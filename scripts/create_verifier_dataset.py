#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
create_verifier_dataset.py

Builds Verifier training data from prior reasoner outputs.

Input (per dataset/split):
  data_intermediate/reasoner_data/<ds>/<ds>_<split>_reasoner.jsonl

Output:
  data_intermediate/verifier_data/<ds>/<ds>_<split>_verifier.jsonl

Each output example contains:
  {
    "id": ...,
    "dataset": ...,
    "split": ...,
    "input_question": ...,
    "plan": ...,
    "reasoner_input": "...",     # possibly perturbed (for hard negatives)
    "verifier_output": "<<VERDICT: ...>>\n<<EVIDENCE: ...>>\n<<FIXED_ANSWER: ...>>",
    "verdict_label": "CORRECT" | "INCORRECT",
    "final_answer_gold": "...",
    "subject": "...",
    "level": "...",
    "qc": {"passed": true/false, "reasons": [...]},
    "source": {...},             # original reasoner record (minimal)
    "meta": {"perturbed": bool, "perturb_kind": "...", "calc_check": {...}}
  }

Run:
  python3 scripts/create_verifier_dataset.py \
    --reasoner_root data_intermediate/reasoner_data \
    --out_dir      data_intermediate/verifier_data \
    --datasets     gsm8k,math,multiarith \
    --splits       train,val,test \
    --perturb_ratio 0.25 \
    --limit 1000
"""

import os
import re
import json
import math
import random
import argparse
from typing import List, Dict, Tuple, Optional
from collections import defaultdict
from tqdm import tqdm
import ast

# ----------------------------
# I/O utils
# ----------------------------
def load_jsonl(path: str) -> List[dict]:
    data = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line: continue
            try:
                data.append(json.loads(line))
            except Exception:
                continue
    return data

def write_jsonl(path: str, rows: List[dict]):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

# ----------------------------
# Safe calculator
# ----------------------------
ALLOWED_FUNCS = {
    "sqrt": math.sqrt,
    "abs": abs,
    "round": round,
}
ALLOWED_NAMES = {"pi": math.pi, "e": math.e}

class SafeEval(ast.NodeVisitor):
    def visit(self, node):
        if isinstance(node, ast.Expression):
            return self.visit(node.body)
        elif isinstance(node, ast.Constant):
            if isinstance(node.value, (int, float)): return node.value
            raise ValueError("bad const")
        elif hasattr(ast, "Num") and isinstance(node, ast.Num):  # Py<3.8 compatibility
            return node.n
        elif isinstance(node, ast.BinOp):
            l = self.visit(node.left); r = self.visit(node.right)
            if isinstance(node.op, ast.Add): return l + r
            if isinstance(node.op, ast.Sub): return l - r
            if isinstance(node.op, ast.Mult): return l * r
            if isinstance(node.op, ast.Div): return l / r
            if isinstance(node.op, ast.FloorDiv): return l // r
            if isinstance(node.op, ast.Mod): return l % r
            if isinstance(node.op, ast.Pow): return l ** r
            raise ValueError("bad op")
        elif isinstance(node, ast.UnaryOp):
            v = self.visit(node.operand)
            if isinstance(node.op, ast.UAdd): return +v
            if isinstance(node.op, ast.USub): return -v
            raise ValueError("bad uop")
        elif isinstance(node, ast.Call):
            if not isinstance(node.func, ast.Name): raise ValueError("bad call")
            fn = node.func.id
            if fn not in ALLOWED_FUNCS: raise ValueError("func not allowed")
            args = [self.visit(a) for a in node.args]
            return ALLOWED_FUNCS[fn](*args)
        elif isinstance(node, ast.Name):
            if node.id in ALLOWED_NAMES: return ALLOWED_NAMES[node.id]
            raise ValueError("name not allowed")
        else:
            raise ValueError("bad expr")

def safe_calc(expr: str) -> Optional[float]:
    expr = expr.strip()
    try:
        tree = ast.parse(expr, mode="eval")
        return float(SafeEval().visit(tree))
    except Exception:
        return None

# ----------------------------
# Parsing helpers
# ----------------------------
CALC_LINE_RE = re.compile(r"\[\[calc:\s*(.*?)\s*\]\]\s*->\s*([+-]?\d+(?:\.\d+)?)")
HASH_ANS_RE = re.compile(r"####\s*(.+)\s*$")

LATEX_CMD = re.compile(r"\\[a-zA-Z]+")
DEG_RE = re.compile(r"(?:\\degree|°)")
FRAC_RE = re.compile(r"\\frac\{([^}]+)\}\{([^}]+)\}")
SQRT_RE = re.compile(r"\\sqrt\{([^}]+)\}")
SET_SPLIT_RE = re.compile(r"\s*,\s*")

def strip_math_wrapping(s: str) -> str:
    s = s.strip()
    if s.startswith("$") and s.endswith("$"):
        s = s[1:-1]
    return s.strip()

def latex_to_numeric_string(s: str) -> str:
    """
    Convert simple LaTeX (frac, sqrt) to a numeric expression string.
    """
    s = strip_math_wrapping(s)
    s = DEG_RE.sub("", s)

    # Replace \frac{a}{b} -> (a)/(b)
    def _frac(m):
        return f"({m.group(1)})/({m.group(2)})"
    s = FRAC_RE.sub(_frac, s)

    # Replace \sqrt{a} -> sqrt(a)
    def _sqrt(m):
        return f"sqrt({m.group(1)})"
    s = SQRT_RE.sub(_sqrt, s)

    # Remove remaining latex commands \something
    s = LATEX_CMD.sub("", s)

    # Remove braces while keeping content
    s = s.replace("{", "(").replace("}", ")")
    # Clean spaces
    s = re.sub(r"\s+", " ", s).strip()
    return s

def parse_final_from_reasoner(text: str) -> Optional[str]:
    if not text: return None
    m = HASH_ANS_RE.search(text)
    if m:
        return m.group(1).strip()
    return None

def numeric_equal(a: float, b: float, eps: float = 1e-6) -> bool:
    return abs(a - b) <= eps

def try_eval_numeric(s: str) -> Optional[float]:
    s = latex_to_numeric_string(s)
    # Quick guard: reject if letters remain (variables)
    if re.search(r"[A-Za-z]", s):
        return None
    return safe_calc(s)

def normalize_answer(ans: str) -> Dict:
    """
    Return dict with:
      {"text": original, "norm": string_norm, "values": List[float] or None}
    """
    if ans is None:
        return {"text": None, "norm": None, "values": None}

    text = ans.strip()
    text = text.strip(".")
    # Remove surrounding $...$
    core = strip_math_wrapping(text)
    # Remove trailing period/units like "kg" only if obviously appended to number?
    core = core.strip()

    # Comma-separated set/list of answers
    parts = [p.strip() for p in SET_SPLIT_RE.split(core)] if "," in core else [core]

    values = []
    all_numeric = True
    for p in parts:
        v = try_eval_numeric(p)
        if v is None:
            all_numeric = False
            break
        values.append(v)

    if all_numeric and values:
        return {"text": text, "norm": "NUMERIC_SET", "values": values}

    # Attempt simple radical like "10\sqrt{3}" -> evaluate
    p_eval = try_eval_numeric(core)
    if p_eval is not None:
        return {"text": text, "norm": "NUMERIC", "values": [p_eval]}

    # Degree handling: strip ° and try again
    core2 = DEG_RE.sub("", core)
    if core2 != core:
        v2 = try_eval_numeric(core2)
        if v2 is not None:
            return {"text": text, "norm": "NUMERIC_DEG", "values": [v2]}

    # Otherwise keep literal
    return {"text": text, "norm": "TEXT", "values": None}

def compare_answers(pred: str, gold: str) -> Tuple[bool, str]:
    """
    Returns (is_equal, reason).
    Numeric sets are compared as unordered (with tolerance).
    """
    if not gold:
        return (False, "missing gold")
    if not pred:
        return (False, "missing pred")

    npred = normalize_answer(pred)
    ngold = normalize_answer(gold)

    # If both numeric sets
    if npred["values"] is not None and ngold["values"] is not None:
        pv = sorted(npred["values"])
        gv = sorted(ngold["values"])
        if len(pv) != len(gv):
            return (False, f"set size mismatch {len(pv)} vs {len(gv)}")
        for a, b in zip(pv, gv):
            if not numeric_equal(a, b):
                return (False, f"value mismatch {a} vs {b}")
        return (True, "numeric match")

    # Fallback: stripped literal compare
    if npred["text"] and ngold["text"] and (npred["text"].strip() == ngold["text"].strip()):
        return (True, "literal match")

    return (False, f"literal mismatch '{npred['text']}' vs '{ngold['text']}'")

# ----------------------------
# Calc verification & evidence
# ----------------------------
def verify_calc_lines(text: str) -> Dict:
    """
    Returns dict:
      {
        "ok": bool,
        "checked": int,
        "mismatches": [ {"expr":..., "claimed":float, "got":float, "line":...}, ... ],
        "parse_fail": [ {"expr":..., "line":...}, ... ]
      }
    """
    out = {"ok": True, "checked": 0, "mismatches": [], "parse_fail": []}
    if not text: return out
    for line in text.splitlines():
        m = CALC_LINE_RE.search(line)
        if not m:
            continue
        expr = m.group(1).strip()
        try:
            claimed = float(m.group(2))
        except Exception:
            out["ok"] = False
            out["parse_fail"].append({"expr": expr, "line": line})
            continue
        got = safe_calc(expr)
        out["checked"] += 1
        if got is None:
            out["ok"] = False
            out["parse_fail"].append({"expr": expr, "line": line})
        else:
            if not numeric_equal(got, claimed):
                out["ok"] = False
                out["mismatches"].append({"expr": expr, "claimed": claimed, "got": got, "line": line})
    return out

def make_auto_evidence(problem: str,
                       plan: str,
                       reasoner: str,
                       final_pred: Optional[str],
                       final_gold: str,
                       calc_report: Dict) -> str:
    items = []
    if final_pred is None:
        items.append("Missing final answer marker '####'.")
    else:
        ok, why = compare_answers(final_pred, final_gold)
        if ok:
            items.append(f"Final answer matches gold ({final_gold}).")
        else:
            items.append(f"Final answer mismatch: predicted '{final_pred}' vs gold '{final_gold}' ({why}).")
    if calc_report["checked"] > 0:
        if calc_report["mismatches"]:
            for mm in calc_report["mismatches"][:3]:
                items.append(f"Calc mismatch: '{mm['expr']}' -> {mm['claimed']} (recalc {mm['got']}).")
        if calc_report["parse_fail"]:
            items.append(f"{len(calc_report['parse_fail'])} calc line(s) not evaluable.")
    return " ".join(items) if items else "No issues found."

# ----------------------------
# Hard negatives (simple, robust)
# ----------------------------
NUM_TOKEN_RE = re.compile(r"(?<![\w.])([+-]?\d+(?:\.\d+)?)(?![\w.])")

def perturb_reasoner_text(text: str) -> Tuple[str, str, bool]:
    """
    Make a subtle but label-changing corruption.
    Strategies (first applicable):
      - If a [[calc: ...]] line exists, alter the arrow value by ±1 (or ±10% for big numbers)
      - Else, flip a '+' to '-' or vice versa in a numeric expression (leaves text plausible)
      - Else, append a wrong '####' answer
    Returns (new_text, kind, changed)
    """
    if not text:
        return text, "none", False

    lines = text.splitlines()
    # 1) Tweak arrow values on calc lines
    changed = False
    for i, line in enumerate(lines):
        m = CALC_LINE_RE.search(line)
        if not m:
            continue
        expr, val = m.group(1), m.group(2)
        try:
            v = float(val)
        except Exception:
            continue
        delta = 1.0 if abs(v) < 50 else max(1.0, abs(v) * 0.1)
        new_v = v + (delta if random.random() < 0.5 else -delta)
        new_line = CALC_LINE_RE.sub(f"[[calc: {expr}]] -> {new_v}", line, count=1)
        lines[i] = new_line
        changed = True
        return "\n".join(lines), "calc_arrow_tweak", True

    # 2) Flip a + / - in a numeric-only expression line
    for i, line in enumerate(lines):
        if "[[calc:" in line:
            continue
        if NUM_TOKEN_RE.search(line) and ("+" in line or "-" in line):
            flipped = line.replace("+", "§PLUS§").replace("-", "+").replace("§PLUS§", "-")
            lines[i] = flipped
            changed = True
            return "\n".join(lines), "op_flip", True

    # 3) Append wrong final answer if hash missing
    has_hash = any(HASH_ANS_RE.search(l) for l in lines)
    if not has_hash:
        lines.append("#### 0")
        return "\n".join(lines), "append_wrong_final", True

    # 4) Replace existing final with wrong one
    for i, line in enumerate(lines):
        m = HASH_ANS_RE.search(line)
        if not m:
            continue
        lines[i] = "#### 0"
        return "\n".join(lines), "replace_final", True

    return text, "none", False

# ----------------------------
# QC
# ----------------------------
def qc_verifier_block(block: str, label: str, gold_fixed: str) -> Tuple[bool, List[str]]:
    errs = []
    if "<<VERDICT:" not in block or ">>" not in block:
        errs.append("missing VERDICT tag")
    if "<<EVIDENCE:" not in block or ">>" not in block:
        errs.append("missing EVIDENCE tag")
    if "<<FIXED_ANSWER:" not in block or ">>" not in block:
        errs.append("missing FIXED_ANSWER tag")
    # Basic consistency
    m = re.search(r"<<VERDICT:\s*(CORRECT|INCORRECT)\s*>>", block)
    if not m:
        errs.append("VERDICT value invalid")
    else:
        if m.group(1) != label:
            errs.append(f"VERDICT label mismatch {m.group(1)} vs {label}")

    m2 = re.search(r"<<FIXED_ANSWER:\s*(.*?)\s*>>", block, flags=re.S)
    if m2:
        # Require fixed equals canonical gold if incorrect
        if label == "INCORRECT":
            ok, _ = compare_answers(m2.group(1).strip(), gold_fixed)
            if not ok:
                errs.append("FIXED_ANSWER does not equal gold")
    return (len(errs) == 0, errs)

# ----------------------------
# Main builder
# ----------------------------
def find_reasoner_file(root: str, ds: str, sp: str) -> Optional[str]:
    cands = [
        os.path.join(root, ds, f"{ds}_{sp}_reasoner.jsonl"),
        os.path.join(root, ds, f"{sp}.jsonl"),
        os.path.join(root, ds, f"{ds}_{sp}.jsonl"),
    ]
    for p in cands:
        if os.path.isfile(p):
            return p
    return None

def process_split(ds: str,
                  sp: str,
                  reasoner_root: str,
                  out_root: str,
                  limit: Optional[int],
                  strict_symbolic: bool,
                  perturb_ratio: float,
                  resume: bool):
    in_file = find_reasoner_file(reasoner_root, ds, sp)
    if not in_file:
        print(f"[WARN] No input for {ds}/{sp} under {reasoner_root}")
        return

    out_dir = os.path.join(out_root, ds)
    out_file = os.path.join(out_dir, f"{ds}_{sp}_verifier.jsonl")
    os.makedirs(out_dir, exist_ok=True)

    seen_ids = set()
    if resume and os.path.exists(out_file):
        for ex in load_jsonl(out_file):
            seen_ids.add(ex.get("id"))

    src = load_jsonl(in_file)
    if limit is not None and limit >= 0:
        src = src[:limit]
        print(f"[INFO] limiting {ds}/{sp} to {len(src)} examples")

    rows = []
    for ex in tqdm(src, desc=f"{ds}/{sp}"):
        ex_id = ex.get("id")
        if resume and ex_id in seen_ids:
            continue

        question = ex.get("input_question", "")
        plan = ex.get("plan") or ex.get("planner_output") or ""
        reasoner = ex.get("reasoner_output", "")
        gold = ex.get("final_answer") or ex.get("final_answer_gold") or ""
        subject = ex.get("subject", "")
        level = ex.get("level", "")

        # Build calc report
        calc_report = verify_calc_lines(reasoner)

        # Extract predicted final
        pred = parse_final_from_reasoner(reasoner)

        # Verdict
        equal, why = compare_answers(pred or "", gold)
        label = "CORRECT" if equal else "INCORRECT"

        # Hard negative?
        perturbed = False
        perturb_kind = ""
        reasoner_in = reasoner
        if random.random() < perturb_ratio:
            r2, kind, changed = perturb_reasoner_text(reasoner)
            if changed:
                reasoner_in = r2
                perturbed = True
                perturb_kind = kind
                # Recompute calc + final on perturbed
                calc_report = verify_calc_lines(reasoner_in)
                pred = parse_final_from_reasoner(reasoner_in)
                equal, why = compare_answers(pred or "", gold)
                label = "CORRECT" if equal else "INCORRECT"

        # Evidence (auto)
        evidence = make_auto_evidence(question, plan, reasoner_in, pred, gold, calc_report)
        fixed_answer = gold if gold is not None else ""

        # For MATH symbolic proofs, optionally relax calc expectations
        if strict_symbolic is False and ds == "math":
            # no action needed here since we don't penalize lack of calc
            pass

        verifier_block = (
            f"<<VERDICT: {label}>>\n"
            f"<<EVIDENCE: {evidence}>>\n"
            f"<<FIXED_ANSWER: {fixed_answer}>>"
        )

        # QC
        qc_pass, qc_reasons = qc_verifier_block(verifier_block, label, fixed_answer)

        rows.append({
            "id": ex_id,
            "dataset": ds,
            "split": sp,
            "input_question": question,
            "plan": plan,
            "reasoner_input": reasoner_in,
            "verifier_output": verifier_block,
            "verdict_label": label,
            "final_answer_gold": gold,
            "subject": subject,
            "level": level,
            "qc": {"passed": qc_pass, "reasons": qc_reasons},
            "source": {
                "id": ex.get("id"),
                "hf_name": ex.get("source", {}).get("hf_name", ""),
                "hf_config": ex.get("source", {}).get("hf_config", ""),
                "original_index": ex.get("source", {}).get("original_index", ""),
                "unique_id": ex.get("source", {}).get("unique_id", "")
            },
            "meta": {
                "perturbed": perturbed,
                "perturb_kind": perturb_kind,
                "calc_check": calc_report
            }
        })

    write_jsonl(out_file, rows)
    print(f"[DONE] Wrote {out_file}  ({len(rows)} examples)")

# ----------------------------
# CLI
# ----------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reasoner_root", type=str, default="data_intermediate/reasoner_data",
                    help="Folder containing per-dataset reasoner jsonls")
    ap.add_argument("--out_dir", type=str, default="data_intermediate/verifier_data")
    ap.add_argument("--datasets", type=str, default="gsm8k,math,multiarith")
    ap.add_argument("--splits", type=str, default="train,val,test")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--perturb_ratio", type=float, default=0.25, help="fraction of samples to create hard negatives")
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--strict_symbolic", action="store_true",
                    help="If set, still expects numeric calc checks; otherwise symbolic (MATH) is calc-optional")
    args = ap.parse_args()

    random.seed(40)

    ds_list = [d.strip() for d in args.datasets.split(",") if d.strip()]
    sp_list = [s.strip() for s in args.splits.split(",") if s.strip()]

    for ds in ds_list:
        for sp in sp_list:
            process_split(
                ds=ds,
                sp=sp,
                reasoner_root=args.reasoner_root,
                out_root=args.out_dir,
                limit=args.limit,
                strict_symbolic=args.strict_symbolic,
                perturb_ratio=args.perturb_ratio,
                resume=True if args.resume else False,
            )

if __name__ == "__main__":
    main()
