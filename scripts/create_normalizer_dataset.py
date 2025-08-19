#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Create Normalizer dataset from reasoner + verifier outputs.

Input (per dataset/split):
  - reasoner: data_intermediate/reasoner_data/<ds>/<ds>_<split>_reasoner.jsonl
  - verifier: data_intermediate/verifier_data/<ds>/<ds>_<split>_verifier.jsonl (optional but recommended)

Output:
  - normalizer: data_intermediate/normalizer_data/<ds>/<ds>_<split>_normalizer.jsonl

Each output row:
{
  "id": ...,
  "dataset": ...,
  "split": ...,
  "input_question": ...,
  "plan": ...,
  "reasoner_output": ...,
  "verifier_output": ...,
  "normalized_rationale": "1. ...\n2. ...\n...\n#### <final_answer>",
  "final_answer_gold": ...,
  "final_answer_norm": ...,
  "subject": ...,
  "level": ...,
  "qc": {...},
  "source_reasoner": {...},   # source rows (optional/minimal depending on --source_mode)
  "source_verifier": {...}
}

Policy:
- If VERDICT=CORRECT → keep final_answer (prefer gold if present), recompute calc arrows,
  remove meaningless calc blocks ([[calc: symbolic]]), renumber steps, ensure one #### line.
- If VERDICT=INCORRECT and FIXED_ANSWER present → use fixed answer; otherwise use gold if available;
  normalize same as above. (No LLM rewrite by default.)
"""

import os
import re
import sys
import json
import math
import ast
import argparse
from typing import Dict, Any, List, Tuple, Optional
from collections import defaultdict
from tqdm import tqdm

# ----------------------------
# IO helpers
# ----------------------------

def load_jsonl(path: str) -> List[Dict[str, Any]]:
    out = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                out.append(json.loads(line))
            except Exception as e:
                print(f"[WARN] Bad JSON line in {path}: {e}", file=sys.stderr)
    return out

def append_jsonl(path: str, row: Dict[str, Any]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")

# ----------------------------
# Parsing & utilities
# ----------------------------

VERDICT_RE = re.compile(r"<<\s*VERDICT:\s*(CORRECT|INCORRECT)\s*>>", flags=re.I)
FIXED_ANS_RE = re.compile(r"<<\s*FIXED_ANSWER:\s*(.*?)\s*>>", flags=re.I | re.S)

# Calc markers: [[calc: expr]] -> value
CALC_LINE_RE = re.compile(r"\[\[\s*calc\s*:\s*(.*?)\s*\]\]\s*->\s*([^\s]+)", flags=re.I)

# Final answer line
FINAL_RE = re.compile(r"^\s*####\s*(.+?)\s*$")

# Let symbolic allowance: numbers, + - * / ( ) . , spaces
# If anything outside allowed & known names (pi,e) appears, treat as symbolic
ALLOWED_CHARS_RE = re.compile(r"^[0-9\.\+\-\*\/\(\)\s,]*$")

# ----------------------------
# Safe calculator (extended)
# ----------------------------

ALLOWED_FUNCS = {
    "sqrt": math.sqrt,
    "abs": abs,
    "round": round,
    "sin": math.sin,
    "cos": math.cos,
    "tan": math.tan,
    "log": math.log,   # natural log
    "ln": math.log,
    "log10": math.log10,
    "exp": math.exp,
}
ALLOWED_NAMES = {"pi": math.pi, "e": math.e}

class SafeEval(ast.NodeVisitor):
    def visit(self, node):
        if isinstance(node, ast.Expression):
            return self.visit(node.body)
        # Py <3.8
        elif hasattr(ast, "Num") and isinstance(node, ast.Num):  # pragma: no cover
            return node.n
        elif isinstance(node, ast.Constant):
            if isinstance(node.value, (int, float)): return node.value
            raise ValueError("bad const")
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
            if not isinstance(node.func, ast.Name):
                raise ValueError("bad call")
            fn = node.func.id
            if fn not in ALLOWED_FUNCS:
                raise ValueError(f"func not allowed: {fn}")
            args = [self.visit(a) for a in node.args]
            return ALLOWED_FUNCS[fn](*args)
        elif isinstance(node, ast.Name):
            if node.id in ALLOWED_NAMES:
                return ALLOWED_NAMES[node.id]
            raise ValueError(f"name not allowed: {node.id}")
        elif isinstance(node, ast.Tuple):
            # Disallow tuples
            raise ValueError("tuple not allowed")
        else:
            raise ValueError("bad expr")

def safe_calc(expr: str) -> Optional[float]:
    expr = expr.strip()
    if not expr:
        return None
    try:
        tree = ast.parse(expr, mode="eval")
        return float(SafeEval().visit(tree))
    except Exception:
        return None

def expr_has_symbol(expr: str) -> bool:
    # Allow pi/e tokens; otherwise, if letters present → symbolic.
    tmp = expr.strip()
    # Quick bail-out: only allowed chars?
    if ALLOWED_CHARS_RE.match(tmp):
        return False
    # Contains letters ⇒ allow 'pi' 'e' only
    # Replace allowed names and re-check
    tmp2 = re.sub(r"\bpi\b", "", tmp)
    tmp2 = re.sub(r"\be\b", "", tmp2)
    return bool(re.search(r"[A-Za-z]", tmp2))

def recompute_calc_line(line: str) -> Tuple[str, List[str]]:
    """
    Fix or remove calc blocks on a single line.
    - If expr has symbols → remove the entire ' [[calc:...]] -> ...' segment
    - Else recompute -> value with safe_calc
    Returns (patched_line, notes)
    """
    notes = []
    def _fix(m: re.Match) -> str:
        expr = m.group(1).strip()
        arrow_val = m.group(2).strip()
        if expr_has_symbol(expr):
            notes.append(f"removed symbolic calc '{expr}'")
            return ""  # remove marker entirely
        val = safe_calc(expr)
        if val is None:
            notes.append(f"calc parse failed '{expr}'; removed")
            return ""
        # prefer minimal float formatting
        if abs(val - round(val)) < 1e-9:
            val_str = str(int(round(val)))
        else:
            val_str = f"{val:.10g}"
        if arrow_val != val_str:
            notes.append(f"fixed '{expr}' from {arrow_val} to {val_str}")
        return f"[[calc: {expr}]] -> {val_str}"
    # Replace each marker
    patched = CALC_LINE_RE.sub(_fix, line)
    # Clean double spaces from removed segments
    patched = re.sub(r"\s{2,}", " ", patched).rstrip()
    return patched, notes

def normalize_steps(text: str) -> Tuple[str, List[str]]:
    """
    - Keep only content before the first '####'
    - Split into lines, drop empty lines
    - Renumber as '1. ...', '2. ...', ...
    - Recompute/clean calc markers per line
    """
    notes = []
    # Keep body before ####
    body = text.split("\n####", 1)[0]
    raw_lines = [ln.rstrip() for ln in body.splitlines()]
    # Keep non-empty lines
    raw_lines = [ln for ln in raw_lines if ln.strip()]

    # If no explicit numbering, treat each as a step
    steps = []
    for ln in raw_lines:
        # Strip leading bullet-like tokens
        ln = re.sub(r"^\s*(\d+\.|\-\s+|\*\s+|[>\u2022]\s+)", "", ln)
        # Fix/clean calc markers
        fixed, notes_line = recompute_calc_line(ln)
        notes.extend(notes_line)
        steps.append(fixed.strip())

    # Remove any now-empty steps (e.g., line was only a removed calc)
    steps = [s for s in steps if s]

    # Renumber
    renum = []
    for i, s in enumerate(steps, 1):
        renum.append(f"{i}. {s}")

    return "\n".join(renum), notes

def extract_final_answer(reasoner_output: str, verifier_output: Optional[str], gold: Optional[str]) -> Tuple[str, List[str]]:
    """
    Decide final answer to use.
    Priority:
      1) If verifier has FIXED_ANSWER and VERDICT=INCORRECT → use FIXED_ANSWER
      2) Else if gold present → use gold
      3) Else parse from reasoner_output '#### ...'
    """
    notes = []
    if verifier_output:
        m_verdict = VERDICT_RE.search(verifier_output)
        m_fixed = FIXED_ANS_RE.search(verifier_output)
        verdict = (m_verdict.group(1).upper() if m_verdict else "")
        if verdict == "INCORRECT" and m_fixed:
            ans = m_fixed.group(1).strip()
            notes.append("used FIXED_ANSWER from verifier")
            return ans, notes

    if gold and str(gold).strip():
        notes.append("used gold final answer")
        return str(gold).strip(), notes

    # Fallback: scrape from reasoner_output
    for ln in reasoner_output.splitlines()[::-1]:
        m = FINAL_RE.match(ln)
        if m:
            ans = m.group(1).strip()
            notes.append("used final from reasoner_output")
            return ans, notes

    notes.append("no final answer found; set to ''")
    return "", notes

def minimal_source(row: Dict[str, Any]) -> Dict[str, Any]:
    # avoid deeply nesting meta to keep files small
    keys = ["id", "dataset", "split", "input_question", "plan",
            "reasoner_output", "final_answer", "rationale_gold",
            "subject", "level"]
    out = {}
    for k in keys:
        if k in row:
            out[k] = row[k]
    return out

# ----------------------------
# Core processing
# ----------------------------

def process_split(
    dataset: str,
    split: str,
    reasoner_file: str,
    verifier_file: Optional[str],
    out_dir: str,
    source_mode: str = "minimal",
    limit: Optional[int] = None,
    resume: bool = True,
) -> None:
    print(f"[PROCESS] {dataset}/{split}  reasoner={reasoner_file}  verifier={verifier_file or '(none)'}")
    reasoner_rows = load_jsonl(reasoner_file)
    if limit is not None and limit >= 0:
        reasoner_rows = reasoner_rows[:limit]
        print(f"[INFO] limiting to {len(reasoner_rows)} examples")

    # Build verifier index by id if present
    verifier_index: Dict[str, Dict[str, Any]] = {}
    if verifier_file and os.path.exists(verifier_file):
        for vrow in load_jsonl(verifier_file):
            vid = str(vrow.get("id", ""))
            if vid:
                verifier_index[vid] = vrow

    out_path = os.path.join(out_dir, dataset)
    os.makedirs(out_path, exist_ok=True)
    out_file = os.path.join(out_path, f"{dataset}_{split}_normalizer.jsonl")

    seen_ids = set()
    if resume and os.path.exists(out_file):
        for ex in load_jsonl(out_file):
            sid = str(ex.get("id", ""))
            if sid:
                seen_ids.add(sid)
        print(f"[RESUME] {len(seen_ids)} rows already in {out_file}; will append new ones.")

    wrote = 0
    for row in tqdm(reasoner_rows, desc=f"{dataset}/{split}"):
        rid = str(row.get("id", ""))
        if not rid:
            continue
        if rid in seen_ids:
            continue

        question = row.get("input_question", "")
        plan = row.get("plan", "")
        reasoner_output = row.get("reasoner_output", "") or row.get("reasoner", "")
        gold = row.get("final_answer_gold", row.get("final_answer", ""))
        subject = row.get("subject", "")
        level = row.get("level", "")

        vrow = verifier_index.get(rid)
        verifier_output = vrow.get("verifier_output", "") if vrow else ""

        # Normalize steps
        norm_steps, notes1 = normalize_steps(reasoner_output or "")
        # Decide final answer
        final_ans, notes2 = extract_final_answer(reasoner_output or "", verifier_output, gold)

        # Ensure single final line appended
        normalized_rationale = norm_steps
        if final_ans:
            normalized_rationale = f"{normalized_rationale}\n#### {final_ans}"
        else:
            # keep empty but mark QC later
            pass

        # QC
        qc = {"passed": True, "reasons": []}
        if not norm_steps.strip():
            qc["passed"] = False
            qc["reasons"].append("no steps")
        # Ensure numbering starts at 1
        if not re.match(r"^1\.\s", normalized_rationale):
            qc["passed"] = False
            qc["reasons"].append("steps not numbered")
        # Check calc markers are numeric only
        for m in CALC_LINE_RE.finditer(normalized_rationale):
            expr = m.group(1)
            if expr_has_symbol(expr):
                qc["passed"] = False
                qc["reasons"].append(f"symbolic calc remained: {expr}")
        # Final answer presence
        if not re.search(r"^\s*####\s+.+", normalized_rationale, flags=re.M):
            qc["passed"] = False
            qc["reasons"].append("missing final ####")

        # Build output row
        out_row = {
            "id": rid,
            "dataset": dataset,
            "split": split,
            "input_question": question,
            "plan": plan,
            "reasoner_output": reasoner_output,
            "verifier_output": verifier_output,
            "normalized_rationale": normalized_rationale,
            "final_answer_gold": gold,
            "final_answer_norm": final_ans,
            "subject": subject,
            "level": level,
            "qc": qc,
        }

        # Source inclusion policy
        if source_mode == "minimal":
            out_row["source_reasoner"] = minimal_source(row)
            if vrow:
                out_row["source_verifier"] = minimal_source(vrow)
        elif source_mode == "raw":
            out_row["source_reasoner"] = row
            if vrow:
                out_row["source_verifier"] = vrow
        # source_mode == "none": omit

        append_jsonl(out_file, out_row)
        wrote += 1

    print(f"[DONE] Wrote {wrote} rows to {out_file}")

# ----------------------------
# Orchestration
# ----------------------------

def find_split_file(root: str, ds: str, split: str, suffix: str) -> Optional[str]:
    """
    Search common patterns:
      <root>/<ds>/<ds>_<split>_<suffix>.jsonl
      <root>/<ds>/<split>.jsonl
    """
    cands = [
        os.path.join(root, ds, f"{ds}_{split}_{suffix}.jsonl"),
        os.path.join(root, ds, f"{split}.jsonl"),
        os.path.join(root, ds, f"{ds}_{split}.jsonl"),
    ]
    for p in cands:
        if os.path.isfile(p):
            return p
    return None

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reasoner_root", type=str, default="data_intermediate/reasoner_data")
    ap.add_argument("--verifier_root", type=str, default="data_intermediate/verifier_data")
    ap.add_argument("--out_dir", type=str, default="data_intermediate/normalizer_data")
    ap.add_argument("--datasets", type=str, default="gsm8k,math,multiarith")
    ap.add_argument("--splits", type=str, default="train,val,test")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--only_dataset", type=str, default=None, choices=["gsm8k","math","multiarith"])
    ap.add_argument("--only_split", type=str, default=None, choices=["train","val","test"])
    ap.add_argument("--source_mode", type=str, default="minimal", choices=["minimal","raw","none"])
    ap.add_argument("--no_resume", action="store_true", help="ignore existing out_file; do not attempt resume")
    args = ap.parse_args()

    ds_list = (args.only_dataset.split(",") if args.only_dataset else args.datasets.split(","))
    sp_list = (args.only_split.split(",") if args.only_split else args.splits.split(","))

    for ds in ds_list:
        ds = ds.strip()
        for sp in sp_list:
            sp = sp.strip()
            reasoner_file = find_split_file(args.reasoner_root, ds, sp, "reasoner")
            if not reasoner_file:
                print(f"[WARN] Missing reasoner for {ds}/{sp} in {args.reasoner_root}", file=sys.stderr)
                continue
            verifier_file = find_split_file(args.verifier_root, ds, sp, "verifier")
            if not verifier_file:
                print(f"[WARN] Missing verifier for {ds}/{sp} in {args.verifier_root} (will proceed).", file=sys.stderr)

            process_split(
                dataset=ds,
                split=sp,
                reasoner_file=reasoner_file,
                verifier_file=verifier_file,
                out_dir=args.out_dir,
                source_mode=args.source_mode,
                limit=args.limit,
                resume=(not args.no_resume),
            )

if __name__ == "__main__":
    main()
