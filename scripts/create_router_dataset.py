#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Create a Router / Role-Selector dataset from existing Planner/Reasoner/Verifier/Normalizer JSONLs.

For each problem, we emit one supervision example that teaches the selector to choose:
- CALLS: which roles to invoke (subset of {planner, reasoner, calculator, verifier, normalizer})
- HALT_AFTER: the last role to run before returning the final answer

Heuristics (tunable via CLI flags):
- planner: if a plan exists with >=2 steps OR question looks multi-step.
- reasoner: always.
- calculator: only if there are valid numeric [[calc: ...]] expressions (ignores blank/symbolic).
- verifier: if miscalc was detected, or verdict == INCORRECT, or (math & level>=4) if --verifier_on_hard_math.
- normalizer: if normalized rationale exists and differs; or --always_normalize.

Router input can optionally include the plan (short) to help selection.

Outputs JSONL rows like:
{
  "id": "...",
  "dataset": "gsm8k",
  "split": "train",
  "input_question": "...",
  "selector_input": "<role:selector>\n{problem}\nPLAN:\n<<...>>",
  "selector_output": "CALLS: planner,reasoner,calculator,verifier; HALT_AFTER: normalizer",
  "calls": ["planner","reasoner","calculator","verifier","normalizer"],
  "halt_after": "normalizer"
  ... (minimal provenance depending on --source_mode)
}
"""

import argparse
import json
import os
import re
import sys
from typing import Any, Dict, Iterable, List, Optional, Tuple
from collections import defaultdict
from tqdm import tqdm
import ast
import math

# ----------------------------
# IO
# ----------------------------

def load_jsonl(path: str) -> List[Dict[str, Any]]:
    out = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line: continue
            try:
                out.append(json.loads(line))
            except Exception as e:
                print(f"[WARN] bad json in {path}: {e}", file=sys.stderr)
    return out

def write_jsonl(path: str, rows: Iterable[Dict[str, Any]], append: bool = True) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    mode = "a" if append else "w"
    with open(path, mode, encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

# ----------------------------
# Safe calc (to decide if calculator is needed)
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
        # Py <3.8
        elif isinstance(node, ast.Num):  # type: ignore[attr-defined]
            return node.n  # type: ignore[attr-defined]
        elif isinstance(node, ast.Constant):
            if isinstance(node.value, (int,float)): return node.value
            raise ValueError("bad const")
        elif isinstance(node, ast.BinOp):
            l = self.visit(node.left); r = self.visit(node.right)
            if isinstance(node.op, ast.Add): return l + r
            if isinstance(node.op, ast.Sub): return l - r
            if isinstance(node.op, ast.Mult): return l * r
            if isinstance(node.op, ast.Div): return l / r
            if isinstance(node.op, ast.Pow): return l ** r
            if isinstance(node.op, ast.Mod): return l % r
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
    try:
        tree = ast.parse(expr.strip(), mode="eval")
        return float(SafeEval().visit(tree))
    except Exception:
        return None

CALC_LINE_RE = re.compile(r"\[\[calc:\s*(.*?)\s*\]\](?:\s*->\s*([^\n]+))?", flags=re.S)

def extract_calc_exprs(text: str) -> List[str]:
    return [m.group(1).strip() for m in CALC_LINE_RE.finditer(text or "")]

def has_numeric_calcs(text: str, ignore_trivial: bool = True) -> bool:
    """
    Return True if there exists at least one [[calc: ...]] whose expression evaluates numerically.
    Trivial ones like blank or a single number (no operator) are ignored if ignore_trivial=True.
    """
    any_numeric = False
    for expr in extract_calc_exprs(text or ""):
        stripped = expr.replace(" ", "")
        if not stripped:
            continue
        # Ignore single number (no operator) if requested
        if ignore_trivial and re.fullmatch(r"[+-]?\d+(\.\d+)?", stripped):
            continue
        # A quick operator hint
        has_op = any(op in stripped for op in ["+","-","*","/","^"])
        if not has_op and ignore_trivial:
            # e.g., 'pi' or variable; treat as non-numeric for our router
            continue
        # Try safe eval
        val = safe_calc(expr)
        if val is not None and (math.isfinite(val)):
            any_numeric = True
            break
    return any_numeric

def detect_calc_mismatch(text: str, tol: float = 1e-6) -> bool:
    """
    True if we can evaluate and find any mismatch between [[calc: expr]] -> claimed value
    """
    mismatch = False
    for m in CALC_LINE_RE.finditer(text or ""):
        expr = (m.group(1) or "").strip()
        arrow = (m.group(2) or "").strip()
        # Try to parse claimed numeric
        try:
            claimed = float(re.findall(r"[+-]?\d+(?:\.\d+)?", arrow)[0])
        except Exception:
            continue
        got = safe_calc(expr)
        if got is None: 
            continue
        if abs(got - claimed) > tol:
            mismatch = True
            break
    return mismatch

# ----------------------------
# Planner / steps detection
# ----------------------------

STEP_RE = re.compile(r"(?m)^\s*\d+\.\s+", flags=0)

def count_steps(plan_text: str) -> int:
    if not plan_text:
        return 0
    # Prefer numbered steps; fallback to line count within << >>
    plan = plan_text
    # If wrapped in << >>, strip
    m = re.search(r"<<(.*)>>", plan_text, flags=re.S)
    if m:
        plan = m.group(1)
    n = len(STEP_RE.findall(plan))
    if n == 0:
        # fallback: splitlines ignoring empties
        lines = [ln.strip() for ln in plan.splitlines() if ln.strip()]
        n = len(lines)
    return n

def looks_multistep_question(q: str) -> bool:
    if not q: return False
    ql = q.lower()
    cues = ["then", "after", "each day", "per day", "per week", "first", "second", "next", "total", "sum of", "difference of"]
    if any(c in ql for c in cues):
        return True
    # Length heuristic
    return len(ql.split()) >= 28

# ----------------------------
# Minimal provenance helper
# ----------------------------

def minimal_source(row: Dict[str, Any]) -> Dict[str, Any]:
    out = {"id": row.get("id")}
    carrier = row.get("source") or row.get("meta") or {}
    if isinstance(carrier, dict):
        nested = carrier.get("source") or carrier.get("orig") or carrier
        prov = {}
        for k in ("unique_id", "hf_name", "hf_config", "original_index"):
            v = nested.get(k) if isinstance(nested, dict) else None
            if v is not None:
                prov[k] = v
        if prov: out["prov"] = prov
    return out

# ----------------------------
# Core merge & label logic
# ----------------------------

def find_file(root: str, ds: str, sp: str, suffix: str) -> Optional[str]:
    cands = [
        os.path.join(root, ds, f"{ds}_{sp}_{suffix}.jsonl"),
        os.path.join(root, ds, f"{sp}.jsonl"),
        os.path.join(root, ds, f"{ds}_{sp}.jsonl"),
    ]
    for p in cands:
        if os.path.isfile(p):
            return p
    return None

def build_index(rows: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    return {str(r.get("id")): r for r in rows if r.get("id") is not None}

def selector_io(problem: str, plan: Optional[str], include_plan: bool) -> Tuple[str, str]:
    inp = "<role:selector>\n" + (problem or "").strip()
    if include_plan and plan and count_steps(plan) >= 2:
        inp += "\nPLAN:\n" + plan.strip()
    return inp, ""  # output filled later

def decide_calls(
    problem: str,
    plan: Optional[str],
    reasoner_text: Optional[str],
    verifier_text: Optional[str],
    normalized_text: Optional[str],
    subject: str,
    level: Any,
    opts: argparse.Namespace,
) -> Tuple[List[str], str, Dict[str, Any]]:
    calls: List[str] = []
    notes: Dict[str, Any] = {}

    # planner?
    n_steps = count_steps(plan or "")
    if (plan and n_steps >= opts.min_plan_steps) or (opts.plan_on_multistep_q and looks_multistep_question(problem)):
        calls.append("planner")
        notes["planner_steps"] = n_steps

    # reasoner (always)
    calls.append("reasoner")

    # calculator?
    # prefer normalized text since it's cleaner; else reasoner
    body_for_calc = normalized_text or reasoner_text or ""
    need_calc = has_numeric_calcs(body_for_calc, ignore_trivial=True)
    if need_calc:
        calls.append("calculator")
    notes["has_numeric_calc"] = need_calc

    # verifier?
    verdict_incorrect = False
    if verifier_text:
        m = re.search(r"<<VERDICT:\s*([A-Z]+)\s*>>", verifier_text)
        verdict_incorrect = (m and m.group(1).upper() == "INCORRECT")
    calc_mismatch = detect_calc_mismatch(body_for_calc) if body_for_calc else False

    hard_math = False
    if (subject or "").lower().strip() and isinstance(level, (int, float, str)):
        try:
            lvl = int(str(level))
            hard_math = (lvl >= opts.hard_math_level_threshold)
        except Exception:
            hard_math = False

    need_verifier = verdict_incorrect or calc_mismatch or (opts.verifier_on_hard_math and hard_math)
    if need_verifier:
        calls.append("verifier")
    notes.update({"verdict_incorrect": verdict_incorrect, "calc_mismatch": calc_mismatch, "hard_math": hard_math})

    # normalizer?
    need_normalizer = False
    if normalized_text and (normalized_text.strip() != (reasoner_text or "").strip()):
        need_normalizer = True
    if opts.always_normalize:
        need_normalizer = True
    if need_normalizer:
        calls.append("normalizer")
    notes["needs_normalizer"] = need_normalizer

    # Halt policy
    if "normalizer" in calls:
        halt_after = "normalizer"
    elif "verifier" in calls:
        halt_after = "verifier"
    else:
        halt_after = "reasoner"

    # Optional aggregator role (off by default)
    if opts.include_aggregator and n_steps >= opts.aggregator_min_steps:
        # keep it after reasoner but before verifier/normalizer in CALLS order if present
        # we don't enforce order in 'calls' (just a set-like list) — HALT_AFTER controls stop.
        if "aggregator" not in calls:
            calls.append("aggregator")
        notes["aggregator_reason"] = f"steps>={opts.aggregator_min_steps}"

    return calls, halt_after, notes

# ----------------------------
# Main
# ----------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_root", type=str, default="data_split",
                    help="Root of raw splits (only used for fallback subject/level if needed).")
    ap.add_argument("--planner_root", type=str, default="data_intermediate/planner_data")
    ap.add_argument("--reasoner_root", type=str, default="data_intermediate/reasoner_data")
    ap.add_argument("--verifier_root", type=str, default="data_intermediate/verifier_data")
    ap.add_argument("--normalizer_root", type=str, default="data_intermediate/normalizer_data")
    ap.add_argument("--out_dir", type=str, default="data_intermediate/router_data")

    ap.add_argument("--datasets", type=str, default="gsm8k,math,multiarith")
    ap.add_argument("--splits", type=str, default="train,val,test")
    ap.add_argument("--limit", type=int, default=-1)
    ap.add_argument("--resume", action="store_true", help="Append and skip seen ids.")
    ap.add_argument("--source_mode", choices=["none","minimal","raw"], default="minimal")
    ap.add_argument("--include_plan_in_input", action="store_true", help="Append PLAN to selector_input if available.")
    ap.add_argument("--min_plan_steps", type=int, default=2)
    ap.add_argument("--plan_on_multistep_q", action="store_true", help="Add planner if question looks multi-step.")
    ap.add_argument("--verifier_on_hard_math", action="store_true", help="Auto include verifier on MATH level>=threshold.")
    ap.add_argument("--hard_math_level_threshold", type=int, default=4)
    ap.add_argument("--always_normalize", action="store_true")
    ap.add_argument("--include_aggregator", action="store_true")
    ap.add_argument("--aggregator_min_steps", type=int, default=5)
    args = ap.parse_args()

    ds_list = [s.strip() for s in args.datasets.split(",") if s.strip()]
    sp_list = [s.strip() for s in args.splits.split(",") if s.strip()]

    for ds in ds_list:
        for sp in sp_list:
            # load reasoner (primary spine)
            reasoner_path = find_file(args.reasoner_root, ds, sp, "reasoner")
            if not reasoner_path:
                print(f"[WARN] Missing reasoner file for {ds}/{sp}; skipping", file=sys.stderr)
                continue
            rows = load_jsonl(reasoner_path)
            if args.limit and args.limit > 0:
                rows = rows[:args.limit]
                print(f"[INFO] limiting {ds}/{sp} to {len(rows)} examples")

            # optional companions
            planner_path = find_file(args.planner_root, ds, sp, "planner")
            verifier_path = find_file(args.verifier_root, ds, sp, "verifier")
            normalizer_path = find_file(args.normalizer_root, ds, sp, "normalizer")

            planner_idx = build_index(load_jsonl(planner_path)) if planner_path else {}
            verifier_idx = build_index(load_jsonl(verifier_path)) if verifier_path else {}
            normalizer_idx = build_index(load_jsonl(normalizer_path)) if normalizer_path else {}

            out_path = os.path.join(args.out_dir, ds, f"{ds}_{sp}_router.jsonl")
            os.makedirs(os.path.dirname(out_path), exist_ok=True)

            seen = set()
            if args.resume and os.path.exists(out_path):
                for ex in load_jsonl(out_path):
                    if ex.get("id"): seen.add(str(ex["id"]))

            to_write: List[Dict[str, Any]] = []

            for row in tqdm(rows, desc=f"{ds}/{sp}", leave=False):
                ex_id = str(row.get("id"))
                if args.resume and ex_id in seen:
                    continue

                q = row.get("input_question") or row.get("question") or ""
                subject = row.get("subject", "")
                level = row.get("level", "")

                # fetch plan (from planner set if present; else from reasoner row if it carried it)
                plan = None
                prow = planner_idx.get(ex_id)
                if prow:
                    plan = prow.get("planner_output") or prow.get("plan") or None
                if not plan:
                    plan = row.get("plan") or row.get("planner_output") or None

                # reasoner text
                reasoner_text = row.get("reasoner_output") or ""

                # verifier text & normalized text
                vrow = verifier_idx.get(ex_id)
                verifier_text = vrow.get("verifier_output") if vrow else row.get("verifier_output")
                nrow = normalizer_idx.get(ex_id)
                normalized_text = nrow.get("normalized_rationale") if nrow else None

                # Decide calls
                calls, halt_after, notes = decide_calls(
                    problem=q,
                    plan=plan,
                    reasoner_text=reasoner_text,
                    verifier_text=verifier_text or "",
                    normalized_text=normalized_text or "",
                    subject=subject or "",
                    level=level,
                    opts=args
                )

                # Build selector input/output strings
                sel_in, _ = selector_io(q, plan, args.include_plan_in_input)
                sel_out = f"CALLS: {','.join(calls)}; HALT_AFTER: {halt_after}"

                out_row: Dict[str, Any] = {
                    "id": ex_id,
                    "dataset": ds,
                    "split": sp,
                    "input_question": q,
                    "selector_input": sel_in,
                    "selector_output": sel_out,
                    "calls": calls,
                    "halt_after": halt_after,
                    # light signals helpful for debug/analysis
                    "subject": subject,
                    "level": level,
                    "signals": notes
                }

                # provenance
                if args.source_mode == "raw":
                    out_row["source_reasoner"] = row
                    if prow: out_row["source_planner"] = prow
                    if vrow: out_row["source_verifier"] = vrow
                    if nrow: out_row["source_normalizer"] = nrow
                elif args.source_mode == "minimal":
                    out_row["source_reasoner"] = minimal_source(row)
                    if prow: out_row["source_planner"] = minimal_source(prow)
                    if vrow: out_row["source_verifier"] = minimal_source(vrow)
                    if nrow: out_row["source_normalizer"] = minimal_source(nrow)
                # else: none

                to_write.append(out_row)

                if len(to_write) >= 1000:
                    write_jsonl(out_path, to_write, append=True)
                    to_write = []

            if to_write:
                write_jsonl(out_path, to_write, append=True)

            print(f"[DONE] {ds}/{sp} -> {out_path}")

if __name__ == "__main__":
    main()
