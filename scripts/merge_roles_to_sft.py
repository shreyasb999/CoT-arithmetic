#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Merge role datasets (planner / reasoner / verifier / normalizer / router)
into a single SFT JSONL:
  {
    "id": "...::<role>",
    "role": "planner|reasoner|verifier|normalizer|selector",
    "dataset": "gsm8k|math|multiarith",
    "split": "train|val|test",
    "prompt": "<role:...> ...",
    "response": "...",
    "meta": {... minimal provenance ...}
  }

Defaults:
- Spine = reasoner set. We attach other roles by shared `id`.
- For reasoner examples, we prefer `normalized_rationale` (if exists) else `reasoner_output`.
- We ensure reasoner responses end with "#### <final>" (best of final_answer_norm, final_answer_gold, or final_answer).
- Minimal provenance to keep file size small (toggle via --source_mode).

No LLM calls. Pure file merge.

Example:
  python3 scripts/merge_roles_to_sft.py \
    --planner_root data_intermediate/planner_data \
    --reasoner_root data_intermediate/reasoner_data \
    --verifier_root data_intermediate/verifier_data \
    --normalizer_root data_intermediate/normalizer_data \
    --router_root data_intermediate/router_data \
    --out_file data_intermediate/unified_sft/all_roles.sft.jsonl \
    --datasets "gsm8k,math,multiarith" --splits "train,val,test" \
    --include_roles "planner,reasoner,verifier,normalizer,selector" \
    --include_plan_in_reasoner
"""

import argparse
import json
import os
import re
import sys
from typing import Any, Dict, Iterable, List, Optional, Tuple
from collections import defaultdict
from tqdm import tqdm

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
# Find role files
# ----------------------------

def find_file(root: str, ds: str, sp: str, suffix: str) -> Optional[str]:
    """
    Try common filename patterns:
      {root}/{ds}/{ds}_{sp}_{suffix}.jsonl
      {root}/{ds}/{sp}.jsonl
      {root}/{ds}/{ds}_{sp}.jsonl
    """
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

# ----------------------------
# Minimal provenance
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
# Prompt builders
# ----------------------------

def ensure_final_suffix(text: str, final: str) -> str:
    """
    Ensure the response ends with '#### <final>'.
    """
    txt = (text or "").rstrip()
    # If already has a trailing #### number/string, keep it.
    if re.search(r"(?m)^\s*####\s+.+\s*$", txt):
        return txt
    if final:
        if not txt.endswith("\n"):
            txt += "\n"
        txt += f"#### {final}"
    return txt

def build_planner_prompt(question: str) -> str:
    return (
        "<role:planner>\n"
        "You write a compact step plan (numbered) inside << >>.\n"
        "Only write the plan; no solution.\n\n"
        f"Problem:\n{question.strip()}\n"
    )

def build_reasoner_prompt(question: str, plan: Optional[str], include_plan: bool) -> str:
    p = "<role:reasoner>\nFollow the given plan. Show numbered steps.\n"
    p += "If you compute, use lines like: [[calc: 12*7-5]] -> 79\n"
    p += "Finish with a line: #### <final answer>\n\n"
    p += f"Problem:\n{question.strip()}\n"
    if include_plan and plan:
        p += "\nPlan:\n" + plan.strip() + "\n"
    return p

def build_verifier_prompt(question: str, candidate: str, gold: Optional[str]) -> str:
    body = (
        "<role:verifier>\n"
        "Given the problem and a candidate rationale + answer, decide correctness.\n"
        "Output exactly:\n"
        "<<VERDICT: CORRECT|INCORRECT>>\n"
        "<<EVIDENCE: ...>>\n"
        "<<FIXED_ANSWER: final_value_if_known_or_blank>>\n\n"
        f"Problem:\n{question.strip()}\n\n"
        "<<CANDIDATE_BEGIN>>\n"
        f"{(candidate or '').strip()}\n"
        "<<CANDIDATE_END>>\n"
    )
    if gold:
        body += f"<<GOLD_ANSWER: {gold}>>\n"
    return body

def build_normalizer_prompt(question: str, candidate: str, verdict: Optional[str]) -> str:
    return (
        "<role:normalizer>\n"
        "Normalize the candidate explanation into compact numbered steps.\n"
        "Keep [[calc: ...]] -> value lines where appropriate.\n"
        "End with '#### <final>'.\n\n"
        f"Problem:\n{question.strip()}\n\n"
        "<<CANDIDATE_BEGIN>>\n"
        f"{(candidate or '').strip()}\n"
        "<<CANDIDATE_END>>\n" +
        (f"<<VERIFIER: {verdict.strip()}>>\n" if verdict else "")
    )

def build_selector_prompt(selector_input: str) -> str:
    """
    Already built by router script, but we still wrap.
    """
    return selector_input if selector_input.startswith("<role:selector>") else "<role:selector>\n" + selector_input

# ----------------------------
# Role example builders (per row)
# ----------------------------

def mk_meta(row: Dict[str, Any], mode: str) -> Dict[str, Any]:
    if mode == "none":
        return {}
    if mode == "raw":
        return {"source": row}
    return {"source": minimal_source(row)}

def make_planner_example(ds: str, sp: str, rid: str, question: str,
                         planner_row: Dict[str, Any], source_mode: str) -> Optional[Dict[str, Any]]:
    plan = planner_row.get("planner_output") or planner_row.get("plan")
    if not plan or not plan.strip():
        return None
    prompt = build_planner_prompt(question)
    response = plan.strip()
    return {
        "id": f"{rid}::planner",
        "role": "planner",
        "dataset": ds, "split": sp,
        "prompt": prompt,
        "response": response,
        "meta": mk_meta(planner_row, source_mode),
    }

def make_reasoner_example(ds: str, sp: str, rid: str, question: str,
                          plan: Optional[str],
                          reasoner_row: Dict[str, Any],
                          normalizer_row: Optional[Dict[str, Any]],
                          source_mode: str,
                          include_plan_in_reasoner: bool) -> Optional[Dict[str, Any]]:
    # Prefer normalized rationale if available
    rationale = None
    final = None
    if normalizer_row and (normalizer_row.get("normalized_rationale")):
        rationale = normalizer_row.get("normalized_rationale")
        final = normalizer_row.get("final_answer_norm") or reasoner_row.get("final_answer_gold") or reasoner_row.get("final_answer")
    else:
        rationale = reasoner_row.get("reasoner_output")
        final = reasoner_row.get("final_answer") or reasoner_row.get("final_answer_gold")
    if not (rationale and rationale.strip()):
        return None
    prompt = build_reasoner_prompt(question, plan, include_plan_in_reasoner)
    response = ensure_final_suffix(rationale, final or "")
    return {
        "id": f"{rid}::reasoner",
        "role": "reasoner",
        "dataset": ds, "split": sp,
        "prompt": prompt,
        "response": response,
        "meta": mk_meta(normalizer_row or reasoner_row, source_mode),
    }

def make_verifier_example(ds: str, sp: str, rid: str, question: str,
                          reasoner_row: Dict[str, Any],
                          verifier_row: Dict[str, Any],
                          source_mode: str) -> Optional[Dict[str, Any]]:
    vout = verifier_row.get("verifier_output")
    if not (vout and vout.strip()):
        return None
    candidate = reasoner_row.get("reasoner_output") or ""
    gold = verifier_row.get("final_answer_gold") or reasoner_row.get("final_answer_gold") or reasoner_row.get("final_answer") or ""
    prompt = build_verifier_prompt(question, candidate, gold)
    response = vout.strip()
    return {
        "id": f"{rid}::verifier",
        "role": "verifier",
        "dataset": ds, "split": sp,
        "prompt": prompt,
        "response": response,
        "meta": mk_meta(verifier_row, source_mode),
    }

def make_normalizer_example(ds: str, sp: str, rid: str, question: str,
                            reasoner_row: Dict[str, Any],
                            verifier_row: Optional[Dict[str, Any]],
                            normalizer_row: Dict[str, Any],
                            source_mode: str) -> Optional[Dict[str, Any]]:
    nr = normalizer_row.get("normalized_rationale")
    if not (nr and nr.strip()):
        return None
    final = normalizer_row.get("final_answer_norm") or reasoner_row.get("final_answer_gold") or reasoner_row.get("final_answer") or ""
    candidate = reasoner_row.get("reasoner_output") or ""
    verdict = (verifier_row or {}).get("verifier_output") or ""
    prompt = build_normalizer_prompt(question, candidate, verdict)
    response = ensure_final_suffix(nr, final)
    return {
        "id": f"{rid}::normalizer",
        "role": "normalizer",
        "dataset": ds, "split": sp,
        "prompt": prompt,
        "response": response,
        "meta": mk_meta(normalizer_row, source_mode),
    }

def make_selector_example(ds: str, sp: str, rid: str,
                          router_row: Dict[str, Any],
                          source_mode: str) -> Optional[Dict[str, Any]]:
    sel_in = router_row.get("selector_input")
    sel_out = router_row.get("selector_output")
    if not (sel_in and sel_out):
        return None
    prompt = build_selector_prompt(sel_in.strip())
    response = sel_out.strip()
    return {
        "id": f"{rid}::selector",
        "role": "selector",
        "dataset": ds, "split": sp,
        "prompt": prompt,
        "response": response,
        "meta": mk_meta(router_row, source_mode),
    }

# ----------------------------
# Main
# ----------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--planner_root", type=str, default="data_intermediate/planner_data")
    ap.add_argument("--reasoner_root", type=str, default="data_intermediate/reasoner_data")
    ap.add_argument("--verifier_root", type=str, default="data_intermediate/verifier_data")
    ap.add_argument("--normalizer_root", type=str, default="data_intermediate/normalizer_data")
    ap.add_argument("--router_root", type=str, default="data_intermediate/router_data")

    ap.add_argument("--out_file", type=str, default="data_intermediate/unified_sft/all_roles.sft.jsonl")
    ap.add_argument("--datasets", type=str, default="gsm8k,math,multiarith")
    ap.add_argument("--splits", type=str, default="train,val,test")
    ap.add_argument("--limit", type=int, default=-1)
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--source_mode", choices=["none","minimal","raw"], default="minimal")
    ap.add_argument("--include_roles", type=str, default="planner,reasoner,verifier,normalizer,selector")
    ap.add_argument("--include_plan_in_reasoner", action="store_true")

    args = ap.parse_args()

    include_roles = set([r.strip() for r in args.include_roles.split(",") if r.strip()])
    ds_list = [s.strip() for s in args.datasets.split(",") if s.strip()]
    sp_list = [s.strip() for s in args.splits.split(",") if s.strip()]

    # Prepare resume set
    seen_ids = set()
    if args.resume and os.path.exists(args.out_file):
        for ex in load_jsonl(args.out_file):
            sid = ex.get("id")
            if sid: seen_ids.add(sid)

    total_written = 0
    for ds in ds_list:
        for sp in sp_list:
            # Spine = reasoner
            reasoner_path = find_file(args.reasoner_root, ds, sp, "reasoner")
            if not reasoner_path:
                print(f"[WARN] Missing reasoner file for {ds}/{sp}; skipping split.", file=sys.stderr)
                continue
            reasoners = load_jsonl(reasoner_path)
            if args.limit and args.limit > 0:
                reasoners = reasoners[:args.limit]
                print(f"[INFO] limiting {ds}/{sp} to {len(reasoners)} rows")

            planner_idx = {}
            if "planner" in include_roles:
                p = find_file(args.planner_root, ds, sp, "planner")
                if p: planner_idx = build_index(load_jsonl(p))

            verifier_idx = {}
            if "verifier" in include_roles:
                v = find_file(args.verifier_root, ds, sp, "verifier")
                if v: verifier_idx = build_index(load_jsonl(v))

            normalizer_idx = {}
            if "normalizer" in include_roles:
                n = find_file(args.normalizer_root, ds, sp, "normalizer")
                if n: normalizer_idx = build_index(load_jsonl(n))

            router_idx = {}
            if "selector" in include_roles:
                r = find_file(args.router_root, ds, sp, "router")
                if r: router_idx = build_index(load_jsonl(r))

            batch: List[Dict[str, Any]] = []

            for row in tqdm(reasoners, desc=f"{ds}/{sp}"):
                rid = str(row.get("id"))
                q = row.get("input_question") or row.get("question") or ""
                plan = row.get("plan") or row.get("planner_output")

                prow = planner_idx.get(rid) if planner_idx else None
                vrow = verifier_idx.get(rid) if verifier_idx else None
                nrow = normalizer_idx.get(rid) if normalizer_idx else None
                srow = router_idx.get(rid) if router_idx else None

                # --- Planner
                if "planner" in include_roles and prow:
                    ex = make_planner_example(ds, sp, rid, q, prow, args.source_mode)
                    if ex and ((not args.resume) or (ex["id"] not in seen_ids)):
                        batch.append(ex); seen_ids.add(ex["id"])

                # --- Reasoner (always from spine, optionally with normalized fallback)
                if "reasoner" in include_roles:
                    ex = make_reasoner_example(ds, sp, rid, q, plan, row, nrow, args.source_mode, args.include_plan_in_reasoner)
                    if ex and ((not args.resume) or (ex["id"] not in seen_ids)):
                        batch.append(ex); seen_ids.add(ex["id"])

                # --- Verifier
                if "verifier" in include_roles and vrow:
                    ex = make_verifier_example(ds, sp, rid, q, row, vrow, args.source_mode)
                    if ex and ((not args.resume) or (ex["id"] not in seen_ids)):
                        batch.append(ex); seen_ids.add(ex["id"])

                # --- Normalizer
                if "normalizer" in include_roles and nrow:
                    ex = make_normalizer_example(ds, sp, rid, q, row, vrow, nrow, args.source_mode)
                    if ex and ((not args.resume) or (ex["id"] not in seen_ids)):
                        batch.append(ex); seen_ids.add(ex["id"])

                # --- Selector (router)
                if "selector" in include_roles and srow:
                    ex = make_selector_example(ds, sp, rid, srow, args.source_mode)
                    if ex and ((not args.resume) or (ex["id"] not in seen_ids)):
                        batch.append(ex); seen_ids.add(ex["id"])

                if len(batch) >= 2000:
                    write_jsonl(args.out_file, batch, append=True)
                    total_written += len(batch)
                    batch = []

            if batch:
                write_jsonl(args.out_file, batch, append=True)
                total_written += len(batch)

            print(f"[DONE] {ds}/{sp} merged. Total so far: {total_written}")

    print(f"[ALL DONE] Wrote {total_written} examples to {args.out_file}")
    if total_written == 0:
        print("[NOTE] No rows written — check roots/filenames and --include_roles filters.", file=sys.stderr)

if __name__ == "__main__":
    main()
