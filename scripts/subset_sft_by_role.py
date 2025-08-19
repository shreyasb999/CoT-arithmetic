#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Subset a unified SFT JSONL file by dataset/role/split with optional
stratification and verifier INCORRECT balancing.

Example:
  python scripts/subset_sft_by_role.py \
    --in data_intermediate/unified_sft/all_roles.sft.jsonl \
    --out data_intermediate/unified_sft/all_roles_math1k.sft.jsonl \
    --include_datasets "math,gsm8k" \
    --include_roles "planner,reasoner,verifier,normalizer,selector" \
    --include_splits "train" \
    --role_max "planner=1500,reasoner=2000,verifier=1000,normalizer=1000,selector=1000" \
    --dataset_role_max "math:reasoner=5000" \
    --stratify_fields subject,level \
    --incorrect_frac 0.50 \
    --seed 42
"""

import argparse
import json
import os
import random
import sys
from collections import defaultdict
from typing import Dict, Iterable, List, Optional, Tuple

# ----------------------------
# IO
# ----------------------------

def load_jsonl(path: str) -> Iterable[dict]:
    with open(path, "r", encoding="utf-8") as f:
        for ln, line in enumerate(f, 1):
            s = line.strip()
            if not s:
                continue
            try:
                yield json.loads(s)
            except Exception as e:
                print(f"[WARN] JSON parse failed at {path}:{ln}: {e}", file=sys.stderr)

def write_jsonl(path: str, rows: Iterable[dict]) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

# ----------------------------
# Helpers
# ----------------------------

def parse_csv_list(s: Optional[str]) -> Optional[List[str]]:
    if not s:
        return None
    return [t.strip() for t in s.split(",") if t.strip()]

def parse_role_max_map(s: Optional[str]) -> Dict[str, int]:
    """
    planner=1500,reasoner=2000
    """
    out: Dict[str, int] = {}
    if not s:
        return out
    for part in s.split(","):
        part = part.strip()
        if not part:
            continue
        if "=" not in part:
            continue
        k, v = part.split("=", 1)
        k = k.strip()
        try:
            out[k] = int(v)
        except Exception:
            pass
    return out

def parse_dataset_role_max(s: Optional[str]) -> Dict[Tuple[str, str], int]:
    """
    math:reasoner=5000,gsm8k:verifier=800
    """
    out: Dict[Tuple[str, str], int] = {}
    if not s:
        return out
    for part in s.split(","):
        part = part.strip()
        if not part:
            continue
        if "=" not in part or ":" not in part:
            continue
        left, v = part.split("=", 1)
        ds, role = left.split(":", 1)
        ds = ds.strip()
        role = role.strip()
        try:
            out[(ds, role)] = int(v)
        except Exception:
            pass
    return out

def _get_nested(d, path, default=None):
    cur = d
    for k in path.split("."):
        if not isinstance(cur, dict) or k not in cur:
            return default
        cur = cur[k]
    return cur

def extract_field_with_fallback(ex: dict, name: str):
    """
    Return a value for 'name' using several fallbacks. If not found, return "unknown".
    """
    v = ex.get(name, None)
    if v not in (None, ""):
        return v
    # Common places in our pipeline
    for p in (
        f"meta.{name}",
        f"meta.source.{name}",
        f"meta.source_reasoner.{name}",
        f"meta.source_verifier.{name}",
        f"meta.prov.{name}",
    ):
        vv = _get_nested(ex, p, None)
        if vv not in (None, ""):
            return vv
    return "unknown"

def is_verifier_incorrect(ex: dict) -> Optional[bool]:
    if ex.get("role") != "verifier":
        return None
    resp = ex.get("response", "") or ""
    if "<<VERDICT: INCORRECT" in resp:
        return True
    if "<<VERDICT: CORRECT" in resp:
        return False
    return None  # unknown

def stratified_pick(pool: List[dict], fields: List[str], k: int, rng: random.Random) -> List[dict]:
    """
    Balanced sampling across groups defined by fields; missing fields map to "unknown".
    Round-robin take one from each group.
    """
    groups: Dict[Tuple, List[dict]] = defaultdict(list)
    for ex in pool:
        key = tuple(extract_field_with_fallback(ex, f) for f in fields)
        groups[key].append(ex)
    # Shuffle groups
    for lst in groups.values():
        rng.shuffle(lst)

    target = len(pool) if (k is None or k <= 0 or k > len(pool)) else k
    picked: List[dict] = []
    keys = list(groups.keys())
    # Round-robin
    while len(picked) < target and keys:
        new_keys = []
        for key in keys:
            if groups[key]:
                picked.append(groups[key].pop())
                if len(picked) >= target:
                    break
                new_keys.append(key)
        keys = new_keys
    return picked

def choose_subset_for_group(
    pool: List[dict],
    cap: Optional[int],
    strat_fields: Optional[List[str]],
    rng: random.Random,
    role: str,
    incorrect_frac: float,
) -> List[dict]:
    """
    Select a subset from pool with optional stratification and, for verifier,
    incorrect/correct balancing.
    """
    if not pool:
        return []

    if cap is None or cap <= 0 or cap >= len(pool):
        # Nothing to cap; keep all (but still optionally re-shuffle to avoid positional bias)
        rng.shuffle(pool)
        return pool

    if role == "verifier":
        inc = [ex for ex in pool if is_verifier_incorrect(ex) is True]
        cor = [ex for ex in pool if is_verifier_incorrect(ex) is False]
        unk = [ex for ex in pool if is_verifier_incorrect(ex) is None]
        # Use unknown as correct by default for balancing purposes
        cor = cor + unk

        want_inc = int(round(cap * max(0.0, min(1.0, incorrect_frac))))
        want_cor = cap - want_inc

        rng.shuffle(inc)
        rng.shuffle(cor)

        picked_inc = inc[:min(want_inc, len(inc))]
        picked_cor = cor[:min(want_cor, len(cor))]

        chosen = picked_inc + picked_cor
        # If still short (not enough INCORRECT/CORRECT), top up from remaining
        remaining = [ex for ex in pool if ex not in chosen]
        rng.shuffle(remaining)
        while len(chosen) < cap and remaining:
            chosen.append(remaining.pop())

        # Apply stratification *after* balancing (to keep exact counts)
        if strat_fields:
            # re-apply strat across chosen if you want even spread; otherwise return chosen
            # Here we keep chosen as-is to preserve the INCORRECT ratio.
            return chosen[:cap]
        else:
            return chosen[:cap]

    # Non-verifier roles
    if strat_fields:
        return stratified_pick(pool, strat_fields, cap, rng)
    else:
        rng.shuffle(pool)
        return pool[:cap]

# ----------------------------
# Main
# ----------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="in_file", required=True, help="Input unified SFT JSONL")
    ap.add_argument("--out", dest="out_file", required=True, help="Output subset JSONL")
    ap.add_argument("--include_datasets", default=None, help='CSV list, e.g. "gsm8k,math"')
    ap.add_argument("--include_roles", default=None, help='CSV list, e.g. "planner,reasoner,verifier,normalizer,selector"')
    ap.add_argument("--include_splits", default=None, help='CSV list, e.g. "train,val"')
    ap.add_argument("--role_max", default=None, help='Per-role cap applied per dataset, e.g. "planner=1500,reasoner=2000"')
    ap.add_argument("--dataset_role_max", default=None, help='Per (dataset,role) override, e.g. "math:reasoner=5000,gsm8k:verifier=800"')
    ap.add_argument("--stratify_fields", default=None, help="CSV fields for stratification (e.g., 'subject,level')")
    ap.add_argument("--incorrect_frac", type=float, default=0.5, help="Fraction of verifier examples that should be INCORRECT (0..1)")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--report_only", action="store_true", help="Only print counts; do not write file")
    args = ap.parse_args()

    rng = random.Random(args.seed)

    include_datasets = parse_csv_list(args.include_datasets)
    include_roles = parse_csv_list(args.include_roles)
    include_splits = parse_csv_list(args.include_splits)
    strat_fields = parse_csv_list(args.stratify_fields)

    role_cap_map = parse_role_max_map(args.role_max)           # role -> cap (per dataset)
    ds_role_cap_map = parse_dataset_role_max(args.dataset_role_max)  # (dataset,role) -> cap

    # Load + filter
    rows = list(load_jsonl(args.in_file))
    print(f"[INFO] Loaded {len(rows)} total examples")

    filt: List[dict] = []
    for ex in rows:
        ds = ex.get("dataset", "")
        rl = ex.get("role", "")
        sp = ex.get("split", "")
        if include_datasets and ds not in include_datasets:
            continue
        if include_roles and rl not in include_roles:
            continue
        if include_splits and sp not in include_splits:
            continue
        filt.append(ex)

    print(f"[INFO] After filters -> {len(filt)} examples")

    # Group by (dataset, role)
    groups: Dict[Tuple[str, str], List[dict]] = defaultdict(list)
    for ex in filt:
        groups[(ex.get("dataset",""), ex.get("role",""))].append(ex)

    # Show available counts
    print("[INFO] Available (dataset,role) counts within filters:")
    for (ds, rl), pool in sorted(groups.items()):
        print(f"  {ds}/{rl}: {len(pool)}")

    # Select per group with caps
    chosen_all: List[dict] = []
    for (ds, rl), pool in sorted(groups.items()):
        # Determine cap for this (dataset, role)
        cap = None
        if (ds, rl) in ds_role_cap_map:
            cap = ds_role_cap_map[(ds, rl)]
        elif rl in role_cap_map:
            cap = role_cap_map[rl]

        picked = choose_subset_for_group(
            pool=pool,
            cap=cap,
            strat_fields=strat_fields,
            rng=rng,
            role=rl,
            incorrect_frac=args.incorrect_frac,
        )
        print(f"[SELECT] {ds}/{rl}: picked {len(picked)} (cap={cap if cap is not None else 'ALL'})")
        chosen_all.extend(picked)

    # Deduplicate (id + content)
    seen = set()
    deduped: List[dict] = []
    for ex in chosen_all:
        key = (ex.get("id",""), ex.get("role",""), ex.get("dataset",""), ex.get("split",""), ex.get("prompt",""), ex.get("response",""))
        if key in seen:
            continue
        seen.add(key)
        deduped.append(ex)

    print(f"[DONE] Total selected (deduped): {len(deduped)}")

    if args.report_only:
        print("[REPORT] report_only set; not writing output file.")
        return

    write_jsonl(args.out_file, deduped)
    print(f"[WROTE] {args.out_file}")

if __name__ == "__main__":
    main()
