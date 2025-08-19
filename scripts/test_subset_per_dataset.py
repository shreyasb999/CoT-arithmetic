#!/usr/bin/env python3
# scripts/subset_per_dataset.py
import argparse, json, random, math
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Tuple

def stratum_key(rec: Dict[str, Any], fields: List[str]) -> Tuple:
    if not fields: 
        return ("__ALL__",)
    return tuple(rec.get(f, "__NA__") for f in fields)

def proportional_sample(items_by_stratum: Dict[Tuple, List[Dict]], k: int, seed: int) -> List[Dict]:
    """Proportional allocation with largest-remainder; then random sample per stratum."""
    total = sum(len(v) for v in items_by_stratum.values())
    if total <= k:
        # nothing to cut
        chosen = []
        for v in items_by_stratum.values():
            chosen.extend(v)
        random.shuffle(chosen)
        return chosen

    # initial quotas (floors) and remainders
    quotas = {}
    remainders = []
    allocated = 0
    for s, items in items_by_stratum.items():
        exact = (len(items) * k) / total
        q = min(len(items), int(math.floor(exact)))
        quotas[s] = q
        allocated += q
        remainders.append((exact - q, s))
    # distribute remaining by largest remainder
    remain = k - allocated
    remainders.sort(reverse=True)  # by remainder
    i = 0
    while remain > 0 and i < len(remainders):
        _, s = remainders[i]
        if quotas[s] < len(items_by_stratum[s]):
            quotas[s] += 1
            remain -= 1
        i += 1
        if i >= len(remainders) and remain > 0:
            i = 0  # loop in case of caps
    # sample within each stratum
    out = []
    for s, items in items_by_stratum.items():
        random.shuffle(items)
        take = min(quotas[s], len(items))
        out.extend(items[:take])
    return out

def main():
    ap = argparse.ArgumentParser(description="Subset unified JSONL to N examples per dataset.")
    ap.add_argument("--in", dest="in_file", required=True, help="Input unified JSONL (with fields: dataset, question, gold_answer, ...)")
    ap.add_argument("--out", dest="out_file", required=True, help="Output JSONL")
    ap.add_argument("--per_dataset", type=int, default=500, help="Max examples per dataset (default: 500)")
    ap.add_argument("--seed", type=int, default=42, help="RNG seed")
    ap.add_argument("--stratify", type=str, default="", help="Comma-separated fields to stratify by (e.g., 'subject,level')")
    args = ap.parse_args()

    random.seed(args.seed)
    fields = [f.strip() for f in args.stratify.split(",") if f.strip()]

    # read
    by_ds: Dict[str, List[Dict]] = defaultdict(list)
    total_in = 0
    with open(args.in_file, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line: continue
            total_in += 1
            try:
                rec = json.loads(line)
            except Exception:
                continue
            ds = rec.get("dataset")
            if not ds:
                continue
            by_ds[ds].append(rec)

    # subset
    selected: List[Dict] = []
    stats = {}
    for ds, items in by_ds.items():
        if not fields:
            # simple random sample
            if len(items) <= args.per_dataset:
                pick = items[:]
                random.shuffle(pick)
            else:
                pick = random.sample(items, args.per_dataset)
        else:
            # stratified across requested fields
            strata: Dict[Tuple, List[Dict]] = defaultdict(list)
            for r in items:
                strata[stratum_key(r, fields)].append(r)
            pick = proportional_sample(strata, args.per_dataset, args.seed)
        selected.extend(pick)
        stats[ds] = len(pick)

    # write
    Path(args.out_file).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out_file, "w", encoding="utf-8") as f:
        for r in selected:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f"[DONE] Read {total_in} lines; wrote {len(selected)} -> {args.out_file}")
    for ds in sorted(stats):
        print(f"  {ds}: {stats[ds]} (cap {args.per_dataset})")

if __name__ == "__main__":
    main()
