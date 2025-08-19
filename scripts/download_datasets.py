#!/usr/bin/env python3
# scripts/download_datasets.py

import argparse, json, os, sys
from typing import Optional, Dict, Any, List
from datasets import load_dataset
from tqdm import tqdm

# Supported datasets with possible HuggingFace IDs/configs
DATA_MAP = {
    "gsm8k": {"candidates":[("gsm8k","main"),("gsm8k","socratic")]},
    "asdiv": {"candidates":[("EleutherAI/asdiv",None),("asdiv",None)]},
    "multiarith": {"candidates":[("ChilleD/MultiArith",None)]},
    "addsub": {"candidates":[("allenai/lila","addsub")]},
    "MATH": {"candidates":[("nlile/hendrycks-MATH-benchmark",None),("MATH",None)]},
    "MathBench": {"candidates":[("opencompass/MathBench",None)]},
    "Omni-MATH": {"candidates":[("KbsdJames/Omni-MATH",None)]},
}

def save_jsonl(records: List[Dict[str, Any]], outpath: str):
    os.makedirs(os.path.dirname(outpath), exist_ok=True)
    with open(outpath, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

def _pick_field(ex, keys):
    for k in keys:
        if k in ex and ex[k] is not None:
            return ex[k]
    for v in ex.values():
        if isinstance(v, str) and "?" in v:
            return v
    return ""

def load_and_write(dataset_key: str, outdir: str,
                   split: str, hf_name: Optional[str],
                   hf_config: Optional[str], data_file: Optional[str],
                   limit: Optional[int]) -> str:
    # 1) Data file override
    if data_file:
        print(f"Loading from data file: {data_file}")
        if data_file.endswith((".json", ".jsonl")):
            ds = load_dataset("json", data_files=data_file, split=split)
        elif data_file.endswith((".csv", ".tsv")):
            ds = load_dataset("csv", data_files=data_file, split=split)
        else:
            ds = load_dataset("json", data_files=data_file, split=split)
    else:
        # 2) Explicit HF override
        candidates = []
        if hf_name:
            candidates = [(hf_name, hf_config)]
        elif dataset_key in DATA_MAP:
            candidates = DATA_MAP[dataset_key]["candidates"]
        else:
            raise ValueError(f"Unknown dataset key '{dataset_key}'")

        last_error = None
        for repo, cfg in candidates:
            try:
                print(f"Trying HF dataset: repo='{repo}', config='{cfg}'")
                ds = load_dataset(repo, cfg, split=split) if cfg else load_dataset(repo, split=split)
                hf_name, hf_config = repo, cfg
                break
            except Exception as e:
                last_error = e
                msg = str(e)
                if "requires a config" in msg or "Dataset scripts are no longer supported" in msg:
                    print(f"  ➡ Error loading '{repo}': {msg}")
                    print("  Suggestion: use --data_file with raw JSON/CSV or retry with correct HF repo/config.")
                    sys.exit(1)
                print(f"Failed {repo}, trying next...")
        else:
            raise RuntimeError(f"All HF candidate attempts failed: {last_error}")

    records = []
    for i, ex in enumerate(tqdm(ds, desc="Iterate dataset")):
        if limit and i >= limit:
            break
        q = _pick_field(ex, ["question", "problem", "input", "text", "question_text"])
        a = _pick_field(ex, ["answer", "label", "output", "solution","final_ans"])
        rec = {
            "id": f"{dataset_key}_{i}",
            "question": q,
            "answer": a,
            "meta": {"hf_name": hf_name, "hf_config": hf_config, "orig": ex}
        }
        records.append(rec)

    outpath = os.path.join(outdir, f"{dataset_key}_{split}.jsonl")
    save_jsonl(records, outpath)
    print(f"Wrote {len(records)} entries to {outpath}")
    return outpath

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True,
                        help="key (e.g. gsm8k, MATH, Omni-MATH) or full HF name")
    parser.add_argument("--outdir", default="data_raw")
    parser.add_argument("--split", default="train")
    parser.add_argument("--hf_name", default=None)
    parser.add_argument("--hf_config", default=None)
    parser.add_argument("--data_file", default=None,
                        help="path or URL to raw JSON/CSV if HF fails")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--info", action="store_true",
                        help="Display dataset mapping and exit")
    args = parser.parse_args()

    if args.info and args.dataset in DATA_MAP:
        print("Key:", args.dataset)
        print("Candidates:", DATA_MAP[args.dataset]["candidates"])
        sys.exit(0)

    try:
        load_and_write(
            dataset_key=args.dataset,
            outdir=args.outdir,
            split=args.split,
            hf_name=args.hf_name,
            hf_config=args.hf_config,
            data_file=args.data_file,
            limit=args.limit)
    except Exception as e:
        print("ERROR:", e, file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    main()
