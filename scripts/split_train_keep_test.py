# scripts/split_train_keep_test.py
import argparse, json, os, random
from tqdm import tqdm

def read_jsonl(path):
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(l) for l in f]

def write_jsonl(recs, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for r in recs:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

def split_train_val(train_path, test_path, outdir, seed=42, val_ratio=0.1):
    recs = read_jsonl(train_path)
    random.Random(seed).shuffle(recs)
    n = len(recs)
    v = int(n * val_ratio)
    val = recs[:v]
    train = recs[v:]
    base = os.path.splitext(os.path.basename(train_path))[0]
    out_base = os.path.join(outdir, base)
    write_jsonl(train, os.path.join(out_base, "train.jsonl"))
    write_jsonl(val, os.path.join(out_base, "val.jsonl"))
    # copy test if provided
    if test_path:
        test_out = os.path.join(out_base, "test.jsonl")
        write_jsonl(read_jsonl(test_path), test_out)
    print(f"Wrote {out_base}/train.jsonl ({len(train)}), val.jsonl ({len(val)}), test.jsonl ({'copied' if test_path else 'none'})")

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("trainfile")
    p.add_argument("--testfile", default=None)
    p.add_argument("--outdir", default="data_split")
    p.add_argument("--seed", default=42, type=int)
    p.add_argument("--val_ratio", default=0.1, type=float)
    args = p.parse_args()
    split_train_val(args.trainfile, args.testfile, args.outdir, seed=args.seed, val_ratio=args.val_ratio)
