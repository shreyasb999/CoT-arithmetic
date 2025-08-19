#!/usr/bin/env python3
# scripts/convert_tests_to_unified.py
import argparse, json, re, sys
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

FINAL_TAIL_RE = re.compile(r"####\s*(.+)$")
BOXED_RE = re.compile(r"\\boxed\{(.+?)\}")
MATH_MODE_RE = re.compile(r"^\$+|\$+$")

def infer_dataset(rec: Dict[str, Any]) -> Optional[str]:
    ds = None
    rid = str(rec.get("id", ""))
    meta = rec.get("meta") or {}
    hf_name = (meta.get("hf_name") or "").lower()

    if rid.lower().startswith("gsm8k_") or hf_name == "gsm8k":
        ds = "gsm8k"
    elif rid.startswith("MATH_") or "math" in hf_name or "hendrycks-math" in hf_name:
        ds = "math"
    elif rid.lower().startswith("multiarith_") or "multiarith" in hf_name:
        ds = "multiarith"
    return ds

def pick_question(rec: Dict[str, Any]) -> Optional[str]:
    if rec.get("question"): return rec["question"]
    meta = rec.get("meta") or {}
    orig = meta.get("orig") or {}
    # MATH sometimes uses "problem"
    if orig.get("problem"): return orig["problem"]
    if orig.get("question"): return orig["question"]
    return None

def extract_final_from_answer_text(ans_text: str) -> Tuple[Optional[str], Optional[str]]:
    """
    Returns (gold_final, rationale_gold)
    If '####' tail exists, we use the segment after it as final and return the
    text before it as rationale. Otherwise final=whole string, rationale=None.
    """
    if ans_text is None:
        return None, None
    # Normalize line endings
    txt = ans_text.strip()
    m = FINAL_TAIL_RE.search(txt)
    if m:
        final_raw = m.group(1).strip()
        rationale = txt[:m.start()].rstrip()
        return final_raw, rationale if rationale else None
    else:
        # Sometimes plain numeric/string answer without steps
        return txt, None

def clean_final(v: Optional[str]) -> Optional[str]:
    if v is None:
        return None
    s = v.strip()
    # Strip TeX \boxed{...}
    m = BOXED_RE.fullmatch(s)
    if m:
        s = m.group(1).strip()
    # Remove surrounding inline $...$ if present
    s = s.strip()
    s = s.strip("$")
    # Remove trailing period for simple numbers/words
    s = s.rstrip()
    return s

def pick_gold_answer(rec: Dict[str, Any]) -> Tuple[Optional[str], Optional[str]]:
    """
    Decide gold answer (final) and rationale_gold from various fields.
    Priority:
      1) top-level 'answer' (parse #### tail if present)
      2) meta.orig.answer (parse #### tail)
      3) meta.orig.final_ans (MultiArith)
    """
    # 1) top-level answer
    if isinstance(rec.get("answer"), str):
        final, rat = extract_final_from_answer_text(rec["answer"])
        return clean_final(final), rat

    meta = rec.get("meta") or {}
    orig = meta.get("orig") or {}

    # 2) orig.answer
    if isinstance(orig.get("answer"), str):
        final, rat = extract_final_from_answer_text(orig["answer"])
        return clean_final(final), rat

    # 3) MultiArith: final_ans
    if isinstance(orig.get("final_ans"), str):
        return clean_final(orig["final_ans"]), None

    # 4) Some MATH files might put it at top-level 'answer' numeric already (handled above)
    # If not found, None
    return None, None

def pick_subject_level(rec: Dict[str, Any]) -> Tuple[Optional[str], Optional[Any]]:
    meta = rec.get("meta") or {}
    orig = meta.get("orig") or {}
    subject = orig.get("subject") or meta.get("subject") or rec.get("subject")
    level = orig.get("level") or meta.get("level") or rec.get("level")
    return subject, level

def unify_record(rec: Dict[str, Any], split: str) -> Optional[Dict[str, Any]]:
    dataset = infer_dataset(rec)
    q = pick_question(rec)
    gold, rationale = pick_gold_answer(rec)
    subject, level = pick_subject_level(rec)

    if not dataset or not q or gold is None:
        return None

    out = {
        "id": str(rec.get("id", "")),
        "dataset": dataset,
        "split": split,
        "question": q.strip(),
        "gold_answer": clean_final(gold),
    }
    if subject is not None:
        out["subject"] = subject
    if level is not None:
        out["level"] = level
    if rationale:
        out["rationale_gold"] = rationale
    return out

def main():
    ap = argparse.ArgumentParser(description="Convert mixed test JSONL files (GSM8K/MATH/MultiArith) to a unified JSONL.")
    ap.add_argument("--inputs", nargs="+", required=True, help="Input JSONL files (one or more).")
    ap.add_argument("--out", required=True, help="Output unified JSONL file.")
    ap.add_argument("--split", default="test", help="Split field for all outputs (default: test).")
    ap.add_argument("--keep_invalid", action="store_true", help="Write invalid lines (that cannot be unified) to a .rejects file next to --out.")
    args = ap.parse_args()

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    rejects_path = out_path.with_suffix(out_path.suffix + ".rejects")

    total_in = 0
    kept = 0
    rejects = 0
    counts = {"gsm8k":0, "math":0, "multiarith":0}

    with out_path.open("w", encoding="utf-8") as fw:
        rej_f = rejects_path.open("w", encoding="utf-8") if args.keep_invalid else None
        try:
            for infile in args.inputs:
                with open(infile, "r", encoding="utf-8") as fr:
                    for line in fr:
                        line = line.strip()
                        if not line:
                            continue
                        total_in += 1
                        try:
                            rec = json.loads(line)
                        except Exception:
                            if rej_f: rej_f.write(line + "\n")
                            rejects += 1
                            continue
                        uni = unify_record(rec, args.split)
                        if uni is None:
                            if rej_f: rej_f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                            rejects += 1
                            continue
                        fw.write(json.dumps(uni, ensure_ascii=False) + "\n")
                        kept += 1
                        ds = uni["dataset"]
                        if ds in counts: counts[ds] += 1
        finally:
            if rej_f: rej_f.close()

    print(f"[DONE] Read {total_in} lines from {len(args.inputs)} file(s).")
    print(f"[OUT]  Wrote {kept} unified examples -> {str(out_path)}")
    if rejects:
        print(f"[WARN] {rejects} lines could not be unified.", file=sys.stderr)
        if args.keep_invalid:
            print(f"[INFO] Rejected lines saved to: {str(rejects_path)}")

    print("[STATS] per dataset:")
    for k,v in counts.items():
        print(f"  {k}: {v}")

if __name__ == "__main__":
    main()
