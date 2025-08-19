#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
eval_multirole.py

Evaluate a multi-role CoT model (selector -> planner -> reasoner -> verifier -> normalizer),
batched with model.generate and deterministic decoding by default.

Outputs:
- predictions.jsonl : per-example raw role outputs + predictions
- summary.json      : overall metrics, per-dataset metrics, per-subject metrics,
                      selector chain histogram, verifier confusion & PRF1

Example:

CUDA_VISIBLE_DEVICES=0,1 \
python scripts/eval_multirole.py \
  --unified_file data_intermediate/eval/unified_test_500.jsonl \
  --model_backend hf \
  --model_name_or_path outputs/merged_qwen3_8b_cot_gsm_multiarith \
  --out_dir eval_runs/qwen3_8b_gsm_multiarith_selector \
  --use_selector --fallback_default \
  --batch_size 4 \
  --planner_tokens 256 --reasoner_tokens 512 --verifier_tokens 256 --normalizer_tokens 256 --selector_tokens 128 \
  --calc_verify --calc_repair
"""

import os
import re
import sys
import json
import argparse
from typing import List, Dict, Any, Tuple, Optional

import torch
from tqdm import tqdm

# ----------------------------
# Prompts (match your SFT format)
# ----------------------------

PLANNER_SYS = (
    "<role:planner>\n"
    "Write a concise plan with numbered imperative steps. Each step must start with an action verb.\n"
    "Do not solve; just outline the steps.\n"
    "Enclose the plan between '<<' and '>>'.\n"
)

REASONER_SYS = (
    "<role:reasoner>\n"
    "Follow the given plan. Show numbered steps.\n"
    "If you compute, use lines like: [[calc: 12*7-5]] -> 79\n"
    "Finish with a line: #### <final answer>\n"
)

VERIFIER_SYS = (
    "<role:verifier>\n"
    "Given the problem and a candidate rationale + answer, decide correctness.\n"
    "Output exactly:\n"
    "<<VERDICT: CORRECT|INCORRECT>>\n"
    "<<EVIDENCE: ...>>\n"
    "<<FIXED_ANSWER: final_value_if_known_or_blank>>\n"
)

NORMALIZER_SYS = (
    "<role:normalizer>\n"
    "Given the problem and a candidate rationale + answer, output a clean, minimal rationale that keeps the correct math.\n"
    "Use calculator lines like [[calc: ...]] -> value when computing.\n"
    "Finish with a line: #### <final answer>\n"
    "Only include math needed to justify the final answer.\n"
)

SELECTOR_SYS = (
    "<role:selector>\n"
    "Decide which roles to call to solve the problem and where to stop.\n"
    "Output exactly one line like: CALLS: planner,reasoner,calculator,normalizer; HALT_AFTER: normalizer\n"
    "Allowed roles: planner,reasoner,verifier,normalizer,calculator.\n"
    "The sequence must be comma-separated and reasonable for the task.\n"
)

DEFAULT_CHAIN = ["planner", "reasoner", "verifier", "normalizer"]


# ----------------------------
# Calculator (safe)
# ----------------------------
import ast
import math as pymath

ALLOWED_FUNCS = {"sqrt": pymath.sqrt, "abs": abs, "round": round}
ALLOWED_NAMES = {"pi": pymath.pi, "e": pymath.e}

class SafeEval(ast.NodeVisitor):
    def visit(self, node):
        if isinstance(node, ast.Expression):
            return self.visit(node.body)
        elif isinstance(node, ast.Constant):  # Py3.8+
            if isinstance(node.value, (int, float)): return node.value
            raise ValueError("bad const")
        elif isinstance(node, ast.Num):  # <=3.7 compat
            return node.n
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
    expr = expr.strip()
    try:
        tree = ast.parse(expr, mode="eval")
        return float(SafeEval().visit(tree))
    except Exception:
        return None

CALC_LINE_RE = re.compile(r"\[\[calc:\s*(.*?)\s*\]\]\s*->\s*([^\n\r]+)")

def verify_and_optionally_repair_calc_blocks(text: str, do_verify: bool, do_repair: bool) -> Tuple[str, Dict[str, Any]]:
    if not (do_verify or do_repair):
        return text, {"checked": 0, "mismatches": 0, "repaired": 0, "parse_fail": 0}

    lines = text.splitlines()
    checked = mismatches = repaired = parse_fail = 0
    out_lines = []

    for line in lines:
        m = CALC_LINE_RE.search(line)
        if not m:
            out_lines.append(line)
            continue
        expr = m.group(1).strip()
        claimed_raw = m.group(2).strip()
        checked += 1

        got = safe_calc(expr)
        try:
            claimed = float(claimed_raw.replace(",", ""))
            claimed_is_num = True
        except:
            claimed_is_num = False

        if got is None:
            parse_fail += 1
            out_lines.append(line)
        else:
            if not claimed_is_num:
                out_lines.append(line)
            else:
                if abs(got - claimed) > 1e-6:
                    mismatches += 1
                    if do_repair:
                        newline = CALC_LINE_RE.sub(f"[[calc: {expr}]] -> {got}", line)
                        out_lines.append(newline)
                        repaired += 1
                    else:
                        out_lines.append(line)
                else:
                    out_lines.append(line)

    return "\n".join(out_lines), {
        "checked": checked,
        "mismatches": mismatches,
        "repaired": repaired,
        "parse_fail": parse_fail,
    }


# ----------------------------
# Data loading
# ----------------------------

def read_jsonl(path: str) -> List[Dict[str, Any]]:
    out = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line: continue
            out.append(json.loads(line))
    return out

def load_unified_or_raw(args) -> List[Dict[str, Any]]:
    if args.unified_file:
        print(f"[LOAD] unified test file: {args.unified_file}")
        items = read_jsonl(args.unified_file)
        print(f"[INFO] Loaded {len(items)} test items.")
        return items

    # legacy path (data_root/datasets/splits)
    items = []
    for ds in args.datasets.split(","):
        for sp in args.splits.split(","):
            path = os.path.join(args.data_root, ds, f"{sp}.jsonl")
            if not os.path.isfile(path):
                print(f"[WARN] Missing file: {path}")
                continue
            rows = read_jsonl(path)
            for r in rows:
                q = r.get("question") # or r.get("input_question") or r.get("problem")
                a = r.get("gold_answer") # or r.get("answer") or r.get("final_answer") 
                items.append({
                    "id": r.get("id", f"{ds}_{sp}_{len(items)}"),
                    "dataset": ds,
                    "split": sp,
                    "question": q,
                    "answer": a,
                    "subject": (r.get("meta", {}) or {}).get("subject", ""),
                    "level": (r.get("meta", {}) or {}).get("level", ""),
                })
    print(f"[INFO] Loaded {len(items)} test items.")
    return items


# ----------------------------
# Parsing helpers
# ----------------------------

FINAL_RE = re.compile(r"####\s*(.+)\s*$")
VERDICT_RE = re.compile(r"<<VERDICT:\s*(CORRECT|INCORRECT)\s*>>", re.I)
FIXED_RE = re.compile(r"<<FIXED_ANSWER:\s*(.*?)\s*>>", re.S)

def extract_final(text: str) -> Optional[str]:
    m = FINAL_RE.search(text or "")
    if not m: return None
    return m.group(1).strip()

def normalize_ans(ans: str) -> str:
    if ans is None: return ""
    return ans.strip().replace("\n", " ").replace(" ", "").rstrip(".")

def match_answer(pred: str, gold: str) -> bool:
    return normalize_ans(pred) == normalize_ans(gold)

def parse_selector(text: str) -> Tuple[List[str], Optional[str]]:
    if not text: return [], None
    m = re.search(r"CALLS:\s*([a-z,\s]+);?\s*HALT_AFTER:\s*([a-z]+)", text, re.I)
    if not m: return [], None
    calls = [p.strip().lower() for p in m.group(1).split(",") if p.strip()]
    halt_after = m.group(2).strip().lower()
    return calls, halt_after


import re

STOP_TOKENS = ["\n<role:", "\n<<", "<<<END>>>", "<<END>>"]  # generic hard stops

PLANNER_BLOCK_RE = re.compile(r"<<(.*?)(?:>>|$)", re.DOTALL)
VERIFIER_RE = re.compile(
    r"<<VERDICT:\s*(CORRECT|INCORRECT)\s*>>\s*"
    r"<<EVIDENCE:\s*(.*?)\s*>>\s*"
    r"<<FIXED_ANSWER:\s*(.*?)\s*>>",
    re.DOTALL | re.IGNORECASE
)
FINAL_LINE_RE = re.compile(r"####[^\n]*")

def _first_idx(text: str, needles):
    idxs = [text.find(n) for n in needles]
    idxs = [i for i in idxs if i >= 0]
    return min(idxs) if idxs else -1

def clamp_generic(text: str) -> str:
    # chop at the first hard stop token to avoid echoes
    i = _first_idx(text, STOP_TOKENS)
    return text[:i] if i >= 0 else text

def sanitize_planner(text: str) -> str:
    # keep only the first << ... >> block; re-wrap if needed
    m = PLANNER_BLOCK_RE.search(text)
    if m:
        block = m.group(1).strip()
        return f"<<\n{block}\n>>"
    # fallback: keep up to first >>
    i = text.find(">>")
    return text[:i+2] if i >= 0 else clamp_generic(text)

def sanitize_reasoner(text: str) -> str:
    """
    Heuristics to clean reasoner outputs:
    - strip role/noise tags
    - collapse duplicate consecutive lines
    - cut trailing chatter after the first '#### <final>' line or '<<<END>>>' if present
    """
    if not text:
        return text

    t = text.replace("\r\n", "\n")

    # Remove obvious role/marker junk
    t = re.sub(r"</?assistant>", "", t, flags=re.IGNORECASE)
    t = re.sub(r"<role:[^>]+>", "", t)
    t = re.sub(r"<<<?[^>]+>>>?", "", t)  # e.g., <<<END>>> or <<VERIFIER:...>>

    # If there's a '#### final' line, keep everything up to and including it
    m = re.search(r"^####\s.*$", t, flags=re.MULTILINE)
    if m:
        t = t[:m.end()]
    else:
        # Else, if there is an <<<END>>> marker, cut there
        m2 = re.search(r"<<<END>>>", t)
        if m2:
            t = t[:m2.start()]

    # Collapse exact duplicate consecutive lines
    lines, prev = [], None
    for ln in t.splitlines():
        s = ln.strip()
        if s == prev:
            continue
        lines.append(ln)
        prev = s

    return "\n".join(lines).strip()


def sanitize_verifier(text: str) -> str:
    # keep the first full triple; normalize casing/spacing
    m = VERIFIER_RE.search(text)
    if not m:
        # soft salvage: try to truncate at the first triple-ish header
        i = _first_idx(text, ["<<VERDICT:", "<<EVIDENCE:", "<<FIXED_ANSWER:"])
        return clamp_generic(text[:i] if i >= 0 else text)
    verdict = m.group(1).upper()
    evidence = m.group(2).strip()
    fixed = m.group(3).strip()
    return f"<<VERDICT: {verdict}>>\n<<EVIDENCE: {evidence}>>\n<<FIXED_ANSWER: {fixed}>>"

def sanitize_normalizer(text: str) -> str:
    """
    Clean the normalizer output:
    - strip role/assistant tags and <<...>> control blocks
    - cut after the first '#### <final>' if present (or '<<<END>>>')
    - collapse consecutive duplicate lines
    """
    if not text:
        return text
    t = text.replace("\r\n", "\n")

    # Remove obvious junk/control tokens
    t = re.sub(r"</?assistant>", "", t, flags=re.IGNORECASE)
    t = re.sub(r"<role:[^>]+>", "", t)
    t = re.sub(r"<<<?[^>]+>>>?", "", t)  # e.g., <<<END>>>, <<VERDICT:...>>

    # Keep everything up to the first final line if present
    m = re.search(r"^####\s.*$", t, flags=re.MULTILINE)
    if m:
        t = t[:m.end()]
    else:
        m2 = re.search(r"<<<END>>>", t)
        if m2:
            t = t[:m2.start()]

    # Collapse exact duplicate consecutive lines
    lines, prev = [], None
    for ln in t.splitlines():
        s = ln.strip()
        if s == prev:
            continue
        lines.append(ln)
        prev = s

    return "\n".join(lines).strip()

SELECTOR_CALLS_RE = re.compile(
    r"CALLS:\s*([a-z, ]+)", re.IGNORECASE
)
SELECTOR_HALT_RE = re.compile(
    r"HALT_AFTER:\s*([a-z]+)", re.IGNORECASE
)

def sanitize_selector(text: str) -> str:
    """
    Normalize a free-form selector output into:
      'CALLS: role1,role2,...; HALT_AFTER: roleX'
    Keeps only allowed roles, dedupes, canonical order, first occurrence.
    """
    if not text:
        return ""

    s = text.strip()
    # strip code fences and xml-ish tags/brackets
    s = re.sub(r"`{3}.*?`{3}", " ", s, flags=re.S)
    s = s.replace("<<", " ").replace(">>", " ")
    s_low = s.lower()

    # canonical role order we allow
    order = ["planner", "reasoner", "calculator", "verifier", "normalizer"]
    allowed = set(order)

    # 1) Try to read a 'calls:' list first
    m_calls = re.search(r"calls\s*:\s*([^\n;><]+)", s_low)
    roles = []
    if m_calls:
        raw = m_calls.group(1)
        roles = [r.strip() for r in raw.split(",") if r.strip()]
    else:
        # fallback: infer from any keywords present (in canonical order)
        roles = [r for r in order if r in s_low]

    # filter to allowed, dedupe preserving canonical order
    seen = set()
    clean_calls = []
    for r in order:
        if r in roles and r in allowed and r not in seen:
            seen.add(r)
            clean_calls.append(r)

    # 2) HALT_AFTER
    m_halt = re.search(r"halt[_\-\s]*after\s*:\s*([a-z_]+)", s_low)
    halt = m_halt.group(1).strip() if m_halt else None
    if halt not in allowed:
        halt = None
    if not halt:
        # sensible defaults
        if "normalizer" in clean_calls:
            halt = "normalizer"
        elif clean_calls:
            halt = clean_calls[-1]  # last role in the chain
        else:
            halt = "normalizer"

    # 3) Build canonical single-line output
    out = ""
    if clean_calls:
        out += f"CALLS: {','.join(clean_calls)}"
    if halt:
        out += ("; " if out else "") + f"HALT_AFTER: {halt}"
    return out

def parse_selector(text: str):
    # keep first CALLS and HALT_AFTER only; drop any extra chatter
    calls = []
    halt = None
    m1 = SELECTOR_CALLS_RE.search(text)
    if m1:
        raw = m1.group(1)
        calls = [c.strip().lower() for c in raw.split(",") if c.strip()]
    m2 = SELECTOR_HALT_RE.search(text)
    if m2:
        halt = m2.group(1).lower()
    # sanitized “selector_output” string for logging
    # sel_out = ""
    # if calls:
    #     sel_out += f"CALLS: {','.join(calls)}"
    # if halt:
    #     sel_out += ("" if not sel_out else "; ") + f"HALT_AFTER: {halt}"
    return calls, halt


def choose_predicted_answer(verifier_text: str,
                            reasoner_text: str,
                            normalizer_text: str) -> Optional[str]:
    """
    Prefer the value the normalizer/ reasoner actually showed on '#### ...'.
    If verifier FIXED_ANSWER is blank, back-fill from those.
    """
    # try normalizer final line
    n_final = normalize_numeric_str(extract_final_line(normalizer_text))
    if n_final:
        return n_final
    # fallback to reasoner final line
    r_final = normalize_numeric_str(extract_final_line(reasoner_text))
    if r_final:
        return r_final
    # fallback to verifier FIXED_ANSWER tag
    if verifier_text:
        m = re.search(r"<<FIXED_ANSWER:\s*(.*?)>>", verifier_text)
        if m:
            v = normalize_numeric_str(m.group(1))
            if v:
                return v
    return None

def clip_after_final(text: str) -> str:
    if not text:
        return text
    lines = text.splitlines()
    out = []
    for ln in lines:
        out.append(ln)
        if ln.strip().startswith("####"):
            break
    return "\n".join(out)

ANS_RE = re.compile(r"####\s*(.+)")
NUM_CLEAN_RE = re.compile(r"[,\s]*(hours?|hrs?|minutes?|mins?|meter(s)?|m|kg|g|dollars?|usd)$", re.I)

def extract_final_line(text: str) -> Optional[str]:
    if not text:
        return None
    m = ANS_RE.search(text)
    if m:
        return m.group(1).strip()
    # sometimes gold is a bare number/string; return stripped
    return text.strip() if text.strip() else None

def normalize_numeric_str(s: Optional[str]) -> Optional[str]:
    if s is None:
        return None
    x = s.strip()
    # remove $ and commas
    x = x.replace("$", "")
    x = x.replace(",", "")
    # drop common trailing units/words
    x = NUM_CLEAN_RE.sub("", x).strip()
    # normalize simple trailing period
    x = x[:-1] if x.endswith(".") else x
    return x

# ----------------------------
# HF model loader & batched generation
# ----------------------------
def load_hf(args):
    from transformers import AutoTokenizer, AutoModelForCausalLM
    extra = {}
    if torch.cuda.device_count() > 1:
        extra["device_map"] = "auto"
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name_or_path,
        torch_dtype=torch.bfloat16 if args.bf16 else None,
        low_cpu_mem_usage=True,
        trust_remote_code=True,
        **extra,
    )
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, use_fast=True, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    if not hasattr(model.config, "use_cache") or model.config.use_cache:
        model.config.use_cache = True
    return tokenizer, model

@torch.no_grad()
def batched_generate(
    model,
    tokenizer,
    prompts: List[str],
    max_new_tokens: int,
    do_sample: bool,
    temperature: Optional[float],
    top_k: Optional[int],
) -> List[str]:
    if len(prompts) == 0:
        return []

    # ---- make decoder-only generation happy ----
    prev_side = getattr(tokenizer, "padding_side", "right")
    tokenizer.padding_side = "left"   # avoid right-padding warning
    if tokenizer.pad_token_id is None and tokenizer.eos_token_id is not None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    enc = tokenizer(
        prompts,
        padding=True,
        truncation=True,
        return_tensors="pt",
    )
    device = model.device if hasattr(model, "device") else torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )
    enc = {k: v.to(device) for k, v in enc.items()}

    # ---- generation args ----
    gen_kwargs = dict(
        max_new_tokens=int(max_new_tokens),
        do_sample=bool(do_sample),
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
        no_repeat_ngram_size=4,     # mild anti-loop
        repetition_penalty=1.05,    # mild anti-loop
    )
    if do_sample:
        if temperature is not None:
            gen_kwargs["temperature"] = float(temperature)
        if top_k is not None:
            gen_kwargs["top_k"] = int(top_k)

    out_ids = model.generate(**enc, **gen_kwargs)

    # slice off the prompt
    cut = enc["input_ids"].shape[1]
    new_ids = out_ids[:, cut:]

    texts = tokenizer.batch_decode(new_ids, skip_special_tokens=True)

    # restore original padding_side (optional)
    tokenizer.padding_side = prev_side

    return texts


# ----------------------------
# Role prompt builders
# ----------------------------
# ---------- Tight, format-locked prompt builders ----------

def build_selector_prompt(question: str) -> str:
    return (
        "<role:selector>\n"
        "You are the SELECTOR in a modular solver.\n"
        "Choose which modules to call and where to stop. No commentary.\n"
        "\n"
        "Rules:\n"
        "- Available modules: planner, reasoner, verifier, normalizer, calculator\n"
        "- Output EXACTLY two lines and NOTHING else.\n"
        "- Line 1: CALLS: <comma-separated subset in execution order>\n"
        "- Line 2: HALT_AFTER: <one module from the CALLS set>\n"
        "- Prefer full chain for word problems: planner,reasoner,calculator,verifier,normalizer\n"
        "- For trivial arithmetic you may skip planner; for already-explicit answers, you may call normalizer only.\n"
        "- Do NOT repeat or explain. Do NOT include angle brackets other than the required ones below.\n"
        "\n"
        "Format:\n"
        "CALLS: planner,reasoner,calculator,verifier,normalizer\n"
        "HALT_AFTER: normalizer\n"
        "\n"
        f"Problem:\n{question}\n"
    )


def build_planner_prompt(question: str) -> str:
    return (
        "<role:planner>\n"
        "Task: Produce a concise plan (3–6 steps) to solve the problem. Do NOT compute.\n"
        "Strict rules:\n"
        "- DO NOT restate the problem.\n"
        "- DO NOT compute or include any numeric results.\n"
        "- Each step must be a single sentence.\n"
        "- Output ONLY the plan block bounded by << and >>. No extra text.\n"
        "\n"
        "Output format (and nothing else):\n"
        "<<\n"
        "1. ...\n"
        "2. ...\n"
        "3. ...\n"
        ">>\n"
        "\n"
        f"Problem:\n{question}\n"
    )


def build_reasoner_prompt(question: str, plan: str) -> str:
    return (
        "<role:reasoner>\n"
        "Follow the plan EXACTLY. Show numbered steps. Use calculator markers ONLY for purely numeric expressions.\n"
        "Strict rules:\n"
        "- Do NOT print the plan text. Do NOT add headers or role tags.\n"
        "- If a step is symbolic (has variables), write the step WITHOUT a calc marker.\n"
        "- If a step is numeric, use: [[calc: <numeric expression>]] -> <value>\n"
        "- Emit exactly one final line: #### <final answer only> (no units, no $ signs in the number itself).\n"
        "- No repetition. No extra lines before or after the final line.\n"
        "\n"
        "Output format example:\n"
        "1. Short step description. [[calc: 12*7-5]] -> 79\n"
        "2. Short step description. [[calc: 79+1]] -> 80\n"
        "#### 80\n"
        "\n"
        f"Problem:\n{question}\n"
        f"Plan:\n{plan}\n"
    )


def build_verifier_prompt(question: str, candidate_rationale: str, gold_answer: str) -> str:
    return (
        "<role:verifier>\n"
        "Decide whether the candidate reaches the correct final answer.\n"
        "Strict rules:\n"
        "- Output EXACTLY three lines. NOTHING else.\n"
        "- Line 1: <<VERDICT: CORRECT>> or <<VERDICT: INCORRECT>>\n"
        "- Line 2: <<EVIDENCE: one short sentence; if any calc mismatch, cite the FIRST one like '30*3 -> 99 (recalc 90)'; "
        "otherwise say 'Final answer matches gold (X).'>>\n"
        "- Line 3: <<FIXED_ANSWER: <final value if known, else blank>>\n"
        "- Do NOT rewrite or quote the candidate. No extra whitespace.\n"
        "\n"
        f"Problem:\n{question}\n"
        "<<CANDIDATE_BEGIN>>\n"
        f"{candidate_rationale}\n"
        "<<CANDIDATE_END>>\n"
        f"<<GOLD_ANSWER: {gold_answer}>>\n"
    )


def build_normalizer_prompt(question: str, candidate_rationale: str) -> str:
    return (
        "<role:normalizer>\n"
        "Rewrite the candidate into a minimal, clean rationale and emit a single final answer line.\n"
        "Strict rules:\n"
        "- Keep at most 8 short lines BEFORE the final answer.\n"
        "- Use calculator markers ONLY for numeric expressions: [[calc: <numeric>]] -> <value> (no $ or units in the value).\n"
        "- Remove role tags, control tokens, and meta markers. No repetition.\n"
        "- End with EXACTLY one line: #### <final answer only> (no units or currency symbols in the number itself).\n"
        "- Output ONLY the normalized rationale and that final line. Nothing else.\n"
        "\n"
        f"Problem:\n{question}\n"
        "Candidate rationale:\n"
        f"{candidate_rationale}\n"
    )



# ----------------------------
# Metrics helpers
# ----------------------------
def _ensure(d: Dict, k: str, init: Dict[str, float]) -> Dict[str, float]:
    if k not in d: d[k] = init.copy()
    return d[k]

def _verifier_counts_dict() -> Dict[str, int]:
    return {"tp": 0, "tn": 0, "fp": 0, "fn": 0}

def _add_verifier_conf(counts: Dict[str, int], expected_correct: bool, predicted: str):
    # expected_correct == True  -> expected verdict CORRECT
    # predicted in {"CORRECT","INCORRECT"}
    if predicted not in ("CORRECT", "INCORRECT"):
        return
    if expected_correct and predicted == "CORRECT":
        counts["tp"] += 1
    elif (not expected_correct) and predicted == "INCORRECT":
        counts["tn"] += 1
    elif (not expected_correct) and predicted == "CORRECT":
        counts["fp"] += 1
    elif expected_correct and predicted == "INCORRECT":
        counts["fn"] += 1

def _summarize_conf(c: Dict[str, int]) -> Dict[str, float | int]:
    tp, tn, fp, fn = c.get("tp",0), c.get("tn",0), c.get("fp",0), c.get("fn",0)
    total = tp + tn + fp + fn
    acc = (tp + tn) / total if total else 0.0
    prec = tp / (tp + fp) if (tp + fp) else 0.0
    rec  = tp / (tp + fn) if (tp + fn) else 0.0
    f1   = (2*prec*rec/(prec+rec)) if (prec+rec) else 0.0
    return {"tp": tp, "tn": tn, "fp": fp, "fn": fn, "total": total,
            "accuracy": round(acc, 6), "precision": round(prec, 6),
            "recall": round(rec, 6), "f1": round(f1, 6)}


# ----------------------------
# Evaluation main
# ----------------------------
def main():
    p = argparse.ArgumentParser()
    # data
    p.add_argument("--unified_file", type=str, default=None, help="Unified jsonl: id,dataset,split,question,answer[,subject,level]")
    p.add_argument("--data_root", type=str, default="data_split")
    p.add_argument("--datasets", type=str, default="gsm8k,math,multiarith")
    p.add_argument("--splits", type=str, default="test")
    p.add_argument("--out_dir", type=str, required=True)

    # model
    p.add_argument("--model_backend", type=str, default="hf", choices=["hf"])
    p.add_argument("--model_name_or_path", type=str, required=True)
    p.add_argument("--bf16", action="store_true", help="bfloat16 inference")

    # batching & generation
    p.add_argument("--batch_size", type=int, default=4)
    p.add_argument("--planner_tokens", type=int, default=192)
    p.add_argument("--reasoner_tokens", type=int, default=512)
    p.add_argument("--verifier_tokens", type=int, default=128)
    p.add_argument("--normalizer_tokens", type=int, default=128)
    p.add_argument("--selector_tokens", type=int, default=64)

    # decoding controls
    p.add_argument("--do_sample", action="store_true", default=False)
    p.add_argument("--temperature", type=float, default=None)
    p.add_argument("--top_k", type=int, default=None)

    # routing & calc
    p.add_argument("--use_selector", action="store_true")
    p.add_argument("--fallback_default", action="store_true")
    p.add_argument("--calc_verify", action="store_true")
    p.add_argument("--calc_repair", action="store_true")

    args = p.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    out_jsonl = os.path.join(args.out_dir, "predictions.jsonl")
    out_summary = os.path.join(args.out_dir, "summary.json")

    # load data
    items = load_unified_or_raw(args)

    # load model
    if args.model_backend != "hf":
        print("[ERROR] Only --model_backend hf is supported.")
        sys.exit(1)
    print(f"[HF] Loading {args.model_name_or_path} ...")
    tokenizer, model = load_hf(args)
    device_descr = str(model.device) if hasattr(model, "device") else "cuda"
    print(f"Device set to use {device_descr}")

    # overall metrics
    total = 0
    correct_final = 0
    correct_reasoner_only = 0
    reasoner_final_parsed = 0
    normalizer_final_parsed = 0

    verifier_total = 0
    verifier_right = 0
    verifier_conf_overall = _verifier_counts_dict()

    selector_stats = {
        "total": 0,
        "has_selector": 0,
        "parsed": 0,
        "called_reasoner": 0,
        "chains": {},
    }

    # per-dataset & per-subject
    per_dataset = {}  # ds -> dict
    per_subject = {}  # subj -> dict

    def ensure_ds(ds: str):
        _ensure(per_dataset, ds, {
            "total": 0,
            "correct_final": 0,
            "reasoner_correct": 0,
            "verifier_total": 0,
        })
        if "verifier_conf" not in per_dataset[ds]:
            per_dataset[ds]["verifier_conf"] = _verifier_counts_dict()
        if "selector" not in per_dataset[ds]:
            per_dataset[ds]["selector"] = {"chains": {}, "has_selector": 0, "parsed": 0, "called_reasoner": 0}

    def ensure_subj(subj: str):
        _ensure(per_subject, subj, {
            "total": 0,
            "correct_final": 0,
            "reasoner_correct": 0,
            "verifier_total": 0,
        })
        if "verifier_conf" not in per_subject[subj]:
            per_subject[subj]["verifier_conf"] = _verifier_counts_dict()

    # open out
    fout = open(out_jsonl, "w", encoding="utf-8")

    # batches
    B = args.batch_size
    num_batches = (len(items) + B - 1) // B
    pbar = tqdm(range(num_batches), total=num_batches, desc="eval", unit="batch")

    for bi in pbar:
        batch = items[bi*B : (bi+1)*B]
        if not batch: break

        # ----- SELECTOR -----
        chains: Dict[str, Dict[str, Any]] = {}
        if args.use_selector:
            selector_prompts = [build_selector_prompt(x["question"]) for x in batch]

            # Deterministic selector for evaluation
            sel_raw_outs = batched_generate(
                model, tokenizer, selector_prompts, args.selector_tokens,
                do_sample=False, temperature=None, top_k=None
            )

            # ⬇️ Sanitize BEFORE parsing
            sel_outs = [sanitize_selector(s) for s in sel_raw_outs]

            for x, s_raw, s in zip(batch, sel_raw_outs, sel_outs):
                selector_stats["total"] += 1
                selector_stats["has_selector"] += 1

                calls, halt_after = parse_selector(s)  # parse sanitized text
                if calls:
                    selector_stats["parsed"] += 1
                if "reasoner" in calls:
                    selector_stats["called_reasoner"] += 1

                chain_key = ",".join(calls) + f";halt={halt_after}"
                selector_stats["chains"][chain_key] = selector_stats["chains"].get(chain_key, 0) + 1

                # Store both (sanitized and raw) if you want to debug
                chains[x["id"]] = {
                    "calls": calls,
                    "halt_after": halt_after,
                    "selector_output": s,         # sanitized
                    "selector_output_raw": s_raw  # optional, for inspection
                }

                ds = x.get("dataset","")
                ensure_ds(ds)
                per_dataset[ds]["selector"]["has_selector"] += 1
                if calls: per_dataset[ds]["selector"]["parsed"] += 1
                if "reasoner" in calls: per_dataset[ds]["selector"]["called_reasoner"] += 1
                per_dataset[ds]["selector"]["chains"][chain_key] = per_dataset[ds]["selector"]["chains"].get(chain_key, 0) + 1
        else:
            for x in batch:
                chains[x["id"]] = {"calls": DEFAULT_CHAIN[:], "halt_after": "normalizer", "selector_output": ""}

        # Fallback to default chain if selector failed/empty
        if args.fallback_default:
            for x in batch:
                ent = chains.get(x["id"], {})
                if not ent.get("calls"):
                    ent["calls"] = DEFAULT_CHAIN[:]
                    ent["halt_after"] = "normalizer"
                    chains[x["id"]] = ent


        # ----- PLANNER -----
        need_planner = [x for x in batch if "planner" in chains[x["id"]]["calls"]]
        planner_map: Dict[str, str] = {}
        if need_planner:
            prompts = [build_planner_prompt(x["question"]) for x in need_planner]

            # Deterministic by default for eval
            outs = batched_generate(
                model, tokenizer, prompts,
                max_new_tokens=args.planner_tokens,
                do_sample=False,                     # <- keep False for evaluation
                temperature=None, top_k=None
            )

            # Optional but recommended: sanitize to remove repetition / enforce format
            outs = [sanitize_planner(o) for o in outs]

            for x, o in zip(need_planner, outs):
                planner_map[x["id"]] = o


        # ----- REASONER -----
        need_reasoner = [x for x in batch if "reasoner" in chains[x["id"]]["calls"]]
        reasoner_map: Dict[str, str] = {}
        reasoner_final_map: Dict[str, Optional[str]] = {}

        if need_reasoner:
            prompts, order_ids = [], []
            for x in need_reasoner:
                plan_txt = planner_map.get(x["id"], "")
                rp = build_reasoner_prompt(x["question"], plan_txt)
                prompts.append(rp); order_ids.append(x["id"])

            outs = batched_generate(
                model, tokenizer, prompts, args.reasoner_tokens,
                args.do_sample, args.temperature, args.top_k
            )

            # ⬇️ sanitize immediately after generation
            outs_sanitized = [sanitize_reasoner(o) for o in outs]

            for rid, o_clean in zip(order_ids, outs_sanitized):
                reasoner_map[rid] = o_clean
                reasoner_final_map[rid] = extract_final(o_clean)

        # Optional calculator verify/repair; sanitize again (idempotent) then re-extract final
        if args.calc_verify or args.calc_repair:
            for rid, text in list(reasoner_map.items()):
                fixed, _ = verify_and_optionally_repair_calc_blocks(
                    text, args.calc_verify, args.calc_repair
                )
                fixed = sanitize_reasoner(fixed)  # keep it tidy after edits
                reasoner_map[rid] = fixed
                reasoner_map[rid] = clip_after_final(reasoner_map[rid])
                reasoner_final_map[rid] = extract_final(fixed)


        # ----- VERIFIER -----
        need_verifier = [x for x in batch if "verifier" in chains[x["id"]]["calls"]]
        verifier_map: Dict[str, str] = {}
        verdict_map: Dict[str, str] = {}
        fixed_ans_map: Dict[str, str] = {}
        if need_verifier:
            prompts, order_ids = [], []
            for x in need_verifier:
                cand = reasoner_map.get(x["id"], "")
                vp = build_verifier_prompt(x["question"], cand, x.get("answer",""))
                prompts.append(vp); order_ids.append(x["id"])
            outs = batched_generate(model, tokenizer, prompts, args.verifier_tokens, args.do_sample, args.temperature, args.top_k)

            outs = [sanitize_verifier(o) for o in outs]

            for rid, o in zip(order_ids, outs):
                verifier_map[rid] = o
                vm = VERDICT_RE.search(o or "")
                verdict = vm.group(1).upper() if vm else ""
                fm = FIXED_RE.search(o or "")
                fval = (fm.group(1).strip() if fm else "") or ""
                verdict_map[rid] = verdict
                fixed_ans_map[rid] = fval

        # ----- NORMALIZER -----
        need_normalizer = [x for x in batch if "normalizer" in chains[x["id"]]["calls"]]
        normalizer_map: Dict[str, str] = {}
        normalizer_final_map: Dict[str, Optional[str]] = {}

        if need_normalizer:
            prompts, order_ids = [], []
            for x in need_normalizer:
                cand = reasoner_map.get(x["id"], "")
                nprompt = build_normalizer_prompt(x["question"], cand)
                prompts.append(nprompt); order_ids.append(x["id"])

            outs = batched_generate(
                model, tokenizer, prompts, args.normalizer_tokens,
                args.do_sample, args.temperature, args.top_k
            )

            # ⬇️ sanitize immediately after generation
            outs_clean = [sanitize_normalizer(o) for o in outs]

            for rid, o in zip(order_ids, outs_clean):
                normalizer_map[rid] = o
                fv = extract_final(o)
                normalizer_final_map[rid] = fv

            # Optional calculator verify/repair; sanitize again (idempotent), then re-extract final
            if args.calc_verify or args.calc_repair:
                for rid, text in list(normalizer_map.items()):
                    fixed, _ = verify_and_optionally_repair_calc_blocks(
                        text, args.calc_verify, args.calc_repair
                    )
                    fixed = sanitize_normalizer(fixed)
                    normalizer_map[rid] = fixed
                    normalizer_map[rid] = clip_after_final(normalizer_map[rid])
                    normalizer_final_map[rid] = extract_final(fixed)


        # ----- finalize & metrics -----
        for x in batch:
            total += 1
            ds = x.get("dataset","")
            subj = (x.get("subject","") or "").strip()
            ensure_ds(ds)
            if subj: ensure_subj(subj)

            qid = x["id"]
            gold = x.get("gold_answer","")
            chain = chains[qid]
            calls = chain.get("calls", []) or []
            halt = chain.get("halt_after", "")
            selector_output = chain.get("selector_output","")

            planner_out = planner_map.get(qid, "")
            reasoner_out = reasoner_map.get(qid, "")
            reasoner_final = reasoner_final_map.get(qid, None)
            if reasoner_final is not None:
                reasoner_final_parsed += 1

            verifier_out = verifier_map.get(qid, "")
            verdict = verdict_map.get(qid, "")
            fixed_ans = fixed_ans_map.get(qid, "")

            normalizer_out = normalizer_map.get(qid, "")
            normalizer_final = normalizer_final_map.get(qid, None)
            if normalizer_out and normalizer_final is not None:
                normalizer_final_parsed += 1

            # predictions
            pred_before_verify = normalizer_final if normalizer_out else (reasoner_final or "")
            pred_after_verify = pred_before_verify
            if verdict == "INCORRECT" and fixed_ans.strip():
                pred_after_verify = fixed_ans.strip()

            # correctness flags
            reasoner_ok = (reasoner_final is not None) and match_answer(reasoner_final, gold)
            final_ok = (pred_after_verify is not None) and match_answer(pred_after_verify, gold)

            if reasoner_ok: correct_reasoner_only += 1
            if final_ok:    correct_final += 1

            # dataset counters
            per_dataset[ds]["total"] += 1
            if reasoner_ok: per_dataset[ds]["reasoner_correct"] += 1
            if final_ok:    per_dataset[ds]["correct_final"] += 1

            # subject counters
            if subj:
                per_subject[subj]["total"] += 1
                if reasoner_ok: per_subject[subj]["reasoner_correct"] += 1
                if final_ok:    per_subject[subj]["correct_final"] += 1

            # verifier metrics
            if verifier_out:
                verifier_total += 1
                cand_ok = reasoner_ok
                expected_verdict = "CORRECT" if cand_ok else "INCORRECT"
                if verdict == expected_verdict:
                    verifier_right += 1

                _add_verifier_conf(verifier_conf_overall, cand_ok, verdict)

                per_dataset[ds]["verifier_total"] += 1
                _add_verifier_conf(per_dataset[ds]["verifier_conf"], cand_ok, verdict)
                if subj:
                    per_subject[subj]["verifier_total"] += 1
                    _add_verifier_conf(per_subject[subj]["verifier_conf"], cand_ok, verdict)


            pred_final = choose_predicted_answer(
                verifier_map.get(x["id"], ""),
                reasoner_map.get(x["id"], ""),
                normalizer_map.get(x["id"], "")
            ) or ""

            # keep the original verifier fixed_answer too (for debugging)
            ver_fixed = ""
            vt = verifier_map.get(x["id"], "")
            if vt:
                m = re.search(r"<<FIXED_ANSWER:\s*(.*?)>>", vt)
                if m:
                    ver_fixed = normalize_numeric_str(m.group(1)) or ""

            # write record
            rec = {
                "id": qid,
                "dataset": ds,
                "split": x.get("split",""),
                "subject": subj,
                "level": x.get("level",""),
                "question": x.get("question",""),
                "answer_gold": gold,

                "selector_output": selector_output,
                "calls": calls,
                "halt_after": halt,

                "planner_output": planner_out,
                "reasoner_output": reasoner_out,
                "verifier_output": verifier_out,
                "normalizer_output": normalizer_out,

                "pred_before_verify": pred_before_verify or "",
                "pred_after_verify":  pred_after_verify or "",

                "reasoner_final": reasoner_final or "",
                "normalizer_final": normalizer_final or "",
                "verdict": verdict or "",
                "fixed_answer": fixed_ans or "",
                "pred_final": pred_final            # <-- THIS is the value you should score against gold
            }
            fout.write(json.dumps(rec, ensure_ascii=False) + "\n")

        # progress bar
        acc = (correct_final / total) if total else 0.0
        pbar.set_postfix_str(f"acc={acc:.3f}")

    fout.close()

    # summary
    summary = {
        "total": total,
        "final_accuracy": round(correct_final / total, 6) if total else 0.0,
        "reasoner_only_accuracy": round(correct_reasoner_only / total, 6) if total else 0.0,
        "verifier": {
            "total": verifier_total,
            "verdict_accuracy": round(verifier_right / verifier_total, 6) if verifier_total else 0.0,
            "confusion": _summarize_conf(verifier_conf_overall),
        },
        "selector": {  # global selector stats
            **selector_stats,
        },
        "notes": {
            "decoding": "greedy" if not args.do_sample else f"sampling (temp={args.temperature}, top_k={args.top_k})",
            "calc_verify": bool(args.calc_verify),
            "calc_repair": bool(args.calc_repair),
        },
        "per_dataset": {},
        "per_subject": {},
    }

    # per-dataset summary
    for ds, d in per_dataset.items():
        ds_total = d["total"]
        ds_final = d["correct_final"]
        ds_reasoner = d["reasoner_correct"]
        vc = d["verifier_conf"]
        ds_sel = d.get("selector", {})
        summary["per_dataset"][ds] = {
            "total": ds_total,
            "final_accuracy": round(ds_final / ds_total, 6) if ds_total else 0.0,
            "reasoner_only_accuracy": round(ds_reasoner / ds_total, 6) if ds_total else 0.0,
            "verifier": {
                "total": d.get("verifier_total", 0),
                "confusion": _summarize_conf(vc),
            },
            "selector": ds_sel,
        }

    # per-subject summary
    for subj, d in per_subject.items():
        stotal = d["total"]
        sfinal = d["correct_final"]
        sreasoner = d["reasoner_correct"]
        vc = d["verifier_conf"]
        summary["per_subject"][subj] = {
            "total": stotal,
            "final_accuracy": round(sfinal / stotal, 6) if stotal else 0.0,
            "reasoner_only_accuracy": round(sreasoner / stotal, 6) if stotal else 0.0,
            "verifier": {
                "total": d.get("verifier_total", 0),
                "confusion": _summarize_conf(vc),
            },
        }

    with open(out_summary, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print("[DONE] Wrote:")
    print(f"  predictions: {out_jsonl}")
    print(f"  summary:     {out_summary}")
    print(f"  final acc:   {summary['final_accuracy']:.4f} | reasoner-only: {summary['reasoner_only_accuracy']:.4f}")


if __name__ == "__main__":
    main()
