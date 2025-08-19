#!/usr/bin/env python3
"""
create_reasoner_dataset.py

Builds the 'reasoner' dataset from raw math corpora (GSM8K, MATH, MultiArith) and
the previously created planner outputs.

For each (dataset, split):
  - Load raw examples from --data_root/{dataset}/{split}.jsonl (or .json)
  - Load planner outputs from --planner_root/{dataset}/{dataset}_{split}_planner.jsonl
  - Join by 'id'
  - Produce a normalized step-by-step Reasoner target that:
      * Executes the given plan
      * Uses calculator markers [[calc: expression]] -> value
      * Ends with a single final line "#### <final_answer>"
  - Validate format and optionally run a safe calculator check to verify calc lines.
  - Write JSONL to --out_dir/{dataset}/{dataset}_{split}_reasoner.jsonl

Two generation modes (choose via --gen_mode):
  - gold_prefer (default): normalize gold rationale when available; otherwise use LLM
  - llm_only: always use LLM from Problem + Plan (+ Final answer for MultiArith)

Backends:
  - HuggingFace (HF) transformers local inference with batching (recommended)
  - CLI subprocess (e.g., ollama) is supported but HF is default.

Requirements (typical):
  pip install -U transformers accelerate bitsandbytes sentencepiece regex tqdm pandas

Example:
python scripts/create_reasoner_dataset.py \
  --data_root ./data \
  --planner_root ./output/planner_data \
  --out_dir ./output/reasoner_data \
  --model_backend hf \
  --model_name_or_path "Qwen/Qwen2.5-Math-7B" \
  --device cuda:0 \
  --batch_size 8 \
  --gen_mode gold_prefer \
  --max_new_tokens 384
"""

import argparse
import os
import re
import sys
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from textwrap import dedent
from tqdm import tqdm

# ----------------------------
# HF client (batch, greedy)
# ----------------------------
try:
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM
    HF_AVAILABLE = True
except Exception:
    HF_AVAILABLE = False

class LLMClient:
    def generate(self, prompt: str, max_tokens: int = 256) -> str:
        raise NotImplementedError
    def generate_many(self, prompts: List[str], max_tokens: int = 256) -> List[str]:
        return [self.generate(p, max_tokens=max_tokens) for p in prompts]

class HFTransformersClient(LLMClient):
    def __init__(self, model_name_or_path: str, device: str = "cuda:0", max_batch_size: int = 8):
        if not HF_AVAILABLE:
            raise RuntimeError("HF transformers not installed.")
        print(f"[HF] Loading {model_name_or_path} on {device} ...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, use_fast=False)
        if self.tokenizer.pad_token_id is None:
            # prefer eos as pad
            if self.tokenizer.eos_token is not None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            else:
                self.tokenizer.add_special_tokens({"pad_token": "[PAD]"})
        self.tokenizer.padding_side = "left"
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name_or_path,
            device_map="auto",
            torch_dtype=torch.float16,
            trust_remote_code=True,
            low_cpu_mem_usage=True,
        )
        if getattr(self.model.config, "pad_token_id", None) is None:
            self.model.config.pad_token_id = self.tokenizer.pad_token_id
        self.max_batch_size = max_batch_size
        print("[HF] Ready.")

    def generate_many(self, prompts: List[str], max_tokens: int = 256) -> List[str]:
        out_texts = []
        for i in range(0, len(prompts), self.max_batch_size):
            chunk = prompts[i:i+self.max_batch_size]
            enc = self.tokenizer(
                chunk,
                return_tensors="pt",
                padding=True,          # left-padded
                truncation=False
            )
            enc = {k: v.to(self.model.device) for k, v in enc.items()}

            with torch.no_grad():
                gen = self.model.generate(
                    **enc,
                    max_new_tokens=max_tokens,
                    min_new_tokens=32,
                    do_sample=False,
                    eos_token_id=self.tokenizer.eos_token_id,
                    pad_token_id=self.tokenizer.pad_token_id,
                )

            # ✅ per-row prompt length using attention mask (sum of 1s)
            prompt_lens = enc["attention_mask"].sum(dim=1)  # shape: (batch,)

            # decode each row using its own slice
            for row_idx in range(gen.size(0)):
                start = int(prompt_lens[row_idx].item())
                out_ids = gen[row_idx, start:]  # continuation only
                text = self.tokenizer.decode(out_ids, skip_special_tokens=True)
                # fallback safety: if empty, decode last max_tokens tokens
                if not text.strip() and out_ids.numel() == 0:
                    tail = gen[row_idx, -max_tokens:]
                    text = self.tokenizer.decode(tail, skip_special_tokens=True)
                out_texts.append(text)
        return out_texts

    def generate(self, prompt: str, max_tokens: int = 256) -> str:
        # single-item convenience wrapper used by the repair pass
        return self.generate_many([prompt], max_tokens=max_tokens)[0]


# ----------------------------
# CLI client (optional)
# ----------------------------
import subprocess, shlex
class CLIClient(LLMClient):
    def __init__(self, cli_cmd_tmpl: str):
        # template must include {prompt}
        if "{prompt}" not in cli_cmd_tmpl:
            raise ValueError("--cli_cmd must contain {prompt}")
        self.tmpl = cli_cmd_tmpl
    def generate(self, prompt: str, max_tokens: int = 256) -> str:
        safe = prompt.replace('"','\\"')
        cmd = self.tmpl.format(prompt=safe)
        proc = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        out, err = proc.communicate(timeout=120)
        if proc.returncode != 0:
            raise RuntimeError(f"CLI error: {err}")
        return out.strip()

# ----------------------------
# IO + utils
# ----------------------------
RE_GSM8K_FINAL = re.compile(r"####\s*([^\n#]+)")
RE_CHEVRON = re.compile(r"<<(.*?)>>", re.S)

def load_jsonl(path: str) -> List[Dict]:
    items = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line: continue
            items.append(json.loads(line))
    return items

def write_jsonl(items: List[Dict], path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for it in items:
            f.write(json.dumps(it, ensure_ascii=False) + "\n")

def normalize_ws(s: str) -> str:
    return re.sub(r"\s+", " ", (s or "")).strip()

def get_in(d, path):
    cur = d
    for k in path:
        if not isinstance(cur, dict) or k not in cur:
            return None
        cur = cur[k]
    return cur

def coalesce(*vals):
    for v in vals:
        if v is not None and v != "":
            return v
    return None

def extract_gsm8k_final(s: str) -> Optional[str]:
    if not s or not isinstance(s, str): return None
    m = RE_GSM8K_FINAL.search(s)
    return m.group(1).strip() if m else None

def parse_subject_from_unique_id(u: str) -> Optional[str]:
    if not u or not isinstance(u, str): return None
    parts = u.strip().split("/")
    if len(parts) >= 2:
        sub = parts[1].strip().replace("_"," ")
        return sub[:1].upper() + sub[1:].lower()
    return None

def detect_fields(example: Dict) -> Tuple[str, Optional[str], Optional[str], str, str]:
    """
    Returns (question, solution, final_answer, subject, level)
    Robust across top-level/meta/orig; parses GSM8K '#### final'.
    """
    q = coalesce(
        example.get("question"), example.get("problem"),
        get_in(example, ("orig","question")), get_in(example, ("orig","problem")),
        get_in(example, ("meta","orig","problem")), get_in(example, ("meta","orig","question"))
    )
    sol = coalesce(
        example.get("solution"),
        get_in(example, ("orig","solution")),
        get_in(example, ("meta","solution")),
        get_in(example, ("meta","orig","solution")),
    )
    final = coalesce(
        example.get("final_ans"), example.get("final_answer"),
        get_in(example, ("orig","final_ans")), get_in(example, ("orig","final_answer")),
        get_in(example, ("meta","final_ans")), get_in(example, ("meta","final_answer")),
        get_in(example, ("meta","orig","final_ans")), get_in(example, ("meta","orig","final_answer")),
    )
    # ANSWER may be rationale (GSM8K) or just final (MATH)
    ans_candidates = [
        example.get("answer"),
        get_in(example, ("orig","answer")),
        get_in(example, ("meta","answer")),
        get_in(example, ("meta","orig","answer")),
    ]
    def looks_step_by_step(s: str) -> bool:
        return bool(s and isinstance(s,str) and (("<<" in s) or ("\n" in s and len(s.splitlines())>=2)))
    for ans in ans_candidates:
        if not ans or not isinstance(ans, str): continue
        if looks_step_by_step(ans) and sol is None:
            sol = ans
            if final is None:
                maybe = extract_gsm8k_final(ans)
                if maybe: final = maybe
        elif final is None:
            if len(ans.splitlines()) == 1 and len(ans) <= 64:
                final = ans

    unique_id = coalesce(
        example.get("unique_id"),
        get_in(example, ("orig","unique_id")),
        get_in(example, ("meta","orig","unique_id")),
    )
    subject = coalesce(
        example.get("subject"),
        get_in(example, ("orig","subject")),
        get_in(example, ("meta","subject")),
        get_in(example, ("meta","orig","subject")),
    )
    level = coalesce(
        example.get("level"),
        get_in(example, ("orig","level")),
        get_in(example, ("meta","level")),
        get_in(example, ("meta","orig","level")),
        example.get("difficulty"),
        get_in(example, ("orig","difficulty")),
        get_in(example, ("meta","orig","difficulty")),
    )
    if (not subject) and unique_id:
        subject = parse_subject_from_unique_id(unique_id)
    if level is not None and not isinstance(level, str):
        level = str(level)
    return q or "", sol, final, (subject or ""), (level or "")

def slim_source_meta(ex: dict) -> dict:
    m = ex.get("meta", {}) if isinstance(ex.get("meta"), dict) else {}
    orig = m.get("orig", {}) if isinstance(m.get("orig"), dict) else {}
    return {
        "id": ex.get("id") or ex.get("idx") or orig.get("unique_id") or "",
        "hf_name": m.get("hf_name", ""),
        "hf_config": m.get("hf_config", ""),
        "original_index": m.get("original_index", None),
        "unique_id": orig.get("unique_id", ""),
    }

CALC_INLINE_RE = re.compile(
    r"""^(\d+\.\s*)(.*?)                 # 1: '1. ' prefix, 2: description
        (?:
          \(\s*([0-9\.\s\+\-\*\/\(\)]+?)\s*\)   # 3: (expr) form
          \s*[\.\,\;\:]*\s*->\s*([+-]?\d+(?:\.\d+)?)   # 4: value
        |
          :\s*([0-9\.\s\+\-\*\/\(\)]+?)\s*      # 5: ': expr' form
          \s*[\.\,\;\:]*\s*->\s*([+-]?\d+(?:\.\d+)?)   # 6: value
        )
        \s*\.?$                                # optional trailing period
    """,
    flags=re.X,
)

# --- unit/symbol normalization (used by NUM_EQ) ---
UNIT_PATTERNS = [
    r"\^\s*\\?circ",   # ^\circ (LaTeX)
    r"°",              # degree symbol
    r"\\,|\\!|\\;|\\:",# LaTeX spacing
]

SYMBOL_MAP = {
    "×": "*",
    "·": "*",
    "⋅": "*",
    "\\cdot": "*",
    "−": "-",          # U+2212
    "—": "-",
    "÷": "/",
}

# Identify names inside expressions and decide if expression is symbolic (has variables/functions)
_IDENT_RE = re.compile(r"[A-Za-z_\\]+")  # includes LaTeX commands like \sin
_ALLOWED_IDENTIFIERS = {"sqrt", "pi", "e"}  # purely numeric-safe names

def _extract_identifiers(expr: str) -> List[str]:
    # remove allowed sqrt( by normalizer; we still allow 'sqrt' token
    return _IDENT_RE.findall(expr)

def _is_symbolic_expr(expr: str) -> bool:
    """
    True if expr contains identifiers other than allowed numeric-safe ones (sqrt, pi, e).
    This catches sin, cos, x, n, etc., and LaTeX commands (\sin, \cos, ...).
    """
    if not expr:
        return False
    ids = set(x.strip("\\") for x in _extract_identifiers(expr))
    # if any identifier besides allowed shows up, treat as symbolic
    return any((tok not in _ALLOWED_IDENTIFIERS) for tok in ids)


def strip_units_and_symbols(s: str) -> str:
    t = s
    for k, v in SYMBOL_MAP.items():
        t = t.replace(k, v)
    for pat in UNIT_PATTERNS:
        t = re.sub(pat, "", t)
    t = t.replace("^", "**")  # caret power -> python power
    return re.sub(r"\s+", " ", t).strip()

def latex_frac_to_ascii(s: str) -> str:
    # \frac{a}{b} -> (a)/(b)
    return re.sub(r"\\frac\s*\{([^{}]+)\}\s*\{([^{}]+)\}", r"(\1)/(\2)", s)

def normalize_numeric_text(s: str) -> str:
    t = latex_frac_to_ascii(s)
    t = strip_units_and_symbols(t)
    return t

# --- NEW regexes (replace old ones) ---
BARE_EXPR_ARROW = re.compile(
    r"([0-9\.\s\+\-\*\/\(\)]+?)\s*[\.\,\;\:]*\s*->\s*([+-]?\d+(?:\.\d+)?)"
)
# numeric equality after normalization (LHS must be numeric-only once units/symbols removed)
NUM_EQ = re.compile(
    r"([0-9\.\s\+\-\*\/\(\)]+?)\s*=\s*([+-]?\d+(?:\.\d+)?)"
)

def inject_calc_markers(text: str) -> str:
    """
    For each *numbered* line lacking [[calc: ...]], append a calc marker if we can mine:
      - 'expr -> value' in that line; or
      - numeric-only equality 'expr = value' after normalization.
    We append to avoid mangling prose.
    """
    if not text.strip():
        return text
    out_lines = []
    for ln in text.splitlines():
        base = ln.rstrip()
        # only number-marked steps need calc markers
        if not re.match(r"^\d+\.\s+\S", base.strip()):
            out_lines.append(base)
            continue
        if "[[calc:" in base:
            out_lines.append(base)
            continue

        # Case A: direct 'expr -> val'
        m_arrow = BARE_EXPR_ARROW.search(base)
        if m_arrow:
            expr = m_arrow.group(1).strip()
            val  = m_arrow.group(2).strip()
            out_lines.append(f"{base}  [[calc: {expr}]] -> {val}")
            continue

        # Case B: numeric-only equality after normalization
        norm = normalize_numeric_text(base)
        m_eq = NUM_EQ.search(norm)
        if m_eq:
            expr = m_eq.group(1).strip()
            val  = m_eq.group(2).strip()
            # avoid variable-led equalities like 'G/16 = 4' by checking original text
            if not re.search(r"[A-Za-z]\s*/?\s*\d*\s*=\s*\d", base):
                out_lines.append(f"{base}  [[calc: {expr}]] -> {val}")
                continue

        out_lines.append(base)
    return "\n".join(out_lines)

_FINAL_SQRT_LATEX = re.compile(r"\\sqrt\s*\{([^{}]+)\}")
_FINAL_SQRT_UNICODE = re.compile(r"√\s*([A-Za-z0-9\(\)]+)")
_FINAL_FRAC = re.compile(r"\\frac\s*\{([^{}]+)\}\s*\{([^{}]+)\}")
_FINAL_PI = re.compile(r"\\pi\b", flags=re.I)

def _final_answer_to_expr(ans: str) -> str:
    """
    Convert a LaTeX-ish final answer into a pythonic expression we can put in [[calc: ...]]:
      - \sqrt{a} -> sqrt(a), √a -> sqrt(a)
      - \frac{a}{b} -> (a)/(b)
      - \pi -> pi
      - insert missing '*' between number and sqrt(...) or parentheses, e.g., 10sqrt(3) -> 10*sqrt(3)
      - map ^ -> **, \cdot, ×, ÷ handled by normalize_numeric_text upstream if you want
      - keep radicals/π; verifier will skip symbolic
    """
    if not ans:
        return ""
    expr = ans.strip()

    # keep the original final for RHS; we only transform the LHS to a "calcable" form
    # sqrt forms
    expr = _FINAL_SQRT_LATEX.sub(r"sqrt(\1)", expr)
    expr = _FINAL_SQRT_UNICODE.sub(r"sqrt(\1)", expr)

    # fractions
    expr = _FINAL_FRAC.sub(r"(\1)/(\2)", expr)

    # constants
    expr = _FINAL_PI.sub("pi", expr)

    # remove degree symbols / \circ for the expr side
    expr = re.sub(r"\^\s*\\?circ", "", expr)
    expr = expr.replace("°", "")

    # ^ -> **, \cdot/×/÷:
    expr = expr.replace("^", "**").replace("\\cdot", "*").replace("×", "*").replace("÷", "/")

    # insert '*' between number and 'sqrt' or '(' when missing (e.g., 10sqrt(3), 2(3+4))
    expr = re.sub(r"(\d)\s*(sqrt)", r"\1*\2", expr)
    expr = re.sub(r"(\d)\s*\(", r"\1*(", expr)

    # compress spaces
    expr = re.sub(r"\s+", " ", expr).strip()
    return expr

def ensure_global_calc_marker(text: str) -> str:
    """
    If there are no [[calc: ...]] in the text:
      1) search ALL lines for either 'expr -> val' or a numeric-only equality (after normalization)
         and append a synthetic numbered step with that calc;
      2) if still none, synthesize a calc from the final answer line (e.g., '10\\sqrt{3}')
         using a pythonic expr ('10*sqrt(3)') and put it before the #### line.
    The synthetic step is placed after the last numbered step, or as step 1 if none exist.
    """
    if not (text or "").strip():
        return text
    if "[[calc:" in text:
        return text

    lines = [ln.rstrip() for ln in text.splitlines()]
    # Find last numbered step index and its number
    last_num_idx = -1
    last_num = 0
    for i, ln in enumerate(lines):
        m = re.match(r"^\s*(\d+)\.\s+\S", ln)
        if m:
            last_num_idx = i
            last_num = int(m.group(1))

    # Find equality/arrow anywhere (paragraphs, bullets, etc.)
    best_expr, best_val = None, None
    for ln in lines:
        # 1) arrow 'expr -> val'
        m_arrow = re.search(r"([0-9\.\s\+\-\*\/\(\)]+?)\s*[\.\,\;\:]*\s*->\s*([^\n]+)", ln)
        if m_arrow:
            best_expr = m_arrow.group(1).strip()
            best_val  = m_arrow.group(2).strip()
            break
        # 2) numeric-only equality after normalization
        norm = normalize_numeric_text(ln)
        m_eq = re.search(r"([0-9\.\s\+\-\*\/\(\)]+?)\s*=\s*([+-]?\d+(?:\.\d+)?)", norm)
        if m_eq and not re.search(r"[A-Za-z]\s*/?\s*\d*\s*=\s*\d", ln):
            best_expr = m_eq.group(1).strip()
            best_val  = m_eq.group(2).strip()
            break

    # If found, append synthetic calc line
    if best_expr is not None and best_val is not None:
        step_no = (last_num + 1) if last_num_idx >= 0 else 1
        synth = f"{step_no}. Computation: [[calc: {best_expr}]] -> {best_val}"
        # insert before final #### line if it exists, else after last numbered or at top
        hash_idx = None
        for i, ln in enumerate(lines):
            if ln.strip().startswith("####"):
                hash_idx = i
                break
        if hash_idx is not None:
            insert_at = hash_idx
        elif last_num_idx >= 0:
            insert_at = last_num_idx + 1
        else:
            insert_at = 0
        lines.insert(insert_at, synth)
        return "\n".join(lines)

    # 3) Fallback: synthesize from final answer
    # Find final answer line
    fin = None
    fin_idx = None
    for i, ln in enumerate(lines):
        if ln.strip().startswith("####"):
            fin = ln.strip()[4:].strip()
            fin_idx = i
            break

    if fin:
        expr_from_final = _final_answer_to_expr(fin)
        if not expr_from_final:
            # try to salvage plain number
            nums = re.findall(r"[+-]?\d+(?:\.\d+)?", fin)
            if nums:
                expr_from_final = nums[-1]

        if expr_from_final:
            step_no = (last_num + 1) if last_num_idx >= 0 else 1
            synth = f"{step_no}. Computation: [[calc: {expr_from_final}]] -> {fin}"
            insert_at = fin_idx if fin_idx is not None else (last_num_idx + 1 if last_num_idx >= 0 else 0)
            lines.insert(insert_at, synth)
            return "\n".join(lines)

    # nothing to do
    return text


def has_calc(text: str) -> bool:
    return "[[calc:" in (text or "")

def count_numbered_steps(text: str) -> int:
    return sum(1 for ln in (text or "").splitlines() if re.match(r"^\d+\.\s+\S", ln.strip()))

# Clean odd markers and enforce final #### line
def cleanup_markers(text: str) -> str:
    if not text:
        return text
    t = text
    # remove BEGIN/END artifacts if model echoed them
    t = re.sub(r"<<<BEGIN>>>", "", t)
    t = re.sub(r"<<<END>>>\>+", "", t)      # handles <<<END>>>> etc.
    # remove leading lone '>' lines (some models prepend that)
    t = re.sub(r"(?m)^\s*>\s*\n", "", t)
    return t.strip()

def enforce_final_line(text: str, fallback_final: Optional[str]) -> Tuple[str, Optional[str]]:
    """
    Ensure the LAST line is exactly '#### <final_answer>'.
    If no explicit #### is present, use fallback_final; else last number seen.
    Removes any existing #### lines and any text after ####.
    """
    t = text or ""

    # If there's already a #### line, trim anything after the last ####
    lines = [ln for ln in t.splitlines()]
    last_hash_idx = None
    for i, ln in enumerate(lines):
        if ln.strip().startswith("####"):
            last_hash_idx = i
    if last_hash_idx is not None:
        lines = lines[: last_hash_idx + 1]
        t = "\n".join(lines)

    # Extract final if present; else try fallback / last number
    final = extract_final_line(t)
    if not final:
        if fallback_final and str(fallback_final).strip():
            final = str(fallback_final).strip()
        else:
            nums = re.findall(r"[+-]?\d+(?:\.\d+)?", t)
            if nums:
                final = nums[-1]

    # Drop any existing #### lines and re-append a single clean one
    lines = [ln for ln in t.splitlines() if not ln.strip().startswith("####")]
    if final:
        lines.append(f"#### {final}")
    return "\n".join(l for l in lines if l.strip()), final


def normalize_reasoner_text(raw_text: str, fallback_final: Optional[str]) -> Tuple[str, Optional[str]]:
    """
    Extract block if present; cleanup; enforce final line; inject calc markers.
    Returns (normalized_text, final_answer_used).
    """
    raw = (raw_text or "").strip()
    mblk = re.search(r"<<<BEGIN>>>(.*?)(?:<<<END>>>|$)", raw, flags=re.S)
    text = mblk.group(1).strip() if mblk else raw
    text = cleanup_markers(text)
    text, final_used = enforce_final_line(text, fallback_final=fallback_final)
    text = inject_calc_markers(text)
    return text, final_used




# ----------------------------
# Calculator (safe) - UPDATED
# ----------------------------
import re
import ast
import math
from typing import Optional, List, Tuple

# --- numeric normalization helpers (safe to keep local to this block) ---
_UNIT_PATTERNS = [
    r"\^\s*\\?circ",   # ^\circ (LaTeX)
    r"°",              # degree symbol
    r"\\,|\\!|\\;|\\:",# LaTeX spacing
]
_SYMBOL_MAP = {
    "×": "*", "·": "*", "⋅": "*", "\\cdot": "*",
    "−": "-", "—": "-",
    "÷": "/",
}
def _latex_frac_to_ascii(s: str) -> str:
    # \frac{a}{b} -> (a)/(b)
    return re.sub(r"\\frac\s*\{([^{}]+)\}\s*\{([^{}]+)\}", r"(\1)/(\2)", s)

def _strip_units_and_symbols(s: str) -> str:
    t = s
    for k, v in _SYMBOL_MAP.items():
        t = t.replace(k, v)
    for pat in _UNIT_PATTERNS:
        t = re.sub(pat, "", t)
    t = t.replace("^", "**")  # caret power -> python power
    return re.sub(r"\s+", " ", t).strip()

def normalize_numeric_text(s: str) -> str:
    return _strip_units_and_symbols(_latex_frac_to_ascii(s or ""))

# --- allow a tiny set of math functions/names if needed ---
ALLOWED_FUNCS = {
    "sqrt": math.sqrt,
    "abs": abs,
    "round": round,
}
ALLOWED_NAMES = {"pi": math.pi, "e": math.e}

# --- safe AST evaluator: numbers, + - * / ** %, unary +/-, parentheses, allowed funcs/names ---
class SafeEval(ast.NodeVisitor):
    def visit(self, node):
        if isinstance(node, ast.Expression):
            return self.visit(node.body)
        elif isinstance(node, ast.Num):  # Py <3.8
            return node.n
        elif isinstance(node, ast.Constant):
            if isinstance(node.value, (int, float)):
                return node.value
            raise ValueError("non-numeric constant")
        elif isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.UAdd, ast.USub)):
            v = self.visit(node.operand)
            return +v if isinstance(node.op, ast.UAdd) else -v
        elif isinstance(node, ast.BinOp) and isinstance(node.op, (ast.Add, ast.Sub, ast.Mult, ast.Div, ast.Pow, ast.Mod)):
            l = self.visit(node.left)
            r = self.visit(node.right)
            if isinstance(node.op, ast.Add):  return l + r
            if isinstance(node.op, ast.Sub):  return l - r
            if isinstance(node.op, ast.Mult): return l * r
            if isinstance(node.op, ast.Div):  return l / r
            if isinstance(node.op, ast.Pow):  return l ** r
            if isinstance(node.op, ast.Mod):  return l % r
            raise ValueError("bad binop")
        elif isinstance(node, ast.Call):
            if not isinstance(node.func, ast.Name):
                raise ValueError("bad call")
            fn = node.func.id
            if fn not in ALLOWED_FUNCS:
                raise ValueError("func not allowed")
            args = [self.visit(a) for a in node.args]
            return ALLOWED_FUNCS[fn](*args)
        elif isinstance(node, ast.Name):
            if node.id in ALLOWED_NAMES:
                return ALLOWED_NAMES[node.id]
            raise ValueError("name not allowed")
        elif isinstance(node, ast.Tuple):
            # Disallow tuples
            raise ValueError("tuple not allowed")
        else:
            raise ValueError(f"bad expr: {type(node).__name__}")

def safe_calc(expr: str) -> Optional[float]:
    """
    Evaluate a numeric expression safely (supports + - * / ** %, parentheses, unary +/-).
    Returns float or None on failure.
    """
    try:
        expr_norm = normalize_numeric_text(expr.strip())
        tree = ast.parse(expr_norm, mode="eval")
        return float(SafeEval().visit(tree))
    except Exception:
        return None

# --- verification of [[calc: ...]] -> ... lines ---
_CALC_PATTERN = re.compile(r"\[\[calc:\s*(.*?)\s*\]\]\s*->\s*([^\n]+)")
_SYMBOLIC_TOKENS_RE = re.compile(r"[A-Za-z√π]")  # treat letters/√/π as symbolic

def _is_symbolic(s: str) -> bool:
    return bool(_SYMBOLIC_TOKENS_RE.search(s or ""))

def _safe_num(s: str) -> Optional[float]:
    """
    Parse a RHS number; supports plain ints/floats and simple rationals like '3/2'.
    Returns float or None if not a plain numeric.
    """
    try:
        t = s.replace(",", "").strip()
        # simple rational
        if re.fullmatch(r"[+-]?\d+(?:\.\d+)?\s*/\s*[+-]?\d+(?:\.\d+)?", t):
            num, den = re.split(r"/", t)
            return float(num) / float(den)
        return float(t)
    except Exception:
        return None

def _format_num(x: float) -> str:
    if abs(x - round(x)) < 1e-9:
        return str(int(round(x)))
    s = f"{x:.10f}".rstrip("0").rstrip(".")
    return s

def verify_calc_lines(text: str) -> Tuple[bool, List[str], str]:
    """
    Verify and optionally patch '[[calc: ...]] -> value' lines.
    - Skips verification if expr or value is symbolic (contains letters, √, π, etc.).
    - Safely evaluates numeric-only expressions.
    - If mismatch is found, patches RHS with the computed value.
    Returns (all_ok, messages, possibly_patched_text)
    """
    if not (text or "").strip():
        return True, [], text

    msgs: List[str] = []
    ok = True
    new_lines: List[str] = []

    for line in text.splitlines():
        m = _CALC_PATTERN.search(line)
        if not m:
            new_lines.append(line)
            continue

        expr_raw = m.group(1).strip()
        val_raw  = m.group(2).strip()

        expr_norm = normalize_numeric_text(expr_raw)
        val_norm  = normalize_numeric_text(val_raw)

        # Skip verification if either side is symbolic (letters, sqrt symbol, π, etc.)
        if _is_symbolic(expr_norm) or _is_symbolic(val_norm) or re.search(r"[A-Za-z_]", expr_norm):
            msgs.append(f"skip symbolic: '{expr_raw}' -> '{val_raw}'")
            new_lines.append(line)
            continue

        # Evaluate LHS
        got = safe_calc(expr_norm)
        if got is None:
            ok = False
            msgs.append(f"eval error: {expr_raw}")
            new_lines.append(line)
            continue

        rhs = _safe_num(val_norm)
        if rhs is None:
            msgs.append(f"skip non-numeric rhs: '{val_raw}'")
            new_lines.append(line)
            continue

        if abs(got - rhs) <= 1e-6 * max(1.0, abs(got), abs(rhs)):
            new_lines.append(line)
        else:
            ok = False
            msgs.append(f"mismatch: {expr_raw} -> {val_raw} (calc={_format_num(got)})")
            # Patch RHS in-place
            start, end = m.span(2)
            patched = line[:start] + _format_num(got) + line[end:]
            new_lines.append(patched)

    return ok, msgs, "\n".join(new_lines)

# ----------------------------
# Calc marker repair helpers
# ----------------------------
import re
from typing import Optional, Tuple, List

# Reuse your safe_calc (already imported/defined)
# from your earlier block, we assume safe_calc(expr) -> Optional[float]

_UNIT_RE = re.compile(r"\b(?:kg|g|lbs?|pounds?|mile?s?|km|m|cm|mm|ft|feet|inch(?:es)?|hrs?|hours?|mins?|minutes?|days?|years?|deg(?:rees)?|°|percent)\b", re.I)
_CURRENCY_RE = re.compile(r"[$£€¥]")
_NUMBER_WITH_COMMAS_RE = re.compile(r"(?<!\d)(\d{1,3}(?:,\d{3})+)(?!\d)")
_SQRT_UNICODE_RE = re.compile(r"√\s*(\d+(?:\.\d+)?)")
_FRAC_LATEX_RE = re.compile(r"\\frac\{([^{}]+)\}\{([^{}]+)\}")
_SUP_LATEX_RE = re.compile(r"\^\{([^{}]+)\}")        # ^{2} -> **2
_CDOT_RE = re.compile(r"(\\cdot|\\times|·|×)")
_LATEX_BRACES_RE = re.compile(r"(\\left|\\right)")
_SUP_PLAIN_RE = re.compile(r"(?<=\d)\s*\^\s*(\d+)")  # 3^2 -> 3**2

CALC_LINE_RE = re.compile(r"\[\[calc:\s*(.*?)\s*\]\]\s*->\s*([+-]?\d+(?:\.\d+)?)")
BLANK_OR_WS_RE = re.compile(r"^\s*$")
NUM_ONLY_RE = re.compile(r"^\s*[+-]?\d+(?:\.\d+)?\s*$")

def _normalize_inline_math(s: str) -> str:
    # Remove currency
    s = _CURRENCY_RE.sub("", s)
    # Remove units
    s = _UNIT_RE.sub("", s)
    # Remove thousands commas
    s = _NUMBER_WITH_COMMAS_RE.sub(lambda m: m.group(1).replace(",", ""), s)
    # LaTeX \frac{a}{b} -> (a)/(b)
    s = _FRAC_LATEX_RE.sub(lambda m: f"({m.group(1)})/({m.group(2)})", s)
    # LaTeX/Unicode operators
    s = _CDOT_RE.sub("*", s)
    s = _LATEX_BRACES_RE.sub("", s)
    # Unicode sqrt -> sqrt()
    s = _SQRT_UNICODE_RE.sub(lambda m: f"sqrt({m.group(1)})", s)
    # LaTeX \sqrt{...} -> sqrt(...)
    s = s.replace("\\sqrt", "sqrt")
    s = s.replace("{", "(").replace("}", ")")
    # Exponent: ^{k} or ^k -> **k
    s = _SUP_LATEX_RE.sub(lambda m: f"**({m.group(1)})", s)
    s = _SUP_PLAIN_RE.sub(lambda m: f"**{m.group(1)}", s)
    # Normalize whitespace
    s = re.sub(r"\s+", " ", s)
    return s.strip()

def _balance_parens(expr: str) -> str:
    opens = expr.count("(")
    closes = expr.count(")")
    if opens > closes:
        expr += ")" * (opens - closes)
    elif closes > opens:
        # rare: drop extras from end if they’re dangling
        while closes > opens and expr.endswith(")"):
            expr = expr[:-1]
            closes -= 1
    return expr

def _clean_calc_expr(expr: str) -> str:
    if not expr:
        return expr
    expr = expr.strip()
    # normalize operators and inline math
    expr = _normalize_inline_math(expr)
    # Some stray tokens sometimes sneak in, strip brackets
    expr = expr.replace("]","").replace("[","")
    # common malformed chunks like '6)**2' -> '(6)**2' best-effort
    expr = re.sub(r"(^|\s)(\d+)\)\s*\*\*", r"\1(\2)**", expr)
    # ensure balanced parens
    expr = _balance_parens(expr)
    return expr

def _numbers_in_line(s: str) -> List[float]:
    # extract floats/ints (ignore inside calc markers)
    s_no_calc = re.sub(r"\[\[calc:.*?\]\]", "", s)
    return [float(x) for x in re.findall(r"[+-]?\d+(?:\.\d+)?", s_no_calc)]

def _try_infer_binary_expr(nums: List[float], target: float, tol: float=1e-6) -> Optional[str]:
    # try last-two first, then all pairs; test +, -, *, /
    def ok(a,b,op,res):
        return abs(res - target) <= tol
    pairs = []
    if len(nums) >= 2:
        pairs.append((nums[-2], nums[-1]))
    # fallback: try any pair (favor ones appearing near end)
    L = len(nums)
    for i in range(max(0, L-6), L):  # limit search to last few
        for j in range(i+1, L):
            pair = (nums[i], nums[j])
            if pair not in pairs:
                pairs.append(pair)
    for a,b in pairs:
        # addition
        if ok(a,b,a+b):
            return f"{a}+{b}"
        # subtraction (try both orders)
        if ok(a,b,a-b):
            return f"{a}-{b}"
        if ok(a,b,b-a):
            return f"{b}-{a}"
        # multiplication
        if ok(a,b,a*b):
            return f"{a}*{b}"
        # division
        if abs(b) > tol and ok(a,b,a/b):
            return f"{a}/{b}"
        if abs(a) > tol and ok(a,b,b/a):
            return f"{b}/{a}"
    return None

def repair_calc_markers_in_text(text: str) -> Tuple[str, List[str]]:
    """
    Repairs [[calc: ...]] lines by:
      - Removing symbolic calc markers (contain variables/functions beyond sqrt, pi, e).
      - Filling blank expressions when a plausible (a op b) can be inferred.
      - Replacing numeric-only expressions with inferred binary when possible.
      - Cleaning malformed expressions and re-validating.
      - If expression still doesn't compute and is blank/numeric-only, drop the marker but keep arrow value.
    Returns (patched_text, list_of_notes)
    """
    notes = []
    out_lines = []

    def _repair_one(match: re.Match) -> str:
        raw_expr, val = match.group(1), match.group(2)
        expr = (raw_expr or "").strip()

        # Parse the arrow value; if it isn't numeric, just remove the calc wrapper
        try:
            target = float(val)
        except Exception:
            notes.append("non-numeric arrow; removed calc wrapper")
            return ""  # drop marker entirely

        # Normalize/clean expression (may still be empty)
        cleaned = _clean_calc_expr(expr)

        # --- NEW: drop symbolic calc markers entirely ---
        if _is_symbolic_expr(cleaned):
            notes.append(f"removed symbolic calc marker: {expr}")
            return ""  # delete the whole '[[calc: ...]] -> val' chunk

        # Blank expression: try to infer; else keep arrow only
        if BLANK_OR_WS_RE.match(cleaned or ""):
            nums = _numbers_in_line(match.string)
            inferred = _try_infer_binary_expr(nums, target)
            if inferred:
                fx = _clean_calc_expr(inferred)
                if safe_calc(fx) is not None:
                    notes.append(f"filled blank calc -> {fx} = {val}")
                    return f"[[calc: {fx}]] -> {val}"
            notes.append("dropped blank calc (kept arrow value)")
            return f"-> {val}"

        # Numeric-only expression: try to infer a/b; else keep numeric
        if NUM_ONLY_RE.match(cleaned):
            nums = _numbers_in_line(match.string)
            inferred = _try_infer_binary_expr(nums, target)
            if inferred:
                fx = _clean_calc_expr(inferred)
                if safe_calc(fx) is not None:
                    notes.append(f"replaced numeric-only calc {cleaned} -> {fx}")
                    return f"[[calc: {fx}]] -> {val}"
            # keep numeric-only if it evaluates; else drop wrapper
            if safe_calc(cleaned) is not None:
                return f"[[calc: {cleaned}]] -> {val}"
            notes.append(f"dropped non-evaluable numeric-only calc {cleaned}")
            return f"-> {val}"

        # Has some expression: validate and fix arrow if needed
        got = safe_calc(cleaned)
        if got is None:
            # try to infer from nearby numbers
            nums = _numbers_in_line(match.string)
            inferred = _try_infer_binary_expr(nums, target)
            if inferred:
                fx = _clean_calc_expr(inferred)
                if safe_calc(fx) is not None:
                    notes.append(f"repaired malformed calc '{expr}' -> {fx}")
                    return f"[[calc: {fx}]] -> {val}"
            # leave as-is (it might be nearly numeric but beyond safe_calc)
            notes.append(f"left malformed calc unchanged: {expr}")
            return f"[[calc: {expr}]] -> {val}"
        else:
            if abs(got - target) > 1e-6:
                notes.append(f"fixed arrow: {cleaned} = {got} (was {val})")
                return f"[[calc: {cleaned}]] -> {got}"
            return f"[[calc: {cleaned}]] -> {val}"

    for raw_line in text.splitlines():
        # Remove all calc markers in the line via callback
        new_line = CALC_LINE_RE.sub(_repair_one, raw_line)
        out_lines.append(new_line)

    return "\n".join(out_lines), notes




# ----------------------------
# Prompts (brace-safe replacement)
# ----------------------------
FEW_SHOTS = dedent("""
Example 1 - simple arithmetic (GSM8K style)
Problem:
Mr. Sanchez found out that 40% of his Grade 5  students got a final grade below B. How many of his students got a final grade of B and above if he has 60 students in Grade 5?
Plan:
<<
1. Compute the percentage of students with B and above as 100% − 40%.
2. Multiply the total number of students by that percentage (as a decimal) to get the count.
>>
Target Reasoner (format to mimic):
1. Percentage with B and above: [[calc: 100-40]] -> 60; decimal form: [[calc: 60/100]] -> 0.6
2. Number of students with B and above: [[calc: 60*0.6]] -> 36
#### 36

Example 2 - unit conversion (MATH)
Problem:
Alice wants to buy 3 pounds of veal, scales show kilograms and 1 kg = 2.20 lb. How many kilograms should she buy? (Answer to nearest hundredth.)
Plan:
<<
1. Recognize conversion: 1 kg = 2.20 lb, so multiply pounds by (1 kg / 2.20 lb).
2. Compute kilograms = 3 * (1 / 2.20) and round to nearest hundredth.
>>
Target Reasoner:
1. Convert pounds to kilograms: [[calc: 3*(1/2.20)]] -> 1.363636
2. Round to the nearest hundredth: [[calc: round(1.363636, 2)]] -> 1.36
#### 1.36

Example 3 - geometry (MATH)
Problem:
Circle $B$ has its center at $(-6, 2)$ and a radius of $10$ units. What is the sum of the $y$-coordinates of the two points on circle $B$ that are also on the $y$-axis?
Plan:
<<
1. Recognize points on the y-axis have x = 0 and substitute x = 0 into the circle equation (x+6)^2 + (y-2)^2 = 100.
2. Solve the resulting equation for the two y-values.
3. Sum the two y-values (or use symmetry: they are 2 ± c, so their sum is 4).
>>
Target Reasoner:
1. Set x=0 (y-axis) and substitute: [[calc: (0+6)**2]] -> 36, so 36 + (y-2)^2 = 100.
2. Isolate the square term: [[calc: 100-36]] -> 64, hence (y-2)^2 = 64 and y-2 = ±[[calc: sqrt(64)]] -> 8.
3. Compute y-values and sum: y1 = [[calc: 2+8]] -> 10, y2 = [[calc: 2-8]] -> -6; sum = [[calc: 10 + (-6)]] -> 4
#### 4

Example 4 - algebra (MATH)
Problem:
Solve x^2 - 5x + 6 = 0.
Plan:
<<
1. Attempt to factor the quadratic into binomials.
2. Set each factor equal to zero and list the roots.
>>
Target Reasoner:
1. Find integers whose product is 6 and sum is -5: [[calc: (-2)*(-3)]] -> 6 and [[calc: (-2)+(-3)]] -> -5, so the factorization is (x - 2)(x - 3) = 0.
2. Set factors to zero: x - 2 = 0 → x = 2; x - 3 = 0 → x = 3
#### 2, 3
                   
Example 5 - fractions (MultiArith-like: no rationale, final ans given)
Problem:
Bianca had 45 coloring books. If she gave away 6 of them, but then bought 20 more, how many would she have total?
Plan:
<<
1. Subtract the number given away from the original total.
2. Add the number of books bought to the remainder to get the final total.
>>
Target Reasoner:
1. After giving away books: [[calc: 45-6]] -> 39
2. After buying more books: [[calc: 39+20]] -> 59
#### 59
                   
Example 6 - percent (MultiArith-like)
Problem:
What is 12% of 250?
Plan:
<<
1. Convert percent to decimal (12% → 0.12).
2. Multiply decimal by 250 to compute the amount.
>>
Target Reasoner:
1. Convert percent to decimal: [[calc: 12/100]] -> 0.12
2. Compute the amount: [[calc: 0.12*250]] -> 30.0
#### 30.0
                   
Example 7 - fractions add (GSM8K / MATH)
Problem:
Simplify (2/3) + (1/6).
Plan:
<<
1. Convert fractions to a common denominator.
2. Add numerators and simplify the final fraction.
>>
Target Reasoner:
1. Use common denominator 6. Convert 2/3 to sixths: [[calc: 2*2]] -> 4; 1/6 remains 1/6.
2. Add numerators over 6: [[calc: 4+1]] -> 5; result = 5/6 (already simplest form).
#### 5/6
                   
Example 8 - inverse function
Problem:
If f(x)=2x+1, find f^{-1}(x).
Plan:
<<
1. Write y = 2x + 1.
2. Swap x and y to prepare for inversion.
3. Solve for y to obtain inverse function formula.
>>
Target Reasoner:
1. Let y = 2x + 1.
2. Swap variables to invert: x = 2y + 1.
3. Solve for y: subtract 1 then divide by 2 → y = (x - 1) / 2.
4. Quick check (composition) with x = 5: f(5) = [[calc: 2*5 + 1]] -> 11; f^{-1}(11) = [[calc: (11-1)/2]] -> 5, which returns the original x.
#### (x - 1)/2


""").strip()

REASONER_INSTRUCT_FROM_PLAN = dedent("""
You are <role:reasoner>. Execute the plan to produce a clear, compact solution.

STRICT FORMAT (must follow exactly):
- Number each step: "1.", "2.", ...
- Every step that performs arithmetic MUST include exactly one line of the form [[calc: <python-style expression>]] -> <value>.
- Prefer 3–12 steps.
- Do NOT restate the plan; execute it. Replace each plan step with concrete computations and results.
- Do NOT include code blocks, Python, NumPy, or external tools.
- Only write [[calc: …]] -> value when the expression is purely numeric (digits, + − × ÷ / parentheses, sqrt, pi, e). Never put variables or functions (e.g., x, n, sin, cos) inside [[calc: …]]. For symbolic steps, write the math in prose/equations without a calc marker.
- Finish with exactly one line: #### <final_answer>
- Output MUST appear only between the markers below, nothing before/after.

<<<BEGIN>>>
1. ...
2. ...
...
#### ...
<<<END>>>

{fewshots}

Now solve this:

Problem:
{problem}

Plan:
{plan}

{maybe_final_hint}
Output:
<<<BEGIN>>>
""").strip()


REASONER_REPAIR_PROMPT = dedent("""
You are <role:reasoner>. The draft below fails formatting. Regenerate it STRICTLY in the required format.

RULES:
- Number each step: "1.", "2.", ...
- Each computational step MUST include one [[calc: ...]] -> ... line.
- End with exactly one line: #### <final_answer>
- Output ONLY between <<<BEGIN>>> and <<<END>>>.

Problem:
{problem}

Plan:
{plan}

Draft (bad):
{bad}

Now produce a corrected version:
<<<BEGIN>>>
""").strip()

REASONER_REPAIR_PROMPT_STRICT = dedent("""
You are <role:reasoner>. The prior output failed the required format.
IGNORE the draft and regenerate from scratch using ONLY the Problem and Plan.

STRICT FORMAT:
- Number each step.
- Each computational step MUST include exactly one [[calc: ...]] -> ... line.
- End with exactly one line: #### <final_answer>
- Output ONLY between <<<BEGIN>>> and <<<END>>>.

Problem:
{problem}

Plan:
{plan}

<<<BEGIN>>>
""").strip()

REASONER_REPAIR_PROMPT_SKELETON = dedent("""
You are <role:reasoner>. The draft failed format.
Regenerate STRICTLY as numbered steps with calc markers and a final line.

RULES:
- Output ONLY between <<<BEGIN>>> and <<<END>>>.
- Number each step "1.", "2.", ...
- Each computational step MUST include exactly one [[calc: <python expression>]] -> <value>.
- No code blocks or Python.
- End with exactly one line: #### <final_answer>

Problem:
{problem}

Plan:
{plan}

Fill this skeleton:
<<<BEGIN>>>
1. 
2. 
3. 
#### 
<<<END>>>
""").strip()


def build_plan_text(plan: str) -> str:
    # ensure the plan block is present; extract <<...>> if needed
    if plan and plan.strip().startswith("<<"):
        return plan.strip()
    # wrap if not delimited
    lines = [ln.strip() for ln in (plan or "").splitlines() if ln.strip()]
    if not lines:
        return "<<\n1. Read problem carefully.\n2. Compute needed quantities.\n>>"
    return "<<\n" + "\n".join(lines) + "\n>>"

def craft_prompt(problem: str, plan: str, gold: Optional[str], final_hint: Optional[str], mode: str) -> str:
    few = FEW_SHOTS
    plan_text = build_plan_text(plan or "")
    hint = f"(For verification, the known final answer is: {final_hint})" if final_hint else ""
    if mode == "gold":
        tmpl = REASONER_INSTRUCT_FROM_GOLD
        return tmpl.replace("{fewshots}", few).replace("{problem}", problem).replace("{plan}", plan_text).replace("{gold}", gold or "").replace("{maybe_final_hint}", "")
    else:
        tmpl = REASONER_INSTRUCT_FROM_PLAN
        return tmpl.replace("{fewshots}", few).replace("{problem}", problem).replace("{plan}", plan_text).replace("{maybe_final_hint}", hint)

# ----------------------------
# QC for reasoner outputs
# ----------------------------
def extract_final_line(text: str) -> Optional[str]:
    # look for last #### line
    last = None
    for ln in text.splitlines():
        m = RE_GSM8K_FINAL.match(ln.strip())
        if m:
            last = m.group(1).strip()
    return last

def validate_reasoner(text: str) -> Tuple[bool, List[str]]:
    reasons = []
    t = (text or "").strip()
    if not t:
        return False, ["empty output"]
    lines = [ln for ln in t.splitlines() if ln.strip()]
    if len(lines) < 2:
        reasons.append("too few lines")
    # numbered steps present?
    numbered = [ln for ln in lines if re.match(r"^\d+\.\s+\S", ln)]
    if len(numbered) < 1:
        reasons.append("no numbered steps")
    # has calc markers?
    if "[[calc:" not in t:
        reasons.append("no calc markers")
    # has final ####
    fin = extract_final_line(t)
    if not fin:
        reasons.append("missing final ####")
    return (len(reasons) == 0), reasons

# ----------------------------
# Planner map loader
# ----------------------------
def load_planner_map(planner_root: str, dataset: str, split: str) -> Dict[str, Dict]:
    path = os.path.join(planner_root, dataset, f"{dataset}_{split}_planner.jsonl")
    if not os.path.exists(path):
        print(f"[WARN] no planner file at {path}")
        return {}
    mp = {}
    for row in load_jsonl(path):
        mp[row["id"]] = row
    return mp

# ----------------------------
# Main orchestration
# ----------------------------
def find_split_file(root: str, dataset: str, split: str) -> Optional[str]:
    candidates = [
        os.path.join(root, dataset, f"{split}.jsonl"),
        os.path.join(root, dataset, f"{split}.json"),
        os.path.join(root, dataset, f"{dataset}_{split}.jsonl"),
        os.path.join(root, dataset, f"{dataset}_{split}.json"),
    ]
    for c in candidates:
        if os.path.exists(c):
            return c
    return None

def main():
    ap = argparse.ArgumentParser(description="Create reasoner dataset")
    ap.add_argument("--data_root", type=str, default="data_split")
    ap.add_argument("--planner_root", type=str, default="data_intermediate/planner_data")
    ap.add_argument("--out_dir", type=str, default="data_intermediate/reasoner_data")
    ap.add_argument("--model_backend", choices=["hf","cli"], default="hf")
    ap.add_argument("--model_name_or_path", type=str, default="Qwen/Qwen2.5-Math-7B")
    ap.add_argument("--cli_cmd", type=str, default='ollama generate {model} --prompt "{prompt}"')
    ap.add_argument("--device", type=str, default="cuda:0")
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--max_new_tokens", type=int, default=384)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--datasets", type=str, default="gsm8k,math,multiarith")
    ap.add_argument("--splits", type=str, default="train,val,test")
    ap.add_argument("--gen_mode", choices=["gold_prefer","llm_only"], default="llm_only")
    ap.add_argument("--verify_calcs", action="store_true", help="check [[calc:...]] lines and patch mismatches")
    ap.add_argument("--source_mode", choices=["minimal","raw","none"], default="minimal")
    args = ap.parse_args()

    # LLM client
    if args.model_backend == "hf":
        client = HFTransformersClient(args.model_name_or_path, device=args.device, max_batch_size=args.batch_size)
    else:
        cmd = args.cli_cmd
        if "{model}" in cmd:
            cmd = cmd.replace("{model}", args.model_name_or_path)
        client = CLIClient(cmd)

    datasets = [d.strip() for d in args.datasets.split(",") if d.strip()]
    splits = [s.strip() for s in args.splits.split(",") if s.strip()]

    for ds in datasets:
        planner_maps = {}
        for sp in splits:
            raw_path = find_split_file(args.data_root, ds, sp)
            if not raw_path:
                print(f"[WARN] missing raw {ds}/{sp}")
                continue
            out_dir_ds = os.path.join(args.out_dir, ds)
            os.makedirs(out_dir_ds, exist_ok=True)
            out_path = os.path.join(out_dir_ds, f"{ds}_{sp}_reasoner.jsonl")

            # resume
            seen_ids = set()
            if os.path.exists(out_path):
                for r in load_jsonl(out_path):
                    seen_ids.add(r["id"])

            # load raw & planner
            examples = load_jsonl(raw_path)
            if args.limit is not None:
                examples = examples[:max(0, args.limit)]
                print(f"[INFO] limiting {ds}/{sp} to {len(examples)} examples")

            if sp not in planner_maps:
                planner_maps[sp] = load_planner_map(args.planner_root, ds, sp)
            planner = planner_maps[sp]

            # batching buffers
            prompts: List[str] = []
            metas: List[Dict] = []

            def flush_batch():
                """
                Generate for the current batch of prompts, normalize outputs, run QC, attempt up to
                two repair passes if formatting fails, optionally verify calc lines, and append
                clean JSONL lines to `out_path`.
                """
                nonlocal prompts, metas
                if not prompts:
                    return

                # --- 1) Batched generation with fallback to per-item ---
                try:
                    gens = client.generate_many(prompts, max_tokens=args.max_new_tokens)
                except Exception as e:
                    print(f"[ERROR] batch gen failed: {e}", file=sys.stderr)
                    gens = []
                    for p in prompts:
                        try:
                            gens.append(client.generate(p, max_tokens=args.max_new_tokens))
                        except Exception as e2:
                            print(f"[ERROR] single gen failed: {e2}", file=sys.stderr)
                            gens.append("")

                # --- 2) Append results to file (resume-friendly) ---
                with open(out_path, "a", encoding="utf-8") as f:
                    for gen, meta in zip(gens, metas):
                        raw = (gen or "").strip()

                        # --- Normalize initial output (extract block, cleanup, enforce final, inject calc) ---
                        text, _ = normalize_reasoner_text(raw, fallback_final=meta.get("final_answer"))

                        # --- Ensure at least one calc marker somewhere (e.g., from a paragraph equality) ---
                        text = ensure_global_calc_marker(text)

                        # --- QC pass #1 ---
                        passed, reasons = validate_reasoner(text)

                        # --- Decide if repair is needed (formatting failures only) ---
                        NEEDS_REPAIR = any(k in reasons for k in [
                            "empty output", "no numbered steps", "no calc markers", "missing final ####"
                        ])

                        # --- Repair #1: use the bad draft as a hint ---
                        if (not passed) and NEEDS_REPAIR:
                            repair_prompt = REASONER_REPAIR_PROMPT.format(
                                problem=meta["question"],
                                plan=meta["plan"],
                                bad=text if text else "(empty)"
                            )
                            try:
                                rep = client.generate(repair_prompt, max_tokens=args.max_new_tokens)
                            except Exception as e_repair:
                                reasons.append(f"repair exception: {type(e_repair).__name__}: {e_repair}")
                                rep = ""

                            if rep:
                                text, _ = normalize_reasoner_text(rep, fallback_final=meta.get("final_answer"))
                                text = ensure_global_calc_marker(text)
                                passed, reasons = validate_reasoner(text)
                            else:
                                reasons.append("repair failed: empty reply")

                        # --- Repair #2: ignore draft, regenerate from scratch ---
                        if (not passed) and NEEDS_REPAIR:
                            repair2_prompt = REASONER_REPAIR_PROMPT_STRICT.format(
                                problem=meta["question"],
                                plan=meta["plan"]
                            )
                            try:
                                rep2 = client.generate(repair2_prompt, max_tokens=args.max_new_tokens)
                            except Exception as e_repair2:
                                reasons.append(f"repair2 exception: {type(e_repair2).__name__}: {e_repair2}")
                                rep2 = ""

                            if rep2:
                                text, _ = normalize_reasoner_text(rep2, fallback_final=meta.get("final_answer"))
                                text = ensure_global_calc_marker(text)
                                passed, reasons = validate_reasoner(text)
                            else:
                                reasons.append("repair2 failed: empty reply")

                        # Repair #3 (skeleton) ...
                        if (not passed) and NEEDS_REPAIR:
                            repair3_prompt = REASONER_REPAIR_PROMPT_SKELETON.format(
                                problem=meta["question"], plan=meta["plan"]
                            )
                            try:
                                rep3 = client.generate(repair3_prompt, max_tokens=args.max_new_tokens)
                            except Exception as e_repair3:
                                reasons.append(f"repair3 exception: {type(e_repair3).__name__}: {e_repair3}")
                                rep3 = ""

                            if rep3:
                                text, _ = normalize_reasoner_text(rep3, fallback_final=meta.get("final_answer"))
                                text = ensure_global_calc_marker(text)
                                passed, reasons = validate_reasoner(text)
                            else:
                                reasons.append("repair3 failed: empty reply")

                        # --- Optional calculator verification & patching ---
                        calc_ok, calc_msgs, patched = (True, [], text)
                        if args.verify_calcs:
                            calc_ok, calc_msgs, patched = verify_calc_lines(text)
                            text = patched  # always carry forward patched text

                        # --- Extract final answer (after any patching) ---
                        fin = extract_final_line(text)

                        # --- Build output object ---
                        out_obj = {
                            "id": meta["id"],
                            "dataset": ds,
                            "split": sp,
                            "input_question": meta["question"],
                            "plan": meta["plan"],
                            "reasoner_output": text,
                            "final_answer": fin or (meta.get("final_answer") or ""),
                            "rationale_gold": meta.get("gold_solution") or "",
                            "subject": meta.get("subject") or "",
                            "level": meta.get("level") or "",
                            "qc": {
                                "passed": bool(passed),
                                "reasons": reasons + ([f"[calc] {m}" for m in calc_msgs] if calc_msgs else [])
                            }
                        }
                        if args.source_mode == "minimal":
                            out_obj["source"] = slim_source_meta(meta["raw"])
                        elif args.source_mode == "raw":
                            out_obj["source"] = meta["raw"]

                        f.write(json.dumps(out_obj, ensure_ascii=False) + "\n")

                # --- 3) Clear batch buffers ---
                prompts, metas = [], []



            with tqdm(total=len(examples), desc=f"{ds}/{sp}") as pbar:
                for idx, ex in enumerate(examples, start=1):
                    pbar.update(1)
                    ex_id = ex.get("id") or ex.get("idx") or f"{ds}_{sp}_{idx}"
                    if ex_id in seen_ids:
                        continue

                    question, solution, final_ans, subject, level = detect_fields(ex)
                    question = normalize_ws(question)
                    final_ans = final_ans or ""
                    # find plan
                    plan_row = planner.get(ex_id)
                    plan_text = plan_row["planner_output"] if plan_row else ""
                    if not plan_text:
                        # allow missing plan by inserting a trivial two-step hint
                        plan_text = "<<\n1. Identify required quantities.\n2. Compute and present final answer.\n>>"

                    # choose generation mode for this example
                    use_gold = (args.gen_mode == "gold_prefer" and bool(solution))
                    prompt = craft_prompt(
                        problem=question,
                        plan=plan_text,
                        gold=solution if use_gold else None,
                        final_hint=final_ans if (not use_gold and final_ans) else None,
                        mode="gold" if use_gold else "plan"
                    )
                    prompts.append(prompt)
                    metas.append({
                        "id": ex_id,
                        "question": question,
                        "final_answer": final_ans,
                        "plan": plan_text,
                        "gold_solution": solution or "",
                        "subject": subject,
                        "level": level,
                        "raw": ex
                    })
                    if len(prompts) >= args.batch_size:
                        flush_batch()
                flush_batch()

            print(f"[DONE] Wrote {out_path}")

if __name__ == "__main__":
    main()
