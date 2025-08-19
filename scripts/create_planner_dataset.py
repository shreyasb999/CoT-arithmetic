#!/usr/bin/env python3
"""
create_planner_dataset.py

Production-grade script to generate 'planner' role dataset for GSM8K, MATH and MultiArith.

Features:
- Reads JSONL train/val/test for each dataset directory (auto-detects fields).
- Uses a local LLM backend (HuggingFace transformers + bitsandbytes or a subprocess to llama.cpp/ollama).
- Unified planner-extractor prompt for MATH (with solution) and MultiArith (no solution).
- Deterministic generation settings (temperature=0.0) for consistent dataset.
- Automatic QC: output contains <<...>>, 1-10 steps, non-empty, step length limits, simple semantic checks.
- Resumeable and batched processing.
- Exports JSONL lines with schema:
  {
    "id": "...",
    "dataset":"gsm8k"|"math"|"multiarith",
    "split":"train"|"val"|"test",
    "input_question": "...",
    "rationale": "<original solution if exists or ''>",
    "planner_output": "<<\n1. ...\n2. ...\n>>",
    "meta": { ... },
    "qc": { "passed": true/false, "reasons": [...] }
  }

Requirements (suggested):
- Python 3.10+
- pip install -U transformers accelerate bitsandbytes sentencepiece regex tqdm
- OR install llama-cpp-python if using local gguf/llama.cpp backend
- Ensure the chosen local model is downloaded (GGUF or HF model + bnb quant files)

Model backends supported (configurable):
- hf: HuggingFace transformers with device_map='auto' + bitsandbytes 4/8-bit quant (recommended if using Qwen or Llama weights via HF).
- llama_cpp: llama-cpp-python (for gguf / ggml) -> CPU/GPU support depending on build.
- cli: call external CLI like 'ollama' or 'vllm' or 'local_server' via subprocess (user must run server).

Usage example:
python create_planner_dataset.py --data_root ./data --out_dir ./output/planner_data \
    --model_backend hf --model_name_or_path qwen-2.5-math-7b --device cuda:0 --batch_size 8
"""

import argparse
import json
import os
import re
import sys
import time
from typing import List, Optional, Dict, Tuple
from pathlib import Path
from tqdm import tqdm

# --- LLM client section (backend-agnostic interface) --------------------------
# The LLMClient class provides two built-in implementations:
#  - HFTransformersClient: uses HuggingFace transformers (with bitsandbytes)
#  - CLIClient: calls a local CLI/server (ollama/vllm) via subprocess (user-managed)
#
# Pick the backend via --model_backend {hf,cli,llama_cpp}
# If you want llama-cpp-python integration, uncomment/implement the LlamaCppClient class.

class LLMClient:
    """Abstract interface - implement generate(prompt:str)->str"""

    def generate(self, prompt: str, max_tokens: int = 256, temperature: float = 0.0) -> str:
        raise NotImplementedError("Implement in subclass")


# -------------------------
# HF Transformers backend
# -------------------------
try:
    from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline, GenerationConfig
    import torch
    HF_AVAILABLE = True
except Exception as e:
    HF_AVAILABLE = False
    # We'll still allow CLI fallback

# --- replace your HFTransformersClient with this ---

class HFTransformersClient(LLMClient):
    """
    HuggingFace local model client using transformers.
    - Batches prompts to the GPU
    - Left-pads for decoder-only models
    - No temperature arg (greedy, deterministic)
    """

    def __init__(self, model_name_or_path: str, device: str = "cuda:0", max_batch_size: int = 8):
        if not HF_AVAILABLE:
            raise RuntimeError("transformers not installed. Install transformers, accelerate, bitsandbytes.")
        self.model_name_or_path = model_name_or_path
        self.device = device
        self.max_batch_size = max_batch_size

        print(f"[HFClient] Loading tokenizer and model from {model_name_or_path} on {device} ...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, use_fast=False)

        # Ensure PAD + left padding for decoder-only
        if self.tokenizer.pad_token_id is None:
            # prefer eos as pad; fall back to unk or define literal "[PAD]"
            if self.tokenizer.eos_token is not None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            else:
                self.tokenizer.add_special_tokens({"pad_token": "[PAD]"})
        self.tokenizer.padding_side = "left"

        # Load model (tweak for 4-bit if you like: load_in_4bit=True)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name_or_path,
            device_map="auto",
            torch_dtype=torch.float16,
            trust_remote_code=True,
            low_cpu_mem_usage=True,
        )
        # align model pad id with tokenizer
        if getattr(self.model.config, "pad_token_id", None) is None:
            self.model.config.pad_token_id = self.tokenizer.pad_token_id

        self.pipe = pipeline(
            task="text-generation",
            model=self.model,
            tokenizer=self.tokenizer,
            device_map="auto"
        )
        print("Device set to use", device)
        print("[HFClient] Model loaded.")

    def _unwrap_outputs(self, outputs):
        """
        pipeline(list_of_prompts) returns a list with one element per prompt.
        Each element is a list of dicts (one dict per returned sequence).
        We take the first sequence per prompt.
        """
        gens = []
        for item in outputs:
            if isinstance(item, list):
                # first sequence for that prompt
                if len(item) > 0 and isinstance(item[0], dict):
                    gens.append(item[0].get("generated_text", ""))
                else:
                    gens.append("")
            elif isinstance(item, dict):
                gens.append(item.get("generated_text", ""))
            else:
                gens.append(str(item) if item is not None else "")
        return gens

    def generate_many(self, prompts, max_tokens: int = 256, temperature: float = 0.0):
        """
        Batched generation. Returns list[str] aligned with prompts.
        We do **greedy** decoding: no temperature arg (avoids pipeline warnings).
        """
        gen_kwargs = {
            "max_new_tokens": max_tokens,
            "do_sample": False,                           # greedy
            "top_p": 1.0,
            "eos_token_id": self.tokenizer.eos_token_id,
            "pad_token_id": self.tokenizer.pad_token_id,
            "return_full_text": False,
            "num_return_sequences": 1,
        }
        outputs = self.pipe(prompts, batch_size=self.max_batch_size, **gen_kwargs)
        return self._unwrap_outputs(outputs)

    def generate(self, prompt: str, max_tokens: int = 256, temperature: float = 0.0) -> str:
        return self.generate_many([prompt], max_tokens=max_tokens, temperature=temperature)[0]



# -------------------------
# CLI backend (ollama / vllm / local server)
# -------------------------
import subprocess, shlex, textwrap
class CLIClient(LLMClient):
    """
    Calls a local CLI endpoint. Example uses `ollama generate <model> --prompt "<prompt>"`.
    User must have ollama/vllm/other local server available in PATH.
    This is a simple wrapper around subprocess - customize as needed.
    """
    def __init__(self, cli_cmd: str):
        """
        cli_cmd: base CLI command formatted with {model} and {prompt}, e.g.
        "ollama generate {model} --prompt \"{prompt}\" --json"
        or "curl -s -X POST http://localhost:8000/generate -d '{{...}}'"
        """
        self.cli_cmd = cli_cmd

    def generate(self, prompt: str, max_tokens: int = 256, temperature: float = 0.0) -> str:
        # Ensure prompt is safely quoted
        safe_prompt = prompt.replace('"', '\\"')
        cmd = self.cli_cmd.format(prompt=safe_prompt)
        # Shell execution
        proc = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        out, err = proc.communicate(timeout=120)
        if proc.returncode != 0:
            print(f"[CLIClient] CLI returned non-zero exit {proc.returncode}. stderr: {err}", file=sys.stderr)
            raise RuntimeError("CLI client error")
        # Heuristics: CLI may return JSON or raw text - try parsing JSON first
        # But for simplicity, return full stdout
        return out.strip()


# -------------------------
# Utilities & parsing
# -------------------------

# Basic regex to find << ... >> blocks
RE_TAGS = re.compile(r"<<(.+?)>>", re.DOTALL)

def extract_between_double_chevrons(text: str) -> List[str]:
    """Return all segments inside << >> in order."""
    return RE_TAGS.findall(text or "")

def remove_between_double_chevrons(text: str) -> str:
    """Return text with <<...>> sections removed (replaced by empty string)."""
    return RE_TAGS.sub("", text or "")

def normalize_whitespace(s: str) -> str:
    return re.sub(r"\s+", " ", s).strip()

def load_jsonl(path: str) -> List[Dict]:
    items = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip(): continue
            items.append(json.loads(line))
    return items

def write_jsonl(items: List[Dict], path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for it in items:
            f.write(json.dumps(it, ensure_ascii=False) + "\n")

# Handy auto-detect field function for datasets (tries common keys)
def detect_fields(example: Dict) -> Tuple[str, Optional[str], Optional[str], Optional[str], Optional[str]]:
    """
    Returns:
      (question_text, solution_text, final_answer, subject, level)
    Tries top-level, orig, and meta.orig variants. Also parses GSM8K '#### final' tail.
    """
    q = None
    sol = None
    final = None
    subject = None
    level = None

    # QUESTIONS (in order of preference)
    q = (
        example.get("question")
        or example.get("problem")
        or get_in(example, ("orig", "question"))
        or get_in(example, ("orig", "problem"))
        or get_in(example, ("meta", "orig", "problem"))
        or get_in(example, ("meta", "orig", "question"))
    )

    # SOLUTIONS / RATIONALES
    sol = (
        example.get("solution")
        or get_in(example, ("orig", "solution"))
        or get_in(example, ("meta", "solution"))
        or get_in(example, ("meta", "orig", "solution"))
    )

    # FINAL ANSWERS (direct)
    final = (
        example.get("final_ans")
        or example.get("final_answer")
        or get_in(example, ("orig", "final_ans"))
        or get_in(example, ("orig", "final_answer"))
        or get_in(example, ("meta", "final_ans"))
        or get_in(example, ("meta", "final_answer"))
        or get_in(example, ("meta", "orig", "final_ans"))
        or get_in(example, ("meta", "orig", "final_answer"))
    )

    # ANSWER fields can be either rationale or final-only depending on dataset
    ans_candidates = [
        example.get("answer"),
        get_in(example, ("orig", "answer")),
        get_in(example, ("meta", "answer")),
        get_in(example, ("meta", "orig", "answer")),
    ]
    for ans in ans_candidates:
        if not ans or not isinstance(ans, str):
            continue
        # If step-by-step (GSM8K style with lines/<< >>), treat as rationale
        if looks_step_by_step(ans) and sol is None:
            sol = ans
            # And still try to parse final from "#### <final>"
            if final is None:
                parsed = extract_gsm8k_final(ans)
                if parsed:
                    final = parsed
        # If short, single-line style (MATH final answer), keep as final if none yet
        elif final is None:
            # avoid swallowing long LaTeX as "final"
            if len(ans.splitlines()) == 1 and len(ans) <= 64:
                final = ans

    # SUBJECT / LEVEL (MATH)
    subject = (
        example.get("subject")
        or get_in(example, ("orig", "subject"))
        or get_in(example, ("meta", "subject"))
        or get_in(example, ("meta", "orig", "subject"))
    )
    level = (
        example.get("level")
        or get_in(example, ("orig", "level"))
        or get_in(example, ("meta", "level"))
        or get_in(example, ("meta", "orig", "level"))
    )

    return q or "", sol, final, subject, level

def get_in(d, path):
    cur = d
    for k in path:
        if not isinstance(cur, dict) or k not in cur:
            return None
        cur = cur[k]
    return cur

RE_GSM8K_FINAL = re.compile(r"####\s*([^\n#]+)")

def extract_gsm8k_final(s: str) -> Optional[str]:
    if not s or not isinstance(s, str): 
        return None
    m = RE_GSM8K_FINAL.search(s)
    return m.group(1).strip() if m else None

def looks_step_by_step(s: str) -> bool:
    if not s or not isinstance(s, str): 
        return False
    # GSM8K often has << >> and multiple lines
    return ("<<" in s) or ("\n" in s and len(s.splitlines()) >= 2)

def slim_source_meta(ex: dict) -> dict:
    """
    Extract minimal, non-redundant provenance.
    Avoids nesting meta->orig->meta recursion.
    """
    m = ex.get("meta", {}) if isinstance(ex.get("meta"), dict) else {}
    orig = m.get("orig", {}) if isinstance(m.get("orig"), dict) else {}

    return {
        "id": ex.get("id") or ex.get("idx") or orig.get("unique_id") or "",
        "hf_name": m.get("hf_name", ""),
        "hf_config": m.get("hf_config", ""),
        "original_index": m.get("original_index", None),
        "unique_id": orig.get("unique_id", ""),
    }


# -------------------------
# Unified Planner prompt (production-grade)
# -------------------------
# We'll export the unified prompt below to be used by the LLM client.
from textwrap import dedent

UNIFIED_PLANNER_PROMPT_PREFIX = dedent("""
You are a Planner Extractor. For each example you will be given:
1) A math problem (the full text).
2) Optionally, a human step-by-step solution (rationale) for this problem.
3) Optionally, a final numeric answer.

Your job is to produce a short ordered plan suitable for a downstream solver. The plan MUST:
- Be concise: prefer 3–6 steps; at most 10 steps.
- Contain short imperative action lines (one line per step) describing WHAT to do, not the numeric calculation.
- Not compute final numeric answers or include long calculations.
- Preserve the logical order and key transforms from the rationale when present.
- If the rationale is absent (e.g., MultiArith), generate a plausible plan that leads to the final answer and ensure it is consistent with the given final answer.
- If unit conversions are required, include an explicit conversion step.
- If the solution requires a key insight (e.g., similar triangles, Pythagorean theorem), include it as a short action.
- Use at most one sentence per step (imperative tone, e.g., "Isolate x", "Compute area using formula", "Express as a single fraction").

OUTPUT RULES (be strict):
- Output only the plan between EXACT delimiters `<<` and `>>` with numbered steps, like:
<<
1. ...
2. ...
>>
- Do NOT output any commentary, metadata, or explanation outside the delimiters.
- Use temperature = 0.0 (deterministic).

Below are examples showing how to convert problems and rationales to planner outputs.
""").lstrip()

# Few-shot examples (production grade) - mix of MATH (has rationale) and MultiArith-like (has final ans only)
# Keep these examples concise and representative (8 examples)
FEW_SHOT_EXAMPLES = [
    # Example 1 - simple arithmetic (GSM8K style)
    {
        "problem":"Mr. Sanchez found out that 40% of his Grade 5  students got a final grade below B. How many of his students got a final grade of B and above if he has 60 students in Grade 5?",
        "solution": "Since 40% of his students got below B, 100% - 40% = 60% of Mr. Sanchez's students got B and above.\nThus, 60 x 60/100 = <<60*60/100=36>>36 students got B and above in their final grade.\n#### 36",
        "planner": "<<\n1. Compute the percentage of students with B and above as 100% − 40%.\n2. Multiply the total number of students by that percentage (as a decimal) to get the count.\n>>"
    },
    # Example 2 - unit conversion (MATH)
    {
        "problem":"Alice wants to buy 3 pounds of veal, scales show kilograms and 1 kg = 2.20 lb. How many kilograms should she buy? (Answer to nearest hundredth.)",
        "solution":"Since Alice wants to buy 3 pounds of veal, we multiply the quantity of 3 pounds by the conversion factor \\frac{1 \\text{kg}}{2.20 \\text{lb}} to obtain 3 lb * (1 kg / 2.20 lb) ≈ 1.36 kg.",
        "planner":"<<\n1. Recognize conversion: 1 kg = 2.20 lb, so multiply pounds by (1 kg / 2.20 lb).\n2. Compute kilograms = 3 * (1 / 2.20) and round to nearest hundredth.\n>>"
    },
    # Example 3 - geometry (MATH)
    {
        "problem":"Circle $B$ has its center at $(-6, 2)$ and a radius of $10$ units. What is the sum of the $y$-coordinates of the two points on circle $B$ that are also on the $y$-axis?",
        "solution":"Note that if the $y$-coordinate of one of the two points is $2+c$, then the $y$-coordinate of the other point must be $2-c$ because the two points must be equidistant from the line $y=2$. Therefore, the sum of the $y$-coordinates of the two points on circle $B$ that are also on the $y$-axis is $\\boxed{4}$.",
        "planner":"<<\n1. Recognize points on the y-axis have x = 0 and substitute x = 0 into the circle equation (x+6)^2 + (y-2)^2 = 100.\n2. USolve the resulting equation for the two y-values.\n3. Sum the two y-values (or use symmetry: they are 2 ± c, so their sum is 4).\n>>"
    },
    # Example 4 - algebra (MATH)
    {
        "problem":"Solve x^2 - 5x + 6 = 0.",
        "solution":"Factor (x-2)(x-3)=0 so x=2 or x=3.",
        "planner":"<<\n1. Attempt to factor the quadratic into binomials.\n2. Set each factor equal to zero and list the roots.\n>>"
    },
    # Example 5 - fractions (MultiArith-like: no rationale, final ans given)
    {
        "problem":"Bianca had 45 coloring books. If she gave away 6 of them, but then bought 20 more, how many would she have total?",
        "solution": None,
        "final_answer":"59",
        "planner":"<<\n1. Subtract the number given away from the original total.\n2. Add the number of books bought to the remainder to get the final total.\n>>"
    },
    # Example 6 - percent (MultiArith-like)
    {
        "problem":"What is 12% of 250?",
        "solution": None,
        "final_answer":"30.0",
        "planner":"<<\n1. Convert percent to decimal (12% → 0.12).\n2. Multiply decimal by 250 to compute the amount.\n>>"
    },
    # Example 7 - fractions add (GSM8K / MATH)
    {
        "problem":"Simplify (2/3) + (1/6).",
        "solution":"Convert to common denominator 6: 4/6 + 1/6 = 5/6.",
        "planner":"<<\n1. Convert fractions to a common denominator.\n2. Add numerators and simplify the final fraction.\n>>"
    },
    # Example 8 - inverse function
    {
        "problem":"If f(x)=2x+1, find f^{-1}(x).",
        "solution":"Set y=2x+1; swap x and y; solve y=(x-1)/2.",
        "planner":"<<\n1. Write y = 2x + 1.\n2. Swap x and y to prepare for inversion.\n3. Solve for y to obtain inverse function formula.\n>>"
    }
]

# Build the few-shot content block used by the LLM
def build_few_shot_block():
    texts = []
    for ex in FEW_SHOT_EXAMPLES:
        p = f"Problem: {ex['problem']}\n"
        if ex.get("solution") is not None:
            p += "Solution: " + ex["solution"] + "\n"
        elif ex.get("final_answer") is not None:
            p += "Final answer: " + ex["final_answer"] + "\n"
        p += "Planner output:\n" + ex["planner"] + "\n\n"
        texts.append(p)
    return "\n".join(texts)

FEW_SHOT_BLOCK = build_few_shot_block()
UNIFIED_PLANNER_PROMPT = UNIFIED_PLANNER_PROMPT_PREFIX + FEW_SHOT_BLOCK + dedent("""
Now produce a planner for the following input. Output ONLY the planner between << and >>, nothing else.

INPUT:
Problem: {problem}
{maybe_solution_or_answer}
Output:
""").lstrip()

# -------------------------
# Planner generation & QC functions
# -------------------------
def craft_prompt(problem: str, solution: Optional[str], final_answer: Optional[str]) -> str:
    """
    Build the unified planner prompt by doing safe literal substitutions.
    Avoids using str.format() on the whole prompt (which can break when
    the prompt contains unescaped curly braces from LaTeX).
    """
    # Prepare the optional block
    maybe = ""
    if solution:
        maybe = "Solution: " + solution
    elif final_answer:
        maybe = "Final answer: " + str(final_answer)

    # Use literal replacement rather than .format() to avoid ValueError on braces.
    # UNIFIED_PLANNER_PROMPT contains literal substrings "{problem}" and
    # "{maybe_solution_or_answer}" which we replace safely.
    # We also guard against None.
    ptext = problem or ""
    mtext = maybe or ""

    # IMPORTANT: Do NOT run .format() on UNIFIED_PLANNER_PROMPT (it contains raw braces).
    # Use replace on the exact placeholder tokens instead.
    prompt = UNIFIED_PLANNER_PROMPT.replace("{problem}", ptext).replace("{maybe_solution_or_answer}", mtext)

    return prompt


def validate_planner(planner_text: str) -> Tuple[bool, List[str]]:
    """
    QC:
      - Errors (block PASS): missing delimiters, empty plan, zero/too many steps,
        malformed numbering, extremely long steps.
      - Warnings (do NOT block PASS): 'may lack action verb' heuristic.

    Returns (passed, reasons) with reasons including both errors and warnings.
    """
    errors = []
    warnings = []

    text = (planner_text or "").strip()
    if not text.startswith("<<") or not text.endswith(">>"):
        errors.append("Missing or incorrect delimiters << >>.")
        return False, errors

    inner = text[2:-2].strip()
    if not inner:
        errors.append("Empty plan inside delimiters.")
        return False, errors

    lines = [l.strip() for l in inner.splitlines() if l.strip()]
    if len(lines) < 1:
        errors.append("No steps detected inside plan.")
    if len(lines) > 10:
        errors.append(f"Too many steps: {len(lines)} (>10).")

    # Broad verb list + UK/US spellings; used as a heuristic only
    VERB_RE = re.compile(
        r"\b("
        r"compute|calculate|determine|find|derive|evaluate|estimate|bound|"
        r"solve|isolate|equate|rearrange|express|rewrite|simplify|factor|factorize|expand|"
        r"convert|normalize|standardize|rationali[sz]e|approximate|round|"
        r"apply|use|compare|note|observe|identify|state|define|let|assume|deduce|infer|"
        r"substitute|differentiate|integrate|complete|(')?complete the square|"
        r"draw|sketch|plot|project|parameteri[sz]e|prove|show|verify|check"
        r")\b",
        flags=re.I
    )

    num_steps = 0
    for i, ln in enumerate(lines, start=1):
        m = re.match(r"^\d+\.\s+(.+)$", ln)
        if not m:
            errors.append(f"Step formatting not recognized on line {i}: {ln[:80]}")
            continue
        num_steps += 1
        step_text = m.group(1).strip()

        # Hard length guard (true error)
        if len(step_text) > 200:
            errors.append(f"Step {i} too long ({len(step_text)} chars).")

        # Heuristic only: flag as WARNING if no obvious action verb
        if not VERB_RE.search(step_text):
            warnings.append(f"Step {i} may lack an action verb: '{step_text[:60]}...'")

    if num_steps == 0:
        errors.append("No numbered steps parsed.")

    passed = (len(errors) == 0)

    # We still return warnings in reasons so you can see them, but they don't block PASS.
    reasons = errors + [f"[warn] {w}" for w in warnings]
    return passed, reasons


# -------------------------
# Main orchestration
# -------------------------
def process_split_for_dataset(dataset_name: str, split: str, input_path: str, out_path: str,
                              client: LLMClient, batch_size: int = 8, resume: bool = True, limit: Optional[int] = None):
    print(f"[PROCESS] dataset={dataset_name} split={split} input={input_path} -> out_dir={out_path}")
    examples = load_jsonl(input_path)

    # Apply --limit if provided
    if limit is not None:
        try:
            if limit < 0:
                limit = None
            else:
                examples = examples[:limit]
                print(f"[INFO] Limiting to first {len(examples)} examples for {dataset_name}/{split}")
        except Exception:
            pass

    out_file = os.path.join(out_path, f"{dataset_name}_{split}_planner.jsonl")
    os.makedirs(out_path, exist_ok=True)

    # Resume support: collect seen ids
    seen_ids = set()
    if resume and os.path.exists(out_file):
        for ex in load_jsonl(out_file):
            seen_ids.add(ex.get("id") or json.dumps(ex.get("input_question",""))[:64])

    pending_prompts = []
    pending_meta = []

    def flush_batch():
        nonlocal pending_prompts, pending_meta
        if not pending_prompts:
            return
        # Try batched first
        try:
            generations = client.generate_many(pending_prompts, max_tokens=350, temperature=0.0)
        except Exception as e:
            print(f"[ERROR] Batch generation failed: {e}", file=sys.stderr)
            # Fallback: one-by-one to salvage progress
            generations = []
            for p in pending_prompts:
                try:
                    generations.append(client.generate(p, max_tokens=256, temperature=0.0))
                except Exception as e2:
                    print(f"[ERROR] Single generation failed: {e2}", file=sys.stderr)
                    generations.append("")

        os.makedirs(out_path, exist_ok=True)
        with open(out_file, "a", encoding="utf-8") as f:
            for gen_text, meta in zip(generations, pending_meta):
                # Keep first <<...>> block if present; else raw (may fail QC if empty)
                m = re.search(r"<<.*?>>", gen_text or "", flags=re.S)
                planner_text = m.group(0) if m else (gen_text or "").strip()

                passed, reasons = validate_planner(planner_text)

                # Build minimal, non-nested provenance
                minimal_src = slim_source_meta(meta["raw"])

                out_obj = {
                    "id": meta["ex_id"],
                    "dataset": dataset_name,
                    "split": split,
                    "input_question": meta["question"],
                    "rationale": meta["solution"] or "",
                    "final_answer": meta["final_answer"] or "",
                    "subject": meta.get("subject") or "",
                    "level": meta.get("level") or "",
                    "planner_output": planner_text,
                    "qc": {"passed": passed, "reasons": reasons},
                    "source": minimal_src          
                }
                f.write(json.dumps(out_obj, ensure_ascii=False) + "\n")

        pending_prompts, pending_meta = [], []


    with tqdm(total=len(examples), desc=f"{dataset_name}/{split}") as pbar:
        for idx, ex in enumerate(examples, start=1):
            pbar.update(1)
            ex_id = ex.get("id") or ex.get("idx") or f"{dataset_name}_{split}_{idx}"
            if ex_id in seen_ids:
                continue

            question, solution, final_answer, subject, level = detect_fields(ex)
            question = normalize_whitespace(question or "")
            solution = solution if solution and isinstance(solution, str) else None
            final_answer = final_answer if final_answer and isinstance(final_answer, str) else None

            prompt = craft_prompt(question, solution, final_answer)
            pending_prompts.append(prompt)
            pending_meta.append({
                "ex_id": ex_id,
                "question": question,
                "solution": solution,
                "final_answer": final_answer,
                "subject": subject,
                "level": level,
                "raw": ex
            })

            if len(pending_prompts) >= batch_size:
                flush_batch()

        # Flush any remainder
        flush_batch()

    print(f"[DONE] Wrote planner dataset to {out_file}")

# -------------------------
# Command line interface
# -------------------------
def find_split_file(root: str, dataset: str, split: str) -> Optional[str]:
    # look for candidate file names
    candidates = [
        os.path.join(root, dataset, f"{split}.jsonl"),
        os.path.join(root, dataset, f"{split}.json"),
        os.path.join(root, dataset, f"{dataset}_{split}.jsonl"),
        os.path.join(root, dataset, f"{dataset}_{split}.json"),
        os.path.join(root, f"{dataset}_{split}.jsonl"),
        os.path.join(root, f"{dataset}_{split}.json"),
        os.path.join(root, dataset, "data.jsonl"),
        os.path.join(root, dataset, "train.jsonl") if split == "train" else None
    ]
    for c in candidates:
        if c and os.path.exists(c):
            return c
    return None

def main():
    parser = argparse.ArgumentParser(description="Create planner dataset for GSM8K, MATH, MultiArith")
    parser.add_argument("--data_root", type=str, default="data_split", help="root directory with dataset subfolders")
    parser.add_argument("--out_dir", type=str, default="data_intermediate/planner_data", help="output directory")
    parser.add_argument("--model_backend", type=str, default="hf", choices=["hf","cli"], help="LLM backend to use")
    parser.add_argument("--model_name_or_path", type=str, default="qwen-2.5-math-7b", help="model name or CLI format")
    parser.add_argument("--cli_cmd", type=str, default='ollama generate {model} --prompt "{prompt}" --json', help="CLI template if model_backend=cli")
    parser.add_argument("--device", type=str, default="cuda:0", help="device for HF backend")
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--limit", type=int, default=None, help="Process at most N examples per split (useful for quick tests)")
    parser.add_argument("--only_dataset", choices=["gsm8k","math","multiarith"], default=None)
    parser.add_argument("--only_split", choices=["train","val","test"], default=None)
    parser.add_argument("--only_file", type=str, default=None)
    parser.add_argument("--source_mode",choices=["minimal", "raw", "none"],default="minimal",help="How much source provenance to store in output JSONL.")
    args = parser.parse_args()

    # instantiate client
    if args.model_backend == "hf":
        client = HFTransformersClient(args.model_name_or_path, device=args.device, max_batch_size=args.batch_size)
    else:
        # create CLI client; replace {model} with model name if needed.
        cli_cmd = args.cli_cmd
        if "{model}" in cli_cmd:
            cli_cmd = cli_cmd.format(model=args.model_name_or_path, prompt="{prompt}")
        client = CLIClient(cli_cmd)

    # if a single file is specified, just process that and return
    if args.only_file:
        ds = args.only_dataset or "custom"
        sp = args.only_split or "custom"
        out_path = os.path.join(args.out_dir, ds)
        process_split_for_dataset(ds, sp, args.only_file, out_path, client,
                                batch_size=args.batch_size, resume=True)
        return

    # otherwise, run the normal loop but respect filters if provided
    datasets = [args.only_dataset] if args.only_dataset else ["gsm8k","math","multiarith"]
    splits = [args.only_split] if args.only_split else ["train","val","test"]

    for ds in datasets:
        for sp in splits:
            file_path = find_split_file(args.data_root, ds, sp)
            if not file_path:
                print(f"[WARN] Skipping missing file for {ds}/{sp} (no file found under {args.data_root}/{ds})")
                continue
            out_path = os.path.join(args.out_dir, ds)
            process_split_for_dataset(ds, sp, file_path, out_path, client,
                                    batch_size=args.batch_size, resume=True, limit=args.limit)

if __name__ == "__main__":
    main()
