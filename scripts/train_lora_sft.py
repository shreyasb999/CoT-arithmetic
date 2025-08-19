#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Supervised fine-tuning with LoRA/QLoRA on unified multi-role SFT JSONL.

Key features:
- No-leak splits: --train_splits and --eval_splits
- Role rebalancing: --role_weights "verifier:3,normalizer:3"
- QLoRA 4-bit or FP16 LoRA
- Optional FlashAttention v2 and gradient checkpointing
- Robust schema ingestion (messages, prompt/response, input_text/target_text)
- Optional packing mode (constant-length concatenation)
"""

import os
import sys
import json
import math
import argparse
import random
from typing import Dict, List, Any, Tuple, Optional, Union

import torch
from datasets import Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    DataCollatorForLanguageModeling,
    default_data_collator,
)
from transformers.trainer_utils import set_seed
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from packaging import version
from dataclasses import dataclass
from transformers import PreTrainedTokenizerBase

# Optional 4-bit
try:
    from bitsandbytes import __version__ as bnb_version  # noqa
    from transformers import BitsAndBytesConfig
    HAS_BNB = True
except Exception:
    HAS_BNB = False


# -----------------------
# Utilities
# -----------------------
def load_jsonl(path: str) -> List[Dict[str, Any]]:
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except Exception as e:
                print(f"[WARN] bad jsonl line: {e}", file=sys.stderr)
    return rows


def filter_rows(rows: List[Dict[str, Any]],
                include_datasets: str,
                include_splits: str,
                include_roles: str) -> List[Dict[str, Any]]:
    out = rows
    if include_datasets:
        allow = {d.strip() for d in include_datasets.split(",") if d.strip()}
        out = [r for r in out if r.get("dataset") in allow]
    if include_splits:
        allow = {s.strip() for s in include_splits.split(",") if s.strip()}
        out = [r for r in out if r.get("split") in allow]
    if include_roles:
        allow = {s.strip() for s in include_roles.split(",") if s.strip()}
        out = [r for r in out if r.get("role") in allow]
    return out


def parse_role_weights(s: str) -> Dict[str, int]:
    if not s:
        return {}
    out: Dict[str, int] = {}
    for tok in s.split(","):
        tok = tok.strip()
        if not tok:
            continue
        k, v = tok.split(":")
        out[k.strip()] = int(v.strip())
    return out


def apply_role_weights(rows: List[Dict[str, Any]], weights: Dict[str, int]) -> List[Dict[str, Any]]:
    if not weights:
        return rows
    boosted: List[Dict[str, Any]] = []
    for r in rows:
        w = weights.get(r.get("role", ""), 1)
        if w <= 1:
            boosted.append(r)
        else:
            boosted.extend([r] * w)
    return boosted


# -----------------------
# Text building
# -----------------------
SYSTEM_FALLBACK = "You are a helpful math assistant that follows role-specific instructions and formats."

def build_from_messages(row: Dict[str, Any]) -> Tuple[str, str]:
    """
    Expect row["messages"] = [{"role": "system/user/assistant", "content": "..."}, ...]
    Target is the **last assistant** message; prompt is everything before it.
    """
    msgs = row.get("messages", [])
    if not msgs:
        raise ValueError("no messages")

    sys_txt = ""
    prompt_parts = []
    target_text = None

    for i, m in enumerate(msgs):
        role = (m.get("role") or "").lower()
        content = m.get("content", "")
        if role == "system":
            sys_txt = content
        elif role == "assistant":
            # define last assistant as target if it's the final message
            if i == len(msgs) - 1:
                target_text = content
            else:
                # assistant in mid turns: keep as part of context
                prompt_parts.append(f"<assistant>\n{content}")
        else:
            # treat anything else as user content
            prompt_parts.append(f"<user>\n{content}")

    if not target_text:
        # fallback: assume last message is target anyway
        last = msgs[-1]
        target_text = last.get("content", "")

    # Compose chat-like prompt
    system = sys_txt if sys_txt else SYSTEM_FALLBACK
    prompt_text = f"<system>\n{system}\n\n" + "\n\n".join(prompt_parts).strip()
    return prompt_text.strip(), (target_text or "").strip()


def build_from_prompt_response(row: Dict[str, Any]) -> Tuple[str, str]:
    prompt = row.get("prompt", "") or row.get("input_text", "") or row.get("source_prompt", "")
    response = row.get("response", "") or row.get("target_text", "") or row.get("output_text", "")
    if not prompt or not response:
        raise ValueError("no prompt/response")
    # add a minimal system header for consistency
    prompt_text = f"<system>\n{SYSTEM_FALLBACK}\n\n<user>\n{prompt}"
    return prompt_text, response


def build_fallback(row: Dict[str, Any]) -> Tuple[str, str]:
    """
    Very defensive fallback if schema is unknown.
    We try to extract role and a sensible input/target.
    """
    role = row.get("role", "assistant")
    # heuristics for common fields from earlier scripts
    problem = row.get("input_question", "") or row.get("problem", "")
    plan = row.get("plan", "")
    reasoner = row.get("reasoner_output", "") or row.get("normalized_rationale", "") or ""
    target = row.get("target_text", "") or row.get("response", "") or row.get("final_answer_norm", "") or row.get("final_answer", "")

    # Compose a compact instruction
    instr = [f"You are acting as <role:{role}>."]
    if problem:
        instr.append("Problem:\n" + problem)
    if plan and role in {"reasoner", "verifier", "normalizer"}:
        instr.append("Planner steps:\n" + plan)
    if reasoner and role in {"verifier", "normalizer"}:
        instr.append("Reasoner output:\n" + reasoner)

    # If we still don't have a target, guess from role-specific fields
    if not target:
        if role == "planner":
            target = row.get("planner_output", "")
        elif role == "reasoner":
            target = row.get("reasoner_output", "")
        elif role == "verifier":
            target = row.get("verifier_output", "")
        elif role in {"normalizer", "aggregator"}:
            target = row.get("normalized_rationale", "") or row.get("aggregated_rationale", "")
        else:
            target = row.get("response", "") or ""

    prompt_text = f"<system>\n{SYSTEM_FALLBACK}\n\n<user>\n" + "\n\n".join(instr)
    return prompt_text.strip(), (target or "").strip()


def extract_io(row: Dict[str, Any]) -> Tuple[str, str]:
    """Robustly produce (prompt, target) pair from a row."""
    # Prefer messages
    try:
        return build_from_messages(row)
    except Exception:
        pass
    # Then prompt/response
    try:
        return build_from_prompt_response(row)
    except Exception:
        pass
    # Fallback heuristic
    return build_fallback(row)


# -----------------------
# Tokenization & datasets
# -----------------------
def make_tokenizer(name_or_path: str):
    tok = AutoTokenizer.from_pretrained(name_or_path, use_fast=True)
    # Set pad/eos if missing
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token if tok.eos_token else "</s>"
    if tok.eos_token is None:
        tok.eos_token = tok.pad_token
    return tok


def tokenize_masked(example: Dict[str, Any], tokenizer, max_len: int) -> Dict[str, Any]:
    prompt, target = extract_io(example)
    full_text = prompt.strip() + "\n\n<assistant>\n" + target.strip()
    # tokenize prompt and full to mask labels before assistant
    prompt_ids = tokenizer(prompt, add_special_tokens=False)["input_ids"]
    full = tokenizer(
        full_text,
        max_length=max_len,
        truncation=True,
        padding=False,
        add_special_tokens=True,
    )
    input_ids = full["input_ids"]
    labels = input_ids.copy()

    # figure prompt length inside 'full'
    # safest: tokenize prompt+assistant tag and take its length
    pre_len = tokenizer(
        prompt.strip() + "\n\n<assistant>\n",
        add_special_tokens=False
    )["input_ids"]
    cutoff = min(len(input_ids), len(pre_len))
    labels[:cutoff] = [-100] * cutoff  # mask prompt
    return {
        "input_ids": input_ids,
        "attention_mask": full.get("attention_mask"),
        "labels": labels,
    }


def tokenize_packed(example: Dict[str, Any], tokenizer, max_len: int) -> Dict[str, Any]:
    # In packing mode, we do a simple casual-LM objective on all tokens (no masking).
    prompt, target = extract_io(example)
    full_text = prompt.strip() + "\n\n<assistant>\n" + target.strip()
    full = tokenizer(
        full_text,
        max_length=max_len,
        truncation=True,
        padding=False,
        add_special_tokens=True,
    )
    input_ids = full["input_ids"]
    labels = input_ids.copy()
    return {
        "input_ids": input_ids,
        "attention_mask": full.get("attention_mask"),
        "labels": labels,
    }


# -----------------------
# Model loading
# -----------------------
def load_model_and_tok(
    model_name: str,
    load_in_4bit: bool,
    bf16: bool,
    flash_attn: bool,
    gradient_checkpointing: bool,
):
    kwargs = {}

    if flash_attn:
        # transformers >= 4.38 supports attn_implementation arg
        kwargs["attn_implementation"] = "flash_attention_2"

    if load_in_4bit:
        if not HAS_BNB:
            raise RuntimeError("bitsandbytes not available but --load_in_4bit was set.")
        compute_dtype = torch.bfloat16 if bf16 else torch.float16
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=compute_dtype,
            bnb_4bit_use_double_quant=True,
        )
        kwargs["quantization_config"] = bnb_config
        kwargs["device_map"] = "auto"
    else:
        kwargs["torch_dtype"] = torch.bfloat16 if bf16 else torch.float16
        kwargs["device_map"] = "auto"

    tok = make_tokenizer(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name, **kwargs)

    if load_in_4bit:
        model = prepare_model_for_kbit_training(
            model,
            use_gradient_checkpointing=gradient_checkpointing
        )

    return model, tok


def attach_lora(
    model,
    r: int,
    alpha: int,
    dropout: float,
    target_modules_csv: str,
):
    targets = [t.strip() for t in target_modules_csv.split(",") if t.strip()]
    if not targets:
        # Attempt sensible defaults for LLaMA/Qwen families
        targets = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]

    peft_config = LoraConfig(
        r=r,
        lora_alpha=alpha,
        lora_dropout=dropout,
        bias="none",
        target_modules=targets,
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()
    return model

# -------- Robust collator for causal LM (pads inputs & labels) --------

@dataclass
class DataCollatorForCausalLMWithPadding:
    tokenizer: PreTrainedTokenizerBase
    pad_to_multiple_of: Optional[int] = 8

    def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
        # Determine max seq length in batch (then round up to multiple for kernel efficiency)
        max_len = max(len(f["input_ids"]) for f in features)
        if self.pad_to_multiple_of:
            rem = max_len % self.pad_to_multiple_of
            if rem != 0:
                max_len += (self.pad_to_multiple_of - rem)

        input_ids_batch, attn_batch, labels_batch = [], [], []

        for f in features:
            ids = list(f["input_ids"])
            am  = list(f.get("attention_mask", [1] * len(ids)))
            lbs = f.get("labels", None)

            pad_len = max_len - len(ids)
            if pad_len > 0:
                ids += [self.tokenizer.pad_token_id] * pad_len
                am  += [0] * pad_len
                if lbs is None:
                    # teach on all non-pad tokens by default
                    lbs = list(f["input_ids"]) + [-100] * pad_len
                else:
                    lbs = list(lbs) + [-100] * pad_len
            else:
                if lbs is None:
                    lbs = list(f["input_ids"])

            input_ids_batch.append(ids)
            attn_batch.append(am)
            labels_batch.append(lbs)

        return {
            "input_ids": torch.tensor(input_ids_batch, dtype=torch.long),
            "attention_mask": torch.tensor(attn_batch, dtype=torch.long),
            "labels": torch.tensor(labels_batch, dtype=torch.long),
        }



# -----------------------
# Main
# -----------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_file", type=str, required=True, help="Path to merged JSONL SFT file")
    ap.add_argument("--include_datasets", type=str, default="", help="CSV of datasets to include")
    ap.add_argument("--include_roles", type=str, default="", help="CSV of roles to include")

    # Split control (no leakage)
    ap.add_argument("--train_splits", type=str, default="train", help="CSV splits used for training")
    ap.add_argument("--eval_splits", type=str, default="", help="CSV splits used for eval; if empty, small slice from train is used")
    ap.add_argument("--eval_ratio", type=float, default=0.01, help="Fallback eval ratio if eval_splits not provided")
    ap.add_argument("--shuffle_before_split", action="store_true", help="Deterministic shuffle (by seed) before slicing")

    # Model / training
    ap.add_argument("--model_name_or_path", type=str, required=True)
    ap.add_argument("--output_dir", type=str, required=True)
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--batch_size", type=int, default=1)
    ap.add_argument("--grad_accum", type=int, default=16)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--warmup_ratio", type=float, default=0.03)
    ap.add_argument("--max_seq_len", type=int, default=2048)
    ap.add_argument("--logging_steps", type=int, default=50)
    ap.add_argument("--save_steps", type=int, default=500)
    ap.add_argument("--seed", type=int, default=42)

    # Precision / speed
    ap.add_argument("--load_in_4bit", action="store_true")
    ap.add_argument("--bf16", action="store_true")
    ap.add_argument("--fp16", action="store_true")  # used only if bf16 is False
    ap.add_argument("--flash_attn", action="store_true")
    ap.add_argument("--gradient_checkpointing", action="store_true")
    ap.add_argument("--packing", action="store_true")

    # LoRA
    ap.add_argument("--lora_r", type=int, default=64)
    ap.add_argument("--lora_alpha", type=int, default=16)
    ap.add_argument("--lora_dropout", type=float, default=0.05)
    ap.add_argument("--lora_target_modules", type=str, default="q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj")

    # Role balancing
    ap.add_argument("--role_weights", type=str, default="",
                    help='e.g. "planner:1,reasoner:1,verifier:3,normalizer:3,selector:1"')

    # Logging / reports
    ap.add_argument("--report_to", type=str, default="tensorboard", help="wandb|tensorboard|none (comma-separated ok)")
    ap.add_argument("--logging_dir", type=str, default="", help="Default: <output_dir>/runs")

    args = ap.parse_args()
    set_seed(args.seed)

    # ---------------- load & filter ----------------
    print(f"[LOAD] {args.data_file}")
    all_rows = load_jsonl(args.data_file)

    # First filter by dataset/roles (splits later)
    base_rows = filter_rows(all_rows, args.include_datasets, include_splits="", include_roles=args.include_roles)

    def filter_by_splits(rows, splits_csv):
        if not splits_csv:
            return rows
        allow = set(s.strip() for s in splits_csv.split(",") if s.strip())
        return [r for r in rows if r.get("split") in allow]

    train_rows = filter_by_splits(base_rows, args.train_splits)
    if len(train_rows) == 0:
        print("[ERR] No training rows after filters. Check --include_* and --train_splits.", file=sys.stderr)
        sys.exit(1)

    if args.shuffle_before_split:
        rng = random.Random(args.seed)
        rng.shuffle(train_rows)

    # Apply role weights
    role_w = parse_role_weights(args.role_weights)
    if role_w:
        train_rows = apply_role_weights(train_rows, role_w)
        print(f"[INFO] After role weighting: Train examples = {len(train_rows)}")

    # Eval rows
    if args.eval_splits:
        eval_rows = filter_by_splits(base_rows, args.eval_splits)
        if len(eval_rows) == 0:
            print("[ERR] No eval rows for --eval_splits; check flags.", file=sys.stderr)
            sys.exit(1)
    else:
        # small diagnostic slice from train (NOT official validation)
        n_eval = max(1, int(len(train_rows) * args.eval_ratio))
        eval_rows = train_rows[-n_eval:] if n_eval < len(train_rows) else train_rows[:1]
        train_rows = train_rows[:-n_eval] if n_eval < len(train_rows) else train_rows

    print(f"[INFO] Train examples: {len(train_rows)} | Eval examples: {len(eval_rows)}")

    # ---------------- model & tok ----------------
    model, tokenizer = load_model_and_tok(
        model_name=args.model_name_or_path,
        load_in_4bit=args.load_in_4bit,
        bf16=args.bf16,
        flash_attn=args.flash_attn,
        gradient_checkpointing=args.gradient_checkpointing
    )

    if args.gradient_checkpointing:
        model.gradient_checkpointing_enable()

    model = attach_lora(
        model,
        r=args.lora_r,
        alpha=args.lora_alpha,
        dropout=args.lora_dropout,
        target_modules_csv=args.lora_target_modules,
    )

    # ---------------- datasets ----------------
    token_fn = tokenize_packed if args.packing else tokenize_masked
    train_ds = Dataset.from_list(train_rows).map(
        lambda ex: token_fn(ex, tokenizer, args.max_seq_len),
        remove_columns=[c for c in set(train_rows[0].keys())],
        desc="Tokenizing train"
    )
    eval_ds = Dataset.from_list(eval_rows).map(
        lambda ex: token_fn(ex, tokenizer, args.max_seq_len),
        remove_columns=[c for c in set(eval_rows[0].keys())],
        desc="Tokenizing eval"
    )

    # --- ensure pad token ---
    if tokenizer.pad_token is None:
        # Qwen/Llama often have no pad token; use EOS as PAD
        tokenizer.pad_token = tokenizer.eos_token
    if model.config.pad_token_id is None:
        model.config.pad_token_id = tokenizer.pad_token_id
    tokenizer.padding_side = "right"

    # ---------------- collator ----------------
    # Use a padding-aware collator for CLM that pads to longest in batch
    data_collator = DataCollatorForCausalLMWithPadding(
        tokenizer=tokenizer,
        pad_to_multiple_of=8,   # or None if you prefer
    )

    # ---- Version-compatible TrainingArguments ----
    from inspect import signature
    from transformers import TrainingArguments

    ta_kwargs = dict(
        output_dir=args.output_dir,
        overwrite_output_dir=True,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        num_train_epochs=args.epochs,
        learning_rate=args.lr,
        warmup_ratio=args.warmup_ratio,
        logging_steps=50,
        save_strategy="steps",
        save_steps=500,
        save_total_limit=2,
        do_eval=True,
        eval_steps=500,
        report_to=["tensorboard"],
        gradient_checkpointing=args.gradient_checkpointing,
        dataloader_pin_memory=False,
    )

    # bf16 / fp16 toggling (keep it simple + safe)
    if getattr(args, "bf16", False):
        ta_kwargs["bf16"] = True
    else:
        # if you had --fp16 flag, honor it; otherwise leave unset
        if getattr(args, "fp16", False):
            ta_kwargs["fp16"] = True

    # Handle API differences: evaluation_strategy vs eval_strategy
    sig = signature(TrainingArguments.__init__)
    if "evaluation_strategy" in sig.parameters:
        ta_kwargs["evaluation_strategy"] = "steps"
    elif "eval_strategy" in sig.parameters:
        ta_kwargs["eval_strategy"] = "steps"
    # else: very old Transformers—no strategy arg; Trainer will still eval if do_eval=True and eval_steps set

    train_args = TrainingArguments(**ta_kwargs)

    # ---------------- trainer ----------------
    trainer = Trainer(
        model=model,
        args=train_args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        tokenizer=tokenizer,
        data_collator=data_collator,
    )

    # ---------------- train ----------------
    trainer.train()

    # Save adapter + tokenizer
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)

    # Final eval & perplexity log
    metrics = trainer.evaluate()
    if "eval_loss" in metrics and metrics["eval_loss"] is not None:
        try:
            ppl = math.exp(metrics["eval_loss"])
            metrics["eval_perplexity"] = ppl
        except Exception:
            pass
    trainer.log_metrics("eval", metrics)
    trainer.save_metrics("eval", metrics)
    trainer.save_state()

    print("[DONE] Training complete. Artifacts in:", args.output_dir)


if __name__ == "__main__":
    main()
