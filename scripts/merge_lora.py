# merge_lora.py
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
base = "Qwen/Qwen3-8B"                      # or your local path
adapter_dir = "outputs/lora_sft_gsm_multiarith"  # your run
out = "outputs/merged_qwen3_8b_cot"         # where to save merged

tok = AutoTokenizer.from_pretrained(base, use_fast=True)
base_model = AutoModelForCausalLM.from_pretrained(base, torch_dtype="auto", device_map="cpu")
model = PeftModel.from_pretrained(base_model, adapter_dir)
merged = model.merge_and_unload()           # bake LoRA weights into the base
merged.save_pretrained(out)
tok.save_pretrained(out)
print("Merged model saved to:", out)
