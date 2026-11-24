"""
Throwaway — ablation: does Engram actually help?
Compare PPL with Engram enabled vs disabled on held-out WikiText.
"""

import os
import math
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import LoraConfig, get_peft_model, TaskType
from safetensors.torch import load_file
from datasets import load_dataset
from model import EngramEnhancedModel

device = "cuda"
ckpt = os.environ.get("CKPT_DIR", "/home/pc/tiny_distill_project/checkpoints")

tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-1.5B")
tokenizer.pad_token = tokenizer.eos_token

print("Loading base + LoRA...")
base = AutoModelForCausalLM.from_pretrained(
    "Qwen/Qwen2.5-1.5B", torch_dtype=torch.bfloat16, attn_implementation="sdpa"
).to(device)
base.config.use_cache = False
base = get_peft_model(base, LoraConfig(
    task_type=TaskType.CAUSAL_LM, r=8, lora_alpha=16, lora_dropout=0.0,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                    "gate_proj", "up_proj", "down_proj"],
    bias="none",
))
base.load_adapter(f"{ckpt}/lora_adapter", adapter_name="default")

print("Loading Engram model...")
model = EngramEnhancedModel(
    base_model=base,
    engram_vocab_size=[151936, 151936],
    max_ngram_size=3, n_embed_per_ngram=1280, n_head_per_ngram=8,
    read_layers=(1, 14), tokenizer_name_or_path="Qwen/Qwen2.5-1.5B",
    pad_id=tokenizer.pad_token_id, seed=0,
).to(device, dtype=torch.bfloat16)

engram_sd = load_file(f"{ckpt}/engram.safetensors")
missing, unexpected = model.memory_layers.load_state_dict(engram_sd, strict=False)
if missing or unexpected:
    print(f"  [WARN] Engram load: {len(missing)} missing keys, {len(unexpected)} unexpected keys")
    print(f"         first missing: {missing[:5]}")
    print(f"         first unexpected: {unexpected[:5]}")
    print(f"         CHECK: does ablate_engram.py config match the checkpoint?")
else:
    print("  Engram weights loaded (all keys matched)")

# Build held-out eval set — dataset chosen by EVAL_DATASET env var
# Usage: EVAL_DATASET=wikitext python ablate_engram.py
#        EVAL_DATASET=gsm8k    python ablate_engram.py
#        EVAL_DATASET=mmlu     python ablate_engram.py
import os
which = os.environ.get("EVAL_DATASET", "wikitext").lower()
print(f"Building eval set from: {which}")

def make_sample(text):
    if not text or len(text.strip()) < 20:
        return None
    ids = tokenizer(text, add_special_tokens=False,
                    truncation=True, max_length=511)["input_ids"]
    if len(ids) < 8:
        return None
    ids.append(tokenizer.eos_token_id)
    if len(ids) < 512:
        ids += [tokenizer.pad_token_id] * (512 - len(ids))
    labels = [i if i != tokenizer.pad_token_id else -100 for i in ids]
    return {
        "input_ids": torch.tensor(ids, dtype=torch.long),
        "labels": torch.tensor(labels, dtype=torch.long),
    }

eval_samples = []

if which == "wikitext":
    ds = load_dataset("wikitext", "wikitext-103-raw-v1", split="test", streaming=True)
    for item in ds:
        text = (item.get("text", "") or "").strip()
        if len(text) < 50:
            continue
        s = make_sample(text)
        if s is not None:
            eval_samples.append(s)
        if len(eval_samples) >= 2000:
            break

elif which == "gsm8k":
    ds = load_dataset("gsm8k", "main", split="test", streaming=True)
    for item in ds:
        q = (item.get("question", "") or "").strip()
        a = (item.get("answer", "") or "").strip()
        if not q or not a:
            continue
        s = make_sample(f"Question: {q}\nAnswer: {a}")
        if s is not None:
            eval_samples.append(s)
        if len(eval_samples) >= 2000:
            break

elif which == "mmlu":
    letters = "ABCD"
    ds = load_dataset("cais/mmlu", "all", split="test", streaming=True)
    for item in ds:
        q = (item.get("question", "") or "").strip()
        choices = item.get("choices", [])
        ans_idx = item.get("answer", 0)
        if not q or not choices or not (0 <= ans_idx < len(choices)):
            continue
        opts = "\n".join(
            f"{letters[i]}) {c}" for i, c in enumerate(choices) if i < len(letters)
        )
        s = make_sample(f"Question: {q}\n{opts}\nAnswer: {letters[ans_idx]}")
        if s is not None:
            eval_samples.append(s)
        if len(eval_samples) >= 2000:
            break

else:
    raise ValueError(f"Unknown EVAL_DATASET: {which}")

print(f"  {len(eval_samples)} eval samples")

def compute_loss_and_ppl(engram_enabled):
    """Returns (token-weighted mean loss, perplexity = exp(loss))."""
    for mem in model.memory_layers.values():
        mem.disabled = not engram_enabled
    model.eval()
    total_loss = 0.0
    total_tokens = 0
    with torch.no_grad():
        for i in range(0, len(eval_samples), 4):
            batch = eval_samples[i:i+4]
            ids = torch.stack([b["input_ids"] for b in batch]).to(device)
            lbl = torch.stack([b["labels"] for b in batch]).to(device)
            attn = (lbl != -100).long()
            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                out = model(input_ids=ids, attention_mask=attn, labels=lbl)
            n_real = (lbl != -100).sum().item()
            total_loss += out.loss.item() * n_real
            total_tokens += n_real
    for mem in model.memory_layers.values():
        mem.disabled = False
    avg_loss = total_loss / max(total_tokens, 1)
    return avg_loss, math.exp(avg_loss)

print("\nComputing with Engram ON...")
loss_on, ppl_on = compute_loss_and_ppl(True)
print(f"  mean loss: {loss_on:.4f}  |  PPL: {ppl_on:.4f}")

print("Computing with Engram OFF...")
loss_off, ppl_off = compute_loss_and_ppl(False)
print(f"  mean loss: {loss_off:.4f}  |  PPL: {ppl_off:.4f}")

loss_delta = loss_off - loss_on
ppl_delta = ppl_off - ppl_on
print(f"\n{'='*60}")
print(f"ABLATION RESULT")
print(f"{'='*60}")
print(f"  Loss with Engram:    {loss_on:.4f}")
print(f"  Loss without Engram: {loss_off:.4f}")
print(f"  Loss delta (off-on): {loss_delta:+.4f}")
print(f"  PPL with Engram:     {ppl_on:.4f}")
print(f"  PPL without Engram:  {ppl_off:.4f}")
print(f"  PPL delta (off-on):  {ppl_delta:+.4f}")
print(f"{'='*60}")

if loss_delta > 0.20:
    print("  → Engram contributes meaningfully.")
elif loss_delta > 0.05:
    print("  → Engram contributes modestly.")
else:
    print("  → Engram contribution is marginal.")
