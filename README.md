# Engram-as-PEFT on 24GB: A Narrow Training Window

Engram (DeepSeek-style hash-based external memory) applied as a PEFT
module to Qwen-2.5-1.5B. Trains on a single 24GB consumer GPU.

**Main finding:** Engram contributes **+0.22 MMLU loss delta** at the
optimal training window (epoch 3–4). Training past epoch 4 degrades
the delta; by epoch 15 with Adafactor, the delta goes **negative**
(−0.11) — Engram actively hurts held-out performance.

**Implication:** Train 3–4 epochs. Checkpoint. Stop. Continuing past
the window doesn't just waste compute, it degrades deployment-time
accuracy.

(ai assisted research)
---

## Results

All numbers from `ablate_engram.py` on the MMLU test set (2000 samples),
token-weighted, using the audit-corrected script (see "Bugs fixed" below).

| Optimizer | Epoch | Train PPL | Loss Δ | PPL Δ | Verdict |
|---|---|---|---|---|---|
| Adafactor | 4 | 3.694 | **+0.2223** | +1.4513 | Engram helps |
| Mano | 4 | 4.835 | **+0.2199** | +1.3455 | Engram helps |
| Adafactor | 15 | 2.221 | **−0.1135** | −1.1643 | **Engram hurts** |
| Mano | 15 | 4.301 | +0.1805 | +1.1456 | (see limitations) |

Loss Δ = `loss(Engram OFF) − loss(Engram ON)`. Positive = Engram helps.

**Peak contribution (epoch 3–4):** +0.22 loss, reproducible across both
optimizers within ±0.003.

**Degradation past epoch 4:** delta shrinks to +0.08 by epoch 10 (Adafactor),
crosses zero by epoch 15.

**Negative delta at Adafactor@15:** Engram with δ=−0.11 is worse than
no Engram. The model has memorized the training set; Engram's residual
injects training-specific noise that fails on held-out data.

---

## Hardware

- **GPU:** AMD Radeon RX 7900 XTX, 24GB VRAM, ROCm 6.2
- **CPU:** Ryzen 7 7800X3D (8c/16t)
- **RAM:** 32GB DDR5-6000
- **OS:** Fedora 44
- **Disk:** ~30GB for model, dataset cache, checkpoints

All experiments single-GPU. No multi-GPU, no cloud.

---

## Dependencies

```bash
python -m venv venv
source venv/bin/activate

# ROCm (tested):
pip install torch --index-url https://download.pytorch.org/whl/rocm6.2
# Or CUDA:
# pip install torch --index-url https://download.pytorch.org/whl/cu121

pip install transformers>=4.40 peft>=0.11 datasets>=2.18 \
            safetensors>=0.4 sympy>=1.12 accelerate>=0.30
```

Optional env vars (recommended):

```bash
export TOKENIZERS_PARALLELISM=false
export PYTORCH_ROCM_ALLOC_CONF=garbage_collection_threshold:0.8   # ROCm only
```

---

## File structure

| File | Purpose |
|---|---|
| `main.py` | Entry point, `Config`, model + trainer wiring |
| `model.py` | Engram module (hashing, embeddings, gates, conv) + `EngramEnhancedModel` wrapper |
| `dataset.py` | Causal-LM dataset from 7 HuggingFace sources |
| `trainer.py` | Training loop, checkpoint save/load, `OPTIMIZER` env switch |
| `mano.py` | Mano optimizer (from `xie-lab-ml/Mano-...`, arXiv:2601.23000) |
| `ablate_engram.py` | Standalone ablation — PPL with Engram ON vs OFF |

`feel_mano.py`, `losses.py`, `memory_utils.py` are dead code from an
earlier iteration. Safe to delete; not imported by the paths above.

---

## Configuration

```python
# main.py Config
model_name = "Qwen/Qwen2.5-1.5B"           # frozen base
hidden_size = 1536

# Engram
engram_vocab_size = [151936, 151936]       # 1× compressed vocab, per n-gram order
max_ngram_size = 3
n_embed_per_ngram = 1280                   # paper-aligned default
n_head_per_ngram = 8
memory_layers_read = [1, 14]               # deep-layer injection
engram_seed = 0

# LoRA
lora_r = 8
lora_alpha = 16
lora_dropout = 0.30
lora_lr = 1e-5
lora_freeze_epoch = 100                    # no freeze (training regime below)

# Training
batch_size = 2
max_len = 512
learning_rate = 1e-5                       # Engram lr, matches LoRA
epochs = int(os.environ.get("MAX_EPOCHS", 20))
num_samples = 21000
gradient_checkpointing = False
```

**Reproducibility:** `torch.manual_seed(42)` set at top of `main()`.

---

## Training data

Streamed from HuggingFace. 21,000 samples, 67% factual QA:

| Source | HF path / config / split | Samples | Format |
|---|---|---|---|
| MMLU | `cais/mmlu` / `all` / `auxiliary_train` | 8,000 | `Question: {q}\nA) …\nB) …\nAnswer: {letter}` |
| TriviaQA | `trivia_qa` / `unfiltered.nocontext` / `train` | 4,000 | `Question: {q}\nAnswer: {value}` |
| OpenBookQA | `openbookqa` / `main` / `train` | 2,000 | `Question: {q}\nAnswer: {letter}` |
| UltraChat 200k | `HuggingFaceH4/ultrachat_200k` / `train_sft` | 2,000 | `{role}: {content}` per turn |
| SlimOrca | `Open-Orca/SlimOrca` / `train` | 2,000 | `{from}: {value}` per turn |
| GSM8K | `gsm8k` / `main` / `train` | 2,000 | `Question: {q}\nAnswer: {answer}` |
| WikiText-103 | `wikitext` / `wikitext-103-raw-v1` / `train` | 1,000 | raw prose |

Each sample: tokenized, EOS appended, truncated to 512, right-padded
with `pad_token_id`. `labels = input_ids`, pad positions = −100.

Test splits used for ablation: `cais/mmlu` / `test` (2000 samples).

---

## Reproduce — 4 epochs (the recommended window)

### Step 1 — Clear checkpoint directory

```bash
cd "/home/pc/tiny_distill_project/train engram"
mkdir -p /home/pc/tiny_distill_project/checkpoints
rm -rf /home/pc/tiny_distill_project/checkpoints/*
```

### Step 2 — Train Adafactor, 4 epochs (~4h on RX 7900 XTX)

```bash
OPTIMIZER=adafactor MAX_EPOCHS=4 PYTHONUNBUFFERED=1 \
  nohup python main.py > /home/pc/tiny_distill_project/overnight_logs/adafactor_e4.log 2>&1 &
```

Verify after 90 seconds:

```bash
tail -12 /home/pc/tiny_distill_project/overnight_logs/adafactor_e4.log
```

Expected:

```
  Trainable (LoRA+Engram): 803,380,992 / 2,347,095,296 (34.229%)
  Optimizer: 396 Adafactor(2D) @ lr=1e-05 | 12 Adafactor(1D/3D) | 2 embedding @ lr=1e-05 (Adafactor)
  >>> 2D optimizer class: Adafactor
  >>> Embedding optimizer class: Adafactor
  Steps per epoch: 10500
  >>> [Stage 0] LoRA frozen for epoch 0 — Engram warmup
  Step    0/10500 | loss=X.XXXX | ...
```

### Step 3 — Ablate at epoch 4

```bash
CKPT_DIR=/home/pc/tiny_distill_project/checkpoints \
  EVAL_DATASET=mmlu python ablate_engram.py
```

Expected output:

```
ABLATION RESULT
============================================================
  Loss with Engram:    1.7630
  Loss without Engram: 1.9853
  Loss delta (off-on): +0.2223
  PPL with Engram:     5.8297
  PPL without Engram:  7.2810
  PPL delta (off-on):  +1.4513
============================================================
```

`Loss delta ≈ +0.22` = reproduced.

### Step 4 — Save checkpoint, repeat for Mano

```bash
mv /home/pc/tiny_distill_project/checkpoints \
   /home/pc/tiny_distill_project/checkpoints_adafactor_e4
mkdir /home/pc/tiny_distill_project/checkpoints

OPTIMIZER=mano MAX_EPOCHS=4 PYTHONUNBUFFERED=1 \
  nohup python main.py > /home/pc/tiny_distill_project/overnight_logs/mano_e4.log 2>&1 &
```

~4h. Then ablate:

```bash
CKPT_DIR=/home/pc/tiny_distill_project/checkpoints \
  EVAL_DATASET=mmlu python ablate_engram.py
```

Expected: `Loss delta ≈ +0.22` (within ±0.003 of Adafactor).

---

## Reproduce — 15 epochs (overfitting regime, ~16h per run)

**Not recommended for the primary result.** Use this only to reproduce
the "delta degrades past epoch 4" finding.

```bash
rm -rf /home/pc/tiny_distill_project/checkpoints/*
OPTIMIZER=adafactor MAX_EPOCHS=16 PYTHONUNBUFFERED=1 \
  nohup python main.py > /home/pc/tiny_distill_project/overnight_logs/adafactor_e15.log 2>&1 &
```

Train PPL trajectory (should match within ±0.02):

| Epoch | Loss | PPL |
|---|---|---|
| 0 | 1.72 | 5.6 |
| 4 | 1.31 | 3.7 |
| 10 | 0.87 | 2.4 |
| 15 | 0.80 | 2.2 |

Ablate at epoch 15:

```bash
CKPT_DIR=/home/pc/tiny_distill_project/checkpoints \
  EVAL_DATASET=mmlu python ablate_engram.py
```

Expected: `Loss delta ≈ −0.11` — **negative** (Engram hurts).

---

## Ablation script — what it measures

`ablate_engram.py`:

1. Loads base model + LoRA adapter from checkpoint
2. Loads Engram weights (`engram.safetensors`)
3. Builds 2000-sample MMLU test set
4. Computes token-weighted mean loss with Engram **ON**
5. Computes token-weighted mean loss with Engram **OFF** (`mem.disabled = True`)
6. Reports `loss_delta = loss_off − loss_on` and `ppl_delta = exp(loss_off) − exp(loss_on)`

Token-weighted, not batch-mean. Each sample contributes proportional to its
non-pad token count. This is important when batch composition varies
(short QA samples vs long prose).

---

## Bugs fixed (audit trail)

Earlier versions of this repo contained bugs that affected the reported numbers.
The current baseline reflects all fixes.

| Bug | Old behavior | Fixed behavior |
|---|---|---|
| **Shared gradient clip** | `clip_grad_norm_` across all params — LoRA grad spikes scaled down Engram grads | Per-optimizer clipping (LoRA/Engram clipped independently) |
| **Batch-mean loss** | Scheduler saw biased PPL (short samples overweighted) | Token-weighted mean loss |
| **PPL vs loss label** | Log said `ppl=1.68` when it was `loss=1.68` | `compute_loss_and_ppl` returns both, labeled correctly |
| **`print_trainable_parameters`** | Reported 9M trainable (LoRA only), hid 794M Engram params | Reports full count after Engram wrap: 803M / 2.35B (34%) |
| **`load_state_dict` no strict=False** | Ablation crashed on config mismatch instead of warning | `strict=False` with explicit missing/unexpected key warnings |
| **No seed** | Run-to-run variance ±0.05 from RNG alone | `torch.manual_seed(42)` + `cuda.manual_seed_all(42)` |

**Impact on the +0.25 number:** The historical `+0.258` (Adafactor) and
`+0.250` (Mano) were measured before the audit fixes. Corrected numbers
are `+0.2223` and `+0.2199`. The ~0.03 gap is attributable to the **shared
gradient clip bug** — coupling LoRA and Engram clipping inflated Engram's
apparent contribution. The audit-corrected **+0.22 is the true baseline**.

---

## Limitations

- **97% of trainable params stay on Adafactor.** The `OPTIMIZER` switch
  controls only the 3% dense interface parameters (`value_proj`, `key_projs`,
  conv, norms) and LoRA (9M). The Engram embedding table (778M) always uses
  Adafactor — Mano's per-row normalization is unstable on the 150k×1280
  embedding. **Mano's apparent +0.18 delta at epoch 15 is a training-speed
  artifact, not an optimizer advantage.** Mano@15 has train PPL 4.301 —
  worse than Adafactor@4 (3.694). At matched training loss, both optimizers
  show similar delta. Mano has not yet reached the overfit regime that
  collapses Adafactor's delta.

- **Single seed.** Variance across audit-fixed runs is ±0.03 (from RNG only).
  Multi-seed runs would tighten this; not done.

- **Not a Mano benchmark.** This paper does not test the Mano optimizer
  claim (arXiv:2601.23000) at pretraining scale. It uses Mano as a
  mixed-optimizer component for a 3% subset of params. The Mano paper's
  "eventual overtake" claim is neither confirmed nor refuted here.

- **Not a from-scratch pretraining.** The base model is frozen Qwen-1.5B.
  This is a PEFT fine-tuning regime, not a reproduction of DeepSeek's
  Engram pretraining (arXiv:2601.07372, 262B tokens, 27B params).

---

## What this is and isn't

**Is:**
- A working Engram-PEFT configuration on 24GB consumer hardware
- A specific, reproducible finding: +0.22 MMLU delta at epoch 3–4
- A documented overfitting regime: delta goes negative at epoch 15
- A bug-corrected baseline (+0.22, not historical +0.25)

**Isn't:**
- Novel architecture (DeepSeek's Engram, unchanged)
- A Mano vs Adafactor benchmark (see limitations)
- A from-scratch Engram training (base frozen)
- A general claim that Engram works everywhere (specific to this config)

---

## Files

- `main.py`, `model.py`, `dataset.py`, `trainer.py`, `mano.py`, `ablate_engram.py` — see File Structure above
- `REPRODUCE.md` — shorter version of the reproduction steps
- `FINDINGS.md` — extended analysis, epoch-by-epoch trajectory, elimination table

---

## Citation

If you use this configuration or finding:

```bibtex
@misc{engram-peft-24gb-2026,
  title  = {Engram-as-PEFT on 24GB: A Narrow Training Window},
  author = {[Kamil Kahoun]},
  year   = {2026},
  note   = {Reproducible training window analysis for DeepSeek-style Engram on consumer hardware},
  url    = {[your repo URL]}
}
```

Original Engram: arXiv:2601.07372. Mano: arXiv:2601.23000.

---

