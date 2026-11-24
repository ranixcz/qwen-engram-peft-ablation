"""
trainer.py — Phase 1 standard causal-LM loop.
==============================================
- Two optimizer groups: LoRA @ lora_lr, Engram @ learning_rate
- Checkpoints: LoRA adapter + Engram safetensors + meta
- Loss = standard causal-LM CE (pads masked via -100 in labels)
- No paired storage/retrieval loop, no credit engine, no sampler

Legacy components (credit_engine, retrieval_tracker, sampler,
routing_tracker, fact_masker) are stored on `self` but not used —
so re-integrating the paired regime later is a loop change, not a
main.py change.
"""
import os
import math
import gc
import datetime
import torch
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader
from transformers import Adafactor


def _grad_max_abs(p):
    """Sparse-safe max(|grad|). Works for both dense and sparse gradients."""
    if p.grad is None:
        return 0.0
    g = p.grad
    if g.is_sparse:
        vals = g.coalesce().values()
        if vals.numel() == 0:
            return 0.0
        return float(vals.abs().max().item())
    return float(g.abs().max().item())


class Trainer:
    def __init__(self, model, dataset, tokenizer, config,
                 credit_engine=None, retrieval_tracker=None, sampler=None,
                 routing_tracker=None, fact_masker=None):
        self.model = model
        self.dataset = dataset
        self.tokenizer = tokenizer
        self.config = config
        self.device = torch.device(config.device)

        # LEGACY (dead code Phase 1) — stored for future re-integration.
        self.credit_engine = credit_engine
        self.retrieval_tracker = retrieval_tracker
        self.sampler = sampler
        self.routing_tracker = routing_tracker
        self.fact_masker = fact_masker

        self.model = self.model.to(
            device=self.device,
            dtype=torch.bfloat16 if self.device.type == "cuda" else torch.float32,
        )
        self.optimizer_engram = None
        self.optimizer_mano = None
        self.scheduler = None

    def _create_optimizer(self):
        # Classify every trainable parameter:
        #   mano_params       — 2D matrices → Mano (or Adafactor when OPTIMIZER=adafactor)
        #   adamw_params      — 1D/3D params → AdamW (or Adafactor when OPTIMIZER=adafactor)
        #   embedding_params  — Engram embedding table → always Adafactor
        lora_params = [p for n, p in self.model.base_model.named_parameters()
                       if p.requires_grad and "lora_" in n]
        lora_params = list(dict.fromkeys(lora_params))

        embedding_params = []
        mano_params = []
        adamw_params = []

        for name, p in self.model.memory_layers.named_parameters():
            if not p.requires_grad:
                continue
            if "multi_head_embedding.embedding" in name:
                embedding_params.append(p)
            elif p.ndim == 2:
                mano_params.append(p)
            else:
                adamw_params.append(p)

        for p in lora_params:
            if p.ndim == 2:
                mano_params.append(p)
            else:
                adamw_params.append(p)

        optimizer_choice = os.environ.get("OPTIMIZER", "mano").lower()
        if optimizer_choice == "mano":
            opt_2d_name = "Mano"
            opt_1d3d_name = "AdamW"
        elif optimizer_choice == "adafactor":
            opt_2d_name = "Adafactor"
            opt_1d3d_name = "Adafactor"
        else:
            raise ValueError(f"Unknown OPTIMIZER: {optimizer_choice}")

        print(f"  Optimizer: "
              f"{len(mano_params)} {opt_2d_name}(2D) @ lr={self.config.lora_lr} | "
              f"{len(adamw_params)} {opt_1d3d_name}(1D/3D) | "
              f"{len(embedding_params)} embedding @ lr={self.config.learning_rate} (Adafactor)")

        if optimizer_choice == "mano":
            from mano import Mano
            self.optimizer_mano = Mano(
                mano_params=mano_params,
                lr=self.config.lora_lr,
                wd=0.01,
                momentum=0.95,
                nesterov=True,
                adamw_params=adamw_params,
                adamw_betas=(0.9, 0.95),
                adamw_eps=1e-8,
            )
        elif optimizer_choice == "adafactor":
            self.optimizer_mano = Adafactor(
                [
                    {"params": mano_params, "lr": self.config.lora_lr},
                    {"params": adamw_params, "lr": self.config.lora_lr},
                ],
                lr=self.config.lora_lr,
                eps=(1e-30, 1e-3),
                clip_threshold=1.0,
                decay_rate=-0.8,
                beta1=None,
                weight_decay=0.0,
                relative_step=False,
                scale_parameter=False,
                warmup_init=False,
            )

        self.optimizer_engram = Adafactor(
            embedding_params,
            lr=self.config.learning_rate,
            eps=(1e-30, 1e-3),
            clip_threshold=1.0,
            decay_rate=-0.8,
            beta1=None,
            weight_decay=0.0,
            relative_step=False,
            scale_parameter=False,
            warmup_init=False,
        )

        self.scheduler = ReduceLROnPlateau(
            self.optimizer_mano, mode="min", factor=0.5, patience=10, min_lr=1e-7
        )

        print(f"  >>> 2D optimizer class: {type(self.optimizer_mano).__name__}")
        print(f"  >>> Embedding optimizer class: {type(self.optimizer_engram).__name__}")

    # ----------------------------------------------------------------
    # Checkpoint save / load
    # ----------------------------------------------------------------
    def _save_checkpoint(self, epoch, best_ppl):
        os.makedirs(self.config.checkpoint_dir, exist_ok=True)

        from peft import PeftModel
        if isinstance(self.model.base_model, PeftModel):
            self.model.base_model.save_pretrained(
                f"{self.config.checkpoint_dir}/lora_adapter"
            )

        from safetensors.torch import save_file
        engram_sd = {
            k: v.detach().cpu().contiguous()
            for k, v in self.model.memory_layers.state_dict().items()
        }
        save_file(engram_sd, f"{self.config.checkpoint_dir}/engram.safetensors")

        torch.save({
            "optimizer_engram_state_dict": self.optimizer_engram.state_dict(),
            "optimizer_mano_state_dict": self.optimizer_mano.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict(),
            "best_ppl": best_ppl,
            "epoch": epoch,
        }, f"{self.config.checkpoint_dir}/training_meta.pt")
        print(f"  -> Checkpoint saved (epoch {epoch}, ppl={best_ppl:.3f})")

    def _load_checkpoint(self):
        ckpt = self.config.checkpoint_dir
        meta_path = f"{ckpt}/training_meta.pt"
        engram_path = f"{ckpt}/engram.safetensors"
        lora_path = f"{ckpt}/lora_adapter"

        if not (os.path.exists(meta_path) and os.path.exists(engram_path)):
            return 0, float("inf")

        print("  Loading checkpoint...")

        from peft import PeftModel
        if os.path.exists(lora_path) and isinstance(self.model.base_model, PeftModel):
            try:
                self.model.base_model.load_adapter(lora_path, adapter_name="default")
                print("  -> LoRA adapter loaded")
            except Exception as e:
                print(f"  [WARNING] Could not load LoRA adapter: {e}")

        from safetensors.torch import load_file
        try:
            engram_sd = load_file(engram_path)
            missing, unexpected = self.model.memory_layers.load_state_dict(
                engram_sd, strict=False
            )
            print(f"  -> Engram weights loaded "
                  f"(missing={len(missing)}, unexpected={len(unexpected)})")
        except Exception as e:
            print(f"  [WARNING] Could not load Engram weights: {e}")

        meta = torch.load(meta_path, map_location="cpu")
        self._create_optimizer()

        # Explicit key checks so legacy checkpoints surface clearly instead of
        # silently dropping optimizer state via a caught KeyError.
        if "optimizer_mano_state_dict" not in meta:
            print(f"  [WARN] legacy checkpoint — no optimizer_mano_state_dict; "
                  f"optimizer state will be fresh. Expected key names: "
                  f"{sorted(meta.keys())}")
        else:
            try:
                self.optimizer_engram.load_state_dict(meta["optimizer_engram_state_dict"])
                self.optimizer_mano.load_state_dict(meta["optimizer_mano_state_dict"])
                self.scheduler.load_state_dict(meta["scheduler_state_dict"])
                print(f"  -> Optimizer + scheduler state restored")
            except Exception as e:
                print(f"  [WARN] optimizer/scheduler restore failed: {e}")

        start_epoch = meta.get("epoch", 0) + 1
        best_ppl = meta.get("best_ppl", float("inf"))
        print(f"  Resuming from epoch {start_epoch}, best ppl {best_ppl:.3f}")
        return start_epoch, best_ppl

    # ----------------------------------------------------------------
    # Main loop
    # ----------------------------------------------------------------
    def run(self):
        config = self.config

        # Gradient checkpointing is incompatible with the current Engram
        # hook (stateful _current_input_ids). Disable it — Phase 1 fits
        # in VRAM without it.
        if config.gradient_checkpointing:
            print("  [WARNING] gradient_checkpointing is incompatible with "
                  "Engram's stateful hook. Disabling for Phase 1.")
            try:
                self.model.base_model.gradient_checkpointing_disable()
            except Exception:
                pass
            config.gradient_checkpointing = False

        start_epoch, best_ppl = self._load_checkpoint()
        if self.optimizer_mano is None:
            self._create_optimizer()

        loader = DataLoader(
            self.dataset,
            batch_size=config.batch_size,
            shuffle=True,
            num_workers=2,
            pin_memory=True,
            drop_last=True,
        )

        print(f"  Steps per epoch: {len(loader)}")

        for epoch in range(start_epoch, config.epochs):
            # --- Stage 0: freeze LoRA for epoch 0, only Engram trains ---
            if epoch == 0:
                for n, p in self.model.base_model.named_parameters():
                    if "lora_" in n:
                        p.requires_grad = False
                print(f"  >>> [Stage 0] LoRA frozen for epoch 0 — Engram warmup")
            elif epoch == 1:
                for n, p in self.model.base_model.named_parameters():
                    if "lora_" in n:
                        p.requires_grad = True
                print(f"  >>> [Stage 1] LoRA unfrozen — joint training begins")
            if epoch == config.lora_freeze_epoch:
                frozen = 0
                for n, p in self.model.base_model.named_parameters():
                    if "lora_" in n and p.requires_grad:
                        p.requires_grad = False
                        frozen += 1
                print(f"  >>> Froze {frozen} LoRA tensors at epoch {epoch}")

            epoch_loss_sum = 0.0
            epoch_tokens = 0
            n_steps = 0
            # Per-mode loss accumulators (for gradient-routing diagnostic)
            mode_loss_sums = {"both": 0.0, "lora_only": 0.0, "engram_only": 0.0}
            mode_loss_counts = {"both": 0, "lora_only": 0, "engram_only": 0}

            for step, batch in enumerate(loader):
                input_ids = batch["input_ids"].to(self.device, non_blocking=True)
                labels = batch["labels"].to(self.device, non_blocking=True)

                # CRITICAL: use labels != -100 for attention_mask, NOT
                # input_ids != pad_id. Qwen's pad_token == eos_token, so the
                # pad comparison would mask out EOS and break generation.
                attn = (labels != -100).long()

                self.optimizer_engram.zero_grad(set_to_none=True)
                self.optimizer_mano.zero_grad(set_to_none=True)

                with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                    out = self.model(
                        input_ids=input_ids,
                        attention_mask=attn,
                        labels=labels,
                    )
                    loss = out.loss

                # --- Gradient routing: 1/1/1 cycle (both / lora_only / engram_only) ---
                r = 0
                mode = "both"
                frozen_lora = []
                frozen_engram = []
                if mode == "lora_only":
                    for p in self.model.memory_layers.parameters():
                        if p.requires_grad:
                            p.requires_grad = False
                            frozen_engram.append(p)
                elif mode == "engram_only":
                    for n, p in self.model.base_model.named_parameters():
                        if "lora_" in n and p.requires_grad:
                            p.requires_grad = False
                            frozen_lora.append(p)

                loss.backward()

                # Restore requires_grad so subsequent steps and checkpointing work
                for p in frozen_lora:
                    p.requires_grad = True
                for p in frozen_engram:
                    p.requires_grad = True

                # ---- Engram health check (BEFORE clip — sees raw gradients) ----
                if step % 100 == 0:
                    engram_parts = []
                    for name, mem in self.model.memory_layers.items():
                        n_grad = sum(
                            1 for p in mem.parameters()
                            if _grad_max_abs(p) > 0
                        )
                        n_total = sum(1 for _ in mem.parameters())

                        emb_param = mem.multi_head_embedding.embedding.weight
                        emb_norm = _grad_max_abs(emb_param)

                        gate = getattr(mem, 'last_gate', None)
                        if gate is not None:
                            gate_str = f"gate={gate.mean().item():.3f}±{gate.std().item():.3f}"
                        else:
                            gate_str = "gate=n/a"

                        engram_parts.append(
                            f"L{name}[{n_grad}/{n_total}, emb={emb_norm:.2e}, {gate_str}]"
                        )

                    lora_total = sum(
                        1 for n, p in self.model.base_model.named_parameters()
                        if "lora_" in n
                    )
                    lora_grad = sum(
                        1 for n, p in self.model.base_model.named_parameters()
                        if "lora_" in n and _grad_max_abs(p) > 0
                    )
                    engram_parts.append(f"LoRA[{lora_grad}/{lora_total}]")
                # ---- end check ----

                # Clip each optimizer group separately so a LoRA grad spike
                # doesn't scale down Engram (or vice versa).
                mano_clip = [p for g in self.optimizer_mano.param_groups
                             for p in g["params"]
                             if p.grad is not None and not p.grad.is_sparse]
                if mano_clip:
                    torch.nn.utils.clip_grad_norm_(mano_clip, 1.0, foreach=False)

                engram_clip = [p for g in self.optimizer_engram.param_groups
                               for p in g["params"]
                               if p.grad is not None and not p.grad.is_sparse]
                if engram_clip:
                    torch.nn.utils.clip_grad_norm_(engram_clip, 1.0, foreach=False)

                self.optimizer_engram.step()
                self.optimizer_mano.step()

                lv = loss.item()
                n_real = (labels != -100).sum().item()
                epoch_loss_sum += lv * n_real
                epoch_tokens += n_real
                n_steps += 1
                mode_loss_sums[mode] += lv
                mode_loss_counts[mode] += 1

                # Print AFTER optimizer.step — keeps the line ordered with loss
                if step % 100 == 0:
                    print(f"  Step {step:4d}/{len(loader)} | loss={loss.item():.4f} | "
                          f"mode={mode} | " + " ".join(engram_parts))

            avg_loss = epoch_loss_sum / max(epoch_tokens, 1)
            try:
                ppl = math.exp(avg_loss)
            except Exception:
                ppl = float("inf")

            self.scheduler.step(ppl)
            lr = self.optimizer_mano.param_groups[0]["lr"]
            ts = datetime.datetime.now().strftime("%H:%M")

            print(f"[{ts}] Epoch {epoch:3d} | loss={avg_loss:.4f} | "
                  f"ppl={ppl:.3f} | best={best_ppl:.3f} | lr={lr:.2e}")
            mode_avg = {
                m: (mode_loss_sums[m] / max(mode_loss_counts[m], 1))
                for m in mode_loss_sums
            }
            active_modes = [m for m in mode_avg if mode_loss_counts[m] > 0]
            parts = [f"{m}={mode_avg[m]:.4f}" for m in active_modes]
            print(f"           | {' '.join(parts)}")

            if ppl < best_ppl or epoch % 5 == 0:
                if ppl < best_ppl:
                    best_ppl = ppl
                self._save_checkpoint(epoch, best_ppl)

            gc.collect()
            torch.cuda.empty_cache()
