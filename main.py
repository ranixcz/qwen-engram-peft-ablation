"""
main.py — Engram Training, Phase 1 (DeepSeek-mirroring architecture)
====================================================================
Entry point. Config + model/dataset/trainer wiring.

Phase 1 goal: verify that hash-based Engram gives decreasing perplexity
on open-source causal-LM data. Legacy components (credit engine, dynamic
retrieval credit, adaptive curriculum sampler, scheduled sampling) are
kept as dead code so they can be re-wired later without rewriting this
file.
"""
import os
import torch
from peft import LoraConfig, get_peft_model, TaskType
from transformers import AutoTokenizer, AutoModelForCausalLM

# ---- Phase 1 active imports ----
from dataset import EngramFocusedDataset
from model import EngramEnhancedModel
from trainer import Trainer

# ---- LEGACY (dead code) — kept so re-wiring is a 2-line change later ----
from losses import CreditEngine, DynamicRetrievalCredit
from memory_utils import (
    RoutingStatsTracker,
    AdaptiveCurriculumSampler,
    EmbeddingFactMasker,
)

def main():
    # Reproducibility: seed all RNGs
    import random as _random
    import numpy as _np
    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)
    _random.seed(42)
    _np.random.seed(42)

    config = Config()

class Config:
    # =================================================================
    # Model & tokenizer
    # =================================================================
    model_name = "Qwen/Qwen2.5-1.5B"
    tokenizer_name = "Qwen/Qwen2.5-1.5B"
    hidden_size = 1536
    vocab_size = 151936
    pad_token_id = 0
    eos_token_id = 151643

    # =================================================================
    # NEW: Engram (DeepSeek-mirroring) — Phase 1
    # These are the only fields the new model uses.
    # =================================================================
    memory_layers_read = [1, 14]                    # read hook layers
    engram_vocab_size = [151936, 151936]   # per n-gram order (2-gram, 3-gram)
    max_ngram_size = 3
    n_embed_per_ngram = 1280
    n_head_per_ngram = 8
    engram_seed = 0

    # =================================================================
    # Training hyperparameters
    # =================================================================
    batch_size = 2
    learning_rate = 1e-5          # Engram modules
    base_lr = 5e-8                # unused in Phase 1 (base is frozen, only LoRA trains)
    epochs = int(os.environ.get("MAX_EPOCHS", 999999))
    checkpoint_dir = "/home/pc/tiny_distill_project/checkpoints"
    num_samples = 21000           # NEW: cap for Phase 1 dataset
    max_len = 512                 # was 256 — longer for causal LM

    # =================================================================
    # LoRA
    # =================================================================
    lora_r = 8
    lora_alpha = 16
    lora_dropout = 0.30
    lora_lr = 1e-5
    lora_freeze_epoch = 100

    # =================================================================
    # LEGACY (dead code) — not used in Phase 1.
    # Instantiated in main() and passed to Trainer, but Trainer ignores
    # them. Kept here so restoring the paired architecture is a matter
    # of un-commenting the training loop, not rebuilding config.
    # =================================================================
    # --- Old memory / SVD fields (unused by DeepSeek Engram) ---
    table_size = 2**22
    val_dim = 384
    key_dim = 64
    top_k = 8
    memory_layer_write = 27
    svd_rank = 512

    # --- Old data paths (unused by Phase 1 dataset) ---
    jsonl_path = "/home/pc/tiny_distill_project/episodic_data.jsonl"
    natural_data_paths = "/home/pc/tiny_distill_project/generated_conversations/episodic_persona.jsonl"

    # --- Stage 2 / output_proj warmup ---
    stage2_epoch = 0
    stage2_lr = 1e-6
    output_proj_unfreeze_epoch = 3

    # --- Loss weights ---
    fact_loss_weight = 1.0
    lang_loss_weight = 1.0
    distill_loss_weight = 0.5

    # --- Credit engine ---
    credit_max_r = 2.0
    credit_min_r = 1.0
    credit_hon = -0.8
    credit_arr_base = -1.6
    credit_arr_max = -4.8
    credit_esc = -0.15
    max_arrogant = 5000
    ageout_batches = 1000
    replay_prob = 0.2

    # --- Dynamic retrieval credit ---
    retrieval_target_accuracy = 0.75
    retrieval_base_lambda = 0.10
    retrieval_max_boost = 0.25
    retrieval_ema_alpha = 0.05

    # --- Adaptive curriculum sampler ---
    sampler_low_threshold = 0.80
    sampler_struggle_threshold = 0.35
    sampler_ema_alpha = 0.30
    sampler_low_priority_reduction = 0.20
    sampler_struggling_boost = 0.10

    # --- Scheduled sampling / paired retrieval-gen ---
    scheduled_sampling_epochs = 15
    use_gen_retrieval = True
    gen_k_tokens = 5

    # --- Self-test ---
    self_test_max_prompts = 20
    self_test_persona_file = "seed_questions/seed_personas.jsonl"
    self_test_knowledge_file = "seed_questions/seed_knowledge.jsonl"

    # =================================================================
    # Misc
    # =================================================================
    device = "cuda" if torch.cuda.is_available() else "cpu"
    use_amp = True
    gradient_checkpointing = False
    debug_route = False


def main():
    config = Config()

    # ---- Tokenizer ----
    tokenizer = AutoTokenizer.from_pretrained(config.tokenizer_name)
    tokenizer.pad_token = tokenizer.eos_token
    config.pad_token_id = tokenizer.pad_token_id
    config.eos_token_id = tokenizer.eos_token_id

    # ---- Dataset ----
    # Phase 1: plain causal-LM dataset. Assumes dataset.py has been
    # replaced with the {input_ids, labels}-only version.
    dataset = EngramFocusedDataset(
        tokenizer=tokenizer,
        max_len=config.max_len,
        num_samples=config.num_samples,
    )

    # ---- Base model ----
    base_model = AutoModelForCausalLM.from_pretrained(
        config.model_name,
        torch_dtype=torch.bfloat16 if config.device == "cuda" else torch.float32,
        attn_implementation="sdpa",
    )
    config.vocab_size = base_model.config.vocab_size
    base_model.config.use_cache = False

    # ---- LoRA ----
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=config.lora_r,
        lora_alpha=config.lora_alpha,
        lora_dropout=config.lora_dropout,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"],
        bias="none",
    )
    base_model = get_peft_model(base_model, lora_config)

    if config.gradient_checkpointing:
        base_model.gradient_checkpointing_enable()
        base_model.enable_input_require_grads()

    # ---- Engram-enhanced model (DeepSeek-mirroring) ----
    model = EngramEnhancedModel(
        base_model=base_model,
        hidden_size=config.hidden_size,
        engram_vocab_size=config.engram_vocab_size,
        max_ngram_size=config.max_ngram_size,
        n_embed_per_ngram=config.n_embed_per_ngram,
        n_head_per_ngram=config.n_head_per_ngram,
        read_layers=config.memory_layers_read,
        tokenizer_name_or_path=config.tokenizer_name,
        pad_id=config.pad_token_id,
        seed=config.engram_seed,
    )

    total_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"  Trainable (LoRA+Engram): {total_trainable:,} / {total_params:,} "
          f"({100.0 * total_trainable / total_params:.3f}%)")

    # ---- LEGACY (dead code) ----
    # Instantiated so imports stay valid and re-wiring is just a matter
    # of passing them into the training loop. Trainer stores them but
    # does not use them in Phase 1.
    credit_engine = CreditEngine(
        vs=config.vocab_size,
        eos_id=config.eos_token_id,
        pad_id=config.pad_token_id,
        max_r=config.credit_max_r,
        min_r=config.credit_min_r,
        hon=config.credit_hon,
        arr_base=config.credit_arr_base,
        arr_max=config.credit_arr_max,
        esc=config.credit_esc,
        max_arrogant=config.max_arrogant,
        ageout_batches=config.ageout_batches,
    )
    retrieval_tracker = DynamicRetrievalCredit(
        target_accuracy=config.retrieval_target_accuracy,
        base_lambda=config.retrieval_base_lambda,
        max_boost=config.retrieval_max_boost,
        ema_alpha=config.retrieval_ema_alpha,
    )
    sampler = AdaptiveCurriculumSampler(
        low_threshold=config.sampler_low_threshold,
        struggle_threshold=config.sampler_struggle_threshold,
        ema_alpha=config.sampler_ema_alpha,
        low_priority_reduction=config.sampler_low_priority_reduction,
        struggling_boost=config.sampler_struggling_boost,
    )
    routing_tracker = RoutingStatsTracker()
    fact_masker = EmbeddingFactMasker(hidden_dim=config.hidden_size)

    # ---- Trainer ----
    trainer = Trainer(
        model=model,
        dataset=dataset,
        tokenizer=tokenizer,
        config=config,
        # Legacy args — Trainer stores them but does not use them in Phase 1.
        credit_engine=credit_engine,
        retrieval_tracker=retrieval_tracker,
        sampler=sampler,
        routing_tracker=routing_tracker,
        fact_masker=fact_masker,
    )
    trainer.run()


if __name__ == "__main__":
    main()
