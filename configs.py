"""All hyperparameter configurations for MoC experiments."""

MOC_CONFIG = {
    # Model
    # Use Qwen3-0.6B: standard attention (not DeltaNet hybrid),
    # supported by transformers 4.51 + PyTorch 2.2 on macOS Intel
    "model_name": "Qwen/Qwen3-0.6B",
    "model_dtype": "float16",       # fp16 for memory; trainable modules match dtype

    # Router & S1 path
    "router_dim": 128,               # Hidden dim of router MLP
    "capacity_factor": 0.5,          # Fraction of tokens routed to S2

    # Auxiliary losses
    "z_loss_weight": 0.01,           # Stabilise router logits
    "balance_loss_weight": 0.01,     # Encourage balanced routing

    # Training
    "lr": 3e-4,
    "weight_decay": 0.01,
    "batch_size": 4,                 # Small for local 4 GB GPU
    "grad_accum_steps": 4,           # Effective batch = 16
    "max_len": 128,
    "n_epochs": 3,
    "warmup_steps": 100,
    "max_grad_norm": 1.0,
    "eval_interval": 200,            # Steps between evaluations
    "save_interval": 500,

    # Consolidation (Phase 4)
    "consolidation_enabled": False,
    "consolidation_check_interval": 500,
    "consolidation_cooldown": 1000,
    "consolidation_min_freq": 50,
    "consolidation_min_s2_ratio": 0.8,
    "consolidation_min_candidates": 5,
    "consolidation_batch_size": 20,
    "consolidation_distill_steps": 100,
    "consolidation_distill_lr": 1e-3,
    "consolidation_router_steps": 50,
    "consolidation_router_lr": 1e-3,
    "ewc_lambda": 100.0,

    # Pattern tracker
    "lsh_n_planes": 32,
    "lsh_table_size": 8192,
    "lsh_emb_ema_alpha": 0.01,

    # Data
    "dataset_name": "trivia_qa",
    "dataset_config": "unfiltered.nocontext",
    "repeat_fraction": 0.3,          # Fraction of train set that are repeated patterns
    "n_repeat_patterns": 50,         # Number of unique patterns to repeat

    # Logging
    "wandb_project": "mixture-of-cognition",
    "checkpoint_dir": "checkpoints",
}
