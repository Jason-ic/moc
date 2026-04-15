"""
Main entry point for MoC experiments.

Usage:
    # Phase 1: basic routing, no consolidation
    python run_experiment.py --phase 1

    # Phase 4: with online consolidation
    python run_experiment.py --phase 4

    # Evaluate a checkpoint
    python run_experiment.py --eval --checkpoint checkpoints/moc_best.pt

    # Quick smoke test (small data subset)
    python run_experiment.py --smoke-test
"""

import argparse
import os
import torch

from configs import MOC_CONFIG
from moc_model import MoCWrapper
from data import TriviaQADataset
from train import train, load_checkpoint
from eval import per_difficulty_metrics, consolidation_gap, flop_savings
from viz import (
    plot_routing_heatmap,
    plot_s2_ratio_over_time,
    plot_consolidation_gap_over_time,
)


def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def build_config(args, base_cfg: dict) -> dict:
    cfg = dict(base_cfg)

    if args.phase >= 4:
        cfg["consolidation_enabled"] = True

    if args.smoke_test:
        cfg["n_epochs"]     = 1
        cfg["batch_size"]   = 2
        cfg["eval_interval"] = 20
        cfg["save_interval"] = 50

    if args.batch_size:
        cfg["batch_size"] = args.batch_size
    if args.lr:
        cfg["lr"] = args.lr

    return cfg


def run_smoke_test(model, cfg, device):
    """Minimal forward pass test — verifies the whole pipeline runs."""
    print("\n=== Smoke test ===")
    tokenizer = model.tokenizer
    text = "Question: What is the capital of France?\nAnswer: Paris"
    enc  = tokenizer(text, return_tensors="pt", max_length=64,
                     truncation=True, padding="max_length")
    input_ids      = enc["input_ids"].to(device)
    attention_mask = enc["attention_mask"].to(device)

    model.eval()
    logits, routing_info = model(input_ids, attention_mask, return_routing_info=True)

    print(f"  logits shape    : {logits.shape}")
    print(f"  n_layers routed : {len(routing_info)}")
    stats = model.routing_summary(routing_info)
    print(f"  mean S2 ratio   : {stats['mean_s2_ratio']:.3f}")
    flops = flop_savings(routing_info, cfg)
    print(f"  FLOP savings    : {flops['flop_savings_pct']:.1f}%")

    from eval import flop_savings as fs
    print("Smoke test PASSED ✓")
    return routing_info


def main():
    parser = argparse.ArgumentParser(description="Mixture of Cognition experiment")
    parser.add_argument("--phase",      type=int, default=1,
                        help="Experiment phase (1-5)")
    parser.add_argument("--eval",       action="store_true",
                        help="Evaluation-only mode")
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="Path to checkpoint (.pt)")
    parser.add_argument("--smoke-test", action="store_true",
                        help="Quick sanity check on a single example")
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--lr",         type=float, default=None)
    parser.add_argument("--max-samples", type=int, default=None,
                        help="Limit dataset size (for fast iteration)")
    args = parser.parse_args()

    device = get_device()
    print(f"Device: {device}")

    cfg = build_config(args, MOC_CONFIG)

    # ------------------------------------------------------------------
    # Build model
    # ------------------------------------------------------------------
    print(f"\nLoading {cfg['model_name']} …")
    model = MoCWrapper(cfg, device)
    print(f"  Base model params     : {model.num_base_params()/1e6:.1f} M")
    print(f"  Trainable MoC params  : {model.num_trainable_params()/1e6:.1f} M")

    # ------------------------------------------------------------------
    # Load checkpoint if provided
    # ------------------------------------------------------------------
    if args.checkpoint:
        load_checkpoint(model, None, args.checkpoint)

    # ------------------------------------------------------------------
    # Smoke test
    # ------------------------------------------------------------------
    if args.smoke_test:
        routing_info = run_smoke_test(model, cfg, device)
        os.makedirs("figures", exist_ok=True)
        plot_routing_heatmap(routing_info, "figures/smoke_heatmap.png",
                             title="Routing heatmap (smoke test)")
        return

    # ------------------------------------------------------------------
    # Load data
    # ------------------------------------------------------------------
    print("\nLoading TriviaQA …")
    train_ds = TriviaQADataset(
        split="train",
        tokenizer=model.tokenizer,
        max_len=cfg["max_len"],
        repeat_fraction=cfg["repeat_fraction"],
        n_repeat_patterns=cfg["n_repeat_patterns"],
        max_samples=args.max_samples,
    )
    val_ds = TriviaQADataset(
        split="validation",
        tokenizer=model.tokenizer,
        max_len=cfg["max_len"],
        repeat_fraction=0.0,    # no repeats in validation
        max_samples=min(2000, args.max_samples or 2000),
    )

    # ------------------------------------------------------------------
    # Eval-only mode
    # ------------------------------------------------------------------
    if args.eval:
        print("\nRunning evaluation …")
        metrics = per_difficulty_metrics(model, val_ds, cfg, device)
        print("\nPer-difficulty metrics:")
        for k, v in metrics.items():
            print(f"  {k}: {v:.4f}")
        return

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------
    print(f"\nStarting training (phase {args.phase}) …")
    os.makedirs("figures", exist_ok=True)

    train(model, train_ds, val_ds, cfg, device)

    # ------------------------------------------------------------------
    # Post-training evaluation
    # ------------------------------------------------------------------
    print("\nFinal evaluation …")
    metrics = per_difficulty_metrics(model, val_ds, cfg, device)
    print("\nFinal per-difficulty metrics:")
    for k, v in metrics.items():
        print(f"  {k}: {v:.4f}")

    # Save final checkpoint
    torch.save({
        "routers":         model.routers.state_dict(),
        "s1_projections":  model.s1_projections.state_dict(),
        "cfg":             cfg,
    }, os.path.join(cfg["checkpoint_dir"], "moc_final.pt"))
    print("Done.")


if __name__ == "__main__":
    main()
