"""Training loop for MoC experiments."""

import os
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm

from router import compute_router_losses
from consolidation import PatternTracker, ConsolidationScheduler, run_consolidation

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False


def train(model, train_dataset, val_dataset, cfg: dict, device: torch.device):
    """
    Main training loop.

    Only router and s1_projection parameters are updated; base model is frozen.
    """
    # ------------------------------------------------------------------
    # Setup
    # ------------------------------------------------------------------
    os.makedirs(cfg["checkpoint_dir"], exist_ok=True)

    use_wandb = WANDB_AVAILABLE and cfg.get("wandb_project")
    if use_wandb:
        try:
            wandb.init(project=cfg["wandb_project"], config=cfg)
        except Exception:
            use_wandb = False
            print("[WandB] Not logged in — running offline, no WandB logging.")

    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg["batch_size"],
        shuffle=True,
        num_workers=0,
        pin_memory=False,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=cfg["batch_size"],
        shuffle=False,
        num_workers=0,
    )

    optimizer = AdamW(
        list(model.trainable_parameters()),
        lr=cfg["lr"],
        weight_decay=cfg["weight_decay"],
    )
    total_steps = len(train_loader) * cfg["n_epochs"] // cfg["grad_accum_steps"]
    scheduler = CosineAnnealingLR(optimizer, T_max=total_steps)

    # ------------------------------------------------------------------
    # Pattern trackers for consolidation (one per layer)
    # ------------------------------------------------------------------
    trackers: list[PatternTracker] = []
    consolidation_scheduler = None
    if cfg.get("consolidation_enabled", False):
        trackers = [
            PatternTracker(
                emb_dim=model.emb_dim,
                n_planes=cfg.get("lsh_n_planes", 32),
                table_size=cfg.get("lsh_table_size", 8192),
            )
            for _ in range(model.n_layers)
        ]
        consolidation_scheduler = ConsolidationScheduler(cfg)

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------
    global_step = 0
    best_val_loss = float("inf")

    for epoch in range(cfg["n_epochs"]):
        model.train()
        epoch_task_loss = 0.0
        epoch_router_loss = 0.0
        n_batches = 0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{cfg['n_epochs']}")
        optimizer.zero_grad()

        for batch_idx, batch in enumerate(pbar):
            input_ids      = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels         = batch["labels"].to(device)

            # Forward with routing info
            logits, routing_info = model(
                input_ids,
                attention_mask=attention_mask,
                return_routing_info=True,
            )

            # Task loss: next-token prediction (ignore padding / -100 labels)
            task_loss = F.cross_entropy(
                logits[:, :-1].float().reshape(-1, logits.shape[-1]),
                labels[:, 1:].reshape(-1),
                ignore_index=-100,
            )

            # Router auxiliary loss
            router_loss = compute_router_losses(routing_info, cfg)

            total_loss = (task_loss + router_loss) / cfg["grad_accum_steps"]
            total_loss.backward()

            epoch_task_loss   += task_loss.item()
            epoch_router_loss += router_loss.item()
            n_batches         += 1

            # Gradient accumulation
            if (batch_idx + 1) % cfg["grad_accum_steps"] == 0:
                torch.nn.utils.clip_grad_norm_(
                    list(model.trainable_parameters()),
                    cfg["max_grad_norm"],
                )
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                global_step += 1

                # Update pattern trackers
                if cfg.get("consolidation_enabled", False):
                    _update_trackers(trackers, routing_info)

                    # Consolidation trigger
                    if consolidation_scheduler.should_consolidate(
                        global_step, trackers
                    ):
                        print(f"\n[Step {global_step}] Running consolidation…")
                        report = run_consolidation(
                            model=model,
                            trackers=trackers,
                            cfg=cfg,
                            step=global_step,
                            device=device,
                        )
                        consolidation_scheduler.record(global_step)
                        _log_consolidation(report, global_step, use_wandb)

                # Logging
                if global_step % 50 == 0:
                    routing_stats = model.routing_summary(routing_info)
                    log_dict = {
                        "train/task_loss":    task_loss.item(),
                        "train/router_loss":  router_loss.item(),
                        "train/total_loss":   (task_loss + router_loss).item(),
                        "train/mean_s2_ratio": routing_stats["mean_s2_ratio"],
                        "train/step":         global_step,
                    }
                    if use_wandb:
                        wandb.log(log_dict, step=global_step)
                    pbar.set_postfix({
                        "loss": f"{task_loss.item():.4f}",
                        "s2%": f"{routing_stats['mean_s2_ratio']*100:.1f}",
                    })

                # Evaluation
                if global_step % cfg["eval_interval"] == 0:
                    val_metrics = evaluate(model, val_loader, cfg, device)
                    val_metrics["step"] = global_step
                    print(f"\n[Eval step {global_step}] {val_metrics}")
                    if use_wandb:
                        wandb.log({f"val/{k}": v for k, v in val_metrics.items()},
                                  step=global_step)
                    if val_metrics["loss"] < best_val_loss:
                        best_val_loss = val_metrics["loss"]
                        _save_checkpoint(model, optimizer, global_step, cfg, tag="best")

                # Periodic checkpoint
                if global_step % cfg["save_interval"] == 0:
                    _save_checkpoint(model, optimizer, global_step, cfg, tag=f"step{global_step}")

        avg_task   = epoch_task_loss   / max(n_batches, 1)
        avg_router = epoch_router_loss / max(n_batches, 1)
        print(f"Epoch {epoch+1} done | task_loss={avg_task:.4f} router_loss={avg_router:.4f}")

    if use_wandb:
        wandb.finish()

    return global_step


# ──────────────────────────────────────────────────────────────────────────────
# Evaluation
# ──────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def evaluate(model, val_loader, cfg: dict, device: torch.device) -> dict:
    model.eval()
    total_loss = 0.0
    total_tokens = 0
    s2_ratios_by_diff: dict[int, list] = {1: [], 2: [], 3: []}

    for batch in val_loader:
        input_ids      = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels         = batch["labels"].to(device)
        difficulties   = batch["difficulty"]

        logits, routing_info = model(
            input_ids, attention_mask=attention_mask, return_routing_info=True
        )

        loss = F.cross_entropy(
            logits[:, :-1].float().reshape(-1, logits.shape[-1]),
            labels[:, 1:].reshape(-1),
            ignore_index=-100,
            reduction="sum",
        )
        n_valid = (labels[:, 1:] != -100).sum().item()
        total_loss   += loss.item()
        total_tokens += max(n_valid, 1)

        # Per-difficulty S2 ratio
        mean_s2 = sum(info["s2_mask"].float().mean().item() for info in routing_info)
        mean_s2 /= len(routing_info)
        for b_idx, diff in enumerate(difficulties):
            s2_ratios_by_diff[diff.item()].append(mean_s2)

    avg_loss = total_loss / max(total_tokens, 1)
    metrics = {"loss": avg_loss, "perplexity": torch.exp(torch.tensor(avg_loss)).item()}
    for diff, vals in s2_ratios_by_diff.items():
        if vals:
            metrics[f"s2_ratio_diff{diff}"] = sum(vals) / len(vals)

    model.train()
    return metrics


# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────

def _update_trackers(trackers, routing_info):
    for info in routing_info:
        idx = info["layer_idx"]
        if idx < len(trackers):
            trackers[idx].update(
                embeddings=info["input_hidden"],
                s2_mask=info["s2_mask"],
            )


def _log_consolidation(report, step, use_wandb):
    n_layers = len(report["layers"])
    print(f"  Consolidation @ step {step}: {n_layers} layers consolidated")
    for lr in report["layers"]:
        print(f"    layer {lr['layer_idx']}: {lr['n_candidates']} patterns, "
              f"distill_loss={lr['distill_loss']:.4f}")
    if use_wandb:
        import wandb
        wandb.log({
            "consolidation/n_layers": n_layers,
            "consolidation/total_candidates": sum(l["n_candidates"] for l in report["layers"]),
            "consolidation/mean_distill_loss": (
                sum(l["distill_loss"] for l in report["layers"]) / max(n_layers, 1)
            ),
        }, step=step)


def _save_checkpoint(model, optimizer, step, cfg, tag="latest"):
    path = os.path.join(cfg["checkpoint_dir"], f"moc_{tag}.pt")
    torch.save({
        "step": step,
        "routers": model.routers.state_dict(),
        "s1_projections": model.s1_projections.state_dict(),
        "optimizer": optimizer.state_dict(),
        "cfg": cfg,
    }, path)
    print(f"  Checkpoint saved → {path}")


def load_checkpoint(model, optimizer, path: str) -> int:
    ckpt = torch.load(path, map_location="cpu")
    model.routers.load_state_dict(ckpt["routers"])
    model.s1_projections.load_state_dict(ckpt["s1_projections"])
    if optimizer and "optimizer" in ckpt:
        optimizer.load_state_dict(ckpt["optimizer"])
    print(f"Loaded checkpoint from {path} (step {ckpt['step']})")
    return ckpt["step"]
