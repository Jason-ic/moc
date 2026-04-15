"""Evaluation utilities: FLOP counting, per-difficulty metrics, consolidation gap."""

import torch
from torch.utils.data import DataLoader
from collections import defaultdict


def flop_savings(routing_info: list, cfg: dict) -> dict:
    """
    Estimate FLOP savings based on how many tokens skip S2 layers.

    FLOPs per token per layer (rough approximation):
      S2 (full attention + FFN): 4*D^2 + 8*D^2 = 12*D^2  (D = emb_dim)
      S1 (single linear):         D^2

    Returns fraction of FLOPs saved relative to a vanilla model (all-S2).
    """
    D = cfg.get("emb_dim", 1024)
    s2_flops = 12 * D * D
    s1_flops =  1 * D * D

    total_flops   = 0
    total_tokens  = 0
    for info in routing_info:
        mask = info["s2_mask"]           # (B, T)
        n_s2 = mask.sum().item()
        n_s1 = mask.numel() - n_s2
        total_flops  += n_s2 * s2_flops + n_s1 * s1_flops
        total_tokens += mask.numel()

    baseline_flops = len(routing_info) * s2_flops   # all layers, all tokens
    avg_flops = total_flops / max(total_tokens, 1)
    savings   = 1.0 - avg_flops / baseline_flops

    return {
        "avg_flops_per_token": avg_flops,
        "baseline_flops":      baseline_flops,
        "flop_savings_pct":    savings * 100,
    }


@torch.no_grad()
def per_difficulty_metrics(
    model,
    val_dataset,
    cfg: dict,
    device: torch.device,
    batch_size: int = 4,
) -> dict:
    """
    Compute accuracy (next-token prediction), S2 ratio, and FLOP savings
    separately for each difficulty level.
    """
    model.eval()
    loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    stats = defaultdict(lambda: {"correct": 0, "total": 0, "s2_ratios": []})

    for batch in loader:
        input_ids      = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels         = batch["labels"].to(device)
        difficulties   = batch["difficulty"]

        logits, routing_info = model(
            input_ids, attention_mask=attention_mask, return_routing_info=True
        )
        preds = logits[:, :-1].argmax(dim=-1)    # (B, T-1)
        tgts  = labels[:, 1:]                    # (B, T-1)

        # Mean S2 ratio across layers for this batch
        mean_s2 = sum(info["s2_mask"].float().mean().item() for info in routing_info)
        mean_s2 /= len(routing_info)

        for b_idx in range(input_ids.shape[0]):
            diff = difficulties[b_idx].item()
            valid = tgts[b_idx] != -100
            n_valid = valid.sum().item()
            if n_valid == 0:
                continue
            n_correct = (preds[b_idx][valid] == tgts[b_idx][valid]).sum().item()
            stats[diff]["correct"] += n_correct
            stats[diff]["total"]   += n_valid
            stats[diff]["s2_ratios"].append(mean_s2)

    results = {}
    for diff in sorted(stats.keys()):
        d = stats[diff]
        acc = d["correct"] / max(d["total"], 1)
        s2r = sum(d["s2_ratios"]) / max(len(d["s2_ratios"]), 1)
        results[f"accuracy_diff{diff}"]  = acc
        results[f"s2_ratio_diff{diff}"]  = s2r

    return results


@torch.no_grad()
def consolidation_gap(
    model,
    repeated_inputs: torch.Tensor,     # (N, T)  inputs seen many times
    novel_inputs:    torch.Tensor,     # (N, T)  inputs never seen before
    cfg: dict,
    device: torch.device,
) -> dict:
    """
    The critical consolidation test:
    do repeated patterns use fewer FLOPs (lower S2 ratio) than novel patterns?

    A positive gap → consolidation is working.
    """
    model.eval()

    def mean_s2(inputs):
        logits, info = model(inputs.to(device), return_routing_info=True)
        return sum(i["s2_mask"].float().mean().item() for i in info) / len(info)

    rep_s2   = mean_s2(repeated_inputs)
    novel_s2 = mean_s2(novel_inputs)
    gap      = novel_s2 - rep_s2       # positive = repeated cheaper than novel

    return {
        "repeated_s2_ratio": rep_s2,
        "novel_s2_ratio":    novel_s2,
        "consolidation_gap": gap,
    }


@torch.no_grad()
def forgetting_rate(
    model,
    reference_inputs:   torch.Tensor,  # patterns consolidated earlier
    reference_labels:   torch.Tensor,
    cfg: dict,
    device: torch.device,
) -> float:
    """
    Measure accuracy on already-consolidated patterns.
    Should stay stable (near 1.0) over training time.
    """
    model.eval()
    import torch.nn.functional as F

    logits = model(reference_inputs.to(device))
    if isinstance(logits, tuple):
        logits = logits[0]
    preds = logits[:, :-1].argmax(-1)
    tgts  = reference_labels[:, 1:].to(device)
    valid = tgts != -100
    correct = (preds[valid] == tgts[valid]).float().mean().item()
    return correct
