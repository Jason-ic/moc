"""MoC Router: per-layer metacognitive routing module."""

import torch
import torch.nn as nn
import torch.nn.functional as F


class MoCRouter(nn.Module):
    """
    Per-layer router.  For each token, produces a scalar score indicating
    how much that token benefits from the full S2 (deep) path.

    High score  →  route to S2 (full frozen transformer layer)
    Low score   →  route to S1 (lightweight skip projection)
    """

    def __init__(self, emb_dim: int, router_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(emb_dim, router_dim),
            nn.GELU(),
            nn.Linear(router_dim, 1),
        )
        # Initialise with small weights so routing starts near 0.5 probability
        for m in self.net.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=0.02)
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, seq_len, emb_dim)  –  float32 hidden states

        Returns:
            router_logits: (batch, seq_len)  –  raw logits (before sigmoid)
        """
        return self.net(x).squeeze(-1)


class MetacognitiveRouter(nn.Module):
    """
    Enhanced router that also predicts its own confidence.
    Used in Phase 3 when we add supervised difficulty prediction.

    Outputs:
        s2_logit   (batch, seq_len)  –  how much this token needs S2
        confidence (batch, seq_len)  –  router's self-confidence in [0, 1]
    """

    def __init__(self, emb_dim: int, router_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(emb_dim, router_dim),
            nn.GELU(),
            nn.Linear(router_dim, router_dim),
            nn.GELU(),
            nn.Linear(router_dim, 2),  # [s2_score, confidence_score]
        )
        for m in self.net.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=0.02)
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor):
        out = self.net(x)                         # (B, T, 2)
        s2_logit = out[..., 0]                    # (B, T)
        confidence = torch.sigmoid(out[..., 1])   # (B, T) ∈ (0, 1)
        return s2_logit, confidence


def compute_router_losses(routing_info_list: list, cfg: dict) -> torch.Tensor:
    """
    Auxiliary losses to prevent router collapse.

    Z-loss:      penalises large logits  →  prevents overconfident routing
    Balance loss: penalises deviation from capacity_factor  →  prevents all-S1 / all-S2
    """
    z_loss_total = 0.0
    balance_loss_total = 0.0
    n = len(routing_info_list)

    for info in routing_info_list:
        logits = info["router_logits"]   # (B, T)

        # Z-loss (from ST-MoE): log(exp(logit))^2 ≈ logit^2 for scalar case
        z_loss = torch.mean(logits ** 2)
        z_loss_total = z_loss_total + z_loss

        # Balance loss: keep S2 fraction near capacity_factor
        s2_frac = info["s2_mask"].mean(dim=-1)          # (B,)
        target = cfg["capacity_factor"]
        balance_loss = torch.mean((s2_frac - target) ** 2)
        balance_loss_total = balance_loss_total + balance_loss

    z_loss_total = z_loss_total / n
    balance_loss_total = balance_loss_total / n

    return (cfg["z_loss_weight"] * z_loss_total +
            cfg["balance_loss_weight"] * balance_loss_total)


def compute_metacognitive_loss(routing_info_list: list) -> torch.Tensor:
    """
    Supervised loss for the metacognitive router (Phase 3).

    Trains the router to predict token-level difficulty, measured as the
    normalised L2 distance between S1 and S2 layer outputs.
    """
    total = 0.0
    n = len(routing_info_list)

    for info in routing_info_list:
        if "s2_logit" not in info or "s2_output" not in info:
            continue

        s2_out = info["s2_output"]   # (B, T, D)  detached teacher signal
        s1_out = info["s1_output"]   # (B, T, D)  detached

        # Difficulty oracle: normalised L2 distance per token
        diff = torch.norm(s2_out.float() - s1_out.float(), dim=-1)  # (B, T)
        d_min = diff.min(dim=-1, keepdim=True).values
        d_max = diff.max(dim=-1, keepdim=True).values
        difficulty = (diff - d_min) / (d_max - d_min + 1e-8)        # (B, T) ∈ [0, 1]

        # Binary target: tokens above median difficulty should use S2
        target = (difficulty > 0.5).float()

        s2_logit = info["s2_logit"]  # (B, T)
        pred_loss = F.binary_cross_entropy_with_logits(s2_logit, target)

        total = total + pred_loss

    return total / max(n, 1)
