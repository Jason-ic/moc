"""
Online S2→S1 Consolidation Mechanism  (Phase 4 – Novel Contribution)

Components
----------
PatternTracker        : Uses LSH to track which embedding clusters consistently
                        activate S2.  No raw inputs stored – only hash counts
                        and EMA representative embeddings.
ConsolidationScheduler: Decides when to trigger a consolidation round.
consolidate_layer     : Distils S2 knowledge into S1 for a single layer,
                        with EWC regularisation to prevent forgetting.
update_router_post_consolidation
                      : Steers the router to send consolidated patterns to S1.
run_consolidation     : Orchestrates a full consolidation pass across all layers.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# ──────────────────────────────────────────────────────────────────────────────
# Pattern Tracker
# ──────────────────────────────────────────────────────────────────────────────

class PatternTracker:
    """
    Locality-Sensitive Hashing tracker.

    For each token embedding seen during training, we:
    1. Compute a binary hash using random projection planes.
    2. Increment the S2 frequency counter for that hash bucket.
    3. Update an EMA representative embedding for that bucket.

    After enough observations, buckets that consistently go to S2
    become consolidation candidates.
    """

    def __init__(self, emb_dim: int, n_planes: int = 32, table_size: int = 8192):
        # Fixed (non-learned) random projection planes
        planes = torch.randn(n_planes, emb_dim)
        planes = planes / planes.norm(dim=-1, keepdim=True)
        self.register_planes(planes)

        self.s2_frequency  = torch.zeros(table_size, dtype=torch.long)
        self.total_frequency = torch.zeros(table_size, dtype=torch.long)
        self.rep_embeddings: dict[int, torch.Tensor] = {}   # bucket → EMA embed

        self.n_planes    = n_planes
        self.table_size  = table_size
        self.ema_alpha   = 0.01

    def register_planes(self, planes: torch.Tensor):
        # Store on CPU always (tracking is offline)
        self._planes = planes.cpu()

    def _hash(self, embeddings: torch.Tensor) -> torch.Tensor:
        """
        embeddings: (N, D)  float32
        returns:    (N,)    long  ∈ [0, table_size)
        """
        emb_cpu = embeddings.cpu().float()
        proj = emb_cpu @ self._planes.T          # (N, n_planes)
        bits = (proj > 0).long()                  # (N, n_planes)  binary
        powers = 2 ** torch.arange(self.n_planes, dtype=torch.long)
        hashes = (bits * powers).sum(dim=-1) % self.table_size
        return hashes

    @torch.no_grad()
    def update(self, embeddings: torch.Tensor, s2_mask: torch.Tensor):
        """
        embeddings : (B, T, D)  – input hidden states to this layer
        s2_mask    : (B, T)     – 1 where token took S2 path
        """
        flat_emb  = embeddings.reshape(-1, embeddings.shape[-1])   # (B*T, D)
        flat_mask = s2_mask.reshape(-1).cpu()                      # (B*T,)

        hashes = self._hash(flat_emb)   # (B*T,)

        for i in range(len(hashes)):
            h = hashes[i].item()
            self.total_frequency[h] += 1
            if flat_mask[i].item() > 0.5:
                self.s2_frequency[h] += 1

            # EMA representative embedding
            emb_i = flat_emb[i].cpu().float()
            if h not in self.rep_embeddings:
                self.rep_embeddings[h] = emb_i.clone()
            else:
                self.rep_embeddings[h] = (
                    (1.0 - self.ema_alpha) * self.rep_embeddings[h]
                    + self.ema_alpha * emb_i
                )

    def get_candidates(
        self,
        min_frequency: int = 50,
        min_s2_ratio: float = 0.8,
    ) -> list[dict]:
        """
        Return buckets that:
        • have been seen at least min_frequency times
        • were routed to S2 at least min_s2_ratio of the time

        Returns list of dicts sorted by frequency (most common first).
        """
        candidates = []
        for h in range(self.table_size):
            freq = self.total_frequency[h].item()
            if freq < min_frequency:
                continue
            s2_ratio = self.s2_frequency[h].item() / freq
            if s2_ratio < min_s2_ratio:
                continue
            if h not in self.rep_embeddings:
                continue
            candidates.append({
                "hash_idx":  h,
                "frequency": freq,
                "s2_ratio":  s2_ratio,
                "embedding": self.rep_embeddings[h],
            })
        candidates.sort(key=lambda x: x["frequency"], reverse=True)
        return candidates

    def decay_candidates(self, candidates: list[dict]):
        """After consolidation, reset S2 count and halve total count."""
        for c in candidates:
            h = c["hash_idx"]
            self.s2_frequency[h]  = 0
            self.total_frequency[h] = self.total_frequency[h] // 2


# ──────────────────────────────────────────────────────────────────────────────
# Consolidation Scheduler
# ──────────────────────────────────────────────────────────────────────────────

class ConsolidationScheduler:
    """Decides when to trigger consolidation."""

    def __init__(self, cfg: dict):
        self.check_interval  = cfg.get("consolidation_check_interval", 500)
        self.cooldown        = cfg.get("consolidation_cooldown", 1000)
        self.min_candidates  = cfg.get("consolidation_min_candidates", 5)
        self.min_freq        = cfg.get("consolidation_min_freq", 50)
        self.min_s2_ratio    = cfg.get("consolidation_min_s2_ratio", 0.8)

        self._last_step      = -self.cooldown   # allow immediate first check
        self.n_consolidations = 0

    def should_consolidate(self, step: int, trackers: list) -> bool:
        if step % self.check_interval != 0:
            return False
        if step - self._last_step < self.cooldown:
            return False
        # Check if any tracker has enough candidates
        for tracker in trackers:
            cands = tracker.get_candidates(self.min_freq, self.min_s2_ratio)
            if len(cands) >= self.min_candidates:
                return True
        return False

    def record(self, step: int):
        self._last_step = step
        self.n_consolidations += 1


# ──────────────────────────────────────────────────────────────────────────────
# EWC Fisher Diagonal
# ──────────────────────────────────────────────────────────────────────────────

def compute_fisher_diagonal(s1_module: nn.Module, device: torch.device,
                             n_samples: int = 200) -> dict[str, torch.Tensor]:
    """
    Approximate the Fisher Information diagonal for an S1Projection module
    by sampling random inputs and accumulating squared gradients.

    Returns dict: param_name → importance weight tensor (same shape as param).
    """
    fisher = {n: torch.zeros_like(p) for n, p in s1_module.named_parameters()}
    s1_module.eval()
    emb_dim = s1_module.proj.in_features

    for _ in range(n_samples):
        x = torch.randn(1, 1, emb_dim, device=device)
        out = s1_module(x)
        loss = out.pow(2).sum()
        s1_module.zero_grad()
        loss.backward()
        for n, p in s1_module.named_parameters():
            if p.grad is not None:
                fisher[n] = fisher[n] + p.grad.detach() ** 2

    for n in fisher:
        fisher[n] = fisher[n] / n_samples
    return fisher


# ──────────────────────────────────────────────────────────────────────────────
# Consolidate a single layer
# ──────────────────────────────────────────────────────────────────────────────

def consolidate_layer(
    layer,          # frozen transformer layer (S2 path)
    s1_proj,        # S1Projection to be updated
    candidates: list[dict],
    cfg: dict,
    device: torch.device,
) -> float:
    """
    Distil S2 knowledge into S1 for the given set of candidate patterns.

    Returns the final distillation loss (float) for logging.
    """
    if not candidates:
        return 0.0

    emb_dim = s1_proj.proj.in_features

    # --- Build input tensor from representative embeddings ---
    emb_stack = torch.stack([c["embedding"] for c in candidates]).to(device)
    # Shape: (N_cand, D)  →  add batch & seq dims: (1, N_cand, D)
    target_input = emb_stack.unsqueeze(0).to(torch.float16 if next(layer.parameters()).dtype == torch.float16 else torch.float32)

    # --- Compute S2 teacher signal (frozen layer, no grad) ---
    layer.eval()
    with torch.no_grad():
        # We pass only hidden_states; the layer handles positional info internally
        # For representative embeddings used as "dummy" inputs this is an approximation.
        # The layer's self-attention has no sequence to attend to except itself,
        # so this gives an approximation of the transformation rather than exact S2.
        try:
            s2_out = layer(
                target_input,
                attention_mask=None,
                position_ids=torch.zeros(1, emb_stack.shape[0], dtype=torch.long, device=device),
            )
        except TypeError:
            # Some layer variants require position_embeddings
            s2_out = layer(target_input)
        if isinstance(s2_out, tuple):
            s2_out = s2_out[0]
    s2_out = s2_out.float().detach()   # (1, N_cand, D)

    # --- EWC: compute Fisher diagonal and save old params ---
    fisher = compute_fisher_diagonal(s1_proj, device)
    old_params = {n: p.clone().detach() for n, p in s1_proj.named_parameters()}

    # --- Distil into S1 ---
    s1_proj.train()
    optimizer = torch.optim.Adam(
        s1_proj.parameters(),
        lr=cfg.get("consolidation_distill_lr", 1e-3),
    )
    ewc_lambda = cfg.get("ewc_lambda", 100.0)
    n_steps = cfg.get("consolidation_distill_steps", 100)
    final_loss = 0.0

    for _ in range(n_steps):
        s1_out = s1_proj(target_input.float())                # (1, N_cand, D)

        distill_loss = F.mse_loss(s1_out, s2_out)

        # EWC penalty: anchor to old important parameters
        ewc_loss = sum(
            (fisher[n] * (p - old_params[n]) ** 2).sum()
            for n, p in s1_proj.named_parameters()
            if n in fisher
        )
        ewc_loss = ewc_lambda / 2.0 * ewc_loss

        loss = distill_loss + ewc_loss
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        final_loss = distill_loss.item()

    return final_loss


# ──────────────────────────────────────────────────────────────────────────────
# Update router after consolidation
# ──────────────────────────────────────────────────────────────────────────────

def update_router_post_consolidation(
    router: nn.Module,
    candidates: list[dict],
    cfg: dict,
    device: torch.device,
):
    """
    Fine-tune the router to send consolidated patterns to S1 (low score).
    """
    if not candidates:
        return

    emb_stack = torch.stack([c["embedding"] for c in candidates]).to(device).float()
    target_input = emb_stack.unsqueeze(0)   # (1, N_cand, D)
    target_logit = torch.zeros(1, emb_stack.shape[0], device=device)  # all → S1

    optimizer = torch.optim.Adam(
        router.parameters(),
        lr=cfg.get("consolidation_router_lr", 1e-3),
    )
    n_steps = cfg.get("consolidation_router_steps", 50)

    router.train()
    for _ in range(n_steps):
        logits = router(target_input)
        if isinstance(logits, tuple):
            logits = logits[0]          # MetacognitiveRouter returns (logit, conf)
        loss = F.binary_cross_entropy_with_logits(logits, target_logit)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()


# ──────────────────────────────────────────────────────────────────────────────
# Full consolidation pass
# ──────────────────────────────────────────────────────────────────────────────

def run_consolidation(
    model,          # MoCWrapper
    trackers: list[PatternTracker],
    cfg: dict,
    step: int,
    device: torch.device,
) -> dict:
    """
    Run a full consolidation pass: for each layer, find candidates, distil S2
    into S1, update the router, and decay pattern frequencies.

    Returns a report dict for logging.
    """
    report = {"step": step, "layers": []}
    batch_size = cfg.get("consolidation_batch_size", 20)

    for layer_idx, (layer, tracker) in enumerate(
        zip(model._layers, trackers)
    ):
        candidates = tracker.get_candidates(
            min_frequency=cfg.get("consolidation_min_freq", 50),
            min_s2_ratio=cfg.get("consolidation_min_s2_ratio", 0.8),
        )
        if len(candidates) < cfg.get("consolidation_min_candidates", 3):
            continue

        # Take only the most frequent candidates this round
        top_candidates = candidates[:batch_size]

        s1_proj = model.s1_projections[layer_idx]
        router  = model.routers[layer_idx]

        # Phase A: Distil S2 → S1
        distill_loss = consolidate_layer(
            layer=layer,
            s1_proj=s1_proj,
            candidates=top_candidates,
            cfg=cfg,
            device=device,
        )

        # Phase B: Steer router toward S1 for these patterns
        update_router_post_consolidation(
            router=router,
            candidates=top_candidates,
            cfg=cfg,
            device=device,
        )

        # Phase C: Decay frequency counts so these patterns need to re-earn S2
        tracker.decay_candidates(top_candidates)

        report["layers"].append({
            "layer_idx":    layer_idx,
            "n_candidates": len(top_candidates),
            "distill_loss": distill_loss,
        })

    return report
