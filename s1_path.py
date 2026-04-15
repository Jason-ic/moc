"""S1 (System 1) lightweight bypass path."""

import torch
import torch.nn as nn


class S1Projection(nn.Module):
    """
    Lightweight S1 bypass that replaces a full transformer layer.

    Initialised as the identity map, so early in training the model
    behaves identically to the frozen base model regardless of routing.

    As training proceeds, S1 learns to approximate what S2 would produce
    for the patterns it is asked to handle.  The consolidation mechanism
    (Phase 4) actively distils S2 knowledge into S1 for repeated patterns.

    Input / output shape: (batch, seq_len, emb_dim)
    """

    def __init__(self, emb_dim: int):
        super().__init__()
        self.proj = nn.Linear(emb_dim, emb_dim, bias=True)
        # Identity initialisation: output ≈ input at the start
        nn.init.eye_(self.proj.weight)
        nn.init.zeros_(self.proj.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj(x)

    @property
    def deviation_from_identity(self) -> float:
        """Diagnostic: how far has S1 drifted from identity?"""
        with torch.no_grad():
            eye = torch.eye(
                self.proj.weight.shape[0],
                device=self.proj.weight.device,
                dtype=self.proj.weight.dtype,
            )
            return (self.proj.weight - eye).norm().item()
