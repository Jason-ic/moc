"""
MoCWrapper: wraps a frozen pre-trained model and injects per-layer
S1/S2 routing via PyTorch forward hooks.

Architecture:
  For every transformer layer i:
    Pre-hook  → compute router score + S1 output, store temporarily
    Post-hook → blend S2 output (from the layer) with stored S1 output

The base model's weight is FROZEN throughout.  Only routers and
s1_projections are trainable.
"""

import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer

from router import MoCRouter, MetacognitiveRouter
from s1_path import S1Projection


class MoCWrapper(nn.Module):
    def __init__(self, cfg: dict, device: torch.device):
        super().__init__()
        self.cfg = cfg
        self.device = device

        # ------------------------------------------------------------------
        # 1. Load pre-trained model
        # ------------------------------------------------------------------
        dtype = torch.float16 if cfg["model_dtype"] == "float16" else torch.float32
        self.base_model = AutoModelForCausalLM.from_pretrained(
            cfg["model_name"],
            torch_dtype=dtype,
            trust_remote_code=True,
        ).to(device)
        self.tokenizer = AutoTokenizer.from_pretrained(
            cfg["model_name"], trust_remote_code=True
        )

        # Freeze ALL base model parameters
        for param in self.base_model.parameters():
            param.requires_grad = False

        # ------------------------------------------------------------------
        # 2. Resolve the layer list (handles both plain LM and VL models)
        # ------------------------------------------------------------------
        self._layers = self._get_layers()
        self.n_layers = len(self._layers)
        self.emb_dim = self.base_model.config.hidden_size

        # ------------------------------------------------------------------
        # 3. Trainable components: one router + one S1 proj per layer
        # ------------------------------------------------------------------
        self.routers = nn.ModuleList([
            MoCRouter(self.emb_dim, cfg["router_dim"])
            for _ in range(self.n_layers)
        ])
        self.s1_projections = nn.ModuleList([
            S1Projection(self.emb_dim)
            for _ in range(self.n_layers)
        ])
        # Trainable modules stay in fp32 for AdamW numerical stability.
        # Blending in hooks casts s1_output to match s2_output dtype before mixing.
        self.routers.to(device=device)           # fp32
        self.s1_projections.to(device=device)   # fp32
        self._trainable_dtype = torch.float32

        # ------------------------------------------------------------------
        # 4. Register hooks
        # ------------------------------------------------------------------
        self._hook_storage: list[dict] = [{}] * self.n_layers
        self._collect_routing_info: bool = False
        self._routing_info_buffer: list[dict] = []
        self._hooks: list = []
        self._register_hooks()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _get_layers(self):
        """Return the list of decoder layers regardless of model variant."""
        model = self.base_model
        # Standard CausalLM: model.model.layers
        if hasattr(model, "model") and hasattr(model.model, "layers"):
            return model.model.layers
        # VL / multimodal wrapper: model.language_model.model.layers
        if hasattr(model, "language_model"):
            lm = model.language_model
            if hasattr(lm, "model") and hasattr(lm.model, "layers"):
                return lm.model.layers
        raise AttributeError(
            f"Cannot find layer list in {type(model).__name__}. "
            "Update _get_layers() in moc_model.py."
        )

    def _get_text_model(self):
        model = self.base_model
        if hasattr(model, "model") and hasattr(model.model, "layers"):
            return model.model
        if hasattr(model, "language_model"):
            return model.language_model.model
        raise AttributeError("Cannot find text model sub-module.")

    def _register_hooks(self):
        for i, layer in enumerate(self._layers):
            pre_h = layer.register_forward_pre_hook(self._make_pre_hook(i))
            post_h = layer.register_forward_hook(self._make_post_hook(i))
            self._hooks.extend([pre_h, post_h])

    def remove_hooks(self):
        """Call this to remove all hooks (e.g. before saving the base model)."""
        for h in self._hooks:
            h.remove()
        self._hooks.clear()

    # ------------------------------------------------------------------
    # Hook factories
    # ------------------------------------------------------------------

    def _make_pre_hook(self, layer_idx: int):
        def pre_hook(module, args):
            # args[0] is hidden_states: (B, T, D) in fp16
            hidden = args[0]  # model dtype (fp16 or fp32)

            # Cast to trainable dtype for router / S1 computation
            hidden_cast = hidden.to(self._trainable_dtype)

            router = self.routers[layer_idx]
            s1_proj = self.s1_projections[layer_idx]

            # --- Router score ---
            router_logits = router(hidden_cast)          # (B, T)
            router_probs = torch.sigmoid(router_logits) # (B, T)

            # Top-k selection: highest-score tokens go to S2
            B, T = router_probs.shape
            k = max(1, int(T * self.cfg["capacity_factor"]))
            _, top_idx = torch.topk(router_probs, k, dim=-1)  # (B, k)
            s2_mask = torch.zeros_like(router_probs)
            s2_mask.scatter_(1, top_idx, 1.0)               # (B, T)  binary

            # --- S1 output ---
            s1_output = s1_proj(hidden_cast)               # (B, T, D) trainable_dtype

            # Store for the matching post-hook
            self._hook_storage[layer_idx] = {
                "router_logits": router_logits,
                "router_probs": router_probs,
                "s2_mask": s2_mask,
                "s1_output": s1_output,
                "input_hidden": hidden_cast.detach(),   # for Pattern Tracker
            }
            return args  # pass inputs unchanged to the layer
        return pre_hook

    def _make_post_hook(self, layer_idx: int):
        def post_hook(module, args, output):
            store = self._hook_storage[layer_idx]

            # output may be a Tensor or a tuple; handle both
            if isinstance(output, tuple):
                s2_output = output[0]
                rest = output[1:]
            else:
                s2_output = output
                rest = None

            # s2_output is fp16 (frozen model); s1/router are fp32 (trainable)
            # Blend in s2_output's dtype to keep MPS happy during backward.
            # Grad still flows: blended_fp16 → s1_fp16 ← .to(fp16) ← s1_fp32
            mix_dtype = s2_output.dtype
            s2_out   = s2_output.detach()                    # fp16, no grad needed
            s1_out   = store["s1_output"].to(mix_dtype)      # fp32→fp16, grad flows back
            probs    = store["router_probs"].to(mix_dtype)   # fp32→fp16
            s2_mask  = store["s2_mask"].to(mix_dtype)

            mask_3d  = s2_mask.unsqueeze(-1)                 # (B, T, 1)

            if self.training:
                probs_3d = probs.unsqueeze(-1)
                mask_st  = mask_3d - probs_3d.detach() + probs_3d
                blended  = mask_st * s2_out + (1.0 - mask_st) * s1_out
            else:
                blended  = mask_3d * s2_out + (1.0 - mask_3d) * s1_out

            # blended is already mix_dtype (= s2_output.dtype), no extra cast needed

            # Optionally accumulate routing info
            if self._collect_routing_info:
                self._routing_info_buffer.append({
                    "layer_idx": layer_idx,
                    "router_logits": store["router_logits"].detach(),
                    "router_probs": probs.detach(),
                    "s2_mask": s2_mask.detach(),
                    "s2_output": s2_out.detach(),
                    "s1_output": s1_out.detach(),
                    "input_hidden": store["input_hidden"],
                })

            if rest is not None:
                return (blended,) + rest
            return blended
        return post_hook

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        return_routing_info: bool = False,
    ):
        self._collect_routing_info = return_routing_info
        self._routing_info_buffer = []

        # Standard causal-LM forward; hooks intercept each layer internally
        outputs = self.base_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )
        logits = outputs.logits  # (B, T, vocab_size)

        if return_routing_info:
            # Sort by layer index (hooks fire in order but be safe)
            info = sorted(self._routing_info_buffer, key=lambda x: x["layer_idx"])
            self._collect_routing_info = False
            return logits, info

        self._collect_routing_info = False
        return logits

    # ------------------------------------------------------------------
    # Convenience
    # ------------------------------------------------------------------

    def trainable_parameters(self):
        """Generator over trainable params (routers + s1_projections only)."""
        yield from self.routers.parameters()
        yield from self.s1_projections.parameters()

    def num_trainable_params(self) -> int:
        return sum(p.numel() for p in self.trainable_parameters())

    def num_base_params(self) -> int:
        return sum(p.numel() for p in self.base_model.parameters())

    def routing_summary(self, routing_info: list) -> dict:
        """Summarise routing info for logging."""
        s2_ratios = [info["s2_mask"].float().mean().item() for info in routing_info]
        return {
            "mean_s2_ratio": sum(s2_ratios) / len(s2_ratios),
            "min_s2_ratio": min(s2_ratios),
            "max_s2_ratio": max(s2_ratios),
        }
