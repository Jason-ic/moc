# Mixture of Cognition (MoC)

> **Online S2→S1 Knowledge Consolidation in a Single Pretrained LLM**

Humans learn to drive by consciously reasoning through each action (System 2), then gradually automating those patterns into intuition (System 1). This project implements an analogous mechanism in a transformer: a lightweight dual-path router directs each token to either a fast S1 path or the full pretrained S2 path, and an online consolidation algorithm progressively transfers frequently-used S2 patterns into S1 — reducing compute for familiar inputs without retraining the base model.

**Key difference from prior work**: Meta FAIR's ["Distilling System 2 into System 1"](https://arxiv.org/abs/2407.06023) performs *offline* distillation into a separate model. MoC performs *online, within a single model*, continuously during fine-tuning.

---

## Architecture

```
Input tokens
    │
    ▼
Embedding (frozen, pretrained)
    │
    ▼  ┌─────────────────────────────────────────────────────┐
    │  │  For each transformer layer l = 1 … L:              │
    │  │                                                     │
    │  │    Router(hidden) ──► score ∈ [0,1] per token       │
    │  │                                                     │
    │  │    top-k tokens (score HIGH) ──► S2 path            │
    │  │    ├── Full pretrained layer  (frozen, ~12D² FLOPs) │
    │  │                                                     │
    │  │    remaining tokens ──────────► S1 path             │
    │  │    └── S1Projection W ≈ I     (trainable, ~1D² FLOPs)│
    │  │                                                     │
    │  │    blended = mask·S2 + (1−mask)·S1                  │
    │  └─────────────────────────────────────────────────────┘
    │
    ▼
LM head (frozen, pretrained)
    │
    ▼
Logits
```

**Trainable parameters only** (~33M out of 596M):
- Per-layer `MoCRouter`: 2-layer MLP → scalar score per token
- Per-layer `S1Projection`: single linear layer, initialized as identity

**Base model is fully frozen.** Hooks intercept each layer's forward pass.

---

## Online Consolidation (Phase 4)

The novel contribution. Every 500 training steps:

1. **Pattern Tracker** (LSH): hashes hidden states into 8192 buckets, tracks how often each bucket is routed to S2
2. **Trigger**: buckets with `total_freq ≥ 50` and `s2_ratio ≥ 0.8` become consolidation candidates
3. **Distillation**: for each candidate pattern, train S1 to match S2 output via MSE loss + EWC regularization (prevents forgetting)
4. **Router update**: fine-tune router to redirect consolidated patterns to S1
5. **Decay**: reset S2 frequency counter for consolidated buckets

Expected outcome: repeated TriviaQA patterns drop from S2 ratio ~0.9 → ~0.2 after consolidation, while novel questions remain at ~0.85.

---

## Project Structure

```
mixture-of-cognition/
├── configs.py          # All hyperparameters
├── moc_model.py        # MoCWrapper: hooks + routing + blending
├── router.py           # MoCRouter, MetacognitiveRouter, aux losses
├── s1_path.py          # S1Projection (identity init)
├── consolidation.py    # PatternTracker (LSH), ConsolidationScheduler, EWC distill
├── data.py             # TriviaQADataset: difficulty labeling + repeat injection
├── train.py            # Training loop: task loss + router loss + consolidation
├── eval.py             # Per-difficulty accuracy, FLOP savings, consolidation gap
├── viz.py              # Routing heatmap, S2 ratio curves, consolidation gap plot
├── run_experiment.py   # Main entry point (CLI)
├── requirements.txt
└── cloud/
    └── setup.sh        # RunPod / Lambda Labs GPU setup
```

---

## Setup

### Local (macOS / CPU)

```bash
conda create -n moc python=3.10 -y
conda activate moc
pip install torch==2.2.2 torchvision
pip install -r requirements.txt
```

### Cloud GPU (RunPod, CUDA 12.1)

```bash
git clone https://github.com/Jason-ic/moc.git
cd moc
bash cloud/setup.sh
conda activate moc
wandb login   # optional
```

**Recommended GPU**: RTX 3090 / 4090 (24GB VRAM). Minimum: 10GB VRAM.

---

## Usage

### Smoke test (verify pipeline end-to-end)

```bash
python run_experiment.py --smoke-test
```

Expected output:
```
logits shape    : torch.Size([1, 64, 151936])
n_layers routed : 28
mean S2 ratio   : 0.500
FLOP savings    : 98.1%
Smoke test PASSED ✓
```

### Phase 1 — Routing only, no consolidation

```bash
# Quick validation (500 samples)
python run_experiment.py --phase 1 --max-samples 500

# Full training
python run_experiment.py --phase 1 --max-samples 50000
```

### Phase 4 — With online consolidation

```bash
python run_experiment.py --phase 4 --max-samples 50000
```

### Evaluation only

```bash
python run_experiment.py --eval --checkpoint checkpoints/moc_best.pt
```

### CLI options

| Flag | Default | Description |
|------|---------|-------------|
| `--phase` | 1 | Experiment phase (1=routing only, 4=with consolidation) |
| `--max-samples` | all | Limit dataset size |
| `--batch-size` | 4 | Override batch size |
| `--lr` | 3e-4 | Override learning rate |
| `--smoke-test` | — | Single forward pass sanity check |
| `--eval` | — | Evaluation only (requires `--checkpoint`) |
| `--checkpoint` | — | Path to `.pt` checkpoint |

---

## Key Hyperparameters (`configs.py`)

| Parameter | Value | Description |
|-----------|-------|-------------|
| `model_name` | `Qwen/Qwen3-0.6B` | Base pretrained model |
| `capacity_factor` | 0.5 | Fraction of tokens routed to S2 |
| `router_dim` | 128 | Hidden dim of router MLP |
| `z_loss_weight` | 0.01 | Stabilise router logits |
| `balance_loss_weight` | 0.01 | Encourage balanced routing |
| `consolidation_min_freq` | 50 | Min S2 visits before consolidation |
| `consolidation_min_s2_ratio` | 0.8 | Min S2 ratio to be a candidate |
| `ewc_lambda` | 100.0 | EWC forgetting penalty |
| `lsh_n_planes` | 32 | LSH random projection planes |

---

## Metrics

| Metric | What it measures |
|--------|-----------------|
| `accuracy_diff{1,2,3}` | Next-token accuracy per difficulty (easy/medium/hard) |
| `s2_ratio_diff{1,2,3}` | Fraction of tokens routed to S2 per difficulty |
| `flop_savings_pct` | FLOP reduction vs all-S2 baseline |
| `consolidation_gap` | S2 ratio difference: novel − repeated patterns |
| Forgetting rate | Accuracy on consolidated patterns over time |

### Target results (Phase 4)

> For TriviaQA patterns seen >50 times:
> - S2 ratio drops from ~0.9 → ~0.2 after consolidation
> - QA accuracy remains >95%
> - Novel questions maintain S2 ratio >0.85

---

## Baselines (Phase 5)

| Baseline | Description |
|----------|-------------|
| Vanilla Qwen3-0.6B | No routing, all tokens through all layers |
| MoC without consolidation | Routing only, no S2→S1 transfer |
| Static MoD | Routing fixed after warmup, no online update |
| **MoC (ours)** | Full dual-path routing + online consolidation |

---

## Implementation Notes

- **MPS (Apple Silicon)**: base model in fp16, trainable modules in fp32. Blending is cast to fp16 before mixing to avoid `mps.select` dtype errors during backward.
- **Gradient flow**: `s2_output.detach()` ensures no gradients flow into the frozen base model parameters. Gradients only update router and S1 projections.
- **Straight-through estimator**: discrete top-k mask is made differentiable via `mask_st = mask - probs.detach() + probs`.
- **EWC**: Fisher diagonal approximated via 200 random-input gradient samples before each consolidation.

---

## Experiment Budget (Cloud GPU)

| Run | Samples | Est. time (RTX 4090) | Cost |
|-----|---------|----------------------|------|
| Smoke test | — | <1 min | ~$0 |
| Phase 1 validation | 5k | ~10 min | ~$0.10 |
| Phase 1 full | 50k | ~1.5h | ~$0.70 |
| Phase 4 full | 50k | ~3h | ~$1.50 |
| Ablations × 3 | 50k each | ~9h | ~$4 |

---

## Related Work

- [Distilling System 2 into System 1](https://arxiv.org/abs/2407.06023) — Meta FAIR, offline distillation
- [Mixture of Depths](https://arxiv.org/abs/2404.02258) — per-token layer skipping
- [Thinking Fast and Slow](https://en.wikipedia.org/wiki/Thinking,_Fast_and_Slow) — Kahneman, cognitive inspiration
- [Elastic Weight Consolidation](https://arxiv.org/abs/1612.00796) — forgetting prevention
