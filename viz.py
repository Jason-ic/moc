"""Visualisation utilities for MoC experiments."""

import torch
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import os


def plot_routing_heatmap(routing_info: list, save_path: str, title: str = ""):
    """
    Heatmap: Layer (y-axis) × Token position (x-axis), colour = S2 probability.
    """
    n_layers = len(routing_info)
    T = routing_info[0]["router_probs"].shape[-1]

    # Average across batch dimension
    matrix = np.zeros((n_layers, T))
    for info in routing_info:
        layer_idx = info["layer_idx"]
        probs = info["router_probs"].float().mean(dim=0).cpu().numpy()  # (T,)
        matrix[layer_idx] = probs

    fig, ax = plt.subplots(figsize=(min(T // 4 + 2, 20), n_layers // 2 + 2))
    im = ax.imshow(matrix, aspect="auto", cmap="RdYlGn_r", vmin=0, vmax=1)
    plt.colorbar(im, ax=ax, label="P(S2)")
    ax.set_xlabel("Token position")
    ax.set_ylabel("Layer index")
    ax.set_title(title or "Routing heatmap (high = S2, low = S1)")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"Saved routing heatmap → {save_path}")


def plot_s2_ratio_over_time(
    step_list: list,
    s2_ratio_by_difficulty: dict[int, list],
    consolidation_steps: list,
    save_path: str,
):
    """
    Line chart: training step → mean S2 ratio, one line per difficulty.
    Vertical dashed lines mark consolidation events.
    """
    fig, ax = plt.subplots(figsize=(12, 5))
    colours = {1: "green", 2: "orange", 3: "red"}
    labels  = {1: "Easy (diff=1)", 2: "Medium (diff=2)", 3: "Hard (diff=3)"}

    for diff, ratios in s2_ratio_by_difficulty.items():
        ax.plot(step_list[:len(ratios)], ratios,
                color=colours.get(diff, "blue"),
                label=labels.get(diff, f"diff={diff}"),
                linewidth=2)

    for cs in consolidation_steps:
        ax.axvline(x=cs, color="purple", linestyle="--", alpha=0.6,
                   label="Consolidation" if cs == consolidation_steps[0] else "")

    ax.set_xlabel("Training step")
    ax.set_ylabel("Mean S2 ratio")
    ax.set_title("S2 Routing Ratio Over Training\n(Medium patterns should drop after consolidation)")
    ax.legend()
    ax.set_ylim(0, 1.05)
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"Saved S2-ratio-over-time → {save_path}")


def plot_flop_savings(
    step_list: list,
    flop_savings_by_difficulty: dict[int, list],
    save_path: str,
):
    """
    Bar chart or line chart showing FLOP savings % per difficulty over time.
    """
    fig, ax = plt.subplots(figsize=(10, 5))
    colours = {1: "green", 2: "orange", 3: "red"}

    for diff, savings in flop_savings_by_difficulty.items():
        ax.plot(step_list[:len(savings)], savings,
                color=colours.get(diff, "blue"),
                label=f"diff={diff}", linewidth=2)

    ax.set_xlabel("Training step")
    ax.set_ylabel("FLOP savings (%)")
    ax.set_title("FLOP Savings Over Training by Difficulty")
    ax.legend()
    ax.set_ylim(0, 100)
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"Saved FLOP savings chart → {save_path}")


def plot_consolidation_gap_over_time(
    step_list: list,
    repeated_s2: list,
    novel_s2: list,
    consolidation_steps: list,
    save_path: str,
):
    """
    The key paper figure: repeated vs novel S2 ratio over time.
    """
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(step_list[:len(repeated_s2)], repeated_s2,
            color="blue", label="Repeated patterns", linewidth=2)
    ax.plot(step_list[:len(novel_s2)], novel_s2,
            color="red",  label="Novel patterns",    linewidth=2)

    for cs in consolidation_steps:
        ax.axvline(x=cs, color="purple", linestyle="--", alpha=0.6,
                   label="Consolidation" if cs == consolidation_steps[0] else "")

    ax.fill_between(
        step_list[:min(len(repeated_s2), len(novel_s2))],
        repeated_s2[:len(novel_s2)],
        novel_s2[:len(repeated_s2)],
        alpha=0.15, color="green", label="Gap (consolidation effect)",
    )

    ax.set_xlabel("Training step")
    ax.set_ylabel("Mean S2 ratio")
    ax.set_title("Consolidation Effect: Repeated vs Novel Patterns\n"
                 "Growing gap → successful S2→S1 knowledge transfer")
    ax.legend()
    ax.set_ylim(0, 1.05)
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"Saved consolidation gap chart → {save_path}")


def plot_forgetting_test(
    step_list: list,
    forgetting_accuracies: list,
    save_path: str,
):
    """
    Line chart of accuracy on consolidated patterns over time.
    Should stay flat (no forgetting).
    """
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(step_list[:len(forgetting_accuracies)], forgetting_accuracies,
            color="navy", linewidth=2)
    ax.axhline(y=0.95, color="green", linestyle="--", alpha=0.7,
               label="95% threshold")
    ax.set_xlabel("Training step")
    ax.set_ylabel("Accuracy on consolidated patterns")
    ax.set_title("Forgetting Test: Accuracy Should Stay Above 95%")
    ax.legend()
    ax.set_ylim(0, 1.05)
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"Saved forgetting test chart → {save_path}")


def plot_s1_deviation(
    step_list: list,
    deviations_by_layer: dict[int, list],
    save_path: str,
):
    """
    How far has each S1 projection drifted from identity?
    Larger deviation = S1 has learned more task-specific content.
    """
    fig, ax = plt.subplots(figsize=(12, 5))
    n_layers = len(deviations_by_layer)
    cmap = plt.cm.viridis(np.linspace(0, 1, n_layers))

    for layer_idx, devs in deviations_by_layer.items():
        ax.plot(step_list[:len(devs)], devs,
                color=cmap[layer_idx], alpha=0.7, linewidth=1.5,
                label=f"Layer {layer_idx}" if layer_idx % 6 == 0 else "")

    ax.set_xlabel("Training step")
    ax.set_ylabel("||W - I||_F  (S1 deviation from identity)")
    ax.set_title("S1 Projection Drift from Identity\n(Higher = more S2 knowledge absorbed)")
    ax.legend(ncol=3, fontsize=8)
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"Saved S1 deviation chart → {save_path}")
