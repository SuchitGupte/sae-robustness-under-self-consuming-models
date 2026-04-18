"""
plot_metrics.py — Plot model collapse metrics from eval_results.json.

Produces a 3-panel figure:
  - Perplexity (held-out, real text)
  - N-gram repetition (4-gram)
  - Unique token ratio (UTR)

Usage
-----
    python plot_metrics.py
    python plot_metrics.py --results ./checkpoints/eval_results.json
    python plot_metrics.py --results ./checkpoints/eval_results.json --out collapse.png --dpi 150
    python plot_metrics.py --metrics perplexity ngram_repetition   # subset of panels

Requirements
------------
    pip install matplotlib numpy
"""

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np


# ──────────────────────────────────────────────────────────────────────────────
# Config
# ──────────────────────────────────────────────────────────────────────────────

METRIC_CONFIG = {
    "perplexity": {
        "label":     "Perplexity",
        "ylabel":    "Perplexity (↑ = more collapse)",
        "color":     "#378ADD",
        "fill":      "#378ADD22",
        "direction": "up",
        "note":      "evaluated on frozen WikiText-2 reference",
    },
    "ngram_repetition": {
        "label":     "N-gram repetition",
        "ylabel":    "Repeated 4-gram fraction (↑ = more collapse)",
        "color":     "#D85A30",
        "fill":      "#D85A3022",
        "direction": "up",
        "note":      "fraction of 4-grams appearing more than once",
    },
    "unique_token_ratio": {
        "label":     "Unique token ratio (UTR)",
        "ylabel":    "Unique / total tokens (↓ = more collapse)",
        "color":     "#1D9E75",
        "fill":      "#1D9E7522",
        "direction": "down",
        "note":      "global unique tokens ÷ total tokens across all outputs",
    },
}

ALL_METRICS = list(METRIC_CONFIG.keys())


# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────

def load_results(path: str) -> list[dict]:
    p = Path(path)
    assert p.exists(), f"Results file not found: {path}"
    with open(p) as f:
        data = json.load(f)
    assert isinstance(data, list) and len(data) > 0, "eval_results.json is empty or malformed"
    return data


def extract_series(results: list[dict], metric: str) -> tuple[list[int], list[float]]:
    """Return (generations, values) for a given metric key."""
    gens, vals = [], []
    for r in results:
        v = r.get(metric)
        if v is not None:
            gens.append(r["generation"])
            vals.append(float(v))
    assert len(vals) > 0, f"No data found for metric '{metric}'"
    return gens, vals


def annotate_collapse(ax, gens, vals, direction: str):
    """Mark the generation with the worst collapse value."""
    if direction == "up":
        worst_idx = int(np.argmax(vals))
    else:
        worst_idx = int(np.argmin(vals))

    gx, vy = gens[worst_idx], vals[worst_idx]
    ax.annotate(
        f"gen {gx}\n{vy:.3g}",
        xy=(gx, vy),
        xytext=(gx + 0.3, vy * (1.04 if direction == "up" else 0.96)),
        fontsize=8,
        color="#555",
        arrowprops=dict(arrowstyle="-", color="#aaa", lw=0.8),
    )


def style_ax(ax, cfg: dict, gens: list[int]):
    """Apply consistent axis styling."""
    ax.set_ylabel(cfg["ylabel"], fontsize=9, labelpad=6, color="#444")
    ax.set_xlabel("generation", fontsize=9, labelpad=4, color="#444")
    ax.set_xticks(gens)
    ax.xaxis.set_major_formatter(ticker.FormatStrFormatter("%d"))
    ax.tick_params(axis="both", labelsize=8, colors="#555")
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["left", "bottom"]].set_color("#ccc")
    ax.grid(axis="y", color="#eee", linewidth=0.8, zorder=0)
    ax.set_facecolor("#fafafa")
    note = cfg.get("note", "")
    if note:
        ax.set_title(note, fontsize=7.5, color="#888", pad=4, style="italic")


# ──────────────────────────────────────────────────────────────────────────────
# Plot
# ──────────────────────────────────────────────────────────────────────────────

def plot(
    results: list[dict],
    metrics: list[str],
    out_path: str,
    dpi: int,
) -> None:
    n = len(metrics)
    fig, axes = plt.subplots(
        n, 1,
        figsize=(8, 3.2 * n),
        sharex=False,
    )
    if n == 1:
        axes = [axes]

    fig.suptitle(
        "Self-consuming loop — model collapse metrics",
        fontsize=13,
        fontweight="medium",
        color="#222",
        y=1.01,
    )

    for ax, metric in zip(axes, metrics):
        cfg = METRIC_CONFIG[metric]
        gens, vals = extract_series(results, metric)

        # Fill area
        ax.fill_between(gens, vals, alpha=0.12, color=cfg["color"], zorder=1)

        # Main line
        ax.plot(
            gens, vals,
            color=cfg["color"],
            linewidth=2,
            marker="o",
            markersize=5,
            markerfacecolor="white",
            markeredgecolor=cfg["color"],
            markeredgewidth=1.5,
            zorder=3,
            label=cfg["label"],
        )

        # Value labels on each point
        for g, v in zip(gens, vals):
            ax.text(
                g, v,
                f"{v:.3g}",
                ha="center",
                va="bottom" if cfg["direction"] == "up" else "top",
                fontsize=7.5,
                color=cfg["color"],
                fontweight="medium",
                zorder=4,
            )

        # Annotate worst generation
        annotate_collapse(ax, gens, vals, cfg["direction"])

        # Trend line (linear fit in log space for perplexity, linear otherwise)
        if len(gens) >= 3:
            x = np.array(gens, dtype=float)
            y = np.array(vals, dtype=float)
            if metric == "perplexity":
                logy = np.log(y)
                coef = np.polyfit(x, logy, 1)
                trend = np.exp(np.polyval(coef, x))
            else:
                coef = np.polyfit(x, y, 1)
                trend = np.polyval(coef, x)
            ax.plot(
                gens, trend,
                color=cfg["color"],
                linewidth=1,
                linestyle="--",
                alpha=0.45,
                zorder=2,
                label="trend",
            )
            ax.legend(fontsize=8, framealpha=0, loc="upper left")

        style_ax(ax, cfg, gens)

    plt.tight_layout(pad=1.4)
    plt.savefig(out_path, dpi=dpi, bbox_inches="tight", facecolor="white")
    print(f"Saved → {out_path}")
    plt.show()


# ──────────────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot self-consuming loop collapse metrics",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--results", default="./checkpoints/eval_results.json",
        help="Path to eval_results.json produced by evaluate.py",
    )
    parser.add_argument(
        "--out", default="collapse_metrics.png",
        help="Output image path (.png / .pdf / .svg)",
    )
    parser.add_argument(
        "--dpi", type=int, default=150,
        help="Output resolution",
    )
    parser.add_argument(
        "--metrics", nargs="+", default=ALL_METRICS,
        choices=ALL_METRICS,
        help="Which metrics to plot (subset of panels)",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    results = load_results(args.results)
    plot(results, args.metrics, args.out, args.dpi)