"""
Calibration Analysis (P1.4).

Reliability diagrams + ECE (Expected Calibration Error) for:
  - gcn-fusion
  - gcn-graph
  - EGARCH(1,1)

Data: research/bootstrap_rerun/final/predictions.csv  (gcn scores + labels)
      paper/revisions/egarch_test_scores.csv           (EGARCH scores)

Outputs:
  paper/revisions/calibration_results.json
  paper/figures/figure_calibration.pdf

Usage:
    cd <repo_root>
    python paper/revisions/compute_calibration.py
"""
from __future__ import annotations

import json
import os

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

PREDICTIONS_CSV   = os.path.join(REPO_ROOT, "research", "bootstrap_rerun", "final", "predictions.csv")
EGARCH_SCORES_CSV = os.path.join(REPO_ROOT, "paper", "revisions", "egarch_test_scores.csv")
OUTPUT_JSON       = os.path.join(REPO_ROOT, "paper", "revisions", "calibration_results.json")
OUTPUT_FIG        = os.path.join(REPO_ROOT, "paper", "figures", "figure_calibration.pdf")

ECE_BINS = 15


def compute_ece(y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 15) -> float:
    """Expected Calibration Error with equal-width bins."""
    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    N = len(y_true)
    for i in range(n_bins):
        lo, hi = bin_edges[i], bin_edges[i + 1]
        # include right edge in last bin
        if i < n_bins - 1:
            mask = (y_prob >= lo) & (y_prob < hi)
        else:
            mask = (y_prob >= lo) & (y_prob <= hi)
        n_bin = mask.sum()
        if n_bin == 0:
            continue
        acc  = y_true[mask].mean()
        conf = y_prob[mask].mean()
        ece += (n_bin / N) * abs(acc - conf)
    return float(ece)


def reliability_diagram_data(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    n_bins: int = 10,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Equal-frequency bins (deciles). Returns (mean_prob, frac_pos, bin_sizes)."""
    quantiles = np.linspace(0, 100, n_bins + 1)
    bin_edges = np.percentile(y_prob, quantiles)
    bin_edges = np.unique(bin_edges)
    mean_probs, frac_pos, sizes = [], [], []
    for i in range(len(bin_edges) - 1):
        lo, hi = bin_edges[i], bin_edges[i + 1]
        if i < len(bin_edges) - 2:
            mask = (y_prob >= lo) & (y_prob < hi)
        else:
            mask = (y_prob >= lo) & (y_prob <= hi)
        if mask.sum() == 0:
            continue
        mean_probs.append(y_prob[mask].mean())
        frac_pos.append(y_true[mask].mean())
        sizes.append(mask.sum())
    return np.array(mean_probs), np.array(frac_pos), np.array(sizes)


def main() -> None:
    print("Loading predictions ...")
    pred_df   = pd.read_csv(PREDICTIONS_CSV,   parse_dates=["timestamp"])
    egarch_df = pd.read_csv(EGARCH_SCORES_CSV, parse_dates=["timestamp"])

    merged = pred_df.merge(
        egarch_df[["symbol", "timestamp", "egarch_score"]],
        on=["symbol", "timestamp"],
        how="left",
    )
    n_miss = merged["egarch_score"].isna().sum()
    if n_miss > 0:
        print(f"  WARNING: {n_miss} missing EGARCH scores -> filling 0.0")
    merged["egarch_score"] = merged["egarch_score"].fillna(0.0)

    y          = merged["label"].to_numpy(dtype=float)
    s_gcn_f    = merged["gcn_fusion_score"].to_numpy(dtype=float)
    s_gcn_g    = merged["gcn_graph_score"].to_numpy(dtype=float)
    s_egarch   = merged["egarch_score"].to_numpy(dtype=float)

    print(f"  Test samples: {len(y):,}  |  Events: {int(y.sum())} ({100*y.mean():.1f}%)")

    # ── ECE ─────────────────────────────────────────────────────────────────────
    ece_gcn_f  = round(compute_ece(y, s_gcn_f,  ECE_BINS), 4)
    ece_gcn_g  = round(compute_ece(y, s_gcn_g,  ECE_BINS), 4)
    ece_egarch = round(compute_ece(y, s_egarch, ECE_BINS), 4)

    print(f"\n  ECE (gcn-fusion):  {ece_gcn_f:.4f}")
    print(f"  ECE (gcn-graph):   {ece_gcn_g:.4f}")
    print(f"  ECE (EGARCH):      {ece_egarch:.4f}")

    # ── Reliability diagram data ────────────────────────────────────────────────
    mp_f, fp_f, sz_f   = reliability_diagram_data(y, s_gcn_f,  n_bins=10)
    mp_g, fp_g, sz_g   = reliability_diagram_data(y, s_gcn_g,  n_bins=10)
    mp_e, fp_e, sz_e   = reliability_diagram_data(y, s_egarch, n_bins=10)

    # ── Figure ───────────────────────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 3, figsize=(12, 4), sharey=False)

    models = [
        ("gcn-fusion",  mp_f, fp_f, sz_f,  ece_gcn_f,  "#1f77b4"),
        ("gcn-graph",   mp_g, fp_g, sz_g,  ece_gcn_g,  "#ff7f0e"),
        ("EGARCH(1,1)", mp_e, fp_e, sz_e,  ece_egarch, "#2ca02c"),
    ]

    for ax, (name, mp, fp, sz, ece, color) in zip(axes, models):
        # Perfect calibration line
        ax.plot([0, 1], [0, 1], "k--", alpha=0.4, linewidth=1.0, label="Perfect calibration")
        # Reliability curve
        ax.plot(mp, fp, "o-", color=color, linewidth=1.8, markersize=5, label=name)
        # Shaded gap
        for xi, yi in zip(mp, fp):
            ax.plot([xi, xi], [xi, yi], color=color, alpha=0.25, linewidth=3)

        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_xlabel("Mean predicted probability", fontsize=8)
        ax.set_ylabel("Fraction of positives", fontsize=8)
        ax.set_title(f"{name}\nECE = {ece:.4f}", fontsize=9)
        ax.tick_params(labelsize=7)
        ax.grid(True, alpha=0.3)

        # Histogram of predicted probs (inset)
        ax2 = ax.inset_axes([0.55, 0.05, 0.4, 0.25])
        ax2.hist(mp, bins=10, weights=sz / sz.sum(), color=color, alpha=0.6)
        ax2.set_xticks([])
        ax2.set_yticks([])
        ax2.set_title("score dist.", fontsize=6)

    fig.suptitle("Reliability Diagrams — Test Split 2022–2026\n(10 equal-frequency bins)",
                 fontsize=9, y=1.02)
    plt.tight_layout()
    plt.savefig(OUTPUT_FIG, bbox_inches="tight")
    plt.close()
    print(f"\nFigure saved: {OUTPUT_FIG}")

    # ── Save JSON ────────────────────────────────────────────────────────────────
    results = {
        "ece": {
            "gcn_fusion":  ece_gcn_f,
            "gcn_graph":   ece_gcn_g,
            "egarch":      ece_egarch,
            "n_bins":      ECE_BINS,
            "note": "Equal-width bins, ECE = sum_b (n_b/N) * |acc_b - conf_b|",
        },
        "reliability": {
            "gcn_fusion":  {"mean_prob": mp_f.tolist(), "frac_pos": fp_f.tolist(), "bin_sizes": sz_f.tolist()},
            "gcn_graph":   {"mean_prob": mp_g.tolist(), "frac_pos": fp_g.tolist(), "bin_sizes": sz_g.tolist()},
            "egarch":      {"mean_prob": mp_e.tolist(), "frac_pos": fp_e.tolist(), "bin_sizes": sz_e.tolist()},
            "n_bins": 10,
            "note": "Equal-frequency (decile) bins",
        },
        "n_test_samples": int(len(y)),
        "event_rate":     round(float(y.mean()), 4),
    }

    with open(OUTPUT_JSON, "w") as f:
        json.dump(results, f, indent=2)
    print(f"JSON saved: {OUTPUT_JSON}")


if __name__ == "__main__":
    main()
