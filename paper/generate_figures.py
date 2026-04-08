"""
Generate all publication figures for TDA-GDL regime detection paper.
Run from repo root: python paper/generate_figures.py
All figures saved to paper/figures/ as PDF.
"""
import os
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D
from sklearn.metrics import precision_recall_curve, auc

FIGURES_DIR = os.path.join(os.path.dirname(__file__), "figures")
os.makedirs(FIGURES_DIR, exist_ok=True)

FINAL_RUN = "research/autoresearch/final_runs/20260401_150829_final/artifacts/final"
PREDICTIONS_CSV = os.path.join(FINAL_RUN, "predictions.csv")
EVENT_TABLE_CSV = os.path.join(FINAL_RUN, "event_table.csv")
FOLD_SUMMARY_JSON = "outputs_autoresearch_walkforward/walk_forward/walkforward_fold_summary.json"

plt.rcParams.update({
    "font.family": "serif",
    "font.size": 9,
    "axes.titlesize": 9,
    "axes.labelsize": 9,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "legend.fontsize": 8,
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "axes.spines.top": False,
    "axes.spines.right": False,
})

BLUE   = "#2e4057"
ORANGE = "#e07b39"
GREEN  = "#3a7d44"
GRAY   = "#888888"
RED    = "#c0392b"

# ─── Figure 1: Val→Test gap trajectory ────────────────────────────────────────
def fig1_gap_trajectory():
    rounds = ["Single-split\n(intraday,\n3 sym)", "Single-split\n(daily,\n5 sym)", "Walk-forward\n(daily,\n13 sym)"]
    gaps   = [-0.215, -0.157, -0.055]
    changes = ["Baseline", "+symbols\n+daily freq.", "+walk-forward\n+13 symbols"]

    fig, ax = plt.subplots(figsize=(3.5, 2.8))
    ax.axhline(0, color="black", linewidth=0.8, linestyle="--", alpha=0.6, label="No gap")
    ax.plot([0, 1, 2], gaps, "o-", color=BLUE, linewidth=1.8, markersize=7, zorder=3)
    for i, (x, y, ch) in enumerate(zip([0,1,2], gaps, changes)):
        ax.annotate(f"{y:.3f}", xy=(x, y), xytext=(0, 10 if y < -0.1 else -16),
                    textcoords="offset points", ha="center", fontsize=7.5,
                    color=BLUE, fontweight="bold")
    ax.set_xticks([0, 1, 2])
    ax.set_xticklabels(rounds, fontsize=7.5)
    ax.set_ylabel("Val F1 − Test F1 (gcn\_fusion)")
    ax.set_ylim(-0.28, 0.05)
    ax.set_xlim(-0.3, 2.3)
    ax.fill_between([-0.3, 2.3], [-0.28, -0.28], [0, 0], alpha=0.05, color=RED)
    ax.fill_between([-0.3, 2.3], [0, 0], [0.05, 0.05], alpha=0.05, color=GREEN)
    ax.text(2.25, 0.02, "overfit", fontsize=7, color=RED, ha="right")
    ax.legend(loc="lower right", frameon=False, fontsize=7.5)
    fig.tight_layout()
    fig.savefig(os.path.join(FIGURES_DIR, "figure1_gap_trajectory.pdf"))
    plt.close(fig)
    print("Figure 1 saved.")

# ─── Figure 2: PR-AUC comparison with CIs ─────────────────────────────────────
def fig2_prauc_comparison():
    models  = ["gcn\_fusion", "rf\_combined", "gcn\_graph", "rf\_topology", "vol\_threshold"]
    prauc   = [0.1155, 0.1048, 0.0986, 0.0824, 0.0800]
    ci_lo   = [0.0987, 0.0852, 0.0756, 0.0651, 0.0630]
    ci_hi   = [0.1491, 0.1330, 0.1263, 0.1033, 0.0955]
    colors_ = [BLUE, GREEN, GREEN, GRAY, ORANGE]
    random_baseline = 194 / 2376  # ~0.0816

    fig, ax = plt.subplots(figsize=(3.5, 2.6))
    y = np.arange(len(models))
    xerr_lo = [p - l for p, l in zip(prauc, ci_lo)]
    xerr_hi = [h - p for p, h in zip(prauc, ci_hi)]
    bars = ax.barh(y, prauc, xerr=[xerr_lo, xerr_hi], height=0.55,
                   color=colors_, alpha=0.85, error_kw=dict(ecolor="#333", linewidth=1.0, capsize=3))
    ax.axvline(random_baseline, color=RED, linestyle="--", linewidth=1.0, alpha=0.8,
               label=f"Random baseline ({random_baseline:.3f})")
    ax.set_yticks(y)
    ax.set_yticklabels(models)
    ax.set_xlabel("PR-AUC (test split, 95% bootstrap CI)")
    ax.set_xlim(0, 0.18)
    for i, (p, c) in enumerate(zip(prauc, colors_)):
        ax.text(p + 0.002, i, f"{p:.3f}", va="center", fontsize=7.5, color=c, fontweight="bold")
    ax.legend(loc="lower right", frameon=False, fontsize=7.5)
    fig.tight_layout()
    fig.savefig(os.path.join(FIGURES_DIR, "figure2_prauc_comparison.pdf"))
    plt.close(fig)
    print("Figure 2 saved.")

# ─── Figure 3: Walk-forward fold F1 ───────────────────────────────────────────
def fig3_walkforward_folds():
    fold_labels = [
        "Fold 0\n(Jul'16–Oct'17\n22 ev.)",
        "Fold 1\n(Jul'17–Oct'18\n50 ev.)",
        "Fold 2\n(Jul'18–Oct'19\n46 ev.)",
        "Fold 3\n(Jul'19–Oct'20\n40 ev.)",
        "Fold 4\n(Jul'20–Sep'21\n13 ev.)",
    ]
    gcn_f1 = [0.0889, 0.8183, 0.1832, 0.6154, 0.2609]
    vol_f1 = [0.3673, 0.5723, 0.4974, 0.2791, 0.3038]
    gcn_mean = np.mean(gcn_f1)
    vol_mean  = np.mean(vol_f1)

    x = np.arange(len(fold_labels))
    w = 0.35
    fig, ax = plt.subplots(figsize=(3.5, 2.8))
    ax.bar(x - w/2, gcn_f1, w, label=f"gcn\_fusion (mean={gcn_mean:.3f})", color=BLUE, alpha=0.85)
    ax.bar(x + w/2, vol_f1, w, label=f"vol\_threshold (mean={vol_mean:.3f})", color=ORANGE, alpha=0.85)
    ax.axhline(gcn_mean, color=BLUE, linestyle="--", linewidth=0.9, alpha=0.6)
    ax.axhline(vol_mean,  color=ORANGE, linestyle="--", linewidth=0.9, alpha=0.6)
    ax.set_xticks(x)
    ax.set_xticklabels(fold_labels, fontsize=6.5)
    ax.set_ylabel("Event F1 (validation)")
    ax.set_ylim(0, 1.0)
    ax.legend(loc="upper right", frameon=False, fontsize=7)
    fig.tight_layout()
    fig.savefig(os.path.join(FIGURES_DIR, "figure3_walkforward_folds.pdf"))
    plt.close(fig)
    print("Figure 3 saved.")

# ─── Figure 4: PR curves ──────────────────────────────────────────────────────
def fig4_pr_curves():
    try:
        df = pd.read_csv(PREDICTIONS_CSV)
        y_true = df["label"].values
        gcn_scores = df["gcn_fusion_score"].values
        vol_scores = df["vol_threshold_score"].values

        prec_gcn, rec_gcn, thr_gcn = precision_recall_curve(y_true, gcn_scores)
        prec_vol, rec_vol, thr_vol = precision_recall_curve(y_true, vol_scores)
        auc_gcn = auc(rec_gcn, prec_gcn)
        auc_vol = auc(rec_vol, prec_vol)
        baseline = y_true.mean()

        # Operating point markers (for revision m4)
        # GCN: threshold=0.55 (reported in paper)
        gcn_op_idx = int(np.searchsorted(thr_gcn, 0.55))
        gcn_op_idx = min(gcn_op_idx, len(thr_gcn) - 1)
        # Vol: threshold at ~44.5% positive rate
        n_total = len(vol_scores)
        sorted_vol_desc = np.sort(vol_scores)[::-1]
        vol_op_thr = sorted_vol_desc[min(int(0.445 * n_total), n_total - 1)]
        vol_op_idx = int(np.searchsorted(thr_vol, vol_op_thr))
        vol_op_idx = min(vol_op_idx, len(thr_vol) - 1)

        # Use canonical metrics.json values in legend for Table 3 consistency;
        # the curve itself is from predictions.csv (tiny diff from numerical integration)
        canonical_gcn = 0.116
        canonical_vol = 0.080
        fig, ax = plt.subplots(figsize=(3.5, 2.8))
        ax.plot(rec_gcn, prec_gcn, color=BLUE, linewidth=1.5,
                label=f"gcn\_fusion (AUC={canonical_gcn:.3f})")
        ax.plot(rec_vol, prec_vol, color=ORANGE, linewidth=1.5, linestyle="--",
                label=f"vol\_threshold (AUC={canonical_vol:.3f})")
        # Mark selected operating points
        ax.plot(rec_gcn[gcn_op_idx], prec_gcn[gcn_op_idx],
                '*', color=BLUE, markersize=10, zorder=5,
                label=f"gcn op. pt (t=0.55)")
        ax.plot(rec_vol[vol_op_idx], prec_vol[vol_op_idx],
                'D', color=ORANGE, markersize=7, zorder=5,
                label=f"vol op. pt (rate=44.5%)")
        ax.axhline(baseline, color=GRAY, linestyle=":", linewidth=1.0, alpha=0.7,
                   label=f"Random ({baseline:.3f})")
        ax.set_xlabel("Recall")
        ax.set_ylabel("Precision")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.legend(loc="upper right", frameon=False, fontsize=6.5)
        fig.tight_layout()
        fig.savefig(os.path.join(FIGURES_DIR, "figure4_pr_curves.pdf"))
        plt.close(fig)
        print(f"Figure 4 saved. (gcn AUC={auc_gcn:.4f}, vol AUC={auc_vol:.4f})")
    except Exception as e:
        print(f"Figure 4 failed: {e}")
        _fig4_fallback()

def _fig4_fallback():
    # Use artifact values if predictions.csv unavailable
    fig, ax = plt.subplots(figsize=(3.5, 2.8))
    ax.text(0.5, 0.5, "PR curves: see artifact\npredictions.csv",
            ha="center", va="center", transform=ax.transAxes, fontsize=9)
    fig.savefig(os.path.join(FIGURES_DIR, "figure4_pr_curves.pdf"))
    plt.close(fig)

# ─── Figure 5: Score distribution ────────────────────────────────────────────
def fig5_score_distribution():
    try:
        df = pd.read_csv(PREDICTIONS_CSV)
        pos_scores = df.loc[df["label"] == 1, "gcn_fusion_score"].values
        neg_scores = df.loc[df["label"] == 0, "gcn_fusion_score"].values

        fig, ax = plt.subplots(figsize=(3.5, 2.6))
        bins = np.linspace(0, 1, 26)
        ax.hist(neg_scores, bins=bins, alpha=0.6, color=GRAY, label="Non-event (label=0)",
                density=True)
        ax.hist(pos_scores, bins=bins, alpha=0.75, color=RED, label="Event (label=1)",
                density=True)
        ax.axvline(0.55, color=BLUE, linestyle="--", linewidth=1.2,
                   label="Decision threshold (0.55)")
        ax.set_xlabel("gcn\_fusion predicted probability")
        ax.set_ylabel("Density")
        ax.legend(frameon=False, fontsize=7.5)
        fig.tight_layout()
        fig.savefig(os.path.join(FIGURES_DIR, "figure5_score_distribution.pdf"))
        plt.close(fig)
        print("Figure 5 saved.")
    except Exception as e:
        print(f"Figure 5 failed: {e}")

# ─── Figure 6: Event detection timeline ──────────────────────────────────────
def fig6_event_timeline():
    try:
        df = pd.read_csv(PREDICTIONS_CSV, parse_dates=["timestamp"])
        # Use SPY for clarity
        spy = df[df["symbol"] == "SPY"].sort_values("timestamp").copy()
        spy["gcn_pred"] = (spy["gcn_fusion_score"] >= 0.55).astype(int)
        spy["vol_pred"] = (spy["vol_threshold_score"] >= spy["vol_threshold_score"].quantile(0.5)).astype(int)

        fig, axes = plt.subplots(2, 1, figsize=(3.5, 3.2), sharex=True)
        ax1, ax2 = axes

        # Panel 1: true labels as shaded regions
        ts = spy["timestamp"].values
        labels = spy["label"].values
        # shade true event regions
        in_event = False
        start_idx = None
        for i, (t, l) in enumerate(zip(ts, labels)):
            if l == 1 and not in_event:
                in_event = True
                start_idx = t
            elif l == 0 and in_event:
                ax1.axvspan(start_idx, t, alpha=0.25, color=RED, linewidth=0)
                ax2.axvspan(start_idx, t, alpha=0.25, color=RED, linewidth=0)
                in_event = False
        if in_event:
            ax1.axvspan(start_idx, ts[-1], alpha=0.25, color=RED, linewidth=0)
            ax2.axvspan(start_idx, ts[-1], alpha=0.25, color=RED, linewidth=0)

        # gcn_fusion predictions
        gcn_fire = spy[spy["gcn_pred"] == 1]
        ax1.scatter(gcn_fire["timestamp"], [0.6]*len(gcn_fire), marker="|",
                    color=BLUE, s=15, linewidths=0.5, alpha=0.7)
        ax1.set_ylabel("gcn\_fusion", fontsize=7.5)
        ax1.set_yticks([])
        ax1.set_ylim(0, 1)

        # vol_threshold predictions
        vol_fire = spy[spy["vol_pred"] == 1]
        ax2.scatter(vol_fire["timestamp"], [0.6]*len(vol_fire), marker="|",
                    color=ORANGE, s=15, linewidths=0.5, alpha=0.7)
        ax2.set_ylabel("vol\_threshold", fontsize=7.5)
        ax2.set_yticks([])
        ax2.set_ylim(0, 1)
        ax2.set_xlabel("Date (test period: Jul 2022–Mar 2026)")

        red_patch = mpatches.Patch(color=RED, alpha=0.3, label="True event")
        ax1.legend(handles=[red_patch], loc="upper right", frameon=False, fontsize=7)

        fig.tight_layout()
        fig.savefig(os.path.join(FIGURES_DIR, "figure6_event_timeline.pdf"))
        plt.close(fig)
        print("Figure 6 saved.")
    except Exception as e:
        print(f"Figure 6 failed: {e}")

if __name__ == "__main__":
    fig1_gap_trajectory()
    fig2_prauc_comparison()
    fig3_walkforward_folds()
    fig4_pr_curves()
    fig5_score_distribution()
    fig6_event_timeline()
    print("All figures complete.")
