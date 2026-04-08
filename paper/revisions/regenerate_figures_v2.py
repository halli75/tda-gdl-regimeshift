"""
Regenerate all publication figures with improved aesthetics for camera-ready.

Fixes:
  - Figures 1 & 2 (PR-AUC bar, EGARCH comparison): value labels moved above
    error bar caps so they never intersect bars.
  - Figure 4 (walk-forward folds): wider figure, shorter labels, rotation fix.
  - Figures 6-8 & 10 (sensitivity + hybrid): enlarged, tighter layout,
    no overlapping text.

Run from repo root:
    cd <repo_root>
    python paper/revisions/regenerate_figures_v2.py
"""
from __future__ import annotations
import json, os, sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_curve, auc

REPO_ROOT   = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
FIGURES_DIR = os.path.join(REPO_ROOT, "paper", "figures")
REVISIONS   = os.path.join(REPO_ROOT, "paper", "revisions")
PREDICTIONS = os.path.join(REPO_ROOT, "research", "bootstrap_rerun", "final", "predictions.csv")

# ── Shared style ──────────────────────────────────────────────────────────────
plt.rcParams.update({
    "font.family":      "serif",
    "font.size":        9,
    "axes.titlesize":   9,
    "axes.labelsize":   8,
    "xtick.labelsize":  8,
    "ytick.labelsize":  8,
    "legend.fontsize":  7.5,
    "figure.dpi":       300,
    "savefig.dpi":      300,
    "savefig.bbox":     "tight",
    "axes.spines.top":  False,
    "axes.spines.right":False,
})

BLUE   = "#2e4057"
ORANGE = "#e07b39"
GREEN  = "#3a7d44"
GRAY   = "#888888"
RED    = "#c0392b"
PURPLE = "#7b3fa0"
BROWN  = "#7f4f24"

# ── Figure 1 (PDF Fig 1): PR-AUC comparison with bootstrap CIs ────────────────
def fig_prauc_comparison():
    """Horizontal bar chart for all 5 models. Labels to right of CI cap."""
    models  = ["gcn-fusion", "rf-combined", "gcn-graph", "rf-topology", "vol-threshold"]
    prauc   = [0.102, 0.105, 0.112, 0.082, 0.080]
    ci_lo   = [0.087, 0.084, 0.081, 0.066, 0.065]
    ci_hi   = [0.126, 0.132, 0.158, 0.106, 0.098]
    colors  = [BLUE, GREEN, GREEN, GRAY, ORANGE]
    random_baseline = 194 / 2376

    fig, ax = plt.subplots(figsize=(4.5, 3.0))
    y       = np.arange(len(models))
    xerr_lo = [p - l for p, l in zip(prauc, ci_lo)]
    xerr_hi = [h - p for p, h in zip(prauc, ci_hi)]

    ax.barh(y, prauc, xerr=[xerr_lo, xerr_hi], height=0.55,
            color=colors, alpha=0.85,
            error_kw=dict(ecolor="#333", linewidth=1.0, capsize=4))
    ax.axvline(random_baseline, color=RED, linestyle="--", linewidth=1.0,
               alpha=0.8, label=f"Random ({random_baseline:.3f})")
    ax.set_yticks(y)
    ax.set_yticklabels(models, fontsize=8)
    ax.set_xlabel("PR-AUC (test split, 95% bootstrap CI)")
    ax.set_xlim(0, 0.21)

    # Labels placed to the RIGHT of each CI upper bound (never on the bar)
    for i, (p, hi, c) in enumerate(zip(prauc, ci_hi, colors)):
        ax.text(hi + 0.004, i, f"{p:.3f}", va="center", fontsize=7.5,
                color=c, fontweight="bold")

    ax.legend(loc="lower right", frameon=False)
    fig.tight_layout()
    out = os.path.join(FIGURES_DIR, "figure2_prauc_comparison.pdf")
    fig.savefig(out)
    plt.close(fig)
    print(f"Saved: {out}")


# ── Figure 2 (PDF Fig 2): EGARCH comparison ───────────────────────────────────
def fig_egarch_comparison():
    """Vertical bar chart: EGARCH, gcn-fusion, vol-threshold. Two panels."""
    with open(os.path.join(REVISIONS, "m5_egarch_results.json")) as f:
        res = json.load(f)

    models = [
        ("EGARCH(1,1)",    res["egarch"]),
        ("gcn-fusion",     res["gcn_fusion"]),
        ("vol-threshold",  res["vol_threshold"]),
    ]
    colors = [GREEN, BLUE, ORANGE]

    fig, axes = plt.subplots(1, 2, figsize=(7, 3.6))

    for ax, (metric_label, key) in zip(axes, [("PR-AUC", "pr_auc"), ("Event F1", "event_f1")]):
        names  = [m[0] for m in models]
        vals   = [m[1][key] for m in models]
        x      = np.arange(len(names))

        bars = ax.bar(x, vals, color=colors, alpha=0.85,
                      edgecolor="black", linewidth=0.6, width=0.55)

        if key == "pr_auc":
            for i, (_, data) in enumerate(models):
                ci = data.get("ci_pr_auc", [vals[i], vals[i]])
                ax.errorbar(i, vals[i],
                            yerr=[[vals[i] - ci[0]], [ci[1] - vals[i]]],
                            fmt="none", color="black", capsize=5, linewidth=1.2)

        # Labels above bars — positioned above CI cap if present, else above bar
        y_top = max(vals)
        ax.set_ylim(0, y_top * 1.35)
        for i, (bar, val, (_, data)) in enumerate(zip(bars, vals, models)):
            if key == "pr_auc":
                ci   = data.get("ci_pr_auc", [val, val])
                label_y = ci[1] + y_top * 0.03
            else:
                label_y = val + y_top * 0.03
            ax.text(bar.get_x() + bar.get_width() / 2, label_y,
                    f"{val:.3f}", ha="center", va="bottom", fontsize=8,
                    fontweight="bold")

        ax.set_xticks(x)
        ax.set_xticklabels(names, fontsize=8.5)
        ax.set_title(f"{metric_label}{' (95% CI)' if key == 'pr_auc' else ''}",
                     fontsize=9)
        ax.set_ylabel(metric_label, fontsize=8)

    fig.suptitle("EGARCH(1,1) vs. vol-threshold vs. gcn-fusion — test split 2022–2026",
                 fontsize=9)
    plt.tight_layout()
    out = os.path.join(FIGURES_DIR, "figure_m5_egarch_comparison.pdf")
    fig.savefig(out)
    plt.close(fig)
    print(f"Saved: {out}")


# ── Figure 4 (PDF Fig 4): Walk-forward folds ─────────────────────────────────
def fig_walkforward_folds():
    """Grouped bar for 5 folds. Wider figure; short labels; no overlap."""
    fold_labels = ["F0\n(Jul'16\n22 ev.)", "F1\n(Jul'17\n50 ev.)",
                   "F2\n(Jul'18\n46 ev.)", "F3\n(Jul'19\n40 ev.)",
                   "F4\n(Jul'20\n13 ev.)"]
    gcn_f1 = [0.0889, 0.8183, 0.1832, 0.6154, 0.2609]
    vol_f1 = [0.3673, 0.5723, 0.4974, 0.2791, 0.3038]
    gcn_mean = np.mean(gcn_f1)
    vol_mean  = np.mean(vol_f1)

    x = np.arange(len(fold_labels))
    w = 0.35
    fig, ax = plt.subplots(figsize=(5.5, 3.2))
    ax.bar(x - w/2, gcn_f1, w,
           label=f"gcn-fusion (mean={gcn_mean:.3f})", color=BLUE, alpha=0.85)
    ax.bar(x + w/2, vol_f1, w,
           label=f"vol-threshold (mean={vol_mean:.3f})", color=ORANGE, alpha=0.85)
    ax.axhline(gcn_mean, color=BLUE, linestyle="--", linewidth=0.9, alpha=0.6)
    ax.axhline(vol_mean,  color=ORANGE, linestyle="--", linewidth=0.9, alpha=0.6)
    ax.set_xticks(x)
    ax.set_xticklabels(fold_labels, fontsize=8)
    ax.set_ylabel("Event F1 (validation fold)")
    ax.set_ylim(0, 1.0)
    ax.legend(loc="upper right", frameon=False)
    fig.tight_layout()
    out = os.path.join(FIGURES_DIR, "figure3_walkforward_folds.pdf")
    fig.savefig(out)
    plt.close(fig)
    print(f"Saved: {out}")


# ── Figure 3 (PDF Fig 3): PR curves ──────────────────────────────────────────
def fig_pr_curves():
    """PR curves with clearly separated x-axis ticks."""
    try:
        df       = pd.read_csv(PREDICTIONS)
        y_true   = df["label"].values
        s_gcn    = df["gcn_fusion_score"].values
        s_vol    = df["vol_threshold_score"].values

        prec_g, rec_g, thr_g = precision_recall_curve(y_true, s_gcn)
        prec_v, rec_v, thr_v = precision_recall_curve(y_true, s_vol)
        baseline = y_true.mean()

        # Operating points
        gcn_idx = min(int(np.searchsorted(thr_g, 0.55)), len(thr_g) - 1)
        n       = len(s_vol)
        vol_thr = np.sort(s_vol)[::-1][min(int(0.445 * n), n - 1)]
        vol_idx = min(int(np.searchsorted(thr_v, vol_thr)), len(thr_v) - 1)

        fig, ax = plt.subplots(figsize=(4.0, 3.2))
        ax.plot(rec_g, prec_g, color=BLUE, linewidth=1.5,
                label="gcn-fusion (AUC=0.102)")
        ax.plot(rec_v, prec_v, color=ORANGE, linewidth=1.5, linestyle="--",
                label="vol-threshold (AUC=0.080)")
        ax.plot(rec_g[gcn_idx], prec_g[gcn_idx], "*", color=BLUE,
                markersize=11, zorder=5, label="gcn op. pt (t=0.55)")
        ax.plot(rec_v[vol_idx], prec_v[vol_idx], "D", color=ORANGE,
                markersize=7, zorder=5, label="vol op. pt (44.5%)")
        ax.axhline(baseline, color=GRAY, linestyle=":", linewidth=1.0,
                   label=f"Random ({baseline:.3f})")
        ax.set_xlabel("Recall")
        ax.set_ylabel("Precision")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_xticks(np.arange(0, 1.1, 0.2))
        ax.set_yticks(np.arange(0, 1.1, 0.2))
        ax.legend(loc="upper right", frameon=False, fontsize=7)
        fig.tight_layout()
        out = os.path.join(FIGURES_DIR, "figure4_pr_curves.pdf")
        fig.savefig(out)
        plt.close(fig)
        print(f"Saved: {out}")
    except Exception as e:
        print(f"PR curves failed: {e}")


# ── Figure 6 (PDF Fig 6): Permutation importance ─────────────────────────────
def fig_permutation_importance():
    """Left: top-20 horizontal bars. Right: group sum bars. No overlap."""
    with open(os.path.join(REVISIONS, "m3_permutation_importance.json")) as f:
        data = json.load(f)

    top20 = data["top20_features"]
    groups = data["group_importance"]

    GROUP_COLORS = {
        "xcorr":     BLUE,
        "classical": GREEN,
        "topology":  PURPLE,
        "sym_onehot":GRAY,
    }

    fig, axes = plt.subplots(1, 2, figsize=(9, 4.5),
                             gridspec_kw={"width_ratios": [2, 1]})

    # Left: top-20 horizontal bars
    ax = axes[0]
    feats  = [f["feature"] for f in top20]
    imps   = [f["importance"] for f in top20]
    stds   = [f["std"] for f in top20]
    gc     = [GROUP_COLORS.get(f["group"], GRAY) for f in top20]
    y      = np.arange(len(feats))

    ax.barh(y, imps, xerr=stds, height=0.7, color=gc, alpha=0.85,
            error_kw=dict(ecolor="#333", linewidth=0.7, capsize=2))
    ax.set_yticks(y)
    ax.set_yticklabels(feats, fontsize=7.5)
    ax.invert_yaxis()
    ax.set_xlabel("Permutation importance\n(avg. precision, 10 repeats)", fontsize=8)
    ax.set_title("Top-20 Features (rf-combined, test split)", fontsize=9)

    # Right: group-level bar chart
    ax2   = axes[1]
    glabels = list(groups.keys())
    gsums   = [groups[g]["sum_importance"] for g in glabels]
    gcols   = [GROUP_COLORS.get(g, GRAY) for g in glabels]
    xg      = np.arange(len(glabels))

    bars = ax2.bar(xg, gsums, color=gcols, alpha=0.85,
                   edgecolor="black", linewidth=0.6, width=0.6)
    ax2.set_xticks(xg)
    ax2.set_xticklabels(glabels, fontsize=8, rotation=20, ha="right")
    ax2.set_ylabel("Group sum importance", fontsize=8)
    ax2.set_title("Feature Group Totals", fontsize=9)
    ymax = max(gsums)
    ax2.set_ylim(0, ymax * 1.30)
    for bar, val in zip(bars, gsums):
        ax2.text(bar.get_x() + bar.get_width() / 2,
                 val + ymax * 0.03,
                 f"{val:.4f}", ha="center", va="bottom", fontsize=8)

    fig.tight_layout(pad=1.5)
    out = os.path.join(FIGURES_DIR, "figure_m3_permutation_importance.pdf")
    fig.savefig(out)
    plt.close(fig)
    print(f"Saved: {out}")


# ── Figure 7 (PDF Fig 7): kNN sensitivity ────────────────────────────────────
def fig_knn_sensitivity():
    """2-bar comparison across 4 metrics. Ample headroom for labels."""
    with open(os.path.join(REVISIONS, "m2_knn_sensitivity.json")) as f:
        data = json.load(f)
    k5  = data["k5"]
    k12 = data["k12"]

    metrics = [
        ("PR-AUC",      "pr_auc",               True),
        ("Event F1",    "event_f1",              False),
        ("FA / day",    "false_alarms_per_day",  False),
        ("Lead (bars)", "mean_lead_bars",         False),
    ]

    fig, axes = plt.subplots(1, 4, figsize=(10, 3.8))
    for ax, (label, key, has_ci) in zip(axes, metrics):
        v5  = k5.get(key)  or 0.0
        v12 = k12.get(key) or 0.0
        vmax = max(v5, v12)

        bars = ax.bar(["k = 5", "k = 12"], [v5, v12],
                      color=[BLUE, ORANGE], width=0.5,
                      edgecolor="black", linewidth=0.6, alpha=0.85)

        if has_ci:
            ax.errorbar(0, v5,
                        yerr=[[v5  - k5["ci_pr_auc"][0]],  [k5["ci_pr_auc"][1]  - v5]],
                        fmt="none", color="black", capsize=5, linewidth=1.2)
            ax.errorbar(1, v12,
                        yerr=[[v12 - k12["ci_pr_auc"][0]], [k12["ci_pr_auc"][1] - v12]],
                        fmt="none", color="black", capsize=5, linewidth=1.2)
            ci_tops = [k5["ci_pr_auc"][1], k12["ci_pr_auc"][1]]
            label_ys = [ct + vmax * 0.05 for ct in ci_tops]
        else:
            label_ys = [v5 + vmax * 0.05, v12 + vmax * 0.05]

        ax.set_ylim(0, vmax * 1.45)
        for bar, val, ly in zip(bars, [v5, v12], label_ys):
            ax.text(bar.get_x() + bar.get_width() / 2, ly,
                    f"{val:.3f}", ha="center", va="bottom", fontsize=8,
                    fontweight="bold")

        ax.set_title(label, fontsize=9)
        ax.set_ylabel(label, fontsize=8)
        ax.tick_params(labelsize=8)

    fig.suptitle("gcn-fusion Sensitivity: kNN Graph Density (k=5 vs k=12)\n"
                 "Error bars: 95% block bootstrap CI on PR-AUC",
                 fontsize=9)
    plt.tight_layout()
    out = os.path.join(FIGURES_DIR, "figure_m2_knn_sensitivity.pdf")
    fig.savefig(out, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out}")


# ── Figure 8 (PDF Fig 8): Lookahead sensitivity ───────────────────────────────
def fig_lookahead_sensitivity():
    """Grouped bar for 5 models × 2 L values. Wider figure, rotated labels."""
    with open(os.path.join(REVISIONS, "m4_lookahead_sensitivity.json")) as f:
        data = json.load(f)
    L5   = data["L5"]
    L20  = data["L20"]

    MODEL_ORDER  = ["vol_threshold", "gcn_fusion", "rf_combined", "gcn_graph", "rf_topology"]
    MODEL_LABELS = {
        "vol_threshold": "vol-thr",
        "gcn_fusion":    "gcn-fusion",
        "rf_combined":   "rf-comb",
        "gcn_graph":     "gcn-graph",
        "rf_topology":   "rf-topo",
    }
    models_present = [m for m in MODEL_ORDER if m in L5 and m in L20]
    labels = [MODEL_LABELS[m] for m in models_present]
    x      = np.arange(len(models_present))
    width  = 0.35

    fig, axes = plt.subplots(1, 2, figsize=(9, 3.8))

    for ax, (title, key) in zip(axes, [("PR-AUC (95% bootstrap CI)", "pr_auc"),
                                        ("Event F1", "event_f1")]):
        vals_L5  = [L5[m][key]  for m in models_present]
        vals_L20 = [L20[m][key] for m in models_present]
        vmax     = max(max(vals_L5), max(vals_L20))

        b5  = ax.bar(x - width/2, vals_L5,  width, label="L = 5",
                     color=BLUE,   alpha=0.85, edgecolor="black", linewidth=0.5)
        b20 = ax.bar(x + width/2, vals_L20, width, label="L = 20",
                     color=ORANGE, alpha=0.85, edgecolor="black", linewidth=0.5)

        if key == "pr_auc":
            for i, m in enumerate(models_present):
                ci5  = L5[m]["ci_pr_auc"]
                ci20 = L20[m]["ci_pr_auc"]
                v5   = vals_L5[i]
                v20  = vals_L20[i]
                ax.errorbar(x[i] - width/2, v5,
                            yerr=[[v5 - ci5[0]], [ci5[1] - v5]],
                            fmt="none", color="black", capsize=3, linewidth=1)
                ax.errorbar(x[i] + width/2, v20,
                            yerr=[[v20 - ci20[0]], [ci20[1] - v20]],
                            fmt="none", color="black", capsize=3, linewidth=1)
            # Labels above CI cap
            for i, m in enumerate(models_present):
                ci5_hi  = L5[m]["ci_pr_auc"][1]
                ci20_hi = L20[m]["ci_pr_auc"][1]
                ax.text(x[i] - width/2, ci5_hi  + vmax * 0.04,
                        f"{vals_L5[i]:.3f}", ha="center", va="bottom", fontsize=7)
                ax.text(x[i] + width/2, ci20_hi + vmax * 0.04,
                        f"{vals_L20[i]:.3f}", ha="center", va="bottom", fontsize=7)
        else:
            for bar in list(b5) + list(b20):
                h = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2, h + vmax * 0.03,
                        f"{h:.3f}", ha="center", va="bottom", fontsize=7)

        ax.set_ylim(0, vmax * 1.50)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=20, ha="right", fontsize=8.5)
        ax.set_title(title, fontsize=9)
        ax.set_ylabel(title.split(" (")[0], fontsize=8)
        ax.legend(fontsize=8, frameon=False)

    fig.suptitle("Lookahead Sensitivity: L=5 vs L=20 — all models, test split 2022–2026",
                 fontsize=9)
    plt.tight_layout()
    out = os.path.join(FIGURES_DIR, "figure_m4_lookahead_sensitivity.pdf")
    fig.savefig(out, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out}")


# ── Figure 10 (PDF Fig 10): Hybrid detector ──────────────────────────────────
def fig_hybrid_detector():
    """Bar chart for 5 models × 2 metrics. Clean labels above CI caps."""
    with open(os.path.join(REVISIONS, "hybrid_detector_results.json")) as f:
        res = json.load(f)

    models = [
        ("EGARCH(1,1)",    res["egarch"]),
        ("Ensemble\n50/50", res["ensemble_50_50"]),
        ("EGARCH-\ngated",  res["egarch_gated"]),
        ("gcn-fusion",     res["gcn_fusion"]),
        ("vol-threshold",  res["vol_threshold"]),
    ]
    colors = [GREEN, PURPLE, BROWN, BLUE, ORANGE]

    fig, axes = plt.subplots(1, 2, figsize=(9, 3.8))

    for ax, (metric_label, key) in zip(axes, [("PR-AUC", "pr_auc"), ("Event F1", "event_f1")]):
        names  = [m[0] for m in models]
        vals   = [m[1].get(key, m[1].get("pr_auc", 0)) for m in models]
        x      = np.arange(len(names))
        vmax   = max(vals)

        bars = ax.bar(x, vals, color=colors, alpha=0.85,
                      edgecolor="black", linewidth=0.6, width=0.55)

        if key == "pr_auc":
            ci_tops = []
            for i, (_, data) in enumerate(models):
                ci  = data.get("ci_pr_auc", [vals[i], vals[i]])
                ax.errorbar(i, vals[i],
                            yerr=[[vals[i] - ci[0]], [ci[1] - vals[i]]],
                            fmt="none", color="black", capsize=4, linewidth=1.2)
                ci_tops.append(ci[1])
            label_ys = [ct + vmax * 0.04 for ct in ci_tops]
        else:
            label_ys = [v + vmax * 0.04 for v in vals]

        ax.set_ylim(0, vmax * 1.45)
        for i, (bar, val, ly) in enumerate(zip(bars, vals, label_ys)):
            ax.text(bar.get_x() + bar.get_width() / 2, ly,
                    f"{val:.3f}", ha="center", va="bottom", fontsize=8,
                    fontweight="bold")

        ax.set_xticks(x)
        ax.set_xticklabels(names, rotation=10, ha="right", fontsize=8.5)
        ax.set_title(f"{metric_label}{' (95% CI)' if key == 'pr_auc' else ''}",
                     fontsize=9)
        ax.set_ylabel(metric_label, fontsize=8)

    fig.suptitle("Hybrid EGARCH-GCN Detector vs. Standalone Models — test split 2022–2026",
                 fontsize=9)
    plt.tight_layout()
    out = os.path.join(FIGURES_DIR, "figure_hybrid_detector.pdf")
    fig.savefig(out, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out}")


if __name__ == "__main__":
    print("Regenerating figures (v2 improved aesthetics) ...")
    fig_prauc_comparison()
    fig_egarch_comparison()
    fig_walkforward_folds()
    fig_pr_curves()
    fig_permutation_importance()
    fig_knn_sensitivity()
    fig_lookahead_sensitivity()
    fig_hybrid_detector()
    print("Done.")
