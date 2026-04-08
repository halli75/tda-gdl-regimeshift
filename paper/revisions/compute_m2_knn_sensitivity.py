"""
M2 kNN Graph Density Sensitivity: gcn_fusion with k=5 vs k=12.

Loads metrics from:
  - k=12 (best config): research/bootstrap_rerun/final/metrics.json
  - k=5  (sensitivity):  research/sensitivity/m2_knn_k5/final/metrics.json

Extracts gcn_fusion metrics: pr_auc, event_f1, false_alarms_per_day,
mean_lead_bars, bootstrap_ci.pr_auc.

Outputs:
  paper/revisions/m2_knn_sensitivity.json
  paper/figures/figure_m2_knn_sensitivity.pdf

Usage:
    cd <repo_root>
    python paper/revisions/compute_m2_knn_sensitivity.py
"""

import json
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

METRICS_K12 = os.path.join(REPO_ROOT, "research", "bootstrap_rerun", "final", "metrics.json")
METRICS_K5  = os.path.join(REPO_ROOT, "research", "sensitivity", "m2_knn_k5", "final", "metrics.json")
OUTPUT_JSON = os.path.join(REPO_ROOT, "paper", "revisions", "m2_knn_sensitivity.json")
OUTPUT_FIG  = os.path.join(REPO_ROOT, "paper", "figures", "figure_m2_knn_sensitivity.pdf")


def extract_gcn_fusion(metrics_path: str) -> dict:
    with open(metrics_path) as f:
        m = json.load(f)
    for model in m["models"]:
        if model["model"] == "gcn_fusion":
            lead = model.get("mean_lead_bars")
            return {
                "pr_auc":               round(float(model["pr_auc"]), 4),
                "event_f1":             round(float(model["event_f1"]), 4),
                "false_alarms_per_day": round(float(model["false_alarms_per_day"]), 4),
                "mean_lead_bars":       round(float(lead), 2) if lead and not (isinstance(lead, float) and np.isnan(lead)) else None,
                "ci_pr_auc":            [round(x, 4) for x in model["bootstrap_ci"]["pr_auc"]],
            }
    raise ValueError(f"gcn_fusion not found in {metrics_path}")


def build_interpretation(k5: dict, k12: dict) -> str:
    delta = k12["pr_auc"] - k5["pr_auc"]
    pct   = delta / k5["pr_auc"] * 100 if k5["pr_auc"] > 0 else 0.0
    ci_overlap = k5["ci_pr_auc"][1] >= k12["ci_pr_auc"][0]

    if abs(delta) < 0.005:
        verdict = (
            "The GCN appears to function primarily as a feature aggregator: "
            f"PR-AUC changes by only {delta:+.4f} ({pct:+.1f}%) from k=5 to k=12, "
            "well within bootstrap CI overlap. Graph density has minimal impact on "
            "probability-ranking ability, suggesting the model is not exploiting "
            "fine-grained geometric structure of the delay-embedded point cloud."
        )
    elif delta > 0:
        verdict = (
            f"PR-AUC improves by {delta:+.4f} ({pct:+.1f}%) from k=5 to k=12. "
            "The denser graph (k=12, density ~46%) yields better probability ranking "
            "than the sparse graph (k=5, density ~19%), consistent with the GCN "
            "exploiting geometric structure rather than acting as a pure feature aggregator. "
            + ("Bootstrap CIs overlap, so the improvement is indicative but not statistically "
               "significant at 95%." if ci_overlap else
               "Bootstrap CIs do not overlap, providing statistical evidence for the improvement.")
        )
    else:
        verdict = (
            f"PR-AUC decreases by {delta:+.4f} ({pct:+.1f}%) from k=5 to k=12. "
            "The sparser graph (k=5) achieves comparable or better probability ranking, "
            "suggesting over-connectivity at k=12 may introduce noise. The GCN may be "
            "functioning primarily as a feature aggregator regardless of graph density."
        )
    return verdict


def make_figure(k5: dict, k12: dict, out_path: str) -> None:
    BLUE   = "#1f77b4"
    ORANGE = "#ff7f0e"

    metrics = [
        ("PR-AUC",     "pr_auc",               1),
        ("Event F1",   "event_f1",              1),
        ("FA / day",   "false_alarms_per_day",  1),
        ("Lead (bars)","mean_lead_bars",         1),
    ]

    fig, axes = plt.subplots(1, 4, figsize=(11, 3.5))

    for ax, (label, key, _) in zip(axes, metrics):
        v5  = k5[key]  if k5[key]  is not None else 0.0
        v12 = k12[key] if k12[key] is not None else 0.0

        bars = ax.bar(["k = 5", "k = 12"], [v5, v12],
                      color=[BLUE, ORANGE], width=0.5, edgecolor="black", linewidth=0.6)

        # Error bars only for PR-AUC
        if key == "pr_auc":
            ax.errorbar(0, v5,  yerr=[[v5  - k5["ci_pr_auc"][0]],  [k5["ci_pr_auc"][1]  - v5]],
                        fmt="none", color="black", capsize=4, linewidth=1.2)
            ax.errorbar(1, v12, yerr=[[v12 - k12["ci_pr_auc"][0]], [k12["ci_pr_auc"][1] - v12]],
                        fmt="none", color="black", capsize=4, linewidth=1.2)

        ax.set_title(label, fontsize=10)
        ax.set_ylabel(label, fontsize=8)
        ax.tick_params(labelsize=8)

        # Value annotations
        for bar, val in zip(bars, [v5, v12]):
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + max(v5, v12) * 0.02,
                    f"{val:.3f}", ha="center", va="bottom", fontsize=7.5)

    fig.suptitle("gcn-fusion Sensitivity: kNN Graph Density (k=5 vs k=12)\n"
                 "Error bars: 95% block bootstrap CI (PR-AUC only)",
                 fontsize=9, y=1.02)
    plt.tight_layout()
    plt.savefig(out_path, bbox_inches="tight")
    plt.close()
    print(f"Figure saved: {out_path}")


if __name__ == "__main__":
    print("Loading k=12 metrics ...")
    k12 = extract_gcn_fusion(METRICS_K12)
    print("Loading k=5  metrics ...")
    k5  = extract_gcn_fusion(METRICS_K5)

    interpretation = build_interpretation(k5, k12)

    results = {
        "k5":  k5,
        "k12": k12,
        "delta_pr_auc": round(k12["pr_auc"] - k5["pr_auc"], 4),
        "pct_change_pr_auc": round((k12["pr_auc"] - k5["pr_auc"]) / k5["pr_auc"] * 100, 1) if k5["pr_auc"] > 0 else None,
        "graph_density_k5":  "~19%",
        "graph_density_k12": "~46%",
        "interpretation": interpretation,
    }

    os.makedirs(os.path.dirname(OUTPUT_JSON), exist_ok=True)
    with open(OUTPUT_JSON, "w") as f:
        json.dump(results, f, indent=2)
    print(f"JSON saved: {OUTPUT_JSON}")

    print("\n--- k=5 ---")
    for k, v in k5.items():
        print(f"  {k}: {v}")
    print("--- k=12 ---")
    for k, v in k12.items():
        print(f"  {k}: {v}")
    print(f"\nDelta PR-AUC: {results['delta_pr_auc']:+.4f} ({results['pct_change_pr_auc']:+.1f}%)")
    print(f"Interpretation:\n  {interpretation}")

    make_figure(k5, k12, OUTPUT_FIG)
