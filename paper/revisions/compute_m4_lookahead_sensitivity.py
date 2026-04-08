"""
M4 Lookahead Horizon Sensitivity: L=5 vs L=20.

Loads metrics from:
  - L=5  (best config): research/bootstrap_rerun/final/metrics.json
  - L=20 (sensitivity): research/sensitivity/m4_lookahead_l20/final/metrics.json

Extracts per-model: pr_auc, event_f1, event_recall, false_alarms_per_day,
bootstrap_ci.pr_auc, and true_event_count (to note label-set change).

Reports whether gcn_fusion PR-AUC advantage over vol_threshold persists at L=20.

Outputs:
  paper/revisions/m4_lookahead_sensitivity.json
  paper/figures/figure_m4_lookahead_sensitivity.pdf

Usage:
    cd <repo_root>
    python paper/revisions/compute_m4_lookahead_sensitivity.py
"""

import json
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

METRICS_L5  = os.path.join(REPO_ROOT, "research", "bootstrap_rerun", "final", "metrics.json")
METRICS_L20 = os.path.join(REPO_ROOT, "research", "sensitivity", "m4_lookahead_l20", "final", "metrics.json")
OUTPUT_JSON = os.path.join(REPO_ROOT, "paper", "revisions", "m4_lookahead_sensitivity.json")
OUTPUT_FIG  = os.path.join(REPO_ROOT, "paper", "figures", "figure_m4_lookahead_sensitivity.pdf")

MODEL_ORDER  = ["vol_threshold", "gcn_fusion", "rf_combined", "gcn_graph", "rf_topology"]
MODEL_LABELS = {
    "vol_threshold": "vol-threshold",
    "gcn_fusion":    "gcn-fusion",
    "rf_combined":   "rf-combined",
    "gcn_graph":     "gcn-graph",
    "rf_topology":   "rf-topology",
}


def extract_all_models(metrics_path: str) -> dict:
    with open(metrics_path) as f:
        m = json.load(f)
    result = {}
    for model in m["models"]:
        name = model["model"]
        lead = model.get("mean_lead_bars")
        result[name] = {
            "pr_auc":               round(float(model["pr_auc"]), 4),
            "event_f1":             round(float(model["event_f1"]), 4),
            "event_recall":         round(float(model["event_recall"]), 4),
            "false_alarms_per_day": round(float(model["false_alarms_per_day"]), 4),
            "true_event_count":     int(model.get("true_event_count", 0)),
            "ci_pr_auc":            [round(x, 4) for x in model["bootstrap_ci"]["pr_auc"]],
            "mean_lead_bars":       round(float(lead), 2) if lead and not (isinstance(lead, float) and np.isnan(lead)) else None,
        }
    return result


def build_interpretation(L5: dict, L20: dict) -> tuple[bool, str]:
    gcn_L5  = L5.get("gcn_fusion",    {})
    vol_L5  = L5.get("vol_threshold", {})
    gcn_L20 = L20.get("gcn_fusion",   {})
    vol_L20 = L20.get("vol_threshold",{})

    adv_L5  = gcn_L5.get("pr_auc",  0) - vol_L5.get("pr_auc",  0)
    adv_L20 = gcn_L20.get("pr_auc", 0) - vol_L20.get("pr_auc", 0)
    persists = adv_L20 > 0

    events_L5  = gcn_L5.get("true_event_count",  "?")
    events_L20 = gcn_L20.get("true_event_count", "?")

    persist_str = "persists" if persists else "does not persist"
    adv_str = (
        f"gcn-fusion PR-AUC advantage over vol-threshold is "
        f"{adv_L5:+.4f} at L=5 and {adv_L20:+.4f} at L=20 ({persist_str})."
    )

    if persists:
        interp = (
            f"{adv_str} "
            f"The topological GCN approach is robust to lookahead horizon: "
            f"superior probability ranking over the rolling volatility baseline "
            f"holds for both short-horizon (L=5, ~1 week) and longer-horizon "
            f"(L=20, ~4 weeks) regime-transition definitions. "
            f"Note that the label set changes substantially between L values "
            f"(test events: L=5 n={events_L5}, L=20 n={events_L20}), "
            f"so direct metric comparisons across L are indicative only."
        )
    else:
        interp = (
            f"{adv_str} "
            f"The PR-AUC advantage of gcn-fusion over vol-threshold does not persist "
            f"at L=20, suggesting the geometric signal is specific to short-horizon "
            f"(~1-week) volatility regime transitions. At L=20 (~4 weeks), the "
            f"rolling volatility baseline achieves equal or superior probability ranking. "
            f"Note that the label set changes between L values "
            f"(test events: L=5 n={events_L5}, L=20 n={events_L20})."
        )
    return persists, interp


def make_figure(L5: dict, L20: dict, out_path: str) -> None:
    BLUE   = "#1f77b4"
    ORANGE = "#ff7f0e"

    models_present = [m for m in MODEL_ORDER if m in L5 and m in L20]
    labels = [MODEL_LABELS.get(m, m) for m in models_present]
    x = np.arange(len(models_present))
    width = 0.35

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))

    for ax, (title, key) in zip(axes, [("PR-AUC (95% bootstrap CI)", "pr_auc"),
                                        ("Event F1", "event_f1")]):
        vals_L5  = [L5[m][key]  for m in models_present]
        vals_L20 = [L20[m][key] for m in models_present]

        b5  = ax.bar(x - width/2, vals_L5,  width, label="L = 5",  color=BLUE,   alpha=0.85, edgecolor="black", linewidth=0.5)
        b20 = ax.bar(x + width/2, vals_L20, width, label="L = 20", color=ORANGE, alpha=0.85, edgecolor="black", linewidth=0.5)

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

        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=15, ha="right", fontsize=8)
        ax.set_title(title, fontsize=10)
        ax.set_ylabel(title.split(" (")[0], fontsize=8)
        ax.legend(fontsize=8)
        ax.tick_params(labelsize=8)

        # Value labels on bars
        for bar in list(b5) + list(b20):
            h = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2, h + 0.005,
                    f"{h:.3f}", ha="center", va="bottom", fontsize=6.5)

    fig.suptitle("Lookahead Sensitivity: L=5 vs L=20\n"
                 "All models; test split 2022–2026",
                 fontsize=9, y=1.02)
    plt.tight_layout()
    plt.savefig(out_path, bbox_inches="tight")
    plt.close()
    print(f"Figure saved: {out_path}")


if __name__ == "__main__":
    print("Loading L=5  metrics ...")
    L5  = extract_all_models(METRICS_L5)
    print("Loading L=20 metrics ...")
    L20 = extract_all_models(METRICS_L20)

    persists, interpretation = build_interpretation(L5, L20)

    results = {
        "L5":  L5,
        "L20": L20,
        "gcn_advantage_persists": persists,
        "gcn_fusion_delta_pr_auc": {
            "L5":  round(L5.get("gcn_fusion", {}).get("pr_auc", 0) - L5.get("vol_threshold", {}).get("pr_auc", 0), 4),
            "L20": round(L20.get("gcn_fusion", {}).get("pr_auc", 0) - L20.get("vol_threshold", {}).get("pr_auc", 0), 4),
        },
        "interpretation": interpretation,
    }

    os.makedirs(os.path.dirname(OUTPUT_JSON), exist_ok=True)
    with open(OUTPUT_JSON, "w") as f:
        json.dump(results, f, indent=2)
    print(f"JSON saved: {OUTPUT_JSON}")

    print("\n--- PR-AUC comparison ---")
    for m in MODEL_ORDER:
        if m in L5 and m in L20:
            print(f"  {m:20s}  L5={L5[m]['pr_auc']:.4f}  L20={L20[m]['pr_auc']:.4f}")
    print(f"\ngcn_advantage_persists: {persists}")
    print(f"Interpretation:\n  {interpretation}")

    make_figure(L5, L20, OUTPUT_FIG)
