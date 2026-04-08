"""
M3 Permutation Feature Importance.

Rebuilds the full feature frame from raw data (same pipeline as rf_combined),
then fits a RandomForestClassifier on train split and evaluates permutation
importance on the test split.

Groups features by:
  topology  — columns starting with "topo_", "betti_", "pi_"
  classical — columns starting with "cls_"
  xcorr     — columns starting with "xcorr_"
  sym_onehot — columns starting with "sym_"

Reports top-20 individual features and group-level mean importance.

Outputs:
  paper/revisions/m3_permutation_importance.json
  paper/figures/figure_m3_permutation_importance.pdf

Usage:
    cd <repo_root>
    PYTHONPATH=src python paper/revisions/compute_m3_permutation_importance.py
"""

from __future__ import annotations

import dataclasses
import json
import os
import sys
import warnings

import numpy as np
import pandas as pd
import yaml
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance

# ── repo root ------------------------------------------------------------------
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(REPO_ROOT, "src"))

from tda_gdl_regime.config import DataConfig, LabelConfig, FeatureConfig       # noqa: E402
from tda_gdl_regime.data_pipeline import load_market_data                       # noqa: E402
from tda_gdl_regime.labels import build_shift_event_labels                      # noqa: E402
from tda_gdl_regime.feature_engineering import (                                # noqa: E402
    build_feature_frame, add_cross_asset_features, feature_groups as get_feature_groups
)

# ── paths ----------------------------------------------------------------------
CONFIG_PATH  = os.path.join(REPO_ROOT, "research", "bootstrap_rerun", "bootstrap_final_config.yaml")
OUTPUT_JSON  = os.path.join(REPO_ROOT, "paper", "revisions", "m3_permutation_importance.json")
OUTPUT_FIG   = os.path.join(REPO_ROOT, "paper", "figures", "figure_m3_permutation_importance.pdf")

# ── split boundaries (must match pipeline config) ------------------------------
TRAIN_END  = pd.Timestamp("2018-10-31")
VAL_END    = pd.Timestamp("2022-06-14")
TEST_START = pd.Timestamp("2022-07-28")

N_REPEATS  = 10
RAND_STATE = 42


def _build_config(raw: dict) -> tuple[DataConfig, LabelConfig, FeatureConfig]:
    def _pick(dcls, raw_sect):
        fields = {f.name for f in dataclasses.fields(dcls)}
        return dcls(**{k: v for k, v in raw_sect.items() if k in fields})
    return (
        _pick(DataConfig,  raw["data"]),
        _pick(LabelConfig, raw["labels"]),
        _pick(FeatureConfig, raw["features"]),
    )


def _feature_groups_by_prefix(combined_cols: list[str]) -> dict[str, list[str]]:
    """Split combined feature columns into interpretable subgroups."""
    groups: dict[str, list[str]] = {
        "topology":  [],
        "classical": [],
        "xcorr":     [],
        "sym_onehot": [],
    }
    for c in combined_cols:
        if c.startswith("top_"):
            groups["topology"].append(c)
        elif c.startswith("cls_"):
            groups["classical"].append(c)
        elif c.startswith("xcorr_"):
            groups["xcorr"].append(c)
        elif c.startswith("sym_"):
            groups["sym_onehot"].append(c)
    return groups


def main() -> None:
    print("Loading config ...")
    with open(CONFIG_PATH) as f:
        raw = yaml.safe_load(f)
    data_cfg, label_cfg, feat_cfg = _build_config(raw)

    print("Loading market data ...")
    price_frame = load_market_data(data_cfg, base_dir=REPO_ROOT)
    print(f"  Loaded {len(price_frame):,} rows for {price_frame['symbol'].nunique()} symbols")

    print("Building labels ...")
    labeled_frame = build_shift_event_labels(price_frame, label_cfg)
    print(f"  Labels: {int(labeled_frame['shift_event'].sum())} positive rows")

    print("Building feature frame (this may take several minutes) ...")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        feature_frame = build_feature_frame(labeled_frame, feat_cfg)

    print("Adding cross-asset features ...")
    feature_frame = add_cross_asset_features(
        feature_frame, price_frame, window=feat_cfg.window_bars
    )
    print(f"  Feature frame: {len(feature_frame):,} rows x {feature_frame.shape[1]} cols")

    # ── identify feature columns using same logic as run_pipeline ---------------
    # feature_groups() selects only top_*, cls_*, xcorr_*, sym_* columns,
    # explicitly excluding label-leaking columns (forward_volatility, etc.)
    pipeline_groups = get_feature_groups(feature_frame)
    feat_cols = pipeline_groups["combined"]   # same as rf_combined in pipeline
    print(f"  Feature columns (combined): {len(feat_cols)}")
    print(f"  topology: {len([c for c in feat_cols if c.startswith('top_')])}, "
          f"classical: {len([c for c in feat_cols if c.startswith('cls_')])}, "
          f"xcorr: {len([c for c in feat_cols if c.startswith('xcorr_')])}, "
          f"sym: {len([c for c in feat_cols if c.startswith('sym_')])}")

    # ── train/val/test split ---------------------------------------------------
    feature_frame = feature_frame.sort_values("timestamp").reset_index(drop=True)
    train_mask = feature_frame["timestamp"] <= TRAIN_END
    test_mask  = feature_frame["timestamp"] >= TEST_START

    train_df = feature_frame[train_mask].copy()
    test_df  = feature_frame[test_mask].copy()
    print(f"  Train: {len(train_df):,}  |  Test: {len(test_df):,}")

    X_train = train_df[feat_cols].fillna(0.0).to_numpy()
    y_train = train_df["label"].to_numpy()
    X_test  = test_df[feat_cols].fillna(0.0).to_numpy()
    y_test  = test_df["label"].to_numpy()
    print(f"  Train pos rate: {y_train.mean():.3f}  |  Test pos rate: {y_test.mean():.3f}")

    # ── fit RF (same hyperparams as rf_combined) --------------------------------
    print("\nFitting RandomForestClassifier (n=200, balanced) ...")
    rf = RandomForestClassifier(
        n_estimators=200,
        class_weight="balanced",
        random_state=RAND_STATE,
        n_jobs=-1,
    )
    rf.fit(X_train, y_train)
    print(f"  Test score (accuracy): {rf.score(X_test, y_test):.4f}")

    # ── permutation importance on test -----------------------------------------
    print(f"\nComputing permutation importance (n_repeats={N_REPEATS}) on test ...")
    result = permutation_importance(
        rf, X_test, y_test,
        n_repeats=N_REPEATS,
        random_state=RAND_STATE,
        scoring="average_precision",
        n_jobs=-1,
    )

    importances     = result.importances_mean
    importances_std = result.importances_std

    # ── group-level summary ----------------------------------------------------
    groups = _feature_groups_by_prefix(feat_cols)
    group_importance: dict[str, dict] = {}
    for gname, gcols in groups.items():
        if not gcols:
            continue
        indices = [feat_cols.index(c) for c in gcols if c in feat_cols]
        if not indices:
            continue
        vals = importances[indices]
        group_importance[gname] = {
            "n_features":  len(indices),
            "mean_importance": round(float(vals.mean()), 6),
            "sum_importance":  round(float(vals.sum()), 6),
            "top_feature":     feat_cols[indices[int(np.argmax(vals))]],
            "top_importance":  round(float(vals.max()), 6),
        }
    print("\nGroup-level importance (mean across features in group):")
    for gname, gdata in sorted(group_importance.items(), key=lambda x: -x[1]["mean_importance"]):
        print(f"  {gname:12s}  n={gdata['n_features']:3d}  mean={gdata['mean_importance']:.6f}  "
              f"sum={gdata['sum_importance']:.6f}  top={gdata['top_feature']}")

    # ── top-20 individual features ---------------------------------------------
    top20_idx  = np.argsort(importances)[::-1][:20]
    top20 = [
        {
            "rank":       int(i + 1),
            "feature":    feat_cols[idx],
            "importance": round(float(importances[idx]), 6),
            "std":        round(float(importances_std[idx]), 6),
            "group":      next(
                (g for g, cols in groups.items() if feat_cols[idx] in cols), "topology"
            ),
        }
        for i, idx in enumerate(top20_idx)
    ]
    print("\nTop-20 features:")
    for t in top20[:10]:
        print(f"  {t['rank']:2d}. {t['feature']:<40s}  {t['importance']:.6f}  ({t['group']})")

    # ── save JSON ---------------------------------------------------------------
    output = {
        "group_importance": group_importance,
        "top20_features":   top20,
        "feature_counts":   {g: len(c) for g, c in groups.items() if c},
        "n_repeats":        N_REPEATS,
        "scoring":          "average_precision",
        "split":            "test (2022-07-28 – 2026-03-16)",
    }
    os.makedirs(os.path.dirname(OUTPUT_JSON), exist_ok=True)
    with open(OUTPUT_JSON, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nJSON saved: {OUTPUT_JSON}")

    # ── figure ------------------------------------------------------------------
    _make_figure(top20, group_importance, OUTPUT_FIG)


def _make_figure(
    top20: list[dict],
    group_importance: dict[str, dict],
    out_path: str,
) -> None:
    GROUP_COLORS = {
        "topology":  "#1f77b4",
        "classical": "#ff7f0e",
        "xcorr":     "#2ca02c",
        "sym_onehot": "#9467bd",
        "other":     "#7f7f7f",
    }

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    # Panel A: top-20 individual features
    ax = axes[0]
    names  = [t["feature"]     for t in top20]
    vals   = [t["importance"]  for t in top20]
    stds   = [t["std"]         for t in top20]
    colors = [GROUP_COLORS.get(t["group"], "#7f7f7f") for t in top20]
    y_pos  = np.arange(len(names))

    ax.barh(y_pos, vals, xerr=stds, align="center",
            color=colors, edgecolor="black", linewidth=0.4, alpha=0.85)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(names, fontsize=6.5)
    ax.invert_yaxis()
    ax.set_xlabel("Mean decrease in avg. precision", fontsize=8)
    ax.set_title("Top-20 Features (permutation importance)", fontsize=9)
    ax.tick_params(labelsize=7)

    # Legend
    from matplotlib.patches import Patch
    legend_elems = [Patch(facecolor=c, label=g) for g, c in GROUP_COLORS.items()
                    if g in {t["group"] for t in top20}]
    ax.legend(handles=legend_elems, fontsize=7, loc="lower right")

    # Panel B: group-level sum importance
    ax2 = axes[1]
    sorted_groups = sorted(group_importance.items(), key=lambda x: -x[1]["sum_importance"])
    gnames  = [g for g, _ in sorted_groups]
    gsums   = [d["sum_importance"] for _, d in sorted_groups]
    gcolors = [GROUP_COLORS.get(g, "#7f7f7f") for g in gnames]
    g_y     = np.arange(len(gnames))

    ax2.barh(g_y, gsums, align="center",
             color=gcolors, edgecolor="black", linewidth=0.5, alpha=0.85)
    ax2.set_yticks(g_y)
    ax2.set_yticklabels(gnames, fontsize=9)
    ax2.invert_yaxis()
    ax2.set_xlabel("Sum of mean importance across features in group", fontsize=8)
    ax2.set_title("Feature Group Importance (sum)", fontsize=9)
    ax2.tick_params(labelsize=8)
    for yp, val, gd in zip(g_y, gsums, [d for _, d in sorted_groups]):
        ax2.text(val + max(gsums) * 0.01, yp,
                 f"n={gd['n_features']}", va="center", fontsize=7.5)

    fig.suptitle("rf-combined Permutation Feature Importance (test split 2022–2026)\n"
                 "Scoring: average precision  |  n_repeats=10",
                 fontsize=9, y=1.02)
    plt.tight_layout()
    plt.savefig(out_path, bbox_inches="tight")
    plt.close()
    print(f"Figure saved: {out_path}")


if __name__ == "__main__":
    main()
