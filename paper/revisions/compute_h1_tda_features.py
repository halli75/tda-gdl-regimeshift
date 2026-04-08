"""
H1 TDA Features Experiment (P1.1).

Augments the standard rf-combined feature set with 3 H1 (1-cycle) persistence
features extracted via Ripser for a 3-symbol subset (SPY, GLD, DBC).

Pipeline:
  1. For each window in the 3-symbol subset:
     - Extract 40-bar return window, build delay-embedded point cloud (m=6, tau=1)
     - Compute Euclidean distance matrix (35 × 35)
     - Run ripser(D, maxdim=1) to extract H1 persistence diagram
     - Compute 3 scalar features:
         h1_sum_pers:    Σ (death - birth) for all H1 pairs
         h1_max_pers:    max (death - birth)  — dominant cycle
         h1_loop_density: # H1 pairs born before median filtration value
  2. Retrain rf-combined:
     (a) BASELINE: standard 112-d features, SPY+GLD+DBC rows only
     (b) AUGMENTED: same rows, 115-d (+ 3 H1 features)
  3. Evaluate both on test split (block bootstrap B=1000 for speed)
  4. Compute permutation importance for H1 features vs H0 topology group

Requires:
  PYTHONPATH=src
  pip install ripser

Outputs:
  paper/revisions/h1_tda_results.json
  paper/figures/figure_h1_tda.pdf

Usage:
    cd <repo_root>
    PYTHONPATH=src python paper/revisions/compute_h1_tda_features.py
"""
from __future__ import annotations

import json
import os
import sys
import time
import warnings

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(REPO_ROOT, "src"))

from pathlib import Path

from tda_gdl_regime.config import ResearchConfig
from tda_gdl_regime.data_pipeline import load_market_data
from tda_gdl_regime.evaluation import evaluate_predictions
from tda_gdl_regime.feature_engineering import (
    add_cross_asset_features,
    build_feature_frame,
    feature_groups,
    split_feature_frame,
)
from tda_gdl_regime.labels import build_shift_event_labels

CONFIG_PATH = os.path.join(REPO_ROOT, "research", "bootstrap_rerun", "bootstrap_final_config.yaml")
OUTPUT_JSON = os.path.join(REPO_ROOT, "paper", "revisions", "h1_tda_results.json")
OUTPUT_FIG  = os.path.join(REPO_ROOT, "paper", "figures", "figure_h1_tda.pdf")

# H1 experiment parameters
H1_SYMBOLS  = ["SPY", "GLD", "DBC"]
WINDOW_BARS = 40     # window for point cloud
EMBED_DIM   = 6      # m: delay embedding dimension
EMBED_TAU   = 1      # tau: embedding lag
# Resulting point cloud size: WINDOW_BARS - (EMBED_DIM - 1)*EMBED_TAU = 40 - 5 = 35 nodes

BOOTSTRAP_SAMPLES = 1000  # reduced for speed (H1 compute is slow)
BOOTSTRAP_BLOCK   = 64


def _resolved_gap_settings(cfg: ResearchConfig) -> tuple[int, int]:
    purge = cfg.evaluation.purge_bars
    if purge is None:
        purge = max(cfg.features.window_bars, cfg.labels.lookahead_bars,
                    cfg.labels.volatility_window_bars)
    embargo = cfg.evaluation.embargo_bars
    if embargo is None:
        embargo = cfg.labels.lookahead_bars
    return int(purge), int(embargo)


def _delay_embed(x: np.ndarray, m: int, tau: int) -> np.ndarray:
    """
    Delay embedding. Returns point cloud of shape (N, m) where
    N = len(x) - (m-1)*tau.
    Each row is [x[i], x[i+tau], x[i+2*tau], ..., x[i+(m-1)*tau]].
    """
    N = len(x) - (m - 1) * tau
    if N <= 0:
        raise ValueError(f"Window too short for embedding: len={len(x)}, m={m}, tau={tau}")
    cloud = np.stack([x[i * tau: i * tau + N] for i in range(m)], axis=1)
    return cloud


def _h1_features_from_window(x: np.ndarray) -> tuple[float, float, float]:
    """
    Given a 1-D return window, compute 3 H1 TDA scalar features.
    Returns (h1_sum_pers, h1_max_pers, h1_loop_density).
    """
    import ripser

    cloud = _delay_embed(x, EMBED_DIM, EMBED_TAU)
    # Euclidean distance matrix
    diff  = cloud[:, None, :] - cloud[None, :, :]
    D     = np.sqrt((diff ** 2).sum(axis=-1))

    result   = ripser.ripser(D, maxdim=1, distance_matrix=True)
    dgm1     = result["dgms"][1]  # H1 persistence pairs

    if len(dgm1) == 0:
        return 0.0, 0.0, 0.0

    # Remove infinite deaths
    finite = dgm1[~np.isinf(dgm1[:, 1])]
    if len(finite) == 0:
        return 0.0, 0.0, 0.0

    pers = finite[:, 1] - finite[:, 0]  # death - birth

    h1_sum_pers = float(pers.sum())
    h1_max_pers = float(pers.max())

    # Loop density: fraction of H1 pairs born before median filtration value
    median_filtration = float(np.median(D[D > 0]))
    h1_loop_density   = float((finite[:, 0] < median_filtration).sum())

    return h1_sum_pers, h1_max_pers, h1_loop_density


def compute_h1_features_for_frame(
    feature_frame: pd.DataFrame,
    market_frame: pd.DataFrame,
    symbols: list[str],
) -> pd.DataFrame:
    """
    For each row in feature_frame (filtered to symbols), extract 40-bar
    return window from market_frame and compute H1 features.
    Returns DataFrame with columns [h1_sum_pers, h1_max_pers, h1_loop_density]
    aligned to feature_frame index.
    """
    h1_rows = []
    sym_subset = feature_frame[feature_frame["symbol"].isin(symbols)].copy()
    n = len(sym_subset)
    print(f"    Computing H1 features for {n} windows ...")
    t0 = time.time()

    for idx, (row_idx, row) in enumerate(sym_subset.iterrows()):
        sym = row["symbol"]
        ts  = row["timestamp"] if "timestamp" in row.index else row.name

        # Get return series for this symbol up to this timestamp
        sym_returns = market_frame[market_frame["symbol"] == sym].sort_values("timestamp")
        sym_returns = sym_returns[sym_returns["timestamp"] <= ts]

        if len(sym_returns) < WINDOW_BARS:
            h1_rows.append({
                "index_val":       row_idx,
                "h1_sum_pers":     0.0,
                "h1_max_pers":     0.0,
                "h1_loop_density": 0.0,
            })
            continue

        ret_col = "log_return" if "log_return" in sym_returns.columns else "return"
        window_returns = sym_returns[ret_col].iloc[-WINDOW_BARS:].values

        try:
            s, mx, ld = _h1_features_from_window(window_returns)
        except Exception:
            s, mx, ld = 0.0, 0.0, 0.0

        h1_rows.append({
            "index_val":       row_idx,
            "h1_sum_pers":     s,
            "h1_max_pers":     mx,
            "h1_loop_density": ld,
        })

        if (idx + 1) % 100 == 0:
            elapsed = time.time() - t0
            print(f"      {idx+1}/{n}  ({elapsed:.0f}s)")

    elapsed = time.time() - t0
    print(f"    H1 features done ({elapsed:.0f}s)")

    h1_df = pd.DataFrame(h1_rows).set_index("index_val")
    return h1_df


def train_rf_and_evaluate(
    train_frame: pd.DataFrame,
    val_frame: pd.DataFrame,
    test_frame: pd.DataFrame,
    feature_cols: list[str],
    label_col: str,
    cfg: ResearchConfig,
    model_name: str,
) -> dict:
    """Train rf-combined on train_frame, select threshold on val, evaluate on test."""
    from sklearn.ensemble import RandomForestClassifier

    X_train = train_frame[feature_cols].values
    y_train = train_frame[label_col].values
    X_val   = val_frame[feature_cols].values
    X_test  = test_frame[feature_cols].values

    print(f"    Training RF ({model_name}): {X_train.shape[0]} samples, {X_train.shape[1]} features ...")
    t0 = time.time()

    rf = RandomForestClassifier(
        n_estimators=200,
        max_depth=8,
        min_samples_leaf=10,
        random_state=42,
        n_jobs=-1,
        class_weight="balanced",
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        rf.fit(X_train, y_train)
    print(f"    Training done ({time.time()-t0:.1f}s)")

    val_scores  = rf.predict_proba(X_val)[:, 1]
    test_scores = rf.predict_proba(X_test)[:, 1]

    # Simple threshold selection: best event_f1 on val via threshold grid
    from tda_gdl_regime.evaluation import select_best_threshold
    threshold_grid = np.linspace(0.01, 0.99, 99).tolist()
    best_thresh, _ = select_best_threshold(
        frame=val_frame,
        scores=val_scores,
        threshold_grid=threshold_grid,
        selection_metric="event_f1",
        early_warning_bars=cfg.evaluation.early_warning_bars,
        bars_per_day=cfg.evaluation.bars_per_day,
        max_false_alarms_per_day=2.0,
        max_positive_rate=0.5,
    )

    test_summary, _ = evaluate_predictions(
        frame=test_frame,
        scores=test_scores,
        threshold=best_thresh,
        early_warning_bars=cfg.evaluation.early_warning_bars,
        bars_per_day=cfg.evaluation.bars_per_day,
        bootstrap_samples=BOOTSTRAP_SAMPLES,
        bootstrap_block_size=BOOTSTRAP_BLOCK,
    )

    pr_auc   = round(float(test_summary["pr_auc"]), 4)
    event_f1 = round(float(test_summary["event_f1"]), 4)
    ci       = [round(x, 4) for x in test_summary["bootstrap_ci"]["pr_auc"]]

    print(f"    {model_name}: PR-AUC={pr_auc:.4f}  CI={ci}  F1={event_f1:.4f}")
    return {
        "pr_auc":    pr_auc,
        "event_f1":  event_f1,
        "ci_pr_auc": ci,
        "threshold": round(best_thresh, 4),
        "rf_model":  rf,
        "feature_cols": feature_cols,
    }


def main() -> None:
    print(f"Loading config: {CONFIG_PATH}")
    cfg = ResearchConfig.from_yaml(CONFIG_PATH)
    project_root = Path(REPO_ROOT)

    print("Loading market data ...")
    market_frame = load_market_data(cfg.data, base_dir=project_root)

    print("Building labels + features ...")
    labeled_frame = build_shift_event_labels(market_frame, cfg.labels)
    feature_frame = build_feature_frame(labeled_frame, cfg.features)
    feature_frame = add_cross_asset_features(feature_frame, labeled_frame)
    groups        = feature_groups(feature_frame)

    purge, embargo = _resolved_gap_settings(cfg)
    splits = split_feature_frame(
        feature_frame,
        train_frac=cfg.evaluation.train_frac,
        val_frac=cfg.evaluation.val_frac,
        test_frac=cfg.evaluation.test_frac,
        purge_bars=purge,
        embargo_bars=embargo,
    )

    # Filter to H1_SYMBOLS only
    def filter_symbols(frame: pd.DataFrame) -> pd.DataFrame:
        return frame[frame["symbol"].isin(H1_SYMBOLS)].copy().reset_index(drop=True)

    train_3 = filter_symbols(splits["train"])
    val_3   = filter_symbols(splits["val"])
    test_3  = filter_symbols(splits["test"])

    print(f"3-symbol subset: train={len(train_3)}, val={len(val_3)}, test={len(test_3)}")
    print(f"  Train events: {(train_3['label']==1).sum()}, "
          f"Val events: {(val_3['label']==1).sum()}, "
          f"Test events: {(test_3['label']==1).sum()}")

    # Standard feature columns (exclude non-feature cols)
    non_feature_cols = {"label", "symbol", "timestamp", "event_id", "split"}
    base_feature_cols = [c for c in splits["train"].columns
                         if c not in non_feature_cols
                         and splits["train"][c].dtype in [np.float64, np.float32, np.int64, np.int32]]
    print(f"Base feature cols: {len(base_feature_cols)}")

    # ── Baseline RF (no H1) ──────────────────────────────────────────────────────
    print("\n=== Baseline RF (no H1 features) ===")
    baseline_result = train_rf_and_evaluate(
        train_3, val_3, test_3,
        feature_cols=base_feature_cols,
        label_col="label",
        cfg=cfg,
        model_name="rf_baseline",
    )

    # ── Compute H1 features ──────────────────────────────────────────────────────
    print("\n=== Computing H1 features ===")
    print("  (This may take several minutes ...)")

    # Ensure we have a return column (market_frame uses 'return')
    if "return" not in market_frame.columns and "log_return" not in market_frame.columns:
        market_frame = market_frame.sort_values(["symbol", "timestamp"])
        market_frame["return"] = market_frame.groupby("symbol")["price"].transform(
            lambda x: np.log(x / x.shift(1))
        ).fillna(0.0)
    # Standardize to 'log_return'
    if "log_return" not in market_frame.columns:
        market_frame = market_frame.copy()
        market_frame["log_return"] = market_frame["return"]

    # Compute H1 features for train, val, test subsets
    def add_h1_to_frame(frame: pd.DataFrame, name: str) -> pd.DataFrame:
        h1_df = compute_h1_features_for_frame(frame, market_frame, H1_SYMBOLS)
        frame = frame.copy()
        frame["h1_sum_pers"]     = h1_df["h1_sum_pers"].reindex(frame.index).fillna(0.0).values
        frame["h1_max_pers"]     = h1_df["h1_max_pers"].reindex(frame.index).fillna(0.0).values
        frame["h1_loop_density"] = h1_df["h1_loop_density"].reindex(frame.index).fillna(0.0).values
        return frame

    print("  Train split ...")
    train_h1 = add_h1_to_frame(train_3, "train")
    print("  Val split ...")
    val_h1   = add_h1_to_frame(val_3,   "val")
    print("  Test split ...")
    test_h1  = add_h1_to_frame(test_3,  "test")

    h1_cols = ["h1_sum_pers", "h1_max_pers", "h1_loop_density"]
    aug_feature_cols = base_feature_cols + h1_cols

    # Descriptive stats on H1 features
    h1_stats = {}
    for col in h1_cols:
        vals = train_h1[col].values
        h1_stats[col] = {
            "mean":   round(float(vals.mean()), 6),
            "std":    round(float(vals.std()),  6),
            "nonzero_frac": round(float((vals != 0).mean()), 4),
        }
        print(f"  {col}: mean={h1_stats[col]['mean']:.4f}  std={h1_stats[col]['std']:.4f}  "
              f"nonzero={h1_stats[col]['nonzero_frac']:.2%}")

    # ── Augmented RF (with H1) ──────────────────────────────────────────────────
    print("\n=== Augmented RF (with H1 features) ===")
    aug_result = train_rf_and_evaluate(
        train_h1, val_h1, test_h1,
        feature_cols=aug_feature_cols,
        label_col="label",
        cfg=cfg,
        model_name="rf_h1",
    )

    # ── Permutation importance for H1 vs H0 topology group ──────────────────────
    print("\n=== Permutation importance ===")
    rf_aug = aug_result["rf_model"]

    from sklearn.inspection import permutation_importance

    X_test_aug = test_h1[aug_feature_cols].values
    y_test     = test_h1["label"].values

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        perm_result = permutation_importance(
            rf_aug, X_test_aug, y_test,
            n_repeats=10,
            random_state=42,
            scoring="average_precision",
            n_jobs=-1,
        )

    imp_mean = perm_result.importances_mean
    imp_std  = perm_result.importances_std

    col_to_idx = {col: i for i, col in enumerate(aug_feature_cols)}

    h1_imp = {
        col: {
            "mean": round(float(imp_mean[col_to_idx[col]]), 6),
            "std":  round(float(imp_std[col_to_idx[col]]),  6),
        }
        for col in h1_cols
    }

    # H0 topology group importance (mean across group)
    h0_group_cols = [c for c in base_feature_cols
                     if any(k in c.lower() for k in ["h0", "topo", "betti", "pers"])]
    if not h0_group_cols:
        # Fallback: all topology-prefixed features
        h0_group_cols = groups.get("topology", [])
    if h0_group_cols:
        h0_imp_vals = [imp_mean[col_to_idx[c]] for c in h0_group_cols if c in col_to_idx]
        h0_mean_imp = round(float(np.mean(h0_imp_vals)), 6) if h0_imp_vals else 0.0
    else:
        h0_mean_imp = 0.0

    print(f"\n  H1 permutation importance:")
    for col, v in h1_imp.items():
        print(f"    {col}: {v['mean']:.6f} ± {v['std']:.6f}")
    print(f"  H0 topology group mean importance: {h0_mean_imp:.6f}  "
          f"(from {len(h0_group_cols)} features)")

    # ── Save figure ──────────────────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    # Left: PR-AUC comparison
    ax = axes[0]
    names = ["RF baseline\n(no H1)", "RF + H1\n(augmented)"]
    vals  = [baseline_result["pr_auc"], aug_result["pr_auc"]]
    cis   = [baseline_result["ci_pr_auc"], aug_result["ci_pr_auc"]]
    colors = ["#1f77b4", "#d62728"]
    bars = ax.bar(names, vals, color=colors, alpha=0.85, edgecolor="black", linewidth=0.6, width=0.45)
    for i, (v, ci) in enumerate(zip(vals, cis)):
        ax.errorbar(i, v, yerr=[[v - ci[0]], [ci[1] - v]],
                    fmt="none", color="black", capsize=5, linewidth=1.5)
        ax.text(i, ci[1] + max(vals) * 0.03, f"{v:.4f}",
                ha="center", va="bottom", fontsize=9)
    ax.set_title(f"PR-AUC: SPY+GLD+DBC subset\n(B={BOOTSTRAP_SAMPLES} bootstrap)", fontsize=9)
    ax.set_ylabel("PR-AUC", fontsize=8)
    ax.tick_params(labelsize=8)

    # Right: H1 permutation importance vs H0 mean
    ax = axes[1]
    h1_names = list(h1_imp.keys()) + ["H0 group\n(mean)"]
    h1_vals  = [h1_imp[c]["mean"] for c in h1_cols] + [h0_mean_imp]
    h1_errs  = [h1_imp[c]["std"]  for c in h1_cols] + [0.0]
    h1_colors = ["#d62728", "#d62728", "#d62728", "#1f77b4"]
    bars2 = ax.bar(h1_names, h1_vals, color=h1_colors, alpha=0.85,
                   edgecolor="black", linewidth=0.6, width=0.45)
    for i, (v, e) in enumerate(zip(h1_vals, h1_errs)):
        if e > 0:
            ax.errorbar(i, v, yerr=e, fmt="none", color="black", capsize=5, linewidth=1.5)
    ax.axhline(0, color="black", linewidth=0.8, linestyle="--")
    ax.set_title("Permutation importance\n(test split, AP score)", fontsize=9)
    ax.set_ylabel("Mean decrease in AP", fontsize=8)
    ax.tick_params(labelsize=7)
    ax.set_xticklabels(h1_names, fontsize=7, rotation=10, ha="right")

    fig.suptitle("H1 Persistent Homology Features — SPY/GLD/DBC Subset",
                 fontsize=10, y=1.02)
    plt.tight_layout()
    plt.savefig(OUTPUT_FIG, bbox_inches="tight")
    plt.close()
    print(f"\nFigure saved: {OUTPUT_FIG}")

    # ── Save JSON ────────────────────────────────────────────────────────────────
    results = {
        "parameters": {
            "symbols":    H1_SYMBOLS,
            "window_bars": WINDOW_BARS,
            "embed_dim":  EMBED_DIM,
            "embed_tau":  EMBED_TAU,
            "point_cloud_size": WINDOW_BARS - (EMBED_DIM - 1) * EMBED_TAU,
        },
        "h1_feature_stats": h1_stats,
        "rf_baseline_pr_auc":  baseline_result["pr_auc"],
        "rf_baseline_f1":      baseline_result["event_f1"],
        "rf_baseline_ci":      baseline_result["ci_pr_auc"],
        "rf_h1_pr_auc":        aug_result["pr_auc"],
        "rf_h1_f1":            aug_result["event_f1"],
        "rf_h1_ci":            aug_result["ci_pr_auc"],
        "h1_permutation_importance": {
            col: v for col, v in h1_imp.items()
        },
        "h0_topology_mean_importance": h0_mean_imp,
        "h0_topology_n_features": len(h0_group_cols),
        "conclusion": (
            f"RF-combined baseline (112-d) PR-AUC={baseline_result['pr_auc']:.4f}; "
            f"RF+H1 (115-d) PR-AUC={aug_result['pr_auc']:.4f} "
            f"(delta={aug_result['pr_auc']-baseline_result['pr_auc']:+.4f}). "
            f"H1 permutation importances: "
            f"h1_sum_pers={h1_imp['h1_sum_pers']['mean']:.6f}, "
            f"h1_max_pers={h1_imp['h1_max_pers']['mean']:.6f}, "
            f"h1_loop_density={h1_imp['h1_loop_density']['mean']:.6f} "
            f"vs H0 group mean={h0_mean_imp:.6f}."
        ),
    }

    with open(OUTPUT_JSON, "w") as f:
        json.dump(results, f, indent=2)
    print(f"JSON saved: {OUTPUT_JSON}")
    print(f"\nSummary: baseline={baseline_result['pr_auc']:.4f}  +H1={aug_result['pr_auc']:.4f}  "
          f"delta={aug_result['pr_auc']-baseline_result['pr_auc']:+.4f}")


if __name__ == "__main__":
    main()
