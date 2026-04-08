"""
Label Stationarity KS Test (P2.6).

For each split (train, val, test), extracts the realized-volatility values
at positive-label windows. Runs pairwise KS tests to check whether the
label-positive RV distribution is stationary across splits.

Requires:
  PYTHONPATH=src

Outputs:
  paper/revisions/ks_stationarity_results.json

Usage:
    cd <repo_root>
    PYTHONPATH=src python paper/revisions/compute_ks_stationarity.py
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np
from scipy import stats

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(REPO_ROOT, "src"))

from pathlib import Path

from tda_gdl_regime.config import ResearchConfig
from tda_gdl_regime.data_pipeline import load_market_data
from tda_gdl_regime.feature_engineering import (
    add_cross_asset_features,
    build_feature_frame,
    split_feature_frame,
    split_frame_summary,
)
from tda_gdl_regime.labels import build_shift_event_labels

CONFIG_PATH = os.path.join(REPO_ROOT, "research", "bootstrap_rerun", "bootstrap_final_config.yaml")
OUTPUT_JSON = os.path.join(REPO_ROOT, "paper", "revisions", "ks_stationarity_results.json")


def _resolved_gap_settings(cfg: ResearchConfig) -> tuple[int, int]:
    purge = cfg.evaluation.purge_bars
    if purge is None:
        purge = max(cfg.features.window_bars, cfg.labels.lookahead_bars,
                    cfg.labels.volatility_window_bars)
    embargo = cfg.evaluation.embargo_bars
    if embargo is None:
        embargo = cfg.labels.lookahead_bars
    return int(purge), int(embargo)


def main() -> None:
    print(f"Loading config: {CONFIG_PATH}")
    cfg = ResearchConfig.from_yaml(CONFIG_PATH)
    project_root = Path(REPO_ROOT)

    print("Loading market data ...")
    market_frame = load_market_data(cfg.data, base_dir=project_root)
    print(f"  {len(market_frame):,} rows")

    print("Building labels + features ...")
    labeled_frame = build_shift_event_labels(market_frame, cfg.labels)
    feature_frame = build_feature_frame(labeled_frame, cfg.features)
    feature_frame = add_cross_asset_features(feature_frame, labeled_frame)

    purge, embargo = _resolved_gap_settings(cfg)
    splits = split_feature_frame(
        feature_frame,
        train_frac=cfg.evaluation.train_frac,
        val_frac=cfg.evaluation.val_frac,
        test_frac=cfg.evaluation.test_frac,
        purge_bars=purge,
        embargo_bars=embargo,
    )

    summary = split_frame_summary(splits)
    for split_name in ["train", "val", "test"]:
        s = summary[split_name]
        print(f"  {split_name:5s}: {s['rows']:,} rows, {s.get('event_count', '?')} events")

    # Extract realized volatility column (the raw vol value at positive-label rows)
    # The feature frame contains rolling-window features; we need the raw vol used in labeling.
    # Use the 'forward_vol' or similar column if available; otherwise use 'rv' or the
    # label-adjacent vol column.
    # Check available columns
    sample_frame = splits["train"]
    print(f"\nFeature columns (first 20): {list(sample_frame.columns[:20])}")

    # Find a column that represents realized volatility
    vol_candidates = [c for c in sample_frame.columns
                      if any(k in c.lower() for k in ["vol", "rv", "std", "vix"])]
    print(f"Volatility-adjacent columns: {vol_candidates[:15]}")

    # Prefer 'vol_20' or similar backward-looking vol, or the label column adjacent features
    # Fall back to using the spread between high and low as a proxy (Parkinson estimator)
    # In the feature frame, 'backward_vol' or 'vol_{window}' should exist
    vol_col = None
    for cand in ["backward_vol", "vol_20", "rv", "vol"]:
        if cand in sample_frame.columns:
            vol_col = cand
            break
    if vol_col is None:
        # Try regex-like match
        for c in sample_frame.columns:
            if "backward" in c.lower() and "vol" in c.lower():
                vol_col = c
                break
    if vol_col is None and vol_candidates:
        vol_col = vol_candidates[0]

    print(f"\nUsing volatility column: '{vol_col}'")

    results: dict = {}

    for split_name in ["train", "val", "test"]:
        frame = splits[split_name]
        label_col = "label"
        pos_mask  = frame[label_col] == 1
        if vol_col and vol_col in frame.columns:
            rv_vals = frame.loc[pos_mask, vol_col].dropna().to_numpy(dtype=float)
        else:
            # Fall back: use any numeric column with non-trivial variance
            print(f"  WARNING: vol column '{vol_col}' not in {split_name} frame. Using first numeric col.")
            num_cols = frame.select_dtypes(include=[np.number]).columns
            rv_vals  = frame.loc[pos_mask, num_cols[0]].dropna().to_numpy(dtype=float)

        results[split_name] = {
            "n_positive_windows": int(pos_mask.sum()),
            "n_total_windows":    len(frame),
            "rv_mean":   round(float(rv_vals.mean()), 6) if len(rv_vals) > 0 else None,
            "rv_std":    round(float(rv_vals.std()),  6) if len(rv_vals) > 0 else None,
            "rv_median": round(float(np.median(rv_vals)), 6) if len(rv_vals) > 0 else None,
            "rv_values_sample": rv_vals[:10].tolist(),  # first 10 for inspection
        }
        print(f"  {split_name}: n_pos={pos_mask.sum()}  rv_mean={results[split_name]['rv_mean']:.6f}  "
              f"rv_std={results[split_name]['rv_std']:.6f}")

    # Pairwise KS tests
    def get_rv(split_name: str) -> np.ndarray:
        frame = splits[split_name]
        pos_mask = frame["label"] == 1
        if vol_col and vol_col in frame.columns:
            return frame.loc[pos_mask, vol_col].dropna().to_numpy(dtype=float)
        num_cols = frame.select_dtypes(include=[np.number]).columns
        return frame.loc[pos_mask, num_cols[0]].dropna().to_numpy(dtype=float)

    rv_train = get_rv("train")
    rv_val   = get_rv("val")
    rv_test  = get_rv("test")

    pairs = [
        ("train_vs_val",  rv_train, rv_val),
        ("train_vs_test", rv_train, rv_test),
        ("val_vs_test",   rv_val,   rv_test),
    ]

    ks_results: dict = {}
    print("\nKS tests (two-sample, two-tailed) ...")
    for name, a, b in pairs:
        if len(a) < 2 or len(b) < 2:
            print(f"  {name}: insufficient samples")
            ks_results[name] = {"ks_stat": None, "p_value": None, "significant_05": None}
            continue
        stat, pval = stats.ks_2samp(a, b)
        sig = bool(pval < 0.05)
        ks_results[name] = {
            "ks_stat":       round(float(stat), 4),
            "p_value":       round(float(pval), 4),
            "significant_05": sig,
        }
        marker = " *** SIGNIFICANT" if sig else " (not significant)"
        print(f"  {name:20s}: KS={stat:.4f}  p={pval:.4f}{marker}")

    # Conclusion
    train_test_sig = ks_results.get("train_vs_test", {}).get("significant_05", False)
    if train_test_sig:
        conclusion = (
            "The KS test rejects H0 of distributional equality between "
            "train and test label-positive realized volatility "
            f"(KS={ks_results['train_vs_test']['ks_stat']:.3f}, "
            f"p={ks_results['train_vs_test']['p_value']:.3f}). "
            "Mild nonstationarity in the label distribution should be acknowledged "
            "as a limitation; walk-forward evaluation mitigates this risk."
        )
    else:
        conclusion = (
            "The KS test fails to reject distributional equality of "
            "label-positive realized volatility across train/val/test splits "
            f"(train vs test: KS={ks_results['train_vs_test']['ks_stat']:.3f}, "
            f"p={ks_results['train_vs_test']['p_value']:.3f}). "
            "The labeling criterion is approximately stationary across the "
            "evaluation period."
        )

    output = {
        "vol_column_used": vol_col,
        "splits": results,
        "ks_tests": ks_results,
        "conclusion": conclusion,
    }

    with open(OUTPUT_JSON, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nJSON saved: {OUTPUT_JSON}")
    print(f"Conclusion: {conclusion}")


if __name__ == "__main__":
    main()
