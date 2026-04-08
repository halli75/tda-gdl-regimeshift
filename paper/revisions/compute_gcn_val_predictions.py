"""
GCN Val Predictions (C3).

Retrains gcn_fusion (single seed) and saves predictions on the val split
to predictions_val.csv. Used by compute_hybrid_detector.py (to replace
EGARCH proxy) and compute_temperature_scaling.py (for calibration).

Requires:
  PYTHONPATH=src
  research/bootstrap_rerun/bootstrap_final_config.yaml

Outputs:
  paper/revisions/predictions_val.csv
    columns: symbol, timestamp, label, gcn_fusion_score

Usage:
    cd <repo_root>
    PYTHONPATH=src python paper/revisions/compute_gcn_val_predictions.py
"""
from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(REPO_ROOT, "src"))

from pathlib import Path

from tda_gdl_regime.config import ResearchConfig
from tda_gdl_regime.data_pipeline import load_market_data
from tda_gdl_regime.feature_engineering import (
    add_cross_asset_features,
    build_feature_frame,
    feature_groups,
    split_feature_frame,
)
from tda_gdl_regime.gdl_models import fit_gdl_model_suite, predict_graph_scores
from tda_gdl_regime.graph_data import build_graph_dataset, split_graph_dataset
from tda_gdl_regime.labels import build_shift_event_labels

CONFIG_PATH  = os.path.join(REPO_ROOT, "research", "bootstrap_rerun", "bootstrap_final_config.yaml")
OUTPUT_CSV   = os.path.join(REPO_ROOT, "paper", "revisions", "predictions_val.csv")


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

    # Use single-seed for speed (only need val scores for threshold selection)
    cfg.models.enabled        = ["gcn_fusion"]
    cfg.models.gdl_n_ensemble = 1

    print("Loading market data ...")
    market_frame = load_market_data(cfg.data, base_dir=project_root)
    print(f"  {len(market_frame):,} rows")

    print("Building features and splits ...")
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

    val_frame  = splits["val"]
    test_frame = splits["test"]
    print(f"  Train: {len(splits['train']):,}  Val: {len(val_frame):,}  Test: {len(test_frame):,}")

    graph_dataset = build_graph_dataset(labeled_frame, cfg.features)
    graph_splits  = split_graph_dataset(graph_dataset, splits)

    print("Training gcn_fusion (single seed) ...")
    fitted = fit_gdl_model_suite(
        graph_splits, splits, groups, cfg.models, cfg.evaluation
    )
    gcn = fitted["gcn_fusion"]

    print("Predicting on val split ...")
    val_scores = predict_graph_scores(
        gcn, graph_splits["val"], val_frame, cfg.models
    )

    out_df = val_frame[["symbol", "timestamp", "label"]].copy().reset_index(drop=True)
    out_df["gcn_fusion_score"] = val_scores

    out_df.to_csv(OUTPUT_CSV, index=False)
    print(f"Val predictions saved: {OUTPUT_CSV}")
    print(f"  Rows: {len(out_df):,}  Positive labels: {out_df['label'].sum()}")
    print(f"  Score range: [{val_scores.min():.4f}, {val_scores.max():.4f}]  Mean: {val_scores.mean():.4f}")


if __name__ == "__main__":
    main()
