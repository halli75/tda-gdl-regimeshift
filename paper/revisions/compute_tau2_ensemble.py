"""
tau=2 Ensemble Validation (C2).

Runs gcn_fusion with a 3-seed ensemble at embed_tau=2 to validate
the single-seed tau=2 result (PR-AUC=0.132) from the embedding
sensitivity experiment.

Requires:
  PYTHONPATH=src
  research/bootstrap_rerun/bootstrap_final_config.yaml

Outputs:
  paper/revisions/tau2_ensemble_results.json

Usage:
    cd <repo_root>
    PYTHONPATH=src python paper/revisions/compute_tau2_ensemble.py
"""
from __future__ import annotations

import json
import os
import sys
import time

import numpy as np

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
from tda_gdl_regime.gdl_models import fit_gdl_model_suite, predict_graph_scores
from tda_gdl_regime.graph_data import build_graph_dataset, split_graph_dataset
from tda_gdl_regime.labels import build_shift_event_labels

CONFIG_PATH  = os.path.join(REPO_ROOT, "research", "bootstrap_rerun", "bootstrap_final_config.yaml")
OUTPUT_JSON  = os.path.join(REPO_ROOT, "paper", "revisions", "tau2_ensemble_results.json")

# Single-seed result from compute_m7_embedding_sensitivity.py for reference
SINGLE_SEED_PR_AUC = 0.1322
SINGLE_SEED_CI     = [0.1067, 0.1757]

# Production tau=1 3-seed ensemble for reference
PRODUCTION_PR_AUC  = 0.1016


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

    # Build cfg variant: tau=2, dim=8, 3-seed ensemble
    cfg_v = ResearchConfig.from_yaml(CONFIG_PATH)
    cfg_v.features.embed_dim    = 8
    cfg_v.features.embed_tau    = 2
    cfg_v.models.enabled        = ["gcn_fusion"]
    cfg_v.models.gdl_n_ensemble = 3

    print(f"\nRunning tau=2 ensemble (embed_dim=8, embed_tau=2, n_ensemble=3) ...")

    labeled_frame = build_shift_event_labels(market_frame, cfg_v.labels)
    feature_frame = build_feature_frame(labeled_frame, cfg_v.features)
    feature_frame = add_cross_asset_features(feature_frame, labeled_frame)
    groups        = feature_groups(feature_frame)

    purge, embargo = _resolved_gap_settings(cfg_v)
    splits = split_feature_frame(
        feature_frame,
        train_frac=cfg_v.evaluation.train_frac,
        val_frac=cfg_v.evaluation.val_frac,
        test_frac=cfg_v.evaluation.test_frac,
        purge_bars=purge,
        embargo_bars=embargo,
    )

    graph_dataset = build_graph_dataset(labeled_frame, cfg_v.features)
    graph_splits  = split_graph_dataset(graph_dataset, splits)

    t0 = time.time()
    fitted = fit_gdl_model_suite(
        graph_splits, splits, groups, cfg_v.models, cfg_v.evaluation
    )
    gcn = fitted["gcn_fusion"]

    test_scores = predict_graph_scores(
        gcn, graph_splits["test"], splits["test"], cfg_v.models
    )

    summary, _ = evaluate_predictions(
        splits["test"],
        scores=test_scores,
        threshold=gcn.threshold,
        early_warning_bars=cfg_v.evaluation.early_warning_bars,
        bars_per_day=cfg_v.evaluation.bars_per_day,
        bootstrap_samples=cfg_v.evaluation.bootstrap_samples,
        bootstrap_block_size=cfg_v.evaluation.bootstrap_block_size,
    )

    pr_auc   = round(float(summary["pr_auc"]), 4)
    event_f1 = round(float(summary["event_f1"]), 4)
    recall   = round(float(summary["event_recall"]), 4)
    fa       = round(float(summary["false_alarms_per_day"]), 4)
    ci       = [round(x, 4) for x in summary["bootstrap_ci"]["pr_auc"]]
    elapsed  = time.time() - t0

    print(f"\n  tau=2 (3-seed ensemble): PR-AUC={pr_auc:.4f}  CI={ci}  F1={event_f1:.4f}  ({elapsed:.0f}s)")
    print(f"  Comparison:")
    print(f"    tau=2 single-seed (prev): PR-AUC={SINGLE_SEED_PR_AUC:.4f}  CI={SINGLE_SEED_CI}")
    print(f"    tau=1 production (3-seed): PR-AUC={PRODUCTION_PR_AUC:.4f}")

    delta_vs_single = round(pr_auc - SINGLE_SEED_PR_AUC, 4)
    delta_vs_prod   = round(pr_auc - PRODUCTION_PR_AUC, 4)

    results = {
        "tau2_ensemble": {
            "embed_dim":            8,
            "embed_tau":            2,
            "n_ensemble":           3,
            "pr_auc":               pr_auc,
            "event_f1":             event_f1,
            "event_recall":         recall,
            "false_alarms_per_day": fa,
            "ci_pr_auc":            ci,
            "threshold":            round(float(gcn.threshold), 4),
            "elapsed_sec":          round(elapsed, 1),
        },
        "tau1_production_reference": {
            "embed_dim":  8,
            "embed_tau":  1,
            "n_ensemble": 3,
            "pr_auc":     PRODUCTION_PR_AUC,
            "note":       "From research/bootstrap_rerun/final/metrics.json",
        },
        "tau2_single_seed_reference": {
            "embed_dim":  8,
            "embed_tau":  2,
            "n_ensemble": 1,
            "pr_auc":     SINGLE_SEED_PR_AUC,
            "ci_pr_auc":  SINGLE_SEED_CI,
            "note":       "From paper/revisions/m7_embedding_sensitivity.json",
        },
        "delta_vs_single_seed": delta_vs_single,
        "delta_vs_production":  delta_vs_prod,
    }

    with open(OUTPUT_JSON, "w") as f:
        json.dump(results, f, indent=2)
    print(f"JSON saved: {OUTPUT_JSON}")


if __name__ == "__main__":
    main()
