"""
Seed variance for gcn_fusion: retrain with n_ensemble=1 for each of seeds 42, 43, 44.

Reports per-seed test-split PR-AUC (with bootstrap CI) and the ensemble mean for
comparison. Data loading, feature engineering, and graph construction are done once;
only the GCN training step is repeated per seed.

The threshold for each single-seed run is val-selected (same protocol as production).
PR-AUC is threshold-independent; event F1 at the val-selected threshold is also reported.

Uses the same config as the production run:
  research/bootstrap_rerun/bootstrap_final_config.yaml
  (but overrides evaluation_mode=final, gdl_n_ensemble=1, enabled=["gcn_fusion"])

Requires:
  PYTHONPATH=src

Outputs:
  paper/revisions/seed_variance_results.json

Usage:
    cd <repo_root>
    PYTHONPATH=src python paper/revisions/compute_seed_variance.py
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
    split_frame_summary,
)
from tda_gdl_regime.gdl_models import fit_gdl_model_suite, predict_graph_scores
from tda_gdl_regime.graph_data import build_graph_dataset, split_graph_dataset
from tda_gdl_regime.labels import build_shift_event_labels

CONFIG_PATH = os.path.join(REPO_ROOT, "research", "bootstrap_rerun", "bootstrap_final_config.yaml")
OUTPUT_JSON = os.path.join(REPO_ROOT, "paper", "revisions", "seed_variance_results.json")

SEEDS = [42, 43, 44]

# Reference ensemble mean PR-AUC (from research/bootstrap_rerun/final/metrics.json)
ENSEMBLE_PR_AUC = 0.1016   # 3-seed ensemble from production run


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

    # ── Build data once ────────────────────────────────────────────────────────
    print("\nLoading market data ...")
    t0 = time.time()
    market_frame = load_market_data(cfg.data, base_dir=project_root)
    print(f"  {len(market_frame):,} rows  ({time.time()-t0:.1f}s)")

    print("Building labels + features + graph dataset ...")
    t0 = time.time()
    labeled_frame  = build_shift_event_labels(market_frame, cfg.labels)
    feature_frame  = build_feature_frame(labeled_frame, cfg.features)
    feature_frame  = add_cross_asset_features(feature_frame, labeled_frame)
    groups         = feature_groups(feature_frame)
    purge, embargo = _resolved_gap_settings(cfg)
    splits         = split_feature_frame(
        feature_frame,
        train_frac=cfg.evaluation.train_frac,
        val_frac=cfg.evaluation.val_frac,
        test_frac=cfg.evaluation.test_frac,
        purge_bars=purge,
        embargo_bars=embargo,
    )
    graph_dataset  = build_graph_dataset(labeled_frame, cfg.features)
    graph_splits   = split_graph_dataset(graph_dataset, splits)
    print(f"  Done  ({time.time()-t0:.1f}s)")

    split_summary = split_frame_summary(splits)
    print(f"  Train: {split_summary['train']['rows']:,} rows  "
          f"Val: {split_summary['val']['rows']:,} rows  "
          f"Test: {split_summary['test']['rows']:,} rows")
    print(f"  Test events: {split_summary['test']['event_count']}")

    # Override config: single-seed mode, gcn_fusion only
    cfg.models.enabled        = ["gcn_fusion"]
    cfg.models.gdl_n_ensemble = 1

    # ── Run per seed ───────────────────────────────────────────────────────────
    results: dict[str, dict] = {}

    for seed in SEEDS:
        cfg.models.random_state = seed
        key = f"seed_{seed}"
        print(f"\n{'='*60}")
        print(f"Seed {seed} (n_ensemble=1) ...")
        t0 = time.time()

        fitted_graph = fit_gdl_model_suite(
            graph_splits, splits, groups, cfg.models, cfg.evaluation
        )
        gcn = fitted_graph["gcn_fusion"]

        test_scores = predict_graph_scores(
            gcn, graph_splits["test"], splits["test"], cfg.models
        )

        summary, _ = evaluate_predictions(
            splits["test"],
            scores=test_scores,
            threshold=gcn.threshold,
            early_warning_bars=cfg.evaluation.early_warning_bars,
            bars_per_day=cfg.evaluation.bars_per_day,
            bootstrap_samples=cfg.evaluation.bootstrap_samples,
            bootstrap_block_size=cfg.evaluation.bootstrap_block_size,
        )

        elapsed = time.time() - t0
        pr_auc   = round(float(summary["pr_auc"]), 4)
        event_f1 = round(float(summary["event_f1"]), 4)
        ci       = [round(x, 4) for x in summary["bootstrap_ci"]["pr_auc"]]

        results[key] = {
            "seed":         seed,
            "pr_auc":       pr_auc,
            "event_f1":     event_f1,
            "ci_pr_auc":    ci,
            "threshold":    round(float(gcn.threshold), 4),
            "elapsed_sec":  round(elapsed, 1),
        }
        print(f"  PR-AUC={pr_auc:.4f}  CI={ci}  F1={event_f1:.4f}  "
              f"threshold={gcn.threshold:.4f}  ({elapsed:.0f}s)")

    # ── Summary ────────────────────────────────────────────────────────────────
    pr_aucs = [results[f"seed_{s}"]["pr_auc"] for s in SEEDS]
    mean_pr = round(float(np.mean(pr_aucs)), 4)
    min_pr  = round(float(np.min(pr_aucs)),  4)
    max_pr  = round(float(np.max(pr_aucs)),  4)
    std_pr  = round(float(np.std(pr_aucs)),  4)

    results["summary"] = {
        "seeds":            SEEDS,
        "mean_pr_auc":      mean_pr,
        "std_pr_auc":       std_pr,
        "min_pr_auc":       min_pr,
        "max_pr_auc":       max_pr,
        "range_pr_auc":     round(max_pr - min_pr, 4),
        "ensemble_3seed_pr_auc": ENSEMBLE_PR_AUC,
        "note": (
            "Per-seed PR-AUCs are from single-member (n_ensemble=1) runs "
            "using the same train/val/test split as the production run. "
            "Threshold is val-selected independently for each seed. "
            "PR-AUC is threshold-independent (full precision-recall curve)."
        ),
    }

    with open(OUTPUT_JSON, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nJSON saved: {OUTPUT_JSON}")

    print(f"\n{'='*60}")
    print(f"Seed PR-AUC summary:")
    for s in SEEDS:
        r = results[f"seed_{s}"]
        print(f"  seed {s}: PR-AUC={r['pr_auc']:.4f}  CI={r['ci_pr_auc']}")
    print(f"  Mean: {mean_pr:.4f}  Std: {std_pr:.4f}  Range: [{min_pr:.4f}, {max_pr:.4f}]")
    print(f"  3-seed ensemble (production): {ENSEMBLE_PR_AUC:.4f}")


if __name__ == "__main__":
    main()
