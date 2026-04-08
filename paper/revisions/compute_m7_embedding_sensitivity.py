"""
Embedding Sensitivity Experiment (P2.7).

Evaluates gcn-fusion (n_ensemble=1 for speed) across embedding parameter variants:
  - m=4,  tau=1  (embed_dim=4,  embed_tau=1)
  - m=8,  tau=1  (production)
  - m=16, tau=1  (embed_dim=16, embed_tau=1)
  - m=8,  tau=2  (embed_dim=8,  embed_tau=2)

Production result (m=8, tau=1) is loaded from existing metrics rather than rerun.
Three new variants are trained from scratch.

Requires:
  PYTHONPATH=src
  research/bootstrap_rerun/bootstrap_final_config.yaml
  research/bootstrap_rerun/final/metrics.json   (for production reference)

Outputs:
  paper/revisions/m7_embedding_sensitivity.json

Usage:
    cd <repo_root>
    PYTHONPATH=src python paper/revisions/compute_m7_embedding_sensitivity.py
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

CONFIG_PATH   = os.path.join(REPO_ROOT, "research", "bootstrap_rerun", "bootstrap_final_config.yaml")
METRICS_JSON  = os.path.join(REPO_ROOT, "research", "bootstrap_rerun", "final", "metrics.json")
OUTPUT_JSON   = os.path.join(REPO_ROOT, "paper", "revisions", "m7_embedding_sensitivity.json")

# Production reference (m=8, tau=1)
PRODUCTION_PR_AUC = 0.1016

# Variants to run (production is m=8 tau=1 — skip if already have it)
VARIANTS = [
    {"embed_dim": 4,  "embed_tau": 1, "label": "m=4, tau=1"},
    {"embed_dim": 16, "embed_tau": 1, "label": "m=16, tau=1"},
    {"embed_dim": 8,  "embed_tau": 2, "label": "m=8, tau=2"},
]


def _resolved_gap_settings(cfg: ResearchConfig) -> tuple[int, int]:
    purge = cfg.evaluation.purge_bars
    if purge is None:
        purge = max(cfg.features.window_bars, cfg.labels.lookahead_bars,
                    cfg.labels.volatility_window_bars)
    embargo = cfg.evaluation.embargo_bars
    if embargo is None:
        embargo = cfg.labels.lookahead_bars
    return int(purge), int(embargo)


def run_variant(
    cfg: ResearchConfig,
    market_frame,
    embed_dim: int,
    embed_tau: int,
    label: str,
) -> dict:
    """Build graph dataset with modified embedding params and evaluate gcn_fusion."""
    print(f"\n  Variant: {label} (embed_dim={embed_dim}, embed_tau={embed_tau})")

    # Override embedding parameters
    cfg_variant = ResearchConfig.from_yaml(CONFIG_PATH)
    cfg_variant.features.embed_dim  = embed_dim
    cfg_variant.features.embed_tau  = embed_tau
    cfg_variant.models.enabled        = ["gcn_fusion"]
    cfg_variant.models.gdl_n_ensemble = 1

    labeled_frame = build_shift_event_labels(market_frame, cfg_variant.labels)
    feature_frame = build_feature_frame(labeled_frame, cfg_variant.features)
    feature_frame = add_cross_asset_features(feature_frame, labeled_frame)
    groups        = feature_groups(feature_frame)

    purge, embargo = _resolved_gap_settings(cfg_variant)
    splits = split_feature_frame(
        feature_frame,
        train_frac=cfg_variant.evaluation.train_frac,
        val_frac=cfg_variant.evaluation.val_frac,
        test_frac=cfg_variant.evaluation.test_frac,
        purge_bars=purge,
        embargo_bars=embargo,
    )

    graph_dataset = build_graph_dataset(labeled_frame, cfg_variant.features)
    graph_splits  = split_graph_dataset(graph_dataset, splits)

    t0 = time.time()
    fitted = fit_gdl_model_suite(
        graph_splits, splits, groups, cfg_variant.models, cfg_variant.evaluation
    )
    gcn = fitted["gcn_fusion"]

    test_scores = predict_graph_scores(
        gcn, graph_splits["test"], splits["test"], cfg_variant.models
    )

    summary, _ = evaluate_predictions(
        splits["test"],
        scores=test_scores,
        threshold=gcn.threshold,
        early_warning_bars=cfg_variant.evaluation.early_warning_bars,
        bars_per_day=cfg_variant.evaluation.bars_per_day,
        bootstrap_samples=cfg_variant.evaluation.bootstrap_samples,
        bootstrap_block_size=cfg_variant.evaluation.bootstrap_block_size,
    )

    pr_auc   = round(float(summary["pr_auc"]), 4)
    event_f1 = round(float(summary["event_f1"]), 4)
    ci       = [round(x, 4) for x in summary["bootstrap_ci"]["pr_auc"]]
    elapsed  = time.time() - t0

    print(f"    PR-AUC={pr_auc:.4f}  CI={ci}  F1={event_f1:.4f}  ({elapsed:.0f}s)")

    return {
        "label":      label,
        "embed_dim":  embed_dim,
        "embed_tau":  embed_tau,
        "pr_auc":     pr_auc,
        "event_f1":   event_f1,
        "ci_pr_auc":  ci,
        "threshold":  round(float(gcn.threshold), 4),
        "elapsed_sec": round(elapsed, 1),
    }


def main() -> None:
    print(f"Loading config: {CONFIG_PATH}")
    cfg = ResearchConfig.from_yaml(CONFIG_PATH)
    project_root = Path(REPO_ROOT)

    print("Loading market data (shared across variants) ...")
    market_frame = load_market_data(cfg.data, base_dir=project_root)
    print(f"  {len(market_frame):,} rows")

    results: dict = {}

    # Production reference
    results["m8_tau1"] = {
        "label":     "m=8, tau=1 (production)",
        "embed_dim": 8,
        "embed_tau": 1,
        "pr_auc":    PRODUCTION_PR_AUC,
        "note":      "3-seed ensemble from production run; single-seed ~0.0992",
        "ci_pr_auc": None,
    }
    print(f"\nProduction (m=8, tau=1): PR-AUC={PRODUCTION_PR_AUC}")

    for variant in VARIANTS:
        key = f"m{variant['embed_dim']}_tau{variant['embed_tau']}"
        try:
            result = run_variant(
                cfg, market_frame,
                embed_dim=variant["embed_dim"],
                embed_tau=variant["embed_tau"],
                label=variant["label"],
            )
            results[key] = result
        except Exception as exc:
            print(f"  ERROR for {variant['label']}: {exc}")
            results[key] = {"label": variant["label"], "error": str(exc)}

    # Summary table
    print(f"\n{'='*60}")
    print("Embedding sensitivity summary:")
    header = f"{'Config':15s}  {'PR-AUC':>8s}  {'CI':>22s}  {'F1':>8s}"
    print(header)
    for key, r in results.items():
        if "error" in r:
            print(f"  {r['label']:15s}  ERROR: {r['error']}")
        else:
            ci_str = f"[{r['ci_pr_auc'][0]:.4f}, {r['ci_pr_auc'][1]:.4f}]" if r.get("ci_pr_auc") else "N/A"
            f1_str = f"{r.get('event_f1', 0):.4f}" if r.get("event_f1") is not None else "N/A"
            print(f"  {r['label']:25s}  {r['pr_auc']:.4f}  {ci_str:22s}  {f1_str}")

    with open(OUTPUT_JSON, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nJSON saved: {OUTPUT_JSON}")


if __name__ == "__main__":
    main()
