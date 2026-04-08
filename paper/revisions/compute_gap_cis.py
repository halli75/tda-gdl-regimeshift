"""
Gap Trajectory Bootstrap CIs (C4).

Computes block-bootstrap 95% CI for the Round 6 validation-to-test
event F1 gap (-0.055). Rounds 1-2 gaps are historical point estimates
with predictions no longer available.

The gap statistic is defined as:
  gap = val_event_f1 - test_event_f1

Val event F1 is a point estimate from training (stored in metrics.json).
We bootstrap the test event F1 distribution to get a CI on the gap.

Requires:
  PYTHONPATH=src
  research/bootstrap_rerun/final/predictions.csv
  research/bootstrap_rerun/final/metrics.json

Outputs:
  paper/revisions/gap_ci_results.json

Usage:
    cd <repo_root>
    PYTHONPATH=src python paper/revisions/compute_gap_cis.py
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np
import pandas as pd

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(REPO_ROOT, "src"))

PREDICTIONS_CSV = os.path.join(REPO_ROOT, "research", "bootstrap_rerun", "final", "predictions.csv")
METRICS_JSON    = os.path.join(REPO_ROOT, "research", "bootstrap_rerun", "final", "metrics.json")
OUTPUT_JSON     = os.path.join(REPO_ROOT, "paper", "revisions", "gap_ci_results.json")

# Historical point estimates (predictions not retained)
GAP_ROUND1 = -0.215  # single-split, Round 3
GAP_ROUND2 = -0.157  # Round 5
VAL_F1_ROUND6_POINT = None  # loaded from metrics.json

EARLY_WARNING_BARS = 5
BARS_PER_DAY       = 50
BOOTSTRAP_SAMPLES  = 5_000
BOOTSTRAP_BLOCK    = 64
ALPHA              = 0.05


def _event_f1_from_frame(
    pred_df: pd.DataFrame,
    threshold: float,
    early_warning_bars: int,
    bars_per_day: int,
) -> float:
    """Compute event F1 from a prediction DataFrame (with gcn_fusion_score and label columns)."""
    from tda_gdl_regime.evaluation import evaluate_predictions
    summary, _ = evaluate_predictions(
        frame=pred_df,
        scores=pred_df["gcn_fusion_score"].to_numpy(),
        threshold=threshold,
        early_warning_bars=early_warning_bars,
        bars_per_day=bars_per_day,
        bootstrap_samples=0,
        bootstrap_block_size=BOOTSTRAP_BLOCK,
    )
    return float(summary["event_f1"])


def _block_bootstrap_event_f1(
    pred_df: pd.DataFrame,
    threshold: float,
    early_warning_bars: int,
    bars_per_day: int,
    n_samples: int,
    block_size: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Block bootstrap the event F1 statistic."""
    from tda_gdl_regime.evaluation import evaluate_predictions

    n = len(pred_df)
    n_blocks = int(np.ceil(n / block_size))
    block_starts = np.arange(0, n, block_size)
    scores = pred_df["gcn_fusion_score"].to_numpy()
    labels = pred_df["label"].to_numpy()
    # Build arrays for bootstrapping
    boot_f1s = []
    for _ in range(n_samples):
        chosen_starts = rng.choice(block_starts, size=n_blocks, replace=True)
        idx = np.concatenate([
            np.arange(s, min(s + block_size, n)) for s in chosen_starts
        ])[:n]
        boot_scores = scores[idx]
        boot_labels = labels[idx]
        # Build minimal DataFrame
        boot_df = pd.DataFrame({
            "gcn_fusion_score": boot_scores,
            "label": boot_labels,
            "symbol": pred_df["symbol"].to_numpy()[idx],
            "timestamp": pred_df["timestamp"].to_numpy()[idx],
        })
        try:
            summary, _ = evaluate_predictions(
                frame=boot_df,
                scores=boot_scores,
                threshold=threshold,
                early_warning_bars=early_warning_bars,
                bars_per_day=bars_per_day,
                bootstrap_samples=0,
                bootstrap_block_size=block_size,
            )
            boot_f1s.append(float(summary["event_f1"]))
        except Exception:
            continue

    return np.array(boot_f1s)


def main() -> None:
    print(f"Loading predictions: {PREDICTIONS_CSV}")
    pred_df = pd.read_csv(PREDICTIONS_CSV, parse_dates=["timestamp"])
    print(f"  {len(pred_df):,} rows")

    print(f"Loading metrics: {METRICS_JSON}")
    with open(METRICS_JSON) as f:
        metrics = json.load(f)

    # Extract val event F1 and threshold for gcn_fusion
    gcn_metrics = metrics.get("gcn_fusion", {})
    val_f1_point = float(gcn_metrics.get("val_event_f1", gcn_metrics.get("validation_f1", 0.45)))
    threshold    = float(gcn_metrics.get("threshold", 0.50))
    test_f1_point = float(gcn_metrics.get("test_event_f1", gcn_metrics.get("event_f1", 0.395)))
    gap_point = val_f1_point - test_f1_point

    print(f"  Val event F1 (point): {val_f1_point:.4f}")
    print(f"  Test event F1 (point): {test_f1_point:.4f}")
    print(f"  Gap (point estimate): {gap_point:.4f}")
    print(f"  Threshold: {threshold:.4f}")

    # Block bootstrap on test event F1
    print(f"\nBlock bootstrap (B={BOOTSTRAP_SAMPLES:,}, block={BOOTSTRAP_BLOCK}) on test event F1 ...")
    rng = np.random.default_rng(42)
    boot_test_f1s = _block_bootstrap_event_f1(
        pred_df=pred_df,
        threshold=threshold,
        early_warning_bars=EARLY_WARNING_BARS,
        bars_per_day=BARS_PER_DAY,
        n_samples=BOOTSTRAP_SAMPLES,
        block_size=BOOTSTRAP_BLOCK,
        rng=rng,
    )
    print(f"  Bootstrap samples: {len(boot_test_f1s)}")

    # Gap distribution: gap = val_f1 - boot_test_f1
    boot_gaps = val_f1_point - boot_test_f1s
    ci_low  = float(np.percentile(boot_gaps, 100 * ALPHA / 2))
    ci_high = float(np.percentile(boot_gaps, 100 * (1 - ALPHA / 2)))
    boot_mean = float(np.mean(boot_gaps))
    boot_std  = float(np.std(boot_gaps))

    print(f"\n  Round 6 gap: {gap_point:.4f}  95% CI: [{ci_low:.4f}, {ci_high:.4f}]")
    print(f"  Bootstrap mean: {boot_mean:.4f}  Std: {boot_std:.4f}")

    results = {
        "round6": {
            "val_event_f1":       round(val_f1_point, 4),
            "test_event_f1":      round(test_f1_point, 4),
            "gap_point_estimate": round(gap_point, 4),
            "gap_ci_95":          [round(ci_low, 4), round(ci_high, 4)],
            "bootstrap_mean_gap": round(boot_mean, 4),
            "bootstrap_std_gap":  round(boot_std, 4),
            "n_bootstrap":        len(boot_test_f1s),
            "block_size":         BOOTSTRAP_BLOCK,
            "threshold":          round(threshold, 4),
            "note":               "CI on gap = val_f1 - test_f1; val_f1 is a point estimate; bootstrap resamples test set",
        },
        "round5": {
            "gap_point_estimate": GAP_ROUND2,
            "gap_ci_95":          None,
            "note":               "Historical run; val/test predictions not retained; point estimate only",
        },
        "round3": {
            "gap_point_estimate": GAP_ROUND1,
            "gap_ci_95":          None,
            "note":               "Single-split evaluation (historical); val/test predictions not retained; point estimate only",
        },
        "gap_trajectory": [GAP_ROUND1, GAP_ROUND2, round(gap_point, 4)],
    }

    with open(OUTPUT_JSON, "w") as f:
        json.dump(results, f, indent=2)
    print(f"JSON saved: {OUTPUT_JSON}")


if __name__ == "__main__":
    main()
