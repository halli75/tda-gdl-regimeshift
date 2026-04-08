"""
Temperature Scaling for GCN-Fusion (C5).

Post-hoc calibration: find optimal temperature T* on val split that
minimizes ECE, then report calibrated test ECE and confirm PR-AUC
is unchanged (temperature scaling is a monotonic transformation).

Requires:
  paper/revisions/predictions_val.csv     (from compute_gcn_val_predictions.py)
  research/bootstrap_rerun/final/predictions.csv

Outputs:
  paper/revisions/temperature_scaling_results.json

Usage:
    cd <repo_root>
    PYTHONPATH=src python paper/revisions/compute_temperature_scaling.py
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(REPO_ROOT, "src"))

PREDICTIONS_CSV  = os.path.join(REPO_ROOT, "research", "bootstrap_rerun", "final", "predictions.csv")
VAL_PRED_CSV     = os.path.join(REPO_ROOT, "paper", "revisions", "predictions_val.csv")
OUTPUT_JSON      = os.path.join(REPO_ROOT, "paper", "revisions", "temperature_scaling_results.json")

ECE_BINS   = 15
T_GRID_MIN = 0.1
T_GRID_MAX = 10.0
T_GRID_N   = 200


def _ece(scores: np.ndarray, labels: np.ndarray, n_bins: int = ECE_BINS) -> float:
    """Equal-width bin ECE."""
    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    n = len(labels)
    for i in range(n_bins):
        mask = (scores >= bin_edges[i]) & (scores < bin_edges[i + 1])
        if mask.sum() == 0:
            continue
        bin_conf  = scores[mask].mean()
        bin_acc   = labels[mask].mean()
        ece += (mask.sum() / n) * abs(bin_conf - bin_acc)
    return float(ece)


def _apply_temperature(scores: np.ndarray, T: float) -> np.ndarray:
    """Temperature scaling: sigmoid(logit / T)."""
    scores = np.clip(scores, 1e-7, 1 - 1e-7)
    logits = np.log(scores / (1.0 - scores))
    return 1.0 / (1.0 + np.exp(-logits / T))


def main() -> None:
    if not os.path.exists(VAL_PRED_CSV):
        print(f"ERROR: {VAL_PRED_CSV} not found.")
        print("Run compute_gcn_val_predictions.py first.")
        sys.exit(1)

    print(f"Loading val predictions: {VAL_PRED_CSV}")
    val_df = pd.read_csv(VAL_PRED_CSV, parse_dates=["timestamp"])
    val_scores = val_df["gcn_fusion_score"].to_numpy()
    val_labels = val_df["label"].to_numpy().astype(float)
    print(f"  Val rows: {len(val_df):,}  Positive: {int(val_labels.sum())}")

    print(f"Loading test predictions: {PREDICTIONS_CSV}")
    pred_df = pd.read_csv(PREDICTIONS_CSV, parse_dates=["timestamp"])
    test_scores = pred_df["gcn_fusion_score"].to_numpy()
    test_labels = pred_df["label"].to_numpy().astype(float)
    print(f"  Test rows: {len(pred_df):,}  Positive: {int(test_labels.sum())}")

    # Baseline ECE (no calibration)
    val_ece_base  = _ece(val_scores, val_labels)
    test_ece_base = _ece(test_scores, test_labels)
    test_pr_auc   = float(average_precision_score(test_labels, test_scores))
    print(f"\nBaseline: val ECE={val_ece_base:.4f}  test ECE={test_ece_base:.4f}  test PR-AUC={test_pr_auc:.4f}")

    # Grid search T on val ECE
    print(f"\nGrid-searching T in [{T_GRID_MIN}, {T_GRID_MAX}] ({T_GRID_N} points) ...")
    T_grid = np.linspace(T_GRID_MIN, T_GRID_MAX, T_GRID_N)
    val_eces = []
    for T in T_grid:
        cal_scores = _apply_temperature(val_scores, T)
        val_eces.append(_ece(cal_scores, val_labels))

    best_idx = int(np.argmin(val_eces))
    T_star   = float(T_grid[best_idx])
    val_ece_cal = float(val_eces[best_idx])
    print(f"  T* = {T_star:.4f}  (val ECE: {val_ece_base:.4f} -> {val_ece_cal:.4f})")

    # Apply T* to test scores
    test_scores_cal = _apply_temperature(test_scores, T_star)
    test_ece_cal    = _ece(test_scores_cal, test_labels)
    test_pr_auc_cal = float(average_precision_score(test_labels, test_scores_cal))
    pr_auc_delta    = abs(test_pr_auc_cal - test_pr_auc)

    print(f"\nTest results after temperature scaling (T*={T_star:.4f}):")
    print(f"  ECE:    {test_ece_base:.4f} -> {test_ece_cal:.4f}  (delta={test_ece_base - test_ece_cal:+.4f})")
    print(f"  PR-AUC: {test_pr_auc:.4f} -> {test_pr_auc_cal:.4f}  (delta={pr_auc_delta:.6f}, should be ~0)")

    if pr_auc_delta > 0.001:
        print("  WARNING: PR-AUC changed by more than 0.001 after temperature scaling.")
        print("           Temperature scaling should be a monotonic transform — check scores.")

    results = {
        "temperature_scaling": {
            "T_star":             round(T_star, 4),
            "val_ece_baseline":   round(val_ece_base, 4),
            "val_ece_calibrated": round(val_ece_cal, 4),
            "test_ece_baseline":  round(test_ece_base, 4),
            "test_ece_calibrated": round(test_ece_cal, 4),
            "test_pr_auc_baseline":   round(test_pr_auc, 4),
            "test_pr_auc_calibrated": round(test_pr_auc_cal, 4),
            "pr_auc_delta":       round(pr_auc_delta, 6),
            "ece_reduction":      round(test_ece_base - test_ece_cal, 4),
            "ece_reduction_pct":  round((test_ece_base - test_ece_cal) / test_ece_base * 100, 1),
        },
        "model":    "gcn_fusion",
        "ece_bins": ECE_BINS,
        "note": (
            f"Temperature scaling T*={T_star:.4f} selected on val ECE. "
            f"Applied as sigmoid(logit/T) to test scores. "
            f"PR-AUC unchanged (monotonic transform, delta={pr_auc_delta:.6f})."
        ),
    }

    with open(OUTPUT_JSON, "w") as f:
        json.dump(results, f, indent=2)
    print(f"JSON saved: {OUTPUT_JSON}")


if __name__ == "__main__":
    main()
