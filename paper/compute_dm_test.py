"""
Diebold-Mariano test for equal predictive ability: gcn_fusion vs vol_threshold.

Two loss functions are used:
  (1) Brier score on gcn_fusion probability vs vol_threshold BINARY prediction
      (vol_threshold is thresholded to 0/1 at its selected threshold 0.00963).
      This tests whether gcn_fusion's calibrated probabilities beat vol's hard forecast
      under a proper scoring rule.
  (2) 0-1 loss (binary error) at each model's selected threshold.
      This tests predictive accuracy at the fixed operating points reported in Table 3.

Long-run variance: Newey-West HAC with Bartlett kernel.

Usage:
    cd <repo_root>
    python paper/compute_dm_test.py

Outputs:
    paper/dm_test_results.json
"""

import json
import os
import numpy as np
import pandas as pd
from scipy import stats

# ── Paths ──────────────────────────────────────────────────────────────────────
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PREDICTIONS_CSV = os.path.join(
    REPO_ROOT,
    "research", "bootstrap_rerun", "final", "predictions.csv",
)
METRICS_JSON = os.path.join(
    REPO_ROOT,
    "research", "autoresearch", "final_runs",
    "20260401_150829_final", "artifacts", "final", "metrics.json",
)
OUTPUT_JSON = os.path.join(REPO_ROOT, "paper", "dm_test_results.json")

# Thresholds from metrics.json (models listed in order: vol, gcn_fusion, gcn_graph, rf_combined, rf_topology)
GCN_THRESHOLD = 0.55
VOL_THRESHOLD = 0.009632050788941530   # grid-searched on validation split


def newey_west_lrv(d: np.ndarray, H: int) -> float:
    """
    Newey-West long-run variance with Bartlett kernel.
    LRV = γ₀ + 2 × Σ_{h=1}^{H} (1 − h/(H+1)) × γ_h
    """
    T = len(d)
    dm = d - d.mean()
    gamma_0 = np.dot(dm, dm) / T
    lrv = gamma_0
    for h in range(1, H + 1):
        gamma_h = np.dot(dm[h:], dm[:-h]) / T
        weight = 1.0 - h / (H + 1)
        lrv += 2.0 * weight * gamma_h
    return lrv


def dm_test(d: np.ndarray, label: str) -> dict:
    """Run DM test on loss differential series d = loss_model1 - loss_model2."""
    T = len(d)
    d_bar = d.mean()
    H = int(np.floor(4.0 * (T / 100.0) ** (2.0 / 9.0)))
    lrv = newey_west_lrv(d, H)

    if lrv <= 0:
        raise RuntimeError(f"Non-positive LRV ({lrv:.6g}) for test '{label}'")

    dm_stat = d_bar / np.sqrt(lrv / T)
    p_two = float(2.0 * (1.0 - stats.norm.cdf(abs(dm_stat))))
    # One-tailed: H1: model1 is WORSE than model2 (d_bar > 0 → model1 loses)
    # For "gcn better than vol": model1=gcn, so d_bar<0 → DM<0 → use left tail
    p_gcn_better = float(stats.norm.cdf(dm_stat))

    return {
        "T": int(T),
        "H_bandwidth": int(H),
        "d_bar": float(d_bar),
        "lrv": float(lrv),
        "dm_stat": float(dm_stat),
        "p_two_tailed": float(p_two),
        "p_one_tailed_gcn_better": float(p_gcn_better),
    }


def run_all_tests(predictions_csv: str) -> dict:
    df = pd.read_csv(predictions_csv)

    y = df["label"].values.astype(float)
    s_gcn = df["gcn_fusion_score"].values.astype(float)
    s_vol = df["vol_threshold_score"].values.astype(float)

    # Binary predictions at selected thresholds
    pred_gcn = (s_gcn >= GCN_THRESHOLD).astype(float)
    pred_vol = (s_vol >= VOL_THRESHOLD).astype(float)

    # ── Test 1: Brier score — gcn probability vs vol binary prediction ─────────
    # vol_threshold is a threshold classifier; its "probability" is 0 or 1.
    # Comparing gcn's calibrated P(y=1) against vol's hard forecast under Brier.
    # NOTE: vol_threshold_score is raw realized vol (not a probability);
    # we use binary predictions at threshold for a valid scoring-rule comparison.
    loss_gcn_brier = (y - s_gcn) ** 2
    loss_vol_brier = (y - pred_vol) ** 2   # pred_vol ∈ {0, 1}
    d_brier = loss_gcn_brier - loss_vol_brier
    brier_result = dm_test(d_brier, "brier")
    brier_result.update({
        "loss_gcn_mean": float(loss_gcn_brier.mean()),
        "loss_vol_mean": float(loss_vol_brier.mean()),
        "direction": "gcn_better" if d_brier.mean() < 0 else "vol_better",
        "conclusion": (
            "gcn_fusion has significantly lower Brier loss than vol_threshold (binary)"
            if brier_result["p_one_tailed_gcn_better"] < 0.05
            else "cannot reject equal Brier loss at p<0.05 (one-tailed)"
        ),
    })

    # ── Test 2: 0-1 loss at selected operating thresholds ─────────────────────
    # Both models thresholded; tests equal classification accuracy at Table 3 points.
    loss_gcn_01 = (pred_gcn != y).astype(float)
    loss_vol_01 = (pred_vol != y).astype(float)
    d_01 = loss_gcn_01 - loss_vol_01
    ol_result = dm_test(d_01, "0-1 loss")
    ol_result.update({
        "loss_gcn_mean": float(loss_gcn_01.mean()),
        "loss_vol_mean": float(loss_vol_01.mean()),
        "gcn_error_rate": float(loss_gcn_01.mean()),
        "vol_error_rate": float(loss_vol_01.mean()),
        "direction": "gcn_better" if d_01.mean() < 0 else "vol_better",
        "conclusion": (
            "gcn_fusion has significantly lower 0-1 loss at operating threshold"
            if ol_result["p_one_tailed_gcn_better"] < 0.05
            else "cannot reject equal 0-1 loss at p<0.05 (one-tailed)"
        ),
    })

    return {
        "gcn_threshold": GCN_THRESHOLD,
        "vol_threshold_value": VOL_THRESHOLD,
        "test_brier_score": brier_result,
        "test_01_loss": ol_result,
        "notes": (
            "vol_threshold_score is raw realized volatility (not a probability). "
            "Test 1 compares gcn probability scores against vol binary predictions "
            "under the Brier scoring rule. Test 2 compares binary accuracy at "
            "each model's selected operating threshold."
        ),
    }


if __name__ == "__main__":
    print(f"Loading predictions: {PREDICTIONS_CSV}\n")
    results = run_all_tests(PREDICTIONS_CSV)

    def print_test(name: str, r: dict) -> None:
        print(f"--- {name} ---")
        print(f"  Mean loss  gcn: {r['loss_gcn_mean']:.6f}")
        print(f"  Mean loss  vol: {r['loss_vol_mean']:.6f}")
        print(f"  d_bar:          {r['d_bar']:.6f}  ({r['direction']})")
        print(f"  DM statistic:   {r['dm_stat']:.4f}")
        print(f"  p (two-tailed): {r['p_two_tailed']:.4f}")
        print(f"  p (gcn better): {r['p_one_tailed_gcn_better']:.4f}")
        print(f"  Conclusion:     {r['conclusion']}\n")

    print_test("Test 1: Brier score (gcn prob vs vol binary)", results["test_brier_score"])
    print_test("Test 2: 0-1 loss at operating thresholds", results["test_01_loss"])

    with open(OUTPUT_JSON, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Results saved to: {OUTPUT_JSON}")
