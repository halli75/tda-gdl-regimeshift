"""
HAR-RV Baseline (P1.2 part 1).

Heterogeneous Autoregressive Realized Volatility (Corsi 2009) baseline.

Per symbol:
  - Compute daily log-RV from returns cache
  - Fit HAR-RV via OLS on train+val:
      log(RV_t) = α + β_d·log(RV_{t-1}) + β_w·log(RV_{5d}) + β_m·log(RV_{22d}) + ε
  - Forecast RV on test split
  - Convert to exceedance probability: p_t = 1 - Fχ²(L·θ²/RV_hat, df=L)
    (same formula as EGARCH baseline)
  - Select threshold on val period (same protocol as production)
  - Evaluate on test with B=5,000 block bootstrap

Requires:
  data/cache/{symbol}_1d_2007-01-02_end.csv
  paper/revisions/egarch_test_scores.csv   (for alignment reference)
  research/bootstrap_rerun/final/predictions.csv

Outputs:
  paper/revisions/har_rv_test_scores.csv
  paper/revisions/har_rv_results.json

Usage:
    cd <repo_root>
    PYTHONPATH=src python paper/revisions/compute_har_rv_baseline.py
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np
import pandas as pd
from scipy.stats import chi2

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(REPO_ROOT, "src"))
sys.path.insert(0, os.path.join(REPO_ROOT, "paper", "revisions"))

from tda_gdl_regime.evaluation import evaluate_predictions, select_best_threshold  # noqa: E402
from compute_m5_egarch_baseline import (  # noqa: E402
    _build_val_frame, _load_returns, _cache_path, align_scores_to_windows,
    SYMBOLS, TRAIN_END, VAL_END, TEST_START,
    EARLY_WARNING_BARS, BARS_PER_DAY, BOOTSTRAP_SAMPLES, BOOTSTRAP_BLOCK, THRESHOLD_GRID,
)

PREDICTIONS_CSV  = os.path.join(REPO_ROOT, "research", "bootstrap_rerun", "final", "predictions.csv")
OUTPUT_JSON      = os.path.join(REPO_ROOT, "paper", "revisions", "har_rv_results.json")
OUTPUT_SCORES    = os.path.join(REPO_ROOT, "paper", "revisions", "har_rv_test_scores.csv")

# HAR-RV lags
LAG_D  = 1    # daily
LAG_W  = 5    # weekly (mean over 5 days)
LAG_M  = 22   # monthly (mean over 22 days)

# Same chi^2 formula as EGARCH baseline (L = lookahead bars)
L = 5   # lookahead bars matching labels.lookahead_bars


def _compute_rv_series(returns: pd.Series) -> pd.Series:
    """Daily realized variance proxy: squared log-returns."""
    rv = returns ** 2
    rv.name = "rv"
    return rv


def _har_features(log_rv: pd.Series) -> pd.DataFrame:
    """Build HAR-RV feature matrix from log-RV series."""
    df = pd.DataFrame({"log_rv": log_rv})
    df["lag1"]  = log_rv.shift(1)
    df["lag5"]  = log_rv.shift(1).rolling(LAG_W).mean()
    df["lag22"] = log_rv.shift(1).rolling(LAG_M).mean()
    return df.dropna()


def _exceedance_prob_rv(rv_hat: np.ndarray, theta: float) -> np.ndarray:
    """
    Convert RV forecast to exceedance probability using χ² CDF.
    p_t = 1 - Fχ²(L·θ²/rv_hat, df=L)
    """
    rv_hat = np.clip(rv_hat, 1e-12, None)
    x = L * theta**2 / rv_hat
    return 1.0 - chi2.cdf(x, df=L)


def fit_har_rv_for_symbol(sym: str) -> tuple[pd.Series, pd.Series]:
    """
    Returns (val_scores, test_scores) as pd.Series indexed by date.
    """
    returns = _load_returns(sym)
    rv      = _compute_rv_series(returns)
    # Clip zero RV (non-trading days) before log
    rv_clipped = rv.clip(lower=1e-12)
    log_rv  = np.log(rv_clipped)

    feat = _har_features(log_rv)

    # Train mask: up to TRAIN_END + VAL_END (fit on full train+val)
    train_mask = feat.index <= VAL_END
    test_mask  = feat.index >= TEST_START

    X_train = feat.loc[train_mask, ["lag1", "lag5", "lag22"]].values
    y_train = feat.loc[train_mask, "log_rv"].values

    if len(X_train) < 30:
        raise RuntimeError(f"{sym}: insufficient train data ({len(X_train)} rows)")

    # OLS: add intercept column
    X_train_int = np.column_stack([np.ones(len(X_train)), X_train])
    # Normal equations
    coeffs = np.linalg.lstsq(X_train_int, y_train, rcond=None)[0]

    # Rolling θ (rolling 90th percentile of returns, using only train period)
    # Use realized vol std proxy for theta
    raw_rv_train = rv.loc[returns.index <= VAL_END]
    theta = float(np.sqrt(raw_rv_train.quantile(0.90)))  # 90th pct of sqrt(rv) = 90th pct of abs returns

    # Predict on val + test
    val_mask = (feat.index >= TRAIN_END) & (feat.index <= VAL_END)

    def _predict_and_score(mask: pd.Series) -> pd.Series:
        X = feat.loc[mask, ["lag1", "lag5", "lag22"]].values
        X_int = np.column_stack([np.ones(len(X)), X])
        log_rv_hat = X_int @ coeffs
        rv_hat = np.exp(log_rv_hat)
        probs  = _exceedance_prob_rv(rv_hat, theta)
        return pd.Series(probs, index=feat.index[mask])

    val_scores  = _predict_and_score(val_mask)
    test_scores = _predict_and_score(test_mask)

    return val_scores, test_scores


def main() -> None:
    print("Loading predictions.csv ...")
    pred_df = pd.read_csv(PREDICTIONS_CSV, parse_dates=["timestamp"])
    test_df = pred_df[pred_df["timestamp"] >= TEST_START].copy().reset_index(drop=True)

    print("Building val frame from cache ...")
    val_df = _build_val_frame()
    print(f"  Test: {len(test_df):,}  |  Val: {len(val_df):,}")

    sym_val_scores:  dict[str, pd.Series] = {}
    sym_test_scores: dict[str, pd.Series] = {}

    for sym in SYMBOLS:
        if not os.path.exists(_cache_path(sym)):
            print(f"  SKIP {sym}: cache not found")
            continue
        print(f"  Fitting HAR-RV for {sym} ...")
        try:
            v, t = fit_har_rv_for_symbol(sym)
            sym_val_scores[sym]  = v
            sym_test_scores[sym] = t
        except Exception as exc:
            print(f"  WARNING {sym}: {exc}")

    def build_score_array(df: pd.DataFrame, sym_scores: dict) -> np.ndarray:
        scores = np.full(len(df), np.nan)
        for sym, sym_score in sym_scores.items():
            mask = (df["symbol"] == sym).to_numpy()
            if mask.sum() == 0:
                continue
            aligned = align_scores_to_windows(df["timestamp"][mask], sym_score)
            scores[mask] = aligned
        return np.nan_to_num(scores, nan=0.0)

    print("\nAligning scores ...")
    val_scores  = build_score_array(val_df,  sym_val_scores)
    test_scores = build_score_array(test_df, sym_test_scores)

    # Save per-sample test scores
    scores_df = test_df[["symbol", "timestamp"]].copy()
    scores_df["har_rv_score"] = test_scores
    scores_df.to_csv(OUTPUT_SCORES, index=False)
    print(f"Test scores saved: {OUTPUT_SCORES}")

    # Select threshold on val
    print("Selecting threshold on val ...")
    best_thresh, val_summary = select_best_threshold(
        frame=val_df,
        scores=val_scores,
        threshold_grid=THRESHOLD_GRID,
        selection_metric="event_f1",
        early_warning_bars=EARLY_WARNING_BARS,
        bars_per_day=BARS_PER_DAY,
        max_false_alarms_per_day=2.0,
        max_positive_rate=0.5,
    )
    print(f"  Best threshold: {best_thresh:.4f}  val event_f1: {val_summary.get('event_f1', 0):.4f}")

    # Evaluate on test
    print("Evaluating on test (B=5,000 bootstrap) ...")
    test_summary, _ = evaluate_predictions(
        frame=test_df,
        scores=test_scores,
        threshold=best_thresh,
        early_warning_bars=EARLY_WARNING_BARS,
        bars_per_day=BARS_PER_DAY,
        bootstrap_samples=BOOTSTRAP_SAMPLES,
        bootstrap_block_size=BOOTSTRAP_BLOCK,
    )

    pr_auc     = round(float(test_summary["pr_auc"]), 4)
    event_f1   = round(float(test_summary["event_f1"]), 4)
    recall     = round(float(test_summary["event_recall"]), 4)
    fa_per_day = round(float(test_summary["false_alarms_per_day"]), 4)
    ci         = [round(x, 4) for x in test_summary["bootstrap_ci"]["pr_auc"]]

    print(f"\n  HAR-RV  PR-AUC={pr_auc:.4f}  CI={ci}  F1={event_f1:.4f}  "
          f"Recall={recall:.4f}  FA/day={fa_per_day:.4f}")

    results = {
        "har_rv": {
            "pr_auc":               pr_auc,
            "event_f1":             event_f1,
            "event_recall":         recall,
            "false_alarms_per_day": fa_per_day,
            "ci_pr_auc":            ci,
            "threshold":            round(best_thresh, 4),
        },
        "model_description": (
            "HAR-RV (Corsi 2009): log(RV_t) = α + β_d·log(RV_{t-1}) "
            "+ β_w·log(RV_{5d}) + β_m·log(RV_{22d}); "
            "OLS fit on train+val; exceedance probability via χ²-CDF."
        ),
    }

    with open(OUTPUT_JSON, "w") as f:
        json.dump(results, f, indent=2)
    print(f"JSON saved: {OUTPUT_JSON}")


if __name__ == "__main__":
    main()
