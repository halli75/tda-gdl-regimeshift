"""
MS-GARCH Baseline (P1.2 part 2).

Markov-Switching variance model (Hamilton 1989 / Haas et al. 2004 spirit)
using statsmodels MarkovAutoregression with switching_variance=True.

Per symbol:
  - Fit 2-state Markov-switching AR(1) with switching variance on train+val returns
  - Identify high-variance state (state with larger σ²)
  - Filtered probability of being in high-variance state = regime score
  - Select threshold on val; evaluate on test with B=5,000 bootstrap

Requires:
  data/cache/{symbol}_1d_2007-01-02_end.csv
  research/bootstrap_rerun/final/predictions.csv

Outputs:
  paper/revisions/ms_garch_test_scores.csv
  paper/revisions/ms_garch_results.json

Usage:
    cd <repo_root>
    PYTHONPATH=src python paper/revisions/compute_ms_garch_baseline.py
"""
from __future__ import annotations

import json
import os
import sys
import warnings

import numpy as np
import pandas as pd

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(REPO_ROOT, "src"))
sys.path.insert(0, os.path.join(REPO_ROOT, "paper", "revisions"))

from tda_gdl_regime.evaluation import evaluate_predictions, select_best_threshold  # noqa: E402
from compute_m5_egarch_baseline import (  # noqa: E402
    _build_val_frame, _load_returns, _cache_path, align_scores_to_windows,
    SYMBOLS, TRAIN_END, VAL_END, TEST_START,
    EARLY_WARNING_BARS, BARS_PER_DAY, BOOTSTRAP_SAMPLES, BOOTSTRAP_BLOCK, THRESHOLD_GRID,
)

PREDICTIONS_CSV = os.path.join(REPO_ROOT, "research", "bootstrap_rerun", "final", "predictions.csv")
OUTPUT_JSON     = os.path.join(REPO_ROOT, "paper", "revisions", "ms_garch_results.json")
OUTPUT_SCORES   = os.path.join(REPO_ROOT, "paper", "revisions", "ms_garch_test_scores.csv")


def fit_ms_garch_for_symbol(sym: str) -> tuple[pd.Series, pd.Series]:
    """
    Returns (val_scores, test_scores) as pd.Series indexed by date.
    Score = filtered probability of being in the high-variance regime.
    """
    from statsmodels.tsa.regime_switching.markov_autoregression import MarkovAutoregression

    returns = _load_returns(sym)

    # Train on train+val period
    train_returns = returns[returns.index <= VAL_END]
    if len(train_returns) < 100:
        raise RuntimeError(f"{sym}: insufficient train data ({len(train_returns)} rows)")

    # Scale returns for numerical stability
    SCALE = 100.0
    y = (train_returns * SCALE).values

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        mod = MarkovAutoregression(
            y,
            k_regimes=2,
            order=1,
            switching_variance=True,
            switching_ar=False,
        )
        res = mod.fit(
            search_reps=20,
            search_scale=0.1,
            disp=False,
            maxiter=200,
        )

    # Identify high-variance state
    # res.params contains variances for each regime
    # Naming varies by statsmodels version; use smoothed_marginal_probabilities
    smoothed = res.smoothed_marginal_probabilities  # shape (T, k_regimes)

    # Variance params: 'sigma2[0]' and 'sigma2[1]' or indexed differently
    # Get regime variances from params
    param_names = res.model.param_names
    var_params = [(i, v) for i, (name, v) in enumerate(zip(param_names, res.params))
                  if "sigma2" in name.lower() or "var" in name.lower()]
    if len(var_params) >= 2:
        # high-variance state = index with larger variance
        high_state = int(np.argmax([v for _, v in var_params]))
    else:
        # Fallback: use regime with higher mean probability during volatile periods
        # Identify high-variance state by variance of returns in each regime
        state0_mask = smoothed[:, 0] > 0.5
        state1_mask = ~state0_mask
        var0 = float(np.var(y[state0_mask])) if state0_mask.sum() > 5 else 0.0
        var1 = float(np.var(y[state1_mask])) if state1_mask.sum() > 5 else 0.0
        high_state = 1 if var1 > var0 else 0

    # Filtered prob of high-variance state on train period
    filtered = res.filtered_marginal_probabilities[:, high_state]
    prob_series = pd.Series(filtered, index=train_returns.index[1:])  # AR(1) loses first obs

    # Re-run on full returns to get test scores (fixed params)
    all_returns = returns[returns.index <= returns.index[-1]]
    y_full = (all_returns * SCALE).values
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        mod_full = MarkovAutoregression(
            y_full,
            k_regimes=2,
            order=1,
            switching_variance=True,
            switching_ar=False,
        )
        res_full = mod_full.smooth(res.params)

    filtered_full = res_full.filtered_marginal_probabilities[:, high_state]
    prob_series_full = pd.Series(filtered_full, index=all_returns.index[1:])

    val_mask  = (prob_series_full.index >= TRAIN_END) & (prob_series_full.index <= VAL_END)
    test_mask = prob_series_full.index >= TEST_START

    return prob_series_full[val_mask], prob_series_full[test_mask]


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
        print(f"  Fitting MS-GARCH for {sym} ...")
        try:
            v, t = fit_ms_garch_for_symbol(sym)
            sym_val_scores[sym]  = v
            sym_test_scores[sym] = t
            print(f"    val={len(v)} scores  test={len(t)} scores  "
                  f"mean_prob={t.mean():.4f}")
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
    scores_df["ms_garch_score"] = test_scores
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

    print(f"\n  MS-GARCH  PR-AUC={pr_auc:.4f}  CI={ci}  F1={event_f1:.4f}  "
          f"Recall={recall:.4f}  FA/day={fa_per_day:.4f}")

    results = {
        "ms_garch": {
            "pr_auc":               pr_auc,
            "event_f1":             event_f1,
            "event_recall":         recall,
            "false_alarms_per_day": fa_per_day,
            "ci_pr_auc":            ci,
            "threshold":            round(best_thresh, 4),
        },
        "model_description": (
            "Markov-Switching AR(1) with 2 regimes and switching variance "
            "(statsmodels MarkovAutoregression, k_regimes=2, order=1, "
            "switching_variance=True). Score = filtered P(high-variance state)."
        ),
    }

    with open(OUTPUT_JSON, "w") as f:
        json.dump(results, f, indent=2)
    print(f"JSON saved: {OUTPUT_JSON}")


if __name__ == "__main__":
    main()
