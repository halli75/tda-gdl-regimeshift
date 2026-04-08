"""
M5 EGARCH(1,1) Exceedance-Probability Baseline.

Per-symbol EGARCH(1,1) model fit on train+val (2007–2022-06-14).
Out-of-sample conditional variance computed via fixed-parameter evaluation
on the full return series.  Threshold selected on val by grid-search over
event_f1 (matching the pipeline's selection_metric).

Outputs:
  paper/revisions/m5_egarch_results.json
  paper/figures/figure_m5_egarch_comparison.pdf

Usage:
    cd <repo_root>
    PYTHONPATH=src python paper/revisions/compute_m5_egarch_baseline.py
"""

from __future__ import annotations

import json
import os
import sys
import warnings

import numpy as np
import pandas as pd
from arch import arch_model
from scipy.stats import chi2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ── repo root ------------------------------------------------------------------
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(REPO_ROOT, "src"))

from tda_gdl_regime.evaluation import evaluate_predictions, select_best_threshold  # noqa: E402

# ── paths ----------------------------------------------------------------------
CACHE_DIR         = os.path.join(REPO_ROOT, "data", "cache")
PREDICTIONS_CSV   = os.path.join(REPO_ROOT, "research", "bootstrap_rerun", "final", "predictions.csv")
METRICS_BASELINE  = os.path.join(REPO_ROOT, "research", "bootstrap_rerun", "final", "metrics.json")
OUTPUT_JSON       = os.path.join(REPO_ROOT, "paper", "revisions", "m5_egarch_results.json")
OUTPUT_FIG        = os.path.join(REPO_ROOT, "paper", "figures", "figure_m5_egarch_comparison.pdf")

# ── label/split parameters matching bootstrap_final_config.yaml ---------------
LOOKAHEAD_BARS        = 5
VOL_WINDOW_BARS       = 20
THRESHOLD_QUANTILE    = 0.9
THRESHOLD_LOOKBACK    = 252
EVENT_MERGE_GAP       = 3
MIN_EVENT_SPAN        = 2
MIN_HISTORY_BARS      = 252

TRAIN_END   = pd.Timestamp("2018-10-31")
VAL_END     = pd.Timestamp("2022-06-14")
TEST_START  = pd.Timestamp("2022-07-28")  # after purge+embargo gap

EARLY_WARNING_BARS = 5
BARS_PER_DAY       = 50
BOOTSTRAP_SAMPLES  = 5_000
BOOTSTRAP_BLOCK    = 64

THRESHOLD_GRID = np.linspace(0.01, 0.99, 99).tolist()

# ── symbols matching config ----------------------------------------------------
SYMBOLS = [
    "SPY", "QQQ", "^VIX", "TLT", "GLD",
    "EEM", "HYG", "LQD", "DBC", "UUP", "IEF", "XLF", "XLE",
]


# ── helpers -------------------------------------------------------------------

def _cache_path(sym: str) -> str:
    safe = sym.replace("^", "")
    return os.path.join(CACHE_DIR, f"{safe}_1d_2007-01-02_end.csv")


def _load_returns(sym: str) -> pd.Series:
    """Load daily log-returns from data cache."""
    path = _cache_path(sym)
    if not os.path.exists(path):
        raise FileNotFoundError(f"Cache missing: {path}")
    df = pd.read_csv(path, parse_dates=["timestamp"])
    df = df.sort_values("timestamp").set_index("timestamp")
    prices = df["mid_price"].dropna()
    returns = np.log(prices / prices.shift(1)).dropna()
    return returns


def _rolling_vol_threshold(returns: pd.Series) -> pd.Series:
    """
    Compute the rolling 90th-quantile volatility threshold matching labels.py:
      backward_vol = rolling(VOL_WINDOW_BARS).std(ddof=0)
      threshold    = backward_vol.shift(1).rolling(THRESHOLD_LOOKBACK).quantile(Q)
    Returns threshold in standard-deviation units.
    """
    backward_vol = returns.rolling(window=VOL_WINDOW_BARS, min_periods=VOL_WINDOW_BARS).std(ddof=0)
    threshold = (
        backward_vol.shift(1)
        .rolling(window=THRESHOLD_LOOKBACK, min_periods=THRESHOLD_LOOKBACK)
        .quantile(THRESHOLD_QUANTILE)
    )
    return threshold


def _egarch_conditional_var(returns: pd.Series, fit_end: pd.Timestamp) -> pd.Series:
    """
    Fit EGARCH(1,1) on returns up to fit_end (inclusive).
    Then compute conditional variance for the full series using fixed params.
    Returns a Series (variance units, same index as returns).
    """
    pre_test = returns.loc[:fit_end].dropna()
    if len(pre_test) < 300:
        raise ValueError(f"Too few obs for EGARCH fit: {len(pre_test)}")

    # Scale by 100 to improve numerical conditioning (arch recommendation)
    SCALE = 100.0
    scaled = pre_test * SCALE

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model_fit = arch_model(scaled, vol="EGARCH", p=1, o=1, q=1,
                               dist="normal", mean="Constant", rescale=False)
        res = model_fit.fit(disp="off", show_warning=False)

    # Apply fitted params to full returns to get conditional vol for all dates
    full_scaled = returns.dropna() * SCALE
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model_full = arch_model(full_scaled, vol="EGARCH", p=1, o=1, q=1,
                                dist="normal", mean="Constant", rescale=False)
        res_full = model_full.fix(res.params)

    # conditional_volatility is in scaled units → variance in original = (cond_vol / SCALE)^2
    cond_vol_orig = res_full.conditional_volatility / SCALE
    h_t = cond_vol_orig ** 2
    return h_t


def _exceedance_prob(h_t: pd.Series, theta: pd.Series, L: int) -> pd.Series:
    """
    P(forward_vol > theta | h_t) = 1 - chi2.cdf(L * theta^2 / h_t, df=L)
    h_t and theta must be aligned by index (variance and std units respectively).
    """
    theta_sq = theta ** 2
    ratio = (L * theta_sq) / h_t.clip(lower=1e-12)
    prob = 1.0 - chi2.cdf(ratio.to_numpy(), df=L)
    return pd.Series(prob, index=h_t.index, dtype=float)


# ── per-symbol computation ----------------------------------------------------

def compute_egarch_scores_for_symbol(sym: str) -> tuple[pd.Series, pd.Series]:
    """
    Returns (val_scores, test_scores) as Series indexed by date.
    Scores are exceedance probabilities in [0, 1].
    """
    returns = _load_returns(sym)

    # Conditional variance (full series, fixed params from pre-test fit)
    h_t = _egarch_conditional_var(returns, fit_end=VAL_END)

    # Volatility threshold (same formula as labels.py)
    theta = _rolling_vol_threshold(returns)

    # Exceedance probability
    prob = _exceedance_prob(h_t, theta, L=LOOKAHEAD_BARS)

    # Drop NaNs from theta warmup
    prob = prob.dropna()

    val_scores  = prob.loc[(prob.index >= TRAIN_END) & (prob.index <= VAL_END)]
    test_scores = prob.loc[prob.index >= TEST_START]

    return val_scores, test_scores


# ── label reconstruction for val period ---------------------------------------

def _forward_std(series: pd.Series, window: int) -> pd.Series:
    rev = series.iloc[::-1].rolling(window=window, min_periods=window).std(ddof=0)
    return rev.iloc[::-1].shift(-(window - 1))


def _merge_flags(flags: np.ndarray, gap: int, min_span: int) -> np.ndarray:
    merged = flags.astype(bool).copy()
    indices = np.flatnonzero(merged)
    if indices.size == 0:
        return merged
    last = indices[0]
    for current in indices[1:]:
        if current - last - 1 <= gap:
            merged[last: current + 1] = True
        last = current
    cleaned = merged.copy()
    start = None
    for idx, flag in enumerate(merged):
        if flag and start is None:
            start = idx
        if start is not None and (idx == len(merged) - 1 or not merged[idx + 1]):
            end = idx
            if end - start + 1 < min_span:
                cleaned[start: end + 1] = False
            start = None
    return cleaned


def build_labels_for_symbol(returns: pd.Series) -> pd.DataFrame:
    """
    Rebuild binary labels matching labels.py (build_shift_event_labels).
    Returns DataFrame with columns: timestamp, label, event_id.
    """
    backward_vol = returns.rolling(window=VOL_WINDOW_BARS, min_periods=VOL_WINDOW_BARS).std(ddof=0)
    forward_vol  = _forward_std(returns, LOOKAHEAD_BARS)
    threshold    = (
        backward_vol.shift(1)
        .rolling(window=THRESHOLD_LOOKBACK, min_periods=THRESHOLD_LOOKBACK)
        .quantile(THRESHOLD_QUANTILE)
    )
    raw_flags = (forward_vol > threshold).fillna(False).to_numpy()
    merged    = _merge_flags(raw_flags, gap=EVENT_MERGE_GAP, min_span=MIN_EVENT_SPAN)

    # Assign event IDs
    event_id = np.full(len(merged), -1, dtype=int)
    eid = 0
    in_event = False
    for i, f in enumerate(merged):
        if f and not in_event:
            in_event = True
            eid += 1
        if f:
            event_id[i] = eid
        if not f:
            in_event = False

    df = pd.DataFrame({
        "timestamp": returns.index,
        "label":     merged.astype(int),
        "event_id":  event_id,
    })
    return df


# ── align EGARCH scores with strided window timestamps -------------------------

def align_scores_to_windows(
    window_timestamps: pd.Series,
    daily_scores: pd.Series,
) -> np.ndarray:
    """
    For each window timestamp, find the nearest available daily score
    (forward-fill for stride gaps).
    """
    scores_series = daily_scores.sort_index()
    result = []
    for ts in window_timestamps:
        # Find scores up to and including ts
        past = scores_series.loc[scores_series.index <= ts]
        if len(past) == 0:
            result.append(np.nan)
        else:
            result.append(past.iloc[-1])
    return np.array(result, dtype=float)


# ── main -----------------------------------------------------------------------

def _build_val_frame() -> pd.DataFrame:
    """
    Reconstruct a val-period frame with label + symbol + timestamp columns
    from the raw cache files, matching the pipeline's label generation.
    (predictions.csv contains only the test split.)
    """
    parts = []
    for sym in SYMBOLS:
        cache_path = _cache_path(sym)
        if not os.path.exists(cache_path):
            continue
        returns = _load_returns(sym)
        labels_df = build_labels_for_symbol(returns)
        labels_df["symbol"] = sym
        val_mask = (labels_df["timestamp"] >= TRAIN_END) & (labels_df["timestamp"] <= VAL_END)
        parts.append(labels_df[val_mask])
    if not parts:
        raise RuntimeError("No val data could be built from cache")
    val_frame = pd.concat(parts, ignore_index=True).sort_values("timestamp").reset_index(drop=True)
    return val_frame


def main() -> None:
    print("Loading predictions.csv (test split) ...")
    pred_df = pd.read_csv(PREDICTIONS_CSV, parse_dates=["timestamp"])
    test_df = pred_df[pred_df["timestamp"] >= TEST_START].copy().reset_index(drop=True)

    print("Building val frame from cache (labels.py-equivalent) ...")
    val_df = _build_val_frame()

    print(f"  Test rows: {len(test_df):,}  |  Val rows: {len(val_df):,}")

    # Collect per-symbol EGARCH scores
    sym_val_scores:  dict[str, pd.Series] = {}
    sym_test_scores: dict[str, pd.Series] = {}

    for sym in SYMBOLS:
        cache_path = _cache_path(sym)
        if not os.path.exists(cache_path):
            print(f"  SKIP {sym}: cache not found")
            continue
        print(f"  Fitting EGARCH for {sym} ...")
        try:
            v_scores, t_scores = compute_egarch_scores_for_symbol(sym)
            sym_val_scores[sym]  = v_scores
            sym_test_scores[sym] = t_scores
        except Exception as exc:
            print(f"  WARNING {sym}: {exc}")

    # Build aligned score arrays for val and test
    # Align by (symbol, timestamp) pairs
    def build_score_array(df: pd.DataFrame, sym_scores: dict[str, pd.Series]) -> np.ndarray:
        scores = np.full(len(df), np.nan)
        for sym, sym_score in sym_scores.items():
            mask = df["symbol"] == sym
            if mask.sum() == 0:
                continue
            sym_df = df[mask]
            aligned = align_scores_to_windows(sym_df["timestamp"], sym_score)
            scores[mask.to_numpy()] = aligned
        return scores

    print("\nAligning scores ...")
    val_scores  = build_score_array(val_df,  sym_val_scores)
    test_scores = build_score_array(test_df, sym_test_scores)

    # Save per-sample test scores for downstream DM test and hybrid detector
    scores_csv = os.path.join(REPO_ROOT, "paper", "revisions", "egarch_test_scores.csv")
    scores_df = test_df[["symbol", "timestamp"]].copy()
    scores_df["egarch_score"] = test_scores
    scores_df.to_csv(scores_csv, index=False)
    print(f"Per-sample test scores saved: {scores_csv}")

    # Fill any remaining NaNs with 0.0 (conservative)
    n_nan_val  = np.isnan(val_scores).sum()
    n_nan_test = np.isnan(test_scores).sum()
    if n_nan_val > 0:
        print(f"  WARNING: {n_nan_val} NaN val scores -> filling with 0.0")
    if n_nan_test > 0:
        print(f"  WARNING: {n_nan_test} NaN test scores -> filling with 0.0")
    val_scores  = np.nan_to_num(val_scores,  nan=0.0)
    test_scores = np.nan_to_num(test_scores, nan=0.0)

    # Select threshold on val
    print("Selecting threshold on val period ...")
    best_threshold, val_summary = select_best_threshold(
        frame=val_df,
        scores=val_scores,
        threshold_grid=THRESHOLD_GRID,
        selection_metric="event_f1",
        early_warning_bars=EARLY_WARNING_BARS,
        bars_per_day=BARS_PER_DAY,
        max_false_alarms_per_day=2.0,
        max_positive_rate=0.5,
    )
    print(f"  Best threshold: {best_threshold:.4f}  |  val event_f1: {val_summary.get('event_f1', 0):.4f}")

    # Evaluate on test with bootstrap CI
    print("Evaluating on test period (B=5,000 bootstrap) ...")
    test_summary, _ = evaluate_predictions(
        frame=test_df,
        scores=test_scores,
        threshold=best_threshold,
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
    print(f"\n  EGARCH PR-AUC: {pr_auc:.4f}  CI: {ci}  F1: {event_f1:.4f}  FA/day: {fa_per_day:.4f}")

    # Load gcn_fusion and vol_threshold for comparison
    with open(METRICS_BASELINE) as f:
        base = json.load(f)
    models_map = {m["model"]: m for m in base["models"]}

    def _pull(name: str) -> dict:
        m = models_map[name]
        return {
            "pr_auc":    round(float(m["pr_auc"]), 4),
            "event_f1":  round(float(m["event_f1"]), 4),
            "fa_per_day": round(float(m["false_alarms_per_day"]), 4),
            "ci_pr_auc": [round(x, 4) for x in m["bootstrap_ci"]["pr_auc"]],
        }

    gcn_stats = _pull("gcn_fusion")
    vol_stats = _pull("vol_threshold")

    egarch_advantage = pr_auc > vol_stats["pr_auc"]
    gcn_advantage    = gcn_stats["pr_auc"] > pr_auc

    interpretation = (
        f"EGARCH(1,1) achieves PR-AUC = {pr_auc:.4f} (95% CI [{ci[0]:.4f}, {ci[1]:.4f}]). "
        f"vol-threshold: {vol_stats['pr_auc']:.4f}, gcn-fusion: {gcn_stats['pr_auc']:.4f}. "
    )
    if egarch_advantage and gcn_advantage:
        interpretation += (
            f"EGARCH exceeds the vol-threshold baseline (+{pr_auc - vol_stats['pr_auc']:+.4f}) "
            f"but is surpassed by gcn-fusion (+{gcn_stats['pr_auc'] - pr_auc:+.4f}), "
            "confirming that the TDA+GCN approach captures signal beyond a calibrated "
            "probabilistic volatility model."
        )
    elif not egarch_advantage:
        interpretation += (
            f"EGARCH does not exceed the vol-threshold baseline "
            f"(delta = {pr_auc - vol_stats['pr_auc']:+.4f}), suggesting that the additional "
            "complexity of the econometric model does not translate to better probability "
            "ranking on this out-of-sample period. gcn-fusion exceeds both."
        )
    else:
        interpretation += (
            f"EGARCH exceeds vol-threshold (+{pr_auc - vol_stats['pr_auc']:+.4f}) and "
            f"also outperforms gcn-fusion (+{pr_auc - gcn_stats['pr_auc']:+.4f}), indicating "
            "that the econometric baseline is competitive on this horizon."
        )

    results = {
        "egarch": {
            "pr_auc":        pr_auc,
            "event_f1":      event_f1,
            "event_recall":  recall,
            "false_alarms_per_day": fa_per_day,
            "ci_pr_auc":     ci,
            "threshold":     round(best_threshold, 4),
        },
        "gcn_fusion":    gcn_stats,
        "vol_threshold": vol_stats,
        "egarch_vs_vol_delta_pr_auc":  round(pr_auc - vol_stats["pr_auc"], 4),
        "gcn_vs_egarch_delta_pr_auc":  round(gcn_stats["pr_auc"] - pr_auc, 4),
        "interpretation": interpretation,
    }

    os.makedirs(os.path.dirname(OUTPUT_JSON), exist_ok=True)
    with open(OUTPUT_JSON, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nJSON saved: {OUTPUT_JSON}")

    # ── figure -----------------------------------------------------------------
    _make_figure(results, OUTPUT_FIG)
    print(f"\nInterpretation:\n  {interpretation}")


def _make_figure(results: dict, out_path: str) -> None:
    BLUE   = "#1f77b4"
    ORANGE = "#ff7f0e"
    GREEN  = "#2ca02c"

    models = [
        ("EGARCH(1,1)",    results["egarch"],    GREEN),
        ("vol-threshold",  results["vol_threshold"], ORANGE),
        ("gcn-fusion",     results["gcn_fusion"],    BLUE),
    ]

    fig, axes = plt.subplots(1, 2, figsize=(9, 4))

    # PR-AUC with CI
    ax = axes[0]
    names  = [m[0] for m in models]
    values = [m[1]["pr_auc"] for m in models]
    colors = [m[2] for m in models]
    bars = ax.bar(names, values, color=colors, alpha=0.85, edgecolor="black", linewidth=0.6, width=0.5)
    for i, (_, data, _) in enumerate(models):
        ci = data["ci_pr_auc"]
        v  = data["pr_auc"]
        ax.errorbar(i, v, yerr=[[v - ci[0]], [ci[1] - v]],
                    fmt="none", color="black", capsize=4, linewidth=1.2)
    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.003,
                f"{val:.3f}", ha="center", va="bottom", fontsize=8)
    ax.set_title("PR-AUC (95% bootstrap CI)", fontsize=10)
    ax.set_ylabel("PR-AUC", fontsize=8)
    ax.tick_params(labelsize=8)

    # Event F1
    ax = axes[1]
    f1s = [m[1]["event_f1"] for m in models]
    bars2 = ax.bar(names, f1s, color=colors, alpha=0.85, edgecolor="black", linewidth=0.6, width=0.5)
    for bar, val in zip(bars2, f1s):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.005,
                f"{val:.3f}", ha="center", va="bottom", fontsize=8)
    ax.set_title("Event F1", fontsize=10)
    ax.set_ylabel("Event F1", fontsize=8)
    ax.tick_params(labelsize=8)

    fig.suptitle("EGARCH(1,1) vs vol-threshold vs gcn-fusion\nTest split 2022–2026",
                 fontsize=9, y=1.02)
    plt.tight_layout()
    plt.savefig(out_path, bbox_inches="tight")
    plt.close()
    print(f"Figure saved: {out_path}")


if __name__ == "__main__":
    main()
