"""
Hybrid EGARCH-GCN Detector.

Two hybrid strategies:
  1. EGARCH-gated GCN:  score = gcn_fusion_score  if egarch_score >= egarch_threshold
                               = 0                  otherwise
  2. Ensemble 50/50:    score = 0.5 * egarch_score + 0.5 * gcn_fusion_score

Threshold for each hybrid is selected on val period (same protocol as pipeline).
Evaluated on test split with B=5,000 block bootstrap CIs.

Requires:
  paper/revisions/egarch_test_scores.csv  (from compute_m5_egarch_baseline.py)
  research/bootstrap_rerun/final/predictions.csv

Outputs:
  paper/revisions/hybrid_detector_results.json
  paper/figures/figure_hybrid_detector.pdf

Usage:
    cd <repo_root>
    PYTHONPATH=src python paper/revisions/compute_hybrid_detector.py
"""

from __future__ import annotations

import json
import os
import sys

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(REPO_ROOT, "src"))

from tda_gdl_regime.evaluation import evaluate_predictions, select_best_threshold  # noqa: E402

# ── Paths ──────────────────────────────────────────────────────────────────────
EGARCH_SCORES_CSV  = os.path.join(REPO_ROOT, "paper", "revisions", "egarch_test_scores.csv")
PREDICTIONS_CSV    = os.path.join(REPO_ROOT, "research", "bootstrap_rerun", "final", "predictions.csv")
VAL_PREDICTIONS_CSV = os.path.join(REPO_ROOT, "paper", "revisions", "predictions_val.csv")
OUTPUT_JSON        = os.path.join(REPO_ROOT, "paper", "revisions", "hybrid_detector_results.json")
OUTPUT_FIG         = os.path.join(REPO_ROOT, "paper", "figures", "figure_hybrid_detector.pdf")

# Import val-building utilities from the EGARCH baseline script
sys.path.insert(0, os.path.join(REPO_ROOT, "paper", "revisions"))
from compute_m5_egarch_baseline import (  # noqa: E402
    _build_val_frame, SYMBOLS, TRAIN_END, VAL_END, TEST_START,
    compute_egarch_scores_for_symbol, align_scores_to_windows,
)

# ── Constants ──────────────────────────────────────────────────────────────────
EGARCH_THRESHOLD   = 0.47        # val-selected from EGARCH baseline
EARLY_WARNING_BARS = 5
BARS_PER_DAY       = 50
BOOTSTRAP_SAMPLES  = 5_000
BOOTSTRAP_BLOCK    = 64
THRESHOLD_GRID     = np.linspace(0.01, 0.99, 99).tolist()


def _build_merged(pred_df: pd.DataFrame, egarch_df: pd.DataFrame) -> pd.DataFrame:
    merged = pred_df.merge(
        egarch_df[["symbol", "timestamp", "egarch_score"]],
        on=["symbol", "timestamp"],
        how="left",
    )
    n_miss = merged["egarch_score"].isna().sum()
    if n_miss > 0:
        print(f"  WARNING: {n_miss} missing EGARCH scores -> filling 0.0")
    merged["egarch_score"] = merged["egarch_score"].fillna(0.0)
    return merged


def _evaluate_model(
    name: str,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    val_scores: np.ndarray,
    test_scores: np.ndarray,
) -> dict:
    """Select threshold on val, evaluate with bootstrap on test."""
    best_thresh, _ = select_best_threshold(
        frame=val_df,
        scores=val_scores,
        threshold_grid=THRESHOLD_GRID,
        selection_metric="event_f1",
        early_warning_bars=EARLY_WARNING_BARS,
        bars_per_day=BARS_PER_DAY,
        max_false_alarms_per_day=2.0,
        max_positive_rate=0.5,
    )
    summary, _ = evaluate_predictions(
        frame=test_df,
        scores=test_scores,
        threshold=best_thresh,
        early_warning_bars=EARLY_WARNING_BARS,
        bars_per_day=BARS_PER_DAY,
        bootstrap_samples=BOOTSTRAP_SAMPLES,
        bootstrap_block_size=BOOTSTRAP_BLOCK,
    )
    result = {
        "pr_auc":        round(float(summary["pr_auc"]), 4),
        "event_f1":      round(float(summary["event_f1"]), 4),
        "event_recall":  round(float(summary["event_recall"]), 4),
        "false_alarms_per_day": round(float(summary["false_alarms_per_day"]), 4),
        "ci_pr_auc":     [round(x, 4) for x in summary["bootstrap_ci"]["pr_auc"]],
        "threshold":     round(best_thresh, 4),
    }
    print(f"  {name:20s}  PR-AUC={result['pr_auc']:.4f}  CI={result['ci_pr_auc']}  "
          f"F1={result['event_f1']:.4f}  FA/day={result['false_alarms_per_day']:.4f}")
    return result


def main() -> None:
    print("Loading data ...")
    pred_df   = pd.read_csv(PREDICTIONS_CSV, parse_dates=["timestamp"])
    egarch_df = pd.read_csv(EGARCH_SCORES_CSV, parse_dates=["timestamp"])

    test_df = pred_df[pred_df["timestamp"] >= TEST_START].copy().reset_index(drop=True)

    # ── Val frame: use predictions_val.csv when available (properly aligned) ────
    gcn_val_available = os.path.exists(VAL_PREDICTIONS_CSV)

    # Re-compute EGARCH val scores (needed for both val frames)
    print("\nComputing EGARCH val scores ...")
    sym_val_scores: dict[str, pd.Series] = {}
    for sym in SYMBOLS:
        cache_path = os.path.join(REPO_ROOT, "data", "cache",
                                  f"{sym.replace('^','')}_1d_2007-01-02_end.csv")
        if not os.path.exists(cache_path):
            continue
        try:
            v_scores, _ = compute_egarch_scores_for_symbol(sym)
            sym_val_scores[sym] = v_scores
        except Exception as exc:
            print(f"  WARNING {sym}: {exc}")

    def _build_val_scores(df: pd.DataFrame) -> np.ndarray:
        scores = np.full(len(df), np.nan)
        for sym, sym_score in sym_val_scores.items():
            mask = df["symbol"] == sym
            if mask.sum() == 0:
                continue
            aligned = align_scores_to_windows(df[mask]["timestamp"], sym_score)
            scores[mask.to_numpy()] = aligned
        return np.nan_to_num(scores, nan=0.0)

    if gcn_val_available:
        # Use predictions_val.csv as val frame: properly aligned to feature windows
        print(f"\nUsing predictions_val.csv as val frame (properly aligned) ...")
        val_gcn_df = pd.read_csv(VAL_PREDICTIONS_CSV, parse_dates=["timestamp"])
        val_df     = val_gcn_df[["symbol", "timestamp", "label"]].copy().reset_index(drop=True)
        val_gcn    = val_gcn_df["gcn_fusion_score"].to_numpy()
        val_egarch = _build_val_scores(val_df)
        gcn_val_source = "predictions_val.csv (actual GCN val predictions, feature-window aligned)"
        print(f"  Val rows: {len(val_df):,}  Positive: {val_df['label'].sum()}")
        print(f"  GCN val mean={val_gcn.mean():.4f}  EGARCH val mean={val_egarch.mean():.4f}")
    else:
        # Fallback: rebuild from daily cache; GCN proxy = EGARCH scores
        print("WARNING: predictions_val.csv not found.")
        print("  Run compute_gcn_val_predictions.py for actual GCN val scores.")
        print("  Falling back to EGARCH proxy for GCN component on val.")
        val_df     = _build_val_frame()
        val_egarch = _build_val_scores(val_df)
        val_gcn    = val_egarch.copy()   # EGARCH proxy
        gcn_val_source = "EGARCH val scores (proxy -- predictions_val.csv missing)"

    # Merge EGARCH scores into test
    test_merged = _build_merged(test_df, egarch_df)
    print(f"  Test: {len(test_df):,}  |  Val (hybrid): {len(val_df):,}")

    test_egarch = test_merged["egarch_score"].to_numpy()
    test_gcn    = test_merged["gcn_fusion_score"].to_numpy()

    # ── Strategy 1: EGARCH-gated GCN ─────────────────────────────────────────
    # hybrid = gcn_score if egarch >= threshold, else 0
    test_gated = np.where(test_egarch >= EGARCH_THRESHOLD, test_gcn, 0.0)
    # Val: use EGARCH for gating signal (always available); apply to val_gcn for the pass-through
    val_gated  = np.where(val_egarch >= EGARCH_THRESHOLD, val_gcn, 0.0)

    # ── Strategy 2: Ensemble 50/50 ────────────────────────────────────────────
    test_ensemble = 0.5 * test_egarch + 0.5 * test_gcn
    val_ensemble  = 0.5 * val_egarch + 0.5 * val_gcn

    print("\nEvaluating hybrids on test split (B=5,000 bootstrap) ...")
    results = {}
    results["egarch_gated"]   = _evaluate_model("EGARCH-gated GCN",   val_df, test_df, val_gated,    test_gated)
    results["ensemble_50_50"] = _evaluate_model("Ensemble 50/50",      val_df, test_df, val_ensemble, test_ensemble)

    # Reference models (from m5_egarch_results.json)
    with open(os.path.join(REPO_ROOT, "paper", "revisions", "m5_egarch_results.json")) as f:
        m5 = json.load(f)
    results["egarch"]        = m5["egarch"]
    results["gcn_fusion"]    = m5["gcn_fusion"]
    results["vol_threshold"] = m5["vol_threshold"]

    # Best hybrid
    best = max(["egarch_gated", "ensemble_50_50"],
               key=lambda k: results[k]["pr_auc"])
    results["best_hybrid"] = best

    results["gcn_val_source"] = gcn_val_source

    with open(OUTPUT_JSON, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nJSON saved: {OUTPUT_JSON}")
    print(f"Best hybrid: {best} (PR-AUC={results[best]['pr_auc']:.4f})")
    print(f"GCN val source: {gcn_val_source}")

    _make_figure(results, OUTPUT_FIG)


def _make_figure(results: dict, out_path: str) -> None:
    COLORS = {
        "EGARCH(1,1)":     "#2ca02c",
        "Hybrid (gated)":  "#9467bd",
        "Ensemble 50/50":  "#8c564b",
        "gcn-fusion":      "#1f77b4",
        "vol-threshold":   "#ff7f0e",
    }

    models = [
        ("EGARCH(1,1)",    results["egarch"]),
        ("Hybrid (gated)", results["egarch_gated"]),
        ("Ensemble 50/50", results["ensemble_50_50"]),
        ("gcn-fusion",     results["gcn_fusion"]),
        ("vol-threshold",  results["vol_threshold"]),
    ]

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))

    for ax, (metric_label, key) in zip(axes, [("PR-AUC", "pr_auc"), ("Event F1", "event_f1")]):
        names  = [m[0] for m in models]
        vals   = [m[1].get(key, m[1].get("pr_auc" if key=="pr_auc" else "event_f1", 0)) for m in models]
        colors = [COLORS[n] for n in names]

        bars = ax.bar(names, vals, color=colors, alpha=0.85,
                      edgecolor="black", linewidth=0.6, width=0.55)

        if key == "pr_auc":
            for i, (_, data) in enumerate(models):
                ci = data.get("ci_pr_auc", [vals[i], vals[i]])
                v  = vals[i]
                ax.errorbar(i, v, yerr=[[v - ci[0]], [ci[1] - v]],
                            fmt="none", color="black", capsize=4, linewidth=1.2)

        for bar, val in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + max(vals) * 0.02,
                    f"{val:.3f}", ha="center", va="bottom", fontsize=7.5)

        ax.set_title(f"{metric_label}{' (95% CI)' if key == 'pr_auc' else ''}", fontsize=10)
        ax.set_ylabel(metric_label, fontsize=8)
        ax.set_xticklabels(names, rotation=12, ha="right", fontsize=8)
        ax.tick_params(labelsize=8)

    fig.suptitle("Hybrid EGARCH-GCN Detector vs Standalone Models\nTest split 2022–2026",
                 fontsize=9, y=1.02)
    plt.tight_layout()
    plt.savefig(out_path, bbox_inches="tight")
    plt.close()
    print(f"Figure saved: {out_path}")


if __name__ == "__main__":
    main()
