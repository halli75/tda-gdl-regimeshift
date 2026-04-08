"""
Per-symbol EGARCH PR-AUC (P3.12).

Computes EGARCH PR-AUC for each symbol on the test split and cross-references
with gcn-fusion per-symbol PR-AUC from per_symbol_metrics.csv.

Requires:
  paper/revisions/egarch_test_scores.csv
  research/bootstrap_rerun/final/predictions.csv
  research/bootstrap_rerun/final/tables/per_symbol_metrics.csv

Outputs:
  paper/revisions/egarch_per_symbol_results.json

Usage:
    cd <repo_root>
    python paper/revisions/compute_egarch_per_symbol.py
"""
from __future__ import annotations

import json
import os

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

EGARCH_SCORES_CSV   = os.path.join(REPO_ROOT, "paper", "revisions", "egarch_test_scores.csv")
PREDICTIONS_CSV     = os.path.join(REPO_ROOT, "research", "bootstrap_rerun", "final", "predictions.csv")
PER_SYMBOL_METRICS  = os.path.join(REPO_ROOT, "research", "bootstrap_rerun", "final", "tables", "per_symbol_metrics.csv")
OUTPUT_JSON         = os.path.join(REPO_ROOT, "paper", "revisions", "egarch_per_symbol_results.json")


def main() -> None:
    print("Loading data ...")
    pred_df   = pd.read_csv(PREDICTIONS_CSV,   parse_dates=["timestamp"])
    egarch_df = pd.read_csv(EGARCH_SCORES_CSV, parse_dates=["timestamp"])

    merged = pred_df.merge(
        egarch_df[["symbol", "timestamp", "egarch_score"]],
        on=["symbol", "timestamp"],
        how="left",
    )
    n_miss = merged["egarch_score"].isna().sum()
    if n_miss > 0:
        print(f"  WARNING: {n_miss} missing EGARCH scores -> filling 0.0")
    merged["egarch_score"] = merged["egarch_score"].fillna(0.0)

    # Load gcn-fusion per-symbol PR-AUC
    try:
        sym_metrics = pd.read_csv(PER_SYMBOL_METRICS)
        gcn_pr_auc_map: dict[str, float] = {}
        for _, row in sym_metrics.iterrows():
            if "gcn_fusion" in str(row.get("model", "")):
                gcn_pr_auc_map[row["symbol"]] = float(row["pr_auc"])
        # Alternative: look for column named gcn_fusion_pr_auc or model column
        if not gcn_pr_auc_map:
            print("  Attempting alternate column layout for per_symbol_metrics.csv ...")
            print(f"  Columns: {list(sym_metrics.columns)}")
            # If there's a single pr_auc column with model column
            for _, row in sym_metrics.iterrows():
                gcn_pr_auc_map[row["symbol"]] = float(row["pr_auc"])
    except Exception as exc:
        print(f"  WARNING: could not load per_symbol_metrics: {exc}")
        gcn_pr_auc_map = {}

    symbols = sorted(merged["symbol"].unique())
    print(f"\nComputing per-symbol EGARCH PR-AUC for {len(symbols)} symbols ...")

    per_symbol: dict[str, dict] = {}
    for sym in symbols:
        df_sym = merged[merged["symbol"] == sym]
        y      = df_sym["label"].to_numpy(dtype=float)
        s_eg   = df_sym["egarch_score"].to_numpy(dtype=float)
        s_gcn  = df_sym["gcn_fusion_score"].to_numpy(dtype=float)

        n_events = int((y == 1).sum())
        n_total  = len(y)

        if n_events == 0:
            eg_pr_auc = float("nan")
        else:
            eg_pr_auc = round(float(average_precision_score(y, s_eg)), 4)

        gcn_pr_auc = round(float(average_precision_score(y, s_gcn)), 4) if n_events > 0 else float("nan")

        delta = round(gcn_pr_auc - eg_pr_auc, 4) if n_events > 0 else float("nan")

        per_symbol[sym] = {
            "n_windows":    n_total,
            "n_events":     n_events,
            "egarch_pr_auc": eg_pr_auc,
            "gcn_pr_auc":    gcn_pr_auc,
            "gcn_minus_egarch": delta,
        }
        marker = " << GCN wins" if isinstance(delta, float) and delta > 0 else (" << EGARCH wins" if isinstance(delta, float) and delta < 0 else "")
        print(f"  {sym:6s}  EGARCH={eg_pr_auc:.4f}  GCN={gcn_pr_auc:.4f}  d={delta:+.4f}{marker}")

    # Summary
    valid = {s: v for s, v in per_symbol.items() if not (isinstance(v["egarch_pr_auc"], float) and v["egarch_pr_auc"] != v["egarch_pr_auc"])}
    eg_vals  = [v["egarch_pr_auc"] for v in valid.values()]
    gcn_vals = [v["gcn_pr_auc"]    for v in valid.values()]
    gcn_wins = sum(1 for s in valid if valid[s]["gcn_minus_egarch"] > 0)

    output = {
        "per_symbol": per_symbol,
        "summary": {
            "n_symbols": len(symbols),
            "mean_egarch_pr_auc": round(float(np.mean(eg_vals)), 4),
            "mean_gcn_pr_auc":    round(float(np.mean(gcn_vals)), 4),
            "gcn_wins_count":     gcn_wins,
            "egarch_wins_count":  len(valid) - gcn_wins,
            "best_egarch_symbol": max(valid, key=lambda s: valid[s]["egarch_pr_auc"]),
            "best_gcn_symbol":    max(valid, key=lambda s: valid[s]["gcn_pr_auc"]),
        },
    }

    with open(OUTPUT_JSON, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nJSON saved: {OUTPUT_JSON}")
    print(f"GCN wins: {gcn_wins}/{len(valid)} symbols  "
          f"|  Mean EGARCH={output['summary']['mean_egarch_pr_auc']:.4f}  "
          f"Mean GCN={output['summary']['mean_gcn_pr_auc']:.4f}")


if __name__ == "__main__":
    main()
