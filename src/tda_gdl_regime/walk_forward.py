from __future__ import annotations

import pandas as pd

from .config import EvaluationConfig


def generate_walk_forward_folds(
    feature_frame: pd.DataFrame,
    cfg_evaluation: EvaluationConfig,
    purge_bars: int,
    embargo_bars: int,
) -> tuple[pd.DataFrame, list[dict[str, pd.DataFrame]]]:
    """
    Generate walk-forward expanding-window folds from a feature frame.

    The test split (last test_frac rows per symbol) is carved out first and never
    appears in any fold. The remaining rows form the walk-forward pool.

    For fold k (0-indexed):
        train = rows [0 : min_train_bars + k * val_bars]
        gap   = purge_bars + embargo_bars  (anti-leakage buffer)
        val   = rows [min_train_bars + k * val_bars + gap :
                      min_train_bars + k * val_bars + gap + val_bars]

    Folds are generated per symbol and then concatenated. Fold k is included only
    when both train and val windows are non-empty for every symbol in the frame.

    Returns:
        test_df: held-out test split (never in any fold)
        folds:   list of {"train": df, "val": df} dicts, one per fold
    """
    bars_per_year: int = cfg_evaluation.bars_per_day
    min_train_bars: int = cfg_evaluation.walk_forward_min_train_years * bars_per_year
    val_bars: int = cfg_evaluation.walk_forward_val_years * bars_per_year
    gap: int = max(purge_bars, 0) + max(embargo_bars, 0)
    test_frac: float = cfg_evaluation.test_frac

    # Accumulate per-symbol contributions for each fold index and the test split.
    # max_folds: upper bound on fold count; computed after first symbol.
    max_folds: int | None = None
    fold_train_parts: list[list[pd.DataFrame]] = []
    fold_val_parts: list[list[pd.DataFrame]] = []
    test_parts: list[pd.DataFrame] = []

    for _, sym_frame in feature_frame.groupby("symbol", sort=True):
        sym_frame = sym_frame.sort_values("sample_row_id").reset_index(drop=True)
        n = len(sym_frame)
        test_start_idx = n - int(n * test_frac)

        test_parts.append(sym_frame.iloc[test_start_idx:].copy())
        available = sym_frame.iloc[:test_start_idx]
        n_avail = len(available)

        # Determine how many folds this symbol supports.
        sym_fold_count = 0
        k = 0
        while True:
            train_end = min_train_bars + k * val_bars
            val_start = train_end + gap
            val_end = val_start + val_bars
            if val_end > n_avail or train_end == 0:
                break
            sym_fold_count += 1
            k += 1

        if max_folds is None:
            max_folds = sym_fold_count
        else:
            max_folds = min(max_folds, sym_fold_count)

        # Ensure fold_*_parts lists are large enough.
        while len(fold_train_parts) < sym_fold_count:
            fold_train_parts.append([])
            fold_val_parts.append([])

        for fold_k in range(sym_fold_count):
            train_end = min_train_bars + fold_k * val_bars
            val_start = train_end + gap
            val_end = val_start + val_bars
            fold_train_parts[fold_k].append(available.iloc[:train_end].copy())
            fold_val_parts[fold_k].append(available.iloc[val_start:val_end].copy())

    if max_folds is None or max_folds == 0:
        test_df = pd.concat(test_parts, ignore_index=True) if test_parts else feature_frame.iloc[0:0].copy()
        return test_df, []

    folds: list[dict[str, pd.DataFrame]] = []
    for fold_k in range(max_folds):
        train_parts = fold_train_parts[fold_k]
        val_parts = fold_val_parts[fold_k]
        if not train_parts or not val_parts:
            continue
        train_df = pd.concat(train_parts, ignore_index=True)
        val_df = pd.concat(val_parts, ignore_index=True)
        if train_df.empty or val_df.empty:
            continue
        folds.append({"train": train_df, "val": val_df})

    test_df = pd.concat(test_parts, ignore_index=True) if test_parts else feature_frame.iloc[0:0].copy()
    return test_df, folds
