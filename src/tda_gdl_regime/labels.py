from __future__ import annotations

import numpy as np
import pandas as pd

from .config import LabelConfig


def _forward_std(series: pd.Series, window: int) -> pd.Series:
    reversed_std = series.iloc[::-1].rolling(window=window, min_periods=window).std(ddof=0)
    return reversed_std.iloc[::-1].shift(-(window - 1))


def _merge_event_flags(flags: np.ndarray, gap: int, min_span: int) -> np.ndarray:
    merged = flags.astype(bool).copy()
    indices = np.flatnonzero(merged)
    if indices.size == 0:
        return merged
    last = indices[0]
    for current in indices[1:]:
        if current - last - 1 <= gap:
            merged[last : current + 1] = True
        last = current
    cleaned = merged.copy()
    start = None
    for idx, flag in enumerate(merged):
        if flag and start is None:
            start = idx
        if start is not None and (idx == len(merged) - 1 or not merged[idx + 1]):
            end = idx
            if end - start + 1 < min_span:
                cleaned[start : end + 1] = False
            start = None
    return cleaned


def event_spans(flags: np.ndarray | pd.Series) -> list[tuple[int, int]]:
    arr = np.asarray(flags, dtype=bool)
    spans: list[tuple[int, int]] = []
    start = None
    for idx, flag in enumerate(arr):
        if flag and start is None:
            start = idx
        if start is not None and (idx == len(arr) - 1 or not arr[idx + 1]):
            spans.append((start, idx))
            start = None
    return spans


def build_shift_event_labels(frame: pd.DataFrame, cfg: LabelConfig) -> pd.DataFrame:
    labeled_parts: list[pd.DataFrame] = []
    for _, symbol_frame in frame.groupby("symbol", sort=True):
        symbol_frame = symbol_frame.sort_values("timestamp").reset_index(drop=True).copy()
        returns = symbol_frame["return"]
        backward_vol = returns.rolling(
            window=cfg.volatility_window_bars, min_periods=cfg.volatility_window_bars
        ).std(ddof=0)
        forward_vol = _forward_std(returns, cfg.lookahead_bars)
        threshold = backward_vol.shift(1).rolling(
            window=cfg.threshold_lookback_bars,
            min_periods=cfg.min_history_bars,
        ).quantile(cfg.threshold_quantile)
        label_available = backward_vol.notna() & forward_vol.notna() & threshold.notna()
        raw_event = forward_vol >= threshold
        if cfg.positive_transition_only:
            raw_event &= backward_vol < threshold
        if cfg.vix_confirmation_col and cfg.vix_confirmation_col in symbol_frame.columns:
            if cfg.vix_confirmation_threshold is None:
                raise ValueError("VIX confirmation threshold is required when confirmation column is set")
            raw_event &= symbol_frame[cfg.vix_confirmation_col] >= cfg.vix_confirmation_threshold
        raw_event &= label_available
        merged_event = _merge_event_flags(
            raw_event.to_numpy(dtype=bool),
            gap=cfg.event_merge_gap,
            min_span=cfg.min_event_span,
        )
        event_ids = np.full(len(symbol_frame), -1, dtype=int)
        for event_id, (start, end) in enumerate(event_spans(merged_event)):
            event_ids[start : end + 1] = event_id
        symbol_frame["backward_volatility"] = backward_vol
        symbol_frame["forward_volatility"] = forward_vol
        symbol_frame["volatility_threshold"] = threshold
        symbol_frame["label_available"] = label_available
        symbol_frame["shift_event"] = merged_event.astype(int)
        symbol_frame["event_id"] = event_ids
        labeled_parts.append(symbol_frame)
    return pd.concat(labeled_parts, ignore_index=True)
