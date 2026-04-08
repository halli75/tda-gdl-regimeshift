from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .config import FeatureConfig
from .labels import event_spans
from .tda_features import topology_feature_vector


@dataclass
class WindowSample:
    symbol: str
    timestamp: object
    sample_row_id: int
    label: int
    event_id: int
    backward_volatility: float
    forward_volatility: float
    volatility_threshold: float
    window_returns: np.ndarray


def _safe_autocorr(values: np.ndarray) -> float:
    if len(values) < 2:
        return 0.0
    left = values[:-1]
    right = values[1:]
    left_std = left.std()
    right_std = right.std()
    if left_std == 0.0 or right_std == 0.0:
        return 0.0
    return float(np.corrcoef(left, right)[0, 1])


def _safe_skew(values: np.ndarray) -> float:
    centered = values - values.mean()
    std = centered.std()
    if std == 0.0:
        return 0.0
    return float(np.mean((centered / std) ** 3))


def _safe_kurtosis(values: np.ndarray) -> float:
    centered = values - values.mean()
    std = centered.std()
    if std == 0.0:
        return 0.0
    return float(np.mean((centered / std) ** 4) - 3.0)


def _trend_slope(values: np.ndarray) -> float:
    if len(values) < 2:
        return 0.0
    cumulative = np.cumsum(values)
    x_axis = np.arange(len(values), dtype=float)
    slope, _ = np.polyfit(x_axis, cumulative, 1)
    return float(slope)


def compute_classical_features(window_returns: np.ndarray) -> dict[str, float]:
    values = np.asarray(window_returns, dtype=float)
    if values.size < 2:
        raise ValueError("Window must contain at least two returns")
    realized_vol = float(values.std(ddof=0))
    adjacent_abs = np.abs(values[1:]) * np.abs(values[:-1])
    bipower = float((np.pi / 2.0) * adjacent_abs.mean()) if adjacent_abs.size else 0.0
    downside = values[values < 0]
    upside = values[values > 0]
    sign_flips = float(np.mean(np.sign(values[1:]) != np.sign(values[:-1]))) if len(values) > 1 else 0.0
    return {
        "cls_realized_volatility": realized_vol,
        "cls_bipower_variation": bipower,
        "cls_skewness": _safe_skew(values),
        "cls_kurtosis": _safe_kurtosis(values),
        "cls_lag1_autocorr": _safe_autocorr(values),
        "cls_downside_semivariance": float(np.mean(downside**2)) if downside.size else 0.0,
        "cls_upside_semivariance": float(np.mean(upside**2)) if upside.size else 0.0,
        "cls_cumulative_return": float(np.exp(values.sum()) - 1.0),
        "cls_trend_slope": _trend_slope(values),
        "cls_sign_flip_rate": sign_flips,
        "cls_mean_abs_return": float(np.mean(np.abs(values))),
    }


def compute_vxx_tailored_features(window_returns: np.ndarray, cfg: FeatureConfig) -> dict[str, float]:
    values = np.asarray(window_returns, dtype=float)
    if values.size < 2:
        raise ValueError("Window must contain at least two returns")
    horizon = max(2, min(int(cfg.vxx_tailored_short_horizon), len(values)))
    recent = values[-horizon:]
    base_scale = float(np.std(values, ddof=0))
    if base_scale == 0.0:
        base_scale = 1.0
    recent_scale = float(np.std(recent, ddof=0))
    normalized = values / base_scale
    normalized_topology, normalized_cols = topology_feature_vector(
        series=normalized,
        embed_dim=cfg.embed_dim,
        embed_tau=cfg.embed_tau,
        radii=cfg.vxx_tailored_radii,
        image_bins=cfg.persistence_image_bins,
        feature_sets=cfg.vxx_tailored_topology_feature_sets,
    )
    features = {
        "vxx_short_realized_volatility": recent_scale,
        "vxx_short_long_vol_ratio": float(recent_scale / (base_scale + 1e-8)),
        "vxx_jump_intensity": float(np.max(np.abs(recent)) / (np.mean(np.abs(values)) + 1e-8)),
        "vxx_recent_abs_return_mean": float(np.mean(np.abs(recent))),
        "vxx_recent_signed_jump_z": float(recent[-1] / (base_scale + 1e-8)),
        "vxx_tail_concentration": float(np.sum(np.abs(recent) >= np.quantile(np.abs(recent), 0.8)) / len(recent)),
    }
    for column, value in zip(normalized_cols, normalized_topology):
        features[f"vxx_{column}"] = float(value)
    return features


def iter_window_samples(frame: pd.DataFrame, cfg: FeatureConfig) -> list[WindowSample]:
    samples: list[WindowSample] = []
    window_bars = cfg.window_bars
    for symbol, symbol_frame in frame.groupby("symbol", sort=True):
        symbol_frame = symbol_frame.sort_values("timestamp").reset_index(drop=True)
        for end_idx in range(window_bars - 1, len(symbol_frame), cfg.stride_bars):
            if not bool(symbol_frame.loc[end_idx, "label_available"]):
                continue
            start_idx = end_idx - window_bars + 1
            samples.append(
                WindowSample(
                    symbol=symbol,
                    timestamp=symbol_frame.loc[end_idx, "timestamp"],
                    sample_row_id=int(symbol_frame.loc[end_idx, "row_id"]),
                    label=int(symbol_frame.loc[end_idx, "shift_event"]),
                    event_id=int(symbol_frame.loc[end_idx, "event_id"]),
                    backward_volatility=float(symbol_frame.loc[end_idx, "backward_volatility"]),
                    forward_volatility=float(symbol_frame.loc[end_idx, "forward_volatility"]),
                    volatility_threshold=float(symbol_frame.loc[end_idx, "volatility_threshold"]),
                    window_returns=symbol_frame.loc[start_idx:end_idx, "return"].to_numpy(dtype=float),
                )
            )
    return samples


def build_feature_frame(frame: pd.DataFrame, cfg: FeatureConfig) -> pd.DataFrame:
    records: list[dict[str, float | int | str]] = []
    for sample in iter_window_samples(frame, cfg):
        classical = compute_classical_features(sample.window_returns)
        topology_values, topology_cols = topology_feature_vector(
            series=sample.window_returns,
            embed_dim=cfg.embed_dim,
            embed_tau=cfg.embed_tau,
            radii=cfg.betti_radii,
            image_bins=cfg.persistence_image_bins,
            feature_sets=cfg.topology_feature_sets,
        )
        record: dict[str, float | int | str] = {
            "symbol": sample.symbol,
            "timestamp": sample.timestamp,
            "sample_row_id": sample.sample_row_id,
            "label": sample.label,
            "event_id": sample.event_id,
            "backward_volatility": sample.backward_volatility,
            "forward_volatility": sample.forward_volatility,
            "volatility_threshold": sample.volatility_threshold,
        }
        record.update(classical)
        if cfg.enable_vxx_tailored:
            record.update(compute_vxx_tailored_features(sample.window_returns, cfg))
        for column, value in zip(topology_cols, topology_values):
            record[column] = float(value)
        records.append(record)
    if not records:
        raise ValueError("Feature extraction produced no samples")
    features = pd.DataFrame.from_records(records)
    if cfg.include_symbol_one_hot:
        symbol_dummies = pd.get_dummies(features["symbol"], prefix="sym", dtype=float)
        features = pd.concat([features, symbol_dummies], axis=1)
    return features


def add_cross_asset_features(
    feature_frame: pd.DataFrame,
    price_frame: pd.DataFrame,
    window: int = 60,
) -> pd.DataFrame:
    """Add rolling cross-asset correlation features to an existing feature frame.

    Computes pairwise rolling Pearson correlations between SPY, QQQ, and VXX
    log returns using the full price frame, then joins them to each sample row
    by timestamp. Columns are prefixed with ``xcorr_`` so they are picked up
    by ``feature_groups`` and included in the ``combined`` feature group.

    For timestamps where VXX is unavailable (pre-2009), the VXX-related
    correlation columns are filled with 0.0 (neutral / no signal).
    """
    returns_wide = (
        price_frame[["timestamp", "symbol", "return"]]
        .drop_duplicates(subset=["timestamp", "symbol"])
        .pivot(index="timestamp", columns="symbol", values="return")
        .sort_index()
    )
    pairs = [
        ("SPY", "^VIX", "xcorr_spy_vix"),
        ("QQQ", "^VIX", "xcorr_qqq_vix"),
        ("SPY", "QQQ",  "xcorr_spy_qqq"),
        ("SPY", "TLT",  "xcorr_spy_tlt"),
        ("SPY", "GLD",  "xcorr_spy_gld"),
        ("TLT", "^VIX", "xcorr_tlt_vix"),
        ("GLD", "^VIX", "xcorr_gld_vix"),
        ("SPY", "HYG",  "xcorr_spy_hyg"),
        ("SPY", "EEM",  "xcorr_spy_eem"),
        ("LQD", "HYG",  "xcorr_lqd_hyg"),
        ("UUP", "^VIX", "xcorr_uup_vix"),
    ]
    corr_records: dict[str, pd.Series] = {}
    for sym_a, sym_b, col_name in pairs:
        if sym_a in returns_wide.columns and sym_b in returns_wide.columns:
            corr_records[col_name] = returns_wide[sym_a].rolling(window).corr(returns_wide[sym_b])
    if not corr_records:
        return feature_frame
    corr_frame = pd.DataFrame(corr_records).fillna(0.0).reset_index()
    return feature_frame.merge(corr_frame, on="timestamp", how="left").fillna(0.0)


def feature_groups(feature_frame: pd.DataFrame) -> dict[str, list[str]]:
    symbol_cols = [column for column in feature_frame.columns if column.startswith("sym_")]
    tailored_cols = [column for column in feature_frame.columns if column.startswith("vxx_")]
    xcorr_cols = [column for column in feature_frame.columns if column.startswith("xcorr_")]
    classical_cols = [column for column in feature_frame.columns if column.startswith("cls_")] + tailored_cols + symbol_cols + xcorr_cols
    topology_cols = [column for column in feature_frame.columns if column.startswith("top_")] + symbol_cols
    combined_cols = classical_cols + [column for column in topology_cols if column not in symbol_cols]
    return {
        "classical": classical_cols,
        "topology": topology_cols,
        "combined": combined_cols,
    }


def split_feature_frame(
    feature_frame: pd.DataFrame,
    train_frac: float,
    val_frac: float,
    test_frac: float,
    purge_bars: int = 0,
    embargo_bars: int = 0,
) -> dict[str, pd.DataFrame]:
    _ = test_frac
    partitions: dict[str, list[pd.DataFrame]] = {"train": [], "val": [], "test": []}
    gap = max(int(purge_bars), 0) + max(int(embargo_bars), 0)
    for _, symbol_frame in feature_frame.groupby("symbol", sort=True):
        symbol_frame = symbol_frame.sort_values("sample_row_id").reset_index(drop=True)
        total = len(symbol_frame)
        train_end = int(total * train_frac)
        val_end = train_end + int(total * val_frac)
        if train_end == 0 or val_end <= train_end or val_end >= total:
            raise ValueError("Chronological split produced an empty partition")
        val_start_id = int(symbol_frame.loc[train_end, "sample_row_id"])
        test_start_id = int(symbol_frame.loc[val_end, "sample_row_id"])

        train_part = symbol_frame.loc[symbol_frame["sample_row_id"] < val_start_id - gap].copy()
        val_part = symbol_frame.loc[
            (symbol_frame["sample_row_id"] >= val_start_id)
            & (symbol_frame["sample_row_id"] < test_start_id - gap)
        ].copy()
        test_part = symbol_frame.loc[symbol_frame["sample_row_id"] >= test_start_id].copy()

        if train_part.empty or val_part.empty or test_part.empty:
            raise ValueError("Purged chronological split produced an empty partition")

        partitions["train"].append(train_part.reset_index(drop=True))
        partitions["val"].append(val_part.reset_index(drop=True))
        partitions["test"].append(test_part.reset_index(drop=True))
    return {
        name: pd.concat(parts, ignore_index=True).sort_values(["symbol", "sample_row_id"]).reset_index(drop=True)
        for name, parts in partitions.items()
    }


def split_frame_summary(splits: dict[str, pd.DataFrame]) -> dict[str, dict[str, object]]:
    summary: dict[str, dict[str, object]] = {}
    for split_name, frame in splits.items():
        labels = frame["label"].to_numpy(dtype=int)
        event_count = 0
        for _, symbol_frame in frame.groupby("symbol", sort=True):
            event_count += len(event_spans(symbol_frame["label"].to_numpy(dtype=int)))
        summary[split_name] = {
            "rows": int(len(frame)),
            "positive_rows": int(labels.sum()),
            "event_count": int(event_count),
            "timestamp_start": str(frame["timestamp"].iloc[0]) if not frame.empty else None,
            "timestamp_end": str(frame["timestamp"].iloc[-1]) if not frame.empty else None,
        }
    return summary
