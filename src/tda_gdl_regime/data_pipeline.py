from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import yfinance as yf

from .config import DataConfig


def _parse_timestamp(series: pd.Series) -> pd.Series:
    if pd.api.types.is_numeric_dtype(series):
        return series
    parsed = pd.to_datetime(series, errors="coerce")
    return parsed if parsed.notna().all() else series


def _load_single_symbol(file_path: Path, symbol: str, cfg: DataConfig) -> pd.DataFrame:
    frame = pd.read_csv(file_path)
    if cfg.timestamp_col not in frame.columns or cfg.price_col not in frame.columns:
        raise ValueError(
            f"{file_path} must contain '{cfg.timestamp_col}' and '{cfg.price_col}' columns"
        )
    frame = frame[[cfg.timestamp_col, cfg.price_col]].copy()
    frame.columns = ["timestamp", "price"]
    frame["timestamp"] = _parse_timestamp(frame["timestamp"])
    frame["price"] = frame["price"].astype(float)
    frame["symbol"] = symbol
    frame = frame.dropna(subset=["timestamp", "price"]).drop_duplicates(subset=["timestamp"])
    frame = frame.sort_values("timestamp").reset_index(drop=True)
    if cfg.regular_hours_only and pd.api.types.is_datetime64_any_dtype(frame["timestamp"]):
        frame = (
            frame.set_index("timestamp")
            .between_time("09:30", "16:00", inclusive="both")
            .reset_index()
        )
    if cfg.returns_mode == "log":
        frame["return"] = np.log(frame["price"]).diff()
    else:
        frame["return"] = frame["price"].pct_change()
    frame["return"] = frame["return"].fillna(0.0)
    frame["row_id"] = np.arange(len(frame))
    return frame


def _yfinance_cache_path(symbol: str, cfg: DataConfig, base_dir: Path) -> Path:
    cache_dir = base_dir / cfg.cache_dir
    cache_dir.mkdir(parents=True, exist_ok=True)
    if cfg.start or cfg.end:
        suffix = f"{cfg.start or 'start'}_{cfg.end or 'end'}"
    else:
        suffix = cfg.period
    safe_suffix = suffix.replace(":", "-").replace("/", "-")
    return cache_dir / f"{symbol}_{cfg.interval}_{safe_suffix}.csv"


def _download_yfinance_history(symbol: str, cfg: DataConfig) -> pd.DataFrame:
    if cfg.start and cfg.end and cfg.chunk_days and cfg.chunk_days > 0:
        start_ts = pd.Timestamp(cfg.start)
        end_ts = pd.Timestamp(cfg.end)
        pieces: list[pd.DataFrame] = []
        cursor = start_ts
        while cursor < end_ts:
            chunk_end = min(cursor + pd.Timedelta(days=cfg.chunk_days), end_ts)
            piece = yf.download(
                tickers=symbol,
                start=cursor.strftime("%Y-%m-%d"),
                end=chunk_end.strftime("%Y-%m-%d"),
                interval=cfg.interval,
                auto_adjust=cfg.auto_adjust,
                prepost=cfg.prepost,
                progress=False,
                group_by="column",
                threads=False,
            )
            if not piece.empty:
                pieces.append(piece)
            cursor = chunk_end
        if not pieces:
            return pd.DataFrame()
        history = pd.concat(pieces).sort_index()
        return history.loc[~history.index.duplicated(keep="last")]
    return yf.download(
        tickers=symbol,
        start=cfg.start,
        end=cfg.end,
        period=None if (cfg.start or cfg.end) else cfg.period,
        interval=cfg.interval,
        auto_adjust=cfg.auto_adjust,
        prepost=cfg.prepost,
        progress=False,
        group_by="column",
        threads=False,
    )


def _load_yfinance_symbol(symbol: str, cfg: DataConfig, base_dir: Path) -> pd.DataFrame:
    cache_path = _yfinance_cache_path(symbol, cfg, base_dir)
    tz_cache_dir = cache_path.parent / "yf_tz_cache"
    tz_cache_dir.mkdir(parents=True, exist_ok=True)
    yf.set_tz_cache_location(str(tz_cache_dir))
    if cache_path.exists() and not cfg.force_refresh:
        return _load_single_symbol(cache_path, symbol, cfg)
    history = _download_yfinance_history(symbol, cfg)
    if history.empty:
        raise ValueError(f"yfinance returned no data for {symbol}")
    if isinstance(history.columns, pd.MultiIndex):
        history.columns = history.columns.get_level_values(0)
    price_column = "Close" if "Close" in history.columns else history.columns[-1]
    frame = history.reset_index()[[history.index.name or "Date", price_column]].copy()
    frame.columns = [cfg.timestamp_col, cfg.price_col]
    frame.to_csv(cache_path, index=False)
    return _load_single_symbol(cache_path, symbol, cfg)


def load_market_data(cfg: DataConfig, base_dir: str | Path | None = None) -> pd.DataFrame:
    base = Path(base_dir or ".")
    frames = []
    if cfg.provider == "local_csv":
        for file_cfg in cfg.files:
            frame = _load_single_symbol(base / file_cfg.path, file_cfg.symbol, cfg)
            frames.append(frame)
    elif cfg.provider == "yfinance":
        for symbol in cfg.symbols:
            frames.append(_load_yfinance_symbol(symbol, cfg, base))
    else:
        raise NotImplementedError(f"Unsupported data provider: {cfg.provider}")
    if not frames:
        raise ValueError("No data files were configured")
    return pd.concat(frames, ignore_index=True)


def write_dataset_manifest(frame: pd.DataFrame, output_path: str | Path, cfg: DataConfig) -> None:
    summary = {
        "provider": cfg.provider,
        "returns_mode": cfg.returns_mode,
        "interval": cfg.interval,
        "period": cfg.period,
        "symbols": {},
    }
    for symbol, symbol_frame in frame.groupby("symbol", sort=True):
        summary["symbols"][symbol] = {
            "rows": int(len(symbol_frame)),
            "timestamp_start": str(symbol_frame["timestamp"].iloc[0]),
            "timestamp_end": str(symbol_frame["timestamp"].iloc[-1]),
            "price_min": float(symbol_frame["price"].min()),
            "price_max": float(symbol_frame["price"].max()),
        }
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)
