from __future__ import annotations

import math

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, f1_score, precision_score, recall_score, roc_auc_score

from .labels import event_spans


def binary_metrics(y_true: np.ndarray, scores: np.ndarray, preds: np.ndarray) -> dict[str, float]:
    metrics = {
        "sample_precision": float(precision_score(y_true, preds, zero_division=0)),
        "sample_recall": float(recall_score(y_true, preds, zero_division=0)),
        "sample_f1": float(f1_score(y_true, preds, zero_division=0)),
        "positive_rate": float(preds.mean()) if len(preds) else 0.0,
    }
    if len(np.unique(y_true)) > 1:
        metrics["pr_auc"] = float(average_precision_score(y_true, scores))
        metrics["roc_auc"] = float(roc_auc_score(y_true, scores))
    else:
        metrics["pr_auc"] = math.nan
        metrics["roc_auc"] = math.nan
    return metrics


def event_level_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    true_events = event_spans(y_true)
    pred_events = event_spans(y_pred)
    matched_true = 0
    for start, end in true_events:
        if any(not (pred_end < start or pred_start > end) for pred_start, pred_end in pred_events):
            matched_true += 1
    matched_pred = 0
    for start, end in pred_events:
        if any(not (true_end < start or true_start > end) for true_start, true_end in true_events):
            matched_pred += 1
    precision = matched_pred / len(pred_events) if pred_events else 0.0
    recall = matched_true / len(true_events) if true_events else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0
    return {
        "event_precision": float(precision),
        "event_recall": float(recall),
        "event_f1": float(f1),
        "true_event_count": float(len(true_events)),
        "pred_event_count": float(len(pred_events)),
    }


def lead_time_analysis(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    early_warning_bars: int,
) -> tuple[dict[str, float], pd.DataFrame]:
    true_events = event_spans(y_true)
    pred_indices = np.flatnonzero(y_pred)
    event_rows: list[dict[str, float | int | bool]] = []
    leads: list[int] = []
    for event_id, (start, end) in enumerate(true_events):
        lower = max(0, start - early_warning_bars)
        candidates = pred_indices[(pred_indices >= lower) & (pred_indices <= end)]
        detected = len(candidates) > 0
        first_pred = int(candidates[0]) if detected else -1
        lead = int(start - first_pred) if detected else math.nan
        if detected:
            leads.append(int(lead))
        event_rows.append(
            {
                "event_id": event_id,
                "start_index": start,
                "end_index": end,
                "duration_bars": int(end - start + 1),
                "is_short_event": bool((end - start + 1) <= 2),
                "detected": detected,
                "first_prediction_index": first_pred,
                "lead_bars": lead,
            }
        )
    summary = {
        "detected_event_rate": float(np.mean([row["detected"] for row in event_rows])) if event_rows else 0.0,
        "mean_lead_bars": float(np.mean(leads)) if leads else math.nan,
    }
    return summary, pd.DataFrame(event_rows)


def false_alarms_per_day(y_true: np.ndarray, y_pred: np.ndarray, bars_per_day: int) -> float:
    true_events = event_spans(y_true)
    pred_events = event_spans(y_pred)
    false_alarm_count = 0
    for start, end in pred_events:
        overlaps = any(not (true_end < start or true_start > end) for true_start, true_end in true_events)
        if not overlaps:
            false_alarm_count += 1
    days = max(len(y_true) / max(bars_per_day, 1), 1e-9)
    return float(false_alarm_count / days)


def bootstrap_ci(
    y_true: np.ndarray,
    scores: np.ndarray,
    preds: np.ndarray,
    n_samples: int,
    block_size: int,
) -> dict[str, tuple[float, float]]:
    if n_samples <= 0 or len(y_true) == 0:
        return {}
    rng = np.random.default_rng(42)
    pr_aucs: list[float] = []
    sample_f1s: list[float] = []
    n = len(y_true)
    block_size = max(1, min(block_size, n))
    for _ in range(n_samples):
        starts = rng.integers(0, n, size=int(np.ceil(n / block_size)))
        pieces = [np.arange(start, start + block_size) % n for start in starts]
        idx = np.concatenate(pieces)[:n]
        boot_true = y_true[idx]
        boot_scores = scores[idx]
        boot_preds = preds[idx]
        if len(np.unique(boot_true)) > 1:
            pr_aucs.append(float(average_precision_score(boot_true, boot_scores)))
        sample_f1s.append(float(f1_score(boot_true, boot_preds, zero_division=0)))
    results: dict[str, tuple[float, float]] = {}
    if pr_aucs:
        results["pr_auc"] = (float(np.quantile(pr_aucs, 0.025)), float(np.quantile(pr_aucs, 0.975)))
    if sample_f1s:
        results["sample_f1"] = (
            float(np.quantile(sample_f1s, 0.025)),
            float(np.quantile(sample_f1s, 0.975)),
        )
    return results


def evaluate_predictions(
    frame: pd.DataFrame,
    scores: np.ndarray,
    threshold: float,
    early_warning_bars: int,
    bars_per_day: int,
    bootstrap_samples: int,
    bootstrap_block_size: int,
) -> tuple[dict[str, float | dict[str, tuple[float, float]]], pd.DataFrame]:
    y_true = frame["label"].to_numpy(dtype=int)
    y_pred = (scores >= threshold).astype(int)
    summary = binary_metrics(y_true, scores, y_pred)
    summary.update(event_level_metrics(y_true, y_pred))
    lead_summary, event_table = lead_time_analysis(y_true, y_pred, early_warning_bars=early_warning_bars)
    summary.update(lead_summary)
    summary["false_alarms_per_day"] = false_alarms_per_day(y_true, y_pred, bars_per_day=bars_per_day)
    summary["threshold"] = float(threshold)
    summary["bootstrap_ci"] = bootstrap_ci(
        y_true,
        scores,
        y_pred,
        n_samples=bootstrap_samples,
        block_size=bootstrap_block_size,
    )
    event_table = event_table.copy()
    event_table["threshold"] = float(threshold)
    return summary, event_table


def short_event_metrics(event_table: pd.DataFrame, max_short_span: int = 2) -> tuple[dict[str, float], pd.DataFrame]:
    if event_table.empty:
        return {
            "short_event_count": 0.0,
            "short_event_detected_count": 0.0,
            "short_event_detection_rate": 0.0,
            "short_event_mean_lead_bars": math.nan,
        }, event_table.copy()
    short_table = event_table.loc[event_table["duration_bars"] <= max_short_span].copy().reset_index(drop=True)
    detected = short_table.loc[short_table["detected"]]
    metrics = {
        "short_event_count": float(len(short_table)),
        "short_event_detected_count": float(len(detected)),
        "short_event_detection_rate": float(detected.shape[0] / len(short_table)) if len(short_table) else 0.0,
        "short_event_mean_lead_bars": float(detected["lead_bars"].mean()) if not detected.empty else math.nan,
    }
    return metrics, short_table


def metrics_pass_gate(summary: dict[str, float], max_false_alarms_per_day: float | None) -> bool:
    if max_false_alarms_per_day is None:
        return True
    return float(summary["false_alarms_per_day"]) < float(max_false_alarms_per_day)


def is_better_summary(
    candidate: dict[str, float],
    incumbent: dict[str, float] | None,
    selection_metric: str,
    max_false_alarms_per_day: float | None,
) -> bool:
    candidate_pass = metrics_pass_gate(candidate, max_false_alarms_per_day)
    incumbent_pass = incumbent is not None and metrics_pass_gate(incumbent, max_false_alarms_per_day)
    if not candidate_pass:
        return False
    if incumbent is None or not incumbent_pass:
        return True

    selection_tolerance = 1e-6
    candidate_value = float(candidate.get(selection_metric, math.nan))
    incumbent_value = float(incumbent.get(selection_metric, math.nan))
    if candidate_value > incumbent_value + selection_tolerance:
        return True
    if candidate_value < incumbent_value - selection_tolerance:
        return False

    if float(candidate.get("event_recall", 0.0)) > float(incumbent.get("event_recall", 0.0)) + selection_tolerance:
        return True
    if float(candidate.get("event_recall", 0.0)) < float(incumbent.get("event_recall", 0.0)) - selection_tolerance:
        return False

    if (
        float(candidate.get("false_alarms_per_day", math.inf))
        < float(incumbent.get("false_alarms_per_day", math.inf)) - selection_tolerance
    ):
        return True
    if (
        float(candidate.get("false_alarms_per_day", math.inf))
        > float(incumbent.get("false_alarms_per_day", math.inf)) + selection_tolerance
    ):
        return False

    if float(candidate.get("pr_auc", -math.inf)) > float(incumbent.get("pr_auc", -math.inf)) + selection_tolerance:
        return True
    if float(candidate.get("pr_auc", -math.inf)) < float(incumbent.get("pr_auc", -math.inf)) - selection_tolerance:
        return False

    return float(candidate.get("mean_lead_bars", -math.inf)) > float(incumbent.get("mean_lead_bars", -math.inf))


def select_best_threshold(
    frame: pd.DataFrame,
    scores: np.ndarray,
    threshold_grid: list[float],
    selection_metric: str,
    early_warning_bars: int,
    bars_per_day: int,
    max_false_alarms_per_day: float | None,
    max_positive_rate: float | None = None,
) -> tuple[float, dict[str, float]]:
    best_threshold = float(threshold_grid[0]) if threshold_grid else 0.5
    best_summary: dict[str, float] | None = None
    for threshold in threshold_grid:
        summary, _ = evaluate_predictions(
            frame,
            scores=scores,
            threshold=float(threshold),
            early_warning_bars=early_warning_bars,
            bars_per_day=bars_per_day,
            bootstrap_samples=0,
            bootstrap_block_size=max(len(frame), 1),
        )
        if max_positive_rate is not None and float(summary.get("positive_rate", 0.0)) > max_positive_rate:
            continue
        if is_better_summary(summary, best_summary, selection_metric, max_false_alarms_per_day):
            best_summary = summary
            best_threshold = float(threshold)
    if best_summary is None:
        fallback_threshold = float(threshold_grid[0]) if threshold_grid else 0.5
        best_summary, _ = evaluate_predictions(
            frame,
            scores=scores,
            threshold=fallback_threshold,
            early_warning_bars=early_warning_bars,
            bars_per_day=bars_per_day,
            bootstrap_samples=0,
            bootstrap_block_size=max(len(frame), 1),
        )
        best_threshold = fallback_threshold
    return best_threshold, best_summary
