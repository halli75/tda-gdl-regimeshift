from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .config import ResearchConfig
from .data_pipeline import load_market_data, write_dataset_manifest
from .evaluation import evaluate_predictions, short_event_metrics
from .feature_engineering import add_cross_asset_features, build_feature_frame, feature_groups, split_feature_frame, split_frame_summary
from .gdl_models import fit_gdl_model_suite, predict_graph_scores
from .graph_data import build_graph_dataset, split_graph_dataset
from .labels import build_shift_event_labels
from .models import fit_model_suite, predict_scores
from .walk_forward import generate_walk_forward_folds


def _output_paths(root: Path, outputs, evaluation_mode: str) -> dict[str, Path]:
    mode_root = root / evaluation_mode
    figure_dir = mode_root / outputs.figure_dir
    table_dir = mode_root / outputs.table_dir
    figure_dir.mkdir(parents=True, exist_ok=True)
    table_dir.mkdir(parents=True, exist_ok=True)
    return {
        "root": mode_root,
        "figure_dir": figure_dir,
        "table_dir": table_dir,
        "metrics": mode_root / outputs.metrics_filename,
        "predictions": mode_root / outputs.predictions_filename,
        "events": mode_root / outputs.event_table_filename,
        "models": table_dir / outputs.model_table_filename,
        "per_symbol": table_dir / outputs.per_symbol_table_filename,
        "short_events": table_dir / outputs.short_event_table_filename,
        "manifest": mode_root / outputs.manifest_filename,
        "summary": mode_root / outputs.summary_filename,
        "split_summary": mode_root / outputs.split_summary_filename,
        "walkforward_folds": mode_root / outputs.walkforward_fold_summary_filename,
        "dataset_manifest": root / "dataset_manifest.json",
    }


def _plot_pr_auc(model_table: pd.DataFrame, output_path: Path) -> None:
    order = model_table.sort_values("pr_auc", ascending=False)
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(order["model"], order["pr_auc"], color="#2f6db0")
    ax.set_ylim(0.0, 1.0)
    ax.set_ylabel("PR-AUC")
    ax.set_title("Model Comparison")
    ax.tick_params(axis="x", rotation=20)
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def _plot_best_model_timeline(predictions: pd.DataFrame, output_path: Path) -> None:
    top = predictions.sort_values(["symbol", "sample_row_id"]).groupby("symbol", sort=True).head(150)
    if top.empty:
        return
    fig, axes = plt.subplots(len(top["symbol"].unique()), 1, figsize=(10, 3 * len(top["symbol"].unique())), sharex=False)
    if not isinstance(axes, np.ndarray):
        axes = np.asarray([axes])
    for axis, (symbol, symbol_frame) in zip(axes, top.groupby("symbol", sort=True)):
        axis.plot(symbol_frame["sample_row_id"], symbol_frame["best_model_score"], label="score", color="#2f6db0")
        axis.plot(symbol_frame["sample_row_id"], symbol_frame["label"], label="label", color="#d24d57", alpha=0.8)
        axis.set_title(f"{symbol}: score vs. label")
        axis.legend(loc="upper right")
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def _per_symbol_metrics(
    predictions: pd.DataFrame,
    model_table: pd.DataFrame,
    cfg: ResearchConfig,
) -> pd.DataFrame:
    rows: list[dict[str, float | str]] = []
    threshold_by_model = {str(row["model"]): float(row["threshold"]) for _, row in model_table.iterrows()}
    model_names = [str(model) for model in model_table["model"].tolist()]
    for model_name in model_names:
        threshold = threshold_by_model[model_name]
        for symbol, symbol_frame in predictions.groupby("symbol", sort=True):
            summary, _ = evaluate_predictions(
                symbol_frame,
                scores=symbol_frame[f"{model_name}_score"].to_numpy(dtype=float),
                threshold=threshold,
                early_warning_bars=cfg.evaluation.early_warning_bars,
                bars_per_day=cfg.evaluation.bars_per_day,
                bootstrap_samples=0,
                bootstrap_block_size=cfg.evaluation.bootstrap_block_size,
            )
            summary["symbol"] = symbol
            summary["model"] = model_name
            rows.append(summary)
    return pd.DataFrame(rows)


def _sort_model_table(model_table: pd.DataFrame, cfg: ResearchConfig) -> pd.DataFrame:
    ranked = model_table.copy()
    max_false_alarms = cfg.evaluation.max_false_alarms_per_day
    if max_false_alarms is None:
        ranked["_passes_gate"] = 1
    else:
        ranked["_passes_gate"] = (ranked["false_alarms_per_day"] < max_false_alarms).astype(int)
    selection_metric = cfg.evaluation.selection_metric
    ranked = ranked.sort_values(
        by=["_passes_gate", selection_metric, "event_recall", "pr_auc", "mean_lead_bars", "false_alarms_per_day"],
        ascending=[False, False, False, False, False, True],
    ).reset_index(drop=True)
    return ranked.drop(columns="_passes_gate")


def _write_run_summary(
    output_path: Path,
    best_model: str,
    model_table: pd.DataFrame,
    feature_frame: pd.DataFrame,
    split_label: str,
    fold_results: list[dict] | None = None,
) -> None:
    best_row = model_table.loc[model_table["model"] == best_model].iloc[0]
    lines = [
        "# Run Summary",
        "",
        f"- Best model: `{best_model}`",
        f"- Evaluation split: `{split_label}`",
        f"- PR-AUC: `{best_row['pr_auc']:.4f}`",
        f"- Event F1: `{best_row['event_f1']:.4f}`",
        f"- Mean lead bars: `{best_row['mean_lead_bars']}`",
        f"- False alarms/day: `{best_row['false_alarms_per_day']}`",
        f"- Feature samples: `{len(feature_frame)}`",
        f"- Positive labels: `{int(feature_frame['label'].sum())}`",
    ]
    if fold_results:
        model_names = list(fold_results[0]["models"].keys()) if fold_results else []
        lines += ["", f"## Walk-Forward Summary ({len(fold_results)} folds)", ""]
        header_models = " | ".join(f"{m} F1" for m in model_names[:3])
        lines.append(f"| Fold | Val Period | Events | {header_models} |")
        lines.append(f"|------|-----------|--------|{'|'.join(['------'] * min(3, len(model_names)))}|")
        for fr in fold_results:
            val_start = str(fr["val_start"])[:10]
            val_end = str(fr["val_end"])[:10]
            model_f1s = " | ".join(
                f"{fr['models'].get(m, {}).get('event_f1', float('nan')):.3f}"
                for m in model_names[:3]
            )
            lines.append(f"| {fr['fold_index']} | {val_start}–{val_end} | {fr['val_event_count']} | {model_f1s} |")
        lines.append("")
        for m in model_names:
            fold_f1s = [fr["models"].get(m, {}).get("event_f1", float("nan")) for fr in fold_results]
            valid = [v for v in fold_f1s if not (v != v)]  # filter NaN
            if valid:
                mean_f1 = sum(valid) / len(valid)
                std_f1 = (sum((v - mean_f1) ** 2 for v in valid) / len(valid)) ** 0.5
                lines.append(f"- Mean {m} event F1: `{mean_f1:.4f} ± {std_f1:.4f}`")
    output_path.write_text("\n".join(lines), encoding="utf-8")


def _resolved_gap_settings(cfg: ResearchConfig) -> tuple[int, int]:
    purge_bars = cfg.evaluation.purge_bars
    if purge_bars is None:
        purge_bars = max(
            cfg.features.window_bars,
            cfg.labels.lookahead_bars,
            cfg.labels.volatility_window_bars,
        )
    embargo_bars = cfg.evaluation.embargo_bars
    if embargo_bars is None:
        embargo_bars = cfg.labels.lookahead_bars
    return int(purge_bars), int(embargo_bars)


def _require_split_sufficiency(split_summary: dict[str, dict[str, object]], cfg: ResearchConfig) -> None:
    val_events = int(split_summary["val"]["event_count"])
    test_events = int(split_summary["test"]["event_count"])
    if val_events < cfg.evaluation.min_validation_events:
        raise ValueError(
            f"Validation split has {val_events} events, below min_validation_events={cfg.evaluation.min_validation_events}"
        )
    if test_events < cfg.evaluation.min_test_events:
        raise ValueError(f"Test split has {test_events} events, below min_test_events={cfg.evaluation.min_test_events}")


def _evaluation_split_name(evaluation_mode: str) -> tuple[str, str]:
    if evaluation_mode == "search":
        return "val", "validation"
    if evaluation_mode == "final":
        return "test", "final_test"
    raise ValueError(f"Unsupported evaluation mode: {evaluation_mode}")


def _run_walk_forward(
    cfg: ResearchConfig,
    paths: dict[str, Path],
    config_path: str,
    labeled_frame: pd.DataFrame,
    feature_frame: pd.DataFrame,
    graph_dataset,
    groups: dict[str, list[str]],
    purge_bars: int,
    embargo_bars: int,
) -> None:
    """Run walk-forward expanding-window evaluation and write all artifacts."""
    test_df, folds = generate_walk_forward_folds(feature_frame, cfg.evaluation, purge_bars, embargo_bars)

    if not folds:
        raise ValueError("Walk-forward produced 0 folds — check walk_forward_min_train_years and data length.")

    # Filter out folds that don't meet the minimum event threshold.
    from .labels import event_spans as _event_spans
    qualified_folds = []
    for fold_idx, fold in enumerate(folds):
        val_events = sum(
            len(_event_spans(sym_frame["label"].to_numpy(dtype=int)))
            for _, sym_frame in fold["val"].groupby("symbol", sort=True)
        )
        if val_events >= cfg.evaluation.min_validation_events:
            qualified_folds.append(fold)
    folds = qualified_folds

    if not folds:
        raise ValueError(
            "Walk-forward produced 0 qualifying folds after applying min_validation_events="
            f"{cfg.evaluation.min_validation_events}. Lower min_validation_events or walk_forward_min_train_years."
        )

    fold_results: list[dict] = []
    # Track per-model scores and metrics across folds for aggregation.
    fold_model_metrics: dict[str, list[dict]] = {}

    for fold_idx, fold in enumerate(folds):
        train_df = fold["train"]
        val_df = fold["val"]
        fold_splits = {"train": train_df, "val": val_df, "test": test_df}
        fold_graph_splits = split_graph_dataset(graph_dataset, fold_splits)
        fitted_classical = fit_model_suite(fold_splits, groups, cfg.models, cfg.evaluation)
        fitted_graph = fit_gdl_model_suite(fold_graph_splits, fold_splits, groups, cfg.models, cfg.evaluation)
        all_fitted = {**fitted_classical, **fitted_graph}

        fold_model_row: dict[str, dict] = {}
        for name, model in all_fitted.items():
            if name in fitted_classical:
                scores = predict_scores(model, val_df)
            else:
                scores = predict_graph_scores(model, fold_graph_splits["val"], val_df, cfg.models)
            summary, _ = evaluate_predictions(
                val_df,
                scores=scores,
                threshold=model.threshold,
                early_warning_bars=cfg.evaluation.early_warning_bars,
                bars_per_day=cfg.evaluation.bars_per_day,
                bootstrap_samples=0,
                bootstrap_block_size=max(len(val_df), 1),
            )
            fold_model_row[name] = {
                "event_f1": float(summary.get("event_f1", 0.0)),
                "pr_auc": float(summary.get("pr_auc", float("nan"))),
                "false_alarms_per_day": float(summary.get("false_alarms_per_day", float("inf"))),
                "event_recall": float(summary.get("event_recall", 0.0)),
                "mean_lead_bars": float(summary.get("mean_lead_bars", float("nan"))),
                "threshold": float(model.threshold),
            }
            if name not in fold_model_metrics:
                fold_model_metrics[name] = []
            fold_model_metrics[name].append(fold_model_row[name])

        # Count val events (positive label spans).
        val_event_count = sum(
            len(_event_spans(sym_frame["label"].to_numpy(dtype=int)))
            for _, sym_frame in val_df.groupby("symbol", sort=True)
        )
        fold_results.append({
            "fold_index": fold_idx,
            "train_start": str(train_df["timestamp"].min()),
            "train_end": str(train_df["timestamp"].max()),
            "val_start": str(val_df["timestamp"].min()),
            "val_end": str(val_df["timestamp"].max()),
            "val_event_count": val_event_count,
            "models": fold_model_row,
        })

    # Build aggregated model table (mean metrics across folds).
    def _safe_mean(values: list[float]) -> float:
        valid = [v for v in values if v == v and v != float("inf") and v != float("-inf")]
        return sum(valid) / len(valid) if valid else float("nan")

    model_rows: list[dict] = []
    for name, fold_metrics_list in fold_model_metrics.items():
        mean_row: dict = {
            "model": name,
            "validation_f1": _safe_mean([m["event_f1"] for m in fold_metrics_list]),
            "feature_count": 0,
            "evaluation_split": "walk_forward",
            "pr_auc": _safe_mean([m["pr_auc"] for m in fold_metrics_list]),
            "event_f1": _safe_mean([m["event_f1"] for m in fold_metrics_list]),
            "event_recall": _safe_mean([m["event_recall"] for m in fold_metrics_list]),
            "false_alarms_per_day": _safe_mean([m["false_alarms_per_day"] for m in fold_metrics_list]),
            "mean_lead_bars": _safe_mean([m["mean_lead_bars"] for m in fold_metrics_list]),
            "threshold": _safe_mean([m["threshold"] for m in fold_metrics_list]),
            "roc_auc": float("nan"),
            "sample_precision": float("nan"),
            "sample_recall": float("nan"),
            "sample_f1": float("nan"),
            "positive_rate": float("nan"),
            "event_precision": float("nan"),
            "true_event_count": float("nan"),
            "pred_event_count": float("nan"),
            "detected_event_rate": float("nan"),
            "bootstrap_ci": {},
        }
        model_rows.append(mean_row)

    model_table = _sort_model_table(pd.DataFrame(model_rows), cfg)
    best_model_name = str(model_table.iloc[0]["model"])

    # Compute per-fold summary stats.
    fold_summary: dict[str, dict] = {}
    for name, fold_metrics_list in fold_model_metrics.items():
        f1s = [m["event_f1"] for m in fold_metrics_list]
        pr_aucs = [m["pr_auc"] for m in fold_metrics_list]
        valid_f1 = [v for v in f1s if v == v]
        valid_pr = [v for v in pr_aucs if v == v]
        mean_f1 = _safe_mean(f1s)
        std_f1 = (sum((v - mean_f1) ** 2 for v in valid_f1) / len(valid_f1)) ** 0.5 if valid_f1 else float("nan")
        mean_pr = _safe_mean(pr_aucs)
        std_pr = (sum((v - mean_pr) ** 2 for v in valid_pr) / len(valid_pr)) ** 0.5 if valid_pr else float("nan")
        fold_summary[name] = {
            "mean_event_f1": mean_f1,
            "std_event_f1": std_f1,
            "mean_pr_auc": mean_pr,
            "std_pr_auc": std_pr,
            "mean_false_alarms_per_day": _safe_mean([m["false_alarms_per_day"] for m in fold_metrics_list]),
        }

    wf_fold_artifact = {
        "n_folds": len(fold_results),
        "fold_val_event_counts": [fr["val_event_count"] for fr in fold_results],
        "folds": fold_results,
        "summary": fold_summary,
    }

    # Build a minimal split_summary for autoresearch compatibility.
    test_event_count = sum(
        len(_event_spans(sym_frame["label"].to_numpy(dtype=int)))
        for _, sym_frame in test_df.groupby("symbol", sort=True)
    )
    split_summary = {
        "walk_forward": {
            "n_folds": len(fold_results),
            "fold_val_event_counts": [fr["val_event_count"] for fr in fold_results],
            "event_count": sum(fr["val_event_count"] for fr in fold_results),
        },
        "test": {
            "rows": len(test_df),
            "event_count": test_event_count,
            "timestamp_start": str(test_df["timestamp"].min()) if not test_df.empty else "",
            "timestamp_end": str(test_df["timestamp"].max()) if not test_df.empty else "",
        },
    }

    # Assemble best_summary for metrics.json (uses mean metrics).
    best_row = model_table.iloc[0]
    best_summary: dict = {
        "best_model": best_model_name,
        "evaluation_mode": "walk_forward",
        "evaluation_split": "walk_forward",
        "selection_metric_source": "walk_forward",
        "selection_metric": cfg.evaluation.selection_metric,
        "max_false_alarms_per_day": cfg.evaluation.max_false_alarms_per_day,
        "purge_bars": purge_bars,
        "embargo_bars": embargo_bars,
        "pr_auc": float(best_row["pr_auc"]),
        "event_f1": float(best_row["event_f1"]),
        "event_recall": float(best_row["event_recall"]),
        "false_alarms_per_day": float(best_row["false_alarms_per_day"]),
        "mean_lead_bars": float(best_row["mean_lead_bars"]),
        "threshold": float(best_row["threshold"]),
        "split_summary": split_summary,
        "models": model_table.to_dict(orient="records"),
        "n_folds": len(fold_results),
        "fold_summary": fold_summary,
    }

    # Minimal placeholder artifacts (required by autoresearch contract).
    empty_predictions = test_df[["symbol", "timestamp", "sample_row_id", "label", "event_id"]].copy()
    empty_predictions["best_model"] = best_model_name
    empty_predictions["best_model_score"] = float("nan")
    empty_predictions["best_model_threshold"] = float(best_row["threshold"])
    empty_predictions["best_model_pred"] = 0

    model_table.to_csv(paths["models"], index=False)
    empty_predictions.to_csv(paths["predictions"], index=False)
    pd.DataFrame().to_csv(paths["events"], index=False)
    pd.DataFrame().to_csv(paths["per_symbol"], index=False)
    pd.DataFrame().to_csv(paths["short_events"], index=False)
    with open(paths["split_summary"], "w", encoding="utf-8") as fh:
        json.dump(split_summary, fh, indent=2)
    with open(paths["metrics"], "w", encoding="utf-8") as fh:
        json.dump(best_summary, fh, indent=2, default=str)
    with open(paths["walkforward_folds"], "w", encoding="utf-8") as fh:
        json.dump(wf_fold_artifact, fh, indent=2, default=str)
    with open(paths["manifest"], "w", encoding="utf-8") as fh:
        json.dump(
            {
                "config_path": str(Path(config_path).resolve()),
                "evaluation_mode": "walk_forward",
                "evaluation_split": "walk_forward",
                "selection_metric_source": "walk_forward",
                "selection_metric": cfg.evaluation.selection_metric,
                "max_false_alarms_per_day": cfg.evaluation.max_false_alarms_per_day,
                "purge_bars": purge_bars,
                "embargo_bars": embargo_bars,
                "best_model": best_model_name,
                "best_model_threshold": float(best_row["threshold"]),
                "n_folds": len(fold_results),
                "model_table_path": str(paths["models"]),
                "predictions_path": str(paths["predictions"]),
                "split_summary_path": str(paths["split_summary"]),
                "walkforward_folds_path": str(paths["walkforward_folds"]),
            },
            fh,
            indent=2,
        )
    _plot_pr_auc(model_table, paths["figure_dir"] / "model_comparison.png")
    _write_run_summary(paths["summary"], best_model_name, model_table, feature_frame, "walk_forward", fold_results)
    print(f"Best model: {best_model_name}")
    print(f"Metrics written to: {paths['metrics']}")


def main(config_path: str, evaluation_mode: str) -> None:
    cfg = ResearchConfig.from_yaml(config_path)
    project_root = Path.cwd().resolve()
    output_root = Path(cfg.outputs.root_dir)
    if not output_root.is_absolute():
        output_root = project_root / output_root
    output_root.mkdir(parents=True, exist_ok=True)
    paths = _output_paths(output_root, cfg.outputs, evaluation_mode)

    market_frame = load_market_data(cfg.data, base_dir=project_root)
    write_dataset_manifest(market_frame, paths["dataset_manifest"], cfg.data)
    purge_bars, embargo_bars = _resolved_gap_settings(cfg)
    labeled_frame = build_shift_event_labels(market_frame, cfg.labels)
    feature_frame = build_feature_frame(labeled_frame, cfg.features)
    feature_frame = add_cross_asset_features(feature_frame, labeled_frame)
    groups = feature_groups(feature_frame)
    splits = split_feature_frame(
        feature_frame,
        train_frac=cfg.evaluation.train_frac,
        val_frac=cfg.evaluation.val_frac,
        test_frac=cfg.evaluation.test_frac,
        purge_bars=purge_bars,
        embargo_bars=embargo_bars,
    )
    split_summary = split_frame_summary(splits)
    _require_split_sufficiency(split_summary, cfg)
    graph_dataset = build_graph_dataset(labeled_frame, cfg.features)

    if evaluation_mode == "walk_forward":
        _run_walk_forward(cfg, paths, config_path, labeled_frame, feature_frame, graph_dataset, groups, purge_bars, embargo_bars)
        return

    graph_splits = split_graph_dataset(graph_dataset, splits)
    fitted_models = fit_model_suite(splits, groups, cfg.models, cfg.evaluation)
    fitted_graph_models = fit_gdl_model_suite(graph_splits, splits, groups, cfg.models, cfg.evaluation)
    all_fitted_models = {**fitted_models, **fitted_graph_models}
    evaluation_split, split_label = _evaluation_split_name(evaluation_mode)

    model_rows: list[dict[str, object]] = []
    all_scores: dict[str, np.ndarray] = {}
    for name, fitted in fitted_models.items():
        scores = predict_scores(fitted, splits[evaluation_split])
        all_scores[name] = scores
        summary, _ = evaluate_predictions(
            splits[evaluation_split],
            scores=scores,
            threshold=fitted.threshold,
            early_warning_bars=cfg.evaluation.early_warning_bars,
            bars_per_day=cfg.evaluation.bars_per_day,
            bootstrap_samples=cfg.evaluation.bootstrap_samples,
            bootstrap_block_size=cfg.evaluation.bootstrap_block_size,
        )
        row = {
            "model": name,
            "validation_f1": fitted.validation_f1,
            "feature_count": len(fitted.feature_columns),
            "evaluation_split": split_label,
        }
        row.update(summary)
        model_rows.append(row)
    for name, fitted in fitted_graph_models.items():
        scores = predict_graph_scores(fitted, graph_splits[evaluation_split], splits[evaluation_split], cfg.models)
        all_scores[name] = scores
        summary, _ = evaluate_predictions(
            splits[evaluation_split],
            scores=scores,
            threshold=fitted.threshold,
            early_warning_bars=cfg.evaluation.early_warning_bars,
            bars_per_day=cfg.evaluation.bars_per_day,
            bootstrap_samples=cfg.evaluation.bootstrap_samples,
            bootstrap_block_size=cfg.evaluation.bootstrap_block_size,
        )
        row = {
            "model": name,
            "validation_f1": fitted.validation_f1,
            "feature_count": len(fitted.feature_columns),
            "evaluation_split": split_label,
        }
        row.update(summary)
        model_rows.append(row)
    model_table = _sort_model_table(pd.DataFrame(model_rows), cfg)
    best_model_name = str(model_table.iloc[0]["model"])
    best_model = all_fitted_models[best_model_name]
    test_predictions = splits[evaluation_split][["symbol", "timestamp", "sample_row_id", "label", "event_id"]].copy()
    for model_name, scores in all_scores.items():
        test_predictions[f"{model_name}_score"] = scores
    test_predictions["best_model"] = best_model_name
    test_predictions["best_model_score"] = all_scores[best_model_name]
    test_predictions["best_model_threshold"] = best_model.threshold
    test_predictions["best_model_pred"] = (
        test_predictions["best_model_score"] >= best_model.threshold
    ).astype(int)

    best_summary, event_table = evaluate_predictions(
        splits[evaluation_split],
        scores=all_scores[best_model_name],
        threshold=best_model.threshold,
        early_warning_bars=cfg.evaluation.early_warning_bars,
        bars_per_day=cfg.evaluation.bars_per_day,
        bootstrap_samples=cfg.evaluation.bootstrap_samples,
        bootstrap_block_size=cfg.evaluation.bootstrap_block_size,
    )
    short_summary, short_event_table = short_event_metrics(event_table)
    best_summary["best_model"] = best_model_name
    best_summary["evaluation_mode"] = evaluation_mode
    best_summary["evaluation_split"] = split_label
    best_summary["selection_metric_source"] = split_label
    best_summary["selection_metric"] = cfg.evaluation.selection_metric
    best_summary["max_false_alarms_per_day"] = cfg.evaluation.max_false_alarms_per_day
    best_summary["purge_bars"] = purge_bars
    best_summary["embargo_bars"] = embargo_bars
    best_summary["split_summary"] = split_summary
    best_summary.update(short_summary)
    best_summary["models"] = model_table.to_dict(orient="records")

    model_table.to_csv(paths["models"], index=False)
    test_predictions.to_csv(paths["predictions"], index=False)
    event_table.to_csv(paths["events"], index=False)
    per_symbol = _per_symbol_metrics(test_predictions, model_table, cfg)
    vxx_row = per_symbol.loc[
        (per_symbol["model"] == best_model_name) & (per_symbol["symbol"].astype(str) == "^VIX")
    ]
    if not vxx_row.empty:
        vxx_metrics = vxx_row.iloc[0].to_dict()
        best_summary["vix_pr_auc"] = float(vxx_metrics["pr_auc"])
        best_summary["vix_event_recall"] = float(vxx_metrics["event_recall"])
        best_summary["vix_event_f1"] = float(vxx_metrics["event_f1"])
        best_summary["vix_detected_event_rate"] = float(vxx_metrics["detected_event_rate"])
    per_symbol.to_csv(paths["per_symbol"], index=False)
    short_event_table.to_csv(paths["short_events"], index=False)
    with open(paths["split_summary"], "w", encoding="utf-8") as handle:
        json.dump(split_summary, handle, indent=2)
    with open(paths["metrics"], "w", encoding="utf-8") as handle:
        json.dump(best_summary, handle, indent=2)
    with open(paths["manifest"], "w", encoding="utf-8") as handle:
        json.dump(
            {
                "config_path": str(Path(config_path).resolve()),
                "evaluation_mode": evaluation_mode,
                "evaluation_split": split_label,
                "selection_metric_source": split_label,
                "selection_metric": cfg.evaluation.selection_metric,
                "max_false_alarms_per_day": cfg.evaluation.max_false_alarms_per_day,
                "purge_bars": purge_bars,
                "embargo_bars": embargo_bars,
                "best_model": best_model_name,
                "best_model_threshold": best_model.threshold,
                "enable_vxx_tailored": cfg.features.enable_vxx_tailored,
                "enable_symbol_offset_calibration": cfg.models.enable_symbol_offset_calibration,
                "model_table_path": str(paths["models"]),
                "predictions_path": str(paths["predictions"]),
                "event_table_path": str(paths["events"]),
                "per_symbol_table_path": str(paths["per_symbol"]),
                "short_event_table_path": str(paths["short_events"]),
                "split_summary_path": str(paths["split_summary"]),
            },
            handle,
            indent=2,
        )
    _plot_pr_auc(model_table, paths["figure_dir"] / "model_comparison.png")
    _plot_best_model_timeline(test_predictions, paths["figure_dir"] / "best_model_timeline.png")
    _write_run_summary(paths["summary"], best_model_name, model_table, feature_frame, split_label)
    print(f"Best model: {best_model_name}")
    print(f"Metrics written to: {paths['metrics']}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the TDA regime-shift detection pipeline")
    parser.add_argument("--config", type=str, default="configs/default.yaml", help="Path to the YAML config")
    parser.add_argument(
        "--evaluation-mode",
        type=str,
        default="search",
        choices=["search", "final", "walk_forward"],
        help="Whether to evaluate models on validation or final held-out test",
    )
    arguments = parser.parse_args()
    main(arguments.config, arguments.evaluation_mode)
