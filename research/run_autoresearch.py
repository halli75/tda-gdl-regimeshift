from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd
import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_TARGET_MODEL = "gcn_fusion"
STATE_DIR = PROJECT_ROOT / "research" / "autoresearch"
RUNS_DIR = STATE_DIR / "runs"
FINAL_RUNS_DIR = STATE_DIR / "final_runs"
STATE_PATH = STATE_DIR / "validation_state.json"
HISTORY_PATH = STATE_DIR / "validation_history.jsonl"
FINALIZATION_PATH = STATE_DIR / "finalization_manifest.json"
SUMMARY_PATH = STATE_DIR / "latest_summary.md"


@dataclass
class TargetMetrics:
    pr_auc: float
    event_f1: float
    false_alarms_per_day: float
    mean_lead_bars: float
    threshold: float
    event_recall: float = 0.0
    vix_pr_auc: float = float("nan")
    vix_event_recall: float = 0.0
    vix_event_f1: float = 0.0

    @classmethod
    def from_sources(cls, row: pd.Series, per_symbol: pd.DataFrame | None) -> "TargetMetrics":
        vxx_row = None
        if per_symbol is not None and not per_symbol.empty:
            matched = per_symbol.loc[
                (per_symbol["model"].astype(str) == str(row["model"]))
                & (per_symbol["symbol"].astype(str) == "^VIX")
            ]
            if not matched.empty:
                vxx_row = matched.iloc[0]
        return cls(
            pr_auc=float(row["pr_auc"]),
            event_f1=float(row["event_f1"]),
            false_alarms_per_day=float(row["false_alarms_per_day"]),
            mean_lead_bars=float(row["mean_lead_bars"]),
            threshold=float(row["threshold"]),
            event_recall=float(row.get("event_recall", 0.0)),
            vix_pr_auc=float(vxx_row["pr_auc"]) if vxx_row is not None else float("nan"),
            vix_event_recall=float(vxx_row["event_recall"]) if vxx_row is not None else 0.0,
            vix_event_f1=float(vxx_row["event_f1"]) if vxx_row is not None else 0.0,
        )


def _load_yaml(path: Path) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


def _write_yaml(path: Path, payload: dict[str, Any]) -> None:
    with open(path, "w", encoding="utf-8") as handle:
        yaml.safe_dump(payload, handle, sort_keys=False)


def _get_nested(payload: dict[str, Any], dotted_key: str) -> Any:
    current = payload
    parts = dotted_key.split(".")
    for part in parts[:-1]:
        current = current[part]
    return current[parts[-1]]


def _set_nested(payload: dict[str, Any], dotted_key: str, value: Any) -> None:
    current = payload
    parts = dotted_key.split(".")
    for part in parts[:-1]:
        current = current[part]
    current[parts[-1]] = value


def _mutation_stages() -> list[list[dict[str, Any]]]:
    return [
        [
            {
                "key": "models.gdl_focal_gamma",
                "values": [1.0, 2.0],
                "hypothesis": "Focal loss down-weights easy negatives, spreading the score distribution and improving precision at low FP rates.",
                "stage": "focal_loss",
            }
        ],
        [
            {
                "key": "labels.lookahead_bars",
                "values": [3, 5, 10],
                "hypothesis": "A shorter or longer forward window may improve early detection at daily resolution.",
                "stage": "lookahead",
            }
        ],
        [
            {
                "key": "labels.volatility_window_bars",
                "values": [10, 20, 30],
                "hypothesis": "A 10-day (fortnight) or 30-day (quarter-month) backward vol window may sharpen regime contrast.",
                "stage": "volatility_window",
            }
        ],
        [
            {
                "key": "labels.threshold_quantile",
                "values": [0.85, 0.90, 0.95],
                "hypothesis": "Adjusting event density vs. severity trade-off may improve the training signal.",
                "stage": "threshold_quantile",
            }
        ],
        [
            {
                "key": "features.window_bars",
                "values": [40, 60, 90],
                "hypothesis": "A monthly (40), quarterly (60), or semi-annual (90) topology window may capture different regime scales.",
                "stage": "window_bars",
            }
        ],
        [
            {
                "key": "features.graph_knn_k",
                "values": [6, 8, 12],
                "hypothesis": "Graph sparsity affects which local geometric patterns are captured in the delay embedding.",
                "stage": "graph_knn",
            }
        ],
        [
            {
                "key": "models.gdl_dropout",
                "values": [0.05, 0.1, 0.2, 0.3, 0.4],
                "hypothesis": "With a larger daily dataset, more regularization may prevent overfitting.",
                "stage": "gdl_dropout",
            }
        ],
        [
            {
                "key": "models.gdl_learning_rate",
                "values": [0.001, 0.003, 0.005, 0.0005],
                "hypothesis": "Higher learning rates spread logit magnitudes faster, escaping the collapsed near-0.5 score distribution.",
                "stage": "gdl_lr",
            }
        ],
        [
            {
                "key": "models.gdl_weight_decay",
                "values": [0.0, 0.00001, 0.0001],
                "hypothesis": "Reducing L2 regularization allows weight magnitudes to grow, widening the score distribution.",
                "stage": "gdl_weight_decay",
            }
        ],
        [
            {
                "key": "models.gdl_hidden_dim",
                "values": [64, 128, 192, 256],
                "hypothesis": "A smaller hidden dim may generalize better; a larger one may capture more complex regime patterns.",
                "stage": "gdl_hidden_dim",
            }
        ],
        [
            {
                "key": "evaluation.purge_bars",
                "values": [10, 20, 30],
                "hypothesis": "Adjusting the anti-leakage gap may change how much training data is available near split boundaries.",
                "stage": "purge_bars",
            }
        ],
    ]


def _signature(delta: dict[str, Any]) -> str:
    return json.dumps(delta, sort_keys=True)


def _load_state(state_path: Path) -> dict[str, Any]:
    if state_path.exists():
        state = json.loads(state_path.read_text(encoding="utf-8"))
        if state.get("metric_source") == "validation":
            state.setdefault("pending_candidate", None)
            state.setdefault("selection_metric", "event_f1")
            state.setdefault("max_false_alarms_per_day", 0.10)
            return state
    return {
        "metric_source": "validation",
        "iteration": 0,
        "selection_metric": "event_f1",
        "max_false_alarms_per_day": 0.10,
        "best_validation_metrics": None,
        "best_run_dir": None,
        "best_config_path": None,
        "best_model_table_path": None,
        "best_per_symbol_table_path": None,
        "best_overall_model": None,
        "pending_candidate": None,
        "tried_mutations": [],
    }


def _write_state(state_path: Path, state: dict[str, Any]) -> None:
    state_path.parent.mkdir(parents=True, exist_ok=True)
    state_path.write_text(json.dumps(state, indent=2), encoding="utf-8")


def _ordered_candidate_values(current_value: Any, values: list[Any]) -> list[Any]:
    def sort_key(value: Any) -> tuple[int, float]:
        if value == current_value:
            return (1, 0.0)
        if isinstance(value, (int, float)) and isinstance(current_value, (int, float)):
            return (0, abs(float(value) - float(current_value)))
        return (0, 0.0)

    return sorted(values, key=sort_key)


def _choose_mutation(base_config: dict[str, Any], tried_signatures: set[str]) -> dict[str, Any]:
    for stage in _mutation_stages():
        for spec in stage:
            current_value = _get_nested(base_config, spec["key"])
            for value in _ordered_candidate_values(current_value, spec["values"]):
                if value == current_value:
                    continue
                signature = _signature({"key": spec["key"], "new_value": value})
                if signature in tried_signatures:
                    continue
                return {
                    "key": spec["key"],
                    "old_value": current_value,
                    "new_value": value,
                    "hypothesis": spec["hypothesis"],
                    "stage": spec["stage"],
                }
    raise RuntimeError("Mutation space exhausted; no unseen bounded experiment remains.")


def _run_pipeline(config_path: Path, evaluation_mode: str) -> None:
    # If the config specifies walk_forward evaluation mode, honour it unless
    # the caller is explicitly requesting the final held-out test ("final").
    cfg_data = _load_yaml(config_path)
    cfg_eval_mode = cfg_data.get("evaluation", {}).get("evaluation_mode", "search")
    actual_mode = cfg_eval_mode if cfg_eval_mode == "walk_forward" and evaluation_mode != "final" else evaluation_mode
    env = {**os.environ, "PYTHONPATH": str(PROJECT_ROOT / "src")}
    subprocess.run(
        [
            sys.executable,
            "-m",
            "tda_gdl_regime.run_pipeline",
            "--config",
            str(config_path),
            "--evaluation-mode",
            actual_mode,
        ],
        cwd=PROJECT_ROOT,
        env=env,
        check=True,
    )


def _search_eval_mode(config_path: Path) -> str:
    """Return the evaluation mode used for search runs ('search' or 'walk_forward')."""
    cfg_data = _load_yaml(config_path)
    mode = cfg_data.get("evaluation", {}).get("evaluation_mode", "search")
    return mode if mode == "walk_forward" else "search"


def _model_table_path(run_dir: Path, evaluation_mode: str) -> Path:
    return run_dir / "artifacts" / evaluation_mode / "tables" / "model_comparison.csv"


def _per_symbol_table_path(run_dir: Path, evaluation_mode: str) -> Path:
    return run_dir / "artifacts" / evaluation_mode / "tables" / "per_symbol_metrics.csv"


def _manifest_path(run_dir: Path, evaluation_mode: str) -> Path:
    return run_dir / "artifacts" / evaluation_mode / "run_manifest.json"


def _load_target_metrics(
    model_table_path: Path,
    per_symbol_table_path: Path,
    target_model: str,
) -> tuple[TargetMetrics, str]:
    table = pd.read_csv(model_table_path)
    row = table.loc[table["model"] == target_model]
    if row.empty:
        raise ValueError(f"Target model {target_model} not found in {model_table_path}")
    per_symbol = None
    if per_symbol_table_path.exists():
        try:
            _ps = pd.read_csv(per_symbol_table_path)
            per_symbol = _ps if not _ps.empty else None
        except Exception:
            per_symbol = None
    target_metrics = TargetMetrics.from_sources(row.iloc[0], per_symbol)
    best_overall_model = str(table.iloc[0]["model"])
    return target_metrics, best_overall_model


def _search_controls(config: dict[str, Any], state: dict[str, Any]) -> tuple[str, float | None]:
    evaluation_cfg = config.get("evaluation", {})
    selection_metric = str(evaluation_cfg.get("selection_metric", state.get("selection_metric", "event_f1")))
    max_false_alarms = evaluation_cfg.get("max_false_alarms_per_day", state.get("max_false_alarms_per_day"))
    return selection_metric, None if max_false_alarms is None else float(max_false_alarms)


def _is_improvement(
    candidate: TargetMetrics,
    best: TargetMetrics | None,
    selection_metric: str,
    max_false_alarms_per_day: float | None,
) -> bool:
    if candidate.mean_lead_bars < 0:
        return False
    if max_false_alarms_per_day is not None and candidate.false_alarms_per_day >= max_false_alarms_per_day:
        return False
    if best is None:
        return True
    candidate_value = float(getattr(candidate, selection_metric))
    best_value = float(getattr(best, selection_metric))
    if candidate_value > best_value + 1e-4:
        return True
    if candidate_value < best_value - 1e-4:
        return False
    if candidate.event_recall > best.event_recall + 1e-4:
        return True
    if candidate.event_recall < best.event_recall - 1e-4:
        return False
    if candidate.false_alarms_per_day < best.false_alarms_per_day - 1e-4:
        return True
    if candidate.false_alarms_per_day > best.false_alarms_per_day + 1e-4:
        return False
    if candidate.pr_auc > best.pr_auc + 1e-4:
        return True
    if candidate.pr_auc < best.pr_auc - 1e-4:
        return False
    if candidate.vix_event_f1 > best.vix_event_f1 + 1e-4:
        return True
    if candidate.vix_event_f1 < best.vix_event_f1 - 1e-4:
        return False
    return candidate.mean_lead_bars > best.mean_lead_bars


def _append_history(history_path: Path, payload: dict[str, Any]) -> None:
    history_path.parent.mkdir(parents=True, exist_ok=True)
    with open(history_path, "a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload) + "\n")


def _promote_best(
    state: dict[str, Any],
    metrics: TargetMetrics,
    run_dir: Path,
    config_path: Path,
    model_table_path: Path,
    per_symbol_table_path: Path,
    best_overall: str,
) -> None:
    state["best_validation_metrics"] = metrics.__dict__
    state["best_run_dir"] = str(run_dir)
    state["best_config_path"] = str(config_path)
    state["best_model_table_path"] = str(model_table_path)
    state["best_per_symbol_table_path"] = str(per_symbol_table_path)
    state["best_overall_model"] = best_overall


def _candidate_payload(
    metrics: TargetMetrics,
    run_dir: Path,
    config_path: Path,
    model_table_path: Path,
    per_symbol_table_path: Path,
    best_overall: str,
    delta: dict[str, Any],
) -> dict[str, Any]:
    return {
        "metrics": metrics.__dict__,
        "run_dir": str(run_dir),
        "config_path": str(config_path),
        "model_table_path": str(model_table_path),
        "per_symbol_table_path": str(per_symbol_table_path),
        "best_overall_model": best_overall,
        "config_delta": {
            "parameter": delta["key"],
            "old_value": delta["old_value"],
            "new_value": delta["new_value"],
            "stage": delta["stage"],
        },
    }


def _prepare_base_run(base_config_path: Path, target_model: str, state: dict[str, Any], state_path: Path) -> dict[str, Any]:
    if state["best_validation_metrics"] is not None:
        return state
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = RUNS_DIR / f"{timestamp}_baseline"
    run_dir.mkdir(parents=True, exist_ok=True)
    config = _load_yaml(base_config_path)
    selection_metric, max_false_alarms = _search_controls(config, state)
    config["outputs"]["root_dir"] = str(run_dir / "artifacts")
    config_path = run_dir / "config.yaml"
    _write_yaml(config_path, config)
    search_mode = _search_eval_mode(config_path)
    _run_pipeline(config_path, evaluation_mode="search")
    model_table_path = _model_table_path(run_dir, search_mode)
    per_symbol_path = _per_symbol_table_path(run_dir, search_mode)
    metrics, best_overall = _load_target_metrics(model_table_path, per_symbol_path, target_model)
    state["selection_metric"] = selection_metric
    state["max_false_alarms_per_day"] = max_false_alarms
    _promote_best(state, metrics, run_dir, config_path, model_table_path, per_symbol_path, best_overall)
    _write_state(state_path, state)
    _append_history(
        HISTORY_PATH,
        {
            "timestamp": timestamp,
            "iteration": 0,
            "run_dir": str(run_dir),
            "target_model": target_model,
            "metric_source": "validation",
            "selection_metric": selection_metric,
            "decision": "baseline",
            "metrics": metrics.__dict__,
            "best_overall_model": best_overall,
            "model_table_path": str(model_table_path),
            "per_symbol_table_path": str(per_symbol_path),
        },
    )
    return state


def _write_summary(state: dict[str, Any], latest_manifest: dict[str, Any]) -> None:
    pending = state.get("pending_candidate")
    summary_lines = [
        "# Autoresearch Summary",
        "",
        f"- Metric source: `validation`",
        f"- Target model: `{latest_manifest['target_model']}`",
        f"- Selection metric: `{state['selection_metric']}`",
        f"- False-alarm cap/day: `{state['max_false_alarms_per_day']}`",
        f"- Latest iteration: `{state['iteration']}`",
        f"- Latest decision: `{latest_manifest['decision']}`",
        f"- Latest stage: `{latest_manifest['config_delta']['stage']}`",
        f"- Latest mutation: `{latest_manifest['config_delta']['parameter']} = {latest_manifest['config_delta']['new_value']}`",
        f"- Latest validation PR-AUC: `{latest_manifest['metrics']['pr_auc']:.4f}`",
        f"- Latest validation event F1: `{latest_manifest['metrics']['event_f1']:.4f}`",
        f"- Latest validation event recall: `{latest_manifest['metrics']['event_recall']:.4f}`",
        f"- Latest VIX event F1: `{latest_manifest['metrics']['vix_event_f1']:.4f}`",
        f"- Best validation PR-AUC: `{state['best_validation_metrics']['pr_auc']:.4f}`",
        f"- Best validation event F1: `{state['best_validation_metrics']['event_f1']:.4f}`",
        f"- Best validation event recall: `{state['best_validation_metrics']['event_recall']:.4f}`",
        f"- Best VIX event F1: `{state['best_validation_metrics'].get('vix_event_f1', 0.0):.4f}`",
        f"- Best run dir: `{state['best_run_dir']}`",
    ]
    if pending is not None:
        summary_lines.append(f"- Pending candidate event F1: `{pending['metrics']['event_f1']:.4f}`")
        summary_lines.append(f"- Pending candidate run dir: `{pending['run_dir']}`")
    SUMMARY_PATH.write_text("\n".join(summary_lines), encoding="utf-8")


def run_iteration(base_config_path: Path, target_model: str, state_path: Path) -> dict[str, Any]:
    state = _load_state(state_path)
    state = _prepare_base_run(base_config_path, target_model, state, state_path)
    base_config_path_for_mutation = Path(
        state["pending_candidate"]["config_path"] if state.get("pending_candidate") else state["best_config_path"]
    )
    base_config = _load_yaml(base_config_path_for_mutation)
    selection_metric, max_false_alarms = _search_controls(base_config, state)
    state["selection_metric"] = selection_metric
    state["max_false_alarms_per_day"] = max_false_alarms

    tried_signatures = set(state.get("tried_mutations", []))
    delta = _choose_mutation(base_config, tried_signatures)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = RUNS_DIR / f"{timestamp}_iter_{int(state['iteration']) + 1:03d}"
    run_dir.mkdir(parents=True, exist_ok=True)
    candidate_config = json.loads(json.dumps(base_config))
    _set_nested(candidate_config, delta["key"], delta["new_value"])
    candidate_config["outputs"]["root_dir"] = str(run_dir / "artifacts")
    candidate_config_path = run_dir / "config.yaml"
    _write_yaml(candidate_config_path, candidate_config)
    _run_pipeline(candidate_config_path, evaluation_mode="search")
    search_mode = _search_eval_mode(candidate_config_path)
    model_table_path = _model_table_path(run_dir, search_mode)
    per_symbol_path = _per_symbol_table_path(run_dir, search_mode)
    candidate_metrics, best_overall = _load_target_metrics(model_table_path, per_symbol_path, target_model)
    current_best = TargetMetrics(**state["best_validation_metrics"]) if state["best_validation_metrics"] else None
    pending_candidate = state.get("pending_candidate")

    if pending_candidate is None:
        keep = _is_improvement(candidate_metrics, current_best, selection_metric, max_false_alarms)
        if keep:
            decision = "candidate_keep"
            state["pending_candidate"] = _candidate_payload(
                candidate_metrics,
                run_dir,
                candidate_config_path,
                model_table_path,
                per_symbol_path,
                best_overall,
                delta,
            )
        else:
            decision = "discard"
    else:
        pending_metrics = TargetMetrics(**pending_candidate["metrics"])
        confirmed = _is_improvement(candidate_metrics, pending_metrics, selection_metric, max_false_alarms) and _is_improvement(
            candidate_metrics,
            current_best,
            selection_metric,
            max_false_alarms,
        )
        if confirmed:
            decision = "keep_confirmed"
            _promote_best(state, candidate_metrics, run_dir, candidate_config_path, model_table_path, per_symbol_path, best_overall)
        else:
            decision = "candidate_rejected"
        state["pending_candidate"] = None

    run_manifest = {
        "timestamp": timestamp,
        "iteration": int(state["iteration"]) + 1,
        "target_model": target_model,
        "metric_source": "validation",
        "selection_metric": selection_metric,
        "max_false_alarms_per_day": max_false_alarms,
        "hypothesis": delta["hypothesis"],
        "config_delta": {
            "parameter": delta["key"],
            "old_value": delta["old_value"],
            "new_value": delta["new_value"],
            "stage": delta["stage"],
        },
        "metrics": candidate_metrics.__dict__,
        "best_overall_model": best_overall,
        "decision": decision,
        "run_dir": str(run_dir),
        "config_path": str(candidate_config_path),
        "model_table_path": str(model_table_path),
        "per_symbol_table_path": str(per_symbol_path),
        "pipeline_manifest_path": str(_manifest_path(run_dir, "search")),
    }
    (run_dir / "autoresearch_manifest.json").write_text(json.dumps(run_manifest, indent=2), encoding="utf-8")
    _append_history(HISTORY_PATH, run_manifest)

    state["iteration"] = int(state["iteration"]) + 1
    state["tried_mutations"] = sorted(
        tried_signatures | {_signature({"key": delta["key"], "new_value": delta["new_value"]})}
    )
    _write_state(state_path, state)
    _write_summary(state, run_manifest)
    return run_manifest


def finalize_best(target_model: str, state_path: Path) -> dict[str, Any]:
    state = _load_state(state_path)
    if state["best_config_path"] is None:
        raise ValueError("No validation-selected best config is available to finalize")
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = FINAL_RUNS_DIR / f"{timestamp}_final"
    run_dir.mkdir(parents=True, exist_ok=True)
    final_config = _load_yaml(Path(state["best_config_path"]))
    final_config["outputs"]["root_dir"] = str(run_dir / "artifacts")
    final_config_path = run_dir / "config.yaml"
    _write_yaml(final_config_path, final_config)
    _run_pipeline(final_config_path, evaluation_mode="final")
    model_table_path = _model_table_path(run_dir, "final")
    per_symbol_path = _per_symbol_table_path(run_dir, "final")
    metrics, best_overall = _load_target_metrics(model_table_path, per_symbol_path, target_model)
    final_manifest = {
        "timestamp": timestamp,
        "target_model": target_model,
        "metric_source": "final_test",
        "selection_metric": state.get("selection_metric", "event_f1"),
        "max_false_alarms_per_day": state.get("max_false_alarms_per_day"),
        "selected_validation_run_dir": state["best_run_dir"],
        "selected_validation_metrics": state["best_validation_metrics"],
        "final_metrics": metrics.__dict__,
        "best_overall_model": best_overall,
        "run_dir": str(run_dir),
        "config_path": str(final_config_path),
        "model_table_path": str(model_table_path),
        "per_symbol_table_path": str(per_symbol_path),
        "pipeline_manifest_path": str(_manifest_path(run_dir, "final")),
    }
    FINAL_RUNS_DIR.mkdir(parents=True, exist_ok=True)
    FINALIZATION_PATH.write_text(json.dumps(final_manifest, indent=2), encoding="utf-8")
    return final_manifest


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run an autoresearch-style experiment loop for the hybrid TDA+GDL model."
    )
    parser.add_argument("--config", default="configs/autoresearch_hybrid.yaml", help="Base config path")
    parser.add_argument("--target-model", default=DEFAULT_TARGET_MODEL, help="Model row to optimize")
    parser.add_argument("--iterations", type=int, default=1, help="Number of foreground iterations to run")
    parser.add_argument("--continuous", action="store_true", help="Keep proposing and running bounded experiments")
    parser.add_argument("--sleep-seconds", type=float, default=0.0, help="Delay between continuous iterations")
    parser.add_argument("--max-iterations", type=int, default=0, help="Optional cap when using --continuous")
    parser.add_argument(
        "--finalize-best",
        action="store_true",
        help="Run the current validation-selected best config once on the untouched final test split",
    )
    args = parser.parse_args()

    STATE_DIR.mkdir(parents=True, exist_ok=True)
    RUNS_DIR.mkdir(parents=True, exist_ok=True)
    FINAL_RUNS_DIR.mkdir(parents=True, exist_ok=True)
    base_config_path = (PROJECT_ROOT / args.config).resolve()

    if args.finalize_best:
        manifest = finalize_best(args.target_model, STATE_PATH)
        print(json.dumps(manifest, indent=2))
        return

    completed = 0
    while True:
        manifest = run_iteration(base_config_path, args.target_model, STATE_PATH)
        print(json.dumps(manifest, indent=2))
        completed += 1
        if not args.continuous and completed >= args.iterations:
            break
        if args.continuous and args.max_iterations > 0 and completed >= args.max_iterations:
            break
        if args.sleep_seconds > 0:
            time.sleep(args.sleep_seconds)


if __name__ == "__main__":
    main()
