from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .config import EvaluationConfig, ModelConfig
from .evaluation import select_best_threshold


@dataclass
class FittedModel:
    name: str
    feature_columns: list[str]
    threshold: float
    estimator: object | None = None
    score_column: str | None = None
    validation_f1: float = 0.0


def _score_quantile_grid(scores: np.ndarray) -> list[float]:
    quantiles = np.linspace(0.1, 0.9, 9)
    return sorted(set(float(value) for value in np.quantile(scores, quantiles)))


def _fit_random_forest(X: np.ndarray, y: np.ndarray, cfg: ModelConfig) -> RandomForestClassifier:
    forest = RandomForestClassifier(
        n_estimators=cfg.rf_estimators,
        max_depth=cfg.rf_max_depth,
        random_state=cfg.random_state,
        class_weight="balanced_subsample",
    )
    forest.fit(X, y)
    return forest


def _fit_mlp(X: np.ndarray, y: np.ndarray, cfg: ModelConfig) -> Pipeline:
    network = Pipeline(
        steps=[
            ("scale", StandardScaler()),
            (
                "mlp",
                MLPClassifier(
                    hidden_layer_sizes=tuple(cfg.mlp_hidden_sizes),
                    activation="relu",
                    solver="adam",
                    max_iter=400,
                    random_state=cfg.random_state,
                ),
            ),
        ]
    )
    network.fit(X, y)
    return network


def fit_model_suite(
    splits: dict[str, pd.DataFrame],
    groups: dict[str, list[str]],
    cfg: ModelConfig,
    evaluation_cfg: EvaluationConfig,
) -> dict[str, FittedModel]:
    train_frame = splits["train"]
    val_frame = splits["val"]
    y_train = train_frame["label"].to_numpy(dtype=int)
    fitted: dict[str, FittedModel] = {}
    if "vol_threshold" in cfg.enabled:
        score_column = "cls_realized_volatility"
        scores = val_frame[score_column].to_numpy(dtype=float)
        threshold_grid = _score_quantile_grid(scores)
        threshold, summary = select_best_threshold(
            val_frame,
            scores,
            threshold_grid,
            selection_metric=evaluation_cfg.selection_metric,
            early_warning_bars=evaluation_cfg.early_warning_bars,
            bars_per_day=evaluation_cfg.bars_per_day,
            max_false_alarms_per_day=evaluation_cfg.max_false_alarms_per_day,
            max_positive_rate=evaluation_cfg.max_positive_rate,
        )
        fitted["vol_threshold"] = FittedModel(
            name="vol_threshold",
            feature_columns=[score_column],
            score_column=score_column,
            threshold=threshold,
            validation_f1=float(summary["sample_f1"]),
        )
    model_specs = {
        "rf_classical": ("classical", _fit_random_forest),
        "rf_topology": ("topology", _fit_random_forest),
        "rf_combined": ("combined", _fit_random_forest),
        "mlp_combined": ("combined", _fit_mlp),
    }
    for model_name, (group_name, trainer) in model_specs.items():
        if model_name not in cfg.enabled:
            continue
        columns = groups[group_name]
        estimator = trainer(
            train_frame[columns].to_numpy(dtype=float),
            y_train,
            cfg,
        )
        scores = estimator.predict_proba(val_frame[columns].to_numpy(dtype=float))[:, 1]
        threshold, summary = select_best_threshold(
            val_frame,
            scores,
            cfg.probability_threshold_grid,
            selection_metric=evaluation_cfg.selection_metric,
            early_warning_bars=evaluation_cfg.early_warning_bars,
            bars_per_day=evaluation_cfg.bars_per_day,
            max_false_alarms_per_day=evaluation_cfg.max_false_alarms_per_day,
            max_positive_rate=evaluation_cfg.max_positive_rate,
        )
        fitted[model_name] = FittedModel(
            name=model_name,
            feature_columns=columns,
            estimator=estimator,
            threshold=threshold,
            validation_f1=float(summary["sample_f1"]),
        )
    return fitted


def predict_scores(model: FittedModel, frame: pd.DataFrame) -> np.ndarray:
    if model.estimator is None:
        if model.score_column is None:
            raise ValueError(f"Model {model.name} is missing both estimator and score column")
        return frame[model.score_column].to_numpy(dtype=float)
    values = frame[model.feature_columns].to_numpy(dtype=float)
    return model.estimator.predict_proba(values)[:, 1]
