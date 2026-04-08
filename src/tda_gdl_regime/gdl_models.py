from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import average_precision_score
from torch import nn
from torch.utils.data import DataLoader, Dataset

from .config import EvaluationConfig, ModelConfig
from .evaluation import is_better_summary, select_best_threshold
from .graph_data import GraphDataset


class FocalBCELoss(nn.Module):
    """BCE loss with focal weighting: down-weights easy negatives to spread score distribution."""

    def __init__(self, gamma: float = 2.0, pos_weight: torch.Tensor | None = None) -> None:
        super().__init__()
        self.gamma = gamma
        self.pos_weight = pos_weight

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        bce = nn.functional.binary_cross_entropy_with_logits(
            logits, targets, pos_weight=self.pos_weight, reduction="none"
        )
        probs = torch.sigmoid(logits)
        pt = torch.where(targets == 1, probs, 1.0 - probs)
        focal_weight = (1.0 - pt) ** self.gamma
        return (focal_weight * bce).mean()


@dataclass
class FittedGraphModel:
    name: str
    feature_columns: list[str]
    threshold: float
    estimator: nn.Module
    standardization_mean: np.ndarray | None = None
    standardization_std: np.ndarray | None = None
    validation_f1: float = 0.0
    symbol_offsets: dict[str, float] | None = None
    ensemble_estimators: list[nn.Module] | None = None


class GraphTensorDataset(Dataset):
    def __init__(
        self,
        node_features: np.ndarray,
        adjacency: np.ndarray,
        labels: np.ndarray,
        tabular: np.ndarray | None = None,
    ) -> None:
        self.node_features = torch.tensor(node_features, dtype=torch.float32)
        self.adjacency = torch.tensor(adjacency, dtype=torch.float32)
        self.labels = torch.tensor(labels, dtype=torch.float32)
        if tabular is None:
            self.tabular = torch.empty((len(labels), 0), dtype=torch.float32)
        else:
            self.tabular = torch.tensor(tabular, dtype=torch.float32)

    def __len__(self) -> int:
        return int(self.labels.shape[0])

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        return self.node_features[idx], self.adjacency[idx], self.labels[idx], self.tabular[idx]


class DenseGraphConv(nn.Module):
    def __init__(self, in_dim: int, out_dim: int) -> None:
        super().__init__()
        self.linear = nn.Linear(in_dim, out_dim)

    def forward(self, x: torch.Tensor, adjacency: torch.Tensor) -> torch.Tensor:
        batch_size, node_count, _ = adjacency.shape
        eye = torch.eye(node_count, device=adjacency.device).unsqueeze(0).expand(batch_size, -1, -1)
        adj = adjacency + eye
        degree = adj.sum(dim=-1).clamp(min=1.0)
        inv_sqrt_degree = degree.pow(-0.5)
        normalized = inv_sqrt_degree.unsqueeze(-1) * adj * inv_sqrt_degree.unsqueeze(-2)
        propagated = torch.bmm(normalized, x)
        return self.linear(propagated)


class GraphClassifier(nn.Module):
    def __init__(
        self,
        node_dim: int,
        hidden_dim: int,
        dropout: float,
        tabular_dim: int = 0,
    ) -> None:
        super().__init__()
        self.conv1 = DenseGraphConv(node_dim, hidden_dim)
        self.conv2 = DenseGraphConv(hidden_dim, hidden_dim)
        self.dropout = nn.Dropout(dropout)
        self.tabular_dim = tabular_dim
        if tabular_dim > 0:
            self.tabular_proj = nn.Sequential(
                nn.Linear(tabular_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
            )
            classifier_in = hidden_dim * 2
        else:
            self.tabular_proj = None
            classifier_in = hidden_dim
        self.classifier = nn.Sequential(
            nn.Linear(classifier_in, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def forward(
        self,
        node_features: torch.Tensor,
        adjacency: torch.Tensor,
        tabular: torch.Tensor | None = None,
    ) -> torch.Tensor:
        hidden = torch.relu(self.conv1(node_features, adjacency))
        hidden = self.dropout(hidden)
        hidden = torch.relu(self.conv2(hidden, adjacency))
        graph_embedding = hidden.mean(dim=1)
        if self.tabular_proj is not None and tabular is not None:
            fused = torch.cat([graph_embedding, self.tabular_proj(tabular)], dim=-1)
        else:
            fused = graph_embedding
        return self.classifier(fused).squeeze(-1)


def _device(cfg: ModelConfig) -> torch.device:
    if cfg.gdl_use_cuda and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def _standardize(
    train: pd.DataFrame,
    other: list[pd.DataFrame],
    columns: list[str],
) -> tuple[np.ndarray, list[np.ndarray], np.ndarray, np.ndarray]:
    train_values = train[columns].to_numpy(dtype=np.float32)
    mean = train_values.mean(axis=0)
    std = train_values.std(axis=0)
    std[std == 0.0] = 1.0
    train_scaled = (train_values - mean) / std
    other_scaled = [(frame[columns].to_numpy(dtype=np.float32) - mean) / std for frame in other]
    return train_scaled, other_scaled, mean, std


def _fit_single_graph_model(
    model_name: str,
    graph_train: GraphDataset,
    graph_val: GraphDataset,
    val_frame: pd.DataFrame,
    tabular_train: np.ndarray | None,
    tabular_val: np.ndarray | None,
    cfg: ModelConfig,
    evaluation_cfg: EvaluationConfig,
    seed: int | None = None,
) -> tuple[nn.Module, float, float]:
    device = _device(cfg)
    actual_seed = seed if seed is not None else cfg.random_state
    torch.manual_seed(actual_seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(actual_seed)
    node_dim = graph_train.node_features.shape[-1]
    tabular_dim = 0 if tabular_train is None else int(tabular_train.shape[-1])
    model = GraphClassifier(
        node_dim=node_dim,
        hidden_dim=cfg.gdl_hidden_dim,
        dropout=cfg.gdl_dropout,
        tabular_dim=tabular_dim,
    ).to(device)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=cfg.gdl_learning_rate,
        weight_decay=cfg.gdl_weight_decay,
    )
    train_labels = graph_train.meta["label"].to_numpy(dtype=np.float32)
    pos_count = float(train_labels.sum())
    neg_count = float(len(train_labels) - pos_count)
    pos_weight = None
    if cfg.gdl_balance_classes and pos_count > 0.0 and neg_count > 0.0:
        pos_weight = torch.tensor([neg_count / pos_count], dtype=torch.float32, device=device)
    if cfg.gdl_focal_gamma is not None:
        criterion = FocalBCELoss(gamma=cfg.gdl_focal_gamma, pos_weight=pos_weight)
    else:
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    train_loader = DataLoader(
        GraphTensorDataset(
            graph_train.node_features,
            graph_train.adjacency,
            graph_train.meta["label"].to_numpy(dtype=np.float32),
            tabular_train,
        ),
        batch_size=cfg.gdl_batch_size,
        shuffle=True,
    )
    best_state = None
    best_score = -1.0
    epochs_without_improvement = 0
    for _ in range(cfg.gdl_epochs):
        model.train()
        for node_features, adjacency, labels, tabular in train_loader:
            node_features = node_features.to(device)
            adjacency = adjacency.to(device)
            labels = labels.to(device)
            tabular = tabular.to(device)
            logits = model(node_features, adjacency, tabular)
            loss = criterion(logits, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        val_scores = predict_graph_scores_raw(model, graph_val, tabular_val, device)
        y_val = graph_val.meta["label"].to_numpy(dtype=int)
        if len(np.unique(y_val)) > 1:
            score = float(average_precision_score(y_val, val_scores))
        else:
            score = 0.0
        if score > best_score:
            best_score = score
            best_state = {key: value.cpu().clone() for key, value in model.state_dict().items()}
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= cfg.gdl_patience:
                break
    if best_state is not None:
        model.load_state_dict(best_state)
    val_scores = predict_graph_scores_raw(model, graph_val, tabular_val, device)
    threshold, summary = select_best_threshold(
        val_frame,
        val_scores,
        cfg.probability_threshold_grid,
        selection_metric=evaluation_cfg.selection_metric,
        early_warning_bars=evaluation_cfg.early_warning_bars,
        bars_per_day=evaluation_cfg.bars_per_day,
        max_false_alarms_per_day=evaluation_cfg.max_false_alarms_per_day,
        max_positive_rate=evaluation_cfg.max_positive_rate,
    )
    return model, threshold, float(summary["sample_f1"])


def _safe_logit(probabilities: np.ndarray) -> np.ndarray:
    clipped = np.clip(probabilities, 1e-6, 1.0 - 1e-6)
    return np.log(clipped / (1.0 - clipped))


def _apply_symbol_offsets(scores: np.ndarray, symbols: pd.Series, offsets: dict[str, float] | None) -> np.ndarray:
    if not offsets:
        return np.asarray(scores, dtype=float)
    logits = _safe_logit(np.asarray(scores, dtype=float))
    adjusted = logits.copy()
    symbol_values = symbols.astype(str).to_numpy()
    for symbol, offset in offsets.items():
        adjusted[symbol_values == symbol] += float(offset)
    return 1.0 / (1.0 + np.exp(-adjusted))


def _fit_symbol_offsets(
    frame: pd.DataFrame,
    base_scores: np.ndarray,
    cfg: ModelConfig,
    evaluation_cfg: EvaluationConfig,
) -> tuple[dict[str, float], float, dict[str, float]]:
    best_offsets = {str(symbol): 0.0 for symbol in sorted(frame["symbol"].astype(str).unique())}
    best_threshold, best_summary = select_best_threshold(
        frame,
        base_scores,
        cfg.probability_threshold_grid,
        selection_metric=evaluation_cfg.selection_metric,
        early_warning_bars=evaluation_cfg.early_warning_bars,
        bars_per_day=evaluation_cfg.bars_per_day,
        max_false_alarms_per_day=evaluation_cfg.max_false_alarms_per_day,
        max_positive_rate=evaluation_cfg.max_positive_rate,
    )
    for _ in range(2):
        improved = False
        for symbol in sorted(best_offsets):
            symbol_best_offsets = dict(best_offsets)
            symbol_best_threshold = best_threshold
            symbol_best_summary = best_summary
            for offset in cfg.symbol_offset_grid:
                candidate_offsets = dict(best_offsets)
                candidate_offsets[symbol] = float(offset)
                candidate_scores = _apply_symbol_offsets(base_scores, frame["symbol"], candidate_offsets)
                threshold, summary = select_best_threshold(
                    frame,
                    candidate_scores,
                    cfg.probability_threshold_grid,
                    selection_metric=evaluation_cfg.selection_metric,
                    early_warning_bars=evaluation_cfg.early_warning_bars,
                    bars_per_day=evaluation_cfg.bars_per_day,
                    max_false_alarms_per_day=evaluation_cfg.max_false_alarms_per_day,
                    max_positive_rate=evaluation_cfg.max_positive_rate,
                )
                if is_better_summary(
                    summary,
                    symbol_best_summary,
                    evaluation_cfg.selection_metric,
                    evaluation_cfg.max_false_alarms_per_day,
                ):
                    symbol_best_offsets = candidate_offsets
                    symbol_best_threshold = threshold
                    symbol_best_summary = summary
            if symbol_best_offsets != best_offsets:
                improved = True
                best_offsets = symbol_best_offsets
                best_threshold = symbol_best_threshold
                best_summary = symbol_best_summary
        if not improved:
            break
    return best_offsets, best_threshold, best_summary


def fit_gdl_model_suite(
    graph_splits: dict[str, GraphDataset],
    feature_splits: dict[str, pd.DataFrame],
    groups: dict[str, list[str]],
    cfg: ModelConfig,
    evaluation_cfg: EvaluationConfig,
) -> dict[str, FittedGraphModel]:
    fitted: dict[str, FittedGraphModel] = {}
    if "gcn_graph" in cfg.enabled:
        estimator, threshold, validation_f1 = _fit_single_graph_model(
            "gcn_graph",
            graph_train=graph_splits["train"],
            graph_val=graph_splits["val"],
            val_frame=feature_splits["val"],
            tabular_train=None,
            tabular_val=None,
            cfg=cfg,
            evaluation_cfg=evaluation_cfg,
        )
        fitted["gcn_graph"] = FittedGraphModel(
            name="gcn_graph",
            feature_columns=[],
            threshold=threshold,
            estimator=estimator,
            validation_f1=validation_f1,
        )
    if "gcn_fusion" in cfg.enabled:
        columns = groups["combined"]
        train_scaled, [val_scaled], mean, std = _standardize(
            feature_splits["train"],
            [feature_splits["val"]],
            columns,
        )
        n_ensemble = max(1, cfg.gdl_n_ensemble)
        ensemble_seeds = [cfg.random_state + i for i in range(n_ensemble)]
        ensemble_estimators: list[nn.Module] = []
        all_val_scores: list[np.ndarray] = []
        for seed in ensemble_seeds:
            member, _, _ = _fit_single_graph_model(
                "gcn_fusion",
                graph_train=graph_splits["train"],
                graph_val=graph_splits["val"],
                val_frame=feature_splits["val"],
                tabular_train=train_scaled,
                tabular_val=val_scaled,
                cfg=cfg,
                evaluation_cfg=evaluation_cfg,
                seed=seed,
            )
            ensemble_estimators.append(member)
            all_val_scores.append(predict_graph_scores_raw(member, graph_splits["val"], val_scaled, _device(cfg)))
        avg_val_scores = np.mean(all_val_scores, axis=0)
        threshold, summary = select_best_threshold(
            feature_splits["val"],
            avg_val_scores,
            cfg.probability_threshold_grid,
            selection_metric=evaluation_cfg.selection_metric,
            early_warning_bars=evaluation_cfg.early_warning_bars,
            bars_per_day=evaluation_cfg.bars_per_day,
            max_false_alarms_per_day=evaluation_cfg.max_false_alarms_per_day,
            max_positive_rate=evaluation_cfg.max_positive_rate,
        )
        validation_f1 = float(summary["sample_f1"])
        estimator = ensemble_estimators[0]
        symbol_offsets = None
        if cfg.enable_symbol_offset_calibration:
            if cfg.symbol_offset_fit_split != "validation":
                raise ValueError("Only validation-based post-hoc symbol offset calibration is supported")
            symbol_offsets, threshold, summary = _fit_symbol_offsets(
                feature_splits["val"],
                avg_val_scores,
                cfg=cfg,
                evaluation_cfg=evaluation_cfg,
            )
            validation_f1 = float(summary["sample_f1"])
        fitted["gcn_fusion"] = FittedGraphModel(
            name="gcn_fusion",
            feature_columns=columns,
            threshold=threshold,
            estimator=estimator,
            standardization_mean=mean,
            standardization_std=std,
            validation_f1=validation_f1,
            symbol_offsets=symbol_offsets,
            ensemble_estimators=ensemble_estimators if n_ensemble > 1 else None,
        )
    return fitted


def predict_graph_scores_raw(
    model: nn.Module,
    graph_split: GraphDataset,
    tabular_values: np.ndarray | None,
    device: torch.device,
) -> np.ndarray:
    model.eval()
    with torch.no_grad():
        node_features = torch.tensor(graph_split.node_features, dtype=torch.float32, device=device)
        adjacency = torch.tensor(graph_split.adjacency, dtype=torch.float32, device=device)
        tabular = None
        if tabular_values is not None:
            tabular = torch.tensor(tabular_values, dtype=torch.float32, device=device)
        else:
            tabular = torch.empty((graph_split.node_features.shape[0], 0), dtype=torch.float32, device=device)
        logits = model(node_features, adjacency, tabular)
        return torch.sigmoid(logits).cpu().numpy()


def predict_graph_scores(
    model: FittedGraphModel,
    graph_split: GraphDataset,
    feature_split: pd.DataFrame,
    cfg: ModelConfig,
) -> np.ndarray:
    tabular_values = None
    if model.feature_columns:
        raw = feature_split[model.feature_columns].to_numpy(dtype=np.float32)
        if model.standardization_mean is None or model.standardization_std is None:
            raise ValueError(f"Model {model.name} is missing tabular standardization state")
        tabular_values = (raw - model.standardization_mean) / model.standardization_std
    estimators = model.ensemble_estimators if model.ensemble_estimators else [model.estimator]
    all_scores = [predict_graph_scores_raw(m, graph_split, tabular_values, _device(cfg)) for m in estimators]
    scores = np.mean(all_scores, axis=0)
    return _apply_symbol_offsets(scores, feature_split["symbol"], model.symbol_offsets)
