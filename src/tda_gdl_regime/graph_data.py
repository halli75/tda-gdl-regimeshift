from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .config import FeatureConfig
from .feature_engineering import iter_window_samples
from .graph_builder import build_knn_graph
from .tda_features import delay_embedding


@dataclass
class GraphDataset:
    meta: pd.DataFrame
    node_features: np.ndarray
    adjacency: np.ndarray


def build_graph_dataset(frame: pd.DataFrame, cfg: FeatureConfig) -> GraphDataset:
    rows: list[dict[str, object]] = []
    node_features: list[np.ndarray] = []
    adjacency: list[np.ndarray] = []
    for sample in iter_window_samples(frame, cfg):
        embedding = delay_embedding(sample.window_returns, cfg.embed_dim, cfg.embed_tau)
        normalized = (embedding - embedding.mean(axis=0)) / (embedding.std(axis=0) + 1e-8)
        rows.append(
            {
                "symbol": sample.symbol,
                "timestamp": sample.timestamp,
                "sample_row_id": sample.sample_row_id,
                "label": sample.label,
                "event_id": sample.event_id,
            }
        )
        node_features.append(normalized.astype(np.float32))
        adjacency.append(build_knn_graph(normalized, cfg.graph_knn_k).astype(np.float32))
    if not rows:
        raise ValueError("Graph dataset is empty")
    return GraphDataset(
        meta=pd.DataFrame(rows),
        node_features=np.stack(node_features),
        adjacency=np.stack(adjacency),
    )


def split_graph_dataset(
    dataset: GraphDataset,
    feature_splits: dict[str, pd.DataFrame],
) -> dict[str, GraphDataset]:
    index_by_key = {
        (row.symbol, int(row.sample_row_id)): idx
        for idx, row in enumerate(dataset.meta.itertuples(index=False))
    }
    split_sets: dict[str, GraphDataset] = {}
    for split_name, split_frame in feature_splits.items():
        indices = [
            index_by_key[(row.symbol, int(row.sample_row_id))]
            for row in split_frame[["symbol", "sample_row_id"]].itertuples(index=False)
        ]
        split_sets[split_name] = GraphDataset(
            meta=dataset.meta.iloc[indices].reset_index(drop=True).copy(),
            node_features=dataset.node_features[indices],
            adjacency=dataset.adjacency[indices],
        )
    return split_sets
