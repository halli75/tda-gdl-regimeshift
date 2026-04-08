import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from tda_gdl_regime.config import EvaluationConfig, ModelConfig  # noqa: E402
from tda_gdl_regime.gdl_models import fit_gdl_model_suite  # noqa: E402
from tda_gdl_regime.graph_data import GraphDataset  # noqa: E402


def _graph_split(symbols: list[str], labels: list[int], offset: int) -> GraphDataset:
    rng = np.random.default_rng(123 + offset)
    node_features = rng.normal(size=(len(labels), 4, 3)).astype(np.float32)
    adjacency = np.tile(np.eye(4, dtype=np.float32), (len(labels), 1, 1))
    meta = pd.DataFrame(
        {
            "symbol": symbols,
            "timestamp": pd.date_range("2025-01-01", periods=len(labels), freq="h"),
            "sample_row_id": np.arange(offset, offset + len(labels)),
            "label": labels,
            "event_id": np.arange(len(labels)),
        }
    )
    return GraphDataset(meta=meta, node_features=node_features, adjacency=adjacency)


class GdlCalibrationTests(unittest.TestCase):
    def test_post_hoc_symbol_offsets_do_not_change_backbone_weights(self) -> None:
        train_symbols = ["SPY", "SPY", "VXX", "VXX", "QQQ", "QQQ", "SPY", "VXX"]
        val_symbols = ["SPY", "VXX", "QQQ", "SPY", "VXX", "QQQ"]
        train = pd.DataFrame(
            {
                "symbol": train_symbols,
                "timestamp": pd.date_range("2025-01-01", periods=len(train_symbols), freq="h"),
                "sample_row_id": np.arange(len(train_symbols)),
                "label": [0, 1, 0, 1, 0, 1, 0, 1],
                "event_id": np.arange(len(train_symbols)),
                "cls_realized_volatility": [0.1, 0.6, 0.2, 0.9, 0.15, 0.7, 0.12, 0.85],
                "sym_QQQ": [0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0],
                "sym_SPY": [1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0],
                "sym_VXX": [0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0, 1.0],
            }
        )
        val = pd.DataFrame(
            {
                "symbol": val_symbols,
                "timestamp": pd.date_range("2025-02-01", periods=len(val_symbols), freq="h"),
                "sample_row_id": np.arange(100, 100 + len(val_symbols)),
                "label": [0, 1, 0, 1, 1, 0],
                "event_id": np.arange(10, 10 + len(val_symbols)),
                "cls_realized_volatility": [0.11, 0.75, 0.16, 0.5, 0.95, 0.2],
                "sym_QQQ": [0.0, 0.0, 1.0, 0.0, 0.0, 1.0],
                "sym_SPY": [1.0, 0.0, 0.0, 1.0, 0.0, 0.0],
                "sym_VXX": [0.0, 1.0, 0.0, 0.0, 1.0, 0.0],
            }
        )
        splits = {"train": train, "val": val}
        graph_splits = {
            "train": _graph_split(train_symbols, train["label"].tolist(), 0),
            "val": _graph_split(val_symbols, val["label"].tolist(), 100),
        }
        groups = {
            "combined": ["cls_realized_volatility", "sym_QQQ", "sym_SPY", "sym_VXX"],
            "classical": ["cls_realized_volatility", "sym_QQQ", "sym_SPY", "sym_VXX"],
            "topology": ["sym_QQQ", "sym_SPY", "sym_VXX"],
        }
        evaluation_cfg = EvaluationConfig(
            early_warning_bars=2,
            bars_per_day=7,
            bootstrap_samples=0,
            bootstrap_block_size=8,
            selection_metric="event_f1",
            max_false_alarms_per_day=1.0,
        )
        base_cfg = ModelConfig(
            enabled=["gcn_fusion"],
            gdl_hidden_dim=8,
            gdl_dropout=0.1,
            gdl_epochs=2,
            gdl_patience=2,
            gdl_batch_size=4,
            gdl_learning_rate=0.001,
            gdl_use_cuda=False,
            probability_threshold_grid=[0.3, 0.5, 0.7],
            symbol_offset_grid=[-0.5, 0.0, 0.5],
        )
        no_calibration = fit_gdl_model_suite(
            graph_splits,
            splits,
            groups,
            base_cfg,
            evaluation_cfg,
        )["gcn_fusion"]
        with_calibration_cfg = ModelConfig(**{**base_cfg.__dict__, "enable_symbol_offset_calibration": True})
        with_calibration = fit_gdl_model_suite(
            graph_splits,
            splits,
            groups,
            with_calibration_cfg,
            evaluation_cfg,
        )["gcn_fusion"]

        no_calibration_state = no_calibration.estimator.state_dict()
        with_calibration_state = with_calibration.estimator.state_dict()
        self.assertIsNotNone(with_calibration.symbol_offsets)
        self.assertTrue(with_calibration.symbol_offsets)
        self.assertEqual(no_calibration_state.keys(), with_calibration_state.keys())
        for key in no_calibration_state:
            self.assertTrue(torch.equal(no_calibration_state[key], with_calibration_state[key]), key)


if __name__ == "__main__":
    unittest.main()
