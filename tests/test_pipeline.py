import json
import os
import shutil
import subprocess
import sys
import unittest
from pathlib import Path

import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]


class PipelineSmokeTests(unittest.TestCase):
    def test_pipeline_runs_and_writes_artifacts(self) -> None:
        temp_dir = PROJECT_ROOT / "test_artifacts" / "pipeline_smoke"
        temp_dir.mkdir(parents=True, exist_ok=True)
        try:
            config = {
                "data": {
                    "provider": "local_csv",
                    "files": [{"symbol": "SPY", "path": "data/example_prices.csv"}],
                    "timestamp_col": "timestamp",
                    "price_col": "mid_price",
                    "returns_mode": "log",
                },
                "labels": {
                    "lookahead_bars": 20,
                    "volatility_window_bars": 20,
                    "threshold_quantile": 0.8,
                    "threshold_lookback_bars": 200,
                    "min_history_bars": 80,
                    "event_merge_gap": 2,
                    "min_event_span": 1,
                    "positive_transition_only": True,
                },
                "features": {
                    "window_bars": 40,
                    "stride_bars": 10,
                    "embed_dim": 5,
                    "embed_tau": 1,
                    "graph_knn_k": 4,
                    "persistence_image_bins": 4,
                    "topology_feature_sets": ["summary", "betti"],
                    "include_symbol_one_hot": True,
                    "enable_vxx_tailored": True,
                    "vxx_tailored_short_horizon": 6,
                },
                "models": {
                    "enabled": [
                        "vol_threshold",
                        "rf_classical",
                        "rf_topology",
                        "rf_combined",
                        "gcn_graph",
                        "gcn_fusion",
                    ],
                    "rf_estimators": 50,
                    "gdl_hidden_dim": 16,
                    "gdl_epochs": 2,
                    "gdl_batch_size": 32,
                    "gdl_use_cuda": False,
                    "enable_symbol_offset_calibration": True,
                },
                "evaluation": {
                    "train_frac": 0.6,
                    "val_frac": 0.2,
                    "test_frac": 0.2,
                    "purge_bars": 5,
                    "embargo_bars": 5,
                    "min_validation_events": 0,
                    "min_test_events": 0,
                    "early_warning_bars": 10,
                    "bars_per_day": 390,
                    "bootstrap_samples": 0,
                    "bootstrap_block_size": 64,
                    "selection_metric": "event_f1",
                    "max_false_alarms_per_day": 1.0,
                },
                "outputs": {
                    "root_dir": str(temp_dir),
                    "figure_dir": "figures",
                    "table_dir": "tables",
                },
            }
            config_path = temp_dir / "config.yaml"
            with open(config_path, "w", encoding="utf-8") as handle:
                yaml.safe_dump(config, handle, sort_keys=False)
            subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "tda_gdl_regime.run_pipeline",
                    "--config",
                    str(config_path),
                    "--evaluation-mode",
                    "search",
                ],
                cwd=PROJECT_ROOT,
                env={
                    **os.environ,
                    "PYTHONPATH": str(PROJECT_ROOT / "src"),
                },
                check=True,
            )
            search_metrics_path = temp_dir / "search" / "metrics.json"
            self.assertTrue(search_metrics_path.exists())
            payload = json.loads(search_metrics_path.read_text(encoding="utf-8"))
            self.assertEqual(payload["evaluation_mode"], "search")
            self.assertEqual(payload["evaluation_split"], "validation")
            self.assertEqual(payload["selection_metric"], "event_f1")
            self.assertTrue((temp_dir / "search" / "tables" / "short_event_metrics.csv").exists())
            per_symbol = temp_dir / "search" / "tables" / "per_symbol_metrics.csv"
            self.assertTrue(per_symbol.exists())
            per_symbol_text = per_symbol.read_text(encoding="utf-8")
            self.assertIn("model", per_symbol_text)

            subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "tda_gdl_regime.run_pipeline",
                    "--config",
                    str(config_path),
                    "--evaluation-mode",
                    "final",
                ],
                cwd=PROJECT_ROOT,
                env={
                    **os.environ,
                    "PYTHONPATH": str(PROJECT_ROOT / "src"),
                },
                check=True,
            )
            final_metrics_path = temp_dir / "final" / "metrics.json"
            self.assertTrue(final_metrics_path.exists())
            final_payload = json.loads(final_metrics_path.read_text(encoding="utf-8"))
            self.assertEqual(final_payload["evaluation_mode"], "final")
            self.assertEqual(final_payload["evaluation_split"], "final_test")
        finally:
            shutil.rmtree(temp_dir, ignore_errors=True)


if __name__ == "__main__":
    unittest.main()
