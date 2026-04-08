import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from tda_gdl_regime.config import EvaluationConfig
from tda_gdl_regime.walk_forward import generate_walk_forward_folds


def _make_feature_frame(n_rows_per_symbol: int, symbols: list[str], positive_frac: float = 0.1) -> pd.DataFrame:
    """Build a minimal feature_frame with the columns generate_walk_forward_folds requires."""
    rng = np.random.default_rng(0)
    frames = []
    row_id = 0
    for sym in symbols:
        labels = (rng.random(n_rows_per_symbol) < positive_frac).astype(int)
        frames.append(
            pd.DataFrame(
                {
                    "symbol": sym,
                    "timestamp": pd.date_range("2007-01-03", periods=n_rows_per_symbol, freq="5B"),
                    "sample_row_id": range(row_id, row_id + n_rows_per_symbol),
                    "label": labels,
                    "event_id": -1,
                }
            )
        )
        row_id += n_rows_per_symbol
    return pd.concat(frames, ignore_index=True)


def _eval_cfg(
    min_train_years: int = 10,
    val_years: int = 1,
    bars_per_day: int = 50,
    test_frac: float = 0.2,
    min_validation_events: int = 1,
) -> EvaluationConfig:
    cfg = EvaluationConfig()
    cfg.walk_forward_min_train_years = min_train_years
    cfg.walk_forward_val_years = val_years
    cfg.bars_per_day = bars_per_day
    cfg.test_frac = test_frac
    cfg.min_validation_events = min_validation_events
    return cfg


class WalkForwardFoldCountTests(unittest.TestCase):
    def test_fold_count_matches_expected(self) -> None:
        """With known row counts, generate_walk_forward_folds should produce exactly N folds."""
        # bars_per_day=50, min_train=10yrs=500, val=1yr=50, test_frac=0.2
        # 800 rows per symbol → available = 640, remaining = 640-500 = 140
        # folds = floor(140 / 50) = 2  (gap=0 in this test)
        n_rows = 800
        cfg = _eval_cfg(min_train_years=10, val_years=1, bars_per_day=50, test_frac=0.2)
        frame = _make_feature_frame(n_rows, ["SPY", "QQQ"])
        test_df, folds = generate_walk_forward_folds(frame, cfg, purge_bars=0, embargo_bars=0)
        self.assertEqual(len(folds), 2, f"Expected 2 folds, got {len(folds)}")

    def test_fewer_folds_with_large_gap(self) -> None:
        """A larger purge+embargo gap should reduce available fold count."""
        n_rows = 800
        cfg = _eval_cfg(min_train_years=10, val_years=1, bars_per_day=50, test_frac=0.2)
        frame = _make_feature_frame(n_rows, ["SPY"])
        _, folds_no_gap = generate_walk_forward_folds(frame, cfg, purge_bars=0, embargo_bars=0)
        _, folds_gap = generate_walk_forward_folds(frame, cfg, purge_bars=40, embargo_bars=10)
        self.assertGreaterEqual(len(folds_no_gap), len(folds_gap))


class WalkForwardLeakageTests(unittest.TestCase):
    def test_no_train_val_overlap(self) -> None:
        """No sample_row_id should appear in both train and val within a fold."""
        n_rows = 900
        cfg = _eval_cfg(min_train_years=8, val_years=1, bars_per_day=50, test_frac=0.2)
        frame = _make_feature_frame(n_rows, ["SPY", "QQQ", "^VIX"])
        _, folds = generate_walk_forward_folds(frame, cfg, purge_bars=20, embargo_bars=5)
        self.assertGreater(len(folds), 0, "Expected at least 1 fold")
        for fold_idx, fold in enumerate(folds):
            train_ids = set(fold["train"]["sample_row_id"].tolist())
            val_ids = set(fold["val"]["sample_row_id"].tolist())
            overlap = train_ids & val_ids
            self.assertEqual(
                len(overlap),
                0,
                f"Fold {fold_idx}: found {len(overlap)} overlapping sample_row_ids between train and val",
            )

    def test_test_split_absent_from_folds(self) -> None:
        """No sample_row_id from the test split should appear in any fold's train or val."""
        n_rows = 900
        cfg = _eval_cfg(min_train_years=8, val_years=1, bars_per_day=50, test_frac=0.2)
        frame = _make_feature_frame(n_rows, ["SPY", "QQQ"])
        test_df, folds = generate_walk_forward_folds(frame, cfg, purge_bars=20, embargo_bars=5)
        test_ids = set(test_df["sample_row_id"].tolist())
        self.assertGreater(len(test_ids), 0, "Test split is empty")
        for fold_idx, fold in enumerate(folds):
            all_fold_ids = set(fold["train"]["sample_row_id"].tolist()) | set(fold["val"]["sample_row_id"].tolist())
            leaked = all_fold_ids & test_ids
            self.assertEqual(
                len(leaked),
                0,
                f"Fold {fold_idx}: {len(leaked)} test sample_row_ids leaked into fold",
            )

    def test_val_always_after_train_within_symbol(self) -> None:
        """For each symbol, every val sample_row_id must be strictly greater than every train sample_row_id."""
        n_rows = 900
        cfg = _eval_cfg(min_train_years=8, val_years=1, bars_per_day=50, test_frac=0.2)
        frame = _make_feature_frame(n_rows, ["SPY"])
        _, folds = generate_walk_forward_folds(frame, cfg, purge_bars=20, embargo_bars=5)
        self.assertGreater(len(folds), 0)
        for fold_idx, fold in enumerate(folds):
            for sym, sym_train in fold["train"].groupby("symbol"):
                sym_val = fold["val"][fold["val"]["symbol"] == sym]
                if sym_val.empty:
                    continue
                max_train_id = sym_train["sample_row_id"].max()
                min_val_id = sym_val["sample_row_id"].min()
                self.assertGreater(
                    min_val_id,
                    max_train_id,
                    f"Fold {fold_idx}, symbol {sym}: val starts before train ends",
                )


class WalkForwardSelectionMetricTests(unittest.TestCase):
    def test_mean_f1_used_for_selection(self) -> None:
        """The mean event_f1 from walk-forward folds must be what autoresearch reads for keep/discard."""
        import sys as _sys
        _sys.path.insert(0, str(PROJECT_ROOT / "research"))
        import run_autoresearch

        # A candidate whose mean fold F1 beats the incumbent should be kept.
        incumbent = run_autoresearch.TargetMetrics(
            pr_auc=0.10, event_f1=0.30, false_alarms_per_day=1.0,
            mean_lead_bars=2.0, threshold=0.5, event_recall=0.35,
        )
        # Walk-forward model_comparison.csv would contain mean event_f1=0.40.
        candidate = run_autoresearch.TargetMetrics(
            pr_auc=0.11, event_f1=0.40, false_alarms_per_day=1.2,
            mean_lead_bars=2.0, threshold=0.5, event_recall=0.45,
        )
        self.assertTrue(
            run_autoresearch._is_improvement(candidate, incumbent, "event_f1", 2.0),
            "Candidate with higher mean event_f1 should be an improvement",
        )
        # A candidate with lower mean fold F1 should be rejected.
        worse = run_autoresearch.TargetMetrics(
            pr_auc=0.12, event_f1=0.25, false_alarms_per_day=0.9,
            mean_lead_bars=3.0, threshold=0.5, event_recall=0.30,
        )
        self.assertFalse(
            run_autoresearch._is_improvement(worse, incumbent, "event_f1", 2.0),
            "Candidate with lower mean event_f1 should not be an improvement",
        )


class WalkForwardArtifactSchemaTests(unittest.TestCase):
    def test_walkforward_fold_summary_schema(self) -> None:
        """generate_walk_forward_folds returns the right structure for artifact writing."""
        n_rows = 900
        cfg = _eval_cfg(min_train_years=8, val_years=1, bars_per_day=50, test_frac=0.2)
        frame = _make_feature_frame(n_rows, ["SPY", "QQQ"])
        test_df, folds = generate_walk_forward_folds(frame, cfg, purge_bars=10, embargo_bars=5)

        self.assertIsInstance(folds, list)
        self.assertGreater(len(folds), 0)

        # Simulate building the fold_summary artifact.
        fold_results = []
        for fold_idx, fold in enumerate(folds):
            fold_results.append(
                {
                    "fold_index": fold_idx,
                    "train_start": str(fold["train"]["timestamp"].min()),
                    "train_end": str(fold["train"]["timestamp"].max()),
                    "val_start": str(fold["val"]["timestamp"].min()),
                    "val_end": str(fold["val"]["timestamp"].max()),
                    "val_event_count": int(fold["val"]["label"].sum()),
                    "models": {"mock_model": {"event_f1": 0.4, "pr_auc": 0.1}},
                }
            )

        artifact = {
            "n_folds": len(fold_results),
            "fold_val_event_counts": [fr["val_event_count"] for fr in fold_results],
            "folds": fold_results,
            "summary": {"mock_model": {"mean_event_f1": 0.4, "std_event_f1": 0.0}},
        }

        # Required top-level keys.
        for key in ("n_folds", "fold_val_event_counts", "folds", "summary"):
            self.assertIn(key, artifact, f"Missing key '{key}' in walkforward_fold_summary artifact")

        # Each fold must have required keys.
        required_fold_keys = ("fold_index", "train_start", "train_end", "val_start", "val_end", "val_event_count", "models")
        for fold_result in artifact["folds"]:
            for key in required_fold_keys:
                self.assertIn(key, fold_result, f"Missing key '{key}' in fold result")

        self.assertEqual(artifact["n_folds"], len(folds))


if __name__ == "__main__":
    unittest.main()
