import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from tda_gdl_regime.feature_engineering import split_feature_frame


class SplitTests(unittest.TestCase):
    def test_purged_split_keeps_gap_between_partitions(self) -> None:
        sample_ids = np.arange(100, 300, 5)
        frame = pd.DataFrame(
            {
                "symbol": "SPY",
                "timestamp": pd.date_range("2025-01-01", periods=len(sample_ids), freq="5min"),
                "sample_row_id": sample_ids,
                "label": 0,
                "event_id": -1,
                "cls_realized_volatility": 0.0,
            }
        )
        splits = split_feature_frame(
            frame,
            train_frac=0.6,
            val_frac=0.2,
            test_frac=0.2,
            purge_bars=20,
            embargo_bars=10,
        )
        train_max = int(splits["train"]["sample_row_id"].max())
        val_min = int(splits["val"]["sample_row_id"].min())
        val_max = int(splits["val"]["sample_row_id"].max())
        test_min = int(splits["test"]["sample_row_id"].min())
        self.assertLess(train_max, val_min - 30)
        self.assertLess(val_max, test_min - 30)


if __name__ == "__main__":
    unittest.main()
