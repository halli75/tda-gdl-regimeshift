import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from tda_gdl_regime.config import LabelConfig
from tda_gdl_regime.labels import build_shift_event_labels


class LabelTests(unittest.TestCase):
    def test_shift_labels_mark_a_volatility_spike(self) -> None:
        rng = np.random.default_rng(7)
        calm = rng.normal(0.0, 0.001, size=500)
        shock = rng.normal(0.0, 0.02, size=80)
        returns = np.concatenate([calm, shock, calm])
        prices = 100 * np.exp(np.cumsum(returns))
        frame = pd.DataFrame(
            {
                "timestamp": np.arange(len(prices)),
                "price": prices,
                "symbol": "SPY",
                "return": np.concatenate([[0.0], np.diff(np.log(prices))]),
                "row_id": np.arange(len(prices)),
            }
        )
        labeled = build_shift_event_labels(
            frame,
            LabelConfig(
                lookahead_bars=20,
                volatility_window_bars=20,
                threshold_quantile=0.8,
                threshold_lookback_bars=120,
                min_history_bars=60,
                event_merge_gap=2,
                min_event_span=1,
                positive_transition_only=False,
            ),
        )
        self.assertGreater(int(labeled["shift_event"].sum()), 0)


if __name__ == "__main__":
    unittest.main()
