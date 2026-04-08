import sys
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "research"))

import run_autoresearch  # noqa: E402


class AutoresearchLogicTests(unittest.TestCase):
    def test_false_alarm_cap_blocks_candidate(self) -> None:
        best = run_autoresearch.TargetMetrics(
            pr_auc=0.40,
            event_f1=0.55,
            false_alarms_per_day=0.05,
            mean_lead_bars=2.0,
            threshold=0.5,
            event_recall=0.50,
        )
        candidate = run_autoresearch.TargetMetrics(
            pr_auc=0.50,
            event_f1=0.70,
            false_alarms_per_day=0.20,
            mean_lead_bars=2.5,
            threshold=0.5,
            event_recall=0.75,
        )
        self.assertFalse(run_autoresearch._is_improvement(candidate, best, "event_f1", 0.10))

    def test_event_objective_beats_pr_auc_tie_break(self) -> None:
        best = run_autoresearch.TargetMetrics(
            pr_auc=0.45,
            event_f1=0.60,
            false_alarms_per_day=0.05,
            mean_lead_bars=2.0,
            threshold=0.5,
            event_recall=0.50,
        )
        candidate = run_autoresearch.TargetMetrics(
            pr_auc=0.60,
            event_f1=0.58,
            false_alarms_per_day=0.04,
            mean_lead_bars=3.0,
            threshold=0.5,
            event_recall=0.67,
        )
        self.assertFalse(run_autoresearch._is_improvement(candidate, best, "event_f1", 0.10))

    def test_event_recall_breaks_event_f1_tie(self) -> None:
        best = run_autoresearch.TargetMetrics(
            pr_auc=0.40,
            event_f1=0.60,
            false_alarms_per_day=0.05,
            mean_lead_bars=2.0,
            threshold=0.5,
            event_recall=0.50,
        )
        candidate = run_autoresearch.TargetMetrics(
            pr_auc=0.39,
            event_f1=0.60,
            false_alarms_per_day=0.05,
            mean_lead_bars=1.5,
            threshold=0.5,
            event_recall=0.67,
        )
        self.assertTrue(run_autoresearch._is_improvement(candidate, best, "event_f1", 0.10))


if __name__ == "__main__":
    unittest.main()
