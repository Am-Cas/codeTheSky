from __future__ import annotations

import sys
from pathlib import Path
import unittest

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from scripts.phase3_scoring import compute_phase3_scores


class Phase3ScoringTests(unittest.TestCase):
    def _base_df(self) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "location_id": ["fx-a", "fx-b", "fx-c"],
                "circuit": [1, 1, 1],
                "F1_level_mV": [100.0, 80.0, 60.0],
                "F2a_mag_mV": [10.0, 50.0, 90.0],
                "F3_slope_mV_per_month": [5.0, -5.0, 0.0],
                "F4_envelope_slope_mV_per_month": [2.0, 1.0, 0.0],
                "n_windows": [26, 22, 18],
                "span_months": [8.0, 8.0, 8.0],
                "F3_ci_width": [8.0, 12.0, 0.5],
            }
        )

    def test_outputs_required_columns(self) -> None:
        out = compute_phase3_scores(self._base_df())
        expected_cols = {
            "fixture_id",
            "circuit",
            "S_triage",
            "S_costress",
            "S_LED",
            "S_driver",
            "rank_F1",
            "rank_F2a",
            "rank_F3",
            "rank_F4",
            "confidence_tier",
            "n_windows",
            "span_months",
        }
        self.assertEqual(set(out.columns), expected_cols)

    def test_triage_does_not_penalize_single_mechanism_extreme(self) -> None:
        out = compute_phase3_scores(self._base_df()).set_index("fixture_id")
        # fx-a: high LED, low driver
        # fx-c: high driver, moderate LED
        self.assertGreater(out.loc["fx-a", "S_triage"], 0.45)
        self.assertGreater(out.loc["fx-c", "S_triage"], 0.30)
        # Geometric diagnostic stays at or below triage and drops on imbalance.
        self.assertLessEqual(out.loc["fx-a", "S_costress"], out.loc["fx-a", "S_triage"])
        self.assertLess(out.loc["fx-a", "S_costress"], out.loc["fx-a", "S_LED"])

    def test_confidence_tiers_match_requested_rules(self) -> None:
        out = compute_phase3_scores(self._base_df()).set_index("fixture_id")
        self.assertEqual(out.loc["fx-a", "confidence_tier"], "High")
        self.assertEqual(out.loc["fx-b", "confidence_tier"], "Medium")
        self.assertEqual(out.loc["fx-c", "confidence_tier"], "Low")


if __name__ == "__main__":
    unittest.main()
