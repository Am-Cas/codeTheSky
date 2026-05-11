from __future__ import annotations

import sys
from pathlib import Path
import unittest

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.indicator import compute_indicator_scores, resolve_confidence_tier


class IndicatorTests(unittest.TestCase):
    def _base_df(self) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "location_id": ["c1-a", "c1-b", "c2-a", "c2-b"],
                "circuit": [1, 1, 2, 2],
                "F1_level_mV": [100.0, 10.0, 100.0, 10.0],
                "F2a_mag_mV": [9.0, 1.0, 8.0, 2.0],
                "F3_slope_mV_per_month": [4.0, 1.0, 3.0, 0.0],
                "F4_envelope_slope_mV_per_month": [5.0, 0.0, 3.0, 1.0],
                "n_windows": [26, 22, 19, 30],
                "span_months": [8.0, 8.0, 8.0, 8.0],
                "F3_ci_width": [2.0, 10.0, 2.0, 1.0],
            }
        )

    def test_missing_required_columns_raises_value_error(self) -> None:
        with self.assertRaises(ValueError):
            compute_indicator_scores(pd.DataFrame({"location_id": ["x"]}))

    def test_within_circuit_ranks_are_independent(self) -> None:
        out = compute_indicator_scores(self._base_df(), within_circuit=True).set_index("fixture_id")
        self.assertEqual(out.loc["c1-a", "rank_F1"], 1.0)
        self.assertEqual(out.loc["c2-a", "rank_F1"], 1.0)
        self.assertEqual(out.loc["c1-b", "rank_F1"], 2.0)
        self.assertEqual(out.loc["c2-b", "rank_F1"], 2.0)

    def test_global_ranks_when_within_circuit_disabled(self) -> None:
        out = compute_indicator_scores(self._base_df(), within_circuit=False).set_index("fixture_id")
        self.assertEqual(out.loc["c1-a", "rank_F1"], 1.5)
        self.assertEqual(out.loc["c2-a", "rank_F1"], 1.5)
        self.assertEqual(out.loc["c1-b", "rank_F1"], 3.5)
        self.assertEqual(out.loc["c2-b", "rank_F1"], 3.5)

    def test_f2b_is_optional_and_nan_when_absent(self) -> None:
        out = compute_indicator_scores(self._base_df())
        self.assertIn("rank_F2b", out.columns)
        self.assertTrue(out["rank_F2b"].isna().all())

    def test_f2b_participates_in_driver_score_when_present(self) -> None:
        df = self._base_df().copy()
        df["F2b_prevalence"] = [0.9, 0.8, 0.2, 0.1]
        out = compute_indicator_scores(df).set_index("fixture_id")

        # With circuit-local ranking and 2 fixtures per circuit, top value stress is 1.0.
        self.assertAlmostEqual(out.loc["c1-a", "S_driver"], 1.0)
        self.assertAlmostEqual(out.loc["c2-b", "S_driver"], 0.0)

    def test_resolve_confidence_tier_uses_ci_bounds_when_width_missing(self) -> None:
        df = self._base_df().copy().drop(columns=["F3_ci_width"])
        df["F3_slope_lo_mV_per_month"] = [1.0, -9.0, -1.0, -0.2]
        df["F3_slope_hi_mV_per_month"] = [5.0, -1.0, 1.0, 0.2]

        tier = resolve_confidence_tier(df)
        self.assertEqual(tier.tolist(), ["High", "Medium", "Low", "Medium"])

    def test_scores_bounded_between_zero_and_one(self) -> None:
        out = compute_indicator_scores(self._base_df())
        numeric_cols = ["S_triage", "S_costress", "S_LED", "S_driver"]
        for col in numeric_cols:
            vals = out[col].to_numpy(dtype=float)
            self.assertTrue(np.nanmin(vals) >= 0.0)
            self.assertTrue(np.nanmax(vals) <= 1.0)


if __name__ == "__main__":
    unittest.main()
