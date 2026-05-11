"""
Phase 3 scoring: dual non-competing outputs.

Primary:
  - S_triage (arithmetic mean of mechanism sub-scores)
Diagnostic:
  - S_costress (geometric mean of mechanism sub-scores)
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
ART = ROOT / "artifacts"

FEATURE_COLUMNS = {
    "fixture_id": "location_id",
    "circuit": "circuit",
    "F1": "F1_level_mV",
    "F2a": "F2a_mag_mV",
    "F3": "F3_slope_mV_per_month",
    "F4": "F4_envelope_slope_mV_per_month",
    "n_windows": "n_windows",
    "span_months": "span_months",
}


def _stress_from_rank_desc(series: pd.Series) -> tuple[pd.Series, pd.Series]:
    """Return descending rank and [0,1] stress where higher values are worse."""
    rank = series.rank(method="average", ascending=False)
    count = rank.notna().sum()
    if count <= 1:
        stress = pd.Series(np.where(rank.notna(), 1.0, np.nan), index=rank.index, dtype="float64")
    else:
        stress = 1.0 - (rank - 1.0) / (count - 1.0)
    return rank, stress


def _resolve_confidence_tier(df: pd.DataFrame) -> pd.Series:
    """
    Confidence tiers:
      - High: n >= 25 AND F3 CI width < 2 * |slope|
      - Medium: 20 <= n < 25 OR F3 CI width >= 2 * |slope|
      - Low: n < 20
    """
    n = pd.to_numeric(df[FEATURE_COLUMNS["n_windows"]], errors="coerce")
    slope = pd.to_numeric(df[FEATURE_COLUMNS["F3"]], errors="coerce").abs()

    has_ci = "F3_ci_width" in df.columns
    if has_ci:
        ci_width = pd.to_numeric(df["F3_ci_width"], errors="coerce")
    elif {"F3_slope_lo_mV_per_month", "F3_slope_hi_mV_per_month"}.issubset(df.columns):
        lo = pd.to_numeric(df["F3_slope_lo_mV_per_month"], errors="coerce")
        hi = pd.to_numeric(df["F3_slope_hi_mV_per_month"], errors="coerce")
        ci_width = (hi - lo).abs()
        has_ci = True
    else:
        ci_width = pd.Series(np.nan, index=df.index, dtype="float64")

    low = n < 20
    high = (n >= 25) & has_ci & (ci_width < (2.0 * slope))
    medium = (~low) & (((n >= 20) & (n < 25)) | (has_ci & (ci_width >= (2.0 * slope))))

    tier = pd.Series("Low", index=df.index, dtype="object")
    tier.loc[medium] = "Medium"
    tier.loc[high] = "High"
    return tier


def compute_phase3_scores(feature_df: pd.DataFrame) -> pd.DataFrame:
    required = [FEATURE_COLUMNS["fixture_id"], FEATURE_COLUMNS["circuit"], FEATURE_COLUMNS["F1"], FEATURE_COLUMNS["F2a"], FEATURE_COLUMNS["F3"], FEATURE_COLUMNS["F4"], FEATURE_COLUMNS["n_windows"], FEATURE_COLUMNS["span_months"]]
    missing = [col for col in required if col not in feature_df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    df = feature_df.copy()
    df["fixture_id"] = df[FEATURE_COLUMNS["fixture_id"]].astype(str)

    rank_cols = {}
    stress_cols = {}
    for feature_key in ("F1", "F2a", "F3", "F4"):
        rank_col = f"rank_{feature_key}"
        stress_col = f"stress_{feature_key}"
        rank_cols[feature_key] = rank_col
        stress_cols[feature_key] = stress_col
        df[rank_col] = np.nan
        df[stress_col] = np.nan

    for _, idx in df.groupby(FEATURE_COLUMNS["circuit"]).groups.items():
        for feature_key in ("F1", "F2a", "F3", "F4"):
            series = pd.to_numeric(df.loc[idx, FEATURE_COLUMNS[feature_key]], errors="coerce")
            rank, stress = _stress_from_rank_desc(series)
            df.loc[idx, rank_cols[feature_key]] = rank
            df.loc[idx, stress_cols[feature_key]] = stress

    df["S_LED"] = df[[stress_cols["F1"], stress_cols["F3"]]].mean(axis=1, skipna=True)
    df["S_driver"] = df[[stress_cols["F2a"], stress_cols["F4"]]].mean(axis=1, skipna=True)

    # Primary triage score: mechanism-disjunctive aggregation.
    df["S_triage"] = (df["S_LED"] + df["S_driver"]) / 2.0

    # Diagnostic co-stress flag: broad vs narrow stress profile.
    df["S_costress"] = np.sqrt(df["S_LED"].clip(lower=0) * df["S_driver"].clip(lower=0))

    df["confidence_tier"] = _resolve_confidence_tier(df)

    out = df[
        [
            "fixture_id",
            FEATURE_COLUMNS["circuit"],
            "S_triage",
            "S_costress",
            "S_LED",
            "S_driver",
            rank_cols["F1"],
            rank_cols["F2a"],
            rank_cols["F3"],
            rank_cols["F4"],
            "confidence_tier",
            FEATURE_COLUMNS["n_windows"],
            FEATURE_COLUMNS["span_months"],
        ]
    ].copy()
    out = out.sort_values("S_triage", ascending=False).reset_index(drop=True)
    return out


def main() -> None:
    feature_path = ART / "phase2_features.csv"
    out_path = ART / "phase3_scores.csv"

    features = pd.read_csv(feature_path)
    scored = compute_phase3_scores(features)
    scored.to_csv(out_path, index=False)

    print(f"Wrote: {out_path}")
    print(scored.head(10).to_string(index=False))


if __name__ == "__main__":
    main()
