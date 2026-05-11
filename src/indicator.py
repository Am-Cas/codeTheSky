"""
Indicator module for per-fixture degradation scoring.

Outputs:
  - S_triage: arithmetic mean of mechanism sub-scores (primary)
  - S_costress: geometric mean of mechanism sub-scores (diagnostic)
  - S_LED, S_driver
  - Feature ranks and confidence tiers
"""
from __future__ import annotations

import numpy as np
import pandas as pd

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
    rank = series.rank(method="average", ascending=False)
    count = rank.notna().sum()
    if count <= 1:
        stress = pd.Series(np.where(rank.notna(), 1.0, np.nan), index=rank.index, dtype="float64")
    else:
        stress = 1.0 - (rank - 1.0) / (count - 1.0)
    return rank, stress


def resolve_confidence_tier(df: pd.DataFrame) -> pd.Series:
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


def compute_indicator_scores(
    feature_df: pd.DataFrame,
    *,
    within_circuit: bool = True,
    include_f2b_in_driver: bool = True,
    min_non_stale_windows: int = 20,
    max_stale_frac: float | None = 0.50,
) -> pd.DataFrame:
    required = [
        FEATURE_COLUMNS["fixture_id"],
        FEATURE_COLUMNS["circuit"],
        FEATURE_COLUMNS["F1"],
        FEATURE_COLUMNS["F2a"],
        FEATURE_COLUMNS["F3"],
        FEATURE_COLUMNS["F4"],
        FEATURE_COLUMNS["n_windows"],
        FEATURE_COLUMNS["span_months"],
    ]
    missing = [col for col in required if col not in feature_df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    df = feature_df.copy()
    df["fixture_id"] = df[FEATURE_COLUMNS["fixture_id"]].astype(str)

    # Optional F2b support for downstream validation displays.
    if "F2b_prevalence" not in df.columns:
        df["F2b_prevalence"] = np.nan

    rank_cols: dict[str, str] = {}
    stress_cols: dict[str, str] = {}
    for feature_key in ("F1", "F2a", "F2b", "F3", "F4"):
        source_col = "F2b_prevalence" if feature_key == "F2b" else FEATURE_COLUMNS.get(feature_key, "")
        if source_col and source_col in df.columns:
            rank_col = f"rank_{feature_key}"
            stress_col = f"stress_{feature_key}"
            rank_cols[feature_key] = rank_col
            stress_cols[feature_key] = stress_col
            df[rank_col] = np.nan
            df[stress_col] = np.nan

    if within_circuit:
        group_iter = df.groupby(FEATURE_COLUMNS["circuit"]).groups.items()
    else:
        group_iter = [(None, df.index)]

    for _, idx in group_iter:
        for feature_key, rank_col in rank_cols.items():
            source_col = "F2b_prevalence" if feature_key == "F2b" else FEATURE_COLUMNS.get(feature_key, "")
            series = pd.to_numeric(df.loc[idx, source_col], errors="coerce")
            rank, stress = _stress_from_rank_desc(series)
            df.loc[idx, rank_col] = rank
            df.loc[idx, stress_cols[feature_key]] = stress

    df["S_LED"] = df[[stress_cols["F1"], stress_cols["F3"]]].mean(axis=1, skipna=True)

    # Driver side optionally includes F2b (which may proxy stale/liveness).
    driver_stresses = [stress_cols["F2a"], stress_cols["F4"]]
    if include_f2b_in_driver and "F2b" in stress_cols:
        driver_stresses.append(stress_cols["F2b"])
    df["S_driver"] = df[driver_stresses].mean(axis=1, skipna=True)

    df["S_triage"] = (df["S_LED"] + df["S_driver"]) / 2.0
    df["S_costress"] = np.sqrt(df["S_LED"].clip(lower=0) * df["S_driver"].clip(lower=0))
    df["confidence_tier"] = resolve_confidence_tier(df)
    n_windows = pd.to_numeric(df[FEATURE_COLUMNS["n_windows"]], errors="coerce")
    stale_frac = pd.to_numeric(df.get("stale_frac", np.nan), errors="coerce")
    non_stale_windows = n_windows * (1.0 - stale_frac)
    df["non_stale_windows"] = non_stale_windows

    qualifies = non_stale_windows >= float(min_non_stale_windows)
    if max_stale_frac is not None:
        qualifies = qualifies | (stale_frac <= float(max_stale_frac))
    df["qualifies_primary_score"] = qualifies.fillna(False)
    df["score_status"] = np.where(
        df["qualifies_primary_score"],
        "Scored",
        "Score suppressed - insufficient data",
    )

    suppressed_mask = ~df["qualifies_primary_score"]
    df.loc[suppressed_mask, ["S_triage", "S_costress"]] = np.nan

    out_cols = [
        "fixture_id",
        FEATURE_COLUMNS["circuit"],
        "S_triage",
        "S_costress",
        "S_LED",
        "S_driver",
        "rank_F1",
        "rank_F2a",
        "rank_F2b",
        "rank_F3",
        "rank_F4",
        "confidence_tier",
        "qualifies_primary_score",
        "score_status",
        "non_stale_windows",
        FEATURE_COLUMNS["n_windows"],
        FEATURE_COLUMNS["span_months"],
    ]
    out = df.reindex(columns=out_cols).copy()
    out = out.sort_values("S_triage", ascending=False).reset_index(drop=True)
    return out
