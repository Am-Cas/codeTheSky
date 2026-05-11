from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from scipy.stats import theilslopes

ROOT = Path(__file__).resolve().parent.parent
ART = ROOT / "artifacts"
CSVS = [
    ROOT / "LED_sensor1" / "yyc_device_1.csv",
    ROOT / "LED_sensor1" / "yyc_device_2.csv",
]

USE = ["timestamp", "location_id", "metric", "numeric_value"]
DTYPE = {
    "location_id": "category",
    "metric": "category",
    "numeric_value": "object",
    "timestamp": "float64",
}


def excel_to_ts(s: pd.Series) -> pd.Series:
    return pd.to_datetime(s, unit="D", origin="1899-12-30", errors="coerce")


def get_circuit(loc: str) -> int | None:
    m = re.match(r"yyc-c(\d+)-r\d+", str(loc))
    return int(m.group(1)) if m else None


def main() -> None:
    frames = []
    for c in CSVS:
        frames.append(pd.read_csv(c, usecols=USE, dtype=DTYPE, low_memory=False))
    raw = pd.concat(frames, ignore_index=True)
    raw["ts"] = excel_to_ts(raw["timestamp"])
    raw["v"] = pd.to_numeric(raw["numeric_value"], errors="coerce")

    bv_metrics = ["boostVoltage", "boostVoltageMinimum", "boostVoltageMaximum"]
    bv = raw[raw["metric"].isin(bv_metrics)].copy()
    bv = bv[bv["v"] > 0].copy()
    bv_wide = (
        bv.pivot_table(index=["ts", "location_id"], columns="metric", values="v", aggfunc="first")
        .reset_index()
    )
    bv_wide.columns.name = None
    bv_wide["envelope"] = bv_wide["boostVoltageMaximum"] - bv_wide["boostVoltageMinimum"]
    bv_wide["date"] = bv_wide["ts"].dt.floor("D")

    # F1 level
    f1 = bv_wide.groupby("location_id")["boostVoltage"].apply(
        lambda s: float(stats.trim_mean(s.dropna().values, 0.05)) if s.notna().any() else np.nan
    )
    f1.name = "F1_level_mV"

    # F2a magnitude (non-zero envelope only)
    f2a = (
        bv_wide[bv_wide["envelope"] > 0]
        .groupby("location_id")["envelope"]
        .median()
        .rename("F2a_mag_mV")
    )

    # F2b prevalence
    env_per = bv_wide.groupby("location_id")["envelope"].agg(
        n_total="count",
        n_nonzero=lambda s: int((s > 0).sum()),
    )
    f2b = (env_per["n_nonzero"] / env_per["n_total"]).rename("F2b_prevalence")

    # F3/F4 from per-batch daily points with revised threshold
    daily_level = (
        bv_wide.groupby(["location_id", "date"])["boostVoltage"]
        .apply(lambda s: float(stats.trim_mean(s.dropna().values, 0.05)))
        .reset_index(name="bv_trim_mean")
    )
    daily_env = (
        bv_wide.groupby(["location_id", "date"])["envelope"]
        .apply(lambda s: float(s[s > 0].median()) if (s > 0).any() else np.nan)
        .reset_index(name="env_nonzero_median")
    )

    t_origin = daily_level["date"].min()
    daily_level["t_months"] = (daily_level["date"] - t_origin).dt.total_seconds() / (3600 * 24 * 30.44)
    daily_env["t_months"] = (daily_env["date"] - t_origin).dt.total_seconds() / (3600 * 24 * 30.44)

    MIN_WINDOWS = 15
    MIN_SPAN_MONTHS = 2.0

    f3_rows = []
    for fx, grp in daily_level.groupby("location_id"):
        grp = grp.dropna(subset=["bv_trim_mean", "t_months"]).sort_values("t_months")
        n = len(grp)
        span = float(grp["t_months"].max() - grp["t_months"].min()) if n > 0 else np.nan
        if n >= MIN_WINDOWS and span >= MIN_SPAN_MONTHS:
            slope, intercept, lo, hi = theilslopes(grp["bv_trim_mean"].values, grp["t_months"].values)
            f3_rows.append((fx, slope, lo, hi, hi - lo, n, span, True))
        else:
            f3_rows.append((fx, np.nan, np.nan, np.nan, np.nan, n, span, False))
    f3_df = pd.DataFrame(
        f3_rows,
        columns=[
            "location_id",
            "F3_slope_mV_per_month",
            "F3_slope_lo_mV_per_month",
            "F3_slope_hi_mV_per_month",
            "F3_ci_width",
            "n_windows",
            "span_months",
            "qualifies_temporal",
        ],
    ).set_index("location_id")

    f4_rows = []
    for fx, grp in daily_env.groupby("location_id"):
        grp = grp.dropna(subset=["env_nonzero_median", "t_months"]).sort_values("t_months")
        n = len(grp)
        span = float(grp["t_months"].max() - grp["t_months"].min()) if n > 0 else np.nan
        if n >= MIN_WINDOWS and span >= MIN_SPAN_MONTHS:
            slope, intercept, lo, hi = theilslopes(grp["env_nonzero_median"].values, grp["t_months"].values)
            f4_rows.append((fx, slope))
        else:
            f4_rows.append((fx, np.nan))
    f4_df = pd.DataFrame(f4_rows, columns=["location_id", "F4_envelope_slope_mV_per_month"]).set_index("location_id")

    # stale fraction
    stale_raw = raw[raw["metric"] == "stale"].copy()
    stale_raw["stale_num"] = pd.to_numeric(stale_raw["numeric_value"], errors="coerce")
    stale = stale_raw.groupby("location_id")["stale_num"].mean().rename("stale_frac")

    # ambient median (optional Arrhenius check)
    amb = raw[raw["metric"] == "ambientTemperature"].copy()
    amb["v"] = pd.to_numeric(amb["numeric_value"], errors="coerce")
    amb = amb[(amb["v"].notna()) & (amb["v"] != 65535)]
    amb_med = amb.groupby("location_id")["v"].median().rename("ambientTemperatureMedian")

    feats = pd.concat([f1, f2a, f2b, f3_df, f4_df, stale, amb_med], axis=1)
    feats["circuit"] = feats.index.map(get_circuit)
    feats = feats.reset_index().rename(columns={"index": "location_id"})

    out = ART / "phase2_features.csv"
    feats.to_csv(out, index=False)
    print(f"Wrote: {out}")
    print(feats.head(10).to_string(index=False))
    print("\nNull counts:")
    print(feats.isnull().sum().to_string())


if __name__ == "__main__":
    main()
