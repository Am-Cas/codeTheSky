"""
Phase 1 follow-up: quantify the data-quality issues that the inventory flagged,
so the stop-and-ask write-up has numbers behind it.

Focus on the signals that survive (boostVoltage, internal/ambient temps,
stale) and the ones that are dead-on-arrival (led*Temperature, heater*,
magnetic*, uint16 sentinels in ambient/internal).
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
ART = ROOT / "artifacts"
CSVS = [ROOT / "LED_sensor1" / "yyc_device_1.csv", ROOT / "LED_sensor1" / "yyc_device_2.csv"]

USE = [
    "timestamp", "device_id", "location_id", "metric", "numeric_value",
    "string_value", "gateway_id",
]

DTYPE = {
    "device_id": "category", "location_id": "category", "metric": "category",
    "gateway_id": "category", "numeric_value": "object", "string_value": "object",
    "timestamp": "float64",
}


def excel_to_ts(s):
    return pd.to_datetime(s, unit="D", origin="1899-12-30", errors="coerce")


def load():
    frames = []
    for c in CSVS:
        df = pd.read_csv(c, usecols=USE, dtype=DTYPE, low_memory=False)
        frames.append(df)
    df = pd.concat(frames, ignore_index=True)
    df["ts"] = excel_to_ts(df["timestamp"])
    df["v"] = pd.to_numeric(df["numeric_value"], errors="coerce")
    return df


def main():
    df = load()
    out = {}

    # ---- sentinel rates on temperature signals ----
    sentinels = {
        "ambientTemperature": 65535,
        "ambientTemperatureMaximum": 65535,
        "internalTemperature": 65535,
        "internalTemperatureMaximum": 65535,
        "led01Temperature": 25,
        "led02Temperature": 25,
        "heater01Temperature": -55,
        "heater02Temperature": -55,
    }
    sent_report = {}
    for m, val in sentinels.items():
        s = df.loc[df["metric"] == m, "v"]
        if len(s) == 0:
            sent_report[m] = {"n": 0, "sentinel_frac": None}
            continue
        sent_report[m] = {
            "n": int(len(s)),
            "sentinel_value": val,
            "sentinel_frac": float((s == val).mean()),
            "distinct_values": int(s.nunique()),
            "p01": float(s.quantile(0.01)),
            "p50": float(s.median()),
            "p99": float(s.quantile(0.99)),
        }
    out["sentinel_audit"] = sent_report
    print("[sentinels]", json.dumps(sent_report, indent=2))

    # ---- distribution of usable signals (filter out sentinels first) ----
    def clean(metric, drop_vals=()):
        s = df.loc[df["metric"] == metric, "v"]
        for v in drop_vals:
            s = s[s != v]
        return s

    usable = {}
    for m, drops in [
        ("boostVoltage", (0,)),
        ("boostVoltageMinimum", (0,)),
        ("boostVoltageMaximum", (0,)),
        ("ambientTemperature", (65535,)),
        ("internalTemperature", (65535,)),
        ("humidity", ()),
        ("pressure", (0,)),
    ]:
        s = clean(m, drops)
        if len(s) == 0:
            continue
        usable[m] = {
            "n": int(len(s)),
            "p01": float(s.quantile(0.01)), "p10": float(s.quantile(0.10)),
            "p50": float(s.median()), "p90": float(s.quantile(0.90)),
            "p99": float(s.quantile(0.99)),
            "min": float(s.min()), "max": float(s.max()),
        }
    out["usable_distributions"] = usable
    print("[usable distributions]", json.dumps(usable, indent=2))

    # ---- per-fixture stale fraction ----
    stale = df[df["metric"] == "stale"].groupby("location_id", observed=True)["v"].agg(
        n="size", stale_frac="mean",
    )
    out["stale_fraction"] = {
        "fleet_mean": float(stale["stale_frac"].mean()),
        "fleet_median": float(stale["stale_frac"].median()),
        "n_fully_stale": int((stale["stale_frac"] == 1.0).sum()),
        "n_never_stale": int((stale["stale_frac"] == 0.0).sum()),
        "p10": float(stale["stale_frac"].quantile(0.10)),
        "p90": float(stale["stale_frac"].quantile(0.90)),
    }
    stale.to_csv(ART / "stale_fraction_per_fixture.csv")
    print("[stale per fixture]", json.dumps(out["stale_fraction"], indent=2))

    # ---- per-fixture boostVoltage coverage ----
    bv = df[df["metric"] == "boostVoltage"].copy()
    bv = bv[bv["v"] > 0]  # drop sentinel zeros
    bv_per = bv.groupby("location_id", observed=True)["v"].agg(
        n="size", p10=lambda s: s.quantile(0.10), p50="median", p90=lambda s: s.quantile(0.90),
        vmin="min", vmax="max",
    )
    bv_per.to_csv(ART / "boostvoltage_per_fixture.csv")
    out["boostvoltage_per_fixture"] = {
        "n_fixtures_with_any_bv": int((bv_per["n"] > 0).sum()),
        "n_fixtures_with_ge_50_samples": int((bv_per["n"] >= 50).sum()),
        "fleet_p50_of_median": float(bv_per["p50"].median()),
        "fleet_p50_of_max": float(bv_per["vmax"].median()),
        "fleet_p90_of_max": float(bv_per["vmax"].quantile(0.90)),
        "fleet_max_of_max": float(bv_per["vmax"].max()),
    }
    print("[boostV per fixture]", json.dumps(out["boostvoltage_per_fixture"], indent=2))

    # ---- internal vs ambient differential coverage ----
    pair = df[df["metric"].isin(["ambientTemperature", "internalTemperature"])].copy()
    pair = pair[pair["v"] != 65535]
    by_fx = pair.groupby(["location_id", "metric"], observed=True)["v"].agg(["count", "median"]).unstack()
    by_fx.columns = ["n_amb", "n_int", "med_amb", "med_int"]
    by_fx["dT_med"] = by_fx["med_int"] - by_fx["med_amb"]
    by_fx.to_csv(ART / "thermal_rise_per_fixture.csv")
    out["thermal_rise"] = {
        "n_fixtures_with_both": int(by_fx[["n_amb", "n_int"]].dropna().shape[0]),
        "fleet_p10_dT": float(by_fx["dT_med"].quantile(0.10)),
        "fleet_p50_dT": float(by_fx["dT_med"].median()),
        "fleet_p90_dT": float(by_fx["dT_med"].quantile(0.90)),
    }
    print("[thermal rise]", json.dumps(out["thermal_rise"], indent=2))

    (ART / "phase1_quality.json").write_text(json.dumps(out, indent=2))
    print("[done]")


if __name__ == "__main__":
    main()
