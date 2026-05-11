"""
Phase 1 — Inventory pass.

Goal: understand the raw stream before any pivoting or modeling.
Outputs land in artifacts/ as parquet + small JSON summaries.

We deliberately do NOT pivot to wide here. Pivoting 2M long-format rows
across ~92 metrics x 127 fixtures blindly is wasteful. First we want to
know what we're pivoting.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
CSVS = [ROOT / "yyc_device_1.csv", ROOT / "yyc_device_2.csv"]
ART = ROOT / "artifacts"
ART.mkdir(exist_ok=True)


def excel_serial_to_ts(s: pd.Series) -> pd.Series:
    # Excel serial dates: days since 1899-12-30 (Excel's epoch quirk).
    return pd.to_datetime(s, unit="D", origin="1899-12-30", errors="coerce")


def load_raw(path: Path) -> pd.DataFrame:
    # Read as strings first so we can audit "null" literals vs real NaN.
    dtype = {
        "timestamp": "float64",
        "device_id": "category",
        "location_id": "category",
        "metric": "category",
        "numeric_value": "object",
        "string_value": "object",
        "gateway_id": "category",
        "asset_type": "category",
        "tenant_id": "category",
        "source_system": "category",
    }
    use = [
        "timestamp",
        "device_id",
        "location_id",
        "metric",
        "numeric_value",
        "string_value",
        "asset_type",
        "gateway_id",
        "inserted_at",
    ]
    df = pd.read_csv(path, usecols=use, dtype=dtype, low_memory=False)
    df["ts"] = excel_serial_to_ts(df["timestamp"])
    df["inserted_ts"] = excel_serial_to_ts(pd.to_numeric(df["inserted_at"], errors="coerce"))
    # numeric_value comes in as object because of "null" strings.
    df["numeric_value"] = pd.to_numeric(df["numeric_value"], errors="coerce")
    # string_value: replace literal 'null' with NaN
    df["string_value"] = df["string_value"].where(df["string_value"].ne("null"), other=np.nan)
    return df


def main() -> None:
    summary: dict = {}
    frames = []
    for csv in CSVS:
        print(f"[load] {csv.name}", flush=True)
        df = load_raw(csv)
        gw = csv.stem
        summary[gw] = {
            "rows": int(len(df)),
            "time_min": str(df["ts"].min()),
            "time_max": str(df["ts"].max()),
            "n_fixtures": int(df["location_id"].nunique()),
            "n_metrics": int(df["metric"].nunique()),
            "n_devices": int(df["device_id"].nunique()),
            "gateway_ids": df["gateway_id"].dropna().unique().tolist(),
        }
        frames.append(df)
        print(f"  rows={len(df):,}  fixtures={df['location_id'].nunique()}  metrics={df['metric'].nunique()}", flush=True)

    raw = pd.concat(frames, ignore_index=True)
    del frames
    print(f"[combined] rows={len(raw):,}", flush=True)

    # --- metric inventory ---
    # For each metric: how many events, how many distinct fixtures, whether values are numeric
    # or categorical, sampling cadence (median delta between observations on a fixture).
    print("[inventory] metric-level summary", flush=True)
    g = raw.groupby("metric", observed=True)
    inv = pd.DataFrame({
        "n_events": g.size(),
        "n_fixtures": g["location_id"].nunique(),
        "n_numeric": g["numeric_value"].apply(lambda s: int(s.notna().sum())),
        "n_string": g["string_value"].apply(lambda s: int(s.notna().sum())),
        "numeric_min": g["numeric_value"].min(),
        "numeric_p50": g["numeric_value"].median(),
        "numeric_max": g["numeric_value"].max(),
    })
    inv = inv.sort_values("n_events", ascending=False)
    inv.to_csv(ART / "metric_inventory.csv")
    print(f"  unique metrics: {len(inv)}", flush=True)
    print(inv.head(25).to_string(), flush=True)
    print("...", flush=True)
    print(inv.tail(15).to_string(), flush=True)

    # --- per-fixture x per-metric cadence (median delta seconds) ---
    print("[cadence] per metric across fleet", flush=True)
    raw_sorted = raw.sort_values(["location_id", "metric", "ts"])
    raw_sorted["dt_s"] = raw_sorted.groupby(["location_id", "metric"], observed=True)["ts"].diff().dt.total_seconds()
    cadence = raw_sorted.groupby("metric", observed=True)["dt_s"].agg(
        cadence_p50_s="median",
        cadence_p10_s=lambda s: s.quantile(0.10),
        cadence_p90_s=lambda s: s.quantile(0.90),
    )
    cadence.to_csv(ART / "metric_cadence.csv")
    print(cadence.sort_values("cadence_p50_s").head(20).to_string(), flush=True)

    # --- fixture roster ---
    fx = raw.groupby("location_id", observed=True).agg(
        gateway=("gateway_id", lambda s: s.dropna().mode().iat[0] if s.notna().any() else None),
        n_events=("metric", "size"),
        n_metrics=("metric", "nunique"),
        first_seen=("ts", "min"),
        last_seen=("ts", "max"),
    ).sort_values("gateway")
    fx.to_csv(ART / "fixture_roster.csv")
    print(f"[fixtures] total={len(fx)}", flush=True)
    print(fx.groupby("gateway").size().to_string(), flush=True)

    # --- legacy-duplicate audit ---
    legacy_pairs = [
        ("ambientTemperature", "ambientTemperatureLegacyInDegC"),
        ("pressure", "pressureLegacyInMbar"),
    ]
    legacy_report = []
    metrics_present = set(inv.index)
    for new, old in legacy_pairs:
        legacy_report.append({
            "new_metric": new,
            "new_present": new in metrics_present,
            "new_events": int(inv.loc[new, "n_events"]) if new in metrics_present else 0,
            "legacy_metric": old,
            "legacy_present": old in metrics_present,
            "legacy_events": int(inv.loc[old, "n_events"]) if old in metrics_present else 0,
        })
    (ART / "legacy_audit.json").write_text(json.dumps(legacy_report, indent=2))
    print("[legacy]", json.dumps(legacy_report, indent=2), flush=True)

    # --- stale / liveness signal ---
    stale_metrics = [m for m in metrics_present if "stale" in m.lower() or "lastSeen" in m or "responseRate" in m or "failsafe" in m.lower()]
    print(f"[liveness candidates] {stale_metrics}", flush=True)
    liveness = raw[raw["metric"].isin(stale_metrics)].groupby(["location_id", "metric"], observed=True)["numeric_value"].agg(["count", "mean", "min", "max"])
    liveness.to_csv(ART / "liveness_per_fixture.csv")

    # --- save raw to parquet for fast reload in later phases ---
    print("[persist] writing combined parquet", flush=True)
    raw[["ts", "location_id", "device_id", "gateway_id", "metric", "numeric_value", "string_value"]].to_parquet(
        ART / "events.parquet", index=False
    )

    (ART / "phase1_summary.json").write_text(json.dumps(summary, indent=2, default=str))
    print("[done] artifacts in", ART)


if __name__ == "__main__":
    sys.exit(main())
