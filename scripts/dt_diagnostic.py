"""
dT diagnostic: correlate thermal rise (internalT - ambientT) with per-fixture
duty cycle (stale fraction) to determine whether the thermal coupling leg is
real or whether the sensors are colocated.
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path(__file__).resolve().parent.parent
CSVS = [ROOT / "LED_sensor1" / "yyc_device_1.csv", ROOT / "LED_sensor1" / "yyc_device_2.csv"]
ART = ROOT / "artifacts"

USE = ["timestamp", "device_id", "location_id", "metric", "numeric_value", "gateway_id"]
DTYPE = {
    "device_id": "category", "location_id": "category", "metric": "category",
    "gateway_id": "category", "numeric_value": "object", "timestamp": "float64",
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
    print("[load] raw data", flush=True)
    df = load()

    print("[compute] per-fixture dT (internalT - ambientT)", flush=True)
    # For each fixture, compute median of (internalT - ambientT).
    # Need to align time-adjacent readings.

    # Pivot: timestamp × (metric, fixture) → value
    pivot = df.pivot_table(
        index="ts", columns=["location_id", "metric"], values="v", aggfunc="first"
    )

    # Compute dT per fixture per timestamp
    fixture_ids = pivot.columns.get_level_values(0).unique()
    dT_per_fx = {}

    for fx in fixture_ids:
        fx_data = pivot[fx]
        if "internalTemperature" not in fx_data.columns or "ambientTemperature" not in fx_data.columns:
            continue

        amb = fx_data["ambientTemperature"]
        int_t = fx_data["internalTemperature"]

        # Drop sentinels (65535) and align
        valid = (amb != 65535) & (int_t != 65535) & amb.notna() & int_t.notna()
        if valid.sum() == 0:
            continue

        dT = int_t[valid] - amb[valid]
        dT_per_fx[fx] = {
            "n": int(valid.sum()),
            "median": float(dT.median()),
            "mean": float(dT.mean()),
            "p10": float(dT.quantile(0.10)),
            "p90": float(dT.quantile(0.90)),
            "std": float(dT.std()),
            "min": float(dT.min()),
            "max": float(dT.max()),
            "dT_series": dT.tolist()[:100],  # first 100 for inspection
        }

    print(f"  {len(dT_per_fx)} fixtures with valid dT", flush=True)

    # Load stale fraction
    stale = pd.read_csv(ART / "stale_fraction_per_fixture.csv", index_col=0)
    stale.index.name = None
    stale = stale[["stale_frac"]]

    # Merge
    result_df = pd.DataFrame({
        "dT_median": pd.Series({fx: v["median"] for fx, v in dT_per_fx.items()}),
        "dT_std": pd.Series({fx: v["std"] for fx, v in dT_per_fx.items()}),
        "dT_n": pd.Series({fx: v["n"] for fx, v in dT_per_fx.items()}),
    })
    result_df = result_df.join(stale)
    result_df = result_df.dropna()

    print(f"  {len(result_df)} fixtures with both dT and stale_frac", flush=True)

    # Correlation
    corr_pearson, pval_pearson = stats.pearsonr(result_df["dT_median"], result_df["stale_frac"])
    corr_spearman, pval_spearman = stats.spearmanr(result_df["dT_median"], result_df["stale_frac"])

    findings = {
        "correlation": {
            "pearson_r": float(corr_pearson),
            "pearson_pval": float(pval_pearson),
            "spearman_rho": float(corr_spearman),
            "spearman_pval": float(pval_spearman),
        },
        "dT_distribution": {
            "n_fixtures": int(len(result_df)),
            "dT_median_p01": float(result_df["dT_median"].quantile(0.01)),
            "dT_median_p10": float(result_df["dT_median"].quantile(0.10)),
            "dT_median_p50": float(result_df["dT_median"].median()),
            "dT_median_p90": float(result_df["dT_median"].quantile(0.90)),
            "dT_median_p99": float(result_df["dT_median"].quantile(0.99)),
            "dT_median_min": float(result_df["dT_median"].min()),
            "dT_median_max": float(result_df["dT_median"].max()),
            "dT_std_p50": float(result_df["dT_std"].median()),
        },
        "stale_frac_range": {
            "min": float(result_df["stale_frac"].min()),
            "p25": float(result_df["stale_frac"].quantile(0.25)),
            "p50": float(result_df["stale_frac"].median()),
            "p75": float(result_df["stale_frac"].quantile(0.75)),
            "max": float(result_df["stale_frac"].max()),
        },
    }

    print("[findings]", json.dumps(findings, indent=2), flush=True)

    # Detailed inspection of low-stale fixtures (high activity)
    low_stale = result_df[result_df["stale_frac"] < 0.40].copy()
    print(f"\n[low-stale fixtures (stale_frac < 0.4)]: {len(low_stale)} fixtures", flush=True)
    if len(low_stale) > 0:
        print("  Top 5 by activity (lowest stale_frac):", flush=True)
        top = low_stale.nsmallest(5, "stale_frac")[["dT_median", "dT_std", "stale_frac"]]
        print(top.to_string(), flush=True)
        print(f"\n  dT stats for active fixtures: p50={low_stale['dT_median'].median():.2f}, p90={low_stale['dT_median'].quantile(0.90):.2f}", flush=True)

    # Save results
    result_df.to_csv(ART / "dt_vs_stale.csv")
    (ART / "dt_diagnostic.json").write_text(json.dumps(findings, indent=2))

    # Plot
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Scatter: dT vs stale_frac
    axes[0].scatter(result_df["stale_frac"], result_df["dT_median"], alpha=0.6, s=50)
    axes[0].set_xlabel("Fixture stale fraction (duty-cycle proxy)")
    axes[0].set_ylabel("Median dT = internalT - ambientT (°C)")
    axes[0].set_title(f"dT vs Duty Cycle (r={corr_pearson:.3f}, p={pval_pearson:.4f})")
    axes[0].grid(True, alpha=0.3)
    axes[0].axhline(0, color="r", linestyle="--", alpha=0.5)

    # Histogram: dT_median distribution
    axes[1].hist(result_df["dT_median"], bins=30, alpha=0.7, edgecolor="black")
    axes[1].set_xlabel("Median dT (°C)")
    axes[1].set_ylabel("Count")
    axes[1].set_title("Per-fixture dT distribution")
    axes[1].axvline(0, color="r", linestyle="--", alpha=0.5, label="dT=0")
    axes[1].axvline(result_df["dT_median"].median(), color="g", linestyle="--", alpha=0.5, label=f"fleet p50={result_df['dT_median'].median():.2f}°C")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    plt.savefig(ART / "dt_diagnostic.png", dpi=100, bbox_inches="tight")
    print("\n[plot] dt_diagnostic.png", flush=True)

    # Interpretation
    interpretation = []
    if abs(corr_pearson) < 0.20:
        interpretation.append("**Weak or no correlation** between dT and stale_frac.")
    elif abs(corr_pearson) < 0.50:
        interpretation.append("**Moderate correlation** between dT and stale_frac.")
    else:
        interpretation.append("**Strong correlation** between dT and stale_frac.")

    if result_df["dT_median"].median() < 0.5:
        interpretation.append("Fleet dT median ~0°C → sensors likely colocated or fixtures mostly off during reporting.")
    else:
        interpretation.append("Fleet dT median >0.5°C → genuine thermal rise detectable.")

    if len(low_stale) > 0 and low_stale["dT_median"].median() > 1.0:
        interpretation.append("Active fixtures (low stale_frac) show dT >1°C → thermal coupling is present in powered state.")
    else:
        interpretation.append("Even active fixtures show dT ~0°C → sensors colocated or driver off/idle.")

    print("\n[interpretation]", flush=True)
    for line in interpretation:
        print(f"  {line}", flush=True)

    findings["interpretation"] = interpretation
    (ART / "dt_diagnostic.json").write_text(json.dumps(findings, indent=2))
    print("\n[done]", flush=True)


if __name__ == "__main__":
    main()
