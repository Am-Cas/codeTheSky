"""
Pre-Phase 3 checks:
  A. Gateway signal comparison (not just sampling differences)
  B. F2b (non-zero envelope prevalence) vs F1 (level) correlation
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path(__file__).resolve().parent.parent
ART  = ROOT / "artifacts"
CSVS = [
    ROOT / "LED_sensor1" / "yyc_device_1.csv",
    ROOT / "LED_sensor1" / "yyc_device_2.csv",
]
USE   = ["timestamp", "location_id", "metric", "numeric_value", "gateway_id"]
DTYPE = {
    "location_id": "category", "metric": "category",
    "gateway_id":  "category", "numeric_value": "object",
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
    df["v"]  = pd.to_numeric(df["numeric_value"], errors="coerce")
    return df

def get_circuit(loc: str) -> int | None:
    m = re.match(r"yyc-c(\d+)-r\d+", str(loc))
    return int(m.group(1)) if m else None

# ──────────────────────────────────────────────────────────────────────────
def main():
    print("[load]", flush=True)
    df = load()

    # Circuit label per row (1 or 2)
    df["circuit"] = df["location_id"].astype(str).map(get_circuit).astype("Int64")

    # ── A: Gateway / circuit signal comparison ─────────────────────────────
    print("\n=== A: Circuit signal comparison ===", flush=True)

    bv_metrics = ["boostVoltage", "boostVoltageMinimum", "boostVoltageMaximum"]
    bv = df[df["metric"].isin(bv_metrics)].copy()
    bv = bv[bv["v"] > 0]

    # Per-circuit boostVoltage distribution
    for circ in [1, 2]:
        s = bv.loc[(bv["circuit"] == circ) & (bv["metric"] == "boostVoltage"), "v"]
        print(f"  Circuit {circ}  n={len(s):,}  p10={s.quantile(0.10):.0f}  "
              f"p50={s.median():.0f}  p90={s.quantile(0.90):.0f}  "
              f"std={s.std():.1f}  max={s.max():.0f} mV", flush=True)

    c1_bv = bv.loc[(bv["circuit"] == 1) & (bv["metric"] == "boostVoltage"), "v"]
    c2_bv = bv.loc[(bv["circuit"] == 2) & (bv["metric"] == "boostVoltage"), "v"]
    u_bv, p_bv = stats.mannwhitneyu(c1_bv, c2_bv, alternative="two-sided")
    print(f"  MWU boostVoltage C1 vs C2: U={u_bv:.0f}, p={p_bv:.4f}", flush=True)

    # Envelope comparison
    bv_wide = bv.pivot_table(
        index=["ts", "location_id", "circuit"],
        columns="metric", values="v", aggfunc="first",
    ).reset_index()
    bv_wide.columns.name = None
    bv_wide["envelope"] = bv_wide["boostVoltageMaximum"] - bv_wide["boostVoltageMinimum"]
    bv_wide_pos = bv_wide[bv_wide["envelope"] > 0]
    for circ in [1, 2]:
        s = bv_wide_pos.loc[bv_wide_pos["circuit"] == circ, "envelope"]
        print(f"  Circuit {circ}  non-zero envelope  n={len(s):,}  "
              f"p50={s.median():.0f}  p90={s.quantile(0.90):.0f}  max={s.max():.0f} mV",
              flush=True)

    # Per-fixture F1 by circuit
    feat = pd.read_csv(ART / "phase2_features.csv", index_col=0)
    feat["circuit"] = feat.index.map(get_circuit)

    f1_c1 = feat.loc[feat["circuit"] == 1, "F1_level_mV"]
    f1_c2 = feat.loc[feat["circuit"] == 2, "F1_level_mV"]
    u_f1, p_f1 = stats.mannwhitneyu(f1_c1.dropna(), f1_c2.dropna(), alternative="two-sided")
    print(f"\n  Per-fixture F1 by circuit:")
    print(f"    Circuit 1:  p10={f1_c1.quantile(0.10):.0f}  p50={f1_c1.median():.0f}  "
          f"p90={f1_c1.quantile(0.90):.0f}  std={f1_c1.std():.1f} mV")
    print(f"    Circuit 2:  p10={f1_c2.quantile(0.10):.0f}  p50={f1_c2.median():.0f}  "
          f"p90={f1_c2.quantile(0.90):.0f}  std={f1_c2.std():.1f} mV")
    print(f"    MWU on per-fixture F1: U={u_f1:.0f}, p={p_f1:.4f}")
    if p_f1 < 0.05:
        d_cohens = (f1_c1.mean() - f1_c2.mean()) / np.sqrt(
            (f1_c1.std()**2 + f1_c2.std()**2) / 2
        )
        print(f"    -> CIRCUITS DIFFER on F1 signal (Cohen's d = {d_cohens:.3f})")
        print(f"       Diff in medians: {f1_c2.median() - f1_c1.median():.0f} mV")
    else:
        print(f"    -> Circuits do NOT differ significantly on F1")

    # Span by circuit (sampling artifact check)
    print(f"\n  n_windows / span_months by circuit:")
    for circ in [1, 2]:
        sub = feat[feat["circuit"] == circ]
        print(f"    Circuit {circ}:  n_windows p50={sub['n_windows'].median():.0f}  "
              f"span p50={sub['span_months'].median():.2f} mo")

    # ── B: F2b vs F1 correlation ───────────────────────────────────────────
    print("\n=== B: F2b (non-zero envelope prevalence) vs F1 (level) ===", flush=True)

    # Compute F2b and F2a per fixture (not in phase2_features yet)
    env_per = bv_wide.groupby("location_id")["envelope"].agg(
        n_total="count",
        n_nonzero=lambda s: (s > 0).sum(),
    )
    env_per["F2b_prevalence"] = env_per["n_nonzero"] / env_per["n_total"]
    env_per["F2a_mag_mV"]     = bv_wide[bv_wide["envelope"] > 0].groupby("location_id")["envelope"].median()
    env_per["F2a_mag_mV"]     = env_per["F2a_mag_mV"].fillna(0)

    feat = feat.join(env_per[["F2b_prevalence", "F2a_mag_mV"]])

    rho_f2b_f1, p_f2b_f1 = stats.spearmanr(
        feat["F1_level_mV"].dropna(),
        feat.loc[feat["F1_level_mV"].notna(), "F2b_prevalence"],
    )
    rho_f2a_f1, p_f2a_f1 = stats.spearmanr(
        feat["F1_level_mV"].dropna(),
        feat.loc[feat["F1_level_mV"].notna(), "F2a_mag_mV"],
    )

    print(f"  F2b (prevalence) vs F1:  Spearman rho={rho_f2b_f1:.3f}, p={p_f2b_f1:.4f}")
    print(f"  F2a (magnitude)  vs F1:  Spearman rho={rho_f2a_f1:.3f}, p={p_f2a_f1:.4f}")

    # F2b distribution
    print(f"\n  F2b fleet distribution:")
    print(f"    p10={feat['F2b_prevalence'].quantile(0.10):.3f}  "
          f"p50={feat['F2b_prevalence'].median():.3f}  "
          f"p90={feat['F2b_prevalence'].quantile(0.90):.3f}  "
          f"max={feat['F2b_prevalence'].max():.3f}")
    print(f"    Fixtures with F2b=0: {(feat['F2b_prevalence']==0).sum()}")
    print(f"    Fixtures with F2b>0.5: {(feat['F2b_prevalence']>0.5).sum()}")

    if abs(rho_f2b_f1) < 0.15 and p_f2b_f1 > 0.10:
        print(f"\n  -> F2b is FLAT relative to F1: zeros are likely firmware artefacts.")
        print(f"     Recommend: demote F2b to data-quality flag, keep F2a only.")
    elif rho_f2b_f1 > 0.20 and p_f2b_f1 < 0.05:
        print(f"\n  -> F2b CO-VARIES with F1: instability and level co-occur.")
        print(f"     Both F2a and F2b are physics-grounded signals.")
    else:
        print(f"\n  -> Weak relationship — interpret cautiously.")

    # Plots
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    # A: Per-fixture F1 by circuit
    for circ, color in [(1, "steelblue"), (2, "tomato")]:
        sub = feat[feat["circuit"] == circ]
        axes[0].hist(sub["F1_level_mV"], bins=20, alpha=0.55, color=color,
                     label=f"Circuit {circ} (n={len(sub)})", edgecolor="none")
    axes[0].set_xlabel("F1 — trimmed mean boostVoltage (mV)")
    axes[0].set_ylabel("Fixture count")
    axes[0].set_title(f"F1 by circuit (MWU p={p_f1:.4f})")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3, axis="y")

    # B: F2b vs F1 scatter
    axes[1].scatter(feat["F1_level_mV"], feat["F2b_prevalence"], alpha=0.5, s=45)
    axes[1].set_xlabel("F1 — level (mV)")
    axes[1].set_ylabel("F2b — non-zero envelope prevalence")
    axes[1].set_title(f"F2b vs F1 (Spearman rho={rho_f2b_f1:.3f}, p={p_f2b_f1:.3f})")
    axes[1].grid(True, alpha=0.3)

    # B: F2a vs F1 scatter
    axes[2].scatter(feat["F1_level_mV"], feat["F2a_mag_mV"], alpha=0.5, s=45, color="darkorange")
    axes[2].set_xlabel("F1 — level (mV)")
    axes[2].set_ylabel("F2a — non-zero envelope magnitude (mV)")
    axes[2].set_title(f"F2a vs F1 (Spearman rho={rho_f2a_f1:.3f}, p={p_f2a_f1:.3f})")
    axes[2].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(ART / "pre_phase3_checks.png", dpi=100, bbox_inches="tight")
    print(f"\n[plot] pre_phase3_checks.png", flush=True)

    # Save updated features
    feat.to_csv(ART / "phase2_features.csv")
    print("[done]", flush=True)

    # Summary json
    summary = {
        "gateway_F1_diff": {
            "C1_median_mV": float(f1_c1.median()),
            "C2_median_mV": float(f1_c2.median()),
            "diff_mV": float(f1_c2.median() - f1_c1.median()),
            "MWU_p": float(p_f1),
            "significant": bool(p_f1 < 0.05),
        },
        "gateway_sampling": {
            "C1_n_windows_p50": float(feat[feat["circuit"]==1]["n_windows"].median()),
            "C1_span_p50_mo":   float(feat[feat["circuit"]==1]["span_months"].median()),
            "C2_n_windows_p50": float(feat[feat["circuit"]==2]["n_windows"].median()),
            "C2_span_p50_mo":   float(feat[feat["circuit"]==2]["span_months"].median()),
        },
        "F2b_vs_F1": {
            "spearman_rho": float(rho_f2b_f1),
            "spearman_p": float(p_f2b_f1),
        },
        "F2a_vs_F1": {
            "spearman_rho": float(rho_f2a_f1),
            "spearman_p": float(p_f2a_f1),
        },
        "F2b_distribution": {
            "p50": float(feat["F2b_prevalence"].median()),
            "p90": float(feat["F2b_prevalence"].quantile(0.90)),
            "n_zero": int((feat["F2b_prevalence"] == 0).sum()),
            "n_gt_half": int((feat["F2b_prevalence"] > 0.5).sum()),
        },
    }
    (ART / "pre_phase3_summary.json").write_text(json.dumps(summary, indent=2))
    print("[summary saved]", flush=True)


if __name__ == "__main__":
    main()
