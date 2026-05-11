"""
Spatial stale check: is stale_frac clustered by circuit or runway position?

location_id pattern: yyc-c{circuit}-r{runway_number}
e.g. yyc-c1-r211, yyc-c2-r40

If stale_frac is spatially clustered, the signal is partly RF-coverage
and should only be used as a filter, not a physics-grounded aging proxy.
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
ART = ROOT / "artifacts"


def main():
    stale = pd.read_csv(ART / "stale_fraction_per_fixture.csv", index_col=0)
    stale.index.name = "location_id"

    # Parse circuit and runway number from location_id
    def parse_loc(loc_id: str):
        m = re.match(r"yyc-c(\d+)-r(\d+)", str(loc_id))
        if m:
            return int(m.group(1)), int(m.group(2))
        return None, None

    stale[["circuit", "runway"]] = pd.DataFrame(
        [parse_loc(x) for x in stale.index], index=stale.index
    )
    stale = stale.dropna(subset=["circuit"])

    print(f"Parsed: {len(stale)} fixtures, circuits: {sorted(stale['circuit'].unique())}", flush=True)

    # --- 1. Circuit-level summary ---
    by_circuit = stale.groupby("circuit")["stale_frac"].agg(
        n="size",
        mean="mean",
        median="median",
        p10=lambda s: s.quantile(0.10),
        p90=lambda s: s.quantile(0.90),
        std="std",
    ).round(4)
    print("\n[circuit-level stale_frac]")
    print(by_circuit.to_string())

    # Statistical test: Mann-Whitney U between circuits
    c1 = stale.loc[stale["circuit"] == 1, "stale_frac"]
    c2 = stale.loc[stale["circuit"] == 2, "stale_frac"]
    if len(c1) > 0 and len(c2) > 0:
        u_stat, p_mw = stats.mannwhitneyu(c1, c2, alternative="two-sided")
        print(f"\n  Mann-Whitney U(c1 vs c2): U={u_stat:.1f}, p={p_mw:.4f}")
        if p_mw < 0.05:
            print("  -> Circuits differ significantly in stale_frac (p<0.05)")
        else:
            print("  -> No significant difference between circuits (p>=0.05)")

    # --- 2. Spatial gradient: does stale_frac correlate with runway number? ---
    corr_c1 = stats.spearmanr(
        stale.loc[stale["circuit"] == 1, "runway"],
        stale.loc[stale["circuit"] == 1, "stale_frac"]
    )
    corr_c2 = stats.spearmanr(
        stale.loc[stale["circuit"] == 2, "runway"],
        stale.loc[stale["circuit"] == 2, "stale_frac"]
    )
    print(f"\n[spatial gradient (runway position vs stale_frac)]")
    print(f"  circuit 1 Spearman rho={corr_c1.statistic:.3f}, p={corr_c1.pvalue:.4f}")
    print(f"  circuit 2 Spearman rho={corr_c2.statistic:.3f}, p={corr_c2.pvalue:.4f}")

    # --- 3. Top/bottom stale fixtures for RF hypothesis ---
    print("\n[10 highest stale_frac (may be RF-shadowed)]")
    print(stale.nlargest(10, "stale_frac")[["stale_frac", "circuit", "runway"]].to_string())

    print("\n[10 lowest stale_frac (high activity, most reliable samples)]")
    print(stale.nsmallest(10, "stale_frac")[["stale_frac", "circuit", "runway"]].to_string())

    # --- 4. Clustering coefficient: variance within circuit vs. total variance ---
    grand_var = stale["stale_frac"].var()
    within_var = stale.groupby("circuit")["stale_frac"].var().mean()
    between_var = grand_var - within_var
    clustering_ratio = between_var / grand_var if grand_var > 0 else 0.0
    print(f"\n[clustering]")
    print(f"  Total variance:   {grand_var:.5f}")
    print(f"  Within-circuit:   {within_var:.5f}")
    print(f"  Between-circuit:  {between_var:.5f}")
    print(f"  Between/(Total):  {clustering_ratio:.3f}  (0=no clustering, 1=all clustering)")

    # --- Plots ---
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    # Box plot by circuit
    stale.boxplot(column="stale_frac", by="circuit", ax=axes[0])
    axes[0].set_title("stale_frac by circuit")
    axes[0].set_xlabel("Circuit")
    axes[0].set_ylabel("stale_frac")
    plt.sca(axes[0])
    plt.title("stale_frac by circuit")

    # Scatter: runway position vs stale_frac for both circuits
    for circ, color in [(1, "steelblue"), (2, "tomato")]:
        sub = stale[stale["circuit"] == circ]
        axes[1].scatter(sub["runway"], sub["stale_frac"], alpha=0.6, s=50,
                        label=f"Circuit {circ}", color=color)
    axes[1].set_xlabel("Runway number")
    axes[1].set_ylabel("stale_frac")
    axes[1].set_title("Runway position vs stale_frac")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    # Histogram by circuit
    for circ, color in [(1, "steelblue"), (2, "tomato")]:
        sub = stale[stale["circuit"] == circ]
        axes[2].hist(sub["stale_frac"], bins=20, alpha=0.5, color=color, label=f"Circuit {circ}")
    axes[2].set_xlabel("stale_frac")
    axes[2].set_ylabel("Count")
    axes[2].set_title("stale_frac distribution")
    axes[2].legend()
    axes[2].grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    plt.savefig(ART / "spatial_stale_check.png", dpi=100, bbox_inches="tight")
    print("\n[plot] spatial_stale_check.png saved")

    # Save summary
    summary = {
        "circuit_summary": by_circuit.to_dict(),
        "circuit_mw_test": {"U": float(u_stat), "p": float(p_mw)} if "p_mw" in dir() else {},
        "spatial_gradient": {
            "c1_spearman_rho": float(corr_c1.statistic),
            "c1_spearman_p": float(corr_c1.pvalue),
            "c2_spearman_rho": float(corr_c2.statistic),
            "c2_spearman_p": float(corr_c2.pvalue),
        },
        "clustering_ratio_between_over_total": float(clustering_ratio),
    }
    (ART / "spatial_stale_summary.json").write_text(json.dumps(summary, indent=2))
    print("[done]")


if __name__ == "__main__":
    main()
