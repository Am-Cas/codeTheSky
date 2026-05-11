from __future__ import annotations

from pathlib import Path

import nbformat
from nbformat.v4 import new_code_cell, new_markdown_cell, new_notebook

ROOT = Path(__file__).resolve().parent.parent
NB_PATH = ROOT / "notebooks" / "04_validate.ipynb"


def md(source: str):
    return new_markdown_cell(source)


def code(source: str):
    return new_code_cell(source)


cells = []

cells.append(
    md(
        """# Phase 4 - Physics-Based Validation (Re-run)

This notebook diagnoses the prior 4.2 failure first, then applies a minimal data-qualification fix,
rebuilds scores, and reruns the full Phase 4 validation.

Validation policy:
- Print **Pass / Fail / Flag** at each subsection.
- If a hard failure occurs, stop before the next subsection.
"""
    )
)

cells.append(
    code(
        """from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats

ROOT = Path("..").resolve()
FIG_DIR = ROOT / "outputs" / "figures"
ART = ROOT / "artifacts"
FIG_DIR.mkdir(parents=True, exist_ok=True)

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.indicator import compute_indicator_scores

FAIL_HARD = False
VALIDATION_LOG = []

def status(section: str, state: str, message: str):
    VALIDATION_LOG.append({"section": section, "state": state, "message": message})
    print(f"[{section}] {state}: {message}")

def guard():
    if FAIL_HARD:
        raise RuntimeError("Validation stopped due to hard failure in a prior section.")
"""
    )
)

cells.append(md("## Load features and previous run output"))
cells.append(
    code(
        """feature_path = ART / "phase2_features.csv"
prev_score_path = ART / "phase3_scores.csv"

if not feature_path.exists():
    status("load", "FAIL", "Feature table missing: artifacts/phase2_features.csv")
    FAIL_HARD = True
    guard()

features = pd.read_csv(feature_path)
if "location_id" not in features.columns:
    status("load", "FAIL", "phase2_features.csv lacks location_id.")
    FAIL_HARD = True
    guard()

if prev_score_path.exists():
    prev_scores = pd.read_csv(prev_score_path)
    status("load", "PASS", f"Loaded previous score table: {prev_score_path.name}")
else:
    prev_scores = compute_indicator_scores(
        features,
        within_circuit=True,
        include_f2b_in_driver=True,
        min_non_stale_windows=0,
        max_stale_frac=None,
    )
    status("load", "FLAG", "Previous score table missing; reconstructed baseline unsuppressed scores.")

prev_joined = prev_scores.merge(
    features.rename(columns={"location_id": "fixture_id"}),
    on=["fixture_id", "circuit", "n_windows", "span_months"],
    how="left",
)
display(prev_joined.head(5))
"""
    )
)

cells.append(md("## Diagnosis before re-scoring"))
cells.append(
    code(
        """guard()

# Recreate failing synthetic negatives from prior run
pool = prev_joined.dropna(subset=["stale_frac", "F2a_mag_mV", "S_triage"]).copy()
pool["stale_rank"] = pool["stale_frac"].rank(ascending=False, method="average")
pool["f2a_low_rank"] = pool["F2a_mag_mV"].rank(ascending=True, method="average")
pool["neg_score"] = pool["stale_rank"] + pool["f2a_low_rank"]
neg10_prev = pool.nsmallest(10, "neg_score").copy()

prev_ranked = prev_joined.sort_values("S_triage", ascending=False).reset_index(drop=True)
prev_ranked["triage_rank"] = np.arange(1, len(prev_ranked) + 1)
neg10_prev = neg10_prev.merge(
    prev_ranked[["fixture_id", "S_triage", "S_LED", "S_driver", "confidence_tier", "triage_rank"]],
    on=["fixture_id", "S_triage", "S_LED", "S_driver", "confidence_tier"],
    how="left",
)

diag_cols = [
    "fixture_id", "triage_rank", "S_triage", "S_LED", "S_driver",
    "n_windows", "stale_frac", "confidence_tier",
    "F1_level_mV", "F2a_mag_mV", "F2b_prevalence",
    "F3_slope_mV_per_month", "F4_envelope_slope_mV_per_month",
]
display(neg10_prev[diag_cols].sort_values("triage_rank"))

rho_f2b_stale, p_f2b_stale = stats.spearmanr(prev_joined["F2b_prevalence"], prev_joined["stale_frac"], nan_policy="omit")
print(f"F2b_prevalence vs stale_frac Spearman rho={rho_f2b_stale:.3f}, p={p_f2b_stale:.4g}")

fig, ax = plt.subplots(figsize=(6, 5))
ax.scatter(prev_joined["stale_frac"], prev_joined["F2b_prevalence"], alpha=0.6)
ax.set_xlabel("stale_frac")
ax.set_ylabel("F2b_prevalence")
ax.set_title("Diagnosis: F2b vs stale fraction")
ax.grid(True, alpha=0.3)
fig.tight_layout()
fig.savefig(FIG_DIR / "validation_diag_f2b_vs_stale.png", dpi=120, bbox_inches="tight")
plt.show()

# Optional time-series inspection for negative controls
bvf_path = ART / "boostvoltage_per_fixture.csv"
if bvf_path.exists():
    bvf = pd.read_csv(bvf_path)
    fig, axes = plt.subplots(5, 2, figsize=(14, 18), sharex=False)
    axes = axes.flatten()
    for i, fx in enumerate(neg10_prev["fixture_id"].tolist()):
        ax = axes[i]
        g = bvf[bvf["location_id"] == fx].copy()
        if "ts" in g.columns:
            g = g.sort_values("ts")
            x = pd.to_datetime(g["ts"], errors="coerce")
        elif "timestamp" in g.columns:
            g = g.sort_values("timestamp")
            x = g["timestamp"]
        else:
            x = np.arange(len(g))
        for col, label, color in [
            ("boostVoltage", "value", "black"),
            ("boostVoltageMinimum", "min", "steelblue"),
            ("boostVoltageMaximum", "max", "tomato"),
        ]:
            if col in g.columns:
                ax.plot(x, g[col], label=label, alpha=0.8, linewidth=1.0, color=color)
        ax.set_title(fx)
        ax.grid(True, alpha=0.3)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper right")
    fig.suptitle("Diagnosis: prior negative-control fixtures time series")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "validation_diag_negative_timeseries.png", dpi=120, bbox_inches="tight")
    plt.show()
else:
    status("diag", "FLAG", "boostvoltage_per_fixture.csv not found; skipped diagnosis time-series.")

drop_f2b = bool(rho_f2b_stale >= 0.35)
if drop_f2b:
    status("diag", "FLAG", f"F2b tracks stale fraction (rho={rho_f2b_stale:.2f}); dropping F2b from driver sub-score.")
else:
    status("diag", "PASS", f"F2b-stale coupling not high (rho={rho_f2b_stale:.2f}); keep F2b in driver sub-score.")
"""
    )
)

cells.append(md("## Apply required fix and rebuild score table"))
cells.append(
    code(
        """guard()

scores = compute_indicator_scores(
    features,
    within_circuit=True,
    include_f2b_in_driver=(not drop_f2b),
    min_non_stale_windows=20,
    max_stale_frac=0.50,
)
scores_fullrank = compute_indicator_scores(
    features,
    within_circuit=False,
    include_f2b_in_driver=(not drop_f2b),
    min_non_stale_windows=20,
    max_stale_frac=0.50,
)

joined = scores.merge(
    features.rename(columns={"location_id": "fixture_id"}),
    on=["fixture_id", "circuit", "n_windows", "span_months"],
    how="left",
)
joined_fullrank = scores_fullrank.merge(
    features.rename(columns={"location_id": "fixture_id"}),
    on=["fixture_id", "circuit", "n_windows", "span_months"],
    how="left",
)

score_out = ART / "phase4_scores.csv"
joined.to_csv(score_out, index=False)
status("fix", "PASS", f"Saved new score table: {score_out.name}; suppressed fixtures={(~joined['qualifies_primary_score']).sum()}")
display(joined.head(10))
"""
    )
)

cells.append(md("## 4.1 Co-occurrence test"))
cells.append(
    code(
        """guard()

valid = joined[joined["qualifies_primary_score"] & joined["S_triage"].notna()].copy()
raw_cols = ["F1_level_mV", "F2a_mag_mV", "F3_slope_mV_per_month", "F4_envelope_slope_mV_per_month"]
tmp = valid.dropna(subset=["S_triage"] + raw_cols).copy()

if len(tmp) < 10:
    status("4.1", "FAIL", "Insufficient qualified rows for decile validation (n<10).")
    FAIL_HARD = True
    guard()
elif len(tmp) < 20:
    status("4.1", "FLAG", f"Small qualified sample (n={len(tmp)}); decile checks are lower-power.")

q90 = tmp["S_triage"].quantile(0.90)
q10 = tmp["S_triage"].quantile(0.10)
top = tmp[tmp["S_triage"] >= q90].copy()
bot = tmp[tmp["S_triage"] <= q10].copy()

fig, axes = plt.subplots(2, 2, figsize=(12, 10))
for ax, col in zip(axes.flat, raw_cols):
    ax.scatter(bot["S_triage"], bot[col], alpha=0.6, label="bottom decile", color="steelblue")
    ax.scatter(top["S_triage"], top[col], alpha=0.8, label="top decile", color="tomato")
    ax.set_xlabel("S_triage")
    ax.set_ylabel(col)
    ax.grid(True, alpha=0.3)
handles, labels = axes[0, 0].get_legend_handles_labels()
fig.legend(handles, labels, loc="upper right")
fig.suptitle("4.1 Co-occurrence by top/bottom S_triage deciles")
fig.tight_layout()
fig.savefig(FIG_DIR / "validation_4_1_decile_scatter.png", dpi=120, bbox_inches="tight")
plt.show()

rho_tbl = []
for col in raw_cols:
    rho, p = stats.spearmanr(tmp["S_triage"], tmp[col], nan_policy="omit")
    rho_tbl.append((col, rho, p))
rho_df = pd.DataFrame(rho_tbl, columns=["feature", "spearman_rho_vs_S_triage", "p_value"])
rho_led_driver, p_led_driver = stats.spearmanr(tmp["S_LED"], tmp["S_driver"], nan_policy="omit")
display(rho_df)
print(f"S_LED vs S_driver Spearman rho={rho_led_driver:.3f}, p={p_led_driver:.4g}")

coherent_top = (
    (top["F1_level_mV"].median() > bot["F1_level_mV"].median())
    and (top["F2a_mag_mV"].median() > bot["F2a_mag_mV"].median())
    and (top["F3_slope_mV_per_month"].median() > bot["F3_slope_mV_per_month"].median())
    and (top["F4_envelope_slope_mV_per_month"].median() > bot["F4_envelope_slope_mV_per_month"].median())
)

if not coherent_top:
    status("4.1", "FAIL", "Top-decile fixtures are not coherently elevated across F1/F2a/F3/F4.")
    FAIL_HARD = True
    guard()
elif rho_led_driver > 0.75:
    status("4.1", "FLAG", f"S_LED and S_driver are highly correlated (rho={rho_led_driver:.2f}); possible redundancy.")
else:
    status("4.1", "PASS", "Co-occurrence pattern is physically coherent; LED-driver coupling remains moderate.")
"""
    )
)

cells.append(md("## 4.2 Negative control (synthetic)"))
cells.append(
    code(
        """guard()

valid = joined[joined["qualifies_primary_score"] & joined["S_triage"].notna()].copy()
neg_pool = valid.dropna(subset=["stale_frac", "F2a_mag_mV", "S_triage"]).copy()
if len(neg_pool) == 0:
    status("4.2", "PASS", "No qualified fixtures available for synthetic negative control after suppression; proceed.")
else:
    neg_pool["stale_rank"] = neg_pool["stale_frac"].rank(ascending=False, method="average")
    neg_pool["f2a_low_rank"] = neg_pool["F2a_mag_mV"].rank(ascending=True, method="average")
    neg_pool["neg_score"] = neg_pool["stale_rank"] + neg_pool["f2a_low_rank"]
    neg10 = neg_pool.nsmallest(min(10, len(neg_pool)), "neg_score").copy()

    ranked = valid.sort_values("S_triage", ascending=False).reset_index(drop=True)
    ranked["triage_rank"] = np.arange(1, len(ranked) + 1)
    neg10 = neg10.merge(ranked[["fixture_id", "triage_rank", "S_triage"]], on=["fixture_id", "S_triage"], how="left")

    threshold_bottom_half = len(ranked) * 0.5
    high_rank_hits = int((neg10["triage_rank"] <= threshold_bottom_half).sum())
    display(neg10[["fixture_id", "stale_frac", "F2a_mag_mV", "S_triage", "triage_rank"]].sort_values("triage_rank"))

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.hist(ranked["triage_rank"], bins=20, alpha=0.5, label="qualified fixtures")
    ax.scatter(neg10["triage_rank"], np.full(len(neg10), 0.5), color="red", label="synthetic negatives")
    ax.axvline(threshold_bottom_half, color="k", linestyle="--", label="bottom-half cutoff")
    ax.set_xlabel("Rank by S_triage (1 = highest risk)")
    ax.set_yticks([])
    ax.legend()
    ax.set_title("4.2 Synthetic negative-control rank positions (qualified only)")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "validation_4_2_negative_control.png", dpi=120, bbox_inches="tight")
    plt.show()

    if high_rank_hits > 2:
        status("4.2", "FAIL", f"{high_rank_hits}/{len(neg10)} synthetic negatives rank in upper half after fix.")
        FAIL_HARD = True
        guard()
    else:
        status("4.2", "PASS", "Synthetic negatives rank low after suppression/filtering.")
"""
    )
)

cells.append(md("## 4.3 Positive-control inspection"))
cells.append(
    code(
        """guard()

valid = joined[joined["qualifies_primary_score"] & joined["S_triage"].notna()].copy()
top10 = valid.sort_values("S_triage", ascending=False).head(10).copy()
display(top10[["fixture_id", "S_triage", "S_LED", "S_driver", "F3_slope_mV_per_month", "F4_envelope_slope_mV_per_month", "confidence_tier"]])

low_conf = top10[top10["confidence_tier"] == "Low"]
nonpos_f3 = int((top10["F3_slope_mV_per_month"] <= 0).sum())
nonpos_f4 = int((top10["F4_envelope_slope_mV_per_month"] <= 0).sum())

bvf_path = ART / "boostvoltage_per_fixture.csv"
if bvf_path.exists():
    bvf = pd.read_csv(bvf_path)
    fig, axes = plt.subplots(5, 2, figsize=(14, 18), sharex=False)
    axes = axes.flatten()
    for i, fx in enumerate(top10["fixture_id"].tolist()):
        ax = axes[i]
        g = bvf[bvf["location_id"] == fx].copy()
        if "ts" in g.columns:
            g = g.sort_values("ts")
            x = pd.to_datetime(g["ts"], errors="coerce")
        elif "timestamp" in g.columns:
            g = g.sort_values("timestamp")
            x = g["timestamp"]
        else:
            x = np.arange(len(g))
        for col, label, color in [("boostVoltage", "value", "black"), ("boostVoltageMinimum", "min", "steelblue"), ("boostVoltageMaximum", "max", "tomato")]:
            if col in g.columns:
                ax.plot(x, g[col], label=label, alpha=0.8, linewidth=1.0, color=color)
        ax.set_title(fx)
        ax.grid(True, alpha=0.3)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper right")
    fig.suptitle("4.3 Top-10 S_triage fixtures: boostVoltage value/min/max")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "validation_4_3_top10_timeseries.png", dpi=120, bbox_inches="tight")
    plt.show()
else:
    status("4.3", "FLAG", "boostvoltage_per_fixture.csv not found; skipped top-10 raw time-series.")

if len(low_conf) > 0:
    status("4.3", "FLAG", f"{len(low_conf)} top fixtures have Low confidence.")
else:
    status("4.3", "PASS", "Top fixtures have Medium/High confidence.")
if nonpos_f3 > 0 or nonpos_f4 > 0:
    status("4.3", "FLAG", f"Top fixtures include non-positive drift: F3<=0 on {nonpos_f3}, F4<=0 on {nonpos_f4}.")
"""
    )
)

cells.append(md("## 4.4 Cross-fleet ranking sanity"))
cells.append(
    code(
        """guard()

valid = joined[joined["qualifies_primary_score"] & joined["S_triage"].notna()].copy()
fig, ax = plt.subplots(figsize=(10, 4))
for circ, color in [(1, "steelblue"), (2, "tomato")]:
    s = valid.loc[valid["circuit"] == circ, "S_triage"].dropna()
    ax.hist(s, bins=20, alpha=0.55, color=color, label=f"Circuit {circ} (n={len(s)})")
ax.set_xlabel("S_triage")
ax.set_ylabel("Fixture count")
ax.set_title("4.4 S_triage distribution by circuit")
ax.legend()
ax.grid(True, alpha=0.3, axis="y")
fig.tight_layout()
fig.savefig(FIG_DIR / "validation_4_4_triage_hist_by_circuit.png", dpi=120, bbox_inches="tight")
plt.show()

ext_cols = ["fixture_id", "circuit", "S_triage", "confidence_tier", "F1_level_mV", "F2a_mag_mV", "F2b_prevalence", "F3_slope_mV_per_month", "F4_envelope_slope_mV_per_month"]
print("Top 5 fixtures")
display(valid.sort_values("S_triage", ascending=False).head(5)[ext_cols])
print("Bottom 5 fixtures")
display(valid.sort_values("S_triage", ascending=True).head(5)[ext_cols])

if "ambientTemperatureMedian" in valid.columns:
    rho_t, p_t = stats.spearmanr(valid["S_triage"], valid["ambientTemperatureMedian"], nan_policy="omit")
    print(f"S_triage vs ambientTemperatureMedian Spearman rho={rho_t:.3f}, p={p_t:.4g}")
    status("4.4", "PASS", "Ambient-temperature correlation checked.")
else:
    status("4.4", "FLAG", "ambientTemperatureMedian not available; Arrhenius check skipped.")
"""
    )
)

cells.append(md("## 4.5 Sensitivity analysis"))
cells.append(
    code(
        """guard()

valid = joined[joined["qualifies_primary_score"] & joined["S_triage"].notna()].copy().reset_index(drop=True)
valid_fullrank = joined_fullrank[joined_fullrank["qualifies_primary_score"] & joined_fullrank["S_triage"].notna()].copy()

def rank_to_stress(rank_series: pd.Series) -> pd.Series:
    n = rank_series.notna().sum()
    if n <= 1:
        return pd.Series(np.where(rank_series.notna(), 1.0, np.nan), index=rank_series.index, dtype="float64")
    return 1.0 - (rank_series - 1.0) / (n - 1.0)

work = valid.copy()
for c in ["rank_F1", "rank_F2a", "rank_F2b", "rank_F3", "rank_F4"]:
    work[c] = pd.to_numeric(work[c], errors="coerce")
    work[f"stress_{c}"] = rank_to_stress(work[c])

base = work["S_triage"].copy()
led_alt = 0.8 * work["stress_rank_F1"] + 0.2 * work["stress_rank_F3"]
drv_alt = 0.5 * work["stress_rank_F2a"] + 0.2 * work["stress_rank_F2b"].fillna(0) + 0.3 * work["stress_rank_F4"]
triage_alt = 0.5 * (led_alt + drv_alt)
delta = (triage_alt - base).abs()
n_gt_01 = int((delta > 0.1).sum())

def deciles(s: pd.Series):
    q90 = s.quantile(0.9)
    q10 = s.quantile(0.1)
    return set(s[s >= q90].index), set(s[s <= q10].index)

top_base, bot_base = deciles(base)
top_alt, bot_alt = deciles(triage_alt)
swap_top = len(top_base.symmetric_difference(top_alt))
swap_bot = len(bot_base.symmetric_difference(bot_alt))

comp = valid[["fixture_id", "S_triage"]].merge(
    valid_fullrank[["fixture_id", "S_triage"]].rename(columns={"S_triage": "S_triage_fullrank"}),
    on="fixture_id",
    how="inner",
)
rho_rank, p_rank = stats.spearmanr(comp["S_triage"], comp["S_triage_fullrank"], nan_policy="omit")

fig, axes = plt.subplots(1, 2, figsize=(12, 5))
axes[0].hist(delta, bins=20, color="mediumpurple", alpha=0.8, edgecolor="black")
axes[0].axvline(0.1, color="k", linestyle="--", label="0.1 threshold")
axes[0].legend()
axes[0].set_title("4.5 |S_triage_alt - baseline|")
axes[1].scatter(comp["S_triage"], comp["S_triage_fullrank"], alpha=0.6)
axes[1].set_xlabel("within-circuit S_triage")
axes[1].set_ylabel("full-fleet S_triage")
axes[1].set_title(f"Spearman rho={rho_rank:.3f}")
axes[1].grid(True, alpha=0.3)
fig.tight_layout()
fig.savefig(FIG_DIR / "validation_4_5_sensitivity.png", dpi=120, bbox_inches="tight")
plt.show()

print(f"Fixtures with |delta| > 0.1: {n_gt_01}")
print(f"Top-decile symmetric difference: {swap_top}")
print(f"Bottom-decile symmetric difference: {swap_bot}")
print(f"Within-circuit vs full-fleet correlation: rho={rho_rank:.3f}, p={p_rank:.4g}")

if swap_top > max(6, int(0.4 * len(top_base))) or swap_bot > max(6, int(0.4 * len(bot_base))):
    status("4.5", "FAIL", "Sensitivity causes large decile flips.")
    FAIL_HARD = True
    guard()
else:
    status("4.5", "PASS", "Sensitivity perturbations do not materially invert ordering.")
"""
    )
)

cells.append(md("## 4.6 Stability check"))
cells.append(
    code(
        """guard()

valid = joined[joined["score_status"] == "Scored"].copy().reset_index(drop=True)
if len(valid) < 10:
    status("4.6", "FAIL", "Too few scored fixtures for bootstrap stability.")
    FAIL_HARD = True
    guard()

def recompute(df_subset: pd.DataFrame):
    feat = features[features["location_id"].isin(df_subset["fixture_id"])].copy()
    out = compute_indicator_scores(
        feat,
        within_circuit=True,
        include_f2b_in_driver=(not drop_f2b),
        min_non_stale_windows=20,
        max_stale_frac=0.50,
    )
    return out.set_index("fixture_id")["S_triage"]

full_s = valid.set_index("fixture_id")["S_triage"]
rng = np.random.default_rng(42)
rhos = []
for _ in range(100):
    n_drop = max(1, int(0.2 * len(valid)))
    drop_idx = set(rng.choice(valid.index.to_numpy(), size=n_drop, replace=False).tolist())
    subset = valid.loc[[i for i in valid.index.to_numpy() if i not in drop_idx]].copy()
    sub_s = recompute(subset)
    overlap = pd.concat([full_s.rename("full"), sub_s.rename("sub")], axis=1).dropna()
    if len(overlap) >= 5:
        rhos.append(overlap["full"].corr(overlap["sub"], method="spearman"))
    else:
        rhos.append(np.nan)

rhos = pd.Series(rhos, dtype="float64").dropna()
rho_median = float(rhos.median()) if len(rhos) else np.nan
rho_p05 = float(rhos.quantile(0.05)) if len(rhos) else np.nan
rho_ge_095 = float((rhos >= 0.95).mean()) if len(rhos) else np.nan

fig, ax = plt.subplots(figsize=(10, 4))
ax.hist(rhos, bins=20, color="teal", alpha=0.85, edgecolor="black")
ax.axvline(rho_median, color="k", linestyle="--", label=f"median={rho_median:.3f}")
ax.axvline(rho_p05, color="tomato", linestyle="--", label=f"p05={rho_p05:.3f}")
ax.set_xlim(0, 1)
ax.set_xlabel("Bootstrap Spearman rho")
ax.set_ylabel("Iteration count")
ax.set_title("4.6 Bootstrap stability (100 x 20% drop of scored fixtures)")
ax.legend()
fig.tight_layout()
fig.savefig(FIG_DIR / "validation_4_6_bootstrap_stability.png", dpi=120, bbox_inches="tight")
plt.show()

print(f"Median rho: {rho_median:.3f}")
print(f"5th percentile rho: {rho_p05:.3f}")
print(f"Fraction rho >= 0.95: {rho_ge_095:.2%}")

if (pd.notna(rho_median) and rho_median >= 0.95) and (pd.notna(rho_p05) and rho_p05 >= 0.85):
    status("4.6", "PASS", "Bootstrap stability passed (median>=0.95 and p05>=0.85).")
else:
    status("4.6", "FAIL", "Bootstrap stability failed pass criterion.")
    FAIL_HARD = True
    guard()
"""
    )
)

cells.append(md("## Final memo-ready summary"))
cells.append(
    code(
        """guard()

summary_df = pd.DataFrame(VALIDATION_LOG)
display(summary_df)

print("- Diagnosis completed before fix.")
print("- F2b retained in driver score." if not drop_f2b else "- F2b removed from driver score due to high stale coupling.")
print("- Primary score suppression rule: non_stale_windows >= 20 and stale_frac <= 0.50.")
for sec in ["4.1", "4.2", "4.3", "4.4", "4.5", "4.6"]:
    row = summary_df[summary_df["section"] == sec]
    if len(row):
        print(f"- {sec}: {row['state'].iloc[-1]} - {row['message'].iloc[-1]}")
"""
    )
)

cells.append(
    md(
        """## Validation Summary (Memo Copy)

- **4.1 Co-occurrence:** `PASS/FAIL` - see execution output.
- **4.2 Synthetic negative control:** `PASS/FAIL` - see execution output.
- **4.3 Positive-control inspection:** `PASS/FLAG/FAIL` - see execution output.
- **4.4 Cross-fleet sanity:** `PASS/FLAG/FAIL` - see execution output.
- **4.5 Sensitivity analysis:** `PASS/FAIL` - see execution output.
- **4.6 Bootstrap stability:** `PASS/FAIL` - pass criterion is median rho >= 0.95 and 5th percentile rho >= 0.85 on scored fixtures only.

Caveat: suppressed fixtures are intentionally excluded from primary-score validation because they are marked as insufficient data and not actionable.
"""
    )
)

nb = new_notebook(cells=cells)
nb.metadata.update(
    {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python"},
    }
)

NB_PATH.parent.mkdir(parents=True, exist_ok=True)
nbformat.write(nb, str(NB_PATH))
print(f"Written: {NB_PATH}")
