"""
Builds notebooks/phase2_physics.ipynb programmatically via nbformat.
All markdown cells contain the physics derivation; code cells compute
the supporting per-fixture statistics but do NOT build the composite
score (that is Phase 3).
"""
from __future__ import annotations

from pathlib import Path

import nbformat
from nbformat.v4 import new_code_cell, new_markdown_cell, new_notebook

ROOT = Path(__file__).resolve().parent.parent
NB_PATH = ROOT / "notebooks" / "phase2_physics.ipynb"

# ---------------------------------------------------------------------------
# Cell factories
# ---------------------------------------------------------------------------

def md(source: str) -> nbformat.NotebookNode:
    return new_markdown_cell(source)

def code(source: str) -> nbformat.NotebookNode:
    return new_code_cell(source)

# ---------------------------------------------------------------------------
# Notebook cells
# ---------------------------------------------------------------------------

cells = []

# ── Header ────────────────────────────────────────────────────────────────
cells.append(md("""\
# Phase 2 — Physics Derivation: LED Degradation from Boost-Voltage Telemetry

## What this notebook is

A physics-first derivation of how LED aging should manifest in the
Axon EQ telemetry that is actually available in this export. Each section:
1. states the physical mechanism from first principles,
2. derives the directional prediction on the observable signal,
3. lists the required assumptions, and
4. names what confounders the signal **cannot** distinguish from true aging.

The derivation governs which features go into the Phase 3 indicator and
how they are normalised.

---

## Signals rejected before Phase 2 — and why

| Signal | Reason rejected |
|---|---|
| `led01Temperature`, `led02Temperature` | Sentinel-flat at exactly 25 °C across all 2 476 readings on all 170 fixtures. Uninitialized register default, not telemetry. |
| `heater01/02Temperature` | Sentinel-flat at −55 °C. |
| `magneticX/Y/Z` | All-zero across fleet. |
| `internalTemperature − ambientTemperature` | dT identically 0.0 °C on all 170 fixtures, **including the 15 most active** (stale_frac ≤ 7 %). Sensors colocated; no thermal gradient recoverable. |
| `inputCurrent` | Not in export. CCR loop current must be taken as an *assumption*, not a measurement. |
| Alarms, failsafe, RF metrics | Not in export (Stream 1 absent). |

---

## Surviving signals and their roles

| Signal | Role |
|---|---|
| `boostVoltage`, `boostVoltageMinimum`, `boostVoltageMaximum` | **Primary** — all four Phase 2 features derived from this triplet |
| `stale`, `lastSeen` | **Filter only** — reporting quality; not scored |

---

## Spatial stale finding (pre-Phase 2 check)

Circuit 2 is structurally ~13 pp more stale than Circuit 1 (median 83 % vs 74 %,
Mann-Whitney p < 0.0001), consistent with RF-coverage differences between gateways.
Between-circuit variance accounts for only 12 % of total stale variance; the
remaining 88 % is within-circuit. A spatial cluster of very active fixtures exists
at `yyc-c1-r205` through `yyc-c1-r227`.

**Implication**: raw stale_frac cannot be used as an absolute threshold across
circuits. Used as a per-fixture sample-count filter only.
"""))

# ── Setup ─────────────────────────────────────────────────────────────────
cells.append(md("## Setup"))
cells.append(code("""\
from __future__ import annotations
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path("..").resolve()
ART  = ROOT / "artifacts"
CSVS = [
    ROOT / "LED_sensor1" / "yyc_device_1.csv",
    ROOT / "LED_sensor1" / "yyc_device_2.csv",
]

DTYPE = {
    "device_id": "category", "location_id": "category", "metric": "category",
    "gateway_id": "category", "numeric_value": "object", "timestamp": "float64",
}
USE = ["timestamp", "location_id", "metric", "numeric_value", "gateway_id"]

def excel_to_ts(s):
    return pd.to_datetime(s, unit="D", origin="1899-12-30", errors="coerce")

frames = []
for c in CSVS:
    df = pd.read_csv(c, usecols=USE, dtype=DTYPE, low_memory=False)
    frames.append(df)
raw = pd.concat(frames, ignore_index=True)
raw["ts"] = excel_to_ts(raw["timestamp"])
raw["v"]  = pd.to_numeric(raw["numeric_value"], errors="coerce")

print(f"Loaded {len(raw):,} rows, {raw['location_id'].nunique()} fixtures")
print(f"Time span: {raw['ts'].min()} to {raw['ts'].max()}")
"""))

# ── Filter boostVoltage ────────────────────────────────────────────────────
cells.append(code("""\
# Extract and clean the boost-voltage triplet
BV_METRICS = ["boostVoltage", "boostVoltageMinimum", "boostVoltageMaximum"]
bv = raw[raw["metric"].isin(BV_METRICS)].copy()
bv = bv[bv["v"] > 0].copy()   # drop sentinel zeros

bv_wide = bv.pivot_table(
    index=["ts", "location_id"],
    columns="metric",
    values="v",
    aggfunc="first",
).reset_index()
bv_wide.columns.name = None
bv_wide["envelope"] = bv_wide["boostVoltageMaximum"] - bv_wide["boostVoltageMinimum"]

print(f"Clean boost-voltage rows: {len(bv_wide):,}")
print(bv_wide.describe().round(1))
"""))

# ── MECHANISM 1: Level ─────────────────────────────────────────────────────
cells.append(md(r"""\
---
## Mechanism 1 — Boost-Voltage Level: LED Forward-Voltage Drift

### Physical mechanism

An LED is a p-n junction diode. Its forward voltage $V_f$ is governed by:

$$V_f(I, T_j) = \frac{n k T_j}{q} \ln\!\left(\frac{I}{I_0}\right) + I R_s$$

where $n$ is the ideality factor (~1–2), $T_j$ the junction temperature,
$I_0$ the dark-saturation current, and $R_s$ the series resistance.

With age, two degradation pathways raise $V_f$:

1. **Defect accumulation in the active layer** increases the ideality factor $n$,
   shifting the ln-term upward.
2. **Contact and bond-wire oxidation / electromigration** raises the bulk series
   resistance $R_s$, adding an ohmic voltage drop proportional to current.

Both pathways shift $V_f$ upward over time at fixed current and temperature.

### CCR loop constraint

The Axon EQ fixtures are driven by a **Constant-Current Regulator (CCR)**, which
holds the series-loop current $I_{\text{CCR}}$ (nominally ~5 000 mA) fixed.
The CCR output voltage compliance $V_{\text{boost}}$ adjusts to maintain $I_{\text{CCR}}$:

$$V_{\text{boost}} = N \cdot V_f(I_{\text{CCR}},\, T_j) + V_{\text{driver,drop}}$$

where $N$ is the number of LED strings in series and $V_{\text{driver,drop}}$
is the internal driver voltage drop (assumed roughly constant).

Therefore:

$$\frac{dV_{\text{boost}}}{dt}\bigg|_{I_{\text{CCR}} = \text{const}} = N \cdot \frac{dV_f}{dt}$$

**LED aging → $V_f$ rises → $V_{\text{boost}}$ must rise to maintain $I_{\text{CCR}}$.**

### Directional prediction

> A fixture with aging LEDs should show a **sustained upward shift in median
> `boostVoltage`** relative to its own commissioning baseline and relative to
> the fleet median.

### Assumptions required

- $I_{\text{CCR}}$ is held constant throughout the export window. *(Cannot verify;
  Stream 1 / `inputCurrent` is absent.)*
- $T_j$ is roughly similar across reporting windows within a fixture, or that
  the thermal state is captured by the internalTemperature sensor. *(Partially
  violated: dT = 0 and we cannot recover $T_j$.)*
- Nominal $V_f$ is similar across the fleet at commissioning. *(Reasonable for
  a single-generation airfield deployment.)*

### Confounders — what elevated level *cannot* distinguish

- **Temperature variation**: $V_f$ decreases with temperature at ~−2 mV/°C.
  A cold fixture reads artificially high; a hot fixture reads artificially low.
  Without LED $T_j$, we cannot correct for this.
- **CCR step changes**: if the regulator was recalibrated or the intensity
  was changed, boost voltage shifts as a step, not a trend.
- **LED bin variation at commissioning**: different optical bins have different
  $V_f$ at ship; initial fleet spread is ~200–400 mV.

### Feature definition

$$F_1(\text{fixture}) = \text{trimmed mean}\!\left(\{V_{\text{boost},t}\}\right)$$

Trimmed mean (5 %–95 %) reduces sensitivity to the sentinel-zero outliers
and transient dips. Normalized cross-fleet in Phase 3.
"""))

cells.append(code("""\
# F1 — per-fixture trimmed mean of boostVoltage
f1 = bv_wide.groupby("location_id")["boostVoltage"].apply(
    lambda s: float(stats.trim_mean(s.dropna().values, 0.05))
).rename("F1_level_mV")

print(f"F1 computed for {f1.notna().sum()} fixtures")
print(f"  Fleet p10={f1.quantile(0.10):.0f} mV  p50={f1.quantile(0.50):.0f} mV  p90={f1.quantile(0.90):.0f} mV")
print(f"  Range: [{f1.min():.0f}, {f1.max():.0f}] mV")

fig, ax = plt.subplots(figsize=(8, 4))
ax.hist(f1, bins=30, edgecolor="black", alpha=0.75)
ax.set_xlabel("F1 — trimmed mean boostVoltage (mV)")
ax.set_ylabel("Fixture count")
ax.set_title("Distribution of per-fixture boost-voltage level")
ax.axvline(f1.median(), color="r", linestyle="--", label=f"Fleet median {f1.median():.0f} mV")
ax.legend()
plt.tight_layout()
plt.savefig(ART / "f1_level_distribution.png", dpi=100)
plt.show()
print("Saved f1_level_distribution.png")
"""))

# ── MECHANISM 2: Envelope width ────────────────────────────────────────────
cells.append(md(r"""\
---
## Mechanism 2 — Envelope Width: Flicker and Instability Precursors

### Physical mechanism

Within each Axon EQ reporting window (~1 hour), the firmware records the
minimum and maximum observed $V_{\text{boost}}$, yielding an **envelope width**:

$$E = V_{\text{boost, max}} - V_{\text{boost, min}}$$

Envelope width is sensitive to two independent aging mechanisms:

#### 2a. Driver output-capacitor degradation

The boost converter's output electrolytic capacitor has capacitance $C_{\text{out}}$
with equivalent series resistance $\text{ESR}$. As electrolyte dries out with
age (Arrhenius-accelerated by temperature):

- $C_{\text{out}}$ decreases → larger voltage ripple $\Delta V = I / (f C_{\text{out}})$
- $\text{ESR}$ increases → additional ripple component $\Delta V_{\text{ESR}} = I \cdot \text{ESR}$

Both mechanisms **widen the envelope at fixed load current**.

#### 2b. LED binning and inter-chip thermal stress

If one LED chip in a series string ages faster (higher $V_f$), the driver
must react more aggressively to stabilise current, causing transient voltage
swings. The min/max envelope captures these transients before the mean shifts.

#### 2c. Partial lamp failure

When one LED chip develops an intermittent open-circuit or high-resistance
fault, current path alternates — producing large transient voltage swings
visible in the min/max envelope before the mean value changes.

### Directional prediction

> A fixture approaching failure should show **widening envelope** before
> a significant shift in mean level (because capacitor degradation and
> intermittent faults manifest first as instability, then as sustained drift).

This makes envelope width a **leading indicator** relative to level shift.

### Assumptions required

- Reporting window duration is consistent across fixtures and over time. *(Cadence
  analysis shows ~1 h median with a long P90 tail — windows of different length
  produce different natural envelope widths even at steady state.)*
- Envelope reflects true driver behaviour, not firmware quantisation artefacts.

### Confounders

- **Load transients**: supply switching or grid events widen the envelope
  transiently without aging. Time-persistent widening is a stronger signal.
- **Window-length variation**: longer reporting windows naturally accumulate
  wider min/max ranges even without aging.
- **Commissioning ripple**: some fixture models have higher baseline ripple by design.

### Feature definition

$$F_2(\text{fixture}) = \text{median}\!\left(\{V_{\text{boost, max}, t} - V_{\text{boost, min}, t}\}\right)$$

Median (rather than mean) is robust to occasional large transients.
Normalized cross-fleet in Phase 3.
"""))

cells.append(code("""\
# F2 — per-fixture median envelope width
f2 = bv_wide.groupby("location_id")["envelope"].median().rename("F2_envelope_mV")

print(f"F2 computed for {f2.notna().sum()} fixtures")
print(f"  Fleet p10={f2.quantile(0.10):.0f} mV  p50={f2.quantile(0.50):.0f} mV  p90={f2.quantile(0.90):.0f} mV")
print(f"  Range: [{f2.min():.0f}, {f2.max():.0f}] mV")

fig, axes = plt.subplots(1, 2, figsize=(12, 4))
axes[0].hist(f2, bins=30, edgecolor="black", alpha=0.75, color="steelblue")
axes[0].set_xlabel("F2 — median envelope width (mV)")
axes[0].set_ylabel("Fixture count")
axes[0].set_title("Per-fixture envelope width distribution")
axes[0].axvline(f2.median(), color="r", linestyle="--", label=f"Fleet median {f2.median():.0f} mV")
axes[0].legend()

# F1 vs F2 scatter
merged = pd.concat([f1, f2], axis=1).dropna()
axes[1].scatter(merged["F1_level_mV"], merged["F2_envelope_mV"], alpha=0.5, s=40)
axes[1].set_xlabel("F1 — trimmed mean level (mV)")
axes[1].set_ylabel("F2 — median envelope width (mV)")
axes[1].set_title("Level vs Envelope (expect weak-moderate correlation)")
r, p = stats.spearmanr(merged["F1_level_mV"], merged["F2_envelope_mV"])
axes[1].text(0.05, 0.95, f"Spearman r={r:.3f}, p={p:.3f}", transform=axes[1].transAxes,
             verticalalignment="top", fontsize=9)
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(ART / "f2_envelope_distribution.png", dpi=100)
plt.show()
print("Saved f2_envelope_distribution.png")
"""))

# ── MECHANISM 3: Temporal drift in level ─────────────────────────────────
cells.append(md(r"""\
---
## Mechanism 3 — Temporal Drift in Boost-Voltage Level

### Physical mechanism

Feature 1 captures *cross-sectional* level (average over the export window).
A fixture with a historically high $V_f$ at commissioning will always rank high,
even if it is not actively degrading. Feature 3 corrects for this by measuring
the **time-rate of change** of median boost voltage within the export window:

$$F_3(\text{fixture}) = \hat{\beta}_1\quad\text{from OLS}\quad
  \overline{V_{\text{boost}, w}} = \beta_0 + \beta_1 \cdot t_w + \varepsilon_w$$

where $\overline{V_{\text{boost}, w}}$ is the trimmed mean over reporting window
$w$ and $t_w$ is the window timestamp in units of **months**.

This controls for fixture-specific baseline $\beta_0$ and isolates the
degradation rate. Under the $V_f$-drift model:

$$F_3 \approx N \cdot \frac{dV_f}{dt}$$

which has units of mV/month and is **zero for a stable fixture and positive for a
degrading one** (assuming constant current and temperature on average).

### Why the time-domain feature is the strongest single aging signal

Cross-sectional level (F1) conflates:
- Initial LED bin differences (captured by $\beta_0$, removed by F3)
- Systematic temperature effects (partly time-averaged out in a long window)
- True aging drift (captured by $\beta_1$ = F3)

F3 separates aging drift from standing offset, making it the most specific
aging signal available in this dataset.

### Assumptions required

- $I_{\text{CCR}}$ is **stable over the export window**. A CCR recalibration
  mid-window creates a step in $\overline{V_{\text{boost}}}$ that OLS will
  interpret as drift.
- Seasonal temperature variation is slow compared to the export window.
  *(The export spans ~months; seasonal $T$ variation of 10–20 °C introduces
  ~20–40 mV seasonal oscillation in $V_f$. At the export-window scale, this
  appears as a slow trend that may partially alias into F3.)*
- A minimum of **N ≥ 30 reporting windows** spread over at least **2 months**
  is required for a reliable slope estimate.

### Fleet coverage assessment

The code cell below determines what fraction of the fleet meets the
minimum-sample requirement.

### Confounders

- Same as F1, plus **OLS is sensitive to outliers** — one anomalous reporting
  window (maintenance event, CCR trip) introduces a slope artefact.
  Robust regression (Theil-Sen) is used instead of OLS.
- **Trend reversal**: a fixture that was declining and then partially recovered
  (e.g., lamp replacement) will show near-zero slope, masking the earlier decline.
"""))

cells.append(code("""\
# F3 — per-fixture temporal drift of trimmed mean boostVoltage
# Requires >= 30 windows across >= 2 months. Use Theil-Sen for robustness.
from scipy.stats import theilslopes

# Bin to daily medians to reduce cadence noise
bv_wide["date"] = bv_wide["ts"].dt.floor("D")
daily = bv_wide.groupby(["location_id", "date"])["boostVoltage"].apply(
    lambda s: float(stats.trim_mean(s.values, 0.05))
).reset_index(name="bv_trim_mean")

# Convert date to months-since-start (float)
t_origin = daily["date"].min()
daily["t_months"] = (daily["date"] - t_origin).dt.total_seconds() / (3600 * 24 * 30.44)

# For each fixture: compute Theil-Sen slope (mV/month)
MIN_WINDOWS = 30
MIN_SPAN_MONTHS = 2.0

results = {}
for fx, grp in daily.groupby("location_id"):
    grp = grp.dropna(subset=["bv_trim_mean"])
    span = grp["t_months"].max() - grp["t_months"].min()
    n = len(grp)
    if n < MIN_WINDOWS or span < MIN_SPAN_MONTHS:
        results[fx] = {"n": n, "span_months": float(span), "qualifies": False,
                       "slope_mV_per_month": np.nan}
        continue
    slope, intercept, lo, hi = theilslopes(grp["bv_trim_mean"].values, grp["t_months"].values)
    results[fx] = {
        "n": int(n), "span_months": float(span), "qualifies": True,
        "slope_mV_per_month": float(slope),
        "slope_lo": float(lo), "slope_hi": float(hi),
        "intercept": float(intercept),
    }

res_df = pd.DataFrame(results).T
res_df.index.name = "location_id"

n_qualify = res_df["qualifies"].sum()
print(f"Fixtures qualifying for F3 ({MIN_WINDOWS}+ windows, {MIN_SPAN_MONTHS}+ months): "
      f"{n_qualify} / {len(res_df)} ({100*n_qualify/len(res_df):.0f}%)")

f3_raw = pd.to_numeric(res_df[res_df["qualifies"]]["slope_mV_per_month"], errors="coerce")
print(f"\\nF3 distribution (mV/month):")
print(f"  p10={f3_raw.quantile(0.10):.2f}  p50={f3_raw.quantile(0.50):.2f}  "
      f"p90={f3_raw.quantile(0.90):.2f}  max={f3_raw.max():.2f}")
print(f"  Fixtures with positive slope (degrading direction): "
      f"{(f3_raw > 0).sum()} / {len(f3_raw)} ({100*(f3_raw>0).mean():.0f}%)")
"""))

cells.append(code("""\
# Visualise F3 slope distribution and time series for extreme fixtures
fig, axes = plt.subplots(1, 2, figsize=(13, 5))

axes[0].hist(f3_raw.dropna(), bins=25, edgecolor="black", alpha=0.75, color="darkorange")
axes[0].axvline(0, color="k", linestyle="--", alpha=0.7, label="zero drift")
axes[0].axvline(f3_raw.median(), color="r", linestyle="--",
                label=f"median {f3_raw.median():.1f} mV/mo")
axes[0].set_xlabel("F3 — Theil-Sen slope (mV/month)")
axes[0].set_ylabel("Fixture count")
axes[0].set_title("Per-fixture temporal drift in boost-voltage level")
axes[0].legend()

# Top-3 and bottom-3 slope fixtures — time series
top3 = f3_raw.nlargest(3).index.tolist()
bot3 = f3_raw.nsmallest(3).index.tolist()
for fx in top3:
    g = daily[daily["location_id"] == fx].sort_values("t_months")
    axes[1].plot(g["t_months"], g["bv_trim_mean"], marker=".", ms=4,
                 label=f"{fx} (+{f3_raw.loc[fx]:.1f}mV/mo)", alpha=0.8)
for fx in bot3:
    g = daily[daily["location_id"] == fx].sort_values("t_months")
    axes[1].plot(g["t_months"], g["bv_trim_mean"], marker=".", ms=4, linestyle="--",
                 label=f"{fx} ({f3_raw.loc[fx]:.1f}mV/mo)", alpha=0.8)
axes[1].set_xlabel("Time (months from export start)")
axes[1].set_ylabel("Daily trimmed mean boostVoltage (mV)")
axes[1].set_title("Fastest-rising and fastest-falling fixtures")
axes[1].legend(fontsize=7)
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(ART / "f3_temporal_drift.png", dpi=100)
plt.show()
print("Saved f3_temporal_drift.png")
"""))

# ── MECHANISM 4: Envelope width temporal trend ────────────────────────────
cells.append(md(r"""\
---
## Mechanism 4 — Temporal Drift in Envelope Width

### Physical mechanism

Feature 2 captures the *cross-sectional* level of envelope width. Feature 4
captures whether the envelope width is **growing over time**, independent of
its absolute level.

The growth of envelope width is tied to capacitor degradation rate:

$$\frac{d E}{d t} = \frac{d}{dt}\left(\frac{I}{f C_{\text{out}}(t)}\right)
  + \frac{d}{dt}\left(I \cdot \text{ESR}(t)\right)$$

Both $C_{\text{out}}$ and $\text{ESR}$ follow Arrhenius kinetics: the degradation
rate doubles roughly every 10 °C above the rated temperature. The key point is
that **the first sign of capacitor wear is widening ripple, before the capacitor
fails**. This means positive slope in envelope width should be a *leading*
indicator even relative to F3 (voltage level drift).

If F3 and F4 both trend upward together: voltage rise **and** instability
growth are co-occurring → strong aging signature.

If F4 is positive but F3 is near zero: instability is growing without a
mean-level shift yet → early-stage degradation.

If F3 is positive but F4 is near zero: mean level rising but envelope
stable → could be temperature drift or step change rather than LED aging.

### Directional prediction

> A degrading fixture should show **increasing envelope width over time**,
> particularly when co-occurring with rising mean level (F3).

### Assumptions required

Same as F3, plus the reporting window duration is stable over time
(otherwise window-length changes alias into apparent envelope trend).

### Feature definition

$$F_4(\text{fixture}) = \hat{\beta}_1^{(E)}\quad\text{from Theil-Sen}\quad
  E_w = \beta_0^{(E)} + \beta_1^{(E)} \cdot t_w + \varepsilon_w$$

Units: mV/month. Zero for a stable fixture; positive for growing instability.
"""))

cells.append(code("""\
# F4 — per-fixture temporal drift in median envelope width
# Same daily binning and Theil-Sen approach as F3

daily_env = bv_wide.groupby(["location_id", "date"])["envelope"].median().reset_index(name="env_median")
daily_env["t_months"] = (daily_env["date"] - t_origin).dt.total_seconds() / (3600 * 24 * 30.44)

f4_results = {}
for fx, grp in daily_env.groupby("location_id"):
    grp = grp.dropna(subset=["env_median"])
    span = grp["t_months"].max() - grp["t_months"].min()
    n = len(grp)
    if n < MIN_WINDOWS or span < MIN_SPAN_MONTHS:
        f4_results[fx] = {"n": n, "span_months": float(span), "qualifies": False,
                          "slope_mV_per_month": np.nan}
        continue
    slope, intercept, lo, hi = theilslopes(grp["env_median"].values, grp["t_months"].values)
    f4_results[fx] = {
        "n": int(n), "span_months": float(span), "qualifies": True,
        "slope_mV_per_month": float(slope),
    }

f4_df = pd.DataFrame(f4_results).T
f4_df.index.name = "location_id"
f4_raw = pd.to_numeric(f4_df[f4_df["qualifies"]]["slope_mV_per_month"], errors="coerce")

print(f"Fixtures qualifying for F4: {len(f4_raw)} / {len(f4_df)}")
print(f"\\nF4 distribution (mV/month envelope drift):")
print(f"  p10={f4_raw.quantile(0.10):.2f}  p50={f4_raw.quantile(0.50):.2f}  "
      f"p90={f4_raw.quantile(0.90):.2f}  max={f4_raw.max():.2f}")
print(f"  Fixtures with positive envelope slope: "
      f"{(f4_raw > 0).sum()} / {len(f4_raw)} ({100*(f4_raw>0).mean():.0f}%)")

# F3 vs F4 correlation — should be moderate and positive if both are aging-driven
shared = pd.concat([f3_raw, f4_raw], axis=1, keys=["F3", "F4"]).dropna()
r34, p34 = stats.spearmanr(shared["F3"], shared["F4"])
print(f"\\nF3 vs F4 Spearman rho={r34:.3f}, p={p34:.4f}")
if r34 > 0.3 and p34 < 0.05:
    print("  -> F3 and F4 co-trend: consistent with a shared aging driver")
elif abs(r34) < 0.2:
    print("  -> F3 and F4 largely independent: measure distinct phenomena")
else:
    print("  -> Weak to moderate relationship; interpret separately")
"""))

cells.append(code("""\
fig, axes = plt.subplots(1, 2, figsize=(13, 5))

axes[0].hist(f4_raw.dropna(), bins=25, edgecolor="black", alpha=0.75, color="mediumpurple")
axes[0].axvline(0, color="k", linestyle="--", alpha=0.7, label="zero drift")
axes[0].axvline(f4_raw.median(), color="r", linestyle="--",
                label=f"median {f4_raw.median():.1f} mV/mo")
axes[0].set_xlabel("F4 — Theil-Sen slope of envelope width (mV/month)")
axes[0].set_ylabel("Fixture count")
axes[0].set_title("Per-fixture temporal drift in envelope width")
axes[0].legend()

axes[1].scatter(shared["F3"], shared["F4"], alpha=0.5, s=45)
axes[1].set_xlabel("F3 — level drift (mV/month)")
axes[1].set_ylabel("F4 — envelope drift (mV/month)")
axes[1].set_title(f"F3 vs F4  (Spearman r={r34:.3f})")
axes[1].axhline(0, color="k", linestyle="--", alpha=0.3)
axes[1].axvline(0, color="k", linestyle="--", alpha=0.3)
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(ART / "f4_envelope_drift.png", dpi=100)
plt.show()
print("Saved f4_envelope_drift.png")
"""))

# ── Duty cycle filter ──────────────────────────────────────────────────────
cells.append(md("""\
---
## Liveness / Duty Cycle — Filter, Not Feature

### Why stale and lastSeen are not scored

`stale` and `lastSeen` capture communication events, not electrical state.
The spatial stale check showed that Circuit 2 is structurally ~13 pp more
stale than Circuit 1 (Mann-Whitney p < 0.0001), consistent with RF-coverage
differences — not LED aging differences.

Treating stale_frac as a scored aging feature would penalise fixtures on
a worse RF circuit for reasons unrelated to their LED health.

### Legitimate uses

1. **Sample-count filter**: fixtures with fewer than `MIN_WINDOWS` (30) daily
   boost-voltage observations are excluded from scored features (F1–F4 are
   unreliable at low N). This filter already encodes the reporting-quality
   constraint without scoring it.

2. **Score confidence qualifier**: a high degradation score from 30 observations
   is less reliable than the same score from 300. Report N alongside the score
   in Phase 3.

3. **Outlier flagging**: fixtures with stale_frac > 0.95 over the entire window
   should be flagged as "insufficient data" rather than scored at all.
"""))

cells.append(code("""\
# Liveness filter thresholds
stale_df = pd.read_csv(ART / "stale_fraction_per_fixture.csv", index_col=0)
stale_df.index.name = "location_id"

n_qualify_bv  = (res_df["qualifies"] == True).sum()
n_high_stale  = (stale_df["stale_frac"] > 0.95).sum()

print(f"Fixtures with >= {MIN_WINDOWS} boost-V windows: {n_qualify_bv}")
print(f"Fixtures with stale_frac > 0.95 (flag as insufficient data): {n_high_stale}")
print(f"\\nPer-circuit reporting quality:")
import re
def get_circuit(loc):
    m = re.match(r"yyc-c(\\d+)-r\\d+", str(loc))
    return int(m.group(1)) if m else None

stale_df["circuit"] = stale_df.index.map(get_circuit)
res_df["circuit"] = res_df.index.map(get_circuit)
print(res_df.groupby("circuit")["qualifies"].value_counts().unstack(fill_value=0))
"""))

# ── Feature summary ────────────────────────────────────────────────────────
cells.append(md("""\
---
## Phase 2 Summary — Feature Specification

| Feature | Signal | Computation | Unit | Physics | Assumptions |
|---|---|---|---|---|---|
| **F1** | Level | Trimmed mean (5–95%) of boostVoltage per fixture | mV | Sustained V_f elevation under constant CCR current | CCR current constant; no T_j correction |
| **F2** | Envelope width | Median of (boostVoltageMax − boostVoltageMin) per fixture | mV | Driver ripple and inter-chip instability | Consistent reporting window duration |
| **F3** | Level drift | Theil-Sen slope of daily trimmed mean boostVoltage vs. time | mV/month | V_f drift rate; controls for commissioning offset | CCR stability over export window; ≥30 obs over ≥2 mo |
| **F4** | Envelope drift | Theil-Sen slope of daily median envelope width vs. time | mV/month | Capacitor degradation rate (leading indicator) | Same as F3 |

All four are independent in theory; moderate F3–F4 correlation expected under
a shared aging driver. Cross-sectional features (F1, F2) cover the whole
fleet; temporal features (F3, F4) require the 30-obs / 2-month qualification.

**Indicator rests entirely on boostVoltage.** The CCR-constant-I premise is
the single load-bearing unverified assumption. Temporal features (F3, F4)
control for per-fixture baseline variation and are the strongest aging signals
available.
"""))

cells.append(code("""\
# Save feature vectors for Phase 3
features = pd.concat([f1, f2], axis=1)
features = features.join(res_df[["slope_mV_per_month"]].rename(columns={"slope_mV_per_month": "F3_slope_mV_per_month"}))
features = features.join(f4_df[["slope_mV_per_month"]].rename(columns={"slope_mV_per_month": "F4_envelope_slope_mV_per_month"}))
features = features.join(res_df[["n", "span_months", "qualifies"]].rename(columns={"n": "n_windows", "span_months": "span_months", "qualifies": "qualifies_temporal"}))
features = features.join(stale_df[["stale_frac"]])
features.index.name = "location_id"

features.to_csv(ART / "phase2_features.csv")
print(features.head(10).to_string())
print(f"\\nSaved to {ART / 'phase2_features.csv'}")
print(f"\\nNull counts:\\n{features.isnull().sum()}")
"""))

# ---------------------------------------------------------------------------
# Build notebook
# ---------------------------------------------------------------------------

nb = new_notebook(cells=cells)
nb.metadata.update({
    "kernelspec": {
        "display_name": "Python 3",
        "language": "python",
        "name": "python3",
    },
    "language_info": {"name": "python", "version": "3.12.0"},
})

NB_PATH.parent.mkdir(exist_ok=True)
nbformat.write(nb, str(NB_PATH))
print(f"Written: {NB_PATH}")
