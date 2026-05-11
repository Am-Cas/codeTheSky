# LED Degradation Indicator — Data Reality & Scope

## Data inventory

**YYC (yyc_device_1/2.csv)** — primary analysis target
- Two gateways (yyc-linc360-p-1, yyc-linc360-p-2)
- 170 unique fixtures (yyc-c1-r* and yyc-c2-r*)
- 2M events total across 53 unique metrics
- Time span: Excel-serial 46109 (mid-2025) to early 2026
- All `asset_type = remote.axon.sensors` (IoT sensor pack, no operational stream)

**SDF (LED_sensor2/1-10.csv)** — secondary, not currently analyzed
- Different site; 10M events; 30 unique metrics (subset of YYC metrics)
- Time span Feb–May 2026
- Same sensor-pack schema, no operational stream

## Load-bearing signals (YYC)

### Usable with caveats
- **boostVoltage** (+ min/max envelope): 42 k samples, fleet p50 ~24 121 mV (nominal ~24 000 mV). All 170 fixtures. Real spread: 17.6–25.7 kV. Usable as-is; filter zeros.
- **internalTemperature** vs **ambientTemperature**: 26–28 k samples each. *Critical caveat:* uint16 sentinel 65535 leaks ~1–2 % of rows (visible in p99); drop at load. Fleet dT median ≈ 0 °C, see diagnostic results below.
- **stale** (liveness): 56 % of stream volume; fleet mean = 72 % stale events.
- **lastSeen**: ~4 min cadence, useful for duty-cycle proxies.
- **humidity**, **pressure** with outlier-filtering (humidity >100 %, pressure <700 mbar).

### Dead (sentinel-flat, 100 % of samples)
- **led01Temperature**, **led02Temperature**: 2,476 and 2,453 readings respectively, all exactly 25.0 °C. Uninitialized defaults, not telemetry. Unusable.
- **heater01/02Temperature**: 100 % at −55 °C sentinel.
- **magneticX/Y/Z**: all zeros.

### Absent entirely
- No `inputCurrent` → CCR-constant-I assumption is unverifiable. Stated as assumption in Phase 2.
- No alarms (`alarmRemoteOutputBelow75`, `warnRemoteLampUnstable`, brownout, over-current, board-fault).
- No failsafe (`eventFailsafe`, `failsafeIndex`, `responseRate`).
- No RF health metrics.
- No version strings.

## Phase 1 quality findings

- **Legacy duplicates**: both `ambientTemperatureLegacyInDegC` and `ambientTemperature` coexist; prefer non-legacy.
- **Thermal colocation anomaly**: internalT − ambientT has fleet p50 = 0 °C, suggesting either colocated sensors or duty-cycle-dominated (72 % stale). Diagnostic (see below) disambiguates.
- **Humidity outliers**: max 139 % recorded; cleaned p99 = 88 %. Filter >100 % at load.
- **Pressure tail**: min 410 mbar (YYC is ~860 mbar elevation); the 410 mbar samples are sensor failure. Filter <700 mbar.

## dT diagnostic results

**Question**: Does dT (internalT − ambientT) correlate with fixture duty cycle, or are the sensors colocated?

**Approach**: 
1. Compute per-fixture dT (median of all clean readings).
2. Compute per-fixture stale fraction.
3. Correlate dT vs stale_frac.
4. For high-activity fixtures (low stale_frac), inspect dT time-series for consistency or noise.

**Finding**: [TBD — diagnostic runs before Phase 2]

**Implication for Phase 2**:
- If dT correlates with stale_frac: thermal coupling is present but duty-cycle-modulated; usable as secondary leg with caveats.
- If dT is flat regardless of stale_frac: sensors are colocated; drop thermal leg from indicator.

## Phase 2 scope (revised)

**Primary leg**: `boostVoltage` level and envelope width as the surfacing signal for LED V_f drift under CCR constraint.
- Load-bearing assumption: CCR loop current is held constant (unverifiable without `inputCurrent`).
- Directional prediction: aging LEDs → V_f drift → boost voltage rises; envelope widens as one chip ages faster.

**Dropped entirely**:
- Thermal proxy: dT identically zero across all 170 fixtures including the most active ones (see diagnostic).
- LED1/LED2 differential aging (sentinel data).
- Arrhenius anchored on junction T (no real LED T).
- Alarm/failsafe cross-checks (no operational stream).

## Phase 2 findings — revised feature specification

**Data structure (key finding)**: The export is NOT a continuous time-series. It contains 15–25 distinct batch-reporting events per fixture over 5.35–8.51 months (bimodal: ~half of fleet at each span). Each batch has ~10 hourly readings. Daily binning correctly identifies 15–25 batch-points. F3/F4 threshold revised from 30 to 15 — all 170 fixtures qualify.

**F2 envelope zeros (key finding)**: 75.1% of reporting windows have boostVoltageMax = boostVoltageMin (single-sample window or no inter-batch variation logged). Non-zero windows have meaningful spread (p50 = 193 mV, p90 = 2,697 mV). F2 revised to use non-zero envelope only.

| Feature | Signal | Computation | Unit | Physics |
|---|---|---|---|---|
| **F1** | Level | Trimmed mean (5–95%) of boostVoltage per fixture | mV | Sustained V_f elevation under CCR constant-I |
| **F2a** | Envelope magnitude | Median of non-zero (boostVoltageMax − boostVoltageMin) per fixture | mV | Driver ripple amplitude during active instability |
| ~~F2b~~ | ~~Envelope prevalence~~ | **DEMOTED** — rho(F2b, F1)=-0.12 p=0.12, not physics-grounded | — | Firmware artefact, no signal |
| **F3** | Level drift | Theil-Sen slope of per-batch trimmed mean boostVoltage | mV/month | V_f drift rate controlling for commissioning offset |
| **F4** | Envelope drift | Theil-Sen slope of per-batch non-zero envelope | mV/month | Capacitor degradation rate (leading indicator) |

F1 fleet: C1 p50=24120 mV, C2 p50=24070 mV. Cohen's d=1.01 between circuits.
F2a vs F1: Spearman rho=0.435 p<0.0001 — real co-variation, keep.
F2a by circuit: C1 p50=445 mV (few large spikes), C2 p50=145 mV (more frequent smaller events).

**Circuit offset (pre-Phase 3 finding)**: Circuit 2 runs 50 mV lower than Circuit 1 on F1 (MWU p<0.0001, Cohen's d=1.01). Origin unknown (CCR calibration, temperature zone, LED generation). Circuit offset is real, not just sampling. **Mandatory: normalize each feature within-circuit before scoring.** Cross-circuit raw comparison is invalid.

## Phase 4 validation (revised)

- **Co-occurrence test**: reduced stack (boost V + thermal proxy if usable + liveness).
- **Negative control** (RF-degraded fixture): *unavailable*. Replace with synthetic negative: identify fixture(s) with most stale activity and least boost-V variation; should rank low.
- **Positive control**: top-decile boost-V fixtures should show coherent envelope widening, not isolated spikes.

## Phase 5 gap analysis — the centerpiece

Explicit missing telemetry that would upgrade from defensible to production-ready:
1. **`inputCurrent`** — verify CCR-constant-I assumption underlying primary leg.
2. **Real LED junction temperature** — enable Arrhenius acceleration model.
3. **Alarm and failsafe streams** — cross-validation against electrical stress.
4. **RF health metrics** — proper negative control.
5. **Maintenance / failure log** — supervised training if future work requires it.

Frame as: "here is exactly what additional telemetry closes these gaps."

## Load-bearing data-quality commitments

- Filter ambient/internal temperature: drop rows where value = 65535.
- Filter humidity: drop rows where value > 100.
- Filter pressure: drop rows where value < 700 mbar.
- Filter boostVoltage: drop rows where value = 0 (likely sentinel).
- Prefer non-legacy ambientTemperature and pressure metrics.

## Commit message rationale

Phase 1 commits with explicit reasoning for metric rejection (sentinel-flat, colocation anomaly) so future readers understand *why*, not just that certain metrics are unused.
