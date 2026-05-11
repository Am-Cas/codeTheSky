# Phase 1 — Findings (stop point)

## Scope of the export vs. the brief

| Brief said | Export actually has |
|---|---|
| 127 fixtures | **170** (gw1: 86, gw2: 84) |
| ~92 metrics per fixture | **53** metrics total across the fleet |
| Two streams: operational (Stream 1) + IoT sensors (Stream 2) | **Stream 2 only** — every row is `asset_type = remote.axon.sensors` |
| Export 30 Mar 2026 | Timestamps span ~mid-2025 to early 2026 (Excel-serial range 46109; full span confirmed in `phase1_summary.json`) |

The single most consequential discovery: **Stream 1 is absent**. Every metric in this dump is a sensor-pack reading.

## What survives, what doesn't

**Usable for physics:**
- `boostVoltage` and its min/max envelope — n = 42 k, fleet p50 = 24 121 mV (matches the documented ~24 000 mV nominal), max 25 745 mV. All 170 fixtures have ≥ 50 samples.
- `ambientTemperature`, `internalTemperature` and their envelopes — n ≈ 27 k each. *Caveat:* uint16 sentinel 65535 leaks in (1–2 % of rows; visible in p99 = 65533). Drop before use.
- `humidity` (mild quality issues — max 101 % after a 139 % outlier was dropped).
- `pressure` (range 410–1 093 mbar; YYC elevation ~860 mbar is plausible; the 410 mbar tail is sensor failure, not weather).
- `stale`, `lastSeen` — liveness only.

**Dead-on-arrival (sentinel-flat, 100 % of samples):**
- `led01Temperature`, `led02Temperature` — every one of ~2 400 readings per channel = exactly **25 °C**. No variance, no fixtures off-trend. These are uninitialized defaults, not telemetry. The **LED1/LED2 differential aging argument cannot be built from this export.**
- `heater01Temperature`, `heater02Temperature` — 100 % at −55 °C.
- `magneticX/Y/Z` — all zero across the fleet.

**Absent entirely (named in the brief, not in either file):**
- `inputCurrent` — the CCR loop current. **Without it, the CCR series-loop argument (current fixed → V rises to compensate aging) becomes an assumption, not a measurement.**
- All alarms: `alarmRemoteOutputBelow75`, `warnRemoteLampUnstable`, brownout, over-current, board-fault.
- `eventFailsafe`, `failsafeIndex`, `responseRate`, version strings, RF health metrics.

## What this breaks in the proposed plan

| Phase 2 mechanism | Status with this export |
|---|---|
| LED V_f drift surfacing as `boostVoltage` rise under CCR constraint | **Still buildable** — boostVoltage is the strongest signal we have. The CCR-constant-I premise becomes an unverifiable assumption. |
| Driver thermal coupling via internal vs ambient `dT` | **Compromised.** Fleet `internalT − ambientT` is p10 / p50 / p90 = −1.5 / 0 / 0 °C. The two sensors track each other to within ±1 °C across all 170 fixtures, which is mechanically wrong for a powered LED driver. Likely the "internal" sensor is colocated with "ambient" on the same board, *or* fixtures spend the bulk of reporting time un-energized. With stale fraction averaging 72 %, both are plausible. |
| LED1 vs LED2 differential aging | **Impossible** — both channels are sentinel-25 °C. |
| Arrhenius acceleration anchored on junction T | **Impossible** without a real LED T. We could substitute internalT, but see row above. |
| Min/max envelope as instability proxy | **Usable for boostVoltage**, mostly broken for temperatures (sentinels widen the envelopes artificially). |
| Duty-cycle inference from `lastSeen`, `stale`, `eventFailsafe`, `failsafeIndex`, `responseRate` | **Reduced.** Only `stale` (72 % fleet mean) and `lastSeen` survive. No failsafe / responseRate. |

## Phase 4 validation implications

- **Co-occurrence test** is doable but weaker: stacked stress signals reduce to (high boost V) + (envelope widening) + (anomalous stale pattern). The LED-temp leg of the stack is gone.
- **Negative control** — *"the documented RF-degraded-but-electrically-nominal fixture should score low"* — **cannot run.** RF metrics are not in the export. I'd need either the fixture ID called out by hand, or the Stream 1 dump.
- **Alarm cross-check** — cannot run, no alarms.

## Other data-quality notes worth recording

- Legacy duplicates confirmed: `ambientTemperatureLegacyInDegC` (26 897 events) and `pressureLegacyInMbar` (29 476 events) coexist with their non-legacy counterparts. The legacy series has ~4 % fewer events but otherwise tracks. Prefer the non-legacy versions.
- 56 % of all rows in the combined stream are `metric = stale`. Of those, 72 % carry value 1. The fleet spends most reporting cycles in stale state.
- Sampling cadence (per fixture, per metric) is ~1 h median for the sensor-pack metrics; `lastSeen` updates every ~4 min.
- Saturation rate on `ambientTemperature` at 65535: ~1.3 % of clean rows live above p99 = 65533, i.e., the sentinel rate sits near 1 %. Filter at load.

## Stop-and-ask

Three forks. Which do you want?

1. **Ship a Stream-2-only indicator.** Drop the LED-T-based and inputCurrent-based legs from Phase 2; lean on boostVoltage (level + envelope width), thermal proxy (best effort, given the sensor colocation issue), and liveness. Carry the missing-Stream-1 caveats explicitly into Phase 5. This is what the export actually supports.
2. **Get the Stream 1 export first.** Re-do exploration once `inputCurrent`, alarms, failsafe, RF and version are in hand. Then Phase 2 can be built as originally specified.
3. **Hybrid: I push ahead on (1) for the boostVoltage / envelope side now, and you check whether Stream 1 is retrievable in parallel.**

I also need the **ID of the RF-degraded-but-electrically-nominal fixture** the brief mentions if you want the negative control done — there's no way to recover that from Stream 2 alone.
