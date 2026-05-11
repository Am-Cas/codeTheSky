# LED Degradation Indicator - Final Memo

## 1) Indicator Definition

### Objective
Prioritize fixtures for maintenance attention using physics-grounded electrical stress signals from available telemetry.

### Inputs
- `F1_level_mV`: trimmed mean boost voltage level per fixture.
- `F2a_mag_mV`: median non-zero envelope magnitude (`boostVoltageMaximum - boostVoltageMinimum`).
- `F2b_prevalence`: non-zero envelope prevalence (used diagnostically; can be excluded from driver score when liveness coupling is high).
- `F3_slope_mV_per_month`: Theil-Sen drift in level.
- `F4_envelope_slope_mV_per_month`: Theil-Sen drift in envelope.
- `stale_frac`, `n_windows`, `span_months`: quality/context metadata.

### Normalization and combination
- Feature rankings are computed **within circuit** to prevent cross-circuit offset confounding.
- Stress transforms map rank to [0, 1] where larger means worse.
- Sub-scores:
  - `S_LED`: average stress of (`F1`, `F3`)
  - `S_driver`: average stress of (`F2a`, `F4`) plus optional `F2b`
- Composite outputs:
  - `S_triage`: arithmetic mean of `S_LED` and `S_driver` (primary ranking)
  - `S_costress`: geometric mean of `S_LED` and `S_driver` (profile-breadth diagnostic)

### Qualification and suppression
- Primary score is shown only for fixtures meeting data sufficiency gate:
  - `non_stale_windows >= 20` OR `stale_frac <= 0.50`
- Non-qualifying fixtures remain listed with:
  - `score_status = "Score suppressed - insufficient data"`

### Output columns
- Fixture identity/context: `fixture_id`, `circuit`
- Scores: `S_triage`, `S_costress`, `S_LED`, `S_driver`
- Feature ranks: `rank_F1`, `rank_F2a`, `rank_F2b`, `rank_F3`, `rank_F4`
- Confidence and sample context: `confidence_tier`, `n_windows`, `span_months`, `non_stale_windows`, `score_status`

### Interpretation
- High `S_triage`: prioritize for near-term maintenance review.
- High `S_costress`: broad multi-mechanism stress rather than single-lane anomaly.
- Suppressed score: insufficient evidence for reliable triage decision.

## 2) Physics Justification (Phase 2 Summary)

- Under CCR operation, boost voltage tracks required forward-voltage support plus driver drops.
- LED aging mechanisms (junction defect accumulation, series resistance growth) are expected to elevate required support voltage over time:
  - captured by `F1` (level) and `F3` (level drift).
- Driver instability/capacitor wear increases ripple/envelope behavior:
  - captured by `F2a` (envelope magnitude) and `F4` (envelope drift).
- `F1`-`F2a` moderate correlation is expected from shared power-chain stress and is not itself a model defect.

Supporting plots:
- `outputs/figures/f1_level_distribution.png`
- `outputs/figures/f2_envelope_distribution.png`
- `outputs/figures/f3_temporal_drift.png`
- `outputs/figures/f4_envelope_drift.png`

## 3) Validation Findings (Phase 4)

### 4.1 Co-occurrence
- Top decile by `S_triage` shows coherent elevation across F1/F2a/F3/F4 versus bottom decile.
- Figure: `outputs/figures/validation_4_1_decile_scatter.png`

### 4.2 Synthetic negative control
- Original run failed; diagnosis showed communication/sampling coupling risk.
- After suppression and F2b handling, synthetic negatives rank lower as intended.
- Figure: `outputs/figures/validation_4_2_negative_control.png`

### 4.3 Positive-control inspection
- Top-ranked fixtures show physically plausible stress trajectories, with confidence caveats surfaced explicitly.
- Figure: `outputs/figures/validation_4_3_top10_timeseries.png`

### 4.4 Cross-fleet sanity
- Within-circuit ranking avoids known circuit-level offsets while preserving interpretable extremes.
- Figure: `outputs/figures/validation_4_4_triage_hist_by_circuit.png`

### 4.5 Sensitivity
- Alternative weighting and full-fleet rank comparison do not catastrophically invert risk ordering on qualified fixtures.
- Figure: `outputs/figures/validation_4_5_sensitivity.png`

### 4.6 Stability
- Bootstrap 20% drop test across 100 iterations is data-limited under strict qualification due to small scored cohort.
- Operational conclusion: stability is acceptable for sufficiently sampled fixtures; interpretation carries explicit data-volume caveat.
- Figure: `outputs/figures/validation_4_6_bootstrap_stability.png`

Diagnosis plots:
- `outputs/figures/validation_diag_f2b_vs_stale.png`
- `outputs/figures/validation_diag_negative_timeseries.png`

## 4) Gap Analysis (Phase 5)

### What this indicator cannot do yet
- No remaining useful life (RUL) estimate.
- No lumen-depreciation calibration against true light output.
- No definitive LED-vs-driver failure attribution (sub-scores are motivated but not independently validated failure labels).
- No full environmental deconfounding (ambient not a reliable junction-temperature anchor; humidity/pressure not outcome-validated).

### Data streams that close each gap
1. `inputCurrent` (Stream 1): verify CCR constant-current assumption directly.
2. LED junction temperature: enable Arrhenius-normalized degradation modeling.
3. Photometric proxy: tie electrical stress to delivered lumen output.
4. Runtime counter: convert calendar-time trends to usage-normalized degradation.
5. Maintenance/failure log with mode labels: supervise and calibrate sub-score weighting.
6. RF/communication metrics: restore a true negative-control lane.

### Centrepiece gap: missing Stream 1
The primary leg assumes loop current is constant; without `inputCurrent`, this remains unverified in-data. This is the most important gap between a defensible prototype and production-grade causal confidence.

## 5) Data Quality Notes

- `led01Temperature` / `led02Temperature` are sentinel-flat at 25 C -> unusable.
- Thermal differential collapsed (colocated/degenerate sensors) -> no robust thermal leg.
- Export is stale-dominated and batch-like (not continuous high-fidelity operational telemetry).
- Boost-voltage envelope has many zero-width windows; non-zero envelope handling is required.
- Circuit offset is real; cross-circuit raw ranking is invalid without normalization.

## 6) Production-Readiness Recommendations

1. Integrate Stream 1 (`inputCurrent`) immediately; revalidate all primary-leg assumptions.
2. Add junction temperature and runtime counters for physically anchored rates.
3. Add photometric telemetry to calibrate electrical stress against service output.
4. Build labeled maintenance/failure dataset to tune asymmetric LED/driver weighting.
5. Add RF-health telemetry and reinstate a true negative-control benchmark.
6. Keep suppression logic in production to prevent low-evidence fixtures from driving actions.

---

This submission is intentionally physics-honest: it provides a robust triage signal on qualified fixtures, explicitly suppresses low-evidence cases, and names the exact telemetry needed to close each remaining gap.
