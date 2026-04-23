# Model Methods and Accuracy Documentation

Last updated: 2026-03-20
Plant config used in this repo: 55 kW rooftop example (`config/plant_config.json`)

This document is the implementation-accurate record of what was planned, what was actually built, and which metrics came from which code path.

## 1. Scope and Artifact Snapshot

Metrics below are taken from these generated files in `output/`:
- `js_physics_calibration.json`
- `js_validation_results.json`
- `test_evaluation_metrics.json`
- `benchmark_metrics.json`
- `benchmark_results.csv`
- `accuracy_report.txt`
- `surrogate_coefficients.json`
- `split_info.json`

Important: not all scripts run the exact same physics variant. The sections below explicitly map each result to the code path that produced it.

## 2. Model Variants Actually Used

| ID | Variant | Primary file(s) | Inverter efficiency | Purpose |
|---|---|---|---|---|
| M1 | Python Physics (full hourly) | `scripts/python_physics/physics_model.py` | Load-dependent curve by default (`use_efficiency_curve=true`) | Reference model and production-style physics run |
| M2 | Python Physics (daily simplified) | `physics_model.py --simplified` | Flat (`flat_efficiency`) | Fast daily estimate |
| M3 | Python Physics (daily synthetic-hourly) | `physics_model.py` daily modes without `--simplified` | Same as M1 | Full pipeline on daily inputs via synthetic hourly reconstruction |
| M4 | JS Physics (calibrated ThingsBoard model) | `scripts/js_physics/physics_model.js` | Flat 0.98 | Edge/runtime estimator calibrated to Python output |
| M5 | JS Surrogate regression | `scripts/js_surrogate/surrogate_model.js` | Implicit in fitted outputs + clipping | Ultra-fast regression approximation |
| H1 | Inline helper physics (batch/benchmark/training) | `scripts/shared/prepare_training_data.py`, `scripts/shared/run_benchmark.py`, `scripts/js_physics/generate_calibration_data.py`, `scripts/python_physics/daily_predictor.py` | Flat 0.98 | Speed and portability in helper workflows |

Key distinction:
- M1 uses the PVsyst inverter curve when enabled.
- H1 scripts use a flat inverter efficiency for simplicity/performance.

## 3. Initial Plan vs Implemented System

`docs/project_plan.md` described a single Solcast-first PVsyst-consistent Python flow.
Implementation expanded to:
- Multi-source weather/irradiance workflows (Solcast, NASA POWER, Open-Meteo, ERA5, CSV)
- Daily-data support with synthetic hourly reconstruction
- Simplified daily mode
- JS physics approximation with calibration workflow
- JS surrogate training/evaluation workflow
- Multiple benchmarking/evaluation scripts with different physics fidelity levels

## 4. Full Physics Method (M1: `physics_model.py`)

Pipeline in execution order:
1. Ingest weather/irradiance (`api`, `csv`, `csv_daily`, `nasa_power`, `nasa_power_daily`).
2. Fill missing weather via fallback chain:
   - Solcast weather if present
   - ERA5 fetch (`fetch_era5_weather`)
   - Constants (`defaults.wind_speed_ms`, `defaults.air_temp_c`)
3. Solar position via pvlib SPA.
4. POA transposition via Perez with `dni_extra`.
5. Optional far-shading multiplier.
6. AOI + IAM interpolation (PVsyst IAM table).
7. SAPM cell temperature from POA-before-IAM.
8. DC conversion with temperature coefficient and DC loss chain.
9. DC-to-AC conversion:
   - Load-dependent inverter curve by default
   - Optional flat efficiency fallback
10. Plant-level clipping and AC wiring loss.
11. Energy integration from timestep.

Configured plant parameters currently in repo:
- 8 orientations, 216 modules total
- Module area 2.556 m2
- STC efficiency 0.2153
- Gamma P -0.00340 /degC
- AC rating 55.0 kW
- DC loss factor product approximately 0.9317

## 5. Daily and Helper Variants (H1/M2/M3)

### 5.1 Synthetic hourly from daily totals (M3)
Implemented in `generate_synthetic_hourly_from_daily()`:
- Builds hourly clear-sky profile (Ineichen)
- Scales hourly profile to match daily totals
- Runs full physics chain afterward

### 5.2 Simplified daily mode (M2)
Implemented in `calculate_daily_energy_simple()`:
- Uses fixed `avg_poa_factor = 1.05`
- Uses flat inverter efficiency
- No sub-hour clipping dynamics

### 5.3 Inline helper physics (H1)
Used by training/benchmark/orchestration scripts for convenience and speed:
- Uses pvlib Perez + SAPM + IAM + losses
- Uses flat inverter efficiency (not the full load-dependent curve)
- Powers many reported benchmark artifacts

## 6. JS Physics Model (M4)

### 6.1 Calibration run (`output/js_physics_calibration.json`)
Source: `scripts/js_physics/optimize_js_params.py`
- Samples used: 50,000
- Baseline RMSE: 2.2449 kW
- Final RMSE: 1.4340 kW
- MAE: 0.6723 kW
- R2: 0.99233
- Energy error: 1.491%
- Improvement: 36.12%
- Iterations: 112
- Function evaluations: 13,677
- Runtime: 396.1 s

Calibrated parameters:
- `circumsolar_factor=0.05`
- `circumsolar_threshold=10.02`
- `aoi_threshold=0.05`
- `diffuse_weight=0.9141`
- `brightness_factor=0.0`
- `sapm_a=-2.0`
- `sapm_b=-0.03`
- `sapm_dt=5.0`

### 6.2 Post-calibration validation (`output/js_validation_results.json`)
Source: `scripts/js_physics/validate_calibration.py`
- Samples: 2,500,000
- RMSE: 1.4577 kW (2.65% of 55 kW)
- MAE: 0.6763 kW
- R2: 0.99214
- MAPE: 7.76%
- Energy error: 1.456%

## 7. JS Surrogate Model (M5)

Training source: `scripts/js_surrogate/fit_surrogate.py`
Evaluation source: `scripts/shared/evaluate_models.py`

From `output/surrogate_coefficients.json`:
- a0 = 0.342118
- a1 = 0.079839
- a2 = -1.293756e-05
- a3 = 0.007314
- a4 = -0.004785
- a5 = -0.001599

Training metrics:
- R2: 0.4214
- MAE: 9.58 kW
- RMSE: 14.08 kW
- MAPE: 116%

Test metrics from `output/test_evaluation_metrics.json`:
- R2: 0.4200
- MAE: 9.44 kW
- RMSE: 14.09 kW
- MAPE: 129.70%
- Energy error: +1.50%

Split metadata note:
- `output/split_info.json` reports 73 selected test days (day-level split in local time).
- `output/test_evaluation_metrics.json` reports 131 unique UTC-normalized days because hourly UTC timestamps can span two UTC dates for one local day.

Interpretation:
- Very weak hourly shape fidelity
- Low net energy error due cancellation of bias
- Suitable only for coarse energy totals, not hourly anomaly detection

## 8. Accuracy Results vs Actual Generation

### 8.1 Raw benchmark output (as generated)
Source: `output/benchmark_metrics.json` from `scripts/shared/run_benchmark.py`

Reported by script:
- Benchmark period label: Dec 11-15, 2025
- Total actual generation reported: 421.342 kWh

Metrics in file:
- Python Physics: MAE 154.1 kWh/day, total error +239.9%, avg time 0.740 ms
- JS Physics: MAE 169.0 kWh/day, total error +268.2%, avg time 0.026 ms
- JS Surrogate: MAE 151.3 kWh/day, total error +238.3%, avg time 0.011 ms

### 8.2 Critical caveat on benchmark totals
`benchmark_results.csv` contains only two rows with actual data (Dec 11 and Dec 12).
The script-level `total_error_pct` in `benchmark_metrics.json` uses 5 days of predictions but only 2 days of non-null actuals, which inflates total error percentages.

Overlap-only totals from `benchmark_results.csv` (2 valid days):
- Actual total: 421.342 kWh
- Python Physics total: 487.941 kWh (error +15.81%)
- JS Physics total: 535.240 kWh (error +27.03%)
- JS Surrogate total: 498.425 kWh (error +18.29%)

Conclusion: keep MAE/RMSE from benchmark for day-level error context, but treat reported `total_error_pct` in `benchmark_metrics.json` as non-comparable unless day filtering is fixed.

### 8.3 Weather-source comparison (`compare_accuracy.py`)
Source: `output/accuracy_report.txt`

Daily metrics over 10 matched days:
- NASA POWER: MAE 96.2 kWh, Bias +32.7, R2 -0.68
- Solcast: MAE 97.4 kWh, Bias +33.9, R2 -0.71
- Open-Meteo: MAE 111.4 kWh, Bias +47.9, R2 -1.55
- ERA5: MAE 96.8 kWh, Bias +33.3, R2 -0.70

Hourly metrics are based on 10 hourly aligned points in that report and should be treated as sparse-window diagnostics, not robust climatological performance.

## 9. Test-Set Provenance Notes

`output/test_evaluation_metrics.json` is generated by `scripts/shared/evaluate_models.py`.
That script creates a temporary inline JS physics implementation (`calcPhysics`) for test execution, not a direct runtime call to `scripts/js_physics/physics_model.js`.

Implication:
- Test-set JS Physics numbers are valid for that evaluation script path.
- They are not a strict one-to-one benchmark of the final calibrated ThingsBoard script file.
- For calibrated M4 parity, prefer `js_validation_results.json` metrics.

## 10. Validation Status

- PVsyst annual parity for M1: pending
- JS calibration parity vs Python labels (synthetic): completed
- Surrogate train/test split (no leakage): completed
- Actual-vs-prediction validation: limited and weather-window sensitive

## 11. Reproducibility Commands

Run from repository root unless noted.

1. Full physics run:
   - `python scripts/python_physics/physics_model.py --source nasa_power --start 20251201 --end 20251215`
2. Daily predictor with source fallbacks:
   - `python scripts/python_physics/daily_predictor.py --start 20260107 --end 20260108`
3. Compare against Meteocontrol actuals:
   - `python scripts/shared/compare_accuracy.py`
4. Prepare train/test data:
   - `python scripts/shared/prepare_training_data.py --year 2024 --test-ratio 0.2 --seed 42`
5. Fit surrogate:
   - `python scripts/js_surrogate/fit_surrogate.py --train-data data/train_data.csv`
6. Evaluate JS models on test set:
   - `python scripts/shared/evaluate_models.py --test-data data/test_data.csv`
7. Generate calibration data:
   - `python scripts/js_physics/generate_calibration_data.py --samples 100000 --seed 42`
8. Optimize JS calibration:
   - `python scripts/js_physics/optimize_js_params.py --maxiter 100`
9. Validate calibrated JS physics:
   - `python scripts/js_physics/validate_calibration.py`

## 12. Key Takeaway

The repository now contains multiple model families and multiple helper implementations. Reported accuracy depends on which code path produced the metric. Always cite both:
- the metric file
- the generating script
