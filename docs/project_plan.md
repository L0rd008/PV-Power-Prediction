# Project Methodology (Code-Aligned)

Last verified against code: 2026-03-20  
Primary production model: `scripts/python_physics/physics_model.py`

This document is the authoritative methodology for the irradiance-to-generation chain implemented in code.

## 0) Scope and Decision

Final chosen model for irradiance -> generation:
- `scripts/python_physics/physics_model.py`

Everything else (JS physics, surrogate, inline helper variants in benchmarking/training scripts) is considered exploratory or operational approximation, not the final reference methodology.

## 1) Configuration Source of Truth

All plant parameters are loaded from:
- `config/plant_config.json`

Key parameter groups used directly in code:
- `location`: `lat`, `lon`, `altitude_m`, `timezone`
- `orientations`: tilt/azimuth/module_count per roof plane
- `module`: `area_m2`, `efficiency_stc`, `gamma_p`
- `inverter`: `ac_rating_kw`, `dc_threshold_kw`, inverter efficiency mode/curve
- `losses`: soiling, LID, module_quality, mismatch, dc_wiring, ac_wiring, albedo, far_shading
- `iam`: AOI angle table and IAM values
- `thermal_model`: SAPM mounting preset
- `defaults`: fallback weather constants

## 2) Supported Input Modes

`physics_model.py` supports five source modes:
- `--source api` (Solcast)
- `--source csv` (hourly local CSV)
- `--source nasa_power` (hourly NASA POWER)
- `--source csv_daily` (daily local CSV)
- `--source nasa_power_daily` (daily NASA POWER)

Daily modes have two branches:
- Default branch: synthetic-hourly reconstruction + full physics chain
- `--simplified` branch: daily simplified energy model (faster, less accurate)

## 3) Weather and Timestamp Handling

### 3.1 Time basis
- Internal model calculation is performed on UTC-indexed timestamps.
- Output is converted to local timezone (`config.location.timezone`) for reporting/export.

### 3.2 Missing weather fallback
If `air_temp` or `wind_speed` are missing:
1. Attempt ERA5 fetch (`fetch_era5_weather`)
2. If ERA5 unavailable, inject constants:
   - `air_temp = defaults.air_temp_c`
   - `wind_speed = defaults.wind_speed_ms`

### 3.3 Interval regularization
- Script checks monotonic timestamps and sorts if needed.
- If interval is irregular, data is resampled to hourly nearest-neighbor before physics computation.

## 4) Physics Chain (Hourly Reference Path)

For each timestamp and each orientation:

### 4.1 Solar geometry
Computed using pvlib location solar position:
- solar zenith, solar azimuth

### 4.2 POA transposition (Perez)
`pvlib.irradiance.get_total_irradiance(..., model="perez", dni_extra=...)`

Total POA conceptually:
- `POA = POA_beam + POA_diffuse + POA_ground`

### 4.3 Far shading
Applied multiplicatively before IAM:
- `POA_shaded = POA_raw * far_shading`
- `far_shading = 1.0` means no shading loss.

### 4.4 AOI + IAM
- AOI from pvlib geometry
- IAM from piecewise linear interpolation of PVsyst IAM table
- Optical POA:
  - `POA_optical = POA_shaded * IAM(AOI)`

### 4.5 Cell temperature (SAPM)
Cell temperature is computed with SAPM using POA before IAM:
- `T_cell = sapm_cell(POA_shaded, T_air, v_wind, a, b, deltaT)`

Important implementation choice:
- Thermal model uses `POA_shaded` (not `POA_optical`) to stay consistent with intended loss ordering.

### 4.6 DC power model
Per-area DC power:
- `P_DC_m2 = POA_optical * eta_STC / 1000`

Temperature correction:
- `P_DC_m2_temp = P_DC_m2 * (1 + gamma_p * (T_cell - 25))`

### 4.7 DC loss chain
Loss factor product in fixed order:
- soiling -> LID -> module_quality -> mismatch -> dc_wiring

Combined factor in code:
- `dc_loss_factor = (1-soiling)*(1-LID)*(1-module_quality)*(1-mismatch)*(1-dc_wiring)`

Then:
- `P_DC_m2_eff = P_DC_m2_temp * dc_loss_factor`

Notes:
- `module_quality` may be negative (gain), so `(1 - module_quality)` can exceed 1.

### 4.8 Area scaling per orientation
- `area_i = module_count_i * module_area`
- `P_DC_i = P_DC_m2_eff * area_i`

### 4.9 Inverter DC threshold
If configured (`dc_threshold_kw > 0`):
- `P_DC_i = 0` when below threshold

### 4.10 DC -> AC conversion
Two available modes in code:
- Load-dependent efficiency curve (default when enabled):
  - `eta_inv = interp(P_DC_i, curve_kw, curve_eta)`
- Flat efficiency fallback:
  - `eta_inv = flat_efficiency`

Then:
- `P_AC_i = P_DC_i * eta_inv`

### 4.11 Plant aggregation and clipping
- Sum all orientations first:
  - `P_AC_plant = sum_i P_AC_i`
- Apply clipping once at plant level:
  - `P_AC_clipped = min(P_AC_plant, inverter_ac_rating_kw)`

### 4.12 AC wiring loss
After clipping:
- `P_AC_net = P_AC_clipped * (1 - ac_wiring)`

### 4.13 Energy integration
If timestep is `dt_hours`:
- `E_kWh = sum_t (P_AC_net(t) * dt_hours)`

## 5) Daily-Input Handling

## 5.1 Synthetic-hourly branch (default)
Function: `generate_synthetic_hourly_from_daily()`

Per day:
1. Generate hourly clear-sky profile (Ineichen)
2. Compute daily clear-sky totals
3. Scale hourly GHI/DNI/DHI to match daily totals
4. Use daily temperature/wind as constant within the day
5. Run full hourly physics chain from Section 4

## 5.2 Simplified daily branch (`--simplified`)
Function: `calculate_daily_energy_simple()`

Approximate approach:
- Uses fixed average POA factor (`avg_poa_factor = 1.05`)
- Uses temperature factor with `gamma_p`
- Uses loss product and flat inverter efficiency
- Applies daily cap based on `24 * ac_rating_kw`

This branch is intentionally lower fidelity.

## 6) Validation Requirements

Recommended verification against PVsyst (same weather basis where possible):
- Annual POA irradiation
- Annual DC energy
- Annual AC energy (E_Grid)
- Performance ratio trend

Operational comparison against SCADA actuals should clearly state:
- weather source used
- evaluation window size
- overlap completeness (missing actual days can distort totals)

## 7) Explicitly Out of Final Methodology

Not part of the final production methodology despite being present in repository:
- `scripts/js_physics/physics_model.js` (calibrated approximation for ThingsBoard)
- `scripts/js_surrogate/surrogate_model.js` (regression surrogate)
- Inline simplified physics copies inside helper scripts:
  - `scripts/shared/prepare_training_data.py`
  - `scripts/shared/run_benchmark.py`
  - `scripts/python_physics/daily_predictor.py`

These are useful for experimentation/operations benchmarking but not the final reference math path.

## 8) One-Page Formula Summary

For each orientation `i` and timestamp `t`:

1. `POA_i = Perez(GHI,DNI,DHI,solar_geom,tilt_i,az_i,albedo)`  
2. `POA_shaded_i = POA_i * far_shading`  
3. `POA_opt_i = POA_shaded_i * IAM(AOI_i)`  
4. `T_cell_i = SAPM(POA_shaded_i, T_air, v_wind)`  
5. `P_DC_m2_i = POA_opt_i * eta_STC / 1000`  
6. `P_DC_m2_temp_i = P_DC_m2_i * (1 + gamma_p * (T_cell_i - 25))`  
7. `P_DC_m2_eff_i = P_DC_m2_temp_i * dc_loss_factor`  
8. `P_DC_i = P_DC_m2_eff_i * area_i`  
9. `P_AC_i = P_DC_i * eta_inv(P_DC_i)`  

Plant level:
- `P_AC_plant = sum_i P_AC_i`  
- `P_AC_clip = min(P_AC_plant, AC_rating)`  
- `P_AC_net = P_AC_clip * (1 - ac_wiring)`  
- `E = sum_t P_AC_net(t) * dt_hours`

---

If this file and code diverge, code is authoritative and this file must be updated immediately.
