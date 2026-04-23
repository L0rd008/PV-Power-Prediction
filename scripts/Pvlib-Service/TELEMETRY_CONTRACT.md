# Pvlib-Service — Telemetry Key Contract

*Version 1.0 — 2026-04-23. This is the source of truth for all telemetry keys written by the service. **Do not rename or remove any key without a 90-day deprecation window.***

---

## Keys Written Per Plant (and Rolled Up to Parent Assets)

| Key | Type | Unit | Description | Widget use |
|---|---|---|---|---|
| `potential_power` | timeseries | **kW** | Physics-model AC output from pvlib pipeline. Primary customer-facing key. | V5 Curtailment (dataset 0 alt), Loss Attribution datasource, any future overlay widget |
| `active_power_pvlib_kw` | timeseries | **kW** | Same value — ops/diagnostic alias. Kept for internal dashboards and debugging. | Ops dashboards only |
| `total_generation_expected_kwh` | timeseries | **kWh** | Daily expected energy (written once at 01:00 UTC for prior day). | Forecast vs Actual Energy widget |
| `pvlib_daily_energy_kwh` | timeseries | **kWh** | Same value — ops alias for `total_generation_expected_kwh`. | Ops dashboards only |
| `pvlib_data_source` | timeseries | string | Tier used: `"tb_station"` / `"solcast"` / `"clearsky"` / `"rollup"` | Diagnostics |
| `pvlib_model_version` | timeseries | string | Always `"pvlib-h-a3-v1"` | Diagnostics, regression detection |
| `ops_expected_unit` | timeseries | string | Always `"kW"` — declares the unit of `potential_power` for widgets that co-plot meter data | Unit normalisation |

---

## Unit Contract

The service **always writes kW** for power keys regardless of what the plant's own `active_power` meter publishes. Some plants publish `active_power` in W (see asset hierarchy notes).

For widgets that co-plot meter `active_power` and `potential_power`:
- If plant attribute `active_power_unit = "W"`: apply a ×0.001 scale on the meter datasource in widget config.
- If plant attribute `active_power_unit = "kW"` (default): no scaling needed.

Recommended: standardise all plants to publish `active_power` in kW in a separate cleanup pass.

---

## Enabling pvlib for a Plant (`pvlib_enabled` Attribute)

Set the following **SERVER_SCOPE** attribute on any plant asset to opt it into pvlib computation:

| Key | Type | Value |
|---|---|---|
| `pvlib_enabled` | boolean | `true` |

The scheduler discovers plants by BFS from the configured root assets, filtering for `isPlant == true AND pvlib_enabled == true`. Plants without this flag are **silently skipped** (a `skipped: pvlib_enabled=false` log line is emitted).

**Rollout sequence (Phase 7):**
1. Set `pvlib_enabled = true` on KSP_Plant only. Monitor 48 h.
2. Add SOU_Plant, SSK_Plant (have weather stations).
3. Add remaining plants (will fall to Solcast or clear-sky tier).
4. Enable parent roll-ups: once all children are stable, the parent assets receive summed `potential_power` automatically.

---

## Required Plant Attributes (Minimum for pvlib_enabled = true)

The service reads these **SERVER_SCOPE** attributes from each plant asset:

### Option A — Single JSON blob (preferred, matches `kebithigollewa_pvlib_config.json` structure)

| Key | Type | Description |
|---|---|---|
| `orientations` | JSON array | `[{"name":"Main","tilt":8,"azimuth":0,"module_count":23296,"use_measured_poa":true}]` |
| `module` | JSON | `{"area_m2":2.5833,"efficiency_stc":0.2248,"gamma_p":-0.0029}` |
| `inverter` | JSON | `{"ac_rating_kw":10000,"dc_threshold_kw":0,"use_efficiency_curve":true,"efficiency_curve_kw":[...],"efficiency_curve_eta":[...],"flat_efficiency":0.9855}` |
| `iam` | JSON | `{"angles":[0,40,50,60,70,75,80,85,90],"values":[1,1,1,1,1,0.984,0.949,0.83,0]}` |
| `station` | JSON | `{"ghi_key":"wstn1_horiz_irradiance","poa_key":"wstn1_tilted_irradiance","air_temp_key":"wstn1_temperature_ambient","wind_speed_key":"wstn1_wind_speed","freshness_minutes":10,"sanity_max_ghi_wm2":1400,"sanity_max_poa_wm2":1500}` |
| `defaults` | JSON | `{"wind_speed_ms":1.0,"air_temp_c":27.96}` |
| `latitude` | double | Decimal degrees |
| `longitude` | double | Decimal degrees |
| `altitude_m` | double | Metres |
| `timezone` | string | IANA tz, e.g. `"Asia/Colombo"` |
| `thermal_model` | string | `"open_rack_glass_glass"` |
| `soiling` | double | Fraction, e.g. `0.03` |
| `lid` | double | Fraction |
| `module_quality` | double | Fraction (negative = gain) |
| `mismatch` | double | Fraction |
| `dc_wiring` | double | Fraction |
| `ac_wiring` | double | Fraction |
| `albedo` | double | Fraction |
| `far_shading` | double | Multiplier (`1.0` = no shading) |
| `isPlant` | boolean | `true` |
| `pvlib_enabled` | boolean | `true` |

### Option B — Flat attributes (legacy, auto-detected by config parser)

All keys above as individual SERVER_SCOPE attributes. The service falls back to this if no `pvlib_config` blob is present.

---

## Plant Hierarchy and Multi-Parent Dedup

The hierarchy in ThingsBoard has some plants (KSP, SOU, SSK) under **multiple parent aggregation assets** (e.g., both "SCADA Power Plants" and "Windforce Groundmount Plants"). The service handles this via dedup-aware roll-up (P3-B):

- Each plant's `potential_power` is computed exactly once.
- Each parent receives a summed `potential_power` that counts each unique child plant exactly once.
- A plant appearing under both parent A and parent B contributes its output to both parents independently (correct for independent regional totals).

If you need a non-double-counted global total, sum `potential_power` across leaf plants only (not aggregation assets).

---

## Key Retirement Policy

Before removing any key listed above:
1. Search `M:\Documents\Projects\MAGICBIT\Widgets\` for widget code referencing that key.
2. Add a 90-day deprecation notice to this document.
3. Remove after confirming zero dashboard usage.

The `pvlib_*` ops aliases (`active_power_pvlib_kw`, `pvlib_daily_energy_kwh`) may be retired after the primary keys (`potential_power`, `total_generation_expected_kwh`) have been stable for 90 days in production.

---

## Related Systems

| System | Location | Key interaction |
|---|---|---|
| Physics config | `config/kebithigollewa_pvlib_config.json` | Template for plant attributes; values stored in TB SERVER_SCOPE |
| Validation script | `scripts/shared/validate_pvlib.py` | Phase 1 accuracy benchmark vs actual `EnergyMeter_active_power` |
| V5 Curtailment widget | `Widgets/Grid & Losses/Curtailment vs Potential Power/V5 TB Timeseries Widget` | Reads `active_power` (meter). Can be configured to overlay `potential_power` as a secondary datasource. |
| Loss Attribution widget | `Widgets/Grid & Losses/Loss Attribution` | Configurable datasource key — point at `potential_power` for expected-vs-actual loss calc. |
| Forecast vs Actual Energy | `Widgets/Forecasts & Risk/Forecast vs Actual Energy` | Reads `total_generation` (actual) and `forecast_p50_daily`. Can also reference `total_generation_expected_kwh` for pvlib expected. |
| Portfolio Status Map | `Widgets/Portfolio/Portfolio Site Status Map` | Uses `isPlant`/`isPlantAgg` attributes for hierarchy — no telemetry keys, unaffected. |
