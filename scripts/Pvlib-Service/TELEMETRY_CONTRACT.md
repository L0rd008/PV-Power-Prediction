# Pvlib-Service — Telemetry Key Contract

*Version 1.0 — 2026-04-23. This is the source of truth for all telemetry keys written by the service. **Do not rename or remove any key without a 90-day deprecation window.***

---

## Keys Written Per Plant (and Rolled Up to Parent Assets)

| Key | Type | Unit | Description | Widget use |
|---|---|---|---|---|
| `potential_power` | timeseries | **kW** | Physics-model AC output from pvlib pipeline. Primary customer-facing key. | V5 Curtailment (dataset 0 alt), Loss Attribution datasource, any future overlay widget |
| `active_power_pvlib_kw` | timeseries | **kW** | Same value — ops/diagnostic alias. Kept for internal dashboards and debugging. | Ops dashboards only |
| `total_generation_expected_kwh` | timeseries | **kWh** | Daily expected energy (written once at 01:00 UTC for prior day). | Forecast vs Actual Energy widget |
| `total_generation_expected_monthly_kwh` | timeseries | kWh | Monthly cumulative expected generation (sum from start of month to today). | Reporting |
| `total_generation_expected_yearly_kwh` | timeseries | kWh | Yearly cumulative expected generation (sum from start of year to today). | Reporting |
| `pvlib_daily_energy_kwh` | timeseries | kWh | Backwards-compatible alias for `total_generation_expected_kwh` (historical systems). | Legacy dashboards |
| `pvlib_data_source` | timeseries | string | Tier used: `"tb_station"` / `"solcast"` / `"clearsky"` / `"rollup"` | Diagnostics |
| `pvlib_model_version` | timeseries | string | Always `"pvlib-h-a3-v1"` | Diagnostics, regression detection |
| `ops_expected_unit` | timeseries | string | Always `"kW"` — declares the unit of `potential_power` for widgets that co-plot meter data | Unit normalisation |

---

## Unit Contract

The service **always writes kW** for all power keys regardless of what the plant's own `active_power` meter publishes. Some plants publish `active_power` in W; the attribute `active_power_unit` on each plant asset declares the meter's unit so widgets can normalise without hard-coding plant IDs.

### Setting the attribute (Gap 9 — H9-D)

Run the one-shot migration script (idempotent — safe to re-run):

```bash
TB_HOST=https://tb.example.com \
TB_USERNAME=admin@tenant.com \
TB_PASSWORD=<secret> \
python scripts/shared/set_active_power_unit.py

# Dry-run to preview without writing:
python scripts/shared/set_active_power_unit.py --dry-run
```

### Plant unit map (as of 2026-04-24)

| Unit | Plants |
|------|--------|
| `kW` | KSP, SSK, SOU, PSP, VPE Plant1, VPE Plant2, SON, SER, SUN, VYD, AKB Welisara 1, AKB Welisara 2, Aerosense, Mouldex 1, Mouldex 2 |
| `W`  | AKB Kelaniya, AKB Exports Mabola, Chris Logix 1, Chris Logix 2, Lina Manufacturing, Quick Tea, Harness, Flinth Admin, Mona Rathmalana, Mona Homagama, Mona Koggala, Hir Agalawaththa, Hir Kahatuduwa 1, Hir Kahatuduwa 2, Hir Kuruvita, Hir Mullaitivu, Hir Eheliyagoda, Hir Seethawaka 1, Hir Seethawaka 2, Hir Maharagama 1, Hir Vavuniya |

### Widget scaling recipe

For any widget that co-plots meter `active_power` alongside `potential_power` (always kW), add the following `postProcessingFunction` on the `active_power` datasource in ThingsBoard widget *Advanced* settings:

```javascript
// Widget datasource → postProcessingFunction
// Scale active_power to kW when the plant publishes it in watts.
// Default: "kW" (no scaling). Missing attribute → treat as kW (safe default).
var unit = entityAttributes.active_power_unit;
return (unit === "W") ? value * 0.001 : value;
```

- If `active_power_unit = "W"` → scale ×0.001 to convert to kW.
- If `active_power_unit = "kW"` (or attribute absent) → no scaling.
- The V5 Curtailment widget already queries capacity in kW; this scaling keeps all series on the same axis.

**Edge cases:**
- *Plant changes meter firmware mid-life:* operator updates the `active_power_unit` attribute; widgets pick up the new scaling automatically on the next refresh.
- *New plant not yet in the map:* attribute defaults to absent → widget assumes kW (no scaling). Add the plant to `set_active_power_unit.py` and re-run.

Recommended long-term: standardise all plants to publish `active_power` in kW in a separate firmware/gateway cleanup pass, then remove the scaling recipe.

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
| V5 Curtailment widget | `Widgets/Grid & Losses/Curtailment vs Potential Power/V5 TB Timeseries Widget` | Gap 8: natively fetches `potential_power` as Dataset 0 (dashed line). Falls back to half-sine model when no TB data. Inline ⚙ settings: **Potential Power Key** (default `potential_power`). |
| Loss Attribution widget | `Widgets/Grid & Losses/Loss Attribution` | Datasource-agnostic — wire the TB datasource key to `potential_power` (instantaneous) or `total_generation_expected_kwh` (daily energy mode). No code change required. |
| Forecast vs Actual Energy | `Widgets/Forecasts & Risk/Forecast vs Actual Energy` | Gap 8: fetches `total_generation_expected_kwh` as Dataset 5 (green dotted "Physics Expected" line). Setting: **pvlibExpectedKey** (default `total_generation_expected_kwh`). kWh auto-converted to MWh for display. |
| Portfolio Status Map | `Widgets/Portfolio/Portfolio Site Status Map` | Uses `isPlant`/`isPlantAgg` attributes for hierarchy — no telemetry keys, unaffected. |

---

## Loss Attribution Daily Keys (Phase L0 — added 2026-05-04)

Written once per calendar day at **local midnight** (ts = Unix-ms of 00:00:00 local) by `app/services/loss_rollup_job.py`. Sentinel = `-1` when data is invalid (< 360 samples, missing potential, etc.). All keys are also rolled up to `isPlantAgg` ancestor assets.

| Key | Type | Unit | Cadence | Description | Widget use |
|---|---|---|---|---|---|
| `loss_grid_daily_kwh` | timeseries | kWh | daily, ts = local midnight | `Σ max(potential − active, 0) × (1/60)` over the calendar day. Sentinel `-1`. | Loss Attribution `grid` mode |
| `loss_curtail_daily_kwh` | timeseries | kWh | daily | `Σ max(potential − max(ceiling, active), 0) × (1/60)` when `setpoint_pct < 99.5`. Sentinel `-1`. | `curtail` mode |
| `loss_revenue_daily_lkr` | timeseries | LKR | daily | `loss_grid_daily_kwh × tariff_rate_lkr` at compute time. `-1` if tariff missing. | `revenue` mode |
| `loss_curtail_revenue_daily_lkr` | timeseries | LKR | daily | `loss_curtail_daily_kwh × tariff_rate_lkr`. | `curtailRevenue` mode |
| `loss_tariff_rate_lkr_at_compute` | timeseries | LKR | daily | The exact `tariff_rate_lkr` attribute value that was used to compute the LKR losses on this day. | Auditing / History |
| `potential_energy_daily_kwh` | timeseries | kWh | daily | Σ potential. Denominator for loss rate / delta footer. | All modes (delta) |
| `exported_energy_daily_kwh` | timeseries | kWh | daily | `Σ active × (1/60)` (after W→kW unit scaling). | Delta + diagnostics |
| `loss_data_source` | timeseries | string | daily | `"ok"` / `"ok:partial"` / `"error:insufficient_samples"` / `"error:no_potential"` / `"error:no_actual"` / `"warn:no_tariff"` / `"rollup"` / `"rollup:partial"` | Diagnostics |
| `loss_model_version` | timeseries | string | daily | `"loss-rollup-v1"` | Regression detection |

**RETIRED keys (removed 2026-05-04 — never consumed by any widget; 90-day deprecation waived for keys that were never in production):**

| Key | Removed | Reason |
|---|---|---|
| `potential_energy_monthly_kwh` | 2026-05-04 | Round-1 deviation from Plan §7 (H2 decision explicitly forbade monthly/yearly precomputed keys). Zero consumers. |
| `potential_energy_yearly_kwh` | 2026-05-04 | Same as above. |

**Today-partial cadence note**: The same six `loss_*_daily_*` keys may be re-written multiple times during the current day by the today-partial cron (default every 5 min, 05:00–19:00 local). These intra-day writes carry `loss_data_source = "ok:partial"` (plant rows) or `"rollup:partial"` (ancestor assets). The finalised value written by the 00:10 cron the next day carries `"ok"` or `"warn:no_tariff"` and is authoritative. Consumers that need only finalised daily totals can filter on `loss_data_source NOT IN ('ok:partial', 'rollup:partial')`.

**Deprecation policy**: same as other keys — 90-day window before removal.

---

## Loss Attribution Lifetime Attributes (Phase L0 — added 2026-05-04)

Written to **SERVER_SCOPE** on each plant asset and rolled up to `isPlantAgg` ancestors. Updated daily by the loss-rollup cron; recomputable on demand via `POST /admin/recompute-lifetime`.

| Attribute | Type | Notes |
|---|---|---|
| `loss_grid_lifetime_kwh` | double | Cumulative gross loss kWh since `commissioning_date` |
| `loss_curtail_lifetime_kwh` | double | Cumulative curtailment loss kWh |
| `loss_revenue_lifetime_lkr` | double | Sum of historical daily LKR values (each computed at tariff in effect that day) |
| `loss_curtail_revenue_lifetime_lkr` | double | Same for curtailment revenue loss |
| `potential_energy_lifetime_kwh` | double | Cumulative potential energy kWh (denominator for loss-rate displays) |
| `exported_energy_lifetime_kwh` | double | Cumulative exported energy kWh |
| `loss_lifetime_anchor_date` | string | ISO date of the most recent day already included (e.g. `"2026-05-03"`) |
| `loss_lifetime_updated_at` | string | ISO datetime of last successful attribute write |

**Reading in widget**: `fetchAttributesWithFallback(entity, [attr_name])` — reads SERVER_SCOPE first, then SHARED_SCOPE. The widget uses the `lossLifetimeAttrPrefix` setting (default `"loss_"`) to compose the six attribute names.

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
| Unit migration script | `scripts/shared/set_active_power_unit.py` | Gap 9: one-shot idempotent script to set `active_power_unit` on all known plants |
| V5 Curtailment widget | `Widgets/Grid & Losses/Curtailment vs Potential Power/V5 TB Timeseries Widget` | Gap 8: natively fetches `potential_power` as Dataset 0 (dashed line). Falls back to half-sine model when no TB data. Inline ⚙ settings: **Potential Power Key** (default `potential_power`). |
| Loss Attribution widget | `Widgets/Grid & Losses/Loss Attribution` | Datasource-agnostic — wire the TB datasource key to `potential_power` (instantaneous) or `total_generation_expected_kwh` (daily energy mode). No code change required. |
| Forecast vs Actual Energy | `Widgets/Forecasts & Risk/Forecast vs Actual Energy` | Gap 8: fetches `total_generation_expected_kwh` as Dataset 5 (green dotted "Physics Expected" line). Setting: **pvlibExpectedKey** (default `total_generation_expected_kwh`). kWh auto-converted to MWh for display. |
| Portfolio Status Map | `Widgets/Portfolio/Portfolio Site Status Map` | Uses `isPlant`/`isPlantAgg` attributes for hierarchy — no telemetry keys, unaffected. |
