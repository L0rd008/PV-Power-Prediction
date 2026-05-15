# Pvlib-Service — Telemetry Key Contract

*Version 1.4 — 2026-05-15. This is the source of truth for all telemetry keys written by the service. **Do not rename or remove any key without a 90-day deprecation window.***

> **v1.4 changes** (Phase 4): New `revenue_job.py` writes 5 LKR revenue timeseries keys per plant. New per-year P-value timeseries (`p50_energy_annual`, `p90_energy_annual`, `p95_energy_annual`) written by `pvalue_job.py` for historical revenue computation. New `pvlib_services` JSON attribute gates per-plant service opt-in/out across all writers. New `commissioning_date`, `onboarding_completed`, `onboarding_completed_at`, `tariff_rate_lkr` (now read by `revenue_job.py` in addition to `loss_rollup_job.py`) attributes. Loss-rollup integration upgraded to native-cadence trapezoidal method (fixes ~5× under-count on 5-min-cadence plants). Plant-attrs TTL cache in `ForecastService` (300 s default).

> **v1.3 changes** (Phase 1.5 + Phases 2/3): `daily_job.py` gains W→kW unit scaling (`active_power_unit` attr) and multi-key support (`actual_power_keys` CSV attr), mirroring `loss_rollup_job.py`. Per-plant timezone (`timezone` attr) now drives daily-record `ts` in both jobs. New §: "Plant SERVER_SCOPE Attributes Read by Service". Daily roll-up rows are stamped at each plant's local midnight (previously always service TZ midnight — no functional change for single-TZ fleets). `set_active_power_unit.py` deprecated — use `tb_config_loader.py` instead.

> **v1.2 changes**: `pvalue_job.py` upgraded to `pvalue-daily-v2` (per-calendar-day P50/P90/P95 percentiles — Phase 2). FDI widget `actualDailyKey` corrected to `actual_daily_energy_kwh`. FDI widget now uses generic `forecastDailyKey` for 3-instance (P50/P90/P95) dashboard pattern. FvA widget default `viewMode` changed to `ytd_weekly` (52 weekly data points with month-name labels, 7-day sensitivity).

### Placeholder Definitions

Command examples in this document use angle brackets (`<...>`) for values you must supply:

| Placeholder | What to replace it with |
|---|---|
| `<secret>` | The password for your ThingsBoard user account (`TB_USERNAME`). |
| `<uuid>` / `<id>` / `<asset_uuid>` | A specific ThingsBoard asset or device UUID. |
| `<root_uuid>` | The ThingsBoard UUID of your root asset. |


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
TB_HOST=https://windforce.thingsnode.cc \
TB_USERNAME=PVLib-Service@thingsnode.cc \
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
| Unit migration script | `scripts/shared/set_active_power_unit.py` | Gap 9: one-shot idempotent script to set `active_power_unit` on all known plants |
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
| `loss_data_source` | timeseries | string | daily | `"ok"` / `"ok:partial"` / `"error:insufficient_samples"` / `"error:no_potential"` / `"error:no_actual"` / `"warn:no_tariff"` / `"warn:no_tariff:partial"` / `"rollup"` / `"rollup:partial"` | Diagnostics |
| `loss_model_version` | timeseries | string | daily | `"loss-rollup-v1"` | Regression detection |

**RETIRED keys (removed 2026-05-04 — never consumed by any widget; 90-day deprecation waived for keys that were never in production):**

| Key | Removed | Reason |
|---|---|---|
| `potential_energy_monthly_kwh` | 2026-05-04 | Round-1 deviation from Plan §7 (H2 decision explicitly forbade monthly/yearly precomputed keys). Zero consumers. |
| `potential_energy_yearly_kwh` | 2026-05-04 | Same as above. |

**Today-partial cadence note**: The same daily keys may be re-written multiple times during the current day by the today-partial cron (default every 5 min, 05:00–19:00 local). These intra-day writes carry `loss_data_source = "ok:partial"` / `"warn:no_tariff:partial"` (plant rows) or `"rollup:partial"` (ancestor assets). The finalised value written by the 00:10 cron the next day carries `"ok"` or `"warn:no_tariff"` and is authoritative. Consumers that need only finalised daily totals can filter on `loss_data_source NOT IN ('ok:partial', 'warn:no_tariff:partial', 'rollup:partial')`.

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


## P-Value Forecast Keys

Written by `app/services/pvalue_job.py`. Source: PVGIS-ERA5 multi-year historical simulation (2005–2023) → monthly percentile calculation → flat daily derivation. Triggered annually (Jan 1, 03:00 local) and on-demand via `/admin/run-pvalues`.

> **Unit note:** Daily and monthly P-value keys are in **MWh** — consistent with the widget display layer, which expects MWh and does not divide. Annual `*_energy` attributes are in **kWh** — consistent with all other energy attributes on the plant asset.

### Daily Timeseries

Written as 365 rows per plant per year. Timestamp = local midnight of each calendar day.

| Key | Unit | Description | Widget |
|---|---|---|---|
| `forecast_p50_daily` | MWh | Median (P50) expected daily energy. 50% of historical days exceeded this. | Forecast vs Actual Energy (P50 line) |
| `forecast_p90_daily` | MWh | P90 expected daily energy. Only 10% of historical days fell below this — conservative lower bound. | Forecast vs Actual Energy (P90 band lower edge) |
| `forecast_p95_daily` | MWh | P95 expected daily energy. Only 5% of historical days fell below this — risk threshold. | Forecast vs Actual Energy (P95 outer band) |

### Monthly Timeseries

Written as 12 rows per plant per year. Timestamp = local midnight of 1st of each month.

| Key | Unit | Description | Widget |
|---|---|---|---|
| `forecast_p50_monthly` | MWh | Median expected monthly energy. | Forecast vs Actual Energy (monthly bar reference) |
| `forecast_p90_monthly` | MWh | P90 expected monthly energy. | Forecast vs Actual Energy (risk threshold) |
| `forecast_p95_monthly` | MWh | P95 expected monthly energy. | Forecast vs Actual Energy (outer risk band) |

### Weekly Timeseries

Written as 52 rows per plant per year. Timestamp = local midnight of each week's start (7-day buckets from Jan 1).

| Key | Unit | Description | Widget |
|---|---|---|---|
| `forecast_p50_weekly` | MWh | Median expected weekly energy. Sum of 7 daily P50 values. | Forecast vs Actual Energy (YTD weekly default mode — P50 line) |
| `forecast_p90_weekly` | MWh | P90 expected weekly energy. Sum of 7 daily P90 values. | Forecast vs Actual Energy (P90 band) |
| `forecast_p95_weekly` | MWh | P95 expected weekly energy. Sum of 7 daily P95 values. | Forecast vs Actual Energy (P95 outer band) |

### MTD Rolling Timeseries

Written as 365 rows per plant per year. Each row = cumulative P-value sum from 1st of month through that day.

| Key | Unit | Description | Widget |
|---|---|---|---|
| `forecast_p50_mtd` | MWh | Cumulative P50 month-to-date expected energy. | FDI card (P50 MTD instance) |
| `forecast_p90_mtd` | MWh | Cumulative P90 month-to-date expected energy. | FDI card (P90 MTD instance) |
| `forecast_p95_mtd` | MWh | Cumulative P95 month-to-date expected energy. | FDI card (P95 MTD instance) |

### Annual SERVER\_SCOPE Attributes

Written once per plant per run. These supersede any manually set values.

| Key | Unit | Description | Widget |
|---|---|---|---|
| `p50_energy` | kWh | Annual P50 energy yield (true 50th percentile of 19-year annual totals). | FDI card (derived-mode denominator) |
| `p90_energy` | kWh | Annual P90 energy yield (10th percentile). | Risk reporting |
| `p95_energy` | kWh | Annual P95 energy yield (5th percentile). | Risk reporting |

### Diagnostic Attributes (also written to SERVER\_SCOPE)

| Key | Type | Description |
|---|---|---|
| `pvalue_model_version` | string | Algorithm version tag — `"pvalue-monthly-v1"`. Increment when algorithm changes. |
| `pvalue_updated_at` | string | ISO-8601 UTC timestamp of the last successful pvalue job run for this plant. |
| `pvalue_target_year` | int | Calendar year the daily/monthly telemetry was stamped against. |

### Monotonicity Contract

The job enforces and logs a warning if violated:

```
p50_energy ≥ p90_energy ≥ p95_energy   (always)
forecast_p50_daily ≥ forecast_p90_daily ≥ forecast_p95_daily   (per day)
```

Any violation indicates bad PVGIS data for that grid cell — inspect logs and re-run after verifying PVGIS availability.

### FDI MTD Derivation (widget-side)

The FDI card v4.0 computes month-to-date deviation by fetching `forecastDailyKey` (e.g. `forecast_p50_daily`) for the current month and summing — it does **not** use the pre-computed `forecast_p50_mtd` key. This gives the widget flexibility to work with any P-level (P50/P90/P95) via the `forecastDailyKey` setting.

> **Note:** The `forecast_p50/p90/p95_mtd` keys **are** still written by `pvalue_job.py` (see §MTD Rolling Timeseries above) and are available for other consumers, but the current FDI v4.0 widget sums daily keys client-side instead.

```
FDI_MTD% = (Σ actual_kwh[day 1..today] − Σ p50_daily_kwh[day 1..today])
            / Σ p50_daily_kwh[day 1..today] × 100
```

Note: `forecast_p50_daily` is in MWh; the widget converts to kWh (×1000) before summing to match `actual_daily_energy_kwh` units.

---

## Plant SERVER\_SCOPE Attributes Read by Service

The service reads (never writes) the following **SERVER_SCOPE** attributes from each plant asset. All default to a safe value when absent — no attribute is mandatory beyond those in §Required Plant Attributes.

> **Daily roll-up ts note**: since Phase 1.5, `actual_daily_energy_kwh` and `total_generation_expected_kwh` rows are stamped at **each plant's local midnight** (derived from the `timezone` attribute). For a single-timezone fleet this is identical to the service-host midnight. For a multi-timezone fleet, roll-up rows for plants in different zones will carry different `ts` values — aggregate widgets should sum leaf-plant rows rather than relying on a single master `ts`.

| Attribute key | Type | Default | Read by | Purpose |
|---|---|---|---|---|
| `actual_power_keys` | string (CSV) | `"active_power"` | `daily_job.py`, `loss_rollup_job.py` | Ordered list of TB telemetry keys to try when fetching meter power. First key that returns non-empty data wins (first-match semantics). Use a comma-separated list when the plant's meter publishes under a non-standard key (e.g. `"EnergyMeter_active_power"` or `"p341_active_power,active_power"`). |
| `actual_power_key` | string | *(none)* | `daily_job.py`, `loss_rollup_job.py` | **Legacy alias (singular).** Honoured with a one-time WARN log per plant. Migrate to `actual_power_keys` (plural CSV). Will be removed after 90 days from first WARN. |
| `active_power_unit` | string | `"kW"` | `daily_job.py`, `loss_rollup_job.py` | Unit in which the plant's meter publishes its active power telemetry. Accepted values: `"kW"` (no scaling) or `"W"` (service multiplies the series by 0.001 before integration). A mismatch here will cause `actual_daily_energy_kwh` and all loss-attribution daily keys to be off by 1000×. |
| `timezone` | string (IANA) | `settings.TZ_LOCAL` | `daily_job.py`, `loss_rollup_job.py` | Plant-local IANA timezone (e.g. `"Asia/Colombo"`). Used to compute the day-boundary `ts` under which the daily roll-up row is written, and to filter the solar window (05:00–19:00 local) for integration. Falls back to the service's `TZ_LOCAL` env var when absent or invalid. |
| `loss_attribution_enabled` | boolean | `true` | `loss_rollup_job.py` | Per-plant opt-out for loss attribution. Set `false` to suppress all `loss_*` daily writes for this plant without needing to remove `pvlib_enabled`. |
| `pvlib_enabled` | boolean | `false` | `forecast_service.py`, all jobs | Master opt-in flag. Plant is discovered and processed only when `isPlant=true AND pvlib_enabled=true`. |
| `setpoint_keys` | string (CSV) | `settings.LOSS_DEFAULT_SETPOINT_KEYS` | `loss_rollup_job.py` | Ordered list of TB keys to query for the active setpoint (curtailment limit). First key with data wins. |
| `tariff_rate_lkr` | double | *(none)* | `loss_rollup_job.py`, `revenue_job.py` | Electricity tariff in LKR/kWh. Used by loss rollup for daily LKR keys and by revenue job for monthly/yearly LKR revenue. Missing tariff → all LKR keys written as -1. |
| `pvlib_services` | JSON object | all-true | all writers | 5-key boolean dict that enables/disables individual service modules per plant. See §`pvlib_services` Attribute below. |
| `commissioning_date` | string (ISO date) | *(none)* | `revenue_job.py`, `auto_onboard.py` | Date the plant was commissioned (e.g. `"2021-03-15"`). Used to sentinel pre-commissioning yearly rows and to anchor the auto-onboard backfill window. Required when `pvlib_services.revenue=true` or `pvlib_services.loss_attribution=true`. |
| `onboarding_completed` | boolean | `false` | `auto_onboard.py` | Set to `true` by the Sunday auto-onboard cron after all backfill steps complete successfully. Plants with this attribute `true` are silently skipped on subsequent cron runs. |
| `onboarding_completed_at` | string (ISO datetime) | *(none)* | `auto_onboard.py` | UTC ISO-8601 timestamp of when onboarding was marked complete. Set atomically alongside `onboarding_completed`. |

### Attribute-setting tooling

As of v1.3, the canonical way to bulk-set plant attributes is via `scripts/shared/tb_config_loader.py` (reads `plants_master.yml`). The legacy script `scripts/shared/set_active_power_unit.py` is **deprecated** — running it will exit with an error pointing to `tb_config_loader.py`. See `ONBOARDING_GUIDE.md` for the full onboarding workflow.

To audit attribute completeness across the fleet:

```bash
python scripts/shared/audit_tb_config.py --root-ids <root_uuid> --format table
# Exit code 1 if any plant has an ERR (missing required attr).
# WARN for missing optional attrs (tariff, actual_power_keys when active_power absent, etc.).
```

---

## `pvlib_services` Attribute (Phase 4 — added 2026-05-15)

Set as a **SERVER_SCOPE** JSON string on each plant asset. Controls which service modules run for that plant. All keys default to `true` when the attribute is absent or a key is missing.

```json
{
  "physics_live":      true,
  "daily_energy":      true,
  "loss_attribution":  true,
  "p_values":          true,
  "revenue":           true
}
```

| Key | Default | Controls |
|---|---|---|
| `physics_live` | `true` | Per-minute `potential_power` writes in `forecast_service.py` |
| `daily_energy` | `true` | `actual_daily_energy_kwh` in `daily_job.py` |
| `loss_attribution` | `true` | All `loss_*_daily_*` and lifetime attrs in `loss_rollup_job.py` |
| `p_values` | `true` | All P-value daily/monthly/annual keys in `pvalue_job.py` |
| `revenue` | `true` | All `*_revenue_*_lkr` and `actual_yearly_energy_kwh` keys in `revenue_job.py` |

The master `pvlib_enabled` flag still gates all services — setting `pvlib_enabled=false` suppresses everything regardless of this attribute.

To disable revenue for one plant without removing pvlib:

```bash
# In ThingsBoard UI: SERVER_SCOPE attribute on the plant asset
pvlib_services = {"physics_live":true,"daily_energy":true,"loss_attribution":true,"p_values":true,"revenue":false}
```

---

## Per-Year P-Value Timeseries (Phase 4 — added 2026-05-15)

Written by `pvalue_job.py` alongside the existing daily/monthly/weekly keys. Each row is stamped at the **1st-of-year local midnight** for the target calendar year, allowing `revenue_job.py` to look up the historical P50 for any past year.

| Key | Unit | Timestamp | Description |
|---|---|---|---|
| `p50_energy_annual` | kWh | 1st Jan midnight local | P50 annual energy for the target year. Same value as the `p50_energy` SERVER_SCOPE attr but retained as a timeseries row so historical years are preserved when the attr is overwritten. |
| `p90_energy_annual` | kWh | 1st Jan midnight local | P90 annual energy for the target year. |
| `p95_energy_annual` | kWh | 1st Jan midnight local | P95 annual energy for the target year. |

**Why timeseries instead of attributes?** The SERVER_SCOPE `p50_energy` attribute is overwritten each time `pvalue_job` runs (typically annually). For years 2016–2024, the attribute only ever holds the most recent year's value. `p50_energy_annual` as a timeseries retains all years in the TB telemetry store and can be queried by year using `start`/`end` window.

**Write-call optimisation (Step 29)**: Daily and MTD records that share the same local-midnight `ts` are merged into a single TB API call by `_merge_records_by_ts()`, halving write-call count from 730 to 365 for the annual pvalue job.

---

## Revenue Telemetry (Phase 4 — added 2026-05-15)

Written by `app/services/revenue_job.py`. Requires `tariff_rate_lkr` attribute on the plant asset and `pvlib_services.revenue != false`. Sentinel = `-1` when tariff is missing or data is unavailable.

### Monthly Revenue

Timestamp = local midnight of **1st of the target month**. Written by the `pvlib_revenue_monthly` cron (1st-of-month 00:15 local) and `/admin/run-revenue-monthly`.

| Key | Unit | Source | Description |
|---|---|---|---|
| `expected_revenue_monthly_lkr` | LKR | `forecast_p50_monthly` (MWh) × 1000 × tariff | Expected monthly revenue at P50 yield. `-1` if tariff or P50 data missing. |
| `actual_revenue_monthly_lkr` | LKR | Σ `actual_daily_energy_kwh` × tariff | Actual monthly revenue based on metered daily energy. `-1` if tariff or daily data missing. |

### Yearly Revenue

Timestamp = local midnight of **1st January of the target year**. Written by the `pvlib_revenue_yearly` cron (1st-of-year 00:20 local) and `/admin/run-revenue-yearly`.

| Key | Unit | Source | Description |
|---|---|---|---|
| `expected_revenue_yearly_lkr` | LKR | `p50_energy_annual` (kWh) × tariff | Expected annual revenue at P50 yield. `-1` if tariff or annual P50 data missing. |
| `actual_revenue_yearly_lkr` | LKR | Σ `actual_daily_energy_kwh` × tariff (full year) | Actual annual revenue. `-1` if tariff missing or year predates `commissioning_date`. |
| `actual_yearly_energy_kwh` | kWh | Σ `actual_daily_energy_kwh` (full year) | Actual annual metered energy. `-1` if year predates `commissioning_date`. |

**Idempotency**: calling either monthly or yearly endpoint twice for the same period produces the same result (overwrite semantics).

**Pre-commissioning sentinel**: if the target year is before the plant's `commissioning_date` year, `actual_revenue_yearly_lkr` and `actual_yearly_energy_kwh` are written as `-1`.

### Admin Endpoints

| Endpoint | Default | Notes |
|---|---|---|
| `POST /admin/run-revenue-monthly?year=N&month=M` | Previous month | Recomputes one month for the full fleet |
| `POST /admin/run-revenue-yearly?year=N` | Previous year | Recomputes one year for the full fleet |
| `POST /admin/run-revenue-backfill?asset_id=<id>&years_back=10` | 10 years | Full backfill for one plant; idempotent |

---

## Auto-Onboard (Phase 4 — added 2026-05-15)

The Sunday 03:00 cron (`pvlib_autoonboard`, enabled via `AUTO_ONBOARD_ENABLED=true` in `.env`) runs a zero-touch full-historical backfill for every plant that lacks `onboarding_completed=true`.

**Backfill chain per plant:**
1. `pvalue_job` for each of the last 10 calendar years
2. `daily_job` from `max(commissioning_date, today−10yr)` to yesterday
3. `loss_rollup_job` for the same date window (if `pvlib_services.loss_attribution`)
4. `recompute_lifetime_for_fleet` to rebuild lifetime attributes from the freshly backfilled rows
5. `revenue_job` backfill (if `pvlib_services.revenue`)
6. Sets `onboarding_completed=true` + `onboarding_completed_at=<UTC ISO>`

**Per-plant timeout**: `AUTOONBOARD_PER_PLANT_TIMEOUT_S` (default 900 s). On timeout the plant is left un-marked and retried the following Sunday.

**Prometheus counters** (exposed via `/metrics`):

| Metric | Type | Description |
|---|---|---|
| `pvlib_autoonboard_attempted_total` | counter | Total plants attempted since process start |
| `pvlib_autoonboard_completed_total` | counter | Plants successfully onboarded |
| `pvlib_autoonboard_failed_total` | counter | Plants that failed or timed out |
| `pvlib_autoonboard_pending` | gauge | Plants currently awaiting onboarding (decremented on success) |

