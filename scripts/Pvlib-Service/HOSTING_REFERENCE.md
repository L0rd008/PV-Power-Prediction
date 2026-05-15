# Pvlib-Service — Hosting Reference

*Comprehensive reference for operators and integrators. Version 2.0 — 2026-05-15.*

> For the step-by-step copy-paste deployment runbook, see **[HOSTING_QUICKSTART.md](HOSTING_QUICKSTART.md)**.

### Placeholder Definitions

Commands and examples in this reference use angle brackets (`<...>`) for values you must supply:

| Placeholder | What to replace it with |
|---|---|
| `<host>` | Your ThingsBoard host domain or IP (e.g., `windforce.thingsnode.cc`). |
| `<UUID>` / `<id>` | A specific ThingsBoard asset or device UUID. |
| `<secret>` | The ThingsBoard password for the service user. |

---

## Environment Variables

All variables are read from the `.env` file (or from the environment). Required variables have no default.

### Connection

| Variable | Default | Description |
|---|---|---|
| `TB_HOST` | *(required)* | ThingsBoard base URL, e.g. `https://windforce.thingsnode.cc`. No trailing slash. |
| `TB_USERNAME` | *(required)* | ThingsBoard user email. This user must have read access to all plant assets and write access for SERVER_SCOPE attributes and telemetry. |
| `TB_PASSWORD` | *(required)* | ThingsBoard user password. |
| `TB_ROOT_ASSET_IDS` | *(required)* | Comma-separated UUIDs of root assets. BFS discovers all plants reachable from these roots via the `Contains` relation where `isPlant=true AND pvlib_enabled=true`. |

### Timezone and Locale

| Variable | Default | Description |
|---|---|---|
| `TZ_LOCAL` | `Asia/Colombo` | IANA timezone used for day-boundary calculations, cron scheduling, and all local-midnight timestamps. Set to the timezone of your fleet's majority. Per-plant overrides via the `timezone` attribute take precedence for that plant's daily roll-up `ts`. |

### Scheduler and Compute

| Variable | Default | Description |
|---|---|---|
| `SCHEDULER_ENABLED` | `true` | Enable/disable automatic scheduler. Set `false` for manual testing (uvicorn only). |
| `SCHEDULER_INTERVAL_MINUTES` | `1` | Minutes between live physics cycles. Minimum effective value: 1. Values below 5 are not recommended for fleets > 100 plants. |
| `READ_LAG_SECONDS` | `30` | Seconds behind 'now' to start the read window (accounts for TB ingestion lag). |
| `READ_WINDOW_SECONDS` | `90` | Width of the telemetry read window in seconds. |
| `MAX_CONCURRENT_PLANTS` | `5` | Maximum number of plants processed concurrently per scheduler cycle. Increase to 10–20 for large fleets on 4+ GB instances. |
| `PLANT_ATTRS_CACHE_TTL_S` | `300` | Plant attribute cache TTL in seconds. The service caches `get_asset_attributes()` results per plant to reduce TB API load. Set `0` to disable caching (full refresh on every cycle). At 1000 plants and 5-min cycle, 300 s TTL reduces attr fetches from ~288k/day to ~58k/day. |

### Logging and Debug

| Variable | Default | Description |
|---|---|---|
| `LOG_LEVEL` | `INFO` | Python logging level for the service. Accepted values: `DEBUG`, `INFO`, `WARNING`, `ERROR`. Set `DEBUG` during initial deployment to surface station-resolution decisions, BFS discovery steps, and roll-up details in the container logs. Revert to `INFO` in steady-state production. |
| `DEBUG` | `false` | When `true`, enables additional debug-mode behaviour in the FastAPI app (uvicorn auto-reload, detailed 500 tracebacks in HTTP responses). Set `false` in production. |
| `MODE` | `pvlib` | Service mode selector. Currently `pvlib` is the only production mode; the value is logged at startup for ops identification and may gate future mode branches. |

### External APIs

| Variable | Default | Description |
|---|---|---|
| `SOLCAST_API_KEY` | *(optional)* | Solcast API key. When present, the service falls back to Solcast hourly forecasts for plants without a working weather station. Without this key, the final fallback is a pvlib clear-sky model (no clouds). |

### Feature Flags

| Variable | Default | Description |
|---|---|---|
| `PVALUE_JOB_ENABLED` | `false` | When `true`, registers the annual Jan-1 03:00 P-value batch and the 01:00 daily new-plant auto-detection cron. Also enables `/admin/run-pvalues*` endpoints. |
| `REVENUE_JOB_ENABLED` | `false` | When `true`, registers the `pvlib_revenue_monthly` (1st-of-month 00:15) and `pvlib_revenue_yearly` (1st-of-year 00:20) crons. Requires `tariff_rate_lkr` and `commissioning_date` on each plant. |
| `AUTO_ONBOARD_ENABLED` | `false` | When `true`, registers the `pvlib_autoonboard` Sunday 03:00 cron for zero-touch historical backfill of newly added plants. |
| `AUTOONBOARD_PER_PLANT_TIMEOUT_S` | `900` | Maximum wall-clock seconds allowed for a single plant's auto-onboard backfill. Plants that exceed this are left un-marked and retried the following Sunday. |
| `AUTO_ONBOARD_BACKFILL_ENABLED` | `false` | When `true`, the 01:00 new-plant detection cron also chains a 30-day daily-energy and loss-rollup backfill for newly P-valued plants. |
| `EVAL_PLANT_IDS` | *(none)* | Comma-separated asset UUIDs of plants to include in the Sunday 02:00 weekly accuracy evaluation (`weekly_eval.py`). The job computes NMAE, NRMSE, and energy-error % against actual meter data and writes `pvlib_accuracy_report_json` to each plant's SERVER_SCOPE. Leave blank to disable the eval. Example: `0e4b4070-50ff-11ef-b4ce-d5aee9e495ad,3c2492b0-5669-11f0-b892-f5acae6d9b71` |

### Loss Attribution

| Variable | Default | Description |
|---|---|---|
| `LOSS_ROLLUP_ENABLED` | `false` | Master flag for the daily loss-rollup job. Set `true` after verifying on a single plant. |
| `LOSS_DEFAULT_SETPOINT_KEYS` | `setpoint_active_power,curtailment_limit,power_limit` | Comma-separated ordered list of TB telemetry keys to query for the active curtailment setpoint. First key with data wins. |
| `LOSS_MIN_VALID_SAMPLES` | `360` | Minimum number of telemetry samples per day required to compute a valid daily loss figure. Days with fewer samples are written as sentinel `-1`. |
| `LOSS_LIFETIME_PAGE_DAYS` | `90` | Page size in days when paging history during `/admin/recompute-lifetime`. |
| `LOSS_TODAY_PARTIAL_ENABLED` | `false` | Master flag for the intra-day today-partial cron. Requires `LOSS_ROLLUP_ENABLED=true`. |
| `LOSS_TODAY_PARTIAL_INTERVAL_MIN` | `5` | Interval in minutes for the today-partial cron. |
| `LOSS_TODAY_PARTIAL_DAY_START_HOUR` | `5` | Local-tz hour at which the today-partial cron starts firing. |
| `LOSS_TODAY_PARTIAL_DAY_END_HOUR` | `19` | Local-tz hour at which the today-partial cron stops firing. |
| `LOSS_TODAY_PARTIAL_MIN_SAMPLES` | `30` | Minimum 1-min samples required by the today-partial path. |

### P-Value Job

| Variable | Default | Description |
|---|---|---|
| `PVGIS_START_YEAR` | `2005` | First year of PVGIS-ERA5 historical data to fetch (inclusive). |
| `PVGIS_END_YEAR` | `2023` | Last year of PVGIS-ERA5 historical data to fetch (inclusive). |
| `PVGIS_RADDATABASE` | `PVGIS-ERA5` | PVGIS radiation database. `PVGIS-ERA5` is globally available. |
| `PVGIS_REQUEST_TIMEOUT_S` | `60` | HTTP timeout in seconds for each PVGIS API call. |
| `PVGIS_RETRY_MAX` | `3` | Maximum retry attempts per PVGIS cell fetch on transient errors. |

---

## ThingsBoard Plant Attributes

### Master Opt-In

| Attribute | Type | Required | Notes |
|---|---|---|---|
| `isPlant` | boolean | Yes | Must be `true`. Set on every plant asset. |
| `pvlib_enabled` | boolean | Yes | Must be `true` for the service to process this plant. |

### Physics Configuration (one of A or B)

**Option A — `pvlib_config` JSON blob (preferred)**

| Attribute | Type | Notes |
|---|---|---|
| `pvlib_config` | JSON | Full plant config. See `config/kebithigollewa_pvlib_config.json` for the template. Fields: `orientations`, `module`, `inverter`, `iam`, `station`, `defaults`, plus all flat fields below as top-level keys. |

**Option B — flat attributes (legacy, auto-detected)**

| Attribute | Type | Notes |
|---|---|---|
| `orientations` | JSON array | `[{"name":"Main","tilt":8,"azimuth":0,"module_count":23296,"use_measured_poa":true}]` |
| `module` | JSON | `{"area_m2":2.5833,"efficiency_stc":0.2248,"gamma_p":-0.0029}` |
| `inverter` | JSON | `{"ac_rating_kw":10000,"dc_threshold_kw":0,"use_efficiency_curve":true,"efficiency_curve_kw":[...],"efficiency_curve_eta":[...],"flat_efficiency":0.9855}` |
| `iam` | JSON | `{"angles":[0,40,50,60,70,75,80,85,90],"values":[1,1,1,1,1,0.984,0.949,0.83,0]}` |
| `station` | JSON | `{"ghi_key":"wstn1_horiz_irradiance","poa_key":"wstn1_tilted_irradiance",...}` |
| `defaults` | JSON | `{"wind_speed_ms":1.0,"air_temp_c":27.96}` |
| `latitude` | double | Decimal degrees |
| `longitude` | double | Decimal degrees |
| `altitude_m` | double | Metres |
| `timezone` | string | IANA tz, e.g. `"Asia/Colombo"` |
| `thermal_model` | string | `"open_rack_glass_glass"` |
| `soiling`, `lid`, `module_quality`, `mismatch`, `dc_wiring`, `ac_wiring` | double | Loss fractions (0–1) |
| `albedo` | double | Ground reflectance fraction |
| `far_shading` | double | Far shading multiplier (`1.0` = no shading) |

### Energy Metering

| Attribute | Type | Default | Notes |
|---|---|---|---|
| `actual_power_keys` | string (CSV) | `"active_power"` | Ordered list of TB telemetry keys to try for meter power. First key with non-empty data wins. |
| `actual_power_key` | string | *(none)* | Legacy singular alias. Deprecated — migrate to `actual_power_keys`. |
| `active_power_unit` | string | `"kW"` | Unit of the meter's power telemetry. `"W"` → service multiplies by 0.001. |

### Loss Attribution

| Attribute | Type | Default | Notes |
|---|---|---|---|
| `setpoint_keys` | string (CSV) | `settings.LOSS_DEFAULT_SETPOINT_KEYS` | TB keys to query for the curtailment setpoint percentage. |
| `tariff_rate_lkr` | double | *(none)* | LKR/kWh electricity tariff. Missing → all revenue/LKR keys written as `-1`. |
| `loss_attribution_enabled` | boolean | `true` | Per-plant opt-out. Set `false` to suppress all `loss_*` writes. |

### Phase 4 Attributes

| Attribute | Type | Default | Notes |
|---|---|---|---|
| `pvlib_services` | JSON object | all-true | Per-service enable/disable flags. Keys: `physics_live`, `daily_energy`, `loss_attribution`, `p_values`, `revenue`. All default `true` when absent. |
| `commissioning_date` | string (ISO date) | *(none)* | Plant commissioning date, e.g. `"2021-03-15"`. Required for revenue job and auto-onboard. Pre-commissioning yearly rows are written as sentinel `-1`. |
| `onboarding_completed` | boolean | `false` | Set to `true` by auto-onboard cron after successful backfill. Plants with `true` are skipped on subsequent runs. |
| `onboarding_completed_at` | string (ISO datetime) | *(none)* | UTC ISO-8601 timestamp set alongside `onboarding_completed`. |

---

## Telemetry Keys Written

For a complete and authoritative table of all written keys, see **[TELEMETRY_CONTRACT.md](TELEMETRY_CONTRACT.md)**.

Quick reference by job:

| Job | Primary written keys |
|---|---|
| `forecast_service.py` (live cycle) | `potential_power`, `active_power_pvlib_kw`, `pvlib_data_source`, `pvlib_model_version`, `ops_expected_unit` |
| `daily_job.py` (daily at 00:05 local) | `actual_daily_energy_kwh`, `total_generation_expected_kwh`, `pvlib_daily_energy_kwh`, `actual_weekly_energy_kwh`, `actual_mtd_energy_kwh` |
| `loss_rollup_job.py` (daily at 00:10 local) | `loss_grid_daily_kwh`, `loss_curtail_daily_kwh`, `loss_revenue_daily_lkr`, `loss_curtail_revenue_daily_lkr`, `potential_energy_daily_kwh`, `exported_energy_daily_kwh`, `loss_tariff_rate_lkr_at_compute`, `loss_data_source`, `loss_model_version` + lifetime attrs |
| `pvalue_job.py` (annual Jan 1, 03:00 local) | `forecast_p50/p90/p95_daily`, `_monthly`, `_weekly`, `_mtd` (all MWh); `p50/p90/p95_energy` attrs (kWh); `p50/p90/p95_energy_annual` timeseries (kWh) |
| `revenue_job.py` (monthly + yearly crons) | `expected_revenue_monthly_lkr`, `actual_revenue_monthly_lkr`, `expected_revenue_yearly_lkr`, `actual_revenue_yearly_lkr`, `actual_yearly_energy_kwh` |

---

## Admin API Reference

All endpoints are `POST` unless noted. Base URL: `http://<host>:8000`.

### Live Cycle

| Endpoint | Notes |
|---|---|
| `POST /admin/run-now` | Trigger an immediate physics cycle (does not wait for current cycle to finish if one is running) |
| `POST /admin/refresh-plants` | Invalidate the plant-discovery BFS cache and the plant-attrs TTL cache. Next cycle re-discovers everything. |
| `GET /pvlib/discover` | List all currently discovered plants with their attributes. |
| `POST /pvlib/run-asset` | Run the physics pipeline for a single plant synchronously. Body: `{"asset_id": "<UUID>"}` (start/end optional). Returns the result JSON. |

### Batch Jobs

| Endpoint | Parameters | Notes |
|---|---|---|
| `POST /admin/run-daily` | `date=YYYY-MM-DD` (default: yesterday) | Recompute `actual_daily_energy_kwh` for the full fleet. |
| `POST /admin/run-pvalues` | `year=N` (default: current year) | Run P-value batch for the full fleet (5–20 min). |
| `POST /admin/run-pvalues-plant` | `asset_id=<UUID>`, `year=N` | Run P-value batch for one plant (smoke test). |
| `POST /admin/run-loss-rollup` | `date=YYYY-MM-DD` | Re-run daily loss-rollup for a specific date. |
| `POST /admin/recompute-lifetime` | *(none)* | Recompute lifetime loss attributes for the full fleet from all existing daily rows. |

### Revenue (requires `REVENUE_JOB_ENABLED=true` or called manually)

| Endpoint | Parameters | Notes |
|---|---|---|
| `POST /admin/run-revenue-monthly` | `year=N`, `month=M` (default: previous month) | Compute monthly LKR revenue for the full fleet. |
| `POST /admin/run-revenue-yearly` | `year=N` (default: previous year) | Compute yearly LKR revenue for the full fleet. |
| `POST /admin/run-revenue-backfill` | `asset_id=<UUID>`, `years_back=10` | Backfill all months + years for one plant. Idempotent. |

### Auto-Onboard

| Endpoint | Parameters | Notes |
|---|---|---|
| `POST /admin/run-autoonboard` | `asset_id=<UUID>` (optional) | Run onboarding backfill. Without `asset_id`: full fleet. With `asset_id`: one plant only. |

### Observability

| Endpoint | Notes |
|---|---|
| `GET /health` | Returns `{"status":"ok", "tb_connected": bool, "plants_discovered": N, ...}` |
| `GET /metrics` | Prometheus text format. Exposes cycle duration, data source distribution, Solcast/discovery cache hit rates, plant failure counters, and auto-onboard counters. |
| `GET /pvlib/status/{job_id}` | Status of an async fleet job. |
| `GET /pvlib/jobs` | List all async jobs. |

---

## Cron Schedule

All crons run inside the process; no external cron daemon is needed.

| Job ID | Schedule | Description |
|---|---|---|
| `pvlib_live_cycle` | Every `SCHEDULER_INTERVAL_MINUTES` min | Main physics pipeline for all enabled plants |
| `pvlib_daily_rollup` | Daily at 00:05 local | Compute `actual_daily_energy_kwh`, weekly + MTD rolling sums from yesterday's meter data |
| `pvlib_loss_rollup` | Daily at 00:10 local | Compute daily loss attribution keys |
| `pvlib_loss_today_partial` | Every 5 min, 05:00–19:00 local | Intra-day partial loss-rollup for current day *(if `LOSS_TODAY_PARTIAL_ENABLED=true`)* |
| `pvlib_pvalue_annual` | Jan 1, 03:00 local | Annual P-value re-computation for full fleet *(if `PVALUE_JOB_ENABLED=true`)* |
| `pvlib_pvalue_newplants` | Daily at 01:00 local | Auto-detect new plants missing P-values and generate *(if `PVALUE_JOB_ENABLED=true`)* |
| `pvlib_weekly_eval` | Sunday 02:00 local | Weekly accuracy evaluation |
| `pvlib_revenue_monthly` | 1st-of-month 00:15 local | Monthly LKR revenue *(if `REVENUE_JOB_ENABLED=true`)* |
| `pvlib_revenue_yearly` | 1st-of-year 00:20 local | Yearly LKR revenue *(if `REVENUE_JOB_ENABLED=true`)* |
| `pvlib_autoonboard` | Sunday 03:00 local | Zero-touch onboarding backfill *(if `AUTO_ONBOARD_ENABLED=true`)* |

---

## Performance Tuning

### Recommended server size by fleet size

| Fleet size | Server specs | RAM | Notes |
|---|---|---|---|
| 1–10 plants | 1 vCPU | 1 GB | Sufficient for live cycle + daily jobs |
| 10–50 plants | 2 vCPU | 2 GB | Recommended default (e.g. Contabo VPS S) |
| 50–200 plants | 2 vCPU | 4 GB | Headroom for P-value annual job (high memory) |
| 200+ plants | 4 vCPU | 8 GB | Run P-value job during off-peak; consider dedicated instance for P-value job |

### Reducing TB API load

- **`PLANT_ATTRS_CACHE_TTL_S`**: Increase to 600–900 s for very large fleets. Plant attributes rarely change — a 15-min cache is safe.
- **`SCHEDULER_INTERVAL_MINUTES`**: Default 1 min. For dashboards refreshing every 5 min, set to 5 to reduce TB write load 5×.
- **Plant-level `pvlib_services`**: Disable services not needed for a plant (e.g. `"revenue":false` for plants without a tariff) to skip unnecessary TB reads.

### P-value job memory usage

The annual P-value job fetches up to 19 years × 8760 hours = 166,440 hourly rows per grid cell from PVGIS, keeps them in memory, and runs a pvlib simulation over each year. Peak memory per unique grid cell is ~200 MB. A fleet where all plants share one grid cell (e.g., one city) peaks at ~200 MB total; a fleet of 50 plants across 10 grid cells peaks at ~2 GB. Use a 4 GB instance if memory is a concern during the Jan 1 run.

### PVGIS rate limits

PVGIS-ERA5 endpoint allows ~10 concurrent requests. The service serialises requests per grid cell and includes a 2 s pause between PVGIS calls. Expect 1–3 min per unique grid cell per run.

---

## Widget Integration Recipes

### Forecast vs Actual Energy

Default `viewMode`: **`ytd_weekly`** — 52 weekly data points with month-name labels (Jan, Feb, ...) on x-axis. 7-day sensitivity.

Widget setting → **pvlibExpectedKey**: `total_generation_expected_kwh`

The widget expects kWh and divides by 1000 to display MWh. Ensure `active_power_unit` is set correctly so `actual_daily_energy_kwh` is in kWh.

### Loss Attribution

Wire the TB datasource to `Capacity` attribute. Widget internally fetches `potential_power` (kW instantaneous) or uses loss-rollup keys (`loss_grid_daily_kwh` etc.) for daily-energy mode. See `Widgets/Grid & Losses/Loss Attribution Unified/`.

Key settings: `actualPowerKeys: "active_power"`, `setpointKeys: "setpoint_active_power"`, `potentialPowerKeys: "potential_power"`, `tariffAttributeKey: "tariff_rate_lkr"`.

### FDI Card (Forecast Deviation Index)

FDI v4.0 operates in **MTD mode**: sums `forecastDailyKey` (MWh, converted to kWh) and `actualDailyKey` (kWh) over the current month client-side, plus today's partial via `actualPartialKey` (kW → kWh via agg=SUM ÷ 60).

Three instances per dashboard: set `forecastDailyKey` to `forecast_p50_daily` / `forecast_p90_daily` / `forecast_p95_daily`.

Requires P-value job to have run for the current year and daily_job to be running.

### Curtailment vs Potential Power (V5)

Widget setting → **potentialPowerKeys**: `potential_power` (default). Renders as dashed green Potential Power line. Falls back to half-sine model when no TB data.

### Expected vs Actual Revenue

Requires `expected_revenue_monthly_lkr` and `actual_revenue_monthly_lkr` (monthly view) or `expected_revenue_yearly_lkr` and `actual_revenue_yearly_lkr` (yearly view). Set `REVENUE_JOB_ENABLED=true` and ensure `tariff_rate_lkr` + `commissioning_date` are set on each plant. Wire `DS[0]` = expected, `DS[1]` = actual.

---

## Troubleshooting

### Service won't start

```bash
docker logs pvlib-service --tail 50
```

Common causes:
- `TB_HOST` unreachable from the server (check firewall, DNS)
- Missing required env vars (`TB_USERNAME`, `TB_PASSWORD`, `TB_ROOT_ASSET_IDS`)
- Port 8000 already in use on the host

### Plants not being discovered

```bash
curl http://localhost:8000/pvlib/discover
```

- Returns empty list → check `TB_ROOT_ASSET_IDS` contains valid UUIDs
- Returns plants but `pvlib_enabled=false` → set `pvlib_enabled=true` on the plant assets in TB
- Returns plants but `isPlant=false` → set `isPlant=true` on the plant assets

### `data_source: clearsky` for all plants

The service falls through to pvlib clear-sky when:
1. No weather station data in TB — check `station.ghi_key` in `pvlib_config`; verify the device has the `Contains` relation to the plant asset
2. Solcast key is absent or invalid — check `SOLCAST_API_KEY` in `.env`
3. Weather station data is stale (beyond `freshness_minutes`) — check the station device is still publishing to TB

### `actual_daily_energy_kwh = -1` or zero

- Check `actual_power_keys` attribute — the key must match exactly what the meter publishes in TB
- Check `active_power_unit` — a `W` plant with `kW` unit setting will have 1000× under-reads
- Check meter data exists in TB for the target day (TB → device → telemetry tab)
- Check for WARN logs: `"no actual power samples for plant"`

### Loss keys all `-1`

- Check `tariff_rate_lkr` is set (for revenue loss keys)
- Check `loss_data_source` key in TB — value explains why the roll-up failed
- `"error:insufficient_samples"` → fewer than `LOSS_MIN_VALID_SAMPLES` meter samples for the day
- `"error:no_potential"` → physics cycle did not write `potential_power` for the plant on that day

### Revenue keys all `-1`

- `tariff_rate_lkr` not set → set it and re-run `/admin/run-revenue-backfill`
- `commissioning_date` not set → year predates commissioning sentinel applies — set the date
- `REVENUE_JOB_ENABLED=false` → enable and restart, or call the admin endpoint manually

### Auto-onboard not completing

- Check `commissioning_date` is set — plants without it are skipped with a WARN log
- Increase `AUTOONBOARD_PER_PLANT_TIMEOUT_S` — large plants with 10 years of history may need 1800+ s
- Run with a single plant first: `POST /admin/run-autoonboard?asset_id=<UUID>`
- Check `/metrics` → `pvlib_autoonboard_failed_total` counter; check logs for the specific step that failed

### High memory usage during P-value job

- Schedule the job during off-peak hours (default Jan 1 03:00 is fine)
- If OOM-killed, upgrade to a 4 GB instance or run the job on a larger instance temporarily
- Check for large number of unique grid cells: `GET /pvlib/discover` → count unique `(lat, lon)` pairs

---

## Security Notes

- **`.env` permissions**: `chmod 600 /opt/pvlib-service/.env` — credentials in plain text.
- **API authentication**: the Pvlib-Service API has no built-in authentication. Place it behind a firewall or nginx with HTTP Basic Auth if accessible from the internet.
- **TB user scope**: create a dedicated ThingsBoard user for the service with the minimum required permissions: read all assets/devices in the tenant, write SERVER_SCOPE attributes and telemetry on plant assets.
- **Secrets rotation**: update `TB_PASSWORD` → update `.env` → restart container. No rolling-restart mechanism is needed for this workload.
