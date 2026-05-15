# Pvlib-Service — Plant Onboarding Guide

*Version 1.1 — 2026-05-15*

This guide covers everything needed to onboard a new plant into Pvlib-Service so that FDI, FvA, Curtailment V5, and Loss Attribution widgets all render correctly. It replaces the older manual API-call workflow and the deprecated `set_active_power_unit.py` script.

> **New in v1.1 (Phase 4):** For VPS/Docker deployment and host-level configuration, see **[HOSTING_QUICKSTART.md](HOSTING_QUICKSTART.md)** (step-by-step copy-paste runbook) and **[HOSTING_REFERENCE.md](HOSTING_REFERENCE.md)** (comprehensive TB attrs, environment variables, and troubleshooting). The auto-onboard feature (`AUTO_ONBOARD_ENABLED=true`) can replace steps 4–7 of this guide for new plants with `commissioning_date` set.

### Placeholder Definitions

Throughout this guide, replace angle-bracket placeholders (`<...>`) with your actual values:

| Placeholder | What to replace it with |
|---|---|
| `<UUID>` / `<asset_uuid>` / `<uuid>` | The ThingsBoard UUID of the plant asset you are onboarding. |
| `<secret>` | The password for your ThingsBoard user account (`TB_USERNAME`). |
| `<root_uuid>` | The ThingsBoard UUID of your root asset. |

---

## 0. Prerequisites

Before onboarding a plant, ensure:

1. The plant **asset exists** in ThingsBoard (TB) with type `ASSET`.
2. The asset is reachable from one of the configured `TB_ROOT_ASSET_IDS` via the `Contains` relation chain.
3. You have the plant's `asset_id` UUID (copy from the TB asset URL).
4. You have TB admin credentials (`TB_USERNAME` / `TB_PASSWORD`).
5. The Pvlib-Service is running (or you can reach it locally at `PVLIB_HOST`).

---

## 1. Add the Plant to `plants_master.yml`

`scripts/shared/plants_master.yml` is the **single source of truth** for all plant configurations. Open it and add a new entry following the documented structure (see `plants_master.schema.json` for the full field reference).

Minimum required fields:

```yaml
- asset_id: "<UUID>"
  name: "My New Plant"
  latitude: 7.1234
  longitude: 80.5678
  altitude_m: 120
  timezone: "Asia/Colombo"
  capacity_kwp: 2000.0
  capacity_unit: "kW"
  active_power_unit: "kW"          # or "W" if meter publishes Watts
  actual_power_keys: "active_power"  # comma-separated, first-match semantics
  commissioning_date: "2025-01-01"
  tariff_rate_lkr: 22.0
  orientations:
    - tilt: 10
      azimuth: 0
      module_count: 3500
  module_template: "astronergy_580w"    # must match a file in templates/
  inverter_template: "huawei_300ktl"
  pvlib_enabled: true
  loss_attribution_enabled: true
```

> **active_power_unit**: Check the plant's meter documentation. If it publishes in Watts, set `"W"` — the service will multiply the series by 0.001 before integrating.

> **actual_power_keys**: Check what TB telemetry key the plant's gateway writes. Common values: `"active_power"` (default), `"EnergyMeter_active_power"`, `"p341_active_power"`, `"power_v3"`.  If you're unsure, use `active_power` first — the audit will WARN if that key has no data.

---

## 2. Load Configuration to ThingsBoard

Use `tb_config_loader.py` to write the plant's `pvlib_config` blob and flat companion attributes to TB SERVER_SCOPE:

```bash
# Dry-run to preview what will be written:
python scripts/shared/tb_config_loader.py \
  --host https://windforce.thingsnode.cc \
  --user PVLib-Service@thingsnode.cc --password <secret> \
  --plant <asset_uuid> --dry-run

# Apply:
python scripts/shared/tb_config_loader.py \
  --host https://windforce.thingsnode.cc \
  --user PVLib-Service@thingsnode.cc --password <secret> \
  --plant <asset_uuid>
```

The script computes a SHA-1 hash of the config blob and skips the plant if the hash matches what's already in TB (idempotent). Use `--force-overwrite` to bypass.

**Deprecated**: `set_active_power_unit.py` now exits with an error. Use `tb_config_loader.py` instead.

---

## 3. Audit Attribute Completeness

```bash
python scripts/shared/audit_tb_config.py \
  --host https://windforce.thingsnode.cc \
  --user PVLib-Service@thingsnode.cc --password <secret> \
  --plant-ids <asset_uuid> --format table
```

Exit code 0 = clean. Exit code 1 = ERR findings (required attrs missing or invalid).

Common ERR causes and fixes:

| ERR | Fix |
|---|---|
| `Capacity` missing | Add `capacity_kwp` to master file + re-run loader |
| `active_power_unit` invalid | Must be exactly `"kW"` or `"W"` |
| `orientations` missing | Add orientations array to master file |
| `latitude`/`longitude` out of range | Verify coordinates |
| `timezone` missing | Add IANA timezone string (e.g. `"Asia/Colombo"`) |

Common WARN causes (non-blocking, fix before production):

| WARN | Impact |
|---|---|
| `tariff_rate_lkr` missing | Loss revenue keys written as -1 |
| `commissioning_date` missing | Lifetime attribute anchor unknown |
| `actual_power_keys` missing | Service defaults to `active_power` |
| `weather_station_id` absent + no `solcast_resource_id` | Falls back to Tier-3 clearsky |
| `forecast_p50_daily` < 360 rows | pvalue_job not yet run for this plant |

---

## 4. Run the Full Onboarding Sequence

`onboard_plant.py` runs 9 idempotent steps: audit, discovery refresh, P-values (current + prior years), daily-energy backfill, loss-rollup backfill, lifetime recompute, and discovery confirmation.

```bash
# Standard onboarding (1 year of historical backfill):
python scripts/shared/onboard_plant.py \
  --asset-id <uuid> \
  --pvlib-host http://localhost:8000 \
  --host https://windforce.thingsnode.cc \
  --user PVLib-Service@thingsnode.cc --password <secret>

# 2-year backfill:
python scripts/shared/onboard_plant.py --asset-id <uuid> --years-back 2

# Skip loss attribution (if plant doesn't have a meter):
python scripts/shared/onboard_plant.py --asset-id <uuid> --skip-loss

# Preview without executing:
python scripts/shared/onboard_plant.py --asset-id <uuid> --dry-run
```

After completion, all of these should be populated in TB:

- `forecast_p50/p90/p95_daily` — 365 rows for current year
- `actual_daily_energy_kwh` — one row per historical day with `active_power` data
- `loss_*_daily_kwh` — one row per historical day (if loss enabled)
- `loss_*_lifetime_*` SERVER_SCOPE attributes
- `onboarding_backfilled_at` SERVER_SCOPE marker (written by auto-backfill cron if `AUTO_ONBOARD_BACKFILL_ENABLED=true`)

---

## 5. Verify Widgets

Open the plant's dashboard and check:

| Widget | Expected |
|---|---|
| **FDI** (P50 instance) | Non-zero FDI% with both forecast and actual context values |
| **FvA** | P50/P90/P95 bands with actual weekly bars |
| **Curtailment V5** | `potential_power` dashed green line visible alongside `active_power` |
| **Loss Attribution** | All four modes (`grid`, `curtail`, `revenue`, `curtailRevenue`) show non-sentinel values |

If FDI or FvA render blank: check `forecast_p50_daily` exists (pvalue_job completed) and `actual_daily_energy_kwh` exists (daily_job has run for at least one historical day).

---

## 6. Auto-Detect New Plants (Steady State)

After initial onboarding, the 01:00 daily cron (`run_pvalue_newplants_now`) automatically detects plants added in the past 24 hours (those lacking `pvalue_updated_at`) and generates P-values.

If `AUTO_ONBOARD_BACKFILL_ENABLED=true` in `.env`, the cron also chains a 30-day daily-energy + loss-rollup backfill for each newly P-valued plant and writes `onboarding_backfilled_at` when complete.

**Default**: `AUTO_ONBOARD_BACKFILL_ENABLED=false`. Enable only after verifying the bounded 30-day backfill window is acceptable for your fleet throughput.

For bulk onboarding of many plants simultaneously, use `onboard_plant.py` for each plant rather than relying on the auto-detect path — the cron serialises plants and may take too long for large batches.

---

## 7. Diagnostic Scripts

### Find Orphan Plants

Plants with `isPlant=true && pvlib_enabled=true` but not reachable from any configured root asset are silently invisible to the service. Detect them with:

```bash
python scripts/shared/find_orphan_plants.py \
  --host https://windforce.thingsnode.cc \
  --user PVLib-Service@thingsnode.cc --password <secret> \
  --pvlib-host http://localhost:8000 \
  --format table
```

Exit code 1 if any orphans found. Fix by connecting the orphaned asset to a root via a `Contains` relation in ThingsBoard.

### Check Config Drift

After running `tb_config_loader.py`, the service writes a `pvlib_config_hash` to each plant. If an operator manually edits TB attributes, the hash will diverge from the master file. Detect and fix drift with:

```bash
# Report drift only:
python scripts/shared/find_config_drift.py \
  --host https://windforce.thingsnode.cc \
  --user PVLib-Service@thingsnode.cc --password <secret> \
  --master scripts/shared/plants_master.yml \
  --report-only

# Fix drift by rewriting diverging plants:
python scripts/shared/find_config_drift.py --fix
```

Exit code 1 on any drift. Schedule this check in CI or run after any manual TB edit.

---

## 8. Capacity Factor, Revenue, and Risk Widgets

These widgets consume keys **not written by Pvlib-Service**:

| Widget | Required keys | Where to set |
|---|---|---|
| Capacity Factor Compliance | `contract_cf_target`, `actual_cf_ytd` | Set via TB UI or PPA-tracking pipeline |
| Grid Outage Timeline | `grid_outage_events`, `insurance_claims_data` | Set via external SCADA/event pipeline or manually |
| Risk Summary Panel | `revenue_at_risk`, `risk_alert_level`, etc. | External risk pipeline |
| Expected vs Actual Revenue | DS[0]/DS[1] wired to any TB key | Operator maps `forecast_p50_monthly × tariff` and `actual_daily_energy_kwh × tariff` to custom keys |

These keys are noted here for completeness; adding them is outside Pvlib-Service scope.

---

## 9. Per-Plant Timezone Note

Since Phase 1.5, daily roll-up rows (`actual_daily_energy_kwh`, `total_generation_expected_kwh`, loss keys) are stamped at each plant's **local midnight** as derived from the `timezone` SERVER_SCOPE attribute. For a single-timezone fleet (all `Asia/Colombo`) this is identical to the service-host midnight. For a multi-timezone fleet, roll-up rows for plants in different zones will carry different `ts` values.

When aggregating across plants from parent asset widgets, sum leaf-plant rows rather than relying on a single master `ts`. The cron fires in `TZ_LOCAL` and must be configured to fire early enough to cover the latest-timezone plant in your fleet (e.g., cron at `Asia/Colombo 00:05` covers any plant from UTC+1 eastward).

---

## 10. Operational SLO

| Metric | Target |
|---|---|
| Cycle p95 duration | < 45 s |
| `pvlib_cycle_plants_per_minute` | > 5 at default `MAX_CONCURRENT_PLANTS=5` |
| `forecast_p50_daily` rows per plant | 365 per year |
| `actual_daily_energy_kwh` sentinel rate | < 5% of calendar days |

Monitor via `GET /metrics` (Prometheus text format). If cycle p95 exceeds 45 s at > 200 plants, increase `MAX_CONCURRENT_PLANTS` (up to ~20 on a 2 vCPU / 2 GB instance, ~30 on a 4 vCPU / 4 GB instance) in `.env`.

---

*See also:* `TELEMETRY_CONTRACT.md` (all keys written/read), `KSP_TEST_RUNBOOK.md` (single-plant manual path), `WIDGET_PVALUE_FIX_PLAN.md` (widget architecture).
