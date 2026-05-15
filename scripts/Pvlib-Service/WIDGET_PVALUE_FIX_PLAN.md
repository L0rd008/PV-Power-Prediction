# Widget P-Value Fix Plan — FDI + FvA

Author: Opus 4.7 · Date: 2026-05-14 · Executor: Sonnet 4.7

This plan covers the small, scoped fixes needed after Pasindu's 2026-05-14 clarification pass. Most prior service-side work is already correct; the remaining defects are widget-side. See §5 for the open-questions list — user has answered the load-bearing ones inline.

---

## 1. What is already correct (do not touch)

| Concern | Status |
|---|---|
| P50/P90/P95 (not P75) across daily/weekly/monthly/MTD timeseries + annual SERVER_SCOPE attrs | ✅ Implemented in `app/services/pvalue_job.py` (pvalue-daily-v2). Per-(month, day) percentiles from 19-year PVGIS-ERA5 replayed through plant physics config. |
| Plant-tuned P-values for ~1000 plants without per-plant manual data | ✅ `pvalue_job` discovers via `pvlib_enabled`, dedups to ~70 ERA5 cells, fetches once at Jan-1 03:00 + daily new-plant cron at 01:00. |
| Future-month P-values visible | ✅ All 12 `forecast_p*_monthly` rows + all 365 `forecast_p*_daily` rows written at year start. |
| Rolling current-month sum + ts-aligned forecast | ✅ `forecast_p*_mtd` cumulative rows exist; **but consumed wrong by FDI — see §3.2**. |
| `actual_daily_energy_kwh`, `actual_weekly_energy_kwh`, `actual_mtd_energy_kwh` | ✅ Written daily at 00:05 by `app/services/daily_job.py` (integrates `active_power` w/ unit-scaling). |
| TB/server load | ✅ Negligible. ~4.7 M total telemetry rows/year across 1000 plants; once-yearly PVGIS fetches; widget reads ~100–800 rows per load. |
| PVsyst PDF role | Validation/sanity only (per user). Service ignores. Use values to compare against `p50_energy` SERVER_SCOPE after a fresh `pvalue_job` run, log mismatch > X%. |
| Physics potential_power = P50? | **NO** — `potential_power` is a single deterministic realization using the current minute's weather. P50 is the **median over 19 years of weather at this plant**. Keep them as separate concepts. FvA already renders `total_generation_expected_kwh` as a green-dotted "Physics Expected" line alongside the P50 blue dashed line. Correct as-is. |

---

## 2. Decisions taken (user-confirmed 2026-05-14)

1. **FvA timeframe**: keep current `ytd_weekly` default — YTD + future months, 52 weekly buckets, 12 month-name labels on x-axis. No change.
2. **FvA bucket size**: keep **7-day**. Drop the earlier "3-day" requirement.
3. **FDI day-1 / early-month empty state**: use today's partial via `active_power` agg=SUM (same pattern FvA already uses), with derived mode as the ultimate fallback.
4. **PVsyst**: validation sanity only; not used for runtime telemetry.

---

## 3. Defects to fix

### 3.1 FDI .js is behind its README

`Widgets/Forecasts & Risk/Forecast Deviation Card (FDI)/.js` header says v2.2 and uses precomputed `forecast_p50_mtd` + `actual_mtd_energy_kwh`. The companion README (v4.0, 2026-05-13) describes a different design: read daily P-keys + a real-meter actual key, sum on the widget. The README is the authoritative spec; the code needs to be brought up to it.

### 3.2 FDI MTD endpoint-alignment bug

`forecast_p*_mtd` is pre-written by `pvalue_job` for **every day of the year** as a cumulative sum. `actual_mtd_energy_kwh` is written by `daily_job` at 00:05 local for the **just-completed** day. With both queried `limit=1&orderBy=DESC` in the current month, the widget compares:

- forecast cumsum through **today** (precomputed in advance)
- actual cumsum through **yesterday**

→ FDI looks 3 – 5 % more negative than reality every day. Confirmed by reading `pvalue_job._build_mtd_records` (writes for all days 1..N) vs `daily_job.run_daily_rollup` (writes for day end_local − 1).

### 3.3 FDI day-1 empty state

When day 1 ≤ ts < 00:05 local on day 2, `actual_mtd_energy_kwh` has no row in the current month. Code falls into the `if (actRows.length === 0) → applyDeviation(0, 0, …)` branch, which shows actual = `--` and FDI = 0%. User wants today's partial integrated in, so the value is meaningful from morning of day 1.

### 3.4 FvA early-year empty state (residual)

`processYtdWeeklyData` succeeds even when `forecast_p*_weekly` returns zero rows for the current year (e.g., pvalue_job not yet run). Chart renders empty P-bands rather than degrading to derived mode. Fix: gate the call on "did any forecast row come back for the current year?"; if not → `tryAttributeFallback`.

### 3.5 FDI 3-instance pattern requires both legacy + new key settings

The README expects three widget copies, one per P-value. The current `.js` reads `forecastMtdKey` — fine for the precomputed-MTD path, but the new design reads `forecastDailyKey`. Settings file needs both keys present with sensible defaults so existing dashboards keep working through the upgrade window.

---

## 4. Approaches considered (per defect)

For each defect, multiple approaches were scored. Confidence ≥ 90 % on each rating — see one-line justification.

### 4.1 FDI MTD endpoint alignment + day-1 partial

| # | Approach | Strengths | Weaknesses | Score |
|---|---|---|---|---|
| A | Widget reads `forecast_p*_daily` + `actual_daily_energy_kwh` for current month, sums both, adds today's partial via `active_power` agg=SUM. | Aligned endpoints by construction. Reuses existing keys. Day-1 path needs no special case. | ~3 REST calls per refresh, ~30 daily + ~1 partial row per call (~ 90 rows total). | **9** |
| B | Pre-write a daily `actual_mtd_energy_kwh` matching `forecast_p*_mtd`'s ts; widget reads MTD pair as today. | One read per side. | Requires either reworking `daily_job` to write current-day partial (cron drift, idempotency mess) or running a 5-min ticker. Forecast still locked to year-precomputed values. | 5 |
| C | Add a today-partial cron to write `forecast_p*_mtd_today` + `actual_mtd_today`; widget reads only those. | Real-time accurate. Cleanest widget. | New telemetry keys; new cron load; can't backfill cleanly. | 6 |

**Pick: A (widget-side aggregation).** Hybrid with C is not warranted at current load. Confidence: 95 %.

### 4.2 FvA empty-state on missing forecasts

| # | Approach | Strengths | Weaknesses | Score |
|---|---|---|---|---|
| A | Detect empty forecast response in `processYtdWeeklyData` → call `tryAttributeFallback`. | Simple. No new code path. Mirrors existing `daily` mode behaviour. | Loses weekly granularity in fallback (flat daily baseline). | **9** |
| B | Add a service-side empty-data sentinel row at year start. | Avoids any client logic. | Telemetry contract bloat; couples failure mode to service. | 4 |
| C | Run `pvalue_job` on first plant-load if missing. | Auto-heal. | Heavy operation triggered by widget render; security/ops risk. | 2 |

**Pick: A.** Confidence: 95 %.

### 4.3 README ↔ code sync for FDI

| # | Approach | Strengths | Weaknesses | Score |
|---|---|---|---|---|
| A | Bring `.js` up to README v4.0 (daily-key summation, 3-instance settings, today-partial, derived fallback). | Single, definitive design. Fixes 3.2 + 3.3 in one pass. | Larger diff. Existing dashboards must adopt new settings names (legacy aliases retained). | **9** |
| B | Revert README to v2.x to match code. | Smaller diff. | Keeps the endpoint-mismatch bug. Locks the widget to precomputed MTD design. | 3 |
| C | Two-mode widget: try precomputed MTD first, fall back to daily summation if mismatch detected. | Backward compat. | Two pipelines to maintain; same bug risk. | 5 |

**Pick: A** with **legacy `forecastMtdKey` retained as ignored alias** to avoid breaking installed instances at upgrade. Confidence: 90 %.

---

## 5. Telemetry contract — proposed additions

**None.** All needed keys already exist. The fix is pure widget-side. No deprecation, no rename, no new write.

If during execution Sonnet discovers a missing key, stop and ask before adding one — telemetry contract changes require a 90-day deprecation window per `TELEMETRY_CONTRACT.md`.

---

## 6. Execution steps (Sonnet 4.7)

Order matters because step 1 unblocks the rest.

### Step 1 — Rewrite `Forecast Deviation Card (FDI)/.js` MTD path

File: `M:\Documents\Projects\MAGICBIT\Widgets\Forecasts & Risk\Forecast Deviation Card (FDI)\.js`

In `fetchMtdData()`:

1. Settings: read `forecastDailyKey` (new, primary), fall back to `forecastP50DailyKey`, then to a synthesised `forecast_p50_daily`. Read `actualDailyKey` (default `actual_daily_energy_kwh` — the daily_job key; user can override to a real meter key).
2. Compute `monthStartMs` (current code is fine — Asia/Colombo UTC+5:30).
3. Compute `todayStartMs` (same offset).
4. Issue three REST calls in parallel (use existing http.get chained pattern):
   - Forecast daily rows: `keys=<forecastDailyKey>&startTs=monthStartMs&endTs=now&limit=40&agg=NONE`
   - Actual daily rows: `keys=<actualDailyKey>&startTs=monthStartMs&endTs=todayStartMs&limit=40&agg=NONE`
   - Today's partial: `keys=<actualPartialKey || 'active_power'>&startTs=todayStartMs&endTs=now&limit=1&agg=SUM&interval=86400000`
5. Sum forecast rows (MWh) up to **today inclusive** → `fcMwh`.
6. Sum actual daily rows (kWh, `actual_daily_energy_kwh`) → `actDayKwh`.
7. Convert today's partial: `kwMin = parseFloat(rows[0].value); todayKwh = kwMin / 60.0`. Add to `actDayKwh`.
8. `actMwh = actDayKwh / 1000.0`.
9. FDI% = `((actMwh - fcMwh) / fcMwh) * 100`.
10. Call `applyDeviation(fdiPct, actMwh, fcMwh, 'mtd')`.
11. **Empty paths:**
    - If forecast daily returns zero rows → `tryAttributeDerived(... )` (existing function).
    - If actual daily AND today's partial are both empty → `applyDeviation(0, 0, fcMwh, 'mtd')` (preserves the day-1 grace look but with a real forecast value).

Remove the legacy `forecast_p*_mtd` read entirely. Bump header comment to `// FDI v4.0`.

### Step 2 — Update `Forecast Deviation Card (FDI)/settings.json`

Add three settings (preserve legacy ones with `helpText: "(legacy — ignored in v4.0)"` so dashboards don't break on upgrade):

```json
{
  "id": "forecastDailyKey",
  "type": "text",
  "default": "forecast_p50_daily",
  "helpText": "Primary: daily P-key. Set per instance to forecast_p50/p90/p95_daily."
},
{
  "id": "actualDailyKey",
  "type": "text",
  "default": "actual_daily_energy_kwh",
  "helpText": "Pre-computed daily kWh from daily_job.py. May be set to total_generation if using a meter-derived key."
},
{
  "id": "actualPartialKey",
  "type": "text",
  "default": "active_power",
  "helpText": "Realtime kW key used to integrate today's partial via agg=SUM."
}
```

Leave `forecastMtdKey`, `actualEnergyKey`, `forecastP50DailyKey` in place (legacy aliases — README documented for 90 days).

### Step 3 — Patch `Forecast vs Actual Energy V1/.js` empty-data fallback

File: `M:\Documents\Projects\MAGICBIT\Widgets\Forecasts & Risk\Forecast vs Actual Energy\V1 TB Latest Values Widget\.js`

In `fetchYtdWeeklyData()` success branch, before calling `processYtdWeeklyData(...)`:

```js
var hasPvalues = (data[p50Key] || []).some(function(r){
    return new Date(parseInt(r.ts)).getFullYear() === year;
});
if (!hasPvalues) {
    tryAttributeFallback(entIdStr, entTypeStr, s);
    return;
}
```

Mirror the same gate inside `processMonthlyData` and `processMtdDailyData` for symmetry.

### Step 4 — README sync

File: `M:\Documents\Projects\MAGICBIT\Widgets\Forecasts & Risk\Forecast Deviation Card (FDI)\README.md`

Confirm settings table matches step 2. Add a one-paragraph "v4.0 migration" note: "If you have an existing FDI instance, set `forecastDailyKey` and `actualDailyKey`; legacy `forecastMtdKey` is ignored. No data loss."

### Step 5 — Telemetry contract — confirm no change

File: `M:\Documents\Projects\MAGICBIT\Power-Prediction\scripts\Pvlib-Service\TELEMETRY_CONTRACT.md`

No edits expected. If Sonnet finds itself wanting to add a key, stop and re-read §5 of this plan.

### Step 6 — Validation

For each plant Sonnet has TB access to (default: KSP_Plant = `0e4b4070-50ff-11ef-b4ce-d5aee9e495ad`):

1. In a TB widget editor, paste new FDI `.js` and `settings.json`.
2. Confirm three instances render P50, P90, P95 from one asset using different `forecastDailyKey` values.
3. Pick a day where `actual_daily_energy_kwh` exists and confirm FDI% matches a hand calc: `((Σ actual − Σ forecast) / Σ forecast) × 100`.
4. Compare against the **old** widget reading on the same day; expect the old FDI to be 3–8 % more negative due to the endpoint-mismatch bug.
5. Validate FvA's empty-data fallback by temporarily renaming the `pvalue_target_year` SERVER_SCOPE attribute or by querying a plant where `pvalue_updated_at` is missing — chart should render derived mode (flat lines) rather than blank.
6. Cross-check P50 annual attribute against the PVsyst PDF for KSP: `Kebithigollewa PVSyst 12.81MW Fixed Tilt.Huawei 300KTL Astronergy 580W MFDG.pdf`. Expect within ±5 % for a healthy plant.

### Step 7 — Document

Update memory file `project_pvlib_service.md` (or append a new memory) with: "FDI widget brought to v4.0; daily-key summation pattern; no service change."

---

## 7. Out of scope (do not do)

- **No 3-day FvA bucket size.** User confirmed 7-day stays.
- **No PVsyst override.** Validation only.
- **No new TB telemetry keys.**
- **No service-side cron changes.** All existing crons (Jan-1 03:00, 01:00 new-plant detection, 00:05 daily, 00:10 loss, weekly_eval 02:00 Sun) remain.
- **No retire of `forecast_p*_mtd` keys.** Will eventually become unused but stay under the 90-day deprecation window. Re-evaluate in plan v2.

---

## 8. Open questions for Pasindu (none blocking — Sonnet may proceed)

These can be deferred to plan v2 after the widget patches land:

1. After 90 days of `forecast_p*_mtd` non-use, should we retire those keys to reduce TB write volume? They cost 1095 rows/plant/year (~ 1 M rows/fleet/year). Trivial but eligible for cleanup.
2. Should `actual_daily_energy_kwh` be co-versioned with `total_generation` so dashboards can pick the source (computed-from-active vs meter-direct)? Not blocking; current daily_job integration is the de-facto truth.
3. PVsyst-as-validation: do we want an automated weekly check that emails a delta report (`p50_energy` attribute vs PVsyst spec)? Could live alongside `weekly_eval`. Lightweight one-shot script.

---

## 9. File index (paths Sonnet will touch)

- `M:\Documents\Projects\MAGICBIT\Widgets\Forecasts & Risk\Forecast Deviation Card (FDI)\.js` — rewrite `fetchMtdData`, bump header to v4.0
- `M:\Documents\Projects\MAGICBIT\Widgets\Forecasts & Risk\Forecast Deviation Card (FDI)\settings.json` — add three new settings, mark legacy
- `M:\Documents\Projects\MAGICBIT\Widgets\Forecasts & Risk\Forecast Deviation Card (FDI)\README.md` — migration note
- `M:\Documents\Projects\MAGICBIT\Widgets\Forecasts & Risk\Forecast vs Actual Energy\V1 TB Latest Values Widget\.js` — empty-data gate in `fetchYtdWeeklyData`, `fetchMonthlyData`, `fetchMtdDailyData`

No Pvlib-Service files are modified by this plan.

---

# Addendum A — Multi-Plant Onboarding & Service ↔ Widget Compatibility (2026-05-14)

Author: Opus 4.7 · Goal: prepare Pvlib-Service + Forecasts/Grid widgets to absorb ~1000 plants of diverse vintage, telemetry, and data-collection methods. Plant discovery driven by `TB_ROOT_ASSET_IDS`; past + current + future data must populate automatically.

This addendum is purely additive — it does not contradict §1–§9 above. Where it touches the same files, it strictly adds steps after the v4.0 widget rewrite lands.

---

## 10. Compatibility audit — what works, what is missing

### 10.1 Service writes ↔ widget reads (verified)

| Widget | Required keys | Service writer | Compatible? |
|---|---|---|---|
| FDI (post v4.0) | `forecast_p*_daily`, `actual_daily_energy_kwh`, `active_power` | `pvalue_job.py`, `daily_job.py`, plant meter | ✅ |
| FvA | `forecast_p*_weekly/monthly/daily`, `actual_daily_energy_kwh`, `actual_weekly_energy_kwh`, `active_power`, `total_generation_expected_kwh` | `pvalue_job.py`, `daily_job.py`, `forecast_service.py` | ✅ |
| Curtailment V5 | `potential_power`, `active_power`, setpoint keys, `Capacity` | `forecast_service.py` (live cycle), plant meter, plant attrs | ✅ |
| Loss Attribution (compute mode) | `potential_power`, `active_power`, setpoint keys, `tariff_rate_lkr`, `Capacity` | service + plant attrs | ✅ |
| Loss Attribution (fast path) | `loss_*_daily_kwh`, lifetime `loss_*_lifetime_*` attrs | `loss_rollup_job.py` | ✅ (when `LOSS_ROLLUP_ENABLED=true`) |
| Capacity Factor Compliance | `contract_cf_target`, `actual_cf_ytd` | **EXTERNAL** (not Pvlib) | ⚠️ See §15 |
| Grid Outage Timeline / Event Summary | `grid_outage_events`, `insurance_claims_data` (JSON arrays) | **EXTERNAL** (manual or rule chain) | ⚠️ See §15 |
| Expected vs Actual Revenue | DS[0] expected, DS[1] actual (any TB key — settings-driven) | **Plant integrator** maps `forecast_p50_monthly` × tariff and `actual_daily_energy_kwh` × tariff into custom keys | ⚠️ See §15 |
| Revenue-at-Risk Breakdown | settings only | n/a | ✅ |
| Risk Summary Panel | `revenue_at_risk`, `risk_alert_level`, `risk_percentile`, `tracking_delta` | **EXTERNAL** | ⚠️ See §15 |

### 10.2 Identified gaps for multi-plant onboarding

| # | Gap | Severity | Where it bites |
|---|---|---|---|
| G1 | `daily_job.py` and `loss_rollup_job.py` read `active_power` hard-coded; some plants publish under `power_v3`, `p341_active_power`, `EnergyMeter_active_power`. | High | Wrong/missing actuals for non-standard meters → all daily/loss/FDI/FvA actual-side numbers wrong. |
| G2 | No bulk plant-config ingestion. `pvlib_config` blob must be manually pasted per plant via TB UI. | High | At 1000 plants this is a wall; onboarding cannot scale. |
| G3 | `audit_tb_config.py` (Phase F) audits Pvlib attrs only — does not validate widget-side attrs (`tariff_rate_lkr`, `commissioning_date`, `Capacity`, `active_power_unit`, `setpoint_keys`). | Medium | Plant looks pvlib-ready but widgets render blanks. |
| G4 | `pvalue_job` writes one calendar year at a time; "last 12 months / rolling window" needs prior year's rows. | Medium | New plant onboarded in late 2026 has no Jan–Apr 2026 P-bands until backfilled. |
| G5 | No single "new-plant playbook" — onboarding requires N manual API calls in the right order. | Medium | Operator error; partial data. |
| G6 | `TB_ROOT_ASSET_IDS` is .env-static. New root needs container restart unless `/admin/refresh-plants` is called AND the new root is already a descendant of an existing root. | Low | Edge case at customer-onboarding. |
| G7 | `set_active_power_unit.py` hardcodes plant names; new plants need either a script edit or manual attr set. | Low | Widgets misscale active_power → curtailment / loss numbers off by 1000×. |
| G8 | New plant with no historical `active_power` → `daily_job` `actual_daily_energy_kwh` is `-1`; FDI/FvA show "no data" instead of using `potential_power`-derived expected. | Low | Cosmetic; the partial-active-power fallback in widgets handles current day. |
| G9 | `MAX_CONCURRENT_PLANTS=5` (default) — at 1000 plants × 1-minute cycle, this caps throughput. | Medium | Cycles overrun 45 s budget; warned in logs; quality degrades quietly. |
| G10 | External widgets (Capacity Factor, Risk Summary, Grid Outage, etc.) consume keys never written by the service. | Medium | Those dashboards stay blank for new plants unless an external pipeline populates them. |

---

## 11. Approaches per onboarding requirement (≥ 90 % confidence)

### REQ-A — Bulk plant-config ingestion (G2)

| # | Approach | Strengths | Weaknesses | Score |
|---|---|---|---|---|
| A1 | One CSV with one row per plant; nested fields as JSON strings in cells; loader script POSTs `pvlib_config` blob to each TB asset. | Tabular = stakeholder-friendly; trivial diff; small files. | Nested JSON in cells is brittle; complex `orientations`/`iam` get messy. | 7 |
| A2 | One YAML/JSON file per plant in `configs/plants/<plant>.yml`; loader script walks the dir. | Highly readable; per-plant diffs; schema enforceable via JSON Schema. | 1000 files; copy-paste effort; rename collisions. | 7 |
| A3 | Single Excel/Yaml master with per-plant rows + per-template module/inverter/IAM definitions referenced by ID. | Stakeholder edits one file; templates reduce duplication; one source of truth. | Two-format author burden; template change cascades. | 8 |
| A4 | **Hybrid (A3 + A2 generator)** — master YAML in repo; tool emits per-plant JSON + bulk-loads into TB; idempotent re-run; dry-run preview. | Best of both; auditable; reproducible; version-controlled; safe to rerun on attr drift. | Extra tooling step; needs `tb_config_loader.py` script. | **9** |
| A5 | TB Asset Profile + Rule Chain templating (TB-native). | Native TB; defaults flow through. | Pushes complexity into TB; harder version control; no diff history. | 6 |

**Pick A4.** Confidence on rating: 92 %. Hybrid avoids A1's nested-JSON brittleness and A2's file explosion.

### REQ-B — Diverse `active_power` telemetry keys (G1)

| # | Approach | Strengths | Weaknesses | Score |
|---|---|---|---|---|
| B1 | Per-plant `actual_power_key` SERVER_SCOPE attr; `daily_job`+`loss_rollup` read it (default `active_power`). | Zero plant-list hardcoding; widgets already support `actualPowerKeys` setting. | Small service patch (~30 lines across 2 jobs). | **9** |
| B2 | Hardcoded fallback list (`active_power` → `power_v3` → `p341_active_power`). | No new attr. | Brittle; non-deterministic; doesn't scale to truly diverse plants. | 4 |
| B3 | TB Rule Chain renames non-standard keys to `active_power` on ingest. | Zero service change. | Per-plant rule chain work; loses provenance; opaque. | 6 |

**Pick B1.** Confidence: 95 %.

### REQ-C — Past / current / future data backfill on new-plant joins (G4, G5, G8)

| # | Approach | Strengths | Weaknesses | Score |
|---|---|---|---|---|
| C1 | Explicit operator playbook + onboarding script (`onboard_plant.py asset_id`) that runs: audit → refresh → pvalues current year → pvalues prior year → daily-range backfill → loss-rollup backfill → recompute-lifetime. | Idempotent; observable per step; reusable; safe to rerun. | Operator must trigger; missing one step leaves gaps. | 8 |
| C2 | Extend `run_pvalue_newplants_now` (01:00 cron) to also chain daily/loss backfill for any plant added in last 24 h. | Zero ops touch; automated. | Hard to bound runtime; spikes if many plants added simultaneously; harder to observe. | 7 |
| C3 | **Hybrid (C1 + C2)** — C2 handles steady-state (95 % case: 1–5 new plants/day); C1 is the documented escape hatch for bulk imports / re-runs / full historical backfill. | Best of both; degrades gracefully. | Two paths; need a "trigger source" stamp on writes for audit. | **9** |

**Pick C3.** Confidence: 92 %.

### REQ-D — `TB_ROOT_ASSET_IDS` discovery (G6)

| # | Approach | Strengths | Weaknesses | Score |
|---|---|---|---|---|
| D1 | Keep .env-static root list; document `/admin/refresh-plants` for cache invalidation. | Today's design; explicit; safe. | New top-level root requires service restart or .env edit. | 8 |
| D2 | `pvlib_root=true` SERVER_SCOPE attr on roots; service queries `/api/tenant/assets?textSearch=...` at every cycle. | Tag-driven; no .env touch. | Tenant-wide queries are heavy; risk of unintended hierarchy capture. | 5 |
| D3 | Hybrid: .env primary (D1) + admin endpoint `POST /admin/add-root?asset_id=...` that appends to runtime list and invalidates cache. | Allows hot-add without restart; keeps explicit baseline. | Runtime list lost on restart unless persisted. | **8** |

**Pick D1 with D3 as Phase-2 stretch.** Confidence: 93 %.

### REQ-E — Widget-side attrs (`tariff_rate_lkr`, `commissioning_date`, `Capacity`, `active_power_unit`, `setpoint_keys`) (G3, G7)

| # | Approach | Strengths | Weaknesses | Score |
|---|---|---|---|---|
| E1 | Extend `audit_tb_config.py` to validate widget-side attrs and report ERR/WARN per plant. | Catches gaps before widgets render blanks. | Audit-only — does not fix. | 8 |
| E2 | E1 + `tb_config_loader.py` from A4 also ingests these attrs (one master file covers everything). | One-shot bulk; auditable; idempotent. | Master schema bloat. | **9** |

**Pick E2.** Confidence: 93 %.

### REQ-F — TB / server load at 1000 plants (G9)

| # | Approach | Strengths | Weaknesses | Score |
|---|---|---|---|---|
| F1 | Raise `MAX_CONCURRENT_PLANTS` (5 → 20–50); profile cycle duration. | Cheap; documented per-cycle metric already in `/metrics`. | Risks TB rate-limit; need observability. | 8 |
| F2 | Shard discovery across plant groups; round-robin per cycle (each plant processed every N cycles). | Lower per-cycle load. | Stale data; complicates widget freshness. | 5 |
| F3 | Hybrid: F1 to lift throughput; instrument with new Prometheus counters; revisit if cycles still > 45 s p95. | Data-driven; reversible. | Requires monitoring follow-through. | **8** |

**Pick F3.** Confidence: 90 %.

---

## 12. Telemetry contract — addendum

**New SERVER_SCOPE attribute** (read by service; not written):

| Key | Type | Default | Read by |
|---|---|---|---|
| `actual_power_key` | string | `"active_power"` | `daily_job.py`, `loss_rollup_job.py` |

**No new telemetry keys.** No deprecations. Append-only.

If REQ-D Phase-2 (D3) is implemented later, add `POST /admin/add-root` documentation to README — runtime root list is ephemeral by design.

---

## 13. Execution steps — onboarding work (Sonnet 4.7)

Order is strict. Steps 1–4 from §6 of the original plan must land first (FDI/FvA widget fixes).

### Step 8 — Add per-plant `actual_power_key` attribute support (B1)

Files:
- `M:\Documents\Projects\MAGICBIT\Power-Prediction\scripts\Pvlib-Service\app\services\daily_job.py`
- `M:\Documents\Projects\MAGICBIT\Power-Prediction\scripts\Pvlib-Service\app\services\loss_rollup_job.py`

Changes:
1. In `_integrate_actual_day` (`daily_job.py`): replace hardcoded `KEY_ACTUAL_METER_POWER = "active_power"` constant lookup with a per-plant key read from `tb_client.get_asset_attributes(plant_id).get("actual_power_key", "active_power")`. Cache the attribute lookup inside the run-loop (one read per plant per cron run is fine).
2. In `loss_rollup_job._process_plant`: same — read `actual_power_key` per plant; fall back to existing `actualPowerKeys` default list.
3. Update `TELEMETRY_CONTRACT.md` with the new attribute (read-only; not written by service).
4. Update `audit_tb_config.py` to emit a WARN when `actual_power_key` is missing and plant publishes a non-canonical active-power key (heuristic: latest-value lookup for `active_power` returns no row).

Test: pick a plant with `power_v3` as meter key; set `actual_power_key=power_v3`; run `/admin/run-daily?date=YYYY-MM-DD`; verify `actual_daily_energy_kwh` is non-zero.

### Step 9 — Onboarding bulk-config tooling (A4 + E2)

Files (new):
- `M:\Documents\Projects\MAGICBIT\Power-Prediction\scripts\shared\plants_master.yml` — single source-of-truth master; one section per plant.
- `M:\Documents\Projects\MAGICBIT\Power-Prediction\scripts\shared\tb_config_loader.py` — script that reads the master, builds per-plant `pvlib_config` JSON + flat SERVER_SCOPE attrs, posts to TB. Flags: `--dry-run`, `--plant <id>`, `--force-overwrite`, `--diff-only`.
- `M:\Documents\Projects\MAGICBIT\Power-Prediction\scripts\shared\templates/module_<name>.yml` — reusable module/inverter/IAM templates referenced by plant rows.
- `M:\Documents\Projects\MAGICBIT\Power-Prediction\scripts\shared\plants_master.schema.json` — JSON Schema for validation.

Master file fields per plant (minimum):
```
asset_id, name, root_asset_id, latitude, longitude, altitude_m, timezone,
capacity_kwp, capacity_unit, active_power_unit, actual_power_key,
commissioning_date, tariff_rate_lkr,
orientations: [{tilt, azimuth, module_count, use_measured_poa}],
module_template, inverter_template, iam_template,
station: {weather_station_id | null, ghi_key, poa_key, air_temp_key, wind_speed_key},
solcast_resource_id (optional),
losses: {soiling, lid, module_quality, mismatch, dc_wiring, ac_wiring, far_shading, albedo},
thermal_model,
setpoint_keys (CSV string),
pvlib_enabled, loss_attribution_enabled
```

Loader output per plant — writes (via `POST /api/plugins/telemetry/ASSET/{id}/attributes/SERVER_SCOPE`):
- `pvlib_config` JSON blob (preferred — the `_from_blob` parser already handles it).
- Flat companions: `Capacity`, `capacityUnit`, `latitude`, `longitude`, `timezone`, `active_power_unit`, `actual_power_key`, `tariff_rate_lkr`, `commissioning_date`, `setpoint_keys`, `isPlant`, `pvlib_enabled`, `loss_attribution_enabled`.

Idempotency: compute SHA-1 of the JSON blob; skip POST if `pvlib_config_hash` attr equals it. Print "no-op" per plant.

### Step 10 — Extend `audit_tb_config.py` (E1)

File: `M:\Documents\Projects\MAGICBIT\Power-Prediction\scripts\shared\audit_tb_config.py`

Add checks per plant:
- ERR: any of `latitude`, `longitude`, `Capacity`, `timezone`, `orientations`, `module`, `inverter` missing.
- ERR: `active_power_unit` neither `"W"` nor `"kW"`.
- WARN: `actual_power_key` missing AND latest-value `active_power` returns no row.
- WARN: `tariff_rate_lkr` missing (blocks Loss Attribution revenue mode).
- WARN: `commissioning_date` missing (blocks lifetime attribute recompute).
- WARN: `setpoint_keys` missing AND no default setpoint key (`setpoint_active_power`, `curtailment_limit`, `power_limit`) returns latest value.
- WARN: `weather_station_id` missing AND no `solcast_resource_id` (falls to Tier-3 clearsky — acceptable but flag it).
- WARN: `forecast_p50_daily` has fewer than 360 rows for current year (pvalue_job hasn't run for this plant yet).

Exit code: 1 if any ERR.

### Step 11 — Onboarding playbook + entry script (C1)

File (new): `M:\Documents\Projects\MAGICBIT\Power-Prediction\scripts\shared\onboard_plant.py`

Args: `--asset-id <UUID> [--years-back 1] [--skip-loss] [--dry-run]`.

Sequence (all idempotent; each step logs OK / FAIL / SKIP):
1. `audit_tb_config.py --plant <asset_id>` — abort if ERR.
2. `POST /admin/refresh-plants` (invalidate cache so the plant is picked up).
3. `POST /admin/run-pvalues-plant?asset_id=<id>&year=<current>` — current year P-values.
4. For each `year in range(current-years_back, current)`: `POST /admin/run-pvalues-plant?asset_id=<id>&year=<y>`.
5. `POST /admin/run-daily-range?start=<commissioning_date|today-365d>&end=<yesterday>` (single fleet endpoint, scoped via `pvlib_enabled` on the target plant only — others noop).
6. If `loss_attribution_enabled` true: `POST /admin/run-loss-rollup?start=<commissioning_date|today-365d>&end=<yesterday>`.
7. If loss enabled: `POST /admin/recompute-lifetime?asset_id=<id>`.
8. `GET /admin/loss-status?asset_id=<id>` (loss-enabled only) — print summary.
9. `GET /pvlib/discover?root_asset_id=<root>` — confirm plant appears.

### Step 12 — Auto-detect new plants on daily cron (C2)

File: `M:\Documents\Projects\MAGICBIT\Power-Prediction\scripts\Pvlib-Service\app\services\scheduler.py`

Extend `run_pvalue_newplants_now` (existing 01:00 cron):
- After P-value generation for the missing-plants list, for each newly P-valued plant also enqueue:
  - One-time historical backfill via internal call to `run_daily_rollup` for the last 30 days (bounded).
  - If `loss_attribution_enabled=true`: one-time `run_loss_rollup` for the last 30 days.
- Add a per-plant `onboarding_backfilled_at` SERVER_SCOPE attribute as the "done" marker; the cron skips plants that already have it.
- New env flag `AUTO_ONBOARD_BACKFILL_ENABLED` (default `false`) — must be opt-in. Document in README.

### Step 13 — Raise concurrency + observability (F3)

File: `M:\Documents\Projects\MAGICBIT\Power-Prediction\scripts\Pvlib-Service\app\config.py`

- Default `MAX_CONCURRENT_PLANTS = 5` → leave default; document that at > 200 plants operators should set `MAX_CONCURRENT_PLANTS=20` in `.env`.
- Add to `/metrics`: `pvlib_cycle_plants_per_minute` (gauge), `pvlib_cycle_duration_p95_ms` (gauge over rolling 60 cycles).
- Document operational SLO in README: cycle p95 should stay < 45 s; if not, increase concurrency or shard.

### Step 14 — Documentation updates

Files:
- `TELEMETRY_CONTRACT.md` — append "Plant SERVER_SCOPE attributes the service reads" table, including `actual_power_key`.
- `scripts/Pvlib-Service/README.md` (or create `ONBOARDING_GUIDE.md` alongside the other plan docs) — explain `plants_master.yml` workflow, the onboarding playbook script, and what each cron does.
- Note in `KSP_TEST_RUNBOOK.md` that the runbook covers the manual single-plant path; refer multi-plant operators to the new onboarding guide.

### Step 15 — Validation

For two real plants from the fleet (one with weather station, one without; one in kW, one in W):

1. Set `pvlib_enabled=true`, push attrs via `tb_config_loader.py --plant <id> --dry-run` then commit.
2. Run `audit_tb_config.py --plant <id>` — expect 0 ERR, optionally some WARN.
3. Run `onboard_plant.py --asset-id <id> --years-back 1`.
4. After completion:
   - `forecast_p*_daily/weekly/monthly/mtd` rows exist for current and prior year.
   - `actual_daily_energy_kwh` exists for every day with `active_power` data.
   - `loss_*_daily_kwh` exists if loss enabled.
   - Open FDI + FvA widgets pointed at the new plant; both render without "no data" states.
   - Open Loss Attribution `grid`/`curtail`/`revenue` modes — fast path served from precomputed keys.
   - Open Curtailment V5 — `potential_power` dashed line visible.

---

## 14. File index — onboarding work (Sonnet 4.7 will touch)

Service code:
- `app/services/daily_job.py` — per-plant `actual_power_key` lookup.
- `app/services/loss_rollup_job.py` — per-plant `actual_power_key` lookup.
- `app/services/scheduler.py` — extend `run_pvalue_newplants_now` (only if `AUTO_ONBOARD_BACKFILL_ENABLED=true`).
- `app/config.py` — new flag `AUTO_ONBOARD_BACKFILL_ENABLED`.
- `app/api/forecast.py` — `/metrics` additions; optional `POST /admin/add-root` for D3 (defer).

Service docs:
- `TELEMETRY_CONTRACT.md` — append SERVER_SCOPE attr table.
- `ONBOARDING_GUIDE.md` (new).

Shared tooling (new):
- `scripts/shared/plants_master.yml`
- `scripts/shared/plants_master.schema.json`
- `scripts/shared/tb_config_loader.py`
- `scripts/shared/onboard_plant.py`
- `scripts/shared/audit_tb_config.py` (extended, not rewritten).

Widgets:
- None — widgets already support per-plant attrs via settings; no widget code change is required for onboarding (FDI/FvA v4.0 changes in §6 cover the widget side independently).

---

## 15. Out of scope (do not silently invent)

These widgets consume keys never produced by Pvlib-Service. Onboarding does **not** populate them; the operator (or an external pipeline) must:

- **Capacity Factor Compliance** — `contract_cf_target` (commercial input, per PPA), `actual_cf_ytd` (commercial calc). Note in onboarding guide: "Set these on plant asset via TB UI or your PPA-tracking pipeline."
- **Grid Outage Timeline / Event Summary** — `grid_outage_events`, `insurance_claims_data`. JSON arrays; either populated by an external SCADA/event pipeline or manually entered.
- **Risk Summary Panel** — `revenue_at_risk`, `risk_alert_level`, `risk_percentile`, `tracking_delta`. Same — external risk pipeline.
- **Expected vs Actual Revenue** — DS[0]/DS[1] are any TB keys; operator chooses (e.g., `forecast_p50_monthly × tariff` and `actual_daily_energy_kwh × tariff`). Document the recommended mapping in `ONBOARDING_GUIDE.md` but don't auto-write.

Sonnet must NOT add service-side computation for these. Adding them would couple Pvlib-Service to commercial/insurance domains that belong elsewhere.

Also out of scope:
- D3 (`POST /admin/add-root`) — defer to a Phase 2 plan; not blocking 1000-plant rollout.
- Any change to PVGIS-ERA5 sourcing, percentile algorithm, or annual cron cadence.
- TB Rule Chain authoring (rule chains belong with the TB integrator, not in this repo).

---

## 16. Open questions for Naveen (non-blocking)

1. **`plants_master.yml` source of truth** — does it live in this repo (alongside the service) or in a separate ops repo? In this repo at `scripts/shared/plants_master.yml`, treated as data not code.
2. **Onboarding cadence** — when a new plant is added, who runs `onboard_plant.py`? Integrator runs it manually for each new plant during commissioning; the C2 auto-detect path (Step 12) handles drift if missed.
3. **Loss attribution default** — should `loss_attribution_enabled` default to `true` when adding a new plant via `tb_config_loader.py`? Plan §6 (Loss Attribution) defaulted to `true` when missing; carry that forward.
4. **`AUTO_ONBOARD_BACKFILL_ENABLED` default** — keep at `false` (safe) and require explicit ops opt-in? Recommended yes; flip to `true` once the bounded-backfill window proves stable in production.
5. **Concurrency tuning** — what is the production EC2 instance class? At t4g.small (per `ec2_userdata.sh`) the cap is ~10 concurrent plants safely; at t4g.medium ~30. Confirm before raising `MAX_CONCURRENT_PLANTS` past 10.

---

# Addendum B — Phase 3 Pvlib Service Hardening (2026-05-14, second pass)

Author: Opus 4.7 · Trigger: deeper review specifically hunting service-side gaps a Sri Lanka → multi-region 1000-plant rollout will hit.

This addendum is purely additive. It introduces a Phase 3 (service hardening) that lands after Phase 2 onboarding plumbing (§11–§16). Some items are critical data-correctness bugs; others are scaling/ergonomics. Each is verified against the current code, not assumed.

---

## 17. Newly-identified Pvlib gaps (verified against code)

| # | Gap | Severity | Evidence |
|---|---|---|---|
| G11 | **`daily_job.py` integrates `active_power` without W→kW unit scaling.** Plants publishing in W produce `actual_daily_energy_kwh` 1000× too high. | **CRITICAL** | `daily_job.py:286–336` reads `active_power` and integrates with no `active_power_unit` lookup. Compare: `loss_rollup_job.py:374–375` correctly applies `w_to_kw = (active_power_unit.upper() == "W")` and scales. Same input, two different transforms. |
| G12 | **`daily_job.py` reads a single hardcoded `active_power` key.** `loss_rollup_job.py` already supports CSV `actual_power_keys` attribute (`loss_rollup_job.py:381–383`). Plants with inverter-sum meters break in daily_job but work in loss_rollup. | High | Inconsistency. |
| G13 | **Day-boundary jobs use `settings.TZ_LOCAL` (global) instead of per-plant `config.timezone`.** Lat/lon already drive a per-plant tz in `PlantConfig`; the physics pipeline respects it, but daily_job + loss_rollup don't. | Medium (low impact while fleet is Asia/Colombo only) | `daily_job.py:69,241,322`, `loss_rollup_job.py:123,256,698,699,829`, `pvalue_job.py:109`, `weekly_eval.py:97`. |
| G14 | **No orphan-plant diagnostic.** Plants with `isPlant=true && pvlib_enabled=true` but not under any `TB_ROOT_ASSET_IDS` ancestor are silently invisible. | Medium | BFS in `thingsboard_client._bfs_discover` only walks from configured roots. |
| G15 | **No config-drift detector.** `forecast_service._attach_config_metadata` already writes `pvlib_config_hash` per cycle, but no tool compares it against the master file (Step 9, A4) to catch manual TB edits diverging from source-of-truth. | Low | `forecast_service.py:KEY_CONFIG_HASH` exists; usage is one-way. |
| G16 | **`set_active_power_unit.py` hardcodes the plant→unit map.** At 1000 plants and ongoing additions this hardcoded list will rot. | Low | `scripts/shared/set_active_power_unit.py`. |
| G17 | **Cron clustering at midnight + 00:05 + 00:10 + 01:00 + 03:00.** No write-storm yet (sequential with sem), but on EC2 t4g.small the combined load is the daily peak. Stagger by plant hash would smooth it. | Low | `scheduler.py:347–438`. |

G11 and G12 are blocking for any non-canonical plant — every plant publishing `active_power` in W (~21 plants per TELEMETRY_CONTRACT.md plant unit map) is silently producing wrong daily energy. **This is a Phase 1.5 fix; it must land before or alongside the widget rewrite or FDI/FvA will display wrong actuals.**

---

## 18. Approaches per Phase-3 requirement (≥ 90 % confidence)

### REQ-G — `daily_job` W→kW scaling (G11) — CRITICAL

| # | Approach | Strengths | Weaknesses | Score |
|---|---|---|---|---|
| G-A | Replicate `loss_rollup_job._process_plant` pattern: read `active_power_unit` SERVER_SCOPE attr, scale × 0.001 when `"W"`. Default `"kW"`. | One-line fix per integration function; mirrors existing tested pattern; restores data correctness. | None. | **10** |
| G-B | Push scaling onto widget side. | Zero service change. | Inconsistent with loss_rollup; widgets read `actual_daily_energy_kwh` and trust it's in kWh; would propagate divergence across all energy widgets. | 2 |
| G-C | TB rule chain rescales `active_power` to a canonical `active_power_kw` key on ingest. | Zero service change. | Per-plant rule chain bloat; doesn't fix existing W-published historical data. | 5 |

**Pick G-A.** Confidence: 98 %.

### REQ-H — `daily_job` multi-key support (G12)

| # | Approach | Strengths | Weaknesses | Score |
|---|---|---|---|---|
| H-A | Read per-plant `actual_power_keys` (CSV, plural — matches loss_rollup convention). First-match wins. Default `"active_power"`. | Aligns the two jobs; widgets already accept `actualPowerKeys` CSV; supports inverter-sum plants via a single canonical key. | ~20 line change. | **9** |
| H-B | Sum across all keys. | Correct when inverter_1+inverter_2 are listed. | Double-counts if plant_total also listed; mode is implicit. | 6 |
| H-C | Hybrid w/ `actual_power_keys_mode = "first_match" \| "sum"` attr. | Explicit. | Yet another attr; rarely needed in practice. | 7 |

**Pick H-A.** Inverter-sum plants get a virtual `active_power_total` via TB rule chain (operator responsibility, not service). Confidence: 92 %.

### REQ-I — Per-plant timezone (G13)

| # | Approach | Strengths | Weaknesses | Score |
|---|---|---|---|---|
| I-A | In day-boundary code paths inside `daily_job._integrate_*_day`, `daily_job.run_daily_rollup`, and `loss_rollup_job._process_plant`, use `tz = ZoneInfo(config.timezone or settings.TZ_LOCAL)`. Cron firing time stays global. | Correct day boundaries per plant; small surgical change; no scheduler refactor. | Cron must fire late enough to cover the easternmost plant. For Sri Lanka-only this is unchanged. Doc note required. | **8** |
| I-B | Separate cron per plant timezone group. | Fully decoupled. | APScheduler complexity; observability harder. | 5 |
| I-C | Defer entirely until first non-Asia/Colombo plant. | Zero work. | Time-bomb; rolled into onboarding playbook means harder later. | 7 |

**Pick I-A** with a documented constraint that the cron's `hour=0` fires in `settings.TZ_LOCAL` which must be ≥ the latest plant timezone − 4 h safety margin. For Sri Lanka fleet (UTC+5:30) firing at `Asia/Colombo 00:05` covers everything from UTC+1 eastward. Confidence: 90 %.

### REQ-J — Orphan plant detection (G14)

| # | Approach | Strengths | Weaknesses | Score |
|---|---|---|---|---|
| J-A | New ops script `scripts/shared/find_orphan_plants.py`: paginate `GET /api/tenant/assets`, filter `pvlib_enabled=true`, subtract the BFS-discovered set, list orphans with their attributes. | Catches misconfigured / mis-parented plants; one-shot ops tool. | Tenant-wide query (~ N pages of 100 assets); acceptable as ops tool, not for every cycle. | **9** |
| J-B | Periodic service-side check; emit `pvlib_orphan_count` gauge in `/metrics`. | Continuous; alertable. | Extra TB query every cycle; risk of false positives during BFS race. | 6 |
| J-C | Hybrid: J-A as primary; optional cron under flag `ORPHAN_CHECK_ENABLED=false`. | Adds eventually-on monitoring. | Minor flag bloat. | 8 |

**Pick J-A** for Phase 3. C is a Phase 4 stretch. Confidence: 92 %.

### REQ-K — Config drift detector (G15)

| # | Approach | Strengths | Weaknesses | Score |
|---|---|---|---|---|
| K-A | New ops script `scripts/shared/find_config_drift.py`: walks `plants_master.yml`, computes the same SHA-1 the service writes as `pvlib_config_hash`, fetches the TB-side value, prints divergences. | Reuses existing telemetry; cheap. | None. | **9** |
| K-B | Service-side periodic comparison + Prometheus gauge. | Continuous. | Master file lives outside the container by default; mount required. | 6 |

**Pick K-A.** Master file is a repo artefact; loader path already mounts it for the bulk loader. Confidence: 95 %.

### REQ-L — Active-power-unit map (G16)

| # | Approach | Strengths | Weaknesses | Score |
|---|---|---|---|---|
| L-A | Retire `set_active_power_unit.py`; absorb the unit mapping into `plants_master.yml`. Bulk loader (Step 9) writes `active_power_unit` from the master file. | Single source of truth; one operator file; no rot. | Loader must be run for any new plant. (Already true.) | **9** |
| L-B | Keep `set_active_power_unit.py` but read the map from a sibling CSV. | Familiar workflow. | Two sources of truth; drift risk. | 5 |

**Pick L-A.** Confidence: 95 %.

### REQ-M — Cron staggering (G17)

Defer. Current single-instance load is acceptable for 1000 plants on t4g.small per Phase 2 metrics. Revisit if `/metrics` shows midnight burst > 80 % CPU sustained > 60 s. No approach scored; capture as Open Question §22.

---

## 19. Telemetry contract additions (Phase 3)

**New SERVER_SCOPE attribute read by service:**

| Key | Type | Default | Read by |
|---|---|---|---|
| `actual_power_keys` | string (CSV) | `"active_power"` | `daily_job.py` (new) + `loss_rollup_job.py` (existing) |

The Phase-2 plan introduced `actual_power_key` (singular). **Override that to `actual_power_keys` (plural CSV)** so daily_job and loss_rollup use the identical attribute name. Update Step 8 of §13 accordingly.

No telemetry keys added. No deprecations.

---

## 20. Execution steps — Phase 3

Order is strict. Steps 1–7 (widget fixes) and Steps 8–14 (onboarding plumbing) must be in place first, EXCEPT Step 16 below which is critical and may land alongside Step 8.

### Step 16 — Fix `daily_job` W→kW scaling + multi-key support (G-A + H-A) — CRITICAL

File: `app/services/daily_job.py`

Changes inside `run_daily_rollup` (per-plant loop, around line 107):

1. Before calling `_integrate_actual_day(...)`, fetch `attrs = await tb_client.get_asset_attributes(plant_id)`. Cache for the rest of the per-plant block.
2. Parse `unit = str(attrs.get("active_power_unit") or "kW").strip().upper()`; `scale_w_to_kw = (unit == "W")`.
3. Parse `keys_csv = attrs.get("actual_power_keys") or "active_power"`; `actual_power_keys = [k.strip() for k in str(keys_csv).split(",") if k.strip()]`.
4. Pass both into `_integrate_actual_day(tb_client, plant_id, day_start_utc, day_end_utc, actual_power_keys, scale_w_to_kw)`.

Changes inside `_integrate_actual_day` (around line 286):

1. Accept the two new args.
2. Iterate `actual_power_keys` until the first key returns non-empty records (first-match semantics).
3. After parsing to `pd.Series`, if `scale_w_to_kw`: `series = series * 0.001`.
4. Integrate as before.

Sentinel/edge-case behavior unchanged. Test: a plant with `active_power_unit="W"` should produce ~0.001× of the previous (wrong) value, matching what loss_rollup_job records for the same day.

Add a single-line WARN log when scaling is applied so the migration is observable.

### Step 17 — Replace `actual_power_key` (singular) with `actual_power_keys` (plural CSV) in plan §13 Step 8

Edit plan §13 Step 8 mention of `actual_power_key` → `actual_power_keys` (CSV). Plan addendum already documents this in §19.

Behavioural note: a plant config that previously used `actual_power_key` should be re-issued as `actual_power_keys`. Bulk loader (Step 9) emits the plural; legacy singular is read as a fallback (one-line read with `or` chain) — add a 6-line legacy adapter inside both daily_job and loss_rollup_job, log WARN once per plant when legacy is used.

### Step 18 — Per-plant timezone for day-boundaries (I-A)

Files:
- `app/services/daily_job.py`
- `app/services/loss_rollup_job.py`

Changes:

1. In each per-plant block, after `attrs = await tb_client.get_asset_attributes(plant_id)`, parse `plant_tz = ZoneInfo(attrs.get("timezone") or settings.TZ_LOCAL)`.
2. Replace `settings.TZ_LOCAL` usages **inside the per-plant integration path** with `plant_tz`. The cron firing time, `service_started_at`, and ancestor roll-up logic continue to use `settings.TZ_LOCAL` — the service is run from one host.
3. `day_ts_ms` (the timestamp under which the daily record is written to TB) must be computed in `plant_tz`'s local midnight, not the service's.

Edge case to verify: a TB asset roll-up that aggregates plants in different timezones will get rows at different `day_ts_ms` values. Document in `TELEMETRY_CONTRACT.md` that "daily roll-up rows are stamped at each plant's local midnight" so widget aggregation is correct.

For the Sri Lanka-only fleet this is a no-op (all plants `Asia/Colombo`). The change is forward-compatible insurance.

### Step 19 — Orphan plant detection (J-A)

File (new): `scripts/shared/find_orphan_plants.py`

Logic:
1. Authenticate to TB using `.env` credentials.
2. Paginated GET `/api/tenant/assets?pageSize=100&page=N` until exhausted.
3. For each asset, fetch SERVER_SCOPE attrs; keep those with `isPlant=true && pvlib_enabled=true`.
4. Call `/pvlib/discover` to get the discovered set.
5. Subtract; print orphans (id, name, root attrs) in a table.

Output formats: `--format table|json|csv`. Exit code 1 if any orphans found (so CI can gate).

### Step 20 — Config drift detector (K-A)

File (new): `scripts/shared/find_config_drift.py`

Logic:
1. Load `plants_master.yml`.
2. For each plant, compute the SHA-1 the same way `forecast_service._config_hash` does (`json.dumps(payload, sort_keys=True, default=str)`, hexdigest()[:12]).
3. Fetch `pvlib_config_hash` from TB SERVER_SCOPE.
4. Print divergence table: `asset_id | name | master_hash | tb_hash | status`.
5. Flags: `--fix` (force-rewrites diverging plants via the bulk loader), `--report-only`.

Verifying that the hash function is identical on both sides is critical — co-locate the helper in `scripts/shared/_config_hash.py` and import it from both the service (via a small refactor in forecast_service) and the script.

### Step 21 — Absorb `active_power_unit` map into master file (L-A)

Files:
- `scripts/shared/plants_master.yml` (extended in Step 9): add `active_power_unit: "kW" | "W"` per plant.
- `scripts/shared/tb_config_loader.py` (Step 9): write the attribute as part of normal load.
- `scripts/shared/set_active_power_unit.py`: deprecate. Keep file with a single line `raise SystemExit("Deprecated; use tb_config_loader.py — see ONBOARDING_GUIDE.md")`. Remove after 90-day deprecation window.

### Step 22 — Documentation

- `TELEMETRY_CONTRACT.md`: under "Plant SERVER_SCOPE attributes read by service" table (added in Step 14), include `actual_power_keys` (CSV), `active_power_unit`, `timezone`, `actual_power_key` (legacy alias).
- `ONBOARDING_GUIDE.md`: document `find_orphan_plants.py`, `find_config_drift.py`, the deprecation of `set_active_power_unit.py`, and the per-plant timezone constraint (cron firing time vs plant tz).
- Note in `KSP_TEST_RUNBOOK.md`: after Phase 1.5 lands, KSP_Plant must have `actual_power_keys="EnergyMeter_active_power"` (or the precise meter key) set on its SERVER_SCOPE; verify with `curl …/admin/run-daily?date=…` matches yesterday's actual within 0.5 %.

### Step 23 — Validation (Phase 3)

For one W-publishing plant (e.g. `AKB Kelaniya`) and one kW-publishing plant (e.g. `KSP_Plant`):

1. Run `POST /admin/run-daily?date=<recent>`.
2. Read TB `actual_daily_energy_kwh` for that day.
3. Confirm both plants have plausible kWh values (between 0 and ~capacity × 8 h × 0.85).
4. For the W-publishing plant, confirm the new value is 1/1000 of the pre-fix value (if a pre-fix value exists in history).
5. Run `find_orphan_plants.py` against the full tenant. Expect either zero orphans or a documented short list.
6. Run `find_config_drift.py` against `plants_master.yml`. Expect zero drift (or a report explaining each divergence).

---

## 21. File index — Phase 3 (Sonnet 4.7 will touch)

Service code:
- `app/services/daily_job.py` — W→kW scaling, multi-key support, per-plant tz.
- `app/services/loss_rollup_job.py` — per-plant tz (W→kW + multi-key already there).
- `app/services/forecast_service.py` — extract `_config_hash` helper into a shared module so the drift script can import it.

Service docs:
- `TELEMETRY_CONTRACT.md` — append attr table; clarify "daily ts is plant-local midnight".
- `ONBOARDING_GUIDE.md` (created in Phase 2) — append diagnostic-script docs.

Shared tooling (new):
- `scripts/shared/_config_hash.py` (extracted helper).
- `scripts/shared/find_orphan_plants.py`.
- `scripts/shared/find_config_drift.py`.

Deprecation:
- `scripts/shared/set_active_power_unit.py` — replace body with deprecation raise.

Widgets:
- None. All Phase 3 changes are service-side. Widgets already do unit normalization for live `active_power` (per `TELEMETRY_CONTRACT.md` §Widget scaling recipe).

---

## 22. Out of scope (Phase 3)

- Cron staggering (REQ-M). Defer until `/metrics` shows actual midnight burst pain.
- Per-plant cron (one job-per-plant). Massive scheduler complexity; not worth it at current load.
- Auto-creation of TB asset entities. Onboarding assumes the asset exists; creating it via API requires tenant-admin scope and is a TB-integrator task.
- Schema migration tooling for `pvlib_config` shape changes. The bulk loader rewrites the blob from the master file, so version-bumping the schema is handled by editing the master.
- Multi-tenancy / multi-customer isolation. Service assumes one tenant per deployment.

---

## 23. Open questions (Phase 3, non-blocking)

1. **W→kW retroactive fix** — should we re-run `/admin/run-daily-range` for the past 12 months for every W-publishing plant to correct historical `actual_daily_energy_kwh`? Recommendation: yes, after Phase 1.5 verification, scoped to the ~21 plants in the W-publishing list.
2. **`timezone` attribute hygiene** — current `PlantConfig.timezone` default is `"UTC"`. Should the audit (Step 10) WARN when a plant has `timezone="UTC"` AND `latitude` is non-zero (likely a forgotten override)? Recommendation: yes.
3. **`actual_power_keys` first-match vs sum** — when a plant exposes both `EnergyMeter_active_power` and `power_v3`, which should win? Recommendation: first key in the CSV wins; document this convention in the contract.
4. **Drift detector severity** — should `find_config_drift.py` exit non-zero on any drift (CI-blocking) or only on schema-breaking drift? Recommendation: non-zero on any drift; provide `--report-only` flag for inspection.
5. **Phase 1.5 deployment gate** — should the widget v4.0 work (Phase 1) be blocked until the daily_job W→kW fix (Step 16) lands? Argument for yes: FDI/FvA's actuals come from `actual_daily_energy_kwh`; if it's wrong for W-plants, the widget's numbers are wrong even with correct code. Recommendation: ship Step 16 first, then Phase 1.

---

# Addendum C — Phase 4 Curtailment Correctness + Revenue + Atomic-Group Switches + Zero-Touch Hosting (2026-05-14)

Author: Opus 4.7 · User-confirmed taxonomy (2026-05-14): 5 atomic service groups; service writes monthly + yearly LKR revenue keys (yearly view spans 10 years); weekly auto-onboard at Sunday 03:00 with full historical backfill since `commissioning_date`; operator-level hosting doc with Linux/EC2/Docker baseline.

This addendum is additive. It introduces Phase 4 (correctness, revenue, automation, hosting) that lands after Phase 3 verifies.

---

## 24. New gaps and requirements

| # | Gap / Requirement | Severity | Evidence |
|---|---|---|---|
| G18 | **`loss_rollup_job._integrate` is cadence-dependent.** Uses `resample("1min").mean().reindex(idx)` → only minute-buckets that have a sample contribute; for 5-min plants the integration loop sums 12 samples/hr × (1/60) = 0.20 × kW per hour, vs the correct ~1.0 × kW per hour. Loss/potential/exported kWh under-reported ~5× on 5-min plants, ~15× on 15-min plants. | **CRITICAL** | `loss_rollup_job.py:578–614`. Compare correct trapezoidal `_time_weighted_kwh` in `daily_job.py:258–283`. |
| G19 | **Expected vs Actual Revenue widget is unwired to service.** Widget expects DS[0]/DS[1] monthly keys; nothing in pvlib-service writes revenue. Widget falls back to a hardcoded baseline of `2.475`. | High (cosmetic until needed; once enabled, blocking) | `Widgets/Forecasts & Risk/Expected vs Actual Revenue/.js:82–113`. |
| G20 | **Revenue 10-year view requires actual + expected yearly LKR.** Today's keys: `total_generation_expected_yearly_kwh` (cumulative-by-day) exists; `actual_yearly_energy_kwh` does NOT exist; no LKR yearly keys exist. | High | `daily_job.py:_safe_write_daily`. |
| G21 | **No atomic service-group switches per plant.** `pvlib_enabled` is binary master only. Operator cannot turn off, e.g., loss-attribution for a plant that has no setpoint device. Every plant runs every job → wasted fetches. | High | `thingsboard_client._visit_node:452–454`. |
| G22 | **Self-onboarding incomplete.** `run_pvalue_newplants_now` (01:00 daily) covers P-value generation only. No automatic daily-energy / loss-attribution / lifetime-recompute backfill. Phase 2 Step 12 added a 30-day window under `AUTO_ONBOARD_BACKFILL_ENABLED`; the user now wants **full historical** since `commissioning_date`. | High | `scheduler.py:262–307`. |
| G23 | **Service-attribute re-read cadence is per-cycle (every minute).** Each cycle hits `get_asset_attributes` per plant. With 1000 plants × 1-minute cycle = 1.44 M attr fetches/day. Cache lives 5 min in `_station_resolution_cache` but is for station/device resolution only — full attrs re-fetched every cycle. | Medium | `forecast_service._load_config:472–490`. |
| G24 | **TB telemetry chunking is partly done.** `forecast_service._build_ac_telemetry` merges keys at the same ts into one record (good). `daily_job._safe_write_daily` posts two separate records (daily + weekly — different ts, justified). `pvalue_job._process_plant` posts 4 separate calls (daily, weekly, monthly, mtd) — but **daily and mtd share the same ts (local midnight per day) and can be merged**, halving the per-plant write count for the annual job. 365 calls → 365 records-in-one-call vs 730 calls. Pure win. | Medium | `pvalue_job.py:276–280`. |
| G25 | **Zero hoster-facing documentation.** Existing docs (`KSP_TEST_RUNBOOK.md`, `TELEMETRY_CONTRACT.md`, `LOSS_ATTRIBUTION_TELEMETRY_PLAN.md`) target the project author. A new operator with EC2 + Docker familiarity has no end-to-end guide. | High | Project tree. |

G18 is data-correctness and must land alongside Phase 1.5 (W→kW). All curtailment loss values today on non-1-min plants are wrong.

---

## 25. Approaches per Phase-4 requirement (≥ 90 % confidence)

### REQ-N — Cadence-correct loss integration (G18) — CRITICAL

| # | Approach | Strengths | Weaknesses | Score |
|---|---|---|---|---|
| N-A | Replace per-minute loop with trapezoidal integration that mirrors `daily_job._time_weighted_kwh`: walk samples in cadence-native order, accumulate `0.5 × (kw[i] + kw[i-1]) × dt_h`, skip gaps > 30 min. Run separately for potential, actual; align on a shared sample-pair timeline for the gross/curtail computations. | Cadence-agnostic; consistent with daily_job (single integration semantics across service); preserves gap-handling. | ~80 LOC rewrite; setpoint step-hold logic still needs the 1-min grid for the curtailment-ceiling lookup. | **9** |
| N-B | Detect median cadence (`Δt = median(diff(ts))`) and weight each non-NaN minute by `Δt/(1/60)`. | Surgical; small diff. | Wrong when cadence varies within the day (e.g., 5-min daytime + 15-min night). Doesn't fix gap handling. | 6 |
| N-C | Forward-fill resample (`.ffill()`) then sum × (1/60). | One-line fix. | Over-counts after outages (last known value held across gap), inflating exported energy. | 5 |

**Pick N-A.** Confidence: 93 %. Keep the 1-min grid only for the setpoint step-hold + curtail-ceiling logic; integrate everything else on the native timeline.

### REQ-O — Atomic service-group switches (G21)

| # | Approach | Strengths | Weaknesses | Score |
|---|---|---|---|---|
| O-A | New SERVER_SCOPE attribute `pvlib_services` (JSON dict, 5 keys). Read once during `discover_plants` BFS into a `PlantRef.services` dict; each cron checks its own flag before processing the plant. `pvlib_enabled` remains the master gate. | One attribute to set per plant; aligns to existing 5 cron boundaries; default `{all true}` when missing. | One new attribute; cron code must read it. | **9** |
| O-B | Bit-flag string (`pvlib_services="LDLPR"`). | Tiny payload. | Cryptic; brittle to ordering; bad ergonomics. | 4 |
| O-C | 5 boolean attributes (`pvlib_physics_live=true`, …). | Simple key-by-key reads. | 5 attrs vs 1; harder to default-set in bulk; visual clutter in TB UI. | 7 |

**Pick O-A.** Confidence: 92 %. Storage shape:
```json
{
  "physics_live":     true,
  "daily_energy":     true,
  "loss_attribution": true,
  "p_values":         true,
  "revenue":          true
}
```
Default when attribute is absent: all `true` (preserve current behaviour). `pvlib_enabled=false` master gate still skips everything.

### REQ-P — Revenue telemetry (monthly + yearly, G19 + G20)

| # | Approach | Strengths | Weaknesses | Score |
|---|---|---|---|---|
| P-A | New `revenue_job.py` writing 4 keys: `expected_revenue_monthly_lkr`, `actual_revenue_monthly_lkr`, `expected_revenue_yearly_lkr`, `actual_revenue_yearly_lkr`. Monthly cron 1st-of-month 00:15; yearly cron 1st-of-year 00:20. Idempotent backfill via `/admin/run-revenue?year=N&month=M`. | Clean separation; widget reads 4 simple keys; service controls tariff snapshotting at compute time. | One new service module + 2 new crons + 4 new TB keys. | **9** |
| P-B | Piggyback into `daily_job` at end-of-month detection. | No new module. | daily_job bloat; month-boundary detection coupling; harder to backfill. | 6 |
| P-C | Widget computes (P50_monthly × tariff + Σ actual_daily × tariff). | Zero new service keys. | Widget rewrite; per-cycle 12–365 row fetch; doesn't satisfy "10-year yearly view" without backfill of historical p_values per year. | 5 |

**Pick P-A.** Confidence: 94 %. **10-year yearly view** requires backfill: for each plant, run `pvalue_job` for years `current−9 .. current−1` (writes annual `p50_energy` attribute is overwritten per-run — keep yearly LKR pre-computed in TB rather than relying on the attribute). Service writes one row per year (Jan-1-of-year ts) into `expected_revenue_yearly_lkr` and `actual_revenue_yearly_lkr`. Pre-commissioning years receive sentinel `-1`.

### REQ-Q — Weekly auto-onboard cron (G22)

| # | Approach | Strengths | Weaknesses | Score |
|---|---|---|---|---|
| Q-A | Sunday 03:00 local cron `pvlib_autoonboard`. For every discovered plant where `onboarding_completed != true`, run the full backfill chain (P-values for `years_back` years, daily-range from `commissioning_date` to yesterday, loss-rollup from `commissioning_date` to yesterday, recompute-lifetime), then set `onboarding_completed=true` + `onboarding_completed_at=<ISO>`. Idempotent; safe to re-run. | Single weekly job covers all newcomers; zero ops touch after attr setup; preserves the existing daily P-value detector for monitoring. | If many plants added in one week, runtime can exceed an hour. Need a per-plant timeout. | **9** |
| Q-B | Daily 01:30 cron (extend `run_pvalue_newplants_now`). | Tighter latency. | More frequent fleet attribute reads; same correctness. | 7 |
| Q-C | On-demand only (no cron). | Maximum control. | Contradicts "hoster touches nothing". | 5 |

**Pick Q-A.** Confidence: 92 %. Bound per-plant runtime to 15 min; on timeout, leave `onboarding_completed` unset so the next Sunday picks it up. Emit Prometheus counter `pvlib_autoonboard_completed_total` and `pvlib_autoonboard_failed_total`.

### REQ-R — TB telemetry chunking by frequency (G24)

| # | Approach | Strengths | Weaknesses | Score |
|---|---|---|---|---|
| R-A | In `pvalue_job._process_plant`, merge daily + mtd records by `ts` before posting (both at local midnight per day → 1 call per day instead of 2). Keep weekly + monthly separate (different ts). Add a `_merge_records_by_ts` helper used by any future writer. | 50 % write-call reduction for the annual job; reusable utility; trivial diff. | None. | **9** |
| R-B | Merge all p_value records by ts and post once. | Smallest payload count. | Different cadences (monthly = 12 rows, weekly = 52, daily = 365) → mismatched ts; TB only accepts shared-ts merges within a single record. Won't work cleanly. | 3 |
| R-C | Move every write through a service-wide "batch_write" queue. | Generic; back-pressure aware. | Heavy refactor; latency cost; not needed at current scale. | 5 |

**Pick R-A.** Confidence: 95 %. Verify `daily_job._safe_write_daily` already merges daily keys (it does); the weekly write is justified separate (different ts).

### REQ-S — Hosting documentation (G25)

| # | Approach | Strengths | Weaknesses | Score |
|---|---|---|---|---|
| S-A | Single mega-doc `ONBOARDING_GUIDE.md`. | One file to read. | Hard to scan; mixes quickstart and deep reference. | 7 |
| S-B | Two-doc split: `HOSTING_QUICKSTART.md` (5 pages, copy-paste runbook) + `HOSTING_REFERENCE.md` (telemetry contract by area, TB attr schemas, widget mapping, troubleshooting). | Layered reading; operator only needs the first to deploy; reference for issues. | Two files; maintenance overhead minor. | **9** |
| S-C | Docs subfolder with topic-per-file. | Most structured. | Discoverability burden for first-time reader. | 7 |

**Pick S-B.** Confidence: 93 %. Quickstart targets an operator with `ssh`, `aws cli`, `docker compose`, basic `.env` editing. Reference enumerates every TB attr / telemetry key / widget setting needed; widget mapping captured as a per-widget recipe.

---

## 26. Telemetry contract — Phase 4 additions

**Read attributes (added):**

| Key | Type | Default | Read by |
|---|---|---|---|
| `pvlib_services` | JSON dict (5 boolean keys) | `{all true}` | Discovery + every cron |
| `commissioning_date` | string (ISO date) | none (auto-onboard requires it) | `revenue_job`, `auto_onboard` |
| `onboarding_completed` | boolean | `false`/absent | `auto_onboard` |
| `onboarding_completed_at` | string (ISO) | absent | Diagnostic only |
| `tariff_rate_lkr` | float | none | Already read by `loss_rollup_job`; now also `revenue_job` |

**Written telemetry (added):**

| Key | Type | Unit | Cadence | Description |
|---|---|---|---|---|
| `expected_revenue_monthly_lkr` | timeseries | LKR | monthly, ts = 1st-of-month midnight local | `forecast_p50_monthly_kwh × tariff_rate_lkr` |
| `actual_revenue_monthly_lkr` | timeseries | LKR | monthly, ts = 1st-of-month midnight local | `Σ actual_daily_energy_kwh in month × tariff` |
| `expected_revenue_yearly_lkr` | timeseries | LKR | yearly, ts = 1st-of-year midnight local | `p50_energy × tariff_rate_lkr` (using year-specific p50 from backfilled `pvalue_job`) |
| `actual_revenue_yearly_lkr` | timeseries | LKR | yearly, ts = 1st-of-year midnight local | `Σ actual_daily_energy_kwh in year × tariff` |
| `actual_yearly_energy_kwh` | timeseries | kWh | yearly, ts = 1st-of-year midnight local | `Σ actual_daily_energy_kwh in year` (intermediate for revenue + dashboards) |

No deprecations.

---

## 27. Execution steps — Phase 4

Phase 4 lands after Phase 3 verifies. Step 24 is data-correctness and must land alongside Phase 1.5 / Phase 3 Step 16 if possible.

### Step 24 — Cadence-correct integration in `loss_rollup_job` (N-A) — CRITICAL

File: `app/services/loss_rollup_job.py`

Changes:

1. Add a helper `_time_weighted_kwh_pair(potential, actual, day_start_utc, day_end_utc, setpoint_step_hold_fn, capacity_kw)`:
   - Walk the union of potential.index and actual.index in ASC order.
   - For each adjacent pair `(t_prev, t_curr)`, compute `dt_h = (t_curr - t_prev).total_seconds()/3600`. If `dt_h > 0.5`, skip (treat as outage).
   - Use the interval midpoint to look up the step-held setpoint percentage.
   - Trapezoidal: `pot_avg = 0.5 × (pot[t_prev] + pot[t_curr])`, same for `act`. Both must be non-NaN and ≥ 0.
   - Accumulate `potential_energy_kwh += pot_avg × dt_h`, `exported_energy_kwh += act_avg × dt_h`, `gross_loss_kwh += max(pot_avg − act_avg, 0) × dt_h`.
   - If `sp_pct < 99.5`: `ceiling_kw = capacity_kw × sp_pct/100`; `curtail_base = max(ceiling_kw, act_avg)`; `curtail_loss_kwh += max(pot_avg − curtail_base, 0) × dt_h`.
2. Replace `_integrate` body with a call to the helper. Keep its return-shape identical so the rest of `_process_plant` is unchanged.
3. Add unit test (or inline assertion via `/admin/run-loss-rollup` against a known 5-min plant) that energy values move ~5× higher post-fix.

Verification: pick one 1-min plant (e.g. KSP) and one 5-min plant. After fix, KSP results unchanged; 5-min plant `potential_energy_daily_kwh` ≈ daily_job's `total_generation_expected_kwh` for the same day (both should match within 1 %).

### Step 25 — `pvlib_services` JSON attribute + gating (O-A)

Files: `app/services/thingsboard_client.py`, `app/services/forecast_service.py`, `app/services/daily_job.py`, `app/services/loss_rollup_job.py`, `app/services/pvalue_job.py`, plus new `revenue_job.py` (Step 26).

Changes:

1. In `thingsboard_client.PlantRef`, add `services: Dict[str, bool] = field(default_factory=lambda: _DEFAULT_SERVICES)`. Default = all `true`.
2. In `_visit_node`, after the `pvlib_enabled` truthy check, parse `attrs.get("pvlib_services")` as JSON (with `_DEFAULT_SERVICES` fallback). Attach to `PlantRef`.
3. Each writer module gets a guard at the per-plant top:
   - `forecast_service.process_single_asset`: skip if `services["physics_live"]` false; log `skip:physics_live=false`. (Note: this requires plumbing `PlantRef.services` through `process_single_asset` — pass a `services` dict alongside the asset_id from `run_fleet_cycle`.)
   - `daily_job.run_daily_rollup` per-plant loop: skip if `services["daily_energy"]` false.
   - `loss_rollup_job._process_plant`: skip if `services["loss_attribution"]` false. (Preserve the existing `loss_attribution_enabled` override as a 6-line legacy adapter that maps to `services["loss_attribution"]`.)
   - `pvalue_job._process_plant`: skip if `services["p_values"]` false.
   - `revenue_job` (Step 26): skip if `services["revenue"]` false.
4. Document defaults in `TELEMETRY_CONTRACT.md` and `HOSTING_REFERENCE.md`.

Idempotent: re-reads on every cron via `discover_plants`, which already caches 5 min.

### Step 26 — Revenue telemetry (P-A)

Files (new): `app/services/revenue_job.py`. Modifies: `app/services/scheduler.py`, `app/api/forecast.py`, `app/config.py`.

Logic:

1. Two public entry points:
   - `run_revenue_monthly(tb_client, year, month)` — writes `expected_revenue_monthly_lkr` + `actual_revenue_monthly_lkr` for that calendar month, ts = local midnight of 1st-of-month.
   - `run_revenue_yearly(tb_client, year)` — writes `expected_revenue_yearly_lkr` + `actual_revenue_yearly_lkr` + `actual_yearly_energy_kwh` for that calendar year, ts = local midnight of 1st-of-year.
2. Algorithm (monthly):
   - For each plant (gated on `services["revenue"]`):
     - Read `tariff_rate_lkr`; if missing → write `-1` sentinels, `data_source = "warn:no_tariff"`.
     - Read `forecast_p50_monthly` row for the target month's 1st-midnight ts. Expected = value × tariff.
     - Sum `actual_daily_energy_kwh` for every day in the target month (skip `< 0`). Actual = sum × tariff.
     - Post merged record at the month-ts.
3. Algorithm (yearly):
   - Read `p50_energy` SERVER_SCOPE for the target year (pvalue_job writes the current-year value annually; backfilled per-year via Step 27).
   - Sum `actual_daily_energy_kwh` for every day in the target year.
   - Multiply both by tariff. Write at 1st-of-year ts.
   - For years before `commissioning_date`, actual = `-1`. Expected still computed (PVGIS-ERA5 doesn't depend on commissioning).
4. Crons:
   - `pvlib_revenue_monthly` cron at 1st-of-month 00:15 local — computes the just-finished month.
   - `pvlib_revenue_yearly` cron at 1st-of-year 00:20 local — computes the just-finished year.
5. Admin endpoints:
   - `POST /admin/run-revenue-monthly?year=N&month=M` (default = previous month).
   - `POST /admin/run-revenue-yearly?year=N` (default = previous year).
   - `POST /admin/run-revenue-backfill?asset_id=<id>&years_back=10` (loops the above for each year + each month in that year).
6. New env flag `REVENUE_JOB_ENABLED` (default `false` until Step 28 enables fleet-wide).

### Step 27 — 10-year P-value backfill (data prerequisite for yearly revenue)

This is an operator action, not a code change. Document in `HOSTING_QUICKSTART.md`:

For each plant during initial onboarding, the auto-onboard cron (Step 28) MUST run `pvalue_job` for each of the last 10 years. PVGIS-ERA5 starts at 2005 → fully covered through 2023. Years 2024 and 2025 fall back to the most recent ERA5 year (2023) for percentile calculation — note this limitation in the contract.

Important: pvalue_job's annual `p50_energy` SERVER_SCOPE attribute is **overwritten per run**. To preserve per-year `p50_energy` values for the yearly revenue computation, the revenue job must write a new SERVER_SCOPE per year: `p50_energy_<year>` (e.g. `p50_energy_2018`). Step 26 algorithm depends on this.

Alternative: write per-year p50 as **timeseries** `p50_energy_annual` with ts = 1st-of-year midnight. Cleaner. Use this instead of per-year attrs.

**Revised contract addition (replacing the per-year attr proposal):**

| Key | Type | Unit | Cadence | Description |
|---|---|---|---|---|
| `p50_energy_annual` | timeseries | kWh | yearly, ts = 1st-of-year midnight | Per-year P50 annual yield (the value `p50_energy` was set to that year — preserved historically) |
| `p90_energy_annual` | timeseries | kWh | yearly | Same for P90 |
| `p95_energy_annual` | timeseries | kWh | yearly | Same for P95 |

`pvalue_job` writes these alongside the SERVER_SCOPE attrs.

### Step 28 — Weekly auto-onboard cron (Q-A)

Files: `app/services/scheduler.py`, plus new `app/services/auto_onboard.py`.

Logic:

1. `run_autoonboard_now(tb_client)`:
   - Discover plants via `tb_client.discover_plants(force=True)`.
   - For each plant where `onboarding_completed != true` (attribute absent or false):
     - Parse `commissioning_date` (ISO date attr). If missing → skip with WARN.
     - Run, in sequence (with a 15-min per-plant timeout):
       a. `pvalue_job.run_pvalue_job(plant_ids=[id], target_year=Y)` for each `Y in range(current_year - 9, current_year + 1)`. Each call writes the yearly p_*_annual timeseries.
       b. `daily_job.run_daily_rollup(date=D)` for each `D` from `max(commissioning_date, today-3650)` to `yesterday`. (Use the existing `/admin/run-daily-range` path internally.)
       c. If `services["loss_attribution"]`: `loss_rollup_job.run_loss_rollup(date=D)` for the same range.
       d. If `services["loss_attribution"]`: `loss_rollup_job.recompute_lifetime_for_fleet(asset_id=id)`.
       e. If `services["revenue"]`: `revenue_job.run_revenue_backfill(asset_id=id, years_back=10)`.
     - On success, set SERVER_SCOPE: `onboarding_completed=true`, `onboarding_completed_at=<ISO>`.
     - On any step failure or timeout, leave attrs unset; next Sunday retries.
2. Cron registration in `scheduler.start_scheduler` when `AUTO_ONBOARD_ENABLED=true` (new env flag, default `false`):
   - Trigger: cron, `day_of_week="sun"`, `hour=3`, `minute=0`, `misfire_grace_time=7200`, `max_instances=1`.
3. Admin endpoint: `POST /admin/run-autoonboard` triggers immediately. `POST /admin/run-autoonboard?asset_id=<id>` for single plant.
4. Prometheus counters: `pvlib_autoonboard_attempted_total`, `pvlib_autoonboard_completed_total`, `pvlib_autoonboard_failed_total`, gauge `pvlib_autoonboard_pending`.

### Step 29 — pvalue_job daily + mtd write merge (R-A)

File: `app/services/pvalue_job.py`

Changes:

1. Replace the four separate `post_telemetry` calls in `_process_plant` (lines ~276–280) with two calls:
   - One merged call for `daily + mtd` records (same ts per day, different keys).
   - Keep `weekly` (different ts) and `monthly` (different ts) separate.
2. New helper `_merge_records_by_ts(*record_lists)` in `pvalue_job.py` (or a shared util) — defensive merge that asserts all input records at the same ts have non-overlapping keys.

Verification: cycle-time metric `pvalue_job_duration_seconds` should drop ~25 % (fewer HTTP round-trips).

### Step 30 — Cache fleet attrs (G23 mitigation)

File: `app/services/forecast_service.py`

Changes:

1. Introduce `_plant_attrs_cache: Dict[str, Tuple[dict, float]]` keyed by `asset_id` with `(attrs, cached_at_monotonic)`.
2. New env flag `PLANT_ATTRS_CACHE_TTL_S` (default `300`).
3. `_load_config` first checks the cache; refreshes on TTL expiry or `force=True`.
4. `/admin/refresh-plants` already invalidates `_discover_cache`; extend to also clear `_plant_attrs_cache`.

Effect: 1.44 M attr fetches/day → ~1000 plants × (1440/5) = 288 k fetches/day. 80 % reduction. Live `potential_power` cadence still 1 min.

### Step 31 — Hosting documentation (S-B)

Files (new):
- `M:\Documents\Projects\MAGICBIT\Power-Prediction\scripts\Pvlib-Service\HOSTING_QUICKSTART.md`
- `M:\Documents\Projects\MAGICBIT\Power-Prediction\scripts\Pvlib-Service\HOSTING_REFERENCE.md`

**Quickstart contents (≈ 5 pages):**

1. Prerequisites — EC2 instance specs (t4g.small minimum for ≤ 200 plants, t4g.medium for 1000), AWS account, Docker installed, ThingsBoard tenant credentials.
2. Provision EC2 — security-group ports (8004 internal only or behind ALB), SSM Parameter Store secrets path, IAM role with SSM read access.
3. Initial container start — clone repo, write `.env` from `ec2_userdata.sh` template, `docker compose up -d --build`, verify `curl :8004/health`.
4. Bootstrap fleet — set root assets in TB via the UI snippet provided; set `pvlib_enabled=true` and `pvlib_services` JSON on each plant via `tb_config_loader.py`; trigger `POST /admin/run-autoonboard` once or wait for Sunday.
5. Verify dashboards — open KSP_Plant dashboard, confirm FDI / FvA / Curtailment V5 / Loss Attribution / Expected vs Actual Revenue all render.
6. Smoke-test commands — copy-paste curl recipes for `/health`, `/pvlib/discover`, `/metrics`, `/admin/loss-status`.

**Reference contents:**

1. Telemetry contract by area (link to `TELEMETRY_CONTRACT.md` + a flat summary table).
2. Per-plant SERVER_SCOPE attribute schema (every key the service reads — name, type, default, semantics, example).
3. Widget mapping recipes — one section per widget (FDI, FvA, Curtailment V5, Loss Attribution × 5 modes, Expected vs Actual Revenue, Capacity Factor Compliance, Grid Outage Timeline, Risk Summary Panel). Each recipe lists: datasource order, exact telemetry keys, units, settings to set, common gotchas.
4. Cron schedule + what each job writes.
5. Troubleshooting — "FDI shows --%", "FvA bands empty in January", "Curtailment shows zero loss when setpoint clearly active", "auto-onboard never completed", etc.
6. Performance tuning — `MAX_CONCURRENT_PLANTS`, `PLANT_ATTRS_CACHE_TTL_S`, `LOSS_TODAY_PARTIAL_INTERVAL_MIN`, EC2 instance sizing.

The hosting docs are pure prose; no code-block JSON dump beyond what an operator needs.

### Step 32 — Documentation updates (cross-cutting)

- `TELEMETRY_CONTRACT.md` — append §"Plant SERVER_SCOPE attributes read by service" (consolidating `actual_power_keys`, `active_power_unit`, `timezone`, `pvlib_services`, `commissioning_date`, `onboarding_completed`, `onboarding_completed_at`, `tariff_rate_lkr`, `Capacity`/`capacityUnit`, `setpoint_keys`). Append §"Revenue keys" + §"Per-year P-value keys".
- `ONBOARDING_GUIDE.md` (created in Phase 2) — point readers to `HOSTING_QUICKSTART.md`; keep Phase 2 multi-plant onboarding section.
- `KSP_TEST_RUNBOOK.md` — add a one-line pointer to `HOSTING_QUICKSTART.md` at the top.

---

## 28. File index — Phase 4 (Sonnet 4.6 will touch)

Service code:
- `app/services/loss_rollup_job.py` — Step 24 (cadence-correct integration); Step 25 (services gate).
- `app/services/daily_job.py` — Step 25 (services gate); Step 28 (auto_onboard internal use).
- `app/services/forecast_service.py` — Step 25 (services gate); Step 30 (attrs cache).
- `app/services/pvalue_job.py` — Step 25 (services gate); Step 27 (per-year p_*_annual timeseries); Step 29 (write merge).
- `app/services/scheduler.py` — Step 26 (revenue crons); Step 28 (auto-onboard cron).
- `app/services/thingsboard_client.py` — Step 25 (PlantRef.services).
- `app/services/revenue_job.py` — Step 26 (new module).
- `app/services/auto_onboard.py` — Step 28 (new module).
- `app/api/forecast.py` — Step 26 (3 admin endpoints); Step 28 (1 admin endpoint).
- `app/config.py` — `REVENUE_JOB_ENABLED`, `AUTO_ONBOARD_ENABLED`, `PLANT_ATTRS_CACHE_TTL_S`, `AUTOONBOARD_PER_PLANT_TIMEOUT_S`.

Service docs:
- `TELEMETRY_CONTRACT.md` — Step 32.
- `HOSTING_QUICKSTART.md` (new) — Step 31.
- `HOSTING_REFERENCE.md` (new) — Step 31.
- `ONBOARDING_GUIDE.md` (Phase 2) — pointer line.
- `KSP_TEST_RUNBOOK.md` — pointer line.

Shared tooling:
- `scripts/shared/plants_master.schema.json` — add `pvlib_services`, `commissioning_date`.
- `scripts/shared/tb_config_loader.py` — write `pvlib_services` + `commissioning_date` from master.
- `scripts/shared/audit_tb_config.py` — validate `pvlib_services` JSON shape; ERR when `commissioning_date` missing for plants with `services["revenue"]=true` or `services["loss_attribution"]=true`.

Widgets:
- `Forecasts & Risk\Expected vs Actual Revenue\settings.json` — add a `viewMode` setting (`monthly` | `yearly`) and four key settings (`expectedMonthlyKey`, `actualMonthlyKey`, `expectedYearlyKey`, `actualYearlyKey`). README updated.
- `Forecasts & Risk\Expected vs Actual Revenue\.js` — branch on `viewMode`; fetch the matching pair of TB keys.
- `Forecasts & Risk\Expected vs Actual Revenue\README.md` — wire to new service keys; document yearly mode.

---

## 29. Out of scope (Phase 4)

- Per-cron staggering (Phase 3 §22 still deferred).
- Multi-tenant isolation.
- Webhook-driven auto-onboard (Q-C alternative). Sunday cron is sufficient.
- Translating widget legacy fallback baselines (`2.475`) — left as-is; once real revenue keys flow, the fallback path is unreachable.
- PVGIS replacement for years > 2023. Accept the documented 2024+ fallback to 2023 ERA5.

---

## 30. Open questions (Phase 4, non-blocking)

1. **Yearly revenue for 2026 (current year)** — should the widget show a partial-year actual (prorated) like it does for the current-month bar in monthly mode? Recommendation: yes; widget computes proration client-side using the latest `actual_yearly_energy_kwh` timeseries row vs day-of-year.
2. **Revenue when `tariff_rate_lkr` changes mid-year** — should historical rows be re-stamped at the new tariff or frozen? Recommendation: frozen at the tariff in effect when each row was first written; never recompute history.
3. **`pvlib_services["physics_live"] = false`** — should the live cycle skip the plant entirely (saves the per-minute discovery + station fetch) or still write sentinels so the absence is observable? Recommendation: skip entirely; ops can observe via `/metrics` `pvlib_skipped_services_total`.
4. **PVGIS rate-limit headroom for 10-year × 1000-plant backfill** — at year-1 onboarding, that's 10 × 70 cells = 700 PVGIS fetches over ~10–20 min. Acceptable. Document in `HOSTING_QUICKSTART.md`.
5. **`commissioning_date` discovery** — if a plant doesn't have the attribute, should auto-onboard default to `today − 365d` for the daily/loss backfill, or skip with WARN? Recommendation: skip with WARN; force operator to set the attr.

