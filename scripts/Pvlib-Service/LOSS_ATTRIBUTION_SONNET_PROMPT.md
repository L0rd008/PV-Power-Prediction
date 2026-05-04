# Hand-off Prompt for Sonnet 4.6

> Paste everything below into a fresh Sonnet 4.6 conversation. Sonnet has no memory of the planning session — every piece of context it needs is in this brief or in the linked files.

---

You are Sonnet 4.6 working on the MAGICBIT power-prediction monorepo at `M:\Documents\Projects\MAGICBIT\`. You have read/write access to:

- `M:\Documents\Projects\MAGICBIT\Power-Prediction\scripts\Pvlib-Service\` — the production EC2 service that writes solar physics telemetry to ThingsBoard.
- `M:\Documents\Projects\MAGICBIT\Widgets\Grid & Losses\Loss Attribution\` — the ThingsBoard widget being optimised.

Your job is to implement the work specified in **`M:\Documents\Projects\MAGICBIT\Power-Prediction\scripts\Pvlib-Service\LOSS_ATTRIBUTION_TELEMETRY_PLAN.md`** (the "Plan"). Read the Plan in full before writing any code. It was written by an Opus planning agent that audited both codebases; it lists every problem, every approach evaluated, the rated decisions, the chosen hybrids, and a file-by-file change list.

## Why this work matters

The Loss Attribution widget currently downloads minute-cadence `active_power` and `potential_power` for the selected range (day/month/year/lifetime/custom) plus a comparator range, then integrates client-side. For year and lifetime ranges this hangs the dashboard. The fix is to move the integration into Pvlib-Service (which already has plant discovery, an attribute reader, a TB client with retry, and a daily cron) and have it write daily aggregated loss telemetry plus a cumulative-lifetime attribute. The widget is changed to read those instead, with graceful fallback to the existing client-side path so plants that haven't been rolled up yet still work.

## Hard constraints (do not violate)

1. **Zero plant-specific constants in the service.** `app/` must not contain a hardcoded plant ID, plant name, plant capacity, or plant tariff. The fleet is 1 000+ plants; everything plant-derived comes from each plant's TB SERVER_SCOPE attributes via the existing `discover_plants` + `get_asset_attributes` pipeline. Slight hardcoding inside the widget (default key names, default mode mapping) is allowed and is documented in the Plan.
2. **Backwards compatibility.** When the new daily keys haven't been written for a plant, the widget must fall back to the existing path (`renderComputedMode` → `calculateLossForRange` → `renderLatestValueFallback`). Operators must be able to disable the new path entirely with `useNewLossKeys = "off"`.
3. **No new infrastructure.** All state lives in ThingsBoard. No new database, no Redis, no S3, no external scheduler. Re-use the existing APScheduler instance in `app/services/scheduler.py`.
4. **No edits to `daily_job.py`, `weekly_eval.py`, or `forecast_service.py`** beyond importing reused constants. Add a new module — don't refactor working code.
5. **Sentinel = `-1` everywhere.** Mirror the existing pattern: when a daily integration is invalid (insufficient samples, missing tariff, etc.) write `-1` for every numeric key for that day with a descriptive `loss_data_source` string.
6. **Telemetry contract is append-only.** Do not rename or remove any key listed in `TELEMETRY_CONTRACT.md`; only add the new keys defined in the Plan §6.

## What to read first (in this order)

1. `M:\Documents\Projects\MAGICBIT\Power-Prediction\scripts\Pvlib-Service\LOSS_ATTRIBUTION_TELEMETRY_PLAN.md` — the Plan. Read end-to-end.
2. `M:\Documents\Projects\MAGICBIT\Power-Prediction\scripts\Pvlib-Service\TELEMETRY_CONTRACT.md` — current TB key contract. The Plan §6 lists the additions you'll append.
3. `M:\Documents\Projects\MAGICBIT\Power-Prediction\scripts\Pvlib-Service\app\services\daily_job.py` — your new `loss_rollup_job.py` mirrors this module's shape.
4. `M:\Documents\Projects\MAGICBIT\Power-Prediction\scripts\Pvlib-Service\app\services\forecast_service.py` — for the ancestor roll-up pattern (`_rollup_parents`) and the telemetry key constants at the top of the file.
5. `M:\Documents\Projects\MAGICBIT\Power-Prediction\scripts\Pvlib-Service\app\services\thingsboard_client.py` — `get_timeseries`, `post_telemetry`, `get_asset_attributes`, `discover_plants`. Note: writing SERVER_SCOPE attributes is **not** in this client today; you'll need to add a `post_attributes(entity_type, entity_id, scope, payload)` method. ThingsBoard's REST API for setting attributes is `POST /api/plugins/telemetry/{entityType}/{entityId}/attributes/{scope}` with body `{"key1": value1, "key2": value2}` (note: `attributes/{scope}`, not the `values/attributes/{scope}` read path used by `get_asset_attributes`). Verify against your local TB v4.3.0 dev instance before deploying. Wrap the new method with the same `tenacity` retry decorator stack used by `_post`.
6. `M:\Documents\Projects\MAGICBIT\Power-Prediction\scripts\Pvlib-Service\app\services\scheduler.py` — register the new cron alongside `pvlib_daily`.
7. `M:\Documents\Projects\MAGICBIT\Power-Prediction\scripts\Pvlib-Service\app\api\forecast.py` — model `/admin/run-loss-rollup`, `/admin/recompute-lifetime`, `/admin/loss-status` after `/admin/run-daily` and `/admin/run-weekly`.
8. `M:\Documents\Projects\MAGICBIT\Power-Prediction\scripts\Pvlib-Service\app\config.py` — add the four new settings listed in the Plan §4.5.
9. `M:\Documents\Projects\MAGICBIT\Power-Prediction\scripts\Pvlib-Service\app\services\weekly_eval.py` — reference pattern for "fetch series, compute, write, with admin endpoint".
10. `M:\Documents\Projects\MAGICBIT\Widgets\Grid & Losses\Loss Attribution\.js` — find `renderComputedMode` and `calculateLossForRange`. Add the new branch as described in Plan §5.2.
11. `M:\Documents\Projects\MAGICBIT\Widgets\Grid & Losses\Loss Attribution\settings.json` — append the new settings listed in Plan §5.1.
12. `M:\Documents\Projects\MAGICBIT\Widgets\Grid & Losses\Loss Attribution\README.md` — add the `curtailRevenue` row + a "Server-side aggregation" section.

## What to build

Follow the Plan. Concretely:

### Server side (Pvlib-Service)

1. **`app/config.py`** — Add `LOSS_ROLLUP_ENABLED`, `LOSS_DEFAULT_SETPOINT_KEYS`, `LOSS_MIN_VALID_SAMPLES`, `LOSS_LIFETIME_PAGE_DAYS` (Plan §4.5). Mirror the existing `field_validator` style.
2. **`app/services/thingsboard_client.py`** — Add `post_attributes(entity_type, entity_id, scope, payload: dict)` using the retry decorator pattern.
3. **`app/services/loss_rollup_job.py`** — New module. Public entry `async def run_loss_rollup(tb_client, date: Optional[datetime] = None) -> dict`. Implement Plan §4.1 step-by-step. Reuse `discover_plants`, `get_asset_attributes`, `get_timeseries`, `post_telemetry`. Concurrency-bounded by `settings.MAX_CONCURRENT_PLANTS`. Sentinel-aware. W → kW scaling on `active_power` driven by the per-plant `active_power_unit` attribute. Returns a summary dict.
4. **Lifetime maintenance** in the same module (Plan §4.2). Two functions: `_lifetime_increment_step` and `_recompute_lifetime_from_history`. Decision logic based on `loss_lifetime_anchor_date`. Write the eight server-scope attributes via the new `post_attributes`.
5. **Ancestor roll-up** for both daily keys and lifetime attributes (Plan §4.1 step 4 and §4.2 last paragraph). Use the same dedup-aware ancestor map shape that `forecast_service._rollup_parents` produces.
6. **`app/services/scheduler.py`** — Add `run_loss_rollup_now(date: Optional[datetime] = None)` mirroring `run_daily_now`. Register a cron at hour=0, minute=10, `misfire_grace_time=3600`, `id="pvlib_loss_rollup"`. Gate on `settings.LOSS_ROLLUP_ENABLED`.
7. **`app/api/forecast.py`** — Add three endpoints (Plan §4.4): `POST /admin/run-loss-rollup` (date-range backfill), `POST /admin/recompute-lifetime` (per-plant or fleet-wide), `GET /admin/loss-status?asset_id=...`. Validate inputs with FastAPI/Pydantic patterns already in use.
8. **`TELEMETRY_CONTRACT.md`** — Append Plan §6 verbatim (the daily-keys table and the lifetime-attributes table). Do not edit any existing rows.
9. **`tests/test_loss_rollup.py`** — New pytest. Unit-test the integration math on synthetic 1-day series, the sentinel paths (no potential / insufficient samples / missing tariff), the W → kW scaling, the curtailment formula at boundary `setpoint_pct == 99.5`, and the lifetime increment vs. recompute decision logic. Use `unittest.mock.AsyncMock` for the TB client (the existing `tests/test_phase_a.py` shows the style).

### Widget side (Loss Attribution)

10. **`settings.json`** — Append the eight new settings entries (Plan §5.1) after `lifetimeStartDate`.
11. **`.js`** — Add `calculateLossForRangePrecomputed(entity, range, attrs)` returning a result object with the same shape as `calculateLossForRange`. Modify `renderComputedMode` to dispatch based on `s.useNewLossKeys` and the today-current-day shortcut (Plan §5.2). Modify `renderComputedResult` only if needed to prefer a directly-read LKR value when present. Reuse `fetchTimeseriesChunked` for daily-key reads (with `useAgg=false`); reuse `fetchAttributesWithFallback` for the lifetime attributes.
12. **`README.md`** — Add the missing `curtailRevenue` row to the Modes table (currently it's in `settings.json` and `.js` but not the README). Add a new "Server-side aggregation" section explaining the new fast path, the auto/force/off setting, and the fallback chain.

## Verification (must run before reporting done)

Plan §6 has the full checklist. Minimum acceptance:

- `pytest scripts/Pvlib-Service/tests/test_loss_rollup.py -v` is green.
- A manual call to `POST /admin/run-loss-rollup` against a local Docker run with KSP's asset ID returns finite values and the keys appear in TB at the local-midnight ts.
- A manual call to `POST /admin/recompute-lifetime?asset_id=<KSP>` agrees with `Σ(daily history)` to within 0.1 %.
- The widget, with `useNewLossKeys = "auto"` and the new keys present on KSP, renders day/month/year/lifetime in under 2 s with no `--` flicker.
- With `useNewLossKeys = "off"`, the widget still works (legacy path).
- With the new keys absent (test against a non-KSP plant), `useNewLossKeys = "auto"` falls back silently and the legacy path renders.

## What you must NOT do

- Do not run the cron against production TB during development. Use the Docker compose setup or the `/admin/run-loss-rollup` endpoint in a controlled environment. The Plan's Phase L1/L2/L3/L4 rollout is operator-driven, not yours.
- Do not flip `LOSS_ROLLUP_ENABLED` to `true` by default. Default = `false`. The operator turns it on per `.env` after manual verification.
- Do not introduce monthly/yearly precomputed keys, an hour-of-day series, a today-partial running tally, or any new external service. They were considered, scored, and ruled out in Plan §2.
- Do not edit `daily_job.py`, `weekly_eval.py`, `forecast_service.py`, the V5 Curtailment widget, or the Forecast vs Actual Energy widget. Those were considered and explicitly ruled out in Plan §7.
- Do not rename or remove any key in the existing `TELEMETRY_CONTRACT.md` — append-only.

## When you finish

1. List every file you changed or created (absolute path).
2. Print `pytest` output proving tests pass.
3. Reproduce the verification checklist results from Plan §6 (the pieces you can do without an EC2 deploy).
4. Note any open questions from Plan §8 you couldn't resolve from the code alone, so Naveen can answer them before Phase L1.

You are evaluated by industry experts before this goes to production. Be rigorous: read the Plan completely, follow the file/line index in Plan §9, mirror the patterns of `daily_job.py` and `weekly_eval.py` rather than inventing new structure, and run the tests before reporting done.
