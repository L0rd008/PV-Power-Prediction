# Hand-off Prompt for Sonnet 4.6 — round 2

> Paste everything below into a fresh Sonnet 4.6 conversation. The only context you need is this brief, the round-1 plan/prompt, and the source files you've already read.

---

You are Sonnet 4.6, continuing work on the MAGICBIT power-prediction monorepo. You completed round 1 of the Loss Attribution work — you implemented `app/services/loss_rollup_job.py` and the supporting changes per `LOSS_ATTRIBUTION_TELEMETRY_PLAN.md` / `LOSS_ATTRIBUTION_SONNET_PROMPT.md` (both still in `scripts/Pvlib-Service/`). Naveen has tested it and reports two real-world problems:

1. **The Grid & Losses tab on ThingsBoard crashes when it loads.** Both the V5 Curtailment widget and the Loss Attribution widget are on this tab; the crash points at the Loss Attribution widget.
2. **The Loss Attribution card flickers every minute** — the value disappears, the card briefly shows `--` / `CALC`, then the value reappears. This repeats indefinitely.
3. The user also asked: if the new precomputed keys only update once per day, why does the widget try so hard to refresh? They want both the service and the widget to update every 30 min instead of per-minute.

The round-2 plan, **`scripts/Pvlib-Service/LOSS_ATTRIBUTION_FIX_PLAN.md`**, audits your round-1 implementation, identifies the bugs, scores fix approaches, and lays out the chosen hybrid bundle. Read it in full before writing any code. It cites your file:line references; trust them but verify.

## Why this matters

Round 1 moved aggregation server-side, but the widget still calls its render function on every TB telemetry tick (`onDataUpdated` fires every minute because DS[0] subscribes to `potential_power`, which the service writes every minute). On every tick the widget calls `setLoadingState` — wiping the displayed value and showing a skeleton — then re-fetches primary + comparator ranges. With 4–6 cards on the dashboard and `useNewLossKeys = "auto"`, when the precomputed keys aren't yet present, the widget falls back to the legacy per-bucket fetch on a Year or Lifetime range, generating hundreds of TB requests per render per card. That's the crash.

You also went off-spec by adding `KEY_POTENTIAL_MONTHLY` / `KEY_POTENTIAL_YEARLY` and a `_get_historical_sum` helper that re-fetches running sums every day. The round-1 Plan §7 explicitly forbade monthly/yearly precomputed keys; revert that.

## Hard constraints (carry over from round 1)

1. **Zero plant-specific constants in the service.** No plant IDs, tariffs, capacities, or names hardcoded in `app/`. Everything plant-derived comes from each plant's TB SERVER_SCOPE attributes.
2. **Backwards compatibility on the widget.** When precomputed keys are missing, the widget must degrade gracefully. For year/lifetime/custom > 60 d a "PENDING — not yet rolled up" placeholder is acceptable; for day/month the legacy per-bucket path stays as a fallback.
3. **Append-only telemetry contract.** Don't remove or rename existing keys. The monthly/yearly keys you added in round 1 may be retired since they were never consumed by any widget — but document the retirement with a one-line entry in `TELEMETRY_CONTRACT.md`.
4. **No new infrastructure.** Re-use the existing APScheduler. No new database, no Redis, no client-side worker.
5. **Both new crons default OFF.** `LOSS_ROLLUP_ENABLED` stays `false` (round-1 default). The new `LOSS_TODAY_PARTIAL_ENABLED` also defaults `false`. The operator turns them on in `.env` after manual verification.
6. **Widget guardrails.** The widget MUST NOT call `setLoadingState` on a background refresh. It MUST NOT fetch attributes more than once per `attrCacheMinutes`. It MUST suspend its refresh timer when the tab is hidden.

## Read first (in this order)

1. `scripts/Pvlib-Service/LOSS_ATTRIBUTION_FIX_PLAN.md` — the round-2 plan. End-to-end. Sections 4 and 5 contain the concrete code-level instructions.
2. `scripts/Pvlib-Service/LOSS_ATTRIBUTION_TELEMETRY_PLAN.md` — the round-1 plan, for the design decisions you already implemented.
3. `scripts/Pvlib-Service/app/services/loss_rollup_job.py` — your round-1 module. The §4 work edits this.
4. `scripts/Pvlib-Service/app/services/scheduler.py` — your round-1 cron registration; you'll add a second job.
5. `scripts/Pvlib-Service/app/config.py` — your round-1 settings; you'll append four new settings.
6. `scripts/Pvlib-Service/app/api/forecast.py` — your round-1 admin endpoints; you'll add `/admin/run-today-partial`.
7. `scripts/Pvlib-Service/TELEMETRY_CONTRACT.md` — append `loss_tariff_rate_lkr_at_compute` row, retire monthly/yearly rows.
8. `scripts/Pvlib-Service/tests/test_loss_rollup.py` — extend with today-partial tests.
9. `Widgets/Grid & Losses/Loss Attribution/.js` — biggest change set. Sections 5.1–5.10 of the Plan walk through every edit.
10. `Widgets/Grid & Losses/Loss Attribution/settings.json` — append two settings (Plan §5.11).
11. `Widgets/Grid & Losses/Loss Attribution/README.md` — add a "Refresh policy" subsection.

## What to build (Plan sections 4 and 5 are the spec — this is the index)

### Server side

1. **Revert the round-1 deviation.** In `loss_rollup_job.py`:
   - Remove `KEY_POTENTIAL_MONTHLY`, `KEY_POTENTIAL_YEARLY`, `_get_historical_sum`.
   - Remove the second `_safe_write_daily(...)` call inside `run_loss_rollup` (the outer-loop one that was added to write the monthly/yearly fields). Single write inside `_process_plant` is enough.
   - Remove the matching ancestor monthly/yearly logic.
   - Drop the keys from `_safe_write_daily`'s `values` dict and from `_sentinel_daily_values()`.
2. **Refactor `_process_plant`** to accept an optional `min_samples_override`. Default = `settings.LOSS_MIN_VALID_SAMPLES`. The today-partial cron will pass the lower threshold from `settings.LOSS_TODAY_PARTIAL_MIN_SAMPLES`.
3. **Add `run_today_partial_rollup(tb_client)`** in the same module per Plan §4.2. Math identical to `run_loss_rollup` but day window = `[today_local_midnight, now_local]`, no lifetime updates, ancestor data-source `"rollup:partial"` and per-plant data-source `"ok:partial"` (Plan §9 Q4).
4. **Add settings** in `app/config.py`: `LOSS_TODAY_PARTIAL_ENABLED`, `LOSS_TODAY_PARTIAL_INTERVAL_MIN`, `LOSS_TODAY_PARTIAL_DAY_START_HOUR`, `LOSS_TODAY_PARTIAL_DAY_END_HOUR`, `LOSS_TODAY_PARTIAL_MIN_SAMPLES`. Defaults per Plan §4.2.
5. **Add `run_today_partial_now()`** in `scheduler.py` (mirrors `run_loss_rollup_now`); registers an `interval` job inside `start_scheduler` gated on both `LOSS_ROLLUP_ENABLED` and `LOSS_TODAY_PARTIAL_ENABLED`.
6. **Add `/admin/run-today-partial`** in `app/api/forecast.py`.
7. **Update `TELEMETRY_CONTRACT.md`**: document `loss_tariff_rate_lkr_at_compute`; add a paragraph about the today-partial cron and the `"ok:partial"` / `"rollup:partial"` data-source values; retire the monthly/yearly rows.
8. **Extend `tests/test_loss_rollup.py`**: drop monthly/yearly tests; add tests for `run_today_partial_rollup` per Plan §4.5.

### Widget side (`Widgets/Grid & Losses/Loss Attribution/.js`)

Plan §5 has the full diff plan. The crucial changes:

9. **Move `window.addEventListener('loss-range-changed', ...)` from `updateDom` into `onInit`.** Single listener for the lifetime of the widget. Range-change handler triggers a non-silent re-render and ensures the refresh timer is alive.
10. **Replace `self.onDataUpdated`'s computed-mode body with `ensureRefreshTimer()`.** No render on TB tick. Insurance / non-computed modes still call `renderLatestValueFallback()`.
11. **Add `ensureRefreshTimer`, `clearRefreshTimer`, `rangeIncludesToday`, and the visibility handler** at module scope per Plan §5.3.
12. **Modify `renderComputedMode` to accept `{silent}`.** Skeleton only on first-render or entity/range change. Track `_hasRendered`, `_activeEntityKey`, `_activeRange`.
13. **Cache `fetchCalculationAttributes`** with a TTL of `_ATTR_TTL_MS` (default 30 min, configurable via the new `attrCacheMinutes` setting).
14. **Replace `isCurrentDay` with `rangeIncludesToday`.** Update the `'auto'` branch in `calcForRange` per Plan §5.7.
15. **Loosen the precomputed `ok` flag** to `hasPotential` only; treat `< 0` in non-potential keys as `0` (Plan §5.8).
16. **Add `legacyFallbackAllowed(range)`** — only `day`, `month`, `custom <= 60 d`. Year and lifetime go to the "PENDING" placeholder if precomputed is empty (Plan §5.7, §5.9).
17. **Add `showNotRolledUpPlaceholder`** for the "PENDING" state (Plan §5.9).
18. **Update `onDestroy`** to clear the refresh timer and the visibility listener (Plan §5.10).
19. **Append two settings to `settings.json`**: `pollIntervalMinutes` (default 30, min 5, max 240) and `attrCacheMinutes` (default 30) (Plan §5.11). Wire them into `_POLL_INTERVAL_MS` and `_ATTR_TTL_MS` at the top of `onInit`.
20. **Update README.md** with a "Refresh policy" subsection per Plan §5.12.

## Verification (must pass before reporting done)

Plan §8 is the full checklist. Minimum acceptance:

- `pytest scripts/Pvlib-Service/tests/test_loss_rollup.py -v` is green; the previously-passing tests still pass.
- `POST /admin/run-today-partial` (with the cron flag enabled in a local dev `.env`) writes the six daily keys at today's local-midnight ts with `loss_data_source = "ok:partial"` (after >30 min of daylight) or sentinel `error:insufficient_samples` (early morning); lifetime attributes unchanged.
- `POST /admin/run-loss-rollup?start=2026-05-03&end=2026-05-03` writes daily keys exactly once per plant per day (no duplicate write); no `potential_energy_monthly_kwh` / `potential_energy_yearly_kwh` keys appear.
- Loss Attribution dashboard with KSP and `useNewLossKeys = "auto"`: cards render once on load with skeleton; no skeleton storm; total TB requests/min from this dashboard ≤ 6 + a 30-min silent burst.
- Hide the tab for 30 min, return — cards do one immediate silent refetch.
- Switch to a non-`pvlib_enabled` plant on Year mode — card shows `--` with `PENDING` and the tooltip "This plant has not been rolled up yet for the selected range."
- V5 Curtailment widget on the same tab is unaffected.

## What you must NOT do

- Do not remove or rename existing `loss_*_daily_*` or `loss_*_lifetime_*` keys (the round-1 contract). Only `potential_energy_monthly_kwh` and `potential_energy_yearly_kwh` may be retired (they were the round-1 deviation).
- Do not change the round-1 daily-job schedule (00:10 local). Only ADD the today-partial interval cron.
- Do not enable any new flag by default. `LOSS_ROLLUP_ENABLED` and `LOSS_TODAY_PARTIAL_ENABLED` both default `false`.
- Do not call `setLoadingState` on a background refresh. The whole point of round 2 is to stop the flicker.
- Do not skip the visibility-change handler. Without it, hidden tabs keep polling and waste both client and TB resources.
- Do not edit `forecast_service.py`, `daily_job.py`, or the V5 Curtailment widget.
- Do not over-fix: leave the legacy `calculateLossForRange` math alone (it's still the day/month fallback path).

## When you finish

1. List every file you changed or created (absolute paths).
2. Print `pytest` output proving the suite is green.
3. Re-list which Plan §8 verification steps you executed locally and which require a TB connection (Naveen will run those).
4. Note any open questions from Plan §9 you couldn't resolve from the code alone.

You are evaluated by industry experts before this goes to production. Be rigorous: read both plans, follow the file/line index in Plan §10, mirror the patterns of `daily_job.py` and your own `run_loss_rollup`, and run the tests before reporting done.
