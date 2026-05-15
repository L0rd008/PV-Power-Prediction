# Sonnet 4.7 Execution Brief — Widget Fix + Multi-Plant Onboarding

> Paste everything below the line into a fresh Sonnet 4.7 conversation. The prompt is self-contained: no prior context required.

---

## ROLE

You are Claude Sonnet 4.7 acting as the executing engineer for a Solar PV monitoring stack. You are reporting to a senior engineer (Pasindu). Your output will be reviewed by an executive panel — quality, correctness, and discipline matter. Do not paraphrase or shortcut the plan. Do not invent telemetry keys, file paths, or behavior. If a step is ambiguous, stop and ask in a single grouped question.

## STACK SUMMARY (read before doing anything)

**Repo:** `M:\Documents\Projects\MAGICBIT\Power-Prediction` (GitHub: `L0rd008/Power-Prediction`).
**Service:** `Pvlib-Service` — FastAPI app in `scripts/Pvlib-Service/`, Docker on EC2 (t4g.small), Python 3.12. Writes solar physics + P-value + loss-attribution telemetry to ThingsBoard (TB) for ~1000 PV plants in Sri Lanka.

**TB host (prod):** `https://windforce.thingsnode.cc`.
**Reference plant:** `KSP_Plant`, asset UUID `0e4b4070-50ff-11ef-b4ce-d5aee9e495ad`.

**Service architecture (H-A3 / H-B6 / H-C4):**
- Discovers plants via BFS from `TB_ROOT_ASSET_IDS` (env), filtering `isPlant=true && pvlib_enabled=true`.
- Crons: minute cycle (`run_cycle_now`), `00:05` daily energy (`daily_job`), `00:10` loss rollup (`loss_rollup_job`, gated), `01:00` new-plant P-value detector, `02:00 Sun` weekly eval, `03:00 Jan-1` annual P-value job.
- Telemetry contract: `scripts/Pvlib-Service/TELEMETRY_CONTRACT.md` (authoritative).
- Live plant discovery cached 5 min; `/admin/refresh-plants` invalidates.
- All admin endpoints documented in `app/api/forecast.py`.

**Widgets repo:** `M:\Documents\Projects\MAGICBIT\Widgets\` — two folders matter for this work:
- `Forecasts & Risk\Forecast Deviation Card (FDI)\`
- `Forecasts & Risk\Forecast vs Actual Energy\V1 TB Latest Values Widget\`
- (Reference only — do not edit:) `Grid & Losses\Curtailment vs Potential Power\V5 TB Timeseries Widget\` and `Grid & Losses\Loss Attribution\`.

Each widget folder contains `.html`, `.css`, `.js`, `settings.json`, `README.md`. The TB widget editor ingests these directly.

## YOUR MISSION (three phases + one critical pre-fix)

### Phase 1.5 — CRITICAL data-correctness fix (lands FIRST, before Phase 1)

**Plan §20 Step 16.** `daily_job.py` integrates `active_power` without W→kW scaling. ~21 plants in the fleet publish in W; their `actual_daily_energy_kwh` is currently 1000× too high. Widgets read this key; FDI/FvA show wrong numbers regardless of widget code. Fix this before widget work — otherwise verification of Phase 1 against W-publishing plants is meaningless.

This is a ~30 LOC change in `daily_job.py` that mirrors the existing pattern in `loss_rollup_job.py:374–375`. After it lands, re-run `/admin/run-daily-range` for the past 12 months for every W-publishing plant (per plan §23 Q1).

### Phase 1 — Widget fixes (lands after 1.5)

Three concrete defects in two widgets. No service code touched.

1. **FDI v4.0 rewrite** — bring `.js` from v2.2 to the v4.0 design described in its README. Use daily-key client-side summation; integrate today's partial via `active_power` agg=SUM; derived fallback as ultimate degradation.
2. **FvA empty-state gate** — when `forecast_p*_weekly` (or `_monthly`, `_daily`) returns zero rows for the current year, fall through to the existing attribute-derived path instead of rendering empty bands.
3. **README ↔ settings ↔ code coherence** for FDI, including legacy-alias retention for installed dashboards.

### Phase 2 — Multi-plant onboarding plumbing (lands after Phase 1 verifies)

Six concrete changes (Steps 8–14 in the plan) to absorb ~1000 plants with heterogeneous configs. **Note:** Step 8 of the plan referred to `actual_power_key` (singular); the Phase 3 addendum revises this to `actual_power_keys` (plural CSV) to align with the existing `loss_rollup_job.py` convention. Use the plural form.

### Phase 3 — Service hardening (lands after Phase 2)

Seven steps (16–22 in the plan) covering: the Phase-1.5 fix already done, multi-key support in daily_job, per-plant timezone, orphan-plant detection script, config-drift script, deprecation of `set_active_power_unit.py`. Step 16 is Phase 1.5 — do not duplicate it here.

### Phase 4 — Correctness + Revenue + Atomic Switches + Zero-Touch (lands after Phase 3)

Nine steps (24–32 in the plan). Notable order: **Step 24 (cadence-correct loss integration) is data-correctness and must land alongside Phase 1.5 / Phase 3 Step 16** — without it, every non-1-min plant has wrong curtailment loss numbers.

Highlights:
- New SERVER_SCOPE JSON `pvlib_services` (5 atomic groups). Default `{all true}`. Each cron checks its group.
- New `revenue_job.py` writing 4 LKR + 1 kWh yearly key (`expected_revenue_monthly_lkr`, `actual_revenue_monthly_lkr`, `expected_revenue_yearly_lkr`, `actual_revenue_yearly_lkr`, `actual_yearly_energy_kwh`).
- New per-year p-value timeseries (`p50_energy_annual`, etc) so the 10-year yearly revenue mode can reference historical P50.
- New weekly Sunday-03:00 `auto_onboard.py` running full historical backfill since `commissioning_date`, marker attr `onboarding_completed=true`. `AUTO_ONBOARD_ENABLED` env flag (default false).
- `pvalue_job` daily+mtd write merge (single TB record per day instead of two).
- `forecast_service` fleet-attr cache (TTL 300 s default).
- `HOSTING_QUICKSTART.md` (5 pages) + `HOSTING_REFERENCE.md` for an operator with Linux/EC2/Docker baseline.
- Expected vs Actual Revenue widget rewired to consume the new service keys + add `viewMode = "monthly" | "yearly"`.

## AUTHORITATIVE PLAN

Read this file in full **before any edit**:

```
M:\Documents\Projects\MAGICBIT\Power-Prediction\scripts\Pvlib-Service\WIDGET_PVALUE_FIX_PLAN.md
```

It contains the scored approach analysis, the chosen approach for each requirement, every step you must execute (Steps 1–15), the file index, and the verification protocol. **Do not re-derive choices** — the analysis is finalised. If you disagree with a choice, stop and raise it; do not silently substitute.

Also read these references (do not edit unless the plan says so):

1. `scripts/Pvlib-Service/TELEMETRY_CONTRACT.md` — source of truth for every TB key the service writes. Read this before considering any telemetry-related change. **Append-only.**
2. `scripts/Pvlib-Service/LOSS_ATTRIBUTION_TELEMETRY_PLAN.md` — context on the loss-rollup design (Phase L0).
3. `scripts/Pvlib-Service/KSP_TEST_RUNBOOK.md` — manual single-plant smoke-test procedure (local Docker).
4. `Widgets/Forecasts & Risk/Forecast Deviation Card (FDI)/README.md` — the v4.0 spec the `.js` must catch up to.
5. `Widgets/Forecasts & Risk/Forecast vs Actual Energy/V1 TB Latest Values Widget/README.md` — describes `ytd_weekly` (the chosen default), `monthly`, `mtd_daily`, `daily` modes.

## EXECUTION ORDER (do in sequence, gate on tests)

1. **Read** the plan end-to-end (Addendums A and B inclusive) + the five reference files above.
2. **Execute Phase 1.5 first (plan §20 Step 16).** Modify `app/services/daily_job.py` to read `active_power_unit` + `actual_power_keys` SERVER_SCOPE attrs and apply W→kW scaling. Add the legacy `actual_power_key` (singular) fallback per §20 Step 17. Test against one kW plant and one W plant; the W plant's new daily kWh should be 1/1000 of any pre-fix historical value (or in the plausible 0–capacity×8×0.85 range if no history exists).
3. **Backfill** historical `actual_daily_energy_kwh` for W-publishing plants per plan §23 Q1 — `POST /admin/run-daily-range?start=<commissioning_date|today-365d>&end=<yesterday>` for each. Verify with `/admin/loss-status` or direct TB latest-telemetry inspection.
4. **Execute Phase 1 (plan Steps 1–7).** Stage all four widget file changes locally before testing:
   - `Forecast Deviation Card (FDI)/.js`
   - `Forecast Deviation Card (FDI)/settings.json`
   - `Forecast Deviation Card (FDI)/README.md`
   - `Forecast vs Actual Energy/V1 TB Latest Values Widget/.js`
5. **Verify Phase 1** following plan §6 Step 6 (validation against KSP and at least one W-publishing plant). If FDI numerical correctness vs hand-calc cannot be reproduced within 0.5 %, stop and report.
6. **Execute Phase 2 (plan Steps 8–14).** This touches service code (`daily_job.py`, `loss_rollup_job.py`, `scheduler.py`, `config.py`), `TELEMETRY_CONTRACT.md`, and creates new files under `scripts/shared/`. Note: the `actual_power_key` (singular) mentioned in plan Step 8 is **superseded** by the plural `actual_power_keys` from Step 16/17 — implement the plural form; keep singular as a 6-line legacy adapter with a one-time WARN log per plant.
7. **Verify Phase 2** following plan §13 Step 15 against two real plants picked by Pasindu.
8. **Execute Phase 3 (plan §20 Steps 18–22).** Per-plant timezone, orphan plant script, config-drift script, master-file absorption of `set_active_power_unit.py`, documentation updates.
9. **Verify Phase 3** following plan §20 Step 23.
10. **Execute Phase 4 (plan §27 Steps 24–32).** Step 24 is critical — land alongside Phase 1.5 / Phase 3 Step 16 if practical. Then Steps 25–32 in order.
11. **Verify Phase 4** per the per-step verification criteria in the plan and §31 Definition of Done.

You may interleave Phase 4 Step 24 with Phase 1.5 (both are correctness bugs in the same service surface — `daily_job` and `loss_rollup_job` integration logic). All other Phase 4 steps strictly follow Phase 3.

## HARD RULES (violating any of these is a regression)

- **Telemetry contract is append-only.** Never rename or remove a key in `TELEMETRY_CONTRACT.md`. New keys require a 90-day deprecation window before any rename.
- **No plant-specific hardcoding in service code.** Read all plant-varying values from TB SERVER_SCOPE attributes. The only acceptable hardcoded "fleet defaults" are the constants documented in `app/config.py`.
- **No assumptions on TB key naming.** A plant publishing `active_power` in W is normal; another publishing it in kW is normal; another using `power_v3` is normal. Per-plant attrs drive behavior.
- **No widget can edit TB attributes.** Widget JS reads attributes; only the service writes them.
- **Idempotency required** for all admin endpoints and onboarding scripts. Running them twice must not corrupt state.
- **Asia/Colombo timezone** for all local-day-boundary logic. Local midnight = UTC+5:30 → use `ZoneInfo("Asia/Colombo")` in Python, the explicit `offsetMs = 330 * 60 * 1000` pattern in widget JS.
- **Sentinel value is `-1`** for service-written numeric keys when data is invalid. Widgets must filter `value < 0` out before summing.
- **`potential_power` is NOT P50.** It is a deterministic single-realization expected power from the live physics pipeline. P50 is the median of 19-year PVGIS replays. They co-exist as separate visualisations (blue dashed P50 vs green dotted "Physics Expected"). Don't conflate them.

## WHEN TO STOP AND ASK

Stop immediately if **any** of the following occur. Group your questions into one message, no more than four bullets.

- The plan instructs you to modify a file you cannot find at the stated path.
- A widget's existing behavior contradicts its README in a way the plan did not anticipate.
- An admin endpoint behaves differently from its docstring.
- You discover a telemetry key the plan didn't mention being consumed by a widget.
- Hand-calc verification (plan §6 Step 6.3) of FDI doesn't reconcile to widget output within 0.5 %.
- A plant in the fleet has `pvlib_enabled=true` but produces sentinels for every cycle.
- You catch yourself wanting to add a new telemetry key not listed in the plan.

Do **not** stop for:
- Style / lint preferences.
- Minor inconsistencies between README versions and code that the plan explicitly notes (e.g., FDI .js says v2.2 while README says v4.0 — the plan accounts for this).
- Whitespace / formatting churn in existing files (leave it alone).

## DEFINITION OF DONE

Phase 1.5 (CRITICAL — must land first):
- `daily_job.py` reads `active_power_unit` and applies `× 0.001` when `"W"`.
- `daily_job.py` reads `actual_power_keys` (plural CSV) with `"active_power"` default; legacy `actual_power_key` (singular) honoured with one-time WARN log.
- A W-publishing plant's `actual_daily_energy_kwh` for a recent day matches the same day's `exported_energy_daily_kwh` from `loss_rollup_job` within 0.5 %.
- Historical backfill issued for every W-publishing plant via `/admin/run-daily-range`; spot-checks confirm corrected values in TB.

Phase 1:
- FDI 3-instance pattern (P50/P90/P95) renders the correct cumulative MTD numbers against KSP for the current month. Hand-calc reconciles within 0.5 %.
- FvA renders P-bands when forecast keys are missing for the current year by falling through to derived mode (flat lines).
- README for FDI matches code (v4.0); legacy `forecastMtdKey` setting remains in settings.json as an ignored alias with help text.
- No service code edited in Phase 1 (Phase 1.5 already covered the service-side dependency).

Phase 2:
- A plant with `actual_power_keys=<non-default>` produces non-zero `actual_daily_energy_kwh` after `/admin/run-daily?date=...`.
- `scripts/shared/tb_config_loader.py --dry-run` prints a coherent per-plant diff against the master file for at least one real plant.
- `scripts/shared/onboard_plant.py --asset-id <id> --dry-run` lists the eight orchestration steps without executing them; with `--dry-run` removed, the steps run in order and the plant has past + current + future telemetry visible in dashboards.
- `audit_tb_config.py` exits non-zero when a plant is missing `latitude`, and exits zero when all required attrs are present.
- Telemetry contract updated to list `actual_power_keys` (and legacy alias `actual_power_key`) as read attributes.
- `AUTO_ONBOARD_BACKFILL_ENABLED` defaults to `false`. Documented in service README.

Phase 3:
- `daily_job` and `loss_rollup_job` use per-plant `config.timezone` for day-boundary math (still default to `settings.TZ_LOCAL` when missing). Cron firing time unchanged.
- `scripts/shared/find_orphan_plants.py` exits 1 when at least one TB asset has `isPlant=true && pvlib_enabled=true` but isn't reachable from any `TB_ROOT_ASSET_IDS` ancestor.
- `scripts/shared/find_config_drift.py` exits 1 when the master-file SHA-1 doesn't match the `pvlib_config_hash` SERVER_SCOPE attribute for any plant.
- `scripts/shared/_config_hash.py` is imported by both the service and the drift script (single source of truth for hashing).
- `scripts/shared/set_active_power_unit.py` is reduced to a deprecation stub.
- `TELEMETRY_CONTRACT.md` updated: daily rollup ts is plant-local midnight; the read-attrs table includes `actual_power_keys`, `active_power_unit`, `timezone`.

Phase 4:
- `loss_rollup_job._integrate` rewritten to cadence-correct trapezoidal integration. A 5-min plant's `potential_energy_daily_kwh` matches `total_generation_expected_kwh` from `daily_job` within 1 %; same for `exported_energy_daily_kwh` vs `actual_daily_energy_kwh`.
- `pvlib_services` JSON attribute parsed in `discover_plants` and attached to `PlantRef`; each cron skips plants whose corresponding group is `false`; `pvlib_enabled=false` master gate behaviour preserved.
- `revenue_job.py` writes 5 keys; the Expected vs Actual Revenue widget renders both `monthly` (default) and `yearly` (10-year) view modes against real data.
- `pvalue_job` writes `p50_energy_annual` / `p90_energy_annual` / `p95_energy_annual` timeseries (per-year) alongside the existing SERVER_SCOPE annual attrs.
- `pvalue_job` per-plant daily+mtd write count halved via record merge by ts.
- `forecast_service` plant-attrs cache (TTL configurable; default 300 s) cuts attribute reads from 1.44 M/day to ~288 k/day at 1000 plants.
- `auto_onboard.py` Sunday-03:00 cron runs full historical backfill chain since `commissioning_date` for every plant where `onboarding_completed != true`, then sets `onboarding_completed=true` + `onboarding_completed_at`. `AUTO_ONBOARD_ENABLED` defaults to `false`.
- `HOSTING_QUICKSTART.md` is a 5-page copy-paste runbook for an operator with Linux + EC2 + Docker baseline. `HOSTING_REFERENCE.md` covers every TB attribute, every telemetry key, every widget mapping recipe, and troubleshooting.

## DELIVERABLES (in your final report)

1. A one-paragraph summary of what changed.
2. A list of files touched, grouped by Phase 1.5 / Phase 1 / Phase 2 / Phase 3 / Phase 4.
3. The exact verification commands you ran with their pass/fail outcome (per phase).
4. Any deviations from the plan (with one-sentence justification each).
5. Open questions for Pasindu (max five, grouped).
6. Memory hook: a one-line summary suitable for a future Opus session to recall this work.

## ANTI-PATTERNS YOU WILL BE PENALISED FOR

- Pasting "I've implemented all the steps" without showing the diff or the verification output.
- Adding new telemetry keys to fix an awkward widget edge case (correct path: extend the widget or ask).
- Hardcoding plant UUIDs anywhere in service code (correct path: SERVER_SCOPE attrs).
- Removing legacy aliases from settings.json without the 90-day deprecation cycle.
- Editing files outside the file index in §9, §14, §21, or §28 of the plan.
- Treating Phase 1.5 as "optional" or "cosmetic". It is data-correctness; Phase 1 verification depends on it.
- Re-using `set_active_power_unit.py` instead of the bulk loader after Phase 2 lands.
- Skipping verification because "it looks right".
- Inferring the timezone offset from the host clock instead of `ZoneInfo("Asia/Colombo")` / the explicit offset.

## OUTPUT FORMAT

- Use plain Markdown.
- Diffs in fenced code blocks tagged with the language (`python`, `javascript`, `json`).
- File paths in backticks, absolute when possible.
- No emoji.
- No fluff.

---

You may begin by reading `WIDGET_PVALUE_FIX_PLAN.md` end-to-end and then summarising back to me, in one paragraph, what you understand the scope to be before you touch any file. I will confirm or correct, then you proceed.
