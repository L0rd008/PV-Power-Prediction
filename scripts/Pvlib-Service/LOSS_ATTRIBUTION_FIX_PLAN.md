# Loss Attribution Fix Plan — round 2
*Pvlib-Service ⇄ Loss Attribution widget*
*Author: planning + audit pass — 2026-05-04*
*Target executor: Sonnet 4.6*

This is the second-round plan. The first-round Plan (`LOSS_ATTRIBUTION_TELEMETRY_PLAN.md`) and the corresponding Sonnet brief (`LOSS_ATTRIBUTION_SONNET_PROMPT.md`) covered the original goal: move per-bucket loss math from the widget into the service. Sonnet 4.6 implemented that work. This document audits the result, identifies the bugs that are crashing the dashboard, scores fix approaches, and lays out the work for a follow-up Sonnet pass.

---

## 1. Audit of Sonnet's first-round implementation

### Files Sonnet added or changed

| File | Status |
|---|---|
| `scripts/Pvlib-Service/app/services/loss_rollup_job.py` | NEW (993 lines) |
| `scripts/Pvlib-Service/app/services/scheduler.py` | edited — registered cron at 00:10 local, gated on `LOSS_ROLLUP_ENABLED` |
| `scripts/Pvlib-Service/app/api/forecast.py` | edited — added `/admin/run-loss-rollup`, `/admin/recompute-lifetime`, `/admin/loss-status` (verify) |
| `scripts/Pvlib-Service/app/services/thingsboard_client.py` | edited — added `post_attributes` (verify) |
| `scripts/Pvlib-Service/app/config.py` | edited — added `LOSS_ROLLUP_ENABLED`, `LOSS_DEFAULT_SETPOINT_KEYS`, `LOSS_MIN_VALID_SAMPLES`, `LOSS_LIFETIME_PAGE_DAYS` |
| `scripts/Pvlib-Service/TELEMETRY_CONTRACT.md` | edited — appended new keys |
| `scripts/Pvlib-Service/tests/test_loss_rollup.py` | NEW |
| `Widgets/Grid & Losses/Loss Attribution/.js` | edited — added precomputed branch |
| `Widgets/Grid & Losses/Loss Attribution/settings.json` | edited — added new settings |
| `Widgets/Grid & Losses/Loss Attribution/README.md` | edited — added `curtailRevenue` row + server-side aggregation section |

### What followed the spec

- `loss_rollup_job.py` implements the daily integration math correctly (lines 409–480).
- Sentinel handling matches the existing pattern (`_make_sentinel_result`, `_sentinel_daily_values`, `_safe_write_daily`).
- W → kW unit scaling is honoured via the `active_power_unit` attribute (lines 288–289, 349–350).
- Per-plant override of `setpoint_keys` is supported (lines 292–293).
- Lifetime maintenance has both an increment-step path and a recompute-from-history path with anchor date (`_update_lifetime`, `_lifetime_increment_step`, `_recompute_lifetime_from_history`, lines 485–658).
- Ancestor roll-up of daily values + lifetime attributes is in place (`_sum_daily_values`, `_sum_lifetime_attrs`).
- Cron at 00:10 local is registered, gated on `LOSS_ROLLUP_ENABLED` (`scheduler.py:230–243`).
- New widget branch dispatches on `useNewLossKeys` and uses `isCurrentDay` to keep the per-minute path for "today" (`.js:188–214, 601–608`).

### Deviations from the original Plan

| # | Deviation | Plan reference | Severity |
|---|---|---|---|
| D1 | Added `KEY_POTENTIAL_MONTHLY` and `KEY_POTENTIAL_YEARLY` keys + a per-day `_get_historical_sum` that re-fetches running monthly/yearly sums for every plant every day. | Plan §2 P2 hybrid H2 explicitly chose "no monthly / yearly precomputed keys"; Plan §7 forbade them by name. | Medium — extra TB load + storage; not the cause of the crash. |
| D2 | `run_loss_rollup` writes daily keys twice per plant: once inside `_process_plant`, then again in the outer loop after the monthly/yearly fields are added (`loss_rollup_job.py:183`). | Implicit from D1. | Low — wasted round-trip per plant per day. |
| D3 | Added a tariff-snapshot key `loss_tariff_rate_lkr_at_compute` written daily. | Not in Plan §6 contract additions. | Low — useful for tariff-history audit but not on the contract; should be documented or removed. |
| D4 | `_recompute_lifetime_from_history` falls back to `2020-10-01` as a fleet-wide commissioning date (line 595). | Plan §4.2 didn't pin a fallback. Reasonable, but should be configurable. | Cosmetic. |

### Bugs introduced or surfaced (the dashboard-killers)

The user reports: "whenever I change to the Grid & Losses tab on ThingsBoard it crashes; the Loss Attribution widget disappears, reloads with no value, then loads again to show the value." The audit traces every symptom to the widget, not the service. The cron itself is dormant by design (`LOSS_ROLLUP_ENABLED = false` until phase L1).

| # | Bug | File:line | Effect |
|---|---|---|---|
| B1 | `onDataUpdated` re-runs on every TB telemetry tick. Because the widget's DS[0] subscribes to a key the service writes every minute (e.g. `potential_power`, the recommended placeholder), TB fires `onDataUpdated` ≈ 60×/hr per card. Each fire calls `debouncedComputedRender()` (150 ms debounce) which calls `setLoadingState()` (wipes value) then re-fetches primary + comparator. | `.js:131–144`, `.js:154–159`, `.js:347–354` | Per-minute flicker on every card; with 4–6 cards this is a request storm. |
| B2 | `setLoadingState` always wipes the value to `--` and adds `skeleton` even on background refreshes. | `.js:347–354` | The visible "disappear → CALC → value" flicker. |
| B3 | When `useNewLossKeys = "auto"` and the precomputed keys are missing (cron not yet enabled), the widget falls back to legacy per-bucket fetch. For Year/Lifetime ranges that's hundreds of TB requests per render per card. With B1 the storm fires every minute. | `.js:203–213`, `.js:610–731`, `.js:869–901` | TB returns 503 / browser tab freezes. |
| B4 | `fetchCalculationAttributes` is called on every render — capacity + tariff are re-fetched every minute per card. | `.js:216–217`, `.js:777–790` | Two extra round-trips per card per minute. |
| B5 | `updateDom` adds a `loss-range-changed` event listener to `window` every time it runs. `onInit` only calls it once today, but if TB ever invokes `updateDom` on settings change the listeners stack indefinitely. | `.js:119–128` | Latent bug; currently masked by single-call lifecycle. |
| B6 | `calculateLossForRangePrecomputed` requires `hasPotential && hasGrid` for `ok=true`. If one daily key is sentinel for the whole range while the other isn't, it returns `ok=false` and the widget falls back to legacy. | `.js:568–569` | Silent fallback to slow path on partial data. |
| B7 | `isCurrentDay` only catches `range.mode === 'day' AND today`. "Current Month" / "Current Year" / "Lifetime" all include today's day-in-progress, but no precomputed key exists for today (the cron only runs at 00:10 the next day). The widget either silently falls back to legacy or shows yesterday-and-earlier-only sums — under-reporting today's contribution. | `.js:601–608` | Stale or wrong values for the most common ranges. |
| B8 | Lifetime branch reads SERVER_SCOPE attributes on every minute tick. | `.js:484–530` | One round-trip × every minute × every card. |
| B9 | For `'force'` mode the widget gives no fallback; if precomputed isn't ready it shows `--`. | `.js:193–195` | Confusing UX during phase L1 rollout. |

The dashboard crash is **B1 + B2 + B3** combined: per-minute renders × multiple cards × full slow path on missing precomputed keys × no tab focus throttling. The flicker is **B2** alone.

### What's still missing relative to the user's update-cadence question

The first-round Plan (P9 / hybrid H9) decided to keep "today current-day" on the legacy per-minute path. That assumed precomputed keys were either there for past days or absent for today only — but the widget treats `Current Month` and `Current Year` as past-range, even though those ranges *include today*. The original design did not solve "current month including today's partial contribution" cleanly. This is the gap the user is pointing at: you can't have the widget show fresh data without either (a) the widget computing today client-side, or (b) the service writing a today-partial roll-up at some cadence.

---

## 2. Problems and approaches (round 2)

### P1 — Update cadence

The user proposed 30-min for both server and widget. We separate the two:

#### P1a — Server cadence (when does the service refresh today's daily key?)

| # | Approach | Pros | Cons | Score (≥90 % conf) |
|---|---|---|---|---|
| C1 | Daily only at 00:10 (status quo) | Cheapest | Today's daily key only exists at 00:10 next day; widgets querying "current month" miss today's contribution all day | 4 |
| C2 | Daily + new "today partial" cron every 30 min | Today's contribution to current-month/year is at most 30 min stale; server cost ≈ 48× daily-job's per-day cost / 24 = 2× cheaper than per-minute | One extra cron; same daily-keys contract; small extra storage churn (today's row overwrites itself) | 9 |
| C3 | Daily + 5-min today-partial | Very fresh today data | 12× the C2 cost; little payoff vs 30 min for human-facing dashboards | 7 |
| C4 | On-demand only (admin endpoint, no cron) | Lowest cost | Today data stale until next-day 00:10 | 5 |
| C5 | Continuous — fold loss math into the existing 1-min `pvlib_cycle` | Live data | Re-runs full integration per plant per minute; defeats the whole reason we moved math to server; same TB churn we're trying to escape | 3 |

**Decision: C2.** A new APScheduler interval job runs every 30 min during local daylight (configurable; default 05:00–19:00 local) and recomputes today's six daily keys. The math is identical to the daily cron — only the date window changes ("today so far" instead of "yesterday full day"). Lifetime attributes are NOT updated by this 30-min job (they only advance once the day is finalised at 00:10).

#### P1b — Widget cadence (when does the widget refetch?)

| # | Approach | Pros | Cons | Score |
|---|---|---|---|---|
| W1 | Refresh on every `onDataUpdated` (status quo) | Always fresh | Causes B1, B2, B3 — dashboard crash | 1 |
| W2 | Refresh only on first render + range change | Stable; cheapest | "Current month/year" view stale until user navigates away and back | 5 |
| W3 | W2 + a 30-min poll throttle that reuses the cached attrs and the precomputed daily keys | Aligns with server cadence; cheap | If user opens dashboard at 00:09, "current month" shows yesterday until +30 min | 7 |
| W4 | W3 + one immediate refetch when range crosses today (current month / current year / lifetime / custom containing today) | Fresh today; cheap past ranges | Slightly more code (range classification helper) | 9 |
| W5 | W4 + page-visibility gate (no polling when tab hidden) | Saves background bandwidth | One extra event listener | 9 |

**Decision (hybrid H1b): W4 + W5.** Refresh policy:

- **Range change** → full refetch with skeleton.
- **First render after `onInit`** → full refetch with skeleton.
- **Periodic refresh, every 30 min while the tab is visible** → silent refetch (no skeleton, value stays visible until new value lands).
- **Range crosses today** (i.e. `endTs >= today_local_midnight`) → silent 30-min refresh active.
- **Range entirely in the past** (`endTs < today_local_midnight`) → no periodic refresh at all (data is immutable until backfill).
- **Tab hidden** (`document.visibilityState === 'hidden'`) → suspend timer; resume on visibility change.

Score: 9.

### P2 — Flicker

| # | Approach | Pros | Cons | Score |
|---|---|---|---|---|
| F1 | Skip `setLoadingState` on background refresh; only show skeleton on first-render or range/entity change | Fixes the visible flicker cleanly | Small extra state (`_hasRendered`, `_isInitialOrRangeChange`) | 9 |
| F2 | Render the new value transactionally over the old DOM | Same as F1 with transition smoothness | More code | 7 |
| F3 | CSS opacity transition over the skeleton | Hides the wipe visually; value still cleared | Half-fix | 4 |

**Decision: F1.**

### P3 — Decoupling the widget from `onDataUpdated`

`onDataUpdated` will still fire on every TB tick because the TB framework wires it to the widget's DS[0]. The cleanest way to stop the per-minute storm is to make `onDataUpdated` a no-op for computed modes (or call it only as the trigger for the legacy fallback path) and rely on our own timer.

| # | Approach | Pros | Cons | Score |
|---|---|---|---|---|
| D1 | Make `onDataUpdated` no-op for computed modes; manage refresh via internal `setInterval` + visibility events | Clean; testable; predictable | One new lifecycle path | 9 |
| D2 | Throttle `onDataUpdated` callbacks at 30 min via timestamp comparison | Less code | Still re-runs body of function on every tick to check throttle; slightly wasteful | 7 |
| D3 | Subscribe to a "heartbeat" datasource at 30 min instead of per-minute | Re-uses TB framework | Requires dashboard-level config changes; couples widget to a specific telemetry key | 4 |

**Decision: D1.**

### P4 — Sonnet's monthly/yearly deviation

| # | Approach | Pros | Cons | Score |
|---|---|---|---|---|
| R1 | Revert: drop `KEY_POTENTIAL_MONTHLY`, `KEY_POTENTIAL_YEARLY`, `_get_historical_sum`, the duplicate write | Matches original Plan; simpler contract | Small risk if some downstream system already depends on these keys (none does — they were just shipped, no consumer yet) | 9 |
| R2 | Keep them and document them | No risk | Bloats contract; doesn't match decision matrix | 5 |
| R3 | Keep them but compute as a TB SUM-aggregation read on the widget side instead of writing them | Widget gets the same numbers without server-side history fetch | Widget already has a fast sum-over-daily approach; this adds a TB AGG dependency | 5 |

**Decision: R1.** Revert.

### P5 — Tariff-snapshot key

Sonnet added `loss_tariff_rate_lkr_at_compute` not in the original contract. It does help auditors verify "what tariff was used on day X" without re-fetching attribute history.

| # | Approach | Pros | Cons | Score |
|---|---|---|---|---|
| T1 | Keep the key, document it in TELEMETRY_CONTRACT.md | Useful audit trail; cheap (one float per day) | Slight contract growth | 8 |
| T2 | Drop it | Clean contract | Loses audit trail | 6 |

**Decision: T1.** Document and keep.

### P6 — Permissiveness of the precomputed `ok` flag

| # | Approach | Pros | Cons | Score |
|---|---|---|---|---|
| O1 | `ok = hasPotential && hasGrid` (current) | Strict | False negatives on partial data → silent fallback to slow path | 6 |
| O2 | `ok = hasPotential` (looser) | Catches the common case | Still false-negative if potential is sentinel one day but the user's range is wider | 8 |
| O3 | `ok = (any of the four kWh keys) >= 0` | Most permissive | Could mask data-quality issues | 6 |

**Decision: O2.** Require `hasPotential`. If grid loss is sentinel, render `0` for grid loss but keep the card up — this matches the existing legacy semantics where missing data ≈ no loss.

### P7 — `isCurrentDay` shortcut should become `rangeIncludesToday`

| # | Approach | Pros | Cons | Score |
|---|---|---|---|---|
| I1 | Rename + broaden: returns true for any range whose `endTs >= today_local_midnight` | Generic; current-month / current-year / lifetime all benefit | Need to differentiate "today only" from "range that ends today" for the periodic-refresh decision | 9 |
| I2 | Add a "split path": for ranges that include today, fetch precomputed for past days + add the today-partial via legacy per-minute math on top | Most accurate when 30-min today cron is unavailable | Complex; doubles the code paths | 6 |

**Decision: I1.** With Fix C2 in place (server writes today-partial every 30 min), the precomputed key for today already covers the day-in-progress. We don't need I2.

### P8 — Crash protection for legacy fallback on huge ranges

Even with everything above, a plant whose precomputed keys are missing (any non-pvlib_enabled plant or first-deploy plants) will hit the legacy path. For Year/Lifetime ranges that path is the original perf disaster.

| # | Approach | Pros | Cons | Score |
|---|---|---|---|---|
| L1 | When fallback would be needed for `range.mode in {year, lifetime, custom > 60 days}` AND precomputed is empty → show a "Data not yet available" placeholder instead of fetching | Prevents the crash | Loses backwards compat for non-pvlib plants for these ranges | 7 |
| L2 | L1 + an "enable precomputation for this plant" inline hint in the widget tooltip | Helpful | Slight UI clutter | 8 |
| L3 | Always allow legacy fallback but cap it: max chunks per render (~10) and warn in tooltip | Backwards compat | Year mode under-reports; user confused | 6 |
| L4 | Legacy fallback only for `day` and `month` ranges; year/lifetime require precomputed | Prevents the worst storms | Same drawback as L1 | 8 |

**Decision (hybrid H8): L1 + L2.** For year/lifetime/custom > 60 d, if precomputed is empty, render `--` with a tooltip "This plant isn't yet rolled up. Ask your operator to enable Loss Attribution rollup for this asset." (or pull the plant name into the message). Day and month ranges keep the legacy fallback.

### P9 — `updateDom` listener leak

| # | Approach | Pros | Cons | Score |
|---|---|---|---|---|
| E1 | Move the `window.addEventListener('loss-range-changed', ...)` registration into `onInit` | One listener, ever | Simple | 10 |

**Decision: E1.**

---

## 3. The chosen hybrid bundle

- C2 — server: 30-min today-partial cron + existing 00:10 daily cron.
- W4 + W5 — widget: refresh on init/range-change/30-min throttle while range includes today and tab is visible; no periodic refresh for past-only ranges.
- F1 — skeleton only on init or range/entity change.
- D1 — `onDataUpdated` becomes no-op for computed modes; widget owns its refresh timer.
- R1 — revert monthly/yearly keys and the duplicate write.
- T1 — keep `loss_tariff_rate_lkr_at_compute`, document it.
- O2 — `ok = hasPotential` for precomputed result.
- I1 — `rangeIncludesToday` replaces `isCurrentDay`.
- L1 + L2 — large ranges with no precomputed data render a guidance message.
- E1 — listener registered in `onInit` only.

Each sub-decision scored ≥ 8/10 with ≥ 90 % confidence; the bundle's worst-case failure mode (precomputed missing on a huge range) is now an explicit user-facing message, not a tab freeze.

---

## 4. Step-by-step server changes

### 4.1 Revert the monthly/yearly deviation (`loss_rollup_job.py`)

- Remove `KEY_POTENTIAL_MONTHLY`, `KEY_POTENTIAL_YEARLY` from the constants and from `_safe_write_daily`.
- Remove `_get_historical_sum`.
- Remove the second `_safe_write_daily` call inside `run_loss_rollup` (the one in the outer `for plant, result in zip(...)` loop at lines 172–185). The single write inside `_process_plant` is sufficient.
- Remove the matching ancestor monthly/yearly logic at lines 195–214.

### 4.2 Add a today-partial cron (`loss_rollup_job.py` + `scheduler.py` + `config.py`)

In `loss_rollup_job.py`:

```python
async def run_today_partial_rollup(tb_client) -> Dict[str, object]:
    """Recompute today-so-far daily values for every pvlib_enabled plant.

    Identical math to run_loss_rollup but the day window is
        [today_local_midnight, now_local].
    Writes the same six daily keys at the today_local_midnight ts (overwriting
    any prior partial). Does NOT update lifetime attributes — those are advanced
    only by the 00:10 cron when the day is finalised.
    """
    tz = ZoneInfo(settings.TZ_LOCAL)
    now_local = datetime.now(tz)
    day_start_local = now_local.replace(hour=0, minute=0, second=0, microsecond=0)
    day_start_utc = day_start_local.astimezone(timezone.utc)
    end_utc = now_local.astimezone(timezone.utc)
    day_ts_ms = int(day_start_local.timestamp() * 1000)

    plants, ancestor_map = await tb_client.discover_plants(settings.root_asset_ids)
    if not plants:
        return {"plants_ok": 0, "plants_failed": 0, "plants_skipped": 0,
                "date": str(day_start_local.date()), "partial": True}

    sem = asyncio.Semaphore(settings.MAX_CONCURRENT_PLANTS)

    async def _process(plant):
        async with sem:
            return await _process_plant(
                tb_client, plant.id, day_start_utc, end_utc,
                day_ts_ms, day_start_local,
            )

    results = await asyncio.gather(*(_process(p) for p in plants), return_exceptions=True)

    plant_daily: Dict[str, Dict[str, float]] = {}
    stats = {"ok": 0, "failed": 0, "skipped": 0}
    for plant, result in zip(plants, results):
        if isinstance(result, Exception):
            stats["failed"] += 1
            continue
        if result.get("skipped"):
            stats["skipped"] += 1
            continue
        if result.get("ok"):
            stats["ok"] += 1
        else:
            stats["failed"] += 1
        plant_daily[plant.id] = result.get("daily_values", _sentinel_daily_values())

    # Ancestor roll-up of daily keys ONLY (no lifetime updates)
    ancestor_children: Dict[str, Set[str]] = {}
    for plant_id, ancestors in ancestor_map.items():
        for anc_id in ancestors:
            ancestor_children.setdefault(anc_id, set()).add(plant_id)

    for ancestor_id, child_ids in ancestor_children.items():
        summed = _sum_daily_values([plant_daily[cid] for cid in child_ids if cid in plant_daily])
        await _safe_write_daily(tb_client, ancestor_id, day_ts_ms, summed, "rollup:partial")

    return {
        "plants_ok": stats["ok"],
        "plants_failed": stats["failed"],
        "plants_skipped": stats["skipped"],
        "date": str(day_start_local.date()),
        "partial": True,
    }
```

Reuse `_process_plant` as-is — it already takes `day_end_utc` as an argument; it will integrate from `day_start_utc` to `now_utc`. The minimum-samples gate (`LOSS_MIN_VALID_SAMPLES = 360`) WILL trip in the early morning before there are 6 hours of data. Add a parameter `min_samples_override: Optional[int]` so the today-partial cron can pass a lower threshold (e.g. 30 — half an hour of data) without changing the daily-job behaviour.

In `app/config.py` add:

```python
LOSS_TODAY_PARTIAL_ENABLED: bool = False
"""Master flag for the 30-min today-partial cron. Default false until phase L1."""

LOSS_TODAY_PARTIAL_INTERVAL_MIN: int = 30
"""Interval in minutes for the today-partial cron (default 30)."""

LOSS_TODAY_PARTIAL_DAY_START_HOUR: int = 5
"""Local-tz hour at which the today-partial cron starts firing (default 05:00)."""

LOSS_TODAY_PARTIAL_DAY_END_HOUR: int = 19
"""Local-tz hour at which the today-partial cron stops firing (default 19:00).
Outside this window the job is a no-op — no solar generation expected."""

LOSS_TODAY_PARTIAL_MIN_SAMPLES: int = 30
"""Minimum 1-min samples required by the today-partial path. Lower than the
daily-job threshold (360) because we want partial values from mid-morning."""
```

In `scheduler.py` register the new job:

```python
async def run_today_partial_now() -> dict:
    if not settings.LOSS_ROLLUP_ENABLED or not settings.LOSS_TODAY_PARTIAL_ENABLED:
        return {"status": "disabled"}
    # Skip outside daylight hours
    tz = ZoneInfo(settings.TZ_LOCAL)
    h = datetime.now(tz).hour
    if h < settings.LOSS_TODAY_PARTIAL_DAY_START_HOUR or h >= settings.LOSS_TODAY_PARTIAL_DAY_END_HOUR:
        return {"status": "outside_window", "hour": h}

    from app.services.loss_rollup_job import run_today_partial_rollup
    from app.services.thingsboard_client import ThingsBoardClient

    effective = _tb_client
    if effective is not None:
        return await run_today_partial_rollup(effective)
    async with ThingsBoardClient(settings.TB_HOST, settings.TB_USERNAME, settings.TB_PASSWORD) as tb:
        return await run_today_partial_rollup(tb)
```

Register inside `start_scheduler` (after the daily-loss-rollup block):

```python
if settings.LOSS_ROLLUP_ENABLED and settings.LOSS_TODAY_PARTIAL_ENABLED:
    scheduler.add_job(
        run_today_partial_now,
        trigger="interval",
        minutes=settings.LOSS_TODAY_PARTIAL_INTERVAL_MIN,
        misfire_grace_time=120,
        max_instances=1,
        id="pvlib_loss_today_partial",
        replace_existing=True,
        coalesce=True,
    )
    log.info("scheduler: today-partial cron registered every %d min %02d:00–%02d:00 %s",
             settings.LOSS_TODAY_PARTIAL_INTERVAL_MIN,
             settings.LOSS_TODAY_PARTIAL_DAY_START_HOUR,
             settings.LOSS_TODAY_PARTIAL_DAY_END_HOUR,
             settings.TZ_LOCAL)
```

### 4.3 New admin endpoint (`app/api/forecast.py`)

```python
@router.post("/admin/run-today-partial")
async def admin_run_today_partial():
    """Manually trigger the 30-min today-partial loss roll-up."""
    from app.services.scheduler import run_today_partial_now
    return await run_today_partial_now()
```

### 4.4 Update `TELEMETRY_CONTRACT.md`

- Document `loss_tariff_rate_lkr_at_compute` (DOUBLE, daily, LKR/kWh, the tariff in effect when the row was written; -1 if no tariff).
- Add a one-paragraph note about the today-partial cron: "The same six `loss_*_daily_*` keys may be re-written multiple times during the current day with `loss_data_source = 'partial'` — the latest write before 00:00 UTC of the next local day represents today-so-far."
- Remove (or retract with a deprecation note) `potential_energy_monthly_kwh` and `potential_energy_yearly_kwh` if Sonnet documented them.

### 4.5 Update tests (`tests/test_loss_rollup.py`)

- Remove tests around `KEY_POTENTIAL_MONTHLY` / `KEY_POTENTIAL_YEARLY` if any.
- Add tests for `run_today_partial_rollup`:
  - Asserts daily keys are written for `today_local_midnight` ts, not yesterday.
  - Asserts lifetime attributes are NOT updated.
  - Asserts low sample count (`< LOSS_TODAY_PARTIAL_MIN_SAMPLES`) writes -1 sentinels.
  - Asserts outside-window short-circuit.

---

## 5. Step-by-step widget changes (`Widgets/Grid & Losses/Loss Attribution/.js`)

### 5.1 `onInit` becomes the single source of side-effect setup

Move the `window.addEventListener('loss-range-changed', self._onRangeChanged)` from `updateDom` (line 128) into `onInit`. Remove it from `updateDom`. This fixes B5.

### 5.2 Add internal refresh state

Add at top of `onInit`:

```javascript
self._hasRendered = false;
self._activeRange = null;        // last range we rendered for
self._activeEntityKey = null;    // last entity id+type we rendered for
self._cachedAttrs = null;        // {capacity, tariffRate, fetchedAt}
self._refreshTimer = null;       // setInterval handle
self._visibilityHandler = null;  // page visibility listener
self._POLL_INTERVAL_MS = 30 * 60 * 1000;     // 30 min
self._ATTR_TTL_MS      = 30 * 60 * 1000;     // 30 min
self._INIT_REFETCH_DELAY_MS = 0;             // first render is immediate
```

### 5.3 New refresh policy

Replace the body of `self.onDataUpdated` with:

```javascript
self.onDataUpdated = function () {
    var mode = getMode();
    if (mode === 'rangeSelector') {
        renderSelectorRange(getActiveRange());
        return;
    }
    var def = getModeDef(mode);
    if (!def.computed) {
        // Insurance and other latest-value modes: light path, no skeleton storm
        renderLatestValueFallback();
        return;
    }
    // Computed modes: NO refresh on TB tick. Our own timer drives it.
    // Just kick the timer in case this is the first datasource update.
    ensureRefreshTimer();
};
```

Add helpers:

```javascript
function ensureRefreshTimer() {
    if (self._refreshTimer) return;
    var range = getActiveRange();
    // No periodic refresh for past-only ranges
    if (!rangeIncludesToday(range)) return;

    self._refreshTimer = setInterval(function () {
        if (document.visibilityState === 'hidden') return;
        var current = getActiveRange();
        if (!rangeIncludesToday(current)) {
            clearRefreshTimer();
            return;
        }
        renderComputedMode({ silent: true });
    }, self._POLL_INTERVAL_MS);

    if (!self._visibilityHandler) {
        self._visibilityHandler = function () {
            if (document.visibilityState === 'visible') {
                // On tab focus, do one immediate silent refresh then resume
                var current = getActiveRange();
                if (rangeIncludesToday(current)) {
                    renderComputedMode({ silent: true });
                }
            }
        };
        document.addEventListener('visibilitychange', self._visibilityHandler);
    }
}

function clearRefreshTimer() {
    if (self._refreshTimer) {
        clearInterval(self._refreshTimer);
        self._refreshTimer = null;
    }
}

function rangeIncludesToday(range) {
    if (!range) return false;
    if (range.mode === 'lifetime') return true;
    var todayStart = new Date();
    todayStart.setHours(0, 0, 0, 0);
    return parseInt(range.endTs, 10) >= todayStart.getTime();
}
```

### 5.4 Range-change handler triggers a non-silent refresh

In `onInit`, after `self._onRangeChanged` is defined, modify it so that after storing the range it forces a non-silent re-render:

```javascript
self._onRangeChanged = function (e) {
    if (!e || !e.detail) return;
    self._activeRangeOverride = e.detail;
    storeRangeLocal(e.detail);
    if (getMode() === 'rangeSelector') return;
    clearRefreshTimer();           // restart timer for the new range
    renderComputedMode({ silent: false });  // show skeleton — range changed
    ensureRefreshTimer();
};
```

### 5.5 `renderComputedMode` accepts a silent flag

Modify the signature:

```javascript
function renderComputedMode(opts) {
    opts = opts || {};
    var silent = !!opts.silent;
    var token = ++self._calcToken;
    var mode = getMode();
    var def = getModeDef(mode);
    var entity = resolveEntity();
    var range = getActiveRange();

    if (!entity || !entity.id || !def.computed) {
        renderLatestValueFallback();
        return;
    }

    var entityKey = (entity.type || 'ASSET') + ':' + entity.id;
    var rangeKey = range.mode + ':' + range.startTs + ':' + range.endTs;
    var entityChanged = self._activeEntityKey !== entityKey;
    var rangeChanged  = self._activeRange      !== rangeKey;
    var firstRender   = !self._hasRendered;
    var showSkeleton  = !silent && (firstRender || entityChanged || rangeChanged);

    if (showSkeleton) {
        setLoadingState(range);
    }

    // ... (rest of the existing function body, BUT do not call setLoadingState inside)
    // At the end of the success branch, set:
    //   self._hasRendered = true;
    //   self._activeEntityKey = entityKey;
    //   self._activeRange = rangeKey;
}
```

Remove the `setLoadingState(range)` call that's currently unconditional at line 174. The new top-of-function gate replaces it.

### 5.6 Cache plant attributes

Replace `fetchCalculationAttributes` with a cached version:

```javascript
function fetchCalculationAttributes(entity) {
    var s = self.ctx.settings || {};
    var entityKey = entity.type + ':' + entity.id;
    var now = Date.now();
    if (self._cachedAttrs &&
        self._cachedAttrs.entityKey === entityKey &&
        (now - self._cachedAttrs.fetchedAt) < self._ATTR_TTL_MS) {
        return Promise.resolve(self._cachedAttrs);
    }
    var keys = uniqueList([
        s.plantCapacityKey || 'Capacity',
        s.tariffAttributeKey || 'tariff_rate_lkr'
    ]);
    return fetchAttributesWithFallback(entity, keys).then(function (attrs) {
        var out = {
            capacity: getAttr(attrs, s.plantCapacityKey || 'Capacity'),
            tariffRate: parseFloat(getAttr(attrs, s.tariffAttributeKey || 'tariff_rate_lkr')),
            entityKey: entityKey,
            fetchedAt: now
        };
        self._cachedAttrs = out;
        return out;
    });
}
```

### 5.7 Replace `isCurrentDay` with `rangeIncludesToday`

Delete `isCurrentDay` (lines 601–608) and use the new `rangeIncludesToday` helper from §5.3. The `'auto'` branch logic in `calcForRange` becomes:

```javascript
function calcForRange(rangeObj, attrs) {
    if (useNew === 'off') return calculateLossForRange(entity, rangeObj, attrs);
    if (useNew === 'force') return calculateLossForRangePrecomputed(entity, rangeObj, attrs);

    // 'auto': always try precomputed first, fall back only when sensible
    return calculateLossForRangePrecomputed(entity, rangeObj, attrs).then(function (preResult) {
        if (preResult && preResult.ok) return preResult;
        // Precomputed unavailable for this plant. Cap legacy fallback.
        if (legacyFallbackAllowed(rangeObj)) {
            return calculateLossForRange(entity, rangeObj, attrs);
        }
        // Too large for legacy without crashing — return ok=false; renderComputedMode
        // will show the user-facing "not yet rolled up" message.
        return { ok: false, tooLargeForLegacy: true };
    }).catch(function () {
        return legacyFallbackAllowed(rangeObj)
            ? calculateLossForRange(entity, rangeObj, attrs)
            : { ok: false, tooLargeForLegacy: true };
    });
}

function legacyFallbackAllowed(range) {
    if (!range) return true;
    if (range.mode === 'day' || range.mode === 'month') return true;
    if (range.mode === 'custom') {
        var spanDays = (range.endTs - range.startTs) / 86400000;
        return spanDays <= 60;
    }
    // year, lifetime → no
    return false;
}
```

### 5.8 Loosen the precomputed `ok` flag

In `calculateLossForRangePrecomputed`, change

```javascript
var hasPotential = potKwh >= 0;
var hasGrid      = gridKwh >= 0;
return { ok: hasPotential && hasGrid, ... }
```

to

```javascript
var hasPotential = potKwh >= 0;
return { ok: hasPotential, fromPrecomputed: true, ... };
```

…and treat any `< 0` in `gridKwh`, `curtailKwh`, etc. as `0` in the result (so the card renders 0 instead of going to fallback).

### 5.9 "Not yet rolled up" message

In `renderComputedMode`, when the result has `tooLargeForLegacy: true`:

```javascript
if (!result || !result.primary || !result.primary.ok) {
    if (result && result.primary && result.primary.tooLargeForLegacy) {
        showNotRolledUpPlaceholder(range);
    } else {
        renderLatestValueFallback();
    }
    return;
}
```

Add:

```javascript
function showNotRolledUpPlaceholder(range) {
    var $el = self.ctx.$widget;
    $el.find('.js-value').text('--').removeClass('skeleton');
    $el.find('.js-status-dot').removeClass('sev-low sev-moderate sev-high').addClass('sev-low');
    $el.find('.js-status-text').text('PENDING');
    $el.find('.js-footer-label').text(range.label || 'Loss Attribution');
    $el.find('.js-tooltip').text(
        'This plant has not been rolled up yet for the selected range. ' +
        'Ask your operator to enable Loss Attribution roll-up for this asset.'
    );
    hideDelta();
    detectChanges();
}
```

### 5.10 `onDestroy` cleanup

```javascript
self.onDestroy = function () {
    if (self._calcTimer) { clearTimeout(self._calcTimer); self._calcTimer = null; }
    self._calcToken++;
    clearRefreshTimer();
    if (self._onRangeChanged) {
        window.removeEventListener('loss-range-changed', self._onRangeChanged);
    }
    if (self._visibilityHandler) {
        document.removeEventListener('visibilitychange', self._visibilityHandler);
        self._visibilityHandler = null;
    }
};
```

### 5.11 `settings.json` additions

Append two settings to give operators control without re-uploading code:

```json
{
  "id": "pollIntervalMinutes",
  "name": "Background Refresh Interval (minutes)",
  "type": "number",
  "default": 30,
  "min": 5,
  "max": 240,
  "step": 5,
  "helpText": "Interval at which the card silently refreshes when the selected range includes today. Past-only ranges never auto-refresh."
},
{
  "id": "attrCacheMinutes",
  "name": "Capacity / Tariff Cache (minutes)",
  "type": "number",
  "default": 30,
  "min": 1,
  "max": 240,
  "step": 1,
  "helpText": "How long to cache the plant's Capacity and tariff_rate_lkr attributes between fetches."
}
```

Wire them into `_POLL_INTERVAL_MS` and `_ATTR_TTL_MS` at top of `onInit`:

```javascript
var pollMin = parseFloat(self.ctx.settings.pollIntervalMinutes);
self._POLL_INTERVAL_MS = (isFinite(pollMin) && pollMin >= 1 ? pollMin : 30) * 60 * 1000;
var attrMin = parseFloat(self.ctx.settings.attrCacheMinutes);
self._ATTR_TTL_MS = (isFinite(attrMin) && attrMin >= 1 ? attrMin : 30) * 60 * 1000;
```

### 5.12 README

Add a "Refresh policy" subsection to the existing "Server-side aggregation" section explaining: range change → immediate refetch; range crosses today + tab visible → 30-min silent refresh; past-only ranges → render once, never poll; tab hidden → suspend polling.

---

## 6. Plant attribute contract (the user's question)

Per `app/physics/config.py::PlantConfig` plus `forecast_service._load_config`, every attribute the service reads from a plant's TB SERVER_SCOPE is below.

### 6.1 Identity (REQUIRED for the plant to be discovered)

| Attribute | Type | Required | Notes |
|---|---|---|---|
| `isPlant` | boolean | yes | `true` to mark the asset as a plant (BFS leaf). |
| `pvlib_enabled` | boolean | yes | `true` to opt the plant into physics computation. Plants without this are silently skipped. |
| `name` | string | inherited | Asset name; usually inherited from the asset object itself, used for logging and for name-prefix WSTN lookup. |

### 6.2 Location (REQUIRED for physics)

| Attribute | Type | Required | Default if missing | Notes |
|---|---|---|---|---|
| `latitude` | double | yes | 0.0 (treated as unset → `no_location` sentinel) | Decimal degrees, −90..+90. |
| `longitude` | double | yes | 0.0 (same) | Decimal degrees, −180..+180. |
| `altitude_m` | double | recommended | 0.0 | Metres above sea level. |
| `timezone` | string | recommended | `"UTC"` | IANA tz; for the Sri Lanka fleet this is `"Asia/Colombo"`. |

### 6.3 Equipment — JSON blob OR flat (REQUIRED)

The service auto-detects which layout is present. The blob layout is preferred for new plants; the flat layout is supported for legacy.

#### 6.3a Preferred — single `pvlib_config` blob

| Attribute | Type | Required | Notes |
|---|---|---|---|
| `pvlib_config` | JSON | yes | Top-level keys: `plant_name`, `location {lat, lon, altitude_m, timezone}`, `orientations[]`, `module {}`, `inverter {}`, `iam {}`, `station {}`, `defaults {}`, `thermal_model`, `losses {}`, `far_shading`, `solcast_resource_id`, `weather_station_id`, `p341_device_id`. |

#### 6.3b Flat layout — each as its own attribute

| Attribute | Type | Required | Notes |
|---|---|---|---|
| `orientations` | JSON array | yes | `[{name, tilt, azimuth, module_count, use_measured_poa}]`. At least one orientation; the service inserts a default if absent. |
| `module` | JSON | yes | `{area_m2, efficiency_stc, gamma_p}`. |
| `inverter` | JSON | yes | `{ac_rating_kw, dc_threshold_kw, use_efficiency_curve, efficiency_curve_kw[], efficiency_curve_eta[], flat_efficiency}`. |
| `iam` | JSON | yes | `{angles[], values[]}`. |
| `station` | JSON | yes | See §7. |
| `defaults` | JSON | recommended | `{wind_speed_ms, air_temp_c}`. Used when station data is missing. |
| `thermal_model` | string | recommended | Default `"open_rack_glass_glass"`. SAPM model key. |

### 6.4 Loss factors (OPTIONAL — default 0.0 except `albedo` and `far_shading`)

All in fraction units; positive = loss; negative = gain (only for `module_quality`).

| Attribute | Type | Default | Notes |
|---|---|---|---|
| `soiling` | double | 0.0 | |
| `lid` | double | 0.0 | Light-induced degradation. |
| `module_quality` | double | 0.0 | Negative = gain. |
| `mismatch` | double | 0.0 | |
| `dc_wiring` | double | 0.0 | |
| `ac_wiring` | double | 0.0 | |
| `albedo` | double | 0.20 | Ground reflectivity. |
| `far_shading` | double | 1.0 | 1.0 = no shading; <1.0 = shaded. |

### 6.5 Device references (RECOMMENDED — auto-discovered if absent)

| Attribute | Type | Notes |
|---|---|---|
| `weather_station_id` | string (UUID) | TB device ID of the WSTN. If absent, the service tries (a) `Contains` relations, then (b) tenant device search by `<plant_prefix>_WSTN*` naming. |
| `p341_device_id` | string (UUID) | TB device ID of the P341 power meter. Optional; auto-discovered same way. |
| `solcast_resource_id` | string | Solcast tier-2 resource ID. If absent, Solcast tier is skipped; clear-sky tier 3 still works. |

### 6.6 Telemetry-unit declaration

| Attribute | Type | Default | Notes |
|---|---|---|---|
| `active_power_unit` | string | `"kW"` | `"W"` for plants whose meter publishes watts. The loss-rollup and the V5 Curtailment widget honour this to scale to kW. Set via `scripts/shared/set_active_power_unit.py` (already in the repo). |

### 6.7 Capacity / tariff (used by widgets, not by the physics pipeline)

| Attribute | Type | Default | Notes |
|---|---|---|---|
| `Capacity` | double | (none) | Plant nameplate capacity. |
| `capacityUnit` | string | `"kW"` | `"kW"` or `"MW"`. |
| `tariff_rate_lkr` | double | (none) | LKR/kWh. Required for the revenue mode of Loss Attribution; loss-rollup writes `-1` for revenue keys when missing. |
| `commissioning_date` | string | `2020-10-01` (fleet fallback) | ISO date `YYYY-MM-DD`. Used by the lifetime-attribute recompute path and the widget's lifetime-range builder. |

### 6.8 Loss-attribution overrides (OPTIONAL — round 1)

| Attribute | Type | Default | Notes |
|---|---|---|---|
| `loss_attribution_enabled` | boolean | `true` (when missing) | Set `false` to opt the plant out of the loss-rollup despite `pvlib_enabled = true`. |
| `setpoint_keys` | CSV string | `"setpoint_active_power, curtailment_limit, power_limit"` | Per-plant override of the default setpoint key list. |
| `actual_power_keys` | CSV string | `"active_power"` | Per-plant override of the actual-power key list. |

### 6.9 Lifetime-attribute outputs (WRITTEN by the service — do not pre-set)

`loss_grid_lifetime_kwh`, `loss_curtail_lifetime_kwh`, `loss_revenue_lifetime_lkr`, `loss_curtail_revenue_lifetime_lkr`, `potential_energy_lifetime_kwh`, `exported_energy_lifetime_kwh`, `loss_lifetime_anchor_date`, `loss_lifetime_updated_at`. These are **outputs**; setting them by hand causes the next increment step to double-count.

---

## 7. Irradiance keys (the user's other question)

The `station` JSON inside `pvlib_config` (or flat `station` attribute) declares which TB telemetry keys on the **weather-station device** carry which physical quantity. The service then reads them via `tb_client.get_timeseries(entity_type="DEVICE", entity_id=weather_station_id, keys=[…])`.

### 7.1 Keys consumed

| Field in `station` JSON | Physical quantity | Default if blank | Required for Tier-1 |
|---|---|---|---|
| `ghi_key` | Global Horizontal Irradiance, W/m² | `"ghi"` | YES |
| `poa_key` | Plane-of-Array irradiance (tilted to module plane), W/m² | `null` (none) | NO — when set together with `use_measured_poa: true` on the orientation, the service skips Perez transposition for that orientation. |
| `air_temp_key` | Ambient air temperature, °C | `"temperature"` | YES |
| `wind_speed_key` | Wind speed, m/s | `null` (none) | NO — falls back to `defaults.wind_speed_ms` (default 1.0). Used by the SAPM thermal model. |
| `freshness_minutes` | Numeric — drop tier-1 if last record older than this. | 10 | metadata, not a key |
| `sanity_max_ghi_wm2` | Numeric — clip GHI to this max. | 1400 | metadata |
| `sanity_max_poa_wm2` | Numeric — clip POA to this max. | 1500 | metadata |

### 7.2 Existing fleet examples (from `TELEMETRY_CONTRACT.md` and `kebithigollewa_pvlib_config.json`)

| Plant | `ghi_key` | `poa_key` | `air_temp_key` | `wind_speed_key` |
|---|---|---|---|---|
| KSP | `wstn1_horiz_irradiance` | `wstn1_tilted_irradiance` | `wstn1_temperature_ambient` | `wstn1_wind_speed` |

The widget tells me other plants in the fleet vary in naming (some use `ghi`, some don't have a station device at all — those fall to clear-sky tier 3).

### 7.3 New plant publishing `solar_irradiance` — what to put in `station`

The mapping depends on **how the sensor is mounted**:

- **Pyranometer mounted horizontally (most common, single-pyranometer stations)** — the reading is GHI. Use:

```json
"station": {
  "ghi_key": "solar_irradiance",
  "air_temp_key": "<whatever the station publishes>",
  "wind_speed_key": "<wind key, or omit>",
  "freshness_minutes": 10,
  "sanity_max_ghi_wm2": 1400,
  "sanity_max_poa_wm2": 1500
}
```

  Leave `poa_key` unset; the service will Perez-transpose GHI → POA per orientation. Each orientation should have `use_measured_poa: false` (the default).

- **Pyranometer mounted on the array plane (rare for 1-sensor stations)** — the reading is POA. Use:

```json
"station": {
  "ghi_key": "solar_irradiance_horizontal_synth_or_omit",
  "poa_key": "solar_irradiance",
  "air_temp_key": "...",
  "freshness_minutes": 10,
  "sanity_max_ghi_wm2": 1400,
  "sanity_max_poa_wm2": 1500
}
```

  …and set `use_measured_poa: true` on every orientation that the sensor is aligned with. If you don't know whether the sensor is on the module plane, default to the horizontal (GHI) case.

If the new station doesn't publish a temperature reading, set `air_temp_key` to a non-existing key (the service will fail tier-1 freshness and fall to Solcast / clear-sky), or set `defaults.air_temp_c` to a sensible local average and accept that thermal correction will use the default.

### 7.4 If you want to confirm before configuring

Before deploying the plant config, run the existing audit:

```
python scripts/shared/audit_tb_config.py --root-ids <root-uuid> --format json
```

This reports `ERR/WARN/MISMATCH` per plant and per attribute. After saving the new plant's `pvlib_config`, re-run; the new plant should appear with no `ERR`s for the station block.

---

## 8. Verification (must pass before you ship)

### Server side
1. `pytest scripts/Pvlib-Service/tests/test_loss_rollup.py -v` — all green, including the new today-partial tests.
2. `POST /admin/run-loss-rollup?start=2026-05-03&end=2026-05-03` (KSP) — six daily keys present at `2026-05-03T00:00 local` ts; no monthly/yearly keys; one write per plant per day.
3. `POST /admin/run-today-partial` — six daily keys present at today's local-midnight ts with `loss_data_source = "ok"` (after >30 min of daylight) or `error:insufficient_samples` (early morning). Lifetime attributes unchanged.
4. With `LOSS_ROLLUP_ENABLED=true` and `LOSS_TODAY_PARTIAL_ENABLED=true`, observe the cron log line at startup; verify no jobs registered when both are false.

### Widget side
5. Load the Grid & Losses dashboard with KSP selected and `useNewLossKeys = "auto"`. The Loss Attribution cards render once, no per-minute flicker, no skeleton storm.
6. Switch range mode in the rangeSelector card — every other card refetches once with skeleton, then settles.
7. Wait 30 min with the tab focused — cards silently refetch; no skeleton flash; values update if any precomputed key changed.
8. Hide the tab for 30 min, then reactivate — cards do one immediate silent refetch on visibility-change.
9. Set range to `Year` and load against a non-`pvlib_enabled` plant — the card shows `--` with status `PENDING` and the "not yet rolled up" tooltip; no TB request storm.
10. Open browser devtools Network tab — total TB requests/min from this dashboard ≤ 6 (one per card on load) + 6 every 30 min (silent refresh). No per-minute TB requests from the Loss Attribution widget.

### Cross-widget
11. Confirm the V5 Curtailment widget on the same tab still works — its `onDataUpdated` flow is independent and shouldn't be touched by these changes. Visual check: chart updates every minute as before for "today"; no regressions in week/month modes.

---

## 9. Open questions for Naveen (please confirm before phase L1)

- **Q1.** The 30-min today-partial cron will run during `LOSS_TODAY_PARTIAL_DAY_START_HOUR..END_HOUR` (default 05:00–19:00 local). For plants that produce outside that window (very early sunrise / late sunset depending on latitude), the value won't update during those hours. Acceptable? (Default plan: yes — Sri Lanka fleet sunrise ≈ 05:30, sunset ≈ 18:30.)
- **Q2.** The widget's default poll interval is 30 min. If you want a faster default for "Today" mode specifically, set `pollIntervalMinutes` per-card. Confirm 30 min is the right default.
- **Q3.** The "PENDING" placeholder for non-rolled-up plants on Year/Lifetime ranges is a visible UX change. Confirm wording or change it (`This plant has not been rolled up yet for the selected range.`).
- **Q4.** Do you want the today-partial values flagged differently in `loss_data_source` (e.g. `"ok:partial"` vs `"ok"`) so reports can filter them out? (Default plan: yes — set `data_source = "ok:partial"` inside `run_today_partial_rollup` for plant rows and `"rollup:partial"` for ancestors.)

---

## 10. File / line index for the implementer

| Concern | File | Where |
|---|---|---|
| New today-partial entry point | `app/services/loss_rollup_job.py` | new function `run_today_partial_rollup` near `run_loss_rollup` |
| Reusable per-plant integration | `app/services/loss_rollup_job.py::_process_plant` | adjust to take `min_samples_override` |
| Cron registration | `app/services/scheduler.py::start_scheduler` | new `add_job` for `pvlib_loss_today_partial` |
| Today-partial trigger function | `app/services/scheduler.py` | new `run_today_partial_now` mirroring `run_loss_rollup_now` |
| New settings | `app/config.py` | append `LOSS_TODAY_PARTIAL_*` block |
| New endpoint | `app/api/forecast.py` | new `/admin/run-today-partial` mirroring `/admin/run-daily` |
| Revert monthly/yearly | `app/services/loss_rollup_job.py` | drop `KEY_POTENTIAL_MONTHLY`, `KEY_POTENTIAL_YEARLY`, `_get_historical_sum`, the duplicate `_safe_write_daily` call in `run_loss_rollup` (≈ lines 172–214) |
| Tariff snapshot doc | `TELEMETRY_CONTRACT.md` | append `loss_tariff_rate_lkr_at_compute` row |
| Widget refresh policy | `Loss Attribution/.js::onInit, onDataUpdated, onDestroy` | add timer + visibility handler; remove the listener-add from `updateDom` |
| Widget skeleton gating | `Loss Attribution/.js::renderComputedMode` | accept `{silent}`; skip `setLoadingState` on silent |
| Widget attr cache | `Loss Attribution/.js::fetchCalculationAttributes` | add 30-min TTL cache |
| Widget pending placeholder | `Loss Attribution/.js` | new `showNotRolledUpPlaceholder` |
| Widget fallback gate | `Loss Attribution/.js::calcForRange` | add `legacyFallbackAllowed` |
| Widget settings | `Loss Attribution/settings.json` | append `pollIntervalMinutes`, `attrCacheMinutes` |

End of plan.
