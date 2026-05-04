# Loss Attribution Telemetry Plan
*Pvlib-Service ⇄ Loss Attribution Widget*
*Author: planning pass — 2026-05-04*
*Target executor: Sonnet 4.6*

---

## 0. Why this document exists

The Loss Attribution widget at `M:\Documents\Projects\MAGICBIT\Widgets\Grid & Losses\Loss Attribution` currently downloads minute-cadence `active_power` and `potential_power` for every selected range (day / month / year / lifetime / custom) **plus the comparator range** for the delta footer, then performs per-bucket arithmetic in the browser. For year and lifetime ranges this is hundreds of thousands of points per card, four cards on a typical dashboard, and a comparator on each — the dashboard hangs.

The fix is to move the per-bucket integration into Pvlib-Service (which already discovers plants from ThingsBoard, reads each plant's config from server-scope attributes, has a 1-min cycle and a 00:05-local daily roll-up), have it write daily aggregated loss telemetry plus a cumulative-lifetime attribute, and have the widget read those instead of recomputing on the client.

Plant-specific behaviour MUST NOT be hardcoded inside the service. Plants are discovered by BFS from `TB_ROOT_ASSET_IDS` and filtered by `pvlib_enabled = true`; everything plant-specific (capacity, tariff, setpoint key list, active_power unit) is read from each plant's TB attributes. Slight hardcoding inside the widget (default key names, default mode mapping) is acceptable.

---

## 1. Current state — facts

### 1.1 Pvlib-Service (already in production)

- `app/services/forecast_service.py` writes per-minute keys: `potential_power` (kW), `active_power_pvlib_kw` (kW alias), `pvlib_data_source`, `pvlib_model_version`, `ops_expected_unit`. Sentinel = `-1` on failure.
- `app/services/daily_job.py::run_daily_rollup` already integrates `potential_power` over the previous local-tz day, writes `total_generation_expected_kwh` and `pvlib_daily_energy_kwh`, and rolls up to all `isPlantAgg` ancestors. Cron 00:05 local in `app/services/scheduler.py`. `MIN_VALID_SAMPLES = 360`. Sentinel `-1` on insufficient samples.
- Plant discovery: `app/services/thingsboard_client.py::discover_plants` (5-min TTL cache, BFS from root assets, filter `isPlant ∧ pvlib_enabled`, dedup-aware ancestor map).
- TB I/O: `get_timeseries`, `post_telemetry`, `get_asset_attributes` (SERVER_SCOPE).
- Settings: `app/config.py` (env-driven; no plant list inside).
- Telemetry contract: `TELEMETRY_CONTRACT.md` — additions to that file are in §6.
- Active-power unit: per-plant `active_power_unit` attribute = `"kW"` or `"W"` (default `"kW"`); fleet split documented in `TELEMETRY_CONTRACT.md` §"Plant unit map". Migration script: `scripts/shared/set_active_power_unit.py`.

### 1.2 Loss Attribution widget

- Modes (`settings.json`): `grid`, `curtail`, `revenue`, `curtailRevenue`, `insurance`, `rangeSelector`. The `curtailRevenue` mode is in `settings.json` and `.js` but NOT in `README.md` — README is stale on that single mode.
- Computed metrics (per range, in browser):
  - `grid` → `Σ max(potential − active, 0) · Δh` [kWh]
  - `curtail` → `Σ max(potential − max(ceiling, active), 0) · Δh` when `setpoint_pct < 99.5`, where `ceiling = capacity · setpoint_pct/100`
  - `revenue` → `grid_kwh · tariff_rate_lkr`
  - `curtailRevenue` → `curtail_kwh · tariff_rate_lkr`
  - `insurance` → latest-value passthrough (no compute)
  - `rangeSelector` → no compute, writes `LossAttributionRange` dashboard state
- Per render the widget runs `calculateLossForRange()` for the primary range AND a comparator range (previous day/month/year, or preceding span for custom) to drive the delta footer. The comparator doubles every cost.
- Power telemetry uses TB `agg=AVG` only when bucket count ≤ 720; otherwise raw chunked + client bucket-average. Setpoint is always raw with 30-day lookback for step-hold.
- Capacity attribute key = `Capacity` (kW or MW per `capacityUnit` setting). Tariff attribute = `tariff_rate_lkr` (SERVER_SCOPE first, then SHARED_SCOPE).
- Lifetime start = `commissioning_date` attribute, fallback `lifetimeStartDate` setting (`2020-10-01`).
- Day range uses solar window 05:00–19:00 local (`SOLAR_DAY_START_HOUR`/`SOLAR_DAY_END_HOUR`).

---

## 2. Problems and approaches

For each problem I list the candidate approaches, pros/cons, and a 1–10 score (≥ 90 % confidence). When no single approach is a clean winner I propose a hybrid that combines the wins and avoids the losses.

### P1. Where to compute the per-bucket loss math

| # | Approach | Pros | Cons | Score |
|---|---|---|---|---|
| A1.1 | Keep client-side compute | Zero backend change | Status quo dashboard hang for year/lifetime; comparator doubles cost | 2 |
| A1.2 | **Pvlib-Service pre-aggregates** | Re-uses BFS discovery, attribute reader, retry, TB client, scheduler; fully generic; one job to test | Requires new code + backfill | 9 |
| A1.3 | TB rule chain | Visual; lives in TB | Hard to express integration over a window across two keys; brittle; hard to test | 3 |
| A1.4 | Custom TB plugin | Native | Heavy build; fleet doesn't operate this way today | 3 |
| A1.5 | External TSDB (Timescale/Influx) joined ad-hoc | Powerful query layer | New service; new auth surface; out of scope | 4 |

**Decision: A1.2.** Existing `daily_job.py` is the precedent — same shape, different formula.

### P2. Aggregation horizons

| # | Approach | Pros | Cons | Score |
|---|---|---|---|---|
| A2.1 | Daily only; widget sums over range | Simplest; daily payload tiny (~365 rows/year) | Lifetime over multi-year reads ~1 800 rows | 7 |
| A2.2 | Daily + monthly + yearly precomputed | Smallest possible widget query | Three writes; backfill complex; double-counting risk | 6 |
| A2.3 | Daily + cumulative-lifetime attribute | Lifetime is constant cost (one attribute read); month/year stay small (~30/365 daily rows) | Maintenance for cumulative correctness on backfills | 8 |
| A2.4 | All four horizons | Minimum query cost everywhere | Maximum maintenance cost | 6 |
| A2.5 | Add hour-of-day key | Sub-day detail | `potential_power` per-minute already covers it | 5 |

**Decision (hybrid H2): A2.3 + per-minute "today" path stays on the client.**
- Today (current day + range mode = `day`) → widget keeps using per-minute fetch (existing `calculateLossForRange`). One day = ≤ 1 440 rows; already fast.
- Past day → one daily-key row.
- Month / Year (current or past) → daily rows (≤ 31 / ≤ 366) summed by widget (or TB `agg=SUM` if practical).
- Lifetime → one server-scope attribute read.
- Custom → daily rows summed by widget; if range crosses today, cap at last-completed day and add a per-minute "today partial" on top.

Score H2: 9.

### P3. Telemetry key shape

| # | Approach | Pros | Cons | Score |
|---|---|---|---|---|
| A3.1 | Flat scalar keys per metric | TB-native; allows `agg=SUM`; simple widget reads | Six new keys per plant | 8 |
| A3.2 | Single JSON blob per day | One write, one read | Cannot use TB `agg=SUM`; widget must `JSON.parse` every record; useless for long ranges | 4 |
| A3.3 | Schemaful keys with horizon suffix (e.g. `loss_grid_daily_kwh`) | Same as A3.1 + clear naming | Slightly more verbose | 9 |

**Decision: A3.3.** See §6 for final names.

### P4. Plant generality (no hardcoding)

| # | Approach | Pros | Cons | Score |
|---|---|---|---|---|
| A4.1 | Reuse existing `pvlib_enabled = true` gate | Zero new attributes; one switch | Cannot disable losses without disabling pvlib | 8 |
| A4.2 | New `loss_attribution_enabled` flag | Independent control | Yet another flag; risk of drift | 7 |
| A4.3 | Trigger by tariff presence | Auto-on when revenue is wanted | Conflates "want this metric" with "have a tariff" | 5 |

**Decision (hybrid H4): A4.1 default, A4.2 optional override.** Default behaviour: any plant with `pvlib_enabled = true` is included; if `loss_attribution_enabled` is explicitly set to `false` the plant is skipped (for cases where loss math is unwanted on a sub-meter plant). Score 9.

### P5. Active-power unit normalisation

| # | Approach | Pros | Cons | Score |
|---|---|---|---|---|
| A5.1 | Read `active_power_unit` per plant; scale W → kW server-side | Generic; matches existing widget recipe; cheap (one extra attr already in cache) | One extra attribute per plant | 9 |
| A5.2 | Require a one-shot kW migration first | Cleanest contract long-term | Op effort; risk of regressions during cut-over | 6 |
| A5.3 | Hardcode the W-publishing plant list | Fast | **Forbidden** by user constraint | 1 |

**Decision: A5.1.**

### P6. Revenue — historical truth vs hypothetical recompute

The tariff today differs from the tariff that was in force last year. The widget today applies the *current* tariff to historical kWh; that's wrong for accounting reports.

| # | Approach | Pros | Cons | Score |
|---|---|---|---|---|
| A6.1 | Pre-compute `loss_revenue_daily_lkr` using tariff at time of write | Historical-accurate; widget reads one number | If a backfill runs after a tariff change, the old number is rebuilt with current tariff (good or bad depending on intent) | 8 |
| A6.2 | Widget multiplies kWh × current tariff each render | Always reflects current tariff; smaller server burden | Not historically accurate; tariff lookup latency on every render | 6 |
| A6.3 | Pre-compute LKR + keep raw kWh | Both historical truth and hypothetical recompute possible | Slight storage overhead; doc burden | 9 |

**Decision (hybrid H6): A6.3.** Service writes BOTH `loss_grid_daily_kwh` AND `loss_revenue_daily_lkr` using the `tariff_rate_lkr` attribute that was current at compute time. Widget defaults to reading the LKR key but can fall back to `kwh × current_tariff` if the LKR key is missing or zero (e.g., new plant with no tariff yet). Same pattern for `curtail` / `curtailRevenue`.

### P7. Lifetime cumulative

| # | Approach | Pros | Cons | Score |
|---|---|---|---|---|
| A7.1 | Widget sums daily over full lifetime each render | No new server keys | 1 800+ rows for 5-year-old plants × 4 cards × 2 (comparator) — slow but tolerable | 5 |
| A7.2 | Maintain server-scope cumulative attributes | One latest-value lookup per card | Backfill correctness needs an anchor-date counter | 8 |
| A7.3 | TB `agg=SUM` over the whole window per render | No new key | TB `agg` over years is expensive; round-trip on every render | 6 |

**Decision (hybrid H7): A7.2 + idempotent recompute path.** Maintain six server-scope attributes (`loss_*_lifetime_*`) plus `loss_lifetime_anchor_date` (the last calendar day already added). Daily job: if `today_day_ts > anchor_date`, add today's daily values and advance the anchor; else (re-run / backfill) recompute fully from history. Add `/admin/recompute-lifetime` endpoint.

### P8. Backfill

| # | Approach | Pros | Cons | Score |
|---|---|---|---|---|
| A8.1 | New `/admin/run-loss-rollup?start&end` endpoint | Mirrors existing `/admin/run-daily` ergonomics | New endpoint to maintain | 9 |
| A8.2 | Re-use `/admin/run-daily` to also do losses | One endpoint | Couples two jobs; harder to backfill losses without redoing energy | 6 |
| A8.3 | One-shot script outside the service | Clean separation | New code path with its own auth | 6 |

**Decision (hybrid H8): A8.1 for date-range backfill + a small one-day fast path on the same endpoint.** Plus `/admin/recompute-lifetime`.

### P9. Same-day "today" partial

| # | Approach | Pros | Cons | Score |
|---|---|---|---|---|
| A9.1 | Service writes today's partial loss every minute | Always fresh on the dashboard | Wasteful integration over partial data; fleet-wide burst every minute | 5 |
| A9.2 | Service writes today's partial every 10 min | Fresh-ish; cheap | Up to 10 min stale | 7 |
| A9.3 | Widget keeps per-minute fetch for today | Zero new server-side cost; today is cheap (≤ 1 440 minutes) | Two code paths in widget (one minor branch) | 9 |

**Decision: A9.3.** Widget detects "range mode = day AND start.date == today.date" and stays on the existing per-minute math. Everything else uses the precomputed daily keys. (Optionally also A9.2 can be added later if a separate dashboard wants today's partial without doing the per-minute math itself; not required for this task.)

### P10. Comparator ranges (delta footer)

No new design — once daily keys exist, both primary and comparator are tiny reads. The widget logic in `getComparatorRange` stays as-is.

### P11. TB `agg=SUM` vs widget client-side sum

For month / year ranges over daily keys, both approaches read 30 / 365 rows. Widget client-side sum is **simpler and avoids edge cases with TB's interval alignment**, so prefer client sum. (TB `agg=SUM` requires interval=range and may bucket awkwardly across DST boundaries; not worth the complexity.)

### P12. Storage / cardinality

Per-plant per-day: 6 scalar keys + 2 string keys (`loss_data_source`, `loss_model_version`). For 1 000 plants × 6 keys × 365 days ≈ 2.19 M points/year — small. Lifetime attributes are 7 strings/numbers per plant.

### P13. Roll-up to ancestor aggregations (fleet/regional totals)

Reuse `_rollup_parents` pattern from `forecast_service.py`. Sum the daily kWh / LKR values of children (skip `-1` sentinels and skip plants with status ≠ ok) and post to ancestors at the same `day_ts_ms`. Lifetime attributes also roll up — sum each child's lifetime attribute into the parent's lifetime attribute (write SERVER_SCOPE).

### P14. Setpoint state-hold across day boundaries

The setpoint is sparse — last value before `start` defines the state for the morning. Pull setpoint records over `[day_start − 30 d, day_end]` to seed the step-hold. Mirror existing widget behaviour. Reuse the configured setpoint key list (server reads from a per-plant attribute `setpoint_keys` if present, otherwise the contract default `setpoint_active_power, curtailment_limit, power_limit`).

### P15. Sentinels and integrity

- Skip records where `potential` or `active` < 0.
- If `valid_minutes < MIN_VALID_SAMPLES = 360`, write `-1` for every loss key for that day (matches `daily_job.py`).
- Set `loss_data_source = "error:insufficient_samples"` (or `"error:no_potential"`, `"error:no_actual"`, `"error:integration_failed"`).
- Lifetime attribute is updated only when today's daily values are non-sentinel.

### P16. Widget changes summary

1. Read order, per mode (computed mode only):
   1. If `range.mode == day` AND date is today and current day → existing per-minute path (no change).
   2. Else if `range.mode == lifetime` → read the lifetime attribute (one read).
   3. Else → fetch the daily key(s) over `[startTs, endTs]` and sum client-side. Comparator goes through the same path.
   4. If the daily key returns nothing OR if any read fails → fall back to existing per-bucket fetch (current behaviour). This preserves backwards compatibility for plants the service hasn't yet covered.
   5. Final fallback (existing): `DS[0]` latest-value.
2. Add two settings to `settings.json`: precomputed-key prefix override (single setting) and a "use precomputed" toggle (default `true`). Six derived key names follow a fixed default pattern; the user can override the prefix only — slight hardcoding here is fine.
3. Comparator path uses precomputed keys too — the comparator delta becomes near-free.
4. `curtailRevenue` mode: add to README so documentation matches code.

### P17. Migration / rollout

- Phase L0 — Service code + tests landed; flag `LOSS_ROLLUP_ENABLED` env defaults `false`. Daily job stays a no-op until flipped.
- Phase L1 — Flip on for KSP only (Gate 0 of this plan); 7-day soak.
- Phase L2 — `/admin/run-loss-rollup` backfill for KSP from `commissioning_date` to today; verify lifetime attribute matches client-summed daily history within ±0.1 %.
- Phase L3 — Widget shipped with `useNewKeys = true` and graceful fallback. Verify both paths agree on KSP for the same range.
- Phase L4 — Enable on SOU + SSK + remaining `pvlib_enabled` plants; backfill each.

### P18. Where the new code goes

New / changed files:
- `scripts/Pvlib-Service/app/services/loss_rollup_job.py` — new module (mirror of `daily_job.py`).
- `scripts/Pvlib-Service/app/services/scheduler.py` — register the cron at 00:10 local + interval-style retry on failure.
- `scripts/Pvlib-Service/app/api/forecast.py` — three new endpoints: `/admin/run-loss-rollup`, `/admin/recompute-lifetime`, optionally `/admin/loss-status`.
- `scripts/Pvlib-Service/app/config.py` — three new settings: `LOSS_ROLLUP_ENABLED`, `LOSS_DEFAULT_SETPOINT_KEYS`, `LOSS_MIN_VALID_SAMPLES` (default 360).
- `scripts/Pvlib-Service/TELEMETRY_CONTRACT.md` — append the new keys + lifetime attributes (§6 of this plan is ready-to-paste).
- `scripts/Pvlib-Service/tests/test_loss_rollup.py` — new pytest covering integration math, sentinel handling, ancestor roll-up.
- `M:\Documents\Projects\MAGICBIT\Widgets\Grid & Losses\Loss Attribution\.js` — new branch in `calculateLossForRange` and `renderComputedMode`.
- `M:\Documents\Projects\MAGICBIT\Widgets\Grid & Losses\Loss Attribution\settings.json` — new settings: precomputed-key prefix, `useNewKeys` toggle, lifetime attribute prefix.
- `M:\Documents\Projects\MAGICBIT\Widgets\Grid & Losses\Loss Attribution\README.md` — add `curtailRevenue` row, document the new fast path and fallback.

### P19. Sonnet constraint reminders

- **Service: zero plant-specific constants.** All plant-derived inputs (capacity, tariff, setpoint key list, lifetime start, active_power_unit) are read from each plant's TB SERVER_SCOPE attributes via the existing `discover_plants` + `get_asset_attributes` calls. Adding a constant like `KSP_TARIFF` anywhere in `app/` is forbidden.
- **Widget: slight hardcoding allowed.** Default key names and default mode mapping are fine; expose them via `settings.json` so an operator can override without re-uploading the widget code.

---

## 3. Hybrid summary (the bundle that goes live)

Precomputation (Pvlib-Service):
- **H2** — Daily aggregates + cumulative-lifetime attribute (no monthly/yearly precompute).
- **H4** — Inclusion gate = `pvlib_enabled = true` AND (`loss_attribution_enabled` ∉ {false, "false", 0}).
- **H6** — Write both kWh and LKR daily keys (LKR uses tariff at compute time).
- **H7** — Lifetime cumulative attribute with anchor date + `/admin/recompute-lifetime`.
- **H8** — Date-range backfill endpoint.

Widget:
- **A9.3** — Today current-day uses existing per-minute path; everything else uses precomputed.
- Graceful fallback chain: precomputed → per-minute compute → DS[0] latest-value.

Each individual sub-decision was rated ≥ 8/10 with ≥ 90 % confidence; the bundle's worst-case failure mode (precomputed key missing) is covered by the existing fallback path.

---

## 4. Step-by-step server changes

### 4.1 New module: `app/services/loss_rollup_job.py`

Public entry: `async def run_loss_rollup(tb_client, date: Optional[datetime] = None) -> Dict[str, object]`

Behaviour mirroring `daily_job.run_daily_rollup`, with these differences:

1. Determine `[day_start_local, day_end_local]` — same as `daily_job` (calendar day in `settings.TZ_LOCAL` that just ended; backfill via `date` arg).
2. `plants, ancestor_map = await tb_client.discover_plants(settings.root_asset_ids)` (re-uses 5-min cache).
3. For each plant (concurrency-bounded by `settings.MAX_CONCURRENT_PLANTS`):
   a. `attrs = await tb_client.get_asset_attributes(plant.id)` — read tariff, capacity, capacity unit, active_power_unit, setpoint key list, plus the per-plant override `loss_attribution_enabled` if set.
   b. Skip plant if `_truthy(attrs.get("loss_attribution_enabled", True)) is False`.
   c. Pull three series for `[day_start_utc, day_end_utc]` (one TB call per series — they live on different keys; do them concurrently with `asyncio.gather`):
      - `potential_power` (kW, already normalised by service)
      - The first key in `actualPowerKeys` that has data — defaults to `active_power`. (Try them in order, like the widget does.)
      - All `setpointKeys` (default contract: `setpoint_active_power, curtailment_limit, power_limit`) for `[day_start_utc − 30 d, day_end_utc]` (raw, for step-hold seed).
   d. Drop `< 0` records (sentinels). If `len(potential) < MIN_VALID_SAMPLES` OR `len(actual) < MIN_VALID_SAMPLES`, integration is invalid — write all loss keys as `-1` with `loss_data_source = "error:insufficient_samples"` and continue.
   e. Apply `active_power_unit` scaling: if `attrs.get("active_power_unit") == "W"`, multiply actual values by `0.001`.
   f. Resample / align series to a 1-minute index over `[day_start_utc, day_end_utc]` using `pd.Series(...).resample("1T").mean()` (drops sentinels naturally). Forward-fill `setpoint` step-hold up to each minute.
   g. Compute per-minute:
      - `gross_loss_kw = max(potential − active, 0)`
      - `ceiling_kw = capacity_kw · setpoint_pct/100` for the active setpoint at that minute (default `100` if no setpoint record yet)
      - `curtail_loss_kw = max(potential − max(ceiling_kw, active), 0)` if `setpoint_pct < 99.5` else `0`
   h. Integrate: each per-minute value contributes `value · (1/60)` kWh.
   i. Compute scalars:
      - `loss_grid_daily_kwh = Σ gross_loss_kwh`
      - `loss_curtail_daily_kwh = Σ curtail_loss_kwh`
      - `potential_energy_daily_kwh = Σ potential · (1/60)`
      - `exported_energy_daily_kwh = Σ active · (1/60)` (after unit scaling)
      - `loss_revenue_daily_lkr = loss_grid_daily_kwh · tariff` (only if tariff is finite; else write `-1` for the LKR key with `loss_data_source = "warn:no_tariff"`)
      - `loss_curtail_revenue_daily_lkr` analogous
   j. Write all keys at `day_ts_ms = local_midnight(day_start)` via `tb_client.post_telemetry`.
4. Ancestor roll-up: same dedup-aware ancestor traversal as `_rollup_parents`. Sum the per-plant daily values; post under the same six keys at the same `day_ts_ms`.
5. Lifetime update (per plant + per ancestor): see §4.2.
6. Return summary `{plants_ok, plants_failed, plants_skipped}` plus per-plant breakdown.

### 4.2 Lifetime attribute maintenance

Server-scope attributes per plant:

| Attribute | Type | Notes |
|---|---|---|
| `loss_grid_lifetime_kwh` | double | Cumulative gross loss kWh |
| `loss_curtail_lifetime_kwh` | double | Cumulative curtailment loss kWh |
| `loss_revenue_lifetime_lkr` | double | Cumulative revenue loss LKR (sum of historical daily LKR values, each computed at the tariff in effect that day) |
| `loss_curtail_revenue_lifetime_lkr` | double | Same for curtailment |
| `potential_energy_lifetime_kwh` | double | Cumulative potential kWh (denominator for loss-rate displays) |
| `exported_energy_lifetime_kwh` | double | Cumulative exported kWh |
| `loss_lifetime_anchor_date` | string | ISO date of the latest day already added (e.g. `"2026-05-03"`) |
| `loss_lifetime_updated_at` | string | ISO datetime of last successful update |

Update logic (in `loss_rollup_job.py`):

```
if today_day == anchor_date_plus_one and all six daily values are finite (≥ 0):
    new_value = old_attribute + today_value
    write attribute, set anchor_date = today_day
elif today_day > anchor_date_plus_one:
    log gap, fall through to recompute path (call _recompute_lifetime_from_history for this plant)
else:
    # backfill / re-run for an already-included date — recompute fully
    call _recompute_lifetime_from_history
```

Endpoint `/admin/recompute-lifetime?asset_id=...` (or fleet-wide if omitted): for each plant, read entire history of the six daily keys via `tb_client.get_timeseries` for `[commissioning_date, today]` paged in 90-day chunks, sum each key, write the six lifetime attributes plus refresh `loss_lifetime_anchor_date` and `loss_lifetime_updated_at`. Use `tb_client.post_attributes` (add this if it doesn't exist; the TB endpoint is `POST /api/plugins/telemetry/{entityType}/{entityId}/SERVER_SCOPE`).

For ancestor lifetime totals, write the same six attributes on each isPlantAgg ancestor as `Σ(child lifetime attrs)`. Read children's existing lifetime attributes (skip child if attribute missing or `-1`).

### 4.3 Scheduler wiring (`app/services/scheduler.py`)

Add a third APScheduler cron job:

```python
scheduler.add_job(
    run_loss_rollup_now,
    trigger="cron",
    hour=0,
    minute=10,                # 5 minutes after the daily energy job at 00:05
    misfire_grace_time=3600,
    max_instances=1,
    id="pvlib_loss_rollup",
    replace_existing=True,
)
```

Add `run_loss_rollup_now(date: Optional[datetime] = None)` mirroring `run_daily_now`. Gate on `settings.LOSS_ROLLUP_ENABLED` — return `{"status": "disabled"}` early if false.

### 4.4 New API endpoints (`app/api/forecast.py`)

- `POST /admin/run-loss-rollup?start=YYYY-MM-DD&end=YYYY-MM-DD` — backfill range; default = yesterday only. Returns `{status, days, plants_ok, plants_failed}`.
- `POST /admin/recompute-lifetime?asset_id=<uuid>` — recompute lifetime attributes; fleet-wide if `asset_id` omitted.
- `GET /admin/loss-status?asset_id=<uuid>` — return latest values for the six daily keys + the eight lifetime attributes for sanity checks during rollout.

### 4.5 New settings (`app/config.py`)

```python
LOSS_ROLLUP_ENABLED: bool = False
"""Master flag for the daily loss-rollup job. Default false until Phase L1."""

LOSS_DEFAULT_SETPOINT_KEYS: str = "setpoint_active_power,curtailment_limit,power_limit"
"""Comma-separated default setpoint keys; per-plant 'setpoint_keys' attribute overrides."""

LOSS_MIN_VALID_SAMPLES: int = 360
"""Minimum 1-min samples per day for a real loss value (else write -1 sentinel)."""

LOSS_LIFETIME_PAGE_DAYS: int = 90
"""Page size when paging history during /admin/recompute-lifetime."""
```

### 4.6 Telemetry contract additions (paste-ready for `TELEMETRY_CONTRACT.md`)

| Key | Type | Unit | Cadence | Description | Widget use |
|---|---|---|---|---|---|
| `loss_grid_daily_kwh` | timeseries | kWh | daily, ts = local midnight | Σ max(potential − active, 0) over the calendar day. Sentinel `-1`. | Loss Attribution `grid` mode |
| `loss_curtail_daily_kwh` | timeseries | kWh | daily | Σ curtailment formula. Sentinel `-1`. | `curtail` mode |
| `loss_revenue_daily_lkr` | timeseries | LKR | daily | `loss_grid_daily_kwh × tariff_rate_lkr` at compute time. `-1` if tariff missing. | `revenue` mode |
| `loss_curtail_revenue_daily_lkr` | timeseries | LKR | daily | `loss_curtail_daily_kwh × tariff_rate_lkr`. | `curtailRevenue` mode |
| `potential_energy_daily_kwh` | timeseries | kWh | daily | Σ potential. Denominator for loss rate / delta footer. | All modes (delta) |
| `exported_energy_daily_kwh` | timeseries | kWh | daily | Σ active (after unit scaling). | Delta + diagnostics |
| `loss_data_source` | timeseries | string | daily | `"ok" / "error:insufficient_samples" / "warn:no_tariff" / "rollup"` | Diagnostics |
| `loss_model_version` | timeseries | string | daily | `"loss-rollup-v1"` | Regression detection |

Server-scope attributes (per plant + per ancestor):

| Attribute | Type | Notes |
|---|---|---|
| `loss_grid_lifetime_kwh` | double | Cumulative since `commissioning_date` |
| `loss_curtail_lifetime_kwh` | double | |
| `loss_revenue_lifetime_lkr` | double | Sum of historical daily LKR values |
| `loss_curtail_revenue_lifetime_lkr` | double | |
| `potential_energy_lifetime_kwh` | double | |
| `exported_energy_lifetime_kwh` | double | |
| `loss_lifetime_anchor_date` | string | ISO date of latest day added |
| `loss_lifetime_updated_at` | string | ISO datetime |

---

## 5. Step-by-step widget changes

### 5.1 New settings entries (`settings.json`)

Append these after the existing `lifetimeStartDate`:

```json
{
  "id": "useNewLossKeys",
  "name": "Use precomputed loss keys",
  "type": "select",
  "default": "auto",
  "items": [
    { "value": "auto",  "label": "Auto (use if available, fall back otherwise)" },
    { "value": "force", "label": "Always use precomputed (no fallback)" },
    { "value": "off",   "label": "Disable — always compute on the client (legacy)" }
  ],
  "helpText": "Reads the daily/lifetime keys written by the Pvlib loss-rollup job."
},
{
  "id": "lossDailyGridKey",
  "name": "Daily Grid Loss Key (kWh)",
  "type": "text",
  "default": "loss_grid_daily_kwh"
},
{
  "id": "lossDailyCurtailKey",
  "name": "Daily Curtailment Loss Key (kWh)",
  "type": "text",
  "default": "loss_curtail_daily_kwh"
},
{
  "id": "lossDailyRevenueKey",
  "name": "Daily Revenue Loss Key (LKR)",
  "type": "text",
  "default": "loss_revenue_daily_lkr"
},
{
  "id": "lossDailyCurtailRevenueKey",
  "name": "Daily Curtailment Revenue Loss Key (LKR)",
  "type": "text",
  "default": "loss_curtail_revenue_daily_lkr"
},
{
  "id": "lossDailyPotentialKey",
  "name": "Daily Potential Energy Key (kWh)",
  "type": "text",
  "default": "potential_energy_daily_kwh"
},
{
  "id": "lossDailyExportedKey",
  "name": "Daily Exported Energy Key (kWh)",
  "type": "text",
  "default": "exported_energy_daily_kwh"
},
{
  "id": "lossLifetimeAttrPrefix",
  "name": "Lifetime Attribute Prefix",
  "type": "text",
  "default": "loss_",
  "helpText": "Prefix used to compose the six lifetime attribute names."
}
```

### 5.2 New widget code (`.js`)

Add a new function `calculateLossForRangePrecomputed(entity, range, attrs)` that:
1. If `range.mode === 'lifetime'`: read seven server-scope attributes (the six lifetime values + `loss_lifetime_anchor_date`). Build a result object with the same shape as `calculateLossForRange` returns: `{ ok, grossLossKWh, curtailLossKWh, potentialEnergyKWh, exportedEnergyKWh, bucketMs }`. For lifetime, `bucketMs` is irrelevant; set to `0`.
2. Else: fetch the four daily keys (`grid`, `curtail`, `potential`, `exported`) over `[range.startTs, range.endTs]` using existing `fetchTimeseriesChunked` with `useAgg=false` (raw — daily keys are already aggregated). Sum each series, dropping any value `< 0` (sentinels).
3. For revenue / curtailRevenue: also fetch `loss_revenue_daily_lkr` / `loss_curtail_revenue_daily_lkr` and sum, dropping `< 0`. If sum is positive, use it directly; else fall back to `kwh × current_tariff` using `attrs.tariffRate`.
4. Return `{ ok: hasPotential, grossLossKWh, curtailLossKWh, potentialEnergyKWh, exportedEnergyKWh, revenueLossLkr, curtailRevenueLossLkr, bucketMs: 0 }`.

Modify `renderComputedMode`:
- Decide path based on `s.useNewLossKeys`:
  - `"off"` → existing `calculateLossForRange` (current behaviour).
  - `"force"` → only `calculateLossForRangePrecomputed`; if it fails, render placeholders (don't fall back).
  - `"auto"` (default):
    - If `range.mode === 'day'` AND start.date is today and current day → existing per-minute `calculateLossForRange` (cheap).
    - Else → try `calculateLossForRangePrecomputed` first; if `result.ok` is false OR any series is empty, fall back to existing `calculateLossForRange`.
- The comparator path uses the same selector. Comparator goes through precomputed when the precomputed path was used for the primary range.

Modify `renderComputedResult` — no change needed; the result shape is the same.

For revenue / curtailRevenue, when the LKR key is read directly from precomputed values, `result.revenueLossLkr` / `result.curtailRevenueLossLkr` is already in LKR — `renderComputedResult` should prefer it over `kwh × tariff` if present.

### 5.3 README addition

- Add a row for `curtailRevenue` (currently missing).
- Add a "Server-side aggregation" section explaining the new fast path, the fallback, and how to disable.

---

## 6. Verification checklist (both for the implementer and the production reviewer)

After server changes:
1. `pytest scripts/Pvlib-Service/tests/test_loss_rollup.py` passes (covers: synthetic 1-day series → expected kWh; sentinel handling; ancestor sum; tariff-missing path; W-unit scaling).
2. `POST /admin/run-loss-rollup` with `start=end=YYYY-MM-DD` for KSP — read back the six daily keys at the local-midnight ts; values > 0; non-negative.
3. `POST /admin/recompute-lifetime?asset_id=<KSP>` — read back the eight attributes; `loss_grid_lifetime_kwh ≈ Σ daily_grid` to within 0.1 %.
4. Run the daily cron once at 00:10 local (or `/admin/run-loss-rollup` for yesterday). The lifetime anchor advances by exactly one day.
5. `/admin/loss-status` for KSP returns reasonable values.
6. `/metrics` continues to expose existing counters — no regressions.

After widget changes:
7. With KSP plant on a dashboard, switch the range selector through Day (today), Day (yesterday), Month (current), Month (previous), Year (current), Lifetime, Custom (last 7 days). Each card renders < 2 s; no `--` flicker.
8. Toggle `useNewLossKeys` to `"off"` — confirm the legacy path still works on the same plant and the numbers match within 1 % (small drift is normal because precomputed buckets are 1-min and legacy widget bucket varies).
9. On a plant where the rollup hasn't run yet, with `useNewLossKeys = "auto"`, the widget falls back to legacy path silently.
10. Revenue card: with `tariff_rate_lkr` set on the plant, value matches `daily_grid_kwh × tariff` for the comparator window.

---

## 7. Things explicitly out of scope (don't do these)

- **Don't rewrite `daily_job.py`.** Keep it intact; the new job is a separate module.
- **Don't backfill `potential_power`** as part of this work — the existing `potential_power` series is the input, not the output. Only the six new daily keys are new.
- **Don't alter the V5 Curtailment widget** or `Forecast vs Actual Energy` widget — they read different keys and are out of scope.
- **Don't add monthly / yearly precomputed keys** in this iteration — daily + lifetime is the agreed bundle.
- **Don't write a per-minute "today partial"** in this iteration — the widget keeps its per-minute path for current day.
- **Don't introduce a new database** or external storage. All state lives in TB.
- **Don't store any plant ID, plant name, capacity, or tariff** as a constant inside `app/`. Read everything from TB attributes.

---

## 8. Open questions to confirm before/while building (Sonnet should ask Naveen if blocked)

1. The widget's `solar day window` for `mode=day` is 05:00–19:00 local. The service's daily roll-up is calendar-day (00:00–24:00). For `mode=day` "today" the widget keeps the solar window (per-minute path). For past `mode=day` the daily key is for the full calendar day. **This is a deliberate small mismatch** — the displayed value for "yesterday" will include any pre-05:00 / post-19:00 anomalies that the live "today" view would hide. Confirm this is acceptable; if not, the fix is to also write a `loss_grid_solar_daily_kwh` keyed to the solar window. **Default plan: accept the calendar-day version.**
2. Tariff change history is not preserved by this design (we capture the tariff in effect on the day of compute, but if the same date is later recomputed, the new tariff applies). If accounting needs frozen historical revenue, add a `tariff_rate_lkr_at_compute` mirror key in a follow-up; not included here.
3. Should `loss_attribution_enabled` default to `true` for plants that have `pvlib_enabled = true` but no explicit value? **Default plan: yes.** Confirm.

---

## 9. File / line index for the implementer

| Concern | File | Where |
|---|---|---|
| Telemetry key constants (`KEY_POTENTIAL_POWER`, etc.) | `app/services/forecast_service.py` | top of file |
| Daily integration pattern to mirror | `app/services/daily_job.py::_integrate_plant_day` | full function |
| Ancestor roll-up pattern | `app/services/forecast_service.py::_rollup_parents` | full method |
| TB client (timeseries read, telemetry post, attribute read) | `app/services/thingsboard_client.py` | `get_timeseries`, `post_telemetry`, `get_asset_attributes` |
| Plant discovery + ancestor map | `app/services/thingsboard_client.py::discover_plants` | full method |
| Cron job registration | `app/services/scheduler.py::start_scheduler` | bottom of function |
| Existing weekly-eval pattern (job + admin endpoint) | `app/services/weekly_eval.py` and `app/api/forecast.py` (`/admin/run-weekly`) | reference |
| Widget computed branch | `Loss Attribution/.js` | `renderComputedMode`, `calculateLossForRange` |
| Widget settings | `Loss Attribution/settings.json` | full file |
| Widget README | `Loss Attribution/README.md` | full file |
| `active_power_unit` widget recipe (replicate in service) | `Pvlib-Service/TELEMETRY_CONTRACT.md` | "Widget scaling recipe" |

---

End of plan.
