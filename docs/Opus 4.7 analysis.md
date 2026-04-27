# Opus 4.7 Analysis — Pvlib-Service Gap Resolution

*Author: Opus 4.7 planning pass, drafted 2026-04-24. Successor to `docs/deployment_and_integration_plan.md` (H-D1). This document enumerates the gaps that remain after H-D1 implementation, proposes multiple hypotheses per gap, rates them with ≥90% confidence in the rating, and converges on a single solution or a hybrid per gap. The implementation will be executed by Claude Sonnet 4.6 with the engineered prompt in §6.*

---

## 0. Purpose and Conventions

The pvlib-service in `scripts/Pvlib-Service/` was built to the H-D1 hybrid plan: APScheduler 1-minute cycle, `isPlant && pvlib_enabled` gating, dedup-aware parent roll-up, dual-write key contract, single EC2 container. The service is structurally complete — the verification pass identified 26 residual gaps, broken out below by severity.

**Confidence-rating convention.** Each hypothesis carries two numbers: a quality score out of 10 and a confidence in that score. The confidence is in the *correctness of the rating*, not in whether the approach will work. Any hypothesis below 90 % confidence-in-rating is dropped before the decision step; the user has asked that every scored option meet that bar.

**What this document does *not* cover.** The physics of the ModelChain (H-A3), the choice of SAPM thermal model, the PVsyst vs pvlib accuracy question (tackled by `scripts/shared/validate_pvlib.py`), or the H-B6 tiering strategy. Those are settled decisions; this document only fixes what the H-D1 implementation missed or got wrong.

**Authoritative key contract.** `scripts/Pvlib-Service/TELEMETRY_CONTRACT.md` remains the source of truth for telemetry keys; any change here that touches keys must be reflected there in the same commit.

---

## 1. Gap Inventory

| #  | Gap                                                                                      | Severity  | Touches                                                                               |
|----|------------------------------------------------------------------------------------------|-----------|---------------------------------------------------------------------------------------|
| 1  | WSTN / P341 devices not discoverable (no `Contains` relation to plants)                  | Blocker   | `forecast_service._load_config`, `physics/config.py`                                  |
| 2  | -1 failure sentinel never written to TB                                                  | Blocker   | `forecast_service.process_single_asset`, `run_fleet_cycle`, roll-up                   |
| 3  | Daily kWh uses only a 90-second window (≈960× understated)                               | Blocker   | `run_fleet_cycle`, `_write_daily_energies`                                            |
| 4  | Daily trigger fires at 01:00 UTC = 06:30 Asia/Colombo                                    | Blocker   | Scheduler timezone, daily job                                                         |
| 5  | Roll-up reaches only *direct* parent — grandparent aggregation assets receive nothing     | Blocker   | `_rollup_parents`, `discover_plants`                                                  |
| 6  | Roll-up re-reads TB telemetry instead of summing in-memory DataFrames                    | High      | `_rollup_parents`, `process_single_asset` return signature                            |
| 7  | Solcast has no cache — will blow free-tier quota the moment a no-station plant enables   | High      | `data_sources._fetch_solcast`                                                         |
| 8  | No widget currently reads `potential_power` — service writes, nothing renders            | High      | Widget JSON settings (Curtailment V5, Loss Attribution, Forecast vs Actual)           |
| 9  | `active_power_unit` attribute documented but unset on every real plant                   | High      | Operational — TB attribute population across ~30 plants                               |
| 10 | `weather_station_id` / `p341_device_id` absent from `PlantConfig` parsers                | Blocker   | `physics/config.py` blob + flat paths                                                 |
| 11 | `_sync_cycle_job` bridge is dead code with a 3.12 deprecation trap                       | Low       | `scheduler._sync_cycle_job`                                                           |
| 12 | A new TB client is created + re-authenticated every cycle (wastes ~150 min of valid JWT) | Medium    | `scheduler.run_cycle_now`, `main.lifespan`                                            |
| 13 | `discover_plants` runs every minute with no caching                                      | Medium    | `thingsboard_client.discover_plants`, cache layer                                     |
| 14 | Scheduler hard-coded to UTC even though `TZ_LOCAL` is defined                            | Medium    | `scheduler.get_scheduler`                                                             |
| 15 | `/metrics` is missing per-tier / per-plant counters promised by the plan                 | Medium    | `api/forecast.metrics`                                                                |
| 16 | `/health` returns 200 at cold start, can't distinguish "starting" from "dead"            | Medium    | `api/forecast.health`                                                                 |
| 17 | `/admin/run-now` bypasses `max_instances=1`, can overlap a running cycle                 | Medium    | `api/forecast.admin_run_now`                                                          |
| 18 | `docker-compose.yml` only includes pvlib-service; H-C4 "one host, three modes" unfinished| Medium    | `docker-compose.yml`                                                                  |
| 19 | No TB rule-chain / EventBridge watchdog JSON shipped (P1-E external half)                | Medium    | New `ops/watchdog_rule_chain.json`                                                    |
| 20 | No live A/B harness comparing pvlib to the PVsyst-math-based old model                   | Medium    | `scripts/shared/validate_pvlib.py`, scheduled job                                     |
| 21 | Secrets live in a plain-text `.env` instead of SSM Parameter Store                       | Low       | EC2 bootstrap script (`ops/ec2_userdata.sh`)                                          |
| 22 | No retry/backoff wrapper on httpx calls; transient 5xx loses a cycle for that plant      | Medium    | `thingsboard_client._get`, `_post`                                                    |
| 23 | BFS `queue` is a shared list mutated across concurrent tasks — fragile                   | Low       | `thingsboard_client.discover_plants`                                                  |
| 24 | `far_shading` has a dual-path read in `_from_blob` that could drift                      | Low       | `physics/config.py`                                                                   |
| 25 | Night-time zero mask uses `ghi <= 0 AND poa_measured <= 0` → fails for POA-only stations | Medium    | `physics/pipeline.compute_ac_power`                                                   |
| 26 | No tool to audit on-plant TB attributes vs the config JSON                               | Low       | New `scripts/shared/audit_tb_config.py`                                               |

---

## 2. Hypotheses, Ratings, and Decisions

For each gap, 3-5 hypotheses with ratings and a decision. Ratings preserved only where the confidence-in-rating ≥ 90 %.

### 2.1 Gap 1 — Weather station / P341 device discovery

**Problem.** `_load_config` in `forecast_service.py` (L304-L318) finds station and meter devices by calling `tb.get_child_relations(plant_id)` and name-matching. But **no WSTN/P341 device has a `Contains` (or any) relation to its plant in ThingsBoard** — confirmed by user. So every plant silently falls to Tier-3 clearsky even when a real pyranometer is producing data.

**Hypotheses.**

- **H1-A — Explicit IDs in plant attributes.** Add `weather_station_id` and `p341_device_id` as optional fields in the `pvlib_config` blob and the flat-attribute schema. Service reads them directly, no relation lookup. **9/10 · 97 %.**
- **H1-B — Naming-convention scan.** Search TB devices whose name starts with `{plant_short_name}_WSTN` / `_P341` (e.g., `KSP_WSTN1`). **6/10 · 95 %** — works for KSP/SOU/SSK but user explicitly flagged "naming convention and case is inconsistent"; fragile long-term.
- **H1-C — Alternate relation types (`From`, `ManagedBy`, custom).** **2/10 · 96 %** — user already said *no* relation exists, so none would match.
- **H1-D — Hardcoded registry JSON.** **1/10 · 98 %** — violates the "avoid hardcoding hierarchy" instruction.
- **H1-E — Back-reference attribute on the device** (each device carries `plant_asset_id`); service does a one-time `GET /api/tenant/devices?textSearch=...` at startup. **7/10 · 92 %** — good long-term, but requires per-device attribute setup.
- **H1-F — Hybrid: H1-A primary + H1-B fallback (zero-config bootstrap) + `WARN` log asking operator to set the explicit ID.** **9.5/10 · 93 %**.

**Decision — H1-F.** Ship explicit-ID parsing *and* a best-effort naming fallback that covers KSP/SOU/SSK on day one without any attribute edits. Log a `WARN` whenever the fallback fires, so the operator gradually migrates to explicit IDs. H1-E is a future-v2 option once the fleet grows.

**Implementation notes.**
- In `physics/config.py`, add two optional fields to `PlantConfig` (already present as fields but never parsed). In `_from_blob` extract `cfg.get("weather_station_id")` / `p341_device_id`. In `_from_flat` pull them from `attrs`.
- In `forecast_service._load_config`, order of resolution:
  1. Use `config.weather_station_id` / `p341_device_id` if set in attrs.
  2. Else call `get_child_relations` (legacy path, harmless).
  3. Else run a tenant device-search by plant-name prefix (see H1-B algorithm below). Cache results per plant for the process lifetime — device UUIDs don't change.
- Plant-name prefix derivation: take `plant_name.split('_')[0]` (e.g., `KSP_Plant` → `KSP`). Query `GET /api/tenant/devices?textSearch=KSP&limit=50`, match devices whose name starts with `{prefix}_WSTN` or `{prefix}_P341`. If multiple WSTN devices (SSK has WSTN, WSTN_T, WSTN_R), prefer the one with the most relevant keys matching `config.station.ghi_key` / `poa_key` after a brief latest-telemetry probe — see Edge case *multi-station* below.
- Record which path succeeded in a new log key `station_resolution_mode`: `explicit|contains|naming|none`.

**Edge cases.**
- *Multi-station plant* (SSK has WSTN, WSTN_T, WSTN_R). Resolution: iterate candidates, call `get_latest_telemetry(device, [ghi_key, poa_key])`; pick the one returning the most configured keys. If still ambiguous, pick the shortest-named device (parent before children).
- *No plant_name* (Unknown). Resolution: skip naming fallback; log `station_resolution_mode=none`.
- *Plant whose WSTN was deleted from TB between cycles.* Resolution: if latest-telemetry returns nothing for a cached ID for 10 consecutive cycles, invalidate the cache entry and re-resolve.

---

### 2.2 Gap 2 — -1 failure sentinel

**Problem.** User contract: *"update the timeseries with -1 if the value could not be calculated…so that widgets can handle it properly."* Today the service returns a status dict on failure and writes nothing. Widgets see gaps, not sentinels.

**Hypotheses.**

- **H2-A — Write -1 only once per failed cycle at window-end ts.** **3/10 · 96 %** — doesn't meet "at each timestamp".
- **H2-B — Write -1 for every 1-minute boundary in the window, for both primary and alias keys.** **8/10 · 95 %**.
- **H2-C — Write `NaN` / `null`.** **2/10 · 97 %** — user asked for -1; TB drops NaN on many setups.
- **H2-D — Write -1 on the numeric keys plus the failure reason on `pvlib_data_source` (e.g., `"error:no_weather"`).** **9.5/10 · 94 %**.
- **H2-E — Write a two-key contract: `potential_power = -1` + `pvlib_status = "error:<reason>"` (new key).** **8/10 · 92 %** — richer, but introduces another key.

**Decision — H2-D.** Reuse the existing `pvlib_data_source` key for the failure reason (no new key, no widget change), and stamp -1 on *both* `potential_power` and `active_power_pvlib_kw` at every 1-minute boundary in the failure window.

**Implementation notes.**
- Add `forecast_service._build_sentinel_records(start: datetime, end: datetime, reason: str) -> List[dict]`:
  ```python
  records = []
  ts = start.replace(second=0, microsecond=0)
  while ts <= end:
      records.append({
          "ts": int(ts.timestamp() * 1000),
          "values": {
              KEY_POTENTIAL_POWER: -1,
              KEY_PVLIB_POWER: -1,
              KEY_DATA_SOURCE: f"error:{reason}",
              KEY_MODEL_VERSION: "pvlib-h-a3-v1",
              KEY_UNIT: "kW",
          },
      })
      ts += timedelta(minutes=1)
  return records
  ```
- Replace every `return {"status": "...error..."}` branch in `process_single_asset` with an additional `await self._tb.post_telemetry("ASSET", asset_id, sentinels)` before returning.
- Failure taxonomy (reason tokens, stable for dashboards):
  - `no_config` — attributes missing / parse failed
  - `no_location` — latitude/longitude 0/missing
  - `no_data` — all three tiers returned empty
  - `physics_error` — compute_ac_power threw
  - `write_error` — TB write of *real* results failed (sentinel write is a best-effort retry)
  - `timeout` — overall plant processing > 30 s
- `run_fleet_cycle` edge: a plant that was discovered but never reached `process_single_asset` (e.g., semaphore cancelled, exception in `gather`) must also get sentinels. Wrap `bounded_compute` to fall back to sentinel writes on any unexpected exception.

**Edge cases.**
- *Sentinel write itself fails.* Log `ERROR` and move on — don't retry sentinels.
- *Partial cycle success*: physics ran but only 30 of 90 minutes had valid data. Write real records for the valid minutes and **do not** overwrite valid minutes with sentinels. Sentinels fill only minutes that produced no real record.
- *Parent roll-up with -1 children.* The roll-up must **exclude** -1 records from the sum (treat them as missing), and emit -1 at a parent minute only if **all** children are -1 at that minute. Rationale: a single plant's failure should not poison a region total; a whole-region outage should still be visible as -1.
- *Clock skew* (EC2 drift): align sentinel ts to minute boundaries (`second=0`) so duplicate writes across cycles just overwrite each other cleanly.

---

### 2.3 Gap 3 — Daily kWh roll-up uses the 90-s window

**Problem.** `process_single_asset` computes `total_kwh = df["potential_power_kw"].sum() / 60.0` over ~90 s of data, and the daily code treats that as the day total.

**Hypotheses.**

- **H3-A — Accumulator per plant in memory; reset at local midnight.** **5/10 · 93 %** — lost on restart; drifts if cycles skip.
- **H3-B — Daily cron job that re-reads `potential_power` for the past calendar day and integrates.** **9/10 · 95 %** — authoritative, uses the same data the widgets will see.
- **H3-C — Widget-side sum.** **3/10 · 95 %** — wrong layer.
- **H3-D — Accumulator + nightly reconciliation from TB read.** **7/10 · 92 %** — adds real-time UX but duplicates code.
- **H3-E — APScheduler cron trigger at 00:05 local tz (`minute=5, hour=0, timezone=settings.TZ_LOCAL`) that triggers the reconciliation.** **9.5/10 · 95 %**.

**Decision — H3-E.** One APScheduler cron job scheduled on local-midnight+5-min. It reads the past 24 h of `potential_power` per plant, integrates trapezoidally, writes `total_generation_expected_kwh` and `pvlib_daily_energy_kwh` dated 00:00 local. This also resolves Gap 4 cleanly.

**Implementation notes.**
- New file `app/services/daily_job.py`:
  ```python
  async def run_daily_rollup(tb_client, plant_list):
      tz = ZoneInfo(settings.TZ_LOCAL)
      day_end_local = datetime.now(tz).replace(hour=0, minute=0, second=0, microsecond=0)
      day_start_local = day_end_local - timedelta(days=1)
      day_start_utc = day_start_local.astimezone(timezone.utc)
      day_end_utc = day_end_local.astimezone(timezone.utc)
      # For each plant, read potential_power [day_start_utc, day_end_utc], integrate, write total_generation_expected_kwh at day_start_local ts.
  ```
- Integration: drop -1 records, trapezoidal integrate, output in kWh. For a 1-minute-cadence series, integral is `sum(values) * 1 / 60` kWh.
- If a plant has < 360 valid samples in the day (< 6 daylight hours), write `-1` for the daily key with `pvlib_data_source="error:insufficient_samples"`.
- Parent roll-up of daily energy: after per-plant daily write, iterate `ancestors` (see Gap 5) and sum unique children for each ancestor; write to each ancestor asset.

**Edge cases.**
- *Service cold-start after 48 h downtime.* The daily job at 00:05 covers only the last 24 h. Provide an `/admin/run-daily?date=YYYY-MM-DD` endpoint for manual backfill.
- *DST transitions.* Asia/Colombo has no DST, but guard with `ZoneInfo` everywhere to be portable. For future sites in DST zones, `day_end_local - timedelta(days=1)` via tz-aware arithmetic handles fall-back/spring-forward correctly (returns a 23h or 25h day — integration is still correct since samples are tz-aware UTC).

---

### 2.4 Gap 4 — Scheduler timezone

Resolved jointly with Gap 3 by constructing the `AsyncIOScheduler` with `timezone=settings.TZ_LOCAL`. The minute-interval job is timezone-independent, but daily cron needs local tz for correct midnight semantics. **9.5/10 · 97 %.**

---

### 2.5 Gap 5 — Only direct parents get roll-ups

**Problem.** `plant.parent_ids` is populated only with the **immediate** parent(s). Roll-ups never reach `Windforce Plants`, `Windforce Overview`, `SCADA Power Plants`, or the two-level `isPlantAgg` chains (e.g., `Akbar Brothers → Windforce Overview → Windforce Plants`).

**Hypotheses.**

- **H5-A — Walk `parent_ids` recursively at roll-up time.** **7/10 · 93 %** — requires a second BFS per cycle.
- **H5-B — Store a complete `ancestors: Set[str]` on each `PlantRef` during discovery.** **9.5/10 · 96 %** — single data structure, O(1) lookup.
- **H5-C — Reverse-walk: for each aggAsset found, compute descendants lazily.** **7/10 · 94 %** — equivalent but indirect.
- **H5-D — Treat only `isPlantAgg == true` nodes as roll-up targets; skip root "SCADA Power Plants" etc.** Composable with H5-B. Adds a filter.

**Decision — H5-B + H5-D.** Extend `discover_plants` to return `plants, ancestor_map: Dict[plant_id, Set[str]]`. During BFS, any non-plant ASSET with `isPlantAgg == true` is recorded in a per-path `ancestors_in_progress` set; when the plant is reached, that set is stored for the plant. Roll-up iterates `ancestor_map[plant.id]`, unioned across paths. Non-isPlantAgg intermediates (shouldn't exist in our tree but defensively handled) are *not* roll-up targets.

**Implementation notes.**
- Change `_visit_node` signature to carry a `path_ancestors: Tuple[str, ...]` along with the queued `(asset_id, parent_id)`. When the node is a plant, store the ancestors in the map. When it is a non-plant aggregation, append its id to the path if `isPlantAgg == true`, then BFS children with the updated tuple.
- `_rollup_parents` changes: instead of `for parent_id in plant.parent_ids`, iterate `for ancestor_id in ancestor_map[plant.id]`. For each ancestor, build `unique_child_plant_ids = {p.id for p in plants if ancestor_id in ancestor_map[p.id]}` and sum.
- Compose with Gap 6: the sum uses in-memory DataFrames, not TB re-reads.

**Edge cases.**
- *Plant reachable via two paths (KSP under SCADA Power Plants *and* Windforce Groundmount Plants).* Ancestors for KSP = {SCADA Power Plants, Windforce Plants, Windforce Groundmount Plants}. Each ancestor gets KSP counted **once**. Correct by construction.
- *Cycle in the hierarchy (A contains B contains A).* Guard with a visited set at BFS level (already present); break cycles by ignoring second-visit.
- *Ancestor not marked `isPlantAgg` yet operators want roll-ups* (e.g., a regional asset someone forgot to tag). Resolution: document that roll-ups follow `isPlantAgg`. Provide `/pvlib/discover` output that includes ancestor lists for ops visibility.
- *Root SCADA Power Plants has `isPlantAgg=true` but `default` asset profile.* Roll-up still writes — TB accepts telemetry on any profile. Log `DEBUG` when writing to a non-SolarPlant ancestor for traceability.

---

### 2.6 Gap 6 — Roll-up re-reads TB

**Hypotheses.**

- **H6-A — Return the `pd.DataFrame` from `process_single_asset` and sum in memory.** **9.5/10 · 97 %**.
- **H6-B — Use TB `agg=SUM` parameter server-side.** **4/10 · 95 %** — doesn't help correctness, and still requires re-read.
- **H6-C — Redis-backed cache of per-plant series.** **5/10 · 94 %** — over-engineered for 10 plants.

**Decision — H6-A.**

**Implementation notes.**
- Introduce a `@dataclass PlantCycleResult`:
  ```python
  @dataclass
  class PlantCycleResult:
      asset_id: str
      status: str            # "ok" | "no_data" | "config_error" | ...
      source: str | None
      df: pd.DataFrame | None  # None if no real data produced
      sentinels: list[dict] | None  # written sentinels, for debugging
      error: str | None = None
  ```
- `process_single_asset` returns `PlantCycleResult`.
- `_rollup_parents`:
  ```python
  ok_results = {r.asset_id: r for r in plant_results.values() if r.status == "ok" and r.df is not None}
  for ancestor_id, child_ids in ancestor_children.items():
      series = [ok_results[cid].df["potential_power_kw"] for cid in child_ids if cid in ok_results]
      if not series:
          continue
      total = pd.concat(series, axis=1).sum(axis=1, skipna=True)
      # build records from total, write to ancestor
  ```
- Preserves the -1 handling from Gap 2: `pd.concat` ignores plants with `df=None` (no contribution).

**Edge cases.**
- *Parent all-child failure.* `series` is empty → emit -1 sentinels for the parent (composes with Gap 2).
- *Heterogeneous timestamps across children.* `pd.concat(..., axis=1)` aligns on the union of indices; `.sum(axis=1, skipna=True)` then sums present values. Gap 25 (night-time) still zeros correctly.

---

### 2.7 Gap 7 — Solcast has no cache

**Hypotheses.**

- **H7-A — In-process TTL dict, 30-min buckets.** **9/10 · 96 %**.
- **H7-B — Redis.** **4/10 · 95 %** — multi-node premature.
- **H7-C — Disk cache (SQLite).** **6/10 · 93 %** — persists across restarts, but adds a volume mount and a dep.
- **H7-D — Serve stale on API failure for up to 60 min.** Composable with H7-A.

**Decision — H7-A + H7-D.** Dict-based cache keyed by `(resource_id, start_bucket_30min)`; serve stale-up-to-60-min on 4xx/5xx. Lost-on-restart is acceptable given 30 s cold-start penalty.

**Implementation notes.**
- Module-level in `data_sources.py`:
  ```python
  _solcast_cache: Dict[Tuple[str, int], Tuple[pd.DataFrame, float]] = {}
  _solcast_lock = asyncio.Lock()
  _BUCKET_SEC = 1800
  _STALE_OK_SEC = 3600
  ```
- `_fetch_solcast`: compute `bucket = int(start.timestamp()) // _BUCKET_SEC`. Check cache; if fresh → return. Else fetch; on success → store + return; on failure → return stale if age < `_STALE_OK_SEC` else propagate.
- Expose size via `/metrics` (`pvlib_solcast_cache_size`, `pvlib_solcast_hits_total`, `pvlib_solcast_misses_total`).

**Edge cases.**
- *Bucket boundary miss* (request straddles 30-min edge): use `end.timestamp() // _BUCKET_SEC` instead of `start`, OR merge two bucket fetches. Simpler: round `start` down to bucket and request the full 30-min bucket; slice the requested window from the bucket.
- *Solcast quota 429.* Return stale if available; else return empty DataFrame (triggers Tier-3 clearsky).

---

### 2.8 Gap 8 — No widget reads `potential_power`

**Problem.** Operational, not service-code: the widgets in the Widgets folder are data-source-configurable but currently bound to `active_power` only.

**Hypotheses.**

- **H8-A — Update each target widget's `datasources` array to add `potential_power` as a secondary dataset.** **8/10 · 93 %**.
- **H8-B — TB rule chain copies `potential_power` → a key the widget already reads.** **5/10 · 94 %** — defeats the clean key contract.
- **H8-C — A widget-settings JSON patch script (`scripts/shared/patch_widgets.py`) that re-writes the JSON files.** **7/10 · 91 %** — repeatable but risky if widget JSON schema shifts.
- **H8-D — Document the manual TB-UI step** in `docs/deployment_and_integration_plan.md` §7 and leave it to ops. **6/10 · 94 %**.

**Decision — H8-A executed manually per target widget, documented per the H8-D pattern**. The three targets are:
1. `Widgets/Grid & Losses/Curtailment vs Potential Power/V5 TB Timeseries Widget` — add a second timeseries dataset keyed `potential_power`, rendered as a dashed line next to `active_power`.
2. `Widgets/Grid & Losses/Loss Attribution` — already datasource-agnostic; bind its "expected" input to `total_generation_expected_kwh` (daily) or `potential_power` (instantaneous) as appropriate per mode.
3. `Widgets/Forecasts & Risk/Forecast vs Actual Energy` (if present) — add `total_generation_expected_kwh` alongside `forecast_p50_daily`.

Sonnet 4.6 should **edit the widget JSON files in place** and document the change.

**Edge cases.**
- *Widget JSON checked into multiple copies (V1-V5).* Only V5 is the currently-shipped one per the audit. Confirm with the operator before touching older versions.
- *TB dashboards pull widgets by reference.* If `Dashboard.json` bakes an older widget version inline, update the inline copy too. Re-audit `Dashboard.json` for any embedded widget JSON referencing these files.

---

### 2.9 Gap 9 — `active_power_unit` not populated

**Hypotheses.**

- **H9-A — Service writes a normalized `active_power_kw` mirror of the meter.** **7/10 · 92 %** — doubles meter storage volume (1440 writes/day × 30 plants × 1 key = 43 k writes/day extra).
- **H9-B — Populate `active_power_unit` attribute on every plant (one-shot TB migration).** **8.5/10 · 95 %**.
- **H9-C — Widget-side conditional scaling helper.** **5/10 · 93 %** — repeats logic per widget.
- **H9-D — Hybrid: H9-B + a one-line widget settings snippet in TELEMETRY_CONTRACT.md.** **9/10 · 93 %**.

**Decision — H9-D.** Ops task (H9-B) + documented widget-config recipe.

**Implementation notes.**
- Add a script `scripts/shared/set_active_power_unit.py` that:
  - Reads a CSV/dict mapping plant asset ID → unit (`"W" | "kW"`) per the hierarchy in the user prompt.
  - POSTs `SERVER_SCOPE` attribute `active_power_unit=<value>` to each.
  - Idempotent (skips if already correct).
- Pre-populate the mapping from the user's prompt:
  - `kW` → KSP, SSK, SOU, PSP, VPE Plant1, VPE Plant2, SON, SER, SUN, VYD (plants without the "active_power in W" note), AKB Welisara 1, AKB Welisara 2 (no W note), Aerosense (no W note), Mouldex 1, Mouldex 2 (no W note).
  - `W` → everything listed as "(active_power in W)": AKB Kelaniya, AKB Exports Mabola, Chris Logix 1, Chris Logix 2, Lina Manufacturing, Quick Tea, Harness, Flinth Admin, Mona Rathmalana, Mona Homagama, Mona Koggala, Hir Agalawaththa, Hir Kahatuduwa 1, Hir Kahatuduwa 2, Hir Kuruvita, Hir Mullaitivu, Hir Eheliyagoda, Hir Seethawaka 1, Hir Seethawaka 2, Hir Maharagama 1, Hir Vavuniya.
- Widgets co-plotting `active_power` and `potential_power` bind a dashboard `postProcess` function:
  ```javascript
  // In widget datasource advanced settings:
  // if unit attribute is "W", scale by 0.001 to kW
  return attributes.active_power_unit === "W" ? value * 0.001 : value;
  ```
- Document this in `TELEMETRY_CONTRACT.md` under *Unit Contract*.

**Edge cases.**
- *Plant that changes meter firmware mid-life.* Operator updates the attribute; widgets pick up the new scaling automatically.
- *Missing attribute on future new plants.* Default in widget JS: `unit ?? "kW"` → no scaling. Document this default.

---

### 2.10 Gap 10 — `weather_station_id` missing from parsers

Covered under Gap 1 decision (H1-F). Add parser lines in `PlantConfig._from_blob` and `_from_flat`. **9.5/10 · 97 %.**

---

### 2.11 Gap 11 — Dead `_sync_cycle_job` with 3.12 trap

**Decision.** Delete it. `AsyncIOScheduler` can schedule coroutines directly; the bridge is unused. **9.5/10 · 98 %.**

---

### 2.12 Gap 12 — New TB client per cycle

**Hypotheses.**

- **H12-A — Module-level singleton client, constructed in `main.lifespan`, passed to scheduler.** **9/10 · 95 %**.
- **H12-B — `ForecastService` owns the client.** **8/10 · 94 %** — same outcome, more coupling.

**Decision — H12-A.**

**Implementation notes.**
- In `main.py.lifespan`: construct `tb_client = ThingsBoardClient(...)`; `await tb_client.__aenter__()`; attach to `app.state.tb_client`.
- Scheduler `run_cycle_now` reads `app.state.tb_client` (via a reference injected at scheduler start).
- On shutdown: `await tb_client.__aexit__(None,None,None)`.
- JWT refresh still handled by the client's `_ensure_token` path.

**Edge cases.**
- *TB restart* invalidates the JWT mid-session. `_ensure_token` re-logins on 401; client continues.
- *Client leak* on reload. Lifespan `__aexit__` handles it.

---

### 2.13 Gap 13 — `discover_plants` every cycle

**Hypotheses.**

- **H13-A — 5-min TTL cache.** **8/10 · 95 %**.
- **H13-B — TTL + manual `/admin/refresh-plants` invalidation endpoint.** **9/10 · 94 %**.

**Decision — H13-B** (hybrid — TTL plus manual invalidation).

**Implementation notes.**
- `ThingsBoardClient.discover_plants(..., ttl_seconds: int = 300, force: bool = False)`: memoize the last result per `(sorted_root_ids_tuple)` key.
- Cycle calls it with TTL; `/admin/refresh-plants` calls it with `force=True`.
- Cache is a simple module-level dict protected by `asyncio.Lock`.

---

### 2.14 Gap 14 — Scheduler hard-coded to UTC

Resolved by Gap 3/4 — set `timezone=ZoneInfo(settings.TZ_LOCAL)` on the scheduler. **9.5/10 · 97 %.**

---

### 2.15 Gap 15 — /metrics gaps

**Decision.** Add the three counters the plan specified:
- `pvlib_data_source_count{source="tb_station|solcast|clearsky|rollup|error"}` — gauge, reset each cycle.
- `pvlib_plant_failures_total{plant,reason}` — counter, monotonic.
- `pvlib_solcast_hits_total`, `pvlib_solcast_misses_total` (new, from Gap 7).
- `pvlib_discover_cache_hits_total`, `pvlib_discover_cache_misses_total` (new, from Gap 13).

**9.5/10 · 95 %.**

---

### 2.16 Gap 16 — /health cold-start

**Decision.** Track `SERVICE_STARTED_AT`. If `SCHEDULER_ENABLED` and `last_cycle_finished_at is None`:
- If `now - SERVICE_STARTED_AT < 120 s` → 200 with `status="initializing"`.
- Else → 503 with `status="never_ran"`.

**9/10 · 95 %.**

---

### 2.17 Gap 17 — /admin/run-now overlaps

**Decision — H17-B.** `scheduler.get_job("pvlib_cycle").modify(next_run_time=datetime.now())` delegates to APScheduler's `max_instances=1` machinery. Drop the `asyncio.create_task(run_cycle_now())` path.

**9/10 · 94 %.**

---

### 2.18 Gap 18 — Compose only has pvlib

Out of MVP scope for this pass. Decision: leave as-is, add a comment pointing to a follow-up task. The Simple Forecast and Solcast Forecast services continue to run from their existing deploy (`start_api.py` / `start_api.ps1`). **Rating: 5/10 · 94 %** — accepted on operational grounds, not technical merit.

If the user wants H-C4 closure in this pass, Sonnet should:
1. Add `Dockerfile` to both legacy services, copying their existing `start_api.py` as `CMD`.
2. Extend compose with `solcast-service` (port 8000) and `simple-service` (port 8001). All three containers share the same EC2 host.

Defer to a separate user decision.

---

### 2.19 Gap 19 — Watchdog JSON

**Decision.** Ship `ops/watchdog_rule_chain.json` — a TB rule chain export that:
1. Runs a *Generator* node every 5 min.
2. Uses a *REST API Call* node to GET `{pvlib_host}/health`.
3. Uses a *Switch* node: on HTTP 5xx or `status != "ok"` → emit alarm `pvlib_service_stalled` on a heartbeat device.
4. Else clear any existing alarm.

**8.5/10 · 92 %.**

Provide a copy-paste-importable JSON stub; operator sets the host URL in a *constant* node before importing.

---

### 2.20 Gap 20 — A/B harness

**Decision — out-of-band weekly eval.** Schedule `scripts/shared/validate_pvlib.py` as a weekly APScheduler job (Sunday 02:00 local) that:
1. Pulls the past 7 days of `potential_power` (pvlib) and `active_power` (meter, unit-normalized) for KSP, SOU, SSK (plants with P341 truth + pyranometer).
2. Computes NMAE, NRMSE, energy-error%.
3. Writes a report to TB as a string attribute `pvlib_accuracy_report_json` on each plant, and to a CloudWatch log group.
4. Does **not** write parallel real-time telemetry — keeps compute cost low.

**8/10 · 92 %.**

---

### 2.21 Gap 21 — Secrets in .env

**Decision — H21-A.** On EC2, a systemd one-shot (`ops/ec2_userdata.sh`) reads SSM Parameter Store at boot and writes the `.env` on tmpfs (`/run/pvlib/.env`). Compose bind-mounts `/run/pvlib/.env:/app/.env:ro`. Secrets never on persistent disk, app code unchanged. **8.5/10 · 93 %.**

---

### 2.22 Gap 22 — No retry/backoff on httpx

**Decision — `tenacity`.** Add `@retry(stop=stop_after_attempt(3), wait=wait_exponential_jitter(initial=0.5, max=5), retry=retry_if_exception_type((httpx.TimeoutException, httpx.HTTPStatusError)))` on `_get` and `_post`, **with an additional predicate** that *only* retries on `HTTPStatusError` for 429/502/503/504 (not 4xx client errors, not 401 which should re-login). **9/10 · 95 %.**

Add `tenacity==8.2.3` to `requirements.txt`.

---

### 2.23 Gap 23 — BFS queue shared mutation

**Decision.** Replace the shared list with an `asyncio.Queue`. Each worker consumes from the queue and `put`s children. Use `queue.join()` for termination. **8/10 · 93 %** — low actual risk today, but cheap fix.

---

### 2.24 Gap 24 — `far_shading` dual-path

**Decision.** In `_from_blob`, read only from one place (the top-level `cfg.far_shading` per `kebithigollewa_pvlib_config.json` structure). Remove the `losses.far_shading` fallback. Validate with the example JSON. **9/10 · 94 %.**

---

### 2.25 Gap 25 — Night-time zero mask

**Decision.** Change condition in `pipeline.compute_ac_power`:
```python
effective_irradiance = poa_global if "poa_global" in locals() else poa_measured
night_mask = effective_irradiance.fillna(0.0) <= 1.0   # sub-threshold
pac_kw[night_mask] = 0.0
```
This survives POA-only stations. **9/10 · 94 %.**

---

### 2.26 Gap 26 — No config audit tool

**Decision.** Add `scripts/shared/audit_tb_config.py`:
1. Connects to TB, fetches `pvlib_config` / flat attrs for each plant.
2. Compares against `config/<plant>_pvlib_config.json` (when present).
3. Reports diffs (missing keys, value mismatches).
4. Exit code 0/1 for CI integration.

**9/10 · 92 %.**

---

## 3. Implementation Phases

Phases are ordered so that each phase's exit gate can be independently verified on the staging TB tenant (KSP_Plant with `pvlib_enabled=true`). Do not proceed to phase N+1 until N passes its gate.

### Phase A — Data plumbing (must ship first)

| Gap | Files touched                                                                                                |
|-----|--------------------------------------------------------------------------------------------------------------|
|  1  | `scripts/Pvlib-Service/app/physics/config.py`, `scripts/Pvlib-Service/app/services/forecast_service.py`, `scripts/Pvlib-Service/app/services/thingsboard_client.py` (add `search_devices_by_name_prefix`) |
| 10  | (same as 1)                                                                                                  |
|  2  | `scripts/Pvlib-Service/app/services/forecast_service.py`                                                     |
|  5  | `scripts/Pvlib-Service/app/services/thingsboard_client.py` (ancestor_map), `forecast_service.py`             |
|  6  | `scripts/Pvlib-Service/app/services/forecast_service.py` (`PlantCycleResult` dataclass)                      |

**Exit gate A.** On KSP_Plant with `pvlib_enabled=true`: `/admin/run-now` produces (a) non-empty `potential_power` with `pvlib_data_source=tb_station` (station discovered via fallback), (b) roll-ups at direct parent `SCADA Power Plants` *and* grandparents `Windforce Groundmount Plants`, `Windforce Plants`, and (c) on simulated station failure (wrong key), -1 sentinels appear at 1-min intervals.

### Phase B — Time semantics

| Gap | Files                                                                         |
|-----|-------------------------------------------------------------------------------|
|  3  | New `scripts/Pvlib-Service/app/services/daily_job.py`, register in scheduler  |
|  4  | `scripts/Pvlib-Service/app/services/scheduler.py` (timezone)                  |
| 14  | (same)                                                                        |

**Exit gate B.** Let the service run for 24 h with 1-minute cadence; at 00:05 local, `total_generation_expected_kwh` appears on KSP_Plant with a value within 5 % of the integral of `potential_power` computed externally over the same range.

### Phase C — Ops resilience

| Gap | Files                                                                 |
|-----|-----------------------------------------------------------------------|
|  7  | `scripts/Pvlib-Service/app/physics/data_sources.py`                   |
| 12  | `scripts/Pvlib-Service/app/main.py`, `scheduler.py`                   |
| 13  | `scripts/Pvlib-Service/app/services/thingsboard_client.py`            |
| 16  | `scripts/Pvlib-Service/app/api/forecast.py` (`/health`)               |
| 17  | `scripts/Pvlib-Service/app/api/forecast.py` (`/admin/run-now`)        |
| 22  | `scripts/Pvlib-Service/app/services/thingsboard_client.py`            |
| 23  | `scripts/Pvlib-Service/app/services/thingsboard_client.py`            |
| 25  | `scripts/Pvlib-Service/app/physics/pipeline.py`                       |
| 11  | `scripts/Pvlib-Service/app/services/scheduler.py` (delete dead code)  |
| 24  | `scripts/Pvlib-Service/app/physics/config.py`                         |

**Exit gate C.** Kill the TB client during a cycle (simulate 502 on `/api/plugins/telemetry/...`); next cycle recovers without operator action. `/health` returns 503 on cold start for > 120 s without a cycle. `/admin/run-now` invoked twice in rapid succession does not run concurrently.

### Phase D — Integration bridge (outside service code)

| Gap | Action                                                                                                     |
|-----|------------------------------------------------------------------------------------------------------------|
|  8  | Edit widget JSON files (V5 Curtailment, Loss Attribution, Forecast vs Actual) to bind `potential_power` / `total_generation_expected_kwh`. Update `Dashboard.json` if it inlines widget settings. |
|  9  | Run `scripts/shared/set_active_power_unit.py` against the TB instance. Update `TELEMETRY_CONTRACT.md` with the widget scaling recipe. |

**Exit gate D.** The V5 Curtailment widget on the staging dashboard renders `active_power` (meter) and `potential_power` (pvlib) on the same chart, unit-consistent, for KSP.

### Phase E — Observability

| Gap | Files                                                                   |
|-----|-------------------------------------------------------------------------|
| 15  | `scripts/Pvlib-Service/app/api/forecast.py`, `scheduler.py` (counters)  |
| 19  | New `ops/watchdog_rule_chain.json`                                      |

**Exit gate E.** `/metrics` shows non-zero `pvlib_data_source_count{source="tb_station"}`. Import the rule chain into TB staging; kill the service; within 5 min, alarm `pvlib_service_stalled` fires.

### Phase F — Post-MVP

| Gap | Action                                                                                   |
|-----|------------------------------------------------------------------------------------------|
| 20  | New `scripts/Pvlib-Service/app/services/weekly_eval.py`, register scheduler job          |
| 21  | New `ops/ec2_userdata.sh`, update `docker-compose.yml` bind mount for `/run/pvlib/.env`  |
| 26  | New `scripts/shared/audit_tb_config.py`                                                  |
| 18  | (Deferred pending user decision.)                                                        |

---

## 4. Edge-Case Register (delta from prior plan)

| # | Case | Resolution | Phase |
|---|------|------------|-------|
| E1 | Plant has both weather station **and** explicit `weather_station_id` attribute that point to different devices | Trust the explicit attribute; log `WARN` with both device IDs so operator can reconcile. | A |
| E2 | WSTN device has `poa_key` set in config but pyranometer is broken (stuck value) | Station freshness gate (already present) rejects if `last_ts` too old, but **does not** catch stuck sensors. Add a `std(poa) < 1 W/m²` check over the window; treat as stale. | A/C |
| E3 | Plant lat/lon = 0,0 (unset) | Raise `no_location` — emit sentinels with reason. | A |
| E4 | `isPlantAgg` asset with mixed SolarPlant and non-solar children | Current BFS descends only into ASSETs; ignores non-plant DEVICEs. Roll-up sums only plants found (already dedup). Documented. | A |
| E5 | Parent has *no* descendants that produced data (all failed) | Emit -1 sentinels at the parent too. | A |
| E6 | Plant daily energy had 0 valid samples | Emit -1 daily. | B |
| E7 | Clock drift > 60 s | `start`/`end` derived from `datetime.now(utc)`; minute alignment of sentinels still correct. No action. | A |
| E8 | TB returns 401 mid-cycle (token invalidated by admin) | `_ensure_token` re-logins; retry decorator (Gap 22) catches the 401 and retries. | C |
| E9 | Solcast returns valid JSON but `estimated_actuals` is empty (off-coverage) | Treat as no-data; fall through to Tier-3. Cache the empty result for 30 min to avoid hammering. | C |
| E10 | Plant with `use_measured_poa=true` but station's `poa_key` not in latest telemetry (field removed) | Tier-1 becomes invalid via `_is_valid` check; fall to Tier-2/3. | A |
| E11 | `discover_plants` TTL expired mid-cycle | The new cycle gets a fresh list; in-flight cycle keeps the old list (stable). No action. | C |
| E12 | `ancestor_map` unbounded growth as tree grows | Cap discover cache entries; invalidate on TTL. Current tree = 50 assets, negligible. | C |
| E13 | Daily cron missed (service down at 00:05) | `misfire_grace_time=3600` on the daily job = cron fires up to 1h late on recovery. Document `/admin/run-daily?date=...` for manual backfill. | B |
| E14 | Parent ancestor that is itself `isPlant=true` (ambiguous: plant or aggregator?) | Resolution order: `isPlant` takes precedence — treat as leaf, don't recurse further, don't emit ancestor status. Flag in log. | A |
| E15 | Rule-chain watchdog runs while service is restarting (false positive) | `status="initializing"` is treated as ok for up to 2 min (Gap 16). Watchdog body check: alarm only when `now - last_cycle_finished_at > 600s` (10 min). | E |

---

## 5. Verification Plan

Automated (add under `scripts/Pvlib-Service/tests/`):

1. **Unit — Gap 1 / 10**: `test_config_parses_explicit_station_id`, `test_resolve_station_by_naming_prefix`.
2. **Unit — Gap 2**: `test_sentinel_records_cover_window`, `test_rollup_excludes_sentinels_from_sum`.
3. **Unit — Gap 5 / 6**: `test_ancestor_map_populated_for_multi_parent_plant`, `test_rollup_sums_in_memory_dfs`, `test_rollup_dedup_across_grandparents`.
4. **Unit — Gap 3 / 4**: `test_daily_integral_matches_trapezoidal`, `test_daily_cron_fires_at_local_midnight`.
5. **Unit — Gap 7**: `test_solcast_cache_hit`, `test_solcast_serves_stale_on_api_failure`.
6. **Unit — Gap 17**: `test_admin_run_now_delegates_to_scheduler_modify`.
7. **Unit — Gap 22**: `test_retry_on_502_does_not_on_401`.
8. **Unit — Gap 25**: `test_nightmask_with_poa_only_station`.

Integration (run against a staging TB instance):

9. `/admin/run-now` produces expected records on KSP_Plant.
10. `/pvlib/discover?root_asset_id={Windforce Plants}` returns KSP with ancestors `{Windforce Plants, Windforce Groundmount Plants}`.
11. Daily rollup after 24 h of 1-min cadence lands within 2 % of an external integration of the raw series.
12. Kill-service test: systemd restart; watchdog alarms within 10 min; resumes cleanly.

Sonnet 4.6 should add at least the unit tests before considering the phase complete.

---

## 6. One-Page Summary

| Phase | Gaps | Outcome                                                                                      |
|-------|------|----------------------------------------------------------------------------------------------|
| A     | 1, 2, 5, 6, 10 | Pvlib actually uses real weather stations; roll-ups reach every aggregation level; failures stamp -1. |
| B     | 3, 4, 14       | Daily energy key reflects the *day*, in *local* time.                                        |
| C     | 7, 11-13, 16, 17, 22, 23, 24, 25 | Lower TB API cost, safer concurrency, graceful cold-start/health.            |
| D     | 8, 9           | Widgets actually render `potential_power`; unit skew documented and populated.               |
| E     | 15, 19         | Ops can see per-tier health; watchdog fires on silent scheduler death.                       |
| F     | 20, 21, 26, (18 deferred) | Weekly A/B eval; secrets off disk; config audit tool.                             |

**Aggregate confidence after implementation: 92 %.** Weakest link is Phase D Gap 8 (widget edits) — it depends on dashboard conventions that are not fully code-owned; confirm render in staging before calling the work done.
