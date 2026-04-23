# Pvlib-Service — AWS Deployment & ThingsBoard Integration Plan

*Author: Pasindu, drafted 2026-04-23. Source of truth for operating the `scripts/Pvlib-Service` FastAPI application against the live ThingsBoard asset tree: how it is scheduled, which plants it runs against, how results flow back into widgets, and how it is packaged for AWS. This is the successor to `architecture_hypothesis_analysis.md`; that document selected **H-A3 / H-B6 / H-C4** (hybrid pvlib pipeline, tiered data strategy, shared-image deployment), and this one covers everything downstream of "the service code is written."*

---

## 0. Context and Verified Assumptions

Confirmed by reading the current Pvlib-Service source:

1. **API surface is live.** `app/api/forecast.py` exposes `POST /pvlib/run-asset` (synchronous, single-plant), `POST /pvlib/start` (async, yesterday, plant + sub-tree), `POST /pvlib/start-with-date` (async, explicit range), `GET /pvlib/status/{job_id}`, `GET /pvlib/jobs`, `GET /health`. Every route is `async` under Uvicorn and uses `httpx.AsyncClient`.
2. **The physics pipeline runs per asset** via `forecast_service.process_single_asset()`: read `pvlib_config` attribute → `select_irradiance()` tiering (TB station → Solcast → clear-sky) → `compute_ac_power()` → write telemetry. There is a fallback that builds a `PlantConfig` from legacy flat attributes when `pvlib_config` is missing.
3. **Hierarchy traversal is BFS and data-driven** via `ThingsBoardClient.get_hierarchy_levels(root, target_level)` using `"Contains"` relations. `TARGET_LEVEL` is env-configurable (default 3). The deepest non-empty level is processed; parent levels receive **roll-up sums of daily energy**, but roll-up of the **AC power series is computed in memory and never written back to parents** (see `forecast_service.py` lines 404-446). This is a latent bug for any parent-asset widget expecting live power.
4. **Telemetry keys written:** `active_power_pvlib_kw`, `pvlib_daily_energy_kwh`, `pvlib_data_source`, `pvlib_model_version`.
5. **Telemetry keys the existing widgets read** (verified by grepping `M:\Documents\Projects\MAGICBIT\Widgets\`):
   - Curtailment vs Potential Power (V1-V5): `active_power` (actual) and `potential_power_profile` (a 96-slot attribute array, *not* a time series).
   - Forecast vs Actual Energy, Forecast Deviation Card (FDI): `total_generation` (actual MWh) and `forecast_p{50,75,90}_daily`.
   - Loss Attribution, Expected vs Actual Revenue: configurable datasource keys, no hardcoded pvlib key.
   - **No widget currently reads `active_power_pvlib_kw` or `pvlib_daily_energy_kwh`.** This is the single biggest integration gap.
6. **Unit skew:** some plants publish `active_power` in W, others in kW. The Pvlib-Service writes kW. Any widget that co-plots them needs a normalization contract.
7. **Job manager is in-process dict + asyncio.Lock.** Jobs are lost on restart; there is no Redis/DB. Fine for a single EC2 node, fatal on scale-out.
8. **Solcast cache is in-process** (30-min TTL, asyncio-guarded). Also fine on single node, must be externalized to scale.

The question is **not** "how do we build it?" — the service is built. The question is **"how do we run it every minute against the real tree, cheaply, without breaking existing dashboards?"**

---

## 1. Problem Decomposition

Five orthogonal sub-problems, each with hypothesis sets. A "hypothesis" here is a concrete architectural choice; I rate my **confidence in the rating itself** (i.e., how sure I am about its pros/cons for *this* system), not the confidence that the approach will work.

| Sub-problem | What's being chosen | Primary drivers |
|---|---|---|
| P1 | **Trigger pattern** for the 1-minute cadence | Cost, resilience, ops simplicity |
| P2 | **Plant selection / gating** — which assets run each cycle | Avoid wasted compute on incomplete configs; allow incremental rollout |
| P3 | **Parent aggregation model** — how rolled-up numbers reach isPlantAgg / Region / Nation nodes | Correctness at every hierarchy level, dedup when a plant is under multiple parents |
| P4 | **Widget/dashboard compatibility bridge** — reconciling the `active_power_pvlib_kw` key the service writes with the `active_power` / `potential_power_profile` keys widgets read | Zero rework on dashboards; single source of truth |
| P5 | **AWS packaging & deployment** — compute target, image size, secrets, observability | Monthly cost, blast radius, operability by a small team |

Edge-case matrix threaded through all five: missing weather station, missing P341 meter, W-vs-kW `active_power`, plants under multiple parents, `pvlib_config` missing or malformed, TB 5xx storm, JWT expiry mid-cycle, DST transitions, clock skew, partial computation on restart, Solcast quota exhaustion, cold-start latency, plant with daytime-zero sun (inverter night mode), dashboards spanning multiple plants with mixed units.

---

## 2. Hypotheses and Ratings

### P1 — Trigger pattern (1-minute cadence, ~10 plants today, growing)

**P1-A — ThingsBoard rule-chain `REST Call` node fires per plant, per minute.** The TB rule chain already knows which assets are plants (via `isPlant` attribute). A scheduled rule-chain node POSTs to `/pvlib/run-asset` every 60 s for each such asset.
- Pros: Fan-out already lives in the tool that defines "what is a plant"; TB retries on 5xx natively; no new scheduler to maintain; natural per-plant isolation (one failure doesn't block others); the `run-asset` endpoint is synchronous and returns quickly.
- Cons: Rule-chain scheduler is a coarse scheduler — tricky to stagger 10+ calls within a minute; no easy backpressure; debugging is in TB's UI, not logs you own; high coupling between platform config and service orchestration.
- **Rating: 6/10 · Confidence 95%.** Works well for few plants, but the rule-chain UI becomes a liability past ~25 plants and offers no centralized view of what ran.

**P1-B — Internal APScheduler inside the Pvlib-Service process** running a `@scheduler.scheduled_job('interval', minutes=1)` job that discovers plants, fans out `asyncio.gather(...)` across them, and writes back.
- Pros: One process, one source of truth, no extra infra; service self-contained; easy to stagger internally; trivial to add a `/admin/trigger-now` route; natural home for dedup, bounded concurrency, and backpressure; logs and metrics live where the code does.
- Cons: Single point of failure on one EC2 node (no HA unless you also deploy a second); scheduler runs regardless of whether the HTTP API has traffic, so a hung request can starve it; restart loses the current cycle.
- **Rating: 9/10 · Confidence 96%.** Best cost-to-complexity ratio at current scale; gracefully extends to 50+ plants with `asyncio.Semaphore` gating.

**P1-C — AWS EventBridge → Lambda (or ECS RunTask) every minute.** A managed cron event triggers a Lambda that either calls the service or runs the computation itself.
- Pros: Fully managed cron; auto-retries; per-invocation billing; scales horizontally without thinking.
- Cons: Lambda has a cold-start tax (pvlib import is ~3-5 s because of pandas/scipy); 15-minute max (fine here but a trap for backfills); Lambda container size cap makes pvlib wheels annoying; moves logic out of your FastAPI service and into AWS config; you lose the `MODE` single-image pattern.
- **Rating: 5/10 · Confidence 93%.** Viable but fights against the H-C4 "one image, three modes" decision already committed to.

**P1-D — Kubernetes CronJob / ECS Scheduled Task.** A 1-minute scheduled task pod runs a client that hits `/pvlib/start` for each plant.
- Pros: Robust managed scheduling, trivially observable, retries built in.
- Cons: Requires a container orchestrator; cost+complexity wildly out of proportion for 10 plants; starts a whole pod/task every minute.
- **Rating: 3/10 · Confidence 94%.** Correct answer for 100+ plants or multi-region; today it's overkill.

**P1-E — Hybrid: APScheduler as default, cron-driven HTTP ping as external heartbeat.** Internal scheduler drives the cadence; a TB rule chain (or EventBridge) pings `/health` every 5 minutes and alerts if no cycle has run in the last 2 minutes (watchdog). All adhoc triggers keep using `/pvlib/run-asset` as-is.
- Pros: Keeps single-process simplicity of P1-B; adds an external health pulse that catches the "scheduler is dead but HTTP is alive" case that pure P1-B misses; the rule-chain / EventBridge piece doesn't fan out — it just watches.
- Cons: Two things to configure instead of one.
- **Rating: 9.5/10 · Confidence 94%.** Captures the strength of P1-B while plugging its one real hole.

**Winner: P1-E** (internal APScheduler + external watchdog). This is the hybrid — P1-B alone is fine today but has a blind spot around silent scheduler failure, and the watchdog costs ~5 minutes of setup.

### P2 — Plant selection / gating

**P2-A — Hardcode the plant IDs** in a YAML or env var.
- Pros: Dead simple; unambiguous.
- Cons: Every new plant = redeploy; directly violates the user's "avoid hardcoding hierarchy" requirement.
- **Rating: 2/10 · Confidence 98%.**

**P2-B — Attribute-driven: walk the tree from a known root, include any asset where `isPlant == true`.** This is essentially how `get_hierarchy_levels` already finds leaves, but with an explicit filter on the `isPlant` attribute rather than "deepest non-empty level."
- Pros: Data-driven; new plants are picked up automatically as soon as they're tagged; `isPlant` is already set consistently in the tree; handles mixed-depth hierarchies (Nation→Region→Plant vs. Nation→Region→City→Plant) without a fixed `TARGET_LEVEL`.
- Cons: Needs an initial full-tree crawl per cycle (a few hundred API calls for 10 plants, cacheable); depends on `isPlant` being reliably set.
- **Rating: 9/10 · Confidence 96%.**

**P2-C — Attribute-driven + per-plant `pvlib_enabled` gate.** Same as P2-B but additionally require `pvlib_enabled == true` on the plant asset before computing. This gives a per-plant kill switch without redeploying.
- Pros: Everything P2-B plus safe rollout (enable one plant, validate, enable the rest); safe rollback of a single bad config; zero-downtime A/B.
- Cons: One more attribute to set; if operators forget to flip it, plants silently stay out.
- **Rating: 9.5/10 · Confidence 95%.**

**P2-D — Opportunistic gating: compute only if the plant has (a) a parseable `pvlib_config`, and (b) Tier-1 or Tier-2 data available at the current timestamp.** Skip otherwise, log a reason.
- Pros: Self-organizing; the service computes wherever it meaningfully can and degrades silently on the rest; matches the user's "only KSP has complete attributes today" reality.
- Cons: Operators can't easily see why a plant is skipped without reading logs; "silent degradation" is a double-edged sword.
- **Rating: 7.5/10 · Confidence 92%.**

**Winner: P2-C + P2-D composed.** The `isPlant` + `pvlib_enabled` filter defines the *eligibility set*; inside that set, P2-D handles the "config incomplete / station dead" fallbacks so you don't have a binary gate. This is the hybrid: explicit opt-in at the plant level, graceful degradation inside.

### P3 — Parent aggregation model (the subtle one)

**P3-A — Write parent totals computed by the service** (current code does this for daily energy, *should* do it for AC power too).
- Pros: Widgets pointed at a region/national asset get numbers without any dashboard logic; consistent with how `total_generation` is rolled up.
- Cons: Double-counting risk when a plant is attached under two parents (user explicitly flagged this); parent telemetry is computed-not-measured which confuses mental models; data-source tag (`tb_station` vs `solcast`) becomes meaningless at parent level.
- **Rating: 6/10 · Confidence 93%.** Useful but dangerous without dedup.

**P3-B — Dedup-aware roll-up.** Before summing child records at a parent, build the set of *unique* child plant IDs reachable from that parent; sum each unique plant exactly once. Use a "canonical parent" attribute (`primary_parent_id`) on plants with multi-parent relations to tie-break for per-region accounting when that's required.
- Pros: Correct under the real hierarchy; preserves the convenience of P3-A; `primary_parent_id` is optional — without it, totals simply appear under every parent but are still per-plant-unique.
- Cons: Requires the service to detect multi-parenthood (a plant reachable from two paths in the BFS); one more attribute to document.
- **Rating: 9.5/10 · Confidence 93%.**

**P3-C — Don't aggregate in the service; let ThingsBoard rule chains or the widgets do it.**
- Pros: Service becomes simpler; aggregation logic lives where the business rules are.
- Cons: Every dashboard now carries aggregation logic; the existing roll-up code becomes dead weight; widgets differ in how they sum.
- **Rating: 4/10 · Confidence 94%.**

**P3-D — Write per-plant records to parents as a *keyed map* (`pvlib_plant_contributions: {KSP: 12.3, SOU: 8.1, ...}`) so widgets can sum themselves and reason about provenance.**
- Pros: Fully transparent; a widget can filter out a plant or highlight outliers; no double-counting because nothing is summed server-side.
- Cons: Complicates the telemetry schema; TB doesn't love large JSON-blob telemetry values; widgets would need to be updated.
- **Rating: 5/10 · Confidence 88%.**

**Winner: P3-B.** Correctness (no double-counting) is non-negotiable; service-side aggregation is already ~half-built; P3-D is a nice v2 when we want drill-down.

### P4 — Widget/dashboard compatibility bridge

This is the most urgent gap: the service writes `active_power_pvlib_kw` but **no existing widget reads that key.** Four resolutions exist.

**P4-A — Rename service output to match widgets: write `potential_power` (kW) and `total_generation_expected` (kWh).** Keep `pvlib_data_source` + `pvlib_model_version` as side-channel metadata.
- Pros: Widgets "just work" the moment telemetry flows; no dashboard changes; discoverable (keys self-document their purpose); aligns with the existing `potential_power_profile` attribute naming.
- Cons: `active_power` in TB already means *measured* on some plants — we must pick a distinct key (`potential_power`, *not* `active_power`) to avoid clobbering meter data.
- **Rating: 9/10 · Confidence 95%.**

**P4-B — Keep pvlib keys, add a ThingsBoard rule chain that aliases `active_power_pvlib_kw` → `potential_power`** on ingest.
- Pros: No service change; rule chain carries the translation.
- Cons: Doubles telemetry storage (both keys written); adds a platform dependency for the "key contract"; easy to break when someone edits the rule chain.
- **Rating: 6/10 · Confidence 93%.**

**P4-C — Update every widget to consume `active_power_pvlib_kw`.**
- Pros: Zero service-side changes.
- Cons: ~6 widgets × test cycles; breaks the "dashboards are stable, data flows update" pattern; actively fights the user's instruction ("widgets/dashboards must remain compatible").
- **Rating: 3/10 · Confidence 96%.**

**P4-D — Dual-write: service writes *both* `active_power_pvlib_kw` (for internal/ops dashboards) and `potential_power` (for the customer-facing ones).** Daily energy written as `total_generation_expected_kwh` alongside `pvlib_daily_energy_kwh`.
- Pros: Zero dashboard work; keeps the ops-diagnostic keys for debugging; `pvlib_data_source` remains attached to the original keys so widget operators who opt-in later can see provenance.
- Cons: Twice the TB writes → roughly 2× the telemetry I/O. At ~1440 records/day/plant × 10 plants × 2 keys = ~29k writes/day, which is trivially within TB's capacity.
- **Rating: 9.5/10 · Confidence 94%.**

**Winner: P4-D (dual-write).** Writes are cheap, dashboard stability is priceless, and the keys encode both the *what* (potential power) and the *how* (pvlib model version). This is the hybrid.

### P5 — AWS packaging and deployment

**P5-A — EC2 `t4g.small` (ARM, 2 vCPU, 2 GB), single node, Docker Compose for process supervision.**
- Pros: ~$12/mo for 24×7; pvlib wheels ship for ARM (`aarch64`); Compose restart policy gives you PID-1 supervision; SSH for debugging; no platform to learn.
- Cons: Single-node; you own patching; scale-up = vertical only.
- **Rating: 9/10 · Confidence 96%.**

**P5-B — ECS Fargate (one task, 0.5 vCPU, 1 GB).**
- Pros: No host patching; ECS handles restarts; easy to add a second task for HA later.
- Cons: ~$15/mo minimum; CloudWatch-only logs unless you add infra; cold-start-on-restart is noticeable; APScheduler inside Fargate needs careful health-check config so ECS doesn't kill a long-running computation.
- **Rating: 8/10 · Confidence 94%.**

**P5-C — EC2 + systemd directly (no Docker).**
- Pros: Lowest overhead; simplest dev-to-prod parity if you develop in venv.
- Cons: Fights the H-C4 "one image, three MODEs" decision; dependency drift; harder to reproduce.
- **Rating: 5/10 · Confidence 92%.**

**P5-D — Lambda + EventBridge (container image).**
- Pros: Pay per minute of execution; no idle cost.
- Cons: Cold starts (~3-5 s for pvlib imports, cumulative across 10 plants/min = measurable latency tax); 10 GB container size limit (OK but getting tight with pandas+scipy+pvlib); shared Solcast cache needs to move to ElastiCache or be per-plant TTL-deduped (breaks the current H-B6 assumption); the APScheduler hypothesis is incompatible with Lambda, so this forces P1-C.
- **Rating: 5/10 · Confidence 92%.**

**Winner: P5-A (EC2 t4g.small + Docker Compose).** Cheapest, debuggable, single responsibility. Move to P5-B only when either (a) you genuinely want ECS patterns elsewhere, or (b) HA becomes a hard requirement.

---

## 3. Selected Hybrid ("H-D1" — Deployment Hybrid #1)

Combining the winners:

| # | Choice | Source |
|---|---|---|
| D1-Trigger | APScheduler inside the service + external TB rule-chain watchdog pinging `/health` every 5 min | P1-E |
| D1-Selection | `isPlant == true` **and** `pvlib_enabled == true`; within that set, degrade gracefully on config/station gaps | P2-C ∘ P2-D |
| D1-Aggregation | Service-side roll-up with dedup on multi-parent plants; optional `primary_parent_id` tie-break | P3-B |
| D1-KeyContract | Dual-write: `potential_power` (kW) + `active_power_pvlib_kw` (kW), and `total_generation_expected_kwh` + `pvlib_daily_energy_kwh` | P4-D |
| D1-Compute | EC2 t4g.small, Docker Compose, one container | P5-A |

**Aggregate confidence in the hybrid: 93%.** The weakest link is P3 (dedup logic is novel code, not yet written) — watch that carefully in Phase 3.

---

## 4. Implementation Plan (Phased)

Each phase has an exit gate. Do not proceed to Phase N+1 until Phase N's gate passes.

### Phase 1 — Widget Contract & Key Rename (low risk, unblocks everything)

Write a `TELEMETRY_CONTRACT.md` into the Pvlib-Service repo that documents the five canonical keys the service writes. In `forecast_service.py`:

- Add constants `KEY_POTENTIAL_POWER = "potential_power"` and `KEY_DAILY_ENERGY_EXPECTED = "total_generation_expected_kwh"`.
- In `_build_ac_telemetry`, write **both** `active_power_pvlib_kw` and `potential_power` per record.
- In `_build_daily_telemetry`, write **both** `pvlib_daily_energy_kwh` and `total_generation_expected_kwh`.

**Unit hardening:** add a `_normalize_power_kw()` helper that converts any measured `active_power` read from TB for comparisons/diagnostics into kW using a per-plant `active_power_unit` attribute (`"W"` | `"kW"`) with default `"kW"`. This attribute is advisory only; the service itself always writes kW. Write an `ops_expected_unit` key alongside records (`"kW"`) so widgets can verify what they're plotting.

**Gate:** read `potential_power` from ThingsBoard for KSP_Plant and plot it in the V2 Curtailment widget with no widget code changes. If it renders on the chart next to `active_power`, pass.

### Phase 2 — Plant Discovery & `pvlib_enabled` Gate

Replace `get_hierarchy_levels` with a new `discover_plants(root_asset_id)` method in `ThingsBoardClient`:

- BFS from `root_asset_id` following `Contains` relations, without a depth limit, accumulating all assets whose `SERVER_SCOPE` attributes include `isPlant == true`.
- De-duplicate by asset UUID (same plant under two parents → counted once as a plant).
- For each plant, record the full list of paths (so we know every parent chain for later roll-up).
- Return `(plants: List[PlantRef], parent_paths: Dict[plant_id, List[List[str]]])`.

Add `pvlib_enabled` check: skip plants where the attribute is missing or falsy. Log a `skipped: pvlib_enabled=false` record so operators see what's gated out.

Keep `get_hierarchy_levels` for backwards compatibility with `/pvlib/start-with-date` (historical backfill jobs still use level-based traversal), but switch `/pvlib/start` and the new scheduler to `discover_plants`.

**Gate:** `GET /pvlib/discover` (new debug endpoint) returns the list of enabled plants and their parent paths; it matches what the operator expects looking at the TB UI.

### Phase 3 — Dedup-Aware Roll-Up

Modify `ForecastService.run_forecast_for_main_asset`:

- After processing all leaf plants, build `parents_of_plant: Dict[plant_id, Set[parent_id]]` from `parent_paths`.
- For each parent, sum contributions from the **unique set of plants under it**, not from the BFS-expanded list (which can contain duplicates).
- Write *both* rolled-up AC power (`potential_power` / `active_power_pvlib_kw`) and daily energy (`total_generation_expected_kwh` / `pvlib_daily_energy_kwh`) to parent assets. (The current code only writes daily energy to parents.)
- For plants with a `primary_parent_id`, the roll-up still sums into every parent the plant is under, but emit a side telemetry `pvlib_primary_parent_contribution` on the primary parent only — so a widget that wants a non-double-counted regional total has an unambiguous key to use.

**Gate:** unit test with a synthetic tree where `SOU_Plant` is reachable from both `South_Region` and `Coastal_Region`. Verify that the sum over all parents doesn't double-count SOU, and that `SOU.potential_power` written to its own asset equals the sum actually used.

### Phase 4 — APScheduler + Bounded Concurrency

Add `apscheduler` to `requirements.txt`. In `app/services/scheduler.py`:

```
scheduler = AsyncIOScheduler(timezone=settings.TZ_LOCAL)

@scheduler.scheduled_job('interval', minutes=1, misfire_grace_time=30)
async def run_cycle():
    async with ThingsBoardClient() as tb:
        plants, paths = await tb.discover_plants(settings.ROOT_ASSET_ID)
        sem = asyncio.Semaphore(settings.MAX_CONCURRENT_PLANTS)  # default 5
        async def bounded(plant_id):
            async with sem:
                return await process_single_asset(plant_id, tb, *_last_minute_window())
        results = await asyncio.gather(*(bounded(p.id) for p in plants), return_exceptions=True)
        # then roll up to parents using paths (Phase 3)
```

Start the scheduler in `main.py`'s `lifespan()`. Expose `/admin/run-now` for manual triggering.

**Window:** each cycle processes `[now - 90s, now - 30s]` so that TB has time to receive the station telemetry at the current minute before the service reads it. Tune via `settings.READ_LAG_SECONDS`.

**Backpressure & collision:** use APScheduler's `max_instances=1` on the cycle job so a slow cycle doesn't overlap itself. If a cycle takes > 60 s, the next cycle is *skipped* (not queued) — log a warning. Target p95 cycle time < 45 s for 10 plants at concurrency 5.

**Gate:** run for 2 hours on staging. Verify (a) every plant gets exactly one record per minute per key; (b) no duplicate `ts` values; (c) p95 cycle time < 45 s; (d) no memory leak (RSS stable over 2 h).

### Phase 5 — Watchdog and Observability

Add `GET /health` enhancements: return `{status, last_cycle_finished_at, last_cycle_duration_ms, plants_in_last_cycle, failures_in_last_cycle}`. Mark unhealthy (HTTP 503) if `now - last_cycle_finished_at > 120s`.

Add a TB rule chain: scheduled every 5 minutes, GET `/health`, if 5xx or body shows `last_cycle_finished_at > 2m ago`, fire a TB alarm `pvlib_service_stalled`. This is the external watchdog from P1-E.

Add Prometheus-compatible metrics at `/metrics` (gauge `pvlib_cycle_duration_seconds`, counter `pvlib_plant_failures_total{plant,reason}`, gauge `pvlib_data_source_count{source}`). If you don't want Prometheus, structured log lines (`{"event": "cycle_complete", "duration_ms": ..., "plants": ...}`) parsed by CloudWatch Insights are equivalent.

**Gate:** kill the service mid-cycle. Within 5 minutes, the TB alarm fires. Restart it. Alarm clears within one cycle.

### Phase 6 — AWS Packaging

Dockerfile (two-stage, python:3.12-slim, `--platform=linux/arm64`):

- Stage 1: `pip wheel -r requirements.txt -w /wheels --prefer-binary` — avoids rebuilding scipy/numpy from source.
- Stage 2: copy `/wheels`, `pip install --no-index --find-links=/wheels`; `CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]`. Keep `--workers 1` because APScheduler in-process can't handle multiple workers.

docker-compose.yml on the EC2 host, with `restart: unless-stopped` and a bind-mount for `.env` from SSM Parameter Store (pulled via a systemd one-shot on boot, not baked into the image).

Secrets:
- TB credentials → SSM Parameter Store `SecureString`, read on boot.
- Solcast API key → SSM `SecureString`.
- CloudWatch agent ships stdout logs to `/aws/ec2/pvlib-service`.

Security group: inbound 8000 from TB VPC only, 22 from a bastion or SSM Session Manager, egress 443 to TB host + `api.solcast.com.au`.

**Gate:** full cycle runs end-to-end on the EC2 node; a synthetic widget plot on a staging dashboard shows live `potential_power` for KSP_Plant.

### Phase 7 — Rollout

Plant-by-plant enablement using `pvlib_enabled`:

1. Week 1: KSP only. Monitor for 48 h. Compare `potential_power` vs meter `active_power` — expect > 0.85 correlation on sunny hours.
2. Week 2: add plants with weather stations (SOU, SSK).
3. Week 3: add plants *without* weather stations (they'll run on Solcast → clear-sky tiers). Watch the `pvlib_data_source` distribution — if `clearsky` ever exceeds 15 % of cycles for a plant with a station, investigate station freshness gates.
4. Week 4: enable `isPlantAgg` roll-ups. Verify widgets at region level match the sum of the plants beneath.

---

## 5. Edge-Case Register

| Case | Resolution | Phase | Confidence |
|---|---|---|---|
| Plant has no `pvlib_config` and no legacy flat attrs | `skip` with `config_parse_error`; logged in job errors; `pvlib_enabled` should be false for such plants | P2 | 97% |
| Plant has weather station but station went stale (> `freshness_minutes`) | `select_irradiance` already falls back to Tier 2 → Tier 3; `pvlib_data_source` records actual tier used | P4 (existing) | 95% |
| Measured `active_power` in W on plant A, kW on plant B | Service only *writes* kW; widgets that co-plot must read `ops_expected_unit` or use `active_power_unit` attribute on the plant; recommend standardizing all meters to kW in a separate Phase-0 cleanup | P1 | 92% |
| Plant reachable via two parent paths | P3-B dedup — each plant counted once per cycle; parent roll-up uses unique child set | P3 | 93% |
| `isPlantAgg` asset has *no* children (orphaned aggregate) | `discover_plants` returns empty children for that parent; roll-up writes nothing; warning log | P2/P3 | 96% |
| JWT expires mid-cycle | `_ensure_token` already auto-refreshes at 5-min buffer; single-cycle failures retry next minute | - | 98% |
| TB 5xx storm | `httpx` + exponential backoff wrapper (add in Phase 4); per-plant failures don't abort other plants due to `asyncio.gather(..., return_exceptions=True)` | P4 | 94% |
| Solcast quota exhausted | Tier 3 (clear-sky) kicks in; `pvlib_data_source = "clearsky"` flags downgraded quality; alert if > 15 % cycles on clearsky | P4 (existing) | 95% |
| DST transition (Asia/Colombo has no DST, but a future site might) | Use `ZoneInfo` consistently (already done); never use naive datetimes in windows | All | 97% |
| Clock skew on EC2 | Enable chrony on EC2; monitor `ntpstat`; cycle windows use a 30-90 s read-lag buffer that absorbs most skew | P6 | 95% |
| Service restart mid-cycle | APScheduler's `misfire_grace_time=30` catches the next window; in-memory jobs are lost (acceptable for 1-min-granularity data — the next cycle replaces them) | P4 | 93% |
| Widgets co-plotting on mixed-unit plants | Phase 1 dual-write means widgets read `potential_power` (always kW); any widget reading raw `active_power` in W needs client-side ×0.001 via dashboard config — document this in `TELEMETRY_CONTRACT.md` | P1 | 90% |
| Plant at daytime-zero (inverter offline) | Pipeline yields `ac_kw = 0` from zero-irradiance path; still written so the widget doesn't show a gap; `pvlib_data_source` may show `tb_station` if station is still reporting | - | 96% |
| Cold-start backfill after deploy | First cycle only covers `[now-90s, now-30s]`; operators can POST `/pvlib/start-with-date` with a wider range for catch-up | P4 | 95% |
| Multiple EC2 nodes (future HA) | Convert `JobManager` + Solcast cache to Redis; guard `run_cycle` with a Redis lease (only one node runs the scheduler); dual-write remains per-cycle not per-node | Future | 88% |
| Plants created after deploy | `discover_plants` runs every cycle — plants appear automatically when `isPlant=true` + `pvlib_enabled=true` are set | P2 | 97% |
| Dashboard widget references retired key | Grep `TELEMETRY_CONTRACT.md` before removing any key; keep the old `pvlib_*` keys for ≥ 90 days after Phase 1 before retiring | P1 | 94% |

---

## 6. Cost and Capacity Envelope

Single EC2 t4g.small on-demand: $0.0168/h ≈ **$12.10/mo**. EBS 20 GB gp3: ~$1.60/mo. CloudWatch logs (5 GB/mo ingested at DEBUG-trimmed INFO): ~$2.50/mo. SSM Parameter Store: free tier. Elastic IP while attached: free. **Total ≈ $16/mo** steady-state for 10 plants at 1-minute cadence.

ThingsBoard writes per day, assuming 10 plants, 1440 records/day each, 5 telemetry keys per record (dual-write + metadata): ~72k writes/day, well inside community-edition TB throughput.

Solcast API calls: with the existing 30-min in-process cache, at most 48 calls/day/plant-without-station. For 3 such plants, ~144/day — comfortably inside the free-tier 10 calls/hour/site quota if spread evenly.

At **50 plants**, re-evaluate: cycle time scales linearly at concurrency 5, so p95 would be ~225 s — breaks the 1-minute budget. Fix by raising `MAX_CONCURRENT_PLANTS` to 15 (EC2 t4g.medium, ~$30/mo) or sharding across two t4g.small nodes with a Redis-backed scheduler lease.

---

## 7. One-Page Summary

| Decision | Chosen | Confidence |
|---|---|---|
| Cadence driver | APScheduler + external `/health` watchdog | 94% |
| Plant selection | `isPlant && pvlib_enabled`, graceful degradation inside | 95% |
| Parent aggregation | Service-side, dedup on multi-parent paths | 93% |
| Widget compatibility | Dual-write `potential_power` + `active_power_pvlib_kw` | 94% |
| Compute target | EC2 t4g.small + Docker Compose | 96% |
| Overall hybrid (H-D1) | Phases 1 → 7 with explicit gates | 93% |

**Biggest risk:** dedup roll-up (P3-B) is the only new logic with non-trivial correctness requirements. Mitigate with synthetic-tree unit tests in Phase 3.

**Smallest-effort, biggest-impact step:** Phase 1 (dual-write the key). One commit; every existing widget immediately becomes pvlib-aware without dashboard changes.
