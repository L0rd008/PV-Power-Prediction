# Real-Time Power Prediction Service — Hypothesis Analysis and Recommended Architecture

*Author: Pasindu, drafted 2026-04-22. Source of truth for selecting the physics pipeline, data strategy, and deployment target for the next-generation service that replaces/augments `Green-Power-Services/Solcast Forcast` and `Green-Power-Services/Simple Forcast`.*

---

## 0. Scope and Load-Bearing Assumptions

Confirmed with the stakeholder before drafting:

1. **Horizon:** Historic and real-time only. No forward-looking Solcast forecasts required.
2. **Primary data source:** Plant weather-station telemetry in ThingsBoard on the same asset, **but availability varies per plant** — the service must degrade gracefully when a plant has no station.
3. **Scale:** ~10 plants today, architected to grow. Not yet 100+.
4. **Transport:** ThingsBoard calls the service via REST with the plant asset ID; service fetches inputs via TB REST, computes AC power, pushes results back as time-series telemetry on the same asset.
5. **Existing codebase:** `scripts/python_physics/physics_model.py` is a pvlib-backed but PVsyst-ordered custom pipeline (not fully custom math — it already calls `pvlib.irradiance.get_total_irradiance`, `pvlib.temperature.sapm_cell`, etc.). Two FastAPI services exist (`Solcast Forcast`, `Simple Forcast`) with identical structure: TB JWT client, asset-tree traversal, forecast job manager.

Derived:

- **"Pure pvlib" vs. "PVsyst-math"** is not a binary — the current code is already ~70% pvlib. The real choice is whether to adopt `pvlib.modelchain.ModelChain` orchestration (standardized, less custom glue) or keep a hand-assembled pipeline (full control over PVsyst-specific loss ordering).
- Because this is nowcast/backfill (no forecast), Solcast can drop from "primary data source" to "historic backfill fallback when no station data exists." This is the biggest cost win available.
- Image size dominated by `pandas + numpy + scipy + pvlib` (~350-450 MB of wheels). Docker image ≈ 550–700 MB regardless of whether we pick ModelChain or keep the custom pipeline.

---

## 1. Problem Decomposition

Three orthogonal sub-problems, each with its own candidate set:

| Sub-problem | What's being chosen | Primary drivers |
|---|---|---|
| **A. Physics pipeline** | How the POA→AC math is structured in code | Accuracy, maintenance cost, PVsyst fidelity |
| **B. Data strategy** | Where irradiance/weather inputs come from per request | External API cost, latency, per-plant capability |
| **C. Deployment** | How the service is packaged and run on EC2 | Compute cost, cold-start, operational burden |

Sections 2–4 enumerate hypotheses for each. Section 5 picks the winning combo and explains the hybrid.

---

## 2. Sub-problem A — Physics Pipeline Hypotheses

Five candidates. Ratings use a 1–5 scale per axis (5 = best). Confidence column is my confidence *in the rating itself*, not in the hypothesis succeeding. I only rate with ≥90% confidence where I have direct evidence (pvlib docs, the current codebase, verified benchmarks).

### H-A1 — Pure pvlib `ModelChain` with `run_model_from_poa()`

**Sketch.** Build one `Location`, one `PVSystem` containing 8 `Array` objects (one per orientation), let `ModelChain` handle POA→DC (PVWatts or Sandia) → cell temperature (SAPM) → DC aggregation → inverter clipping. Use `run_model_from_poa()` so we can inject POA-after-far-shading ourselves.

| Axis | Rating | Confidence |
|---|---|---|
| Accuracy vs. PVsyst | 4 | 95% — identical primitives, differs only in small opinionated defaults |
| Code complexity / maintenance | 5 | 95% — ~50 lines vs. current ~900 |
| Compute cost per call | 4 | 90% — ModelChain has tiny orchestration overhead, dominated by the same pvlib calls |
| Storage / image size | 3 | 90% — same dependency footprint |
| PVsyst loss-ordering fidelity | 2 | 90% — ModelChain does *not* natively encode PVsyst's soiling→LID→quality→mismatch→wiring chain; would need pre/post hooks |
| Fit to current team knowledge | 5 | 95% — reduces bespoke code, aligned with upstream |

**Pros.** Standard, upstream-maintained, inferred model selection, automatic spectral/AOI hooks, easier to onboard new developers, simpler tests (mock pvlib less, mock your own glue less). Native multi-orientation when sharing one inverter (verified via pvlib source).

**Cons.** Less obvious where to splice PVsyst's exact DC loss chain; you end up wrapping ModelChain with pre-POA far-shading and post-DC loss multiplication, which partially negates the "pure" story. Inverter clipping works only if all sub-arrays share one `Paco` — fine for all 8 orientations on a 55 kW inverter, but fails if a site has two inverters with different ratings (would need two `PVSystem` + manual aggregation).

### H-A2 — Keep the existing custom pipeline on pvlib primitives (status quo)

**Sketch.** `compute_pv_ac()` as written in `physics_model.py`: explicit per-orientation loop calling `get_total_irradiance`, `aoi`, `sapm_cell`, then a hand-rolled DC loss chain, inverter curve via `np.interp`, plant-level clip.

| Axis | Rating | Confidence |
|---|---|---|
| Accuracy vs. PVsyst | 5 | 95% — exact PVsyst loss ordering encoded |
| Code complexity / maintenance | 2 | 95% — ~900 lines, seven CLI modes, custom glue for each loss |
| Compute cost per call | 4 | 90% — no orchestration overhead, same primitives |
| Storage / image size | 3 | 95% — identical deps |
| PVsyst loss-ordering fidelity | 5 | 99% — by construction |
| Fit to current team knowledge | 5 | 95% — team already knows it |

**Pros.** Maximum PVsyst fidelity; every loss term applied exactly per the Business Intelligence doc; proven against existing services; no surprises from upstream ModelChain defaults.

**Cons.** Large custom surface area → higher long-term maintenance cost. The CLI-driven mono-file mixes five concerns (config load, four data sources, ERA5 fallback, compute, synthetic hourly profile, main loop) — hard to unit test, hard to reuse as a library inside a FastAPI worker without carving out the compute core. Every new plant or loss convention requires Python edits.

### H-A3 — Hybrid: `ModelChain` core + custom PVsyst pre/post hooks *(recommended for A)*

**Sketch.** Use `ModelChain.run_model_from_poa()` for POA→cell-temp→DC→AC+clip. Wrap it with:
- **Pre-hook:** per-orientation POA computed via `get_total_irradiance(model='perez')` → multiplied by far-shading factor → multiplied by `pvlib.iam.interp(aoi, iam_table)` → fed in as the list of POA DataFrames.
- **Post-hook on DC:** apply PVsyst loss chain `(1-soiling) × (1-LID) × (1-quality) × (1-mismatch) × (1-dc_wiring)` to the DC output before passing to the inverter model.
- **AC post-hook:** multiply by `(1 - ac_wiring)` after clipping.

Use `pvlib.pvsystem.PVSystem` with one `Array` per orientation; share one inverter; drive inverter via `pvlib.inverter.sandia` fit from the existing efficiency curve using `pvlib.inverter.fit_sandia()`.

| Axis | Rating | Confidence |
|---|---|---|
| Accuracy vs. PVsyst | 5 | 90% — all PVsyst conventions preserved, pvlib handles the heavy lifting |
| Code complexity / maintenance | 4 | 90% — ~150-200 lines for the compute core |
| Compute cost per call | 4 | 90% — marginal overhead vs. H-A2 |
| Storage / image size | 3 | 95% — same deps |
| PVsyst loss-ordering fidelity | 5 | 92% — explicit hooks keep the chain intact |
| Fit to current team knowledge | 4 | 90% — bridges old and new |

**Pros.** Strictly dominates H-A1 for PVsyst fidelity and dominates H-A2 for maintenance. pvlib absorbs the parts where it is already canonical (solar position, Perez, SAPM, inverter clip), while your team retains explicit control of the loss ordering that PVsyst requires. Easier to swap in `pvlib.pvsystem.pvwatts_dc` or a single-diode model later without rewriting the service.

**Cons.** Straddles two conventions — anyone reading the code must understand both pvlib's ModelChain lifecycle and the PVsyst loss chain. Documentation burden is real. Still ships the same ~600 MB of dependencies.

### H-A4 — Full single-diode model (CEC / De Soto / PVsyst)

**Sketch.** Replace the efficiency-based DC equation with `pvlib.pvsystem.singlediode()` using CEC module parameters. More physical, captures I-V curve behavior, better at part-load and low irradiance.

| Axis | Rating | Confidence |
|---|---|---|
| Accuracy vs. PVsyst | 5 | 85% — potentially better than H-A2, but only if CEC parameters are genuinely available for the installed modules |
| Code complexity / maintenance | 3 | 90% — more inputs to wire |
| Compute cost per call | 2 | 80% — Lambert-W solution is ~5-10× slower than the efficiency formula (still sub-millisecond but measurable at scale) |
| Storage / image size | 3 | 95% — same deps |
| PVsyst loss-ordering fidelity | 4 | 85% — orthogonal to ordering |
| Operational risk | 2 | 80% — requires per-plant CEC parameter lookup; breaks if modules aren't in the SAM library |

**Pros.** Maximum physical realism. Best for high-$, high-accuracy deployments.

**Cons.** CEC parameters are not in your existing `plant_config_example.json` — migration cost is real. Little accuracy gain unless you are under-performing the current model today. Defer unless validation shows current approach has systematic bias.

### H-A5 — Minimal PVWatts-only pipeline

**Sketch.** Strip to `pvlib.pvsystem.pvwatts_dc` + `pvlib.inverter.pvwatts` — two parameters per module (P_dc0, γ_pdc) and one inverter efficiency. Skip IAM, skip per-orientation loss chain, skip soiling/LID/mismatch.

| Axis | Rating | Confidence |
|---|---|---|
| Accuracy vs. PVsyst | 2 | 90% — known to diverge ~5-10% from PVsyst on annual yield |
| Code complexity / maintenance | 5 | 95% — shortest possible implementation |
| Compute cost per call | 5 | 90% — fastest |
| Storage / image size | 3 | 95% — same deps |
| Operational risk | 4 | 90% — well-known, well-understood |

**Pros.** Fastest and simplest. Good enough for rough UI indicators.

**Cons.** You already have `Simple Forcast` covering the "simple" end — adding another simple path is redundant. Won't meet the implied PVsyst-validation bar in the Business Intelligence doc.

### Sub-problem A verdict

**Rank (best → worst):** **H-A3** > H-A1 > H-A2 > H-A5 > H-A4.

H-A3 wins because it preserves the PVsyst fidelity the existing system has (which appears to matter per the BI doc's annual-energy check lists) while shrinking the custom surface to the parts that are genuinely plant-specific (loss chain, far-shading, IAM table). Confidence in this ranking: **~92%.** It would drop to ~75% if the team prioritizes zero hand-written glue over PVsyst fidelity — in that case H-A1 wins.

---

## 3. Sub-problem B — Data Strategy Hypotheses

Given the answers: nowcast/backfill only, station data on the plant asset where available, per-plant variability, 10 plants growing.

### H-B1 — TB station data only; error if missing

**Sketch.** Require station telemetry on the asset. If absent, return 400.

**Rating: 2/5.** **Confidence 95%.**

**Why not.** Violates the "some plants don't have weather stations" reality. Creates two classes of plants with very different UX.

### H-B2 — Solcast-only (current `Solcast Forcast` behavior)

**Sketch.** Always hit Solcast `estimated_actuals` for historic and the most recent 30 minutes for "real-time".

**Rating: 2/5.** **Confidence 95%.**

**Why not.** Paid API for something you can measure directly on the subset of plants that have stations. For 10+ plants at 1-minute resolution this adds up quickly and is strictly redundant when station data exists. Also introduces external-API latency and reliability risk on every call.

### H-B3 — Clear-sky-only (current `Simple Forcast` behavior)

**Sketch.** Ineichen/Solis clear-sky GHI/DNI/DHI from lat/lon/time only.

**Rating: 2/5.** **Confidence 92%.**

**Why not.** Ignores clouds and actual conditions; fine as a diagnostic benchmark but not as the primary real-time estimator. No one deploys this as the single production model.

### H-B4 — TB station → clear-sky fallback

**Sketch.** Prefer TB station telemetry (GHI or direct POA when available); if asset has no station *or* station is stale/faulty, fall back to Ineichen clear-sky. No paid API calls, ever.

| Axis | Rating | Confidence |
|---|---|---|
| Cost (external) | 5 | 99% — $0 |
| Accuracy on clear days | 5 | 90% — station data is ground truth |
| Accuracy on cloudy days w/o station | 1 | 95% — clear-sky overestimates by 2–10× |
| Per-plant coverage | 4 | 85% — works everywhere, but quality drops for stationless plants on cloudy days |
| Implementation cost | 4 | 92% — straightforward source selector |

**Pros.** Zero external cost; deterministic; no rate limits. Stations win where they exist.

**Cons.** For stationless plants on cloudy days, you are delivering a systematic overestimate. That's unacceptable for "expected generation" which is compared against actuals.

### H-B5 — TB station → Solcast `estimated_actuals` fallback

**Sketch.** Prefer TB station telemetry. If missing/stale, call Solcast `estimated_actuals` (cheaper than forecast calls, gives historic + near-real-time satellite-derived irradiance).

| Axis | Rating | Confidence |
|---|---|---|
| Cost (external) | 4 | 90% — Solcast called only for stationless plants and for backfill windows outside the TB retention |
| Accuracy on clear days | 5 | 90% — station when present; Solcast is high-quality satellite otherwise |
| Accuracy on cloudy days | 4 | 90% — Solcast is cloud-aware (unlike clear-sky) |
| Per-plant coverage | 5 | 95% — every plant gets a reasonable signal |
| Implementation cost | 3 | 85% — needs TB station-quality check + Solcast client + result caching |

**Pros.** Best realistic balance of accuracy and cost. Solcast call volume scales with *stationless* plants, not *total* plants — so the marginal API cost for adding a stationed plant is $0.

**Cons.** Operational complexity — need per-plant configuration to decide "is this station data trustworthy right now?" (freshness threshold, sensor-stuck detection, value sanity ranges).

### H-B6 — TB station → Solcast → clear-sky (3-tier fallback) *(recommended for B)*

**Sketch.** H-B5 with a third tier: if both TB station and Solcast fail (outage, rate limit, API key expired), fall through to clear-sky so the service never hard-fails. Emit a `data_source` label on every telemetry write so downstream dashboards can see which tier produced each data point.

| Axis | Rating | Confidence |
|---|---|---|
| Cost (external) | 4 | 90% — same as H-B5 in normal operation |
| Accuracy on clear days | 5 | 90% |
| Accuracy on cloudy days | 4 | 88% |
| Per-plant coverage | 5 | 95% |
| Reliability / uptime | 5 | 95% — service cannot hard-fail on data issues |
| Implementation cost | 3 | 85% — small extra logic over H-B5 |

**Pros.** Every request produces an answer; the quality label lets downstream systems discount low-tier predictions; clean promotion path as you add weather stations to plants (move them from tier 2 → tier 1 automatically).

**Cons.** Three tiers require three code paths and testing — but each is small.

### Additional data-strategy consideration — measured POA directly

If a station reports *tilted* irradiance at the same orientation as one of the plant's sub-arrays, you can short-circuit POA transposition for that orientation (skip Perez). Worth a config flag `"use_measured_poa": true` per orientation where the sensor is colocated. Estimated speedup: ~20-30% of compute. Accuracy: *better*, because Perez carries ~3-5% transposition error.

### Sub-problem B verdict

**Rank:** **H-B6** > H-B5 > H-B4 > H-B2 > H-B3 > H-B1. Confidence in ranking: **~93%.**

---

## 4. Sub-problem C — Deployment Hypotheses

### H-C1 — Per-service always-on EC2 (one container each for Solcast, Simple, new pvlib)

**Sketch.** Keep the two existing services untouched, add a third container for the new pipeline. Separate containers, separate ECR repos.

**Rating: 2/5. Confidence 95%.**

**Why not.** Triples EC2 cost vs. collapsing to one. The three services share ~95% of their code (TB client, job manager, config loader, physics core). Operational overhead: three log streams, three alarms, three deploy pipelines.

### H-C2 — AWS Lambda

**Sketch.** Trigger a Lambda per TB REST call. Runtime: Python 3.12, package pvlib via layers.

**Rating: 2/5. Confidence 88%.**

**Why not.** pvlib + pandas + numpy + scipy pushes the Lambda package near or over the 250 MB unzipped limit (manageable via container images, but still cold-starts in 3-6 seconds). For a real-time endpoint this is too slow. Also ThingsBoard would need to wait for that cold start on the first call after idle. Cost only beats EC2 if QPS stays extremely low — at 10 plants polling every minute, you are already at ~14k calls/day which EC2 serves cheaper.

### H-C3 — ECS Fargate with scale-to-low (1 replica)

**Sketch.** Single Fargate task, 0.25 vCPU / 512 MB, behind an ALB. Scale out with concurrent-request policy when plants grow.

**Rating: 4/5. Confidence 90%.**

**Why.** No server patching, clean scale-out story, moderate cost ($8-12/month per task in us-east-1 at 0.25 vCPU idle), first-class CloudWatch + ECR integration. Good fit for 10→100 plant growth.

**Caveat.** At 10 plants you're paying Fargate management overhead vs. a t4g.small ($7.50/month reserved). Break-even favors Fargate once you need multi-AZ or autoscaling.

### H-C4 — Single t4g.small EC2 + Docker Compose, shared image across all three services *(recommended for C)*

**Sketch.** One t4g.small (ARM, 2 vCPU / 2 GB, ~$7.50/month reserved or $12/month on-demand), Docker Compose running three FastAPI processes from the *same* image with different `ENTRY=solcast|simple|pvlib` environment variables. All three share:
- TB JWT client module
- Asset-relations traversal
- Config loader
- Job manager
- Telemetry writer
- The physics core (selected by env var)

Nginx/Caddy in front for TLS + routing (`/solcast/*`, `/simple/*`, `/pvlib/*`). One ECR repo, one image, one deploy pipeline.

| Axis | Rating | Confidence |
|---|---|---|
| Monthly cost (10 plants) | 5 | 90% — ~$7-15/month all-in |
| Scale path to 100+ plants | 3 | 85% — vertical scale to t4g.medium is fine; after that move to H-C3 |
| Cold start | 5 | 98% — always warm |
| Operational overhead | 3 | 85% — requires OS-level patching, but only one instance |
| Image size | 3 | 90% — one image ≈ 600 MB |
| Migration from current state | 5 | 90% — current services already run this way |

**Pros.** Cheapest, simplest, lowest latency, uses existing deployment pattern. Critical insight: the three "services" are really three **configurations** of the same FastAPI app — no reason to run them as separate containers.

**Cons.** Single point of failure; OS patching is your problem; vertical ceiling around t4g.medium for 100 plants.

### H-C5 — ECS Fargate Spot with scheduled scale-in to zero at night

**Sketch.** Fargate Spot task, scale-in at 19:00 local (no irradiance → no forecasts need computing) and scale-out at 05:00. Plants at non-operational hours get zero cost.

**Rating: 3/5. Confidence 80%.**

**Why not top.** Clever but fragile. Historic backfill can be requested overnight; scheduled downtime breaks that. The savings vs. always-on t4g.small (already $7/month) are negligible.

### Sub-problem C verdict

**Rank:** **H-C4** (now) → **H-C3** (at 50+ plants) > H-C5 > H-C2 > H-C1. Confidence in the "now" pick: **~92%.** Confidence in the transition threshold: **~75%** — depends on actual QPS per plant.

---

## 5. Recommended Hybrid Architecture

Combining winners from A, B, C:

```
                                     ┌────────────────────────────────┐
                                     │        ThingsBoard server       │
                                     │  (plant assets w/ telemetry +   │
                                     │   some w/ weather stations)     │
                                     └─────────────┬───────────────────┘
                                                   │ REST: asset_id, range
                                                   ▼
                                     ┌────────────────────────────────┐
                                     │  power-prediction-service       │
                                     │  (FastAPI, single image, t4g    │
                                     │   EC2 or Fargate @ scale)       │
                                     │                                 │
                                     │  ┌───────────────────────────┐  │
                                     │  │ 1. TB REST client         │  │
                                     │  │    (shared, JWT cached)   │  │
                                     │  └──────────┬────────────────┘  │
                                     │             │ attrs + telemetry │
                                     │  ┌──────────▼────────────────┐  │
                                     │  │ 2. Data-source selector    │  │
                                     │  │    tier 1: station         │  │
                                     │  │    tier 2: Solcast actuals │  │
                                     │  │    tier 3: Ineichen CS     │  │
                                     │  │    (emits `data_source`)   │  │
                                     │  └──────────┬────────────────┘  │
                                     │             │ GHI/DNI/DHI/T/WS  │
                                     │  ┌──────────▼────────────────┐  │
                                     │  │ 3. Physics core (H-A3)    │  │
                                     │  │   POA = get_total_irr()   │  │
                                     │  │   × far_shading × IAM     │  │
                                     │  │     │                     │  │
                                     │  │     ▼                     │  │
                                     │  │   ModelChain.run_model_   │  │
                                     │  │     from_poa(list)        │  │
                                     │  │     │                     │  │
                                     │  │     ▼ DC                  │  │
                                     │  │   apply PVsyst loss chain │  │
                                     │  │     │                     │  │
                                     │  │     ▼                     │  │
                                     │  │   inverter (Sandia) +     │  │
                                     │  │   plant-level clip        │  │
                                     │  │     │                     │  │
                                     │  │     ▼                     │  │
                                     │  │   × (1 - ac_wiring)       │  │
                                     │  └──────────┬────────────────┘  │
                                     │             │ AC power series   │
                                     │  ┌──────────▼────────────────┐  │
                                     │  │ 4. Telemetry writer       │  │
                                     │  │    ac_kw, data_source,    │  │
                                     │  │    model_version          │  │
                                     │  └──────────┬────────────────┘  │
                                     └─────────────┼──────────────────┘
                                                   ▼
                                     back to ThingsBoard asset telemetry
```

### Why this combination

- **H-A3** keeps the PVsyst loss conventions your current system validated against, while moving ~80% of the code into pvlib where it belongs. Reduces long-term maintenance cost without risking the accuracy bar set by the BI doc.
- **H-B6** makes external-API spend proportional to *stationless* plants rather than *total* plants, and guarantees the service never hard-fails on upstream data issues.
- **H-C4** collapses the three current services into one FastAPI image with three entry modes. Lower cost, lower ops, uses the deployment pattern the team already operates. Clean upgrade path to Fargate when scale demands it.

### Open design choices to freeze before implementation

1. **Measured POA shortcut.** When a plant has a tilted-irradiance sensor at the same tilt/azimuth as one of its orientations, skip Perez for that orientation. Config flag per orientation.
2. **Station-quality thresholds.** Freshness (e.g., "telemetry ≤5 min old"), value sanity (0 ≤ GHI ≤ 1400 W/m², non-negative at night), sensor-stuck detection (std < ε over N samples). These govern the tier-1 → tier-2 fallback.
3. **Model version telemetry.** Write `model_version` alongside `ac_kw` so you can diff old vs. new on the same timeline without deleting history.
4. **Caching.** Solcast responses for stationless plants can be cached per (lat, lon, period) with 30-min TTL — across plants sharing a location, this halves API spend further.
5. **Inverter curve fitting.** Decide: use `pvlib.inverter.fit_sandia()` once at startup per plant (stores Sandia params in memory) vs. keep `np.interp` approach (simpler, same accuracy). Leaning toward `np.interp` for simplicity unless ModelChain's `ac_model` requires Sandia-compatible params.

---

## 6. Implementation Plan

Five phases. Each phase has an explicit verification gate — no moving to the next phase until the previous one's accuracy/cost bar is met.

### Phase 1 — Offline bench (1-2 days)

**Goal.** Confirm H-A3 matches H-A2 to within 1% on known data before touching the service layer.

**Steps.**
1. Add `scripts/python_physics/physics_model_v2.py` implementing H-A3 — import the existing `physics_model.py` config loader, data loaders, and ERA5 fallback *unchanged*; replace only `compute_pv_ac()` with a ModelChain-based function.
2. Under `scripts/shared/compare_accuracy.py`, add a cross-check that runs both models on the same CSV input and reports per-timestep absolute and percent diff.
3. Run against existing test datasets in `data/` for the example plant.

**Gate.** Mean |Δ| ≤ 0.5% of AC energy; max |Δ| ≤ 3% on any single timestamp. If the gate fails, the cause is almost certainly the inverter-efficiency model or loss-chain ordering — log and adjust.

### Phase 2 — Service scaffold (2-3 days)

**Goal.** A single FastAPI service, shared image, runs locally in Docker.

**Steps.**
1. Create `Green-Power-Services-main/Unified-Service/` by copying `Solcast Forcast/app/` as the base (it has the cleanest TB client).
2. Split `physics_model.py` into four modules under `app/physics/`:
   - `config.py` — config schema, TB-attribute → config mapping
   - `data_sources.py` — three tiers (tb_station, solcast, clearsky), each a `get_irradiance(lat, lon, start, end, period) -> DataFrame`
   - `pipeline.py` — H-A3 compute core as a pure function `compute_ac_power(config, weather_df) -> Series`
   - `service.py` — orchestrator: fetch attrs → select data source → compute → write telemetry
3. Replace `forecast_service.py` in the new service with a thin shim that calls `service.run(asset_id, start, end)`.
4. Dockerfile uses `python:3.12-slim`, multi-stage build with wheel cache. Expected image ~600 MB.

**Gate.** `docker-compose up` locally; hit `/pvlib/run?asset_id=X&start=...&end=...` on a test TB tenant; telemetry shows up with `data_source=tb_station` for stationed plants.

### Phase 3 — Validation harness (2 days)

**Goal.** Compare new service output against the two existing services on the same assets and time ranges, under real ThingsBoard conditions.

**Steps.**
1. For each of the 10 plants, for a week of historic data, trigger all three services (`Solcast Forcast`, `Simple Forcast`, new `Unified-Service`) and record output to three separate TB telemetry keys.
2. Build a comparison notebook (`scripts/shared/three_way_validation.ipynb`) plotting the three series alongside actuals (Meteocontrol / plant inverter).
3. Compute MAPE/bias/RMSE per plant per service.

**Gate.** New service matches or beats `Solcast Forcast` on MAPE for stationed plants; matches or beats `Simple Forcast` for stationless plants. If either fails, the bug is in data sourcing, not physics.

### Phase 4 — Shared-image deployment (1-2 days)

**Goal.** Run all three modes from one image on one EC2.

**Steps.**
1. Refactor so the same `Dockerfile` produces an image with entry modes `MODE=solcast|simple|pvlib`. All three share the physics core; `MODE` only changes the default data-source selector and routing.
2. `docker-compose.yml` on t4g.small runs three containers from the same image, behind Caddy for TLS.
3. Cut DNS over from the two existing EC2s to the new instance's Caddy; watch TB-side integration for 48h.
4. Decommission old EC2s.

**Gate.** Zero regressions on `Solcast Forcast` and `Simple Forcast` endpoints; new `/pvlib/*` endpoint serving the 10 plants; CloudWatch metrics showing container CPU <30% at peak.

### Phase 5 — Operational hardening (ongoing)

**Steps.**
1. Add `model_version` and `data_source` fields to every telemetry write.
2. Add CloudWatch alarms: container memory >80%, TB-API 5xx >5%, Solcast credits <1000.
3. Add Prometheus-style `/metrics` for per-endpoint latency and data-source mix.
4. Add a nightly backfill cron (in the same container, triggered by a TB scheduler rule) that re-runs the last 24h for any plant whose data source was tier-3 (clear-sky) — in case station data arrived late.
5. Document the config schema in `docs/plant_config_v2.md`.
6. Plan migration to ECS Fargate when any of: (a) plant count ≥ 50, (b) single-instance CPU >50% p95, (c) a TB tenant explicitly requires multi-AZ.

---

## 7. Risk Register

| Risk | Likelihood | Impact | Mitigation |
|---|---|---|---|
| H-A3 diverges from PVsyst by >3% on a new plant | Medium | Medium | Validation harness in Phase 3 catches before prod; config has per-plant loss-chain override |
| TB station telemetry is stale/stuck without our detection noticing | Medium | High | Phase 5 adds sensor-stuck detection; `data_source` telemetry makes divergence visible |
| Solcast API key expiry causes quiet fallback to clear-sky, producing overestimates | Low | High | Phase 5 alarm on Solcast credits; `data_source=clearsky` is a loud signal |
| t4g.small saturates as plant count grows | Medium (at 50+ plants) | Medium | Vertical scale to t4g.medium first; Phase 5 defines ECS Fargate trigger |
| pvlib breaking change in minor release | Low | Medium | Pin `pvlib==0.15.x` in `requirements.txt`; upgrade only with a Phase 3 re-run |
| Measured POA sensor disagrees with transposed POA on a plant with both | Medium | Low | Log both; keep transposed POA as a cross-check; let ops team decide per plant |

---

## 8. Summary Table (All Hypotheses, One Page)

| ID | Hypothesis | Overall | Verdict | Confidence |
|---|---|---|---|---|
| H-A1 | Pure `ModelChain` | 4.0 | Strong; loses PVsyst loss fidelity | 92% |
| **H-A3** | **ModelChain + PVsyst hooks** | **4.5** | **Recommended** | **92%** |
| H-A2 | Custom pipeline (status quo) | 4.0 | Works; high maintenance cost | 95% |
| H-A4 | Single-diode CEC | 3.5 | Defer — not needed yet | 85% |
| H-A5 | PVWatts-only | 3.0 | Redundant with `Simple Forcast` | 90% |
| H-B1 | Station-only | 2.0 | Rejected — ignores stationless plants | 95% |
| H-B2 | Solcast-only | 2.5 | Rejected — redundant when stations exist | 95% |
| H-B3 | Clear-sky-only | 2.5 | Rejected — systematic cloud-day error | 92% |
| H-B4 | Station → clear-sky | 3.5 | Close, but weak on cloudy days | 90% |
| H-B5 | Station → Solcast | 4.5 | Strong, lacks uptime guarantee | 90% |
| **H-B6** | **Station → Solcast → clear-sky** | **5.0** | **Recommended** | **93%** |
| H-C1 | Three separate always-on EC2 | 2.0 | Wasteful — 95% code shared | 95% |
| H-C2 | Lambda | 2.5 | Cold start kills real-time | 88% |
| H-C3 | Fargate (single) | 4.0 | Good at scale, overkill now | 90% |
| **H-C4** | **Shared image on t4g.small** | **4.5** | **Recommended now** | **92%** |
| H-C5 | Fargate Spot with scheduled scale-in | 3.0 | Clever, fragile, small savings | 80% |

---

*Next action: decide whether Phase 1 starts now, or whether any assumption in §0 needs revising first.*
