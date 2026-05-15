"""
Pvlib forecast API routes.

Endpoints:
  POST /pvlib/run-asset          — synchronous single-plant compute
  POST /pvlib/start              — async fleet job (current cycle window)
  POST /pvlib/start-with-date    — async fleet job (explicit date range)
  GET  /pvlib/status/{job_id}    — job status
  GET  /pvlib/jobs               — all jobs list
  GET  /pvlib/discover           — debug: list discovered plants
  GET  /health                   — enhanced health with last_cycle info
  POST /admin/run-now            — trigger immediate cycle
  GET  /metrics                  — Prometheus-compatible text metrics
  POST /admin/run-revenue-monthly   — compute LKR revenue for a calendar month
  POST /admin/run-revenue-yearly    — compute LKR revenue for a calendar year
  POST /admin/run-revenue-backfill  — backfill all months + years for one plant
  POST /admin/run-autoonboard       — run zero-touch onboarding for pending plants
"""
from __future__ import annotations

import time
from datetime import datetime, timedelta, timezone
from typing import Optional

from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel

from app.config import settings
from app.services.job_manager import job_manager
from app.services.scheduler import (
    cycle_state,
    run_cycle_now,
    run_daily_now,
    run_loss_rollup_now,
    run_pvalue_job_now,
    run_today_partial_now,
)

router = APIRouter()


# ── Request / Response models ───────────────────────────────────────────────

class RunAssetRequest(BaseModel):
    asset_id: str
    start: Optional[datetime] = None
    end: Optional[datetime] = None


class StartJobRequest(BaseModel):
    root_asset_id: Optional[str] = None   # overrides TB_ROOT_ASSET_IDS if provided


class StartWithDateRequest(BaseModel):
    start: datetime
    end: datetime
    root_asset_id: Optional[str] = None


# ── Routes ──────────────────────────────────────────────────────────────────

@router.post("/pvlib/run-asset")
async def run_asset(req: RunAssetRequest):
    """Synchronous single-plant compute. Returns result immediately."""
    from app.services.thingsboard_client import ThingsBoardClient
    from app.services.forecast_service import ForecastService

    now = datetime.now(timezone.utc)
    end = req.end or now - timedelta(seconds=settings.READ_LAG_SECONDS)
    start = req.start or end - timedelta(seconds=settings.READ_WINDOW_SECONDS)

    async with ThingsBoardClient(
        settings.TB_HOST, settings.TB_USERNAME, settings.TB_PASSWORD
    ) as tb:
        svc = ForecastService(tb, solcast_api_key=settings.SOLCAST_API_KEY)
        result = await svc.process_single_asset(req.asset_id, start, end)

    result_dict = result.to_dict()
    if result.status not in ("ok", "no_data"):
        raise HTTPException(status_code=500, detail=result_dict)
    return result_dict


@router.post("/pvlib/start")
async def start_fleet_job(req: StartJobRequest):
    """Start async fleet job for the current 1-minute window."""
    import asyncio
    from app.services.thingsboard_client import ThingsBoardClient
    from app.services.forecast_service import ForecastService

    root_ids = [req.root_asset_id] if req.root_asset_id else settings.root_asset_ids
    if not root_ids:
        raise HTTPException(status_code=400, detail="No root_asset_id configured")

    now = datetime.now(timezone.utc)
    end = now - timedelta(seconds=settings.READ_LAG_SECONDS)
    start = end - timedelta(seconds=settings.READ_WINDOW_SECONDS)

    job = await job_manager.create()

    async def _run():
        async with ThingsBoardClient(
            settings.TB_HOST, settings.TB_USERNAME, settings.TB_PASSWORD
        ) as tb:
            svc = ForecastService(tb, solcast_api_key=settings.SOLCAST_API_KEY)
            await svc.run_fleet_cycle(root_ids, start, end,
                                      max_concurrent=settings.MAX_CONCURRENT_PLANTS,
                                      job_mgr=job_manager)

    asyncio.create_task(_run())
    return {"job_id": job.job_id, "status": "started", "window_start": start.isoformat(), "window_end": end.isoformat()}


@router.post("/pvlib/start-with-date")
async def start_with_date(req: StartWithDateRequest):
    """Start async fleet job for an explicit date range (backfill)."""
    import asyncio
    from app.services.thingsboard_client import ThingsBoardClient
    from app.services.forecast_service import ForecastService

    root_ids = [req.root_asset_id] if req.root_asset_id else settings.root_asset_ids
    if not root_ids:
        raise HTTPException(status_code=400, detail="No root_asset_id configured")

    job = await job_manager.create()

    async def _run():
        async with ThingsBoardClient(
            settings.TB_HOST, settings.TB_USERNAME, settings.TB_PASSWORD
        ) as tb:
            svc = ForecastService(tb, solcast_api_key=settings.SOLCAST_API_KEY)
            await svc.run_fleet_cycle(root_ids, req.start, req.end,
                                      max_concurrent=settings.MAX_CONCURRENT_PLANTS,
                                      job_mgr=job_manager)

    asyncio.create_task(_run())
    return {"job_id": job.job_id, "status": "started", "start": req.start.isoformat(), "end": req.end.isoformat()}


@router.get("/pvlib/status/{job_id}")
async def get_status(job_id: str):
    job = await job_manager.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail=f"Job {job_id} not found")
    return job.to_dict()


@router.get("/pvlib/jobs")
async def list_jobs():
    jobs = await job_manager.list_all()
    return [j.to_dict() for j in jobs]


@router.get("/pvlib/discover")
async def discover_plants(root_asset_id: Optional[str] = Query(None)):
    """Debug endpoint: return list of pvlib-enabled plants discovered from root(s)."""
    from app.services.thingsboard_client import ThingsBoardClient

    root_ids = [root_asset_id] if root_asset_id else settings.root_asset_ids
    if not root_ids:
        raise HTTPException(status_code=400, detail="No root_asset_id configured")

    async with ThingsBoardClient(
        settings.TB_HOST, settings.TB_USERNAME, settings.TB_PASSWORD
    ) as tb:
        plants, parent_map = await tb.discover_plants(root_ids)

    return {
        "plant_count": len(plants),
        "plants": [
            {
                "id": p.id,
                "name": p.name,
                "parent_count": len(p.parent_ids),
                "parent_ids": list(p.parent_ids),
                "ancestor_count": len(parent_map.get(p.id, set())),
                "ancestor_ids": list(parent_map.get(p.id, set())),
            }
            for p in plants
        ],
    }


@router.get("/health")
async def health():
    """Enhanced health check (Gap 16).

    Cold-start behaviour:
      - If SCHEDULER_ENABLED and no cycle has finished yet:
          < 120 s since start  → 200  status="initializing"
          ≥ 120 s since start  → 503  status="never_ran"
      - If last cycle is stale (> 120 s since finished) → 503 status="degraded"
      - Otherwise → 200 status="ok"
    """
    from fastapi.responses import JSONResponse

    now = datetime.now(timezone.utc)
    last_finished = cycle_state.last_cycle_finished_at
    started_at = cycle_state.service_started_at
    stalled = False
    stale_seconds = None
    status = "ok"

    if settings.SCHEDULER_ENABLED:
        if last_finished is None:
            # No cycle has run yet — are we still starting up?
            uptime = (now - started_at).total_seconds() if started_at else 9999
            if uptime < 120:
                status = "initializing"
            else:
                status = "never_ran"
                stalled = True
        else:
            stale_seconds = (now - last_finished).total_seconds()
            if stale_seconds > 120:
                stalled = True
                status = "degraded"

    body = {
        "status": status,
        "scheduler_enabled": settings.SCHEDULER_ENABLED,
        "service_started_at": started_at.isoformat() if started_at else None,
        "last_cycle_finished_at": last_finished.isoformat() if last_finished else None,
        "last_cycle_duration_ms": cycle_state.last_cycle_duration_ms,
        "last_cycle_plants": cycle_state.last_cycle_plants,
        "last_cycle_failures": cycle_state.last_cycle_failures,
        "total_cycles": cycle_state.total_cycles,
        "total_failures": cycle_state.total_failures,
        "stale_seconds": round(stale_seconds, 1) if stale_seconds is not None else None,
    }

    if stalled:
        return JSONResponse(content=body, status_code=503)
    return body


@router.post("/admin/run-now")
async def admin_run_now():
    """Manually trigger an immediate scheduler cycle (Gap 17).

    Delegates to APScheduler.modify(next_run_time=now) so max_instances=1
    is respected — a second call while a cycle is in flight is a no-op.
    """
    from app.services.scheduler import get_scheduler
    scheduler = get_scheduler()
    job = scheduler.get_job("pvlib_cycle")
    if job:
        job.modify(next_run_time=datetime.now(timezone.utc))
        return {"status": "scheduled", "triggered_at": datetime.now(timezone.utc).isoformat()}
    # Scheduler not running (e.g., SCHEDULER_ENABLED=false) — fire directly
    import asyncio
    asyncio.create_task(run_cycle_now())
    return {"status": "triggered_direct", "triggered_at": datetime.now(timezone.utc).isoformat()}


@router.post("/admin/refresh-plants")
async def admin_refresh_plants():
    """Force-invalidate the discover_plants cache (Gap 13).

    The next cycle will re-run BFS regardless of the 5-min TTL.
    """
    from app.services.thingsboard_client import _discover_cache
    from app.services.forecast_service import ForecastService
    _discover_cache.clear()
    ForecastService._plant_attrs_cache.clear()
    return {"status": "cache_cleared", "triggered_at": datetime.now(timezone.utc).isoformat()}


@router.post("/admin/run-weekly")
async def admin_run_weekly():
    """Manually trigger the weekly accuracy evaluation job (Gap 20).

    Equivalent to the Sunday 02:00 cron firing immediately.
    Returns the evaluation summary once it completes.
    """
    from app.services.scheduler import run_weekly_eval_now
    result = await run_weekly_eval_now()
    return result


@router.post("/admin/run-daily")
async def admin_run_daily(date: Optional[str] = Query(None)):
    """Manually trigger the daily energy roll-up job.

    date: optional ISO date string (YYYY-MM-DD).  If omitted, rolls up yesterday.
    """
    import asyncio
    from datetime import date as _date

    target_dt: Optional[datetime] = None
    if date:
        try:
            parsed = _date.fromisoformat(date)
            # Treat as local midnight
            from zoneinfo import ZoneInfo
            target_dt = datetime(parsed.year, parsed.month, parsed.day,
                                 tzinfo=ZoneInfo(settings.TZ_LOCAL))
        except ValueError:
            raise HTTPException(status_code=400, detail=f"Invalid date format: {date!r}. Use YYYY-MM-DD.")

    asyncio.create_task(run_daily_now(date=target_dt))
    return {
        "status": "triggered",
        "date": date or "yesterday",
        "triggered_at": datetime.now(timezone.utc).isoformat(),
    }


@router.post("/admin/run-daily-range")
async def admin_run_daily_range(
    start: Optional[str] = Query(None, description="Start date YYYY-MM-DD (inclusive). Defaults to yesterday."),
    end:   Optional[str] = Query(None, description="End date YYYY-MM-DD (inclusive). Defaults to start."),
):
    """Backfill daily energy roll-up (expected + actual) for a date range.

    Runs synchronously per day and returns a combined summary once all days complete.
    Writes both total_generation_expected_kwh AND actual_daily_energy_kwh for each day.

    Examples
    --------
    /admin/run-daily-range                              -> yesterday only
    /admin/run-daily-range?start=2026-01-01             -> 2026-01-01 only
    /admin/run-daily-range?start=2026-01-01&end=2026-05-12  -> 4-month backfill

    Note: for plants with historical active_power data, this will compute
    actual_daily_energy_kwh from that data. The pvlib expected values are also
    recomputed from potential_power for the same days.
    """
    from datetime import date as _date
    from zoneinfo import ZoneInfo
    from app.services.daily_job import run_daily_rollup
    from app.services.thingsboard_client import ThingsBoardClient

    tz = ZoneInfo(settings.TZ_LOCAL)

    def _parse(s: str) -> _date:
        try:
            return _date.fromisoformat(s)
        except ValueError:
            raise HTTPException(status_code=400, detail=f"Invalid date format: {s!r}. Use YYYY-MM-DD.")

    today_local = datetime.now(tz).date()
    start_date  = _parse(start) if start else today_local - timedelta(days=1)
    end_date    = _parse(end)   if end   else start_date

    if end_date < start_date:
        raise HTTPException(status_code=400, detail="end must be >= start")
    if end_date >= today_local:
        raise HTTPException(status_code=400, detail="end must be before today (only complete days can be backfilled)")

    results = []
    cursor  = start_date

    async with ThingsBoardClient(
        settings.TB_HOST, settings.TB_USERNAME, settings.TB_PASSWORD
    ) as tb:
        while cursor <= end_date:
            target_dt = datetime(cursor.year, cursor.month, cursor.day, tzinfo=tz) + timedelta(days=1)
            try:
                summary = await run_daily_rollup(tb, date=target_dt)
                summary["date"] = str(cursor)
            except Exception as exc:
                summary = {"date": str(cursor), "status": "error", "error": str(exc)}
            results.append(summary)
            cursor += timedelta(days=1)

    return {
        "status": "done",
        "days": len(results),
        "plants_ok_total":     sum(r.get("ok",     0) for r in results),
        "plants_failed_total": sum(r.get("failed", 0) for r in results),
        "triggered_at": datetime.now(timezone.utc).isoformat(),
        "results": results,
    }


@router.post("/admin/run-loss-rollup")
async def admin_run_loss_rollup(
    start: Optional[str] = Query(None, description="Start date YYYY-MM-DD (inclusive). Defaults to yesterday."),
    end: Optional[str] = Query(None, description="End date YYYY-MM-DD (inclusive). Defaults to start."),
):
    """Trigger the daily loss-rollup job for a date range (backfill) or for yesterday.

    Runs synchronously per day and returns a combined summary once all days complete.
    For large backfills (many years), consider running one month at a time.

    Examples
    --------
    /admin/run-loss-rollup                         → yesterday only
    /admin/run-loss-rollup?start=2026-04-01        → 2026-04-01 only
    /admin/run-loss-rollup?start=2026-01-01&end=2026-04-30  → 4-month backfill
    """
    import asyncio
    from datetime import date as _date
    from zoneinfo import ZoneInfo

    tz = ZoneInfo(settings.TZ_LOCAL)

    def _parse_date(s: str) -> _date:
        try:
            return _date.fromisoformat(s)
        except ValueError:
            raise HTTPException(status_code=400, detail=f"Invalid date format: {s!r}. Use YYYY-MM-DD.")

    today_local = datetime.now(tz).date()

    if start is None:
        start_date = today_local - timedelta(days=1)
    else:
        start_date = _parse_date(start)

    end_date = _parse_date(end) if end else start_date

    if end_date < start_date:
        raise HTTPException(status_code=400, detail="end must be >= start")

    if not settings.LOSS_ROLLUP_ENABLED:
        # Allow backfill even when the daily cron is disabled (operator opt-in)
        pass  # run_loss_rollup_now will handle the flag for the cron; backfill bypasses it

    results = []
    cursor = start_date

    from app.services.loss_rollup_job import run_loss_rollup
    from app.services.thingsboard_client import ThingsBoardClient

    async with ThingsBoardClient(
        settings.TB_HOST, settings.TB_USERNAME, settings.TB_PASSWORD
    ) as tb:
        while cursor <= end_date:
            # Align to local midnight of the day AFTER cursor (the day we want to compute)
            target_dt = datetime(
                cursor.year, cursor.month, cursor.day,
                tzinfo=tz,
            ) + timedelta(days=1)
            try:
                summary = await run_loss_rollup(tb, date=target_dt)
            except Exception as exc:
                summary = {"date": str(cursor), "status": "error", "error": str(exc)}
            results.append(summary)
            cursor += timedelta(days=1)

    total_ok = sum(r.get("plants_ok", 0) for r in results)
    total_failed = sum(r.get("plants_failed", 0) for r in results)

    return {
        "status": "done",
        "days": len(results),
        "plants_ok_total": total_ok,
        "plants_failed_total": total_failed,
        "triggered_at": datetime.now(timezone.utc).isoformat(),
        "results": results,
    }


@router.post("/admin/run-today-partial")
async def admin_run_today_partial():
    """Manually trigger the intra-day today-partial loss roll-up.

    Computes today-so-far daily loss keys for all pvlib-enabled plants and writes
    them at today's local-midnight timestamp with ``loss_data_source = "ok:partial"``.
    Lifetime attributes are NOT updated by this endpoint.

    Requires both LOSS_ROLLUP_ENABLED=true and LOSS_TODAY_PARTIAL_ENABLED=true in .env,
    otherwise returns ``{"status": "disabled"}``.  Outside the configured daylight window
    returns ``{"status": "outside_window"}``.
    """
    return await run_today_partial_now()


@router.post("/admin/recompute-lifetime")
async def admin_recompute_lifetime(
    asset_id: Optional[str] = Query(None, description="Asset UUID to recompute. Omit for fleet-wide."),
):
    """Recompute lifetime cumulative loss attributes by summing the entire daily-key history.

    Fleet-wide when asset_id is omitted; single asset when provided.
    For large fleets this may take several minutes — returns once complete.

    Verification: loss_grid_lifetime_kwh should equal Σ(loss_grid_daily_kwh) to within 0.1%.
    """
    from app.services.loss_rollup_job import recompute_lifetime_for_fleet
    from app.services.thingsboard_client import ThingsBoardClient

    async with ThingsBoardClient(
        settings.TB_HOST, settings.TB_USERNAME, settings.TB_PASSWORD
    ) as tb:
        result = await recompute_lifetime_for_fleet(tb, asset_id=asset_id)

    result["triggered_at"] = datetime.now(timezone.utc).isoformat()
    return result


@router.get("/admin/loss-status")
async def admin_loss_status(
    asset_id: str = Query(..., description="Asset UUID to inspect."),
):
    """Return latest daily loss key values and all lifetime attributes for one asset.

    Useful for sanity checks during Phase L1–L4 rollout.
    """
    from app.services.loss_rollup_job import get_loss_status
    from app.services.thingsboard_client import ThingsBoardClient

    if not asset_id:
        raise HTTPException(status_code=400, detail="asset_id is required")

    async with ThingsBoardClient(
        settings.TB_HOST, settings.TB_USERNAME, settings.TB_PASSWORD
    ) as tb:
        result = await get_loss_status(tb, asset_id)

    return result


@router.get("/metrics")
async def metrics():
    """Prometheus-compatible plain-text metrics (Gap 15)."""
    from fastapi.responses import PlainTextResponse
    from app.physics.data_sources import _solcast_hits_total, _solcast_misses_total
    from app.services.thingsboard_client import (
        _discover_cache_hits_total, _discover_cache_misses_total,
    )
    from app.metrics import data_source_count, plant_failures_total
    import app.services.auto_onboard as _ao

    lines = [
        # ── Scheduler cycle health ────────────────────────────────────────
        "# HELP pvlib_cycle_duration_seconds Duration of last scheduler cycle",
        "# TYPE pvlib_cycle_duration_seconds gauge",
        f'pvlib_cycle_duration_seconds {(cycle_state.last_cycle_duration_ms or 0) / 1000:.3f}',
        "# HELP pvlib_total_cycles_total Total scheduler cycles run",
        "# TYPE pvlib_total_cycles_total counter",
        f'pvlib_total_cycles_total {cycle_state.total_cycles}',
        "# HELP pvlib_total_failures_total Total scheduler cycle failures",
        "# TYPE pvlib_total_failures_total counter",
        f'pvlib_total_failures_total {cycle_state.total_failures}',
        "# HELP pvlib_last_cycle_plants Plants processed in last cycle",
        "# TYPE pvlib_last_cycle_plants gauge",
        f'pvlib_last_cycle_plants {cycle_state.last_cycle_plants}',
        "# HELP pvlib_last_cycle_failures Plant failures in last cycle",
        "# TYPE pvlib_last_cycle_failures gauge",
        f'pvlib_last_cycle_failures {cycle_state.last_cycle_failures}',
        "",
        # ── Data-source distribution (gauge, current cycle) ──────────────
        "# HELP pvlib_data_source_count Plants per data source in the last cycle",
        "# TYPE pvlib_data_source_count gauge",
    ]
    for source, count in sorted(data_source_count.items()):
        lines.append(f'pvlib_data_source_count{{source="{source}"}} {count}')

    lines += [
        "",
        # ── Per-plant failure totals (monotonic) ─────────────────────────
        "# HELP pvlib_plant_failures_total Cumulative plant failures by reason",
        "# TYPE pvlib_plant_failures_total counter",
    ]
    for (plant_id, reason), count in sorted(plant_failures_total.items()):
        lines.append(
            f'pvlib_plant_failures_total{{plant="{plant_id}",reason="{reason}"}} {count}'
        )

    lines += [
        "",
        # ── Solcast cache (monotonic) ─────────────────────────────────────
        "# HELP pvlib_solcast_hits_total Solcast in-process cache hits",
        "# TYPE pvlib_solcast_hits_total counter",
        f'pvlib_solcast_hits_total {_solcast_hits_total}',
        "# HELP pvlib_solcast_misses_total Solcast cache misses (API calls made)",
        "# TYPE pvlib_solcast_misses_total counter",
        f'pvlib_solcast_misses_total {_solcast_misses_total}',
        "",
        # ── Discover cache (monotonic) ────────────────────────────────────
        "# HELP pvlib_discover_cache_hits_total Plant-discovery BFS cache hits",
        "# TYPE pvlib_discover_cache_hits_total counter",
        f'pvlib_discover_cache_hits_total {_discover_cache_hits_total}',
        "# HELP pvlib_discover_cache_misses_total Plant-discovery BFS cache misses",
        "# TYPE pvlib_discover_cache_misses_total counter",
        f'pvlib_discover_cache_misses_total {_discover_cache_misses_total}',
        "",
        # ── Throughput gauges (Step 13 — F3) ─────────────────────────────
        # pvlib_cycle_plants_per_minute: average plants processed per minute over
        # the last cycle.  0 when no cycle has run yet.
        "# HELP pvlib_cycle_plants_per_minute Average plants processed per minute in last cycle",
        "# TYPE pvlib_cycle_plants_per_minute gauge",
    ]
    _dur_s = (cycle_state.last_cycle_duration_ms or 0) / 1000.0
    _plants = cycle_state.last_cycle_plants or 0
    _ppm = (_plants / (_dur_s / 60.0)) if _dur_s > 0 else 0.0
    lines.append(f'pvlib_cycle_plants_per_minute {_ppm:.2f}')

    # pvlib_cycle_duration_p95_ms: p95 of cycle durations over the last 60 cycles.
    # Computed from cycle_state.recent_cycle_durations_ms (ring buffer, populated below).
    # Falls back to last_cycle_duration_ms when fewer than 5 samples are available.
    lines += [
        "# HELP pvlib_cycle_duration_p95_ms p95 cycle duration in ms over last 60 cycles",
        "# TYPE pvlib_cycle_duration_p95_ms gauge",
    ]
    _recent = getattr(cycle_state, "recent_cycle_durations_ms", [])
    if len(_recent) >= 5:
        import statistics as _stats
        _sorted = sorted(_recent)
        _p95_idx = max(0, int(len(_sorted) * 0.95) - 1)
        _p95 = _sorted[_p95_idx]
    else:
        _p95 = cycle_state.last_cycle_duration_ms or 0
    lines.append(f'pvlib_cycle_duration_p95_ms {_p95}')

    lines += [
        "",
        # ── Auto-onboard counters (Step 28) ───────────────────────────────
        "# HELP pvlib_autoonboard_attempted_total Total auto-onboard plants attempted",
        "# TYPE pvlib_autoonboard_attempted_total counter",
        f'pvlib_autoonboard_attempted_total {_ao.autoonboard_attempted_total}',
        "# HELP pvlib_autoonboard_completed_total Auto-onboard plants successfully completed",
        "# TYPE pvlib_autoonboard_completed_total counter",
        f'pvlib_autoonboard_completed_total {_ao.autoonboard_completed_total}',
        "# HELP pvlib_autoonboard_failed_total Auto-onboard plants that failed or timed out",
        "# TYPE pvlib_autoonboard_failed_total counter",
        f'pvlib_autoonboard_failed_total {_ao.autoonboard_failed_total}',
        "# HELP pvlib_autoonboard_pending Plants awaiting onboarding (gauge)",
        "# TYPE pvlib_autoonboard_pending gauge",
        f'pvlib_autoonboard_pending {_ao.autoonboard_pending_gauge}',
    ]

    return PlainTextResponse("\n".join(lines) + "\n", media_type="text/plain; version=0.0.4")


@router.post("/admin/run-pvalues")
async def admin_run_pvalues(
    year: Optional[int] = Query(
        None,
        description="Target calendar year for daily/monthly telemetry timestamps. "
                    "Defaults to current local year. Use for backfill (e.g. year=2025).",
    ),
):
    """Trigger the P-value batch job for the full fleet.

    Fetches PVGIS-ERA5 multi-year hourly data per unique grid cell, runs the
    existing pvlib physics pipeline for each historical year, computes monthly
    P50/P90/P95 percentiles, and writes 9 telemetry keys per plant to TB:

      Daily timeseries   : forecast_p50_daily, forecast_p90_daily, forecast_p95_daily  (MWh)
      Monthly timeseries : forecast_p50_monthly, forecast_p90_monthly, forecast_p95_monthly (MWh)
      SERVER_SCOPE attrs : p50_energy, p90_energy, p95_energy  (kWh)

    This endpoint runs synchronously and returns once all plants are processed.
    For a single-plant smoke test before fleet-wide backfill, use /admin/run-pvalues-plant.

    Typical runtime: 5–20 min depending on fleet size and PVGIS API latency.
    """
    result = await run_pvalue_job_now(target_year=year)
    return result


@router.post("/admin/run-pvalues-plant")
async def admin_run_pvalues_plant(
    asset_id: str = Query(..., description="Asset UUID of the plant to compute P-values for."),
    year: Optional[int] = Query(
        None,
        description="Target calendar year. Defaults to current local year.",
    ),
):
    """Trigger the P-value batch job for a single plant (smoke test).

    Identical to /admin/run-pvalues but restricted to one asset UUID.
    Run this first on a known-good plant (e.g. KSP) to validate output
    before triggering the full fleet with /admin/run-pvalues.

    Verify the result:
      - plants_ok == 1, plants_failed == 0
      - Check TB: forecast_p50_daily exists with 365 rows for the target year
      - Sanity: p50_energy > p90_energy > p95_energy in SERVER_SCOPE attributes
      - Spot-check: sum(forecast_p50_daily for Jan) ≈ forecast_p50_monthly for Jan
    """
    if not asset_id:
        from fastapi import HTTPException
        raise HTTPException(status_code=400, detail="asset_id is required")

    result = await run_pvalue_job_now(target_year=year, plant_ids=[asset_id])
    return result


# ── Revenue admin endpoints (Step 26) ─────────────────────────────────────────

@router.post("/admin/run-revenue-monthly")
async def admin_run_revenue_monthly(
    year: Optional[int] = Query(None, description="Calendar year (default: previous month's year)"),
    month: Optional[int] = Query(None, description="Calendar month 1–12 (default: previous month)"),
):
    """Compute and write expected + actual LKR revenue for a calendar month.

    Reads ``forecast_p50_monthly`` (MWh) and ``actual_daily_energy_kwh`` from TB;
    writes ``expected_revenue_monthly_lkr`` and ``actual_revenue_monthly_lkr`` for
    every plant where ``pvlib_services.revenue != false``.

    Defaults to the previous calendar month so it can be called on the 1st of
    each month without arguments (same as the automatic cron).
    """
    from app.services.revenue_job import run_revenue_monthly
    from zoneinfo import ZoneInfo
    tz = ZoneInfo(settings.TZ_LOCAL)
    now = datetime.now(tz)
    if year is None or month is None:
        prev = now.replace(day=1) - timedelta(days=1)
        year = year or prev.year
        month = month or prev.month

    from app.services.thingsboard_client import ThingsBoardClient
    async with ThingsBoardClient(
        settings.TB_HOST, settings.TB_USERNAME, settings.TB_PASSWORD
    ) as tb:
        result = await run_revenue_monthly(tb, year=year, month=month)
    return result


@router.post("/admin/run-revenue-yearly")
async def admin_run_revenue_yearly(
    year: Optional[int] = Query(None, description="Calendar year (default: previous year)"),
):
    """Compute and write expected + actual LKR revenue for a calendar year.

    Reads ``p50_energy_annual`` timeseries and ``actual_daily_energy_kwh`` from TB;
    writes ``expected_revenue_yearly_lkr``, ``actual_revenue_yearly_lkr``, and
    ``actual_yearly_energy_kwh`` for every plant where ``pvlib_services.revenue != false``.

    Defaults to the previous calendar year so it can be called on 1st January
    without arguments (same as the automatic cron).
    """
    from app.services.revenue_job import run_revenue_yearly
    from zoneinfo import ZoneInfo
    tz = ZoneInfo(settings.TZ_LOCAL)
    if year is None:
        year = datetime.now(tz).year - 1

    from app.services.thingsboard_client import ThingsBoardClient
    async with ThingsBoardClient(
        settings.TB_HOST, settings.TB_USERNAME, settings.TB_PASSWORD
    ) as tb:
        result = await run_revenue_yearly(tb, year=year)
    return result


@router.post("/admin/run-revenue-backfill")
async def admin_run_revenue_backfill(
    asset_id: str = Query(..., description="Asset UUID of the plant to backfill"),
    years_back: int = Query(10, description="Number of calendar years to backfill (default: 10)"),
):
    """Backfill all months + years for a single plant over the last N years.

    Idempotent: running twice produces the same result.  Useful to call
    immediately after a new plant is onboarded or after ``tariff_rate_lkr``
    is set for the first time.
    """
    if not asset_id:
        raise HTTPException(status_code=400, detail="asset_id is required")

    from app.services.revenue_job import run_revenue_backfill
    from app.services.thingsboard_client import ThingsBoardClient
    async with ThingsBoardClient(
        settings.TB_HOST, settings.TB_USERNAME, settings.TB_PASSWORD
    ) as tb:
        result = await run_revenue_backfill(tb, asset_id=asset_id, years_back=years_back)
    return result


# ── Auto-onboard admin endpoints (Step 28) ────────────────────────────────────

@router.post("/admin/run-autoonboard")
async def admin_run_autoonboard(
    asset_id: Optional[str] = Query(
        None,
        description="Restrict to a single plant UUID.  Omit to process the full fleet.",
    ),
):
    """Run zero-touch onboarding backfill for all pending plants (or one plant).

    For every plant where ``onboarding_completed != true``:
      1. Runs P-values for each of the last 10 years
      2. Backfills ``actual_daily_energy_kwh`` from commissioning date to yesterday
      3. Backfills loss-rollup (if ``pvlib_services.loss_attribution`` enabled)
      4. Recomputes lifetime loss attributes
      5. Backfills LKR revenue (if ``pvlib_services.revenue`` enabled)
      6. Sets ``onboarding_completed=true`` + ``onboarding_completed_at=<ISO>``

    Plants that already have ``onboarding_completed=true`` are silently skipped
    (idempotent).  Each plant is subject to a per-plant timeout
    (``AUTOONBOARD_PER_PLANT_TIMEOUT_S``, default 900 s); on timeout the plant
    is left un-marked and will be retried on the next Sunday cron.
    """
    from app.services.auto_onboard import run_autoonboard_now
    from app.services.thingsboard_client import ThingsBoardClient
    async with ThingsBoardClient(
        settings.TB_HOST, settings.TB_USERNAME, settings.TB_PASSWORD
    ) as tb:
        result = await run_autoonboard_now(tb, asset_id=asset_id)
    return result
