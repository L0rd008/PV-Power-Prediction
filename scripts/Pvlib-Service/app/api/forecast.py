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
"""
from __future__ import annotations

import time
from datetime import datetime, timedelta, timezone
from typing import Optional

from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel

from app.config import settings
from app.services.job_manager import job_manager
from app.services.scheduler import cycle_state, run_cycle_now, run_daily_now

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
    _discover_cache.clear()
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


@router.get("/metrics")
async def metrics():
    """Prometheus-compatible plain-text metrics (Gap 15)."""
    from fastapi.responses import PlainTextResponse
    from app.physics.data_sources import _solcast_hits_total, _solcast_misses_total
    from app.services.thingsboard_client import (
        _discover_cache_hits_total, _discover_cache_misses_total,
    )
    from app.metrics import data_source_count, plant_failures_total

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
    ]
    return PlainTextResponse("\n".join(lines) + "\n", media_type="text/plain; version=0.0.4")
