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
from app.services.scheduler import cycle_state, run_cycle_now

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

    if result.get("status") not in ("ok", "no_data"):
        raise HTTPException(status_code=500, detail=result)
    return result


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
            }
            for p in plants
        ],
    }


@router.get("/health")
async def health():
    """Enhanced health check. Returns 503 if scheduler has stalled > 120 s."""
    now = datetime.now(timezone.utc)
    last_finished = cycle_state.last_cycle_finished_at
    stalled = False
    stale_seconds = None

    if settings.SCHEDULER_ENABLED and last_finished:
        stale_seconds = (now - last_finished).total_seconds()
        stalled = stale_seconds > 120

    body = {
        "status": "degraded" if stalled else "ok",
        "scheduler_enabled": settings.SCHEDULER_ENABLED,
        "last_cycle_finished_at": last_finished.isoformat() if last_finished else None,
        "last_cycle_duration_ms": cycle_state.last_cycle_duration_ms,
        "last_cycle_plants": cycle_state.last_cycle_plants,
        "last_cycle_failures": cycle_state.last_cycle_failures,
        "total_cycles": cycle_state.total_cycles,
        "total_failures": cycle_state.total_failures,
        "stale_seconds": round(stale_seconds, 1) if stale_seconds is not None else None,
    }

    if stalled:
        from fastapi.responses import JSONResponse
        return JSONResponse(content=body, status_code=503)
    return body


@router.post("/admin/run-now")
async def admin_run_now():
    """Manually trigger an immediate scheduler cycle."""
    import asyncio
    asyncio.create_task(run_cycle_now())
    return {"status": "triggered", "triggered_at": datetime.now(timezone.utc).isoformat()}


@router.get("/metrics")
async def metrics():
    """Prometheus-compatible plain-text metrics."""
    from fastapi.responses import PlainTextResponse

    lines = [
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
    ]
    return PlainTextResponse("\n".join(lines) + "\n", media_type="text/plain; version=0.0.4")
