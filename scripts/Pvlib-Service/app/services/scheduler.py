"""
APScheduler-based 1-minute cycle scheduler (P1-E).

Design decisions:
  - AsyncIOScheduler runs inside the same event loop as FastAPI/Uvicorn (single-worker)
  - max_instances=1 ensures cycles never overlap; a slow cycle is skipped, not queued
  - misfire_grace_time=30 catches one missed cycle after a brief hiccup
  - Bounded concurrency via asyncio.Semaphore(MAX_CONCURRENT_PLANTS)
  - Read window: [now - READ_LAG_SECONDS - READ_WINDOW_SECONDS, now - READ_LAG_SECONDS]
    so TB has time to receive station telemetry before we read it
  - last_cycle state is written here and exposed to /health
"""
from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Optional

from apscheduler.schedulers.asyncio import AsyncIOScheduler

from app.config import settings

log = logging.getLogger(__name__)


@dataclass
class CycleState:
    """Mutable state shared with the /health endpoint."""
    last_cycle_started_at: Optional[datetime] = None
    last_cycle_finished_at: Optional[datetime] = None
    last_cycle_duration_ms: Optional[float] = None
    last_cycle_plants: int = 0
    last_cycle_failures: int = 0
    total_cycles: int = 0
    total_failures: int = 0


# Module-level singleton — read by /health
cycle_state = CycleState()
_scheduler: Optional[AsyncIOScheduler] = None


def get_scheduler() -> AsyncIOScheduler:
    global _scheduler
    if _scheduler is None:
        _scheduler = AsyncIOScheduler(timezone="UTC")
    return _scheduler


async def run_cycle_now(tb_client=None) -> dict:
    """Trigger one cycle immediately (used by /admin/run-now and the scheduler job)."""
    from app.services.thingsboard_client import ThingsBoardClient
    from app.services.forecast_service import ForecastService

    now_utc = datetime.now(timezone.utc)
    window_end = now_utc - timedelta(seconds=settings.READ_LAG_SECONDS)
    window_start = window_end - timedelta(seconds=settings.READ_WINDOW_SECONDS)

    root_ids = settings.root_asset_ids
    if not root_ids:
        log.warning("scheduler: TB_ROOT_ASSET_IDS is empty — no plants to process")
        return {"status": "no_roots"}

    cycle_state.last_cycle_started_at = now_utc
    t0 = asyncio.get_event_loop().time()

    try:
        if tb_client is not None:
            svc = ForecastService(tb_client, solcast_api_key=settings.SOLCAST_API_KEY)
            summary = await svc.run_fleet_cycle(
                root_asset_ids=root_ids,
                start=window_start,
                end=window_end,
                max_concurrent=settings.MAX_CONCURRENT_PLANTS,
            )
        else:
            async with ThingsBoardClient(
                settings.TB_HOST, settings.TB_USERNAME, settings.TB_PASSWORD
            ) as tb:
                svc = ForecastService(tb, solcast_api_key=settings.SOLCAST_API_KEY)
                summary = await svc.run_fleet_cycle(
                    root_asset_ids=root_ids,
                    start=window_start,
                    end=window_end,
                    max_concurrent=settings.MAX_CONCURRENT_PLANTS,
                )

        duration_ms = (asyncio.get_event_loop().time() - t0) * 1000
        cycle_state.last_cycle_finished_at = datetime.now(timezone.utc)
        cycle_state.last_cycle_duration_ms = round(duration_ms, 1)
        cycle_state.last_cycle_plants = summary.get("plants_processed", 0)
        cycle_state.last_cycle_failures = summary.get("plants_failed", 0)
        cycle_state.total_cycles += 1

        if duration_ms > 45_000:
            log.warning("scheduler: cycle took %.1f s (target p95 < 45 s)", duration_ms / 1000)
        else:
            log.info("scheduler: cycle complete in %.1f s — %d plants, %d failed",
                     duration_ms / 1000,
                     cycle_state.last_cycle_plants,
                     cycle_state.last_cycle_failures)
        return summary

    except Exception as exc:
        cycle_state.last_cycle_failures = (cycle_state.last_cycle_failures or 0) + 1
        cycle_state.total_failures += 1
        log.exception("scheduler: cycle failed: %s", exc)
        return {"status": "error", "error": str(exc)}


def _sync_cycle_job():
    """APScheduler calls sync functions; bridge to async."""
    loop = asyncio.get_event_loop()
    if loop.is_running():
        asyncio.ensure_future(run_cycle_now())
    else:
        loop.run_until_complete(run_cycle_now())


def start_scheduler() -> None:
    """Start the APScheduler if SCHEDULER_ENABLED is true."""
    if not settings.SCHEDULER_ENABLED:
        log.info("scheduler: disabled (SCHEDULER_ENABLED=false)")
        return

    scheduler = get_scheduler()
    scheduler.add_job(
        run_cycle_now,
        trigger="interval",
        minutes=settings.SCHEDULER_INTERVAL_MINUTES,
        misfire_grace_time=30,
        max_instances=1,
        id="pvlib_cycle",
        replace_existing=True,
        coalesce=True,
    )
    scheduler.start()
    log.info("scheduler: started — interval %d min, max_concurrent %d plants",
             settings.SCHEDULER_INTERVAL_MINUTES, settings.MAX_CONCURRENT_PLANTS)


def stop_scheduler() -> None:
    global _scheduler
    if _scheduler and _scheduler.running:
        _scheduler.shutdown(wait=False)
        log.info("scheduler: stopped")
