"""
APScheduler-based scheduler (P1-E, Phase B).

Jobs:
  pvlib_cycle  — interval every SCHEDULER_INTERVAL_MINUTES, runs run_cycle_now()
  pvlib_daily  — cron 00:05 local time (TZ_LOCAL), runs run_daily_rollup()

Design decisions:
  - AsyncIOScheduler uses timezone=ZoneInfo(TZ_LOCAL) so cron midnight triggers
    fire at local midnight regardless of server TZ (Gap 4/14).
  - max_instances=1 ensures cycles never overlap.
  - misfire_grace_time=30s on the minute job; 3600s on the daily job (Gap B Edge E13).
  - The old _sync_cycle_job bridge is removed (Gap 11 — dead code with 3.12 trap).
  - Singleton TB client injected at start_scheduler() call (Phase C, Gap 12).
  - last_cycle_state written here and exposed to /health.
"""
from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Optional
from zoneinfo import ZoneInfo

from apscheduler.schedulers.asyncio import AsyncIOScheduler

from app.config import settings

log = logging.getLogger(__name__)


@dataclass
class CycleState:
    """Mutable state shared with the /health endpoint."""
    service_started_at: Optional[datetime] = None
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

# Injected singleton TB client (set by main.py lifespan in Phase C)
_tb_client = None


def get_scheduler() -> AsyncIOScheduler:
    global _scheduler
    if _scheduler is None:
        # Use local timezone so daily cron fires at correct local midnight (Gap 4/14)
        _scheduler = AsyncIOScheduler(timezone=ZoneInfo(settings.TZ_LOCAL))
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

    # Prefer the injected singleton; fall back to per-cycle construction (Phase A compat)
    effective_client = tb_client or _tb_client

    try:
        if effective_client is not None:
            svc = ForecastService(effective_client, solcast_api_key=settings.SOLCAST_API_KEY)
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


async def run_weekly_eval_now(tb_client=None) -> dict:
    """Trigger the weekly accuracy evaluation immediately (used by /admin/run-weekly and the cron)."""
    from app.services.weekly_eval import run_weekly_eval

    effective_client = tb_client or _tb_client
    try:
        return await run_weekly_eval(tb_client=effective_client)
    except Exception as exc:
        log.exception("run_weekly_eval_now: failed: %s", exc)
        return {"status": "error", "error": str(exc)}


async def run_loss_rollup_now(date: Optional[datetime] = None) -> dict:
    """Trigger the daily loss-rollup job immediately (used by /admin/run-loss-rollup and the cron)."""
    if not settings.LOSS_ROLLUP_ENABLED:
        log.debug("run_loss_rollup_now: LOSS_ROLLUP_ENABLED=false — skipping")
        return {"status": "disabled"}

    from app.services.loss_rollup_job import run_loss_rollup
    from app.services.thingsboard_client import ThingsBoardClient

    effective_client = _tb_client

    try:
        if effective_client is not None:
            return await run_loss_rollup(effective_client, date=date)
        else:
            async with ThingsBoardClient(
                settings.TB_HOST, settings.TB_USERNAME, settings.TB_PASSWORD
            ) as tb:
                return await run_loss_rollup(tb, date=date)
    except Exception as exc:
        log.exception("run_loss_rollup_now: failed: %s", exc)
        return {"status": "error", "error": str(exc)}


async def run_today_partial_now() -> dict:
    """Trigger the today-partial loss roll-up immediately.

    Gated on both LOSS_ROLLUP_ENABLED and LOSS_TODAY_PARTIAL_ENABLED.
    Also honours the daylight-hours window (DAY_START_HOUR..DAY_END_HOUR).
    """
    if not settings.LOSS_ROLLUP_ENABLED or not settings.LOSS_TODAY_PARTIAL_ENABLED:
        log.debug("run_today_partial_now: disabled — skipping")
        return {"status": "disabled"}

    tz = ZoneInfo(settings.TZ_LOCAL)
    h = datetime.now(tz).hour
    if h < settings.LOSS_TODAY_PARTIAL_DAY_START_HOUR or h >= settings.LOSS_TODAY_PARTIAL_DAY_END_HOUR:
        log.debug(
            "run_today_partial_now: outside window (%02d:00–%02d:00), current hour=%d",
            settings.LOSS_TODAY_PARTIAL_DAY_START_HOUR,
            settings.LOSS_TODAY_PARTIAL_DAY_END_HOUR,
            h,
        )
        return {"status": "outside_window", "hour": h}

    from app.services.loss_rollup_job import run_today_partial_rollup
    from app.services.thingsboard_client import ThingsBoardClient

    effective_client = _tb_client
    try:
        if effective_client is not None:
            return await run_today_partial_rollup(effective_client)
        else:
            async with ThingsBoardClient(
                settings.TB_HOST, settings.TB_USERNAME, settings.TB_PASSWORD
            ) as tb:
                return await run_today_partial_rollup(tb)
    except Exception as exc:
        log.exception("run_today_partial_now: failed: %s", exc)
        return {"status": "error", "error": str(exc)}


async def run_daily_now(date: Optional[datetime] = None) -> dict:
    """Trigger the daily roll-up job immediately (used by /admin/run-daily and the cron)."""
    from app.services.daily_job import run_daily_rollup
    from app.services.thingsboard_client import ThingsBoardClient

    effective_client = _tb_client

    try:
        if effective_client is not None:
            return await run_daily_rollup(effective_client, date=date)
        else:
            async with ThingsBoardClient(
                settings.TB_HOST, settings.TB_USERNAME, settings.TB_PASSWORD
            ) as tb:
                return await run_daily_rollup(tb, date=date)
    except Exception as exc:
        log.exception("run_daily_now: failed: %s", exc)
        return {"status": "error", "error": str(exc)}


def start_scheduler(tb_client=None) -> None:
    """Start APScheduler if SCHEDULER_ENABLED is true.

    Parameters
    ----------
    tb_client : ThingsBoardClient, optional
        Singleton client injected from main.py lifespan (Phase C, Gap 12).
        If None, a new client is created per cycle (Phase A/B behaviour).
    """
    global _tb_client
    if tb_client is not None:
        _tb_client = tb_client

    if not settings.SCHEDULER_ENABLED:
        log.info("scheduler: disabled (SCHEDULER_ENABLED=false)")
        return

    cycle_state.service_started_at = datetime.now(timezone.utc)
    scheduler = get_scheduler()

    # 1-minute cycle job
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

    # Daily energy job: cron at 00:05 local time (Gap 3 / B)
    # misfire_grace_time=3600 handles the case where the service was down at 00:05
    # and restarts within the hour — the job fires immediately (Edge E13).
    scheduler.add_job(
        run_daily_now,
        trigger="cron",
        hour=0,
        minute=5,
        misfire_grace_time=3600,
        max_instances=1,
        id="pvlib_daily",
        replace_existing=True,
    )

    # Daily loss roll-up: cron at 00:10 local time (5 min after energy job at 00:05).
    # Gated on LOSS_ROLLUP_ENABLED — no-op when false so flag is safe to deploy disabled.
    if settings.LOSS_ROLLUP_ENABLED:
        scheduler.add_job(
            run_loss_rollup_now,
            trigger="cron",
            hour=0,
            minute=10,
            misfire_grace_time=3600,
            max_instances=1,
            id="pvlib_loss_rollup",
            replace_existing=True,
        )
        log.info("scheduler: loss rollup cron registered at 00:10 %s", settings.TZ_LOCAL)

        if settings.LOSS_TODAY_PARTIAL_ENABLED:
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
            log.info(
                "scheduler: today-partial cron registered every %d min %02d:00–%02d:00 %s",
                settings.LOSS_TODAY_PARTIAL_INTERVAL_MIN,
                settings.LOSS_TODAY_PARTIAL_DAY_START_HOUR,
                settings.LOSS_TODAY_PARTIAL_DAY_END_HOUR,
                settings.TZ_LOCAL,
            )
        else:
            log.info("scheduler: today-partial cron NOT registered (LOSS_TODAY_PARTIAL_ENABLED=false)")
    else:
        log.info("scheduler: loss rollup cron NOT registered (LOSS_ROLLUP_ENABLED=false)")

    # Weekly accuracy evaluation: cron at 02:00 every Sunday (Gap 20 / F)
    # misfire_grace_time=7200 (2 h): catches a restart within 2 hours of the window.
    scheduler.add_job(
        run_weekly_eval_now,
        trigger="cron",
        day_of_week="sun",
        hour=2,
        minute=0,
        misfire_grace_time=7200,
        max_instances=1,
        id="pvlib_weekly_eval",
        replace_existing=True,
    )

    scheduler.start()
    log.info(
        "scheduler: started — interval %d min, daily cron 00:05 %s, max_concurrent %d plants",
        settings.SCHEDULER_INTERVAL_MINUTES,
        settings.TZ_LOCAL,
        settings.MAX_CONCURRENT_PLANTS,
    )


def stop_scheduler() -> None:
    global _scheduler
    if _scheduler and _scheduler.running:
        _scheduler.shutdown(wait=False)
        log.info("scheduler: stopped")
