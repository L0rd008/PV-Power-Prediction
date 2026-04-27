"""
Pvlib-Service FastAPI application entry point.

Phase C changes:
  - Gap 12: Singleton ThingsBoardClient constructed in lifespan, injected into scheduler.
            JWT is re-used across cycles instead of creating a new client every minute.
  - Gap 16: SERVICE_STARTED_AT set in lifespan, exposed to /health for cold-start detection.
  - Gap 17: /admin/run-now triggers via scheduler.modify (max_instances=1 respected).

Lifespan:
  startup  — construct TB client, inject into scheduler, start APScheduler
  shutdown — stop APScheduler, close TB client
"""
from __future__ import annotations

import logging
import sys
from contextlib import asynccontextmanager

from fastapi import FastAPI

from app.config import settings

# Configure logging before importing anything else
logging.basicConfig(
    level=getattr(logging, settings.LOG_LEVEL.upper(), logging.INFO),
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    stream=sys.stdout,
)
log = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Start/stop scheduler and singleton TB client around the FastAPI lifecycle."""
    log.info("pvlib-service starting (mode=%s, scheduler=%s)",
             settings.MODE, settings.SCHEDULER_ENABLED)

    from app.services.thingsboard_client import ThingsBoardClient
    from app.services.scheduler import start_scheduler, stop_scheduler, cycle_state
    from datetime import datetime, timezone

    # Gap 12: construct singleton TB client; JWT lasts ~2.5 h, reused every minute
    tb_client = ThingsBoardClient(
        settings.TB_HOST, settings.TB_USERNAME, settings.TB_PASSWORD
    )
    await tb_client.__aenter__()
    app.state.tb_client = tb_client

    # Gap 16: record service start time so /health can distinguish "initializing" from "dead"
    cycle_state.service_started_at = datetime.now(timezone.utc)

    # Pass singleton client into scheduler
    start_scheduler(tb_client=tb_client)

    yield

    stop_scheduler()
    await tb_client.__aexit__(None, None, None)
    log.info("pvlib-service shut down")


app = FastAPI(
    title="Pvlib Power Prediction Service",
    description=(
        "Real-time solar power estimation using pvlib physics (H-A3) "
        "with 3-tier data strategy (H-B6). "
        "Writes potential_power and active_power_pvlib_kw to ThingsBoard."
    ),
    version="1.1.0",
    lifespan=lifespan,
)

# Register routes
from app.api.forecast import router as forecast_router  # noqa: E402
app.include_router(forecast_router)

log.info("pvlib-service app created — routes registered")
