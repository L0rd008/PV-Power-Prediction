"""
Pvlib-Service FastAPI application entry point.

Lifespan:
  startup  — start APScheduler if SCHEDULER_ENABLED
  shutdown — stop APScheduler gracefully
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
    """Start/stop scheduler around the FastAPI lifecycle."""
    log.info("pvlib-service starting (mode=%s, scheduler=%s)",
             settings.MODE, settings.SCHEDULER_ENABLED)

    from app.services.scheduler import start_scheduler, stop_scheduler
    start_scheduler()

    yield

    stop_scheduler()
    log.info("pvlib-service shut down")


app = FastAPI(
    title="Pvlib Power Prediction Service",
    description=(
        "Real-time solar power estimation using pvlib physics (H-A3) "
        "with 3-tier data strategy (H-B6). "
        "Writes potential_power and active_power_pvlib_kw to ThingsBoard."
    ),
    version="1.0.0",
    lifespan=lifespan,
)

# Register routes
from app.api.forecast import router as forecast_router  # noqa: E402
app.include_router(forecast_router)

log.info("pvlib-service app created — routes registered")
