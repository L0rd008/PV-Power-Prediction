"""
Daily energy roll-up job (Gap 3, Phase B).

Scheduled at 00:05 local time (Asia/Colombo by default) via APScheduler cron.
This replaces the old 01:00 UTC trigger inside run_fleet_cycle._write_daily_energies.

Algorithm:
  1. Determine the just-completed calendar day in local timezone.
  2. For each pvlib-enabled plant, read potential_power telemetry for that 24-hour window.
  3. Integrate: for 1-min cadence, integral ≈ sum(kW_values) / 60  (kWh).
  4. Write total_generation_expected_kwh + pvlib_daily_energy_kwh at midnight local ts.
  5. Roll up to all isPlantAgg ancestors.

Edge cases handled:
  - < 360 valid samples (< 6 h of daylight): write -1 with error:insufficient_samples
  - Backfill: /admin/run-daily?date=YYYY-MM-DD calls run_daily_rollup(date=<date>)
  - DST transitions: tz-aware arithmetic via ZoneInfo; no special-casing needed
"""
from __future__ import annotations

import logging
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Optional, Set
from zoneinfo import ZoneInfo

from app.config import settings
from app.services.forecast_service import (
    KEY_DATA_SOURCE,
    KEY_DAILY_ENERGY_EXPECTED,
    KEY_MODEL_VERSION,
    KEY_PVLIB_DAILY_ENERGY,
    MODEL_VERSION,
)

log = logging.getLogger(__name__)

# Minimum valid samples in a day to produce a real daily total.
# 360 = 6 h × 60 min.  Below this, likely a partial day or major service outage.
MIN_VALID_SAMPLES = 360


async def run_daily_rollup(
    tb_client,
    date: Optional[datetime] = None,
) -> Dict[str, object]:
    """Compute and write daily expected energy for all pvlib-enabled plants.

    Parameters
    ----------
    tb_client : ThingsBoardClient
        Authenticated TB client (singleton from app.state).
    date : datetime, optional
        Specific UTC or local midnight to compute for.  If None, computes
        for the calendar day that just ended (yesterday in local tz).

    Returns
    -------
    dict
        Summary: {plants_ok, plants_failed, plants_skipped}
    """
    tz = ZoneInfo(settings.TZ_LOCAL)
    now_local = datetime.now(tz)

    if date is None:
        # Day that just ended: local midnight today - 1 day
        local_midnight_today = now_local.replace(
            hour=0, minute=0, second=0, microsecond=0
        )
        day_end_local = local_midnight_today
    else:
        # Caller provided a specific date; align to local midnight
        day_end_local = date.astimezone(tz).replace(
            hour=0, minute=0, second=0, microsecond=0
        )

    day_start_local = day_end_local - timedelta(days=1)
    day_start_utc = day_start_local.astimezone(timezone.utc)
    day_end_utc   = day_end_local.astimezone(timezone.utc)

    log.info("run_daily_rollup: computing %s → %s (local: %s → %s)",
             day_start_utc.isoformat(), day_end_utc.isoformat(),
             day_start_local.isoformat(), day_end_local.isoformat())

    # Timestamp to stamp the daily value: local midnight (start of the day)
    day_ts_ms = int(day_start_local.timestamp() * 1000)

    # Discover plants + ancestor map
    from app.config import settings as _s
    plants, ancestor_map = await tb_client.discover_plants(_s.root_asset_ids)

    if not plants:
        log.warning("run_daily_rollup: no pvlib-enabled plants found")
        return {"plants_ok": 0, "plants_failed": 0, "plants_skipped": 0}

    from app.services.forecast_service import KEY_POTENTIAL_POWER
    stats = {"ok": 0, "failed": 0, "skipped": 0}
    plant_kwh: Dict[str, float] = {}  # plant_id → kwh (or -1 for invalid)

    for plant in plants:
        try:
            kwh = await _integrate_plant_day(
                tb_client, plant.id, day_start_utc, day_end_utc
            )
        except Exception as exc:
            log.error("run_daily_rollup: failed for %s: %s", plant.id, exc)
            kwh = -1.0
            stats["failed"] += 1

        plant_kwh[plant.id] = kwh

        if kwh == -1.0:
            val = {"pvlib_data_source": "error:integration_failed"}
            await _safe_write_daily(tb_client, plant.id, day_ts_ms, -1.0, -1.0, -1.0)
        else:
            month_start_utc = day_start_local.replace(day=1).astimezone(timezone.utc)
            year_start_utc = day_start_local.replace(month=1, day=1).astimezone(timezone.utc)
            
            monthly_history = await _get_historical_sum(tb_client, plant.id, month_start_utc, day_start_utc, KEY_DAILY_ENERGY_EXPECTED)
            yearly_history = await _get_historical_sum(tb_client, plant.id, year_start_utc, day_start_utc, KEY_DAILY_ENERGY_EXPECTED)
            
            monthly_kwh = monthly_history + kwh
            yearly_kwh = yearly_history + kwh

            await _safe_write_daily(tb_client, plant.id, day_ts_ms, kwh, monthly_kwh, yearly_kwh)
            stats["ok"] += 1

    # Ancestor roll-up
    ancestor_children: Dict[str, Set[str]] = {}
    for plant_id, ancestors in ancestor_map.items():
        for anc_id in ancestors:
            ancestor_children.setdefault(anc_id, set()).add(plant_id)

    for ancestor_id, child_ids in ancestor_children.items():
        valid_kwhs = [
            plant_kwh[cid] for cid in child_ids
            if cid in plant_kwh and plant_kwh[cid] >= 0
        ]
        total_kwh = sum(valid_kwhs) if valid_kwhs else -1.0
        
        if total_kwh != -1.0:
            month_start_utc = day_start_local.replace(day=1).astimezone(timezone.utc)
            year_start_utc = day_start_local.replace(month=1, day=1).astimezone(timezone.utc)
            monthly_history = await _get_historical_sum(tb_client, ancestor_id, month_start_utc, day_start_utc, KEY_DAILY_ENERGY_EXPECTED)
            yearly_history = await _get_historical_sum(tb_client, ancestor_id, year_start_utc, day_start_utc, KEY_DAILY_ENERGY_EXPECTED)
            await _safe_write_daily(tb_client, ancestor_id, day_ts_ms, total_kwh, monthly_history + total_kwh, yearly_history + total_kwh)
        else:
            await _safe_write_daily(tb_client, ancestor_id, day_ts_ms, -1.0, -1.0, -1.0)

    log.info("run_daily_rollup: done — ok=%d failed=%d skipped=%d",
             stats["ok"], stats["failed"], stats["skipped"])
    return stats


async def _integrate_plant_day(
    tb_client,
    plant_id: str,
    day_start_utc: datetime,
    day_end_utc: datetime,
) -> float:
    """Read potential_power for one plant over the day, integrate, return kWh.

    Returns -1.0 if data is insufficient (< MIN_VALID_SAMPLES valid records).
    """
    from app.services.forecast_service import KEY_POTENTIAL_POWER

    raw = await tb_client.get_timeseries(
        "ASSET", plant_id,
        [KEY_POTENTIAL_POWER],
        start=day_start_utc,
        end=day_end_utc,
        limit=100_000,
    )

    records = raw.get(KEY_POTENTIAL_POWER, [])
    if not records:
        log.info("_integrate_plant_day: no potential_power records for %s", plant_id)
        return -1.0

    import pandas as pd
    series = pd.Series({
        pd.Timestamp(r["ts"], unit="ms", tz="UTC"): float(r["value"])
        for r in records
        if float(r["value"]) >= 0  # exclude -1 sentinels
    })

    # Filter to 05:00-19:00 local time
    tz = ZoneInfo(settings.TZ_LOCAL)
    series_local = series.index.tz_convert(tz)
    series = series[(series_local.hour >= 5) & (series_local.hour < 19)]

    if len(series) < MIN_VALID_SAMPLES:
        log.warning(
            "_integrate_plant_day: only %d valid samples for %s (min %d) — writing -1",
            len(series), plant_id, MIN_VALID_SAMPLES,
        )
        return -1.0

    # For 1-minute cadence: integral = sum(kW) / 60  (kWh)
    kwh = float(series.sum()) / 60.0
    return round(kwh, 3)


async def _safe_write_daily(
    tb_client, entity_id: str, ts_ms: int, kwh: float, monthly_kwh: float = -1.0, yearly_kwh: float = -1.0
) -> None:
    """Write daily energy keys; best-effort (logs on failure)."""
    if kwh == -1.0:
        values = {
            KEY_DAILY_ENERGY_EXPECTED: -1,
            "total_generation_expected_monthly_kwh": -1,
            "total_generation_expected_yearly_kwh": -1,
            KEY_PVLIB_DAILY_ENERGY: -1,
            KEY_DATA_SOURCE: "error:insufficient_samples",
            KEY_MODEL_VERSION: MODEL_VERSION,
        }
    else:
        values = {
            KEY_DAILY_ENERGY_EXPECTED: round(kwh, 3),
            "total_generation_expected_monthly_kwh": round(monthly_kwh, 3) if monthly_kwh >= 0 else -1,
            "total_generation_expected_yearly_kwh": round(yearly_kwh, 3) if yearly_kwh >= 0 else -1,
            KEY_PVLIB_DAILY_ENERGY: round(kwh, 3),
            KEY_MODEL_VERSION: MODEL_VERSION,
        }
    try:
        await tb_client.post_telemetry("ASSET", entity_id, [{"ts": ts_ms, "values": values}])
    except Exception as exc:
        log.error("_safe_write_daily: failed for %s: %s", entity_id, exc)

async def _get_historical_sum(
    tb_client, entity_id: str, start_utc: datetime, end_utc: datetime, key: str
) -> float:
    """Fetch historical daily values up to end_utc (exclusive) and return their sum."""
    if start_utc >= end_utc:
        return 0.0
    try:
        raw = await tb_client.get_timeseries(
            "ASSET", entity_id, [key],
            start=start_utc, end=end_utc, limit=1000
        )
        records = raw.get(key, [])
        return sum(float(r["value"]) for r in records if float(r["value"]) >= 0)
    except Exception as exc:
        log.error("_get_historical_sum: failed for %s: %s", entity_id, exc)
        return 0.0
