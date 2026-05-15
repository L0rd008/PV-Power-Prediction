"""
revenue_job.py — Monthly + yearly LKR revenue telemetry (Phase 4 Step 26 — P-A).

Writes five telemetry keys to ThingsBoard:

  Per plant, per calendar month (ts = 1st-of-month midnight local):
    expected_revenue_monthly_lkr   [LKR] = forecast_p50_monthly_kwh × tariff
    actual_revenue_monthly_lkr     [LKR] = Σ actual_daily_energy_kwh in month × tariff

  Per plant, per calendar year (ts = 1st-of-year midnight local):
    expected_revenue_yearly_lkr    [LKR] = p50_energy_annual_kwh × tariff  (from pvalue_job)
    actual_revenue_yearly_lkr      [LKR] = Σ actual_daily_energy_kwh in year × tariff
    actual_yearly_energy_kwh       [kWh] = Σ actual_daily_energy_kwh in year

Sentinels:
  - -1 for any LKR key when tariff_rate_lkr is missing.
  - -1 for yearly keys when the year predates commissioning_date.

Idempotency:
  Each call re-computes and overwrites the target month/year row.
  Run /admin/run-revenue-monthly twice for the same month → same result.

Gating:
  - REVENUE_JOB_ENABLED must be true in .env for crons to be registered.
  - Per-plant pvlib_services["revenue"] = false skips that plant (Step 25).

Crons (registered by scheduler.py when REVENUE_JOB_ENABLED=true):
  pvlib_revenue_monthly — 1st-of-month 00:15 local (computes just-finished month)
  pvlib_revenue_yearly  — 1st-of-year  00:20 local (computes just-finished year)

Admin endpoints (registered in forecast.py):
  POST /admin/run-revenue-monthly?year=N&month=M  (default = previous month)
  POST /admin/run-revenue-yearly?year=N           (default = previous year)
  POST /admin/run-revenue-backfill?asset_id=<id>&years_back=10
"""
from __future__ import annotations

import asyncio
import logging
from calendar import monthrange
from datetime import datetime, timezone, timedelta
from typing import Any, Dict, List, Optional
from zoneinfo import ZoneInfo

from app.config import settings

log = logging.getLogger(__name__)

# ── Telemetry key constants ──────────────────────────────────────────────────

KEY_EXPECTED_MONTHLY = "expected_revenue_monthly_lkr"
KEY_ACTUAL_MONTHLY   = "actual_revenue_monthly_lkr"
KEY_EXPECTED_YEARLY  = "expected_revenue_yearly_lkr"
KEY_ACTUAL_YEARLY    = "actual_revenue_yearly_lkr"
KEY_ACTUAL_YEARLY_KWH = "actual_yearly_energy_kwh"

# Source keys read from TB
KEY_FORECAST_MONTHLY = "forecast_p50_monthly"   # MWh — written by pvalue_job
KEY_ACTUAL_DAILY     = "actual_daily_energy_kwh" # kWh — written by daily_job
KEY_P50_ANNUAL       = "p50_energy_annual"        # kWh — written by pvalue_job (Step 27)

SENTINEL = -1


# ── Helpers ──────────────────────────────────────────────────────────────────

def _tz() -> ZoneInfo:
    return ZoneInfo(settings.TZ_LOCAL)


def _month_ts_ms(year: int, month: int) -> int:
    """Unix ms of 1st-of-month local midnight."""
    tz = _tz()
    dt = datetime(year, month, 1, 0, 0, 0, tzinfo=tz)
    return int(dt.timestamp() * 1000)


def _year_ts_ms(year: int) -> int:
    """Unix ms of 1st-of-year local midnight."""
    return _month_ts_ms(year, 1)


def _month_window_utc(year: int, month: int):
    """Return (start_utc, end_utc) covering [1st-of-month, 1st-of-next-month)."""
    tz = _tz()
    start_local = datetime(year, month, 1, 0, 0, 0, tzinfo=tz)
    if month == 12:
        end_local = datetime(year + 1, 1, 1, 0, 0, 0, tzinfo=tz)
    else:
        end_local = datetime(year, month + 1, 1, 0, 0, 0, tzinfo=tz)
    return start_local.astimezone(timezone.utc), end_local.astimezone(timezone.utc)


def _year_window_utc(year: int):
    """Return (start_utc, end_utc) covering [1st-of-year, 1st-of-next-year)."""
    tz = _tz()
    start_local = datetime(year, 1, 1, 0, 0, 0, tzinfo=tz)
    end_local   = datetime(year + 1, 1, 1, 0, 0, 0, tzinfo=tz)
    return start_local.astimezone(timezone.utc), end_local.astimezone(timezone.utc)


def _parse_float(v) -> Optional[float]:
    if v is None:
        return None
    try:
        f = float(v)
        return f if f >= 0 else None
    except (TypeError, ValueError):
        return None


async def _sum_daily_kwh(tb_client, asset_id: str, start_utc: datetime, end_utc: datetime) -> float:
    """Sum actual_daily_energy_kwh rows in [start_utc, end_utc). Skip sentinels (< 0)."""
    try:
        raw = await tb_client.get_timeseries(
            "ASSET", asset_id,
            keys=[KEY_ACTUAL_DAILY],
            start=start_utc,
            end=end_utc,
            limit=400,
            agg="NONE",
        )
    except Exception as exc:
        log.warning("_sum_daily_kwh: fetch failed for %s: %s", asset_id, exc)
        return SENTINEL

    rows = raw.get(KEY_ACTUAL_DAILY, [])
    total = 0.0
    for r in rows:
        v = _parse_float(r.get("value"))
        if v is not None and v >= 0:
            total += v
    return total


# ── Monthly revenue ──────────────────────────────────────────────────────────

async def run_revenue_monthly(
    tb_client,
    year: int,
    month: int,
    plant_ids: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """Compute and write expected + actual monthly LKR revenue for all plants.

    Parameters
    ----------
    year, month : int
        Target calendar month (1-based).
    plant_ids : list[str], optional
        Restrict to specific plants (default: full fleet).
    """
    log.info("run_revenue_monthly: %d-%02d", year, month)

    plants, _ = await tb_client.discover_plants(settings.root_asset_ids)
    if plant_ids:
        plants = [p for p in plants if p.id in plant_ids]

    if not plants:
        log.warning("run_revenue_monthly: no plants")
        return {"plants_ok": 0, "plants_failed": 0, "year": year, "month": month}

    start_utc, end_utc = _month_window_utc(year, month)
    ts_ms = _month_ts_ms(year, month)

    stats = {"ok": 0, "failed": 0}

    for plant in plants:
        # Step 25 gate
        if not plant.services.get("revenue", True):
            log.debug("run_revenue_monthly: skipping %s (revenue=false)", plant.id)
            continue
        try:
            await _compute_write_monthly(tb_client, plant.id, year, month,
                                         start_utc, end_utc, ts_ms)
            stats["ok"] += 1
        except Exception as exc:
            log.error("run_revenue_monthly: failed for %s: %s", plant.id, exc)
            stats["failed"] += 1

    log.info("run_revenue_monthly: %d-%02d done — ok=%d failed=%d",
             year, month, stats["ok"], stats["failed"])
    return {**stats, "year": year, "month": month}


async def _compute_write_monthly(
    tb_client,
    asset_id: str,
    year: int,
    month: int,
    start_utc: datetime,
    end_utc: datetime,
    ts_ms: int,
) -> None:
    attrs = await tb_client.get_asset_attributes(asset_id)
    tariff = _parse_float(attrs.get("tariff_rate_lkr"))

    # Expected: read forecast_p50_monthly row at the 1st-of-month ts
    expected_lkr = SENTINEL
    try:
        raw_fc = await tb_client.get_timeseries(
            "ASSET", asset_id,
            keys=[KEY_FORECAST_MONTHLY],
            start=start_utc - timedelta(hours=1),
            end=start_utc + timedelta(hours=1),
            limit=5,
            agg="NONE",
        )
        fc_rows = raw_fc.get(KEY_FORECAST_MONTHLY, [])
        if fc_rows and tariff is not None:
            fc_mwh = _parse_float(fc_rows[0].get("value"))
            if fc_mwh is not None and fc_mwh >= 0:
                expected_lkr = round(fc_mwh * 1000 * tariff, 2)  # MWh → kWh × tariff
    except Exception as exc:
        log.warning("_compute_write_monthly: forecast fetch failed for %s: %s", asset_id, exc)

    # Actual: sum daily kWh
    actual_kwh = await _sum_daily_kwh(tb_client, asset_id, start_utc, end_utc)
    actual_lkr = SENTINEL
    if actual_kwh >= 0 and tariff is not None:
        actual_lkr = round(actual_kwh * tariff, 2)

    record = {"ts": ts_ms, "values": {
        KEY_EXPECTED_MONTHLY: expected_lkr,
        KEY_ACTUAL_MONTHLY:   actual_lkr,
    }}
    await tb_client.post_telemetry("ASSET", asset_id, [record])
    log.debug("_compute_write_monthly: %s %d-%02d → expected=%.0f actual=%.0f LKR",
              asset_id, year, month, expected_lkr, actual_lkr)


# ── Yearly revenue ────────────────────────────────────────────────────────────

async def run_revenue_yearly(
    tb_client,
    year: int,
    plant_ids: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """Compute and write expected + actual yearly LKR revenue + yearly kWh for all plants."""
    log.info("run_revenue_yearly: %d", year)

    plants, _ = await tb_client.discover_plants(settings.root_asset_ids)
    if plant_ids:
        plants = [p for p in plants if p.id in plant_ids]

    if not plants:
        log.warning("run_revenue_yearly: no plants")
        return {"plants_ok": 0, "plants_failed": 0, "year": year}

    start_utc, end_utc = _year_window_utc(year)
    ts_ms = _year_ts_ms(year)

    stats = {"ok": 0, "failed": 0}

    for plant in plants:
        if not plant.services.get("revenue", True):
            log.debug("run_revenue_yearly: skipping %s (revenue=false)", plant.id)
            continue
        try:
            await _compute_write_yearly(tb_client, plant.id, year,
                                         start_utc, end_utc, ts_ms)
            stats["ok"] += 1
        except Exception as exc:
            log.error("run_revenue_yearly: failed for %s: %s", plant.id, exc)
            stats["failed"] += 1

    log.info("run_revenue_yearly: %d done — ok=%d failed=%d", year, stats["ok"], stats["failed"])
    return {**stats, "year": year}


async def _compute_write_yearly(
    tb_client,
    asset_id: str,
    year: int,
    start_utc: datetime,
    end_utc: datetime,
    ts_ms: int,
) -> None:
    attrs = await tb_client.get_asset_attributes(asset_id)
    tariff = _parse_float(attrs.get("tariff_rate_lkr"))

    # Check commissioning date — pre-commissioning years get actual = sentinel
    commissioning_raw = attrs.get("commissioning_date")
    before_commissioning = False
    if commissioning_raw:
        try:
            from datetime import date
            cd = date.fromisoformat(str(commissioning_raw)[:10])
            if cd.year > year:
                before_commissioning = True
        except Exception:
            pass

    # Expected: read p50_energy_annual timeseries row at the 1st-of-year ts (Step 27)
    expected_lkr = SENTINEL
    try:
        raw_p50 = await tb_client.get_timeseries(
            "ASSET", asset_id,
            keys=[KEY_P50_ANNUAL],
            start=start_utc - timedelta(hours=1),
            end=start_utc + timedelta(hours=1),
            limit=5,
            agg="NONE",
        )
        p50_rows = raw_p50.get(KEY_P50_ANNUAL, [])
        if p50_rows and tariff is not None:
            p50_kwh = _parse_float(p50_rows[0].get("value"))
            if p50_kwh is not None and p50_kwh >= 0:
                expected_lkr = round(p50_kwh * tariff, 2)
    except Exception as exc:
        log.warning("_compute_write_yearly: p50_annual fetch failed for %s: %s", asset_id, exc)

    # Actual: sum daily kWh for the year
    actual_kwh: float = SENTINEL
    actual_lkr: float = SENTINEL
    if not before_commissioning:
        actual_kwh = await _sum_daily_kwh(tb_client, asset_id, start_utc, end_utc)
        if actual_kwh >= 0 and tariff is not None:
            actual_lkr = round(actual_kwh * tariff, 2)

    record = {"ts": ts_ms, "values": {
        KEY_EXPECTED_YEARLY:  expected_lkr,
        KEY_ACTUAL_YEARLY:    actual_lkr,
        KEY_ACTUAL_YEARLY_KWH: round(actual_kwh, 3) if actual_kwh >= 0 else SENTINEL,
    }}
    await tb_client.post_telemetry("ASSET", asset_id, [record])
    log.debug("_compute_write_yearly: %s %d → expected=%.0f actual=%.0f LKR (kwh=%.0f)",
              asset_id, year, expected_lkr, actual_lkr, actual_kwh if actual_kwh >= 0 else -1)


# ── Revenue backfill ──────────────────────────────────────────────────────────

async def run_revenue_backfill(
    tb_client,
    asset_id: str,
    years_back: int = 10,
) -> Dict[str, Any]:
    """Backfill all months + years for a single plant over the last years_back years.

    Idempotent: running twice produces the same result.
    """
    tz = _tz()
    current_year = datetime.now(tz).year
    current_month = datetime.now(tz).month

    log.info("run_revenue_backfill: %s — %d years back", asset_id, years_back)

    months_ok = 0
    months_failed = 0
    years_ok = 0
    years_failed = 0

    # Discover the plant to get its services dict
    plants, _ = await tb_client.discover_plants(settings.root_asset_ids)
    plant_map = {p.id: p for p in plants}
    plant = plant_map.get(asset_id)
    if plant and not plant.services.get("revenue", True):
        log.info("run_revenue_backfill: %s skipped (revenue=false)", asset_id)
        return {"skipped": True}

    for y in range(current_year - years_back, current_year + 1):
        # Monthly backfill
        max_month = current_month if y == current_year else 12
        for m in range(1, max_month + 1):
            try:
                start_utc, end_utc = _month_window_utc(y, m)
                ts_ms = _month_ts_ms(y, m)
                await _compute_write_monthly(tb_client, asset_id, y, m,
                                              start_utc, end_utc, ts_ms)
                months_ok += 1
            except Exception as exc:
                log.warning("run_revenue_backfill: monthly %d-%02d failed for %s: %s",
                            y, m, asset_id, exc)
                months_failed += 1

        # Yearly backfill (only completed years)
        if y < current_year:
            try:
                start_utc, end_utc = _year_window_utc(y)
                ts_ms = _year_ts_ms(y)
                await _compute_write_yearly(tb_client, asset_id, y,
                                             start_utc, end_utc, ts_ms)
                years_ok += 1
            except Exception as exc:
                log.warning("run_revenue_backfill: yearly %d failed for %s: %s",
                            y, asset_id, exc)
                years_failed += 1

    log.info("run_revenue_backfill: %s done — months ok=%d fail=%d | years ok=%d fail=%d",
             asset_id, months_ok, months_failed, years_ok, years_failed)
    return {
        "asset_id": asset_id,
        "months_ok": months_ok, "months_failed": months_failed,
        "years_ok": years_ok, "years_failed": years_failed,
    }
