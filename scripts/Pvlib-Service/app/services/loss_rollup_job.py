"""
Daily loss-attribution roll-up job (Phase L0).

Scheduled at 00:10 local time (TZ_LOCAL) via APScheduler cron — 5 minutes
after the daily energy job (00:05) so potential_power daily keys are already
written before we read them.

Algorithm per plant:
  1. Fetch potential_power, active_power (or first matching actualPowerKey),
     and all setpointKeys for [day_start_utc, day_end_utc] + 30-day setpoint lookback.
  2. Normalise active_power W→kW via per-plant active_power_unit attribute.
  3. Drop records with value < 0 (sentinels).
  4. If valid samples < LOSS_MIN_VALID_SAMPLES → write -1 sentinels for the day.
  5. Resample / align to 1-min grid; step-hold setpoint to each minute.
  6. Integrate:
       gross_loss_kWh   = Σ max(potential − active, 0) × (1/60)
       curtail_loss_kWh = Σ max(potential − max(ceiling, active), 0) × (1/60)
                         when setpoint_pct < 99.5, else 0
       potential_energy_kWh = Σ potential × (1/60)
       exported_energy_kWh  = Σ active   × (1/60)
  7. Revenue = loss_kWh × tariff_rate_lkr  (using tariff at compute time).
  8. Write six daily timeseries keys + 2 metadata strings at day local-midnight ts.
  9. Update lifetime cumulative attributes (increment or recompute).
 10. Roll up daily values + lifetime attributes to all isPlantAgg ancestors.

Sentinel = -1 for every numeric key when data is invalid.
No plant-specific constants anywhere in this file.
"""
from __future__ import annotations

import asyncio
import logging
import math
from datetime import date as _date
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional, Set, Tuple
from zoneinfo import ZoneInfo

import pandas as pd

from app.config import settings

log = logging.getLogger(__name__)

# ── Telemetry key names (daily timeseries) ──────────────────────────────────

KEY_LOSS_GRID_DAILY       = "loss_grid_daily_kwh"
KEY_LOSS_CURTAIL_DAILY    = "loss_curtail_daily_kwh"
KEY_LOSS_REVENUE_DAILY    = "loss_revenue_daily_lkr"
KEY_LOSS_CURTAILREV_DAILY = "loss_curtail_revenue_daily_lkr"
KEY_LOSS_TARIFF_DAILY     = "loss_tariff_rate_lkr_at_compute"
KEY_POTENTIAL_DAILY       = "potential_energy_daily_kwh"
KEY_EXPORTED_DAILY        = "exported_energy_daily_kwh"
KEY_LOSS_DATA_SOURCE      = "loss_data_source"
KEY_LOSS_MODEL_VERSION    = "loss_model_version"

# RETIRED (round-1 deviation — never consumed by any widget):
#   potential_energy_monthly_kwh  — removed 2026-05-04
#   potential_energy_yearly_kwh   — removed 2026-05-04

# ── Lifetime attribute names (SERVER_SCOPE) ──────────────────────────────────

ATTR_GRID_LIFETIME        = "loss_grid_lifetime_kwh"
ATTR_CURTAIL_LIFETIME     = "loss_curtail_lifetime_kwh"
ATTR_REVENUE_LIFETIME     = "loss_revenue_lifetime_lkr"
ATTR_CURTAILREV_LIFETIME  = "loss_curtail_revenue_lifetime_lkr"
ATTR_POTENTIAL_LIFETIME   = "potential_energy_lifetime_kwh"
ATTR_EXPORTED_LIFETIME    = "exported_energy_lifetime_kwh"
ATTR_ANCHOR_DATE          = "loss_lifetime_anchor_date"
ATTR_UPDATED_AT           = "loss_lifetime_updated_at"

LIFETIME_ATTRS = [
    ATTR_GRID_LIFETIME,
    ATTR_CURTAIL_LIFETIME,
    ATTR_REVENUE_LIFETIME,
    ATTR_CURTAILREV_LIFETIME,
    ATTR_POTENTIAL_LIFETIME,
    ATTR_EXPORTED_LIFETIME,
]

LOSS_MODEL_VERSION = "loss-rollup-v1"


async def _empty_dict() -> dict:
    """Async no-op that returns an empty dict; used in asyncio.gather calls."""
    return {}


# Mirror of forecast_service.KEY_POTENTIAL_POWER — defined here to avoid importing
# forecast_service (which drags in pvlib) from this module.  The value is a contract
# constant and will not change without a 90-day deprecation notice in TELEMETRY_CONTRACT.md.
_KEY_POTENTIAL_POWER = "potential_power"

# Setpoint threshold below which we compute curtailment ceiling (mirror widget)
CURTAIL_SETPOINT_THRESHOLD = 99.5


# ── Public entry point ───────────────────────────────────────────────────────

async def run_loss_rollup(
    tb_client,
    date: Optional[datetime] = None,
) -> Dict[str, object]:
    """Compute and write daily loss attribution for all pvlib-enabled plants.

    Parameters
    ----------
    tb_client : ThingsBoardClient
        Authenticated TB client (singleton from app.state).
    date : datetime, optional
        Specific date to compute.  If None, computes for the calendar day that
        just ended (yesterday in local tz).  Pass a tz-aware or tz-naive UTC/local
        datetime; the function aligns it to local midnight.

    Returns
    -------
    dict
        {plants_ok, plants_failed, plants_skipped, date, per_plant}
    """
    tz = ZoneInfo(settings.TZ_LOCAL)
    now_local = datetime.now(tz)

    if date is None:
        local_midnight_today = now_local.replace(hour=0, minute=0, second=0, microsecond=0)
        day_end_local = local_midnight_today
    else:
        day_end_local = date.astimezone(tz).replace(hour=0, minute=0, second=0, microsecond=0)

    day_start_local = day_end_local - timedelta(days=1)
    day_start_utc   = day_start_local.astimezone(timezone.utc)
    day_end_utc     = day_end_local.astimezone(timezone.utc)

    # Timestamp to stamp daily values: local midnight of the calendar day
    day_ts_ms = int(day_start_local.timestamp() * 1000)

    log.info(
        "run_loss_rollup: computing %s → %s (local: %s → %s)",
        day_start_utc.isoformat(), day_end_utc.isoformat(),
        day_start_local.date(), day_end_local.date(),
    )

    plants, ancestor_map = await tb_client.discover_plants(settings.root_asset_ids)

    if not plants:
        log.warning("run_loss_rollup: no pvlib-enabled plants found")
        return {"plants_ok": 0, "plants_failed": 0, "plants_skipped": 0,
                "date": str(day_start_local.date())}

    stats: Dict[str, int] = {"ok": 0, "failed": 0, "skipped": 0}
    per_plant: Dict[str, dict] = {}

    # Process plants with concurrency limit
    sem = asyncio.Semaphore(settings.MAX_CONCURRENT_PLANTS)

    async def _process(plant):
        async with sem:
            return await _process_plant(
                tb_client, plant.id, day_start_utc, day_end_utc, day_ts_ms, day_start_local
            )

    results = await asyncio.gather(*(_process(p) for p in plants), return_exceptions=True)

    plant_daily: Dict[str, Dict[str, float]] = {}

    for plant, result in zip(plants, results):
        if isinstance(result, Exception):
            log.error("run_loss_rollup: unhandled exception for %s: %s", plant.id, result)
            result = _make_sentinel_result("error:unhandled_exception")
            stats["failed"] += 1
        elif result.get("skipped"):
            stats["skipped"] += 1
        elif result.get("ok"):
            stats["ok"] += 1
        else:
            stats["failed"] += 1

        per_plant[plant.id] = result
        daily_vals = result.get("daily_values", _sentinel_daily_values())
        plant_daily[plant.id] = daily_vals

    # ── Ancestor roll-up of daily keys ──────────────────────────────────────
    ancestor_children: Dict[str, Set[str]] = {}
    for plant_id, ancestors in ancestor_map.items():
        for anc_id in ancestors:
            ancestor_children.setdefault(anc_id, set()).add(plant_id)

    for ancestor_id, child_ids in ancestor_children.items():
        summed = _sum_daily_values([
            plant_daily[cid]
            for cid in child_ids
            if cid in plant_daily
        ])
        await _safe_write_daily(tb_client, ancestor_id, day_ts_ms, summed, "rollup")

    # ── Lifetime update: plants ──────────────────────────────────────────────
    for plant in plants:
        if per_plant.get(plant.id, {}).get("skipped"):
            continue
        daily = plant_daily[plant.id]
        await _update_lifetime(tb_client, plant.id, day_start_local.date(), daily)

    # ── Lifetime update: ancestors ───────────────────────────────────────────
    for ancestor_id, child_ids in ancestor_children.items():
        # Sum children's lifetime attributes; skip sentinel children
        child_lifetimes = []
        for cid in child_ids:
            lt = await _read_lifetime_attrs(tb_client, cid)
            if lt is not None:
                child_lifetimes.append(lt)

        if not child_lifetimes:
            continue

        summed_lt = _sum_lifetime_attrs(child_lifetimes)
        anchor = str(day_start_local.date())
        updated_at = datetime.now(timezone.utc).isoformat()
        summed_lt[ATTR_ANCHOR_DATE] = anchor
        summed_lt[ATTR_UPDATED_AT] = updated_at
        await _safe_post_attributes(tb_client, ancestor_id, summed_lt)

    log.info(
        "run_loss_rollup: done — ok=%d failed=%d skipped=%d date=%s",
        stats["ok"], stats["failed"], stats["skipped"], day_start_local.date(),
    )

    return {
        "plants_ok": stats["ok"],
        "plants_failed": stats["failed"],
        "plants_skipped": stats["skipped"],
        "date": str(day_start_local.date()),
        "per_plant": {k: {k2: v2 for k2, v2 in v.items() if k2 != "daily_values"}
                      for k, v in per_plant.items()},
    }


# ── Today-partial roll-up (5-min cadence, daylight hours only) ───────────────

async def run_today_partial_rollup(tb_client) -> Dict[str, object]:
    """Recompute today-so-far daily keys for every pvlib-enabled plant.

    Writes the same six daily keys at today's local-midnight ts, overwriting any
    prior partial.  Does NOT update lifetime attributes — those are advanced only
    by the 00:10 cron when the day is finalised.

    Data source value: ``"ok:partial"`` for plant rows; ``"rollup:partial"`` for
    ancestor roll-ups.  Downstream reports can filter on this flag.
    """
    tz = ZoneInfo(settings.TZ_LOCAL)
    now_local = datetime.now(tz)
    day_start_local = now_local.replace(hour=0, minute=0, second=0, microsecond=0)
    day_start_utc = day_start_local.astimezone(timezone.utc)
    end_utc = now_local.astimezone(timezone.utc)
    day_ts_ms = int(day_start_local.timestamp() * 1000)

    log.info(
        "run_today_partial_rollup: computing today-so-far %s → %s (local: %s)",
        day_start_utc.isoformat(), end_utc.isoformat(), day_start_local.date(),
    )

    plants, ancestor_map = await tb_client.discover_plants(settings.root_asset_ids)
    if not plants:
        log.warning("run_today_partial_rollup: no pvlib-enabled plants found")
        return {"plants_ok": 0, "plants_failed": 0, "plants_skipped": 0,
                "date": str(day_start_local.date()), "partial": True}

    sem = asyncio.Semaphore(settings.MAX_CONCURRENT_PLANTS)
    min_samples = settings.LOSS_TODAY_PARTIAL_MIN_SAMPLES

    async def _process(plant):
        async with sem:
            return await _process_plant(
                tb_client, plant.id, day_start_utc, end_utc,
                day_ts_ms, day_start_local,
                min_samples_override=min_samples,
            )

    results = await asyncio.gather(*(_process(p) for p in plants), return_exceptions=True)

    stats: Dict[str, int] = {"ok": 0, "failed": 0, "skipped": 0}
    plant_daily: Dict[str, Dict[str, float]] = {}

    for plant, result in zip(plants, results):
        if isinstance(result, Exception):
            log.error("run_today_partial_rollup: unhandled exception for %s: %s", plant.id, result)
            stats["failed"] += 1
            plant_daily[plant.id] = _sentinel_daily_values()
            continue

        if result.get("skipped"):
            stats["skipped"] += 1
        elif result.get("ok"):
            stats["ok"] += 1
            # Re-stamp data_source as "ok:partial" (or "warn:no_tariff") to distinguish
            # these intra-day rows from the finalised daily values written at 00:10.
            ds = result.get("data_source", "ok")
            partial_ds = "warn:no_tariff" if ds == "warn:no_tariff" else "ok:partial"
            await _safe_write_daily(
                tb_client, plant.id, day_ts_ms,
                result["daily_values"], partial_ds,
            )
        else:
            stats["failed"] += 1
        plant_daily[plant.id] = result.get("daily_values", _sentinel_daily_values())

    # ── Ancestor roll-up (daily only — no lifetime) ──────────────────────────
    ancestor_children: Dict[str, Set[str]] = {}
    for plant_id, ancestors in ancestor_map.items():
        for anc_id in ancestors:
            ancestor_children.setdefault(anc_id, set()).add(plant_id)

    for ancestor_id, child_ids in ancestor_children.items():
        summed = _sum_daily_values([plant_daily[cid] for cid in child_ids if cid in plant_daily])
        await _safe_write_daily(tb_client, ancestor_id, day_ts_ms, summed, "rollup:partial")

    log.info(
        "run_today_partial_rollup: done — ok=%d failed=%d skipped=%d date=%s",
        stats["ok"], stats["failed"], stats["skipped"], day_start_local.date(),
    )

    return {
        "plants_ok": stats["ok"],
        "plants_failed": stats["failed"],
        "plants_skipped": stats["skipped"],
        "date": str(day_start_local.date()),
        "partial": True,
    }


# ── Single-plant processing ──────────────────────────────────────────────────

async def _process_plant(
    tb_client,
    plant_id: str,
    day_start_utc: datetime,
    day_end_utc: datetime,
    day_ts_ms: int,
    day_start_local_date,  # datetime in local tz (for anchor date)
    min_samples_override: Optional[int] = None,
) -> Dict[str, object]:
    """Compute and write one plant's daily loss keys.  Returns status dict."""
    try:
        attrs = await tb_client.get_asset_attributes(plant_id)
    except Exception as exc:
        log.error("_process_plant: failed to read attrs for %s: %s", plant_id, exc)
        return _make_sentinel_result("error:attribute_read_failed")

    # Check per-plant loss attribution override
    loss_attr_enabled = attrs.get("loss_attribution_enabled")
    if loss_attr_enabled is not None and not _truthy(loss_attr_enabled):
        log.info("_process_plant: skipping %s (loss_attribution_enabled=false)", plant_id)
        return {"skipped": True, "reason": "loss_attribution_enabled=false",
                "daily_values": _sentinel_daily_values()}

    # Parse plant attributes
    tariff = _parse_float(attrs.get("tariff_rate_lkr"))
    capacity_raw = _parse_float(attrs.get("Capacity"))
    capacity_unit = str(attrs.get("capacityUnit") or attrs.get("capacity_unit") or "kW").strip()
    capacity_kw = (capacity_raw * 1000) if capacity_unit == "MW" else capacity_raw
    if not _is_positive(capacity_kw):
        capacity_kw = float(settings.dict().get("fallbackPower", 1000))

    active_power_unit = str(attrs.get("active_power_unit") or "kW").strip()
    w_to_kw = (active_power_unit.upper() == "W")

    # Setpoint keys: per-plant 'setpoint_keys' attribute overrides the service default
    sp_keys_raw = attrs.get("setpoint_keys") or settings.LOSS_DEFAULT_SETPOINT_KEYS
    setpoint_keys = [k.strip() for k in str(sp_keys_raw).split(",") if k.strip()]

    # Actual power keys: per-plant 'actual_power_keys' attribute, else default
    actual_keys_raw = attrs.get("actual_power_keys") or "active_power"
    actual_power_keys = [k.strip() for k in str(actual_keys_raw).split(",") if k.strip()]

    # ── Fetch series concurrently ────────────────────────────────────────────
    setpoint_start_utc = day_start_utc - timedelta(days=30)

    try:
        potential_raw, actual_raw, setpoint_raw = await asyncio.gather(
            tb_client.get_timeseries(
                "ASSET", plant_id, [_KEY_POTENTIAL_POWER],
                start=day_start_utc, end=day_end_utc, limit=100_000,
            ),
            _fetch_first_actual(
                tb_client, plant_id, actual_power_keys,
                day_start_utc, day_end_utc,
            ),
            tb_client.get_timeseries(
                "ASSET", plant_id, setpoint_keys,
                start=setpoint_start_utc, end=day_end_utc, limit=100_000,
            ) if setpoint_keys else _empty_dict(),
        )
    except Exception as exc:
        log.error("_process_plant: timeseries fetch failed for %s: %s", plant_id, exc)
        await _safe_write_daily(
            tb_client, plant_id, day_ts_ms,
            _sentinel_daily_values(), "error:fetch_failed",
        )
        return _make_sentinel_result("error:fetch_failed")

    # ── Parse and validate series ────────────────────────────────────────────
    potential_records = potential_raw.get(_KEY_POTENTIAL_POWER, [])
    actual_records = actual_raw  # already a list from _fetch_first_actual

    if not potential_records:
        log.info("_process_plant: no potential_power for %s — writing -1", plant_id)
        await _safe_write_daily(
            tb_client, plant_id, day_ts_ms,
            _sentinel_daily_values(), "error:no_potential",
        )
        return _make_sentinel_result("error:no_potential")

    if not actual_records:
        log.info("_process_plant: no actual_power for %s — writing -1", plant_id)
        await _safe_write_daily(
            tb_client, plant_id, day_ts_ms,
            _sentinel_daily_values(), "error:no_actual",
        )
        return _make_sentinel_result("error:no_actual")

    # Build pandas Series, drop sentinels (< 0)
    potential_series = _records_to_series(potential_records, skip_negative=True)
    actual_series = _records_to_series(actual_records, skip_negative=True)

    if w_to_kw and len(actual_series) > 0:
        actual_series = actual_series * 0.001

    min_samples = min_samples_override if min_samples_override is not None else settings.LOSS_MIN_VALID_SAMPLES
    if len(potential_series) < min_samples or len(actual_series) < min_samples:
        log.warning(
            "_process_plant: insufficient samples for %s "
            "(potential=%d, actual=%d, min=%d) — writing -1",
            plant_id, len(potential_series), len(actual_series), min_samples,
        )
        await _safe_write_daily(
            tb_client, plant_id, day_ts_ms,
            _sentinel_daily_values(), "error:insufficient_samples",
        )
        return _make_sentinel_result("error:insufficient_samples")

    # ── Build setpoint step-hold series ─────────────────────────────────────
    setpoint_series = _build_setpoint_series(setpoint_raw, setpoint_keys)

    # ── Integration ──────────────────────────────────────────────────────────
    try:
        daily = _integrate(
            potential_series, actual_series, setpoint_series,
            capacity_kw, day_start_utc, day_end_utc,
        )
    except Exception as exc:
        log.error("_process_plant: integration failed for %s: %s", plant_id, exc)
        await _safe_write_daily(
            tb_client, plant_id, day_ts_ms,
            _sentinel_daily_values(), "error:integration_failed",
        )
        return _make_sentinel_result("error:integration_failed")

    # ── Revenue computation ──────────────────────────────────────────────────
    if _is_positive_or_zero(tariff) and tariff is not None:
        daily[KEY_LOSS_REVENUE_DAILY]    = round(daily[KEY_LOSS_GRID_DAILY] * tariff, 2)
        daily[KEY_LOSS_CURTAILREV_DAILY] = round(daily[KEY_LOSS_CURTAIL_DAILY] * tariff, 2)
        daily[KEY_LOSS_TARIFF_DAILY]     = float(tariff)
        data_source = "ok"
    else:
        daily[KEY_LOSS_REVENUE_DAILY]    = -1
        daily[KEY_LOSS_CURTAILREV_DAILY] = -1
        daily[KEY_LOSS_TARIFF_DAILY]     = -1
        data_source = "warn:no_tariff"
        log.info("_process_plant: tariff missing for %s — LKR keys = -1", plant_id)

    # ── Write daily keys ─────────────────────────────────────────────────────
    await _safe_write_daily(tb_client, plant_id, day_ts_ms, daily, data_source)

    return {
        "ok": True,
        "daily_values": daily,
        "data_source": data_source,
        "w_to_kw": w_to_kw,
        "tariff": tariff,
    }


# ── Integration math ─────────────────────────────────────────────────────────

def _integrate(
    potential: pd.Series,
    actual: pd.Series,
    setpoint: pd.Series,
    capacity_kw: float,
    day_start_utc: datetime,
    day_end_utc: datetime,
) -> Dict[str, float]:
    """Resample to 1-min grid and compute loss integrals.

    Returns dict with six scalar float values (kWh / LKR keys NOT set here —
    revenue is computed after this function based on tariff availability).
    """
    # 1-minute DatetimeIndex spanning the day
    idx = pd.date_range(day_start_utc, day_end_utc, freq="1min", inclusive="left", tz="UTC")

    # Filter to solar window: 05:00 to 19:00 local time
    idx_local = idx.tz_convert(settings.TZ_LOCAL)
    idx_filtered = idx[(idx_local.hour >= 5) & (idx_local.hour < 19)]

    # Resample onto 1-min grid: mean within each minute bucket
    pot_1min = potential.resample("1min").mean().reindex(idx)
    act_1min = actual.resample("1min").mean().reindex(idx)

    # Forward-fill setpoint: step-hold the last known value into each minute
    if setpoint is not None and len(setpoint) > 0:
        # Reindex onto the full minute range (which extends back 30 days for lookback)
        combined_idx = setpoint.index.union(idx)
        sp_full = setpoint.reindex(combined_idx, method="ffill")
        sp_1min = sp_full.reindex(idx)
    else:
        sp_1min = pd.Series(100.0, index=idx)

    # Fill remaining NaNs in setpoint with 100 (no curtailment)
    sp_1min = sp_1min.fillna(100.0)

    # Integrate: each 1-min slot contributes value × (1/60) kWh
    h_per_min = 1.0 / 60.0

    gross_loss_kwh = 0.0
    curtail_loss_kwh = 0.0
    potential_energy_kwh = 0.0
    exported_energy_kwh = 0.0

    for ts in idx_filtered:
        pot_v = pot_1min.get(ts, float("nan"))
        act_v = act_1min.get(ts, float("nan"))
        sp_v = float(sp_1min.get(ts, 100.0))

        if not math.isnan(pot_v) and pot_v >= 0:
            potential_energy_kwh += pot_v * h_per_min

        if not math.isnan(act_v) and act_v >= 0:
            exported_energy_kwh += act_v * h_per_min

        if not math.isnan(pot_v) and not math.isnan(act_v) and pot_v >= 0 and act_v >= 0:
            gross_loss_kwh += max(pot_v - act_v, 0.0) * h_per_min

            if sp_v < CURTAIL_SETPOINT_THRESHOLD:
                ceiling_kw = capacity_kw * (sp_v / 100.0)
                curtail_base_kw = max(ceiling_kw, act_v)
                curtail_loss_kwh += max(pot_v - curtail_base_kw, 0.0) * h_per_min

    return {
        KEY_LOSS_GRID_DAILY:       round(gross_loss_kwh, 3),
        KEY_LOSS_CURTAIL_DAILY:    round(curtail_loss_kwh, 3),
        KEY_POTENTIAL_DAILY:       round(potential_energy_kwh, 3),
        KEY_EXPORTED_DAILY:        round(exported_energy_kwh, 3),
        # Revenue keys filled by caller after tariff check
        KEY_LOSS_REVENUE_DAILY:    -1,
        KEY_LOSS_CURTAILREV_DAILY: -1,
    }


# ── Lifetime attribute maintenance ────────────────────────────────────────────

async def _update_lifetime(
    tb_client,
    entity_id: str,
    today: _date,
    daily: Dict[str, float],
) -> None:
    """Increment or recompute lifetime cumulative attributes for one entity.

    Logic:
      - If today == anchor_date + 1 day AND all six daily values ≥ 0:
            new_lifetime = old_lifetime + today_daily; advance anchor.
      - If today > anchor_date + 1 day (gap detected):
            fall through to full recompute.
      - If today ≤ anchor_date (re-run or backfill):
            full recompute.
    """
    # Skip if today's daily is all-sentinel
    if _all_sentinel(daily):
        log.debug("_update_lifetime: all sentinel for %s on %s — skipping", entity_id, today)
        return

    try:
        attrs = await tb_client.get_asset_attributes(entity_id)
    except Exception as exc:
        log.error("_update_lifetime: failed to read attrs for %s: %s", entity_id, exc)
        return

    anchor_str = attrs.get(ATTR_ANCHOR_DATE)
    anchor: Optional[_date] = None
    if anchor_str:
        try:
            anchor = _date.fromisoformat(str(anchor_str))
        except ValueError:
            pass

    if anchor is not None and today == anchor + timedelta(days=1):
        # Happy path: increment by today's values
        await _lifetime_increment_step(tb_client, entity_id, attrs, daily, today)
    else:
        if anchor is not None:
            if today <= anchor:
                log.info("_update_lifetime: re-run for %s (today=%s <= anchor=%s) — recomputing",
                         entity_id, today, anchor)
            else:
                log.warning("_update_lifetime: gap detected for %s "
                            "(today=%s, anchor=%s) — recomputing", entity_id, today, anchor)
        else:
            log.info("_update_lifetime: no anchor for %s — recomputing", entity_id)
        await _recompute_lifetime_from_history(tb_client, entity_id)


async def _lifetime_increment_step(
    tb_client,
    entity_id: str,
    existing_attrs: dict,
    daily: Dict[str, float],
    today: _date,
) -> None:
    """Add today's daily values to existing lifetime attributes."""
    daily_to_lifetime = {
        KEY_LOSS_GRID_DAILY:       ATTR_GRID_LIFETIME,
        KEY_LOSS_CURTAIL_DAILY:    ATTR_CURTAIL_LIFETIME,
        KEY_LOSS_REVENUE_DAILY:    ATTR_REVENUE_LIFETIME,
        KEY_LOSS_CURTAILREV_DAILY: ATTR_CURTAILREV_LIFETIME,
        KEY_POTENTIAL_DAILY:       ATTR_POTENTIAL_LIFETIME,
        KEY_EXPORTED_DAILY:        ATTR_EXPORTED_LIFETIME,
    }

    new_attrs: Dict[str, object] = {}
    for daily_key, lifetime_attr in daily_to_lifetime.items():
        daily_val = daily.get(daily_key, -1)
        if daily_val < 0:
            # Partial sentinel: keep existing lifetime (don't regress)
            new_attrs[lifetime_attr] = _parse_float(existing_attrs.get(lifetime_attr)) or 0.0
        else:
            old = _parse_float(existing_attrs.get(lifetime_attr)) or 0.0
            if old < 0:
                old = 0.0
            new_attrs[lifetime_attr] = round(old + daily_val, 3)

    new_attrs[ATTR_ANCHOR_DATE] = str(today)
    new_attrs[ATTR_UPDATED_AT] = datetime.now(timezone.utc).isoformat()

    await _safe_post_attributes(tb_client, entity_id, new_attrs)
    log.debug("_lifetime_increment_step: incremented %s for %s", today, entity_id)


async def _recompute_lifetime_from_history(
    tb_client,
    entity_id: str,
    commissioning_date: Optional[_date] = None,
) -> Dict[str, float]:
    """Sum all daily loss keys from commissioning_date to today.

    Pages history in LOSS_LIFETIME_PAGE_DAYS-day chunks to avoid huge TB reads.
    Returns the summed values dict (also writes to TB).
    """
    today = datetime.now(ZoneInfo(settings.TZ_LOCAL)).date()
    tz = ZoneInfo(settings.TZ_LOCAL)

    if commissioning_date is None:
        try:
            attrs = await tb_client.get_asset_attributes(entity_id)
            cod_raw = attrs.get("commissioning_date")
            if cod_raw:
                commissioning_date = _date.fromisoformat(str(cod_raw)[:10])
        except Exception:
            pass

    if commissioning_date is None:
        commissioning_date = _date(2020, 10, 1)  # fleet-wide fallback

    daily_keys = [
        KEY_LOSS_GRID_DAILY,
        KEY_LOSS_CURTAIL_DAILY,
        KEY_LOSS_REVENUE_DAILY,
        KEY_LOSS_CURTAILREV_DAILY,
        KEY_POTENTIAL_DAILY,
        KEY_EXPORTED_DAILY,
    ]

    totals: Dict[str, float] = {k: 0.0 for k in daily_keys}
    page_days = settings.LOSS_LIFETIME_PAGE_DAYS

    cursor = commissioning_date
    while cursor <= today:
        page_end = min(cursor + timedelta(days=page_days), today + timedelta(days=1))
        start_utc = datetime(cursor.year, cursor.month, cursor.day, tzinfo=tz).astimezone(timezone.utc)
        end_utc   = datetime(page_end.year, page_end.month, page_end.day, tzinfo=tz).astimezone(timezone.utc)

        try:
            raw = await tb_client.get_timeseries(
                "ASSET", entity_id, daily_keys,
                start=start_utc, end=end_utc, limit=500,
            )
        except Exception as exc:
            log.error("_recompute_lifetime: fetch failed for %s [%s→%s]: %s",
                      entity_id, cursor, page_end, exc)
            cursor = page_end
            continue

        for key in daily_keys:
            for rec in raw.get(key, []):
                try:
                    v = float(rec["value"])
                    if v >= 0:
                        totals[key] += v
                except (KeyError, ValueError, TypeError):
                    pass

        cursor = page_end

    new_attrs: Dict[str, object] = {
        ATTR_GRID_LIFETIME:       round(totals[KEY_LOSS_GRID_DAILY], 3),
        ATTR_CURTAIL_LIFETIME:    round(totals[KEY_LOSS_CURTAIL_DAILY], 3),
        ATTR_REVENUE_LIFETIME:    round(totals[KEY_LOSS_REVENUE_DAILY], 3),
        ATTR_CURTAILREV_LIFETIME: round(totals[KEY_LOSS_CURTAILREV_DAILY], 3),
        ATTR_POTENTIAL_LIFETIME:  round(totals[KEY_POTENTIAL_DAILY], 3),
        ATTR_EXPORTED_LIFETIME:   round(totals[KEY_EXPORTED_DAILY], 3),
        ATTR_ANCHOR_DATE:         str(today),
        ATTR_UPDATED_AT:          datetime.now(timezone.utc).isoformat(),
    }

    await _safe_post_attributes(tb_client, entity_id, new_attrs)
    log.info("_recompute_lifetime: wrote lifetime attrs for %s (anchor=%s)", entity_id, today)

    return {
        ATTR_GRID_LIFETIME:       new_attrs[ATTR_GRID_LIFETIME],
        ATTR_CURTAIL_LIFETIME:    new_attrs[ATTR_CURTAIL_LIFETIME],
        ATTR_REVENUE_LIFETIME:    new_attrs[ATTR_REVENUE_LIFETIME],
        ATTR_CURTAILREV_LIFETIME: new_attrs[ATTR_CURTAILREV_LIFETIME],
        ATTR_POTENTIAL_LIFETIME:  new_attrs[ATTR_POTENTIAL_LIFETIME],
        ATTR_EXPORTED_LIFETIME:   new_attrs[ATTR_EXPORTED_LIFETIME],
    }


# ── Fleet-wide recompute (called from /admin/recompute-lifetime) ─────────────

async def recompute_lifetime_for_fleet(
    tb_client,
    asset_id: Optional[str] = None,
) -> Dict[str, object]:
    """Recompute lifetime attributes for a single plant or the entire fleet.

    Parameters
    ----------
    tb_client : ThingsBoardClient
    asset_id : str, optional
        If provided, recompute only this asset.  If None, recompute all plants
        discovered from TB_ROOT_ASSET_IDS.

    Returns
    -------
    dict  {status, entities_ok, entities_failed}
    """
    if asset_id:
        try:
            await _recompute_lifetime_from_history(tb_client, asset_id)
            return {"status": "ok", "entities_ok": 1, "entities_failed": 0}
        except Exception as exc:
            return {"status": "error", "error": str(exc), "entities_ok": 0, "entities_failed": 1}

    plants, ancestor_map = await tb_client.discover_plants(settings.root_asset_ids)
    ok = failed = 0

    for plant in plants:
        try:
            await _recompute_lifetime_from_history(tb_client, plant.id)
            ok += 1
        except Exception as exc:
            log.error("recompute_lifetime: failed for %s: %s", plant.id, exc)
            failed += 1

    # Recompute ancestor lifetimes as sum of children
    ancestor_children: Dict[str, Set[str]] = {}
    for plant_id, ancestors in ancestor_map.items():
        for anc_id in ancestors:
            ancestor_children.setdefault(anc_id, set()).add(plant_id)

    for ancestor_id, child_ids in ancestor_children.items():
        child_lifetimes = []
        for cid in child_ids:
            lt = await _read_lifetime_attrs(tb_client, cid)
            if lt is not None:
                child_lifetimes.append(lt)
        if not child_lifetimes:
            continue
        summed = _sum_lifetime_attrs(child_lifetimes)
        today = str(datetime.now(ZoneInfo(settings.TZ_LOCAL)).date())
        summed[ATTR_ANCHOR_DATE] = today
        summed[ATTR_UPDATED_AT] = datetime.now(timezone.utc).isoformat()
        await _safe_post_attributes(tb_client, ancestor_id, summed)
        ok += 1

    return {"status": "ok", "entities_ok": ok, "entities_failed": failed}


# ── Loss status reader (for /admin/loss-status) ──────────────────────────────

async def get_loss_status(tb_client, asset_id: str) -> Dict[str, object]:
    """Return latest daily key values + all lifetime attributes for one asset."""
    daily_keys = [
        KEY_LOSS_GRID_DAILY,
        KEY_LOSS_CURTAIL_DAILY,
        KEY_LOSS_REVENUE_DAILY,
        KEY_LOSS_CURTAILREV_DAILY,
        KEY_LOSS_TARIFF_DAILY,
        KEY_POTENTIAL_DAILY,
        KEY_EXPORTED_DAILY,
        KEY_LOSS_DATA_SOURCE,
        KEY_LOSS_MODEL_VERSION,
    ]
    lifetime_attr_names = [
        ATTR_GRID_LIFETIME,
        ATTR_CURTAIL_LIFETIME,
        ATTR_REVENUE_LIFETIME,
        ATTR_CURTAILREV_LIFETIME,
        ATTR_POTENTIAL_LIFETIME,
        ATTR_EXPORTED_LIFETIME,
        ATTR_ANCHOR_DATE,
        ATTR_UPDATED_AT,
    ]

    latest_ts, server_attrs = await asyncio.gather(
        tb_client.get_latest_telemetry("ASSET", asset_id, daily_keys),
        tb_client.get_asset_attributes(asset_id),
        return_exceptions=True,
    )

    if isinstance(latest_ts, Exception):
        latest_ts = {}
    if isinstance(server_attrs, Exception):
        server_attrs = {}

    lifetime_out = {k: server_attrs.get(k) for k in lifetime_attr_names}

    return {
        "asset_id": asset_id,
        "latest_daily": latest_ts,
        "lifetime_attributes": lifetime_out,
    }


# ── Helper: fetch first matching actual power series ─────────────────────────

async def _fetch_first_actual(
    tb_client,
    plant_id: str,
    actual_keys: List[str],
    start: datetime,
    end: datetime,
) -> List[dict]:
    """Try each actual power key in order; return records from the first that has data."""
    for key in actual_keys:
        try:
            raw = await tb_client.get_timeseries(
                "ASSET", plant_id, [key],
                start=start, end=end, limit=100_000,
            )
            records = raw.get(key, [])
            if records:
                return records
        except Exception as exc:
            log.debug("_fetch_first_actual: key %s failed for %s: %s", key, plant_id, exc)
    return []


# ── Helper: write daily timeseries ───────────────────────────────────────────

async def _safe_write_daily(
    tb_client,
    entity_id: str,
    ts_ms: int,
    daily: Dict[str, float],
    data_source: str,
) -> None:
    """Write loss daily keys at ts_ms; best-effort (logs on failure)."""
    values: Dict[str, object] = {
        KEY_LOSS_GRID_DAILY:       daily.get(KEY_LOSS_GRID_DAILY, -1),
        KEY_LOSS_CURTAIL_DAILY:    daily.get(KEY_LOSS_CURTAIL_DAILY, -1),
        KEY_LOSS_REVENUE_DAILY:    daily.get(KEY_LOSS_REVENUE_DAILY, -1),
        KEY_LOSS_CURTAILREV_DAILY: daily.get(KEY_LOSS_CURTAILREV_DAILY, -1),
        KEY_LOSS_TARIFF_DAILY:     daily.get(KEY_LOSS_TARIFF_DAILY, -1),
        KEY_POTENTIAL_DAILY:       daily.get(KEY_POTENTIAL_DAILY, -1),
        KEY_EXPORTED_DAILY:        daily.get(KEY_EXPORTED_DAILY, -1),
        KEY_LOSS_DATA_SOURCE:      data_source,
        KEY_LOSS_MODEL_VERSION:    LOSS_MODEL_VERSION,
    }
    try:
        await tb_client.post_telemetry(
            "ASSET", entity_id,
            [{"ts": ts_ms, "values": values}],
        )
    except Exception as exc:
        log.error("_safe_write_daily: failed for %s: %s", entity_id, exc)

async def _safe_post_attributes(tb_client, entity_id: str, payload: dict) -> None:
    """Write SERVER_SCOPE attributes; best-effort."""
    try:
        await tb_client.post_attributes("ASSET", entity_id, "SERVER_SCOPE", payload)
    except Exception as exc:
        log.error("_safe_post_attributes: failed for %s: %s", entity_id, exc)


# ── Lifetime read / sum helpers ───────────────────────────────────────────────

async def _read_lifetime_attrs(tb_client, entity_id: str) -> Optional[Dict[str, float]]:
    """Return lifetime attribute floats for an entity; None if all missing."""
    try:
        attrs = await tb_client.get_asset_attributes(entity_id)
    except Exception:
        return None
    result = {}
    for attr in LIFETIME_ATTRS:
        v = _parse_float(attrs.get(attr))
        if v is None or v < 0:
            return None  # sentinel or missing — skip this entity from ancestor sums
        result[attr] = v
    return result if result else None


def _sum_lifetime_attrs(lifetimes: List[Dict[str, float]]) -> Dict[str, object]:
    """Sum a list of lifetime attr dicts, returning a new dict."""
    out: Dict[str, float] = {k: 0.0 for k in LIFETIME_ATTRS}
    for lt in lifetimes:
        for k in LIFETIME_ATTRS:
            out[k] = round(out[k] + lt.get(k, 0.0), 3)
    return out  # type: ignore


# ── Series helpers ────────────────────────────────────────────────────────────

def _records_to_series(records: List[dict], skip_negative: bool = True) -> pd.Series:
    """Convert TB timeseries record list to a UTC-indexed pandas Series."""
    data = {}
    for r in records:
        try:
            ts = pd.Timestamp(int(r["ts"]), unit="ms", tz="UTC")
            v = float(r["value"])
            if skip_negative and v < 0:
                continue
            data[ts] = v
        except (KeyError, ValueError, TypeError):
            continue
    return pd.Series(data, dtype=float).sort_index()


def _build_setpoint_series(raw: dict, keys: List[str]) -> pd.Series:
    """Merge all setpoint key records into a single step-hold series."""
    combined = {}
    for key in keys:
        for rec in raw.get(key, []):
            try:
                ts = pd.Timestamp(int(rec["ts"]), unit="ms", tz="UTC")
                v = float(rec["value"])
                if 0 <= v <= 100:
                    # Take the minimum setpoint at each timestamp (most restrictive)
                    combined[ts] = min(combined.get(ts, v), v)
            except (KeyError, ValueError, TypeError):
                continue
    if not combined:
        return pd.Series(dtype=float)
    return pd.Series(combined, dtype=float).sort_index()


def _sum_daily_values(values_list: List[Dict[str, float]]) -> Dict[str, float]:
    """Sum valid (≥0) daily values across a list; result is -1 if all sentinels."""
    keys = [
        KEY_LOSS_GRID_DAILY,
        KEY_LOSS_CURTAIL_DAILY,
        KEY_LOSS_REVENUE_DAILY,
        KEY_LOSS_CURTAILREV_DAILY,
        KEY_LOSS_TARIFF_DAILY,
        KEY_POTENTIAL_DAILY,
        KEY_EXPORTED_DAILY,
    ]
    totals: Dict[str, float] = {k: 0.0 for k in keys}
    has_valid: Dict[str, bool] = {k: False for k in keys}

    for dv in values_list:
        for k in keys:
            v = dv.get(k, -1)
            if v >= 0:
                totals[k] += v
                has_valid[k] = True

    result = {}
    for k in keys:
        result[k] = round(totals[k], 3) if has_valid[k] else -1.0
    return result


# ── Sentinel helpers ──────────────────────────────────────────────────────────

def _sentinel_daily_values() -> Dict[str, float]:
    return {
        KEY_LOSS_GRID_DAILY:       -1.0,
        KEY_LOSS_CURTAIL_DAILY:    -1.0,
        KEY_LOSS_REVENUE_DAILY:    -1.0,
        KEY_LOSS_CURTAILREV_DAILY: -1.0,
        KEY_LOSS_TARIFF_DAILY:     -1.0,
        KEY_POTENTIAL_DAILY:       -1.0,
        KEY_EXPORTED_DAILY:        -1.0,
    }


def _make_sentinel_result(data_source: str) -> Dict[str, object]:
    return {"ok": False, "daily_values": _sentinel_daily_values(), "data_source": data_source}


def _all_sentinel(daily: Dict[str, float]) -> bool:
    return all(v < 0 for v in daily.values() if isinstance(v, (int, float)))


# ── Misc utilities ────────────────────────────────────────────────────────────

def _parse_float(value) -> Optional[float]:
    if value is None:
        return None
    try:
        f = float(value)
        return f if math.isfinite(f) else None
    except (ValueError, TypeError):
        return None


def _is_positive(value) -> bool:
    f = _parse_float(value)
    return f is not None and f > 0


def _is_positive_or_zero(value) -> bool:
    f = _parse_float(value)
    return f is not None and f >= 0


def _truthy(value) -> bool:
    """Convert TB attribute values to bool safely (mirrors thingsboard_client._truthy)."""
    if value is None:
        return False
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return value != 0
    s = str(value).strip().lower()
    return s in ("true", "1", "yes", "y", "on")
