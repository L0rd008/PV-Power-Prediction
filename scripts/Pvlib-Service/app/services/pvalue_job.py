"""
P-value probabilistic forecast job — P50 / P90 / P95 daily + monthly telemetry.

Algorithm (Phase 2 — per-calendar-day percentiles):
  1. Discover pvlib-enabled plants via tb_client.discover_plants().
  2. Deduplicate by ERA5 grid cell (lat/lon rounded to nearest 0.25°).
     ~1000 Sri Lanka plants → ~70 unique cells → ~70 PVGIS API calls.
  3. Per unique cell: fetch PVGIS-ERA5 hourly data (PVGIS_START_YEAR–PVGIS_END_YEAR).
     Single HTTP call returns all years in one response.
  4. Per plant (uses shared cell weather, different physics config each time):
       a. Prepare DataFrame: rename PVGIS columns → pipeline canonical names.
       b. Run compute_ac_power() for each calendar year slice.
       c. Integrate daily kWh: resample('D').sum()
          CRITICAL: hourly cadence → each kW row = 1 kWh; do NOT divide by 60.
       d. Concatenate all years → full daily series.
  5. Per-calendar-day percentiles (Phase 2):
       Group daily series by (month, day_of_month) across all 19 years.
       Each group has up to 19 values → P50/P90/P95 per calendar day.
       Result: 365 unique daily P values (intra-month variation captured).
       Note: with n=19, P95=quantile(0.05) ≈ minimum value — conservative but acceptable
       for risk signalling. Phase 3 upgrade: extend to 30-year ERA5 dataset.
  6. Monthly P values: sum of per-day P values within each month (consistent totals).
  7. Annual P50/P90/P95: percentile of full-year totals (19 values — true annual stat).
  8. Write to ThingsBoard:
       Timeseries (365 daily rows, ts = local midnight of target year):
         forecast_p50_daily, forecast_p90_daily, forecast_p95_daily  [MWh]
       Timeseries (12 monthly rows, ts = 1st-of-month midnight of target year):
         forecast_p50_monthly, forecast_p90_monthly, forecast_p95_monthly [MWh]
       SERVER_SCOPE attributes:
         p50_energy, p90_energy, p95_energy  [kWh — annual total]

PVGIS column mapping:
  PVGIS 'G(h)'   → pipeline 'ghi'        [W/m²]
  PVGIS 'T2m'    → pipeline 'air_temp'   [°C]
  PVGIS 'WS10m'  → pipeline 'wind_speed' [m/s]
  (Gb(n)/Gd(h) available but pipeline handles decomposition internally via Erbs)
"""
from __future__ import annotations

import asyncio
import calendar
import logging
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Set, Tuple
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

from app.config import settings
from app.physics.config import PlantConfig
from app.physics.pipeline import compute_ac_power

log = logging.getLogger(__name__)

# ── Telemetry key constants ──────────────────────────────────────────────────

KEY_P50_DAILY     = "forecast_p50_daily"      # MWh — daily P50 expected energy
KEY_P90_DAILY     = "forecast_p90_daily"      # MWh — daily P90 expected energy
KEY_P95_DAILY     = "forecast_p95_daily"      # MWh — daily P95 expected energy

KEY_P50_MONTHLY   = "forecast_p50_monthly"    # MWh — monthly P50 expected energy
KEY_P90_MONTHLY   = "forecast_p90_monthly"    # MWh — monthly P90 expected energy
KEY_P95_MONTHLY   = "forecast_p95_monthly"    # MWh — monthly P95 expected energy

ATTR_P50_ENERGY   = "p50_energy"              # kWh — annual P50 (SERVER_SCOPE attr)
ATTR_P90_ENERGY   = "p90_energy"              # kWh — annual P90
ATTR_P95_ENERGY   = "p95_energy"              # kWh — annual P95

PVALUE_MODEL_VER  = "pvalue-daily-v2"          # Phase 2: per-calendar-day percentiles

# ERA5 grid resolution — plants within the same cell share a PVGIS fetch
ERA5_GRID_DEG = 0.25


# ── Public entry point ───────────────────────────────────────────────────────

async def run_pvalue_job(
    tb_client,
    target_year: Optional[int] = None,
    plant_ids: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """Compute and write P50/P90/P95 forecast telemetry for all pvlib-enabled plants.

    Parameters
    ----------
    tb_client : ThingsBoardClient
        Authenticated TB client (singleton from app.state).
    target_year : int, optional
        Calendar year to stamp daily/monthly timeseries against.
        Defaults to current local year. Pass explicitly for backfill.
    plant_ids : list[str], optional
        Restrict run to specific asset UUIDs (smoke-test / single-plant mode).
        None (default) processes the full fleet.

    Returns
    -------
    dict
        Summary: {plants_ok, plants_failed, cells_fetched, target_year}
    """
    tz = ZoneInfo(settings.TZ_LOCAL)
    year = target_year or datetime.now(tz).year
    log.info("run_pvalue_job: start — target_year=%d start=%d end=%d db=%s",
             year, settings.PVGIS_START_YEAR, settings.PVGIS_END_YEAR,
             settings.PVGIS_RADDATABASE)

    # ── 1. Discover plants ────────────────────────────────────────────────────
    from app.config import settings as _s
    plants, _ = await tb_client.discover_plants(_s.root_asset_ids, force=True)
    if not plants:
        log.warning("run_pvalue_job: no pvlib-enabled plants found")
        return {"plants_ok": 0, "plants_failed": 0, "cells_fetched": 0, "target_year": year}

    # Optional single-plant filter (smoke-test mode)
    if plant_ids:
        plants = [p for p in plants if p.id in plant_ids]
        if not plants:
            log.warning("run_pvalue_job: none of provided plant_ids match discovered plants")
            return {"plants_ok": 0, "plants_failed": 0, "cells_fetched": 0, "target_year": year}

    log.info("run_pvalue_job: processing %d plant(s)", len(plants))

    # ── 2. Load plant configs and deduplicate by ERA5 grid cell ──────────────
    plant_configs: Dict[str, PlantConfig] = {}
    for plant in plants:
        try:
            attrs = await tb_client.get_asset_attributes(plant.id)
            if not attrs:
                log.warning("run_pvalue_job: no attributes for plant %s — skipping", plant.id)
                continue
            config = PlantConfig.from_tb_attributes(plant.id, attrs)
            if config.latitude == 0.0 and config.longitude == 0.0:
                log.warning("run_pvalue_job: plant %s has lat/lon = 0,0 — skipping", plant.id)
                continue
            plant_configs[plant.id] = config
        except Exception as exc:
            log.error("run_pvalue_job: config load failed for %s: %s", plant.id, exc)

    if not plant_configs:
        log.error("run_pvalue_job: no valid plant configs loaded — aborting")
        return {"plants_ok": 0, "plants_failed": len(plants), "cells_fetched": 0, "target_year": year}

    # Build cell → [plant_ids] mapping
    cell_plants: Dict[Tuple[float, float], List[str]] = {}
    for pid, cfg in plant_configs.items():
        cell = _era5_cell(cfg.latitude, cfg.longitude)
        cell_plants.setdefault(cell, []).append(pid)

    log.info("run_pvalue_job: %d plant(s) → %d unique ERA5 cell(s)",
             len(plant_configs), len(cell_plants))

    # ── 3. Fetch PVGIS per unique cell (sequential — rate limit friendly) ─────
    cell_weather: Dict[Tuple[float, float], pd.DataFrame] = {}
    cells_fetched = 0

    for cell_lat, cell_lon in cell_plants:
        try:
            df_hourly = await _fetch_pvgis_cell(cell_lat, cell_lon)
            cell_weather[(cell_lat, cell_lon)] = df_hourly
            cells_fetched += 1
            log.info("run_pvalue_job: fetched cell (%.2f, %.2f) — %d rows",
                     cell_lat, cell_lon, len(df_hourly))
        except Exception as exc:
            log.error("run_pvalue_job: PVGIS fetch failed for cell (%.2f, %.2f): %s — "
                      "skipping %d plant(s) in this cell",
                      cell_lat, cell_lon, exc, len(cell_plants[(cell_lat, cell_lon)]))

    # ── 4–7. Simulate per plant and write to TB ───────────────────────────────
    stats = {"ok": 0, "failed": 0}

    for plant_id, config in plant_configs.items():
        cell = _era5_cell(config.latitude, config.longitude)
        df_hourly = cell_weather.get(cell)
        if df_hourly is None:
            log.error("run_pvalue_job: no weather data for plant %s (cell fetch failed)", plant_id)
            stats["failed"] += 1
            continue

        try:
            await _process_plant(tb_client, plant_id, config, df_hourly, year)
            stats["ok"] += 1
            log.info("run_pvalue_job: wrote P-values for plant %s (%s)",
                     plant_id, config.plant_name)
        except Exception as exc:
            log.error("run_pvalue_job: failed for plant %s (%s): %s",
                      plant_id, config.plant_name, exc)
            stats["failed"] += 1

    summary = {
        "plants_ok": stats["ok"],
        "plants_failed": stats["failed"],
        "cells_fetched": cells_fetched,
        "target_year": year,
    }
    log.info("run_pvalue_job: done — %s", summary)
    return summary


# ── Per-plant processing ─────────────────────────────────────────────────────

async def _process_plant(
    tb_client,
    plant_id: str,
    config: PlantConfig,
    df_pvgis: pd.DataFrame,
    target_year: int,
) -> None:
    """Simulate, compute percentiles, and write all 9 P-value outputs for one plant."""

    tz_str = config.timezone
    tz = ZoneInfo(tz_str)

    # ── Prepare PVGIS weather DataFrame ──────────────────────────────────────
    # PVGIS-ERA5 column names (outputformat='json', components=True):
    #   G(h) = GHI [W/m²], T2m = air temp [°C], WS10m = wind speed [m/s]
    df_weather = _prepare_weather(df_pvgis, config)

    # ── Run per-year simulation ───────────────────────────────────────────────
    all_daily_kwh = _simulate_all_years(config, df_weather, tz)

    if all_daily_kwh.empty:
        raise RuntimeError("simulation produced no daily data across all years")

    # ── Per-calendar-day percentiles (Phase 2) ────────────────────────────────
    daily_p_kwh = _daily_percentiles(all_daily_kwh)

    # ── Monthly P values: sum of per-day P values within each month ───────────
    monthly_p_kwh = _monthly_from_daily(daily_p_kwh, target_year)

    # ── Annual true percentiles (from full-year totals) ───────────────────────
    annual_by_year = all_daily_kwh.resample("YE").sum()
    p50_annual_kwh = float(np.quantile(annual_by_year.values, 0.50))
    p90_annual_kwh = float(np.quantile(annual_by_year.values, 0.10))
    p95_annual_kwh = float(np.quantile(annual_by_year.values, 0.05))

    log.info(
        "_process_plant [%s]: annual P50=%.0f P90=%.0f P95=%.0f kWh | years=%d | daily_groups=%d",
        config.plant_name,
        p50_annual_kwh, p90_annual_kwh, p95_annual_kwh,
        len(annual_by_year),
        len(daily_p_kwh),
    )

    # Sanity: P50 > P90 > P95 (monotonicity guard)
    if not (p50_annual_kwh >= p90_annual_kwh >= p95_annual_kwh):
        log.warning(
            "_process_plant [%s]: P-value monotonicity violated — "
            "P50=%.0f P90=%.0f P95=%.0f. Check PVGIS data quality.",
            config.plant_name, p50_annual_kwh, p90_annual_kwh, p95_annual_kwh,
        )

    # ── Build TB records ──────────────────────────────────────────────────────
    daily_records   = _build_daily_records(daily_p_kwh, target_year, tz)
    monthly_records = _build_monthly_records(monthly_p_kwh, target_year, tz)

    annual_attrs = {
        ATTR_P50_ENERGY: round(p50_annual_kwh, 1),
        ATTR_P90_ENERGY: round(p90_annual_kwh, 1),
        ATTR_P95_ENERGY: round(p95_annual_kwh, 1),
        "pvalue_model_version": PVALUE_MODEL_VER,
        "pvalue_updated_at": datetime.now(timezone.utc).isoformat(),
        "pvalue_target_year": target_year,
    }

    # ── Write to TB ───────────────────────────────────────────────────────────
    await tb_client.post_telemetry("ASSET", plant_id, daily_records)
    await tb_client.post_telemetry("ASSET", plant_id, monthly_records)
    await tb_client.post_attributes("ASSET", plant_id, "SERVER_SCOPE", annual_attrs)


# ── Grid-cell deduplication ──────────────────────────────────────────────────

def _era5_cell(lat: float, lon: float) -> Tuple[float, float]:
    """Round lat/lon to nearest ERA5 grid node (0.25° resolution)."""
    return (
        round(lat / ERA5_GRID_DEG) * ERA5_GRID_DEG,
        round(lon / ERA5_GRID_DEG) * ERA5_GRID_DEG,
    )


# ── PVGIS fetch with retry ───────────────────────────────────────────────────

async def _fetch_pvgis_cell(cell_lat: float, cell_lon: float) -> pd.DataFrame:
    """Fetch PVGIS-ERA5 multi-year hourly data for a grid cell.

    Runs pvlib.iotools.get_pvgis_hourly() in a thread executor
    (it is synchronous / blocking HTTP).

    Retries up to PVGIS_RETRY_MAX times with exponential back-off on failure.
    """
    loop = asyncio.get_event_loop()
    last_exc: Optional[Exception] = None

    for attempt in range(settings.PVGIS_RETRY_MAX):
        try:
            # pvlib 0.11+ returns (data, inputs, metadata) — 3 values.
            # Earlier versions returned (data, months_optimal, inputs, metadata) — 4.
            result = await loop.run_in_executor(
                None,
                lambda: _pvgis_call(cell_lat, cell_lon),
            )
            df = result[0]   # always the DataFrame regardless of tuple length
            return df
        except Exception as exc:
            last_exc = exc
            if attempt < settings.PVGIS_RETRY_MAX - 1:
                wait_s = 2 ** (attempt + 1)   # 2 s, 4 s, 8 s
                log.warning(
                    "_fetch_pvgis_cell (%.2f, %.2f): attempt %d failed (%s) — retry in %ds",
                    cell_lat, cell_lon, attempt + 1, exc, wait_s,
                )
                await asyncio.sleep(wait_s)
            else:
                log.error(
                    "_fetch_pvgis_cell (%.2f, %.2f): all %d attempts failed",
                    cell_lat, cell_lon, settings.PVGIS_RETRY_MAX,
                )

    raise RuntimeError(
        f"PVGIS fetch failed for cell ({cell_lat}, {cell_lon}) "
        f"after {settings.PVGIS_RETRY_MAX} attempts"
    ) from last_exc


def _pvgis_call(lat: float, lon: float):
    """Synchronous PVGIS API call — run in executor, not in event loop."""
    import pvlib.iotools  # noqa: PLC0415 — deferred to avoid import cost at module load

    return pvlib.iotools.get_pvgis_hourly(
        latitude=lat,
        longitude=lon,
        start=settings.PVGIS_START_YEAR,
        end=settings.PVGIS_END_YEAR,
        raddatabase=settings.PVGIS_RADDATABASE,
        components=True,        # return GHI + DNI + DHI components
        outputformat="json",
        usehorizon=True,
        timeout=settings.PVGIS_REQUEST_TIMEOUT_S,
    )


# ── Weather DataFrame preparation ────────────────────────────────────────────

def _prepare_weather(df_pvgis: pd.DataFrame, config: PlantConfig) -> pd.DataFrame:
    """Rename/reconstruct PVGIS columns → pipeline canonical names.

    Handles two pvlib column name conventions:

    pvlib < 0.11 (legacy PVGIS API column names):
      'G(h)'   — Global Horizontal Irradiance [W/m²]
      'T2m'    — Air temperature at 2 m [°C]
      'WS10m'  — Wind speed at 10 m [m/s]

    pvlib 0.11+ (standardized pvlib column names, angle=0 horizontal fetch):
      'poa_direct'        — Beam irradiance on horizontal plane [W/m²]
      'poa_sky_diffuse'   — Sky diffuse on horizontal [W/m²]
      'poa_ground_diffuse'— Ground reflected on horizontal [W/m²]
      → GHI = sum of all three (on horizontal surface POA = GHI)
      'temp_air'          — Air temperature [°C]
      'wind_speed'        — Wind speed [m/s]  (name unchanged in pipeline)

    The fetch uses angle=0 (horizontal tilt) so no Perez transposition is
    applied by PVGIS; we receive GHI-equivalent components and reassemble.
    The pipeline then performs Perez transposition using the plant's actual
    tilt/azimuth during compute_ac_power().
    """
    cols = set(df_pvgis.columns)

    # ── GHI ──────────────────────────────────────────────────────────────────
    if "G(h)" in cols:
        # Legacy pvlib / raw PVGIS column name
        ghi = df_pvgis["G(h)"].clip(lower=0.0)
    elif all(c in cols for c in ("poa_direct", "poa_sky_diffuse", "poa_ground_diffuse")):
        # pvlib 0.11+: horizontal fetch → sum of POA components = GHI
        ghi = (
            df_pvgis["poa_direct"].clip(lower=0.0)
            + df_pvgis["poa_sky_diffuse"].clip(lower=0.0)
            + df_pvgis["poa_ground_diffuse"].clip(lower=0.0)
        )
    elif "poa_direct" in cols and "poa_sky_diffuse" in cols:
        # Partial match — best effort
        poa_cols = [c for c in ("poa_direct", "poa_sky_diffuse", "poa_ground_diffuse") if c in cols]
        ghi = df_pvgis[poa_cols].clip(lower=0.0).sum(axis=1)
    else:
        raise ValueError(
            f"PVGIS response has no usable GHI column. "
            f"Expected 'G(h)' or 'poa_direct'+'poa_sky_diffuse'+'poa_ground_diffuse'. "
            f"Available: {sorted(cols)}"
        )

    # ── Air temperature ───────────────────────────────────────────────────────
    if "temp_air" in cols:          # pvlib 0.11+
        air_temp = df_pvgis["temp_air"].fillna(config.defaults.air_temp_c)
    elif "T2m" in cols:             # legacy
        air_temp = df_pvgis["T2m"].fillna(config.defaults.air_temp_c)
    else:
        log.warning("_prepare_weather: no air_temp column in PVGIS data — using default %.1f°C",
                    config.defaults.air_temp_c)
        air_temp = pd.Series(config.defaults.air_temp_c, index=df_pvgis.index)

    # ── Wind speed ────────────────────────────────────────────────────────────
    if "wind_speed" in cols:        # pvlib 0.11+ (already correct name)
        wind_speed = df_pvgis["wind_speed"].fillna(config.defaults.wind_speed_ms)
    elif "WS10m" in cols:           # legacy
        wind_speed = df_pvgis["WS10m"].fillna(config.defaults.wind_speed_ms)
    else:
        log.warning("_prepare_weather: no wind_speed column in PVGIS data — using default %.1f m/s",
                    config.defaults.wind_speed_ms)
        wind_speed = pd.Series(config.defaults.wind_speed_ms, index=df_pvgis.index)

    # ── Assemble canonical DataFrame ──────────────────────────────────────────
    df = pd.DataFrame(
        {"ghi": ghi, "air_temp": air_temp, "wind_speed": wind_speed},
        index=df_pvgis.index,
    )

    # Ensure UTC-aware index then convert to plant local timezone
    if df.index.tzinfo is None:
        df.index = df.index.tz_localize("UTC")
    df.index = df.index.tz_convert(config.timezone)

    # Apply GHI sanity clip (matches live pipeline behaviour)
    df["ghi"] = df["ghi"].clip(lower=0.0, upper=config.station.sanity_max_ghi_wm2)

    return df


# ── Multi-year simulation ────────────────────────────────────────────────────

def _simulate_all_years(
    config: PlantConfig,
    df_weather: pd.DataFrame,
    tz: ZoneInfo,
) -> pd.Series:
    """Run compute_ac_power() for each calendar year in df_weather.

    Returns a Series indexed by local date, values = daily kWh.
    Uses local-timezone day boundaries (not UTC) so calendar days align correctly.

    Integration formula for HOURLY data:
      daily_kwh = Σ(kW_per_hour)  ← each row = 1 kW × 1 h = 1 kWh
      DO NOT divide by 60 — that formula applies to 1-minute data only.
    """
    all_daily: List[pd.Series] = []
    years = sorted(df_weather.index.year.unique())

    for year in years:
        df_year = df_weather[df_weather.index.year == year]
        if df_year.empty:
            log.debug("_simulate_all_years [%s]: year %d is empty — skipping",
                      config.plant_name, year)
            continue

        try:
            df_ac = compute_ac_power(config, df_year, data_source="pvgis_historical")
        except Exception as exc:
            log.warning("_simulate_all_years [%s]: year %d failed (%s) — skipping",
                        config.plant_name, year, exc)
            continue

        if df_ac.empty:
            continue

        # Daily integration — each row = 1 kWh (hourly cadence)
        # resample('D') respects tz-aware index → day = local calendar day
        daily_kwh = (
            df_ac["potential_power_kw"]
            .clip(lower=0.0)           # exclude any residual negatives (night sentinels)
            .resample("D")
            .sum()
        )
        all_daily.append(daily_kwh)

    if not all_daily:
        return pd.Series(dtype=float)

    return pd.concat(all_daily).sort_index()


# ── Percentile computation ───────────────────────────────────────────────────

def _daily_percentiles(
    all_daily_kwh: pd.Series,
) -> Dict[Tuple[int, int], Dict[float, float]]:
    """Phase 2 — per-calendar-day P50/P90/P95 from multi-year daily series.

    Groups daily values by (month, day_of_month) across all historical years.
    Each group has up to n=19 values (one per simulated year).
    Percentile of those n values = P-value for that calendar day.

    Returns
    -------
    dict
        {(month, day): {0.50: kwh, 0.10: kwh, 0.05: kwh}}
        Values = daily energy in kWh.
    """
    result: Dict[Tuple[int, int], Dict[float, float]] = {}

    # Build a DataFrame with month and day columns for grouping
    df = pd.DataFrame({"kwh": all_daily_kwh})
    df["month"] = df.index.month
    df["day"]   = df.index.day

    for (month, day), group in df.groupby(["month", "day"]):
        vals = group["kwh"].values
        if len(vals) == 0:
            log.warning("_daily_percentiles: no data for month=%d day=%d", month, day)
            result[(month, day)] = {0.50: 0.0, 0.10: 0.0, 0.05: 0.0}
            continue

        result[(month, day)] = {
            0.50: float(np.quantile(vals, 0.50)),
            0.10: float(np.quantile(vals, 0.10)),
            0.05: float(np.quantile(vals, 0.05)),
        }

        log.debug(
            "_daily_percentiles: %02d-%02d — P50=%.1f P90=%.1f P95=%.1f kWh (n=%d)",
            month, day,
            result[(month, day)][0.50],
            result[(month, day)][0.10],
            result[(month, day)][0.05],
            len(vals),
        )

    log.info(
        "_daily_percentiles: computed %d calendar-day groups across %d years",
        len(result),
        len(df["kwh"].resample("YE").sum()) if hasattr(all_daily_kwh, 'resample') else 0,
    )
    return result


def _monthly_from_daily(
    daily_p_kwh: Dict[Tuple[int, int], Dict[float, float]],
    target_year: int,
) -> Dict[int, Dict[float, float]]:
    """Derive monthly P values by summing per-day P values within each calendar month.

    Uses the actual days-in-month for target_year (handles leap years and Feb 29).
    Monotonicity is maintained because sum of P-values at same quantile is monotone.

    Returns
    -------
    dict
        {month (1–12): {0.50: kwh, 0.10: kwh, 0.05: kwh}}
    """
    result: Dict[int, Dict[float, float]] = {}
    for month in range(1, 13):
        _, days_in_month = calendar.monthrange(target_year, month)
        p50_sum = p90_sum = p95_sum = 0.0
        for day in range(1, days_in_month + 1):
            key = (month, day)
            if key not in daily_p_kwh:
                # Feb 29 in non-leap years: use Feb 28 value as best estimate
                fallback = (month, days_in_month)
                if fallback in daily_p_kwh:
                    pvals = daily_p_kwh[fallback]
                    log.debug("_monthly_from_daily: %02d-%02d missing — using %02d-%02d",
                              month, day, month, days_in_month)
                else:
                    log.warning("_monthly_from_daily: no data for %02d-%02d — using 0", month, day)
                    continue
            else:
                pvals = daily_p_kwh[key]
            p50_sum += pvals[0.50]
            p90_sum += pvals[0.10]
            p95_sum += pvals[0.05]

        result[month] = {0.50: p50_sum, 0.10: p90_sum, 0.05: p95_sum}
        log.debug(
            "_monthly_from_daily: month %02d — P50=%.0f P90=%.0f P95=%.0f kWh",
            month, p50_sum, p90_sum, p95_sum,
        )
    return result


# ── TB record builders ───────────────────────────────────────────────────────

def _build_daily_records(
    daily_p_kwh: Dict[Tuple[int, int], Dict[float, float]],
    target_year: int,
    tz: ZoneInfo,
) -> List[dict]:
    """Build 365/366 daily TB timeseries records for the target year.

    Phase 2: each day gets its own P value from per-calendar-day percentiles.
    ts = local midnight of that calendar day (ms epoch).
    Values written in MWh (÷ 1000) — widget expects MWh.
    """
    records: List[dict] = []
    for month in range(1, 13):
        _, days_in_month = calendar.monthrange(target_year, month)
        for day in range(1, days_in_month + 1):
            key = (month, day)
            if key not in daily_p_kwh:
                # Feb 29 in non-leap target year (target_year is always 365-day or 366-day)
                # Since target_year is real, monthrange is authoritative — no missing days.
                # But if ERA5 data never had Feb 29 (non-leap source years), use Feb 28.
                fallback = (month, days_in_month - 1) if day == 29 else (month, day)
                pvals = daily_p_kwh.get(fallback, {0.50: 0.0, 0.10: 0.0, 0.05: 0.0})
                log.warning("_build_daily_records: no per-day data for %02d-%02d — using fallback",
                            month, day)
            else:
                pvals = daily_p_kwh[key]

            ts_local = datetime(target_year, month, day, 0, 0, 0, tzinfo=tz)
            ts_ms = int(ts_local.timestamp() * 1000)
            records.append({
                "ts": ts_ms,
                "values": {
                    KEY_P50_DAILY: round(pvals[0.50] / 1000.0, 4),
                    KEY_P90_DAILY: round(pvals[0.10] / 1000.0, 4),
                    KEY_P95_DAILY: round(pvals[0.05] / 1000.0, 4),
                },
            })

    return records


def _build_monthly_records(
    monthly_p_kwh: Dict[int, Dict[float, float]],
    target_year: int,
    tz: ZoneInfo,
) -> List[dict]:
    """Build 12 monthly TB timeseries records for the target year.

    ts = local midnight of the 1st of each month.
    Values in MWh — widget expects MWh.
    """
    records: List[dict] = []
    for month in range(1, 13):
        ts_local = datetime(target_year, month, 1, 0, 0, 0, tzinfo=tz)
        ts_ms = int(ts_local.timestamp() * 1000)
        pvals = monthly_p_kwh[month]
        records.append({
            "ts": ts_ms,
            "values": {
                KEY_P50_MONTHLY: round(pvals[0.50] / 1000.0, 3),
                KEY_P90_MONTHLY: round(pvals[0.10] / 1000.0, 3),
                KEY_P95_MONTHLY: round(pvals[0.05] / 1000.0, 3),
            },
        })
    return records
