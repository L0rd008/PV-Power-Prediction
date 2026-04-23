"""
3-tier irradiance / weather data strategy (H-B6).

Tier 1 — TB weather station (freshest, highest accuracy)
Tier 2 — Solcast estimated_actuals API (satellite, ~15-min delay)
Tier 3 — pvlib Ineichen clear-sky model (always available, zero accuracy)

The caller receives a single DataFrame plus a string indicating which tier was used.
"""
from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Optional, Tuple

import httpx
import pandas as pd
import pvlib
from pvlib.location import Location

from app.physics.config import PlantConfig, StationConfig

log = logging.getLogger(__name__)

DataResult = Tuple[pd.DataFrame, str]   # (df_weather, source_label)


async def select_irradiance(
    config: PlantConfig,
    start: datetime,
    end: datetime,
    tb_client,                     # ThingsBoardClient (injected to avoid circular import)
    solcast_api_key: Optional[str] = None,
) -> DataResult:
    """Try each tier in order, return first that produces valid data."""

    # ── Tier 1: TB weather station ─────────────────────────────────────
    if config.weather_station_id:
        try:
            df = await _fetch_tb_station(config, start, end, tb_client)
            if _is_valid(df):
                log.debug("%s: using Tier-1 TB station data", config.plant_name)
                return df, "tb_station"
            else:
                log.info("%s: TB station data invalid/stale, falling to Tier-2", config.plant_name)
        except Exception as exc:
            log.warning("%s: TB station fetch failed (%s), falling to Tier-2", config.plant_name, exc)

    # ── Tier 2: Solcast ────────────────────────────────────────────────
    if solcast_api_key and config.solcast_resource_id:
        try:
            df = await _fetch_solcast(config, start, end, solcast_api_key)
            if _is_valid(df):
                log.debug("%s: using Tier-2 Solcast data", config.plant_name)
                return df, "solcast"
            else:
                log.info("%s: Solcast data empty, falling to Tier-3", config.plant_name)
        except Exception as exc:
            log.warning("%s: Solcast fetch failed (%s), falling to Tier-3", config.plant_name, exc)

    # ── Tier 3: Clear-sky ──────────────────────────────────────────────
    log.debug("%s: using Tier-3 clear-sky", config.plant_name)
    df = _clearsky(config, start, end)
    return df, "clearsky"


# ── Tier 1 implementation ──────────────────────────────────────────────────

async def _fetch_tb_station(
    config: PlantConfig,
    start: datetime,
    end: datetime,
    tb_client,
) -> pd.DataFrame:
    sc = config.station
    keys = [sc.ghi_key, sc.air_temp_key]
    if sc.poa_key:
        keys.append(sc.poa_key)
    if sc.wind_speed_key:
        keys.append(sc.wind_speed_key)

    raw = await tb_client.get_timeseries(
        entity_type="DEVICE",
        entity_id=config.weather_station_id,
        keys=keys,
        start=start,
        end=end,
    )

    if not raw:
        return pd.DataFrame()

    frames = {}
    for key, records in raw.items():
        if records:
            s = pd.Series(
                {pd.Timestamp(r["ts"], unit="ms", tz="UTC"): float(r["value"]) for r in records}
            )
            frames[key] = s

    if not frames:
        return pd.DataFrame()

    df = pd.DataFrame(frames)
    df.index = df.index.tz_convert(config.timezone)
    df = df.sort_index()

    # Rename to canonical column names
    rename = {sc.ghi_key: "ghi", sc.air_temp_key: "air_temp"}
    if sc.poa_key and sc.poa_key in df.columns:
        rename[sc.poa_key] = "poa"
    if sc.wind_speed_key and sc.wind_speed_key in df.columns:
        rename[sc.wind_speed_key] = "wind_speed"
    df = df.rename(columns=rename)

    # Sanity clipping
    if "ghi" in df.columns:
        df["ghi"] = df["ghi"].clip(lower=0, upper=sc.sanity_max_ghi_wm2)
    if "poa" in df.columns:
        df["poa"] = df["poa"].clip(lower=0, upper=sc.sanity_max_poa_wm2)

    # Freshness check: reject if last record is too old
    if not df.empty:
        now_local = datetime.now(timezone.utc).astimezone(
            __import__("zoneinfo", fromlist=["ZoneInfo"]).ZoneInfo(config.timezone)
        )
        last_ts = df.index[-1]
        age_minutes = (now_local - last_ts).total_seconds() / 60
        if age_minutes > sc.freshness_minutes:
            log.info(
                "%s: station data is %.1f min old (limit %d min), treating as stale",
                config.plant_name, age_minutes, sc.freshness_minutes,
            )
            return pd.DataFrame()

    return df


# ── Tier 2 implementation ──────────────────────────────────────────────────

async def _fetch_solcast(
    config: PlantConfig,
    start: datetime,
    end: datetime,
    api_key: str,
) -> pd.DataFrame:
    url = (
        f"https://api.solcast.com.au/data/historic/radiation_and_weather"
        f"?resource_id={config.solcast_resource_id}"
        f"&start={start.strftime('%Y-%m-%dT%H:%M:%SZ')}"
        f"&end={end.strftime('%Y-%m-%dT%H:%M:%SZ')}"
        f"&format=json&output_parameters=ghi,air_temp_2m,wind_speed_10m"
    )
    headers = {"Authorization": f"Bearer {api_key}"}
    async with httpx.AsyncClient(timeout=30) as client:
        resp = await client.get(url, headers=headers)
        resp.raise_for_status()

    data = resp.json()
    records = data.get("estimated_actuals", [])
    if not records:
        return pd.DataFrame()

    rows = []
    for r in records:
        rows.append({
            "ts": pd.Timestamp(r["period_end"]).tz_convert("UTC"),
            "ghi": float(r.get("ghi", 0)),
            "air_temp": float(r.get("air_temp_2m", config.defaults.air_temp_c)),
            "wind_speed": float(r.get("wind_speed_10m", config.defaults.wind_speed_ms)),
        })

    df = pd.DataFrame(rows).set_index("ts").sort_index()
    df.index = df.index.tz_convert(config.timezone)
    return df


# ── Tier 3 implementation ──────────────────────────────────────────────────

def _clearsky(config: PlantConfig, start: datetime, end: datetime) -> pd.DataFrame:
    loc = Location(
        latitude=config.latitude,
        longitude=config.longitude,
        altitude=config.altitude_m,
        tz=config.timezone,
    )
    times = pd.date_range(start=start, end=end, freq="1min", tz=config.timezone)
    cs = loc.get_clearsky(times, model="ineichen")
    df = pd.DataFrame({
        "ghi": cs["ghi"],
        "air_temp": config.defaults.air_temp_c,
        "wind_speed": config.defaults.wind_speed_ms,
    }, index=times)
    return df


# ── Validation helper ──────────────────────────────────────────────────────

def _is_valid(df: pd.DataFrame) -> bool:
    if df is None or df.empty:
        return False
    if "ghi" not in df.columns and "poa" not in df.columns:
        return False
    return True
