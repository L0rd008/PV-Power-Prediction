"""
3-tier irradiance / weather data strategy (H-B6).

Tier 1 — TB weather station (freshest, highest accuracy)
Tier 2 — Solcast estimated_actuals API (satellite, ~15-min delay)  [Gap 7: cached]
Tier 3 — pvlib Ineichen clear-sky model (always available, zero accuracy)

Gap 7 (Phase C): Solcast responses are cached in a module-level dict keyed by
(resource_id, 30-min bucket).  Stale results are served for up to 60 min on
4xx/5xx API failures.  Cache is lost on restart (acceptable: ~30 s penalty).

The caller receives a single DataFrame plus a string indicating which tier was used.
"""
from __future__ import annotations

import asyncio
import logging
import time
from datetime import datetime, timezone
from typing import Dict, Optional, Tuple
from zoneinfo import ZoneInfo

import httpx
import pandas as pd
import pvlib
from pvlib.location import Location

from app.physics.config import PlantConfig, StationConfig

log = logging.getLogger(__name__)

DataResult = Tuple[pd.DataFrame, str]   # (df_weather, source_label)

# ── Solcast in-process cache (Gap 7) ────────────────────────────────────────
# Key:   (resource_id, bucket_epoch)  where bucket_epoch = int(start_ts) // 1800
# Value: (DataFrame, stored_at_epoch_float)
_solcast_cache: Dict[Tuple[str, int], Tuple[pd.DataFrame, float]] = {}
_solcast_lock: asyncio.Lock = None   # lazily initialised

_BUCKET_SEC   = 1800    # 30-min bucket window
_STALE_OK_SEC = 3600    # serve stale for up to 60 min on failure

# Prometheus-style counters (read by /metrics in Phase E)
_solcast_hits_total    = 0
_solcast_misses_total  = 0


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
            ZoneInfo(config.timezone)
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


# ── Tier 2 implementation with cache (Gap 7) ───────────────────────────────

async def _fetch_solcast(
    config: PlantConfig,
    start: datetime,
    end: datetime,
    api_key: str,
) -> pd.DataFrame:
    """Fetch Solcast estimated_actuals with 30-min bucket caching.

    On API failure, returns stale data if < _STALE_OK_SEC old.
    Edge case E9: empty estimated_actuals cached for 30 min to avoid hammering.
    """
    global _solcast_lock, _solcast_hits_total, _solcast_misses_total

    if _solcast_lock is None:
        _solcast_lock = asyncio.Lock()

    resource_id = config.solcast_resource_id
    # Use the end-of-range bucket to handle window-straddling correctly
    bucket = int(end.timestamp()) // _BUCKET_SEC

    cache_key = (resource_id, bucket)

    async with _solcast_lock:
        cached = _solcast_cache.get(cache_key)
        now_ts = time.time()
        if cached:
            df_cached, stored_at = cached
            age = now_ts - stored_at
            if age < _BUCKET_SEC:
                _solcast_hits_total += 1
                log.debug("_fetch_solcast: cache hit for %s (age=%.0fs)", resource_id, age)
                return df_cached

    # Cache miss or expired — fetch from API
    _solcast_misses_total += 1
    url = (
        f"https://api.solcast.com.au/data/historic/radiation_and_weather"
        f"?resource_id={resource_id}"
        f"&start={start.strftime('%Y-%m-%dT%H:%M:%SZ')}"
        f"&end={end.strftime('%Y-%m-%dT%H:%M:%SZ')}"
        f"&format=json&output_parameters=ghi,air_temp_2m,wind_speed_10m"
    )
    headers = {"Authorization": f"Bearer {api_key}"}

    try:
        async with httpx.AsyncClient(timeout=30) as client:
            resp = await client.get(url, headers=headers)
            resp.raise_for_status()
        data = resp.json()
        records = data.get("estimated_actuals", [])

        if not records:
            # Edge case E9: cache empty result to avoid hammering
            empty_df = pd.DataFrame()
            async with _solcast_lock:
                _solcast_cache[cache_key] = (empty_df, time.time())
            return empty_df

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

        async with _solcast_lock:
            _solcast_cache[cache_key] = (df, time.time())

        return df

    except Exception as exc:
        # API failure — serve stale if available and not too old (H7-D)
        async with _solcast_lock:
            cached = _solcast_cache.get(cache_key)
        if cached:
            df_stale, stored_at = cached
            age = time.time() - stored_at
            if age < _STALE_OK_SEC:
                log.warning(
                    "_fetch_solcast: API failed (%s), serving stale data (age=%.0fs)",
                    exc, age,
                )
                return df_stale
        raise


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
    """Return True if df contains usable irradiance data.

    Rejects:
      - empty or missing ghi/poa
      - stuck pyranometer (std < 1 W/m² over a non-zero mean) — Edge case E2
    """
    if df is None or df.empty:
        return False
    if "ghi" not in df.columns and "poa" not in df.columns:
        return False
    # Edge case E2: detect stuck pyranometer sensor
    if "poa" in df.columns and len(df) > 2:
        poa_std = float(df["poa"].std(skipna=True))
        poa_mean = float(df["poa"].mean(skipna=True))
        if poa_std < 1.0 and poa_mean > 10.0:
            # Non-zero stuck reading — sensor fault, not just night-time
            log.warning(
                "_is_valid: POA pyranometer appears stuck "
                "(std=%.3f W/m², mean=%.1f W/m²) — treating station data as invalid",
                poa_std, poa_mean,
            )
            return False
    return True
