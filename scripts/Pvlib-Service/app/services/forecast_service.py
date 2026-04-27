"""
ForecastService — orchestrates physics computation for one plant or a fleet.

Responsibilities:
  1. Read PlantConfig from ThingsBoard asset attributes
  2. Discover related devices (weather station, P341 meter) — H1-F:
       a. Explicit IDs from pvlib_config / flat attrs (primary)
       b. Contains-relation lookup (legacy fallback)
       c. Tenant device search by plant-name prefix (zero-config bootstrap)
  3. Fetch weather data (3-tier: TB station → Solcast → clear-sky)
  4. Run compute_ac_power() physics pipeline
  5. Dual-write telemetry:
       potential_power         (kW) — primary widget key
       active_power_pvlib_kw   (kW) — ops/diagnostic alias
       total_generation_expected_kwh — daily energy expected (written by daily_job.py)
       pvlib_daily_energy_kwh        — ops alias
       pvlib_data_source, pvlib_model_version
  6. Dedup-aware full-depth roll-up (Gap 5/6):
       ancestor_map[plant_id] = Set of all isPlantAgg ancestors (not just direct parent)
       in-memory DataFrames summed — no TB re-read
  7. -1 sentinel writes on any error (Gap 2):
       _build_sentinel_records() / _write_sentinels() cover every 1-min boundary
"""
from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional, Set

import pandas as pd

from app.metrics import data_source_count, plant_failures_total
from app.physics.config import PlantConfig
from app.physics.data_sources import select_irradiance
from app.physics.pipeline import compute_ac_power
from app.services.job_manager import JobManager, JobStatus, job_manager as _default_job_manager

log = logging.getLogger(__name__)

# ── Telemetry key constants (P4-D dual-write) ──────────────────────────────
KEY_POTENTIAL_POWER          = "potential_power"           # primary widget key (kW)
KEY_PVLIB_POWER              = "active_power_pvlib_kw"     # ops alias (kW)
KEY_DAILY_ENERGY_EXPECTED    = "total_generation_expected_kwh"  # primary daily key (kWh)
KEY_PVLIB_DAILY_ENERGY       = "pvlib_daily_energy_kwh"    # ops alias (kWh)
KEY_DATA_SOURCE              = "pvlib_data_source"
KEY_MODEL_VERSION            = "pvlib_model_version"
KEY_UNIT                     = "ops_expected_unit"

MODEL_VERSION = "pvlib-h-a3-v1"


# ── Result dataclass ────────────────────────────────────────────────────────

@dataclass
class PlantCycleResult:
    """Result from processing one plant in one scheduler cycle."""

    asset_id: str
    status: str                          # "ok" | "no_data" | "config_error" |
                                         # "no_location" | "data_error" |
                                         # "physics_error" | "write_error" | "timeout"
    source: Optional[str] = None
    df: Optional[pd.DataFrame] = None   # None when no real data produced
    sentinels: Optional[List[dict]] = None  # sentinel records written (for debugging)
    error: Optional[str] = None
    records: int = 0
    total_kwh: float = 0.0
    peak_kw: float = 0.0

    def to_dict(self) -> dict:
        """JSON-serialisable summary (excludes DataFrame)."""
        return {
            "asset_id": self.asset_id,
            "status": self.status,
            "source": self.source,
            "records": self.records,
            "total_kwh": self.total_kwh,
            "peak_kw": self.peak_kw,
            "error": self.error,
        }


# ── ForecastService ─────────────────────────────────────────────────────────

class ForecastService:
    """Stateless orchestrator; instantiated per-request or per-cycle.

    The station resolution cache (_station_resolution_cache) is a class variable
    so it persists across instances for the process lifetime.  Device UUIDs in TB
    do not change, so this is safe.
    """

    # Process-lifetime station resolution cache
    # key: plant asset_id
    # value: {"weather_station_id": str|None, "p341_device_id": str|None,
    #          "mode": "explicit|contains|naming|none", "miss_count": int}
    _station_resolution_cache: Dict[str, Dict[str, Any]] = {}

    def __init__(self, tb_client, solcast_api_key: Optional[str] = None):
        self._tb = tb_client
        self._solcast_key = solcast_api_key

    # ── Public: single plant ───────────────────────────────────────────────

    async def process_single_asset(
        self,
        asset_id: str,
        start: datetime,
        end: datetime,
    ) -> PlantCycleResult:
        """Compute and write pvlib forecast for one plant.

        On any failure, -1 sentinel records are written to TB for every 1-minute
        boundary in [start, end] before returning.
        """
        log.info("process_single_asset: %s  [%s → %s]",
                 asset_id, start.isoformat(), end.isoformat())

        # ── Config load ────────────────────────────────────────────────────
        try:
            config = await self._load_config(asset_id)
        except Exception as exc:
            log.error("process_single_asset: config load failed for %s: %s", asset_id, exc)
            sentinels = self._build_sentinel_records(start, end, "no_config")
            await self._write_sentinels(asset_id, sentinels)
            return PlantCycleResult(
                asset_id=asset_id, status="config_error",
                error=str(exc), sentinels=sentinels,
            )

        # ── Location guard (Edge case E3) ──────────────────────────────────
        if config.latitude == 0.0 and config.longitude == 0.0:
            msg = "latitude/longitude both 0.0 (unset)"
            log.error("process_single_asset: %s for %s", msg, asset_id)
            sentinels = self._build_sentinel_records(start, end, "no_location")
            await self._write_sentinels(asset_id, sentinels)
            return PlantCycleResult(
                asset_id=asset_id, status="no_location",
                error=msg, sentinels=sentinels,
            )

        # ── Weather data fetch ─────────────────────────────────────────────
        try:
            df_weather, source = await select_irradiance(
                config, start, end, self._tb, self._solcast_key
            )
        except Exception as exc:
            log.error("process_single_asset: irradiance fetch failed for %s: %s", asset_id, exc)
            sentinels = self._build_sentinel_records(start, end, "no_data")
            await self._write_sentinels(asset_id, sentinels)
            return PlantCycleResult(
                asset_id=asset_id, status="data_error",
                error=str(exc), sentinels=sentinels,
            )

        if df_weather.empty:
            log.warning("process_single_asset: no weather data for %s", asset_id)
            sentinels = self._build_sentinel_records(start, end, "no_data")
            await self._write_sentinels(asset_id, sentinels)
            return PlantCycleResult(
                asset_id=asset_id, status="no_data",
                source=source, sentinels=sentinels,
            )

        # ── Physics ────────────────────────────────────────────────────────
        try:
            df_result = compute_ac_power(config, df_weather, data_source=source)
        except Exception as exc:
            log.error("process_single_asset: physics failed for %s: %s", asset_id, exc)
            sentinels = self._build_sentinel_records(start, end, "physics_error")
            await self._write_sentinels(asset_id, sentinels)
            return PlantCycleResult(
                asset_id=asset_id, status="physics_error",
                error=str(exc), sentinels=sentinels,
            )

        # ── Telemetry write ────────────────────────────────────────────────
        try:
            records = _build_ac_telemetry(df_result)
            await self._tb.post_telemetry("ASSET", asset_id, records)
            log.info("process_single_asset: wrote %d records for %s (source=%s)",
                     len(records), asset_id, source)
        except Exception as exc:
            log.error("process_single_asset: telemetry write failed for %s: %s", asset_id, exc)
            sentinels = self._build_sentinel_records(start, end, "write_error")
            await self._write_sentinels(asset_id, sentinels)
            return PlantCycleResult(
                asset_id=asset_id, status="write_error",
                error=str(exc), sentinels=sentinels,
            )

        # ── Partial-coverage gap fill (Edge case: station has missing minutes) ──
        await self._fill_coverage_gaps(asset_id, df_result, start, end)

        total_kwh = df_result["potential_power_kw"].sum() / 60.0  # kW × 1/60 hr
        return PlantCycleResult(
            asset_id=asset_id,
            status="ok",
            source=source,
            df=df_result,
            records=len(records),
            total_kwh=round(total_kwh, 3),
            peak_kw=round(float(df_result["potential_power_kw"].max()), 2),
        )

    # ── Public: fleet (scheduler cycle) ───────────────────────────────────

    async def run_fleet_cycle(
        self,
        root_asset_ids: List[str],
        start: datetime,
        end: datetime,
        max_concurrent: int = 5,
        job_mgr: Optional[JobManager] = None,
    ) -> Dict[str, Any]:
        """Discover enabled plants, compute all, roll up to ancestors. Returns cycle summary."""
        jm = job_mgr or _default_job_manager
        job = await jm.create()
        await jm.update(
            job.job_id,
            status=JobStatus.RUNNING,
            started_at=datetime.now(timezone.utc),
        )

        try:
            plants, ancestor_map = await self._tb.discover_plants(root_asset_ids)
            await jm.update(job.job_id, plants_total=len(plants))

            if not plants:
                log.warning("run_fleet_cycle: no pvlib-enabled plants found")
                await jm.update(job.job_id, status=JobStatus.COMPLETED,
                                finished_at=datetime.now(timezone.utc))
                return {"job_id": job.job_id, "plants": 0}

            sem = asyncio.Semaphore(max_concurrent)
            plant_results: Dict[str, PlantCycleResult] = {}

            async def bounded_compute(plant):
                async with sem:
                    try:
                        res = await self.process_single_asset(plant.id, start, end)
                    except Exception as exc:
                        # Unexpected exception not caught inside process_single_asset
                        log.error("bounded_compute: unexpected error for %s: %s", plant.id, exc)
                        sentinels = self._build_sentinel_records(start, end, "timeout")
                        await self._write_sentinels(plant.id, sentinels)
                        res = PlantCycleResult(
                            asset_id=plant.id, status="timeout",
                            error=str(exc), sentinels=sentinels,
                        )
                    plant_results[plant.id] = res
                    if res.status == "ok":
                        await jm.update(job.job_id,
                                        plants_completed=job.plants_completed + 1)
                    else:
                        await jm.update(
                            job.job_id,
                            plants_failed=job.plants_failed + 1,
                            errors=job.errors + [f"{plant.id}: {res.error or res.status}"],
                        )

            await asyncio.gather(*(bounded_compute(p) for p in plants),
                                 return_exceptions=True)

            # ── Gap 15: update Prometheus-style counters ──────────────────
            data_source_count.clear()   # gauge — reset each cycle
            for asset_id, res in plant_results.items():
                if res.status == "ok" and res.source:
                    data_source_count[res.source] = (
                        data_source_count.get(res.source, 0) + 1
                    )
                elif res.status != "ok":
                    # monotonic per-(plant, reason) failure counter
                    key = (asset_id, res.status)
                    plant_failures_total[key] = plant_failures_total.get(key, 0) + 1
                    # also bucket into data_source_count so /metrics shows error share
                    err_label = f"error:{res.status}"
                    data_source_count[err_label] = (
                        data_source_count.get(err_label, 0) + 1
                    )

            # ── Full-depth ancestor roll-up (Gaps 5+6) ─────────────────────
            await self._rollup_parents(plants, plant_results, ancestor_map, start, end)

            # ── Daily energy write at 01:00 UTC (replaced by daily_job.py in Phase B) ──
            now_utc = datetime.now(timezone.utc)
            if now_utc.minute == 0 and now_utc.hour == 1:
                await self._write_daily_energies(plant_results, now_utc, ancestor_map, plants)

            failed = sum(1 for r in plant_results.values() if r.status != "ok")
            final_status = JobStatus.COMPLETED if failed == 0 else JobStatus.PARTIAL
            summary = {
                "job_id": job.job_id,
                "plants_processed": len(plant_results),
                "plants_ok": len(plant_results) - failed,
                "plants_failed": failed,
                "results": {k: v.to_dict() for k, v in plant_results.items()},
            }
            await jm.update(job.job_id,
                            status=final_status,
                            finished_at=datetime.now(timezone.utc),
                            result_summary=summary)
            return summary

        except Exception as exc:
            log.exception("run_fleet_cycle: unhandled error: %s", exc)
            await jm.update(job.job_id,
                            status=JobStatus.FAILED,
                            finished_at=datetime.now(timezone.utc),
                            errors=job.errors + [str(exc)])
            raise

    # ── Full-depth ancestor roll-up (Gaps 5 + 6) ──────────────────────────

    async def _rollup_parents(
        self,
        plants,
        plant_results: Dict[str, PlantCycleResult],
        ancestor_map: Dict[str, Set[str]],
        start: datetime,
        end: datetime,
    ) -> None:
        """Sum potential_power to every isPlantAgg ancestor using in-memory DataFrames.

        Each unique plant is counted exactly once per ancestor.
        Plants with df=None (failures) are excluded from the sum.
        If all children of an ancestor failed, -1 sentinels are emitted for the ancestor.
        """
        # Build inverse: ancestor_id → set of plant_ids under it
        ancestor_children: Dict[str, Set[str]] = {}
        for plant_id, ancestors in ancestor_map.items():
            for anc_id in ancestors:
                ancestor_children.setdefault(anc_id, set()).add(plant_id)

        if not ancestor_children:
            return

        ok_results = {
            r.asset_id: r
            for r in plant_results.values()
            if r.status == "ok" and r.df is not None
        }

        for ancestor_id, child_ids in ancestor_children.items():
            try:
                series = [
                    ok_results[cid].df["potential_power_kw"]
                    for cid in child_ids
                    if cid in ok_results
                ]

                if not series:
                    # All children failed — emit -1 sentinels at the ancestor too (Edge E5)
                    log.debug("rollup: all children failed for ancestor %s, writing sentinels",
                              ancestor_id)
                    sentinels = self._build_sentinel_records(start, end, "all_children_failed")
                    await self._write_sentinels(ancestor_id, sentinels)
                    continue

                # Sum across children; skipna so a plant missing a minute doesn't zero the sum
                total = pd.concat(series, axis=1).sum(axis=1, skipna=True)

                rollup_records = [
                    {
                        "ts": int(ts.timestamp() * 1000),
                        "values": {
                            KEY_POTENTIAL_POWER: round(float(v), 3),
                            KEY_PVLIB_POWER: round(float(v), 3),
                            KEY_DATA_SOURCE: "rollup",
                            KEY_MODEL_VERSION: MODEL_VERSION,
                            KEY_UNIT: "kW",
                        },
                    }
                    for ts, v in total.items()
                    if float(v) >= 0
                ]
                await self._tb.post_telemetry("ASSET", ancestor_id, rollup_records)
                log.debug(
                    "rollup: wrote %d records to ancestor %s (sum of %d/%d plants)",
                    len(rollup_records), ancestor_id, len(series), len(child_ids),
                )

            except Exception as exc:
                log.warning("rollup: failed for ancestor %s: %s", ancestor_id, exc)

    async def _write_daily_energies(
        self,
        plant_results: Dict[str, PlantCycleResult],
        ts: datetime,
        ancestor_map: Dict[str, Set[str]],
        plants: list,
    ) -> None:
        """Write daily expected energy totals to each plant and its ancestors.

        NOTE: Phase B replaces this with daily_job.py (00:05 local-tz cron).
        This remains as a fallback for the 01:00 UTC trigger during Phase A.
        """
        ts_ms = int(ts.replace(hour=0, minute=0, second=0, microsecond=0).timestamp() * 1000)

        # Per-plant
        for plant_id, result in plant_results.items():
            if result.status != "ok":
                continue
            kwh = result.total_kwh
            try:
                await self._tb.post_telemetry("ASSET", plant_id, [{
                    "ts": ts_ms,
                    "values": {
                        KEY_DAILY_ENERGY_EXPECTED: round(kwh, 3),
                        KEY_PVLIB_DAILY_ENERGY: round(kwh, 3),
                    },
                }])
            except Exception as exc:
                log.warning("daily energy write failed for %s: %s", plant_id, exc)

        # Ancestor roll-up of daily energy
        ancestor_children: Dict[str, Set[str]] = {}
        for plant_id, ancestors in ancestor_map.items():
            for anc_id in ancestors:
                ancestor_children.setdefault(anc_id, set()).add(plant_id)

        for ancestor_id, child_ids in ancestor_children.items():
            total_kwh = sum(
                plant_results[cid].total_kwh
                for cid in child_ids
                if cid in plant_results and plant_results[cid].status == "ok"
            )
            try:
                await self._tb.post_telemetry("ASSET", ancestor_id, [{
                    "ts": ts_ms,
                    "values": {
                        KEY_DAILY_ENERGY_EXPECTED: round(total_kwh, 3),
                        KEY_PVLIB_DAILY_ENERGY: round(total_kwh, 3),
                    },
                }])
            except Exception as exc:
                log.warning("daily energy ancestor write failed for %s: %s", ancestor_id, exc)

    # ── Config loader (H1-F) ───────────────────────────────────────────────

    async def _load_config(self, asset_id: str) -> PlantConfig:
        """Load PlantConfig from TB attributes, then resolve related devices.

        Device resolution order (H1-F):
          1. Explicit IDs in pvlib_config blob or flat attrs (already parsed by config.py)
          2. Contains-relation lookup (legacy TB hierarchy path)
          3. Tenant device search by plant-name prefix (zero-config bootstrap)
        Logs station_resolution_mode for ops visibility.
        """
        attrs = await self._tb.get_asset_attributes(asset_id)
        if not attrs:
            raise ValueError(f"No attributes found for asset {asset_id}")

        config = PlantConfig.from_tb_attributes(asset_id, attrs)

        # ── Step 1: Explicit IDs (already in config from parsers) ──────────
        if config.weather_station_id:
            log.debug("_load_config [%s]: weather station from explicit attr: %s",
                      asset_id, config.weather_station_id)
            self._update_station_cache(asset_id, config, "explicit")
            return config

        # ── Check process-lifetime cache ───────────────────────────────────
        cached = ForecastService._station_resolution_cache.get(asset_id)
        if cached:
            config.weather_station_id = cached.get("weather_station_id")
            config.p341_device_id = cached.get("p341_device_id")
            log.debug("_load_config [%s]: station from process cache (mode=%s)",
                      asset_id, cached.get("mode"))
            return config

        # ── Step 2: Contains-relation lookup ───────────────────────────────
        station_found_via = None
        try:
            children = await self._tb.get_child_relations(asset_id)
            for rel in children:
                child_id = rel.get("to", {}).get("id") or rel.get("toId", "")
                child_type = (
                    rel.get("to", {}).get("entityType") or rel.get("toEntityType", "")
                )
                if not child_id or child_type != "DEVICE":
                    continue
                child_info = await self._tb.get_entity_info(child_id, "DEVICE")
                child_name = (child_info or {}).get("name", "")
                upper = child_name.upper()
                if ("WSTN" in upper or "WEATHER" in upper or "STATION" in upper):
                    if not config.weather_station_id:
                        config.weather_station_id = child_id
                        station_found_via = "contains"
                        log.debug("_load_config [%s]: WSTN via Contains relation: %s (%s)",
                                  asset_id, child_id, child_name)
                elif "P341" in upper or "METER" in upper:
                    if not config.p341_device_id:
                        config.p341_device_id = child_id
                        log.debug("_load_config [%s]: P341 via Contains relation: %s (%s)",
                                  asset_id, child_id, child_name)
        except Exception as exc:
            log.debug("_load_config [%s]: Contains-relation lookup failed: %s", asset_id, exc)

        # ── Step 3: Name-prefix search (H1-B) ──────────────────────────────
        if not config.weather_station_id:
            plant_name = config.plant_name
            if plant_name not in ("Unknown", "Unknown Plant"):
                await self._resolve_station_by_name_prefix(asset_id, config)
                if config.weather_station_id:
                    station_found_via = "naming"

        # ── Log resolution outcome ─────────────────────────────────────────
        if config.weather_station_id:
            mode = station_found_via or "explicit"
            if mode != "explicit":
                log.warning(
                    "_load_config [%s]: weather_station_id resolved via '%s' fallback. "
                    "Add 'weather_station_id' to pvlib_config to remove this warning.",
                    asset_id, mode,
                )
            self._update_station_cache(asset_id, config, mode)
        else:
            log.warning("_load_config [%s]: no weather station found (station_resolution_mode=none)",
                        asset_id)
            self._update_station_cache(asset_id, config, "none")

        return config

    def _update_station_cache(
        self, asset_id: str, config: PlantConfig, mode: str
    ) -> None:
        ForecastService._station_resolution_cache[asset_id] = {
            "weather_station_id": config.weather_station_id,
            "p341_device_id": config.p341_device_id,
            "mode": mode,
            "miss_count": 0,
        }

    async def _resolve_station_by_name_prefix(
        self, asset_id: str, config: PlantConfig
    ) -> None:
        """H1-B: find WSTN/P341 devices by plant-name prefix search.

        Example: plant_name='KSP_Plant' → prefix='KSP', searches for KSP_WSTN*, KSP_P341*.
        """
        prefix = config.plant_name.split("_")[0] if "_" in config.plant_name else config.plant_name
        if not prefix:
            return

        try:
            devices = await self._tb.search_devices_by_name_prefix(prefix)
        except Exception as exc:
            log.debug("_resolve_station_by_name_prefix [%s]: search failed: %s", asset_id, exc)
            return

        if not devices:
            return

        prefix_upper = prefix.upper()
        wstn_candidates: List[dict] = []
        p341_candidates: List[dict] = []

        for dev in devices:
            raw_id = dev.get("id", {})
            dev_id = raw_id.get("id") if isinstance(raw_id, dict) else str(raw_id)
            dev_name = (dev.get("name") or "").upper()
            if not dev_id:
                continue
            # Match pattern: PREFIX_WSTN* or PREFIX_P341*
            if dev_name.startswith(f"{prefix_upper}_WSTN") or (
                "WSTN" in dev_name and dev_name.startswith(prefix_upper)
            ):
                wstn_candidates.append(dev)
            elif dev_name.startswith(f"{prefix_upper}_P341") or (
                "P341" in dev_name and dev_name.startswith(prefix_upper)
            ):
                p341_candidates.append(dev)

        # Multi-WSTN resolution (Edge case: SSK has WSTN, WSTN_T, WSTN_R)
        best_wstn: Optional[dict] = None
        if len(wstn_candidates) == 1:
            best_wstn = wstn_candidates[0]
        elif len(wstn_candidates) > 1:
            best_wstn = await self._pick_best_wstn(wstn_candidates, config)

        if best_wstn:
            raw_id = best_wstn.get("id", {})
            wstn_id = raw_id.get("id") if isinstance(raw_id, dict) else str(raw_id)
            config.weather_station_id = wstn_id
            log.warning(
                "_resolve_station_by_name_prefix [%s]: matched WSTN '%s' (%s) via "
                "name-prefix fallback. Set 'weather_station_id' in pvlib_config to silence.",
                asset_id, best_wstn.get("name"), wstn_id,
            )

        if p341_candidates and not config.p341_device_id:
            dev = p341_candidates[0]
            raw_id = dev.get("id", {})
            p341_id = raw_id.get("id") if isinstance(raw_id, dict) else str(raw_id)
            config.p341_device_id = p341_id

    async def _pick_best_wstn(
        self, candidates: List[dict], config: PlantConfig
    ) -> Optional[dict]:
        """Score WSTN candidates by how many configured telemetry keys they have data for."""
        keys_to_check = [
            k for k in [
                config.station.ghi_key,
                config.station.poa_key,
                config.station.air_temp_key,
            ] if k
        ]

        best: Optional[dict] = None
        best_score = -1

        for dev in candidates:
            raw_id = dev.get("id", {})
            dev_id = raw_id.get("id") if isinstance(raw_id, dict) else str(raw_id)
            if not dev_id:
                continue
            try:
                latest = await self._tb.get_latest_telemetry("DEVICE", dev_id, keys_to_check)
                score = len(latest)
            except Exception:
                score = 0
            if score > best_score:
                best_score = score
                best = dev

        # Fallback: shortest name (parent device before derived children)
        if best is None and candidates:
            best = min(candidates, key=lambda d: len(d.get("name", "")))
        return best

    # ── Sentinel helpers ───────────────────────────────────────────────────

    def _build_sentinel_records(
        self, start: datetime, end: datetime, reason: str
    ) -> List[dict]:
        """Build -1 sentinel records for every 1-minute boundary in [start, end].

        Aligned to second=0 so repeated writes across cycles overwrite cleanly.
        """
        records: List[dict] = []
        ts = start.replace(second=0, microsecond=0)
        while ts <= end:
            records.append({
                "ts": int(ts.timestamp() * 1000),
                "values": {
                    KEY_POTENTIAL_POWER: -1,
                    KEY_PVLIB_POWER: -1,
                    KEY_DATA_SOURCE: f"error:{reason}",
                    KEY_MODEL_VERSION: MODEL_VERSION,
                    KEY_UNIT: "kW",
                },
            })
            ts += timedelta(minutes=1)
        return records

    async def _write_sentinels(self, asset_id: str, sentinels: List[dict]) -> None:
        """Best-effort sentinel write — logs on failure but does not re-raise."""
        if not sentinels:
            return
        try:
            await self._tb.post_telemetry("ASSET", asset_id, sentinels)
        except Exception as exc:
            log.error("_write_sentinels: write failed for %s: %s — moving on", asset_id, exc)

    async def _fill_coverage_gaps(
        self,
        asset_id: str,
        df_result: pd.DataFrame,
        start: datetime,
        end: datetime,
    ) -> None:
        """Write sentinels for any 1-min boundaries in [start, end] absent from df_result.

        Handles partial station coverage (e.g., 5-minute outage within the window).
        Real records are never overwritten — only missing minutes get sentinels.

        Rules:
          1. Walk only minute boundaries that fall strictly inside [start, end].
             Flooring `start` to the minute would include a boundary before the
             window and emit a spurious sentinel for a minute we never asked about.
          2. Coverage is matched by minute-bucket (floor real ts to its minute),
             not by exact-ms equality. The physics pipeline emits at station-native
             timestamps (e.g., 10:47:45), and we still want that to "cover" the
             10:47:00 minute.
          3. Never emit a sentinel with a ts LATER than the most recent real record.
             TB's "Latest telemetry" tab picks the row with the max ts — a future-
             minute sentinel would clobber the real value on the live widget even
             though the station is healthy. Historical gaps (strictly earlier than
             the last real record) are still filled so the sparkline stays complete.
        """
        if df_result.empty:
            return

        # Rule 2: coverage by minute-bucket (ms).
        covered_minutes_ms: Set[int] = set()
        for ts in df_result.index:
            t = pd.Timestamp(ts)
            if t.tzinfo is None:
                t = t.tz_localize("UTC")
            minute_floor = t.floor("min")
            covered_minutes_ms.add(int(minute_floor.timestamp() * 1000))

        # Rule 3: clamp to "not after the last real record".
        last_real_ts_ms = int(pd.Timestamp(df_result.index[-1]).timestamp() * 1000)

        # Rule 1: walk only full minute boundaries inside the window.
        cur = start.replace(second=0, microsecond=0)
        if cur < start:
            cur += timedelta(minutes=1)

        gap_records: List[dict] = []
        while cur <= end:
            ts_ms = int(cur.timestamp() * 1000)
            if ts_ms > last_real_ts_ms:
                # Don't clobber "Latest telemetry" with a future-minute sentinel.
                break
            if ts_ms not in covered_minutes_ms:
                gap_records.append({
                    "ts": ts_ms,
                    "values": {
                        KEY_POTENTIAL_POWER: -1,
                        KEY_PVLIB_POWER: -1,
                        KEY_DATA_SOURCE: "error:partial_coverage",
                        KEY_MODEL_VERSION: MODEL_VERSION,
                        KEY_UNIT: "kW",
                    },
                })
            cur += timedelta(minutes=1)

        if gap_records:
            log.debug("_fill_coverage_gaps: %d gap sentinels for %s", len(gap_records), asset_id)
            await self._write_sentinels(asset_id, gap_records)


# ── Module-level helpers ───────────────────────────────────────────────────

def _build_ac_telemetry(df: pd.DataFrame) -> List[dict]:
    """Convert physics result DataFrame to TB telemetry record list."""
    records: List[dict] = []
    for ts, row in df.iterrows():
        ts_ms = int(pd.Timestamp(ts).timestamp() * 1000)
        kw = float(row["potential_power_kw"])
        if kw < 0:
            kw = 0.0
        records.append({
            "ts": ts_ms,
            "values": {
                KEY_POTENTIAL_POWER: round(kw, 3),
                KEY_PVLIB_POWER: round(kw, 3),
                KEY_DATA_SOURCE: str(row.get("data_source", "unknown")),
                KEY_MODEL_VERSION: str(row.get("model_version", MODEL_VERSION)),
                KEY_UNIT: "kW",
            },
        })
    return records
