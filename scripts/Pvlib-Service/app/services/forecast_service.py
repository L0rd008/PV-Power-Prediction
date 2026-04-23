"""
ForecastService — orchestrates physics computation for one plant or a fleet.

Responsibilities:
  1. Read PlantConfig from ThingsBoard asset attributes
  2. Discover related devices (weather station, P341 meter)
  3. Fetch weather data (3-tier: TB station → Solcast → clear-sky)
  4. Run compute_ac_power() physics pipeline
  5. Dual-write telemetry:
       potential_power         (kW) — primary widget key
       active_power_pvlib_kw   (kW) — ops/diagnostic alias
       total_generation_expected_kwh — daily energy expected
       pvlib_daily_energy_kwh        — ops alias
       pvlib_data_source, pvlib_model_version
  6. Dedup-aware parent roll-up (P3-B): each unique plant counted once per parent
"""
from __future__ import annotations

import asyncio
import logging
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional, Set, Tuple

import pandas as pd

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


class ForecastService:
    """Stateless orchestrator; instantiated per-request or per-cycle."""

    def __init__(self, tb_client, solcast_api_key: Optional[str] = None):
        self._tb = tb_client
        self._solcast_key = solcast_api_key

    # ── Public: single plant ───────────────────────────────────────────────

    async def process_single_asset(
        self,
        asset_id: str,
        start: datetime,
        end: datetime,
    ) -> Dict[str, Any]:
        """Compute and write pvlib forecast for one plant. Returns a result summary."""
        log.info("process_single_asset: %s  [%s → %s]", asset_id, start.isoformat(), end.isoformat())

        try:
            config = await self._load_config(asset_id)
        except Exception as exc:
            log.error("process_single_asset: config load failed for %s: %s", asset_id, exc)
            return {"asset_id": asset_id, "status": "config_error", "error": str(exc)}

        try:
            df_weather, source = await select_irradiance(
                config, start, end, self._tb, self._solcast_key
            )
        except Exception as exc:
            log.error("process_single_asset: irradiance fetch failed for %s: %s", asset_id, exc)
            return {"asset_id": asset_id, "status": "data_error", "error": str(exc)}

        if df_weather.empty:
            log.warning("process_single_asset: no weather data for %s, skipping write", asset_id)
            return {"asset_id": asset_id, "status": "no_data", "source": source}

        try:
            df_result = compute_ac_power(config, df_weather, data_source=source)
        except Exception as exc:
            log.error("process_single_asset: physics failed for %s: %s", asset_id, exc)
            return {"asset_id": asset_id, "status": "physics_error", "error": str(exc)}

        try:
            records = _build_ac_telemetry(df_result)
            await self._tb.post_telemetry("ASSET", asset_id, records)
            log.info("process_single_asset: wrote %d records for %s (source=%s)", len(records), asset_id, source)
        except Exception as exc:
            log.error("process_single_asset: telemetry write failed for %s: %s", asset_id, exc)
            return {"asset_id": asset_id, "status": "write_error", "error": str(exc)}

        total_kwh = df_result["potential_power_kw"].sum() / 60.0  # kW × (1/60 hr) = kWh for 1-min data
        return {
            "asset_id": asset_id,
            "status": "ok",
            "source": source,
            "records": len(records),
            "total_kwh": round(total_kwh, 3),
            "peak_kw": round(df_result["potential_power_kw"].max(), 2),
        }

    # ── Public: fleet (scheduler cycle) ───────────────────────────────────

    async def run_fleet_cycle(
        self,
        root_asset_ids: List[str],
        start: datetime,
        end: datetime,
        max_concurrent: int = 5,
        job_mgr: Optional[JobManager] = None,
    ) -> Dict[str, Any]:
        """Discover enabled plants, compute all, roll up to parents. Returns cycle summary."""
        jm = job_mgr or _default_job_manager
        job = await jm.create()
        await jm.update(
            job.job_id,
            status=JobStatus.RUNNING,
            started_at=datetime.now(timezone.utc),
        )

        try:
            plants, parent_map = await self._tb.discover_plants(root_asset_ids)
            await jm.update(job.job_id, plants_total=len(plants))

            if not plants:
                log.warning("run_fleet_cycle: no pvlib-enabled plants found")
                await jm.update(job.job_id, status=JobStatus.COMPLETED,
                                finished_at=datetime.now(timezone.utc))
                return {"job_id": job.job_id, "plants": 0}

            sem = asyncio.Semaphore(max_concurrent)
            plant_results: Dict[str, Dict] = {}

            async def bounded_compute(plant):
                async with sem:
                    res = await self.process_single_asset(plant.id, start, end)
                    plant_results[plant.id] = res
                    if res.get("status") == "ok":
                        await jm.update(job.job_id,
                                        plants_completed=job.plants_completed + 1)
                    else:
                        await jm.update(job.job_id,
                                        plants_failed=job.plants_failed + 1,
                                        errors=job.errors + [f"{plant.id}: {res.get('error','?')}"])

            await asyncio.gather(*(bounded_compute(p) for p in plants), return_exceptions=True)

            # ── Parent roll-up ─────────────────────────────────────────────
            await self._rollup_parents(plants, plant_results, parent_map, start, end)

            # ── Daily energy write ─────────────────────────────────────────
            now_utc = datetime.now(timezone.utc)
            if now_utc.minute == 0 and now_utc.hour == 1:
                # Write daily totals at 01:00 UTC (adjust to local midnight if needed)
                await self._write_daily_energies(plant_results, now_utc, parent_map, plants)

            failed = sum(1 for r in plant_results.values() if r.get("status") != "ok")
            final_status = JobStatus.COMPLETED if failed == 0 else JobStatus.PARTIAL
            summary = {
                "job_id": job.job_id,
                "plants_processed": len(plant_results),
                "plants_ok": len(plant_results) - failed,
                "plants_failed": failed,
                "results": plant_results,
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

    # ── Parent roll-up (P3-B dedup) ────────────────────────────────────────

    async def _rollup_parents(
        self,
        plants,
        plant_results: Dict[str, Dict],
        parent_map: Dict[str, Set[str]],
        start: datetime,
        end: datetime,
    ) -> None:
        """Sum potential_power to each parent, counting each plant exactly once."""
        # Build: parent_id → set of unique plant IDs under it
        parent_children: Dict[str, Set[str]] = {}
        for plant in plants:
            for parent_id in plant_map_ids(plant, parent_map):
                parent_children.setdefault(parent_id, set()).add(plant.id)

        # For each parent, fetch the AC series for its unique children and sum
        for parent_id, child_ids in parent_children.items():
            ok_children = [
                cid for cid in child_ids
                if plant_results.get(cid, {}).get("status") == "ok"
            ]
            if not ok_children:
                continue

            # Re-fetch each child's just-written telemetry and sum
            # (simpler than keeping the full DataFrame in memory per plant)
            try:
                series_list = []
                for cid in ok_children:
                    raw = await self._tb.get_timeseries(
                        "ASSET", cid,
                        [KEY_POTENTIAL_POWER],
                        start=start, end=end,
                    )
                    records = raw.get(KEY_POTENTIAL_POWER, [])
                    if records:
                        s = pd.Series({
                            pd.Timestamp(r["ts"], unit="ms", tz="UTC"): float(r["value"])
                            for r in records
                        })
                        series_list.append(s)

                if not series_list:
                    continue

                total = sum(series_list).fillna(0.0)
                rollup_records = [
                    {
                        "ts": int(ts.timestamp() * 1000),
                        "values": {
                            KEY_POTENTIAL_POWER: round(float(v), 3),
                            KEY_PVLIB_POWER: round(float(v), 3),
                            KEY_DATA_SOURCE: "rollup",
                            KEY_MODEL_VERSION: "pvlib-h-a3-v1",
                            KEY_UNIT: "kW",
                        },
                    }
                    for ts, v in total.items()
                    if v >= 0
                ]
                await self._tb.post_telemetry("ASSET", parent_id, rollup_records)
                log.debug("rollup: wrote %d records to parent %s (sum of %d plants)",
                          len(rollup_records), parent_id, len(ok_children))

            except Exception as exc:
                log.warning("rollup: failed for parent %s: %s", parent_id, exc)

    async def _write_daily_energies(self, plant_results, ts: datetime, parent_map, plants) -> None:
        """Write daily expected energy totals to each plant and its parents."""
        ts_ms = int(ts.replace(hour=0, minute=0, second=0, microsecond=0).timestamp() * 1000)

        # Per-plant
        for plant_id, result in plant_results.items():
            if result.get("status") != "ok":
                continue
            kwh = result.get("total_kwh", 0.0)
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

        # Parent roll-ups (dedup)
        parent_children: Dict[str, Set[str]] = {}
        for plant in plants:
            for parent_id in plant_map_ids(plant, parent_map):
                parent_children.setdefault(parent_id, set()).add(plant.id)

        for parent_id, child_ids in parent_children.items():
            total_kwh = sum(
                plant_results.get(cid, {}).get("total_kwh", 0.0)
                for cid in child_ids
                if plant_results.get(cid, {}).get("status") == "ok"
            )
            try:
                await self._tb.post_telemetry("ASSET", parent_id, [{
                    "ts": ts_ms,
                    "values": {
                        KEY_DAILY_ENERGY_EXPECTED: round(total_kwh, 3),
                        KEY_PVLIB_DAILY_ENERGY: round(total_kwh, 3),
                    },
                }])
            except Exception as exc:
                log.warning("daily energy parent write failed for %s: %s", parent_id, exc)

    # ── Config loader ──────────────────────────────────────────────────────

    async def _load_config(self, asset_id: str) -> PlantConfig:
        """Load PlantConfig from TB attributes; discover related devices."""
        attrs = await self._tb.get_asset_attributes(asset_id)
        if not attrs:
            raise ValueError(f"No attributes found for asset {asset_id}")

        config = PlantConfig.from_tb_attributes(asset_id, attrs)

        # Discover related devices via 'Contains' relations on the plant
        children = await self._tb.get_child_relations(asset_id)
        for rel in children:
            child_id = rel.get("to", {}).get("id") or rel.get("toId", "")
            child_type = rel.get("to", {}).get("entityType") or rel.get("toEntityType", "")
            if child_id and child_type == "DEVICE":
                child_info = await self._tb.get_entity_info(child_id, "DEVICE")
                child_name = (child_info or {}).get("name", "")
                child_name_upper = child_name.upper()
                if "WSTN" in child_name_upper or "WEATHER" in child_name_upper or "STATION" in child_name_upper:
                    config.weather_station_id = child_id
                    log.debug("_load_config: found weather station %s (%s)", child_id, child_name)
                elif "P341" in child_name_upper or "METER" in child_name_upper:
                    config.p341_device_id = child_id
                    log.debug("_load_config: found P341 meter %s (%s)", child_id, child_name)

        return config


# ── Helpers ────────────────────────────────────────────────────────────────

def _build_ac_telemetry(df: pd.DataFrame) -> List[dict]:
    """Convert physics result DataFrame to TB telemetry record list."""
    records = []
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
                KEY_MODEL_VERSION: str(row.get("model_version", "pvlib-h-a3-v1")),
                KEY_UNIT: "kW",
            },
        })
    return records


def plant_map_ids(plant, parent_map: Dict[str, Set[str]]) -> Set[str]:
    """Return the set of parent IDs for a plant (from plant.parent_ids or parent_map)."""
    return parent_map.get(plant.id, plant.parent_ids)
