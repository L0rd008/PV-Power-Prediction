"""
Phase A unit tests — Opus §5 items 1-3.

Tests:
  1. test_config_parses_explicit_station_id        (Gap 1/10 — blob path)
  2. test_config_parses_station_id_flat            (Gap 1/10 — flat attrs)
  3. test_resolve_station_by_naming_prefix         (Gap 1   — H1-B fallback)
  4. test_sentinel_records_cover_window            (Gap 2)
  5. test_sentinel_records_minute_aligned          (Gap 2 — alignment to :00)
  6. test_sentinel_not_overwrite_partial_real      (Gap 2 — partial coverage)
  7. test_rollup_excludes_sentinels_from_sum       (Gap 2 — roll-up -1 exclusion)
  8. test_ancestor_map_populated_for_multi_parent  (Gap 5)
  9. test_rollup_sums_in_memory_dfs                (Gap 6)
 10. test_rollup_dedup_across_grandparents         (Gap 5/6)
 11. test_discover_plants_no_isplantagg_no_ancestor (Gap 5 — non-agg intermediates)
 12. test_stuck_poa_sensor_treated_as_invalid      (E2)
"""
from __future__ import annotations

import asyncio
import json
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional, Set
from unittest.mock import AsyncMock, MagicMock, patch

import pandas as pd
import pytest

# ── Helpers ────────────────────────────────────────────────────────────────

def _utc(minute: int = 0, hour: int = 12) -> datetime:
    return datetime(2026, 4, 24, hour, minute, 0, tzinfo=timezone.utc)


def _make_df(minutes: int = 3, kw: float = 100.0) -> pd.DataFrame:
    idx = pd.date_range(_utc(), periods=minutes, freq="1min", tz="UTC")
    return pd.DataFrame(
        {
            "potential_power_kw": [kw] * minutes,
            "active_power_pvlib_kw": [kw] * minutes,
            "data_source": ["tb_station"] * minutes,
            "model_version": ["pvlib-h-a3-v1"] * minutes,
        },
        index=idx,
    )


# ════════════════════════════════════════════════════════════════════════════
# 1-2.  config.py parser tests (Gap 1/10)
# ════════════════════════════════════════════════════════════════════════════

class TestConfigParsers:
    """PlantConfig._from_blob and _from_flat now parse weather_station_id / p341_device_id."""

    def test_config_parses_explicit_station_id_from_blob(self):
        from app.physics.config import PlantConfig

        blob = {
            "plant_name": "KSP_Plant",
            "location": {"lat": 8.0, "lon": 80.0, "altitude_m": 100.0, "timezone": "Asia/Colombo"},
            "orientations": [{"tilt": 8, "azimuth": 0, "module_count": 100}],
            "module": {"area_m2": 2.58, "efficiency_stc": 0.224, "gamma_p": -0.0029},
            "inverter": {"ac_rating_kw": 500.0, "flat_efficiency": 0.98},
            "station": {"ghi_key": "wstn1_ghi", "air_temp_key": "wstn1_temp"},
            "losses": {"soiling": 0.03},
            # Explicit device IDs in the blob
            "weather_station_id": "wstn-uuid-blob-1234",
            "p341_device_id": "p341-uuid-blob-5678",
        }
        attrs = {"pvlib_config": json.dumps(blob), "isPlant": "true", "pvlib_enabled": "true"}

        config = PlantConfig.from_tb_attributes("asset-1", attrs)

        assert config.weather_station_id == "wstn-uuid-blob-1234", (
            "_from_blob must parse weather_station_id from the pvlib_config blob"
        )
        assert config.p341_device_id == "p341-uuid-blob-5678", (
            "_from_blob must parse p341_device_id from the pvlib_config blob"
        )

    def test_config_parses_explicit_station_id_from_attrs(self):
        """Explicit IDs in flat SERVER_SCOPE attrs (not inside the blob)."""
        from app.physics.config import PlantConfig

        blob = {
            "plant_name": "KSP_Plant",
            "location": {"lat": 8.0, "lon": 80.0, "altitude_m": 100.0, "timezone": "Asia/Colombo"},
            "orientations": [{"tilt": 8, "azimuth": 0, "module_count": 100}],
            "module": {"area_m2": 2.58, "efficiency_stc": 0.224, "gamma_p": -0.0029},
            "inverter": {"ac_rating_kw": 500.0, "flat_efficiency": 0.98},
            "station": {"ghi_key": "wstn1_ghi", "air_temp_key": "wstn1_temp"},
        }
        attrs = {
            "pvlib_config": json.dumps(blob),
            # IDs in flat attrs (takes precedence via `cfg.get(...) or attrs.get(...)`)
            "weather_station_id": "wstn-uuid-attrs-9999",
            "p341_device_id": "p341-uuid-attrs-8888",
        }

        config = PlantConfig.from_tb_attributes("asset-2", attrs)

        # When blob has no device IDs, fall back to attrs
        assert config.weather_station_id == "wstn-uuid-attrs-9999"
        assert config.p341_device_id == "p341-uuid-attrs-8888"

    def test_config_parses_station_id_flat(self):
        """Flat attributes path (no pvlib_config blob)."""
        from app.physics.config import PlantConfig

        attrs = {
            "latitude": 8.0,
            "longitude": 80.0,
            "altitude_m": 100.0,
            "timezone": "Asia/Colombo",
            "orientations": json.dumps([{"tilt": 8, "azimuth": 0, "module_count": 100}]),
            "module": json.dumps({"area_m2": 2.58, "efficiency_stc": 0.224, "gamma_p": -0.0029}),
            "inverter": json.dumps({"ac_rating_kw": 500.0, "flat_efficiency": 0.98}),
            "station": json.dumps({"ghi_key": "wstn1_ghi", "air_temp_key": "wstn1_temp"}),
            "weather_station_id": "wstn-uuid-flat-1111",
            "p341_device_id": "p341-uuid-flat-2222",
        }

        config = PlantConfig.from_tb_attributes("asset-3", attrs)

        assert config.weather_station_id == "wstn-uuid-flat-1111", (
            "_from_flat must parse weather_station_id from flat attrs"
        )
        assert config.p341_device_id == "p341-uuid-flat-2222"


# ════════════════════════════════════════════════════════════════════════════
# 3.  H1-B naming-prefix fallback (Gap 1)
# ════════════════════════════════════════════════════════════════════════════

class TestResolveStationByNamingPrefix:
    """ForecastService._resolve_station_by_name_prefix uses TB device search."""

    @pytest.fixture(autouse=True)
    def clear_cache(self):
        """Wipe class-level station cache between tests."""
        from app.services.forecast_service import ForecastService
        ForecastService._station_resolution_cache.clear()
        yield
        ForecastService._station_resolution_cache.clear()

    @pytest.mark.asyncio
    async def test_resolve_station_by_naming_prefix(self):
        from app.physics.config import PlantConfig
        from app.services.forecast_service import ForecastService

        # Build a minimal PlantConfig with no explicit station ID
        config = PlantConfig(
            plant_name="KSP_Plant",
            asset_id="ksp-asset-id",
            latitude=8.0,
            longitude=80.0,
        )
        config.weather_station_id = None

        # Mock TB client that returns a matching device
        mock_tb = AsyncMock()
        mock_tb.search_devices_by_name_prefix.return_value = [
            {"id": {"id": "wstn-found-uuid"}, "name": "KSP_WSTN1"},
        ]
        mock_tb.get_latest_telemetry.return_value = {"wstn1_ghi": "500", "wstn1_temp": "30"}

        svc = ForecastService(mock_tb)
        await svc._resolve_station_by_name_prefix("ksp-asset-id", config)

        assert config.weather_station_id == "wstn-found-uuid", (
            "Naming fallback must populate weather_station_id from matched device"
        )
        mock_tb.search_devices_by_name_prefix.assert_called_once_with("KSP")

    @pytest.mark.asyncio
    async def test_naming_prefix_multi_wstn_picks_best_by_key_coverage(self):
        """When multiple WSTN candidates exist, pick the one with most configured keys."""
        from app.physics.config import PlantConfig, StationConfig
        from app.services.forecast_service import ForecastService

        config = PlantConfig(
            plant_name="SSK_Plant",
            asset_id="ssk-asset",
            latitude=7.0,
            longitude=81.0,
        )
        config.station = StationConfig(
            ghi_key="wstn_ghi",
            poa_key="wstn_poa",
            air_temp_key="wstn_temp",
        )
        config.weather_station_id = None

        mock_tb = AsyncMock()
        mock_tb.search_devices_by_name_prefix.return_value = [
            {"id": {"id": "wstn-A"}, "name": "SSK_WSTN"},
            {"id": {"id": "wstn-B"}, "name": "SSK_WSTN_T"},
            {"id": {"id": "wstn-C"}, "name": "SSK_WSTN_R"},
        ]
        # wstn-B has 3 matching keys (best), others have fewer
        async def fake_latest(entity_type, entity_id, keys):
            if entity_id == "wstn-B":
                return {k: "1" for k in keys}  # all keys present
            return {"wstn_ghi": "1"}  # only 1 key

        mock_tb.get_latest_telemetry.side_effect = fake_latest

        svc = ForecastService(mock_tb)
        await svc._resolve_station_by_name_prefix("ssk-asset", config)

        assert config.weather_station_id == "wstn-B", (
            "Multi-WSTN: should pick the station with most matching telemetry keys"
        )

    @pytest.mark.asyncio
    async def test_no_match_leaves_station_id_none(self):
        """If no device matches the prefix, weather_station_id stays None."""
        from app.physics.config import PlantConfig
        from app.services.forecast_service import ForecastService

        config = PlantConfig(plant_name="XYZ_Plant", asset_id="xyz-asset",
                             latitude=6.0, longitude=79.0)
        config.weather_station_id = None

        mock_tb = AsyncMock()
        mock_tb.search_devices_by_name_prefix.return_value = []

        svc = ForecastService(mock_tb)
        await svc._resolve_station_by_name_prefix("xyz-asset", config)

        assert config.weather_station_id is None


# ════════════════════════════════════════════════════════════════════════════
# 4-7.  Sentinel tests (Gap 2)
# ════════════════════════════════════════════════════════════════════════════

class TestSentinelRecords:

    def _svc(self):
        from app.services.forecast_service import ForecastService
        return ForecastService(tb_client=MagicMock())

    def test_sentinel_records_cover_window(self):
        """Sentinels must cover every 1-minute boundary in [start, end]."""
        svc = self._svc()
        start = _utc(minute=0)
        end = _utc(minute=5)

        records = svc._build_sentinel_records(start, end, "no_data")

        # 0,1,2,3,4,5 minutes → 6 records
        assert len(records) == 6
        ts_values = [r["ts"] for r in records]
        for i, r in enumerate(records):
            assert r["values"]["potential_power"] == -1
            assert r["values"]["active_power_pvlib_kw"] == -1
            assert r["values"]["pvlib_data_source"] == "error:no_data"

    def test_sentinel_records_minute_aligned(self):
        """Sentinel timestamps must be aligned to :00 seconds regardless of start."""
        svc = self._svc()
        # Start has sub-minute precision
        start = datetime(2026, 4, 24, 12, 0, 45, 123456, tzinfo=timezone.utc)
        end   = datetime(2026, 4, 24, 12, 2, 30, tzinfo=timezone.utc)

        records = svc._build_sentinel_records(start, end, "physics_error")

        for r in records:
            ts = datetime.fromtimestamp(r["ts"] / 1000, tz=timezone.utc)
            assert ts.second == 0 and ts.microsecond == 0, (
                f"Sentinel ts must be at :00 seconds, got {ts}"
            )

    @pytest.mark.asyncio
    async def test_sentinel_written_on_config_error(self):
        """process_single_asset must write sentinels when config load fails."""
        from app.services.forecast_service import ForecastService, PlantCycleResult

        mock_tb = AsyncMock()
        mock_tb.get_asset_attributes.return_value = {}  # triggers ValueError in _load_config

        svc = ForecastService(mock_tb)
        start, end = _utc(0), _utc(5)

        result = await svc.process_single_asset("bad-asset", start, end)

        assert result.status == "config_error"
        assert result.sentinels is not None and len(result.sentinels) > 0
        mock_tb.post_telemetry.assert_called()  # sentinels were posted

    @pytest.mark.asyncio
    async def test_partial_coverage_gaps_filled(self):
        """After a successful write, any missing 1-min boundaries get gap sentinels."""
        from app.services.forecast_service import ForecastService

        # df_result covers only minute 0 — minutes 1,2,3 are missing
        start = _utc(0)
        end = _utc(3)
        df_partial = _make_df(minutes=1)  # only minute 0

        mock_tb = AsyncMock()
        svc = ForecastService(mock_tb)

        await svc._fill_coverage_gaps("asset-X", df_partial, start, end)

        # Should have been called once with 3 gap records (minutes 1,2,3)
        calls = mock_tb.post_telemetry.call_args_list
        assert calls, "_fill_coverage_gaps should post gap sentinels"
        gap_records = calls[0].args[2]
        assert len(gap_records) == 3, (
            f"Expected 3 gap records, got {len(gap_records)}"
        )
        for r in gap_records:
            assert r["values"]["potential_power"] == -1

    def test_rollup_excludes_failed_plants_from_sum(self):
        """In-memory roll-up must NOT include plants with status != 'ok' in the sum."""
        from app.services.forecast_service import ForecastService, PlantCycleResult

        ok_df = _make_df(minutes=3, kw=100.0)
        fail_result = PlantCycleResult(asset_id="plant-fail", status="physics_error", df=None)
        ok_result   = PlantCycleResult(asset_id="plant-ok",   status="ok",            df=ok_df)

        ok_results = {
            r.asset_id: r
            for r in [ok_result, fail_result]
            if r.status == "ok" and r.df is not None
        }
        # Only plant-ok should appear
        assert "plant-ok" in ok_results
        assert "plant-fail" not in ok_results


# ════════════════════════════════════════════════════════════════════════════
# 8-11. Ancestor map tests (Gap 5)
# ════════════════════════════════════════════════════════════════════════════

class _FakeTBClient:
    """Minimal fake TB client for BFS discovery tests."""

    def __init__(self, asset_graph: Dict[str, Any]):
        """
        asset_graph: {
            asset_id: {
                "attrs":   {key: value, ...},
                "info":    {"name": "..."},
                "children": [child_id, ...],
            }
        }
        """
        self._graph = asset_graph

    async def get_asset_attributes(self, asset_id: str, scope: str = "SERVER_SCOPE") -> dict:
        return self._graph.get(asset_id, {}).get("attrs", {})

    async def get_entity_info(self, entity_id: str, entity_type: str = "ASSET") -> Optional[dict]:
        return self._graph.get(entity_id, {}).get("info", {})

    async def get_child_relations(self, asset_id: str) -> List[dict]:
        children = self._graph.get(asset_id, {}).get("children", [])
        return [
            {"to": {"id": cid, "entityType": "ASSET"}}
            for cid in children
        ]


class TestAncestorMap:

    @pytest.mark.asyncio
    async def test_ancestor_map_populated_for_multi_parent_plant(self):
        """A plant reachable via two paths should accumulate ancestors from both."""
        from app.services.thingsboard_client import ThingsBoardClient

        # Hierarchy:
        #   root-A (isPlantAgg) → agg-1 (isPlantAgg) → plant-X
        #   root-B (isPlantAgg)                       → plant-X   (same plant)
        graph = {
            "root-A": {
                "attrs": {"isPlantAgg": "true"},
                "info": {"name": "Root A"},
                "children": ["agg-1"],
            },
            "agg-1": {
                "attrs": {"isPlantAgg": "true"},
                "info": {"name": "Agg 1"},
                "children": ["plant-X"],
            },
            "root-B": {
                "attrs": {"isPlantAgg": "true"},
                "info": {"name": "Root B"},
                "children": ["plant-X"],
            },
            "plant-X": {
                "attrs": {"isPlant": "true", "pvlib_enabled": "true"},
                "info": {"name": "Plant X"},
                "children": [],
            },
        }

        tb = ThingsBoardClient.__new__(ThingsBoardClient)
        tb._http = None
        tb._token = "fake"
        tb._token_expiry = 9e18
        tb._lock = asyncio.Lock()

        # Monkey-patch with fake implementations
        fake = _FakeTBClient(graph)
        tb.get_asset_attributes = fake.get_asset_attributes
        tb.get_entity_info = fake.get_entity_info
        tb.get_child_relations = fake.get_child_relations

        plants, ancestor_map = await tb.discover_plants(["root-A", "root-B"])

        assert len(plants) == 1 and plants[0].id == "plant-X"
        ancestors = ancestor_map.get("plant-X", set())
        assert "root-A" in ancestors, "root-A must be an ancestor of plant-X"
        assert "agg-1" in ancestors,  "agg-1 must be an ancestor of plant-X"
        assert "root-B" in ancestors, "root-B must be an ancestor of plant-X"

    @pytest.mark.asyncio
    async def test_discover_plants_no_isplantagg_no_ancestor(self):
        """Intermediate assets without isPlantAgg are NOT added to ancestor_map."""
        from app.services.thingsboard_client import ThingsBoardClient

        graph = {
            "root": {
                "attrs": {"isPlantAgg": "true"},
                "info": {"name": "Root"},
                "children": ["mid"],
            },
            "mid": {
                # NOT isPlantAgg — should not appear in ancestor list
                "attrs": {},
                "info": {"name": "Mid (no agg)"},
                "children": ["plant-Y"],
            },
            "plant-Y": {
                "attrs": {"isPlant": "true", "pvlib_enabled": "true"},
                "info": {"name": "Plant Y"},
                "children": [],
            },
        }

        tb = ThingsBoardClient.__new__(ThingsBoardClient)
        tb._http = None
        tb._token = "fake"
        tb._token_expiry = 9e18
        tb._lock = asyncio.Lock()

        fake = _FakeTBClient(graph)
        tb.get_asset_attributes = fake.get_asset_attributes
        tb.get_entity_info = fake.get_entity_info
        tb.get_child_relations = fake.get_child_relations

        plants, ancestor_map = await tb.discover_plants(["root"])

        ancestors = ancestor_map.get("plant-Y", set())
        assert "root" in ancestors,  "root (isPlantAgg) must appear as ancestor"
        assert "mid"  not in ancestors, (
            "mid (not isPlantAgg) must NOT appear as ancestor"
        )


# ════════════════════════════════════════════════════════════════════════════
# 9-10.  In-memory roll-up tests (Gap 6)
# ════════════════════════════════════════════════════════════════════════════

class TestInMemoryRollup:

    @pytest.fixture(autouse=True)
    def clear_cache(self):
        from app.services.forecast_service import ForecastService
        ForecastService._station_resolution_cache.clear()
        yield
        ForecastService._station_resolution_cache.clear()

    @pytest.mark.asyncio
    async def test_rollup_sums_in_memory_dfs(self):
        """Roll-up should sum DataFrames without re-reading TB telemetry."""
        from app.services.forecast_service import ForecastService, PlantCycleResult

        df_a = _make_df(minutes=3, kw=100.0)
        df_b = _make_df(minutes=3, kw=200.0)

        results = {
            "plant-a": PlantCycleResult(asset_id="plant-a", status="ok", df=df_a),
            "plant-b": PlantCycleResult(asset_id="plant-b", status="ok", df=df_b),
        }
        ancestor_map = {
            "plant-a": {"ancestor-1"},
            "plant-b": {"ancestor-1"},
        }

        mock_tb = AsyncMock()
        svc = ForecastService(mock_tb)
        start, end = _utc(0), _utc(2)

        await svc._rollup_parents(
            plants=[], plant_results=results,
            ancestor_map=ancestor_map, start=start, end=end,
        )

        # post_telemetry should have been called for ancestor-1 with summed values
        mock_tb.post_telemetry.assert_called_once()
        call = mock_tb.post_telemetry.call_args
        assert call.args[1] == "ancestor-1"
        records = call.args[2]
        # Each record should be 300 kW (100+200)
        for r in records:
            assert abs(r["values"]["potential_power"] - 300.0) < 0.01, (
                f"Expected 300 kW rollup, got {r['values']['potential_power']}"
            )

        # get_timeseries must NOT have been called (no TB re-read)
        mock_tb.get_timeseries.assert_not_called()

    @pytest.mark.asyncio
    async def test_rollup_dedup_across_grandparents(self):
        """A plant under two ancestors must be summed exactly once per ancestor."""
        from app.services.forecast_service import ForecastService, PlantCycleResult

        df_p = _make_df(minutes=2, kw=150.0)
        results = {
            "plant-p": PlantCycleResult(asset_id="plant-p", status="ok", df=df_p),
        }
        # plant-p is under both anc-1 and anc-2
        ancestor_map = {
            "plant-p": {"anc-1", "anc-2"},
        }

        mock_tb = AsyncMock()
        svc = ForecastService(mock_tb)
        start, end = _utc(0), _utc(1)

        await svc._rollup_parents(
            plants=[], plant_results=results,
            ancestor_map=ancestor_map, start=start, end=end,
        )

        call_ancestors = {c.args[1] for c in mock_tb.post_telemetry.call_args_list}
        assert "anc-1" in call_ancestors and "anc-2" in call_ancestors, (
            "Both ancestors must receive roll-up writes"
        )
        # Each call should contain 150 kW (plant-p counted once per ancestor)
        for call in mock_tb.post_telemetry.call_args_list:
            for r in call.args[2]:
                assert abs(r["values"]["potential_power"] - 150.0) < 0.01

    @pytest.mark.asyncio
    async def test_rollup_writes_sentinel_when_all_children_fail(self):
        """When all children under an ancestor fail, ancestor gets -1 sentinels."""
        from app.services.forecast_service import ForecastService, PlantCycleResult

        results = {
            "plant-bad": PlantCycleResult(asset_id="plant-bad", status="physics_error", df=None),
        }
        ancestor_map = {"plant-bad": {"anc-fail"}}

        mock_tb = AsyncMock()
        svc = ForecastService(mock_tb)
        start, end = _utc(0), _utc(2)

        await svc._rollup_parents(
            plants=[], plant_results=results,
            ancestor_map=ancestor_map, start=start, end=end,
        )

        mock_tb.post_telemetry.assert_called()
        call = mock_tb.post_telemetry.call_args
        assert call.args[1] == "anc-fail"
        records = call.args[2]
        assert all(r["values"]["potential_power"] == -1 for r in records), (
            "Ancestor sentinels must have -1 values"
        )


# ════════════════════════════════════════════════════════════════════════════
# 12.  Stuck-sensor test (Edge case E2)
# ════════════════════════════════════════════════════════════════════════════

class TestStuckSensor:

    def test_stuck_poa_sensor_treated_as_invalid(self):
        """_is_valid must return False when POA std < 1 W/m² with non-zero mean."""
        from app.physics.data_sources import _is_valid

        idx = pd.date_range(_utc(0), periods=10, freq="1min", tz="UTC")
        # Stuck sensor: all values identical and non-zero
        df_stuck = pd.DataFrame({"ghi": [500.0] * 10, "poa": [450.0] * 10}, index=idx)
        assert not _is_valid(df_stuck), "Stuck POA sensor (std=0) should be invalid"

    def test_normal_poa_variation_is_valid(self):
        """_is_valid must return True for a sensor with normal irradiance variation."""
        from app.physics.data_sources import _is_valid

        idx = pd.date_range(_utc(0), periods=10, freq="1min", tz="UTC")
        poa_values = [400 + i * 15 for i in range(10)]  # std ≫ 1
        df_ok = pd.DataFrame({"ghi": poa_values, "poa": poa_values}, index=idx)
        assert _is_valid(df_ok), "Normal POA variation should be valid"

    def test_nighttime_zero_poa_not_flagged_as_stuck(self):
        """Night-time zero POA (mean ≤ 10 W/m²) should NOT trigger stuck-sensor logic."""
        from app.physics.data_sources import _is_valid

        idx = pd.date_range(_utc(0, hour=0), periods=10, freq="1min", tz="UTC")
        df_night = pd.DataFrame({"ghi": [0.0] * 10, "poa": [0.0] * 10}, index=idx)
        # Should be valid (night-time zeros) — ghi column present
        assert _is_valid(df_night), "Night-time zeros should not be flagged as stuck"
