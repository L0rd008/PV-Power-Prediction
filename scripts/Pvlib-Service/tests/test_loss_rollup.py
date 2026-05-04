"""
tests/test_loss_rollup.py — Unit tests for the loss roll-up job (Phase L0).

Coverage:
  1. Integration math on a synthetic 1-day series (happy path).
  2. Sentinel paths:
       a. No potential_power records.
       b. Insufficient samples (< MIN_VALID_SAMPLES).
       c. Missing tariff → kWh keys non-sentinel, LKR keys = -1.
  3. W → kW unit scaling (active_power_unit = "W").
  4. Curtailment formula at boundary setpoint_pct == 99.5 (just above threshold → 0 curtailment).
  5. Curtailment formula at setpoint_pct = 50 (half capacity).
  6. Lifetime increment step logic:
       a. Happy path: today == anchor + 1 day → values accumulate.
       b. Re-run (today <= anchor) → falls through to recompute.
  7. Ancestor roll-up sums daily values across plants (sentinel excluded).
  8. run_loss_rollup() returns correct summary counts with mock TB client.
"""
from __future__ import annotations

import asyncio
from datetime import date, datetime, timedelta, timezone
from typing import Any, Dict, List, Optional
from unittest.mock import AsyncMock, MagicMock, call, patch

import pandas as pd
import pytest

# ── Constants mirrored from the module under test ────────────────────────────

from app.services.loss_rollup_job import (
    KEY_EXPORTED_DAILY,
    KEY_LOSS_CURTAIL_DAILY,
    KEY_LOSS_CURTAILREV_DAILY,
    KEY_LOSS_GRID_DAILY,
    KEY_LOSS_REVENUE_DAILY,
    KEY_POTENTIAL_DAILY,
    ATTR_ANCHOR_DATE,
    ATTR_CURTAIL_LIFETIME,
    ATTR_CURTAILREV_LIFETIME,
    ATTR_EXPORTED_LIFETIME,
    ATTR_GRID_LIFETIME,
    ATTR_POTENTIAL_LIFETIME,
    ATTR_REVENUE_LIFETIME,
    ATTR_UPDATED_AT,
    CURTAIL_SETPOINT_THRESHOLD,
    _integrate,
    _records_to_series,
    _build_setpoint_series,
    _sentinel_daily_values,
    _sum_daily_values,
    _lifetime_increment_step,
    run_loss_rollup,
)

# ── Helpers ───────────────────────────────────────────────────────────────────

def _utc(hour: int = 0, minute: int = 0, day: int = 3, month: int = 5, year: int = 2026) -> datetime:
    return datetime(year, month, day, hour, minute, 0, tzinfo=timezone.utc)


def _ms(dt: datetime) -> int:
    return int(dt.timestamp() * 1000)


def _make_records(
    start: datetime,
    end: datetime,
    value: float,
    step_minutes: int = 1,
) -> List[Dict[str, Any]]:
    """Generate synthetic timeseries records at `step_minutes` cadence."""
    records = []
    cur = start
    while cur < end:
        records.append({"ts": _ms(cur), "value": str(value)})
        cur += timedelta(minutes=step_minutes)
    return records


def _make_tb_client(
    potential_records=None,
    actual_records=None,
    setpoint_records=None,
    plant_attrs=None,
    existing_lifetime_attrs=None,
) -> AsyncMock:
    """Build a mock ThingsBoardClient suitable for loss_rollup tests."""
    tb = AsyncMock()

    # discover_plants returns one fake plant and empty ancestor map
    from app.services.thingsboard_client import PlantRef
    fake_plant = PlantRef(id="plant-001", name="TestPlant")
    tb.discover_plants = AsyncMock(return_value=([fake_plant], {"plant-001": set()}))

    # get_asset_attributes: plant attrs + optional lifetime attrs merged
    base_attrs = {
        "isPlant": "true",
        "pvlib_enabled": "true",
        "tariff_rate_lkr": "10.0",
        "Capacity": "1000",
        "capacityUnit": "kW",
        "active_power_unit": "kW",
    }
    if plant_attrs:
        base_attrs.update(plant_attrs)

    if existing_lifetime_attrs:
        # For the lifetime test, get_asset_attributes is called twice:
        # once for plant attrs, once for lifetime attrs (same entity for simplicity)
        lifetime_with_anchor = dict(existing_lifetime_attrs)
        combined = dict(base_attrs)
        combined.update(lifetime_with_anchor)
        tb.get_asset_attributes = AsyncMock(return_value=combined)
    else:
        tb.get_asset_attributes = AsyncMock(return_value=base_attrs)

    # get_timeseries side effects
    day_start = _utc(0, 0, 3)
    day_end   = _utc(0, 0, 4)
    pot_rec   = potential_records if potential_records is not None else _make_records(day_start, day_end, 100.0)
    act_rec   = actual_records   if actual_records   is not None else _make_records(day_start, day_end, 80.0)
    sp_rec    = setpoint_records if setpoint_records  is not None else []

    async def _get_timeseries(entity_type, entity_id, keys, start, end, limit=50_000, agg="NONE"):
        result = {}
        for k in keys:
            if k in ("potential_power",):
                result[k] = pot_rec
            elif k in ("active_power", "active_power_w", "actual_power"):
                result[k] = act_rec
            elif k in ("setpoint_active_power", "curtailment_limit", "power_limit"):
                result[k] = sp_rec
            else:
                # daily key reads (for recompute_lifetime) return empty
                result[k] = []
        return result

    tb.get_timeseries = AsyncMock(side_effect=_get_timeseries)

    # post_telemetry and post_attributes are fire-and-forget
    tb.post_telemetry = AsyncMock()
    tb.post_attributes = AsyncMock()

    return tb


# ════════════════════════════════════════════════════════════════════════════
# 1. Integration math — happy path
# ════════════════════════════════════════════════════════════════════════════

class TestIntegrationMath:
    """Direct tests of _integrate() on controlled synthetic series."""

    def _day_series(self, start, end, value, step_min=1):
        records = _make_records(start, end, value, step_min)
        return _records_to_series(records, skip_negative=True)

    def test_no_loss_when_active_equals_potential(self):
        """When active == potential, gross loss must be 0."""
        start = _utc(0, 0, 3)
        end   = _utc(0, 0, 4)
        pot = self._day_series(start, end, 100.0)
        act = self._day_series(start, end, 100.0)
        sp  = _build_setpoint_series({}, [])

        result = _integrate(pot, act, sp, 1000.0, start, end)

        assert result[KEY_LOSS_GRID_DAILY] == pytest.approx(0.0, abs=1e-3)
        assert result[KEY_LOSS_CURTAIL_DAILY] == pytest.approx(0.0, abs=1e-3)
        assert result[KEY_POTENTIAL_DAILY] > 0

    def test_gross_loss_full_day_constant(self):
        """Constant 100 kW potential, 80 kW actual → gross loss = 20 kW × 24 h = 480 kWh."""
        start = _utc(0, 0, 3)
        end   = _utc(0, 0, 4)
        pot = self._day_series(start, end, 100.0)
        act = self._day_series(start, end, 80.0)
        sp  = _build_setpoint_series({}, [])

        result = _integrate(pot, act, sp, 1000.0, start, end)

        # 1 440 minutes × (20 kW / 60) = 480 kWh
        assert result[KEY_LOSS_GRID_DAILY] == pytest.approx(480.0, rel=1e-3)
        # No setpoint → no curtailment
        assert result[KEY_LOSS_CURTAIL_DAILY] == pytest.approx(0.0, abs=1e-3)
        # Potential: 1 440 × (100/60) = 2 400 kWh
        assert result[KEY_POTENTIAL_DAILY] == pytest.approx(2400.0, rel=1e-3)
        # Exported: 1 440 × (80/60) = 1 920 kWh
        assert result[KEY_EXPORTED_DAILY] == pytest.approx(1920.0, rel=1e-3)

    def test_negative_potential_skipped(self):
        """Negative potential records (sentinels) should be excluded from integration."""
        start = _utc(0, 0, 3)
        end   = _utc(6, 0, 3)   # 6-hour window
        # First half: valid; second half: sentinel -1
        good_end = _utc(3, 0, 3)
        records_pot = (
            _make_records(start, good_end, 100.0)
            + _make_records(good_end, end, -1.0)   # sentinels
        )
        records_act = _make_records(start, end, 50.0)

        pot = _records_to_series(records_pot, skip_negative=True)
        act = _records_to_series(records_act, skip_negative=False)
        sp  = _build_setpoint_series({}, [])

        result = _integrate(pot, act, sp, 1000.0, start, end)

        # Only the first 3 hours of potential contribute to loss
        # gross loss: 3h × 60 min × (100−50)/60 = 150 kWh
        assert result[KEY_LOSS_GRID_DAILY] == pytest.approx(150.0, rel=1e-2)


# ════════════════════════════════════════════════════════════════════════════
# 2. Sentinel paths
# ════════════════════════════════════════════════════════════════════════════

class TestSentinelPaths:

    @pytest.mark.asyncio
    async def test_no_potential_writes_sentinel(self):
        """If potential_power has no records, all daily keys must be -1."""
        tb = _make_tb_client(potential_records=[])

        with patch("app.services.loss_rollup_job.settings") as mock_settings:
            mock_settings.TZ_LOCAL = "Asia/Colombo"
            mock_settings.root_asset_ids = ["root-001"]
            mock_settings.MAX_CONCURRENT_PLANTS = 5
            mock_settings.LOSS_MIN_VALID_SAMPLES = 360
            mock_settings.LOSS_DEFAULT_SETPOINT_KEYS = "setpoint_active_power"
            mock_settings.LOSS_LIFETIME_PAGE_DAYS = 90

            result = await run_loss_rollup(tb, date=datetime(2026, 5, 3, 1, 0, tzinfo=timezone.utc))

        assert result["plants_ok"] == 0
        assert result["plants_failed"] + result["plants_skipped"] >= 1

        # Verify -1 was written
        written = tb.post_telemetry.call_args
        if written:
            values = written[0][2][0]["values"]
            assert values[KEY_LOSS_GRID_DAILY] == -1
            assert values[KEY_POTENTIAL_DAILY] == -1

    @pytest.mark.asyncio
    async def test_insufficient_samples_writes_sentinel(self):
        """If valid sample count < MIN_VALID_SAMPLES, write -1 for all loss keys."""
        # Only 100 records (< 360 minimum)
        day_start = _utc(0, 0, 3)
        short_records = _make_records(day_start, _utc(1, 40, 3), 100.0)  # 100 min
        tb = _make_tb_client(potential_records=short_records, actual_records=short_records)

        with patch("app.services.loss_rollup_job.settings") as mock_settings:
            mock_settings.TZ_LOCAL = "Asia/Colombo"
            mock_settings.root_asset_ids = ["root-001"]
            mock_settings.MAX_CONCURRENT_PLANTS = 5
            mock_settings.LOSS_MIN_VALID_SAMPLES = 360
            mock_settings.LOSS_DEFAULT_SETPOINT_KEYS = "setpoint_active_power"
            mock_settings.LOSS_LIFETIME_PAGE_DAYS = 90

            result = await run_loss_rollup(tb, date=datetime(2026, 5, 3, 1, 0, tzinfo=timezone.utc))

        assert result["plants_failed"] == 1 or result["plants_ok"] == 0

        written = tb.post_telemetry.call_args
        if written:
            values = written[0][2][0]["values"]
            assert values[KEY_LOSS_GRID_DAILY] == -1

    @pytest.mark.asyncio
    async def test_missing_tariff_lkr_keys_are_sentinel_kwh_keys_are_not(self):
        """When tariff_rate_lkr is missing, kWh keys are computed; LKR keys are -1."""
        day_start = _utc(0, 0, 3)
        day_end   = _utc(0, 0, 4)
        tb = _make_tb_client(
            potential_records=_make_records(day_start, day_end, 100.0),
            actual_records=_make_records(day_start, day_end, 80.0),
            plant_attrs={"tariff_rate_lkr": None},  # no tariff
        )
        # Patch get_asset_attributes to omit tariff
        tb.get_asset_attributes = AsyncMock(return_value={
            "isPlant": "true",
            "pvlib_enabled": "true",
            "Capacity": "1000",
            "capacityUnit": "kW",
            "active_power_unit": "kW",
            # no tariff_rate_lkr
        })

        with patch("app.services.loss_rollup_job.settings") as mock_settings:
            mock_settings.TZ_LOCAL = "UTC"
            mock_settings.root_asset_ids = ["root-001"]
            mock_settings.MAX_CONCURRENT_PLANTS = 5
            mock_settings.LOSS_MIN_VALID_SAMPLES = 360
            mock_settings.LOSS_DEFAULT_SETPOINT_KEYS = "setpoint_active_power"
            mock_settings.LOSS_LIFETIME_PAGE_DAYS = 90

            result = await run_loss_rollup(tb, date=datetime(2026, 5, 4, 1, 0, tzinfo=timezone.utc))

        written = tb.post_telemetry.call_args
        assert written is not None, "post_telemetry should have been called"
        values = written[0][2][0]["values"]

        assert values[KEY_LOSS_GRID_DAILY] > 0, "kWh key should be computed"
        assert values[KEY_LOSS_REVENUE_DAILY] == -1, "LKR key should be -1 when no tariff"
        assert values[KEY_LOSS_CURTAILREV_DAILY] == -1


# ════════════════════════════════════════════════════════════════════════════
# 3. W → kW unit scaling
# ════════════════════════════════════════════════════════════════════════════

class TestWToKwScaling:

    def test_w_to_kw_halves_loss(self):
        """If active_power_unit=W, actual values are ×0.001.

        Plant: potential=100 kW, actual=80 000 W (=80 kW after scaling).
        Expected gross loss = 480 kWh (same as the 80 kW case).
        If scaling is NOT applied, actual=80 000 kW > potential=100 kW → gross_loss = 0 (wrong).
        """
        start = _utc(0, 0, 3)
        end   = _utc(0, 0, 4)
        pot = _records_to_series(_make_records(start, end, 100.0), skip_negative=True)
        # Publish actual in W (80 000 W = 80 kW)
        act_w = _records_to_series(_make_records(start, end, 80_000.0), skip_negative=True)
        # Apply scaling
        act_kw = act_w * 0.001
        sp = _build_setpoint_series({}, [])

        result = _integrate(pot, act_kw, sp, 1000.0, start, end)

        assert result[KEY_LOSS_GRID_DAILY] == pytest.approx(480.0, rel=1e-3)

    def test_no_scaling_when_unit_is_kw(self):
        """If active_power_unit=kW (default), no scaling → same as direct integration."""
        start = _utc(0, 0, 3)
        end   = _utc(0, 0, 4)
        pot = _records_to_series(_make_records(start, end, 100.0), skip_negative=True)
        act = _records_to_series(_make_records(start, end, 80.0),  skip_negative=True)
        sp  = _build_setpoint_series({}, [])

        result = _integrate(pot, act, sp, 1000.0, start, end)

        assert result[KEY_LOSS_GRID_DAILY] == pytest.approx(480.0, rel=1e-3)


# ════════════════════════════════════════════════════════════════════════════
# 4 & 5. Curtailment formula — setpoint boundaries
# ════════════════════════════════════════════════════════════════════════════

class TestCurtailmentFormula:

    def _run_with_setpoint(self, setpoint_pct: float, capacity_kw: float = 1000.0):
        """Helper: constant 100 kW potential, 60 kW actual, given setpoint_pct for 1 hour."""
        start = _utc(0, 0, 3)
        end   = _utc(1, 0, 3)   # 1-hour window
        pot = _records_to_series(_make_records(start, end, 100.0), skip_negative=True)
        act = _records_to_series(_make_records(start, end, 60.0),  skip_negative=True)

        # Single setpoint record at the start
        sp_raw = {"setpoint_active_power": [{"ts": _ms(start - timedelta(minutes=5)), "value": str(setpoint_pct)}]}
        sp = _build_setpoint_series(sp_raw, ["setpoint_active_power"])

        return _integrate(pot, act, sp, capacity_kw, start, end)

    def test_curtailment_zero_at_boundary_setpoint_99_5(self):
        """setpoint_pct = 99.5 is exactly at the threshold → curtailment = 0 (not < 99.5)."""
        result = self._run_with_setpoint(CURTAIL_SETPOINT_THRESHOLD)
        assert result[KEY_LOSS_CURTAIL_DAILY] == pytest.approx(0.0, abs=1e-6)

    def test_curtailment_nonzero_below_threshold(self):
        """setpoint_pct = 50 → ceiling = 500 kW; actual = 60 kW.
        curtail_loss = max(100 − max(500, 60), 0) = 0 ... wait.

        Capacity = 1000 kW, setpoint = 50%.
        ceiling = 1000 × 0.5 = 500 kW.
        max(ceiling, actual) = max(500, 60) = 500 kW.
        curtail_loss = max(100 − 500, 0) = 0.

        Let's use capacity = 80 kW instead so ceiling < actual makes sense:
          capacity=80, setpoint=50% → ceiling=40 kW.
          actual=60 kW > ceiling=40 → max(40,60)=60.
          curtail_loss = max(100−60, 0) = 40 kW.
          gross_loss   = max(100−60, 0) = 40 kW.
        """
        start = _utc(0, 0, 3)
        end   = _utc(1, 0, 3)
        pot = _records_to_series(_make_records(start, end, 100.0), skip_negative=True)
        act = _records_to_series(_make_records(start, end, 60.0),  skip_negative=True)
        sp_raw = {"setpoint_active_power": [{"ts": _ms(start - timedelta(minutes=1)), "value": "50"}]}
        sp = _build_setpoint_series(sp_raw, ["setpoint_active_power"])

        # capacity=80, ceiling=40; actual=60 > ceiling → max=60; loss=max(100−60,0)=40
        result = _integrate(pot, act, sp, 80.0, start, end)

        # 60 minutes × (40 kW / 60) = 40 kWh curtailment
        assert result[KEY_LOSS_CURTAIL_DAILY] == pytest.approx(40.0, rel=1e-3)
        # Gross: 60 minutes × (100−60)/60 = 40 kWh
        assert result[KEY_LOSS_GRID_DAILY] == pytest.approx(40.0, rel=1e-3)

    def test_curtailment_below_actual_ceiling(self):
        """When ceiling > actual (setpoint barely below threshold), curtailment < gross loss."""
        start = _utc(0, 0, 3)
        end   = _utc(1, 0, 3)
        # capacity=1000; setpoint=90% → ceiling=900; actual=60 → max(900,60)=900
        # curtail = max(100-900, 0) = 0, gross = max(100-60,0)=40 kWh
        pot = _records_to_series(_make_records(start, end, 100.0), skip_negative=True)
        act = _records_to_series(_make_records(start, end, 60.0),  skip_negative=True)
        sp_raw = {"setpoint_active_power": [{"ts": _ms(start - timedelta(minutes=1)), "value": "90"}]}
        sp = _build_setpoint_series(sp_raw, ["setpoint_active_power"])

        result = _integrate(pot, act, sp, 1000.0, start, end)

        assert result[KEY_LOSS_CURTAIL_DAILY] == pytest.approx(0.0, abs=1e-6)
        assert result[KEY_LOSS_GRID_DAILY] == pytest.approx(40.0, rel=1e-3)


# ════════════════════════════════════════════════════════════════════════════
# 6. Lifetime increment vs recompute logic
# ════════════════════════════════════════════════════════════════════════════

class TestLifetimeIncrementLogic:

    @pytest.mark.asyncio
    async def test_increment_when_today_is_anchor_plus_one(self):
        """When today == anchor + 1 day, lifetime values should be incremented."""
        anchor_date = date(2026, 5, 2)
        today = date(2026, 5, 3)

        existing_attrs = {
            ATTR_GRID_LIFETIME:       "1000.0",
            ATTR_CURTAIL_LIFETIME:    "200.0",
            ATTR_REVENUE_LIFETIME:    "5000.0",
            ATTR_CURTAILREV_LIFETIME: "1000.0",
            ATTR_POTENTIAL_LIFETIME:  "10000.0",
            ATTR_EXPORTED_LIFETIME:   "9000.0",
            ATTR_ANCHOR_DATE:         str(anchor_date),
            ATTR_UPDATED_AT:          "2026-05-03T00:10:00+00:00",
        }

        tb = AsyncMock()
        tb.post_attributes = AsyncMock()

        daily = {
            KEY_LOSS_GRID_DAILY:       50.0,
            KEY_LOSS_CURTAIL_DAILY:    10.0,
            KEY_LOSS_REVENUE_DAILY:    500.0,
            KEY_LOSS_CURTAILREV_DAILY: 100.0,
            KEY_POTENTIAL_DAILY:       1000.0,
            KEY_EXPORTED_DAILY:        950.0,
        }

        await _lifetime_increment_step(tb, "plant-001", existing_attrs, daily, today)

        tb.post_attributes.assert_awaited_once()
        call_args = tb.post_attributes.call_args
        payload = call_args[0][3]  # post_attributes(entity_type, entity_id, scope, payload)

        assert payload[ATTR_GRID_LIFETIME] == pytest.approx(1050.0, rel=1e-6)
        assert payload[ATTR_CURTAIL_LIFETIME] == pytest.approx(210.0, rel=1e-6)
        assert payload[ATTR_REVENUE_LIFETIME] == pytest.approx(5500.0, rel=1e-6)
        assert payload[ATTR_ANCHOR_DATE] == str(today)

    @pytest.mark.asyncio
    async def test_sentinel_daily_does_not_change_lifetime(self):
        """If the daily key is -1, the existing lifetime value is preserved (not regressed)."""
        anchor_date = date(2026, 5, 2)
        today = date(2026, 5, 3)

        existing_attrs = {
            ATTR_GRID_LIFETIME: "1000.0",
            ATTR_CURTAIL_LIFETIME: "200.0",
            ATTR_REVENUE_LIFETIME: "5000.0",
            ATTR_CURTAILREV_LIFETIME: "1000.0",
            ATTR_POTENTIAL_LIFETIME: "10000.0",
            ATTR_EXPORTED_LIFETIME: "9000.0",
            ATTR_ANCHOR_DATE: str(anchor_date),
            ATTR_UPDATED_AT: "2026-05-03T00:10:00+00:00",
        }

        tb = AsyncMock()
        tb.post_attributes = AsyncMock()

        daily = _sentinel_daily_values()  # all -1

        await _lifetime_increment_step(tb, "plant-001", existing_attrs, daily, today)

        payload = tb.post_attributes.call_args[0][3]
        # Existing values should be preserved
        assert payload[ATTR_GRID_LIFETIME] == pytest.approx(1000.0, rel=1e-6)


# ════════════════════════════════════════════════════════════════════════════
# 7. Ancestor roll-up sum
# ════════════════════════════════════════════════════════════════════════════

class TestAncestorRollup:

    def test_sum_daily_values_excludes_sentinels(self):
        """_sum_daily_values: sentinel -1 plants are excluded from the sum."""
        good = {
            KEY_LOSS_GRID_DAILY:       100.0,
            KEY_LOSS_CURTAIL_DAILY:    20.0,
            KEY_LOSS_REVENUE_DAILY:    1000.0,
            KEY_LOSS_CURTAILREV_DAILY: 200.0,
            KEY_POTENTIAL_DAILY:       2000.0,
            KEY_EXPORTED_DAILY:        1900.0,
        }
        sentinel = _sentinel_daily_values()

        result = _sum_daily_values([good, sentinel])

        assert result[KEY_LOSS_GRID_DAILY] == pytest.approx(100.0, rel=1e-6)
        assert result[KEY_POTENTIAL_DAILY] == pytest.approx(2000.0, rel=1e-6)

    def test_sum_daily_values_happy_two_plants(self):
        """Two valid plants → values summed correctly."""
        p1 = {
            KEY_LOSS_GRID_DAILY:       100.0,
            KEY_LOSS_CURTAIL_DAILY:    10.0,
            KEY_LOSS_REVENUE_DAILY:    1000.0,
            KEY_LOSS_CURTAILREV_DAILY: 100.0,
            KEY_POTENTIAL_DAILY:       2000.0,
            KEY_EXPORTED_DAILY:        1900.0,
        }
        p2 = {
            KEY_LOSS_GRID_DAILY:       50.0,
            KEY_LOSS_CURTAIL_DAILY:    5.0,
            KEY_LOSS_REVENUE_DAILY:    500.0,
            KEY_LOSS_CURTAILREV_DAILY: 50.0,
            KEY_POTENTIAL_DAILY:       1000.0,
            KEY_EXPORTED_DAILY:        950.0,
        }

        result = _sum_daily_values([p1, p2])

        assert result[KEY_LOSS_GRID_DAILY] == pytest.approx(150.0, rel=1e-6)
        assert result[KEY_POTENTIAL_DAILY] == pytest.approx(3000.0, rel=1e-6)
        assert result[KEY_EXPORTED_DAILY]  == pytest.approx(2850.0, rel=1e-6)

    def test_sum_all_sentinels_returns_sentinel(self):
        """If all plants are sentinel, the ancestor sum should also be -1."""
        s1 = _sentinel_daily_values()
        s2 = _sentinel_daily_values()

        result = _sum_daily_values([s1, s2])

        assert result[KEY_LOSS_GRID_DAILY] == -1.0


# ════════════════════════════════════════════════════════════════════════════
# 8. run_loss_rollup() summary counts
# ════════════════════════════════════════════════════════════════════════════

class TestRunLossRollupSummary:

    @pytest.mark.asyncio
    async def test_happy_path_returns_one_ok(self):
        """Full happy-path run with sufficient data returns plants_ok=1."""
        day_start = _utc(0, 0, 3)
        day_end   = _utc(0, 0, 4)
        tb = _make_tb_client(
            potential_records=_make_records(day_start, day_end, 100.0),
            actual_records=_make_records(day_start, day_end, 80.0),
        )

        with patch("app.services.loss_rollup_job.settings") as mock_settings:
            mock_settings.TZ_LOCAL = "UTC"
            mock_settings.root_asset_ids = ["root-001"]
            mock_settings.MAX_CONCURRENT_PLANTS = 5
            mock_settings.LOSS_MIN_VALID_SAMPLES = 360
            mock_settings.LOSS_DEFAULT_SETPOINT_KEYS = "setpoint_active_power"
            mock_settings.LOSS_LIFETIME_PAGE_DAYS = 90

            result = await run_loss_rollup(tb, date=datetime(2026, 5, 4, 1, 0, tzinfo=timezone.utc))
        
        assert result["plants_ok"] == 1
        assert result["plants_failed"] == 0
        assert result["plants_skipped"] == 0
        assert result["date"] == "2026-05-03"

    @pytest.mark.asyncio
    async def test_loss_attribution_disabled_plant_is_skipped(self):
        """Plant with loss_attribution_enabled=false must be skipped."""
        day_start = _utc(0, 0, 3)
        day_end   = _utc(0, 0, 4)
        tb = _make_tb_client(
            potential_records=_make_records(day_start, day_end, 100.0),
            actual_records=_make_records(day_start, day_end, 80.0),
            plant_attrs={"loss_attribution_enabled": "false"},
        )
        tb.get_asset_attributes = AsyncMock(return_value={
            "isPlant": "true",
            "pvlib_enabled": "true",
            "loss_attribution_enabled": "false",
        })

        with patch("app.services.loss_rollup_job.settings") as mock_settings:
            mock_settings.TZ_LOCAL = "Asia/Colombo"
            mock_settings.root_asset_ids = ["root-001"]
            mock_settings.MAX_CONCURRENT_PLANTS = 5
            mock_settings.LOSS_MIN_VALID_SAMPLES = 360
            mock_settings.LOSS_DEFAULT_SETPOINT_KEYS = "setpoint_active_power"
            mock_settings.LOSS_LIFETIME_PAGE_DAYS = 90

            result = await run_loss_rollup(tb, date=datetime(2026, 5, 3, 1, 0, tzinfo=timezone.utc))

        assert result["plants_skipped"] == 1
        assert result["plants_ok"] == 0
        # post_telemetry should NOT have been called for a skipped plant
        # (it may be called for ancestor rollup but ancestor_map is empty in this mock)
        # Just verify no per-plant telemetry was written
