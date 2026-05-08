"""
SSK pvlib configuration hotfix utility.

Dry-run by default. Use --apply to write SERVER_SCOPE attrs.
Optional --run-asset computes a current-window smoke test after the attr write.
Optional --backfill-start/--backfill-end recomputes potential_power for date range.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
from copy import deepcopy
from datetime import date, datetime, time, timedelta, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

SERVICE_ROOT = Path(__file__).resolve().parents[1]
if str(SERVICE_ROOT) not in sys.path:
    sys.path.insert(0, str(SERVICE_ROOT))

from dotenv import load_dotenv

from app.config import settings
from app.services.forecast_service import ForecastService
from app.services.thingsboard_client import ThingsBoardClient

SSK_ASSET_ID = "3c2492b0-5669-11f0-b892-f5acae6d9b71"
SSK_WSTN_ID = "d43fa340-6853-11f0-9429-751fa577e736"
TZ_LOCAL = "Asia/Colombo"


def _jparse(value, default):
    if value is None:
        return deepcopy(default)
    if isinstance(value, str):
        try:
            return json.loads(value)
        except ValueError:
            return deepcopy(default)
    return deepcopy(value)


def _canonical_payload(attrs: dict) -> dict:
    station = _jparse(attrs.get("station"), {})
    station.update({
        "ghi_key": None,
        "poa_key": "wstn1_tilted_irradiance",
        "air_temp_key": station.get("air_temp_key") or "wstn1_temperature_ambient",
        "wind_speed_key": station.get("wind_speed_key") or "wstn1_wind_speed",
        "freshness_minutes": int(station.get("freshness_minutes", 10)),
        "sanity_max_ghi_wm2": float(station.get("sanity_max_ghi_wm2", 1400)),
        "sanity_max_poa_wm2": float(station.get("sanity_max_poa_wm2", 1500)),
    })

    orientations = _jparse(attrs.get("orientations"), [])
    if not orientations:
        orientations = [{"name": "Main", "tilt": 0, "azimuth": 0}]
    orientations[0].update({
        "module_count": 22040,
        "use_measured_poa": True,
    })

    inverter = _jparse(attrs.get("inverter"), {})
    inverter["ac_rating_kw"] = 10200

    payload = {
        "station": station,
        "orientations": orientations,
        "inverter": inverter,
        "Capacity": 10000,
        "capacityUnit": "kW",
        "use_measured_poa": True,
        "weather_station_id": SSK_WSTN_ID,
    }
    if attrs.get("pvlib_config") is not None:
        payload["pvlib_config"] = None
    return payload


def _summarize(label: str, value) -> None:
    print(f"{label}: {json.dumps(value, default=str, sort_keys=True)}")


async def _run_asset(tb: ThingsBoardClient, asset_id: str) -> dict:
    svc = ForecastService(tb, solcast_api_key=settings.SOLCAST_API_KEY)
    now = datetime.now(timezone.utc)
    end = now - timedelta(seconds=settings.READ_LAG_SECONDS)
    start = end - timedelta(seconds=settings.READ_WINDOW_SECONDS)
    return (await svc.process_single_asset(asset_id, start, end)).to_dict()


async def _backfill(tb: ThingsBoardClient, asset_id: str, start_day: date, end_day: date) -> list[dict]:
    svc = ForecastService(tb, solcast_api_key=settings.SOLCAST_API_KEY)
    tz = ZoneInfo(TZ_LOCAL)
    results = []
    cursor = start_day
    while cursor <= end_day:
        start_local = datetime.combine(cursor, time.min, tzinfo=tz)
        end_local = start_local + timedelta(days=1)
        result = await svc.process_single_asset(
            asset_id,
            start_local.astimezone(timezone.utc),
            end_local.astimezone(timezone.utc),
        )
        results.append({"date": str(cursor), **result.to_dict()})
        cursor += timedelta(days=1)
    return results


async def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--asset-id", default=SSK_ASSET_ID)
    parser.add_argument("--apply", action="store_true")
    parser.add_argument("--run-asset", action="store_true")
    parser.add_argument("--backfill-start", type=date.fromisoformat)
    parser.add_argument("--backfill-end", type=date.fromisoformat)
    args = parser.parse_args()

    load_dotenv()
    host = os.getenv("TB_HOST")
    username = os.getenv("TB_USERNAME")
    password = os.getenv("TB_PASSWORD")
    if not host or not username or not password:
        raise SystemExit("Missing TB_HOST/TB_USERNAME/TB_PASSWORD")

    async with ThingsBoardClient(host, username, password) as tb:
        attrs = await tb.get_asset_attributes(args.asset_id)
        payload = _canonical_payload(attrs)
        backup = Path("C:/tmp") / f"ssk_attrs_backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        backup.write_text(json.dumps(attrs, indent=2, default=str), encoding="utf-8")

        _summarize("backup", str(backup))
        _summarize("current_orientations", _jparse(attrs.get("orientations"), []))
        _summarize("current_station", _jparse(attrs.get("station"), {}))
        _summarize("write_payload", payload)

        if args.apply:
            await tb.post_attributes("ASSET", args.asset_id, "SERVER_SCOPE", payload)
            print("attrs_written: true")
        else:
            print("dry_run: true")

        if args.apply and args.run_asset:
            _summarize("run_asset", await _run_asset(tb, args.asset_id))

        if args.apply and args.backfill_start:
            end_day = args.backfill_end or args.backfill_start
            _summarize("backfill", await _backfill(tb, args.asset_id, args.backfill_start, end_day))


if __name__ == "__main__":
    asyncio.run(main())
