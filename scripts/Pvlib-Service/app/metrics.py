"""
app/metrics.py — Central Prometheus-style counter store (Phase E, Gap 15).

All values are plain Python ints/dicts.  They are written by individual
subsystems (data_sources, thingsboard_client, forecast_service) and read
by GET /metrics in app/api/forecast.py.

No external dependencies — this module must be importable before any
service module initialises so it can be used as a shared sink.
"""
from __future__ import annotations

from typing import Dict, Tuple

# ── Solcast cache counters (written by app/physics/data_sources.py) ─────────
# These mirror the module-level vars already in data_sources.py.
# /metrics reads from data_sources directly (see forecast.py) — these
# aliases exist for documentation and future consolidation only.

# ── Discover cache counters (written by app/services/thingsboard_client.py) ─
discover_cache_hits_total:   int = 0
discover_cache_misses_total: int = 0

# ── Per-cycle data-source distribution (gauge, reset at cycle start) ─────────
# key: source label ("tb_station" | "solcast" | "clearsky" | "rollup" | "error:<reason>")
# value: number of plants that used this source in the *current* cycle
data_source_count: Dict[str, int] = {}

# ── Per-plant failure totals (monotonic counter, never reset) ────────────────
# key: (plant_asset_id, reason_string)   e.g. ("abc-123", "config_error")
# value: cumulative failure count
plant_failures_total: Dict[Tuple[str, str], int] = {}
