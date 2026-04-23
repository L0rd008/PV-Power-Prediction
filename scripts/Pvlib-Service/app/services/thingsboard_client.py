"""
Async ThingsBoard REST client (v3.x / v4.x API).

Key responsibilities:
  - JWT authentication with automatic refresh (5-min buffer before expiry)
  - Asset attribute reads (SERVER_SCOPE)
  - Latest-values telemetry reads for both ASSET and DEVICE entity types
  - Timeseries history reads
  - Telemetry writes to ASSET SERVER_SCOPE
  - BFS hierarchy traversal
  - Plant discovery: BFS + isPlant==true + pvlib_enabled==true filter
"""
from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Set, Tuple

import httpx

log = logging.getLogger(__name__)


@dataclass
class PlantRef:
    """Lightweight plant descriptor returned by discover_plants()."""
    id: str
    name: str
    # All parent asset IDs in every path from root → this plant
    parent_ids: Set[str] = field(default_factory=set)


class ThingsBoardClient:
    """Async context-manager ThingsBoard client.

    Usage:
        async with ThingsBoardClient(host, username, password) as tb:
            attrs = await tb.get_asset_attributes(asset_id)
    """

    def __init__(self, host: str, username: str, password: str):
        self._host = host.rstrip("/")
        self._username = username
        self._password = password
        self._token: Optional[str] = None
        self._token_expiry: float = 0.0   # epoch seconds
        self._http: Optional[httpx.AsyncClient] = None
        self._lock = asyncio.Lock()

    async def __aenter__(self) -> "ThingsBoardClient":
        self._http = httpx.AsyncClient(timeout=30, follow_redirects=True)
        await self._ensure_token()
        return self

    async def __aexit__(self, *_):
        if self._http:
            await self._http.aclose()

    # ── Auth ────────────────────────────────────────────────────────────────

    async def _ensure_token(self) -> None:
        """Refresh JWT if missing or within 5 minutes of expiry."""
        async with self._lock:
            if self._token and time.time() < self._token_expiry - 300:
                return
            await self._login()

    async def _login(self) -> None:
        resp = await self._http.post(
            f"{self._host}/api/auth/login",
            json={"username": self._username, "password": self._password},
        )
        resp.raise_for_status()
        data = resp.json()
        self._token = data["token"]
        # TB tokens are valid for 2.5 hours by default; decode expiry from JWT
        try:
            import base64, json as _json
            payload = self._token.split(".")[1]
            # Add padding
            payload += "=" * (-len(payload) % 4)
            claims = _json.loads(base64.urlsafe_b64decode(payload))
            self._token_expiry = float(claims.get("exp", time.time() + 7200))
        except Exception:
            self._token_expiry = time.time() + 7200
        log.debug("TB login OK, token valid until %s",
                  datetime.fromtimestamp(self._token_expiry, tz=timezone.utc).isoformat())

    def _headers(self) -> dict:
        return {"X-Authorization": f"Bearer {self._token}"}

    async def _get(self, path: str, params: dict | None = None) -> Any:
        await self._ensure_token()
        url = f"{self._host}{path}"
        resp = await self._http.get(url, headers=self._headers(), params=params)
        if resp.status_code == 404:
            return None
        resp.raise_for_status()
        return resp.json()

    async def _post(self, path: str, payload: Any) -> None:
        await self._ensure_token()
        url = f"{self._host}{path}"
        resp = await self._http.post(url, headers=self._headers(), json=payload)
        resp.raise_for_status()

    # ── Attributes ──────────────────────────────────────────────────────────

    async def get_asset_attributes(self, asset_id: str, scope: str = "SERVER_SCOPE") -> dict:
        """Return a flat dict of {key: value} for an asset's attributes."""
        data = await self._get(
            f"/api/plugins/telemetry/ASSET/{asset_id}/values/attributes/{scope}"
        )
        if not data:
            return {}
        return {entry["key"]: entry["value"] for entry in data}

    async def get_entity_info(self, entity_id: str, entity_type: str = "ASSET") -> Optional[dict]:
        """Return basic entity info (name, assetProfileName, etc.)."""
        if entity_type == "ASSET":
            return await self._get(f"/api/asset/{entity_id}")
        elif entity_type == "DEVICE":
            return await self._get(f"/api/device/{entity_id}")
        return None

    # ── Telemetry reads ─────────────────────────────────────────────────────

    async def get_latest_telemetry(
        self,
        entity_type: str,
        entity_id: str,
        keys: list[str],
    ) -> dict:
        """Return {key: latest_value} for an ASSET or DEVICE."""
        params = {"keys": ",".join(keys)}
        data = await self._get(
            f"/api/plugins/telemetry/{entity_type}/{entity_id}/values/timeseries",
            params=params,
        )
        if not data:
            return {}
        result = {}
        for key, records in data.items():
            if records:
                result[key] = records[0]["value"]
        return result

    async def get_timeseries(
        self,
        entity_type: str,
        entity_id: str,
        keys: list[str],
        start: datetime,
        end: datetime,
        limit: int = 50_000,
        agg: str = "NONE",
    ) -> dict:
        """Return {key: [{ts, value}, ...]} for a time range."""
        start_ms = int(start.timestamp() * 1000)
        end_ms = int(end.timestamp() * 1000)
        params = {
            "keys": ",".join(keys),
            "startTs": start_ms,
            "endTs": end_ms,
            "limit": limit,
            "agg": agg,
            "orderBy": "ASC",
        }
        data = await self._get(
            f"/api/plugins/telemetry/{entity_type}/{entity_id}/values/timeseries",
            params=params,
        )
        return data or {}

    # ── Telemetry writes ────────────────────────────────────────────────────

    async def post_telemetry(
        self,
        entity_type: str,
        entity_id: str,
        records: list[dict],
    ) -> None:
        """Write telemetry records.

        records format: [{"ts": epoch_ms, "values": {"key1": v1, "key2": v2}}, ...]
        """
        # TB accepts batches up to ~1000 records; chunk if needed
        chunk_size = 500
        for i in range(0, len(records), chunk_size):
            chunk = records[i:i + chunk_size]
            await self._post(
                f"/api/plugins/telemetry/{entity_type}/{entity_id}/timeseries/SERVER_SCOPE",
                {"telemetry": chunk},
            )

    # ── Relations / hierarchy ───────────────────────────────────────────────

    async def get_child_relations(self, asset_id: str) -> list[dict]:
        """Return list of child entity descriptors via 'Contains' relations."""
        data = await self._get(
            "/api/relations",
            params={
                "fromId": asset_id,
                "fromType": "ASSET",
                "relationType": "Contains",
            },
        )
        if not data:
            return []
        return data if isinstance(data, list) else []

    # ── Plant discovery ─────────────────────────────────────────────────────

    async def discover_plants(
        self,
        root_asset_ids: list[str],
    ) -> Tuple[List[PlantRef], Dict[str, Set[str]]]:
        """BFS from root assets to find all pvlib-enabled plants.

        Returns
        -------
        plants : List[PlantRef]
            De-duplicated list of plants where isPlant==true AND pvlib_enabled==true.
        parent_map : Dict[plant_id, Set[parent_id]]
            All direct parent asset IDs of each plant (for roll-up dedup).
        """
        visited: Set[str] = set()
        plant_map: Dict[str, PlantRef] = {}   # id → PlantRef (dedup)
        parent_map: Dict[str, Set[str]] = {}  # plant_id → set of parent ids
        queue: List[Tuple[str, Optional[str]]] = [(rid, None) for rid in root_asset_ids]

        while queue:
            batch, queue = queue[:20], queue[20:]
            await asyncio.gather(*(
                self._visit_node(
                    asset_id, parent_id,
                    visited, plant_map, parent_map, queue,
                )
                for asset_id, parent_id in batch
            ), return_exceptions=True)

        log.info("discover_plants: found %d enabled plants from %d roots",
                 len(plant_map), len(root_asset_ids))
        return list(plant_map.values()), parent_map

    async def _visit_node(
        self,
        asset_id: str,
        parent_id: Optional[str],
        visited: Set[str],
        plant_map: Dict[str, PlantRef],
        parent_map: Dict[str, Set[str]],
        queue: list,
    ) -> None:
        if asset_id in visited:
            # Still record the parent path for already-discovered plants
            if asset_id in plant_map and parent_id:
                plant_map[asset_id].parent_ids.add(parent_id)
                parent_map.setdefault(asset_id, set()).add(parent_id)
            return
        visited.add(asset_id)

        # Fetch attributes and entity info concurrently
        attrs_task = asyncio.create_task(self.get_asset_attributes(asset_id))
        info_task = asyncio.create_task(self.get_entity_info(asset_id))
        attrs, info = await asyncio.gather(attrs_task, info_task, return_exceptions=True)
        if isinstance(attrs, Exception):
            attrs = {}
        if isinstance(info, Exception):
            info = {}

        info = info or {}
        attrs = attrs or {}

        is_plant = _truthy(attrs.get("isPlant"))
        pvlib_enabled = _truthy(attrs.get("pvlib_enabled"))

        if is_plant:
            if pvlib_enabled:
                ref = plant_map.setdefault(asset_id, PlantRef(
                    id=asset_id,
                    name=info.get("name", attrs.get("name", asset_id)),
                ))
                if parent_id:
                    ref.parent_ids.add(parent_id)
                    parent_map.setdefault(asset_id, set()).add(parent_id)
                log.debug("discover_plants: plant %s (%s) enabled", asset_id, ref.name)
            else:
                log.debug("discover_plants: plant %s skipped (pvlib_enabled=false/missing)", asset_id)
            # Don't traverse children of a plant (plants are leaves in the physical hierarchy)
            return

        # Non-plant asset → BFS into children
        try:
            children = await self.get_child_relations(asset_id)
        except Exception as exc:
            log.warning("discover_plants: failed to get children of %s: %s", asset_id, exc)
            return

        for rel in children:
            to_id = rel.get("to", {}).get("id") or rel.get("toId") or ""
            to_type = rel.get("to", {}).get("entityType") or rel.get("toEntityType", "ASSET")
            if to_id and to_type == "ASSET":
                queue.append((to_id, asset_id))

    # ── Legacy compatibility ────────────────────────────────────────────────

    async def get_hierarchy_levels(self, root_asset_id: str, target_level: int = 3) -> list:
        """Depth-limited BFS for backward compatibility with start-with-date jobs."""
        result = []
        queue = [(root_asset_id, 0)]
        visited: Set[str] = set()
        while queue:
            current_id, depth = queue.pop(0)
            if current_id in visited:
                continue
            visited.add(current_id)
            if depth == target_level:
                result.append(current_id)
                continue
            children = await self.get_child_relations(current_id)
            for rel in children:
                child_id = rel.get("to", {}).get("id") or rel.get("toId", "")
                if child_id:
                    queue.append((child_id, depth + 1))
        return result


def _truthy(value) -> bool:
    """Convert TB attribute values to bool safely."""
    if value is None:
        return False
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return value != 0
    s = str(value).strip().lower()
    return s in ("true", "1", "yes", "y", "on")
