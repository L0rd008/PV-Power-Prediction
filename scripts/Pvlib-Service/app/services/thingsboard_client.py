"""
Async ThingsBoard REST client (v3.x / v4.x API).

Phase C hardening:
  - Gap 22: tenacity retry on _get/_post (3 attempts, exp-jitter backoff,
    only on 429/502/503/504 and network timeouts; never retries 401/4xx client errors)
  - Gap 23: BFS uses asyncio.Queue instead of shared list mutation
  - Gap 13: discover_plants result cached with 5-min TTL; /admin/refresh-plants invalidates

Key responsibilities:
  - JWT authentication with automatic refresh (5-min buffer before expiry)
  - Asset attribute reads (SERVER_SCOPE)
  - Latest-values + history telemetry reads
  - Telemetry writes
  - BFS hierarchy traversal
  - Plant discovery: BFS + isPlant==true + pvlib_enabled==true filter,
    building full ancestor_map (isPlantAgg ancestors)
  - Tenant device search by name prefix (H1-B fallback)
"""
from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Set, Tuple

import httpx
from tenacity import (
    retry,
    retry_if_exception,
    stop_after_attempt,
    wait_exponential_jitter,
)

log = logging.getLogger(__name__)

# ── Retry predicate ─────────────────────────────────────────────────────────

def _should_retry(exc: BaseException) -> bool:
    """Retry on network timeouts and specific HTTP status codes (Gap 22).

    Do NOT retry on:
      - 4xx client errors (our bug — fix the request)
      - 401 Unauthorized (handled separately by _ensure_token re-login)
    """
    if isinstance(exc, httpx.TimeoutException):
        return True
    if isinstance(exc, httpx.HTTPStatusError):
        return exc.response.status_code in (429, 502, 503, 504)
    return False


# ── Discover-plants cache ────────────────────────────────────────────────────

@dataclass
class _DiscoverCache:
    plants: List["PlantRef"]
    ancestor_map: Dict[str, Set[str]]
    cached_at: float       # time.monotonic()


_discover_cache: Dict[str, _DiscoverCache] = {}   # key: sorted root_ids string
_discover_cache_lock: asyncio.Lock = None          # initialised lazily

# ── Prometheus-style counters (Gap 15) — read by /metrics ───────────────────
_discover_cache_hits_total:   int = 0
_discover_cache_misses_total: int = 0


def _cache_key(root_asset_ids: list[str]) -> str:
    return ",".join(sorted(root_asset_ids))


# ── PlantRef ────────────────────────────────────────────────────────────────

@dataclass
class PlantRef:
    """Lightweight plant descriptor returned by discover_plants()."""
    id: str
    name: str
    parent_ids: Set[str] = field(default_factory=set)


# ── ThingsBoardClient ────────────────────────────────────────────────────────

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
        self._token_expiry: float = 0.0
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
        try:
            import base64, json as _json
            payload = self._token.split(".")[1]
            payload += "=" * (-len(payload) % 4)
            claims = _json.loads(base64.urlsafe_b64decode(payload))
            self._token_expiry = float(claims.get("exp", time.time() + 7200))
        except Exception:
            self._token_expiry = time.time() + 7200
        log.debug("TB login OK, token valid until %s",
                  datetime.fromtimestamp(self._token_expiry, tz=timezone.utc).isoformat())

    def _headers(self) -> dict:
        return {"X-Authorization": f"Bearer {self._token}"}

    # ── Core HTTP with retry (Gap 22) ────────────────────────────────────────

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential_jitter(initial=0.5, max=5.0),
        retry=retry_if_exception(_should_retry),
        reraise=True,
    )
    async def _get(self, path: str, params: dict | None = None) -> Any:
        await self._ensure_token()
        url = f"{self._host}{path}"
        resp = await self._http.get(url, headers=self._headers(), params=params)
        if resp.status_code == 404:
            return None
        resp.raise_for_status()
        return resp.json()

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential_jitter(initial=0.5, max=5.0),
        retry=retry_if_exception(_should_retry),
        reraise=True,
    )
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

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential_jitter(initial=0.5, max=5.0),
        retry=retry_if_exception(_should_retry),
        reraise=True,
    )
    async def post_attributes(
        self,
        entity_type: str,
        entity_id: str,
        scope: str,
        payload: dict,
    ) -> None:
        """Write SERVER_SCOPE (or any scope) attributes for an entity.

        TB REST endpoint:
          POST /api/plugins/telemetry/{entityType}/{entityId}/attributes/{scope}
          body: {"key1": value1, "key2": value2}

        Note: the write path is /attributes/{scope} (NOT /values/attributes/{scope}
        which is the read path used by get_asset_attributes).

        Used by the loss-rollup job to persist the six lifetime cumulative attributes
        and the anchor-date / updated-at tracking fields.
        """
        await self._ensure_token()
        url = f"{self._host}/api/plugins/telemetry/{entity_type}/{entity_id}/attributes/{scope}"
        resp = await self._http.post(url, headers=self._headers(), json=payload)
        resp.raise_for_status()

    async def post_telemetry(
        self,
        entity_type: str,
        entity_id: str,
        records: list[dict],
    ) -> None:
        """Write telemetry records in chunks of 500.

        TB REST contract for timeseries writes:
          POST /api/plugins/telemetry/{entityType}/{entityId}/timeseries/ANY
          body: [{"ts": <ms>, "values": {<key>: <val>, ...}}, ...]

        The array is posted directly — do NOT wrap it as {"telemetry": [...]}.
        Wrapping causes TB to store the entire payload under a single key
        literally named "telemetry" with the stringified array as its value,
        breaking all downstream widget queries. This matches the Solcast +
        Simple Forecast reference clients.
        """
        chunk_size = 500
        for i in range(0, len(records), chunk_size):
            chunk = records[i:i + chunk_size]
            await self._post(
                f"/api/plugins/telemetry/{entity_type}/{entity_id}/timeseries/ANY",
                chunk,
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

    # ── Device search (H1-B, Gap 1) ─────────────────────────────────────────

    async def search_devices_by_name_prefix(
        self,
        prefix: str,
        limit: int = 50,
    ) -> List[Dict[str, Any]]:
        """Search tenant devices whose name contains the given prefix string.

        Handles both TB v3.x (dict with 'data') and v4.x (raw list) response shapes.
        """
        data = await self._get(
            "/api/tenant/devices",
            params={"textSearch": prefix, "pageSize": limit, "page": 0},
        )
        if not data:
            return []
        if isinstance(data, dict) and "data" in data:
            return data["data"]
        if isinstance(data, list):
            return data
        return []

    # ── Plant discovery with TTL cache (Gap 13) ─────────────────────────────

    async def discover_plants(
        self,
        root_asset_ids: list[str],
        ttl_seconds: int = 300,
        force: bool = False,
    ) -> Tuple[List[PlantRef], Dict[str, Set[str]]]:
        """BFS from root assets to find all pvlib-enabled plants.

        Results are cached for ttl_seconds (default 5 min).
        Pass force=True to invalidate the cache immediately (used by /admin/refresh-plants).

        Returns
        -------
        plants : List[PlantRef]
            De-duplicated list of plants where isPlant==true AND pvlib_enabled==true.
        ancestor_map : Dict[plant_id, Set[ancestor_id]]
            All isPlantAgg ancestor asset IDs for each plant.
        """
        global _discover_cache_lock
        if _discover_cache_lock is None:
            _discover_cache_lock = asyncio.Lock()

        key = _cache_key(root_asset_ids)

        global _discover_cache_hits_total, _discover_cache_misses_total

        async with _discover_cache_lock:
            cached = _discover_cache.get(key)
            now = time.monotonic()
            if not force and cached and (now - cached.cached_at) < ttl_seconds:
                _discover_cache_hits_total += 1
                log.debug("discover_plants: cache hit (age=%.0fs)", now - cached.cached_at)
                return cached.plants, cached.ancestor_map

        # Cache miss — run BFS outside the lock so concurrent callers don't deadlock
        _discover_cache_misses_total += 1
        plants, ancestor_map = await self._bfs_discover(root_asset_ids)

        async with _discover_cache_lock:
            _discover_cache[key] = _DiscoverCache(
                plants=plants,
                ancestor_map=ancestor_map,
                cached_at=time.monotonic(),
            )

        log.info("discover_plants: found %d enabled plants from %d roots",
                 len(plants), len(root_asset_ids))
        return plants, ancestor_map

    async def _bfs_discover(
        self,
        root_asset_ids: list[str],
    ) -> Tuple[List[PlantRef], Dict[str, Set[str]]]:
        """Inner BFS using asyncio.Queue to avoid shared-list mutation (Gap 23)."""
        visited: Set[str] = set()
        plant_map: Dict[str, PlantRef] = {}
        ancestor_map: Dict[str, Set[str]] = {}

        # Queue items: (asset_id, direct_parent_id, path_ancestors_tuple)
        q: asyncio.Queue[Tuple[str, Optional[str], Tuple[str, ...]]] = asyncio.Queue()
        for rid in root_asset_ids:
            await q.put((rid, None, ()))

        # Process in batches of up to 20 concurrent visits
        while not q.empty():
            batch: List[Tuple[str, Optional[str], Tuple[str, ...]]] = []
            while not q.empty() and len(batch) < 20:
                batch.append(await q.get())

            await asyncio.gather(*(
                self._visit_node(
                    asset_id, parent_id, path_ancestors,
                    visited, plant_map, ancestor_map, q,
                )
                for asset_id, parent_id, path_ancestors in batch
            ), return_exceptions=True)

            # Mark tasks done
            for _ in batch:
                q.task_done()

        return list(plant_map.values()), ancestor_map

    async def _visit_node(
        self,
        asset_id: str,
        parent_id: Optional[str],
        path_ancestors: Tuple[str, ...],
        visited: Set[str],
        plant_map: Dict[str, PlantRef],
        ancestor_map: Dict[str, Set[str]],
        q: asyncio.Queue,
    ) -> None:
        if asset_id in visited:
            if asset_id in plant_map:
                if parent_id:
                    plant_map[asset_id].parent_ids.add(parent_id)
                ancestor_map.setdefault(asset_id, set()).update(path_ancestors)
            return
        visited.add(asset_id)

        attrs_task = asyncio.create_task(self.get_asset_attributes(asset_id))
        info_task = asyncio.create_task(self.get_entity_info(asset_id))
        attrs, info = await asyncio.gather(attrs_task, info_task, return_exceptions=True)
        if isinstance(attrs, Exception):
            attrs = {}
        if isinstance(info, Exception):
            info = {}
        info = info or {}
        attrs = attrs or {}

        is_plant     = _truthy(attrs.get("isPlant"))
        pvlib_enabled = _truthy(attrs.get("pvlib_enabled"))
        is_plant_agg = _truthy(attrs.get("isPlantAgg"))

        if is_plant:
            if pvlib_enabled:
                ref = plant_map.setdefault(asset_id, PlantRef(
                    id=asset_id,
                    name=info.get("name", attrs.get("name", asset_id)),
                ))
                if parent_id:
                    ref.parent_ids.add(parent_id)
                ancestor_map.setdefault(asset_id, set()).update(path_ancestors)
                if is_plant_agg:
                    log.warning(
                        "discover_plants: asset %s (%s) has both isPlant=true and "
                        "isPlantAgg=true — treating as leaf (Edge E14).",
                        asset_id, ref.name,
                    )
                log.debug("discover_plants: plant %s (%s) enabled, ancestors=%s",
                          asset_id, ref.name, path_ancestors)
            else:
                log.debug("discover_plants: plant %s skipped (pvlib_enabled=false/missing)", asset_id)
            return   # plants are leaves

        new_path: Tuple[str, ...] = (
            path_ancestors + (asset_id,) if is_plant_agg else path_ancestors
        )

        try:
            children = await self.get_child_relations(asset_id)
        except Exception as exc:
            log.warning("discover_plants: failed to get children of %s: %s", asset_id, exc)
            return

        for rel in children:
            to_id   = rel.get("to", {}).get("id") or rel.get("toId") or ""
            to_type = rel.get("to", {}).get("entityType") or rel.get("toEntityType", "ASSET")
            if to_id and to_type == "ASSET":
                await q.put((to_id, asset_id, new_path))

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
