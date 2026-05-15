"""
find_orphan_plants.py — Detect pvlib-enabled plants not reachable from any
configured root asset.

A plant is an "orphan" when it has SERVER_SCOPE attributes isPlant=true AND
pvlib_enabled=true but does NOT appear in the Pvlib-Service discovery response
(/pvlib/discover). This means the service silently ignores it — no potential_power,
no actual_daily_energy_kwh, no loss attribution.

Typical causes:
  - Plant asset exists but is not connected to any root via a "Contains" relation.
  - Plant is under a root not listed in TB_ROOT_ASSET_IDS.
  - pvlib_enabled was set before the asset was added to the hierarchy.

Fix: connect the orphaned asset to its parent via a "Contains" relation in
ThingsBoard Asset Groups or via the TB REST API.

USAGE
-----
  python find_orphan_plants.py --pvlib-host http://localhost:8004 --format table

EXIT CODES
----------
  0  No orphans found
  1  Orphans detected
  2  Connection / auth failure

ENVIRONMENT VARIABLES
---------------------
  TB_HOST       ThingsBoard base URL (default http://localhost:8080)
  TB_USERNAME   Tenant admin email
  TB_PASSWORD   Tenant admin password
  PVLIB_HOST    Pvlib-Service base URL (default http://localhost:8004)
"""
from __future__ import annotations

import argparse
import csv
import io
import json
import logging
import os
import sys
from typing import Any, Dict, List, Optional

import requests

log = logging.getLogger("find_orphan_plants")


# ── ThingsBoard REST client (sync / minimal) ──────────────────────────────────

class TBClient:
    PAGE_SIZE = 100

    def __init__(self, host: str, username: str, password: str) -> None:
        self.base = host.rstrip("/")
        self._s = requests.Session()
        self._s.headers["Content-Type"] = "application/json"
        resp = self._s.post(f"{self.base}/api/auth/login",
                            json={"username": username, "password": password},
                            timeout=20)
        resp.raise_for_status()
        self._s.headers["X-Authorization"] = f"Bearer {resp.json()['token']}"

    def _get(self, path: str, params=None) -> Any:
        resp = self._s.get(f"{self.base}{path}", params=params, timeout=30)
        if resp.status_code == 404:
            return None
        resp.raise_for_status()
        return resp.json()

    def tenant_assets_pvlib_enabled(self) -> List[Dict]:
        """Paginate all tenant assets; return those with isPlant=true && pvlib_enabled=true."""
        pvlib_plants = []
        page = 0
        while True:
            data = self._get("/api/tenant/assets",
                             params={"pageSize": self.PAGE_SIZE, "page": page})
            if not data:
                break
            items = data.get("data", [])
            for asset in items:
                asset_id = asset.get("id", {}).get("id", "")
                if not asset_id:
                    continue
                attrs = self._server_attrs(asset_id)
                is_plant   = _truthy(attrs.get("isPlant", False))
                pvlib_on   = _truthy(attrs.get("pvlib_enabled", False))
                if is_plant and pvlib_on:
                    pvlib_plants.append({
                        "id":   asset_id,
                        "name": asset.get("name", asset_id),
                        "attrs": attrs,
                    })
            if data.get("hasNext", False):
                page += 1
            else:
                break
        return pvlib_plants

    def _server_attrs(self, asset_id: str) -> dict:
        data = self._get(
            f"/api/plugins/telemetry/ASSET/{asset_id}/values/attributes/SERVER_SCOPE"
        )
        if not data:
            return {}
        return {item["key"]: item["value"] for item in data}


def _truthy(v) -> bool:
    if isinstance(v, bool): return v
    if isinstance(v, (int, float)): return v != 0
    if isinstance(v, str): return v.strip().lower() in ("true", "yes", "1")
    return bool(v)


def get_discovered_ids(pvlib_base: str) -> set:
    """Call /pvlib/discover and return the set of discovered plant asset IDs."""
    try:
        resp = requests.get(f"{pvlib_base.rstrip('/')}/pvlib/discover", timeout=30)
        resp.raise_for_status()
        data = resp.json()
        plants = data.get("plants", data) if isinstance(data, dict) else data
        if isinstance(plants, list):
            ids = set()
            for p in plants:
                pid = p.get("id") or p.get("asset_id") or ""
                if pid:
                    ids.add(pid)
            return ids
        return set()
    except requests.RequestException as exc:
        log.warning("Could not reach pvlib-service /pvlib/discover: %s", exc)
        return set()


# ── Formatters ────────────────────────────────────────────────────────────────

def _fmt_table(orphans: List[dict]) -> str:
    if not orphans:
        return "No orphan plants found.\n"
    lines = [f"{'─'*72}", f"  {'NAME':<35} {'ASSET_ID'}", f"{'─'*72}"]
    for o in orphans:
        lines.append(f"  {o['name']:<35} {o['id']}")
    lines.append(f"{'─'*72}")
    lines.append(f"  {len(orphans)} orphan(s) found")
    return "\n".join(lines) + "\n"


def _fmt_json(orphans: List[dict]) -> str:
    return json.dumps(
        {"orphans": [{"id": o["id"], "name": o["name"]} for o in orphans],
         "count": len(orphans)},
        indent=2,
    )


def _fmt_csv(orphans: List[dict]) -> str:
    buf = io.StringIO()
    w = csv.writer(buf)
    w.writerow(["asset_id", "name"])
    for o in orphans:
        w.writerow([o["id"], o["name"]])
    return buf.getvalue()


# ── CLI ───────────────────────────────────────────────────────────────────────

def main() -> int:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--host",       default=os.getenv("TB_HOST", "http://localhost:8080"))
    p.add_argument("--user",       default=os.getenv("TB_USERNAME", ""))
    p.add_argument("--password",   default=os.getenv("TB_PASSWORD", ""))
    p.add_argument("--pvlib-host", default=os.getenv("PVLIB_HOST", "http://localhost:8004"))
    p.add_argument("--format", choices=["table", "json", "csv"], default="table")
    p.add_argument("--output", default="", help="Write output to file instead of stdout")
    args = p.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s",
                        stream=sys.stderr)

    if not args.user or not args.password:
        log.error("TB_USERNAME and TB_PASSWORD must be set")
        return 2

    try:
        tb = TBClient(args.host, args.user, args.password)
        log.info("Connected to ThingsBoard at %s", args.host)
    except requests.HTTPError as exc:
        log.error("Authentication failed: %s", exc)
        return 2
    except requests.ConnectionError as exc:
        log.error("Cannot reach ThingsBoard: %s", exc)
        return 2

    log.info("Fetching all tenant pvlib-enabled plants (paginated)…")
    all_pvlib = tb.tenant_assets_pvlib_enabled()
    log.info("Found %d pvlib-enabled plant(s) in tenant", len(all_pvlib))

    log.info("Fetching discovered plant IDs from /pvlib/discover at %s…", args.pvlib_host)
    discovered = get_discovered_ids(args.pvlib_host)
    log.info("Pvlib-service has discovered %d plant(s)", len(discovered))

    orphans = [p for p in all_pvlib if p["id"] not in discovered]

    if args.format == "json":
        output = _fmt_json(orphans)
    elif args.format == "csv":
        output = _fmt_csv(orphans)
    else:
        output = _fmt_table(orphans)

    if args.output:
        with open(args.output, "w", encoding="utf-8") as fh:
            fh.write(output)
        log.info("Output written to %s", args.output)
    else:
        print(output, end="")

    return 1 if orphans else 0


if __name__ == "__main__":
    sys.exit(main())
