"""
set_active_power_unit.py — Gap 9 (H9-D)

One-shot migration script: writes the SERVER_SCOPE attribute
``active_power_unit`` on every known solar-plant asset in ThingsBoard.

Widgets that co-plot ``active_power`` (meter) and ``potential_power`` (pvlib,
always kW) must normalise the meter reading when the plant publishes watts.
By setting this attribute once, widgets can apply a conditional scale without
embedding plant-specific logic.

Usage
─────
    # Basic (reads TB credentials from environment variables):
    python set_active_power_unit.py

    # Explicit TB host + credentials:
    python set_active_power_unit.py --host https://thingsboard.example.com \
                                    --user admin@tenant.com \
                                    --password <secret>

    # Dry-run (print what would be written, no API calls):
    python set_active_power_unit.py --dry-run

    # Override for a single plant by name (partial match):
    python set_active_power_unit.py --override "AKB Kelaniya=kW"

Environment variables (precedence: CLI > env > defaults)
─────────────────────────────────────────────────────────
    TB_HOST      ThingsBoard base URL (e.g. https://tb.example.com)
    TB_USERNAME  Tenant admin email
    TB_PASSWORD  Tenant admin password

Plant → unit mapping (from Opus §2.9 + user asset hierarchy notes)
────────────────────────────────────────────────────────────────────
kW plants: active_power already in kilowatts — no widget scaling needed.
W  plants: active_power published in watts  — widgets apply ×0.001 scale.

The mapping uses plant *name substrings* to handle minor name variations
(e.g. "KSP Plant" vs "KSP_Plant").  Exact-match is attempted first; if no
exact hit, the first asset whose name contains the key string is used.
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from typing import Dict, Optional

import requests

log = logging.getLogger("set_active_power_unit")

# ── Plant → unit mapping ────────────────────────────────────────────────────
# Key = substring that uniquely identifies the plant name in ThingsBoard.
# Value = "kW" | "W"
# Source: Opus 4.7 analysis §2.9 + operator-supplied hierarchy.
PLANT_UNIT_MAP: Dict[str, str] = {
    # ── kW plants (no scaling needed) ───────────────────────────────────────
    "KSP":             "kW",
    "SSK":             "kW",
    "SOU":             "kW",
    "PSP":             "kW",
    "VPE Plant1":      "kW",
    "VPE Plant2":      "kW",
    "SON":             "kW",
    "SER":             "kW",
    "SUN":             "kW",
    "VYD":             "kW",
    "AKB Welisara 1":  "kW",
    "AKB Welisara 2":  "kW",
    "Aerosense":       "kW",
    "Mouldex 1":       "kW",
    "Mouldex 2":       "kW",
    # ── W plants (active_power in watts — widget must scale ×0.001) ─────────
    "AKB Kelaniya":        "W",
    "AKB Exports Mabola":  "W",
    "Chris Logix 1":       "W",
    "Chris Logix 2":       "W",
    "Lina Manufacturing":  "W",
    "Quick Tea":           "W",
    "Harness":             "W",
    "Flinth Admin":        "W",
    "Mona Rathmalana":     "W",
    "Mona Homagama":       "W",
    "Mona Koggala":        "W",
    "Hir Agalawaththa":    "W",
    "Hir Kahatuduwa 1":    "W",
    "Hir Kahatuduwa 2":    "W",
    "Hir Kuruvita":        "W",
    "Hir Mullaitivu":      "W",
    "Hir Eheliyagoda":     "W",
    "Hir Seethawaka 1":    "W",
    "Hir Seethawaka 2":    "W",
    "Hir Maharagama 1":    "W",
    "Hir Vavuniya":        "W",
}

ATTRIBUTE_KEY = "active_power_unit"


# ── ThingsBoard REST client (minimal — no async needed for a one-shot script) ─

class TBClient:
    def __init__(self, host: str, username: str, password: str) -> None:
        self.base = host.rstrip("/")
        self._session = requests.Session()
        self._session.headers["Content-Type"] = "application/json"
        self._token: Optional[str] = None
        self._login(username, password)

    def _login(self, username: str, password: str) -> None:
        resp = self._session.post(
            f"{self.base}/api/auth/login",
            json={"username": username, "password": password},
            timeout=20,
        )
        resp.raise_for_status()
        self._token = resp.json()["token"]
        self._session.headers["X-Authorization"] = f"Bearer {self._token}"
        log.info("Authenticated as %s", username)

    def search_assets(self, text_search: str, page_size: int = 100) -> list[dict]:
        """Return all assets whose name matches text_search (TB prefix search)."""
        assets: list[dict] = []
        page = 0
        while True:
            resp = self._session.get(
                f"{self.base}/api/tenant/assets",
                params={
                    "textSearch": text_search,
                    "pageSize": page_size,
                    "page": page,
                    "sortProperty": "name",
                    "sortOrder": "ASC",
                },
                timeout=20,
            )
            resp.raise_for_status()
            body = resp.json()
            data = body.get("data", [])
            assets.extend(data)
            if body.get("hasNext", False):
                page += 1
            else:
                break
        return assets

    def get_server_attributes(self, entity_type: str, entity_id: str, keys: list[str]) -> dict:
        """Return {key: value} for the given SERVER_SCOPE attributes."""
        resp = self._session.get(
            f"{self.base}/api/plugins/telemetry/{entity_type}/{entity_id}/values/attributes/SERVER_SCOPE",
            params={"keys": ",".join(keys)},
            timeout=20,
        )
        resp.raise_for_status()
        return {item["key"]: item["value"] for item in resp.json()}

    def save_server_attribute(self, entity_type: str, entity_id: str, key: str, value: str) -> None:
        """Write a single SERVER_SCOPE attribute."""
        resp = self._session.post(
            f"{self.base}/api/plugins/telemetry/{entity_type}/{entity_id}/SERVER_SCOPE",
            json={key: value},
            timeout=20,
        )
        resp.raise_for_status()


# ── Resolution logic ─────────────────────────────────────────────────────────

def _find_asset(tb: TBClient, search_key: str) -> Optional[dict]:
    """
    Find an asset by name.

    Strategy:
      1. Search TB with the exact search_key string.
      2. First try exact case-insensitive name match.
      3. Fall back to first asset whose name contains the search_key substring.
    """
    candidates = tb.search_assets(search_key)
    if not candidates:
        return None
    lower_key = search_key.lower()
    # Exact match first
    for a in candidates:
        if a.get("name", "").lower() == lower_key:
            return a
    # Substring match
    for a in candidates:
        if lower_key in a.get("name", "").lower():
            return a
    return None


# ── Main ─────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--host",     default=os.getenv("TB_HOST", "http://localhost:8080"),
                   help="ThingsBoard base URL (env: TB_HOST)")
    p.add_argument("--user",     default=os.getenv("TB_USERNAME", ""),
                   help="Tenant admin email (env: TB_USERNAME)")
    p.add_argument("--password", default=os.getenv("TB_PASSWORD", ""),
                   help="Tenant admin password (env: TB_PASSWORD)")
    p.add_argument("--dry-run",  action="store_true",
                   help="Print what would be written without making any API calls")
    p.add_argument("--override", action="append", default=[], metavar="NAME=UNIT",
                   help="Override unit for a plant: 'AKB Kelaniya=kW'. Repeatable.")
    p.add_argument("--verbose",  action="store_true")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)s  %(message)s",
        stream=sys.stdout,
    )

    # Apply CLI overrides to the mapping
    plant_map = dict(PLANT_UNIT_MAP)
    for override in args.override:
        if "=" not in override:
            log.error("Bad --override format (expected NAME=UNIT): %r", override)
            return 1
        name, unit = override.split("=", 1)
        unit = unit.strip().upper()
        if unit not in ("KW", "W"):
            log.error("Unit must be kW or W, got %r", unit)
            return 1
        # TB is case-sensitive for the value — preserve user casing
        plant_map[name.strip()] = unit if unit != "KW" else "kW"

    if args.dry_run:
        log.info("DRY-RUN mode — no API calls will be made")
        for name, unit in sorted(plant_map.items()):
            log.info("  WOULD SET  %-35s  active_power_unit = %s", name, unit)
        return 0

    if not args.user or not args.password:
        log.error("TB_USERNAME and TB_PASSWORD must be set (env vars or --user / --password)")
        return 1

    tb = TBClient(args.host, args.user, args.password)

    ok = skipped = failed = not_found = 0

    for search_key, desired_unit in sorted(plant_map.items()):
        asset = _find_asset(tb, search_key)
        if asset is None:
            log.warning("NOT FOUND  %-35s — skipping", search_key)
            not_found += 1
            continue

        asset_id = asset["id"]["id"]
        asset_name = asset.get("name", search_key)

        # Idempotency check
        try:
            current = tb.get_server_attributes("ASSET", asset_id, [ATTRIBUTE_KEY])
            if current.get(ATTRIBUTE_KEY) == desired_unit:
                log.info("SKIP       %-35s  already %s", asset_name, desired_unit)
                skipped += 1
                continue
        except requests.HTTPError as exc:
            log.warning("Could not read existing attribute for %s (%s) — will overwrite", asset_name, exc)

        try:
            tb.save_server_attribute("ASSET", asset_id, ATTRIBUTE_KEY, desired_unit)
            log.info("SET        %-35s  active_power_unit = %s", asset_name, desired_unit)
            ok += 1
        except requests.HTTPError as exc:
            log.error("FAILED     %-35s  %s", asset_name, exc)
            failed += 1

    print(f"\nSummary: {ok} written, {skipped} already-correct (skipped), "
          f"{not_found} not found, {failed} failed.")

    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
