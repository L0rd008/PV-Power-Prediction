"""
tb_config_loader.py — Bulk plant configuration loader for ThingsBoard.

Reads scripts/shared/plants_master.yml (the single source of truth), builds the
pvlib_config JSON blob + flat SERVER_SCOPE companion attributes for each plant, and
writes them to ThingsBoard via REST.

Idempotency:
  Each plant's pvlib_config blob is SHA-1-hashed (first 12 hex chars); the hash is
  stored as pvlib_config_hash SERVER_SCOPE.  On subsequent runs the script compares
  master-computed hash vs TB-stored hash and skips unchanged plants unless
  --force-overwrite is set.

USAGE
-----
  # Dry-run all plants (preview without writing):
  python tb_config_loader.py --dry-run

  # Apply all plants:
  python tb_config_loader.py

  # Single plant:
  python tb_config_loader.py --plant 0e4b4070-50ff-11ef-b4ce-d5aee9e495ad

  # Show diff of what would change without writing:
  python tb_config_loader.py --diff-only

  # Force overwrite even if hash matches:
  python tb_config_loader.py --force-overwrite

ENVIRONMENT VARIABLES
---------------------
  TB_HOST           ThingsBoard base URL (default http://localhost:8080)
  TB_USERNAME       Tenant admin email
  TB_PASSWORD       Tenant admin password
  PLANTS_MASTER     Path to plants_master.yml (default: <this_dir>/plants_master.yml)
  TEMPLATES_DIR     Path to templates/ directory (default: <this_dir>/templates/)
"""
from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import requests

try:
    import yaml
except ImportError:
    print("ERROR: PyYAML is required — pip install pyyaml --break-system-packages", file=sys.stderr)
    sys.exit(1)

log = logging.getLogger("tb_config_loader")

_HERE = Path(__file__).parent

# ── Template loading ──────────────────────────────────────────────────────────

def _load_templates(templates_dir: Path) -> dict:
    """Load all YAML files in templates_dir into a dict keyed by stem name."""
    tmpl = {}
    if not templates_dir.exists():
        return tmpl
    for f in templates_dir.glob("*.yml"):
        with f.open(encoding="utf-8") as fh:
            data = yaml.safe_load(fh)
        tmpl[f.stem] = data
    return tmpl


# ── Config builder ────────────────────────────────────────────────────────────

def _resolve(templates: dict, key: Optional[str], category: str) -> Optional[dict]:
    """Return the resolved template dict for a given key, or None."""
    if not key:
        return None
    full_key = f"{category}_{key}" if not key.startswith(category) else key
    return templates.get(full_key) or templates.get(key)


def build_pvlib_config(plant: dict, templates: dict) -> dict:
    """Assemble the pvlib_config JSON blob from a plant master entry.

    This is the same shape as kebithigollewa_pvlib_config.json so the existing
    PlantConfig._from_blob parser can deserialise it without changes.
    """
    cfg: Dict[str, Any] = {}

    # Location
    cfg["location"] = {
        "lat":       plant["latitude"],
        "lon":       plant["longitude"],
        "altitude_m": plant.get("altitude_m", 0),
        "timezone":  plant.get("timezone", "Asia/Colombo"),
    }

    # Capacity
    unit = plant.get("capacity_unit", "kW")
    kwp  = plant["capacity_kwp"]
    cfg["capacity_kw"] = kwp * 1000 if unit == "MW" else kwp

    # Orientations
    cfg["orientations"] = plant["orientations"]

    # Module
    module = _resolve(templates, plant.get("module_template"), "module")
    if module:
        cfg["module"] = module
    else:
        log.warning("No module template resolved for '%s'", plant.get("module_template"))

    # Inverter
    inverter = _resolve(templates, plant.get("inverter_template"), "inverter")
    if inverter:
        cfg["inverter"] = inverter
    else:
        log.warning("No inverter template resolved for '%s'", plant.get("inverter_template"))

    # IAM
    iam = _resolve(templates, plant.get("iam_template"), "iam")
    if iam:
        cfg["iam"] = iam

    # Station (weather data source)
    station = plant.get("station")
    if station:
        cfg["station"] = station

    # Losses
    losses = plant.get("losses", {})
    for k in ("soiling", "lid", "module_quality", "mismatch", "dc_wiring",
              "ac_wiring", "far_shading", "albedo"):
        if k in losses:
            cfg[k] = losses[k]

    # Thermal model
    if plant.get("thermal_model"):
        cfg["thermal_model"] = plant["thermal_model"]

    # Defaults fallback
    cfg.setdefault("defaults", {"wind_speed_ms": 1.0, "air_temp_c": 27.96})

    return cfg


def build_flat_attrs(plant: dict, pvlib_config: dict, config_hash: str) -> dict:
    """Build the flat SERVER_SCOPE attributes to write alongside pvlib_config."""
    attrs: Dict[str, Any] = {
        # Core pvlib config blob + hash
        "pvlib_config":      json.dumps(pvlib_config, separators=(",", ":")),
        "pvlib_config_hash": config_hash,

        # Flat companions that widgets / audit / service reads directly
        "isPlant":           True,
        "pvlib_enabled":     plant.get("pvlib_enabled", True),
        "loss_attribution_enabled": plant.get("loss_attribution_enabled", True),

        "latitude":     plant["latitude"],
        "longitude":    plant["longitude"],
        "altitude_m":   plant.get("altitude_m", 0),
        "timezone":     plant.get("timezone", "Asia/Colombo"),

        "Capacity":      plant["capacity_kwp"],
        "capacityUnit":  plant.get("capacity_unit", "kW"),

        "active_power_unit": plant["active_power_unit"],
        "actual_power_keys": plant["actual_power_keys"],
    }

    if plant.get("commissioning_date"):
        attrs["commissioning_date"] = plant["commissioning_date"]
    if plant.get("tariff_rate_lkr") is not None:
        attrs["tariff_rate_lkr"] = plant["tariff_rate_lkr"]
    if plant.get("setpoint_keys"):
        attrs["setpoint_keys"] = plant["setpoint_keys"]
    if plant.get("solcast_resource_id"):
        attrs["solcast_resource_id"] = plant["solcast_resource_id"]

    # Weather station ID (top-level shorthand if station block present)
    station = plant.get("station", {})
    if station.get("weather_station_id"):
        attrs["weather_station_id"] = station["weather_station_id"]

    return attrs


def config_hash(pvlib_config: dict) -> str:
    """Thin wrapper that delegates to the shared _config_hash helper.

    Both tb_config_loader.py and find_config_drift.py call this so the
    algorithm is guaranteed identical on both sides.
    """
    from _config_hash import compute_hash
    return compute_hash(pvlib_config)


# ── ThingsBoard REST client ───────────────────────────────────────────────────

class TBClient:
    def __init__(self, host: str, username: str, password: str) -> None:
        self.base = host.rstrip("/")
        self._s = requests.Session()
        self._s.headers["Content-Type"] = "application/json"
        self._login(username, password)

    def _login(self, username: str, password: str) -> None:
        resp = self._s.post(f"{self.base}/api/auth/login",
                            json={"username": username, "password": password},
                            timeout=20)
        resp.raise_for_status()
        self._s.headers["X-Authorization"] = f"Bearer {resp.json()['token']}"

    def get_server_attributes(self, asset_id: str) -> dict:
        resp = self._s.get(
            f"{self.base}/api/plugins/telemetry/ASSET/{asset_id}/values/attributes/SERVER_SCOPE",
            timeout=20,
        )
        if resp.status_code == 404:
            return {}
        resp.raise_for_status()
        return {item["key"]: item["value"] for item in (resp.json() or [])}

    def post_server_attributes(self, asset_id: str, attrs: dict) -> None:
        resp = self._s.post(
            f"{self.base}/api/plugins/telemetry/ASSET/{asset_id}/attributes/SERVER_SCOPE",
            json=attrs,
            timeout=30,
        )
        resp.raise_for_status()


# ── Diff helper ───────────────────────────────────────────────────────────────

def _diff_attrs(old: dict, new: dict) -> List[str]:
    """Return human-readable lines describing changes from old to new."""
    lines = []
    all_keys = set(old) | set(new)
    for k in sorted(all_keys):
        ov = old.get(k, "<absent>")
        nv = new.get(k, "<absent>")
        if k == "pvlib_config":
            # Don't dump huge blobs in diff; just note if hash changed
            continue
        if str(ov) != str(nv):
            ov_repr = repr(ov)[:80] if not isinstance(ov, str) else ov[:80]
            nv_repr = repr(nv)[:80] if not isinstance(nv, str) else nv[:80]
            lines.append(f"  ~ {k}: {ov_repr!r} → {nv_repr!r}")
    return lines


# ── Main logic ────────────────────────────────────────────────────────────────

def load_master(master_path: Path) -> List[dict]:
    with master_path.open(encoding="utf-8") as fh:
        data = yaml.safe_load(fh)
    return data.get("plants", [])


def process_plant(
    tb: TBClient,
    plant: dict,
    templates: dict,
    *,
    dry_run: bool = False,
    diff_only: bool = False,
    force_overwrite: bool = False,
) -> str:
    """Process one plant entry. Returns 'ok' | 'skip' | 'dry' | 'error'."""
    asset_id   = plant["asset_id"]
    plant_name = plant.get("name", asset_id)

    pvlib_cfg  = build_pvlib_config(plant, templates)
    master_hash = config_hash(pvlib_cfg)
    flat_attrs  = build_flat_attrs(plant, pvlib_cfg, master_hash)

    try:
        existing = tb.get_server_attributes(asset_id)
    except requests.RequestException as exc:
        log.error("[%s] could not read attrs: %s", plant_name, exc)
        return "error"

    tb_hash = existing.get("pvlib_config_hash", "")

    if not force_overwrite and tb_hash == master_hash:
        log.info("[%s] no-op (hash match: %s)", plant_name, master_hash)
        return "skip"

    if diff_only or dry_run:
        diff = _diff_attrs(existing, flat_attrs)
        action = "DRY-RUN" if dry_run else "DIFF"
        print(f"\n{'─'*60}")
        print(f"  {action}: {plant_name} ({asset_id})")
        print(f"  master_hash={master_hash}  tb_hash={tb_hash or '(none)'}")
        if diff:
            for line in diff:
                print(line)
        else:
            print("  (no flat-attr changes — pvlib_config blob may have changed)")
        if dry_run:
            return "dry"

    try:
        tb.post_server_attributes(asset_id, flat_attrs)
        log.info("[%s] written (%d attrs, hash=%s)", plant_name, len(flat_attrs), master_hash)
        return "ok"
    except requests.RequestException as exc:
        log.error("[%s] write failed: %s", plant_name, exc)
        return "error"


def main() -> int:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--host",      default=os.getenv("TB_HOST", "http://localhost:8080"))
    p.add_argument("--user",      default=os.getenv("TB_USERNAME", ""))
    p.add_argument("--password",  default=os.getenv("TB_PASSWORD", ""))
    p.add_argument("--master",    default=os.getenv("PLANTS_MASTER",
                                  str(_HERE / "plants_master.yml")),
                   help="Path to plants_master.yml")
    p.add_argument("--templates", default=os.getenv("TEMPLATES_DIR",
                                  str(_HERE / "templates")),
                   help="Path to templates/ directory")
    p.add_argument("--plant",     default="",
                   help="Process only this asset UUID (skips all others)")
    p.add_argument("--dry-run",   action="store_true",
                   help="Preview changes without writing to ThingsBoard")
    p.add_argument("--diff-only", action="store_true",
                   help="Show a diff of every plant; do not write")
    p.add_argument("--force-overwrite", action="store_true",
                   help="Write even when the hash already matches")
    args = p.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s",
                        stream=sys.stderr)

    if not args.user or not args.password:
        log.error("TB_USERNAME and TB_PASSWORD must be set")
        return 2

    master_path    = Path(args.master)
    templates_dir  = Path(args.templates)

    if not master_path.exists():
        log.error("plants_master.yml not found: %s", master_path)
        return 2

    plants    = load_master(master_path)
    templates = _load_templates(templates_dir)

    if args.plant:
        plants = [p for p in plants if p.get("asset_id") == args.plant.strip()]
        if not plants:
            log.error("Plant %s not found in master file", args.plant)
            return 2

    try:
        tb = TBClient(args.host, args.user, args.password)
        log.info("Connected to %s", args.host)
    except requests.HTTPError as exc:
        log.error("Authentication failed: %s", exc)
        return 2
    except requests.ConnectionError as exc:
        log.error("Cannot reach ThingsBoard at %s: %s", args.host, exc)
        return 2

    counts = {"ok": 0, "skip": 0, "dry": 0, "error": 0}
    for plant in plants:
        result = process_plant(
            tb, plant, templates,
            dry_run=args.dry_run,
            diff_only=args.diff_only,
            force_overwrite=args.force_overwrite,
        )
        counts[result] = counts.get(result, 0) + 1

    print(f"\n{'='*60}")
    print(f"  {len(plants)} plant(s)  —  "
          f"ok={counts['ok']}  skip={counts['skip']}  "
          f"dry={counts['dry']}  error={counts['error']}")

    return 1 if counts["error"] > 0 else 0


if __name__ == "__main__":
    sys.exit(main())
