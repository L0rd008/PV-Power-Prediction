"""
find_config_drift.py — Detect ThingsBoard plants whose pvlib_config has drifted
from the source-of-truth in plants_master.yml.

The Pvlib-Service writes pvlib_config_hash as a SERVER_SCOPE attribute each time
tb_config_loader.py applies a plant config. If an operator manually edits TB
attributes after that, or applies a YAML change without re-running the loader,
the TB-side hash will diverge from what the master file would produce.

This script:
  1. Loads plants_master.yml.
  2. Recomputes each plant's pvlib_config hash via the shared _config_hash helper.
  3. Fetches the current pvlib_config_hash from TB SERVER_SCOPE.
  4. Reports plants where the two hashes differ.

FLAGS
-----
  --report-only   Print drift table, exit 1 on any drift (default)
  --fix           Rewrite diverging plants via tb_config_loader logic, then exit 0

EXIT CODES
----------
  0  No drift (or --fix applied all diverging plants successfully)
  1  Drift detected (in --report-only mode) or fix failed for some plant
  2  Connection / auth failure

USAGE
-----
  python find_config_drift.py --master scripts/shared/plants_master.yml

ENVIRONMENT VARIABLES
---------------------
  TB_HOST       ThingsBoard base URL (default http://localhost:8080)
  TB_USERNAME   Tenant admin email
  TB_PASSWORD   Tenant admin password
  PLANTS_MASTER Path to plants_master.yml
  TEMPLATES_DIR Path to templates/ directory
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from pathlib import Path
from typing import Dict, List

import requests

try:
    import yaml
except ImportError:
    print("ERROR: PyYAML required — pip install pyyaml --break-system-packages", file=sys.stderr)
    sys.exit(1)

# Ensure scripts/shared is on sys.path so relative imports work when run as a script
_HERE = Path(__file__).parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

from _config_hash import compute_hash
from tb_config_loader import (
    TBClient, build_pvlib_config, build_flat_attrs, load_master, _load_templates, process_plant,
)

log = logging.getLogger("find_config_drift")


# ── Drift detection ───────────────────────────────────────────────────────────

def detect_drift(tb: TBClient, plants: list, templates: dict) -> List[dict]:
    """Return list of drift dicts for plants whose hashes diverge from master."""
    drifts = []
    for plant in plants:
        asset_id = plant["asset_id"]
        name     = plant.get("name", asset_id)
        try:
            pvlib_cfg    = build_pvlib_config(plant, templates)
            master_hash  = compute_hash(pvlib_cfg)
            existing     = tb.get_server_attributes(asset_id)
            tb_hash      = existing.get("pvlib_config_hash", "(none)")
            status       = "OK" if master_hash == tb_hash else "DRIFT"
        except Exception as exc:
            log.warning("[%s] could not compute/fetch hash: %s", name, exc)
            master_hash = "(error)"
            tb_hash     = "(unknown)"
            status      = "ERROR"

        entry = {
            "asset_id":    asset_id,
            "name":        name,
            "master_hash": master_hash,
            "tb_hash":     tb_hash,
            "status":      status,
        }
        drifts.append(entry)

    return drifts


def _print_table(drifts: List[dict]) -> None:
    header = f"{'NAME':<35} {'MASTER_HASH':<14} {'TB_HASH':<14} STATUS"
    print(f"\n{'─'*80}")
    print(f"  {header}")
    print(f"{'─'*80}")
    for d in drifts:
        icon = {"OK": " ✓", "DRIFT": "❌", "ERROR": "⚠ "}.get(d["status"], "  ")
        print(f"  {icon} {d['name']:<33} {d['master_hash']:<14} {d['tb_hash']:<14} {d['status']}")
    drift_count = sum(1 for d in drifts if d["status"] != "OK")
    print(f"{'─'*80}")
    print(f"  {len(drifts)} plant(s)  —  {drift_count} drift(s)\n")


# ── CLI ───────────────────────────────────────────────────────────────────────

def main() -> int:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--host",       default=os.getenv("TB_HOST", "http://localhost:8080"))
    p.add_argument("--user",       default=os.getenv("TB_USERNAME", ""))
    p.add_argument("--password",   default=os.getenv("TB_PASSWORD", ""))
    p.add_argument("--master",     default=os.getenv("PLANTS_MASTER",
                                   str(_HERE / "plants_master.yml")))
    p.add_argument("--templates",  default=os.getenv("TEMPLATES_DIR",
                                   str(_HERE / "templates")))
    p.add_argument("--plant",      default="",
                   help="Check only this asset UUID")
    p.add_argument("--report-only", action="store_true", default=True,
                   help="Print drift report without writing (default)")
    p.add_argument("--fix",        action="store_true",
                   help="Rewrite diverging plants to restore hash alignment")
    p.add_argument("--format", choices=["table", "json"], default="table")
    args = p.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s",
                        stream=sys.stderr)

    if not args.user or not args.password:
        log.error("TB_USERNAME and TB_PASSWORD must be set")
        return 2

    master_path   = Path(args.master)
    templates_dir = Path(args.templates)

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
        log.error("Cannot reach ThingsBoard: %s", exc)
        return 2

    drifts = detect_drift(tb, plants, templates)

    if args.format == "json":
        print(json.dumps(drifts, indent=2))
    else:
        _print_table(drifts)

    diverging = [d for d in drifts if d["status"] == "DRIFT"]

    if not diverging:
        log.info("No config drift detected.")
        return 0

    if not args.fix:
        log.warning("%d plant(s) have config drift. Run with --fix to rewrite.", len(diverging))
        return 1

    # ── Fix mode: rewrite diverging plants ──────────────────────────────────
    log.info("Fixing %d plant(s)…", len(diverging))
    drift_ids = {d["asset_id"] for d in diverging}
    fix_plants = [p for p in plants if p.get("asset_id") in drift_ids]

    errors = 0
    for plant in fix_plants:
        result = process_plant(tb, plant, templates, force_overwrite=True)
        if result == "error":
            errors += 1

    if errors:
        log.error("%d plant(s) failed to fix", errors)
        return 1

    log.info("All diverging plants fixed successfully.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
