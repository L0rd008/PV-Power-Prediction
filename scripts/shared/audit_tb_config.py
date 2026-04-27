"""
audit_tb_config.py — Gap 26 (Phase F)

Audits ThingsBoard plant assets against the pvlib-service attribute contract.

For every pvlib-enabled plant the script checks:
  1. Required attributes are present and have the correct type.
  2. Numeric values are within sensible physics ranges.
  3. Optional but recommended attributes are flagged as WARN when missing.
  4. If a reference config JSON is supplied (--config-file), specific values are
     compared and MISMATCH is reported on divergence.

EXIT CODES
----------
  0  All plants clean (no ERR findings)
  1  One or more ERR findings (required attrs missing or out-of-range)
  2  Connection / auth failure

USAGE
-----
  # Discover all pvlib plants from root assets (TB_* env vars):
  python audit_tb_config.py

  # Explicit credentials:
  python audit_tb_config.py --host https://tb.example.com \\
                             --user admin@tenant.com \\
                             --password <secret>

  # Audit specific plants by asset ID:
  python audit_tb_config.py --plant-ids <uuid1>,<uuid2>

  # Compare attributes against a reference JSON (plant_config.json schema):
  python audit_tb_config.py --config-file config/kebithigollewa_pvlib_config.json \\
                             --plant-ids <uuid>

  # Machine-readable output:
  python audit_tb_config.py --format json
  python audit_tb_config.py --format csv

  # Output to a file instead of stdout:
  python audit_tb_config.py --format json --output audit_report.json

  # Discover from an explicit root asset (overrides TB_ROOT_ASSET_IDS):
  python audit_tb_config.py --root-ids <root_uuid1>,<root_uuid2>

ENVIRONMENT VARIABLES
---------------------
  TB_HOST           ThingsBoard base URL (default http://localhost:8080)
  TB_USERNAME       Tenant admin email
  TB_PASSWORD       Tenant admin password
  TB_ROOT_ASSET_IDS Comma-separated root asset UUIDs for BFS discovery
"""
from __future__ import annotations

import argparse
import csv
import io
import json
import logging
import os
import sys
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import requests

log = logging.getLogger("audit_tb_config")

# ── Attribute contract ────────────────────────────────────────────────────────

# Each entry: (attribute_key, expected_type, severity, range_check_fn, note)
#   severity  "ERR"  = required; missing or wrong-type blocks pvlib
#             "WARN" = recommended; missing degrades quality but won't crash
#   range_check_fn: callable(value) -> bool (True = valid) or None to skip

def _bool_check(v): return isinstance(v, bool) or str(v).lower() in ("true", "false", "1", "0")
def _json_array(v): return isinstance(v, (list, str)) and (isinstance(v, list) or _parseable_json(v))
def _json_obj(v):   return isinstance(v, (dict, str)) and (isinstance(v, dict) or _parseable_json(v))
def _parseable_json(v):
    try:
        json.loads(v); return True
    except (ValueError, TypeError): return False

def _lat(v):   return -90.0 <= float(v) <= 90.0
def _lon(v):   return -180.0 <= float(v) <= 180.0
def _alt(v):   return -500.0 <= float(v) <= 9000.0
def _loss(v):  return 0.0 <= float(v) <= 0.5        # loss fractions never > 50 %
def _shade(v): return 0.0 < float(v) <= 1.0         # far_shading multiplier

@dataclass
class _AttrSpec:
    key: str
    expected_types: Tuple[type, ...]   # acceptable Python types AFTER optional JSON parse
    severity: str                      # "ERR" | "WARN"
    range_fn: Any = None               # callable(raw_value) -> bool  or  None
    note: str = ""


REQUIRED_ATTRS: List[_AttrSpec] = [
    _AttrSpec("pvlib_enabled",  (bool,),       "ERR",  _bool_check,
              "Must be true on every pvlib-enabled plant (discovery filter)"),
    _AttrSpec("isPlant",        (bool,),        "ERR",  _bool_check,
              "BFS filter flag; must be true on leaf plants"),
    _AttrSpec("latitude",       (float, int),   "ERR",  _lat,
              "Decimal degrees [-90, 90]"),
    _AttrSpec("longitude",      (float, int),   "ERR",  _lon,
              "Decimal degrees [-180, 180]"),
    _AttrSpec("altitude_m",     (float, int),   "ERR",  _alt,
              "Metres above sea level [-500, 9000]"),
    _AttrSpec("timezone",       (str,),         "ERR",  None,
              "IANA timezone string, e.g. 'Asia/Colombo'"),
    _AttrSpec("orientations",   (list, str),    "ERR",  _json_array,
              "JSON array of panel orientations"),
    _AttrSpec("module",         (dict, str),    "ERR",  _json_obj,
              "JSON object with area_m2, efficiency_stc, gamma_p"),
    _AttrSpec("inverter",       (dict, str),    "ERR",  _json_obj,
              "JSON object with ac_rating_kw, etc."),
]

RECOMMENDED_ATTRS: List[_AttrSpec] = [
    _AttrSpec("iam",            (dict, list, str), "WARN", None,
              "Incidence-angle modifier lookup; defaults to Ashrae if absent"),
    _AttrSpec("thermal_model",  (str,),         "WARN", None,
              "pvlib PVSystem thermal model name"),
    _AttrSpec("soiling",        (float, int),   "WARN", _loss,   "Loss fraction [0, 0.5]"),
    _AttrSpec("lid",            (float, int),   "WARN", _loss,   "Loss fraction [0, 0.5]"),
    _AttrSpec("mismatch",       (float, int),   "WARN", _loss,   "Loss fraction [0, 0.5]"),
    _AttrSpec("dc_wiring",      (float, int),   "WARN", _loss,   "Loss fraction [0, 0.5]"),
    _AttrSpec("ac_wiring",      (float, int),   "WARN", _loss,   "Loss fraction [0, 0.5]"),
    _AttrSpec("albedo",         (float, int),   "WARN", lambda v: 0.0 <= float(v) <= 1.0,
              "Ground reflectance [0, 1]"),
    _AttrSpec("far_shading",    (float, int),   "WARN", _shade,  "Shading multiplier (0, 1]"),
    _AttrSpec("defaults",       (dict, str),    "WARN", _json_obj,
              "JSON: {wind_speed_ms, air_temp_c} used when station data absent"),
    _AttrSpec("active_power_unit", (str,),      "WARN", lambda v: str(v) in ("kW", "W"),
              "Set by set_active_power_unit.py — must be 'kW' or 'W'"),
]

ALL_SPECS: List[_AttrSpec] = REQUIRED_ATTRS + RECOMMENDED_ATTRS


# ── Finding data class ────────────────────────────────────────────────────────

@dataclass
class Finding:
    plant_id:   str
    plant_name: str
    severity:   str        # ERR | WARN | INFO | OK
    attribute:  str
    status:     str        # MISSING | TYPE_ERROR | RANGE_ERROR | MISMATCH | OK
    expected:   str = ""
    actual:     str = ""
    note:       str = ""


# ── ThingsBoard REST client (sync, minimal — no async needed for CLI) ─────────

class TBClient:
    def __init__(self, host: str, username: str, password: str) -> None:
        self.base = host.rstrip("/")
        self._s = requests.Session()
        self._s.headers["Content-Type"] = "application/json"
        self._token: Optional[str] = None
        self._login(username, password)

    def _login(self, username: str, password: str) -> None:
        resp = self._s.post(f"{self.base}/api/auth/login",
                            json={"username": username, "password": password},
                            timeout=20)
        resp.raise_for_status()
        self._token = resp.json()["token"]
        self._s.headers["X-Authorization"] = f"Bearer {self._token}"

    def _get(self, path: str, params: dict | None = None) -> Any:
        resp = self._s.get(f"{self.base}{path}", params=params, timeout=20)
        if resp.status_code == 404:
            return None
        resp.raise_for_status()
        return resp.json()

    def get_asset_info(self, asset_id: str) -> Optional[dict]:
        return self._get(f"/api/asset/{asset_id}")

    def get_server_attributes(self, asset_id: str) -> dict:
        data = self._get(
            f"/api/plugins/telemetry/ASSET/{asset_id}/values/attributes/SERVER_SCOPE"
        )
        if not data:
            return {}
        return {item["key"]: item["value"] for item in data}

    def bfs_discover_pvlib_plants(self, root_ids: List[str]) -> List[dict]:
        """BFS over asset relations; return assets with pvlib_enabled=true."""
        visited: set[str] = set()
        queue = list(root_ids)
        plants = []

        while queue:
            asset_id = queue.pop(0)
            if asset_id in visited:
                continue
            visited.add(asset_id)

            info = self.get_asset_info(asset_id)
            if not info:
                continue

            attrs = self.get_server_attributes(asset_id)
            is_plant   = _truthy(attrs.get("isPlant", False))
            pvlib_on   = _truthy(attrs.get("pvlib_enabled", False))

            if is_plant and pvlib_on:
                plants.append({"id": asset_id, "name": info.get("name", asset_id),
                                "attrs": attrs})

            # Queue children
            children = self._get_children(asset_id)
            for child_id in children:
                if child_id not in visited:
                    queue.append(child_id)

        return plants

    def _get_children(self, asset_id: str) -> List[str]:
        """Return IDs of assets directly related to this asset as children."""
        data = self._get(
            "/api/relations",
            params={"fromId": asset_id, "fromType": "ASSET",
                    "relationTypeGroup": "COMMON", "relationType": "Contains"},
        )
        if not data:
            return []
        ids = []
        for rel in (data if isinstance(data, list) else data.get("data", [])):
            to = rel.get("to", {})
            if to.get("entityType") == "ASSET":
                ids.append(to["id"])
        return ids

    def get_plants_by_ids(self, asset_ids: List[str]) -> List[dict]:
        """Fetch plant info + attrs for a given list of asset IDs."""
        plants = []
        for asset_id in asset_ids:
            info  = self.get_asset_info(asset_id)
            attrs = self.get_server_attributes(asset_id)
            name  = info.get("name", asset_id) if info else asset_id
            plants.append({"id": asset_id, "name": name, "attrs": attrs})
        return plants


# ── Audit logic ───────────────────────────────────────────────────────────────

def _truthy(v) -> bool:
    if isinstance(v, bool): return v
    if isinstance(v, (int, float)): return v != 0
    if isinstance(v, str): return v.strip().lower() in ("true", "yes", "1")
    return bool(v)


def _coerce(raw, types: Tuple[type, ...]):
    """Try to coerce raw value into one of the expected types.

    Handles the common case where TB returns a JSON-serialised string for
    dict/list attributes (e.g. orientations stored as a JSON string).
    """
    if isinstance(raw, types):
        return raw, True
    # Try JSON parse for string → dict/list
    if isinstance(raw, str) and (dict in types or list in types):
        try:
            parsed = json.loads(raw)
            if isinstance(parsed, types):
                return parsed, True
        except (ValueError, TypeError):
            pass
    # Try numeric coercion
    if (float in types or int in types) and isinstance(raw, str):
        try:
            return float(raw), True
        except ValueError:
            pass
    # Try bool coercion
    if bool in types and isinstance(raw, str):
        if raw.strip().lower() in ("true", "false", "1", "0"):
            return raw.strip().lower() in ("true", "1"), True
    return raw, False


def audit_plant(
    plant: dict,
    ref_config: Optional[dict] = None,
) -> List[Finding]:
    """Produce a list of findings for one plant."""
    pid   = plant["id"]
    pname = plant["name"]
    attrs = plant["attrs"]
    findings: List[Finding] = []

    for spec in ALL_SPECS:
        raw = attrs.get(spec.key)

        if raw is None:
            findings.append(Finding(
                plant_id=pid, plant_name=pname,
                severity=spec.severity, attribute=spec.key,
                status="MISSING",
                expected=f"<{spec.expected_types[0].__name__}>",
                actual="(absent)",
                note=spec.note,
            ))
            continue

        coerced, ok = _coerce(raw, spec.expected_types)

        if not ok:
            findings.append(Finding(
                plant_id=pid, plant_name=pname,
                severity=spec.severity, attribute=spec.key,
                status="TYPE_ERROR",
                expected="|".join(t.__name__ for t in spec.expected_types),
                actual=repr(raw)[:80],
                note=spec.note,
            ))
            continue

        # Range check
        if spec.range_fn is not None:
            try:
                in_range = spec.range_fn(raw)
            except (TypeError, ValueError):
                in_range = False
            if not in_range:
                findings.append(Finding(
                    plant_id=pid, plant_name=pname,
                    severity=spec.severity, attribute=spec.key,
                    status="RANGE_ERROR",
                    expected="within valid range",
                    actual=repr(raw)[:80],
                    note=spec.note,
                ))
                continue

        # Reference-config comparison (optional)
        if ref_config:
            ref_val = _ref_value(ref_config, spec.key)
            if ref_val is not None:
                match = _values_match(coerced, ref_val)
                if not match:
                    findings.append(Finding(
                        plant_id=pid, plant_name=pname,
                        severity="WARN", attribute=spec.key,
                        status="MISMATCH",
                        expected=repr(ref_val)[:80],
                        actual=repr(coerced)[:80],
                        note="Differs from reference config",
                    ))
                    continue

        # All checks passed
        findings.append(Finding(
            plant_id=pid, plant_name=pname,
            severity="OK", attribute=spec.key,
            status="OK",
            actual=repr(coerced)[:60],
        ))

    return findings


def _ref_value(ref_config: dict, key: str) -> Any:
    """Extract a comparable value from the reference config JSON.

    Handles both flat and nested config schemas.
    """
    # Flat key first
    if key in ref_config:
        return ref_config[key]
    # Nested paths (plant_config.json schema)
    _nested_map = {
        "latitude":  ("location", "lat"),
        "longitude": ("location", "lon"),
        "altitude_m": ("location", "altitude_m"),
        "timezone":  ("location", "timezone"),
    }
    if key in _nested_map:
        section, sub = _nested_map[key]
        sec = ref_config.get(section, {})
        if sub in sec:
            return sec[sub]
    return None


def _values_match(actual: Any, expected: Any, tol: float = 1e-4) -> bool:
    """Numeric tolerance comparison; falls back to equality for other types."""
    try:
        return abs(float(actual) - float(expected)) <= tol
    except (TypeError, ValueError):
        pass
    # For JSON objects/arrays: compare serialised forms
    if isinstance(actual, (dict, list)):
        return json.dumps(actual, sort_keys=True) == json.dumps(expected, sort_keys=True)
    return str(actual).strip() == str(expected).strip()


# ── Output formatters ─────────────────────────────────────────────────────────

def _print_table(findings: List[Finding], plants: List[dict]) -> None:
    """Pretty console table grouped by plant."""
    plant_order = [p["id"] for p in plants]
    by_plant: Dict[str, List[Finding]] = {}
    for f in findings:
        by_plant.setdefault(f.plant_id, []).append(f)

    # Summary counters
    err_total = warn_total = ok_total = 0

    for pid in plant_order:
        pname = next((p["name"] for p in plants if p["id"] == pid), pid)
        pf = by_plant.get(pid, [])
        errs  = [f for f in pf if f.severity == "ERR"]
        warns = [f for f in pf if f.severity == "WARN"]
        oks   = [f for f in pf if f.severity == "OK"]
        err_total += len(errs); warn_total += len(warns); ok_total += len(oks)

        status_icon = "✓" if not errs else "✗"
        print(f"\n{'─'*72}")
        print(f"  {status_icon}  {pname}  ({pid})")
        print(f"     {len(oks)} OK   {len(warns)} WARN   {len(errs)} ERR")

        for f in sorted(pf, key=lambda x: (x.severity == "OK", x.severity)):
            if f.status == "OK":
                continue   # skip noisy OK lines unless --verbose requested
            icon = "❌" if f.severity == "ERR" else "⚠ "
            print(f"     {icon} [{f.status:12s}] {f.attribute}")
            if f.expected:
                print(f"              expected : {f.expected}")
            if f.actual:
                print(f"              actual   : {f.actual}")
            if f.note:
                print(f"              note     : {f.note}")

    print(f"\n{'='*72}")
    print(f"  TOTAL — {len(plants)} plants   {ok_total} OK  "
          f"{warn_total} WARN  {err_total} ERR")


def _print_table_verbose(findings: List[Finding], plants: List[dict]) -> None:
    """Like _print_table but includes OK rows."""
    plant_order = [p["id"] for p in plants]
    by_plant: Dict[str, List[Finding]] = {}
    for f in findings:
        by_plant.setdefault(f.plant_id, []).append(f)

    for pid in plant_order:
        pname = next((p["name"] for p in plants if p["id"] == pid), pid)
        pf = by_plant.get(pid, [])
        errs = [f for f in pf if f.severity == "ERR"]
        print(f"\n{'─'*72}")
        print(f"  {'✓' if not errs else '✗'}  {pname}  ({pid})")
        for f in sorted(pf, key=lambda x: (x.severity == "OK", x.severity, x.attribute)):
            icon = {"OK": "  ✓", "ERR": "  ❌", "WARN": "  ⚠ "}.get(f.severity, "   ")
            print(f"  {icon} [{f.status:12s}] {f.attribute:<30s}  {f.actual}")


def _to_json(findings: List[Finding], plants: List[dict]) -> str:
    by_plant = {}
    for f in findings:
        by_plant.setdefault(f.plant_id, []).append({
            "attribute": f.attribute,
            "severity":  f.severity,
            "status":    f.status,
            "expected":  f.expected,
            "actual":    f.actual,
            "note":      f.note,
        })
    output = {
        "plants": [
            {
                "id":       p["id"],
                "name":     p["name"],
                "findings": by_plant.get(p["id"], []),
                "err_count":  sum(1 for f in by_plant.get(p["id"], []) if f["severity"] == "ERR"),
                "warn_count": sum(1 for f in by_plant.get(p["id"], []) if f["severity"] == "WARN"),
                "ok_count":   sum(1 for f in by_plant.get(p["id"], []) if f["severity"] == "OK"),
            }
            for p in plants
        ],
        "summary": {
            "plants_audited": len(plants),
            "total_err":  sum(1 for f in findings if f.severity == "ERR"),
            "total_warn": sum(1 for f in findings if f.severity == "WARN"),
            "total_ok":   sum(1 for f in findings if f.severity == "OK"),
        },
    }
    return json.dumps(output, indent=2)


def _to_csv(findings: List[Finding]) -> str:
    buf = io.StringIO()
    w = csv.writer(buf)
    w.writerow(["plant_id", "plant_name", "severity", "attribute",
                "status", "expected", "actual", "note"])
    for f in findings:
        w.writerow([f.plant_id, f.plant_name, f.severity, f.attribute,
                    f.status, f.expected, f.actual, f.note])
    return buf.getvalue()


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--host",     default=os.getenv("TB_HOST", "http://localhost:8080"))
    p.add_argument("--user",     default=os.getenv("TB_USERNAME", ""))
    p.add_argument("--password", default=os.getenv("TB_PASSWORD", ""))
    p.add_argument("--root-ids", default=os.getenv("TB_ROOT_ASSET_IDS", ""),
                   help="Comma-separated root asset UUIDs for BFS discovery")
    p.add_argument("--plant-ids", default="",
                   help="Comma-separated asset UUIDs to audit directly (skips BFS)")
    p.add_argument("--config-file", default="",
                   help="Path to a reference plant_config.json to compare against")
    p.add_argument("--format", choices=["table", "json", "csv"], default="table")
    p.add_argument("--output", default="",
                   help="Write output to this file instead of stdout")
    p.add_argument("--warn-as-err", action="store_true",
                   help="Treat WARN findings as errors for exit-code purposes")
    p.add_argument("--verbose", action="store_true",
                   help="Include OK attribute rows in table output")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    logging.basicConfig(
        level=logging.INFO,
        format="%(levelname)s  %(message)s",
        stream=sys.stderr,
    )

    if not args.user or not args.password:
        log.error("TB_USERNAME and TB_PASSWORD must be set (env vars or --user/--password)")
        return 2

    # ── Connect ──────────────────────────────────────────────────────────────
    try:
        tb = TBClient(args.host, args.user, args.password)
        log.info("Connected to %s", args.host)
    except requests.HTTPError as exc:
        log.error("Authentication failed: %s", exc)
        return 2
    except requests.ConnectionError as exc:
        log.error("Cannot reach ThingsBoard at %s: %s", args.host, exc)
        return 2

    # ── Discover or use explicit IDs ─────────────────────────────────────────
    if args.plant_ids:
        ids = [x.strip() for x in args.plant_ids.split(",") if x.strip()]
        log.info("Auditing %d plant(s) by ID", len(ids))
        plants = tb.get_plants_by_ids(ids)
    elif args.root_ids:
        roots = [x.strip() for x in args.root_ids.split(",") if x.strip()]
        log.info("BFS discover from %d root asset(s) …", len(roots))
        plants = tb.bfs_discover_pvlib_plants(roots)
        log.info("Discovered %d pvlib-enabled plant(s)", len(plants))
    else:
        log.error("Provide --plant-ids or --root-ids (or set TB_ROOT_ASSET_IDS)")
        return 2

    if not plants:
        log.warning("No plants found — nothing to audit")
        return 0

    # ── Load reference config ─────────────────────────────────────────────────
    ref_config: Optional[dict] = None
    if args.config_file:
        try:
            with open(args.config_file, encoding="utf-8") as fh:
                ref_config = json.load(fh)
            log.info("Reference config loaded: %s", args.config_file)
        except (OSError, json.JSONDecodeError) as exc:
            log.error("Cannot load config file %r: %s", args.config_file, exc)
            return 2

    # ── Audit ─────────────────────────────────────────────────────────────────
    all_findings: List[Finding] = []
    for plant in plants:
        findings = audit_plant(plant, ref_config=ref_config)
        all_findings.extend(findings)
        err_count  = sum(1 for f in findings if f.severity == "ERR")
        warn_count = sum(1 for f in findings if f.severity == "WARN")
        log.info("  %-40s  ERR=%d  WARN=%d", plant["name"], err_count, warn_count)

    # ── Format output ─────────────────────────────────────────────────────────
    if args.format == "json":
        output_str = _to_json(all_findings, plants)
    elif args.format == "csv":
        output_str = _to_csv(all_findings)
    else:
        # table: write to stdout directly; capture only for --output
        if args.output:
            import io as _io
            buf = _io.StringIO()
            old_stdout = sys.stdout
            sys.stdout = buf
        if args.verbose:
            _print_table_verbose(all_findings, plants)
        else:
            _print_table(all_findings, plants)
        if args.output:
            sys.stdout = old_stdout
            output_str = buf.getvalue()
        else:
            output_str = None

    if output_str is not None:
        if args.output:
            with open(args.output, "w", encoding="utf-8") as fh:
                fh.write(output_str)
            log.info("Report written to %s", args.output)
        else:
            print(output_str)

    # ── Exit code ──────────────────────────────────────────────────────────────
    has_err = any(f.severity == "ERR" for f in all_findings)
    has_warn = any(f.severity == "WARN" for f in all_findings)
    if has_err or (args.warn_as_err and has_warn):
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
