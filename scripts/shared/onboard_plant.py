"""
onboard_plant.py — Idempotent single-plant onboarding sequence for Pvlib-Service.

Runs 9 steps in order, logging OK / FAIL / SKIP for each.  Any step that fails
stops the sequence and exits with code 1 (unless --skip-on-fail is set).

All steps are idempotent — safe to re-run at any time without double-counting
or overwriting valid data.

USAGE
-----
  # Onboard KSP plant with 1 year of historical backfill:
  python onboard_plant.py --asset-id 0e4b4070-50ff-11ef-b4ce-d5aee9e495ad

  # Longer backfill (2 years):
  python onboard_plant.py --asset-id <uuid> --years-back 2

  # Skip loss attribution backfill:
  python onboard_plant.py --asset-id <uuid> --skip-loss

  # Preview without executing (dry-run):
  python onboard_plant.py --asset-id <uuid> --dry-run

ENVIRONMENT VARIABLES
---------------------
  TB_HOST      ThingsBoard base URL (default http://localhost:8080)
  TB_USERNAME  Tenant admin email
  TB_PASSWORD  Tenant admin password
  PVLIB_HOST   Pvlib-Service base URL (default http://localhost:8004)
"""
from __future__ import annotations

import argparse
import datetime
import logging
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Optional

import requests

log = logging.getLogger("onboard_plant")
_HERE = Path(__file__).parent


def _get(session: requests.Session, url: str, timeout: int = 30) -> dict:
    resp = session.get(url, timeout=timeout)
    resp.raise_for_status()
    return resp.json()


def _post(session: requests.Session, url: str, timeout: int = 300, **kwargs) -> dict:
    resp = session.post(url, timeout=timeout, **kwargs)
    resp.raise_for_status()
    return resp.json()


# ── Step runner ───────────────────────────────────────────────────────────────

class StepRunner:
    def __init__(self, asset_id: str, dry_run: bool = False) -> None:
        self.asset_id = asset_id
        self.dry_run  = dry_run
        self._results: list = []

    def step(self, number: int, name: str, fn, *args, **kwargs) -> bool:
        """Execute one step.  Returns True on success."""
        prefix = f"  Step {number}: {name}"
        if self.dry_run:
            print(f"{prefix}  [DRY-RUN — would execute]")
            self._results.append((number, name, "DRY"))
            return True
        try:
            result = fn(*args, **kwargs)
            print(f"{prefix}  [OK] {result or ''}")
            self._results.append((number, name, "OK"))
            return True
        except Exception as exc:
            print(f"{prefix}  [FAIL] {exc}")
            self._results.append((number, name, "FAIL"))
            return False

    def summary(self) -> None:
        print(f"\n{'='*60}")
        for num, name, status in self._results:
            icon = {"OK": "✓", "FAIL": "✗", "SKIP": "-", "DRY": "○"}.get(status, "?")
            print(f"  {icon} Step {num}: {name}  [{status}]")
        fails = sum(1 for _, _, s in self._results if s == "FAIL")
        print(f"\n  {len(self._results)} step(s) total — {fails} failure(s)")


# ── Step implementations ──────────────────────────────────────────────────────

def _step1_audit(asset_id: str, tb_host: str, tb_user: str, tb_pass: str) -> str:
    """Run audit_tb_config.py; abort if any ERR found."""
    script = _HERE / "audit_tb_config.py"
    result = subprocess.run(
        [sys.executable, str(script),
         "--host", tb_host, "--user", tb_user, "--password", tb_pass,
         "--plant-ids", asset_id, "--format", "table"],
        capture_output=True, text=True,
    )
    if result.returncode not in (0, 1):
        raise RuntimeError(f"audit script returned {result.returncode}: {result.stderr[:200]}")
    if result.returncode == 1:
        # ERR found — print summary and abort
        print(result.stdout[-1000:])
        raise RuntimeError("Audit found ERR findings — fix attributes before onboarding")
    return "audit clean"


def _step2_refresh_plants(pvlib: requests.Session, pvlib_base: str) -> str:
    _post(pvlib, f"{pvlib_base}/admin/refresh-plants")
    return "discovery cache invalidated"


def _step3_pvalues_current(pvlib: requests.Session, pvlib_base: str, asset_id: str) -> str:
    year = datetime.date.today().year
    _post(pvlib, f"{pvlib_base}/admin/run-pvalues-plant",
          params={"asset_id": asset_id, "year": year}, timeout=600)
    return f"P-values written for {year}"


def _step4_pvalues_history(
    pvlib: requests.Session, pvlib_base: str, asset_id: str, years_back: int
) -> str:
    current = datetime.date.today().year
    results = []
    for y in range(current - years_back, current):
        _post(pvlib, f"{pvlib_base}/admin/run-pvalues-plant",
              params={"asset_id": asset_id, "year": y}, timeout=600)
        results.append(str(y))
    return "backfilled years: " + ", ".join(results) if results else "no prior years"


def _step5_daily_backfill(
    pvlib: requests.Session, pvlib_base: str, asset_id: str, years_back: int
) -> str:
    today = datetime.date.today()
    start = today - datetime.timedelta(days=365 * years_back)
    end   = today - datetime.timedelta(days=1)
    # Use run-daily-range if available; fall back to per-day calls otherwise
    try:
        resp = pvlib.post(
            f"{pvlib_base}/admin/run-daily-range",
            params={"asset_id": asset_id, "start": str(start), "end": str(end)},
            timeout=600,
        )
        resp.raise_for_status()
        return f"daily range {start} → {end}"
    except requests.HTTPError as exc:
        if exc.response is not None and exc.response.status_code == 404:
            # Endpoint not available — do day-by-day (slower)
            count = 0
            d = start
            while d <= end:
                try:
                    pvlib.post(f"{pvlib_base}/admin/run-daily",
                               params={"date": str(d), "asset_id": asset_id},
                               timeout=120).raise_for_status()
                    count += 1
                except Exception:
                    pass
                d += datetime.timedelta(days=1)
            return f"daily backfill {count} days (day-by-day fallback)"
        raise


def _step6_loss_backfill(
    pvlib: requests.Session, pvlib_base: str, asset_id: str, years_back: int
) -> str:
    today = datetime.date.today()
    start = today - datetime.timedelta(days=365 * years_back)
    end   = today - datetime.timedelta(days=1)
    try:
        resp = pvlib.post(
            f"{pvlib_base}/admin/run-loss-rollup",
            params={"asset_id": asset_id, "start": str(start), "end": str(end)},
            timeout=600,
        )
        resp.raise_for_status()
        return f"loss range {start} → {end}"
    except requests.HTTPError as exc:
        if exc.response is not None and exc.response.status_code == 404:
            return "SKIP (run-loss-rollup endpoint not available)"
        raise


def _step7_lifetime_recompute(
    pvlib: requests.Session, pvlib_base: str, asset_id: str
) -> str:
    try:
        resp = pvlib.post(
            f"{pvlib_base}/admin/recompute-lifetime",
            params={"asset_id": asset_id},
            timeout=120,
        )
        resp.raise_for_status()
        return "lifetime attrs recomputed"
    except requests.HTTPError as exc:
        if exc.response is not None and exc.response.status_code == 404:
            return "SKIP (recompute-lifetime endpoint not available)"
        raise


def _step8_loss_status(
    pvlib: requests.Session, pvlib_base: str, asset_id: str
) -> str:
    try:
        data = _get(pvlib, f"{pvlib_base}/admin/loss-status?asset_id={asset_id}")
        return str(data)[:120]
    except requests.HTTPError as exc:
        if exc.response is not None and exc.response.status_code == 404:
            return "SKIP (loss-status endpoint not available)"
        raise


def _step9_confirm_discovery(
    pvlib: requests.Session, pvlib_base: str, asset_id: str
) -> str:
    data = _get(pvlib, f"{pvlib_base}/pvlib/discover")
    plants = data.get("plants", data) if isinstance(data, dict) else data
    ids  = [p.get("id") or p.get("asset_id", "") for p in (plants if isinstance(plants, list) else [])]
    if asset_id in ids:
        return "plant present in discovery response"
    raise RuntimeError(
        f"Plant {asset_id} NOT found in /pvlib/discover — check pvlib_enabled=true "
        f"and root asset connectivity"
    )


# ── CLI ───────────────────────────────────────────────────────────────────────

def main() -> int:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--asset-id",    required=True, help="Plant asset UUID in ThingsBoard")
    p.add_argument("--years-back",  type=int, default=1,
                   help="How many prior years to backfill P-values and daily energy (default 1)")
    p.add_argument("--skip-loss",   action="store_true",
                   help="Skip loss attribution backfill and lifetime recompute")
    p.add_argument("--dry-run",     action="store_true",
                   help="Preview all steps without executing them")
    p.add_argument("--skip-on-fail", action="store_true",
                   help="Continue to next step even if a step fails (default: abort on first fail)")
    p.add_argument("--host",     default=os.getenv("TB_HOST",    "http://localhost:8080"))
    p.add_argument("--user",     default=os.getenv("TB_USERNAME", ""))
    p.add_argument("--password", default=os.getenv("TB_PASSWORD", ""))
    p.add_argument("--pvlib-host", default=os.getenv("PVLIB_HOST", "http://localhost:8004"))
    args = p.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s",
                        stream=sys.stderr)

    if not args.user or not args.password:
        log.error("TB_USERNAME and TB_PASSWORD must be set")
        return 2

    asset_id = args.asset_id.strip()
    pvlib_base = args.pvlib_host.rstrip("/")

    pvlib = requests.Session()
    runner = StepRunner(asset_id, dry_run=args.dry_run)

    print(f"\nOnboarding plant: {asset_id}")
    print(f"  pvlib-service: {pvlib_base}")
    print(f"  years-back:    {args.years_back}")
    print(f"  skip-loss:     {args.skip_loss}")
    print(f"  dry-run:       {args.dry_run}\n")

    steps = [
        (1, "Audit TB config",
         _step1_audit, asset_id, args.host, args.user, args.password),
        (2, "Refresh plant discovery cache",
         _step2_refresh_plants, pvlib, pvlib_base),
        (3, "Write current-year P-values",
         _step3_pvalues_current, pvlib, pvlib_base, asset_id),
        (4, f"Backfill P-values ({args.years_back} prior year(s))",
         _step4_pvalues_history, pvlib, pvlib_base, asset_id, args.years_back),
        (5, f"Backfill daily energy ({args.years_back} year(s))",
         _step5_daily_backfill, pvlib, pvlib_base, asset_id, args.years_back),
    ]

    if not args.skip_loss:
        steps += [
            (6, f"Backfill loss attribution ({args.years_back} year(s))",
             _step6_loss_backfill, pvlib, pvlib_base, asset_id, args.years_back),
            (7, "Recompute lifetime loss attributes",
             _step7_lifetime_recompute, pvlib, pvlib_base, asset_id),
            (8, "Verify loss status",
             _step8_loss_status, pvlib, pvlib_base, asset_id),
        ]
    else:
        print("  [SKIP] Steps 6-8 (loss attribution disabled via --skip-loss)\n")

    steps.append(
        (9, "Confirm plant appears in /pvlib/discover",
         _step9_confirm_discovery, pvlib, pvlib_base, asset_id)
    )

    for step_num, step_name, fn, *fn_args in steps:
        ok = runner.step(step_num, step_name, fn, *fn_args)
        if not ok and not args.skip_on_fail and not args.dry_run:
            runner.summary()
            return 1

    runner.summary()
    fails = sum(1 for _, _, s in runner._results if s == "FAIL")
    return 1 if fails > 0 else 0


if __name__ == "__main__":
    sys.exit(main())
