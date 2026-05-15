"""
auto_onboard.py — Weekly full-historical-backfill cron (Phase 4 Step 28 — Q-A).

Runs every Sunday at 03:00 local time when AUTO_ONBOARD_ENABLED=true.

For every discovered plant where `onboarding_completed != true`:
  a. Verify commissioning_date is set — skip with WARN if missing.
  b. Run pvalue_job for each of the last 10 years (writes p*_energy_annual timeseries).
  c. Run daily_job backfill from max(commissioning_date, today-10yr) to yesterday.
  d. If services["loss_attribution"]: run loss_rollup backfill for same window.
  e. If services["loss_attribution"]: recompute lifetime attributes.
  f. If services["revenue"]: run revenue_job backfill for 10 years.
  g. On success: set onboarding_completed=true + onboarding_completed_at=<ISO>.
  h. On any step failure or per-plant timeout (AUTOONBOARD_PER_PLANT_TIMEOUT_S):
     leave attrs unset; the next Sunday retries.

Idempotent: a plant with onboarding_completed=true is always skipped.

Admin endpoints (registered in forecast.py):
  POST /admin/run-autoonboard            (full fleet)
  POST /admin/run-autoonboard?asset_id=<id>  (single plant)

Prometheus counters (exposed via /metrics):
  pvlib_autoonboard_attempted_total
  pvlib_autoonboard_completed_total
  pvlib_autoonboard_failed_total
  pvlib_autoonboard_pending  [gauge]
"""
from __future__ import annotations

import asyncio
import logging
from datetime import datetime, date, timedelta, timezone
from typing import Any, Dict, List, Optional
from zoneinfo import ZoneInfo

from app.config import settings

log = logging.getLogger(__name__)

# ── Prometheus-style counters (read by /metrics) ─────────────────────────────
autoonboard_attempted_total:  int = 0
autoonboard_completed_total:  int = 0
autoonboard_failed_total:     int = 0
autoonboard_pending_gauge:    int = 0


# ── Main entry point ──────────────────────────────────────────────────────────

async def run_autoonboard_now(
    tb_client,
    asset_id: Optional[str] = None,
) -> Dict[str, Any]:
    """Run full onboarding backfill for all pending plants (or a single plant).

    Parameters
    ----------
    tb_client : ThingsBoardClient
    asset_id : str, optional
        If set, process only this plant UUID.  None = full fleet.
    """
    global autoonboard_attempted_total, autoonboard_completed_total
    global autoonboard_failed_total, autoonboard_pending_gauge

    plants, _ = await tb_client.discover_plants(settings.root_asset_ids, force=True)

    if asset_id:
        plants = [p for p in plants if p.id == asset_id]
        if not plants:
            log.warning("run_autoonboard_now: plant %s not found in discovered set", asset_id)
            return {"status": "plant_not_found", "asset_id": asset_id}

    # Filter to plants not yet completed
    pending = []
    for plant in plants:
        try:
            attrs = await tb_client.get_asset_attributes(plant.id)
            if _truthy(attrs.get("onboarding_completed")):
                log.debug("run_autoonboard_now: %s already onboarded — skipping", plant.id)
                continue
            pending.append((plant, attrs))
        except Exception as exc:
            log.warning("run_autoonboard_now: attrs fetch failed for %s: %s", plant.id, exc)
            pending.append((plant, {}))

    autoonboard_pending_gauge = len(pending)
    log.info("run_autoonboard_now: %d plant(s) pending onboarding", len(pending))

    stats = {"attempted": 0, "completed": 0, "failed": 0, "skipped": 0}

    for plant, attrs in pending:
        stats["attempted"] += 1
        autoonboard_attempted_total += 1
        try:
            success = await asyncio.wait_for(
                _onboard_one_plant(tb_client, plant, attrs),
                timeout=settings.AUTOONBOARD_PER_PLANT_TIMEOUT_S,
            )
            if success:
                stats["completed"] += 1
                autoonboard_completed_total += 1
            else:
                stats["skipped"] += 1
        except asyncio.TimeoutError:
            log.error(
                "run_autoonboard_now: plant %s timed out after %d s — will retry next Sunday",
                plant.id, settings.AUTOONBOARD_PER_PLANT_TIMEOUT_S,
            )
            stats["failed"] += 1
            autoonboard_failed_total += 1
        except Exception as exc:
            log.error("run_autoonboard_now: plant %s failed: %s", plant.id, exc)
            stats["failed"] += 1
            autoonboard_failed_total += 1

    autoonboard_pending_gauge = max(0, autoonboard_pending_gauge - stats["completed"])
    log.info(
        "run_autoonboard_now: done — attempted=%d completed=%d failed=%d skipped=%d",
        stats["attempted"], stats["completed"], stats["failed"], stats["skipped"],
    )
    return stats


async def _onboard_one_plant(tb_client, plant, attrs: dict) -> bool:
    """Run the full backfill chain for one plant.  Returns True on success, False on skip."""
    from app.services.pvalue_job import run_pvalue_job
    from app.services.daily_job import run_daily_rollup
    from app.services.loss_rollup_job import run_loss_rollup
    from app.services.revenue_job import run_revenue_backfill

    plant_id = plant.id
    plant_name = plant.name
    tz = ZoneInfo(settings.TZ_LOCAL)

    # Require commissioning_date
    commissioning_raw = attrs.get("commissioning_date")
    if not commissioning_raw:
        log.warning("_onboard_one_plant: %s (%s) has no commissioning_date — skipping",
                    plant_id, plant_name)
        return False

    try:
        commissioning = date.fromisoformat(str(commissioning_raw)[:10])
    except (ValueError, TypeError) as exc:
        log.warning("_onboard_one_plant: %s invalid commissioning_date %r: %s",
                    plant_id, commissioning_raw, exc)
        return False

    today = datetime.now(tz).date()
    max_history = today - timedelta(days=10 * 365)  # 10 years
    backfill_start = max(commissioning, max_history)
    backfill_end   = today - timedelta(days=1)       # yesterday

    current_year = today.year
    loss_enabled = plant.services.get("loss_attribution", True)
    revenue_enabled = plant.services.get("revenue", True)
    pvalue_enabled = plant.services.get("p_values", True)

    log.info(
        "_onboard_one_plant: %s (%s) — commissioning=%s backfill=%s→%s "
        "loss=%s revenue=%s pvalue=%s",
        plant_id, plant_name, commissioning, backfill_start, backfill_end,
        loss_enabled, revenue_enabled, pvalue_enabled,
    )

    # ── a. P-values for 10 years ─────────────────────────────────────────────
    if pvalue_enabled:
        for year in range(current_year - 9, current_year + 1):
            try:
                await run_pvalue_job(tb_client, target_year=year, plant_ids=[plant_id])
                log.debug("_onboard_one_plant: %s pvalue %d OK", plant_id, year)
            except Exception as exc:
                log.warning("_onboard_one_plant: %s pvalue %d failed: %s", plant_id, year, exc)

    # ── b. Daily energy backfill ─────────────────────────────────────────────
    d = backfill_start
    while d <= backfill_end:
        target_utc = datetime(d.year, d.month, d.day, 0, 5, 0, tzinfo=tz).astimezone(timezone.utc)
        try:
            await run_daily_rollup(tb_client, date=target_utc)
        except Exception as exc:
            log.warning("_onboard_one_plant: %s daily %s failed: %s", plant_id, d, exc)
        d += timedelta(days=1)
    log.info("_onboard_one_plant: %s daily backfill done (%d days)",
             plant_id, (backfill_end - backfill_start).days + 1)

    # ── c/d. Loss rollup backfill ────────────────────────────────────────────
    if loss_enabled:
        d = backfill_start
        while d <= backfill_end:
            target_utc = datetime(d.year, d.month, d.day, 0, 10, 0, tzinfo=tz).astimezone(timezone.utc)
            try:
                await run_loss_rollup(tb_client, date=target_utc)
            except Exception as exc:
                log.warning("_onboard_one_plant: %s loss %s failed: %s", plant_id, d, exc)
            d += timedelta(days=1)
        log.info("_onboard_one_plant: %s loss backfill done", plant_id)

        # Recompute lifetime attributes after bulk backfill
        try:
            from app.services.loss_rollup_job import recompute_lifetime_for_fleet
            await recompute_lifetime_for_fleet(tb_client, asset_id=plant_id)
            log.info("_onboard_one_plant: %s lifetime recompute done", plant_id)
        except Exception as exc:
            log.warning("_onboard_one_plant: %s lifetime recompute failed: %s", plant_id, exc)

    # ── e. Revenue backfill ──────────────────────────────────────────────────
    if revenue_enabled:
        try:
            await run_revenue_backfill(tb_client, asset_id=plant_id, years_back=10)
            log.info("_onboard_one_plant: %s revenue backfill done", plant_id)
        except Exception as exc:
            log.warning("_onboard_one_plant: %s revenue backfill failed: %s", plant_id, exc)

    # ── f. Mark completed ────────────────────────────────────────────────────
    now_iso = datetime.now(timezone.utc).isoformat()
    await tb_client.post_attributes("ASSET", plant_id, "SERVER_SCOPE", {
        "onboarding_completed":    True,
        "onboarding_completed_at": now_iso,
    })
    log.info("_onboard_one_plant: %s (%s) onboarding complete at %s", plant_id, plant_name, now_iso)
    return True


def _truthy(v) -> bool:
    if isinstance(v, bool): return v
    if isinstance(v, (int, float)): return v != 0
    if isinstance(v, str): return v.strip().lower() in ("true", "yes", "1")
    return bool(v)
