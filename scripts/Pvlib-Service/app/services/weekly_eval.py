"""
app/services/weekly_eval.py — Weekly accuracy evaluation job (Gap 20, Phase F).

Scheduled every Sunday at 02:00 local time via APScheduler cron.

Evaluates the 7 completed calendar days ending at the most recent local midnight
(i.e. Mon–Sun of the week just finished).

METRICS per plant
-----------------
  NMAE             = mean(|predicted − actual|) / mean(actual) × 100   [%]
  NRMSE            = sqrt(mean((predicted − actual)²)) / mean(actual) × 100  [%]
  energy_error_pct = (sum(predicted) − sum(actual)) / sum(actual) × 100  [%]

  where:
    predicted  = total_generation_expected_kwh  (written at 00:05 by daily_job.py)
    actual     = EVAL_ACTUAL_DAILY_KEY          (env; default: daily_energy_kwh)

OUTPUT
------
  Key: pvlib_accuracy_report_json
  Type: JSON string (TB timeseries, ASSET entity)
  Timestamp: Unix-ms of the Monday that started the eval window

  Written to:
    - Each plant in EVAL_PLANT_IDS
    - The first root asset as a fleet-level summary

  Schema (per plant entry):
    {
      "eval_window": { "start": "ISO", "end": "ISO", "days": 7 },
      "plants": {
        "<asset_id>": {
          "name": "<str>",
          "days_matched": <int>,        # days with both predicted + actual
          "NMAE_pct": <float|null>,
          "NRMSE_pct": <float|null>,
          "energy_error_pct": <float|null>,
          "sum_predicted_kwh": <float>,
          "sum_actual_kwh": <float>
        }
      },
      "fleet": {
        "NMAE_pct": <float|null>,
        "NRMSE_pct": <float|null>,
        "energy_error_pct": <float|null>
      },
      "generated_at": "ISO",
      "model_version": "<str>"
    }

GUARD RAILS
-----------
  - Needs ≥ 3 matched days per plant to emit meaningful metrics; fewer → metrics = null.
  - Actual values ≤ 0 excluded from denominator (guards against missing-meter days).
  - Job is a no-op if EVAL_PLANT_IDS is empty.
  - misfire_grace_time = 7200 s (2 h): if service was down at Sunday 02:00, fires
    if it restarts within 2 hours.
"""
from __future__ import annotations

import json
import logging
import math
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Optional, Tuple
from zoneinfo import ZoneInfo

from app.config import settings
from app.services.forecast_service import KEY_DAILY_ENERGY_EXPECTED, MODEL_VERSION

log = logging.getLogger(__name__)

# ── constants ────────────────────────────────────────────────────────────────

KEY_ACCURACY_REPORT = "pvlib_accuracy_report_json"
MIN_MATCHED_DAYS = 3        # minimum aligned days to produce non-null metrics
_MS_PER_DAY = 86_400_000    # used for timestamp boundary queries


# ── public entry point ───────────────────────────────────────────────────────

async def run_weekly_eval(tb_client=None) -> Dict[str, object]:
    """Entry point called by APScheduler and /admin/run-weekly.

    Parameters
    ----------
    tb_client : ThingsBoardClient, optional
        Authenticated singleton.  If None a new client is created.
    """
    if not settings.eval_plant_ids:
        log.info("weekly_eval: EVAL_PLANT_IDS is empty — skipping")
        return {"status": "skipped", "reason": "EVAL_PLANT_IDS empty"}

    from app.services.thingsboard_client import ThingsBoardClient

    tz = ZoneInfo(settings.TZ_LOCAL)
    now_local = datetime.now(tz)

    # Eval window: 7 completed days ending at local midnight today
    end_local = now_local.replace(hour=0, minute=0, second=0, microsecond=0)
    start_local = end_local - timedelta(days=7)
    end_utc   = end_local.astimezone(timezone.utc)
    start_utc = start_local.astimezone(timezone.utc)

    # Timestamp to stamp the report: start of the eval window (Monday midnight)
    report_ts_ms = int(start_local.timestamp() * 1000)

    log.info(
        "weekly_eval: evaluating %s → %s (%s → %s local)",
        start_utc.isoformat(), end_utc.isoformat(),
        start_local.date(), end_local.date(),
    )

    try:
        if tb_client is not None:
            return await _evaluate(tb_client, start_utc, end_utc,
                                   start_local, end_local, report_ts_ms)
        async with ThingsBoardClient(
            settings.TB_HOST, settings.TB_USERNAME, settings.TB_PASSWORD
        ) as tb:
            return await _evaluate(tb, start_utc, end_utc,
                                   start_local, end_local, report_ts_ms)
    except Exception as exc:
        log.exception("weekly_eval: failed: %s", exc)
        return {"status": "error", "error": str(exc)}


# ── internal ──────────────────────────────────────────────────────────────────

async def _evaluate(
    tb_client,
    start_utc: datetime,
    end_utc: datetime,
    start_local: datetime,
    end_local: datetime,
    report_ts_ms: int,
) -> Dict[str, object]:
    """Fetch data, compute metrics, write to TB."""

    plant_ids = settings.eval_plant_ids
    actual_key = settings.EVAL_ACTUAL_DAILY_KEY

    # Resolve asset names (best-effort; fall back to ID if TB lookup fails)
    plant_names = await _resolve_names(tb_client, plant_ids)

    all_errors: List[float] = []       # for fleet NMAE / NRMSE accumulation
    all_actuals: List[float] = []
    fleet_pred_sum = 0.0
    fleet_act_sum  = 0.0

    plant_results: Dict[str, dict] = {}

    for plant_id in plant_ids:
        try:
            pred_series, act_series = await _fetch_series(
                tb_client, plant_id, start_utc, end_utc, actual_key
            )
        except Exception as exc:
            log.error("weekly_eval: fetch failed for %s: %s", plant_id, exc)
            plant_results[plant_id] = {
                "name": plant_names.get(plant_id, plant_id),
                "error": str(exc),
                "days_matched": 0,
                "NMAE_pct": None,
                "NRMSE_pct": None,
                "energy_error_pct": None,
                "sum_predicted_kwh": None,
                "sum_actual_kwh": None,
            }
            continue

        matched = _align_series(pred_series, act_series)
        metrics = _compute_metrics(matched)

        for _, pred, act in matched:
            if act > 0:
                all_errors.append(abs(pred - act))
                all_actuals.append(act)
        fleet_pred_sum += metrics.get("sum_predicted_kwh", 0.0) or 0.0
        fleet_act_sum  += metrics.get("sum_actual_kwh", 0.0) or 0.0

        plant_results[plant_id] = {
            "name": plant_names.get(plant_id, plant_id),
            "days_matched": len(matched),
            **metrics,
        }
        log.info(
            "weekly_eval: %s (%s) — %d days matched, NMAE=%.1f%%, energy_err=%.1f%%",
            plant_id, plant_names.get(plant_id, "?"),
            len(matched),
            metrics.get("NMAE_pct") or float("nan"),
            metrics.get("energy_error_pct") or float("nan"),
        )

    # Fleet-level metrics
    fleet = _compute_metrics_from_lists(all_errors, all_actuals,
                                        fleet_pred_sum, fleet_act_sum)

    report = {
        "eval_window": {
            "start": start_local.isoformat(),
            "end":   end_local.isoformat(),
            "days":  7,
        },
        "plants": plant_results,
        "fleet": fleet,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "model_version": MODEL_VERSION,
    }

    report_json = json.dumps(report, default=str)
    await _write_report(tb_client, plant_ids, report_json, report_ts_ms)

    log.info(
        "weekly_eval: done — fleet NMAE=%.1f%% energy_err=%.1f%%",
        fleet.get("NMAE_pct") or float("nan"),
        fleet.get("energy_error_pct") or float("nan"),
    )
    return {"status": "ok", "fleet": fleet, "plants_evaluated": len(plant_ids)}


async def _resolve_names(tb_client, plant_ids: List[str]) -> Dict[str, str]:
    """Return {asset_id: name} for a list of plant IDs. Best-effort."""
    names: Dict[str, str] = {}
    for pid in plant_ids:
        try:
            info = await tb_client.get_asset_info(pid)
            if info:
                names[pid] = info.get("name", pid)
        except Exception:
            pass  # name resolution is cosmetic — never break eval on it
    return names


async def _fetch_series(
    tb_client,
    plant_id: str,
    start_utc: datetime,
    end_utc: datetime,
    actual_key: str,
) -> Tuple[List[Tuple[int, float]], List[Tuple[int, float]]]:
    """Return (predicted_series, actual_series).

    Each series is a list of (ts_ms, value_kwh) tuples with value > 0.
    """
    raw = await tb_client.get_timeseries(
        "ASSET", plant_id,
        [KEY_DAILY_ENERGY_EXPECTED, actual_key],
        start=start_utc,
        end=end_utc,
        limit=30,
    )

    def _parse(records, exclude_negative: bool = True):
        out = []
        for r in (records or []):
            try:
                v = float(r["value"])
                if exclude_negative and v < 0:
                    continue
                out.append((int(r["ts"]), v))
            except (KeyError, ValueError, TypeError):
                continue
        return out

    predicted = _parse(raw.get(KEY_DAILY_ENERGY_EXPECTED, []))
    actual    = _parse(raw.get(actual_key, []))
    return predicted, actual


def _align_series(
    predicted: List[Tuple[int, float]],
    actual:    List[Tuple[int, float]],
    tolerance_ms: int = 4 * 3_600_000,   # 4-hour slack for timestamp alignment
) -> List[Tuple[int, float, float]]:
    """Return [(day_ts_ms, pred_kwh, act_kwh)] for days present in both series.

    Daily roll-up timestamps are written at local midnight but the exact
    millisecond may drift slightly between jobs; the tolerance window handles this.
    """
    if not predicted or not actual:
        return []

    # Build lookup: round predicted ts to nearest half-day bucket
    act_lookup: Dict[int, float] = {}
    for ts, val in actual:
        bucket = _day_bucket(ts)
        act_lookup[bucket] = val

    matched = []
    for ts, pred_val in predicted:
        bucket = _day_bucket(ts)
        if bucket in act_lookup:
            act_val = act_lookup[bucket]
            if act_val > 0:   # skip zero/missing meter days
                matched.append((bucket, pred_val, act_val))

    return sorted(matched, key=lambda x: x[0])


def _day_bucket(ts_ms: int) -> int:
    """Round a timestamp down to the nearest 24-hour boundary (UTC)."""
    return (ts_ms // _MS_PER_DAY) * _MS_PER_DAY


def _compute_metrics(
    matched: List[Tuple[int, float, float]],
) -> Dict[str, object]:
    """Compute per-plant accuracy metrics from aligned (ts, pred, act) triples."""
    if len(matched) < MIN_MATCHED_DAYS:
        n = len(matched)
        sum_p = sum(p for _, p, _ in matched)
        sum_a = sum(a for _, _, a in matched)
        return {
            "NMAE_pct": None,
            "NRMSE_pct": None,
            "energy_error_pct": None,
            "sum_predicted_kwh": round(sum_p, 3) if matched else None,
            "sum_actual_kwh":    round(sum_a, 3) if matched else None,
        }

    errors  = [abs(p - a) for _, p, a in matched]
    sq_errs = [(p - a) ** 2 for _, p, a in matched]
    actuals = [a for _, _, a in matched]
    preds   = [p for _, p, _ in matched]

    mean_act = sum(actuals) / len(actuals)
    if mean_act <= 0:
        return {
            "NMAE_pct": None, "NRMSE_pct": None, "energy_error_pct": None,
            "sum_predicted_kwh": round(sum(preds), 3),
            "sum_actual_kwh":    round(sum(actuals), 3),
        }

    nmae  = (sum(errors) / len(errors)) / mean_act * 100
    nrmse = math.sqrt(sum(sq_errs) / len(sq_errs)) / mean_act * 100
    energy_err = (sum(preds) - sum(actuals)) / sum(actuals) * 100

    return {
        "NMAE_pct":         round(nmae, 2),
        "NRMSE_pct":        round(nrmse, 2),
        "energy_error_pct": round(energy_err, 2),
        "sum_predicted_kwh": round(sum(preds), 3),
        "sum_actual_kwh":    round(sum(actuals), 3),
    }


def _compute_metrics_from_lists(
    errors: List[float],
    actuals: List[float],
    sum_predicted: float,
    sum_actual: float,
) -> Dict[str, object]:
    """Compute fleet-level metrics from accumulated per-plant error lists."""
    if len(errors) < MIN_MATCHED_DAYS or not actuals:
        return {
            "NMAE_pct": None,
            "NRMSE_pct": None,
            "energy_error_pct": None,
            "sum_predicted_kwh": round(sum_predicted, 3),
            "sum_actual_kwh":    round(sum_actual, 3),
        }

    mean_act = sum(actuals) / len(actuals)
    if mean_act <= 0:
        return {
            "NMAE_pct": None, "NRMSE_pct": None, "energy_error_pct": None,
            "sum_predicted_kwh": round(sum_predicted, 3),
            "sum_actual_kwh":    round(sum_actual, 3),
        }

    nmae  = (sum(errors) / len(errors)) / mean_act * 100
    sq_errs = [(e ** 2) for e in errors]
    nrmse = math.sqrt(sum(sq_errs) / len(sq_errs)) / mean_act * 100

    if sum_actual > 0:
        energy_err = (sum_predicted - sum_actual) / sum_actual * 100
    else:
        energy_err = None

    return {
        "NMAE_pct":         round(nmae, 2) if nmae is not None else None,
        "NRMSE_pct":        round(nrmse, 2) if nrmse is not None else None,
        "energy_error_pct": round(energy_err, 2) if energy_err is not None else None,
        "sum_predicted_kwh": round(sum_predicted, 3),
        "sum_actual_kwh":    round(sum_actual, 3),
    }


async def _write_report(
    tb_client,
    plant_ids: List[str],
    report_json: str,
    report_ts_ms: int,
) -> None:
    """Write pvlib_accuracy_report_json to each eval plant and the first root asset."""
    payload = [{"ts": report_ts_ms, "values": {KEY_ACCURACY_REPORT: report_json}}]

    targets = list(plant_ids)
    if settings.root_asset_ids:
        targets.append(settings.root_asset_ids[0])  # fleet summary on root

    written = 0
    for entity_id in targets:
        try:
            await tb_client.post_telemetry("ASSET", entity_id, payload)
            written += 1
        except Exception as exc:
            log.error("weekly_eval: write failed for %s: %s", entity_id, exc)

    log.info("weekly_eval: wrote %s to %d/%d entities",
             KEY_ACCURACY_REPORT, written, len(targets))
