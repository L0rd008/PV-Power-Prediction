#!/usr/bin/env python3
"""
validate_pvlib.py - Phase 1 + Phase 3 accuracy benchmark
=========================================================

Compares:
  1. Old model  : scripts/python_physics/physics_model.py  (globals patched)
  2. New model  : scripts/Pvlib-Service/app/physics/pipeline.py  (H-A3)

against real ThingsBoard active_power telemetry for Kebithigollewa 10MWac.

Default irradiance: NASA POWER hourly (free, no API key).

USAGE:
  python validate_pvlib.py --jwt YOUR_JWT_TOKEN
  python validate_pvlib.py --tb-user admin@x.com --tb-pass secret
  python validate_pvlib.py --jwt ... --start 2024-06-01 --end 2024-08-31
  python validate_pvlib.py --actual-csv /path/to/power.csv
  python validate_pvlib.py --jwt ... --weather-csv output/validation/nasa_power_weather.csv
  python validate_pvlib.py --jwt ... --no-old-model
  python validate_pvlib.py --jwt ... --no-plot

Phase 1 gate: |new_energy_err% - old_energy_err%| <= 0.5%

OUTPUTS (output/validation/):
  model_comparison.csv   - hourly predictions + actual
  metrics.json           - MAPE, MAE, RMSE, bias, energy error per model
  validation_plot.png    - time-series + scatter (requires matplotlib)
  actual_power.csv       - cached TB telemetry
  nasa_power_weather.csv - cached NASA POWER data
"""

import argparse
import json
import sys
import importlib.util
import requests
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime, timezone, timedelta

# ------------------------------------------------------------------------------
# Directory layout
# ------------------------------------------------------------------------------

SCRIPT_DIR    = Path(__file__).parent.resolve()
PROJECT_DIR   = SCRIPT_DIR.parent.parent
CONFIG_DIR    = PROJECT_DIR / "config"
OUTPUT_DIR    = PROJECT_DIR / "output" / "validation"
OLD_MODEL_DIR = PROJECT_DIR / "scripts" / "python_physics"
NEW_MODEL_DIR = PROJECT_DIR / "scripts" / "Pvlib-Service" / "app" / "physics"

KEBITH_CONFIG = CONFIG_DIR / "kebithigollewa_pvlib_config.json"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ------------------------------------------------------------------------------
# ThingsBoard helpers
# ------------------------------------------------------------------------------

def tb_fetch_jwt(tb_host, username, password):
    """Exchange username/password for a ThingsBoard JWT token."""
    r = requests.post(
        f"{tb_host}/api/auth/login",
        json={"username": username, "password": password},
        timeout=15,
    )
    r.raise_for_status()
    return r.json()["token"]


def tb_fetch_timeseries(tb_host, entity_type, entity_id, keys,
                        start_ms, end_ms, jwt, limit=50_000):
    """
    Fetch historical timeseries from a ThingsBoard entity via JWT.

    entity_type : 'DEVICE' or 'ASSET'
    """
    headers = {"X-Authorization": f"Bearer {jwt}"}
    url = (
        f"{tb_host}/api/plugins/telemetry/{entity_type}/{entity_id}/values/timeseries"
        f"?keys={','.join(keys)}&startTs={start_ms}&endTs={end_ms}&limit={limit}"
        f"&agg=NONE&orderBy=ASC"
    )
    r = requests.get(url, headers=headers, timeout=60)
    r.raise_for_status()
    return r.json()


def parse_tb_timeseries(raw, key):
    """Convert TB API response dict to a pandas Series with UTC DatetimeIndex."""
    rows = raw.get(key, [])
    if not rows:
        return pd.Series(dtype=float, name=key)
    ts   = [int(r["ts"])      for r in rows]
    vals = [float(r["value"]) for r in rows]
    idx  = pd.to_datetime(ts, unit="ms", utc=True)
    return pd.Series(vals, index=idx, name=key).sort_index()


def to_hourly_mean(series):
    """Aggregate a power series to hourly mean values."""
    if series is None:
        return None
    return series.sort_index().resample("1h").mean()


def repair_legacy_nasa_power_index(df):
    """
    Repair cached NASA POWER CSV files generated with the legacy %Y%m%d%H%M
    parser bug, which compressed hourly keys into a few clock hours per day.
    """
    if df is None or len(df.index) < 24:
        return df

    idx = pd.DatetimeIndex(df.index)
    if idx.tz is None:
        idx = idx.tz_localize("UTC")

    looks_legacy = (
        idx.hour.max() <= 2
        and idx.minute.max() <= 9
        and idx.second.max() == 0
        and idx.microsecond.max() == 0
    )
    if not looks_legacy:
        return df

    repaired = df.copy()
    repaired.index = pd.date_range(
        start=idx.min().normalize(),
        periods=len(df),
        freq="1h",
        tz="UTC",
    )
    print("  Repaired legacy NASA POWER hourly timestamps from cached CSV")
    return repaired


def load_indexed_timeseries_csv(path):
    """Load a cached CSV with a timestamp index in mixed ISO formats."""
    df = pd.read_csv(path, index_col=0)
    df.index = pd.to_datetime(df.index, utc=True, format="mixed")
    return df.sort_index()


# ------------------------------------------------------------------------------
# Irradiance: NASA POWER hourly (free, no API key)
# ------------------------------------------------------------------------------

def fetch_nasa_power(lat, lon, start, end):
    """
    Fetch hourly GHI/DHI/DNI/temperature/wind from NASA POWER API.
    Returns DataFrame with UTC DatetimeIndex; columns: ghi, dhi, dni,
    air_temp, wind_speed.
    """
    fmt = "%Y%m%d"
    url = (
        "https://power.larc.nasa.gov/api/temporal/hourly/point"
        "?parameters=ALLSKY_SFC_SW_DWN,ALLSKY_SFC_SW_DIFF,ALLSKY_SFC_SW_DNI,T2M,WS2M"
        f"&community=RE&longitude={lon}&latitude={lat}"
        f"&start={start.strftime(fmt)}&end={end.strftime(fmt)}&format=JSON"
    )
    print(f"  Fetching NASA POWER: {start.date()} -> {end.date()} ...")
    r = requests.get(url, timeout=180)
    r.raise_for_status()
    data = r.json()["properties"]["parameter"]

    records, timestamps = [], []
    for ts_str, ghi in data["ALLSKY_SFC_SW_DWN"].items():
        dt   = datetime.strptime(ts_str, "%Y%m%d%H").replace(tzinfo=timezone.utc)
        dhi  = data["ALLSKY_SFC_SW_DIFF"].get(ts_str, 0.0)
        dni  = data["ALLSKY_SFC_SW_DNI"].get(ts_str, 0.0)
        temp = data["T2M"].get(ts_str, 25.0)
        wind = data["WS2M"].get(ts_str, 1.0)
        timestamps.append(dt)
        records.append({
            "ghi":        max(float(ghi), 0.0),
            "dhi":        max(float(dhi), 0.0),
            "dni":        max(float(dni), 0.0),
            "air_temp":   float(temp),
            "wind_speed": float(wind),
        })

    df = pd.DataFrame(records, index=pd.DatetimeIndex(timestamps, tz="UTC"))
    df = df.sort_index()
    print(f"  NASA POWER: {len(df)} hourly rows")
    return df


# ------------------------------------------------------------------------------
# Old model runner
# ------------------------------------------------------------------------------

def run_old_model(keb_cfg, weather_df):
    """
    Load physics_model.py dynamically, patch its module-level globals with
    Kebithigollewa parameters, then call compute_pv_ac(weather_df).
    """
    import pvlib

    spec = importlib.util.spec_from_file_location(
        "physics_model", OLD_MODEL_DIR / "physics_model.py"
    )
    mod = importlib.util.module_from_spec(spec)
    sys.path.insert(0, str(OLD_MODEL_DIR))
    try:
        spec.loader.exec_module(mod)
    finally:
        if str(OLD_MODEL_DIR) in sys.path:
            sys.path.remove(str(OLD_MODEL_DIR))

    loc   = keb_cfg["location"]
    m     = keb_cfg["module"]
    inv   = keb_cfg["inverter"]
    loss  = keb_cfg["losses"]
    iam_c = keb_cfg["iam"]
    defs  = keb_cfg.get("defaults", {})

    mod.lat             = loc["lat"]
    mod.lon             = loc["lon"]
    mod.SITE_ALTITUDE_M = loc.get("altitude_m", 0)
    mod.orientations    = keb_cfg["orientations"]

    mod.module_area           = m["area_m2"]
    mod.total_module_area     = sum(
        o["module_count"] * m["area_m2"] for o in keb_cfg["orientations"]
    )
    mod.module_efficiency_stc = m["efficiency_stc"]
    mod.gamma_p               = m["gamma_p"]

    mod.INV_AC_RATING_KW = inv["ac_rating_kw"]
    mod.PDC_THRESHOLD_KW = inv["dc_threshold_kw"]

    soiling = loss.get("soiling", 0.0)
    lid     = loss.get("lid", 0.0)
    mq      = loss.get("module_quality", 0.0)
    mm      = loss.get("mismatch", 0.0)
    dcw     = loss.get("dc_wiring", 0.0)
    mod.ac_wiring_loss = loss.get("ac_wiring", 0.0)
    mod.albedo         = loss.get("albedo", 0.20)
    mod.far_shading    = loss.get("far_shading", 1.0)
    # PVsyst convention: negative mq = quality gain
    mod.dc_loss_factor = (1.0 - soiling) * (1.0 - lid) * (1.0 - mq) * (1.0 - mm) * (1.0 - dcw)

    mod.iam_angles = np.array(iam_c["angles"])
    mod.iam_values = np.array(iam_c["values"])

    thermal_key = keb_cfg.get("thermal_model", "open_rack_glass_glass")
    mod.sapm_params = pvlib.temperature.TEMPERATURE_MODEL_PARAMETERS["sapm"][thermal_key]

    if inv.get("use_efficiency_curve", False):
        curve_kw  = np.array(inv["efficiency_curve_kw"])
        curve_eta = np.array(inv["efficiency_curve_eta"])
        def _eff_curve(pdc_kw):
            return np.interp(pdc_kw, curve_kw, curve_eta)
        mod.inverter_efficiency = _eff_curve
    else:
        flat_eta = inv.get("flat_efficiency", 0.98)
        def _eff_flat(pdc_kw):
            return flat_eta
        mod.inverter_efficiency = _eff_flat

    mod.DEFAULT_WIND_SPEED_MS = defs.get("wind_speed_ms", 1.0)
    mod.DEFAULT_AIR_TEMP_C    = defs.get("air_temp_c", 25.0)

    print("  Patched globals, calling compute_pv_ac() ...")
    ac = mod.compute_pv_ac(weather_df.copy())
    return ac.clip(lower=0.0)


# ------------------------------------------------------------------------------
# New model runner
# ------------------------------------------------------------------------------

def run_new_model(keb_cfg, weather_df):
    """
    Import the H-A3 pipeline and run compute_ac_power() with the Kebithigollewa
    PlantConfig. Returns a Series of AC power in kW.
    """
    service_root = str(NEW_MODEL_DIR.parent.parent)   # scripts/Pvlib-Service
    sys.path.insert(0, service_root)
    try:
        from app.physics.config import PlantConfig
        from app.physics.pipeline import compute_ac_power
    finally:
        if service_root in sys.path:
            sys.path.remove(service_root)

    config = PlantConfig.model_validate(keb_cfg)
    result = compute_ac_power(config, weather_df.copy())
    return result["ac_kw"].clip(lower=0.0)


# ------------------------------------------------------------------------------
# Metrics
# ------------------------------------------------------------------------------

def compute_metrics(actual, predicted, label="model"):
    """Compute MAE, RMSE, MAPE, bias, and energy error."""
    aligned = pd.concat(
        [actual.rename("actual"), predicted.rename("pred")], axis=1
    ).dropna()

    if len(aligned) < 10:
        print(f"  WARNING: {label}: only {len(aligned)} aligned points")

    a   = aligned["actual"]
    p   = aligned["pred"]
    err = p - a

    with np.errstate(divide="ignore", invalid="ignore"):
        pct_errs = (np.abs(err) / a.where(a > 0)).replace([np.inf, -np.inf], np.nan)
        mape = float(pct_errs.mean() * 100)

    mae  = float(np.abs(err).mean())
    rmse = float(np.sqrt((err ** 2).mean()))
    bias = float(err.mean())

    e_actual    = float(a.sum())
    e_predicted = float(p.sum())
    energy_err_pct = (
        (e_predicted - e_actual) / e_actual * 100
        if e_actual > 0 else float("nan")
    )

    return {
        "label":                label,
        "n_points":             len(aligned),
        "mape_pct":             round(mape, 3),
        "mae_kw":               round(mae, 2),
        "rmse_kw":              round(rmse, 2),
        "bias_kw":              round(bias, 2),
        "energy_actual_kwh":    round(e_actual, 1),
        "energy_predicted_kwh": round(e_predicted, 1),
        "energy_error_pct":     round(energy_err_pct, 3),
    }


def phase1_gate(old_m, new_m):
    """Phase 1 gate: |delta_energy_err| <= 0.5 percentage points."""
    delta  = abs(new_m["energy_error_pct"] - old_m["energy_error_pct"])
    passed = delta <= 0.5
    marker = "PASS" if passed else "FAIL"
    print(f"\n  Phase 1 gate [{marker}]:  |delta_energy_err| = {delta:.3f}%  (threshold <= 0.5%)")
    return passed


# ------------------------------------------------------------------------------
# Plot
# ------------------------------------------------------------------------------

def make_plot(actual, old_pred, new_pred, out_path):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import matplotlib.dates as mdates
    except ImportError:
        print("  matplotlib not available - skipping plot")
        return

    fig, axes = plt.subplots(2, 1, figsize=(16, 10))

    ax = axes[0]
    if actual is not None:
        ax.plot(actual.index, actual.values, color="black", lw=1.0,
                label="Actual (ThingsBoard)", alpha=0.85)
    if old_pred is not None:
        ax.plot(old_pred.index, old_pred.values, color="#e74c3c", lw=0.8,
                label="Old model (physics_model.py)", alpha=0.70)
    ax.plot(new_pred.index, new_pred.values, color="#3498db", lw=0.8,
            label="New model (H-A3 pipeline)", alpha=0.70)
    ax.set_ylabel("AC Power (kW)")
    ax.set_title("Kebithigollewa 10 MWac - Actual vs Predicted")
    ax.legend(loc="upper right")
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d"))
    fig.autofmt_xdate()
    ax.grid(True, alpha=0.3)

    ax2 = axes[1]
    if actual is not None:
        aligned = pd.concat(
            [actual.rename("a"), new_pred.rename("p")], axis=1
        ).dropna()
        if len(aligned) > 0:
            ax2.scatter(aligned["a"], aligned["p"], s=3, alpha=0.3, color="#3498db",
                        label="New model vs Actual")
            lim = max(aligned["a"].max(), aligned["p"].max()) * 1.05
            ax2.plot([0, lim], [0, lim], "k--", lw=0.8, label="1:1 line")
    ax2.set_xlabel("Actual AC Power (kW)")
    ax2.set_ylabel("Predicted AC Power (kW)")
    ax2.set_title("Scatter - New Model vs Actual")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Plot saved: {out_path}")


# ------------------------------------------------------------------------------
# Main
# ------------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(
        description="Validate old vs new pvlib model against ThingsBoard actual power",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    ap.add_argument("--tb-host",      help="ThingsBoard base URL (overrides config _tb_host)")
    ap.add_argument("--jwt",          help="Tenant JWT token")
    ap.add_argument("--tb-user",      help="TB username (to obtain JWT)")
    ap.add_argument("--tb-pass",      help="TB password (to obtain JWT)")
    ap.add_argument("--start",        default="2024-01-01",
                    help="Window start YYYY-MM-DD (default 2024-01-01)")
    ap.add_argument("--end",          default="2024-03-31",
                    help="Window end YYYY-MM-DD (default 2024-03-31)")
    ap.add_argument("--actual-csv",   help="Pre-exported actual power CSV (skip TB fetch)")
    ap.add_argument("--weather-csv",  help="Pre-fetched weather CSV (skip NASA POWER fetch)")
    ap.add_argument("--config",       default=str(KEBITH_CONFIG),
                    help="Path to pvlib plant config JSON")
    ap.add_argument("--no-old-model", action="store_true",
                    help="Skip old model (benchmark new model only)")
    ap.add_argument("--no-plot",      action="store_true",
                    help="Skip matplotlib plot")
    args = ap.parse_args()

    print("\n" + "=" * 70)
    print("   KEBITHIGOLLEWA PVLIB VALIDATION  -  OLD vs NEW MODEL")
    print("=" * 70)

    config_path = Path(args.config)
    if not config_path.exists():
        sys.exit(f"Config not found: {config_path}")
    with open(config_path) as f:
        keb_cfg = json.load(f)

    lat = keb_cfg["location"]["lat"]
    lon = keb_cfg["location"]["lon"]

    # TB entity routing:
    # - Preferred: KSP_Plant ASSET / EnergyMeter_active_power (kW)
    # - Fallback:  KSP_P341 DEVICE / p341_active_power (W * 0.001 = kW)
    tb_host = args.tb_host or keb_cfg.get("_tb_host", "")

    asset_id        = keb_cfg.get("_tb_asset_id")
    asset_pw_key    = keb_cfg.get("_tb_plant_active_power_key", "EnergyMeter_active_power")
    asset_pw_scale  = 1.0

    device_id       = keb_cfg.get("_tb_device_id")
    device_pw_key   = keb_cfg.get("_tb_active_power_key", "p341_active_power")
    device_pw_scale = float(keb_cfg.get("_tb_active_power_scale", 0.001))

    print(f"\nPlant  : {keb_cfg.get('plant_name')}")
    print(f"Config : {config_path}")
    print(f"TB host: {tb_host or '(not set)'}")
    print(f"Asset  : {asset_id}  key={asset_pw_key} (kW)")
    print(f"Device : {device_id}  key={device_pw_key} (W, x{device_pw_scale}=kW)")

    start_dt = datetime.strptime(args.start, "%Y-%m-%d").replace(tzinfo=timezone.utc)
    end_dt   = datetime.strptime(args.end,   "%Y-%m-%d").replace(tzinfo=timezone.utc)
    start_ms = int(start_dt.timestamp() * 1000)
    end_ms   = int((end_dt + timedelta(days=1)).timestamp() * 1000)
    print(f"Window : {args.start} -> {args.end}")

    # -- actual power ----------------------------------------------------------
    actual = None

    if args.actual_csv:
        print(f"\nLoading actual power: {args.actual_csv}")
        raw = load_indexed_timeseries_csv(args.actual_csv)
        actual = raw.iloc[:, 0].rename("active_power").astype(float)
        print(f"  {len(actual)} rows loaded")

    elif tb_host:
        jwt = args.jwt
        if not jwt and args.tb_user and args.tb_pass:
            print(f"\nObtaining JWT from {tb_host} ...")
            jwt = tb_fetch_jwt(tb_host, args.tb_user, args.tb_pass)
            print("  JWT obtained")

        if jwt:
            # Preferred: plant ASSET (EnergyMeter_active_power, kW)
            if asset_id:
                print(f"\nFetching {asset_pw_key} from KSP_Plant ASSET ...")
                try:
                    raw_ts = tb_fetch_timeseries(
                        tb_host, "ASSET", asset_id, [asset_pw_key],
                        start_ms, end_ms, jwt,
                    )
                    s = parse_tb_timeseries(raw_ts, asset_pw_key)
                    if len(s) == 0:
                        raise ValueError("empty response")
                    actual = (s * asset_pw_scale).rename("active_power_kw")
                    print(f"  {len(actual)} data points (kW)")
                except Exception as exc:
                    print(f"  WARNING: ASSET fetch failed ({exc}), falling back to device ...")
                    actual = None

            # Fallback: KSP_P341 DEVICE (W -> kW)
            if actual is None and device_id:
                print(f"\nFetching {device_pw_key} from KSP_P341 DEVICE ...")
                raw_ts = tb_fetch_timeseries(
                    tb_host, "DEVICE", device_id, [device_pw_key],
                    start_ms, end_ms, jwt,
                )
                s = parse_tb_timeseries(raw_ts, device_pw_key)
                actual = (s * device_pw_scale).rename("active_power_kw")
                print(f"  {len(actual)} data points (W -> kW, x{device_pw_scale})")

            if actual is not None and len(actual) > 0:
                cache_path = OUTPUT_DIR / "actual_power.csv"
                actual.to_csv(cache_path)
                print(f"  Cached: {cache_path}")
            else:
                print("  WARNING: No actual power data retrieved.")
        else:
            print("\n  WARNING: No JWT/credentials provided - skipping TB fetch.")
            print("     Re-run with --jwt or --tb-user + --tb-pass, or use --actual-csv.")

    else:
        print("\n  WARNING: No TB host configured - skipping actual power fetch.")
        print("     Models will be compared against each other only.")

    # -- weather ---------------------------------------------------------------
    if args.weather_csv:
        print(f"\nLoading weather: {args.weather_csv}")
        weather_raw = load_indexed_timeseries_csv(args.weather_csv)
        weather_df = repair_legacy_nasa_power_index(weather_raw)
        if not weather_df.index.equals(weather_raw.index):
            weather_df.to_csv(args.weather_csv)
            print(f"  Re-cached repaired weather CSV: {args.weather_csv}")
        print(f"  {len(weather_df)} rows loaded")
    else:
        print("\nFetching weather (NASA POWER) ...")
        weather_df = fetch_nasa_power(lat, lon, start_dt, end_dt)
        wx_path = OUTPUT_DIR / "nasa_power_weather.csv"
        weather_df.to_csv(wx_path)
        print(f"  Cached: {wx_path}")

    w_mask = (
        (weather_df.index >= start_dt)
        & (weather_df.index < end_dt + timedelta(days=1))
    )
    weather_df = weather_df[w_mask]
    print(f"  Weather window: {len(weather_df)} hourly rows")

    # -- run NEW model ---------------------------------------------------------
    print("\nRunning NEW model (H-A3 pipeline) ...")
    new_pred = run_new_model(keb_cfg, weather_df)
    print(f"  {len(new_pred)} timestamps  |  peak = {new_pred.max():.1f} kW"
          f"  |  total = {new_pred.sum()/1000:.1f} MWh")

    # -- run OLD model ---------------------------------------------------------
    old_pred = None
    if not args.no_old_model:
        print("\nRunning OLD model (physics_model.py) ...")
        try:
            old_pred = run_old_model(keb_cfg, weather_df)
            print(f"  {len(old_pred)} timestamps  |  peak = {old_pred.max():.1f} kW"
                  f"  |  total = {old_pred.sum()/1000:.1f} MWh")
        except Exception as exc:
            print(f"  WARNING: Old model failed: {exc}")
            old_pred = None

    # -- metrics ---------------------------------------------------------------
    print("\n" + "=" * 70)
    print("  METRICS")
    print("=" * 70)

    actual_h = to_hourly_mean(actual) if actual is not None and len(actual) > 0 else None
    new_pred_h = to_hourly_mean(new_pred)
    old_pred_h = to_hourly_mean(old_pred) if old_pred is not None else None

    metrics_out = {}
    fmt = (
        "  {label:<30}  MAPE={mape_pct:.2f}%  MAE={mae_kw:,.0f} kW"
        "  RMSE={rmse_kw:,.0f} kW  Bias={bias_kw:+,.0f} kW"
        "  Energy: {energy_actual_kwh:,.0f}->{energy_predicted_kwh:,.0f} kWh"
        "  ({energy_error_pct:+.2f}%)"
    )

    if actual_h is not None and len(actual_h) > 0:
        new_m = compute_metrics(actual_h, new_pred_h, "New model (H-A3)")
        metrics_out["new_model"] = new_m
        print(fmt.format(**new_m))

        if old_pred_h is not None:
            old_m = compute_metrics(actual_h, old_pred_h, "Old model")
            metrics_out["old_model"] = old_m
            print(fmt.format(**old_m))
            phase1_gate(old_m, new_m)
        else:
            print("\n  (Old model not available - Phase 1 gate skipped)")

    else:
        print("  No actual power data - comparing models against each other.")
        if old_pred is not None:
            delta = (new_pred - old_pred).dropna()
            e_new = float(new_pred.sum())
            e_old = float(old_pred.sum())
            diff_pct = (e_new - e_old) / e_old * 100 if e_old else float("nan")
            print(f"\n  New vs Old:")
            print(f"    Mean diff  : {delta.mean():+.2f} kW")
            print(f"    Max |diff| : {delta.abs().max():.2f} kW")
            print(f"    Energy new : {e_new/1000:,.1f} MWh")
            print(f"    Energy old : {e_old/1000:,.1f} MWh")
            print(f"    delta energy: {diff_pct:+.3f}%")
            delta_e = abs(diff_pct)
            marker  = "PASS" if delta_e <= 0.5 else "FAIL"
            print(f"\n  Phase 1 gate [{marker}]:  |delta_energy| = {delta_e:.3f}%  (threshold <= 0.5%)")

    targets = keb_cfg.get("_validation_targets", {})
    if targets and actual is not None:
        print(f"\n  PVsyst P50 targets (soiling excluded from PVsyst):")
        print(f"    Annual E_Grid  : {targets.get('annual_e_grid_kwh', '?'):,} kWh")
        print(f"    Annual PR      : {targets.get('annual_pr', '?'):.1%}")
        print(f"    Specific yield : {targets.get('annual_specific_prod_kwh_kwp', '?'):,} kWh/kWp")

    # -- save ------------------------------------------------------------------
    print(f"\nSaving to {OUTPUT_DIR} ...")

    out_frames = {"new_model_kw": new_pred_h}
    if old_pred_h is not None:
        out_frames["old_model_kw"] = old_pred_h
    if actual_h is not None:
        out_frames["actual_kw"] = actual_h
    pd.DataFrame(out_frames).to_csv(OUTPUT_DIR / "model_comparison.csv")
    print("  model_comparison.csv")

    if metrics_out:
        with open(OUTPUT_DIR / "metrics.json", "w") as f:
            json.dump(metrics_out, f, indent=2)
        print("  metrics.json")

    if not args.no_plot:
        make_plot(actual_h, old_pred_h, new_pred_h, OUTPUT_DIR / "validation_plot.png")

    print("\n" + "=" * 70)
    print("  DONE")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()
