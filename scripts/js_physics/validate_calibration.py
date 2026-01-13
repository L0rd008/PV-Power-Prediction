#!/usr/bin/env python3
"""
================================================================================
VALIDATE JS PHYSICS MODEL CALIBRATION
================================================================================

Validates the calibrated JS physics model against the Python pvlib reference
using holdout test data or fresh samples.

Produces:
- Accuracy metrics (RMSE, MAE, R², energy error)
- Scatter plot: JS vs Python predictions
- Residual distribution histogram
- Time series comparison plot

USAGE:
------
  # Validate using calibration data
  python validate_calibration.py

  # Validate with specific data file
  python validate_calibration.py --data ../../data/js_calibration_data.csv

  # Use separate test set
  python validate_calibration.py --data ../../data/test_data.csv

================================================================================
"""

import argparse
import json
import sys
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

# =====================================================================
# PATHS
# =====================================================================

SCRIPT_DIR = Path(__file__).parent
PROJECT_DIR = SCRIPT_DIR.parent.parent
CONFIG_DIR = PROJECT_DIR / "config"
DATA_DIR = PROJECT_DIR / "data"
OUTPUT_DIR = PROJECT_DIR / "output"


# =====================================================================
# CONFIGURATION
# =====================================================================

def load_config():
    """Load plant configuration."""
    with open(CONFIG_DIR / "plant_config.json") as f:
        return json.load(f)


def load_calibration():
    """Load calibration results."""
    calib_path = OUTPUT_DIR / "js_physics_calibration.json"
    if not calib_path.exists():
        return None
    with open(calib_path) as f:
        return json.load(f)


# =====================================================================
# JS MODEL REIMPLEMENTATION (with calibrated parameters)
# =====================================================================

DEG2RAD = np.pi / 180
RAD2DEG = 180 / np.pi


def clip(v, vmin, vmax):
    return np.clip(v, vmin, vmax)


def calc_dni_extra(timestamps):
    """Calculate extraterrestrial DNI (Spencer's formula)."""
    day_of_year = timestamps.dayofyear
    b = 2 * np.pi * day_of_year / 365
    dni_extra = 1367 * (1.00011 + 0.034221 * np.cos(b) + 0.00128 * np.sin(b)
                       + 0.000719 * np.cos(2*b) + 0.000077 * np.sin(2*b))
    return dni_extra


def solar_pos_js(timestamps, lat, lon):
    """Simplified solar position (JS algorithm)."""
    ts_unix = timestamps.astype(np.int64) / 1e9
    jd = ts_unix / 86400 + 2440587.5
    jc = (jd - 2451545) / 36525
    
    m = (357.52911 + jc * 35999.05029) * DEG2RAD
    c = ((1.914602 - jc * 0.004817) * np.sin(m) + 
         0.019993 * np.sin(2 * m) + 
         0.000289 * np.sin(3 * m))
    sun_lon = (280.46646 + jc * 36000.76983 + c) * DEG2RAD
    
    obl = (23.439291 - jc * 0.0130042) * DEG2RAD
    dec = np.arcsin(np.sin(obl) * np.sin(sun_lon))
    
    eot = 4 * (sun_lon * RAD2DEG - np.arctan2(
        np.cos(obl) * np.sin(sun_lon),
        np.cos(sun_lon)
    ) * RAD2DEG)
    
    utc_h = timestamps.hour + timestamps.minute / 60 + timestamps.second / 3600
    solar_t = utc_h + lon / 15 + eot / 60
    ha = (solar_t - 12) * 15 * DEG2RAD
    
    lat_r = lat * DEG2RAD
    sin_el = np.sin(lat_r) * np.sin(dec) + np.cos(lat_r) * np.cos(dec) * np.cos(ha)
    el = np.arcsin(clip(sin_el, -1, 1))
    
    cos_az = (np.sin(dec) - np.sin(lat_r) * sin_el) / (np.cos(lat_r) * np.cos(el))
    az = np.arccos(clip(cos_az, -1, 1))
    az = np.where(ha > 0, 2 * np.pi - az, az)
    
    return {'zen': 90 - el * RAD2DEG, 'az': az * RAD2DEG, 'el': el * RAD2DEG}


def interp_iam(aoi, iam_angles, iam_values):
    return np.interp(aoi, iam_angles, iam_values)


def perez_poa_js(ghi, dni, dhi, sun, tilt, azim, albedo, dni_extra, params):
    """Calibrated Perez POA approximation."""
    zen_r = sun['zen'] * DEG2RAD
    tilt_r = tilt * DEG2RAD
    az_r = azim * DEG2RAD
    sun_az_r = sun['az'] * DEG2RAD
    
    cos_aoi = (np.cos(zen_r) * np.cos(tilt_r) +
               np.sin(zen_r) * np.sin(tilt_r) * np.cos(sun_az_r - az_r))
    cos_aoi = clip(cos_aoi, -1, 1)
    aoi = np.arccos(cos_aoi) * RAD2DEG
    
    beam = np.where((cos_aoi > 0) & (sun['el'] > 0), dni * cos_aoi, 0)
    
    f = 0.5 + 0.5 * np.cos(tilt_r)
    diff = dhi * f * params.get('diffuse_weight', 1.0)
    
    kt = np.where(dni_extra > 0, ghi / dni_extra, 0)
    kt = clip(kt, 0, 1.2)
    brightness_adj = 1.0 + params.get('brightness_factor', 1.0) * (kt - 0.5)
    brightness_adj = clip(brightness_adj, 0.5, 1.5)
    
    circumsolar_threshold = params.get('circumsolar_threshold', 50)
    aoi_threshold = params.get('aoi_threshold', 0.087)
    circumsolar_factor = params.get('circumsolar_factor', 0.2)
    
    cos_zen_safe = np.maximum(aoi_threshold, np.cos(zen_r))
    circumsolar_mask = (dni > circumsolar_threshold) & (cos_aoi > aoi_threshold)
    circum = np.where(
        circumsolar_mask,
        dhi * circumsolar_factor * brightness_adj * (cos_aoi / cos_zen_safe),
        0
    )
    diff = diff + circum
    
    ground = ghi * albedo * (1 - np.cos(tilt_r)) * 0.5
    poa = np.maximum(0, beam + diff + ground)
    
    return {'poa': poa, 'aoi': aoi}


def calc_pv_js(df, config, params):
    """JS physics model with calibrated parameters."""
    lat = config["location"]["lat"]
    lon = config["location"]["lon"]
    orientations = config["orientations"]
    module_area = config["module"]["area_m2"]
    module_efficiency = config["module"]["efficiency_stc"]
    gamma_p = config["module"]["gamma_p"]
    inv_ac_rating = config["inverter"]["ac_rating_kw"]
    inv_eff = config["inverter"]["flat_efficiency"]
    albedo = config["losses"]["albedo"]
    far_shading = config["losses"].get("far_shading", 1.0)
    
    losses = config["losses"]
    dc_loss_factor = (
        (1 - losses["soiling"]) *
        (1 - losses["lid"]) *
        (1 + losses["module_quality"]) *
        (1 - losses["mismatch"]) *
        (1 - losses["dc_wiring"]) *
        far_shading
    )
    ac_wiring_loss = losses["ac_wiring"]
    
    iam_angles = np.array(config["iam"]["angles"])
    iam_values = np.array(config["iam"]["values"])
    total_module_area = sum(o["module_count"] * module_area for o in orientations)
    
    ghi = df["ghi"].values
    dni = df["dni"].values
    dhi = df["dhi"].values
    air_temp = df["air_temp"].values
    wind_speed = df["wind_speed"].values
    
    sun = solar_pos_js(df.index, lat, lon)
    dni_extra = calc_dni_extra(df.index)
    
    total_dc = np.zeros(len(df))
    
    sapm_a = params.get('sapm_a', -3.56)
    sapm_b = params.get('sapm_b', -0.075)
    sapm_dt = params.get('sapm_dt', 3)
    
    for o in orientations:
        tilt = o["tilt"]
        azimuth = o["azimuth"]
        area_fraction = (o["module_count"] * module_area) / total_module_area
        
        poa_result = perez_poa_js(ghi, dni, dhi, sun, tilt, azimuth, albedo, dni_extra, params)
        poa = poa_result['poa'] * far_shading
        aoi = poa_result['aoi']
        
        iam = interp_iam(aoi, iam_angles, iam_values)
        poa_optical = poa * iam
        
        e0 = poa / 1000
        t_cell = air_temp + sapm_a * e0 + sapm_b * e0 * wind_speed + sapm_dt
        
        dc_kw_m2 = poa_optical * module_efficiency / 1000
        dc_kw_m2 = dc_kw_m2 * (1 + gamma_p * (t_cell - 25))
        dc_kw_m2 = dc_kw_m2 * dc_loss_factor
        
        total_dc += dc_kw_m2 * total_module_area * area_fraction
    
    ac = np.minimum(total_dc * inv_eff, inv_ac_rating)
    ac = ac * (1 - ac_wiring_loss)
    ac = np.where((ghi < 1) | (sun['el'] < 0), 0, ac)
    
    return ac


# =====================================================================
# VALIDATION
# =====================================================================

def compute_metrics(js_ac, py_ac, inv_rating):
    """Compute validation metrics."""
    rmse = np.sqrt(np.mean((js_ac - py_ac) ** 2))
    mae = np.mean(np.abs(js_ac - py_ac))
    
    ss_res = np.sum((js_ac - py_ac) ** 2)
    ss_tot = np.sum((py_ac - np.mean(py_ac)) ** 2)
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0
    
    total_py = py_ac.sum()
    total_js = js_ac.sum()
    energy_error_pct = 100 * abs(total_py - total_js) / total_py if total_py > 0 else 0
    
    # MAPE (excluding near-zero values)
    nonzero_mask = py_ac > 1.0
    if nonzero_mask.sum() > 0:
        mape = 100 * np.mean(np.abs((py_ac[nonzero_mask] - js_ac[nonzero_mask]) / py_ac[nonzero_mask]))
    else:
        mape = np.nan
    
    return {
        'rmse_kw': round(rmse, 4),
        'rmse_pct': round(100 * rmse / inv_rating, 3),
        'mae_kw': round(mae, 4),
        'r2': round(r2, 5),
        'mape_pct': round(mape, 2) if not np.isnan(mape) else None,
        'energy_error_pct': round(energy_error_pct, 3),
        'total_py_kwh': round(total_py, 1),
        'total_js_kwh': round(total_js, 1)
    }


def create_validation_plots(df, js_ac, py_ac, metrics, config, output_dir):
    """Create validation visualization plots."""
    inv_rating = config["inverter"]["ac_rating_kw"]
    
    fig = plt.figure(figsize=(14, 10))
    gs = GridSpec(2, 2, figure=fig, hspace=0.3, wspace=0.25)
    
    # Plot 1: Scatter plot (JS vs Python)
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.scatter(py_ac, js_ac, alpha=0.3, s=8, c='steelblue')
    ax1.plot([0, inv_rating], [0, inv_rating], 'r--', linewidth=2, label='Perfect Match')
    ax1.set_xlabel('Python (pvlib) Power [kW]', fontsize=11)
    ax1.set_ylabel('JS (calibrated) Power [kW]', fontsize=11)
    ax1.set_title(f'JS vs Python Model\nR² = {metrics["r2"]:.4f}, RMSE = {metrics["rmse_kw"]:.3f} kW', 
                  fontsize=12, fontweight='bold')
    ax1.legend(loc='upper left')
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim(0, inv_rating * 1.1)
    ax1.set_ylim(0, inv_rating * 1.1)
    ax1.set_aspect('equal')
    
    # Plot 2: Residual distribution
    ax2 = fig.add_subplot(gs[0, 1])
    residuals = js_ac - py_ac
    ax2.hist(residuals, bins=50, color='steelblue', alpha=0.7, edgecolor='black')
    ax2.axvline(x=0, color='red', linestyle='--', linewidth=2)
    ax2.axvline(x=np.mean(residuals), color='orange', linestyle='-', linewidth=2, label=f'Mean: {np.mean(residuals):.3f} kW')
    ax2.set_xlabel('Residual (JS - Python) [kW]', fontsize=11)
    ax2.set_ylabel('Frequency', fontsize=11)
    ax2.set_title(f'Residual Distribution\nMAE = {metrics["mae_kw"]:.3f} kW', fontsize=12, fontweight='bold')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: Time series comparison (if timestamps available)
    ax3 = fig.add_subplot(gs[1, 0])
    
    # Sample for visibility (max 500 points)
    n_samples = min(500, len(df))
    sample_indices = np.linspace(0, len(df)-1, n_samples, dtype=int)
    
    x_vals = range(n_samples)
    ax3.plot(x_vals, py_ac[sample_indices], 'b-', linewidth=1, alpha=0.8, label='Python (pvlib)')
    ax3.plot(x_vals, js_ac[sample_indices], 'r--', linewidth=1, alpha=0.8, label='JS (calibrated)')
    ax3.set_xlabel('Sample Index', fontsize=11)
    ax3.set_ylabel('AC Power [kW]', fontsize=11)
    ax3.set_title('Time Series Comparison (sampled)', fontsize=12, fontweight='bold')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    ax3.set_ylim(0, inv_rating * 1.1)
    
    # Plot 4: Error vs irradiance
    ax4 = fig.add_subplot(gs[1, 1])
    ghi = df["ghi"].values
    abs_error = np.abs(residuals)
    
    # Bin by GHI
    ghi_bins = np.linspace(0, 1200, 13)
    bin_centers = (ghi_bins[:-1] + ghi_bins[1:]) / 2
    bin_errors = []
    
    for i in range(len(ghi_bins) - 1):
        mask = (ghi >= ghi_bins[i]) & (ghi < ghi_bins[i+1])
        if mask.sum() > 0:
            bin_errors.append(np.mean(abs_error[mask]))
        else:
            bin_errors.append(np.nan)
    
    ax4.bar(bin_centers, bin_errors, width=80, color='steelblue', alpha=0.7, edgecolor='black')
    ax4.set_xlabel('GHI [W/m²]', fontsize=11)
    ax4.set_ylabel('Mean Absolute Error [kW]', fontsize=11)
    ax4.set_title('Error by Irradiance Level', fontsize=12, fontweight='bold')
    ax4.grid(True, alpha=0.3, axis='y')
    
    plt.suptitle(f'JS Physics Model Calibration Validation\n{config.get("plant_name", "Plant")}',
                 fontsize=14, fontweight='bold', y=1.02)
    
    plt.tight_layout()
    
    plot_path = output_dir / "js_calibration_validation.png"
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    print(f"Saved plot: {plot_path}")
    
    plt.close()
    
    return plot_path


# =====================================================================
# MAIN
# =====================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Validate JS physics model calibration",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument(
        "--data",
        default=None,
        help="Path to test data CSV (default: data/js_calibration_data.csv)"
    )
    parser.add_argument(
        "--samples",
        type=int,
        default=None,
        help="Number of samples to use (default: all)"
    )
    parser.add_argument(
        "--no-plot",
        action="store_true",
        help="Skip generating plots"
    )
    
    args = parser.parse_args()
    
    print("=" * 70)
    print("VALIDATE JS PHYSICS MODEL CALIBRATION")
    print("=" * 70)
    
    # Load configuration
    config = load_config()
    print(f"Plant: {config.get('plant_name', 'Unknown')}")
    print(f"Inverter rating: {config['inverter']['ac_rating_kw']} kW")
    
    # Load calibration
    calib = load_calibration()
    if calib:
        print(f"\nCalibration loaded from: {OUTPUT_DIR / 'js_physics_calibration.json'}")
        params = calib["parameters"]
        print("Calibrated parameters:")
        for name, value in params.items():
            print(f"  {name}: {value}")
    else:
        print("\nNo calibration found - using default parameters")
        params = {
            'circumsolar_factor': 0.2,
            'circumsolar_threshold': 50,
            'aoi_threshold': 0.087,
            'diffuse_weight': 1.0,
            'brightness_factor': 1.0,
            'sapm_a': -3.56,
            'sapm_b': -0.075,
            'sapm_dt': 3
        }
    
    # Load test data
    data_path = Path(args.data) if args.data else DATA_DIR / "js_calibration_data.csv"
    
    if not data_path.exists():
        print(f"\nError: Data file not found: {data_path}")
        print("Run generate_calibration_data.py first!")
        sys.exit(1)
    
    print(f"\nLoading data from: {data_path}")
    df = pd.read_csv(data_path)
    
    # Prepare data
    if "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
        df = df.set_index("timestamp")
    
    # Sample if requested
    if args.samples and args.samples < len(df):
        df = df.sample(n=args.samples, random_state=42)
    
    print(f"  Total samples: {len(df):,}")
    
    # Run JS model
    print("\nRunning calibrated JS model...")
    js_ac = calc_pv_js(df, config, params)
    
    # Get Python reference
    py_ac = df["ac_power_kw"].values
    
    # Compute metrics
    metrics = compute_metrics(js_ac, py_ac, config["inverter"]["ac_rating_kw"])
    
    # Print results
    print("\n" + "=" * 70)
    print("VALIDATION RESULTS")
    print("=" * 70)
    
    print("\nAccuracy Metrics:")
    print(f"  RMSE:           {metrics['rmse_kw']:.4f} kW ({metrics['rmse_pct']:.2f}% of rated)")
    print(f"  MAE:            {metrics['mae_kw']:.4f} kW")
    print(f"  R² Score:       {metrics['r2']:.5f}")
    if metrics['mape_pct']:
        print(f"  MAPE (>1 kW):   {metrics['mape_pct']:.2f}%")
    
    print("\nEnergy Comparison:")
    print(f"  Python total:   {metrics['total_py_kwh']:.1f} kWh")
    print(f"  JS total:       {metrics['total_js_kwh']:.1f} kWh")
    print(f"  Energy error:   {metrics['energy_error_pct']:.3f}%")
    
    # Quality assessment
    print("\n" + "=" * 70)
    print("QUALITY ASSESSMENT")
    print("=" * 70)
    
    if metrics['rmse_pct'] < 1.0:
        print("  RMSE < 1%: EXCELLENT match to Python model")
    elif metrics['rmse_pct'] < 2.0:
        print("  RMSE < 2%: GOOD match to Python model")
    elif metrics['rmse_pct'] < 5.0:
        print("  RMSE < 5%: ACCEPTABLE for operational use")
    else:
        print("  RMSE >= 5%: NEEDS IMPROVEMENT - consider recalibrating")
    
    if metrics['r2'] > 0.99:
        print("  R² > 0.99: EXCELLENT correlation")
    elif metrics['r2'] > 0.95:
        print("  R² > 0.95: GOOD correlation")
    else:
        print("  R² < 0.95: NEEDS ATTENTION")
    
    # Create plots
    if not args.no_plot:
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        create_validation_plots(df, js_ac, py_ac, metrics, config, OUTPUT_DIR)
    
    # Save metrics
    validation_results = {
        "metrics": metrics,
        "parameters_used": params,
        "data_source": str(data_path),
        "samples": len(df),
        "validated_at": datetime.now().isoformat()
    }
    
    results_path = OUTPUT_DIR / "js_validation_results.json"
    with open(results_path, "w") as f:
        json.dump(validation_results, f, indent=2)
    print(f"\nSaved results: {results_path}")
    
    print("\n" + "=" * 70)
    print("VALIDATION COMPLETE")
    print("=" * 70)
    
    return metrics


if __name__ == "__main__":
    main()


