#!/usr/bin/env python3
"""
================================================================================
OPTIMIZE JS PHYSICS MODEL PARAMETERS
================================================================================

Uses differential evolution to find optimal parameters for the JS physics model
that minimize the error against the Python pvlib reference model.

The JS model is re-implemented in Python with tunable parameters for optimization.

TUNABLE PARAMETERS:
- circumsolar_factor: Perez approximation multiplier (default: 0.2)
- circumsolar_threshold: DNI threshold for circumsolar (default: 50)
- aoi_threshold: AOI cosine threshold (default: 0.087)
- diffuse_weight: Sky view factor adjustment (default: 1.0)
- brightness_factor: dni_extra effect scaling (default: 1.0)
- sapm_a: Thermal irradiance coefficient (default: -3.56)
- sapm_b: Thermal wind coefficient (default: -0.075)
- sapm_dt: Thermal offset (default: 3)

USAGE:
------
  # Optimize using default calibration data
  python optimize_js_params.py

  # Use specific data file and sample size
  python optimize_js_params.py --data ../../data/js_calibration_data.csv --samples 50000

  # Faster optimization with fewer iterations
  python optimize_js_params.py --maxiter 50

================================================================================
"""

import argparse
import json
import sys
import time
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd
from scipy.optimize import differential_evolution
from tqdm import tqdm

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


# =====================================================================
# JS MODEL REIMPLEMENTATION (with tunable parameters)
# =====================================================================

DEG2RAD = np.pi / 180
RAD2DEG = 180 / np.pi


def clip(v, vmin, vmax):
    """Clip value to range."""
    return np.clip(v, vmin, vmax)


def calc_dni_extra(timestamps):
    """
    Calculate extraterrestrial DNI (Spencer's formula).
    
    This is the NEW feature being added to JS model.
    
    Parameters:
    -----------
    timestamps : pd.DatetimeIndex
        UTC timestamps
    
    Returns:
    --------
    np.ndarray with dni_extra values in W/m²
    """
    day_of_year = timestamps.dayofyear
    b = 2 * np.pi * day_of_year / 365
    
    # Spencer's formula for Earth-Sun distance correction
    dni_extra = 1367 * (1.00011 + 0.034221 * np.cos(b) + 0.00128 * np.sin(b)
                       + 0.000719 * np.cos(2*b) + 0.000077 * np.sin(2*b))
    
    return dni_extra


def solar_pos_js(timestamps, lat, lon):
    """
    Simplified solar position (JS algorithm in Python).
    
    Parameters:
    -----------
    timestamps : pd.DatetimeIndex
        UTC timestamps
    lat, lon : float
        Location coordinates
    
    Returns:
    --------
    dict with 'zen', 'az', 'el' arrays
    """
    # Convert to numpy for vectorized operations
    ts_unix = timestamps.astype(np.int64) / 1e9  # seconds since epoch
    
    # Julian day
    jd = ts_unix / 86400 + 2440587.5
    jc = (jd - 2451545) / 36525
    
    # Mean anomaly
    m = (357.52911 + jc * 35999.05029) * DEG2RAD
    
    # Sun true longitude
    c = ((1.914602 - jc * 0.004817) * np.sin(m) + 
         0.019993 * np.sin(2 * m) + 
         0.000289 * np.sin(3 * m))
    sun_lon = (280.46646 + jc * 36000.76983 + c) * DEG2RAD
    
    # Declination
    obl = (23.439291 - jc * 0.0130042) * DEG2RAD
    dec = np.arcsin(np.sin(obl) * np.sin(sun_lon))
    
    # Equation of time
    eot = 4 * (sun_lon * RAD2DEG - np.arctan2(
        np.cos(obl) * np.sin(sun_lon),
        np.cos(sun_lon)
    ) * RAD2DEG)
    
    # Hour angle
    utc_h = timestamps.hour + timestamps.minute / 60 + timestamps.second / 3600
    solar_t = utc_h + lon / 15 + eot / 60
    ha = (solar_t - 12) * 15 * DEG2RAD
    
    # Elevation
    lat_r = lat * DEG2RAD
    sin_el = (np.sin(lat_r) * np.sin(dec) + 
              np.cos(lat_r) * np.cos(dec) * np.cos(ha))
    el = np.arcsin(clip(sin_el, -1, 1))
    
    # Azimuth
    cos_az = (np.sin(dec) - np.sin(lat_r) * sin_el) / (np.cos(lat_r) * np.cos(el))
    az = np.arccos(clip(cos_az, -1, 1))
    az = np.where(ha > 0, 2 * np.pi - az, az)
    
    return {
        'zen': 90 - el * RAD2DEG,
        'az': az * RAD2DEG,
        'el': el * RAD2DEG
    }


def interp_iam(aoi, iam_angles, iam_values):
    """Interpolate IAM from lookup table."""
    return np.interp(aoi, iam_angles, iam_values)


def perez_poa_js(ghi, dni, dhi, sun, tilt, azim, albedo, dni_extra, params):
    """
    Simplified Perez POA with tunable parameters.
    
    Parameters:
    -----------
    ghi, dni, dhi : np.ndarray
        Irradiance components in W/m²
    sun : dict
        Solar position with 'zen', 'az', 'el' keys
    tilt, azim : float
        Surface orientation in degrees
    albedo : float
        Ground reflectance
    dni_extra : np.ndarray
        Extraterrestrial DNI
    params : dict
        Tunable parameters
    
    Returns:
    --------
    dict with 'poa' and 'aoi' arrays
    """
    zen_r = sun['zen'] * DEG2RAD
    tilt_r = tilt * DEG2RAD
    az_r = azim * DEG2RAD
    sun_az_r = sun['az'] * DEG2RAD
    
    # AOI
    cos_aoi = (np.cos(zen_r) * np.cos(tilt_r) +
               np.sin(zen_r) * np.sin(tilt_r) * np.cos(sun_az_r - az_r))
    cos_aoi = clip(cos_aoi, -1, 1)
    aoi = np.arccos(cos_aoi) * RAD2DEG
    
    # Beam component
    beam = np.where((cos_aoi > 0) & (sun['el'] > 0), dni * cos_aoi, 0)
    
    # Simplified diffuse (isotropic + circumsolar approximation)
    f = 0.5 + 0.5 * np.cos(tilt_r)  # Sky view factor
    diff = dhi * f * params.get('diffuse_weight', 1.0)
    
    # Circumsolar enhancement (with dni_extra for brightness)
    circumsolar_factor = params.get('circumsolar_factor', 0.2)
    circumsolar_threshold = params.get('circumsolar_threshold', 50)
    aoi_threshold = params.get('aoi_threshold', 0.087)
    brightness_factor = params.get('brightness_factor', 1.0)
    
    # Calculate clearness index approximation using dni_extra
    kt = np.where(dni_extra > 0, ghi / dni_extra, 0)
    kt = clip(kt, 0, 1.2)
    
    # Brightness adjustment based on clearness
    brightness_adj = 1.0 + brightness_factor * (kt - 0.5)
    brightness_adj = clip(brightness_adj, 0.5, 1.5)
    
    # Circumsolar component
    cos_zen = np.cos(zen_r)
    cos_zen_safe = np.maximum(aoi_threshold, cos_zen)
    
    circumsolar_mask = (dni > circumsolar_threshold) & (cos_aoi > aoi_threshold)
    circum = np.where(
        circumsolar_mask,
        dhi * circumsolar_factor * brightness_adj * (cos_aoi / cos_zen_safe),
        0
    )
    diff = diff + circum
    
    # Ground reflection
    ground = ghi * albedo * (1 - np.cos(tilt_r)) * 0.5
    
    # Total POA
    poa = np.maximum(0, beam + diff + ground)
    
    return {'poa': poa, 'aoi': aoi}


def calc_pv_js(df, config, params):
    """
    JS physics model reimplemented in Python with tunable parameters.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Weather data with ghi, dni, dhi, air_temp, wind_speed columns
        Index must be DatetimeIndex
    config : dict
        Plant configuration
    params : dict
        Tunable parameters
    
    Returns:
    --------
    np.ndarray with AC power in kW
    """
    # Extract config
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
    
    # DC losses (pre-computed)
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
    
    # IAM table
    iam_angles = np.array(config["iam"]["angles"])
    iam_values = np.array(config["iam"]["values"])
    
    # Total module area
    total_module_area = sum(o["module_count"] * module_area for o in orientations)
    
    # Get inputs as numpy arrays
    ghi = df["ghi"].values
    dni = df["dni"].values
    dhi = df["dhi"].values
    air_temp = df["air_temp"].values
    wind_speed = df["wind_speed"].values
    
    # Solar position
    sun = solar_pos_js(df.index, lat, lon)
    
    # Extraterrestrial radiation (NEW feature)
    dni_extra = calc_dni_extra(df.index)
    
    # Initialize total DC power
    total_dc = np.zeros(len(df))
    total_tcell = np.zeros(len(df))
    
    # SAPM thermal parameters (tunable)
    sapm_a = params.get('sapm_a', -3.56)
    sapm_b = params.get('sapm_b', -0.075)
    sapm_dt = params.get('sapm_dt', 3)
    
    # Process each orientation
    for o in orientations:
        tilt = o["tilt"]
        azimuth = o["azimuth"]
        area_fraction = (o["module_count"] * module_area) / total_module_area
        
        # POA with tunable Perez approximation
        poa_result = perez_poa_js(ghi, dni, dhi, sun, tilt, azimuth, albedo, dni_extra, params)
        poa = poa_result['poa'] * far_shading
        aoi = poa_result['aoi']
        
        # IAM
        iam = interp_iam(aoi, iam_angles, iam_values)
        poa_optical = poa * iam
        
        # Cell temperature (tunable SAPM)
        e0 = poa / 1000
        t_cell = air_temp + sapm_a * e0 + sapm_b * e0 * wind_speed + sapm_dt
        total_tcell += t_cell
        
        # DC power
        dc_kw_m2 = poa_optical * module_efficiency / 1000
        dc_kw_m2 = dc_kw_m2 * (1 + gamma_p * (t_cell - 25))
        dc_kw_m2 = dc_kw_m2 * dc_loss_factor
        
        total_dc += dc_kw_m2 * total_module_area * area_fraction
    
    # Inverter efficiency and clipping
    ac = np.minimum(total_dc * inv_eff, inv_ac_rating)
    
    # AC wiring loss
    ac = ac * (1 - ac_wiring_loss)
    
    # Nighttime / low irradiance = 0
    ac = np.where((ghi < 1) | (sun['el'] < 0), 0, ac)
    
    return ac


# =====================================================================
# OPTIMIZATION
# =====================================================================

def compute_rmse(params_array, df_sample, config, param_names):
    """
    Compute RMSE between JS model (with params) and Python reference.
    
    Parameters:
    -----------
    params_array : np.ndarray
        Parameter values in order of param_names
    df_sample : pd.DataFrame
        Sample data with weather inputs and ac_power_kw target
    config : dict
        Plant configuration
    param_names : list
        Names of parameters being optimized
    
    Returns:
    --------
    float: RMSE in kW
    """
    # Convert array to dict
    params = dict(zip(param_names, params_array))
    
    # Run JS model
    js_ac = calc_pv_js(df_sample, config, params)
    
    # Get Python reference values
    py_ac = df_sample["ac_power_kw"].values
    
    # Compute RMSE
    rmse = np.sqrt(np.mean((js_ac - py_ac) ** 2))
    
    return rmse


def optimize_parameters(df, config, maxiter=100, sample_size=50000, seed=42):
    """
    Run differential evolution to find optimal parameters.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Full calibration dataset
    config : dict
        Plant configuration
    maxiter : int
        Maximum iterations for optimizer
    sample_size : int
        Number of samples to use for optimization (for speed)
    seed : int
        Random seed
    
    Returns:
    --------
    dict: Optimal parameters and metrics
    """
    np.random.seed(seed)
    
    # Sample data for faster optimization
    if len(df) > sample_size:
        df_sample = df.sample(n=sample_size, random_state=seed)
    else:
        df_sample = df
    
    # Convert timestamp column to index if needed
    if "timestamp" in df_sample.columns:
        df_sample = df_sample.copy()
        df_sample["timestamp"] = pd.to_datetime(df_sample["timestamp"], utc=True)
        df_sample = df_sample.set_index("timestamp")
    
    print(f"\nOptimizing with {len(df_sample):,} samples...")
    
    # Parameter names and bounds
    param_names = [
        'circumsolar_factor',
        'circumsolar_threshold', 
        'aoi_threshold',
        'diffuse_weight',
        'brightness_factor',
        'sapm_a',
        'sapm_b',
        'sapm_dt'
    ]
    
    bounds = [
        (0.05, 0.5),      # circumsolar_factor (default: 0.2)
        (10, 100),        # circumsolar_threshold (default: 50)
        (0.05, 0.15),     # aoi_threshold (default: 0.087)
        (0.8, 1.2),       # diffuse_weight (default: 1.0)
        (0.0, 2.0),       # brightness_factor (NEW)
        (-5.0, -2.0),     # sapm_a (default: -3.56)
        (-0.15, -0.03),   # sapm_b (default: -0.075)
        (1, 5),           # sapm_dt (default: 3)
    ]
    
    # Initial baseline (default values)
    default_params = {
        'circumsolar_factor': 0.2,
        'circumsolar_threshold': 50,
        'aoi_threshold': 0.087,
        'diffuse_weight': 1.0,
        'brightness_factor': 1.0,
        'sapm_a': -3.56,
        'sapm_b': -0.075,
        'sapm_dt': 3
    }
    
    # Compute baseline RMSE
    baseline_rmse = compute_rmse(
        [default_params[n] for n in param_names],
        df_sample, config, param_names
    )
    print(f"Baseline RMSE (default params): {baseline_rmse:.4f} kW")
    
    print(f"\nRunning differential evolution (maxiter={maxiter})...")
    print("This may take 5-30 minutes depending on sample size...")
    
    start_time = time.perf_counter()
    
    # Progress callback
    iteration_count = [0]
    best_rmse = [baseline_rmse]
    
    def callback(xk, convergence):
        iteration_count[0] += 1
        current_rmse = compute_rmse(xk, df_sample, config, param_names)
        if current_rmse < best_rmse[0]:
            best_rmse[0] = current_rmse
            print(f"  Iteration {iteration_count[0]}: RMSE = {current_rmse:.4f} kW (improved)")
    
    result = differential_evolution(
        compute_rmse,
        bounds,
        args=(df_sample, config, param_names),
        maxiter=maxiter,
        workers=-1,  # Use all CPU cores
        updating='deferred',
        callback=callback,
        disp=True,
        seed=seed,
        tol=0.0001,
        atol=0.0001
    )
    
    elapsed = time.perf_counter() - start_time
    
    # Extract optimal parameters
    optimal_params = dict(zip(param_names, result.x))
    
    # Round for cleaner output
    optimal_params_rounded = {
        'circumsolar_factor': round(optimal_params['circumsolar_factor'], 4),
        'circumsolar_threshold': round(optimal_params['circumsolar_threshold'], 2),
        'aoi_threshold': round(optimal_params['aoi_threshold'], 4),
        'diffuse_weight': round(optimal_params['diffuse_weight'], 4),
        'brightness_factor': round(optimal_params['brightness_factor'], 4),
        'sapm_a': round(optimal_params['sapm_a'], 3),
        'sapm_b': round(optimal_params['sapm_b'], 4),
        'sapm_dt': round(optimal_params['sapm_dt'], 2),
    }
    
    # Compute final metrics
    final_rmse = result.fun
    inv_rating = config["inverter"]["ac_rating_kw"]
    rmse_pct = 100 * final_rmse / inv_rating
    
    # Compute additional metrics on full sample
    js_ac = calc_pv_js(df_sample, config, optimal_params)
    py_ac = df_sample["ac_power_kw"].values
    
    mae = np.mean(np.abs(js_ac - py_ac))
    r2 = 1 - np.sum((js_ac - py_ac) ** 2) / np.sum((py_ac - np.mean(py_ac)) ** 2)
    
    # Energy comparison
    total_py = py_ac.sum()
    total_js = js_ac.sum()
    energy_error_pct = 100 * abs(total_py - total_js) / total_py if total_py > 0 else 0
    
    results = {
        "parameters": optimal_params_rounded,
        "metrics": {
            "baseline_rmse_kw": round(baseline_rmse, 4),
            "final_rmse_kw": round(final_rmse, 4),
            "rmse_pct_of_rating": round(rmse_pct, 3),
            "mae_kw": round(mae, 4),
            "r2": round(r2, 5),
            "energy_error_pct": round(energy_error_pct, 3),
            "improvement_pct": round(100 * (baseline_rmse - final_rmse) / baseline_rmse, 2)
        },
        "optimization": {
            "iterations": result.nit,
            "function_evaluations": result.nfev,
            "success": result.success,
            "elapsed_seconds": round(elapsed, 1),
            "samples_used": len(df_sample)
        },
        "created_at": datetime.now().isoformat(),
        "plant_name": config.get("plant_name", "Unknown")
    }
    
    return results


# =====================================================================
# OUTPUT
# =====================================================================

def generate_js_code(params):
    """Generate JavaScript code snippet for physics_model.js."""
    js_code = f"""// =====================================================================
// CALIBRATED PARAMETERS (optimized to match Python pvlib model)
// =====================================================================
// Generated: {datetime.now().isoformat()}
// DO NOT EDIT MANUALLY - regenerate using optimize_js_params.py

var CALIBRATION = {{
    // Perez POA approximation
    circumsolar_factor: {params['circumsolar_factor']},    // Circumsolar brightening multiplier
    circumsolar_threshold: {params['circumsolar_threshold']},   // DNI threshold for circumsolar (W/m²)
    aoi_threshold: {params['aoi_threshold']},          // AOI cosine threshold
    diffuse_weight: {params['diffuse_weight']},          // Sky view factor adjustment
    brightness_factor: {params['brightness_factor']},      // dni_extra effect on brightness
    
    // SAPM thermal model
    sapm_a: {params['sapm_a']},              // Irradiance coefficient
    sapm_b: {params['sapm_b']},            // Wind coefficient
    sapm_dt: {params['sapm_dt']}                // Temperature offset
}};

// Extraterrestrial radiation calculation (Spencer's formula)
function calcDniExtra(timestamp) {{
    var d = new Date(timestamp);
    var start = new Date(d.getFullYear(), 0, 0);
    var diff = d - start;
    var dayOfYear = Math.floor(diff / 86400000);
    var b = 2 * Math.PI * dayOfYear / 365;
    return 1367 * (1.00011 + 0.034221 * Math.cos(b) + 0.00128 * Math.sin(b)
                  + 0.000719 * Math.cos(2*b) + 0.000077 * Math.sin(2*b));
}}
"""
    return js_code


# =====================================================================
# MAIN
# =====================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Optimize JS physics model parameters",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python optimize_js_params.py
  python optimize_js_params.py --samples 20000 --maxiter 100
  python optimize_js_params.py --data ../../data/js_calibration_data.csv
        """
    )
    
    parser.add_argument(
        "--data",
        default=None,
        help="Path to calibration data CSV (default: data/js_calibration_data.csv)"
    )
    parser.add_argument(
        "--samples",
        type=int,
        default=50000,
        help="Number of samples to use for optimization (default: 50000)"
    )
    parser.add_argument(
        "--maxiter",
        type=int,
        default=1000,
        help="Maximum optimizer iterations (default: 1000)"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed (default: 42)"
    )
    
    args = parser.parse_args()
    
    print("=" * 70)
    print("OPTIMIZE JS PHYSICS MODEL PARAMETERS")
    print("=" * 70)
    
    # Load config
    config = load_config()
    print(f"Plant: {config.get('plant_name', 'Unknown')}")
    print(f"Inverter rating: {config['inverter']['ac_rating_kw']} kW")
    
    # Load calibration data
    data_path = Path(args.data) if args.data else DATA_DIR / "js_calibration_data.csv"
    
    if not data_path.exists():
        print(f"\nError: Calibration data not found: {data_path}")
        print("Run generate_calibration_data.py first!")
        sys.exit(1)
    
    print(f"\nLoading data from: {data_path}")
    df = pd.read_csv(data_path)
    print(f"  Total samples: {len(df):,}")
    
    # Run optimization
    results = optimize_parameters(
        df, config,
        maxiter=args.maxiter,
        sample_size=args.samples,
        seed=args.seed
    )
    
    # Print results
    print("\n" + "=" * 70)
    print("OPTIMIZATION RESULTS")
    print("=" * 70)
    
    print("\nOptimized Parameters:")
    for name, value in results["parameters"].items():
        print(f"  {name}: {value}")
    
    print("\nMetrics:")
    for name, value in results["metrics"].items():
        print(f"  {name}: {value}")
    
    print("\nOptimization Info:")
    print(f"  Iterations: {results['optimization']['iterations']}")
    print(f"  Function evaluations: {results['optimization']['function_evaluations']}")
    print(f"  Elapsed time: {results['optimization']['elapsed_seconds']:.1f}s")
    print(f"  Success: {results['optimization']['success']}")
    
    # Save results
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    output_path = OUTPUT_DIR / "js_physics_calibration.json"
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved results: {output_path}")
    
    # Generate and save JS code
    js_code = generate_js_code(results["parameters"])
    js_path = OUTPUT_DIR / "js_calibration_code.js"
    with open(js_path, "w") as f:
        f.write(js_code)
    print(f"Saved JS code: {js_path}")
    
    # Print JS code
    print("\n" + "=" * 70)
    print("JAVASCRIPT CODE (copy to physics_model.js)")
    print("=" * 70)
    print(js_code)
    
    print("=" * 70)
    print("COMPLETE")
    print("=" * 70)
    print("\nNext steps:")
    print("  1. Copy the CALIBRATION object to physics_model.js")
    print("  2. Update perezPOA() to use CALIBRATION.* values")
    print("  3. Add calcDniExtra() function")
    print("  4. Run validate_calibration.py to verify")
    print("=" * 70)
    
    return results


if __name__ == "__main__":
    main()


