#!/usr/bin/env python3
"""
================================================================================
OPTIMIZE JS PHYSICS MODEL PARAMETERS (GPU-ACCELERATED)
================================================================================

Uses PyTorch with CUDA for GPU-accelerated optimization of JS physics model
parameters. Evaluates the model on GPU for 10-100x speedup on large datasets.

REQUIREMENTS:
- PyTorch with CUDA support
- RTX GPU with CUDA drivers

Install PyTorch with CUDA:
    pip install torch --index-url https://download.pytorch.org/whl/cu121

USAGE:
------
  # Optimize using GPU (default if available)
  python optimize_js_params_gpu.py

  # Force CPU (for comparison)
  python optimize_js_params_gpu.py --device cpu

  # Use specific sample size
  python optimize_js_params_gpu.py --samples 100000

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

try:
    import torch
    import torch.nn as nn
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    print("PyTorch not installed. Install with: pip install torch --index-url https://download.pytorch.org/whl/cu121")
    sys.exit(1)

from scipy.optimize import differential_evolution

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


def check_gpu():
    """Check GPU availability and print info."""
    print("\n" + "=" * 70)
    print("GPU STATUS")
    print("=" * 70)
    
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"  GPU Available: YES")
        print(f"  GPU Name: {gpu_name}")
        print(f"  GPU Memory: {gpu_memory:.1f} GB")
        print(f"  CUDA Version: {torch.version.cuda}")
        return "cuda"
    else:
        print("  GPU Available: NO")
        print("  Falling back to CPU")
        return "cpu"


# =====================================================================
# GPU-ACCELERATED JS MODEL (PyTorch)
# =====================================================================

class JSPhysicsModel(nn.Module):
    """
    JS Physics Model implemented in PyTorch for GPU acceleration.
    
    All calculations are vectorized and run on GPU for maximum throughput.
    """
    
    def __init__(self, config, device="cuda"):
        super().__init__()
        self.device = device
        
        # Extract config values as tensors
        self.lat = torch.tensor(config["location"]["lat"], device=device)
        self.lon = torch.tensor(config["location"]["lon"], device=device)
        
        # Orientations
        orientations = config["orientations"]
        self.tilts = torch.tensor([o["tilt"] for o in orientations], device=device)
        self.azimuths = torch.tensor([o["azimuth"] for o in orientations], device=device)
        
        module_area = config["module"]["area_m2"]
        module_counts = torch.tensor([o["module_count"] for o in orientations], device=device)
        total_module_area = (module_counts * module_area).sum()
        self.area_fractions = (module_counts * module_area) / total_module_area
        self.total_module_area = total_module_area
        
        # Module parameters
        self.module_efficiency = torch.tensor(config["module"]["efficiency_stc"], device=device)
        self.gamma_p = torch.tensor(config["module"]["gamma_p"], device=device)
        
        # Inverter
        self.inv_ac_rating = torch.tensor(config["inverter"]["ac_rating_kw"], device=device)
        self.inv_eff = torch.tensor(config["inverter"]["flat_efficiency"], device=device)
        
        # Losses
        losses = config["losses"]
        self.albedo = torch.tensor(losses["albedo"], device=device)
        self.far_shading = torch.tensor(losses.get("far_shading", 1.0), device=device)
        
        dc_loss_factor = (
            (1 - losses["soiling"]) *
            (1 - losses["lid"]) *
            (1 + losses["module_quality"]) *
            (1 - losses["mismatch"]) *
            (1 - losses["dc_wiring"]) *
            losses.get("far_shading", 1.0)
        )
        self.dc_loss_factor = torch.tensor(dc_loss_factor, device=device)
        self.ac_wiring_loss = torch.tensor(losses["ac_wiring"], device=device)
        
        # IAM table
        self.iam_angles = torch.tensor(config["iam"]["angles"], device=device, dtype=torch.float32)
        self.iam_values = torch.tensor(config["iam"]["values"], device=device, dtype=torch.float32)
        
        # Constants
        self.DEG2RAD = torch.tensor(np.pi / 180, device=device)
        self.RAD2DEG = torch.tensor(180 / np.pi, device=device)
    
    def calc_dni_extra(self, day_of_year):
        """Calculate extraterrestrial DNI (Spencer's formula)."""
        b = 2 * np.pi * day_of_year / 365
        dni_extra = 1367 * (1.00011 + 0.034221 * torch.cos(b) + 0.00128 * torch.sin(b)
                           + 0.000719 * torch.cos(2*b) + 0.000077 * torch.sin(2*b))
        return dni_extra
    
    def solar_position(self, unix_timestamps, utc_hours):
        """Simplified solar position calculation."""
        jd = unix_timestamps / 86400 + 2440587.5
        jc = (jd - 2451545) / 36525
        
        m = (357.52911 + jc * 35999.05029) * self.DEG2RAD
        c = ((1.914602 - jc * 0.004817) * torch.sin(m) + 
             0.019993 * torch.sin(2 * m) + 
             0.000289 * torch.sin(3 * m))
        sun_lon = (280.46646 + jc * 36000.76983 + c) * self.DEG2RAD
        
        obl = (23.439291 - jc * 0.0130042) * self.DEG2RAD
        dec = torch.asin(torch.sin(obl) * torch.sin(sun_lon))
        
        eot = 4 * (sun_lon * self.RAD2DEG - torch.atan2(
            torch.cos(obl) * torch.sin(sun_lon),
            torch.cos(sun_lon)
        ) * self.RAD2DEG)
        
        solar_t = utc_hours + self.lon / 15 + eot / 60
        ha = (solar_t - 12) * 15 * self.DEG2RAD
        
        lat_r = self.lat * self.DEG2RAD
        sin_el = (torch.sin(lat_r) * torch.sin(dec) + 
                  torch.cos(lat_r) * torch.cos(dec) * torch.cos(ha))
        sin_el = torch.clamp(sin_el, -1, 1)
        el = torch.asin(sin_el)
        
        cos_az = (torch.sin(dec) - torch.sin(lat_r) * sin_el) / (torch.cos(lat_r) * torch.cos(el))
        cos_az = torch.clamp(cos_az, -1, 1)
        az = torch.acos(cos_az)
        az = torch.where(ha > 0, 2 * np.pi - az, az)
        
        zen = np.pi / 2 - el
        
        return {'zen': zen, 'az': az, 'el': el}
    
    def interp_iam(self, aoi):
        """Interpolate IAM values."""
        # Clamp AOI to valid range
        aoi_clamped = torch.clamp(aoi, 0, 90)
        
        # Find indices for interpolation
        indices = torch.searchsorted(self.iam_angles, aoi_clamped)
        indices = torch.clamp(indices, 1, len(self.iam_angles) - 1)
        
        # Linear interpolation
        x0 = self.iam_angles[indices - 1]
        x1 = self.iam_angles[indices]
        y0 = self.iam_values[indices - 1]
        y1 = self.iam_values[indices]
        
        t = (aoi_clamped - x0) / (x1 - x0 + 1e-10)
        iam = y0 + t * (y1 - y0)
        
        return iam
    
    def perez_poa(self, ghi, dni, dhi, sun, tilt, azim, dni_extra, params):
        """Calibrated Perez POA approximation on GPU."""
        zen_r = sun['zen']
        tilt_r = tilt * self.DEG2RAD
        az_r = azim * self.DEG2RAD
        sun_az_r = sun['az']
        
        # AOI
        cos_aoi = (torch.cos(zen_r) * torch.cos(tilt_r) +
                   torch.sin(zen_r) * torch.sin(tilt_r) * torch.cos(sun_az_r - az_r))
        cos_aoi = torch.clamp(cos_aoi, -1, 1)
        aoi = torch.acos(cos_aoi) * self.RAD2DEG
        
        # Beam
        beam = torch.where((cos_aoi > 0) & (sun['el'] > 0), dni * cos_aoi, torch.zeros_like(dni))
        
        # Diffuse
        f = 0.5 + 0.5 * torch.cos(tilt_r)
        diff = dhi * f * params['diffuse_weight']
        
        # Clearness index
        kt = torch.where(dni_extra > 0, ghi / dni_extra, torch.zeros_like(ghi))
        kt = torch.clamp(kt, 0, 1.2)
        brightness_adj = 1.0 + params['brightness_factor'] * (kt - 0.5)
        brightness_adj = torch.clamp(brightness_adj, 0.5, 1.5)
        
        # Circumsolar
        cos_zen = torch.cos(zen_r)
        cos_zen_safe = torch.maximum(params['aoi_threshold'], cos_zen)
        circumsolar_mask = (dni > params['circumsolar_threshold']) & (cos_aoi > params['aoi_threshold'])
        circum = torch.where(
            circumsolar_mask,
            dhi * params['circumsolar_factor'] * brightness_adj * (cos_aoi / cos_zen_safe),
            torch.zeros_like(dhi)
        )
        diff = diff + circum
        
        # Ground reflection
        ground = ghi * self.albedo * (1 - torch.cos(tilt_r)) * 0.5
        
        poa = torch.clamp(beam + diff + ground, min=0)
        
        return {'poa': poa, 'aoi': aoi}
    
    def forward(self, ghi, dni, dhi, air_temp, wind_speed, unix_ts, utc_hours, day_of_year, params):
        """
        Forward pass: compute AC power for all samples.
        
        All inputs are GPU tensors of shape (batch_size,).
        params is a dict of scalar tensors.
        """
        batch_size = ghi.shape[0]
        
        # Solar position
        sun = self.solar_position(unix_ts, utc_hours)
        
        # Extraterrestrial radiation
        dni_extra = self.calc_dni_extra(day_of_year)
        
        # Initialize total DC power
        total_dc = torch.zeros(batch_size, device=self.device)
        
        # Process each orientation
        for i in range(len(self.tilts)):
            tilt = self.tilts[i]
            azimuth = self.azimuths[i]
            area_fraction = self.area_fractions[i]
            
            # POA
            poa_result = self.perez_poa(ghi, dni, dhi, sun, tilt, azimuth, dni_extra, params)
            poa = poa_result['poa'] * self.far_shading
            aoi = poa_result['aoi']
            
            # IAM
            iam = self.interp_iam(aoi)
            poa_optical = poa * iam
            
            # Cell temperature
            e0 = poa / 1000
            t_cell = air_temp + params['sapm_a'] * e0 + params['sapm_b'] * e0 * wind_speed + params['sapm_dt']
            
            # DC power
            dc_kw_m2 = poa_optical * self.module_efficiency / 1000
            dc_kw_m2 = dc_kw_m2 * (1 + self.gamma_p * (t_cell - 25))
            dc_kw_m2 = dc_kw_m2 * self.dc_loss_factor
            
            total_dc = total_dc + dc_kw_m2 * self.total_module_area * area_fraction
        
        # Inverter + clipping
        ac = torch.minimum(total_dc * self.inv_eff, self.inv_ac_rating)
        
        # AC wiring loss
        ac = ac * (1 - self.ac_wiring_loss)
        
        # Nighttime = 0
        ac = torch.where((ghi < 1) | (sun['el'] < 0), torch.zeros_like(ac), ac)
        
        return ac


# =====================================================================
# GPU OPTIMIZATION
# =====================================================================

class GPUOptimizer:
    """GPU-accelerated parameter optimizer."""
    
    def __init__(self, config, df, device="cuda"):
        self.device = device
        self.config = config
        self.inv_rating = config["inverter"]["ac_rating_kw"]
        
        # Create GPU model
        self.model = JSPhysicsModel(config, device=device)
        
        # Prepare data tensors (move to GPU once)
        print(f"\nMoving {len(df):,} samples to GPU...")
        
        # Parse timestamps
        if "timestamp" in df.columns:
            timestamps = pd.to_datetime(df["timestamp"], utc=True)
        else:
            timestamps = df.index
        
        # Calculate time features
        unix_ts = timestamps.astype(np.int64) // 10**9
        utc_hours = timestamps.dt.hour + timestamps.dt.minute / 60 + timestamps.dt.second / 3600
        day_of_year = timestamps.dt.dayofyear
        
        # Create GPU tensors
        self.ghi = torch.tensor(df["ghi"].values, device=device, dtype=torch.float32)
        self.dni = torch.tensor(df["dni"].values, device=device, dtype=torch.float32)
        self.dhi = torch.tensor(df["dhi"].values, device=device, dtype=torch.float32)
        self.air_temp = torch.tensor(df["air_temp"].values, device=device, dtype=torch.float32)
        self.wind_speed = torch.tensor(df["wind_speed"].values, device=device, dtype=torch.float32)
        self.unix_ts = torch.tensor(unix_ts.values, device=device, dtype=torch.float32)
        self.utc_hours = torch.tensor(utc_hours.values, device=device, dtype=torch.float32)
        self.day_of_year = torch.tensor(day_of_year.values, device=device, dtype=torch.float32)
        self.target = torch.tensor(df["ac_power_kw"].values, device=device, dtype=torch.float32)
        
        print(f"  GPU memory used: {torch.cuda.memory_allocated() / 1e9:.2f} GB")
    
    def compute_rmse(self, params_array):
        """Compute RMSE on GPU."""
        # Convert array to param dict with GPU tensors
        params = {
            'circumsolar_factor': torch.tensor(params_array[0], device=self.device),
            'circumsolar_threshold': torch.tensor(params_array[1], device=self.device),
            'aoi_threshold': torch.tensor(params_array[2], device=self.device),
            'diffuse_weight': torch.tensor(params_array[3], device=self.device),
            'brightness_factor': torch.tensor(params_array[4], device=self.device),
            'sapm_a': torch.tensor(params_array[5], device=self.device),
            'sapm_b': torch.tensor(params_array[6], device=self.device),
            'sapm_dt': torch.tensor(params_array[7], device=self.device),
        }
        
        # Forward pass on GPU
        with torch.no_grad():
            pred = self.model(
                self.ghi, self.dni, self.dhi, 
                self.air_temp, self.wind_speed,
                self.unix_ts, self.utc_hours, self.day_of_year,
                params
            )
        
        # Compute RMSE
        rmse = torch.sqrt(torch.mean((pred - self.target) ** 2))
        
        return rmse.cpu().item()
    
    def optimize(self, maxiter=100, seed=42):
        """Run differential evolution with GPU-accelerated objective."""
        
        param_names = [
            'circumsolar_factor', 'circumsolar_threshold', 'aoi_threshold',
            'diffuse_weight', 'brightness_factor', 'sapm_a', 'sapm_b', 'sapm_dt'
        ]
        
        bounds = [
            (0.05, 0.5),      # circumsolar_factor
            (10, 100),        # circumsolar_threshold
            (0.05, 0.15),     # aoi_threshold
            (0.8, 1.2),       # diffuse_weight
            (0.0, 2.0),       # brightness_factor
            (-5.0, -2.0),     # sapm_a
            (-0.15, -0.03),   # sapm_b
            (1, 5),           # sapm_dt
        ]
        
        # Baseline
        default_params = [0.2, 50, 0.087, 1.0, 1.0, -3.56, -0.075, 3]
        baseline_rmse = self.compute_rmse(default_params)
        print(f"Baseline RMSE: {baseline_rmse:.4f} kW")
        
        print(f"\nRunning GPU-accelerated optimization (maxiter={maxiter})...")
        
        start_time = time.perf_counter()
        iteration_count = [0]
        best_rmse = [baseline_rmse]
        
        def callback(xk, convergence):
            iteration_count[0] += 1
            current_rmse = self.compute_rmse(xk)
            if current_rmse < best_rmse[0]:
                best_rmse[0] = current_rmse
                print(f"  Iteration {iteration_count[0]}: RMSE = {current_rmse:.4f} kW (improved)")
        
        # Note: workers=1 because GPU is already parallel
        result = differential_evolution(
            self.compute_rmse,
            bounds,
            maxiter=maxiter,
            workers=1,  # GPU handles parallelism
            callback=callback,
            disp=True,
            seed=seed,
            tol=0.0001,
            atol=0.0001,
            updating='deferred'  # Match CPU version for consistent convergence
        )
        
        elapsed = time.perf_counter() - start_time
        
        # Extract results
        optimal_params = {
            'circumsolar_factor': round(result.x[0], 4),
            'circumsolar_threshold': round(result.x[1], 2),
            'aoi_threshold': round(result.x[2], 4),
            'diffuse_weight': round(result.x[3], 4),
            'brightness_factor': round(result.x[4], 4),
            'sapm_a': round(result.x[5], 3),
            'sapm_b': round(result.x[6], 4),
            'sapm_dt': round(result.x[7], 2),
        }
        
        # Final metrics
        final_rmse = result.fun
        rmse_pct = 100 * final_rmse / self.inv_rating
        
        # Compute additional metrics (matching CPU version)
        params = {
            'circumsolar_factor': torch.tensor(result.x[0], device=self.device),
            'circumsolar_threshold': torch.tensor(result.x[1], device=self.device),
            'aoi_threshold': torch.tensor(result.x[2], device=self.device),
            'diffuse_weight': torch.tensor(result.x[3], device=self.device),
            'brightness_factor': torch.tensor(result.x[4], device=self.device),
            'sapm_a': torch.tensor(result.x[5], device=self.device),
            'sapm_b': torch.tensor(result.x[6], device=self.device),
            'sapm_dt': torch.tensor(result.x[7], device=self.device),
        }
        
        with torch.no_grad():
            pred = self.model(
                self.ghi, self.dni, self.dhi,
                self.air_temp, self.wind_speed,
                self.unix_ts, self.utc_hours, self.day_of_year,
                params
            )
        
        js_ac = pred.cpu().numpy()
        py_ac = self.target.cpu().numpy()
        
        mae = float(np.mean(np.abs(js_ac - py_ac)))
        r2 = float(1 - np.sum((js_ac - py_ac) ** 2) / np.sum((py_ac - np.mean(py_ac)) ** 2))
        
        total_py = float(py_ac.sum())
        total_js = float(js_ac.sum())
        energy_error_pct = 100 * abs(total_py - total_js) / total_py if total_py > 0 else 0
        
        return {
            "parameters": optimal_params,
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
                "samples_used": len(self.ghi),
                "device": self.device
            },
            "created_at": datetime.now().isoformat(),
            "plant_name": self.config.get("plant_name", "Unknown")
        }


# =====================================================================
# OUTPUT
# =====================================================================

def generate_js_code(params):
    """Generate JavaScript code snippet."""
    js_code = f"""// =====================================================================
// CALIBRATED PARAMETERS (GPU-optimized to match Python pvlib model)
// =====================================================================
// Generated: {datetime.now().isoformat()}
// DO NOT EDIT MANUALLY - regenerate using optimize_js_params_gpu.py

var CALIBRATION = {{
    // Perez POA approximation
    circumsolar_factor: {params['circumsolar_factor']},
    circumsolar_threshold: {params['circumsolar_threshold']},
    aoi_threshold: {params['aoi_threshold']},
    diffuse_weight: {params['diffuse_weight']},
    brightness_factor: {params['brightness_factor']},
    
    // SAPM thermal model
    sapm_a: {params['sapm_a']},
    sapm_b: {params['sapm_b']},
    sapm_dt: {params['sapm_dt']}
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
        description="GPU-accelerated JS physics model optimization",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python optimize_js_params_gpu.py
  python optimize_js_params_gpu.py --samples 100000
  python optimize_js_params_gpu.py --device cpu  # Force CPU
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
        default=10000,
        help="Number of samples to use for optimization (default: 10000)"
    )
    parser.add_argument(
        "--maxiter",
        type=int,
        default=100,
        help="Maximum optimizer iterations (default: 100)"
    )
    parser.add_argument(
        "--device",
        default=None,
        choices=["cuda", "cpu"],
        help="Force device (default: auto-detect GPU)"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed (default: 42)"
    )
    
    args = parser.parse_args()
    
    print("=" * 70)
    print("GPU-ACCELERATED JS PHYSICS MODEL OPTIMIZATION")
    print("=" * 70)
    
    # Check GPU
    if args.device:
        device = args.device
    else:
        device = check_gpu()
    
    print(f"\nUsing device: {device.upper()}")
    
    # Load config
    config = load_config()
    print(f"Plant: {config.get('plant_name', 'Unknown')}")
    print(f"Inverter rating: {config['inverter']['ac_rating_kw']} kW")
    
    # Load calibration data
    data_path = Path(args.data) if args.data else DATA_DIR / "js_calibration_data.csv"
    
    if not data_path.exists():
        print(f"\nError: Data not found: {data_path}")
        print("Run generate_calibration_data.py first!")
        sys.exit(1)
    
    print(f"\nLoading data from: {data_path}")
    df = pd.read_csv(data_path)
    print(f"  Total samples: {len(df):,}")
    
    # Sample data for faster optimization (matching CPU version behavior)
    if args.samples and len(df) > args.samples:
        df_sample = df.sample(n=args.samples, random_state=args.seed)
        print(f"  Using {len(df_sample):,} samples for optimization")
    else:
        df_sample = df
    
    # Create GPU optimizer
    optimizer = GPUOptimizer(config, df_sample, device=device)
    
    # Run optimization
    results = optimizer.optimize(maxiter=args.maxiter, seed=args.seed)
    
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
    print(f"  Device: {results['optimization']['device'].upper()}")
    
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

