#!/usr/bin/env python3
"""
================================================================================
GENERATE CALIBRATION DATA FOR JS PHYSICS MODEL
================================================================================

Generates training data from the Python pvlib physics model for calibrating
the JS physics model parameters.

Creates 100K-1M samples with:
- Random timestamps across a full year (daylight hours)
- Physically realistic irradiance combinations (GHI, DNI, DHI)
- Temperature and wind speed variations
- Python model AC power output (ground truth)

USAGE:
------
  # Generate 100K samples (default)
  python generate_calibration_data.py

  # Generate 1M samples
  python generate_calibration_data.py --samples 1000000

  # Use specific random seed for reproducibility
  python generate_calibration_data.py --samples 100000 --seed 42

================================================================================
"""

import argparse
import json
import sys
import time
from pathlib import Path
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import pvlib
from pvlib.location import Location
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
# PHYSICS MODEL (from prepare_training_data.py)
# =====================================================================

def run_physics_model(df_weather, config):
    """
    Run physics-based PV model on weather data.
    
    Uses the same physics as physics_model.py (full Perez with dni_extra).
    
    Parameters:
    -----------
    df_weather : pd.DataFrame
        Hourly weather data with columns: ghi, dni, dhi, air_temp, wind_speed
        Index must be UTC DatetimeIndex
    config : dict
        Plant configuration from plant_config.json
    
    Returns:
    --------
    pd.Series with AC power output in kW, indexed by timestamp
    """
    # Extract config values
    lat = config["location"]["lat"]
    lon = config["location"]["lon"]
    altitude = config["location"]["altitude_m"]
    orientations = config["orientations"]
    module_area = config["module"]["area_m2"]
    module_efficiency = config["module"]["efficiency_stc"]
    gamma_p = config["module"]["gamma_p"]
    inv_ac_rating = config["inverter"]["ac_rating_kw"]
    albedo = config["losses"]["albedo"]
    far_shading = config["losses"].get("far_shading", 1.0)
    
    # Calculate loss factors
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
    
    # SAPM thermal model
    thermal_model = config.get("thermal_model", "close_mount_glass_glass")
    sapm_params = pvlib.temperature.TEMPERATURE_MODEL_PARAMETERS["sapm"][thermal_model]
    
    # IAM table
    iam_angles = np.array(config["iam"]["angles"])
    iam_values = np.array(config["iam"]["values"])
    
    # Total module area
    total_module_area = sum(o["module_count"] * module_area for o in orientations)
    
    # Site location
    site = Location(lat, lon, altitude=altitude)
    solpos = site.get_solarposition(df_weather.index)
    
    # Extraterrestrial radiation for Perez model accuracy
    dni_extra = pvlib.irradiance.get_extra_radiation(df_weather.index)
    
    # Initialize output
    plant_ac = pd.Series(0.0, index=df_weather.index)
    
    for o in orientations:
        tilt = o["tilt"]
        azimuth = o["azimuth"]
        area_fraction = (o["module_count"] * module_area) / total_module_area
        
        # POA irradiance using Perez model with dni_extra
        irr = pvlib.irradiance.get_total_irradiance(
            surface_tilt=tilt,
            surface_azimuth=azimuth,
            solar_zenith=solpos["zenith"],
            solar_azimuth=solpos["azimuth"],
            dni=df_weather["dni"],
            ghi=df_weather["ghi"],
            dhi=df_weather["dhi"],
            dni_extra=dni_extra,
            model="perez",
            albedo=albedo
        )
        
        poa = irr["poa_global"].clip(lower=0)
        
        # Apply far-shading
        if far_shading < 1.0:
            poa = poa * far_shading
        
        # AOI and IAM
        aoi = pvlib.irradiance.aoi(tilt, azimuth, solpos["zenith"], solpos["azimuth"])
        aoi_clipped = aoi.clip(0, 90)
        iam = np.interp(aoi_clipped, iam_angles, iam_values)
        
        # Effective POA
        poa_optical = poa * iam
        
        # Cell temperature (SAPM)
        cell_temp = pvlib.temperature.sapm_cell(
            poa_global=poa,
            temp_air=df_weather["air_temp"],
            wind_speed=df_weather["wind_speed"],
            a=sapm_params["a"],
            b=sapm_params["b"],
            deltaT=sapm_params["deltaT"]
        )
        
        # DC power per m²
        pdc_kwm2 = poa_optical * module_efficiency / 1000
        
        # Temperature coefficient
        pdc_kwm2_temp = pdc_kwm2 * (1 + gamma_p * (cell_temp - 25))
        
        # Apply DC losses
        pdc_kwm2_eff = pdc_kwm2_temp * dc_loss_factor
        
        # Scale to orientation area
        area_i = total_module_area * area_fraction
        pdc_total = pdc_kwm2_eff * area_i
        
        # Inverter efficiency (simplified flat efficiency from config)
        inv_eff = config["inverter"]["flat_efficiency"]
        pac = pdc_total * inv_eff
        
        plant_ac += pac
    
    # Inverter clipping
    plant_ac = plant_ac.clip(upper=inv_ac_rating)
    
    # AC wiring losses
    plant_ac = plant_ac * (1 - ac_wiring_loss)
    
    return plant_ac


# =====================================================================
# DATA GENERATION
# =====================================================================

def generate_random_samples(n_samples, config, seed=42):
    """
    Generate random, physically realistic weather samples.
    
    Parameters:
    -----------
    n_samples : int
        Number of samples to generate
    config : dict
        Plant configuration
    seed : int
        Random seed for reproducibility
    
    Returns:
    --------
    pd.DataFrame with columns: timestamp, ghi, dni, dhi, air_temp, wind_speed, ac_power_kw
    """
    np.random.seed(seed)
    
    print(f"\nGenerating {n_samples:,} random weather samples...")
    
    # Base year for timestamps
    base_year = 2025
    
    # Generate random timestamps across the year (daylight hours 5:00-19:00)
    day_of_year = np.random.randint(1, 366, size=n_samples)
    hour = np.random.uniform(5, 19, size=n_samples)
    minute = np.random.randint(0, 60, size=n_samples)
    
    # Create timestamps
    timestamps = []
    for i in range(n_samples):
        ts = datetime(base_year, 1, 1) + timedelta(
            days=int(day_of_year[i]) - 1,
            hours=float(hour[i]),
            minutes=int(minute[i])
        )
        timestamps.append(ts)
    
    # 1. Localize to Plant Timezone first (Asia/Colombo)
    # 2. Convert to UTC for consistency with the model
    timezone = config["location"]["timezone"] # "Asia/Colombo"
    timestamps = pd.to_datetime(timestamps).tz_localize(timezone).tz_convert("UTC")
    
    # Generate irradiance with physical constraints
    # Clearness factor determines overall conditions
    clearness = np.random.uniform(0, 1, size=n_samples)
    
    # Hour factor (solar elevation proxy)
    hour_factor = np.sin(np.pi * (hour - 5) / 14)  # 0 at sunrise/sunset, 1 at noon
    hour_factor = np.clip(hour_factor, 0, 1)
    
    # GHI: 0-1200 W/m², depends on time of day and clearness
    max_ghi = 1200 * hour_factor * (0.3 + 0.7 * clearness)
    ghi = np.random.uniform(0, np.maximum(1, max_ghi))
    
    # DNI: correlates with clearness
    # Clear sky: high DNI, overcast: low DNI
    dni_factor = np.where(clearness > 0.4, 
                          np.random.uniform(0.7, 1.2, n_samples),
                          np.random.uniform(0, 0.4, n_samples))
    dni = ghi * dni_factor * clearness
    dni = np.clip(dni, 0, 1000)
    
    # DHI: diffuse component
    # Higher under overcast, lower under clear sky
    dhi_factor = np.where(clearness < 0.5,
                          np.random.uniform(0.5, 0.9, n_samples),
                          np.random.uniform(0.1, 0.4, n_samples))
    dhi = ghi * dhi_factor
    dhi = np.clip(dhi, 0, 600)
    
    # Ensure GHI >= DNI*cos(zenith) + DHI approximately
    # (simplified - full constraint requires solar position)
    
    # Air temperature: 15-45°C (tropical climate)
    # Correlates slightly with irradiance (hotter during peak sun)
    base_temp = np.random.uniform(20, 35, size=n_samples)
    temp_variation = hour_factor * np.random.uniform(0, 10, size=n_samples)
    air_temp = base_temp + temp_variation
    air_temp = np.clip(air_temp, 15, 45)
    
    # Wind speed: exponential distribution, 0-15 m/s
    wind_speed = np.random.exponential(3, size=n_samples)
    wind_speed = np.clip(wind_speed, 0.1, 15)
    
    # Create DataFrame
    df = pd.DataFrame({
        "ghi": ghi,
        "dni": dni,
        "dhi": dhi,
        "air_temp": air_temp,
        "wind_speed": wind_speed
    }, index=timestamps)
    
    df = df.sort_index()
    
    print(f"  GHI range: {df['ghi'].min():.1f} - {df['ghi'].max():.1f} W/m²")
    print(f"  DNI range: {df['dni'].min():.1f} - {df['dni'].max():.1f} W/m²")
    print(f"  DHI range: {df['dhi'].min():.1f} - {df['dhi'].max():.1f} W/m²")
    print(f"  Temp range: {df['air_temp'].min():.1f} - {df['air_temp'].max():.1f} °C")
    print(f"  Wind range: {df['wind_speed'].min():.1f} - {df['wind_speed'].max():.1f} m/s")
    
    return df


def generate_calibration_data(n_samples, config, seed=42, batch_size=10000):
    """
    Generate calibration dataset with Python model outputs.
    
    Parameters:
    -----------
    n_samples : int
        Number of samples to generate
    config : dict
        Plant configuration
    seed : int
        Random seed
    batch_size : int
        Process in batches for memory efficiency
    
    Returns:
    --------
    pd.DataFrame with weather inputs and ac_power_kw from Python model
    """
    # Generate random weather samples
    df_weather = generate_random_samples(n_samples, config, seed)
    
    print(f"\nRunning Python physics model on {n_samples:,} samples...")
    print("  (This may take a few minutes for large datasets)")
    
    start_time = time.perf_counter()
    
    # Process in batches for progress indication
    all_ac_power = []
    
    n_batches = (n_samples + batch_size - 1) // batch_size
    
    for i in tqdm(range(n_batches), desc="Processing batches"):
        start_idx = i * batch_size
        end_idx = min((i + 1) * batch_size, n_samples)
        
        batch_df = df_weather.iloc[start_idx:end_idx]
        batch_ac = run_physics_model(batch_df, config)
        all_ac_power.append(batch_ac)
    
    # Combine results
    ac_power = pd.concat(all_ac_power)
    
    elapsed = time.perf_counter() - start_time
    print(f"  Completed in {elapsed:.1f}s ({n_samples/elapsed:.0f} samples/sec)")
    
    # Add AC power to dataframe
    df_weather["ac_power_kw"] = ac_power
    
    # Add timestamp as column for CSV export
    df_weather = df_weather.reset_index()
    df_weather = df_weather.rename(columns={"index": "timestamp"})
    
    # Round values for cleaner output
    df_weather["ghi"] = df_weather["ghi"].round(2)
    df_weather["dni"] = df_weather["dni"].round(2)
    df_weather["dhi"] = df_weather["dhi"].round(2)
    df_weather["air_temp"] = df_weather["air_temp"].round(2)
    df_weather["wind_speed"] = df_weather["wind_speed"].round(3)
    df_weather["ac_power_kw"] = df_weather["ac_power_kw"].round(6)
    
    return df_weather


# =====================================================================
# MAIN
# =====================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Generate calibration data for JS physics model",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python generate_calibration_data.py --samples 100000
  python generate_calibration_data.py --samples 1000000 --seed 42
        """
    )
    
    parser.add_argument(
        "--samples",
        type=int,
        default=100000,
        help="Number of samples to generate (default: 100000)"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42)"
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Output CSV path (default: data/js_calibration_data.csv)"
    )
    
    args = parser.parse_args()
    
    print("=" * 70)
    print("GENERATE CALIBRATION DATA FOR JS PHYSICS MODEL")
    print("=" * 70)
    print(f"Samples: {args.samples:,}")
    print(f"Random seed: {args.seed}")
    print("=" * 70)
    
    # Load config
    config = load_config()
    print(f"\nPlant: {config.get('plant_name', 'Unknown')}")
    print(f"Location: {config['location']['lat']:.4f}, {config['location']['lon']:.4f}")
    print(f"Inverter rating: {config['inverter']['ac_rating_kw']} kW")
    
    # Generate data
    df = generate_calibration_data(
        n_samples=args.samples,
        config=config,
        seed=args.seed
    )
    
    # Statistics
    print("\n" + "=" * 70)
    print("DATA STATISTICS")
    print("=" * 70)
    
    nonzero = df[df["ac_power_kw"] > 0]
    print(f"Total samples: {len(df):,}")
    print(f"Non-zero power samples: {len(nonzero):,} ({100*len(nonzero)/len(df):.1f}%)")
    print(f"AC power range: {df['ac_power_kw'].min():.3f} - {df['ac_power_kw'].max():.3f} kW")
    print(f"AC power mean (non-zero): {nonzero['ac_power_kw'].mean():.2f} kW")
    
    # Save output
    output_path = Path(args.output) if args.output else DATA_DIR / "js_calibration_data.csv"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    df.to_csv(output_path, index=False)
    
    print(f"\nSaved to: {output_path}")
    print(f"File size: {output_path.stat().st_size / 1024 / 1024:.1f} MB")
    
    # Save metadata
    metadata = {
        "samples": args.samples,
        "seed": args.seed,
        "created_at": datetime.now().isoformat(),
        "plant_name": config.get("plant_name", "Unknown"),
        "inverter_rating_kw": config["inverter"]["ac_rating_kw"],
        "statistics": {
            "nonzero_samples": int(len(nonzero)),
            "ac_power_min": float(df["ac_power_kw"].min()),
            "ac_power_max": float(df["ac_power_kw"].max()),
            "ac_power_mean_nonzero": float(nonzero["ac_power_kw"].mean())
        }
    }
    
    metadata_path = output_path.with_suffix(".json")
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)
    
    print(f"Metadata: {metadata_path}")
    
    print("\n" + "=" * 70)
    print("COMPLETE")
    print("=" * 70)
    print("\nNext step: Run optimize_js_params.py or optimize_js_params_gpu.py to calibrate JS model")
    print("=" * 70)
    
    return df


if __name__ == "__main__":
    main()

