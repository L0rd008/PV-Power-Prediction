#!/usr/bin/env python3
"""
================================================================================
UNIFIED PHYSICS-BASED PV GENERATION MODEL
================================================================================

Production-grade Solcast → PV Generation pipeline that reproduces PVsyst physics
and loss ordering as closely as possible using:

  - Perez POA transposition (with dni_extra for accuracy)
  - AOI + IAM from PVsyst
  - SAPM temperature matched to PVsyst Uc/Uv
  - Far-shading (when configured in plant_config.json)
  - Sequential DC loss chain (PVsyst order)
  - Plant-level inverter clipping (PVsyst methodology)
  - AC wiring losses
  - Correct PVsyst orientation weighting

USAGE:
------
  # Fetch from Solcast API (forecast endpoint)
  python physics_model.py --source api

  # Fetch from Solcast API (estimated_actuals - last 7 days)
  python physics_model.py --source api --endpoint estimated_actuals

  # Read from local CSV file (hourly data)
  python physics_model.py --source csv

  # Read from custom CSV path (hourly data)
  python physics_model.py --source csv --csv-path ../data/custom.csv

  # Fetch from NASA POWER hourly (free, no API key needed)
  python physics_model.py --source nasa_power --start 20251201 --end 20251215

  # Read from daily CSV file (full physics via synthetic hourly profiles)
  python physics_model.py --source csv_daily --csv-path ../data/daily.csv

  # Fetch from NASA POWER daily (full physics via synthetic hourly profiles)
  python physics_model.py --source nasa_power_daily --start 20251201 --end 20251215

  # Use simplified daily model (faster, less accurate)
  python physics_model.py --source nasa_power_daily --start 20251201 --end 20251215 --simplified

  # Show help
  python physics_model.py --help

NOTE:
-----
Exact numerical agreement with PVsyst requires validation against the
plant's PVsyst report and, where available, inverter efficiency curves.

================================================================================
"""

import argparse
import requests
import pandas as pd
import numpy as np
import pvlib
from pvlib.location import Location
import sys
import os
import json
from pathlib import Path
from datetime import datetime, timedelta

# Import NASA POWER fetcher
from fetch_nasa_power import fetch_nasa_power_hourly, fetch_nasa_power_daily

# =====================================================================
# CONFIGURATION LOADER
# =====================================================================

def load_config(config_path=None):
    """
    Load plant configuration from JSON file.
    
    Args:
        config_path: Path to config file. If None, uses default location.
    
    Returns:
        dict: Plant configuration parameters
    """
    if config_path is None:
        # Path: scripts/python_physics/ -> ../../config/
        config_path = Path(__file__).parent.parent.parent / "config" / "plant_config.json"
    
    with open(config_path, "r") as f:
        return json.load(f)

# Load configuration
config = load_config()

# =====================================================================
# ⚠️ CRITICAL USER INPUTS — MUST BE VERIFIED PER PLANT
# =====================================================================
# These parameters are NOT generic defaults.
# They MUST be verified for EACH new solar plant.
#
# If ANY of the following values are incorrect or assumed without
# validation, the generation estimate will be physically inconsistent
# and may deviate from PVsyst by >10–30%.
#
# IMPORTANT:
# PVsyst is the REFERENCE model.
# This script is a PVsyst-consistent reimplementation, not an optimizer.
#
# Sources for each parameter are explicitly noted below.
#
# ---------------------------------------------------------------------
# 1) Solcast API + Location
# ---------------------------------------------------------------------
# API_KEY:
#   → Your own Solcast API key
#   → Obtained from Solcast dashboard (NOT from PVsyst)
#
# lat, lon:
#   → Exact plant latitude / longitude
#   → PVsyst Simulation Report → Page 2 → "Geographical Site"
#   → Must match the coordinates used in the PVsyst simulation
#
# period:
#   → Time resolution of Solcast data
#   → Independent of PVsyst internal timestep
#   → Choose based on modeling intent:
#       - "PT60M" for hourly energy modeling
#       - "PT30M" / "PT15M" only if sub-hourly dynamics are required
#
# TIMEZONE:
#   → Local plant timezone (used ONLY for labeling / reporting)
#   → PVsyst Simulation Report → Page 2 → "Time zone"
#   → ALL internal calculations are performed in UTC
#
# ---------------------------------------------------------------------
# 2) Array Orientations (HIGH-RISK INPUT)
# ---------------------------------------------------------------------
# orientations:
#   → MUST be extracted from PVsyst "Orientation" sections
#   → Each UNIQUE (tilt, azimuth) combination is a separate entry
#
# For EACH orientation:
#   - tilt (degrees):
#       PVsyst report → "General parameters" → Orientation #n
#
#   - azimuth (degrees):
#       PVsyst convention:
#         0 = South
#         +90 = West
#         -90 = East
#         ±180 = North
#
#   - module_count:
#       → Derived from plant layout / as-built drawings
#       → Count = strings × modules per string per roof/array plane
#       → MUST satisfy:
#           sum(module_count) == total modules reported by PVsyst
#
# ⚠ Do NOT guess module counts.
# ⚠ Do NOT evenly split unless documentation is unavailable
#    (and clearly mark as approximation).
#
# ---------------------------------------------------------------------
# 3) Module Geometry & Electrical Parameters
# ---------------------------------------------------------------------
# module_area (m² per module):
#   → PRIMARY source: Manufacturer datasheet / as-built drawings
#   → Secondary cross-check: PVsyst report → "Total PV power" → Module area
#   → PVsyst reports TOTAL module area, not per-module geometry
#
# module_efficiency_stc:
#   → PVsyst report → Loss Diagram
#       ("efficiency at STC = xx.xx% PV conversion")
#   → This value is already PVsyst-consistent and SHOULD be used directly
#   → NOTE: May differ slightly from datasheet (use PVsyst value for validation)
#
# gamma_p (power temperature coefficient):
#   → NOT explicitly printed in the PVsyst report
#   → Source MUST be one of:
#       - PV module datasheet, OR
#       - PVsyst internal module database
#   → Typical crystalline silicon range:
#       -0.003 to -0.004 per °C
#
# ---------------------------------------------------------------------
# 4) Inverter & System-Level Parameters
# ---------------------------------------------------------------------
# INV_AC_RATING_KW:
#   → PVsyst report → Page 2 → "System summary"
#   → Use TOTAL inverter AC power (sum of all inverters): Pnom total
#   → Value in kW for direct use with kW-based power calculations
#
# PDC_THRESHOLD_KW:
#   → Inverter startup / minimum operating power (Pmin / Pthresh)
#   → NOT numerically exposed in PVsyst simulation reports
#   → Obtain from:
#       - PVsyst report → Loss Diagram → "Power threshold loss"
#       - If reported as 0.0%, set PDC_THRESHOLD_KW = 0.0
#       - Otherwise obtain from PVsyst inverter component (.OND), OR
#       - Manufacturer datasheet
#
# ---------------------------------------------------------------------
# 5) Optical & Shading Parameters
# ---------------------------------------------------------------------
# albedo:
#   → PVsyst report → Page 2 → Project settings → Albedo
#
# far_shading:
#   → PVsyst report → Loss Diagram → "Far shading" or "Horizon" loss
#   → Value = 1 - (loss % / 100)
#   → Set to 1.0 if no shading is defined in PVsyst
#   → Applied multiplicatively to POA irradiance BEFORE IAM
#
# NOTE ON SHADING / HORIZON LOSSES:
# Apply shading factor IF PVsyst report shows:
# - Near shading loss (from shading scene)
# - Far shading loss (from horizon profile)
# - Explicit shading loss percentage in Loss Diagram
#
# Do NOT apply shading factor if:
# - PVsyst shows "free horizon"
# - No shading scene is defined
# - Only "Global incident in coll. plane" loss appears
#   (this is transposition geometry, already in Perez)
#
# ---------------------------------------------------------------------
# 6) DC-Side Loss Factors (PVsyst-derived)
# ---------------------------------------------------------------------
# soiling, LID, mismatch, dc_wiring, module_quality:
#   → PVsyst report → "Array losses" section / Loss Diagram
#   → These are ANNUAL AVERAGE losses
#   → Applied on the DC side only
#   → Order must remain:
#       Soiling → LID → Module quality → Mismatch → DC wiring
#   → module_quality is often a GAIN (negative loss), e.g., +0.8%
#   → Extract EXACT values from PVsyst loss diagram
#
# ---------------------------------------------------------------------
# 7) AC-Side Loss Factors (PVsyst-derived)
# ---------------------------------------------------------------------
# ac_wiring_loss:
#   → PVsyst report → Loss Diagram → "AC ohmic loss"
#   → Applied AFTER inverter conversion, BEFORE reporting
#   → Typically 0.2–0.5%
#
# ---------------------------------------------------------------------
# 8) IAM Table (MODULE-SPECIFIC)
# ---------------------------------------------------------------------
# iam_angles / iam_values:
#   → PVsyst report → "IAM loss factor" → User-defined profile
#   → Module-specific optical behavior
#   → MUST NOT be reused across different module models
#   → Interpolated against angle-of-incidence (AOI)
#
# ---------------------------------------------------------------------
# 9) Thermal Model Parameters
# ---------------------------------------------------------------------
# sapm_params:
#   → Chosen to approximate PVsyst thermal model (Uc / Uv)
#   → PVsyst report → "Thermal Loss factor"
#   → PVsyst Uc/Uv values guide SAPM mounting selection
#   → IMPORTANT: SAPM and PVsyst use different thermal equations
#   → Exact cell temperature match is NOT possible
#   → Validation is performed at ANNUAL energy/PR level, not hourly
#
# VALIDATION RULE:
#   If annual AC energy or PR deviates from PVsyst by >3–5%,
#   revisit thermal model selection FIRST.
#
# ---------------------------------------------------------------------
# FINAL WARNING
# ---------------------------------------------------------------------
# This script is NOT plug-and-play.
# It is a PVsyst-consistent forward model.
#
# ALWAYS validate against PVsyst:
#   - Annual POA irradiation
#   - Annual DC energy (EArray)
#   - Annual AC energy (E_Grid)
#   - Performance Ratio (PR)
#
# BEFORE using outputs for BI, forecasting, or operations.
# =====================================================================


# =====================================================================
# 1) PARAMETERS FROM CONFIG FILE
# =====================================================================
# All plant-specific parameters are loaded from config/plant_config.json
# Modify that file for different plants - do not hardcode values here.

# API & Location
API_KEY = config["api"]["solcast_key"]
lat = config["location"]["lat"]
lon = config["location"]["lon"]
period = config["api"]["period"]
TIMEZONE = config["location"]["timezone"]
SITE_ALTITUDE_M = config["location"]["altitude_m"]

# Orientations
orientations = config["orientations"]

# Module parameters
module_area = config["module"]["area_m2"]
total_module_area = sum(o["module_count"] * module_area for o in orientations)
module_efficiency_stc = config["module"]["efficiency_stc"]
gamma_p = config["module"]["gamma_p"]

# Inverter parameters
INV_AC_RATING_KW = config["inverter"]["ac_rating_kw"]
PDC_THRESHOLD_KW = config["inverter"]["dc_threshold_kw"]

# Losses
soiling = config["losses"]["soiling"]
LID = config["losses"]["lid"]
module_quality = config["losses"]["module_quality"]
mismatch = config["losses"]["mismatch"]
dc_wiring = config["losses"]["dc_wiring"]
ac_wiring_loss = config["losses"]["ac_wiring"]
albedo = config["losses"]["albedo"]
far_shading = config["losses"].get("far_shading", 1.0)  # Default to 1.0 (no shading)

# DC loss factor (computed from individual losses)
dc_loss_factor = (1 - soiling) * (1 - LID) * (1 - module_quality) * (1 - mismatch) * (1 - dc_wiring)

# IAM table
iam_angles = np.array(config["iam"]["angles"])
iam_values = np.array(config["iam"]["values"])

# Default weather values
DEFAULT_WIND_SPEED_MS = config["defaults"]["wind_speed_ms"]
DEFAULT_AIR_TEMP_C = config["defaults"]["air_temp_c"]

# Thermal model
sapm_params = pvlib.temperature.TEMPERATURE_MODEL_PARAMETERS["sapm"][config["thermal_model"]]

# =====================================================================
# INVERTER EFFICIENCY MODEL (from config)
# =====================================================================
USE_INVERTER_CURVE = config["inverter"]["use_efficiency_curve"]
inverter_eff_curve_kw = np.array(config["inverter"]["efficiency_curve_kw"])
inverter_eff_curve_eta = np.array(config["inverter"]["efficiency_curve_eta"])
FLAT_EFFICIENCY = config["inverter"]["flat_efficiency"]

# Date range for Solcast data
start_date = config["date_range"]["start"]
end_date = config["date_range"]["end"]

def inverter_efficiency(pdc_kw):
    """
    Returns inverter efficiency η as a function of DC power (kW).
    Uses curve interpolation or flat efficiency based on config.
    """
    if USE_INVERTER_CURVE:
        return np.interp(np.asarray(pdc_kw), inverter_eff_curve_kw, inverter_eff_curve_eta)
    else:
        return FLAT_EFFICIENCY


# =====================================================================
# 2) SOLCAST API FETCH
# =====================================================================

# NOTE TO USERS:
# - This script supports multiple Solcast endpoints via --endpoint flag
# - For PVsyst validation, prefer Solcast TMY if available
# - For backtesting or SCADA comparison, use estimated_actuals endpoint
# - Change the endpoint URL below accordingly
#
# IMPORTANT: estimated_actuals endpoint only provides data for LAST 7 DAYS
# Ensure start_date and end_date fall within this window.
#
# If wind_speed or air_temp are unavailable in your Solcast plan,
# the model falls back to ERA5 or constant values.

def fetch_solcast_forecast(lat, lon, period, api_key):
    """Fetch Solcast forecast data (requires paid plan for weather)."""
    url = "https://api.solcast.com.au/data/forecast/radiation_and_weather"
    
    params = {
        "latitude": lat,
        "longitude": lon,
        "period": period,
        "api_key": api_key,
        "output_parameters": "ghi,dni,dhi,air_temp,wind_speed_10m",
        "format": "json"
    }
    
    print("Fetching Solcast forecast data...")
    r = requests.get(url, params=params)
    j = r.json()
    
    # If wind_speed is not allowed, retry without it
    if "response_status" in j:
        if j["response_status"].get("message", "").lower().startswith("invalid"):
            print("⚠️  Weather parameters not available, retrying with irradiance only...")
            params["output_parameters"] = "ghi,dni,dhi,air_temp"
            r = requests.get(url, params=params)
            j = r.json()
    
    return j


def fetch_solcast_estimated_actuals(lat, lon, period, api_key):
    """Fetch Solcast estimated actuals (last 7 days of historical data)."""
    url = "https://api.solcast.com.au/world_radiation/estimated_actuals"
    
    params = {
        "latitude": lat,
        "longitude": lon,
        "start": start_date,
        "end": end_date,
        "period": period,
        "api_key": api_key,
        "output_parameters": "ghi,dni,dhi",  # Basic tier parameters only
        "format": "json"
    }
    
    print(f"Fetching Solcast estimated actuals from {start_date} to {end_date}...")
    r = requests.get(url, params=params)
    
    if r.status_code != 200:
        print(f"\n⚠️ API ERROR (HTTP {r.status_code}):", r.text)
        sys.exit(1)
    
    return r.json()


def solcast_to_df(j):
    """Convert Solcast JSON response to DataFrame."""
    # Handle different Solcast response formats
    if "estimated_actuals" in j:
        rows = j["estimated_actuals"]
    elif "forecasts" in j:
        rows = j["forecasts"]
    elif "data" in j:
        rows = j["data"]
    else:
        raise ValueError(f"Solcast API did not return valid data. Keys found: {j.keys()}")
    
    df = pd.DataFrame(rows)
    
    # Normalize column names
    if "wind_speed_10m" in df.columns:
        df.rename(columns={"wind_speed_10m": "wind_speed"}, inplace=True)
    
    # Parse timestamps
    idx = pd.to_datetime(df["period_end"], utc=True)
    df.index = pd.DatetimeIndex(idx)
    
    return df


# =====================================================================
# 3) ERA5 WEATHER FALLBACK
# =====================================================================
# ERA5 is used as a secondary fallback when Solcast API does not provide
# air temperature or wind speed data. This provides physically consistent
# weather data for thermal modeling instead of constant values.
#
# Fallback hierarchy:
# 1. Solcast (primary) - if air_temp & wind_speed present
# 2. ERA5 (secondary) - if Solcast missing weather, fetch from ERA5
# 3. Constants (last resort) - if both fail, use DEFAULT values
#
# ERA5 Configuration:
# - Requires cdsapi library and CDS API credentials
# - Automatically attempts to fetch data for date range matching Solcast
# - Spatial averaging applied for point location extraction

def fetch_era5_weather(lat, lon, start_date_str, end_date_str):
    """
    Fetch ERA5 weather data (air temperature & wind speed) as fallback.
    
    Parameters:
    -----------
    lat, lon : float
        Location coordinates
    start_date_str, end_date_str : str
        ISO 8601 date strings (UTC), e.g., "2025-12-10T00:00:00Z"
    
    Returns:
    --------
    pd.DataFrame with columns: air_temp (°C), wind_speed (m/s)
    Index: UTC timestamps
    
    Returns None if ERA5 fetch fails.
    """
    try:
        import cdsapi
        import xarray as xr
        
        print("\n" + "="*60)
        print("ERA5 FALLBACK ACTIVATED")
        print("="*60)
        print("Solcast API did not provide air_temp or wind_speed.")
        print("Attempting to fetch from ERA5 (Copernicus CDS)...")
        
        # Parse date range from ISO strings
        start_dt = pd.to_datetime(start_date_str).tz_localize(None)
        end_dt = pd.to_datetime(end_date_str).tz_localize(None)
        
        OUTPUT_NETCDF = "era5_weather_temp.nc"
        
        # Initialize CDS API client
        c = cdsapi.Client()
        
        # Determine year/month/day ranges for ERA5 request
        if start_dt.month != end_dt.month:
            print("⚠️  ERA5 fetch currently supports single-month ranges only")
            return None

        year = str(start_dt.year)
        month = f"{start_dt.month:02d}"
        
        # Generate day list
        days = []
        current = start_dt
        while current <= end_dt:
            days.append(f"{current.day:02d}")
            current += timedelta(days=1)
        
        print(f"ERA5 request: {start_dt.date()} → {end_dt.date()}")
        print(f"Location: lat={lat}, lon={lon}")
        
        # Request ERA5 data
        c.retrieve(
            "reanalysis-era5-single-levels",
            {
                "product_type": "reanalysis",
                "variable": [
                    "2m_temperature",
                    "10m_u_component_of_wind",
                    "10m_v_component_of_wind",
                ],
                "year": year,
                "month": month,
                "day": days,
                "time": [f"{h:02d}:00" for h in range(24)],
                "area": [
                    lat + 0.1,  # North
                    lon - 0.1,  # West
                    lat - 0.1,  # South
                    lon + 0.1,  # East
                ],
                "format": "netcdf",
            },
            OUTPUT_NETCDF,
        )
        
        print("ERA5 download complete. Processing...")
        
        # Process NetCDF to DataFrame
        ds = xr.open_dataset(OUTPUT_NETCDF)
        df = ds.to_dataframe().reset_index()
        
        # Detect time column (ERA5-safe)
        if "time" in df.columns:
            time_col = "time"
        elif "valid_time" in df.columns:
            time_col = "valid_time"
        else:
            raise RuntimeError(f"Cannot find time column in ERA5 data. Columns: {df.columns}")
        
        # Spatial averaging (lat/lon grid → point)
        df = df.groupby(time_col, as_index=False).mean(numeric_only=True)
        
        # Unit conversions
        df["air_temp"] = df["t2m"] - 273.15  # Kelvin → Celsius
        df["wind_speed"] = np.sqrt(df["u10"]**2 + df["v10"]**2)  # m/s magnitude
        
        # Create final weather DataFrame
        weather_df = (
            df[[time_col, "air_temp", "wind_speed"]]
            .rename(columns={time_col: "time"})
            .set_index("time")
            .sort_index()
        )
        weather_df.index = pd.to_datetime(weather_df.index, utc=True)
        
        # Clean up temporary file
        if os.path.exists(OUTPUT_NETCDF):
            os.remove(OUTPUT_NETCDF)
        
        print("✓ ERA5 weather data successfully retrieved")
        print(f"  Temperature range: {weather_df['air_temp'].min():.1f}°C to {weather_df['air_temp'].max():.1f}°C")
        print(f"  Wind speed range: {weather_df['wind_speed'].min():.1f} to {weather_df['wind_speed'].max():.1f} m/s")
        print("="*60 + "\n")
        
        return weather_df
        
    except ImportError:
        print("⚠️  ERA5 fallback failed: cdsapi library not installed")
        print("   Install with: pip install cdsapi")
        return None
    except Exception as e:
        print(f"⚠️  ERA5 fallback failed: {str(e)}")
        print("   Check CDS API credentials: ~/.cdsapirc")
        return None


# =====================================================================
# 4) CSV DATA LOADING
# =====================================================================

def load_csv_data(csv_path):
    """
    Load irradiance data from local CSV file.
    
    Parameters:
    -----------
    csv_path : str or Path
        Path to CSV file containing irradiance data
    
    Returns:
    --------
    pd.DataFrame with DatetimeIndex (UTC) and columns: ghi, dni, dhi, [air_temp, wind_speed]
    """
    print(f"Loading irradiance data from CSV: {csv_path}")
    df = pd.read_csv(csv_path)
    
    # Find timestamp column
    time_col_candidates = ["time", "timestamp", "period_end", "datetime"]
    
    for col in time_col_candidates:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], utc=True)
            df = df.set_index(col)
            break
    else:
        raise RuntimeError("No valid timestamp column found in CSV. Expected one of: " + 
                          ", ".join(time_col_candidates))
    
    # Sort & validate
    df = df.sort_index()
    required_cols = ["ghi", "dni", "dhi"]
    missing = [c for c in required_cols if c not in df.columns]
    
    if missing:
        raise RuntimeError(f"CSV missing required irradiance columns: {missing}")
    
    # Normalize wind speed column names
    if "wind_speed" not in df.columns and "wind_speed_10m" in df.columns:
        print("ℹ️  Using wind_speed_10m as wind_speed")
        df["wind_speed"] = df["wind_speed_10m"]
    
    print(f"✓ Loaded {len(df)} records from {df.index.min()} to {df.index.max()}")
    
    return df


def load_csv_daily_data(csv_path):
    """
    Load daily irradiance data from local CSV file.
    
    Parameters:
    -----------
    csv_path : str or Path
        Path to CSV file containing daily irradiance data
    
    Returns:
    --------
    pd.DataFrame with DatetimeIndex (daily) and columns:
        - ghi_kwh_m2_day: Daily GHI in kWh/m²/day
        - dni_kwh_m2_day: Daily DNI in kWh/m²/day (optional)
        - dhi_kwh_m2_day: Daily DHI in kWh/m²/day (optional)
        - temp_c_daily: Daily average temperature in °C (optional)
        - wind_mps_daily: Daily average wind speed in m/s (optional)
    """
    print(f"Loading daily irradiance data from CSV: {csv_path}")
    df = pd.read_csv(csv_path)
    
    # Find date/timestamp column
    date_col_candidates = ["date", "time", "timestamp", "datetime", "day"]
    
    date_col = None
    for col in date_col_candidates:
        if col in df.columns:
            date_col = col
            break
    
    if date_col is None:
        raise RuntimeError("No valid date column found in CSV. Expected one of: " + 
                          ", ".join(date_col_candidates))
    
    # Parse dates and set as index
    df[date_col] = pd.to_datetime(df[date_col])
    df = df.set_index(date_col)
    df = df.sort_index()
    
    # Check for required columns - support multiple naming conventions
    ghi_col_candidates = ["ghi_kwh_m2_day", "ghi_daily", "ghi", "ALLSKY_SFC_SW_DWN"]
    dni_col_candidates = ["dni_kwh_m2_day", "dni_daily", "dni", "ALLSKY_SFC_SW_DNI"]
    dhi_col_candidates = ["dhi_kwh_m2_day", "dhi_daily", "dhi", "ALLSKY_SFC_SW_DIFF"]
    temp_col_candidates = ["temp_c_daily", "temp_daily", "temperature", "air_temp", "T2M"]
    wind_col_candidates = ["wind_mps_daily", "wind_daily", "wind_speed", "WS10M"]
    
    def find_and_rename(df, candidates, target_name):
        """Find column from candidates and rename to target."""
        for col in candidates:
            if col in df.columns:
                if col != target_name:
                    df = df.rename(columns={col: target_name})
                return df, True
        return df, False
    
    # Map columns to standard names
    df, has_ghi = find_and_rename(df, ghi_col_candidates, "ghi_kwh_m2_day")
    df, has_dni = find_and_rename(df, dni_col_candidates, "dni_kwh_m2_day")
    df, has_dhi = find_and_rename(df, dhi_col_candidates, "dhi_kwh_m2_day")
    df, has_temp = find_and_rename(df, temp_col_candidates, "temp_c_daily")
    df, has_wind = find_and_rename(df, wind_col_candidates, "wind_mps_daily")
    
    if not has_ghi:
        raise RuntimeError(f"CSV missing required GHI column. Expected one of: {ghi_col_candidates}")
    
    # Log what was found
    print(f"  GHI: {'✓' if has_ghi else '✗'}")
    print(f"  DNI: {'✓' if has_dni else '✗ (will be estimated)'}")
    print(f"  DHI: {'✓' if has_dhi else '✗ (will be estimated)'}")
    print(f"  Temperature: {'✓' if has_temp else '✗ (using default)'}")
    print(f"  Wind speed: {'✓' if has_wind else '✗ (using default)'}")
    
    print(f"✓ Loaded {len(df)} daily records from {df.index.min()} to {df.index.max()}")
    
    return df


# =====================================================================
# 5) WEATHER DATA PROCESSING
# =====================================================================

def ensure_weather_data(df):
    """
    Ensure air_temp and wind_speed are present in DataFrame.
    Uses fallback hierarchy: Solcast → ERA5 → Constants
    
    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame with irradiance data
    
    Returns:
    --------
    pd.DataFrame with air_temp and wind_speed columns filled
    """
    missing_temp = "air_temp" not in df.columns
    missing_wind = "wind_speed" not in df.columns
    
    if not missing_temp and not missing_wind:
        print("✓ Weather data available (air_temp & wind_speed)")
        return df
    
    print("\n" + "="*60)
    print("WEATHER DATA MISSING")
    print("="*60)
    if missing_temp:
        print("⚠️  Missing: air_temp")
    if missing_wind:
        print("⚠️  Missing: wind_speed")
    print("="*60)
    
    # Attempt ERA5 fallback
    era5_weather = fetch_era5_weather(lat, lon, start_date, end_date)
    
    if era5_weather is not None:
        # ERA5 fetch successful - merge with irradiance data
        print("Merging ERA5 weather data with irradiance data...")
        
        # Align ERA5 timestamps with irradiance (nearest neighbor merge)
        df = df.join(era5_weather, how="left")
        
        # Forward-fill any remaining gaps (typical for timestamp misalignment)
        if df["air_temp"].isna().any() or df["wind_speed"].isna().any():
            print("⚠️  Minor timestamp gaps detected - forward filling...")
            df["air_temp"] = df["air_temp"].ffill()
            df["wind_speed"] = df["wind_speed"].ffill()
        
        print("✓ Weather data successfully integrated from ERA5")
        
    else:
        # ERA5 fetch failed - fall back to constants
        print("\n" + "="*60)
        print("ERA5 FALLBACK FAILED")
        print("="*60)
        print("Falling back to constant weather values.")
        print("⚠️  WARNING: Constant weather is NOT valid for PVsyst validation")
        print("="*60 + "\n")
        
        if missing_temp:
            print(f"ℹ️  Injecting default temperature ({DEFAULT_AIR_TEMP_C}°C)")
            df["air_temp"] = DEFAULT_AIR_TEMP_C
        
        if missing_wind:
            print(f"ℹ️  Injecting default wind speed ({DEFAULT_WIND_SPEED_MS} m/s)")
            df["wind_speed"] = DEFAULT_WIND_SPEED_MS
    
    return df


# =====================================================================
# 6) PV MODEL (Perez POA → IAM → SAPM → DC → AC → Clipping)
# =====================================================================

def compute_pv_ac(df_weather):
    """
    Compute AC power output from weather data using physics-based model.
    
    Parameters:
    -----------
    df_weather : pd.DataFrame
        DataFrame with DatetimeIndex (UTC) and columns: ghi, dni, dhi, air_temp, wind_speed
    
    Returns:
    --------
    pd.Series with AC power output in kW
    """
    # Create site location
    site = Location(
        lat,
        lon,
        altitude=SITE_ALTITUDE_M if SITE_ALTITUDE_M is not None else 0
    )
    solpos = site.get_solarposition(df_weather.index)
    
    plant_ac = pd.Series(0.0, index=df_weather.index)
    
    # Extraterrestrial radiation for Perez model accuracy
    dni_extra = pvlib.irradiance.get_extra_radiation(df_weather.index)
    
    for o in orientations:
        tilt = o["tilt"]
        azimuth = o["azimuth"]
        area_fraction = (o["module_count"] * module_area) / total_module_area
        
        # Perez POA transposition (GHI/DNI/DHI → POA)
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
        
        # Apply far-shading BEFORE IAM (if configured)
        # far_shading = 1.0 means no shading, < 1.0 means shading loss
        if far_shading < 1.0:
            poa = poa * far_shading
        
        # AOI + IAM (optical correction)
        aoi = pvlib.irradiance.aoi(
            tilt, azimuth,
            solpos["zenith"], solpos["azimuth"]
        )
        iam = np.interp(aoi, iam_angles, iam_values)
        poa_optical = poa * iam
        
        # SAPM cell temperature (uses POA BEFORE IAM, per PVsyst methodology)
        # CRITICAL: Do NOT use poa_optical here; thermal losses depend on
        # total absorbed irradiance, not optically corrected irradiance
        cell_temp = pvlib.temperature.sapm_cell(
            poa,  # POA before IAM (correct per methodology)
            df_weather["air_temp"],
            df_weather["wind_speed"],
            **sapm_params
        )
        
        # DC power per m² (kW/m²)
        pdc_kwm2 = poa_optical * module_efficiency_stc / 1000
        
        # Apply temperature coefficient
        pdc_kwm2_temp = pdc_kwm2 * (1 + gamma_p * (cell_temp - 25))
        
        # Apply DC losses (PVsyst order: soiling → LID → module quality → mismatch → DC wiring)
        pdc_kwm2_eff = pdc_kwm2_temp * dc_loss_factor
        
        # Scale to orientation's total area
        area_i = total_module_area * area_fraction
        pdc_total_kw = pdc_kwm2_eff * area_i
        
        # Apply inverter DC threshold (Pmin / Pthresh)
        if PDC_THRESHOLD_KW > 0:
            pdc_total_kw = pdc_total_kw.where(pdc_total_kw >= PDC_THRESHOLD_KW, 0.0)
        
        # Convert DC → AC using inverter efficiency
        pac_kw = pdc_total_kw * inverter_efficiency(pdc_total_kw)
        
        # Accumulate per-orientation AC power (clipping happens later)
        plant_ac += pac_kw
    
    # Plant-level inverter clipping (AFTER orientation aggregation, per PVsyst)
    # CRITICAL: Do NOT clip per-orientation; PVsyst clips at plant level
    plant_ac = plant_ac.clip(upper=INV_AC_RATING_KW)
    
    # Apply AC wiring losses (AFTER inverter, BEFORE reporting)
    plant_ac = plant_ac * (1 - ac_wiring_loss)
    
    return plant_ac


# =====================================================================
# 6b) SIMPLIFIED DAILY MODEL (for NASA POWER daily data)
# =====================================================================

def calculate_daily_energy_simple(df_daily):
    """
    Simplified daily energy model using daily irradiance totals.
    
    This is a simplified approach for quick estimates when only daily
    irradiance data is available (e.g., NASA POWER daily endpoint).
    
    Less accurate than hourly model because:
    - No hour-by-hour Perez POA transposition
    - Uses average temperature for entire day
    - No sub-daily inverter clipping effects
    
    Parameters:
    -----------
    df_daily : pd.DataFrame
        Daily data with columns:
        - ghi_kwh_m2_day: Daily GHI in kWh/m²/day
        - dhi_kwh_m2_day: Daily DHI in kWh/m²/day
        - dni_kwh_m2_day: Daily DNI in kWh/m²/day
        - temp_c_daily: Daily average temperature in °C
        - wind_mps_daily: Daily average wind speed in m/s
    
    Returns:
    --------
    pd.Series with daily AC energy output in kWh
    """
    print("\n" + "="*60)
    print("SIMPLIFIED DAILY MODEL")
    print("="*60)
    
    # System parameters from config
    total_module_area = sum(o["module_count"] for o in orientations) * module_area
    
    # Average POA factor (ratio of POA irradiance to GHI for tilted arrays)
    # This is a simplification - actual value depends on tilt, azimuth, and season
    # For tropical locations with multiple orientations, 1.0-1.1 is typical
    avg_poa_factor = 1.05
    
    # System efficiency (simplified)
    # Includes: module efficiency, temperature losses, DC losses, inverter losses, AC losses
    temp_ref = 25.0
    temp_coeff = config["module"]["gamma_p"]  # Typically -0.0034
    
    # Daily losses (product of individual loss factors)
    loss_factors = [
        1 - config["losses"]["soiling"],
        1 - config["losses"]["lid"],
        1 + config["losses"]["module_quality"],  # Can be positive (quality gain)
        1 - config["losses"]["mismatch"],
        1 - config["losses"]["dc_wiring"],
        1 - config["losses"]["ac_wiring"],
        config["losses"]["far_shading"],
    ]
    dc_loss_factor = 1.0
    for f in loss_factors:
        dc_loss_factor *= f
    
    # Module STC efficiency
    module_eff = config["module"]["efficiency_stc"]
    
    # Inverter efficiency (simplified flat value)
    inv_eff = config["inverter"]["flat_efficiency"]
    
    # Calculate daily energy for each day
    daily_energy = []
    
    for idx, row in df_daily.iterrows():
        # Get irradiance (kWh/m²/day)
        ghi = row.get("ghi_kwh_m2_day", 0)
        
        # Skip invalid days
        if pd.isna(ghi) or ghi <= 0:
            daily_energy.append(np.nan)
            continue
        
        # Estimate POA irradiance (kWh/m²/day)
        poa_daily = ghi * avg_poa_factor
        
        # Temperature correction
        temp = row.get("temp_c_daily", temp_ref)
        if pd.isna(temp):
            temp = temp_ref
        temp_factor = 1 + temp_coeff * (temp - temp_ref)
        
        # DC energy before losses (kWh)
        dc_energy = poa_daily * total_module_area * module_eff * temp_factor
        
        # Apply DC losses
        dc_energy_net = dc_energy * dc_loss_factor
        
        # Apply inverter efficiency
        ac_energy = dc_energy_net * inv_eff
        
        # Apply inverter clipping (daily energy cap = 24h * AC rating)
        max_daily_energy = INV_AC_RATING_KW * 24
        ac_energy = min(ac_energy, max_daily_energy)
        
        daily_energy.append(ac_energy)
    
    result = pd.Series(daily_energy, index=df_daily.index, name="energy_kWh")
    
    print(f"Total days: {len(df_daily)}")
    print(f"Valid days: {result.notna().sum()}")
    print(f"Total energy: {result.sum():.1f} kWh")
    print(f"Average daily: {result.mean():.1f} kWh")
    print("="*60)
    
    return result


# =====================================================================
# 6c) SYNTHETIC HOURLY PROFILE GENERATOR (for daily data → full physics)
# =====================================================================

def generate_synthetic_hourly_from_daily(df_daily):
    """
    Generate synthetic hourly irradiance profiles from daily totals.
    
    Uses pvlib clear-sky models to create realistic temporal distributions,
    then scales them to match measured daily totals. This allows the full
    physics model to be applied to daily data.
    
    Parameters:
    -----------
    df_daily : pd.DataFrame
        Daily data with DatetimeIndex and columns:
        - ghi_kwh_m2_day: Daily GHI in kWh/m²/day
        - dni_kwh_m2_day: Daily DNI in kWh/m²/day
        - dhi_kwh_m2_day: Daily DHI in kWh/m²/day
        - temp_c_daily: Daily average temperature in °C (optional)
        - wind_mps_daily: Daily average wind speed in m/s (optional)
    
    Returns:
    --------
    pd.DataFrame with hourly data (DatetimeIndex UTC) and columns:
        ghi, dni, dhi (W/m²), air_temp (°C), wind_speed (m/s)
    """
    print("\n" + "="*60)
    print("SYNTHETIC HOURLY PROFILE GENERATOR")
    print("="*60)
    print("Converting daily irradiance to synthetic hourly profiles...")
    print("Using pvlib Ineichen clear-sky model for temporal distribution")
    print("="*60 + "\n")
    
    # Create site location for clear-sky calculation
    site = Location(
        lat, lon,
        altitude=SITE_ALTITUDE_M if SITE_ALTITUDE_M is not None else 0
    )
    
    # Collect all hourly data
    hourly_records = []
    
    for day_idx, row in df_daily.iterrows():
        # Parse the day (handle both datetime and date objects)
        if hasattr(day_idx, 'date'):
            day = day_idx.date() if hasattr(day_idx, 'date') else day_idx
        else:
            day = pd.to_datetime(day_idx).date()
        
        # Get daily irradiance totals (kWh/m²/day)
        ghi_daily = row.get("ghi_kwh_m2_day", np.nan)
        dni_daily = row.get("dni_kwh_m2_day", np.nan)
        dhi_daily = row.get("dhi_kwh_m2_day", np.nan)
        
        # Get weather data (use defaults if not available)
        temp_daily = row.get("temp_c_daily", DEFAULT_AIR_TEMP_C)
        wind_daily = row.get("wind_mps_daily", DEFAULT_WIND_SPEED_MS)
        
        if pd.isna(temp_daily):
            temp_daily = DEFAULT_AIR_TEMP_C
        if pd.isna(wind_daily):
            wind_daily = DEFAULT_WIND_SPEED_MS
        
        # Skip days with invalid irradiance
        if pd.isna(ghi_daily) or ghi_daily <= 0:
            print(f"  Skipping {day}: invalid GHI data")
            continue
        
        # Generate hourly timestamps for this day (UTC)
        day_start = pd.Timestamp(day, tz="UTC")
        hourly_times = pd.date_range(
            start=day_start,
            end=day_start + pd.Timedelta(hours=23),
            freq="h"
        )
        
        # Get clear-sky irradiance for this day
        clearsky = site.get_clearsky(hourly_times, model="ineichen")
        
        # Calculate clear-sky daily totals (W/m² → kWh/m²/day, 1 hour per step)
        cs_ghi_total = clearsky["ghi"].sum() / 1000  # W·h/m² → kWh/m²
        cs_dni_total = clearsky["dni"].sum() / 1000
        cs_dhi_total = clearsky["dhi"].sum() / 1000
        
        # Handle edge cases: very cloudy days or nighttime-only data
        MIN_CLEARSKY_THRESHOLD = 0.01  # kWh/m²/day
        
        # Calculate scaling factors (measured / clear-sky)
        if cs_ghi_total > MIN_CLEARSKY_THRESHOLD:
            ghi_scale = ghi_daily / cs_ghi_total
        else:
            ghi_scale = 0.0
        
        if cs_dni_total > MIN_CLEARSKY_THRESHOLD and not pd.isna(dni_daily):
            dni_scale = dni_daily / cs_dni_total
        else:
            # If DNI not provided or clear-sky is zero, estimate from GHI
            dni_scale = ghi_scale * 0.7 if not pd.isna(dni_daily) else 0.0
        
        if cs_dhi_total > MIN_CLEARSKY_THRESHOLD and not pd.isna(dhi_daily):
            dhi_scale = dhi_daily / cs_dhi_total
        else:
            # If DHI not provided, estimate as remainder
            dhi_scale = ghi_scale * 0.3 if not pd.isna(dhi_daily) else ghi_scale
        
        # Apply scaling to get synthetic hourly values (W/m²)
        ghi_hourly = clearsky["ghi"] * ghi_scale
        dni_hourly = clearsky["dni"] * dni_scale
        dhi_hourly = clearsky["dhi"] * dhi_scale
        
        # Clamp negative values (can occur with unusual scaling)
        ghi_hourly = ghi_hourly.clip(lower=0)
        dni_hourly = dni_hourly.clip(lower=0)
        dhi_hourly = dhi_hourly.clip(lower=0)
        
        # Build hourly records for this day
        for i, ts in enumerate(hourly_times):
            hourly_records.append({
                "timestamp": ts,
                "ghi": ghi_hourly.iloc[i],
                "dni": dni_hourly.iloc[i],
                "dhi": dhi_hourly.iloc[i],
                "air_temp": temp_daily,  # Constant for the day
                "wind_speed": wind_daily,  # Constant for the day
            })
    
    if not hourly_records:
        raise RuntimeError("No valid daily data to convert to hourly profiles")
    
    # Create DataFrame from records
    df_hourly = pd.DataFrame(hourly_records)
    df_hourly = df_hourly.set_index("timestamp")
    df_hourly.index = pd.DatetimeIndex(df_hourly.index, tz="UTC")
    df_hourly = df_hourly.sort_index()
    
    # Summary statistics
    total_days = len(df_daily)
    valid_days = len(df_hourly) // 24
    
    print(f"✓ Generated synthetic hourly profiles")
    print(f"  Input days: {total_days}")
    print(f"  Valid days processed: {valid_days}")
    print(f"  Total hourly records: {len(df_hourly)}")
    print(f"  GHI range: {df_hourly['ghi'].min():.1f} - {df_hourly['ghi'].max():.1f} W/m²")
    print(f"  DNI range: {df_hourly['dni'].min():.1f} - {df_hourly['dni'].max():.1f} W/m²")
    print(f"  DHI range: {df_hourly['dhi'].min():.1f} - {df_hourly['dhi'].max():.1f} W/m²")
    
    return df_hourly


# =====================================================================
# 7) MAIN EXECUTION
# =====================================================================

def main():
    """Main entry point with CLI argument parsing."""
    parser = argparse.ArgumentParser(
        description="Physics-based PV generation model",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python physics_model.py --source api
  python physics_model.py --source api --endpoint estimated_actuals
  python physics_model.py --source csv
  python physics_model.py --source csv --csv-path ../data/custom.csv
  python physics_model.py --source csv_daily --csv-path ../data/daily.csv
  python physics_model.py --source nasa_power_daily --start 20251201 --end 20251215
  python physics_model.py --source nasa_power_daily --start 20251201 --end 20251215 --simplified
        """
    )
    
    parser.add_argument(
        "--source",
        choices=["api", "csv", "csv_daily", "nasa_power", "nasa_power_daily"],
        default="csv",
        help="Data source: 'api' for Solcast API, 'csv' for hourly CSV, "
             "'csv_daily' for daily CSV, 'nasa_power' for NASA POWER hourly, "
             "'nasa_power_daily' for NASA POWER daily (default: csv)"
    )
    
    parser.add_argument(
        "--endpoint",
        choices=["forecast", "estimated_actuals"],
        default="forecast",
        help="Solcast API endpoint (default: forecast)"
    )
    
    parser.add_argument(
        "--csv-path",
        default=str(Path(__file__).parent.parent.parent / "data" / "solcast_irradiance.csv"),
        help="Path to CSV file when using --source csv"
    )
    
    parser.add_argument(
        "--start",
        default=None,
        help="Start date in YYYYMMDD format (for NASA POWER)"
    )
    
    parser.add_argument(
        "--end",
        default=None,
        help="End date in YYYYMMDD format (for NASA POWER)"
    )
    
    parser.add_argument(
        "--output",
        default=str(Path(__file__).parent.parent.parent / "output" / "pv_generation.csv"),
        help="Output CSV path"
    )
    
    parser.add_argument(
        "--simplified",
        action="store_true",
        help="Use simplified daily model instead of synthetic hourly profiles "
             "(only for daily sources: csv_daily, nasa_power_daily)"
    )
    
    args = parser.parse_args()
    
    print("\n" + "="*60)
    print("PHYSICS-BASED PV GENERATION MODEL")
    print("="*60)
    print(f"Data source: {args.source}")
    if args.source == "api":
        print(f"API endpoint: {args.endpoint}")
    elif args.source in ["csv", "csv_daily"]:
        print(f"CSV path: {args.csv_path}")
    if args.source in ["nasa_power", "nasa_power_daily"]:
        print(f"NASA POWER dates: {args.start} to {args.end}")
    if args.source in ["csv_daily", "nasa_power_daily"]:
        if args.simplified:
            print("Model: Simplified daily (fast, less accurate)")
        else:
            print("Model: Full physics via synthetic hourly profiles")
    print(f"Far-shading factor: {far_shading}")
    print("="*60 + "\n")
    
    # =====================================================================
    # LOAD DATA
    # =====================================================================
    
    if args.source == "api":
        # Fetch from Solcast API
        if args.endpoint == "forecast":
            raw = fetch_solcast_forecast(lat, lon, period, API_KEY)
        else:
            raw = fetch_solcast_estimated_actuals(lat, lon, period, API_KEY)
        
        # Check for API errors
        if "forecasts" not in raw and "data" not in raw and "estimated_actuals" not in raw:
            print("\n⚠️ API ERROR RECEIVED:")
            print(raw)
            sys.exit(1)
        
        df = solcast_to_df(raw)
    
    elif args.source == "nasa_power":
        # Fetch hourly data from NASA POWER (W/m², same units as Solcast)
        if not args.start or not args.end:
            print("⚠️ ERROR: --start and --end required for NASA POWER source")
            print("   Example: --start 20251201 --end 20251215")
            sys.exit(1)
        
        df = fetch_nasa_power_hourly(lat, lon, args.start, args.end)
        
        # Rename columns to match expected format
        df = df.rename(columns={"wind_speed": "wind_speed_10m"})
        
        # Set index to period_end
        df["period_end"] = pd.to_datetime(df["period_end"])
        df = df.set_index("period_end")
        
        # Rename for compatibility
        if "wind_speed_10m" in df.columns:
            df = df.rename(columns={"wind_speed_10m": "wind_speed"})
    
    elif args.source == "csv_daily":
        # Load daily data from CSV file
        df_daily = load_csv_daily_data(args.csv_path)
        
        if args.simplified:
            # Use simplified daily model (fast, less accurate)
            daily_energy = calculate_daily_energy_simple(df_daily)
            
            # Convert to local timezone for output
            daily_energy_local = daily_energy.tz_localize(TIMEZONE)
            
            # Save output
            output_path = Path(args.output)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            out = pd.DataFrame({"energy_kWh": daily_energy_local})
            out.to_csv(output_path)
            
            # Results summary
            print("\n" + "="*60)
            print("RESULTS (DAILY SIMPLIFIED MODEL)")
            print("="*60)
            print(f"Output saved: {output_path}")
            print(f"Total energy (kWh): {daily_energy.sum():.1f}")
            print(f"Period: {df_daily.index.min()} to {df_daily.index.max()}")
            print("="*60)
            
            print("\nDaily energy output:")
            print(out)
            
            return daily_energy
        else:
            # Use full physics model via synthetic hourly profiles
            df = generate_synthetic_hourly_from_daily(df_daily)
    
    elif args.source == "nasa_power_daily":
        # Fetch daily data from NASA POWER
        if not args.start or not args.end:
            print("⚠️ ERROR: --start and --end required for NASA POWER source")
            print("   Example: --start 20251201 --end 20251215")
            sys.exit(1)
        
        df_daily = fetch_nasa_power_daily(lat, lon, args.start, args.end)
        
        if args.simplified:
            # Use simplified daily model (fast, less accurate)
            daily_energy = calculate_daily_energy_simple(df_daily)
            
            # Convert to local timezone for output
            daily_energy_local = daily_energy.tz_localize(TIMEZONE)
            
            # Save output
            output_path = Path(args.output)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            out = pd.DataFrame({"energy_kWh": daily_energy_local})
            out.to_csv(output_path)
            
            # Results summary
            print("\n" + "="*60)
            print("RESULTS (DAILY SIMPLIFIED MODEL)")
            print("="*60)
            print(f"Output saved: {output_path}")
            print(f"Total energy (kWh): {daily_energy.sum():.1f}")
            print(f"Period: {df_daily.index.min()} to {df_daily.index.max()}")
            print("="*60)
            
            print("\nDaily energy output:")
            print(out)
            
            return daily_energy
        else:
            # Use full physics model via synthetic hourly profiles
            df = generate_synthetic_hourly_from_daily(df_daily)
        
    else:
        # Load from CSV
        df = load_csv_data(args.csv_path)
    
    # =====================================================================
    # ENSURE WEATHER DATA
    # =====================================================================
    
    df = ensure_weather_data(df)
    
    # =====================================================================
    # TIMESTAMP VALIDATION
    # =====================================================================
    
    # Ensure time index is strictly increasing
    if not df.index.is_monotonic_increasing:
        print("⚠️  Timestamp order issue detected – sorting index")
        df = df.sort_index()
    
    # Enforce regular hourly grid if source data drifts
    EXPECTED_FREQ = "h"
    inferred_freq = pd.infer_freq(df.index)
    if inferred_freq != EXPECTED_FREQ and inferred_freq != "H":
        print(f"⚠️  Irregular timestep detected (inferred: {inferred_freq})")
        print("   Resampling to nearest hourly timestamps")
        df = df.resample(EXPECTED_FREQ).nearest()
    
    # =====================================================================
    # COMPUTE PV OUTPUT
    # =====================================================================
    
    print("\nComputing PV AC output...")
    plant_ac = compute_pv_ac(df)
    
    # Compute timestep duration for energy integration
    dt_hours = (df.index[1] - df.index[0]).total_seconds() / 3600
    
    # Convert to local timezone for reporting/output only
    plant_ac_local = plant_ac.tz_convert(TIMEZONE)
    
    # Integrate power to energy (kWh)
    energy = (plant_ac_local * dt_hours).sum()
    
    # =====================================================================
    # SAVE OUTPUT
    # =====================================================================
    
    # Ensure output directory exists
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    out = pd.DataFrame({"AC_kW": plant_ac_local})
    out.to_csv(output_path)
    
    # =====================================================================
    # RESULTS SUMMARY
    # =====================================================================
    
    print("\n" + "="*60)
    print("RESULTS")
    print("="*60)
    print(f"Output saved: {output_path}")
    print(f"Total energy (kWh): {energy:.1f}")
    print(f"Period: {df.index.min()} to {df.index.max()}")
    print(f"Data source: {args.source}")
    print("="*60)
    
    print("\nSample output (first 10 rows):")
    print(out.head(10))
    
    print("\n" + "="*60)
    print("VALIDATION CHECKLIST (compare with PVsyst report)")
    print("="*60)
    print("□ Annual POA irradiation (kWh/m²)")
    print("□ Annual DC energy (EArray)")
    print("□ Annual AC energy (E_Grid)")
    print("□ Performance Ratio (PR)")
    print("□ Acceptable deviation: ±2-5%")
    print("="*60)
    
    return plant_ac


if __name__ == "__main__":
    main()
