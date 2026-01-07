#!/usr/bin/env python3
"""
================================================================================
DAILY POWER PREDICTOR
================================================================================

Runs daily to predict power generation for today and tomorrow using 3 different
primary data sources with cascade fallbacks for missing fields.

OUTPUT FILES:
-------------
  output/predictions_nasa_power.csv  - Primary: NASA POWER, Fallback: Solcast → Open-Meteo → ERA5
  output/predictions_solcast.csv     - Primary: Solcast, Fallback: Open-Meteo → NASA → ERA5
  output/predictions_openmeteo.csv   - Primary: Open-Meteo, Fallback: Solcast → NASA → ERA5
  output/predictions_era5.csv        - Primary: ERA5, Fallback: Solcast → Open-Meteo → NASA

Each file contains:
  timestamp, predicted_power_kw, ghi, dni, dhi, air_temp, wind_speed,
  irradiance_source, temp_source, wind_source

USAGE:
------
  # Default: predict for today + tomorrow
  python daily_predictor.py

  # Today only
  python daily_predictor.py --today-only

  # Tomorrow only
  python daily_predictor.py --tomorrow-only

  # Custom date range
  python daily_predictor.py --start 20260107 --end 20260108

================================================================================
"""

import argparse
import pandas as pd
import numpy as np
import json
import logging
from pathlib import Path
from datetime import datetime, timedelta, timezone
import sys

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

# =====================================================================
# CONFIGURATION
# =====================================================================

SCRIPT_DIR = Path(__file__).parent
PROJECT_DIR = SCRIPT_DIR.parent.parent
CONFIG_PATH = PROJECT_DIR / "config" / "plant_config.json"
OUTPUT_DIR = PROJECT_DIR / "output"

# Output directories for each granularity
HOURLY_DIR = OUTPUT_DIR / "hourly"
DAILY_DIR = OUTPUT_DIR / "daily"
MONTHLY_DIR = OUTPUT_DIR / "monthly"

# Output file paths per source (hourly is the base)
OUTPUT_FILES = {
    "nasa": "predictions_nasa_power.csv",
    "solcast": "predictions_solcast.csv",
    "openmeteo": "predictions_openmeteo.csv",
    "era5": "predictions_era5.csv",
}

# Fallback priority order for each primary source
FALLBACK_ORDER = {
    "nasa": ["nasa", "solcast", "openmeteo", "era5"],
    "solcast": ["solcast", "openmeteo", "nasa", "era5"],
    "openmeteo": ["openmeteo", "solcast", "nasa", "era5"],
    "era5": ["era5", "solcast", "openmeteo", "nasa"],
}


def load_config():
    """Load plant configuration from JSON file."""
    with open(CONFIG_PATH, "r") as f:
        return json.load(f)


# =====================================================================
# DATA FETCHERS
# =====================================================================

def fetch_nasa_power_data(lat, lon, start_date, end_date):
    """
    Fetch hourly data from NASA POWER API.
    
    Returns DataFrame with columns: ghi, dni, dhi, air_temp, wind_speed
    Index: UTC datetime
    """
    from fetch_nasa_power import fetch_nasa_power_hourly
    
    try:
        logger.info("Fetching NASA POWER data...")
        start_str = start_date.strftime("%Y%m%d")
        end_str = end_date.strftime("%Y%m%d")
        
        df = fetch_nasa_power_hourly(lat, lon, start_str, end_str)
        
        # Set index
        df["period_end"] = pd.to_datetime(df["period_end"])
        df = df.set_index("period_end")
        
        # Standardize column names
        df = df.rename(columns={"wind_speed": "wind_speed_nasa"})
        df = df.rename(columns={"wind_speed_nasa": "wind_speed"})
        
        logger.info(f"✓ NASA POWER: {len(df)} records fetched")
        return df[["ghi", "dni", "dhi", "air_temp", "wind_speed"]]
        
    except Exception as e:
        logger.warning(f"⚠️ NASA POWER fetch failed: {e}")
        return None


def fetch_solcast_data(lat, lon, api_key, start_date, end_date, endpoint="estimated_actuals"):
    """
    Fetch hourly data from Solcast API.
    
    For today: uses estimated_actuals endpoint
    For tomorrow: uses forecast endpoint
    
    Returns DataFrame with columns: ghi, dni, dhi, air_temp, wind_speed
    Index: UTC datetime
    """
    import requests
    
    try:
        logger.info(f"Fetching Solcast data ({endpoint})...")
        
        if endpoint == "estimated_actuals":
            url = "https://api.solcast.com.au/world_radiation/estimated_actuals"
            params = {
                "latitude": lat,
                "longitude": lon,
                "start": start_date.isoformat(),
                "end": end_date.isoformat(),
                "period": "PT60M",
                "api_key": api_key,
                "output_parameters": "ghi,dni,dhi",
                "format": "json"
            }
        else:
            url = "https://api.solcast.com.au/data/forecast/radiation_and_weather"
            params = {
                "latitude": lat,
                "longitude": lon,
                "period": "PT60M",
                "api_key": api_key,
                "output_parameters": "ghi,dni,dhi,air_temp,wind_speed_10m",
                "format": "json"
            }
        
        resp = requests.get(url, params=params, timeout=30)
        
        if resp.status_code != 200:
            logger.warning(f"Solcast API error: HTTP {resp.status_code}")
            return None
        
        data = resp.json()
        
        # Parse response
        if "estimated_actuals" in data:
            rows = data["estimated_actuals"]
        elif "forecasts" in data:
            rows = data["forecasts"]
        else:
            logger.warning("Solcast API returned unexpected format")
            return None
        
        df = pd.DataFrame(rows)
        
        # Parse timestamps
        df["period_end"] = pd.to_datetime(df["period_end"], utc=True)
        df = df.set_index("period_end")
        
        # Standardize column names
        if "wind_speed_10m" in df.columns:
            df = df.rename(columns={"wind_speed_10m": "wind_speed"})
        
        # Add missing columns with NaN
        for col in ["ghi", "dni", "dhi", "air_temp", "wind_speed"]:
            if col not in df.columns:
                df[col] = np.nan
        
        logger.info(f"✓ Solcast: {len(df)} records fetched")
        return df[["ghi", "dni", "dhi", "air_temp", "wind_speed"]]
        
    except Exception as e:
        logger.warning(f"⚠️ Solcast fetch failed: {e}")
        return None


def fetch_era5_data(lat, lon, start_date, end_date):
    """
    Fetch hourly data from ERA5 (Copernicus CDS).
    
    Note: ERA5 only provides air_temp and wind_speed, not irradiance.
    Irradiance would require separate processing of SSRD variable.
    
    Returns DataFrame with columns: air_temp, wind_speed (irradiance as NaN)
    Index: UTC datetime
    """
    try:
        import cdsapi
        import xarray as xr
        import os
        
        logger.info("Fetching ERA5 data...")
        
        # Temporary file for download
        temp_file = OUTPUT_DIR / "era5_temp.nc"
        
        # ERA5 needs dates as strings
        year = str(start_date.year)
        month = f"{start_date.month:02d}"
        
        # Generate day list
        days = []
        current = start_date
        while current <= end_date:
            days.append(f"{current.day:02d}")
            current += timedelta(days=1)
        
        # Initialize CDS API client
        c = cdsapi.Client()
        
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
            str(temp_file),
        )
        
        # Process NetCDF - use context manager to ensure file is closed
        with xr.open_dataset(temp_file) as ds:
            df = ds.to_dataframe().reset_index()
        
        # Detect time column
        if "time" in df.columns:
            time_col = "time"
        elif "valid_time" in df.columns:
            time_col = "valid_time"
        else:
            raise RuntimeError(f"Cannot find time column: {df.columns}")
        
        # Spatial averaging
        df = df.groupby(time_col, as_index=False).mean(numeric_only=True)
        
        # Unit conversions
        df["air_temp"] = df["t2m"] - 273.15  # K → °C
        df["wind_speed"] = np.sqrt(df["u10"]**2 + df["v10"]**2)
        
        # Create result DataFrame
        result = pd.DataFrame({
            "air_temp": df["air_temp"].values,
            "wind_speed": df["wind_speed"].values,
            "ghi": np.nan,  # ERA5 doesn't provide these directly
            "dni": np.nan,
            "dhi": np.nan,
        })
        result.index = pd.to_datetime(df[time_col], utc=True)
        result.index.name = "period_end"
        
        # Cleanup - file should be closed now
        try:
            if temp_file.exists():
                os.remove(temp_file)
        except Exception as cleanup_err:
            logger.warning(f"Could not remove temp file: {cleanup_err}")
        
        logger.info(f"✓ ERA5: {len(result)} records fetched")
        return result
        
    except ImportError:
        logger.warning("⚠️ ERA5 fetch failed: cdsapi not installed")
        return None
    except Exception as e:
        logger.warning(f"⚠️ ERA5 fetch failed: {e}")
        return None


def fetch_openmeteo_data(lat, lon, start_date, end_date):
    """
    Fetch forecast data from Open-Meteo (free, no API key required).
    
    Provides: GHI, DNI, DHI, temperature, wind speed
    Available for forecasts (up to 16 days ahead) and recent past (~1 week)
    
    Returns DataFrame with columns: ghi, dni, dhi, air_temp, wind_speed
    Index: UTC datetime
    """
    import requests
    
    try:
        logger.info("Fetching Open-Meteo data...")
        
        url = "https://api.open-meteo.com/v1/forecast"
        params = {
            "latitude": lat,
            "longitude": lon,
            "hourly": "direct_normal_irradiance,diffuse_radiation,shortwave_radiation,temperature_2m,wind_speed_10m",
            "timezone": "UTC",
            "start_date": start_date.strftime("%Y-%m-%d"),
            "end_date": end_date.strftime("%Y-%m-%d")
        }
        
        resp = requests.get(url, params=params, timeout=30)
        
        if resp.status_code != 200:
            logger.warning(f"Open-Meteo API error: HTTP {resp.status_code}")
            return None
        
        data = resp.json()
        
        if "hourly" not in data:
            logger.warning("Open-Meteo returned unexpected format")
            return None
        
        hourly = data["hourly"]
        
        # Build DataFrame
        df = pd.DataFrame({
            "time": pd.to_datetime(hourly["time"]),
            "ghi": hourly.get("shortwave_radiation", [np.nan] * len(hourly["time"])),
            "dni": hourly.get("direct_normal_irradiance", [np.nan] * len(hourly["time"])),
            "dhi": hourly.get("diffuse_radiation", [np.nan] * len(hourly["time"])),
            "air_temp": hourly.get("temperature_2m", [np.nan] * len(hourly["time"])),
            "wind_speed": hourly.get("wind_speed_10m", [np.nan] * len(hourly["time"])),
        })
        
        # Set index
        df["time"] = df["time"].dt.tz_localize("UTC")
        df = df.set_index("time")
        df.index.name = "period_end"
        
        # Clean data - clip negative irradiance
        for col in ["ghi", "dni", "dhi"]:
            if col in df.columns:
                df[col] = df[col].clip(lower=0)
        
        logger.info(f"✓ Open-Meteo: {len(df)} records fetched")
        return df[["ghi", "dni", "dhi", "air_temp", "wind_speed"]]
        
    except Exception as e:
        logger.warning(f"⚠️ Open-Meteo fetch failed: {e}")
        return None


# =====================================================================
# CASCADE FALLBACK LOGIC
# =====================================================================

def get_field_with_fallback(timestamp, field_name, sources_priority, all_data):
    """
    Try sources in priority order until valid value found.
    
    Args:
        timestamp: The hour to get data for
        field_name: 'ghi', 'dni', 'dhi', 'air_temp', or 'wind_speed'
        sources_priority: ['nasa', 'solcast', 'era5'] in order
        all_data: Dict of DataFrames from each source
    
    Returns:
        (value, source_name) or (None, None) if all fail
    """
    for source in sources_priority:
        df = all_data.get(source)
        if df is None:
            continue
        
        if timestamp not in df.index:
            continue
        
        val = df.loc[timestamp, field_name]
        
        # Check for valid value (not NaN, and >= 0 for irradiance)
        if pd.notna(val):
            if field_name in ["ghi", "dni", "dhi"] and val < 0:
                continue
            return val, source
    
    return None, None


def merge_data_with_fallback(timestamps, primary_source, all_data):
    """
    Merge data from multiple sources using cascade fallback.
    
    Args:
        timestamps: List of timestamps to process
        primary_source: 'nasa', 'solcast', or 'era5'
        all_data: Dict of DataFrames from each source
    
    Returns:
        DataFrame with merged data and source tracking columns
    """
    fallback_order = FALLBACK_ORDER[primary_source]
    
    rows = []
    for ts in timestamps:
        row = {"timestamp": ts}
        
        # Get irradiance components
        for field in ["ghi", "dni", "dhi"]:
            val, src = get_field_with_fallback(ts, field, fallback_order, all_data)
            row[field] = val
            if field == "ghi":  # Track irradiance source
                row["irradiance_source"] = src
        
        # Get temperature
        val, src = get_field_with_fallback(ts, "air_temp", fallback_order, all_data)
        row["air_temp"] = val
        row["temp_source"] = src
        
        # Get wind speed
        val, src = get_field_with_fallback(ts, "wind_speed", fallback_order, all_data)
        row["wind_speed"] = val
        row["wind_source"] = src
        
        rows.append(row)
    
    df = pd.DataFrame(rows)
    df = df.set_index("timestamp")
    
    return df


# =====================================================================
# PHYSICS MODEL (simplified from physics_model.py)
# =====================================================================

def compute_pv_power(df, config):
    """
    Compute AC power output from weather data using physics-based model.
    
    This is a simplified version of the full physics_model.py for daily use.
    
    Args:
        df: DataFrame with ghi, dni, dhi, air_temp, wind_speed
        config: Plant configuration dict
    
    Returns:
        Series with AC power in kW
    """
    import pvlib
    from pvlib.location import Location
    
    # Skip if no valid data
    if df[["ghi", "dni", "dhi"]].isna().all().all():
        return pd.Series(0.0, index=df.index, name="predicted_power_kw")
    
    # Extract config parameters
    lat = config["location"]["lat"]
    lon = config["location"]["lon"]
    altitude = config["location"]["altitude_m"]
    orientations = config["orientations"]
    module_area = config["module"]["area_m2"]
    module_eff = config["module"]["efficiency_stc"]
    gamma_p = config["module"]["gamma_p"]
    inv_ac_rating = config["inverter"]["ac_rating_kw"]
    albedo = config["losses"]["albedo"]
    far_shading = config["losses"].get("far_shading", 1.0)
    
    # IAM table
    iam_angles = np.array(config["iam"]["angles"])
    iam_values = np.array(config["iam"]["values"])
    
    # DC losses
    dc_loss_factor = (
        (1 - config["losses"]["soiling"]) *
        (1 - config["losses"]["lid"]) *
        (1 - config["losses"]["module_quality"]) *
        (1 - config["losses"]["mismatch"]) *
        (1 - config["losses"]["dc_wiring"])
    )
    ac_wiring_loss = config["losses"]["ac_wiring"]
    
    # Thermal model
    sapm_params = pvlib.temperature.TEMPERATURE_MODEL_PARAMETERS["sapm"][config["thermal_model"]]
    
    # Total module area
    total_module_area = sum(o["module_count"] * module_area for o in orientations)
    
    # Create site location
    site = Location(lat, lon, altitude=altitude)
    
    # Fill NaN weather with defaults
    df = df.copy()
    df["air_temp"] = df["air_temp"].fillna(config["defaults"]["air_temp_c"])
    df["wind_speed"] = df["wind_speed"].fillna(config["defaults"]["wind_speed_ms"])
    df["ghi"] = df["ghi"].fillna(0)
    df["dni"] = df["dni"].fillna(0)
    df["dhi"] = df["dhi"].fillna(0)
    
    # Get solar position
    solpos = site.get_solarposition(df.index)
    dni_extra = pvlib.irradiance.get_extra_radiation(df.index)
    
    plant_ac = pd.Series(0.0, index=df.index)
    
    for o in orientations:
        tilt = o["tilt"]
        azimuth = o["azimuth"]
        area_fraction = (o["module_count"] * module_area) / total_module_area
        
        # Perez POA transposition
        irr = pvlib.irradiance.get_total_irradiance(
            surface_tilt=tilt,
            surface_azimuth=azimuth,
            solar_zenith=solpos["zenith"],
            solar_azimuth=solpos["azimuth"],
            dni=df["dni"],
            ghi=df["ghi"],
            dhi=df["dhi"],
            dni_extra=dni_extra,
            model="perez",
            albedo=albedo
        )
        
        poa = irr["poa_global"].clip(lower=0)
        
        # Apply far-shading
        if far_shading < 1.0:
            poa = poa * far_shading
        
        # AOI + IAM
        aoi = pvlib.irradiance.aoi(tilt, azimuth, solpos["zenith"], solpos["azimuth"])
        iam = np.interp(aoi, iam_angles, iam_values)
        poa_optical = poa * iam
        
        # Cell temperature (SAPM)
        cell_temp = pvlib.temperature.sapm_cell(
            poa, df["air_temp"], df["wind_speed"], **sapm_params
        )
        
        # DC power
        pdc_kwm2 = poa_optical * module_eff / 1000
        pdc_kwm2_temp = pdc_kwm2 * (1 + gamma_p * (cell_temp - 25))
        pdc_kwm2_eff = pdc_kwm2_temp * dc_loss_factor
        
        # Scale to orientation area
        area_i = total_module_area * area_fraction
        pdc_total_kw = pdc_kwm2_eff * area_i
        
        # Inverter efficiency (simplified flat)
        pac_kw = pdc_total_kw * config["inverter"]["flat_efficiency"]
        
        plant_ac += pac_kw
    
    # Plant-level clipping
    plant_ac = plant_ac.clip(upper=inv_ac_rating)
    
    # AC wiring losses
    plant_ac = plant_ac * (1 - ac_wiring_loss)
    
    return plant_ac.rename("predicted_power_kw")


# =====================================================================
# FILE APPEND LOGIC
# =====================================================================

def aggregate_hourly_to_daily(hourly_df):
    """
    Aggregate hourly predictions to daily totals.
    
    Args:
        hourly_df: DataFrame with hourly predictions (index=timestamp, columns include predicted_power_kw)
    
    Returns:
        DataFrame with daily aggregated data
    """
    if hourly_df.empty:
        return pd.DataFrame(columns=["date", "predicted_energy_kwh", "hours_count", 
                                      "avg_ghi", "avg_dni", "avg_dhi", "avg_temp", "avg_wind"])
    
    # Group by date
    daily = hourly_df.copy()
    daily["date"] = daily.index.date
    
    # Aggregate
    agg_dict = {
        "predicted_power_kw": "sum",  # Sum of hourly kW = kWh for 1-hour intervals
    }
    
    # Add optional columns if present
    for col in ["ghi", "dni", "dhi"]:
        if col in daily.columns:
            agg_dict[col] = "mean"
    if "air_temp" in daily.columns:
        agg_dict["air_temp"] = "mean"
    if "wind_speed" in daily.columns:
        agg_dict["wind_speed"] = "mean"
    
    result = daily.groupby("date").agg(agg_dict).reset_index()
    
    # Count hours per day
    hours_count = daily.groupby("date").size()
    result["hours_count"] = result["date"].map(hours_count)
    
    # Rename columns
    result = result.rename(columns={
        "predicted_power_kw": "predicted_energy_kwh",
        "ghi": "avg_ghi",
        "dni": "avg_dni", 
        "dhi": "avg_dhi",
        "air_temp": "avg_temp",
        "wind_speed": "avg_wind"
    })
    
    # Set date as index
    result = result.set_index("date")
    
    return result


def aggregate_daily_to_monthly(daily_df):
    """
    Aggregate daily predictions to monthly totals.
    
    Args:
        daily_df: DataFrame with daily predictions (index=date, columns include predicted_energy_kwh)
    
    Returns:
        DataFrame with monthly aggregated data
    """
    if daily_df.empty:
        return pd.DataFrame(columns=["month", "predicted_energy_kwh", "days_count",
                                      "avg_ghi", "avg_dni", "avg_dhi", "avg_temp", "avg_wind"])
    
    # Convert index to datetime if needed
    monthly = daily_df.copy()
    if not isinstance(monthly.index, pd.DatetimeIndex):
        monthly.index = pd.to_datetime(monthly.index)
    
    # Group by month (YYYY-MM format)
    monthly["month"] = monthly.index.to_period("M").astype(str)
    
    # Aggregate
    agg_dict = {
        "predicted_energy_kwh": "sum",
    }
    
    # Add optional columns if present
    for col in ["avg_ghi", "avg_dni", "avg_dhi"]:
        if col in monthly.columns:
            agg_dict[col] = "mean"
    if "avg_temp" in monthly.columns:
        agg_dict["avg_temp"] = "mean"
    if "avg_wind" in monthly.columns:
        agg_dict["avg_wind"] = "mean"
    
    result = monthly.groupby("month").agg(agg_dict).reset_index()
    
    # Count days per month
    days_count = monthly.groupby("month").size()
    result["days_count"] = result["month"].map(days_count)
    
    # Set month as index
    result = result.set_index("month")
    
    return result


def append_to_file(output_path, new_df, index_col="timestamp"):
    """
    Append new data to a file, avoiding duplicates.
    
    Args:
        output_path: Path to output CSV
        new_df: DataFrame with new data
        index_col: Name of the index column in file
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    if output_path.exists():
        # Load existing data
        existing = pd.read_csv(output_path, parse_dates=[index_col] if index_col == "timestamp" else None)
        existing = existing.set_index(existing.columns[0])
        
        # Combine, keeping newest for duplicates
        combined = pd.concat([existing, new_df])
        combined = combined[~combined.index.duplicated(keep="last")]
    else:
        combined = new_df
    
    # Sort by index and save
    combined = combined.sort_index()
    combined.to_csv(output_path)
    
    return len(combined)


def append_predictions(source_name, new_predictions_df):
    """
    Append new predictions to hourly, daily, and monthly files.
    
    Args:
        source_name: Data source name (e.g., "nasa", "solcast")
        new_predictions_df: DataFrame with hourly predictions
    """
    filename = OUTPUT_FILES[source_name]
    
    # 1. Save hourly data
    hourly_path = HOURLY_DIR / filename
    hourly_count = append_to_file(hourly_path, new_predictions_df, "timestamp")
    logger.info(f"✓ Hourly: {len(new_predictions_df)} new → {hourly_count} total in {hourly_path.name}")
    
    # 2. Aggregate and save daily data
    daily_df = aggregate_hourly_to_daily(new_predictions_df)
    daily_path = DAILY_DIR / filename
    daily_count = append_to_file(daily_path, daily_df, "date")
    logger.info(f"✓ Daily: {len(daily_df)} days → {daily_count} total in {daily_path.name}")
    
    # 3. Aggregate and save monthly data
    monthly_df = aggregate_daily_to_monthly(daily_df)
    monthly_path = MONTHLY_DIR / filename
    monthly_count = append_to_file(monthly_path, monthly_df, "month")
    logger.info(f"✓ Monthly: {len(monthly_df)} months → {monthly_count} total in {monthly_path.name}")


# =====================================================================
# MAIN EXECUTION
# =====================================================================

def main():
    """Main entry point with CLI argument parsing."""
    parser = argparse.ArgumentParser(
        description="Daily power predictor with cascade fallbacks",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python daily_predictor.py                          # Today + tomorrow
  python daily_predictor.py --today-only             # Today only
  python daily_predictor.py --tomorrow-only          # Tomorrow only
  python daily_predictor.py --start 20260107 --end 20260108  # Custom range
        """
    )
    
    parser.add_argument(
        "--today-only",
        action="store_true",
        help="Predict for today only"
    )
    parser.add_argument(
        "--tomorrow-only",
        action="store_true",
        help="Predict for tomorrow only"
    )
    parser.add_argument(
        "--start",
        help="Start date in YYYYMMDD format"
    )
    parser.add_argument(
        "--end",
        help="End date in YYYYMMDD format"
    )
    
    args = parser.parse_args()
    
    # Create output directories
    HOURLY_DIR.mkdir(parents=True, exist_ok=True)
    DAILY_DIR.mkdir(parents=True, exist_ok=True)
    MONTHLY_DIR.mkdir(parents=True, exist_ok=True)
    
    # Load configuration
    config = load_config()
    lat = config["location"]["lat"]
    lon = config["location"]["lon"]
    timezone_str = config["location"]["timezone"]
    api_key = config["api"]["solcast_key"]
    
    # Determine date range
    now = datetime.now(timezone.utc)
    today = now.replace(hour=0, minute=0, second=0, microsecond=0)
    tomorrow = today + timedelta(days=1)
    
    if args.start and args.end:
        start_date = datetime.strptime(args.start, "%Y%m%d").replace(tzinfo=timezone.utc)
        end_date = datetime.strptime(args.end, "%Y%m%d").replace(tzinfo=timezone.utc)
        end_date = end_date.replace(hour=23)  # End of day
    elif args.today_only:
        start_date = today
        end_date = today.replace(hour=23)
    elif args.tomorrow_only:
        start_date = tomorrow
        end_date = tomorrow.replace(hour=23)
    else:
        # Default: today + tomorrow
        start_date = today
        end_date = tomorrow.replace(hour=23)
    
    logger.info("=" * 60)
    logger.info("DAILY POWER PREDICTOR")
    logger.info("=" * 60)
    logger.info(f"Date range: {start_date.date()} to {end_date.date()}")
    logger.info(f"Location: lat={lat}, lon={lon}")
    logger.info("=" * 60)
    
    # =====================================================================
    # FETCH DATA FROM ALL SOURCES
    # =====================================================================
    
    all_data = {}
    
    # NASA POWER (free, no API key)
    nasa_data = fetch_nasa_power_data(lat, lon, start_date.date(), end_date.date())
    if nasa_data is not None:
        all_data["nasa"] = nasa_data
    
    # Solcast (requires API key)
    if api_key and api_key != "YOUR_SOLCAST_API_KEY":
        # For today, use estimated_actuals; for tomorrow, use forecast
        if start_date.date() <= now.date():
            solcast_data = fetch_solcast_data(lat, lon, api_key, start_date, end_date, "estimated_actuals")
        else:
            solcast_data = fetch_solcast_data(lat, lon, api_key, start_date, end_date, "forecast")
        
        if solcast_data is not None:
            all_data["solcast"] = solcast_data
    else:
        logger.warning("⚠️ Solcast API key not configured, skipping Solcast")
    
    # Open-Meteo (free, no API key - great for forecasts!)
    openmeteo_data = fetch_openmeteo_data(lat, lon, start_date.date(), end_date.date())
    if openmeteo_data is not None:
        all_data["openmeteo"] = openmeteo_data
    
    # ERA5 (requires CDS API credentials)
    era5_data = fetch_era5_data(lat, lon, start_date.date(), end_date.date())
    if era5_data is not None:
        all_data["era5"] = era5_data
    
    # Check if we have any data
    if not all_data:
        logger.error("❌ No data sources available! Cannot proceed.")
        sys.exit(1)
    
    logger.info(f"\nAvailable data sources: {list(all_data.keys())}")
    
    # =====================================================================
    # GENERATE TIMESTAMPS FOR PREDICTION PERIOD
    # =====================================================================
    
    timestamps = pd.date_range(start=start_date, end=end_date, freq="h", tz="UTC")
    logger.info(f"Prediction timestamps: {len(timestamps)} hours")
    
    # =====================================================================
    # PROCESS EACH PRIMARY SOURCE
    # =====================================================================
    
    for primary_source in ["nasa", "solcast", "openmeteo", "era5"]:
        logger.info(f"\n{'='*60}")
        logger.info(f"Processing: {primary_source.upper()} as primary source")
        logger.info(f"Fallback order: {FALLBACK_ORDER[primary_source]}")
        logger.info(f"{'='*60}")
        
        # Merge data with fallbacks
        merged_df = merge_data_with_fallback(timestamps, primary_source, all_data)
        
        # Check if we have valid irradiance data
        valid_irradiance = merged_df[["ghi", "dni", "dhi"]].notna().any(axis=1).sum()
        
        if valid_irradiance == 0:
            logger.warning(f"⚠️ No valid irradiance data for {primary_source}, skipping")
            continue
        
        logger.info(f"Valid irradiance hours: {valid_irradiance}/{len(merged_df)}")
        
        # Compute power predictions
        power = compute_pv_power(merged_df, config)
        
        # Ensure power is always numeric (fill NaN with 0)
        power = power.fillna(0).clip(lower=0)
        
        # Build output DataFrame
        output_df = merged_df.copy()
        output_df["predicted_power_kw"] = power
        
        # Convert to local timezone for output
        output_df.index = output_df.index.tz_convert(timezone_str)
        
        # Reorder columns
        output_df = output_df[[
            "predicted_power_kw", "ghi", "dni", "dhi", "air_temp", "wind_speed",
            "irradiance_source", "temp_source", "wind_source"
        ]]
        
        # Append to hourly, daily, and monthly files
        append_predictions(primary_source, output_df)
        
        # Log summary
        total_energy = (power.clip(lower=0) * 1).sum()  # kWh (1-hour intervals)
        logger.info(f"Total predicted energy: {total_energy:.1f} kWh")
    
    # =====================================================================
    # SUMMARY
    # =====================================================================
    
    logger.info("\n" + "=" * 60)
    logger.info("DAILY PREDICTOR COMPLETE")
    logger.info("=" * 60)
    
    for granularity, folder in [("Hourly", HOURLY_DIR), ("Daily", DAILY_DIR), ("Monthly", MONTHLY_DIR)]:
        logger.info(f"\n{granularity} output files:")
        for source, filename in OUTPUT_FILES.items():
            path = folder / filename
            if path.exists():
                df = pd.read_csv(path)
                logger.info(f"  {filename}: {len(df)} records")
    
    logger.info("\n" + "=" * 60)


if __name__ == "__main__":
    main()

