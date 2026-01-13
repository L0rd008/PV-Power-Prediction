"""
DNI Prediction from GHI, DHI using Solar Position
==================================================

Predicts Direct Normal Irradiance (DNI) using:
- GHI (Global Horizontal Irradiance)
- DHI (Diffuse Horizontal Irradiance)  
- Solar zenith angle (calculated from location + time)

Physical relationship: GHI = DHI + DNI × cos(θz)
Rearranged: DNI = (GHI - DHI) / cos(θz)

Output: JSON with predictions and accuracy metrics
"""

import json
import math
import csv
from datetime import datetime
from pathlib import Path

# Configuration from plant_config.json
CONFIG = {
    "lat": 8.342368984714714,
    "lon": 80.37623529556957,
    "data_file": "../data/solcast_irradiance.csv",
    "output_file": "../data/dni_prediction_results.json"
}


def julian_day(year, month, day, hour):
    """Calculate Julian Day number"""
    if month <= 2:
        year -= 1
        month += 12
    A = int(year / 100)
    B = 2 - A + int(A / 4)
    JD = int(365.25 * (year + 4716)) + int(30.6001 * (month + 1)) + day + hour/24 + B - 1524.5
    return JD


def solar_position(lat, lon, dt):
    """
    Calculate solar zenith and azimuth angles
    Returns: (zenith_deg, azimuth_deg)
    """
    lat_rad = math.radians(lat)
    
    # Julian day
    JD = julian_day(dt.year, dt.month, dt.day, dt.hour + dt.minute/60 + dt.second/3600)
    
    # Julian century
    JC = (JD - 2451545.0) / 36525.0
    
    # Sun's mean longitude (degrees)
    L0 = (280.46646 + JC * (36000.76983 + 0.0003032 * JC)) % 360
    
    # Sun's mean anomaly (degrees)
    M = (357.52911 + JC * (35999.05029 - 0.0001537 * JC)) % 360
    M_rad = math.radians(M)
    
    # Eccentricity of Earth's orbit
    e = 0.016708634 - JC * (0.000042037 + 0.0000001267 * JC)
    
    # Sun's equation of center
    C = (1.914602 - JC * (0.004817 + 0.000014 * JC)) * math.sin(M_rad) \
        + (0.019993 - 0.000101 * JC) * math.sin(2 * M_rad) \
        + 0.000289 * math.sin(3 * M_rad)
    
    # Sun's true longitude
    sun_lon = L0 + C
    
    # Sun's apparent longitude
    omega = 125.04 - 1934.136 * JC
    apparent_lon = sun_lon - 0.00569 - 0.00478 * math.sin(math.radians(omega))
    
    # Mean obliquity of the ecliptic
    obliquity = 23.439291 - 0.0130042 * JC - 0.00000016 * JC**2 + 0.000000504 * JC**3
    obliquity_corr = obliquity + 0.00256 * math.cos(math.radians(omega))
    obliquity_rad = math.radians(obliquity_corr)
    
    # Sun's declination
    declination = math.degrees(math.asin(math.sin(obliquity_rad) * math.sin(math.radians(apparent_lon))))
    decl_rad = math.radians(declination)
    
    # Equation of time (minutes)
    y = math.tan(obliquity_rad / 2)**2
    EoT = 4 * math.degrees(
        y * math.sin(2 * math.radians(L0))
        - 2 * e * math.sin(M_rad)
        + 4 * e * y * math.sin(M_rad) * math.cos(2 * math.radians(L0))
        - 0.5 * y**2 * math.sin(4 * math.radians(L0))
        - 1.25 * e**2 * math.sin(2 * M_rad)
    )
    
    # Solar time
    time_offset = EoT + 4 * lon  # minutes
    solar_time = dt.hour * 60 + dt.minute + dt.second/60 + time_offset
    
    # Hour angle
    hour_angle = (solar_time / 4) - 180  # degrees
    hour_angle_rad = math.radians(hour_angle)
    
    # Solar zenith angle
    cos_zenith = (math.sin(lat_rad) * math.sin(decl_rad) +
                  math.cos(lat_rad) * math.cos(decl_rad) * math.cos(hour_angle_rad))
    cos_zenith = max(-1, min(1, cos_zenith))  # Clamp for numerical stability
    zenith = math.degrees(math.acos(cos_zenith))
    
    # Solar azimuth angle
    if zenith > 0.1:
        sin_azimuth = -math.cos(decl_rad) * math.sin(hour_angle_rad) / math.sin(math.radians(zenith))
    else:
        sin_azimuth = 0
    sin_azimuth = max(-1, min(1, sin_azimuth))
    azimuth = math.degrees(math.asin(sin_azimuth))
    
    return zenith, azimuth


def predict_dni(ghi, dhi, cos_zenith, min_cos_zenith=0.05):
    """
    Predict DNI using physics model: DNI = (GHI - DHI) / cos(θz)
    
    Args:
        ghi: Global Horizontal Irradiance
        dhi: Diffuse Horizontal Irradiance
        cos_zenith: Cosine of solar zenith angle
        min_cos_zenith: Minimum cos(zenith) to avoid division issues
    
    Returns:
        Predicted DNI (clamped to >= 0)
    """
    if cos_zenith < min_cos_zenith:
        return 0.0
    
    dni_pred = (ghi - dhi) / cos_zenith
    return max(0.0, dni_pred)


def calculate_metrics(actual, predicted):
    """Calculate accuracy metrics"""
    n = len(actual)
    if n == 0:
        return {}
    
    # Filter pairs where both have values (daytime only for meaningful comparison)
    pairs = [(a, p) for a, p in zip(actual, predicted) if a > 0 or p > 0]
    
    if len(pairs) == 0:
        return {"note": "No non-zero values to compare"}
    
    actual_f = [p[0] for p in pairs]
    pred_f = [p[1] for p in pairs]
    n_f = len(actual_f)
    
    # Mean Absolute Error
    mae = sum(abs(a - p) for a, p in pairs) / n_f
    
    # Root Mean Square Error
    rmse = math.sqrt(sum((a - p)**2 for a, p in pairs) / n_f)
    
    # Mean Bias Error
    mbe = sum(p - a for a, p in pairs) / n_f
    
    # R-squared (coefficient of determination)
    mean_actual = sum(actual_f) / n_f
    ss_tot = sum((a - mean_actual)**2 for a in actual_f)
    ss_res = sum((a - p)**2 for a, p in pairs)
    r2 = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
    
    # Mean Absolute Percentage Error (for non-zero actuals)
    mape_pairs = [(a, p) for a, p in pairs if a > 0]
    if mape_pairs:
        mape = sum(abs(a - p) / a for a, p in mape_pairs) / len(mape_pairs) * 100
    else:
        mape = None
    
    return {
        "total_samples": n,
        "evaluated_samples": n_f,
        "mae_wm2": round(mae, 2),
        "rmse_wm2": round(rmse, 2),
        "mbe_wm2": round(mbe, 2),
        "r_squared": round(r2, 4),
        "mape_percent": round(mape, 2) if mape else None
    }


def load_csv_data(filepath):
    """Load data from CSV file"""
    data = []
    with open(filepath, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            data.append({
                "air_temp": float(row["air_temp"]),
                "dhi": float(row["dhi"]),
                "dni": float(row["dni"]),
                "ghi": float(row["ghi"]),
                "wind_speed_10m": float(row["wind_speed_10m"]),
                "period_end": row["period_end"]
            })
    return data


def main():
    script_dir = Path(__file__).parent
    data_path = script_dir / CONFIG["data_file"]
    output_path = script_dir / CONFIG["output_file"]
    
    print(f"Loading data from: {data_path}")
    data = load_csv_data(data_path)
    
    results = []
    actual_dni = []
    predicted_dni = []
    
    for row in data:
        # Parse timestamp (remove timezone for simplicity, already UTC)
        ts_str = row["period_end"].replace("+00:00", "").replace("Z", "")
        dt = datetime.fromisoformat(ts_str)
        
        # Calculate solar position
        zenith, azimuth = solar_position(CONFIG["lat"], CONFIG["lon"], dt)
        cos_zenith = math.cos(math.radians(zenith))
        
        # Predict DNI
        dni_pred = predict_dni(row["ghi"], row["dhi"], cos_zenith)
        
        actual_dni.append(row["dni"])
        predicted_dni.append(dni_pred)
        
        results.append({
            "timestamp": row["period_end"],
            "ghi": row["ghi"],
            "dhi": row["dhi"],
            "dni_actual": row["dni"],
            "dni_predicted": round(dni_pred, 2),
            "solar_zenith_deg": round(zenith, 2),
            "cos_zenith": round(cos_zenith, 4),
            "error": round(dni_pred - row["dni"], 2)
        })
    
    # Calculate metrics
    metrics = calculate_metrics(actual_dni, predicted_dni)
    
    output = {
        "model": "Physics-based DNI prediction",
        "formula": "DNI = (GHI - DHI) / cos(solar_zenith)",
        "location": {
            "lat": CONFIG["lat"],
            "lon": CONFIG["lon"]
        },
        "metrics": metrics,
        "predictions": results
    }
    
    # Save results
    with open(output_path, 'w') as f:
        json.dump(output, f, indent=2)
    
    print(f"\nResults saved to: {output_path}")
    print(f"\n=== ACCURACY METRICS ===")
    print(json.dumps(metrics, indent=2))
    
    # Print sample predictions
    print(f"\n=== SAMPLE PREDICTIONS (daytime hours with DNI > 0) ===")
    daytime = [r for r in results if r["dni_actual"] > 0 or r["dni_predicted"] > 10]
    for r in daytime[:10]:
        print(f"  {r['timestamp'][:16]} | Actual: {r['dni_actual']:6.1f} | Pred: {r['dni_predicted']:6.1f} | Error: {r['error']:+6.1f}")


if __name__ == "__main__":
    main()





