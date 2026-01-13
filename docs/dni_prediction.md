# DNI Prediction Script

## Purpose
Predicts Direct Normal Irradiance (DNI) using physics-based solar radiation decomposition and evaluates accuracy against actual measurements.

## Input
- `data/solcast_irradiance.csv` - Contains DHI, GHI, air_temp, wind_speed_10m, DNI (for validation)

## Output
- `data/dni_prediction_results.json` - Predictions and accuracy metrics

## Method
Uses the fundamental solar radiation equation:
```
GHI = DHI + DNI × cos(θz)
```
Rearranged to:
```
DNI = (GHI - DHI) / cos(θz)
```
Where θz is the solar zenith angle calculated from:
- Latitude/Longitude (from plant_config.json)
- Timestamp of each measurement

## Metrics
- **MAE**: Mean Absolute Error (W/m²)
- **RMSE**: Root Mean Square Error (W/m²)
- **MBE**: Mean Bias Error (W/m²)
- **R²**: Coefficient of determination
- **MAPE**: Mean Absolute Percentage Error (%)

## Usage
```bash
cd Power-Prediction/scripts
python predict_dni.py
```

## Notes
- Nighttime predictions (when cos(θz) < 0.05) are set to 0
- No external dependencies required (uses only Python standard library)





