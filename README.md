# Power Prediction

Solar PV power generation prediction system using physics-based and surrogate modeling approaches.

## Overview

This project provides tools to predict AC power output from solar PV installations using:
1. **Physics-based models** - Full PVsyst-consistent calculations using pvlib
2. **Surrogate models** - Fast regression-based approximations trained on physics model outputs

---

## Folder Structure

```
Power Prediction/
├── scripts/
│   ├── python_physics/     # Python physics-based model
│   │   ├── physics_model.py    # Main physics model (pvlib)
│   │   ├── daily_predictor.py  # Daily predictions with cascade fallbacks
│   │   ├── fetch_nasa_power.py # NASA POWER API fetcher
│   │   └── fetch_era5_weather.py
│   ├── js_physics/         # JavaScript physics model
│   │   └── physics_model.js    # ThingsBoard physics version
│   ├── js_surrogate/       # JavaScript surrogate model
│   │   ├── surrogate_model.js  # ThingsBoard regression model
│   │   └── fit_surrogate.py    # Training script
│   └── shared/             # Shared utilities
│       ├── compare_accuracy.py
│       ├── prepare_training_data.py
│       ├── evaluate_models.py
│       ├── timezone_utils.py
│       ├── validate_model.py
│       ├── run_benchmark.py
│       ├── benchmark_js.js
│       └── plot_inverter_curve.py
├── config/            # Plant configuration files
├── data/              # Input data files
├── output/            # Generated outputs
│   ├── hourly/        # Hourly predictions per source
│   ├── daily/         # Daily aggregated predictions
│   └── monthly/       # Monthly aggregated predictions
└── docs/              # Documentation
```

---

## Scripts

### Python Physics Model (`scripts/python_physics/`)

| File | Description |
|------|-------------|
| `physics_model.py` | **Unified physics model** - Full physics-based PV prediction using pvlib, Perez POA transposition, SAPM thermal model, inverter efficiency curves, far-shading support. Supports Solcast, NASA POWER, and CSV data sources via CLI. |
| `daily_predictor.py` | **Daily predictor** - Automated daily predictions with cascade fallbacks across 4 data sources (NASA POWER, Solcast, Open-Meteo, ERA5). Outputs to hourly/daily/monthly directories. |
| `fetch_nasa_power.py` | **NASA POWER API** - Fetches free irradiance data (no API key required). Supports hourly (W/m²) and daily (kWh/m²/day) modes. |
| `fetch_era5_weather.py` | Downloads ERA5 reanalysis weather data (temperature, wind speed) from Copernicus CDS API. |

**CLI Usage:**
```bash
cd scripts/python_physics

# Read from local CSV (default)
python physics_model.py --source csv

# Fetch from Solcast API (forecast)
python physics_model.py --source api

# Fetch from Solcast API (estimated_actuals - last 7 days)
python physics_model.py --source api --endpoint estimated_actuals

# Fetch from NASA POWER hourly (free, no API key needed)
python physics_model.py --source nasa_power --start 20251201 --end 20251215

# Fetch from NASA POWER daily (simplified model, quick estimates)
python physics_model.py --source nasa_power_daily --start 20251201 --end 20251215

# Custom CSV path
python physics_model.py --source csv --csv-path ../../data/custom.csv

# Show all options
python physics_model.py --help
```

**Daily Predictor CLI Usage:**
```bash
cd scripts/python_physics

# Predict for today + tomorrow (default)
python daily_predictor.py

# Today only
python daily_predictor.py --today-only

# Tomorrow only  
python daily_predictor.py --tomorrow-only

# Custom date range
python daily_predictor.py --start 20260107 --end 20260108

# Show all options
python daily_predictor.py --help
```

### JavaScript Physics Model (`scripts/js_physics/`)

| File | Description |
|------|-------------|
| `physics_model.js` | **Physics-based** - Simplified Perez model with solar position calculations. Optimized for ThingsBoard rule chains. Good for real-time SCADA monitoring. |

### JavaScript Surrogate Model (`scripts/js_surrogate/`)

| File | Description |
|------|-------------|
| `surrogate_model.js` | **Regression-based** - Fast approximation using fitted coefficients. Requires training first. Best for high-frequency predictions. |
| `fit_surrogate.py` | Trains regression coefficients for the surrogate model using sklearn. Outputs JavaScript-ready coefficients. |

### Shared Utilities (`scripts/shared/`)

| File | Description |
|------|-------------|
| `compare_accuracy.py` | **Accuracy comparison** - Compares predictions from all data sources against Meteocontrol actuals. Generates metrics (MAE, RMSE, MAPE, R²), charts, and reports. |
| `prepare_training_data.py` | **Training pipeline** - Fetches NASA POWER data, runs physics model, creates train/test split. |
| `evaluate_models.py` | **Model evaluation** - Evaluates all models on test data only (no data leakage). |
| `timezone_utils.py` | **Timezone handling** - Consistent UTC/local conversion. Loads Meteocontrol data with proper timestamps. |
| `validate_model.py` | Compares model predictions against actual Meteocontrol generation data. Calculates MAE, MAPE, and bias. |
| `run_benchmark.py` | Benchmarks all three models (Python physics, JS physics, JS surrogate) against actual data. |
| `benchmark_js.js` | Node.js runner for JavaScript model benchmarking. |
| `plot_inverter_curve.py` | Visualizes and validates the inverter efficiency curve from datasheet digitization. |

---

## Data Sources

The system supports multiple irradiance data sources:

| Source | API Key | Resolution | Units | Best For |
|--------|---------|------------|-------|----------|
| **Solcast** | Required | Hourly | W/m² | Production forecasts |
| **NASA POWER** | Not required | Hourly/Daily | W/m² or kWh/m²/day | Free historical data |
| **Open-Meteo** | Not required | Hourly | W/m² | Free forecasts (16 days ahead) |
| **ERA5** | Required (CDS) | Hourly | W/m² | Weather validation |

### NASA POWER (Recommended for free access)

NASA POWER provides satellite-derived irradiance data from 1981 to present (7-day lag).

```bash
# Fetch hourly data (W/m², compatible with physics model)
python fetch_nasa_power.py --start 20251201 --end 20251215 --mode hourly

# Fetch daily data (kWh/m²/day, for quick estimates)
python fetch_nasa_power.py --start 20251201 --end 20251215 --mode daily
```

---

## Data Files

| File | Description |
|------|-------------|
| `data/solcast_irradiance.csv` | Hourly irradiance data (GHI, DNI, DHI, temperature, wind) from Solcast API |
| `data/nasa_power_hourly.csv` | Hourly irradiance from NASA POWER (generated by fetch_nasa_power.py) |
| `data/nasa_power_daily.csv` | Daily irradiance from NASA POWER (generated by fetch_nasa_power.py) |
| `data/meteocontrol_actual.csv` | Actual daily energy generation from Meteocontrol monitoring system |
| `data/era5_weather.csv` | ERA5 reanalysis weather data (air temperature, wind speed) |
| `data/train_data.csv` | Training data (80% of days, generated by prepare_training_data.py) |
| `data/test_data.csv` | Test data (20% of days, for final evaluation) |
| `data/physics_predictions_all.csv` | Complete physics model predictions for all data |

---

## Configuration

| File | Description |
|------|-------------|
| `config/plant_config.json` | Plant parameters (location, modules, inverter, losses, far-shading) |

### Key Configuration Parameters

```json
{
  "losses": {
    "far_shading": 1.0  // 1.0 = no shading, <1.0 = shading loss applied
  }
}
```

---

## Output Files

Generated files are saved to `output/`:

### Prediction Files (by granularity)

The `daily_predictor.py` script generates predictions in three subdirectories:

| Directory | Contents | Description |
|-----------|----------|-------------|
| `output/hourly/` | `predictions_*.csv` | Hourly power predictions (kW) per data source |
| `output/daily/` | `predictions_*.csv` | Daily energy totals (kWh) per data source |
| `output/monthly/` | `predictions_*.csv` | Monthly energy totals (kWh) per data source |

Each directory contains four files:
- `predictions_nasa_power.csv` - NASA POWER as primary source
- `predictions_solcast.csv` - Solcast as primary source
- `predictions_openmeteo.csv` - Open-Meteo as primary source
- `predictions_era5.csv` - ERA5 as primary source

### Other Output Files

| File | Generated By | Description |
|------|--------------|-------------|
| `pv_generation.csv` | `physics_model.py` | Hourly AC power predictions (kW) |
| `accuracy_comparison.png` | `compare_accuracy.py` | Bar chart comparing all sources |
| `daily_comparison.png` | `compare_accuracy.py` | Line chart: actual vs predicted |
| `accuracy_report.txt` | `compare_accuracy.py` | Detailed text report |
| `validation_results.csv` | `fit_surrogate.py` | Detailed comparison of actual vs predicted |
| `validation_plots.png` | `fit_surrogate.py` | Visual validation plots |
| `js_coefficients.txt` | `fit_surrogate.py` | Fitted coefficients for JavaScript surrogate |
| `surrogate_coefficients.json` | `fit_surrogate.py` | Trained coefficients with metadata |
| `split_info.json` | `prepare_training_data.py` | Train/test split information for reproducibility |
| `test_evaluation_metrics.json` | `evaluate_models.py` | Test set performance metrics |
| `test_evaluation.png` | `evaluate_models.py` | Test set comparison plots |
| `benchmark_results.csv` | `run_benchmark.py` | Daily energy comparison across models |
| `benchmark_comparison.png` | `run_benchmark.py` | Visual benchmark comparison |
| `benchmark_report.md` | `run_benchmark.py` | Human-readable benchmark summary |

---

## Quick Start

### 1. Generate predictions (from CSV)

```bash
cd scripts/python_physics
python physics_model.py --source csv
```

### 2. Generate predictions (from Solcast API)

```bash
cd scripts/python_physics
python physics_model.py --source api
```

### 3. Validate against actual data

```bash
cd scripts/shared
python validate_model.py
```

### 4. Train surrogate model (proper train/test split)

```bash
# Step 1: Prepare training data from NASA POWER
cd scripts/shared
python prepare_training_data.py --year 2024 --test-ratio 0.2 --seed 42

# Step 2: Train surrogate on training data only
cd ../js_surrogate
python fit_surrogate.py --train-data ../../data/train_data.csv

# Step 3: Evaluate all models on test data (no leakage)
cd ../shared
python evaluate_models.py --test-data ../../data/test_data.csv
```

### 5. Run model benchmark

```bash
cd scripts/shared

# Using local Solcast CSV data (default)
python run_benchmark.py

# Using NASA POWER data
python run_benchmark.py --data-source nasa_power --start 20251210 --end 20251215
```

### 6. Run daily predictions (with cascade fallbacks)

```bash
cd scripts/python_physics

# Predict for today + tomorrow (default)
python daily_predictor.py

# Today only
python daily_predictor.py --today-only

# Tomorrow only
python daily_predictor.py --tomorrow-only

# Custom date range
python daily_predictor.py --start 20260107 --end 20260108
```

### 7. Compare accuracy against actual data

```bash
cd scripts/shared

# Default comparison (all granularities)
python compare_accuracy.py

# Filter by day range
python compare_accuracy.py --days 1-15

# Output as JSON
python compare_accuracy.py --json

# Skip graphical output
python compare_accuracy.py --no-plot

# Specific granularity only
python compare_accuracy.py --granularity daily
```

### 8. Deploy to ThingsBoard

Copy the contents of `scripts/js_surrogate/surrogate_model.js` or `scripts/js_physics/physics_model.js` to a ThingsBoard Script node in your rule chain.

---

## Training Pipeline (No Data Leakage)

The training pipeline ensures proper train/test separation to avoid overfitting:

```
NASA POWER API (1 year)
        │
        ▼
┌─────────────────────────┐
│ Python Physics Model    │  (generate ground truth)
└─────────────────────────┘
        │
        ▼
┌─────────────────────────┐
│ Random Split by DAY     │  (80% train / 20% test)
└─────────────────────────┘
   ┌────┴────┐
   ▼         ▼
┌──────┐  ┌──────┐
│Train │  │Test  │
│ 80%  │  │ 20%  │
└──────┘  └──────┘
   │         │
   ▼         ▼
Surrogate   Evaluate ALL
Training    models here
```

### Key Points:
- **Days are split**, not individual hours (prevents temporal leakage)
- **Surrogate sees training data only** during fitting
- **Test metrics are unbiased** - reported on completely unseen data
- **Random seed** ensures reproducibility

### Typical Results (with proper split):

| Model | Test R² | Test MAE | Notes |
|-------|---------|----------|-------|
| Python Physics | Baseline | Baseline | Ground truth |
| JS Physics | ~0.90 | ~2-3 kW | Good physics approximation |
| JS Surrogate | ~0.25-0.40 | ~6-10 kW | Limited by training data |

The surrogate model shows lower test performance than training performance - this is **expected** and correct. High training R² (0.95+) with low test R² indicates overfitting.

---

## Plant Configuration

The models are configured for a **55kW rooftop solar installation** in Sri Lanka:

- **Location**: 8.342°N, 80.376°E
- **Modules**: 216 × 550W (Canadian Solar CS6W-550MS)
- **Inverter**: 55kW AC rating
- **Orientations**: 8 different tilt/azimuth combinations
- **Module efficiency**: 21.53% at STC

See `docs/config_notes.md` for detailed PVsyst parameters.

---

## Model Comparison

| Aspect | Python Physics | JS Physics | JS Surrogate |
|--------|----------------|------------|--------------|
| **Approach** | Full pvlib + Perez | Simplified Perez | Regression |
| **Speed** | ~0.5ms/prediction | ~0.02ms/prediction | ~0.005ms/prediction |
| **Accuracy** | Reference model | ±3-5% vs Python | ±2-3% vs Python |
| **Flexibility** | Full physics | ThingsBoard optimized | Training-bound |
| **Use case** | Engineering validation | Real-time monitoring | High-frequency |

---

## Requirements

### Python
```
pandas
numpy
pvlib
requests
matplotlib
scikit-learn
cdsapi (for ERA5)
xarray (for ERA5)
```

### JavaScript
- ThingsBoard (for deployment)
- Node.js (optional, for local testing/benchmarking)

---

## Timezone Handling

All scripts use consistent timezone handling via `timezone_utils.py`:

| Data Source | Native Timezone | Notes |
|-------------|-----------------|-------|
| Solcast API | UTC | Explicit in timestamps |
| NASA POWER | UTC | Converted from YYYYMMDDHH format |
| Open-Meteo | UTC | Explicit in timestamps |
| ERA5 weather | UTC | Explicit in timestamps |
| Meteocontrol actual | Asia/Colombo (+05:30) | Day numbers only in raw CSV |
| **Model outputs** | **Asia/Colombo (+05:30)** | All final outputs in local time |

**Key functions:**
```python
from timezone_utils import load_meteocontrol, to_utc, to_local

# Load Meteocontrol with proper timestamps
df = load_meteocontrol("../../data/meteocontrol_actual.csv")

# Convert to UTC for internal processing
df_utc = to_utc(df)

# Convert to local for output
df_local = to_local(df)
```

---

## Notes

- All Python scripts support CLI arguments - use `--help` for options
- Internal processing uses UTC; final outputs are in Asia/Colombo (+05:30)
- The surrogate model requires training before use - see `fit_surrogate.py`
- Far-shading is controlled via `plant_config.json` and applied when value < 1.0
- NASA POWER data is free but has ~7-day lag; Solcast requires paid API key for weather
- NASA POWER daily mode uses a simplified model (less accurate but no API needed)
