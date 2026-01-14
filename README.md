# Power Prediction

Solar PV power generation prediction system using physics-based and surrogate modeling approaches.

---

## What This Project Does

This project predicts AC power output from solar PV installations using three different approaches:

| Model | File | Use Case | Accuracy | Speed |
|-------|------|----------|----------|-------|
| **Python Physics** | `scripts/python_physics/physics_model.py` | Engineering validation, ground truth | Highest (reference) | ~0.5ms/prediction |
| **JS Physics** | `scripts/js_physics/physics_model.js` | ThingsBoard real-time SCADA | ±3-5% vs Python | ~0.02ms/prediction |
| **JS Surrogate** | `scripts/js_surrogate/surrogate_model.js` | High-frequency monitoring | Training-dependent | ~0.005ms/prediction |

The **Python Physics model** is the source of truth, implementing full PVsyst-consistent calculations using pvlib. The **JavaScript models** are optimized approximations designed for deployment in ThingsBoard rule chains.

---

## Quick Start

### 1. Install Dependencies

```bash
cd Power-Prediction
pip install -r requirements.txt
```

### 2. Configure Your Plant

Edit `config/plant_config.json` with your solar plant parameters (location, modules, inverter, losses). See `config/plant_config_example.json` for reference.

### 3. Run a Prediction

```bash
cd scripts/python_physics

# Using free NASA POWER hourly data (no API key needed)
python physics_model.py --source nasa_power --start 20251201 --end 20251215

# Using free NASA POWER daily data (full physics via synthetic hourly profiles)
python physics_model.py --source nasa_power_daily --start 20251201 --end 20251215

# Using local CSV file (hourly data)
python physics_model.py --source csv

# Using local CSV file (daily data, full physics via synthetic hourly profiles)
python physics_model.py --source csv_daily --csv-path ../data/daily.csv

# Using Solcast API (requires API key)
python physics_model.py --source api

# Use simplified daily model for faster but less accurate results (daily sources only)
python physics_model.py --source nasa_power_daily --start 20251201 --end 20251215 --simplified
```

**Note:** Daily data sources (`csv_daily`, `nasa_power_daily`) use synthetic hourly profiles generated from daily totals, enabling the full physics model for better accuracy. Use `--simplified` for faster but less accurate daily calculations.

Output is saved to `output/pv_generation.csv`.

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           DATA SOURCES                                       │
├─────────────┬─────────────┬─────────────┬─────────────┬─────────────────────┤
│ NASA POWER  │   Solcast   │  Open-Meteo │    ERA5     │    Local CSV        │
│ (free)      │ (API key)   │   (free)    │ (CDS API)   │                     │
└──────┬──────┴──────┬──────┴──────┬──────┴──────┬──────┴──────────┬──────────┘
       │             │             │             │                 │
       └─────────────┴─────────────┴─────────────┴─────────────────┘
                                   │
                     ┌─────────────▼─────────────┐
                     │     plant_config.json     │
                     │  (location, modules, etc) │
                     └─────────────┬─────────────┘
                                   │
       ┌───────────────────────────┼───────────────────────────┐
       │                           │                           │
       ▼                           ▼                           ▼
┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
│  Python Physics │     │   JS Physics    │     │   JS Surrogate  │
│   (pvlib)       │────▶│  (ThingsBoard)  │     │  (regression)   │
│                 │     │                 │     │                 │
│ REFERENCE MODEL │     │  CALIBRATED TO  │     │   TRAINED ON    │
│                 │     │  MATCH PYTHON   │     │   PYTHON OUTPUT │
└────────┬────────┘     └────────┬────────┘     └────────┬────────┘
         │                       │                       │
         └───────────────────────┼───────────────────────┘
                                 │
                     ┌───────────▼───────────┐
                     │   Validation Tools    │
                     │  (compare_accuracy,   │
                     │   validate_model,     │
                     │   run_benchmark)      │
                     └───────────────────────┘
```

---

## Workflows

### Workflow 1: Daily Predictions (Recommended for Operations)

Run daily predictions using multiple data sources with automatic fallbacks:

```bash
cd scripts/python_physics

# Predict for today + tomorrow (default)
python daily_predictor.py

# Today only
python daily_predictor.py --today-only

# Custom date range
python daily_predictor.py --start 20260107 --end 20260108
```

**Output files:**
- `output/hourly/predictions_*.csv` - Hourly power (kW)
- `output/daily/predictions_*.csv` - Daily energy (kWh)
- `output/monthly/predictions_*.csv` - Monthly energy (kWh)

**Compare against actual generation:**
```bash
cd scripts/shared
python compare_accuracy.py
```

```
┌──────────────────┐     ┌──────────────────┐     ┌──────────────────┐
│ plant_config.json│────▶│daily_predictor.py│────▶│ output/hourly/   │
└──────────────────┘     └────────┬─────────┘     │ output/daily/    │
                                  │               │ output/monthly/  │
                                  ▼               └────────┬─────────┘
                         ┌──────────────────┐              │
                         │ NASA/Solcast/    │              │
                         │ OpenMeteo/ERA5   │              │
                         │ (with fallbacks) │              │
                         └──────────────────┘              │
                                                           ▼
                         ┌──────────────────┐     ┌──────────────────┐
                         │meteocontrol_     │────▶│compare_accuracy  │
                         │actual.csv        │     │.py               │
                         └──────────────────┘     └────────┬─────────┘
                                                           │
                                                           ▼
                                                  ┌──────────────────┐
                                                  │accuracy_report   │
                                                  │.txt / .json      │
                                                  └──────────────────┘
```

---

### Workflow 2: Train Surrogate Model (For ThingsBoard Deployment)

Train a fast regression model using physics model outputs:

```bash
# Step 1: Prepare training data (fetches NASA POWER + runs physics model)
cd scripts/shared
python prepare_training_data.py --year 2024 --test-ratio 0.2 --seed 42

# Step 2: Train surrogate model on training data only
cd ../js_surrogate
python fit_surrogate.py --train-data ../../data/train_data.csv

# Step 3: Evaluate on held-out test data
cd ../shared
python evaluate_models.py --test-data ../../data/test_data.csv
```

**Output files:**
- `data/train_data.csv` - Training set (80% of days)
- `data/test_data.csv` - Test set (20% of days)
- `output/surrogate_coefficients.json` - Trained coefficients
- `output/js_coefficients.txt` - Copy-paste ready for ThingsBoard
- `output/test_evaluation_metrics.json` - Test performance metrics

```
┌──────────────────┐
│prepare_training_ │
│data.py           │
└────────┬─────────┘
         │ fetches NASA POWER
         │ runs physics model
         │ splits by DAY (no leakage)
         ▼
┌──────────────────┐     ┌──────────────────┐
│ train_data.csv   │     │ test_data.csv    │
│ (80% of days)    │     │ (20% of days)    │
└────────┬─────────┘     └────────┬─────────┘
         │                        │
         ▼                        │
┌──────────────────┐              │
│ fit_surrogate.py │              │
│ (sklearn linear  │              │
│  regression)     │              │
└────────┬─────────┘              │
         │                        │
         ▼                        ▼
┌──────────────────┐     ┌──────────────────┐
│surrogate_        │────▶│evaluate_models   │
│coefficients.json │     │.py               │
└──────────────────┘     └────────┬─────────┘
                                  │
                                  ▼
                         ┌──────────────────┐
                         │test_evaluation_  │
                         │metrics.json      │
                         └──────────────────┘
```

---

### Workflow 3: Calibrate JS Physics Model

Optimize the JavaScript physics model to match Python pvlib output:

```bash
# Step 1: Generate calibration data (100K samples from physics model)
cd scripts/js_physics
python generate_calibration_data.py --samples 100000 --seed 42

# Step 2: Optimize parameters using differential evolution
python optimize_js_params.py --maxiter 100

# Step 3: Validate calibration results
python validate_calibration.py
```

**Output files:**
- `data/js_calibration_data.csv` - Calibration samples
- `output/js_physics_calibration.json` - Optimized parameters
- `output/js_calibration_validation.png` - Validation plots

```
┌──────────────────┐
│generate_         │
│calibration_      │
│data.py           │
└────────┬─────────┘
         │ 100K random weather samples
         │ + Python physics model output
         ▼
┌──────────────────┐
│js_calibration_   │
│data.csv          │
└────────┬─────────┘
         │
         ▼
┌──────────────────┐
│optimize_js_      │
│params.py         │
│ (scipy DE)       │
└────────┬─────────┘
         │ minimizes error vs Python
         ▼
┌──────────────────┐
│js_physics_       │
│calibration.json  │
└────────┬─────────┘
         │
         ▼
┌──────────────────┐     ┌──────────────────┐
│validate_         │────▶│js_calibration_   │
│calibration.py    │     │validation.png    │
└──────────────────┘     └──────────────────┘
```

---

### Workflow 4: Benchmark All Models

Compare all three models against actual Meteocontrol data:

```bash
cd scripts/shared

# Using local Solcast CSV data
python run_benchmark.py

# Using NASA POWER data
python run_benchmark.py --data-source nasa_power --start 20251210 --end 20251215
```

**Output files:**
- `output/benchmark_results.csv` - Daily comparison table
- `output/benchmark_comparison.png` - Visual comparison
- `output/benchmark_metrics.json` - Machine-readable metrics

---

## File Reference

### Configuration (`config/`)

| File | Purpose | When to Use |
|------|---------|-------------|
| `plant_config.json` | Plant parameters (location, modules, inverter, losses) | **Edit first** for any new plant |
| `plant_config_example.json` | Example configuration with comments | Reference when setting up |

### Data Fetchers (`scripts/python_physics/`)

| File | Purpose | Dependencies | API Key |
|------|---------|--------------|---------|
| `fetch_nasa_power.py` | Fetch free irradiance data from NASA POWER | None | Not required |
| `fetch_era5_weather.py` | Fetch ERA5 temperature/wind from Copernicus | cdsapi, xarray | CDS API required |

### Physics Models

| File | Purpose | When to Use |
|------|---------|-------------|
| `scripts/python_physics/physics_model.py` | Full pvlib physics model (ground truth). Supports both hourly and daily data. Daily data uses synthetic hourly profiles for full physics accuracy. Use `--simplified` flag for faster daily calculations. | Engineering validation, training data, daily or hourly irradiance |
| `scripts/python_physics/daily_predictor.py` | Automated daily predictions with fallbacks | Production daily forecasts |
| `scripts/js_physics/physics_model.js` | Simplified physics for ThingsBoard | Real-time SCADA monitoring |
| `scripts/js_surrogate/surrogate_model.js` | Fast regression model | High-frequency predictions |

### Training Pipeline (`scripts/shared/` and `scripts/js_surrogate/`)

| File | Purpose | Prerequisites |
|------|---------|---------------|
| `prepare_training_data.py` | Fetch data + run physics + create train/test split | `plant_config.json` |
| `fit_surrogate.py` | Train regression coefficients | `train_data.csv` |
| `evaluate_models.py` | Evaluate all models on test data | `test_data.csv`, trained coefficients |

### Calibration Pipeline (`scripts/js_physics/`)

| File | Purpose | Prerequisites |
|------|---------|---------------|
| `generate_calibration_data.py` | Generate random samples with physics output | `plant_config.json` |
| `optimize_js_params.py` | Optimize JS model parameters | Calibration data |
| `optimize_js_params_gpu.py` | GPU-accelerated optimization (optional) | Calibration data, CUDA |
| `validate_calibration.py` | Validate calibrated parameters | Calibration results |

### Validation & Benchmarking (`scripts/shared/`)

| File | Purpose | Prerequisites |
|------|---------|---------------|
| `compare_accuracy.py` | Compare predictions vs Meteocontrol actuals | Prediction files, `meteocontrol_actual.csv` |
| `validate_model.py` | Quick model validation | `pv_generation.csv`, `meteocontrol_actual.csv` |
| `run_benchmark.py` | Full benchmark: Python + JS models vs actuals | All models, actual data |
| `benchmark_js.js` | Node.js runner for JS model benchmarking | Called by `run_benchmark.py` |

### Utilities (`scripts/shared/`)

| File | Purpose |
|------|---------|
| `timezone_utils.py` | Consistent UTC/local time conversion |
| `plot_inverter_curve.py` | Visualize inverter efficiency curve |

---

## Data Sources

| Source | API Key | Resolution | Cost | Best For |
|--------|---------|------------|------|----------|
| **NASA POWER** | Not required | Hourly/Daily | Free | Historical data, testing |
| **Solcast** | Required | Hourly | Paid | Production forecasts |
| **Open-Meteo** | Not required | Hourly | Free | Forecasts (16 days ahead) |
| **ERA5** | CDS API required | Hourly | Free | Weather validation |

**Resolution Options:** The physics model supports both hourly and daily data. Daily data sources (`csv_daily`, `nasa_power_daily`) use synthetic hourly profiles generated from daily totals via pvlib clear-sky models, enabling full physics calculations with better accuracy than simplified daily models. Use `--simplified` flag for faster but less accurate daily calculations.

### NASA POWER (Recommended for Free Access)

NASA POWER provides satellite-derived irradiance from 1981 to ~7 days ago.

```bash
cd scripts/python_physics

# Fetch hourly data (direct physics model)
python fetch_nasa_power.py --start 20251201 --end 20251215 --mode hourly
python physics_model.py --source nasa_power --start 20251201 --end 20251215

# Fetch daily data (full physics via synthetic hourly profiles)
python fetch_nasa_power.py --start 20251201 --end 20251215 --mode daily
python physics_model.py --source nasa_power_daily --start 20251201 --end 20251215
```

---

## Plant Configuration

The `config/plant_config.json` file contains all plant-specific parameters. Key sections:

```json
{
  "location": {
    "lat": 8.342,           // Latitude (decimal degrees)
    "lon": 80.376,          // Longitude (decimal degrees)
    "altitude_m": 88,       // Altitude (meters)
    "timezone": "Asia/Colombo"
  },
  
  "orientations": [         // Each unique tilt/azimuth combination
    {"tilt": 18, "azimuth": 148, "name": "O1", "module_count": 18}
  ],
  
  "module": {
    "area_m2": 2.556,       // Module area (m²)
    "efficiency_stc": 0.2153,// STC efficiency (from PVsyst)
    "gamma_p": -0.00340     // Temperature coefficient (per °C)
  },
  
  "inverter": {
    "ac_rating_kw": 55.0,   // Total AC rating
    "use_efficiency_curve": true,
    "efficiency_curve_kw": [...],
    "efficiency_curve_eta": [...]
  },
  
  "losses": {
    "soiling": 0.03,
    "lid": 0.014,
    "module_quality": -0.008, // Negative = gain
    "mismatch": 0.017,
    "dc_wiring": 0.009,
    "ac_wiring": 0.003,
    "far_shading": 1.0      // 1.0 = no shading
  }
}
```

---

## Data Files

### Input Data (`data/`)

| File | Description | Source |
|------|-------------|--------|
| `solcast_irradiance.csv` | Hourly GHI, DNI, DHI, temp, wind | Solcast API |
| `nasa_power_hourly.csv` | Hourly irradiance | `fetch_nasa_power.py` |
| `nasa_power_2024.csv` | Year 2024 data | `prepare_training_data.py` |
| `meteocontrol_actual.csv` | Actual daily generation | Meteocontrol export |
| `era5_weather.csv` | Temperature, wind from ERA5 | `fetch_era5_weather.py` |
| `train_data.csv` | Training set (80%) | `prepare_training_data.py` |
| `test_data.csv` | Test set (20%) | `prepare_training_data.py` |

### Output Files (`output/`)

| File | Description | Generated By |
|------|-------------|--------------|
| `pv_generation.csv` | Hourly AC power predictions | `physics_model.py` |
| `hourly/*.csv` | Hourly predictions per source | `daily_predictor.py` |
| `daily/*.csv` | Daily energy per source | `daily_predictor.py` |
| `monthly/*.csv` | Monthly energy per source | `daily_predictor.py` |
| `surrogate_coefficients.json` | Trained regression coefficients | `fit_surrogate.py` |
| `js_physics_calibration.json` | Calibrated JS parameters | `optimize_js_params.py` |
| `benchmark_*.{csv,png,json,md}` | Benchmark results | `run_benchmark.py` |
| `accuracy_report.txt` | Accuracy comparison | `compare_accuracy.py` |

---

## Timezone Handling

All scripts use consistent timezone handling via `timezone_utils.py`:

| Data Source | Native Timezone | Notes |
|-------------|-----------------|-------|
| Solcast API | UTC | Explicit in timestamps |
| NASA POWER | UTC | Converted from YYYYMMDDHH format |
| Open-Meteo | UTC | Explicit in timestamps |
| ERA5 | UTC | Explicit in timestamps |
| Meteocontrol | Asia/Colombo (+05:30) | Day numbers only in CSV |
| **Model outputs** | **Asia/Colombo (+05:30)** | All final outputs in local time |

---

## Requirements

### Python

```bash
pip install -r requirements.txt
```

Core dependencies:
- pandas, numpy - Data manipulation
- pvlib - Solar physics calculations
- requests - API calls
- matplotlib - Plotting
- scikit-learn - Regression training
- scipy - Optimization
- tqdm - Progress bars

Optional (for ERA5):
- cdsapi - Copernicus CDS API
- xarray - NetCDF processing

### JavaScript / Node.js

Required only for:
- Running JS model benchmarks (`run_benchmark.py`, `evaluate_models.py`)
- ThingsBoard deployment

```bash
# Verify Node.js is installed
node --version
```

---

## Typical Model Performance

| Model | Test R² | Test MAE | Notes |
|-------|---------|----------|-------|
| Python Physics | Baseline | Baseline | Ground truth |
| JS Physics | ~0.90+ | ~2-3 kW | After calibration |
| JS Surrogate | ~0.25-0.40 | ~6-10 kW | Limited by training data |

The surrogate model typically shows lower test performance than training performance. High training R² (0.95+) with low test R² indicates overfitting - this is expected and correct behavior.

---

## ThingsBoard Deployment

1. **For Physics Model:** Copy contents of `scripts/js_physics/physics_model.js` to a ThingsBoard Script node
2. **For Surrogate Model:** Copy contents of `scripts/js_surrogate/surrogate_model.js` to a ThingsBoard Script node

Update the `CALIBRATION` or `REGRESSION_COEFFICIENTS` objects with values from:
- `output/js_physics_calibration.json` (for physics model)
- `output/surrogate_coefficients.json` (for surrogate model)

---

## Notes

- All Python scripts support CLI arguments - use `--help` for options
- Internal processing uses UTC; final outputs are in Asia/Colombo (+05:30)
- NASA POWER data is free but has ~7-day lag
- Surrogate model requires training before use
- Far-shading is controlled via `plant_config.json` (`far_shading < 1.0` enables it)
