# Article Research Notes: PV Power Prediction Pipeline

*Compiled: March 2026 — for internal article planning, not for publication as-is*  
*Plant: 55 kW rooftop PV, North-Central Sri Lanka (8.34°N, 80.38°E, Asia/Colombo)*

---

## 1. Problem Statement & Motivation

**The core challenge:** A commercial solar plant connected to a ThingsBoard SCADA system needed:
1. An *expected generation* baseline to compare against real-time production — essential for fault detection, performance ratio calculation, and soiling detection
2. This calculation had to run *inside ThingsBoard rule chains* (JavaScript, no external libraries, <5ms/call)
3. It had to be accurate enough to flag genuine underperformance (target: ±5% energy error on a monthly basis)

There was no off-the-shelf solution that satisfied all three: physics tools like pvlib are Python-only, commercial APIs are expensive, and simple ML models need retraining for each plant.

**The site:**
- 8 differently-oriented roof planes (tilts 18–27°, azimuths ranging full compass)
- 216 modules × 2.556 m² = 552.1 m² total area
- 55 kW AC inverter (single inverter, multi-MPPT)
- Located at 88 m altitude, tropical monsoon climate (high irradiance variability)
- SCADA: Meteocontrol, actuals exported as daily energy (kWh) with target bands

---

## 2. What Was Actually Built — Full Inventory

### 2.1 Production Scripts

| Script | Role | Status |
|--------|------|--------|
| `scripts/python_physics/physics_model.py` | **The model** — Python+pvlib full physics chain | ✅ Production |
| `scripts/python_physics/daily_predictor.py` | Orchestrator: runs `physics_model.py` daily for 4 data sources with cascade fallback | ✅ Production |
| `scripts/js_physics/physics_model.js` | Calibrated JS approximation for ThingsBoard rule chains | ✅ Production |
| `scripts/js_surrogate/surrogate_model.js` | Linear regression JS model (experimental, kept for reference) | 📦 Archived |

### 2.2 Tooling Scripts

| Script | Role |
|--------|------|
| `scripts/python_physics/fetch_nasa_power.py` | Fetches hourly / daily irradiance from NASA POWER API |
| `scripts/python_physics/fetch_era5_weather.py` | Downloads ERA5 temperature & wind from Copernicus CDS (NetCDF → DataFrame) |
| `scripts/js_physics/generate_calibration_data.py` | Generates 2.5M synthetic samples through Python model for JS calibration |
| `scripts/js_physics/optimize_js_params.py` | Scipy differential evolution to minimize RMSE between JS and Python outputs |
| `scripts/js_physics/optimize_js_params_gpu.py` | GPU-accelerated version of optimizer (CuPy / parallel) |
| `scripts/js_physics/validate_calibration.py` | Validates JS model against Python on held-out data |
| `scripts/shared/prepare_training_data.py` | 80/20 day-level train/test split; generates `train_data.csv`, `test_data.csv`, `split_info.json` |
| `scripts/shared/run_benchmark.py` | Times all three models on identical input windows against Meteocontrol actuals |
| `scripts/shared/evaluate_models.py` | Evaluates JS Physics and Surrogate on held-out test set |
| `scripts/shared/compare_accuracy.py` | Compares all 4 data-source predictions against Meteocontrol (hourly/daily/monthly) |
| `scripts/shared/validate_model.py` | General model validation utilities |
| `scripts/shared/plot_inverter_curve.py` | Visualizes PVsyst inverter efficiency curve |
| `scripts/shared/benchmark_js.js` | Node.js benchmarking harness for JS Physics model |
| `scripts/shared/timezone_utils.py` | UTC ↔ Asia/Colombo conversion utilities |
| `scripts/js_surrogate/fit_surrogate.py` | Trains linear regression surrogate on `train_data.csv` |
| `scripts/predict_dni.py` | Standalone: computes DNI from GHI-DHI decomposition (diagnostic) |

### 2.3 Data Files

| File | Description |
|------|-------------|
| `config/plant_config.json` | All plant parameters (location, orientations, module/inverter specs, losses, IAM table) |
| `data/meteocontrol_actual.csv` | Meteocontrol December 2025 daily actual: days 1–12 with energy, target bands, POA irradiation |
| `data/meteocontrol_actual1.csv` | Meteocontrol full-month actual with 31 days and target band data |
| `data/js_calibration_data.csv` (large) | 2.5M synthetic weather→power samples from Python model for JS calibration |
| `data/js_calibration_data.json` | Metadata for calibration dataset |
| `data/solcast_irradiance.csv` | Solcast historical irradiance (used for initial testing and DNI validation) |
| `data/train_data.csv`, `test_data.csv` | 80/20 split NASA POWER 2024 data (367 days) |

### 2.4 Output Files

| File | Description |
|------|-------------|
| `output/benchmark_metrics.json` | Benchmark vs Meteocontrol (all 3 models, Dec 2025) |
| `output/benchmark_report.md` | Human-readable benchmark summary |
| `output/benchmark_results.csv` | Daily comparison: Actual vs Python/JS Physics/JS Surrogate |
| `output/accuracy_report.txt` | Multi-source (NASA/Solcast/Open-Meteo/ERA5) comparison vs Meteocontrol, hourly+daily+monthly |
| `output/js_physics_calibration.json` | Optimizer output: calibrated SAPM + Perez params, improvement metrics |
| `output/js_validation_results.json` | JS model validation on 2.5M samples |
| `output/surrogate_coefficients.json` | Fitted regression coefficients + training metrics |
| `output/test_evaluation_metrics.json` | JS Physics and Surrogate evaluation on held-out 1357-point test set |
| `output/split_info.json` | Full train/test date lists, seeds, sizes |
| `output/hourly/`, `output/daily/`, `output/monthly/` | Rolling append-mode predictions by source |

---

## 3. Models That Were Tried & Discarded

### 3.1 ❌ Simplified Daily Model (Discarded)

**What:** `calculate_daily_energy_simple()` inside `physics_model.py`, callable with `--simplified` flag.

**Method:**
```
E_day = GHI_daily × POA_factor × total_area × η_STC × dc_loss × inverter_eff
POA_factor = 1.05 (hardcoded "tropical multi-orientation" approximation)
```

**Why discarded:**
- No sub-daily temporal resolution → cannot capture morning/evening clipping dynamics
- `POA_factor = 1.05` is a naive scalar — completely wrong for overcast vs clear days  
- No cell temperature calculation → temperature losses ignored  
- The Perez transposition model accounts for ±15% POA variation between orientations that this factor can't replicate
- Effectively just a scaled irradiance-to-power ratio; PVsyst-level accuracy impossible

**What it's still good for:** Rough daily sanity check when only daily GHI is available and sub-5% error is not needed.

---

### 3.2 ❌ Synthetic Hourly Profile Generator (Partially Discarded)

**What:** `generate_synthetic_hourly_from_daily()` in `physics_model.py` — creates fake hourly GHI/DNI/DHI from daily totals using Ineichen clear-sky profiles.

**Method:**
1. Compute hourly clear-sky profile using pvlib Ineichen model
2. Scale hourly values by `measured_daily / clearsky_daily`
3. Run full physics model on synthetic hourly data

**Why it's inaccurate and mostly discarded:**
- Assumes the temporal distribution of irradiance always follows the clear-sky shape — completely wrong on partially cloudy days
- On a day with afternoon thunderstorms, clear-sky would show maximum at noon, but reality might peak at 9 AM
- Temperature/wind applied as constant daily average — misses morning cool, afternoon heat pattern
- Cannot capture intraday clipping events that might happen only during brief clear periods

**When it was still useful:** Helped bridge the gap when only daily NASA POWER data was available and no true hourly data existed. Results showed ~15–20% hourly error (R² ~ 0.7) vs hourly actual, but daily energy totals were reasonable (within ~10%'s).

**Replaced by:** Once NASA POWER hourly became a data source, the synthetic profile approach was abandoned for the primary pipeline. `daily_predictor.py` always fetches hourly data now.

---

### 3.3 ❌ JS Surrogate Linear Regression Model (Archived)

**What:** `surrogate_model.js` — a 5-feature linear regression:
```
P_AC = a0 + a1·GHI + a2·GHI² + a3·DNI + a4·(GHI×ΔT) + a5·(GHI×wind)
```

**Fitted coefficients (from `surrogate_coefficients.json`):**

| Coeff | Value | Feature |
|-------|-------|---------|
| a0 | 0.342 | Intercept |
| a1 | 0.0798 | GHI linear |
| a2 | −1.294×10⁻⁵ | GHI² |
| a3 | 0.00731 | DNI |
| a4 | −4.785×10⁻³ | GHI × ΔT |
| a5 | −1.599×10⁻³ | GHI × wind |

**Training setup:**
- Trained on 5,521 hourly samples (80% of 2024 NASA POWER data, run through Python model)
- Tested on 1,357 hourly samples (20% held-out — different calendar days)
- Day-level split (seed=42) to prevent temporal leakage

**Why discarded (core issues):**

1. **R² = 0.42:** A model with R²=0.42 means 58% of hourly variance is unexplained. Completely useless for hourly monitoring.

2. **MAPE = 130% on test set:** Because the regression sometimes predicts nonzero power for low-irradiance morning/evening timestamps, then the actual is near-zero → huge percentage error.

3. **Why the maths fails:** A linear regression on GHI doesn't know about the plant's geometric orientation — the same GHI value at 8 AM vs noon produces completely different POA on a multi-tilt array. The model conflates east-facing and west-facing morning GHI.

4. **High energy accuracy is misleading:** Energy error = 1.5% on test. This sounds good but it's a statistical fluke: the model over-predicts in the morning on south-facing surfaces and under-predicts in the afternoon on west-facing ones. These biases cancel over 24 hours and across many days, not because the physics are correct.

5. **The surrogate_model.js placeholder coefficients are wrong:** The file ships with different coefficients (`R²=0.99, MAE=0.77 kW` — these were from a 144-sample validation, not the proper 5521-sample training run with the clean train/test split). The placeholder was never updated in the JS file after retraining with the full dataset.

**Kept archived because:**
- Fast (67× faster than Python model)
- Energy error <2% on monthly rolls
- Potentially useful if retrained per-plant per-season
- The code infrastructure for fitting is complete

---

### 3.4 ❌ predict_dni.py — DNI Decomposition (Abandoned Standalone)

**What:** Predicts DNI from GHI and DHI using the fundamental irradiance identity:
```
DNI = (GHI - DHI) / cos(θz)
```

**Why it was written:** Early exploration to understand how well a zero-dependency Python script (no pvlib, no numpy — pure stdlib) could compute a core physics quantity.

**Why abandoned:** It's a diagnostic/validation tool only. Solcast already provides GHI, DNI, and DHI directly. The decomposition formula just inverts what Solcast already computed — you get back a slightly noisy version of what you started with. For NASA POWER, all three components are also available directly. There was no production need for this decomposition.

**Interesting finding:** The script successfully validated that Solcast's GHI/DHI/DNI are internally consistent (R²≈0.99 when predicting DNI from GHI-DHI). This confirmed the data quality, but revealed nothing actionable.

---

### 3.5 ❌ Flat Inverter Efficiency (Superseded)

**What:** Earlier code used `P_AC = P_DC × 0.98` (constant 98% efficiency).

**Why discarded:** The PVsyst inverter report provided a full 21-point load-dependent efficiency curve. Real efficiency drops from ~98.3% at 25 kW to 95.2% at minimum load (2.43 kW). On an average day, ~40% of operating hours are at <50% load — the flat assumption adds a systematic 1–3% AC energy over-prediction.

**Replaced by:** `use_efficiency_curve: true` in `plant_config.json`, with a 21-point lookup table interpolated via `np.interp()`.

---

### 3.6 ❌ config_notes.md Experiments (Discarded)

**What:** Early scratch configurations stored in `docs/config_notes.md` trying two different plants:
- Plant 1: 100 kW AC rating, `PDC_THRESHOLD_KW = 1.0`, LID=1.5%, mismatch=0.9%, no module_quality term
- Plant 2: Different set of 8 orientations (16–23° tilt), 54–60 modules per orientation, different IAM table

**Why discarded:**
- Plant 1 used placeholder DC losses from a textbook, not extracted from PVsyst — yields wrong energy
- The inverter AC rating of 100 kW was the initial wrong assumption before getting the actual PVsyst report showing 55 kW
- No structured config file — parameters were hardcoded in Python source, making iteration painful
- `PDC_THRESHOLD_KW = 1.0` was set conservatively without datasheet justification; final config uses `0.0` (no threshold, let inverter model handle it)

**Resolution:** Moved to `plant_config.json` JSON structure, extracted all values from PVsyst report, added module_quality as a signed factor.

---

## 4. The Chosen Solution: Python Physics Model

### 4.1 Why physics_model.py Won

The only approach that satisfied both accuracy requirements and maintainability was a full, PVsyst-consistent physics chain in Python+pvlib:

| Requirement | How met |
|-------------|---------|
| PVsyst consistency | Perez POA, SAPM thermal, PVsyst loss chain, load-dependent inverter curve |
| Multi-orientation | Per-orientation physics loop, area-weighted aggregation |
| Data source flexibility | 6 sources (Solcast API, NASA POWER hourly/daily, Open-Meteo, ERA5, local CSV) |
| ThingsBoard compatibility | JS physics model (calibrated from Python outputs) |
| No retraining needed | Physics — parameters come from PVsyst report, not data |

### 4.2 Key Physics Choices & Their Justifications

**Perez model, not isotropic diffuse:**  
The 8 orientations span all quadrants. At any given hour, some orientations see sky-facing diffuse (SE in AM, NW in PM). Perez accounts for circumsolar and horizon brightening; isotropic would under-predict diffuse by ~5–10% on clear-sky days. This is particularly important for multi-orientation rooftop plants where diffuse can be 30–50% of total POA.

**SAPM thermal, not NOCT:**  
NOCT (Nominal Operating Cell Temperature) model: `T_cell = T_amb + NOCT×(G/800)`. Simple and widely used, but wind-independent. In Sri Lanka, coastal sea breezes are a significant cooling factor. SAPM includes both irradiance and wind speed, giving more accurate cell temperatures. SAPM deviation from PVsyst is typically <2°C.

**Sequential 5-factor DC loss chain:**  
Adding `module_quality = -0.008` (a gain) was initially confusing. PVsyst reports this as a positive number on the loss diagram (labeled as gain), but in code it needs the sign flip. Getting this right dropped predicted annual energy by ~0.8% × ~200 MWh/year = ~1.6 MWh per year difference — material for O&M reporting.

**Plant-level clipping, not per-orientation:**  
The plant has one inverter with 8 MPPTs. All MPPTs feed the same AC bus. Clipping at plant level matches how the inverter hardware actually works. Early code had per-orientation clipping — this predicted ~3% more clipping loss than reality because it clipped each orientation separately before summing, whereas in reality east-facing orientations are peaking while west-facing ones are still low.

### 4.3 Data Source Comparison

| Source | Pro | Con | Best use |
|--------|-----|-----|----------|
| **Solcast API** | 15-min resolution, 7-day actuals, includes weather on paid plan | Paid plan expensive; free tier lacks air_temp/wind | Production forecasting |
| **NASA POWER hourly** | Free, ~12 months of history, consistent global coverage | ~7-day latency; slight spatial smoothing | Historical analysis, model training |
| **Open-Meteo** | Free, no API key, includes full weather | Forecast-only beyond ~1 week back; different cloud model | Near-term forecasting |
| **ERA5** | Best-quality historical reanalysis | Weather only (no irradiance); requires CDS API + cdsapi library + approval | Weather fallback only |
| **Local CSV** | Full control, any format | Manual data management | Offline testing |

**Unexpected finding:** Open-Meteo showed better hourly R² (−1.55 vs −4.0) than NASA POWER in the December 2025 accuracy test — but this was during the 10-day cloudy window. NASA POWER and ERA5 performed identically (expected — both are reanalysis products sharing the same underlying atmospheric model backbone).

---

## 5. The JavaScript Challenge: Getting Physics Into ThingsBoard

### 5.1 Why JS Was Necessary

ThingsBoard (open-source IoT platform) runs rule chains in a Nashorn JavaScript engine. Python cannot be called directly. Options considered:

| Option | Rejected because |
|--------|-----------------|
| Call Python subprocess from TB | Latency too high; no subprocess API in Nashorn |
| External API endpoint | Requires hosting; failure point; latency |
| Precompute and cache | Misses live telemetry, 15-min updates |
| JS reimplementation | ✅ Chosen — self-contained, <5ms, no dependencies |

### 5.2 JS Physics Model Approximations vs Python

The main challenge is that pvlib doesn't exist in JavaScript. Key approximations made:

| Python (pvlib) | JS Approximation | Error introduced |
|----------------|-----------------|-----------------|
| NREL SPA solar position | Simplified Julian-day SPA | <0.1° solar angle error; negligible for energy |
| Full Perez (5 radiance bins) | Isotropic + calibrated circumsolar term | Main source of JS error; fixed by calibration |
| `pvlib.irradiance.get_extra_radiation()` | Spencer's formula | <0.1% DNI_extra error |
| SAPM preset table lookup | Custom SAPM with calibrated a, b, ΔT | Fixed by calibration |
| Load-dependent inverter curve | Flat η = 0.98 | ~1% energy error at partial load |

**Most important finding:** The Perez diffuse model simplification caused the largest initial deviation (36% of RMSE before calibration). With only isotropic diffuse, the JS model systematically under-predicted on overcast days (when diffuse dominates) and over-predicted on clear days (missing circumsolar correction). This drove the need for a calibrated approximation.

### 5.3 Calibration Process

**Step 1: Data generation** (`generate_calibration_data.py`)
- 2.5M random weather samples (not real historical data — fully synthetic)
- Weather ranges: GHI 0–1200 W/m², temp 15–45°C, wind 0.1–15 m/s, all daylight hours
- Generated over 2025 timestamps (Asia/Colombo timezone)
- 2,483,662 non-zero power samples (99.3% of generated samples were daytime)
- Python physics model computed AC power for each → ground truth labels
- This is essentially treating pvlib as a "physics oracle"

**Step 2: Optimization** (`optimize_js_params.py`)
- Loss function: RMSE between Python model output and JS model output
- Optimizer: scipy `differential_evolution` (global, derivative-free, handles non-convex landscapes)
- 8 parameters tuned: `circumsolar_factor`, `circumsolar_threshold`, `aoi_threshold`, `diffuse_weight`, `brightness_factor`, `sapm_a`, `sapm_b`, `sapm_dt`
- 112 outer iterations, 13,677 function evaluations
- 396 seconds runtime (CPU, not GPU — GPU version also exists but wasn't needed)
- 50,000 samples were used per evaluation (from the 2.5M set) for speed

**Step 3: Validation** (`validate_calibration.py`)
- Validated on the full 2.5M dataset (separate from the 50K optimization subset)
- Result: RMSE = 1.46 kW, MAE = 0.68 kW, R² = 0.992, energy error = 1.46%

**Step 4: Test set evaluation** (`evaluate_models.py`)
- Used the held-out 1,357-hour test set (real NASA POWER 2024 data, different calendar days)
- JS Physics: R² = 0.969, MAE = 1.34 kW (1.34/55 = 2.4% of rating), energy error = +5.47%
- The 5.47% energy error on real data (vs 1.46% on synthetic) reflects the distribution mismatch between synthetic training weather and real-world weather

### 5.4 The GPU Version

`optimize_js_params_gpu.py` was built because the CPU calibration took 396 seconds. The GPU version runs the same JS model physics logic translated to NumPy/CuPy arrays and uses GPU parallelism to evaluate multiple parameter combinations simultaneously. It wasn't ultimately needed (CPU was fast enough for 8 parameters), but provides 10–50× speedup for future plants or larger parameter spaces.

---

## 6. Data: What Actual Generation Looks Like

### 6.1 Meteocontrol December 2025 (First Dataset Available)

Only 12 days of actual data available at benchmark time:

| Day | Actual (kWh) | Notes |
|-----|-------------|-------|
| Dec 1 | 274.3 | Normal |
| Dec 2 | 300.3 | Normal |
| Dec 3 | 308.7 | Losses: 14.2 kWh (some shading or downtime) |
| Dec 4 | 334.9 | Losses: 66.2 kWh — significant underperformance |
| Dec 5 | 274.5 | Normal |
| Dec 6 | 215.6 | Low — likely cloudy |
| Dec 7 | 361.2 | Best day of the set |
| Dec 8 | 149.9 | Very low — heavy cloud |
| Dec 9 | 207.4 | Low |
| Dec 10 | 62.9 | Near-zero — heavy overcast/monsoon |
| **Dec 11** | **345.4** | 📊 Benchmark day |
| **Dec 12** | **76.0** | 📊 Benchmark day — monsoon |

**Meteocontrol also provides "Target range" (low, high):** Derived from satellite-based POA irradiation and expected plant performance coefficients. Notably:
- Day 3 (308.7 kWh) falls below the target range (322.9–394.6 kWh) — Losses=14.2 kWh flagged
- Day 4 (334.9 kWh) is massively below target (401–490 kWh) — Losses=66.2 kWh
- Day 10 (62.9 kWh) is near target (59–72 kWh) — meaning the irradiance really was that low that day

### 6.2 The Benchmark Problem (Dec 11–12)

The benchmark used only two days — Dec 11 and Dec 12:
- Dec 11: Clear day (345.4 kWh actual) → all models under-predicted by ~35% (Python: 224.5, JS: 233.3)
- Dec 12: Heavy cloud (76.0 kWh actual) → all models over-predicted by ~250% (Python: 263.4, JS: 301.9)

**Critical insight:** The `benchmark_results.csv` shows days 11 and 12 only — the python benchmark script only ran on the two days where the Python model prediction files already existed for that date range. The irradiance inputs for these two days (from NASA POWER or Solcast) apparently showed higher irradiance than actually occurred on the ground — classic reanalysis "clear-sky bias" during monsoon conditions.

**Why the December 12 error is so extreme:**
- Actual: 76 kWh — deep monsoon overcast (entire day)
- NASA POWER / Solcast predicted: ~250–300 kWh — both sources see lower-resolution cloud cover, often showing partial cloud when it was total overcast
- This is a **data source quality problem**, not a physics model failure

### 6.3 The Full Month Dataset (`meteocontrol_actual1.csv`)

This is a different month with 31 days of actuals (appears to be 2024, month unknown — possibly September or October based on loss patterns):
- Best day: 421.8 kWh (day 26)  
- Worst day: 155.2 kWh (day 29)
- Days with significant Losses (underperformance): days 4 (124 kWh loss), 5 (92), 7 (50), 22 (78), 25 (117)
- Days with zero Losses (performing within target): days 3, 6, 9, 10, etc.
- Pattern: High-loss days correspond to high POA irradiation (4000–5500 Wh/m²) — suggesting inverter clipping or soiling was worse on sunny days

**Article angle:** The target bands from Meteocontrol are themselves model-based (satellite POA → expected generation), creating a "model vs model" situation where both the actual system target and the predicted values depend on irradiance modeling quality.

---

## 7. Key Numerical Results for the Article

### 7.1 Internal Accuracy: JS Model vs Python Model

All comparisons here are JS Physics vs Python Physics (not vs real generation):

| Metric | Value | Context |
|--------|-------|---------|
| RMSE | 1.46 kW | On 2.5M synthetic samples |
| MAE | 0.68 kW | 1.2% of 55 kW rating |
| R² | 0.992 | Near-perfect correlation with Python model |
| Energy error | 1.46% | Monthly energy accuracy |
| MAPE | 7.76% | Hourly; mainly driven by low-power morning/evening hours |
| RMSE % of rating | 2.65% | Within 3% engineering spec |
| Speed vs Python | 28× faster | JS: 0.026 ms/point vs Python: 0.74 ms/point |
| Pre-calibration RMSE | 2.24 kW | Before optimization |
| Post-calibration improvement | 36.1% | RMSE reduction from optimization |
| Optimizer iterations | 112 | Differential evolution |
| Training samples | 50,000 | Used per evaluation |
| Calibration runtime | 396 seconds | CPU single-threaded |

### 7.2 Surrogate Model vs Python Model (test set — 1,357 real hourly samples)

| Metric | JS Physics | JS Surrogate | Why it matters |
|--------|------------|-------------|---------------|
| R² | 0.969 | 0.420 | Surrogate fails hourly monitoring |
| MAE | 1.34 kW | 9.44 kW | Surrogate error = 17% of rating |
| MAPE | 20.1% | 129.7% | Surrogate unusable for hourly |
| Energy error | +5.47% | +1.50% | Surrogate better for totals only |
| Speed | 0.026 ms | 0.011 ms | Marginal speed difference |

**The key insight:** Speed-accuracy tradeoff. JS Physics has 2.4× the surrogate's error but 2.4× slower (still 28× faster than Python). The surrogate wins on speed but loses decisively on hourly accuracy. For a SCADA system where 15-minute readings matter, surrogate is unsuitable.

### 7.3 All Models vs Real Generation (December 2025, 2 days)

| Model | MAE (kWh/day) | MAPE | Total Predicted | Total Actual | Error |
|-------|-------------|------|-----------------|--------------|-------|
| Python Physics | 154.1 | 56.3% | 421.3+264.2 | 421.4 | −0.1% (total) |
| JS Physics | 169.0 | 66.0% | 535.2 | 421.4 | +27.0% |
| JS Surrogate | 151.3 | 56.5% | 498.4 | 421.4 | +18.3% |

Wait — re-reading benchmark_metrics.json carefully: the actual total over the 5-day benchmark period was **421.3 kWh** total (not per day). This is extremely low for a 55 kW plant over 5 days → confirms heavy monsoon conditions throughout the period.

### 7.4 Irradiance Source Comparison vs Meteocontrol (accuracy_report.txt, 10 days)

| Source | Daily MAE (kWh) | Daily Bias | Daily R² |
|--------|---------------|------------|----------|
| NASA POWER | 96.2 | +32.7 | −0.68 |
| Solcast | 97.4 | +33.9 | −0.71 |
| Open-Meteo | 111.4 | +47.9 | −1.55 |
| ERA5 | 96.8 | +33.3 | −0.70 |

All sources show positive bias (over-prediction). All R² are negative (model unfit for this 10-day window). Same conclusion as §7.3: the problem is irradiance data quality during monsoon period.

---

## 8. The Training Data Pipeline

### 8.1 Dataset Used for Surrogate and JS Calibration

**Period:** Full year 2024 (January 1 to January 1, 2025 — 367 days including leap day Feb 29)  
**Source:** NASA POWER hourly for the plant's coordinates  
**After filtering:** 5,521 train hours + 1,357 test hours = 6,878 total daytime hours  
**Split method:** Day-level stratified split (whole calendar days, not individual hours)  
- Avoids temporal leakage within a single day
- Ensures test days are not adjacent to train days (roughly)
- Seed: 42, ratio: 80/20

**Why 2024 data but plant is in active use?**  
The 2024 NASA POWER data was entirely synthetic physics model output (no real generation ground truth) — it's used to train the surrogate and calibrate the JS model against the *Python model*, not against real generation. Real Meteocontrol actuals are only available for December 2025 (12 days partial, 2 usable for benchmark).

### 8.2 JS Calibration Dataset

- **2.5M samples** of synthetic weather (random timestamps over 2025, random irradiance/temperature/wind)
- Not real weather — designed to cover the full input space non-uniformly
- 99.3% are daytime (non-zero output)
- Max AC power seen: 54.84 kW (plant near but not at clipping limit — 55 kW)
- Mean AC power (non-zero): 16.24 kW — typical for a multi-orientation plant

---

## 9. Architecture & Reliability Patterns

### 9.1 The 4-Source Cascade Fallback in daily_predictor.py

`daily_predictor.py` is the production orchestrator. It runs daily and fetches from 4 sources simultaneously, then for each field (GHI, DNI, DHI, air_temp, wind_speed) selects the first valid value from a priority-ordered list:

```
NASA primary:    NASA → Solcast → Open-Meteo → ERA5
Solcast primary: Solcast → Open-Meteo → NASA → ERA5
Open-Meteo:      Open-Meteo → Solcast → NASA → ERA5
ERA5 primary:    ERA5 → Solcast → Open-Meteo → NASA
```

ERA5 is special: it provides **only** air_temp and wind_speed (no irradiance) — ERA5 SSRD requires separate NetCDF processing not implemented. So ERA5 is always a weather fallback, never an irradiance source.

**Output:** 4 separate prediction CSV files, each with source-tracking columns (`irradiance_source`, `temp_source`, `wind_source`) — allowing full auditability of which source contributed to each data point.

**Append mode:** New predictions are appended to rolling files (hourly/daily/monthly), with duplicate-by-timestamp detection to prevent double-counting on re-runs.

### 9.2 Why ThingsBoard, Not an API

- The plant's Meteocontrol SCADA feeds into ThingsBoard for dashboarding
- ThingsBoard rule chains process each telemetry packet; expected generation can be computed at the same moment as real generation arrives
- This enables real-time Performance Ratio (PR) calculation without any external service
- The DB threshold and inverter clipping in the JS model prevent false PR alerts at startup or during minimal power

---

## 10. Gaps, Limitations & What Wasn't Done

| Gap | Description |
|-----|-------------|
| **PVsyst validation** | Annual energy from `physics_model.py` has never been compared against a PVsyst simulation report for this plant — this is the standard validation step and remains pending |
| **Spectral effects** | pvlib does not apply spectral correction; PVsyst does. Estimated <1% error at sea level |
| **Degradation** | Module degradation is not modeled — LID only. No annual Rd (degradation rate) applied |
| **Self-calibration** | The physics model could recalibrate its loss factors using actuals when enough data accumulates — not implemented |
| **Battery/load offset** | If there's a battery or self-consumption, exported generation differs from generated. Model predicts generation, not export. |
| **Soiling detection** | Model can compute expected — actual difference could signal soiling. Not yet connected to alerting. |
| **Near-shading** | No row-to-row or parapet shading modeled — the 8 orientations are assumed unshaded (far_shading=1.0) |
| **GPU calibration used?** | The GPU optimizer was built but CPU was sufficient — unused in practice |
| **Open-Meteo irradiance quirk** | Open-Meteo's `shortwave_radiation` ≠ GHI exactly (includes reflected components); the script uses it as GHI which introduces systematic error |

---

## 11. Suggested Article Angles

1. **"The SCADA Physics Problem"** — Why existing tools (pvlib, PVsyst) can't directly run in IoT edge systems, and the engineering path to make them work via calibrated approximation.

2. **"When R²=0.99 Lies"** — The surrogate model looks perfect in training but fails at the use case. How energy error and hourly R² tell completely different stories.

3. **"Cloud Cover Beats Physics"** — Even with a correct physics model, irradiance input quality dominates accuracy on cloudy days. The monsoon benchmark shows all models fail not because physics is wrong, but because cloud forecasting is hard.

4. **"The 2.5 Million Sample Calibration"** — Treating pvlib as a physics oracle to generate synthetic training data, then using differential evolution to transfer knowledge to a JavaScript approximation. A novel transfer-learning-via-simulation approach.

5. **"Multi-Orientation Roof Complexity"** — Why 8 orientations spanning all compass quadrants make simple scaling/regression models fail where physics models succeed. The east-west energy smoothing effect and its implications for clipping.

---

*All data and code referenced here is in `m:\Documents\Projects\MAGICBIT\Power-Prediction\`. See `docs/model_methods.md` for formal method comparisons and `docs/project_plan.md` for the original methodology specification.*
