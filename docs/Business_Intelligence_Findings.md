# Irradiance to Generation

*Always up-to-date version: Business Intelligence Findings*

---

## 1. Extract Tilt/Azimuth Orientations & Module Counts from PVsyst + Plant Documentation

### a. Identify All Roof Planes From Plant Documentation

Each roof plane with a different:
- tilt,
- azimuth, or
- module direction

must be treated as a separate orientation.

### b. Extract Tilt(β) & Azimuth(γ) Per Orientation From PVsyst

For each orientation from Orientation and Shading Scene Description.

### c. Count Modules Per Roof Plane Using Documentation

For each roof plane:
- Identify all strings located on that plane
- Count modules = (modules per string) × (number of strings on that plane)
- Assign the plane to the PVsyst orientation with the closest tilt/azimuth

Verify Module Counts With PVsyst: `sum(modules_per_orientation) == total_modules_from_documentation`

### d. Compute Area Fractions

Each orientation contributes to plant power based on its share of total module area:

```
area_i = module_count_i × module_area_m²
area_fraction_i = area_i / total_area
```

---

## 2. Extract Solcast Irradiance and Weather Data (GHI, DNI, DHI, Air Temperature, Wind Speed)

### a. Required Solcast Parameters

```
ghi, dni, dhi, air_temp, wind_speed
```

### b. If subscription includes POA we extract:

```
ghi, dni, dhi, poa_global, air_temp, wind_speed
```

### c. POA Availability Note

Solcast POA is NOT available unless the user has a paid commercial plan. In case it's not provided, compute using: GHI, DNI, DHI, Sun Position, Perez Transposition.

### d. Period Matching

In the Solcast API Call, period must match the modeling timestep (30min or 60min). (`"period": "PT60M"`, or `"period": "PT30M"`). Use the same period as the PVsyst simulation or SCADA requirements. If PVsyst is simulated hourly, use PT60M. If finer resolution is needed, use PT30M. When computing energy, multiply power by the timestep duration (0.5h or 1h). Check that the Solcast period matches the modeling timestep.

### e. Timezone

Solcast timestamps are always in UTC. Convert them to local time zone with:

```python
df.index = df.index.tz_convert("Asia/Colombo")
```

### f. Shading Note

Solcast does NOT include optical/horizon/terrain shading. Therefore always apply PVsyst far-shading correction to POA before AOI/IAM.

### g. Data Quality

Always ensure GHI + DNI + DHI are non-negative and consistent.

### h. Solar Geometry

Compute solar geometry explicitly for every timestamp. For each timestamp t, compute solar zenith θz(t) and solar azimuth γs(t) from timestamp, latitude, longitude, and elevation. These angles are required inputs for:
- Perez POA transposition
- AOI computation
- Horizon / far-shading modeling (if time-dependent)
- IAM correction

---

## 3. POA Computation

Start with a time series of POA irradiance (plane-of-array). We MUST compute POA (independently for each orientation (tilt β, azimuth γ) before any area-weighting or aggregation is applied) ourselves from Solcast GHI/DNI/DHI because PVsyst does not provide time-series POA. We deliberately avoid Solcast PV power endpoints and use only irradiance + weather to preserve PVsyst-consistent physics and loss ordering.

### a. Plane-of-Array (POA) Computation Using Perez Transposition (If Solcast does not provide poa_global)

```
POA = POA_beam + POA_diffuse + POA_ground
POA_beam = DNI × cos(AOI)
POA_diffuse = DHI × F1 + GHI × F2
F1 = max(0, A + B·cos(θ_i) + C·θ_z)
F2 = D·sin(β) + E
θ_i = Angle of incidence (AOI)
θ_z = Solar zenith angle
β  = Panel tilt
A, B, C = Circumsolar weighting terms
D, E = Horizon brightening terms
```

Perez coefficients A, B, C, D, E are not user-defined constants. They are selected per timestamp from Perez sky clearness (ε) and brightness (Δ) bins, as implemented in pvlib and PVsyst.

### b. Ground-Reflected Component

```
POA_ground = GHI × ρ × (1 − cos β) / 2
```

ρ = ground albedo (from PVsyst)

### c. Apply Far-Shading / Horizon Loss Factors from PVsyst

Losses due to horizon profile obstructions (hills, trees, etc.) are modeled in PVsyst as Far Shading or Horizon Shading, and the annual impact is reported as a percentage loss. In the Orientation and Shading Scene Description page, we find the Far Shading Loss value. This loss must be applied before any optical modeling (AOI, IAM), because shading reduces the irradiance reaching the module surface.

### d. Define a shading factor

```
shading_factor = 1 − (far_shading_loss% / 100)
```

### e. Apply it to every timestamp

```
POA_shaded(t) = POA_raw(t) × shading_factor
```

### f. Notation

```
G_POA,shaded = POA after far-shading, before IAM
```

### g. Ordering Requirement

Always apply horizon / far-shading losses to POA irradiance BEFORE any further modeling steps (IAM, temperature, DC conversion). Solcast does not include terrain shading, therefore this correction must be applied manually using the PVsyst value. This ordering is mandatory:

```
POA → far-shading → IAM → temperature → DC → AC
```

Any deviation from this order will invalidate comparison with PVsyst.

### h. Measured Horizon File

If the plant has a measured horizon file, we can replace the flat far shading loss% loss with per-timestamp shading factors, and sun-position-dependent shading modeling.

---

## 4. Apply IAM

It corrects for optical losses at high incidence angles. Compute the incidence angle (AOI) and multiply POA by IAM(AOI).

### a. AOI Formula

```
AOI = arccos(cos(θ_z)·cos(β) + sin(θ_z)·sin(β)·cos(γ_s − γ_p))

θ_z = solar zenith angle
β   = panel tilt
γ_s = solar azimuth
γ_p = panel azimuth
```

Solar position → computed from timestamps  
Panel tilt and azimuth → extracted from PVsyst orientation section

### b. Extract IAM Table from PVsyst

On Array Losses section under IAM loss factor. This table is module-specific. Extract this column pair and interpolate:

```
IAM(AOI) = interp(AOI, AOI_table, IAM_table)
```

### c. Apply IAM

```
G_POA,optical = G_POA,shaded × IAM(θ_i)     [θ_i = AOI]
```

IAM is applied only after far-shading correction and before temperature and DC power modeling.

---

## 5. Compute Module Cell Temperature

Since it affects panel efficiency, our pipeline uses pvlib SAPM cell temperature. The following inputs are needed:
- POA (W/m²)
- Ambient temperature (°C)
- Wind speed (m/s), if available
- SAPM parameters (a, b, deltaT) for chosen mounting type

> **Important:** Cell temperature MUST be computed using POA before IAM. Using optically corrected POA would underestimate thermal losses and break consistency with PVsyst.

```
T_cell = f_SAPM(G_POA,shaded, T_amb, v_wind)
```

IAM-corrected POA (G_POA,optical) MUST NOT be used for thermal modeling.

---

## 6. Extract Thermal Parameters from PVsyst & Map Them to SAPM

### a. Module Model Parameters page — find:
- **Uc** = heat loss coefficient (W/m²·K)
- **Uv** = wind dependence term (W/m²·K per m/s)

### b. SAPM Mounting Selection

Since PVsyst does not provide SAPM parameters, we must choose a SAPM mounting type that best matches the physical installation shown in plant documents.

### c. Validation Requirement

PVsyst's thermal model (Uc/Uv) cannot be matched exactly using SAPM. SAPM Requires a Different Set of Parameters (a, b, ΔT). The chosen SAPM mounting type should approximate PVsyst's cooling behaviour. Always validate energy against PVsyst; adjust SAPM mounting choice if deviations >3–5%.

### d. Reference thermal models (for validation and substitution)

PVsyst's thermal model:
```
T_cell = T_amb + G_poa / (Uc + Uv · v_wind)
```

SAPM:
```
T_cell = T_amb + a · G_poa + b · G_poa · v_wind + ΔT
```

> Code hard-selects a SAPM preset and does NOT validate deviation vs PVsyst.

### e. Roof-Mounted Systems

For roof-mounted systems:
- `SAPM_MOUNT = 'close_mount_glass_glass'`
- or `SAPM_MOUNT = 'insulated_back_glass_glass'`

More context linked.

### f. Temperature Deviation Fallback

If SAPM produces temperature deviations >3%, consider replacing SAPM with a direct PVsyst-style cell temperature model using Uc and Uv from the simulation report:

```
T_cell = T_amb + G_poa / 20
```

---

## 7. Apply Module Power Temperature Coefficient

Module output decreases with cell temperature:

```
P(T) = P_STC · (1 + γ_P · (T_cell − 25))
η(T) = η_STC · (1 + γ_P · (T_cell − 25))
```

---

## 8. Compute DC Power per Timestamp (PVWatts-style)

Using optical-corrected POA and temperature-corrected efficiency:

```
P_DC,perm²(t) = G_POA,opt(t) · η_STC / 1000
P_DC,temp(t)  = P_DC,perm²(t) · (1 + γ_P · (T_cell(t) − 25))
A_coll         = total module area
G_POA,opt      = POA × IAM
η(T)           = temperature-adjusted module efficiency relative to η_STC
P_DC(t)        = G_POA,opt(t) × A_coll × η(T)
```

---

## 9. Apply DC-Side Array Losses (Extracted from PVsyst)

### a. Loss Application

PVsyst lists DC-side losses on the Detailed Losses / Array Loss Diagram page of the simulation report which represents annual average DC power reductions (e.g., Soiling loss, Light-Induced Degradation (LID), Module quality / mismatch, DC ohmic wiring loss):

```
P_DC,eff(t) = P_DC(t) × (1 − soiling) × (1 − LID) × (1 − mismatch) × (1 − wiring_DC)
```

Losses are applied sequentially on the DC side and must not be reordered or aggregated.

### b. Mandatory Loss Ordering

DC loss factors in PVsyst are not optional tuning parameters. They must be applied in the same order and on the DC side to reproduce PVsyst energy results.

### c. Sequential Application

DC losses are applied sequentially on DC power because PVsyst reports them as DC-side losses. Reordering these terms changes annual yield and is not permitted.

---

## 10. Convert DC → AC and Apply Inverter Losses

### a. Extract and Modeling Inverter Performance from PVsyst Inverter Parameters / Losses page:
- AC Nominal Power (e.g., 200 kW)
- Inverter efficiency curve (symbol η(Pdc))
- Threshold power (Pmin or Pthresh)
- Max output power (clipping limit)
- MPPT voltage range
- Wiring and AC losses
- AC rating (kW)
- Efficiency (%)
- AC wiring loss (%)
- Inverter loss (%)

### b. Starting Point

Start with effective DC power after all DC-side losses.

### c. Apply inverter conversion loss

```
P_AC,raw = P_DC,eff × (1 − inverter_loss)
```

### d. Apply AC wiring losses

```
P_AC,nom = P_AC,raw × (1 − AC_wiring_loss)
```

### e. Plant-Level Aggregation and Clipping

Sum AC power from all orientations first. Then apply inverter clipping once at the plant level. (Inverter clipping must occur after orientation aggregation):

```
P_AC,plant(t) = min(P_AC,nom,plant(t), P_inv,AC,rating)
```

Per-orientation clipping is forbidden; clipping must occur only after full plant-level aggregation.

### f. Clipping Rule

Clipping must occur after summing all orientations because PVsyst clips at the inverter terminals, not per sub-array.

### g. Inverter Efficiency Model

We currently have P_AC = P_DC × η. But PVsyst uses a curved efficiency model, not a single value. So we can use a BI-SCADA friendly approximation for plants where partial-load behavior matters; but we must extract the inverter efficiency curve from PVsyst and implement a load-dependent efficiency model:

```
η_inv = 0.97 (nominal)
```

### h. Efficiency Curve

If an inverter efficiency curve is available from PVsyst, η_inv is evaluated as a function of DC loading. If not available, a constant nominal efficiency may be used as a BI-SCADA approximation.

---

## 11. Aggregate All Orientations

For each orientation i (area fractions are based on module counts or equal split):

```
P_AC,plant(t) = Σ_i P_AC,i(t)
```

---

## 12. Integrate Over Time to Get Total Energy

(timestep = Δt hours)

```
E_AC,model = Σ_t P_AC,plant(t) × Δt_hours
```

---

## 13. Compare to PVsyst Annual Output

The pipeline should match closely (±2–3%) if the same weather data is used.

### a. Extract Key Reference Values From PVsyst:
- Annual Effective Irradiation on Plane of Array (kWh/m²)
- Annual DC Energy (Earray)
- Annual AC Energy (E_Grid)
- Performance Ratio (PR)

### b. POA Comparison

After generating modeled POA time series and applying horizon loss, IAM, and all transposition steps, compute:

```
H_POA,model = Σ POA(t) × Δt
```

Compare against PVsyst's H_POA,PVsyst. Acceptable variance: ±2–3%.

### c. If modeled POA differs by >5%, the issue is likely:
- missing diffuse calculation
- incorrect tilt/azimuth mapping
- incorrect Solcast timezone
- missing far-shading adjustment

### d. AC Energy Comparison

Compare Annual AC Energy E_AC,model to PVsyst's E_Grid. Acceptable error band: ±2–5%.

### e. If >5% deviation, the most common causes are:
- incorrect SAPM thermal model
- incorrect DC loss factors
- clipping performed per-orientation instead of plant-level
- incorrect inverter efficiency assumption
- shaded POA not applied
- wrong module area allocation (orientation mapping)

### f. Performance Ratio (PR)

Compute PR with:

```
PR_model = E_AC,model / (H_POA,model × P_rated)
```

Compare with PVsyst PR. Acceptable variance: ±2–3%.

### g. Larger differences indicate that:
- a loss factor is missing
- AC or DC loss ordering is wrong
- inverter efficiency is not applied correctly
- module temperature is underestimated/overestimated

### h. Validation Requirement

Always validate the modeled annual AC energy against PVsyst before deploying the model. A mismatch within ±2–5% is acceptable for hourly-resolution forecasts. If mismatch exceeds 5%, revisit the thermal model, irradiance preprocessing, orientation fractions, or inverter clipping implementation.

### i. Checklist

| Check | Criterion |
|---|---|
| **POA Check** | Does modeled POA (kWh/m²) match PVsyst within ±3%? |
| **DC Energy Check** | Does modeled DC energy match PVsyst EArray within ±3–5%? |
| **AC Energy Check** | Does final AC energy match PVsyst E_Grid within ±5%? |
| **PR Check** | Does modeled PR match PVsyst PR within ±3%? |
| **Clipping Check** | Is clipping applied once at plant level? |
| **Thermal Check** | Is SAPM or PVsyst temperature model correctly used? |
| **Loss Factor Ordering** | Are soiling → LID → mismatch → DC wiring applied sequentially? |

### j. Validation Importance

Without validation against PVsyst, the methodology cannot be trusted. PV modeling is sensitive to temperature, irradiance transposition, and inverter behavior. Even a small mistake can cause large (>10%) yield drift. Validation ensures the pipeline is reliable for SCADA dashboards and energy forecasts.

---

## Important Notes & Cautions

- **Timestamps/timezones:** Solcast returns UTC. The code keeps internal calculations in UTC; it converts to local timezone only when printing. Always align SCADA timestamps to UTC before comparing.
- **Area assignment:** To reproduce PVsyst closely, set `module_area_m2` and per-orientation `module_count` to the exact values used in PVsyst. Equal splits are approximate.
- **Losses are static:** Soiling, LID, mismatch, wiring are applied as fixed fractions in the script. If you have seasonal or event-based soiling data, model this dynamically.
- **Validation:** Always validate monthly totals against PVsyst monthly table when using the TMY that PVsyst used. If differences > ~5%, check IAM curve, area split, NOCT/SAPM choice, inverter mapping, or the weather file.
- **Resolution tradeoffs:** Hourly data is usually fine for annual energy. Use 15-min for generation forecasting or grid services; remember to adjust `period` and dt handling.

---

## Implementation Summary (Code–Method Mapping)

This implementation summary exists to clarify how the theoretical methodology is realized in code.

It does not introduce new modeling logic; it maps each conceptual step to its concrete computational implementation in the Python pipeline.

Validation against PVsyst annual and monthly results remains mandatory before operational use.

### 1) Weather ingestion & timestamp normalization

- The implementation fetches irradiance and weather data from Solcast using the `radiation_and_weather` endpoint at the requested temporal resolution (e.g., PT60M).
- Solcast timestamps (`period_end` / `period_start`) are parsed into a timezone-aware UTC pandas index. All internal calculations are performed in UTC; conversion to local time is done only for reporting or visualization.

### 2) Solar geometry (sun position)

- For each timestamp, solar zenith and solar azimuth angles are computed using `pvlib.location.Location.get_solarposition`.
- These angles are used consistently across POA transposition, AOI calculation, and shading logic.

### 3) POA transposition (Perez diffuse model)

- For each array orientation (tilt β, azimuth γ), plane-of-array irradiance is computed using `pvlib.irradiance.get_total_irradiance` with the Perez diffuse model.
- The implementation internally combines beam, diffuse, and ground-reflected components to obtain `poa_global`, matching the transposition approach used by PVsyst.

### 4) Angle of incidence (AOI) and IAM correction

- The angle of incidence (AOI) is computed using pvlib's geometric formulation.
- The module-specific IAM table extracted from PVsyst is interpolated as a function of AOI, and the resulting IAM factor is applied to POA irradiance to obtain the optically effective POA (`poa_optical`).
- This represents the irradiance actually absorbed by the module after angular reflection losses.

### 5) Cell temperature (SAPM thermal model)

- Module cell temperature is computed using pvlib's SAPM thermal model with mounting parameters selected to approximate the PVsyst Uc/Uv thermal behavior.
- The implementation intentionally uses POA before IAM for temperature calculations, matching PVsyst's convention and avoiding underestimation of thermal losses at high AOI.

### 6) DC power computation (area-based, temperature-corrected)

- Instantaneous DC power is computed per unit module area using optically corrected POA, STC efficiency, and the module power temperature coefficient.
- This yields temperature-corrected DC power density, which is later scaled by total module area.

### 7) DC-side array losses

- DC-side losses (soiling, LID, mismatch, DC wiring) are applied as a multiplicative loss chain using annual loss fractions extracted from the PVsyst loss diagram.
- Loss ordering is preserved to match PVsyst's DC-side energy accounting.

### 8) DC → AC nominal (apply inverter/AC losses fraction)

- Effective DC power is converted to nominal AC power using a fixed inverter + AC loss fraction derived from PVsyst.
- In the production script, inverter modeling and clipping are handled explicitly at the plant level rather than per orientation.

### 9) Orientation weighting & aggregation

- Each orientation's contribution is weighted by its share of total module area, derived from module counts and module area.
- Per-orientation AC power is scaled accordingly and summed to obtain total plant-level AC power before clipping.

### 10) Inverter clipping at plant level

- Plant-level AC power is clipped once against the installed inverter AC rating.
- This reflects PVsyst's system-level clipping behavior and avoids per-orientation clipping artifacts.
- Optional extensions include replacing flat efficiency with inverter efficiency curves if detailed inverter maps are available.

### 11) Energy integration (kWh) & outputs

- Energy is computed by integrating plant-level AC power over time using the inferred timestep duration.
- The implementation outputs a time-series CSV of AC power and reports total energy for the modeled period.

---

## ThingsBoard Rule Chain Integration

This project provides two implementations of the same irradiance-to-generation methodology, designed for different operational needs.

### 1. Python Reference Model (local-poa.py)

The Python implementation is the authoritative, engineering-grade reference:

- Uses pvlib for:
  - solar position
  - Perez POA transposition
  - AOI / IAM handling
  - SAPM temperature modeling
- Intended for:
  - validation against PVsyst
  - offline analysis
  - batch forecasting
  - model tuning and calibration

This model prioritizes physical accuracy and traceability, and should always be used as the reference when validating results.

### 2. ThingsBoard Rule Chain Model (local-poa.js)

The JavaScript implementation (`local-poa.js`) is a ThingsBoard rule-chain-ready operational estimator.

Key characteristics:
- Designed to run inside ThingsBoard's JavaScript execution environment
- Uses a simplified but physics-consistent version of the methodology
- Optimized for:
  - real-time dashboards
  - SCADA KPIs
  - operational alerts
- Trades some physical fidelity for:
  - execution speed
  - deterministic runtime
  - scalability inside rule chains

This version must not be treated as a PVsyst replacement. It is an estimator for operational use, not a simulation engine.

### 3. Hybrid Architecture (Recommended)

In production deployments, a hybrid approach is recommended:

- **ThingsBoard (`local-poa.js`)**
  - Runs continuously
  - Produces real-time estimates
  - Feeds dashboards and alarms
- **Hosted Python Service (`local-poa.py`)**
  - Runs periodically (hourly / daily)
  - Produces high-accuracy reference values
  - Used for validation, reconciliation, and bias correction

The ThingsBoard rule chain can optionally call the hosted Python service via REST when higher accuracy is required (e.g., forecasts, engineering views), but this should not be done for every telemetry point due to latency and scalability constraints.

### 4. Design Rationale

This separation is intentional:
- ThingsBoard rule chains are optimized for fast, lightweight execution
- Python + pvlib is optimized for accurate physical modeling
- Combining both directly inside rule chains is not feasible

By keeping the methodology identical and separating only the execution environment, the system achieves both accuracy and operational robustness.

---

## Computation

### Transpose locally from GHI/DNI/DHI

```python
import requests
import pandas as pd
import numpy as np
import pvlib
from pvlib.location import Location
import sys
import os
from datetime import datetime, timedelta, UTC

API_KEY = "YOUR_SOLCAST_API_KEY"
lat, lon = 8.342368984714714, 80.37623529556957
period = "PT60M"
TIMEZONE = "Asia/Colombo"

orientations = [
    {"tilt": 18, "azimuth": 148,  "name": "O1", "module_count": 18},
    {"tilt": 18, "azimuth": -32,  "name": "O2", "module_count": 18},
    {"tilt": 19, "azimuth": 55,   "name": "O3", "module_count": 36},
    {"tilt": 19, "azimuth": -125, "name": "O4", "module_count": 36},
    {"tilt": 18, "azimuth": -125, "name": "O5", "module_count": 36},
    {"tilt": 18, "azimuth": 55,   "name": "O6", "module_count": 36},
    {"tilt": 27, "azimuth": -125, "name": "O7", "module_count": 18},
    {"tilt": 27, "azimuth": 55,   "name": "O8", "module_count": 18},
]

module_area = 2.556
total_module_area = sum(o["module_count"] * module_area for o in orientations)
module_efficiency_stc = 0.2153
gamma_p = -0.00340
INV_AC_RATING_KW = 55.0
albedo = 0.20
PDC_THRESHOLD_KW = 0.0

soiling = 0.03
LID = 0.014
module_quality = -0.008
mismatch = 0.017
dc_wiring = 0.009
dc_loss_factor = (1 - soiling) * (1 - LID) * (1 - module_quality) * (1 - mismatch) * (1 - dc_wiring)

ac_wiring_loss = 0.003

iam_angles = np.array([0, 25, 45, 60, 65, 70, 75, 80, 90])
iam_values = np.array([1.000, 1.000, 0.995, 0.962, 0.936, 0.903, 0.851, 0.754, 0.000])

DEFAULT_WIND_SPEED_MS = 1.0
DEFAULT_AIR_TEMP_C = 25.0
SITE_ALTITUDE_M = 88

sapm_params = pvlib.temperature.TEMPERATURE_MODEL_PARAMETERS["sapm"]["close_mount_glass_glass"]

USE_INVERTER_CURVE = True
inverter_eff_curve_kw = np.array([
    2.43, 4.74, 4.99, 7.18, 9.99, 12.23, 14.92, 17.42, 20.04, 22.28, 24.91,
    27.53, 30.02, 32.46, 34.96, 37.52, 40.07, 42.57, 45.19, 47.50, 50.00
])
inverter_eff_curve_eta = np.array([
    0.95238, 0.97151, 0.97270, 0.97724, 0.98016, 0.98168, 0.98222, 0.98254,
    0.98276, 0.98286, 0.98276, 0.98254, 0.98222, 0.98189, 0.98146, 0.98103,
    0.98059, 0.98016, 0.97973, 0.97930, 0.97876
])

start_date = "2025-12-10T00:00:00Z"
end_date   = "2025-12-16T23:00:00Z"

def inverter_efficiency(pdc_kw):
    if USE_INVERTER_CURVE:
        return np.interp(np.asarray(pdc_kw), inverter_eff_curve_kw, inverter_eff_curve_eta)
    else:
        return 0.98

# print("Fetching Solcast data…")
# raw = fetch_solcast(lat, lon, period, API_KEY)
# if "forecasts" not in raw and "data" not in raw and "estimated_actuals" not in raw:
#     print("\n⚠️ API ERROR RECEIVED:")
#     print(raw)
#     sys.exit(1)
# df = solcast_to_df(raw)

def fetch_solcast(lat, lon, period, api_key):
    url = "https://api.solcast.com.au/world_radiation/estimated_actuals"
    params = {
        "latitude": lat,
        "longitude": lon,
        "start": start_date,
        "end": end_date,
        "period": period,
        "api_key": api_key,
        "output_parameters": "ghi,dni,dhi",
        "format": "json"
    }
    print(f"Requesting estimated actuals from {start_date} to {end_date}...")
    r = requests.get(url, params=params)
    if r.status_code != 200:
        print("\n⚠️ API ERROR:", r.text)
        sys.exit(1)
    return r.json()

def solcast_to_df(j):
    if "estimated_actuals" in j:
        rows = j["estimated_actuals"]
    elif "forecasts" in j:
        rows = j["forecasts"]
    elif "data" in j:
        rows = j["data"]
    else:
        raise ValueError(f"Solcast API did not return valid data. Keys found: {j.keys()}")
    df = pd.DataFrame(rows)
    idx = pd.to_datetime(df["period_end"], utc=True)
    df.index = pd.DatetimeIndex(idx)
    return df

def fetch_era5_weather(lat, lon, start_date_str, end_date_str):
    try:
        import cdsapi
        import xarray as xr
        start_dt = pd.to_datetime(start_date_str).tz_localize(None)
        end_dt   = pd.to_datetime(end_date_str).tz_localize(None)
        OUTPUT_NETCDF = "era5_weather_temp.nc"
        c = cdsapi.Client()
        if start_dt.month != end_dt.month:
            raise RuntimeError("ERA5 fetch currently supports single-month ranges only")
        year  = str(start_dt.year)
        month = f"{start_dt.month:02d}"
        days  = []
        current = start_dt
        while current <= end_dt:
            days.append(f"{current.day:02d}")
            current += timedelta(days=1)
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
                "area": [lat + 0.1, lon - 0.1, lat - 0.1, lon + 0.1],
                "format": "netcdf",
            },
            OUTPUT_NETCDF,
        )
        ds = xr.open_dataset(OUTPUT_NETCDF)
        df = ds.to_dataframe().reset_index()
        time_col = "time" if "time" in df.columns else "valid_time"
        df = df.groupby(time_col, as_index=False).mean(numeric_only=True)
        df["air_temp"]   = df["t2m"] - 273.15
        df["wind_speed"] = np.sqrt(df["u10"]**2 + df["v10"]**2)
        weather_df = (
            df[[time_col, "air_temp", "wind_speed"]]
            .rename(columns={time_col: "time"})
            .set_index("time")
            .sort_index()
        )
        weather_df.index = pd.to_datetime(weather_df.index, utc=True)
        if os.path.exists(OUTPUT_NETCDF):
            os.remove(OUTPUT_NETCDF)
        return weather_df
    except Exception:
        return None

def compute_pv_ac(df_weather):
    site = Location(lat, lon, altitude=SITE_ALTITUDE_M if SITE_ALTITUDE_M is not None else 0)
    solpos = site.get_solarposition(df_weather.index)
    plant_ac = pd.Series(0, index=df_weather.index)
    dni_extra = pvlib.irradiance.get_extra_radiation(df_weather.index)

    for o in orientations:
        tilt     = o["tilt"]
        azimuth  = o["azimuth"]
        area_fraction = (o["module_count"] * module_area) / total_module_area

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

        aoi = pvlib.irradiance.aoi(tilt, azimuth, solpos["zenith"], solpos["azimuth"])
        iam = np.interp(aoi, iam_angles, iam_values)
        poa_optical = poa * iam

        cell_temp = pvlib.temperature.sapm_cell(
            poa,
            df_weather["air_temp"],
            df_weather["wind_speed"],
            **sapm_params
        )

        pdc_kwm2      = poa_optical * module_efficiency_stc / 1000
        pdc_kwm2_temp = pdc_kwm2 * (1 + gamma_p * (cell_temp - 25))
        pdc_kwm2_eff  = pdc_kwm2_temp * dc_loss_factor

        area_i       = (total_module_area * area_fraction)
        pdc_total_kw = pdc_kwm2_eff * area_i

        if PDC_THRESHOLD_KW > 0:
            pdc_total_kw = pdc_total_kw.where(pdc_total_kw >= PDC_THRESHOLD_KW, 0.0)

        pac_kw    = pdc_total_kw * inverter_efficiency(pdc_total_kw)
        plant_ac += pac_kw

    plant_ac = plant_ac.clip(upper=INV_AC_RATING_KW)
    plant_ac = plant_ac * (1 - ac_wiring_loss)
    return plant_ac

CSV_PATH = r"path\to\.csv"
print("Loading irradiance data from CSV...")
df = pd.read_csv(CSV_PATH)
time_col_candidates = ["time", "timestamp", "period_end", "datetime"]
for col in time_col_candidates:
    if col in df.columns:
        df[col] = pd.to_datetime(df[col], utc=True)
        df = df.set_index(col)
        break
else:
    raise RuntimeError("No valid timestamp column found in CSV")

df = df.sort_index()
required_cols = ["ghi", "dni", "dhi"]
missing = [c for c in required_cols if c not in df.columns]
if missing:
    raise RuntimeError(f"CSV missing required irradiance columns: {missing}")

if "wind_speed" not in df.columns and "wind_speed_10m" in df.columns:
    df["wind_speed"] = df["wind_speed_10m"]

missing_temp = "air_temp" not in df.columns
missing_wind = "wind_speed" not in df.columns

if missing_temp or missing_wind:
    era5_weather = fetch_era5_weather(lat, lon, start_date, end_date)
    if era5_weather is not None:
        df = df.join(era5_weather, how="left")
        df["air_temp"]   = df["air_temp"].fillna(method="ffill")
        df["wind_speed"] = df["wind_speed"].fillna(method="ffill")
    else:
        if missing_temp:
            df["air_temp"]   = DEFAULT_AIR_TEMP_C
        if missing_wind:
            df["wind_speed"] = DEFAULT_WIND_SPEED_MS

if not df.index.is_monotonic_increasing:
    df = df.sort_index()

EXPECTED_FREQ = "1H"
if pd.infer_freq(df.index) != EXPECTED_FREQ:
    df = df.resample(EXPECTED_FREQ).nearest()

print("Computing PV AC output…")
plant_ac = compute_pv_ac(df)
dt_hours = (df.index[1] - df.index[0]).total_seconds() / 3600
plant_ac_local = plant_ac.tz_convert(TIMEZONE)
energy = (plant_ac_local * dt_hours).sum()
out = pd.DataFrame({"AC_kW": plant_ac_local})
out.to_csv("pv_generation_output.csv")
print(f"Total energy (kWh): {energy:.1f}")
```

File can be found at: `local-poa.py`  
ThingsBoard Rule Chain ready version: `local-poa.js`

---

### Inputs

| Parameter | Description |
|---|---|
| `API_KEY` | Solcast API key |
| `lat / lon` | Site coordinates of the solar plant |
| `period` | Time resolution (ISO-8601). Example: PT60M = 60 minutes, PT15M = 15 minutes |
| `TIMEZONE` | Local timezone used for reporting (e.g., Asia/Colombo) |
| `orientations` | List of array orientations. Each item contains: `tilt` (panel tilt in degrees), `azimuth` (panel azimuth: 0 = north, 90 = east, 180 = south, -90 = west), `name` (identifier for the orientation), `module_count` (number of modules in that orientation, used to compute area share) |
| `module_area_m2` | Area of one PV module in m² |
| `area_total_m2` | Total collector area of the plant (sum of all module areas) |
| `module_eff_stc` | Module efficiency at STC (decimal) |
| `module_gamma_p` | Power temperature coefficient (/°C) |
| `INVERTER_AC_RATING_KW` | Total inverter AC capacity (kW) |
| `INVERTER_EFFICIENCY` | Nominal inverter efficiency (decimal) |
| `soiling` | Soiling loss fraction |
| `LID` | Light-induced degradation fraction |
| `mismatch` | Module mismatch loss fraction |
| `wiring_dc` | DC wiring loss fraction |
| `inv_and_ac_losses_fraction` | Inverter + AC wiring losses (fraction) |
| `iam_angles` | Incidence angle breakpoints (degrees) |
| `iam_values` | IAM values corresponding to the angles |
| `SAPM_MOUNT` | SAPM thermal model mounting type (e.g., open_rack_cell_glassback) |

### Example Response

```
Fetching Solcast forecast (GHI/DNI/DHI/air_temp [wind_speed optional]) ...
Computing POA for: Orient1 tilt: 16 az: -120
Computing POA for: Orient2 tilt: 16 az: 60
...
Saved hourly plant AC power to: plant_pac_hourly_accurate.csv
Total energy (kWh) for fetched period: 318870.4

Sample hourly output (local timezone):
                           P_ac_kW
2025-01-01 00:00:00+05:30    0.000
2025-01-01 01:00:00+05:30    0.000
2025-01-01 02:00:00+05:30    0.000
2025-01-01 03:00:00+05:30    0.000
2025-01-01 04:00:00+05:30    0.000
2025-01-01 05:00:00+05:30    0.015
2025-01-01 06:00:00+05:30    1.102
2025-01-01 07:00:00+05:30    9.845
2025-01-01 08:00:00+05:30   35.241
2025-01-01 09:00:00+05:30   78.102
...
```

### Important Notes & Cautions

- **Mounting & thermal params:** Choose the SAPM mounting parameters that best match your install (roof, open rack, glass-back, etc.). Wrong mounting type changes cell temp and yield by several percent.
- **Inverter modeling:** The code uses plant AC clipping at `INVERTER_AC_RATING_KW` and a flat inverter loss fraction. For more accuracy (clipping behavior and partial-load efficiency) provide inverter efficiency curves or Sandia inverter model parameters.
- **Spectral / advanced optical effects:** PVsyst includes some additional corrections (spectral, glass, detailed shading). The pvlib+SAPM approach reproduces most effects but may differ slightly (<1–2%) on edge cases. Use PVsyst if you need its full set of optical models.

---

## Terminology

| Term | Explanation |
|---|---|
| **Azimuth (Angle)** | The horizontal compass direction the PV modules face, measured in degrees (typically from true north). It determines when during the day the array receives maximum irradiance and is a key input for transposing GHI/DNI/DHI into POA. |
| **Solar Zenith Angle** | The angle between the sun and the vertical direction at a given location and time. It defines how high the sun is in the sky and directly affects DNI projection, AOI, and POA irradiance calculations in PV performance models. |
| **Solar Azimuth Angle** | The compass direction of the sun at a given time, measured on the horizontal plane (from true north or south, by convention). Used with solar zenith to determine AOI, shading, and POA irradiance relative to the array's azimuth. |
| **Horizon Modeling** | A geometric method to represent terrain and distant obstacle profiles around a PV site. It determines when the sun is blocked based on solar azimuth and zenith, enabling accurate far-shading losses in POA and power calculations. |
| **POA irradiance (Plane-of-Array)** | Solar irradiance measured on the same tilt and orientation as the PV modules. It represents the actual sunlight hitting the panels, making it the most relevant irradiance input for forecasting energy yield and evaluating performance ratios. |
| **GHI (Global Horizontal Irradiance)** | Total solar irradiance received on a horizontal surface. It includes direct + diffuse sunlight and is commonly used as the base irradiance input for PV modeling before converting to POA using tilt/orientation. |
| **DNI (Direct Normal Irradiance)** | Sunlight coming directly from the sun, measured on a surface perpendicular to the sun's rays. Critical for modeling tracking systems and decomposing irradiance components. |
| **DHI (Diffuse Horizontal Irradiance)** | Sunlight scattered by the atmosphere, measured on a horizontal surface. Used along with DNI to compute GHI and to estimate POA under cloudy or partially shaded conditions. |
| **GlobInc** (cannot replace POA time series) | Global incident irradiance on the tilted plane → POA BEFORE losses |
| **GlobEff** (cannot replace POA time series) | Effective POA after IAM + soiling + some shading corrections |
| **STC (Standard Test Conditions)** | The fixed laboratory conditions used to rate PV module performance: Irradiance: 1000 W/m², Cell temperature: 25°C, Air mass: 1.5 |
| **IAM (Incidence Angle Modifier)** | A correction factor that adjusts irradiance based on the angle between sunlight and the module surface. At steep angles (morning/evening), more light is reflected instead of absorbed, so IAM reduces the effective POA irradiance. |
| **AOI (Angle of Incidence)** | The angle between the incoming sunlight and the perpendicular (normal) to the PV module surface. Higher AOI (more oblique light) → more reflection → lower effective irradiance. It's a key input for IAM corrections. |
| **Module Cell Temperature** | The actual operating temperature of the PV cells, not the ambient air. It strongly affects power output: hotter cells → lower efficiency. Calculated using irradiance, ambient temperature, wind, and module thermal coefficients (e.g., NOCT model). |
| **SAPM (Sandia Array Performance Model)** | A detailed PV performance model developed by Sandia that uses experiment-based coefficients to predict real-world module output. It accounts for irradiance, cell temperature, angle-of-incidence effects, spectral shifts, and module-specific behavior—making it more accurate than simple STC/NOCT-based models for SCADA BI calculations. |
| **Ambient Temperature** | The air temperature surrounding the PV array. Used as an input (along with irradiance and wind) to estimate module cell temperature, which directly impacts power output. |
| **Module Power Temperature Coefficient** | The rate at which a PV module's power output drops for each °C increase in cell temperature above 25°C (STC). Example: −0.36%/°C. Higher temperature → lower power. This coefficient is essential for correcting DC power predictions in BI/SCADA models. |
| **DC Power per Timestamp (PVWatts-style)** | The instantaneous DC output of the PV array calculated at each time step using: POA irradiance, Module temperature, Nameplate capacity, Temperature coefficient. PVWatts uses a simplified formula to estimate DC power under real-world conditions, making it easy to integrate into SCADA BI pipelines. |
| **Total Module Area** | The combined surface area of all PV modules in the array (length × width × number of modules). Used to compute area-based metrics like W/m², efficiency, and irradiance-to-power conversion performance. |
| **Far-Shading Correction** | A loss factor that reduces irradiance or power to account for distant obstacles (terrain, buildings, trees) that block the sun at certain solar angles. Unlike near-shading, it is usually angle-based and time-dependent, applied during POA or power calculation in SCADA/BI models. |
| **Ground Albedo** | The fraction of sunlight reflected by the ground surface (0–1). It contributes to the reflected irradiance component of POA, especially for tilted arrays and high-albedo surfaces (sand, concrete, snow). |
| **Optical Modeling** | The simulation of how sunlight interacts with the PV module surface (reflection, refraction, absorption). Used to adjust POA via AOI, IAM, glass properties, and soiling, improving accuracy of irradiance-to-power calculations in SCADA/BI systems. |
| **Panel Azimuth** | The compass direction the PV modules face on the horizontal plane. It defines the array's orientation relative to the sun and is used with solar azimuth to compute AOI and POA irradiance. |
| **Interpolate** | Estimate values at missing or finer time steps by using surrounding data points (e.g., irradiance, temperature). In SCADA/BI pipelines, interpolation is used to align weather data with power timestamps without altering underlying trends. |
| **Cell Temperature** | The actual temperature of the PV cells inside the module. It directly affects voltage and power output and is a key input for temperature-derated DC power calculations in SCADA BI models. |
| **Soiling Loss** | The reduction in effective irradiance or power caused by dust, dirt, or pollutants on the module surface. Modeled as a percentage loss factor applied to POA or DC power in SCADA/BI performance calculations. |
| **Light-Induced Degradation (LID)** | The initial, irreversible drop in module power that occurs after first exposure to sunlight. Typically a small percentage loss (≈1–3%) applied as a fixed derate in long-term performance and SCADA BI baseline models. |
| **AC Nominal Power** | The rated maximum AC output of the PV system at the inverter output under specified conditions. It defines the system's AC capacity, is used for clipping analysis, and serves as the reference for AC-based KPIs in SCADA dashboards. |
| **Threshold Power (Pmin / Pthresh)** | The minimum DC input power required for the inverter to turn on and start producing AC power. Below this level, output is zero (used to filter low-irradiance noise in BI metrics). |
| **Max Output Power (Clipping Limit)** | The upper AC power cap imposed by the inverter. Any DC power above this limit is clipped, causing lost energy during high-irradiance periods. |
| **MPPT Voltage Range** | The DC voltage window within which the inverter can track the maximum power point. Operation outside this range results in reduced power or inverter shutdown. |
| **AC Rating (kW)** | The inverter's nameplate continuous AC power capacity. Used as the reference for AC nominal power, clipping detection, and capacity-based KPIs in SCADA BI systems. |
| **apply inverter clipping once at the plant level** | Sum all DC power first, then cap the result at the plant's AC rating one time, to avoid double-counting clipping losses. |
| **Per-Orientation Clipping** | Applying inverter clipping separately for each array orientation (e.g., east, west, south) before aggregating power. Used when different orientations are connected to different inverters or MPPTs, to model orientation-specific clipping behavior accurately in SCADA BI calculations. |
| **Partial-Load Behavior** | How an inverter performs below its rated power, especially at low irradiance. It captures efficiency drop-off, startup thresholds, and non-linear response, which affects early morning, evening, and cloudy-period power accuracy in SCADA BI models. |
| **Load-Dependent Efficiency Model** | An inverter efficiency representation where efficiency varies with output power (load fraction). Used to convert DC → AC power more accurately across low, medium, and high load conditions instead of assuming constant efficiency. |
| **Annual Effective Irradiation on Plane of Array** | The yearly sum of usable solar energy incident on the module plane, after applying losses such as IAM, soiling, shading, and horizon effects. It represents the actual irradiance available for energy production and is a key normalization metric for annual yield and performance KPIs. |

---

## pvlib SAPM Mounting Presets

### `open_rack_glass_glass` / `open_rack_glass_back`
- Modules mounted on open racks with strong rear ventilation (ground-mount or open structures).
- Best cooling, lowest cell temps.

**Use for:**
- ✔ Ground-mounted arrays
- ✔ Carports
- ✔ Well-ventilated rooftop racks with significant air gap

### `close_mount_glass_glass` / `close_mount_glass_back`
- Modules mounted close to a surface but not sealed, with limited airflow.
- Warmer than open rack but not the worst.

**Use for:**
- ✔ Roof-mounted systems with ~10–20 cm rear gap
- ✔ Metal sheet roofs with rails
- ✔ Typical residential rooftops not fully insulated behind

### `insulated_back_glass_back`
- Modules with very little or no rear ventilation—the back is insulated or very close to a hot surface.
- Highest cell temps.

**Use for:**
- ✔ Building-integrated PV (BIPV)
- ✔ Modules mounted directly on walls or roofs
- ✔ Installations with foam insulation or sealed backs
