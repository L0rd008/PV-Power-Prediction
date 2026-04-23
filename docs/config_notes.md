# =====================================================================
# 1) USER INPUTS (Solcast + PVsyst + Plant Geometry): MUST ADJUST
# =====================================================================

API_KEY = "YOUR_SOLCAST_API_KEY"
lat, lon = 8.342368984714714, 80.37623529556957
period = "PT60M"
TIMEZONE = "Asia/Colombo"

# Orientation list extracted from PVsyst PDF
orientations = [
    {"tilt": 18, "azimuth": 148, "name": "O1", "module_count": 18},
    {"tilt": 18, "azimuth": -32,   "name": "O2", "module_count": 18},
    {"tilt": 19, "azimuth": 55, "name": "O3", "module_count": 36},
    {"tilt": 19, "azimuth": -125,   "name": "O4", "module_count": 36},
    {"tilt": 18, "azimuth": -125,  "name": "O5", "module_count": 36},
    {"tilt": 18, "azimuth": 55,  "name": "O6", "module_count": 36},
    {"tilt": 27, "azimuth": -125, "name": "O7", "module_count": 18},
    {"tilt": 27, "azimuth": 55,   "name": "O8", "module_count": 18},
]

module_area = 2.556    # m² per module
total_module_area = sum(o["module_count"] * module_area for o in orientations)

module_efficiency_stc = 0.2153
gamma_p = -0.003       # per °C: This is a placeholder value, get the real value from module datasheet 

INV_AC_RATING = 100_000  # W Pnom total
far_shading_factor = 0.982   # PVsyst = 1-(IAM factor on global)%/100
albedo = 0.20                # PVsyst value

# Inverter DC threshold power (Pmin / Pthresh)
# NOTE: PVsyst simulation report does not expose numeric Pmin.
# Value below is a conservative placeholder for BI/SCADA realism.
# Replace with value from PVsyst inverter component (.OND) or datasheet.
PDC_THRESHOLD_KW = 1.0

# PVsyst DC losses (multiplicative)
soiling = 0.03
LID = 0.015
mismatch = 0.009
dc_wiring = 0.014
dc_loss_factor = (1 - soiling)*(1 - LID)*(1 - mismatch)*(1 - dc_wiring)

# IAM VALUES FROM PVsyst
iam_angles = np.array([0, 25, 45, 60, 65, 70, 75, 80, 90])
iam_values = np.array([1.000, 1.000, 0.995, 0.962, 0.936, 0.903, 0.851, 0.754, 0.000])

# Default wind speed used ONLY if Solcast does not provide wind_speed
# Typical calm-wind assumption: 1–2 m/s
# PVsyst implicitly assumes similar values in absence of detailed data
DEFAULT_WIND_SPEED_MS = 1.0

# Site altitude above mean sea level (meters)
# OPTIONAL INPUT:
# - PVsyst uses altitude in air-mass and DNI calculations.
# - Extract from PVsyst report → "Geographical Site" section.
# - If not provided (None), model falls back to sea level (0 m),
#   which is acceptable for low-altitude sites but may introduce
#   small DNI errors for high-altitude plants.
# NOTE ON ALTITUDE:
# PVsyst includes site altitude in solar geometry and irradiance modeling.
# pvlib defaults to sea level if altitude is not specified.
#
# This script allows passing altitude explicitly via SITE_ALTITUDE_M.
# If left as None, altitude = 0 m is assumed.
#
# For low-altitude sites (<500 m), impact is usually small.
# For high-altitude plants, providing altitude improves DNI accuracy
# and PVsyst agreement.
SITE_ALTITUDE_M = None  # e.g. 480 for Ranabima; set to None to assume sea level

# SAPM matching PVsyst Uc=20, Uv=0 → close-mount
sapm_params = pvlib.temperature.TEMPERATURE_MODEL_PARAMETERS["sapm"]["close_mount_glass_glass"]
# NOTE: Validate annual energy vs PVsyst; adjust SAPM mount if >3–5% deviation

# =====================================================================
# INVERTER EFFICIENCY MODEL (TODO: load-dependent η(Pdc))
# =====================================================================
# TODO:
# PVsyst uses a load-dependent inverter efficiency curve η(Pdc),
# not a constant value. This stub allows replacing the flat efficiency
# with a curve-based model extracted from PVsyst if/when available.
#
# If inverter efficiency curve data (Pdc vs η) is provided,
# populate inverter_eff_curve_kw and inverter_eff_curve_eta
# and set USE_INVERTER_CURVE = True.
USE_INVERTER_CURVE = False  # Set True when inverter curve is available

# Example placeholder curve (to be replaced with PVsyst data)
# Replace with curve values from PVsyst inverter component (.OND) or datasheet.
inverter_eff_curve_kw  = np.array([0, 20, 50, 100, 150, 200])   # kW DC
inverter_eff_curve_eta = np.array([0.0, 0.93, 0.965, 0.975, 0.973, 0.970])



# =====================================================================
# 1) USER INPUTS (Solcast + PVsyst + Plant Geometry): MUST ADJUST
# =====================================================================

API_KEY = "NSBpiyy3hMS5I6A52Gbnq0iwusBiOpXN"
lat, lon = 7.28, 80.59
period = "PT60M"
TIMEZONE = "Asia/Colombo"

# Orientation list extracted from PVsyst PDF
orientations = [
    {"tilt": 16, "azimuth": -120, "name": "O1", "module_count": 54},
    {"tilt": 16, "azimuth": 60,   "name": "O2", "module_count": 54},
    {"tilt": 22, "azimuth": -165, "name": "O3", "module_count": 54},
    {"tilt": 22, "azimuth": 15,   "name": "O4", "module_count": 54},
    {"tilt": 23, "azimuth": 165,  "name": "O5", "module_count": 55},
    {"tilt": 23, "azimuth": -15,  "name": "O6", "module_count": 55},
    {"tilt": 12, "azimuth": -109, "name": "O7", "module_count": 60},
    {"tilt": 12, "azimuth": 71,   "name": "O8", "module_count": 60},
]

module_area = 2.586    # m² per module (PV Model legend-> Size gives measures in mm)
total_module_area = sum(o["module_count"] * module_area for o in orientations)

module_efficiency_stc = 0.2151
gamma_p = -0.003       # per °C: This is a placeholder value, get the real value from module datasheet 

INV_AC_RATING = 100_000  # W Pnom total
far_shading_factor = 0.982   # PVsyst = 1-(IAM factor on global)%/100
albedo = 0.20                # PVsyst value

# Inverter DC threshold power (Pmin / Pthresh)
# NOTE: PVsyst simulation report does not expose numeric Pmin.
# Value below is a conservative placeholder for BI/SCADA realism.
# Replace with value from PVsyst inverter component (.OND) or datasheet.
PDC_THRESHOLD_KW = 1.0

# PVsyst DC losses (multiplicative)
soiling = 0.03
LID = 0.014
mismatch = 0.0085
dc_wiring = 0.009
dc_loss_factor = (1 - soiling)*(1 - LID)*(1 - mismatch)*(1 - dc_wiring)

# IAM VALUES FROM PVsyst
iam_angles = np.array([0, 30, 50, 60, 70, 75, 80, 85, 90])
iam_values = np.array([1.000, 0.999, 0.987, 0.963, 0.892, 0.814, 0.679, 0.438, 0])

# Default wind speed used ONLY if Solcast does not provide wind_speed
# Typical calm-wind assumption: 1–2 m/s
# PVsyst implicitly assumes similar values in absence of detailed data
DEFAULT_WIND_SPEED_MS = 1.0

# Site altitude above mean sea level (meters)
# OPTIONAL INPUT:
# - PVsyst uses altitude in air-mass and DNI calculations.
# - Extract from PVsyst report → "Geographical Site" section.
# - If not provided (None), model falls back to sea level (0 m),
#   which is acceptable for low-altitude sites but may introduce
#   small DNI errors for high-altitude plants.
# NOTE ON ALTITUDE:
# PVsyst includes site altitude in solar geometry and irradiance modeling.
# pvlib defaults to sea level if altitude is not specified.
#
# This script allows passing altitude explicitly via SITE_ALTITUDE_M.
# If left as None, altitude = 0 m is assumed.
#
# For low-altitude sites (<500 m), impact is usually small.
# For high-altitude plants, providing altitude improves DNI accuracy
# and PVsyst agreement.
SITE_ALTITUDE_M = None  # e.g. 480 for Ranabima; set to None to assume sea level

# SAPM matching PVsyst Uc=20, Uv=0 → close-mount
sapm_params = pvlib.temperature.TEMPERATURE_MODEL_PARAMETERS["sapm"]["close_mount_glass_glass"]
# NOTE: Validate annual energy vs PVsyst; adjust SAPM mount if >3–5% deviation

# =====================================================================
# INVERTER EFFICIENCY MODEL (TODO: load-dependent η(Pdc))
# =====================================================================
# TODO:
# PVsyst uses a load-dependent inverter efficiency curve η(Pdc),
# not a constant value. This stub allows replacing the flat efficiency
# with a curve-based model extracted from PVsyst if/when available.
#
# If inverter efficiency curve data (Pdc vs η) is provided,
# populate inverter_eff_curve_kw and inverter_eff_curve_eta
# and set USE_INVERTER_CURVE = True.
USE_INVERTER_CURVE = False  # Set True when inverter curve is available

# Example placeholder curve (to be replaced with PVsyst data)
# Replace with curve values from PVsyst inverter component (.OND) or datasheet.
inverter_eff_curve_kw  = np.array([0, 20, 50, 100, 150, 200])   # kW DC
inverter_eff_curve_eta = np.array([0.0, 0.93, 0.965, 0.975, 0.973, 0.970])
