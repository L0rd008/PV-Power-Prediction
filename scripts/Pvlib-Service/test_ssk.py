import pandas as pd
import json
from app.physics.config import PlantConfig
from app.physics.pipeline import compute_ac_power

attrs = {
    'dashboardMode': 5,
    'use_measured_poa': True,
    'station': {'poa_key': 'wstn1_tilted_irradiance', 'air_temp_key': 'wstn1_temperature_ambient', 'wind_speed_key': 'wstn1_wind_speed'},
    'orientations': [{'name': 'Main', 'tilt': 0, 'azimuth': 0, 'module_count': 22040, 'use_measured_poa': False}],
    'solcast_resource_id': None,
    'defaults': {'wind_speed_ms': 1, 'air_temp_c': 28.43},
    'dc_wiring': 0.015,
    'mismatch': 0.021,
    'module_quality': -0.008,
    'lid': 0,
    'soiling': 0.03,
    'ac_wiring': 0,
    'albedo': 0.2,
    'far_shading': 1,
    'thermal_model': {'Uc': 29, 'Uv': 0},
    'iam': {'angles': [0, 40, 50, 60, 70, 75, 80, 85, 90], 'values': [1, 1, 1, 1, 1, 0.984, 0.949, 0.83, 0]},
    'inverter': {'ac_rating_kw': 10200, 'dc_threshold_kw': 0, 'use_efficiency_curve': True, 'efficiency_curve_kw': [506, 10573], 'efficiency_curve_eta': [0.979, 0.9886]},
    'module': {'area_m2': 2.7012, 'efficiency_stc': 0.235, 'gamma_p': -0.0029},
    'timezone': 'Asia/Colombo',
    'altitude_m': 5,
    'name': 'Sooryashakthi',
    'latitude': 7.527987,
    'longitude': 81.73268,
    'Capacity': 10000
}

config = PlantConfig.from_tb_attributes('test', attrs)

idx = pd.date_range('2026-05-06 12:52:38', periods=1, tz='Asia/Colombo')
df_weather = pd.DataFrame({'poa': [943.9], 'air_temp': [30], 'wind_speed': [2]}, index=idx)

res = compute_ac_power(config, df_weather, 'tb_station')
print(res['potential_power_kw'])
