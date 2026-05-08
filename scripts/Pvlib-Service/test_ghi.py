import sys
import pandas as pd
from app.physics.config import PlantConfig
from app.physics.pipeline import compute_ac_power

attrs = {
    'orientations': '[{"name":"Main","tilt":10,"azimuth":180,"use_measured_poa":false}]',
    'station': '{"ghi_key":"wstn1_horiz_irradiance"}',
    'module': '{"area_m2":2.0,"efficiency_stc":0.2,"gamma_p":-0.003}',
    'inverter': '{"ac_rating_kw":1000,"flat_efficiency":0.98}',
    'latitude': 6.9, 'longitude': 79.8, 'timezone': 'Asia/Colombo',
    'defaults': '{"wind_speed_ms":1,"air_temp_c":30}',
    'capacity_kwp': 1000,
    'pvlib_enabled': 'true', 'isPlant': 'true'
}
config = PlantConfig.from_tb_attributes('test', attrs)

idx = pd.date_range('2026-05-06 12:00:00', periods=5, freq='1min', tz='Asia/Colombo')
df_weather = pd.DataFrame({'ghi': [800, 810, 820, 830, 840]}, index=idx)

try:
    res = compute_ac_power(config, df_weather, 'tb_station')
    print(res)
except Exception as e:
    import traceback
    traceback.print_exc()
