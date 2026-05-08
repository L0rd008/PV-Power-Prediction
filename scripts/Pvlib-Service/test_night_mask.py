import pandas as pd
from app.physics.config import PlantConfig
from app.physics.pipeline import compute_ac_power

attrs = {
    'orientations': '[{"name":"Main","tilt":10,"azimuth":180,"use_measured_poa":false}]',
    'station': '{"ghi_key":"ghi", "poa_key":"poa"}',
    'module': '{"area_m2":2.0,"efficiency_stc":0.2,"gamma_p":-0.003}',
    'inverter': '{"ac_rating_kw":1000,"flat_efficiency":0.98}',
    'latitude': 6.9, 'longitude': 79.8, 'timezone': 'Asia/Colombo',
    'defaults': '{"wind_speed_ms":1,"air_temp_c":30}',
    'capacity_kwp': 1000,
    'pvlib_enabled': 'true', 'isPlant': 'true'
}
config = PlantConfig.from_tb_attributes('test', attrs)

idx = pd.date_range('2026-05-06 12:00:00', periods=3, freq='1min', tz='Asia/Colombo')
df_weather = pd.DataFrame({'ghi': [800, 800, 800], 'poa': [0.0, 0.0, 0.0]}, index=idx)

res = compute_ac_power(config, df_weather, 'tb_station')
print(res['potential_power_kw'])
