import pandas as pd
import numpy as np
from app.physics.config import PlantConfig
from app.physics.pipeline import compute_ac_power

attrs = {
    "orientations": '[{"name":"Main","tilt":0,"azimuth":0,"module_count":22040,"use_measured_poa":false}]',
    "solcast_resource_id": "null",
    "defaults": '{"wind_speed_ms":1,"air_temp_c":28.43}',
    "dc_wiring": 0.015,
    "mismatch": 0.021,
    "module_quality": -0.008,
    "lid": 0,
    "soiling": 0.03,
    "ac_wiring": 0,
    "albedo": 0.2,
    "far_shading": 1,
    "thermal_model": '{"Uc":29,"Uv":0}',
    "station": '{"poa_key":"wstn1_horiz_irradiance","air_temp_key":"wstn1_temperature_ambient","wind_speed_key":"wstn1_wind_speed","freshness_minutes":10,"sanity_max_ghi_wm2":1400,"sanity_max_poa_wm2":1500}',
    "iam": '{"angles":[0,40,50,60,70,75,80,85,90],"values":[1,1,1,1,1,0.984,0.949,0.83,0]}',
    "inverter": '{"ac_rating_kw":10200,"dc_threshold_kw":0,"use_efficiency_curve":true,"efficiency_curve_kw":[506,610,701,804,934,1090,1232,1388,1557,1712,1881,2102,2322,2504,2698,2958,3191,3477,3762,4086,4450,4800,5228,5643,6097,6409,6746,7070,7330,7836,8368,8796,9133,9548,9898,10249,10573],"efficiency_curve_eta":[0.979,0.98048,0.98197,0.98346,0.98494,0.98606,0.98699,0.98755,0.98829,0.98866,0.98922,0.98978,0.99015,0.99015,0.99015,0.98996,0.98996,0.98996,0.98996,0.98996,0.98996,0.98978,0.98978,0.98959,0.98959,0.98941,0.98922,0.98922,0.98903,0.98885,0.98885,0.98885,0.98866,0.98866,0.98866,0.98866,0.98866],"flat_efficiency":0.9855}',
    "module": '{"area_m2":2.7012,"efficiency_stc":0.235,"gamma_p":-0.0029}',
    "timezone": "Asia/Colombo",
    "altitude_m": 5,
    "name": "Sooryashakthi",
    "pvlib_enabled": "true",
    "weather_station_id": "d43fa340-6853-11f0-9429-751fa577e736",
    "tariff_rate_lkr": 15,
    "Capacity": 10000,
    "latitude": 7.527987068215883,
    "longitude": 81.73268058029599,
    "isPlant": "true"
}
config = PlantConfig.from_tb_attributes("test", attrs)
print(config)

idx = pd.date_range("2026-05-06 12:00:00", periods=5, freq="1min", tz="Asia/Colombo")
df_weather = pd.DataFrame({
    "poa": [800, 810, 820, 830, 840],
    "air_temp": [30, 30, 30, 30, 30],
    "wind_speed": [2, 2, 2, 2, 2]
}, index=idx)

try:
    res = compute_ac_power(config, df_weather, "tb_station")
    print("SUCCESS")
    print(res)
except Exception as e:
    import traceback
    traceback.print_exc()
