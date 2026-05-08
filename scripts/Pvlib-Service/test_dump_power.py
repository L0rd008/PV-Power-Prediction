import asyncio
import os
import sys
import pandas as pd
from dotenv import load_dotenv

sys.path.append(os.path.dirname(__file__))
from app.services.thingsboard_client import ThingsBoardClient
from app.physics.config import PlantConfig
from app.physics.pipeline import compute_ac_power

async def test():
    load_dotenv()
    tb_host = os.getenv("TB_HOST")
    tb_user = os.getenv("TB_USERNAME")
    tb_pass = os.getenv("TB_PASSWORD")
    
    async with ThingsBoardClient(tb_host, tb_user, tb_pass) as tb:
        attrs = await tb.get_asset_attributes("3c2492b0-5669-11f0-b892-f5acae6d9b71")
        config = PlantConfig.from_tb_attributes("3c2492b0-5669-11f0-b892-f5acae6d9b71", attrs)
        
        idx = pd.date_range('2026-05-06 12:52:38', periods=1, tz='Asia/Colombo')
        df_weather = pd.DataFrame({'poa': [943.9], 'air_temp': [30], 'wind_speed': [2]}, index=idx)

        res = compute_ac_power(config, df_weather, 'tb_station')
        print("Calculated Power:")
        print(res['potential_power_kw'])

if __name__ == "__main__":
    asyncio.run(test())
