import asyncio
import os
import sys
from dotenv import load_dotenv

sys.path.append(os.path.dirname(__file__))
from app.services.thingsboard_client import ThingsBoardClient

async def test():
    load_dotenv()
    tb_host = os.getenv("TB_HOST")
    tb_user = os.getenv("TB_USERNAME")
    tb_pass = os.getenv("TB_PASSWORD")
    
    async with ThingsBoardClient(tb_host, tb_user, tb_pass) as tb:
        keys = ["ghi", "wstn1_horiz_irradiance"]
        latest = await tb.get_latest_telemetry("DEVICE", "d43fa340-6853-11f0-9429-751fa577e736", keys)
        print("Latest for SSK WSTN:", latest)

if __name__ == "__main__":
    asyncio.run(test())
