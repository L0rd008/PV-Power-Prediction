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
        attrs = await tb.get_asset_attributes("3c2492b0-5669-11f0-b892-f5acae6d9b71")
        if "pvlib_config" in attrs:
            print("FOUND PVLIB_CONFIG!")
            print(attrs["pvlib_config"])
        else:
            print("NO pvlib_config found.")
        print("ORIENTATIONS string:")
        print(repr(attrs.get("orientations")))

if __name__ == "__main__":
    asyncio.run(test())
