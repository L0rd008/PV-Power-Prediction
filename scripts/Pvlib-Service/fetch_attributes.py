import asyncio
import os
import sys
from datetime import datetime
from dotenv import load_dotenv

# Add the app directory to sys.path so we can import ThingsBoardClient
sys.path.append(os.path.dirname(__file__))

from app.services.thingsboard_client import ThingsBoardClient

async def fetch_attributes(asset_id: str):
    load_dotenv()
    
    tb_host = os.getenv("TB_HOST")
    tb_user = os.getenv("TB_USERNAME")
    tb_pass = os.getenv("TB_PASSWORD")
    
    if not all([tb_host, tb_user, tb_pass]):
        print("Error: Missing ThingsBoard credentials in .env file.")
        return

    print(f"Connecting to ThingsBoard at {tb_host}...")
    
    async with ThingsBoardClient(tb_host, tb_user, tb_pass) as tb:
        print(f"Fetching SERVER_SCOPE attributes for asset: {asset_id}\n")
        
        # We use the raw _get method to keep the 'lastUpdateTs' field
        raw_data = await tb._get(f"/api/plugins/telemetry/ASSET/{asset_id}/values/attributes/SERVER_SCOPE")
        
        if not raw_data:
            print("No attributes found.")
            return
            
        # Sort by lastUpdateTs descending (most recent first)
        sorted_data = sorted(raw_data, key=lambda x: x.get('lastUpdateTs', 0), reverse=True)
        
        print(f"{'LAST UPDATED':<22} | {'KEY':<35} | {'VALUE'}")
        print("-" * 80)
        
        for entry in sorted_data:
            ts = entry.get('lastUpdateTs', 0)
            dt_str = datetime.fromtimestamp(ts / 1000.0).strftime('%Y-%m-%d %H:%M:%S') if ts else "Unknown"
            
            key = entry.get('key', '')
            val = str(entry.get('value', ''))
            
            # Truncate long values for display purposes if needed, though usually fine
            if len(val) > 100:
                val = val[:97] + "..."
                
            print(f"{dt_str:<22} | {key:<35} | {val}")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python fetch_attributes.py <ASSET_ID>")
        sys.exit(1)
        
    asset_id = sys.argv[1]
    asyncio.run(fetch_attributes(asset_id))
