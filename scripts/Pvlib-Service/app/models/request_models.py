"""Request models for the Pvlib-Service API."""

from datetime import date
from typing import Optional

from pydantic import BaseModel, Field


class ForecastStartRequest(BaseModel):
    asset_id: str = Field(..., description="ThingsBoard plant asset UUID")
    already_have: bool = Field(
        False,
        description="Set to True if the asset already has historical telemetry "
                    "(skips existence check and goes straight to computation).",
    )


class ForecastWithDateRequest(BaseModel):
    asset_id: str
    already_have: bool = False
    start_date: Optional[date] = Field(None, description="ISO date (defaults to yesterday)")
    end_date: Optional[date] = Field(None, description="ISO date (defaults to yesterday)")


class NewAssetRequest(BaseModel):
    asset_id: str
    already_have: bool = False
