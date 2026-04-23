"""Response models for the Pvlib-Service API."""

from datetime import datetime
from typing import Any, Dict, List, Optional

from pydantic import BaseModel


class ForecastStartResponse(BaseModel):
    job_id: str
    status: str
    asset_id: str
    date_range: Dict[str, str]
    already_have: bool
    message: str = "Forecast job started in background"


class ForecastStatusResponse(BaseModel):
    job_id: str
    status: str
    asset_id: str
    already_have: bool
    date_range: Optional[Dict[str, str]] = None
    progress: Optional[str] = None
    started_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    error: Optional[str] = None
    total_assets: Optional[int] = None
    processed_assets: int = 0
    successful_assets: int = 0
    failed_assets: int = 0
    skipped_assets: int = 0
    current_step: Optional[str] = None
    errors: List[Dict[str, Any]] = []


class NewAssetResponse(BaseModel):
    status: str
    asset_id: str
    message: str
    ac_records_written: Optional[int] = None
    daily_records_written: Optional[int] = None
    error: Optional[str] = None


class ErrorResponse(BaseModel):
    error: str
    message: str
    asset_id: Optional[str] = None
    missing_attributes: Optional[List[str]] = None


class HealthResponse(BaseModel):
    status: str
    service: str
    version: str
    model: str = "pvlib-h-a3-v1"
