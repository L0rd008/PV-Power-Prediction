"""Health-check endpoint."""

from fastapi import APIRouter

from app.models.response_models import HealthResponse
from app.utils.config import settings

router = APIRouter(tags=["Health"])


@router.get("/health", response_model=HealthResponse, summary="Service health check")
async def health() -> HealthResponse:
    return HealthResponse(
        status="ok",
        service=settings.APP_NAME,
        version=settings.APP_VERSION,
        model="pvlib-h-a3-v1",
    )
