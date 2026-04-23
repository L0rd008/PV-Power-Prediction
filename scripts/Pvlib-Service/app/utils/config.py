"""
Centralised application settings loaded from environment variables.

All runtime secrets (TB credentials, Solcast API key) come from the
environment — never hardcoded.  See ``.env.example`` for the full list.
"""

import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()


class Settings:
    # ── Application ────────────────────────────────────────────────────────
    APP_NAME: str = "pvlib-power-prediction-service"
    APP_VERSION: str = "1.0.0"
    DEBUG: bool = os.getenv("DEBUG", "false").lower() == "true"

    # ── ThingsBoard ────────────────────────────────────────────────────────
    TB_HOST: str = os.getenv("TB_HOST", "http://localhost:8080")
    TB_USERNAME: str = os.getenv("TB_USERNAME", "")
    TB_PASSWORD: str = os.getenv("TB_PASSWORD", "")
    TB_TOKEN_EXPIRY: int = 3600  # seconds; refresh 5 min before expiry

    # ── Asset hierarchy ────────────────────────────────────────────────────
    # How many levels deep to traverse from the requested root asset.
    # Level 1 = direct children, Level 2 = grandchildren, etc.
    DEFAULT_TARGET_LEVEL: int = int(os.getenv("TARGET_LEVEL", "3"))

    # ── Timezone ───────────────────────────────────────────────────────────
    TZ_LOCAL: str = os.getenv("TZ_LOCAL", "Asia/Colombo")

    # ── Solcast (Tier-2 fallback) ──────────────────────────────────────────
    # Leave empty to disable Solcast and go straight to clear-sky (Tier 3).
    SOLCAST_API_KEY: str = os.getenv("SOLCAST_API_KEY", "")

    # ── Logging ────────────────────────────────────────────────────────────
    LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO")

    # ── Service mode (future unified image) ───────────────────────────────
    MODE: str = os.getenv("MODE", "pvlib")  # pvlib | solcast | simple

    @classmethod
    def validate(cls) -> None:
        missing = [k for k in ("TB_HOST", "TB_USERNAME", "TB_PASSWORD") if not getattr(cls, k)]
        if missing:
            raise ValueError(
                f"Required environment variables not set: {missing}. "
                "Check your .env file or container environment."
            )


settings = Settings()
