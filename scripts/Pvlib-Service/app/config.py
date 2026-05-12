"""
Application settings using pydantic-settings v2.

Loads configuration from environment variables and .env file.
Supports ThingsBoard connection, scheduler settings, data sources, and service mode.
"""
from __future__ import annotations

from typing import Optional

from pydantic import field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application configuration loaded from environment variables and .env file."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # ThingsBoard connection
    TB_HOST: str = "https://localhost"
    """ThingsBoard API endpoint URL."""

    TB_USERNAME: str = ""
    """ThingsBoard username for authentication."""

    TB_PASSWORD: str = ""
    """ThingsBoard password for authentication."""

    TB_ROOT_ASSET_IDS: str = ""
    """Comma-separated list of root asset IDs to discover plants from."""

    # Scheduler configuration
    SCHEDULER_ENABLED: bool = True
    """Enable/disable automatic scheduler."""

    SCHEDULER_INTERVAL_MINUTES: int = 1
    """Interval between scheduler cycles in minutes."""

    READ_LAG_SECONDS: int = 30
    """Seconds behind 'now' to start the read window (accounts for TB ingestion lag)."""

    READ_WINDOW_SECONDS: int = 90
    """Width of the telemetry read window in seconds."""

    MAX_CONCURRENT_PLANTS: int = 5
    """Maximum number of plants processed concurrently per scheduler cycle."""

    # Data sources
    SOLCAST_API_KEY: Optional[str] = None
    """API key for Solcast solar forecast service (optional)."""

    TZ_LOCAL: str = "Asia/Colombo"
    """Local timezone for the service (used for scheduling and reporting)."""

    # Weekly accuracy evaluation (Gap 20)
    EVAL_PLANT_IDS: str = ""
    """Comma-separated asset IDs of plants to include in the weekly accuracy report.
    Set to KSP / SOU / SSK asset UUIDs in production.  Empty = job is a no-op."""

    EVAL_ACTUAL_DAILY_KEY: str = "daily_energy_kwh"
    """TB timeseries key holding the metered daily energy (kWh) for each plant.
    Written by the site data pipeline; used as ground truth in weekly_eval."""

    # Service configuration
    MODE: str = "pvlib"
    """Service operation mode (e.g., 'pvlib', 'simple', 'solcast')."""

    LOG_LEVEL: str = "INFO"
    """Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)."""

    DEBUG: bool = False
    """Enable debug mode for verbose logging and development features."""

    @property
    def eval_plant_ids(self) -> list[str]:
        """Parse comma-separated eval plant IDs into a list of stripped strings."""
        return [x.strip() for x in self.EVAL_PLANT_IDS.split(",") if x.strip()]

    @field_validator("DEBUG", mode="before")
    @classmethod
    def _parse_debug_flag(cls, value):
        if isinstance(value, str):
            normalized = value.strip().lower()
            if normalized in {"release", "prod", "production", "false", "0", "no", "off"}:
                return False
            if normalized in {"debug", "dev", "development", "true", "1", "yes", "on"}:
                return True
        return value

    @property
    def root_asset_ids(self) -> list[str]:
        """Parse comma-separated root asset IDs into a list of stripped strings."""
        return [x.strip() for x in self.TB_ROOT_ASSET_IDS.split(",") if x.strip()]

    # Loss roll-up job (Phase L0+)
    LOSS_ROLLUP_ENABLED: bool = False
    """Master flag for the daily loss-rollup job. Default false until Phase L1.
    Set to true in .env after manual verification of a single plant (Phase L1)."""

    LOSS_DEFAULT_SETPOINT_KEYS: str = "setpoint_active_power,curtailment_limit,power_limit"
    """Comma-separated default setpoint keys used for curtailment ceiling calculation.
    Per-plant 'setpoint_keys' attribute overrides this when present."""

    LOSS_MIN_VALID_SAMPLES: int = 360
    """Minimum 1-min samples per day to produce a real daily loss value.
    Below this threshold (< 6 h of data), write -1 sentinel for all keys."""

    LOSS_LIFETIME_PAGE_DAYS: int = 90
    """Page size in days when paging history during /admin/recompute-lifetime.
    Keeps individual TB timeseries reads under ~130 k rows per call."""

    # Today-partial roll-up job (Phase L1+)
    LOSS_TODAY_PARTIAL_ENABLED: bool = False
    """Master flag for the intra-day today-partial cron.  Default false until Phase L1.
    Requires LOSS_ROLLUP_ENABLED=true as well; no-op when either flag is false."""

    LOSS_TODAY_PARTIAL_INTERVAL_MIN: int = 5
    """Interval in minutes for the today-partial cron (default 5)."""

    LOSS_TODAY_PARTIAL_DAY_START_HOUR: int = 5
    """Local-tz hour at which the today-partial cron starts firing (default 05:00).
    Sri Lanka fleet: sunrise ≈ 05:30."""

    LOSS_TODAY_PARTIAL_DAY_END_HOUR: int = 19
    """Local-tz hour at which the today-partial cron stops firing (default 19:00).
    Sri Lanka fleet: sunset ≈ 18:30.  Outside this window the job is a no-op."""

    LOSS_TODAY_PARTIAL_MIN_SAMPLES: int = 30
    """Minimum 1-min samples required by the today-partial path (default 30 — half an hour).
    Lower than the daily-job threshold (360) because we want partial values from mid-morning."""

    # P-value job (P50/P90/P95 probabilistic forecast telemetry)
    PVALUE_JOB_ENABLED: bool = False
    """Master flag for the annual P-value batch job.
    When true, registers a Jan-1 03:00 local-tz cron that fetches PVGIS multi-year
    data and writes forecast_p50/p90/p95 daily + monthly telemetry to TB.
    Default false — set true after smoke-testing /admin/run-pvalues-plant."""

    PVGIS_START_YEAR: int = 2005
    """First year of PVGIS-ERA5 historical data to fetch (inclusive).
    ERA5 within PVGIS starts at 2005."""

    PVGIS_END_YEAR: int = 2023
    """Last year of PVGIS-ERA5 historical data to fetch (inclusive).
    Keep at 2023 until PVGIS confirms a full 2024 ERA5 dataset."""

    PVGIS_RADDATABASE: str = "PVGIS-ERA5"
    """PVGIS radiation database to use.  'PVGIS-ERA5' is globally available.
    Alternative: 'PVGIS-SARAH2' (higher resolution but not global)."""

    PVGIS_REQUEST_TIMEOUT_S: int = 60
    """HTTP timeout in seconds for each PVGIS API call.
    PVGIS returns multi-year hourly data in one response — 60 s is generous."""

    PVGIS_RETRY_MAX: int = 3
    """Maximum retry attempts per PVGIS cell fetch on transient network errors.
    Exponential back-off: 2 s, 4 s, 8 s between attempts."""

    @field_validator("SCHEDULER_INTERVAL_MINUTES", "READ_LAG_SECONDS", "READ_WINDOW_SECONDS", "MAX_CONCURRENT_PLANTS")
    @classmethod
    def validate_positive_int(cls, v: int) -> int:
        """Ensure positive integer values for time and concurrency parameters."""
        if v <= 0:
            raise ValueError("Value must be positive")
        return v

    @field_validator("LOG_LEVEL")
    @classmethod
    def validate_log_level(cls, v: str) -> str:
        """Validate log level against standard logging levels."""
        valid_levels = {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}
        if v.upper() not in valid_levels:
            raise ValueError(f"LOG_LEVEL must be one of {valid_levels}")
        return v.upper()


# Global singleton settings instance
settings = Settings()
