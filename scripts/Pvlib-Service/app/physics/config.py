"""
Plant configuration models for pvlib-based power prediction.

Supports two attribute layouts in ThingsBoard:
  1. Single JSON blob: SERVER_SCOPE attribute "pvlib_config" (preferred,
     produced by kebithigollewa_pvlib_config.json)
  2. Flat attributes: individual SERVER_SCOPE keys (latitude, longitude,
     module JSON, inverter JSON, etc.)

The from_tb_attributes() classmethod auto-detects which layout is present.
"""
from __future__ import annotations

import json
import logging
from typing import Optional

from pydantic import BaseModel, field_validator

log = logging.getLogger(__name__)


class OrientationConfig(BaseModel):
    """Configuration for a single array orientation (tilt/azimuth pair)."""

    name: str = "Main"
    """Orientation name for logging and reporting."""

    tilt: float = 20.0
    """Tilt angle in degrees (0=horizontal, 90=vertical)."""

    azimuth: float = 0.0
    """Azimuth in degrees from South (pvlib convention: 0=South, 90=West, -90=East)."""

    module_count: int = 1000
    """Number of modules in this orientation."""

    use_measured_poa: bool = False
    """True → use POA sensor directly, skip Perez transposition."""


class ModuleConfig(BaseModel):
    """Photovoltaic module specifications."""

    area_m2: float = 2.0
    """Physical area of one module in square meters."""

    efficiency_stc: float = 0.20
    """Efficiency at STC in fraction (e.g., 0.2248)."""

    gamma_p: float = -0.0040
    """Power temperature coefficient in /°C (e.g., -0.0029 for c-Si)."""


class InverterConfig(BaseModel):
    """Inverter specifications and efficiency characteristics."""

    ac_rating_kw: float = 1000.0
    """AC power rating of the inverter in kilowatts."""

    dc_threshold_kw: float = 0.0
    """Minimum DC power threshold for inverter operation (kW)."""

    use_efficiency_curve: bool = False
    """Use efficiency curve interpolation (True) or flat efficiency (False)."""

    efficiency_curve_kw: list[float] = []
    """DC input power points for efficiency curve (kW), in ascending order."""

    efficiency_curve_eta: list[float] = []
    """Efficiency values at each power point (fraction), matching length of efficiency_curve_kw."""

    flat_efficiency: float = 0.98
    """Fixed efficiency used when use_efficiency_curve=False."""

    @field_validator("efficiency_curve_kw", "efficiency_curve_eta", mode="before")
    @classmethod
    def ensure_list(cls, v):
        """Ensure efficiency curve parameters are lists."""
        if v is None:
            return []
        if isinstance(v, str):
            try:
                v = json.loads(v)
            except json.JSONDecodeError:
                return []
        return v if isinstance(v, list) else []


class IAMConfig(BaseModel):
    """Incidence Angle Modifier (IAM) profile for Fresnel losses."""

    angles: list[float] = [0, 40, 50, 60, 70, 75, 80, 85, 90]
    """AOI breakpoints in degrees."""

    values: list[float] = [1.0, 1.0, 1.0, 1.0, 1.0, 0.984, 0.949, 0.830, 0.0]
    """IAM factors at each breakpoint (1.0=no loss, 0.0=complete loss)."""


class StationConfig(BaseModel):
    """Weather station data mapping and quality gates."""

    ghi_key: str = "ghi"
    """ThingsBoard telemetry key for Global Horizontal Irradiance (W/m²)."""

    poa_key: Optional[str] = None
    """Optional key for measured Plane-of-Array irradiance (W/m²)."""

    air_temp_key: str = "temperature"
    """Key for ambient air temperature (°C)."""

    wind_speed_key: Optional[str] = None
    """Optional key for wind speed (m/s)."""

    freshness_minutes: int = 10
    """Maximum age of telemetry to accept as valid (minutes)."""

    sanity_max_ghi_wm2: float = 1400.0
    """Maximum acceptable GHI value for sanity checking (W/m²)."""

    sanity_max_poa_wm2: float = 1500.0
    """Maximum acceptable POA value for sanity checking (W/m²)."""


class DefaultsConfig(BaseModel):
    """Default values for missing weather data."""

    wind_speed_ms: float = 1.0
    """Default wind speed when unavailable (m/s)."""

    air_temp_c: float = 25.0
    """Default ambient temperature when unavailable (°C)."""


class PlantConfig(BaseModel):
    """Complete plant configuration including location, equipment, and losses."""

    # Identity
    plant_name: str = "Unknown Plant"
    """Human-readable plant name."""

    asset_id: str = ""
    """ThingsBoard asset UUID."""

    # Location
    latitude: float = 0.0
    """Latitude in degrees (-90 to +90)."""

    longitude: float = 0.0
    """Longitude in degrees (-180 to +180)."""

    altitude_m: float = 0.0
    """Altitude above sea level in meters."""

    timezone: str = "UTC"
    """Local timezone (e.g., 'Asia/Colombo')."""

    # Equipment
    orientations: list[OrientationConfig] = []
    """List of array orientations (tilt/azimuth pairs)."""

    module: ModuleConfig = ModuleConfig()
    """Module specifications."""

    inverter: InverterConfig = InverterConfig()
    """Inverter specifications."""

    iam: IAMConfig = IAMConfig()
    """Incidence angle modifier profile."""

    station: StationConfig = StationConfig()
    """Weather station data mapping."""

    defaults: DefaultsConfig = DefaultsConfig()
    """Default values for missing data."""

    thermal_model: str = "open_rack_glass_glass"
    """SAPM thermal model key (e.g., 'open_rack_glass_glass')."""

    # Losses (fractions: positive = loss, negative = gain)
    soiling: float = 0.0
    """Soiling loss fraction (0.0 to 1.0)."""

    lid: float = 0.0
    """Light-induced degradation loss fraction (0.0 to 1.0)."""

    module_quality: float = 0.0
    """Module quality factor (negative = gain, positive = loss)."""

    mismatch: float = 0.0
    """Mismatch loss fraction (0.0 to 1.0)."""

    dc_wiring: float = 0.0
    """DC wiring loss fraction (0.0 to 1.0)."""

    ac_wiring: float = 0.0
    """AC wiring loss fraction (0.0 to 1.0)."""

    albedo: float = 0.20
    """Ground albedo for reflected irradiance (0.0 to 1.0)."""

    far_shading: float = 1.0
    """Far shading multiplier (1.0=no shading, <1.0=shaded)."""

    # Related devices and metadata
    weather_station_id: Optional[str] = None
    """ThingsBoard device ID for external weather station (optional)."""

    p341_device_id: Optional[str] = None
    """Device ID for P341 power meter (optional)."""

    active_power_unit: str = "kW"
    """Unit for plant's active power telemetry ('W' or 'kW')."""

    solcast_resource_id: Optional[str] = None
    """Solcast resource ID for this plant (optional)."""

    capacity_kwp: Optional[float] = None
    """Installed capacity in kWp (optional, for scaling checks)."""

    def model_post_init(self, __context) -> None:
        """Ensure at least one default orientation if none provided."""
        if not self.orientations:
            self.orientations = [OrientationConfig()]

    # ────────────────────────────────────────────────────────────────────
    # Factory method
    # ────────────────────────────────────────────────────────────────────

    @classmethod
    def from_tb_attributes(cls, asset_id: str, attrs: dict) -> "PlantConfig":
        """Build PlantConfig from ThingsBoard SERVER_SCOPE attributes.

        Auto-detects whether the configuration is stored as a single JSON blob
        (pvlib_config) or spread across flat attributes. Tries the blob first,
        then falls back to flat parsing.

        Parameters
        ----------
        asset_id : str
            ThingsBoard asset UUID.
        attrs : dict
            Flat dictionary of SERVER_SCOPE attributes (values may be strings
            or native types).

        Returns
        -------
        PlantConfig
            Fully validated plant configuration.

        Raises
        ------
        ValueError
            If required attributes are missing or JSON parsing fails.
        """

        def jparse(key, default=None):
            """Parse JSON value or return as-is if not JSON."""
            val = attrs.get(key, default)
            if isinstance(val, str):
                try:
                    return json.loads(val)
                except (json.JSONDecodeError, ValueError):
                    return val  # Return as-is if not JSON
            return val if val is not None else default

        pvlib_cfg = jparse("pvlib_config")
        if isinstance(pvlib_cfg, dict):
            log.debug("asset %s: loading config from pvlib_config blob", asset_id)
            return cls._from_blob(asset_id, pvlib_cfg, attrs)

        log.debug("asset %s: loading config from flat attributes", asset_id)
        return cls._from_flat(asset_id, attrs)

    @classmethod
    def _from_blob(cls, asset_id: str, cfg: dict, attrs: dict) -> "PlantConfig":
        """Parse PlantConfig from a pvlib_config JSON blob.

        Parameters
        ----------
        asset_id : str
            ThingsBoard asset UUID.
        cfg : dict
            Parsed pvlib_config dictionary.
        attrs : dict
            Flat attributes (used as fallback for missing fields).

        Returns
        -------
        PlantConfig
        """

        def _float(d, key, default=0.0):
            """Safely convert value to float."""
            try:
                return float(d.get(key, default))
            except (TypeError, ValueError):
                return default

        loc = cfg.get("location", {})
        losses = cfg.get("losses", {})
        raw_orients = cfg.get("orientations", [{}])

        return cls(
            plant_name=cfg.get("plant_name", attrs.get("name", "Unknown")),
            asset_id=asset_id,
            latitude=_float(loc, "lat", attrs.get("latitude", 0.0)),
            longitude=_float(loc, "lon", attrs.get("longitude", 0.0)),
            altitude_m=_float(loc, "altitude_m", attrs.get("altitude_m", 0.0)),
            timezone=loc.get("timezone", attrs.get("timezone", "UTC")),
            orientations=[OrientationConfig(**o) for o in raw_orients],
            module=ModuleConfig(**cfg["module"]) if "module" in cfg else ModuleConfig(),
            inverter=InverterConfig(**cfg["inverter"]) if "inverter" in cfg else InverterConfig(),
            iam=IAMConfig(**cfg["iam"]) if "iam" in cfg else IAMConfig(),
            station=StationConfig(**cfg["station"]) if "station" in cfg else StationConfig(),
            defaults=DefaultsConfig(**cfg["defaults"]) if "defaults" in cfg else DefaultsConfig(),
            thermal_model=cfg.get("thermal_model", "open_rack_glass_glass"),
            soiling=_float(losses, "soiling", attrs.get("soiling", 0.0)),
            lid=_float(losses, "lid", attrs.get("lid", 0.0)),
            module_quality=_float(losses, "module_quality", attrs.get("module_quality", 0.0)),
            mismatch=_float(losses, "mismatch", attrs.get("mismatch", 0.0)),
            dc_wiring=_float(losses, "dc_wiring", attrs.get("dc_wiring", 0.0)),
            ac_wiring=_float(losses, "ac_wiring", attrs.get("ac_wiring", 0.0)),
            albedo=_float(losses, "albedo", cfg.get("albedo", attrs.get("albedo", 0.20))),
            far_shading=_float(losses, "far_shading", cfg.get("far_shading", attrs.get("far_shading", 1.0))),
            active_power_unit=str(attrs.get("active_power_unit", "kW")),
            solcast_resource_id=cfg.get("solcast_resource_id") or attrs.get("solcast_resource_id"),
            capacity_kwp=float(attrs["Capacity"]) if attrs.get("Capacity") is not None else None,
        )

    @classmethod
    def _from_flat(cls, asset_id: str, attrs: dict) -> "PlantConfig":
        """Parse PlantConfig from flat attributes.

        Parameters
        ----------
        asset_id : str
            ThingsBoard asset UUID.
        attrs : dict
            Flat attributes dictionary.

        Returns
        -------
        PlantConfig
        """

        def jparse(key, default=None):
            """Parse JSON value or return as-is if not JSON."""
            val = attrs.get(key, default)
            if isinstance(val, str):
                try:
                    return json.loads(val)
                except (json.JSONDecodeError, ValueError):
                    return default
            return val if val is not None else default

        def _float(key, default=0.0):
            """Safely convert attribute to float."""
            try:
                return float(attrs.get(key, default))
            except (TypeError, ValueError):
                return default

        raw_orients = jparse("orientations") or []
        module_raw = jparse("module") or {}
        inverter_raw = jparse("inverter") or {}
        iam_raw = jparse("iam") or {}
        station_raw = jparse("station") or {}
        defaults_raw = jparse("defaults") or {}

        return cls(
            plant_name=str(attrs.get("name", "Unknown")),
            asset_id=asset_id,
            latitude=_float("latitude"),
            longitude=_float("longitude"),
            altitude_m=_float("altitude_m"),
            timezone=str(attrs.get("timezone", "UTC")),
            orientations=[OrientationConfig(**o) for o in raw_orients] if raw_orients else [OrientationConfig()],
            module=ModuleConfig(**module_raw) if module_raw else ModuleConfig(),
            inverter=InverterConfig(**inverter_raw) if inverter_raw else InverterConfig(),
            iam=IAMConfig(**iam_raw) if iam_raw else IAMConfig(),
            station=StationConfig(**station_raw) if station_raw else StationConfig(),
            defaults=DefaultsConfig(**defaults_raw) if defaults_raw else DefaultsConfig(),
            thermal_model=str(attrs.get("thermal_model", "open_rack_glass_glass")),
            soiling=_float("soiling"),
            lid=_float("lid"),
            module_quality=_float("module_quality"),
            mismatch=_float("mismatch"),
            dc_wiring=_float("dc_wiring"),
            ac_wiring=_float("ac_wiring"),
            albedo=_float("albedo", 0.20),
            far_shading=_float("far_shading", 1.0),
            active_power_unit=str(attrs.get("active_power_unit", "kW")),
            solcast_resource_id=attrs.get("solcast_resource_id"),
            capacity_kwp=float(attrs["Capacity"]) if attrs.get("Capacity") is not None else None,
        )
