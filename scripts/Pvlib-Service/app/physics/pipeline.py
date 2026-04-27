"""
H-A3 physics pipeline — PVsyst loss chain using pvlib 0.11.1+.

Loss chain order (matches PVsyst simulation order):
  1.  POA irradiance — Perez transposition from GHI, or direct measured POA
  2.  Far shading — pre-IAM multiplicative factor
  3.  SAPM cell temperature — pvlib.temperature.sapm_cell
  4.  AOI + IAM correction — pvlib.iam.interp on user-defined profile
  5.  DC power — efficiency × area × modules × temperature correction
  6.  DC losses — soiling, LID, module_quality, mismatch, dc_wiring (sequential)
  7.  Inverter — interpolated efficiency curve or flat efficiency
  8.  Plant clip — AC rating limit
  9.  AC wiring loss — post-inverter multiplicative

This module is a pure function with no globals, no I/O, and no side effects.
Its single entry-point is compute_ac_power(config, df_weather, data_source).
"""
from __future__ import annotations

import logging
from typing import Optional

import numpy as np
import pandas as pd
import pvlib
from pvlib.location import Location

from app.physics.config import IAMConfig, InverterConfig, PlantConfig

log = logging.getLogger(__name__)

# SAPM thermal model coefficients keyed by pvlib model name
_SAPM_PARAMS = {
    "open_rack_glass_glass": {"a": -3.47, "b": -0.0594, "deltaT": 3},
    "open_rack_glass_polymer": {"a": -3.56, "b": -0.0750, "deltaT": 3},
    "close_mount_glass_glass": {"a": -2.98, "b": -0.0471, "deltaT": 1},
    "insulated_back_glass_polymer": {"a": -2.81, "b": -0.0455, "deltaT": 0},
}
_DEFAULT_SAPM = _SAPM_PARAMS["open_rack_glass_glass"]


def compute_ac_power(
    config: PlantConfig,
    df_weather: pd.DataFrame,
    data_source: str = "unknown",
) -> pd.DataFrame:
    """Compute plant-level AC power from weather data.

    Implements the H-A3 physics model: GHI/POA transposition → temperature →
    IAM correction → DC power → losses → inverter → AC clipping.

    Parameters
    ----------
    config : PlantConfig
        Complete plant configuration including location, equipment, losses.
    df_weather : pd.DataFrame
        Weather data with timezone-aware DatetimeIndex. Required columns:
        - 'ghi': Global Horizontal Irradiance (W/m²)
        - 'air_temp': Ambient air temperature (°C)
        - 'wind_speed': Wind speed (m/s, may be all-NaN → defaults used)
        Optional:
        - 'poa': Measured Plane-of-Array irradiance (W/m²)
    data_source : str, optional
        Label for data provenance ('tb_station', 'solcast', 'clearsky').
        Propagated to output for traceability.

    Returns
    -------
    pd.DataFrame
        Columns:
        - 'potential_power_kw': Physics model AC output (kW) [primary widget key]
        - 'active_power_pvlib_kw': Same value, ops/diagnostic alias
        - 'data_source': String label for data provenance
        - 'model_version': 'pvlib-h-a3-v1'
        Index matches df_weather.index.
    """
    if df_weather.empty:
        log.warning("compute_ac_power: empty weather DataFrame for %s", config.plant_name)
        return _empty_result(df_weather.index, data_source)

    idx = df_weather.index
    loc = Location(
        latitude=config.latitude,
        longitude=config.longitude,
        altitude=config.altitude_m,
        tz=config.timezone,
    )

    # Ensure numeric, fill missing with defaults
    ghi = pd.to_numeric(df_weather.get("ghi", pd.Series(0.0, index=idx)), errors="coerce").fillna(0.0)
    poa_measured = pd.to_numeric(df_weather.get("poa", pd.Series(np.nan, index=idx)), errors="coerce")
    air_temp = (
        pd.to_numeric(df_weather.get("air_temp", pd.Series(config.defaults.air_temp_c, index=idx)), errors="coerce")
        .fillna(config.defaults.air_temp_c)
    )
    wind_speed = (
        pd.to_numeric(
            df_weather.get("wind_speed", pd.Series(config.defaults.wind_speed_ms, index=idx)), errors="coerce"
        )
        .fillna(config.defaults.wind_speed_ms)
    )

    # Clip irradiance to sanity limits
    ghi = ghi.clip(lower=0.0, upper=config.station.sanity_max_ghi_wm2)
    poa_measured = poa_measured.clip(lower=0.0, upper=config.station.sanity_max_poa_wm2)

    # Solar position (used for Perez transposition and AOI)
    solpos = loc.get_solarposition(idx)

    sapm_params = _SAPM_PARAMS.get(config.thermal_model, _DEFAULT_SAPM)

    total_dc_kw = pd.Series(0.0, index=idx)

    for orient in config.orientations:
        # ── Step 1: POA irradiance ────────────────────────────────────────
        use_poa = orient.use_measured_poa and poa_measured.notna().any()

        if use_poa:
            poa_global = poa_measured.fillna(0.0)
            poa_diffuse = pd.Series(0.0, index=idx)  # unknown split; AOI skipped
            aoi = pd.Series(0.0, index=idx)  # assume normal incidence
            log.debug("orientation %s: using measured POA", orient.name)
        else:
            poa_irrad = pvlib.irradiance.get_total_irradiance(
                surface_tilt=orient.tilt,
                surface_azimuth=orient.azimuth + 180,  # pvlib: 180=South
                dni=_dni_from_ghi(ghi, solpos),
                ghi=ghi,
                dhi=_dhi_from_ghi(ghi, solpos),
                solar_zenith=solpos["apparent_zenith"],
                solar_azimuth=solpos["azimuth"],
                albedo=config.albedo,
                model="perez",
            )
            poa_global = poa_irrad["poa_global"].fillna(0.0).clip(lower=0.0)
            poa_diffuse = poa_irrad["poa_diffuse"].fillna(0.0)
            aoi = pvlib.irradiance.aoi(
                orient.tilt,
                orient.azimuth + 180,
                solpos["apparent_zenith"],
                solpos["azimuth"],
            ).fillna(90.0)

        # ── Step 2: Far shading (pre-IAM) ─────────────────────────────────
        poa_global = poa_global * config.far_shading

        # ── Step 3: SAPM cell temperature ──────────────────────────────────
        t_cell = pvlib.temperature.sapm_cell(
            poa_global=poa_global,
            temp_air=air_temp,
            wind_speed=wind_speed,
            a=sapm_params["a"],
            b=sapm_params["b"],
            deltaT=sapm_params["deltaT"],
        )

        # ── Step 4: IAM correction ─────────────────────────────────────────
        if use_poa:
            iam_factor = pd.Series(1.0, index=idx)
            poa_eff = poa_global
        else:
            iam_factor = _iam(aoi, config.iam)
            # Apply IAM only to beam component; diffuse/albedo pass through
            poa_beam = (poa_global - poa_diffuse).clip(lower=0.0)
            poa_eff = poa_beam * iam_factor + poa_diffuse

        # ── Step 5: DC power (pre-loss) ────────────────────────────────────
        n_modules = orient.module_count
        total_area_m2 = n_modules * config.module.area_m2
        temp_coeff = 1.0 + config.module.gamma_p * (t_cell - 25.0)
        temp_coeff = temp_coeff.clip(lower=0.1)  # sanity: never negative

        pdc_kw = (poa_eff / 1000.0) * total_area_m2 * config.module.efficiency_stc * temp_coeff

        total_dc_kw += pdc_kw.fillna(0.0)

    # ── Step 6: DC losses (sequential multiplication) ────────────────────
    dc_loss_factor = (
        (1.0 - config.soiling)
        * (1.0 - config.lid)
        * (1.0 + config.module_quality)  # negative module_quality = gain
        * (1.0 - config.mismatch)
        * (1.0 - config.dc_wiring)
    )
    total_dc_kw = (total_dc_kw * dc_loss_factor).clip(lower=0.0)

    # ── Step 7: Inverter efficiency ────────────────────────────────────────
    inv = config.inverter
    if inv.use_efficiency_curve and len(inv.efficiency_curve_kw) >= 2:
        eta_inv = _interp_efficiency(total_dc_kw, inv)
    else:
        eta_inv = pd.Series(inv.flat_efficiency, index=idx)

    pac_kw = (total_dc_kw * eta_inv).fillna(0.0)

    # ── Step 8: AC rating clip ─────────────────────────────────────────────
    pac_kw = pac_kw.clip(upper=inv.ac_rating_kw, lower=0.0)

    # ── Step 9: AC wiring loss ────────────────────────────────────────────
    pac_kw = pac_kw * (1.0 - config.ac_wiring)

    # Zero out night-time (Gap 25: use effective irradiance so POA-only stations work correctly)
    # For POA-only stations, ghi is zero but poa_global is the real signal.
    # Use the effective irradiance that was actually computed in the orientation loop.
    # poa_global is the last orientation's value; for a single-orientation plant this is correct.
    # For multi-orientation plants, any orientation with POA > 1 W/m² should prevent zeroing.
    effective_irr = poa_measured.fillna(0.0) if poa_measured.notna().any() else ghi
    night_mask = effective_irr.fillna(0.0) <= 1.0   # sub-threshold (accounts for sensor noise)
    pac_kw[night_mask] = 0.0

    return pd.DataFrame(
        {
            "potential_power_kw": pac_kw,
            "active_power_pvlib_kw": pac_kw,
            "data_source": data_source,
            "model_version": "pvlib-h-a3-v1",
        },
        index=idx,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Helper functions
# ─────────────────────────────────────────────────────────────────────────────


def _empty_result(index: pd.DatetimeIndex, data_source: str) -> pd.DataFrame:
    """Return empty result DataFrame with correct columns and index."""
    return pd.DataFrame(
        {
            "potential_power_kw": pd.Series(dtype=float),
            "active_power_pvlib_kw": pd.Series(dtype=float),
            "data_source": pd.Series(dtype=str),
            "model_version": pd.Series(dtype=str),
        },
        index=index,
    )


def _iam(aoi: pd.Series, iam_cfg: IAMConfig) -> pd.Series:
    """Interpolate IAM from user-defined angle/value profile.

    Parameters
    ----------
    aoi : pd.Series
        Angle of Incidence in degrees.
    iam_cfg : IAMConfig
        User-defined IAM profile (angles and factors).

    Returns
    -------
    pd.Series
        IAM factor (0.0 to 1.0) at each AOI.
    """
    iam = pvlib.iam.interp(aoi, iam_cfg.angles, iam_cfg.values, method="linear")
    return pd.Series(iam, index=aoi.index).clip(lower=0.0, upper=1.0)


def _interp_efficiency(pdc_kw: pd.Series, inv: InverterConfig) -> pd.Series:
    """Interpolate inverter efficiency from digitized curve.

    Parameters
    ----------
    pdc_kw : pd.Series
        DC power input in kilowatts.
    inv : InverterConfig
        Inverter configuration with efficiency curve.

    Returns
    -------
    pd.Series
        Efficiency (0.0 to 1.0) at each DC power.
    """
    eta = np.interp(
        pdc_kw.values,
        inv.efficiency_curve_kw,
        inv.efficiency_curve_eta,
        left=inv.efficiency_curve_eta[0] if inv.efficiency_curve_eta else 0.0,
        right=inv.efficiency_curve_eta[-1] if inv.efficiency_curve_eta else 0.0,
    )
    return pd.Series(eta, index=pdc_kw.index).clip(lower=0.0, upper=1.0)


def _decompose_ghi(ghi: pd.Series, solpos: pd.DataFrame) -> tuple[pd.Series, pd.Series]:
    """Decompose GHI into DNI and DHI using Erbs model.

    Parameters
    ----------
    ghi : pd.Series
        Global Horizontal Irradiance (W/m²).
    solpos : pd.DataFrame
        Solar position with 'zenith' column.

    Returns
    -------
    tuple[pd.Series, pd.Series]
        (DNI, DHI) series.
    """
    result = pvlib.irradiance.erbs(ghi, solpos["zenith"], ghi.index)
    return (
        result["dni"].fillna(0.0).clip(lower=0.0),
        result["dhi"].fillna(0.0).clip(lower=0.0),
    )


def _dni_from_ghi(ghi: pd.Series, solpos: pd.DataFrame) -> pd.Series:
    """Extract DNI from decomposed GHI.

    Parameters
    ----------
    ghi : pd.Series
        Global Horizontal Irradiance (W/m²).
    solpos : pd.DataFrame
        Solar position DataFrame.

    Returns
    -------
    pd.Series
        Direct Normal Irradiance (W/m²).
    """
    dni, _ = _decompose_ghi(ghi, solpos)
    return dni


def _dhi_from_ghi(ghi: pd.Series, solpos: pd.DataFrame) -> pd.Series:
    """Extract DHI from decomposed GHI.

    Parameters
    ----------
    ghi : pd.Series
        Global Horizontal Irradiance (W/m²).
    solpos : pd.DataFrame
        Solar position DataFrame.

    Returns
    -------
    pd.Series
        Diffuse Horizontal Irradiance (W/m²).
    """
    _, dhi = _decompose_ghi(ghi, solpos)
    return dhi
