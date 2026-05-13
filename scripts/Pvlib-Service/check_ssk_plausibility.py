"""
SSK Timeseries Plausibility Check
Verifies that the potential_power values in ssk_timeseries.csv are 
consistent with the plant config in ssk.txt and the physics pipeline.
"""
import math

# ── Plant config from ssk.txt ─────────────────────────────────────────────────
module_area_m2   = 2.7012
efficiency_stc   = 0.235
gamma_p          = -0.0029
module_count     = 22040   # from orientations
tilt             = 0       # flat mount (0° tilt)
ac_rating_kw     = 10200  # inverter ac_rating_kw

# Losses
soiling          = 0.03
lid              = 0.0
module_quality   = -0.008   # gain
mismatch         = 0.021
dc_wiring        = 0.015
ac_wiring        = 0.0

# Thermal model: open_rack_glass_glass
a, b, deltaT     = -3.47, -0.0594, 3
defaults_air_temp_c = 28.43
defaults_wind_ms    = 1.0

# ── Reproduce pipeline at a high-irradiance sample ────────────────────────────
# Peak sample from CSV: 12:14 → POA ~984 W/m²
# Since tilt=0, ghi ≈ poa for a flat array
# station uses poa_key='wstn1_tilted_irradiance' which equals ghi_key
# → PlantConfig._normalize_poa_only_station() sets ghi_key=None
# → use_poa=True path is taken (poa is not NaN, ghi_key is None so no GHI)
# So: poa_global = poa_measured (directly)

print("=" * 60)
print("SSK Pvlib-Service Plausibility Check")
print("=" * 60)

def check_sample(label, poa_wm2, air_temp=28.43, wind_ms=1.0):
    # Step 1: POA (use_poa path — ghi_key=None after normalization)
    poa_global = poa_wm2  # direct

    # Step 2: Far shading = 1 (no loss)
    poa_global *= 1.0

    # Step 3: SAPM cell temperature
    t_cell = poa_global * math.exp(a + b * wind_ms) + air_temp + (poa_global / 1000) * deltaT
    
    # Step 4: IAM = 1.0 (use_poa path skips AOI correction)
    poa_eff = poa_global

    # Step 5: DC power
    total_area = module_count * module_area_m2  # m²
    temp_coeff = 1.0 + gamma_p * (t_cell - 25.0)
    temp_coeff = max(temp_coeff, 0.1)
    pdc_kw = (poa_eff / 1000.0) * total_area * efficiency_stc * temp_coeff

    # Step 6: DC losses
    dc_loss_factor = (
        (1.0 - soiling)
        * (1.0 - lid)
        * (1.0 + module_quality)   # negative = gain
        * (1.0 - mismatch)
        * (1.0 - dc_wiring)
    )
    pdc_after_loss_kw = max(pdc_kw * dc_loss_factor, 0.0)

    # Step 7: Inverter — use_efficiency_curve=True
    # Efficiency curve from ssk.txt inverter attribute (truncated in display)
    # We'll use a representative ~98% efficiency for now (typical at high load)
    # The actual curve is in TB; we use 0.98 as approximation
    eta_inv = 0.98
    pac_kw = pdc_after_loss_kw * eta_inv

    # Step 8: AC clip at 10200 kW
    pac_kw = min(pac_kw, ac_rating_kw)

    # Step 9: AC wiring = 0 (no loss)
    pac_kw *= (1.0 - ac_wiring)

    print(f"\n--- {label} ---")
    print(f"  POA irradiance:       {poa_wm2:.1f} W/m²")
    print(f"  Total array area:     {total_area:.0f} m² ({module_count} × {module_area_m2} m²)")
    print(f"  Cell temperature:     {t_cell:.1f} °C")
    print(f"  Temp coefficient:     {temp_coeff:.4f}")
    print(f"  PDC (pre-loss):       {pdc_kw:.1f} kW")
    print(f"  DC loss factor:       {dc_loss_factor:.4f}  ({(1-dc_loss_factor)*100:.2f}% total loss)")
    print(f"  PDC (post-loss):      {pdc_after_loss_kw:.1f} kW")
    print(f"  PAC (after inv η):    {pac_kw:.1f} kW")
    print(f"  AC clip limit:        {ac_rating_kw} kW")
    return pac_kw

# ── Sample checks ─────────────────────────────────────────────────────────────
# Peak noon sample: POA ~984 W/m², potential_power in CSV = 1247 kW  ← SUSPICIOUS
p1 = check_sample("Noon peak (12:14) — POA 984 W/m²", poa_wm2=984.1)
print(f"  CSV potential_power:  ~1247 kW")
print(f"  ✓ Match?  {'YES' if abs(p1 - 1247) < 50 else 'NO — MISMATCH'} (model={p1:.1f} kW vs csv=1247 kW)")

# Mid-afternoon: POA ~760 W/m², potential_power in CSV = ~973 kW
p2 = check_sample("Afternoon (14:17) — POA 761 W/m²", poa_wm2=761.1)
print(f"  CSV potential_power:  ~973 kW")
print(f"  ✓ Match?  {'YES' if abs(p2 - 973) < 50 else 'NO — MISMATCH'} (model={p2:.1f} kW vs csv=973 kW)")

# Late afternoon: POA ~437 W/m², potential_power in CSV = ~648 kW
p3 = check_sample("Late afternoon (15:57) — POA 437 W/m²", poa_wm2=437.6)
print(f"  CSV potential_power:  ~648 kW")
print(f"  ✓ Match?  {'YES' if abs(p3 - 648) < 50 else 'NO — MISMATCH'} (model={p3:.1f} kW vs csv=648 kW)")

# Low irradiance: POA ~98 W/m², potential_power in CSV = 204 kW
p4 = check_sample("End of day (17:28) — POA 98.8 W/m²", poa_wm2=98.8)
print(f"  CSV potential_power:  ~204 kW")
print(f"  ✓ Match?  {'YES' if abs(p4 - 204) < 30 else 'NO — MISMATCH'} (model={p4:.1f} kW vs csv=204 kW)")

print("\n" + "=" * 60)
print("RATIO ANALYSIS (CSV potential_power / POA)")
print("=" * 60)
samples = [
    ("12:02", 868.0, 1111.393),
    ("12:08", 965.3, 1236.935),
    ("12:13", 958.0, 1227.539),
    ("12:18", 973.4, 1247.362),
    ("12:23", 973.2, 1247.105),
    ("12:27", 953.8, 1222.133),
    ("13:08", 835.1, 1068.961),
    ("14:03", 701.8, 897.112),
    ("14:17", 761.1, 973.556),
    ("15:02", 640.3, 817.741),
    ("15:23", 564.4, 719.824),
    ("15:57", 437.6, 648.372),
    ("16:07", 401.3, 547.056),
    ("17:07", 179.2, 354.232),
    ("17:28", 98.8, 204.081),
]
print(f"{'Time':<6} {'POA (W/m²)':<12} {'CSV kW':<10} {'Ratio kW/POA':<14} {'Expected kW':<12}")
print("-" * 60)
expected_ratio = None
for time, poa, csv_kw in samples:
    ratio = csv_kw / poa if poa > 0 else 0
    # Simple model estimate
    total_area = module_count * module_area_m2
    # rough estimate: pdc = (poa/1000) * area * eff * temp_correction
    t_cell_est = poa * math.exp(a + b * defaults_wind_ms) + defaults_air_temp_c + (poa/1000)*deltaT
    temp_c = 1 + gamma_p * (t_cell_est - 25)
    dc_loss = (1-soiling)*(1-lid)*(1+module_quality)*(1-mismatch)*(1-dc_wiring)
    est = (poa/1000) * total_area * efficiency_stc * temp_c * dc_loss * 0.98
    est = min(est, ac_rating_kw)
    print(f"{time:<6} {poa:<12.1f} {csv_kw:<10.1f} {ratio:<14.4f} {est:<12.1f}")

print("\n" + "=" * 60)
print("KEY FINDINGS")
print("=" * 60)
total_area = module_count * module_area_m2
print(f"  Plant capacity (rated):      10,000 kWp (10 MW)")
print(f"  Inverter AC rating:          {ac_rating_kw} kW")
print(f"  Module count:                {module_count}")
print(f"  Module area:                 {module_area_m2} m²")
print(f"  Total array area:            {total_area:.0f} m²")
print(f"  STC DC peak (no losses):     {total_area * efficiency_stc:.0f} kW")
dc_loss_factor = (1-soiling)*(1-lid)*(1+module_quality)*(1-mismatch)*(1-dc_wiring)
print(f"  DC loss factor:              {dc_loss_factor:.4f}")
print(f"  Net DC at STC (after loss):  {total_area * efficiency_stc * dc_loss_factor:.0f} kW")
print(f"  Expected PAC at STC (~1000 W/m²): {min(total_area * efficiency_stc * dc_loss_factor * 0.98, ac_rating_kw):.0f} kW")
print(f"\n  NOTE: Active power in CSV peaks at ~10,655 kW → clipped to inverter rating")
print(f"  NOTE: potential_power peaks at ~1,247 kW → MUCH lower than 10,000 kWp")
print(f"  ⚠ ISSUE: potential_power appears to be in kW but seems ~8x too low vs active_power")
print(f"         active_power peaks ~10,500 kW, potential_power peaks ~1,247 kW")
print(f"         Ratio: {10500/1247:.1f}x  — this is the known SSK bug")
