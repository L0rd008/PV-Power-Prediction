"""
set_active_power_unit.py — DEPRECATED

This script is no longer maintained.

Use ``tb_config_loader.py`` instead, which sets ``active_power_unit`` (and all
other SERVER_SCOPE plant attributes) as part of a single idempotent workflow
driven by ``plants_master.yml``.

See ONBOARDING_GUIDE.md for the current onboarding procedure.

This file will be removed after the 90-day deprecation window (≥ 2026-08-14).
"""
raise SystemExit(
    "DEPRECATED: set_active_power_unit.py is no longer supported.\n"
    "Use tb_config_loader.py instead — see ONBOARDING_GUIDE.md."
)
