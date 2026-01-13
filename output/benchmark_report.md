# Model Benchmark Report

Generated: 2026-01-12 12:27:34

## Summary

Comparison of three PV prediction models against actual Meteocontrol generation data.

**Comparison Period:** December 11-15, 2025 (5 complete days)
**Timezone:** Asia/Colombo (UTC+5:30)

## Results

| Model | MAE (kWh) | MAPE (%) | Bias (%) | Total Error (%) | Avg Time (ms) |
|-------|-----------|----------|----------|-----------------|---------------|
| Python Physics | 154.1 | 56.3 | 42.3 | 239.9 | 0.740 |
| JS Physics | 169.0 | 66.0 | 53.0 | 268.2 | 0.026 |
| JS Surrogate | 151.3 | 56.5 | 43.4 | 238.3 | 0.011 |

## Daily Energy Comparison (kWh)

| Date | Actual | Python Physics | JS Physics | JS Surrogate |
|------|--------|----------------|------------|--------------|
| 2025-12-11 | 345.4 | 224.5 | 233.3 | 232.6 |
| 2025-12-12 | 76.0 | 263.4 | 301.9 | 265.8 |

## Notes

- **Python Physics**: Full pvlib model with Perez POA transposition, SAPM thermal model
- **JS Physics**: Simplified Perez approximation, designed for ThingsBoard
- **JS Surrogate**: Regression-based model (coefficients may need training)
- All timestamps converted to Asia/Colombo before daily aggregation
- Only complete days (Dec 11-15) included in comparison
