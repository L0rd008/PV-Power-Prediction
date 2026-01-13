// =====================================================================
// CALIBRATED PARAMETERS (optimized to match Python pvlib model)
// =====================================================================
// Generated: 2026-01-13T12:25:23.046172
// DO NOT EDIT MANUALLY - regenerate using optimize_js_params.py

var CALIBRATION = {
    // Perez POA approximation
    circumsolar_factor: 0.05,    // Circumsolar brightening multiplier
    circumsolar_threshold: 10.02,   // DNI threshold for circumsolar (W/m²)
    aoi_threshold: 0.05,          // AOI cosine threshold
    diffuse_weight: 0.9141,          // Sky view factor adjustment
    brightness_factor: 0.0,      // dni_extra effect on brightness
    
    // SAPM thermal model
    sapm_a: -2.0,              // Irradiance coefficient
    sapm_b: -0.03,            // Wind coefficient
    sapm_dt: 5.0                // Temperature offset
};

// Extraterrestrial radiation calculation (Spencer's formula)
function calcDniExtra(timestamp) {
    var d = new Date(timestamp);
    var start = new Date(d.getFullYear(), 0, 0);
    var diff = d - start;
    var dayOfYear = Math.floor(diff / 86400000);
    var b = 2 * Math.PI * dayOfYear / 365;
    return 1367 * (1.00011 + 0.034221 * Math.cos(b) + 0.00128 * Math.sin(b)
                  + 0.000719 * Math.cos(2*b) + 0.000077 * Math.sin(2*b));
}
