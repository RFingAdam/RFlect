"""Frequency extrapolation of antenna patterns (#38).

Extracted from the oversized ``calculations.py`` as the first of several planned
leaf-module splits. These two functions depend only on numpy (no other
``calculations`` internals and no ``file_utils`` — the former ``read_passive_file``
reference was only in a docstring), so the extraction is free of circular-import
risk. ``calculations`` re-exports both names, so existing
``from .calculations import extrapolate_pattern`` imports keep working.
"""

from __future__ import annotations

import numpy as np


def extrapolate_pattern(
    hpol_data,
    vpol_data,
    target_frequency,
    fit_degree=2,
    min_frequencies=5,
):
    """
    Extrapolate antenna pattern to a target frequency using polynomial fitting.

    For each spatial point (theta, phi), fits gain-vs-frequency and phase-vs-frequency
    curves across the measured band, then evaluates at the target frequency.

    Parameters:
        hpol_data (list): List of dicts from read_passive_file(), each with
                          'frequency', 'theta', 'phi', 'mag', 'phase'.
        vpol_data (list): Matched VPOL data (same structure).
        target_frequency (float): Target frequency in MHz.
        fit_degree (int): Polynomial order for magnitude fitting (default 2).
        min_frequencies (int): Minimum number of frequency points required (default 5).

    Returns:
        dict with keys:
            'hpol': dict with 'frequency', 'theta', 'phi', 'mag', 'phase'
            'vpol': same structure
            'confidence': dict with 'extrapolation_ratio', 'mean_r_squared',
                          'max_estimated_error_dB', 'quality', 'warning'
            'is_extrapolated': True

    Raises:
        ValueError: If fewer than min_frequencies data points available.
    """
    if len(hpol_data) < min_frequencies:
        raise ValueError(
            f"Need at least {min_frequencies} frequency points for extrapolation, "
            f"got {len(hpol_data)}"
        )

    # Build frequency array from measured data
    freqs = np.array([d["frequency"] for d in hpol_data])
    measured_min = float(np.min(freqs))
    measured_max = float(np.max(freqs))
    measured_bw = measured_max - measured_min

    # Number of spatial points (from the first entry)
    n_points = len(hpol_data[0]["mag"])

    # Output arrays
    h_mag_out = np.zeros(n_points)
    h_phase_out = np.zeros(n_points)
    v_mag_out = np.zeros(n_points)
    v_phase_out = np.zeros(n_points)

    r_squared_list = []

    for i in range(n_points):
        # --- HPOL magnitude ---
        h_gains = np.array([d["mag"][i] for d in hpol_data])
        h_coeffs = np.polyfit(freqs, h_gains, fit_degree)
        h_mag_out[i] = np.clip(np.polyval(h_coeffs, target_frequency), -60.0, 30.0)

        # R² for HPOL magnitude fit
        h_fitted = np.polyval(h_coeffs, freqs)
        ss_res = np.sum((h_gains - h_fitted) ** 2)
        ss_tot = np.sum((h_gains - np.mean(h_gains)) ** 2)
        r2_h = 1.0 - ss_res / ss_tot if ss_tot > 0 else 1.0
        r_squared_list.append(r2_h)

        # --- HPOL phase (linear fit on unwrapped phase) ---
        h_phases = np.array([d["phase"][i] for d in hpol_data])
        h_phases_rad = np.unwrap(np.deg2rad(h_phases))
        h_phase_coeffs = np.polyfit(freqs, h_phases_rad, 1)
        h_phase_out[i] = np.rad2deg(np.polyval(h_phase_coeffs, target_frequency))

        # --- VPOL magnitude ---
        v_gains = np.array([d["mag"][i] for d in vpol_data])
        v_coeffs = np.polyfit(freqs, v_gains, fit_degree)
        v_mag_out[i] = np.clip(np.polyval(v_coeffs, target_frequency), -60.0, 30.0)

        # R² for VPOL magnitude fit
        v_fitted = np.polyval(v_coeffs, freqs)
        ss_res_v = np.sum((v_gains - v_fitted) ** 2)
        ss_tot_v = np.sum((v_gains - np.mean(v_gains)) ** 2)
        r2_v = 1.0 - ss_res_v / ss_tot_v if ss_tot_v > 0 else 1.0
        r_squared_list.append(r2_v)

        # --- VPOL phase (linear fit on unwrapped phase) ---
        v_phases = np.array([d["phase"][i] for d in vpol_data])
        v_phases_rad = np.unwrap(np.deg2rad(v_phases))
        v_phase_coeffs = np.polyfit(freqs, v_phases_rad, 1)
        v_phase_out[i] = np.rad2deg(np.polyval(v_phase_coeffs, target_frequency))

    # Confidence metrics
    nearest_measured = float(freqs[np.argmin(np.abs(freqs - target_frequency))])
    distance_to_nearest = abs(target_frequency - nearest_measured)
    extrapolation_ratio = distance_to_nearest / measured_bw if measured_bw > 0 else 1.0
    mean_r2 = float(np.mean(r_squared_list))

    # Estimate max error from residuals scaled by extrapolation distance
    max_estimated_error = distance_to_nearest * (1.0 - mean_r2) * 10.0  # heuristic dB

    if extrapolation_ratio < 0.25:
        quality = "good"
        warning = None
    elif extrapolation_ratio < 0.50:
        quality = "fair"
        warning = "Moderate extrapolation — verify against nearby measurements."
    elif extrapolation_ratio < 0.75:
        quality = "poor"
        warning = "Large extrapolation distance — results may be unreliable."
    else:
        quality = "unreliable"
        warning = (
            "Extrapolation exceeds 75% of measured bandwidth — "
            "treat results as rough estimates only."
        )

    # Use theta/phi from the first entry (same grid for all frequencies)
    return {
        "hpol": {
            "frequency": target_frequency,
            "theta": list(hpol_data[0]["theta"]),
            "phi": list(hpol_data[0]["phi"]),
            "mag": h_mag_out.tolist(),
            "phase": h_phase_out.tolist(),
        },
        "vpol": {
            "frequency": target_frequency,
            "theta": list(vpol_data[0]["theta"]),
            "phi": list(vpol_data[0]["phi"]),
            "mag": v_mag_out.tolist(),
            "phase": v_phase_out.tolist(),
        },
        "confidence": {
            "extrapolation_ratio": round(extrapolation_ratio, 4),
            "mean_r_squared": round(mean_r2, 4),
            "max_estimated_error_dB": round(max_estimated_error, 2),
            "quality": quality,
            "warning": warning,
        },
        "is_extrapolated": True,
    }


def validate_extrapolation(
    hpol_data,
    vpol_data,
    holdout_frequency,
    fit_degree=2,
):
    """
    Validate extrapolation accuracy by holding out a known frequency.

    Removes holdout_frequency from both datasets, runs extrapolate_pattern(),
    and compares with the actual measurement.

    Parameters:
        hpol_data (list): Full list of HPOL frequency entries.
        vpol_data (list): Full list of VPOL frequency entries.
        holdout_frequency (float): Frequency to hold out and predict.
        fit_degree (int): Polynomial order for magnitude fitting.

    Returns:
        dict with 'holdout_frequency', 'rms_error_dB', 'max_error_dB', 'mean_error_dB'

    Raises:
        ValueError: If holdout_frequency not found in data.
    """
    # Find and remove the holdout frequency
    h_holdout = None
    v_holdout = None
    h_train = []
    v_train = []

    for h, v in zip(hpol_data, vpol_data):
        if np.isclose(h["frequency"], holdout_frequency, atol=0.5):
            h_holdout = h
            v_holdout = v
        else:
            h_train.append(h)
            v_train.append(v)

    if h_holdout is None:
        raise ValueError(f"Holdout frequency {holdout_frequency} MHz not found in data.")

    # Run extrapolation on training data
    result = extrapolate_pattern(
        h_train,
        v_train,
        holdout_frequency,
        fit_degree=fit_degree,
        min_frequencies=max(5, len(h_train)),
    )

    # Compare extrapolated vs actual (using total gain = HPOL + VPOL in linear)
    actual_h = np.array(h_holdout["mag"])
    actual_v = np.array(v_holdout["mag"])
    extrap_h = np.array(result["hpol"]["mag"])
    extrap_v = np.array(result["vpol"]["mag"])

    # Errors per polarization
    h_errors = extrap_h - actual_h
    v_errors = extrap_v - actual_v
    all_errors = np.concatenate([h_errors, v_errors])

    return {
        "holdout_frequency": holdout_frequency,
        "rms_error_dB": float(np.sqrt(np.mean(all_errors**2))),
        "max_error_dB": float(np.max(np.abs(all_errors))),
        "mean_error_dB": float(np.mean(np.abs(all_errors))),
        "confidence": result["confidence"],
    }
