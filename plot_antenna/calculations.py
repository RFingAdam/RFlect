# calculations.py

import numpy as np
from scipy.signal import windows


def calculate_trp(power_dBm_2d, theta_angles_rad, inc_theta, inc_phi):
    """
    Calculate Total Radiated Power (TRP) using IEEE-correct solid angle integration.

    The TRP is calculated by integrating radiated power density over a closed sphere:
        TRP = (1/4π) ∫∫ P(θ,φ) · sin(θ) dθ dφ

    For discrete measurements, this becomes:
        TRP = Σ P(θ,φ) · sin(θ) · Δθ · Δφ / (4π)

    Parameters:
        power_dBm_2d: 2D array of power values in dBm (theta x phi)
        theta_angles_rad: 1D array of theta angles in radians
        inc_theta: Theta increment in degrees
        inc_phi: Phi increment in degrees

    Returns:
        TRP_dBm: Total radiated power in dBm
    """
    # Convert power from dBm to linear (mW)
    power_mW = 10 ** (power_dBm_2d / 10)

    # sin(θ) weighting for spherical coordinate Jacobian
    theta_weight = np.sin(theta_angles_rad)

    # IEEE-correct solid angle integration
    # TRP = sum(P * sin(θ)) * dθ * dφ / (4π)
    dtheta = np.deg2rad(inc_theta)
    dphi = np.deg2rad(inc_phi)
    TRP_mW = np.sum(power_mW * theta_weight[:, np.newaxis]) * dtheta * dphi / (4 * np.pi)

    TRP_dBm = 10 * np.log10(np.maximum(TRP_mW, 1e-12))  # Protect against log(0)
    return TRP_dBm


def calculate_active_variables(
    start_phi, stop_phi, start_theta, stop_theta, inc_phi, inc_theta, h_power_dBm, v_power_dBm
):
    theta_points = int((stop_theta - start_theta) / inc_theta + 1)
    phi_points = int((stop_phi - start_phi) / inc_phi + 1)
    data_points = theta_points * phi_points

    # Calculate theta and phi angles in degrees
    theta_angles_deg = np.linspace(start_theta, stop_theta, theta_points)
    phi_angles_deg = np.linspace(start_phi, stop_phi, phi_points)

    # Reshape data into 2D arrays for calculations
    h_power_dBm_2d = h_power_dBm.reshape((theta_points, phi_points))
    v_power_dBm_2d = v_power_dBm.reshape((theta_points, phi_points))

    # Convert angles to radians for calculations
    theta_angles_rad = np.deg2rad(theta_angles_deg)
    phi_angles_rad = np.deg2rad(phi_angles_deg)

    # Calculate total power in dBm
    total_power_dBm_2d = 10 * np.log10(10 ** (v_power_dBm_2d / 10) + 10 ** (h_power_dBm_2d / 10))

    # Calculate TRP using IEEE solid angle integration
    TRP_dBm = calculate_trp(total_power_dBm_2d, theta_angles_rad, inc_theta, inc_phi)
    h_TRP_dBm = calculate_trp(h_power_dBm_2d, theta_angles_rad, inc_theta, inc_phi)
    v_TRP_dBm = calculate_trp(v_power_dBm_2d, theta_angles_rad, inc_theta, inc_phi)

    # For plotting, create extended arrays
    phi_angles_deg_plot = np.append(phi_angles_deg, 360)
    phi_angles_rad_plot = np.deg2rad(phi_angles_deg_plot)

    h_power_dBm_2d_plot = np.hstack((h_power_dBm_2d, h_power_dBm_2d[:, [0]]))
    v_power_dBm_2d_plot = np.hstack((v_power_dBm_2d, v_power_dBm_2d[:, [0]]))
    total_power_dBm_2d_plot = np.hstack((total_power_dBm_2d, total_power_dBm_2d[:, [0]]))

    # Calculate min and nominal values for plotting
    total_power_dBm_min = np.min(total_power_dBm_2d)
    total_power_dBm_nom = total_power_dBm_2d_plot - total_power_dBm_min
    h_power_dBm_min = np.min(h_power_dBm_2d)
    h_power_dBm_nom = h_power_dBm_2d_plot - h_power_dBm_min
    v_power_dBm_min = np.min(v_power_dBm_2d)
    v_power_dBm_nom = v_power_dBm_2d_plot - v_power_dBm_min

    # Return both original and extended arrays
    return (
        data_points,
        theta_angles_deg,
        phi_angles_deg,
        theta_angles_rad,
        phi_angles_rad,
        total_power_dBm_2d,
        h_power_dBm_2d,
        v_power_dBm_2d,
        phi_angles_deg_plot,
        phi_angles_rad_plot,
        total_power_dBm_2d_plot,
        h_power_dBm_2d_plot,
        v_power_dBm_2d_plot,
        total_power_dBm_min,
        total_power_dBm_nom,
        h_power_dBm_min,
        h_power_dBm_nom,
        v_power_dBm_min,
        v_power_dBm_nom,
        TRP_dBm,
        h_TRP_dBm,
        v_TRP_dBm,
    )


# _____________Passive Calculation Functions___________
# Auto Determine Polarization for HPOL & VPOL Files
def determine_polarization(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        content = f.read()
        if "Horizontal Polarization" in content:
            return "HPol"
        else:
            return "VPol"


# Verify angle data and frequencies are not mismatched
def angles_match(
    start_phi_h,
    stop_phi_h,
    inc_phi_h,
    start_theta_h,
    stop_theta_h,
    inc_theta_h,
    start_phi_v,
    stop_phi_v,
    inc_phi_v,
    start_theta_v,
    stop_theta_v,
    inc_theta_v,
):

    return bool(
        np.isclose(start_phi_h, start_phi_v)
        and np.isclose(stop_phi_h, stop_phi_v)
        and np.isclose(inc_phi_h, inc_phi_v)
        and np.isclose(start_theta_h, start_theta_v)
        and np.isclose(stop_theta_h, stop_theta_v)
        and np.isclose(inc_theta_h, inc_theta_v)
    )


# Extract Frequency points for selection in the drop-down menu
def extract_passive_frequencies(file_path):
    with open(file_path, "r", encoding="utf-8") as file:
        content = file.readlines()

    # Extracting frequencies
    frequencies = [
        float(line.split("=")[1].split()[0]) for line in content if "Test Frequency" in line
    ]

    return frequencies


# Calculate Total Gain Vector and add cable loss etc - Use Phase for future implementation?
def calculate_passive_variables(
    hpol_data,
    vpol_data,
    cable_loss,
    start_phi,
    stop_phi,
    inc_phi,
    start_theta,
    stop_theta,
    inc_theta,
    freq_list,
    selected_frequency,
):
    theta_points = int((stop_theta - start_theta) / inc_theta + 1)
    phi_points = int((stop_phi - start_phi) / inc_phi + 1)
    data_points = theta_points * phi_points

    theta_angles_deg = np.zeros((phi_points * theta_points, len(freq_list)))
    phi_angles_deg = np.zeros((phi_points * theta_points, len(freq_list)))
    v_gain_dB = np.zeros((phi_points * theta_points, len(freq_list)))
    h_gain_dB = np.zeros((phi_points * theta_points, len(freq_list)))
    v_phase = np.zeros((phi_points * theta_points, len(freq_list)))
    h_phase = np.zeros((phi_points * theta_points, len(freq_list)))

    for m, (hpol_entry, vpol_entry) in enumerate(zip(hpol_data, vpol_data)):
        if not np.isclose(hpol_entry.get("frequency", 0), vpol_entry.get("frequency", 0)):
            raise ValueError(
                f"Frequency mismatch at index {m}: "
                f"HPOL={hpol_entry.get('frequency')} MHz, VPOL={vpol_entry.get('frequency')} MHz"
            )
        for n, (theta_h, phi_h, mag_h, phase_h, theta_v, phi_v, mag_v, phase_v) in enumerate(
            zip(
                hpol_entry["theta"],
                hpol_entry["phi"],
                hpol_entry["mag"],
                hpol_entry["phase"],
                vpol_entry["theta"],
                vpol_entry["phi"],
                vpol_entry["mag"],
                vpol_entry["phase"],
            )
        ):
            v_gain = mag_v
            h_gain = mag_h
            v_ph = phase_v
            h_ph = phase_h

            theta_angles_deg[n, m] = theta_h
            phi_angles_deg[n, m] = phi_h
            v_gain_dB[n, m] = v_gain
            h_gain_dB[n, m] = h_gain
            v_phase[n, m] = v_ph
            h_phase[n, m] = h_ph

    cable_loss_matrix = np.ones((phi_points * theta_points, len(freq_list))) * cable_loss
    v_gain_dB += cable_loss_matrix
    h_gain_dB += cable_loss_matrix

    Total_Gain_dB = 10 * np.log10(
        np.maximum(10 ** (v_gain_dB / 10) + 10 ** (h_gain_dB / 10), 1e-12)
    )

    return theta_angles_deg, phi_angles_deg, v_gain_dB, h_gain_dB, Total_Gain_dB


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


def get_window(window_type, shape):
    """
    Generates a windowing function based on the specified type and shape.

    Parameters:
        window_type (str): Type of window ('none', 'hanning', 'hamming', etc.).
        shape (tuple): Shape of the window (rows, cols).

    Returns:
        window (2D np.array or None): Windowing matrix or None if 'none'.
    """
    if window_type.lower() == "none":
        return None
    elif window_type.lower() == "hanning":
        window_1d_theta = windows.hann(shape[0])
        window_1d_phi = windows.hann(shape[1])
    elif window_type.lower() == "hamming":
        window_1d_theta = windows.hamming(shape[0])
        window_1d_phi = windows.hamming(shape[1])
    else:
        raise ValueError(f"Unsupported window type: {window_type}")

    # Create 2D window by outer product
    window = np.outer(window_1d_theta, window_1d_phi)
    return window


# Helper function to process gain data for plotting.
def process_data(selected_data, selected_phi_angles_deg, selected_theta_angles_deg):
    """
    Helper function to process data (gain or power) for plotting using FFT-based plane wave decomposition.
    """
    # Convert angles to radians
    theta = np.deg2rad(selected_theta_angles_deg)
    phi = np.deg2rad(selected_phi_angles_deg)

    # Create a 2D grid of theta and phi values
    theta_grid, phi_grid = np.meshgrid(theta, phi, indexing="ij")

    # Perform Plane Wave Decomposition
    far_field_complex = np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(selected_data)))

    # Normalize the far-field pattern
    far_field_mag = np.abs(far_field_complex)
    far_field_phase = np.angle(far_field_complex, deg=True)

    return far_field_mag, far_field_phase


# _____________Human Body Shadowing Model___________
# Function to draw a translucent cone representing human shadow direction.
def draw_shadow_cone(ax, direction="-X", angle_deg=45, scale=1.2):
    """
    Draw a translucent cone to represent the human shadow direction.

    Args:
        ax (matplotlib 3D axis): The 3D axis to plot on.
        direction (str): '+X' or '-X' axis alignment.
        angle_deg (float): Half-angle of the cone.
        scale (float): Length of the cone in normalized units.
    """
    import numpy as np
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection

    # Define cone orientation vector
    if direction == "+X":
        tip = np.array([scale, 0, 0])
        base_normal = np.array([1, 0, 0])
    elif direction == "-X":
        tip = np.array([-scale, 0, 0])
        base_normal = np.array([-1, 0, 0])
    else:
        raise ValueError("direction must be '+X' or '-X'")

    num_points = 60
    theta = np.linspace(0, 2 * np.pi, num_points)
    cone_angle_rad = np.deg2rad(angle_deg)
    radius = scale * np.tan(cone_angle_rad)

    # Create orthogonal vectors for base circle
    v = np.array([0, 1, 0]) if direction == "+X" else np.array([0, 1, 0])
    if np.allclose(base_normal, v):
        v = np.array([0, 0, 1])
    ortho1 = np.cross(base_normal, v)
    ortho1 /= np.linalg.norm(ortho1)
    ortho2 = np.cross(base_normal, ortho1)

    # Build circular base
    circle = [tip + radius * (np.cos(t) * ortho1 + np.sin(t) * ortho2) for t in theta]

    # Build faces
    faces = [[tip, circle[i], circle[(i + 1) % num_points]] for i in range(num_points)]

    # Draw cone
    cone = Poly3DCollection(faces, color="gray", alpha=0.15)
    ax.add_collection3d(cone)


# Helper function to get tissue properties based on frequency.
def get_tissue_properties(frequency_mhz):
    """
    Returns dielectric properties of human muscle tissue as a function of frequency.

    Parameters:
        frequency_mhz (float): Frequency in MHz

    Returns:
        tuple: (epsilon_r, conductivity) where
            - epsilon_r: relative permittivity
            - conductivity: in S/m
    """
    f_ghz = frequency_mhz / 1000.0

    if f_ghz < 0.5:
        return 57.1, 0.8
    elif f_ghz < 1.0:
        return 54.0, 0.95
    elif f_ghz < 2.4:
        return 52.0, 1.3
    elif f_ghz < 5.0:
        return 48.0, 1.9
    elif f_ghz < 6.0:
        return 46.5, 2.3
    elif f_ghz < 10.0:
        return 43.5, 3.1
    else:
        return 39.0, 5.0


# Apply human body fading model in the ±X axis direction.
def apply_directional_human_shadow(
    gain_dB,
    theta_deg,
    phi_deg,
    frequency_mhz,
    target_axis="+X",
    cone_half_angle_deg=45,
    tissue_thickness_cm=3.5,
):
    """
    Apply human body fading model in the ±X axis direction.

    Args:
        gain_dB (ndarray): gain or power values (dB).
        theta_deg (ndarray): theta angles in degrees (same shape as gain).
        phi_deg (ndarray): phi angles in degrees (same shape as gain).
        frequency_mhz (float): frequency.
        target_axis (str): '+X' or '-X' direction for shadow.
        cone_half_angle_deg (float): cone radius in degrees.
        tissue_thickness_cm (float): typical depth of body path (~3.5 cm).

    Returns:
        gain_dB_modified (ndarray): attenuated gain pattern
    """
    import numpy as np
    from math import log10, pi

    shadow_theta = 90
    shadow_phi = 0 if target_axis == "+X" else 180

    theta_rad = np.deg2rad(theta_deg)
    phi_rad = np.deg2rad(phi_deg)
    shadow_theta_rad = np.deg2rad(shadow_theta)
    shadow_phi_rad = np.deg2rad(shadow_phi)

    angle_diff = np.arccos(
        np.sin(theta_rad) * np.sin(shadow_theta_rad) * np.cos(phi_rad - shadow_phi_rad)
        + np.cos(theta_rad) * np.cos(shadow_theta_rad)
    )

    in_shadow = angle_diff <= np.deg2rad(cone_half_angle_deg)

    # Lookup permittivity and conductivity
    eps_r, sigma = get_tissue_properties(frequency_mhz)
    omega = 2 * pi * frequency_mhz * 1e6
    epsilon = eps_r * 8.854e-12  # ε0
    mu = 4e-7 * pi  # μ₀ (tissue is non-magnetic)
    # General skin depth for lossy media:
    #   α = ω√(με/2) · √(√(1 + (σ/ωε)²) - 1)
    #   δ = 1/α
    loss_tangent_sq = (sigma / (omega * epsilon)) ** 2
    alpha = omega * np.sqrt(mu * epsilon / 2) * np.sqrt(np.sqrt(1 + loss_tangent_sq) - 1)
    delta = 1.0 / alpha  # skin depth

    depth = (tissue_thickness_cm / 100.0) / np.cos(angle_diff)
    depth = np.clip(depth, 0, 0.25)

    attenuation_linear = np.exp(-2 * depth / delta)
    attenuation_dB = 10 * np.log10(attenuation_linear)

    gain_dB_modified = np.array(gain_dB)
    gain_dB_modified[in_shadow] += attenuation_dB[in_shadow]

    return gain_dB_modified


# _____________Diversity Gain Calculation___________
# Function to compute Diversity Gain from Envelope Correlation Coefficient (ECC)
def diversity_gain(ecc):
    """
    Compute Diversity Gain in dB from ECC using the Vaughan-Andersen formula.
      DG = 10·√(1 − ECC²)
    ecc : array-like of envelope correlation coefficients (0 ≤ ecc < 1)
    returns: array of DG in dB
    """
    ecc = np.clip(ecc, 0, 0.9999)
    return 10.0 * np.sqrt(1.0 - np.asarray(ecc) ** 2)


# Function to compute MIMO capacity under AWGN + correlation
# This is a closed-form solution for 2x2 MIMO systems.
def capacity_awgn(ecc, snr_db):
    """
    Closed-form 2×2 MIMO ergodic capacity (b/s/Hz) under AWGN + correlation.
      C = log2(1+ρ/2·(1+ECC)) + log2(1+ρ/2·(1−ECC))
    ecc   : array of ECC (|ρ₁₂|)
    snr_db: scalar SNR in dB
    """
    rho = 10 ** (snr_db / 10.0)
    lam1 = 1 + ecc
    lam2 = 1 - ecc
    c1 = np.log2(1 + 0.5 * rho * lam1)
    c2 = np.log2(1 + 0.5 * rho * lam2)
    return c1 + c2


# Function to estimate MIMO capacity using Monte-Carlo simulation
# This is for 2x2 MIMO systems under correlated fading.
def capacity_monte_carlo(ecc, snr_db, fading="rayleigh", K=10, trials=2000):
    """
    Monte-Carlo estimate of 2×2 MIMO capacity under correlated fading.
    ecc      : scalar ECC value (|ρ₁₂|), must be a single float
    snr_db   : scalar SNR in dB
    fading   : 'rayleigh' or 'rician'
    K        : Rician K-factor (linear) if fading='rician'
    trials   : number of channel realizations
    returns  : average capacity (b/s/Hz)
    """
    ecc = float(ecc)  # Enforce scalar — 2x2 correlation matrix requires single value
    rho = 10 ** (snr_db / 10.0)
    # correlation matrix R
    R = np.array([[1, ecc], [ecc, 1]], dtype=complex)
    # sqrtm(R)
    eigv, eigvec = np.linalg.eigh(R)
    Rhalf = eigvec @ np.diag(np.sqrt(eigv)) @ eigvec.conj().T

    caps = []
    for _ in range(trials):
        # generate H0
        if fading == "rayleigh":
            H0 = (np.random.randn(2, 2) + 1j * np.random.randn(2, 2)) / np.sqrt(2)
        else:  # 'rician'
            mu = np.sqrt(K / (K + 1))
            sigma = np.sqrt(1 / (K + 1))
            LOS = np.ones((2, 2)) * mu
            scat = (np.random.randn(2, 2) + 1j * np.random.randn(2, 2)) / np.sqrt(2) * sigma
            H0 = LOS + scat

        H = H0 @ Rhalf
        M = H @ H.conj().T
        caps.append(np.log2(np.linalg.det(np.eye(2) + (rho / 2.0) * M).real))

    return np.mean(caps)


# ——— POLARIZATION ANALYSIS FUNCTIONS ——————————————————————————————
def calculate_polarization_parameters(hpol_data, vpol_data, cable_loss=0.0):
    """
    Calculate polarization parameters from HPOL and VPOL passive measurement data.

    Parameters:
        hpol_data: List of dictionaries containing HPOL measurements
                   Each dict has 'frequency', 'theta', 'phi', 'mag', 'phase'
        vpol_data: List of dictionaries containing VPOL measurements
        cable_loss: Cable loss in dB to apply to measurements

    Returns:
        results: List of dictionaries, one per frequency, containing:
            - 'frequency': Frequency in MHz
            - 'theta': Theta angles array
            - 'phi': Phi angles array
            - 'axial_ratio_dB': Axial ratio in dB (shape: theta x phi)
            - 'tilt_angle_deg': Tilt angle in degrees (shape: theta x phi)
            - 'sense': Polarization sense: +1 for LHCP, -1 for RHCP (shape: theta x phi)
            - 'cross_pol_discrimination_dB': XPD in dB (shape: theta x phi)
    """
    results = []

    for h_entry, v_entry in zip(hpol_data, vpol_data):
        if abs(h_entry["frequency"] - v_entry["frequency"]) > 0.1:
            continue  # Skip if frequencies don't match

        freq = h_entry["frequency"]

        # Extract arrays
        theta = np.array(h_entry["theta"])
        phi = np.array(h_entry["phi"])

        # Apply cable loss compensation (add back the loss) and convert to linear
        h_mag = np.array(h_entry["mag"]) + cable_loss
        h_phase = np.radians(h_entry["phase"])
        v_mag = np.array(v_entry["mag"]) + cable_loss
        v_phase = np.radians(v_entry["phase"])

        # Construct complex E-field components (E_phi and E_theta)
        # HPOL → E_φ (azimuthal component)
        # VPOL → E_θ (elevation component)
        E_phi = 10 ** (h_mag / 20) * np.exp(1j * h_phase)
        E_theta = 10 ** (v_mag / 20) * np.exp(1j * v_phase)

        # Calculate polarization parameters using Ludwig-3 definition
        # This is the standard for spherical coordinate systems

        # Phase difference
        delta = np.angle(E_phi) - np.angle(E_theta)

        # Amplitude ratio
        A_phi = np.abs(E_phi)
        A_theta = np.abs(E_theta)

        # Prevent division by zero
        epsilon = 1e-12
        A_theta = np.where(A_theta < epsilon, epsilon, A_theta)
        A_phi = np.where(A_phi < epsilon, epsilon, A_phi)

        # Axial Ratio (AR) via polarization ellipse semi-axes.
        # The full formula accounts for the phase difference δ between
        # orthogonal field components:
        #   AR = |E_major| / |E_minor|
        # where the semi-axes are derived from:
        #   a² = ½(A_θ² + A_φ² + √(A_θ⁴ + A_φ⁴ + 2A_θ²A_φ²cos(2δ)))
        #   b² = ½(A_θ² + A_φ² - √(A_θ⁴ + A_φ⁴ + 2A_θ²A_φ²cos(2δ)))
        A2_t = A_theta**2
        A2_p = A_phi**2
        discriminant = np.sqrt(
            np.maximum(A2_t**2 + A2_p**2 + 2 * A2_t * A2_p * np.cos(2 * delta), 0)
        )
        a_sq = 0.5 * (A2_t + A2_p + discriminant)
        b_sq = np.maximum(0.5 * (A2_t + A2_p - discriminant), epsilon**2)
        AR_linear = np.sqrt(a_sq / b_sq)
        axial_ratio_dB = 20 * np.log10(AR_linear)

        # Amplitude ratio for other calculations
        m = A_theta / A_phi

        # Tilt angle (τ) - orientation of polarization ellipse major axis
        # tan(2τ) = (2·m·cos(δ)) / (1 - m²)
        numerator_tilt = 2 * m * np.cos(delta)
        denominator_tilt = 1 - m**2
        tilt_angle_rad = 0.5 * np.arctan2(numerator_tilt, denominator_tilt)
        tilt_angle_deg = np.degrees(tilt_angle_rad)

        # Polarization sense (handedness)
        # For Ludwig-3 (IEEE) definition in antenna measurements:
        # RHCP (Right-Hand): E_φ leads E_θ by +90° → δ > 0, m > 0 → sense < 0
        # LHCP (Left-Hand): E_φ lags E_θ by -90° → δ < 0, m > 0 → sense > 0
        # We negate to match convention: Negative for RHCP, Positive for LHCP
        sense = -np.sign(m * np.sin(delta))

        # Cross-Polarization Discrimination (XPD)
        # XPD is the field (voltage) ratio of co-pol to cross-pol:
        #   XPD = 20·log₁₀[(AR + 1) / (AR - 1)]
        # AR_linear >= 1 always. When AR = 1 (circular), XPD → ∞ (clamped by epsilon).
        XPD_dB = 20 * np.log10((AR_linear + 1) / (AR_linear - 1 + epsilon))

        results.append(
            {
                "frequency": freq,
                "theta": theta,
                "phi": phi,
                "axial_ratio_dB": axial_ratio_dB,
                "tilt_angle_deg": tilt_angle_deg,
                "sense": sense,
                "cross_pol_discrimination_dB": XPD_dB,
                "E_theta_mag": A_theta,
                "E_phi_mag": A_phi,
                "phase_diff_deg": np.degrees(delta),
            }
        )

    return results


def export_polarization_data(pol_results, output_path, format="csv"):
    """
    Export polarization analysis results to file.

    Parameters:
        pol_results: Results from calculate_polarization_parameters
        output_path: Base path for output files
        format: 'csv' or 'txt'
    """
    import os
    import csv

    for result in pol_results:
        freq = result["frequency"]
        filename = f"polarization_{freq}MHz.{format}"
        filepath = os.path.join(output_path, filename)

        # Flatten arrays for export
        n_points = len(result["theta"])

        if format == "csv":
            with open(filepath, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(
                    [
                        "Theta(deg)",
                        "Phi(deg)",
                        "Axial_Ratio(dB)",
                        "Tilt_Angle(deg)",
                        "Sense",
                        "XPD(dB)",
                        "E_theta_mag",
                        "E_phi_mag",
                        "Phase_Diff(deg)",
                    ]
                )

                for i in range(n_points):
                    writer.writerow(
                        [
                            f"{result['theta'][i]:.2f}",
                            f"{result['phi'][i]:.2f}",
                            f"{result['axial_ratio_dB'][i]:.3f}",
                            f"{result['tilt_angle_deg'][i]:.2f}",
                            f"{result['sense'][i]:+.0f}",
                            f"{result['cross_pol_discrimination_dB'][i]:.3f}",
                            f"{result['E_theta_mag'][i]:.6f}",
                            f"{result['E_phi_mag'][i]:.6f}",
                            f"{result['phase_diff_deg'][i]:.2f}",
                        ]
                    )
        else:  # txt format
            with open(filepath, "w") as f:
                f.write(f"Polarization Analysis Results at {freq} MHz\n")
                f.write("=" * 80 + "\n\n")
                f.write(
                    f"{'Theta':>8} {'Phi':>8} {'AR(dB)':>10} {'Tilt(°)':>10} "
                    f"{'Sense':>6} {'XPD(dB)':>10}\n"
                )
                f.write("-" * 80 + "\n")

                for i in range(n_points):
                    sense_str = "LHCP" if result["sense"][i] > 0 else "RHCP"
                    f.write(
                        f"{result['theta'][i]:8.2f} {result['phi'][i]:8.2f} "
                        f"{result['axial_ratio_dB'][i]:10.3f} "
                        f"{result['tilt_angle_deg'][i]:10.2f} "
                        f"{sense_str:>6} "
                        f"{result['cross_pol_discrimination_dB'][i]:10.3f}\n"
                    )


# ——— LINK BUDGET & RANGE ESTIMATION ——————————————————————————————

# Protocol presets: {name: (rx_sensitivity_dBm, tx_power_dBm, freq_mhz)}
PROTOCOL_PRESETS = {
    "Custom": (None, None, None),
    "BLE 1Mbps": (-98.0, 0.0, 2450.0),
    "BLE 2Mbps": (-92.0, 0.0, 2450.0),
    "BLE Long Range (Coded)": (-103.0, 0.0, 2450.0),
    "WiFi 802.11n (MCS0)": (-82.0, 20.0, 2450.0),
    "WiFi 802.11ac (MCS0)": (-82.0, 20.0, 5500.0),
    "Zigbee / Thread": (-100.0, 0.0, 2450.0),
    "LoRa SF12": (-137.0, 14.0, 915.0),
    "LoRa SF7": (-123.0, 14.0, 915.0),
    "LTE Cat-M1": (-108.0, 23.0, 700.0),
    "NB-IoT": (-141.0, 23.0, 800.0),
}

# Environment presets: {name: (path_loss_n, shadow_sigma_dB, fading_model, K_factor, typical_walls)}
ENVIRONMENT_PRESETS = {
    "Free Space": (2.0, 0.0, "none", 0, 0),
    "Office": (3.0, 5.0, "rician", 6, 1),
    "Residential": (2.8, 4.0, "rician", 4, 1),
    "Commercial": (2.2, 3.0, "rician", 10, 0),
    "Hospital": (3.5, 7.0, "rayleigh", 0, 2),
    "Industrial": (3.0, 8.0, "rayleigh", 0, 1),
    "Outdoor Urban": (3.5, 6.0, "rayleigh", 0, 0),
    "Outdoor LOS": (2.0, 2.0, "rician", 15, 0),
}


def free_space_path_loss(freq_mhz, distance_m):
    """Free-space path loss (Friis) in dB.

    FSPL = 20·log10(d) + 20·log10(f) + 20·log10(4π/c)
         = 20·log10(d) + 20·log10(f) - 147.55   (d in m, f in Hz)

    Parameters:
        freq_mhz: frequency in MHz
        distance_m: distance in metres (must be > 0)

    Returns:
        Path loss in dB (positive value).
    """
    if distance_m <= 0:
        return 0.0
    freq_hz = freq_mhz * 1e6
    return 20 * np.log10(distance_m) + 20 * np.log10(freq_hz) - 147.55


def friis_range_estimate(pt_dbm, pr_dbm, gt_dbi, gr_dbi, freq_mhz,
                         path_loss_exp=2.0, misc_loss_db=0.0):
    """Solve Friis / log-distance model for maximum range.

    Allowable path loss:
        PL_max = Pt + Gt + Gr - Pr - L

    Using log-distance model with d0 = 1 m:
        PL(d) = FSPL(d0) + 10·n·log10(d/d0)
        d = d0 · 10^((PL_max - FSPL_d0) / (10·n))

    Parameters:
        pt_dbm: transmit power (dBm)
        pr_dbm: receiver sensitivity (dBm, negative value)
        gt_dbi: transmit antenna gain (dBi)
        gr_dbi: receive antenna gain (dBi)
        freq_mhz: frequency (MHz)
        path_loss_exp: path loss exponent n (2.0 = free space)
        misc_loss_db: additional system losses (dB)

    Returns:
        Maximum range in metres.
    """
    pl_max = pt_dbm + gt_dbi + gr_dbi - pr_dbm - misc_loss_db
    fspl_d0 = free_space_path_loss(freq_mhz, 1.0)  # FSPL at 1 m
    if path_loss_exp <= 0:
        return float("inf")
    exponent = (pl_max - fspl_d0) / (10.0 * path_loss_exp)
    return 10.0 ** exponent  # d0 = 1 m, so d = 10^exponent


def min_tx_gain_for_range(target_range_m, pt_dbm, pr_dbm, gr_dbi,
                          freq_mhz, path_loss_exp=2.0, misc_loss_db=0.0):
    """Solve Friis for minimum Tx antenna gain to achieve target range.

    Parameters:
        target_range_m: desired range in metres
        pt_dbm: transmit power (dBm)
        pr_dbm: receiver sensitivity (dBm)
        gr_dbi: receive antenna gain (dBi)
        freq_mhz: frequency (MHz)
        path_loss_exp: path loss exponent
        misc_loss_db: additional system losses (dB)

    Returns:
        Minimum Gt in dBi.
    """
    fspl_d0 = free_space_path_loss(freq_mhz, 1.0)
    pl_at_range = fspl_d0 + 10.0 * path_loss_exp * np.log10(max(target_range_m, 0.01))
    # Gt = PL + Pr + L - Pt - Gr
    return pl_at_range + pr_dbm + misc_loss_db - pt_dbm - gr_dbi


def link_margin(pt_dbm, gt_dbi, gr_dbi, freq_mhz, distance_m,
                path_loss_exp=2.0, misc_loss_db=0.0, pr_sensitivity_dbm=-98.0):
    """Calculate link margin at a given distance.

    Link margin = Pr_received - Pr_sensitivity
    where Pr_received = Pt + Gt + Gr - PL(d) - L

    Parameters:
        pt_dbm: transmit power (dBm)
        gt_dbi: transmit antenna gain (dBi)
        gr_dbi: receive antenna gain (dBi)
        freq_mhz: frequency (MHz)
        distance_m: distance in metres
        path_loss_exp: path loss exponent
        misc_loss_db: system losses (dB)
        pr_sensitivity_dbm: receiver sensitivity (dBm)

    Returns:
        Link margin in dB (positive = link closes, negative = link fails).
    """
    fspl_d0 = free_space_path_loss(freq_mhz, 1.0)
    pl = fspl_d0 + 10.0 * path_loss_exp * np.log10(max(distance_m, 0.01))
    pr_received = pt_dbm + gt_dbi + gr_dbi - pl - misc_loss_db
    return pr_received - pr_sensitivity_dbm


def range_vs_azimuth(gain_2d, theta_deg, phi_deg, freq_mhz,
                     pt_dbm, pr_dbm, gr_dbi,
                     path_loss_exp=2.0, misc_loss_db=0.0):
    """Compute maximum range for each azimuth direction at the horizon.

    Uses gain at the theta closest to 90° for each phi.

    Parameters:
        gain_2d: 2D gain/EIRP array (n_theta, n_phi) in dB
        theta_deg: 1D theta angles in degrees
        phi_deg: 1D phi angles in degrees
        freq_mhz: frequency in MHz
        pt_dbm: transmit power (dBm). For active (EIRP) data, set to 0.
        pr_dbm: receiver sensitivity (dBm)
        gr_dbi: receive antenna gain (dBi)
        path_loss_exp: path loss exponent
        misc_loss_db: system losses (dB)

    Returns:
        range_m: 1D array of max range per phi angle (metres)
        horizon_gain: 1D array of gain/EIRP at horizon per phi (dB)
    """
    # Find theta index closest to 90°
    theta_90_idx = np.argmin(np.abs(theta_deg - 90.0))
    horizon_gain = gain_2d[theta_90_idx, :]

    range_m = np.array([
        friis_range_estimate(pt_dbm, pr_dbm, g, gr_dbi, freq_mhz,
                             path_loss_exp, misc_loss_db)
        for g in horizon_gain
    ])
    return range_m, horizon_gain


# ——— INDOOR / ENVIRONMENTAL PROPAGATION ——————————————————————————

def log_distance_path_loss(freq_mhz, distance_m, n=2.0, d0=1.0, sigma_db=0.0):
    """Log-distance path loss model with optional shadow fading margin.

    PL(d) = FSPL(d0) + 10·n·log10(d/d0) + X_sigma

    Parameters:
        freq_mhz: frequency in MHz
        distance_m: distance in metres (scalar or array)
        n: path loss exponent (2.0 = free space, 3.0 = typical indoor)
        d0: reference distance in metres (default 1 m)
        sigma_db: shadow fading margin in dB (added to path loss).
                  For probabilistic use, pass the desired margin
                  (e.g., 1.28·σ for 90th percentile).

    Returns:
        Path loss in dB (scalar or array matching distance_m).
    """
    distance_m = np.asarray(distance_m, dtype=float)
    d_safe = np.maximum(distance_m, 0.01)
    fspl_d0 = free_space_path_loss(freq_mhz, d0)
    return fspl_d0 + 10.0 * n * np.log10(d_safe / d0) + sigma_db


# ITU-R P.1238 distance power loss coefficient N per environment
_ITU_P1238_N = {
    # (environment, freq_band_ghz_lower): N
    "office": {0.9: 33, 1.2: 32, 1.8: 30, 2.4: 28, 5.0: 31},
    "residential": {0.9: 28, 1.8: 28, 2.4: 28, 5.0: 28},
    "commercial": {0.9: 22, 1.8: 22, 2.4: 22, 5.0: 22},
    "hospital": {0.9: 33, 1.8: 30, 2.4: 28, 5.0: 28},
    "industrial": {0.9: 30, 1.8: 30, 2.4: 30, 5.0: 30},
}

# ITU-R P.1238 floor penetration loss factor Lf(n_floors) in dB
_ITU_P1238_LF = {
    "office": lambda n: 15 + 4 * (n - 1) if n > 0 else 0,
    "residential": lambda n: 4 * n if n > 0 else 0,
    "commercial": lambda n: 6 + 3 * (n - 1) if n > 0 else 0,
    "hospital": lambda n: 15 + 4 * (n - 1) if n > 0 else 0,
    "industrial": lambda n: 10 + 3 * (n - 1) if n > 0 else 0,
}


def _itu_get_N(environment, freq_mhz):
    """Look up ITU-R P.1238 distance power loss coefficient N."""
    env = environment.lower()
    if env not in _ITU_P1238_N:
        env = "office"
    freq_ghz = freq_mhz / 1000.0
    table = _ITU_P1238_N[env]
    # Find closest frequency band
    bands = sorted(table.keys())
    closest = min(bands, key=lambda b: abs(b - freq_ghz))
    return table[closest]


def itu_indoor_path_loss(freq_mhz, distance_m, n_floors=0,
                         environment="office"):
    """ITU-R P.1238 indoor propagation model.

    PL = 20·log10(f_MHz) + N·log10(d) + Lf(n_floors) - 28

    Parameters:
        freq_mhz: frequency in MHz
        distance_m: distance in metres (scalar or array)
        n_floors: number of floors between Tx and Rx
        environment: 'office', 'residential', 'commercial', 'hospital', 'industrial'

    Returns:
        Path loss in dB.
    """
    distance_m = np.asarray(distance_m, dtype=float)
    d_safe = np.maximum(distance_m, 0.1)  # P.1238 valid for d > 1m typically
    N = _itu_get_N(environment, freq_mhz)
    env = environment.lower() if environment.lower() in _ITU_P1238_LF else "office"
    Lf = _ITU_P1238_LF[env](n_floors)
    return 20 * np.log10(freq_mhz) + N * np.log10(d_safe) + Lf - 28


# ITU-R P.2040 material penetration loss (dB) at 2.4 GHz baseline
# Scaled with frequency: loss ∝ sqrt(f/f_ref) approximately
_WALL_LOSS_DB_2G4 = {
    "drywall": 3.0,
    "wood": 4.0,
    "glass": 2.0,
    "brick": 8.0,
    "concrete": 12.0,
    "metal": 20.0,
    "reinforced_concrete": 18.0,
}


def wall_penetration_loss(freq_mhz, material="drywall"):
    """Material penetration loss per ITU-R P.2040 (simplified).

    Loss scales approximately as sqrt(f/2400) from 2.4 GHz reference values.

    Parameters:
        freq_mhz: frequency in MHz
        material: wall material ('drywall', 'wood', 'glass', 'brick',
                  'concrete', 'metal', 'reinforced_concrete')

    Returns:
        Penetration loss in dB per wall.
    """
    mat = material.lower().replace(" ", "_")
    base_loss = _WALL_LOSS_DB_2G4.get(mat, 5.0)
    freq_scale = np.sqrt(freq_mhz / 2400.0)
    return base_loss * freq_scale


def apply_indoor_propagation(gain_2d, theta_deg, phi_deg, freq_mhz,
                             pt_dbm, distance_m, n=3.0, n_walls=1,
                             wall_material="drywall", sigma_db=0.0):
    """Apply indoor propagation model to a measured antenna pattern.

    Computes received power at a given distance for every (theta, phi) direction:
        Pr(θ,φ) = Pt + G(θ,φ) - PL(d) - n_walls·L_wall

    Parameters:
        gain_2d: 2D gain array (n_theta, n_phi) in dBi
        theta_deg: 1D theta angles
        phi_deg: 1D phi angles
        freq_mhz: frequency in MHz
        pt_dbm: transmit power in dBm
        distance_m: distance in metres
        n: path loss exponent
        n_walls: number of wall penetrations
        wall_material: wall material type
        sigma_db: shadow fading margin in dB

    Returns:
        received_power_2d: 2D array (n_theta, n_phi) of received power in dBm
        path_loss_total: total path loss in dB (scalar)
    """
    pl = log_distance_path_loss(freq_mhz, distance_m, n=n, sigma_db=sigma_db)
    wall_loss = n_walls * wall_penetration_loss(freq_mhz, wall_material)
    path_loss_total = float(pl + wall_loss)
    received_power_2d = pt_dbm + gain_2d - path_loss_total
    return received_power_2d, path_loss_total


# ——— MULTIPATH FADING MODELS ———————————————————————————————————

def rayleigh_cdf(power_db, mean_power_db=0.0):
    """Rayleigh fading CDF: probability that received power < x.

    For Rayleigh fading, the power (envelope squared) follows an
    exponential distribution:
        P(power < x) = 1 - exp(-x_linear / mean_linear)

    Parameters:
        power_db: power levels in dB (scalar or array)
        mean_power_db: mean received power in dB

    Returns:
        CDF values (probability between 0 and 1).
    """
    power_db = np.asarray(power_db, dtype=float)
    x_lin = 10 ** (power_db / 10.0)
    mean_lin = 10 ** (mean_power_db / 10.0)
    return 1.0 - np.exp(-x_lin / mean_lin)


def rician_cdf(power_db, mean_power_db=0.0, K_factor=10.0):
    """Rician fading CDF (Marcum Q-function approximation).

    For Rician fading with K-factor, uses a Gaussian approximation
    that is accurate for moderate-to-high K:
        mean = 10·log10((K+1)/exp(K)) + mean_power_dB  (approx)
        σ ≈ 4.34/sqrt(2K+1) dB

    For exact results, requires scipy.stats, but this approximation
    is sufficient for engineering analysis and avoids the dependency.

    Parameters:
        power_db: power levels in dB (scalar or array)
        mean_power_db: mean received power in dB (without fading)
        K_factor: Rician K-factor (linear, ratio of LOS to scattered)

    Returns:
        CDF values (probability between 0 and 1).
    """
    power_db = np.asarray(power_db, dtype=float)
    if K_factor <= 0:
        return rayleigh_cdf(power_db, mean_power_db)

    # Nakagami-m approximation: m = (K+1)^2 / (2K+1)
    # For high K, the distribution approaches Gaussian in dB domain
    # σ_dB ≈ 4.34 / sqrt(2K + 1)
    sigma_db = 4.34 / np.sqrt(2 * K_factor + 1)
    # Mean shift due to K-factor (Rician mean power = total power)
    # No shift needed since mean_power_db already includes LOS component
    z = (power_db - mean_power_db) / sigma_db
    return 0.5 * (1.0 + _erf_approx(z / np.sqrt(2)))


def _erf_approx(x):
    """Abramowitz-Stegun approximation of erf(x), max error < 1.5e-7."""
    x = np.asarray(x, dtype=float)
    sign = np.sign(x)
    x = np.abs(x)
    t = 1.0 / (1.0 + 0.3275911 * x)
    poly = t * (0.254829592 + t * (-0.284496736 + t * (1.421413741
           + t * (-1.453152027 + t * 1.061405429))))
    result = 1.0 - poly * np.exp(-x * x)
    return sign * result


def fade_margin_for_reliability(reliability_pct, fading="rayleigh", K=10):
    """Required fade margin (dB) to achieve target reliability.

    Computes the fade margin below the mean power such that the
    probability of the signal being above (mean - margin) equals
    the target reliability.

    Parameters:
        reliability_pct: target reliability percentage (e.g., 99.0)
        fading: 'rayleigh' or 'rician'
        K: Rician K-factor (linear), used only if fading='rician'

    Returns:
        Fade margin in dB (positive value to subtract from link budget).
    """
    outage = 1.0 - reliability_pct / 100.0
    if outage <= 0:
        return float("inf")
    if outage >= 1:
        return 0.0

    if fading == "rayleigh":
        # Rayleigh: P(power < x) = 1 - exp(-x/mean), solve for x/mean
        # x/mean = -ln(1 - outage)
        # In dB: margin = -10·log10(-ln(1-outage))
        ratio_lin = -np.log(1.0 - outage)
        return -10.0 * np.log10(ratio_lin)
    else:  # rician
        # Use inverse of Gaussian approximation
        sigma_db = 4.34 / np.sqrt(2 * K + 1)
        # z such that Φ(z) = outage → z = Φ^(-1)(outage)
        # Using Beasley-Springer-Moro approximation for inverse normal
        z = _norm_ppf_approx(outage)
        return -z * sigma_db  # margin below mean


def _norm_ppf_approx(p):
    """Rational approximation for inverse normal CDF (probit function).

    Accurate to ~4.5e-4 for 0.0001 < p < 0.9999.
    Uses Abramowitz-Stegun 26.2.23.
    """
    if p <= 0:
        return -10.0
    if p >= 1:
        return 10.0
    if p > 0.5:
        return -_norm_ppf_approx(1 - p)
    t = np.sqrt(-2.0 * np.log(p))
    c0, c1, c2 = 2.515517, 0.802853, 0.010328
    d1, d2, d3 = 1.432788, 0.189269, 0.001308
    return -(t - (c0 + c1 * t + c2 * t**2) / (1.0 + d1 * t + d2 * t**2 + d3 * t**3))


def apply_statistical_fading(gain_2d, theta_deg, phi_deg,
                             fading="rayleigh", K=10, realizations=1000):
    """Apply statistical fading to a measured pattern via Monte-Carlo.

    For each (theta, phi) direction, generates `realizations` fading
    samples and computes statistics.

    Parameters:
        gain_2d: 2D gain array (n_theta, n_phi) in dB
        theta_deg: 1D theta angles
        phi_deg: 1D phi angles
        fading: 'rayleigh' or 'rician'
        K: Rician K-factor (linear)
        realizations: number of Monte-Carlo trials

    Returns:
        mean_db: 2D mean gain (dB) per angle
        std_db: 2D standard deviation (dB) per angle
        outage_5pct_db: 2D 5th-percentile gain (dB) — worst 5% fade
    """
    n_theta, n_phi = gain_2d.shape
    gain_lin = 10 ** (gain_2d / 10.0)

    if fading == "rayleigh":
        # Rayleigh: |h|^2 ~ Exp(1), so faded power = gain * |h|^2
        h_sq = np.random.exponential(1.0, (realizations, n_theta, n_phi))
    else:
        # Rician: |h|^2 where h = sqrt(K/(K+1)) + sqrt(1/(K+1))·CN(0,1)
        mu = np.sqrt(K / (K + 1.0))
        sigma = np.sqrt(1.0 / (2.0 * (K + 1.0)))
        h = mu + sigma * (np.random.randn(realizations, n_theta, n_phi)
                          + 1j * np.random.randn(realizations, n_theta, n_phi))
        h_sq = np.abs(h) ** 2

    # Faded power: gain_linear × fading_coefficient
    faded_lin = gain_lin[np.newaxis, :, :] * h_sq
    faded_db = 10 * np.log10(np.maximum(faded_lin, 1e-20))

    mean_db = np.mean(faded_db, axis=0)
    std_db = np.std(faded_db, axis=0)
    outage_5pct_db = np.percentile(faded_db, 5, axis=0)

    return mean_db, std_db, outage_5pct_db


def delay_spread_estimate(distance_m, environment="indoor"):
    """Estimate RMS delay spread for a given environment.

    Based on typical measured values from literature.

    Parameters:
        distance_m: Tx-Rx distance in metres
        environment: 'indoor', 'office', 'residential', 'urban', 'suburban'

    Returns:
        RMS delay spread in nanoseconds.
    """
    # Typical base delay spreads (ns) and distance scaling
    base_spreads = {
        "indoor": 25.0,
        "office": 35.0,
        "residential": 30.0,
        "hospital": 40.0,
        "industrial": 50.0,
        "urban": 200.0,
        "suburban": 100.0,
    }
    env = environment.lower()
    base = base_spreads.get(env, 35.0)
    # Delay spread scales roughly as sqrt(distance) for indoor
    return base * np.sqrt(max(distance_m, 1.0) / 10.0)


# ——— ENHANCED MIMO ANALYSIS ——————————————————————————————————————

def envelope_correlation_from_patterns(E1_theta, E1_phi, E2_theta, E2_phi,
                                       theta_deg, phi_deg):
    """Compute Envelope Correlation Coefficient from 3D far-field patterns.

    IEEE definition:
        ρe = |∫∫ (E1θ·E2θ* + E1φ·E2φ*) sinθ dθ dφ|²
             / (∫∫|E1|² sinθ dθ dφ · ∫∫|E2|² sinθ dθ dφ)

    where E1, E2 are complex E-field components of antennas 1 and 2.

    Parameters:
        E1_theta, E1_phi: complex E-field components of antenna 1 (n_theta, n_phi)
        E2_theta, E2_phi: complex E-field components of antenna 2 (n_theta, n_phi)
        theta_deg: 1D theta angles in degrees
        phi_deg: 1D phi angles in degrees

    Returns:
        ECC value (float, 0 to 1).
    """
    theta_rad = np.deg2rad(theta_deg)
    sin_theta = np.sin(theta_rad)

    # Cross-correlation integral
    integrand_cross = (E1_theta * np.conj(E2_theta) + E1_phi * np.conj(E2_phi))
    cross = np.sum(integrand_cross * sin_theta[:, np.newaxis])

    # Self-correlation integrals
    self1 = np.sum((np.abs(E1_theta)**2 + np.abs(E1_phi)**2) * sin_theta[:, np.newaxis])
    self2 = np.sum((np.abs(E2_theta)**2 + np.abs(E2_phi)**2) * sin_theta[:, np.newaxis])

    denom = self1 * self2
    if denom == 0:
        return 1.0  # Degenerate case
    return float(np.abs(cross)**2 / denom)


def combining_gain(gains_db, method="mrc"):
    """Compute combined output for multi-antenna receiving.

    Parameters:
        gains_db: 1D array of antenna element gains in dB (one per element)
        method: 'mrc' (maximal ratio combining),
                'egc' (equal gain combining),
                'sc' (selection combining)

    Returns:
        combined_db: combined output in dB
        improvement_db: improvement over single best antenna (dB)
    """
    gains_lin = 10 ** (np.asarray(gains_db, dtype=float) / 10.0)
    best_single = np.max(gains_lin)

    if method == "mrc":
        # MRC: sum of linear powers (optimal when noise is equal)
        combined_lin = np.sum(gains_lin)
    elif method == "egc":
        # EGC: (sum of amplitudes)^2 / N
        combined_lin = (np.sum(np.sqrt(gains_lin)))**2 / len(gains_lin)
    elif method == "sc":
        # SC: select the best branch
        combined_lin = best_single
    else:
        combined_lin = best_single

    combined_db = 10 * np.log10(max(combined_lin, 1e-20))
    improvement_db = 10 * np.log10(max(combined_lin / best_single, 1e-20))
    return combined_db, improvement_db


def mimo_capacity_vs_snr(ecc, snr_range_db=(-5, 30), num_points=36,
                         fading="rayleigh", K=10):
    """Compute MIMO capacity curves over an SNR range.

    Returns capacity for SISO, 2×2 AWGN, and 2×2 fading channels.
    Uses existing capacity_awgn and capacity_monte_carlo functions.

    Parameters:
        ecc: envelope correlation coefficient (scalar, 0-1)
        snr_range_db: (min_snr, max_snr) tuple in dB
        num_points: number of SNR points
        fading: 'rayleigh' or 'rician' for Monte-Carlo
        K: Rician K-factor

    Returns:
        snr_axis: 1D array of SNR values (dB)
        siso_cap: 1D SISO capacity (b/s/Hz)
        awgn_cap: 1D 2×2 AWGN capacity (b/s/Hz)
        fading_cap: 1D 2×2 fading capacity (b/s/Hz)
    """
    snr_axis = np.linspace(snr_range_db[0], snr_range_db[1], num_points)
    siso_cap = np.log2(1 + 10 ** (snr_axis / 10.0))
    awgn_cap = np.array([capacity_awgn(ecc, s) for s in snr_axis])
    fading_cap = np.array([
        capacity_monte_carlo(ecc, s, fading=fading, K=K, trials=500)
        for s in snr_axis
    ])
    return snr_axis, siso_cap, awgn_cap, fading_cap


def mean_effective_gain_mimo(gain_2d_list, theta_deg, phi_deg, xpr_db=6.0):
    """Compute Mean Effective Gain per antenna element (Taga model).

    MEG accounts for the cross-polarization ratio (XPR) of the
    propagation environment and the antenna's polarization characteristics.

    For a single-pol measurement:
        MEG = (1/(2π)) ∫∫ G(θ,φ)·P(θ,φ)·sinθ dθ dφ
    where P(θ,φ) is the incoming power distribution.

    Simplified uniform environment model:
        MEG ≈ weighted average gain with sin(θ) weighting.
        When separate Eθ/Eφ components are available (future),
        xpr_db weights V vs H: V_weight = XPR/(1+XPR), H_weight = 1/(1+XPR).

    Parameters:
        gain_2d_list: list of 2D gain arrays (one per antenna element)
        theta_deg: 1D theta angles
        phi_deg: 1D phi angles
        xpr_db: cross-polarization ratio in dB (reserved for V/H decomposition)

    Returns:
        meg_list: list of MEG values in dB (one per element)
    """
    _ = xpr_db  # reserved for future V/H polarization weighting
    theta_rad = np.deg2rad(theta_deg)
    sin_w = np.sin(theta_rad)
    n_phi = len(phi_deg)

    meg_list = []
    for g2d in gain_2d_list:
        g_lin = 10 ** (g2d / 10.0)
        # Uniform azimuth, sin(theta) elevation weighting
        # XPR scaling: total_weight = XPR/(1+XPR) for V + 1/(1+XPR) for H
        # For total power: effectively just sin-weighted average
        weighted = np.sum(g_lin * sin_w[:, np.newaxis])
        norm = np.sum(sin_w) * n_phi
        meg_lin = weighted / norm if norm > 0 else 0
        meg_db = 10 * np.log10(max(meg_lin, 1e-20))
        meg_list.append(meg_db)
    return meg_list


# ——— WEARABLE / MEDICAL DEVICE ANALYSIS ——————————————————————————

BODY_POSITIONS = {
    "wrist": {"axis": "+X", "cone_deg": 50, "tissue_cm": 2.0},
    "chest": {"axis": "-X", "cone_deg": 60, "tissue_cm": 4.0},
    "hip": {"axis": "-X", "cone_deg": 45, "tissue_cm": 3.5},
    "head": {"axis": "+Z", "cone_deg": 40, "tissue_cm": 2.5},
}


def body_worn_pattern_analysis(gain_2d, theta_deg, phi_deg, freq_mhz,
                               body_positions=None):
    """Analyze antenna pattern across multiple body-worn positions.

    For each position, applies the directional human shadow model and
    computes TRP and efficiency delta.

    Parameters:
        gain_2d: 2D gain array (n_theta, n_phi) in dBi
        theta_deg: 1D theta angles
        phi_deg: 1D phi angles
        freq_mhz: frequency in MHz
        body_positions: list of position names (default: all in BODY_POSITIONS)

    Returns:
        results: dict keyed by position name, each containing:
            'pattern': modified 2D gain array
            'trp_delta_db': change in TRP vs free-space (dB)
            'peak_delta_db': change in peak gain vs free-space (dB)
            'avg_gain_db': sin-weighted average gain after shadowing
    """
    if body_positions is None:
        body_positions = list(BODY_POSITIONS.keys())

    theta_rad = np.deg2rad(theta_deg)
    sin_w = np.sin(theta_rad)

    # Free-space reference
    g_lin = 10 ** (gain_2d / 10.0)
    ref_avg = np.sum(g_lin * sin_w[:, np.newaxis]) / (np.sum(sin_w) * len(phi_deg))
    ref_avg_db = 10 * np.log10(max(ref_avg, 1e-20))
    ref_peak = np.max(gain_2d)

    results = {}
    for pos in body_positions:
        if pos not in BODY_POSITIONS:
            continue
        cfg = BODY_POSITIONS[pos]

        # Expand gain_2d to match theta/phi shape if needed
        theta_2d = np.broadcast_to(theta_deg[:, np.newaxis], gain_2d.shape)
        phi_2d = np.broadcast_to(phi_deg[np.newaxis, :], gain_2d.shape)

        modified = apply_directional_human_shadow(
            gain_2d.copy(),
            theta_2d,
            phi_2d,
            freq_mhz,
            target_axis=cfg["axis"],
            cone_half_angle_deg=cfg["cone_deg"],
            tissue_thickness_cm=cfg["tissue_cm"],
        )

        mod_lin = 10 ** (modified / 10.0)
        mod_avg = np.sum(mod_lin * sin_w[:, np.newaxis]) / (np.sum(sin_w) * len(phi_deg))
        mod_avg_db = 10 * np.log10(max(mod_avg, 1e-20))

        results[pos] = {
            "pattern": modified,
            "trp_delta_db": mod_avg_db - ref_avg_db,
            "peak_delta_db": float(np.max(modified)) - ref_peak,
            "avg_gain_db": mod_avg_db,
        }

    return results


def dense_device_interference(num_devices, tx_power_dbm, freq_mhz,
                              bandwidth_mhz=2.0, room_size_m=(10, 10, 3)):
    """Estimate aggregate interference in a dense device deployment.

    Monte-Carlo: places N co-channel devices at random positions in a room
    and computes interference power at the center.

    Parameters:
        num_devices: number of interfering devices
        tx_power_dbm: per-device transmit power (dBm)
        freq_mhz: frequency (MHz)
        bandwidth_mhz: channel bandwidth (MHz) — for noise floor calc
        room_size_m: (length, width, height) in metres

    Returns:
        avg_interference_dbm: average aggregate interference (dBm)
        sinr_distribution: 1D array of SINR values (dB) from Monte-Carlo
        noise_floor_dbm: thermal noise floor (dBm)
    """
    n_trials = 500
    lx, ly, lz = room_size_m

    # Thermal noise floor: kTB
    noise_floor_dbm = -174 + 10 * np.log10(bandwidth_mhz * 1e6)

    sinr_values = []
    for _ in range(n_trials):
        # Random device positions
        positions = np.column_stack([
            np.random.uniform(0, lx, num_devices),
            np.random.uniform(0, ly, num_devices),
            np.random.uniform(0, lz, num_devices),
        ])
        # Receiver at room center
        rx = np.array([lx / 2, ly / 2, lz / 2])
        distances = np.linalg.norm(positions - rx, axis=1)
        distances = np.maximum(distances, 0.1)  # min 10 cm

        # Path loss per device (indoor n=3)
        pl = np.array([log_distance_path_loss(freq_mhz, d, n=3.0) for d in distances])
        rx_power = tx_power_dbm - pl  # dBm per device

        # One device is the "desired", rest are interferers
        # Take closest as desired signal
        desired_idx = np.argmin(distances)
        desired_power = rx_power[desired_idx]
        interferer_mask = np.ones(num_devices, dtype=bool)
        interferer_mask[desired_idx] = False
        interference_lin = np.sum(10 ** (rx_power[interferer_mask] / 10.0))
        noise_lin = 10 ** (noise_floor_dbm / 10.0)
        sinr = desired_power - 10 * np.log10(interference_lin + noise_lin)
        sinr_values.append(sinr)

    sinr_distribution = np.array(sinr_values)
    return float(np.mean(sinr_distribution)), sinr_distribution, noise_floor_dbm


def sar_exposure_estimate(tx_power_mw, antenna_gain_dbi, distance_cm,
                          freq_mhz, tissue_type="muscle"):
    """Simplified SAR estimation for regulatory screening.

    Uses far-field power density and tissue absorption:
        S = (Pt · Gt) / (4π·d²)          [W/m²]
        SAR ≈ σ · S / (ρ · penetration)  [W/kg]

    NOTE: This is an indicative estimate only. Full SAR compliance
    requires 3D EM simulation per IEC 62209 / IEEE 1528.

    Parameters:
        tx_power_mw: transmit power in milliwatts
        antenna_gain_dbi: antenna gain in dBi
        distance_cm: distance from antenna to tissue surface (cm)
        freq_mhz: frequency in MHz
        tissue_type: 'muscle', 'skin', 'fat', 'bone'

    Returns:
        sar_w_per_kg: estimated SAR (W/kg)
        fcc_limit: FCC limit for comparison (1.6 W/kg for 1g average)
        icnirp_limit: ICNIRP limit (2.0 W/kg for 10g average)
        compliant: True if below both limits (indicative only)
    """
    # Tissue density (kg/m³)
    tissue_props = {
        "muscle": {"density": 1040, "depth_cm": 2.0},
        "skin": {"density": 1100, "depth_cm": 0.5},
        "fat": {"density": 920, "depth_cm": 1.5},
        "bone": {"density": 1850, "depth_cm": 1.0},
    }
    props = tissue_props.get(tissue_type, tissue_props["muscle"])
    sigma = get_tissue_properties(freq_mhz)[1]

    # Power density at distance
    gt_lin = 10 ** (antenna_gain_dbi / 10.0)
    pt_w = tx_power_mw / 1000.0
    d_m = max(distance_cm / 100.0, 0.001)
    S = (pt_w * gt_lin) / (4 * np.pi * d_m ** 2)  # W/m²

    # SAR ≈ σ · E² / ρ, and S = E²/(2·η), so SAR ≈ 2·σ·S/ρ
    # This is the surface SAR, averaged over penetration depth
    rho = props["density"]
    sar = 2 * sigma * S / rho  # W/kg (surface estimate)

    fcc_limit = 1.6  # W/kg (1g average)
    icnirp_limit = 2.0  # W/kg (10g average)

    return sar, fcc_limit, icnirp_limit, bool(sar < fcc_limit and sar < icnirp_limit)


def wban_link_budget(tx_power_dbm, freq_mhz, body_channel="on_body",
                     distance_cm=30):
    """IEEE 802.15.6 WBAN channel model link budget.

    Simplified path loss models from IEEE 802.15.6 standard.

    Parameters:
        tx_power_dbm: transmit power in dBm
        freq_mhz: frequency in MHz
        body_channel: 'on_body', 'in_body', or 'off_body'
        distance_cm: distance in cm

    Returns:
        path_loss_db: estimated path loss (dB)
        received_power_dbm: estimated received power (dBm)
    """
    d_m = max(distance_cm / 100.0, 0.01)

    if body_channel == "on_body":
        # CM3 model (on-body to on-body): PL = a·log10(d) + b + N
        # Typical at 2.4 GHz: a=6.6, b=36.1 (from IEEE 802.15.6)
        a, b = 6.6, 36.1
        path_loss = a * np.log10(d_m) + b
    elif body_channel == "in_body":
        # CM1 model (implant to implant): very high loss
        # PL ≈ 47.14 + 4.26·f_GHz + 29.0·d_cm
        f_ghz = freq_mhz / 1000.0
        d_cm = distance_cm
        path_loss = 47.14 + 4.26 * f_ghz + 0.29 * d_cm
    else:  # off_body
        # CM4 model (on-body to off-body): essentially indoor propagation
        # Use log-distance with n=2.5 (body-nearby effects)
        path_loss = free_space_path_loss(freq_mhz, d_m) + 5.0  # +5dB body effect

    received_power = tx_power_dbm - path_loss
    return float(path_loss), float(received_power)
