# calculations.py

import numpy as np
from scipy.constants import c  # Speed of light
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
    with open(file_path, "r") as f:
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

    return (
        start_phi_h == start_phi_v
        and stop_phi_h == stop_phi_v
        and inc_phi_h == inc_phi_v
        and start_theta_h == start_theta_v
        and stop_theta_h == stop_theta_v
        and inc_theta_h == inc_theta_v
    )


# Extract Frequency points for selection in the drop-down menu
def extract_passive_frequencies(file_path):
    with open(file_path, "r") as file:
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

    Total_Gain_dB = 10 * np.log10(10 ** (v_gain_dB / 10) + 10 ** (h_gain_dB / 10))

    return theta_angles_deg, phi_angles_deg, v_gain_dB, h_gain_dB, Total_Gain_dB


# Enhanced NF to FF Transformation
def apply_nf2ff_transformation(
    hpol_data,
    vpol_data,
    frequency,
    start_phi,
    stop_phi,
    inc_phi,
    start_theta,
    stop_theta,
    inc_theta,
    measurement_distance,
    window_function="none",
):
    """
    Applies Near-Field to Far-Field transformation using Plane Wave Decomposition.

    Parameters:
        hpol_data (list): List of dictionaries with 'mag' and 'phase' for horizontal polarization.
        vpol_data (list): List of dictionaries with 'mag' and 'phase' for vertical polarization.
        frequency (float): Frequency in MHz.
        start_phi, stop_phi, inc_phi (float): Phi angle range and increment in degrees.
        start_theta, stop_theta, inc_theta (float): Theta angle range and increment in degrees.
        measurement_distance (float): Distance from antenna to probe in meters.
        window_function (str): Type of window to apply ('none', 'hanning', 'hamming', etc.).

    Returns:
        hpol_far_field (list): Far-field data for horizontal polarization.
        vpol_far_field (list): Far-field data for vertical polarization.
    """
    # Calculate wavelength
    wavelength = c / (frequency * 1e6)  # Convert MHz to Hz

    # Prepare theta and phi grids in radians
    theta = np.deg2rad(np.arange(start_theta, stop_theta + inc_theta, inc_theta))
    phi = np.deg2rad(np.arange(start_phi, stop_phi + inc_phi, inc_phi))
    theta_grid, phi_grid = np.meshgrid(theta, phi, indexing="ij")

    # Calculate Plane Wave Decomposition coefficients
    k = 2 * np.pi / wavelength  # Wave number

    # Initialize far-field lists
    hpol_far_field = []
    vpol_far_field = []

    # Select window function
    window = get_window(window_function, theta_grid.shape)

    for hpol_entry, vpol_entry in zip(hpol_data, vpol_data):
        # Convert magnitude and phase to complex near-field data
        h_near_field = hpol_entry["mag"] * np.exp(1j * np.deg2rad(hpol_entry["phase"]))
        v_near_field = vpol_entry["mag"] * np.exp(1j * np.deg2rad(vpol_entry["phase"]))

        # Apply windowing if selected
        if window is not None:
            h_near_field *= window
            v_near_field *= window

        # Apply Plane Wave Decomposition
        h_ff = plane_wave_decomposition(h_near_field, theta_grid, phi_grid, k, measurement_distance)
        v_ff = plane_wave_decomposition(v_near_field, theta_grid, phi_grid, k, measurement_distance)

        # Scale far-field based on wavelength and distance
        scaling_factor = wavelength / (4 * np.pi * measurement_distance)
        h_ff *= scaling_factor
        v_ff *= scaling_factor

        # Convert to magnitude and phase
        h_far_field_mag = np.abs(h_ff)
        h_far_field_phase = np.angle(h_ff, deg=True)

        v_far_field_mag = np.abs(v_ff)
        v_far_field_phase = np.angle(v_ff, deg=True)

        # Append to far-field lists
        hpol_far_field.append({"mag": h_far_field_mag, "phase": h_far_field_phase})
        vpol_far_field.append({"mag": v_far_field_mag, "phase": v_far_field_phase})

    return hpol_far_field, vpol_far_field


def plane_wave_decomposition(near_field, theta, phi, k, distance):
    """
    Performs Plane Wave Decomposition on near-field data to obtain far-field.

    Parameters:
        near_field (2D np.array): Complex near-field data.
        theta (2D np.array): Theta angles in radians.
        phi (2D np.array): Phi angles in radians.
        k (float): Wave number.
        distance (float): Measurement distance in meters.

    Returns:
        far_field (2D np.array): Complex far-field data.
    """
    # Calculate the phase shift based on the distance and angle
    # Assuming spherical measurement, phase shift is -j * k * r * cos(theta)
    exponent = -1j * k * distance * np.cos(theta)

    # Apply the phase shift to decompose into far-field
    far_field = near_field * np.exp(exponent)

    return far_field


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
    delta = np.sqrt(
        2 / (omega * 4e-7 * pi * np.sqrt(1 + (sigma / (omega * epsilon)) ** 2))
    )  # skin depth

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
    ecc      : scalar or array of ECC (|ρ₁₂|)
    snr_db   : scalar SNR in dB
    fading   : 'rayleigh' or 'rician'
    K        : Rician K-factor (linear) if fading='rician'
    trials   : number of channel realizations
    returns  : average capacity (b/s/Hz)
    """
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

        # Apply cable loss and convert to linear
        h_mag = np.array(h_entry["mag"]) - cable_loss
        h_phase = np.radians(h_entry["phase"])
        v_mag = np.array(v_entry["mag"]) - cable_loss
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
