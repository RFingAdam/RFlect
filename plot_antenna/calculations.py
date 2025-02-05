# calculations.py

import numpy as np
from scipy.constants import c  # Speed of light
from scipy.signal import windows

def calculate_active_variables(start_phi, stop_phi, start_theta, stop_theta, inc_phi, inc_theta, h_power_dBm, v_power_dBm):
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
    total_power_dBm_2d = 10 * np.log10(10**(v_power_dBm_2d / 10) + 10**(h_power_dBm_2d / 10))

    # Calculate TRP using original arrays
    power_mW = 10 ** (total_power_dBm_2d / 10)
    theta_weight = np.sin(theta_angles_rad)
    TRP_mW = np.sum(power_mW * theta_weight[:, np.newaxis]) * (np.pi / phi_points)
    TRP_dBm = 10 * np.log10(TRP_mW)

    # Similarly calculate h_TRP_dBm and v_TRP_dBm
    h_TRP_mW = np.sum(10**(h_power_dBm_2d / 10) * theta_weight[:, np.newaxis]) * (np.pi / phi_points)
    h_TRP_dBm = 10 * np.log10(h_TRP_mW)
    v_TRP_mW = np.sum(10**(v_power_dBm_2d / 10) * theta_weight[:, np.newaxis]) * (np.pi / phi_points)
    v_TRP_dBm = 10 * np.log10(v_TRP_mW)

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
    return (data_points, theta_angles_deg, phi_angles_deg, theta_angles_rad, phi_angles_rad,
            total_power_dBm_2d, h_power_dBm_2d, v_power_dBm_2d,
            phi_angles_deg_plot, phi_angles_rad_plot,
            total_power_dBm_2d_plot, h_power_dBm_2d_plot, v_power_dBm_2d_plot,
            total_power_dBm_min, total_power_dBm_nom,
            h_power_dBm_min, h_power_dBm_nom,
            v_power_dBm_min, v_power_dBm_nom,
            TRP_dBm, h_TRP_dBm, v_TRP_dBm)



# _____________Passive Calculation Functions___________
# Auto Determine Polarization for HPOL & VPOL Files
def determine_polarization(file_path):
    with open(file_path, 'r') as f:
        content = f.read()
        if "Horizontal Polarization" in content:
            return "HPol"
        else:
            return "VPol"

# Verify angle data and frequencies are not mismatched      
def angles_match(start_phi_h, stop_phi_h, inc_phi_h, start_theta_h, stop_theta_h, inc_theta_h,
                            start_phi_v, stop_phi_v, inc_phi_v, start_theta_v, stop_theta_v, inc_theta_v):

    return (start_phi_h == start_phi_v and stop_phi_h == stop_phi_v and inc_phi_h == inc_phi_v and
            start_theta_h == start_theta_v and stop_theta_h == stop_theta_v and inc_theta_h == inc_theta_v)

# Extract Frequency points for selection in the drop-down menu      
def extract_passive_frequencies(file_path):
    with open(file_path, 'r') as file:
        content = file.readlines()

    # Extracting frequencies
    frequencies = [float(line.split("=")[1].split()[0]) for line in content if "Test Frequency" in line]

    return frequencies

# Calculate Total Gain Vector and add cable loss etc - Use Phase for future implementation?
def calculate_passive_variables(hpol_data, vpol_data, cable_loss, start_phi, stop_phi, inc_phi, start_theta, stop_theta, inc_theta, freq_list, selected_frequency):
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
        for n, (theta_h, phi_h, mag_h, phase_h, theta_v, phi_v, mag_v, phase_v) in enumerate(zip(hpol_entry['theta'], hpol_entry['phi'], hpol_entry['mag'], hpol_entry['phase'], vpol_entry['theta'], vpol_entry['phi'], vpol_entry['mag'], vpol_entry['phase'])):
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

    Total_Gain_dB = 10 * np.log10(10**(v_gain_dB/10) + 10**(h_gain_dB/10))
    
   
    return theta_angles_deg, phi_angles_deg, v_gain_dB, h_gain_dB, Total_Gain_dB

# Enhanced NF to FF Transformation
def apply_nf2ff_transformation(hpol_data, vpol_data, frequency,
                               start_phi, stop_phi, inc_phi,
                               start_theta, stop_theta, inc_theta,
                               measurement_distance, window_function='none'):
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
    theta_grid, phi_grid = np.meshgrid(theta, phi, indexing='ij')

    # Calculate Plane Wave Decomposition coefficients
    k = 2 * np.pi / wavelength  # Wave number

    # Initialize far-field lists
    hpol_far_field = []
    vpol_far_field = []

    # Select window function
    window = get_window(window_function, theta_grid.shape)

    for hpol_entry, vpol_entry in zip(hpol_data, vpol_data):
        # Convert magnitude and phase to complex near-field data
        h_near_field = hpol_entry['mag'] * np.exp(1j * np.deg2rad(hpol_entry['phase']))
        v_near_field = vpol_entry['mag'] * np.exp(1j * np.deg2rad(vpol_entry['phase']))

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
        hpol_far_field.append({'mag': h_far_field_mag, 'phase': h_far_field_phase})
        vpol_far_field.append({'mag': v_far_field_mag, 'phase': v_far_field_phase})

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
    if window_type.lower() == 'none':
        return None
    elif window_type.lower() == 'hanning':
        window_1d_theta = windows.hann(shape[0])
        window_1d_phi = windows.hann(shape[1])
    elif window_type.lower() == 'hamming':
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
    Helper function to process data (gain or power) for plotting using interp2d.
    """
    # Convert angles to radians
    theta = np.deg2rad(selected_theta_angles_deg)
    phi = np.deg2rad(selected_phi_angles_deg)

    # Create a 2D grid of theta and phi values
    theta_grid, phi_grid = np.meshgrid(theta, phi, indexing='ij')

    # Perform Plane Wave Decomposition
    far_field_complex = np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(selected_data)))

    # Normalize the far-field pattern
    far_field_mag = np.abs(far_field_complex)
    far_field_phase = np.angle(far_field_complex, deg=True)

    return far_field_mag, far_field_phase
