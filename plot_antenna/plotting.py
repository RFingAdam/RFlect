from config import THETA_RESOLUTION, PHI_RESOLUTION, polar_dB_max, polar_dB_min
from file_utils import parse_2port_data

import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import pandas as pd
import os
import numpy as np
from matplotlib import cm
import scipy.interpolate as spi
from tkinter import messagebox
from tkinter import simpledialog

# _____________Active Plotting Functions___________
# Function called by gui.py to start the active plotting procedure after parsing and calculating variables
def plot_active_2d_data(data_points, theta_angles_rad, phi_angles_rad, total_power_dBm_2d, frequency, save_path=None):
    # Plot Azimuth cuts for different theta values on one plot
    theta_values_to_plot = [0, 15, 30, 45, 60, 75, 90, 105, 120, 135, 150, 165]
    
    # Save 2D Azimuth Power Cuts with naming consistency if save_path is provided
    if save_path:
        plot_2d_azimuth_power_cuts(phi_angles_rad, theta_angles_rad, total_power_dBm_2d, theta_values_to_plot, frequency, save_path=save_path)
        plot_additional_active_polar_plots(phi_angles_rad, theta_angles_rad, total_power_dBm_2d, frequency, save_path=save_path)
    else:
        plot_2d_azimuth_power_cuts(phi_angles_rad, theta_angles_rad, total_power_dBm_2d, theta_values_to_plot, frequency)
        plot_additional_active_polar_plots(phi_angles_rad, theta_angles_rad, total_power_dBm_2d, frequency)
    
    return

# Plot 2D Azimuth Power Cuts vs Phi for various value of theta to plot
def plot_2d_azimuth_power_cuts(phi_angles_rad, theta_angles_rad, power_dBm_2d, theta_values_to_plot, frequency, save_path=None):
    """
    Plot azimuth power pattern cuts for specified theta values.

    Parameters:
    - theta_angles_rad: 1D array of theta angles in radians.
    - phi_angles_rad: 1D array of phi angles in radians.
    - power_dBm_2d: 2D array of power values, shape should be (num_theta, num_phi).
    - theta_values_to_plot_deg: List of theta angles in degrees for which to plot azimuth cuts.
    """
    plt.figure(figsize=(10, 6))
    ax = plt.subplot(111, projection='polar')

    for theta_deg in theta_values_to_plot:
        # Find the closest index to the desired theta value
        theta_idx = np.argmin(np.abs(np.rad2deg(theta_angles_rad) - theta_deg))
        
        # Extract the corresponding phi and power values
        phi_values = phi_angles_rad
        power_values = power_dBm_2d[theta_idx, :]
        
        # Append the first phi and power value to the end to wrap the data
        phi_values = np.append(phi_values, phi_values[0])
        power_values = np.append(power_values, power_values[0])
        
        ax.plot(phi_values, power_values, label=f'Theta = {theta_deg} deg')

    # Gain Summary
    max_power = np.max(power_dBm_2d)
    min_power = np.min(power_dBm_2d)
    # Average Gain Summary
    power_mW = 10**(power_dBm_2d/10)
    avg_power_mW = np.mean(power_mW)
    avg_power_dBm = 10*np.log10(avg_power_mW)


    # Add a text box or similar to your plot to display these values. For example:
    textstr = f'Power Summary at {frequency} (MHz): Max: {max_power:.1f} (dBm) Min: {min_power:.1f} (dBm) Avg: {avg_power_dBm:.1f} (dBm)'
    ax.annotate(textstr, xy=(-0.1, -0.1), xycoords='axes fraction', fontsize=10,
                bbox=dict(facecolor='none', edgecolor='black', boxstyle='round,pad=0.5'))

    ax.set_title("Azimuth Power Pattern Cuts vs. Phi")
    ax.legend(loc="upper right", bbox_to_anchor=(1.5, 1))
    ax.grid(True, which='both', linestyle='--', linewidth=0.5)

    if save_path:
        # Save the plot with appropriate naming
        azimuth_plot_path = os.path.join(save_path, f"2D_Azimuth_Cuts_{frequency}_MHz.png")
        plt.savefig(azimuth_plot_path, format='png')
        plt.close()  # Close the plot after saving
    else:
        plt.show()

# Plot Additional Elevation and Azimuth cuts for the 3-planes Theta=90deg, Phi=0deg/180deg, and Phi=90deg/270deg
def plot_additional_active_polar_plots(phi_angles_rad, theta_angles_rad, total_power_dBm_2d, frequency, save_path=None):
    # Sanitize the frequency string to avoid special characters
    sanitized_frequency = str(frequency).replace('/', '_').replace('=', '_').replace(':', '_')

    # Azimuth Power Pattern at Theta = 90deg
    idx_theta_90 = np.argwhere(np.isclose(theta_angles_rad, np.pi / 2, atol=1e-6)).flatten()
    if idx_theta_90.size > 0:
        max_power = np.max(total_power_dBm_2d[idx_theta_90, :])
        min_power = np.min(total_power_dBm_2d[idx_theta_90, :])
        avg_power = 10 * np.log10(np.mean(10 ** (total_power_dBm_2d[idx_theta_90, :] / 10)))

        title = f"Total Power in Theta = 90deg Plane, Freq = {sanitized_frequency}MHz"
        plot_polar_power_pattern(title, phi_angles_rad, total_power_dBm_2d[idx_theta_90, :].flatten(), "azimuth", max_power, min_power, avg_power, save_path=save_path)

    # Elevation Power Pattern in the Phi = 270deg/90deg plane
    idx_phi_90 = np.argwhere(np.isclose(phi_angles_rad, 3 * np.pi / 2, atol=1e-6)).flatten()
    idx_phi_270 = np.argwhere(np.isclose(phi_angles_rad, np.pi / 2, atol=1e-6)).flatten()

    if idx_phi_90.size > 0 and idx_phi_270.size > 0:
        theta_phi_90 = theta_angles_rad
        power_phi_90 = total_power_dBm_2d[:, idx_phi_90].flatten()

        theta_phi_270 = 2 * np.pi - theta_angles_rad
        power_phi_270 = total_power_dBm_2d[:, idx_phi_270].flatten()

        power_combined_calc = np.concatenate((power_phi_90, power_phi_270))
        max_power = np.max(power_combined_calc)
        min_power = np.min(power_combined_calc)
        avg_power = 10 * np.log10(np.mean(10 ** (power_combined_calc / 10)))

        theta_combined = np.concatenate((theta_phi_90, [np.nan], theta_phi_270))
        power_combined = np.concatenate((power_phi_90, [np.nan], power_phi_270))

        annotations = [
            {'text': 'Phi=90', 'xy': (3 * np.pi / 2, max_power), 'xytext': (0, 20)},
            {'text': 'Phi=270', 'xy': (np.pi / 2, max_power), 'xytext': (0, 20)}
        ]

        title = f"Elevation Power Pattern at Phi = 270/90deg, Freq = {frequency}MHz"
        plot_polar_power_pattern(title, theta_combined, power_combined, "elevation", max_power, min_power, avg_power, annotations=annotations, save_path=save_path)

    # Elevation Power Pattern in the Phi = 180deg/0deg plane
    idx_phi_0 = np.argwhere(np.isclose(phi_angles_rad, np.pi, atol=1e-6)).flatten()
    idx_phi_180 = np.argwhere(np.isclose(phi_angles_rad, 0, atol=1e-6)).flatten()

    if idx_phi_0.size > 0 and idx_phi_180.size > 0:
        theta_phi_0 = theta_angles_rad
        power_phi_0 = total_power_dBm_2d[:, idx_phi_0].flatten()

        theta_phi_180 = 2 * np.pi - theta_angles_rad
        power_phi_180 = total_power_dBm_2d[:, idx_phi_180].flatten()

        power_combined_calc = np.concatenate((power_phi_0, power_phi_180))
        max_power = np.max(power_combined_calc)
        min_power = np.min(power_combined_calc)
        avg_power = 10 * np.log10(np.mean(10 ** (power_combined_calc / 10)))

        theta_combined = np.concatenate((theta_phi_0, [np.nan], theta_phi_180))
        power_combined = np.concatenate((power_phi_0, [np.nan], power_phi_180))

        annotations = [
            {'text': 'Phi=0', 'xy': (3 * np.pi / 2, max_power), 'xytext': (0, 20)},
            {'text': 'Phi=180', 'xy': (np.pi / 2, max_power), 'xytext': (0, 20)}
        ]

        title = f"Elevation Power Pattern at Phi = 180/0deg, Freq = {frequency}MHz"
        plot_polar_power_pattern(title, theta_combined, power_combined, "elevation", max_power, min_power, avg_power, annotations=annotations, save_path=save_path)
        
# General Polar Plotting Function for Active Plots
def plot_polar_power_pattern(title, angles_rad, power_dBm, plane, max_power, min_power, avg_power, summary_position='bottom', annotations=None, save_path=None):
    fig, ax = plt.subplots(subplot_kw={'projection': 'polar'})
        
    if plane == "azimuth":
        # Append first data point to end for continuous plot
        angles_rad = np.concatenate((angles_rad, [angles_rad[0]]))
        power_dBm = np.concatenate((power_dBm, [power_dBm[0]]))
        ax.set_theta_zero_location("E")
        ax.set_theta_direction(1)
        ax.set_rlabel_position(90)
        ax.set_xticks(np.radians([0, 30, 60, 90, 120, 150, 180, 210, 240, 270, 300, 330]))
        ax.set_xticklabels(['0°', '30°', '60°', '90°', '120°', '150°', '180°', '210°', '240°', '270°', '300°', '330°'])
    elif plane == "elevation":
        ax.set_theta_zero_location("N")
        ax.set_theta_direction(1)
        ax.set_rlabel_position(0)
        ax.set_xticks(np.radians([0, 30, 60, 90, 120, 150, 180, 210, 240, 270, 300, 330]))
        ax.set_xticklabels(['0°', '30°', '60°', '90°', '120°', '150°', '180°', '150°', '120°', '90°', '60°', '30°'])
    
    ax.plot(angles_rad, power_dBm, label=title)
    ax.grid(True, linestyle='--', linewidth=0.5)
    
    # Set labels and title
    ax.set_title(title, va='bottom')
    
    summary_text = f"Max: {max_power:.1f} (dBm) Min: {min_power:.1f} (dBm) Avg: {avg_power:.1f} (dBm)"
    
    if summary_position == 'bottom':
        ax.annotate(summary_text, xy=(0.5, -0.11), xycoords='axes fraction', ha='center', va='center',
                    bbox=dict(boxstyle="round", alpha=0.1))
    elif summary_position == 'top':
        ax.annotate(summary_text, xy=(0.5, 1.15), xycoords='axes fraction', ha='center', va='center',
                    bbox=dict(boxstyle="round", alpha=0.1))
    
    # Add annotations if provided
    if annotations:
        for annotation in annotations:
            ax.annotate(annotation['text'], xy=annotation['xy'], xycoords='data', textcoords='offset points', xytext=annotation['xytext'], ha='center', fontsize=12, bbox=dict(boxstyle="round", alpha=0.5, facecolor='white'))
    
    if save_path:
        # Sanitize the title string to avoid special characters in file names
        sanitized_title = title.replace(' ', '_').replace('=', '_').replace(':', '_').replace('/', '_')
        plot_path = os.path.join(save_path, f"{sanitized_title}.png")
        plt.savefig(plot_path, format='png')
        plt.close(fig)
    else:
        plt.show()

def plot_active_3d_data(theta_angles_deg, phi_angles_deg, total_power_dBm_2d,
                        phi_angles_deg_plot, total_power_dBm_2d_plot,
                        frequency, power_type='total', interpolate=True, save_path=None):
    """
    Plot a 3D representation of the active data (TRP).

    Parameters:
    - theta_angles_deg (array): Theta angles in degrees.
    - phi_angles_deg (array): Phi angles in degrees.
    - total_power_dBm_2d (array): 2D array of power values in dBm.
    - frequency (float): Frequency in MHz.
    - power_type (str): Type of power ('total', 'hpol', 'vpol'). Default is 'total'.
    - interpolate (bool): Whether to interpolate the data for smoother visualization.
    - save_path (str): Path to save the plots. If None, the plot will be displayed.
    """
    # Ensure theta and phi angles are 1D arrays
    theta_angles_deg = np.squeeze(theta_angles_deg)
    phi_angles_deg = np.squeeze(phi_angles_deg)

    # Ensure the Max EIRP is calculated before interpolation
    max_eirp_dBm = np.max(total_power_dBm_2d)

    # Adjusted Python TRP Calculation with theta weighting
    theta_angles_rad = np.deg2rad(theta_angles_deg)  # Ensure theta is in radians
    phi_angles_rad = np.deg2rad(phi_angles_deg)  # Ensure phi is in radians

    # Convert power from dBm to linear scale (mW)
    power_mW = 10**(total_power_dBm_2d / 10)

    # Apply sin(theta) weighting and integrate over theta and phi
    theta_weight = np.sin(theta_angles_rad)
    TRP_mW = np.sum(power_mW * theta_weight[:, np.newaxis] * np.pi / 2) / (len(theta_angles_rad) * len(phi_angles_rad))

    # Convert back to dBm
    TRP_dBm = 10 * np.log10(TRP_mW)

    if interpolate:
        # Create meshgrid of theta and phi angles
        THETA_deg, PHI_deg = np.meshgrid(theta_angles_deg, phi_angles_deg, indexing='ij')

         # Flatten the data for interpolation
        theta_flat = THETA_deg.flatten()
        phi_flat = PHI_deg.flatten()
        power_flat = total_power_dBm_2d.flatten()

        # Check for NaN values in the data
        if np.isnan(power_flat).any():
            print("Warning: NaN values found in power_flat data.")

        # Use the process_data function to interpolate the data
        X, Y, Z, data_interp, R, theta_interp, phi_interp = process_data(power_flat, phi_flat, theta_flat)

        if X is None:
            print("Error in process_data. Exiting plot function.")
            return

        power_data = data_interp

        # Check if power_data contains NaNs after interpolation
        if np.isnan(power_data).all():
            print("Error: Interpolated power_data contains all NaN values.")
            return

        # Normalize the power data
        max_power_value = np.max(power_data)
        min_power_value = np.min(power_data)
        denominator = max_power_value - min_power_value
        if denominator == 0:
            print("Warning: max_power_value equals min_power_value. Normalizing data to zeros.")
            power_normalized = np.zeros_like(power_data)
        else:
            power_normalized = (power_data - min_power_value) / denominator
        
        # Convert to spherical coordinates using normalized values
        PHI, THETA = np.meshgrid(phi_interp, theta_interp)
        theta_rad = np.deg2rad(THETA)
        phi_rad = np.deg2rad(PHI)
        R = 0.75 * power_normalized
        X = R * np.sin(theta_rad) * np.cos(phi_rad)
        Y = R * np.sin(theta_rad) * np.sin(phi_rad)
        Z = R * np.cos(theta_rad)

    else:
        # Use the raw data without interpolation
        # Create a meshgrid of theta and phi angles
        THETA_deg, PHI_deg = np.meshgrid(theta_angles_deg, phi_angles_deg, indexing='ij')
        PHI_rad = np.deg2rad(PHI_deg)
        THETA_rad = np.deg2rad(THETA_deg)

        # Ensure total_power_dBm_2d has correct shape
        if total_power_dBm_2d.shape != THETA_deg.shape:
            total_power_dBm_2d = total_power_dBm_2d.T

        power_data = total_power_dBm_2d

        # Normalize the power data
        max_power_value = np.nanmax(power_data)
        min_power_value = np.nanmin(power_data)
        denominator = max_power_value - min_power_value
        if denominator == 0:
            print("Warning: max_power_value equals min_power_value. Normalizing data to zeros.")
            power_normalized = np.zeros_like(power_data)
        else:
            power_normalized = (power_data - min_power_value) / denominator
        power_normalized = np.nan_to_num(power_normalized)

        # Adjust the radius for visualization
        R_normalized = 0.75 * power_normalized

        # Convert to Cartesian coordinates
        X = R_normalized * np.sin(THETA_rad) * np.cos(PHI_rad)
        Y = R_normalized * np.sin(THETA_rad) * np.sin(PHI_rad)
        Z = R_normalized * np.cos(THETA_rad)

        # For consistency, define theta_interp and phi_interp
        theta_interp = theta_angles_deg
        phi_interp = phi_angles_deg

    # Proceed with plotting as before
    plt.style.use('default')
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(1,1,1, projection='3d')
    
    # Apply coloring based on actual power values (not normalized)
    norm = plt.Normalize(min_power_value, max_power_value)
    mappable = cm.ScalarMappable(norm=norm, cmap=cm.jet)
    mappable.set_array(power_data)

    # Plot the surface
    surf = ax.plot_surface(
        X, Y, Z,
        facecolors=cm.jet(norm(power_data)),
        linewidth=0.5, antialiased=True, shade=False, rstride=1, cstride=1, zorder=10
    )

    # Set the view angle
    ax.view_init(elev=20, azim=-30)

    # Remove axis tick labels but retain grid
    ax.grid(True)
    xticks = ax.get_xticks()
    yticks = ax.get_yticks()
    zticks = ax.get_zticks()

    ax.set_xticklabels(['' if -1.2 < val < 1.2 else f'{val:.1f}' for val in xticks])
    ax.set_yticklabels(['' if -1.2 < val < 1.2 else f'{val:.1f}' for val in yticks])
    ax.set_zticklabels(['' if -1.2 < val < 1.2 else f'{val:.1f}' for val in zticks])

    # Add quiver arrows (axes)
    # Extract indices corresponding to X, Y, Z directions
    idx_theta_90 = np.argmin(np.abs(theta_interp - 90))
    idx_theta_0 = np.argmin(np.abs(theta_interp - 0))
    idx_phi_0 = np.argmin(np.abs(phi_interp - 0))
    idx_phi_90 = np.argmin(np.abs(phi_interp - 90))

    # Get starting points for quivers
    start_x = X[idx_theta_90, idx_phi_0]
    start_y = Y[idx_theta_90, idx_phi_90]
    start_z = Z[idx_theta_0, 0]  # phi index can be 0 since theta is 0

    # Ensure starting points are not too close to zero
    min_offset = 0.05
    if np.abs(start_z) < min_offset:
        start_z = min_offset

    # Calculate the distances from each intersection point to the origin
    dist_x = np.abs(start_x)
    dist_y = np.abs(start_y)
    dist_z = np.abs(start_z)

    # Compute quiver lengths such that they don't exceed plot area
    quiver_length = 0.25 * max(dist_x, dist_y, dist_z)  # Base length for X and Y axes
    min_quiver_length = 0.1  # Minimum length to ensure visibility
    quiver_length = max(quiver_length, min_quiver_length)

    # Make Z-axis quiver longer to enhance visibility
    quiver_length_z = quiver_length * 3.2  # Adjust the factor as needed

    # Plot adjusted quiver arrows
    ax.quiver(start_x, 0, 0, quiver_length*2.2, 0, 0, color='green', arrow_length_ratio=0.1, zorder=20)  # X-axis
    ax.quiver(0, start_y, 0, 0, quiver_length*2.4, 0, color='red', arrow_length_ratio=0.1, zorder=20)    # Y-axis
    ax.quiver(0, 0, start_z, 0, 0, quiver_length_z, color='blue', arrow_length_ratio=0.1, zorder=20)  # Z-axis

    # Set Title based on power_type with rounded TRP values
    if power_type == 'total':
        plot_title = f"Total Radiated Power at {frequency} MHz, TRP = {TRP_dBm:.2f} dBm, Max EIRP = {max_eirp_dBm:.2f} dBm"
    elif power_type == 'hpol':
        plot_title = f"Phi Polarization: Radiated Power at {frequency} MHz, TRP = {TRP_dBm:.2f} dBm"
    elif power_type == 'vpol':
        plot_title = f"Theta Polarization: Radiated Power at {frequency} MHz, TRP = {TRP_dBm:.2f} dBm"
    else:
        plot_title = f"3D Radiation Pattern - {power_type} at {frequency} MHz, TRP = {TRP_dBm:.2f} dBm"
    ax.set_title(plot_title, fontsize=16)


    # Add a colorbar
    cbar = fig.colorbar(mappable, ax=ax, pad=0.1, shrink=0.75)
    cbar.set_label('Power (dBm)', rotation=270, labelpad=20, fontsize=14)
    cbar.ax.tick_params(labelsize=12)

    # Add Max Power to top of Legend
    ax.text2D(1.12, 0.90, f"{max_eirp_dBm:.2f} dBm", transform=ax.transAxes, fontsize=12, weight='bold')

    # If save path provided, save the plot
    if save_path:
        # Save the first view
        plot_3d_path_1 = os.path.join(save_path, f"3D_TRP_{power_type}_{frequency}MHz_1of2.png")
        fig.savefig(plot_3d_path_1, format='png')

        # Adjust view angle to get the rear side of the 3D plot
        ax.view_init(elev=20, azim=150)

        # Save the second view
        plot_3d_path_2 = os.path.join(save_path, f"3D_TRP_{power_type}_{frequency}MHz_2of2.png")
        fig.savefig(plot_3d_path_2, format='png')

        plt.close(fig)
    else:
        plt.show()


# _____________Passive Plotting Functions___________
# Generic Plotting Function
def plot_data(data, title, x_label, y_label, legend_labels=None, x_data=None):
    """
    A generic function to plot data.
    
    Parameters:
    - data: list of data arrays to plot.
    - title: title of the plot.
    - x_label: label for the x-axis.
    - y_label: label for the y-axis.
    - legend_labels: labels for the legend (default is None).
    
    Returns:
    - fig: a matplotlib figure object.
    """
    fig, ax = plt.subplots()
    for d in data:
        if x_data:
            ax.plot(x_data, d)
        else:
            ax.plot(d)
    ax.set_title(title)
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    if legend_labels:
        ax.legend(legend_labels)
    return fig

# Plot passive data
def plot_2d_passive_data(theta_angles_deg, phi_angles_deg, v_gain_dB, h_gain_dB, Total_Gain_dB, freq_list, selected_frequency, datasheet_plots, save_path=None):
    """
    Plot 2D passive data for the given parameters.

    Parameters:
    - theta_angles_deg (array): Theta angles in degrees.
    - phi_angles_deg (array): Phi angles in degrees.
    - v_gain_dB (array): Vertical gain in dB.
    - h_gain_dB (array): Horizontal gain in dB.
    - Total_Gain_dB (array): Total gain in dB.
    - freq_list (list): List of frequencies.
    - selected_frequency (float): Selected frequency for plotting.

    Returns:
    None
    """

    # Convert angles from degrees to radians
    plot_theta_rad = np.radians(theta_angles_deg)
    plot_phi_rad = np.radians(phi_angles_deg)

    # Calculate Average Gain per Frequency in dB
    Average_Gain_dB = 10 * np.log10(np.sum(np.pi/2 * np.sin(plot_theta_rad) * 10**(Total_Gain_dB/10), axis=0) / (theta_angles_deg.shape[0]))
    
    # Plot Efficiency in dB
    fig = plot_data([Average_Gain_dB], 
          "Average Radiated Efficiency Versus Frequency (dB)", 
          "Frequency (MHz)", 
          "Efficiency (dB)",
          x_data=freq_list)
    fig.gca().grid(True, which='both', linestyle='--', linewidth=0.5)
    
    #If save path specified, save otherwise show
    if save_path:
        eff_db_path = os.path.join(save_path, "efficiency_db.png")
        fig.savefig(eff_db_path, format='png')
        plt.close(fig)
    else:
        plt.show()

    # Convert Average_Gain_dB to Efficiency Percentage and Plot
    Average_Gain_percentage = 100 * 10**(Average_Gain_dB / 10)
    
    #Plot Eff in %
    fig = plot_data([Average_Gain_percentage], 
            "Average Radiated Efficiency Versus Frequency (%)", 
            "Frequency (MHz)", 
            "Efficiency (%)",
            x_data=freq_list)
    fig.gca().grid(True, which='both', linestyle='--', linewidth=0.5)
    
    #If save path specified, save otherwise show
    if save_path:
        eff_db_path = os.path.join(save_path, "efficiency_%.png")
        fig.savefig(eff_db_path, format='png')
        plt.close(fig)
    else:
        plt.show()

    # Calculate Peak Gain as the maximum gain across all angles for each frequency
    Peak_Gain_dB = np.max(Total_Gain_dB, axis=0)
    
    # Plot Peak Gain
    fig = plot_data([Peak_Gain_dB], 
              "Total Gain Versus Frequency", 
              "Frequency (MHz)", 
              "Peak Gain (dBi)",
            x_data=freq_list)
    fig.gca().grid(True, which='both', linestyle='--', linewidth=0.5)
    
    '''# Gain summary for Peak Gain
    peak_total_gain = np.max(Peak_Gain_dB)
    peak_total_freq = freq_list[np.argmax(Peak_Gain_dB)]
    fig.gca().annotate(f"Max Total Gain: {peak_total_gain:.2f} dBi at {peak_total_freq} MHz",
                       xy=(0.5, 0), xycoords='axes fraction', fontsize=10,
                       xytext=(0, 10), textcoords='offset points',
                       ha='center', va='bottom', bbox=dict(boxstyle='round,pad=0.5', facecolor='white', alpha=0.7))
    '''
    
    # If save path specified, save otherwise show
    if save_path:
        eff_db_path = os.path.join(save_path, "tota_gain.png")
        fig.savefig(eff_db_path, format='png')
        plt.close(fig)
    else:
        plt.show()

    # If datasheet plots setting is selected in the settings menu
    if datasheet_plots:
        # Calculate the maximum gain across all angles for each frequency
        max_phi_gain_dB = np.max(h_gain_dB, axis=0)
        max_theta_gain_dB = np.max(v_gain_dB, axis=0)
        
        # Plot Phi Gain
        fig_phi = plot_data([max_phi_gain_dB], 
                "Phi Gain Versus Frequency", 
                "Frequency (MHz)", 
                "Phi Gain (dBi)",
                x_data=freq_list)
        fig_phi.gca().grid(True, which='both', linestyle='--', linewidth=0.5)

        # Plot Theta Gain
        fig_theta = plot_data([max_theta_gain_dB], 
                "Theta Gain Versus Frequency", 
                "Frequency (MHz)", 
                "Theta Gain (dBi)",
                x_data=freq_list)
        fig_theta.gca().grid(True, which='both', linestyle='--', linewidth=0.5)

        num_bands = simpledialog.askinteger("Input", "Enter the number of bands:")
        bands = []
        for i in range(num_bands):
            band_range = simpledialog.askstring("Input", f"Enter the frequency range for band {i+1} (e.g., 2400,2480):")
            bands.append(list(map(float, band_range.split(','))))
        
         # Convert freq_list to a NumPy array
        freq_array = np.array(freq_list)
        
        annotation_height = 10 # Initial offset for annotations
        for i, band in enumerate(bands):
            start_freq, end_freq = band
            indices = np.where((freq_array >= start_freq) & (freq_array <= end_freq))

            if len(indices[0]) == 0:
                print(f"No data points found between {start_freq} MHz and {end_freq} MHz.")
                continue  # Skip to the next band if no data points are found in this band

            max_index_phi = np.argmax(h_gain_dB[:, indices], axis=None)
            max_index_theta = np.argmax(v_gain_dB[:, indices], axis=None)

            # TODO Didn't look like this held the correct freq., so removed from annotation 
            peak_band_phi_freq = freq_array[indices[0]][max_index_phi % len(indices[0])]
            peak_band_theta_freq = freq_array[indices[0]][max_index_theta % len(indices[0])]

            peak_band_phi_gain = h_gain_dB[:, indices].flatten()[max_index_phi]
            peak_band_theta_gain = v_gain_dB[:, indices].flatten()[max_index_theta]

        # Annotate the max gain in the band for each polarization in the summary
            fig_phi.gca().annotate(
                f"Band {start_freq}-{end_freq} MHz: Max Phi Gain: {peak_band_phi_gain:.2f} dBi",
                xy=(0.5, 0), xycoords='axes fraction', fontsize=10,
                xytext=(0, annotation_height), textcoords='offset points',
                ha='center', va='bottom', bbox=dict(boxstyle='round,pad=0.5', facecolor='white', alpha=0.7)
            )
            fig_theta.gca().annotate(
                f"Band {start_freq}-{end_freq} MHz: Max Theta Gain: {peak_band_theta_gain:.2f} dBi",
                xy=(0.5, 0), xycoords='axes fraction', fontsize=10,
                xytext=(0, annotation_height), textcoords='offset points',
                ha='center', va='bottom', bbox=dict(boxstyle='round,pad=0.5', facecolor='white', alpha=0.7)
            )
            annotation_height += 20  # Increase offset for each annotation
            
        # If save path specified, save otherwise show
        if save_path:
            fig_phi.savefig(os.path.join(save_path, "phi_gain.png"), format='png')
            fig_theta.savefig(os.path.join(save_path, "theta_gain.png"), format='png')
            plt.close(fig_phi)
            plt.close(fig_theta)
        else:
            plt.show()
                
    # Plot Azimuth cuts for different theta values
    if selected_frequency in freq_list:
        freq_idx = np.where(np.array(freq_list) == selected_frequency)[0][0]
    else:
        print(f"Error: Selected frequency {selected_frequency} not found in the frequency list.")
        return

    selected_azimuth_freq = Total_Gain_dB[:, freq_idx]
    plot_phi_rad = plot_phi_rad[:, freq_idx]
        
    theta_values_to_plot = [0, 15, 30, 45, 60, 75, 90, 105, 120, 135, 150, 165]
    
    plt.figure(figsize=(10,6))
    ax = plt.subplot(111, projection='polar')
    
    for theta in theta_values_to_plot:
        mask = np.abs(theta_angles_deg[:, freq_idx] - theta) < 0.01
        if np.any(mask):
            phi_values = plot_phi_rad[mask]
            gain_values = selected_azimuth_freq[mask]
            
            # Append the first phi and gain value to the end to wrap the data
            phi_values = np.append(phi_values, phi_values[0])
            gain_values = np.append(gain_values, gain_values[0])
            
            ax.plot(phi_values, gain_values, label=f'Theta {theta}°')

    ax.set_title(f"Gain Pattern Azimuth Cuts - Total Gain at {selected_frequency} MHz")
    ax.legend(loc="upper right", bbox_to_anchor=(1.4, 1))
    ax.grid(True, which='both', linestyle='--', linewidth=0.5)

    # Gain Summary
    max_gain = np.max(selected_azimuth_freq)
    min_gain = np.min(selected_azimuth_freq)
    # Average Gain Summary
    power_mW = 10**(selected_azimuth_freq/10)
    avg_gain_lin = np.mean(power_mW)
    avg_gain_dBi = 10*np.log10(avg_gain_lin)


    # Add a text box or similar to your plot to display these values. For example:
    textstr = f'Gain Summary at {selected_frequency} (MHz): Max: {max_gain:.1f} (dBm) Min: {min_gain:.1f} (dBm) Avg: {avg_gain_dBi:.1f} (dBm)'
    ax.annotate(textstr, xy=(-0.1, -0.1), xycoords='axes fraction', fontsize=10,
                bbox=dict(facecolor='none', edgecolor='black', boxstyle='round,pad=0.5'))
    
    if save_path:  
        azimuth_plot_path = os.path.join(save_path, f"Azimuth_Cuts_{selected_frequency}MHz.png")
        plt.savefig(azimuth_plot_path, format='png')
        plt.close()  # Close the plot after saving
    else:
        plt.show()  # Display the plot

    # Check if save_path is provided
    if datasheet_plots:
        # TODO Save Individual Gain Plots at Theta=90, Phi=0, Phi=90
        plot_additional_polar_patterns(plot_phi_rad, theta_angles_deg, selected_azimuth_freq, selected_frequency, save_path)
  

def plot_additional_polar_patterns(plot_phi_rad, plot_theta_deg, plot_Total_Gain_dB, selected_frequency, save_path=None):

    # Define gain summary function for reuse
    def gain_summary(gain_values):
        this_min = np.nanmin(gain_values)
        this_max = np.nanmax(gain_values)
        this_mean = 10 * np.log10(np.nanmean(10**(gain_values/10)))
        return this_min, this_max, this_mean
    
    # Create polar plot for specific conditions
    def create_polar_plot(title, theta_values, gain_values, freq, plot_type, save_path=None, annotations=None):
        plt.figure()
        ax = plt.subplot(111, projection='polar')

        # Common settings
        ax.plot(theta_values, gain_values, linewidth=2) #,color='blue')
        ax.grid(True, color='gray', linestyle='--', linewidth=0.5)
        
        # Plot type-specific settings
        settings = {
            'azimuth': {
                'rlabel_position': 90,
                'theta_zero_location': 'E',
                'theta_direction': 1,
                'ylim': [polar_dB_min, polar_dB_max],
                'yticks': np.arange(polar_dB_min, polar_dB_max + 1, 5),
                'xticks': np.deg2rad(np.arange(0, 360, 30)),
                'xticklabels': ['0°', '30°', '60°', '90°', '120°', '150°', '180°', '210°', '240°', '270°', '300°', '330°']
            },
            'elevation': {
                'rlabel_position': 0,
                'theta_zero_location': 'N',
                'theta_direction': 1,
                'ylim': [polar_dB_min, polar_dB_max],
                'yticks': np.arange(polar_dB_min, polar_dB_max + 1, 5),
                'xticks': np.deg2rad(np.arange(0, 360, 30)),
                'xticklabels': ['0°', '30°', '60°', '90°', '120°', '150°', '180°', '150°', '120°', '90°', '60°', '30°']
            }
        }

        ax.set(**settings[plot_type])
        ax.set_title(title + f" at {freq} MHz")
        
        # Create Gain Summary below plot
        this_min, this_max, this_mean = gain_summary(gain_values)

        summary_text = (f"Gain Summary at {freq} MHz"
                f" min: {this_min:.1f} (dBi)"
                f" max: {this_max:.1f} (dBi)"
                f" avg: {this_mean:.1f} (dBi)")
        ax.annotate(summary_text, xy=(0.5, -0.11), xycoords='axes fraction', ha='center', va='center',
                    bbox=dict(boxstyle="round", alpha=0.1))
        
        # Add annotations if provided
        if annotations:
            for annotation in annotations:
                ax.annotate(annotation['text'], xy=annotation['xy'], xycoords='data', textcoords='offset points', xytext=annotation['xytext'], ha='center')

        if save_path:
            plt.savefig(os.path.join(save_path, title.replace(" ", "_") + f"_at_{freq}_MHz.png"))
            plt.close()
        else:
            plt.show()
    
    if save_path:
        # Create Subfolder to save additional 2D gain cuts        
        two_d_data_subfolder = os.path.join(save_path, f'2D Gain Cuts at {selected_frequency} MHz')
        os.makedirs(two_d_data_subfolder, exist_ok=True)
    else:
        two_d_data_subfolder = None
        
    # 1. Azimuth Gain Pattern for Theta = 90
    index = np.where(np.abs(plot_theta_deg - 90) < 0.01)[0]
    if index.size != 0:  # Check if index is not empty
        phi_values = np.append(plot_phi_rad[index], plot_phi_rad[index][0])  # Append first value to end
        gain_values = np.append(plot_Total_Gain_dB[index], plot_Total_Gain_dB[index][0])  # Append first value to end
        create_polar_plot("Total Gain in Theta = 90 Degree Plane", phi_values, gain_values, selected_frequency, 'azimuth', two_d_data_subfolder)
    else:
        print("No data found for Azimuth Gain Pattern Theta = 90")

    # 2. Elevation Gain Pattern Phi = 180/0 (XZ-Plane: Phi=0° Plane)
    index_phi_0 = np.where(np.isclose(plot_phi_rad, np.pi, atol=1e-6))[0]
    index_phi_180 = np.where(np.isclose(plot_phi_rad, 0, atol=1e-6))[0]

    if index_phi_0.size != 0 and index_phi_180.size != 0:
        # Adjusting theta_values_phi_0 and theta_values_phi_180 to match the reference.
        theta_values_phi_0 = np.radians(plot_theta_deg[index_phi_0])
        gain_values_phi_0 = plot_Total_Gain_dB[index_phi_0]
        
        theta_values_phi_180 = 2 * np.pi - np.radians(plot_theta_deg[index_phi_180])
        gain_values_phi_180 = plot_Total_Gain_dB[index_phi_180]
        
        # Flatten the 2D arrays to 1D arrays
        theta_values_phi_0_flat = theta_values_phi_0.flatten()
        theta_values_phi_180_flat = theta_values_phi_180.flatten()

        # Ensure that gain arrays are also 1D and of matching shape
        gain_values_phi_0_flat = np.repeat(gain_values_phi_0, theta_values_phi_0.shape[1])
        gain_values_phi_180_flat = np.repeat(gain_values_phi_180, theta_values_phi_180.shape[1])

        # Concatenating the flattened arrays with np.nan to introduce a break in the plot
        theta_values = np.concatenate([theta_values_phi_0_flat, [np.nan], theta_values_phi_180_flat])
        gain_values = np.concatenate([gain_values_phi_0_flat, [np.nan], gain_values_phi_180_flat])

        # Now, both theta_values and gain_values should be 1D arrays of the same shape.
        annotations = [
            {'text': 'Phi=180', 'xy': (np.pi / 2, polar_dB_max), 'xytext': (0, 20)},
            {'text': 'Phi=0', 'xy': (3 * np.pi / 2, polar_dB_max), 'xytext': (0, 20)}
        ]
        create_polar_plot("Total Gain in Phi = 180 & 0 Degrees Plane", theta_values, gain_values, selected_frequency, 'elevation', two_d_data_subfolder, annotations)

    # 3. Elevation Gain Pattern Phi = 270/90 (YZ-Plane: Phi=90° Plane)
    index_phi_90 = np.where(np.isclose(plot_phi_rad, 3 * np.pi / 2, atol=1e-6))[0]
    index_phi_270 = np.where(np.isclose(plot_phi_rad, np.pi / 2, atol=1e-6))[0]

    if index_phi_90.size != 0 and index_phi_270.size != 0:
        # Keeping the same as it aligns with the reference.
        theta_values_phi_90 = np.radians(plot_theta_deg[index_phi_90])
        gain_values_phi_90 = plot_Total_Gain_dB[index_phi_90]

        theta_values_phi_270 = 2 * np.pi - np.radians(plot_theta_deg[index_phi_270])
        gain_values_phi_270 = plot_Total_Gain_dB[index_phi_270]

        # Flatten the 2D arrays to 1D arrays
        theta_values_phi_90_flat = theta_values_phi_90.flatten()
        theta_values_phi_270_flat = theta_values_phi_270.flatten()

        # Ensure that gain arrays are also 1D and of matching shape
        gain_values_phi_90_flat = np.repeat(gain_values_phi_90, theta_values_phi_90.shape[1])
        gain_values_phi_270_flat = np.repeat(gain_values_phi_270, theta_values_phi_270.shape[1])

        # Concatenating the flattened arrays with np.nan to introduce a break in the plot
        theta_values = np.concatenate([theta_values_phi_90_flat, [np.nan], theta_values_phi_270_flat])
        gain_values = np.concatenate([gain_values_phi_90_flat, [np.nan], gain_values_phi_270_flat])

        # Now, both theta_values and gain_values should be 1D arrays of the same shape.
        annotations = [
            {'text': 'Phi=270', 'xy': (np.pi / 2, polar_dB_max), 'xytext': (0, 20)},
            {'text': 'Phi=90', 'xy': (3 * np.pi / 2, polar_dB_max), 'xytext': (0, 20)}
        ]
        create_polar_plot("Total Gain in Phi = 270 & 90 Degrees Plane", theta_values, gain_values, selected_frequency, 'elevation', two_d_data_subfolder, annotations)
        
def db_to_linear(db_value):
    return 10 ** (db_value / 10)

# Helper function to process gain data for plotting.
def process_data(selected_data, selected_phi_angles_deg, selected_theta_angles_deg):
    # Get unique phi and theta values
    unique_phi = np.unique(selected_phi_angles_deg)
    unique_theta = np.unique(selected_theta_angles_deg)

    # Append 360° to phi if not already included (for wrapping)
    if unique_phi[0] == 0 and unique_phi[-1] < 360:
        unique_phi = np.append(unique_phi, 360)
        # Append the first column of data to the end to maintain continuity
        data_grid = selected_data.reshape((len(unique_theta), len(unique_phi) - 1))
        data_grid = np.column_stack((data_grid, data_grid[:, 0]))  # Append first column to wrap
    else:
        data_grid = selected_data.reshape((len(unique_theta), len(unique_phi)))

    # Interpolate the data for smoother gradient shading
    theta_interp = np.linspace(unique_theta.min(), unique_theta.max(), THETA_RESOLUTION)
    phi_interp = np.linspace(unique_phi.min(), unique_phi.max(), PHI_RESOLUTION)

    # Create interpolation function
    f_interp = spi.interp2d(unique_phi, unique_theta, data_grid, kind='linear')

    # Interpolate data
    data_interp = f_interp(phi_interp, theta_interp)

    # Convert to spherical coordinates
    PHI_interp_grid, THETA_interp_grid = np.meshgrid(phi_interp, theta_interp)
    theta_rad = np.deg2rad(THETA_interp_grid)
    phi_rad = np.deg2rad(PHI_interp_grid)
    R = db_to_linear(data_interp)
    X = R * np.sin(theta_rad) * np.cos(phi_rad)
    Y = R * np.sin(theta_rad) * np.sin(phi_rad)
    Z = R * np.cos(theta_rad)

    return X, Y, Z, data_interp, R, theta_interp, phi_interp

def normalize_gain(gain_dB):
    """
    Normalize the gain values to be between 0 and 1.
    """
    gain_min = np.min(gain_dB)
    gain_max = np.max(gain_dB)
    return (gain_dB - gain_min) / (gain_max - gain_min)
 
def plot_passive_3d_component(theta_angles_deg, phi_angles_deg, h_gain_dB, v_gain_dB, Total_Gain_dB, freq_list, selected_frequency, gain_type, save_path=None):
    """
    Plot a 3D representation of the passive component data.

    This function visualizes the passive antenna data in a 3D representation. 
    It takes in gain values (in dB) along with theta and phi angles and creates 
    a 3D plot to showcase the data for a selected frequency.

    Parameters:
    - theta_angles_deg (ndarray): Array of theta angles in degrees.
    - phi_angles_deg (ndarray): Array of phi angles in degrees.
    - v_gain_dB (ndarray): Vertical gain values in dB.
    - h_gain_dB (ndarray): Horizontal gain values in dB.
    - Total_Gain_dB (ndarray): Total gain values in dB.
    - freq_list (list): List of frequencies.
    - selected_frequency (float): The frequency value for which the 3D plot should be generated.

    Returns:
    None. The function directly displays the 3D plot.
    """

    # Check if the selected frequency is present in the frequency list
    if selected_frequency in freq_list:
        freq_idx = np.where(np.array(freq_list) == selected_frequency)[0][0]
    else:
        print(f"Error: Selected frequency {selected_frequency} not found in the frequency list.")
        return
    
    if gain_type == "total":
        selected_gain = Total_Gain_dB[:, freq_idx]
        plot_title = f"3D Radiation Pattern - Total Gain at {selected_frequency} MHz"
    elif gain_type == "hpol":
        selected_gain = h_gain_dB[:, freq_idx]
        plot_title = f"3D Radiation Pattern - Phi Polarization Gain at {selected_frequency} MHz"
    elif gain_type == "vpol":
        selected_gain = v_gain_dB[:, freq_idx]
        plot_title = f"3D Radiation Pattern - Theta Polarization Gain at {selected_frequency} MHz"
    else:
        print(f"Error: Invalid gain type {gain_type}.")
        return

    selected_theta_angles_deg = theta_angles_deg[:, freq_idx]
    selected_phi_angles_deg = phi_angles_deg[:, freq_idx]
    
    # Process gain data
    X, Y, Z, gain_interp, R, theta_interp, phi_interp = process_data(selected_gain, selected_phi_angles_deg, selected_theta_angles_deg)
    
    # Normalize the gain values
    max_gain_value = np.max(gain_interp)
    min_gain_value = np.min(gain_interp)
    gain_normalized = (gain_interp - min_gain_value) / (max_gain_value - min_gain_value)

    # Convert to spherical coordinates using normalized values
    PHI, THETA = np.meshgrid(phi_interp, theta_interp)
    theta_rad = np.deg2rad(THETA)
    phi_rad = np.deg2rad(PHI)
    R = 0.75 * gain_normalized
    X = R * np.sin(theta_rad) * np.cos(phi_rad)
    Y = R * np.sin(theta_rad) * np.sin(phi_rad)
    Z = R * np.cos(theta_rad)

    # Plotting
    plt.style.use('default') 
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(1, 1, 1, projection='3d')
    
    # Remove axis tick labels but retain grid
    ax.grid(True)
    xticks = ax.get_xticks()
    yticks = ax.get_yticks()
    zticks = ax.get_zticks()

    ax.set_xticklabels(['' if -1.2 < val < 1.2 else f'{val:.1f}' for val in xticks])
    ax.set_yticklabels(['' if -1.2 < val < 1.2 else f'{val:.1f}' for val in yticks])
    ax.set_zticklabels(['' if -1.2 < val < 1.2 else f'{val:.1f}' for val in zticks])

    # Apply coloring based on actual gain values (not normalized)
    norm = plt.Normalize(selected_gain.min(), selected_gain.max())
    surf = ax.plot_surface(X, Y, Z, facecolors=cm.jet(norm(gain_interp)), linewidth=0.5, antialiased=True, shade=False, rstride=1, cstride=1, zorder=10)
    
    # Extract gains for the respective directions
    gain_x_idx = np.argmin(np.abs(theta_interp - 90))
    gain_y_idx = np.argmin(np.abs(theta_interp - 90))
    gain_z_idx = np.argmin(np.abs(theta_interp - 0))

    gain_x_idx_phi = np.argmin(np.abs(phi_interp - 0))
    gain_y_idx_phi = np.argmin(np.abs(phi_interp - 90))
    
    # Calculate the starting points of the quivers to be where the gain plot intersects with the axes
    start_x = X[gain_x_idx, gain_x_idx_phi]
    start_y = Y[gain_y_idx, gain_y_idx_phi]
    start_z = Z[gain_z_idx, 0]

    # Ensure starting points are not too close to zero (which might hide the axis)
    min_offset = 0.05  # Small offset to ensure visibility
    if np.abs(start_z) < min_offset:
        start_z = min_offset

    # Calculate the distances from each intersection point to the origin
    dist_x = np.abs(start_x)
    dist_y = np.abs(start_y)
    dist_z = np.abs(start_z)

    # Compute quiver lengths such that they don't exceed plot area
    quiver_length = 0.25 * max(dist_x, dist_y, dist_z)  # Base length for X and Y axes
    min_quiver_length = 0.1  # Minimum length to ensure visibility
    quiver_length = max(quiver_length, min_quiver_length)

    # Make Z-axis quiver longer to enhance visibility
    quiver_length_z = quiver_length * 3.2  # Adjust the factor as needed

    # Plot adjusted quiver arrows
    ax.quiver(start_x, 0, 0, quiver_length*2.2, 0, 0, color='green', arrow_length_ratio=0.1, zorder=20)  # X-axis
    ax.quiver(0, start_y, 0, 0, quiver_length*2.4, 0, color='red', arrow_length_ratio=0.1, zorder=20)    # Y-axis
    ax.quiver(0, 0, start_z, 0, 0, quiver_length_z, color='blue', arrow_length_ratio=0.1, zorder=20)  # Z-axis

    #Adjust the view angle for a top-down view
    #ax.view_init(elev=10, azim=-25)
    ax.view_init(elev=20, azim=-30)  # Tweaking the view angle for a better perspective

    # Set Title
    ax.set_title(plot_title, fontsize=16)

    # Add a colorbar
    mappable = cm.ScalarMappable(norm=norm, cmap=cm.jet)
    mappable.set_array(gain_interp)
    cbar = fig.colorbar(mappable, ax=ax, pad=0.1, shrink=0.75)
    cbar.set_label('Gain (dBi)', rotation=270, labelpad=20, fontsize=14)
    cbar.ax.tick_params(labelsize=12)

    # Add Max Gain to top of Legend
    max_gain = selected_gain.max()
    ax.text2D(1.12, 0.90, f"{max_gain:.2f} dBi", transform=ax.transAxes, fontsize=12, weight='bold')

    #If Path provided, save otherwise show
    if save_path:
        # Save the first view
        plot_3d_path_1 = os.path.join(save_path, f"3D_{gain_type}_1of2.png")
        fig.savefig(plot_3d_path_1, format='png')
        
        # Adjust view angle to get the rear side of the 3D plot
        ax.view_init(elev=20, azim=150)  # Adjust the azimuthal angle to get the rear view
        
        # Save the second view
        plot_3d_path_2 = os.path.join(save_path, f"3D_{gain_type}_2of2.png")
        fig.savefig(plot_3d_path_2, format='png')
        
        plt.close(fig)
    else:
        plt.show()

# _____________Passive Plotting G&D Functions___________
def plot_gd_data(datasets, labels, x_range):
        # Initialize x_min and x_max to None (auto-scale)
        x_min, x_max = None, None

        # Parse the input if it's not empty
        if x_range:
            try:
                x_values = x_range.split(',')
                x_min = float(x_values[0])
                x_max = float(x_values[1])
            except:
                messagebox.showwarning("Warning", "Invalid input for x-axis range. Using auto-scale.")
        # For each type of data, plot datasets together
        data_types = ['Gain', 'Directivity', 'Efficiency', 'Efficiency_dB']
        y_labels = ['Gain (dBi)', 'Directivity (dB)', 'Efficiency (%)', 'Efficiency (dB)']
        titles = ['Gain vs Frequency', 'Directivity vs Frequency', 'Efficiency (%) vs Frequency', 'Efficiency (dB) vs Frequency']

        for data_type, y_label, title in zip(data_types, y_labels, titles):
            plt.figure()
            for data, label in zip(datasets, labels):
                if data_type == 'Efficiency_dB':
                    y_data = 10 * np.log10(np.array(data['Efficiency'])/100)
                else:
                    y_data = data[data_type]
                plt.plot(data['Frequency'], y_data, label=label)
            plt.title(title)
            plt.ylabel(y_label)
            plt.xlabel("Frequency (MHz)")
            plt.legend()
            plt.grid(True)
            if x_min is not None and x_max is not None:
                plt.xlim(x_min, x_max)

        plt.show()

# _____________CSV (VSWR/S11) Plotting Functions___________
def process_vswr_files(file_paths, saved_limit1_freq1, saved_limit1_freq2, saved_limit1_start, saved_limit1_stop, saved_limit2_freq1, saved_limit2_freq2, saved_limit2_start, saved_limit2_stop ):
    # Try to read the first line to determine if it's a 2-port measurement with the header
    # TODO Define Consistent Axis Methodology for Plotting S-Parameters
    # TODO Define Limit 1, Limit 2, & S21 or S12 Limits
    # TODO Handle Other Parameters for 2-port analysis like VSWR etc
    with open(file_paths[0], 'r', encoding='utf-8-sig') as f:  # encoding='utf-8-sig' to handle BOM characters
        first_line = f.readline()

    if '!' in first_line: # Process S2VNA
        # Collect all the data from the files into a list
        all_data = [parse_2port_data(file_path) for file_path in file_paths]
        
       # Define a preferred order for the S-parameters
        preferred_order = ['S11(dB)', 'S22(dB)', 'S21(dB)', 'S12(dB)', 'S21(s)', 'S12(s)']
        
        # Extract all the unique S-parameters across the files
        unique_columns = set(col for data in all_data for col in data.columns if col != '! Stimulus(Hz)')

        # Order the unique_columns based on the preferred order
        ordered_columns = sorted(unique_columns, key=lambda x: preferred_order.index(x) if x in preferred_order else len(preferred_order))

        # Create a subplot for each S-parameter
        # TODO Dynmamic Figure Size based on number of plots or fixed
        # fig, axes = plt.subplots(nrows=len(ordered_columns), figsize=(10, 3.5 * len(ordered_columns)))
        fig, axes = plt.subplots(nrows=len(ordered_columns), figsize=(10, 10))

        # If only one subplot, make sure 'axes' is a list for consistent indexing
        if len(ordered_columns) == 1:
            axes = [axes]

        # Iterate over each S-parameter and plot the data from each file
        for ax, column in zip(axes, ordered_columns):
            for data, file_path in zip(all_data, file_paths):
                if column in data:
                    freqs_ghz = data['! Stimulus(Hz)'] / 1e9  # Convert to GHz
                    values = data[column]
                    ax.plot(freqs_ghz, values, label=f"{os.path.basename(file_path)}")

                    ax.set_title(column)
                    ax.set_xlabel('Frequency (GHz)')
                    if 'dB' in column:
                        ax.set_ylabel('Magnitude (dB)')
                    elif 's' in column:
                        ax.set_ylabel('Group Delay (ns)')    
                    else:
                        ax.set_ylabel('Value')
                    ax.grid(True)
                    ax.legend()

        plt.tight_layout()
        plt.show()

    else: # Process RVNA Files
        fig, ax = plt.subplots(figsize=(10, 6))

        for file_path in file_paths:
            data = pd.read_csv(file_path)
            freqs_ghz = data.iloc[:, 0] / 1e9
            values = data.iloc[:, 1]
            ax.plot(freqs_ghz, values, label=os.path.basename(file_path))

            if values.mean() < 0:
                ax.set_ylabel("Return Loss (dB)")
                ax.set_title("S11, LogMag vs. Frequency")
            else:
                ax.set_ylabel("VSWR")
                ax.set_title("VSWR vs. Frequency")

            # Setting the x-axis limits based on the data frequency range
            ax.set_xlim(freqs_ghz.min(), freqs_ghz.max()) 

            # Check if the saved limit values are available and not zero
            if saved_limit1_freq1 and saved_limit1_freq2 and saved_limit1_start and saved_limit1_stop:
                print(f"Limit 1: {saved_limit1_freq1}, {saved_limit1_freq2}, {saved_limit1_start}, {saved_limit1_stop}")
                ax.plot([saved_limit1_freq1, saved_limit1_freq2], [saved_limit1_start, saved_limit1_stop], linewidth=2, zorder=100, color='red', alpha=0.8)
            
            if saved_limit2_freq1 and saved_limit2_freq2 and saved_limit2_start and saved_limit2_stop:
                print(f"Limit 2: {saved_limit2_freq1}, {saved_limit2_freq2}, {saved_limit2_start}, {saved_limit2_stop}")
                ax.plot([saved_limit2_freq1, saved_limit2_freq2], [saved_limit2_start, saved_limit2_stop], linewidth=2, zorder=100, color ='red', alpha=0.8)

            ax.set_xlabel("Frequency (GHz)")
            ax.legend()
            ax.grid(True)
            plt.tight_layout()
        plt.show()


# Plots M5090 S2VNA S-Parameter Files Depending On what S-parameters are available
def plot_2port_data(file_paths):
    # Collect all the data from the files into a list
    all_data = [parse_2port_data(file_path) for file_path in file_paths]
    
    # Extract all the unique S-parameters across the files
    unique_columns = set(col for data in all_data for col in data.columns if col != '! Stimulus(Hz)')

    # Create a subplot for each S-parameter
    fig, axes = plt.subplots(nrows=len(unique_columns), figsize=(10, 6 * len(unique_columns)))

    # If only one subplot, make sure 'axes' is a list for consistent indexing
    if len(unique_columns) == 1:
        axes = [axes]

    # Iterate over each S-parameter and plot the data from each file
    for ax, column in zip(axes, unique_columns):
        for data, file_path in zip(all_data, file_paths):
            if column in data:
                freqs_ghz = data['! Stimulus(Hz)'] / 1e9  # Convert to GHz
                values = data[column]
                ax.plot(freqs_ghz, values, label=f"{os.path.basename(file_path)}")

                ax.set_title(column)
                ax.set_xlabel('Frequency (GHz)')
                if 'dB' in column:
                    ax.set_ylabel('Magnitude (dB)')
                else:
                    ax.set_ylabel('Value')
                ax.grid(True)
                ax.legend()

    plt.tight_layout()
    plt.show()
