from config import THETA_RESOLUTION, PHI_RESOLUTION

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
import scipy.interpolate as spi
from scipy.ndimage import gaussian_filter

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

#plot passive data
def plot_2d_passive_data(theta_angles_deg, phi_angles_deg, v_gain_dB, h_gain_dB, Total_Gain_dB, freq_list, selected_frequency):
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
    
    # Convert Average_Gain_dB to Efficiency Percentage and Plot
    Average_Gain_percentage = 100 * 10**(Average_Gain_dB / 10)
    
    #Plot Eff in %
    fig = plot_data([Average_Gain_percentage], 
            "Average Radiated Efficiency Versus Frequency (%)", 
            "Frequency (MHz)", 
            "Efficiency (%)",
            x_data=freq_list)
    fig.gca().grid(True, which='both', linestyle='--', linewidth=0.5)

    # Calculate Peak Gain as the maximum gain across all angles for each frequency
    Peak_Gain_dB = np.max(Total_Gain_dB, axis=0)
    
    # Plot Peak Gain
    fig = plot_data([Peak_Gain_dB], 
              "Peak Gain Versus Frequency", 
              "Frequency (MHz)", 
              "Peak Gain (dBi)",
            x_data=freq_list)
    fig.gca().grid(True, which='both', linestyle='--', linewidth=0.5)
  
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
    ax.legend(loc="upper right", bbox_to_anchor=(1.3, 1))
    ax.grid(True, which='both', linestyle='--', linewidth=0.5)
    plt.show()

def db_to_linear(db_value):
    return 10 ** (db_value / 10)

# Helper function to process gain data for plotting.
def process_gain_data(selected_gain, selected_phi_angles_deg, selected_theta_angles_deg):
    """
    Helper function to process gain data for plotting.
    """
    # Reshape and mesh the data
    unique_phi = np.unique(selected_phi_angles_deg)
    unique_theta = np.unique(selected_theta_angles_deg)
    reshaped_gain = selected_gain.reshape((len(unique_theta), len(unique_phi)))
    reshaped_gain = np.column_stack((reshaped_gain, reshaped_gain[:, 0]))
    unique_phi = np.append(unique_phi, 360)
    
    # Interpolate the data for smoother gradient shading
    theta_interp = np.linspace(0, 180, THETA_RESOLUTION)
    phi_interp = np.linspace(0, 360, PHI_RESOLUTION)
    f_interp = spi.interp2d(unique_phi, unique_theta, reshaped_gain, kind='linear')
    gain_interp = f_interp(phi_interp, theta_interp)
   # f_interp = spi.RectBivariateSpline(unique_theta, unique_phi, reshaped_gain)
   # gain_interp = f_interp(phi_interp, theta_interp)

    PHI, THETA = np.meshgrid(phi_interp, theta_interp)
    
    # Convert to spherical coordinates
    theta_rad = np.deg2rad(THETA)
    phi_rad = np.deg2rad(PHI)
    R = db_to_linear(gain_interp)
    X = R * np.sin(theta_rad) * np.cos(phi_rad)
    Y = R * np.sin(theta_rad) * np.sin(phi_rad)
    Z = R * np.cos(theta_rad)
    
    return X, Y, Z, gain_interp, R, theta_interp, phi_interp

def normalize_gain(gain_dB):
    """
    Normalize the gain values to be between 0 and 1.
    """
    gain_min = np.min(gain_dB)
    gain_max = np.max(gain_dB)
    return (gain_dB - gain_min) / (gain_max - gain_min)
 
def plot_passive_3d_component(theta_angles_deg, phi_angles_deg, v_gain_dB, h_gain_dB, Total_Gain_dB, freq_list, selected_frequency, gain_type):
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
    X, Y, Z, gain_interp, R, theta_interp, phi_interp = process_gain_data(selected_gain, selected_phi_angles_deg, selected_theta_angles_deg)
    
    # Normalize the gain values
    max_gain_value = np.max(gain_interp)
    min_gain_value = np.min(gain_interp)
    gain_normalized = (gain_interp - min_gain_value) / (max_gain_value - min_gain_value)

    # Convert to spherical coordinates using normalized values
    PHI, THETA = np.meshgrid(phi_interp, theta_interp)
    theta_rad = np.deg2rad(THETA)
    phi_rad = np.deg2rad(PHI)
    R = 0.75 * gain_normalized  # Adjusted to scale the gain to 75% of the usable area
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

    # Calculate the distances from each intersection point to the origin
    dist_x = np.sqrt(start_x**2 + start_y**2 + start_z**2)
    dist_y = np.sqrt(start_x**2 + start_y**2 + start_z**2)
    dist_z = np.sqrt(start_x**2 + start_y**2 + start_z**2)

    # Compute quiver lengths such that they don't exceed plot area
    quiver_length = 0.25 * max(dist_x, dist_y, dist_z) # making them extend 25% further than the plots

    # Plot adjusted quiver arrows
    ax.quiver(start_x, 0, 0, quiver_length, 0, 0, color='green', arrow_length_ratio=0.1, zorder=0)  # X-axis
    ax.quiver(0, start_y, 0, 0, quiver_length, 0, color='red', arrow_length_ratio=0.1, zorder=0)  # Y-axis
    ax.quiver(0, 0, start_z, 0, 0, quiver_length, color='blue', arrow_length_ratio=0.1, zorder=0)  # Z-axis

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

    plt.show()

