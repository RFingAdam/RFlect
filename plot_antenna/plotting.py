from config import THETA_RESOLUTION, PHI_RESOLUTION, polar_dB_max, polar_dB_min

import pandas as pd
import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
import scipy.interpolate as spi
from tkinter import messagebox

# _____________Active Plotting Functions___________
# TODO

# _____________Passive Plotting Functions___________
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
def plot_2d_passive_data(theta_angles_deg, phi_angles_deg, v_gain_dB, h_gain_dB, Total_Gain_dB, freq_list, selected_frequency, save_path=None):
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
              "Peak Gain Versus Frequency", 
              "Frequency (MHz)", 
              "Peak Gain (dBi)",
            x_data=freq_list)
    fig.gca().grid(True, which='both', linestyle='--', linewidth=0.5)
      
    #If save path specified, save otherwise show
    if save_path:
        eff_db_path = os.path.join(save_path, "gain_dBi.png")
        fig.savefig(eff_db_path, format='png')
        plt.close(fig)
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
    ax.legend(loc="upper right", bbox_to_anchor=(1.3, 1))
    ax.grid(True, which='both', linestyle='--', linewidth=0.5)
    
    # Check if save_path is provided
    if save_path:
        #Save Individual Gain Plots at Theta=90, Phi=0, Phi=90
        # TODO plot_additional_polar_patterns(plot_phi_rad, theta_angles_deg, selected_azimuth_freq, selected_frequency, save_path)
        
        azimuth_plot_path = os.path.join(save_path, f"Azimuth_Cuts_{selected_frequency}MHz.png")
        plt.savefig(azimuth_plot_path, format='png')
        plt.close()  # Close the plot after saving
    else:
        plt.show()  # Display the plot

def plot_additional_polar_patterns(plot_phi_rad, plot_theta_deg, plot_Total_Gain_dB, selected_frequency, save_path=None):

    # Define gain summary function for reuse
    def gain_summary(gain_values):
        this_min = np.min(gain_values)
        this_max = np.max(gain_values)
        this_mean = 10 * np.log10(np.mean(10**(gain_values/10)))
        return this_min, this_max, this_mean
    
    # Create polar plot for specific conditions
    def create_polar_plot(title, theta_values, gain_values, freq, plot_type, save_path=None):
        plt.figure()
        ax = plt.subplot(111, projection='polar')

        # Common settings
        ax.plot(theta_values, gain_values, linewidth=2, color='black')
        ax.grid(True, color='gray', linestyle='--', linewidth=0.5)
        ax.set_rlabel_position(90)
        
        # Plot type-specific settings
        settings = {
            'azimuth': {
                'theta_zero_location': 'E',
                'theta_direction': 1,
                'ylim': [polar_dB_min, polar_dB_max],
                'yticks': np.arange(polar_dB_min, polar_dB_max + 1, 5),
                'xticks': np.deg2rad(np.arange(0, 360, 30)),
                'xticklabels': ['0°', '30°', '60°', '90°', '120°', '150°', '180°', '210°', '240°', '270°', '300°', '330°']
            },
            'elevation': {
                'theta_zero_location': 'N',
                'theta_direction': -1,
                'ylim': [polar_dB_min, polar_dB_max],
                'yticks': np.arange(polar_dB_min, polar_dB_max + 1, 5),
                'xticks': np.deg2rad(np.arange(0, 360, 30)),
                'xticklabels': ['90°', '60°', '30°', '0°', '330°', '300°', '270°', '240°', '210°', '180°', '150°', '120°']
            }
        }

        ax.set(**settings[plot_type])
        ax.set_title(title + f" at {freq} MHz")
        
        # Create Gain Summary below plot
        this_min, this_max, this_mean = gain_summary(gain_values)
        ax.text(0.5, -0.10, f"Gain Summary at {freq} MHz   min: {this_min:.1f} dBi   max: {this_max:.1f} dBi   avg: {this_mean:.1f} dBi", 
            horizontalalignment='center', verticalalignment='center', transform=ax.transAxes, color='black', bbox=dict(facecolor='white', alpha=0.7))

        if save_path:
            plt.savefig(os.path.join(save_path, title.replace(" ", "_") + f"_at_{freq}_MHz.png"))
            plt.close()
        else:
            plt.show()
    
    # Create Subfolder to save additional 2D gain cuts        
    two_d_data_subfolder = os.path.join(save_path, f'2D Gain Cuts at {selected_frequency} MHz')
    os.makedirs(two_d_data_subfolder, exist_ok=True)

    # 1. Azimuth Gain Pattern for Theta = 90
    index = np.where(np.abs(plot_theta_deg - 90) < 0.01)[0]
    if index.size != 0:  # Check if index is not empty
        phi_values = np.append(plot_phi_rad[index], plot_phi_rad[index][0])  # Append first value to end
        gain_values = np.append(plot_Total_Gain_dB[index], plot_Total_Gain_dB[index][0])  # Append first value to end
        create_polar_plot("Azimuth Gain, Theta = 90 Degree Plane", phi_values, gain_values, selected_frequency, 'azimuth', two_d_data_subfolder)
    else:
        print("No data found for Azimuth Gain Pattern Theta = 90")

    # 2. Elevation Gain Pattern Phi = 0/180
    index_phi_0 = np.where(np.abs(plot_phi_rad - 0) < 0.01)[0]
    index_phi_180 = np.where(np.abs(plot_phi_rad - np.pi) < 0.01)[0]
    if index_phi_0.size != 0 and index_phi_180.size != 0:  # Check if both indexes are not empty

        # Adjust theta values for phi = 0 slice
        theta_values_phi_0 = 2 * np.pi - np.radians(plot_theta_deg[index_phi_0])
        gain_values_phi_0 = plot_Total_Gain_dB[index_phi_0]

        # For phi = 180 slice, no adjustment to theta values
        theta_values_phi_180 = np.radians(plot_theta_deg[index_phi_180])
        gain_values_phi_180 = plot_Total_Gain_dB[index_phi_180]

        # Concatenate data for plotting
        theta_values = np.concatenate([theta_values_phi_0, theta_values_phi_180])
        gain_values = np.concatenate([gain_values_phi_0, gain_values_phi_180])

        # Adjust gain values by subtracting the minimum value from the entire dataset
        gain_values = gain_values - np.min(plot_Total_Gain_dB)

        create_polar_plot("Elevation Gain, Phi = 0 & 180 Degrees Plane", theta_values, gain_values, selected_frequency, 'elevation', two_d_data_subfolder)
    else:
        print("No data found for Elevation Gain Pattern Phi = 0/180")

    # 3. Elevation Gain Pattern Phi = 90/270
    index = np.where(np.abs(plot_phi_rad - (np.pi / 2)) < 0.01)[0]  # Use np.pi/2 for 90 degrees in radians
    if index.size != 0:  # Check if both indexes are not empty
        theta_values = plot_theta_deg[index]
        gain_values = plot_Total_Gain_dB[index]
        create_polar_plot("Elevation Gain, Phi = 90 & 270 Degrees Plane", theta_values, gain_values, selected_frequency, 'elevation', two_d_data_subfolder)
    else:
        print("No data found for Elevation Gain Pattern Phi = 90/270")


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
 
def plot_passive_3d_component(theta_angles_deg, phi_angles_deg, v_gain_dB, h_gain_dB, Total_Gain_dB, freq_list, selected_frequency, gain_type, save_path=None):
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