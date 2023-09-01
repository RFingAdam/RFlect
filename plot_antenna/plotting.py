import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm

def plot_data(data, title, x_label, y_label, legend_labels=None):
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
    plot_data([Average_Gain_dB], 
          "Average Radiated Efficiency Versus Frequency (dB)", 
          "Frequency (MHz)", 
          "Efficiency (dB)")

    
    # Convert Average_Gain_dB to Efficiency Percentage and Plot
    Average_Gain_percentage = 100 * 10**(Average_Gain_dB / 10)
    
    #Plot Eff in %
    plot_data([Average_Gain_percentage], 
            "Average Radiated Efficiency Versus Frequency (%)", 
            "Frequency (MHz)", 
            "Efficiency (%)")
    
    # Calculate Peak Gain as the maximum gain across all angles for each frequency
    Peak_Gain_dB = np.max(Total_Gain_dB, axis=0)
    
    # Plot Peak Gain
    plot_data([Peak_Gain_dB], 
              "Peak Gain Versus Frequency", 
              "Frequency (MHz)", 
              "Peak Gain (dBi)")
    
   # Plot Azimuth cuts for different theta values
    # Check if the selected frequency is present in the frequency list
    if selected_frequency in freq_list:
        freq_idx = np.where(np.array(freq_list) == selected_frequency)[0][0]
    else:
        print(f"Error: Selected frequency {selected_frequency} not found in the frequency list.")
        return

    # Extract the total gain data for the selected frequency
    selected_azimuth_freq = Total_Gain_dB[:, freq_idx]
        
    # Add these theta values based on which ones you want to plot
    theta_values_to_plot = [0, 15, 30, 45, 60, 75, 90, 105, 120, 135, 150, 165] 
    lines = []
    labels = []
    
    plt.figure(figsize=(10,6))
    ax = plt.subplot(111, projection='polar')
    for theta in theta_values_to_plot:
        mask = np.abs(theta_angles_deg[:, freq_idx] - theta) < 0.01
        if np.any(mask):
            lines = ax.plot(plot_phi_rad[mask], selected_azimuth_freq[mask], label=f'Theta {theta}°')
            lines.append(lines)
            labels.append(f'Theta {theta}°')

    ax.set_title(f"Gain Pattern Azimuth Cuts - Total Gain at {selected_frequency} MHz")
    ax.legend(lines, labels, loc="upper right", bbox_to_anchor=(1.3, 1))

    # Display the plot
    plt.show()
    
def plot_passive_3d_component(theta_angles_deg, phi_angles_deg, v_gain_dB, h_gain_dB, Total_Gain_dB, freq_list, selected_frequency):
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
    
    # Find the index for the selected frequency
    selected_index = freq_list.index(selected_frequency)
     # Convert frequency list to numpy array
    freq_list = np.array(freq_list)

    # Select values for the selected frequency
    v_gain_selected = v_gain_dB[:, selected_index]
    h_gain_selected = h_gain_dB[:, selected_index]
    total_gain_selected = Total_Gain_dB[:, selected_index]
    selected_theta_angles_deg = theta_angles_deg[:, selected_index]
    selected_phi_angles_deg = phi_angles_deg[:, selected_index]
    
    PHI, THETA = np.meshgrid(selected_phi_angles_deg, selected_theta_angles_deg)

    # Convert angles to radians
    selected_theta_angles_rad = np.deg2rad(THETA)
    selected_phi_angles_rad = np.deg2rad(PHI)

    R = db_to_linear(v_gain_selected)  # Convert gain from dB to linear scale

    # Create a meshgrid
    interp_factor = 1  

    X = R * np.sin(selected_theta_angles_rad) * np.cos(selected_phi_angles_rad)
    Y = R * np.sin(selected_theta_angles_rad) * np.sin(selected_phi_angles_rad)
    Z = R * np.cos(selected_theta_angles_rad)

    for counter in range(interp_factor):  # Interpolate between points to increase number of faces
        X = interp_array(X)
        Y = interp_array(Y)
        Z = interp_array(Z)

    # Plotting
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1, projection='3d')

    N = np.sqrt(X**2 + Y**2 + Z**2)
    Rmax = np.max(N)
    N = N / Rmax
    N = interp_array(N)[1::2, 1::2]  # Interpolate for color mapping

    axes_length = 0.65
    ax.plot([0, axes_length*Rmax], [0, 0], [0, 0], linewidth=2, color='red')
    ax.plot([0, 0], [0, axes_length*Rmax], [0, 0], linewidth=2, color='green')
    ax.plot([0, 0], [0, 0], [0, axes_length*Rmax], linewidth=2, color='blue')
    
    mycol = cm.jet(N)
    ax.plot_surface(X, Y, Z, rstride=1, cstride=1, facecolors=mycol, linewidth=0.5, antialiased=True)

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.view_init(azim=300, elev=30)  # Set view angle

    m = cm.ScalarMappable(cmap=cm.jet)
    m.set_array(R)
    fig.colorbar(m, shrink=0.8)

    plt.show()

def interp_array(N1):  # add interpolated rows and columns to array
        N2 = np.empty([int(N1.shape[0]), int(2*N1.shape[1] - 1)])  # insert interpolated columns
        N2[:, 0] = N1[:, 0]  # original column
        for k in range(N1.shape[1] - 1):  # loop through columns
            N2[:, 2*k+1] = np.mean(N1[:, [k, k + 1]], axis=1)  # interpolated column
            N2[:, 2*k+2] = N1[:, k+1]  # original column
        N3 = np.empty([int(2*N2.shape[0]-1), int(N2.shape[1])])  # insert interpolated columns
        N3[0] = N2[0]  # original row
        for k in range(N2.shape[0] - 1):  # loop through rows
            N3[2*k+1] = np.mean(N2[[k, k + 1]], axis=0)  # interpolated row
            N3[2*k+2] = N2[k+1]  # original row
        return N3

def db_to_linear(db_value):
    return 10 ** (db_value / 10)


