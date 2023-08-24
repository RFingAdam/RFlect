import os
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
import pandas as pd
import requests
import json
import webbrowser
import sys

# Set a modern blue color (#2596be)
DARK_BG_COLOR = "#2E2E2E"
LIGHT_TEXT_COLOR = "#FFFFFF"
ACCENT_BLUE_COLOR = "#4A90E2"
BUTTON_COLOR = "#3A3A3A"
HOVER_COLOR = "#4A4A4A"

# Fonts
HEADER_FONT = ("Arial", 14, "bold")
LABEL_FONT = ("Arial", 12)

#___________________________Helper Functions_____________________________________________________

#Read in TRP/Active Scan File
def read_active_file(file_path):
    with open(file_path, 'r') as file:
        data = file.readlines()
    return data

# Read in Passive HPOL/VPOL Files
def read_passive_file(file_path):
    with open(file_path, 'r') as file:
        content = file.readlines()
    return parse_passive_file(content)

# Function to parse the data from HPOL/VPOL files
def parse_passive_file(content):
    # Initialize variables to store PHI and THETA details
    all_data = []
    start_phi, stop_phi, inc_phi = None, None, None
    start_theta, stop_theta, inc_theta = None, None, None
    
    # Extract Theta & PHI details
    for line in content:
        if "Axis1 Start Angle" in line:
            start_phi = float(line.split(":")[1].split("Deg")[0].strip())
        elif "Axis1 Stop Angle" in line:
            stop_phi = float(line.split(":")[1].split("Deg")[0].strip())
        elif "Axis1 Increment" in line:
            inc_phi = float(line.split(":")[1].split("Deg")[0].strip())
        elif "Axis2 Start Angle" in line:
            start_theta = float(line.split(":")[1].split("Deg")[0].strip())
        elif "Axis2 Stop Angle" in line:
            stop_theta = float(line.split(":")[1].split("Deg")[0].strip())
        elif "Axis2 Increment" in line:
            inc_theta = float(line.split(":")[1].split("Deg")[0].strip())
    
    # Calculate the expected number of data points
    theta_points = (stop_theta - start_theta) / inc_theta + 1
    phi_points = (stop_phi - start_phi) / inc_phi + 1
    data_points = int(theta_points * phi_points)

    # Extract data for each frequency
    while content:
        cal_factor_line = next((line for line in content if "Cal Std Antenna Peak Gain Factor" in line), None)
        if not cal_factor_line:
            break
        cal_factor_value = float(cal_factor_line.split('=')[1].strip().split(' ')[0])

        freq_line = next((line for line in content if "Test Frequency = " in line), None)
        freq_value = float(freq_line.split('=')[1].strip().split(' ')[0])

        start_index = next((i for i, line in enumerate(content) if "THETA\t  PHI\t  Mag\t Phase" in line), None)
        if start_index is None:
            break
        start_index += 2

        data_lines = content[start_index:start_index+data_points]
        theta, phi, mag, phase = [], [], [], []
        for line in data_lines:
            line_data = line.strip().split('\t')
            if len(line_data) != 4:
                break
            t, p, m, ph = map(float, line_data)
            theta.append(t)
            phi.append(p)
            mag.append(m)
            phase.append(ph)

        freq_data = {
            "frequency": freq_value,
            "cal_factor": cal_factor_value,
            "theta": theta,
            "phi": phi,
            "mag": mag,
            "phase": phase
        }
        all_data.append(freq_data)
        content = content[start_index + data_points:]

    return all_data, start_phi, stop_phi, inc_phi, start_theta, stop_theta, inc_theta

#Auto Determine Polarization for HPOL & VPOL Files
def determine_polarization(file_path):
    with open(file_path, 'r') as f:
        content = f.read()
        if "Horizontal Polarization" in content:
            return "HPol"
        else:
            return "VPol"
        
#Verify angle data and frequencies are not mismatched      
def angles_match(start_phi_h, stop_phi_h, inc_phi_h, start_theta_h, stop_theta_h, inc_theta_h,
                            start_phi_v, stop_phi_v, inc_phi_v, start_theta_v, stop_theta_v, inc_theta_v):

    return (start_phi_h == start_phi_v and stop_phi_h == stop_phi_v and inc_phi_h == inc_phi_v and
            start_theta_h == start_theta_v and stop_theta_h == stop_theta_v and inc_theta_h == inc_theta_v)

#Extract Frequency points for selection in the drop-down menu      
def extract_passive_frequencies(file_path):
    with open(file_path, 'r') as file:
        content = file.readlines()

    # Extracting frequencies
    frequencies = [float(line.split("=")[1].split()[0]) for line in content if "Test Frequency" in line]

    return frequencies

#Calculate Total Gain Vector and add cable loss etc - Use Phase for future implementation?
def calculate_passive_variables(hpol_data, vpol_data, cable_loss, start_phi, stop_phi, inc_phi, start_theta, stop_theta, inc_theta, freq_list, selected_frequency):
    theta_points = int((stop_theta - start_theta) / inc_theta + 1)
    phi_points = int((stop_phi - start_phi) / inc_phi + 1)
    data_points = theta_points * phi_points

    theta_angles_deg = np.zeros((data_points, len(freq_list)))
    phi_angles_deg = np.zeros((data_points, len(freq_list)))
    v_gain_dB = np.zeros((data_points, len(freq_list)))
    h_gain_dB = np.zeros((data_points, len(freq_list)))
    v_phase = np.zeros((data_points, len(freq_list)))
    h_phase = np.zeros((data_points, len(freq_list)))

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
    
#plot passive data
def plot_2d_passive_data(theta_angles_deg, phi_angles_deg, v_gain_dB, h_gain_dB, Total_Gain_dB, freq_list, selected_frequency):
    # Convert angles from degrees to radians
    plot_theta_rad = np.radians(theta_angles_deg)
    plot_phi_rad = np.radians(phi_angles_deg)

    # Calculate Average Gain per Frequency in dB
    Average_Gain_dB = 10 * np.log10(np.sum(np.pi/2 * np.sin(plot_theta_rad) * 10**(Total_Gain_dB/10), axis=0) / (theta_angles_deg.shape[0]))
    
     # Plot Efficiency in dB
    plt.figure(figsize=(10,6))
    plt.plot(freq_list, Average_Gain_dB, color='b', linewidth=2)
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    plt.title("Average Radiated Efficiency Versus Frequency (dB)")
    plt.xlabel("Frequency (MHz)")
    plt.ylabel("Efficiency (dB)")
    plt.tight_layout()
    plt.show()
    
    # Convert Average_Gain_dB to Efficiency Percentage and Plot
    Average_Gain_percentage = 100 * 10**(Average_Gain_dB / 10)
    plt.figure(figsize=(10,6))
    plt.plot(freq_list, Average_Gain_percentage, color='b', linewidth=2)
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    plt.title("Average Radiated Efficiency Versus Frequency (%)")
    plt.xlabel("Frequency (MHz)")
    plt.ylabel("Efficiency (%)")
    plt.ylim([10*round(0.08*min(Average_Gain_percentage)), 100])
    plt.tight_layout()
    plt.show()
    
    # Calculate Peak Gain as the maximum gain across all angles for each frequency
    Peak_Gain_dB = np.max(Total_Gain_dB, axis=0)
    
    # Plot Peak Gain
    plt.figure(figsize=(10,6))
    plt.plot(freq_list, Peak_Gain_dB, color='r', linewidth=2)
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    plt.title("Peak Gain Versus Frequency")
    plt.xlabel("Frequency (MHz)")
    plt.ylabel("Peak Gain (dBi)")
    plt.ylim([min(Peak_Gain_dB)-2, max(Peak_Gain_dB)+2])
    plt.tight_layout()
    plt.show()
    
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

def spherical_to_cartesian(r, theta, phi):
    """
    Convert spherical coordinates to cartesian coordinates.
    """
    x = r * np.sin(theta) * np.cos(phi)
    y = r * np.sin(theta) * np.sin(phi)
    z = r * np.cos(theta)
    return x, y, z



# GUI Class Implementation______________________________________________________________________
class AntennaPlotGUI:
    def resource_path(relative_path):
        """ Get absolute path to resource, works for dev and for PyInstaller """
        try:
            # PyInstaller creates a temp folder and stores path in _MEIPASS
            base_path = sys._MEIPASS
        except Exception:
            base_path = os.path.abspath(".")

        return os.path.join(base_path, relative_path) 
    
    settings_path = resource_path("settings.json")
    with open(settings_path, "r") as file:
        settings = json.load(file)
        CURRENT_VERSION = settings["CURRENT_VERSION"]

    def __init__(self, root):
        self.root = root
        self.root.title("RFlect")
        self.root.geometry("450x300")  # Set a reasonable window size

        self.frequency_var = tk.StringVar(self.root)
        self.freq_list = []      
        # Attributes for file paths
        self.hpol_file_path = None
        self.active_file_path = None
        self.vpol_file_path = None

        #initializing VSWR/S11 settings
        self.saved_limit1_freq1 = 0.0
        self.saved_limit1_freq2 = 0.0
        self.saved_limit1_start = 0.0
        self.saved_limit1_stop = 0.0
        self.saved_limit2_freq1 = 0.0
        self.saved_limit2_freq2 = 0.0
        self.saved_limit2_start = 0.0
        self.saved_limit2_stop = 0.0

        self.plot_type_var = tk.StringVar()
        self.plot_type_var.set("HPOL/VPOL")  # default value
        
        self.passive_scan_type = tk.StringVar()
        self.passive_scan_type.set("HPOL/VPOL")  # Default value

        
        # Determine the path of the current script or packaged executable
        current_path = os.path.dirname(os.path.abspath(__file__))

        # Construct the full path to the logo
        logo_path = os.path.join(current_path, 'assets/smith_logo.png')

        # Load the logo
        self.logo_image = tk.PhotoImage(file=logo_path)
        self.root.iconphoto(False, self.logo_image)  # Set the logo as the window icon

        # GUI Elements with updated styling
        self.label_scan_type = tk.Label(self.root, text="Select Measurement Type:", bg=DARK_BG_COLOR, fg=LIGHT_TEXT_COLOR)
        self.label_scan_type.grid(row=1, column=0, pady=10, columnspan=2)

        self.scan_type = tk.StringVar()
        self.scan_type.set("passive")

        active_rb = tk.Radiobutton(self.root, text="Active", variable=self.scan_type, value="active",
                                   background=BUTTON_COLOR, foreground=LIGHT_TEXT_COLOR, selectcolor=DARK_BG_COLOR,
                                   activebackground=ACCENT_BLUE_COLOR, activeforeground=LIGHT_TEXT_COLOR,
                                   command=self.update_visibility)
        active_rb.grid(row=2, column=0, pady=5)

        passive_rb = tk.Radiobutton(self.root, text="Passive", variable=self.scan_type, value="passive",
                                    background=BUTTON_COLOR, foreground=LIGHT_TEXT_COLOR, selectcolor=DARK_BG_COLOR,
                                    activebackground=ACCENT_BLUE_COLOR, activeforeground=LIGHT_TEXT_COLOR,
                                    command=self.update_visibility)
        passive_rb.grid(row=2, column=1, pady=5)

        return_loss = tk.Radiobutton(self.root, text="VNA (.csv)", variable=self.scan_type, value="vswr",
                                     background=BUTTON_COLOR, foreground=LIGHT_TEXT_COLOR, selectcolor=DARK_BG_COLOR,
                                     activebackground=ACCENT_BLUE_COLOR, activeforeground=LIGHT_TEXT_COLOR,
                                     command=self.update_visibility)
        return_loss.grid(row=2, column=2, pady=5)

        self.btn_import = tk.Button(self.root, text="Import File(s)", command=self.import_files, bg=ACCENT_BLUE_COLOR, fg=LIGHT_TEXT_COLOR)
        self.btn_import.grid(row=2, column=3, columnspan=2, pady=10, padx=15)
        
       
        # Cable Loss input for Passive scans
        self.label_cable_loss = tk.Label(self.root, text='Cable Loss:', bg=DARK_BG_COLOR, fg=LIGHT_TEXT_COLOR)
        self.cable_loss = tk.StringVar(self.root, value="0.0")
        self.cable_loss_input = tk.Entry(self.root, textvariable=self.cable_loss, bg=DARK_BG_COLOR, fg=LIGHT_TEXT_COLOR)
        
        # Initially, since Passive is default, show the cable loss input
        self.label_cable_loss.grid(row=4, column=0, pady=5)
        self.cable_loss_input.grid(row=4, column=1, pady=5, padx=5)

        # Attributes for dropdown (Combobox) to select frequency
        self.available_frequencies = []
        self.selected_frequency = tk.StringVar()

        # Dropdown (Combobox) for selecting frequency
        self.label_frequency = tk.Label(self.root, text='Select Frequency:', bg=DARK_BG_COLOR, fg=LIGHT_TEXT_COLOR)
        self.label_frequency.grid(row=3, column=0, pady=5)
        self.frequency_dropdown = ttk.Combobox(self.root, textvariable=self.selected_frequency, values=self.available_frequencies, state='readonly')
        self.frequency_dropdown.grid(row=3, column=1, pady=5)

        self.btn_view_results = tk.Button(self.root, text="View Results", command=self.process_data, bg=ACCENT_BLUE_COLOR, fg=LIGHT_TEXT_COLOR)
        self.btn_view_results.grid(row=5, column=0, pady=10, padx=10)

        self.btn_save_to_file = tk.Button(self.root, text="Save Results to File", command=self.save_to_results_folder, bg=ACCENT_BLUE_COLOR, fg=LIGHT_TEXT_COLOR)
        self.btn_save_to_file.grid(row=5, column=1, pady=10, padx=10)
        self.btn_settings = tk.Button(self.root, text="Settings", command=self.show_settings, bg=ACCENT_BLUE_COLOR, fg=LIGHT_TEXT_COLOR)
        self.btn_settings.grid(row=5, column=2, pady=10, padx=10)

        # Update background color for the entire window
        self.root.config(bg=DARK_BG_COLOR)
    
        # Bind hover effect to the buttons
        buttons = [self.btn_import, self.btn_view_results, self.btn_save_to_file, self.btn_settings]
        for btn in buttons:
            btn.bind("<Enter>", self.on_enter)
            btn.bind("<Leave>", self.on_leave)

        # Title
        self.title_label = tk.Label(self.root, text="RFlect - Antenna Plot Tool", bg=DARK_BG_COLOR, fg=LIGHT_TEXT_COLOR, font=HEADER_FONT)
        self.title_label.grid(row=0, column=0, pady=(10, 0), columnspan=6, sticky="n")
        self.update_visibility()

        # Check for updates
        self.check_for_updates()

        #Check Release Version on Github
    def get_latest_release(self):
        owner = "RFingAdam"
        repo = "RFlect"
        url = f"https://api.github.com/repos/{owner}/{repo}/releases/latest"
        
        response = requests.get(url)
        
        if response.status_code == 200:
            data = response.json()
            latest_version = data["tag_name"]
            release_url = data["html_url"]
            return latest_version, release_url
        else:
            return None, None
        
    
      
    #Downloads latest release   
    def download_latest_release(self, url):
        """
        This function opens the given URL in the default web browser, 
        effectively letting the user download the release.
        """
        webbrowser.open(url)


    def check_for_updates(self):
        latest_version, release_url = self.get_latest_release()
        if latest_version and latest_version != AntennaPlotGUI.CURRENT_VERSION:
            # Create a simple tkinter window for the dialog
            root = tk.Tk()
            root.withdraw()  # Hide the main window

            answer = messagebox.askyesno("Update Available",
                                        f"A new version {latest_version} is available! Would you like to download it?")

            if answer:
              self.download_latest_release(release_url)
    # Hover effect functions
    def on_enter(self, e):
        e.widget.original_color = e.widget['background']
        e.widget['background'] = HOVER_COLOR

    def on_leave(self, e):
        e.widget['background'] = e.widget.original_color

    #Updates GUI Visibility depending on selections
    def update_visibility(self):
        scan_type_value = self.scan_type.get()
        if scan_type_value == "active":
            # Hide the cable loss input for Active scan type
            self.label_cable_loss.grid_forget()
            self.cable_loss_input.grid_forget()
            
            # Hide the frequency-related widgets
            self.label_frequency.grid_forget()
            self.frequency_dropdown.grid_forget()

            self.btn_view_results.grid(row=5, column=0, pady=10)
        elif scan_type_value == "passive":
            # Check the passive scan type and adjust visibility
            if self.passive_scan_type.get() == "G&D":
                self.label_frequency.grid_forget()
                self.frequency_dropdown.grid_forget()
            else:
                self.label_frequency.grid(row=3, column=0, pady=5)
                self.frequency_dropdown.grid(row=3, column=1, pady=5)
            
            # Show the cable loss input
            self.label_cable_loss.grid(row=4, column=0, pady=5)
            self.cable_loss_input.grid(row=4, column=1, pady=5, padx=5)
            self.btn_view_results.grid(row=5, column=0, pady=10)
                            
        elif scan_type_value == "vswr":
            # Hide the cable loss input for Active scan type
            self.label_cable_loss.grid_forget()
            self.cable_loss_input.grid_forget()
            
            # Hide the frequency-related widgets
            self.label_frequency.grid_forget()
            self.frequency_dropdown.grid_forget()
            self.btn_view_results.grid_forget()


    def show_settings(self):
        scan_type_value = self.scan_type.get()
        settings_window = tk.Toplevel(self.root)
        settings_window.geometry("600x200")  # Increase the size
        settings_window.title(f"{scan_type_value.capitalize()} Settings")
        if scan_type_value == "active":
            # Show settings specific to active scan
            label = tk.Label(settings_window, text="Placeholder for Active settings")
            label.grid(row=0, column=0, columnspan=2, pady=20)
            # Add more active-specific settings here
        elif scan_type_value == "passive":
          # Show settings specific to passive scan
            label = tk.Label(settings_window, text="Passive Plot Settings")
            label.grid(row=0, column=0, columnspan=2, pady=10)

             # Radiobuttons for choosing plot type
            self.plot_type_var = tk.StringVar(value=self.passive_scan_type.get())

            r1 = tk.Radiobutton(settings_window, text="G&D", variable=self.plot_type_var, value="G&D")
            r1.grid(row=1, column=1, sticky=tk.W, padx=20)

            r2 = tk.Radiobutton(settings_window, text="VPOL/HPOL", variable=self.plot_type_var, value="VPOL/HPOL")
            r2.grid(row=1, column=0, sticky=tk.W, padx=20)

            # Update the radiobuttons with saved values if they exist
            if self.passive_scan_type.get() == "G&D":
                r1.select()
            else:
                r2.select()

            def save_passive_settings():
                # Save the chosen plot type
                self.passive_scan_type.set(self.plot_type_var.get())
                # Update the visibility of the main GUI
                self.update_visibility()
                # Close the settings window after saving
                settings_window.destroy()


            save_button = tk.Button(settings_window, text="Save Settings", command=save_passive_settings, bg=ACCENT_BLUE_COLOR, fg=LIGHT_TEXT_COLOR)
            save_button.grid(row=2, column=0, columnspan=2, pady=20)
            
        elif scan_type_value == "vswr":
            # Show settings specific to VNA
            label = tk.Label(settings_window, text="VSWR/Return Loss Plot Settings")
            label.grid(row=0, column=0, columnspan=2, pady=10)
            
            def save_vswr_settings():
               # Save the entered values with checks
                self.saved_limit1_freq1 = self.limit1_freq1.get()
                self.saved_limit1_freq2 = self.limit1_freq2.get()
                self.saved_limit1_start = self.limit1_val1.get()
                self.saved_limit1_stop = self.limit1_val2.get()

                self.saved_limit2_freq1 = self.limit2_freq1.get()
                self.saved_limit2_freq2 = self.limit2_freq2.get()
                self.saved_limit2_start = self.limit2_val1.get()
                self.saved_limit2_stop = self.limit2_val2.get()
                # Close the settings window after saving
                settings_window.destroy()

            def default_vswr_settings():
                # Set the entry values to default
                DEFAULT_LIMIT1_FREQ1 = 0.0
                DEFAULT_LIMIT1_FREQ2 = 0.0
                DEFAULT_LIMIT1_START = 0.0
                DEFAULT_LIMIT1_STOP = 0.0

                DEFAULT_LIMIT2_FREQ1 = 0.0
                DEFAULT_LIMIT2_FREQ2 = 0.0
                DEFAULT_LIMIT2_START = 0.0
                DEFAULT_LIMIT2_STOP = 0.0
                self.limit1_freq1.set(DEFAULT_LIMIT1_FREQ1)
                self.limit1_freq2.set(DEFAULT_LIMIT1_FREQ2)
                self.limit1_val1.set(DEFAULT_LIMIT1_START)
                self.limit1_val2.set(DEFAULT_LIMIT1_STOP)
                self.limit2_freq1.set(DEFAULT_LIMIT2_FREQ1)
                self.limit2_freq2.set(DEFAULT_LIMIT2_FREQ2)
                self.limit2_val1.set(DEFAULT_LIMIT2_START)
                self.limit2_val2.set(DEFAULT_LIMIT2_STOP)

                # Update the saved settings variables to default values
                self.saved_limit1_freq1 = DEFAULT_LIMIT1_FREQ1
                self.saved_limit1_freq2 = DEFAULT_LIMIT1_FREQ2
                self.saved_limit1_start = DEFAULT_LIMIT1_START
                self.saved_limit1_stop = DEFAULT_LIMIT1_STOP
                self.saved_limit2_freq1 = DEFAULT_LIMIT2_FREQ1
                self.saved_limit2_freq2 = DEFAULT_LIMIT2_FREQ2
                self.saved_limit2_start = DEFAULT_LIMIT2_START
                self.saved_limit2_stop = DEFAULT_LIMIT2_STOP

            # Limit 1
            tk.Label(settings_window, text="Limit 1 Frequency Start (GHz):").grid(row=1, column=0)
            self.limit1_freq1 = tk.DoubleVar()
            tk.Entry(settings_window, textvariable=self.limit1_freq1).grid(row=1, column=1)
            
            tk.Label(settings_window, text="Limit 1 Value Start:").grid(row=1, column=2)
            self.limit1_val1 = tk.DoubleVar()
            tk.Entry(settings_window, textvariable=self.limit1_val1).grid(row=1, column=3)
            
            tk.Label(settings_window, text="Limit 1 Frequency End (GHz):").grid(row=2, column=0)
            self.limit1_freq2 = tk.DoubleVar()
            tk.Entry(settings_window, textvariable=self.limit1_freq2).grid(row=2, column=1)
            
            tk.Label(settings_window, text="Limit 1 Value End:").grid(row=2, column=2)
            self.limit1_val2 = tk.DoubleVar()
            tk.Entry(settings_window, textvariable=self.limit1_val2).grid(row=2, column=3)
            
            # Limit 2
            tk.Label(settings_window, text="Limit 2 Frequency Start (GHz):").grid(row=3, column=0)
            self.limit2_freq1 = tk.DoubleVar()
            tk.Entry(settings_window, textvariable=self.limit2_freq1).grid(row=3, column=1)
            
            tk.Label(settings_window, text="Limit 2 Value Start:").grid(row=3, column=2)
            self.limit2_val1 = tk.DoubleVar()
            tk.Entry(settings_window, textvariable=self.limit2_val1).grid(row=3, column=3)
            
            tk.Label(settings_window, text="Limit 2 Frequency End (GHz):").grid(row=4, column=0)
            self.limit2_freq2 = tk.DoubleVar()
            tk.Entry(settings_window, textvariable=self.limit2_freq2).grid(row=4, column=1)
            
            tk.Label(settings_window, text="Limit 2 Value End:").grid(row=4, column=2)
            self.limit2_val2 = tk.DoubleVar()
            tk.Entry(settings_window, textvariable=self.limit2_val2).grid(row=4, column=3)
        
            # Update the input fields with saved values if they exist
            if hasattr(self, 'saved_limit1_freq1'):
                self.limit1_freq1.set(self.saved_limit1_freq1)
                self.limit1_freq2.set(self.saved_limit1_freq2)
                self.limit1_val1.set(self.saved_limit1_start)
                self.limit1_val2.set(self.saved_limit1_stop)
                self.limit2_freq1.set(self.saved_limit2_freq1)
                self.limit2_freq2.set(self.saved_limit2_freq2)
                self.limit2_val1.set(self.saved_limit2_start)
                self.limit2_val2.set(self.saved_limit2_stop)

            save_button = tk.Button(settings_window, text="Save Settings", command=save_vswr_settings, bg=ACCENT_BLUE_COLOR, fg=LIGHT_TEXT_COLOR)
            save_button.grid(row=5, column=0, columnspan=2, pady=20)
            default_button = tk.Button(settings_window, text="Default Settings", command=default_vswr_settings, bg=ACCENT_BLUE_COLOR, fg=LIGHT_TEXT_COLOR)
            default_button.grid(row=5, column=2, columnspan=2, pady=20)
    
    def process_gd_file(self, filepath):
        """
        Reads the G&D file and extracts frequency, gain, directivity, and efficiency data.
        """
        frequency = []
        gain = []
        directivity = []
        efficiency = []

        with open(filepath, 'r') as file:
            lines = file.readlines()

        # Find the start of the data
        for i, line in enumerate(lines):
            if "Freq" in line:
                data_start_line = i + 10
                break
        else:
            raise Exception("Could not find the start of the data in the provided file.")

        # Extract data
        for line in lines[data_start_line:]:
            if line.strip() == "":
                break
            parts = line.split()
            frequency.append(float(parts[0]))
            gain.append(float(parts[1]))
            directivity.append(float(parts[2]))
            efficiency.append(float(parts[3]))

        return {
            'Frequency': frequency,
            'Gain': gain,
            'Directivity': directivity,
            'Efficiency': efficiency
        }    
    
    def plot_gd_data(self, datasets, labels):
         # Prompt user for x-axis range
        x_range_input = tk.simpledialog.askstring("Input", "Enter x-axis range as 'min,max' (or leave blank for auto-scale):", parent=self.root)

        # Initialize x_min and x_max to None (auto-scale)
        x_min, x_max = None, None

        # Parse the input if it's not empty
        if x_range_input:
            try:
                x_values = x_range_input.split(',')
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

    def update_passive_frequency_list(self):
        # Extracting frequencies from the HPOL file
        self.freq_list = extract_passive_frequencies(self.hpol_file_path)
        
        # Update the frequency dropdown with available frequencies
        if self.freq_list:
            # Set the values for the dropdown
            self.frequency_dropdown['values'] = self.freq_list
            
            # Set the default start frequency
            self.selected_frequency.set(self.freq_list[0])
            
            # Enable the dropdown
            self.frequency_dropdown['state'] = 'readonly'
        else:
            # If no frequencies found, clear the dropdown
            self.frequency_dropdown['values'] = []
            self.selected_frequency.set('')
            self.frequency_dropdown['state'] = 'disabled'
    
    #Method checks for matching data between two passive scan files HPOL and VPOL to ensure they are from the same dataset
    def check_matching_files(self, file1, file2):
        # Extract filename without extension for comparison
        filename1 = os.path.splitext(os.path.basename(file1))[0]
        filename2 = os.path.splitext(os.path.basename(file2))[0]
        
        # Check if filenames match excluding the last 4 characters (polarization part)
        if filename1[:-4] != filename2[:-4]:
            return False, "File names do not match."

        # Extract frequency and angular data from files
        with open(file1, 'r') as f1, open(file2, 'r') as f2:
            content1 = f1.readlines()
            content2 = f2.readlines()

        # Extracting required data for comparison
        freq1 = content1[33]
        freq2 = content2[33]
        phi1 = content1[42:44]
        phi2 = content2[42:44]
        theta1 = content1[49:51]
        theta2 = content2[49:51]

        if freq1 != freq2 or phi1 != phi2 or theta1 != theta2:
            return False, "The selected files have mismatched frequency or angle data."

        return True, ""     

    #Method to import TRP or HPOL/VPOL data files for analysis       
    def import_files(self):
        if self.scan_type.get() == "active":
            self.TRP_file_path = filedialog.askopenfilename(title="Select the TRP Data File",
                                                            filetypes=[("Text files", "*.txt"), ("All files", "*.*")])
            self.update_active_frequency_list()
        elif self.scan_type.get() == "passive" and self.passive_scan_type.get() == "G&D":
            num_files = tk.simpledialog.askinteger("Input", "How many files would you like to import?", parent=self.root)
            if not num_files:
                return
            datasets = []
            file_names = []
            for _ in range(num_files):
                filepath = filedialog.askopenfilename(filetypes=[("Text files", "*.txt")])
                if not filepath:
                    return
                file_name = os.path.basename(filepath).replace('.txt', '')  # Extract filename without extension
                file_names.append(file_name)
                data = self.process_gd_file(filepath)
                datasets.append(data)
                
            self.plot_gd_data(datasets, file_names)

        elif self.scan_type.get() == "passive" and self.passive_scan_type.get() == "VPOL/HPOL":
            
            first_file = filedialog.askopenfilename(title="Select the First Data File",
                                                filetypes=[("Text files", "*.txt"), ("All files", "*.*")])
            second_file = filedialog.askopenfilename(title="Select the Second Data File",
                                                    filetypes=[("Text files", "*.txt"), ("All files", "*.*")])
            if first_file and second_file:
                # Determine file polarizations
                first_polarization = determine_polarization(first_file)
                second_polarization = determine_polarization(second_file)
                
                 # Check if both files have the same polarization
                if first_polarization == second_polarization:
                    messagebox.showerror("Error", "Both files cannot be of the same polarization.")
                    return

                if first_polarization == "HPol":
                    self.hpol_file_path = first_file
                    self.vpol_file_path = second_file
                else:
                    self.hpol_file_path = second_file
                    self.vpol_file_path = first_file

                #Check if File names match and data is consistent between files
                match, message = self.check_matching_files(self.hpol_file_path, self.vpol_file_path)
                if not match:
                    messagebox.showerror("Error", message)
                    return
                self.update_passive_frequency_list()

        elif self.scan_type.get() == "vswr":
            num_files = tk.simpledialog.askinteger("Input", "How many files do you want to import?")
            if not num_files:
                return

            file_paths = []
            for _ in range(num_files):
                file_path = filedialog.askopenfilename(title="Select the VSWR/Return Loss File(s)",
                                                       filetypes=[("CSV files", "*.csv"), ("All files", "*.*")])
                if not file_path:  # if user cancels the dialog
                    return
                file_paths.append(file_path)

            self.process_vswr_files(file_paths)

    def process_vswr_files(self, file_paths):
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
            if self.saved_limit1_freq1 and self.saved_limit1_freq2 and self.saved_limit1_start and self.saved_limit1_stop:
                print(f"Limit 1: {self.saved_limit1_freq1}, {self.saved_limit1_freq2}, {self.saved_limit1_start}, {self.saved_limit1_stop}")
                ax.plot([self.saved_limit1_freq1, self.saved_limit1_freq2], [self.saved_limit1_start, self.saved_limit1_stop], linewidth=2, zorder=100, color='red', alpha=0.8)
            
            if self.saved_limit2_freq1 and self.saved_limit2_freq2 and self.saved_limit2_start and self.saved_limit2_stop:
                print(f"Limit 2: {self.saved_limit2_freq1}, {self.saved_limit2_freq2}, {self.saved_limit2_start}, {self.saved_limit2_stop}")
                ax.plot([self.saved_limit2_freq1, self.saved_limit2_freq2], [self.saved_limit2_start, self.saved_limit2_stop], linewidth=2, zorder=100, color ='red', alpha=0.8)

            ax.set_xlabel("Frequency (GHz)")
            ax.legend()
            ax.grid(True)
            plt.tight_layout()
        plt.show()
    

    def process_data(self):
        if self.scan_type.get() == "active":
            #perform active calculations and plotting method calls
            #future implementaiton
            return
        elif self.scan_type.get() == "passive":
            #After reading & parsing, hpol_data and vpol_data will be lists of dictionaries. 
            #Each dictionary will represent a frequency point and will contain:
                #'frequency': The frequency value
                #'cal_factor': Calibration factor for that frequency
                #'data': A list of tuples, where each tuple contains (theta, phi, mag, phase).
            parsed_hpol_data, start_phi_h, stop_phi_h, inc_phi_h, start_theta_h, stop_theta_h, inc_theta_h = read_passive_file(self.hpol_file_path)
            hpol_data = parsed_hpol_data

            parsed_vpol_data, start_phi_v, stop_phi_v, inc_phi_v, start_theta_v, stop_theta_v, inc_theta_v = read_passive_file(self.vpol_file_path)
            vpol_data = parsed_vpol_data
            
            #check to see if selected files have mismatched frequency or angle data
            angles_match(start_phi_h, stop_phi_h, inc_phi_h, start_theta_h, stop_theta_h, inc_theta_h,
                            start_phi_v, stop_phi_v, inc_phi_v, start_theta_v, stop_theta_v, inc_theta_v)

            #Call Methods to Calculate Required Variables and Set up variables for plotting 
            passive_variables = calculate_passive_variables(hpol_data, vpol_data, float(self.cable_loss.get()), start_phi_h, stop_phi_h, inc_phi_h, start_theta_h, stop_theta_h, inc_theta_h, self.freq_list, float(self.selected_frequency.get()))
            theta_angles_deg, phi_angles_deg, v_gain_dB, h_gain_dB, Total_Gain_dB = passive_variables
            
            #Call Method to Plot Passive Data
            plot_2d_passive_data(theta_angles_deg, phi_angles_deg, v_gain_dB, h_gain_dB, Total_Gain_dB, self.freq_list, float(self.selected_frequency.get()))
            
            #plot_passive_3d_component(theta_angles_deg, phi_angles_deg, v_gain_dB, h_gain_dB, Total_Gain_dB, self.freq_list, float(self.selected_frequency.get()))

            return
        
    #Save results to folder called when Button pressed
    def save_to_results_folder(self):
        #future implementation
        #save_passive_data(passive_variables, self.freq_list)
        return
    
def main():
    root = tk.Tk()
    app = AntennaPlotGUI(root)
    root.mainloop()

if __name__ == "__main__":
    main()