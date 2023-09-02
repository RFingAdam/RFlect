from file_utils import save_to_results_folder, read_active_file, read_passive_file
from calculations import determine_polarization, angles_match, extract_passive_frequencies, calculate_passive_variables
from plotting import plot_2d_passive_data, plot_passive_3d_component
from config import *

import os
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import requests
import json
import sys
import webbrowser


class AntennaPlotGUI:
    """
    Main GUI class for the RFlect application.

    This class manages the creation and interactions of the main application window and its components.
    """
    
    #Get Current Version from settings.json file
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

        
       # Determine if the script is being run as a standalone script or packaged executable
        if getattr(sys, 'frozen', False):
            # If packaged with PyInstaller, use the temporary folder where PyInstaller extracted the assets
            current_path = sys._MEIPASS
        else:
            # If running as a standalone script, use the script's directory
            current_path = os.path.dirname(os.path.abspath(__file__))

        # Construct the path to the logo
        logo_path = os.path.join(current_path, 'assets', 'smith_logo.png')

        # Load the logo
        self.logo_image = tk.PhotoImage(file=logo_path)

        self.root.iconphoto(False, self.logo_image)  # Set the logo as the window icon
        
        # GUI Elements with updated styling
        self.label_scan_type = tk.Label(self.root, text="Select Measurement Type:", bg=DARK_BG_COLOR, fg=LIGHT_TEXT_COLOR)
        self.label_scan_type.grid(row=1, column=0, pady=10, columnspan=2)

        #Initialize Default Scan Settings
        self.scan_type = tk.StringVar()
        self.scan_type.set("passive")
        self.passive_scan_type = tk.StringVar()
        self.passive_scan_type.set("VPOL/HPOL")

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

        self.btn_save_to_file = tk.Button(self.root, text="Save Results to File", command=save_to_results_folder, bg=ACCENT_BLUE_COLOR, fg=LIGHT_TEXT_COLOR)
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
            #Check if user canceled the first selection
            if not first_file:
                return
            
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
            
            plot_passive_3d_component(theta_angles_deg, phi_angles_deg, v_gain_dB, h_gain_dB, Total_Gain_dB, self.freq_list, float(self.selected_frequency.get()), gain_type="total")
            plot_passive_3d_component(theta_angles_deg, phi_angles_deg, v_gain_dB, h_gain_dB, Total_Gain_dB, self.freq_list, float(self.selected_frequency.get()), gain_type="hpol")
            plot_passive_3d_component(theta_angles_deg, phi_angles_deg, v_gain_dB, h_gain_dB, Total_Gain_dB, self.freq_list, float(self.selected_frequency.get()), gain_type="vpol")

            return
        
    