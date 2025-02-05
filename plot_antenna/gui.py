from file_utils import read_active_file, read_passive_file, check_matching_files, process_gd_file, convert_HpolVpol_files, generate_active_cal_file
from save import save_to_results_folder,generate_report, RFAnalyzer
from calculations import determine_polarization, angles_match, extract_passive_frequencies, calculate_passive_variables, calculate_active_variables, apply_nf2ff_transformation
from plotting import plot_2d_passive_data, plot_passive_3d_component, plot_gd_data, process_vswr_files, plot_active_2d_data, plot_active_3d_data
from config import *
from groupdelay import process_groupdelay_files
from file_utils import parse_2port_data 

import os
import tkinter as tk
from tkinter import filedialog, messagebox, ttk, Menu, simpledialog
from tkinter.simpledialog import askstring
import tkinter.scrolledtext as ScrolledText
import requests
import json
import sys
import webbrowser
import matplotlib
import pandas as pd
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import numpy as np

# Custom class to write to both console and Text widget
class DualOutput:
    def __init__(self, widget, stream=None):
        self.widget = widget
        self.stream = stream

    def write(self, string):
        self.widget.configure(state='normal')
        self.widget.insert('end', string)
        self.widget.configure(state='disabled')
        self.widget.see('end')
        if self.stream:  # Only write to the stream if it's valid
            self.stream.write(string)
            self.stream.flush()

    def flush(self):
        pass

def calculate_min_max_parameters(file_paths, bands, param_name):
    """
    Calculate Min/Max for a specified parameter (e.g., VSWR or S21) across defined frequency bands.
    """
    results = []

    for i, (freq_min, freq_max) in enumerate(bands, start=1):
        band_label = f"({freq_min}-{freq_max} MHz)"
        band_data = []

        for file_path in file_paths:
            # Parse data from file
            data = parse_2port_data(file_path)

            # Ensure '! Stimulus(Hz)' exists
            if '! Stimulus(Hz)' not in data.columns:
                raise ValueError(f"File '{file_path}' does not have a '! Stimulus(Hz)' column.")

            freqs_mhz = data['! Stimulus(Hz)'] / 1e6

            # Determine the parameter column based on param_name
            if param_name.upper() == "VSWR":
                candidates = [c for c in data.columns if 'SWR' in c.upper()]
                if not candidates:
                    raise ValueError(f"No SWR columns found in '{file_path}' for VSWR calculation.")
            else:
                candidates = [c for c in data.columns if param_name.upper() in c.upper()]
                if not candidates:
                    raise ValueError(f"No columns containing '{param_name}' found in '{file_path}'.")
            
            # Use the first matching column
            desired_col = candidates[0]

            # Extract values and filter by frequency range
            values = pd.to_numeric(data[desired_col], errors='coerce').dropna()
            within_range = (freqs_mhz >= freq_min) & (freqs_mhz <= freq_max)

            # Calculate min/max for the parameter
            if np.any(within_range):
                min_val = values[within_range].min()
                max_val = values[within_range].max()
            else:
                min_val = None
                max_val = None

            band_data.append((file_path, min_val, max_val))

        results.append((band_label, band_data))

    return results


def display_parameter_table(results, param_name, parent=None):
    """
    Display the results from calculate_min_max_parameters in a Tkinter Toplevel window.

    Parameters:
        results (list): The results list from calculate_min_max_parameter.
        param_name (str): Name of the parameter (e.g., "VSWR").
        parent: The parent Tk widget (e.g., self.root from your main GUI class).
    """
    if parent is None:
        parent = tk.Tk()
    else:
        parent = tk.Toplevel(parent)

    parent.title(f"Min/Max {param_name} Results by Band")

    row = 0
    for band_label, band_data in results:
        # Display band header
        tk.Label(parent, text=band_label, font=("Arial", 12, "bold")).grid(row=row, column=0, columnspan=3, pady=5)
        row += 1

        # Table headers for each band
        headers = ["File", f"Min {param_name}", f"Max {param_name}"]
        for col, header in enumerate(headers):
            tk.Label(parent, text=header, font=("Arial", 10, "bold")).grid(row=row, column=col, padx=10, pady=5)
        row += 1

        # Data rows for each file in the band
        for file, min_val, max_val in band_data:
            tk.Label(parent, text=os.path.basename(file)).grid(row=row, column=0, padx=10, pady=5)
            tk.Label(parent, text=f"{min_val:.2f}" if min_val is not None else "N/A").grid(row=row, column=1, padx=10, pady=5)
            tk.Label(parent, text=f"{max_val:.2f}" if max_val is not None else "N/A").grid(row=row, column=2, padx=10, pady=5)
            row += 1

        # Add some space between bands
        row += 1

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
            base_path = os.path.join(os.path.dirname(__file__), '..')

        return os.path.abspath(os.path.join(base_path, relative_path))
    
    settings_path = resource_path("settings.json")
    with open(settings_path, "r") as file:

        settings = json.load(file)
        CURRENT_VERSION = settings["CURRENT_VERSION"]

    def __init__(self, root):
        self.root = root
        self.root.title("RFlect")
        self.root.geometry("600x400")  # Set a reasonable window size

        self.frequency_var = tk.StringVar(self.root)
        self.freq_list = []      
        # Attributes for file paths
        self.hpol_file_path = None
        self.TRP_file_path = None
        self.vpol_file_path = None
        self.min_max_vswr_var = tk.BooleanVar(value=getattr(self, 'saved_min_max_vswr', False))  # Default to False if not set
        
        self.min_max_eff_gain_var = tk.BooleanVar(value=False)
        
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
        
        self.interpolate_3d_plots = True  # Default value; set to True if you want interpolation enabled by default

        # Determine if the script is being run as a standalone script or packaged executable
        if getattr(sys, 'frozen', False):
            # If packaged with PyInstaller, use the temporary folder where PyInstaller extracted the assets
            current_path = sys._MEIPASS
        else:
            # If running as a standalone script, use the script's directory and go up one level to the parent directory
            current_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

        # Construct the path to the logo
        logo_path = os.path.join(current_path, 'assets', 'smith_logo.png')

        # Load the logo
        self.logo_image = tk.PhotoImage(file=logo_path)

        self.root.iconphoto(False, self.logo_image)  # Set the logo as the window icon

        # GUI Elements with updated styling
        self.label_scan_type = tk.Label(self.root, text="Select Measurement Type:", bg=DARK_BG_COLOR, fg=LIGHT_TEXT_COLOR)
        self.label_scan_type.grid(row=1, column=0, pady=10, columnspan=2)

        # Initialize Default Scan Settings
        self.scan_type = tk.StringVar()
        self.scan_type.set("passive")
        self.passive_scan_type = tk.StringVar()
        self.passive_scan_type.set("VPOL/HPOL")
        self.datasheet_plots_var = tk.BooleanVar(value=False)
        self.cb_groupdelay_sff_var = tk.BooleanVar(value=False)

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

        # Button for View Results Routine
        self.btn_view_results = tk.Button(self.root, text="View Results", command=self.process_data, bg=ACCENT_BLUE_COLOR, fg=LIGHT_TEXT_COLOR)
        self.btn_view_results.grid(row=5, column=0, pady=10, padx=10)

        # Button for Save Results to file Routine
        self.btn_save_to_file = tk.Button(
            self.root,
            text="Save Results to File",
            command=lambda: self.save_results_to_file(),
            bg=ACCENT_BLUE_COLOR,
            fg=LIGHT_TEXT_COLOR
        )
        self.btn_save_to_file.grid(row=5, column=1, pady=10, padx=10)
        
        # Button for Settings
        self.btn_settings = tk.Button(self.root, text="Settings", command=self.show_settings, bg=ACCENT_BLUE_COLOR, fg=LIGHT_TEXT_COLOR)
        self.btn_settings.grid(row=5, column=3, pady=10, padx=10)
        
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

        # Create a new menu bar
        menubar = Menu(root)

        # Create a new drop-down menu
        tools_menu = Menu(menubar, tearoff=0)  # 'tearoff=0' means the menu can't be separated from the window
        tools_menu.add_command(label="HPOL/VPOL->CST FFS Converter", command=self.open_hpol_vpol_converter)
        tools_menu.add_command(label="Active Chamber Calibration", command=self.open_active_chamber_cal)
        tools_menu.add_command(label="Generate Report", command=self.generate_report_from_directory)

        # Load environment variables from the openai.env file
        openai_api_key = os.getenv('OPENAI_API_KEY') or os.getenv('OPENAI_API_KEY2')

        if openai_api_key:
            tools_menu.add_command(label="Generate Report with AI", command=self.generate_ai_report_from_directory)
        else:
            print("OpenAI API key not found. AI functionality will be disabled.")
        # TODO add more tools here

        # Add the dropdown menu to the menu bar
        menubar.add_cascade(label="Additional Tools", menu=tools_menu)

        # Display the menu bar
        root.config(menu=menubar)

        # Check for updates
        self.check_for_updates()

        # Create a frame to contain the log text area
        log_frame = tk.Frame(self.root)
        log_frame.grid(row=6, column=0, columnspan=4, sticky='nsew')
        
        # Add a ScrolledText widget for log output
        self.log_text = ScrolledText.ScrolledText(log_frame, wrap=tk.WORD, height=10, bg=DARK_BG_COLOR, fg=LIGHT_TEXT_COLOR)
        self.log_text.grid(row=0, column=0, sticky='nsew')
        
        # Make the log frame expandable
        self.root.grid_rowconfigure(6, weight=1)
        self.root.grid_columnconfigure(0, weight=1)

        # Redirect stdout and stderr
        sys.stdout = DualOutput(self.log_text, sys.stdout)
        sys.stderr = DualOutput(self.log_text, sys.stderr)

    # Class Methods
    # Recursively collect image files from the selected directory and subdirectories
    def collect_image_paths(self, directory):
        image_paths = []
        for root, dirs, files in os.walk(directory):
            for file in files:
                if file.endswith(".png"):
                    image_paths.append(os.path.join(root, file))
        return image_paths
    
    # Function to import files from directory to be saved to .docx file
    def generate_report_from_directory(self):
        directory = filedialog.askdirectory(title="Select Directory Containing Measurement Images")
        if not directory:
            messagebox.showerror("Error", "No directory selected.")
            return
        
        # Recursively collect image paths using the instance method
        image_paths = self.collect_image_paths(directory)
        
        if not image_paths:
            messagebox.showerror("Error", "No images found in the selected directory.")
            return

        report_title = simpledialog.askstring("Input", "Enter Report Title:")
        if not report_title:
            messagebox.showerror("Error", "Report title is required.")
            return

        # Initialize the RFAnalyzer without AI
        analyzer = RFAnalyzer(use_ai=False)
        save_path = filedialog.askdirectory(title="Select Directory to Save Report")
        if save_path:
            generate_report(report_title, image_paths, save_path, analyzer)
            messagebox.showinfo("Success", "Report generated successfully.")
        else:
            messagebox.showerror("Error", "No directory selected to save the report.")

    def generate_ai_report_from_directory(self):
        directory = filedialog.askdirectory(title="Select Directory Containing Measurement Images")
        if not directory:
            messagebox.showerror("Error", "No directory selected.")
            return
        
        # Recursively collect image paths using the instance method
        image_paths = self.collect_image_paths(directory)
        
        if not image_paths:
            messagebox.showerror("Error", "No images found in the selected directory.")
            return

        report_title = simpledialog.askstring("Input", "Enter Report Title:")
        if not report_title:
            messagebox.showerror("Error", "Report title is required.")
            return

        # Initialize the RFAnalyzer with AI flag enabled
        analyzer = RFAnalyzer(use_ai=True)
        save_path = filedialog.askdirectory(title="Select Directory to Save Report")
        if save_path:
            generate_report(report_title, image_paths, save_path, analyzer)
            messagebox.showinfo("Success", "AI Report generated successfully.")
        else:
            messagebox.showerror("Error", "No directory selected to save the report.")

    # Function to assign frequency for active or passive scans, since active scans don't require a selected_frequency(measurement is at one freq.)
    def save_results_to_file(self):
        try:
            scan_type = self.scan_type.get()
            
            if scan_type == "active":
                # For active measurements, no need to get the frequency from user input
                frequency = None  # Frequency will come from the active file
                cable_loss = None  # Cable loss is not applicable for active scans
            else:
                # For passive measurements, ensure a frequency is selected
                frequency = self.selected_frequency.get()
                cable_loss = float(self.cable_loss.get())
                if not frequency:
                    messagebox.showerror("Error", "Please select a frequency before saving.")
                    return
                frequency = float(frequency)  # Convert to float
            
            # Now proceed with saving results
            save_to_results_folder(
                frequency,  # Pass frequency (None for active, actual value for passive)
                self.freq_list,
                scan_type,
                self.hpol_file_path,
                self.vpol_file_path,
                self.TRP_file_path,
                cable_loss,
                self.datasheet_plots_var.get(),
                word=False
            )
            
        except ValueError as ve:
            messagebox.showerror("Conversion Error", f"Invalid frequency or cable loss value. Error: {ve}")

    # Function to start HPOL and VPOL gain text file conversion to the format readable by cst for FFS/ Efield data
    def open_hpol_vpol_converter(self):
        # Implement the logic of the converter or open a new window for it
        self.log_message("HPOL/VPOL to CST FFS Converter started...")
        first_file = filedialog.askopenfilename(title="Select the First Data File",
                                                filetypes=[("Text files", "*.txt"), ("All files", "*.*")])
        # Check if user canceled the first selection
        if not first_file:
            self.log_message("File selection canceled.")
            return
        
        second_file = filedialog.askopenfilename(title="Select the Second Data File",
                                                filetypes=[("Text files", "*.txt"), ("All files", "*.*")])
        
        self.vswr_file_path = filedialog.askopenfilename(title="Select the VSWR Data File for Accepted P(W,rms) calculations")
        
        if first_file and second_file:
            # Determine file polarizations
            first_polarization = determine_polarization(first_file)
            second_polarization = determine_polarization(second_file)
            
            # Check if both files have the same polarization
            if first_polarization == second_polarization:
                self.log_message("Error", "Both files cannot be of the same polarization.")
                return

            if first_polarization == "HPol":
                self.hpol_file_path = first_file
                self.vpol_file_path = second_file
            else:
                self.hpol_file_path = second_file
                self.vpol_file_path = first_file

            #Check if File names match and data is consistent between files
            match, message = check_matching_files(self.hpol_file_path, self.vpol_file_path)
            if not match:
                self.log_message("Error", message)
                return
            self.update_passive_frequency_list()

            # Hide buttons not related to this routine
            self.btn_view_results.grid_remove()
            self.btn_save_to_file.grid_remove()
            self.btn_settings.grid_remove()
            self.btn_import.grid_remove()
            self.log_message("CST .ffs file created successfully.")

            # Create the new button if it doesn't exist, or just show it if it does
            if not hasattr(self, 'convert_files_button'):
                self.convert_files_button = tk.Button(self.root, text="Convert Files", command=lambda: convert_HpolVpol_files(self.vswr_file_path, self.hpol_file_path, self.vpol_file_path, float(self.cable_loss.get()), self.freq_list, float(self.selected_frequency.get()), callback=self.update_visibility), bg=ACCENT_BLUE_COLOR, fg=LIGHT_TEXT_COLOR)
                self.convert_files_button.grid(column=0, row=5)  # Adjust grid position as necessary
            else:
                self.convert_files_button.grid()  # This shows the button
    
    # Function to start Active Chamber Calibration routine
    def open_active_chamber_cal(self):
        self.log_message("Active Chamber Calibration routine started....")

        # Implement the logic of the calibration file import
        power_measurement = filedialog.askopenfilename(title="Select the Signal Generator/Power Meter Measurement File",
                                                filetypes=[("Text files", "*.txt"), ("All files", "*.*")])
        # Check if user canceled the first selection
        if not power_measurement:
            self.log_message("File selection canceled.")
            return
        self.power_measurement = power_measurement

        BLPA_HORN_GAIN_STD = filedialog.askopenfilename(title="Select the BLPA or Horn Antenna Gain Standard File",
                                                filetypes=[("Text files", "*.txt"), ("All files", "*.*")])
        # Check if user canceled the first selection
        if not BLPA_HORN_GAIN_STD:
            self.log_message("File selection canceled.")
            return
        
        self.BLPA_HORN_GAIN_STD = BLPA_HORN_GAIN_STD

        first_file = filedialog.askopenfilename(title="Select the BLPA or Horn Antenna HPOL or VPOL File",
                                                    filetypes=[("Text files", "*.txt"), ("All files", "*.*")])
            # Check if user canceled the first selection
        if not first_file:
            self.log_message("File selection canceled.")
            return
        
        second_file = filedialog.askopenfilename(title="Select the other HPOL or VPOL File",
                                                filetypes=[("Text files", "*.txt"), ("All files", "*.*")])
        
        if first_file and second_file:
            # Determine file polarizations
            first_polarization = determine_polarization(first_file)
            second_polarization = determine_polarization(second_file)
            
            # Check if both files have the same polarization
            if first_polarization == second_polarization:
                self.log_message("Error", "Both files cannot be of the same polarization.")
                return

            if first_polarization == "HPol":
                self.hpol_file_path = first_file
                self.vpol_file_path = second_file
            else:
                self.hpol_file_path = second_file
                self.vpol_file_path = first_file

            #Check if File names match and data is consistent between files
            match, message = check_matching_files(self.hpol_file_path, self.vpol_file_path)
            if not match:
                self.log_message("Error", message)
                return
            self.update_passive_frequency_list()

            # Hide buttons not related to this routine
            self.btn_view_results.grid_remove()
            self.btn_save_to_file.grid_remove()
            self.btn_settings.grid_remove()
            self.btn_import.grid_remove()
            
            self.log_message("Active Chamber Calibration File Created Successfully.")

            # Create the new button if it doesn't exist, or just show it if it does
            if not hasattr(self, 'generate_active_cal_button'):
                self.convert_files_button = tk.Button(self.root, text="Generate Calibration File", command=lambda: generate_active_cal_file(self.power_measurement, self.BLPA_HORN_GAIN_STD, self.hpol_file_path, self.vpol_file_path, float(self.cable_loss.get()), self.freq_list, callback=self.update_visibility), bg=ACCENT_BLUE_COLOR, fg=LIGHT_TEXT_COLOR)
                self.convert_files_button.grid(column=0, row=5)  # Adjust grid position as necessary
            else:
                self.convert_files_button.grid()  # This shows the button

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
        if latest_version and latest_version > AntennaPlotGUI.CURRENT_VERSION:
            # Create a simple tkinter window for the dialog
            root = tk.Tk()
            root.withdraw()  # Hide the main window
            self.log_message("Update Available. A new version {latest_version} is available!")
                             
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

    # Updates GUI Visibility depending on selections
    def update_visibility(self):
        # If Convert routing was just run, Remove Convert button
        if hasattr(self, 'convert_files_button'):
            self.convert_files_button.grid_remove()
        if hasattr(self, 'generate_active_cal_button'):
            self.convert_files_button.grid_remove()

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
                # Hide the save results button for G&D passive scans
                self.btn_save_to_file.grid_forget()
            else:
                self.label_frequency.grid(row=3, column=0, pady=5)
                self.frequency_dropdown.grid(row=3, column=1, pady=5)
                # Show the save results button for non-G&D passive scans
                self.btn_save_to_file.grid(row=5, column=1, pady=10)  # Adjusting the row index
            # Show the cable loss input
            self.label_cable_loss.grid(row=4, column=0, pady=5)
            self.cable_loss_input.grid(row=4, column=1, pady=5, padx=5)
            self.btn_view_results.grid(row=5, column=0, pady=10)
            self.btn_settings.grid()

        elif scan_type_value == "vswr":
            # Hide the cable loss input for Active scan type
            self.label_cable_loss.grid_forget()
            self.cable_loss_input.grid_forget()
            
            # Hide the frequency-related widgets
            self.label_frequency.grid_forget()
            self.frequency_dropdown.grid_forget()
            self.btn_view_results.grid_forget()

            # Hide the view results button
            self.btn_view_results.grid_forget()
            # Hide the save results button for .csv/VSWR files
            self.btn_save_to_file.grid_forget()
            
    def show_settings(self):
        # Show Saved or Default Scan Type
        scan_type_value = self.scan_type.get()
        settings_window = tk.Toplevel(self.root)
        settings_window.geometry("600x200")  # Increase the size
        settings_window.title(f"{scan_type_value.capitalize()} Settings")
        if scan_type_value == "active":
            # Show settings specific to active scan
            label = tk.Label(settings_window, text="Active Plot Settings")
            label.grid(row=0, column=0, columnspan=2, pady=20)
            
            # Add checkbox for interpolation setting
            self.interpolate_var = tk.BooleanVar(value=self.interpolate_3d_plots)
            cb_interpolate = tk.Checkbutton(settings_window, text="Interpolate 3D Plots", variable=self.interpolate_var)
            cb_interpolate.grid(row=1, column=0, sticky=tk.W, padx=20)


            def save_active_settings():
                # Save the interpolate setting
                self.interpolate_3d_plots = self.interpolate_var.get()
                # Update the visibility of the main GUI
                self.update_visibility()
                # Close the settings window after saving
                settings_window.destroy()
            save_button = tk.Button(settings_window, text="Save Settings", command=save_active_settings, bg=ACCENT_BLUE_COLOR, fg=LIGHT_TEXT_COLOR)
            save_button.grid(row=3, column=0, columnspan=2, pady=20)

        elif scan_type_value == "passive":
            # Show settings specific to passive scan
            label = tk.Label(settings_window, text="Passive Plot Settings")
            label.grid(row=0, column=0, columnspan=3, pady=10)  # Adjust columnspan if needed
            
            # Radiobuttons for choosing plot type
            self.plot_type_var = tk.StringVar(value=self.passive_scan_type.get())

            r2 = tk.Radiobutton(settings_window, text="VPOL/HPOL", variable=self.plot_type_var, value="VPOL/HPOL")
            r2.grid(row=1, column=0, sticky=tk.W, padx=20)

            r1 = tk.Radiobutton(settings_window, text="G&D", variable=self.plot_type_var, value="G&D")
            r1.grid(row=1, column=1, sticky=tk.W, padx=20)

            # Create the "Datasheet Plots" Checkbutton
            self.cb_datasheet_plots = tk.Checkbutton(settings_window, text="Datasheet Plots", variable=self.datasheet_plots_var)

            # Create the "Min/Max Eff & Gain" Checkbutton
            self.cb_min_max_eff_gain = tk.Checkbutton(settings_window, text="Min/Max Eff & Gain", variable=self.min_max_eff_gain_var)

            def on_radiobutton_change():
                if self.plot_type_var.get() == "G&D":
                    # Hide Datasheet Plots
                    self.cb_datasheet_plots.grid_remove()
                    # Show Min/Max Eff & Gain
                    self.cb_min_max_eff_gain.grid(row=2, column=1, sticky=tk.W, padx=20)
                else:
                    # VPOL/HPOL selected
                    # Show Datasheet Plots
                    self.cb_datasheet_plots.grid(row=2, column=0, sticky=tk.W, padx=20)
                    # Hide Min/Max Eff & Gain
                    self.cb_min_max_eff_gain.grid_remove()

            # Initially set visibility based on current selection
            if self.passive_scan_type.get() == "G&D":
                self.cb_min_max_eff_gain.grid(row=2, column=1, sticky=tk.W, padx=20)
                # Datasheet Plots hidden
            else:
                self.cb_datasheet_plots.grid(row=2, column=0, sticky=tk.W, padx=20)
                # Min/Max Eff & Gain hidden

            r1.config(command=on_radiobutton_change)
            r2.config(command=on_radiobutton_change)

            def save_passive_settings():
                # Save the chosen plot type
                self.passive_scan_type.set(self.plot_type_var.get())
                # Update the visibility of the main GUI
                self.update_visibility()
                # Close the settings window after saving
                settings_window.destroy()

            save_button = tk.Button(settings_window, text="Save Settings", command=save_passive_settings, bg=ACCENT_BLUE_COLOR, fg=LIGHT_TEXT_COLOR)
            save_button.grid(row=3, column=0, columnspan=3, pady=20)
                    
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

                self.cb_groupdelay_sff = self.cb_groupdelay_sff_var.get()
                self.saved_min_max_vswr = self.min_max_vswr_var.get()  # Save checkbox state                
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

                self.cb_groupdelay_sff_var.set(False)
                self.saved_min_max_vswr = False 
                if hasattr(self, 'saved_min_max_vswr'):
                                self.min_max_vswr_var.set(self.saved_min_max_vswr)

            # Create the "Group Delay Setting" Checkbutton
            self.cb_groupdelay_sff = tk.Checkbutton(settings_window, text="Group Delay & SFF", variable=self.cb_groupdelay_sff_var)
            self.cb_groupdelay_sff.grid(row=1, column=0, sticky=tk.W)  # Show checkbox
            
            # Create the Min/Max VSWR Checkbutton
            self.cb_min_max_vswr = tk.Checkbutton(settings_window, text="Tabled Min/Max VSWR", variable=self.min_max_vswr_var)
            self.cb_min_max_vswr.grid(row=1, column=2, sticky=tk.W)  # Adjust the row/column as necessary
            # If there’s a saved value, use it to initialize the checkbox
            if hasattr(self, 'saved_min_max_vswr'):
                self.min_max_vswr_var.set(self.saved_min_max_vswr)

            # Limit 1
            tk.Label(settings_window, text="Limit 1 Frequency Start (GHz):").grid(row=2, column=0)
            self.limit1_freq1 = tk.DoubleVar()
            tk.Entry(settings_window, textvariable=self.limit1_freq1).grid(row=2, column=1)
            
            tk.Label(settings_window, text="Limit 1 Value Start:").grid(row=2, column=2)
            self.limit1_val1 = tk.DoubleVar()
            tk.Entry(settings_window, textvariable=self.limit1_val1).grid(row=2, column=3)
            
            tk.Label(settings_window, text="Limit 1 Frequency End (GHz):").grid(row=3, column=0)
            self.limit1_freq2 = tk.DoubleVar()
            tk.Entry(settings_window, textvariable=self.limit1_freq2).grid(row=3, column=1)
            
            tk.Label(settings_window, text="Limit 1 Value End:").grid(row=3, column=2)
            self.limit1_val2 = tk.DoubleVar()
            tk.Entry(settings_window, textvariable=self.limit1_val2).grid(row=3, column=3)
            
            # Limit 2
            tk.Label(settings_window, text="Limit 2 Frequency Start (GHz):").grid(row=4, column=0)
            self.limit2_freq1 = tk.DoubleVar()
            tk.Entry(settings_window, textvariable=self.limit2_freq1).grid(row=4, column=1)
            
            tk.Label(settings_window, text="Limit 2 Value Start:").grid(row=4, column=2)
            self.limit2_val1 = tk.DoubleVar()
            tk.Entry(settings_window, textvariable=self.limit2_val1).grid(row=4, column=3)
            
            tk.Label(settings_window, text="Limit 2 Frequency End (GHz):").grid(row=5, column=0)
            self.limit2_freq2 = tk.DoubleVar()
            tk.Entry(settings_window, textvariable=self.limit2_freq2).grid(row=5, column=1)
            
            tk.Label(settings_window, text="Limit 2 Value End:").grid(row=5, column=2)
            self.limit2_val2 = tk.DoubleVar()
            tk.Entry(settings_window, textvariable=self.limit2_val2).grid(row=5, column=3)
        
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

            # Create the Save Settings & Default Settings button within VSWR Settings
            save_button = tk.Button(settings_window, text="Save Settings", command=save_vswr_settings, bg=ACCENT_BLUE_COLOR, fg=LIGHT_TEXT_COLOR)
            save_button.grid(row=6, column=0, columnspan=2, pady=20)
            default_button = tk.Button(settings_window, text="Default Settings", command=default_vswr_settings, bg=ACCENT_BLUE_COLOR, fg=LIGHT_TEXT_COLOR)
            default_button.grid(row=6, column=2, columnspan=2, pady=20)

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

    # Method to reset the necessary variables and GUI elements upon file import
    def reset_data(self):
        plt.close('all')
        self.data = None
        self.hpol_file_path = None
        self.vpol_file_path = None
        self.freq_list = None
        self.TRP_file_path = None
        
    def calculate_min_max_eff_gain(self, file_paths, bands):
        """
        For each band, calculate the min, max, and average of Gain (dBi) and Efficiency (both as fraction and dB).
        - Gain in the G&D file is already in dBi.
        - Efficiency is in percentage; convert from % to fraction and dB:
        Efficiency_dB = 10 * log10(Efficiency_%/100).
        """
        results = []
        all_data = []
        for fpath in file_paths:
            gd_data = process_gd_file(fpath)
            freq = np.array(gd_data['Frequency'])
            
            gain_dBi = np.array(gd_data['Gain'])  # Gain already in dBi
            eff_percent = np.array(gd_data['Efficiency'])
            eff_fraction = eff_percent / 100.0
            eff_fraction[eff_fraction <= 0] = 1e-12  # Avoid log issues
            eff_dB = 10 * np.log10(eff_fraction)

            avg_eff_fraction = np.mean(eff_fraction)
            avg_eff_dB = 10 * np.log10(avg_eff_fraction)

            all_data.append((fpath, freq, gain_dBi, eff_dB, avg_eff_fraction, avg_eff_dB))
        
        for i, (freq_min, freq_max) in enumerate(bands, start=1):
            band_label = f"({freq_min}-{freq_max} MHz)"
            band_results = []
            for fpath, freq, gain_dBi, eff_dB, avg_eff_fraction, avg_eff_dB in all_data:
                within_range = (freq >= freq_min) & (freq <= freq_max)
                min_gain = gain_dBi[within_range].min() if np.any(within_range) else None
                max_gain = gain_dBi[within_range].max() if np.any(within_range) else None
                min_eff = eff_dB[within_range].min() if np.any(within_range) else None
                max_eff = eff_dB[within_range].max() if np.any(within_range) else None
                band_results.append((os.path.basename(fpath), min_gain, max_gain, min_eff, max_eff, avg_eff_fraction, avg_eff_dB))
            results.append((band_label, band_results))
        
        return results

    def display_eff_gain_table(self, results):
        """
        Display the calculated min, max, and average efficiency (percentage and dB) along with gain (dBi).
        """
        result_window = tk.Toplevel()
        result_window.title("Min/Max Gain & Efficiency Results by Band")
        
        row = 0
        for band_label, band_data in results:
            tk.Label(result_window, text=band_label, font=("Arial", 12, "bold")).grid(row=row, column=0, columnspan=7, pady=5)
            row += 1

            headers = ["File", "Min Gain(dB)", "Max Gain(dB)", "Min Eff(dB)", "Max Eff(dB)", "Avg Eff(%)", "Avg Eff(dB)"]
            for col, header in enumerate(headers):
                tk.Label(result_window, text=header, font=("Arial", 10, "bold")).grid(row=row, column=col, padx=10, pady=5)
            row += 1

            for file, min_gain, max_gain, min_eff, max_eff, avg_eff_fraction, avg_eff_dB in band_data:
                avg_eff_percent = avg_eff_fraction * 100
                tk.Label(result_window, text=file).grid(row=row, column=0, padx=10, pady=5)
                tk.Label(result_window, text=f"{min_gain:.2f}" if min_gain is not None else "N/A").grid(row=row, column=1, padx=10, pady=5)
                tk.Label(result_window, text=f"{max_gain:.2f}" if max_gain is not None else "N/A").grid(row=row, column=2, padx=10, pady=5)
                tk.Label(result_window, text=f"{min_eff:.2f}" if min_eff is not None else "N/A").grid(row=row, column=3, padx=10, pady=5)
                tk.Label(result_window, text=f"{max_eff:.2f}" if max_eff is not None else "N/A").grid(row=row, column=4, padx=10, pady=5)
                tk.Label(result_window, text=f"{avg_eff_percent:.2f}").grid(row=row, column=5, padx=10, pady=5)
                tk.Label(result_window, text=f"{avg_eff_dB:.2f}").grid(row=row, column=6, padx=10, pady=5)
                row += 1

            row += 1

    # Method to import TRP or HPOL/VPOL data files for analysis       
    def import_files(self):
        # Reset variables to clean any previous file imports
        self.reset_data()

        if self.scan_type.get() == "active":
            self.TRP_file_path = filedialog.askopenfilename(title="Select the TRP Data File",
                                                            filetypes=[("Text files", "*.txt"), ("All files", "*.*")])
        elif self.scan_type.get() == "passive" and self.passive_scan_type.get() == "G&D":
            if self.min_max_eff_gain_var.get():
                # Prompt the user for scenario/type of measurement when min_max_eff_gain is enabled
                scenario_window = tk.Toplevel(self.root)
                scenario_window.title("Select Measurement Scenario")
                scenario_window.geometry("400x300")
                scenario_window.grab_set()  # Modal

                scenario_var = tk.StringVar(value="")

                # TODO Example scenarios
                #lora_863 = tk.Radiobutton(scenario_window, text="Single-band LoRa 863-870 MHz", variable=scenario_var, value="LoRa_863")
                #lora_863.grid(row=0, column=0, sticky='w', padx=10, pady=5)

                #lora_902 = tk.Radiobutton(scenario_window, text="Single-band LoRa 902-928 MHz", variable=scenario_var, value="LoRa_902")
                #lora_902.grid(row=1, column=0, sticky='w', padx=10, pady=5)

                #lora_dual = tk.Radiobutton(scenario_window, text="Dual-band LoRa 863-928 MHz", variable=scenario_var, value="LoRa_863_928")
                #lora_dual.grid(row=2, column=0, sticky='w', padx=10, pady=5)

                wifi_6e = tk.Radiobutton(scenario_window, text="WiFi 6e (2.4, 5, 6 GHz)", variable=scenario_var, value="WiFi_6e")
                wifi_6e.grid(row=3, column=0, sticky='w', padx=10, pady=5)

                tk.Label(scenario_window, text="Number of files per band:").grid(row=4, column=0, sticky='w', padx=10, pady=5)
                files_per_band_var = tk.IntVar(value=4)  # default to 4
                tk.Entry(scenario_window, textvariable=files_per_band_var, width=5).grid(row=4, column=1, padx=5, pady=5)

                def on_scenario_ok():
                    chosen = scenario_var.get()
                    if chosen == "":
                        messagebox.showerror("Error", "Please select a measurement scenario.")
                        return

                    self.files_per_band = files_per_band_var.get()

                    # Define selected_bands based on scenario
                    if chosen == "LoRa_863":
                        self.selected_bands = [(863.0, 870.0)]
                    elif chosen == "LoRa_902":
                        self.selected_bands = [(902.0, 928.0)]
                    elif chosen == "LoRa_863_928":
                        self.selected_bands = [(863.0, 928.0)]
                    elif chosen == "WiFi_6e":
                        # WiFi6e triple-band
                        self.selected_bands = [
                            (2400.0, 2500.0),
                            (4900.0, 5925.0),
                            (5925.0, 7125.0)
                        ]
                    else:
                        # If no predefined scenario selected, self.selected_bands = []
                        # We'll handle in fallback logic
                        self.selected_bands = []

                    self.measurement_scenario = chosen
                    scenario_window.destroy()

                ok_button = tk.Button(scenario_window, text="OK", command=on_scenario_ok, bg=ACCENT_BLUE_COLOR, fg=LIGHT_TEXT_COLOR)
                ok_button.grid(row=5, column=0, pady=20, padx=10)

                self.root.wait_window(scenario_window)

                # Now handle chosen scenario
                if self.measurement_scenario in ["LoRa_863", "LoRa_902", "LoRa_863_928", "WiFi_6e"] and self.selected_bands:
                    # We have predefined bands and files_per_band from scenario.
                    all_band_results = []
                    for i, (f_min, f_max) in enumerate(self.selected_bands, start=1):
                        band_label = f"({f_min}-{f_max} MHz)"
                        messagebox.showinfo("Band Selection", f"Please select {self.files_per_band} G&D files for {band_label}")
                        file_paths = []
                        for _ in range(self.files_per_band):
                            filepath = filedialog.askopenfilename(filetypes=[("Text files", "*.txt")], title=f"Select File for {band_label}")
                            if not filepath:
                                return
                            file_paths.append(filepath)

                        band_results = self.calculate_min_max_eff_gain(file_paths, [(f_min, f_max)])
                        self.display_eff_gain_table(band_results)
                        all_band_results.extend(band_results)

                    self.display_final_summary(all_band_results)

                else:
                    # Fallback: original logic if no predefined scenario or user wants to define bands manually
                    num_files = tk.simpledialog.askinteger("Input", "How many G&D files would you like to import?", parent=self.root)
                    if not num_files:
                        return
                    file_paths = []
                    for _ in range(num_files):
                        filepath = filedialog.askopenfilename(filetypes=[("Text files", "*.txt")])
                        if not filepath:
                            return
                        file_paths.append(filepath)

                    num_bands = tk.simpledialog.askinteger("Input", "How many frequency bands would you like to define?")
                    if not num_bands:
                        return

                    bands = []
                    for i in range(num_bands):
                        freq_range_input = tk.simpledialog.askstring("Input", f"Enter frequency range for Band {i+1} as 'min,max' (MHz):", parent=self.root)
                        try:
                            freq_min, freq_max = map(float, freq_range_input.split(','))
                            bands.append((freq_min, freq_max))
                        except ValueError:
                            messagebox.showerror("Error", "Invalid frequency range. Please enter values in 'min,max' format.")
                            return

                    results = self.calculate_min_max_eff_gain(file_paths, bands)
                    self.display_eff_gain_table(results)
                    self.display_final_summary(results)

            else:
                # Original non-min_max_eff_gain logic for G&D
                num_files = tk.simpledialog.askinteger("Input", "How many files would you like to import?", parent=self.root)
                if not num_files:
                    return
                datasets = []
                file_names = []
                for _ in range(num_files):
                    filepath = filedialog.askopenfilename(filetypes=[("Text files", "*.txt")])
                    if not filepath:
                        return
                    file_name = os.path.basename(filepath).replace('.txt', '')
                    file_names.append(file_name)
                    data = process_gd_file(filepath)
                    datasets.append(data)

                x_range_input = tk.simpledialog.askstring("Input", "Enter x-axis range as 'min,max' (or leave blank for auto-scale):", parent=self.root)
                plot_gd_data(datasets, file_names, x_range_input)

        elif self.scan_type.get() == "passive" and self.passive_scan_type.get() == "VPOL/HPOL":
            first_file = filedialog.askopenfilename(title="Select the First Data File",
                                                filetypes=[("Text files", "*.txt"), ("All files", "*.*")])
            if not first_file:
                return
            
            second_file = filedialog.askopenfilename(title="Select the Second Data File",
                                                    filetypes=[("Text files", "*.txt"), ("All files", "*.*")])
            if first_file and second_file:
                first_polarization = determine_polarization(first_file)
                second_polarization = determine_polarization(second_file)
                
                if first_polarization == second_polarization:
                    messagebox.showerror("Error", "Both files cannot be of the same polarization.")
                    return

                if first_polarization == "HPol":
                    self.hpol_file_path = first_file
                    self.vpol_file_path = second_file
                else:
                    self.hpol_file_path = second_file
                    self.vpol_file_path = first_file

                match, message = check_matching_files(self.hpol_file_path, self.vpol_file_path)
                if not match:
                    messagebox.showerror("Error", message)
                    return
                self.update_passive_frequency_list()

        elif self.scan_type.get() == "vswr":
            # Check for conflicting options
            if self.cb_groupdelay_sff_var.get() and self.min_max_vswr_var.get():
                messagebox.showerror("Error", "Cannot have both Group Delay/SFF and Min/Max VSWR selected.")
                return

            # Scenario 1: Group Delay = False, Min/Max VSWR = False
            # Just plot VSWR/Return Loss files
            if not self.cb_groupdelay_sff_var.get() and not self.min_max_vswr_var.get():
                num_files = tk.simpledialog.askinteger("Input", "How many files do you want to import?")
                if not num_files:
                    return

                file_paths = []
                for _ in range(num_files):
                    file_path = filedialog.askopenfilename(title="Select the VSWR/Return Loss File(s)",
                                                        filetypes=[("CSV files", "*.csv"), ("All files", "*.*")])
                    if not file_path:
                        return
                    file_paths.append(file_path)

                process_vswr_files(file_paths,
                                self.saved_limit1_freq1, self.saved_limit1_freq2, self.saved_limit1_start, self.saved_limit1_stop,
                                self.saved_limit2_freq1, self.saved_limit2_freq2, self.saved_limit2_start, self.saved_limit2_stop)

            # Scenario 2: Group Delay = True, Min/Max VSWR = False
            # Process group delay and SFF from 2-port data
            elif self.cb_groupdelay_sff_var.get() and not self.min_max_vswr_var.get():
                num_files = tk.simpledialog.askinteger("Input", "How many files do you want to import? (e.g. 8 for Theta=0° to 315° in 45° steps)")
                if not num_files:
                    return

                file_paths = []
                for _ in range(num_files):
                    file_path = filedialog.askopenfilename(title="Select the 2-Port S-Parameter File(s)",
                                                        filetypes=[("CSV files", "*.csv"), ("All files", "*.*")])
                    if not file_path:
                        return
                    file_paths.append(file_path)

                root = tk.Tk()
                root.withdraw()

                min_freq = askstring("Input", "Enter minimum frequency (GHz) or leave blank for default:")
                max_freq = askstring("Input", "Enter maximum frequency (GHz) or leave blank for default:")

                min_freq = float(min_freq) if min_freq else None
                max_freq = float(max_freq) if max_freq else None

                process_groupdelay_files(file_paths,
                                        self.saved_limit1_freq1, self.saved_limit1_freq2, self.saved_limit1_start, self.saved_limit1_stop,
                                        self.saved_limit2_freq1, self.saved_limit2_freq2, self.saved_limit2_start, self.saved_limit2_stop,
                                        min_freq, max_freq)

            # Scenario 3: Group Delay = False, Min/Max VSWR = True
            # Calculate and display min/max VSWR over defined bands
            elif not self.cb_groupdelay_sff_var.get() and self.min_max_vswr_var.get():
                num_files = tk.simpledialog.askinteger("Input", "How many VSWR files do you want to import?")
                if not num_files:
                    return

                num_bands = tk.simpledialog.askinteger("Input", "How many frequency bands would you like to define?")
                if not num_bands:
                    return

                bands = []
                for i in range(num_bands):
                    freq_range_input = tk.simpledialog.askstring("Input", f"Enter frequency range for Band {i+1} as 'min,max' (in MHz):", parent=self.root)
                    try:
                        freq_min, freq_max = map(float, freq_range_input.split(','))
                        bands.append((freq_min, freq_max))
                    except ValueError:
                        messagebox.showerror("Error", "Invalid frequency range. Please enter values in 'min,max' format.")
                        return

                file_paths = []
                for _ in range(num_files):
                    file_path = filedialog.askopenfilename(title="Select the VSWR 1-Port File(s)",
                                                        filetypes=[("CSV files", "*.csv"), ("All files", "*.*")])
                    if not file_path:
                        return
                    file_paths.append(file_path)

                # Using the previously discussed generic parameter min/max calculation
                # Assuming you've implemented calculate_min_max_parameter and display_parameter_table
                results = calculate_min_max_parameters(file_paths, bands, "VSWR")
                display_parameter_table(results, "VSWR", parent=self.root)

    def log_message(self, message):
        self.log_text.configure(state='normal')
        self.log_text.insert('end', message + '\n')
        self.log_text.configure(state='disabled')
        self.log_text.see('end')

    def process_data(self):
        try:
            if self.scan_type.get() == "active":
                # Call read_active_file and retrieve data variables 
                # Assuming `file_content` is a string containing the content of the selected file
                data = read_active_file(self.TRP_file_path)
                
                # Unpacking the data for further use
                self.log_message("Processing active data...")
                (frequency,start_phi,start_theta,stop_phi,stop_theta,inc_phi,inc_theta,
                calc_trp,theta_angles_deg,phi_angles_deg,h_power_dBm,v_power_dBm
                ) = (data["Frequency"],data["Start Phi"],data["Start Theta"],data["Stop Phi"],
                data["Stop Theta"],data["Inc Phi"],data["Inc Theta"], data["Calculated TRP(dBm)"],
                data["Theta_Angles_Deg"],data["Phi_Angles_Deg"],data["H_Power_dBm"],data["V_Power_dBm"])

                # Calculate Variables for TRP/Active Measurement Plotting        
                active_variables = calculate_active_variables(start_phi, stop_phi, start_theta, stop_theta, inc_phi, inc_theta, h_power_dBm, v_power_dBm)
                
                #Define Active Variables
                (data_points, theta_angles_deg, phi_angles_deg, theta_angles_rad, phi_angles_rad,
                total_power_dBm_2d, h_power_dBm_2d, v_power_dBm_2d,
                phi_angles_deg_plot, phi_angles_rad_plot,
                total_power_dBm_2d_plot, h_power_dBm_2d_plot, v_power_dBm_2d_plot,
                total_power_dBm_min, total_power_dBm_nom,
                h_power_dBm_min, h_power_dBm_nom,
                v_power_dBm_min, v_power_dBm_nom,
                TRP_dBm, h_TRP_dBm, v_TRP_dBm) = active_variables



                # Plot Azimuth cuts for different theta values on one plot from theta_values_to_plot = [0, 15, 30, 45, 60, 75, 90, 105, 120, 135, 150, 165] like below
                # Plot Elevation and Azimuth cuts for the 3-planes Theta=90deg, Phi=0deg/180deg, and Phi=90deg/270deg
                plot_active_2d_data(data_points, theta_angles_rad, phi_angles_rad_plot, total_power_dBm_2d_plot, frequency)
                # Plot Elevation and Azimuth cuts for the 3-planes Theta=90deg, Phi=0deg/180deg, and Phi=90deg/270deg
                #plot_active_2d_data(data_points, theta_angles_rad, phi_angles_rad, h_power_dBm_2d, frequency)
                # Plot Elevation and Azimuth cuts for the 3-planes Theta=90deg, Phi=0deg/180deg, and Phi=90deg/270deg
                #plot_active_2d_data(data_points, theta_angles_rad, phi_angles_rad, v_power_dBm_2d, frequency)

                # 3D TRP Surface Plots similar to the passive 3D data, but instead of gain TRP for Phi, Theta pol and Total Radiated Power(TRP)
                # For total power
                plot_active_3d_data(
                    theta_angles_deg, phi_angles_deg, total_power_dBm_2d,
                    phi_angles_deg_plot, total_power_dBm_2d_plot,
                    frequency, power_type='total', interpolate=self.interpolate_3d_plots
                )

                # For horizontal polarization (hpol)
                plot_active_3d_data(
                    theta_angles_deg, phi_angles_deg, h_power_dBm_2d,
                    phi_angles_deg_plot, h_power_dBm_2d_plot,
                    frequency, power_type='hpol', interpolate=self.interpolate_3d_plots
                )

                # For vertical polarization (vpol)
                plot_active_3d_data(
                    theta_angles_deg, phi_angles_deg, v_power_dBm_2d,
                    phi_angles_deg_plot, v_power_dBm_2d_plot,
                    frequency, power_type='vpol', interpolate=self.interpolate_3d_plots
                )
                self.log_message("Active data processed successfully.")
                return
            
            elif self.scan_type.get() == "passive":
                self.log_message("Processing passive data...")

                #After reading & parsing, hpol_data and vpol_data will be lists of dictionaries. 
                #Each dictionary will represent a frequency point and will contain:
                    #'frequency': The frequency value
                    #'cal_factor': Calibration factor for that frequency
                    #'data': A list of tuples, where each tuple contains (theta, phi, mag, phase).
                parsed_hpol_data, start_phi_h, stop_phi_h, inc_phi_h, start_theta_h, stop_theta_h, inc_theta_h = read_passive_file(self.hpol_file_path)
                hpol_data = parsed_hpol_data

                parsed_vpol_data, start_phi_v, stop_phi_v, inc_phi_v, start_theta_v, stop_theta_v, inc_theta_v = read_passive_file(self.vpol_file_path)
                vpol_data = parsed_vpol_data
                
                # Check if angle data matches
                if not angles_match(start_phi_h, stop_phi_h, inc_phi_h, start_theta_h, stop_theta_h, inc_theta_h,
                                start_phi_v, stop_phi_v, inc_phi_v, start_theta_v, stop_theta_v, inc_theta_v):
                    messagebox.showerror("Error", "Angle data mismatch between HPol and VPol files.")
                    self.log_message("Error: Angle data mismatch between HPol and VPol files.")
                    return
                
                #Call Methods to Calculate Required Variables and Set up variables for plotting 
                passive_variables = calculate_passive_variables(hpol_data, vpol_data, float(self.cable_loss.get()), start_phi_h, stop_phi_h, inc_phi_h, start_theta_h, stop_theta_h, inc_theta_h, self.freq_list, float(self.selected_frequency.get()))
                theta_angles_deg, phi_angles_deg, v_gain_dB, h_gain_dB, Total_Gain_dB = passive_variables
                
                # If frequency is below 500MHz, convert NF measurements to FF
                if float(self.selected_frequency.get()) < 500:
                    # Measurement distance is fixed at 1 meter
                    measurement_distance = 1.0  # meters
                    # Window function is fixed to 'none'
                    window_function = 'none'

                    self.log_message("Applying NF2FF Transformation...")
                    hpol_far_field, vpol_far_field = apply_nf2ff_transformation(
                        hpol_data, vpol_data, float(self.selected_frequency.get()),
                        start_phi_h, stop_phi_h, inc_phi_h,
                        start_theta_h, stop_theta_h, inc_theta_h,
                        measurement_distance, window_function
                    )
                else: # if frequency is above 500MHz, plot passive data normally
                    hpol_far_field = h_gain_dB
                    vpol_far_field = v_gain_dB
                self.log_message("Passive data processed successfully.")
                plot_2d_passive_data(theta_angles_deg, phi_angles_deg, hpol_far_field, vpol_far_field, Total_Gain_dB, self.freq_list, float(self.selected_frequency.get()), self.datasheet_plots_var.get())
                plot_passive_3d_component(theta_angles_deg, phi_angles_deg, hpol_far_field, vpol_far_field, Total_Gain_dB, self.freq_list, float(self.selected_frequency.get()), gain_type="total")
                plot_passive_3d_component(theta_angles_deg, phi_angles_deg, hpol_far_field, vpol_far_field, Total_Gain_dB, self.freq_list, float(self.selected_frequency.get()), gain_type="hpol") 
                plot_passive_3d_component(theta_angles_deg, phi_angles_deg, hpol_far_field, vpol_far_field, Total_Gain_dB, self.freq_list, float(self.selected_frequency.get()), gain_type="vpol")
                
                return
        except Exception as e:
            self.log_message(f"Error: {e}")    
    def display_final_summary(self, results):
        """
        Create a final summary table with Min, Max, and Average Efficiency (as % and dB) for each band.
        """
        # Calculate overall averages for each band
        band_summary = {}
        for band_label, band_data in results:
            eff_fractions = [x[5] for x in band_data if x[5] is not None]
            eff_dBs = [x[6] for x in band_data if x[6] is not None]
            gain_mins = [x[1] for x in band_data if x[1] is not None]
            gain_maxs = [x[2] for x in band_data if x[2] is not None]

            # Calculate global min/max/average values for the band
            avg_eff_fraction = np.mean(eff_fractions) if eff_fractions else None
            avg_eff_percent = avg_eff_fraction * 100 if avg_eff_fraction else None
            avg_eff_dB = np.mean(eff_dBs) if eff_dBs else None
            band_min_gain = min(gain_mins) if gain_mins else None
            band_max_gain = max(gain_maxs) if gain_maxs else None
            band_summary[band_label] = (avg_eff_percent, avg_eff_dB, band_min_gain, band_max_gain)

        # Create summary window
        summary_window = tk.Toplevel(self.root)
        summary_window.title("Final Summary")

        # Header
        tk.Label(summary_window, text="Final Summary of All Bands", font=("Arial", 12, "bold")).grid(row=0, column=0, columnspan=len(results) + 1, pady=10)

        # Band labels as headers
        tk.Label(summary_window, text="Parameter", font=("Arial", 10, "bold")).grid(row=1, column=0, padx=10, pady=5)
        for col, (band_label, _) in enumerate(results, start=1):
            tk.Label(summary_window, text=band_label, font=("Arial", 10, "bold")).grid(row=1, column=col, padx=10, pady=5)

        # Rows for parameters
        parameters = ["Avg Eff(%)", "Avg Eff(dB)", "Min Gain(dBi)", "Max Gain(dBi)"]
        for row, param in enumerate(parameters, start=2):
            tk.Label(summary_window, text=param, font=("Arial", 10, "bold")).grid(row=row, column=0, sticky='w', padx=10, pady=5)

        # Fill in values for each band
        for col, (band_label, _) in enumerate(results, start=1):
            avg_eff_percent, avg_eff_dB, min_gain, max_gain = band_summary[band_label]
            values = [avg_eff_percent, avg_eff_dB, min_gain, max_gain]
            for row, value in enumerate(values, start=2):
                tk.Label(summary_window, text=f"{value:.2f}" if value is not None else "N/A").grid(row=row, column=col, padx=10, pady=5)