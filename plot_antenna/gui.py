from file_utils import read_active_file, read_passive_file, check_matching_files, process_gd_file, convert_HpolVpol_files, generate_active_cal_file
from save import save_to_results_folder
from calculations import determine_polarization, angles_match, extract_passive_frequencies, calculate_passive_variables, calculate_active_variables, apply_nf2ff_transformation
from plotting import plot_2d_passive_data, plot_passive_3d_component, plot_gd_data, process_vswr_files, plot_active_2d_data, plot_active_3d_data
from config import *
from groupdelay import process_groupdelay_files

import os
import tkinter as tk
from tkinter import filedialog, messagebox, ttk, Menu
from tkinter.simpledialog import askstring
import tkinter.scrolledtext as ScrolledText
import requests
import json
import sys
import webbrowser
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

# Custom class to write to both console and Text widget
class DualOutput:
    def __init__(self, widget, stream):
        self.widget = widget
        self.stream = stream

    def write(self, string):
        self.widget.configure(state='normal')
        self.widget.insert('end', string)
        self.widget.configure(state='disabled')
        self.widget.see('end')
        self.stream.write(string)
        self.stream.flush()

    def flush(self):
        pass

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
        
        # TODO Button for Save Results to word document Routine
        """self.btn_save_to_word = tk.Button(self.root, text="Save Results to Word", command=lambda: save_to_results_folder(float(self.selected_frequency.get()), self.freq_list, self.scan_type.get(), self.hpol_file_path, self.vpol_file_path, self.TRP_file_path, float(self.cable_loss.get()), self.datasheet_plots_var.get(), word=True), bg=ACCENT_BLUE_COLOR, fg=LIGHT_TEXT_COLOR)
        self.btn_save_to_word.grid(row=5, column=2, pady=10, padx=10)"""
        
        # Update background color for the entire window
        self.root.config(bg=DARK_BG_COLOR)
    
        # TODO Bind hover effect to the buttons
        """buttons = [self.btn_import, self.btn_view_results, self.btn_save_to_file, self.btn_save_to_word, self.btn_settings]
        for btn in buttons:
            btn.bind("<Enter>", self.on_enter)
            btn.bind("<Leave>", self.on_leave)
"""
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
    def save_results_to_file(self):
        try:
            scan_type = self.scan_type.get()
            
            if scan_type == "active":
                # For active measurements, no need to get the frequency from user input
                frequency = None  # Frequency will come from the active file
            else:
                # For passive measurements, ensure a frequency is selected
                frequency = self.selected_frequency.get()
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
                float(self.cable_loss.get()),
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
            # TODO self.btn_save_to_word.grid_remove()
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
            # TODO self.btn_save_to_word.grid_remove
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
                # TODO self.btn_save_to_word.grid_forget()
            else:
                self.label_frequency.grid(row=3, column=0, pady=5)
                self.frequency_dropdown.grid(row=3, column=1, pady=5)
                # Show the save results button for non-G&D passive scans
                self.btn_save_to_file.grid(row=5, column=1, pady=10)  # Adjusting the row index
                # TODO self.btn_save_to_word.grid(row=5, column=2, pady=10)  # Adjusting the row index
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
            # TODO self.btn_save_to_word.grid_forget()
            
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
            
            # Create the "Datasheet Plots" Checkbutton
            self.cb_datasheet_plots = tk.Checkbutton(settings_window, text="Datasheet Plots", variable=self.datasheet_plots_var)
            
            if self.passive_scan_type.get() == "VPOL/HPOL":
                self.cb_datasheet_plots.grid(row=2, column=0, sticky=tk.W, padx=20)  # Show checkbox
            else:
                self.cb_datasheet_plots.grid_remove()  # Hide checkbox
            
            def save_passive_settings():
                # Save the chosen plot type
                self.passive_scan_type.set(self.plot_type_var.get())
                # Update the visibility of the main GUI
                self.update_visibility()
                # Close the settings window after saving
                settings_window.destroy()


            save_button = tk.Button(settings_window, text="Save Settings", command=save_passive_settings, bg=ACCENT_BLUE_COLOR, fg=LIGHT_TEXT_COLOR)
            save_button.grid(row=3, column=0, columnspan=2, pady=20)
            
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

            # Create the "Group Delay Setting" Checkbutton
            self.cb_groupdelay_sff = tk.Checkbutton(settings_window, text="Group Delay & SFF", variable=self.cb_groupdelay_sff_var)
            self.cb_groupdelay_sff.grid(row=1, column=0, sticky=tk.W)  # Show checkbox

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
        

    # Method to import TRP or HPOL/VPOL data files for analysis       
    def import_files(self):
        # Reset variables to clean any previous file imports
        self.reset_data()

        if self.scan_type.get() == "active":
            self.TRP_file_path = filedialog.askopenfilename(title="Select the TRP Data File",
                                                            filetypes=[("Text files", "*.txt"), ("All files", "*.*")])
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
                data = process_gd_file(filepath)
                datasets.append(data)

            # Prompt user for x-axis range
            x_range_input = tk.simpledialog.askstring("Input", "Enter x-axis range as 'min,max' (or leave blank for auto-scale):", parent=self.root)
            plot_gd_data(datasets, file_names, x_range_input)

        elif self.scan_type.get() == "passive" and self.passive_scan_type.get() == "VPOL/HPOL":
            
            first_file = filedialog.askopenfilename(title="Select the First Data File",
                                                filetypes=[("Text files", "*.txt"), ("All files", "*.*")])
            # Check if user canceled the first selection
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
                match, message = check_matching_files(self.hpol_file_path, self.vpol_file_path)
                if not match:
                    messagebox.showerror("Error", message)
                    return
                self.update_passive_frequency_list()

        elif self.scan_type.get() == "vswr" and self.cb_groupdelay_sff_var.get() == False:
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

            process_vswr_files(file_paths, self.saved_limit1_freq1, self.saved_limit1_freq2, self.saved_limit1_start, self.saved_limit1_stop, self.saved_limit2_freq1, self.saved_limit2_freq2, self.saved_limit2_start, self.saved_limit2_stop)    
        
        elif self.scan_type.get() == "vswr" and self.cb_groupdelay_sff_var.get() == True:
            # Group Delay and SFF Routine
            num_files = tk.simpledialog.askinteger("Input", "How many files do you want to import? (8 typ. for Theta=0deg to 315deg in 45deg steps)")
            if not num_files:
                return

            file_paths = []
            for _ in range(num_files):
                file_path = filedialog.askopenfilename(title="Select the 2-Port S-Parameter File(s) of format S11(dB), S22(dB), S21(dB), and S21(s)",
                                                    filetypes=[("CSV files", "*.csv"), ("All files", "*.*")])
                if not file_path:  # if user cancels the dialog
                    return
                file_paths.append(file_path)

            # Prompt User for X-axis/Frequency Plot Limits
            root = tk.Tk()
            root.withdraw()  # Don't need a full GUI, so keep the root window from appearing
            
            min_freq = askstring("Input", "Enter minimum frequency (GHz) or leave blank for default:")
            max_freq = askstring("Input", "Enter maximum frequency (GHz) or leave blank for default:")
            
            # Convert to float or set to None if blank
            min_freq = float(min_freq) if min_freq else None
            max_freq = float(max_freq) if max_freq else None
            
            process_groupdelay_files(file_paths, self.saved_limit1_freq1, self.saved_limit1_freq2, self.saved_limit1_start, self.saved_limit1_stop, self.saved_limit2_freq1, self.saved_limit2_freq2, self.saved_limit2_start, self.saved_limit2_stop, min_freq, max_freq)    
    
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
                (data_points, theta_angles_deg, phi_angles_deg, theta_angles_rad, phi_angles_rad, total_power_dBm_2d,
                total_power_dBm_min, total_power_dBm_nom, h_power_dBm_2d, h_power_dBm_min, v_power_dBm_2d,
                v_power_dBm_min, h_power_dBm_nom, v_power_dBm_nom, TRP_dBm, h_TRP_dBm, v_TRP_dBm) = active_variables

                # Plot Azimuth cuts for different theta values on one plot from theta_values_to_plot = [0, 15, 30, 45, 60, 75, 90, 105, 120, 135, 150, 165] like below
                # Plot Elevation and Azimuth cuts for the 3-planes Theta=90deg, Phi=0deg/180deg, and Phi=90deg/270deg
                plot_active_2d_data(data_points, theta_angles_rad, phi_angles_rad, total_power_dBm_2d, frequency)
                # Plot Elevation and Azimuth cuts for the 3-planes Theta=90deg, Phi=0deg/180deg, and Phi=90deg/270deg
                #plot_active_2d_data(data_points, theta_angles_rad, phi_angles_rad, h_power_dBm_2d, frequency)
                # Plot Elevation and Azimuth cuts for the 3-planes Theta=90deg, Phi=0deg/180deg, and Phi=90deg/270deg
                #plot_active_2d_data(data_points, theta_angles_rad, phi_angles_rad, v_power_dBm_2d, frequency)

                # 3D TRP Surface Plots similar to the passive 3D data, but instead of gain TRP for Phi, Theta pol and Total Radiated Power(TRP)
                plot_active_3d_data(theta_angles_deg, phi_angles_deg, total_power_dBm_2d, frequency, power_type='total', interpolate=self.interpolate_3d_plots)
                plot_active_3d_data(theta_angles_deg, phi_angles_deg, h_power_dBm_2d, frequency, power_type='hpol', interpolate=self.interpolate_3d_plots)
                plot_active_3d_data(theta_angles_deg, phi_angles_deg, v_power_dBm_2d, frequency, power_type='vpol', interpolate=self.interpolate_3d_plots)

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
    