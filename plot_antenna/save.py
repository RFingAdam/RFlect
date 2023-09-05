from calculations import angles_match, calculate_passive_variables
from plotting import plot_passive_3d_component, plot_2d_passive_data
from file_utils import read_active_file, read_passive_file

from tkinter import simpledialog, filedialog, Tk
import os

def save_to_results_folder(selected_frequency, freq_list, scan_type, hpol_path, vpol_path, active_path, cable_loss):
    user_selected_frequency = selected_frequency
    # Initialize the GUI
    root = Tk()
    root.withdraw()  # Hide the main window
       
    # Call the modified plotting functions with save_path argument to save the plots
    if scan_type == "active":
            # TODO Perform active calculations and plotting method calls
            #future implementaiton
            return
    elif scan_type == "passive":
        #After reading & parsing, hpol_data and vpol_data will be lists of dictionaries. 
        #Each dictionary will represent a frequency point and will contain:
            #'frequency': The frequency value
            #'cal_factor': Calibration factor for that frequency
            #'data': A list of tuples, where each tuple contains (theta, phi, mag, phase).
            # Prompt the user for the project name
        project_name = simpledialog.askstring("Input", "Enter Project Name:")
        
        # Prompt the user to select a directory
        directory = filedialog.askdirectory(title="Select Directory to Save Project")
        
        # Check if user provided a project name and directory
        if not project_name or not directory:
            print("Project name or directory not provided. Exiting...")
            return
        
        # Create the directory structure
        project_path = os.path.join(directory, project_name)
        two_d_data_path = os.path.join(project_path, "2D Plots")
              
        user_selected_frequency_folder_name = f"3D Plots at {selected_frequency} MHz"
        user_selected_frequency_path = os.path.join(project_path, user_selected_frequency_folder_name)

        # Create directories if they don't exist
        os.makedirs(two_d_data_path, exist_ok=True)
        os.makedirs(user_selected_frequency_path, exist_ok=True)
        
        parsed_hpol_data, start_phi_h, stop_phi_h, inc_phi_h, start_theta_h, stop_theta_h, inc_theta_h = read_passive_file(hpol_path)
        hpol_data = parsed_hpol_data

        parsed_vpol_data, start_phi_v, stop_phi_v, inc_phi_v, start_theta_v, stop_theta_v, inc_theta_v = read_passive_file(vpol_path)
        vpol_data = parsed_vpol_data
        
        #check to see if selected files have mismatched frequency or angle data
        angles_match(start_phi_h, stop_phi_h, inc_phi_h, start_theta_h, stop_theta_h, inc_theta_h,
                        start_phi_v, stop_phi_v, inc_phi_v, start_theta_v, stop_theta_v, inc_theta_v)

        #Call Methods to Calculate Required Variables and Set up variables for plotting 
        passive_variables = calculate_passive_variables(hpol_data, vpol_data, cable_loss, start_phi_h, stop_phi_h, inc_phi_h, start_theta_h, stop_theta_h, inc_theta_h, freq_list, selected_frequency)
        theta_angles_deg, phi_angles_deg, v_gain_dB, h_gain_dB, Total_Gain_dB = passive_variables
        
        #Call Method to Plot Passive Data
        plot_2d_passive_data(theta_angles_deg, phi_angles_deg, v_gain_dB, h_gain_dB, Total_Gain_dB, freq_list, selected_frequency, save_path=two_d_data_path)
        
        plot_passive_3d_component(theta_angles_deg, phi_angles_deg, v_gain_dB, h_gain_dB, Total_Gain_dB, freq_list, float(selected_frequency), gain_type="total", save_path=user_selected_frequency_path)
        plot_passive_3d_component(theta_angles_deg, phi_angles_deg, v_gain_dB, h_gain_dB, Total_Gain_dB, freq_list, float(selected_frequency), gain_type="hpol", save_path=user_selected_frequency_path)
        plot_passive_3d_component(theta_angles_deg, phi_angles_deg, v_gain_dB, h_gain_dB, Total_Gain_dB, freq_list, float(selected_frequency), gain_type="vpol", save_path=user_selected_frequency_path)
    
    print(f"Data saved to {project_path}")