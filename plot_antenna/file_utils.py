from config import ACCENT_GREEN_COLOR, ACCENT_BLUE_COLOR

import os

#Read in TRP/Active Scan File
def read_active_file(file_path):
    with open(file_path, 'r') as file:
        content = file.readlines()
    return parse_active_file(content)

# Read in Passive HPOL/VPOL Files
def read_passive_file(file_path):
    with open(file_path, 'r') as file:
        content = file.readlines()
    return parse_passive_file(content)

# Function to parse the data from active TRP files
def parse_active_file(content):
    #tbd
    return

# Function to parse the data from HPOL/VPOL files
def parse_passive_file(content):
    """
    Parse passive files and return a consolidated dataframe.

    This function reads and processes passive scan files (typically in .csv format). 
    It consolidates the data from these files into a single dataframe.

    Parameters:
    - files (list): List of file paths to be parsed.

    Returns:
    - DataFrame: A pandas DataFrame containing the consolidated data from the files.
    """
    
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
    
def save_to_results_folder(filename, content, widget):
    """
    Save the given data to the specified file within the results folder.

    Parameters:
    - filename (str): The name of the file to save to.
    - data (Any): The data to be saved.

    Returns:
    - str: The path to the saved file.
    """

    # Check if a results folder exists or not
    if not os.path.exists("results"):
        os.makedirs("results")

    # Save the content to the file in the results folder
    with open(os.path.join("results", filename), "w") as f:
        f.write(content)

    # Provide feedback through the widget (in this case, a button)
    widget.config(text="Save Results to File (Saved!)", bg=ACCENT_GREEN_COLOR)
    widget.after(2000, lambda: widget.config(text="Save Results to File", bg=ACCENT_BLUE_COLOR))
