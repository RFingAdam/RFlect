import os
import numpy as np
import pandas as pd

# Read in TRP/Active Scan File 
def read_active_file(file_path):
    with open(file_path, 'r') as file:
        content = file.readlines()
    return parse_active_file(content)

# Read in Passive HPOL/VPOL Files
def read_passive_file(file_path):
    with open(file_path, 'r') as file:
        content = file.readlines()
    return parse_passive_file(content)

# Function to parse the data from active TRP files Extracted from Howland Antenna Chamber and WTL Program
def parse_active_file(content):
    """
    Parse the TRP data file. Calculates some intermediate data to return less parameters to functions downstream

    Parameters:
    - content (str): Content of the data file.

    Returns:
    - dict: A dictionary containing parsed data and metadata.
    """
    # Validating file type
    if "Total Radiated Power Test" not in content[5]:
        raise ValueError("The selected file does not contain TRP data.")
    
    # Validate test type
    test_type = content[15].split(":")[1].strip()
    if test_type not in ["Discrete Test", "Semi-Discrete Angle Based Test"]:
        raise ValueError("Currently only Discrete and Semi-Discrete Angle TRP data is supported. Please select a DISCRETE or SEMI-DISCRETE ANGLE TRP file.")
    
    # Extracting metadata and initializing data arrays
    f = float(content[13].split(":")[1].split("MHz")[0].strip())

    # Extract the file format
    file_format = content[0].split()[-1]
    
    # Depending on the file format, the information is located on different lines
    # TODO Consider a more dynamic approach vs relying on hard coded number for extraction
    if file_format == "V5.02":
        start_phi = float(content[34].split(":")[1].split("Deg")[0].strip())
        stop_phi = float(content[35].split(":")[1].split("Deg")[0].strip())
        inc_phi = float(content[34].split(":")[1].split("Deg")[0].strip())
        
        start_theta = float(content[39].split(":")[1].split("Deg")[0].strip())
        stop_theta = float(content[40].split(":")[1].split("Deg")[0].strip())
        inc_theta = float(content[41].split(":")[1].split("Deg")[0].strip())
        
        calc_TRP = float(content[50].split("=")[1].split(" ")[0].strip())
        data_start_line = 55

        v_cal_fact = float(content[48].split('=')[1].strip().split(' ')[0])
        h_cal_fact = float(content[47].split('=')[1].strip().split(' ')[0])

    else: # if file_format == "V5.03" or newer
        start_phi = float(content[31].split(":")[1].split("Deg")[0].strip())
        stop_phi = float(content[32].split(":")[1].split("Deg")[0].strip())
        inc_phi = float(content[33].split(":")[1].split("Deg")[0].strip())
        
        start_theta = float(content[38].split(":")[1].split("Deg")[0].strip())
        stop_theta = float(content[39].split(":")[1].split("Deg")[0].strip())
        inc_theta = float(content[40].split(":")[1].split("Deg")[0].strip())
        
        calc_TRP = float(content[49].split('=')[1].strip().split(' ')[0])
        data_start_line = 54

        v_cal_fact = float(content[47].split('=')[1].strip().split(' ')[0])
        h_cal_fact = float(content[46].split('=')[1].strip().split(' ')[0])

        
    # Extracting data points
    theta_angles_deg = []
    phi_angles_deg = []
    h_unc_power_dBm = []
    v_unc_power_dBm = []

    for line in content[data_start_line:]:
        # Assuming the data is space-separated; adjust as necessary
        try:  # Add a try/except block to handle potential conversion errors
            theta, phi, h_unc_power, v_unc_power = map(float, line.split())
            theta_angles_deg.append(theta)
            phi_angles_deg.append(phi)
            h_unc_power_dBm.append(h_unc_power)
            v_unc_power_dBm.append(v_unc_power)
        except ValueError as e:
            raise ValueError(f"Error parsing data line: {line}. {str(e)}")
    
    
    # Calculate Intermediary Data to Return less variables downstream
    h_power_dBm = np.array(h_unc_power_dBm) + h_cal_fact
    v_power_dBm = np.array(v_unc_power_dBm) + v_cal_fact
    
    data = {
        "Frequency": f,
        "Start Phi": start_phi,
        "Start Theta": start_theta,
        "Stop Phi": stop_phi,
        "Stop Theta": stop_theta,
        "Inc Phi": inc_phi,
        "Inc Theta": inc_theta,
        "Calculated TRP(dBm)":calc_TRP,
        "Theta_Angles_Deg": np.array(theta_angles_deg),
        "Phi_Angles_Deg": np.array(phi_angles_deg),
        "H_Power_dBm": np.array(h_power_dBm),
        "V_Power_dBm": np.array(v_power_dBm),
    }
    
    return data

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
    
 # Checks for matching data between two passive scan files HPOL and VPOL to ensure they are from the same dataset
def check_matching_files(file1, file2):
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

def process_gd_file(filepath):
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

# Parse Group Delay .csv file consisting of S11(dB), S22(dB), S21(dB) or S12(dB), and S21(s)or S12(s) data
def parse_2port_data(file_path):
    # Load data considering the third row as header
    data = pd.read_csv(file_path, skiprows=2)

    # Remove leading/trailing whitespace from column names
    data.columns = [col.strip() for col in data.columns]

    # Check which columns are available in the data
    available_columns = [col for col in ['! Stimulus(Hz)', 'S11(dB)', 'S22(dB)', 'S21(dB)', 'S12(dB)', 'S21(s)', 'S12(s)'] if col in data.columns]
    
    # If not all columns are available, handle it gracefully
    if len(available_columns) < 5:
        print(f"Warning: Not all expected columns are available in {file_path}. Available columns: {', '.join(available_columns)}.")
    
    # Use only the available columns
    organized_data = data[available_columns]
    
    return organized_data