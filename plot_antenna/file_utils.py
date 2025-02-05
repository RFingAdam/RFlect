from calculations import angles_match, calculate_passive_variables

import os
import numpy as np
import pandas as pd
import datetime

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
        inc_phi = float(content[36].split(":")[1].split("Deg")[0].strip())
        
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

        data_start_line = None
        for i, line in enumerate(lines):
            if "Freq" in line and "Gain" in line and "Directivity" in line and "Efficiency" in line:
                # The next non-blank line after this should be data_start_line
                data_start_line = i + 1
                # Now find the first truly numeric line
                while data_start_line < len(lines) and (lines[data_start_line].strip() == "" or not lines[data_start_line].strip()[0].isdigit()):
                    data_start_line += 1
                break

        if data_start_line is None:
            raise Exception("Could not find the data header line in the provided file.")

        # Extract data
        for line in lines[data_start_line:]:
            line = line.strip()
            if not line:  # Blank line means end of data
                break
            parts = line.split()
            # Check if we have four columns and the first is numeric
            if len(parts) < 4 or not parts[0].replace('.', '', 1).isdigit():
                # Not a valid data line
                continue  # or break, depending on file structure
            
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
    available_columns = [col for col in [
        '! Stimulus(Hz)', 'S11(SWR)', 'S22(SWR)', 'S11(dB)', 'S22(dB)', 
        'S21(dB)', 'S12(dB)', 'S21(s)', 'S12(s)'
    ] if col in data.columns]
    
    # If not all columns are available, handle it gracefully
    if len(available_columns) < 5:
        print(f"Warning: Not all expected columns are available in {file_path}. Available columns: {', '.join(available_columns)}.")
    
    # Use only the available columns
    organized_data = data[available_columns]
    
    return organized_data

# Function to parse group delay measurements from Agilent VNA
def parse_agilent_data(file_path):
    # Read the data, assuming the first row is a header row
    # and the file is space or tab-delimited
    data = pd.read_csv(file_path, skiprows=2)

    # Rename columns for consistency with other data
    # Assuming the first column is frequency and the second is group delay
    data = data.rename(columns={
        data.columns[0]: '! Stimulus(Hz)',
        data.columns[1]: 'S21(s)'
    })

    # Check which columns are available in the data
    available_columns = [col for col in ['! Stimulus(Hz)', 'S21(s)', 'S12(s)'] if col in data.columns]
    # Use only the available columns
    organized_data = data[available_columns]

    return organized_data

# File Utility to Conver Vpol and Hpol Gain Measurement to FFS file for CST Import/Visualization
def convert_HpolVpol_files(vswr_file_path, hpol_file_path, vpol_file_path, cable_loss, freq_list_MHz, selected_freq, callback=None):
    # Convert selected frequency from MHz to GHz
    selected_freq_ghz = selected_freq / 1000.0

    parsed_hpol_data, start_phi_h, stop_phi_h, inc_phi_h, start_theta_h, stop_theta_h, inc_theta_h = read_passive_file(hpol_file_path)
    hpol_data = parsed_hpol_data

    parsed_vpol_data, start_phi_v, stop_phi_v, inc_phi_v, start_theta_v, stop_theta_v, inc_theta_v = read_passive_file(vpol_file_path)
    vpol_data = parsed_vpol_data
    
    # Check to see if selected files have mismatched frequency or angle data
    angles_match(start_phi_h, stop_phi_h, inc_phi_h, start_theta_h, stop_theta_h, inc_theta_h,
                    start_phi_v, stop_phi_v, inc_phi_v, start_theta_v, stop_theta_v, inc_theta_v)

    #Call Methods to Calculate Required Variables and Set up variables for plotting 
    passive_variables = calculate_passive_variables(hpol_data, vpol_data, cable_loss, start_phi_h, stop_phi_h, inc_phi_h, start_theta_h, stop_theta_h, inc_theta_h, freq_list_MHz, selected_freq)
    theta_angles_deg, phi_angles_deg, v_gain_dB, h_gain_dB, Total_Gain_dB = passive_variables
    
    # Read VSWR Data
    vswr_data = pd.read_csv(vswr_file_path)
    freqs_ghz = vswr_data.iloc[:, 0] / 1e9
    values = vswr_data.iloc[:, 1]

    # Check if selected_freq exists in freqs_ghz using precision checking
    if np.any(np.isclose(freqs_ghz, selected_freq_ghz, atol=1e-6)):
        selected_freq_idx_vswr = np.where(np.isclose(freqs_ghz, selected_freq_ghz, atol=1e-6))[0][0]

        vswr_at_selected_freq = values[selected_freq_idx_vswr]
    else:
        print(f"Error: Selected frequency {selected_freq_ghz} not found in the VSWR frequency list.")
        return

    # Calculate the reflection coefficient
    reflection_coefficient = (vswr_at_selected_freq - 1) / (vswr_at_selected_freq + 1)

    # Calculate the transmission coefficient
    transmission_coefficient = 1 - reflection_coefficient**2

    # Calculate the accepted power
    stimulated_power = 1e-3  # Default 0.5 W Or 0.001W=0dBm?
    accepted_power = transmission_coefficient * stimulated_power

    # Ensure selected frequency exists in the provided frequency list
    if np.any(np.isclose(freq_list_MHz, selected_freq, atol=1e-6)):
        freq_idx_other = np.where(np.isclose(freq_list_MHz, selected_freq, atol=1e-6))[0][0]
    else:
        print(f"Error: Selected frequency {selected_freq_ghz} not found in the frequency list.")
        return
    
    selected_theta_angles_deg = theta_angles_deg[:, freq_idx_other]
    selected_phi_angles_deg = phi_angles_deg[:, freq_idx_other]
    selected_h_gain = h_gain_dB[:, freq_idx_other]
    selected_v_gain = v_gain_dB[:, freq_idx_other]
  
    # Find the dictionary corresponding to the selected_freq for hpol and vpol
    selected_hpol_data = next((data for data in parsed_hpol_data if np.isclose(data["frequency"], selected_freq, atol=1e-6)), None)  # Change 4: Consistent frequency check
    selected_vpol_data = next((data for data in parsed_vpol_data if np.isclose(data["frequency"], selected_freq, atol=1e-6)), None)  # Change 4: Consistent frequency check

    # Check if we found the data for the selected frequency
    if not selected_hpol_data or not selected_vpol_data:
        print(f"Error: Data for the selected frequency {selected_freq} not found.")  
        return
    
    # Extract the phase information
    selected_h_phase = selected_hpol_data['phase']
    selected_v_phase = selected_vpol_data['phase']

  # Ensure data integrity by checking if all arrays have the same length
    if not (len(selected_theta_angles_deg) == len(selected_phi_angles_deg) == len(selected_h_gain) == len(selected_v_gain)):
        print("Error: Data arrays do not have the same length.")
        return

    # Combine the selected data into a single DataFrame
    combined_data = pd.DataFrame({
        'Theta': selected_theta_angles_deg,
        'Phi': selected_phi_angles_deg,
        'H_Gain': selected_h_gain,
        'V_Gain': selected_v_gain,
        'H_Phase': selected_h_phase,
        'V_Phase': selected_v_phase
    })

    # Add Phi=360 data
    phi_360_data = combined_data[combined_data['Phi'] == 0].copy()
    phi_360_data['Phi'] = 360
    combined_data = pd.concat([combined_data, phi_360_data], ignore_index=True)

    # Add Theta=180 data
    theta_180_data = combined_data.drop_duplicates(subset=['Phi']).copy()
    theta_180_data['Theta'] = 180
    theta_180_data['H_Gain'] = -30
    theta_180_data['V_Gain'] = -30
    theta_180_data['H_Phase'] = 0
    theta_180_data['V_Phase'] = 0
    combined_data = pd.concat([combined_data, theta_180_data], ignore_index=True)

    # Calculate Total Phi and Theta Samples
    total_phi_samples = int((stop_phi_h - start_phi_h) / inc_phi_h) + 1 + 1 # add 1 for Phi=360
    total_theta_samples = int((stop_theta_h - start_theta_h) / inc_theta_h) + 1 + 1 # add 1 for Theta=180

    # Convert (Theta, Phi, H_Gain, V_gain to // >> Phi, Theta, Re(E_Theta), Im(E_Theta), Re(E_Phi), Im(E_Phi):)
    # Convert gain from dB to linear scale and then to complex representation
    # Gain in linear scale = 10 ^ (Gain_dB / 10)
    G_linear_theta = 10 ** (combined_data['V_Gain'] / 10)
    G_linear_phi = 10 ** (combined_data['H_Gain'] / 10)
    
    # Calculate the sum of the new V-POL and H-POL magnitudes, multiplied by the sine of the Theta angle (converted to radians)
    theta_rad = np.deg2rad(combined_data['Theta'])
    sum_values = (G_linear_theta + G_linear_phi) * np.sin(theta_rad)

    # Calculate the radiated power using the formula
    M = total_phi_samples - 0
    N = total_theta_samples - 0
    radiated_power_mw = (np.pi / 2) * sum_values.sum() / (N * M)
    radiated_power_w = radiated_power_mw / 1000  # Convert to Watts

    # Calculate the electric field strength
    E_Theta_strength = np.sqrt(377 * G_linear_theta / 4 / np.pi)  # using V_Gain for E_Theta (V/m)
    E_Phi_strength = np.sqrt(377 * G_linear_phi / 4 / np.pi)   # using H_Gain for E_Phi (V/m)

    # Convert phase from degrees to radians for the calculation
    combined_data['V_Phase'] = np.radians(combined_data['V_Phase'])
    combined_data['H_Phase'] = np.radians(combined_data['H_Phase'])
    
    # Calculate the electric field components in complex form
    E_Theta_complex = E_Theta_strength * (np.cos(combined_data['V_Phase']) + 1j * np.sin(combined_data['V_Phase']))
    E_Phi_complex = E_Phi_strength * (np.cos(combined_data['H_Phase']) + 1j * np.sin(combined_data['H_Phase']))

    # Separate real and imaginary parts
    combined_data['Re(E_Theta)'] = np.real(E_Theta_complex)
    combined_data['Im(E_Theta)'] = np.imag(E_Theta_complex)
    combined_data['Re(E_Phi)'] = np.real(E_Phi_complex)
    combined_data['Im(E_Phi)'] = np.imag(E_Phi_complex)

    # Sort the DataFrame by 'Theta' and 'Phi'
    combined_data.sort_values(by=['Phi', 'Theta'], inplace=True)

    # Round values to 2 decimal places
    combined_data = combined_data.round({'Phi': 2, 'Theta': 2, 'Re(E_Theta)': 2, 'Im(E_Theta)': 2, 'Re(E_Phi)': 2, 'Im(E_Phi)': 2})

    # Format the DataFrame for output
    df_final = combined_data[['Phi', 'Theta', 'Re(E_Theta)', 'Im(E_Theta)', 'Re(E_Phi)', 'Im(E_Phi)']].copy()

    # Prepare the data in the CST .ffs format
    # Construct the header
    
    # Ensure the selected_freq is treated as a float for formatting
    frequency_hz = float(selected_freq) * 1e6

    header = f"""// CST Farfield Source File
    
// Version:
3.0

// Data Type
Farfield

// #Frequencies
1

//Position
0.0 0.0 0.0

//z-Axis
0.0 0.0 1.0

//x-Axis
1.0 0.0 0.0

// Radiated Power [W,rms]
{radiated_power_w}

// Accepted Power [W,rms]
{accepted_power}

// Stimulated Power [W,rms]
{stimulated_power}

// Frequency[Hz]
{frequency_hz}

// >> Total #phi samples, total #theta samples
{total_phi_samples}   {total_theta_samples}

// >> Phi, Theta, Re(E_Theta), Im(E_Theta), Re(E_Phi), Im(E_Phi): 
"""

    # Construct the data lines
    # You may need to adjust the following line depending on the exact structure of 'combined_data' and the .ffs file format requirements
    data_lines = "\n".join(df_final.astype(str).apply("\t".join, axis=1))

    # Save to a file
    # Extract the base name without extension
    base_name_hpol = os.path.splitext(os.path.basename(hpol_file_path))[0]

    # Construct the new file name
    new_file_name = f"{base_name_hpol.replace('AP_HPol', '')}_{selected_freq}MHz.ffs"

    # Join with the directory path
    output_path = os.path.join(os.path.dirname(hpol_file_path), new_file_name)
    with open(output_path, 'w') as f:
        f.write(header)
        f.writelines(data_lines)

    if callback:
        callback()

def read_power_measurement(file_path):
    """
    Reads power measurement data from a file.

    Args:
        file_path (str): The path to the power measurement file.

    Returns:
        dict: A dictionary with frequencies as keys and power measurements as values.
    """
    data = {}
    with open(file_path, 'r') as file:
        frequency = None
        read_values = False  # Initialize read_values at the start
        for line in file:
            if "Test Frequency" in line:
                frequency = float(line.split('=')[1].strip().replace('MHz', ''))
                read_values = False  # Reset the flag
                # print(f"Found frequency: {frequency} MHz")
            elif "H-Pol" in line or "V-Pol" in line and frequency is not None:
                read_values = True  # Next lines will contain the measurements
            elif read_values and frequency is not None and line.strip():
                try:
                    parts = line.split()
                    if len(parts) == 3 and parts[-1].replace('-', '').replace('.', '').isdigit():
                        power = float(parts[-1].strip())  # Assume the last part is the power value
                        data[frequency] = power
                        # print(f"Added power measurement: {power} dBm at {frequency} MHz")
                        read_values = False  # Reset the flag after reading the power value
                        frequency = None  # Reset frequency to None after reading the power value
                except ValueError as e:
                    print(f"Error converting power value to float on line: {line.strip()}. Error: {e}")
    return data

def read_gain_standard(file_path):
    """
    Reads gain standard data from a file.

    Args:
        file_path (str): The path to the gain standard file.

    Returns:
        dict: A dictionary with frequencies as keys and gain values as values.
    """
    data = {}
    with open(file_path, 'r') as file:
        for line in file:
            if line.strip() and line[0].isdigit():  # Check if the line starts with a digit
                parts = line.split()
                if len(parts) >= 2:
                    frequency = float(parts[0].strip())
                    gain = float(parts[1].strip())
                    data[frequency] = gain
                    # print(f"Frequency: {frequency} MHz, Gain: {gain} dBi")
    return data

def read_polarization_data(file_path):
    """
    Reads polarization data from a file.

    Args:
        file_path (str): The path to the polarization data file.

    Returns:
        dict: A dictionary with frequencies as keys and polarization values as values.
    """
    data = {}
    with open(file_path, 'r') as file:
        frequency = None
        read_values = False
        for line in file:
            if "Test Frequency" in line:
                frequency = float(line.split('=')[1].strip().replace('MHz', ''))
                read_values = False
            elif "H-Pol" in line or "V-Pol" in line and frequency is not None:
                read_values = True
            elif read_values and frequency is not None and line.strip():
                try:
                    parts = line.split()
                    if len(parts) == 3 and parts[-1].replace('-', '').replace('.', '').isdigit():
                        value = float(parts[-1].strip())
                        data[frequency] = value
                        # print(f"Added polarization data: {value} dBm at {frequency} MHz")
                        read_values = False
                        frequency = None
                except ValueError as e:
                    print(f"Error converting polarization value to float on line: {line.strip()}. Error: {e}")
    return data

def generate_active_cal_file(power_measurement_file, gain_standard_file, hpol_file, vpol_file, cable_loss, freq_list, callback=None):
    """
    Generates an active calibration file and a summary file.

    Args:
        power_measurement_file (str): Path to the power measurement file.
        gain_standard_file (str): Path to the gain standard file.
        hpol_file (str): Path to the horizontal polarization data file.
        vpol_file (str): Path to the vertical polarization data file.
        cable_loss (float): Cable loss value -  Not used, but could be.
        freq_list (list): List of frequencies to include in the calibration file.
        callback (function, optional): Callback function to update visibility in the GUI.

    Returns:
        None
    """
    power_measurement_data = read_power_measurement(power_measurement_file)
    gain_standard_data = read_gain_standard(gain_standard_file)
    hpol_data = read_polarization_data(hpol_file)
    vpol_data = read_polarization_data(vpol_file)

    # Create a DataFrame for easy computation
    df = pd.DataFrame(freq_list, columns=["Frequency"])
    df["Power"] = df["Frequency"].map(power_measurement_data)
    df["Gain"] = df["Frequency"].map(gain_standard_data)
    df["H-Pol"] = df["Frequency"].map(hpol_data)
    df["V-Pol"] = df["Frequency"].map(vpol_data)

    # Compute P+G-H and P+G-V
    df["P+G-H"] = df["Power"] + df["Gain"] - df["H-Pol"]
    df["P+G-V"] = df["Power"] + df["Gain"] - df["V-Pol"]
 
    # Filter out rows with NaN values
    df_filtered = df.dropna()

    # Determine the frequency range from the data
    start_frequency = df["Frequency"].min()
    stop_frequency = df["Frequency"].max()

    # Determine today's date and time
    now = datetime.datetime.now()
    today = now.strftime("%m-%d-%y")
    current_time = now.strftime("%H:%M:%S")

    # Extracting the file names
    hpol_basename = os.path.basename(hpol_file)
    vpol_basename = os.path.basename(vpol_file)
    common_part = os.path.commonprefix([hpol_basename, vpol_basename]).split('.')[0].replace("AP_", "")

    # Create the output file name based on the common part of the HPOL and VPOL files and the date
    output_file = f"TRP Cal {common_part}1Amp 0 dBm {start_frequency}-{stop_frequency} {today}.txt"
    output_path = os.path.join(os.path.dirname(hpol_file), output_file)

    # Write the result to a file
    with open(output_path, 'w') as file:
        file.write(f"WTL Calibration File\n\nDate: {today}\nTime: {current_time}\n\n*******  Cal Data  ********\n")
        file.write("Freq\tH-Pol\tV-Pol\n(MHz)\t(dBm)\t(dBm)\n")
        for _, row in df_filtered.iterrows():
            file.write(f"{int(row['Frequency'])}\t{row['P+G-H']:.2f}\t{row['P+G-V']:.2f}\n")

    # Generate a summary file
    summary_file = f"TRP Cal Summary {common_part}1Amp 0 dBm {start_frequency}-{stop_frequency} {today}.txt"
    summary_path = os.path.join(os.path.dirname(hpol_file), summary_file)

    with open(summary_path, 'w') as file:
        file.write(f"Calibration Summary\n\nDate: {today}\nTime: {current_time}\n\n")
        file.write(f"Power Measurement File: {power_measurement_file}\n")
        file.write(f"Gain Standard File: {gain_standard_file}\n")
        file.write(f"HPol File: {hpol_file}\n")
        file.write(f"VPol File: {vpol_file}\n")
        file.write(f"Output File: {output_path}\n")
        file.write(f"Frequency Range: {start_frequency} MHz - {stop_frequency} MHz\n\n")
        file.write("Missing Data Frequencies:\n")
        missing_data = df[df.isna().any(axis=1)]
        for freq in missing_data["Frequency"]:
            file.write(f"{freq} MHz\n")

    # Callback to update visibility in the GUI
    if callback:
        callback()