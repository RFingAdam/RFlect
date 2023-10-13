import os

# TODO Read in TRP/Active Scan File 
def read_active_file(file_path):
    with open(file_path, 'r') as file:
        content = file.readlines()
    return parse_active_file(content)

# Read in Passive HPOL/VPOL Files
def read_passive_file(file_path):
    with open(file_path, 'r') as file:
        content = file.readlines()
    return parse_passive_file(content)

# TODO Function to parse the data from active TRP files
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