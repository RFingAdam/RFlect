from .calculations import angles_match, calculate_passive_variables

import os
import numpy as np
import pandas as pd
import datetime


# Read in TRP/Active Scan File
def read_active_file(file_path):
    with open(file_path, "r", encoding="utf-8") as file:
        content = file.readlines()
    return parse_active_file(content)


# Read in Passive HPOL/VPOL Files
def read_passive_file(file_path):
    with open(file_path, "r", encoding="utf-8") as file:
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
        raise ValueError(
            "Currently only Discrete and Semi-Discrete Angle TRP data is supported. Please select a DISCRETE or SEMI-DISCRETE ANGLE TRP file."
        )

    # Extracting metadata and initializing data arrays
    f = float(content[13].split(":")[1].split("MHz")[0].strip())

    # Extract the file format
    file_format = content[0].split()[-1]

    # Depending on the file format, the information is located on different lines
    # TODO Consider a more dynamic approach vs relying on hard coded number for extraction
    try:
        if file_format == "V5.02":
            start_phi = float(content[34].split(":")[1].split("Deg")[0].strip())
            stop_phi = float(content[35].split(":")[1].split("Deg")[0].strip())
            inc_phi = float(content[36].split(":")[1].split("Deg")[0].strip())

            start_theta = float(content[39].split(":")[1].split("Deg")[0].strip())
            stop_theta = float(content[40].split(":")[1].split("Deg")[0].strip())
            inc_theta = float(content[41].split(":")[1].split("Deg")[0].strip())

            calc_TRP = float(content[50].split("=")[1].split(" ")[0].strip())
            data_start_line = 55

            v_cal_fact = float(content[48].split("=")[1].strip().split(" ")[0])
            h_cal_fact = float(content[47].split("=")[1].strip().split(" ")[0])

        else:  # if file_format == "V5.03" or newer
            start_phi = float(content[31].split(":")[1].split("Deg")[0].strip())
            stop_phi = float(content[32].split(":")[1].split("Deg")[0].strip())
            inc_phi = float(content[33].split(":")[1].split("Deg")[0].strip())

            start_theta = float(content[38].split(":")[1].split("Deg")[0].strip())
            stop_theta = float(content[39].split(":")[1].split("Deg")[0].strip())
            inc_theta = float(content[40].split(":")[1].split("Deg")[0].strip())

            calc_TRP = float(content[49].split("=")[1].strip().split(" ")[0])
            data_start_line = 54

            v_cal_fact = float(content[47].split("=")[1].strip().split(" ")[0])
            h_cal_fact = float(content[46].split("=")[1].strip().split(" ")[0])
    except (IndexError, AttributeError) as e:
        raise ValueError(f"Malformed TRP file header (format {file_format}): {e}")

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
        "Calculated TRP(dBm)": calc_TRP,
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
    start_phi: float | None = None
    stop_phi: float | None = None
    inc_phi: float | None = None
    start_theta: float | None = None
    stop_theta: float | None = None
    inc_theta: float | None = None

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

    # Validate that all required values were extracted
    if (
        start_phi is None
        or stop_phi is None
        or inc_phi is None
        or start_theta is None
        or stop_theta is None
        or inc_theta is None
    ):
        raise ValueError("Failed to extract all required angle parameters from passive file")

    # Calculate the expected number of data points
    theta_points = (stop_theta - start_theta) / inc_theta + 1
    phi_points = (stop_phi - start_phi) / inc_phi + 1
    data_points = int(theta_points * phi_points)

    # Extract data for each frequency
    while content:
        cal_factor_line = next(
            (line for line in content if "Cal Std Antenna Peak Gain Factor" in line), None
        )
        if not cal_factor_line:
            break
        cal_factor_value = float(cal_factor_line.split("=")[1].strip().split(" ")[0])
        freq_line = next((line for line in content if "Test Frequency = " in line), None)
        if not freq_line:
            break
        freq_value = float(freq_line.split("=")[1].strip().split(" ")[0])

        start_index = next(
            (i for i, line in enumerate(content) if "THETA\t  PHI\t  Mag\t Phase" in line), None
        )
        if start_index is None:
            break
        start_index += 2

        data_lines = content[start_index : start_index + data_points]
        theta, phi, mag, phase = [], [], [], []
        for line in data_lines:
            line_data = line.strip().split("\t")
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
            "phase": phase,
        }
        all_data.append(freq_data)
        # Ensure forward progress to prevent infinite loops
        min_advance = max(start_index + 1, 1) if start_index is not None else 1
        content = content[max(start_index + data_points, min_advance) :]

    return all_data, start_phi, stop_phi, inc_phi, start_theta, stop_theta, inc_theta


# Checks for matching data between two passive scan files HPOL and VPOL to ensure they are from the same dataset
def check_matching_files(file_path1, file_path2):
    """Verify that two passive measurement files have matching parameters."""

    def _extract_params(content_lines):
        freq = None
        angles = []
        for line in content_lines:
            if "Test Frequency" in line:
                try:
                    freq = float(line.split("=")[1].split()[0])
                except (IndexError, ValueError):
                    pass
            if any(kw in line for kw in ("Start", "Stop", "Inc")) and any(
                kw in line for kw in ("Phi", "Theta", "Axis")
            ):
                angles.append(line.strip())
        return freq, angles

    # Extract filename without extension for comparison
    filename1 = os.path.splitext(os.path.basename(file_path1))[0]
    filename2 = os.path.splitext(os.path.basename(file_path2))[0]

    # Check if filenames match excluding the last 4 characters (polarization part)
    if filename1[:-4] != filename2[:-4]:
        return False, "File names do not match."

    with open(file_path1, "r", encoding="utf-8") as f:
        content1 = f.readlines()
    with open(file_path2, "r", encoding="utf-8") as f:
        content2 = f.readlines()

    freq1, angles1 = _extract_params(content1)
    freq2, angles2 = _extract_params(content2)

    # Angle configurations must match for passive pair files
    if angles1 != angles2:
        return False, "The selected files have mismatched angle data."

    return True, ""


def validate_hpol_vpol_files(hpol_file, vpol_file):
    """
    Validates that HPOL and VPOL files are correctly paired and assigned.

    Checks:
    1. Files are from the same measurement set (matching filenames except polarization)
    2. Files have matching frequency and angle data
    3. Files are correctly identified as HPOL and VPOL (not swapped)

    Parameters:
        hpol_file: Path to file selected as HPOL
        vpol_file: Path to file selected as VPOL

    Returns:
        (is_valid, error_message) tuple
        - is_valid: True if validation passes, False otherwise
        - error_message: String describing the error, empty if valid
    """
    # Extract filenames
    hpol_filename = os.path.splitext(os.path.basename(hpol_file))[0].upper()
    vpol_filename = os.path.splitext(os.path.basename(vpol_file))[0].upper()

    # Check 1: Verify polarization identifiers in filenames
    hpol_has_hpol = "HPOL" in hpol_filename or "H_POL" in hpol_filename or "_H_" in hpol_filename
    vpol_has_vpol = "VPOL" in vpol_filename or "V_POL" in vpol_filename or "_V_" in vpol_filename

    hpol_has_vpol = "VPOL" in hpol_filename or "V_POL" in hpol_filename or "_V_" in hpol_filename
    vpol_has_hpol = "HPOL" in vpol_filename or "H_POL" in vpol_filename or "_H_" in vpol_filename

    # Check if files are swapped
    if hpol_has_vpol and vpol_has_hpol:
        return False, (
            "⚠️ FILES ARE SWAPPED!\n\n"
            f"The file you selected as HPOL appears to be VPOL: '{os.path.basename(hpol_file)}'\n"
            f"The file you selected as VPOL appears to be HPOL: '{os.path.basename(vpol_file)}'\n\n"
            "Please swap your file selections."
        )

    # Check if HPOL file is actually HPOL
    if not hpol_has_hpol and hpol_has_vpol:
        return False, (
            f"⚠️ INCORRECT FILE!\n\n"
            f"The file you selected as HPOL appears to be VPOL:\n'{os.path.basename(hpol_file)}'\n\n"
            "Please select the correct HPOL file."
        )

    # Check if VPOL file is actually VPOL
    if not vpol_has_vpol and vpol_has_hpol:
        return False, (
            f"⚠️ INCORRECT FILE!\n\n"
            f"The file you selected as VPOL appears to be HPOL:\n'{os.path.basename(vpol_file)}'\n\n"
            "Please select the correct VPOL file."
        )

    # Check 2: Verify files are from same measurement set
    # Extract base filename (before polarization identifier)
    hpol_base = (
        hpol_filename.replace("HPOL", "")
        .replace("H_POL", "")
        .replace("_H_", "_")
        .replace("_H", "")
        .replace("H_", "")
    )
    vpol_base = (
        vpol_filename.replace("VPOL", "")
        .replace("V_POL", "")
        .replace("_V_", "_")
        .replace("_V", "")
        .replace("V_", "")
    )

    # More lenient base name matching (after removing polarization indicators)
    # Compare last 10 chars or so to allow for different prefixes
    if len(hpol_base) >= 10 and len(vpol_base) >= 10:
        if hpol_base[-10:] != vpol_base[-10:]:
            # Try the original check_matching_files method
            base_match, _ = check_matching_files(hpol_file, vpol_file)
            if not base_match:
                return False, (
                    f"⚠️ FILE MISMATCH!\n\n"
                    f"Files may not be from the same measurement:\n"
                    f"HPOL: '{os.path.basename(hpol_file)}'\n"
                    f"VPOL: '{os.path.basename(vpol_file)}'\n\n"
                    "Please verify you selected matching files."
                )

    # Check 3: Use existing check_matching_files for frequency/angle validation
    files_match, match_error = check_matching_files(hpol_file, vpol_file)
    if not files_match:
        return (
            False,
            f"⚠️ DATA MISMATCH!\n\n{match_error}\n\nFiles must have matching frequency and angle parameters.",
        )

    return True, ""


def process_gd_file(filepath):
    """
    Reads the G&D file and extracts frequency, gain, directivity, and efficiency data.
    """
    frequency = []
    gain = []
    directivity = []
    efficiency = []

    with open(filepath, "r", encoding="utf-8") as file:
        lines = file.readlines()

    data_start_line = None
    for i, line in enumerate(lines):
        if "Freq" in line and "Gain" in line and "Directivity" in line and "Efficiency" in line:
            # The next non-blank line after this should be data_start_line
            data_start_line = i + 1
            # Now find the first truly numeric line
            while data_start_line < len(lines) and (
                lines[data_start_line].strip() == ""
                or not lines[data_start_line].strip()[0].isdigit()
            ):
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
        if len(parts) < 4 or not parts[0].replace(".", "", 1).isdigit():
            # Not a valid data line
            continue  # or break, depending on file structure

        frequency.append(float(parts[0]))
        gain.append(float(parts[1]))
        directivity.append(float(parts[2]))
        efficiency.append(float(parts[3]))

    return {
        "Frequency": frequency,
        "Gain": gain,
        "Directivity": directivity,
        "Efficiency": efficiency,
    }


# Parse Group Delay .csv file consisting of S11(dB), S22(dB), S21(dB) or S12(dB), and S21(s)or S12(s) data
def parse_2port_data(file_path):
    # Load data considering the third row as header
    data = pd.read_csv(file_path, skiprows=2)

    # Remove leading/trailing whitespace from column names
    data.columns = [col.strip() for col in data.columns]

    # Check which columns are available in the data
    available_columns = [
        col
        for col in [
            "! Stimulus(Hz)",
            "S11(SWR)",
            "S22(SWR)",
            "S11(dB)",
            "S22(dB)",
            "S21(dB)",
            "S12(dB)",
            "S21(s)",
            "S12(s)",
        ]
        if col in data.columns
    ]

    # If not all columns are available, handle it gracefully
    if len(available_columns) < 5:
        print(
            f"Warning: Not all expected columns are available in {file_path}. Available columns: {', '.join(available_columns)}."
        )

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
    data = data.rename(columns={data.columns[0]: "! Stimulus(Hz)", data.columns[1]: "S21(s)"})

    # Check which columns are available in the data
    available_columns = [
        col for col in ["! Stimulus(Hz)", "S21(s)", "S12(s)"] if col in data.columns
    ]
    # Use only the available columns
    organized_data = data[available_columns]

    return organized_data


def parse_touchstone_to_dataframe(filepath):
    """Parse a Touchstone .s2p file into a DataFrame compatible with the group delay pipeline.

    Converts complex S-parameters back to the column format used by parse_2port_data:
    '! Stimulus(Hz)', 'S11(dB)', 'S22(dB)', 'S21(dB)', 'S21(s)'.

    Group delay is computed from the S21 phase: tau = -dphi/(2*pi*df).

    Args:
        filepath: Path to the .s2p file.

    Returns:
        pd.DataFrame with columns matching the S2VNA CSV convention.
    """
    from .uwb_analysis import parse_touchstone, compute_group_delay_from_s21

    ts = parse_touchstone(filepath)
    freq = ts["freq_hz"]
    s11 = ts["s11"]
    s21 = ts["s21"]
    s22 = ts["s22"]

    s11_dB = 20.0 * np.log10(np.maximum(np.abs(s11), 1e-30))
    s22_dB = 20.0 * np.log10(np.maximum(np.abs(s22), 1e-30))
    s21_dB = 20.0 * np.log10(np.maximum(np.abs(s21), 1e-30))

    # Compute group delay from S21 phase
    gd_result = compute_group_delay_from_s21(freq, s21)
    group_delay = gd_result["group_delay_s"]

    df = pd.DataFrame(
        {
            "! Stimulus(Hz)": freq,
            "S11(dB)": s11_dB,
            "S22(dB)": s22_dB,
            "S21(dB)": s21_dB,
            "S21(s)": group_delay,
        }
    )

    return df


# File Utility to Conver Vpol and Hpol Gain Measurement to FFS file for CST Import/Visualization
def convert_HpolVpol_files(
    vswr_file_path,
    hpol_file_path,
    vpol_file_path,
    cable_loss,
    freq_list_MHz,
    selected_freq,
    callback=None,
):
    # Convert selected frequency from MHz to GHz
    selected_freq_ghz = selected_freq / 1000.0

    (
        parsed_hpol_data,
        start_phi_h,
        stop_phi_h,
        inc_phi_h,
        start_theta_h,
        stop_theta_h,
        inc_theta_h,
    ) = read_passive_file(hpol_file_path)
    hpol_data = parsed_hpol_data

    (
        parsed_vpol_data,
        start_phi_v,
        stop_phi_v,
        inc_phi_v,
        start_theta_v,
        stop_theta_v,
        inc_theta_v,
    ) = read_passive_file(vpol_file_path)
    vpol_data = parsed_vpol_data

    # Check to see if selected files have mismatched frequency or angle data
    angles_match(
        start_phi_h,
        stop_phi_h,
        inc_phi_h,
        start_theta_h,
        stop_theta_h,
        inc_theta_h,
        start_phi_v,
        stop_phi_v,
        inc_phi_v,
        start_theta_v,
        stop_theta_v,
        inc_theta_v,
    )

    # Call Methods to Calculate Required Variables and Set up variables for plotting
    passive_variables = calculate_passive_variables(
        hpol_data,
        vpol_data,
        cable_loss,
        start_phi_h,
        stop_phi_h,
        inc_phi_h,
        start_theta_h,
        stop_theta_h,
        inc_theta_h,
        freq_list_MHz,
        selected_freq,
    )
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
        print(
            f"Error: Selected frequency {selected_freq_ghz} not found in the VSWR frequency list."
        )
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
    selected_hpol_data = next(
        (
            data
            for data in parsed_hpol_data
            if np.isclose(data["frequency"], selected_freq, atol=1e-6)
        ),
        None,
    )  # Change 4: Consistent frequency check
    selected_vpol_data = next(
        (
            data
            for data in parsed_vpol_data
            if np.isclose(data["frequency"], selected_freq, atol=1e-6)
        ),
        None,
    )  # Change 4: Consistent frequency check

    # Check if we found the data for the selected frequency
    if not selected_hpol_data or not selected_vpol_data:
        print(f"Error: Data for the selected frequency {selected_freq} not found.")
        return

    # Extract the phase information
    selected_h_phase = selected_hpol_data["phase"]
    selected_v_phase = selected_vpol_data["phase"]

    # Ensure data integrity by checking if all arrays have the same length
    if not (
        len(selected_theta_angles_deg)
        == len(selected_phi_angles_deg)
        == len(selected_h_gain)
        == len(selected_v_gain)
    ):
        print("Error: Data arrays do not have the same length.")
        return

    # Combine the selected data into a single DataFrame
    combined_data = pd.DataFrame(
        {
            "Theta": selected_theta_angles_deg,
            "Phi": selected_phi_angles_deg,
            "H_Gain": selected_h_gain,
            "V_Gain": selected_v_gain,
            "H_Phase": selected_h_phase,
            "V_Phase": selected_v_phase,
        }
    )

    # Add Phi=360 data
    phi_360_data = combined_data[combined_data["Phi"] == 0].copy()
    phi_360_data["Phi"] = 360
    combined_data = pd.concat([combined_data, phi_360_data], ignore_index=True)

    # Add Theta=180 data
    theta_180_data = combined_data.drop_duplicates(subset=["Phi"]).copy()
    theta_180_data["Theta"] = 180
    theta_180_data["H_Gain"] = -30
    theta_180_data["V_Gain"] = -30
    theta_180_data["H_Phase"] = 0
    theta_180_data["V_Phase"] = 0
    combined_data = pd.concat([combined_data, theta_180_data], ignore_index=True)

    # Calculate Total Phi and Theta Samples
    total_phi_samples = int((stop_phi_h - start_phi_h) / inc_phi_h) + 1 + 1  # add 1 for Phi=360
    total_theta_samples = (
        int((stop_theta_h - start_theta_h) / inc_theta_h) + 1 + 1
    )  # add 1 for Theta=180

    # Convert (Theta, Phi, H_Gain, V_gain to // >> Phi, Theta, Re(E_Theta), Im(E_Theta), Re(E_Phi), Im(E_Phi):)
    # Convert gain from dB to linear scale and then to complex representation
    # Gain in linear scale = 10 ^ (Gain_dB / 10)
    G_linear_theta = 10 ** (combined_data["V_Gain"] / 10)
    G_linear_phi = 10 ** (combined_data["H_Gain"] / 10)

    # Calculate the sum of the new V-POL and H-POL magnitudes, multiplied by the sine of the Theta angle (converted to radians)
    theta_rad = np.deg2rad(combined_data["Theta"])
    sum_values = (G_linear_theta + G_linear_phi) * np.sin(theta_rad)

    # Calculate the radiated power using the formula
    M = total_phi_samples - 0
    N = total_theta_samples - 0
    radiated_power_mw = (np.pi / 2) * sum_values.sum() / (N * M)
    radiated_power_w = radiated_power_mw / 1000  # Convert to Watts

    # Calculate the electric field strength
    E_Theta_strength = np.sqrt(377 * G_linear_theta / 4 / np.pi)  # using V_Gain for E_Theta (V/m)
    E_Phi_strength = np.sqrt(377 * G_linear_phi / 4 / np.pi)  # using H_Gain for E_Phi (V/m)

    # Convert phase from degrees to radians for the calculation
    combined_data["V_Phase"] = np.radians(combined_data["V_Phase"])
    combined_data["H_Phase"] = np.radians(combined_data["H_Phase"])

    # Calculate the electric field components in complex form
    E_Theta_complex = E_Theta_strength * (
        np.cos(combined_data["V_Phase"]) + 1j * np.sin(combined_data["V_Phase"])
    )
    E_Phi_complex = E_Phi_strength * (
        np.cos(combined_data["H_Phase"]) + 1j * np.sin(combined_data["H_Phase"])
    )

    # Separate real and imaginary parts
    combined_data["Re(E_Theta)"] = np.real(E_Theta_complex)
    combined_data["Im(E_Theta)"] = np.imag(E_Theta_complex)
    combined_data["Re(E_Phi)"] = np.real(E_Phi_complex)
    combined_data["Im(E_Phi)"] = np.imag(E_Phi_complex)

    # Sort the DataFrame by 'Theta' and 'Phi'
    combined_data.sort_values(by=["Phi", "Theta"], inplace=True)

    # Round values to 2 decimal places
    combined_data = combined_data.round(
        {"Phi": 2, "Theta": 2, "Re(E_Theta)": 2, "Im(E_Theta)": 2, "Re(E_Phi)": 2, "Im(E_Phi)": 2}
    )

    # Format the DataFrame for output
    df_final = combined_data[
        ["Phi", "Theta", "Re(E_Theta)", "Im(E_Theta)", "Re(E_Phi)", "Im(E_Phi)"]
    ].copy()

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
    with open(output_path, "w", encoding="utf-8") as f:
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
    with open(file_path, "r", encoding="utf-8") as file:
        frequency = None
        read_values = False  # Initialize read_values at the start
        for line in file:
            if "Test Frequency" in line:
                frequency = float(line.split("=")[1].strip().replace("MHz", ""))
                read_values = False  # Reset the flag
                # print(f"Found frequency: {frequency} MHz")
            elif ("H-Pol" in line or "V-Pol" in line) and frequency is not None:
                read_values = True  # Next lines will contain the measurements
            elif read_values and frequency is not None and line.strip():
                try:
                    parts = line.split()
                    if len(parts) == 3 and parts[-1].replace("-", "").replace(".", "").isdigit():
                        power = float(parts[-1].strip())  # Assume the last part is the power value
                        data[frequency] = power
                        # print(f"Added power measurement: {power} dBm at {frequency} MHz")
                        read_values = False  # Reset the flag after reading the power value
                        frequency = None  # Reset frequency to None after reading the power value
                except ValueError as e:
                    print(
                        f"Error converting power value to float on line: {line.strip()}. Error: {e}"
                    )
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
    with open(file_path, "r", encoding="utf-8") as file:
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
    with open(file_path, "r", encoding="utf-8") as file:
        frequency = None
        read_values = False
        for line in file:
            if "Test Frequency" in line:
                frequency = float(line.split("=")[1].strip().replace("MHz", ""))
                read_values = False
            elif ("H-Pol" in line or "V-Pol" in line) and frequency is not None:
                read_values = True
            elif read_values and frequency is not None and line.strip():
                try:
                    parts = line.split()
                    if len(parts) == 3 and parts[-1].replace("-", "").replace(".", "").isdigit():
                        value = float(parts[-1].strip())
                        data[frequency] = value
                        # print(f"Added polarization data: {value} dBm at {frequency} MHz")
                        read_values = False
                        frequency = None
                except ValueError as e:
                    print(
                        f"Error converting polarization value to float on line: {line.strip()}. Error: {e}"
                    )
    return data


def generate_active_cal_file(
    power_measurement_file,
    gain_standard_file,
    hpol_file,
    vpol_file,
    cable_loss,
    freq_list,
    callback=None,
):
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
    common_part = (
        os.path.commonprefix([hpol_basename, vpol_basename]).split(".")[0].replace("AP_", "")
    )

    # Create the output file name based on the common part of the HPOL and VPOL files and the date
    output_file = f"TRP Cal {common_part}1Amp 0 dBm {start_frequency}-{stop_frequency} {today}.txt"
    output_path = os.path.join(os.path.dirname(hpol_file), output_file)

    # Write the result to a file
    with open(output_path, "w", encoding="utf-8") as file:
        file.write(
            f"WTL Calibration File\n\nDate: {today}\nTime: {current_time}\n\n*******  Cal Data  ********\n"
        )
        file.write("Freq\tH-Pol\tV-Pol\n(MHz)\t(dBm)\t(dBm)\n")
        for _, row in df_filtered.iterrows():
            file.write(f"{int(row['Frequency'])}\t{row['P+G-H']:.2f}\t{row['P+G-V']:.2f}\n")

    # Generate a summary file
    summary_file = (
        f"TRP Cal Summary {common_part}1Amp 0 dBm {start_frequency}-{stop_frequency} {today}.txt"
    )
    summary_path = os.path.join(os.path.dirname(hpol_file), summary_file)

    with open(summary_path, "w", encoding="utf-8") as file:
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


# --------------------------------------------------------------------------
# Bulk passive scan processing
# --------------------------------------------------------------------------
# NOTE: plotting imports happen inside the function to avoid circular imports.
def batch_process_passive_scans(
    folder_path,
    freq_list,
    selected_frequencies,
    cable_loss=0.0,
    datasheet_plots=False,
    save_base=None,
    axis_mode="auto",
    zmin: float = -15.0,
    zmax: float = 15.0,
    maritime_plots_enabled=False,
    maritime_theta_min=60.0,
    maritime_theta_max=120.0,
    maritime_theta_cuts=None,
    maritime_gain_threshold=-3.0,
):
    """
    Batch‑process all HPOL/VPOL pairs in a directory.

    Parameters:
        folder_path (str): Directory containing measurement files.
        freq_list (list of float): Full list of available frequencies (MHz).
        selected_frequencies (list of float): Frequencies to process for each pair (MHz).
        cable_loss (float): Cable loss applied to all datasets.
        datasheet_plots (bool): Whether to generate datasheet‑style plots.
        save_base (str or None): Optional directory to write results; a subfolder per pair will be created.
        axis_mode (str): 'auto' or 'manual' axis scaling for 3D plots.
        zmin (float): Minimum z‑axis limit when axis_mode='manual'.
        zmax (float): Maximum z‑axis limit when axis_mode='manual'.

    This routine scans ``folder_path`` for files ending in ``AP_HPol.txt`` and
    ``AP_VPol.txt``.  For each matching pair it computes passive gain data
    using :func:`calculate_passive_variables` and then generates 2D and 3D plots
    using :mod:`plotting`.  Results are saved into per‑pair subfolders in
    ``save_base`` if provided.
    """
    import os
    from .plotting import plot_2d_passive_data, plot_passive_3d_component

    # Find all HPOL and VPOL files
    files = os.listdir(folder_path)
    hpol_files = [f for f in files if f.endswith("AP_HPol.txt")]
    vpol_files = [f for f in files if f.endswith("AP_VPol.txt")]

    # Match by filename prefix
    pairs = []
    for h_file in hpol_files:
        base = h_file.replace("AP_HPol.txt", "")
        match = base + "AP_VPol.txt"
        if match in vpol_files:
            pairs.append((os.path.join(folder_path, h_file), os.path.join(folder_path, match)))

    if not pairs:
        print(f"No HPOL/VPOL pairs found in {folder_path}.")
        return

    for h_path, v_path in pairs:
        print(f"Processing pair: {os.path.basename(h_path)}, {os.path.basename(v_path)}")
        # Parse both files
        parsed_h, start_phi_h, stop_phi_h, inc_phi_h, start_theta_h, stop_theta_h, inc_theta_h = (
            read_passive_file(h_path)
        )
        parsed_v, start_phi_v, stop_phi_v, inc_phi_v, start_theta_v, stop_theta_v, inc_theta_v = (
            read_passive_file(v_path)
        )

        # Verify angle grids match
        if not angles_match(
            start_phi_h,
            stop_phi_h,
            inc_phi_h,
            start_theta_h,
            stop_theta_h,
            inc_theta_h,
            start_phi_v,
            stop_phi_v,
            inc_phi_v,
            start_theta_v,
            stop_theta_v,
            inc_theta_v,
        ):
            print(f"  Warning: angle mismatch between {h_path} and {v_path}.  Skipping.")
            continue

        for sel_freq in selected_frequencies:
            print(f"  Processing frequency {sel_freq} MHz…")
            # Compute gains for this frequency
            theta_deg, phi_deg, v_gain_dB, h_gain_dB, total_gain_dB = calculate_passive_variables(
                parsed_h,
                parsed_v,
                cable_loss,
                start_phi_h,
                stop_phi_h,
                inc_phi_h,
                start_theta_h,
                stop_theta_h,
                inc_theta_h,
                freq_list,
                sel_freq,
            )

            # Create per‑pair/frequency subfolder if requested
            if save_base:
                base_name = os.path.splitext(os.path.basename(h_path))[0].replace("AP_HPol", "")
                subfolder = os.path.join(save_base, f"{base_name}_{sel_freq}MHz")
                os.makedirs(subfolder, exist_ok=True)
            else:
                subfolder = None

            # 2D plots
            plot_2d_passive_data(
                theta_deg,
                phi_deg,
                v_gain_dB,
                h_gain_dB,
                total_gain_dB,
                freq_list,
                sel_freq,
                datasheet_plots,
                save_path=subfolder,
            )

            # 3D plots (total, hpol and vpol)
            for pol in ("total", "hpol", "vpol"):
                plot_passive_3d_component(
                    theta_deg,
                    phi_deg,
                    h_gain_dB,
                    v_gain_dB,
                    total_gain_dB,
                    freq_list,
                    sel_freq,
                    pol,
                    axis_mode=axis_mode,
                    zmin=zmin,
                    zmax=zmax,
                    save_path=subfolder,
                )

            # Maritime / Horizon plots
            if maritime_plots_enabled and subfolder:
                from .plotting import _prepare_gain_grid, generate_maritime_plots

                freq_idx = freq_list.index(sel_freq) if sel_freq in freq_list else 0
                unique_theta, unique_phi, gain_grid = _prepare_gain_grid(
                    theta_deg, phi_deg, total_gain_dB, freq_idx
                )
                if gain_grid is not None:
                    maritime_sub = os.path.join(subfolder, "Maritime Plots")
                    os.makedirs(maritime_sub, exist_ok=True)
                    generate_maritime_plots(
                        unique_theta, unique_phi, gain_grid, sel_freq,
                        data_label="Gain", data_unit="dBi",
                        theta_min=maritime_theta_min,
                        theta_max=maritime_theta_max,
                        theta_cuts=maritime_theta_cuts,
                        gain_threshold=maritime_gain_threshold,
                        axis_mode=axis_mode,
                        zmin=zmin,
                        zmax=zmax,
                        save_path=maritime_sub,
                    )


def batch_process_active_scans(
    folder_path,
    save_base=None,
    interpolate=True,
    axis_mode="auto",
    zmin: float = -15.0,
    zmax: float = 15.0,
    maritime_plots_enabled=False,
    maritime_theta_min=60.0,
    maritime_theta_max=120.0,
    maritime_theta_cuts=None,
    maritime_gain_threshold=-3.0,
):
    """
    Batch‑process all active TRP measurement files in a directory.

    Parameters:
        folder_path (str): Directory containing TRP measurement files.
        save_base (str or None): Optional directory to write results; a subfolder per file will be created.
        interpolate (bool): Whether to interpolate 3D plots for smoother visualization.
        axis_mode (str): 'auto' or 'manual' axis scaling for 3D plots.
        zmin (float): Minimum z‑axis limit (dBm) when axis_mode='manual'.
        zmax (float): Maximum z‑axis limit (dBm) when axis_mode='manual'.

    This routine scans ``folder_path`` for TRP data files (e.g., files ending in ``.txt``).
    For each file, it:
      1. Reads and parses the TRP data using :func:`read_active_file`.
      2. Calculates active variables using :func:`calculate_active_variables`.
      3. Generates 2D azimuth/elevation cuts and 3D TRP plots.
      4. Saves results to per‑file subfolders in ``save_base`` if provided.
    """
    import os
    from .plotting import plot_active_2d_data, plot_active_3d_data
    from .calculations import calculate_active_variables

    # Find all TRP files in the folder
    files = os.listdir(folder_path)
    trp_files = [f for f in files if f.endswith(".txt") and "TRP" in f.upper()]

    if not trp_files:
        print(f"No TRP files found in {folder_path}.")
        return

    for trp_file in trp_files:
        file_path = os.path.join(folder_path, trp_file)
        print(f"Processing TRP file: {trp_file}")

        try:
            # Read active file
            data = read_active_file(file_path)

            # Extract data
            frequency = data["Frequency"]
            start_phi = data["Start Phi"]
            start_theta = data["Start Theta"]
            stop_phi = data["Stop Phi"]
            stop_theta = data["Stop Theta"]
            inc_phi = data["Inc Phi"]
            inc_theta = data["Inc Theta"]
            h_power_dBm = data["H_Power_dBm"]
            v_power_dBm = data["V_Power_dBm"]

            # Calculate active variables
            active_vars = calculate_active_variables(
                start_phi,
                stop_phi,
                start_theta,
                stop_theta,
                inc_phi,
                inc_theta,
                h_power_dBm,
                v_power_dBm,
            )

            # Unpack variables
            (
                data_points,
                theta_angles_deg,
                phi_angles_deg,
                theta_angles_rad,
                phi_angles_rad,
                total_power_dBm_2d,
                h_power_dBm_2d,
                v_power_dBm_2d,
                phi_angles_deg_plot,
                phi_angles_rad_plot,
                total_power_dBm_2d_plot,
                h_power_dBm_2d_plot,
                v_power_dBm_2d_plot,
                total_power_dBm_min,
                total_power_dBm_nom,
                h_power_dBm_min,
                h_power_dBm_nom,
                v_power_dBm_min,
                v_power_dBm_nom,
                TRP_dBm,
                h_TRP_dBm,
                v_TRP_dBm,
            ) = active_vars

            # Create subfolder for this file if save_base is provided
            if save_base:
                base_name = os.path.splitext(trp_file)[0]
                subfolder = os.path.join(save_base, f"{base_name}_{frequency}MHz")
                os.makedirs(subfolder, exist_ok=True)
            else:
                subfolder = None

            # Generate 2D plots
            plot_active_2d_data(
                data_points,
                theta_angles_rad,
                phi_angles_rad_plot,
                total_power_dBm_2d_plot,
                frequency,
                save_path=subfolder,
            )

            # Generate 3D plots for total, hpol, and vpol
            for power_type, power_2d, power_2d_plot in [
                ("total", total_power_dBm_2d, total_power_dBm_2d_plot),
                ("hpol", h_power_dBm_2d, h_power_dBm_2d_plot),
                ("vpol", v_power_dBm_2d, v_power_dBm_2d_plot),
            ]:
                plot_active_3d_data(
                    theta_angles_deg,
                    phi_angles_deg,
                    power_2d,
                    phi_angles_deg_plot,
                    power_2d_plot,
                    frequency,
                    power_type=power_type,
                    interpolate=interpolate,
                    axis_mode=axis_mode,
                    zmin=zmin,
                    zmax=zmax,
                    save_path=subfolder,
                )

            # Maritime / Horizon plots
            if maritime_plots_enabled and subfolder:
                from .plotting import generate_maritime_plots

                maritime_sub = os.path.join(subfolder, "Maritime Plots")
                os.makedirs(maritime_sub, exist_ok=True)
                generate_maritime_plots(
                    theta_angles_deg, phi_angles_deg, total_power_dBm_2d, frequency,
                    data_label="Power", data_unit="dBm",
                    theta_min=maritime_theta_min,
                    theta_max=maritime_theta_max,
                    theta_cuts=maritime_theta_cuts,
                    gain_threshold=maritime_gain_threshold,
                    axis_mode=axis_mode,
                    zmin=zmin,
                    zmax=zmax,
                    save_path=maritime_sub,
                )

            print(f"  ✓ Completed {trp_file} at {frequency} MHz (TRP={TRP_dBm:.2f} dBm)")

        except Exception as e:
            print(f"  ✗ Error processing {trp_file}: {e}")
