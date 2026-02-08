from .file_utils import parse_2port_data, parse_agilent_data, parse_touchstone_to_dataframe
from .uwb_analysis import build_complex_s21_from_s2vna, calculate_sff, calculate_sff_vs_angle
from .uwb_plotting import plot_sff_vs_angle, plot_input_vs_output_pulse
from matplotlib.ticker import ScalarFormatter

import matplotlib

matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
import os
import re
import numpy as np


# Process & Plot 2-Port S-Parameter Files for Group Delay and System Fidelity Factor
def process_groupdelay_files(
    file_paths,
    saved_limit1_freq1,
    saved_limit1_freq2,
    saved_limit1_start,
    saved_limit1_stop,
    saved_limit2_freq1,
    saved_limit2_freq2,
    saved_limit2_start,
    saved_limit2_stop,
    min_freq,
    max_freq,
):
    # Placeholder for data storage with theta values as keys
    data_dict = {}

    # Regular expression to match a number followed by 'deg'
    pattern = re.compile(r"(\d+)(deg|DEG)", re.IGNORECASE)

    # Extract data from files
    for file_path in file_paths:
        # Extracting theta from file name using regex
        filename = os.path.basename(file_path)
        match = pattern.search(filename)
        if match:
            theta = match.group(1)  # This will be the number matched by (\d+)
        else:
            print(f"Warning: Could not extract theta from filename: {filename}")
            continue  # Skip this file if no match is found

        # Determine file type and parse accordingly
        if file_path.lower().endswith(".s2p"):
            # Process Touchstone .s2p file
            data = parse_touchstone_to_dataframe(file_path)
        else:
            # Read the first line to determine the source, if Agilent or S2VNA
            with open(file_path, "r", encoding="utf-8-sig") as f:
                first_line = f.readline()  # Read the first line
            if "!" in first_line:
                # Process S2VNA 2-Port Measurement
                data = parse_2port_data(file_path)
            elif "#" in first_line:
                # Process Agilent Group Delay Measurement
                data = parse_agilent_data(file_path)
            else:
                print(f"Warning: Could not recognize file structure of: {filename}")
                data = None

        if data is not None:
            # Storing data
            data_dict[theta] = data
    # Plotting:
    # Group Delay vs Frequency for Various Theta, Group Delay Difference vs Theta, & Max. Distance Error vs Theta
    plot_group_delay_error(data_dict, min_freq, max_freq)

    # System Fidelity Factor (requires S21(dB) + S21(s) group delay columns)
    plot_total_system_fidelity(data_dict, min_freq, max_freq)

    return


# Function to plot Peak-to-peak Group Delay Difference & Corresponding Error
def plot_group_delay_error(data_dict, min_freq=None, max_freq=None):
    # Plot Group Delay Vs Frequency
    plt.figure(figsize=(10, 6))
    for theta, data in data_dict.items():
        if min_freq is not None and max_freq is not None:
            # Convert frequencies to GHz since your min_freq and max_freq are probably in GHz
            data = data[
                (data["! Stimulus(Hz)"] >= min_freq * 1e9)
                & (data["! Stimulus(Hz)"] <= max_freq * 1e9)
            ]

        if "S21(s)" or "S12(s)" in data.columns:
            if "S21(s)" in data.columns:
                data_group_delay = data["S21(s)"]
            else:
                data_group_delay = data["S12(s)"]
            plt.plot(data["! Stimulus(Hz)"], data_group_delay, label=f"Theta={theta} deg")
        else:
            print(f"Warning: 'S21(s)' column not available for Theta={theta} deg, skipping plot.")

    # Set frequency range if specified
    if min_freq is not None and max_freq is not None:
        plt.xlim(min_freq * 1e9, max_freq * 1e9)
    # Initialize an empty list to store the filtered frequency points
    filtered_freq_points = []

    # If min_freq and max_freq are specified, filter the frequency points
    if min_freq is not None and max_freq is not None:
        for theta, data in data_dict.items():
            data = data[
                (data["! Stimulus(Hz)"] >= min_freq * 1e9)
                & (data["! Stimulus(Hz)"] <= max_freq * 1e9)
            ]
            data_dict[theta] = data  # Store the filtered data back in the data_dict

            if (
                len(filtered_freq_points) == 0
            ):  # If the list is empty, populate it with the first set of filtered frequency points
                filtered_freq_points = data["! Stimulus(Hz)"].to_numpy()

    else:  # If no frequency range is specified, use all frequency points from the first dataset
        filtered_freq_points = data_dict[next(iter(data_dict))]["! Stimulus(Hz)"].to_numpy()
        plt.ylim(0, 5e-9)  # This sets the Y-axis limits from 0 to 5 ns
    plt.xlabel("Frequency (GHz)")
    plt.ylabel("Group Delay (ns)")

    # Create a new formatter
    formatter = ScalarFormatter(useOffset=False)
    formatter.set_scientific(True)
    formatter.set_powerlimits((9, 9))  # This line forces the scientific notation to 10^9

    # Apply the formatter
    plt.gca().xaxis.set_major_formatter(formatter)

    plt.legend()
    plt.title("Group Delay vs Frequency for Various Theta (Azimuthal) Rotation")
    plt.grid(True, which="both", linestyle="--", linewidth=0.5)
    plt.show()

    # Plot Group Delay Difference & Error Vs Frequency
    difference_data = []
    error_data = []
    variance_data = []
    std_dev_data = []

    # Extracting all available frequency points from the data
    freq_points = data_dict[next(iter(data_dict))]["! Stimulus(Hz)"].to_numpy()

    # Calculating difference for each frequency point across all theta
    for freq in freq_points:
        group_delays_at_freq = [
            data[data["! Stimulus(Hz)"] == freq][
                ("S21(s)" if "S21(s)" in data.columns else "S12(s)")
            ].values[0]
            for theta, data in data_dict.items()
            if ("S21(s)" in data.columns or "S12(s)" in data.columns)
        ]
        difference_at_freq = np.max(group_delays_at_freq) - np.min(group_delays_at_freq)

        difference_data.append(difference_at_freq * 1e12)
        error_at_freq = (difference_at_freq) * 29979245800
        error_data.append(error_at_freq)
        """
        # TODO Calculate the variance in picoseconds
        variance_at_freq = np.var(group_delays_at_freq_ps)
        variance_data.append(variance_at_freq)

        # TODO Calculate the standard deviation
        std_dev_at_freq = np.std(group_delays_at_freq_ps)
        std_dev_data.append(std_dev_at_freq)
        """
    # Plot Group Delay difference
    plt.figure(figsize=(10, 6))
    plt.plot(freq_points, difference_data, label="Max Group Delay Difference over Theta")

    # Set frequency range if specified
    if min_freq is not None and max_freq is not None:
        plt.xlim(min_freq * 1e9, max_freq * 1e9)

    plt.xlabel("Frequency (GHz)")
    plt.ylabel("Peak-to-Peak Group Delay Difference over Theta (ps)")

    # Apply the formatter
    plt.gca().xaxis.set_major_formatter(formatter)

    plt.legend()
    plt.title("Max Group Delay Difference over Theta")
    plt.grid(True, which="both", linestyle="--", linewidth=0.5)
    # Use plain formatting (not scientific notation) for the y-axis
    plt.ticklabel_format(style="plain", axis="y", scilimits=(0, 0))
    plt.show()

    # Plot Max Distance Error Vs Theta
    plt.figure(figsize=(10, 6))
    plt.plot(freq_points, error_data, label="Max Distance Error over Theta")

    # Set frequency range if specified
    if min_freq is not None and max_freq is not None:
        plt.xlim(min_freq * 1e9, max_freq * 1e9)

    plt.xlabel("Frequency (GHz)")
    plt.ylabel("Max Distance Error (cm)")

    # Apply the formatter
    plt.gca().xaxis.set_major_formatter(formatter)

    plt.legend()
    plt.title("Max Distance Error over Theta")
    plt.grid(True, which="both", linestyle="--", linewidth=0.5)
    plt.ticklabel_format(style="plain", axis="y", scilimits=(0, 0))
    plt.show()


# Function to Plot Total System Fidelity
def plot_total_system_fidelity(data_dict, min_freq=None, max_freq=None):
    """
    Compute and plot the System Fidelity Factor (SFF) across all theta angles.

    Requires both S21(dB) and S21(s) (group delay) columns. Files without
    group delay are skipped with a warning.

    Parameters:
        data_dict: dict mapping theta labels to DataFrames.
        min_freq: float, optional — minimum frequency in GHz.
        max_freq: float, optional — maximum frequency in GHz.
    """
    angle_data = []
    last_sff_result = None

    for theta, data in data_dict.items():
        # Check for required columns
        has_s21_dB = "S21(dB)" in data.columns or "S12(dB)" in data.columns
        has_group_delay = "S21(s)" in data.columns or "S12(s)" in data.columns

        if not has_s21_dB or not has_group_delay:
            print(f"Warning: Theta={theta} deg missing S21(dB) or group delay column, "
                  f"skipping SFF. Available: {list(data.columns)}")
            continue

        freq = data["! Stimulus(Hz)"].values
        s21_dB = data[("S21(dB)" if "S21(dB)" in data.columns else "S12(dB)")].values
        gd = data[("S21(s)" if "S21(s)" in data.columns else "S12(s)")].values

        # Apply frequency filter
        if min_freq is not None and max_freq is not None:
            mask = (freq >= min_freq * 1e9) & (freq <= max_freq * 1e9)
            freq = freq[mask]
            s21_dB = s21_dB[mask]
            gd = gd[mask]

        if len(freq) < 2:
            print(f"Warning: Theta={theta} deg has insufficient data after filtering, skipping.")
            continue

        # Build complex S21 from magnitude + group delay
        s21_complex = build_complex_s21_from_s2vna(freq, s21_dB, gd)

        angle_data.append({
            'angle_deg': float(theta),
            'freq_hz': freq,
            's21_complex': s21_complex,
        })

    if not angle_data:
        print("Warning: No angles had both S21(dB) and group delay data. SFF not computed.")
        return

    # Compute SFF for all angles
    multi_result = calculate_sff_vs_angle(angle_data)

    for angle, sff_val, quality in zip(
        multi_result['angles'], multi_result['sff_values'], multi_result['qualities']
    ):
        print(f"System Fidelity Factor for Theta={angle:.0f} deg: {sff_val:.4f} ({quality})")

    print(f"Mean SFF across all angles: {multi_result['mean_sff']:.4f}")

    # Plot SFF vs angle bar chart
    fig_sff = plot_sff_vs_angle(multi_result['angles'], multi_result['sff_values'])
    plt.show()

    # Plot input vs output pulse for the last angle (boresight-like)
    last_entry = angle_data[-1]
    last_sff_result = calculate_sff(
        last_entry['freq_hz'], last_entry['s21_complex']
    )
    fig_pulse = plot_input_vs_output_pulse(
        last_sff_result['time_s'],
        last_sff_result['input_pulse'],
        last_sff_result['output_pulse'],
        last_sff_result['sff'],
        last_sff_result['peak_delay_s'],
    )
    plt.show()


# Calculate System Fidelity Factor from S-parameters
def calculate_SFF_with_gaussian_pulse(freq, S_param, group_delay_s=None):
    """
    Calculate the System Fidelity Factor (SFF) from S-parameters.

    Uses cross-correlation based SFF via uwb_analysis. If group_delay_s is
    provided, phase is reconstructed from group delay for a proper complex S21.
    Otherwise falls back to magnitude-only (less accurate).

    Parameters:
    - freq: 1D array of frequency points (Hz)
    - S_param: 1D array of S21 magnitude in dB
    - group_delay_s: 1D array of group delay in seconds (optional)

    Returns:
    - SFF: The System Fidelity Factor (float)
    - input_pulse: 1D array of the input pulse
    - output_pulse: 1D array of the output pulse
    - time_s: 1D time array
    """
    freq = np.asarray(freq, dtype=float)
    S_param = np.asarray(S_param, dtype=float)

    if group_delay_s is not None:
        s21_complex = build_complex_s21_from_s2vna(freq, S_param, group_delay_s)
    else:
        # Fallback: magnitude only with zero phase (less accurate)
        s21_complex = 10 ** (S_param / 20.0) + 0j

    result = calculate_sff(freq, s21_complex)

    return result['sff'], result['input_pulse'], result['output_pulse'], result['time_s']
