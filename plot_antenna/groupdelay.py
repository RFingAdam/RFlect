from .file_utils import parse_2port_data, parse_agilent_data
from matplotlib.ticker import ScalarFormatter

import matplotlib

matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
import os
import re
import numpy as np
from numpy.fft import fft, ifft


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

    # TODO System Fidelity Factor
    # plot_total_system_fidelity(data_dict, min_freq, max_freq)

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
    Function to plot the total system fidelity factor.

    Parameters:
        data_dict: dict
            Dictionary containing data for different theta angles.
        min_freq: float, optional
            Minimum frequency for plotting, in GHz. If None, use the smallest frequency available.
        max_freq: float, optional
            Maximum frequency for plotting, in GHz. If None, use the largest frequency available.
    """

    SFF_results = []
    theta_values = []

    # Iterating through each orientation and calculating SFF
    for theta, data in data_dict.items():
        # Extracting frequency and S-parameter (S21 or S12) data
        freq = data["! Stimulus(Hz)"].values
        S_param = data[("S21(dB)" if "S21(dB)" in data.columns else "S12(dB)")].values

        # Calculating SFF, Gaussian pulse and system impulse response
        SFF, p_t, h_sys, t = calculate_SFF_with_gaussian_pulse(freq, S_param)

        # Storing results
        theta_values.append(theta)
        SFF_results.append(SFF)

        print(f"System Fidelity Factor for Theta={theta} deg: {SFF}")

    # Plot Gaussian pulse and system impulse response
    plt.figure(figsize=(10, 6))
    plt.plot(t, p_t, label="Gaussian pulse p(t)")
    plt.plot(t[: len(h_sys)], h_sys, label="System impulse response h_sys(t)")
    plt.xlabel("Time (ns)")
    plt.ylabel("Amplitude")
    plt.legend()
    plt.title("Gaussian pulse and System Impulse Response")
    plt.grid(True, which="both", linestyle="--", linewidth=0.5)
    plt.ticklabel_format(style="plain", axis="y", scilimits=(0, 0))

    plt.show()

    # Plotting SFF vs Theta
    plt.figure(figsize=(10, 6))
    plt.plot(theta_values, SFF_results, marker="o", linestyle="-")

    plt.xlabel("Theta (deg)")
    plt.ylabel("System Fidelity Factor")
    plt.title("System Fidelity Factor vs Theta")
    plt.grid(True, which="both", linestyle="--", linewidth=0.5)
    plt.show()


# Calculate System Fidelity Factor from S-parameters
def calculate_SFF_with_gaussian_pulse(freq, S_param):
    """
    Calculate the System Fidelity Factor (SFF) from S-parameters,
    comparing the system response to a Gaussian pulse.

    Parameters:
    - freq: 1D array of frequency points
    - S_param: 1D array of S-parameters (complex) at the given frequency points
    - tau: Time constant for the Gaussian pulse

    Returns:
    - SFF: The System Fidelity Factor
    """
    # 1. Ensure S_param is in linear scale and complex form
    S_param_lin = 10 ** (S_param / 20)

    # 2. Inverse Fourier Transform to get impulse response
    h_t = ifft(S_param_lin)

    # 3. Generate Reference Gaussian pulse
    # Desire Parameters
    pulse_start = 1e-9
    pulse_width = 3e-9
    center = pulse_start + pulse_width / 2

    # Time vector
    t = np.linspace(-6e-9, 6e-9, len(h_t))
    p_t = np.exp(-(((t - center) / (pulse_width / 2)) ** 2))

    # 4. Obtain system impulse response
    h_sys = np.convolve(h_t, p_t, mode="same")

    # 5. Normalizing the pulses
    p_t = p_t / np.max(np.abs(p_t))
    h_sys = h_sys / np.max(np.abs(h_sys))

    # 6. Calculate System Fidelity Factor comparing h_sys and p_t
    SFF = np.abs(np.trapezoid(h_sys * p_t)) ** 2 / (
        np.trapezoid(np.abs(h_sys) ** 2) * np.trapezoid(np.abs(p_t) ** 2)
    )

    # Update time vector to match the length of h_sys
    t = np.linspace(-6e-9, 6e-9, len(h_sys))

    return SFF, p_t, h_sys, t
