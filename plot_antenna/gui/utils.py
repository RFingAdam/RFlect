"""
Utility classes and functions for the RFlect GUI.

Contains shared helper classes and functions used across the GUI modules.
"""

import os
import sys
import json
import tkinter as tk
from ..file_utils import parse_2port_data
import pandas as pd
import numpy as np


class DualOutput:
    """Custom class to write to both console and Text widget."""

    def __init__(self, widget, stream=None):
        self.widget = widget
        self.stream = stream

    def write(self, string):
        self.widget.configure(state="normal")
        self.widget.insert("end", string)
        self.widget.configure(state="disabled")
        self.widget.see("end")
        if self.stream:  # Only write to the stream if it's valid
            self.stream.write(string)
            self.stream.flush()

    def flush(self):
        pass


def resource_path(relative_path):
    """Get absolute path to resource, works for dev and for PyInstaller."""
    try:
        # PyInstaller creates a temp folder and stores path in _MEIPASS
        base_path = sys._MEIPASS  # type: ignore
    except Exception:
        base_path = os.path.join(os.path.dirname(__file__), "..", "..")

    return os.path.abspath(os.path.join(base_path, relative_path))


def get_user_data_dir():
    """Get user-specific data directory for storing settings, API keys, recent files."""
    if sys.platform == "win32":
        # Windows: Use AppData\Local\RFlect
        app_data = os.getenv("LOCALAPPDATA", os.path.expanduser("~"))
        user_dir = os.path.join(app_data, "RFlect")
    elif sys.platform == "darwin":
        # macOS: Use ~/Library/Application Support/RFlect
        user_dir = os.path.expanduser("~/Library/Application Support/RFlect")
    else:
        # Linux: Use ~/.config/RFlect
        user_dir = os.path.expanduser("~/.config/RFlect")

    # Create directory if it doesn't exist
    os.makedirs(user_dir, exist_ok=True)
    return user_dir


def get_current_version():
    """Get the current version from settings.json."""
    settings_path = resource_path("settings.json")
    try:
        with open(settings_path, "r", encoding="utf-8") as file:
            settings = json.load(file)
            return settings.get("CURRENT_VERSION", "Unknown")
    except Exception:
        return "Unknown"


def calculate_min_max_parameters(file_paths, bands, param_name):
    """
    Calculate Min/Max for a specified parameter (e.g., VSWR or S21) across defined frequency bands.

    Args:
        file_paths: List of file paths to analyze
        bands: List of (freq_min, freq_max) tuples defining frequency bands
        param_name: Parameter name to analyze (e.g., "VSWR", "S21")

    Returns:
        List of (band_label, band_data) tuples where band_data contains
        [(file_path, min_val, max_val), ...]
    """
    results = []

    for i, (freq_min, freq_max) in enumerate(bands, start=1):
        band_label = f"({freq_min}-{freq_max} MHz)"
        band_data = []

        for file_path in file_paths:
            # Parse data from file
            data = parse_2port_data(file_path)

            # Ensure '! Stimulus(Hz)' exists
            if "! Stimulus(Hz)" not in data.columns:
                raise ValueError(f"File '{file_path}' does not have a '! Stimulus(Hz)' column.")

            freqs_mhz = data["! Stimulus(Hz)"] / 1e6

            # Determine the parameter column based on param_name
            if param_name.upper() == "VSWR":
                candidates = [c for c in data.columns if "SWR" in c.upper()]
                if not candidates:
                    raise ValueError(f"No SWR columns found in '{file_path}' for VSWR calculation.")
            else:
                candidates = [c for c in data.columns if param_name.upper() in c.upper()]
                if not candidates:
                    raise ValueError(
                        f"No columns containing '{param_name}' found in '{file_path}'."
                    )

            # Use the first matching column
            desired_col = candidates[0]

            # Extract values and filter by frequency range
            values = pd.to_numeric(data[desired_col], errors="coerce").dropna()
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


def display_parameter_table(results, param_name, parent):
    """Display parameter results in a table format."""
    row = 0
    for band_label, band_data in results:
        # Band label
        tk.Label(parent, text=band_label, font=("Arial", 12, "bold")).grid(
            row=row, column=0, columnspan=3, pady=10
        )
        row += 1

        # Table headers for each band
        headers = ["File", f"Min {param_name}", f"Max {param_name}"]
        for col, header in enumerate(headers):
            tk.Label(parent, text=header, font=("Arial", 10, "bold")).grid(
                row=row, column=col, padx=10, pady=5
            )
        row += 1

        # Data rows for each file in the band
        for file, min_val, max_val in band_data:
            tk.Label(parent, text=os.path.basename(file)).grid(row=row, column=0, padx=10, pady=5)
            tk.Label(parent, text=f"{min_val:.2f}" if min_val is not None else "N/A").grid(
                row=row, column=1, padx=10, pady=5
            )
            tk.Label(parent, text=f"{max_val:.2f}" if max_val is not None else "N/A").grid(
                row=row, column=2, padx=10, pady=5
            )
            row += 1

        # Add some space between bands
        row += 1
