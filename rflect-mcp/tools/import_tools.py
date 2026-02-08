"""
Import Tools for RFlect MCP Server

Handles importing antenna measurement files and folders.
"""

import os
import glob
import threading
import numpy as np
from typing import Optional, List, Dict, Any
from dataclasses import dataclass, field

# Import RFlect file utilities
from plot_antenna.file_utils import read_passive_file, read_active_file
from plot_antenna.calculations import calculate_passive_variables, calculate_active_variables


@dataclass
class LoadedMeasurement:
    """Represents a loaded antenna measurement."""
    file_path: str
    scan_type: str  # 'passive' or 'active'
    frequencies: List[float]
    data: Dict[str, Any]


# Global storage for loaded measurements
_loaded_measurements: Dict[str, LoadedMeasurement] = {}
_measurements_lock = threading.Lock()


def get_loaded_data_summary() -> str:
    """Get summary of currently loaded data."""
    with _measurements_lock:
        if not _loaded_measurements:
            return "No data loaded. Use import_antenna_file or import_antenna_folder to load measurements."

        summary = f"Loaded {len(_loaded_measurements)} measurement(s):\n\n"
        for name, measurement in _loaded_measurements.items():
            summary += f"- {name}\n"
            summary += f"  Type: {measurement.scan_type}\n"
            summary += f"  Frequencies: {measurement.frequencies}\n"
            summary += f"  File: {measurement.file_path}\n\n"

        return summary


def register_import_tools(mcp):
    """Register import tools with the MCP server."""

    @mcp.tool()
    def import_antenna_file(file_path: str, scan_type: str = "auto") -> str:
        """
        Import a single antenna measurement file.

        Args:
            file_path: Path to the measurement file (.csv, .txt, etc.)
            scan_type: Type of scan - "passive", "active", or "auto" (detect from file)

        Returns:
            Summary of imported data including frequencies and data points.
        """
        if not os.path.exists(file_path):
            return f"Error: File not found: {file_path}"

        try:
            # Auto-detect scan type from filename if needed
            if scan_type == "auto":
                fname = os.path.basename(file_path).lower()
                if "active" in fname or "trp" in fname:
                    scan_type = "active"
                else:
                    scan_type = "passive"

            # Read the file based on type
            if scan_type == "passive":
                # read_passive_file returns (all_data, start_phi, stop_phi, inc_phi,
                #                            start_theta, stop_theta, inc_theta)
                result = read_passive_file(file_path)
                parsed_data, start_phi, stop_phi, inc_phi, start_theta, stop_theta, inc_theta = result
                frequencies = [d['frequency'] for d in parsed_data if isinstance(d, dict) and 'frequency' in d]
                data = {
                    'parsed_data': parsed_data,
                    'start_phi': start_phi,
                    'stop_phi': stop_phi,
                    'inc_phi': inc_phi,
                    'start_theta': start_theta,
                    'stop_theta': stop_theta,
                    'inc_theta': inc_theta,
                }
            else:
                data = read_active_file(file_path)
                # Active data is a dict with 'Frequency' key
                frequencies = []
                if isinstance(data, dict) and 'Frequency' in data:
                    frequencies = [data['Frequency']]
                elif isinstance(data, dict) and 'frequency' in data:
                    frequencies = [data['frequency']]
                elif isinstance(data, list):
                    frequencies = [d.get('frequency', 0) for d in data if isinstance(d, dict)]

            # Store the measurement
            name = os.path.basename(file_path)
            with _measurements_lock:
                _loaded_measurements[name] = LoadedMeasurement(
                    file_path=file_path,
                    scan_type=scan_type,
                    frequencies=frequencies,
                    data=data
                )

            return f"Successfully imported: {name}\nType: {scan_type}\nFrequencies: {frequencies}\nData points loaded."

        except Exception as e:
            return f"Error importing file: {str(e)}"

    @mcp.tool()
    def import_antenna_folder(
        folder_path: str,
        pattern: str = "*.csv",
        scan_type: str = "auto"
    ) -> str:
        """
        Import all antenna measurement files from a folder.

        Args:
            folder_path: Path to folder containing measurement files
            pattern: File pattern to match (default: "*.csv")
            scan_type: Type of scan - "passive", "active", or "auto"

        Returns:
            Summary of all imported files.
        """
        if not os.path.exists(folder_path):
            return f"Error: Folder not found: {folder_path}"

        # Find matching files
        search_pattern = os.path.join(folder_path, pattern)
        files = glob.glob(search_pattern)

        if not files:
            # Try recursive search
            search_pattern = os.path.join(folder_path, "**", pattern)
            files = glob.glob(search_pattern, recursive=True)

        if not files:
            return f"No files matching '{pattern}' found in {folder_path}"

        results = []
        success_count = 0
        error_count = 0

        for file_path in sorted(files):
            try:
                result = import_antenna_file(file_path, scan_type)
                if "Successfully" in result:
                    success_count += 1
                else:
                    error_count += 1
                results.append(f"  {os.path.basename(file_path)}: OK")
            except Exception as e:
                error_count += 1
                results.append(f"  {os.path.basename(file_path)}: FAILED - {str(e)}")

        summary = f"Imported {success_count} file(s), {error_count} error(s)\n\n"
        summary += "Files:\n" + "\n".join(results)

        return summary

    @mcp.tool()
    def import_passive_pair(
        hpol_file: str,
        vpol_file: str,
        cable_loss: float = 0.0,
        name: str = "auto"
    ) -> str:
        """
        Import a matched HPOL + VPOL passive measurement pair and compute combined gain.

        This is the recommended way to import passive data for analysis.
        It processes both files together using calculate_passive_variables()
        to produce total gain, H-gain, and V-gain arrays ready for analysis.

        Args:
            hpol_file: Path to the H-polarization measurement file
            vpol_file: Path to the V-polarization measurement file
            cable_loss: Cable loss in dB to compensate (default: 0)
            name: Measurement name (default: auto-generated from filename)

        Returns:
            Summary of imported pair including frequencies and data shape.
        """
        for p in (hpol_file, vpol_file):
            if not os.path.exists(p):
                return f"Error: File not found: {p}"

        try:
            h_result = read_passive_file(hpol_file)
            hpol_data, start_phi, stop_phi, inc_phi, start_theta, stop_theta, inc_theta = h_result

            v_result = read_passive_file(vpol_file)
            vpol_data = v_result[0]  # Only need parsed_data, angles should match

            frequencies = [d['frequency'] for d in hpol_data if isinstance(d, dict) and 'frequency' in d]

            if not frequencies:
                return "Error: No frequencies found in the HPOL file."

            # Process all frequencies into gain arrays
            theta_deg, phi_deg, v_gain_dB, h_gain_dB, total_gain_dB = calculate_passive_variables(
                hpol_data, vpol_data, cable_loss,
                start_phi, stop_phi, inc_phi,
                start_theta, stop_theta, inc_theta,
                frequencies, frequencies[0]
            )

            # Build analyzer-compatible data dict
            analyzer_data = {
                'total_gain': total_gain_dB,
                'h_gain': h_gain_dB,
                'v_gain': v_gain_dB,
                'theta': theta_deg,
                'phi': phi_deg,
            }

            if name == "auto":
                # Extract common base name (e.g. "PassiveTest_BLE AP")
                h_base = os.path.basename(hpol_file)
                name = h_base.replace("_HPol", "").replace("_VPol", "").replace(".txt", "").strip()

            with _measurements_lock:
                _loaded_measurements[name] = LoadedMeasurement(
                    file_path=f"{hpol_file} + {vpol_file}",
                    scan_type="passive",
                    frequencies=frequencies,
                    data=analyzer_data,
                )

            n_spatial = total_gain_dB.shape[0]
            n_freqs = total_gain_dB.shape[1] if total_gain_dB.ndim == 2 else 1

            return (
                f"Successfully imported passive pair: {name}\n"
                f"HPOL: {os.path.basename(hpol_file)}\n"
                f"VPOL: {os.path.basename(vpol_file)}\n"
                f"Frequencies: {len(frequencies)} ({frequencies[0]:.1f} - {frequencies[-1]:.1f} MHz)\n"
                f"Spatial points: {n_spatial}\n"
                f"Gain array shape: {total_gain_dB.shape}\n"
                f"Peak total gain: {float(np.max(total_gain_dB)):.2f} dBi\n"
                f"Cable loss applied: {cable_loss} dB"
            )

        except Exception as e:
            return f"Error importing passive pair: {str(e)}"

    @mcp.tool()
    def import_active_processed(file_path: str, name: str = "auto") -> str:
        """
        Import an active TRP measurement file and compute total power + TRP.

        This is the recommended way to import active data for analysis.
        It processes the file through calculate_active_variables() to produce
        total power arrays and TRP values ready for analysis.

        Args:
            file_path: Path to the active TRP measurement file
            name: Measurement name (default: auto-generated from filename)

        Returns:
            Summary including TRP value, power range, and data shape.
        """
        if not os.path.exists(file_path):
            return f"Error: File not found: {file_path}"

        try:
            raw = read_active_file(file_path)

            frequency = raw.get('Frequency', raw.get('frequency', 0))

            result = calculate_active_variables(
                raw['Start Phi'], raw['Stop Phi'],
                raw['Start Theta'], raw['Stop Theta'],
                raw['Inc Phi'], raw['Inc Theta'],
                np.array(raw['H_Power_dBm']),
                np.array(raw['V_Power_dBm']),
            )

            (data_points, theta_deg, phi_deg, theta_rad, phi_rad,
             total_power_2d, h_power_2d, v_power_2d,
             phi_deg_plot, phi_rad_plot,
             total_power_2d_plot, h_power_2d_plot, v_power_2d_plot,
             total_power_min, total_power_nom,
             h_power_min, h_power_nom,
             v_power_min, v_power_nom,
             TRP_dBm, h_TRP_dBm, v_TRP_dBm) = result

            # Build analyzer-compatible data dict
            analyzer_data = {
                # Flattened arrays for AntennaAnalyzer backward compatibility
                'total_power': total_power_2d.flatten(),
                'h_power': h_power_2d.flatten(),
                'v_power': v_power_2d.flatten(),
                'TRP_dBm': float(TRP_dBm),
                'H_TRP_dBm': float(h_TRP_dBm),
                'V_TRP_dBm': float(v_TRP_dBm),
                'theta': theta_deg,
                'phi': phi_deg,
                # 2D arrays for plotting
                'data_points': data_points,
                'theta_rad': theta_rad,
                'phi_rad': phi_rad,
                'total_power_2d': total_power_2d,
                'h_power_2d': h_power_2d,
                'v_power_2d': v_power_2d,
                # Extended arrays for 3D plot wrapping
                'phi_deg_plot': phi_deg_plot,
                'phi_rad_plot': phi_rad_plot,
                'total_power_2d_plot': total_power_2d_plot,
                'h_power_2d_plot': h_power_2d_plot,
                'v_power_2d_plot': v_power_2d_plot,
            }

            if name == "auto":
                name = os.path.basename(file_path).replace(".txt", "").strip()

            with _measurements_lock:
                _loaded_measurements[name] = LoadedMeasurement(
                    file_path=file_path,
                    scan_type="active",
                    frequencies=[frequency],
                    data=analyzer_data,
                )

            return (
                f"Successfully imported active file: {name}\n"
                f"Frequency: {frequency} MHz\n"
                f"Data points: {data_points}\n"
                f"TRP: {TRP_dBm:.2f} dBm\n"
                f"H-TRP: {h_TRP_dBm:.2f} dBm | V-TRP: {v_TRP_dBm:.2f} dBm\n"
                f"Max total power: {float(np.max(total_power_2d)):.2f} dBm\n"
                f"Min total power: {float(np.min(total_power_2d)):.2f} dBm"
            )

        except Exception as e:
            return f"Error importing active file: {str(e)}"

    @mcp.tool()
    def list_loaded_data() -> str:
        """
        List all currently loaded antenna measurements.

        Returns:
            Summary of loaded measurements with frequencies and types.
        """
        return get_loaded_data_summary()

    @mcp.tool()
    def clear_data() -> str:
        """
        Clear all loaded measurement data.

        Returns:
            Confirmation message.
        """
        with _measurements_lock:
            count = len(_loaded_measurements)
            _loaded_measurements.clear()
        return f"Cleared {count} measurement(s) from memory."

    @mcp.tool()
    def get_measurement_details(measurement_name: str) -> str:
        """
        Get detailed information about a specific loaded measurement.

        Args:
            measurement_name: Name of the measurement (filename)

        Returns:
            Detailed information about the measurement data.
        """
        with _measurements_lock:
            if measurement_name not in _loaded_measurements:
                available = list(_loaded_measurements.keys())
                return f"Measurement '{measurement_name}' not found. Available: {available}"

            m = _loaded_measurements[measurement_name]

        details = f"Measurement: {measurement_name}\n"
        details += f"File: {m.file_path}\n"
        details += f"Type: {m.scan_type}\n"
        details += f"Frequencies: {m.frequencies}\n\n"

        # Add data structure info
        if isinstance(m.data, dict):
            details += f"Data keys: {list(m.data.keys())}\n"
        elif isinstance(m.data, list):
            details += f"Data records: {len(m.data)}\n"

        return details


def get_loaded_measurements() -> Dict[str, LoadedMeasurement]:
    """Get a snapshot of all loaded measurements (for use by other tools)."""
    with _measurements_lock:
        return dict(_loaded_measurements)
