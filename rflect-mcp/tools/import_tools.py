"""
Import Tools for RFlect MCP Server

Handles importing antenna measurement files and folders.
"""

import os
import glob
from typing import Optional, List, Dict, Any
from dataclasses import dataclass, field

# Import RFlect file utilities
from plot_antenna.file_utils import read_passive_file, read_active_file


@dataclass
class LoadedMeasurement:
    """Represents a loaded antenna measurement."""
    file_path: str
    scan_type: str  # 'passive' or 'active'
    frequencies: List[float]
    data: Dict[str, Any]


# Global storage for loaded measurements
_loaded_measurements: Dict[str, LoadedMeasurement] = {}


def get_loaded_data_summary() -> str:
    """Get summary of currently loaded data."""
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
                data = read_passive_file(file_path)
            else:
                data = read_active_file(file_path)

            # Extract frequencies
            frequencies = []
            if isinstance(data, dict) and 'frequency' in data:
                frequencies = [data['frequency']]
            elif isinstance(data, list):
                frequencies = [d.get('frequency', 0) for d in data if isinstance(d, dict)]

            # Store the measurement
            name = os.path.basename(file_path)
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
    """Get all loaded measurements (for use by other tools)."""
    return _loaded_measurements
