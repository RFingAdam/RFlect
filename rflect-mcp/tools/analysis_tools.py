"""
Analysis Tools for RFlect MCP Server

Provides pattern analysis, gain statistics, polarization comparison,
and frequency extrapolation.
Uses plot_antenna.ai_analysis.AntennaAnalyzer for core functionality.
"""

import os
from typing import Optional, List, Dict, Any
import json
import numpy as np

# Import from RFlect
from plot_antenna.ai_analysis import AntennaAnalyzer
from plot_antenna.calculations import extrapolate_pattern, validate_extrapolation
from plot_antenna.file_utils import read_passive_file
from .import_tools import get_loaded_measurements, LoadedMeasurement


def _fmt(value, fmt=".2f"):
    """Safely format a numeric value, returning 'N/A' for non-numeric types."""
    if value is None or value == "N/A":
        return "N/A"
    try:
        return f"{float(value):{fmt}}"
    except (TypeError, ValueError):
        return str(value)


def _get_analyzer_for_measurement(measurement_name: Optional[str] = None) -> tuple:
    """
    Get an AntennaAnalyzer for the specified or first loaded measurement.

    Returns:
        Tuple of (AntennaAnalyzer, measurement_name, error_message)
    """
    measurements = get_loaded_measurements()

    if not measurements:
        return None, None, "No data loaded. Use import_antenna_file first."

    # If no name specified, use the first measurement
    if measurement_name is None:
        measurement_name = list(measurements.keys())[0]

    if measurement_name not in measurements:
        available = list(measurements.keys())
        return None, None, f"Measurement '{measurement_name}' not found. Available: {available}"

    m = measurements[measurement_name]

    try:
        # Create analyzer from loaded data
        analyzer = AntennaAnalyzer(
            measurement_data=m.data,
            scan_type=m.scan_type,
            frequencies=m.frequencies
        )
        return analyzer, measurement_name, None
    except Exception as e:
        return None, None, f"Error creating analyzer: {str(e)}"


# ---- Standalone analysis functions (callable without MCP registration) ----

def list_frequencies(measurement_name: Optional[str] = None) -> str:
    """
    List available frequencies in the loaded measurement data.

    Args:
        measurement_name: Optional specific measurement (uses first if not specified)

    Returns:
        List of frequencies in MHz.
    """
    measurements = get_loaded_measurements()

    if not measurements:
        return "No data loaded. Use import_antenna_file first."

    if measurement_name:
        if measurement_name not in measurements:
            return f"Measurement '{measurement_name}' not found."
        freqs = measurements[measurement_name].frequencies
        return f"Frequencies in {measurement_name}: {freqs} MHz"

    # List all frequencies from all measurements
    result = "Available frequencies:\n\n"
    for name, m in measurements.items():
        result += f"{name}: {m.frequencies} MHz\n"

    return result


def analyze_pattern(
    frequency: Optional[float] = None,
    polarization: str = "total",
    measurement_name: Optional[str] = None
) -> str:
    """
    Analyze radiation pattern characteristics.

    Args:
        frequency: Frequency in MHz (uses first available if not specified)
        polarization: "total", "hpol", or "vpol"
        measurement_name: Optional specific measurement

    Returns:
        Pattern analysis including pattern type, HPBW, F/B ratio, and null info.
    """
    analyzer, name, error = _get_analyzer_for_measurement(measurement_name)
    if error:
        return error

    try:
        # Use first frequency if not specified
        if frequency is None:
            measurements = get_loaded_measurements()
            freqs = measurements[name].frequencies
            frequency = freqs[0] if freqs else None

        if frequency is None:
            return "No frequency available in the data."

        result = analyzer.analyze_pattern(frequency=frequency)

        output = f"Pattern Analysis for {name} @ {frequency} MHz\n"
        output += f"Polarization: {polarization}\n"
        output += "-" * 40 + "\n\n"

        output += f"Pattern Type: {result.get('pattern_type', 'Unknown')}\n"
        output += f"Peak Gain: {_fmt(result.get('peak_gain_dBi'))} dBi\n"
        output += f"HPBW (E-plane): {_fmt(result.get('hpbw_e_plane'), '.1f')}°\n"
        output += f"HPBW (H-plane): {_fmt(result.get('hpbw_h_plane'), '.1f')}°\n"
        output += f"Front-to-Back Ratio: {_fmt(result.get('front_to_back_dB'), '.1f')} dB\n"

        main_theta = result.get('main_beam_theta')
        main_phi = result.get('main_beam_phi')
        if main_theta is not None and main_phi is not None:
            output += f"Main Beam Direction: \u03b8={_fmt(main_theta, '.1f')}°, \u03c6={_fmt(main_phi, '.1f')}°\n"

        num_nulls = result.get('num_nulls', 0)
        if num_nulls:
            output += f"\nNulls: {num_nulls} found\n"
            deepest = result.get('deepest_null_dB')
            if deepest is not None:
                output += f"Deepest Null: {_fmt(deepest)} dB\n"

        return output

    except Exception as e:
        return f"Error analyzing pattern: {str(e)}"


def get_gain_statistics(
    frequency: Optional[float] = None,
    measurement_name: Optional[str] = None
) -> str:
    """
    Get gain statistics for the antenna measurement.

    Args:
        frequency: Frequency in MHz (uses first available if not specified)
        measurement_name: Optional specific measurement

    Returns:
        Gain statistics including min, max, average gain in dBi.
    """
    analyzer, name, error = _get_analyzer_for_measurement(measurement_name)
    if error:
        return error

    try:
        # Use first frequency if not specified
        if frequency is None:
            measurements = get_loaded_measurements()
            freqs = measurements[name].frequencies
            frequency = freqs[0] if freqs else None

        if frequency is None:
            return "No frequency available in the data."

        result = analyzer.get_gain_statistics(frequency=frequency)

        output = f"Gain Statistics for {name} @ {frequency} MHz\n"
        output += "-" * 40 + "\n\n"

        # Handle both passive (dBi) and active (dBm) scan types
        if result.get('scan_type') == 'active':
            output += f"Maximum Power: {_fmt(result.get('max_power_dBm'))} dBm\n"
            output += f"Minimum Power: {_fmt(result.get('min_power_dBm'))} dBm\n"
            output += f"Average Power: {_fmt(result.get('avg_power_dBm'))} dBm\n"
            output += f"Std Dev: {_fmt(result.get('std_dev_dB'))} dB\n"
            if 'TRP_dBm' in result:
                output += f"TRP: {_fmt(result.get('TRP_dBm'))} dBm\n"
        else:
            output += f"Maximum Gain: {_fmt(result.get('max_gain_dBi'))} dBi\n"
            output += f"Minimum Gain: {_fmt(result.get('min_gain_dBi'))} dBi\n"
            output += f"Average Gain: {_fmt(result.get('avg_gain_dBi'))} dBi\n"
            output += f"Std Dev: {_fmt(result.get('std_dev_dB'))} dB\n"

            # Polarization-specific if available
            if 'max_hpol_gain_dBi' in result:
                output += f"\nHPOL Max Gain: {_fmt(result.get('max_hpol_gain_dBi'))} dBi\n"
            if 'max_vpol_gain_dBi' in result:
                output += f"VPOL Max Gain: {_fmt(result.get('max_vpol_gain_dBi'))} dBi\n"

        return output

    except Exception as e:
        return f"Error getting gain statistics: {str(e)}"


def compare_polarizations(
    frequency: Optional[float] = None,
    measurement_name: Optional[str] = None
) -> str:
    """
    Compare HPOL and VPOL polarization components.

    Args:
        frequency: Frequency in MHz (uses first available if not specified)
        measurement_name: Optional specific measurement

    Returns:
        Polarization comparison including XPD (cross-polar discrimination).
    """
    analyzer, name, error = _get_analyzer_for_measurement(measurement_name)
    if error:
        return error

    try:
        # Use first frequency if not specified
        if frequency is None:
            measurements = get_loaded_measurements()
            freqs = measurements[name].frequencies
            frequency = freqs[0] if freqs else None

        if frequency is None:
            return "No frequency available in the data."

        result = analyzer.compare_polarizations(frequency=frequency)

        output = f"Polarization Comparison for {name} @ {frequency} MHz\n"
        output += "-" * 40 + "\n\n"

        output += f"Average XPD: {_fmt(result.get('avg_xpd_dB'), '.1f')} dB\n"
        output += f"Max XPD: {_fmt(result.get('max_xpd_dB'), '.1f')} dB\n\n"

        output += f"HPOL Max Gain: {_fmt(result.get('max_hpol_gain_dBi'))} dBi\n"
        output += f"VPOL Max Gain: {_fmt(result.get('max_vpol_gain_dBi'))} dBi\n"

        balance = result.get('polarization_balance_dB')
        if balance is not None:
            output += f"Polarization Balance: {_fmt(balance, '.1f')} dB\n"

        note = result.get('polarization_note')
        if note:
            output += f"Assessment: {note}\n"

        return output

    except Exception as e:
        return f"Error comparing polarizations: {str(e)}"


def get_all_analysis(
    frequency: Optional[float] = None,
    measurement_name: Optional[str] = None
) -> str:
    """
    Get comprehensive analysis of the antenna measurement.
    Combines pattern analysis, gain statistics, and polarization comparison.

    Args:
        frequency: Frequency in MHz (uses first available if not specified)
        measurement_name: Optional specific measurement

    Returns:
        Complete analysis summary suitable for report generation.
    """
    results = []

    # Get each analysis
    results.append(get_gain_statistics(frequency, measurement_name))
    results.append("")
    results.append(analyze_pattern(frequency, "total", measurement_name))
    results.append("")
    results.append(compare_polarizations(frequency, measurement_name))

    return "\n".join(results)


def extrapolate_to_frequency(
    hpol_file: str,
    vpol_file: str,
    target_frequency: float,
    fit_degree: int = 2,
) -> str:
    """
    Extrapolate antenna pattern to a target frequency outside the measured range.

    Fits polynomial curves to gain-vs-frequency for each spatial point across
    the measured band, then evaluates at the target frequency. Useful for
    estimating performance at frequencies below the chamber's calibration range.

    Args:
        hpol_file: Path to the H-polarization measurement file
        vpol_file: Path to the V-polarization measurement file
        target_frequency: Target frequency in MHz
        fit_degree: Polynomial order for magnitude fitting (default 2)

    Returns:
        Extrapolation summary with estimated gain stats and confidence metrics.
    """
    for p in (hpol_file, vpol_file):
        if not os.path.exists(p):
            return f"Error: File not found: {p}"

    try:
        h_result = read_passive_file(hpol_file)
        hpol_data = h_result[0]
        v_result = read_passive_file(vpol_file)
        vpol_data = v_result[0]

        freqs = [d["frequency"] for d in hpol_data]

        result = extrapolate_pattern(
            hpol_data, vpol_data, target_frequency, fit_degree=fit_degree
        )

        h_mag = np.array(result["hpol"]["mag"])
        v_mag = np.array(result["vpol"]["mag"])
        total_linear = 10 ** (h_mag / 10) + 10 ** (v_mag / 10)
        total_dB = 10 * np.log10(np.maximum(total_linear, 1e-12))

        conf = result["confidence"]

        output = f"Frequency Extrapolation to {target_frequency} MHz\n"
        output += "=" * 50 + "\n\n"
        output += f"Measured range: {min(freqs):.1f} - {max(freqs):.1f} MHz "
        output += f"({len(freqs)} frequencies)\n"
        output += f"Target: {target_frequency} MHz\n\n"

        output += "Estimated Gain Statistics\n"
        output += "-" * 30 + "\n"
        output += f"  Peak Total Gain: {_fmt(float(np.max(total_dB)))} dBi\n"
        output += f"  Avg Total Gain:  {_fmt(float(np.mean(total_dB)))} dBi\n"
        output += f"  Peak HPOL Gain:  {_fmt(float(np.max(h_mag)))} dBi\n"
        output += f"  Peak VPOL Gain:  {_fmt(float(np.max(v_mag)))} dBi\n\n"

        output += "Confidence\n"
        output += "-" * 30 + "\n"
        output += f"  Quality: {conf['quality']}\n"
        output += f"  Extrapolation Ratio: {conf['extrapolation_ratio']:.3f}\n"
        output += f"  Mean R²: {conf['mean_r_squared']:.4f}\n"
        output += f"  Est. Max Error: {conf['max_estimated_error_dB']:.1f} dB\n"
        if conf.get("warning"):
            output += f"  Warning: {conf['warning']}\n"

        return output

    except Exception as e:
        return f"Error during extrapolation: {str(e)}"


def get_horizon_statistics(
    frequency: Optional[float] = None,
    theta_min: float = 60.0,
    theta_max: float = 120.0,
    gain_threshold: float = -3.0,
    measurement_name: Optional[str] = None,
) -> str:
    """
    Get horizon-band statistics for maritime/on-water antenna applications.

    Analyzes the antenna pattern in the horizon region (default theta 60-120 deg)
    and returns coverage, MEG, null detection, and gain statistics.

    Args:
        frequency: Frequency in MHz (uses first available if not specified)
        theta_min: Minimum theta angle for horizon band (default 60 deg)
        theta_max: Maximum theta angle for horizon band (default 120 deg)
        gain_threshold: dB threshold below peak for coverage calculation (default -3)
        measurement_name: Optional specific measurement (uses first if not specified)

    Returns:
        Horizon statistics including min/max/avg gain, coverage %, MEG, and null info.
    """
    analyzer, name, error = _get_analyzer_for_measurement(measurement_name)
    if error:
        return error

    try:
        if frequency is None:
            measurements = get_loaded_measurements()
            freqs = measurements[name].frequencies
            frequency = freqs[0] if freqs else None

        if frequency is None:
            return "No frequency available in the data."

        result = analyzer.get_horizon_statistics(
            frequency=frequency,
            theta_min=theta_min,
            theta_max=theta_max,
            gain_threshold=gain_threshold,
        )

        if "error" in result:
            return f"Error: {result['error']}"

        unit = result.get("unit", "dB")
        output = f"Horizon Statistics for {name} @ {frequency} MHz\n"
        output += f"Theta Range: {theta_min}° - {theta_max}°\n"
        output += "=" * 50 + "\n\n"

        output += f"Max {unit}: {_fmt(result.get('max_gain_dB'))} {unit}\n"
        output += f"Min {unit}: {_fmt(result.get('min_gain_dB'))} {unit}\n"
        output += f"Avg {unit} (linear): {_fmt(result.get('avg_gain_dB'))} {unit}\n"
        output += f"MEG (sin-θ weighted): {_fmt(result.get('meg_dB'))} {unit}\n\n"

        output += f"Coverage (>{_fmt(result.get('max_gain_dB', 0) + gain_threshold)} {unit}): "
        output += f"{_fmt(result.get('coverage_pct'), '.1f')}%\n"
        output += f"Null Depth: {_fmt(result.get('null_depth_dB'), '.1f')} dB\n"

        null_loc = result.get("null_location", {})
        if null_loc:
            output += f"Null Location: θ={_fmt(null_loc.get('theta_deg'), '.0f')}°, "
            output += f"φ={_fmt(null_loc.get('phi_deg'), '.0f')}°\n"

        return output

    except Exception as e:
        return f"Error calculating horizon statistics: {str(e)}"


# ---- MCP tool registration (wraps the standalone functions above) ----

def register_analysis_tools(mcp):
    """Register analysis tools with the MCP server."""

    mcp.tool()(list_frequencies)
    mcp.tool()(analyze_pattern)
    mcp.tool()(get_gain_statistics)
    mcp.tool()(compare_polarizations)
    mcp.tool()(get_all_analysis)
    mcp.tool()(extrapolate_to_frequency)
    mcp.tool()(get_horizon_statistics)
