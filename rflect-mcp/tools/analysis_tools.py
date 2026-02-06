"""
Analysis Tools for RFlect MCP Server

Provides pattern analysis, gain statistics, and polarization comparison.
Uses plot_antenna.ai_analysis.AntennaAnalyzer for core functionality.
"""

from typing import Optional, List, Dict, Any
import json

# Import from RFlect
from plot_antenna.ai_analysis import AntennaAnalyzer
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


# ---- MCP tool registration (wraps the standalone functions above) ----

def register_analysis_tools(mcp):
    """Register analysis tools with the MCP server."""

    mcp.tool()(list_frequencies)
    mcp.tool()(analyze_pattern)
    mcp.tool()(get_gain_statistics)
    mcp.tool()(compare_polarizations)
    mcp.tool()(get_all_analysis)
