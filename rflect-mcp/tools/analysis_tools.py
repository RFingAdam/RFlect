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


def register_analysis_tools(mcp):
    """Register analysis tools with the MCP server."""

    @mcp.tool()
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

    @mcp.tool()
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
            Pattern analysis including:
            - Pattern type (directional, omnidirectional, etc.)
            - Half-power beamwidth (HPBW)
            - Front-to-back ratio
            - Null locations
            - Sidelobe levels
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
            output += f"HPBW (E-plane): {result.get('hpbw_e_plane', 'N/A')}°\n"
            output += f"HPBW (H-plane): {result.get('hpbw_h_plane', 'N/A')}°\n"
            output += f"Front-to-Back Ratio: {result.get('front_to_back_dB', 'N/A')} dB\n"
            output += f"Main Beam Direction: θ={result.get('main_beam_theta', 'N/A')}°, φ={result.get('main_beam_phi', 'N/A')}°\n"

            nulls = result.get('nulls', [])
            if nulls:
                output += f"\nNull Locations: {len(nulls)} found\n"
                for i, null in enumerate(nulls[:5]):  # Show first 5
                    output += f"  Null {i+1}: θ={null.get('theta', 'N/A')}°, φ={null.get('phi', 'N/A')}°\n"

            return output

        except Exception as e:
            return f"Error analyzing pattern: {str(e)}"

    @mcp.tool()
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

            output += f"Maximum Gain: {result.get('max_gain_dBi', 'N/A'):.2f} dBi\n"
            output += f"  at θ={result.get('max_gain_theta', 'N/A')}°, φ={result.get('max_gain_phi', 'N/A')}°\n"
            output += f"Minimum Gain: {result.get('min_gain_dBi', 'N/A'):.2f} dBi\n"
            output += f"Average Gain: {result.get('avg_gain_dBi', 'N/A'):.2f} dBi\n"
            output += f"Gain Range: {result.get('gain_range_dB', 'N/A'):.2f} dB\n"

            return output

        except Exception as e:
            return f"Error getting gain statistics: {str(e)}"

    @mcp.tool()
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

            output += f"Dominant Polarization: {result.get('dominant_pol', 'Unknown')}\n"
            output += f"Average XPD: {result.get('avg_xpd_dB', 'N/A'):.1f} dB\n"
            output += f"Max XPD: {result.get('max_xpd_dB', 'N/A'):.1f} dB\n"
            output += f"Min XPD: {result.get('min_xpd_dB', 'N/A'):.1f} dB\n\n"

            output += f"HPOL Max Gain: {result.get('hpol_max_dBi', 'N/A'):.2f} dBi\n"
            output += f"VPOL Max Gain: {result.get('vpol_max_dBi', 'N/A'):.2f} dBi\n"

            pol_balance = result.get('polarization_balance', 'N/A')
            output += f"\nPolarization Balance: {pol_balance}\n"

            return output

        except Exception as e:
            return f"Error comparing polarizations: {str(e)}"

    @mcp.tool()
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
