"""
AI Analysis Module for RFlect

This module provides reusable antenna analysis functions for AI-powered features.
Designed to be GUI-independent and suitable for both GUI and future MCP server use.

Status: EXPERIMENTAL / NOT PRODUCTION-READY
- Core analysis logic is functional
- System prompts need refinement for antenna domain knowledge
- Report templating system is ~90% complete
- Needs more testing with real-world antenna data

TODO (Future v4.1+):
- Add batch frequency analysis across all loaded frequencies
- Improve pattern classification (omnidirectional, directional, sectoral)
- Add antenna benchmarking (compare to typical dipole, patch, horn)
- Add design recommendations based on pattern analysis
- Complete report templating system
- Add MCP server integration
"""

import json
import numpy as np
from typing import Dict, List, Optional, Tuple, Any


class AntennaAnalyzer:
    """
    Antenna measurement analysis for AI-powered features.

    This class provides pure analysis functions that operate on measurement data
    without depending on GUI state. Designed for reuse in both GUI and API contexts.

    Args:
        measurement_data: Dictionary containing antenna measurement data
        scan_type: Type of scan ("passive", "active", or "vna")
        frequencies: List of measured frequencies in MHz

    Example:
        >>> data = {
        ...     'phi': phi_angles,
        ...     'theta': theta_angles,
        ...     'total_gain': gain_array,
        ...     'h_gain': h_pol_array,
        ...     'v_gain': v_pol_array
        ... }
        >>> analyzer = AntennaAnalyzer(data, scan_type='passive', frequencies=[2400, 2450])
        >>> stats = analyzer.get_gain_statistics(frequency=2400)
    """

    def __init__(
        self,
        measurement_data: Dict[str, Any],
        scan_type: str,
        frequencies: List[float]
    ):
        """
        Initialize analyzer with measurement data.

        Args:
            measurement_data: Dict with keys like 'phi', 'theta', 'total_gain', 'h_gain', 'v_gain'
            scan_type: One of 'passive', 'active', 'vna'
            frequencies: List of measured frequencies in MHz
        """
        self.data = measurement_data
        self.scan_type = scan_type
        self.frequencies = frequencies

    def get_gain_statistics(self, frequency: Optional[float] = None) -> Dict[str, Any]:
        """
        Calculate gain/power statistics for a given frequency.

        Args:
            frequency: Target frequency in MHz. If None, uses first available frequency.

        Returns:
            Dictionary containing min, max, avg, and std dev of gain/power

        Example:
            >>> stats = analyzer.get_gain_statistics(2400.0)
            >>> print(f"Max gain: {stats['max_gain_dBi']:.2f} dBi")
        """
        if frequency is None and self.frequencies:
            frequency = self.frequencies[0]

        # Find closest frequency
        freq_array = np.array(self.frequencies)
        freq_idx = np.argmin(np.abs(freq_array - frequency))
        actual_freq = self.frequencies[freq_idx]

        stats = {
            "frequency_requested": frequency,
            "frequency_actual": actual_freq,
            "scan_type": self.scan_type
        }

        # Passive scan: Gain in dBi
        if self.scan_type == "passive":
            if 'total_gain' in self.data and self.data['total_gain'] is not None:
                # Handle 2D array indexed by frequency
                if self.data['total_gain'].ndim == 2:
                    gain_data = self.data['total_gain'][:, freq_idx]
                else:
                    gain_data = self.data['total_gain']

                stats.update({
                    "max_gain_dBi": float(np.max(gain_data)),
                    "min_gain_dBi": float(np.min(gain_data)),
                    "avg_gain_dBi": float(np.mean(gain_data)),
                    "std_dev_dB": float(np.std(gain_data))
                })

                # Add polarization-specific stats if available
                if 'h_gain' in self.data and self.data['h_gain'] is not None:
                    h_data = self.data['h_gain'][:, freq_idx] if self.data['h_gain'].ndim == 2 else self.data['h_gain']
                    stats["max_hpol_gain_dBi"] = float(np.max(h_data))

                if 'v_gain' in self.data and self.data['v_gain'] is not None:
                    v_data = self.data['v_gain'][:, freq_idx] if self.data['v_gain'].ndim == 2 else self.data['v_gain']
                    stats["max_vpol_gain_dBi"] = float(np.max(v_data))

        # Active scan: Power in dBm
        elif self.scan_type == "active":
            if 'total_power' in self.data and self.data['total_power'] is not None:
                power_data = self.data['total_power']
                stats.update({
                    "max_power_dBm": float(np.max(power_data)),
                    "min_power_dBm": float(np.min(power_data)),
                    "avg_power_dBm": float(np.mean(power_data)),
                    "std_dev_dB": float(np.std(power_data))
                })

                # Add TRP if available
                if 'TRP_dBm' in self.data:
                    stats["TRP_dBm"] = float(self.data['TRP_dBm'])
                if 'h_TRP_dBm' in self.data:
                    stats["h_TRP_dBm"] = float(self.data['h_TRP_dBm'])
                if 'v_TRP_dBm' in self.data:
                    stats["v_TRP_dBm"] = float(self.data['v_TRP_dBm'])

        return stats

    def analyze_pattern(self, frequency: Optional[float] = None) -> Dict[str, Any]:
        """
        Analyze radiation pattern characteristics.

        Analyzes:
        - Null locations and depths
        - Beamwidth (HPBW - Half Power Beamwidth)
        - Front-to-back ratio
        - Pattern type classification (omnidirectional, directional, etc.)

        Args:
            frequency: Target frequency in MHz

        Returns:
            Dictionary containing pattern analysis results

        TODO:
        - Improve null detection algorithm
        - Add pattern symmetry analysis
        - Add sidelobe level detection
        - Classify antenna type from pattern
        """
        if frequency is None and self.frequencies:
            frequency = self.frequencies[0]

        # Find closest frequency
        freq_array = np.array(self.frequencies)
        freq_idx = np.argmin(np.abs(freq_array - frequency))
        actual_freq = self.frequencies[freq_idx]

        analysis = {
            "frequency": actual_freq,
            "scan_type": self.scan_type
        }

        if self.scan_type == "passive" and 'total_gain' in self.data:
            gain_data = self.data['total_gain']
            if gain_data.ndim == 2:
                gain_data = gain_data[:, freq_idx]

            # Find peak gain location
            max_idx = np.argmax(gain_data)
            analysis["peak_gain_dBi"] = float(np.max(gain_data))

            # Find nulls (gain below -10 dB from peak)
            peak_gain = np.max(gain_data)
            null_threshold = peak_gain - 10
            null_indices = np.where(gain_data < null_threshold)[0]
            analysis["num_nulls"] = len(null_indices)
            analysis["deepest_null_dB"] = float(np.min(gain_data)) if len(gain_data) > 0 else None

            # TODO: Calculate HPBW (Half Power Beamwidth)
            # Requires finding -3dB points from peak in E-plane and H-plane

            # TODO: Calculate front-to-back ratio
            # Requires identifying front and back directions

            # Pattern type classification (basic)
            gain_range = np.max(gain_data) - np.min(gain_data)
            if gain_range < 6:
                analysis["pattern_type"] = "omnidirectional"
            elif gain_range < 12:
                analysis["pattern_type"] = "sectoral"
            else:
                analysis["pattern_type"] = "directional"

        return analysis

    def compare_polarizations(self, frequency: Optional[float] = None) -> Dict[str, Any]:
        """
        Compare HPOL and VPOL characteristics.

        Calculates:
        - Cross-polarization discrimination (XPD)
        - Polarization balance
        - Axial ratio (for circular polarization)

        Args:
            frequency: Target frequency in MHz

        Returns:
            Dictionary containing polarization comparison results

        TODO:
        - Add axial ratio calculation
        - Add tilt angle calculation
        - Add polarization sense (LHCP vs RHCP)
        """
        if frequency is None and self.frequencies:
            frequency = self.frequencies[0]

        # Find closest frequency
        freq_array = np.array(self.frequencies)
        freq_idx = np.argmin(np.abs(freq_array - frequency))
        actual_freq = self.frequencies[freq_idx]

        comparison = {
            "frequency": actual_freq,
            "scan_type": self.scan_type
        }

        if self.scan_type == "passive":
            if 'h_gain' in self.data and 'v_gain' in self.data:
                h_data = self.data['h_gain']
                v_data = self.data['v_gain']

                if h_data.ndim == 2:
                    h_data = h_data[:, freq_idx]
                    v_data = v_data[:, freq_idx]

                # Peak gains
                comparison["max_hpol_gain_dBi"] = float(np.max(h_data))
                comparison["max_vpol_gain_dBi"] = float(np.max(v_data))

                # Cross-pol discrimination (XPD)
                # XPD = co-pol / cross-pol ratio in dB
                xpd = h_data - v_data  # In dB, this is ratio
                comparison["avg_xpd_dB"] = float(np.mean(np.abs(xpd)))
                comparison["max_xpd_dB"] = float(np.max(np.abs(xpd)))

                # Polarization balance
                balance = comparison["max_hpol_gain_dBi"] - comparison["max_vpol_gain_dBi"]
                comparison["polarization_balance_dB"] = float(balance)

                if abs(balance) < 3:
                    comparison["polarization_note"] = "Well-balanced polarization"
                elif abs(balance) < 6:
                    comparison["polarization_note"] = "Moderate polarization imbalance"
                else:
                    comparison["polarization_note"] = "Significant polarization imbalance"

        return comparison

    def analyze_all_frequencies(self) -> Dict[str, Any]:
        """
        Analyze gain trends across all measured frequencies.

        Returns:
            Dictionary containing frequency-dependent analysis

        TODO: Implement this function for v4.1
        - Frequency response (gain vs frequency)
        - 3dB bandwidth calculation
        - Resonance frequency detection
        - Frequency stability analysis
        """
        analysis = {
            "frequencies_MHz": self.frequencies,
            "num_frequencies": len(self.frequencies)
        }

        # Placeholder for future implementation
        analysis["status"] = "Not implemented - planned for v4.1"
        analysis["TODO"] = [
            "Calculate gain vs frequency trend",
            "Find 3dB bandwidth",
            "Detect resonance frequency",
            "Analyze frequency stability"
        ]

        return analysis


# Utility functions for AI prompts

def get_antenna_domain_knowledge() -> str:
    """
    Get antenna engineering domain knowledge for AI system prompts.

    Returns:
        String containing antenna benchmarks and design guidelines

    TODO: Expand with more antenna types and applications
    """
    knowledge = """
Antenna Engineering Reference:

Common Antenna Benchmarks:
- Isotropic radiator: 0 dBi (theoretical reference)
- Short dipole: 1.76 dBi
- Half-wave dipole: 2.15 dBi
- Patch antenna: 6-8 dBi (typical)
- Horn antenna: 10-25 dBi
- Parabolic reflector: 20-50+ dBi

Gain Classifications:
- Low gain: <6 dBi (omnidirectional, wide coverage)
- Medium gain: 6-12 dBi (sectoral coverage)
- High gain: >12 dBi (directional, narrow beam)

Efficiency Guidelines:
- Excellent: >90% (-0.5 dB)
- Good: 70-90% (-1.5 to -0.5 dB)
- Fair: 50-70% (-3 to -1.5 dB)
- Poor: <50% (>-3 dB)

Cross-Polarization Discrimination (XPD):
- Excellent: >30 dB
- Good: 20-30 dB
- Acceptable: 10-20 dB
- Poor: <10 dB

VSWR Guidelines:
- Excellent: <1.5:1 (return loss >14 dB)
- Good: 1.5-2.0:1 (return loss 14-10 dB)
- Acceptable: 2.0-3.0:1 (return loss 10-6 dB)
- Poor: >3.0:1 (return loss <6 dB)
"""
    return knowledge


def create_analysis_system_prompt(scan_type: str) -> str:
    """
    Create system prompt for AI analysis with antenna domain knowledge.

    Args:
        scan_type: Type of scan being analyzed

    Returns:
        System prompt string for AI

    TODO: Refine prompts based on user feedback
    """
    base_prompt = f"""You are an RF antenna engineering expert analyzing {scan_type} antenna measurements.

{get_antenna_domain_knowledge()}

When analyzing antenna data:
1. Compare measurements to typical benchmarks
2. Identify strengths and potential issues
3. Suggest improvements if patterns show suboptimal performance
4. Use precise technical terminology
5. Be concise but thorough

Focus on practical insights that help engineers improve antenna designs.
"""
    return base_prompt


# Report templating status
REPORT_TEMPLATE_STATUS = {
    "status": "~90% complete",
    "working_features": [
        "Executive summary generation",
        "Gain statistics reporting",
        "Pattern classification",
        "Basic recommendations"
    ],
    "incomplete_features": [
        "Custom branding integration",
        "Multi-frequency comparison tables",
        "Automated figure captioning",
        "Compliance checklist (FCC, CE, etc.)"
    ],
    "known_issues": [
        "Some AI recommendations too generic",
        "Template formatting inconsistent across sections",
        "Missing integration with plotting.py for automated figure insertion"
    ]
}
