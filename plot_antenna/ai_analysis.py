"""
AI Analysis Module for RFlect

This module provides reusable antenna analysis functions for AI-powered features.
Designed to be GUI-independent and suitable for both GUI and MCP server use.

Status: EXPERIMENTAL
- Core analysis logic is functional
- Pattern analysis includes HPBW and F/B ratio calculations
- Batch frequency analysis implemented
- MCP server integration complete
"""

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

    def __init__(self, measurement_data: Dict[str, Any], scan_type: str, frequencies: List[float]):
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
            "scan_type": self.scan_type,
        }

        # Passive scan: Gain in dBi
        if self.scan_type == "passive":
            if "total_gain" in self.data and self.data["total_gain"] is not None:
                # Handle 2D array indexed by frequency
                if self.data["total_gain"].ndim == 2:
                    gain_data = self.data["total_gain"][:, freq_idx]
                else:
                    gain_data = self.data["total_gain"]

                # Spherical average gain: weight by sin(θ) for correct solid-angle
                # integration on a sphere. Points near poles represent less area.
                # Only applies when theta array matches gain data length.
                gain_linear = 10.0 ** (gain_data / 10.0)
                theta_data = self.data.get("theta")
                if theta_data is not None:
                    theta_col = np.asarray(theta_data)
                    if theta_col.ndim == 2:
                        theta_col = theta_col[:, freq_idx]
                    if len(theta_col) == len(gain_data):
                        sin_weights = np.sin(np.deg2rad(theta_col))
                        weight_sum = float(np.sum(sin_weights))
                        if weight_sum > 0:
                            avg_gain_linear = float(np.sum(gain_linear * sin_weights) / weight_sum)
                        else:
                            avg_gain_linear = float(np.mean(gain_linear))
                    else:
                        avg_gain_linear = float(np.mean(gain_linear))
                else:
                    avg_gain_linear = float(np.mean(gain_linear))
                avg_gain_dBi = (
                    10.0 * np.log10(avg_gain_linear)
                    if avg_gain_linear > 0
                    else float(np.mean(gain_data))
                )

                stats.update(
                    {
                        "max_gain_dBi": float(np.max(gain_data)),
                        "min_gain_dBi": float(np.min(gain_data)),
                        "avg_gain_dBi": avg_gain_dBi,
                        "std_dev_dB": float(np.std(gain_data)),
                    }
                )

                # Add polarization-specific stats if available
                if "h_gain" in self.data and self.data["h_gain"] is not None:
                    h_data = (
                        self.data["h_gain"][:, freq_idx]
                        if self.data["h_gain"].ndim == 2
                        else self.data["h_gain"]
                    )
                    stats["max_hpol_gain_dBi"] = float(np.max(h_data))

                if "v_gain" in self.data and self.data["v_gain"] is not None:
                    v_data = (
                        self.data["v_gain"][:, freq_idx]
                        if self.data["v_gain"].ndim == 2
                        else self.data["v_gain"]
                    )
                    stats["max_vpol_gain_dBi"] = float(np.max(v_data))

        # Active scan: Power in dBm
        elif self.scan_type == "active":
            if "total_power" in self.data and self.data["total_power"] is not None:
                power_data = self.data["total_power"]
                # Average power must be computed in the linear domain (mW)
                power_linear = 10.0 ** (power_data / 10.0)
                avg_power_linear = float(np.mean(power_linear))
                avg_power_dBm = (
                    10.0 * np.log10(avg_power_linear)
                    if avg_power_linear > 0
                    else float(np.mean(power_data))
                )

                stats.update(
                    {
                        "max_power_dBm": float(np.max(power_data)),
                        "min_power_dBm": float(np.min(power_data)),
                        "avg_power_dBm": avg_power_dBm,
                        "std_dev_dB": float(np.std(power_data)),
                    }
                )

                # Add TRP if available
                if "TRP_dBm" in self.data:
                    stats["TRP_dBm"] = float(self.data["TRP_dBm"])
                if "h_TRP_dBm" in self.data:
                    stats["h_TRP_dBm"] = float(self.data["h_TRP_dBm"])
                if "v_TRP_dBm" in self.data:
                    stats["v_TRP_dBm"] = float(self.data["v_TRP_dBm"])

        return stats

    def _get_gain_grid(
        self, gain_data_1d: np.ndarray
    ) -> Optional[Tuple[np.ndarray, np.ndarray, np.ndarray]]:
        """
        Reshape 1D gain data into a 2D grid using phi/theta angle arrays.

        Returns:
            Tuple of (gain_2d, unique_theta, unique_phi) or None if insufficient data.
            gain_2d shape is (num_theta, num_phi).
        """
        phi = self.data.get("phi")
        theta = self.data.get("theta")
        if phi is None or theta is None:
            return None

        phi = np.asarray(phi)
        theta = np.asarray(theta)

        unique_theta = np.unique(theta)
        unique_phi = np.unique(phi)
        n_theta = len(unique_theta)
        n_phi = len(unique_phi)

        if n_theta * n_phi != len(gain_data_1d):
            # Data does not fit expected grid shape — cannot form 2D grid
            return None

        try:
            gain_2d = gain_data_1d.reshape((n_theta, n_phi))
            return gain_2d, unique_theta, unique_phi
        except ValueError:
            return None

    def _calculate_hpbw(self, cut_angles: np.ndarray, cut_gain: np.ndarray) -> Optional[float]:
        """
        Calculate Half-Power Beamwidth from a 1D gain cut.

        Walks outward from the peak to find the contiguous -3dB region,
        correctly handling wrapping around 0/360° boundaries.

        Args:
            cut_angles: Angle values in degrees along the cut
            cut_gain: Gain values in dBi along the cut

        Returns:
            HPBW in degrees, or None if can't determine.
        """
        if len(cut_gain) < 3:
            return None

        peak_gain = np.max(cut_gain)
        threshold = peak_gain - 3.0  # -3 dB from peak

        peak_idx = np.argmax(cut_gain)
        n = len(cut_gain)

        # Walk left from peak to find first crossing below threshold
        left_bound = peak_idx
        for i in range(1, n):
            idx = (peak_idx - i) % n
            if cut_gain[idx] < threshold:
                left_bound = (peak_idx - i + 1) % n
                break
        else:
            # Entire cut is above threshold (omnidirectional)
            return float(cut_angles[-1] - cut_angles[0])

        # Walk right from peak to find first crossing below threshold
        right_bound = peak_idx
        for i in range(1, n):
            idx = (peak_idx + i) % n
            if cut_gain[idx] < threshold:
                right_bound = (peak_idx + i - 1) % n
                break

        # Interpolate left crossing
        left_below_idx = (left_bound - 1) % n
        g_above = cut_gain[left_bound]
        g_below = cut_gain[left_below_idx]
        if g_above != g_below and left_bound != peak_idx:
            frac = (threshold - g_below) / (g_above - g_below)
            left_angle = cut_angles[left_below_idx] + frac * (
                cut_angles[left_bound] - cut_angles[left_below_idx]
            )
        else:
            left_angle = cut_angles[left_bound]

        # Interpolate right crossing
        right_above_idx = (right_bound + 1) % n
        g_above = cut_gain[right_bound]
        g_below = cut_gain[right_above_idx]
        if g_above != g_below and right_bound != peak_idx:
            frac = (threshold - g_below) / (g_above - g_below)
            right_angle = cut_angles[right_above_idx] - frac * (
                cut_angles[right_above_idx] - cut_angles[right_bound]
            )
        else:
            right_angle = cut_angles[right_bound]

        # Handle wrapping: if right_angle < left_angle, the beam wraps around
        angle_span = cut_angles[-1] - cut_angles[0]
        if angle_span > 0:
            step = angle_span / (n - 1) if n > 1 else 0
            full_range = angle_span + step  # e.g., 360 for 0..345 with 15° step
        else:
            full_range = 360.0

        hpbw = right_angle - left_angle
        if hpbw < 0:
            hpbw += full_range

        return float(hpbw) if hpbw > 0 else None

    def _detect_sidelobes(
        self, cut_angles: np.ndarray, cut_gain: np.ndarray, num_sidelobes: int = 3
    ) -> List[Dict[str, Any]]:
        """
        Detect sidelobes in a 1D gain cut.

        Args:
            cut_angles: Angle values in degrees along the cut
            cut_gain: Gain values in dBi along the cut
            num_sidelobes: Maximum number of sidelobes to return

        Returns:
            List of dicts with keys: angle_deg, gain_dBi, sll_dB (relative to main lobe)
        """
        if len(cut_gain) < 5:
            return []

        # Find local maxima: points higher than both neighbors
        local_max = []
        for i in range(1, len(cut_gain) - 1):
            if cut_gain[i] > cut_gain[i - 1] and cut_gain[i] > cut_gain[i + 1]:
                local_max.append(i)

        # Check endpoints
        if cut_gain[0] > cut_gain[1]:
            local_max.insert(0, 0)
        if cut_gain[-1] > cut_gain[-2]:
            local_max.append(len(cut_gain) - 1)

        if len(local_max) < 2:
            return []

        # Sort by gain value (descending)
        local_max.sort(key=lambda i: cut_gain[i], reverse=True)

        main_gain = float(cut_gain[local_max[0]])
        sidelobes = []
        for idx in local_max[1 : num_sidelobes + 1]:
            sll = float(cut_gain[idx]) - main_gain
            sidelobes.append(
                {
                    "angle_deg": float(cut_angles[idx]),
                    "gain_dBi": float(cut_gain[idx]),
                    "sll_dB": sll,
                }
            )

        return sidelobes

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
        """
        if frequency is None and self.frequencies:
            frequency = self.frequencies[0]

        # Find closest frequency
        freq_array = np.array(self.frequencies)
        freq_idx = np.argmin(np.abs(freq_array - frequency))
        actual_freq = self.frequencies[freq_idx]

        analysis = {"frequency": actual_freq, "scan_type": self.scan_type}

        if self.scan_type == "passive" and "total_gain" in self.data:
            gain_data = self.data["total_gain"]
            if gain_data.ndim == 2:
                gain_data = gain_data[:, freq_idx]

            # Peak gain
            analysis["peak_gain_dBi"] = float(np.max(gain_data))

            # Find nulls (gain below -10 dB from peak)
            peak_gain = np.max(gain_data)
            null_threshold = peak_gain - 10
            null_indices = np.where(gain_data < null_threshold)[0]
            analysis["num_nulls"] = len(null_indices)
            analysis["deepest_null_dB"] = float(np.min(gain_data)) if len(gain_data) > 0 else None

            # Pattern type classification
            gain_range = np.max(gain_data) - np.min(gain_data)
            if gain_range < 6:
                analysis["pattern_type"] = "omnidirectional"
            elif gain_range < 12:
                analysis["pattern_type"] = "sectoral"
            else:
                analysis["pattern_type"] = "directional"

            # HPBW and F/B ratio require 2D grid data
            grid = self._get_gain_grid(gain_data)
            if grid is not None:
                gain_2d, unique_theta, unique_phi = grid
                peak_2d_idx = np.unravel_index(np.argmax(gain_2d), gain_2d.shape)
                peak_theta_idx, peak_phi_idx = peak_2d_idx

                analysis["main_beam_theta"] = float(unique_theta[peak_theta_idx])
                analysis["main_beam_phi"] = float(unique_phi[peak_phi_idx])

                # HPBW E-plane: vary theta at peak phi
                e_plane_cut = gain_2d[:, peak_phi_idx]
                hpbw_e = self._calculate_hpbw(unique_theta, e_plane_cut)
                analysis["hpbw_e_plane"] = hpbw_e

                # HPBW H-plane: vary phi at peak theta
                h_plane_cut = gain_2d[peak_theta_idx, :]
                hpbw_h = self._calculate_hpbw(unique_phi, h_plane_cut)
                analysis["hpbw_h_plane"] = hpbw_h

                # Front-to-back ratio
                opposite_phi = (unique_phi[peak_phi_idx] + 180.0) % 360.0
                back_phi_idx = np.argmin(np.abs(unique_phi - opposite_phi))
                back_gain = gain_2d[peak_theta_idx, back_phi_idx]
                analysis["front_to_back_dB"] = float(peak_gain - back_gain)

                # Sidelobe detection on principal plane cuts
                e_sidelobes = self._detect_sidelobes(unique_theta, e_plane_cut)
                if e_sidelobes:
                    analysis["e_plane_sidelobes"] = e_sidelobes
                    analysis["first_sll_e_plane_dB"] = e_sidelobes[0]["sll_dB"]

                h_sidelobes = self._detect_sidelobes(unique_phi, h_plane_cut)
                if h_sidelobes:
                    analysis["h_plane_sidelobes"] = h_sidelobes
                    analysis["first_sll_h_plane_dB"] = h_sidelobes[0]["sll_dB"]

                # Directivity and efficiency estimation (Kraus approximation)
                # Kraus is only valid for directive patterns; when either HPBW
                # exceeds 180° the approximation breaks down and can yield η>100%.
                if (
                    hpbw_e is not None
                    and hpbw_h is not None
                    and hpbw_e > 0
                    and hpbw_h > 0
                    and hpbw_e <= 180
                    and hpbw_h <= 180
                ):
                    directivity_linear = 41253.0 / (hpbw_e * hpbw_h)
                    directivity_dBi = float(10.0 * np.log10(directivity_linear))
                    analysis["estimated_directivity_dBi"] = round(directivity_dBi, 2)

                    # Efficiency: η = G/D (in dB: G_dBi - D_dBi)
                    eta_dB = peak_gain - directivity_dBi
                    eta_linear = 10.0 ** (eta_dB / 10.0)
                    eta_pct = eta_linear * 100.0
                    if 0 < eta_pct <= 100:
                        analysis["estimated_efficiency_pct"] = round(eta_pct, 1)
                        analysis["estimated_efficiency_dB"] = round(float(eta_dB), 2)

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
        """
        if frequency is None and self.frequencies:
            frequency = self.frequencies[0]

        # Find closest frequency
        freq_array = np.array(self.frequencies)
        freq_idx = np.argmin(np.abs(freq_array - frequency))
        actual_freq = self.frequencies[freq_idx]

        comparison = {"frequency": actual_freq, "scan_type": self.scan_type}

        if self.scan_type == "passive":
            if "h_gain" in self.data and "v_gain" in self.data:
                h_data = self.data["h_gain"]
                v_data = self.data["v_gain"]

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

                # Proper XPD at co-pol peak direction
                h_peak_val = float(np.max(h_data))
                v_peak_val = float(np.max(v_data))
                if h_peak_val >= v_peak_val:
                    copol_peak_idx = int(np.argmax(h_data))
                    xpd_at_peak = float(h_data[copol_peak_idx] - v_data[copol_peak_idx])
                    comparison["dominant_pol"] = "H"
                else:
                    copol_peak_idx = int(np.argmax(v_data))
                    xpd_at_peak = float(v_data[copol_peak_idx] - h_data[copol_peak_idx])
                    comparison["dominant_pol"] = "V"
                comparison["xpd_at_copol_peak_dB"] = xpd_at_peak

                # Polarization balance
                balance = float(comparison["max_hpol_gain_dBi"]) - float(comparison["max_vpol_gain_dBi"])  # type: ignore[arg-type]
                comparison["polarization_balance_dB"] = balance

                if abs(balance) < 3:
                    comparison["polarization_note"] = "Well-balanced polarization"
                elif abs(balance) < 6:
                    comparison["polarization_note"] = "Moderate polarization imbalance"
                else:
                    comparison["polarization_note"] = "Significant polarization imbalance"

        return comparison

    def get_horizon_statistics(
        self,
        frequency: Optional[float] = None,
        theta_min: float = 60.0,
        theta_max: float = 120.0,
        gain_threshold: float = -3.0,
    ) -> Dict[str, Any]:
        """
        Calculate horizon-band statistics for maritime/on-water applications.

        Returns:
            Dictionary containing:
            - min_gain_dB, max_gain_dB, avg_gain_dB: Gain extremes and linear-domain average
            - coverage_pct: Percentage of horizon band above (peak + threshold)
            - meg_dB: Mean Effective Gain with sin(theta) weighting
            - null_depth_dB: Depth of deepest null relative to peak
            - null_location: (theta, phi) of the null
            - theta_range: (theta_min, theta_max) used
        """
        stats: Dict[str, Any] = {
            "theta_min": theta_min,
            "theta_max": theta_max,
            "gain_threshold_dB": gain_threshold,
        }

        if self.scan_type == "passive":
            gain_key = "total_gain"
            unit = "dBi"
        else:
            gain_key = "total_power"
            unit = "dBm"

        gain_data = self.data.get(gain_key)
        if gain_data is None:
            stats["error"] = f"No {gain_key} data available"
            return stats

        # Get the gain grid for the specified frequency
        if frequency is not None and self.frequencies:
            if frequency in self.frequencies:
                freq_idx = self.frequencies.index(frequency)
            else:
                stats["error"] = f"Frequency {frequency} not found"
                return stats
        else:
            freq_idx = 0

        grid_result = self._get_gain_grid(
            gain_data[:, freq_idx] if gain_data.ndim == 2 else gain_data
        )
        if grid_result is None:
            stats["error"] = "Cannot form 2D gain grid"
            return stats

        gain_2d, unique_theta, unique_phi = grid_result

        # Extract horizon band
        mask = (unique_theta >= theta_min) & (unique_theta <= theta_max)
        horizon_theta = unique_theta[mask]
        horizon_gain = gain_2d[mask, :]

        if horizon_gain.size == 0:
            stats["error"] = f"No data in theta range {theta_min}-{theta_max}"
            return stats

        stats["frequency_MHz"] = (
            frequency if frequency else (self.frequencies[0] if self.frequencies else None)
        )
        stats["unit"] = unit

        # Min / max / avg (linear-domain average)
        stats["max_gain_dB"] = float(np.max(horizon_gain))
        stats["min_gain_dB"] = float(np.min(horizon_gain))
        lin = 10.0 ** (horizon_gain / 10.0)
        stats["avg_gain_dB"] = float(10.0 * np.log10(np.mean(lin)))

        # Coverage: % of points above (peak + threshold)
        coverage_limit = stats["max_gain_dB"] + gain_threshold
        stats["coverage_pct"] = float(
            100.0 * np.sum(horizon_gain >= coverage_limit) / horizon_gain.size
        )

        # MEG: Mean Effective Gain with sin(theta) weighting
        theta_rad = np.deg2rad(horizon_theta)
        sin_weights = np.sin(theta_rad)
        weighted_lin = lin * sin_weights[:, np.newaxis]
        total_weight = np.sum(np.tile(sin_weights[:, np.newaxis], (1, horizon_gain.shape[1])))
        meg_lin = np.sum(weighted_lin) / total_weight if total_weight > 0 else 0
        stats["meg_dB"] = float(10.0 * np.log10(meg_lin)) if meg_lin > 0 else None

        # Null detection
        null_flat_idx = int(np.argmin(horizon_gain))
        null_theta_idx, null_phi_idx = np.unravel_index(null_flat_idx, horizon_gain.shape)
        stats["null_depth_dB"] = float(np.min(horizon_gain) - np.max(horizon_gain))
        stats["null_location"] = {
            "theta_deg": float(horizon_theta[null_theta_idx]),
            "phi_deg": float(unique_phi[null_phi_idx]),
        }

        return stats

    def analyze_all_frequencies(self) -> Dict[str, Any]:
        """
        Analyze gain trends across all measured frequencies.

        Returns:
            Dictionary containing:
            - peak_gain_per_freq: Peak gain at each frequency
            - resonance_frequency_MHz: Frequency with highest peak gain
            - bandwidth_3dB_MHz: 3dB bandwidth (if determinable)
            - gain_variation_dB: Total gain variation across band
            - avg_peak_gain_dBi: Average peak gain
            - gain_std_dev_dB: Standard deviation of peak gain
        """
        analysis = {
            "frequencies_MHz": self.frequencies,
            "num_frequencies": len(self.frequencies),
        }

        if len(self.frequencies) == 0:
            return analysis

        # Get peak gain at each frequency
        peak_gains = []
        for freq in self.frequencies:
            stats = self.get_gain_statistics(frequency=freq)
            gain = stats.get("max_gain_dBi", stats.get("max_power_dBm", None))
            peak_gains.append(gain)
        analysis["peak_gain_per_freq"] = peak_gains

        # Filter to valid (non-None) entries
        valid = [(g, f) for g, f in zip(peak_gains, self.frequencies) if g is not None]
        if not valid:
            return analysis

        valid_gains = [g for g, _ in valid]

        # Find resonance frequency (max peak gain)
        best_gain, best_freq = max(valid, key=lambda x: x[0])
        analysis["resonance_frequency_MHz"] = best_freq
        analysis["peak_gain_at_resonance_dBi"] = best_gain

        # 3dB bandwidth
        threshold = best_gain - 3
        in_band = [f for g, f in valid if g >= threshold]
        if len(in_band) >= 2:
            analysis["bandwidth_3dB_MHz"] = max(in_band) - min(in_band)
        else:
            analysis["bandwidth_3dB_MHz"] = None

        # Frequency stability metrics
        analysis["gain_variation_dB"] = max(valid_gains) - min(valid_gains)
        # Average peak gain in linear domain to avoid dB-averaging error
        gains_linear = [10.0 ** (g / 10.0) for g in valid_gains]
        avg_linear = float(np.mean(gains_linear))
        analysis["avg_peak_gain_dBi"] = (
            10.0 * np.log10(avg_linear) if avg_linear > 0 else float(np.mean(valid_gains))
        )
        analysis["gain_std_dev_dB"] = float(np.std(valid_gains))

        return analysis


# Utility functions for AI prompts


def get_antenna_domain_knowledge() -> str:
    """
    Get antenna engineering domain knowledge for AI system prompts.

    Returns:
        String containing antenna benchmarks and design guidelines
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
        "HPBW calculation",
        "Front-to-back ratio",
        "Batch frequency analysis",
        "Basic recommendations",
    ],
    "incomplete_features": [
        "Custom branding integration",
        "Multi-frequency comparison tables",
        "Automated figure captioning",
        "Compliance checklist (FCC, CE, etc.)",
    ],
    "known_issues": [
        "Some AI recommendations too generic",
        "Template formatting inconsistent across sections",
        "Missing integration with plotting.py for automated figure insertion",
    ],
}
