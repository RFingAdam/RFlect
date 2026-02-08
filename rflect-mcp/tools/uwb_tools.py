"""
UWB Analysis Tools for RFlect MCP Server

Provides tools for System Fidelity Factor, UWB channel analysis,
and impedance bandwidth computation.
"""

import os
import json
import numpy as np

from plot_antenna.uwb_analysis import (
    calculate_sff,
    build_complex_s21_from_s2vna,
    compute_group_delay_from_s21,
    extract_transfer_function,
    compute_impulse_response,
    analyze_return_loss,
    calculate_sff_vs_angle,
    parse_touchstone,
)
from plot_antenna.file_utils import parse_2port_data, parse_touchstone_to_dataframe


def register_uwb_tools(mcp):
    """Register UWB analysis tools with the MCP server."""

    @mcp.tool()
    def calculate_sff_from_files(
        file_paths: list[str],
        pulse_type: str = "gaussian_monocycle",
        min_freq_ghz: float | None = None,
        max_freq_ghz: float | None = None,
    ) -> str:
        """
        Calculate System Fidelity Factor (SFF) from 2-port S-parameter files.

        Supports S2VNA CSV files (with S21(dB) + S21(s) group delay columns)
        and Touchstone .s2p files. Each file should have an angle in the
        filename (e.g., 'GroupDelay_45deg.csv').

        Args:
            file_paths: List of file paths to process.
            pulse_type: Pulse type — 'gaussian_monocycle', 'modulated_gaussian',
                        or '5th_derivative_gaussian'.
            min_freq_ghz: Optional minimum frequency filter in GHz.
            max_freq_ghz: Optional maximum frequency filter in GHz.

        Returns:
            JSON string with SFF results per angle and mean SFF.
        """
        import re

        angle_data = []
        pattern = re.compile(r"(\d+)(deg|DEG)", re.IGNORECASE)

        for fp in file_paths:
            if not os.path.exists(fp):
                return json.dumps({"error": f"File not found: {fp}"})

            filename = os.path.basename(fp)
            match = pattern.search(filename)
            angle = float(match.group(1)) if match else len(angle_data) * 45.0

            # Parse file
            if fp.lower().endswith('.s2p'):
                df = parse_touchstone_to_dataframe(fp)
            else:
                df = parse_2port_data(fp)

            freq = df["! Stimulus(Hz)"].values
            has_s21_dB = "S21(dB)" in df.columns or "S12(dB)" in df.columns
            has_gd = "S21(s)" in df.columns or "S12(s)" in df.columns

            if not has_s21_dB:
                continue

            s21_dB = df[("S21(dB)" if "S21(dB)" in df.columns else "S12(dB)")].values

            # Apply frequency filter
            if min_freq_ghz is not None and max_freq_ghz is not None:
                mask = (freq >= min_freq_ghz * 1e9) & (freq <= max_freq_ghz * 1e9)
                freq = freq[mask]
                s21_dB = s21_dB[mask]

            if len(freq) < 10:
                continue

            if has_gd:
                gd = df[("S21(s)" if "S21(s)" in df.columns else "S12(s)")].values
                if min_freq_ghz is not None and max_freq_ghz is not None:
                    gd = gd[mask]
                s21_complex = build_complex_s21_from_s2vna(freq, s21_dB, gd)
            else:
                # Magnitude-only fallback
                s21_complex = 10 ** (s21_dB / 20.0) + 0j

            angle_data.append({
                'angle_deg': angle,
                'freq_hz': freq,
                's21_complex': s21_complex,
            })

        if not angle_data:
            return json.dumps({"error": "No valid data found in provided files."})

        result = calculate_sff_vs_angle(angle_data, pulse_type=pulse_type)

        return json.dumps({
            "angles_deg": result['angles'],
            "sff_values": result['sff_values'],
            "qualities": result['qualities'],
            "mean_sff": result['mean_sff'],
            "pulse_type": pulse_type,
            "num_files": len(angle_data),
        }, indent=2)

    @mcp.tool()
    def analyze_uwb_channel(
        file_path: str,
        distance_m: float = 1.0,
        pulse_type: str = "gaussian_monocycle",
    ) -> str:
        """
        Full UWB channel analysis from a single S-parameter file.

        Computes SFF, group delay, transfer function, and impulse response.

        Args:
            file_path: Path to S2VNA CSV or Touchstone .s2p file.
            distance_m: Measurement distance in meters.
            pulse_type: Pulse type for SFF calculation.

        Returns:
            JSON string with comprehensive channel analysis.
        """
        if not os.path.exists(file_path):
            return json.dumps({"error": f"File not found: {file_path}"})

        if file_path.lower().endswith('.s2p'):
            ts = parse_touchstone(file_path)
            freq = ts['freq_hz']
            s21_complex = ts['s21']
            s11 = ts['s11']
            s11_dB = 20.0 * np.log10(np.maximum(np.abs(s11), 1e-30))
        else:
            df = parse_2port_data(file_path)
            freq = df["! Stimulus(Hz)"].values

            has_s21_dB = "S21(dB)" in df.columns or "S12(dB)" in df.columns
            has_gd = "S21(s)" in df.columns or "S12(s)" in df.columns

            if not has_s21_dB:
                return json.dumps({"error": "File missing S21(dB) column."})

            s21_dB_arr = df[("S21(dB)" if "S21(dB)" in df.columns else "S12(dB)")].values

            if has_gd:
                gd = df[("S21(s)" if "S21(s)" in df.columns else "S12(s)")].values
                s21_complex = build_complex_s21_from_s2vna(freq, s21_dB_arr, gd)
            else:
                s21_complex = 10 ** (s21_dB_arr / 20.0) + 0j

            s11_dB = df["S11(dB)"].values if "S11(dB)" in df.columns else None

        analysis = {}

        # SFF
        sff_result = calculate_sff(freq, s21_complex, pulse_type=pulse_type)
        analysis["sff"] = {
            "value": sff_result['sff'],
            "quality": sff_result['quality'],
            "peak_delay_ns": sff_result['peak_delay_s'] * 1e9,
        }

        # Group delay
        gd_result = compute_group_delay_from_s21(freq, s21_complex)
        analysis["group_delay"] = {
            "mean_ns": float(np.mean(gd_result['group_delay_s'])) * 1e9,
            "variation_ps": gd_result['variation_s'] * 1e12,
            "distance_error_cm": gd_result['distance_error_m'] * 100,
        }

        # Transfer function
        tf_result = extract_transfer_function(freq, s21_complex, distance_m=distance_m)
        analysis["transfer_function"] = {
            "mag_range_dB": [float(np.min(tf_result['H_mag_dB'])),
                             float(np.max(tf_result['H_mag_dB']))],
            "flatness_dB": float(np.ptp(tf_result['H_mag_dB'])),
        }

        # Impulse response
        ir_result = compute_impulse_response(freq, tf_result['H_complex'])
        analysis["impulse_response"] = {
            "pulse_width_ps": ir_result['pulse_width_s'] * 1e12,
            "ringing_dB": ir_result['ringing_dB'],
        }

        # S11/VSWR (if available)
        if s11_dB is not None:
            rl_result = analyze_return_loss(freq, s11_dB)
            analysis["return_loss"] = {
                "min_s11_dB": rl_result['min_s11_dB'],
                "bandwidth_mhz": rl_result['bandwidth_hz'] / 1e6,
                "band_start_ghz": rl_result['band_start_hz'] / 1e9,
                "band_stop_ghz": rl_result['band_stop_hz'] / 1e9,
                "fractional_bandwidth_pct": rl_result['fractional_bandwidth'] * 100,
            }

        analysis["frequency_range_ghz"] = [float(freq[0] / 1e9), float(freq[-1] / 1e9)]
        analysis["num_points"] = len(freq)

        return json.dumps(analysis, indent=2)

    @mcp.tool()
    def get_impedance_bandwidth(
        file_path: str,
        threshold_dB: float = -10.0,
    ) -> str:
        """
        Compute impedance bandwidth from S11 data.

        Args:
            file_path: Path to S2VNA CSV or Touchstone .s2p file.
            threshold_dB: S11 threshold for bandwidth (default: -10 dB).

        Returns:
            JSON string with VSWR and bandwidth information.
        """
        if not os.path.exists(file_path):
            return json.dumps({"error": f"File not found: {file_path}"})

        if file_path.lower().endswith('.s2p'):
            ts = parse_touchstone(file_path)
            freq = ts['freq_hz']
            s11_dB = 20.0 * np.log10(np.maximum(np.abs(ts['s11']), 1e-30))
        else:
            df = parse_2port_data(file_path)
            freq = df["! Stimulus(Hz)"].values
            if "S11(dB)" not in df.columns:
                return json.dumps({"error": "File missing S11(dB) column."})
            s11_dB = df["S11(dB)"].values

        result = analyze_return_loss(freq, s11_dB, threshold_dB=threshold_dB)

        return json.dumps({
            "min_s11_dB": result['min_s11_dB'],
            "bandwidth_mhz": result['bandwidth_hz'] / 1e6,
            "band_start_ghz": result['band_start_hz'] / 1e9,
            "band_stop_ghz": result['band_stop_hz'] / 1e9,
            "fractional_bandwidth_pct": result['fractional_bandwidth'] * 100,
            "threshold_dB": threshold_dB,
            "frequency_range_ghz": [float(freq[0] / 1e9), float(freq[-1] / 1e9)],
        }, indent=2)
