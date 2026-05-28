"""
VNA / S-parameter MCP tools for RFlect.

Wrappers over the deterministic S-parameter math in plot_antenna.uwb_analysis
(return-loss / impedance bandwidth, group delay). No LLM, no network. Each
tool returns a structured dict and never raises; failures populate `warnings`.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import numpy as np

from plot_antenna.uwb_analysis import analyze_return_loss, compute_group_delay_from_s21


def register_vna_tools(mcp):
    """Register VNA / S-parameter tools with the MCP server."""

    @mcp.tool()
    def analyze_s11(
        freq_hz: List[float],
        s11_db: List[float],
        threshold_db: float = -10.0,
        include_vswr_curve: bool = False,
    ) -> Dict[str, Any]:
        """
        Analyze an S11 sweep for impedance bandwidth, VSWR, and resonances.

        Args:
            freq_hz: Frequencies in Hz (1D, ascending).
            s11_db: S11 magnitude in dB at each frequency (negative values).
            threshold_db: Return-loss threshold defining the band (default -10 dB,
                i.e. VSWR ~2:1).
            include_vswr_curve: If True, also return the full per-point VSWR list
                (omitted by default to keep the response compact).

        Returns:
            Dict: min_s11_db, resonance_freq_hz (freq of min S11), vswr_min,
            band_start_hz, band_stop_hz, bandwidth_hz, center_freq_hz,
            fractional_bandwidth_pct, n_points, plus optional vswr_curve. Never
            raises; failures populate `warnings`.
        """
        result: Dict[str, Any] = {
            "min_s11_db": None,
            "resonance_freq_hz": None,
            "vswr_min": None,
            "band_start_hz": None,
            "band_stop_hz": None,
            "bandwidth_hz": None,
            "center_freq_hz": None,
            "fractional_bandwidth_pct": None,
            "n_points": 0,
            "warnings": [],
        }

        f = np.asarray(freq_hz, dtype=float)
        s = np.asarray(s11_db, dtype=float)
        if f.size == 0 or s.size == 0:
            result["warnings"].append("empty_input")
            return result
        if f.size != s.size:
            result["warnings"].append(f"length_mismatch: freq_hz={f.size} s11_db={s.size}")
            return result
        result["n_points"] = int(f.size)

        try:
            rl = analyze_return_loss(f, s, threshold_dB=threshold_db)
        except Exception as exc:
            result["warnings"].append(f"analyze_return_loss_failed: {exc}")
            return result

        vswr = np.asarray(rl.get("vswr", []), dtype=float)
        res_idx = int(np.argmin(s))
        bw = rl.get("bandwidth_hz", 0.0) or 0.0
        b0 = rl.get("band_start_hz", 0.0) or 0.0
        b1 = rl.get("band_stop_hz", 0.0) or 0.0
        center = (b0 + b1) / 2.0 if (b0 or b1) else None

        result.update(
            {
                "min_s11_db": float(np.min(s)),
                "resonance_freq_hz": float(f[res_idx]),
                "vswr_min": float(np.min(vswr)) if vswr.size else None,
                "band_start_hz": float(b0),
                "band_stop_hz": float(b1),
                "bandwidth_hz": float(bw),
                "center_freq_hz": float(center) if center else None,
                "fractional_bandwidth_pct": (float(rl.get("fractional_bandwidth", 0.0)) * 100.0),
            }
        )
        if bw <= 0:
            result["warnings"].append(
                f"no_band_below_threshold: nothing at or below {threshold_db} dB"
            )
        if include_vswr_curve and vswr.size:
            result["vswr_curve"] = [float(v) for v in vswr]
        return result

    @mcp.tool()
    def analyze_group_delay(
        freq_hz: List[float],
        phase_deg: List[float],
        band_start_hz: Optional[float] = None,
        band_stop_hz: Optional[float] = None,
    ) -> Dict[str, Any]:
        """
        Compute group delay and in-band flatness from a transmission-phase sweep.

        Group delay depends only on phase, so supply the measured S21 (or S12)
        phase in degrees; magnitude is not required. Optionally restrict the
        flatness statistics to an in-band window.

        Args:
            freq_hz: Frequencies in Hz (1D, ascending).
            phase_deg: S21 phase in degrees at each frequency (will be unwrapped).
            band_start_hz: Optional lower edge for in-band flatness stats.
            band_stop_hz: Optional upper edge for in-band flatness stats.

        Returns:
            Dict: mean_group_delay_ns, group_delay_variation_ns (peak-to-peak),
            distance_error_m, in_band_variation_ns (if a band is given), n_points.
            Never raises; failures populate `warnings`.
        """
        result: Dict[str, Any] = {
            "mean_group_delay_ns": None,
            "group_delay_variation_ns": None,
            "distance_error_m": None,
            "n_points": 0,
            "warnings": [],
        }
        f = np.asarray(freq_hz, dtype=float)
        ph = np.asarray(phase_deg, dtype=float)
        if f.size < 2 or ph.size < 2:
            result["warnings"].append("need_at_least_2_points")
            return result
        if f.size != ph.size:
            result["warnings"].append(f"length_mismatch: freq_hz={f.size} phase_deg={ph.size}")
            return result
        result["n_points"] = int(f.size)

        try:
            s21 = np.exp(1j * np.deg2rad(ph))  # unit magnitude; phase carries GD
            gd = compute_group_delay_from_s21(f, s21)
        except Exception as exc:
            result["warnings"].append(f"group_delay_failed: {exc}")
            return result

        gds = np.asarray(gd["group_delay_s"], dtype=float)
        result["mean_group_delay_ns"] = float(np.mean(gds) * 1e9)
        result["group_delay_variation_ns"] = float(gd["variation_s"] * 1e9)
        result["distance_error_m"] = float(gd["distance_error_m"])

        if band_start_hz is not None and band_stop_hz is not None:
            mask = (f >= band_start_hz) & (f <= band_stop_hz)
            if np.count_nonzero(mask) >= 2:
                result["in_band_variation_ns"] = float(np.ptp(gds[mask]) * 1e9)
            else:
                result["warnings"].append("band_window_has_too_few_points")
        return result
