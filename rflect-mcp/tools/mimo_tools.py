"""
MIMO / diversity MCP tools for RFlect.

Wrappers over the deterministic MIMO math in plot_antenna.calculations
(envelope correlation, Vaughan-Andersen diversity gain, 2x2 capacity). No
LLM, no network. Returns a structured dict; never raises.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import numpy as np

from plot_antenna.calculations import (
    diversity_gain,
    capacity_awgn,
    mimo_capacity_vs_snr,
)


def register_mimo_tools(mcp):
    """Register MIMO / diversity tools with the MCP server."""

    @mcp.tool()
    def analyze_mimo_diversity(
        ecc: float,
        snr_db: float = 15.0,
        snr_sweep_db: Optional[List[float]] = None,
        fading: str = "rayleigh",
        rician_k: float = 10.0,
    ) -> Dict[str, Any]:
        """
        Compute 2x2 MIMO diversity gain and capacity from an envelope correlation.

        Given the envelope correlation coefficient (ECC) between two antennas,
        report the Vaughan-Andersen diversity gain and the 2x2 ergodic capacity.
        ECC can be obtained from measured patterns (e.g. compute_ecc_from_farfield
        on a pair of far-field files, or derived from compare_polarizations).

        Rule of thumb: ECC < 0.5 is generally considered good isolation for
        diversity/MIMO; ECC < 0.3 is excellent.

        Args:
            ecc: Envelope correlation coefficient in [0, 1).
            snr_db: SNR (dB) at which to report the single-point capacity.
            snr_sweep_db: Optional list of SNRs (dB) for a capacity-vs-SNR curve.
            fading: 'rayleigh' | 'rician' for the Monte-Carlo capacity sweep.
            rician_k: Rician K-factor (linear) when fading='rician'.

        Returns:
            Dict: ecc, diversity_gain_db, capacity_bps_hz (at snr_db),
            isolation_rating ('excellent'|'good'|'marginal'|'poor'), plus an
            optional capacity_curve [{snr_db, capacity_bps_hz}]. Never raises;
            failures populate `warnings`.
        """
        result: Dict[str, Any] = {
            "ecc": None,
            "diversity_gain_db": None,
            "capacity_bps_hz": None,
            "isolation_rating": None,
            "warnings": [],
        }
        try:
            ecc_v = float(ecc)
        except (TypeError, ValueError):
            result["warnings"].append(f"invalid_ecc: {ecc!r}")
            return result
        if not (0.0 <= ecc_v < 1.0):
            result["warnings"].append(f"ecc_out_of_range: {ecc_v} (expected 0 <= ecc < 1)")
            ecc_v = min(max(ecc_v, 0.0), 0.9999)

        result["ecc"] = ecc_v
        result["diversity_gain_db"] = float(np.asarray(diversity_gain(ecc_v)))
        try:
            result["capacity_bps_hz"] = float(capacity_awgn(ecc_v, snr_db))
        except Exception as exc:
            result["warnings"].append(f"capacity_failed: {exc}")

        if ecc_v < 0.3:
            result["isolation_rating"] = "excellent"
        elif ecc_v < 0.5:
            result["isolation_rating"] = "good"
        elif ecc_v < 0.7:
            result["isolation_rating"] = "marginal"
        else:
            result["isolation_rating"] = "poor"

        if snr_sweep_db:
            try:
                lo, hi = float(min(snr_sweep_db)), float(max(snr_sweep_db))
                snrs, siso_cap, awgn_cap, fading_cap = mimo_capacity_vs_snr(
                    ecc_v,
                    snr_range_db=(lo, hi),
                    num_points=len(snr_sweep_db),
                    fading=fading,
                    K=rician_k,
                )
                result["capacity_curve"] = [
                    {
                        "snr_db": float(s),
                        "capacity_bps_hz": float(awgn),  # 2x2 AWGN+correlation
                        "siso_bps_hz": float(siso),
                        "fading_bps_hz": float(fade),
                    }
                    for s, siso, awgn, fade in zip(
                        np.asarray(snrs),
                        np.asarray(siso_cap),
                        np.asarray(awgn_cap),
                        np.asarray(fading_cap),
                    )
                ]
            except Exception as exc:
                result["warnings"].append(f"capacity_sweep_failed: {exc}")

        return result
