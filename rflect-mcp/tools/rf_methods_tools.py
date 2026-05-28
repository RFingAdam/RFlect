"""
Advanced RF-method MCP tools for RFlect (v6.0).

Thin wrappers over plot_antenna.rf_methods (axial ratio, 3-antenna gain, array
synthesis, planar NF2FF, S-param time-gating/de-embedding, CTIA TIS) plus a
CTIA OTA test-plan template generator. Deterministic; never raise; failures in
`warnings`.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import numpy as np

from plot_antenna.rf_methods import (
    axial_ratio_from_hv,
    three_antenna_gain,
    array_factor_uniform,
    planar_nf2ff,
    time_gate_sparam,
    port_extension_deembed,
    total_isotropic_sensitivity,
)


def register_rf_methods_tools(mcp):
    """Register the advanced RF-method tools with the MCP server."""

    @mcp.tool()
    def analyze_axial_ratio(mag_v: float, mag_h: float, phase_diff_deg: float) -> Dict[str, Any]:
        """Circular-polarization axial ratio / tilt / sense from orthogonal V & H.

        Args: mag_v, mag_h (linear field magnitudes), phase_diff_deg = phase(V) -
        phase(H). Returns: axial_ratio_db (0 = circular), tilt_deg, sense
        (RHCP/LHCP/linear).
        """
        try:
            return axial_ratio_from_hv(mag_v, mag_h, phase_diff_deg)
        except Exception as exc:  # noqa: BLE001
            return {"error": f"axial_ratio failed: {exc}"}

    @mcp.tool()
    def three_antenna_gain_method(
        m_ab_db: float, m_ac_db: float, m_bc_db: float, range_m: float, freq_mhz: float
    ) -> Dict[str, Any]:
        """Absolute gain of 3 antennas from 3 pairwise S21 ratios (classic method).

        Each M_xy is the pair (x,y) received/transmitted power ratio (dB) at the
        given range. Returns gain_a/b/c_dbi + the FSPL used.
        """
        try:
            return three_antenna_gain(m_ab_db, m_ac_db, m_bc_db, range_m, freq_mhz)
        except Exception as exc:  # noqa: BLE001
            return {"error": f"three_antenna_gain failed: {exc}"}

    @mcp.tool()
    def synthesize_array(
        n_elements: int,
        spacing_lambda: float,
        steer_deg: float = 0.0,
        taper: str = "uniform",
    ) -> Dict[str, Any]:
        """Uniform linear-array factor with electronic steering + grating check.

        Returns the normalized |AF| vs angle (theta from broadside), realized
        main-beam angle, HPBW, peak sidelobe level, and a grating-lobe-risk flag.
        """
        try:
            return array_factor_uniform(
                n_elements, spacing_lambda, steer_deg=steer_deg, taper=taper
            )
        except Exception as exc:  # noqa: BLE001
            return {"error": f"synthesize_array failed: {exc}"}

    @mcp.tool()
    def near_field_to_far_field(
        near_field_real: List[List[float]],
        near_field_imag: List[List[float]],
        dx_m: float,
        dy_m: float,
        freq_mhz: float,
        pad: int = 4,
    ) -> Dict[str, Any]:
        """Planar near-field -> far-field via the 2D FFT spectral method.

        Supply the complex aperture field as separate real/imag 2D grids. Returns
        the normalized far-field magnitude pattern (dB) over the FFT grid and the
        boresight index.
        """
        try:
            nf = np.asarray(near_field_real, dtype=float) + 1j * np.asarray(
                near_field_imag, dtype=float
            )
            return planar_nf2ff(nf, dx_m, dy_m, freq_mhz, pad=pad)
        except Exception as exc:  # noqa: BLE001
            return {"error": f"nf2ff failed: {exc}"}

    @mcp.tool()
    def time_gate_sparameters(
        freq_hz: List[float],
        s_real: List[float],
        s_imag: List[float],
        t_start_s: float,
        t_stop_s: float,
    ) -> Dict[str, Any]:
        """Time-gate an S-parameter sweep (IFFT -> window -> FFT) to remove
        out-of-gate responses such as chamber reflections.

        Returns the gated S (real/imag), the time axis, and the time-domain
        magnitude response.
        """
        try:
            s = np.asarray(s_real, dtype=float) + 1j * np.asarray(s_imag, dtype=float)
            out = time_gate_sparam(np.asarray(freq_hz, dtype=float), s, t_start_s, t_stop_s)
            sg = np.asarray(out.pop("s_gated_complex"))
            out["s_gated_real"] = np.real(sg).tolist()
            out["s_gated_imag"] = np.imag(sg).tolist()
            return out
        except Exception as exc:  # noqa: BLE001
            return {"error": f"time_gate failed: {exc}"}

    @mcp.tool()
    def deembed_port_extension(
        freq_hz: List[float],
        s_real: List[float],
        s_imag: List[float],
        delay_s: float,
        two_way: bool = True,
    ) -> Dict[str, Any]:
        """Reference-plane de-embedding (electrical-length removal) of a fixture/
        cable delay. two_way=True for a reflection (S11). Returns de-embedded S."""
        try:
            s = np.asarray(s_real, dtype=float) + 1j * np.asarray(s_imag, dtype=float)
            d = port_extension_deembed(
                np.asarray(freq_hz, dtype=float), s, delay_s, two_way=two_way
            )
            return {"s_real": np.real(d).tolist(), "s_imag": np.imag(d).tolist()}
        except Exception as exc:  # noqa: BLE001
            return {"error": f"deembed failed: {exc}"}

    @mcp.tool()
    def calculate_tis(
        eis_dbm_2d: List[List[float]],
        inc_theta_deg: float,
        inc_phi_deg: float,
    ) -> Dict[str, Any]:
        """CTIA Total Isotropic Sensitivity from a sphere of EIS (dBm) samples.

        eis_dbm_2d is theta x phi EIS in dBm; theta spans 0..180 in inc_theta
        steps. Returns tis_dbm (the sphere harmonic mean).
        """
        try:
            arr = np.asarray(eis_dbm_2d, dtype=float)
            theta = np.deg2rad(np.arange(0.0, 180.0 + inc_theta_deg, inc_theta_deg)[: arr.shape[0]])
            tis = total_isotropic_sensitivity(arr, theta, inc_theta_deg, inc_phi_deg)
            return {"tis_dbm": tis, "n_theta": int(arr.shape[0]), "n_phi": int(arr.shape[1])}
        except Exception as exc:  # noqa: BLE001
            return {"error": f"calculate_tis failed: {exc}"}

    @mcp.tool()
    def ctia_test_plan_template(
        band: str = "2.4GHz",
        grid_step_deg: int = 15,
        kind: str = "TRP",
    ) -> Dict[str, Any]:
        """Generate a CTIA-style OTA TRP/TIS test-plan template.

        Returns a structured measurement plan: the theta/phi grid, the channels
        to test for the band, and the partial-sphere definitions — a starting
        scaffold for a CTIA-aligned campaign (verify against the current CTIA
        test plan revision for certification).

        Args:
            band: "2.4GHz" | "5GHz".
            grid_step_deg: angular step for the full-sphere grid (CTIA uses 15).
            kind: "TRP" | "TIS".
        """
        channels = {
            "2.4GHz": {"low": 2412, "mid": 2437, "high": 2462},
            "5GHz": {"low": 5180, "mid": 5500, "high": 5825},
        }
        if band not in channels:
            return {"error": f"unknown band {band!r}; use 2.4GHz|5GHz", "warnings": []}
        if kind not in ("TRP", "TIS"):
            return {"error": f"unknown kind {kind!r}; use TRP|TIS", "warnings": []}
        thetas = list(range(0, 180 + grid_step_deg, grid_step_deg))
        phis = list(range(0, 360, grid_step_deg))
        return {
            "kind": kind,
            "band": band,
            "channels_mhz": channels[band],
            "grid": {
                "theta_deg": thetas,
                "phi_deg": phis,
                "n_points": len(thetas) * len(phis),
                "step_deg": grid_step_deg,
            },
            "partial_sphere": {
                "near_horizon_theta_band_deg": [60, 120],
                "note": "near-horizon partial TRP/TIS uses theta in [60,120]",
            },
            "metric": (
                "TRP (dBm) sphere-integrated"
                if kind == "TRP"
                else "TIS (dBm) sphere harmonic-mean of EIS"
            ),
            "warnings": [
                "scaffold only — confirm grid, channels, and pass/fail vs the "
                "current CTIA OTA test-plan revision for certification"
            ],
        }
