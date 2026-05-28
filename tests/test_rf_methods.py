"""Analytic-oracle tests for plot_antenna.rf_methods (v6.0 RF methods)."""

from __future__ import annotations

import numpy as np
import pytest

from plot_antenna.rf_methods import (
    axial_ratio_from_hv,
    three_antenna_gain,
    array_factor_uniform,
    planar_nf2ff,
    time_gate_sparam,
    port_extension_deembed,
    total_isotropic_sensitivity,
)


# ----------------------------- #48 axial ratio -----------------------------


def test_axial_ratio_circular():
    out = axial_ratio_from_hv(mag_v=1.0, mag_h=1.0, phase_diff_deg=90.0)
    assert out["axial_ratio_db"] == pytest.approx(0.0, abs=0.01)
    assert out["sense"] in ("RHCP", "LHCP")


def test_axial_ratio_linear_single_component():
    out = axial_ratio_from_hv(mag_v=1.0, mag_h=0.0, phase_diff_deg=0.0)
    assert out["axial_ratio_db"] == float("inf")
    assert out["sense"] == "linear"


def test_axial_ratio_45deg_linear():
    # Equal magnitudes, in phase -> linear at 45 deg (AR -> inf).
    out = axial_ratio_from_hv(mag_v=1.0, mag_h=1.0, phase_diff_deg=0.0)
    assert out["axial_ratio_db"] == float("inf")
    assert out["tilt_deg"] == pytest.approx(45.0, abs=0.01)


def test_axial_ratio_sense_sign():
    rh = axial_ratio_from_hv(1.0, 1.0, -90.0)
    lh = axial_ratio_from_hv(1.0, 1.0, +90.0)
    assert rh["sense"] != lh["sense"]


# ----------------------------- #48 three-antenna gain -----------------------------


def test_three_antenna_recovers_known_gains():
    g = {"A": 2.0, "B": 3.0, "C": 5.0}
    R, f_mhz = 3.0, 2450.0
    fspl = 20 * np.log10(4 * np.pi * R * f_mhz * 1e6 / 299792458.0)
    m_ab = g["A"] + g["B"] - fspl
    m_ac = g["A"] + g["C"] - fspl
    m_bc = g["B"] + g["C"] - fspl
    out = three_antenna_gain(m_ab, m_ac, m_bc, R, f_mhz)
    assert out["gain_a_dbi"] == pytest.approx(2.0, abs=1e-3)
    assert out["gain_b_dbi"] == pytest.approx(3.0, abs=1e-3)
    assert out["gain_c_dbi"] == pytest.approx(5.0, abs=1e-3)


# ----------------------------- #46 array factor -----------------------------


def test_array_broadside_main_beam():
    out = array_factor_uniform(n_elements=8, spacing_lambda=0.5, steer_deg=0.0)
    assert out["main_beam_deg"] == pytest.approx(0.0, abs=1.0)
    assert out["grating_lobe_risk"] is False


def test_array_steered_beam_moves():
    out = array_factor_uniform(n_elements=16, spacing_lambda=0.5, steer_deg=30.0)
    assert out["main_beam_deg"] == pytest.approx(30.0, abs=2.0)


def test_array_grating_lobe_flag_at_full_wavelength():
    out = array_factor_uniform(n_elements=8, spacing_lambda=1.0, steer_deg=0.0)
    assert out["grating_lobe_risk"] is True


def test_array_more_elements_narrower_beam():
    w8 = array_factor_uniform(8, 0.5, 0.0)["hpbw_deg"]
    w32 = array_factor_uniform(32, 0.5, 0.0)["hpbw_deg"]
    assert w32 < w8


# ----------------------------- #47 planar NF2FF -----------------------------


def test_nf2ff_uniform_aperture_peaks_at_boresight():
    # Uniform-illumination square aperture -> far-field peak at center (boresight).
    nf = np.ones((16, 16), dtype=complex)
    out = planar_nf2ff(nf, dx_m=0.01, dy_m=0.01, freq_mhz=10000.0, pad=2)
    cy, cx = out["shape"][0] // 2, out["shape"][1] // 2
    by, bx = out["boresight_index"]
    assert abs(by - cy) <= 1 and abs(bx - cx) <= 1
    assert out["peak_db"] == pytest.approx(0.0, abs=1e-6)


# ----------------------------- #43 time-gating + de-embed -----------------------------


def test_time_gating_keeps_in_gate_removes_out_of_gate():
    f = np.linspace(1e9, 2e9, 256)
    tau0 = 5e-9
    s = np.exp(-1j * 2 * np.pi * f * tau0)  # a single response at delay tau0
    out = time_gate_sparam(f, s, t_start_s=tau0 - 1e-9, t_stop_s=tau0 + 1e-9)
    # The time response should peak near tau0.
    t = np.array(out["time_s"])
    mag = np.array(out["time_response_mag"])
    assert abs(t[int(np.argmax(mag))] - tau0) < 0.5e-9
    # Gating a window that excludes tau0 strongly suppresses the signal
    # (a hard rectangular gate leaks, so this is ~30+ dB down, not exactly 0).
    out2 = time_gate_sparam(f, s, t_start_s=20e-9, t_stop_s=30e-9)
    sg = np.array(out2["s_gated_complex"])
    assert np.max(np.abs(sg)) < 0.05  # > 26 dB suppression vs the ~1.0 input


def test_port_extension_flattens_linear_phase():
    f = np.linspace(1e9, 2e9, 64)
    tau = 2e-9
    s = np.exp(-1j * 2 * np.pi * f * 2 * tau)  # S11 with a 2-way delay tau
    deemb = port_extension_deembed(f, s, delay_s=tau, two_way=True)
    # Phase should be ~flat (0) after removing the round-trip delay.
    assert np.allclose(np.angle(deemb), np.angle(deemb)[0], atol=1e-6)


# ----------------------------- #42 TIS -----------------------------


def test_tis_constant_eis_returns_that_eis():
    inc = 5.0
    theta_deg = np.arange(0, 180 + inc, inc)
    n_phi = int(360 / inc)
    eis = np.full((len(theta_deg), n_phi), -95.0)  # constant -95 dBm sensitivity
    tis = total_isotropic_sensitivity(eis, np.deg2rad(theta_deg), inc, inc)
    assert tis == pytest.approx(-95.0, abs=0.05)


def test_tis_worse_at_one_pole_raises_tis():
    inc = 5.0
    theta_deg = np.arange(0, 180 + inc, inc)
    n_phi = int(360 / inc)
    eis = np.full((len(theta_deg), n_phi), -95.0)
    eis[0, :] = -80.0  # one bad region (worse sensitivity)
    tis = total_isotropic_sensitivity(eis, np.deg2rad(theta_deg), inc, inc)
    # TIS is dominated by the worst (harmonic mean) -> >= -95 (worse) but the bad
    # region has near-zero solid angle (pole), so only slightly worse.
    assert tis >= -95.0
