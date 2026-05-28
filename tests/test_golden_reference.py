"""Golden-reference regression tests for the TRP / gain integration core (issue #12).

These lock the analysis math against analytic oracles so a future refactor can
never silently shift TRP (e.g. the 1/(4pi) EIRP normalization). The key oracle:
an isotropic EIRP pattern integrates back to itself.
"""

from __future__ import annotations

import numpy as np
import pytest

from plot_antenna.calculations import calculate_trp, calculate_partial_trp


def _grid(inc_theta=5.0, inc_phi=5.0):
    theta_deg = np.arange(0.0, 180.0 + inc_theta, inc_theta)
    n_phi = int(round(360.0 / inc_phi))
    return theta_deg, np.deg2rad(theta_deg), n_phi, inc_theta, inc_phi


@pytest.mark.parametrize("inc", [10.0, 5.0, 2.0])
def test_isotropic_eirp_integrates_to_itself(inc):
    """Constant EIRP over the sphere -> TRP == that EIRP (the core oracle)."""
    theta_deg, theta_rad, n_phi, it, ip = _grid(inc, inc)
    for level in (0.0, 5.0, -7.5):
        power = np.full((len(theta_deg), n_phi), level)
        trp = calculate_trp(power, theta_rad, it, ip)
        # Quadrature tightens as the grid refines.
        tol = 0.05 if inc <= 5.0 else 0.15
        assert trp == pytest.approx(level, abs=tol), f"inc={inc} level={level} -> {trp}"


def test_upper_hemisphere_is_minus_3dB():
    """EIRP = P0 over the upper hemisphere, ~0 below -> TRP = P0 - 3.01 dB."""
    inc = 2.0
    theta_deg, theta_rad, n_phi, it, ip = _grid(inc, inc)
    P0 = 0.0  # dBm
    power = np.full((len(theta_deg), n_phi), -200.0)  # ~0 mW everywhere
    power[theta_deg <= 90.0, :] = P0  # upper hemisphere radiates P0
    trp = calculate_trp(power, theta_rad, it, ip)
    assert trp == pytest.approx(P0 - 3.0103, abs=0.1)


def test_trp_scales_linearly_in_dB():
    """Adding X dB to every EIRP sample adds X dB to TRP."""
    theta_deg, theta_rad, n_phi, it, ip = _grid()
    base = np.full((len(theta_deg), n_phi), 0.0)
    trp0 = calculate_trp(base, theta_rad, it, ip)
    trp5 = calculate_trp(base + 5.0, theta_rad, it, ip)
    assert (trp5 - trp0) == pytest.approx(5.0, abs=1e-6)


def test_partial_trp_full_sphere_isotropic():
    """calculate_partial_trp over the full sphere on a flat pattern is finite + stable."""
    inc = 5.0
    theta_deg = np.arange(0.0, 180.0 + inc, inc)
    phi_deg = np.arange(0.0, 360.0, inc)
    pattern = np.zeros((len(theta_deg), len(phi_deg)))  # 0 dB everywhere
    full = calculate_partial_trp(pattern, theta_deg, phi_deg)
    upper = calculate_partial_trp(pattern, theta_deg, phi_deg, theta_min=0.0, theta_max=90.0)
    # Upper hemisphere integral is ~half the full-sphere integral -> ~3 dB lower.
    assert (full - upper) == pytest.approx(3.0103, abs=0.2)
