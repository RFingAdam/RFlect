"""Tests for plot_antenna.plot_group_delay_cst pure math (issue #14)."""

from __future__ import annotations

import numpy as np
import pytest

import matplotlib

matplotlib.use("Agg")

from plot_antenna.plot_group_delay_cst import phase_to_tau


def test_phase_to_tau_constant_delay():
    """Linear phase vs frequency -> constant group delay = tau."""
    freqs = np.linspace(2.0e9, 3.0e9, 201)
    tau_s = 5e-9
    # phase (deg) for a pure delay: phi = -2*pi*f*tau, one angle row (axis=1 = freq).
    phase_deg = np.rad2deg(-2 * np.pi * freqs * tau_s).reshape(1, -1)
    tau_ns = phase_to_tau(phase_deg, freqs)
    # Returns ns; should be ~5 ns everywhere.
    assert np.allclose(tau_ns, 5.0, atol=0.05)


def test_phase_to_tau_shape_preserved():
    freqs = np.linspace(2.0e9, 2.5e9, 51)
    phase_deg = np.zeros((3, 51))  # 3 angle cuts
    out = phase_to_tau(phase_deg, freqs)
    assert out.shape == (3, 51)
    # Flat phase -> ~0 group delay.
    assert np.allclose(out, 0.0, atol=1e-6)
