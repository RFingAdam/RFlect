"""Tests for plot_antenna.uwb_plotting (issue #14) — pure helper + plot smoke."""

from __future__ import annotations

import os

import numpy as np

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from plot_antenna.uwb_plotting import _sff_color, plot_s11_vswr


def test_sff_color_thresholds():
    # High SFF -> a "good" color; low SFF -> the fallback red.
    hi = _sff_color(0.99)
    lo = _sff_color(0.0)
    assert isinstance(hi, str) and hi.startswith("#")
    assert lo == "#e74c3c"  # documented fallback for poor fidelity


def test_plot_s11_vswr_smoke(tmp_path):
    """plot_s11_vswr renders without error on the Agg backend and can be saved."""
    f = np.linspace(2.0e9, 3.0e9, 101)
    s11 = -2.0 - 18.0 * np.exp(-(((f - 2.45e9) / 60e6) ** 2))
    vswr = (1 + 10 ** (s11 / 20)) / (1 - 10 ** (s11 / 20))
    bw = {
        "band_start_hz": 2.4e9,
        "band_stop_hz": 2.5e9,
        "bandwidth_hz": 1e8,
        "fractional_bandwidth": 1e8 / 2.45e9,
    }
    try:
        plot_s11_vswr(f, s11, vswr, bw)
        out = tmp_path / "s11.png"
        plt.savefig(out)
        assert os.path.getsize(out) > 0
    finally:
        plt.close("all")
