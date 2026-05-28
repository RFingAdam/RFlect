"""Tests for plot_antenna.plotting pure-compute helpers (issue #13).

plotting.py (3710 lines) had no tests; this covers the deterministic compute
paths (unit conversion, normalization, partial-sphere TRP) that feed both the
GUI and the MCP report pipeline.
"""

from __future__ import annotations

import numpy as np
import pytest

import matplotlib

matplotlib.use("Agg")

from plot_antenna.plotting import db_to_linear, normalize_gain, _compute_partial_trp


class TestDbToLinear:
    def test_zero_db(self):
        assert db_to_linear(0.0) == pytest.approx(1.0)

    def test_ten_db(self):
        assert db_to_linear(10.0) == pytest.approx(10.0)

    def test_minus_three_db(self):
        assert db_to_linear(-3.0103) == pytest.approx(0.5, abs=1e-3)


class TestNormalizeGain:
    def test_unit_range(self):
        g = np.array([-10.0, -5.0, 0.0])
        n = normalize_gain(g)
        assert n.min() == pytest.approx(0.0)
        assert n.max() == pytest.approx(1.0)

    def test_midpoint(self):
        g = np.array([0.0, 5.0, 10.0])
        assert normalize_gain(g)[1] == pytest.approx(0.5)


class TestComputePartialTRP:
    def _grid(self, inc=5.0):
        theta = np.arange(0.0, 180.0 + inc, inc)
        phi = np.arange(0.0, 360.0, inc)
        return theta, phi

    def test_isotropic_full_sphere(self):
        theta, phi = self._grid(2.0)
        gain = np.zeros((len(theta), len(phi)))  # 0 dB isotropic
        trp = _compute_partial_trp(theta, phi, gain)
        assert trp == pytest.approx(0.0, abs=0.1)

    def test_upper_hemisphere_minus_3dB(self):
        theta, phi = self._grid(2.0)
        gain = np.zeros((len(theta), len(phi)))
        full = _compute_partial_trp(theta, phi, gain)
        upper = _compute_partial_trp(theta, phi, gain, theta_min=0.0, theta_max=90.0)
        assert (full - upper) == pytest.approx(3.0103, abs=0.2)

    def test_empty_selection_returns_neg_inf(self):
        theta, phi = self._grid()
        gain = np.zeros((len(theta), len(phi)))
        # A theta window outside the data selects nothing.
        out = _compute_partial_trp(theta, phi, gain, theta_min=200.0, theta_max=300.0)
        assert out == float("-inf")


class TestPlotStyleHelpers:
    """#41 — shared plot style/save helpers."""

    def test_colormap_registry(self):
        from plot_antenna.plotting import colormap_for

        assert colormap_for("gain") == "viridis"
        assert colormap_for("power") == "turbo"
        assert colormap_for("unknown_kind") == "viridis"  # default

    def test_save_figure_writes_png(self, tmp_path):
        import matplotlib.pyplot as plt
        from plot_antenna.plotting import save_figure, PLOT_DPI

        fig = plt.figure()
        plt.plot([0, 1, 2], [0, 1, 4])
        out = tmp_path / "fig.png"
        save_figure(fig, str(out))
        import os

        assert os.path.getsize(out) > 0
        plt.close("all")
        assert PLOT_DPI == 300
