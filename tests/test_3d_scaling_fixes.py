"""Regression tests for the v6.1.0 3D rendering/scaling fixes.

Covers:
  #1  save_to_results_folder feeds per-polarization power arrays into the
      active H-pol / V-pol 3D plots (was passing total power).
  #2  plot_passive_3d_component is robust to degenerate input (constant gain /
      NaN) — no divide-by-zero / invalid-value / all-NaN warnings, mirroring the
      active route's guards.
  #3  _setup_3d_axes never collapses the bounding box to lim=0 / lim=NaN for a
      flat or all-NaN pattern.
  #4  process_data (the 3D render helper) returns only (grid, theta, phi); the
      dead Cartesian X/Y/Z + db_to_linear radius it used to return are gone.
"""

from __future__ import annotations

import ast
import pathlib
import warnings

import numpy as np
import pytest

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

import plot_antenna  # noqa: E402
from plot_antenna.plotting import (  # noqa: E402
    PHI_RESOLUTION,
    THETA_RESOLUTION,
    _setup_3d_axes,
    plot_passive_3d_component,
    process_data,
)


def _point_grid(theta_step: float = 30.0, phi_step: float = 30.0):
    """Flattened (theta, phi) point cloud spanning a full sphere."""
    thetas = np.arange(0.0, 180.0 + theta_step, theta_step)
    phis = np.arange(0.0, 360.0, phi_step)
    TH, PH = np.meshgrid(thetas, phis, indexing="ij")
    return TH.flatten(), PH.flatten()


# ---------------------------------------------------------------- #4
class TestProcessDataContract:
    def test_returns_grid_and_axes_only(self):
        theta_pts, phi_pts = _point_grid()
        data = np.full(theta_pts.shape, -3.0)
        out = process_data(data, phi_pts, theta_pts)

        assert len(out) == 3, "process_data should return (grid, theta, phi) only"
        data_interp, theta_interp, phi_interp = out
        assert data_interp.shape == (THETA_RESOLUTION, PHI_RESOLUTION)
        assert theta_interp.shape == (THETA_RESOLUTION,)
        assert phi_interp.shape == (PHI_RESOLUTION,)


# ---------------------------------------------------------------- #3
class TestSetup3DAxesDegenerate:
    @staticmethod
    def _axes():
        fig = plt.figure()
        ax = fig.add_subplot(111, projection="3d")
        return fig, ax

    def test_flat_pattern_gives_nondegenerate_box(self):
        fig, ax = self._axes()
        zeros = np.zeros((8, 8))
        _setup_3d_axes(ax, zeros, zeros, zeros)
        lo, hi = ax.get_xlim3d()
        assert np.isfinite(lo) and np.isfinite(hi)
        assert hi - lo >= 1.0, "flat pattern collapsed the box"
        plt.close(fig)

    def test_all_nan_gives_finite_box(self):
        fig, ax = self._axes()
        nan = np.full((8, 8), np.nan)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            _setup_3d_axes(ax, nan, nan, nan)
        for getlim in (ax.get_xlim3d, ax.get_ylim3d, ax.get_zlim3d):
            lo, hi = getlim()
            assert np.isfinite(lo) and np.isfinite(hi), "NaN extent gave non-finite limits"
            assert hi - lo >= 1.0
        plt.close(fig)


# ---------------------------------------------------------------- #2
class TestPassive3DDegenerateInput:
    def test_nan_gain_produces_finite_geometry(self, tmp_path, monkeypatch):
        """A NaN in the gain data (bad / out-of-hull point) must not poison the
        whole surface via non-NaN-aware min/max. Spy on the coords handed to
        _setup_3d_axes and require them finite."""
        import plot_antenna.plotting as plotting

        captured: dict[str, np.ndarray] = {}
        orig = plotting._setup_3d_axes

        def spy(ax, X, Y, Z):
            captured["X"], captured["Y"], captured["Z"] = X, Y, Z
            return orig(ax, X, Y, Z)

        monkeypatch.setattr(plotting, "_setup_3d_axes", spy)

        theta_pts, phi_pts = _point_grid()
        n = theta_pts.size
        theta2d = theta_pts.reshape(n, 1)
        phi2d = phi_pts.reshape(n, 1)
        gain = (np.linspace(-12.0, -2.0, n)).reshape(n, 1)
        gain[0, 0] = np.nan  # one bad data point

        plot_passive_3d_component(
            theta2d,
            phi2d,
            gain,  # h_gain_dB
            gain,  # v_gain_dB
            gain,  # Total_Gain_dB
            [2400.0],
            2400.0,
            "total",
            save_path=str(tmp_path),
        )

        assert captured, "surface was never configured"
        for key in ("X", "Y", "Z"):
            assert np.all(np.isfinite(captured[key])), f"{key} has non-finite values"
        assert (tmp_path / "3D_total_1of2.png").exists()
        assert (tmp_path / "3D_total_2of2.png").exists()


# ---------------------------------------------------------------- #1
def _save_module_source() -> str:
    return (pathlib.Path(plot_antenna.__file__).parent / "save.py").read_text()


def _active_3d_calls_by_pol():
    tree = ast.parse(_save_module_source())
    out: dict[str, list[ast.Call]] = {}
    for node in ast.walk(tree):
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == "plot_active_3d_data"
        ):
            kw = {k.arg: k.value for k in node.keywords}
            ptype = kw.get("power_type")
            if isinstance(ptype, ast.Constant):
                out.setdefault(ptype.value, []).append(node)
    return out


class TestActiveSavePerPolArrays:
    """The active 3D save must pass per-pol power, not total (data-selection bug)."""

    def test_hpol_save_uses_hpol_power(self):
        calls = _active_3d_calls_by_pol()
        assert len(calls.get("hpol", [])) == 1
        node = calls["hpol"][0]
        assert isinstance(node.args[2], ast.Name) and node.args[2].id == "h_power_dBm_2d"
        assert isinstance(node.args[4], ast.Name) and node.args[4].id == "h_power_dBm_2d_plot"

    def test_vpol_save_uses_vpol_power(self):
        calls = _active_3d_calls_by_pol()
        assert len(calls.get("vpol", [])) == 1
        node = calls["vpol"][0]
        assert isinstance(node.args[2], ast.Name) and node.args[2].id == "v_power_dBm_2d"
        assert isinstance(node.args[4], ast.Name) and node.args[4].id == "v_power_dBm_2d_plot"
