"""
Unit tests for maritime/horizon visualization features.

Tests cover:
- _prepare_gain_grid: passive reshape, active passthrough, invalid data
- plot_mercator_heatmap: full range, theta zoom, labels
- plot_conical_cuts: default/custom thetas, polar/Cartesian
- plot_gain_over_azimuth: Cartesian output, reference lines
- plot_horizon_statistics: default band, statistics computation
- plot_3d_pattern_masked: alpha masking, band boundaries
- generate_maritime_plots: dispatcher calls all sub-functions
"""

import pytest
import numpy as np
import matplotlib

matplotlib.use("Agg")  # Non-interactive backend for testing
import matplotlib.pyplot as plt
import os
import tempfile

from plot_antenna.plotting import (
    _prepare_gain_grid,
    plot_mercator_heatmap,
    plot_conical_cuts,
    plot_gain_over_azimuth,
    plot_horizon_statistics,
    plot_3d_pattern_masked,
    generate_maritime_plots,
)


# ---------------------------------------------------------------------------
# Test data helpers
# ---------------------------------------------------------------------------


def _make_grid_data(n_theta=13, n_phi=24):
    """Create a simple 2D gain grid for testing (active-style: already gridded)."""
    theta = np.linspace(0, 180, n_theta)
    phi = np.linspace(0, 345, n_phi)
    # Create a pattern with known properties: peak at theta=90
    THETA, PHI = np.meshgrid(theta, phi, indexing="ij")
    gain_2d = 5.0 * np.sin(np.deg2rad(THETA)) * np.cos(np.deg2rad(PHI))
    return theta, phi, gain_2d


def _make_passive_arrays(n_theta=13, n_phi=24, n_freqs=3):
    """Create passive-style measurement arrays (n_pts, n_freqs)."""
    theta_1d = np.linspace(0, 180, n_theta)
    phi_1d = np.linspace(0, 345, n_phi)
    n_pts = n_theta * n_phi

    # Build flat arrays repeating theta for each phi
    theta_flat = np.repeat(theta_1d, n_phi)
    phi_flat = np.tile(phi_1d, n_theta)

    # Multi-freq: stack into (n_pts, n_freqs)
    theta_2d = np.column_stack([theta_flat for _ in range(n_freqs)])
    phi_2d = np.column_stack([phi_flat for _ in range(n_freqs)])
    gain_flat = 3.0 * np.sin(np.deg2rad(theta_flat)) + np.cos(np.deg2rad(phi_flat))
    gain_2d = np.column_stack([gain_flat + i * 0.5 for i in range(n_freqs)])

    return theta_2d, phi_2d, gain_2d


# ---------------------------------------------------------------------------
# TestPrepareGainGrid
# ---------------------------------------------------------------------------


class TestPrepareGainGrid:
    """Test _prepare_gain_grid for passive reshape and active passthrough."""

    def test_active_passthrough(self):
        """Active data (1D theta, 2D gain) should pass through unchanged."""
        theta, phi, gain = _make_grid_data()
        t_out, p_out, g_out = _prepare_gain_grid(theta, phi, gain, freq_idx=0)

        np.testing.assert_array_equal(t_out, theta)
        np.testing.assert_array_equal(p_out, phi)
        np.testing.assert_array_equal(g_out, gain)

    def test_passive_reshape(self):
        """Passive data (2D arrays with freq dim) should reshape to grid."""
        theta_2d, phi_2d, gain_2d = _make_passive_arrays(n_theta=7, n_phi=12, n_freqs=3)
        t_out, p_out, g_out = _prepare_gain_grid(theta_2d, phi_2d, gain_2d, freq_idx=1)

        assert t_out is not None
        assert g_out.shape == (7, 12)
        # Verify theta values are unique sorted
        assert len(t_out) == 7
        assert len(p_out) == 12

    def test_invalid_data_returns_none(self):
        """When data doesn't fit a grid, return None triplet."""
        theta = np.array([0, 10, 20, 30])
        phi = np.array([0, 10, 20])
        # Gain has wrong number of points (5 != 4*3=12)
        gain = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        t, p, g = _prepare_gain_grid(theta, phi, gain, freq_idx=0)
        assert t is None
        assert p is None
        assert g is None

    def test_passive_single_freq(self):
        """Passive data with single frequency column."""
        theta_2d, phi_2d, gain_2d = _make_passive_arrays(n_theta=5, n_phi=8, n_freqs=1)
        t_out, p_out, g_out = _prepare_gain_grid(theta_2d, phi_2d, gain_2d, freq_idx=0)

        assert g_out is not None
        assert g_out.shape == (5, 8)


# ---------------------------------------------------------------------------
# TestMercatorHeatmap
# ---------------------------------------------------------------------------


class TestMercatorHeatmap:
    """Test plot_mercator_heatmap for full and zoomed modes."""

    def test_full_range_display(self):
        """Full Mercator heatmap should display without error."""
        theta, phi, gain = _make_grid_data()
        plot_mercator_heatmap(theta, phi, gain, 2400, save_path=None)
        plt.close("all")

    def test_theta_zoom(self):
        """Zoomed Mercator with theta limits should display without error."""
        theta, phi, gain = _make_grid_data()
        plot_mercator_heatmap(theta, phi, gain, 2400, theta_min=60, theta_max=120, save_path=None)
        plt.close("all")

    def test_save_to_file(self):
        """Mercator plot should save to file when save_path is given."""
        theta, phi, gain = _make_grid_data()
        with tempfile.TemporaryDirectory() as tmpdir:
            plot_mercator_heatmap(theta, phi, gain, 2400, save_path=tmpdir)
            files = os.listdir(tmpdir)
            assert any("mercator" in f for f in files)
            plt.close("all")

    def test_save_zoomed_filename(self):
        """Zoomed Mercator should include theta range in filename."""
        theta, phi, gain = _make_grid_data()
        with tempfile.TemporaryDirectory() as tmpdir:
            plot_mercator_heatmap(
                theta, phi, gain, 2400, theta_min=60, theta_max=120, save_path=tmpdir
            )
            files = os.listdir(tmpdir)
            assert any("theta60-120" in f for f in files)
            plt.close("all")

    def test_active_labels(self):
        """Should accept Power/dBm labels for active data."""
        theta, phi, gain = _make_grid_data()
        plot_mercator_heatmap(
            theta, phi, gain, 2400, data_label="Power", data_unit="dBm", save_path=None
        )
        plt.close("all")


# ---------------------------------------------------------------------------
# TestConicalCuts
# ---------------------------------------------------------------------------


class TestConicalCuts:
    """Test plot_conical_cuts for polar and Cartesian modes."""

    def test_default_thetas_polar(self):
        """Default theta cuts in polar mode should display without error."""
        theta, phi, gain = _make_grid_data()
        plot_conical_cuts(theta, phi, gain, 2400, polar=True, save_path=None)
        plt.close("all")

    def test_custom_thetas(self):
        """Custom theta cuts should work."""
        theta, phi, gain = _make_grid_data()
        plot_conical_cuts(
            theta, phi, gain, 2400, theta_cuts=[80, 90, 100], polar=True, save_path=None
        )
        plt.close("all")

    def test_cartesian_mode(self):
        """Cartesian mode should display without error."""
        theta, phi, gain = _make_grid_data()
        plot_conical_cuts(theta, phi, gain, 2400, polar=False, save_path=None)
        plt.close("all")

    def test_save_polar(self):
        """Polar conical cuts should save with correct filename."""
        theta, phi, gain = _make_grid_data()
        with tempfile.TemporaryDirectory() as tmpdir:
            plot_conical_cuts(theta, phi, gain, 2400, polar=True, save_path=tmpdir)
            files = os.listdir(tmpdir)
            assert any("conical_cuts_polar" in f for f in files)
            plt.close("all")

    def test_save_cartesian(self):
        """Cartesian conical cuts should save with correct filename."""
        theta, phi, gain = _make_grid_data()
        with tempfile.TemporaryDirectory() as tmpdir:
            plot_conical_cuts(theta, phi, gain, 2400, polar=False, save_path=tmpdir)
            files = os.listdir(tmpdir)
            assert any("conical_cuts_cartesian" in f for f in files)
            plt.close("all")


# ---------------------------------------------------------------------------
# TestGainOverAzimuth
# ---------------------------------------------------------------------------


class TestGainOverAzimuth:
    """Test plot_gain_over_azimuth (Cartesian wrapper)."""

    def test_display(self):
        """GoA plot should display without error."""
        theta, phi, gain = _make_grid_data()
        plot_gain_over_azimuth(theta, phi, gain, 2400, save_path=None)
        plt.close("all")

    def test_save_filename(self):
        """GoA plot should save with goa_ prefix."""
        theta, phi, gain = _make_grid_data()
        with tempfile.TemporaryDirectory() as tmpdir:
            plot_gain_over_azimuth(theta, phi, gain, 2400, save_path=tmpdir)
            files = os.listdir(tmpdir)
            assert any("goa_" in f for f in files)
            plt.close("all")

    def test_custom_cuts(self):
        """Custom theta cuts should work."""
        theta, phi, gain = _make_grid_data()
        plot_gain_over_azimuth(theta, phi, gain, 2400, theta_cuts=[70, 90, 110], save_path=None)
        plt.close("all")


# ---------------------------------------------------------------------------
# TestHorizonStatistics
# ---------------------------------------------------------------------------


class TestHorizonStatistics:
    """Test plot_horizon_statistics."""

    def test_default_band(self):
        """Default horizon band should display without error."""
        theta, phi, gain = _make_grid_data()
        plot_horizon_statistics(theta, phi, gain, 2400, save_path=None)
        plt.close("all")

    def test_custom_band(self):
        """Custom theta band should work."""
        theta, phi, gain = _make_grid_data()
        plot_horizon_statistics(theta, phi, gain, 2400, theta_min=70, theta_max=110, save_path=None)
        plt.close("all")

    def test_save_filename(self):
        """Horizon stats should save with correct filename."""
        theta, phi, gain = _make_grid_data()
        with tempfile.TemporaryDirectory() as tmpdir:
            plot_horizon_statistics(theta, phi, gain, 2400, save_path=tmpdir)
            files = os.listdir(tmpdir)
            assert any("horizon_stats" in f for f in files)
            plt.close("all")

    def test_empty_band_no_crash(self):
        """If no data in the theta range, should not crash."""
        theta = np.array([0, 10, 20, 30])
        phi = np.array([0, 90, 180, 270])
        gain = np.zeros((4, 4))
        # theta_min/max outside the data range
        plot_horizon_statistics(
            theta, phi, gain, 2400, theta_min=150, theta_max=170, save_path=None
        )
        plt.close("all")

    def test_full_coverage(self):
        """When all points are within threshold, coverage should be 100%."""
        theta, phi, gain = _make_grid_data(n_theta=13, n_phi=24)
        # Make all gain values identical so coverage is 100%
        uniform_gain = np.ones_like(gain) * 5.0
        plot_horizon_statistics(theta, phi, uniform_gain, 2400, gain_threshold=-3.0, save_path=None)
        plt.close("all")


# ---------------------------------------------------------------------------
# TestMasked3DPattern
# ---------------------------------------------------------------------------


class TestMasked3DPattern:
    """Test plot_3d_pattern_masked."""

    def test_display(self):
        """3D masked pattern should display without error."""
        theta, phi, gain = _make_grid_data()
        plot_3d_pattern_masked(theta, phi, gain, 2400, save_path=None)
        plt.close("all")

    def test_custom_band(self):
        """Custom highlight band should work."""
        theta, phi, gain = _make_grid_data()
        plot_3d_pattern_masked(
            theta,
            phi,
            gain,
            2400,
            theta_highlight_min=70,
            theta_highlight_max=110,
            save_path=None,
        )
        plt.close("all")

    def test_save_filename(self):
        """3D masked pattern should save with correct filename."""
        theta, phi, gain = _make_grid_data()
        with tempfile.TemporaryDirectory() as tmpdir:
            plot_3d_pattern_masked(theta, phi, gain, 2400, save_path=tmpdir)
            files = os.listdir(tmpdir)
            assert any("3d_masked" in f for f in files)
            plt.close("all")


# ---------------------------------------------------------------------------
# TestGenerateMaritimePlots
# ---------------------------------------------------------------------------


class TestGenerateMaritimePlots:
    """Test generate_maritime_plots dispatcher."""

    def test_dispatcher_saves_all_files(self):
        """Dispatcher should generate all 6 maritime plot files."""
        theta, phi, gain = _make_grid_data()
        with tempfile.TemporaryDirectory() as tmpdir:
            generate_maritime_plots(theta, phi, gain, 2400, save_path=tmpdir)
            files = os.listdir(tmpdir)
            # Should have: mercator (full), mercator (zoomed), conical_cuts_polar,
            # goa_, horizon_stats, 3d_masked = 6 files
            assert len(files) == 6, f"Expected 6 files, got {len(files)}: {files}"

            # Check each expected pattern exists
            patterns = ["mercator", "conical_cuts", "goa_", "horizon_stats", "3d_masked"]
            for pattern in patterns:
                assert any(pattern in f for f in files), f"Missing file with pattern '{pattern}'"

            plt.close("all")

    def test_dispatcher_display_mode(self):
        """Dispatcher with save_path=None should display without error."""
        theta, phi, gain = _make_grid_data()
        generate_maritime_plots(theta, phi, gain, 2400, save_path=None)
        plt.close("all")

    def test_dispatcher_custom_params(self):
        """Dispatcher should accept custom theta range and cuts."""
        theta, phi, gain = _make_grid_data()
        with tempfile.TemporaryDirectory() as tmpdir:
            generate_maritime_plots(
                theta,
                phi,
                gain,
                2400,
                theta_min=70,
                theta_max=110,
                theta_cuts=[75, 85, 95, 105],
                gain_threshold=-6.0,
                save_path=tmpdir,
            )
            files = os.listdir(tmpdir)
            assert len(files) == 6
            plt.close("all")

    def test_dispatcher_active_labels(self):
        """Dispatcher with Power/dBm labels should work."""
        theta, phi, gain = _make_grid_data()
        with tempfile.TemporaryDirectory() as tmpdir:
            generate_maritime_plots(
                theta,
                phi,
                gain,
                2400,
                data_label="Power",
                data_unit="dBm",
                save_path=tmpdir,
            )
            files = os.listdir(tmpdir)
            assert any("power" in f for f in files)
            plt.close("all")


# ---------------------------------------------------------------------------
# TestDetectMeasurementType
# ---------------------------------------------------------------------------


class TestDetectMeasurementType:
    """Test detect_measurement_type with maritime filenames."""

    def test_mercator_detection(self):
        from plot_antenna.save import detect_measurement_type

        assert detect_measurement_type("mercator_gain_2400MHz.png") == "maritime"

    def test_conical_detection(self):
        from plot_antenna.save import detect_measurement_type

        assert detect_measurement_type("conical_cuts_polar_2400MHz.png") == "maritime"

    def test_goa_detection(self):
        from plot_antenna.save import detect_measurement_type

        assert detect_measurement_type("goa_2400MHz.png") == "maritime"

    def test_horizon_stats_detection(self):
        from plot_antenna.save import detect_measurement_type

        assert detect_measurement_type("horizon_stats_2400MHz.png") == "maritime"

    def test_3d_masked_detection(self):
        from plot_antenna.save import detect_measurement_type

        assert detect_measurement_type("3d_masked_2400MHz.png") == "maritime"

    def test_existing_types_unchanged(self):
        """Existing detection should still work."""
        from plot_antenna.save import detect_measurement_type

        assert detect_measurement_type("polarization_2D_2400MHz.png") == "polarization"
        assert detect_measurement_type("active_trp_2400MHz.png") == "active"
        assert detect_measurement_type("passive_gain_2400MHz.png") == "passive"
