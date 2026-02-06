"""
Tests for plot_antenna.calculations module

Covers:
- Import verification
- Diversity Gain (diversity_gain)
- MIMO Capacity: AWGN closed-form and Monte Carlo (capacity_awgn, capacity_monte_carlo)
- Angle matching (angles_match)
- Total Radiated Power (calculate_trp)
- Data processing / FFT (process_data)
"""

import pytest
import numpy as np

from plot_antenna.calculations import (
    calculate_passive_variables,
    calculate_active_variables,
    diversity_gain,
    extract_passive_frequencies,
    capacity_awgn,
    capacity_monte_carlo,
    angles_match,
    calculate_trp,
    process_data,
)


# ---------------------------------------------------------------------------
# 1. TestImports
# ---------------------------------------------------------------------------
class TestImports:
    """Test that all calculation functions can be imported"""

    def test_imports_successful(self):
        """Verify all calculation functions import without errors"""
        assert callable(calculate_passive_variables)
        assert callable(calculate_active_variables)
        assert callable(diversity_gain)
        assert callable(extract_passive_frequencies)
        assert callable(capacity_awgn)
        assert callable(capacity_monte_carlo)
        assert callable(angles_match)
        assert callable(calculate_trp)
        assert callable(process_data)


# ---------------------------------------------------------------------------
# 2. TestDiversityGain
# ---------------------------------------------------------------------------
class TestDiversityGain:
    """Tests for the diversity_gain(ecc) function.

    Formula: DG = 10 * log10(1 / (1 - ecc)), ecc clipped to [0, 0.9999].
    """

    def test_zero_ecc_returns_zero(self):
        """diversity_gain(0.0) should be 0 dB (no diversity gain when uncorrelated)"""
        result = diversity_gain(0.0)
        assert result == pytest.approx(0.0, abs=1e-10)

    def test_half_ecc(self):
        """diversity_gain(0.5) should be approximately 3.01 dB"""
        # 10*log10(1/(1-0.5)) = 10*log10(2) ≈ 3.0103
        expected = 10 * np.log10(2.0)
        result = diversity_gain(0.5)
        assert result == pytest.approx(expected, abs=1e-4)

    def test_near_one_ecc_clipped(self):
        """diversity_gain(0.9999) should not crash and should be ~40 dB"""
        # 10*log10(1/(1-0.9999)) = 10*log10(10000) = 40.0
        result = diversity_gain(0.9999)
        assert np.isfinite(result)
        assert result == pytest.approx(40.0, abs=0.1)

    def test_array_input(self):
        """diversity_gain should accept numpy arrays and return an array"""
        ecc_array = np.array([0.0, 0.5])
        result = diversity_gain(ecc_array)
        assert isinstance(result, np.ndarray)
        assert len(result) == 2
        assert result[0] == pytest.approx(0.0, abs=1e-10)
        assert result[1] == pytest.approx(10 * np.log10(2.0), abs=1e-4)


# ---------------------------------------------------------------------------
# 3. TestCapacity
# ---------------------------------------------------------------------------
class TestCapacity:
    """Tests for capacity_awgn and capacity_monte_carlo functions."""

    def test_capacity_awgn_basic(self):
        """capacity_awgn returns a positive float for ecc=0.5, snr=10 dB"""
        result = capacity_awgn(0.5, 10)
        assert isinstance(result, (float, np.floating))
        assert result > 0.0

    def test_capacity_awgn_zero_ecc(self):
        """Zero ECC (uncorrelated) should give higher capacity than high ECC (correlated)"""
        cap_low_ecc = capacity_awgn(0.0, 10)
        cap_high_ecc = capacity_awgn(0.9, 10)
        assert cap_low_ecc > cap_high_ecc

    def test_capacity_awgn_high_snr(self):
        """Higher SNR should yield higher capacity"""
        cap_high_snr = capacity_awgn(0.5, 30)
        cap_low_snr = capacity_awgn(0.5, 0)
        assert cap_high_snr > cap_low_snr

    def test_capacity_monte_carlo_basic(self):
        """capacity_monte_carlo returns a positive float (use low trials for speed)"""
        np.random.seed(42)
        result = capacity_monte_carlo(0.5, 10, fading="rayleigh", trials=100)
        assert isinstance(result, (float, np.floating))
        assert result > 0.0


# ---------------------------------------------------------------------------
# 4. TestAnglesMatch
# ---------------------------------------------------------------------------
class TestAnglesMatch:
    """Tests for the angles_match function."""

    def test_matching_returns_true(self):
        """Identical angle parameters for H and V polarizations should return True"""
        result = angles_match(
            start_phi_h=0, stop_phi_h=360, inc_phi_h=15,
            start_theta_h=0, stop_theta_h=180, inc_theta_h=5,
            start_phi_v=0, stop_phi_v=360, inc_phi_v=15,
            start_theta_v=0, stop_theta_v=180, inc_theta_v=5,
        )
        assert result is True

    def test_mismatched_returns_false(self):
        """Different angle parameters should return False"""
        result = angles_match(
            start_phi_h=0, stop_phi_h=360, inc_phi_h=15,
            start_theta_h=0, stop_theta_h=180, inc_theta_h=5,
            start_phi_v=0, stop_phi_v=350, inc_phi_v=10,
            start_theta_v=0, stop_theta_v=170, inc_theta_v=10,
        )
        assert result is False


# ---------------------------------------------------------------------------
# 5. TestCalculateTRP
# ---------------------------------------------------------------------------
class TestCalculateTRP:
    """Tests for calculate_trp (Total Radiated Power)."""

    @staticmethod
    def _make_uniform_sphere():
        """Build deterministic uniform power sphere for TRP tests.

        Returns power_dBm_2d (theta x phi), theta_angles_rad, inc_theta, inc_phi.
        """
        inc_theta = 15
        inc_phi = 15
        theta_deg = np.arange(0, 180 + inc_theta, inc_theta)
        phi_deg = np.arange(0, 360, inc_phi)
        theta_rad = np.deg2rad(theta_deg)
        # Uniform 0 dBm at every point
        power_dBm_2d = np.zeros((len(theta_deg), len(phi_deg)))
        return power_dBm_2d, theta_rad, inc_theta, inc_phi

    def test_uniform_power(self):
        """Uniform power distribution returns a finite float"""
        power, theta_rad, inc_theta, inc_phi = self._make_uniform_sphere()
        result = calculate_trp(power, theta_rad, inc_theta, inc_phi)
        assert isinstance(result, (float, np.floating))
        assert np.isfinite(result)

    def test_trp_returns_finite(self):
        """TRP result is finite (not inf or nan) for a non-trivial input"""
        inc_theta = 10
        inc_phi = 10
        theta_deg = np.arange(0, 180 + inc_theta, inc_theta)
        phi_deg = np.arange(0, 360, inc_phi)
        theta_rad = np.deg2rad(theta_deg)
        # Deterministic varying power pattern: linearly spaced values
        power_dBm_2d = np.linspace(-20, 10, len(theta_deg) * len(phi_deg)).reshape(
            len(theta_deg), len(phi_deg)
        )
        result = calculate_trp(power_dBm_2d, theta_rad, inc_theta, inc_phi)
        assert np.isfinite(result)


# ---------------------------------------------------------------------------
# 6. TestProcessData
# ---------------------------------------------------------------------------
class TestProcessData:
    """Tests for process_data (FFT-based far-field processing)."""

    @staticmethod
    def _make_deterministic_data():
        """Build a small deterministic 2-D gain grid for process_data tests.

        Returns selected_data, phi_angles_deg, theta_angles_deg.
        """
        theta_deg = np.arange(0, 181, 15, dtype=float)   # 13 points
        phi_deg = np.arange(0, 360, 15, dtype=float)      # 24 points
        # Deterministic data: outer-product pattern (no randomness)
        selected_data = np.outer(
            np.cos(np.deg2rad(theta_deg)),
            np.sin(np.deg2rad(phi_deg)),
        )
        return selected_data, phi_deg, theta_deg

    def test_returns_mag_and_phase(self):
        """process_data returns a tuple of two arrays (magnitude and phase)"""
        data, phi_deg, theta_deg = self._make_deterministic_data()
        result = process_data(data, phi_deg, theta_deg)
        assert isinstance(result, tuple)
        assert len(result) == 2
        mag, phase = result
        assert isinstance(mag, np.ndarray)
        assert isinstance(phase, np.ndarray)

    def test_output_shapes(self):
        """Output array shapes must match the input data shape"""
        data, phi_deg, theta_deg = self._make_deterministic_data()
        mag, phase = process_data(data, phi_deg, theta_deg)
        assert mag.shape == data.shape
        assert phase.shape == data.shape
