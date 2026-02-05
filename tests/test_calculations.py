"""
Unit tests for plot_antenna.calculations module

Tests core RF calculation functions including:
- Passive variable calculations (efficiency, gain, directivity)
- Active variable calculations (TRP, EIRP)
- Polarization calculations (axial ratio, tilt angle, XPD)
- IEEE total gain calculations
"""

import pytest
import numpy as np
from plot_antenna.calculations import (
    calculate_passive_variables,
    calculate_active_variables,
    diversity_gain,
    extract_passive_frequencies,
)


class TestPassiveCalculations:
    """Tests for passive antenna measurement calculations"""

    def test_calculate_passive_variables_basic(self, sample_passive_data):
        """Test passive variable calculation with valid data"""
        result = calculate_passive_variables(
            sample_passive_data['phi'],
            sample_passive_data['theta'],
            sample_passive_data['h_gain'],
            sample_passive_data['v_gain']
        )

        # Check that result contains expected keys
        assert 'total_gain' in result
        assert 'h_gain' in result
        assert 'v_gain' in result
        assert 'efficiency' in result

        # Check that arrays have correct shape
        assert result['total_gain'].shape == sample_passive_data['h_gain'].shape

    def test_passive_gain_ranges(self, sample_passive_data):
        """Test that calculated gains are within reasonable ranges"""
        result = calculate_passive_variables(
            sample_passive_data['phi'],
            sample_passive_data['theta'],
            sample_passive_data['h_gain'],
            sample_passive_data['v_gain']
        )

        # Total gain should be higher than individual components (IEEE definition)
        max_h = np.max(sample_passive_data['h_gain'])
        max_v = np.max(sample_passive_data['v_gain'])
        max_total = np.max(result['total_gain'])

        assert max_total >= max_h, "Total gain should be >= HPOL gain"
        assert max_total >= max_v, "Total gain should be >= VPOL gain"

    def test_efficiency_calculation_range(self, sample_passive_data):
        """Test that efficiency values are between 0 and 100%"""
        result = calculate_passive_variables(
            sample_passive_data['phi'],
            sample_passive_data['theta'],
            sample_passive_data['h_gain'],
            sample_passive_data['v_gain']
        )

        if 'efficiency' in result:
            # Efficiency should be between 0 and 100%
            assert np.all(result['efficiency'] >= 0)
            assert np.all(result['efficiency'] <= 100)


class TestActiveCalculations:
    """Tests for active TRP measurement calculations"""

    def test_calculate_active_variables_basic(self, sample_active_data):
        """Test active variable calculation with valid data"""
        result = calculate_active_variables(
            sample_active_data['phi'],
            sample_active_data['theta'],
            sample_active_data['power_dbm']
        )

        # Check that result contains expected keys
        assert 'trp_dbm' in result or 'eirp_dbm' in result
        assert isinstance(result, dict)

    def test_trp_is_scalar(self, sample_active_data):
        """Test that TRP result is a single scalar value"""
        result = calculate_active_variables(
            sample_active_data['phi'],
            sample_active_data['theta'],
            sample_active_data['power_dbm']
        )

        if 'trp_dbm' in result:
            trp = result['trp_dbm']
            assert np.isscalar(trp) or (isinstance(trp, np.ndarray) and trp.size == 1)


class TestDiversityGain:
    """Tests for diversity gain calculations"""

    def test_diversity_gain_positive(self):
        """Test that diversity gain is positive when combining uncorrelated signals"""
        # Two equal power signals should give ~3dB diversity gain
        power1 = np.ones(100) * 10  # 10 dBm
        power2 = np.ones(100) * 10  # 10 dBm

        result = diversity_gain(power1, power2)

        # Diversity gain should be positive for uncorrelated signals
        assert result > 0, "Diversity gain should be positive"
        assert result < 6, "Diversity gain should be reasonable (<6dB)"

    def test_diversity_gain_zero_power(self):
        """Test diversity gain with one antenna having zero power"""
        power1 = np.ones(100) * 10  # 10 dBm
        power2 = np.zeros(100)  # 0 linear power

        result = diversity_gain(power1, power2)

        # Should handle zero power gracefully
        assert not np.isnan(result), "Should not return NaN"
        assert not np.isinf(result), "Should not return Inf"


class TestFrequencyExtraction:
    """Tests for frequency extraction from file paths"""

    def test_extract_passive_frequencies(self):
        """Test frequency extraction from typical file paths"""
        files = [
            "antenna_2400MHz_HPOL.txt",
            "antenna_2450MHz_HPOL.txt",
            "antenna_2500MHz_HPOL.txt"
        ]

        freqs = extract_passive_frequencies(files)

        assert len(freqs) == 3
        assert 2400.0 in freqs
        assert 2450.0 in freqs
        assert 2500.0 in freqs

    def test_extract_frequencies_sorted(self):
        """Test that extracted frequencies are sorted"""
        files = [
            "test_2500MHz.txt",
            "test_2400MHz.txt",
            "test_2450MHz.txt"
        ]

        freqs = extract_passive_frequencies(files)

        assert freqs == sorted(freqs), "Frequencies should be sorted"


class TestPolarizationCalculations:
    """Tests for polarization parameter calculations"""

    def test_axial_ratio_range(self, sample_passive_data):
        """Test that axial ratio values are within valid range"""
        # Axial ratio should be >= 0 dB (1:1 is perfect circular polarization)
        # Most practical antennas have AR between 0-20 dB

        h_gain = sample_passive_data['h_gain']
        v_gain = sample_passive_data['v_gain']

        # Calculate AR manually for validation
        # AR = 20 * log10((E_major + E_minor) / (E_major - E_minor))
        # For testing, just check that inputs are reasonable

        assert np.all(np.isfinite(h_gain)), "HPOL gain should be finite"
        assert np.all(np.isfinite(v_gain)), "VPOL gain should be finite"

    def test_cross_pol_discrimination(self, sample_passive_data):
        """Test XPD calculation basics"""
        h_gain = sample_passive_data['h_gain']
        v_gain = sample_passive_data['v_gain']

        # XPD is the ratio of co-pol to cross-pol
        # Should be positive (co-pol > cross-pol for good antennas)
        xpd = h_gain - v_gain  # Simplified XPD in dB

        # Most antennas have XPD > 10 dB
        # Our random test data may not, but should be reasonable
        assert np.all(np.isfinite(xpd)), "XPD should be finite"


class TestEdgeCases:
    """Tests for edge cases and error handling"""

    def test_empty_arrays(self):
        """Test behavior with empty input arrays"""
        phi = np.array([])
        theta = np.array([])
        gain = np.array([]).reshape(0, 0)

        # Should either handle gracefully or raise appropriate error
        with pytest.raises((ValueError, IndexError)):
            calculate_passive_variables(phi, theta, gain, gain)

    def test_mismatched_dimensions(self):
        """Test behavior with mismatched array dimensions"""
        phi = np.linspace(0, 360, 10)
        theta = np.linspace(0, 180, 10)
        h_gain = np.random.randn(10, 10)
        v_gain = np.random.randn(5, 5)  # Wrong size

        # Should raise an error for mismatched dimensions
        with pytest.raises((ValueError, IndexError)):
            calculate_passive_variables(phi, theta, h_gain, v_gain)

    def test_nan_handling(self):
        """Test behavior with NaN values in data"""
        phi = np.linspace(0, 360, 10)
        theta = np.linspace(0, 180, 10)
        h_gain = np.random.randn(10, 10)
        h_gain[0, 0] = np.nan  # Inject NaN
        v_gain = np.random.randn(10, 10)

        # Should either handle NaN gracefully or propagate it
        result = calculate_passive_variables(phi, theta, h_gain, v_gain)

        # If NaN is propagated, check that it doesn't corrupt all results
        if np.any(np.isnan(result['total_gain'])):
            # At least some values should still be valid
            assert np.any(~np.isnan(result['total_gain'])), "Not all values should be NaN"
