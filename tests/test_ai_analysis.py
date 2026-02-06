"""
Unit tests for plot_antenna.ai_analysis module

Tests cover:
- AntennaAnalyzer initialization for passive and active scans
- Gain/power statistics calculation
- Radiation pattern analysis and classification
- Polarization comparison
- Utility functions (domain knowledge, system prompts)

All test data uses deterministic numpy arrays with known properties.
"""

import pytest
import numpy as np

from plot_antenna.ai_analysis import (
    AntennaAnalyzer,
    get_antenna_domain_knowledge,
    create_analysis_system_prompt,
)


# ---------------------------------------------------------------------------
# Deterministic test data helpers
# ---------------------------------------------------------------------------

def _make_passive_data_1d():
    """Create passive measurement data with 1D gain arrays (single frequency)."""
    num_points = 37  # 10-degree phi steps
    total_gain = np.linspace(-5.0, 10.0, num_points)  # known min=-5, max=10
    h_gain = np.linspace(-3.0, 8.0, num_points)
    v_gain = np.linspace(-6.0, 6.0, num_points)
    return {
        "phi": np.linspace(0, 360, num_points),
        "theta": np.linspace(0, 180, 19),
        "total_gain": total_gain,
        "h_gain": h_gain,
        "v_gain": v_gain,
    }


def _make_passive_data_2d(num_points=37, num_freqs=3):
    """Create passive measurement data with 2D gain arrays (multiple frequencies).

    Shape: (num_points, num_freqs)
    """
    total_gain = np.column_stack(
        [np.linspace(-5.0, 10.0, num_points) + i for i in range(num_freqs)]
    )
    h_gain = np.column_stack(
        [np.linspace(-3.0, 8.0, num_points) + i for i in range(num_freqs)]
    )
    v_gain = np.column_stack(
        [np.linspace(-6.0, 6.0, num_points) + i for i in range(num_freqs)]
    )
    return {
        "phi": np.linspace(0, 360, num_points),
        "theta": np.linspace(0, 180, 19),
        "total_gain": total_gain,
        "h_gain": h_gain,
        "v_gain": v_gain,
    }


def _make_active_data_1d():
    """Create active measurement data with 1D power array."""
    num_points = 37
    total_power = np.linspace(-10.0, 5.0, num_points)
    return {
        "phi": np.linspace(0, 360, num_points),
        "theta": np.linspace(0, 180, 19),
        "total_power": total_power,
        "TRP_dBm": 2.5,
        "h_TRP_dBm": 1.0,
        "v_TRP_dBm": 0.8,
    }


# ---------------------------------------------------------------------------
# TestAntennaAnalyzerInit
# ---------------------------------------------------------------------------

class TestAntennaAnalyzerInit:
    """Test AntennaAnalyzer construction for different scan types."""

    def test_init_passive_scan(self):
        """Verify that a passive-scan analyzer stores data, scan_type, and frequencies."""
        data = _make_passive_data_1d()
        freqs = [2400.0, 2450.0, 2500.0]
        analyzer = AntennaAnalyzer(data, scan_type="passive", frequencies=freqs)

        assert analyzer.data is data
        assert analyzer.scan_type == "passive"
        assert analyzer.frequencies == freqs

    def test_init_active_scan(self):
        """Verify that an active-scan analyzer stores data, scan_type, and frequencies."""
        data = _make_active_data_1d()
        freqs = [2400.0]
        analyzer = AntennaAnalyzer(data, scan_type="active", frequencies=freqs)

        assert analyzer.data is data
        assert analyzer.scan_type == "active"
        assert analyzer.frequencies == freqs


# ---------------------------------------------------------------------------
# TestGainStatistics
# ---------------------------------------------------------------------------

class TestGainStatistics:
    """Test get_gain_statistics for passive and active data."""

    def test_passive_gain_stats_basic(self):
        """Check that all expected keys are present and values are correct for 1D passive data."""
        data = _make_passive_data_1d()
        analyzer = AntennaAnalyzer(data, scan_type="passive", frequencies=[2400.0])
        stats = analyzer.get_gain_statistics(frequency=2400.0)

        expected_keys = {
            "frequency_requested",
            "frequency_actual",
            "scan_type",
            "max_gain_dBi",
            "min_gain_dBi",
            "avg_gain_dBi",
            "std_dev_dB",
        }
        assert expected_keys.issubset(stats.keys())
        assert stats["scan_type"] == "passive"
        assert stats["max_gain_dBi"] == pytest.approx(10.0)
        assert stats["min_gain_dBi"] == pytest.approx(-5.0)
        assert stats["avg_gain_dBi"] == pytest.approx(2.5, abs=0.01)

    def test_passive_gain_stats_with_frequency(self):
        """Check that 2D data is sliced by the correct frequency index."""
        data = _make_passive_data_2d(num_points=37, num_freqs=3)
        freqs = [2400.0, 2450.0, 2500.0]
        analyzer = AntennaAnalyzer(data, scan_type="passive", frequencies=freqs)

        # Request the second frequency (index 1); gain offset is +1 relative to base
        stats = analyzer.get_gain_statistics(frequency=2450.0)
        assert stats["frequency_actual"] == 2450.0
        # Base range is -5..10; with offset +1 the range becomes -4..11
        assert stats["max_gain_dBi"] == pytest.approx(11.0)
        assert stats["min_gain_dBi"] == pytest.approx(-4.0)

    def test_passive_gain_stats_with_polarization(self):
        """Verify polarization-specific stats are included when h_gain/v_gain are present."""
        data = _make_passive_data_1d()
        analyzer = AntennaAnalyzer(data, scan_type="passive", frequencies=[2400.0])
        stats = analyzer.get_gain_statistics(frequency=2400.0)

        assert "max_hpol_gain_dBi" in stats
        assert "max_vpol_gain_dBi" in stats
        assert stats["max_hpol_gain_dBi"] == pytest.approx(8.0)
        assert stats["max_vpol_gain_dBi"] == pytest.approx(6.0)

    def test_active_power_stats_basic(self):
        """Check that active-scan statistics contain power keys with correct values."""
        data = _make_active_data_1d()
        analyzer = AntennaAnalyzer(data, scan_type="active", frequencies=[2400.0])
        stats = analyzer.get_gain_statistics(frequency=2400.0)

        expected_keys = {
            "max_power_dBm",
            "min_power_dBm",
            "avg_power_dBm",
            "std_dev_dB",
        }
        assert expected_keys.issubset(stats.keys())
        assert stats["scan_type"] == "active"
        assert stats["max_power_dBm"] == pytest.approx(5.0)
        assert stats["min_power_dBm"] == pytest.approx(-10.0)

    def test_active_trp_stats(self):
        """Verify that TRP values are included when present in the measurement data."""
        data = _make_active_data_1d()
        analyzer = AntennaAnalyzer(data, scan_type="active", frequencies=[2400.0])
        stats = analyzer.get_gain_statistics(frequency=2400.0)

        assert stats["TRP_dBm"] == pytest.approx(2.5)
        assert stats["h_TRP_dBm"] == pytest.approx(1.0)
        assert stats["v_TRP_dBm"] == pytest.approx(0.8)


# ---------------------------------------------------------------------------
# TestPatternAnalysis
# ---------------------------------------------------------------------------

class TestPatternAnalysis:
    """Test analyze_pattern for pattern classification and null detection."""

    def test_omnidirectional_classification(self):
        """A flat gain array (range < 6 dB) should be classified as omnidirectional."""
        num_points = 37
        # Constant gain with a very small spread (max - min = 2 dB)
        gain = np.full(num_points, 5.0)
        gain[0] = 4.0   # min
        gain[-1] = 6.0   # max -> range = 2 dB
        data = {
            "phi": np.linspace(0, 360, num_points),
            "theta": np.linspace(0, 180, 19),
            "total_gain": gain,
        }
        analyzer = AntennaAnalyzer(data, scan_type="passive", frequencies=[2400.0])
        result = analyzer.analyze_pattern(frequency=2400.0)

        assert result["pattern_type"] == "omnidirectional"

    def test_sectoral_classification(self):
        """A gain array with 6 <= range < 12 dB should be classified as sectoral."""
        num_points = 37
        gain = np.full(num_points, 5.0)
        gain[0] = 2.0    # min
        gain[-1] = 10.0   # max -> range = 8 dB
        data = {
            "phi": np.linspace(0, 360, num_points),
            "theta": np.linspace(0, 180, 19),
            "total_gain": gain,
        }
        analyzer = AntennaAnalyzer(data, scan_type="passive", frequencies=[2400.0])
        result = analyzer.analyze_pattern(frequency=2400.0)

        assert result["pattern_type"] == "sectoral"

    def test_directional_classification(self):
        """A gain array with range >= 12 dB should be classified as directional."""
        num_points = 37
        gain = np.full(num_points, 0.0)
        gain[0] = -5.0    # min
        gain[-1] = 10.0   # max -> range = 15 dB
        data = {
            "phi": np.linspace(0, 360, num_points),
            "theta": np.linspace(0, 180, 19),
            "total_gain": gain,
        }
        analyzer = AntennaAnalyzer(data, scan_type="passive", frequencies=[2400.0])
        result = analyzer.analyze_pattern(frequency=2400.0)

        assert result["pattern_type"] == "directional"

    def test_null_detection(self):
        """Points more than 10 dB below peak should be counted as nulls."""
        num_points = 37
        # Peak at 10 dBi, null threshold = 10 - 10 = 0 dBi
        gain = np.full(num_points, 5.0)  # all at 5 dBi (above threshold)
        gain[18] = 10.0                  # peak
        gain[0] = -5.0                   # null (below 0)
        gain[1] = -2.0                   # null (below 0)
        gain[2] = -1.0                   # null (below 0)
        data = {
            "phi": np.linspace(0, 360, num_points),
            "theta": np.linspace(0, 180, 19),
            "total_gain": gain,
        }
        analyzer = AntennaAnalyzer(data, scan_type="passive", frequencies=[2400.0])
        result = analyzer.analyze_pattern(frequency=2400.0)

        assert result["num_nulls"] == 3
        assert result["deepest_null_dB"] == pytest.approx(-5.0)

    def test_peak_gain_value(self):
        """The reported peak_gain_dBi should match the maximum of the gain array."""
        num_points = 37
        gain = np.linspace(-10.0, 15.0, num_points)
        data = {
            "phi": np.linspace(0, 360, num_points),
            "theta": np.linspace(0, 180, 19),
            "total_gain": gain,
        }
        analyzer = AntennaAnalyzer(data, scan_type="passive", frequencies=[2400.0])
        result = analyzer.analyze_pattern(frequency=2400.0)

        assert result["peak_gain_dBi"] == pytest.approx(15.0)


# ---------------------------------------------------------------------------
# TestPolarizationComparison
# ---------------------------------------------------------------------------

class TestPolarizationComparison:
    """Test compare_polarizations for balanced, imbalanced, and XPD calculations."""

    def test_balanced_polarization(self):
        """When max H-pol and V-pol gains are within 3 dB, note should say well-balanced."""
        num_points = 37
        h_gain = np.full(num_points, 8.0)
        v_gain = np.full(num_points, 7.0)  # 1 dB difference
        data = {
            "phi": np.linspace(0, 360, num_points),
            "theta": np.linspace(0, 180, 19),
            "h_gain": h_gain,
            "v_gain": v_gain,
        }
        analyzer = AntennaAnalyzer(data, scan_type="passive", frequencies=[2400.0])
        result = analyzer.compare_polarizations(frequency=2400.0)

        assert result["polarization_balance_dB"] == pytest.approx(1.0)
        assert result["polarization_note"] == "Well-balanced polarization"

    def test_imbalanced_polarization(self):
        """When the balance exceeds 6 dB, note should indicate significant imbalance."""
        num_points = 37
        h_gain = np.full(num_points, 12.0)
        v_gain = np.full(num_points, 3.0)  # 9 dB difference
        data = {
            "phi": np.linspace(0, 360, num_points),
            "theta": np.linspace(0, 180, 19),
            "h_gain": h_gain,
            "v_gain": v_gain,
        }
        analyzer = AntennaAnalyzer(data, scan_type="passive", frequencies=[2400.0])
        result = analyzer.compare_polarizations(frequency=2400.0)

        assert result["polarization_balance_dB"] == pytest.approx(9.0)
        assert result["polarization_note"] == "Significant polarization imbalance"

    def test_xpd_calculation(self):
        """Verify avg_xpd_dB is the mean of |h_gain - v_gain| across all points."""
        num_points = 5
        h_gain = np.array([10.0, 8.0, 6.0, 4.0, 2.0])
        v_gain = np.array([5.0, 5.0, 5.0, 5.0, 5.0])
        # |h - v| = [5, 3, 1, 1, 3], mean = 2.6
        data = {
            "phi": np.linspace(0, 360, num_points),
            "theta": np.linspace(0, 180, 3),
            "h_gain": h_gain,
            "v_gain": v_gain,
        }
        analyzer = AntennaAnalyzer(data, scan_type="passive", frequencies=[2400.0])
        result = analyzer.compare_polarizations(frequency=2400.0)

        expected_avg_xpd = np.mean(np.abs(h_gain - v_gain))
        assert result["avg_xpd_dB"] == pytest.approx(expected_avg_xpd)
        assert result["max_hpol_gain_dBi"] == pytest.approx(10.0)
        assert result["max_vpol_gain_dBi"] == pytest.approx(5.0)


# ---------------------------------------------------------------------------
# TestAnalyzeAllFrequencies
# ---------------------------------------------------------------------------

class TestAnalyzeAllFrequencies:
    """Test analyze_all_frequencies across multiple frequency points."""

    def test_multi_frequency_analysis(self):
        """Verify resonance detection, gain variation, and bandwidth across frequencies."""
        num_points = 37
        # Build 2D gain: freq 0 peaks at 10, freq 1 peaks at 12, freq 2 peaks at 11
        col0 = np.full(num_points, 5.0); col0[0] = 10.0
        col1 = np.full(num_points, 5.0); col1[0] = 12.0
        col2 = np.full(num_points, 5.0); col2[0] = 11.0
        total_gain = np.column_stack([col0, col1, col2])
        data = {
            "phi": np.linspace(0, 360, num_points),
            "theta": np.linspace(0, 180, 19),
            "total_gain": total_gain,
        }
        freqs = [2400.0, 2450.0, 2500.0]
        analyzer = AntennaAnalyzer(data, scan_type="passive", frequencies=freqs)
        result = analyzer.analyze_all_frequencies()

        assert result["num_frequencies"] == 3
        assert result["frequencies_MHz"] == freqs
        assert result["resonance_frequency_MHz"] == 2450.0
        assert result["peak_gain_at_resonance_dBi"] == pytest.approx(12.0)
        assert result["gain_variation_dB"] == pytest.approx(2.0)
        assert len(result["peak_gain_per_freq"]) == 3


# ---------------------------------------------------------------------------
# TestUtilityFunctions
# ---------------------------------------------------------------------------

class TestUtilityFunctions:
    """Test standalone utility functions for AI prompts."""

    def test_domain_knowledge_returns_string(self):
        """get_antenna_domain_knowledge should return a non-empty string with reference data."""
        result = get_antenna_domain_knowledge()

        assert isinstance(result, str)
        assert len(result) > 0
        # Verify it contains key antenna benchmarks
        assert "dBi" in result
        assert "dipole" in result.lower() or "Dipole" in result

    def test_system_prompt_includes_scan_type(self):
        """create_analysis_system_prompt should embed the scan_type in the returned prompt."""
        for scan_type in ("passive", "active", "vna"):
            prompt = create_analysis_system_prompt(scan_type)

            assert isinstance(prompt, str)
            assert scan_type in prompt
            # Should also embed domain knowledge
            assert "dBi" in prompt
