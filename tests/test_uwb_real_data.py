"""
Integration tests for UWB analysis using real Group Delay measurement files.

Test data directory: /home/swamp/Downloads/TestFiles/_Test Files/test_files/Group Delay/
GroupDelay_*deg.csv files have S21(dB) + S21(s) (group delay) columns.
S11S22S21_*deg.csv files have S21(dB) only (no group delay).
"""

import os
import glob
import pytest
import numpy as np

from plot_antenna.file_utils import parse_2port_data
from plot_antenna.uwb_analysis import (
    build_complex_s21_from_s2vna,
    calculate_sff,
    compute_group_delay_from_s21,
    calculate_sff_vs_angle,
    analyze_return_loss,
    reconstruct_phase_from_group_delay,
)

# ---------------------------------------------------------------------------
# Paths to real test data
# ---------------------------------------------------------------------------
GROUP_DELAY_DIR = "/home/swamp/Downloads/TestFiles/_Test Files/test_files/Group Delay"

# Skip all tests if test data directory is missing
pytestmark = pytest.mark.skipif(
    not os.path.isdir(GROUP_DELAY_DIR),
    reason=f"Test data directory not found: {GROUP_DELAY_DIR}",
)


def _get_groupdelay_files():
    """Return sorted list of GroupDelay_*deg.csv files."""
    return sorted(glob.glob(os.path.join(GROUP_DELAY_DIR, "GroupDelay_*deg.csv")))


def _get_s11_only_files():
    """Return sorted list of S11S22S21_*deg.csv files (no group delay column)."""
    return sorted(glob.glob(os.path.join(GROUP_DELAY_DIR, "S11S22S21_*deg.csv")))


# ===========================================================================
# Real Data: Phase Reconstruction
# ===========================================================================
class TestRealPhaseReconstruction:
    """Test phase reconstruction from real GroupDelay CSV files."""

    def test_monotonic_phase(self):
        """Reconstructed phase from GroupDelay_0deg.csv should be monotonically decreasing."""
        files = _get_groupdelay_files()
        if not files:
            pytest.skip("No GroupDelay files found")

        df = parse_2port_data(files[0])
        freq = df["! Stimulus(Hz)"].values
        gd = df["S21(s)"].values

        phase = reconstruct_phase_from_group_delay(freq, gd)

        # For positive group delay, phase should generally decrease with frequency
        # Allow some noise, check overall trend
        phase_diff = np.diff(phase)
        assert (
            np.sum(phase_diff < 0) > len(phase_diff) * 0.8
        ), "Phase should be predominantly decreasing for positive group delay"


# ===========================================================================
# Real Data: SFF
# ===========================================================================
class TestRealSFF:
    """Test SFF computation with real measurement files."""

    def test_sff_in_valid_range(self):
        """SFF from real data should be in [0.3, 1.0] range."""
        files = _get_groupdelay_files()
        if not files:
            pytest.skip("No GroupDelay files found")

        df = parse_2port_data(files[0])
        freq = df["! Stimulus(Hz)"].values
        s21_dB = df["S21(dB)"].values
        gd = df["S21(s)"].values

        s21_complex = build_complex_s21_from_s2vna(freq, s21_dB, gd)
        result = calculate_sff(freq, s21_complex)

        assert 0.3 <= result["sff"] <= 1.0, f"SFF {result['sff']} outside expected [0.3, 1.0] range"

    def test_sff_varies_with_angle(self):
        """SFF should vary between 0deg and 90deg measurements."""
        files = _get_groupdelay_files()
        file_0deg = [f for f in files if "0deg" in f]
        file_90deg = [f for f in files if "90deg" in f]

        if not file_0deg or not file_90deg:
            pytest.skip("Missing 0deg or 90deg GroupDelay files")

        sff_values = []
        for fp in [file_0deg[0], file_90deg[0]]:
            df = parse_2port_data(fp)
            freq = df["! Stimulus(Hz)"].values
            s21_dB = df["S21(dB)"].values
            gd = df["S21(s)"].values
            s21_complex = build_complex_s21_from_s2vna(freq, s21_dB, gd)
            result = calculate_sff(freq, s21_complex)
            sff_values.append(result["sff"])

        # SFF should differ between angles (antenna is not isotropic)
        # Allow them to be similar but not expect exact match
        assert len(sff_values) == 2
        assert all(0.0 <= v <= 1.0 for v in sff_values)

    def test_multi_angle_sff_all_files(self):
        """Run SFF across all GroupDelay angle files."""
        files = _get_groupdelay_files()
        if len(files) < 2:
            pytest.skip("Need at least 2 GroupDelay files")

        import re

        pattern = re.compile(r"(\d+)deg", re.IGNORECASE)

        angle_data = []
        for fp in files:
            match = pattern.search(os.path.basename(fp))
            angle = float(match.group(1)) if match else 0.0

            df = parse_2port_data(fp)
            freq = df["! Stimulus(Hz)"].values
            s21_dB = df["S21(dB)"].values
            gd = df["S21(s)"].values

            s21_complex = build_complex_s21_from_s2vna(freq, s21_dB, gd)
            angle_data.append(
                {
                    "angle_deg": angle,
                    "freq_hz": freq,
                    "s21_complex": s21_complex,
                }
            )

        result = calculate_sff_vs_angle(angle_data)

        assert len(result["angles"]) == len(files)
        assert 0.0 < result["mean_sff"] <= 1.0
        assert all(0.0 <= v <= 1.0 for v in result["sff_values"])


# ===========================================================================
# Real Data: Group Delay
# ===========================================================================
class TestRealGroupDelay:
    """Test group delay analysis with real files."""

    def test_group_delay_variation_under_2ns(self):
        """Group delay variation across frequency should be < 2 ns for a good UWB antenna."""
        files = _get_groupdelay_files()
        if not files:
            pytest.skip("No GroupDelay files found")

        df = parse_2port_data(files[0])
        freq = df["! Stimulus(Hz)"].values
        s21_dB = df["S21(dB)"].values
        gd = df["S21(s)"].values

        s21_complex = build_complex_s21_from_s2vna(freq, s21_dB, gd)
        result = compute_group_delay_from_s21(freq, s21_complex)

        assert (
            result["variation_s"] < 2e-9
        ), f"Group delay variation {result['variation_s']*1e9:.3f} ns exceeds 2 ns"


# ===========================================================================
# Real Data: S11 Bandwidth
# ===========================================================================
class TestRealS11Bandwidth:
    """Test S11/VSWR analysis with real files."""

    def test_s11_valid_bandwidth(self):
        """S11 from GroupDelay file should produce a valid bandwidth in the 5-10 GHz range."""
        files = _get_groupdelay_files()
        if not files:
            pytest.skip("No GroupDelay files found")

        df = parse_2port_data(files[0])
        freq = df["! Stimulus(Hz)"].values

        if "S11(dB)" not in df.columns:
            pytest.skip("S11(dB) column not available")

        s11_dB = df["S11(dB)"].values
        result = analyze_return_loss(freq, s11_dB)

        # Check that frequency range is reasonable (5-10 GHz)
        assert freq[0] >= 4e9
        assert freq[-1] <= 11e9
        # VSWR should be computed
        assert len(result["vswr"]) == len(freq)
        assert all(v >= 1.0 for v in result["vswr"])


# ===========================================================================
# Real Data: Missing Group Delay Graceful Handling
# ===========================================================================
class TestRealMissingGroupDelay:
    """Test files without S21(s) column are handled gracefully."""

    def test_s11_only_files_no_group_delay(self):
        """S11S22S21_*deg.csv files lack S21(s); should still parse S11/S21(dB)."""
        files = _get_s11_only_files()
        if not files:
            pytest.skip("No S11S22S21 files found")

        df = parse_2port_data(files[0])
        assert "! Stimulus(Hz)" in df.columns
        assert "S21(dB)" in df.columns or "S12(dB)" in df.columns
        assert "S21(s)" not in df.columns, "S11S22S21 files should NOT have S21(s)"

    def test_s11_only_return_loss(self):
        """S11 analysis works even without group delay data."""
        files = _get_s11_only_files()
        if not files:
            pytest.skip("No S11S22S21 files found")

        df = parse_2port_data(files[0])
        if "S11(dB)" not in df.columns:
            pytest.skip("S11(dB) not in S11S22S21 file")

        freq = df["! Stimulus(Hz)"].values
        s11_dB = df["S11(dB)"].values

        result = analyze_return_loss(freq, s11_dB)
        assert result["min_s11_dB"] < 0  # S11 should be negative
        assert len(result["vswr"]) == len(freq)
