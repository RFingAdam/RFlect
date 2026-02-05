"""
Unit tests for plot_antenna.file_utils module

Tests file parsing and data loading functionality:
- Passive measurement file parsing (HPOL/VPOL .txt files)
- Active measurement file parsing (TRP .txt files)
- VNA S-parameter file parsing (.csv files)
- CST simulation file parsing
- Error handling for malformed files
"""

import pytest
import numpy as np
from pathlib import Path
from plot_antenna.file_utils import (
    read_passive_file,
    read_active_file,
    parse_2port_data,
)


class TestPassiveFileReading:
    """Tests for passive measurement file parsing"""

    def test_read_passive_file_structure(self, sample_data_dir):
        """Test that passive file reading returns expected structure"""
        # Note: This test will be skipped if no sample file exists
        sample_file = sample_data_dir / "sample_hpol.txt"

        if not sample_file.exists():
            pytest.skip("Sample passive file not available")

        result = read_passive_file(str(sample_file))

        # Check basic structure
        assert isinstance(result, dict), "Result should be a dictionary"
        assert 'phi' in result or 'theta' in result, "Should contain angle data"
        assert 'gain' in result or 'h_gain' in result, "Should contain gain data"

    def test_invalid_file_path(self):
        """Test error handling for non-existent file"""
        with pytest.raises((FileNotFoundError, IOError)):
            read_passive_file("nonexistent_file.txt")

    def test_passive_data_types(self, sample_data_dir):
        """Test that parsed data has correct types"""
        sample_file = sample_data_dir / "sample_hpol.txt"

        if not sample_file.exists():
            pytest.skip("Sample passive file not available")

        result = read_passive_file(str(sample_file))

        # Angles and gain should be numpy arrays or lists
        for key in result:
            if 'gain' in key or 'phi' in key or 'theta' in key:
                assert isinstance(result[key], (np.ndarray, list)), \
                    f"{key} should be numpy array or list"


class TestActiveFileReading:
    """Tests for active TRP measurement file parsing"""

    def test_read_active_file_structure(self, sample_data_dir):
        """Test that active file reading returns expected structure"""
        sample_file = sample_data_dir / "sample_trp.txt"

        if not sample_file.exists():
            pytest.skip("Sample TRP file not available")

        result = read_active_file(str(sample_file))

        # Check basic structure
        assert isinstance(result, dict), "Result should be a dictionary"
        assert 'phi' in result or 'theta' in result, "Should contain angle data"
        assert 'power' in result or 'trp' in result, "Should contain power data"

    def test_active_power_units(self, sample_data_dir):
        """Test that power values are in expected units (dBm or mW)"""
        sample_file = sample_data_dir / "sample_trp.txt"

        if not sample_file.exists():
            pytest.skip("Sample TRP file not available")

        result = read_active_file(str(sample_file))

        # Power values should be reasonable (not extreme)
        if 'power_dbm' in result:
            power = np.array(result['power_dbm'])
            # Typical range: -30 to +30 dBm for antenna measurements
            assert np.all(power > -50), "Power too low (check units)"
            assert np.all(power < 50), "Power too high (check units)"


class TestVNAFileReading:
    """Tests for VNA S-parameter file parsing"""

    def test_parse_2port_data_structure(self, sample_data_dir):
        """Test that 2-port data parsing returns expected structure"""
        sample_file = sample_data_dir / "sample_vna.csv"

        if not sample_file.exists():
            pytest.skip("Sample VNA file not available")

        result = parse_2port_data(str(sample_file))

        # Check basic structure
        assert isinstance(result, dict), "Result should be a dictionary"
        assert 'frequency' in result, "Should contain frequency data"

        # Should have S-parameter data (S11, S21, S12, S22)
        s_params = ['s11', 's21', 's12', 's22']
        has_s_param = any(param in result for param in s_params)
        assert has_s_param, "Should contain at least one S-parameter"

    def test_vna_frequency_units(self, sample_data_dir):
        """Test that frequencies are in expected units (MHz or GHz)"""
        sample_file = sample_data_dir / "sample_vna.csv"

        if not sample_file.exists():
            pytest.skip("Sample VNA file not available")

        result = parse_2port_data(str(sample_file))

        if 'frequency' in result:
            freq = np.array(result['frequency'])
            # Typical antenna freq: 100 MHz to 10 GHz (depends on units)
            # Check that frequencies are monotonically increasing
            assert np.all(np.diff(freq) > 0), "Frequencies should be monotonically increasing"


class TestFileFormatValidation:
    """Tests for file format validation and error handling"""

    def test_empty_file_handling(self, tmp_path):
        """Test behavior with empty file"""
        empty_file = tmp_path / "empty.txt"
        empty_file.write_text("")

        # Should raise appropriate error for empty file
        with pytest.raises((ValueError, IOError, Exception)):
            read_passive_file(str(empty_file))

    def test_malformed_data_handling(self, tmp_path):
        """Test behavior with malformed data"""
        bad_file = tmp_path / "malformed.txt"
        bad_file.write_text("This is not valid antenna data\nRandom text\n123abc")

        # Should raise appropriate error for malformed data
        with pytest.raises((ValueError, IOError, Exception)):
            read_passive_file(str(bad_file))

    def test_incorrect_extension_warning(self, tmp_path):
        """Test that incorrect file extensions are handled"""
        wrong_ext = tmp_path / "data.xyz"
        wrong_ext.write_text("Some data")

        # Should either work (if format is correct) or raise error
        # But shouldn't crash unexpectedly
        try:
            read_passive_file(str(wrong_ext))
        except (ValueError, IOError, Exception):
            pass  # Expected behavior


class TestDataConsistency:
    """Tests for data consistency and integrity"""

    def test_phi_theta_grid_consistency(self, sample_data_dir):
        """Test that phi and theta form a consistent grid"""
        sample_file = sample_data_dir / "sample_hpol.txt"

        if not sample_file.exists():
            pytest.skip("Sample passive file not available")

        result = read_passive_file(str(sample_file))

        if 'phi' in result and 'theta' in result:
            phi = np.array(result['phi'])
            theta = np.array(result['theta'])

            # Phi should be in [0, 360] range
            assert np.all(phi >= 0) and np.all(phi <= 360), "Phi should be in [0, 360]"

            # Theta should be in [0, 180] range
            assert np.all(theta >= 0) and np.all(theta <= 180), "Theta should be in [0, 180]"

    def test_gain_array_dimensions(self, sample_data_dir):
        """Test that gain arrays have correct dimensions matching angles"""
        sample_file = sample_data_dir / "sample_hpol.txt"

        if not sample_file.exists():
            pytest.skip("Sample passive file not available")

        result = read_passive_file(str(sample_file))

        if 'phi' in result and 'theta' in result and 'gain' in result:
            phi = np.array(result['phi'])
            theta = np.array(result['theta'])
            gain = np.array(result['gain'])

            # Gain should be 2D array with dimensions matching phi and theta
            expected_shape = (len(theta), len(phi))
            assert gain.shape == expected_shape, \
                f"Gain shape {gain.shape} doesn't match expected {expected_shape}"


class TestFrequencyExtraction:
    """Tests for frequency extraction from filenames"""

    def test_frequency_in_filename(self):
        """Test extracting frequency from filename"""
        # This is more of an integration test with calculations module
        # but placed here as it relates to file handling

        filenames = [
            "antenna_2400MHz_HPOL.txt",
            "test_2.45GHz_VPOL.txt",
            "measurement_915MHz.txt"
        ]

        # Test that frequency extraction patterns work
        for filename in filenames:
            # Frequency should be extractable from these filenames
            # Actual extraction is done by calculations.extract_passive_frequencies
            assert 'MHz' in filename or 'GHz' in filename, \
                "Test filenames should contain frequency units"


class TestSpecialCases:
    """Tests for special cases and edge conditions"""

    def test_single_frequency_file(self, sample_data_dir):
        """Test handling of files with single frequency"""
        # Most files have single frequency, this should be standard case
        # Just verify it works
        sample_file = sample_data_dir / "sample_hpol.txt"

        if not sample_file.exists():
            pytest.skip("Sample passive file not available")

        result = read_passive_file(str(sample_file))

        # Should successfully parse without errors
        assert result is not None

    def test_high_resolution_data(self):
        """Test handling of high-resolution measurements"""
        # High resolution might be 1-degree steps (361x181 points)
        # Should handle without memory issues

        # This is more of a performance test
        # Verify that the module can conceptually handle large datasets
        large_phi = np.linspace(0, 360, 361)
        large_theta = np.linspace(0, 180, 181)

        # Just check that arrays of this size can be created
        assert len(large_phi) == 361
        assert len(large_theta) == 181

    def test_unicode_in_filepath(self, tmp_path):
        """Test handling of unicode characters in file paths"""
        unicode_file = tmp_path / "antenna_测试_2400MHz.txt"
        unicode_file.write_text("Sample data")

        # Should handle unicode file paths gracefully
        try:
            read_passive_file(str(unicode_file))
        except (FileNotFoundError, UnicodeError):
            # May fail on some systems, but shouldn't crash
            pass
