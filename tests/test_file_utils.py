"""
Core tests for plot_antenna.file_utils module.
"""

import numpy as np
import pytest

# Verify imports work
from plot_antenna.file_utils import (
    read_passive_file,
    read_active_file,
    parse_2port_data,
)


def _build_minimal_passive_file_text():
    """Build a tiny passive export text that parse_passive_file can consume."""
    lines = [
        "Axis1 Start Angle: 0 Deg",
        "Axis1 Stop Angle: 180 Deg",
        "Axis1 Increment: 180 Deg",
        "Axis2 Start Angle: 0 Deg",
        "Axis2 Stop Angle: 90 Deg",
        "Axis2 Increment: 90 Deg",
        "Cal Std Antenna Peak Gain Factor = 2.5 dB",
        "Test Frequency = 2450 MHz",
        "THETA\t  PHI\t  Mag\t Phase",
        "",
        "0\t0\t1.0\t0.0",
        "0\t180\t2.0\t10.0",
        "90\t0\t3.0\t20.0",
        "90\t180\t4.0\t30.0",
    ]
    return "\n".join(lines) + "\n"


def _build_minimal_active_file_text():
    """Build a minimal V5.03 active TRP export with one data sample."""
    lines = [f"Header line {i}" for i in range(55)]
    lines[0] = "Howland Chamber Export V5.03"
    lines[5] = "Total Radiated Power Test"
    lines[13] = "Test Frequency: 2450 MHz"
    lines[15] = "Test Type: Discrete Test"
    lines[31] = "Start Phi: 0 Deg"
    lines[32] = "Stop Phi: 360 Deg"
    lines[33] = "Inc Phi: 180 Deg"
    lines[38] = "Start Theta: 0 Deg"
    lines[39] = "Stop Theta: 180 Deg"
    lines[40] = "Inc Theta: 90 Deg"
    lines[46] = "H Cal Factor = 1.0 dB"
    lines[47] = "V Cal Factor = 2.0 dB"
    lines[49] = "Calculated TRP = -3.0 dBm"
    lines[54] = "0 0 -10 -20"
    return "\n".join(lines) + "\n"


class TestImports:
    """Test that all file_utils functions can be imported"""

    def test_imports_successful(self):
        """Verify all file parsing functions import without errors"""
        assert callable(read_passive_file)
        assert callable(read_active_file)
        assert callable(parse_2port_data)


class TestErrorHandling:
    """Test error handling for invalid inputs"""

    def test_nonexistent_passive_file(self):
        """Test that nonexistent passive file raises appropriate error"""
        with pytest.raises((FileNotFoundError, IOError, ValueError)):
            read_passive_file("nonexistent_file_12345.txt")

    def test_nonexistent_active_file(self):
        """Test that nonexistent active file raises appropriate error"""
        with pytest.raises((FileNotFoundError, IOError, ValueError)):
            read_active_file("nonexistent_file_12345.txt")

    def test_nonexistent_vna_file(self):
        """Test that nonexistent VNA file raises appropriate error"""
        with pytest.raises((FileNotFoundError, IOError, ValueError)):
            parse_2port_data("nonexistent_file_12345.csv")


class TestPassiveFileParsing:
    """Validate passive file parsing with a minimal fixture."""

    def test_read_passive_file_minimal_fixture(self, tmp_path):
        file_path = tmp_path / "sample_AP_HPol.txt"
        file_path.write_text(_build_minimal_passive_file_text(), encoding="utf-8")

        all_data, start_phi, stop_phi, inc_phi, start_theta, stop_theta, inc_theta = (
            read_passive_file(str(file_path))
        )

        assert start_phi == 0.0
        assert stop_phi == 180.0
        assert inc_phi == 180.0
        assert start_theta == 0.0
        assert stop_theta == 90.0
        assert inc_theta == 90.0
        assert len(all_data) == 1

        freq_data = all_data[0]
        assert freq_data["frequency"] == 2450.0
        assert freq_data["cal_factor"] == 2.5
        assert freq_data["theta"] == [0.0, 0.0, 90.0, 90.0]
        assert freq_data["phi"] == [0.0, 180.0, 0.0, 180.0]
        assert freq_data["mag"] == [1.0, 2.0, 3.0, 4.0]
        assert freq_data["phase"] == [0.0, 10.0, 20.0, 30.0]


class TestActiveFileParsing:
    """Validate active file parsing with a minimal fixture."""

    def test_read_active_file_minimal_fixture(self, tmp_path):
        file_path = tmp_path / "sample_TRP.txt"
        file_path.write_text(_build_minimal_active_file_text(), encoding="utf-8")

        data = read_active_file(str(file_path))

        assert data["Frequency"] == 2450.0
        assert data["Start Phi"] == 0.0
        assert data["Stop Phi"] == 360.0
        assert data["Inc Phi"] == 180.0
        assert data["Start Theta"] == 0.0
        assert data["Stop Theta"] == 180.0
        assert data["Inc Theta"] == 90.0
        assert data["Calculated TRP(dBm)"] == -3.0
        np.testing.assert_allclose(data["Theta_Angles_Deg"], np.array([0.0]))
        np.testing.assert_allclose(data["Phi_Angles_Deg"], np.array([0.0]))
        # Parsed values include calibration factors from the header lines.
        np.testing.assert_allclose(data["H_Power_dBm"], np.array([-9.0]))
        np.testing.assert_allclose(data["V_Power_dBm"], np.array([-18.0]))


class TestVNAFileParsing:
    """Validate CSV VNA parsing with a minimal 2-port fixture."""

    def test_parse_2port_data_minimal_fixture(self, tmp_path):
        file_path = tmp_path / "sample_vna.csv"
        file_path.write_text(
            "\n".join(
                [
                    "Metadata row 1",
                    "Metadata row 2",
                    "! Stimulus(Hz),S11(SWR),S22(SWR),S11(dB),S22(dB),S21(dB),S21(s)",
                    "1000000000,1.5,1.4,-10.0,-11.0,-3.0,1.2e-9",
                    "2000000000,1.7,1.6,-9.5,-10.2,-2.7,1.1e-9",
                ]
            )
            + "\n",
            encoding="utf-8",
        )

        parsed = parse_2port_data(str(file_path))

        assert list(parsed.columns) == [
            "! Stimulus(Hz)",
            "S11(SWR)",
            "S22(SWR)",
            "S11(dB)",
            "S22(dB)",
            "S21(dB)",
            "S21(s)",
        ]
        assert parsed.shape == (2, 7)
        np.testing.assert_allclose(parsed["S21(dB)"].to_numpy(), np.array([-3.0, -2.7]))


class TestAllImports:
    """Test that all file_utils public functions can be imported"""

    def test_check_matching_files_importable(self):
        from plot_antenna.file_utils import check_matching_files

        assert callable(check_matching_files)

    def test_batch_process_passive_importable(self):
        from plot_antenna.file_utils import batch_process_passive_scans

        assert callable(batch_process_passive_scans)

    def test_batch_process_active_importable(self):
        from plot_antenna.file_utils import batch_process_active_scans

        assert callable(batch_process_active_scans)

    def test_convert_hpolvpol_importable(self):
        from plot_antenna.file_utils import convert_HpolVpol_files

        assert callable(convert_HpolVpol_files)

    def test_validate_hpol_vpol_importable(self):
        from plot_antenna.file_utils import validate_hpol_vpol_files

        assert callable(validate_hpol_vpol_files)
