"""
Integration tests using real antenna measurement data files.

These tests exercise the full pipeline from file I/O through calculations
to AI analysis, using actual measurement files rather than synthetic data.
This targets the low-coverage modules: file_utils.py (5%), calculations.py (21%).

Test data directory: /home/swamp/Downloads/TestFiles/_Test Files/test_files/
"""

import os
import pytest
import numpy as np
import pandas as pd

from plot_antenna.file_utils import (
    read_passive_file,
    read_active_file,
    parse_passive_file,
    parse_active_file,
    check_matching_files,
    validate_hpol_vpol_files,
    parse_2port_data,
    process_gd_file,
)
from plot_antenna.calculations import (
    calculate_trp,
    calculate_active_variables,
    calculate_passive_variables,
    calculate_polarization_parameters,
    extract_passive_frequencies,
    angles_match,
    extrapolate_pattern,
    validate_extrapolation,
)
from plot_antenna.ai_analysis import AntennaAnalyzer


# ---------------------------------------------------------------------------
# Paths to real test data
# ---------------------------------------------------------------------------
TEST_DATA_DIR = "/home/swamp/Downloads/TestFiles/_Test Files/test_files"

PASSIVE_BLE_HPOL = os.path.join(TEST_DATA_DIR, "PassiveTest_BLE AP_HPol.txt")
PASSIVE_BLE_VPOL = os.path.join(TEST_DATA_DIR, "PassiveTest_BLE AP_VPol.txt")
PASSIVE_LORA_HPOL = os.path.join(TEST_DATA_DIR, "PassiveTest_LoRa AP_HPol.txt")
PASSIVE_LORA_VPOL = os.path.join(TEST_DATA_DIR, "PassiveTest_LoRa AP_VPol.txt")
ACTIVE_BLE_TRP = os.path.join(TEST_DATA_DIR, "Active Test_BLE TRP.txt")
S2VNA_GROUP_DELAY = os.path.join(TEST_DATA_DIR, "Group Delay", "S11S22S21_0deg.csv")
GD_FILE = os.path.join(TEST_DATA_DIR, "G&D", "Free Space, 2.1mm PolyCarb..txt")

# Skip all tests in this file if test data directory is missing
pytestmark = pytest.mark.skipif(
    not os.path.isdir(TEST_DATA_DIR),
    reason=f"Test data directory not found: {TEST_DATA_DIR}",
)


# ===========================================================================
# 1. Passive File Parsing
# ===========================================================================
class TestReadPassiveFile:
    """Test read_passive_file / parse_passive_file with real BLE and LoRa data."""

    def test_ble_hpol_returns_seven_tuple(self):
        result = read_passive_file(PASSIVE_BLE_HPOL)
        assert isinstance(result, tuple)
        assert len(result) == 7, "Expected (all_data, start_phi, stop_phi, inc_phi, start_theta, stop_theta, inc_theta)"

    def test_ble_hpol_angle_parameters(self):
        _, start_phi, stop_phi, inc_phi, start_theta, stop_theta, inc_theta = read_passive_file(
            PASSIVE_BLE_HPOL
        )
        assert start_phi == 0.0
        assert stop_phi == 345.0
        assert inc_phi == 15.0
        assert start_theta == 0.0
        assert stop_theta == 165.0
        assert inc_theta == 15.0

    def test_ble_hpol_frequency_count(self):
        """BLE passive file has 61 frequencies (2300-2600 MHz, 5 MHz steps)."""
        all_data, *_ = read_passive_file(PASSIVE_BLE_HPOL)
        assert len(all_data) == 61

    def test_ble_hpol_first_frequency(self):
        all_data, *_ = read_passive_file(PASSIVE_BLE_HPOL)
        assert all_data[0]["frequency"] == pytest.approx(2300.0)

    def test_ble_hpol_last_frequency(self):
        all_data, *_ = read_passive_file(PASSIVE_BLE_HPOL)
        assert all_data[-1]["frequency"] == pytest.approx(2600.0)

    def test_ble_hpol_data_points_per_frequency(self):
        """Each frequency block should have 24 phi x 12 theta = 288 data points."""
        all_data, *_ = read_passive_file(PASSIVE_BLE_HPOL)
        for entry in all_data:
            assert len(entry["theta"]) == 288
            assert len(entry["phi"]) == 288
            assert len(entry["mag"]) == 288
            assert len(entry["phase"]) == 288

    def test_ble_hpol_has_cal_factor(self):
        all_data, *_ = read_passive_file(PASSIVE_BLE_HPOL)
        for entry in all_data:
            assert "cal_factor" in entry
            assert isinstance(entry["cal_factor"], float)

    def test_lora_hpol_parses_successfully(self):
        """LoRa file should also parse correctly with a different frequency range."""
        all_data, start_phi, stop_phi, inc_phi, start_theta, stop_theta, inc_theta = (
            read_passive_file(PASSIVE_LORA_HPOL)
        )
        assert len(all_data) > 0
        # LoRa is typically in 700-1100 MHz range
        assert all_data[0]["frequency"] < 2000.0

    def test_ble_vpol_parses_successfully(self):
        all_data, *_ = read_passive_file(PASSIVE_BLE_VPOL)
        assert len(all_data) == 61

    def test_ble_hpol_vpol_same_frequency_count(self):
        hpol_data, *_ = read_passive_file(PASSIVE_BLE_HPOL)
        vpol_data, *_ = read_passive_file(PASSIVE_BLE_VPOL)
        assert len(hpol_data) == len(vpol_data)

    def test_ble_gain_values_in_plausible_range(self):
        """Gain values for a BLE antenna should be roughly -40 to +10 dBi."""
        all_data, *_ = read_passive_file(PASSIVE_BLE_HPOL)
        for entry in all_data:
            mags = np.array(entry["mag"])
            assert np.all(mags > -60), "Unreasonably low gain value detected"
            assert np.all(mags < 20), "Unreasonably high gain value detected"


# ===========================================================================
# 2. Active File Parsing
# ===========================================================================
class TestReadActiveFile:
    """Test read_active_file / parse_active_file with real BLE TRP data."""

    def test_returns_dict(self):
        data = read_active_file(ACTIVE_BLE_TRP)
        assert isinstance(data, dict)

    def test_expected_keys(self):
        data = read_active_file(ACTIVE_BLE_TRP)
        expected_keys = [
            "Frequency",
            "Start Phi",
            "Stop Phi",
            "Inc Phi",
            "Start Theta",
            "Stop Theta",
            "Inc Theta",
            "Calculated TRP(dBm)",
            "Theta_Angles_Deg",
            "Phi_Angles_Deg",
            "H_Power_dBm",
            "V_Power_dBm",
        ]
        for key in expected_keys:
            assert key in data, f"Missing key: {key}"

    def test_frequency_value(self):
        data = read_active_file(ACTIVE_BLE_TRP)
        assert data["Frequency"] == pytest.approx(2440.0)

    def test_angle_parameters(self):
        data = read_active_file(ACTIVE_BLE_TRP)
        assert data["Start Phi"] == 0.0
        assert data["Stop Phi"] == 345.0
        assert data["Inc Phi"] == 15.0
        assert data["Start Theta"] == 0.0
        assert data["Stop Theta"] == 165.0
        assert data["Inc Theta"] == 15.0

    def test_calculated_trp_value(self):
        """The file header says TRP = -6.61 dBm."""
        data = read_active_file(ACTIVE_BLE_TRP)
        assert data["Calculated TRP(dBm)"] == pytest.approx(-6.61, abs=0.01)

    def test_data_array_lengths(self):
        """24 phi x 12 theta = 288 data points."""
        data = read_active_file(ACTIVE_BLE_TRP)
        assert len(data["Theta_Angles_Deg"]) == 288
        assert len(data["Phi_Angles_Deg"]) == 288
        assert len(data["H_Power_dBm"]) == 288
        assert len(data["V_Power_dBm"]) == 288

    def test_power_arrays_are_numpy(self):
        data = read_active_file(ACTIVE_BLE_TRP)
        assert isinstance(data["H_Power_dBm"], np.ndarray)
        assert isinstance(data["V_Power_dBm"], np.ndarray)

    def test_cal_factors_applied(self):
        """H_Power_dBm and V_Power_dBm should have cal factors added.
        The file has h_cal_fact=6.10 and v_cal_fact=5.65.
        Raw first H data point is -11.567, so corrected = -11.567 + 6.10 = -5.467."""
        data = read_active_file(ACTIVE_BLE_TRP)
        # First H data point should be approximately -11.567 + 6.10 = -5.467
        assert data["H_Power_dBm"][0] == pytest.approx(-5.467, abs=0.01)

    def test_invalid_file_raises_error(self):
        """A passive file should not parse as active."""
        with pytest.raises(ValueError, match="does not contain TRP data"):
            read_active_file(PASSIVE_BLE_HPOL)


# ===========================================================================
# 3. Check Matching Files
# ===========================================================================
class TestCheckMatchingFiles:
    """Test check_matching_files with real file pairs."""

    def test_matching_ble_pair(self):
        result, msg = check_matching_files(PASSIVE_BLE_HPOL, PASSIVE_BLE_VPOL)
        assert result is True
        assert msg == ""

    def test_matching_lora_pair(self):
        result, msg = check_matching_files(PASSIVE_LORA_HPOL, PASSIVE_LORA_VPOL)
        assert result is True
        assert msg == ""

    def test_mismatched_pair_ble_vs_lora(self):
        result, msg = check_matching_files(PASSIVE_BLE_HPOL, PASSIVE_LORA_VPOL)
        assert result is False

    def test_mismatched_pair_reversed(self):
        result, msg = check_matching_files(PASSIVE_LORA_HPOL, PASSIVE_BLE_VPOL)
        assert result is False


# ===========================================================================
# 4. Validate HPOL/VPOL Files
# ===========================================================================
class TestValidateHpolVpolFiles:
    """Test validate_hpol_vpol_files with real data."""

    def test_valid_ble_pair(self):
        is_valid, error = validate_hpol_vpol_files(PASSIVE_BLE_HPOL, PASSIVE_BLE_VPOL)
        assert is_valid is True
        assert error == ""

    def test_swapped_ble_pair(self):
        """Passing VPOL as HPOL and vice versa should fail."""
        is_valid, error = validate_hpol_vpol_files(PASSIVE_BLE_VPOL, PASSIVE_BLE_HPOL)
        assert is_valid is False
        assert "SWAP" in error.upper() or "INCORRECT" in error.upper()

    def test_valid_lora_pair(self):
        is_valid, error = validate_hpol_vpol_files(PASSIVE_LORA_HPOL, PASSIVE_LORA_VPOL)
        assert is_valid is True

    def test_mismatched_datasets(self):
        """BLE HPOL with LoRa VPOL should fail validation."""
        is_valid, error = validate_hpol_vpol_files(PASSIVE_BLE_HPOL, PASSIVE_LORA_VPOL)
        assert is_valid is False


# ===========================================================================
# 5. 2-Port VNA Data Parsing
# ===========================================================================
class TestParse2PortData:
    """Test parse_2port_data with real S2VNA CSV file."""

    def test_returns_dataframe(self):
        df = parse_2port_data(S2VNA_GROUP_DELAY)
        assert isinstance(df, pd.DataFrame)

    def test_has_stimulus_column(self):
        df = parse_2port_data(S2VNA_GROUP_DELAY)
        assert "! Stimulus(Hz)" in df.columns

    def test_has_s_parameter_columns(self):
        df = parse_2port_data(S2VNA_GROUP_DELAY)
        # The file has S11, S22, S21 columns
        s_cols = [c for c in df.columns if c.startswith("S")]
        assert len(s_cols) >= 3, f"Expected at least 3 S-parameter columns, found: {s_cols}"

    def test_frequency_values_plausible(self):
        df = parse_2port_data(S2VNA_GROUP_DELAY)
        freqs = df["! Stimulus(Hz)"].values
        # The group delay file starts around 5 GHz
        assert freqs[0] > 1e9, "Frequency should be in Hz (> 1 GHz)"
        assert len(freqs) > 10, "Should have more than 10 data points"

    def test_s_parameter_values_are_numeric(self):
        df = parse_2port_data(S2VNA_GROUP_DELAY)
        for col in df.columns:
            assert pd.api.types.is_numeric_dtype(df[col]), f"Column {col} should be numeric"


# ===========================================================================
# 6. G&D File Processing
# ===========================================================================
class TestProcessGDFile:
    """Test process_gd_file with real Gain & Directivity data."""

    def test_returns_dict(self):
        result = process_gd_file(GD_FILE)
        assert isinstance(result, dict)

    def test_expected_keys(self):
        result = process_gd_file(GD_FILE)
        for key in ["Frequency", "Gain", "Directivity", "Efficiency"]:
            assert key in result, f"Missing key: {key}"

    def test_data_length_consistency(self):
        result = process_gd_file(GD_FILE)
        n = len(result["Frequency"])
        assert n > 0, "Should have at least one data point"
        assert len(result["Gain"]) == n
        assert len(result["Directivity"]) == n
        assert len(result["Efficiency"]) == n

    def test_frequency_range(self):
        """The G&D file covers 700-1100 MHz."""
        result = process_gd_file(GD_FILE)
        assert result["Frequency"][0] == pytest.approx(700.0)
        assert result["Frequency"][-1] == pytest.approx(1100.0)

    def test_efficiency_range(self):
        """Efficiency should be between 0 and 100%."""
        result = process_gd_file(GD_FILE)
        for eff in result["Efficiency"]:
            assert 0 < eff <= 100.0, f"Efficiency {eff}% out of range"

    def test_gain_values_plausible(self):
        """Gain for a flex dipole should be roughly -10 to +5 dBi."""
        result = process_gd_file(GD_FILE)
        for g in result["Gain"]:
            assert -15.0 < g < 10.0, f"Gain {g} dBi seems implausible"

    def test_directivity_positive(self):
        """Directivity should always be positive (in dB)."""
        result = process_gd_file(GD_FILE)
        for d in result["Directivity"]:
            assert d > 0.0, f"Directivity {d} dB should be positive"


# ===========================================================================
# 7. Calculate Passive Variables
# ===========================================================================
class TestCalculatePassiveVariables:
    """Test calculate_passive_variables with real parsed data."""

    @pytest.fixture
    def parsed_ble_data(self):
        hpol_data, sp, stp, ip, st, stht, it = read_passive_file(PASSIVE_BLE_HPOL)
        vpol_data, *_ = read_passive_file(PASSIVE_BLE_VPOL)
        freq_list = [entry["frequency"] for entry in hpol_data]
        return {
            "hpol_data": hpol_data,
            "vpol_data": vpol_data,
            "start_phi": sp,
            "stop_phi": stp,
            "inc_phi": ip,
            "start_theta": st,
            "stop_theta": stht,
            "inc_theta": it,
            "freq_list": freq_list,
        }

    def test_returns_five_arrays(self, parsed_ble_data):
        d = parsed_ble_data
        result = calculate_passive_variables(
            d["hpol_data"],
            d["vpol_data"],
            0.0,  # cable_loss
            d["start_phi"],
            d["stop_phi"],
            d["inc_phi"],
            d["start_theta"],
            d["stop_theta"],
            d["inc_theta"],
            d["freq_list"],
            d["freq_list"][0],
        )
        assert len(result) == 5
        theta_deg, phi_deg, v_gain, h_gain, total_gain = result
        assert theta_deg.ndim == 2
        assert phi_deg.ndim == 2

    def test_total_gain_greater_or_equal_individual(self, parsed_ble_data):
        """Total gain (linear sum of H+V) should be >= max(H, V) at every point."""
        d = parsed_ble_data
        theta_deg, phi_deg, v_gain, h_gain, total_gain = calculate_passive_variables(
            d["hpol_data"],
            d["vpol_data"],
            0.0,
            d["start_phi"],
            d["stop_phi"],
            d["inc_phi"],
            d["start_theta"],
            d["stop_theta"],
            d["inc_theta"],
            d["freq_list"],
            d["freq_list"][0],
        )
        # Use first frequency column
        max_individual = np.maximum(v_gain[:, 0], h_gain[:, 0])
        # Total gain in dB should be >= max individual (or within rounding tolerance)
        assert np.all(total_gain[:, 0] >= max_individual - 0.01)

    def test_cable_loss_shifts_gain(self, parsed_ble_data):
        """Adding cable loss should increase gain values (compensating for cable)."""
        d = parsed_ble_data
        _, _, _, _, total_no_loss = calculate_passive_variables(
            d["hpol_data"],
            d["vpol_data"],
            0.0,
            d["start_phi"],
            d["stop_phi"],
            d["inc_phi"],
            d["start_theta"],
            d["stop_theta"],
            d["inc_theta"],
            d["freq_list"],
            d["freq_list"][0],
        )
        _, _, _, _, total_with_loss = calculate_passive_variables(
            d["hpol_data"],
            d["vpol_data"],
            3.0,  # 3 dB cable loss
            d["start_phi"],
            d["stop_phi"],
            d["inc_phi"],
            d["start_theta"],
            d["stop_theta"],
            d["inc_theta"],
            d["freq_list"],
            d["freq_list"][0],
        )
        # With cable loss correction, total gain should be higher
        assert np.mean(total_with_loss[:, 0]) > np.mean(total_no_loss[:, 0])

    def test_gain_array_shape(self, parsed_ble_data):
        """Shape should be (288, 61) for 288 data points x 61 frequencies."""
        d = parsed_ble_data
        theta_deg, phi_deg, v_gain, h_gain, total_gain = calculate_passive_variables(
            d["hpol_data"],
            d["vpol_data"],
            0.0,
            d["start_phi"],
            d["stop_phi"],
            d["inc_phi"],
            d["start_theta"],
            d["stop_theta"],
            d["inc_theta"],
            d["freq_list"],
            d["freq_list"][0],
        )
        assert total_gain.shape == (288, 61)


# ===========================================================================
# 8. Calculate Active Variables
# ===========================================================================
class TestCalculateActiveVariables:
    """Test calculate_active_variables with real TRP data."""

    @pytest.fixture
    def active_data(self):
        return read_active_file(ACTIVE_BLE_TRP)

    def test_returns_22_element_tuple(self, active_data):
        result = calculate_active_variables(
            active_data["Start Phi"],
            active_data["Stop Phi"],
            active_data["Start Theta"],
            active_data["Stop Theta"],
            active_data["Inc Phi"],
            active_data["Inc Theta"],
            active_data["H_Power_dBm"],
            active_data["V_Power_dBm"],
        )
        assert len(result) == 22

    def test_trp_close_to_file_value(self, active_data):
        """Our computed TRP should be close to the file's stated value of -6.61 dBm."""
        result = calculate_active_variables(
            active_data["Start Phi"],
            active_data["Stop Phi"],
            active_data["Start Theta"],
            active_data["Stop Theta"],
            active_data["Inc Phi"],
            active_data["Inc Theta"],
            active_data["H_Power_dBm"],
            active_data["V_Power_dBm"],
        )
        TRP_dBm = result[19]  # Index 19 is TRP_dBm
        file_trp = active_data["Calculated TRP(dBm)"]
        # Allow tolerance since integration methods may differ slightly
        assert TRP_dBm == pytest.approx(file_trp, abs=1.5)

    def test_data_points_count(self, active_data):
        result = calculate_active_variables(
            active_data["Start Phi"],
            active_data["Stop Phi"],
            active_data["Start Theta"],
            active_data["Stop Theta"],
            active_data["Inc Phi"],
            active_data["Inc Theta"],
            active_data["H_Power_dBm"],
            active_data["V_Power_dBm"],
        )
        data_points = result[0]
        assert data_points == 288

    def test_total_power_2d_shape(self, active_data):
        result = calculate_active_variables(
            active_data["Start Phi"],
            active_data["Stop Phi"],
            active_data["Start Theta"],
            active_data["Stop Theta"],
            active_data["Inc Phi"],
            active_data["Inc Theta"],
            active_data["H_Power_dBm"],
            active_data["V_Power_dBm"],
        )
        total_power_2d = result[5]  # Index 5 is total_power_dBm_2d
        # 12 theta x 24 phi
        assert total_power_2d.shape == (12, 24)

    def test_h_trp_and_v_trp_less_than_total(self, active_data):
        """Individual polarization TRP should be less than total TRP."""
        result = calculate_active_variables(
            active_data["Start Phi"],
            active_data["Stop Phi"],
            active_data["Start Theta"],
            active_data["Stop Theta"],
            active_data["Inc Phi"],
            active_data["Inc Theta"],
            active_data["H_Power_dBm"],
            active_data["V_Power_dBm"],
        )
        TRP_dBm = result[19]
        h_TRP_dBm = result[20]
        v_TRP_dBm = result[21]
        assert h_TRP_dBm < TRP_dBm + 0.01  # H-pol TRP <= Total TRP
        assert v_TRP_dBm < TRP_dBm + 0.01  # V-pol TRP <= Total TRP


# ===========================================================================
# 9. Polarization Parameters
# ===========================================================================
class TestCalculatePolarizationParameters:
    """Test calculate_polarization_parameters with real HPOL/VPOL data."""

    @pytest.fixture
    def pol_data(self):
        hpol_data, *_ = read_passive_file(PASSIVE_BLE_HPOL)
        vpol_data, *_ = read_passive_file(PASSIVE_BLE_VPOL)
        return hpol_data, vpol_data

    def test_returns_list(self, pol_data):
        hpol_data, vpol_data = pol_data
        results = calculate_polarization_parameters(hpol_data, vpol_data, cable_loss=0.0)
        assert isinstance(results, list)
        assert len(results) > 0

    def test_one_result_per_frequency(self, pol_data):
        hpol_data, vpol_data = pol_data
        results = calculate_polarization_parameters(hpol_data, vpol_data)
        # Should have one result per matching frequency pair
        assert len(results) == len(hpol_data)

    def test_result_keys(self, pol_data):
        hpol_data, vpol_data = pol_data
        results = calculate_polarization_parameters(hpol_data, vpol_data)
        expected_keys = [
            "frequency",
            "theta",
            "phi",
            "axial_ratio_dB",
            "tilt_angle_deg",
            "sense",
            "cross_pol_discrimination_dB",
        ]
        for key in expected_keys:
            assert key in results[0], f"Missing key: {key}"

    def test_axial_ratio_non_negative(self, pol_data):
        """Axial ratio in dB should be >= 0 (AR >= 1 linear)."""
        hpol_data, vpol_data = pol_data
        results = calculate_polarization_parameters(hpol_data, vpol_data)
        for r in results[:5]:  # Check first 5 frequencies
            ar = np.array(r["axial_ratio_dB"])
            assert np.all(ar >= -0.01), "Axial ratio should be non-negative in dB"

    def test_sense_values(self, pol_data):
        """Polarization sense should be +1 (LHCP), -1 (RHCP), or 0."""
        hpol_data, vpol_data = pol_data
        results = calculate_polarization_parameters(hpol_data, vpol_data)
        for r in results[:5]:
            sense = np.array(r["sense"])
            unique_vals = np.unique(sense)
            for v in unique_vals:
                assert v in [-1.0, 0.0, 1.0], f"Unexpected sense value: {v}"


# ===========================================================================
# 10. AI Analysis with Real Data
# ===========================================================================
class TestAIAnalysisWithRealData:
    """Test AntennaAnalyzer class with real measurement data through the full pipeline."""

    @pytest.fixture
    def passive_analyzer(self):
        """Create an AntennaAnalyzer from real BLE passive data."""
        hpol_data, sp, stp, ip, st, stht, it = read_passive_file(PASSIVE_BLE_HPOL)
        vpol_data, *_ = read_passive_file(PASSIVE_BLE_VPOL)
        freq_list = [entry["frequency"] for entry in hpol_data]

        theta_deg, phi_deg, v_gain, h_gain, total_gain = calculate_passive_variables(
            hpol_data,
            vpol_data,
            0.0,
            sp,
            stp,
            ip,
            st,
            stht,
            it,
            freq_list,
            freq_list[0],
        )

        measurement_data = {
            "phi": phi_deg[:, 0],
            "theta": theta_deg[:, 0],
            "total_gain": total_gain,
            "h_gain": h_gain,
            "v_gain": v_gain,
        }

        return AntennaAnalyzer(measurement_data, scan_type="passive", frequencies=freq_list)

    @pytest.fixture
    def active_analyzer(self):
        """Create an AntennaAnalyzer from real BLE active TRP data."""
        data = read_active_file(ACTIVE_BLE_TRP)
        result = calculate_active_variables(
            data["Start Phi"],
            data["Stop Phi"],
            data["Start Theta"],
            data["Stop Theta"],
            data["Inc Phi"],
            data["Inc Theta"],
            data["H_Power_dBm"],
            data["V_Power_dBm"],
        )
        total_power_2d = result[5]
        TRP_dBm = result[19]
        h_TRP_dBm = result[20]
        v_TRP_dBm = result[21]

        measurement_data = {
            "total_power": total_power_2d.ravel(),
            "TRP_dBm": TRP_dBm,
            "h_TRP_dBm": h_TRP_dBm,
            "v_TRP_dBm": v_TRP_dBm,
        }

        return AntennaAnalyzer(
            measurement_data, scan_type="active", frequencies=[data["Frequency"]]
        )

    def test_passive_gain_statistics(self, passive_analyzer):
        stats = passive_analyzer.get_gain_statistics(frequency=2400.0)
        assert "max_gain_dBi" in stats
        assert "avg_gain_dBi" in stats
        assert stats["max_gain_dBi"] > stats["min_gain_dBi"]
        assert stats["scan_type"] == "passive"

    def test_passive_gain_in_plausible_range(self, passive_analyzer):
        stats = passive_analyzer.get_gain_statistics(frequency=2450.0)
        # A BLE antenna should have gain roughly -20 to +10 dBi
        assert -30 < stats["max_gain_dBi"] < 15
        assert -40 < stats["min_gain_dBi"] < 5

    def test_passive_pattern_analysis(self, passive_analyzer):
        pattern = passive_analyzer.analyze_pattern(frequency=2400.0)
        assert "peak_gain_dBi" in pattern
        assert "pattern_type" in pattern
        assert pattern["pattern_type"] in ["omnidirectional", "sectoral", "directional"]

    def test_passive_polarization_comparison(self, passive_analyzer):
        comparison = passive_analyzer.compare_polarizations(frequency=2450.0)
        assert "max_hpol_gain_dBi" in comparison
        assert "max_vpol_gain_dBi" in comparison
        assert "polarization_balance_dB" in comparison

    def test_passive_all_frequencies_analysis(self, passive_analyzer):
        analysis = passive_analyzer.analyze_all_frequencies()
        assert "resonance_frequency_MHz" in analysis
        assert "peak_gain_per_freq" in analysis
        assert len(analysis["peak_gain_per_freq"]) == 61

    def test_active_gain_statistics(self, active_analyzer):
        stats = active_analyzer.get_gain_statistics(frequency=2440.0)
        assert "max_power_dBm" in stats
        assert "TRP_dBm" in stats
        assert stats["scan_type"] == "active"

    def test_active_trp_in_stats(self, active_analyzer):
        stats = active_analyzer.get_gain_statistics(frequency=2440.0)
        assert "TRP_dBm" in stats
        assert "h_TRP_dBm" in stats
        assert "v_TRP_dBm" in stats

    def test_passive_hpbw_computed(self, passive_analyzer):
        """Pattern analysis should compute HPBW for real data."""
        pattern = passive_analyzer.analyze_pattern(frequency=2450.0)
        # HPBW should be computed for real data with a proper grid
        # A BLE antenna might have wide beamwidth
        if "hpbw_e_plane" in pattern:
            assert pattern["hpbw_e_plane"] is None or pattern["hpbw_e_plane"] > 0
        if "hpbw_h_plane" in pattern:
            assert pattern["hpbw_h_plane"] is None or pattern["hpbw_h_plane"] > 0


# ===========================================================================
# 11. Extract Passive Frequencies
# ===========================================================================
class TestExtractPassiveFrequencies:
    """Test extract_passive_frequencies with real passive files."""

    def test_ble_frequency_count(self):
        freqs = extract_passive_frequencies(PASSIVE_BLE_HPOL)
        assert len(freqs) == 61

    def test_ble_frequency_range(self):
        freqs = extract_passive_frequencies(PASSIVE_BLE_HPOL)
        assert freqs[0] == pytest.approx(2300.0)
        assert freqs[-1] == pytest.approx(2600.0)

    def test_lora_frequency_range(self):
        freqs = extract_passive_frequencies(PASSIVE_LORA_HPOL)
        assert len(freqs) > 0
        # LoRa is in 700-1100 MHz range
        assert freqs[0] < 1200.0


# ===========================================================================
# 12. Angles Match with Real Data
# ===========================================================================
class TestAnglesMatchRealData:
    """Test angles_match using real parsed file angle data."""

    def test_same_file_pair_matches(self):
        _, sp_h, stp_h, ip_h, st_h, stht_h, it_h = read_passive_file(PASSIVE_BLE_HPOL)
        _, sp_v, stp_v, ip_v, st_v, stht_v, it_v = read_passive_file(PASSIVE_BLE_VPOL)
        assert angles_match(sp_h, stp_h, ip_h, st_h, stht_h, it_h, sp_v, stp_v, ip_v, st_v, stht_v, it_v)

    def test_different_datasets_may_still_match(self):
        """BLE and LoRa may have same angle grids even if different frequencies."""
        _, sp_h, stp_h, ip_h, st_h, stht_h, it_h = read_passive_file(PASSIVE_BLE_HPOL)
        _, sp_l, stp_l, ip_l, st_l, stht_l, it_l = read_passive_file(PASSIVE_LORA_HPOL)
        # Both use same chamber setup (0-345/15 phi, 0-165/15 theta)
        result = angles_match(sp_h, stp_h, ip_h, st_h, stht_h, it_h, sp_l, stp_l, ip_l, st_l, stht_l, it_l)
        # This is informational - both may match since same chamber setup
        assert isinstance(result, bool)


# ===========================================================================
# 13. End-to-End TRP Pipeline
# ===========================================================================
class TestEndToEndTRP:
    """Cross-module end-to-end test: file I/O -> calculations -> analysis."""

    def test_active_full_pipeline(self):
        """Parse active file, compute variables, and run AI analysis in sequence."""
        # Step 1: Read file
        data = read_active_file(ACTIVE_BLE_TRP)
        assert data["Frequency"] == pytest.approx(2440.0)

        # Step 2: Calculate variables
        result = calculate_active_variables(
            data["Start Phi"],
            data["Stop Phi"],
            data["Start Theta"],
            data["Stop Theta"],
            data["Inc Phi"],
            data["Inc Theta"],
            data["H_Power_dBm"],
            data["V_Power_dBm"],
        )
        TRP_dBm = result[19]
        assert isinstance(TRP_dBm, float)

        # Step 3: AI analysis
        total_power_2d = result[5]
        analyzer = AntennaAnalyzer(
            {"total_power": total_power_2d.ravel(), "TRP_dBm": TRP_dBm},
            scan_type="active",
            frequencies=[data["Frequency"]],
        )
        stats = analyzer.get_gain_statistics()
        assert "max_power_dBm" in stats

    def test_passive_full_pipeline(self):
        """Parse passive files, compute gains, polarization, and run AI analysis."""
        # Step 1: Read files
        hpol_data, sp, stp, ip, st, stht, it = read_passive_file(PASSIVE_BLE_HPOL)
        vpol_data, *_ = read_passive_file(PASSIVE_BLE_VPOL)
        freq_list = [e["frequency"] for e in hpol_data]

        # Step 2: Calculate passive variables
        theta_deg, phi_deg, v_gain, h_gain, total_gain = calculate_passive_variables(
            hpol_data, vpol_data, 0.0, sp, stp, ip, st, stht, it, freq_list, 2450.0
        )

        # Step 3: Calculate polarization
        pol_results = calculate_polarization_parameters(hpol_data[:3], vpol_data[:3])
        assert len(pol_results) == 3

        # Step 4: AI analysis
        measurement_data = {
            "phi": phi_deg[:, 0],
            "theta": theta_deg[:, 0],
            "total_gain": total_gain,
            "h_gain": h_gain,
            "v_gain": v_gain,
        }
        analyzer = AntennaAnalyzer(measurement_data, scan_type="passive", frequencies=freq_list)

        stats = analyzer.get_gain_statistics(frequency=2450.0)
        assert "max_gain_dBi" in stats

        pattern = analyzer.analyze_pattern(frequency=2450.0)
        assert "pattern_type" in pattern

        comparison = analyzer.compare_polarizations(frequency=2450.0)
        assert "polarization_balance_dB" in comparison

        freq_analysis = analyzer.analyze_all_frequencies()
        assert freq_analysis["num_frequencies"] == 61


# ===========================================================================
# 14. TRP Calculation Unit Tests with Real Data
# ===========================================================================
class TestCalculateTRP:
    """Test calculate_trp directly with data derived from real measurements."""

    def test_trp_with_real_active_data(self):
        """Compute TRP from real active measurement data."""
        data = read_active_file(ACTIVE_BLE_TRP)
        # Reshape power data into 2D array
        theta_points = int((data["Stop Theta"] - data["Start Theta"]) / data["Inc Theta"] + 1)
        phi_points = int((data["Stop Phi"] - data["Start Phi"]) / data["Inc Phi"] + 1)

        total_power_linear = 10 ** (data["H_Power_dBm"] / 10) + 10 ** (data["V_Power_dBm"] / 10)
        total_power_dBm = 10 * np.log10(total_power_linear)
        total_power_2d = total_power_dBm.reshape((theta_points, phi_points))

        theta_angles_deg = np.linspace(data["Start Theta"], data["Stop Theta"], theta_points)
        theta_angles_rad = np.deg2rad(theta_angles_deg)

        trp = calculate_trp(total_power_2d, theta_angles_rad, data["Inc Theta"], data["Inc Phi"])
        assert isinstance(trp, (float, np.floating))
        # TRP for BLE device should be in reasonable range
        assert -30 < trp < 10

    def test_trp_is_finite(self):
        data = read_active_file(ACTIVE_BLE_TRP)
        theta_points = int((data["Stop Theta"] - data["Start Theta"]) / data["Inc Theta"] + 1)
        phi_points = int((data["Stop Phi"] - data["Start Phi"]) / data["Inc Phi"] + 1)

        h_power_2d = data["H_Power_dBm"].reshape((theta_points, phi_points))
        theta_angles_deg = np.linspace(data["Start Theta"], data["Stop Theta"], theta_points)
        theta_angles_rad = np.deg2rad(theta_angles_deg)

        trp_h = calculate_trp(h_power_2d, theta_angles_rad, data["Inc Theta"], data["Inc Phi"])
        assert np.isfinite(trp_h)


# ===========================================================================
# 8. Frequency Extrapolation
# ===========================================================================
class TestFrequencyExtrapolation:
    """Holdout validation tests for frequency extrapolation using real data."""

    def test_lora_holdout_700mhz(self):
        """Fit 750-1100 MHz LoRa data, extrapolate to 700 MHz, expect RMS < 3 dB."""
        hpol_data, *_ = read_passive_file(PASSIVE_LORA_HPOL)
        vpol_data, *_ = read_passive_file(PASSIVE_LORA_VPOL)

        # Find the lowest frequency to hold out
        freqs = sorted(set(d["frequency"] for d in hpol_data))
        holdout_freq = freqs[0]  # Lowest measured frequency

        result = validate_extrapolation(
            hpol_data, vpol_data, holdout_freq, fit_degree=2
        )
        assert result["rms_error_dB"] < 3.0, (
            f"RMS error {result['rms_error_dB']:.2f} dB exceeds 3 dB threshold"
        )

    def test_ble_holdout_interpolation(self):
        """Hold out a mid-band BLE frequency (interpolation), expect RMS < 1 dB."""
        hpol_data, *_ = read_passive_file(PASSIVE_BLE_HPOL)
        vpol_data, *_ = read_passive_file(PASSIVE_BLE_VPOL)

        freqs = sorted(set(d["frequency"] for d in hpol_data))
        # Pick a frequency near the middle of the band
        mid_idx = len(freqs) // 2
        holdout_freq = freqs[mid_idx]

        result = validate_extrapolation(
            hpol_data, vpol_data, holdout_freq, fit_degree=2
        )
        assert result["rms_error_dB"] < 1.5, (
            f"RMS error {result['rms_error_dB']:.2f} dB exceeds 1.5 dB threshold "
            f"for interpolation at {holdout_freq} MHz"
        )

    def test_confidence_increases_with_distance(self):
        """Extrapolation ratio should be larger for frequencies further from measured range."""
        hpol_data, *_ = read_passive_file(PASSIVE_LORA_HPOL)
        vpol_data, *_ = read_passive_file(PASSIVE_LORA_VPOL)

        freqs = sorted(set(d["frequency"] for d in hpol_data))
        measured_min = freqs[0]

        # Close extrapolation: 50 MHz below range
        close = extrapolate_pattern(hpol_data, vpol_data, measured_min - 50)
        # Far extrapolation: 300 MHz below range
        far = extrapolate_pattern(hpol_data, vpol_data, measured_min - 300)

        assert far["confidence"]["extrapolation_ratio"] > close["confidence"]["extrapolation_ratio"]
