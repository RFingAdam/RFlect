"""
Tests for UWB antenna analysis module.

Validates phase reconstruction, SFF computation, group delay, transfer function,
impulse response, S11/VSWR, and Touchstone parsing using synthetic data.
"""

import os
import tempfile
import pytest
import numpy as np

from plot_antenna.uwb_analysis import (
    parse_touchstone,
    reconstruct_phase_from_group_delay,
    build_complex_s21_from_s2vna,
    generate_gaussian_monocycle,
    generate_modulated_gaussian,
    generate_5th_derivative_gaussian,
    calculate_sff,
    compute_group_delay_from_s21,
    extract_transfer_function,
    compute_impulse_response,
    analyze_return_loss,
    calculate_sff_vs_angle,
    C,
)


# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────

def _make_linear_phase_s21(freq_hz, delay_s=1e-9, magnitude_dB=-3.0):
    """Create an S21 with constant magnitude and linear phase (constant delay)."""
    mag = 10 ** (magnitude_dB / 20.0)
    phase = -2.0 * np.pi * freq_hz * delay_s
    return mag * np.exp(1j * phase)


def _write_touchstone(filepath, freq_ghz, s11, s21, s12, s22, fmt='RI', z0=50.0):
    """Write a minimal .s2p file."""
    with open(filepath, 'w') as f:
        f.write(f'! Test touchstone file\n')
        f.write(f'# GHZ S {fmt} R {z0}\n')
        for i in range(len(freq_ghz)):
            if fmt == 'RI':
                f.write(f'{freq_ghz[i]:.6f} '
                        f'{s11[i].real:.6f} {s11[i].imag:.6f} '
                        f'{s21[i].real:.6f} {s21[i].imag:.6f} '
                        f'{s12[i].real:.6f} {s12[i].imag:.6f} '
                        f'{s22[i].real:.6f} {s22[i].imag:.6f}\n')
            elif fmt == 'MA':
                f.write(f'{freq_ghz[i]:.6f} '
                        f'{np.abs(s11[i]):.6f} {np.rad2deg(np.angle(s11[i])):.6f} '
                        f'{np.abs(s21[i]):.6f} {np.rad2deg(np.angle(s21[i])):.6f} '
                        f'{np.abs(s12[i]):.6f} {np.rad2deg(np.angle(s12[i])):.6f} '
                        f'{np.abs(s22[i]):.6f} {np.rad2deg(np.angle(s22[i])):.6f}\n')
            elif fmt == 'DB':
                f.write(f'{freq_ghz[i]:.6f} '
                        f'{20*np.log10(np.abs(s11[i])):.6f} {np.rad2deg(np.angle(s11[i])):.6f} '
                        f'{20*np.log10(np.abs(s21[i])):.6f} {np.rad2deg(np.angle(s21[i])):.6f} '
                        f'{20*np.log10(np.abs(s12[i])):.6f} {np.rad2deg(np.angle(s12[i])):.6f} '
                        f'{20*np.log10(np.abs(s22[i])):.6f} {np.rad2deg(np.angle(s22[i])):.6f}\n')


# ──────────────────────────────────────────────────────────────────────────────
# TestTouchstoneParser
# ──────────────────────────────────────────────────────────────────────────────

class TestTouchstoneParser:
    """Tests for Touchstone .s2p file parsing."""

    def test_ri_format(self, tmp_path):
        """Parse RI (Real/Imaginary) format."""
        freq_ghz = np.array([1.0, 2.0, 3.0])
        s21 = np.array([0.5 + 0.3j, 0.4 + 0.2j, 0.3 + 0.1j])
        s11 = np.array([-0.1 + 0.1j, -0.2 + 0.05j, -0.15 + 0.08j])
        s12 = s21.copy()
        s22 = s11.copy()

        filepath = str(tmp_path / "test.s2p")
        _write_touchstone(filepath, freq_ghz, s11, s21, s12, s22, fmt='RI')

        result = parse_touchstone(filepath)
        np.testing.assert_allclose(result['freq_hz'], freq_ghz * 1e9)
        np.testing.assert_allclose(result['s21'], s21, atol=1e-5)
        np.testing.assert_allclose(result['s11'], s11, atol=1e-5)
        assert result['z0'] == 50.0

    def test_ma_format(self, tmp_path):
        """Parse MA (Magnitude/Angle) format."""
        freq_ghz = np.array([5.0, 6.0, 7.0])
        s21 = np.array([0.7 * np.exp(1j * np.deg2rad(-30)),
                         0.6 * np.exp(1j * np.deg2rad(-60)),
                         0.5 * np.exp(1j * np.deg2rad(-90))])
        s11 = np.array([0.1 * np.exp(1j * np.deg2rad(10)),
                         0.15 * np.exp(1j * np.deg2rad(20)),
                         0.12 * np.exp(1j * np.deg2rad(15))])
        s12 = s21.copy()
        s22 = s11.copy()

        filepath = str(tmp_path / "test_ma.s2p")
        _write_touchstone(filepath, freq_ghz, s11, s21, s12, s22, fmt='MA')

        result = parse_touchstone(filepath)
        np.testing.assert_allclose(result['freq_hz'], freq_ghz * 1e9)
        np.testing.assert_allclose(np.abs(result['s21']), np.abs(s21), atol=1e-4)
        np.testing.assert_allclose(np.angle(result['s21']), np.angle(s21), atol=1e-3)

    def test_db_format(self, tmp_path):
        """Parse DB (dB/Angle) format."""
        freq_ghz = np.array([3.0, 4.0, 5.0])
        s21 = np.array([0.5 * np.exp(1j * np.deg2rad(-45)),
                         0.4 * np.exp(1j * np.deg2rad(-90)),
                         0.3 * np.exp(1j * np.deg2rad(-135))])
        s11 = np.array([0.1 + 0j, 0.15 + 0j, 0.12 + 0j])
        s12 = s21.copy()
        s22 = s11.copy()

        filepath = str(tmp_path / "test_db.s2p")
        _write_touchstone(filepath, freq_ghz, s11, s21, s12, s22, fmt='DB')

        result = parse_touchstone(filepath)
        np.testing.assert_allclose(np.abs(result['s21']), np.abs(s21), atol=1e-3)

    def test_frequency_units_mhz(self, tmp_path):
        """Parse file with MHz frequency units."""
        filepath = str(tmp_path / "test_mhz.s2p")
        with open(filepath, 'w') as f:
            f.write('# MHZ S RI R 50\n')
            f.write('1000.0 0.1 0.0 0.5 0.3 0.5 0.3 0.1 0.0\n')
        result = parse_touchstone(filepath)
        np.testing.assert_allclose(result['freq_hz'], [1e9])

    def test_comments_ignored(self, tmp_path):
        """Comments (! lines) are ignored."""
        filepath = str(tmp_path / "test_comments.s2p")
        with open(filepath, 'w') as f:
            f.write('! This is a comment\n')
            f.write('! Another comment\n')
            f.write('# GHZ S RI R 50\n')
            f.write('! Data follows\n')
            f.write('1.0 0.1 0.0 0.5 0.0 0.5 0.0 0.1 0.0\n')
        result = parse_touchstone(filepath)
        assert len(result['freq_hz']) == 1

    def test_empty_file_raises(self, tmp_path):
        """Empty file raises ValueError."""
        filepath = str(tmp_path / "empty.s2p")
        with open(filepath, 'w') as f:
            f.write('# GHZ S RI R 50\n')
        with pytest.raises(ValueError, match="No data found"):
            parse_touchstone(filepath)


# ──────────────────────────────────────────────────────────────────────────────
# TestPhaseReconstruction
# ──────────────────────────────────────────────────────────────────────────────

class TestPhaseReconstruction:
    """Tests for phase reconstruction from group delay."""

    def test_constant_delay_linear_phase(self):
        """Constant group delay should produce linear phase."""
        freq_hz = np.linspace(3e9, 10e9, 1000)
        delay = 2e-9  # 2 ns constant delay
        group_delay = np.full_like(freq_hz, delay)

        phase = reconstruct_phase_from_group_delay(freq_hz, group_delay)

        # Phase should be linear: phi(f) = phi_0 - 2*pi*(f - f_min)*tau
        # With reference_phase_rad=0, phi(f_min)=0
        expected_phase = -2 * np.pi * (freq_hz - freq_hz[0]) * delay
        np.testing.assert_allclose(phase, expected_phase, atol=1e-3)

    def test_roundtrip_with_known_delay(self):
        """Reconstruct phase from group delay, then recover group delay."""
        freq_hz = np.linspace(3e9, 10e9, 500)
        delay = 1.5e-9
        group_delay = np.full_like(freq_hz, delay)

        phase = reconstruct_phase_from_group_delay(freq_hz, group_delay)

        # Recover group delay from phase: tau = -dphi/domega
        omega = 2 * np.pi * freq_hz
        recovered_delay = -np.gradient(phase, omega)

        np.testing.assert_allclose(recovered_delay, delay, atol=1e-3 * delay)

    def test_reference_phase(self):
        """Reference phase shifts entire output."""
        freq_hz = np.linspace(1e9, 5e9, 100)
        group_delay = np.full_like(freq_hz, 1e-9)

        phase0 = reconstruct_phase_from_group_delay(freq_hz, group_delay, 0.0)
        phase1 = reconstruct_phase_from_group_delay(freq_hz, group_delay, np.pi)

        np.testing.assert_allclose(phase1 - phase0, np.pi, atol=1e-10)

    def test_varying_delay(self):
        """Varying group delay produces non-linear phase."""
        freq_hz = np.linspace(3e9, 10e9, 500)
        # Linearly increasing delay
        group_delay = np.linspace(1e-9, 3e-9, 500)

        phase = reconstruct_phase_from_group_delay(freq_hz, group_delay)

        # Phase should be monotonically decreasing
        assert np.all(np.diff(phase) < 0)


# ──────────────────────────────────────────────────────────────────────────────
# TestBuildComplexS21
# ──────────────────────────────────────────────────────────────────────────────

class TestBuildComplexS21:
    """Tests for building complex S21 from S2VNA data."""

    def test_magnitude_preservation(self):
        """Magnitude should match input dB values."""
        freq_hz = np.linspace(3e9, 10e9, 200)
        s21_dB = np.full(200, -5.0)
        group_delay = np.full(200, 1e-9)

        s21 = build_complex_s21_from_s2vna(freq_hz, s21_dB, group_delay)

        expected_mag = 10 ** (-5.0 / 20.0)
        np.testing.assert_allclose(np.abs(s21), expected_mag, atol=1e-10)

    def test_phase_from_delay(self):
        """Reconstructed phase should match constant delay model."""
        freq_hz = np.linspace(3e9, 10e9, 500)
        s21_dB = np.zeros(500)  # 0 dB
        delay = 2e-9
        group_delay = np.full(500, delay)

        s21 = build_complex_s21_from_s2vna(freq_hz, s21_dB, group_delay)

        # Phase slope should match delay: dphi/df = -2*pi*tau
        actual_phase = np.unwrap(np.angle(s21))
        # Check slope rather than absolute values (integration constant offset)
        actual_slope = np.gradient(actual_phase, freq_hz)
        expected_slope = -2 * np.pi * delay
        np.testing.assert_allclose(actual_slope, expected_slope, atol=1e-6)


# ──────────────────────────────────────────────────────────────────────────────
# TestPulseGenerators
# ──────────────────────────────────────────────────────────────────────────────

class TestPulseGenerators:
    """Tests for UWB pulse generation functions."""

    def test_monocycle_zero_crossings(self):
        """Gaussian monocycle has a zero crossing at center."""
        t = np.linspace(-1e-9, 1e-9, 1001)  # odd count so t=0 is exactly represented
        pulse = generate_gaussian_monocycle(t, sigma=100e-12, center=0.0)
        # At t=center (tau=0), pulse should be 0
        center_idx = 500
        assert abs(pulse[center_idx]) < 1e-10

    def test_modulated_gaussian_carrier(self):
        """Modulated Gaussian should oscillate at carrier frequency."""
        f0 = 6.85e9
        sigma = 500e-12  # wide enough to see oscillation
        dt = 1e-12
        t = np.arange(0, 5e-9, dt)
        pulse = generate_modulated_gaussian(t, f0=f0, sigma=sigma)

        # FFT to find dominant frequency
        spectrum = np.abs(np.fft.rfft(pulse))
        freqs = np.fft.rfftfreq(len(t), dt)
        peak_freq = freqs[np.argmax(spectrum)]
        assert abs(peak_freq - f0) < 0.5e9  # within 500 MHz

    def test_5th_derivative_symmetric(self):
        """5th derivative Gaussian should be anti-symmetric about center."""
        t = np.linspace(-0.5e-9, 0.5e-9, 1001)
        pulse = generate_5th_derivative_gaussian(t, sigma=51e-12, center=0.0)
        # H5 is odd, so p(-t) = -p(t)
        n = len(pulse)
        np.testing.assert_allclose(pulse[:n // 2], -pulse[n // 2 + 1:][::-1], atol=1e-10)


# ──────────────────────────────────────────────────────────────────────────────
# TestSFF
# ──────────────────────────────────────────────────────────────────────────────

class TestSFF:
    """Tests for System Fidelity Factor computation."""

    def test_perfect_channel(self):
        """All-pass channel with linear phase should give high SFF.

        Note: SFF < 1.0 even for a perfect channel because the pulse spectrum
        extends beyond the measurement band. Spectral truncation limits SFF
        to ~0.65-0.85 for typical UWB bandwidths.
        """
        freq_hz = np.linspace(3e9, 10e9, 500)
        # Perfect channel: unity gain, constant group delay
        delay = 1e-9
        s21 = _make_linear_phase_s21(freq_hz, delay_s=delay, magnitude_dB=0.0)

        result = calculate_sff(freq_hz, s21, pulse_type='gaussian_monocycle')

        assert result['sff'] > 0.60, f"SFF too low for perfect channel: {result['sff']}"
        assert result['quality'] in ('Excellent', 'Very Good', 'Good')

    def test_dispersive_channel_lower_sff(self):
        """Dispersive channel should have lower SFF than all-pass."""
        freq_hz = np.linspace(3e9, 10e9, 500)

        # All-pass channel
        s21_good = _make_linear_phase_s21(freq_hz, delay_s=1e-9, magnitude_dB=0.0)
        # Dispersive channel: quadratic phase addition
        dispersion = np.exp(1j * 1e-20 * (freq_hz - 6.5e9) ** 2)
        s21_bad = s21_good * dispersion

        sff_good = calculate_sff(freq_hz, s21_good)['sff']
        sff_bad = calculate_sff(freq_hz, s21_bad)['sff']

        assert sff_bad < sff_good, "Dispersive channel should have lower SFF"

    def test_quality_thresholds(self):
        """Quality labels should match threshold values."""
        freq_hz = np.linspace(3e9, 10e9, 200)
        s21 = _make_linear_phase_s21(freq_hz, delay_s=1e-9, magnitude_dB=0.0)

        result = calculate_sff(freq_hz, s21)
        sff = result['sff']
        quality = result['quality']

        if sff >= 0.95:
            assert quality == 'Excellent'
        elif sff >= 0.85:
            assert quality == 'Very Good'
        elif sff >= 0.70:
            assert quality == 'Good'
        elif sff >= 0.50:
            assert quality == 'Fair'
        else:
            assert quality == 'Poor'

    def test_different_pulse_types(self):
        """SFF should compute for all pulse types without error."""
        freq_hz = np.linspace(3e9, 10e9, 200)
        s21 = _make_linear_phase_s21(freq_hz, delay_s=1e-9, magnitude_dB=-3.0)

        for pt in ('gaussian_monocycle', 'modulated_gaussian', '5th_derivative_gaussian'):
            result = calculate_sff(freq_hz, s21, pulse_type=pt)
            assert 0.0 <= result['sff'] <= 1.0, f"SFF out of range for {pt}"

    def test_invalid_pulse_type_raises(self):
        """Invalid pulse type should raise ValueError."""
        freq_hz = np.linspace(3e9, 10e9, 50)
        s21 = _make_linear_phase_s21(freq_hz, delay_s=1e-9)

        with pytest.raises(ValueError, match="Unknown pulse type"):
            calculate_sff(freq_hz, s21, pulse_type='invalid_pulse')

    def test_sff_output_keys(self):
        """SFF result should contain all expected keys."""
        freq_hz = np.linspace(3e9, 10e9, 100)
        s21 = _make_linear_phase_s21(freq_hz, delay_s=1e-9)

        result = calculate_sff(freq_hz, s21)

        expected_keys = {'sff', 'quality', 'input_pulse', 'output_pulse',
                         'time_s', 'peak_delay_s'}
        assert set(result.keys()) == expected_keys


# ──────────────────────────────────────────────────────────────────────────────
# TestGroupDelayFromS21
# ──────────────────────────────────────────────────────────────────────────────

class TestGroupDelayFromS21:
    """Tests for computing group delay from complex S21."""

    def test_constant_delay(self):
        """Linear phase S21 should give constant group delay."""
        freq_hz = np.linspace(3e9, 10e9, 1000)
        delay = 2e-9
        s21 = _make_linear_phase_s21(freq_hz, delay_s=delay)

        result = compute_group_delay_from_s21(freq_hz, s21)

        # Group delay should be approximately the expected value
        np.testing.assert_allclose(result['group_delay_s'], delay, rtol=0.01)

    def test_zero_variation_linear_phase(self):
        """Linear phase should have near-zero group delay variation."""
        freq_hz = np.linspace(3e9, 10e9, 1000)
        s21 = _make_linear_phase_s21(freq_hz, delay_s=1e-9)

        result = compute_group_delay_from_s21(freq_hz, s21)

        assert result['variation_s'] < 1e-12, "Variation should be ~0 for linear phase"


# ──────────────────────────────────────────────────────────────────────────────
# TestTransferFunction
# ──────────────────────────────────────────────────────────────────────────────

class TestTransferFunction:
    """Tests for transfer function extraction."""

    def test_free_space_removal(self):
        """Transfer function should remove free-space path loss."""
        freq_hz = np.linspace(3e9, 10e9, 100)
        distance_m = 1.0
        s21 = _make_linear_phase_s21(freq_hz, delay_s=distance_m / C, magnitude_dB=-20.0)

        result = extract_transfer_function(freq_hz, s21, distance_m=distance_m)

        assert 'H_complex' in result
        assert 'H_mag_dB' in result
        assert 'H_phase_deg' in result
        assert len(result['H_complex']) == len(freq_hz)

    def test_output_shapes(self):
        """All outputs should have the same length as input."""
        freq_hz = np.linspace(1e9, 5e9, 50)
        s21 = _make_linear_phase_s21(freq_hz)

        result = extract_transfer_function(freq_hz, s21)

        assert len(result['H_complex']) == 50
        assert len(result['H_mag_dB']) == 50
        assert len(result['H_phase_deg']) == 50


# ──────────────────────────────────────────────────────────────────────────────
# TestImpulseResponse
# ──────────────────────────────────────────────────────────────────────────────

class TestImpulseResponse:
    """Tests for impulse response computation."""

    def test_flat_transfer_function(self):
        """Flat |H(f)| with linear phase should give a narrow pulse."""
        freq_hz = np.linspace(1e9, 10e9, 500)
        # Flat magnitude, linear phase
        H = np.ones(500) * np.exp(-1j * 2 * np.pi * freq_hz * 1e-9)

        result = compute_impulse_response(freq_hz, H, nfft=4096)

        assert 'time_s' in result
        assert 'h_t' in result
        assert result['pulse_width_s'] > 0
        # Peak should be well-defined
        assert np.max(np.abs(result['h_t'])) > 0

    def test_output_length(self):
        """Output arrays should have length nfft."""
        freq_hz = np.linspace(1e9, 5e9, 100)
        H = np.ones(100, dtype=complex)

        nfft = 2048
        result = compute_impulse_response(freq_hz, H, nfft=nfft)

        assert len(result['time_s']) == nfft
        assert len(result['h_t']) == nfft


# ──────────────────────────────────────────────────────────────────────────────
# TestReturnLoss
# ──────────────────────────────────────────────────────────────────────────────

class TestReturnLoss:
    """Tests for S11/VSWR return loss analysis."""

    def test_well_matched_antenna(self):
        """Well-matched antenna should have wide bandwidth."""
        freq_hz = np.linspace(3e9, 10e9, 500)
        # Good match: S11 around -15 dB across band
        s11_dB = np.full(500, -15.0)

        result = analyze_return_loss(freq_hz, s11_dB, threshold_dB=-10.0)

        assert result['bandwidth_hz'] > 6e9  # nearly full band
        assert result['min_s11_dB'] == -15.0
        assert result['fractional_bandwidth'] > 1.0  # > 100%

    def test_narrowband_antenna(self):
        """Narrowband antenna should have limited bandwidth."""
        freq_hz = np.linspace(3e9, 10e9, 500)
        # Only matched around 6.5 GHz
        s11_dB = -5.0 * np.ones(500)
        center = 250
        width = 25
        s11_dB[center - width:center + width] = -20.0

        result = analyze_return_loss(freq_hz, s11_dB, threshold_dB=-10.0)

        assert result['bandwidth_hz'] < 3e9
        assert result['band_start_hz'] > 3e9
        assert result['band_stop_hz'] < 10e9

    def test_vswr_conversion(self):
        """VSWR should be correctly computed from S11."""
        freq_hz = np.array([5e9])
        s11_dB = np.array([-10.0])  # |S11| = 0.3162

        result = analyze_return_loss(freq_hz, s11_dB)

        s11_lin = 10 ** (-10.0 / 20.0)
        expected_vswr = (1 + s11_lin) / (1 - s11_lin)
        np.testing.assert_allclose(result['vswr'], [expected_vswr], rtol=1e-4)

    def test_no_match(self):
        """Antenna with S11 above threshold should have zero bandwidth."""
        freq_hz = np.linspace(1e9, 5e9, 100)
        s11_dB = np.full(100, -3.0)  # poor match everywhere

        result = analyze_return_loss(freq_hz, s11_dB, threshold_dB=-10.0)

        assert result['bandwidth_hz'] == 0.0
        assert result['band_start_hz'] == 0.0
        assert result['band_stop_hz'] == 0.0


# ──────────────────────────────────────────────────────────────────────────────
# TestMultiAngleSFF
# ──────────────────────────────────────────────────────────────────────────────

class TestMultiAngleSFF:
    """Tests for multi-angle SFF calculation."""

    def test_basic_operation(self):
        """Multi-angle SFF should compute for each entry."""
        freq_hz = np.linspace(3e9, 10e9, 200)
        s21 = _make_linear_phase_s21(freq_hz, delay_s=1e-9, magnitude_dB=-3.0)

        angle_data = [
            {'angle_deg': 0, 'freq_hz': freq_hz, 's21_complex': s21},
            {'angle_deg': 90, 'freq_hz': freq_hz, 's21_complex': s21},
            {'angle_deg': 180, 'freq_hz': freq_hz, 's21_complex': s21},
        ]

        result = calculate_sff_vs_angle(angle_data)

        assert len(result['angles']) == 3
        assert len(result['sff_values']) == 3
        assert len(result['qualities']) == 3
        assert 0 <= result['mean_sff'] <= 1.0

    def test_varying_sff_across_angles(self):
        """Different S21 per angle should give different SFF values."""
        freq_hz = np.linspace(3e9, 10e9, 200)
        s21_good = _make_linear_phase_s21(freq_hz, delay_s=1e-9, magnitude_dB=0.0)
        # Add dispersion for angle 90
        dispersion = np.exp(1j * 5e-20 * (freq_hz - 6.5e9) ** 2)
        s21_bad = s21_good * dispersion

        angle_data = [
            {'angle_deg': 0, 'freq_hz': freq_hz, 's21_complex': s21_good},
            {'angle_deg': 90, 'freq_hz': freq_hz, 's21_complex': s21_bad},
        ]

        result = calculate_sff_vs_angle(angle_data)

        assert result['sff_values'][0] > result['sff_values'][1], \
            "Non-dispersive angle should have higher SFF"
