"""
UWB Antenna Analysis Module

Core calculation functions for Ultra-Wideband antenna characterization:
- Touchstone .s2p parsing
- Phase reconstruction from group delay
- System Fidelity Factor (SFF) computation
- Group delay analysis
- Transfer function extraction
- Impulse response computation
- S11/VSWR return loss analysis
- Multi-angle SFF sweeps

All functions are pure numpy/scipy — no matplotlib imports.
"""

import numpy as np
from scipy.integrate import cumulative_trapezoid
from scipy.signal import correlate

C = 299792458.0  # speed of light (m/s)


# ──────────────────────────────────────────────────────────────────────────────
# 1a. Touchstone .s2p Parser
# ──────────────────────────────────────────────────────────────────────────────

def parse_touchstone(filepath: str) -> dict:
    """Parse a Touchstone .s2p file into complex S-parameter arrays.

    Handles RI (Real/Imaginary), MA (Magnitude/Angle), and DB (dB/Angle)
    formats, and Hz/kHz/MHz/GHz frequency units.

    Args:
        filepath: Path to the .s2p file.

    Returns:
        dict with keys:
            'freq_hz': 1D array of frequencies in Hz
            's11', 's21', 's12', 's22': 1D complex arrays
            'z0': reference impedance (float)
    """
    freq_mult = {'HZ': 1.0, 'KHZ': 1e3, 'MHZ': 1e6, 'GHZ': 1e9}
    data_format = 'MA'
    z0 = 50.0
    multiplier = 1e9  # default GHz

    rows = []

    with open(filepath, 'r', encoding='utf-8-sig') as f:
        for line in f:
            stripped = line.strip()
            if not stripped or stripped.startswith('!'):
                continue
            if stripped.startswith('#'):
                # Option line: # <freq_unit> S <format> R <z0>
                parts = stripped[1:].split()
                for i, p in enumerate(parts):
                    pu = p.upper()
                    if pu in freq_mult:
                        multiplier = freq_mult[pu]
                    elif pu in ('RI', 'MA', 'DB'):
                        data_format = pu
                    elif pu == 'R' and i + 1 < len(parts):
                        try:
                            z0 = float(parts[i + 1])
                        except ValueError:
                            pass
                continue

            # Data line: freq s11_p1 s11_p2 s21_p1 s21_p2 s12_p1 s12_p2 s22_p1 s22_p2
            values = stripped.split()
            if len(values) < 9:
                continue
            try:
                row = [float(v) for v in values[:9]]
            except ValueError:
                continue
            rows.append(row)

    if not rows:
        raise ValueError(f"No data found in Touchstone file: {filepath}")

    data = np.array(rows)
    freq_hz = data[:, 0] * multiplier

    def _to_complex(p1, p2, fmt):
        if fmt == 'RI':
            return p1 + 1j * p2
        elif fmt == 'MA':
            return p1 * np.exp(1j * np.deg2rad(p2))
        elif fmt == 'DB':
            mag = 10 ** (p1 / 20.0)
            return mag * np.exp(1j * np.deg2rad(p2))
        raise ValueError(f"Unknown Touchstone format: {fmt}")

    s11 = _to_complex(data[:, 1], data[:, 2], data_format)
    s21 = _to_complex(data[:, 3], data[:, 4], data_format)
    s12 = _to_complex(data[:, 5], data[:, 6], data_format)
    s22 = _to_complex(data[:, 7], data[:, 8], data_format)

    return {
        'freq_hz': freq_hz,
        's11': s11,
        's21': s21,
        's12': s12,
        's22': s22,
        'z0': z0,
    }


# ──────────────────────────────────────────────────────────────────────────────
# 1b. Phase Reconstruction from Group Delay
# ──────────────────────────────────────────────────────────────────────────────

def reconstruct_phase_from_group_delay(freq_hz, group_delay_s, reference_phase_rad=0.0):
    """Reconstruct phase from group delay via cumulative integration.

    phi(f) = phi_0 - 2*pi * integral(tau_g(f') df')

    Args:
        freq_hz: 1D array of frequencies in Hz.
        group_delay_s: 1D array of group delay in seconds.
        reference_phase_rad: Phase at the first frequency point (radians).

    Returns:
        1D array of reconstructed phase in radians.
    """
    freq_hz = np.asarray(freq_hz, dtype=float)
    group_delay_s = np.asarray(group_delay_s, dtype=float)

    integral = cumulative_trapezoid(group_delay_s, freq_hz, initial=0.0)
    phase_rad = reference_phase_rad - 2.0 * np.pi * integral
    return phase_rad


# ──────────────────────────────────────────────────────────────────────────────
# 1c. Complex S21 from S2VNA Data
# ──────────────────────────────────────────────────────────────────────────────

def build_complex_s21_from_s2vna(freq_hz, s21_dB, group_delay_s, reference_phase_rad=0.0):
    """Build complex S21 from S2VNA magnitude + group delay data.

    Phase is reconstructed from group delay, then combined with magnitude.

    Args:
        freq_hz: 1D array of frequencies in Hz.
        s21_dB: 1D array of S21 magnitude in dB.
        group_delay_s: 1D array of group delay in seconds.
        reference_phase_rad: Phase at the first frequency point (radians).

    Returns:
        1D complex array of S21.
    """
    phase_rad = reconstruct_phase_from_group_delay(
        freq_hz, group_delay_s, reference_phase_rad
    )
    magnitude = 10 ** (np.asarray(s21_dB, dtype=float) / 20.0)
    return magnitude * np.exp(1j * phase_rad)


# ──────────────────────────────────────────────────────────────────────────────
# 1d. Input Pulse Library
# ──────────────────────────────────────────────────────────────────────────────

def generate_gaussian_monocycle(t, sigma=65e-12, center=None):
    """Generate a Gaussian monocycle (1st derivative of Gaussian).

    Args:
        t: 1D time array in seconds.
        sigma: Pulse width parameter in seconds.
        center: Pulse center time; defaults to middle of t.

    Returns:
        1D array of the pulse waveform.
    """
    t = np.asarray(t, dtype=float)
    if center is None:
        center = (t[0] + t[-1]) / 2.0
    tau = (t - center) / sigma
    return -tau * np.exp(-0.5 * tau ** 2)


def generate_modulated_gaussian(t, f0=6.85e9, sigma=150e-12, center=None):
    """Generate a carrier-modulated Gaussian pulse.

    Args:
        t: 1D time array in seconds.
        f0: Carrier frequency in Hz.
        sigma: Gaussian envelope width in seconds.
        center: Pulse center time; defaults to middle of t.

    Returns:
        1D array of the pulse waveform.
    """
    t = np.asarray(t, dtype=float)
    if center is None:
        center = (t[0] + t[-1]) / 2.0
    envelope = np.exp(-0.5 * ((t - center) / sigma) ** 2)
    return np.cos(2 * np.pi * f0 * (t - center)) * envelope


def generate_5th_derivative_gaussian(t, sigma=51e-12, center=None):
    """Generate 5th derivative Gaussian pulse (FCC mask compliant).

    Args:
        t: 1D time array in seconds.
        sigma: Pulse width parameter in seconds.
        center: Pulse center time; defaults to middle of t.

    Returns:
        1D array of the pulse waveform.
    """
    t = np.asarray(t, dtype=float)
    if center is None:
        center = (t[0] + t[-1]) / 2.0
    tau = (t - center) / sigma
    tau2 = tau ** 2
    # H5(x) = x^5 - 10*x^3 + 15*x  (probabilist's Hermite)
    h5 = tau ** 5 - 10 * tau ** 3 + 15 * tau
    return h5 * np.exp(-0.5 * tau2)


_PULSE_GENERATORS = {
    'gaussian_monocycle': generate_gaussian_monocycle,
    'modulated_gaussian': generate_modulated_gaussian,
    '5th_derivative_gaussian': generate_5th_derivative_gaussian,
}


# ──────────────────────────────────────────────────────────────────────────────
# 1e. System Fidelity Factor
# ──────────────────────────────────────────────────────────────────────────────

def calculate_sff(freq_hz, s21_complex, pulse_type='gaussian_monocycle',
                  sigma=65e-12, nfft=8192):
    """Compute System Fidelity Factor (SFF) using cross-correlation.

    SFF = max_tau |<s(t), r(t-tau)>| / (||s|| * ||r||)

    The input pulse is always carrier-modulated to center on the measurement
    band so that its spectrum overlaps with the S21 data. For 'gaussian_monocycle'
    and '5th_derivative_gaussian', a modulated Gaussian centered on the band is
    used instead, with sigma derived from the bandwidth.

    Args:
        freq_hz: 1D array of measurement frequencies in Hz.
        s21_complex: 1D complex array of S21 transfer function.
        pulse_type: One of 'gaussian_monocycle', 'modulated_gaussian',
                    '5th_derivative_gaussian'.
        sigma: Pulse width parameter passed to the generator.
        nfft: FFT size for time-domain conversion.

    Returns:
        dict with keys:
            'sff': float in [0, 1]
            'quality': str quality label
            'input_pulse': 1D array
            'output_pulse': 1D array
            'time_s': 1D time array
            'peak_delay_s': float estimated delay
    """
    freq_hz = np.asarray(freq_hz, dtype=float)
    s21_complex = np.asarray(s21_complex, dtype=complex)

    if pulse_type not in _PULSE_GENERATORS:
        raise ValueError(f"Unknown pulse type: {pulse_type}. "
                         f"Choose from {list(_PULSE_GENERATORS.keys())}")

    f_min, f_max = freq_hz[0], freq_hz[-1]
    f0 = (f_min + f_max) / 2.0
    bw = f_max - f_min

    # Use 4x the max frequency for adequate sampling
    fs = 4.0 * f_max
    dt = 1.0 / fs
    t = np.arange(nfft) * dt

    # All pulse types use a modulated Gaussian centered on the measurement band.
    # sigma is set to match the bandwidth for monocycle/5th-derivative types.
    if pulse_type == 'modulated_gaussian':
        input_pulse = generate_modulated_gaussian(t, f0=f0, sigma=sigma)
    else:
        # Auto-scale sigma to match measurement bandwidth
        band_sigma = 1.0 / bw
        input_pulse = generate_modulated_gaussian(t, f0=f0, sigma=band_sigma)

    # Input spectrum
    X_f = np.fft.fft(input_pulse, n=nfft)
    freq_fft = np.fft.fftfreq(nfft, dt)

    # Interpolate S21 onto FFT grid
    s21_interp_mag = np.interp(np.abs(freq_fft), freq_hz,
                               np.abs(s21_complex), left=0.0, right=0.0)
    s21_interp_phase = np.interp(np.abs(freq_fft), freq_hz,
                                 np.unwrap(np.angle(s21_complex)),
                                 left=0.0, right=0.0)
    S21_interp = s21_interp_mag * np.exp(1j * s21_interp_phase)

    # Output spectrum and time-domain signal
    Y_f = S21_interp * X_f
    output_pulse = np.real(np.fft.ifft(Y_f))

    # Normalized cross-correlation
    corr = correlate(output_pulse, input_pulse, mode='full')
    norm = np.sqrt(np.sum(input_pulse ** 2) * np.sum(output_pulse ** 2))

    if norm < 1e-30:
        return {
            'sff': 0.0,
            'quality': 'Poor',
            'input_pulse': input_pulse,
            'output_pulse': output_pulse,
            'time_s': t,
            'peak_delay_s': 0.0,
        }

    corr_norm = corr / norm
    sff = float(np.max(np.abs(corr_norm)))

    # Peak delay
    peak_idx = np.argmax(np.abs(corr_norm))
    lags = np.arange(-(len(input_pulse) - 1), len(output_pulse)) * dt
    peak_delay_s = float(lags[peak_idx])

    # Quality classification
    if sff >= 0.95:
        quality = 'Excellent'
    elif sff >= 0.85:
        quality = 'Very Good'
    elif sff >= 0.70:
        quality = 'Good'
    elif sff >= 0.50:
        quality = 'Fair'
    else:
        quality = 'Poor'

    return {
        'sff': sff,
        'quality': quality,
        'input_pulse': input_pulse,
        'output_pulse': output_pulse,
        'time_s': t,
        'peak_delay_s': peak_delay_s,
    }


# ──────────────────────────────────────────────────────────────────────────────
# 1f. Group Delay from Complex S21
# ──────────────────────────────────────────────────────────────────────────────

def compute_group_delay_from_s21(freq_hz, s21_complex):
    """Compute group delay from complex S21.

    tau_g = -d(unwrap(angle(S21))) / (2*pi*df)

    Args:
        freq_hz: 1D array of frequencies in Hz.
        s21_complex: 1D complex array of S21.

    Returns:
        dict with keys:
            'group_delay_s': 1D array of group delay in seconds
            'variation_s': float peak-to-peak variation
            'distance_error_m': float corresponding distance error
    """
    freq_hz = np.asarray(freq_hz, dtype=float)
    s21_complex = np.asarray(s21_complex, dtype=complex)

    phase = np.unwrap(np.angle(s21_complex))
    omega = 2.0 * np.pi * freq_hz
    group_delay = -np.gradient(phase, omega)

    variation = float(np.ptp(group_delay))
    distance_error = variation * C

    return {
        'group_delay_s': group_delay,
        'variation_s': variation,
        'distance_error_m': distance_error,
    }


# ──────────────────────────────────────────────────────────────────────────────
# 1g. Transfer Function Extraction
# ──────────────────────────────────────────────────────────────────────────────

def extract_transfer_function(freq_hz, s21_complex, distance_m=1.0):
    """Extract antenna transfer function by removing free-space channel.

    H(f) = S21(f) * (4*pi*f*d/c) * exp(j*2*pi*f*d/c)

    Args:
        freq_hz: 1D array of frequencies in Hz.
        s21_complex: 1D complex array of S21.
        distance_m: Measurement distance in meters.

    Returns:
        dict with keys:
            'H_complex': 1D complex transfer function
            'H_mag_dB': 1D magnitude in dB
            'H_phase_deg': 1D phase in degrees
    """
    freq_hz = np.asarray(freq_hz, dtype=float)
    s21_complex = np.asarray(s21_complex, dtype=complex)

    free_space = (4.0 * np.pi * freq_hz * distance_m) / C
    phase_shift = np.exp(1j * 2.0 * np.pi * freq_hz * distance_m / C)

    H = s21_complex * free_space * phase_shift

    H_mag = np.abs(H)
    H_mag_dB = np.where(H_mag > 0, 20.0 * np.log10(H_mag), -200.0)
    H_phase_deg = np.rad2deg(np.unwrap(np.angle(H)))

    return {
        'H_complex': H,
        'H_mag_dB': H_mag_dB,
        'H_phase_deg': H_phase_deg,
    }


# ──────────────────────────────────────────────────────────────────────────────
# 1h. Impulse Response
# ──────────────────────────────────────────────────────────────────────────────

def compute_impulse_response(freq_hz, transfer_function, nfft=8192, window='blackman'):
    """Compute impulse response from transfer function via IFFT.

    Args:
        freq_hz: 1D array of frequencies in Hz.
        transfer_function: 1D complex array H(f).
        nfft: FFT size.
        window: Window function name ('blackman', 'hann', 'hamming', 'none').

    Returns:
        dict with keys:
            'time_s': 1D time array
            'h_t': 1D impulse response array
            'pulse_width_s': float -3dB pulse width
            'ringing_dB': float peak sidelobe level relative to main peak
    """
    freq_hz = np.asarray(freq_hz, dtype=float)
    transfer_function = np.asarray(transfer_function, dtype=complex)

    f_max = freq_hz[-1]
    dt = 1.0 / (2.0 * f_max)

    # Build frequency-domain array with Hermitian symmetry
    freq_fft = np.fft.fftfreq(nfft, dt)
    H_interp_mag = np.interp(np.abs(freq_fft), freq_hz,
                              np.abs(transfer_function), left=0.0, right=0.0)
    H_interp_phase = np.interp(np.abs(freq_fft), freq_hz,
                                np.unwrap(np.angle(transfer_function)),
                                left=0.0, right=0.0)
    H_fft = H_interp_mag * np.exp(1j * H_interp_phase)

    # Apply window
    if window != 'none':
        win_func = getattr(np, window, None)
        if win_func is not None:
            w = win_func(nfft)
        else:
            w = np.ones(nfft)
        H_fft *= w
    else:
        w = np.ones(nfft)

    h_t = np.real(np.fft.ifft(H_fft))
    t = np.arange(nfft) * dt

    # Pulse width at -3dB
    h_abs = np.abs(h_t)
    peak_val = np.max(h_abs)
    if peak_val > 0:
        threshold = peak_val / np.sqrt(2)  # -3dB
        above = h_abs >= threshold
        indices = np.where(above)[0]
        if len(indices) >= 2:
            pulse_width_s = float((indices[-1] - indices[0]) * dt)
        else:
            pulse_width_s = float(dt)

        # Ringing: peak sidelobe relative to main peak
        peak_idx = np.argmax(h_abs)
        # Zero out main lobe region
        h_copy = h_abs.copy()
        # Find main lobe extent
        left = peak_idx
        while left > 0 and h_copy[left] > 0.1 * peak_val:
            left -= 1
        right = peak_idx
        while right < len(h_copy) - 1 and h_copy[right] > 0.1 * peak_val:
            right += 1
        h_copy[left:right + 1] = 0.0
        sidelobe_peak = np.max(h_copy) if np.any(h_copy > 0) else 1e-30
        ringing_dB = float(20.0 * np.log10(sidelobe_peak / peak_val))
    else:
        pulse_width_s = 0.0
        ringing_dB = -np.inf

    return {
        'time_s': t,
        'h_t': h_t,
        'pulse_width_s': pulse_width_s,
        'ringing_dB': ringing_dB,
    }


# ──────────────────────────────────────────────────────────────────────────────
# 1i. S11/VSWR Analysis
# ──────────────────────────────────────────────────────────────────────────────

def analyze_return_loss(freq_hz, s11_dB, threshold_dB=-10.0):
    """Analyze return loss to determine impedance bandwidth.

    Args:
        freq_hz: 1D array of frequencies in Hz.
        s11_dB: 1D array of S11 magnitude in dB (negative values).
        threshold_dB: Return loss threshold for bandwidth (default -10 dB).

    Returns:
        dict with keys:
            'vswr': 1D array of VSWR
            'bandwidth_hz': float impedance bandwidth in Hz
            'band_start_hz': float lower band edge
            'band_stop_hz': float upper band edge
            'fractional_bandwidth': float (BW / center_freq)
            'min_s11_dB': float minimum S11
    """
    freq_hz = np.asarray(freq_hz, dtype=float)
    s11_dB = np.asarray(s11_dB, dtype=float)

    # Convert S11(dB) to VSWR
    s11_linear = 10 ** (s11_dB / 20.0)
    # Clamp to avoid division by zero for |S11| near 1
    s11_linear = np.clip(s11_linear, 0.0, 0.9999)
    vswr = (1.0 + s11_linear) / (1.0 - s11_linear)

    # Find bandwidth where S11 < threshold
    below = s11_dB <= threshold_dB
    if np.any(below):
        indices = np.where(below)[0]
        # Find the widest contiguous band
        splits = np.split(indices, np.where(np.diff(indices) > 1)[0] + 1)
        widest = max(splits, key=len)
        band_start_hz = float(freq_hz[widest[0]])
        band_stop_hz = float(freq_hz[widest[-1]])
        bandwidth_hz = band_stop_hz - band_start_hz
        center_freq = (band_start_hz + band_stop_hz) / 2.0
        fractional_bandwidth = bandwidth_hz / center_freq if center_freq > 0 else 0.0
    else:
        band_start_hz = 0.0
        band_stop_hz = 0.0
        bandwidth_hz = 0.0
        fractional_bandwidth = 0.0

    return {
        'vswr': vswr,
        'bandwidth_hz': bandwidth_hz,
        'band_start_hz': band_start_hz,
        'band_stop_hz': band_stop_hz,
        'fractional_bandwidth': fractional_bandwidth,
        'min_s11_dB': float(np.min(s11_dB)),
    }


# ──────────────────────────────────────────────────────────────────────────────
# 1j. Multi-angle SFF
# ──────────────────────────────────────────────────────────────────────────────

def calculate_sff_vs_angle(angle_data, pulse_type='gaussian_monocycle', sigma=65e-12):
    """Compute SFF at multiple angle orientations.

    Args:
        angle_data: list of dicts, each with keys:
            'angle_deg': float angle in degrees
            'freq_hz': 1D array of frequencies
            's21_complex': 1D complex S21
        pulse_type: Pulse type for SFF calculation.
        sigma: Pulse width parameter.

    Returns:
        dict with keys:
            'angles': list of angle values
            'sff_values': list of SFF values
            'qualities': list of quality strings
            'mean_sff': float mean SFF
    """
    angles = []
    sff_values = []
    qualities = []

    for entry in angle_data:
        angle = entry['angle_deg']
        result = calculate_sff(
            entry['freq_hz'],
            entry['s21_complex'],
            pulse_type=pulse_type,
            sigma=sigma,
        )
        angles.append(angle)
        sff_values.append(result['sff'])
        qualities.append(result['quality'])

    mean_sff = float(np.mean(sff_values)) if sff_values else 0.0

    return {
        'angles': angles,
        'sff_values': sff_values,
        'qualities': qualities,
        'mean_sff': mean_sff,
    }
