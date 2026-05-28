"""
Advanced RF analysis methods (v6.0).

Pure, deterministic functions — no IO, no plotting, no LLM. Each has an analytic
oracle covered by tests:

- axial_ratio_from_hv         (#48) circular-polarization axial ratio / tilt / sense
- three_antenna_gain          (#48) absolute gain via the 3-antenna method
- array_factor_uniform        (#46) uniform linear array factor + steering + grating check
- planar_nf2ff                (#47) planar near-field -> far-field via 2D FFT
- time_gate_sparam            (#43) IFFT time-gating of an S-parameter sweep
- port_extension_deembed      (#43) reference-plane (electrical-length) de-embedding
- total_isotropic_sensitivity (#42) CTIA TIS (sphere-harmonic-mean of EIS)
"""

from __future__ import annotations

from typing import Dict, Tuple

import numpy as np

C_M_PER_S = 299792458.0


# ---------------------------------------------------------------------------
# #48 — circular polarization axial ratio + 3-antenna absolute gain
# ---------------------------------------------------------------------------


def axial_ratio_from_hv(mag_v: float, mag_h: float, phase_diff_deg: float) -> Dict[str, float]:
    """Polarization-ellipse axial ratio / tilt / sense from orthogonal H & V.

    Given the two orthogonal linear field magnitudes (E_theta = V, E_phi = H)
    and their phase difference delta = phase(V) - phase(H), compute the axial
    ratio of the polarization ellipse.

        AR_linear = sqrt( (S + R) / (S - R) ),  S = E1^2+E2^2,
                    R = sqrt(E1^4 + E2^4 + 2 E1^2 E2^2 cos(2 delta))
        tilt = 0.5 * atan2(2 E1 E2 cos(delta), E1^2 - E2^2)

    Returns AR in dB (0 dB = perfect circular, large = linear), tilt (deg), and
    sense ("RHCP"/"LHCP"/"linear"). For equal magnitudes and delta = +/-90 deg
    the result is circular (AR = 0 dB).
    """
    e1 = abs(float(mag_v))
    e2 = abs(float(mag_h))
    delta = np.deg2rad(float(phase_diff_deg))
    s = e1 * e1 + e2 * e2
    r = np.sqrt(e1**4 + e2**4 + 2 * e1 * e1 * e2 * e2 * np.cos(2 * delta))
    denom = s - r
    if s <= 0:
        return {"axial_ratio_db": float("inf"), "tilt_deg": 0.0, "sense": "undefined"}
    if denom <= 1e-15:
        ar_linear = float("inf")
    else:
        ar_linear = np.sqrt((s + r) / denom)
    ar_db = float("inf") if not np.isfinite(ar_linear) else 20.0 * np.log10(ar_linear)
    tilt = 0.5 * np.arctan2(2 * e1 * e2 * np.cos(delta), e1 * e1 - e2 * e2)

    sin2d = np.sin(delta)
    if ar_db > 20.0 or abs(sin2d) < 1e-6:
        sense = "linear"
    elif sin2d > 0:
        sense = "LHCP"
    else:
        sense = "RHCP"
    return {
        "axial_ratio_db": round(float(ar_db), 4) if np.isfinite(ar_db) else float("inf"),
        "tilt_deg": round(float(np.rad2deg(tilt)), 4),
        "sense": sense,
    }


def three_antenna_gain(
    m_ab_db: float, m_ac_db: float, m_bc_db: float, range_m: float, freq_mhz: float
) -> Dict[str, float]:
    """Absolute gain of three antennas via the classic 3-antenna method.

    Each measurement M_xy (dB) is the received/transmitted power ratio of the
    pair (x,y) at the given range. Friis: M_xy = G_x + G_y - FSPL. Solving:
        G_A = 0.5 (M_AB + M_AC - M_BC + FSPL), and cyclically for G_B, G_C.

    Returns gains (dBi) of antennas A, B, C plus the FSPL used.
    """
    freq_hz = float(freq_mhz) * 1e6
    fspl_db = 20.0 * np.log10(4.0 * np.pi * float(range_m) * freq_hz / C_M_PER_S)
    g_a = 0.5 * (m_ab_db + m_ac_db - m_bc_db + fspl_db)
    g_b = 0.5 * (m_ab_db + m_bc_db - m_ac_db + fspl_db)
    g_c = 0.5 * (m_ac_db + m_bc_db - m_ab_db + fspl_db)
    return {
        "gain_a_dbi": round(g_a, 4),
        "gain_b_dbi": round(g_b, 4),
        "gain_c_dbi": round(g_c, 4),
        "fspl_db": round(fspl_db, 4),
    }


# ---------------------------------------------------------------------------
# #46 — uniform linear array factor + steering + grating-lobe check
# ---------------------------------------------------------------------------


def array_factor_uniform(
    n_elements: int,
    spacing_lambda: float,
    steer_deg: float = 0.0,
    taper: str = "uniform",
    n_points: int = 721,
) -> Dict[str, object]:
    """Array factor of a uniform linear array (electronic beam steering).

    theta is measured from broadside (0 deg = broadside, +/-90 deg = endfire).
    Progressive phase beta = -k d sin(steer). Reports the normalized |AF| in dB
    vs angle, the realized main-beam angle, HPBW, peak sidelobe level, and a
    grating-lobe flag (spacing-lambda >= 1/(1+|sin(steer)|) => grating lobe).

    taper: "uniform" or "hamming".
    """
    n = int(n_elements)
    d = float(spacing_lambda)
    k = 2 * np.pi  # in units of lambda (k*d uses d in wavelengths)
    theta = np.linspace(-90.0, 90.0, int(n_points))
    th = np.deg2rad(theta)
    steer = np.deg2rad(steer_deg)
    beta = -k * d * np.sin(steer)
    if taper == "hamming":
        a = np.hamming(n)
    else:
        a = np.ones(n)
    a = a / np.sum(a)

    nidx = np.arange(n)
    psi = k * d * np.sin(th)[:, None] + beta  # (n_points, 1) broadcast
    af = (a[None, :] * np.exp(1j * nidx[None, :] * psi)).sum(axis=1)
    mag = np.abs(af)
    mag_db = 20 * np.log10(np.maximum(mag / np.max(mag), 1e-6))

    peak_idx = int(np.argmax(mag))
    main_angle = float(theta[peak_idx])

    # HPBW: -3 dB about the main beam.
    half = mag_db - mag_db[peak_idx]  # 0 at peak
    left = peak_idx
    while left > 0 and half[left] > -3.0:
        left -= 1
    right = peak_idx
    while right < len(half) - 1 and half[right] > -3.0:
        right += 1
    hpbw = float(theta[right] - theta[left])

    # Peak sidelobe: highest local max outside the main lobe span.
    sll = -999.0
    for i in range(1, len(mag_db) - 1):
        if not (left <= i <= right) and mag_db[i] > mag_db[i - 1] and mag_db[i] > mag_db[i + 1]:
            sll = max(sll, float(mag_db[i]))

    grating = d >= 1.0 / (1.0 + abs(np.sin(steer)) + 1e-12)
    return {
        "theta_deg": theta.tolist(),
        "af_db": mag_db.tolist(),
        "main_beam_deg": main_angle,
        "hpbw_deg": hpbw,
        "peak_sll_db": None if sll <= -998 else round(sll, 2),
        "grating_lobe_risk": bool(grating),
        "n_elements": n,
        "spacing_lambda": d,
        "steer_deg": float(steer_deg),
    }


# ---------------------------------------------------------------------------
# #47 — planar near-field -> far-field (2D FFT)
# ---------------------------------------------------------------------------


def planar_nf2ff(
    near_field: np.ndarray, dx_m: float, dy_m: float, freq_mhz: float, pad: int = 4
) -> Dict[str, object]:
    """Planar near-field to far-field transform via the 2D FFT (spectral method).

    The (complex) tangential aperture field sampled on a regular planar grid is
    transformed to the far-field angular spectrum E(kx, ky); the far-field
    magnitude pattern is |FFT(E_aperture)| over the visible region kx^2+ky^2<=k^2.

    Args:
        near_field: 2D complex array of the aperture field.
        dx_m, dy_m: sample spacing (m).
        freq_mhz: frequency (MHz).
        pad: zero-padding factor for angular resolution.

    Returns:
        Dict: ff_db (2D normalized far-field magnitude in dB over the FFT grid),
        boresight_index, peak_db (0 after normalization), shape, k_m. The peak of
        a uniform aperture is at boresight (array center).
    """
    e = np.asarray(near_field, dtype=complex)
    if e.ndim != 2:
        raise ValueError("near_field must be a 2D array")
    ny, nx = e.shape
    nfx, nfy = nx * int(pad), ny * int(pad)
    spectrum = np.fft.fftshift(np.fft.fft2(e, s=(nfy, nfx)))
    mag = np.abs(spectrum)
    mag_db = 20 * np.log10(np.maximum(mag / np.max(mag), 1e-9))
    peak_idx = np.unravel_index(int(np.argmax(mag)), mag.shape)
    freq_hz = float(freq_mhz) * 1e6
    k = 2 * np.pi * freq_hz / C_M_PER_S
    return {
        "ff_db": mag_db.tolist(),
        "shape": list(mag_db.shape),
        "boresight_index": [int(peak_idx[0]), int(peak_idx[1])],
        "peak_db": float(mag_db[peak_idx]),
        "k_per_m": k,
    }


# ---------------------------------------------------------------------------
# #43 — S-parameter time-gating + reference-plane de-embedding
# ---------------------------------------------------------------------------


def time_gate_sparam(
    freq_hz: np.ndarray, s_complex: np.ndarray, t_start_s: float, t_stop_s: float
) -> Dict[str, object]:
    """Time-gate an S-parameter sweep: IFFT to time, window [t_start, t_stop], FFT back.

    Removes out-of-gate responses (e.g. chamber reflections) from a frequency
    sweep. Returns the gated S vs frequency and the time-domain response.
    """
    f = np.asarray(freq_hz, dtype=float)
    s = np.asarray(s_complex, dtype=complex)
    if f.size != s.size or f.size < 2:
        raise ValueError("freq_hz and s_complex must be equal length >= 2")
    df = float(np.mean(np.diff(f)))
    n = f.size
    # Time axis for the IFFT of a one-sided sweep.
    t = np.fft.fftfreq(n, d=df)
    td = np.fft.ifft(s)
    gate = (t >= t_start_s) & (t <= t_stop_s)
    td_gated = td * gate
    s_gated = np.fft.fft(td_gated)
    return {
        "s_gated_complex": [complex(v) for v in s_gated],
        "time_s": t.tolist(),
        "time_response_mag": np.abs(td).tolist(),
        "n_gated_bins": int(np.count_nonzero(gate)),
    }


def port_extension_deembed(
    freq_hz: np.ndarray, s_complex: np.ndarray, delay_s: float, two_way: bool = True
) -> np.ndarray:
    """Reference-plane de-embedding via port extension (electrical-length removal).

    Rotates out the linear phase of a cable / fixture delay: multiply by
    exp(+j 2π f * delay) (one-way) or exp(+j 2π f * 2*delay) for a reflection
    (two-way, the default for S11). Returns the de-embedded complex S.
    """
    f = np.asarray(freq_hz, dtype=float)
    s = np.asarray(s_complex, dtype=complex)
    factor = 2.0 if two_way else 1.0
    return s * np.exp(1j * 2 * np.pi * f * factor * float(delay_s))


# ---------------------------------------------------------------------------
# #42 — CTIA Total Isotropic Sensitivity
# ---------------------------------------------------------------------------


def total_isotropic_sensitivity(
    eis_dbm_2d: np.ndarray, theta_angles_rad: np.ndarray, inc_theta: float, inc_phi: float
) -> float:
    """CTIA Total Isotropic Sensitivity (TIS) from EIS over the sphere.

    TIS is the sphere harmonic mean of the Effective Isotropic Sensitivity:
        1/TIS = (1/4π) ∫∫ (1/EIS(θ,φ)) sin(θ) dθ dφ
    (linear power). For a constant EIS over the sphere, TIS == that EIS.

    Args:
        eis_dbm_2d: 2D EIS in dBm (theta x phi). EIS is a sensitivity (negative
            dBm); lower (more negative) is better.
        theta_angles_rad, inc_theta, inc_phi: as in calculate_trp.

    Returns:
        TIS in dBm.
    """
    eis_mw = 10 ** (np.asarray(eis_dbm_2d, dtype=float) / 10.0)
    w = np.sin(np.asarray(theta_angles_rad, dtype=float))[:, None]
    dtheta = np.deg2rad(inc_theta)
    dphi = np.deg2rad(inc_phi)
    inv_mean = np.sum((1.0 / eis_mw) * w) * dtheta * dphi / (4 * np.pi)
    tis_mw = 1.0 / inv_mean
    return float(10.0 * np.log10(max(tis_mw, 1e-30)))
