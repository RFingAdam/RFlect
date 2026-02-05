# RFlect - Antenna Plot Tool - Group Delay & Fidelity Analyzer

import os
import re
import time
from functools import wraps
from io import StringIO
from collections import defaultdict
from functools import wraps
from functools import lru_cache

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import correlate, fftconvolve
from tqdm import tqdm
import tkinter as tk
from tkinter import filedialog, messagebox, simpledialog

# ——— CONSTANTS ——————————————————————————————————
STATIC_COLS = [
    'Theta [deg.]','Phi [deg.]','Abs(Dir.)[]',
    'Abs(Theta)[]','Phase(Theta)[deg.]','Abs(Phi)[]',
    'Phase(Phi)[deg.]','Ax.Ratio[]'
]
C = 3e8  # speed of light (m/s)

# ——— TIMING DECORATOR ——————————————————————————————
def timed(name):
    def decorator(fn):
        @wraps(fn)
        def wrapper(*args, **kwargs):
            t0 = time.time()
            res = fn(*args, **kwargs)
            print(f"[⏱ {name}] {time.time()-t0:.2f}s")
            return res
        return wrapper
    return decorator

# ——— PARSING HELPERS —————————————————————————————
"""
Read a far-field data file and extract the frequency and data.
The data is expected to be in a specific format with 8 columns.

Parameters:
    path (str): Path to the far-field data file.

Returns:
    tuple: A tuple containing the frequency (float) and a DataFrame with the data.
        The DataFrame has columns for theta, phi, gain, and phase information.
"""
def find_data_block(lines):
    for i, L in enumerate(lines):
        if L.strip().startswith('---'):
            return lines[i+1:]
    raise RuntimeError("No '---' delimiter found")

@lru_cache(maxsize=None)
def extract_frequency(path):
    m = re.search(r'\(f=([\d.]+)\)', os.path.basename(path))
    if not m:
        raise ValueError(f"No frequency tag in {path}")
    return float(m.group(1)) * 1e9

# add lru_cache to your imports
from functools import wraps, lru_cache

@lru_cache(maxsize=256) # Maximum number of cached files to 256 to limit memory usage
def read_farfield_file(path: str) -> tuple[float, pd.DataFrame]:
    """
    Read a CST far‑field .txt file, parse out the data block,
    return (frequency_Hz, DataFrame).
    Caches results so each file is only parsed once.
    """
    with open(path, 'r') as f:
        lines = f.readlines()
    data = find_data_block(lines)
    df = pd.read_csv(
        StringIO(''.join(data)),
        delim_whitespace=True,
        header=None,
        engine='python'
    )
    if df.shape[1] != 8:
        raise ValueError(f"{os.path.basename(path)}: expected 8 cols, got {df.shape[1]}")
    df.columns = STATIC_COLS
    return extract_frequency(path), df

# ——— CORE: build phase‑vs‑freq matrix per plane ——————————————————
"""
Compute the phase data for a given plane (horizontal or vertical) from the far-field files.
Parameters:
    files (list): List of file paths to the far-field data files.
    phase_col (str): Column name for the phase data.
    fixed_col (str): Column name for the fixed angle data.
    fixed_val (float): Fixed angle value.
    group_col (str): Column name for the grouping angle data.
Returns:
    tuple: A tuple containing the angles, frequencies, and phase data matrix.
"""
@timed("Compute plane data")
def compute_plane_data(files, phase_col, fixed_col, fixed_val, group_col):
    freq_df = []
    for p in tqdm(sorted(files), desc="Reading far‑field files"):
        try:
            f, df = read_farfield_file(p)
        except Exception as e:
            print(f"[WARN] skipping {p}: {e}")
            continue
        freq_df.append((f, df))

    freq_df.sort(key=lambda x: x[0])
    freqs = np.array([f for f, _ in freq_df])
    angles = np.sort(freq_df[0][1][group_col].unique())
    P = np.zeros((len(angles), len(freqs)))

    for j, (f, df) in enumerate(freq_df):
        grp = df.groupby(group_col)
        for i, ang in enumerate(angles):
            sub = grp.get_group(ang)
            idx = (sub[fixed_col] - fixed_val).abs().idxmin()
            P[i, j] = sub.loc[idx, phase_col]
    return angles, freqs, P

def phase_to_tau(P_deg, freqs):
    P_rad = np.unwrap(np.deg2rad(P_deg), axis=1)
    ω = 2 * np.pi * freqs
    dphi_dω = np.gradient(P_rad, ω, axis=1)
    return -dphi_dω * 1e9  # ns

# ——— φ‑based group‑delay calculation ——————————————————
"""
Compute the far-field data for both horizontal and vertical polarizations.

Parameters:
    files (list): List of file paths to the far-field data files.
    dominant_polarization (str): The dominant polarization ('Theta' or 'Phi').
    theta_target (float): The target theta angle for the far-field data.
    gain_threshold (float): The gain threshold for filtering.

Returns:
    phis (list): List of phi angles.
    delays (list): List of group delays corresponding to the phi angles.
    all_delays (list): List of all group delays.
"""
@timed("Compute group delay")
def compute_group_delay(files,
                        dominant_polarization: str = 'Phi',
                        theta_target: float = 90.0,
                        gain_threshold: float = 0.0):
    """
    Compute azimuthal group delay for the chosen polarization:
      - Theta, Phi, or Total (vector sum of Eθ+Eφ).
    """
    # prepare containers
    phi_dict = defaultdict(list)
    gain_col = 'Abs(Dir.)[]'

    # loop over frequency snapshots
    for path in sorted(files):
        try:
            freq, df = read_farfield_file(path)
        except Exception as e:
            print(f"[WARN] Skipping {path}: {e}")
            continue

        # for each azimuth
        for phi, sub in df.groupby('Phi [deg.]'):
            # find the row closest to θ=theta_target
            idx = (sub['Theta [deg.]'] - theta_target).abs().idxmin()
            row = sub.loc[idx]

            # skip if below gain threshold
            gain_val = float(row[gain_col])  # type: ignore
            if gain_val < gain_threshold:
                continue

            # pick the phase value
            if dominant_polarization in ('Theta', 'Phi'):
                phase_val = row[f'Phase({dominant_polarization})[deg.]']
            else:  # Total → vector‐sum of the two fields
                Eθ = row['Abs(Theta)[]'] * np.exp(1j * np.deg2rad(row['Phase(Theta)[deg.]']))
                Eφ = row['Abs(Phi)[]']   * np.exp(1j * np.deg2rad(row['Phase(Phi)[deg.]']))
                phase_val = np.rad2deg(np.angle(Eθ + Eφ))

            # store for later differentiation
            phi_dict[phi].append((freq, phase_val))

    # now compute τ for each phi
    phis, delays, all_delays = [], [], []
    for phi in sorted(phi_dict):
        pts = sorted(phi_dict[phi], key=lambda x: x[0])
        if len(pts) < 2:
            continue
        freqs_arr = np.array([p[0] for p in pts])
        phases   = np.unwrap(np.deg2rad([p[1] for p in pts]))
        dphi_dω  = np.gradient(phases, 2 * np.pi * freqs_arr)
        tau_ns   = -np.mean(dphi_dω) * 1e9

        phis.append(phi)
        delays.append(tau_ns)
        all_delays.append(tau_ns)

    return phis, delays, all_delays


# ——— PLOTTING FUNCTIONS —————————————————————————————————
def plot_group_delay_vs_phi(phis, delays):
    plt.figure(figsize=(10,4))
    plt.plot(phis, delays, 'o-', lw=1.5)
    plt.axhline(np.mean(delays), ls='--',
                label=f"Mean = {np.mean(delays):.2f} ns")
    plt.title(f"Group Delay vs Azimuth (θ≈90°, Δτ={max(delays)-min(delays):.2f} ns)")
    plt.xlabel("φ (deg)"); plt.ylabel("τ (ns)")
    plt.grid(True); plt.legend(); plt.tight_layout(); plt.show()

def plot_cdf(delays):
    sd = np.sort(delays)
    cdf = np.linspace(0,1,len(sd))
    plt.figure(figsize=(6,4))
    plt.plot(sd*1e3, cdf, lw=2)
    plt.title("CDF of Absolute Group Delay (θ≈90°)")
    plt.xlabel("τ (ps)"); plt.ylabel("CDF")
    plt.grid(True); plt.tight_layout(); plt.show()

def plot_group_delay_vs_theta(thetas, delays):
    plt.figure(figsize=(10,4))
    plt.plot(thetas, delays, 's-', lw=1.5)
    plt.axhline(np.mean(delays), ls='--',
                label=f"Mean = {np.mean(delays):.2f} ns")
    plt.title(f"Group Delay vs Elevation (φ=0°, Δτ={max(delays)-min(delays):.2f} ns)")
    plt.xlabel("θ (deg)"); plt.ylabel("τ (ns)")
    plt.grid(True); plt.legend(); plt.tight_layout(); plt.show()

def plot_group_delay_vs_frequency_annotated(freqs, τ0):
    i_min = np.argmin(τ0); i_max = np.argmax(τ0)
    f_low, τ_low = freqs[i_min]/1e9, τ0[i_min]
    f_hi, τ_hi = freqs[i_max]/1e9, τ0[i_max]
    Δτ_ps = (τ_hi - τ_low)*1e3

    plt.figure(figsize=(8,4))
    plt.plot(freqs/1e9, τ0, 'o-', label="Boresight τ")
    plt.hlines([τ_low, τ_hi], freqs[0]/1e9, freqs[-1]/1e9,
               colors='gray', linestyles='--')
    plt.annotate("", xy=(freqs[-1]/1e9, τ_hi),
                 xytext=(freqs[-1]/1e9, τ_low),
                 arrowprops=dict(arrowstyle='<->', color='gray'))
    plt.text(freqs[-1]/1e9+0.005, (τ_low+τ_hi)/2, f"{Δτ_ps:.1f} ps",
             va='center')
    plt.plot(f_low, τ_low, 'rv')
    plt.plot(f_hi, τ_hi, 'gv')
    plt.title("Group Delay vs Frequency at Boresight (θ=90°, φ=0°)")
    plt.xlabel("Frequency (GHz)"); plt.ylabel("τ (ns)")
    plt.grid(True); plt.tight_layout(); plt.show()

def plot_envelope_vs_frequency(freqs, τ, plane_name):
    minτ, maxτ = τ.min(axis=0), τ.max(axis=0)
    meanτ = τ.mean(axis=0)
    plt.figure(figsize=(8,4))
    plt.fill_between(freqs/1e9, minτ, maxτ, alpha=0.3,
                     label=f"{plane_name} envelope")
    plt.plot(freqs/1e9, meanτ, '-', label=f"{plane_name} mean τ")
    plt.title(f"{plane_name} Delay Envelope vs Frequency")
    plt.xlabel("Frequency (GHz)"); plt.ylabel("τ (ns)")
    plt.grid(True); plt.legend(); plt.tight_layout(); plt.show()

def plot_variation_vs_frequency(freqs, τ_phi, τ_theta):
    var_phi = τ_phi.ptp(axis=0)*1e3
    var_theta = τ_theta.ptp(axis=0)*1e3
    plt.figure(figsize=(8,4))
    plt.plot(freqs/1e9, var_phi, 'o-', label="φ‑plane Δτ")
    plt.plot(freqs/1e9, var_theta, 's-', label="θ‑plane Δτ")
    plt.title("Peak‑to‑Peak Δτ vs Frequency")
    plt.xlabel("Frequency (GHz)"); plt.ylabel("Δτ (ps)")
    plt.grid(True); plt.legend(); plt.tight_layout(); plt.show()

def plot_max_error_vs_frequency(freqs, τ_phi, τ_theta):
    err_phi = τ_phi.ptp(axis=0)*1e-9*C
    err_theta = τ_theta.ptp(axis=0)*1e-9*C
    plt.figure(figsize=(8,4))
    plt.plot(freqs/1e9, err_phi, 'o-', label="φ‑plane error (m)")
    plt.plot(freqs/1e9, err_theta, 's-', label="θ‑plane error (m)")
    plt.title("Max Distance Error vs Frequency")
    plt.xlabel("Frequency (GHz)"); plt.ylabel("Error (m)")
    plt.grid(True); plt.legend(); plt.tight_layout(); plt.show()

def plot_cdf_vs_threshold_spread(angles_phi, angles_theta, τ_phi, τ_theta):
    idx_phi0 = np.argmin(np.abs(angles_phi - 0.0))
    Δτ_phi0 = τ_phi[idx_phi0].ptp()*1e3
    var_theta = τ_theta.ptp(axis=1)*1e3
    sd = np.sort(var_theta)
    cdf = np.linspace(0,1,len(sd))
    cdf_val = np.interp(Δτ_phi0, sd, cdf)

    plt.figure(figsize=(6,4))
    plt.plot(sd, cdf, label='θ‑plane Δτ CDF')
    plt.axvline(Δτ_phi0, color='C1', ls='--',
                label=f"φ‑plane Δτ = {Δτ_phi0:.1f} ps")
    plt.title("CDF of Δτ Spread across θ‑plane")
    plt.xlabel("Δτ (ps)"); plt.ylabel("CDF")
    plt.grid(True); plt.legend(); plt.tight_layout(); plt.show()

def plot_group_delay_vs_freq_for_phis(freqs, τ_phi, angles_phi):
    sel = [0,45,90,135,180,225,270,315]
    plt.figure(figsize=(10,6))
    for angle in sel:
        idx = np.argmin(np.abs(angles_phi - angle))
        plt.plot(freqs/1e9, τ_phi[idx], label=f"φ={angles_phi[idx]:.0f}°")
    plt.title("Group Delay vs Frequency for Various φ (θ=90°)")
    plt.xlabel("Frequency (GHz)"); plt.ylabel("τ (ns)")
    plt.grid(True); plt.legend(); plt.tight_layout(); plt.show()

def plot_peak_to_peak_over_phi(freqs, τ_phi):
    var_phi = τ_phi.ptp(axis=0)*1e3
    plt.figure(figsize=(8,4))
    plt.plot(freqs/1e9, var_phi, '-o')
    plt.title("φ‑plane Peak‑to‑Peak Δτ vs Frequency")
    plt.xlabel("Frequency (GHz)"); plt.ylabel("Δτ (ps)")
    plt.grid(True); plt.tight_layout(); plt.show()

def plot_max_error_over_phi(freqs, τ_phi):
    err_cm = τ_phi.ptp(axis=0)*1e-9*C*100
    plt.figure(figsize=(8,4))
    plt.plot(freqs/1e9, err_cm, '-o')
    plt.title("φ‑plane Max Distance Error vs Frequency")
    plt.xlabel("Frequency (GHz)"); plt.ylabel("Error (cm)")
    plt.grid(True); plt.tight_layout(); plt.show()

# 2D equirectangular map of τ(θ,φ)      
@timed("Compute group delay map")
def compute_group_delay_map(files, dom_pol='Phi'):
    """
    Builds a 2D map τ(θ,φ) by differentiating the unwrapped phase
    at each (θ,φ) point across frequency.

    Returns:
      thetas : 1D array of θ values in degrees (0…180)
      phis   : 1D array of φ values in degrees (0…360)
      tau    : 2D array shape (len(thetas), len(phis)) in ns
    """
    # 1) load all data
    freq_df = []
    for path in tqdm(sorted(files), desc="Reading far-field files"):
        f, df = read_farfield_file(path)
        freq_df.append((f, df))
    freq_df.sort(key=lambda x: x[0])
    freqs = np.array([f for f,_ in freq_df])
    ω     = 2*np.pi*freqs

    # 2) collect all θ,φ grid points
    all_thetas = np.sort(freq_df[0][1]['Theta [deg.]'].unique())
    all_phis   = np.sort(freq_df[0][1]['Phi [deg.]'].unique())
    Nθ, Nφ = len(all_thetas), len(all_phis)

    # 3) build phase array: shape (θ,φ,f)
    P_deg = np.zeros((Nθ, Nφ, len(freqs)))
    for k, (_, df) in enumerate(freq_df):
        # pivot into 2D
        if dom_pol in ('Theta','Phi'):
            valcol = f'Phase({dom_pol})[deg.]'
            grid = df.pivot(index='Theta [deg.]', columns='Phi [deg.]', values=valcol)
        else:
            # Total: compute combined phase
            df['PhaseTot'] = np.rad2deg(np.angle(
                df['Abs(Theta)[]'] * np.exp(1j*np.deg2rad(df['Phase(Theta)[deg.]'])) +
                df['Abs(Phi)[]']   * np.exp(1j*np.deg2rad(df['Phase(Phi)[deg.]']))
            ))
            grid = df.pivot(index='Theta [deg.]', columns='Phi [deg.]', values='PhaseTot')

        P_deg[:,:,k] = grid.reindex(
            index=all_thetas, columns=all_phis
        ).values


    # 4) unwrap & differentiate along freq axis
    tau = np.zeros((Nθ, Nφ))
    P_rad = np.unwrap(np.deg2rad(P_deg), axis=2)
    dphi_dω = np.gradient(P_rad, ω, axis=2)    # dφ/dω [rad⋅s]
    tau = -np.mean(dphi_dω, axis=2) * 1e9      # ns

    return all_thetas, all_phis, tau


def plot_group_delay_map(thetas, phis, tau):
    """
    Equirectangular colormap of τ(θ,φ).
    θ runs vertically from 0° (bottom) to 180° (top),
    φ runs horizontally from 0° (left) to 360° (right).
    """
    plt.figure(figsize=(8,4))
    # note: imshow wants [row, col] = [θ,φ]
    im = plt.imshow(tau,
                    extent=(phis[0], phis[-1], thetas[0], thetas[-1]),
                    origin='lower',
                    aspect='auto',
                    cmap='viridis')
    plt.xlabel('φ (deg)')
    plt.ylabel('θ (deg)')
    Δτ = tau.max() - tau.min()
    plt.title(f"Group Delay Map τ(θ,φ) [ns], Δτ={Δτ:.2f} ns")
    plt.colorbar(im, label='τ (ns)')
    plt.tight_layout()
    plt.show()

# ——— NEW FIDELITY SECTION (auto‑detect band) ——————————————————
"""
Compute the fidelity of the antenna at boresight.
Parameters:
    files (list): List of file paths to the far-field data files.
    dom_pol (str): Dominant polarization ('Theta' or 'Phi').
    theta0 (float): Target theta angle for the boresight.
    phi0 (float): Target phi angle for the boresight.
    band (tuple): Frequency band (min, max) for the fidelity calculation.
    Nfft (int): Number of FFT points.
Returns:
    tuple: A tuple containing the fidelity (float), time array (np.ndarray),
    input pulse (np.ndarray), and output pulse (np.ndarray).
"""
# ——— PLOTTING —————————————————————————————————
def _split_clusters(idx: np.ndarray) -> list[np.ndarray]:
    """Split a sorted index array into contiguous runs."""
    if idx.size == 0:
        return []
    cuts = np.where(np.diff(idx) > 1)[0] + 1
    return np.split(idx, cuts)

def plot_io_pulse(tt: np.ndarray,
                  inp: np.ndarray,
                  out_raw: np.ndarray,
                  delay_ns: float) -> None:
    """
    Plot the input pulse (at t≈0) and the antenna output (at t≈delay_ns),
    cropping away everything else.
    """
    # time in ns
    t_ns = tt * 1e9

    # — pad the TX pulse so it’s the same length as the RX —
    vin = np.real(inp)
    if vin.shape[0] < out_raw.shape[0]:
        vin_full      = np.zeros_like(out_raw)
        vin_full[:vin.shape[0]] = vin
        vin           = vin_full
    else:
        vin = vin[:out_raw.shape[0]]

    vout = np.real(out_raw)

    # normalize
    vin  /= np.max(np.abs(vin))
    vout /= np.max(np.abs(vout))

    # threshold clusters >5%
    thr      = 0.05
    idx_in   = np.where(np.abs(vin)  > thr)[0]
    idx_out  = np.where(np.abs(vout) > thr)[0]

    def split(idx):
        if idx.size == 0: return []
        cuts = np.where(np.diff(idx) > 1)[0] + 1
        return np.split(idx, cuts)

    in_clust  = split(idx_in)
    out_clust = split(idx_out)

    # pick the first TX cluster
    c_in = in_clust[0] if in_clust else np.array([0])

    # pick the RX cluster nearest to the expected delay
    dt    = t_ns[1] - t_ns[0]
    exp_i = int(np.round(delay_ns/dt))
    if out_clust:
        c_out = min(out_clust, key=lambda c: abs(np.mean(c) - exp_i))
    else:
        c_out = c_in

    # pad 10% on each side
    pad = 0.1*(t_ns[c_out[-1]] - t_ns[c_in[0]])
    x0  = max(0,            t_ns[c_in[0]] - pad)
    x1  = min(t_ns[-1], t_ns[c_out[-1]] + pad + delay_ns)

    # shift the RX time‐axis by delay_ns
    t_out = t_ns + delay_ns

    # plot
    plt.figure(figsize=(6,4))
    plt.plot(t_ns,    vin,  'b-',  label='Input')
    plt.plot(t_out,   vout, 'r--', label=f'Output (delay {delay_ns:.2f} ns)')
    plt.xlim(x0, x1)
    plt.xlabel('Time (ns)')
    plt.ylabel('Normalized amplitude')
    plt.title('Input vs Output Pulse @ Boresight')
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()

def plot_fidelity_vs_phi(phis: np.ndarray,
                         fids: np.ndarray,
                         band: tuple[float,float]) -> None:
    """Plot fidelity vs φ, with y‑axis fixed to 50–100%."""
    fmin, fmax = band
    plt.figure(figsize=(6,4))
    plt.plot(phis, fids*100, 'o-', lw=1.5)
    plt.ylim(50,100)
    plt.xlabel('φ (deg)')
    plt.ylabel('Fidelity (%)')
    plt.title(f'Fidelity vs φ ({fmin/1e9:.1f}–{fmax/1e9:.1f} GHz)')
    plt.grid(True)
    plt.tight_layout()
    plt.show()

def plot_fidelity_vs_theta(thetas: np.ndarray,
                           fids: np.ndarray,
                           band: tuple[float,float]) -> None:
    """Same as φ but vs θ."""
    fmin, fmax = band
    plt.figure(figsize=(6,4))
    plt.plot(thetas, fids*100, 's-', lw=1.5)
    plt.ylim(50,100)
    plt.xlabel('θ (deg)')
    plt.ylabel('Fidelity (%)')
    plt.title(f'Fidelity vs θ ({fmin/1e9:.1f}–{fmax/1e9:.1f} GHz)')
    plt.grid(True)
    plt.tight_layout()
    plt.show()

# ——— NEW FIDELITY SECTION (auto‑detect band) ——————————————————
@timed("Compute fidelity")
def compute_fidelity_at_boresight(
    files: list[str],
    dom_pol: str = 'Phi',
    theta0: float = 90.0,
    phi0: float = 0.0,
    band: tuple[float,float] | None = None,
    Nfft: int = 16384
) -> tuple[float, np.ndarray, np.ndarray, np.ndarray, float]:
    """
    Returns:
    Fmax    : fidelity scalar
    t_lin   : time vector for R_lin (s)
    pulse   : transmitted test pulse
    R_lin   : antenna output (linear convolution)
    delay_s : estimated one-way group delay (s)
    """
    # — determine the band —
    all_fs = sorted(extract_frequency(p) for p in files)
    if band is None:
        band = (all_fs[0], all_fs[-1])

    # — gather boresight amp & phase —
    amp_phase = []
    for p in sorted(files):
        f, df = read_farfield_file(p)
        if not (band[0] <= f <= band[1]):
            continue
        grp  = df.groupby('Phi [deg.]')
        φval = min(grp.groups, key=lambda x: abs(float(x)-phi0))  # type: ignore
        sub  = grp.get_group(φval)
        idx  = (sub['Theta [deg.]'] - theta0).abs().idxmin()
        amp  = sub.loc[idx, 'Abs(Dir.)[]']
        pdg  = np.deg2rad(float(sub.loc[idx, f'Phase({dom_pol})[deg.]']))  # type: ignore
        amp_phase.append((f, amp, pdg))

    if len(amp_phase) < 2:
        raise RuntimeError("Not enough points for fidelity!")

    amp_phase.sort(key=lambda x: x[0])
    Fs, A_vals, phases = map(np.array, zip(*amp_phase))

    # — build the UWB Gaussian pulse —
    fs     = 4 * band[1]
    dt     = 1.0 / fs
    t_p    = np.arange(Nfft) * dt
    f0     = np.mean(band)
    bw     = band[1] - band[0]
    σ      = 1.0 / bw
    center = 5 * σ
    carrier= np.cos(2*np.pi * f0 * (t_p - center))
    gauss  = np.exp(-0.5 * ((t_p - center)/σ)**2)
    pulse  = carrier * gauss

    # — build H(f) with Hermitian symmetry —
    win    = np.blackman(Nfft)
    Pp     = np.fft.fft(pulse * win)
    freq_v = np.fft.fftfreq(Nfft, dt)
    A_i   = np.interp(freq_v, Fs, A_vals,   left=0.0, right=0.0)

    phi_i = np.interp(freq_v, Fs, phases,   left=0.0, right=0.0)

    Hf     = A_i * np.exp(1j*phi_i)
    half   = Nfft//2
    Hf[half+1:] = np.conj(Hf[1:half][::-1])

    # — get antenna impulse response h(n) —
    h = np.fft.ifft(Hf)
    peak = np.argmax(np.abs(h))
    h = np.roll(h, -peak)      # move main lobe to n=0

   #  — window the impulse response so only the main lobe remains —
    thr_h   = 0.02 * np.max(np.abs(h))
    idx_h   = np.where(np.abs(h) > thr_h)[0]

    if idx_h.size:
        # split into contiguous runs and keep only the FIRST one
        cuts     = np.where(np.diff(idx_h) > 1)[0] + 1
        clusters = np.split(idx_h, cuts)
        main     = clusters[0]
        h_clean  = np.zeros_like(h)
        h_clean[main] = h[main]
        h = h_clean
    # else: if detection fails, leave h alone
    else:
        print("[WARN] No main lobe detected in h(n)")
  
    # — **linear** convolution to avoid wrap-around —
    R_lin = fftconvolve(pulse, h, mode='full')
    t_lin = np.arange(len(R_lin)) * dt

    # — cross-correlate & normalize —
    corr = correlate(R_lin, pulse, mode='full')
    norm = np.sqrt(np.sum(np.abs(pulse)**2) * np.sum(np.abs(R_lin)**2))
    corr /= norm

    # — choose the first positive peak for one-way delay —
    lags = np.arange(-len(pulse)+1, len(pulse)) * dt
    pos  = np.where(lags > 0)[0]
    best = pos[np.argmax(np.abs(corr[pos]))]
    delay_s = lags[best]

    Fmax = float(np.max(np.abs(corr)))
    return Fmax, t_lin, pulse, R_lin, float(delay_s)

def plot_fidelity(tt: np.ndarray, inp: np.ndarray, out: np.ndarray, F: float) -> None:
    center = len(tt)//2
    w = int(0.2 * len(tt))
    rng = slice(center - w, center + w)
    plt.figure(figsize=(8,4))
    plt.plot(tt[rng]*1e9, inp[rng]/np.max(np.abs(inp)),  label='Input')
    plt.plot(tt[rng]*1e9, out[rng]/np.max(np.abs(out)), '--', label='Output')
    plt.xlabel('Time (ns)')
    plt.ylabel('Normalized Amplitude')
    plt.title(f'Impulse Fidelity @ Boresight = {F*100:.2f}%')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

def plot_correlation(corr: np.ndarray, dt: float) -> None:
    """
    Plot the full cross-correlation sequence vs time lag.
    corr : output of scipy.signal.correlate()
    dt   : sample period (s) used to build your pulse
    """
    # full-length, so lags run from -(N-1) to +(N-1)
    N = len(corr)
    lags = np.arange(- (N//2), N//2 + (N%2)) * dt
    # if corr isn't centered at zero lag, roll it
    corr_centered = np.roll(corr, -np.argmax(np.abs(corr)) + N//2)
    plt.figure(figsize=(8,4))
    plt.plot(lags*1e9, corr_centered, lw=1.5)
    plt.xlabel("Time Lag (ns)")
    plt.ylabel("Cross-correlation amplitude")
    plt.title("Full Cross‑Correlation (Pulse Dispersion)")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

# ——— ECC FUNCTIONS ———————————————————————————————
def compute_ecc_from_farfield(path1: str,
                              path2: str,
                              freq_Hz: float,
                              tol_Hz: float = 1e6) -> float:
    """
    Compute full‐field ECC by integrating both Eθ and Eφ over the sphere:
      ECC = |∬ [E1θ·E2θ* + E1φ·E2φ*] sinθ dθ dφ|²
            / [ (∬ (|E1θ|²+|E1φ|²) sinθ dθ dφ)
                · (∬ (|E2θ|²+|E2φ|²) sinθ dθ dφ) ]
    """
    def load_components(path):
        f, df = read_farfield_file(path)
        if abs(f - freq_Hz) > tol_Hz:
            raise ValueError(f"{os.path.basename(path)} not at {freq_Hz/1e9:.3f}GHz")
        θ = np.deg2rad(df['Theta [deg.]'].values)  # type: ignore
        φ = np.deg2rad(df['Phi [deg.]'].values)  # type: ignore
        Eθ = df['Abs(Theta)[]'].values * np.exp(1j*np.deg2rad(df['Phase(Theta)[deg.]'].values))  # type: ignore
        Eφ = df['Abs(Phi)[]'].values   * np.exp(1j*np.deg2rad(df['Phase(Phi)[deg.]'].values))  # type: ignore
        return θ, φ, Eθ, Eφ

    θ1, φ1, E1θ, E1φ = load_components(path1)
    θ2, φ2, E2θ, E2φ = load_components(path2)

    # verify same sampling
    if not (np.allclose(θ1, θ2) and np.allclose(φ1, φ2)):
        raise RuntimeError("Angle grids do not match")

    thetas = np.unique(θ1)
    phis   = np.unique(φ1)
    dθ, dφ = thetas[1] - thetas[0], phis[1] - phis[0]

    w = np.sin(θ1) * dθ * dφ

    # numerator: cross‐term of full vector fields
    inner = E1θ*np.conj(E2θ) + E1φ*np.conj(E2φ)
    num   = abs(np.sum(inner * w))**2

    # denom: total power in each antenna
    P1 = np.sum((abs(E1θ)**2 + abs(E1φ)**2) * w)
    P2 = np.sum((abs(E2θ)**2 + abs(E2φ)**2) * w)
    den = P1 * P2

    return float(num/den)


def plot_ecc_summary(ecc_results: list[tuple[float,float]]):
    """
    Plot ECC vs. frequency.

    Parameters:
      ecc_results : list of (freq_Hz, ecc_scalar) tuples
    """
    freqs = np.array([f for f, ecc in ecc_results]) / 1e9
    eccs   = np.array([ecc for f, ecc in ecc_results])

    plt.figure(figsize=(8,4))
    plt.plot(freqs, eccs, 'o-', lw=1.5)
    plt.ylim(0,1)
    plt.xlabel("Frequency (GHz)")
    plt.ylabel("ECC")
    plt.title("Envelope Correlation Coefficient vs. Frequency")
    plt.grid(True)
    plt.tight_layout()
    plt.show()


# ——— MAIN UI —————————————————————————————————
def main():
    root = tk.Tk()
    root.title("RFlect – Group Delay, Fidelity & ECC Analyzer")

    # — Polarization selection —
    tk.Label(root, text="Select dominant polarization:").pack(pady=(10,0))
    var_pol = tk.StringVar(value='Total')          # default to Total
    for v in ('Theta','Phi','Total'):               # add Total
        tk.Radiobutton(root, text=v, variable=var_pol, value=v)\
            .pack(anchor='w', padx=20)


    # — Analysis mode —
    tk.Label(root, text="Select analysis type:").pack(pady=(10,0))
    var_mode = tk.StringVar(value='GroupDelay')
    for text, val in [
        ("Group Delay & Fidelity", 'GroupDelay'),
        ("Envelope Correlation Coefficient (ECC)", 'ECC')
    ]:
        tk.Radiobutton(root, text=text, variable=var_mode, value=val).pack(anchor='w', padx=20)

    # — ECC frequency sampling (only used if ECC chosen) —
    tk.Label(root, text="ECC frequency sampling:").pack(pady=(10,0))
    var_ecc_freq = tk.StringVar(value='FullBand')
    for text, val in [
        ("Full sweep (all frequencies)", 'FullBand'),
        ("Low / Mid / High only",            'Sample3Freq'),
        ("Single frequency",                 'SingleFreq')
    ]:
        tk.Radiobutton(root, text=text, variable=var_ecc_freq, value=val).pack(anchor='w', padx=40)

    @timed("Full analysis")
    def proceed():
        mode = var_mode.get()

        if mode == 'GroupDelay':
            # pick the files
            files = filedialog.askopenfilenames(
                title="Select CST far-field .txt files",
                filetypes=[("Text files","*.txt")]
            )
            if not files:
                messagebox.showwarning("Cancelled","No files selected.")
                return
            if len(files) < 2 and var_mode.get() == 'GroupDelay':
                messagebox.showerror("Need ≥2 files","Select two or more exports.")
                return
            root.destroy()

            # — Existing group-delay & fidelity pipeline —
            all_fs = sorted(extract_frequency(p) for p in files)
            full_band = (all_fs[0], all_fs[-1])

            thetas, phis, tau_map = compute_group_delay_map(files, dom_pol=var_pol.get())
            plot_group_delay_map(thetas, phis, tau_map)

            phis, delays, all_delays = compute_group_delay(files, dominant_polarization=var_pol.get())
            angles_phi, freqs, P_phi = compute_plane_data(
                files,
                phase_col = f"Phase({var_pol.get()})[deg.]",
                fixed_col = "Theta [deg.]",
                fixed_val = 90.0,
                group_col = "Phi [deg.]"
            )
            angles_theta, _, P_theta = compute_plane_data(
                files, f"Phase({var_pol.get()})[deg.]",
                "Phi [deg.]", 0.0, "Theta [deg.]"
            )
            
            τ_phi   = phase_to_tau(P_phi, freqs)
            τ_theta = phase_to_tau(P_theta, freqs)
            idx0    = np.argmin(np.abs(angles_phi - 0.0))
            τ0      = τ_phi[idx0]
            delay_ns = np.mean(τ_phi[idx0])       # in ns
            delay_s  = delay_ns * 1e-9            # back to seconds

            plot_group_delay_vs_phi(phis, delays)
            plot_cdf(all_delays)
            plot_group_delay_vs_theta(angles_theta, τ_theta.mean(axis=1))
            plot_group_delay_vs_frequency_annotated(freqs, τ0)
            plot_envelope_vs_frequency(freqs, τ_phi, plane_name="φ‑plane (θ=90°)")
            plot_envelope_vs_frequency(freqs, τ_theta, plane_name="θ‑plane (φ=0°)")
            plot_variation_vs_frequency(freqs, τ_phi, τ_theta)
            plot_max_error_vs_frequency(freqs, τ_phi, τ_theta)
            #plot_cdf_vs_threshold_spread(angles_phi, angles_theta, τ_phi, τ_theta)
            plot_group_delay_vs_freq_for_phis(freqs, τ_phi, angles_phi)
            plot_peak_to_peak_over_phi(freqs, τ_phi)
            plot_max_error_over_phi(freqs, τ_phi)
            
            # — FIDELITY: full‑band I/O pulse @ φ=0° —
            NFFT_FID = 16384  # ← bump for finer time resolution

            F0, t_lin, pulse, R_lin, delay_s = compute_fidelity_at_boresight(
                list(files),
                dom_pol=var_pol.get(),
                theta0=90.0,
                phi0=0.0,
                band=None,
                Nfft=NFFT_FID
            )

            messagebox.showinfo(
                "Fidelity",
                f"Impulse fidelity @ φ=0° = {F0*100:.2f}%\n"
                f"Using phase-derived delay = {delay_ns:.2f} ns"
            )
            plot_io_pulse(t_lin, pulse, R_lin, delay_ns)

            # — FIDELITY vs φ at 30° steps over full band -----------------------
            # fidelity vs φ
            phis30 = np.arange(0,360,30)
            fid30  = []
            for φ in phis30:
                f, *_ = compute_fidelity_at_boresight(
                    list(files), dom_pol=var_pol.get(), theta0=90, phi0=float(φ),
                    band=None, Nfft=NFFT_FID
                )
                fid30.append(f)
            plot_fidelity_vs_phi(phis30, np.array(fid30), full_band)

            # — FIDELITY vs θ at 15° steps over full band —
            thetas = np.arange(0,181,15)
            fid_th = []
            for θ in thetas:
                f, *_ = compute_fidelity_at_boresight(
                    list(files), dom_pol=var_pol.get(), theta0=float(θ), phi0=0.0,
                    band=None, Nfft=NFFT_FID
                )
                fid_th.append(f)
            plot_fidelity_vs_theta(thetas, np.array(fid_th), full_band)
        else: # ECC mmode    
            # — pick ANT1 files —
            ant1_files = filedialog.askopenfilenames(
                title="Select ANT 1 CST far-field .txt files",
                filetypes=[("Text files","*.txt")])
            if not ant1_files:
                return
            # — pick ANT2 files —
            ant2_files = filedialog.askopenfilenames(
                title="Select ANT 2 CST far-field .txt files",
                filetypes=[("Text files","*.txt")])
            if not ant2_files:
                return

            # find all center‐frequencies in each set
            f1 = {extract_frequency(p):p for p in ant1_files}
            f2 = {extract_frequency(p):p for p in ant2_files}
            common_fs = sorted(set(f1).intersection(f2))
            if not common_fs:
                messagebox.showerror("No match",
                    "No overlapping frequencies between ANT1 and ANT2 files.")
                return

            # sample according to Low/Mid/High/Single
            if var_ecc_freq.get() == 'Sample3Freq':
                freqs = [common_fs[0],
                         common_fs[len(common_fs)//2],
                         common_fs[-1]]
            elif var_ecc_freq.get() == 'SingleFreq':
                idx = simpledialog.askinteger(
                    "Pick frequency index",
                    f"Select index 0–{len(common_fs)-1}:")
                if idx is None or idx<0 or idx>=len(common_fs):
                    return
                freqs = [common_fs[idx]]
            else:  # FullBand
                freqs = common_fs

            # compute ECC at each selected frequency
            ecc_results = []
            for f in freqs:
                p1 = f1[f]
                p2 = f2[f]
                ecc = compute_ecc_from_farfield(p1, p2, f)
                ecc_results.append((f, ecc))

            plot_ecc_summary(ecc_results)
    
    
    tk.Button(root, text="Run Analysis", command=proceed).pack(pady=20)
    root.mainloop()

if __name__=="__main__":
    main()
