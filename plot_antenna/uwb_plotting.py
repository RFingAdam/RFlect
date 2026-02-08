"""
UWB Antenna Plotting Module

Visualization functions for UWB antenna analysis results.
All functions take computed dicts from uwb_analysis and return matplotlib Figure objects.
"""

import numpy as np
import matplotlib

matplotlib.use("TkAgg")
import matplotlib.pyplot as plt


# Quality color thresholds for SFF bar charts
_SFF_COLORS = [
    (0.95, "#2ecc71"),  # Excellent - green
    (0.85, "#27ae60"),  # Very Good - dark green
    (0.70, "#f39c12"),  # Good - orange
    (0.50, "#e67e22"),  # Fair - dark orange
    (0.00, "#e74c3c"),  # Poor - red
]


def _sff_color(sff_value):
    """Return color based on SFF quality threshold."""
    for threshold, color in _SFF_COLORS:
        if sff_value >= threshold:
            return color
    return "#e74c3c"


def plot_sff_vs_angle(angles, sff_values):
    """Bar chart of SFF vs angle with quality color bands.

    Args:
        angles: list/array of angle values in degrees.
        sff_values: list/array of corresponding SFF values.

    Returns:
        matplotlib.figure.Figure
    """
    fig, ax = plt.subplots(figsize=(10, 5))
    colors = [_sff_color(v) for v in sff_values]

    bars = ax.bar(
        [str(a) for a in angles], sff_values, color=colors, edgecolor="black", linewidth=0.5
    )

    # Quality band shading
    ax.axhspan(0.95, 1.0, alpha=0.08, color="#2ecc71", label="Excellent (>0.95)")
    ax.axhspan(0.85, 0.95, alpha=0.08, color="#27ae60", label="Very Good (0.85-0.95)")
    ax.axhspan(0.70, 0.85, alpha=0.08, color="#f39c12", label="Good (0.70-0.85)")
    ax.axhspan(0.50, 0.70, alpha=0.08, color="#e67e22", label="Fair (0.50-0.70)")
    ax.axhspan(0.00, 0.50, alpha=0.08, color="#e74c3c", label="Poor (<0.50)")

    mean_sff = np.mean(sff_values)
    ax.axhline(
        mean_sff, color="navy", linestyle="--", linewidth=1.5, label=f"Mean SFF = {mean_sff:.3f}"
    )

    ax.set_xlabel("Angle (deg)")
    ax.set_ylabel("System Fidelity Factor")
    ax.set_title("System Fidelity Factor vs Angle")
    ax.set_ylim(0, 1.05)
    ax.legend(loc="lower left", fontsize=8)
    ax.grid(True, axis="y", linestyle="--", alpha=0.5)
    fig.tight_layout()

    return fig


def plot_group_delay_vs_freq(freq_hz, group_delay_s, angles_dict=None):
    """Group delay vs frequency with optional multi-angle overlay.

    Args:
        freq_hz: 1D array of frequencies in Hz.
        group_delay_s: 1D array of group delay in seconds (used if angles_dict is None).
        angles_dict: optional dict {angle_label: group_delay_array} for multi-angle plot.

    Returns:
        matplotlib.figure.Figure
    """
    fig, ax = plt.subplots(figsize=(10, 5))

    if angles_dict is not None:
        for label, gd in angles_dict.items():
            ax.plot(freq_hz / 1e9, gd * 1e9, label=label, linewidth=0.8)
        # Compute mean and variation
        all_gd = np.array(list(angles_dict.values()))
        mean_gd = np.mean(all_gd, axis=0)
        ax.plot(freq_hz / 1e9, mean_gd * 1e9, "k--", linewidth=1.5, label="Mean")
        variation_ps = np.ptp(all_gd) * 1e12
        ax.set_title(f"Group Delay vs Frequency (pk-pk variation: {variation_ps:.1f} ps)")
    else:
        ax.plot(freq_hz / 1e9, group_delay_s * 1e9, "b-", linewidth=1.0)
        mean_val = np.mean(group_delay_s) * 1e9
        variation_ps = np.ptp(group_delay_s) * 1e12
        ax.axhline(mean_val, color="red", linestyle="--", label=f"Mean = {mean_val:.2f} ns")
        ax.set_title(f"Group Delay vs Frequency (variation: {variation_ps:.1f} ps)")

    ax.set_xlabel("Frequency (GHz)")
    ax.set_ylabel("Group Delay (ns)")
    ax.legend(loc="best", fontsize=8)
    ax.grid(True, linestyle="--", alpha=0.5)
    fig.tight_layout()

    return fig


def plot_impulse_response(time_s, h_t):
    """Impulse response with -3dB width annotation.

    Args:
        time_s: 1D time array in seconds.
        h_t: 1D impulse response array.

    Returns:
        matplotlib.figure.Figure
    """
    fig, ax = plt.subplots(figsize=(10, 5))

    t_ns = time_s * 1e9
    h_norm = h_t / np.max(np.abs(h_t)) if np.max(np.abs(h_t)) > 0 else h_t

    ax.plot(t_ns, h_norm, "b-", linewidth=0.8)

    # Find -3dB width
    threshold = 1.0 / np.sqrt(2)
    above = np.abs(h_norm) >= threshold
    indices = np.where(above)[0]
    if len(indices) >= 2:
        t_start = t_ns[indices[0]]
        t_end = t_ns[indices[-1]]
        width = t_end - t_start
        ax.axhline(threshold, color="red", linestyle=":", alpha=0.5, label=f"-3dB level")
        ax.axhline(-threshold, color="red", linestyle=":", alpha=0.5)
        ax.annotate(
            f"Width = {width:.2f} ns",
            xy=((t_start + t_end) / 2, threshold),
            xytext=(0, 15),
            textcoords="offset points",
            ha="center",
            fontsize=9,
            arrowprops=dict(arrowstyle="->", color="red"),
        )

    ax.set_xlabel("Time (ns)")
    ax.set_ylabel("Normalized Amplitude")
    ax.set_title("Impulse Response h(t)")

    # Auto-zoom to signal region
    peak_idx = np.argmax(np.abs(h_norm))
    window = min(len(t_ns) // 4, 500)
    left = max(0, peak_idx - window)
    right = min(len(t_ns), peak_idx + window)
    ax.set_xlim(t_ns[left], t_ns[right])

    ax.legend(loc="best")
    ax.grid(True, linestyle="--", alpha=0.5)
    fig.tight_layout()

    return fig


def plot_transfer_function(freq_hz, H_mag_dB, H_phase_deg):
    """Dual-axis plot of transfer function magnitude and phase.

    Args:
        freq_hz: 1D array of frequencies in Hz.
        H_mag_dB: 1D magnitude array in dB.
        H_phase_deg: 1D phase array in degrees.

    Returns:
        matplotlib.figure.Figure
    """
    fig, ax1 = plt.subplots(figsize=(10, 5))

    color_mag = "#2980b9"
    color_phase = "#e74c3c"

    ax1.plot(freq_hz / 1e9, H_mag_dB, color=color_mag, linewidth=1.0)
    ax1.set_xlabel("Frequency (GHz)")
    ax1.set_ylabel("|H(f)| (dB)", color=color_mag)
    ax1.tick_params(axis="y", labelcolor=color_mag)
    ax1.grid(True, linestyle="--", alpha=0.3)

    ax2 = ax1.twinx()
    ax2.plot(freq_hz / 1e9, H_phase_deg, color=color_phase, linewidth=0.8, linestyle="--")
    ax2.set_ylabel("Phase (deg)", color=color_phase)
    ax2.tick_params(axis="y", labelcolor=color_phase)

    ax1.set_title("Antenna Transfer Function H(f)")
    fig.tight_layout()

    return fig


def plot_input_vs_output_pulse(time_s, input_pulse, output_pulse, sff, delay_s):
    """Overlay of input and output pulses with SFF annotation.

    Args:
        time_s: 1D time array in seconds.
        input_pulse: 1D input pulse array.
        output_pulse: 1D output pulse array.
        sff: float SFF value.
        delay_s: float estimated delay in seconds.

    Returns:
        matplotlib.figure.Figure
    """
    fig, ax = plt.subplots(figsize=(10, 5))

    t_ns = time_s * 1e9
    delay_ns = delay_s * 1e9

    # Normalize
    inp = (
        input_pulse / np.max(np.abs(input_pulse))
        if np.max(np.abs(input_pulse)) > 0
        else input_pulse
    )
    out = (
        output_pulse / np.max(np.abs(output_pulse))
        if np.max(np.abs(output_pulse)) > 0
        else output_pulse
    )

    # Trim output to same length
    n = min(len(inp), len(out))

    ax.plot(t_ns[:n], inp[:n], "b-", linewidth=1.0, label="Input Pulse")
    ax.plot(
        t_ns[:n] + delay_ns,
        out[:n],
        "r--",
        linewidth=1.0,
        label=f"Output Pulse (delay = {delay_ns:.2f} ns)",
    )

    ax.set_xlabel("Time (ns)")
    ax.set_ylabel("Normalized Amplitude")
    ax.set_title(f"Input vs Output Pulse — SFF = {sff:.4f}")
    ax.legend(loc="best")
    ax.grid(True, linestyle="--", alpha=0.5)

    # Auto-zoom to pulse region
    sig = np.abs(inp[:n])
    above_thresh = np.where(sig > 0.05 * np.max(sig))[0]
    if len(above_thresh) > 0:
        margin = max(len(t_ns) // 20, 50)
        left = max(0, above_thresh[0] - margin)
        right = min(
            n,
            above_thresh[-1] + margin + int(delay_ns / (t_ns[1] - t_ns[0]) if len(t_ns) > 1 else 0),
        )
        right = min(right, n)
        ax.set_xlim(t_ns[left], t_ns[right] + delay_ns)

    fig.tight_layout()
    return fig


def plot_s11_vswr(freq_hz, s11_dB, vswr, bandwidth_info):
    """Dual plot of S11 and VSWR with bandwidth shading.

    Args:
        freq_hz: 1D array of frequencies in Hz.
        s11_dB: 1D array of S11 in dB.
        vswr: 1D array of VSWR values.
        bandwidth_info: dict with 'band_start_hz', 'band_stop_hz',
                        'bandwidth_hz', 'fractional_bandwidth'.

    Returns:
        matplotlib.figure.Figure
    """
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

    f_ghz = freq_hz / 1e9

    # S11 plot
    ax1.plot(f_ghz, s11_dB, "b-", linewidth=1.0)
    ax1.axhline(-10, color="red", linestyle="--", alpha=0.7, label="S11 = -10 dB")
    ax1.set_ylabel("S11 (dB)")
    ax1.set_title("Return Loss (S11)")
    ax1.grid(True, linestyle="--", alpha=0.5)
    ax1.legend(loc="best")

    # Bandwidth shading
    if bandwidth_info["bandwidth_hz"] > 0:
        f_start = bandwidth_info["band_start_hz"] / 1e9
        f_stop = bandwidth_info["band_stop_hz"] / 1e9
        bw_mhz = bandwidth_info["bandwidth_hz"] / 1e6
        frac_bw = bandwidth_info["fractional_bandwidth"] * 100

        ax1.axvspan(
            f_start,
            f_stop,
            alpha=0.15,
            color="green",
            label=f"BW = {bw_mhz:.0f} MHz ({frac_bw:.0f}%)",
        )
        ax1.legend(loc="best")

        ax2.axvspan(f_start, f_stop, alpha=0.15, color="green")

    # VSWR plot
    ax2.plot(f_ghz, vswr, "r-", linewidth=1.0)
    ax2.axhline(2.0, color="blue", linestyle="--", alpha=0.7, label="VSWR = 2:1")
    ax2.set_xlabel("Frequency (GHz)")
    ax2.set_ylabel("VSWR")
    ax2.set_title("Voltage Standing Wave Ratio")
    ax2.set_ylim(1, min(10, np.max(vswr) * 1.1))
    ax2.grid(True, linestyle="--", alpha=0.5)
    ax2.legend(loc="best")

    fig.tight_layout()
    return fig


def plot_group_delay_variation(freq_hz, angles_dict):
    """Peak-to-peak group delay variation across angles.

    Args:
        freq_hz: 1D array of frequencies in Hz.
        angles_dict: dict {angle_label: group_delay_array}.

    Returns:
        matplotlib.figure.Figure
    """
    fig, ax = plt.subplots(figsize=(10, 5))

    all_gd = np.array(list(angles_dict.values()))
    variation_ps = (np.max(all_gd, axis=0) - np.min(all_gd, axis=0)) * 1e12

    ax.plot(freq_hz / 1e9, variation_ps, "b-", linewidth=1.0)
    mean_var = np.mean(variation_ps)
    ax.axhline(mean_var, color="red", linestyle="--", label=f"Mean variation = {mean_var:.1f} ps")

    ax.set_xlabel("Frequency (GHz)")
    ax.set_ylabel("Group Delay Variation (ps)")
    ax.set_title("Peak-to-Peak Group Delay Variation Across Angles")
    ax.legend(loc="best")
    ax.grid(True, linestyle="--", alpha=0.5)
    fig.tight_layout()

    return fig
