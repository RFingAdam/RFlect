"""Advanced-analysis plot family (#38).

Extracted verbatim from plotting.py to shrink that module: the link-budget,
indoor-coverage, fading, MIMO and wearable plot builders. They are self-contained
(numpy + pyplot + RF-math helpers from calculations.py; no other plotting.py
internals), so the move is free of circular-import risk. plotting.py re-exports
these names, so `from .plotting import plot_link_budget_summary` still works, and
the advanced-analysis dispatcher (which stays in plotting.py) calls them via that
re-export.
"""

from __future__ import annotations

import os

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize

from .config import MIMO_SNR_RANGE_DB

from .calculations import (
    friis_range_estimate,
    min_tx_gain_for_range,
    link_margin,
    range_vs_azimuth,
    free_space_path_loss,
    log_distance_path_loss,
    wall_penetration_loss,
    rayleigh_cdf,
    rician_cdf,
    fade_margin_for_reliability,
    apply_statistical_fading,
    combining_gain,
    mimo_capacity_vs_snr,
    mean_effective_gain_mimo,
    body_worn_pattern_analysis,
    dense_device_interference,
)


def plot_link_budget_summary(
    freq_mhz,
    gain_2d,
    theta_deg,
    phi_deg,
    pt_dbm=0.0,
    pr_dbm=-98.0,
    gr_dbi=0.0,
    path_loss_exp=2.0,
    misc_loss_db=10.0,
    target_range_m=5.0,
    data_label="Gain",
    data_unit="dBi",
    save_path=None,
):
    """
    Link budget summary: waterfall chart + range-vs-azimuth polar plot.

    Left panel:  Link budget table showing all parameters and derived values
    Right panel: Range vs azimuth polar plot at θ=90° (horizon)
    """
    _ = data_unit  # kept in signature for API consistency
    # Determine if active (EIRP) or passive (Gain) data
    is_active = data_label != "Gain"

    # Get horizon gain/EIRP and range per azimuth
    range_m, horizon_gain = range_vs_azimuth(
        gain_2d,
        theta_deg,
        phi_deg,
        freq_mhz,
        pt_dbm=0.0 if is_active else pt_dbm,  # EIRP already includes Pt
        pr_dbm=pr_dbm,
        gr_dbi=gr_dbi,
        path_loss_exp=path_loss_exp,
        misc_loss_db=misc_loss_db,
    )

    peak_gain = float(np.max(horizon_gain))
    worst_gain = float(np.min(horizon_gain))

    # Peak and worst-case range
    peak_range = friis_range_estimate(
        pt_dbm if not is_active else 0.0,
        pr_dbm,
        peak_gain,
        gr_dbi,
        freq_mhz,
        path_loss_exp,
        misc_loss_db,
    )
    worst_range = friis_range_estimate(
        pt_dbm if not is_active else 0.0,
        pr_dbm,
        worst_gain,
        gr_dbi,
        freq_mhz,
        path_loss_exp,
        misc_loss_db,
    )

    # Min Tx gain for target range
    min_gt = min_tx_gain_for_range(
        target_range_m,
        pt_dbm if not is_active else 0.0,
        pr_dbm,
        gr_dbi,
        freq_mhz,
        path_loss_exp,
        misc_loss_db,
    )

    # Link margin at target range with peak gain
    margin = link_margin(
        pt_dbm if not is_active else 0.0,
        peak_gain,
        gr_dbi,
        freq_mhz,
        target_range_m,
        path_loss_exp,
        misc_loss_db,
        pr_dbm,
    )

    # FSPL at 1m reference
    fspl_1m = free_space_path_loss(freq_mhz, 1.0)
    pl_target = fspl_1m + 10 * path_loss_exp * np.log10(max(target_range_m, 0.01))

    # ---- Figure ----
    fig = plt.figure(figsize=(16, 7))
    fig_gs = fig.add_gridspec(1, 2, width_ratios=[1, 1.2])

    # --- Left: Link budget table ---
    ax_table = fig.add_subplot(fig_gs[0])
    ax_table.axis("off")

    gt_label = "Peak Horizon EIRP (θ=90°)" if is_active else "Peak Horizon Gain (θ=90°)"
    gt_value = f"{peak_gain:.1f} dBm" if is_active else f"{peak_gain:.1f} dBi"

    min_gt_label = "Min EIRP for target" if is_active else "Min Gt for target range"
    min_gt_unit = "dBm" if is_active else "dBi"

    table_data = []
    if not is_active:
        table_data.append(["Tx Power (Pt)", f"{pt_dbm:.1f} dBm"])
    table_data.extend(
        [
            [gt_label, gt_value],
            [f"Path Loss @ {target_range_m:.1f}m", f"{-abs(pl_target):.1f} dB"],
            ["Misc Losses", f"{-abs(misc_loss_db):.1f} dB"],
            ["Rx Gain (Gr)", f"{gr_dbi:.1f} dBi"],
            ["Rx Sensitivity", f"{pr_dbm:.1f} dBm"],
            ["", ""],
            ["Link Margin @ target", f"{margin:+.1f} dB"],
            ["Peak Range", f"{peak_range:.1f} m"],
            ["Worst-Case Range", f"{worst_range:.1f} m"],
            [min_gt_label, f"{min_gt:.1f} {min_gt_unit}"],
            ["Frequency", f"{freq_mhz:.1f} MHz"],
            ["Path Loss Exponent (n)", f"{path_loss_exp:.1f}"],
        ]
    )

    table = ax_table.table(
        cellText=table_data,
        colLabels=["Parameter", "Value"],
        cellLoc="left",
        loc="center",
        colWidths=[0.55, 0.45],
    )
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 1.4)

    for j in range(2):
        table[0, j].set_facecolor("#4A90E2")
        table[0, j].set_text_props(color="white", fontweight="bold")

    # Highlight margin row — find explicitly by content rather than fragile offset
    margin_row_data_idx = next(
        (i for i, row in enumerate(table_data) if "Link Margin" in row[0]), None
    )
    if margin_row_data_idx is not None:
        margin_row_tbl = margin_row_data_idx + 1  # +1 for header row in table
        color = "#4CAF50" if margin >= 0 else "#F44336"
        for j in range(2):
            table[margin_row_tbl, j].set_facecolor(color)
            table[margin_row_tbl, j].set_text_props(color="white", fontweight="bold")

    ax_table.set_title(
        f"Link Budget Summary — {freq_mhz} MHz",
        fontsize=13,
        fontweight="bold",
        pad=20,
    )

    # --- Right: Range vs Azimuth polar plot ---
    ax_polar = fig.add_subplot(fig_gs[1], projection="polar")
    phi_rad = np.deg2rad(phi_deg)

    # Colour-code by whether range meets target
    colors = np.where(range_m >= target_range_m, "#4CAF50", "#F44336")
    bar_width = np.deg2rad(np.mean(np.diff(phi_deg))) if len(phi_deg) > 1 else np.deg2rad(5)
    ax_polar.bar(
        phi_rad, range_m, width=bar_width, color=colors, alpha=0.7, edgecolor="gray", linewidth=0.3
    )

    # Target range ring — close the loop so the dashed circle is complete
    phi_rad_closed = np.append(phi_rad, phi_rad[0])
    target_ring = np.full_like(phi_rad_closed, target_range_m)
    ax_polar.plot(
        phi_rad_closed, target_ring, "k--", linewidth=1.5, label=f"Target: {target_range_m:.0f} m"
    )

    ax_polar.set_title(
        "Range vs Azimuth (θ=90°)\nGreen = meets target, Red = below",
        fontsize=11,
        fontweight="bold",
        pad=15,
    )
    ax_polar.set_theta_zero_location("N")
    ax_polar.set_theta_direction(-1)
    ax_polar.set_rlabel_position(67.5)  # move radial labels clear of data
    ax_polar.legend(loc="upper right", bbox_to_anchor=(1.3, 1.1), fontsize=9)

    plt.tight_layout()

    if save_path:
        fname = f"link_budget_{freq_mhz}MHz.png"
        fig.savefig(os.path.join(save_path, fname), dpi=300, bbox_inches="tight")
        plt.close(fig)
    else:
        plt.show()


# ——— INDOOR PROPAGATION PLOTS ————————————————————————————————————


def plot_indoor_coverage_map(
    freq_mhz,
    gain_2d,
    theta_deg,
    phi_deg,
    pt_dbm=0.0,
    pr_sensitivity_dbm=-98.0,
    environment="Office",
    path_loss_exp=3.0,
    n_walls=1,
    wall_material="drywall",
    shadow_fading_db=5.0,
    max_distance_m=30.0,
    data_label="Gain",
    data_unit="dBi",
    save_path=None,
):
    """
    Indoor coverage analysis: path loss curves + azimuthal coverage heatmap + range contour.
    """
    _ = data_unit  # kept in signature for API consistency
    is_active = data_label != "Gain"
    distances = np.linspace(0.5, max_distance_m, 60)

    # Path loss models
    pl_free = free_space_path_loss(freq_mhz, distances)
    pl_indoor = log_distance_path_loss(freq_mhz, distances, n=path_loss_exp)
    wl = wall_penetration_loss(freq_mhz, wall_material)
    pl_walls = pl_indoor + n_walls * wl
    pl_shadow = pl_walls + shadow_fading_db

    # Horizon gain per azimuth
    theta_90_idx = np.argmin(np.abs(theta_deg - 90.0))
    horizon_gain = gain_2d[theta_90_idx, :]

    # Received power heatmap: Pr(phi, d) = Pt + G(phi) - PL(d) - shadow_margin
    effective_pt = 0.0 if is_active else pt_dbm
    pl_total = pl_walls + shadow_fading_db  # include shadow fading margin
    pr_map = effective_pt + horizon_gain[np.newaxis, :] - pl_total[:, np.newaxis]

    # Coverage range per azimuth (with shadow fading margin applied)
    coverage_range = np.zeros(len(phi_deg))
    fspl_1m = free_space_path_loss(freq_mhz, 1.0)
    total_wall_loss = n_walls * wl + shadow_fading_db
    for i, g in enumerate(horizon_gain):
        allowed_pl = effective_pt + g - pr_sensitivity_dbm
        net_pl = allowed_pl - total_wall_loss
        if path_loss_exp > 0 and net_pl > fspl_1m:
            coverage_range[i] = 10 ** ((net_pl - fspl_1m) / (10 * path_loss_exp))
        else:
            coverage_range[i] = 0.5

    # ---- Figure ----
    fig = plt.figure(figsize=(18, 6.5))
    fig_gs = fig.add_gridspec(1, 3, width_ratios=[1, 1.3, 1])

    # --- Left: Path Loss vs Distance ---
    ax_pl = fig.add_subplot(fig_gs[0])
    ax_pl.plot(distances, pl_free, "b--", linewidth=1.5, label="Free Space (n=2)")
    ax_pl.plot(distances, pl_indoor, "g-", linewidth=1.5, label=f"Indoor (n={path_loss_exp})")
    ax_pl.plot(
        distances,
        pl_walls,
        "r-",
        linewidth=2,
        label=f"+ {n_walls}× {wall_material} ({n_walls * wl:.1f} dB)",
    )
    ax_pl.plot(
        distances, pl_shadow, "r:", linewidth=1, label=f"+ {shadow_fading_db:.0f} dB shadow margin"
    )
    ax_pl.set_xlabel("Distance (m)")
    ax_pl.set_ylabel("Path Loss (dB)")
    ax_pl.set_title("Path Loss Models", fontsize=11, fontweight="bold")
    ax_pl.legend(fontsize=8, loc="upper left")
    ax_pl.grid(True, alpha=0.3)
    ax_pl.invert_yaxis()

    # --- Center: Received Power Heatmap ---
    ax_hm = fig.add_subplot(fig_gs[1])
    dphi = phi_deg[1] - phi_deg[0] if len(phi_deg) > 1 else 15.0
    extent = [phi_deg[0] - dphi / 2, phi_deg[-1] + dphi / 2, distances[0], distances[-1]]
    vmin = pr_sensitivity_dbm - 20
    vmax = float(np.max(pr_map))
    im = ax_hm.imshow(
        pr_map,
        aspect="auto",
        origin="lower",
        extent=extent,
        cmap="RdYlGn",
        vmin=vmin,
        vmax=vmax,
    )
    ax_hm.contour(
        phi_deg,
        distances,
        pr_map,
        levels=[pr_sensitivity_dbm],
        colors=["black"],
        linewidths=[2],
        linestyles=["--"],
    )
    ax_hm.set_xlabel("Azimuth φ (°)")
    ax_hm.set_ylabel("Distance (m)")
    ax_hm.set_title(
        f"Received Power at Horizon (θ=90°)\n{environment} — {freq_mhz} MHz",
        fontsize=11,
        fontweight="bold",
    )
    cbar = fig.colorbar(im, ax=ax_hm, shrink=0.8)
    cbar.set_label("Received Power (dBm)")

    # --- Right: Coverage Range Polar ---
    ax_polar = fig.add_subplot(fig_gs[2], projection="polar")
    phi_rad_closed = np.append(np.deg2rad(phi_deg), np.deg2rad(phi_deg[0]))
    cr_closed = np.append(coverage_range, coverage_range[0])
    ax_polar.fill(phi_rad_closed, cr_closed, alpha=0.3, color="#4CAF50")
    ax_polar.plot(phi_rad_closed, cr_closed, "g-", linewidth=2, label="Coverage range")
    ax_polar.set_title(
        f"Coverage Range @ {pr_sensitivity_dbm:.0f} dBm\n"
        f"{n_walls}× {wall_material} + {shadow_fading_db:.0f} dB shadow margin",
        fontsize=11,
        fontweight="bold",
        pad=15,
    )
    ax_polar.set_theta_zero_location("N")
    ax_polar.set_theta_direction(-1)

    avg_range = np.mean(coverage_range)
    min_range = np.min(coverage_range)
    max_range = np.max(coverage_range)
    summary = f"Avg: {avg_range:.1f}m  Min: {min_range:.1f}m  Max: {max_range:.1f}m"
    fig.text(
        0.5,
        0.02,
        summary,
        ha="center",
        fontsize=10,
        bbox=dict(facecolor="white", edgecolor="gray", alpha=0.8, boxstyle="round,pad=0.3"),
    )

    plt.tight_layout(rect=[0, 0.06, 1, 1])

    if save_path:
        fname = f"indoor_coverage_{freq_mhz}MHz.png"
        fig.savefig(os.path.join(save_path, fname), dpi=300, bbox_inches="tight")
        plt.close(fig)
    else:
        plt.show()


# ——— MULTIPATH FADING PLOTS ——————————————————————————————————————


def plot_fading_analysis(
    freq_mhz,
    gain_2d,
    theta_deg,
    phi_deg,
    pr_sensitivity_dbm=-98.0,
    pt_dbm=0.0,
    target_reliability=99.0,
    fading_model="rayleigh",
    fading_rician_k=10.0,
    realizations=1000,
    target_distance_m=5.0,
    path_loss_exp=2.0,
    misc_loss_db=0.0,
    gr_dbi=0.0,
    data_label="Gain",
    data_unit="dBi",
    save_path=None,
):
    """
    Fading analysis: CDF curves, fade margin chart, pattern with fading envelope,
    and outage probability bar chart.

    The outage subplot computes received power at ``target_distance_m`` using
    a log-distance path loss model so results are physically meaningful.
    """
    is_active = data_label != "Gain"
    model = str(fading_model).strip().lower()
    if model not in ("rayleigh", "rician"):
        model = "rayleigh"
    k_factor = max(float(fading_rician_k), 0.0)
    n_realizations = max(int(realizations), 10)

    # Peak direction
    peak_idx = np.unravel_index(np.argmax(gain_2d), gain_2d.shape)
    peak_val = gain_2d[peak_idx]

    # Power range for CDF
    power_range = np.linspace(peak_val - 40, peak_val + 5, 200)

    # CDF curves
    if model == "rayleigh":
        cdf_selected = rayleigh_cdf(power_range, peak_val)
        cdf_reference = rician_cdf(power_range, peak_val, K_factor=max(k_factor, 1.0))
        selected_label = "Rayleigh (selected)"
        reference_label = f"Rician K={max(k_factor, 1.0):.1f}"
    else:
        cdf_selected = rician_cdf(power_range, peak_val, K_factor=max(k_factor, 0.1))
        cdf_reference = rayleigh_cdf(power_range, peak_val)
        selected_label = f"Rician K={max(k_factor, 0.1):.1f} (selected)"
        reference_label = "Rayleigh"

    # Fade margins for reliability range
    reliability_range = np.linspace(50, 99.99, 100)
    margin_selected = [
        fade_margin_for_reliability(r, model, K=max(k_factor, 0.1)) for r in reliability_range
    ]
    if model == "rayleigh":
        margin_reference = [
            fade_margin_for_reliability(r, "rician", K=max(k_factor, 1.0))
            for r in reliability_range
        ]
        margin_reference_label = f"Rician K={max(k_factor, 1.0):.1f}"
    else:
        margin_reference = [fade_margin_for_reliability(r, "rayleigh") for r in reliability_range]
        margin_reference_label = "Rayleigh"

    # Monte-Carlo fading at horizon
    theta_90_idx = np.argmin(np.abs(theta_deg - 90.0))
    horizon_slice = gain_2d[theta_90_idx : theta_90_idx + 1, :]
    mean_db, std_db, p5_db = apply_statistical_fading(
        horizon_slice,
        theta_deg[theta_90_idx : theta_90_idx + 1],
        phi_deg,
        fading=model,
        K=max(k_factor, 0.1),
        realizations=n_realizations,
    )

    # ---- Figure ----
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # --- Top-left: CDF curves ---
    ax = axes[0, 0]
    ax.semilogy(power_range, 1 - cdf_selected, "b-", linewidth=2, label=selected_label)
    ax.semilogy(power_range, 1 - cdf_reference, "k--", linewidth=1.2, label=reference_label)
    ax.axhline(y=0.01, color="gray", linestyle=":", alpha=0.7, label="99% reliability")
    ax.set_xlabel(f"{data_label} ({data_unit})")
    ax.set_ylabel("P(signal > x) - CCDF")
    ax.set_title("Fading CCDF at Peak Direction", fontweight="bold")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3, which="both")
    ax.set_ylim(1e-4, 1)

    # --- Top-right: Fade Margin vs Reliability ---
    ax = axes[0, 1]
    ax.plot(reliability_range, margin_selected, "b-", linewidth=2, label=selected_label)
    ax.plot(
        reliability_range,
        margin_reference,
        "k--",
        linewidth=1.2,
        label=margin_reference_label,
    )
    ax.axvline(x=target_reliability, color="gray", linestyle=":", alpha=0.7)
    target_margin = fade_margin_for_reliability(target_reliability, model, K=max(k_factor, 0.1))
    ax.plot(target_reliability, target_margin, "bo", markersize=8)
    ax.annotate(
        f"{target_margin:.1f} dB",
        (target_reliability, target_margin),
        textcoords="offset points",
        xytext=(10, 5),
        fontsize=9,
    )
    ax.set_xlabel("Reliability (%)")
    ax.set_ylabel("Required Fade Margin (dB)")
    ax.set_title("Fade Margin vs Reliability", fontweight="bold")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # --- Bottom-left: Pattern with fading envelope at horizon ---
    ax = axes[1, 0]
    mean_flat = mean_db.flatten()
    std_flat = std_db.flatten()
    p5_flat = p5_db.flatten()
    ax.fill_between(
        phi_deg,
        mean_flat - std_flat,
        mean_flat + std_flat,
        alpha=0.2,
        color="blue",
        label=r"$\pm 1\sigma$ envelope",
    )
    ax.plot(phi_deg, mean_flat, "b-", linewidth=2, label="Mean (faded)")
    ax.plot(phi_deg, gain_2d[theta_90_idx, :], "k--", linewidth=1, label="Free-space")
    ax.plot(phi_deg, p5_flat, "r:", linewidth=1, label="5th percentile")
    ax.set_xlabel("Azimuth phi (deg)")
    ax.set_ylabel(f"{data_label} ({data_unit})")
    if model == "rician":
        fade_title = f"Rician Fading Envelope at theta=90 deg (K={max(k_factor, 0.1):.1f})"
    else:
        fade_title = "Rayleigh Fading Envelope at theta=90 deg"
    ax.set_title(f"{fade_title}\n({n_realizations} realizations)", fontweight="bold")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # --- Bottom-right: Outage probability per azimuth ---
    ax = axes[1, 1]
    horizon_vals = gain_2d[theta_90_idx, :]
    effective_pt = 0.0 if is_active else pt_dbm
    # Compute mean received power including path loss at target distance
    fspl_d0 = free_space_path_loss(freq_mhz, 1.0)
    pl_at_target = fspl_d0 + 10.0 * path_loss_exp * np.log10(max(target_distance_m, 0.01))
    mean_rx = effective_pt + horizon_vals + gr_dbi - pl_at_target - misc_loss_db
    rx_threshold = np.full_like(horizon_vals, pr_sensitivity_dbm)
    if model == "rician":
        outage_prob = rician_cdf(rx_threshold, mean_rx, K_factor=max(k_factor, 0.1))
        outage_title = f"Rician Outage per Azimuth (K={max(k_factor, 0.1):.1f})"
    else:
        outage_prob = rayleigh_cdf(rx_threshold, mean_rx)
        outage_title = "Rayleigh Outage per Azimuth"
    target_outage_pct = 100.0 - target_reliability
    bar_width = np.mean(np.diff(phi_deg)) * 0.8 if len(phi_deg) > 1 else 3.0
    colors_out = []
    for op in outage_prob:
        if op < target_outage_pct / 100.0:
            colors_out.append("#4CAF50")
        elif op < 0.1:
            colors_out.append("#FFC107")
        else:
            colors_out.append("#F44336")
    ax.bar(
        phi_deg,
        outage_prob * 100,
        width=bar_width,
        color=colors_out,
        edgecolor="gray",
        linewidth=0.3,
    )
    ax.axhline(
        y=target_outage_pct,
        color="r",
        linestyle="--",
        linewidth=1,
        label=f"{target_outage_pct:.0g}% outage",
    )
    ax.set_xlabel("Azimuth phi (deg)")
    ax.set_ylabel("Outage Probability (%)")
    ax.set_title(
        f"{outage_title}\n(d={target_distance_m:.0f} m, Rx={pr_sensitivity_dbm:.0f} dBm)",
        fontweight="bold",
    )
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    fig.suptitle(f"Multipath Fading Analysis - {freq_mhz} MHz", fontsize=14, fontweight="bold")
    plt.tight_layout(rect=[0, 0, 1, 0.96])

    if save_path:
        fname = f"fading_analysis_{freq_mhz}MHz.png"
        fig.savefig(os.path.join(save_path, fname), dpi=300, bbox_inches="tight")
        plt.close(fig)
    else:
        plt.show()


def plot_mimo_analysis(
    ecc_values,
    freq_list,
    gain_data_list,
    theta_deg,
    phi_deg,
    snr_db=20,
    snr_range_db=MIMO_SNR_RANGE_DB,
    fading="rayleigh",
    K=10,
    xpr_db=6.0,
    save_path=None,
):
    """
    MIMO analysis: capacity curves, combining gain comparison, pattern overlay.
    """
    _ = freq_list  # reserved for per-frequency analysis in future
    if ecc_values is None:
        ecc_values = []
    if gain_data_list is None:
        gain_data_list = []

    fig = plt.figure(figsize=(18, 6.5))
    fig_gs = fig.add_gridspec(1, 3, width_ratios=[1, 1, 1])

    # --- Left: Capacity vs SNR ---
    ax_cap = fig.add_subplot(fig_gs[0])
    ecc_arr = np.asarray(ecc_values, dtype=float).reshape(-1)
    ecc_arr = ecc_arr[np.isfinite(ecc_arr)]
    ecc_median = float(np.median(ecc_arr)) if ecc_arr.size > 0 else 0.3
    ecc_median = float(np.clip(ecc_median, 0.0, 1.0))

    try:
        snr_lo, snr_hi = snr_range_db
    except Exception:
        snr_lo, snr_hi = -5, 30
    if snr_lo >= snr_hi:
        snr_lo, snr_hi = -5, 30

    snr_axis, siso_cap, awgn_cap, fading_cap = mimo_capacity_vs_snr(
        ecc_median,
        snr_range_db=(snr_lo, snr_hi),
        fading=fading,
        K=K,
    )
    ax_cap.plot(snr_axis, siso_cap, "k--", linewidth=1.5, label="SISO")
    ax_cap.plot(snr_axis, awgn_cap, "b-", linewidth=2, label="2x2 AWGN")
    ax_cap.plot(snr_axis, fading_cap, "r-", linewidth=2, label=f"2x2 {fading.capitalize()}")
    ax_cap.axvline(x=snr_db, color="gray", linestyle=":", alpha=0.7)
    ax_cap.set_xlabel("SNR (dB)")
    ax_cap.set_ylabel("Capacity (b/s/Hz)")
    ax_cap.set_title(f"Channel Capacity (ECC={ecc_median:.3f})", fontweight="bold")
    ax_cap.legend(fontsize=8)
    ax_cap.grid(True, alpha=0.3)

    # --- Center: Combining Gain Comparison ---
    ax_comb = fig.add_subplot(fig_gs[1])
    if len(gain_data_list) >= 2:
        theta_90_idx = np.argmin(np.abs(theta_deg - 90.0))
        n_phi = len(phi_deg)
        mrc_imp = np.zeros(n_phi)
        egc_imp = np.zeros(n_phi)
        sc_imp = np.zeros(n_phi)

        for i in range(n_phi):
            element_gains = [g[theta_90_idx, i] for g in gain_data_list]
            _, mrc_imp[i] = combining_gain(element_gains, method="mrc")
            _, egc_imp[i] = combining_gain(element_gains, method="egc")
            _, sc_imp[i] = combining_gain(element_gains, method="sc")

        ax_comb.plot(phi_deg, mrc_imp, "b-", linewidth=2, label="MRC")
        ax_comb.plot(phi_deg, egc_imp, "g--", linewidth=1.5, label="EGC")
        ax_comb.plot(phi_deg, sc_imp, "r:", linewidth=1.5, label="Selection")
        ax_comb.set_xlabel("Azimuth phi (deg)")
        ax_comb.set_ylabel("Combining Improvement (dB)")
        ax_comb.set_title("Combining Gain at theta=90 deg", fontweight="bold")
        ax_comb.legend(fontsize=8)
        ax_comb.grid(True, alpha=0.3)
    else:
        ax_comb.text(
            0.5,
            0.5,
            "Insufficient element patterns\n(need 2+ patterns)",
            ha="center",
            va="center",
            fontsize=12,
            transform=ax_comb.transAxes,
        )
        ax_comb.set_title("Combining Gain", fontweight="bold")

    # --- Right: Pattern Correlation (overlaid polar) ---
    ax_polar = fig.add_subplot(fig_gs[2], projection="polar")
    if len(gain_data_list) >= 2:
        phi_rad = np.deg2rad(phi_deg)
        theta_90_idx = np.argmin(np.abs(theta_deg - 90.0))

        colors_list = ["#4A90E2", "#E63946", "#4CAF50", "#FFC107"]
        for idx, g2d in enumerate(gain_data_list[:4]):
            color = colors_list[idx % len(colors_list)]
            pattern = g2d[theta_90_idx, :]
            pattern_norm = pattern - np.min(pattern)
            ax_polar.plot(phi_rad, pattern_norm, color=color, linewidth=1.5, label=f"Ant {idx + 1}")
        ax_polar.legend(loc="upper right", bbox_to_anchor=(1.3, 1.1), fontsize=8)
    else:
        ax_polar.text(
            0.5,
            0.5,
            "Insufficient element patterns\n(need 2+ patterns)",
            ha="center",
            va="center",
            fontsize=11,
            transform=ax_polar.transAxes,
        )

    ax_polar.set_title("Pattern Overlay (theta=90 deg)\nNormalized", fontweight="bold", pad=15)
    ax_polar.set_theta_zero_location("N")
    ax_polar.set_theta_direction(-1)

    if len(gain_data_list) > 0:
        meg_vals = mean_effective_gain_mimo(gain_data_list, theta_deg, phi_deg, xpr_db=xpr_db)
        meg_summary = ", ".join([f"A{i + 1}:{v:.1f} dB" for i, v in enumerate(meg_vals[:4])])
        ax_cap.annotate(
            f"MEG @ XPR={xpr_db:.1f} dB: {meg_summary}",
            xy=(0.02, 0.03),
            xycoords="axes fraction",
            fontsize=8,
            bbox=dict(facecolor="white", alpha=0.75, edgecolor="gray"),
        )

    fig.suptitle("MIMO / Diversity Analysis", fontsize=14, fontweight="bold")
    plt.tight_layout()

    if save_path:
        fname = "mimo_analysis.png"
        fig.savefig(os.path.join(save_path, fname), dpi=300, bbox_inches="tight")
        plt.close(fig)
    else:
        plt.show()


def plot_wearable_assessment(
    freq_mhz,
    gain_2d,
    theta_deg,
    phi_deg,
    body_positions=None,
    tx_power_mw=1.0,
    num_devices=20,
    room_size=(10, 10, 3),
    data_label="Gain",
    data_unit="dBi",
    save_path=None,
):
    """
    Wearable/medical device assessment: body position comparison,
    overlaid patterns, and dense device SINR analysis.
    """
    if body_positions is None:
        body_positions = ["wrist", "chest", "hip", "head"]

    # Body-worn analysis
    bw_results = body_worn_pattern_analysis(
        gain_2d,
        theta_deg,
        phi_deg,
        freq_mhz,
        body_positions,
    )

    # Dense device interference
    tx_dbm = 10 * np.log10(max(tx_power_mw, 0.001))
    avg_sinr, sinr_dist, noise_floor = dense_device_interference(
        num_devices,
        tx_dbm,
        freq_mhz,
        room_size_m=room_size,
    )

    # ---- Figure ----
    fig = plt.figure(figsize=(18, 7))
    fig_gs = fig.add_gridspec(1, 3, width_ratios=[1.3, 1, 1])

    # --- Left: Body position comparison table ---
    ax_table = fig.add_subplot(fig_gs[0])
    ax_table.axis("off")

    table_data = []
    for pos in body_positions:
        if pos not in bw_results:
            continue
        r = bw_results[pos]
        table_data.append(
            [
                pos.capitalize(),
                f"{r['avg_gain_db']:.1f} {data_unit}",
                f"{r['trp_delta_db']:+.1f} dB",
                f"{r['peak_delta_db']:+.1f} dB",
            ]
        )

    # Add free-space reference
    ref_lin = 10 ** (gain_2d / 10.0)
    sin_w = np.sin(np.deg2rad(theta_deg))
    ref_avg = 10 * np.log10(np.sum(ref_lin * sin_w[:, np.newaxis]) / (np.sum(sin_w) * len(phi_deg)))
    table_data.insert(0, ["Free Space", f"{ref_avg:.1f} {data_unit}", "ref", "ref"])

    table = ax_table.table(
        cellText=table_data,
        colLabels=["Position", f"Avg {data_label}", "TRP Δ", "Peak Δ"],
        cellLoc="center",
        loc="center",
        colWidths=[0.25, 0.25, 0.25, 0.25],
    )
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 1.6)

    for j in range(4):
        table[0, j].set_facecolor("#4A90E2")
        table[0, j].set_text_props(color="white", fontweight="bold")
    for j in range(4):
        table[1, j].set_facecolor("#E8F0FE")

    ax_table.set_title(
        f"Body-Worn Performance — {freq_mhz} MHz",
        fontsize=12,
        fontweight="bold",
        pad=20,
    )

    # --- Center: Overlaid polar patterns ---
    ax_polar = fig.add_subplot(fig_gs[1], projection="polar")
    phi_rad = np.deg2rad(phi_deg)
    theta_90_idx = np.argmin(np.abs(theta_deg - 90.0))

    fs_pattern = gain_2d[theta_90_idx, :]
    ax_polar.plot(phi_rad, fs_pattern - np.min(fs_pattern), "k-", linewidth=2, label="Free Space")

    colors_pos = {
        "wrist": "#4A90E2",
        "chest": "#E63946",
        "hip": "#4CAF50",
        "head": "#FFC107",
    }
    for pos in body_positions:
        if pos not in bw_results:
            continue
        pattern = bw_results[pos]["pattern"][theta_90_idx, :]
        ax_polar.plot(
            phi_rad,
            pattern - np.min(fs_pattern),
            color=colors_pos.get(pos, "gray"),
            linewidth=1.5,
            label=pos.capitalize(),
        )

    ax_polar.set_title("Pattern at θ=90°\n(Normalized)", fontweight="bold", pad=15)
    ax_polar.set_theta_zero_location("N")
    ax_polar.set_theta_direction(-1)
    ax_polar.legend(loc="upper right", bbox_to_anchor=(1.35, 1.15), fontsize=8)

    # --- Right: Dense device SINR distribution ---
    ax_sinr = fig.add_subplot(fig_gs[2])
    ax_sinr.hist(sinr_dist, bins=30, color="#4A90E2", edgecolor="white", alpha=0.8, density=True)
    ax_sinr.axvline(
        x=avg_sinr, color="red", linestyle="--", linewidth=2, label=f"Mean SINR: {avg_sinr:.1f} dB"
    )
    ax_sinr.axvline(x=0, color="gray", linestyle=":", linewidth=1, label="0 dB (breakeven)")
    ax_sinr.set_xlabel("SINR (dB)")
    ax_sinr.set_ylabel("Probability Density")
    ax_sinr.set_title(
        f"Dense Device Coexistence\n{num_devices} devices in "
        f"{room_size[0]}×{room_size[1]}×{room_size[2]}m",
        fontweight="bold",
    )
    ax_sinr.legend(fontsize=8)
    ax_sinr.grid(True, alpha=0.3)
    ax_sinr.annotate(
        f"Noise floor: {noise_floor:.0f} dBm",
        xy=(0.02, 0.95),
        xycoords="axes fraction",
        fontsize=8,
        bbox=dict(facecolor="white", edgecolor="gray", alpha=0.8),
    )

    plt.tight_layout()

    if save_path:
        fname = f"wearable_assessment_{freq_mhz}MHz.png"
        fig.savefig(os.path.join(save_path, fname), dpi=300, bbox_inches="tight")
        plt.close(fig)
    else:
        plt.show()
