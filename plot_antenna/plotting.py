import os
import warnings
from tkinter import messagebox, simpledialog

import matplotlib

try:
    matplotlib.use("TkAgg")  # noqa: E402 — must precede pyplot import
except ImportError:
    # No display available (e.g. headless CI test collection) — fall back
    # to a non-interactive backend instead of crashing on import.
    matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

plt.ion()  # Non-blocking show() — avoids "main thread is not in main loop" with Tkinter GUI
from matplotlib import cm
from matplotlib.colors import Normalize
import numpy as np
import scipy.interpolate as spi

from .config import (
    THETA_RESOLUTION,
    PHI_RESOLUTION,
    polar_dB_max,
    polar_dB_min,
)

# Shared plot style + save helpers (#41) — single source of truth for output
# dpi/format and a small colormap registry, so styling isn't duplicated across
# the many plot_* functions. Existing call sites migrate to save_figure()
# incrementally; new plots should use it.
PLOT_DPI = 300
PLOT_FORMAT = "png"

_COLORMAPS = {
    "gain": "viridis",
    "power": "turbo",
    "phase": "twilight",
    "error": "RdYlGn_r",
    "diverging": "RdBu_r",
}


def colormap_for(kind: str):
    """Return the registered colormap name for a plot kind (default: viridis)."""
    return _COLORMAPS.get(kind, "viridis")


def save_figure(fig_or_path, path=None, *, dpi=PLOT_DPI, fmt=PLOT_FORMAT):
    """Save a figure with the project-standard dpi/format (#41).

    Usage: save_figure(fig, path) or save_figure(path) to save the current
    pyplot figure. Centralizes the dpi=300 / format="png" convention.
    """
    if path is None:
        plt.savefig(fig_or_path, format=fmt, dpi=dpi)
        return fig_or_path
    fig_or_path.savefig(path, format=fmt, dpi=dpi)
    return path


from .file_utils import parse_2port_data
from .calculations import (
    calculate_trp,
    calculate_partial_trp,
    calculate_spherical_band_statistics,
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

# Suppress noisy warnings during batch processing worker threads.
warnings.filterwarnings("ignore", message="Starting a Matplotlib GUI outside of the main thread")


# _____________Active Plotting Functions___________
# Function called by gui.py to start the active plotting procedure after parsing and calculating variables
def plot_active_2d_data(
    data_points, theta_angles_rad, phi_angles_rad, total_power_dBm_2d, frequency, save_path=None
):
    # Plot Azimuth cuts for different theta values on one plot
    theta_values_to_plot = [0, 15, 30, 45, 60, 75, 90, 105, 120, 135, 150, 165]

    # Save 2D Azimuth Power Cuts with naming consistency if save_path is provided
    if save_path:
        plot_2d_azimuth_power_cuts(
            phi_angles_rad,
            theta_angles_rad,
            total_power_dBm_2d,
            theta_values_to_plot,
            frequency,
            save_path=save_path,
        )
        plot_additional_active_polar_plots(
            phi_angles_rad, theta_angles_rad, total_power_dBm_2d, frequency, save_path=save_path
        )
    else:
        plot_2d_azimuth_power_cuts(
            phi_angles_rad, theta_angles_rad, total_power_dBm_2d, theta_values_to_plot, frequency
        )
        plot_additional_active_polar_plots(
            phi_angles_rad, theta_angles_rad, total_power_dBm_2d, frequency
        )

    return


# Plot 2D Azimuth Power Cuts vs Phi for various value of theta to plot
def plot_2d_azimuth_power_cuts(
    phi_angles_rad, theta_angles_rad, power_dBm_2d, theta_values_to_plot, frequency, save_path=None
):
    """
    Plot azimuth power pattern cuts for specified theta values.

    Parameters:
    - theta_angles_rad: 1D array of theta angles in radians.
    - phi_angles_rad: 1D array of phi angles in radians.
    - power_dBm_2d: 2D array of power values, shape should be (num_theta, num_phi).
    - theta_values_to_plot_deg: List of theta angles in degrees for which to plot azimuth cuts.
    """
    plt.figure(figsize=(10, 6))
    ax = plt.subplot(111, projection="polar")

    for theta_deg in theta_values_to_plot:
        # Find the closest index to the desired theta value
        theta_idx = np.argmin(np.abs(np.rad2deg(theta_angles_rad) - theta_deg))

        # Extract the corresponding phi and power values
        phi_values = phi_angles_rad
        power_values = power_dBm_2d[theta_idx, :]

        # Append the first phi and power value to the end to wrap the data
        phi_values = np.append(phi_values, phi_values[0])
        power_values = np.append(power_values, power_values[0])

        ax.plot(phi_values, power_values, label=f"Theta = {theta_deg} deg")

    # Gain Summary
    max_power = np.max(power_dBm_2d)
    min_power = np.min(power_dBm_2d)
    # Average Gain Summary
    power_mW = 10 ** (power_dBm_2d / 10)
    avg_power_mW = np.mean(power_mW)
    avg_power_dBm = 10 * np.log10(avg_power_mW)

    # Add a text box or similar to your plot to display these values. For example:
    textstr = f"Power Summary at {frequency} (MHz): Max: {max_power:.1f} (dBm) Min: {min_power:.1f} (dBm) Avg: {avg_power_dBm:.1f} (dBm)"
    ax.annotate(
        textstr,
        xy=(-0.1, -0.1),
        xycoords="axes fraction",
        fontsize=10,
        bbox=dict(facecolor="none", edgecolor="black", boxstyle="round,pad=0.5"),
    )

    ax.set_title("Azimuth Power Pattern Cuts vs. Phi")
    ax.legend(loc="upper right", bbox_to_anchor=(1.5, 1))
    ax.grid(True, which="both", linestyle="--", linewidth=0.5)

    if save_path:
        # Save the plot with appropriate naming
        azimuth_plot_path = os.path.join(save_path, f"2D_Azimuth_Cuts_{frequency}_MHz.png")
        plt.savefig(azimuth_plot_path, format="png", dpi=300)
        plt.close()  # Close the plot after saving
    else:
        plt.show()


# Plot Additional Elevation and Azimuth cuts for the 3-planes Theta=90deg, Phi=0deg/180deg, and Phi=90deg/270deg
def plot_additional_active_polar_plots(
    phi_angles_rad, theta_angles_rad, total_power_dBm_2d, frequency, save_path=None
):
    # Sanitize the frequency string to avoid special characters
    sanitized_frequency = str(frequency).replace("/", "_").replace("=", "_").replace(":", "_")

    # Azimuth Power Pattern at Theta = 90deg
    idx_theta_90 = np.argwhere(np.isclose(theta_angles_rad, np.pi / 2, atol=1e-6)).flatten()
    if idx_theta_90.size > 0:
        max_power = np.max(total_power_dBm_2d[idx_theta_90, :])
        min_power = np.min(total_power_dBm_2d[idx_theta_90, :])
        avg_power = 10 * np.log10(np.mean(10 ** (total_power_dBm_2d[idx_theta_90, :] / 10)))

        title = f"Total Power in Theta = 90deg Plane, Freq = {sanitized_frequency}MHz"
        plot_polar_power_pattern(
            title,
            phi_angles_rad,
            total_power_dBm_2d[idx_theta_90, :].flatten(),
            "azimuth",
            max_power,
            min_power,
            avg_power,
            save_path=save_path,
        )

    # Elevation Power Pattern in the Phi = 270deg/90deg plane
    # Note: Variable names match the angle they represent
    idx_phi_270 = np.argwhere(
        np.isclose(phi_angles_rad, 3 * np.pi / 2, atol=1e-6)
    ).flatten()  # 270° = 3π/2
    idx_phi_90 = np.argwhere(
        np.isclose(phi_angles_rad, np.pi / 2, atol=1e-6)
    ).flatten()  # 90° = π/2

    if idx_phi_90.size > 0 and idx_phi_270.size > 0:
        theta_phi_270 = theta_angles_rad
        power_phi_270 = total_power_dBm_2d[:, idx_phi_270].flatten()

        theta_phi_90 = 2 * np.pi - theta_angles_rad
        power_phi_90 = total_power_dBm_2d[:, idx_phi_90].flatten()

        power_combined_calc = np.concatenate((power_phi_90, power_phi_270))
        max_power = np.max(power_combined_calc)
        min_power = np.min(power_combined_calc)
        avg_power = 10 * np.log10(np.mean(10 ** (power_combined_calc / 10)))

        theta_combined = np.concatenate((theta_phi_90, [np.nan], theta_phi_270))
        power_combined = np.concatenate((power_phi_90, [np.nan], power_phi_270))

        annotations = [
            {"text": "Phi=90", "xy": (3 * np.pi / 2, max_power), "xytext": (0, 20)},
            {"text": "Phi=270", "xy": (np.pi / 2, max_power), "xytext": (0, 20)},
        ]

        title = f"Elevation Power Pattern at Phi = 270/90deg, Freq = {frequency}MHz"
        plot_polar_power_pattern(
            title,
            theta_combined,
            power_combined,
            "elevation",
            max_power,
            min_power,
            avg_power,
            annotations=annotations,
            save_path=save_path,
        )

    # Elevation Power Pattern in the Phi = 180deg/0deg plane
    # Note: Variable names match the angle they represent
    idx_phi_180 = np.argwhere(np.isclose(phi_angles_rad, np.pi, atol=1e-6)).flatten()  # 180° = π
    idx_phi_0 = np.argwhere(np.isclose(phi_angles_rad, 0, atol=1e-6)).flatten()  # 0° = 0

    if idx_phi_0.size > 0 and idx_phi_180.size > 0:
        theta_phi_180 = theta_angles_rad
        power_phi_180 = total_power_dBm_2d[:, idx_phi_180].flatten()

        theta_phi_0 = 2 * np.pi - theta_angles_rad
        power_phi_0 = total_power_dBm_2d[:, idx_phi_0].flatten()

        power_combined_calc = np.concatenate((power_phi_0, power_phi_180))
        max_power = np.max(power_combined_calc)
        min_power = np.min(power_combined_calc)
        avg_power = 10 * np.log10(np.mean(10 ** (power_combined_calc / 10)))

        theta_combined = np.concatenate((theta_phi_0, [np.nan], theta_phi_180))
        power_combined = np.concatenate((power_phi_0, [np.nan], power_phi_180))

        annotations = [
            {"text": "Phi=0", "xy": (3 * np.pi / 2, max_power), "xytext": (0, 20)},
            {"text": "Phi=180", "xy": (np.pi / 2, max_power), "xytext": (0, 20)},
        ]

        title = f"Elevation Power Pattern at Phi = 180/0deg, Freq = {frequency}MHz"
        plot_polar_power_pattern(
            title,
            theta_combined,
            power_combined,
            "elevation",
            max_power,
            min_power,
            avg_power,
            annotations=annotations,
            save_path=save_path,
        )


# General Polar Plotting Function for Active Plots
def plot_polar_power_pattern(
    title,
    angles_rad,
    power_dBm,
    plane,
    max_power,
    min_power,
    avg_power,
    summary_position="bottom",
    annotations=None,
    save_path=None,
):
    fig, ax = plt.subplots(subplot_kw={"projection": "polar"})

    if plane == "azimuth":
        # Append first data point to end for continuous plot
        angles_rad = np.concatenate((angles_rad, [angles_rad[0]]))
        power_dBm = np.concatenate((power_dBm, [power_dBm[0]]))
        ax.set_theta_zero_location("E")  # type: ignore
        ax.set_theta_direction(1)  # type: ignore
        ax.set_rlabel_position(90)  # type: ignore
        ax.set_xticks(np.radians([0, 30, 60, 90, 120, 150, 180, 210, 240, 270, 300, 330]))
        ax.set_xticklabels(
            [
                "0°",
                "30°",
                "60°",
                "90°",
                "120°",
                "150°",
                "180°",
                "210°",
                "240°",
                "270°",
                "300°",
                "330°",
            ]
        )
    elif plane == "elevation":
        ax.set_theta_zero_location("N")  # type: ignore
        ax.set_theta_direction(1)  # type: ignore
        ax.set_rlabel_position(0)  # type: ignore
        ax.set_xticks(np.radians([0, 30, 60, 90, 120, 150, 180, 210, 240, 270, 300, 330]))
        ax.set_xticklabels(
            ["0°", "30°", "60°", "90°", "120°", "150°", "180°", "150°", "120°", "90°", "60°", "30°"]
        )

    ax.plot(angles_rad, power_dBm, label=title)
    ax.grid(True, linestyle="--", linewidth=0.5)

    # Set labels and title
    ax.set_title(title, va="bottom")

    summary_text = (
        f"Max: {max_power:.1f} (dBm) Min: {min_power:.1f} (dBm) Avg: {avg_power:.1f} (dBm)"
    )

    if summary_position == "bottom":
        ax.annotate(
            summary_text,
            xy=(0.5, -0.11),
            xycoords="axes fraction",
            ha="center",
            va="center",
            bbox=dict(boxstyle="round", alpha=0.1),
        )
    elif summary_position == "top":
        ax.annotate(
            summary_text,
            xy=(0.5, 1.15),
            xycoords="axes fraction",
            ha="center",
            va="center",
            bbox=dict(boxstyle="round", alpha=0.1),
        )

    # Add annotations if provided
    if annotations:
        for annotation in annotations:
            ax.annotate(
                annotation["text"],
                xy=annotation["xy"],
                xycoords="data",
                textcoords="offset points",
                xytext=annotation["xytext"],
                ha="center",
                fontsize=12,
                bbox=dict(boxstyle="round", alpha=0.5, facecolor="white"),
            )

    if save_path:
        # Sanitize the title string to avoid special characters in file names
        sanitized_title = (
            title.replace(" ", "_").replace("=", "_").replace(":", "_").replace("/", "_")
        )
        plot_path = os.path.join(save_path, f"{sanitized_title}.png")
        plt.savefig(plot_path, format="png", dpi=300)
        plt.close(fig)
    else:
        plt.show()


def _setup_3d_axes(ax, X, Y, Z):
    """Configure a 3D Axes for antenna pattern display.

    Professional-style 3-D antenna gain plot with DUT orientation:

    * **Equal aspect ratio** via ``set_box_aspect([1,1,1])`` — no
      axis stretching.
    * **Symmetric limits** centred on the origin so the pattern sits
      in the middle of the bounding box.
    * **Box-edge axis labels** via ``set_xlabel / ylabel / zlabel``
      — rendered in the 2D overlay layer, never occluded.
    * **Short origin arrows** that mirror the physical orientation
      marker used in the anechoic chamber (X green, Y red, Z blue).
      They extend from the *negative* side of each axis toward the
      origin so they sit in the empty space behind the pattern and
      remain visible regardless of view angle.

    Parameters
    ----------
    ax : mpl_toolkits.mplot3d.axes3d.Axes3D
        The 3-D axes to decorate.
    X, Y, Z : ndarray
        Cartesian coordinate arrays of the plotted surface.
    """
    # ---- Data extent (finite values only) ----
    _ext = np.abs(np.concatenate([np.ravel(X), np.ravel(Y), np.ravel(Z)]))
    _ext = _ext[np.isfinite(_ext)]
    max_ext = float(_ext.max()) if _ext.size else 0.0
    # Guard degenerate extents (all-NaN or flat/zero pattern) so the bounding
    # box never collapses to lim=0 / lim=NaN, which would blank the axes.
    if not np.isfinite(max_ext) or max_ext <= 0:
        max_ext = 1.0

    # ---- Symmetric limits — centres the pattern in the box ----
    lim = 1.02 * max_ext
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    ax.set_zlim(-lim, lim)
    ax.set_box_aspect([1, 1, 1])

    # ---- Transparent panes ----
    ax.xaxis.pane.fill = False  # type: ignore
    ax.yaxis.pane.fill = False  # type: ignore
    ax.zaxis.pane.fill = False  # type: ignore
    ax.xaxis.pane.set_edgecolor("gray")  # type: ignore
    ax.yaxis.pane.set_edgecolor("gray")  # type: ignore
    ax.zaxis.pane.set_edgecolor("gray")  # type: ignore
    ax.xaxis.pane.set_alpha(0.2)  # type: ignore
    ax.yaxis.pane.set_alpha(0.2)  # type: ignore
    ax.zaxis.pane.set_alpha(0.2)  # type: ignore

    # ---- Grid + hide tick labels ----
    ax.grid(True)
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.set_zticklabels([])  # type: ignore

    # ---- Box-edge axis labels (2D overlay — never occluded) ----
    _label_kw = dict(fontsize=14, fontweight="bold", labelpad=2)
    ax.set_xlabel("X", color="green", **_label_kw)
    ax.set_ylabel("Y", color="red", **_label_kw)
    ax.set_zlabel("Z", color="blue", **_label_kw)  # type: ignore

    # ---- DUT orientation triad ----
    # A three-arrow tripod anchored at the (-lim, -lim, -lim) corner,
    # pointing in +X, +Y, +Z — mirrors the physical orientation
    # marker placed on the DUT in the anechoic chamber.
    # All three arrows share the same origin so they form a
    # recognisable XYZ triad from every viewing angle.
    corner = -lim  # common origin at the negative corner
    arrow_len = 0.35 * lim  # short — just an orientation hint

    _arrow_spec = [
        # (dx, dy, dz, colour)
        (1, 0, 0, "green"),  # +X
        (0, 1, 0, "red"),  # +Y
        (0, 0, 1, "blue"),  # +Z
    ]
    for dx, dy, dz, colour in _arrow_spec:
        ax.quiver(
            corner,
            corner,
            corner,  # all start at the same corner
            dx * arrow_len,
            dy * arrow_len,
            dz * arrow_len,
            color=colour,
            arrow_length_ratio=0.3,
            linewidth=2.5,
            alpha=0.9,
        )


def plot_active_3d_data(
    theta_angles_deg,
    phi_angles_deg,
    total_power_dBm_2d,
    phi_angles_deg_plot,
    total_power_dBm_2d_plot,
    frequency,
    power_type="total",
    interpolate=True,
    axis_mode="auto",
    zmin: float = -15.0,
    zmax: float = 15.0,
    save_path=None,
):
    """
    Plot a 3D representation of the active data (TRP).
    Added axis_mode, zmin, zmax to allow manual z-axis scaling.

    Parameters:
    - theta_angles_deg (array): Theta angles in degrees.
    - phi_angles_deg (array): Phi angles in degrees.
    - total_power_dBm_2d (array): 2D array of power values in dBm.
    - frequency (float): Frequency in MHz.
    - power_type (str): Type of power ('total', 'hpol', 'vpol'). Default is 'total'.
    - interpolate (bool): Whether to interpolate the data for smoother visualization.
    - save_path (str): Path to save the plots. If None, the plot will be displayed.
    """
    # Ensure theta and phi angles are 1D arrays
    theta_angles_deg = np.squeeze(theta_angles_deg)
    phi_angles_deg = np.squeeze(phi_angles_deg)

    # Ensure the Max EIRP is calculated before interpolation
    max_eirp_dBm = np.max(total_power_dBm_2d)

    # TRP Calculation using configurable formula from config
    theta_angles_rad = np.deg2rad(theta_angles_deg)
    phi_angles_rad = np.deg2rad(phi_angles_deg)

    # Calculate angular increments
    if len(theta_angles_deg) > 1:
        inc_theta = theta_angles_deg[1] - theta_angles_deg[0]
    else:
        inc_theta = 15.0  # Default increment
    if len(phi_angles_deg) > 1:
        inc_phi = phi_angles_deg[1] - phi_angles_deg[0]
    else:
        inc_phi = 15.0  # Default increment

    # Calculate TRP using IEEE solid angle integration
    TRP_dBm = calculate_trp(total_power_dBm_2d, theta_angles_rad, inc_theta, inc_phi)

    if interpolate:
        # Create meshgrid of theta and phi angles
        THETA_deg, PHI_deg = np.meshgrid(theta_angles_deg, phi_angles_deg, indexing="ij")

        # Flatten the data for interpolation
        theta_flat = THETA_deg.flatten()
        phi_flat = PHI_deg.flatten()
        power_flat = total_power_dBm_2d.flatten()

        # Check for NaN values in the data
        if np.isnan(power_flat).any():
            print("Warning: NaN values found in power_flat data.")

        # Interpolate the data onto the render grid.
        data_interp, theta_interp, phi_interp = process_data(power_flat, phi_flat, theta_flat)

        power_data = data_interp

        # Check if power_data contains NaNs after interpolation
        if np.isnan(power_data).all():
            print("Error: Interpolated power_data contains all NaN values.")
            return

        # Normalize the power data
        max_power_value = np.max(power_data)
        min_power_value = np.min(power_data)
        denominator = max_power_value - min_power_value
        if denominator == 0:
            print("Warning: max_power_value equals min_power_value. Normalizing data to zeros.")
            power_normalized = np.zeros_like(power_data)
        else:
            power_normalized = (power_data - min_power_value) / denominator

        # Convert to spherical coordinates using normalized values
        PHI, THETA = np.meshgrid(phi_interp, theta_interp)
        theta_rad = np.deg2rad(THETA)
        phi_rad = np.deg2rad(PHI)
        R = 0.75 * power_normalized
        X = R * np.sin(theta_rad) * np.cos(phi_rad)
        Y = R * np.sin(theta_rad) * np.sin(phi_rad)
        Z = R * np.cos(theta_rad)

    else:
        # Use the raw data without interpolation
        # Create a meshgrid of theta and phi angles
        THETA_deg, PHI_deg = np.meshgrid(theta_angles_deg, phi_angles_deg, indexing="ij")
        PHI_rad = np.deg2rad(PHI_deg)
        THETA_rad = np.deg2rad(THETA_deg)

        # Ensure total_power_dBm_2d has correct shape
        if total_power_dBm_2d.shape != THETA_deg.shape:
            total_power_dBm_2d = total_power_dBm_2d.T

        power_data = total_power_dBm_2d

        # Normalize the power data
        max_power_value = np.nanmax(power_data)
        min_power_value = np.nanmin(power_data)
        denominator = max_power_value - min_power_value
        if denominator == 0:
            print("Warning: max_power_value equals min_power_value. Normalizing data to zeros.")
            power_normalized = np.zeros_like(power_data)
        else:
            power_normalized = (power_data - min_power_value) / denominator
        power_normalized = np.nan_to_num(power_normalized)

        # Adjust the radius for visualization
        R_normalized = 0.75 * power_normalized

        # Convert to Cartesian coordinates
        X = R_normalized * np.sin(THETA_rad) * np.cos(PHI_rad)
        Y = R_normalized * np.sin(THETA_rad) * np.sin(PHI_rad)
        Z = R_normalized * np.cos(THETA_rad)

        # For consistency, define theta_interp and phi_interp
        theta_interp = theta_angles_deg
        phi_interp = phi_angles_deg
        data_interp = power_data  # Use power_data for consistency

    # Now plot (use data_interp which is defined in both branches)
    plt.style.use("default")
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(1, 1, 1, projection="3d")
    # Expand the 3D axes to fill more of the figure
    fig.subplots_adjust(left=0.0, right=0.88, bottom=0.0, top=0.92)

    # Color‐mapping alone obeys manual limits
    if axis_mode == "manual":
        norm = Normalize(zmin, zmax)
    else:
        norm = Normalize(data_interp.min(), data_interp.max())
    mappable = cm.ScalarMappable(norm=norm, cmap=cm.turbo)  # type: ignore
    mappable.set_array(data_interp)

    surf = ax.plot_surface(
        X,
        Y,
        Z,
        facecolors=cm.turbo(norm(data_interp)),  # type: ignore
        linewidth=0.5,
        antialiased=True,
        shade=False,
        rstride=1,
        cstride=1,
        zorder=1,
    )
    # Set the view angle
    ax.view_init(elev=20, azim=-30)

    # Configure axes: equal aspect, panes, grid, arrows
    _setup_3d_axes(ax, X, Y, Z)

    # Set Title based on power_type with rounded TRP values
    if power_type == "total":
        plot_title = f"Total Radiated Power at {frequency} MHz, TRP = {TRP_dBm:.2f} dBm, Max EIRP = {max_eirp_dBm:.2f} dBm"
    elif power_type == "hpol":
        plot_title = f"Phi Polarization: Radiated Power at {frequency} MHz, TRP = {TRP_dBm:.2f} dBm"
    elif power_type == "vpol":
        plot_title = (
            f"Theta Polarization: Radiated Power at {frequency} MHz, TRP = {TRP_dBm:.2f} dBm"
        )
    else:
        plot_title = (
            f"3D Radiation Pattern - {power_type} at {frequency} MHz, TRP = {TRP_dBm:.2f} dBm"
        )
    fig.suptitle(plot_title, fontsize=14, y=0.97)

    # Add a colorbar
    cbar = fig.colorbar(mappable, ax=ax, pad=0.08, shrink=0.65)
    cbar.set_label("Power (dBm)", rotation=270, labelpad=20, fontsize=12)
    cbar.ax.tick_params(labelsize=10)

    # Add Max Power to top of colorbar
    cbar.ax.set_title(f"{max_eirp_dBm:.2f} dBm", fontsize=10, weight="bold", pad=4)

    # If save path provided, save the plot
    if save_path:
        # Save the first view
        plot_3d_path_1 = os.path.join(save_path, f"3D_TRP_{power_type}_{frequency}MHz_1of2.png")
        fig.savefig(plot_3d_path_1, format="png", dpi=300)

        # Adjust view angle to get the rear side of the 3D plot
        ax.view_init(elev=20, azim=150)

        # Save the second view
        plot_3d_path_2 = os.path.join(save_path, f"3D_TRP_{power_type}_{frequency}MHz_2of2.png")
        fig.savefig(plot_3d_path_2, format="png", dpi=300)

        plt.close(fig)
    else:
        plt.show()


# _____________Passive Plotting Functions___________
def plot_data(data, title, x_label, y_label, legend_labels=None, x_data=None):
    """
    A generic function to plot data.

    Parameters:
    - data: list of data arrays to plot.
    - title: title of the plot.
    - x_label: label for the x-axis.
    - y_label: label for the y-axis.
    - legend_labels: labels for the legend (default is None).

    Returns:
    - fig: a matplotlib figure object.
    """
    fig, ax = plt.subplots()
    for d in data:
        if x_data:
            ax.plot(x_data, d)
        else:
            ax.plot(d)
    ax.set_title(title)
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    if legend_labels:
        ax.legend(legend_labels)
    return fig


# Plot passive data
def plot_2d_passive_data(
    theta_angles_deg,
    phi_angles_deg,
    v_gain_dB,
    h_gain_dB,
    Total_Gain_dB,
    freq_list,
    selected_frequency,
    datasheet_plots,
    save_path=None,
):
    """
    Plot 2D passive data for the given parameters.
    Saves efficiency and total gain plots using unique filenames containing
    the minimum and maximum frequencies to avoid overwriting.
    """
    # Convert angles from degrees to radians
    plot_theta_rad = np.radians(theta_angles_deg)
    plot_phi_rad = np.radians(phi_angles_deg)

    # Calculate angular increments for proper solid angle integration
    # Get unique theta and phi values to determine increments
    unique_theta = (
        np.unique(theta_angles_deg[:, 0])
        if theta_angles_deg.ndim > 1
        else np.unique(theta_angles_deg)
    )
    unique_phi = (
        np.unique(phi_angles_deg[:, 0]) if phi_angles_deg.ndim > 1 else np.unique(phi_angles_deg)
    )

    if len(unique_theta) > 1:
        dtheta = np.deg2rad(unique_theta[1] - unique_theta[0])
    else:
        dtheta = np.deg2rad(15.0)  # Default
    if len(unique_phi) > 1:
        dphi = np.deg2rad(unique_phi[1] - unique_phi[0])
    else:
        dphi = np.deg2rad(15.0)  # Default

    # Calculate Average Gain per Frequency using proper spherical integration
    # Average = (1/4π) ∫∫ G(θ,φ) · sin(θ) dθ dφ
    # For antenna efficiency, this gives the average gain over the sphere
    gain_linear = 10 ** (Total_Gain_dB / 10)
    sin_theta = np.sin(plot_theta_rad)

    # Sum with proper solid angle weighting
    weighted_sum = np.sum(gain_linear * sin_theta, axis=0) * dtheta * dphi
    Average_Gain_dB = 10 * np.log10(np.maximum(weighted_sum / (4 * np.pi), 1e-12))

    # Determine the frequency range for naming outputs
    # Cast to int for clean file names; fall back to first/last if not numeric
    try:
        min_freq = int(min(freq_list))
        max_freq = int(max(freq_list))
    except Exception:
        min_freq = freq_list[0]
        max_freq = freq_list[-1]

    # ---- Efficiency (dB) plot ----
    fig = plot_data(
        [Average_Gain_dB],
        "Average Radiated Efficiency Versus Frequency (dB)",
        "Frequency (MHz)",
        "Efficiency (dB)",
        x_data=freq_list,
    )
    fig.gca().grid(True, which="both", linestyle="--", linewidth=0.5)
    if save_path:
        eff_db_path = os.path.join(save_path, f"efficiency_db_{min_freq}-{max_freq}MHz.png")
        fig.savefig(eff_db_path, format="png", dpi=300)
        plt.close(fig)
    else:
        plt.show()

    # ---- Efficiency (%) plot ----
    Average_Gain_percentage = 100 * 10 ** (Average_Gain_dB / 10)
    fig = plot_data(
        [Average_Gain_percentage],
        "Average Radiated Efficiency Versus Frequency (%)",
        "Frequency (MHz)",
        "Efficiency (%)",
        x_data=freq_list,
    )
    fig.gca().grid(True, which="both", linestyle="--", linewidth=0.5)
    if save_path:
        eff_percent_path = os.path.join(
            save_path, f"efficiency_percent_{min_freq}-{max_freq}MHz.png"
        )
        fig.savefig(eff_percent_path, format="png", dpi=300)
        plt.close(fig)
    else:
        plt.show()

    # ---- Peak Gain plot ----
    Peak_Gain_dB = np.max(Total_Gain_dB, axis=0)
    fig = plot_data(
        [Peak_Gain_dB],
        "Total Gain Versus Frequency",
        "Frequency (MHz)",
        "Peak Gain (dBi)",
        x_data=freq_list,
    )
    fig.gca().grid(True, which="both", linestyle="--", linewidth=0.5)
    if save_path:
        total_gain_path = os.path.join(save_path, f"total_gain_{min_freq}-{max_freq}MHz.png")
        fig.savefig(total_gain_path, format="png", dpi=300)
        plt.close(fig)
    else:
        plt.show()

    # If datasheet plots setting is selected in the settings menu
    if datasheet_plots:
        # Calculate the maximum gain across all angles for each frequency
        max_phi_gain_dB = np.max(h_gain_dB, axis=0)
        max_theta_gain_dB = np.max(v_gain_dB, axis=0)

        # Plot Phi Gain
        fig_phi = plot_data(
            [max_phi_gain_dB],
            "Phi Gain Versus Frequency",
            "Frequency (MHz)",
            "Phi Gain (dBi)",
            x_data=freq_list,
        )
        fig_phi.gca().grid(True, which="both", linestyle="--", linewidth=0.5)

        # Plot Theta Gain
        fig_theta = plot_data(
            [max_theta_gain_dB],
            "Theta Gain Versus Frequency",
            "Frequency (MHz)",
            "Theta Gain (dBi)",
            x_data=freq_list,
        )
        fig_theta.gca().grid(True, which="both", linestyle="--", linewidth=0.5)

        num_bands = simpledialog.askinteger("Input", "Enter the number of bands:")
        if not num_bands:
            return
        bands = []
        for i in range(num_bands):
            band_range = simpledialog.askstring(
                "Input", f"Enter the frequency range for band {i+1} (e.g., 2400,2480):"
            )
            if not band_range:
                messagebox.showerror("Error", "No frequency range entered.")
                return
            bands.append(list(map(float, band_range.split(","))))

        # Convert freq_list to a NumPy array
        freq_array = np.array(freq_list)

        annotation_height = 10  # Initial offset for annotations
        for i, band in enumerate(bands):
            start_freq, end_freq = band
            indices = np.where((freq_array >= start_freq) & (freq_array <= end_freq))

            if len(indices[0]) == 0:
                print(f"No data points found between {start_freq} MHz and {end_freq} MHz.")
                continue  # Skip to the next band if no data points are found in this band

            max_index_phi = np.argmax(h_gain_dB[:, indices], axis=None)
            max_index_theta = np.argmax(v_gain_dB[:, indices], axis=None)

            peak_band_phi_gain = h_gain_dB[:, indices].flatten()[max_index_phi]
            peak_band_theta_gain = v_gain_dB[:, indices].flatten()[max_index_theta]

            # Annotate the max gain in the band for each polarization in the summary
            fig_phi.gca().annotate(
                f"Band {start_freq}-{end_freq} MHz: Max Phi Gain: {peak_band_phi_gain:.2f} dBi",
                xy=(0.5, 0),
                xycoords="axes fraction",
                fontsize=10,
                xytext=(0, annotation_height),
                textcoords="offset points",
                ha="center",
                va="bottom",
                bbox=dict(boxstyle="round,pad=0.5", facecolor="white", alpha=0.7),
            )
            fig_theta.gca().annotate(
                f"Band {start_freq}-{end_freq} MHz: Max Theta Gain: {peak_band_theta_gain:.2f} dBi",
                xy=(0.5, 0),
                xycoords="axes fraction",
                fontsize=10,
                xytext=(0, annotation_height),
                textcoords="offset points",
                ha="center",
                va="bottom",
                bbox=dict(boxstyle="round,pad=0.5", facecolor="white", alpha=0.7),
            )
            annotation_height += 20  # Increase offset for each annotation

        # If save path specified, save otherwise show
        if save_path:
            fig_phi.savefig(os.path.join(save_path, "phi_gain.png"), format="png", dpi=300)
            fig_theta.savefig(os.path.join(save_path, "theta_gain.png"), format="png", dpi=300)
            plt.close(fig_phi)
            plt.close(fig_theta)
        else:
            plt.show()

    # Plot Azimuth cuts for different theta values
    if selected_frequency in freq_list:
        freq_idx = np.where(np.array(freq_list) == selected_frequency)[0][0]
    else:
        print(f"Error: Selected frequency {selected_frequency} not found in the frequency list.")
        return

    selected_azimuth_freq = Total_Gain_dB[:, freq_idx]
    plot_phi_rad = plot_phi_rad[:, freq_idx]

    theta_values_to_plot = [0, 15, 30, 45, 60, 75, 90, 105, 120, 135, 150, 165]

    plt.figure(figsize=(10, 6))
    ax = plt.subplot(111, projection="polar")

    for theta in theta_values_to_plot:
        mask = np.abs(theta_angles_deg[:, freq_idx] - theta) < 0.01
        if np.any(mask):
            phi_values = plot_phi_rad[mask]
            gain_values = selected_azimuth_freq[mask]

            # Append the first phi and gain value to the end to wrap the data
            phi_values = np.append(phi_values, phi_values[0])
            gain_values = np.append(gain_values, gain_values[0])

            ax.plot(phi_values, gain_values, label=f"Theta {theta}°")

    ax.set_title(f"Gain Pattern Azimuth Cuts - Total Gain at {selected_frequency} MHz")
    ax.legend(loc="upper right", bbox_to_anchor=(1.4, 1))
    ax.grid(True, which="both", linestyle="--", linewidth=0.5)

    # Gain Summary
    max_gain = np.max(selected_azimuth_freq)
    min_gain = np.min(selected_azimuth_freq)
    # Average Gain Summary
    power_mW = 10 ** (selected_azimuth_freq / 10)
    avg_gain_lin = np.mean(power_mW)
    avg_gain_dBi = 10 * np.log10(avg_gain_lin)

    # Add a text box or similar to your plot to display these values. For example:
    # Note: Antenna gain is in dBi (relative to isotropic), not dBm (power)
    textstr = f"Gain Summary at {selected_frequency} (MHz): Max: {max_gain:.1f} (dBi) Min: {min_gain:.1f} (dBi) Avg: {avg_gain_dBi:.1f} (dBi)"
    ax.annotate(
        textstr,
        xy=(-0.1, -0.1),
        xycoords="axes fraction",
        fontsize=10,
        bbox=dict(facecolor="none", edgecolor="black", boxstyle="round,pad=0.5"),
    )

    if save_path:
        azimuth_plot_path = os.path.join(save_path, f"Azimuth_Cuts_{selected_frequency}MHz.png")
        plt.savefig(azimuth_plot_path, format="png", dpi=300)
        plt.close()  # Close the plot after saving
    else:
        plt.show()  # Display the plot

    # Check if save_path is provided
    if datasheet_plots:
        plot_additional_polar_patterns(
            plot_phi_rad, theta_angles_deg, selected_azimuth_freq, selected_frequency, save_path
        )


def plot_additional_polar_patterns(
    plot_phi_rad, plot_theta_deg, plot_Total_Gain_dB, selected_frequency, save_path=None
):

    # Define gain summary function for reuse
    def gain_summary(gain_values):
        this_min = np.nanmin(gain_values)
        this_max = np.nanmax(gain_values)
        this_mean = 10 * np.log10(np.nanmean(10 ** (gain_values / 10)))
        return this_min, this_max, this_mean

    # Create polar plot for specific conditions
    def create_polar_plot(
        title, theta_values, gain_values, freq, plot_type, save_path=None, annotations=None
    ):
        plt.figure()
        ax = plt.subplot(111, projection="polar")

        # Common settings
        ax.plot(theta_values, gain_values, linewidth=2)  # ,color='blue')
        ax.grid(True, color="gray", linestyle="--", linewidth=0.5)

        # Plot type-specific settings
        settings = {
            "azimuth": {
                "rlabel_position": 90,
                "theta_zero_location": "E",
                "theta_direction": 1,
                "ylim": [polar_dB_min, polar_dB_max],
                "yticks": np.arange(polar_dB_min, polar_dB_max + 1, 5),
                "xticks": np.deg2rad(np.arange(0, 360, 30)),
                "xticklabels": [
                    "0°",
                    "30°",
                    "60°",
                    "90°",
                    "120°",
                    "150°",
                    "180°",
                    "210°",
                    "240°",
                    "270°",
                    "300°",
                    "330°",
                ],
            },
            "elevation": {
                "rlabel_position": 0,
                "theta_zero_location": "N",
                "theta_direction": 1,
                "ylim": [polar_dB_min, polar_dB_max],
                "yticks": np.arange(polar_dB_min, polar_dB_max + 1, 5),
                "xticks": np.deg2rad(np.arange(0, 360, 30)),
                "xticklabels": [
                    "0°",
                    "30°",
                    "60°",
                    "90°",
                    "120°",
                    "150°",
                    "180°",
                    "150°",
                    "120°",
                    "90°",
                    "60°",
                    "30°",
                ],
            },
        }

        ax.set(**settings[plot_type])
        ax.set_title(title + f" at {freq} MHz")

        # Create Gain Summary below plot
        this_min, this_max, this_mean = gain_summary(gain_values)

        summary_text = (
            f"Gain Summary at {freq} MHz"
            f" min: {this_min:.1f} (dBi)"
            f" max: {this_max:.1f} (dBi)"
            f" avg: {this_mean:.1f} (dBi)"
        )
        ax.annotate(
            summary_text,
            xy=(0.5, -0.11),
            xycoords="axes fraction",
            ha="center",
            va="center",
            bbox=dict(boxstyle="round", alpha=0.1),
        )

        # Add annotations if provided
        if annotations:
            for annotation in annotations:
                ax.annotate(
                    annotation["text"],
                    xy=annotation["xy"],
                    xycoords="data",
                    textcoords="offset points",
                    xytext=annotation["xytext"],
                    ha="center",
                )

        if save_path:
            plt.savefig(
                os.path.join(save_path, title.replace(" ", "_") + f"_at_{freq}_MHz.png"), dpi=300
            )
            plt.close()
        else:
            plt.show()

    if save_path:
        # Create Subfolder to save additional 2D gain cuts
        two_d_data_subfolder = os.path.join(save_path, f"2D Gain Cuts at {selected_frequency} MHz")
        os.makedirs(two_d_data_subfolder, exist_ok=True)
    else:
        two_d_data_subfolder = None

    # 1. Azimuth Gain Pattern for Theta = 90
    index = np.where(np.abs(plot_theta_deg - 90) < 0.01)[0]
    if index.size != 0:  # Check if index is not empty
        phi_values = np.append(
            plot_phi_rad[index], plot_phi_rad[index][0]
        )  # Append first value to end
        gain_values = np.append(
            plot_Total_Gain_dB[index], plot_Total_Gain_dB[index][0]
        )  # Append first value to end
        create_polar_plot(
            "Total Gain in Theta = 90 Degree Plane",
            phi_values,
            gain_values,
            selected_frequency,
            "azimuth",
            two_d_data_subfolder,
        )
    else:
        print("No data found for Azimuth Gain Pattern Theta = 90")

    # 2. Elevation Gain Pattern Phi = 180/0 (XZ-Plane: Phi=0° Plane)
    index_phi_180 = np.where(np.isclose(plot_phi_rad, np.pi, atol=1e-6))[0]
    index_phi_0 = np.where(np.isclose(plot_phi_rad, 0, atol=1e-6))[0]

    if index_phi_180.size != 0 and index_phi_0.size != 0:
        # phi=180° data plotted on the right half (theta 0→π)
        theta_values_phi_180 = np.radians(plot_theta_deg[index_phi_180])
        gain_values_phi_180 = plot_Total_Gain_dB[index_phi_180]

        # phi=0° data mirrored to the left half (theta 2π→π)
        theta_values_phi_0 = 2 * np.pi - np.radians(plot_theta_deg[index_phi_0])
        gain_values_phi_0 = plot_Total_Gain_dB[index_phi_0]

        # Flatten the 2D arrays to 1D arrays
        theta_values_phi_180_flat = theta_values_phi_180.flatten()
        theta_values_phi_0_flat = theta_values_phi_0.flatten()

        # Ensure that gain arrays are also 1D and of matching shape
        gain_values_phi_180_flat = np.repeat(gain_values_phi_180, theta_values_phi_180.shape[1])
        gain_values_phi_0_flat = np.repeat(gain_values_phi_0, theta_values_phi_0.shape[1])

        # Concatenating the flattened arrays with np.nan to introduce a break in the plot
        theta_values = np.concatenate(
            [theta_values_phi_180_flat, [np.nan], theta_values_phi_0_flat]
        )
        gain_values = np.concatenate([gain_values_phi_180_flat, [np.nan], gain_values_phi_0_flat])

        # Now, both theta_values and gain_values should be 1D arrays of the same shape.
        annotations = [
            {"text": "Phi=180", "xy": (np.pi / 2, polar_dB_max), "xytext": (0, 20)},
            {"text": "Phi=0", "xy": (3 * np.pi / 2, polar_dB_max), "xytext": (0, 20)},
        ]
        create_polar_plot(
            "Total Gain in Phi = 180 & 0 Degrees Plane",
            theta_values,
            gain_values,
            selected_frequency,
            "elevation",
            two_d_data_subfolder,
            annotations,
        )

    # 3. Elevation Gain Pattern Phi = 270/90 (YZ-Plane: Phi=90° Plane)
    index_phi_270 = np.where(np.isclose(plot_phi_rad, 3 * np.pi / 2, atol=1e-6))[0]
    index_phi_90 = np.where(np.isclose(plot_phi_rad, np.pi / 2, atol=1e-6))[0]

    if index_phi_270.size != 0 and index_phi_90.size != 0:
        # phi=270° data plotted on the right half (theta 0→π)
        theta_values_phi_270 = np.radians(plot_theta_deg[index_phi_270])
        gain_values_phi_270 = plot_Total_Gain_dB[index_phi_270]

        # phi=90° data mirrored to the left half (theta 2π→π)
        theta_values_phi_90 = 2 * np.pi - np.radians(plot_theta_deg[index_phi_90])
        gain_values_phi_90 = plot_Total_Gain_dB[index_phi_90]

        # Flatten the 2D arrays to 1D arrays
        theta_values_phi_270_flat = theta_values_phi_270.flatten()
        theta_values_phi_90_flat = theta_values_phi_90.flatten()

        # Ensure that gain arrays are also 1D and of matching shape
        gain_values_phi_270_flat = np.repeat(gain_values_phi_270, theta_values_phi_270.shape[1])
        gain_values_phi_90_flat = np.repeat(gain_values_phi_90, theta_values_phi_90.shape[1])

        # Concatenating the flattened arrays with np.nan to introduce a break in the plot
        theta_values = np.concatenate(
            [theta_values_phi_270_flat, [np.nan], theta_values_phi_90_flat]
        )
        gain_values = np.concatenate([gain_values_phi_270_flat, [np.nan], gain_values_phi_90_flat])

        # Now, both theta_values and gain_values should be 1D arrays of the same shape.
        annotations = [
            {"text": "Phi=270", "xy": (np.pi / 2, polar_dB_max), "xytext": (0, 20)},
            {"text": "Phi=90", "xy": (3 * np.pi / 2, polar_dB_max), "xytext": (0, 20)},
        ]
        create_polar_plot(
            "Total Gain in Phi = 270 & 90 Degrees Plane",
            theta_values,
            gain_values,
            selected_frequency,
            "elevation",
            two_d_data_subfolder,
            annotations,
        )


def db_to_linear(db_value):
    return 10 ** (db_value / 10)


# Helper function to process gain data for plotting.
def process_data(selected_data, selected_phi_angles_deg, selected_theta_angles_deg):
    # Get unique phi and theta values
    unique_phi = np.unique(selected_phi_angles_deg)
    unique_theta = np.unique(selected_theta_angles_deg)

    # Determine phi increment from data
    phi_inc = unique_phi[1] - unique_phi[0] if len(unique_phi) > 1 else 5.0

    # Check if phi data needs wrapping to complete the sphere
    # Data should span 360° (e.g., 0-355 with 5° increments needs 360° appended)
    phi_span = unique_phi[-1] - unique_phi[0]
    needs_wrapping = phi_span < 360 and (360 - phi_span) <= phi_inc * 1.5

    if needs_wrapping:
        # Append 360° to phi for wrapping
        unique_phi = np.append(unique_phi, unique_phi[0] + 360)
        # Reshape data to (theta, phi) grid
        data_grid = selected_data.reshape((len(unique_theta), len(unique_phi) - 1))
        # Append first column to wrap around the sphere
        data_grid = np.column_stack((data_grid, data_grid[:, 0]))
    else:
        data_grid = selected_data.reshape((len(unique_theta), len(unique_phi)))

    # Interpolate the data for smoother gradient shading
    theta_interp = np.linspace(unique_theta.min(), unique_theta.max(), THETA_RESOLUTION)
    phi_interp = np.linspace(unique_phi.min(), unique_phi.max(), PHI_RESOLUTION)

    # Create interpolation function using RegularGridInterpolator (replacement for deprecated interp2d)
    f_interp = spi.RegularGridInterpolator(
        (unique_theta, unique_phi),
        data_grid,
        method="linear",
        bounds_error=False,
        fill_value=np.nan,
    )

    # Create meshgrid for interpolation points
    phi_mesh, theta_mesh = np.meshgrid(phi_interp, theta_interp, indexing="ij")
    interp_points = np.array([theta_mesh.ravel(), phi_mesh.ravel()]).T

    # Interpolate data
    data_interp = f_interp(interp_points).reshape(len(phi_interp), len(theta_interp)).T

    # Callers map the interpolated dB field to a normalized render radius
    # themselves (R = 0.75 * minmax(dB)), so process_data only needs to hand
    # back the interpolated grid and its axes.
    return data_interp, theta_interp, phi_interp


def normalize_gain(gain_dB):
    """
    Normalize the gain values to be between 0 and 1.
    """
    gain_min = np.min(gain_dB)
    gain_max = np.max(gain_dB)
    return (gain_dB - gain_min) / (gain_max - gain_min)


def plot_passive_3d_component(
    theta_angles_deg,
    phi_angles_deg,
    h_gain_dB,
    v_gain_dB,
    Total_Gain_dB,
    freq_list,
    selected_frequency,
    gain_type,
    axis_mode="auto",
    zmin: float = -15.0,
    zmax: float = 15.0,
    save_path=None,
):
    """
    Plot a 3D representation of the passive component data.

    This function visualizes the passive antenna data in a 3D representation.
    It takes in gain values (in dB) along with theta and phi angles and creates
    a 3D plot to showcase the data for a selected frequency.

    Parameters:
    - theta_angles_deg (ndarray): Array of theta angles in degrees.
    - phi_angles_deg (ndarray): Array of phi angles in degrees.
    - v_gain_dB (ndarray): Vertical gain values in dB.
    - h_gain_dB (ndarray): Horizontal gain values in dB.
    - Total_Gain_dB (ndarray): Total gain values in dB.
    - freq_list (list): List of frequencies.
    - selected_frequency (float): The frequency value for which the 3D plot should be generated.

    Returns:
    None. The function directly displays the 3D plot.
    """

    # Check if the selected frequency is present in the frequency list
    if selected_frequency in freq_list:
        freq_idx = np.where(np.array(freq_list) == selected_frequency)[0][0]
    else:
        print(f"Error: Selected frequency {selected_frequency} not found in the frequency list.")
        return

    if gain_type == "total":
        selected_gain = Total_Gain_dB[:, freq_idx]
        plot_title = f"3D Radiation Pattern - Total Gain at {selected_frequency} MHz"
    elif gain_type == "hpol":
        selected_gain = h_gain_dB[:, freq_idx]
        plot_title = f"3D Radiation Pattern - Phi Polarization Gain at {selected_frequency} MHz"
    elif gain_type == "vpol":
        selected_gain = v_gain_dB[:, freq_idx]
        plot_title = f"3D Radiation Pattern - Theta Polarization Gain at {selected_frequency} MHz"
    else:
        print(f"Error: Invalid gain type {gain_type}.")
        return

    selected_theta_angles_deg = theta_angles_deg[:, freq_idx]
    selected_phi_angles_deg = phi_angles_deg[:, freq_idx]

    # Process gain data → interpolated dB grid + its interpolation axes.
    gain_interp, theta_interp, phi_interp = process_data(
        selected_gain, selected_phi_angles_deg, selected_theta_angles_deg
    )

    # Guard fully-invalid data (mirrors plot_active_3d_data).
    if np.isnan(gain_interp).all():
        print("Error: interpolated gain is all-NaN; skipping 3D plot.")
        return

    # Map gain → radius (normalized) so the shape stays correct. NaN-robust
    # min/max + a zero-radius fallback for the degenerate constant-gain case
    # (amax == amin), matching the active route's guards.
    amin, amax = np.nanmin(gain_interp), np.nanmax(gain_interp)
    denom = amax - amin
    if denom == 0:
        R_unit = np.zeros_like(gain_interp)
    else:
        R_unit = np.nan_to_num((gain_interp - amin) / denom)  # [0..1], NaN→0
    R = 0.75 * R_unit  # global scale factor

    # 2) Build Cartesian coords from R
    PHI, THETA = np.meshgrid(phi_interp, theta_interp)
    theta_rad = np.deg2rad(THETA)
    phi_rad = np.deg2rad(PHI)
    X = R * np.sin(theta_rad) * np.cos(phi_rad)
    Y = R * np.sin(theta_rad) * np.sin(phi_rad)
    Z = R * np.cos(theta_rad)

    # Plotting
    plt.style.use("default")
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(1, 1, 1, projection="3d")
    fig.subplots_adjust(left=0.0, right=0.88, bottom=0.0, top=0.92)

    # Color-scale respects Manual or Auto limits
    if axis_mode == "manual":
        norm = Normalize(zmin, zmax)
    else:
        norm = Normalize(float(np.nanmin(gain_interp)), float(np.nanmax(gain_interp)))
    surf = ax.plot_surface(
        X,
        Y,
        Z,
        facecolors=cm.turbo(norm(gain_interp)),  # type: ignore
        linewidth=0.5,
        antialiased=True,
        shade=False,
        rstride=1,
        cstride=1,
        zorder=1,
    )

    ax.view_init(elev=20, azim=-30)

    # Configure axes: equal aspect, panes, grid, arrows
    _setup_3d_axes(ax, X, Y, Z)

    # Set Title
    fig.suptitle(plot_title, fontsize=14, y=0.97)

    # Add a colorbar
    mappable = cm.ScalarMappable(norm=norm, cmap=cm.turbo)  # type: ignore
    mappable.set_array(gain_interp)
    cbar = fig.colorbar(mappable, ax=ax, pad=0.08, shrink=0.65)
    cbar.set_label("Gain (dBi)", rotation=270, labelpad=20, fontsize=12)
    cbar.ax.tick_params(labelsize=10)

    # Add Max Gain to top of colorbar
    max_gain = float(np.nanmax(selected_gain))
    cbar.ax.set_title(f"{max_gain:.2f} dBi", fontsize=10, weight="bold", pad=4)

    # If Path provided, save otherwise show
    if save_path:
        # Save the first view
        plot_3d_path_1 = os.path.join(save_path, f"3D_{gain_type}_1of2.png")
        fig.savefig(plot_3d_path_1, format="png", dpi=300)

        # Adjust view angle to get the rear side of the 3D plot
        ax.view_init(elev=20, azim=150)  # Adjust the azimuthal angle to get the rear view

        # Save the second view
        plot_3d_path_2 = os.path.join(save_path, f"3D_{gain_type}_2of2.png")
        fig.savefig(plot_3d_path_2, format="png", dpi=300)

        plt.close(fig)
    else:
        plt.show()


# _____________Passive Plotting G&D Functions___________
def plot_gd_data(datasets, labels, x_range):
    # Initialize x_min and x_max to None (auto-scale)
    x_min, x_max = None, None

    # Parse the input if it's not empty
    if x_range:
        try:
            x_values = x_range.split(",")
            x_min = float(x_values[0])
            x_max = float(x_values[1])
        except (ValueError, IndexError) as e:
            print(f"[WARNING] Invalid x-axis range: {e}")
            messagebox.showwarning("Warning", "Invalid input for x-axis range. Using auto-scale.")
    # For each type of data, plot datasets together
    data_types = ["Gain", "Directivity", "Efficiency", "Efficiency_dB"]
    y_labels = ["Gain (dBi)", "Directivity (dB)", "Efficiency (%)", "Efficiency (dB)"]
    titles = [
        "Gain vs Frequency",
        "Directivity vs Frequency",
        "Efficiency (%) vs Frequency",
        "Efficiency (dB) vs Frequency",
    ]

    for data_type, y_label, title in zip(data_types, y_labels, titles):
        plt.figure()
        for data, label in zip(datasets, labels):
            if data_type == "Efficiency_dB":
                y_data = 10 * np.log10(np.array(data["Efficiency"]) / 100)
            else:
                y_data = data[data_type]
            plt.plot(data["Frequency"], y_data, label=label)
        plt.title(title)
        plt.ylabel(y_label)
        plt.xlabel("Frequency (MHz)")
        plt.legend()
        plt.grid(True)
        if x_min is not None and x_max is not None:
            plt.xlim(x_min, x_max)

    plt.show()


# _____________CSV (VSWR/S11) Plotting Functions___________
def process_vswr_files(
    file_paths,
    saved_limit1_freq1,
    saved_limit1_freq2,
    saved_limit1_start,
    saved_limit1_stop,
    saved_limit2_freq1,
    saved_limit2_freq2,
    saved_limit2_start,
    saved_limit2_stop,
):
    import matplotlib.pyplot as plt
    import os
    import pandas as pd

    with open(file_paths[0], "r", encoding="utf-8-sig") as f:
        first_line = f.readline()

    # --- CASE 1: S2VNA files (with '!' in header) ---
    if "!" in first_line:
        all_data = [parse_2port_data(fp) for fp in file_paths]
        preferred_order = ["S11(dB)", "S11(SWR)", "S22(dB)", "S22(SWR)", "S21(dB)", "S12(dB)"]
        unique_columns = sorted(
            {c for d in all_data for c in d.columns if c != "! Stimulus(Hz)"},
            key=lambda x: preferred_order.index(x) if x in preferred_order else 99,
        )

        fig, axes = plt.subplots(nrows=len(unique_columns), figsize=(10, 8))
        if len(unique_columns) == 1:
            axes = [axes]

        for ax, column in zip(axes, unique_columns):
            for df, fp in zip(all_data, file_paths):
                if column in df:
                    freqs_ghz = df["! Stimulus(Hz)"] / 1e9
                    ax.plot(freqs_ghz, df[column], label=os.path.basename(fp))

            ax.set_title(column)
            ax.set_xlabel("Frequency (GHz)")
            if "dB" in column:
                ax.set_ylabel("Magnitude (dB)")
            elif "SWR" in column:
                ax.set_ylabel("VSWR")
                ax.set_ylim(top=5)

            # Draw limit lines if frequency range is defined
            if saved_limit1_freq1 != 0 and saved_limit1_freq2 != 0:
                ax.plot(
                    [saved_limit1_freq1, saved_limit1_freq2],
                    [saved_limit1_start, saved_limit1_stop],
                    color="red",
                    linestyle="--",
                    linewidth=2,
                    zorder=100,
                    label="Limit 1",
                )

            if saved_limit2_freq1 != 0 and saved_limit2_freq2 != 0:
                ax.plot(
                    [saved_limit2_freq1, saved_limit2_freq2],
                    [saved_limit2_start, saved_limit2_stop],
                    color="green",
                    linestyle="--",
                    linewidth=2,
                    zorder=100,
                    label="Limit 2",
                )

            ax.grid(True)
            ax.legend()

        plt.tight_layout()
        plt.show()

    # --- CASE 2: RVNA files (basic CSV: Frequency, S11) ---
    else:
        fig, ax = plt.subplots(figsize=(10, 6))
        for fp in file_paths:
            df = pd.read_csv(fp)
            freqs_ghz = df.iloc[:, 0] / 1e9
            values = df.iloc[:, 1]

            ax.plot(freqs_ghz, values, label=os.path.basename(fp))

        # Auto-label: detect if S11 is negative → it's in dB
        if values.mean() < 0:
            ax.set_ylabel("Return Loss (dB)")
            ax.set_title("S11 LogMag vs Frequency")
        else:
            ax.set_ylabel("VSWR")
            ax.set_ylim(top=5)
            ax.set_title("VSWR vs Frequency")

        ax.set_xlabel("Frequency (GHz)")

        # Apply limits if defined
        if saved_limit1_freq1 != 0 and saved_limit1_freq2 != 0:
            ax.plot(
                [saved_limit1_freq1, saved_limit1_freq2],
                [saved_limit1_start, saved_limit1_stop],
                color="red",
                linestyle="--",
                linewidth=2,
                zorder=100,
                label="Limit 1",
            )

        if saved_limit2_freq1 != 0 and saved_limit2_freq2 != 0:
            ax.plot(
                [saved_limit2_freq1, saved_limit2_freq2],
                [saved_limit2_start, saved_limit2_stop],
                color="green",
                linestyle="--",
                linewidth=2,
                zorder=100,
                label="Limit 2",
            )

        ax.legend()
        ax.grid(True)
        plt.tight_layout()
        plt.show()


# Plots M5090 S2VNA S-Parameter Files Depending On what S-parameters are available
def plot_2port_data(file_paths):
    all_data = [parse_2port_data(file_path) for file_path in file_paths]
    unique_columns = set(
        col for data in all_data for col in data.columns if col != "! Stimulus(Hz)"
    )
    fig, axes = plt.subplots(nrows=len(unique_columns), figsize=(10, 6 * len(unique_columns)))
    if len(unique_columns) == 1:
        axes = [axes]

    for ax, column in zip(axes, unique_columns):
        for data, file_path in zip(all_data, file_paths):
            if column in data:
                freqs_ghz = data["! Stimulus(Hz)"] / 1e9
                values = data[column]
                ax.plot(freqs_ghz, values, label=f"{os.path.basename(file_path)}")

                ax.set_title(column)
                ax.set_xlabel("Frequency (GHz)")
                if "dB" in column:
                    ax.set_ylabel("Magnitude (dB)")
                else:
                    ax.set_ylabel("Value")
                ax.grid(True)
                ax.legend()

    plt.tight_layout()
    plt.show()


def plot_ecc_data(ecc_results):
    """
    Plot Envelope Correlation Coefficient vs Frequency.
    ecc_results: list of (freq_MHz, ecc_value)
    """
    if not ecc_results:
        raise ValueError("No ECC results to plot")

    freqs, eccs = zip(*sorted(ecc_results))
    plt.figure(figsize=(8, 5))
    plt.plot(freqs, eccs, marker="o", linestyle="-")
    plt.title("Envelope Correlation Coefficient vs Frequency")
    plt.xlabel("Frequency (MHz)")
    plt.ylabel("ECC")
    plt.grid(True, linestyle="--")
    plt.tight_layout()
    plt.show()


# _____________Polarization Analysis Plotting Functions___________


def plot_polarization_2d(
    theta_deg, phi_deg, ar_db, tilt_deg, sense, xpd_db, frequency, save_path=None
):
    """
    Plot 2D polar representations of polarization parameters.

    Parameters:
    - theta_deg: 1D array of theta angles in degrees
    - phi_deg: 1D array of phi angles in degrees
    - ar_db: 2D array of axial ratio in dB
    - tilt_deg: 2D array of tilt angle in degrees
    - sense: 2D array of polarization sense (+1=LHCP, -1=RHCP)
    - xpd_db: 2D array of cross-polarization discrimination in dB
    - frequency: Frequency in MHz
    - save_path: Optional path to save figures
    """
    from matplotlib.colors import TwoSlopeNorm

    # Create figure with subplots
    fig = plt.figure(figsize=(16, 12))

    # Convert to meshgrid for plotting
    phi_mesh, theta_mesh = np.meshgrid(phi_deg, theta_deg)

    # 1. Axial Ratio
    ax1 = fig.add_subplot(2, 3, 1)
    # Limit AR display to reasonable range
    ar_display = np.clip(ar_db, 0, 40)
    c1 = ax1.contourf(phi_mesh, theta_mesh, ar_display, levels=20, cmap="viridis")
    ax1.set_xlabel("Phi (degrees)")
    ax1.set_ylabel("Theta (degrees)")
    ax1.set_title(f"Axial Ratio (dB) @ {frequency} MHz")
    cbar1 = plt.colorbar(c1, ax=ax1)
    cbar1.set_label("AR (dB)")
    ax1.grid(True, alpha=0.3)

    # 2. Tilt Angle
    ax2 = fig.add_subplot(2, 3, 2)
    # Use diverging colormap centered at 0
    norm = TwoSlopeNorm(vmin=-90, vcenter=0, vmax=90)
    c2 = ax2.contourf(phi_mesh, theta_mesh, tilt_deg, levels=20, cmap="RdBu_r", norm=norm)
    ax2.set_xlabel("Phi (degrees)")
    ax2.set_ylabel("Theta (degrees)")
    ax2.set_title(f"Tilt Angle (degrees) @ {frequency} MHz")
    cbar2 = plt.colorbar(c2, ax=ax2)
    cbar2.set_label("Tilt (deg)")
    ax2.grid(True, alpha=0.3)

    # 3. Polarization Sense
    ax3 = fig.add_subplot(2, 3, 3)
    c3 = ax3.contourf(
        phi_mesh,
        theta_mesh,
        sense,
        levels=[-1.5, -0.5, 0.5, 1.5],
        colors=["blue", "white", "red"],
        alpha=0.7,
    )
    ax3.set_xlabel("Phi (degrees)")
    ax3.set_ylabel("Theta (degrees)")
    ax3.set_title(f"Polarization Sense @ {frequency} MHz")
    cbar3 = plt.colorbar(c3, ax=ax3, ticks=[-1, 0, 1])
    cbar3.ax.set_yticklabels(["RHCP", "Linear", "LHCP"])
    ax3.grid(True, alpha=0.3)

    # 4. XPD (with nan handling)
    ax4 = fig.add_subplot(2, 3, 4)
    # Replace nan/inf with 40 dB (essentially "perfect" linear polarization)
    xpd_display = np.where(np.isfinite(xpd_db), xpd_db, 40.0)
    xpd_display = np.clip(xpd_display, 0, 40)
    c4 = ax4.contourf(phi_mesh, theta_mesh, xpd_display, levels=20, cmap="plasma")
    ax4.set_xlabel("Phi (degrees)")
    ax4.set_ylabel("Theta (degrees)")
    ax4.set_title(f"XPD (dB) @ {frequency} MHz")
    cbar4 = plt.colorbar(c4, ax=ax4)
    cbar4.set_label("XPD (dB)")
    ax4.grid(True, alpha=0.3)

    # 5. AR Polar Plot (Azimuth cut at theta=90°)
    ax5 = fig.add_subplot(2, 3, 5, projection="polar")
    theta_90_idx = np.argmin(np.abs(theta_deg - 90))
    ar_cut = ar_display[theta_90_idx, :]
    phi_rad = np.deg2rad(phi_deg)
    ax5.plot(phi_rad, ar_cut, "b-", linewidth=2)
    ax5.set_theta_zero_location("N")  # type: ignore
    ax5.set_theta_direction(-1)  # type: ignore
    ax5.set_title(f"AR (dB) - Azimuth Cut (θ=90°)", pad=20)
    ax5.grid(True)

    # 6. Tilt Polar Plot (Azimuth cut at theta=90°)
    ax6 = fig.add_subplot(2, 3, 6, projection="polar")
    tilt_cut = tilt_deg[theta_90_idx, :]
    # Normalize tilt to 0-180 for plotting
    tilt_plot = np.abs(tilt_cut)
    ax6.plot(phi_rad, tilt_plot, "r-", linewidth=2)
    ax6.set_theta_zero_location("N")  # type: ignore
    ax6.set_theta_direction(-1)  # type: ignore
    ax6.set_title(f"Tilt Angle - Azimuth Cut (θ=90°)", pad=20)
    ax6.grid(True)

    plt.tight_layout()

    if save_path:
        fig.savefig(
            os.path.join(save_path, f"polarization_2D_{frequency}MHz.png"),
            dpi=300,
            bbox_inches="tight",
        )
        plt.close(fig)
    else:
        plt.show()


def plot_polarization_3d(theta_deg, phi_deg, ar_db, tilt_deg, sense, frequency, save_path=None):
    """
    Plot 3D spherical representations of polarization parameters.

    Parameters:
    - theta_deg: 1D array of theta angles in degrees
    - phi_deg: 1D array of phi angles in degrees
    - ar_db: 2D array of axial ratio in dB
    - tilt_deg: 2D array of tilt angle in degrees
    - sense: 2D array of polarization sense (+1=LHCP, -1=RHCP)
    - frequency: Frequency in MHz
    - save_path: Optional path to save figures
    """
    # Interpolate for smoother visualization
    from scipy.interpolate import RegularGridInterpolator

    # Wrap phi to 360° for complete sphere (if not already wrapped)
    if phi_deg[0] == 0 and phi_deg[-1] < 360:
        phi_deg_wrapped = np.append(phi_deg, 360)
        # Append first row of data to wrap the sphere
        ar_db_wrapped = np.column_stack((ar_db, ar_db[:, 0]))
        sense_wrapped = np.column_stack((sense, sense[:, 0]))
    else:
        phi_deg_wrapped = phi_deg
        ar_db_wrapped = ar_db
        sense_wrapped = sense

    # Use config resolution settings for consistency with other 3D plots
    theta_interp = np.linspace(theta_deg.min(), theta_deg.max(), THETA_RESOLUTION)
    phi_interp = np.linspace(phi_deg_wrapped.min(), phi_deg_wrapped.max(), PHI_RESOLUTION)

    # Create interpolators
    interp_ar = RegularGridInterpolator(
        (theta_deg, phi_deg_wrapped),
        ar_db_wrapped,
        method="linear",
        bounds_error=False,
        fill_value=np.nan,
    )
    interp_sense = RegularGridInterpolator(
        (theta_deg, phi_deg_wrapped),
        sense_wrapped,
        method="linear",
        bounds_error=False,
        fill_value=0,
    )

    # Create meshgrid for interpolation
    phi_mesh_interp, theta_mesh_interp = np.meshgrid(phi_interp, theta_interp)
    interp_points = np.array([theta_mesh_interp.ravel(), phi_mesh_interp.ravel()]).T

    # Interpolate data
    ar_interp = interp_ar(interp_points).reshape(len(theta_interp), len(phi_interp))
    sense_interp = interp_sense(interp_points).reshape(len(theta_interp), len(phi_interp))

    # Convert angles to radians
    theta_rad = np.deg2rad(theta_interp)
    phi_rad = np.deg2rad(phi_interp)

    # Create meshgrid
    phi_mesh, theta_mesh = np.meshgrid(phi_rad, theta_rad)

    # Limit AR for better visualization
    ar_display = np.clip(ar_interp, 0, 30)

    # Convert to Cartesian coordinates for 3D plotting
    # Use AR as radius (normalized for better shape visibility)
    r = 10 + ar_display / 3  # Offset + scaled AR
    x = r * np.sin(theta_mesh) * np.cos(phi_mesh)
    y = r * np.sin(theta_mesh) * np.sin(phi_mesh)
    z = r * np.cos(theta_mesh)

    # Create figure with two 3D subplots
    fig = plt.figure(figsize=(16, 7))

    # 1. Axial Ratio in 3D
    ax1 = fig.add_subplot(1, 2, 1, projection="3d")
    # Create colormap
    cmap_viridis = matplotlib.colormaps.get_cmap("viridis")
    surf1 = ax1.plot_surface(
        x,
        y,
        z,
        facecolors=cmap_viridis(ar_display / 30),
        rstride=1,
        cstride=1,
        linewidth=0,
        antialiased=True,
        shade=True,
        alpha=0.95,
    )
    ax1.view_init(elev=20, azim=-30)
    _setup_3d_axes(ax1, x, y, z)
    ax1.set_title(f"Axial Ratio (3D) @ {frequency} MHz\nRadius = f(AR)", fontsize=14, weight="bold")

    # Add colorbar
    m = cm.ScalarMappable(cmap="viridis")
    m.set_array(ar_display)
    m.set_clim(0, 30)
    cbar1 = plt.colorbar(m, ax=ax1, shrink=0.5, aspect=5, pad=0.1)
    cbar1.set_label("AR (dB)", rotation=270, labelpad=20, fontsize=12)

    # 2. Polarization Sense in 3D
    ax2 = fig.add_subplot(1, 2, 2, projection="3d")
    # Color by sense: blue=RHCP, red=LHCP
    sense_colors = np.where(sense_interp > 0, 1, 0)  # 1=LHCP, 0=RHCP
    cmap_rdbu = matplotlib.colormaps.get_cmap("RdBu")
    surf2 = ax2.plot_surface(
        x,
        y,
        z,
        facecolors=cmap_rdbu(sense_colors),
        rstride=1,
        cstride=1,
        linewidth=0,
        antialiased=True,
        shade=True,
        alpha=0.95,
    )
    ax2.view_init(elev=20, azim=-30)
    _setup_3d_axes(ax2, x, y, z)
    ax2.set_title(
        f"Polarization Sense (3D) @ {frequency} MHz\nRed=LHCP, Blue=RHCP",
        fontsize=14,
        weight="bold",
    )

    plt.tight_layout()

    if save_path:
        fig.savefig(
            os.path.join(save_path, f"polarization_3D_{frequency}MHz.png"),
            dpi=300,
            bbox_inches="tight",
        )
        plt.close(fig)
    else:
        plt.show()


# ─────────────────────────────────────────────────────────────────────────────
# MARITIME / HORIZON VISUALIZATION FUNCTIONS
# ─────────────────────────────────────────────────────────────────────────────


def _prepare_gain_grid(theta_angles_deg, phi_angles_deg, gain_data, freq_idx):
    """
    Reshape flat passive measurement data into a 2D gain grid.

    For passive data, theta/phi/gain arrays are 2D with shape (n_points, n_freqs).
    For active data, gain_2d is already (n_theta, n_phi) — passed through unchanged.

    Parameters:
        theta_angles_deg: Theta angle array (1D for active, 2D for passive)
        phi_angles_deg: Phi angle array (1D for active, 2D for passive)
        gain_data: Gain/power array (2D grid for active, 2D (n_pts, n_freqs) for passive)
        freq_idx: Frequency index (used only for passive multi-freq arrays)

    Returns:
        Tuple of (unique_theta, unique_phi, gain_2d) or (None, None, None) on failure.
    """
    # Active data: gain_data is already a 2D grid (n_theta, n_phi)
    if gain_data.ndim == 2 and theta_angles_deg.ndim == 1:
        return theta_angles_deg, phi_angles_deg, gain_data

    # Passive data: extract single frequency column
    if gain_data.ndim == 2:
        gain_1d = gain_data[:, freq_idx]
        theta_1d = theta_angles_deg[:, freq_idx]
        phi_1d = phi_angles_deg[:, freq_idx]
    else:
        gain_1d = gain_data
        theta_1d = theta_angles_deg
        phi_1d = phi_angles_deg

    unique_theta = np.sort(np.unique(theta_1d))
    unique_phi = np.sort(np.unique(phi_1d))
    n_theta = len(unique_theta)
    n_phi = len(unique_phi)

    if n_theta * n_phi != len(gain_1d):
        return None, None, None

    try:
        gain_2d = gain_1d.reshape((n_theta, n_phi))
        return unique_theta, unique_phi, gain_2d
    except ValueError:
        return None, None, None


def plot_mercator_heatmap(
    theta_deg,
    phi_deg,
    gain_2d,
    frequency,
    data_label="Gain",
    data_unit="dBi",
    theta_min=None,
    theta_max=None,
    cmap="turbo",
    levels=30,
    save_path=None,
):
    """
    Plot a Mercator/cylindrical projection heatmap of antenna gain or power.

    Parameters:
        theta_deg: 1D array of theta angles in degrees
        phi_deg: 1D array of phi angles in degrees
        gain_2d: 2D array of gain/power values (n_theta, n_phi)
        frequency: Frequency in MHz
        data_label: Label for data ("Gain" or "Power")
        data_unit: Unit string ("dBi" or "dBm")
        theta_min: Optional min theta for Y-axis zoom
        theta_max: Optional max theta for Y-axis zoom
        cmap: Colormap name
        levels: Number of contour levels
        save_path: Optional path to save figure
    """
    fig, ax = plt.subplots(figsize=(12, 6))

    phi_mesh, theta_mesh = np.meshgrid(phi_deg, theta_deg)
    c = ax.contourf(phi_mesh, theta_mesh, gain_2d, levels=levels, cmap=cmap)
    cbar = plt.colorbar(c, ax=ax)
    cbar.set_label(f"{data_label} ({data_unit})")

    ax.set_xlabel("Phi (degrees)")
    ax.set_ylabel("Theta (degrees)")
    ax.invert_yaxis()  # 0=zenith at top

    zoomed = theta_min is not None and theta_max is not None
    if zoomed:
        ax.set_ylim(theta_max, theta_min)
        ax.axhline(y=theta_min, color="white", linestyle="--", alpha=0.7, linewidth=1)
        ax.axhline(y=theta_max, color="white", linestyle="--", alpha=0.7, linewidth=1)
        title_suffix = f" (Theta {theta_min:.0f}-{theta_max:.0f} deg)"
    else:
        title_suffix = ""

    ax.set_title(
        f"Mercator {data_label} Heatmap @ {frequency} MHz{title_suffix}",
        fontsize=14,
    )
    ax.grid(True, alpha=0.3)

    # Gain summary annotation
    if zoomed:
        mask = (theta_mesh >= theta_min) & (theta_mesh <= theta_max)
        visible = gain_2d[(theta_deg >= theta_min) & (theta_deg <= theta_max), :]
    else:
        visible = gain_2d

    max_val = np.max(visible)
    min_val = np.min(visible)
    lin = 10 ** (visible / 10)
    avg_val = 10 * np.log10(np.mean(lin))

    textstr = (
        f"Max: {max_val:.1f} {data_unit}  "
        f"Min: {min_val:.1f} {data_unit}  "
        f"Avg: {avg_val:.1f} {data_unit}"
    )
    ax.annotate(
        textstr,
        xy=(0.5, -0.15),
        xycoords="axes fraction",
        fontsize=9,
        ha="center",
        bbox=dict(facecolor="white", edgecolor="gray", alpha=0.8, boxstyle="round,pad=0.3"),
    )

    plt.tight_layout()

    if save_path:
        suffix = f"_theta{theta_min:.0f}-{theta_max:.0f}" if zoomed else ""
        fname = f"mercator_{data_label.lower()}_{frequency}MHz{suffix}.png"
        fig.savefig(os.path.join(save_path, fname), dpi=300, bbox_inches="tight")
        plt.close(fig)
    else:
        plt.show()


def plot_conical_cuts(
    theta_deg,
    phi_deg,
    gain_2d,
    frequency,
    theta_cuts=None,
    data_label="Gain",
    data_unit="dBi",
    gain_threshold=-3.0,
    polar=True,
    save_path=None,
):
    """
    Plot conical (constant-theta) cuts of gain/power vs phi.

    Parameters:
        theta_deg: 1D array of theta angles in degrees
        phi_deg: 1D array of phi angles in degrees
        gain_2d: 2D array of gain/power values (n_theta, n_phi)
        frequency: Frequency in MHz
        theta_cuts: List of theta angles for cuts (default: maritime horizon cuts)
        data_label: Label for data ("Gain" or "Power")
        data_unit: Unit string ("dBi" or "dBm")
        polar: If True, plot on polar projection; if False, Cartesian
        save_path: Optional path to save figure
    """
    if theta_cuts is None:
        theta_cuts = [60, 70, 80, 90, 100, 110, 120]

    fig = plt.figure(figsize=(10, 8) if polar else (12, 6))
    if polar:
        ax = fig.add_subplot(111, projection="polar")
    else:
        ax = fig.add_subplot(111)

    colors = matplotlib.colormaps.get_cmap("viridis")(np.linspace(0, 1, len(theta_cuts)))

    for i, theta_cut in enumerate(theta_cuts):
        theta_idx = np.argmin(np.abs(theta_deg - theta_cut))
        cut_gain = gain_2d[theta_idx, :]

        if polar:
            phi_rad = np.deg2rad(phi_deg)
            # Wrap for closure
            phi_plot = np.append(phi_rad, phi_rad[0])
            gain_plot = np.append(cut_gain, cut_gain[0])
            ax.plot(phi_plot, gain_plot, color=colors[i], label=f"θ={theta_cut}°", linewidth=1.5)
        else:
            ax.plot(phi_deg, cut_gain, color=colors[i], label=f"θ={theta_cut}°", linewidth=1.5)

    theta_range_str = f"θ={theta_cuts[0]}–{theta_cuts[-1]}°"

    if polar:
        ax.set_theta_zero_location("N")
        ax.set_theta_direction(-1)
        ax.set_title(
            f"Conical Cuts - {data_label} ({data_unit}) @ {frequency} MHz\n{theta_range_str}",
            pad=20,
            fontsize=14,
        )
    else:
        ax.set_xlabel("Phi (degrees)")
        ax.set_ylabel(f"{data_label} ({data_unit})")
        ax.set_title(
            f"{data_label} over Azimuth @ {frequency} MHz — {theta_range_str}",
            fontsize=14,
        )
        ax.grid(True, alpha=0.3)
        # Reference line relative to peak using the coverage threshold setting
        all_cut_data = np.concatenate(
            [gain_2d[np.argmin(np.abs(theta_deg - tc)), :] for tc in theta_cuts]
        )
        peak_val = np.max(all_cut_data)
        ref_line = peak_val + gain_threshold  # gain_threshold is negative
        ax.axhline(
            y=ref_line,
            color="red",
            linestyle="--",
            alpha=0.5,
            linewidth=1,
            label=f"{gain_threshold:.0f} dB ref ({ref_line:.1f} {data_unit})",
        )

    ax.legend(loc="upper right", bbox_to_anchor=(1.3, 1.0), fontsize=8)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        mode = "polar" if polar else "cartesian"
        fname = f"conical_cuts_{mode}_{frequency}MHz.png"
        fig.savefig(os.path.join(save_path, fname), dpi=300, bbox_inches="tight")
        plt.close(fig)
    else:
        plt.show()


def plot_gain_over_azimuth(
    theta_deg,
    phi_deg,
    gain_2d,
    frequency,
    theta_cuts=None,
    data_label="Gain",
    data_unit="dBi",
    gain_threshold=-3.0,
    save_path=None,
):
    """
    Plot data over Azimuth (Cartesian view of conical cuts) with -3 dB reference.

    Uses data_label/data_unit to label axes and title correctly for Gain (dBi)
    or Power (dBm) depending on whether passive or active data is provided.
    """
    if theta_cuts is None:
        theta_cuts = [60, 70, 80, 90, 100, 110, 120]

    # Use conical cuts in Cartesian mode
    fig = plt.figure(figsize=(12, 6))
    ax = fig.add_subplot(111)

    colors = matplotlib.colormaps.get_cmap("viridis")(np.linspace(0, 1, len(theta_cuts)))

    for i, theta_cut in enumerate(theta_cuts):
        theta_idx = np.argmin(np.abs(theta_deg - theta_cut))
        cut_gain = gain_2d[theta_idx, :]
        ax.plot(phi_deg, cut_gain, color=colors[i], label=f"θ={theta_cut}°", linewidth=1.5)

    # Compute stats across all cuts first (used for ref line and summary)
    all_cuts = np.concatenate([gain_2d[np.argmin(np.abs(theta_deg - tc)), :] for tc in theta_cuts])
    peak_val = np.max(all_cuts)
    ref_line = peak_val + gain_threshold  # gain_threshold is negative
    ax.axhline(
        y=ref_line,
        color="red",
        linestyle="--",
        alpha=0.5,
        linewidth=1,
        label=f"{gain_threshold:.0f} dB ref ({ref_line:.1f} {data_unit})",
    )
    ax.set_xlabel("Phi (degrees)")
    ax.set_ylabel(f"{data_label} ({data_unit})")
    theta_range_str = f"θ={theta_cuts[0]}–{theta_cuts[-1]}°"
    ax.set_title(f"{data_label} over Azimuth @ {frequency} MHz — {theta_range_str}", fontsize=14)
    ax.legend(loc="upper right", bbox_to_anchor=(1.2, 1.0), fontsize=8)
    ax.grid(True, alpha=0.3)

    # Summary annotation reusing all_cuts computed above
    min_v = np.min(all_cuts)
    avg_v = 10 * np.log10(np.mean(10 ** (all_cuts / 10)))
    summary = f"Max: {peak_val:.1f} {data_unit}   Min: {min_v:.1f} {data_unit}   Avg: {avg_v:.1f} {data_unit}"
    ax.annotate(
        summary,
        xy=(0.5, -0.12),
        xycoords="axes fraction",
        fontsize=9,
        ha="center",
        bbox=dict(facecolor="white", edgecolor="gray", alpha=0.8, boxstyle="round,pad=0.3"),
    )

    plt.tight_layout()

    if save_path:
        fname = f"goa_{frequency}MHz.png"
        fig.savefig(os.path.join(save_path, fname), dpi=300, bbox_inches="tight")
        plt.close(fig)
    else:
        plt.show()


def _compute_partial_trp(theta_deg, phi_deg, gain_2d, theta_min=None, theta_max=None):
    """
    Compute TRP (Total Radiated Power) over a partial or full sphere.

    Uses numerical integration with sin(theta) weighting:
        TRP = ΔΘ·ΔΦ / (4π) · ΣΣ P_linear(θ,φ) · sin(θ)

    Parameters:
        theta_deg: 1D array of theta angles in degrees
        phi_deg: 1D array of phi angles in degrees
        gain_2d: 2D array (n_theta, n_phi) of gain/power in dB scale
        theta_min: Optional lower bound (degrees). None = use full range.
        theta_max: Optional upper bound (degrees). None = use full range.

    Returns:
        TRP in dB (10*log10 of linear TRP), or -inf if no data.
    """
    if theta_min is not None and theta_max is not None:
        mask = (theta_deg >= theta_min) & (theta_deg <= theta_max)
        sel_theta = theta_deg[mask]
        sel_gain = gain_2d[mask, :]
    else:
        sel_theta = theta_deg
        sel_gain = gain_2d

    if sel_gain.size == 0:
        return float("-inf")

    d_theta = np.deg2rad(np.mean(np.diff(sel_theta))) if len(sel_theta) > 1 else np.deg2rad(5.0)
    d_phi = np.deg2rad(np.mean(np.diff(phi_deg))) if len(phi_deg) > 1 else np.deg2rad(5.0)

    lin = 10 ** (sel_gain / 10)
    sin_w = np.sin(np.deg2rad(sel_theta))
    # Integrate: ΔΘ·ΔΦ/(4π) · Σ_θ Σ_φ P(θ,φ)·sin(θ)
    integrated = d_theta * d_phi / (4 * np.pi) * np.sum(lin * sin_w[:, np.newaxis])
    return 10 * np.log10(integrated) if integrated > 0 else float("-inf")


def plot_horizon_statistics(
    theta_deg,
    phi_deg,
    gain_2d,
    frequency,
    theta_min=60,
    theta_max=120,
    gain_threshold=-3.0,
    data_label="Gain",
    data_unit="dBi",
    conducted_power_dBm=None,
    save_path=None,
):
    """
    Plot horizon coverage statistics as a table + mini polar plot.

    Parameters:
        theta_deg: 1D array of theta angles in degrees
        phi_deg: 1D array of phi angles in degrees
        gain_2d: 2D array of gain/power values (n_theta, n_phi)
        frequency: Frequency in MHz
        theta_min: Minimum theta for horizon band
        theta_max: Maximum theta for horizon band
        gain_threshold: dB threshold below peak for coverage calculation
        data_label: Label for data
        data_unit: Unit string
        save_path: Optional path to save figure
    """
    try:
        band_stats = calculate_spherical_band_statistics(
            gain_2d,
            theta_deg,
            phi_deg,
            theta_min=theta_min,
            theta_max=theta_max,
            gain_threshold=gain_threshold,
        )
    except ValueError:
        print(f"[Maritime] No data in theta range {theta_min}-{theta_max} deg")
        return

    max_gain = band_stats["max_dB"]
    min_gain = band_stats["min_dB"]
    avg_gain = band_stats["avg_dB"]
    trp_horizon_dB = band_stats["band_trp_dB"]
    trp_full_dB = band_stats["full_trp_dB"]
    coverage_pct = band_stats["coverage_pct"]
    coverage_limit = band_stats["coverage_threshold_dB"]
    null_depth = band_stats["null_depth_dB"]
    null_location = band_stats["null_location"]
    null_location_text = (
        f"theta={null_location['theta_deg']:.0f} deg, phi={null_location['phi_deg']:.0f} deg"
    )

    # ----- Figure: table (left) + multi-cut polar (right) -----
    # Dynamic height: extra rows for fraction + efficiency
    _n_extra = 0
    if np.isfinite(trp_horizon_dB) and np.isfinite(trp_full_dB):
        _n_extra += 1  # Maritime Power Fraction
    if conducted_power_dBm is not None:
        _n_extra += 3  # Conducted Power + Total Eff + Maritime Eff
    _fig_h = 8 + _n_extra * 0.45
    fig = plt.figure(figsize=(16, _fig_h))
    gs = fig.add_gridspec(1, 2, width_ratios=[1.4, 1])

    # --- Left: statistics table ---
    ax_table = fig.add_subplot(gs[0])
    ax_table.axis("off")

    if data_label == "Gain":
        trp_label = "Maritime Integrated Gain"
        trp_full_label = "Full-Sphere Integrated Gain"
    else:
        trp_label = "Maritime TRP"
        trp_full_label = "Full-Sphere TRP"

    table_data = [
        ["Max " + data_label, f"{max_gain:.1f} {data_unit}"],
        ["Min " + data_label, f"{min_gain:.1f} {data_unit}"],
        ["Avg " + data_label + " (linear)", f"{avg_gain:.1f} {data_unit}"],
        [trp_full_label, f"{trp_full_dB:.1f} {data_unit}"],
        [trp_label, f"{trp_horizon_dB:.1f} {data_unit}"],
    ]

    # Maritime power fraction (pattern-only metric, always shown)
    if np.isfinite(trp_horizon_dB) and np.isfinite(trp_full_dB):
        mar_frac_dB = trp_horizon_dB - trp_full_dB
        mar_frac_pct = 10 ** (mar_frac_dB / 10) * 100.0
        table_data.append(["Maritime Power Fraction", f"{mar_frac_pct:.1f}%"])

    # Add efficiency rows when conducted power is provided
    if conducted_power_dBm is not None:
        table_data.append(["Conducted Power", f"{conducted_power_dBm:.1f} dBm"])
        full_eff_dB = trp_full_dB - conducted_power_dBm
        full_eff_pct = 10 ** (full_eff_dB / 10) * 100.0
        table_data.append(["Total Efficiency", f"{full_eff_dB:+.1f} dB ({full_eff_pct:.1f}%)"])
        mar_eff_dB = trp_horizon_dB - conducted_power_dBm
        mar_eff_pct = 10 ** (mar_eff_dB / 10) * 100.0
        table_data.append(["Maritime Efficiency", f"{mar_eff_dB:+.1f} dB ({mar_eff_pct:.1f}%)"])

    table_data += [
        [
            f"Coverage (>{coverage_limit:.1f} {data_unit})",
            f"{coverage_pct:.1f}%",
        ],
        ["Null Depth", f"{null_depth:.1f} dB"],
        ["Null Location", null_location_text],
        ["Theta Range", f"{theta_min} deg - {theta_max} deg"],
        ["Frequency", f"{frequency} MHz"],
    ]

    table = ax_table.table(
        cellText=table_data,
        colLabels=["Metric", "Value"],
        cellLoc="left",
        loc="center",
        colWidths=[0.55, 0.45],
    )
    table.auto_set_font_size(False)
    table.set_fontsize(13)
    table.scale(1, 1.8)

    # Style header row
    for j in range(2):
        table[0, j].set_facecolor("#4A90E2")
        table[0, j].set_text_props(color="white", fontweight="bold", fontsize=14)

    ax_table.set_title(
        f"Maritime Band {data_label} Statistics @ {frequency} MHz ({data_unit})",
        fontsize=16,
        fontweight="bold",
        pad=20,
    )

    # --- Right: multi-cut polar plot across horizon band ---
    ax_polar = fig.add_subplot(gs[1], projection="polar")
    # Pick 3-5 representative cuts spanning the horizon band
    band_thetas = np.array(
        [
            t
            for t in [theta_min, (theta_min + 90) / 2, 90, (90 + theta_max) / 2, theta_max]
            if theta_min <= t <= theta_max
        ]
    )
    # Remove near-duplicates
    band_thetas = np.unique(np.round(band_thetas, 1))
    cmap = matplotlib.colormaps.get_cmap("viridis")
    colors = cmap(np.linspace(0, 1, len(band_thetas)))

    phi_rad = np.deg2rad(phi_deg)
    phi_plot = np.append(phi_rad, phi_rad[0])

    for i, tc in enumerate(band_thetas):
        t_idx = np.argmin(np.abs(theta_deg - tc))
        cut = gain_2d[t_idx, :]
        cut_plot = np.append(cut, cut[0])
        ax_polar.plot(phi_plot, cut_plot, color=colors[i], linewidth=1.5, label=f"θ={tc:.0f}°")

    ax_polar.set_theta_zero_location("N")
    ax_polar.set_theta_direction(-1)
    ax_polar.set_title(
        f"Azimuth Cuts Across Maritime Band ({data_unit})\n{theta_min:.0f}-{theta_max:.0f} deg",
        pad=20,
        fontsize=13,
    )
    ax_polar.legend(loc="upper right", bbox_to_anchor=(1.35, 1.0), fontsize=9)
    ax_polar.grid(True, alpha=0.3)
    ax_polar.tick_params(labelsize=11)

    fig.tight_layout(rect=(0.0, 0.0, 0.93, 0.95))

    if save_path:
        fname = f"horizon_stats_{frequency}MHz.png"
        fig.savefig(os.path.join(save_path, fname), dpi=300, bbox_inches="tight")
        plt.close(fig)
    else:
        plt.show()


def plot_3d_pattern_masked(
    theta_deg,
    phi_deg,
    gain_2d,
    frequency,
    theta_highlight_min=60,
    theta_highlight_max=120,
    mask_alpha=0.15,
    data_label="Gain",
    data_unit="dBi",
    interpolate=True,
    axis_mode="auto",
    zmin=-15,
    zmax=15,
    conducted_power_dBm=None,
    save_path=None,
):
    """
    Plot 3D radiation pattern with alpha transparency outside the horizon band.

    Parameters:
        theta_deg: 1D array of theta angles in degrees
        phi_deg: 1D array of phi angles in degrees
        gain_2d: 2D array of gain/power values (n_theta, n_phi)
        frequency: Frequency in MHz
        theta_highlight_min: Min theta for full-opacity band
        theta_highlight_max: Max theta for full-opacity band
        mask_alpha: Alpha for regions outside highlight band
        data_label: Label for data
        data_unit: Unit string
        interpolate: Whether to interpolate for smoother rendering
        axis_mode: "auto" or "manual" axis scaling
        zmin: Min gain value for manual scaling
        zmax: Max gain value for manual scaling
        save_path: Optional path to save figure
    """
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

    # --- Phi wrapping to close the sphere (matches process_data() logic) ---
    phi_inc = phi_deg[1] - phi_deg[0] if len(phi_deg) > 1 else 5.0
    phi_span = phi_deg[-1] - phi_deg[0]
    needs_wrapping = phi_span < 360 and (360 - phi_span) <= phi_inc * 1.5

    if needs_wrapping:
        wrapped_phi = np.append(phi_deg, phi_deg[0] + 360)
        wrapped_gain = np.column_stack((gain_2d, gain_2d[:, 0]))
    else:
        wrapped_phi = phi_deg
        wrapped_gain = gain_2d

    if interpolate:
        theta_interp = np.linspace(theta_deg.min(), theta_deg.max(), THETA_RESOLUTION)
        phi_interp = np.linspace(wrapped_phi.min(), wrapped_phi.max(), PHI_RESOLUTION)
        try:
            interp_func = spi.RegularGridInterpolator(
                (theta_deg, wrapped_phi),
                wrapped_gain,
                method="linear",
                bounds_error=False,
                fill_value=np.nan,
            )
            PHI_grid, THETA_grid = np.meshgrid(phi_interp, theta_interp)
            pts = np.column_stack([THETA_grid.ravel(), PHI_grid.ravel()])
            gain_interp = interp_func(pts).reshape(THETA_grid.shape)
            use_theta = theta_interp
            use_phi = phi_interp
            use_gain = gain_interp
        except Exception:
            use_theta = theta_deg
            use_phi = wrapped_phi
            use_gain = wrapped_gain
    else:
        use_theta = theta_deg
        use_phi = wrapped_phi
        use_gain = wrapped_gain

    # Build spherical coordinates
    PHI, THETA = np.meshgrid(use_phi, use_theta)
    theta_rad = np.deg2rad(THETA)
    phi_rad = np.deg2rad(PHI)

    # Map gain to radius
    gmin, gmax = np.nanmin(use_gain), np.nanmax(use_gain)
    if gmax == gmin:
        R_unit = np.ones_like(use_gain)
    else:
        R_unit = (use_gain - gmin) / (gmax - gmin)
    R = 0.75 * R_unit

    X = R * np.sin(theta_rad) * np.cos(phi_rad)
    Y = R * np.sin(theta_rad) * np.sin(phi_rad)
    Z = R * np.cos(theta_rad)

    # Color mapping
    if axis_mode == "manual":
        norm = Normalize(zmin, zmax)
    else:
        norm = Normalize(gmin, gmax)

    face_colors = cm.turbo(norm(use_gain))

    # Desaturate regions outside the horizon band instead of using alpha
    # transparency (matplotlib's 3D renderer can't depth-sort transparent faces,
    # causing back-surface bleed-through artifacts).
    in_band = (THETA >= theta_highlight_min) & (THETA <= theta_highlight_max)
    gray = np.array([0.82, 0.82, 0.82, 1.0])
    face_colors[~in_band] = mask_alpha * face_colors[~in_band] + (1 - mask_alpha) * gray

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection="3d")
    fig.subplots_adjust(left=0.0, right=0.88, bottom=0.0, top=0.92)

    ax.plot_surface(
        X,
        Y,
        Z,
        facecolors=face_colors,
        linewidth=0.3,
        antialiased=True,
        shade=False,
        rstride=1,
        cstride=1,
    )

    # Ring lines at band boundaries
    for boundary_theta in [theta_highlight_min, theta_highlight_max]:
        t_rad = np.deg2rad(boundary_theta)
        ring_phi = np.linspace(0, 2 * np.pi, 100)
        # Use average radius at that theta for ring placement
        t_idx = np.argmin(np.abs(use_theta - boundary_theta))
        r_ring = np.mean(R[t_idx, :]) if t_idx < R.shape[0] else 0.5
        ring_x = r_ring * np.sin(t_rad) * np.cos(ring_phi)
        ring_y = r_ring * np.sin(t_rad) * np.sin(ring_phi)
        ring_z = r_ring * np.cos(t_rad) * np.ones_like(ring_phi)
        ax.plot(ring_x, ring_y, ring_z, color="yellow", linewidth=2, alpha=0.9)

    ax.view_init(elev=20, azim=-30)

    # Configure axes: equal aspect, panes, grid, arrows
    _setup_3d_axes(ax, X, Y, Z)
    fig.suptitle(
        f"3D {data_label} Pattern - Maritime Band {theta_highlight_min}-{theta_highlight_max} deg "
        f"@ {frequency} MHz",
        fontsize=14,
        y=0.97,
    )

    # Colorbar
    mappable = cm.ScalarMappable(norm=norm, cmap=cm.turbo)
    mappable.set_array(use_gain)
    cbar = fig.colorbar(mappable, ax=ax, pad=0.08, shrink=0.65)
    cbar.set_label(f"{data_label} ({data_unit})")

    # ----- Maritime band statistics annotation -----
    try:
        band_stats = calculate_spherical_band_statistics(
            gain_2d,
            theta_deg,
            phi_deg,
            theta_min=theta_highlight_min,
            theta_max=theta_highlight_max,
        )
    except ValueError:
        band_stats = None

    if band_stats is not None:
        if data_label == "Gain":
            band_trp_label = "Maritime Int. Gain"
            full_trp_label = "Full-Sphere Int. Gain"
        else:
            band_trp_label = "Maritime TRP"
            full_trp_label = "Full-Sphere TRP"

        stats_text = (
            f"Maritime Band ({theta_highlight_min:.0f}-{theta_highlight_max:.0f} deg)\n"
            f"Max: {band_stats['max_dB']:.1f}  Min: {band_stats['min_dB']:.1f}  "
            f"Avg: {band_stats['avg_dB']:.1f} {data_unit}\n"
            f"{full_trp_label}: {band_stats['full_trp_dB']:.1f} {data_unit}   "
            f"{band_trp_label}: {band_stats['band_trp_dB']:.1f} {data_unit}"
        )
        # Maritime power fraction (pattern-only metric)
        _frac_dB = band_stats["band_trp_dB"] - band_stats["full_trp_dB"]
        if np.isfinite(_frac_dB):
            _frac_pct = 10 ** (_frac_dB / 10) * 100
            stats_text += f"\nMaritime Power Fraction: {_frac_pct:.1f}%"
        if conducted_power_dBm is not None:
            _full_eff_dB = band_stats["full_trp_dB"] - conducted_power_dBm
            _mar_eff_dB = band_stats["band_trp_dB"] - conducted_power_dBm
            _full_eff = 10 ** (_full_eff_dB / 10) * 100
            _mar_eff = 10 ** (_mar_eff_dB / 10) * 100
            stats_text += (
                f"\nTotal Eff: {_full_eff_dB:+.1f} dB ({_full_eff:.1f}%)"
                f"   Maritime Eff: {_mar_eff_dB:+.1f} dB ({_mar_eff:.1f}%)"
            )
        ax.text2D(
            0.02,
            0.02,
            stats_text,
            transform=ax.transAxes,
            fontsize=8.5,
            verticalalignment="bottom",
            bbox=dict(facecolor="white", edgecolor="gray", alpha=0.85, boxstyle="round,pad=0.4"),
        )

    plt.tight_layout()

    if save_path:
        fname = f"3d_masked_{frequency}MHz.png"
        fig.savefig(os.path.join(save_path, fname), dpi=300, bbox_inches="tight")
        plt.close(fig)
    else:
        plt.show()


def generate_maritime_plots(
    theta_deg,
    phi_deg,
    gain_2d,
    frequency,
    data_label="Gain",
    data_unit="dBi",
    theta_min=60.0,
    theta_max=120.0,
    theta_cuts=None,
    gain_threshold=-3.0,
    axis_mode="auto",
    zmin=-15.0,
    zmax=15.0,
    conducted_power_dBm=None,
    save_path=None,
):
    """
    Generate all maritime/horizon visualization plots.

    This dispatcher calls all maritime plot functions with proper parameters,
    avoiding duplication across the 4 entry points (View, Save, Bulk Passive, Bulk Active).
    """
    if theta_cuts is None:
        theta_cuts = [60, 70, 80, 90, 100, 110, 120]

    # 1. Full-range Mercator
    plot_mercator_heatmap(
        theta_deg,
        phi_deg,
        gain_2d,
        frequency,
        data_label=data_label,
        data_unit=data_unit,
        save_path=save_path,
    )

    # 2. Zoomed Mercator (horizon band)
    plot_mercator_heatmap(
        theta_deg,
        phi_deg,
        gain_2d,
        frequency,
        data_label=data_label,
        data_unit=data_unit,
        theta_min=theta_min,
        theta_max=theta_max,
        save_path=save_path,
    )

    # 3. Conical cuts (polar)
    plot_conical_cuts(
        theta_deg,
        phi_deg,
        gain_2d,
        frequency,
        theta_cuts=theta_cuts,
        data_label=data_label,
        data_unit=data_unit,
        gain_threshold=gain_threshold,
        polar=True,
        save_path=save_path,
    )

    # 4. Data-over-Azimuth (Cartesian) — Gain/dBi for passive, Power/dBm for active
    plot_gain_over_azimuth(
        theta_deg,
        phi_deg,
        gain_2d,
        frequency,
        theta_cuts=theta_cuts,
        data_label=data_label,
        data_unit=data_unit,
        gain_threshold=gain_threshold,
        save_path=save_path,
    )

    # 5. Horizon statistics
    plot_horizon_statistics(
        theta_deg,
        phi_deg,
        gain_2d,
        frequency,
        theta_min=theta_min,
        theta_max=theta_max,
        gain_threshold=gain_threshold,
        data_label=data_label,
        data_unit=data_unit,
        conducted_power_dBm=conducted_power_dBm,
        save_path=save_path,
    )

    # 6. 3D masked pattern
    plot_3d_pattern_masked(
        theta_deg,
        phi_deg,
        gain_2d,
        frequency,
        theta_highlight_min=theta_min,
        theta_highlight_max=theta_max,
        data_label=data_label,
        data_unit=data_unit,
        axis_mode=axis_mode,
        zmin=zmin,
        zmax=zmax,
        conducted_power_dBm=conducted_power_dBm,
        save_path=save_path,
    )


# ——— LINK BUDGET & RANGE ESTIMATION PLOTS ——————————————————————————


# Advanced-analysis plot family extracted to advanced_plots.py (#38); re-exported
# so existing `from .plotting import plot_link_budget_summary` imports keep working
# and the dispatcher below can call them as module-level names.
from .advanced_plots import (
    plot_link_budget_summary,
    plot_indoor_coverage_map,
    plot_fading_analysis,
    plot_mimo_analysis,
    plot_wearable_assessment,
)


# ——— DISPATCHER FOR ADVANCED ANALYSIS PLOTS ——————————————————————


def generate_advanced_analysis_plots(
    theta_deg,
    phi_deg,
    gain_2d,
    frequency,
    data_label="Gain",
    data_unit="dBi",
    save_path=None,
    # Link Budget params
    link_budget_enabled=False,
    lb_pt_dbm=0.0,
    lb_pr_dbm=-98.0,
    lb_gr_dbi=0.0,
    lb_path_loss_exp=2.0,
    lb_misc_loss_db=10.0,
    lb_target_range_m=5.0,
    # Indoor params
    indoor_enabled=False,
    indoor_environment="Office",
    indoor_path_loss_exp=3.0,
    indoor_n_walls=1,
    indoor_wall_material="drywall",
    indoor_shadow_fading_db=5.0,
    indoor_max_distance_m=30.0,
    # Fading params
    fading_enabled=False,
    fading_pr_sensitivity_dbm=-98.0,
    fading_pt_dbm=0.0,
    fading_target_reliability=99.0,
    fading_model="rayleigh",
    fading_rician_k=10.0,
    fading_realizations=1000,
    # MIMO params
    mimo_enabled=False,
    mimo_snr_db=20.0,
    mimo_fading_model="rayleigh",
    mimo_rician_k=10.0,
    mimo_xpr_db=6.0,
    mimo_ecc_values=None,
    mimo_gain_data_list=None,
    # Wearable params
    wearable_enabled=False,
    wearable_body_positions=None,
    wearable_tx_power_mw=1.0,
    wearable_num_devices=20,
    wearable_room_size=(10, 10, 3),
):
    """
    Dispatcher for all advanced analysis plots. Called from callbacks/batch
    after pattern processing, analogous to generate_maritime_plots().

    The flat keyword arguments are retained for backward compatibility; they are
    bundled into grouped config dataclasses (#37) and forwarded to the structured
    dispatcher :func:`dispatch_advanced_analysis_plots`. New callers should build
    an :class:`~plot_antenna.advanced_analysis_config.AdvancedAnalysisConfig`
    directly and call the structured dispatcher.
    """
    from .advanced_analysis_config import configs_from_flat_kwargs

    config = configs_from_flat_kwargs(
        link_budget_enabled=link_budget_enabled,
        lb_pt_dbm=lb_pt_dbm,
        lb_pr_dbm=lb_pr_dbm,
        lb_gr_dbi=lb_gr_dbi,
        lb_path_loss_exp=lb_path_loss_exp,
        lb_misc_loss_db=lb_misc_loss_db,
        lb_target_range_m=lb_target_range_m,
        indoor_enabled=indoor_enabled,
        indoor_environment=indoor_environment,
        indoor_path_loss_exp=indoor_path_loss_exp,
        indoor_n_walls=indoor_n_walls,
        indoor_wall_material=indoor_wall_material,
        indoor_shadow_fading_db=indoor_shadow_fading_db,
        indoor_max_distance_m=indoor_max_distance_m,
        fading_enabled=fading_enabled,
        fading_pr_sensitivity_dbm=fading_pr_sensitivity_dbm,
        fading_pt_dbm=fading_pt_dbm,
        fading_target_reliability=fading_target_reliability,
        fading_model=fading_model,
        fading_rician_k=fading_rician_k,
        fading_realizations=fading_realizations,
        mimo_enabled=mimo_enabled,
        mimo_snr_db=mimo_snr_db,
        mimo_fading_model=mimo_fading_model,
        mimo_rician_k=mimo_rician_k,
        mimo_xpr_db=mimo_xpr_db,
        mimo_ecc_values=mimo_ecc_values,
        mimo_gain_data_list=mimo_gain_data_list,
        wearable_enabled=wearable_enabled,
        wearable_body_positions=wearable_body_positions,
        wearable_tx_power_mw=wearable_tx_power_mw,
        wearable_num_devices=wearable_num_devices,
        wearable_room_size=wearable_room_size,
    )
    dispatch_advanced_analysis_plots(
        theta_deg,
        phi_deg,
        gain_2d,
        frequency,
        config,
        data_label=data_label,
        data_unit=data_unit,
        save_path=save_path,
    )


def dispatch_advanced_analysis_plots(
    theta_deg,
    phi_deg,
    gain_2d,
    frequency,
    config,
    *,
    data_label="Gain",
    data_unit="dBi",
    save_path=None,
):
    """Structured dispatcher for the advanced-analysis plots (#37).

    Operates on the grouped :class:`AdvancedAnalysisConfig` instead of 30+ flat
    keyword arguments. Each ``if <group>.enabled:`` branch is independent. Note
    that the indoor and fading branches intentionally reuse link-budget
    parameters (Tx/Rx power, range, path-loss, losses) — this mirrors the GUI,
    where those panels share the link-budget inputs.
    """
    lb = config.link_budget
    indoor = config.indoor
    fading = config.fading
    mimo = config.mimo
    wearable = config.wearable

    if lb.enabled:
        plot_link_budget_summary(
            frequency,
            gain_2d,
            theta_deg,
            phi_deg,
            pt_dbm=lb.pt_dbm,
            pr_dbm=lb.pr_dbm,
            gr_dbi=lb.gr_dbi,
            path_loss_exp=lb.path_loss_exp,
            misc_loss_db=lb.misc_loss_db,
            target_range_m=lb.target_range_m,
            data_label=data_label,
            data_unit=data_unit,
            save_path=save_path,
        )

    if indoor.enabled:
        plot_indoor_coverage_map(
            frequency,
            gain_2d,
            theta_deg,
            phi_deg,
            pt_dbm=lb.pt_dbm,
            pr_sensitivity_dbm=lb.pr_dbm,
            environment=indoor.environment,
            path_loss_exp=indoor.path_loss_exp,
            n_walls=indoor.n_walls,
            wall_material=indoor.wall_material,
            shadow_fading_db=indoor.shadow_fading_db,
            max_distance_m=indoor.max_distance_m,
            data_label=data_label,
            data_unit=data_unit,
            save_path=save_path,
        )

    if fading.enabled:
        plot_fading_analysis(
            frequency,
            gain_2d,
            theta_deg,
            phi_deg,
            pr_sensitivity_dbm=fading.pr_sensitivity_dbm,
            pt_dbm=fading.pt_dbm,
            target_reliability=fading.target_reliability,
            fading_model=fading.model,
            fading_rician_k=fading.rician_k,
            realizations=fading.realizations,
            target_distance_m=lb.target_range_m,
            path_loss_exp=lb.path_loss_exp,
            misc_loss_db=lb.misc_loss_db,
            gr_dbi=lb.gr_dbi,
            data_label=data_label,
            data_unit=data_unit,
            save_path=save_path,
        )

    if mimo.enabled:
        gain_list = mimo.gain_data_list if mimo.gain_data_list is not None else [gain_2d]
        if not isinstance(gain_list, (list, tuple)):
            gain_list = [gain_list]
        gain_list = [g for g in gain_list if g is not None]
        if not gain_list:
            gain_list = [gain_2d]

        plot_mimo_analysis(
            ecc_values=mimo.ecc_values if mimo.ecc_values is not None else [],
            freq_list=[frequency],
            gain_data_list=list(gain_list),
            theta_deg=theta_deg,
            phi_deg=phi_deg,
            snr_db=mimo.snr_db,
            fading=mimo.fading_model,
            K=mimo.rician_k,
            xpr_db=mimo.xpr_db,
            save_path=save_path,
        )

    if wearable.enabled:
        plot_wearable_assessment(
            frequency,
            gain_2d,
            theta_deg,
            phi_deg,
            body_positions=wearable.body_positions,
            tx_power_mw=wearable.tx_power_mw,
            num_devices=wearable.num_devices,
            room_size=wearable.room_size,
            data_label=data_label,
            data_unit=data_unit,
            save_path=save_path,
        )
