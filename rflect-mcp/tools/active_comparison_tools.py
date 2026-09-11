"""Compare active power cuts without changing their absolute power scale."""

from pathlib import Path
from typing import Any

import numpy as np
from matplotlib.figure import Figure

from .import_tools import get_loaded_measurements

COLORS = ("#0072B2", "#D55E00", "#009E73", "#CC79A7", "#E69F00", "#56B4E9")


def compare_active_overlay(
    measurement_names: list[str],
    output_path: str,
    polarization: str = "total",
    plane: str = "azimuth",
    cut_angle_deg: float = 90.0,
    max_envelope: bool = False,
    labels: list[str] | None = None,
) -> dict[str, Any]:
    """Export 2-6 active measurements as a PNG on one absolute dBm scale.

    Azimuth varies phi at fixed theta; elevation varies theta at fixed phi.
    The requested cut must be present in every measurement. No interpolation,
    normalization, or cross-frequency comparison is performed. Generic labels
    keep source filenames out of the image unless labels are supplied explicitly.
    """
    if not 2 <= len(measurement_names) <= 6 or len(set(measurement_names)) != len(
        measurement_names
    ):
        raise ValueError("Select 2-6 distinct active measurements")
    if polarization not in {"total", "hpol", "vpol"}:
        raise ValueError("polarization must be total, hpol, or vpol")
    if plane not in {"azimuth", "elevation"} or not np.isfinite(cut_angle_deg):
        raise ValueError("Select azimuth or elevation and a finite cut angle")
    if labels is not None and (
        len(labels) != len(measurement_names) or any(not label.strip() for label in labels)
    ):
        raise ValueError("Provide one nonempty label per measurement")
    path = Path(output_path)
    if path.suffix.lower() != ".png":
        raise ValueError("output_path must end in .png")
    measurements = get_loaded_measurements()
    traces = []
    frequencies = []
    key, trp_key = {
        "total": ("total_power_2d", "TRP_dBm"),
        "hpol": ("h_power_2d", "h_TRP_dBm"),
        "vpol": ("v_power_2d", "v_TRP_dBm"),
    }[polarization]
    for index, name in enumerate(measurement_names):
        measurement = measurements.get(name)
        if measurement is None or measurement.scan_type != "active":
            raise ValueError("Every selected measurement must be loaded active data")
        if len(measurement.frequencies) != 1 or not np.isfinite(measurement.frequencies[0]):
            raise ValueError("Each active measurement must have one finite frequency")
        frequencies.append(measurement.frequencies[0])
        data = measurement.data
        theta = np.asarray(data.get("theta", []), dtype=float)
        phi = np.asarray(data.get("phi", []), dtype=float)
        power = np.asarray(data.get(key, []), dtype=float)
        if theta.ndim != 1 or phi.ndim != 1 or power.shape != (theta.size, phi.size):
            raise ValueError("Active data must contain a power grid matching theta and phi")
        if not all(np.all(np.isfinite(value)) for value in (theta, phi, power)):
            raise ValueError("Active power grids and angles must contain finite values")
        fixed, angles = (theta, phi) if plane == "azimuth" else (phi, theta)
        matches = np.flatnonzero(np.isclose(fixed, cut_angle_deg, atol=1e-6, rtol=0))
        if not matches.size or angles.size < 2 or np.any(np.diff(angles) <= 0):
            raise ValueError("Requested cut must exist with strictly increasing sample angles")
        cut = power[matches[0], :] if plane == "azimuth" else power[:, matches[0]]
        label = labels[index] if labels else f"Measurement {index + 1}"
        trp = data.get(trp_key)
        if isinstance(trp, (int, float)) and np.isfinite(trp):
            label += f" (TRP {trp:.2f} dBm)"
        traces.append((angles, cut, label))
    if not np.allclose(frequencies, frequencies[0], atol=1e-6, rtol=0):
        raise ValueError("Compare measurements at the same frequency")
    envelope = None
    if max_envelope:
        reference = traces[0][0]
        if any(
            angles.shape != reference.shape or not np.allclose(angles, reference, atol=1e-6, rtol=0)
            for angles, _, _ in traces
        ):
            raise ValueError("Maximum envelope requires matching angular grids")
        envelope = np.max([cut for _, cut, _ in traces], axis=0)

    low = float(np.floor(min(np.min(cut) for _, cut, _ in traces) / 5) * 5)
    high = float(np.ceil(max(np.max(cut) for _, cut, _ in traces) / 5) * 5)
    if high <= low:
        high = low + 5
    figure = Figure(figsize=(8, 7), layout="constrained")
    axis = figure.add_subplot(111, projection="polar")
    for index, (angles, cut, label) in enumerate(traces):
        axis.plot(np.deg2rad(angles), cut, color=COLORS[index], label=label, linewidth=1.8)
    if envelope is not None:
        axis.plot(
            np.deg2rad(traces[0][0]),
            envelope,
            color="#333333",
            linestyle="--",
            label="Maximum envelope",
        )
    axis.set_ylim(low, high)
    axis.set_theta_zero_location("N")
    axis.set_theta_direction(-1)
    if plane == "elevation":
        axis.set_thetamax(180)
    axis.set_title(
        f"{polarization.upper()} power at {frequencies[0]:g} MHz\n{plane.capitalize()} cut at {cut_angle_deg:g} degrees, dBm",
        pad=24,
    )
    axis.legend(loc="upper left", bbox_to_anchor=(1.05, 1.0))
    path.parent.mkdir(parents=True, exist_ok=True)
    # Exclusive creation preserves an existing report or figure.
    with path.open("xb") as output:
        figure.savefig(output, format="png", dpi=160, bbox_inches="tight", pad_inches=0.25)
    return {
        "output_path": str(path.resolve()),
        "frequency_mhz": frequencies[0],
        "radial_limits_dbm": [low, high],
        "trace_count": len(traces),
        "max_envelope": max_envelope,
    }


def register_active_comparison_tools(mcp):
    mcp.tool()(compare_active_overlay)
