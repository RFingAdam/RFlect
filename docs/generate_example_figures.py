"""Generate example figures for the v6.0 RF-method docs (#27).

Reproducible, headless (Agg) generator — run from the repo root:

    python docs/generate_example_figures.py

It writes PNGs into docs/assets/screenshots/ using the same pure RF functions the
MCP tools call, so the figures are authentic outputs (not mock-ups). Re-run after
changing the methods to refresh the docs gallery.
"""

from __future__ import annotations

import os
import sys

# Allow running directly from the repo root (`python docs/generate_example_figures.py`).
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import matplotlib

matplotlib.use("Agg")

import numpy as np
from matplotlib.figure import Figure
from matplotlib.backends.backend_agg import FigureCanvasAgg

from plot_antenna.rf_methods import array_factor_uniform, axial_ratio_from_hv

OUT_DIR = os.path.join(os.path.dirname(__file__), "assets", "screenshots")


def _save(fig, name):
    os.makedirs(OUT_DIR, exist_ok=True)
    path = os.path.join(OUT_DIR, name)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    return path


def fig_array_factor_steering():
    """Uniform linear array factor at several steer angles (#43)."""
    fig = Figure(figsize=(8, 5))
    FigureCanvasAgg(fig)
    ax = fig.add_subplot(111)
    for steer in (0.0, 20.0, 45.0):
        r = array_factor_uniform(n_elements=8, spacing_lambda=0.5, steer_deg=steer)
        ax.plot(
            r["theta_deg"],
            r["af_db"],
            label=f"steer {steer:.0f}° (main {r['main_beam_deg']:.0f}°, HPBW {r['hpbw_deg']:.0f}°)",
        )
    ax.set_xlabel("Angle from broadside (°)")
    ax.set_ylabel("Normalized |AF| (dB)")
    ax.set_title("Uniform Linear Array (8 elements, 0.5 λ) — Electronic Steering")
    ax.set_ylim(-40, 2)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)
    return _save(fig, "example_array_factor.png")


def fig_array_taper_compare():
    """Uniform vs Hamming taper sidelobe trade (#43)."""
    fig = Figure(figsize=(8, 5))
    FigureCanvasAgg(fig)
    ax = fig.add_subplot(111)
    for taper in ("uniform", "hamming"):
        r = array_factor_uniform(n_elements=16, spacing_lambda=0.5, taper=taper)
        ax.plot(r["theta_deg"], r["af_db"], label=f"{taper} (SLL {r['peak_sll_db']} dB)")
    ax.set_xlabel("Angle from broadside (°)")
    ax.set_ylabel("Normalized |AF| (dB)")
    ax.set_title("Aperture Taper Trade-off (16 elements, 0.5 λ)")
    ax.set_ylim(-60, 2)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)
    return _save(fig, "example_array_taper.png")


def fig_axial_ratio_sweep():
    """Axial ratio vs H/V phase difference at equal amplitude (#42)."""
    fig = Figure(figsize=(8, 5))
    FigureCanvasAgg(fig)
    ax = fig.add_subplot(111)
    phases = np.linspace(0, 180, 181)
    ar = [axial_ratio_from_hv(1.0, 1.0, p)["axial_ratio_db"] for p in phases]
    ar = [min(a, 40.0) for a in ar]  # clip the linear-pol asymptote for display
    ax.plot(phases, ar)
    ax.axhline(3.0, color="r", ls="--", alpha=0.6, label="3 dB CP threshold")
    ax.axvline(90.0, color="g", ls=":", alpha=0.6, label="90° → circular")
    ax.set_xlabel("H/V phase difference (°)")
    ax.set_ylabel("Axial ratio (dB)")
    ax.set_title("Axial Ratio vs Phase (equal H/V amplitude)")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)
    return _save(fig, "example_axial_ratio.png")


def main():
    paths = [
        fig_array_factor_steering(),
        fig_array_taper_compare(),
        fig_axial_ratio_sweep(),
    ]
    for p in paths:
        print(f"wrote {p}")


if __name__ == "__main__":
    main()
