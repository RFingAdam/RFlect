"""
DialogsMixin - Dialog-related methods for RFlect GUI

This mixin provides all dialog-related functionality:
- About dialog
- Scan type settings dialogs (Active, Passive, VSWR)
"""

from __future__ import annotations

import os
import datetime
import webbrowser
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from typing import TYPE_CHECKING, Optional

from ..config import (
    DARK_BG_COLOR,
    LIGHT_TEXT_COLOR,
    ACCENT_BLUE_COLOR,
    BUTTON_COLOR,
    SURFACE_COLOR,
    SECTION_HEADER_FONT,
)
from ..calculations import PROTOCOL_PRESETS, ENVIRONMENT_PRESETS

# Import utility functions
from .utils import resource_path

if TYPE_CHECKING:
    from .base_protocol import AntennaPlotGUIProtocol


class DialogsMixin:
    """Mixin class providing dialog methods for AntennaPlotGUI.

    Type hints reference AntennaPlotGUIProtocol for IDE support.
    At runtime, this mixin is combined with the main GUI class.
    """

    # Type hints for IDE support (these are satisfied by the main class)
    root: tk.Tk
    scan_type: tk.StringVar
    passive_scan_type: tk.StringVar
    interpolate_3d_plots: bool
    axis_scale_mode_total: tk.StringVar
    axis_min_total: tk.DoubleVar
    axis_max_total: tk.DoubleVar
    axis_scale_mode_hpol: tk.StringVar
    axis_min_hpol: tk.DoubleVar
    axis_max_hpol: tk.DoubleVar
    axis_scale_mode_vpol: tk.StringVar
    axis_min_vpol: tk.DoubleVar
    axis_max_vpol: tk.DoubleVar
    datasheet_plots_var: tk.BooleanVar
    min_max_eff_gain_var: tk.BooleanVar
    min_max_vswr_var: tk.BooleanVar
    cb_groupdelay_sff_var: tk.BooleanVar
    ecc_analysis_enabled: bool
    shadowing_enabled: bool
    shadow_direction: str
    saved_limit1_freq1: float
    saved_limit1_freq2: float
    saved_limit1_start: float
    saved_limit1_stop: float
    saved_limit2_freq1: float
    saved_limit2_freq2: float
    saved_limit2_start: float
    saved_limit2_stop: float
    saved_min_max_vswr: bool
    cb_groupdelay_sff: bool
    CURRENT_VERSION: str

    hpol_file_path: Optional[str]
    vpol_file_path: Optional[str]

    # Method declarations for type checking only (not defined at runtime to avoid MRO conflicts)
    if TYPE_CHECKING:

        def resource_path(self, relative_path: str) -> str: ...
        def get_user_data_dir(self) -> str: ...
        def update_visibility(self) -> None: ...
        def _run_extrapolation(self, target_frequency: float) -> None: ...

    # ────────────────────────────────────────────────────────────────────────
    # CONDUCTED POWER CSV BROWSE
    # ────────────────────────────────────────────────────────────────────────

    def _browse_conducted_power_csv(self):
        """Open file dialog to select a CSV with per-frequency conducted power."""
        path = filedialog.askopenfilename(
            title="Select Conducted Power CSV",
            filetypes=[("CSV files", "*.csv"), ("All files", "*.*")],
        )
        if path:
            self.conducted_power_csv_path.set(path)

    # ────────────────────────────────────────────────────────────────────────
    # ABOUT DIALOG
    # ────────────────────────────────────────────────────────────────────────

    def show_about_dialog(self):
        """Show the About RFlect dialog with version and credits."""
        about_window = tk.Toplevel(self.root)
        about_window.title("About RFlect")
        about_window.geometry("500x400")
        about_window.resizable(False, False)
        about_window.configure(bg=DARK_BG_COLOR)

        # Center the window
        about_window.transient(self.root)
        about_window.grab_set()

        # Header Frame with Logo and Name
        header_frame = tk.Frame(about_window, bg=DARK_BG_COLOR)
        header_frame.pack(pady=(20, 10))

        # Logo (smith_logo.png)
        try:
            logo_path = resource_path(os.path.join("assets", "smith_logo.png"))
            if os.path.exists(logo_path):
                from PIL import Image, ImageTk

                logo_img = Image.open(logo_path)
                logo_img = logo_img.resize((80, 80), Image.Resampling.LANCZOS)
                logo_photo = ImageTk.PhotoImage(logo_img)
                logo_label = tk.Label(header_frame, image=logo_photo, bg=DARK_BG_COLOR)
                logo_label.image = logo_photo  # type: ignore # Keep a reference
                logo_label.pack(side=tk.LEFT, padx=(0, 15))
        except (FileNotFoundError, ImportError, OSError) as e:
            print(f"[INFO] Could not load logo: {e}")
            pass  # No logo, that's okay

        # App Name (red color to match logo)
        name_label = tk.Label(
            header_frame,
            text="RFlect",
            font=("Arial", 28, "bold"),
            bg=DARK_BG_COLOR,
            fg="#E63946",  # Red color similar to smith_logo.png
        )
        name_label.pack(side=tk.LEFT)

        # Version
        version_label = tk.Label(
            about_window,
            text=f"Version {self.CURRENT_VERSION}",
            font=("Arial", 12),
            bg=DARK_BG_COLOR,
            fg=LIGHT_TEXT_COLOR,
        )
        version_label.pack()

        # Description
        desc_label = tk.Label(
            about_window,
            text="Antenna Measurement & Analysis Tool",
            font=("Arial", 10, "italic"),
            bg=DARK_BG_COLOR,
            fg=LIGHT_TEXT_COLOR,
        )
        desc_label.pack(pady=(5, 20))

        # Credits
        credits_frame = tk.Frame(about_window, bg=DARK_BG_COLOR)
        credits_frame.pack(pady=10)

        credits_text = """
Developed by: Adam Engelbrecht

Features:
- Active & Passive Antenna Measurements
- 2D & 3D Radiation Pattern Visualization
- Polarization Analysis (AR, Tilt, Sense, XPD)
- AI-Powered Report Generation
- Group Delay & Fidelity Analysis
- VSWR & S-Parameter Analysis
"""

        credits_label = tk.Label(
            credits_frame,
            text=credits_text,
            font=("Arial", 9),
            bg=DARK_BG_COLOR,
            fg=LIGHT_TEXT_COLOR,
            justify=tk.LEFT,
        )
        credits_label.pack()

        # License info
        license_label = tk.Label(
            about_window,
            text="Licensed under GNU General Public License v3.0",
            font=("Arial", 8),
            bg=DARK_BG_COLOR,
            fg=LIGHT_TEXT_COLOR,
        )
        license_label.pack(pady=(5, 0))

        # Links
        links_frame = tk.Frame(about_window, bg=DARK_BG_COLOR)
        links_frame.pack(pady=10)

        github_btn = tk.Button(
            links_frame,
            text="View on GitHub",
            command=lambda: webbrowser.open("https://github.com/RFingAdam/RFlect"),
            bg=ACCENT_BLUE_COLOR,
            fg=LIGHT_TEXT_COLOR,
            relief=tk.FLAT,
            padx=10,
            pady=5,
        )
        github_btn.pack(side=tk.LEFT, padx=5)

        license_btn = tk.Button(
            links_frame,
            text="View License",
            command=lambda: webbrowser.open(
                "https://github.com/RFingAdam/RFlect/blob/main/LICENSE"
            ),
            bg=ACCENT_BLUE_COLOR,
            fg=LIGHT_TEXT_COLOR,
            relief=tk.FLAT,
            padx=10,
            pady=5,
        )
        license_btn.pack(side=tk.LEFT, padx=5)

        # Close button
        close_btn = tk.Button(
            about_window,
            text="Close",
            command=about_window.destroy,
            bg=ACCENT_BLUE_COLOR,
            fg=LIGHT_TEXT_COLOR,
            width=10,
        )
        close_btn.pack(pady=20)

    # ────────────────────────────────────────────────────────────────────────
    # API KEY MANAGEMENT
    # ────────────────────────────────────────────────────────────────────────

    def _cleanup_advanced_analysis_traces(self):
        """Remove variable trace handlers registered by the advanced settings UI."""
        handles = getattr(self, "_advanced_trace_handles", [])
        for var, mode, trace_id in handles:
            try:
                var.trace_remove(mode, trace_id)
            except Exception:
                # Ignore stale handles; this cleanup is best-effort.
                pass
        self._advanced_trace_handles = []

    def _build_advanced_analysis_frames(self, parent, start_row):
        """Build all advanced analysis LabelFrame sections.

        Returns the next available row index and a callback to read values.
        """
        # Ensure we don't accumulate callbacks if settings is reopened repeatedly.
        self._cleanup_advanced_analysis_traces()
        self._advanced_trace_handles = []
        row = start_row

        # ── Link Budget / Range Estimation ──
        lb_frame = tk.LabelFrame(
            parent,
            text="Link Budget / Range Estimation",
            bg=DARK_BG_COLOR,
            fg=ACCENT_BLUE_COLOR,
            font=SECTION_HEADER_FONT,
        )
        lb_frame.grid(row=row, column=0, columnspan=4, sticky="ew", padx=15, pady=5)
        row += 1

        self._cb_link_budget_var = tk.BooleanVar(value=getattr(self, "link_budget_enabled", False))
        tk.Checkbutton(
            lb_frame,
            text="Enable Link Budget Analysis",
            variable=self._cb_link_budget_var,
            bg=DARK_BG_COLOR,
            fg=LIGHT_TEXT_COLOR,
            selectcolor=SURFACE_COLOR,
            activebackground=DARK_BG_COLOR,
            activeforeground=LIGHT_TEXT_COLOR,
        ).grid(row=0, column=0, columnspan=4, sticky=tk.W, padx=5, pady=2)

        # Protocol preset dropdown
        tk.Label(lb_frame, text="Protocol:", bg=DARK_BG_COLOR, fg=LIGHT_TEXT_COLOR).grid(
            row=1, column=0, sticky=tk.W, padx=5
        )
        protocol_options = list(PROTOCOL_PRESETS.keys())
        self._lb_protocol_menu = ttk.Combobox(
            lb_frame,
            textvariable=self.lb_protocol_preset,
            values=protocol_options,
            width=22,
            state="readonly",
        )
        self._lb_protocol_menu.grid(row=1, column=1, columnspan=2, sticky=tk.W, padx=5)

        def _on_protocol_change(*_args):
            preset = self.lb_protocol_preset.get()
            if preset in PROTOCOL_PRESETS and preset != "Custom":
                sens, pwr, _freq = PROTOCOL_PRESETS[preset]
                if sens is not None:
                    self.lb_rx_sensitivity.set(sens)
                if pwr is not None:
                    self.lb_tx_power.set(pwr)

        trace_id = self.lb_protocol_preset.trace_add("write", _on_protocol_change)
        self._advanced_trace_handles.append((self.lb_protocol_preset, "write", trace_id))

        # Tx Power
        tk.Label(lb_frame, text="Tx Power (dBm):", bg=DARK_BG_COLOR, fg=LIGHT_TEXT_COLOR).grid(
            row=2, column=0, sticky=tk.W, padx=5
        )
        tk.Entry(
            lb_frame,
            textvariable=self.lb_tx_power,
            width=8,
            bg=SURFACE_COLOR,
            fg=LIGHT_TEXT_COLOR,
            insertbackground=LIGHT_TEXT_COLOR,
        ).grid(row=2, column=1, padx=5)

        # Rx Sensitivity
        tk.Label(
            lb_frame, text="Rx Sensitivity (dBm):", bg=DARK_BG_COLOR, fg=LIGHT_TEXT_COLOR
        ).grid(row=2, column=2, sticky=tk.W, padx=5)
        tk.Entry(
            lb_frame,
            textvariable=self.lb_rx_sensitivity,
            width=8,
            bg=SURFACE_COLOR,
            fg=LIGHT_TEXT_COLOR,
            insertbackground=LIGHT_TEXT_COLOR,
        ).grid(row=2, column=3, padx=5)

        # Rx Gain
        tk.Label(lb_frame, text="Rx Gain (dBi):", bg=DARK_BG_COLOR, fg=LIGHT_TEXT_COLOR).grid(
            row=3, column=0, sticky=tk.W, padx=5
        )
        tk.Entry(
            lb_frame,
            textvariable=self.lb_rx_gain,
            width=8,
            bg=SURFACE_COLOR,
            fg=LIGHT_TEXT_COLOR,
            insertbackground=LIGHT_TEXT_COLOR,
        ).grid(row=3, column=1, padx=5)

        # Path loss exponent
        tk.Label(lb_frame, text="Path Loss Exp (n):", bg=DARK_BG_COLOR, fg=LIGHT_TEXT_COLOR).grid(
            row=3, column=2, sticky=tk.W, padx=5
        )
        tk.Entry(
            lb_frame,
            textvariable=self.lb_path_loss_exp,
            width=8,
            bg=SURFACE_COLOR,
            fg=LIGHT_TEXT_COLOR,
            insertbackground=LIGHT_TEXT_COLOR,
        ).grid(row=3, column=3, padx=5)

        # Misc loss + target range
        tk.Label(lb_frame, text="Misc Loss (dB):", bg=DARK_BG_COLOR, fg=LIGHT_TEXT_COLOR).grid(
            row=4, column=0, sticky=tk.W, padx=5
        )
        tk.Entry(
            lb_frame,
            textvariable=self.lb_misc_loss,
            width=8,
            bg=SURFACE_COLOR,
            fg=LIGHT_TEXT_COLOR,
            insertbackground=LIGHT_TEXT_COLOR,
        ).grid(row=4, column=1, padx=5)

        tk.Label(lb_frame, text="Target Range (m):", bg=DARK_BG_COLOR, fg=LIGHT_TEXT_COLOR).grid(
            row=4, column=2, sticky=tk.W, padx=5
        )
        tk.Entry(
            lb_frame,
            textvariable=self.lb_target_range,
            width=8,
            bg=SURFACE_COLOR,
            fg=LIGHT_TEXT_COLOR,
            insertbackground=LIGHT_TEXT_COLOR,
        ).grid(row=4, column=3, padx=5)

        # ── Indoor Propagation ──
        indoor_frame = tk.LabelFrame(
            parent,
            text="Indoor Propagation",
            bg=DARK_BG_COLOR,
            fg=ACCENT_BLUE_COLOR,
            font=SECTION_HEADER_FONT,
        )
        indoor_frame.grid(row=row, column=0, columnspan=4, sticky="ew", padx=15, pady=5)
        row += 1

        self._cb_indoor_var = tk.BooleanVar(value=getattr(self, "indoor_analysis_enabled", False))
        tk.Checkbutton(
            indoor_frame,
            text="Enable Indoor Analysis",
            variable=self._cb_indoor_var,
            bg=DARK_BG_COLOR,
            fg=LIGHT_TEXT_COLOR,
            selectcolor=SURFACE_COLOR,
            activebackground=DARK_BG_COLOR,
            activeforeground=LIGHT_TEXT_COLOR,
        ).grid(row=0, column=0, columnspan=4, sticky=tk.W, padx=5, pady=2)

        # Environment dropdown
        tk.Label(indoor_frame, text="Environment:", bg=DARK_BG_COLOR, fg=LIGHT_TEXT_COLOR).grid(
            row=1, column=0, sticky=tk.W, padx=5
        )
        env_options = list(ENVIRONMENT_PRESETS.keys())
        self._indoor_env_menu = ttk.Combobox(
            indoor_frame,
            textvariable=self.indoor_environment,
            values=env_options,
            width=18,
            state="readonly",
        )
        self._indoor_env_menu.grid(row=1, column=1, sticky=tk.W, padx=5)

        def _on_env_change(*_args):
            env = self.indoor_environment.get()
            if env in ENVIRONMENT_PRESETS:
                n, sigma, fading_m, k, walls = ENVIRONMENT_PRESETS[env]
                self.indoor_path_loss_exp.set(n)
                self.indoor_shadow_fading.set(sigma)
                self.indoor_num_walls.set(walls)
                if fading_m != "none":
                    self.fading_model.set(fading_m)
                    if k > 0:
                        self.fading_rician_k.set(float(k))

        trace_id = self.indoor_environment.trace_add("write", _on_env_change)
        self._advanced_trace_handles.append((self.indoor_environment, "write", trace_id))

        tk.Label(
            indoor_frame, text="Path Loss Exp (n):", bg=DARK_BG_COLOR, fg=LIGHT_TEXT_COLOR
        ).grid(row=1, column=2, sticky=tk.W, padx=5)
        tk.Entry(
            indoor_frame,
            textvariable=self.indoor_path_loss_exp,
            width=8,
            bg=SURFACE_COLOR,
            fg=LIGHT_TEXT_COLOR,
            insertbackground=LIGHT_TEXT_COLOR,
        ).grid(row=1, column=3, padx=5)

        # Walls + material
        tk.Label(indoor_frame, text="Walls:", bg=DARK_BG_COLOR, fg=LIGHT_TEXT_COLOR).grid(
            row=2, column=0, sticky=tk.W, padx=5
        )
        tk.Spinbox(
            indoor_frame,
            textvariable=self.indoor_num_walls,
            from_=0,
            to=10,
            width=4,
            bg=SURFACE_COLOR,
            fg=LIGHT_TEXT_COLOR,
        ).grid(row=2, column=1, sticky=tk.W, padx=5)

        tk.Label(indoor_frame, text="Material:", bg=DARK_BG_COLOR, fg=LIGHT_TEXT_COLOR).grid(
            row=2, column=2, sticky=tk.W, padx=5
        )
        wall_options = ["drywall", "wood", "glass", "brick", "concrete", "metal"]
        ttk.Combobox(
            indoor_frame,
            textvariable=self.indoor_wall_material,
            values=wall_options,
            width=12,
            state="readonly",
        ).grid(row=2, column=3, sticky=tk.W, padx=5)

        # Shadow fading + max distance
        tk.Label(indoor_frame, text="Shadow σ (dB):", bg=DARK_BG_COLOR, fg=LIGHT_TEXT_COLOR).grid(
            row=3, column=0, sticky=tk.W, padx=5
        )
        tk.Entry(
            indoor_frame,
            textvariable=self.indoor_shadow_fading,
            width=8,
            bg=SURFACE_COLOR,
            fg=LIGHT_TEXT_COLOR,
            insertbackground=LIGHT_TEXT_COLOR,
        ).grid(row=3, column=1, padx=5)

        tk.Label(
            indoor_frame, text="Max Distance (m):", bg=DARK_BG_COLOR, fg=LIGHT_TEXT_COLOR
        ).grid(row=3, column=2, sticky=tk.W, padx=5)
        tk.Entry(
            indoor_frame,
            textvariable=self.indoor_max_distance,
            width=8,
            bg=SURFACE_COLOR,
            fg=LIGHT_TEXT_COLOR,
            insertbackground=LIGHT_TEXT_COLOR,
        ).grid(row=3, column=3, padx=5)

        # ── Multipath Fading ──
        fading_frame = tk.LabelFrame(
            parent,
            text="Multipath Fading",
            bg=DARK_BG_COLOR,
            fg=ACCENT_BLUE_COLOR,
            font=SECTION_HEADER_FONT,
        )
        fading_frame.grid(row=row, column=0, columnspan=4, sticky="ew", padx=15, pady=5)
        row += 1

        self._cb_fading_var = tk.BooleanVar(value=getattr(self, "fading_analysis_enabled", False))
        tk.Checkbutton(
            fading_frame,
            text="Enable Fading Analysis",
            variable=self._cb_fading_var,
            bg=DARK_BG_COLOR,
            fg=LIGHT_TEXT_COLOR,
            selectcolor=SURFACE_COLOR,
            activebackground=DARK_BG_COLOR,
            activeforeground=LIGHT_TEXT_COLOR,
        ).grid(row=0, column=0, columnspan=4, sticky=tk.W, padx=5, pady=2)

        tk.Label(fading_frame, text="Model:", bg=DARK_BG_COLOR, fg=LIGHT_TEXT_COLOR).grid(
            row=1, column=0, sticky=tk.W, padx=5
        )
        ttk.Combobox(
            fading_frame,
            textvariable=self.fading_model,
            values=["rayleigh", "rician"],
            width=12,
            state="readonly",
        ).grid(row=1, column=1, sticky=tk.W, padx=5)

        tk.Label(fading_frame, text="K-factor:", bg=DARK_BG_COLOR, fg=LIGHT_TEXT_COLOR).grid(
            row=1, column=2, sticky=tk.W, padx=5
        )
        tk.Entry(
            fading_frame,
            textvariable=self.fading_rician_k,
            width=8,
            bg=SURFACE_COLOR,
            fg=LIGHT_TEXT_COLOR,
            insertbackground=LIGHT_TEXT_COLOR,
        ).grid(row=1, column=3, padx=5)

        tk.Label(
            fading_frame, text="Target Reliability (%):", bg=DARK_BG_COLOR, fg=LIGHT_TEXT_COLOR
        ).grid(row=2, column=0, columnspan=2, sticky=tk.W, padx=5)
        tk.Entry(
            fading_frame,
            textvariable=self.fading_target_reliability,
            width=8,
            bg=SURFACE_COLOR,
            fg=LIGHT_TEXT_COLOR,
            insertbackground=LIGHT_TEXT_COLOR,
        ).grid(row=2, column=2, padx=5)

        tk.Label(fading_frame, text="Realizations:", bg=DARK_BG_COLOR, fg=LIGHT_TEXT_COLOR).grid(
            row=3, column=0, sticky=tk.W, padx=5
        )
        tk.Spinbox(
            fading_frame,
            textvariable=self.fading_realizations,
            from_=50,
            to=10000,
            increment=50,
            width=8,
            bg=SURFACE_COLOR,
            fg=LIGHT_TEXT_COLOR,
        ).grid(row=3, column=1, sticky=tk.W, padx=5)

        # ── MIMO / Diversity ──
        mimo_frame = tk.LabelFrame(
            parent,
            text="MIMO / Diversity",
            bg=DARK_BG_COLOR,
            fg=ACCENT_BLUE_COLOR,
            font=SECTION_HEADER_FONT,
        )
        mimo_frame.grid(row=row, column=0, columnspan=4, sticky="ew", padx=15, pady=5)
        row += 1

        self._cb_mimo_var = tk.BooleanVar(value=getattr(self, "mimo_analysis_enabled", False))
        tk.Checkbutton(
            mimo_frame,
            text="Enable MIMO Analysis",
            variable=self._cb_mimo_var,
            bg=DARK_BG_COLOR,
            fg=LIGHT_TEXT_COLOR,
            selectcolor=SURFACE_COLOR,
            activebackground=DARK_BG_COLOR,
            activeforeground=LIGHT_TEXT_COLOR,
        ).grid(row=0, column=0, columnspan=4, sticky=tk.W, padx=5, pady=2)

        tk.Label(mimo_frame, text="SNR (dB):", bg=DARK_BG_COLOR, fg=LIGHT_TEXT_COLOR).grid(
            row=1, column=0, sticky=tk.W, padx=5
        )
        tk.Entry(
            mimo_frame,
            textvariable=self.mimo_snr,
            width=8,
            bg=SURFACE_COLOR,
            fg=LIGHT_TEXT_COLOR,
            insertbackground=LIGHT_TEXT_COLOR,
        ).grid(row=1, column=1, padx=5)

        tk.Label(mimo_frame, text="Fading:", bg=DARK_BG_COLOR, fg=LIGHT_TEXT_COLOR).grid(
            row=1, column=2, sticky=tk.W, padx=5
        )
        ttk.Combobox(
            mimo_frame,
            textvariable=self.mimo_fading_model,
            values=["rayleigh", "rician"],
            width=12,
            state="readonly",
        ).grid(row=1, column=3, sticky=tk.W, padx=5)

        tk.Label(mimo_frame, text="K-factor:", bg=DARK_BG_COLOR, fg=LIGHT_TEXT_COLOR).grid(
            row=2, column=0, sticky=tk.W, padx=5
        )
        tk.Entry(
            mimo_frame,
            textvariable=self.mimo_rician_k,
            width=8,
            bg=SURFACE_COLOR,
            fg=LIGHT_TEXT_COLOR,
            insertbackground=LIGHT_TEXT_COLOR,
        ).grid(row=2, column=1, padx=5)

        tk.Label(mimo_frame, text="XPR (dB):", bg=DARK_BG_COLOR, fg=LIGHT_TEXT_COLOR).grid(
            row=2, column=2, sticky=tk.W, padx=5
        )
        tk.Entry(
            mimo_frame,
            textvariable=self.mimo_xpr,
            width=8,
            bg=SURFACE_COLOR,
            fg=LIGHT_TEXT_COLOR,
            insertbackground=LIGHT_TEXT_COLOR,
        ).grid(row=2, column=3, padx=5)

        # ── Wearable / Medical ──
        wear_frame = tk.LabelFrame(
            parent,
            text="Wearable / Medical",
            bg=DARK_BG_COLOR,
            fg=ACCENT_BLUE_COLOR,
            font=SECTION_HEADER_FONT,
        )
        wear_frame.grid(row=row, column=0, columnspan=4, sticky="ew", padx=15, pady=5)
        row += 1

        self._cb_wearable_var = tk.BooleanVar(
            value=getattr(self, "wearable_analysis_enabled", False)
        )
        tk.Checkbutton(
            wear_frame,
            text="Enable Wearable Assessment",
            variable=self._cb_wearable_var,
            bg=DARK_BG_COLOR,
            fg=LIGHT_TEXT_COLOR,
            selectcolor=SURFACE_COLOR,
            activebackground=DARK_BG_COLOR,
            activeforeground=LIGHT_TEXT_COLOR,
        ).grid(row=0, column=0, columnspan=4, sticky=tk.W, padx=5, pady=2)

        # Body positions checkboxes
        pos_frame = tk.Frame(wear_frame, bg=DARK_BG_COLOR)
        pos_frame.grid(row=1, column=0, columnspan=4, sticky=tk.W, padx=5)
        for i, (pos, var) in enumerate(self.wearable_positions_var.items()):
            tk.Checkbutton(
                pos_frame,
                text=pos.capitalize(),
                variable=var,
                bg=DARK_BG_COLOR,
                fg=LIGHT_TEXT_COLOR,
                selectcolor=SURFACE_COLOR,
                activebackground=DARK_BG_COLOR,
                activeforeground=LIGHT_TEXT_COLOR,
            ).pack(side=tk.LEFT, padx=8)

        tk.Label(wear_frame, text="Tx Power (mW):", bg=DARK_BG_COLOR, fg=LIGHT_TEXT_COLOR).grid(
            row=2, column=0, sticky=tk.W, padx=5
        )
        tk.Entry(
            wear_frame,
            textvariable=self.wearable_tx_power_mw,
            width=8,
            bg=SURFACE_COLOR,
            fg=LIGHT_TEXT_COLOR,
            insertbackground=LIGHT_TEXT_COLOR,
        ).grid(row=2, column=1, padx=5)

        tk.Label(wear_frame, text="Nearby Devices:", bg=DARK_BG_COLOR, fg=LIGHT_TEXT_COLOR).grid(
            row=2, column=2, sticky=tk.W, padx=5
        )
        tk.Spinbox(
            wear_frame,
            textvariable=self.wearable_device_count,
            from_=1,
            to=100,
            width=5,
            bg=SURFACE_COLOR,
            fg=LIGHT_TEXT_COLOR,
        ).grid(row=2, column=3, sticky=tk.W, padx=5)

        # Room size
        room_frame = tk.Frame(wear_frame, bg=DARK_BG_COLOR)
        room_frame.grid(row=3, column=0, columnspan=4, sticky=tk.W, padx=5, pady=2)
        tk.Label(room_frame, text="Room (m):", bg=DARK_BG_COLOR, fg=LIGHT_TEXT_COLOR).pack(
            side=tk.LEFT
        )
        tk.Entry(
            room_frame,
            textvariable=self.wearable_room_x,
            width=5,
            bg=SURFACE_COLOR,
            fg=LIGHT_TEXT_COLOR,
            insertbackground=LIGHT_TEXT_COLOR,
        ).pack(side=tk.LEFT, padx=2)
        tk.Label(room_frame, text="×", bg=DARK_BG_COLOR, fg=LIGHT_TEXT_COLOR).pack(side=tk.LEFT)
        tk.Entry(
            room_frame,
            textvariable=self.wearable_room_y,
            width=5,
            bg=SURFACE_COLOR,
            fg=LIGHT_TEXT_COLOR,
            insertbackground=LIGHT_TEXT_COLOR,
        ).pack(side=tk.LEFT, padx=2)
        tk.Label(room_frame, text="×", bg=DARK_BG_COLOR, fg=LIGHT_TEXT_COLOR).pack(side=tk.LEFT)
        tk.Entry(
            room_frame,
            textvariable=self.wearable_room_z,
            width=5,
            bg=SURFACE_COLOR,
            fg=LIGHT_TEXT_COLOR,
            insertbackground=LIGHT_TEXT_COLOR,
        ).pack(side=tk.LEFT, padx=2)

        return row

    def _save_advanced_analysis_settings(self):
        """Read advanced analysis checkbox values back to self attributes."""
        self.link_budget_enabled = self._cb_link_budget_var.get()
        self.indoor_analysis_enabled = self._cb_indoor_var.get()
        self.fading_analysis_enabled = self._cb_fading_var.get()
        self.mimo_analysis_enabled = self._cb_mimo_var.get()
        self.wearable_analysis_enabled = self._cb_wearable_var.get()

    # ────────────────────────────────────────────────────────────────────────
    # SCAN TYPE SETTINGS DIALOG
    # ────────────────────────────────────────────────────────────────────────

    def show_settings(self):
        """Show settings dialog based on current scan type."""
        scan_type_value = self.scan_type.get()
        outer_window = tk.Toplevel(self.root)
        outer_window.geometry("650x800")
        outer_window.title(f"{scan_type_value.capitalize()} Settings")
        outer_window.configure(bg=DARK_BG_COLOR)
        outer_window.resizable(True, True)

        # Scrollable content area
        _canvas = tk.Canvas(outer_window, bg=DARK_BG_COLOR, highlightthickness=0)
        _scrollbar = ttk.Scrollbar(outer_window, orient="vertical", command=_canvas.yview)
        settings_window = tk.Frame(_canvas, bg=DARK_BG_COLOR)

        settings_window.bind(
            "<Configure>",
            lambda e: _canvas.configure(scrollregion=_canvas.bbox("all")),
        )
        _cw = _canvas.create_window((0, 0), window=settings_window, anchor="nw")
        _canvas.configure(yscrollcommand=_scrollbar.set)
        _canvas.bind("<Configure>", lambda e: _canvas.itemconfig(_cw, width=e.width))

        def _on_mousewheel(event):
            try:
                _canvas.yview_scroll(int(-1 * (event.delta / 120)), "units")
            except tk.TclError:
                _canvas.unbind_all("<MouseWheel>")

        def _on_enter(_e):
            _canvas.bind_all("<MouseWheel>", _on_mousewheel)

        def _on_leave(_e):
            _canvas.unbind_all("<MouseWheel>")

        _canvas.bind("<Enter>", _on_enter)
        _canvas.bind("<Leave>", _on_leave)

        def _on_outer_destroy(event):
            if event.widget is outer_window:
                _canvas.unbind_all("<MouseWheel>")
                self._cleanup_advanced_analysis_traces()

        outer_window.bind("<Destroy>", _on_outer_destroy)

        _scrollbar.pack(side="right", fill="y")
        _canvas.pack(side="left", fill="both", expand=True)

        # ────────────────────────────────────
        #  ACTIVE  (TRP) SETTINGS
        # ────────────────────────────────────
        if scan_type_value == "active":
            tk.Label(
                settings_window, text="Active Plot Settings", bg=DARK_BG_COLOR, fg=LIGHT_TEXT_COLOR
            ).grid(row=0, column=0, columnspan=4, pady=20)

            # 3-D interpolation
            self.interpolate_var = tk.BooleanVar(value=self.interpolate_3d_plots)
            tk.Checkbutton(
                settings_window,
                text="Interpolate 3-D Plots",
                variable=self.interpolate_var,
                bg=DARK_BG_COLOR,
                fg=LIGHT_TEXT_COLOR,
                selectcolor=SURFACE_COLOR,
                activebackground=DARK_BG_COLOR,
                activeforeground=LIGHT_TEXT_COLOR,
            ).grid(row=1, column=0, sticky=tk.W, padx=20)

            # Per-type 3D Z-axis scaling
            tk.Label(
                settings_window, text="3-D Z-Axis Scale:", bg=DARK_BG_COLOR, fg=LIGHT_TEXT_COLOR
            ).grid(row=2, column=0, sticky=tk.W, padx=20)
            # Column headers
            for ci, hdr in enumerate(["Mode", "Min", "Max"], start=1):
                tk.Label(
                    settings_window,
                    text=hdr,
                    bg=DARK_BG_COLOR,
                    fg=LIGHT_TEXT_COLOR,
                ).grid(row=2, column=ci, sticky=tk.W)

            _scale_rows = [
                ("Total:", self.axis_scale_mode_total, self.axis_min_total, self.axis_max_total),
                ("H-Pol:", self.axis_scale_mode_hpol, self.axis_min_hpol, self.axis_max_hpol),
                ("V-Pol:", self.axis_scale_mode_vpol, self.axis_min_vpol, self.axis_max_vpol),
            ]
            for idx, (lbl, mode_var, min_var, max_var) in enumerate(_scale_rows):
                r = 3 + idx
                tk.Label(
                    settings_window,
                    text=lbl,
                    bg=DARK_BG_COLOR,
                    fg=LIGHT_TEXT_COLOR,
                ).grid(row=r, column=0, sticky=tk.W, padx=30)
                _mode_frame = tk.Frame(settings_window, bg=DARK_BG_COLOR)
                _mode_frame.grid(row=r, column=1, sticky=tk.W)
                tk.Radiobutton(
                    _mode_frame,
                    text="Auto",
                    variable=mode_var,
                    value="auto",
                    bg=DARK_BG_COLOR,
                    fg=LIGHT_TEXT_COLOR,
                    selectcolor=SURFACE_COLOR,
                    activebackground=DARK_BG_COLOR,
                    activeforeground=LIGHT_TEXT_COLOR,
                ).pack(side=tk.LEFT)
                tk.Radiobutton(
                    _mode_frame,
                    text="Man",
                    variable=mode_var,
                    value="manual",
                    bg=DARK_BG_COLOR,
                    fg=LIGHT_TEXT_COLOR,
                    selectcolor=SURFACE_COLOR,
                    activebackground=DARK_BG_COLOR,
                    activeforeground=LIGHT_TEXT_COLOR,
                ).pack(side=tk.LEFT)
                tk.Entry(
                    settings_window,
                    textvariable=min_var,
                    width=6,
                    bg=SURFACE_COLOR,
                    fg=LIGHT_TEXT_COLOR,
                    insertbackground=LIGHT_TEXT_COLOR,
                ).grid(row=r, column=2)
                tk.Entry(
                    settings_window,
                    textvariable=max_var,
                    width=6,
                    bg=SURFACE_COLOR,
                    fg=LIGHT_TEXT_COLOR,
                    insertbackground=LIGHT_TEXT_COLOR,
                ).grid(row=r, column=3)

            # Maritime / Horizon plot settings
            maritime_frame = tk.LabelFrame(
                settings_window,
                text="Maritime / Horizon Plots",
                bg=DARK_BG_COLOR,
                fg=ACCENT_BLUE_COLOR,
                font=SECTION_HEADER_FONT,
            )
            maritime_frame.grid(row=6, column=0, columnspan=4, sticky="ew", padx=15, pady=5)

            self.cb_maritime_var = tk.BooleanVar(
                value=getattr(self, "maritime_plots_enabled", False)
            )
            tk.Checkbutton(
                maritime_frame,
                text="Enable Maritime Plots",
                variable=self.cb_maritime_var,
                bg=DARK_BG_COLOR,
                fg=LIGHT_TEXT_COLOR,
                selectcolor=SURFACE_COLOR,
                activebackground=DARK_BG_COLOR,
                activeforeground=LIGHT_TEXT_COLOR,
            ).grid(row=0, column=0, columnspan=2, sticky=tk.W, padx=5, pady=2)

            tk.Label(
                maritime_frame, text="Theta Min (°):", bg=DARK_BG_COLOR, fg=LIGHT_TEXT_COLOR
            ).grid(row=1, column=0, sticky=tk.W, padx=5)
            tk.Entry(
                maritime_frame,
                textvariable=self.horizon_theta_min,
                width=6,
                bg=SURFACE_COLOR,
                fg=LIGHT_TEXT_COLOR,
                insertbackground=LIGHT_TEXT_COLOR,
            ).grid(row=1, column=1, padx=5)
            tk.Label(
                maritime_frame, text="Theta Max (°):", bg=DARK_BG_COLOR, fg=LIGHT_TEXT_COLOR
            ).grid(row=1, column=2, sticky=tk.W, padx=5)
            tk.Entry(
                maritime_frame,
                textvariable=self.horizon_theta_max,
                width=6,
                bg=SURFACE_COLOR,
                fg=LIGHT_TEXT_COLOR,
                insertbackground=LIGHT_TEXT_COLOR,
            ).grid(row=1, column=3, padx=5)

            tk.Label(
                maritime_frame,
                text="Coverage Threshold (dB):",
                bg=DARK_BG_COLOR,
                fg=LIGHT_TEXT_COLOR,
            ).grid(row=2, column=0, columnspan=2, sticky=tk.W, padx=5)
            tk.Entry(
                maritime_frame,
                textvariable=self.horizon_gain_threshold,
                width=6,
                bg=SURFACE_COLOR,
                fg=LIGHT_TEXT_COLOR,
                insertbackground=LIGHT_TEXT_COLOR,
            ).grid(row=2, column=2, padx=5)

            tk.Label(
                maritime_frame, text="Theta Cuts (°):", bg=DARK_BG_COLOR, fg=LIGHT_TEXT_COLOR
            ).grid(row=3, column=0, sticky=tk.W, padx=5)
            tk.Entry(
                maritime_frame,
                textvariable=self.horizon_theta_cuts_var,
                width=25,
                bg=SURFACE_COLOR,
                fg=LIGHT_TEXT_COLOR,
                insertbackground=LIGHT_TEXT_COLOR,
            ).grid(row=3, column=1, columnspan=3, sticky=tk.W, padx=5, pady=2)

            tk.Label(
                maritime_frame,
                text="Conducted Power (dBm):",
                bg=DARK_BG_COLOR,
                fg=LIGHT_TEXT_COLOR,
            ).grid(row=4, column=0, columnspan=2, sticky=tk.W, padx=5)
            tk.Entry(
                maritime_frame,
                textvariable=self.conducted_power_dBm,
                width=8,
                bg=SURFACE_COLOR,
                fg=LIGHT_TEXT_COLOR,
                insertbackground=LIGHT_TEXT_COLOR,
            ).grid(row=4, column=2, padx=5, pady=2)
            tk.Label(
                maritime_frame,
                text="(single freq)",
                bg=DARK_BG_COLOR,
                fg="#A0A0A0",
                font=("Arial", 9),
            ).grid(row=4, column=3, sticky=tk.W)

            # CSV file for per-frequency conducted power (batch processing)
            tk.Label(
                maritime_frame,
                text="Per-Freq CSV:",
                bg=DARK_BG_COLOR,
                fg=LIGHT_TEXT_COLOR,
            ).grid(row=5, column=0, sticky=tk.W, padx=5)
            csv_entry = tk.Entry(
                maritime_frame,
                textvariable=self.conducted_power_csv_path,
                width=20,
                bg=SURFACE_COLOR,
                fg=LIGHT_TEXT_COLOR,
                insertbackground=LIGHT_TEXT_COLOR,
            )
            csv_entry.grid(row=5, column=1, columnspan=2, sticky=tk.W, padx=5, pady=2)
            tk.Button(
                maritime_frame,
                text="Browse",
                command=lambda: self._browse_conducted_power_csv(),
                bg=BUTTON_COLOR,
                fg=LIGHT_TEXT_COLOR,
                width=7,
            ).grid(row=5, column=3, padx=2)

            # Advanced analysis settings (Link Budget, Indoor, Fading, Wearable)
            _adv_next_row = self._build_advanced_analysis_frames(settings_window, start_row=7)

            def save_active_settings():
                self.interpolate_3d_plots = self.interpolate_var.get()
                self.maritime_plots_enabled = self.cb_maritime_var.get()
                self._save_advanced_analysis_settings()
                self.update_visibility()
                outer_window.destroy()

            tk.Button(
                settings_window,
                text="Save Settings",
                command=save_active_settings,
                bg=ACCENT_BLUE_COLOR,
                fg=LIGHT_TEXT_COLOR,
            ).grid(row=_adv_next_row, column=0, columnspan=4, pady=20)

        # ────────────────────────────────────
        #  PASSIVE  (HPOL/VPOL  or  G&D) SETTINGS
        # ────────────────────────────────────
        elif scan_type_value == "passive":
            tk.Label(
                settings_window, text="Passive Plot Settings", bg=DARK_BG_COLOR, fg=LIGHT_TEXT_COLOR
            ).grid(row=0, column=0, columnspan=4, pady=10)

            # VPOL/HPOL  vs  G&D
            self.plot_type_var = tk.StringVar(value=self.passive_scan_type.get())
            r_hv = tk.Radiobutton(
                settings_window,
                text="VPOL / HPOL",
                variable=self.plot_type_var,
                value="VPOL/HPOL",
                bg=DARK_BG_COLOR,
                fg=LIGHT_TEXT_COLOR,
                selectcolor=SURFACE_COLOR,
                activebackground=DARK_BG_COLOR,
                activeforeground=LIGHT_TEXT_COLOR,
            )
            r_gd = tk.Radiobutton(
                settings_window,
                text="G&D",
                variable=self.plot_type_var,
                value="G&D",
                bg=DARK_BG_COLOR,
                fg=LIGHT_TEXT_COLOR,
                selectcolor=SURFACE_COLOR,
                activebackground=DARK_BG_COLOR,
                activeforeground=LIGHT_TEXT_COLOR,
            )
            r_hv.grid(row=1, column=0, sticky=tk.W, padx=20)
            r_gd.grid(row=1, column=1, sticky=tk.W, padx=20)

            # Datasheet-style plots  (only for VPOL/HPOL)
            self.cb_datasheet_plots = tk.Checkbutton(
                settings_window,
                text="Datasheet Plots",
                variable=self.datasheet_plots_var,
                bg=DARK_BG_COLOR,
                fg=LIGHT_TEXT_COLOR,
                selectcolor=SURFACE_COLOR,
                activebackground=DARK_BG_COLOR,
                activeforeground=LIGHT_TEXT_COLOR,
            )

            # ECC calculation  (only for VPOL/HPOL)
            self.cb_ecc_analysis_var = tk.BooleanVar(
                value=getattr(self, "ecc_analysis_enabled", False)
            )
            self.cb_ecc_analysis = tk.Checkbutton(
                settings_window,
                text="Calculate Envelope Correlation Coefficient (ECC)",
                variable=self.cb_ecc_analysis_var,
                bg=DARK_BG_COLOR,
                fg=LIGHT_TEXT_COLOR,
                selectcolor=SURFACE_COLOR,
                activebackground=DARK_BG_COLOR,
                activeforeground=LIGHT_TEXT_COLOR,
            )

            # Min/Max Eff & Gain  (only for G&D)
            self.cb_min_max_eff_gain = tk.Checkbutton(
                settings_window,
                text="Min/Max Eff & Gain",
                variable=self.min_max_eff_gain_var,
                bg=DARK_BG_COLOR,
                fg=LIGHT_TEXT_COLOR,
                selectcolor=SURFACE_COLOR,
                activebackground=DARK_BG_COLOR,
                activeforeground=LIGHT_TEXT_COLOR,
            )

            # Human-torso shadowing model
            self.cb_shadowing_var = tk.BooleanVar(value=getattr(self, "shadowing_enabled", False))
            tk.Checkbutton(
                settings_window,
                text="Apply Human Torso Shadow",
                variable=self.cb_shadowing_var,
                bg=DARK_BG_COLOR,
                fg=LIGHT_TEXT_COLOR,
                selectcolor=SURFACE_COLOR,
                activebackground=DARK_BG_COLOR,
                activeforeground=LIGHT_TEXT_COLOR,
            ).grid(row=8, column=0, sticky=tk.W, padx=20)
            tk.Label(
                settings_window, text="Shadow Direction:", bg=DARK_BG_COLOR, fg=LIGHT_TEXT_COLOR
            ).grid(row=8, column=1, sticky=tk.E)
            self.shadow_direction_var = tk.StringVar(value=getattr(self, "shadow_direction", "-X"))
            ttk.Combobox(
                settings_window,
                textvariable=self.shadow_direction_var,
                values=["+X", "-X"],
                width=4,
                state="readonly",
            ).grid(row=8, column=2)

            # Per-type 3D Z-axis scaling
            tk.Label(
                settings_window,
                text="3-D Z-Axis Scale:",
                bg=DARK_BG_COLOR,
                fg=LIGHT_TEXT_COLOR,
            ).grid(row=3, column=0, sticky=tk.W, padx=20)
            for ci, hdr in enumerate(["Mode", "Min", "Max"], start=1):
                tk.Label(
                    settings_window,
                    text=hdr,
                    bg=DARK_BG_COLOR,
                    fg=LIGHT_TEXT_COLOR,
                ).grid(row=3, column=ci, sticky=tk.W)

            _scale_rows = [
                ("Total:", self.axis_scale_mode_total, self.axis_min_total, self.axis_max_total),
                ("H-Pol:", self.axis_scale_mode_hpol, self.axis_min_hpol, self.axis_max_hpol),
                ("V-Pol:", self.axis_scale_mode_vpol, self.axis_min_vpol, self.axis_max_vpol),
            ]
            for idx, (lbl, mode_var, min_var, max_var) in enumerate(_scale_rows):
                r = 4 + idx
                tk.Label(
                    settings_window,
                    text=lbl,
                    bg=DARK_BG_COLOR,
                    fg=LIGHT_TEXT_COLOR,
                ).grid(row=r, column=0, sticky=tk.W, padx=30)
                _mode_frame = tk.Frame(settings_window, bg=DARK_BG_COLOR)
                _mode_frame.grid(row=r, column=1, sticky=tk.W)
                tk.Radiobutton(
                    _mode_frame,
                    text="Auto",
                    variable=mode_var,
                    value="auto",
                    bg=DARK_BG_COLOR,
                    fg=LIGHT_TEXT_COLOR,
                    selectcolor=SURFACE_COLOR,
                    activebackground=DARK_BG_COLOR,
                    activeforeground=LIGHT_TEXT_COLOR,
                ).pack(side=tk.LEFT)
                tk.Radiobutton(
                    _mode_frame,
                    text="Man",
                    variable=mode_var,
                    value="manual",
                    bg=DARK_BG_COLOR,
                    fg=LIGHT_TEXT_COLOR,
                    selectcolor=SURFACE_COLOR,
                    activebackground=DARK_BG_COLOR,
                    activeforeground=LIGHT_TEXT_COLOR,
                ).pack(side=tk.LEFT)
                tk.Entry(
                    settings_window,
                    textvariable=min_var,
                    width=6,
                    bg=SURFACE_COLOR,
                    fg=LIGHT_TEXT_COLOR,
                    insertbackground=LIGHT_TEXT_COLOR,
                ).grid(row=r, column=2)
                tk.Entry(
                    settings_window,
                    textvariable=max_var,
                    width=6,
                    bg=SURFACE_COLOR,
                    fg=LIGHT_TEXT_COLOR,
                    insertbackground=LIGHT_TEXT_COLOR,
                ).grid(row=r, column=3)

            # Extrapolation controls (VPOL/HPOL only)
            extrap_frame = tk.Frame(settings_window, bg=DARK_BG_COLOR)
            self.extrap_freq_var = tk.StringVar(value="")
            tk.Label(
                extrap_frame,
                text="Extrapolate to Freq (MHz):",
                bg=DARK_BG_COLOR,
                fg=LIGHT_TEXT_COLOR,
            ).pack(side=tk.LEFT, padx=(0, 5))
            tk.Entry(
                extrap_frame,
                textvariable=self.extrap_freq_var,
                width=10,
                bg=SURFACE_COLOR,
                fg=LIGHT_TEXT_COLOR,
                insertbackground=LIGHT_TEXT_COLOR,
            ).pack(side=tk.LEFT, padx=(0, 5))

            has_files = bool(self.hpol_file_path and self.vpol_file_path)

            def _do_extrapolate():
                val = self.extrap_freq_var.get().strip()
                if not val:
                    from tkinter import messagebox as mb

                    mb.showwarning("Input Required", "Enter a target frequency in MHz.")
                    return
                try:
                    freq = float(val)
                except ValueError:
                    from tkinter import messagebox as mb

                    mb.showerror("Invalid Input", "Frequency must be a number.")
                    return
                self._run_extrapolation(freq)

            extrap_btn = tk.Button(
                extrap_frame,
                text="Extrapolate",
                command=_do_extrapolate,
                bg=ACCENT_BLUE_COLOR,
                fg=LIGHT_TEXT_COLOR,
                state=tk.NORMAL if has_files else tk.DISABLED,
            )
            extrap_btn.pack(side=tk.LEFT)
            extrap_frame.grid(row=9, column=0, columnspan=4, sticky=tk.W, padx=20, pady=5)

            # Helper to show / hide controls depending on radio-selection
            def refresh_passive_ui():
                if self.plot_type_var.get() == "G&D":
                    # hide VPOL/HPOL-only controls
                    self.cb_datasheet_plots.grid_remove()
                    self.cb_ecc_analysis.grid_remove()
                    extrap_frame.grid_remove()
                    # show G&D-specific
                    self.cb_min_max_eff_gain.grid(row=2, column=1, sticky=tk.W, padx=20)
                else:  # VPOL/HPOL
                    self.cb_min_max_eff_gain.grid_remove()
                    self.cb_datasheet_plots.grid(row=2, column=0, sticky=tk.W, padx=20)
                    self.cb_ecc_analysis.grid(row=7, column=0, sticky=tk.W, padx=20)
                    extrap_frame.grid(row=9, column=0, columnspan=4, sticky=tk.W, padx=20, pady=5)

            # first run + connect
            refresh_passive_ui()
            r_hv.config(command=refresh_passive_ui)
            r_gd.config(command=refresh_passive_ui)

            # Maritime / Horizon plot settings
            maritime_frame_p = tk.LabelFrame(
                settings_window,
                text="Maritime / Horizon Plots",
                bg=DARK_BG_COLOR,
                fg=ACCENT_BLUE_COLOR,
                font=SECTION_HEADER_FONT,
            )
            maritime_frame_p.grid(row=10, column=0, columnspan=4, sticky="ew", padx=15, pady=5)

            self.cb_maritime_var = tk.BooleanVar(
                value=getattr(self, "maritime_plots_enabled", False)
            )
            tk.Checkbutton(
                maritime_frame_p,
                text="Enable Maritime Plots",
                variable=self.cb_maritime_var,
                bg=DARK_BG_COLOR,
                fg=LIGHT_TEXT_COLOR,
                selectcolor=SURFACE_COLOR,
                activebackground=DARK_BG_COLOR,
                activeforeground=LIGHT_TEXT_COLOR,
            ).grid(row=0, column=0, columnspan=2, sticky=tk.W, padx=5, pady=2)

            tk.Label(
                maritime_frame_p, text="Theta Min (°):", bg=DARK_BG_COLOR, fg=LIGHT_TEXT_COLOR
            ).grid(row=1, column=0, sticky=tk.W, padx=5)
            tk.Entry(
                maritime_frame_p,
                textvariable=self.horizon_theta_min,
                width=6,
                bg=SURFACE_COLOR,
                fg=LIGHT_TEXT_COLOR,
                insertbackground=LIGHT_TEXT_COLOR,
            ).grid(row=1, column=1, padx=5)
            tk.Label(
                maritime_frame_p, text="Theta Max (°):", bg=DARK_BG_COLOR, fg=LIGHT_TEXT_COLOR
            ).grid(row=1, column=2, sticky=tk.W, padx=5)
            tk.Entry(
                maritime_frame_p,
                textvariable=self.horizon_theta_max,
                width=6,
                bg=SURFACE_COLOR,
                fg=LIGHT_TEXT_COLOR,
                insertbackground=LIGHT_TEXT_COLOR,
            ).grid(row=1, column=3, padx=5)

            tk.Label(
                maritime_frame_p,
                text="Coverage Threshold (dB):",
                bg=DARK_BG_COLOR,
                fg=LIGHT_TEXT_COLOR,
            ).grid(row=2, column=0, columnspan=2, sticky=tk.W, padx=5)
            tk.Entry(
                maritime_frame_p,
                textvariable=self.horizon_gain_threshold,
                width=6,
                bg=SURFACE_COLOR,
                fg=LIGHT_TEXT_COLOR,
                insertbackground=LIGHT_TEXT_COLOR,
            ).grid(row=2, column=2, padx=5)

            tk.Label(
                maritime_frame_p, text="Theta Cuts (°):", bg=DARK_BG_COLOR, fg=LIGHT_TEXT_COLOR
            ).grid(row=3, column=0, sticky=tk.W, padx=5)
            tk.Entry(
                maritime_frame_p,
                textvariable=self.horizon_theta_cuts_var,
                width=25,
                bg=SURFACE_COLOR,
                fg=LIGHT_TEXT_COLOR,
                insertbackground=LIGHT_TEXT_COLOR,
            ).grid(row=3, column=1, columnspan=3, sticky=tk.W, padx=5, pady=2)

            # Note: Conducted power fields are not shown for passive scans —
            # VNA-based gain measurements don't involve a radio transmitter.

            # Advanced analysis settings (Link Budget, Indoor, Fading, Wearable)
            _adv_next_row_p = self._build_advanced_analysis_frames(settings_window, start_row=11)

            # Save button
            def save_passive_settings():
                self.passive_scan_type.set(self.plot_type_var.get())
                self.ecc_analysis_enabled = self.cb_ecc_analysis_var.get()
                self.shadowing_enabled = self.cb_shadowing_var.get()
                self.shadow_direction = self.shadow_direction_var.get()
                self.maritime_plots_enabled = self.cb_maritime_var.get()
                self._save_advanced_analysis_settings()
                self.update_visibility()
                outer_window.destroy()

            tk.Button(
                settings_window,
                text="Save Settings",
                command=save_passive_settings,
                bg=ACCENT_BLUE_COLOR,
                fg=LIGHT_TEXT_COLOR,
            ).grid(row=_adv_next_row_p, column=0, columnspan=4, pady=20)

        elif scan_type_value == "vswr":
            # Show settings specific to VNA with organized LabelFrame sections
            title = tk.Label(
                settings_window,
                text="VSWR/Return Loss Settings",
                font=("Arial", 12, "bold"),
                bg=DARK_BG_COLOR,
                fg=LIGHT_TEXT_COLOR,
            )
            title.grid(row=0, column=0, columnspan=2, pady=10)

            def save_vswr_settings():
                try:
                    f1 = self.limit1_freq1.get()
                    f2 = self.limit1_freq2.get()
                    if f1 != 0 and f2 != 0 and f1 >= f2:
                        messagebox.showwarning(
                            "Invalid", "Limit 1: Freq Start must be less than Freq End"
                        )
                        return
                    f1_2 = self.limit2_freq1.get()
                    f2_2 = self.limit2_freq2.get()
                    if f1_2 != 0 and f2_2 != 0 and f1_2 >= f2_2:
                        messagebox.showwarning(
                            "Invalid", "Limit 2: Freq Start must be less than Freq End"
                        )
                        return
                    self.saved_limit1_freq1 = self.limit1_freq1.get()
                    self.saved_limit1_freq2 = self.limit1_freq2.get()
                    self.saved_limit1_start = self.limit1_val1.get()
                    self.saved_limit1_stop = self.limit1_val2.get()
                    self.saved_limit2_freq1 = self.limit2_freq1.get()
                    self.saved_limit2_freq2 = self.limit2_freq2.get()
                    self.saved_limit2_start = self.limit2_val1.get()
                    self.saved_limit2_stop = self.limit2_val2.get()
                    self.cb_groupdelay_sff = self.cb_groupdelay_sff_var.get()
                    self.saved_min_max_vswr = self.min_max_vswr_var.get()
                    outer_window.destroy()
                except tk.TclError:
                    messagebox.showerror("Invalid Input", "Please enter valid numeric values.")

            def default_vswr_settings():
                for var in (
                    self.limit1_freq1,
                    self.limit1_freq2,
                    self.limit1_val1,
                    self.limit1_val2,
                    self.limit2_freq1,
                    self.limit2_freq2,
                    self.limit2_val1,
                    self.limit2_val2,
                ):
                    var.set(0.0)
                self.saved_limit1_freq1 = 0.0
                self.saved_limit1_freq2 = 0.0
                self.saved_limit1_start = 0.0
                self.saved_limit1_stop = 0.0
                self.saved_limit2_freq1 = 0.0
                self.saved_limit2_freq2 = 0.0
                self.saved_limit2_start = 0.0
                self.saved_limit2_stop = 0.0
                self.cb_groupdelay_sff_var.set(False)
                self.saved_min_max_vswr = False
                self.min_max_vswr_var.set(False)

            # Options section
            opts_frame = tk.LabelFrame(
                settings_window,
                text="Options",
                padx=10,
                pady=5,
                bg=DARK_BG_COLOR,
                fg=LIGHT_TEXT_COLOR,
            )
            opts_frame.grid(row=1, column=0, columnspan=2, padx=10, pady=5, sticky="ew")
            tk.Checkbutton(
                opts_frame,
                text="Group Delay & SFF",
                variable=self.cb_groupdelay_sff_var,
                bg=DARK_BG_COLOR,
                fg=LIGHT_TEXT_COLOR,
                selectcolor=SURFACE_COLOR,
                activebackground=DARK_BG_COLOR,
                activeforeground=LIGHT_TEXT_COLOR,
            ).pack(anchor=tk.W)
            tk.Checkbutton(
                opts_frame,
                text="Tabled Min/Max VSWR",
                variable=self.min_max_vswr_var,
                bg=DARK_BG_COLOR,
                fg=LIGHT_TEXT_COLOR,
                selectcolor=SURFACE_COLOR,
                activebackground=DARK_BG_COLOR,
                activeforeground=LIGHT_TEXT_COLOR,
            ).pack(anchor=tk.W)
            if hasattr(self, "saved_min_max_vswr"):
                self.min_max_vswr_var.set(self.saved_min_max_vswr)

            # Limit Line 1 section
            limit1_frame = tk.LabelFrame(
                settings_window,
                text="Limit Line 1",
                padx=10,
                pady=5,
                bg=DARK_BG_COLOR,
                fg=LIGHT_TEXT_COLOR,
            )
            limit1_frame.grid(row=2, column=0, columnspan=2, padx=10, pady=5, sticky="ew")

            self.limit1_freq1 = tk.DoubleVar()
            self.limit1_freq2 = tk.DoubleVar()
            self.limit1_val1 = tk.DoubleVar()
            self.limit1_val2 = tk.DoubleVar()

            tk.Label(
                limit1_frame, text="Freq Start (GHz):", bg=DARK_BG_COLOR, fg=LIGHT_TEXT_COLOR
            ).grid(row=0, column=0, sticky="e", padx=5, pady=2)
            tk.Entry(
                limit1_frame,
                textvariable=self.limit1_freq1,
                width=10,
                bg=SURFACE_COLOR,
                fg=LIGHT_TEXT_COLOR,
                insertbackground=LIGHT_TEXT_COLOR,
            ).grid(row=0, column=1, padx=5, pady=2)
            tk.Label(limit1_frame, text="Value Start:", bg=DARK_BG_COLOR, fg=LIGHT_TEXT_COLOR).grid(
                row=0, column=2, sticky="e", padx=5, pady=2
            )
            tk.Entry(
                limit1_frame,
                textvariable=self.limit1_val1,
                width=10,
                bg=SURFACE_COLOR,
                fg=LIGHT_TEXT_COLOR,
                insertbackground=LIGHT_TEXT_COLOR,
            ).grid(row=0, column=3, padx=5, pady=2)

            tk.Label(
                limit1_frame, text="Freq End (GHz):", bg=DARK_BG_COLOR, fg=LIGHT_TEXT_COLOR
            ).grid(row=1, column=0, sticky="e", padx=5, pady=2)
            tk.Entry(
                limit1_frame,
                textvariable=self.limit1_freq2,
                width=10,
                bg=SURFACE_COLOR,
                fg=LIGHT_TEXT_COLOR,
                insertbackground=LIGHT_TEXT_COLOR,
            ).grid(row=1, column=1, padx=5, pady=2)
            tk.Label(limit1_frame, text="Value End:", bg=DARK_BG_COLOR, fg=LIGHT_TEXT_COLOR).grid(
                row=1, column=2, sticky="e", padx=5, pady=2
            )
            tk.Entry(
                limit1_frame,
                textvariable=self.limit1_val2,
                width=10,
                bg=SURFACE_COLOR,
                fg=LIGHT_TEXT_COLOR,
                insertbackground=LIGHT_TEXT_COLOR,
            ).grid(row=1, column=3, padx=5, pady=2)

            # Limit Line 2 section
            limit2_frame = tk.LabelFrame(
                settings_window,
                text="Limit Line 2",
                padx=10,
                pady=5,
                bg=DARK_BG_COLOR,
                fg=LIGHT_TEXT_COLOR,
            )
            limit2_frame.grid(row=3, column=0, columnspan=2, padx=10, pady=5, sticky="ew")

            self.limit2_freq1 = tk.DoubleVar()
            self.limit2_freq2 = tk.DoubleVar()
            self.limit2_val1 = tk.DoubleVar()
            self.limit2_val2 = tk.DoubleVar()

            tk.Label(
                limit2_frame, text="Freq Start (GHz):", bg=DARK_BG_COLOR, fg=LIGHT_TEXT_COLOR
            ).grid(row=0, column=0, sticky="e", padx=5, pady=2)
            tk.Entry(
                limit2_frame,
                textvariable=self.limit2_freq1,
                width=10,
                bg=SURFACE_COLOR,
                fg=LIGHT_TEXT_COLOR,
                insertbackground=LIGHT_TEXT_COLOR,
            ).grid(row=0, column=1, padx=5, pady=2)
            tk.Label(limit2_frame, text="Value Start:", bg=DARK_BG_COLOR, fg=LIGHT_TEXT_COLOR).grid(
                row=0, column=2, sticky="e", padx=5, pady=2
            )
            tk.Entry(
                limit2_frame,
                textvariable=self.limit2_val1,
                width=10,
                bg=SURFACE_COLOR,
                fg=LIGHT_TEXT_COLOR,
                insertbackground=LIGHT_TEXT_COLOR,
            ).grid(row=0, column=3, padx=5, pady=2)

            tk.Label(
                limit2_frame, text="Freq End (GHz):", bg=DARK_BG_COLOR, fg=LIGHT_TEXT_COLOR
            ).grid(row=1, column=0, sticky="e", padx=5, pady=2)
            tk.Entry(
                limit2_frame,
                textvariable=self.limit2_freq2,
                width=10,
                bg=SURFACE_COLOR,
                fg=LIGHT_TEXT_COLOR,
                insertbackground=LIGHT_TEXT_COLOR,
            ).grid(row=1, column=1, padx=5, pady=2)
            tk.Label(limit2_frame, text="Value End:", bg=DARK_BG_COLOR, fg=LIGHT_TEXT_COLOR).grid(
                row=1, column=2, sticky="e", padx=5, pady=2
            )
            tk.Entry(
                limit2_frame,
                textvariable=self.limit2_val2,
                width=10,
                bg=SURFACE_COLOR,
                fg=LIGHT_TEXT_COLOR,
                insertbackground=LIGHT_TEXT_COLOR,
            ).grid(row=1, column=3, padx=5, pady=2)

            # Restore saved values
            if hasattr(self, "saved_limit1_freq1"):
                self.limit1_freq1.set(self.saved_limit1_freq1)
                self.limit1_freq2.set(self.saved_limit1_freq2)
                self.limit1_val1.set(self.saved_limit1_start)
                self.limit1_val2.set(self.saved_limit1_stop)
                self.limit2_freq1.set(self.saved_limit2_freq1)
                self.limit2_freq2.set(self.saved_limit2_freq2)
                self.limit2_val1.set(self.saved_limit2_start)
                self.limit2_val2.set(self.saved_limit2_stop)

            # Buttons
            btn_frame = tk.Frame(settings_window, bg=DARK_BG_COLOR)
            btn_frame.grid(row=4, column=0, columnspan=2, pady=15)
            tk.Button(
                btn_frame,
                text="Save",
                command=save_vswr_settings,
                bg=ACCENT_BLUE_COLOR,
                fg=LIGHT_TEXT_COLOR,
                width=12,
            ).pack(side=tk.LEFT, padx=5)
            tk.Button(
                btn_frame,
                text="Defaults",
                command=default_vswr_settings,
                bg=BUTTON_COLOR,
                fg=LIGHT_TEXT_COLOR,
                width=12,
            ).pack(side=tk.LEFT, padx=5)
