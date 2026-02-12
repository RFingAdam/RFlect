"""
Main Window - Core AntennaPlotGUI class for RFlect

This module contains the main GUI class that combines all mixins:
- DialogsMixin: Dialogs (About, API Key, Settings)
- AIChatMixin: AI Chat functionality
- ToolsMixin: Bulk processing, polarization, converters
- CallbacksMixin: File import, data processing, save operations
"""

from __future__ import annotations

import os
import sys
import json
import tkinter as tk
from tkinter import filedialog, messagebox, ttk, Menu
import tkinter.scrolledtext as ScrolledText
import webbrowser
from typing import Optional, List, Any

import matplotlib

matplotlib.use("TkAgg")
import matplotlib.pyplot as plt

from ..config import (
    DARK_BG_COLOR,
    LIGHT_TEXT_COLOR,
    ACCENT_BLUE_COLOR,
    ERROR_COLOR,
    WARNING_COLOR,
    SUCCESS_COLOR,
    HOVER_COLOR,
    SURFACE_COLOR,
    SURFACE_LIGHT_COLOR,
    BORDER_COLOR,
    DIVIDER_COLOR,
    LOG_BG_COLOR,
    HEADER_BAR_COLOR,
    HEADER_ACCENT_COLOR,
    DISABLED_FG_COLOR,
    FOCUS_BORDER_COLOR,
    HEADER_BAR_FONT,
    HEADER_VERSION_FONT,
    SECTION_HEADER_FONT,
    LABEL_FONT_MODERN,
    BUTTON_FONT,
    LOG_FONT,
    STATUS_FONT,
    BTN_PADX,
    BTN_PADY,
    SECTION_PAD,
    WIDGET_GAP,
    PAD_SM,
    PAD_MD,
    PAD_LG,
)
from ..calculations import extract_passive_frequencies

from ..api_keys import is_api_key_configured, initialize_keys, clear_env_keys

# Import mixins
from .dialogs_mixin import DialogsMixin
from .ai_chat_mixin import AIChatMixin  # kept for mixin class — methods unused until re-enabled
from .tools_mixin import ToolsMixin
from .callbacks_mixin import CallbacksMixin

# Import utility classes and functions
from .utils import DualOutput, resource_path, get_user_data_dir, get_current_version


class AntennaPlotGUI(DialogsMixin, AIChatMixin, ToolsMixin, CallbacksMixin):  # type: ignore[misc]
    """
    Main GUI class for the RFlect application.

    This class manages the creation and interactions of the main application window.
    Functionality is organized into mixins for maintainability:
    - DialogsMixin: About, API key, settings dialogs
    - AIChatMixin: AI chat and analysis functions
    - ToolsMixin: Bulk processing, polarization, converters
    - CallbacksMixin: File import, data processing
    """

    # ────────────────────────────────────────────────────────────────────────
    # CLASS INITIALIZATION & VERSION
    # ────────────────────────────────────────────────────────────────────────

    # Load version from settings.json
    settings_path = resource_path("settings.json")
    with open(settings_path, "r", encoding="utf-8") as file:
        settings = json.load(file)
        CURRENT_VERSION = settings["CURRENT_VERSION"]

    # ────────────────────────────────────────────────────────────────────────
    # INITIALIZATION
    # ────────────────────────────────────────────────────────────────────────

    def __init__(self, root):
        self.root = root
        self.root.title("RFlect")
        self.root.geometry("850x600")
        self.root.minsize(700, 500)

        self.frequency_var = tk.StringVar(self.root)
        self.freq_list = []

        # File path attributes
        self.hpol_file_path = None
        self.TRP_file_path = None
        self.vpol_file_path = None

        # Processing state
        self._processing_lock = False
        self._active_figures = []
        self._extrapolation_cache = {}

        # Measurement context for AI awareness
        self._measurement_context = {
            "files_loaded": [],
            "scan_type": None,
            "frequencies": [],
            "data_shape": None,
            "cable_loss_applied": 0.0,
            "extrapolation_applied": False,
            "extrapolation_confidence": None,
            "processing_complete": False,
            "key_metrics": {},
        }

        # VSWR settings
        self.saved_min_max_vswr = False
        self.min_max_vswr_var = tk.BooleanVar(value=self.saved_min_max_vswr)
        self.min_max_eff_gain_var = tk.BooleanVar(value=False)

        # G&D settings
        self.files_per_band = 0
        self.selected_bands: list[tuple[float, float]] = []
        self.measurement_scenario = ""

        # Recent files
        self.recent_files = []
        self.max_recent_files = 5
        self.load_recent_files()

        # Initialize API key storage (loads keys from keyring/file/env)
        initialize_keys()

        # VSWR limit settings
        self.saved_limit1_freq1 = 0.0
        self.saved_limit1_freq2 = 0.0
        self.saved_limit1_start = 0.0
        self.saved_limit1_stop = 0.0
        self.saved_limit2_freq1 = 0.0
        self.saved_limit2_freq2 = 0.0
        self.saved_limit2_start = 0.0
        self.saved_limit2_stop = 0.0

        # Load persisted user settings (overrides defaults above)
        self._load_user_settings()

        # Scan type settings
        self.plot_type_var = tk.StringVar()
        self.plot_type_var.set("HPOL/VPOL")

        self.passive_scan_type = tk.StringVar()
        self.passive_scan_type.set("VPOL/HPOL")

        self.interpolate_3d_plots = True

        # Load logo
        self._load_logo()

        # Configure ttk dark theme before creating widgets
        self._setup_ttk_theme()

        # Create GUI elements
        self._create_widgets()
        self._create_menu()
        self._create_log_area()
        self._create_status_bar()
        self._bind_shortcuts()

        # Set up window close handler
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)

        # Check for updates on startup
        self.check_for_updates()

    def _load_logo(self):
        """Load application logo."""
        if getattr(sys, "frozen", False):
            current_path = sys._MEIPASS  # type: ignore
        else:
            current_path = os.path.dirname(
                os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            )

        logo_path = os.path.join(current_path, "assets", "smith_logo.png")
        self.logo_image = tk.PhotoImage(file=logo_path)
        self.root.iconphoto(False, self.logo_image)

    def _setup_ttk_theme(self):
        """Configure a modern dark theme for all ttk widgets."""
        style = ttk.Style()
        style.theme_use("clam")

        # --- TButton ---
        style.configure(
            "TButton",
            background=ACCENT_BLUE_COLOR,
            foreground=LIGHT_TEXT_COLOR,
            font=BUTTON_FONT,
            padding=(BTN_PADX, BTN_PADY),
            relief="flat",
            borderwidth=0,
        )
        style.map(
            "TButton",
            background=[("active", HOVER_COLOR), ("disabled", SURFACE_COLOR)],
            foreground=[("disabled", DISABLED_FG_COLOR)],
        )

        # Secondary button variant
        style.configure(
            "Secondary.TButton",
            background=SURFACE_COLOR,
            foreground=LIGHT_TEXT_COLOR,
            font=BUTTON_FONT,
            padding=(BTN_PADX, BTN_PADY),
            relief="flat",
        )
        style.map("Secondary.TButton", background=[("active", HOVER_COLOR)])

        # --- TRadiobutton ---
        style.configure(
            "TRadiobutton",
            background=DARK_BG_COLOR,
            foreground=LIGHT_TEXT_COLOR,
            font=LABEL_FONT_MODERN,
            indicatorcolor=SURFACE_LIGHT_COLOR,
            indicatorrelief="flat",
        )
        style.map(
            "TRadiobutton",
            background=[("active", SURFACE_COLOR)],
            indicatorcolor=[("selected", ACCENT_BLUE_COLOR)],
        )

        # --- TCheckbutton ---
        style.configure(
            "TCheckbutton",
            background=DARK_BG_COLOR,
            foreground=LIGHT_TEXT_COLOR,
            font=LABEL_FONT_MODERN,
            indicatorcolor=SURFACE_LIGHT_COLOR,
        )
        style.map(
            "TCheckbutton",
            background=[("active", SURFACE_COLOR)],
            indicatorcolor=[("selected", ACCENT_BLUE_COLOR)],
        )

        # --- TCombobox ---
        style.configure(
            "TCombobox",
            fieldbackground=SURFACE_LIGHT_COLOR,
            background=SURFACE_COLOR,
            foreground=LIGHT_TEXT_COLOR,
            arrowcolor=LIGHT_TEXT_COLOR,
            bordercolor=BORDER_COLOR,
            lightcolor=BORDER_COLOR,
            darkcolor=BORDER_COLOR,
            selectbackground=ACCENT_BLUE_COLOR,
            selectforeground=LIGHT_TEXT_COLOR,
        )
        style.map(
            "TCombobox",
            fieldbackground=[("readonly", SURFACE_LIGHT_COLOR)],
            selectbackground=[("readonly", ACCENT_BLUE_COLOR)],
        )
        self.root.option_add("*TCombobox*Listbox.background", SURFACE_LIGHT_COLOR)
        self.root.option_add("*TCombobox*Listbox.foreground", LIGHT_TEXT_COLOR)
        self.root.option_add("*TCombobox*Listbox.selectBackground", ACCENT_BLUE_COLOR)
        self.root.option_add("*TCombobox*Listbox.selectForeground", LIGHT_TEXT_COLOR)

        # --- TProgressbar ---
        style.configure(
            "TProgressbar",
            background=ACCENT_BLUE_COLOR,
            troughcolor=SURFACE_COLOR,
            bordercolor=BORDER_COLOR,
            lightcolor=ACCENT_BLUE_COLOR,
            darkcolor=ACCENT_BLUE_COLOR,
            thickness=6,
        )

        # --- TLabelframe ---
        style.configure(
            "TLabelframe",
            background=DARK_BG_COLOR,
            foreground=LIGHT_TEXT_COLOR,
            bordercolor=BORDER_COLOR,
            relief="flat",
        )
        style.configure(
            "TLabelframe.Label",
            background=DARK_BG_COLOR,
            foreground=ACCENT_BLUE_COLOR,
            font=SECTION_HEADER_FONT,
        )

        # --- TLabel ---
        style.configure(
            "TLabel",
            background=DARK_BG_COLOR,
            foreground=LIGHT_TEXT_COLOR,
            font=LABEL_FONT_MODERN,
        )

        # --- TEntry ---
        style.configure(
            "TEntry",
            fieldbackground=SURFACE_LIGHT_COLOR,
            foreground=LIGHT_TEXT_COLOR,
            insertcolor=LIGHT_TEXT_COLOR,
            bordercolor=BORDER_COLOR,
            lightcolor=BORDER_COLOR,
            darkcolor=BORDER_COLOR,
        )
        style.map(
            "TEntry",
            bordercolor=[("focus", FOCUS_BORDER_COLOR)],
            lightcolor=[("focus", FOCUS_BORDER_COLOR)],
        )

        # --- Vertical.TScrollbar ---
        style.configure(
            "Vertical.TScrollbar",
            background=SURFACE_COLOR,
            troughcolor=DARK_BG_COLOR,
            arrowcolor=LIGHT_TEXT_COLOR,
            bordercolor=DARK_BG_COLOR,
        )
        style.map("Vertical.TScrollbar", background=[("active", HOVER_COLOR)])

        # --- TSeparator ---
        style.configure("TSeparator", background=DIVIDER_COLOR)

        # --- Menu styling ---
        self.root.option_add("*Menu.background", SURFACE_COLOR)
        self.root.option_add("*Menu.foreground", LIGHT_TEXT_COLOR)
        self.root.option_add("*Menu.activeBackground", ACCENT_BLUE_COLOR)
        self.root.option_add("*Menu.activeForeground", LIGHT_TEXT_COLOR)
        self.root.option_add("*Menu.selectColor", ACCENT_BLUE_COLOR)

    # ────────────────────────────────────────────────────────────────────────
    # WIDGET CREATION
    # ────────────────────────────────────────────────────────────────────────

    def _create_widgets(self):
        """Create main GUI widgets with modern paneled layout."""

        # ── ROW 0: BRANDED HEADER BAR ──────────────────────────────────────
        self.header_frame = tk.Frame(self.root, bg=HEADER_BAR_COLOR, height=56)
        self.header_frame.grid(row=0, column=0, sticky="ew")
        self.header_frame.grid_propagate(False)

        # Logo in header
        try:
            from PIL import Image, ImageTk

            logo_path = resource_path(os.path.join("assets", "smith_logo.png"))
            if os.path.exists(logo_path):
                logo_img = Image.open(logo_path)
                logo_img = logo_img.resize((36, 36), Image.Resampling.LANCZOS)
                self._header_logo = ImageTk.PhotoImage(logo_img)
                logo_label = tk.Label(
                    self.header_frame, image=self._header_logo, bg=HEADER_BAR_COLOR
                )
                logo_label.pack(side=tk.LEFT, padx=(PAD_LG, PAD_SM), pady=PAD_SM)
        except (FileNotFoundError, ImportError, OSError):
            pass

        self.title_label = tk.Label(
            self.header_frame,
            text="RFlect",
            bg=HEADER_BAR_COLOR,
            fg=HEADER_ACCENT_COLOR,
            font=HEADER_BAR_FONT,
        )
        self.title_label.pack(side=tk.LEFT, pady=PAD_SM)

        tk.Label(
            self.header_frame,
            text="  Antenna Measurement Tool",
            bg=HEADER_BAR_COLOR,
            fg="#AAAAAA",
            font=LABEL_FONT_MODERN,
        ).pack(side=tk.LEFT, pady=PAD_SM)

        tk.Label(
            self.header_frame,
            text=self.CURRENT_VERSION,
            bg=HEADER_BAR_COLOR,
            fg="#777777",
            font=HEADER_VERSION_FONT,
        ).pack(side=tk.RIGHT, padx=PAD_LG, pady=PAD_SM)

        # ── ROW 1: MEASUREMENT TYPE SECTION ────────────────────────────────
        self.controls_frame = ttk.LabelFrame(
            self.root, text="  Measurement Type", padding=SECTION_PAD
        )
        self.controls_frame.grid(row=1, column=0, sticky="ew", padx=PAD_LG, pady=(PAD_MD, PAD_SM))

        radio_frame = tk.Frame(self.controls_frame, bg=DARK_BG_COLOR)
        radio_frame.pack(side=tk.LEFT, fill=tk.X, expand=True)

        # Initialize scan settings
        self.scan_type = tk.StringVar()
        self.scan_type.set("passive")
        self.passive_scan_type = tk.StringVar()
        self.passive_scan_type.set("VPOL/HPOL")
        self.datasheet_plots_var = tk.BooleanVar(value=False)
        self.cb_groupdelay_sff_var = tk.BooleanVar(value=False)
        self.ecc_analysis_enabled = False

        self.label_scan_type = ttk.Label(radio_frame, text="Type:")
        self.label_scan_type.pack(side=tk.LEFT, padx=(0, PAD_SM))

        ttk.Radiobutton(
            radio_frame,
            text="Active",
            variable=self.scan_type,
            value="active",
            command=self.update_visibility,
        ).pack(side=tk.LEFT, padx=PAD_SM)

        ttk.Radiobutton(
            radio_frame,
            text="Passive",
            variable=self.scan_type,
            value="passive",
            command=self.update_visibility,
        ).pack(side=tk.LEFT, padx=PAD_SM)

        ttk.Radiobutton(
            radio_frame,
            text="VNA (.csv)",
            variable=self.scan_type,
            value="vswr",
            command=self.update_visibility,
        ).pack(side=tk.LEFT, padx=PAD_SM)

        # Import button (right side)
        self.btn_import = tk.Button(
            self.controls_frame,
            text="\U0001f4c2  Import File(s)",
            command=self.import_files,
            bg=ACCENT_BLUE_COLOR,
            fg=LIGHT_TEXT_COLOR,
            font=BUTTON_FONT,
            relief="flat",
            bd=0,
            padx=BTN_PADX,
            pady=BTN_PADY,
            cursor="hand2",
        )
        self.btn_import.pack(side=tk.RIGHT, padx=PAD_SM)

        # ── ROW 2: PARAMETERS SECTION ──────────────────────────────────────
        self.params_frame = ttk.LabelFrame(self.root, text="  Parameters", padding=SECTION_PAD)
        self.params_frame.grid(row=2, column=0, sticky="ew", padx=PAD_LG, pady=PAD_SM)

        self.available_frequencies = []
        self.selected_frequency = tk.StringVar()
        self.label_frequency = ttk.Label(self.params_frame, text="Frequency:")
        self.label_frequency.grid(row=0, column=0, sticky="w", padx=(0, PAD_SM), pady=2)
        self.frequency_dropdown = ttk.Combobox(
            self.params_frame,
            textvariable=self.selected_frequency,
            values=self.available_frequencies,
            state="readonly",
            width=20,
        )
        self.frequency_dropdown.grid(row=0, column=1, sticky="w", padx=PAD_SM, pady=2)

        self.label_cable_loss = ttk.Label(self.params_frame, text="Cable Loss (dB):")
        self.cable_loss = tk.StringVar(self.root, value="0.0")
        self.cable_loss_input = ttk.Entry(self.params_frame, textvariable=self.cable_loss, width=10)
        self.label_cable_loss.grid(row=1, column=0, sticky="w", padx=(0, PAD_SM), pady=2)
        self.cable_loss_input.grid(row=1, column=1, sticky="w", padx=PAD_SM, pady=2)
        self.params_frame.columnconfigure(2, weight=1)

        # ── ROW 3: ACTION BUTTONS BAR ──────────────────────────────────────
        self.actions_frame = tk.Frame(self.root, bg=DARK_BG_COLOR)
        self.actions_frame.grid(row=3, column=0, sticky="ew", padx=PAD_LG, pady=PAD_SM)

        self.btn_view_results = tk.Button(
            self.actions_frame,
            text="\u25b6  View Results",
            command=self.process_data,
            bg=ACCENT_BLUE_COLOR,
            fg=LIGHT_TEXT_COLOR,
            font=BUTTON_FONT,
            relief="flat",
            bd=0,
            padx=BTN_PADX,
            pady=BTN_PADY,
            cursor="hand2",
        )
        self.btn_view_results.pack(side=tk.LEFT, padx=(0, WIDGET_GAP))

        self.btn_save_to_file = tk.Button(
            self.actions_frame,
            text="\U0001f4be  Save Results",
            command=lambda: self.save_results_to_file(),
            bg=ACCENT_BLUE_COLOR,
            fg=LIGHT_TEXT_COLOR,
            font=BUTTON_FONT,
            relief="flat",
            bd=0,
            padx=BTN_PADX,
            pady=BTN_PADY,
            cursor="hand2",
        )
        self.btn_save_to_file.pack(side=tk.LEFT, padx=WIDGET_GAP)

        self.btn_settings = tk.Button(
            self.actions_frame,
            text="\u2699  Settings",
            command=self.show_settings,
            bg=SURFACE_COLOR,
            fg=LIGHT_TEXT_COLOR,
            font=BUTTON_FONT,
            relief="flat",
            bd=0,
            padx=BTN_PADX,
            pady=BTN_PADY,
            cursor="hand2",
        )
        self.btn_settings.pack(side=tk.RIGHT, padx=(WIDGET_GAP, 0))

        # 3D plotting settings
        self.axis_scale_mode = tk.StringVar(value="auto")
        self.axis_min = tk.DoubleVar(value=-20.0)
        self.axis_max = tk.DoubleVar(value=6.0)

        # Human shadowing settings
        self.shadowing_enabled = False
        self.shadow_direction = "-X"
        self.cb_shadowing_var = tk.BooleanVar(value=False)
        self.shadow_direction_var = tk.StringVar(value="-X")

        # Maritime / Horizon plot settings
        self.maritime_plots_enabled = False
        self.horizon_theta_min = tk.DoubleVar(value=60.0)
        self.horizon_theta_max = tk.DoubleVar(value=120.0)
        self.horizon_gain_threshold = tk.DoubleVar(value=-3.0)
        self.horizon_theta_cuts_var = tk.StringVar(value="60,70,80,90,100,110,120")

        # Link Budget / Range Estimation settings
        self.link_budget_enabled = False
        self.lb_protocol_preset = tk.StringVar(value="BLE 1Mbps")
        self.lb_tx_power = tk.DoubleVar(value=0.0)
        self.lb_rx_sensitivity = tk.DoubleVar(value=-98.0)
        self.lb_rx_gain = tk.DoubleVar(value=0.0)
        self.lb_path_loss_exp = tk.DoubleVar(value=2.0)
        self.lb_misc_loss = tk.DoubleVar(value=10.0)
        self.lb_target_range = tk.DoubleVar(value=5.0)

        # Indoor Propagation settings
        self.indoor_analysis_enabled = False
        self.indoor_environment = tk.StringVar(value="Office")
        self.indoor_num_walls = tk.IntVar(value=1)
        self.indoor_wall_material = tk.StringVar(value="drywall")
        self.indoor_shadow_fading = tk.DoubleVar(value=5.0)
        self.indoor_max_distance = tk.DoubleVar(value=30.0)

        # Multipath Fading Analysis settings
        self.fading_analysis_enabled = False
        self.fading_model = tk.StringVar(value="rayleigh")
        self.fading_rician_k = tk.DoubleVar(value=10.0)
        self.fading_target_reliability = tk.DoubleVar(value=99.0)

        # MIMO / Diversity Analysis settings
        self.mimo_analysis_enabled = False
        self.mimo_snr = tk.DoubleVar(value=20.0)
        self.mimo_xpr = tk.DoubleVar(value=6.0)

        # Wearable / Medical Device settings
        self.wearable_analysis_enabled = False
        self.wearable_positions_var = {
            pos: tk.BooleanVar(value=True) for pos in ["wrist", "chest", "hip", "head"]
        }
        self.wearable_tx_power_mw = tk.DoubleVar(value=1.0)
        self.wearable_device_count = tk.IntVar(value=20)
        self.wearable_room_x = tk.DoubleVar(value=10.0)
        self.wearable_room_y = tk.DoubleVar(value=10.0)
        self.wearable_room_z = tk.DoubleVar(value=3.0)

        # Configure background
        self.root.config(bg=DARK_BG_COLOR)

        # Bind hover effects
        buttons = [self.btn_import, self.btn_view_results, self.btn_save_to_file, self.btn_settings]
        for btn in buttons:
            btn.bind("<Enter>", self.on_enter)
            btn.bind("<Leave>", self.on_leave)

        # Add tooltips
        from .tooltip import ToolTip

        ToolTip(self.btn_import, "Import HPOL/VPOL or TRP measurement files (Ctrl+O)")
        ToolTip(self.btn_view_results, "Process loaded data and display plots")
        ToolTip(self.btn_save_to_file, "Save processed results and plots to disk")
        ToolTip(self.btn_settings, "Configure plot settings for current scan type")

        # Initialize visibility
        self.update_visibility()

    def _create_menu(self):
        """Create menu bar."""
        menubar = Menu(self.root)

        # File Menu
        file_menu = Menu(menubar, tearoff=0)
        file_menu.add_command(
            label="Import File(s)...", command=self.import_files, accelerator="Ctrl+O"
        )
        file_menu.add_separator()

        # Recent Files submenu
        self.recent_menu = Menu(file_menu, tearoff=0)
        file_menu.add_cascade(label="Recent Files", menu=self.recent_menu)
        self.update_recent_files_menu()

        file_menu.add_separator()
        file_menu.add_command(label="Exit", command=self.on_closing, accelerator="Ctrl+Q")
        menubar.add_cascade(label="File", menu=file_menu)

        # Tools Menu
        tools_menu = Menu(menubar, tearoff=0)
        tools_menu.add_command(
            label="Bulk Passive Processing", command=self.open_bulk_passive_processing
        )
        tools_menu.add_command(
            label="Bulk Active Processing", command=self.open_bulk_active_processing
        )
        tools_menu.add_separator()
        tools_menu.add_command(
            label="Polarization Analysis (Export)", command=self.open_polarization_analysis
        )
        tools_menu.add_command(
            label="Polarization Analysis (Interactive)", command=self.open_polarization_interactive
        )
        tools_menu.add_separator()
        tools_menu.add_command(
            label="HPOL/VPOL->CST FFS Converter", command=self.open_hpol_vpol_converter
        )
        tools_menu.add_command(
            label="Active Chamber Calibration", command=self.open_active_chamber_cal
        )
        tools_menu.add_separator()
        tools_menu.add_command(label="Generate Report", command=self.generate_report_from_directory)

        # AI tools
        tools_menu.add_separator()
        tools_menu.add_command(label="Manage API Keys...", command=self.manage_api_key)
        tools_menu.add_command(label="AI Settings...", command=self.manage_ai_settings)
        any_key = is_api_key_configured("openai") or is_api_key_configured("anthropic")
        if any_key:
            tools_menu.add_command(
                label="Generate Report with AI", command=self.generate_ai_report_from_directory
            )
            tools_menu.add_command(label="AI Chat Assistant...", command=self.open_ai_chat)
        else:
            print("[INFO] No AI API key configured. AI report/chat disabled.")
            print("       Configure via: Tools -> Manage API Keys")

        menubar.add_cascade(label="Tools", menu=tools_menu)

        # Help Menu
        help_menu = Menu(menubar, tearoff=0)
        help_menu.add_command(label="About RFlect", command=self.show_about_dialog)
        help_menu.add_separator()
        help_menu.add_command(label="Check for Updates", command=self.check_for_updates)
        help_menu.add_separator()
        help_menu.add_command(
            label="View on GitHub",
            command=lambda: webbrowser.open("https://github.com/RFingAdam/RFlect"),
        )
        help_menu.add_command(
            label="Report an Issue",
            command=lambda: webbrowser.open("https://github.com/RFingAdam/RFlect/issues"),
        )
        help_menu.add_separator()
        help_menu.add_command(
            label="MCP Server Setup",
            command=lambda: webbrowser.open(
                "https://github.com/RFingAdam/RFlect/tree/main/rflect-mcp#readme"
            ),
        )

        menubar.add_cascade(label="Help", menu=help_menu)

        self.root.config(menu=menubar)

    def _create_log_area(self):
        """Create log text area with header."""
        log_outer = tk.Frame(self.root, bg=DARK_BG_COLOR)
        log_outer.grid(row=4, column=0, sticky="nsew", padx=PAD_LG, pady=(PAD_SM, 0))

        # Log header bar
        log_header = tk.Frame(log_outer, bg=SURFACE_COLOR, height=28)
        log_header.pack(fill=tk.X)
        log_header.pack_propagate(False)
        tk.Label(
            log_header,
            text="  Output Log",
            bg=SURFACE_COLOR,
            fg="#AAAAAA",
            font=STATUS_FONT,
            anchor="w",
        ).pack(side=tk.LEFT, padx=PAD_SM, pady=2)

        # Log text area
        log_frame = tk.Frame(log_outer, bg=LOG_BG_COLOR)
        log_frame.pack(fill=tk.BOTH, expand=True)

        self.log_text = ScrolledText.ScrolledText(
            log_frame,
            wrap=tk.WORD,
            height=10,
            bg=LOG_BG_COLOR,
            fg=LIGHT_TEXT_COLOR,
            font=LOG_FONT,
            relief=tk.FLAT,
            borderwidth=0,
            padx=8,
            pady=4,
            insertbackground=LIGHT_TEXT_COLOR,
        )
        self.log_text.grid(row=0, column=0, sticky="nsew")
        log_frame.grid_rowconfigure(0, weight=1)
        log_frame.grid_columnconfigure(0, weight=1)

        # Row/column weights
        self.root.grid_rowconfigure(4, weight=1)
        self.root.grid_columnconfigure(0, weight=1)

        # Configure log text tags for color-coded messages
        self.log_text.tag_config("log_info", foreground=LIGHT_TEXT_COLOR)
        self.log_text.tag_config("log_success", foreground=SUCCESS_COLOR)
        self.log_text.tag_config("log_warning", foreground=WARNING_COLOR)
        self.log_text.tag_config("log_error", foreground=ERROR_COLOR)

        # Redirect stdout and stderr
        sys.stdout = DualOutput(self.log_text, sys.stdout)
        sys.stderr = DualOutput(self.log_text, sys.stderr)

    def _create_status_bar(self):
        """Create modern status bar with separator."""
        ttk.Separator(self.root, orient="horizontal").grid(row=5, column=0, sticky="ew")

        status_frame = tk.Frame(self.root, bg=HEADER_BAR_COLOR)
        status_frame.grid(row=6, column=0, sticky="ew")

        self.status_bar = tk.Label(
            status_frame,
            text="\u2713  Ready",
            bd=0,
            relief=tk.FLAT,
            anchor=tk.W,
            bg=HEADER_BAR_COLOR,
            fg="#AAAAAA",
            font=STATUS_FONT,
            padx=PAD_SM,
        )
        self.status_bar.pack(side=tk.LEFT, fill=tk.X, expand=True)

        self.progress_bar = ttk.Progressbar(status_frame, mode="indeterminate", length=120)

    def _bind_shortcuts(self):
        """Bind keyboard shortcuts."""
        self.root.bind("<Control-q>", lambda e: self.on_closing())
        self.root.bind("<Control-o>", lambda e: self.import_files())
        self.root.bind("<Control-r>", lambda e: self.process_data())
        self.root.bind("<F5>", lambda e: self.process_data())

    # ────────────────────────────────────────────────────────────────────────
    # WINDOW MANAGEMENT
    # ────────────────────────────────────────────────────────────────────────

    def on_closing(self):
        """Properly cleanup resources before closing the application."""
        try:
            self._save_user_settings()
        except Exception:
            pass
        try:
            clear_env_keys()
        except Exception:
            pass
        try:
            plt.close("all")
        except Exception as e:
            print(f"Error closing matplotlib figures: {e}")
        finally:
            self.root.quit()
            self.root.destroy()

    def _load_user_settings(self):
        """Load persisted user settings (cable loss, VSWR limits)."""
        try:
            settings_path = os.path.join(get_user_data_dir(), "user_settings.json")
            if os.path.exists(settings_path):
                with open(settings_path, "r", encoding="utf-8") as f:
                    settings = json.load(f)
                # VSWR limits
                for key in (
                    "saved_limit1_freq1",
                    "saved_limit1_freq2",
                    "saved_limit1_start",
                    "saved_limit1_stop",
                    "saved_limit2_freq1",
                    "saved_limit2_freq2",
                    "saved_limit2_start",
                    "saved_limit2_stop",
                ):
                    if key in settings:
                        setattr(self, key, float(settings[key]))
        except Exception:
            pass  # silently fall back to defaults

    def _save_user_settings(self):
        """Persist user settings to disk."""
        try:
            settings = {}
            for key in (
                "saved_limit1_freq1",
                "saved_limit1_freq2",
                "saved_limit1_start",
                "saved_limit1_stop",
                "saved_limit2_freq1",
                "saved_limit2_freq2",
                "saved_limit2_start",
                "saved_limit2_stop",
            ):
                if hasattr(self, key):
                    settings[key] = getattr(self, key)
            settings_path = os.path.join(get_user_data_dir(), "user_settings.json")
            os.makedirs(os.path.dirname(settings_path), exist_ok=True)
            with open(settings_path, "w", encoding="utf-8") as f:
                json.dump(settings, f, indent=2)
        except Exception:
            pass

    def update_status(self, message):
        """Update the status bar with a new message."""
        self.status_bar.config(text=message)
        self.root.update_idletasks()

    # ────────────────────────────────────────────────────────────────────────
    # RECENT FILES MANAGEMENT
    # ────────────────────────────────────────────────────────────────────────

    def load_recent_files(self):
        """Load recent files from user data directory."""
        try:
            recent_files_path = os.path.join(get_user_data_dir(), "recent_files.json")
            if os.path.exists(recent_files_path):
                with open(recent_files_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    self.recent_files = data.get("recent_files", [])
        except Exception as e:
            print(f"Could not load recent files: {e}")
            self.recent_files = []

    def save_recent_files(self):
        """Save recent files to user data directory."""
        try:
            recent_files_path = os.path.join(get_user_data_dir(), "recent_files.json")
            with open(recent_files_path, "w", encoding="utf-8") as f:
                json.dump({"recent_files": self.recent_files}, f, indent=2)
        except Exception as e:
            print(f"Could not save recent files: {e}")

    def add_recent_file(self, filepath):
        """Add a file to the recent files list."""
        if filepath in self.recent_files:
            self.recent_files.remove(filepath)
        self.recent_files.insert(0, filepath)
        self.recent_files = self.recent_files[: self.max_recent_files]
        self.save_recent_files()
        self.update_recent_files_menu()

    def update_recent_files_menu(self):
        """Update the recent files submenu."""
        self.recent_menu.delete(0, tk.END)

        if not self.recent_files:
            self.recent_menu.add_command(label="(No recent files)", state=tk.DISABLED)
        else:
            for filepath in self.recent_files:
                filename = os.path.basename(filepath)
                self.recent_menu.add_command(
                    label=filename, command=lambda f=filepath: self.open_recent_file(f)
                )
            self.recent_menu.add_separator()
            self.recent_menu.add_command(
                label="Clear Recent Files", command=self.clear_recent_files
            )

    def open_recent_file(self, filepath):
        """Open a file from the recent files list."""
        if not os.path.exists(filepath):
            messagebox.showerror("File Not Found", f"The file no longer exists:\n{filepath}")
            self.recent_files.remove(filepath)
            self.save_recent_files()
            self.update_recent_files_menu()
            return

        try:
            with open(filepath, "r", encoding="utf-8") as f:
                content = f.read(500)

            self.reset_data()

            if "Total Radiated Power Test" in content:
                self.scan_type.set("active")
                self.TRP_file_path = filepath
                self.update_status(f"Loaded TRP file: {os.path.basename(filepath)}")
                print(f"Loaded TRP file from recent: {filepath}")
                self.update_visibility()

            elif "Horizontal Polarization" in content or "Vertical Polarization" in content:
                self.scan_type.set("passive")

                if "Horizontal Polarization" in content:
                    self.hpol_file_path = filepath
                    self.update_status(f"Loaded HPOL file: {os.path.basename(filepath)}")
                    print(f"Loaded HPOL file from recent: {filepath}")
                else:
                    self.vpol_file_path = filepath
                    self.update_status(f"Loaded VPOL file: {os.path.basename(filepath)}")
                    print(f"Loaded VPOL file from recent: {filepath}")

                self.update_visibility()
            else:
                messagebox.showwarning(
                    "Unknown File Type",
                    "Could not determine the file type.\n\nExpected one of:\n"
                    "- Total Radiated Power Test (Active)\n"
                    "- Horizontal Polarization (Passive)\n"
                    "- Vertical Polarization (Passive)",
                )

        except FileNotFoundError:
            messagebox.showerror(
                "File Not Found",
                "The selected file could not be found.\nIt may have been moved or deleted.",
            )
        except PermissionError:
            messagebox.showerror(
                "Permission Denied",
                "Cannot read the selected file.\nCheck file permissions.",
            )
        except ValueError as e:
            messagebox.showerror(
                "Invalid File",
                f"The file format could not be recognized:\n{str(e)}",
            )
        except Exception as e:
            messagebox.showerror("Error", f"Could not open file:\n{type(e).__name__}: {str(e)}")

    def clear_recent_files(self):
        """Clear the recent files list."""
        self.recent_files = []
        self.save_recent_files()
        self.update_recent_files_menu()
        self.update_status("Recent files cleared")

    # ────────────────────────────────────────────────────────────────────────
    # VISIBILITY MANAGEMENT
    # ────────────────────────────────────────────────────────────────────────

    def update_visibility(self):
        """Update widget visibility based on current scan type.

        The params_frame is shown/hidden as a whole for active/vswr modes.
        Within params_frame, individual widgets toggle for passive sub-modes.
        Action buttons use state=DISABLED to stay visible but inactive.
        """
        # Remove converter button if present (pack-based in actions_frame)
        if hasattr(self, "convert_files_button"):
            self.convert_files_button.pack_forget()

        # Re-pack standard buttons if they were removed by converter tools
        if not self.btn_view_results.winfo_manager():
            self.btn_view_results.pack(side=tk.LEFT, padx=(0, WIDGET_GAP))
        if not self.btn_save_to_file.winfo_manager():
            self.btn_save_to_file.pack(side=tk.LEFT, padx=WIDGET_GAP)
        if not self.btn_settings.winfo_manager():
            self.btn_settings.pack(side=tk.RIGHT, padx=(WIDGET_GAP, 0))
        if not self.btn_import.winfo_manager():
            self.btn_import.pack(side=tk.RIGHT, padx=PAD_SM)

        scan_type_value = self.scan_type.get()

        if scan_type_value == "active":
            self.params_frame.grid_remove()
            self.btn_view_results.config(state=tk.NORMAL)
            self.btn_save_to_file.config(state=tk.DISABLED)
            self.btn_settings.config(state=tk.NORMAL)

        elif scan_type_value == "passive":
            self.params_frame.grid(row=2, column=0, sticky="ew", padx=PAD_LG, pady=PAD_SM)
            if self.passive_scan_type.get() == "G&D":
                self.label_frequency.grid_remove()
                self.frequency_dropdown.grid_remove()
                self.btn_save_to_file.config(state=tk.DISABLED)
            else:
                self.label_frequency.grid(row=0, column=0, sticky="w", padx=(0, PAD_SM), pady=2)
                self.frequency_dropdown.grid(row=0, column=1, sticky="w", padx=PAD_SM, pady=2)
                self.btn_save_to_file.config(state=tk.NORMAL)

            self.label_cable_loss.grid(row=1, column=0, sticky="w", padx=(0, PAD_SM), pady=2)
            self.cable_loss_input.grid(row=1, column=1, sticky="w", padx=PAD_SM, pady=2)
            self.btn_view_results.config(state=tk.NORMAL)
            self.btn_settings.config(state=tk.NORMAL)

        elif scan_type_value == "vswr":
            self.params_frame.grid_remove()
            self.btn_view_results.config(state=tk.DISABLED)
            self.btn_save_to_file.config(state=tk.DISABLED)

    def update_passive_frequency_list(self):
        """Update the frequency dropdown from HPOL file."""
        self.freq_list = extract_passive_frequencies(self.hpol_file_path)

        if self.freq_list:
            self.frequency_dropdown["values"] = self.freq_list
            self.selected_frequency.set(str(self.freq_list[0]))
            self.frequency_dropdown["state"] = "readonly"
            self._measurement_context["frequencies"] = list(self.freq_list)
        else:
            self.frequency_dropdown["values"] = []
            self.selected_frequency.set("")
            self.frequency_dropdown["state"] = "disabled"
