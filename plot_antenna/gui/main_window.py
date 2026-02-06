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
    DARK_BG_COLOR, LIGHT_TEXT_COLOR, ACCENT_BLUE_COLOR, BUTTON_COLOR, HEADER_FONT,
    ERROR_COLOR, WARNING_COLOR, SUCCESS_COLOR, HOVER_COLOR,
)
from ..calculations import extract_passive_frequencies

# Import centralized API key management
from ..api_keys import load_api_key, get_api_key, is_api_key_configured

# Import mixins
from .dialogs_mixin import DialogsMixin
from .ai_chat_mixin import AIChatMixin
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
        self._nf2ff_cache = {}

        # Measurement context for AI awareness
        self._measurement_context = {
            "files_loaded": [],
            "scan_type": None,
            "frequencies": [],
            "data_shape": None,
            "cable_loss_applied": 0.0,
            "nf2ff_applied": False,
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

        # Load API key using centralized module (handles keyring, .env, user file, etc.)
        # The api_keys module auto-loads on import, but we call it explicitly here
        # to ensure the key is available and to log the status
        if not is_api_key_configured():
            # Try loading again (may find .env file now that GUI is initializing)
            load_api_key()

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

    def _create_widgets(self):
        """Create main GUI widgets."""
        # Title
        self.title_label = tk.Label(
            self.root,
            text="RFlect - Antenna Plot Tool",
            bg=DARK_BG_COLOR,
            fg=LIGHT_TEXT_COLOR,
            font=HEADER_FONT,
        )
        self.title_label.grid(row=0, column=0, pady=(10, 0), columnspan=6, sticky="n")

        # Scan type label
        self.label_scan_type = tk.Label(
            self.root, text="Select Measurement Type:", bg=DARK_BG_COLOR, fg=LIGHT_TEXT_COLOR
        )
        self.label_scan_type.grid(row=1, column=0, pady=10, columnspan=2)

        # Initialize scan settings
        self.scan_type = tk.StringVar()
        self.scan_type.set("passive")
        self.passive_scan_type = tk.StringVar()
        self.passive_scan_type.set("VPOL/HPOL")
        self.datasheet_plots_var = tk.BooleanVar(value=False)
        self.cb_groupdelay_sff_var = tk.BooleanVar(value=False)
        self.ecc_analysis_enabled = False

        # Radio buttons for scan type
        active_rb = tk.Radiobutton(
            self.root,
            text="Active",
            variable=self.scan_type,
            value="active",
            background=BUTTON_COLOR,
            foreground=LIGHT_TEXT_COLOR,
            selectcolor=DARK_BG_COLOR,
            activebackground=ACCENT_BLUE_COLOR,
            activeforeground=LIGHT_TEXT_COLOR,
            command=self.update_visibility,
        )
        active_rb.grid(row=2, column=0, pady=5)

        passive_rb = tk.Radiobutton(
            self.root,
            text="Passive",
            variable=self.scan_type,
            value="passive",
            background=BUTTON_COLOR,
            foreground=LIGHT_TEXT_COLOR,
            selectcolor=DARK_BG_COLOR,
            activebackground=ACCENT_BLUE_COLOR,
            activeforeground=LIGHT_TEXT_COLOR,
            command=self.update_visibility,
        )
        passive_rb.grid(row=2, column=1, pady=5)

        return_loss = tk.Radiobutton(
            self.root,
            text="VNA (.csv)",
            variable=self.scan_type,
            value="vswr",
            background=BUTTON_COLOR,
            foreground=LIGHT_TEXT_COLOR,
            selectcolor=DARK_BG_COLOR,
            activebackground=ACCENT_BLUE_COLOR,
            activeforeground=LIGHT_TEXT_COLOR,
            command=self.update_visibility,
        )
        return_loss.grid(row=2, column=2, pady=5)

        # Import button
        self.btn_import = tk.Button(
            self.root,
            text="Import File(s)",
            command=self.import_files,
            bg=ACCENT_BLUE_COLOR,
            fg=LIGHT_TEXT_COLOR,
        )
        self.btn_import.grid(row=2, column=3, columnspan=2, pady=10, padx=15)

        # Cable Loss input
        self.label_cable_loss = tk.Label(
            self.root, text="Cable Loss:", bg=DARK_BG_COLOR, fg=LIGHT_TEXT_COLOR
        )
        self.cable_loss = tk.StringVar(self.root, value="0.0")
        self.cable_loss_input = tk.Entry(
            self.root, textvariable=self.cable_loss, bg=DARK_BG_COLOR, fg=LIGHT_TEXT_COLOR
        )
        self.label_cable_loss.grid(row=4, column=0, pady=5)
        self.cable_loss_input.grid(row=4, column=1, pady=5, padx=5)

        # Frequency selection
        self.available_frequencies = []
        self.selected_frequency = tk.StringVar()
        self.label_frequency = tk.Label(
            self.root, text="Select Frequency:", bg=DARK_BG_COLOR, fg=LIGHT_TEXT_COLOR
        )
        self.label_frequency.grid(row=3, column=0, pady=5)
        self.frequency_dropdown = ttk.Combobox(
            self.root,
            textvariable=self.selected_frequency,
            values=self.available_frequencies,
            state="readonly",
        )
        self.frequency_dropdown.grid(row=3, column=1, pady=5)

        # View Results button
        self.btn_view_results = tk.Button(
            self.root,
            text="View Results",
            command=self.process_data,
            bg=ACCENT_BLUE_COLOR,
            fg=LIGHT_TEXT_COLOR,
        )
        self.btn_view_results.grid(row=5, column=0, pady=10, padx=10)

        # Save Results button
        self.btn_save_to_file = tk.Button(
            self.root,
            text="Save Results to File",
            command=lambda: self.save_results_to_file(),
            bg=ACCENT_BLUE_COLOR,
            fg=LIGHT_TEXT_COLOR,
        )
        self.btn_save_to_file.grid(row=5, column=1, pady=10, padx=10)

        # Settings button
        self.btn_settings = tk.Button(
            self.root,
            text="Settings",
            command=self.show_settings,
            bg=ACCENT_BLUE_COLOR,
            fg=LIGHT_TEXT_COLOR,
        )
        self.btn_settings.grid(row=5, column=3, pady=10, padx=10)

        # 3D plotting settings
        self.axis_scale_mode = tk.StringVar(value="auto")
        self.axis_min = tk.DoubleVar(value=-20.0)
        self.axis_max = tk.DoubleVar(value=6.0)

        # Human shadowing settings
        self.shadowing_enabled = False
        self.shadow_direction = "-X"
        self.cb_shadowing_var = tk.BooleanVar(value=False)
        self.shadow_direction_var = tk.StringVar(value="-X")

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
        if is_api_key_configured():
            tools_menu.add_command(
                label="Generate Report with AI", command=self.generate_ai_report_from_directory
            )
            tools_menu.add_command(label="AI Chat Assistant...", command=self.open_ai_chat)
        else:
            print("[INFO] No AI API key configured. AI features disabled.")
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

        menubar.add_cascade(label="Help", menu=help_menu)

        self.root.config(menu=menubar)

    def _create_log_area(self):
        """Create log text area."""
        log_frame = tk.Frame(self.root)
        log_frame.grid(row=6, column=0, columnspan=4, sticky="nsew")

        self.log_text = ScrolledText.ScrolledText(
            log_frame, wrap=tk.WORD, height=10, bg=DARK_BG_COLOR, fg=LIGHT_TEXT_COLOR
        )
        self.log_text.grid(row=0, column=0, sticky="nsew")

        self.root.grid_rowconfigure(6, weight=1)
        self.root.grid_columnconfigure(0, weight=1)
        self.root.grid_columnconfigure(1, weight=1)
        self.root.grid_columnconfigure(2, weight=1)
        self.root.grid_columnconfigure(3, weight=1)

        # Configure log text tags for color-coded messages
        self.log_text.tag_config("log_info", foreground=LIGHT_TEXT_COLOR)
        self.log_text.tag_config("log_success", foreground=SUCCESS_COLOR)
        self.log_text.tag_config("log_warning", foreground=WARNING_COLOR)
        self.log_text.tag_config("log_error", foreground=ERROR_COLOR)

        # Redirect stdout and stderr
        sys.stdout = DualOutput(self.log_text, sys.stdout)
        sys.stderr = DualOutput(self.log_text, sys.stderr)

    def _create_status_bar(self):
        """Create status bar with progress indicator."""
        status_frame = tk.Frame(self.root, bg=DARK_BG_COLOR)
        status_frame.grid(row=7, column=0, columnspan=4, sticky="ew")

        self.status_bar = tk.Label(
            status_frame,
            text="Ready",
            bd=1,
            relief=tk.SUNKEN,
            anchor=tk.W,
            bg=DARK_BG_COLOR,
            fg=LIGHT_TEXT_COLOR,
            font=("Arial", 9),
        )
        self.status_bar.pack(side=tk.LEFT, fill=tk.X, expand=True)

        self.progress_bar = ttk.Progressbar(
            status_frame, mode="indeterminate", length=120
        )

    def _bind_shortcuts(self):
        """Bind keyboard shortcuts."""
        self.root.bind("<Control-q>", lambda e: self.on_closing())
        self.root.bind("<Control-o>", lambda e: self.import_files())

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
                    "saved_limit1_freq1", "saved_limit1_freq2",
                    "saved_limit1_start", "saved_limit1_stop",
                    "saved_limit2_freq1", "saved_limit2_freq2",
                    "saved_limit2_start", "saved_limit2_stop",
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
                "saved_limit1_freq1", "saved_limit1_freq2",
                "saved_limit1_start", "saved_limit1_stop",
                "saved_limit2_freq1", "saved_limit2_freq2",
                "saved_limit2_start", "saved_limit2_stop",
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

        Labels/inputs use grid_remove() when not applicable.
        Action buttons use state=DISABLED to stay visible but inactive.
        """
        # Remove converter button if present
        if hasattr(self, "convert_files_button"):
            self.convert_files_button.grid_remove()

        scan_type_value = self.scan_type.get()

        if scan_type_value == "active":
            self.label_cable_loss.grid_remove()
            self.cable_loss_input.grid_remove()
            self.label_frequency.grid_remove()
            self.frequency_dropdown.grid_remove()
            self.btn_view_results.config(state=tk.NORMAL)
            self.btn_view_results.grid(row=5, column=0, pady=10)
            self.btn_save_to_file.config(state=tk.DISABLED)
            self.btn_settings.config(state=tk.DISABLED)

        elif scan_type_value == "passive":
            if self.passive_scan_type.get() == "G&D":
                self.label_frequency.grid_remove()
                self.frequency_dropdown.grid_remove()
                self.btn_save_to_file.config(state=tk.DISABLED)
            else:
                self.label_frequency.grid(row=3, column=0, pady=5)
                self.frequency_dropdown.grid(row=3, column=1, pady=5)
                self.btn_save_to_file.config(state=tk.NORMAL)
                self.btn_save_to_file.grid(row=5, column=1, pady=10)

            self.label_cable_loss.grid(row=4, column=0, pady=5)
            self.cable_loss_input.grid(row=4, column=1, pady=5, padx=5)
            self.btn_view_results.config(state=tk.NORMAL)
            self.btn_view_results.grid(row=5, column=0, pady=10)
            self.btn_settings.config(state=tk.NORMAL)
            self.btn_settings.grid()

        elif scan_type_value == "vswr":
            self.label_cable_loss.grid_remove()
            self.cable_loss_input.grid_remove()
            self.label_frequency.grid_remove()
            self.frequency_dropdown.grid_remove()
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
