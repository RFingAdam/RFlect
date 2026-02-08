"""
Base Protocol - Type definitions for RFlect GUI mixins

This module provides type hints and protocol definitions so that
IDE type checkers can understand the mixin pattern used in the GUI.
"""

from typing import TYPE_CHECKING, Optional, List, Any, Protocol
import tkinter as tk
from tkinter import ttk

if TYPE_CHECKING:
    import numpy.typing as npt


class AntennaPlotGUIProtocol(Protocol):
    """
    Protocol defining the interface that mixins expect from the main GUI class.

    This allows IDEs to understand the mixin pattern and provide proper
    type checking and autocompletion.
    """

    # ────────────────────────────────────────────────────────────────────────
    # Tkinter Root and Widgets
    # ────────────────────────────────────────────────────────────────────────
    root: tk.Tk
    log_text: tk.Text
    status_bar: tk.Label
    progress_bar: Any  # ttk.Progressbar

    # Container frames
    header_frame: tk.Frame
    controls_frame: Any  # ttk.LabelFrame
    params_frame: Any  # ttk.LabelFrame
    actions_frame: tk.Frame

    # Buttons
    btn_import: tk.Button
    btn_view_results: tk.Button
    btn_save_to_file: tk.Button
    btn_settings: tk.Button

    # Labels
    title_label: tk.Label
    label_scan_type: Any  # ttk.Label
    label_cable_loss: Any  # ttk.Label
    label_frequency: Any  # ttk.Label

    # Entry widgets
    cable_loss_input: Any  # ttk.Entry
    frequency_dropdown: ttk.Combobox

    # ────────────────────────────────────────────────────────────────────────
    # Tkinter Variables
    # ────────────────────────────────────────────────────────────────────────
    scan_type: tk.StringVar
    passive_scan_type: tk.StringVar
    selected_frequency: tk.StringVar
    frequency_var: tk.StringVar
    cable_loss: tk.StringVar

    # Settings variables
    axis_scale_mode: tk.StringVar
    axis_min: tk.DoubleVar
    axis_max: tk.DoubleVar
    datasheet_plots_var: tk.BooleanVar
    min_max_vswr_var: tk.BooleanVar
    min_max_eff_gain_var: tk.BooleanVar
    cb_groupdelay_sff_var: tk.BooleanVar
    cb_shadowing_var: tk.BooleanVar
    shadow_direction_var: tk.StringVar

    # ────────────────────────────────────────────────────────────────────────
    # File Paths
    # ────────────────────────────────────────────────────────────────────────
    hpol_file_path: Optional[str]
    vpol_file_path: Optional[str]
    TRP_file_path: Optional[str]
    vswr_file_path: Optional[str]
    power_measurement: Optional[str]
    BLPA_HORN_GAIN_STD: Optional[str]

    # ────────────────────────────────────────────────────────────────────────
    # Data Arrays (populated after file loading)
    # ────────────────────────────────────────────────────────────────────────
    freq_list: List[float]
    theta_list: Any  # numpy array
    phi_list: Any  # numpy array
    h_gain_dB: Any  # numpy array
    v_gain_dB: Any  # numpy array
    total_gain_dB: Any  # numpy array
    hpol_far_field: Any  # numpy array
    vpol_far_field: Any  # numpy array

    # Active scan data
    data_points: Any
    theta_angles_deg: Any
    phi_angles_deg: Any
    theta_angles_rad: Any
    phi_angles_rad: Any
    total_power_dBm_2d: Any
    h_power_dBm_2d: Any
    v_power_dBm_2d: Any
    phi_angles_deg_plot: Any
    total_power_dBm_2d_plot: Any

    # ────────────────────────────────────────────────────────────────────────
    # Settings State
    # ────────────────────────────────────────────────────────────────────────
    interpolate_3d_plots: bool
    ecc_analysis_enabled: bool
    shadowing_enabled: bool
    shadow_direction: str

    # VSWR limits
    saved_limit1_freq1: float
    saved_limit1_freq2: float
    saved_limit1_start: float
    saved_limit1_stop: float
    saved_limit2_freq1: float
    saved_limit2_freq2: float
    saved_limit2_start: float
    saved_limit2_stop: float
    saved_min_max_vswr: bool

    # G&D settings
    files_per_band: int
    selected_bands: List[tuple]
    measurement_scenario: str

    # Recent files
    recent_files: List[str]
    max_recent_files: int

    # Version
    CURRENT_VERSION: str

    # ────────────────────────────────────────────────────────────────────────
    # Methods that mixins expect to exist
    # ────────────────────────────────────────────────────────────────────────

    @staticmethod
    def resource_path(relative_path: str) -> str:
        """Get absolute path to resource."""
        ...

    @staticmethod
    def get_user_data_dir() -> str:
        """Get user-specific data directory."""
        ...

    def update_visibility(self) -> None:
        """Update widget visibility based on scan type."""
        ...

    def update_passive_frequency_list(self) -> None:
        """Update frequency dropdown from HPOL file."""
        ...

    def process_data(self) -> None:
        """Process loaded measurement data."""
        ...

    def _process_data_without_plotting(self) -> bool:
        """Process data without generating plots."""
        ...

    def log_message(self, message: str) -> None:
        """Log a message to the GUI log area."""
        ...

    def update_status(self, message: str) -> None:
        """Update the status bar."""
        ...

    def load_recent_files(self) -> None:
        """Load recent files list."""
        ...

    def save_recent_files(self) -> None:
        """Save recent files list."""
        ...

    def add_recent_file(self, filepath: str) -> None:
        """Add a file to recent files."""
        ...

    def update_recent_files_menu(self) -> None:
        """Update the recent files menu."""
        ...
