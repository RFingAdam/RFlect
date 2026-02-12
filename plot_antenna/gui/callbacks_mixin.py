"""
CallbacksMixin - Core callback methods for RFlect GUI

This mixin provides the main data processing callbacks:
- import_files: File import logic for active/passive/VSWR scans
- process_data: Main data processing and visualization
- Data reset and helper functions
"""

from __future__ import annotations

import os
import tkinter as tk
from tkinter import filedialog, messagebox
from tkinter.simpledialog import askinteger, askstring
from typing import TYPE_CHECKING, Optional, List, Any

import numpy as np
import matplotlib.pyplot as plt

from ..config import ACCENT_BLUE_COLOR, LIGHT_TEXT_COLOR

from ..file_utils import (
    read_passive_file,
    read_active_file,
    check_matching_files,
    process_gd_file,
    parse_touchstone_to_dataframe,
)
from ..calculations import (
    determine_polarization,
    calculate_passive_variables,
    calculate_active_variables,
    extrapolate_pattern,
    apply_directional_human_shadow,
    angles_match,
)
from ..plotting import (
    plot_2d_passive_data,
    plot_passive_3d_component,
    plot_active_2d_data,
    plot_active_3d_data,
    plot_gd_data,
    process_vswr_files,
    _prepare_gain_grid,
    generate_maritime_plots,
    generate_advanced_analysis_plots,
)
from ..groupdelay import process_groupdelay_files
from .utils import calculate_min_max_parameters, display_parameter_table

if TYPE_CHECKING:
    from .base_protocol import AntennaPlotGUIProtocol


class CallbacksMixin:
    """Mixin class providing core callback methods for AntennaPlotGUI."""

    # Type hints for IDE support (satisfied by main class)
    root: tk.Tk
    scan_type: tk.StringVar
    passive_scan_type: tk.StringVar
    selected_frequency: tk.StringVar
    freq_list: Optional[List[float]]  # Can be None after reset
    cable_loss: tk.StringVar
    datasheet_plots_var: tk.BooleanVar
    min_max_eff_gain_var: tk.BooleanVar
    axis_scale_mode: tk.StringVar
    axis_min: tk.DoubleVar
    axis_max: tk.DoubleVar
    interpolate_3d_plots: bool
    ecc_analysis_enabled: bool
    shadowing_enabled: bool
    shadow_direction: str

    # File paths
    hpol_file_path: Optional[str]
    vpol_file_path: Optional[str]
    TRP_file_path: Optional[str]
    vswr_file_path: Optional[str]
    data: Any

    # Data arrays
    theta_list: Any
    phi_list: Any
    h_gain_dB: Any
    v_gain_dB: Any
    total_gain_dB: Any
    hpol_far_field: Any
    vpol_far_field: Any
    theta_angles_deg: Any
    phi_angles_deg: Any
    theta_angles_rad: Any
    phi_angles_rad: Any
    total_power_dBm_2d: Any
    h_power_dBm_2d: Any
    v_power_dBm_2d: Any
    data_points: Any
    phi_angles_deg_plot: Any
    total_power_dBm_2d_plot: Any

    # Widgets
    btn_view_results: tk.Button
    btn_save_to_file: tk.Button
    frequency_dropdown: Any  # ttk.Combobox
    log_text: tk.Text

    # G&D settings
    files_per_band: int
    selected_bands: List[tuple]
    measurement_scenario: str
    cb_groupdelay_sff_var: tk.BooleanVar
    min_max_vswr_var: tk.BooleanVar

    # VSWR limit settings
    saved_limit1_freq1: float
    saved_limit1_freq2: float
    saved_limit1_start: float
    saved_limit1_stop: float
    saved_limit2_freq1: float
    saved_limit2_freq2: float
    saved_limit2_start: float
    saved_limit2_stop: float

    # GUI attributes (defined by main_window.py, declared here for type checking)
    btn_import: tk.Button
    btn_view_results: tk.Button
    btn_save_to_file: tk.Button
    btn_settings: tk.Button
    progress_bar: Any
    _processing_lock: bool
    _active_figures: List[Any]
    _extrapolation_cache: dict
    _measurement_context: dict

    # Method declarations for type checking only (not defined at runtime to avoid MRO conflicts)
    if TYPE_CHECKING:

        def update_visibility(self) -> None: ...
        def update_passive_frequency_list(self) -> None: ...
        def add_recent_file(self, filepath: str) -> None: ...
        def update_status(self, message: str) -> None: ...

    # ────────────────────────────────────────────────────────────────────────
    # MARITIME HELPERS
    # ────────────────────────────────────────────────────────────────────────

    def _parse_theta_cuts(self):
        """Parse the comma-separated theta cuts string into a list of floats."""
        try:
            return [
                float(x.strip()) for x in self.horizon_theta_cuts_var.get().split(",") if x.strip()
            ]
        except (ValueError, AttributeError):
            return [60, 70, 80, 90, 100, 110, 120]

    # ────────────────────────────────────────────────────────────────────────
    # DATA RESET
    # ────────────────────────────────────────────────────────────────────────

    def reset_data(self):
        """Reset all data variables and close matplotlib figures."""
        for fig in self._active_figures:
            try:
                plt.close(fig)
            except Exception:
                pass
        self._active_figures.clear()
        self._extrapolation_cache.clear()
        self.data = None
        self.hpol_file_path = None
        self.vpol_file_path = None
        self.freq_list = None
        self.TRP_file_path = None

    # ────────────────────────────────────────────────────────────────────────
    # G&D MIN/MAX CALCULATIONS
    # ────────────────────────────────────────────────────────────────────────

    def calculate_min_max_eff_gain(self, file_paths, bands):
        """
        For each band, calculate the min, max, and average of Gain (dBi) and Efficiency.
        Efficiency is in percentage; convert from % to fraction and dB.
        """
        results = []
        all_data = []
        for fpath in file_paths:
            gd_data = process_gd_file(fpath)
            freq = np.array(gd_data["Frequency"])

            gain_dBi = np.array(gd_data["Gain"])
            eff_percent = np.array(gd_data["Efficiency"])
            eff_fraction = eff_percent / 100.0
            eff_fraction[eff_fraction <= 0] = 1e-12
            eff_dB = 10 * np.log10(eff_fraction)

            avg_eff_fraction = np.mean(eff_fraction)
            avg_eff_dB = 10 * np.log10(avg_eff_fraction)

            all_data.append((fpath, freq, gain_dBi, eff_dB, avg_eff_fraction, avg_eff_dB))

        for i, (freq_min, freq_max) in enumerate(bands, start=1):
            band_label = f"({freq_min}-{freq_max} MHz)"
            band_results = []
            for fpath, freq, gain_dBi, eff_dB, avg_eff_fraction, avg_eff_dB in all_data:
                within_range = (freq >= freq_min) & (freq <= freq_max)
                min_gain = gain_dBi[within_range].min() if np.any(within_range) else None
                max_gain = gain_dBi[within_range].max() if np.any(within_range) else None
                min_eff = eff_dB[within_range].min() if np.any(within_range) else None
                max_eff = eff_dB[within_range].max() if np.any(within_range) else None
                band_results.append(
                    (
                        os.path.basename(fpath),
                        min_gain,
                        max_gain,
                        min_eff,
                        max_eff,
                        avg_eff_fraction,
                        avg_eff_dB,
                    )
                )
            results.append((band_label, band_results))

        return results

    def display_eff_gain_table(self, results):
        """Display the calculated min, max, and average efficiency along with gain."""
        result_window = tk.Toplevel()
        result_window.title("Min/Max Gain & Efficiency Results by Band")

        row = 0
        for band_label, band_data in results:
            tk.Label(result_window, text=band_label, font=("Arial", 12, "bold")).grid(
                row=row, column=0, columnspan=7, pady=5
            )
            row += 1

            headers = [
                "File",
                "Min Gain(dB)",
                "Max Gain(dB)",
                "Min Eff(dB)",
                "Max Eff(dB)",
                "Avg Eff(%)",
                "Avg Eff(dB)",
            ]
            for col, header in enumerate(headers):
                tk.Label(result_window, text=header, font=("Arial", 10, "bold")).grid(
                    row=row, column=col, padx=10, pady=5
                )
            row += 1

            for (
                file,
                min_gain,
                max_gain,
                min_eff,
                max_eff,
                avg_eff_fraction,
                avg_eff_dB,
            ) in band_data:
                avg_eff_percent = avg_eff_fraction * 100
                tk.Label(result_window, text=file).grid(row=row, column=0, padx=10, pady=5)
                tk.Label(
                    result_window, text=f"{min_gain:.2f}" if min_gain is not None else "N/A"
                ).grid(row=row, column=1, padx=10, pady=5)
                tk.Label(
                    result_window, text=f"{max_gain:.2f}" if max_gain is not None else "N/A"
                ).grid(row=row, column=2, padx=10, pady=5)
                tk.Label(
                    result_window, text=f"{min_eff:.2f}" if min_eff is not None else "N/A"
                ).grid(row=row, column=3, padx=10, pady=5)
                tk.Label(
                    result_window, text=f"{max_eff:.2f}" if max_eff is not None else "N/A"
                ).grid(row=row, column=4, padx=10, pady=5)
                tk.Label(result_window, text=f"{avg_eff_percent:.2f}").grid(
                    row=row, column=5, padx=10, pady=5
                )
                tk.Label(result_window, text=f"{avg_eff_dB:.2f}").grid(
                    row=row, column=6, padx=10, pady=5
                )
                row += 1

            row += 1

    def display_final_summary(self, results):
        """Create a final summary table with Min, Max, and Average Efficiency for each band."""
        band_summary = {}
        for band_label, band_data in results:
            eff_fractions = [x[5] for x in band_data if x[5] is not None]
            eff_dBs = [x[6] for x in band_data if x[6] is not None]
            gain_mins = [x[1] for x in band_data if x[1] is not None]
            gain_maxs = [x[2] for x in band_data if x[2] is not None]

            avg_eff_fraction = np.mean(eff_fractions) if eff_fractions else None
            avg_eff_percent = avg_eff_fraction * 100 if avg_eff_fraction else None
            avg_eff_dB = np.mean(eff_dBs) if eff_dBs else None
            band_min_gain = min(gain_mins) if gain_mins else None
            band_max_gain = max(gain_maxs) if gain_maxs else None
            band_summary[band_label] = (avg_eff_percent, avg_eff_dB, band_min_gain, band_max_gain)

        summary_window = tk.Toplevel(self.root)
        summary_window.title("Final Summary")

        tk.Label(
            summary_window, text="Final Summary of All Bands", font=("Arial", 12, "bold")
        ).grid(row=0, column=0, columnspan=len(results) + 1, pady=10)

        tk.Label(summary_window, text="Parameter", font=("Arial", 10, "bold")).grid(
            row=1, column=0, padx=10, pady=5
        )
        for col, (band_label, _) in enumerate(results, start=1):
            tk.Label(summary_window, text=band_label, font=("Arial", 10, "bold")).grid(
                row=1, column=col, padx=10, pady=5
            )

        parameters = ["Avg Eff(%)", "Avg Eff(dB)", "Min Gain(dBi)", "Max Gain(dBi)"]
        for row, param in enumerate(parameters, start=2):
            tk.Label(summary_window, text=param, font=("Arial", 10, "bold")).grid(
                row=row, column=0, sticky="w", padx=10, pady=5
            )

        for col, (band_label, _) in enumerate(results, start=1):
            avg_eff_percent, avg_eff_dB, min_gain, max_gain = band_summary[band_label]
            values = [avg_eff_percent, avg_eff_dB, min_gain, max_gain]
            for row, value in enumerate(values, start=2):
                tk.Label(summary_window, text=f"{value:.2f}" if value is not None else "N/A").grid(
                    row=row, column=col, padx=10, pady=5
                )

    # ────────────────────────────────────────────────────────────────────────
    # FILE IMPORT
    # ────────────────────────────────────────────────────────────────────────

    def import_files(self):
        """Import TRP or HPOL/VPOL data files for analysis."""
        if self.hpol_file_path or self.TRP_file_path:
            if not messagebox.askyesno("Confirm", "This will discard current data. Continue?"):
                return
        self.reset_data()

        if self.scan_type.get() == "active":
            self.TRP_file_path = filedialog.askopenfilename(
                title="Select the TRP Data File",
                filetypes=[("Text files", "*.txt"), ("All files", "*.*")],
            )
            if self.TRP_file_path:
                self.add_recent_file(self.TRP_file_path)
                self.update_status(f"Loaded TRP file: {os.path.basename(self.TRP_file_path)}")
                self._measurement_context["files_loaded"].append(
                    {
                        "path": os.path.basename(self.TRP_file_path),
                        "type": "TRP",
                    }
                )
                self._measurement_context["scan_type"] = "active"

        elif self.scan_type.get() == "passive" and self.passive_scan_type.get() == "G&D":
            self._import_gd_files()

        elif (
            self.scan_type.get() == "passive"
            and self.passive_scan_type.get() == "VPOL/HPOL"
            and self.ecc_analysis_enabled
        ):
            self._import_ecc_files()

        elif (
            self.scan_type.get() == "passive"
            and self.passive_scan_type.get() == "VPOL/HPOL"
            and not self.ecc_analysis_enabled
        ):
            self._import_vpol_hpol_files()

        elif self.scan_type.get() == "vswr":
            self._import_vswr_files()

    def _import_gd_files(self):
        """Handle G&D file imports with optional min/max analysis."""
        if self.min_max_eff_gain_var.get():
            # Show scenario selection dialog
            scenario_window = tk.Toplevel(self.root)
            scenario_window.title("Select Measurement Scenario")
            scenario_window.geometry("400x300")
            scenario_window.grab_set()

            scenario_var = tk.StringVar(value="")

            tk.Label(scenario_window, text="Select scenario:").grid(
                row=0, column=0, sticky="w", padx=10, pady=(10, 5)
            )

            tk.Radiobutton(
                scenario_window,
                text="WiFi 6e (2.4, 5, 6 GHz)",
                variable=scenario_var,
                value="WiFi_6e",
            ).grid(row=1, column=0, sticky="w", padx=20, pady=2)

            tk.Radiobutton(
                scenario_window,
                text="LoRa 863 MHz (863-870 MHz)",
                variable=scenario_var,
                value="LoRa_863",
            ).grid(row=2, column=0, sticky="w", padx=20, pady=2)

            tk.Radiobutton(
                scenario_window,
                text="LoRa 902 MHz (902-928 MHz)",
                variable=scenario_var,
                value="LoRa_902",
            ).grid(row=3, column=0, sticky="w", padx=20, pady=2)

            tk.Radiobutton(
                scenario_window,
                text="LoRa 863-928 MHz (dual band)",
                variable=scenario_var,
                value="LoRa_863_928",
            ).grid(row=4, column=0, sticky="w", padx=20, pady=2)

            tk.Label(scenario_window, text="Number of files per band:").grid(
                row=5, column=0, sticky="w", padx=10, pady=5
            )
            files_per_band_var = tk.IntVar(value=4)
            tk.Entry(scenario_window, textvariable=files_per_band_var, width=5).grid(
                row=5, column=1, padx=5, pady=5
            )

            def on_scenario_ok():
                chosen = scenario_var.get()
                if chosen == "":
                    messagebox.showerror("Error", "Please select a measurement scenario.")
                    return

                self.files_per_band = files_per_band_var.get()

                if chosen == "LoRa_863":
                    self.selected_bands = [(863.0, 870.0)]
                elif chosen == "LoRa_902":
                    self.selected_bands = [(902.0, 928.0)]
                elif chosen == "LoRa_863_928":
                    self.selected_bands = [(863.0, 928.0)]
                elif chosen == "WiFi_6e":
                    self.selected_bands = [(2400.0, 2500.0), (4900.0, 5925.0), (5925.0, 7125.0)]
                else:
                    self.selected_bands = []

                self.measurement_scenario = chosen
                scenario_window.destroy()

            ok_button = tk.Button(
                scenario_window,
                text="OK",
                command=on_scenario_ok,
                bg=ACCENT_BLUE_COLOR,
                fg=LIGHT_TEXT_COLOR,
            )
            ok_button.grid(row=6, column=0, pady=20, padx=10)

            self.root.wait_window(scenario_window)

            if (
                self.measurement_scenario in ["LoRa_863", "LoRa_902", "LoRa_863_928", "WiFi_6e"]
                and self.selected_bands
            ):
                all_band_results = []
                for i, (f_min, f_max) in enumerate(self.selected_bands, start=1):
                    band_label = f"({f_min}-{f_max} MHz)"
                    messagebox.showinfo(
                        "Band Selection",
                        f"Please select {self.files_per_band} G&D files for {band_label}",
                    )
                    file_paths = []
                    for _ in range(self.files_per_band):
                        filepath = filedialog.askopenfilename(
                            filetypes=[("Text files", "*.txt")],
                            title=f"Select File for {band_label}",
                        )
                        if not filepath:
                            return
                        file_paths.append(filepath)

                    band_results = self.calculate_min_max_eff_gain(file_paths, [(f_min, f_max)])
                    self.display_eff_gain_table(band_results)
                    all_band_results.extend(band_results)

                self.display_final_summary(all_band_results)
            else:
                # Fallback manual band definition
                num_files = askinteger(
                    "Input", "How many G&D files would you like to import?", parent=self.root
                )
                if not num_files:
                    return
                file_paths = []
                for _ in range(num_files):
                    filepath = filedialog.askopenfilename(filetypes=[("Text files", "*.txt")])
                    if not filepath:
                        return
                    file_paths.append(filepath)

                num_bands = askinteger(
                    "Input", "How many frequency bands would you like to define?"
                )
                if not num_bands:
                    return

                bands = []
                for i in range(num_bands):
                    freq_range_input = askstring(
                        "Input",
                        f"Enter frequency range for Band {i+1} as 'min,max' (MHz):",
                        parent=self.root,
                    )
                    if not freq_range_input:
                        messagebox.showerror("Error", "No frequency range entered.")
                        return
                    try:
                        freq_min, freq_max = map(float, freq_range_input.split(","))
                        bands.append((freq_min, freq_max))
                    except ValueError:
                        messagebox.showerror(
                            "Error",
                            "Invalid frequency range. Please enter values in 'min,max' format.",
                        )
                        return

                results = self.calculate_min_max_eff_gain(file_paths, bands)
                self.display_eff_gain_table(results)
                self.display_final_summary(results)
        else:
            # Original non-min_max_eff_gain logic for G&D
            num_files = askinteger(
                "Input", "How many files would you like to import?", parent=self.root
            )
            if not num_files:
                return
            datasets = []
            file_names = []
            for _ in range(num_files):
                filepath = filedialog.askopenfilename(filetypes=[("Text files", "*.txt")])
                if not filepath:
                    return
                file_name = os.path.basename(filepath).replace(".txt", "")
                file_names.append(file_name)
                data = process_gd_file(filepath)
                datasets.append(data)

            x_range_input = askstring(
                "Input",
                "Enter x-axis range as 'min,max' (or leave blank for auto-scale):",
                parent=self.root,
            )
            plot_gd_data(datasets, file_names, x_range_input)

    def _import_ecc_files(self):
        """Handle ECC (Envelope Correlation Coefficient) analysis file imports."""
        n_scans = askinteger(
            "ECC Analysis", "How many antenna placements to compare?", parent=self.root
        )
        if not n_scans:
            return

        scan_labels = []
        for i in range(n_scans):
            lbl = askstring(
                "Placement Label",
                f"Enter a short label for placement #{i+1} (e.g. 'Pos1', 'Pos2'):",
                parent=self.root,
            )
            scan_labels.append(lbl or f"Scan {i+1}")

        plt.figure(figsize=(8, 5))

        for idx, placement in enumerate(scan_labels, start=1):
            nbands = askinteger(
                f"ECC Analysis - {placement}",
                f"How many bands for '{placement}'?",
                parent=self.root,
            )
            if not nbands:
                continue

            for b in range(1, nbands + 1):
                a1_h = filedialog.askopenfilename(title=f"{placement} - Band {b}: Ant1 HPOL")
                a1_v = filedialog.askopenfilename(title=f"{placement} - Band {b}: Ant1 VPOL")
                a2_h = filedialog.askopenfilename(title=f"{placement} - Band {b}: Ant2 HPOL")
                a2_v = filedialog.askopenfilename(title=f"{placement} - Band {b}: Ant2 VPOL")
                if not all((a1_h, a1_v, a2_h, a2_v)):
                    messagebox.showwarning("ECC", f"{placement}, Band {b}: incomplete, skipping")
                    continue

                h1, *_ = read_passive_file(a1_h)
                v1, *_ = read_passive_file(a1_v)
                h2, *_ = read_passive_file(a2_h)
                v2, *_ = read_passive_file(a2_v)

                ecc_results = []
                for h1p, v1p, h2p, v2p in zip(h1, v1, h2, v2):
                    if abs(h1p["frequency"] - h2p["frequency"]) > 0.1:
                        continue
                    f = h1p["frequency"]

                    # Ludwig-3 convention: HPOL → E_φ, VPOL → E_θ
                    mag_h1 = np.array(h1p["mag"])
                    ph_h1 = np.radians(h1p["phase"])
                    E1_phi = 10 ** (mag_h1 / 20) * np.exp(1j * ph_h1)

                    mag_v1 = np.array(v1p["mag"])
                    ph_v1 = np.radians(v1p["phase"])
                    E1_theta = 10 ** (mag_v1 / 20) * np.exp(1j * ph_v1)

                    mag_h2 = np.array(h2p["mag"])
                    ph_h2 = np.radians(h2p["phase"])
                    E2_phi = 10 ** (mag_h2 / 20) * np.exp(1j * ph_h2)

                    mag_v2 = np.array(v2p["mag"])
                    ph_v2 = np.radians(v2p["phase"])
                    E2_theta = 10 ** (mag_v2 / 20) * np.exp(1j * ph_v2)

                    theta = np.array(h1p["theta"])
                    sin_theta = np.sin(np.radians(theta))

                    inner = E1_theta * np.conj(E2_theta) + E1_phi * np.conj(E2_phi)
                    numerator = np.abs(np.sum(inner * sin_theta)) ** 2

                    D1 = np.sum((np.abs(E1_theta) ** 2 + np.abs(E1_phi) ** 2) * sin_theta)
                    D2 = np.sum((np.abs(E2_theta) ** 2 + np.abs(E2_phi) ** 2) * sin_theta)
                    denom = D1 * D2

                    ecc = numerator / denom if denom else 0.0
                    ecc_results.append((f, ecc))

                if ecc_results:
                    freqs, vals = zip(*sorted(ecc_results))
                    plt.plot(freqs, vals, marker="o", linestyle="-", label=f"{placement} - B{b}")

        plt.title("Envelope Correlation Coefficient vs Frequency")
        plt.xlabel("Frequency (MHz)")
        plt.ylabel("ECC")
        plt.grid(True, linestyle="--")
        plt.legend(loc="best")
        plt.tight_layout()
        plt.show()

    def _import_vpol_hpol_files(self):
        """Handle standard VPOL/HPOL file imports."""
        first_file = filedialog.askopenfilename(
            title="Select the First Data File",
            filetypes=[("Text files", "*.txt"), ("All files", "*.*")],
        )
        if not first_file:
            return

        second_file = filedialog.askopenfilename(
            title="Select the Second Data File",
            filetypes=[("Text files", "*.txt"), ("All files", "*.*")],
        )
        if first_file and second_file:
            first_polarization = determine_polarization(first_file)
            second_polarization = determine_polarization(second_file)

            if first_polarization == second_polarization:
                messagebox.showerror("Error", "Both files cannot be of the same polarization.")
                return

            if first_polarization == "HPol":
                self.hpol_file_path = first_file
                self.vpol_file_path = second_file
            else:
                self.hpol_file_path = second_file
                self.vpol_file_path = first_file

            self.add_recent_file(self.hpol_file_path)
            self.add_recent_file(self.vpol_file_path)
            self.update_status("Loaded HPOL and VPOL files")
            self._measurement_context["files_loaded"].extend(
                [
                    {"path": os.path.basename(self.hpol_file_path), "type": "HPOL"},
                    {"path": os.path.basename(self.vpol_file_path), "type": "VPOL"},
                ]
            )
            self._measurement_context["scan_type"] = "passive"

            match, message = check_matching_files(self.hpol_file_path, self.vpol_file_path)
            if not match:
                messagebox.showerror("Error", message)
                return
            self.update_passive_frequency_list()

    def _import_vswr_files(self):
        """Handle VSWR file imports."""
        if self.cb_groupdelay_sff_var.get() and self.min_max_vswr_var.get():
            messagebox.showerror(
                "Error", "Cannot have both Group Delay/SFF and Min/Max VSWR selected."
            )
            return

        # Scenario 1: Group Delay = False, Min/Max VSWR = False
        if not self.cb_groupdelay_sff_var.get() and not self.min_max_vswr_var.get():
            num_files = askinteger("Input", "How many files do you want to import?")
            if not num_files:
                return

            file_paths = []
            for _ in range(num_files):
                file_path = filedialog.askopenfilename(
                    title="Select the VSWR/Return Loss File(s)",
                    filetypes=[("CSV files", "*.csv"), ("All files", "*.*")],
                )
                if not file_path:
                    return
                file_paths.append(file_path)

            process_vswr_files(
                file_paths,
                self.saved_limit1_freq1,
                self.saved_limit1_freq2,
                self.saved_limit1_start,
                self.saved_limit1_stop,
                self.saved_limit2_freq1,
                self.saved_limit2_freq2,
                self.saved_limit2_start,
                self.saved_limit2_stop,
            )

        # Scenario 2: Group Delay = True, Min/Max VSWR = False
        elif self.cb_groupdelay_sff_var.get() and not self.min_max_vswr_var.get():
            num_files = askinteger(
                "Input",
                "How many files do you want to import? (e.g. 8 for Theta=0 deg to 315 deg in 45 deg steps)",
            )
            if not num_files:
                return

            file_paths = []
            for _ in range(num_files):
                file_path = filedialog.askopenfilename(
                    title="Select the 2-Port S-Parameter File(s)",
                    filetypes=[
                        ("CSV files", "*.csv"),
                        ("Touchstone", "*.s2p"),
                        ("All files", "*.*"),
                    ],
                )
                if not file_path:
                    return
                file_paths.append(file_path)

            min_freq = askstring(
                "Input", "Enter minimum frequency (GHz) or leave blank for default:"
            )
            max_freq = askstring(
                "Input", "Enter maximum frequency (GHz) or leave blank for default:"
            )

            min_freq = float(min_freq) if min_freq else None
            max_freq = float(max_freq) if max_freq else None

            process_groupdelay_files(
                file_paths,
                self.saved_limit1_freq1,
                self.saved_limit1_freq2,
                self.saved_limit1_start,
                self.saved_limit1_stop,
                self.saved_limit2_freq1,
                self.saved_limit2_freq2,
                self.saved_limit2_start,
                self.saved_limit2_stop,
                min_freq,
                max_freq,
            )

        # Scenario 3: Group Delay = False, Min/Max VSWR = True
        elif not self.cb_groupdelay_sff_var.get() and self.min_max_vswr_var.get():
            num_files = askinteger("Input", "How many VSWR files do you want to import?")
            if not num_files:
                return

            num_bands = askinteger("Input", "How many frequency bands would you like to define?")
            if not num_bands:
                return

            bands = []
            for i in range(num_bands):
                freq_range_input = askstring(
                    "Input",
                    f"Enter frequency range for Band {i+1} as 'min,max' (in MHz):",
                    parent=self.root,
                )
                if not freq_range_input:
                    messagebox.showerror("Error", "No frequency range entered.")
                    return
                try:
                    freq_min, freq_max = map(float, freq_range_input.split(","))
                    bands.append((freq_min, freq_max))
                except ValueError:
                    messagebox.showerror(
                        "Error", "Invalid frequency range. Please enter values in 'min,max' format."
                    )
                    return

            file_paths = []
            for _ in range(num_files):
                file_path = filedialog.askopenfilename(
                    title="Select the VSWR 1-Port File(s)",
                    filetypes=[("CSV files", "*.csv"), ("All files", "*.*")],
                )
                if not file_path:
                    return
                file_paths.append(file_path)

            results = calculate_min_max_parameters(file_paths, bands, "VSWR")
            display_parameter_table(results, "VSWR", parent=self.root)

    # ────────────────────────────────────────────────────────────────────────
    # LOGGING
    # ────────────────────────────────────────────────────────────────────────

    def log_message(self, message, level="info"):
        """Log a message to the GUI log text area with optional color coding.

        Args:
            message: Text to display in the log
            level: One of "info", "success", "warning", "error"
        """
        if not hasattr(self, "log_text"):
            return
        self.log_text.configure(state="normal")
        tag = f"log_{level}"
        self.log_text.insert("end", message + "\n", tag)
        self.log_text.configure(state="disabled")
        self.log_text.see("end")

    # ────────────────────────────────────────────────────────────────────────
    # FREQUENCY EXTRAPOLATION
    # ────────────────────────────────────────────────────────────────────────

    def _run_extrapolation(self, target_frequency):
        """Run frequency extrapolation and display results as a standalone operation.

        Args:
            target_frequency: Target frequency in MHz to extrapolate to.
        """
        if not self.hpol_file_path or not self.vpol_file_path:
            messagebox.showerror("Error", "No passive HPOL/VPOL files loaded.")
            return

        try:
            hpol_data, *_ = read_passive_file(self.hpol_file_path)
            vpol_data, *_ = read_passive_file(self.vpol_file_path)
        except Exception as e:
            messagebox.showerror("Error", f"Failed to read passive files: {e}")
            return

        # Check cache
        cache_key = (target_frequency, self.hpol_file_path, self.vpol_file_path)
        if cache_key in self._extrapolation_cache:
            extrap = self._extrapolation_cache[cache_key]
            self.log_message("Using cached extrapolation results.", level="info")
        else:
            try:
                self.log_message(f"Extrapolating pattern to {target_frequency} MHz...")
                extrap = extrapolate_pattern(hpol_data, vpol_data, target_frequency)
                self._extrapolation_cache[cache_key] = extrap
            except ValueError as e:
                messagebox.showerror("Extrapolation Failed", str(e))
                self.log_message(f"Extrapolation failed: {e}", level="error")
                return

        # Compute summary stats
        h_mag = np.array(extrap["hpol"]["mag"])
        v_mag = np.array(extrap["vpol"]["mag"])
        # Total gain: power sum of H and V in linear
        h_lin = 10 ** (h_mag / 10)
        v_lin = 10 ** (v_mag / 10)
        total_dB = 10 * np.log10(h_lin + v_lin)

        peak_total = float(np.max(total_dB))
        peak_hpol = float(np.max(h_mag))
        peak_vpol = float(np.max(v_mag))
        # Spherical average with sin(theta) weighting
        theta_deg = np.array(extrap["hpol"]["theta"])
        sin_theta = np.sin(np.radians(theta_deg))
        total_lin = 10 ** (total_dB / 10)
        if np.sum(sin_theta) > 0:
            avg_gain = float(10 * np.log10(np.sum(total_lin * sin_theta) / np.sum(sin_theta)))
        else:
            avg_gain = float(np.mean(total_dB))

        confidence = extrap["confidence"]
        quality = confidence["quality"]

        # Log results
        self.log_message(f"─── Extrapolation to {target_frequency} MHz ───", level="info")
        self.log_message(f"  Peak Total Gain: {peak_total:.2f} dBi", level="info")
        self.log_message(f"  Peak HPOL Gain:  {peak_hpol:.2f} dBi", level="info")
        self.log_message(f"  Peak VPOL Gain:  {peak_vpol:.2f} dBi", level="info")
        self.log_message(f"  Avg Total Gain:  {avg_gain:.2f} dBi", level="info")
        self.log_message(
            f"  Confidence: {quality} (R²={confidence['mean_r_squared']:.3f})",
            level="info" if quality in ("high", "moderate") else "warning",
        )
        if confidence.get("warning"):
            self.log_message(f"  Warning: {confidence['warning']}", level="warning")

        # Update measurement context
        self._measurement_context["extrapolation_applied"] = True
        self._measurement_context["extrapolation_confidence"] = confidence

        # Show summary dialog
        messagebox.showinfo(
            f"Extrapolation to {target_frequency} MHz",
            f"Peak Total Gain: {peak_total:.2f} dBi\n"
            f"Peak HPOL Gain: {peak_hpol:.2f} dBi\n"
            f"Peak VPOL Gain: {peak_vpol:.2f} dBi\n"
            f"Avg Total Gain: {avg_gain:.2f} dBi\n\n"
            f"Confidence: {quality}\n"
            f"R²: {confidence['mean_r_squared']:.3f}\n"
            f"Est. Max Error: {confidence['max_estimated_error_dB']:.1f} dB",
        )

    # ────────────────────────────────────────────────────────────────────────
    # DATA PROCESSING
    # ────────────────────────────────────────────────────────────────────────

    def _process_data_without_plotting(self):
        """Process antenna data and store arrays WITHOUT automatically displaying plots.
        Used by AI functions to prepare data for selective plotting."""
        try:
            if self.scan_type.get() == "passive":
                # Read passive files
                (
                    parsed_hpol_data,
                    start_phi_h,
                    stop_phi_h,
                    inc_phi_h,
                    start_theta_h,
                    stop_theta_h,
                    inc_theta_h,
                ) = read_passive_file(self.hpol_file_path)
                hpol_data = parsed_hpol_data

                (
                    parsed_vpol_data,
                    start_phi_v,
                    stop_phi_v,
                    inc_phi_v,
                    start_theta_v,
                    stop_theta_v,
                    inc_theta_v,
                ) = read_passive_file(self.vpol_file_path)
                vpol_data = parsed_vpol_data

                # Check if angle data matches
                if not angles_match(
                    start_phi_h,
                    stop_phi_h,
                    inc_phi_h,
                    start_theta_h,
                    stop_theta_h,
                    inc_theta_h,
                    start_phi_v,
                    stop_phi_v,
                    inc_phi_v,
                    start_theta_v,
                    stop_theta_v,
                    inc_theta_v,
                ):
                    raise Exception("Angle data mismatch between HPol and VPol files")

                # Calculate variables for all frequencies
                passive_variables = calculate_passive_variables(
                    hpol_data,
                    vpol_data,
                    float(self.cable_loss.get()),
                    start_phi_h,
                    stop_phi_h,
                    inc_phi_h,
                    start_theta_h,
                    stop_theta_h,
                    inc_theta_h,
                    self.freq_list,
                    float(self.selected_frequency.get()),
                )
                theta_angles_deg, phi_angles_deg, v_gain_dB, h_gain_dB, Total_Gain_dB = (
                    passive_variables
                )

                # Store as instance variables for AI access
                self.theta_list = theta_angles_deg
                self.phi_list = phi_angles_deg
                self.v_gain_dB = v_gain_dB
                self.h_gain_dB = h_gain_dB
                self.total_gain_dB = Total_Gain_dB

                self.hpol_far_field = h_gain_dB
                self.vpol_far_field = v_gain_dB

                return True

            elif self.scan_type.get() == "active":
                # Process active data without plotting
                data = read_active_file(self.TRP_file_path)

                (
                    frequency,
                    start_phi,
                    start_theta,
                    stop_phi,
                    stop_theta,
                    inc_phi,
                    inc_theta,
                    calc_trp,
                    theta_angles_deg_raw,
                    phi_angles_deg_raw,
                    h_power_dBm,
                    v_power_dBm,
                ) = (
                    data["Frequency"],
                    data["Start Phi"],
                    data["Start Theta"],
                    data["Stop Phi"],
                    data["Stop Theta"],
                    data["Inc Phi"],
                    data["Inc Theta"],
                    data["Calculated TRP(dBm)"],
                    data["Theta_Angles_Deg"],
                    data["Phi_Angles_Deg"],
                    data["H_Power_dBm"],
                    data["V_Power_dBm"],
                )

                # Calculate Variables for TRP/Active Measurement
                active_variables = calculate_active_variables(
                    start_phi,
                    stop_phi,
                    start_theta,
                    stop_theta,
                    inc_phi,
                    inc_theta,
                    h_power_dBm,
                    v_power_dBm,
                )

                (
                    data_points,
                    theta_angles_deg,
                    phi_angles_deg,
                    theta_angles_rad,
                    phi_angles_rad,
                    total_power_dBm_2d,
                    h_power_dBm_2d,
                    v_power_dBm_2d,
                    phi_angles_deg_plot,
                    phi_angles_rad_plot,
                    total_power_dBm_2d_plot,
                    h_power_dBm_2d_plot,
                    v_power_dBm_2d_plot,
                    total_power_dBm_min,
                    total_power_dBm_nom,
                    h_power_dBm_min,
                    h_power_dBm_nom,
                    v_power_dBm_min,
                    v_power_dBm_nom,
                    TRP_dBm,
                    h_TRP_dBm,
                    v_TRP_dBm,
                ) = active_variables

                # Store as instance variables for AI access
                self.data_points = data_points
                self.theta_angles_deg = theta_angles_deg
                self.phi_angles_deg = phi_angles_deg
                self.theta_angles_rad = theta_angles_rad
                self.phi_angles_rad = phi_angles_rad
                self.total_power_dBm_2d = total_power_dBm_2d
                self.h_power_dBm_2d = h_power_dBm_2d
                self.v_power_dBm_2d = v_power_dBm_2d
                self.phi_angles_deg_plot = phi_angles_deg_plot
                self.phi_angles_rad_plot = phi_angles_rad_plot
                self.total_power_dBm_2d_plot = total_power_dBm_2d_plot
                self.TRP_dBm = TRP_dBm
                self.h_TRP_dBm = h_TRP_dBm
                self.v_TRP_dBm = v_TRP_dBm
                self.active_frequency = frequency

                # Set freq_list for active scan (single frequency)
                self.freq_list = [frequency]

                return True

            return False
        except Exception as e:
            print(f"[AI Error] Data processing failed: {str(e)}")
            return False

    def process_data(self):
        """Main data processing method - processes and displays all plots.

        Uses deferred execution so the UI can show progress feedback before
        the (synchronous) processing begins. Matplotlib requires main-thread
        execution, so we schedule via root.after() instead of threading.
        """
        if self._processing_lock:
            return
        self._processing_lock = True
        self._set_buttons_busy(True)
        self.update_status("Processing...")
        self.progress_bar.pack(side=tk.RIGHT, padx=5)
        self.progress_bar.start(15)

        # Defer actual work so the UI updates first
        self.root.after(50, self._do_process)

    def _do_process(self):
        """Execute data processing (called from main thread after UI update)."""
        try:
            # Close tracked figures or fall back to close all
            for fig in self._active_figures:
                try:
                    plt.close(fig)
                except Exception:
                    pass
            self._active_figures.clear()

            if self.scan_type.get() == "active":
                self._process_active_data()
            elif self.scan_type.get() == "passive":
                self._process_passive_data()

        except Exception as e:
            self.log_message(f"Error: {e}", level="error")
        finally:
            self._on_processing_done()

    def _set_buttons_busy(self, busy):
        """Disable/enable action buttons during processing."""
        state = tk.DISABLED if busy else tk.NORMAL
        for btn in (
            self.btn_import,
            self.btn_view_results,
            self.btn_save_to_file,
            self.btn_settings,
        ):
            try:
                btn.config(state=state)
            except Exception:
                pass

    def _on_processing_done(self):
        """Re-enable UI after processing completes."""
        self.progress_bar.stop()
        self.progress_bar.pack_forget()
        self._processing_lock = False
        self._set_buttons_busy(False)
        self.update_status("Ready")
        self.update_visibility()

    def _process_active_data(self):
        """Process and display active measurement data."""
        data = read_active_file(self.TRP_file_path)

        self.log_message("Processing active data...")
        (
            frequency,
            start_phi,
            start_theta,
            stop_phi,
            stop_theta,
            inc_phi,
            inc_theta,
            calc_trp,
            theta_angles_deg,
            phi_angles_deg,
            h_power_dBm,
            v_power_dBm,
        ) = (
            data["Frequency"],
            data["Start Phi"],
            data["Start Theta"],
            data["Stop Phi"],
            data["Stop Theta"],
            data["Inc Phi"],
            data["Inc Theta"],
            data["Calculated TRP(dBm)"],
            data["Theta_Angles_Deg"],
            data["Phi_Angles_Deg"],
            data["H_Power_dBm"],
            data["V_Power_dBm"],
        )

        # Calculate Variables for TRP/Active Measurement Plotting
        active_variables = calculate_active_variables(
            start_phi,
            stop_phi,
            start_theta,
            stop_theta,
            inc_phi,
            inc_theta,
            h_power_dBm,
            v_power_dBm,
        )

        (
            data_points,
            theta_angles_deg,
            phi_angles_deg,
            theta_angles_rad,
            phi_angles_rad,
            total_power_dBm_2d,
            h_power_dBm_2d,
            v_power_dBm_2d,
            phi_angles_deg_plot,
            phi_angles_rad_plot,
            total_power_dBm_2d_plot,
            h_power_dBm_2d_plot,
            v_power_dBm_2d_plot,
            total_power_dBm_min,
            total_power_dBm_nom,
            h_power_dBm_min,
            h_power_dBm_nom,
            v_power_dBm_min,
            v_power_dBm_nom,
            TRP_dBm,
            h_TRP_dBm,
            v_TRP_dBm,
        ) = active_variables

        # Plot 2D data
        plot_active_2d_data(
            data_points, theta_angles_rad, phi_angles_rad_plot, total_power_dBm_2d_plot, frequency
        )

        # Plot 3D data - Total power
        plot_active_3d_data(
            theta_angles_deg,
            phi_angles_deg,
            total_power_dBm_2d,
            phi_angles_deg_plot,
            total_power_dBm_2d_plot,
            frequency,
            power_type="total",
            interpolate=self.interpolate_3d_plots,
            axis_mode=self.axis_scale_mode.get(),
            zmin=self.axis_min.get(),
            zmax=self.axis_max.get(),
        )

        # H-pol
        plot_active_3d_data(
            theta_angles_deg,
            phi_angles_deg,
            h_power_dBm_2d,
            phi_angles_deg_plot,
            h_power_dBm_2d_plot,
            frequency,
            power_type="hpol",
            interpolate=self.interpolate_3d_plots,
            axis_mode=self.axis_scale_mode.get(),
            zmin=self.axis_min.get(),
            zmax=self.axis_max.get(),
        )

        # V-pol
        plot_active_3d_data(
            theta_angles_deg,
            phi_angles_deg,
            v_power_dBm_2d,
            phi_angles_deg_plot,
            v_power_dBm_2d_plot,
            frequency,
            power_type="vpol",
            interpolate=self.interpolate_3d_plots,
            axis_mode=self.axis_scale_mode.get(),
            zmin=self.axis_min.get(),
            zmax=self.axis_max.get(),
        )

        self.log_message("Active data processed successfully.", level="success")

        # Maritime / Horizon plots (active)
        if self.maritime_plots_enabled:
            self.log_message("Generating maritime plots (active)...")
            generate_maritime_plots(
                np.rad2deg(theta_angles_rad),
                np.rad2deg(phi_angles_rad),
                total_power_dBm_2d,
                frequency,
                data_label="Power",
                data_unit="dBm",
                theta_min=self.horizon_theta_min.get(),
                theta_max=self.horizon_theta_max.get(),
                theta_cuts=self._parse_theta_cuts(),
                gain_threshold=self.horizon_gain_threshold.get(),
                axis_mode=self.axis_scale_mode.get(),
                zmin=self.axis_min.get(),
                zmax=self.axis_max.get(),
                save_path=None,
            )

        # Advanced analysis plots (active)
        _any_advanced = (
            self.link_budget_enabled
            or self.indoor_analysis_enabled
            or self.fading_analysis_enabled
            or self.wearable_analysis_enabled
        )
        if _any_advanced:
            self.log_message("Generating advanced analysis plots (active)...")
            generate_advanced_analysis_plots(
                np.rad2deg(theta_angles_rad),
                np.rad2deg(phi_angles_rad),
                total_power_dBm_2d,
                frequency,
                data_label="Power",
                data_unit="dBm",
                save_path=None,
                link_budget_enabled=self.link_budget_enabled,
                lb_pt_dbm=self.lb_tx_power.get(),
                lb_pr_dbm=self.lb_rx_sensitivity.get(),
                lb_gr_dbi=self.lb_rx_gain.get(),
                lb_path_loss_exp=self.lb_path_loss_exp.get(),
                lb_misc_loss_db=self.lb_misc_loss.get(),
                lb_target_range_m=self.lb_target_range.get(),
                indoor_enabled=self.indoor_analysis_enabled,
                indoor_environment=self.indoor_environment.get(),
                indoor_path_loss_exp=self.lb_path_loss_exp.get(),
                indoor_n_walls=self.indoor_num_walls.get(),
                indoor_wall_material=self.indoor_wall_material.get(),
                indoor_shadow_fading_db=self.indoor_shadow_fading.get(),
                indoor_max_distance_m=self.indoor_max_distance.get(),
                fading_enabled=self.fading_analysis_enabled,
                fading_pr_sensitivity_dbm=self.lb_rx_sensitivity.get(),
                fading_pt_dbm=self.lb_tx_power.get(),
                fading_target_reliability=self.fading_target_reliability.get(),
                wearable_enabled=self.wearable_analysis_enabled,
                wearable_body_positions=[
                    pos for pos, var in self.wearable_positions_var.items()
                    if var.get()
                ],
                wearable_tx_power_mw=self.wearable_tx_power_mw.get(),
                wearable_num_devices=self.wearable_device_count.get(),
                wearable_room_size=(
                    self.wearable_room_x.get(),
                    self.wearable_room_y.get(),
                    self.wearable_room_z.get(),
                ),
            )

        # Update measurement context for AI awareness
        self._measurement_context["processing_complete"] = True
        self._measurement_context["key_metrics"] = {
            "Peak Power": f"{np.max(total_power_dBm_2d):.1f} dBm",
            "Min Power": f"{np.min(total_power_dBm_2d):.1f} dBm",
            "TRP": f"{TRP_dBm:.1f} dBm",
        }
        self._measurement_context["data_shape"] = f"{data_points} points"

    def _process_passive_data(self):
        """Process and display passive measurement data."""
        self.log_message("Processing passive data...")

        (
            parsed_hpol_data,
            start_phi_h,
            stop_phi_h,
            inc_phi_h,
            start_theta_h,
            stop_theta_h,
            inc_theta_h,
        ) = read_passive_file(self.hpol_file_path)
        hpol_data = parsed_hpol_data

        (
            parsed_vpol_data,
            start_phi_v,
            stop_phi_v,
            inc_phi_v,
            start_theta_v,
            stop_theta_v,
            inc_theta_v,
        ) = read_passive_file(self.vpol_file_path)
        vpol_data = parsed_vpol_data

        # Check if angle data matches
        if not angles_match(
            start_phi_h,
            stop_phi_h,
            inc_phi_h,
            start_theta_h,
            stop_theta_h,
            inc_theta_h,
            start_phi_v,
            stop_phi_v,
            inc_phi_v,
            start_theta_v,
            stop_theta_v,
            inc_theta_v,
        ):
            messagebox.showerror("Error", "Angle data mismatch between HPol and VPol files.")
            self.log_message("Error: Angle data mismatch between HPol and VPol files.")
            return

        # Calculate passive variables
        passive_variables = calculate_passive_variables(
            hpol_data,
            vpol_data,
            float(self.cable_loss.get()),
            start_phi_h,
            stop_phi_h,
            inc_phi_h,
            start_theta_h,
            stop_theta_h,
            inc_theta_h,
            self.freq_list,
            float(self.selected_frequency.get()),
        )
        theta_angles_deg, phi_angles_deg, v_gain_dB, h_gain_dB, Total_Gain_dB = passive_variables

        # Store as instance variables for AI access
        self.theta_list = theta_angles_deg
        self.phi_list = phi_angles_deg
        self.v_gain_dB = v_gain_dB
        self.h_gain_dB = h_gain_dB
        self.total_gain_dB = Total_Gain_dB

        hpol_far_field = h_gain_dB
        vpol_far_field = v_gain_dB
        self.log_message("Passive data processed successfully.", level="success")

        # Update measurement context for AI awareness
        self._measurement_context["processing_complete"] = True
        self._measurement_context["cable_loss_applied"] = float(self.cable_loss.get())
        freq_idx = (
            self.freq_list.index(float(self.selected_frequency.get())) if self.freq_list else 0
        )
        if Total_Gain_dB is not None:
            gain = Total_Gain_dB[:, freq_idx] if Total_Gain_dB.ndim == 2 else Total_Gain_dB
            self._measurement_context["key_metrics"] = {
                "Peak Gain": f"{np.max(gain):.1f} dBi",
                "Min Gain": f"{np.min(gain):.1f} dBi",
                "Avg Gain": f"{np.mean(gain):.1f} dBi",
            }
            self._measurement_context["data_shape"] = (
                f"{Total_Gain_dB.shape[0]} points x {Total_Gain_dB.shape[1]} frequencies"
                if Total_Gain_dB.ndim == 2
                else f"{len(Total_Gain_dB)} points"
            )

        # Apply human shadowing if enabled
        if self.shadowing_enabled:
            idx = np.where(np.array(self.freq_list) == float(self.selected_frequency.get()))[0][0]
            gain_before = Total_Gain_dB[:, idx].copy()

            Total_Gain_dB[:, idx] = apply_directional_human_shadow(
                Total_Gain_dB[:, idx],
                theta_angles_deg[:, idx],
                phi_angles_deg[:, idx],
                float(self.selected_frequency.get()),
                target_axis=self.shadow_direction,
                cone_half_angle_deg=45,
                tissue_thickness_cm=3.5,
            )

            # Efficiency calculation
            theta_rad = np.deg2rad(theta_angles_deg[:, idx])
            gain_before_lin = 10 ** (gain_before / 10)
            gain_after_lin = 10 ** (Total_Gain_dB[:, idx] / 10)
            weight = np.sin(theta_rad)

            avg_before_lin = np.sum(gain_before_lin * weight) / np.sum(weight)
            avg_after_lin = np.sum(gain_after_lin * weight) / np.sum(weight)

            eff_before_dB = 10 * np.log10(avg_before_lin)
            eff_after_dB = 10 * np.log10(avg_after_lin)
            delta_dB = eff_after_dB - eff_before_dB

            self.log_message(f"[Efficiency Shadowing Impact]")
            self.log_message(f"Before Shadowing: {eff_before_dB:.2f} dB")
            self.log_message(f"After Shadowing : {eff_after_dB:.2f} dB")
            self.log_message(f"Delta Efficiency : {delta_dB:+.2f} dB")

        # Plot 2D and 3D passive data
        plot_2d_passive_data(
            theta_angles_deg,
            phi_angles_deg,
            hpol_far_field,
            vpol_far_field,
            Total_Gain_dB,
            self.freq_list,
            float(self.selected_frequency.get()),
            self.datasheet_plots_var.get(),
        )

        # Plot Total Gain in 3D
        plot_passive_3d_component(
            theta_angles_deg,
            phi_angles_deg,
            hpol_far_field,
            vpol_far_field,
            Total_Gain_dB,
            self.freq_list,
            float(self.selected_frequency.get()),
            gain_type="total",
            axis_mode=self.axis_scale_mode.get(),
            zmin=self.axis_min.get(),
            zmax=self.axis_max.get(),
            save_path=None,
            shadowing_enabled=self.shadowing_enabled,
            shadow_direction=self.shadow_direction,
        )

        # Plot H-pol in 3D
        plot_passive_3d_component(
            theta_angles_deg,
            phi_angles_deg,
            hpol_far_field,
            vpol_far_field,
            Total_Gain_dB,
            self.freq_list,
            float(self.selected_frequency.get()),
            gain_type="hpol",
            axis_mode=self.axis_scale_mode.get(),
            zmin=self.axis_min.get(),
            zmax=self.axis_max.get(),
            save_path=None,
        )

        # Plot V-pol in 3D
        plot_passive_3d_component(
            theta_angles_deg,
            phi_angles_deg,
            hpol_far_field,
            vpol_far_field,
            Total_Gain_dB,
            self.freq_list,
            float(self.selected_frequency.get()),
            gain_type="vpol",
            axis_mode=self.axis_scale_mode.get(),
            zmin=self.axis_min.get(),
            zmax=self.axis_max.get(),
            save_path=None,
        )

        # Maritime / Horizon plots (passive)
        if self.maritime_plots_enabled:
            self.log_message("Generating maritime plots (passive)...")
            freq_idx = (
                self.freq_list.index(float(self.selected_frequency.get())) if self.freq_list else 0
            )
            unique_theta, unique_phi, gain_grid = _prepare_gain_grid(
                theta_angles_deg, phi_angles_deg, Total_Gain_dB, freq_idx
            )
            if gain_grid is not None:
                generate_maritime_plots(
                    unique_theta,
                    unique_phi,
                    gain_grid,
                    float(self.selected_frequency.get()),
                    data_label="Gain",
                    data_unit="dBi",
                    theta_min=self.horizon_theta_min.get(),
                    theta_max=self.horizon_theta_max.get(),
                    theta_cuts=self._parse_theta_cuts(),
                    gain_threshold=self.horizon_gain_threshold.get(),
                    axis_mode=self.axis_scale_mode.get(),
                    zmin=self.axis_min.get(),
                    zmax=self.axis_max.get(),
                    save_path=None,
                )
            else:
                self.log_message(
                    "Maritime: Could not reshape gain data to 2D grid.", level="warning"
                )

        # Advanced analysis plots (passive)
        _any_advanced_p = (
            self.link_budget_enabled
            or self.indoor_analysis_enabled
            or self.fading_analysis_enabled
            or self.wearable_analysis_enabled
        )
        if _any_advanced_p:
            self.log_message("Generating advanced analysis plots (passive)...")
            _adv_freq_idx = (
                self.freq_list.index(float(self.selected_frequency.get()))
                if self.freq_list
                else 0
            )
            _adv_theta, _adv_phi, _adv_grid = _prepare_gain_grid(
                theta_angles_deg, phi_angles_deg, Total_Gain_dB, _adv_freq_idx
            )
            if _adv_grid is not None:
                generate_advanced_analysis_plots(
                    _adv_theta,
                    _adv_phi,
                    _adv_grid,
                    float(self.selected_frequency.get()),
                    data_label="Gain",
                    data_unit="dBi",
                    save_path=None,
                    link_budget_enabled=self.link_budget_enabled,
                    lb_pt_dbm=self.lb_tx_power.get(),
                    lb_pr_dbm=self.lb_rx_sensitivity.get(),
                    lb_gr_dbi=self.lb_rx_gain.get(),
                    lb_path_loss_exp=self.lb_path_loss_exp.get(),
                    lb_misc_loss_db=self.lb_misc_loss.get(),
                    lb_target_range_m=self.lb_target_range.get(),
                    indoor_enabled=self.indoor_analysis_enabled,
                    indoor_environment=self.indoor_environment.get(),
                    indoor_path_loss_exp=self.lb_path_loss_exp.get(),
                    indoor_n_walls=self.indoor_num_walls.get(),
                    indoor_wall_material=self.indoor_wall_material.get(),
                    indoor_shadow_fading_db=self.indoor_shadow_fading.get(),
                    indoor_max_distance_m=self.indoor_max_distance.get(),
                    fading_enabled=self.fading_analysis_enabled,
                    fading_pr_sensitivity_dbm=self.lb_rx_sensitivity.get(),
                    fading_pt_dbm=self.lb_tx_power.get(),
                    fading_target_reliability=self.fading_target_reliability.get(),
                    wearable_enabled=self.wearable_analysis_enabled,
                    wearable_body_positions=[
                        pos for pos, var in self.wearable_positions_var.items()
                        if var.get()
                    ],
                    wearable_tx_power_mw=self.wearable_tx_power_mw.get(),
                    wearable_num_devices=self.wearable_device_count.get(),
                    wearable_room_size=(
                        self.wearable_room_x.get(),
                        self.wearable_room_y.get(),
                        self.wearable_room_z.get(),
                    ),
                )
            else:
                self.log_message(
                    "Advanced analysis: Could not reshape gain data to 2D grid.",
                    level="warning",
                )
