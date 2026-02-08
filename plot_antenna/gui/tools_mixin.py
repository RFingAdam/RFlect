"""
ToolsMixin - Tools and utility methods for RFlect GUI

This mixin provides tool-related functionality:
- Report generation (standard and AI-enhanced)
- Bulk passive/active processing
- Polarization analysis (standard and interactive)
- HPOL/VPOL converter
- Active chamber calibration
- Update checking
"""

from __future__ import annotations

import os
import datetime
import webbrowser
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from tkinter.simpledialog import askstring
from typing import TYPE_CHECKING, Optional, List, Any

import numpy as np
import requests

from ..config import (
    ACCENT_BLUE_COLOR,
    DARK_BG_COLOR,
    LIGHT_TEXT_COLOR,
    HOVER_COLOR,
    BUTTON_FONT,
    BTN_PADX,
    BTN_PADY,
    WIDGET_GAP,
    SURFACE_COLOR,
)

from ..file_utils import (
    read_passive_file,
    check_matching_files,
    convert_HpolVpol_files,
    generate_active_cal_file,
    validate_hpol_vpol_files,
    batch_process_passive_scans,
    batch_process_active_scans,
)
from ..calculations import (
    determine_polarization,
    calculate_polarization_parameters,
    export_polarization_data,
)
from ..plotting import plot_polarization_2d, plot_polarization_3d
from ..save import generate_report, RFAnalyzer
from ..ai_analysis import AntennaAnalyzer

if TYPE_CHECKING:
    from .base_protocol import AntennaPlotGUIProtocol


class ToolsMixin:
    """Mixin class providing tools functionality for AntennaPlotGUI."""

    # Type hints for IDE support (satisfied by main class)
    root: tk.Tk
    scan_type: tk.StringVar
    selected_frequency: tk.StringVar
    freq_list: List[float]
    cable_loss: tk.StringVar
    datasheet_plots_var: tk.BooleanVar
    axis_scale_mode: tk.StringVar
    axis_min: tk.DoubleVar
    axis_max: tk.DoubleVar
    interpolate_3d_plots: bool
    hpol_file_path: Optional[str]
    vpol_file_path: Optional[str]
    TRP_file_path: Optional[str]
    vswr_file_path: Optional[str]
    power_measurement: Optional[str]
    BLPA_HORN_GAIN_STD: Optional[str]
    CURRENT_VERSION: str
    btn_import: tk.Button
    btn_view_results: tk.Button
    btn_save_to_file: tk.Button
    btn_settings: tk.Button

    # Method declarations for type checking only (not defined at runtime to avoid MRO conflicts)
    if TYPE_CHECKING:

        def log_message(self, message: str) -> None: ...
        def update_visibility(self) -> None: ...
        def update_passive_frequency_list(self) -> None: ...

    # ────────────────────────────────────────────────────────────────────────
    # IMAGE COLLECTION
    # ────────────────────────────────────────────────────────────────────────

    def collect_image_paths(self, directory):
        """Recursively collect PNG image paths from directory."""
        image_paths = []
        for root, dirs, files in os.walk(directory):
            for file in files:
                if file.endswith(".png"):
                    image_paths.append(os.path.join(root, file))
        return image_paths

    def _build_report_analyzer(self):
        """Build an AntennaAnalyzer from current GUI state for report tables.

        Returns None if no measurement data is loaded.
        """
        scan_type = self.scan_type.get() if hasattr(self, "scan_type") else None
        if not scan_type:
            return None

        data = {}
        if scan_type == "passive":
            theta = getattr(self, "theta_list", None)
            phi = getattr(self, "phi_list", None)
            if theta is None or phi is None:
                return None
            # GUI stores 2D arrays (points x freqs); AntennaAnalyzer expects 1D angles
            if hasattr(theta, "ndim") and theta.ndim == 2:
                theta = theta[:, 0]
            if hasattr(phi, "ndim") and phi.ndim == 2:
                phi = phi[:, 0]
            data["phi"] = phi
            data["theta"] = theta
            data["total_gain"] = getattr(self, "total_gain_dB", None)
            data["h_gain"] = getattr(self, "h_gain_dB", None)
            data["v_gain"] = getattr(self, "v_gain_dB", None)
        elif scan_type == "active":
            data["phi"] = getattr(self, "phi_angles_deg", None)
            data["theta"] = getattr(self, "theta_angles_deg", None)
            data["total_power"] = getattr(self, "total_power_dBm_2d", None)
            if hasattr(self, "TRP_dBm"):
                data["TRP_dBm"] = getattr(self, "TRP_dBm", 0)
        else:
            return None

        frequencies = list(self.freq_list) if hasattr(self, "freq_list") and self.freq_list else []
        if not frequencies:
            return None

        return AntennaAnalyzer(data, scan_type=scan_type, frequencies=frequencies)

    # ────────────────────────────────────────────────────────────────────────
    # REPORT GENERATION
    # ────────────────────────────────────────────────────────────────────────

    def generate_report_from_directory(self):
        """Generate a report with enhanced prompts for project context."""
        directory = filedialog.askdirectory(title="Select Directory Containing Measurement Images")
        if not directory:
            messagebox.showerror("Error", "No directory selected.")
            return

        # Recursively collect image paths
        image_paths = self.collect_image_paths(directory)

        if not image_paths:
            messagebox.showerror("Error", "No images found in the selected directory.")
            return

        # Enhanced input dialog for report metadata
        report_dialog = tk.Toplevel(self.root)
        report_dialog.title("Report Configuration")
        report_dialog.geometry("500x450")
        report_dialog.transient(self.root)
        report_dialog.grab_set()

        # Title
        tk.Label(report_dialog, text="Report Generation", font=("Arial", 14, "bold")).pack(pady=10)

        # Report title
        tk.Label(report_dialog, text="Report Title:", font=("Arial", 10)).pack(anchor="w", padx=20)
        title_var = tk.StringVar(value="Antenna Measurement Report")
        tk.Entry(report_dialog, textvariable=title_var, width=50).pack(padx=20, pady=5)

        # Project name
        tk.Label(report_dialog, text="Project Name:", font=("Arial", 10)).pack(anchor="w", padx=20)
        project_var = tk.StringVar()
        tk.Entry(report_dialog, textvariable=project_var, width=50).pack(padx=20, pady=5)

        # Antenna type
        tk.Label(
            report_dialog, text="Antenna Type (e.g., Patch, Monopole, PIFA):", font=("Arial", 10)
        ).pack(anchor="w", padx=20)
        antenna_var = tk.StringVar()
        tk.Entry(report_dialog, textvariable=antenna_var, width=50).pack(padx=20, pady=5)

        # Frequency range
        tk.Label(
            report_dialog, text="Frequency Range (e.g., 2.4-2.5 GHz):", font=("Arial", 10)
        ).pack(anchor="w", padx=20)
        freq_var = tk.StringVar()
        tk.Entry(report_dialog, textvariable=freq_var, width=50).pack(padx=20, pady=5)

        # Author
        tk.Label(report_dialog, text="Author/Engineer:", font=("Arial", 10)).pack(
            anchor="w", padx=20
        )
        author_var = tk.StringVar()
        tk.Entry(report_dialog, textvariable=author_var, width=50).pack(padx=20, pady=5)

        # Options
        include_summary_var = tk.BooleanVar(value=False)
        tk.Checkbutton(
            report_dialog, text="Include Executive Summary", variable=include_summary_var
        ).pack(anchor="w", padx=20, pady=5)

        result = {}
        result["cancelled"] = True

        def on_generate():
            result["cancelled"] = False
            result["title"] = title_var.get()
            result["project"] = project_var.get()
            result["antenna"] = antenna_var.get()
            result["freq"] = freq_var.get()
            result["author"] = author_var.get()
            result["summary"] = include_summary_var.get()
            report_dialog.destroy()

        def on_cancel():
            report_dialog.destroy()

        # Buttons
        button_frame = tk.Frame(report_dialog)
        button_frame.pack(pady=20)
        tk.Button(
            button_frame,
            text="Generate Report",
            command=on_generate,
            bg="#4CAF50",
            fg="white",
            padx=20,
        ).pack(side="left", padx=10)
        tk.Button(button_frame, text="Cancel", command=on_cancel, padx=20).pack(
            side="left", padx=10
        )

        # Wait for dialog to close
        self.root.wait_window(report_dialog)

        if result["cancelled"]:
            return

        # Build metadata dictionary
        metadata = {
            "project_name": result["project"],
            "antenna_type": result["antenna"],
            "frequency_range": result["freq"],
            "author": result["author"],
            "date": datetime.datetime.now().strftime("%B %d, %Y"),
        }

        # Build project context for AI
        project_context = {"antenna_type": result["antenna"], "frequency_range": result["freq"]}

        # Initialize the RFAnalyzer without AI
        analyzer = RFAnalyzer(use_ai=False, project_context=project_context)

        # Build AntennaAnalyzer for gain/pattern tables (None if no data loaded)
        ant_analyzer = self._build_report_analyzer()

        save_path = filedialog.askdirectory(title="Select Directory to Save Report")
        if save_path:
            generate_report(
                result["title"],
                image_paths,
                save_path,
                analyzer,
                metadata=metadata,
                include_summary=result["summary"],
                antenna_analyzer=ant_analyzer,
            )
            messagebox.showinfo("Success", f"Report '{result['title']}' generated successfully!")
        else:
            messagebox.showerror("Error", "No directory selected to save the report.")

    def generate_ai_report_from_directory(self):
        """Generate an AI-enhanced report with project context."""
        directory = filedialog.askdirectory(title="Select Directory Containing Measurement Images")
        if not directory:
            messagebox.showerror("Error", "No directory selected.")
            return

        # Recursively collect image paths
        image_paths = self.collect_image_paths(directory)

        if not image_paths:
            messagebox.showerror("Error", "No images found in the selected directory.")
            return

        # Enhanced input dialog for AI report metadata
        report_dialog = tk.Toplevel(self.root)
        report_dialog.title("AI Report Configuration")
        report_dialog.geometry("550x550")
        report_dialog.transient(self.root)
        report_dialog.grab_set()

        # Title
        tk.Label(
            report_dialog, text="AI-Enhanced Report Generation", font=("Arial", 14, "bold")
        ).pack(pady=10)
        tk.Label(
            report_dialog,
            text="AI will analyze each measurement image and generate detailed technical insights.",
            font=("Arial", 9),
            wraplength=500,
        ).pack(pady=5)

        # Report title
        tk.Label(report_dialog, text="Report Title:", font=("Arial", 10)).pack(anchor="w", padx=20)
        title_var = tk.StringVar(value="Antenna Measurement Report - AI Analysis")
        tk.Entry(report_dialog, textvariable=title_var, width=60).pack(padx=20, pady=5)

        # Project name
        tk.Label(report_dialog, text="Project Name:", font=("Arial", 10)).pack(anchor="w", padx=20)
        project_var = tk.StringVar()
        tk.Entry(report_dialog, textvariable=project_var, width=60).pack(padx=20, pady=5)

        # Antenna type
        tk.Label(
            report_dialog, text="Antenna Type (helps AI understand context):", font=("Arial", 10)
        ).pack(anchor="w", padx=20)
        antenna_var = tk.StringVar()
        tk.Entry(report_dialog, textvariable=antenna_var, width=60).pack(padx=20, pady=5)

        # Frequency range
        tk.Label(
            report_dialog, text="Frequency Range (e.g., 2.4-2.5 GHz):", font=("Arial", 10)
        ).pack(anchor="w", padx=20)
        freq_var = tk.StringVar()
        tk.Entry(report_dialog, textvariable=freq_var, width=60).pack(padx=20, pady=5)

        # Application/Requirements
        tk.Label(
            report_dialog, text="Application & Requirements (optional):", font=("Arial", 10)
        ).pack(anchor="w", padx=20)
        req_var = tk.StringVar()
        tk.Entry(report_dialog, textvariable=req_var, width=60).pack(padx=20, pady=5)

        # Author
        tk.Label(report_dialog, text="Author/Engineer:", font=("Arial", 10)).pack(
            anchor="w", padx=20
        )
        author_var = tk.StringVar()
        tk.Entry(report_dialog, textvariable=author_var, width=60).pack(padx=20, pady=5)

        # Options
        include_summary_var = tk.BooleanVar(value=True)
        tk.Checkbutton(
            report_dialog,
            text="Include AI-Generated Executive Summary",
            variable=include_summary_var,
        ).pack(anchor="w", padx=20, pady=5)

        result = {}
        result["cancelled"] = True

        def on_generate():
            result["cancelled"] = False
            result["title"] = title_var.get()
            result["project"] = project_var.get()
            result["antenna"] = antenna_var.get()
            result["freq"] = freq_var.get()
            result["requirements"] = req_var.get()
            result["author"] = author_var.get()
            result["summary"] = include_summary_var.get()
            report_dialog.destroy()

        def on_cancel():
            report_dialog.destroy()

        # Buttons
        button_frame = tk.Frame(report_dialog)
        button_frame.pack(pady=15)
        tk.Button(
            button_frame,
            text="Generate AI Report",
            command=on_generate,
            bg="#2196F3",
            fg="white",
            padx=20,
            pady=5,
        ).pack(side="left", padx=10)
        tk.Button(button_frame, text="Cancel", command=on_cancel, padx=20, pady=5).pack(
            side="left", padx=10
        )

        # Wait for dialog to close
        self.root.wait_window(report_dialog)

        if result["cancelled"]:
            return

        # Build metadata dictionary
        metadata = {
            "project_name": result["project"],
            "antenna_type": result["antenna"],
            "frequency_range": result["freq"],
            "author": result["author"],
            "date": datetime.datetime.now().strftime("%B %d, %Y"),
        }

        # Build enhanced project context for AI
        project_context = {
            "antenna_type": result["antenna"],
            "frequency_range": result["freq"],
            "application": "General RF Application",
        }

        if result["requirements"]:
            project_context["requirements"] = result["requirements"]

        # Initialize the RFAnalyzer WITH AI enabled
        analyzer = RFAnalyzer(use_ai=True, project_context=project_context)

        # Build AntennaAnalyzer for gain/pattern tables (None if no data loaded)
        ant_analyzer = self._build_report_analyzer()

        save_path = filedialog.askdirectory(title="Select Directory to Save AI Report")
        if save_path:
            self.log_message("Starting AI-enhanced report generation...")
            self.log_message(f"Processing {len(image_paths)} images with AI analysis...")

            generate_report(
                result["title"],
                image_paths,
                save_path,
                analyzer,
                metadata=metadata,
                include_summary=result["summary"],
                antenna_analyzer=ant_analyzer,
            )

            messagebox.showinfo(
                "Success",
                f"AI-enhanced report '{result['title']}' generated successfully!\n\n"
                f"Each measurement has been analyzed by AI for technical insights.",
            )
        else:
            messagebox.showerror("Error", "No directory selected to save the report.")

    # ────────────────────────────────────────────────────────────────────────
    # BULK PROCESSING
    # ────────────────────────────────────────────────────────────────────────

    def open_bulk_passive_processing(self):
        """Prompt the user for a directory of HPOL/VPOL files and process them in bulk."""
        directory = filedialog.askdirectory(title="Select Folder Containing HPOL/VPOL Files")
        if not directory:
            return

        # Ask the user for selected frequency points
        freq_input = askstring(
            "Input", "Enter frequency points (MHz) to plot, comma-separated (e.g., 2400,2480,4900):"
        )
        if not freq_input:
            return
        try:
            selected_freqs = [float(x.strip()) for x in freq_input.split(",") if x.strip()]
            if not selected_freqs:
                raise ValueError
        except Exception:
            messagebox.showerror(
                "Error", "Invalid frequency list. Please enter comma-separated numbers."
            )
            return

        # Determine the full list of frequencies from the first HPOL file
        try:
            hpol_files = [f for f in os.listdir(directory) if f.endswith("AP_HPol.txt")]
            if not hpol_files:
                raise FileNotFoundError
            # Use the first HPOL file to extract available frequencies
            first_hpol_path = os.path.join(directory, hpol_files[0])
            parsed_data, *_ = read_passive_file(first_hpol_path)
            freq_list = sorted({entry["frequency"] for entry in parsed_data})
        except Exception:
            messagebox.showerror("Error", "Unable to determine frequency list from files.")
            return

        # Ask user where to save the results
        save_dir = filedialog.askdirectory(title="Select Directory to Save Results")
        if not save_dir:
            return
        project_name = askstring("Input", "Enter Project Name for bulk processing:")
        if project_name:
            save_base = os.path.join(save_dir, project_name)
        else:
            save_base = save_dir
        os.makedirs(save_base, exist_ok=True)

        # Retrieve cable loss from the entry
        try:
            cable_loss = (
                float(self.cable_loss.get())
                if isinstance(self.cable_loss, tk.StringVar)
                else float(self.cable_loss)
            )
        except Exception:
            cable_loss = 0.0

        # Determine whether to generate datasheet plots
        datasheet_plots = (
            bool(self.datasheet_plots_var.get()) if hasattr(self, "datasheet_plots_var") else False
        )

        # Axis scaling settings
        axis_mode = self.axis_scale_mode.get() if hasattr(self, "axis_scale_mode") else "auto"
        zmin = float(self.axis_min.get()) if hasattr(self, "axis_min") else -15
        zmax = float(self.axis_max.get()) if hasattr(self, "axis_max") else 15

        # Invoke batch processing with progress feedback
        progress_window = tk.Toplevel(self.root)
        progress_window.title("Processing...")
        progress_window.geometry("300x100")
        progress_window.configure(bg=DARK_BG_COLOR)
        progress_window.transient(self.root)
        tk.Label(progress_window, text="Bulk processing in progress...", bg=DARK_BG_COLOR, fg=LIGHT_TEXT_COLOR).pack(pady=10)
        progress_bar = ttk.Progressbar(progress_window, mode="indeterminate", length=250)
        progress_bar.pack(pady=10)
        progress_bar.start()

        def _process_worker():
            try:
                batch_process_passive_scans(
                    folder_path=directory,
                    freq_list=freq_list,
                    selected_frequencies=selected_freqs,
                    cable_loss=cable_loss,
                    datasheet_plots=datasheet_plots,
                    save_base=save_base,
                    axis_mode=axis_mode,
                    zmin=zmin,
                    zmax=zmax,
                )
                self.root.after(0, lambda: _process_done(True, None))
            except Exception as e:
                err = str(e)
                self.root.after(0, lambda: _process_done(False, err))

        def _process_done(success, error_msg):
            progress_bar.stop()
            progress_window.destroy()
            if success:
                messagebox.showinfo("Success", f"Bulk processing complete. Results saved to {save_base}")
            else:
                messagebox.showerror("Error", f"An error occurred during processing: {error_msg}")

        import threading
        threading.Thread(target=_process_worker, daemon=True).start()

    def open_bulk_active_processing(self):
        """Prompt the user for a directory of TRP files and process them in bulk."""
        directory = filedialog.askdirectory(title="Select Folder Containing TRP Files")
        if not directory:
            return

        # Ask user where to save the results
        save_dir = filedialog.askdirectory(title="Select Directory to Save Results")
        if not save_dir:
            return

        project_name = askstring("Input", "Enter Project Name for bulk active processing:")
        if project_name:
            save_base = os.path.join(save_dir, project_name)
        else:
            save_base = save_dir
        os.makedirs(save_base, exist_ok=True)

        # Get interpolation setting
        interpolate = self.interpolate_3d_plots if hasattr(self, "interpolate_3d_plots") else True

        # Axis scaling settings
        axis_mode = self.axis_scale_mode.get() if hasattr(self, "axis_scale_mode") else "auto"
        zmin = float(self.axis_min.get()) if hasattr(self, "axis_min") else -15.0
        zmax = float(self.axis_max.get()) if hasattr(self, "axis_max") else 15.0

        # Invoke batch processing with progress feedback
        progress_window = tk.Toplevel(self.root)
        progress_window.title("Processing...")
        progress_window.geometry("300x100")
        progress_window.configure(bg=DARK_BG_COLOR)
        progress_window.transient(self.root)
        tk.Label(progress_window, text="Bulk processing in progress...", bg=DARK_BG_COLOR, fg=LIGHT_TEXT_COLOR).pack(pady=10)
        progress_bar = ttk.Progressbar(progress_window, mode="indeterminate", length=250)
        progress_bar.pack(pady=10)
        progress_bar.start()

        def _process_worker():
            try:
                batch_process_active_scans(
                    folder_path=directory,
                    save_base=save_base,
                    interpolate=interpolate,
                    axis_mode=axis_mode,
                    zmin=zmin,
                    zmax=zmax,
                )
                self.root.after(0, lambda: _process_done(True, None))
            except Exception as e:
                err = str(e)
                self.root.after(0, lambda: _process_done(False, err))

        def _process_done(success, error_msg):
            progress_bar.stop()
            progress_window.destroy()
            if success:
                messagebox.showinfo("Success", f"Bulk active processing complete. Results saved to {save_base}")
            else:
                messagebox.showerror("Error", f"An error occurred during processing: {error_msg}")

        import threading
        threading.Thread(target=_process_worker, daemon=True).start()

    # ────────────────────────────────────────────────────────────────────────
    # POLARIZATION ANALYSIS
    # ────────────────────────────────────────────────────────────────────────

    def open_polarization_analysis(self):
        """
        Calculate and export polarization parameters (Axial Ratio, Tilt Angle,
        Sense, XPD) from HPOL/VPOL passive measurement data.
        """
        self.log_message("Starting Polarization Analysis...")

        # Import HPOL file
        hpol_file = filedialog.askopenfilename(
            title="Select HPOL File", filetypes=[("Text files", "*.txt"), ("All files", "*.*")]
        )
        if not hpol_file:
            return

        # Import VPOL file
        vpol_file = filedialog.askopenfilename(
            title="Select VPOL File", filetypes=[("Text files", "*.txt"), ("All files", "*.*")]
        )
        if not vpol_file:
            return

        # Validate HPOL/VPOL file pairing
        self.log_message("Validating HPOL and VPOL file selection...")
        is_valid, error_msg = validate_hpol_vpol_files(hpol_file, vpol_file)
        if not is_valid:
            self.log_message(f"Validation failed: {error_msg}")
            messagebox.showerror("File Validation Error", error_msg)
            return
        self.log_message("[OK] File validation passed - files correctly paired")

        # Get cable loss
        cable_loss_str = askstring("Input", "Enter cable loss (dB):", initialvalue="0.0")
        if cable_loss_str is None:
            return
        try:
            cable_loss = float(cable_loss_str)
        except ValueError:
            messagebox.showerror("Error", "Invalid cable loss value")
            return

        # Select output directory
        output_dir = filedialog.askdirectory(title="Select Output Directory for Results")
        if not output_dir:
            return

        # Ask for output format
        format_choice = messagebox.askyesno(
            "Output Format", "Export as CSV?\n\nYes = CSV format\nNo = TXT format"
        )
        output_format = "csv" if format_choice else "txt"

        try:
            # Parse files
            self.log_message("Parsing HPOL and VPOL files...")
            hpol_data, *_ = read_passive_file(hpol_file)
            vpol_data, *_ = read_passive_file(vpol_file)

            # Calculate polarization parameters
            self.log_message("Calculating polarization parameters...")
            pol_results = calculate_polarization_parameters(hpol_data, vpol_data, cable_loss)

            # Export results
            self.log_message(f"Exporting results as {output_format.upper()} files...")
            export_polarization_data(pol_results, output_dir, format=output_format)

            # Generate summary
            summary_lines = ["Polarization Analysis Summary", "=" * 60, ""]
            for result in pol_results:
                freq = result["frequency"]
                ar_mean = np.mean(result["axial_ratio_dB"])
                ar_min = np.min(result["axial_ratio_dB"])
                ar_max = np.max(result["axial_ratio_dB"])

                tilt_mean = np.mean(result["tilt_angle_deg"])

                # Count LHCP vs RHCP points
                lhcp_count = np.sum(result["sense"] > 0)
                rhcp_count = np.sum(result["sense"] < 0)
                total_points = len(result["sense"])
                lhcp_pct = (lhcp_count / total_points) * 100
                rhcp_pct = (rhcp_count / total_points) * 100

                summary_lines.append(f"Frequency: {freq} MHz")
                summary_lines.append(
                    f"  Axial Ratio: Min={ar_min:.2f} dB, Max={ar_max:.2f} dB, Avg={ar_mean:.2f} dB"
                )
                summary_lines.append(f"  Tilt Angle: Avg={tilt_mean:.2f} deg")
                summary_lines.append(f"  Polarization: LHCP={lhcp_pct:.1f}%, RHCP={rhcp_pct:.1f}%")
                summary_lines.append("")

            summary_text = "\n".join(summary_lines)

            # Save summary
            summary_path = os.path.join(output_dir, "polarization_summary.txt")
            with open(summary_path, "w", encoding="utf-8") as f:
                f.write(summary_text)

            # Show summary to user
            self.log_message("Polarization analysis complete!")
            self.log_message(summary_text)

            messagebox.showinfo(
                "Success",
                f"Polarization analysis complete!\n\n"
                f"Results saved to:\n{output_dir}\n\n"
                f"Files generated:\n"
                f"- polarization_<freq>MHz.{output_format} (one per frequency)\n"
                f"- polarization_summary.txt",
            )

        except Exception as e:
            self.log_message(f"Error during polarization analysis: {e}")
            messagebox.showerror("Error", f"An error occurred: {e}")

    def open_polarization_interactive(self):
        """
        Interactive polarization analysis with frequency selection and live plotting.
        Allows user to visualize polarization parameters (AR, Tilt, Sense, XPD)
        in 2D/3D and optionally export data.
        """
        self.log_message("Starting Interactive Polarization Analysis...")

        # Import HPOL file
        hpol_file = filedialog.askopenfilename(
            title="Select HPOL File", filetypes=[("Text files", "*.txt"), ("All files", "*.*")]
        )
        if not hpol_file:
            return

        # Import VPOL file
        vpol_file = filedialog.askopenfilename(
            title="Select VPOL File", filetypes=[("Text files", "*.txt"), ("All files", "*.*")]
        )
        if not vpol_file:
            return

        # Validate HPOL/VPOL file pairing
        self.log_message("Validating HPOL and VPOL file selection...")
        is_valid, error_msg = validate_hpol_vpol_files(hpol_file, vpol_file)
        if not is_valid:
            self.log_message(f"Validation failed: {error_msg}")
            messagebox.showerror("File Validation Error", error_msg)
            return
        self.log_message("[OK] File validation passed - files correctly paired")

        # Get cable loss
        cable_loss_str = askstring("Input", "Enter cable loss (dB):", initialvalue="0.0")
        if cable_loss_str is None:
            return
        try:
            cable_loss = float(cable_loss_str)
        except ValueError:
            messagebox.showerror("Error", "Invalid cable loss value")
            return

        try:
            # Parse files
            self.log_message("Parsing HPOL and VPOL files...")
            hpol_data, *_ = read_passive_file(hpol_file)
            vpol_data, *_ = read_passive_file(vpol_file)

            # Calculate polarization parameters for all frequencies
            self.log_message("Calculating polarization parameters...")
            pol_results = calculate_polarization_parameters(hpol_data, vpol_data, cable_loss)

            if not pol_results:
                messagebox.showerror("Error", "No polarization results calculated")
                return

            # Create frequency selection dialog
            freq_list = [result["frequency"] for result in pol_results]

            # Show frequency selection dialog
            freq_dialog = tk.Toplevel(self.root)
            freq_dialog.title("Select Frequency and Visualization Options")
            freq_dialog.geometry("450x400")
            freq_dialog.transient(self.root)
            freq_dialog.grab_set()

            # Frequency selection
            tk.Label(freq_dialog, text="Select Frequency (MHz):", font=("Arial", 11, "bold")).pack(
                pady=10
            )
            freq_var = tk.StringVar(value=str(freq_list[0]))
            freq_listbox = tk.Listbox(freq_dialog, height=6, font=("Arial", 10))
            for freq in freq_list:
                freq_listbox.insert(tk.END, f"{freq} MHz")
            freq_listbox.selection_set(0)
            freq_listbox.pack(pady=5)

            # Plot options
            tk.Label(freq_dialog, text="Visualization Options:", font=("Arial", 11, "bold")).pack(
                pady=10
            )
            plot_2d_var = tk.BooleanVar(value=True)
            plot_3d_var = tk.BooleanVar(value=True)

            tk.Checkbutton(
                freq_dialog,
                text="Show 2D Plots (Contour + Polar)",
                variable=plot_2d_var,
                font=("Arial", 10),
            ).pack()
            tk.Checkbutton(
                freq_dialog,
                text="Show 3D Plots (Spherical)",
                variable=plot_3d_var,
                font=("Arial", 10),
            ).pack()

            # Export option
            tk.Label(freq_dialog, text="Export Options:", font=("Arial", 11, "bold")).pack(pady=10)
            export_var = tk.BooleanVar(value=False)
            tk.Checkbutton(
                freq_dialog, text="Export data to file", variable=export_var, font=("Arial", 10)
            ).pack()

            # Results container
            result_container = {"cancelled": True}

            def on_ok():
                result_container["cancelled"] = False
                selection = freq_listbox.curselection()
                if selection:
                    result_container["freq_idx"] = selection[0]
                    result_container["plot_2d"] = plot_2d_var.get()
                    result_container["plot_3d"] = plot_3d_var.get()
                    result_container["export"] = export_var.get()
                freq_dialog.destroy()

            def on_cancel():
                result_container["cancelled"] = True
                freq_dialog.destroy()

            # Buttons
            button_frame = tk.Frame(freq_dialog)
            button_frame.pack(pady=20)
            tk.Button(
                button_frame, text="OK", command=on_ok, width=10, font=("Arial", 10, "bold")
            ).pack(side=tk.LEFT, padx=5)
            tk.Button(
                button_frame, text="Cancel", command=on_cancel, width=10, font=("Arial", 10)
            ).pack(side=tk.LEFT, padx=5)

            # Wait for dialog to close
            self.root.wait_window(freq_dialog)

            if result_container["cancelled"]:
                return

            # Get selected frequency data
            freq_idx = result_container["freq_idx"]
            selected_result = pol_results[freq_idx]
            frequency = selected_result["frequency"]

            self.log_message(f"Visualizing polarization data for {frequency} MHz...")

            # Extract data (use correct keys from calculate_polarization_parameters)
            theta_deg = selected_result["theta"]
            phi_deg = selected_result["phi"]
            ar_db = selected_result["axial_ratio_dB"]
            tilt_deg = selected_result["tilt_angle_deg"]
            sense = selected_result["sense"]
            xpd_db = selected_result["cross_pol_discrimination_dB"]

            # Get unique theta and phi values for plotting
            unique_theta = np.unique(theta_deg)
            unique_phi = np.unique(phi_deg)

            # Reshape 1D arrays to 2D for plotting
            n_theta = len(unique_theta)
            n_phi = len(unique_phi)
            ar_db_2d = ar_db.reshape(n_theta, n_phi)
            tilt_deg_2d = tilt_deg.reshape(n_theta, n_phi)
            sense_2d = sense.reshape(n_theta, n_phi)
            xpd_db_2d = xpd_db.reshape(n_theta, n_phi)

            # Generate plots
            if result_container["plot_2d"]:
                self.log_message("Generating 2D polarization plots...")
                plot_polarization_2d(
                    unique_theta, unique_phi, ar_db_2d, tilt_deg_2d, sense_2d, xpd_db_2d, frequency
                )

            if result_container["plot_3d"]:
                self.log_message("Generating 3D polarization plots...")
                plot_polarization_3d(
                    unique_theta, unique_phi, ar_db_2d, tilt_deg_2d, sense_2d, frequency
                )

            # Export if requested
            if result_container["export"]:
                output_dir = filedialog.askdirectory(title="Select Output Directory for Export")
                if output_dir:
                    format_choice = messagebox.askyesno(
                        "Output Format", "Export as CSV?\n\nYes = CSV format\nNo = TXT format"
                    )
                    output_format = "csv" if format_choice else "txt"

                    self.log_message(
                        f"Exporting {frequency} MHz data as {output_format.upper()}..."
                    )
                    export_polarization_data([selected_result], output_dir, format=output_format)

                    messagebox.showinfo(
                        "Export Complete",
                        f"Polarization data exported to:\n{output_dir}\n\n"
                        f"File: polarization_{frequency}MHz.{output_format}",
                    )

            self.log_message("Interactive polarization analysis complete!")

        except Exception as e:
            self.log_message(f"Error during interactive polarization analysis: {e}")
            messagebox.showerror("Error", f"An error occurred: {e}")

    # ────────────────────────────────────────────────────────────────────────
    # FILE CONVERTERS AND CALIBRATION
    # ────────────────────────────────────────────────────────────────────────

    def save_results_to_file(self):
        """Save measurement results to file."""
        from ..save import save_to_results_folder

        try:
            scan_type = self.scan_type.get()

            if scan_type == "active":
                # For active measurements, no need to get the frequency from user input
                frequency = None
                cable_loss = None
            else:
                # For passive measurements, ensure a frequency is selected
                frequency = self.selected_frequency.get()
                cable_loss = float(self.cable_loss.get())
                if not frequency:
                    messagebox.showerror("Error", "Please select a frequency before saving.")
                    return
                frequency = float(frequency)

            # Now proceed with saving results
            save_to_results_folder(
                frequency,
                self.freq_list,
                scan_type,
                self.hpol_file_path,
                self.vpol_file_path,
                self.TRP_file_path,
                cable_loss,
                self.datasheet_plots_var.get(),
                self.axis_scale_mode.get(),
                self.axis_min.get(),
                self.axis_max.get(),
                word=False,
            )

        except ValueError as ve:
            messagebox.showerror(
                "Conversion Error", f"Invalid frequency or cable loss value. Error: {ve}"
            )

    def open_hpol_vpol_converter(self):
        """Start HPOL and VPOL gain text file conversion to CST FFS format."""
        self.log_message("HPOL/VPOL to CST FFS Converter started...")
        first_file = filedialog.askopenfilename(
            title="Select the First Data File",
            filetypes=[("Text files", "*.txt"), ("All files", "*.*")],
        )
        if not first_file:
            self.log_message("File selection canceled.")
            return

        second_file = filedialog.askopenfilename(
            title="Select the Second Data File",
            filetypes=[("Text files", "*.txt"), ("All files", "*.*")],
        )

        self.vswr_file_path = filedialog.askopenfilename(
            title="Select the VSWR Data File for Accepted P(W,rms) calculations"
        )

        if first_file and second_file:
            # Determine file polarizations
            first_polarization = determine_polarization(first_file)
            second_polarization = determine_polarization(second_file)

            # Check if both files have the same polarization
            if first_polarization == second_polarization:
                self.log_message("Error: Both files cannot be of the same polarization.")
                return

            if first_polarization == "HPol":
                self.hpol_file_path = first_file
                self.vpol_file_path = second_file
            else:
                self.hpol_file_path = second_file
                self.vpol_file_path = first_file

            # Check if File names match and data is consistent between files
            match, message = check_matching_files(self.hpol_file_path, self.vpol_file_path)
            if not match:
                self.log_message(f"Error: {message}")
                return
            self.update_passive_frequency_list()

            # Hide buttons not related to this routine
            self.btn_view_results.pack_forget()
            self.btn_save_to_file.pack_forget()
            self.btn_settings.pack_forget()
            self.btn_import.pack_forget()
            self.log_message("CST .ffs file created successfully.")

            # Create the new button if it doesn't exist, or just show it if it does
            if not hasattr(self, "convert_files_button"):
                self.convert_files_button = tk.Button(
                    self.actions_frame,
                    text="Convert Files",
                    command=lambda: convert_HpolVpol_files(
                        self.vswr_file_path,
                        self.hpol_file_path,
                        self.vpol_file_path,
                        float(self.cable_loss.get()),
                        self.freq_list,
                        float(self.selected_frequency.get()),
                        callback=self.update_visibility,
                    ),
                    bg=ACCENT_BLUE_COLOR,
                    fg=LIGHT_TEXT_COLOR,
                    font=BUTTON_FONT,
                    relief="flat",
                    bd=0,
                    padx=BTN_PADX,
                    pady=BTN_PADY,
                    cursor="hand2",
                )
                self.convert_files_button.pack(side=tk.LEFT, padx=WIDGET_GAP)
            else:
                self.convert_files_button.pack(side=tk.LEFT, padx=WIDGET_GAP)

    def open_active_chamber_cal(self):
        """Start Active Chamber Calibration routine."""
        self.log_message("Active Chamber Calibration routine started....")

        # Implement the logic of the calibration file import
        power_measurement = filedialog.askopenfilename(
            title="Select the Signal Generator/Power Meter Measurement File",
            filetypes=[("Text files", "*.txt"), ("All files", "*.*")],
        )
        if not power_measurement:
            self.log_message("File selection canceled.")
            return
        self.power_measurement = power_measurement

        BLPA_HORN_GAIN_STD = filedialog.askopenfilename(
            title="Select the BLPA or Horn Antenna Gain Standard File",
            filetypes=[("Text files", "*.txt"), ("All files", "*.*")],
        )
        if not BLPA_HORN_GAIN_STD:
            self.log_message("File selection canceled.")
            return

        self.BLPA_HORN_GAIN_STD = BLPA_HORN_GAIN_STD

        first_file = filedialog.askopenfilename(
            title="Select the BLPA or Horn Antenna HPOL or VPOL File",
            filetypes=[("Text files", "*.txt"), ("All files", "*.*")],
        )
        if not first_file:
            self.log_message("File selection canceled.")
            return

        second_file = filedialog.askopenfilename(
            title="Select the other HPOL or VPOL File",
            filetypes=[("Text files", "*.txt"), ("All files", "*.*")],
        )

        if first_file and second_file:
            # Determine file polarizations
            first_polarization = determine_polarization(first_file)
            second_polarization = determine_polarization(second_file)

            # Check if both files have the same polarization
            if first_polarization == second_polarization:
                self.log_message("Error: Both files cannot be of the same polarization.")
                return

            if first_polarization == "HPol":
                self.hpol_file_path = first_file
                self.vpol_file_path = second_file
            else:
                self.hpol_file_path = second_file
                self.vpol_file_path = first_file

            # Check if File names match and data is consistent between files
            match, message = check_matching_files(self.hpol_file_path, self.vpol_file_path)
            if not match:
                self.log_message(f"Error: {message}")
                return
            self.update_passive_frequency_list()

            # Hide buttons not related to this routine
            self.btn_view_results.pack_forget()
            self.btn_save_to_file.pack_forget()
            self.btn_settings.pack_forget()
            self.btn_import.pack_forget()

            self.log_message("Active Chamber Calibration File Created Successfully.")

            # Create the new button if it doesn't exist, or just show it if it does
            if not hasattr(self, "convert_files_button"):
                self.convert_files_button = tk.Button(
                    self.actions_frame,
                    text="Generate Calibration File",
                    command=lambda: generate_active_cal_file(
                        self.power_measurement,
                        self.BLPA_HORN_GAIN_STD,
                        self.hpol_file_path,
                        self.vpol_file_path,
                        float(self.cable_loss.get()),
                        self.freq_list,
                        callback=self.update_visibility,
                    ),
                    bg=ACCENT_BLUE_COLOR,
                    fg=LIGHT_TEXT_COLOR,
                    font=BUTTON_FONT,
                    relief="flat",
                    bd=0,
                    padx=BTN_PADX,
                    pady=BTN_PADY,
                    cursor="hand2",
                )
                self.convert_files_button.pack(side=tk.LEFT, padx=WIDGET_GAP)
            else:
                self.convert_files_button.pack(side=tk.LEFT, padx=WIDGET_GAP)

    # ────────────────────────────────────────────────────────────────────────
    # UPDATE CHECKING
    # ────────────────────────────────────────────────────────────────────────

    def get_latest_release(self):
        """Get the latest release from GitHub."""
        owner = "RFingAdam"
        repo = "RFlect"
        url = f"https://api.github.com/repos/{owner}/{repo}/releases/latest"

        response = requests.get(url, timeout=5)

        if response.status_code == 200:
            data = response.json()
            latest_version = data["tag_name"]
            release_url = data["html_url"]
            return latest_version, release_url
        else:
            return None, None

    def download_latest_release(self, url):
        """Open the given URL in the default web browser to download the release."""
        webbrowser.open(url)

    def check_for_updates(self):
        """Check for software updates."""

        def _parse_version(v):
            """Parse version string like 'v4.2.0' into comparable tuple."""
            return tuple(int(x) for x in v.lstrip("v").split("."))

        latest_version, release_url = self.get_latest_release()
        if latest_version and _parse_version(latest_version) > _parse_version(self.CURRENT_VERSION):
            self.log_message(f"Update Available. A new version {latest_version} is available!")

            answer = messagebox.askyesno(
                "Update Available",
                f"A new version {latest_version} is available! Would you like to download it?",
            )

            if answer:
                self.download_latest_release(release_url)

    # ────────────────────────────────────────────────────────────────────────
    # HOVER EFFECTS
    # ────────────────────────────────────────────────────────────────────────

    def on_enter(self, e):
        """Mouse hover enter effect — brighten button."""
        if str(e.widget["state"]) == "disabled":
            return
        e.widget._original_bg = e.widget["background"]
        bg = e.widget["background"]
        if bg == ACCENT_BLUE_COLOR:
            e.widget["background"] = "#5AA0F2"
        elif bg == SURFACE_COLOR:
            e.widget["background"] = HOVER_COLOR
        else:
            e.widget["background"] = HOVER_COLOR

    def on_leave(self, e):
        """Mouse hover leave effect — restore original."""
        if hasattr(e.widget, "_original_bg"):
            e.widget["background"] = e.widget._original_bg
