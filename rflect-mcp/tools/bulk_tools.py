"""
Bulk Processing Tools for RFlect MCP Server

Exposes existing batch processing functions from plot_antenna.file_utils
as headless MCP tools for VNA automation workflows.
"""

import os
from typing import Optional, List

from .import_tools import get_loaded_measurements


def register_bulk_tools(mcp):
    """Register bulk processing tools with the MCP server."""

    @mcp.tool()
    def bulk_process_passive(
        folder_path: str,
        frequencies: Optional[List[float]] = None,
        cable_loss: float = 0.0,
        save_path: Optional[str] = None,
        datasheet_plots: bool = False,
    ) -> str:
        """
        Batch process all HPOL/VPOL passive measurement pairs in a folder.

        Scans for files ending in AP_HPol.txt and AP_VPol.txt, matches pairs
        by filename prefix, and processes each pair at the specified frequencies.

        Args:
            folder_path: Directory containing AP_HPol.txt and AP_VPol.txt files
            frequencies: List of frequencies in MHz to process (None = all found)
            cable_loss: Cable loss in dB to apply (default: 0.0)
            save_path: Directory for output plots (None = same as input folder)
            datasheet_plots: Generate datasheet-style plots (default: False)

        Returns:
            Summary of processed files and any errors.
        """
        if not os.path.isdir(folder_path):
            return f"Error: '{folder_path}' is not a valid directory."

        try:
            from plot_antenna.file_utils import read_passive_file
            from plot_antenna.calculations import extract_passive_frequencies

            # Find HPOL/VPOL pairs
            files = os.listdir(folder_path)
            hpol_files = [f for f in files if f.endswith("AP_HPol.txt")]
            vpol_files = [f for f in files if f.endswith("AP_VPol.txt")]

            if not hpol_files:
                return f"No HPOL files (*AP_HPol.txt) found in {folder_path}."

            # Extract available frequencies from first HPOL file
            first_hpol = os.path.join(folder_path, hpol_files[0])
            freq_list = extract_passive_frequencies(first_hpol)

            if not freq_list:
                return "Could not extract frequencies from measurement files."

            # If no frequencies specified, use all
            selected_frequencies = frequencies if frequencies else freq_list

            # Validate requested frequencies exist
            invalid = [f for f in selected_frequencies if f not in freq_list]
            if invalid:
                return (
                    f"Frequencies not found in data: {invalid} MHz\n"
                    f"Available: {freq_list} MHz"
                )

            output_dir = save_path or folder_path

            from plot_antenna.file_utils import batch_process_passive_scans

            batch_process_passive_scans(
                folder_path=folder_path,
                freq_list=freq_list,
                selected_frequencies=selected_frequencies,
                cable_loss=cable_loss,
                datasheet_plots=datasheet_plots,
                save_base=output_dir,
            )

            # Build summary
            pairs_found = min(len(hpol_files), len(vpol_files))
            summary = f"Bulk passive processing complete.\n\n"
            summary += f"Folder: {folder_path}\n"
            summary += f"Pairs found: {pairs_found}\n"
            summary += f"Frequencies processed: {selected_frequencies} MHz\n"
            summary += f"Cable loss: {cable_loss} dB\n"
            summary += f"Output: {output_dir}\n"

            return summary

        except Exception as e:
            return f"Error during bulk passive processing: {str(e)}"

    @mcp.tool()
    def bulk_process_active(
        folder_path: str,
        save_path: Optional[str] = None,
        interpolate: bool = True,
    ) -> str:
        """
        Batch process all TRP (Total Radiated Power) active measurement files in a folder.

        Scans for .txt files containing "TRP" in the filename and processes each one.

        Args:
            folder_path: Directory containing TRP measurement files
            save_path: Directory for output plots (None = same as input folder)
            interpolate: Interpolate 3D plots for smoother visualization (default: True)

        Returns:
            Summary of processed files and any errors.
        """
        if not os.path.isdir(folder_path):
            return f"Error: '{folder_path}' is not a valid directory."

        try:
            # Check for TRP files
            files = os.listdir(folder_path)
            trp_files = [f for f in files if f.endswith(".txt") and "TRP" in f.upper()]

            if not trp_files:
                return f"No TRP files found in {folder_path}."

            output_dir = save_path or folder_path

            from plot_antenna.file_utils import batch_process_active_scans

            batch_process_active_scans(
                folder_path=folder_path,
                save_base=output_dir,
                interpolate=interpolate,
            )

            summary = f"Bulk active processing complete.\n\n"
            summary += f"Folder: {folder_path}\n"
            summary += f"TRP files found: {len(trp_files)}\n"
            for f in trp_files:
                summary += f"  - {f}\n"
            summary += f"Interpolation: {'Yes' if interpolate else 'No'}\n"
            summary += f"Output: {output_dir}\n"

            return summary

        except Exception as e:
            return f"Error during bulk active processing: {str(e)}"

    @mcp.tool()
    def validate_file_pair(hpol_path: str, vpol_path: str) -> str:
        """
        Validate that HPOL and VPOL measurement files are correctly paired.

        Checks that files are from the same measurement set, have matching
        frequency/angle data, and are correctly identified as HPOL vs VPOL.

        Args:
            hpol_path: Path to the HPOL measurement file
            vpol_path: Path to the VPOL measurement file

        Returns:
            Validation result: pass/fail with details.
        """
        if not os.path.isfile(hpol_path):
            return f"Error: HPOL file not found: {hpol_path}"
        if not os.path.isfile(vpol_path):
            return f"Error: VPOL file not found: {vpol_path}"

        try:
            from plot_antenna.file_utils import validate_hpol_vpol_files

            is_valid, error_message = validate_hpol_vpol_files(hpol_path, vpol_path)

            if is_valid:
                return (
                    f"VALIDATION PASSED\n\n"
                    f"HPOL: {os.path.basename(hpol_path)}\n"
                    f"VPOL: {os.path.basename(vpol_path)}\n"
                    f"Files are correctly paired and matched."
                )
            else:
                return (
                    f"VALIDATION FAILED\n\n"
                    f"HPOL: {os.path.basename(hpol_path)}\n"
                    f"VPOL: {os.path.basename(vpol_path)}\n\n"
                    f"Issue: {error_message}"
                )

        except Exception as e:
            return f"Error during validation: {str(e)}"

    @mcp.tool()
    def convert_to_cst(
        hpol_path: str,
        vpol_path: str,
        vswr_path: str,
        frequency: float,
        cable_loss: float = 0.0,
        output_path: Optional[str] = None,
    ) -> str:
        """
        Convert HPOL/VPOL measurement files to CST Farfield Source (.ffs) format.

        Combines HPOL, VPOL, and VSWR data to generate a CST-compatible
        farfield source file at the specified frequency.

        Args:
            hpol_path: Path to HPOL passive measurement file
            vpol_path: Path to VPOL passive measurement file
            vswr_path: Path to VSWR measurement CSV file
            frequency: Frequency in MHz to export
            cable_loss: Cable loss in dB (default: 0.0)
            output_path: Custom output directory (None = same as input)

        Returns:
            Path to generated .ffs file and summary.
        """
        for path, label in [(hpol_path, "HPOL"), (vpol_path, "VPOL"), (vswr_path, "VSWR")]:
            if not os.path.isfile(path):
                return f"Error: {label} file not found: {path}"

        try:
            from plot_antenna.calculations import extract_passive_frequencies
            from plot_antenna.file_utils import convert_HpolVpol_files

            # Get available frequencies
            freq_list = extract_passive_frequencies(hpol_path)

            if not freq_list:
                return "Could not extract frequencies from HPOL file."

            if frequency not in freq_list:
                return (
                    f"Frequency {frequency} MHz not found in data.\n"
                    f"Available: {freq_list} MHz"
                )

            convert_HpolVpol_files(
                vswr_file_path=vswr_path,
                hpol_file_path=hpol_path,
                vpol_file_path=vpol_path,
                cable_loss=cable_loss,
                freq_list_MHz=freq_list,
                selected_freq=frequency,
            )

            # The output file is created in the same directory as input
            base_name = os.path.splitext(os.path.basename(hpol_path))[0]
            base_name = base_name.replace("AP_HPol", "").rstrip("_")
            output_file = os.path.join(
                os.path.dirname(hpol_path),
                f"{base_name}_{int(frequency)}MHz.ffs"
            )

            summary = f"CST conversion complete.\n\n"
            summary += f"Input files:\n"
            summary += f"  HPOL: {os.path.basename(hpol_path)}\n"
            summary += f"  VPOL: {os.path.basename(vpol_path)}\n"
            summary += f"  VSWR: {os.path.basename(vswr_path)}\n"
            summary += f"Frequency: {frequency} MHz\n"
            summary += f"Cable loss: {cable_loss} dB\n"
            summary += f"Output: {output_file}\n"

            return summary

        except Exception as e:
            return f"Error during CST conversion: {str(e)}"

    @mcp.tool()
    def list_measurement_files(folder_path: str) -> str:
        """
        Scan a folder and list all recognized antenna measurement files.

        Identifies HPOL/VPOL pairs, TRP files, and VSWR files.

        Args:
            folder_path: Directory to scan

        Returns:
            Categorized list of measurement files found.
        """
        if not os.path.isdir(folder_path):
            return f"Error: '{folder_path}' is not a valid directory."

        files = os.listdir(folder_path)
        txt_files = [f for f in files if f.endswith(".txt")]
        csv_files = [f for f in files if f.endswith(".csv")]

        hpol = [f for f in txt_files if "AP_HPol" in f]
        vpol = [f for f in txt_files if "AP_VPol" in f]
        trp = [f for f in txt_files if "TRP" in f.upper()]
        vswr = [f for f in csv_files if "VSWR" in f.upper() or "SWR" in f.upper()]
        other = [f for f in txt_files if f not in hpol + vpol + trp]

        output = f"Measurement files in: {folder_path}\n"
        output += "=" * 50 + "\n\n"

        output += f"PASSIVE PAIRS ({len(hpol)} HPOL, {len(vpol)} VPOL):\n"
        for f in sorted(hpol):
            output += f"  [HPOL] {f}\n"
        for f in sorted(vpol):
            output += f"  [VPOL] {f}\n"

        if trp:
            output += f"\nACTIVE/TRP FILES ({len(trp)}):\n"
            for f in sorted(trp):
                output += f"  [TRP]  {f}\n"

        if vswr:
            output += f"\nVSWR FILES ({len(vswr)}):\n"
            for f in sorted(vswr):
                output += f"  [VSWR] {f}\n"

        if other:
            output += f"\nOTHER .txt FILES ({len(other)}):\n"
            for f in sorted(other):
                output += f"  {f}\n"

        # Check pairing
        if hpol and vpol:
            paired = min(len(hpol), len(vpol))
            output += f"\nPAIRING STATUS: {paired} pair(s) detected"
            if len(hpol) != len(vpol):
                output += f" (WARNING: {abs(len(hpol) - len(vpol))} unmatched files)"
            output += "\n"

        return output
