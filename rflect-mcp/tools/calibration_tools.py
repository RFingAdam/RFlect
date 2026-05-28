"""
Active chamber calibration MCP tool for RFlect.

Scriptable wrapper over plot_antenna.file_utils.generate_active_cal_file so the
chamber active-calibration workflow can be driven by an MCP agent. The
underlying routine also records the run into the cal-drift history, so this
pairs with the cal_drift_* tools. No LLM, no network. Returns a structured
dict and never raises; failures populate `warnings`.
"""

from __future__ import annotations

import os
from typing import Any, Dict, List

from plot_antenna.file_utils import generate_active_cal_file


def register_calibration_tools(mcp):
    """Register active-calibration tools with the MCP server."""

    @mcp.tool()
    def generate_active_cal(
        power_measurement_file: str,
        gain_standard_file: str,
        hpol_file: str,
        vpol_file: str,
        freq_list: List[float],
        cable_loss: float = 0.0,
    ) -> Dict[str, Any]:
        """
        Generate an active chamber calibration file from reference-antenna data.

        Computes the per-frequency path-loss calibration (P + G − HPol and
        P + G − VPol) from a power-measurement file, a gain-standard file, and the
        reference antenna's HPol/VPol scans, writing a TRP cal file + summary. The
        run is automatically recorded into the cal-drift history (see the
        cal_drift_* tools to compare against prior calibrations).

        Note: the gain-standard reference is rotated 90 deg between polarizations,
        so HPol/VPol angle headers legitimately differ — this routine accounts for
        that (it does not require matching angle grids).

        Args:
            power_measurement_file: Path to the power-measurement file.
            gain_standard_file: Path to the gain-standard (reference antenna) file.
            hpol_file: Path to the reference HPol data file.
            vpol_file: Path to the reference VPol data file.
            freq_list: Frequencies (MHz) to include in the calibration.
            cable_loss: Cable loss (dB); reserved for future use.

        Returns:
            Dict: output_path, summary_path, rows_written, rows_missing. Never
            raises; missing files / generation errors populate `warnings`.
        """
        result: Dict[str, Any] = {
            "output_path": None,
            "summary_path": None,
            "rows_written": None,
            "rows_missing": None,
            "warnings": [],
        }

        for label, path in (
            ("power_measurement_file", power_measurement_file),
            ("gain_standard_file", gain_standard_file),
            ("hpol_file", hpol_file),
            ("vpol_file", vpol_file),
        ):
            if not path or not os.path.isfile(path):
                result["warnings"].append(f"file_not_found: {label}={path!r}")
        if result["warnings"]:
            return result
        if not freq_list:
            result["warnings"].append("empty_freq_list")
            return result

        try:
            cal = generate_active_cal_file(
                power_measurement_file=power_measurement_file,
                gain_standard_file=gain_standard_file,
                hpol_file=hpol_file,
                vpol_file=vpol_file,
                cable_loss=cable_loss,
                freq_list=list(freq_list),
            )
        except Exception as exc:
            result["warnings"].append(f"generate_active_cal_failed: {exc}")
            return result

        if isinstance(cal, dict):
            result.update(
                {
                    "output_path": cal.get("output_path"),
                    "summary_path": cal.get("summary_path"),
                    "rows_written": cal.get("rows_written"),
                    "rows_missing": cal.get("rows_missing"),
                }
            )
            if cal.get("rows_missing"):
                result["warnings"].append(
                    f"{cal['rows_missing']} frequencies had no data and were skipped"
                )
        return result
