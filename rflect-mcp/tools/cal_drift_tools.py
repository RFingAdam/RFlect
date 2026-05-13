"""
Calibration Drift Tools for RFlect MCP Server

Exposes the cal-drift history: ingest archived TRP Cal files, list recorded
runs, compare any two runs, and export drift reports. Backed by
plot_antenna.cal_drift (same logic the RFlect GUI uses).
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from plot_antenna import cal_drift


def cal_drift_ingest(directory: str) -> Dict[str, Any]:
    """Walk a directory and record every TRP Cal file found into drift history.

    Pairs each cal file with its sibling summary when present; duplicates
    (by output SHA-256) are skipped. Returns counts + new run_ids.

    Args:
        directory: Absolute path to scan (e.g. "~/Downloads/Calibration Data/").

    Returns:
        {ingested, skipped_duplicate, failed, run_ids: [...]}.
    """
    return cal_drift.import_historical_dir(directory)


def cal_drift_list_runs(
    antenna: Optional[str] = None,
    band: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """List recorded calibration runs, optionally filtered.

    Args:
        antenna: Optional filter on the antenna code (e.g. "BLPA", "HORN").
        band: Optional filter on the band_label (e.g. "690-2700").

    Returns:
        Sorted list of run metadata dicts (one per CalRunMeta).
    """
    df = cal_drift.list_runs(antenna=antenna, band=band)
    return df.to_dict(orient="records")


def cal_drift_compare(
    baseline_run_id: str,
    current_run_id: str,
    max_delta_rows: int = 500,
) -> Dict[str, Any]:
    """Compute per-frequency ΔdB between two recorded runs, with consistency diff.

    Args:
        baseline_run_id: run_id of the earlier/reference calibration.
        current_run_id: run_id of the calibration being checked.
        max_delta_rows: Cap on rows returned in the "deltas" list (default 500;
            use `cal_drift_report` for full-detail output).

    Returns:
        JSON-safe dict with baseline/current metadata, summary stats (H/V),
        method-consistency map, missing-frequency audit, and per-frequency
        deltas (truncated if over the cap).
    """
    result = cal_drift.compute_drift(baseline_run_id, current_run_id)
    return cal_drift.result_to_dict(result, max_delta_rows=max_delta_rows)


def cal_drift_report(
    baseline_run_id: str,
    current_run_id: str,
    output_path: str,
    format: str = "markdown",
) -> str:
    """Generate a drift report file and return its path.

    Args:
        baseline_run_id: run_id of the baseline.
        current_run_id: run_id of the current run.
        output_path: Absolute path where the report is written.
        format: "markdown" | "pdf" | "png".

    Returns:
        The absolute output path on success (raises ValueError on unknown format).
    """
    fmt = format.lower().strip()
    result = cal_drift.compute_drift(baseline_run_id, current_run_id)
    if fmt == "markdown":
        cal_drift.export_markdown(result, output_path)
    elif fmt == "pdf":
        cal_drift.export_pdf(result, output_path)
    elif fmt == "png":
        cal_drift.render_delta_plot(result, out_path=output_path)
    else:
        raise ValueError(f"Unknown report format {format!r}; use markdown|pdf|png")
    return output_path


def cal_drift_history_dir() -> str:
    """Return the currently-active cal-drift history directory."""
    return str(cal_drift.history_dir())


def cal_drift_set_history_dir(directory: str) -> str:
    """Persist a new cal-drift history directory (written to user_settings.json).

    Args:
        directory: Absolute path. Created if it does not exist.

    Returns:
        The resolved absolute path now in effect.
    """
    return str(cal_drift.set_history_dir(directory))


def cal_drift_set_setup_group(run_id: str, setup_group: str) -> bool:
    """Assign a methodology-epoch tag to a recorded run.

    The setup_group is free-text (e.g. "pre-2024-cable-change",
    "2026-v2-mount") and is flagged as mismatched on the consistency tab
    whenever two runs in different groups are compared — a loud visual
    signal that the comparison may not be apples-to-apples.

    Args:
        run_id: run_id from `cal_drift_list_runs`.
        setup_group: New setup-group tag (empty string to clear).

    Returns:
        True on success, False if the run_id is not found.
    """
    return cal_drift.set_setup_group(run_id, setup_group)


def cal_drift_set_notes(run_id: str, notes: str) -> bool:
    """Update the operator_notes field on a recorded run.

    Args:
        run_id: run_id from `cal_drift_list_runs`.
        notes: New free-text notes.

    Returns:
        True on success, False if the run_id is not found.
    """
    return cal_drift.update_notes(run_id, notes)


def register_cal_drift_tools(mcp):
    """Register calibration-drift tools with the MCP server."""
    mcp.tool()(cal_drift_ingest)
    mcp.tool()(cal_drift_list_runs)
    mcp.tool()(cal_drift_compare)
    mcp.tool()(cal_drift_report)
    mcp.tool()(cal_drift_history_dir)
    mcp.tool()(cal_drift_set_history_dir)
    mcp.tool()(cal_drift_set_setup_group)
    mcp.tool()(cal_drift_set_notes)
