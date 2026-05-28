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
) -> Dict[str, Any]:
    """Generate a drift report file.

    Args:
        baseline_run_id: run_id of the baseline.
        current_run_id: run_id of the current run.
        output_path: Absolute path where the report is written.
        format: "markdown" | "pdf" | "png".

    Returns:
        On success: {"output_path": str, "format": str}. On failure:
        {"error": str}. Never raises — failures are returned, not thrown,
        per the MCP tool contract.
    """
    fmt = format.lower().strip()
    valid = ("markdown", "pdf", "png")
    if fmt not in valid:
        return {"error": f"unknown report format {format!r}; use {'|'.join(valid)}"}
    try:
        result = cal_drift.compute_drift(baseline_run_id, current_run_id)
        if fmt == "markdown":
            cal_drift.export_markdown(result, output_path)
        elif fmt == "pdf":
            cal_drift.export_pdf(result, output_path)
        else:  # png
            cal_drift.render_delta_plot(result, out_path=output_path)
    except Exception as exc:  # noqa: BLE001 — MCP tools never raise to the client
        return {"error": f"cal_drift_report failed: {exc}"}
    return {"output_path": output_path, "format": fmt}


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


def cal_drift_alert(
    baseline_run_id: str,
    current_run_id: str,
    warn_db: float = 0.5,
    alert_db: float = 1.0,
) -> Dict[str, Any]:
    """Compare two cal runs and classify the drift as OK / WARN / ALERT (#4).

    Computes the drift between baseline and current and applies warn/alert
    thresholds to the worst per-polarization mean and max-absolute drift; also
    flags a setup_group mismatch.

    Args:
        baseline_run_id, current_run_id: run_ids from cal_drift_list_runs.
        warn_db: drift at/above this -> WARN (default 0.5 dB).
        alert_db: drift at/above this -> ALERT (default 1.0 dB).

    Returns:
        Dict: level, worst_mean_abs_db, worst_max_abs_db, reasons, warn_db,
        alert_db. On failure: {error}. Never raises.
    """
    try:
        result = cal_drift.compute_drift(baseline_run_id, current_run_id)
    except Exception as exc:
        return {"error": f"cal_drift_alert failed: {exc}"}
    out = cal_drift.evaluate_drift_alert(result, warn_db=warn_db, alert_db=alert_db)
    out["baseline_run_id"] = baseline_run_id
    out["current_run_id"] = current_run_id
    return out


def cal_drift_monitor(
    baseline_run_id: str,
    antenna: Optional[str] = None,
    band: Optional[str] = None,
    warn_db: float = 0.5,
    alert_db: float = 1.0,
) -> Dict[str, Any]:
    """Compare the most-recent cal run to a baseline and alert (#33).

    Cron-friendly: finds the latest recorded run (optionally filtered by
    antenna/band), compares it to the baseline, and returns the drift alert.
    Intended to be scheduled so chamber drift is caught automatically.

    Args:
        baseline_run_id: the reference run to compare against.
        antenna, band: optional filters to select the latest run.
        warn_db, alert_db: alert thresholds (dB).

    Returns:
        Dict: latest_run_id, plus the cal_drift_alert fields. On failure:
        {error}. Never raises.
    """
    try:
        df = cal_drift.list_runs(antenna=antenna, band=band)
        if df is None or len(df) == 0:
            return {"error": "no_recorded_runs", "warnings": ["nothing to monitor"]}
        # Latest by date then time if present.
        sort_cols = [c for c in ("date", "time") if c in df.columns]
        latest = df.sort_values(sort_cols).iloc[-1] if sort_cols else df.iloc[-1]
        latest_id = str(latest["run_id"])
    except Exception as exc:
        return {"error": f"cal_drift_monitor failed: {exc}"}
    if latest_id == baseline_run_id:
        return {
            "latest_run_id": latest_id,
            "level": "OK",
            "reasons": ["latest run is the baseline; nothing newer to compare"],
        }
    out = cal_drift_alert(baseline_run_id, latest_id, warn_db=warn_db, alert_db=alert_db)
    out["latest_run_id"] = latest_id
    return out


def cal_drift_recert_check(
    cal_date: str,
    interval_months: int = 12,
    last_recert_date: Optional[str] = None,
) -> Dict[str, Any]:
    """Gain-standard recertification status for a cal run (#3).

    Reports whether the gain standard's certification was current as of the
    cal date, given its last recert and the recert interval.

    Args:
        cal_date: calibration date "YYYY-MM-DD".
        interval_months: recert validity interval (default 12).
        last_recert_date: last recertification date "YYYY-MM-DD" (required for
            a definite verdict; otherwise status is "unknown").

    Returns:
        Dict: status ("ok"|"due"|"unknown"), months_since_recert, due_date,
        interval_months, cal_date, last_recert_date. Never raises.
    """
    return cal_drift.gain_standard_recert_status(
        cal_date, interval_months=interval_months, last_recert_date=last_recert_date
    )


def cal_drift_cable_loss_history(
    antenna: Optional[str] = None, band: Optional[str] = None
) -> List[Dict[str, Any]]:
    """Cable-loss .s2p references tracked across recorded cal runs (#5).

    Returns one entry per run that recorded a cable-loss file, date-ordered, so
    a cable swap (a common chamber-drift source) is auditable.

    Returns: list of {run_id, date, cal_type, cable_loss_file, cable_loss_sha256}.
    """
    return cal_drift.cable_loss_history(antenna=antenna, band=band)


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
    mcp.tool()(cal_drift_alert)
    mcp.tool()(cal_drift_monitor)
    mcp.tool()(cal_drift_recert_check)
    mcp.tool()(cal_drift_cable_loss_history)
