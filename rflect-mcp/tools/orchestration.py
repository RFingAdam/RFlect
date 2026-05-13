"""
Orchestration tools for RFlect MCP Server.

Provides `process_folder` — a single entry point that scans a folder of
chamber output, picks the right workflow (passive pair, active TRP, cal
drift, UWB), runs it, and optionally produces a DOCX report. Designed so
an AI client (Claude, Cline, etc.) can run a standard RFlect procedure
with one call instead of chaining list + bulk + analyze + report.
"""

from __future__ import annotations

import os
from typing import Any, Dict, List, Optional


VALID_INTENTS = {"auto", "passive", "active", "cal_drift", "uwb"}


def _scan_folder(folder_path: str) -> Dict[str, List[str]]:
    """Categorize files in a folder by intent. Names only (no paths)."""
    files = os.listdir(folder_path)
    txt_files = [f for f in files if f.lower().endswith(".txt")]
    s2p_files = [f for f in files if f.lower().endswith(".s2p")]
    csv_files = [f for f in files if f.lower().endswith(".csv")]

    cal_files = [
        f for f in txt_files
        if f.startswith("TRP Cal ") and not f.startswith("TRP Cal Summary")
    ]
    hpol = [f for f in txt_files if "_HPol" in f]
    vpol = [f for f in txt_files if "_VPol" in f]
    # Active TRP: contains TRP but not "TRP Cal" prefix
    trp = [
        f for f in txt_files
        if "TRP" in f.upper() and not f.startswith("TRP Cal")
    ]
    uwb_csv = [
        f for f in csv_files
        if any(tok in f for tok in ("S21", "GroupDelay")) or "deg" in f.lower()
    ]

    return {
        "cal_files": cal_files,
        "hpol": hpol,
        "vpol": vpol,
        "trp": trp,
        "uwb": s2p_files + uwb_csv,
    }


def _detect_intent(scan: Dict[str, List[str]]) -> tuple[Optional[str], List[str]]:
    """Pick the dominant intent. Priority: cal_drift > passive > active > uwb.

    Returns (intent, mixed_intents_summary[]). The summary is non-empty when
    more than one category had files, so callers can surface a warning.
    """
    categories = []
    if scan["cal_files"]:
        categories.append(("cal_drift", len(scan["cal_files"])))
    paired = min(len(scan["hpol"]), len(scan["vpol"]))
    if paired > 0:
        categories.append(("passive", paired))
    if scan["trp"]:
        categories.append(("active", len(scan["trp"])))
    if scan["uwb"]:
        categories.append(("uwb", len(scan["uwb"])))

    if not categories:
        return None, []

    priority = {"cal_drift": 0, "passive": 1, "active": 2, "uwb": 3}
    categories.sort(key=lambda c: priority[c[0]])
    chosen = categories[0][0]

    mixed: List[str] = []
    if len(categories) > 1:
        summary = ", ".join(f"{name}({count})" for name, count in categories)
        mixed.append(f"mixed_intents_detected: {summary} — chose {chosen}")
    return chosen, mixed


def _run_passive(
    folder_path: str,
    scan: Dict[str, List[str]],
    freqs: Optional[List[float]],
) -> Dict[str, Any]:
    """Process every matched HPOL/VPOL pair in the folder."""
    from plot_antenna.file_utils import batch_process_passive_scans
    from plot_antenna.calculations import extract_passive_frequencies

    warnings: List[str] = []
    paired = min(len(scan["hpol"]), len(scan["vpol"]))
    if paired == 0:
        return {
            "files_processed": 0,
            "frequencies_loaded": [],
            "warnings": ["no_files_matched_intent: passive"],
        }

    if len(scan["hpol"]) != len(scan["vpol"]):
        unmatched_h = sorted(set(scan["hpol"]) - {h for h in scan["hpol"][:paired]})
        unmatched_v = sorted(set(scan["vpol"]) - {v for v in scan["vpol"][:paired]})
        if unmatched_h:
            warnings.append(f"unmatched_hpol: {unmatched_h}")
        if unmatched_v:
            warnings.append(f"unmatched_vpol: {unmatched_v}")

    first_hpol = os.path.join(folder_path, sorted(scan["hpol"])[0])
    available = extract_passive_frequencies(first_hpol) or []
    if not available:
        return {
            "files_processed": 0,
            "frequencies_loaded": [],
            "warnings": warnings + ["could_not_extract_frequencies"],
        }

    if freqs:
        invalid = [f for f in freqs if f not in available]
        valid = [f for f in freqs if f in available]
        if invalid:
            warnings.append(f"invalid_frequencies_dropped: {invalid}")
        selected = valid or available
    else:
        selected = available

    batch_process_passive_scans(
        folder_path=folder_path,
        freq_list=available,
        selected_frequencies=selected,
        cable_loss=0.0,
        datasheet_plots=False,
        save_base=folder_path,
    )

    return {
        "files_processed": paired * 2,
        "frequencies_loaded": selected,
        "warnings": warnings,
    }


def _run_active(
    folder_path: str,
    scan: Dict[str, List[str]],
) -> Dict[str, Any]:
    """Batch-process every TRP file in the folder."""
    from plot_antenna.file_utils import batch_process_active_scans

    if not scan["trp"]:
        return {
            "files_processed": 0,
            "frequencies_loaded": [],
            "warnings": ["no_files_matched_intent: active"],
        }

    batch_process_active_scans(
        folder_path=folder_path,
        save_base=folder_path,
        interpolate=True,
    )
    return {
        "files_processed": len(scan["trp"]),
        "frequencies_loaded": [],
        "warnings": [],
    }


def _run_cal_drift(folder_path: str) -> Dict[str, Any]:
    """Ingest TRP Cal files into the cal-drift history."""
    from plot_antenna import cal_drift

    result = cal_drift.import_historical_dir(folder_path)
    warnings: List[str] = []
    if result.get("failed"):
        warnings.append(f"cal_drift_failed_count: {result['failed']}")
    if result.get("ingested") == 0 and result.get("skipped_duplicate") == 0:
        warnings.append("no_files_matched_intent: cal_drift")
    return {
        "files_processed": result.get("ingested", 0),
        "frequencies_loaded": [],
        "warnings": warnings,
        "extra": {
            "skipped_duplicate": result.get("skipped_duplicate", 0),
            "failed": result.get("failed", 0),
            "run_ids": result.get("run_ids", []),
        },
    }


def _run_uwb(
    folder_path: str,
    scan: Dict[str, List[str]],
) -> Dict[str, Any]:
    """Run channel analysis on each S-parameter file in the folder."""
    from .uwb_tools import _analyze_uwb_channel_dict

    if not scan["uwb"]:
        return {
            "files_processed": 0,
            "frequencies_loaded": [],
            "warnings": ["no_files_matched_intent: uwb"],
        }

    warnings: List[str] = []
    per_file: List[Dict[str, Any]] = []
    processed = 0
    for name in sorted(scan["uwb"]):
        path = os.path.join(folder_path, name)
        try:
            res = _analyze_uwb_channel_dict(path)
            if "error" in res:
                warnings.append(f"uwb_failed: {name}: {res['error']}")
                continue
            per_file.append({"file": name, "analysis": res})
            processed += 1
        except Exception as exc:  # pragma: no cover - defensive
            warnings.append(f"uwb_failed: {name}: {exc}")

    return {
        "files_processed": processed,
        "frequencies_loaded": [],
        "warnings": warnings,
        "extra": {"results": per_file},
    }


def _maybe_generate_report(
    requested: bool,
    report_path: Optional[str],
    folder_path: str,
) -> tuple[Optional[str], List[str]]:
    """Generate a default report when requested, else (None, [])."""
    if not requested:
        return None, []

    from .import_tools import get_loaded_measurements

    if not get_loaded_measurements():
        return None, ["report_skipped_no_loaded_measurements"]

    target = report_path or os.path.join(folder_path, "process_folder_report.docx")
    try:
        from tools.report_tools import (
            _prepare_report_data,
            _generate_plots,
            _create_llm_provider,
            _build_branded_docx,
            ReportOptions,
        )
        import tempfile

        measurements = get_loaded_measurements()
        opts = ReportOptions()
        report_data = _prepare_report_data(measurements, opts)
        if not report_data["measurements"]:
            return None, ["report_no_measurements_after_filter"]

        plot_images: Dict[str, List[str]] = {}
        temp_dir = tempfile.mkdtemp(prefix="rflect_process_folder_report_")
        if opts.include_2d_plots or opts.include_3d_plots:
            plot_images = _generate_plots(measurements, opts, temp_dir)

        provider = _create_llm_provider(opts)
        _build_branded_docx(
            target, report_data, plot_images,
            opts, provider, {}, measurements,
        )
        return target, []
    except Exception as exc:
        return None, [f"report_generation_failed: {exc}"]


def register_orchestration_tools(mcp):
    """Register the `process_folder` orchestration tool."""

    @mcp.tool()
    def process_folder(
        folder_path: str,
        intent: str = "auto",
        report: bool = False,
        freqs: Optional[List[float]] = None,
        report_path: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Run a standard RFlect procedure on every relevant file in a folder.

        One-call workflow: scan → pick intent → import + plot/analyze → (optional) report.
        Auto-detect uses priority cal_drift > passive > active > uwb.

        Args:
            folder_path: Absolute path to scan (non-recursive).
            intent: 'auto' | 'passive' | 'active' | 'cal_drift' | 'uwb'.
            report: If True and measurements were loaded, generate a default DOCX.
            freqs: For passive intent, restrict processing to these MHz values.
            report_path: Explicit DOCX output path (defaults to <folder>/process_folder_report.docx).

        Returns:
            Dict with keys: intent_used, files_scanned, files_processed,
            frequencies_loaded, warnings, report_path. Never raises;
            failures surface in `warnings`.
        """
        result: Dict[str, Any] = {
            "intent_used": None,
            "files_scanned": 0,
            "files_processed": 0,
            "frequencies_loaded": [],
            "warnings": [],
            "report_path": None,
        }

        if intent not in VALID_INTENTS:
            result["warnings"].append(
                f"invalid_intent: {intent!r}; expected one of {sorted(VALID_INTENTS)}"
            )
            return result

        if not os.path.isdir(folder_path):
            result["warnings"].append(f"folder_not_found: {folder_path}")
            return result

        scan = _scan_folder(folder_path)
        result["files_scanned"] = sum(len(v) for v in scan.values())

        chosen = intent
        if intent == "auto":
            detected, mixed = _detect_intent(scan)
            result["warnings"].extend(mixed)
            if detected is None:
                result["warnings"].append("no_files_matched_intent: auto")
                return result
            chosen = detected

        result["intent_used"] = chosen

        try:
            if chosen == "passive":
                run = _run_passive(folder_path, scan, freqs)
            elif chosen == "active":
                run = _run_active(folder_path, scan)
            elif chosen == "cal_drift":
                run = _run_cal_drift(folder_path)
            elif chosen == "uwb":
                run = _run_uwb(folder_path, scan)
            else:  # pragma: no cover - VALID_INTENTS guards this
                result["warnings"].append(f"unhandled_intent: {chosen}")
                return result
        except Exception as exc:
            result["warnings"].append(f"{chosen}_failed: {exc}")
            return result

        result["files_processed"] = run.get("files_processed", 0)
        result["frequencies_loaded"] = run.get("frequencies_loaded", [])
        result["warnings"].extend(run.get("warnings", []))
        if "extra" in run:
            result["extra"] = run["extra"]

        path, report_warnings = _maybe_generate_report(report, report_path, folder_path)
        result["report_path"] = path
        result["warnings"].extend(report_warnings)
        return result
