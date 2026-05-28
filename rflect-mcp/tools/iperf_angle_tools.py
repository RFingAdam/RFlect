"""
Multi-angle iperf-sweep analysis tool for the RFlect MCP server.

Ingests a pair of bench session folders (an installed-antenna session and a
matched reference-antenna session) produced by an external angle-sweep
runner. Each session is expected to contain a ``session.json`` manifest in
the documented bench format described in ``analyze_iperf_angle_sweep``.

The tool computes per-angle deltas between the installed and reference
antennas at matching (channel, mode) cells, emits a CSV / JSON summary,
renders a polar plot per (channel, mode), and writes a short markdown
report. It is intentionally decoupled from any specific bench harness — it
only reads JSON.
"""

from __future__ import annotations

import csv
import json
import os
import statistics
from collections import defaultdict
from typing import Any, Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


# ---------------------------------------------------------------------------
# Session-JSON loader (no bench-code dependency)
# ---------------------------------------------------------------------------


def _load_session_runs(session_dir: str) -> Tuple[List[Dict[str, Any]], List[str]]:
    """Parse ``<session_dir>/session.json`` and return its wifi-only runs.

    Returns (runs, warnings). Each run is filtered to those with a numeric
    ``angle_deg``, ``wifi_channel``, ``iperf_mode``, and
    ``aggregate.overall_mbps`` populated; anything missing those fields is
    skipped with a warning.
    """
    warnings: List[str] = []
    manifest_path = os.path.join(session_dir, "session.json")
    if not os.path.isfile(manifest_path):
        return [], [f"manifest_missing: {manifest_path}"]
    try:
        with open(manifest_path, "r") as fh:
            manifest = json.load(fh)
    except (OSError, json.JSONDecodeError) as exc:
        return [], [f"manifest_parse_failed: {exc}"]

    runs_raw = manifest.get("runs", [])
    if not isinstance(runs_raw, list):
        return [], [f"manifest_runs_not_list: {type(runs_raw).__name__}"]

    keepers: List[Dict[str, Any]] = []
    for idx, rec in enumerate(runs_raw):
        if not isinstance(rec, dict):
            warnings.append(f"run_{idx}_not_dict")
            continue
        if rec.get("kind") != "wifi_only":
            continue  # caller is interested in iperf-only cells
        angle = rec.get("angle_deg")
        channel = rec.get("wifi_channel")
        mode = rec.get("iperf_mode")
        aggregate = rec.get("aggregate") or {}
        mbps = aggregate.get("overall_mbps")
        if angle is None or channel is None or mode is None or mbps is None:
            warnings.append(
                f"run_{idx}_missing_field: "
                f"angle_deg={angle!r} wifi_channel={channel!r} "
                f"iperf_mode={mode!r} overall_mbps={mbps!r}"
            )
            continue
        try:
            keepers.append({
                "angle_deg": float(angle),
                "wifi_channel": int(channel),
                "iperf_mode": str(mode),
                "overall_mbps": float(mbps),
                "loss_percent": (
                    None if aggregate.get("loss_percent") is None
                    else float(aggregate["loss_percent"])
                ),
                "retransmits": aggregate.get("retransmits"),
            })
        except (TypeError, ValueError) as exc:
            warnings.append(f"run_{idx}_field_coerce_failed: {exc}")
    return keepers, warnings


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------


def _index_by_cell(runs: List[Dict[str, Any]]) -> Dict[Tuple[int, str], Dict[float, float]]:
    """Group runs into {(channel, mode) -> {angle_deg -> overall_mbps}}.

    If multiple runs share the same (channel, mode, angle) cell, the mean is
    used. This is robust against retries during a sweep.
    """
    grouped: Dict[Tuple[int, str], Dict[float, List[float]]] = defaultdict(lambda: defaultdict(list))
    for r in runs:
        grouped[(r["wifi_channel"], r["iperf_mode"])][r["angle_deg"]].append(r["overall_mbps"])
    return {
        cell: {a: sum(v) / len(v) for a, v in by_angle.items()}
        for cell, by_angle in grouped.items()
    }


def _summarize_cell(
    delta_by_angle: Dict[float, float],
    installed_by_angle: Dict[float, float],
    reference_by_angle: Dict[float, float],
) -> Dict[str, Any]:
    angles = sorted(delta_by_angle)
    deltas = [delta_by_angle[a] for a in angles]
    if not deltas:
        return {
            "n_angles": 0,
            "mean_delta_mbps": None,
            "median_delta_mbps": None,
            "worst_delta_mbps": None,
            "worst_angle_deg": None,
            "best_delta_mbps": None,
            "best_angle_deg": None,
            "spread_mbps": None,
            "p10_delta_mbps": None,
            "p90_delta_mbps": None,
            "mean_installed_mbps": None,
            "mean_reference_mbps": None,
        }
    arr = np.asarray(deltas, dtype=float)
    min_i = int(np.argmin(arr))
    max_i = int(np.argmax(arr))
    return {
        "n_angles": int(arr.size),
        "mean_delta_mbps": float(np.mean(arr)),
        "median_delta_mbps": float(np.median(arr)),
        "worst_delta_mbps": float(arr.min()),
        "worst_angle_deg": float(angles[min_i]),
        "best_delta_mbps": float(arr.max()),
        "best_angle_deg": float(angles[max_i]),
        "spread_mbps": float(arr.max() - arr.min()),
        "p10_delta_mbps": float(np.percentile(arr, 10)),
        "p90_delta_mbps": float(np.percentile(arr, 90)),
        "mean_installed_mbps": float(np.mean(list(installed_by_angle.values()))),
        "mean_reference_mbps": float(np.mean(list(reference_by_angle.values()))),
    }


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------


def _render_polar(
    installed_by_angle: Dict[float, float],
    reference_by_angle: Dict[float, float],
    *,
    channel: int,
    mode: str,
    out_path: str,
) -> None:
    angles_deg = sorted(set(installed_by_angle).union(reference_by_angle))
    theta = np.radians(angles_deg)

    inst_vals = [installed_by_angle.get(a, np.nan) for a in angles_deg]
    ref_vals = [reference_by_angle.get(a, np.nan) for a in angles_deg]

    # Close the polar contour by repeating the first sample.
    if angles_deg:
        theta_closed = np.append(theta, theta[0])
        inst_closed = inst_vals + [inst_vals[0]]
        ref_closed = ref_vals + [ref_vals[0]]
    else:
        theta_closed = theta
        inst_closed = inst_vals
        ref_closed = ref_vals

    fig = plt.figure(figsize=(6, 6))
    ax = fig.add_subplot(111, projection="polar")
    ax.set_theta_zero_location("N")
    ax.set_theta_direction(-1)
    ax.plot(theta_closed, inst_closed, "-o", label="installed", linewidth=2)
    ax.plot(theta_closed, ref_closed, "--s", label="reference", linewidth=1.5)
    if ref_vals:
        ref_mean = float(np.nanmean(ref_vals))
        ax.plot(np.linspace(0, 2 * np.pi, 360), [ref_mean] * 360,
                ":", color="gray", linewidth=1, label=f"reference mean ({ref_mean:.1f})")
    ax.set_title(f"Ch {channel}  {mode}  Mbps vs azimuth")
    ax.legend(loc="lower left", bbox_to_anchor=(1.05, 0.0), fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Markdown report
# ---------------------------------------------------------------------------


def _render_report_md(
    summary_rows: List[Dict[str, Any]],
    polar_plot_paths: Dict[Tuple[int, str], str],
    *,
    installed_session: str,
    reference_session: str,
    mean_threshold_mbps: float,
    worst_threshold_mbps: float,
) -> str:
    L: List[str] = []
    L.append("# Multi-angle iperf sweep — installed vs reference")
    L.append("")
    L.append(f"- installed session: `{installed_session}`")
    L.append(f"- reference session: `{reference_session}`")
    L.append(f"- verdict thresholds: mean Δ ≥ {mean_threshold_mbps:+.1f} Mbps, "
             f"worst Δ ≥ {worst_threshold_mbps:+.1f} Mbps → adequate")
    L.append("")
    L.append("## Methodology")
    L.append("")
    L.append("Throughput is reported as a single ``overall_mbps`` per "
             "(channel, mode, angle) cell, drawn from the per-run "
             "``aggregate.overall_mbps`` field in the bench manifest. Cells "
             "are matched between the two sessions on (channel, mode, angle); "
             "angles present in only one session are dropped. Statistics are "
             "computed over the per-angle deltas (installed − reference).")
    L.append("")
    L.append("## Per-cell summary")
    L.append("")
    L.append("| Ch | Mode | n | mean Δ Mbps | median Δ | worst Δ @ angle | best Δ @ angle | spread | p10/p90 | verdict |")
    L.append("|---:|------|--:|------------:|---------:|----------------:|---------------:|-------:|--------:|---------|")
    for r in summary_rows:
        if r["n_angles"] == 0:
            L.append(f"| {r['wifi_channel']} | {r['iperf_mode']} | 0 | — | — | — | — | — | — | (no data) |")
            continue
        verdict = (
            "adequate"
            if (r["mean_delta_mbps"] >= mean_threshold_mbps
                and r["worst_delta_mbps"] >= worst_threshold_mbps)
            else "investigate"
        )
        L.append(
            f"| {r['wifi_channel']} | {r['iperf_mode']} | {r['n_angles']} "
            f"| {r['mean_delta_mbps']:+.2f} "
            f"| {r['median_delta_mbps']:+.2f} "
            f"| {r['worst_delta_mbps']:+.2f} @ {r['worst_angle_deg']:.0f}° "
            f"| {r['best_delta_mbps']:+.2f} @ {r['best_angle_deg']:.0f}° "
            f"| {r['spread_mbps']:.2f} "
            f"| {r['p10_delta_mbps']:+.2f} / {r['p90_delta_mbps']:+.2f} "
            f"| {verdict} |"
        )
    L.append("")
    if polar_plot_paths:
        L.append("## Polar plots")
        L.append("")
        for (ch, mode), path in sorted(polar_plot_paths.items()):
            L.append(f"### Ch {ch} — {mode}")
            L.append("")
            L.append(f"![Ch {ch} {mode}]({os.path.basename(path)})")
            L.append("")
    return "\n".join(L) + "\n"


# ---------------------------------------------------------------------------
# Tool registration
# ---------------------------------------------------------------------------


def register_iperf_angle_tools(mcp):
    """Register the iperf-angle sweep tools with the MCP server."""

    @mcp.tool()
    def analyze_iperf_angle_sweep(
        session_dir: str,
        reference_session_dir: str,
        out_dir: str,
        mean_threshold_mbps: float = -5.0,
        worst_threshold_mbps: float = -10.0,
    ) -> Dict[str, Any]:
        """
        Compute per-angle delta statistics between an installed-antenna
        session and a matched reference-antenna session.

        Inputs are two bench session folders, each containing a
        ``session.json`` manifest with this minimal shape:

            {
              "session_name": "...",
              "runs": [
                {
                  "kind": "wifi_only",
                  "wifi_channel": 6,
                  "iperf_mode": "tcp_up",
                  "angle_deg": 45.0,
                  "aggregate": {"overall_mbps": 123.4, ...}
                },
                ...
              ]
            }

        Only ``kind == "wifi_only"`` runs are considered. Cells without all
        four fields populated (``wifi_channel``, ``iperf_mode``,
        ``angle_deg``, ``aggregate.overall_mbps``) are skipped with a
        warning in the response.

        Args:
            session_dir: Folder containing the installed-antenna session.
            reference_session_dir: Folder containing the reference-antenna session.
            out_dir: Directory to write summary.csv, summary.json, polar PNGs,
                and report.md. Created if it does not exist.
            mean_threshold_mbps: A cell is flagged "adequate" if its mean
                delta is at least this value AND the worst delta is at least
                ``worst_threshold_mbps``. Both default to negative values
                because the typical sign is installed - reference < 0.
            worst_threshold_mbps: See above.

        Returns:
            Dict with keys: summary_csv, summary_json, report_md, polar_pngs,
            n_cells, warnings. Never raises; failures surface in ``warnings``.
        """
        result: Dict[str, Any] = {
            "summary_csv": None,
            "summary_json": None,
            "report_md": None,
            "polar_pngs": [],
            "n_cells": 0,
            "warnings": [],
        }

        installed_runs, w1 = _load_session_runs(session_dir)
        reference_runs, w2 = _load_session_runs(reference_session_dir)
        result["warnings"].extend([f"installed: {w}" for w in w1])
        result["warnings"].extend([f"reference: {w}" for w in w2])

        if not installed_runs:
            result["warnings"].append("no_installed_runs")
        if not reference_runs:
            result["warnings"].append("no_reference_runs")
        if not installed_runs or not reference_runs:
            return result

        try:
            os.makedirs(out_dir, exist_ok=True)
        except OSError as exc:
            result["warnings"].append(f"out_dir_create_failed: {exc}")
            return result

        installed_by_cell = _index_by_cell(installed_runs)
        reference_by_cell = _index_by_cell(reference_runs)
        common_cells = sorted(set(installed_by_cell).intersection(reference_by_cell))
        if not common_cells:
            result["warnings"].append("no_common_channel_mode_cells")
            return result

        summary_rows: List[Dict[str, Any]] = []
        polar_paths: Dict[Tuple[int, str], str] = {}

        for (ch, mode) in common_cells:
            inst = installed_by_cell[(ch, mode)]
            ref = reference_by_cell[(ch, mode)]
            common_angles = sorted(set(inst).intersection(ref))
            if not common_angles:
                result["warnings"].append(
                    f"ch{ch}_{mode}: no common angles between sessions"
                )
                summary_rows.append({
                    "wifi_channel": ch, "iperf_mode": mode,
                    **_summarize_cell({}, {}, {}),
                })
                continue
            delta_by_angle = {a: inst[a] - ref[a] for a in common_angles}
            inst_common = {a: inst[a] for a in common_angles}
            ref_common = {a: ref[a] for a in common_angles}
            summary = _summarize_cell(delta_by_angle, inst_common, ref_common)
            summary_rows.append({
                "wifi_channel": ch,
                "iperf_mode": mode,
                **summary,
            })

            # Polar PNG
            png_name = f"polar_ch{ch:02d}_{mode}_mbps.png"
            png_path = os.path.join(out_dir, png_name)
            try:
                _render_polar(inst_common, ref_common,
                              channel=ch, mode=mode, out_path=png_path)
                polar_paths[(ch, mode)] = png_path
            except Exception as exc:
                result["warnings"].append(f"polar_render_failed_ch{ch}_{mode}: {exc}")

        # CSV
        csv_path = os.path.join(out_dir, "summary.csv")
        csv_fields = [
            "wifi_channel", "iperf_mode", "n_angles",
            "mean_delta_mbps", "median_delta_mbps",
            "worst_delta_mbps", "worst_angle_deg",
            "best_delta_mbps", "best_angle_deg",
            "spread_mbps", "p10_delta_mbps", "p90_delta_mbps",
            "mean_installed_mbps", "mean_reference_mbps",
        ]
        try:
            with open(csv_path, "w", newline="") as fh:
                writer = csv.DictWriter(fh, fieldnames=csv_fields)
                writer.writeheader()
                for r in summary_rows:
                    writer.writerow({k: r.get(k) for k in csv_fields})
            result["summary_csv"] = csv_path
        except OSError as exc:
            result["warnings"].append(f"csv_write_failed: {exc}")

        # JSON
        json_path = os.path.join(out_dir, "summary.json")
        try:
            with open(json_path, "w") as fh:
                json.dump({
                    "installed_session": os.path.basename(session_dir.rstrip("/")),
                    "reference_session": os.path.basename(reference_session_dir.rstrip("/")),
                    "mean_threshold_mbps": mean_threshold_mbps,
                    "worst_threshold_mbps": worst_threshold_mbps,
                    "cells": summary_rows,
                }, fh, indent=2)
            result["summary_json"] = json_path
        except OSError as exc:
            result["warnings"].append(f"json_write_failed: {exc}")

        # Markdown report
        report_path = os.path.join(out_dir, "report.md")
        try:
            with open(report_path, "w") as fh:
                fh.write(_render_report_md(
                    summary_rows,
                    polar_paths,
                    installed_session=os.path.basename(session_dir.rstrip("/")),
                    reference_session=os.path.basename(reference_session_dir.rstrip("/")),
                    mean_threshold_mbps=mean_threshold_mbps,
                    worst_threshold_mbps=worst_threshold_mbps,
                ))
            result["report_md"] = report_path
        except OSError as exc:
            result["warnings"].append(f"report_md_write_failed: {exc}")

        result["polar_pngs"] = sorted(polar_paths.values())
        result["n_cells"] = len(summary_rows)
        return result
