"""
Cross-measurement comparison MCP tool for RFlect.

Overlays N loaded measurements and reports a per-frequency comparison of peak
gain / TRP with deltas versus a reference and the best performer per frequency.
Reuses the deterministic AntennaAnalyzer; no LLM, no network. Returns a
structured dict and never raises; failures populate `warnings`.

Closes the long-standing request for a cross-file overlay comparison tool.
"""

from __future__ import annotations

import csv
import os
from typing import Any, Dict, List, Optional

from plot_antenna.analysis_engine import AntennaAnalyzer

from .import_tools import get_loaded_measurements


def _peak_metric(stats: Dict[str, Any], scan_type: str):
    """Return (value, unit) of the headline 'peak' metric for a scan type."""
    if scan_type == "active":
        return stats.get("max_power_dBm"), "dBm"
    return stats.get("max_gain_dBi"), "dBi"


def register_comparison_tools(mcp):
    """Register cross-measurement comparison tools with the MCP server."""

    @mcp.tool()
    def compare_antennas(
        measurement_names: Optional[List[str]] = None,
        reference: Optional[str] = None,
        out_csv: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Compare loaded antenna measurements head-to-head across frequency.

        For every measurement, reports the peak metric (gain dBi for passive,
        TRP/peak power dBm for active) at each common frequency, the delta versus
        a reference measurement, and the best performer per frequency. Import the
        measurements first (import_antenna_file / import_passive_pair /
        import_antenna_folder), then call this to overlay them.

        Args:
            measurement_names: Names to compare (default: all loaded). Must share
                a scan type for the metric to be apples-to-apples.
            reference: Measurement to compute deltas against (default: the first).
            out_csv: Optional path to also write the comparison table as CSV.

        Returns:
            Dict: metric, unit, measurements, frequencies, rows (per-frequency:
            {freq_mhz, values:{name:peak}, best:{name,value}, deltas_vs_ref}),
            best_overall (name with the highest mean peak), out_csv. Never raises;
            failures populate `warnings`.
        """
        result: Dict[str, Any] = {
            "metric": None,
            "unit": None,
            "measurements": [],
            "frequencies": [],
            "rows": [],
            "best_overall": None,
            "out_csv": None,
            "warnings": [],
        }

        loaded = get_loaded_measurements()
        if not loaded:
            result["warnings"].append("no_data_loaded")
            return result

        names = measurement_names or list(loaded.keys())
        names = [n for n in names if n in loaded]
        missing = [n for n in (measurement_names or []) if n not in loaded]
        for n in missing:
            result["warnings"].append(f"measurement_not_found: {n}")
        if len(names) < 2:
            result["warnings"].append("need_at_least_2_measurements_to_compare")
            return result

        scan_types = {loaded[n].scan_type for n in names}
        if len(scan_types) > 1:
            result["warnings"].append(
                f"mixed_scan_types: {sorted(scan_types)} — peak metric not comparable"
            )
        scan_type = loaded[names[0]].scan_type
        result["metric"] = "peak_power" if scan_type == "active" else "peak_gain"
        result["unit"] = "dBm" if scan_type == "active" else "dBi"
        result["measurements"] = names

        ref = reference if reference in names else names[0]
        if reference and reference not in names:
            result["warnings"].append(f"reference_not_in_set: {reference}; using {ref}")

        # Build analyzers + the union of frequencies (rounded to MHz).
        analyzers: Dict[str, AntennaAnalyzer] = {}
        for n in names:
            m = loaded[n]
            try:
                analyzers[n] = AntennaAnalyzer(
                    measurement_data=m.data, scan_type=m.scan_type, frequencies=m.frequencies
                )
            except Exception as exc:
                result["warnings"].append(f"analyzer_failed:{n}: {exc}")

        all_freqs = sorted({round(float(fq), 1) for n in names for fq in loaded[n].frequencies})
        result["frequencies"] = all_freqs

        sums: Dict[str, float] = {n: 0.0 for n in names}
        counts: Dict[str, int] = {n: 0 for n in names}

        for fq in all_freqs:
            values: Dict[str, float] = {}
            for n in names:
                az = analyzers.get(n)
                if az is None:
                    continue
                try:
                    stats = az.get_gain_statistics(frequency=fq)
                    val, _ = _peak_metric(stats, scan_type)
                    if val is not None:
                        values[n] = float(val)
                        sums[n] += float(val)
                        counts[n] += 1
                except Exception as exc:
                    result["warnings"].append(f"stats_failed:{n}@{fq}: {exc}")
            if not values:
                continue
            best_name = max(values, key=values.get)
            ref_val = values.get(ref)
            deltas = (
                {k: round(v - ref_val, 2) for k, v in values.items()} if ref_val is not None else {}
            )
            result["rows"].append(
                {
                    "freq_mhz": fq,
                    "values": {k: round(v, 2) for k, v in values.items()},
                    "best": {"name": best_name, "value": round(values[best_name], 2)},
                    "deltas_vs_ref": deltas,
                }
            )

        means = {n: (sums[n] / counts[n]) for n in names if counts[n] > 0}
        if means:
            result["best_overall"] = max(means, key=means.get)
            result["mean_peak_by_measurement"] = {k: round(v, 2) for k, v in means.items()}

        if out_csv:
            try:
                with open(out_csv, "w", newline="", encoding="utf-8") as fh:
                    w = csv.writer(fh)
                    w.writerow(["freq_mhz"] + names + ["best", "best_value"])
                    for row in result["rows"]:
                        vals = [row["values"].get(n, "") for n in names]
                        w.writerow(
                            [row["freq_mhz"]] + vals + [row["best"]["name"], row["best"]["value"]]
                        )
                result["out_csv"] = out_csv
            except (OSError, UnicodeError) as exc:
                result["warnings"].append(f"csv_write_failed: {exc}")

        return result
