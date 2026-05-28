"""
Statistical pattern-averaging MCP tool for RFlect (#32).

Averages repeat measurements of the same radiation pattern and reports
per-point spread / repeatability — the pattern-level complement to the
cal-drift framework. Pure (operates on supplied arrays); no LLM, no network.
Returns a structured dict and never raises.
"""

from __future__ import annotations

from typing import Any, Dict, List

import numpy as np


def register_statistics_tools(mcp):
    """Register statistical-averaging tools with the MCP server."""

    @mcp.tool()
    def average_patterns(patterns: List[List[List[float]]]) -> Dict[str, Any]:
        """
        Average N repeat radiation-pattern grids and report repeatability.

        Each pattern is a 2D grid (theta x phi) in dB for the same DUT measured
        repeatedly under nominally identical conditions. All grids must share the
        same shape. Returns the per-point mean grid plus repeatability statistics
        (per-point standard deviation across repeats).

        Args:
            patterns: list of 2D arrays (theta x phi), all the same shape, in dB.

        Returns:
            Dict: n_repeats, shape, mean_pattern_db (2D), max_std_db,
            mean_std_db, peak_to_peak_at_max_db (spread of repeats at the
            mean-peak direction), warnings. Never raises.
        """
        result: Dict[str, Any] = {
            "n_repeats": 0,
            "shape": None,
            "mean_pattern_db": None,
            "max_std_db": None,
            "mean_std_db": None,
            "peak_to_peak_at_max_db": None,
            "warnings": [],
        }
        if not patterns or len(patterns) < 2:
            result["warnings"].append("need_at_least_2_patterns")
            return result
        try:
            stack = np.asarray(patterns, dtype=float)  # (n, theta, phi)
        except (TypeError, ValueError) as exc:
            result["warnings"].append(f"bad_pattern_arrays: {exc}")
            return result
        if stack.ndim != 3:
            result["warnings"].append(f"patterns must be a list of 2D grids; got ndim={stack.ndim}")
            return result

        mean = np.mean(stack, axis=0)
        std = np.std(stack, axis=0, ddof=1) if stack.shape[0] >= 2 else np.zeros_like(mean)

        # Repeatability at the strongest direction (where the mean pattern peaks).
        peak_idx = np.unravel_index(int(np.argmax(mean)), mean.shape)
        repeats_at_peak = stack[:, peak_idx[0], peak_idx[1]]
        p2p = float(np.max(repeats_at_peak) - np.min(repeats_at_peak))

        result["n_repeats"] = int(stack.shape[0])
        result["shape"] = list(mean.shape)
        result["mean_pattern_db"] = mean.tolist()
        result["max_std_db"] = float(np.max(std))
        result["mean_std_db"] = float(np.mean(std))
        result["peak_to_peak_at_max_db"] = p2p
        return result
