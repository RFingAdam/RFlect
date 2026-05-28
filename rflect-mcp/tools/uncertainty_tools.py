"""
Measurement-uncertainty MCP tool for RFlect (#30).

Combines individual uncertainty contributors (all in dB) into a combined
standard uncertainty and an expanded uncertainty using a coverage factor,
following the CTIA / ISO-GUM root-sum-square approach. Deterministic; no LLM,
no network. Returns a structured dict and never raises.
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional

# Type B divisors: convert a stated half-width to a standard uncertainty.
#   normal      -> value is already 1 sigma (divide by 1)
#   rectangular -> half-width / sqrt(3)   (uniform within +/- a)
#   u-shaped    -> half-width / sqrt(2)   (mismatch ripple)
#   triangular  -> half-width / sqrt(6)
_DIVISORS = {
    "normal": 1.0,
    "rectangular": math.sqrt(3.0),
    "u-shaped": math.sqrt(2.0),
    "triangular": math.sqrt(6.0),
}


def register_uncertainty_tools(mcp):
    """Register the measurement-uncertainty tool with the MCP server."""

    @mcp.tool()
    def compute_uncertainty_budget(
        contributors: List[Dict[str, Any]],
        coverage_factor: float = 2.0,
        measured_value_db: Optional[float] = None,
    ) -> Dict[str, Any]:
        """
        Build a TRP/gain measurement-uncertainty budget (dB).

        Each contributor is converted to a standard uncertainty (1 sigma) using
        its distribution, combined in root-sum-square, and scaled by the
        coverage factor to an expanded uncertainty (k=2 ~ 95% confidence).

        Args:
            contributors: list of {name, value_db, distribution}, where value_db
                is the contributor's half-width (or 1-sigma for "normal") in dB
                and distribution is one of: normal, rectangular, u-shaped,
                triangular. Typical contributors: mismatch, cable drift,
                positioner repeatability, reference-antenna gain, instrument
                linearity, chamber/range error.
            coverage_factor: k for the expanded uncertainty (default 2.0).
            measured_value_db: optional nominal value; if given, the response
                includes value +/- expanded uncertainty bounds.

        Returns:
            Dict: combined_standard_uncertainty_db, expanded_uncertainty_db,
            coverage_factor, n_contributors, per_contributor (with each std
            uncertainty), optional value_low_db/value_high_db, warnings.
        """
        result: Dict[str, Any] = {
            "combined_standard_uncertainty_db": None,
            "expanded_uncertainty_db": None,
            "coverage_factor": coverage_factor,
            "n_contributors": 0,
            "per_contributor": [],
            "warnings": [],
        }
        if not contributors:
            result["warnings"].append("no_contributors")
            return result
        if coverage_factor <= 0:
            result["warnings"].append(f"invalid_coverage_factor: {coverage_factor}")
            return result

        sum_sq = 0.0
        per: List[Dict[str, Any]] = []
        for i, c in enumerate(contributors):
            try:
                name = str(c.get("name", f"contributor_{i}"))
                value = float(c.get("value_db"))
                dist = str(c.get("distribution", "rectangular")).lower()
            except (TypeError, ValueError) as exc:
                result["warnings"].append(f"contributor_{i}_bad: {exc}")
                continue
            divisor = _DIVISORS.get(dist)
            if divisor is None:
                result["warnings"].append(
                    f"{name}: unknown distribution {dist!r}; using rectangular"
                )
                divisor = _DIVISORS["rectangular"]
            std = abs(value) / divisor
            sum_sq += std * std
            per.append(
                {
                    "name": name,
                    "value_db": value,
                    "distribution": dist,
                    "standard_uncertainty_db": round(std, 4),
                }
            )

        if not per:
            result["warnings"].append("no_valid_contributors")
            return result

        u_c = math.sqrt(sum_sq)
        u_exp = coverage_factor * u_c
        result["combined_standard_uncertainty_db"] = round(u_c, 4)
        result["expanded_uncertainty_db"] = round(u_exp, 4)
        result["n_contributors"] = len(per)
        result["per_contributor"] = per
        if measured_value_db is not None:
            result["value_low_db"] = round(measured_value_db - u_exp, 4)
            result["value_high_db"] = round(measured_value_db + u_exp, 4)
        return result
