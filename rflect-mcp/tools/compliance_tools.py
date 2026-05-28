"""
Compliance / spec-limit MCP tools for RFlect.

- check_spec_compliance: generic PASS/FAIL of measured metrics against
  user-supplied limit lines (#28).
- check_regulatory_eirp: measured EIRP vs built-in FCC Part 15 / ETSI
  reference EIRP limits for common unlicensed bands (#29).

Pure / deterministic; no LLM, no network. Each tool returns a structured dict
and never raises; failures populate `warnings`.

NOTE: the regulatory values are convenience references for common configs and
are NOT a substitute for the current rule text. Always confirm against the
live FCC/ETSI regulation for your exact device class and configuration.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

# Each limit: how the measured value must relate to the threshold.
# ("max", x): measured must be <= x. ("min", x): measured must be >= x.
_SPEC_DIRECTION = {
    "min_gain_dbi": "min",
    "min_trp_dbm": "min",
    "min_efficiency_pct": "min",
    "max_vswr": "max",
    "max_return_loss_db": "max",  # e.g. -10 dB: measured S11 must be <= -10
    "max_sll_db": "max",  # sidelobe level relative to main (dB)
    "max_axial_ratio_db": "max",
    "max_ecc": "max",
    "max_eirp_dbm": "max",
}

# Built-in reference EIRP ceilings (dBm) for common unlicensed bands.
# band key -> {ruleset -> (eirp_dbm_limit, note)}
_EIRP_LIMITS = {
    "2.4GHz": {
        "FCC_15.247": (36.0, "FCC 15.247 DTS/FHSS point-to-multipoint EIRP ceiling"),
        "ETSI_EN300328": (20.0, "ETSI EN 300 328 2.4 GHz EIRP ceiling (100 mW)"),
    },
    "5GHz_UNII1": {  # 5150-5250
        "FCC_15.407": (30.0, "FCC 15.407 U-NII-1 EIRP (250 mW) for std power"),
        "ETSI_EN301893": (23.0, "ETSI EN 301 893 U-NII-1 EIRP ceiling (200 mW)"),
    },
    "5GHz_UNII3": {  # 5725-5850
        "FCC_15.407": (36.0, "FCC 15.407 U-NII-3 EIRP ceiling (4 W)"),
        "ETSI_EN302502": (33.0, "ETSI EN 302 502 5.8 GHz EIRP ceiling (2 W)"),
    },
}


def _freq_to_band(freq_mhz: float) -> Optional[str]:
    if 2400.0 <= freq_mhz <= 2500.0:
        return "2.4GHz"
    if 5150.0 <= freq_mhz <= 5250.0:
        return "5GHz_UNII1"
    if 5725.0 <= freq_mhz <= 5850.0:
        return "5GHz_UNII3"
    return None


def register_compliance_tools(mcp):
    """Register compliance / spec-limit tools with the MCP server."""

    @mcp.tool()
    def check_spec_compliance(
        measured: Dict[str, float],
        limits: Dict[str, float],
    ) -> Dict[str, Any]:
        """
        PASS/FAIL measured antenna metrics against spec limit lines.

        Supply measured values and the spec limits using matching keys; each
        recognized key has a fixed direction (min vs max). The tool reports the
        margin and a PASS/FAIL per metric plus an overall verdict.

        Recognized keys (direction):
          min_gain_dbi, min_trp_dbm, min_efficiency_pct (measured must be >=);
          max_vswr, max_return_loss_db, max_sll_db, max_axial_ratio_db,
          max_ecc, max_eirp_dbm (measured must be <=).

        Args:
            measured: {key: measured_value}.
            limits: {key: limit_value} for the keys you want checked.

        Returns:
            Dict: overall ("PASS"|"FAIL"|"NO_CHECKS"), checks [{metric, measured,
            limit, direction, margin, verdict}], warnings.
        """
        result: Dict[str, Any] = {"overall": "NO_CHECKS", "checks": [], "warnings": []}
        checks: List[Dict[str, Any]] = []
        any_fail = False
        any_checked = False

        for key, limit in limits.items():
            direction = _SPEC_DIRECTION.get(key)
            if direction is None:
                result["warnings"].append(f"unknown_limit_key: {key}")
                continue
            if key not in measured:
                result["warnings"].append(f"no_measured_value_for: {key}")
                continue
            m = float(measured[key])
            lim = float(limit)
            if direction == "min":
                margin = m - lim  # positive = pass margin
                verdict = "PASS" if m >= lim else "FAIL"
            else:  # max
                margin = lim - m  # positive = pass margin
                verdict = "PASS" if m <= lim else "FAIL"
            any_checked = True
            if verdict == "FAIL":
                any_fail = True
            checks.append(
                {
                    "metric": key,
                    "measured": m,
                    "limit": lim,
                    "direction": direction,
                    "margin": round(margin, 3),
                    "verdict": verdict,
                }
            )

        result["checks"] = checks
        if any_checked:
            result["overall"] = "FAIL" if any_fail else "PASS"
        return result

    @mcp.tool()
    def check_regulatory_eirp(
        eirp_dbm: float,
        freq_mhz: float,
        ruleset: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Check a measured EIRP against built-in FCC/ETSI reference limits.

        Maps the frequency to a common unlicensed band (2.4 GHz, 5 GHz U-NII-1,
        U-NII-3) and compares the measured EIRP to the reference ceiling(s).

        Args:
            eirp_dbm: Measured peak EIRP (dBm). Often max_gain_dbi + conducted
                power, or the chamber peak EIRP.
            freq_mhz: Frequency (MHz) used to select the band.
            ruleset: Optional specific ruleset key (e.g. "FCC_15.247"); default
                checks all rulesets known for the band.

        Returns:
            Dict: band, checks [{ruleset, limit_dbm, measured_dbm, margin_db,
            verdict, note}], overall, warnings. Reference values only — verify
            against the live regulation for your device class.
        """
        result: Dict[str, Any] = {
            "band": None,
            "checks": [],
            "overall": "NO_CHECKS",
            "warnings": [],
        }
        band = _freq_to_band(freq_mhz)
        if band is None:
            result["warnings"].append(
                f"no_builtin_band_for_freq_mhz: {freq_mhz} (supported: 2.4 GHz, "
                f"5 GHz U-NII-1 5150-5250, U-NII-3 5725-5850)"
            )
            return result
        result["band"] = band

        rulesets = _EIRP_LIMITS[band]
        items = {ruleset: rulesets[ruleset]} if ruleset and ruleset in rulesets else rulesets
        if ruleset and ruleset not in rulesets:
            result["warnings"].append(
                f"unknown_ruleset_for_band: {ruleset}; have {sorted(rulesets)}"
            )
            return result

        any_fail = False
        for rs, (limit, note) in items.items():
            margin = limit - eirp_dbm
            verdict = "PASS" if eirp_dbm <= limit else "FAIL"
            if verdict == "FAIL":
                any_fail = True
            result["checks"].append(
                {
                    "ruleset": rs,
                    "limit_dbm": limit,
                    "measured_dbm": eirp_dbm,
                    "margin_db": round(margin, 2),
                    "verdict": verdict,
                    "note": note,
                }
            )
        result["overall"] = "FAIL" if any_fail else "PASS"
        result["warnings"].append(
            "reference_limits_only: confirm against current FCC/ETSI rule text "
            "for your exact device class and configuration"
        )
        return result
