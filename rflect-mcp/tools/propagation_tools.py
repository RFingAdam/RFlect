"""
Propagation / link-budget MCP tools for RFlect.

Thin wrappers over the deterministic models in plot_antenna.calculations
(Friis / log-distance / ITU indoor path loss, Rayleigh/Rician fade margin).
No LLM, no network. Each tool returns a structured dict and never raises;
failures surface in a `warnings` list.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

from plot_antenna.calculations import (
    free_space_path_loss,
    friis_range_estimate,
    log_distance_path_loss,
    itu_indoor_path_loss,
    fade_margin_for_reliability,
)


def register_propagation_tools(mcp):
    """Register propagation / link-budget tools with the MCP server."""

    @mcp.tool()
    def estimate_link_budget(
        tx_power_dbm: float,
        rx_sensitivity_dbm: float,
        tx_gain_dbi: float,
        rx_gain_dbi: float,
        freq_mhz: float,
        path_loss_exp: float = 2.0,
        misc_loss_db: float = 0.0,
        model: str = "log_distance",
        target_range_m: Optional[float] = None,
        reliability_pct: Optional[float] = None,
        fading: str = "rayleigh",
        rician_k: float = 10.0,
        n_floors: int = 0,
        environment: str = "office",
    ) -> Dict[str, Any]:
        """
        Estimate a wireless link budget and maximum range from measured antenna gains.

        Combines a transmit/receive gain pair with a path-loss model to solve for
        the maximum range at which the received power still meets the receiver
        sensitivity. Pair this with measured TRP/gain (e.g. use the peak gain from
        get_gain_statistics as tx_gain_dbi) for a measurement-grounded estimate.

        Args:
            tx_power_dbm: Transmit power (dBm).
            rx_sensitivity_dbm: Receiver sensitivity (dBm, typically negative).
            tx_gain_dbi: Transmit antenna gain (dBi) — e.g. measured peak gain.
            rx_gain_dbi: Receive antenna gain (dBi).
            freq_mhz: Frequency (MHz).
            path_loss_exp: Path-loss exponent n (2.0 free space, ~3.0 indoor).
            misc_loss_db: Additional fixed system losses (dB).
            model: 'friis' | 'log_distance' | 'itu_indoor'. Selects how the
                allowable path loss is converted to range. 'friis' and
                'log_distance' use path_loss_exp; 'itu_indoor' uses n_floors +
                environment.
            target_range_m: If given, also report the link margin at this range.
            reliability_pct: If given (e.g. 99.0), report the fade margin needed
                for this link reliability and the de-rated range after applying it.
            fading: 'rayleigh' | 'rician' (used only when reliability_pct given).
            rician_k: Rician K-factor (linear), used when fading='rician'.
            n_floors: Floors between Tx/Rx (model='itu_indoor').
            environment: 'office'|'residential'|'commercial' (model='itu_indoor').

        Returns:
            Dict: allowable_path_loss_db, fspl_at_1m_db, max_range_m, model_used,
            plus optional link_margin_db (with target_range_m) and
            fade_margin_db / reliable_range_m (with reliability_pct). Never raises;
            failures populate `warnings`.
        """
        result: Dict[str, Any] = {
            "model_used": model,
            "allowable_path_loss_db": None,
            "fspl_at_1m_db": None,
            "max_range_m": None,
            "warnings": [],
        }

        if model not in ("friis", "log_distance", "itu_indoor"):
            result["warnings"].append(
                f"invalid_model: {model!r}; expected friis|log_distance|itu_indoor"
            )
            return result
        if freq_mhz <= 0:
            result["warnings"].append(f"invalid_freq_mhz: {freq_mhz}")
            return result

        # Allowable path loss: PL_max = Pt + Gt + Gr - Pr - L
        pl_max = tx_power_dbm + tx_gain_dbi + rx_gain_dbi - rx_sensitivity_dbm - misc_loss_db
        result["allowable_path_loss_db"] = float(pl_max)
        result["fspl_at_1m_db"] = float(free_space_path_loss(freq_mhz, 1.0))

        try:
            n = 2.0 if model == "friis" else path_loss_exp
            if model == "itu_indoor":
                # Solve range numerically for the ITU model by scanning distance.
                import numpy as np

                dists = np.logspace(-1, 3, 4000)  # 0.1 m .. 1000 m
                pl = itu_indoor_path_loss(
                    freq_mhz, dists, n_floors=n_floors, environment=environment
                )
                ok = np.where(pl <= pl_max)[0]
                result["max_range_m"] = float(dists[ok[-1]]) if len(ok) else 0.0
            else:
                result["max_range_m"] = float(
                    friis_range_estimate(
                        tx_power_dbm,
                        rx_sensitivity_dbm,
                        tx_gain_dbi,
                        rx_gain_dbi,
                        freq_mhz,
                        path_loss_exp=n,
                        misc_loss_db=misc_loss_db,
                    )
                )
        except Exception as exc:
            result["warnings"].append(f"range_solve_failed: {exc}")
            return result

        if target_range_m is not None and target_range_m > 0:
            try:
                if model == "itu_indoor":
                    pl_at = float(
                        itu_indoor_path_loss(
                            freq_mhz, target_range_m, n_floors=n_floors, environment=environment
                        )
                    )
                else:
                    pl_at = float(
                        log_distance_path_loss(
                            freq_mhz, target_range_m, n=(2.0 if model == "friis" else path_loss_exp)
                        )
                    )
                result["path_loss_at_target_db"] = pl_at
                result["link_margin_db"] = float(pl_max - pl_at)
            except Exception as exc:
                result["warnings"].append(f"target_margin_failed: {exc}")

        if reliability_pct is not None:
            try:
                fm = float(fade_margin_for_reliability(reliability_pct, fading=fading, K=rician_k))
                result["fade_margin_db"] = fm
                # De-rate range: subtract fade margin from allowable PL, re-solve.
                pl_eff = pl_max - fm
                n_eff = 2.0 if model == "friis" else path_loss_exp
                fspl_d0 = float(free_space_path_loss(freq_mhz, 1.0))
                if n_eff > 0:
                    exponent = (pl_eff - fspl_d0) / (10.0 * n_eff)
                    result["reliable_range_m"] = float(10.0**exponent)
            except Exception as exc:
                result["warnings"].append(f"fade_margin_failed: {exc}")

        return result
