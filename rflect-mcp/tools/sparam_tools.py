"""
Multiport S-parameter MCP tools for RFlect (#31).

Reads N-port Touchstone files (.s2p/.s3p/.s4p ...) and reports per-port return
loss / impedance bandwidth; for 4-ports it can convert single-ended S to
mixed-mode (Sdd/Scc/Sdc/Scd) for differential / MIMO-feed analysis.

Self-contained Touchstone parser (no external skrf dependency). Deterministic;
returns a structured dict and never raises; failures populate `warnings`.
"""

from __future__ import annotations

import os
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

_FREQ_MULT = {"hz": 1.0, "khz": 1e3, "mhz": 1e6, "ghz": 1e9}


def _parse_touchstone(path: str) -> Tuple[np.ndarray, np.ndarray, int, str]:
    """Parse an N-port Touchstone file.

    Returns (freq_hz, S, n_ports, fmt) where S has shape (n_freq, n, n) complex.
    Port count is inferred from the .sNp extension (falls back to data width).
    """
    ext = os.path.splitext(path)[1].lower()
    n_ports = None
    if len(ext) >= 4 and ext.startswith(".s") and ext.endswith("p"):
        try:
            n_ports = int(ext[2:-1])
        except ValueError:
            n_ports = None

    freq_unit_mult = 1e9  # default GHz per Touchstone spec
    fmt = "ma"
    tokens: List[float] = []
    with open(path, "r", encoding="utf-8-sig", errors="replace") as fh:
        for raw in fh:
            line = raw.strip()
            if not line or line.startswith("!"):
                continue
            if line.startswith("#"):
                parts = line[1:].lower().split()
                for p in parts:
                    if p in _FREQ_MULT:
                        freq_unit_mult = _FREQ_MULT[p]
                    if p in ("ri", "ma", "db"):
                        fmt = p
                continue
            # strip inline comments
            if "!" in line:
                line = line.split("!", 1)[0]
            tokens.extend(float(x) for x in line.replace(",", " ").split())

    if not tokens:
        raise ValueError("no numeric data in Touchstone file")

    # Infer n_ports from row width if not from extension.
    # A record is: 1 freq + 2*n^2 values.
    if n_ports is None:
        for n in (1, 2, 3, 4, 8):
            if len(tokens) % (1 + 2 * n * n) == 0:
                n_ports = n
                break
        if n_ports is None:
            raise ValueError("could not infer port count from data width")

    rec_len = 1 + 2 * n_ports * n_ports
    if len(tokens) % rec_len != 0:
        raise ValueError(
            f"data length {len(tokens)} not a multiple of record length {rec_len} "
            f"for {n_ports}-port"
        )
    n_freq = len(tokens) // rec_len
    arr = np.asarray(tokens, dtype=float).reshape(n_freq, rec_len)

    freq_hz = arr[:, 0] * freq_unit_mult
    S = np.empty((n_freq, n_ports, n_ports), dtype=complex)
    pairs = arr[:, 1:].reshape(n_freq, n_ports * n_ports, 2)

    if fmt == "ri":
        vals = pairs[:, :, 0] + 1j * pairs[:, :, 1]
    elif fmt == "db":
        mag = 10 ** (pairs[:, :, 0] / 20.0)
        vals = mag * np.exp(1j * np.deg2rad(pairs[:, :, 1]))
    else:  # ma
        vals = pairs[:, :, 0] * np.exp(1j * np.deg2rad(pairs[:, :, 1]))

    # Touchstone 2-port orders entries S11 S21 S12 S22; N!=2 is row-major.
    if n_ports == 2:
        for k in range(n_freq):
            S[k, 0, 0] = vals[k, 0]
            S[k, 1, 0] = vals[k, 1]
            S[k, 0, 1] = vals[k, 2]
            S[k, 1, 1] = vals[k, 3]
    else:
        S = vals.reshape(n_freq, n_ports, n_ports)
    return freq_hz, S, n_ports, fmt


def _mixed_mode_4port(S: np.ndarray) -> Dict[str, np.ndarray]:
    """Single-ended 4-port -> mixed-mode blocks (Sdd, Scc, Sdc, Scd).

    Port mapping: differential pair 1 = ports (1,3), pair 2 = ports (2,4)
    (0-based 0,2 and 1,3) — the common balanced convention. Each block is 2x2.
    """
    # M S M^-1 with the standard mixed-mode transform; do it blockwise.
    # Indices (0-based): pair1 = (0,2), pair2 = (1,3).
    i = [0, 2, 1, 3]  # reorder to [p1+, p1-, p2+, p2-]
    Sr = S[np.ix_(range(S.shape[0]), i, i)]
    # Transform matrix M (per IEEE mixed-mode): rows d1,d2,c1,c2.
    Mt = (1 / np.sqrt(2)) * np.array(
        [
            [1, -1, 0, 0],
            [0, 0, 1, -1],
            [1, 1, 0, 0],
            [0, 0, 1, 1],
        ],
        dtype=float,
    )
    Minv = np.linalg.inv(Mt)
    Smm = Mt @ Sr @ Minv  # broadcast over freq axis
    return {
        "Sdd": Smm[:, 0:2, 0:2],
        "Sdc": Smm[:, 0:2, 2:4],
        "Scd": Smm[:, 2:4, 0:2],
        "Scc": Smm[:, 2:4, 2:4],
    }


def register_sparam_tools(mcp):
    """Register multiport S-parameter tools with the MCP server."""

    @mcp.tool()
    def analyze_multiport_touchstone(
        path: str,
        return_loss_threshold_db: float = -10.0,
    ) -> Dict[str, Any]:
        """
        Read an N-port Touchstone file and summarize per-port match (+ mixed-mode
        for 4-ports).

        Supports .s2p/.s3p/.s4p (and higher), RI/MA/DB formats, any frequency
        unit. Reports per-port return loss (min Sii in dB, worst across band) and,
        for a 4-port, the differential return loss Sdd11 and differential
        insertion loss Sdd21 (pairs = ports (1,3) and (2,4)).

        Args:
            path: Path to the Touchstone file.
            return_loss_threshold_db: threshold for the "matched" band per port.

        Returns:
            Dict: n_ports, n_points, freq_start_hz, freq_stop_hz, per_port
            [{port, min_return_loss_db}], mixed_mode (for 4-port:
            {sdd11_min_db, sdd21_mean_db}), warnings. Never raises.
        """
        result: Dict[str, Any] = {
            "n_ports": None,
            "n_points": 0,
            "freq_start_hz": None,
            "freq_stop_hz": None,
            "per_port": [],
            "warnings": [],
        }
        if not os.path.isfile(path):
            result["warnings"].append(f"file_not_found: {path}")
            return result
        try:
            freq, S, n_ports, fmt = _parse_touchstone(path)
        except Exception as exc:
            result["warnings"].append(f"parse_failed: {exc}")
            return result

        result["n_ports"] = n_ports
        result["n_points"] = int(freq.size)
        result["freq_start_hz"] = float(freq[0])
        result["freq_stop_hz"] = float(freq[-1])

        for p in range(n_ports):
            sii_db = 20 * np.log10(np.maximum(np.abs(S[:, p, p]), 1e-12))
            result["per_port"].append(
                {
                    "port": p + 1,
                    "min_return_loss_db": float(np.min(sii_db)),  # worst (closest to 0)
                    "matched": bool(np.any(sii_db <= return_loss_threshold_db)),
                }
            )

        if n_ports == 4:
            try:
                mm = _mixed_mode_4port(S)
                sdd11_db = 20 * np.log10(np.maximum(np.abs(mm["Sdd"][:, 0, 0]), 1e-12))
                sdd21_db = 20 * np.log10(np.maximum(np.abs(mm["Sdd"][:, 1, 0]), 1e-12))
                result["mixed_mode"] = {
                    "sdd11_min_db": float(np.min(sdd11_db)),
                    "sdd21_mean_db": float(np.mean(sdd21_db)),
                    "note": "differential pairs = ports (1,3) and (2,4)",
                }
            except Exception as exc:
                result["warnings"].append(f"mixed_mode_failed: {exc}")
        return result
