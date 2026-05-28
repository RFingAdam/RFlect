"""Tests for instrument/positioner automation (v6.0 #44/#45) via mock backends."""

from __future__ import annotations

import os
import sys

import numpy as np
import pytest

from plot_antenna.instruments import (
    MockVna,
    MockPositioner,
    VnaController,
    PositionerController,
)


# ----------------------------- #44 VNA control -----------------------------


def test_vna_controller_reads_trace_via_mock():
    vna = VnaController(MockVna(resonance_hz=2.45e9))
    idn = vna.connect()
    assert "MockVNA" in idn
    vna.configure_sweep(2.0e9, 3.0e9, 101)
    trace = vna.read_s_parameter("S11")
    assert len(trace["freq_hz"]) == 101
    assert len(trace["s_real"]) == 101 and len(trace["s_imag"]) == 101
    # The mock resonance dips near 2.45 GHz -> min |S| there.
    mag = np.hypot(trace["s_real"], trace["s_imag"])
    f = np.array(trace["freq_hz"])
    assert abs(f[int(np.argmin(mag))] - 2.45e9) < 60e6
    vna.disconnect()


def test_vna_configure_sweep_emits_scpi():
    backend = MockVna()
    VnaController(backend).configure_sweep(2.4e9, 2.5e9, 51)
    joined = " ".join(backend.writes).upper()
    assert "FREQ:STAR" in joined and "FREQ:STOP" in joined and "SWE:POIN" in joined


# ----------------------------- #45 positioner -----------------------------


def test_positioner_home_and_goto():
    ctrl = PositionerController(MockPositioner())
    assert ctrl.home() == (0.0, 0.0)
    assert ctrl.goto(90.0, 45.0) == (90.0, 45.0)


def test_positioner_scan_grid_visits_all_points():
    ctrl = PositionerController(MockPositioner())
    ctrl.home()
    visited = ctrl.scan_grid([0, 90, 180], [0, 120, 240])
    assert len(visited) == 9
    assert visited[0] == (0.0, 0.0)
    assert (90.0, 120.0) in visited


# ----------------------------- MCP tool layer -----------------------------


def test_mcp_instrument_tools():
    pytest.importorskip("mcp", reason="mcp package not installed")
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(__file__)), "rflect-mcp"))
    from mcp.server.fastmcp import FastMCP
    from tools.instrument_tools import register_instrument_tools

    mcp = FastMCP("instr-test")
    register_instrument_tools(mcp)
    reg = mcp._tool_manager._tools

    vt = reg["vna_read_trace"].fn(start_hz=2.0e9, stop_hz=3.0e9, points=51)
    assert vt["backend"] == "mock"
    assert len(vt["freq_hz"]) == 51

    pg = reg["positioner_scan_grid"].fn(theta_steps=[0, 90], phi_steps=[0, 180])
    assert pg["n_points"] == 4
    assert pg["visited"][0] == [0.0, 0.0]
