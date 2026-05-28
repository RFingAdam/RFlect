"""Tests for the v5.0.0 MCP tools: link budget, S11, group delay, MIMO,
antenna comparison, and active-cal generation."""

from __future__ import annotations

import math
import os
import sys

import numpy as np
import pytest

import matplotlib

matplotlib.use("Agg")

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "rflect-mcp"))
sys.path.insert(0, PROJECT_ROOT)

pytest.importorskip("mcp", reason="mcp package not installed")
from mcp.server.fastmcp import FastMCP


@pytest.fixture(scope="module")
def tools():
    from tools.propagation_tools import register_propagation_tools
    from tools.vna_tools import register_vna_tools
    from tools.mimo_tools import register_mimo_tools
    from tools.comparison_tools import register_comparison_tools
    from tools.calibration_tools import register_calibration_tools
    from tools.cal_drift_tools import register_cal_drift_tools

    mcp = FastMCP("rflect-new-tools-test")
    register_propagation_tools(mcp)
    register_vna_tools(mcp)
    register_mimo_tools(mcp)
    register_comparison_tools(mcp)
    register_calibration_tools(mcp)
    register_cal_drift_tools(mcp)
    reg = mcp._tool_manager._tools
    return {name: reg[name].fn for name in reg}


# ---------------------------------------------------------------------------
# cal_drift_report — never-raise contract (issue #9)
# ---------------------------------------------------------------------------


def test_cal_drift_report_unknown_format_returns_error_dict(tools, tmp_path):
    """Unknown format must return {'error': ...}, never raise (MCP contract)."""
    out = tools["cal_drift_report"](
        baseline_run_id="nope",
        current_run_id="nope",
        output_path=str(tmp_path / "r.md"),
        format="exe",
    )
    assert isinstance(out, dict)
    assert "error" in out and "unknown report format" in out["error"]


def test_cal_drift_report_bad_run_ids_returns_error_not_raise(tools, tmp_path):
    """A valid format but unknown run_ids returns an error dict, not an exception."""
    out = tools["cal_drift_report"](
        baseline_run_id="does-not-exist",
        current_run_id="also-not",
        output_path=str(tmp_path / "r.md"),
        format="markdown",
    )
    assert isinstance(out, dict)
    assert "error" in out  # compute_drift failed -> returned, not raised


# ---------------------------------------------------------------------------
# estimate_link_budget
# ---------------------------------------------------------------------------


def test_link_budget_basic_range(tools):
    out = tools["estimate_link_budget"](
        tx_power_dbm=20,
        rx_sensitivity_dbm=-90,
        tx_gain_dbi=2,
        rx_gain_dbi=2,
        freq_mhz=2450,
        path_loss_exp=2.0,
        model="friis",
    )
    assert out["warnings"] == []
    # PL_max = 20 + 2 + 2 - (-90) - 0 = 114 dB
    assert out["allowable_path_loss_db"] == pytest.approx(114.0)
    # Free-space at 2450 MHz over a clear link -> kilometers
    assert out["max_range_m"] > 1000


def test_link_budget_target_margin_and_fade(tools):
    out = tools["estimate_link_budget"](
        tx_power_dbm=10,
        rx_sensitivity_dbm=-85,
        tx_gain_dbi=0,
        rx_gain_dbi=0,
        freq_mhz=2450,
        path_loss_exp=3.0,
        model="log_distance",
        target_range_m=10.0,
        reliability_pct=99.0,
        fading="rayleigh",
    )
    assert "link_margin_db" in out
    assert "fade_margin_db" in out and out["fade_margin_db"] > 0
    # Adding a fade margin must not increase the reliable range vs raw max.
    assert out["reliable_range_m"] <= out["max_range_m"] + 1e-6


def test_link_budget_invalid_model(tools):
    out = tools["estimate_link_budget"](
        tx_power_dbm=20,
        rx_sensitivity_dbm=-90,
        tx_gain_dbi=2,
        rx_gain_dbi=2,
        freq_mhz=2450,
        model="teleport",
    )
    assert any("invalid_model" in w for w in out["warnings"])


# ---------------------------------------------------------------------------
# analyze_s11
# ---------------------------------------------------------------------------


def test_analyze_s11_finds_resonance_and_band(tools):
    f = np.linspace(2.0e9, 3.0e9, 201)
    # A resonance dip centered at 2.45 GHz reaching about -20 dB.
    s11 = -2.0 - 18.0 * np.exp(-(((f - 2.45e9) / 60e6) ** 2))
    out = tools["analyze_s11"](freq_hz=f.tolist(), s11_db=s11.tolist(), threshold_db=-10.0)
    assert out["warnings"] == []
    assert out["min_s11_db"] < -15
    assert out["resonance_freq_hz"] == pytest.approx(2.45e9, abs=10e6)
    assert out["bandwidth_hz"] > 0
    assert out["vswr_min"] is not None and out["vswr_min"] < 1.5


def test_analyze_s11_length_mismatch(tools):
    out = tools["analyze_s11"](freq_hz=[1, 2, 3], s11_db=[-5, -6])
    assert any("length_mismatch" in w for w in out["warnings"])


# ---------------------------------------------------------------------------
# analyze_group_delay
# ---------------------------------------------------------------------------


def test_group_delay_linear_phase_is_flat(tools):
    f = np.linspace(2.0e9, 3.0e9, 201)
    # Linear phase vs frequency -> constant group delay.
    tau = 5e-9  # 5 ns
    phase_deg = np.degrees(-2 * np.pi * f * tau)
    out = tools["analyze_group_delay"](freq_hz=f.tolist(), phase_deg=phase_deg.tolist())
    assert out["warnings"] == []
    assert out["mean_group_delay_ns"] == pytest.approx(5.0, abs=0.2)
    assert abs(out["group_delay_variation_ns"]) < 0.5


# ---------------------------------------------------------------------------
# analyze_mimo_diversity
# ---------------------------------------------------------------------------


def test_mimo_zero_ecc_max_diversity(tools):
    out = tools["analyze_mimo_diversity"](ecc=0.0, snr_db=15.0)
    assert out["diversity_gain_db"] == pytest.approx(10.0, abs=0.01)
    assert out["isolation_rating"] == "excellent"
    assert out["capacity_bps_hz"] is not None


def test_mimo_high_ecc_poor(tools):
    out = tools["analyze_mimo_diversity"](ecc=0.85, snr_db=15.0)
    assert out["isolation_rating"] == "poor"
    assert out["diversity_gain_db"] < 6.0


def test_mimo_capacity_curve(tools):
    out = tools["analyze_mimo_diversity"](ecc=0.2, snr_sweep_db=[0, 10, 20, 30])
    assert "capacity_curve" in out
    caps = [p["capacity_bps_hz"] for p in out["capacity_curve"]]
    assert caps == sorted(caps)  # capacity increases with SNR


# ---------------------------------------------------------------------------
# generate_active_cal (validation path — no real chamber files in CI)
# ---------------------------------------------------------------------------


def test_active_cal_missing_files(tools, tmp_path):
    out = tools["generate_active_cal"](
        power_measurement_file=str(tmp_path / "missing_p.txt"),
        gain_standard_file=str(tmp_path / "missing_g.txt"),
        hpol_file=str(tmp_path / "missing_h.txt"),
        vpol_file=str(tmp_path / "missing_v.txt"),
        freq_list=[2400.0, 2450.0],
    )
    assert out["output_path"] is None
    assert sum("file_not_found" in w for w in out["warnings"]) == 4


# ---------------------------------------------------------------------------
# compare_antennas
# ---------------------------------------------------------------------------


def _mock_passive(seed, gain_offset):
    from tools.import_tools import LoadedMeasurement

    n_theta, n_phi = 19, 37
    theta = np.linspace(0, 180, n_theta)
    phi = np.linspace(0, 360, n_phi)
    rng = np.random.default_rng(seed)
    total = rng.standard_normal((n_theta * n_phi, 3)) * 2 + gain_offset
    h = rng.standard_normal((n_theta * n_phi, 3)) * 2
    v = rng.standard_normal((n_theta * n_phi, 3)) * 2
    return LoadedMeasurement(
        file_path=f"mock_{seed}.txt",
        scan_type="passive",
        frequencies=[2400.0, 2450.0, 2500.0],
        data={"theta": theta, "phi": phi, "total_gain": total, "h_gain": h, "v_gain": v},
    )


def test_compare_antennas_ranks_and_deltas(tools):
    from tools.import_tools import _loaded_measurements, _measurements_lock

    with _measurements_lock:
        _loaded_measurements.clear()
        _loaded_measurements["AntLow"] = _mock_passive(1, gain_offset=0.0)
        _loaded_measurements["AntHigh"] = _mock_passive(2, gain_offset=4.0)
    try:
        out = tools["compare_antennas"](measurement_names=["AntLow", "AntHigh"], reference="AntLow")
        assert out["warnings"] == [] or all("mixed_scan_types" not in w for w in out["warnings"])
        assert out["metric"] == "peak_gain" and out["unit"] == "dBi"
        assert set(out["measurements"]) == {"AntLow", "AntHigh"}
        assert len(out["rows"]) == 3  # three frequencies
        # AntHigh has +4 dB offset -> should win overall.
        assert out["best_overall"] == "AntHigh"
        # Deltas are vs AntLow (reference); AntLow's own delta is 0.
        for row in out["rows"]:
            assert row["deltas_vs_ref"]["AntLow"] == 0.0
    finally:
        with _measurements_lock:
            _loaded_measurements.clear()


def test_compare_antennas_needs_two(tools):
    from tools.import_tools import _loaded_measurements, _measurements_lock

    with _measurements_lock:
        _loaded_measurements.clear()
        _loaded_measurements["OnlyOne"] = _mock_passive(3, gain_offset=1.0)
    try:
        out = tools["compare_antennas"](measurement_names=["OnlyOne"])
        assert any("need_at_least_2" in w for w in out["warnings"])
    finally:
        with _measurements_lock:
            _loaded_measurements.clear()
