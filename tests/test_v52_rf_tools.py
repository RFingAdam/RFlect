"""Tests for the v5.2 RF MCP tools: compliance (#28/#29), uncertainty (#30),
multiport S-params (#31), statistical averaging (#32)."""

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
    from tools.compliance_tools import register_compliance_tools
    from tools.uncertainty_tools import register_uncertainty_tools
    from tools.sparam_tools import register_sparam_tools
    from tools.statistics_tools import register_statistics_tools

    mcp = FastMCP("rflect-v52-test")
    register_compliance_tools(mcp)
    register_uncertainty_tools(mcp)
    register_sparam_tools(mcp)
    register_statistics_tools(mcp)
    reg = mcp._tool_manager._tools
    return {name: reg[name].fn for name in reg}


# ----------------------------- #28 spec compliance -----------------------------


def test_spec_compliance_pass_and_fail(tools):
    out = tools["check_spec_compliance"](
        measured={"min_gain_dbi": 3.0, "max_vswr": 1.8, "max_ecc": 0.6},
        limits={"min_gain_dbi": 2.0, "max_vswr": 2.0, "max_ecc": 0.5},
    )
    verdicts = {c["metric"]: c["verdict"] for c in out["checks"]}
    assert verdicts["min_gain_dbi"] == "PASS"  # 3.0 >= 2.0
    assert verdicts["max_vswr"] == "PASS"  # 1.8 <= 2.0
    assert verdicts["max_ecc"] == "FAIL"  # 0.6 > 0.5
    assert out["overall"] == "FAIL"


def test_spec_compliance_unknown_key_warns(tools):
    out = tools["check_spec_compliance"](measured={"foo": 1}, limits={"foo": 1})
    assert any("unknown_limit_key" in w for w in out["warnings"])
    assert out["overall"] == "NO_CHECKS"


# ----------------------------- #29 regulatory EIRP -----------------------------


def test_regulatory_eirp_24ghz_fail_over_etsi(tools):
    out = tools["check_regulatory_eirp"](eirp_dbm=25.0, freq_mhz=2450.0)
    assert out["band"] == "2.4GHz"
    by_rule = {c["ruleset"]: c["verdict"] for c in out["checks"]}
    assert by_rule["ETSI_EN300328"] == "FAIL"  # 25 > 20
    assert by_rule["FCC_15.247"] == "PASS"  # 25 <= 36
    assert out["overall"] == "FAIL"


def test_regulatory_eirp_unknown_band(tools):
    out = tools["check_regulatory_eirp"](eirp_dbm=10.0, freq_mhz=900.0)
    assert out["band"] is None
    assert any("no_builtin_band" in w for w in out["warnings"])


# ----------------------------- #30 uncertainty -----------------------------


def test_uncertainty_rss_and_expanded(tools):
    # Two normal 1-sigma contributors of 0.3 and 0.4 dB -> RSS = 0.5 dB.
    out = tools["compute_uncertainty_budget"](
        contributors=[
            {"name": "mismatch", "value_db": 0.3, "distribution": "normal"},
            {"name": "cable", "value_db": 0.4, "distribution": "normal"},
        ],
        coverage_factor=2.0,
        measured_value_db=12.0,
    )
    assert out["combined_standard_uncertainty_db"] == pytest.approx(0.5, abs=1e-3)
    assert out["expanded_uncertainty_db"] == pytest.approx(1.0, abs=1e-3)
    assert out["value_low_db"] == pytest.approx(11.0, abs=1e-3)
    assert out["value_high_db"] == pytest.approx(13.0, abs=1e-3)


def test_uncertainty_rectangular_divisor(tools):
    # Rectangular half-width 1.0 dB -> std = 1/sqrt(3).
    out = tools["compute_uncertainty_budget"](
        contributors=[{"name": "quant", "value_db": 1.0, "distribution": "rectangular"}],
    )
    assert out["combined_standard_uncertainty_db"] == pytest.approx(1.0 / math.sqrt(3), abs=1e-3)


def test_uncertainty_empty(tools):
    out = tools["compute_uncertainty_budget"](contributors=[])
    assert any("no_contributors" in w for w in out["warnings"])


# ----------------------------- #31 multiport Touchstone -----------------------------


def _write_s2p(path):
    # Minimal 2-port .s2p in MA format: a -15 dB S11 dip, ~ -1 dB S21.
    lines = ["! test 2-port", "# GHz S MA R 50"]
    for f in (2.0, 2.45, 3.0):
        s11_mag = 0.5 if abs(f - 2.45) > 0.3 else 0.178  # -15 dB at center
        # freq S11(mag,ang) S21(mag,ang) S12 S22
        lines.append(f"{f} {s11_mag} 0 0.89 0 0.89 0 0.5 0")
    path.write_text("\n".join(lines) + "\n")


def test_multiport_s2p_per_port(tools, tmp_path):
    p = tmp_path / "dut.s2p"
    _write_s2p(p)
    out = tools["analyze_multiport_touchstone"](str(p))
    assert out["warnings"] == []
    assert out["n_ports"] == 2
    assert out["n_points"] == 3
    assert len(out["per_port"]) == 2
    # Port 1 sees a -15 dB dip somewhere -> matched at -10 dB threshold.
    p1 = next(pp for pp in out["per_port"] if pp["port"] == 1)
    assert p1["matched"] is True
    assert p1["min_return_loss_db"] < -10.0


def test_multiport_missing_file(tools, tmp_path):
    out = tools["analyze_multiport_touchstone"](str(tmp_path / "nope.s4p"))
    assert any("file_not_found" in w for w in out["warnings"])


def test_multiport_4port_mixed_mode(tools, tmp_path):
    # Synthetic 4-port: identity-ish (matched, low coupling) just to exercise
    # the mixed-mode path end-to-end.
    p = tmp_path / "diff.s4p"
    lines = ["! 4-port", "# GHz S MA R 50"]
    for f in (2.4, 2.5):
        vals = []
        for r in range(4):
            for c in range(4):
                mag = 0.05 if r != c else 0.1  # low everything
                vals += [f"{mag}", "0"]
        lines.append(f"{f} " + " ".join(vals))
    p.write_text("\n".join(lines) + "\n")
    out = tools["analyze_multiport_touchstone"](str(p))
    assert out["n_ports"] == 4
    assert "mixed_mode" in out
    assert "sdd11_min_db" in out["mixed_mode"]


# ----------------------------- #32 statistical averaging -----------------------------


def test_average_patterns_mean_and_repeatability(tools):
    base = [[0.0, -3.0], [-6.0, -10.0]]
    # Two repeats: identical except a +0.4/-0.4 dB jitter at the peak cell.
    p1 = [[0.4, -3.0], [-6.0, -10.0]]
    p2 = [[-0.4, -3.0], [-6.0, -10.0]]
    out = tools["average_patterns"]([p1, p2])
    assert out["n_repeats"] == 2
    assert out["shape"] == [2, 2]
    # Mean at the peak cell is 0.0; peak-to-peak across repeats there is 0.8 dB.
    assert out["mean_pattern_db"][0][0] == pytest.approx(0.0, abs=1e-6)
    assert out["peak_to_peak_at_max_db"] == pytest.approx(0.8, abs=1e-6)


def test_average_patterns_needs_two(tools):
    out = tools["average_patterns"]([[[0.0]]])
    assert any("need_at_least_2" in w for w in out["warnings"])


# ----------------------------- #34/#7 n-antenna summary -----------------------------


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


def test_summarize_antennas_minmaxmean():
    from mcp.server.fastmcp import FastMCP
    from tools.comparison_tools import register_comparison_tools
    from tools.import_tools import _loaded_measurements, _measurements_lock

    mcp = FastMCP("summ-test")
    register_comparison_tools(mcp)
    fn = mcp._tool_manager._tools["summarize_antennas"].fn

    with _measurements_lock:
        _loaded_measurements.clear()
        _loaded_measurements["AntA"] = _mock_passive(1, 0.0)
        _loaded_measurements["AntB"] = _mock_passive(2, 4.0)
    try:
        out = fn(measurement_names=["AntA", "AntB"])
        assert out["metric"] == "peak_gain" and out["unit"] == "dBi"
        rows = {r["name"]: r for r in out["rows"]}
        assert set(rows) == {"AntA", "AntB"}
        for r in rows.values():
            assert r["n_freqs"] == 3
            assert r["min"] <= r["mean"] <= r["max"]
            assert r["spread"] == pytest.approx(r["max"] - r["min"], abs=0.01)
        # AntB has +4 dB offset -> higher mean peak gain.
        assert rows["AntB"]["mean"] > rows["AntA"]["mean"]
    finally:
        with _measurements_lock:
            _loaded_measurements.clear()


def test_summarize_antennas_no_data():
    from mcp.server.fastmcp import FastMCP
    from tools.comparison_tools import register_comparison_tools
    from tools.import_tools import _loaded_measurements, _measurements_lock

    mcp = FastMCP("summ-empty")
    register_comparison_tools(mcp)
    fn = mcp._tool_manager._tools["summarize_antennas"].fn
    with _measurements_lock:
        _loaded_measurements.clear()
    out = fn()
    assert any("no_data_loaded" in w for w in out["warnings"])
