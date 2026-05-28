"""Tests for the ``analyze_iperf_angle_sweep`` MCP tool."""

from __future__ import annotations

import csv
import json
import os
import sys
from pathlib import Path

import pytest

import matplotlib
matplotlib.use("Agg")

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MCP_DIR = os.path.join(PROJECT_ROOT, "rflect-mcp")
sys.path.insert(0, MCP_DIR)
sys.path.insert(0, PROJECT_ROOT)

pytest.importorskip("mcp", reason="mcp package not installed")
from mcp.server.fastmcp import FastMCP


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def mcp_server():
    from tools.iperf_angle_tools import register_iperf_angle_tools

    mcp = FastMCP("rflect-iperf-angle-test")
    register_iperf_angle_tools(mcp)
    return mcp


@pytest.fixture
def analyze(mcp_server):
    return mcp_server._tool_manager._tools["analyze_iperf_angle_sweep"].fn


def _write_session(
    session_dir: Path,
    *,
    cells: dict,
) -> None:
    """Write a minimal session.json with a list of ``wifi_only`` runs.

    ``cells`` is ``{(channel, mode, angle_deg) -> overall_mbps}``.
    """
    session_dir.mkdir(parents=True, exist_ok=True)
    runs = []
    for (ch, mode, angle), mbps in cells.items():
        runs.append({
            "timestamp_utc": "2026-01-01T00:00:00+00:00",
            "kind": "wifi_only",
            "wifi_channel": ch,
            "iperf_mode": mode,
            "angle_deg": angle,
            "aggregate": {"overall_mbps": mbps,
                          "retransmits": 0,
                          "loss_percent": None},
        })
    manifest = {
        "session_name": session_dir.name,
        "created_utc": "2026-01-01T00:00:00+00:00",
        "runs": runs,
        "wifi_channels_tested": sorted({ch for (ch, _, _) in cells}),
        "iperf_modes": sorted({mode for (_, mode, _) in cells}),
        "angles_tested": sorted({a for (_, _, a) in cells}),
    }
    (session_dir / "session.json").write_text(json.dumps(manifest, indent=2))


def _flat_cells(channels, modes, angles, *, base_mbps, gain_at_angle=None):
    """Build {(ch, mode, angle) -> mbps} with optional per-angle gain."""
    out = {}
    for ch in channels:
        for mode in modes:
            for a in angles:
                gain = 0.0 if gain_at_angle is None else gain_at_angle(a)
                out[(ch, mode, a)] = float(base_mbps + gain)
    return out


# ---------------------------------------------------------------------------
# Happy-path delta computation
# ---------------------------------------------------------------------------


def test_per_angle_delta_and_summary(analyze, tmp_path):
    angles = [0.0, 90.0, 180.0, 270.0]
    channels = [6]
    modes = ["tcp_up"]

    installed_cells = _flat_cells(channels, modes, angles,
                                  base_mbps=100.0,
                                  gain_at_angle={0.0: 0.0, 90.0: -10.0,
                                                  180.0: -2.0, 270.0: -5.0}.get)
    reference_cells = _flat_cells(channels, modes, angles, base_mbps=110.0)

    inst_dir = tmp_path / "installed"
    ref_dir = tmp_path / "reference"
    out_dir = tmp_path / "out"
    _write_session(inst_dir, cells=installed_cells)
    _write_session(ref_dir, cells=reference_cells)

    result = analyze(str(inst_dir), str(ref_dir), str(out_dir))

    assert result["warnings"] == []
    assert result["n_cells"] == 1
    assert os.path.isfile(result["summary_csv"])
    assert os.path.isfile(result["summary_json"])
    assert os.path.isfile(result["report_md"])
    assert len(result["polar_pngs"]) == 1
    assert os.path.isfile(result["polar_pngs"][0])

    with open(result["summary_json"]) as fh:
        data = json.load(fh)
    cell = data["cells"][0]
    # installed = 100 + {0,-10,-2,-5}; reference = 110 always.
    # deltas: -10, -20, -12, -15. mean = -14.25; worst at 90 (-20); best at 0 (-10).
    assert cell["wifi_channel"] == 6
    assert cell["iperf_mode"] == "tcp_up"
    assert cell["n_angles"] == 4
    assert cell["mean_delta_mbps"] == pytest.approx(-14.25)
    assert cell["worst_delta_mbps"] == pytest.approx(-20.0)
    assert cell["worst_angle_deg"] == pytest.approx(90.0)
    assert cell["best_delta_mbps"] == pytest.approx(-10.0)
    assert cell["best_angle_deg"] == pytest.approx(0.0)
    assert cell["spread_mbps"] == pytest.approx(10.0)


# ---------------------------------------------------------------------------
# CSV shape
# ---------------------------------------------------------------------------


def test_summary_csv_shape(analyze, tmp_path):
    angles = [0.0, 90.0]
    inst_dir = tmp_path / "installed"
    ref_dir = tmp_path / "reference"
    _write_session(inst_dir, cells={(6, "tcp_up", 0.0): 100.0, (6, "tcp_up", 90.0): 80.0})
    _write_session(ref_dir, cells={(6, "tcp_up", 0.0): 110.0, (6, "tcp_up", 90.0): 105.0})
    out_dir = tmp_path / "out"

    analyze(str(inst_dir), str(ref_dir), str(out_dir))
    csv_path = out_dir / "summary.csv"
    with open(csv_path) as fh:
        reader = csv.DictReader(fh)
        rows = list(reader)
    assert len(rows) == 1
    row = rows[0]
    assert row["wifi_channel"] == "6"
    assert row["iperf_mode"] == "tcp_up"
    assert float(row["mean_delta_mbps"]) == pytest.approx(-17.5)
    assert float(row["worst_delta_mbps"]) == pytest.approx(-25.0)


# ---------------------------------------------------------------------------
# Cell matching
# ---------------------------------------------------------------------------


def test_only_common_cells_are_compared(analyze, tmp_path):
    inst_dir = tmp_path / "installed"
    ref_dir = tmp_path / "reference"
    _write_session(inst_dir, cells={
        (6, "tcp_up", 0.0): 90.0,
        (11, "tcp_up", 0.0): 80.0,        # not present in reference
    })
    _write_session(ref_dir, cells={
        (6, "tcp_up", 0.0): 100.0,
        (1, "tcp_up", 0.0): 95.0,         # not present in installed
    })
    out_dir = tmp_path / "out"

    result = analyze(str(inst_dir), str(ref_dir), str(out_dir))
    assert result["n_cells"] == 1
    with open(result["summary_json"]) as fh:
        data = json.load(fh)
    assert data["cells"][0]["wifi_channel"] == 6
    assert data["cells"][0]["iperf_mode"] == "tcp_up"


def test_angles_present_in_only_one_session_are_dropped(analyze, tmp_path):
    inst_dir = tmp_path / "installed"
    ref_dir = tmp_path / "reference"
    _write_session(inst_dir, cells={
        (6, "tcp_up", 0.0): 100.0,
        (6, "tcp_up", 45.0): 90.0,
        (6, "tcp_up", 90.0): 80.0,
    })
    _write_session(ref_dir, cells={
        (6, "tcp_up", 0.0): 110.0,
        (6, "tcp_up", 90.0): 100.0,
        (6, "tcp_up", 180.0): 95.0,        # installed has no 180°
    })
    out_dir = tmp_path / "out"

    result = analyze(str(inst_dir), str(ref_dir), str(out_dir))
    with open(result["summary_json"]) as fh:
        data = json.load(fh)
    cell = data["cells"][0]
    # Only 0° and 90° intersect, so n_angles == 2.
    assert cell["n_angles"] == 2


# ---------------------------------------------------------------------------
# Verdict threshold smoke
# ---------------------------------------------------------------------------


def test_report_verdict_changes_with_threshold(analyze, tmp_path):
    inst_dir = tmp_path / "installed"
    ref_dir = tmp_path / "reference"
    _write_session(inst_dir, cells={(6, "tcp_up", a): 90.0 for a in (0.0, 90.0, 180.0, 270.0)})
    _write_session(ref_dir, cells={(6, "tcp_up", a): 100.0 for a in (0.0, 90.0, 180.0, 270.0)})
    out_dir = tmp_path / "out"

    analyze(str(inst_dir), str(ref_dir), str(out_dir),
            mean_threshold_mbps=-5.0, worst_threshold_mbps=-15.0)
    report_strict = (out_dir / "report.md").read_text(encoding="utf-8")

    out2 = tmp_path / "out_loose"
    analyze(str(inst_dir), str(ref_dir), str(out2),
            mean_threshold_mbps=-50.0, worst_threshold_mbps=-50.0)
    report_loose = (out2 / "report.md").read_text(encoding="utf-8")

    # Strict thresholds: mean Δ = -10, worse than -5 -> investigate.
    assert "investigate" in report_strict
    # Loose thresholds: -10 better than -50 -> adequate.
    assert "adequate" in report_loose


# ---------------------------------------------------------------------------
# Error paths
# ---------------------------------------------------------------------------


def test_missing_session_dir_surfaces_warning(analyze, tmp_path):
    out_dir = tmp_path / "out"
    result = analyze(str(tmp_path / "missing_installed"),
                     str(tmp_path / "missing_reference"),
                     str(out_dir))
    assert any("manifest_missing" in w for w in result["warnings"])
    assert result["n_cells"] == 0


def test_runs_missing_fields_are_skipped(analyze, tmp_path):
    inst_dir = tmp_path / "installed"
    ref_dir = tmp_path / "reference"
    inst_dir.mkdir(parents=True)
    ref_dir.mkdir(parents=True)
    # Installed run is missing angle_deg; should be skipped with a warning.
    (inst_dir / "session.json").write_text(json.dumps({
        "session_name": "inst",
        "created_utc": "x",
        "runs": [{
            "timestamp_utc": "x", "kind": "wifi_only",
            "wifi_channel": 6, "iperf_mode": "tcp_up",
            "aggregate": {"overall_mbps": 100.0},
        }],
    }))
    _write_session(ref_dir, cells={(6, "tcp_up", 0.0): 110.0})
    out_dir = tmp_path / "out"

    result = analyze(str(inst_dir), str(ref_dir), str(out_dir))
    assert any("missing_field" in w for w in result["warnings"])
    # Only the malformed run was skipped; no valid installed cells remain.
    assert any("no_common_channel_mode_cells" in w
               or "no_installed_runs" in w
               for w in result["warnings"]) or result["n_cells"] == 1


def test_non_wifi_only_runs_are_ignored(analyze, tmp_path):
    inst_dir = tmp_path / "installed"
    ref_dir = tmp_path / "reference"
    inst_dir.mkdir(parents=True)
    ref_dir.mkdir(parents=True)
    (inst_dir / "session.json").write_text(json.dumps({
        "session_name": "inst",
        "created_utc": "x",
        "runs": [
            # baseline kind — should be ignored entirely
            {"timestamp_utc": "x", "kind": "baseline",
             "wifi_channel": 6, "iperf_mode": "tcp_up", "angle_deg": 0.0,
             "aggregate": {"overall_mbps": 999.0}},
            # the only iperf cell we care about
            {"timestamp_utc": "x", "kind": "wifi_only",
             "wifi_channel": 6, "iperf_mode": "tcp_up", "angle_deg": 0.0,
             "aggregate": {"overall_mbps": 90.0}},
        ],
    }))
    _write_session(ref_dir, cells={(6, "tcp_up", 0.0): 100.0})
    out_dir = tmp_path / "out"

    result = analyze(str(inst_dir), str(ref_dir), str(out_dir))
    with open(result["summary_json"]) as fh:
        data = json.load(fh)
    # The baseline 999.0 must NOT contribute — installed mean should be 90, not 544.5.
    assert data["cells"][0]["mean_installed_mbps"] == pytest.approx(90.0)
