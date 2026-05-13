"""Tests for the `process_folder` orchestration MCP tool.

Most delegates (batch_process_passive_scans, batch_process_active_scans,
analyze_uwb_channel) need real chamber data which is not available in CI.
We monkeypatch those delegates and assert the orchestration logic itself —
intent detection, error handling, mixed-folder priority, warnings shape.

One real-delegate test exercises the cal_drift path using the existing
synthetic fixtures under tests/fixtures/cal_drift/ (same approach as
tests/test_cal_drift.py).
"""

from __future__ import annotations

import os
import shutil
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


CAL_FIXTURES = Path(__file__).parent / "fixtures" / "cal_drift"


@pytest.fixture(scope="module")
def mcp_server():
    from tools.import_tools import register_import_tools
    from tools.bulk_tools import register_bulk_tools
    from tools.uwb_tools import register_uwb_tools
    from tools.cal_drift_tools import register_cal_drift_tools
    from tools.report_tools import register_report_tools
    from tools.orchestration import register_orchestration_tools

    mcp = FastMCP("rflect-orchestration-test")
    register_import_tools(mcp)
    register_bulk_tools(mcp)
    register_uwb_tools(mcp)
    register_cal_drift_tools(mcp)
    register_report_tools(mcp)
    register_orchestration_tools(mcp)
    return mcp


@pytest.fixture
def process_folder(mcp_server):
    return mcp_server._tool_manager._tools["process_folder"].fn


@pytest.fixture(autouse=True)
def _clear_state():
    from tools.import_tools import _loaded_measurements
    _loaded_measurements.clear()
    yield
    _loaded_measurements.clear()


def _touch(p: Path) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text("")


# --------------------------------------------------------------------- #
# Intent detection / scanning
# --------------------------------------------------------------------- #


def test_folder_not_found(process_folder, tmp_path):
    missing = tmp_path / "does_not_exist"
    out = process_folder(str(missing))
    assert out["intent_used"] is None
    assert out["files_processed"] == 0
    assert any("folder_not_found" in w for w in out["warnings"])


def test_invalid_intent(process_folder, tmp_path):
    out = process_folder(str(tmp_path), intent="bogus")
    assert out["intent_used"] is None
    assert any("invalid_intent" in w for w in out["warnings"])


def test_no_match_returns_warning(process_folder, tmp_path):
    (tmp_path / "random.bin").write_text("nothing")
    out = process_folder(str(tmp_path), intent="auto")
    assert out["intent_used"] is None
    assert any("no_files_matched_intent" in w for w in out["warnings"])


def test_auto_detect_picks_passive_over_active(process_folder, tmp_path, monkeypatch):
    # Mixed folder: one passive pair + one active TRP. Priority should pick passive.
    _touch(tmp_path / "PassiveTest_X_HPol.txt")
    _touch(tmp_path / "PassiveTest_X_VPol.txt")
    _touch(tmp_path / "Active Test_Y TRP.txt")

    monkeypatch.setattr(
        "tools.orchestration._run_passive",
        lambda folder, scan, freqs: {
            "files_processed": 2,
            "frequencies_loaded": [2400.0],
            "warnings": [],
        },
    )

    out = process_folder(str(tmp_path), intent="auto")
    assert out["intent_used"] == "passive"
    assert out["files_processed"] == 2
    assert any("mixed_intents_detected" in w for w in out["warnings"])


def test_explicit_active_intent(process_folder, tmp_path, monkeypatch):
    _touch(tmp_path / "Active Test_BLE TRP.txt")
    monkeypatch.setattr(
        "tools.orchestration._run_active",
        lambda folder, scan: {
            "files_processed": 1,
            "frequencies_loaded": [],
            "warnings": [],
        },
    )

    out = process_folder(str(tmp_path), intent="active")
    assert out["intent_used"] == "active"
    assert out["files_processed"] == 1
    assert out["warnings"] == []


def test_explicit_uwb_intent_monkeypatched(process_folder, tmp_path, monkeypatch):
    _touch(tmp_path / "ant_45deg.s2p")
    _touch(tmp_path / "ant_90deg.s2p")

    def fake_uwb(path, *args, **kwargs):
        return {"sff": {"value": 0.9, "quality": "good", "peak_delay_ns": 1.0}}

    monkeypatch.setattr(
        "tools.uwb_tools._analyze_uwb_channel_dict",
        fake_uwb,
    )

    out = process_folder(str(tmp_path), intent="uwb")
    assert out["intent_used"] == "uwb"
    assert out["files_processed"] == 2
    assert out["warnings"] == []
    assert "extra" in out and len(out["extra"]["results"]) == 2


# --------------------------------------------------------------------- #
# Real-delegate test: cal_drift
# --------------------------------------------------------------------- #


def test_cal_drift_intent_with_real_fixtures(process_folder, tmp_path):
    staging = tmp_path / "archive"
    staging.mkdir()
    shutil.copy(
        CAL_FIXTURES / "cal_baseline.txt",
        staging / "TRP Cal BLPA 690-2700 1Amp 0 dBm 700-1600 01-15-24.txt",
    )
    shutil.copy(
        CAL_FIXTURES / "cal_shifted.txt",
        staging / "TRP Cal BLPA 690-2700 1Amp 0 dBm 700-1600 04-14-26.txt",
    )

    out = process_folder(str(staging), intent="cal_drift")
    assert out["intent_used"] == "cal_drift"
    assert out["files_processed"] == 2
    assert out["warnings"] == []
    assert "extra" in out and len(out["extra"]["run_ids"]) == 2


# --------------------------------------------------------------------- #
# Report-failure isolation
# --------------------------------------------------------------------- #


def test_report_requested_but_no_loaded_data(process_folder, tmp_path, monkeypatch):
    # Use UWB intent: it never loads into _loaded_measurements, so report=True
    # should record a warning and leave report_path None without raising.
    _touch(tmp_path / "ant_0deg.s2p")

    monkeypatch.setattr(
        "tools.uwb_tools._analyze_uwb_channel_dict",
        lambda *args, **kwargs: {"sff": {"value": 0.5, "quality": "fair", "peak_delay_ns": 0.0}},
    )

    out = process_folder(str(tmp_path), intent="uwb", report=True)
    assert out["intent_used"] == "uwb"
    assert out["report_path"] is None
    assert any("report_skipped_no_loaded_measurements" in w for w in out["warnings"])


def test_report_generation_failure_does_not_break_processing(
    process_folder, tmp_path, monkeypatch
):
    # Stub passive delegate, stub get_loaded_measurements to be non-empty so
    # the report path is attempted; stub _prepare_report_data to raise.
    _touch(tmp_path / "PassiveTest_X_HPol.txt")
    _touch(tmp_path / "PassiveTest_X_VPol.txt")

    monkeypatch.setattr(
        "tools.orchestration._run_passive",
        lambda folder, scan, freqs: {
            "files_processed": 2,
            "frequencies_loaded": [2400.0],
            "warnings": [],
        },
    )
    monkeypatch.setattr(
        "tools.import_tools.get_loaded_measurements",
        lambda: {"dummy": object()},
    )

    def boom(*args, **kwargs):
        raise RuntimeError("simulated report failure")

    monkeypatch.setattr("tools.report_tools._prepare_report_data", boom)

    out = process_folder(str(tmp_path), intent="passive", report=True)
    assert out["intent_used"] == "passive"
    assert out["files_processed"] == 2
    assert out["report_path"] is None
    assert any("report_generation_failed" in w for w in out["warnings"])
