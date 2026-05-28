"""v5.2 cal-drift extensions: drift alert (#4), recert reminder (#3),
monitor (#33)."""

from __future__ import annotations

import os
import sys

import pytest

import matplotlib

matplotlib.use("Agg")

from plot_antenna.cal_drift import (
    DriftResult,
    evaluate_drift_alert,
    gain_standard_recert_status,
)


def _dr(h_mean, h_max, v_mean=0.0, v_max=0.0, consistency=None):
    return DriftResult(
        deltas=None,
        stats={
            "H": {"mean": h_mean, "max_abs": h_max},
            "V": {"mean": v_mean, "max_abs": v_max},
        },
        consistency=consistency or {},
        missing_audit={},
        baseline=None,
        current=None,
    )


# ----------------------------- #4 drift alert -----------------------------


def test_alert_levels():
    assert evaluate_drift_alert(_dr(0.2, 0.3))["level"] == "OK"
    assert evaluate_drift_alert(_dr(0.2, 0.7))["level"] == "WARN"  # max 0.7 >= 0.5
    assert evaluate_drift_alert(_dr(0.2, 1.3))["level"] == "ALERT"  # max 1.3 >= 1.0
    assert evaluate_drift_alert(_dr(1.2, 0.3))["level"] == "ALERT"  # mean 1.2 >= 1.0


def test_alert_setup_group_mismatch_escalates_ok_to_warn():
    out = evaluate_drift_alert(_dr(0.1, 0.2, consistency={"setup_group": ("mismatch", "a", "b")}))
    assert out["level"] == "WARN"
    assert any("setup_group mismatch" in r for r in out["reasons"])


def test_alert_custom_thresholds():
    # With a tighter warn threshold, a 0.3 dB drift trips WARN.
    out = evaluate_drift_alert(_dr(0.0, 0.3), warn_db=0.2, alert_db=0.5)
    assert out["level"] == "WARN"


# ----------------------------- #3 recert reminder -----------------------------


def test_recert_ok_within_interval():
    out = gain_standard_recert_status(
        "2026-03-01", interval_months=12, last_recert_date="2025-06-01"
    )
    assert out["status"] == "ok"
    assert out["due_date"] == "2026-06-01"


def test_recert_due_past_interval():
    out = gain_standard_recert_status(
        "2026-09-01", interval_months=12, last_recert_date="2025-06-01"
    )
    assert out["status"] == "due"


def test_recert_unknown_without_reference():
    assert gain_standard_recert_status("2026-03-01")["status"] == "unknown"


def test_recert_bad_date_is_unknown():
    assert gain_standard_recert_status("not-a-date", 12, "2025-06-01")["status"] == "unknown"


# ----------------------------- #4/#33 MCP never-raise -----------------------------


# ----------------------------- #5/#6 cal-type + cable-loss history -----------------------------

from pathlib import Path

import plot_antenna.cal_drift as cal_drift

_FIXTURES = Path(__file__).parent / "fixtures" / "cal_drift"
_BASELINE_CAL = _FIXTURES / "cal_baseline.txt"
_SHIFTED_CAL = _FIXTURES / "cal_shifted.txt"
_REF_HPOL = _FIXTURES / "ref_hpol.txt"


@pytest.fixture
def _hist_dir(tmp_path, monkeypatch):
    d = tmp_path / "cal_drift"
    monkeypatch.setenv("RFLECT_CAL_DRIFT_DIR", str(d))
    return d


def test_record_run_persists_cal_type_and_cable_loss(_hist_dir, tmp_path):
    cable = tmp_path / "cable_loss.s2p"
    cable.write_text("! mock cable loss\n# GHz S MA R 50\n2.4 0.1 0\n")
    meta = cal_drift.record_run(
        cal_result={
            "output_path": str(_BASELINE_CAL),
            "summary_path": "",
            "rows_written": 0,
            "rows_missing": 0,
        },
        hpol_ref_file=str(_REF_HPOL),
        cal_type="passive",
        cable_loss_file=str(cable),
    )
    assert meta is not None
    assert meta.cal_type == "passive"
    assert meta.cable_loss_file == str(cable)
    assert meta.cable_loss_sha256  # sha computed for an existing file
    # Persisted + readable back.
    got = cal_drift.get_run(meta.run_id)
    assert got.cal_type == "passive"
    assert got.cable_loss_file == str(cable)


def test_cable_loss_history(_hist_dir, tmp_path):
    cable = tmp_path / "cl.s2p"
    cable.write_text("! cl\n# GHz S MA R 50\n2.4 0.1 0\n")
    cal_drift.record_run(
        cal_result={
            "output_path": str(_BASELINE_CAL),
            "summary_path": "",
            "rows_written": 0,
            "rows_missing": 0,
        },
        hpol_ref_file=str(_REF_HPOL),
        cable_loss_file=str(cable),
    )
    # A second run without a cable-loss file should not appear in the history.
    cal_drift.record_run(
        cal_result={
            "output_path": str(_SHIFTED_CAL),
            "summary_path": "",
            "rows_written": 0,
            "rows_missing": 0,
        },
        hpol_ref_file=str(_REF_HPOL),
    )
    hist = cal_drift.cable_loss_history()
    assert len(hist) == 1
    assert hist[0]["cable_loss_file"] == str(cable)
    assert hist[0]["cable_loss_sha256"]


def test_cal_type_defaults_to_active(_hist_dir):
    meta = cal_drift.record_run(
        cal_result={
            "output_path": str(_BASELINE_CAL),
            "summary_path": "",
            "rows_written": 0,
            "rows_missing": 0,
        },
        hpol_ref_file=str(_REF_HPOL),
    )
    assert meta.cal_type == "active"


def test_mcp_alert_and_monitor_never_raise_on_missing_runs():
    pytest.importorskip("mcp", reason="mcp package not installed")
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(__file__)), "rflect-mcp"))
    from mcp.server.fastmcp import FastMCP
    from tools.cal_drift_tools import register_cal_drift_tools

    mcp = FastMCP("cd-v52")
    register_cal_drift_tools(mcp)
    reg = mcp._tool_manager._tools

    alert = reg["cal_drift_alert"].fn("nope-a", "nope-b")
    assert isinstance(alert, dict) and "error" in alert

    recert = reg["cal_drift_recert_check"].fn("2026-03-01", 12, "2025-06-01")
    assert recert["status"] == "ok"
