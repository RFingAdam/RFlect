"""Tests for the calibration drift tracker (`plot_antenna.cal_drift`).

Uses synthetic fixtures under tests/fixtures/cal_drift/ so CI does not depend
on the real 2026-04-14 archive. One end-to-end test is guarded by the same
`~/Downloads/Active Calibration` skip as `tests/test_active_calibration.py`.
"""

from __future__ import annotations

import os
import shutil
from pathlib import Path

import pandas as pd
import pytest

from plot_antenna import cal_drift


FIXTURES = Path(__file__).parent / "fixtures" / "cal_drift"
BASELINE_CAL = FIXTURES / "cal_baseline.txt"
SHIFTED_CAL = FIXTURES / "cal_shifted.txt"
MISSING_CAL = FIXTURES / "cal_missing_freq.txt"
REF_HPOL = FIXTURES / "ref_hpol.txt"
SUMMARY = FIXTURES / "cal_summary.txt"


# ──────────────────────────────────────────────────────────────────────────
# Fixtures
# ──────────────────────────────────────────────────────────────────────────


@pytest.fixture
def history_dir(tmp_path, monkeypatch):
    """Isolated history directory per test (via env var override)."""
    d = tmp_path / "cal_drift"
    monkeypatch.setenv("RFLECT_CAL_DRIFT_DIR", str(d))
    return d


def _record_fixture(cal_path: Path, hpol: Path = REF_HPOL) -> cal_drift.CalRunMeta:
    meta = cal_drift.record_run(
        cal_result={
            "output_path": str(cal_path),
            "summary_path": str(SUMMARY),
            "rows_written": 10,
            "rows_missing": 0,
        },
        hpol_ref_file=str(hpol),
    )
    assert meta is not None
    return meta


# ──────────────────────────────────────────────────────────────────────────
# Parser tests
# ──────────────────────────────────────────────────────────────────────────


def test_parse_trp_cal_file_golden():
    df = cal_drift.parse_trp_cal_file(BASELINE_CAL)
    assert list(df.columns) == ["freq_mhz", "hpol_dbm", "vpol_dbm"]
    assert len(df) == 10
    assert df.iloc[0]["freq_mhz"] == 700.0
    assert df.iloc[0]["hpol_dbm"] == pytest.approx(21.50)
    assert df.iloc[-1]["freq_mhz"] == 1600.0
    assert df.iloc[-1]["vpol_dbm"] == pytest.approx(24.80)


def test_parse_cal_file_header_extracts_date_time():
    hdr = cal_drift.parse_cal_file_header(BASELINE_CAL)
    assert hdr["date"] == "01-15-24"
    assert hdr["time"] == "10:30:00"


def test_parse_ref_file_header_extracts_method_fields():
    hdr = cal_drift.parse_ref_file_header(REF_HPOL)
    assert hdr["signal_source"] == "Rohde & Schwarz SMV03 Signal Generator"
    assert hdr["output_level_dBm"] == pytest.approx(0.0)
    assert hdr["receiver"] == "Rohde & Schwarz NRX Power Meter Sensor A"
    assert hdr["start_mhz"] == pytest.approx(700.0)
    assert hdr["stop_mhz"] == pytest.approx(1600.0)


def test_parse_cal_summary_extracts_source_paths_and_missing():
    s = cal_drift.parse_cal_summary_file(SUMMARY)
    assert s["power_file"].endswith("power_file.txt")
    assert s["gain_std_file"].endswith("gain_std.txt")
    assert s["hpol_file"].endswith("ref_hpol.txt")
    assert s["vpol_file"].endswith("ref_vpol.txt")
    assert s["freq_start_mhz"] == 700.0
    assert s["freq_stop_mhz"] == 1600.0
    assert s["missing_freqs"] == [965.0, 970.0, 975.0]


# ──────────────────────────────────────────────────────────────────────────
# Record / list / load
# ──────────────────────────────────────────────────────────────────────────


def test_record_run_appends_row_and_points(history_dir):
    meta = _record_fixture(BASELINE_CAL)
    runs = cal_drift.list_runs()
    assert len(runs) == 1
    assert runs.iloc[0]["run_id"] == meta.run_id
    assert runs.iloc[0]["antenna"] == "UNKNOWN"  # fixture doesn't match TRP-Cal filename pattern
    points = cal_drift.load_points(meta.run_id)
    assert len(points) == 10


def test_record_run_is_idempotent(history_dir):
    meta1 = _record_fixture(BASELINE_CAL)
    dup = cal_drift.record_run(
        cal_result={"output_path": str(BASELINE_CAL), "summary_path": "",
                    "rows_written": 10, "rows_missing": 0},
    )
    assert dup is None
    runs = cal_drift.list_runs()
    assert len(runs) == 1
    assert runs.iloc[0]["run_id"] == meta1.run_id


def test_record_run_captures_ref_header_metadata(history_dir):
    meta = _record_fixture(BASELINE_CAL, hpol=REF_HPOL)
    assert meta.signal_source == "Rohde & Schwarz SMV03 Signal Generator"
    assert meta.source_level_dBm == pytest.approx(0.0)


# ──────────────────────────────────────────────────────────────────────────
# Drift compute
# ──────────────────────────────────────────────────────────────────────────


def test_compute_drift_flags_injected_shift(history_dir):
    base = _record_fixture(BASELINE_CAL)
    cur = _record_fixture(SHIFTED_CAL)
    result = cal_drift.compute_drift(base.run_id, cur.run_id)
    # cal_shifted.txt is baseline + 0.75 dB on H-pol, V-pol unchanged
    assert result.stats["H"]["pct_gt_0_5"] == pytest.approx(100.0)
    assert result.stats["H"]["max_abs"] == pytest.approx(0.75, abs=0.001)
    assert result.stats["V"]["max_abs"] == pytest.approx(0.0, abs=0.001)
    # Outlier column should be True everywhere for H, False for V
    assert result.deltas["outlier_h"].all()
    assert not result.deltas["outlier_v"].any()


def test_compute_drift_missing_frequency_audit(history_dir):
    base = _record_fixture(BASELINE_CAL)
    cur = _record_fixture(MISSING_CAL)
    result = cal_drift.compute_drift(base.run_id, cur.run_id)
    # cal_missing_freq.txt drops 900 MHz
    assert 900.0 in result.missing_audit["disappeared_in_current"]
    assert result.missing_audit["appeared_in_current"] == []


def test_consistency_flags_gain_standard_change(history_dir, tmp_path):
    # Record baseline with REF_HPOL
    base = _record_fixture(BASELINE_CAL, hpol=REF_HPOL)

    # Create a second ref file with a different "Signal Source" value so
    # signal_source mismatches between the two runs.
    alt_ref = tmp_path / "alt_ref_hpol.txt"
    alt_ref.write_text(
        REF_HPOL.read_text().replace(
            "Rohde & Schwarz SMV03 Signal Generator",
            "Keysight N5183B Signal Generator",
        )
    )
    cur = _record_fixture(SHIFTED_CAL, hpol=alt_ref)
    result = cal_drift.compute_drift(base.run_id, cur.run_id)
    assert result.consistency["signal_source"]["state"] == "mismatch"
    assert result.consistency["antenna"]["state"] == "match"


# ──────────────────────────────────────────────────────────────────────────
# Historical import
# ──────────────────────────────────────────────────────────────────────────


def test_import_historical_dir_skips_summary_files(history_dir, tmp_path):
    staging = tmp_path / "archive"
    staging.mkdir()
    # Copy fixtures using TRP-Cal naming so import picks them up
    shutil.copy(BASELINE_CAL, staging / "TRP Cal BLPA 690-2700 1Amp 0 dBm 700-1600 01-15-24.txt")
    shutil.copy(SHIFTED_CAL, staging / "TRP Cal BLPA 690-2700 1Amp 0 dBm 700-1600 04-14-26.txt")
    # Summary file should be ignored by the scanner
    shutil.copy(
        SUMMARY,
        staging / "TRP Cal Summary BLPA 690-2700 1Amp 0 dBm 700-1600 01-15-24.txt",
    )

    result = cal_drift.import_historical_dir(staging)
    assert result["ingested"] == 2
    assert result["skipped_duplicate"] == 0
    assert result["failed"] == 0

    runs = cal_drift.list_runs()
    assert len(runs) == 2
    assert set(runs["antenna"]) == {"BLPA"}
    assert set(runs["band_label"]) == {"690-2700"}


def test_import_historical_dir_is_idempotent(history_dir, tmp_path):
    staging = tmp_path / "archive"
    staging.mkdir()
    shutil.copy(BASELINE_CAL, staging / "TRP Cal BLPA 690-2700 1Amp 0 dBm 700-1600 01-15-24.txt")
    r1 = cal_drift.import_historical_dir(staging)
    r2 = cal_drift.import_historical_dir(staging)
    assert r1["ingested"] == 1
    assert r2["ingested"] == 0
    assert r2["skipped_duplicate"] == 1


# ──────────────────────────────────────────────────────────────────────────
# Reporting
# ──────────────────────────────────────────────────────────────────────────


def test_export_markdown_produces_expected_sections(history_dir, tmp_path):
    base = _record_fixture(BASELINE_CAL)
    cur = _record_fixture(SHIFTED_CAL)
    result = cal_drift.compute_drift(base.run_id, cur.run_id)
    out = tmp_path / "drift.md"
    cal_drift.export_markdown(result, out)
    md = out.read_text()
    assert "Summary stats" in md
    assert "Method consistency" in md
    assert "Worst 10 points" in md


def test_render_delta_plot_returns_figure(history_dir):
    import matplotlib
    matplotlib.use("Agg", force=True)
    base = _record_fixture(BASELINE_CAL)
    cur = _record_fixture(SHIFTED_CAL)
    result = cal_drift.compute_drift(base.run_id, cur.run_id)
    fig = cal_drift.render_delta_plot(result)
    assert fig is not None
    assert len(fig.axes) == 2


def test_result_to_dict_is_json_safe(history_dir):
    import json
    base = _record_fixture(BASELINE_CAL)
    cur = _record_fixture(SHIFTED_CAL)
    result = cal_drift.compute_drift(base.run_id, cur.run_id)
    d = cal_drift.result_to_dict(result)
    json.dumps(d)  # must not raise


# ──────────────────────────────────────────────────────────────────────────
# Update / delete
# ──────────────────────────────────────────────────────────────────────────


def test_update_notes(history_dir):
    meta = _record_fixture(BASELINE_CAL)
    assert cal_drift.update_notes(meta.run_id, "chamber foam replaced")
    assert cal_drift.get_run(meta.run_id).operator_notes == "chamber foam replaced"


def test_set_setup_group_and_consistency_flag(history_dir):
    base = _record_fixture(BASELINE_CAL)
    cur = _record_fixture(SHIFTED_CAL)
    assert cal_drift.set_setup_group(base.run_id, "pre-2024-setup")
    assert cal_drift.set_setup_group(cur.run_id, "2024-current")
    assert cal_drift.get_run(base.run_id).setup_group == "pre-2024-setup"
    result = cal_drift.compute_drift(base.run_id, cur.run_id)
    # Cross-epoch comparison must be flagged
    assert result.consistency["setup_group"]["state"] == "mismatch"
    assert result.consistency["setup_group"]["baseline"] == "pre-2024-setup"
    assert result.consistency["setup_group"]["current"] == "2024-current"


def test_setup_group_matches_when_same(history_dir):
    base = _record_fixture(BASELINE_CAL)
    cur = _record_fixture(SHIFTED_CAL)
    cal_drift.set_setup_group(base.run_id, "2024-current")
    cal_drift.set_setup_group(cur.run_id, "2024-current")
    result = cal_drift.compute_drift(base.run_id, cur.run_id)
    assert result.consistency["setup_group"]["state"] == "match"


def test_delete_run(history_dir):
    meta = _record_fixture(BASELINE_CAL)
    assert cal_drift.delete_run(meta.run_id)
    assert cal_drift.list_runs().empty
    assert not cal_drift._points_csv_path(meta.run_id).exists()


# ──────────────────────────────────────────────────────────────────────────
# End-to-end with the real Active Cal flow
# ──────────────────────────────────────────────────────────────────────────


_DOWNLOADS = Path(
    os.path.expanduser("~/Downloads/Active Calibration/2026-4-14 Active Calibration")
)
_REAL_SAMPLES = [
    _DOWNLOADS / "BLPA 690-2700 AP_HPol.txt",
    _DOWNLOADS / "BLPA 690-2700 AP_VPol.txt",
    _DOWNLOADS / "P_690-2700 2026-04-09 AP_VPol.txt",
    _DOWNLOADS / "Howland BLPA-19 3100 Gain 2021-10-14.txt",
]


@pytest.mark.skipif(
    not all(p.exists() for p in _REAL_SAMPLES),
    reason="Real Active Calibration samples not available",
)
def test_generate_active_cal_file_auto_captures(tmp_path, monkeypatch):
    """Hook into the real generate_active_cal_file and verify drift auto-capture."""
    from plot_antenna.file_utils import generate_active_cal_file
    from plot_antenna.calculations import extract_passive_frequencies

    # Isolate the history directory for this test
    hist = tmp_path / "cal_drift"
    monkeypatch.setenv("RFLECT_CAL_DRIFT_DIR", str(hist))

    # Stage inputs so the generated output lands in tmp_path
    hpol = tmp_path / _REAL_SAMPLES[0].name
    vpol = tmp_path / _REAL_SAMPLES[1].name
    power = tmp_path / _REAL_SAMPLES[2].name
    gain = tmp_path / _REAL_SAMPLES[3].name
    for src, dst in zip(_REAL_SAMPLES, (hpol, vpol, power, gain)):
        shutil.copyfile(src, dst)

    freq_list = extract_passive_frequencies(str(hpol))
    result = generate_active_cal_file(
        str(power), str(gain), str(hpol), str(vpol), 0.0, freq_list
    )
    assert result is not None

    # Auto-capture should have written to runs.csv
    runs = cal_drift.list_runs()
    assert len(runs) == 1
    row = runs.iloc[0]
    assert row["antenna"] == "BLPA"
    assert row["band_label"] == "690-2700"
    assert row["rows_written"] >= 250
