"""Tests for the Active Chamber Calibration flow.

Uses the real 2026-04-14 reference files in the user's Downloads folder; skips
when those files are not present so CI does not require them.
"""

import os
from pathlib import Path

import pytest

from plot_antenna.file_utils import (
    check_matching_files,
    generate_active_cal_file,
)
from plot_antenna.calculations import extract_passive_frequencies


SAMPLES_DIR = Path(
    os.path.expanduser("~/Downloads/Active Calibration/2026-4-14 Active Calibration")
)

HPOL = SAMPLES_DIR / "BLPA 690-2700 AP_HPol.txt"
VPOL = SAMPLES_DIR / "BLPA 690-2700 AP_VPol.txt"
POWER = SAMPLES_DIR / "P_690-2700 2026-04-09 AP_VPol.txt"
GAIN_STD = SAMPLES_DIR / "Howland BLPA-19 3100 Gain 2021-10-14.txt"


pytestmark = pytest.mark.skipif(
    not all(p.exists() for p in (HPOL, VPOL, POWER, GAIN_STD)),
    reason="Active Calibration reference files not available",
)


def test_check_matching_files_rejects_active_cal_pair_in_strict_mode():
    """Passive-scan behavior must be preserved: Axis1 0° vs 90° should fail."""
    ok, message = check_matching_files(str(HPOL), str(VPOL))
    assert ok is False
    assert "mismatched angle data" in message


def test_check_matching_files_accepts_active_cal_pair_in_relaxed_mode():
    """Active cal flow passes strict_angles=False to allow the H/V 90° offset."""
    ok, message = check_matching_files(str(HPOL), str(VPOL), strict_angles=False)
    assert ok is True, f"unexpected rejection: {message}"


def test_generate_active_cal_file_end_to_end(tmp_path):
    """Full active-cal generation: produces non-empty cal file with expected row count."""
    import shutil

    # Copy inputs into tmp_path so generated output doesn't pollute Downloads
    hpol = tmp_path / HPOL.name
    vpol = tmp_path / VPOL.name
    power = tmp_path / POWER.name
    gain = tmp_path / GAIN_STD.name
    for src, dst in ((HPOL, hpol), (VPOL, vpol), (POWER, power), (GAIN_STD, gain)):
        shutil.copyfile(src, dst)

    freq_list = extract_passive_frequencies(str(hpol))
    assert len(freq_list) > 0

    result = generate_active_cal_file(
        str(power), str(gain), str(hpol), str(vpol), 0.0, freq_list
    )

    assert result is not None
    assert result["rows_written"] >= 250  # BLPA 690-2700 band has ~252 valid points
    assert Path(result["output_path"]).exists()
    assert Path(result["summary_path"]).exists()

    # Spot-check the first data line of the output has the expected 3-column format
    with open(result["output_path"], "r", encoding="utf-8") as f:
        lines = f.readlines()
    data_lines = [ln for ln in lines if ln and ln[0].isdigit()]
    assert len(data_lines) == result["rows_written"]
    first = data_lines[0].split("\t")
    assert len(first) == 3  # Freq, P+G-H, P+G-V
    float(first[0])
    float(first[1])
    float(first[2])
