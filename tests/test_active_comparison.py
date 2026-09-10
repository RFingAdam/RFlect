"""Active overlays retain measured levels and reject incompatible inputs."""

import sys
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "rflect-mcp"))
from tools import active_comparison_tools as comparison_tools
from tools.import_tools import LoadedMeasurement
from tools.summary_stats import active_summary


@pytest.fixture
def measurements(monkeypatch):
    loaded = {}
    for index in range(3):
        values = (
            np.array(
                [[-30, -20, -10, -20, -30], [-25, -15, -5, -15, -25], [-30, -20, -10, -20, -30]]
            )
            + index
        )
        loaded[f"private-source-{index}"] = LoadedMeasurement(
            file_path=f"private/source-{index}.txt",
            scan_type="active",
            frequencies=[2450.0],
            data={
                "theta": np.array([0, 90, 180]),
                "phi": np.array([0, 90, 180, 270, 360]),
                "total_power_2d": values,
                "h_power_2d": values - 3,
                "v_power_2d": values - 4,
                "TRP_dBm": float(index),
            },
        )
    monkeypatch.setattr(comparison_tools, "get_loaded_measurements", lambda: loaded)
    return loaded


def test_three_trace_export_preserves_values(measurements, tmp_path, monkeypatch):
    figures = []
    original = comparison_tools.Figure

    def capture(*args, **kwargs):
        figure = original(*args, **kwargs)
        figures.append(figure)
        return figure

    monkeypatch.setattr(comparison_tools, "Figure", capture)
    target = tmp_path / "overlay.png"
    result = comparison_tools.compare_active_overlay(
        list(measurements), str(target), max_envelope=True
    )
    lines = figures[0].axes[0].lines
    np.testing.assert_array_equal(lines[0].get_ydata(), [-25, -15, -5, -15, -25])
    np.testing.assert_array_equal(lines[3].get_ydata(), [-23, -13, -3, -13, -23])
    assert result["radial_limits_dbm"] == [-25, 0]
    assert all("private" not in line.get_label() for line in lines)
    with Image.open(target) as image:
        assert image.format == "PNG"
        assert image.convert("RGB").getextrema()[0][0] < 255
    with pytest.raises(FileExistsError):
        comparison_tools.compare_active_overlay(list(measurements), str(target))


@pytest.mark.parametrize(
    "change,message",
    [
        ("frequency", "same frequency"),
        ("cut", "Requested cut"),
        ("nan", "finite"),
        ("shape", "power grid"),
        ("grid", "matching angular"),
    ],
)
def test_incompatible_data_is_rejected(measurements, tmp_path, change, message):
    measurement = list(measurements.values())[1]
    if change == "frequency":
        measurement.frequencies = [2400.0]
    elif change == "cut":
        measurement.data["theta"] = np.array([0, 80, 180])
    elif change == "nan":
        measurement.data["total_power_2d"] = np.full((3, 5), np.nan)
    elif change == "shape":
        measurement.data["total_power_2d"] = np.zeros((2, 5))
    else:
        measurement.data["phi"] = np.array([0, 80, 180, 270, 360])
    target = tmp_path / "bad.png"
    with pytest.raises(ValueError, match=message):
        comparison_tools.compare_active_overlay(list(measurements), str(target), max_envelope=True)
    assert not target.exists()


def test_summary_groups_twenty_files_without_leaking_names():
    measurements = [
        LoadedMeasurement(
            file_path="private-customer-project.txt",
            scan_type="active",
            frequencies=[frequency],
            data={"TRP_dBm": float(index), "dut_id": dut},
        )
        for dut in ("private-device-a", "private-device-b")
        for frequency in (2400, 2480)
        for index in range(5)
    ]
    summary = " ".join(active_summary(measurements))
    assert summary.count("5 measurement(s)") == 4
    assert summary.count("0.00 to 4.00 dBm, spread 4.00 dB") == 4
    assert "private" not in summary
    assert "uncertainty" in summary
    assert "pass" not in summary.lower()


def test_summary_dut_labels_are_independent_of_measurement_order():
    measurements = [
        LoadedMeasurement("", "active", [2450], {"TRP_dBm": value, "dut_id": dut})
        for dut, value in [("z-device", 2.0), ("a-device", 1.0)]
    ]
    forward = active_summary(measurements)
    reverse = active_summary(list(reversed(measurements)))
    assert forward == reverse
    assert "DUT 1 at 2450 MHz: 1 measurement(s), TRP 1.00 to 1.00 dBm" in " ".join(forward)


def test_summary_rejects_incomplete_explicit_groups(measurements):
    with pytest.raises(ValueError, match="one DUT group"):
        active_summary(list(measurements.values()), dut_ids=["only-one"])


def test_summary_excludes_nonfinite_trp_and_filters_frequency():
    measurements = [
        LoadedMeasurement("", "active", [freq], {"TRP_dBm": value})
        for freq, value in [(2400, -3), (2400, float("nan")), (2480, 5)]
    ]
    summary = " ".join(active_summary(measurements, frequencies=[2400]))
    assert "-3.00 to -3.00" in summary
    assert "1 measurement(s) excluded" in summary
    assert "2480" not in summary


def test_report_options_group_without_modifying_loaded_measurements(measurements):
    from tools.report_tools import ReportOptions, _build_executive_summary

    options = ReportOptions(measurement_groups={name: "internal-group" for name in measurements})
    summary = " ".join(_build_executive_summary(measurements, options))
    assert "DUT 1 at 2450 MHz: 3 measurement(s)" in summary
    assert "internal-group" not in summary
    assert all("dut_id" not in value.data for value in measurements.values())
