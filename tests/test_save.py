"""Tests for plot_antenna.save pure helper functions (issue #14)."""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")

from plot_antenna.save import (
    detect_measurement_type,
    extract_frequency_from_filename,
    extract_plot_subtype,
    deduplicate_images,
    _fmt,
)


class TestDetectMeasurementType:
    def test_maritime_before_polarization(self):
        # "conical...polar" must classify as maritime, not polarization.
        assert detect_measurement_type("conical_cuts_polar_2450.png") == "maritime"
        assert detect_measurement_type("mercator_view.png") == "maritime"

    def test_polarization(self):
        assert detect_measurement_type("axial_ratio_ar_2450.png") == "polarization"
        assert detect_measurement_type("xpd_plot.png") == "polarization"

    def test_active(self):
        assert detect_measurement_type("Active_TRP_2440.png") == "active"
        assert detect_measurement_type("total_power_3d.png") == "active"

    def test_passive(self):
        assert detect_measurement_type("passive_gain_2400.png") == "passive"

    def test_unknown(self):
        assert detect_measurement_type("random_file.png") is None


class TestExtractFrequency:
    def test_basic_mhz(self):
        # Returns a frequency token from the filename (exact format is the
        # contract under test — just assert it finds the 2450 value).
        out = extract_frequency_from_filename("gain_2450MHz.png")
        assert "2450" in str(out)

    def test_no_frequency(self):
        # Should not raise on a filename without a frequency.
        extract_frequency_from_filename("no_freq_here.png")


class TestFmt:
    def test_float(self):
        assert _fmt(10.123) == "10.12"

    def test_none(self):
        assert _fmt(None) == "N/A"

    def test_suffix(self):
        assert _fmt(3.0, ".1f", " dBi") == "3.0 dBi"


def test_deduplicate_images_removes_repeats():
    pairs = [("a.png", "active"), ("a.png", "active"), ("b.png", "passive")]
    out = deduplicate_images(pairs)
    # All unique basenames preserved; duplicate dropped.
    names = [p[0] for p in out]
    assert names.count("a.png") == 1
    assert "b.png" in names


def test_extract_plot_subtype_runs():
    # Smoke: returns a string/None without raising for representative names.
    for n in ("2D_Azimuth_Cuts_2450.png", "3D_TRP_total_2450.png", "weird.png"):
        extract_plot_subtype(n)
