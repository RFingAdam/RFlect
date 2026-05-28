"""Tests for plot_antenna.groupdelay pure-compute functions.

Covers the group-delay dispersion metric (issue #8) and starts coverage for
the previously-untested groupdelay module (issue #14).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import matplotlib

matplotlib.use("Agg")

from plot_antenna.groupdelay import compute_group_delay_dispersion


def _cut(freqs_hz, gd_seconds, col="S21(s)"):
    return pd.DataFrame({"! Stimulus(Hz)": freqs_hz, col: gd_seconds})


def test_dispersion_known_values():
    freqs = [2.40e9, 2.45e9, 2.50e9]
    # Three theta cuts; at each freq the GD values across cuts are known.
    data = {
        0: _cut(freqs, [5e-9, 6e-9, 7e-9]),
        90: _cut(freqs, [7e-9, 6e-9, 5e-9]),
        180: _cut(freqs, [6e-9, 6e-9, 6e-9]),
    }
    out = compute_group_delay_dispersion(data)
    assert list(out["freq_hz"]) == pytest.approx(freqs)
    # At 2.40 GHz the cut values are {5,7,6} ns -> spread 2 ns, var of [5,7,6]e-9.
    vals = np.array([5e-9, 7e-9, 6e-9])
    assert out["max_minus_min_s"][0] == pytest.approx(2e-9)
    assert out["variance_s2"][0] == pytest.approx(np.var(vals))
    assert out["std_s"][0] == pytest.approx(np.std(vals))
    # Distance error = spread * c.
    assert out["distance_error_m"][0] == pytest.approx(2e-9 * 299792458.0)


def test_dispersion_zero_when_all_cuts_identical():
    freqs = [2.40e9, 2.45e9]
    data = {0: _cut(freqs, [5e-9, 5e-9]), 90: _cut(freqs, [5e-9, 5e-9])}
    out = compute_group_delay_dispersion(data)
    assert np.allclose(out["variance_s2"], 0.0)
    assert np.allclose(out["std_s"], 0.0)
    assert np.allclose(out["max_minus_min_s"], 0.0)


def test_dispersion_band_filter():
    freqs = [2.0e9, 2.5e9, 3.0e9]
    data = {0: _cut(freqs, [5e-9, 5e-9, 5e-9]), 90: _cut(freqs, [6e-9, 6e-9, 6e-9])}
    out = compute_group_delay_dispersion(data, min_freq=2.4, max_freq=2.6)
    assert list(out["freq_hz"]) == pytest.approx([2.5e9])


def test_dispersion_accepts_s12_column():
    freqs = [5.0e9]
    data = {0: _cut(freqs, [3e-9], col="S12(s)"), 45: _cut(freqs, [4e-9], col="S12(s)")}
    out = compute_group_delay_dispersion(data)
    assert out["max_minus_min_s"][0] == pytest.approx(1e-9)


def test_dispersion_empty_input():
    out = compute_group_delay_dispersion({})
    assert out["freq_hz"].size == 0
    assert out["std_s"].size == 0
