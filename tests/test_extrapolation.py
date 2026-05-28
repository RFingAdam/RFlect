"""CI-runnable tests for plot_antenna.extrapolation (#38).

The data-gated integration tests skip without chamber files; these use synthetic
patterns with a known linear gain-vs-frequency trend so the extraction is locked
in CI. Also asserts calculations.py still re-exports the same objects.
"""

from __future__ import annotations

import numpy as np
import pytest

from plot_antenna.extrapolation import extrapolate_pattern, validate_extrapolation


def _synthetic(n_freqs=6, n_points=4, slope=0.01, base=900.0, step=20.0):
    """Build matched HPOL/VPOL entries whose magnitude rises linearly with freq."""
    theta = list(range(n_points))
    phi = [0] * n_points
    hpol, vpol = [], []
    for k in range(n_freqs):
        f = base + k * step
        mag = [(-10.0 + slope * (f - base) + 0.5 * i) for i in range(n_points)]
        phase = [0.0] * n_points
        entry = {"frequency": f, "theta": theta, "phi": phi, "mag": mag, "phase": phase}
        hpol.append(dict(entry))
        vpol.append(dict(entry))
    return hpol, vpol


class TestExtrapolatePattern:
    def test_too_few_points_raises(self):
        hpol, vpol = _synthetic(n_freqs=3)
        with pytest.raises(ValueError):
            extrapolate_pattern(hpol, vpol, 1100.0, min_frequencies=5)

    def test_linear_trend_recovered(self):
        hpol, vpol = _synthetic(n_freqs=6, slope=0.02)
        out = extrapolate_pattern(hpol, vpol, 1040.0, fit_degree=1)
        # At f=1040 (140 above base 900), point 0 mag = -10 + 0.02*140 = -7.2
        assert out["hpol"]["mag"][0] == pytest.approx(-7.2, abs=0.05)
        assert out["is_extrapolated"] is True
        assert out["confidence"]["mean_r_squared"] == pytest.approx(1.0, abs=1e-6)

    def test_confidence_quality_field_present(self):
        hpol, vpol = _synthetic()
        out = extrapolate_pattern(hpol, vpol, 1020.0)
        assert out["confidence"]["quality"] in {"good", "fair", "poor", "unreliable"}


class TestValidateExtrapolation:
    def test_holdout_recovers_low_error(self):
        hpol, vpol = _synthetic(n_freqs=6, slope=0.02)
        # Hold out an interior frequency; a clean linear trend should predict it well.
        res = validate_extrapolation(hpol, vpol, holdout_frequency=960.0, fit_degree=1)
        assert res["holdout_frequency"] == 960.0
        assert res["rms_error_dB"] < 0.5

    def test_missing_holdout_raises(self):
        hpol, vpol = _synthetic()
        with pytest.raises(ValueError):
            validate_extrapolation(hpol, vpol, holdout_frequency=12345.0)


def test_calculations_reexports_same_objects():
    from plot_antenna import calculations, extrapolation

    assert calculations.extrapolate_pattern is extrapolation.extrapolate_pattern
    assert calculations.validate_extrapolation is extrapolation.validate_extrapolation
