"""v5.2 RF-correctness tests: sidelobe main-lobe guard (#22) + exact Rician
fade margin (#23)."""

from __future__ import annotations

import numpy as np
import pytest

import matplotlib

matplotlib.use("Agg")

from plot_antenna.analysis_engine import AntennaAnalyzer
from plot_antenna.calculations import fade_margin_for_reliability


# ---------------------------------------------------------------------------
# #22 — sidelobe detection must not report main-lobe shoulders as sidelobes
# ---------------------------------------------------------------------------


def _detect(cut_angles, cut_gain, n=3):
    # _detect_sidelobes is an instance method but uses no instance state.
    az = AntennaAnalyzer.__new__(AntennaAnalyzer)
    return az._detect_sidelobes(np.asarray(cut_angles, float), np.asarray(cut_gain, float), n)


def test_shoulder_on_main_lobe_is_not_a_sidelobe():
    # Linear-angle cut (no wrap). Broad main lobe at the center index with a
    # shoulder ripple within 3 dB of the peak, plus a genuine sidelobe far out.
    angles = np.arange(0, 181, 10.0)  # 0..180
    gain = np.full(angles.shape, -25.0)
    # Main lobe around 90 deg with a shoulder at 70 deg (a local max but within
    # 3 dB of the 0 dB peak).
    gain[angles == 70] = -1.5  # shoulder (local max, inside main lobe)
    gain[angles == 80] = -2.0  # dip between shoulder and peak
    gain[angles == 90] = 0.0  # main-lobe peak
    gain[angles == 100] = -2.0
    gain[angles == 110] = -1.0  # shoulder on the other side, still within 3 dB
    gain[angles == 120] = -2.5
    gain[angles == 20] = -15.0  # a genuine sidelobe well outside the main lobe
    sl = _detect(angles, gain)
    # The true sidelobe at 20 deg should be found; no reported sidelobe may sit
    # within the main lobe (>= peak-3dB) region near the 90 deg peak.
    assert any(abs(s["angle_deg"] - 20.0) < 1e-6 for s in sl)
    for s in sl:
        assert s["gain_dBi"] < -3.0  # nothing within 3 dB of the 0 dB peak


def test_clean_two_lobe_pattern_reports_the_sidelobe():
    angles = np.arange(0, 360, 5.0)
    gain = np.full(angles.shape, -30.0)
    gain[angles == 0] = 0.0  # main lobe
    gain[angles == 90] = -12.0  # sidelobe at 90
    sl = _detect(angles, gain, n=3)
    assert len(sl) >= 1
    top = sl[0]
    assert top["angle_deg"] == pytest.approx(90.0)
    assert top["sll_dB"] == pytest.approx(-12.0, abs=0.01)


# ---------------------------------------------------------------------------
# #23 — exact Rician fade margin
# ---------------------------------------------------------------------------


def test_rician_reduces_to_rayleigh_at_k0():
    for rel in (90.0, 99.0, 99.9):
        ray = fade_margin_for_reliability(rel, fading="rayleigh")
        ric = fade_margin_for_reliability(rel, fading="rician", K=0.0)
        assert ric == pytest.approx(ray, abs=0.05), f"K=0 should match Rayleigh at {rel}%"


def test_rician_margin_decreases_with_higher_K():
    # More dominant LOS (higher K) => shallower fades => smaller required margin.
    rel = 99.0
    m_k1 = fade_margin_for_reliability(rel, fading="rician", K=1.0)
    m_k5 = fade_margin_for_reliability(rel, fading="rician", K=5.0)
    m_k20 = fade_margin_for_reliability(rel, fading="rician", K=20.0)
    assert m_k1 > m_k5 > m_k20 > 0.0


def test_higher_reliability_needs_more_margin():
    m90 = fade_margin_for_reliability(90.0, fading="rician", K=3.0)
    m999 = fade_margin_for_reliability(99.9, fading="rician", K=3.0)
    assert m999 > m90
