"""Tests for the advanced-analysis config dataclasses + structured dispatcher (#37).

Verifies the flat-kwargs -> grouped-config bundling preserves every value
(including the deliberate cross-group reuse of link-budget params by the indoor
and fading branches) and that the structured dispatcher dispatches to exactly
the plot functions whose group is enabled.
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")

import numpy as np
import pytest

from plot_antenna.advanced_analysis_config import (
    AdvancedAnalysisConfig,
    FadingConfig,
    LinkBudgetConfig,
    MimoConfig,
    configs_from_flat_kwargs,
)


class TestConfigsFromFlatKwargs:
    def test_defaults_all_disabled(self):
        cfg = configs_from_flat_kwargs()
        assert cfg.link_budget.enabled is False
        assert cfg.indoor.enabled is False
        assert cfg.fading.enabled is False
        assert cfg.mimo.enabled is False
        assert cfg.wearable.enabled is False

    def test_values_routed_to_correct_groups(self):
        cfg = configs_from_flat_kwargs(
            link_budget_enabled=True,
            lb_pt_dbm=12.0,
            lb_target_range_m=7.5,
            fading_enabled=True,
            fading_model="rician",
            fading_rician_k=4.0,
            mimo_enabled=True,
            mimo_xpr_db=9.0,
        )
        assert cfg.link_budget.pt_dbm == 12.0
        assert cfg.link_budget.target_range_m == 7.5
        assert cfg.fading.model == "rician"
        assert cfg.fading.rician_k == 4.0
        assert cfg.mimo.xpr_db == 9.0

    def test_frozen(self):
        with pytest.raises(Exception):
            configs_from_flat_kwargs().link_budget.pt_dbm = 1.0  # type: ignore[misc]


class TestStructuredDispatcher:
    """The dispatcher should call only the enabled groups' plot functions."""

    def _grid(self):
        theta = np.linspace(0, 180, 19)
        phi = np.linspace(0, 350, 36)
        gain = np.zeros((theta.size, phi.size))
        return theta, phi, gain

    def test_dispatches_only_enabled_branches(self, monkeypatch):
        from plot_antenna import plotting

        called = []
        for name in (
            "plot_link_budget_summary",
            "plot_indoor_coverage_map",
            "plot_fading_analysis",
            "plot_mimo_analysis",
            "plot_wearable_assessment",
        ):
            monkeypatch.setattr(plotting, name, (lambda n: lambda *a, **k: called.append(n))(name))

        theta, phi, gain = self._grid()
        cfg = AdvancedAnalysisConfig(
            link_budget=LinkBudgetConfig(enabled=True),
            fading=FadingConfig(enabled=True),
        )
        plotting.dispatch_advanced_analysis_plots(theta, phi, gain, 2450.0, cfg)

        assert set(called) == {"plot_link_budget_summary", "plot_fading_analysis"}

    def test_flat_entrypoint_forwards_to_structured(self, monkeypatch):
        from plot_antenna import plotting

        called = []
        monkeypatch.setattr(plotting, "plot_mimo_analysis", lambda *a, **k: called.append("mimo"))
        theta, phi, gain = self._grid()
        plotting.generate_advanced_analysis_plots(theta, phi, gain, 2450.0, mimo_enabled=True)
        assert called == ["mimo"]
