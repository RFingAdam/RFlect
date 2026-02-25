"""Integration and dispatcher tests for advanced analysis wiring."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

import plot_antenna.calculations as calculations
import plot_antenna.file_utils as file_utils
import plot_antenna.plotting as plotting
from plot_antenna.gui.callbacks_mixin import CallbacksMixin
from plot_antenna.gui.dialogs_mixin import DialogsMixin


class DummyVar:
    """Simple stand-in for Tkinter variables in unit tests."""

    def __init__(self, value):
        self._value = value

    def get(self):
        return self._value

    def set(self, value):
        self._value = value


class _CallbacksHarness(CallbacksMixin):
    """Lightweight harness to call callback logic without a GUI."""

    def __init__(self):
        self.TRP_file_path = "dummy_trp.txt"
        self.interpolate_3d_plots = False
        self.axis_scale_mode = DummyVar("auto")
        self.axis_min = DummyVar(-20.0)
        self.axis_max = DummyVar(6.0)
        self.maritime_plots_enabled = False
        self.horizon_theta_min = DummyVar(60.0)
        self.horizon_theta_max = DummyVar(120.0)
        self.horizon_gain_threshold = DummyVar(-3.0)
        self.horizon_theta_cuts_var = DummyVar("60,90,120")

        # Advanced analysis toggles
        self.link_budget_enabled = True
        self.indoor_analysis_enabled = True
        self.fading_analysis_enabled = True
        self.mimo_analysis_enabled = True
        self.wearable_analysis_enabled = False

        # Link budget / indoor
        self.lb_tx_power = DummyVar(3.0)
        self.lb_rx_sensitivity = DummyVar(-96.0)
        self.lb_rx_gain = DummyVar(0.5)
        self.lb_path_loss_exp = DummyVar(2.2)
        self.lb_misc_loss = DummyVar(8.0)
        self.lb_target_range = DummyVar(15.0)
        self.indoor_environment = DummyVar("Office")
        self.indoor_path_loss_exp = DummyVar(3.1)
        self.indoor_num_walls = DummyVar(2)
        self.indoor_wall_material = DummyVar("drywall")
        self.indoor_shadow_fading = DummyVar(4.5)
        self.indoor_max_distance = DummyVar(25.0)

        # Fading / MIMO
        self.fading_target_reliability = DummyVar(98.0)
        self.fading_model = DummyVar("rician")
        self.fading_rician_k = DummyVar(7.5)
        self.fading_realizations = DummyVar(321)
        self.mimo_snr = DummyVar(16.0)
        self.mimo_fading_model = DummyVar("rician")
        self.mimo_rician_k = DummyVar(6.0)
        self.mimo_xpr = DummyVar(4.0)

        # Wearable values are always collected even if disabled
        self.wearable_positions_var = {
            "wrist": DummyVar(True),
            "chest": DummyVar(False),
            "hip": DummyVar(False),
            "head": DummyVar(False),
        }
        self.wearable_tx_power_mw = DummyVar(1.0)
        self.wearable_device_count = DummyVar(10)
        self.wearable_room_x = DummyVar(10.0)
        self.wearable_room_y = DummyVar(8.0)
        self.wearable_room_z = DummyVar(3.0)

        self._measurement_context = {"processing_complete": False, "key_metrics": {}}

    def log_message(self, message: str, level: str = "info") -> None:
        _ = (message, level)


class _DialogHarness(DialogsMixin):
    """Harness for trace cleanup tests."""


class _TraceVar:
    """Trace var stub that records removals."""

    def __init__(self):
        self.removals = []

    def trace_remove(self, mode, trace_id):
        self.removals.append((mode, trace_id))


def _simple_grid(n_theta=7, n_phi=13, value=0.0):
    theta = np.linspace(0, 180, n_theta)
    phi = np.linspace(0, 360, n_phi)
    gain = np.full((n_theta, n_phi), value, dtype=float)
    return theta, phi, gain


class _AxisStub:
    """Minimal matplotlib axis stub for non-GUI plotting tests."""

    def semilogy(self, *args, **kwargs): ...
    def axhline(self, *args, **kwargs): ...
    def set_xlabel(self, *args, **kwargs): ...
    def set_ylabel(self, *args, **kwargs): ...
    def set_title(self, *args, **kwargs): ...
    def legend(self, *args, **kwargs): ...
    def grid(self, *args, **kwargs): ...
    def set_ylim(self, *args, **kwargs): ...
    def plot(self, *args, **kwargs): ...
    def axvline(self, *args, **kwargs): ...
    def annotate(self, *args, **kwargs): ...
    def fill_between(self, *args, **kwargs): ...
    def bar(self, *args, **kwargs): ...


class _FigureStub:
    """Minimal matplotlib figure stub for non-GUI plotting tests."""

    def suptitle(self, *args, **kwargs): ...

    def savefig(self, path, *args, **kwargs):
        Path(path).write_text("stub")


def test_generate_advanced_analysis_plots_all_enabled_creates_expected_artifacts(
    monkeypatch, tmp_path
):
    theta, phi, gain_2d = _simple_grid()

    def _artifact_stub(filename):
        def _fn(*args, save_path=None, **kwargs):
            Path(save_path, filename).write_text("ok")

        return _fn

    monkeypatch.setattr(
        plotting,
        "plot_link_budget_summary",
        _artifact_stub("link_budget_2450.0MHz.png"),
    )
    monkeypatch.setattr(
        plotting,
        "plot_indoor_coverage_map",
        _artifact_stub("indoor_coverage_2450.0MHz.png"),
    )
    monkeypatch.setattr(
        plotting,
        "plot_fading_analysis",
        _artifact_stub("fading_analysis_2450.0MHz.png"),
    )
    monkeypatch.setattr(
        plotting,
        "plot_mimo_analysis",
        _artifact_stub("mimo_analysis.png"),
    )
    monkeypatch.setattr(
        plotting,
        "plot_wearable_assessment",
        _artifact_stub("wearable_assessment_2450.0MHz.png"),
    )

    plotting.generate_advanced_analysis_plots(
        theta,
        phi,
        gain_2d,
        2450.0,
        data_label="Gain",
        data_unit="dBi",
        save_path=str(tmp_path),
        link_budget_enabled=True,
        indoor_enabled=True,
        fading_enabled=True,
        fading_model="rician",
        fading_rician_k=6.0,
        fading_realizations=80,
        mimo_enabled=True,
        mimo_gain_data_list=[gain_2d],
        wearable_enabled=True,
        wearable_body_positions=["wrist"],
    )

    expected = {
        "link_budget_2450.0MHz.png",
        "indoor_coverage_2450.0MHz.png",
        "fading_analysis_2450.0MHz.png",
        "mimo_analysis.png",
        "wearable_assessment_2450.0MHz.png",
    }
    generated = {p.name for p in tmp_path.iterdir() if p.is_file()}
    assert expected.issubset(generated)


@pytest.mark.parametrize(
    "model, expected_outage_fn",
    [
        ("rayleigh", "rayleigh"),
        ("rician", "rician"),
    ],
)
def test_plot_fading_analysis_uses_selected_runtime_model(
    model, expected_outage_fn, monkeypatch, tmp_path
):
    theta, phi, gain_2d = _simple_grid()
    calls = {"apply": [], "rayleigh": [], "rician": []}

    def _fake_apply(g2d, theta_deg, phi_deg, fading="rayleigh", K=10, realizations=1000):
        calls["apply"].append((fading, K, realizations, g2d.shape))
        return np.zeros_like(g2d), np.ones_like(g2d), np.zeros_like(g2d)

    def _fake_rayleigh(power_db, mean_power_db=0.0):
        shape = np.asarray(power_db).shape
        calls["rayleigh"].append(shape)
        return np.full(shape, 0.5, dtype=float)

    def _fake_rician(power_db, mean_power_db=0.0, K_factor=10.0):
        shape = np.asarray(power_db).shape
        calls["rician"].append(shape)
        return np.full(shape, 0.5, dtype=float)

    monkeypatch.setattr(plotting, "apply_statistical_fading", _fake_apply)
    monkeypatch.setattr(plotting, "rayleigh_cdf", _fake_rayleigh)
    monkeypatch.setattr(plotting, "rician_cdf", _fake_rician)
    monkeypatch.setattr(
        plotting.plt,
        "subplots",
        lambda *args, **kwargs: (
            _FigureStub(),
            np.array([[_AxisStub(), _AxisStub()], [_AxisStub(), _AxisStub()]], dtype=object),
        ),
    )
    monkeypatch.setattr(plotting.plt, "tight_layout", lambda *args, **kwargs: None)
    monkeypatch.setattr(plotting.plt, "close", lambda *args, **kwargs: None)
    monkeypatch.setattr(plotting.plt, "show", lambda *args, **kwargs: None)

    plotting.plot_fading_analysis(
        2450.0,
        gain_2d,
        theta,
        phi,
        fading_model=model,
        fading_rician_k=5.5,
        realizations=111,
        save_path=str(tmp_path),
    )

    assert calls["apply"], "apply_statistical_fading should be invoked"
    fading_name, k_used, n_used, shape_used = calls["apply"][0]
    assert fading_name == model
    assert shape_used == (1, len(phi))
    assert n_used == 111
    if model == "rician":
        assert abs(k_used - 5.5) < 1e-12

    # The outage branch uses horizon-size arrays (len(phi),).
    assert (len(phi),) in calls[expected_outage_fn]


def test_generate_advanced_analysis_plots_invokes_mimo_with_runtime_controls(monkeypatch):
    theta, phi, gain_2d = _simple_grid()
    captured = {}

    def _spy_mimo(*args, **kwargs):
        captured["args"] = args
        captured["kwargs"] = kwargs

    monkeypatch.setattr(plotting, "plot_mimo_analysis", _spy_mimo)

    plotting.generate_advanced_analysis_plots(
        theta,
        phi,
        gain_2d,
        5800.0,
        mimo_enabled=True,
        mimo_snr_db=18.0,
        mimo_fading_model="rician",
        mimo_rician_k=4.0,
        mimo_xpr_db=7.0,
        mimo_ecc_values=[0.11, 0.22],
        mimo_gain_data_list=[gain_2d],
    )

    assert "kwargs" in captured
    assert captured["kwargs"]["snr_db"] == 18.0
    assert captured["kwargs"]["fading"] == "rician"
    assert captured["kwargs"]["K"] == 4.0
    assert captured["kwargs"]["xpr_db"] == 7.0
    assert captured["kwargs"]["ecc_values"] == [0.11, 0.22]
    assert len(captured["kwargs"]["gain_data_list"]) == 1


def test_callbacks_active_forwarding_passes_full_advanced_params(monkeypatch):
    harness = _CallbacksHarness()
    theta = np.array([0.0, 90.0, 180.0])
    phi = np.array([0.0, 180.0, 360.0])
    theta_rad = np.deg2rad(theta)
    phi_rad = np.deg2rad(phi)
    grid = np.zeros((len(theta), len(phi)))

    monkeypatch.setattr(
        plotting,
        "plot_active_2d_data",
        lambda *args, **kwargs: None,
    )
    monkeypatch.setattr(
        plotting,
        "plot_active_3d_data",
        lambda *args, **kwargs: None,
    )
    monkeypatch.setattr(
        plotting,
        "generate_maritime_plots",
        lambda *args, **kwargs: None,
    )

    captured = {}

    def _spy_advanced(*args, **kwargs):
        captured["args"] = args
        captured["kwargs"] = kwargs

    monkeypatch.setattr(
        "plot_antenna.gui.callbacks_mixin.generate_advanced_analysis_plots",
        _spy_advanced,
    )
    monkeypatch.setattr(
        "plot_antenna.gui.callbacks_mixin.plot_active_2d_data",
        lambda *args, **kwargs: None,
    )
    monkeypatch.setattr(
        "plot_antenna.gui.callbacks_mixin.plot_active_3d_data",
        lambda *args, **kwargs: None,
    )
    monkeypatch.setattr(
        "plot_antenna.gui.callbacks_mixin.generate_maritime_plots",
        lambda *args, **kwargs: None,
    )

    monkeypatch.setattr(
        "plot_antenna.gui.callbacks_mixin.read_active_file",
        lambda _path: {
            "Frequency": 2450.0,
            "Start Phi": 0.0,
            "Start Theta": 0.0,
            "Stop Phi": 360.0,
            "Stop Theta": 180.0,
            "Inc Phi": 180.0,
            "Inc Theta": 90.0,
            "Calculated TRP(dBm)": 0.0,
            "Theta_Angles_Deg": theta,
            "Phi_Angles_Deg": phi,
            "H_Power_dBm": grid,
            "V_Power_dBm": grid,
        },
    )
    monkeypatch.setattr(
        "plot_antenna.gui.callbacks_mixin.calculate_active_variables",
        lambda *args, **kwargs: (
            9,
            theta,
            phi,
            theta_rad,
            phi_rad,
            grid,
            grid,
            grid,
            phi,
            phi_rad,
            grid,
            grid,
            grid,
            -1.0,
            1.0,
            -1.0,
            1.0,
            -1.0,
            1.0,
            0.0,
            0.0,
            0.0,
        ),
    )

    harness._process_active_data()
    assert "kwargs" in captured
    kwargs = captured["kwargs"]
    assert kwargs["fading_model"] == "rician"
    assert kwargs["fading_rician_k"] == 7.5
    assert kwargs["fading_realizations"] == 321
    assert kwargs["mimo_enabled"] is True
    assert kwargs["mimo_fading_model"] == "rician"
    assert kwargs["mimo_rician_k"] == 6.0
    assert kwargs["mimo_xpr_db"] == 4.0
    assert len(kwargs["mimo_gain_data_list"]) == 1


def test_batch_passive_forwarding_injects_mimo_gain_data(monkeypatch, tmp_path):
    theta, phi, grid = _simple_grid(n_theta=3, n_phi=3)
    base_dir = tmp_path / "in"
    base_dir.mkdir()
    out_dir = tmp_path / "out"

    monkeypatch.setattr(file_utils.os, "listdir", lambda _path: ["DemoAP_HPol.txt", "DemoAP_VPol.txt"])
    monkeypatch.setattr(
        file_utils,
        "read_passive_file",
        lambda _path: ([{"frequency": 2450.0}], 0, 360, 180, 0, 180, 90),
    )
    monkeypatch.setattr(file_utils, "angles_match", lambda *args, **kwargs: True)
    monkeypatch.setattr(
        file_utils,
        "calculate_passive_variables",
        lambda *args, **kwargs: (theta, phi, grid, grid, grid),
    )

    monkeypatch.setattr(plotting, "plot_2d_passive_data", lambda *args, **kwargs: None)
    monkeypatch.setattr(plotting, "plot_passive_3d_component", lambda *args, **kwargs: None)
    monkeypatch.setattr(plotting, "_prepare_gain_grid", lambda *args, **kwargs: (theta, phi, grid))

    captured = {}

    def _spy_advanced(*args, **kwargs):
        captured["kwargs"] = kwargs

    monkeypatch.setattr(plotting, "generate_advanced_analysis_plots", _spy_advanced)

    file_utils.batch_process_passive_scans(
        folder_path=str(base_dir),
        freq_list=[2450.0],
        selected_frequencies=[2450.0],
        save_base=str(out_dir),
        advanced_analysis_params={
            "fading_enabled": True,
            "fading_model": "rician",
            "fading_rician_k": 4.0,
            "fading_realizations": 77,
            "mimo_enabled": True,
        },
    )

    assert "kwargs" in captured
    kwargs = captured["kwargs"]
    assert kwargs["fading_model"] == "rician"
    assert kwargs["fading_rician_k"] == 4.0
    assert kwargs["fading_realizations"] == 77
    assert kwargs["mimo_enabled"] is True
    assert "mimo_gain_data_list" in kwargs
    assert len(kwargs["mimo_gain_data_list"]) == 1
    assert np.array_equal(kwargs["mimo_gain_data_list"][0], grid)


def test_batch_active_forwarding_injects_mimo_gain_data(monkeypatch, tmp_path):
    theta = np.array([0.0, 90.0, 180.0])
    phi = np.array([0.0, 180.0, 360.0])
    phi_rad = np.deg2rad(phi)
    grid = np.zeros((len(theta), len(phi)))
    base_dir = tmp_path / "in"
    base_dir.mkdir()
    out_dir = tmp_path / "out"

    monkeypatch.setattr(file_utils.os, "listdir", lambda _path: ["Demo_TRP.txt"])
    monkeypatch.setattr(
        file_utils,
        "read_active_file",
        lambda _path: {
            "Frequency": 2450.0,
            "Start Phi": 0.0,
            "Start Theta": 0.0,
            "Stop Phi": 360.0,
            "Stop Theta": 180.0,
            "Inc Phi": 180.0,
            "Inc Theta": 90.0,
            "H_Power_dBm": grid,
            "V_Power_dBm": grid,
        },
    )
    monkeypatch.setattr(
        calculations,
        "calculate_active_variables",
        lambda *args, **kwargs: (
            9,
            theta,
            phi,
            np.deg2rad(theta),
            phi_rad,
            grid,
            grid,
            grid,
            phi,
            phi_rad,
            grid,
            grid,
            grid,
            -1.0,
            1.0,
            -1.0,
            1.0,
            -1.0,
            1.0,
            0.0,
            0.0,
            0.0,
        ),
    )

    monkeypatch.setattr(plotting, "plot_active_2d_data", lambda *args, **kwargs: None)
    monkeypatch.setattr(plotting, "plot_active_3d_data", lambda *args, **kwargs: None)

    captured = {}

    def _spy_advanced(*args, **kwargs):
        captured["kwargs"] = kwargs

    monkeypatch.setattr(plotting, "generate_advanced_analysis_plots", _spy_advanced)

    file_utils.batch_process_active_scans(
        folder_path=str(base_dir),
        save_base=str(out_dir),
        advanced_analysis_params={
            "fading_enabled": True,
            "fading_model": "rician",
            "fading_rician_k": 8.0,
            "fading_realizations": 66,
            "mimo_enabled": True,
        },
    )

    assert "kwargs" in captured
    kwargs = captured["kwargs"]
    assert kwargs["fading_model"] == "rician"
    assert kwargs["fading_rician_k"] == 8.0
    assert kwargs["fading_realizations"] == 66
    assert kwargs["mimo_enabled"] is True
    assert "mimo_gain_data_list" in kwargs
    assert len(kwargs["mimo_gain_data_list"]) == 1
    assert np.array_equal(kwargs["mimo_gain_data_list"][0], grid)


def test_batch_active_summary_tracks_per_file_failures(monkeypatch, tmp_path):
    base_dir = tmp_path / "in"
    base_dir.mkdir()

    monkeypatch.setattr(file_utils.os, "listdir", lambda _path: ["Broken_TRP.txt"])
    monkeypatch.setattr(
        file_utils,
        "read_active_file",
        lambda _path: (_ for _ in ()).throw(ValueError("malformed active file")),
    )

    summary = file_utils.batch_process_active_scans(folder_path=str(base_dir))

    assert summary["total_files"] == 1
    assert summary["processed"] == 0
    assert summary["failed"] == 1
    assert len(summary["errors"]) == 1
    assert summary["errors"][0]["file"] == "Broken_TRP.txt"


def test_batch_passive_summary_tracks_pair_read_failures(monkeypatch, tmp_path):
    base_dir = tmp_path / "in"
    base_dir.mkdir()

    monkeypatch.setattr(file_utils.os, "listdir", lambda _path: ["DemoAP_HPol.txt", "DemoAP_VPol.txt"])
    monkeypatch.setattr(
        file_utils,
        "read_passive_file",
        lambda _path: (_ for _ in ()).throw(ValueError("malformed passive file")),
    )

    summary = file_utils.batch_process_passive_scans(
        folder_path=str(base_dir),
        freq_list=[2450.0, 5800.0],
        selected_frequencies=[2450.0, 5800.0],
    )

    assert summary["total_pairs"] == 1
    assert summary["total_jobs"] == 2
    assert summary["processed"] == 0
    assert summary["failed"] == 2
    assert summary["skipped"] == 0
    assert len(summary["errors"]) == 1
    assert summary["errors"][0]["pair"].endswith("DemoAP_HPol.txt | DemoAP_VPol.txt")


def test_cleanup_advanced_analysis_traces_clears_registered_handlers():
    harness = _DialogHarness()
    trace_var = _TraceVar()

    harness._advanced_trace_handles = [
        (trace_var, "write", "trace_a"),
        (trace_var, "write", "trace_b"),
    ]
    harness._cleanup_advanced_analysis_traces()
    assert trace_var.removals == [("write", "trace_a"), ("write", "trace_b")]
    assert harness._advanced_trace_handles == []

    harness._advanced_trace_handles = [(trace_var, "write", "trace_c")]
    harness._cleanup_advanced_analysis_traces()
    assert trace_var.removals[-1] == ("write", "trace_c")
    assert harness._advanced_trace_handles == []
