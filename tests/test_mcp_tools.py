"""
Tests for RFlect MCP Server tools

Tests the MCP dataclasses (LoadedMeasurement, ReportOptions)
and standalone helper functions (_fmt) from the rflect-mcp tools package.

The rflect-mcp directory uses sys.path manipulation and relative imports
internally, so we add it to sys.path and import the tools as a package.
This requires plot_antenna to be importable (editable install).
"""

import sys
import os
import pytest

# Add rflect-mcp to path so we can import the tools package
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "rflect-mcp"))


class TestLoadedMeasurement:
    """Test the LoadedMeasurement dataclass from import_tools."""

    def test_dataclass_creation(self):
        """Verify LoadedMeasurement can be instantiated with required fields."""
        from tools.import_tools import LoadedMeasurement

        m = LoadedMeasurement(
            file_path="test.csv",
            scan_type="passive",
            frequencies=[2400.0],
            data={"key": "val"},
        )
        assert m.file_path == "test.csv"

    def test_dataclass_fields(self):
        """Verify all fields are accessible and store correct values."""
        from tools.import_tools import LoadedMeasurement

        m = LoadedMeasurement(
            file_path="a.csv",
            scan_type="active",
            frequencies=[2400.0, 2500.0],
            data={},
        )
        assert m.scan_type == "active"
        assert len(m.frequencies) == 2


class TestReportOptions:
    """Test the ReportOptions dataclass from report_tools."""

    def test_default_options(self):
        """Verify ReportOptions defaults match expected values."""
        from tools.report_tools import ReportOptions

        opts = ReportOptions()
        assert opts.frequencies is None
        assert opts.include_2d_plots is True
        assert opts.include_3d_plots is False
        assert opts.ai_model == "gpt-4o-mini"

    def test_custom_options(self):
        """Verify ReportOptions can be overridden with custom values."""
        from tools.report_tools import ReportOptions

        opts = ReportOptions(frequencies=[2400.0], include_3d_plots=True)
        assert opts.frequencies == [2400.0]
        assert opts.include_3d_plots is True


class TestFmtHelper:
    """Test the _fmt helper function from analysis_tools."""

    def test_fmt_float(self):
        """Verify _fmt formats a float to 2 decimal places by default."""
        from tools.analysis_tools import _fmt

        assert _fmt(10.123) == "10.12"

    def test_fmt_none(self):
        """Verify _fmt returns 'N/A' for None input."""
        from tools.analysis_tools import _fmt

        assert _fmt(None) == "N/A"

    def test_fmt_string_na(self):
        """Verify _fmt passes through 'N/A' string unchanged."""
        from tools.analysis_tools import _fmt

        assert _fmt("N/A") == "N/A"

    def test_fmt_custom_format(self):
        """Verify _fmt respects custom format specifier."""
        from tools.analysis_tools import _fmt

        assert _fmt(10.123, ".1f") == "10.1"


class TestActiveBulkImportProducesProcessedData:
    """Regression test for issue #63.

    Verifies that an active file loaded via the shared `_build_active_analyzer_data`
    helper (used by both `import_antenna_folder` and `import_active_processed`)
    populates the post-processed keys that report_tools._generate_active_plots needs.
    Without this, the report generator silently produces 0 embedded plots.
    """

    @staticmethod
    def _build_active_fixture(path):
        lines = [f"Header line {i}" for i in range(55)]
        lines[0] = "Howland Chamber Export V5.03"
        lines[5] = "Total Radiated Power Test"
        lines[13] = "Test Frequency: 2440 MHz"
        lines[15] = "Test Type: Semi-Discrete Angle Based Test"
        lines[31] = "Start Phi: 0 Deg"
        lines[32] = "Stop Phi: 345 Deg"
        lines[33] = "Inc Phi: 15 Deg"
        lines[38] = "Start Theta: 0 Deg"
        lines[39] = "Stop Theta: 165 Deg"
        lines[40] = "Inc Theta: 15 Deg"
        lines[46] = "H Cal Factor = 1.0 dB"
        lines[47] = "V Cal Factor = 2.0 dB"
        lines[49] = "Calculated TRP = -1.0 dBm"
        data_rows = []
        for ti in range(12):
            for pi in range(24):
                data_rows.append(f"{ti * 15} {pi * 15} -10 -20")
        path.write_text("\n".join(lines[:54] + data_rows) + "\n", encoding="utf-8")

    def test_helper_produces_all_processed_keys(self, tmp_path):
        from tools.import_tools import _build_active_analyzer_data

        fixture = tmp_path / "Test_TRP.txt"
        self._build_active_fixture(fixture)

        data, freq = _build_active_analyzer_data(str(fixture))

        assert freq == 2440.0
        for required in ("data_points", "theta_rad", "phi_rad_plot",
                         "total_power_2d_plot", "theta", "phi",
                         "total_power_2d", "phi_deg_plot"):
            assert required in data and data[required] is not None, \
                f"missing key {required!r} (would break report plot generation)"
        assert isinstance(data["TRP_dBm"], float)

    def test_import_antenna_file_active_stores_processed_data(self, tmp_path):
        from tools.import_tools import (
            register_import_tools,
            _loaded_measurements,
            _measurements_lock,
        )

        fixture = tmp_path / "BulkTest_TRP.txt"
        self._build_active_fixture(fixture)

        class _Capture:
            def __init__(self):
                self.tools = {}

            def tool(self):
                def _wrap(fn):
                    self.tools[fn.__name__] = fn
                    return fn
                return _wrap

        mcp = _Capture()
        register_import_tools(mcp)

        with _measurements_lock:
            _loaded_measurements.clear()

        result = mcp.tools["import_antenna_file"](str(fixture), scan_type="active")
        assert "Successfully imported" in result
        assert "TRP:" in result, "active import response should surface TRP value"

        name = os.path.basename(str(fixture))
        with _measurements_lock:
            assert name in _loaded_measurements
            stored = _loaded_measurements[name].data
            assert "total_power_2d_plot" in stored
            assert "TRP_dBm" in stored
            _loaded_measurements.clear()


class TestMCPHorizonStats:
    """Test the get_horizon_statistics MCP tool function registration."""

    def test_function_exists(self):
        """Verify get_horizon_statistics is importable from analysis_tools."""
        from tools.analysis_tools import get_horizon_statistics

        assert callable(get_horizon_statistics)

    def test_function_signature(self):
        """Verify get_horizon_statistics accepts expected parameters."""
        import inspect
        from tools.analysis_tools import get_horizon_statistics

        sig = inspect.signature(get_horizon_statistics)
        params = list(sig.parameters.keys())
        assert "frequency" in params
        assert "theta_min" in params
        assert "theta_max" in params
        assert "gain_threshold" in params
        assert "measurement_name" in params

    def test_no_data_returns_error(self):
        """Without loaded data, should return an error message."""
        from tools.analysis_tools import get_horizon_statistics

        result = get_horizon_statistics()
        assert "No data loaded" in result or "not found" in result
