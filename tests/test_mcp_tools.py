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
