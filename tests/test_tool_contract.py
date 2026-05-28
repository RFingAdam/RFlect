"""Return-contract tests for the string-contract MCP tool modules (#10).

analysis_tools and bulk_tools follow the human-readable *string* contract: every
tool returns a ``str`` on success and on failure, and NEVER raises. This module
locks that behaviour on the error/precondition paths, which need no chamber data
and therefore run in CI (unlike test_mcp_integration.py, which is data-gated).

It deliberately avoids importing the ``mcp`` package: the analysis tools are
module-level functions, and the bulk tools are captured via a tiny fake-mcp
shim, so the contract is verified even where FastMCP is not installed.
"""

from __future__ import annotations

import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "rflect-mcp"))


@pytest.fixture(autouse=True)
def _no_data_loaded():
    """Ensure the shared measurement store is empty for the error-path tests."""
    from tools.import_tools import _loaded_measurements

    saved = dict(_loaded_measurements)
    _loaded_measurements.clear()
    try:
        yield
    finally:
        _loaded_measurements.clear()
        _loaded_measurements.update(saved)


def _looks_like_failure(text: str) -> bool:
    """A failure is detectable by substring under the two-tier convention:
    an ``Error:`` prefix (exceptions) or a known precondition sentence."""
    low = text.lower()
    return (
        "error" in low
        or "no data" in low
        or "not found" in low
        or "no frequency" in low
        or "could not" in low
        or "not a valid directory" in low
        or "no hpol" in low
        or "no trp" in low
    )


class TestAnalysisToolsContract:
    """Module-level analysis tools, called directly with no data loaded."""

    def test_list_frequencies_no_data(self):
        from tools.analysis_tools import list_frequencies

        out = list_frequencies()
        assert isinstance(out, str)
        assert _looks_like_failure(out)

    def test_analyze_pattern_no_data(self):
        from tools.analysis_tools import analyze_pattern

        out = analyze_pattern()
        assert isinstance(out, str)
        assert _looks_like_failure(out)

    def test_gain_statistics_no_data(self):
        from tools.analysis_tools import get_gain_statistics

        out = get_gain_statistics()
        assert isinstance(out, str)
        assert _looks_like_failure(out)

    def test_compare_polarizations_no_data(self):
        from tools.analysis_tools import compare_polarizations

        out = compare_polarizations()
        assert isinstance(out, str)
        assert _looks_like_failure(out)

    def test_horizon_statistics_no_data(self):
        from tools.analysis_tools import get_horizon_statistics

        out = get_horizon_statistics()
        assert isinstance(out, str)
        assert _looks_like_failure(out)

    def test_extrapolate_missing_file(self):
        from tools.analysis_tools import extrapolate_to_frequency

        out = extrapolate_to_frequency(
            hpol_file="/nonexistent/does_not_exist_AP_HPol.txt",
            vpol_file="/nonexistent/does_not_exist_AP_VPol.txt",
            target_frequency=2450.0,
        )
        assert isinstance(out, str)
        assert _looks_like_failure(out)


class _FakeMCP:
    """Captures functions registered via ``@mcp.tool()`` without needing FastMCP."""

    def __init__(self):
        self.tools = {}

    def tool(self, *args, **kwargs):
        def deco(fn):
            self.tools[fn.__name__] = fn
            return fn

        return deco

    def resource(self, *args, **kwargs):
        def deco(fn):
            return fn

        return deco


@pytest.fixture(scope="module")
def bulk_tools():
    from tools.bulk_tools import register_bulk_tools

    mcp = _FakeMCP()
    register_bulk_tools(mcp)
    return mcp.tools


class TestBulkToolsContract:
    """Bulk tools, captured via the fake-mcp shim, exercised on error paths."""

    def test_bulk_process_passive_bad_dir(self, bulk_tools):
        out = bulk_tools["bulk_process_passive"]("/nonexistent/dir_xyz")
        assert isinstance(out, str)
        assert _looks_like_failure(out)

    def test_bulk_process_active_bad_dir(self, bulk_tools):
        out = bulk_tools["bulk_process_active"]("/nonexistent/dir_xyz")
        assert isinstance(out, str)
        assert _looks_like_failure(out)

    def test_validate_file_pair_missing(self, bulk_tools):
        out = bulk_tools["validate_file_pair"]("/no/hpol.txt", "/no/vpol.txt")
        assert isinstance(out, str)
        assert _looks_like_failure(out)

    def test_convert_to_cst_missing_files(self, bulk_tools):
        out = bulk_tools["convert_to_cst"](
            hpol_path="/no/hpol.txt",
            vpol_path="/no/vpol.txt",
            vswr_path="/no/vswr.csv",
            frequency=2450.0,
        )
        assert isinstance(out, str)
        assert _looks_like_failure(out)

    def test_list_measurement_files_bad_dir(self, bulk_tools):
        out = bulk_tools["list_measurement_files"]("/nonexistent/dir_xyz")
        assert isinstance(out, str)
        assert _looks_like_failure(out)
