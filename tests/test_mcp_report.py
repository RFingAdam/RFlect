"""
Tests for RFlect MCP Report Pipeline

Tests the branded DOCX report generation: plot generation, DOCX building,
gain tables, LLM provider fallback, and end-to-end report flow.
"""

import sys
import os
import tempfile
import shutil

import pytest
import numpy as np

# Add rflect-mcp to path so we can import the tools package
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'rflect-mcp'))


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def mock_passive_measurement():
    """Create a LoadedMeasurement with realistic passive data."""
    from tools.import_tools import LoadedMeasurement

    n_theta, n_phi = 19, 37
    theta = np.linspace(0, 180, n_theta)
    phi = np.linspace(0, 360, n_phi)

    np.random.seed(42)
    total_gain = np.random.randn(n_theta * n_phi, 3) * 5 + 2
    h_gain = np.random.randn(n_theta * n_phi, 3) * 5 + 0
    v_gain = np.random.randn(n_theta * n_phi, 3) * 5 + 0

    return LoadedMeasurement(
        file_path="test_hpol.txt + test_vpol.txt",
        scan_type="passive",
        frequencies=[2400.0, 2450.0, 2500.0],
        data={
            "theta": theta,
            "phi": phi,
            "total_gain": total_gain,
            "h_gain": h_gain,
            "v_gain": v_gain,
        },
    )


@pytest.fixture
def mock_active_measurement():
    """Create a LoadedMeasurement with realistic active data and plot arrays."""
    from tools.import_tools import LoadedMeasurement

    n_theta, n_phi = 13, 25
    theta_deg = np.linspace(0, 180, n_theta)
    phi_deg = np.linspace(0, 345, n_phi)
    theta_rad = np.deg2rad(theta_deg)
    phi_rad = np.deg2rad(phi_deg)

    np.random.seed(123)
    total_power_2d = np.random.randn(n_theta, n_phi) * 3 + 5
    h_power_2d = np.random.randn(n_theta, n_phi) * 3 + 3
    v_power_2d = np.random.randn(n_theta, n_phi) * 3 + 3

    # Extended arrays for 3D wrapping
    phi_deg_plot = np.append(phi_deg, 360)
    phi_rad_plot = np.deg2rad(phi_deg_plot)
    total_power_2d_plot = np.hstack((total_power_2d, total_power_2d[:, [0]]))
    h_power_2d_plot = np.hstack((h_power_2d, h_power_2d[:, [0]]))
    v_power_2d_plot = np.hstack((v_power_2d, v_power_2d[:, [0]]))

    return LoadedMeasurement(
        file_path="test_active.txt",
        scan_type="active",
        frequencies=[2450.0],
        data={
            "total_power": total_power_2d.flatten(),
            "h_power": h_power_2d.flatten(),
            "v_power": v_power_2d.flatten(),
            "TRP_dBm": 5.0,
            "H_TRP_dBm": 3.0,
            "V_TRP_dBm": 3.0,
            "theta": theta_deg,
            "phi": phi_deg,
            "data_points": n_theta * n_phi,
            "theta_rad": theta_rad,
            "phi_rad": phi_rad,
            "total_power_2d": total_power_2d,
            "h_power_2d": h_power_2d,
            "v_power_2d": v_power_2d,
            "phi_deg_plot": phi_deg_plot,
            "phi_rad_plot": phi_rad_plot,
            "total_power_2d_plot": total_power_2d_plot,
            "h_power_2d_plot": h_power_2d_plot,
            "v_power_2d_plot": v_power_2d_plot,
        },
    )


@pytest.fixture
def default_opts():
    """Create default ReportOptions with AI disabled for deterministic tests."""
    from tools.report_tools import ReportOptions

    return ReportOptions(
        ai_executive_summary=False,
        ai_section_analysis=False,
        ai_recommendations=False,
        include_2d_plots=False,
        include_3d_plots=False,
    )


@pytest.fixture
def temp_dir():
    """Create and clean up a temp directory for test outputs."""
    d = tempfile.mkdtemp(prefix="rflect_test_")
    yield d
    shutil.rmtree(d, ignore_errors=True)


@pytest.fixture
def _inject_passive(mock_passive_measurement):
    """Inject a passive measurement into the MCP global store."""
    from tools.import_tools import _loaded_measurements, _measurements_lock

    with _measurements_lock:
        _loaded_measurements["TestPassive"] = mock_passive_measurement
    yield
    with _measurements_lock:
        _loaded_measurements.pop("TestPassive", None)


@pytest.fixture
def _inject_active(mock_active_measurement):
    """Inject an active measurement into the MCP global store."""
    from tools.import_tools import _loaded_measurements, _measurements_lock

    with _measurements_lock:
        _loaded_measurements["TestActive"] = mock_active_measurement
    yield
    with _measurements_lock:
        _loaded_measurements.pop("TestActive", None)


# ---------------------------------------------------------------------------
# Tests: ReportOptions
# ---------------------------------------------------------------------------

class TestReportOptionsBranded:
    """Test ReportOptions dataclass with new fields."""

    def test_default_options(self):
        from tools.report_tools import ReportOptions

        opts = ReportOptions()
        assert opts.include_cover_page is True
        assert opts.include_table_of_contents is True
        assert opts.include_gain_tables is True
        assert opts.ai_model == "gpt-4o-mini"

    def test_custom_metadata_passthrough(self):
        from tools.report_tools import ReportOptions

        opts = ReportOptions(company_name="TestCo", logo_path="/tmp/logo.png")
        assert opts.company_name == "TestCo"
        assert opts.logo_path == "/tmp/logo.png"


# ---------------------------------------------------------------------------
# Tests: _fmt helper
# ---------------------------------------------------------------------------

class TestFmtReport:
    """Test the report-specific _fmt helper."""

    def test_fmt_float(self):
        from tools.report_tools import _fmt
        assert _fmt(10.123) == "10.12"

    def test_fmt_none(self):
        from tools.report_tools import _fmt
        assert _fmt(None) == "N/A"

    def test_fmt_with_suffix(self):
        from tools.report_tools import _fmt
        assert _fmt(10.0, ".1f", " dBi") == "10.0 dBi"

    def test_fmt_non_numeric(self):
        from tools.report_tools import _fmt
        assert _fmt("hello") == "hello"


# ---------------------------------------------------------------------------
# Tests: _create_llm_provider
# ---------------------------------------------------------------------------

class TestCreateLLMProvider:
    """Test LLM provider creation with graceful fallback."""

    def test_no_key_returns_none(self, monkeypatch):
        """Verify graceful None return when no API key is configured."""
        from tools.report_tools import _create_llm_provider, ReportOptions

        # Patch get_api_key to return None
        monkeypatch.setattr(
            "tools.report_tools.get_all_analysis",
            lambda *a, **kw: "",
        )
        # _create_llm_provider imports internally, so we need to patch at source
        import plot_antenna.api_keys as ak
        monkeypatch.setattr(ak, "get_api_key", lambda _: None)

        provider = _create_llm_provider(ReportOptions())
        assert provider is None


# ---------------------------------------------------------------------------
# Tests: _prepare_report_data
# ---------------------------------------------------------------------------

class TestPrepareReportData:
    """Test measurement filtering and frequency collection."""

    def test_all_measurements(self, mock_passive_measurement):
        from tools.report_tools import _prepare_report_data, ReportOptions

        measurements = {"passive1": mock_passive_measurement}
        opts = ReportOptions()
        data = _prepare_report_data(measurements, opts)

        assert "passive1" in data["measurements"]
        assert data["frequencies"] == [2400.0, 2450.0, 2500.0]

    def test_frequency_filter(self, mock_passive_measurement):
        from tools.report_tools import _prepare_report_data, ReportOptions

        measurements = {"passive1": mock_passive_measurement}
        opts = ReportOptions(frequencies=[2450.0])
        data = _prepare_report_data(measurements, opts)

        assert data["frequencies"] == [2450.0]

    def test_measurement_filter(self, mock_passive_measurement, mock_active_measurement):
        from tools.report_tools import _prepare_report_data, ReportOptions

        measurements = {
            "passive1": mock_passive_measurement,
            "active1": mock_active_measurement,
        }
        opts = ReportOptions(measurements=["active1"])
        data = _prepare_report_data(measurements, opts)

        assert data["measurements"] == ["active1"]
        assert "passive1" not in data["measurements"]

    def test_empty_measurements(self):
        from tools.report_tools import _prepare_report_data, ReportOptions

        data = _prepare_report_data({}, ReportOptions())
        assert data["measurements"] == []
        assert data["frequencies"] == []


# ---------------------------------------------------------------------------
# Tests: _filter_frequencies
# ---------------------------------------------------------------------------

class TestFilterFrequencies:

    def test_no_filter(self):
        from tools.report_tools import _filter_frequencies, ReportOptions
        opts = ReportOptions(frequencies=None)
        assert _filter_frequencies([1.0, 2.0, 3.0], opts) == [1.0, 2.0, 3.0]

    def test_subset(self):
        from tools.report_tools import _filter_frequencies, ReportOptions
        opts = ReportOptions(frequencies=[2.0])
        assert _filter_frequencies([1.0, 2.0, 3.0], opts) == [2.0]


# ---------------------------------------------------------------------------
# Tests: _safe_filename
# ---------------------------------------------------------------------------

class TestSafeFilename:

    def test_safe_chars_preserved(self):
        from tools.report_tools import _safe_filename
        assert _safe_filename("BLE_AP_Test") == "BLE_AP_Test"

    def test_special_chars_replaced(self):
        from tools.report_tools import _safe_filename
        result = _safe_filename("test/path:file")
        assert "/" not in result
        assert ":" not in result


# ---------------------------------------------------------------------------
# Tests: _add_gain_stats_table
# ---------------------------------------------------------------------------

class TestGainStatsTable:

    def test_passive_table(self):
        from docx import Document
        from docx.shared import RGBColor
        from tools.report_tools import _add_gain_stats_table

        doc = Document()
        stats = {
            "scan_type": "passive",
            "frequency_actual": 2450.0,
            "max_gain_dBi": 5.12,
            "min_gain_dBi": -10.5,
            "avg_gain_dBi": 1.23,
            "std_dev_dB": 3.45,
            "max_hpol_gain_dBi": 4.0,
            "max_vpol_gain_dBi": 3.5,
        }
        _add_gain_stats_table(doc, stats, RGBColor(50, 50, 50))

        # Should have one table with 8 rows (header + 5 base + 2 pol)
        assert len(doc.tables) == 1
        table = doc.tables[0]
        assert table.rows[0].cells[0].text == "Parameter"
        assert "5.12" in table.rows[2].cells[1].text

    def test_active_table_with_trp(self):
        from docx import Document
        from docx.shared import RGBColor
        from tools.report_tools import _add_gain_stats_table

        doc = Document()
        stats = {
            "scan_type": "active",
            "max_power_dBm": 10.0,
            "min_power_dBm": -5.0,
            "avg_power_dBm": 3.0,
            "std_dev_dB": 2.0,
            "TRP_dBm": 5.5,
        }
        _add_gain_stats_table(doc, stats, RGBColor(50, 50, 50))

        table = doc.tables[0]
        # Should have TRP row
        last_row = table.rows[-1]
        assert "TRP" in last_row.cells[0].text
        assert "5.50" in last_row.cells[1].text


# ---------------------------------------------------------------------------
# Tests: _build_branded_docx (no AI, no plots)
# ---------------------------------------------------------------------------

class TestBuildBrandedDocx:

    def test_docx_no_ai_no_plots(self, mock_passive_measurement, temp_dir, default_opts):
        """Verify DOCX is created with correct sections when AI is off."""
        from tools.report_tools import _build_branded_docx, _prepare_report_data
        from tools.import_tools import _loaded_measurements, _measurements_lock

        measurements = {"TestPassive": mock_passive_measurement}
        with _measurements_lock:
            _loaded_measurements["TestPassive"] = mock_passive_measurement

        try:
            report_data = _prepare_report_data(measurements, default_opts)
            output = os.path.join(temp_dir, "test_report.docx")

            _build_branded_docx(
                output, report_data, {},
                default_opts, None, None, measurements,
            )

            assert os.path.exists(output)
            assert os.path.getsize(output) > 0

            # Verify DOCX structure
            from docx import Document
            doc = Document(output)
            text = "\n".join(p.text for p in doc.paragraphs)

            assert "Antenna Radiation Pattern Test Report" in text
            assert "Test Configuration" in text
            assert "Conclusions and Recommendations" in text
        finally:
            with _measurements_lock:
                _loaded_measurements.pop("TestPassive", None)

    def test_docx_with_metadata(self, mock_passive_measurement, temp_dir, default_opts):
        """Verify metadata appears on cover page."""
        from tools.report_tools import _build_branded_docx, _prepare_report_data
        from tools.import_tools import _loaded_measurements, _measurements_lock

        measurements = {"TestPassive": mock_passive_measurement}
        with _measurements_lock:
            _loaded_measurements["TestPassive"] = mock_passive_measurement

        try:
            report_data = _prepare_report_data(measurements, default_opts)
            output = os.path.join(temp_dir, "meta_report.docx")

            metadata = {
                "title": "BLE Antenna Report",
                "project_name": "Project X",
                "antenna_type": "PCB Trace",
                "author": "Test Engineer",
            }

            _build_branded_docx(
                output, report_data, {},
                default_opts, None, metadata, measurements,
            )

            from docx import Document
            doc = Document(output)
            text = "\n".join(p.text for p in doc.paragraphs)

            assert "BLE Antenna Report" in text
            assert "Project X" in text
            assert "PCB Trace" in text
            assert "Test Engineer" in text
        finally:
            with _measurements_lock:
                _loaded_measurements.pop("TestPassive", None)

    def test_docx_with_gain_tables(self, mock_passive_measurement, temp_dir):
        """Verify gain tables are present when enabled."""
        from tools.report_tools import (
            _build_branded_docx, _prepare_report_data, ReportOptions,
        )
        from tools.import_tools import _loaded_measurements, _measurements_lock

        measurements = {"TestPassive": mock_passive_measurement}
        with _measurements_lock:
            _loaded_measurements["TestPassive"] = mock_passive_measurement

        try:
            opts = ReportOptions(
                include_gain_tables=True,
                ai_executive_summary=False,
                ai_section_analysis=False,
                ai_recommendations=False,
                include_2d_plots=False,
                include_3d_plots=False,
            )
            report_data = _prepare_report_data(measurements, opts)
            output = os.path.join(temp_dir, "tables_report.docx")

            _build_branded_docx(
                output, report_data, {},
                opts, None, None, measurements,
            )

            from docx import Document
            doc = Document(output)

            # Should have at least one table (gain stats)
            assert len(doc.tables) >= 1
            # First table should have "Parameter" header
            assert doc.tables[0].rows[0].cells[0].text == "Parameter"
        finally:
            with _measurements_lock:
                _loaded_measurements.pop("TestPassive", None)

    def test_fallback_conclusions(self, mock_passive_measurement, temp_dir, default_opts):
        """Verify fallback conclusions are written when AI is off."""
        from tools.report_tools import _build_branded_docx, _prepare_report_data
        from tools.import_tools import _loaded_measurements, _measurements_lock

        measurements = {"TestPassive": mock_passive_measurement}
        with _measurements_lock:
            _loaded_measurements["TestPassive"] = mock_passive_measurement

        try:
            report_data = _prepare_report_data(measurements, default_opts)
            output = os.path.join(temp_dir, "conclusions_report.docx")

            _build_branded_docx(
                output, report_data, {},
                default_opts, None, None, measurements,
            )

            from docx import Document
            doc = Document(output)
            text = "\n".join(p.text for p in doc.paragraphs)

            assert "Review all performance metrics" in text
        finally:
            with _measurements_lock:
                _loaded_measurements.pop("TestPassive", None)


# ---------------------------------------------------------------------------
# Tests: _generate_ai_text
# ---------------------------------------------------------------------------

class TestGenerateAIText:

    def test_none_provider_returns_none(self):
        from tools.report_tools import _generate_ai_text, ReportOptions
        result = _generate_ai_text(None, "test", {"frequencies": []}, ReportOptions())
        assert result is None


# ---------------------------------------------------------------------------
# Tests: End-to-end generate_report (via direct function call)
# ---------------------------------------------------------------------------

class TestEndToEnd:

    def test_report_no_data_loaded(self):
        """Verify error message when no data is loaded."""
        from tools.import_tools import _loaded_measurements, _measurements_lock
        from tools.report_tools import _prepare_report_data, ReportOptions

        with _measurements_lock:
            _loaded_measurements.clear()

        data = _prepare_report_data({}, ReportOptions())
        assert data["measurements"] == []

    def test_full_pipeline_passive(self, mock_passive_measurement, temp_dir):
        """End-to-end: import passive data, build branded DOCX, verify output."""
        from tools.report_tools import (
            _build_branded_docx, _prepare_report_data, ReportOptions,
        )
        from tools.import_tools import _loaded_measurements, _measurements_lock

        measurements = {"BLE_Test": mock_passive_measurement}
        with _measurements_lock:
            _loaded_measurements["BLE_Test"] = mock_passive_measurement

        try:
            opts = ReportOptions(
                frequencies=[2450.0],
                ai_executive_summary=False,
                ai_section_analysis=False,
                ai_recommendations=False,
                include_gain_tables=True,
                include_2d_plots=False,
                include_3d_plots=False,
            )
            report_data = _prepare_report_data(measurements, opts)
            output = os.path.join(temp_dir, "e2e_report.docx")

            metadata = {
                "title": "E2E Test Report",
                "project_name": "Test",
                "frequency_range": "2400-2500 MHz",
            }

            _build_branded_docx(
                output, report_data, {},
                opts, None, metadata, measurements,
            )

            assert os.path.exists(output)

            from docx import Document
            doc = Document(output)
            text = "\n".join(p.text for p in doc.paragraphs)

            # Cover page
            assert "E2E Test Report" in text
            # Test config
            assert "BLE_Test" in text
            # Gain tables present
            assert len(doc.tables) >= 1
            # Conclusions present
            assert "Conclusions and Recommendations" in text
        finally:
            with _measurements_lock:
                _loaded_measurements.pop("BLE_Test", None)

    def test_full_pipeline_active(self, mock_active_measurement, temp_dir):
        """End-to-end: active data with gain tables."""
        from tools.report_tools import (
            _build_branded_docx, _prepare_report_data, ReportOptions,
        )
        from tools.import_tools import _loaded_measurements, _measurements_lock

        measurements = {"TRP_Test": mock_active_measurement}
        with _measurements_lock:
            _loaded_measurements["TRP_Test"] = mock_active_measurement

        try:
            opts = ReportOptions(
                ai_executive_summary=False,
                ai_section_analysis=False,
                ai_recommendations=False,
                include_gain_tables=True,
                include_2d_plots=False,
                include_3d_plots=False,
            )
            report_data = _prepare_report_data(measurements, opts)
            output = os.path.join(temp_dir, "active_report.docx")

            _build_branded_docx(
                output, report_data, {},
                opts, None, None, measurements,
            )

            assert os.path.exists(output)

            from docx import Document
            doc = Document(output)

            # Should have tables for active stats
            assert len(doc.tables) >= 1
            # Check for TRP in table
            found_trp = False
            for table in doc.tables:
                for row in table.rows:
                    if "TRP" in row.cells[0].text:
                        found_trp = True
                        break
            assert found_trp
        finally:
            with _measurements_lock:
                _loaded_measurements.pop("TRP_Test", None)


# ---------------------------------------------------------------------------
# Tests: _add_freq_comparison_table
# ---------------------------------------------------------------------------

class TestFreqComparisonTable:

    def test_single_freq_skipped(self, mock_active_measurement):
        """With < 2 frequencies, comparison table should not be added."""
        from docx import Document
        from docx.shared import RGBColor
        from tools.report_tools import _add_freq_comparison_table
        from plot_antenna.ai_analysis import AntennaAnalyzer

        analyzer = AntennaAnalyzer(
            measurement_data=mock_active_measurement.data,
            scan_type="active",
            frequencies=[2450.0],
        )
        doc = Document()

        def heading(d, t, level=1):
            d.add_heading(t, level)

        _add_freq_comparison_table(doc, analyzer, heading, RGBColor(50, 50, 50))
        assert len(doc.tables) == 0

    def test_multi_freq_table(self, mock_passive_measurement):
        """With >= 2 frequencies, comparison table should be added."""
        from docx import Document
        from docx.shared import RGBColor
        from tools.report_tools import _add_freq_comparison_table
        from plot_antenna.ai_analysis import AntennaAnalyzer

        analyzer = AntennaAnalyzer(
            measurement_data=mock_passive_measurement.data,
            scan_type="passive",
            frequencies=[2400.0, 2450.0, 2500.0],
        )
        doc = Document()

        def heading(d, t, level=1):
            d.add_heading(t, level)

        _add_freq_comparison_table(doc, analyzer, heading, RGBColor(50, 50, 50))
        assert len(doc.tables) == 1
        table = doc.tables[0]
        # Header row + 3 data rows
        assert len(table.rows) == 4
        assert "Freq (MHz)" in table.rows[0].cells[0].text
