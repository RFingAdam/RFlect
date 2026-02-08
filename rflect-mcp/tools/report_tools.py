"""
Report Tools for RFlect MCP Server

Generates professional branded DOCX reports with embedded plots, gain tables,
AI-generated summaries/conclusions/captions, and graceful fallback when no
AI provider is configured.
"""

import os
import tempfile
from typing import Optional, List, Dict, Any
from dataclasses import dataclass, field
from datetime import datetime

from .import_tools import get_loaded_measurements, LoadedMeasurement
from .analysis_tools import (
    _get_analyzer_for_measurement,
    get_gain_statistics,
    analyze_pattern,
    compare_polarizations,
    get_all_analysis,
)


@dataclass
class ReportOptions:
    """Configuration options for report generation."""
    # Content filtering
    frequencies: Optional[List[float]] = None  # None = all frequencies
    polarizations: List[str] = field(default_factory=lambda: ["total"])  # total, hpol, vpol
    measurements: Optional[List[str]] = None  # None = all loaded measurements

    # Plot filtering (key for managing complexity)
    include_2d_plots: bool = True
    include_3d_plots: bool = False  # Default off - they're large/complex
    include_polar_plots: bool = True
    include_cartesian_plots: bool = False

    # Data filtering
    include_raw_data_tables: bool = False
    include_gain_tables: bool = True
    max_frequencies_in_table: int = 10  # Limit table rows

    # AI content
    ai_executive_summary: bool = True
    ai_section_analysis: bool = True
    ai_recommendations: bool = True
    ai_model: str = "gpt-4o-mini"  # Cost-effective default

    # Output
    output_format: str = "docx"  # docx, pdf (future)
    include_cover_page: bool = True
    include_table_of_contents: bool = True

    # Template
    template_path: Optional[str] = None  # Path to YAML template (None = default)

    # Branding (loaded from config)
    company_name: Optional[str] = None
    logo_path: Optional[str] = None


# ---------------------------------------------------------------------------
# LLM Provider Helper
# ---------------------------------------------------------------------------

def _create_llm_provider(opts: ReportOptions):
    """Create an LLM provider for report generation based on config.

    Returns BaseLLMProvider or None if no API key is configured.
    """
    try:
        from plot_antenna.api_keys import get_api_key
        from plot_antenna.llm_provider import create_provider
        from plot_antenna import config

        ai_provider = getattr(config, "AI_PROVIDER", "openai")

        if ai_provider == "openai":
            api_key = get_api_key("openai")
            if not api_key:
                return None
            model = getattr(config, "AI_OPENAI_MODEL", opts.ai_model)
            return create_provider("openai", api_key=api_key, model=model)
        elif ai_provider == "anthropic":
            api_key = get_api_key("anthropic")
            if not api_key:
                return None
            model = getattr(config, "AI_ANTHROPIC_MODEL", "claude-sonnet-4-20250514")
            return create_provider("anthropic", api_key=api_key, model=model)
        elif ai_provider == "ollama":
            model = getattr(config, "AI_OLLAMA_MODEL", "llama3.1")
            base_url = getattr(config, "AI_OLLAMA_URL", "http://localhost:11434")
            return create_provider("ollama", model=model, base_url=base_url)
    except Exception:
        pass
    return None


# ---------------------------------------------------------------------------
# Plot Generation
# ---------------------------------------------------------------------------

def _generate_plots(measurements: Dict[str, LoadedMeasurement], opts: ReportOptions,
                    plot_dir: str) -> Dict[str, List[str]]:
    """Generate PNG plots for all measurements in headless mode.

    Returns dict mapping measurement_name -> list of image paths.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from plot_antenna.plotting import (
        plot_2d_passive_data,
        plot_passive_3d_component,
        plot_active_2d_data,
        plot_active_3d_data,
    )

    images: Dict[str, List[str]] = {}

    for name, m in measurements.items():
        if opts.measurements is not None and name not in opts.measurements:
            continue

        meas_dir = os.path.join(plot_dir, _safe_filename(name))
        os.makedirs(meas_dir, exist_ok=True)

        meas_images: List[str] = []

        if m.scan_type == "passive":
            meas_images.extend(
                _generate_passive_plots(m, opts, meas_dir,
                                        plot_2d_passive_data, plot_passive_3d_component, plt)
            )
        elif m.scan_type == "active":
            meas_images.extend(
                _generate_active_plots(m, opts, meas_dir,
                                       plot_active_2d_data, plot_active_3d_data, plt)
            )

        images[name] = meas_images

    return images


def _generate_passive_plots(m: LoadedMeasurement, opts: ReportOptions,
                            meas_dir: str, plot_2d_fn, plot_3d_fn, plt) -> List[str]:
    """Generate passive measurement plots. Returns list of image paths."""
    paths: List[str] = []
    data = m.data
    theta = data.get("theta")
    phi = data.get("phi")
    v_gain = data.get("v_gain")
    h_gain = data.get("h_gain")
    total_gain = data.get("total_gain")

    if theta is None or phi is None or total_gain is None:
        return paths

    freqs = _filter_frequencies(m.frequencies, opts)

    for freq in freqs:
        freq_label = f"{freq:.0f}MHz"

        if opts.include_2d_plots:
            save_dir_2d = os.path.join(meas_dir, "2d")
            os.makedirs(save_dir_2d, exist_ok=True)
            try:
                plot_2d_fn(
                    theta, phi, v_gain, h_gain, total_gain,
                    m.frequencies, freq,
                    datasheet_plots=False,
                    save_path=save_dir_2d,
                )
                plt.close("all")
                # Collect generated PNGs
                for f in sorted(os.listdir(save_dir_2d)):
                    full = os.path.join(save_dir_2d, f)
                    if full.endswith(".png") and full not in paths:
                        paths.append(full)
            except Exception:
                plt.close("all")

        if opts.include_3d_plots:
            save_dir_3d = os.path.join(meas_dir, "3d")
            os.makedirs(save_dir_3d, exist_ok=True)
            for gain_type in ("total", "hpol", "vpol"):
                if gain_type != "total" and gain_type not in opts.polarizations:
                    continue
                try:
                    plot_3d_fn(
                        theta, phi, h_gain, v_gain, total_gain,
                        m.frequencies, freq,
                        gain_type=gain_type,
                        save_path=save_dir_3d,
                    )
                    plt.close("all")
                    for f in sorted(os.listdir(save_dir_3d)):
                        full = os.path.join(save_dir_3d, f)
                        if full.endswith(".png") and full not in paths:
                            paths.append(full)
                except Exception:
                    plt.close("all")

    return paths


def _generate_active_plots(m: LoadedMeasurement, opts: ReportOptions,
                           meas_dir: str, plot_2d_fn, plot_3d_fn, plt) -> List[str]:
    """Generate active measurement plots. Returns list of image paths."""
    paths: List[str] = []
    data = m.data

    # Need the 2D arrays stored by import_active_processed
    data_points = data.get("data_points")
    theta_rad = data.get("theta_rad")
    phi_rad_plot = data.get("phi_rad_plot")
    total_power_2d_plot = data.get("total_power_2d_plot")

    # For 3D
    theta_deg = data.get("theta")
    phi_deg = data.get("phi")
    total_power_2d = data.get("total_power_2d")
    phi_deg_plot = data.get("phi_deg_plot")

    freq = m.frequencies[0] if m.frequencies else 0

    if opts.include_2d_plots and all(v is not None for v in
                                     [data_points, theta_rad, phi_rad_plot, total_power_2d_plot]):
        save_dir_2d = os.path.join(meas_dir, "2d")
        os.makedirs(save_dir_2d, exist_ok=True)
        try:
            plot_2d_fn(data_points, theta_rad, phi_rad_plot,
                       total_power_2d_plot, freq, save_path=save_dir_2d)
            plt.close("all")
            for f in sorted(os.listdir(save_dir_2d)):
                full = os.path.join(save_dir_2d, f)
                if full.endswith(".png") and full not in paths:
                    paths.append(full)
        except Exception:
            plt.close("all")

    if opts.include_3d_plots and all(v is not None for v in
                                     [theta_deg, phi_deg, total_power_2d,
                                      phi_deg_plot, total_power_2d_plot]):
        save_dir_3d = os.path.join(meas_dir, "3d")
        os.makedirs(save_dir_3d, exist_ok=True)
        try:
            plot_3d_fn(theta_deg, phi_deg, total_power_2d,
                       phi_deg_plot, total_power_2d_plot, freq,
                       save_path=save_dir_3d)
            plt.close("all")
            for f in sorted(os.listdir(save_dir_3d)):
                full = os.path.join(save_dir_3d, f)
                if full.endswith(".png") and full not in paths:
                    paths.append(full)
        except Exception:
            plt.close("all")

    return paths


def _filter_frequencies(all_freqs: List[float], opts: ReportOptions) -> List[float]:
    """Return frequencies filtered by opts, or all if no filter."""
    if opts.frequencies is None:
        return list(all_freqs)
    return [f for f in all_freqs if f in opts.frequencies]


def _safe_filename(name: str) -> str:
    """Convert a measurement name to a safe directory name."""
    return "".join(c if c.isalnum() or c in " _-" else "_" for c in name).strip()


# ---------------------------------------------------------------------------
# AI Text Generation
# ---------------------------------------------------------------------------

def _generate_ai_text(provider, prompt: str, data: Dict, opts: ReportOptions,
                      max_tokens: int = 500) -> Optional[str]:
    """Generate AI text using an LLM provider. Returns None on failure."""
    if provider is None:
        return None
    try:
        from plot_antenna.llm_provider import LLMMessage

        all_analysis = []
        for freq in data["frequencies"][:3]:
            all_analysis.append(get_all_analysis(freq))

        full_prompt = f"{prompt}\n\nMeasurement Data:\n" + "\n".join(all_analysis)
        response = provider.chat(
            [LLMMessage(role="user", content=full_prompt)], max_tokens=max_tokens
        )
        return response.content or None
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Branded DOCX Builder
# ---------------------------------------------------------------------------

def _fmt(val, fmt=".2f", suffix=""):
    """Format a value for table display, handling None gracefully."""
    if val is None:
        return "N/A"
    try:
        return f"{float(val):{fmt}}{suffix}"
    except (ValueError, TypeError):
        return str(val)


def _build_branded_docx(output_path: str, report_data: Dict,
                        plot_images: Dict[str, List[str]],
                        opts: ReportOptions, provider, metadata: Optional[Dict],
                        measurements: Dict[str, LoadedMeasurement]):
    """Build a professional branded DOCX report.

    Produces the same section structure and formatting as the GUI report
    pipeline in save.py, adapted for MCP server use.
    """
    from docx import Document
    from docx.shared import Inches, Pt, RGBColor
    from docx.enum.text import WD_PARAGRAPH_ALIGNMENT
    from plot_antenna import config
    from plot_antenna.ai_analysis import AntennaAnalyzer

    # Brand colors
    BRAND_PRIMARY = (
        RGBColor(*config.BRAND_PRIMARY_COLOR)
        if getattr(config, "BRAND_PRIMARY_COLOR", None)
        else RGBColor(70, 130, 180)
    )
    BRAND_DARK = (
        RGBColor(*config.BRAND_DARK_COLOR)
        if getattr(config, "BRAND_DARK_COLOR", None)
        else RGBColor(50, 50, 50)
    )
    BRAND_LIGHT = (
        RGBColor(*config.BRAND_LIGHT_COLOR)
        if getattr(config, "BRAND_LIGHT_COLOR", None)
        else RGBColor(128, 128, 128)
    )

    brand_name = getattr(config, "BRAND_NAME", None)
    brand_tagline = getattr(config, "BRAND_TAGLINE", None)
    brand_website = getattr(config, "BRAND_WEBSITE", None)
    report_subtitle = getattr(config, "REPORT_SUBTITLE", "Antenna Measurement & Analysis Report")

    def add_branded_heading(doc, text, level=1):
        heading = doc.add_heading(text, level=level)
        for run in heading.runs:
            run.font.color.rgb = BRAND_DARK
            if level == 1:
                run.font.size = Pt(getattr(config, "HEADING1_FONT_SIZE", 18))
            else:
                run.font.size = Pt(getattr(config, "HEADING2_FONT_SIZE", 14))
            run.font.bold = True
        return heading

    doc = Document()

    # Margins
    for section in doc.sections:
        section.top_margin = Inches(0.75)
        section.bottom_margin = Inches(0.75)
        section.left_margin = Inches(1)
        section.right_margin = Inches(1)

    # Logo in header
    logo_path = opts.logo_path
    if not logo_path and getattr(config, "LOGO_FILENAME", None):
        candidates = [
            os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
                         "assets", config.LOGO_FILENAME),
        ]
        for p in candidates:
            if os.path.exists(p):
                logo_path = p
                break

    if logo_path and os.path.exists(logo_path):
        header = doc.sections[0].header
        header_para = header.paragraphs[0]
        header_run = header_para.add_run()
        logo_width = getattr(config, "LOGO_WIDTH_INCHES", 2.0)
        header_run.add_picture(logo_path, width=Inches(logo_width))
        logo_align = getattr(config, "LOGO_ALIGNMENT", "LEFT")
        align_map = {"CENTER": WD_PARAGRAPH_ALIGNMENT.CENTER,
                     "RIGHT": WD_PARAGRAPH_ALIGNMENT.RIGHT}
        header_para.alignment = align_map.get(logo_align, WD_PARAGRAPH_ALIGNMENT.LEFT)

    # ------------------------------------------------------------------ #
    # SECTION 1: Title Page
    # ------------------------------------------------------------------ #
    if opts.include_cover_page:
        title_text = (metadata or {}).get("title", "Antenna Radiation Pattern Test Report")
        doc.add_paragraph()
        doc.add_paragraph()
        doc.add_paragraph()

        title = doc.add_heading(title_text, 0)
        title.alignment = WD_PARAGRAPH_ALIGNMENT.CENTER
        title_run = title.runs[0]
        title_run.font.color.rgb = BRAND_DARK
        title_run.font.size = Pt(getattr(config, "TITLE_FONT_SIZE", 28))
        title_run.font.bold = True

        if report_subtitle:
            sub = doc.add_paragraph()
            sub.alignment = WD_PARAGRAPH_ALIGNMENT.CENTER
            sub_run = sub.add_run(report_subtitle)
            sub_run.font.color.rgb = BRAND_LIGHT
            sub_run.font.size = Pt(getattr(config, "SUBTITLE_FONT_SIZE", 14))
            sub_run.italic = True

        doc.add_paragraph()
        doc.add_paragraph()

        if metadata:
            meta_fields = [
                ("Project:", metadata.get("project_name")),
                ("Antenna Type:", metadata.get("antenna_type")),
                ("Frequency Range:", metadata.get("frequency_range")),
                ("Date:", metadata.get("date", datetime.now().strftime("%B %d, %Y"))),
                ("Prepared by:", metadata.get("author")),
            ]
            for label, value in meta_fields:
                if value:
                    mp = doc.add_paragraph()
                    mp.alignment = WD_PARAGRAPH_ALIGNMENT.CENTER
                    lr = mp.add_run(f"{label} ")
                    lr.font.bold = True
                    lr.font.color.rgb = BRAND_DARK
                    lr.font.size = Pt(12)
                    vr = mp.add_run(value)
                    vr.font.color.rgb = BRAND_LIGHT
                    vr.font.size = Pt(12)
                    mp.paragraph_format.space_after = Pt(3)
        else:
            dp = doc.add_paragraph()
            dp.alignment = WD_PARAGRAPH_ALIGNMENT.CENTER
            dr = dp.add_run(f"Date: {datetime.now().strftime('%B %d, %Y')}")
            dr.font.color.rgb = BRAND_LIGHT
            dr.font.size = Pt(12)

        if brand_website:
            doc.add_paragraph()
            doc.add_paragraph()
            wp = doc.add_paragraph()
            wp.alignment = WD_PARAGRAPH_ALIGNMENT.CENTER
            wr = wp.add_run(brand_website)
            wr.font.color.rgb = BRAND_LIGHT
            wr.font.size = Pt(10)
            wr.italic = True

        doc.add_page_break()

    # ------------------------------------------------------------------ #
    # SECTION 2: Table of Contents
    # ------------------------------------------------------------------ #
    if opts.include_table_of_contents:
        add_branded_heading(doc, "Table of Contents", level=1)
        toc = doc.add_paragraph(
            "< Table of Contents will be generated when document is opened in Word >"
        )
        toc.paragraph_format.space_before = Pt(6)
        doc.add_page_break()

    # ------------------------------------------------------------------ #
    # SECTION 3: Executive Summary
    # ------------------------------------------------------------------ #
    if opts.ai_executive_summary:
        add_branded_heading(doc, "Executive Summary", level=1)
        summary = _generate_ai_text(
            provider,
            "You are an RF engineer analyzing antenna test data.\n"
            "Write a concise executive summary (2-3 paragraphs) highlighting "
            "key performance characteristics, any concerns, and overall assessment.",
            report_data, opts, max_tokens=500,
        )
        if summary:
            for para_text in summary.split("\n"):
                if para_text.strip():
                    doc.add_paragraph(para_text.strip())
        else:
            doc.add_paragraph(
                "This report presents the results of antenna radiation pattern "
                "measurements. The following sections detail gain statistics, "
                "radiation patterns, and polarization characteristics."
            )
        doc.add_page_break()

    # ------------------------------------------------------------------ #
    # SECTION 4: Test Configuration
    # ------------------------------------------------------------------ #
    add_branded_heading(doc, "Test Configuration", level=1)
    doc.add_paragraph(f"Measurements included: {len(report_data['measurements'])}")
    for meas_name in report_data["measurements"]:
        m = measurements.get(meas_name)
        if m:
            doc.add_paragraph(
                f"  {meas_name} ({m.scan_type}) - {len(m.frequencies)} frequency point(s)",
                style="List Bullet",
            )
    doc.add_paragraph(f"Frequencies: {', '.join(f'{f:.1f} MHz' for f in report_data['frequencies'])}")
    doc.add_paragraph()

    # ------------------------------------------------------------------ #
    # SECTION 5: Gain Summary Tables
    # ------------------------------------------------------------------ #
    if opts.include_gain_tables:
        add_branded_heading(doc, "Measurement Analysis", level=1)

        for meas_name in report_data["measurements"]:
            m = measurements.get(meas_name)
            if not m:
                continue

            analyzer, _, err = _get_analyzer_for_measurement(meas_name)
            if analyzer is None:
                continue

            add_branded_heading(doc, f"Gain Statistics - {meas_name}", level=2)

            freqs = _filter_frequencies(m.frequencies, opts)
            for freq in freqs[:opts.max_frequencies_in_table]:
                stats = analyzer.get_gain_statistics(frequency=freq)
                _add_gain_stats_table(doc, stats, BRAND_DARK)

            # Multi-frequency comparison table
            if len(analyzer.frequencies) >= 2:
                _add_freq_comparison_table(doc, analyzer, add_branded_heading, BRAND_DARK)

        doc.add_page_break()

    # ------------------------------------------------------------------ #
    # SECTION 6: Measurement Results (embedded plots)
    # ------------------------------------------------------------------ #
    total_images = sum(len(imgs) for imgs in plot_images.values())
    if total_images > 0:
        add_branded_heading(doc, "Measurement Results", level=1)
        figure_num = 1

        for meas_name, img_paths in plot_images.items():
            if not img_paths:
                continue
            add_branded_heading(doc, meas_name, level=2)

            for img_path in img_paths:
                if not os.path.exists(img_path):
                    continue
                doc.add_picture(img_path, width=Inches(6))
                doc.paragraphs[-1].alignment = WD_PARAGRAPH_ALIGNMENT.CENTER

                caption = doc.add_paragraph()
                cap_run = caption.add_run(f"Figure {figure_num}: ")
                cap_run.bold = True
                cap_run.font.color.rgb = BRAND_DARK
                cap_run.font.size = Pt(11)

                fname_run = caption.add_run(os.path.basename(img_path))
                fname_run.font.color.rgb = BRAND_LIGHT
                fname_run.font.size = Pt(11)
                caption.alignment = WD_PARAGRAPH_ALIGNMENT.CENTER

                figure_num += 1

        doc.add_page_break()

    # ------------------------------------------------------------------ #
    # SECTION 7: Pattern Analysis
    # ------------------------------------------------------------------ #
    if opts.ai_section_analysis:
        add_branded_heading(doc, "Pattern Analysis", level=1)

        for freq in report_data["frequencies"][:opts.max_frequencies_in_table]:
            analysis_text = analyze_pattern(freq)
            add_branded_heading(doc, f"Pattern at {freq:.1f} MHz", level=2)
            for line in analysis_text.split("\n"):
                if line.strip():
                    doc.add_paragraph(line.strip())

        # AI commentary
        ai_commentary = _generate_ai_text(
            provider,
            "Analyze the radiation patterns and comment on pattern classification, "
            "beamwidth characteristics, front-to-back ratio, and any anomalies.",
            report_data, opts, max_tokens=400,
        )
        if ai_commentary:
            doc.add_paragraph()
            for line in ai_commentary.split("\n"):
                if line.strip():
                    doc.add_paragraph(line.strip())

    # ------------------------------------------------------------------ #
    # SECTION 8: Polarization Analysis
    # ------------------------------------------------------------------ #
    has_polarization = any(
        measurements[n].scan_type == "passive" for n in report_data["measurements"]
        if n in measurements
    )
    if has_polarization:
        add_branded_heading(doc, "Polarization Analysis", level=1)

        for freq in report_data["frequencies"][:opts.max_frequencies_in_table]:
            pol_text = compare_polarizations(freq)
            add_branded_heading(doc, f"Polarization at {freq:.1f} MHz", level=2)
            for line in pol_text.split("\n"):
                if line.strip():
                    doc.add_paragraph(line.strip())

    # ------------------------------------------------------------------ #
    # SECTION 9: Conclusions and Recommendations
    # ------------------------------------------------------------------ #
    doc.add_page_break()
    add_branded_heading(doc, "Conclusions and Recommendations", level=1)

    if opts.ai_recommendations:
        conclusions = _generate_ai_text(
            provider,
            "Based on the measurements, provide 4-6 bullet-point conclusions and "
            "recommendations for the antenna design. Be specific and actionable.",
            report_data, opts, max_tokens=500,
        )
        if conclusions:
            doc.add_paragraph("Based on the measurement results presented in this report:")
            for line in conclusions.split("\n"):
                line = line.strip().lstrip("-").lstrip("*").lstrip()
                if line:
                    doc.add_paragraph(f"{line}", style="List Bullet")
        else:
            _add_fallback_conclusions(doc)
    else:
        _add_fallback_conclusions(doc)

    # Brand footer
    if brand_tagline or brand_website:
        doc.add_paragraph()
        if brand_tagline and brand_name:
            fp = doc.add_paragraph()
            fp.alignment = WD_PARAGRAPH_ALIGNMENT.CENTER
            fr = fp.add_run(f"{brand_name} | {brand_tagline}")
            fr.font.color.rgb = BRAND_LIGHT
            fr.font.size = Pt(10)
            fr.italic = True
        if brand_website:
            wp = doc.add_paragraph()
            wp.alignment = WD_PARAGRAPH_ALIGNMENT.CENTER
            wr = wp.add_run(brand_website)
            wr.font.color.rgb = BRAND_PRIMARY
            wr.font.size = Pt(10)
            wr.bold = True

    doc.save(output_path)


def _add_fallback_conclusions(doc):
    """Add generic conclusion bullets when AI is not available."""
    doc.add_paragraph("Based on the measurement results presented in this report:")
    doc.add_paragraph(
        "Review all performance metrics against specification requirements",
        style="List Bullet",
    )
    doc.add_paragraph(
        "Verify antenna performance meets application needs across the operational bandwidth",
        style="List Bullet",
    )
    doc.add_paragraph(
        "Consider additional measurements or design iterations if performance gaps are identified",
        style="List Bullet",
    )


def _add_gain_stats_table(doc, stats: Dict, brand_dark):
    """Add a formatted gain statistics table to the document."""
    scan_type = stats.get("scan_type", "passive")

    if scan_type == "passive":
        row_data = [
            ("Parameter", "Value"),
            ("Frequency", f"{_fmt(stats.get('frequency_actual'), '.1f')} MHz"),
            ("Peak Gain", f"{_fmt(stats.get('max_gain_dBi'))} dBi"),
            ("Minimum Gain", f"{_fmt(stats.get('min_gain_dBi'))} dBi"),
            ("Average Gain", f"{_fmt(stats.get('avg_gain_dBi'))} dBi"),
            ("Std Deviation", f"{_fmt(stats.get('std_dev_dB'))} dB"),
        ]
        if stats.get("max_hpol_gain_dBi") is not None:
            row_data.append(("H-pol Peak Gain", f"{_fmt(stats.get('max_hpol_gain_dBi'))} dBi"))
        if stats.get("max_vpol_gain_dBi") is not None:
            row_data.append(("V-pol Peak Gain", f"{_fmt(stats.get('max_vpol_gain_dBi'))} dBi"))
    else:
        row_data = [
            ("Parameter", "Value"),
            ("Peak Power", f"{_fmt(stats.get('max_power_dBm'))} dBm"),
            ("Minimum Power", f"{_fmt(stats.get('min_power_dBm'))} dBm"),
            ("Average Power", f"{_fmt(stats.get('avg_power_dBm'))} dBm"),
            ("Std Deviation", f"{_fmt(stats.get('std_dev_dB'))} dB"),
        ]
        if stats.get("TRP_dBm") is not None:
            row_data.append(("TRP", f"{_fmt(stats.get('TRP_dBm'))} dBm"))

    table = doc.add_table(rows=len(row_data), cols=2)
    table.style = "Light Shading Accent 1"

    for i, (label, value) in enumerate(row_data):
        table.rows[i].cells[0].text = label
        table.rows[i].cells[1].text = value
        if i == 0:
            for cell in table.rows[i].cells:
                for para in cell.paragraphs:
                    for run in para.runs:
                        run.bold = True
                        run.font.color.rgb = brand_dark

    doc.add_paragraph()


def _add_freq_comparison_table(doc, antenna_analyzer, add_branded_heading, brand_dark):
    """Add a multi-frequency comparison table to the document."""
    from docx.shared import Pt

    freqs = antenna_analyzer.frequencies
    if not freqs or len(freqs) < 2:
        return

    add_branded_heading(doc, "Multi-Frequency Comparison", level=2)

    headers = [
        "Freq (MHz)", "Peak Gain (dBi)", "Pattern Type",
        "HPBW-E (\u00b0)", "HPBW-H (\u00b0)", "F/B (dB)",
    ]
    table = doc.add_table(rows=1 + len(freqs), cols=len(headers))
    table.style = "Light Shading Accent 1"

    for j, h in enumerate(headers):
        cell = table.rows[0].cells[j]
        cell.text = h
        for para in cell.paragraphs:
            for run in para.runs:
                run.bold = True
                run.font.color.rgb = brand_dark

    for i, freq in enumerate(freqs):
        pattern = antenna_analyzer.analyze_pattern(frequency=freq)
        row = table.rows[i + 1]
        row.cells[0].text = _fmt(freq, ".1f")
        row.cells[1].text = _fmt(pattern.get("peak_gain_dBi"))
        row.cells[2].text = str(pattern.get("pattern_type", "N/A"))
        row.cells[3].text = _fmt(pattern.get("hpbw_e_plane"), ".1f")
        row.cells[4].text = _fmt(pattern.get("hpbw_h_plane"), ".1f")
        row.cells[5].text = _fmt(pattern.get("front_to_back_dB"), ".1f")

    overall = antenna_analyzer.analyze_all_frequencies()
    if overall.get("resonance_frequency_MHz"):
        parts = [
            f"Resonance frequency: {_fmt(overall['resonance_frequency_MHz'], '.1f')} MHz",
            f"Peak gain at resonance: {_fmt(overall.get('peak_gain_at_resonance_dBi'))} dBi",
            f"Gain variation across band: {_fmt(overall.get('gain_variation_dB'))} dB",
        ]
        if overall.get("bandwidth_3dB_MHz") is not None:
            parts.append(f"3 dB bandwidth: {_fmt(overall['bandwidth_3dB_MHz'], '.1f')} MHz")

        summary_para = doc.add_paragraph(" | ".join(parts))
        summary_para.paragraph_format.space_before = Pt(6)
        for run in summary_para.runs:
            run.font.size = Pt(9)
            run.italic = True

    doc.add_paragraph()


# ---------------------------------------------------------------------------
# Data Preparation
# ---------------------------------------------------------------------------

def _prepare_report_data(measurements: Dict[str, LoadedMeasurement],
                         opts: ReportOptions) -> Dict:
    """Prepare data for report generation based on options."""
    data: Dict[str, Any] = {
        "measurements": [],
        "frequencies": [],
    }

    for name, m in measurements.items():
        if opts.measurements is None or name in opts.measurements:
            data["measurements"].append(name)
            for freq in m.frequencies:
                if opts.frequencies is None or freq in opts.frequencies:
                    if freq not in data["frequencies"]:
                        data["frequencies"].append(freq)

    data["frequencies"].sort()
    return data


# ---------------------------------------------------------------------------
# MCP Tool Registration
# ---------------------------------------------------------------------------

def register_report_tools(mcp):
    """Register report generation tools with the MCP server."""

    @mcp.tool()
    def get_report_options() -> str:
        """
        Get available report configuration options.

        Returns:
            Documentation of all available filtering and configuration options.
        """
        return """
REPORT FILTERING OPTIONS
========================

Use these options with generate_report() to control what's included.

CONTENT FILTERING:
- frequencies: List of frequencies to include (MHz), or null for all
  Example: [2400, 2450, 2500]

- polarizations: Which polarization data to include
  Options: ["total"], ["hpol", "vpol"], ["total", "hpol", "vpol"]
  Default: ["total"]

- measurements: Specific measurement files to include, or null for all

PLOT FILTERING (manages complexity):
- include_2d_plots: true/false (default: true)
  Includes 2D azimuth/elevation pattern cuts

- include_3d_plots: true/false (default: FALSE)
  3D surface plots are large - disabled by default

DATA FILTERING:
- include_gain_tables: true/false (default: true)
  Summary gain tables per frequency

- max_frequencies_in_table: number (default: 10)
  Limits table rows for readability

AI CONTENT:
- ai_executive_summary: true/false (default: true)
  AI-generated executive summary

- ai_section_analysis: true/false (default: true)
  AI commentary on each section

- ai_recommendations: true/false (default: true)
  AI-generated design recommendations

- ai_model: "gpt-4o-mini", "gpt-4o", "o3", etc.
  Default: "gpt-4o-mini" (cost-effective)

OUTPUT:
- include_cover_page: true/false (default: true)
- include_table_of_contents: true/false (default: true)

METADATA (optional dict):
- title: Custom report title
- project_name: Project name for cover page
- antenna_type: Antenna type description
- frequency_range: Frequency range string
- author: Report author name
- date: Report date string

EXAMPLE - Minimal Report:
{
    "frequencies": [2450],
    "polarizations": ["total"],
    "include_2d_plots": true,
    "include_3d_plots": false,
    "ai_executive_summary": false,
    "ai_section_analysis": false
}

EXAMPLE - Full Report:
{
    "frequencies": null,
    "polarizations": ["total", "hpol", "vpol"],
    "include_2d_plots": true,
    "include_3d_plots": true,
    "include_gain_tables": true,
    "ai_executive_summary": true,
    "ai_section_analysis": true,
    "ai_recommendations": true,
    "metadata": {
        "title": "BLE Antenna Test Report",
        "project_name": "Product X",
        "antenna_type": "PCB Trace Antenna"
    }
}
"""

    @mcp.tool()
    def generate_report(
        output_path: str,
        options: Optional[Dict[str, Any]] = None,
        title: Optional[str] = None,
        metadata: Optional[Dict[str, str]] = None,
    ) -> str:
        """
        Generate a professional branded antenna test report (DOCX).

        Produces a DOCX with cover page, gain tables, embedded 2D/3D plots,
        AI-generated summaries and conclusions, and branded formatting.

        Args:
            output_path: Path for the output DOCX file
            options: Report options (see get_report_options for details)
            title: Custom report title (default: "Antenna Radiation Pattern Test Report")
            metadata: Dict with project_name, antenna_type, frequency_range, author, date

        Returns:
            Path to generated report and summary of contents.
        """
        measurements = get_loaded_measurements()

        if not measurements:
            return "No data loaded. Use import_antenna_file or import_passive_pair first."

        # Parse options
        opts = ReportOptions()
        if options:
            for key, value in options.items():
                if hasattr(opts, key):
                    setattr(opts, key, value)

        # Merge title into metadata
        if metadata is None:
            metadata = {}
        if title:
            metadata.setdefault("title", title)

        try:
            # 1. Prepare report data
            report_data = _prepare_report_data(measurements, opts)

            if not report_data["measurements"]:
                return "No measurements match the specified filters."

            # 2. Generate plots in temp dir
            plot_images: Dict[str, List[str]] = {}
            temp_dir = tempfile.mkdtemp(prefix="rflect_report_")

            if opts.include_2d_plots or opts.include_3d_plots:
                plot_images = _generate_plots(measurements, opts, temp_dir)

            # 3. Create AI provider (optional)
            provider = _create_llm_provider(opts)

            # 4. Build branded DOCX
            _build_branded_docx(
                output_path, report_data, plot_images,
                opts, provider, metadata, measurements,
            )

            # 5. Summary
            total_plots = sum(len(imgs) for imgs in plot_images.values())
            summary = f"Report generated: {output_path}\n\n"
            summary += "Contents:\n"
            summary += f"- Measurements: {len(report_data['measurements'])}\n"
            summary += f"- Frequencies: {report_data['frequencies']}\n"
            summary += f"- Embedded plots: {total_plots}\n"
            summary += f"- Gain tables: {'Yes' if opts.include_gain_tables else 'No'}\n"
            summary += f"- AI provider: {'Connected' if provider else 'None (fallback text used)'}\n"
            summary += f"- Cover page: {'Yes' if opts.include_cover_page else 'No'}\n"

            return summary

        except Exception as e:
            return f"Error generating report: {str(e)}"

    @mcp.tool()
    def preview_report(options: Optional[Dict[str, Any]] = None) -> str:
        """
        Preview what would be included in a report without generating it.

        Args:
            options: Same options as generate_report

        Returns:
            Summary of what the report would contain.
        """
        measurements = get_loaded_measurements()

        if not measurements:
            return "No data loaded. Use import_antenna_file or import_passive_pair first."

        # Parse options
        opts = ReportOptions()
        if options:
            for key, value in options.items():
                if hasattr(opts, key):
                    setattr(opts, key, value)

        report_data = _prepare_report_data(measurements, opts)

        preview = "REPORT PREVIEW\n"
        preview += "=" * 40 + "\n\n"

        # Measurements
        preview += f"MEASUREMENTS ({len(report_data['measurements'])})\n"
        for name in report_data["measurements"]:
            m = measurements.get(name)
            scan = m.scan_type if m else "?"
            preview += f"  - {name} ({scan})\n"

        # Frequencies
        preview += f"\nFREQUENCIES ({len(report_data['frequencies'])})\n"
        for freq in report_data["frequencies"][:10]:
            preview += f"  - {freq} MHz\n"
        if len(report_data["frequencies"]) > 10:
            preview += f"  ... and {len(report_data['frequencies']) - 10} more\n"

        # Sections
        preview += "\nSECTIONS\n"
        preview += f"  [{'x' if opts.include_cover_page else ' '}] Cover Page (branded)\n"
        preview += f"  [{'x' if opts.include_table_of_contents else ' '}] Table of Contents\n"
        preview += f"  [{'x' if opts.ai_executive_summary else ' '}] Executive Summary (AI)\n"
        preview += "  [x] Test Configuration\n"
        preview += f"  [{'x' if opts.include_gain_tables else ' '}] Gain Summary Tables\n"
        preview += f"  [{'x' if opts.include_2d_plots else ' '}] 2D Pattern Plots\n"
        preview += f"  [{'x' if opts.include_3d_plots else ' '}] 3D Pattern Plots\n"
        preview += f"  [{'x' if opts.ai_section_analysis else ' '}] Pattern Analysis (AI)\n"
        preview += f"  [{'x' if opts.ai_recommendations else ' '}] Conclusions (AI)\n"

        # Estimate plot count
        n_meas = len(report_data["measurements"])
        n_freqs = len(report_data["frequencies"])
        plot_count = 0
        if opts.include_2d_plots:
            plot_count += n_freqs * n_meas
        if opts.include_3d_plots:
            plot_count += n_freqs * n_meas * len(opts.polarizations)

        preview += f"\nESTIMATED COMPLEXITY\n"
        preview += f"  Plots: ~{plot_count}\n"
        preview += f"  Tables: ~{n_freqs if opts.include_gain_tables else 0}\n"
        ai_count = sum([opts.ai_executive_summary, opts.ai_section_analysis, opts.ai_recommendations])
        preview += f"  AI Sections: {ai_count}\n"

        if plot_count > 20:
            preview += f"\nWarning: {plot_count} plots may make the report very large.\n"
            preview += "Consider reducing frequencies or disabling 3D plots.\n"

        return preview
