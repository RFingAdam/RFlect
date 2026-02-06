"""
Report Tools for RFlect MCP Server

Generates DOCX reports with smart filtering options.
AI analyzes data and generates narrative content.
"""

import os
import json
from typing import Optional, List, Dict, Any
from dataclasses import dataclass, field
from datetime import datetime

from .import_tools import get_loaded_measurements
from .analysis_tools import (
    get_gain_statistics,
    analyze_pattern,
    compare_polarizations,
    get_all_analysis
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

- include_polar_plots: true/false (default: true)
  Standard polar radiation patterns

- include_cartesian_plots: true/false (default: false)
  Cartesian (rectangular) gain plots

DATA FILTERING:
- include_raw_data_tables: true/false (default: false)
  Full measurement data tables (can be very long)

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
- output_format: "docx" (pdf coming soon)
- include_cover_page: true/false
- include_table_of_contents: true/false

EXAMPLE - Minimal Report:
{
    "frequencies": [2450],
    "polarizations": ["total"],
    "include_2d_plots": true,
    "include_3d_plots": false,
    "include_raw_data_tables": false,
    "ai_executive_summary": true,
    "ai_section_analysis": false
}

EXAMPLE - Full Report:
{
    "frequencies": null,  // all
    "polarizations": ["total", "hpol", "vpol"],
    "include_2d_plots": true,
    "include_3d_plots": true,
    "include_gain_tables": true,
    "ai_executive_summary": true,
    "ai_section_analysis": true,
    "ai_recommendations": true
}
"""

    @mcp.tool()
    def generate_report(
        output_path: str,
        options: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Generate an antenna test report with smart filtering.

        Args:
            output_path: Path for the output DOCX file
            options: Report options (see get_report_options for details)
                - frequencies: List of frequencies to include
                - polarizations: ["total", "hpol", "vpol"]
                - include_2d_plots: Include 2D pattern plots
                - include_3d_plots: Include 3D plots (default false)
                - ai_executive_summary: Generate AI summary
                - ai_section_analysis: AI commentary per section

        Returns:
            Path to generated report and summary of contents.
        """
        measurements = get_loaded_measurements()

        if not measurements:
            return "No data loaded. Use import_antenna_file first."

        # Parse options
        opts = ReportOptions()
        if options:
            for key, value in options.items():
                if hasattr(opts, key):
                    setattr(opts, key, value)

        try:
            # Import RFlect report generation
            from plot_antenna.save import generate_report as rflect_generate_report
            from plot_antenna import config

            # Prepare report data
            report_data = _prepare_report_data(measurements, opts)

            # Generate the report using RFlect's existing function
            # For now, we'll create a simplified version
            report_content = _generate_report_content(report_data, opts)

            # Write to file
            _write_docx_report(output_path, report_content, opts)

            # Summary of what was included
            summary = f"Report generated: {output_path}\n\n"
            summary += "Contents:\n"
            summary += f"- Measurements: {len(report_data['measurements'])}\n"
            summary += f"- Frequencies: {report_data['frequencies']}\n"
            summary += f"- Polarizations: {opts.polarizations}\n"
            summary += f"- 2D Plots: {'Yes' if opts.include_2d_plots else 'No'}\n"
            summary += f"- 3D Plots: {'Yes' if opts.include_3d_plots else 'No'}\n"
            summary += f"- AI Summary: {'Yes' if opts.ai_executive_summary else 'No'}\n"

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
            return "No data loaded. Use import_antenna_file first."

        # Parse options
        opts = ReportOptions()
        if options:
            for key, value in options.items():
                if hasattr(opts, key):
                    setattr(opts, key, value)

        # Calculate what would be included
        report_data = _prepare_report_data(measurements, opts)

        preview = "REPORT PREVIEW\n"
        preview += "=" * 40 + "\n\n"

        # Measurements
        preview += f"MEASUREMENTS ({len(report_data['measurements'])})\n"
        for name in report_data['measurements']:
            preview += f"  - {name}\n"

        # Frequencies
        preview += f"\nFREQUENCIES ({len(report_data['frequencies'])})\n"
        for freq in report_data['frequencies'][:10]:
            preview += f"  - {freq} MHz\n"
        if len(report_data['frequencies']) > 10:
            preview += f"  ... and {len(report_data['frequencies']) - 10} more\n"

        # Sections
        preview += f"\nSECTIONS\n"
        preview += f"  [{'x' if opts.include_cover_page else ' '}] Cover Page\n"
        preview += f"  [{'x' if opts.include_table_of_contents else ' '}] Table of Contents\n"
        preview += f"  [{'x' if opts.ai_executive_summary else ' '}] Executive Summary (AI)\n"
        preview += f"  [x] Test Configuration\n"
        preview += f"  [{'x' if opts.include_gain_tables else ' '}] Gain Summary Tables\n"
        preview += f"  [{'x' if opts.include_2d_plots else ' '}] 2D Pattern Plots\n"
        preview += f"  [{'x' if opts.include_3d_plots else ' '}] 3D Pattern Plots\n"
        preview += f"  [{'x' if opts.include_polar_plots else ' '}] Polar Plots\n"
        preview += f"  [{'x' if opts.ai_section_analysis else ' '}] Pattern Analysis (AI)\n"
        preview += f"  [{'x' if opts.ai_recommendations else ' '}] Recommendations (AI)\n"
        preview += f"  [{'x' if opts.include_raw_data_tables else ' '}] Raw Data Tables\n"

        # Estimate complexity
        plot_count = 0
        if opts.include_2d_plots:
            plot_count += len(report_data['frequencies']) * len(opts.polarizations)
        if opts.include_3d_plots:
            plot_count += len(report_data['frequencies']) * len(opts.polarizations)
        if opts.include_polar_plots:
            plot_count += len(report_data['frequencies']) * 2  # E and H plane

        preview += f"\nESTIMATED COMPLEXITY\n"
        preview += f"  Plots: ~{plot_count}\n"
        preview += f"  Tables: ~{len(report_data['frequencies']) if opts.include_gain_tables else 0}\n"
        preview += f"  AI Sections: {sum([opts.ai_executive_summary, opts.ai_section_analysis, opts.ai_recommendations])}\n"

        if plot_count > 20:
            preview += f"\n⚠️  Warning: {plot_count} plots may make the report very large.\n"
            preview += f"   Consider reducing frequencies or disabling 3D plots.\n"

        return preview


def _load_template(template_path: Optional[str] = None) -> Optional[Dict]:
    """Load a YAML report template.

    Args:
        template_path: Path to template YAML. None uses default template.

    Returns:
        Parsed template dict, or None if loading fails.
    """
    try:
        import yaml
    except ImportError:
        return None

    if template_path is None:
        # Use default template from templates/ directory
        template_path = os.path.join(
            os.path.dirname(os.path.dirname(__file__)),
            "templates", "default.yaml"
        )

    if not os.path.isfile(template_path):
        return None

    try:
        with open(template_path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f)
    except Exception:
        return None


def _template_section_enabled(section: Dict, opts: ReportOptions) -> bool:
    """Check whether a template section should be included based on options."""
    if not section.get("enabled", True):
        return False

    section_id = section.get("id", "")

    # Map section IDs to ReportOptions flags
    flag_map = {
        "cover_page": opts.include_cover_page,
        "table_of_contents": opts.include_table_of_contents,
        "executive_summary": opts.ai_executive_summary,
        "gain_summary": opts.include_gain_tables,
        "recommendations": opts.ai_recommendations,
        "appendix_data": opts.include_raw_data_tables,
    }

    if section_id in flag_map:
        return flag_map[section_id]

    return True


def _prepare_report_data(measurements: Dict, opts: ReportOptions) -> Dict:
    """Prepare data for report generation based on options."""
    data = {
        'measurements': [],
        'frequencies': [],
        'analyses': {}
    }

    # Filter measurements
    for name, m in measurements.items():
        if opts.measurements is None or name in opts.measurements:
            data['measurements'].append(name)

            # Collect frequencies
            for freq in m.frequencies:
                if opts.frequencies is None or freq in opts.frequencies:
                    if freq not in data['frequencies']:
                        data['frequencies'].append(freq)

    data['frequencies'].sort()

    return data


def _generate_report_content(data: Dict, opts: ReportOptions) -> Dict:
    """Generate report content including AI analysis.

    If a YAML template is available, uses it to determine section order and
    content.  Falls back to hardcoded sections when no template is loaded.
    """
    template = _load_template(opts.template_path)

    content = {
        'title': 'Antenna Test Report',
        'date': datetime.now().strftime('%Y-%m-%d'),
        'sections': []
    }

    if template:
        content['title'] = template.get('metadata', {}).get('title', content['title'])
        return _generate_from_template(content, data, opts, template)

    return _generate_hardcoded_content(content, data, opts)


def _generate_from_template(
    content: Dict, data: Dict, opts: ReportOptions, template: Dict
) -> Dict:
    """Build report sections by iterating over the YAML template."""
    sections = template.get('sections', [])

    # Map of section IDs to content generators
    section_handlers = {
        'executive_summary': lambda: _generate_ai_summary(data, opts),
        'gain_summary': lambda: _build_gain_sections(data, opts),
        'radiation_patterns': lambda: _build_pattern_sections(data, opts),
        'pattern_analysis': lambda: _build_analysis_sections(data, opts),
        'polarization_analysis': lambda: _build_polarization_sections(data, opts),
        'recommendations': lambda: _generate_ai_section(
            data, opts,
            template.get('sections', [{}])[-1].get('ai_prompt', '')
        ),
    }

    for section in sections:
        if not _template_section_enabled(section, opts):
            continue

        section_id = section.get('id', '')
        title = section.get('title', section_id)

        # Handle AI-generated sections
        if section.get('ai_generated') and section_id in section_handlers:
            text = section_handlers[section_id]()
            if isinstance(text, list):
                content['sections'].extend(text)
            else:
                content['sections'].append({
                    'title': title,
                    'content': text,
                    'type': 'text'
                })
            continue

        # Handle data sections with per-frequency expansion
        if section_id in section_handlers:
            result = section_handlers[section_id]()
            if isinstance(result, list):
                content['sections'].extend(result)
            else:
                content['sections'].append({
                    'title': title,
                    'content': result,
                    'type': 'text'
                })
            continue

        # Handle subsections (e.g., radiation_patterns has 2d, 3d, polar)
        if 'subsections' in section:
            for sub in section['subsections']:
                if sub.get('enabled', True):
                    content['sections'].append({
                        'title': f"{title} - {sub.get('title', '')}",
                        'content': f"[{sub.get('title', 'Section')} placeholder]",
                        'type': 'text'
                    })

    return content


def _build_gain_sections(data: Dict, opts: ReportOptions) -> list:
    """Build gain statistics sections for each frequency."""
    sections = []
    for freq in data['frequencies'][:opts.max_frequencies_in_table]:
        stats = get_gain_statistics(freq)
        sections.append({
            'title': f'Gain Statistics - {freq} MHz',
            'content': stats,
            'type': 'text'
        })
    return sections


def _build_pattern_sections(data: Dict, opts: ReportOptions) -> list:
    """Build pattern cut sections for each frequency."""
    sections = []
    for freq in data['frequencies'][:opts.max_frequencies_in_table]:
        analysis = analyze_pattern(freq)
        sections.append({
            'title': f'Radiation Pattern - {freq} MHz',
            'content': analysis,
            'type': 'text'
        })
    return sections


def _build_analysis_sections(data: Dict, opts: ReportOptions) -> list:
    """Build pattern analysis sections for each frequency."""
    sections = []
    for freq in data['frequencies'][:opts.max_frequencies_in_table]:
        analysis = analyze_pattern(freq)
        sections.append({
            'title': f'Pattern Analysis - {freq} MHz',
            'content': analysis,
            'type': 'text'
        })
    return sections


def _build_polarization_sections(data: Dict, opts: ReportOptions) -> list:
    """Build polarization comparison sections for each frequency."""
    sections = []
    for freq in data['frequencies'][:opts.max_frequencies_in_table]:
        comparison = compare_polarizations(freq)
        sections.append({
            'title': f'Polarization Analysis - {freq} MHz',
            'content': comparison,
            'type': 'text'
        })
    return sections


def _generate_ai_section(data: Dict, opts: ReportOptions, prompt: str) -> str:
    """Generate an AI section using a custom prompt from the template."""
    if not prompt:
        return "[No AI prompt provided in template]"

    try:
        from plot_antenna.api_keys import get_api_key
        from openai import OpenAI

        api_key = get_api_key()
        if not api_key:
            return "[AI section requires OpenAI API key]"

        # Gather context data
        all_analysis = []
        for freq in data['frequencies'][:3]:
            all_analysis.append(get_all_analysis(freq))

        client = OpenAI(api_key=api_key)
        full_prompt = f"{prompt}\n\nMeasurement Data:\n{chr(10).join(all_analysis)}"

        response = client.chat.completions.create(
            model=opts.ai_model,
            messages=[{"role": "user", "content": full_prompt}],
            max_tokens=500
        )

        return response.choices[0].message.content or "[AI section unavailable]"

    except Exception as e:
        return f"[AI section generation failed: {str(e)}]"


def _generate_hardcoded_content(content: Dict, data: Dict, opts: ReportOptions) -> Dict:
    """Fallback: generate report with hardcoded section order (no template)."""
    # Executive Summary (AI)
    if opts.ai_executive_summary:
        summary_text = _generate_ai_summary(data, opts)
        content['sections'].append({
            'title': 'Executive Summary',
            'content': summary_text,
            'type': 'text'
        })

    # Gain Statistics per frequency
    if opts.include_gain_tables:
        content['sections'].extend(_build_gain_sections(data, opts))

    # Pattern Analysis
    content['sections'].extend(_build_analysis_sections(data, opts))

    return content


def _generate_ai_summary(data: Dict, opts: ReportOptions) -> str:
    """Generate AI executive summary."""
    # For now, return a placeholder
    # In full implementation, this would call OpenAI API
    try:
        from plot_antenna.api_keys import get_api_key
        from openai import OpenAI

        api_key = get_api_key()
        if not api_key:
            return "[AI Summary requires OpenAI API key]"

        # Get analysis data
        all_analysis = []
        for freq in data['frequencies'][:3]:  # Limit to first 3 for summary
            all_analysis.append(get_all_analysis(freq))

        client = OpenAI(api_key=api_key)

        prompt = f"""You are an RF engineer analyzing antenna test data.
Based on the following measurements, write a concise executive summary (2-3 paragraphs)
highlighting key performance characteristics, any concerns, and overall assessment.

Measurements:
{chr(10).join(all_analysis)}

Write the executive summary:"""

        response = client.chat.completions.create(
            model=opts.ai_model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=500
        )

        return response.choices[0].message.content or "[AI summary unavailable]"

    except Exception as e:
        return f"[AI Summary generation failed: {str(e)}]"


def _write_docx_report(output_path: str, content: Dict, opts: ReportOptions):
    """Write the report content to a DOCX file."""
    try:
        from docx import Document
        from docx.shared import Inches, Pt
        from docx.enum.text import WD_PARAGRAPH_ALIGNMENT

        doc = Document()

        # Title
        title = doc.add_heading(content['title'], 0)
        title.alignment = WD_PARAGRAPH_ALIGNMENT.CENTER

        # Date
        date_para = doc.add_paragraph(f"Report Date: {content['date']}")
        date_para.alignment = WD_PARAGRAPH_ALIGNMENT.CENTER

        doc.add_page_break()

        # Sections
        for section in content['sections']:
            doc.add_heading(section['title'], 1)

            if section['type'] == 'text':
                # Split content into paragraphs
                for para in section['content'].split('\n'):
                    if para.strip():
                        doc.add_paragraph(para)

            doc.add_paragraph()  # Spacing

        # Save
        doc.save(output_path)

    except ImportError:
        raise Exception("python-docx not installed. Run: pip install python-docx")
