"""
Shared DOCX report helpers (#36).

Single source of truth for the small formatting helpers that were duplicated
between the GUI report builder (plot_antenna/save.py) and the MCP report builder
(rflect-mcp/tools/report_tools.py). Pure formatting; no IO.

The full table-builder functions intentionally remain in each builder — their
signatures diverged over time and unifying them would risk the (tested) MCP
report layout — but the trivial, identical helpers below are now shared.
"""

from __future__ import annotations


def fmt_value(val, fmt: str = ".2f", suffix: str = "") -> str:
    """Format a value for table display, handling None gracefully.

    Canonical implementation of the helper both report builders call ``_fmt``.
    """
    if val is None:
        return "N/A"
    try:
        return f"{float(val):{fmt}}{suffix}"
    except (ValueError, TypeError):
        return str(val)


def style_header_row(table, brand_dark) -> None:
    """Apply branded (bold + brand color) styling to a python-docx table's
    header row. Shared by both report builders."""
    for cell in table.rows[0].cells:
        for para in cell.paragraphs:
            for run in para.runs:
                run.bold = True
                run.font.color.rgb = brand_dark
