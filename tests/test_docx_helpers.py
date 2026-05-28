"""Tests for the shared DOCX helpers (#36).

These helpers were de-duplicated out of the GUI report builder (save.py) and the
MCP report builder (report_tools.py); both now import the canonical versions here.
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "rflect-mcp"))

from plot_antenna.docx_helpers import fmt_value, style_header_row


class TestFmtValue:
    def test_none_is_na(self):
        assert fmt_value(None) == "N/A"

    def test_default_two_decimals(self):
        assert fmt_value(3.14159) == "3.14"

    def test_custom_format_and_suffix(self):
        assert fmt_value(2412.0, ".1f", " MHz") == "2412.0 MHz"

    def test_non_numeric_falls_back_to_str(self):
        assert fmt_value("passive") == "passive"

    def test_integer_input(self):
        assert fmt_value(5, ".0f") == "5"


class TestStyleHeaderRow:
    def test_styles_first_row_bold_and_colored(self):
        # Lightweight fakes mirroring the python-docx run/paragraph/cell/row API.
        class Font:
            def __init__(self):
                self.color = type("C", (), {"rgb": None})()

        class Run:
            def __init__(self):
                self.bold = False
                self.font = Font()

        class Para:
            def __init__(self):
                self.runs = [Run()]

        class Cell:
            def __init__(self):
                self.paragraphs = [Para()]

        class Table:
            def __init__(self):
                self.rows = [type("R", (), {"cells": [Cell(), Cell()]})()]

        table = Table()
        style_header_row(table, brand_dark="DARK")
        for cell in table.rows[0].cells:
            run = cell.paragraphs[0].runs[0]
            assert run.bold is True
            assert run.font.color.rgb == "DARK"

    def test_report_tools_and_save_share_the_same_helper(self):
        # Regression guard for #36: both builders must delegate to this module,
        # not carry private copies that can drift.
        import importlib

        rt = importlib.import_module("tools.report_tools")
        save = importlib.import_module("plot_antenna.save")
        assert rt._fmt is fmt_value
        assert save._fmt is fmt_value
        assert rt._style_header_row is style_header_row
