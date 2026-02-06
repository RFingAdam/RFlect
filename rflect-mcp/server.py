"""
RFlect MCP Server

Provides AI-powered antenna analysis and report generation via Model Context Protocol.
Enables Claude Code, Cline, and other AI tools to process antenna measurements programmatically.

Usage:
    # Add to Claude Code MCP settings:
    {
        "mcpServers": {
            "rflect": {
                "command": "python",
                "args": ["path/to/rflect-mcp/server.py"]
            }
        }
    }
"""

import sys
import os

# Add parent directory to path to import plot_antenna
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from mcp.server.fastmcp import FastMCP

# Import RFlect tools
from tools.import_tools import register_import_tools
from tools.analysis_tools import register_analysis_tools
from tools.report_tools import register_report_tools

# Create MCP server
mcp = FastMCP("rflect")

# Register all tools
register_import_tools(mcp)
register_analysis_tools(mcp)
register_report_tools(mcp)


@mcp.resource("rflect://status")
def get_status() -> str:
    """Get current RFlect MCP server status and loaded data summary."""
    from tools.import_tools import get_loaded_data_summary
    return get_loaded_data_summary()


@mcp.resource("rflect://help")
def get_help() -> str:
    """Get help information about available tools."""
    return """
RFlect MCP Server - Antenna Analysis Tools

IMPORT TOOLS:
- import_antenna_file(file_path) - Import single measurement file
- import_antenna_folder(folder_path, pattern) - Import all files from folder
- list_loaded_data() - Show currently loaded measurements
- clear_data() - Clear all loaded data

ANALYSIS TOOLS:
- list_frequencies() - Get available frequencies
- analyze_pattern(frequency, polarization) - Pattern analysis (HPBW, F/B, nulls)
- get_gain_statistics(frequency) - Min/max/avg gain
- compare_polarizations(frequency) - HPOL vs VPOL comparison

REPORT TOOLS:
- generate_report(output_path, options) - Generate DOCX report
- get_report_options() - Show available report configuration options

FILTERING OPTIONS (for generate_report):
- frequencies: List of specific frequencies to include (default: all)
- polarizations: ["total", "hpol", "vpol"] (default: ["total"])
- plots_2d: Include 2D pattern plots (default: true)
- plots_3d: Include 3D pattern plots (default: false - they're large)
- include_raw_data: Include data tables (default: false)
- ai_analysis: Let AI analyze and comment (default: true)
- ai_summary: Generate executive summary (default: true)
"""


if __name__ == "__main__":
    mcp.run()
