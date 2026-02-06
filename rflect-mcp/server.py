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
from tools.bulk_tools import register_bulk_tools

# Create MCP server
mcp = FastMCP("rflect")

# Register all tools
register_import_tools(mcp)
register_analysis_tools(mcp)
register_report_tools(mcp)
register_bulk_tools(mcp)


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
- get_all_analysis(frequency) - Combined gain + pattern + polarization analysis

REPORT TOOLS:
- generate_report(output_path, options) - Generate DOCX report
- preview_report(options) - Preview report contents without generating
- get_report_options() - Show available report configuration options

BULK PROCESSING TOOLS:
- list_measurement_files(folder_path) - Scan folder for measurement files
- bulk_process_passive(folder_path, frequencies, cable_loss) - Batch process HPOL/VPOL pairs
- bulk_process_active(folder_path) - Batch process TRP files
- validate_file_pair(hpol_path, vpol_path) - Validate HPOL/VPOL file pairing
- convert_to_cst(hpol_path, vpol_path, vswr_path, frequency) - Convert to CST .ffs format
"""


if __name__ == "__main__":
    mcp.run()
