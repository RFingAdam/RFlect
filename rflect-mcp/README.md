# RFlect MCP Server

AI-powered antenna analysis and report generation via Model Context Protocol (MCP).

Enables Claude Code, Cline, and other AI assistants to programmatically analyze antenna measurements and generate reports.

## Quick Start

Add to your Claude Code configuration file (`~/.claude/settings.json`):

```json
{
  "mcpServers": {
    "rflect": {
      "command": "python",
      "args": ["f:/Personal/_Projects/_live/RFlect/RFlect/rflect-mcp/server.py"],
      "env": {}
    }
  }
}
```

**Important**: Use the absolute path to `server.py` on your system.

Then restart Claude Code and you'll have access to 20 antenna analysis tools. See [Quick-Start Workflow](#quick-start-workflow) below.

## Installation

```bash
# From the rflect-mcp directory
pip install -r requirements.txt

# Also install RFlect from parent directory
pip install -e ..
```

## Configuration

### Claude Code

Add to your Claude Code MCP settings (`~/.claude/settings.json`):

```json
{
  "mcpServers": {
    "rflect": {
      "command": "python",
      "args": ["/absolute/path/to/rflect-mcp/server.py"],
      "env": {}
    }
  }
}
```

**Note**: Replace `/absolute/path/to/` with the full path on your system. On Windows, use forward slashes or escaped backslashes.

### Cline Configuration

For Cline users, add to `.cline/mcp_settings.json` in your project root:

```json
{
  "mcpServers": {
    "rflect": {
      "command": "python",
      "args": ["f:/Personal/_Projects/_live/RFlect/RFlect/rflect-mcp/server.py"],
      "env": {}
    }
  }
}
```

### Other MCP Clients

Most MCP-compatible clients use similar JSON configuration. Consult your client's documentation for the config file location.

## Available Tools

The RFlect MCP server provides 20 tools across four categories:

### Import Tools (6 tools)

| Tool | Description | Parameters |
|------|-------------|------------|
| `import_antenna_file` | Import single antenna measurement file | `file_path` (str), `scan_type` (passive/active) |
| `import_antenna_folder` | Import all measurement files from a folder | `folder_path` (str), `pattern` (optional glob pattern) |
| `import_passive_pair` | Import and process HPOL/VPOL passive pair with full calculation pipeline | `hpol_file` (str), `vpol_file` (str) |
| `import_active_processed` | Import and process active TRP file with power calculations | `file_path` (str), `name` (optional str) |
| `list_loaded_data` | List all currently loaded measurements with frequencies | None |
| `clear_data` | Clear all loaded data from memory | None |

**Typical usage**: Import data before analysis or report generation. For passive HPOL/VPOL data, `import_passive_pair` is recommended as it runs the full calculation pipeline during import.

### Analysis Tools (5 tools)

| Tool | Description | Parameters |
|------|-------------|------------|
| `list_frequencies` | Get all available frequencies from loaded data | None |
| `analyze_pattern` | Analyze radiation pattern (HPBW, F/B ratio, nulls, sidelobes) | `frequency` (float), `polarization` (hpol/vpol/total) |
| `get_gain_statistics` | Calculate min/max/average gain statistics | `frequency` (float) |
| `compare_polarizations` | Compare HPOL vs VPOL at a frequency | `frequency` (float) |
| `get_all_analysis` | Run complete analysis suite for a frequency | `frequency` (float) |

**Typical usage**: Analyze patterns to understand antenna performance before generating reports.

### Report Tools (3 tools)

| Tool | Description | Parameters |
|------|-------------|------------|
| `generate_report` | Generate formatted DOCX report with plots and AI analysis | `output_path` (str), `options` (dict) |
| `preview_report` | Preview report contents without generating file | `options` (dict) |
| `get_report_options` | Show all available filtering/customization options | None |

**Typical usage**: Generate professional reports after importing and analyzing data.

### Bulk Processing Tools (6 tools)

| Tool | Description | Parameters |
|------|-------------|------------|
| `list_measurement_files` | Scan folder and categorize measurement files | `folder_path` (str) |
| `bulk_process_passive` | Batch process HPOL/VPOL file pairs | `folder_path` (str), `frequencies` (list) |
| `bulk_process_active` | Batch process TRP active measurement files | `folder_path` (str) |
| `validate_file_pair` | Validate that HPOL and VPOL files match | `hpol_path` (str), `vpol_path` (str) |
| `convert_to_cst` | Convert measurement data to CST .ffs format | `hpol_path` (str), `vpol_path` (str), `vswr_path` (optional), `frequency` (float) |
| `batch_analyze_frequencies` | Analyze all frequencies in loaded data | None |

**Typical usage**: Process multiple antennas or frequencies in one operation.

## Quick-Start Workflow

Here's a typical 3-step workflow for analyzing antenna data and generating a report:

### Step 1: Import Data Files

```python
# Option A: Import HPOL/VPOL passive pair (recommended for passive data)
import_passive_pair("C:/measurements/2.4GHz_HPOL.txt", "C:/measurements/2.4GHz_VPOL.txt")

# Option B: Import a single file
import_antenna_file("C:/measurements/2.4GHz_HPOL.txt", "passive")

# Option C: Import entire folder
import_antenna_folder("C:/measurements/wifi_antenna/")

# Option D: Import active TRP file with processing
import_active_processed("C:/measurements/TRP_2.4GHz.txt")

# Verify what was loaded
list_loaded_data()
# Returns: "Loaded 6 measurements at frequencies: [2400, 2450, 2500] MHz"
```

### Step 2: Analyze Patterns

```python
# Check available frequencies
list_frequencies()
# Returns: [2400.0, 2450.0, 2500.0]

# Analyze a specific frequency
analyze_pattern(2450, "total")
# Returns: HPBW, front-to-back ratio, null depths, sidelobe levels

# Or get comprehensive analysis
get_all_analysis(2450)
# Returns: Pattern analysis + gain statistics + polarization comparison
```

### Step 3: Generate Report

```python
# Generate a focused report (recommended for first run)
generate_report("wifi_antenna_report.docx", {
    "frequencies": [2450],
    "polarizations": ["total"],
    "include_2d_plots": True,
    "include_3d_plots": False,
    "ai_executive_summary": True,
    "ai_section_analysis": False
})

# Or generate a comprehensive report with all data
generate_report("full_report.docx", {
    "frequencies": None,  # All frequencies
    "polarizations": ["total", "hpol", "vpol"],
    "include_3d_plots": True,
    "ai_executive_summary": True,
    "ai_section_analysis": True,
    "ai_recommendations": True
})
```

**Result**: Professional DOCX report with plots, tables, and AI-generated insights.

## Report Filtering

The key feature is **smart filtering** to control report complexity.

### Example: Minimal Report

```python
generate_report("antenna_report.docx", {
    "frequencies": [2450],           # Just one frequency
    "polarizations": ["total"],      # Only total gain
    "include_2d_plots": True,
    "include_3d_plots": False,       # Skip large 3D plots
    "ai_executive_summary": True,
    "ai_section_analysis": False     # Skip per-section AI
})
```

### Example: Full Report

```python
generate_report("full_report.docx", {
    "frequencies": None,             # All frequencies
    "polarizations": ["total", "hpol", "vpol"],
    "include_2d_plots": True,
    "include_3d_plots": True,
    "include_gain_tables": True,
    "ai_executive_summary": True,
    "ai_section_analysis": True,
    "ai_recommendations": True
})
```

### Filtering Options

**Content:**
- `frequencies`: List of MHz values, or `null` for all
- `polarizations`: `["total"]`, `["hpol", "vpol"]`, or all three
- `measurements`: Specific files, or `null` for all loaded

**Plots (manage complexity):**
- `include_2d_plots`: 2D pattern cuts (default: true)
- `include_3d_plots`: 3D surface plots (default: **false** - large)
- `include_polar_plots`: Polar radiation patterns (default: true)
- `include_cartesian_plots`: Rectangular plots (default: false)

**Data:**
- `include_gain_tables`: Summary tables (default: true)
- `include_raw_data_tables`: Full data (default: false)
- `max_frequencies_in_table`: Limit rows (default: 10)

**AI Content:**
- `ai_executive_summary`: AI-generated summary (default: true)
- `ai_section_analysis`: AI commentary per section (default: true)
- `ai_recommendations`: Design recommendations (default: true)
- `ai_model`: OpenAI model (default: "gpt-4o-mini")

## Usage Examples

### With Claude Code

```
User: "Import all antenna files from C:/measurements/wifi_antenna/ and generate a report"

Claude: I'll use RFlect MCP to process those files.
[Uses import_antenna_folder tool]
Found 8 measurement files.

[Uses preview_report to check complexity]
This would generate 24 plots. Let me reduce that.

[Uses generate_report with filtering]
Report generated: wifi_antenna_report.docx

Contents:
- 3 frequencies included (2400, 2450, 2500 MHz)
- 2D plots only (3D disabled)
- AI executive summary included
```

### Batch Processing

```
User: "Generate individual reports for each antenna in /project/antennas/"

Claude: I'll process each antenna folder.
[Loops through subfolders]
[Generates report for each]

Generated 5 reports:
- antenna_a_report.docx
- antenna_b_report.docx
- ...
```

## Troubleshooting

### "No data loaded"
**Cause**: Attempting analysis or report generation before importing data.

**Solution**: Run `import_antenna_file` or `import_antenna_folder` first. Verify with `list_loaded_data()`.

### "AI Summary requires API key"
**Cause**: AI features enabled but no AI provider API key configured.

**Solution**:
- **Via RFlect GUI**: Launch RFlect GUI, go to Tools → Manage API Keys, and enter your API key for any supported provider (OpenAI, Anthropic, or Ollama)
- **Via Environment Variable**: Set `OPENAI_API_KEY` or `ANTHROPIC_API_KEY` in your environment, or configure a local Ollama instance
- **Alternative**: Disable AI features in report options: `"ai_executive_summary": false`

### Report too large / Out of memory
**Cause**: Generating too many 3D plots or processing too many frequencies at once.

**Solution**: Use filtering options to reduce complexity:
- Reduce frequencies: `"frequencies": [2450]`
- Disable 3D plots: `"include_3d_plots": false`
- Limit tables: `"max_frequencies_in_table": 5`
- Process one polarization: `"polarizations": ["total"]`

### MCP server not connecting
**Cause**: Incorrect path in config or Python environment issues.

**Solution**:
- Verify absolute path to `server.py` in your MCP config
- Ensure Python environment has required packages: `pip install -r requirements.txt`
- Test server manually: `python server.py` (should start without errors)
- Check Claude Code logs for connection errors

### File import fails
**Cause**: Unsupported file format or corrupted measurement file.

**Solution**:
- Ensure files are in supported format (RFlect native, SATIMO, Orbit FR, etc.)
- Use `list_measurement_files` to validate folder contents before import
- Check file encoding (should be UTF-8 or ASCII)

## Development

### Thread Safety

The MCP server uses `threading.Lock` to protect the `_loaded_measurements` dictionary. All import, analysis, and report tools acquire this lock before reading or writing measurement data. When adding new tools that access shared state, ensure proper lock usage.

### Adding New Tools

1. Create function in appropriate `tools/*.py` file
2. Register with `@mcp.tool()` decorator
3. Ensure thread safety when accessing `_loaded_measurements`
4. Document in this README

### Testing

```bash
# Run server directly
python server.py

# Test with MCP inspector
npx @anthropic/mcp-inspector python server.py
```

## License

Same as RFlect - see parent repository.
