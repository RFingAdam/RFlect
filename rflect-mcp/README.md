# RFlect MCP Server

AI-powered antenna analysis and report generation via Model Context Protocol (MCP).

Enables Claude Code, Cline, and other AI assistants to programmatically analyze antenna measurements and generate reports.

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
      "args": ["path/to/rflect-mcp/server.py"]
    }
  }
}
```

### Cline / Other MCP Clients

Configure similarly with the path to `server.py`.

## Available Tools

### Import Tools

| Tool | Description |
|------|-------------|
| `import_antenna_file(file_path, scan_type)` | Import single measurement file |
| `import_antenna_folder(folder_path, pattern)` | Import all files from folder |
| `list_loaded_data()` | Show loaded measurements |
| `clear_data()` | Clear all loaded data |

### Analysis Tools

| Tool | Description |
|------|-------------|
| `list_frequencies()` | Get available frequencies |
| `analyze_pattern(frequency, polarization)` | Pattern analysis (HPBW, F/B ratio, nulls) |
| `get_gain_statistics(frequency)` | Min/max/avg gain |
| `compare_polarizations(frequency)` | HPOL vs VPOL comparison |
| `get_all_analysis(frequency)` | Complete analysis |

### Report Tools

| Tool | Description |
|------|-------------|
| `generate_report(output_path, options)` | Generate DOCX report |
| `preview_report(options)` | Preview what report would contain |
| `get_report_options()` | Show available filtering options |

### Bulk Processing Tools

| Tool | Description |
|------|-------------|
| `list_measurement_files(folder_path)` | Scan folder for measurement files |
| `bulk_process_passive(folder_path, frequencies)` | Batch process HPOL/VPOL pairs |
| `bulk_process_active(folder_path)` | Batch process TRP files |
| `validate_file_pair(hpol_path, vpol_path)` | Validate HPOL/VPOL file pairing |
| `convert_to_cst(hpol_path, vpol_path, vswr_path, frequency)` | Convert to CST .ffs format |

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
Run `import_antenna_file` or `import_antenna_folder` first.

### "AI Summary requires API key"
Set your OpenAI API key via RFlect GUI (Help → Manage API Key) or environment variable.

### Report too large
Use filtering options:
- Reduce frequencies: `"frequencies": [2450]`
- Disable 3D plots: `"include_3d_plots": false`
- Limit tables: `"max_frequencies_in_table": 5`

## Development

### Adding New Tools

1. Create function in appropriate `tools/*.py` file
2. Register with `@mcp.tool()` decorator
3. Document in this README

### Testing

```bash
# Run server directly
python server.py

# Test with MCP inspector
npx @anthropic/mcp-inspector python server.py
```

## License

Same as RFlect - see parent repository.
