# AI Reports

DOCX reports with embedded plots, gain tables, and optional AI-generated executive summaries.

## What gets included

| Section                | AI involvement                                    |
|------------------------|---------------------------------------------------|
| Cover page             | None — template                                   |
| Measurement summary    | None — tables generated from loaded data          |
| 2D / 3D plots          | None — matplotlib renders                         |
| Gain tables            | None — computed                                   |
| Executive summary      | **AI** — prose from the measurement metrics       |
| Per-section commentary | **AI** — describes what each plot shows           |
| Design recommendations | **AI** — heuristics + LLM judgment                |

## In the GUI

File → Generate Report (DOCX). Pick output path. Options dialog lets you toggle each section.

## Programmatic / MCP

```python
generate_report("/tmp/wifi_report.docx", {
    "frequencies":          [2400, 2450, 2500],
    "polarizations":        ["total", "hpol", "vpol"],
    "include_2d_plots":     True,
    "include_3d_plots":     False,        # large
    "include_gain_tables":  True,
    "ai_executive_summary": True,
    "ai_section_analysis":  True,
    "ai_recommendations":   True,
})
```

`preview_report(options)` returns what would be included without writing the DOCX.

## Filtering options

| Option                     | Default | Notes                                       |
|----------------------------|---------|---------------------------------------------|
| `frequencies`              | None    | List of MHz, or None for all                |
| `polarizations`            | `["total"]` | `total` / `hpol` / `vpol`               |
| `include_2d_plots`         | true    | Polar + cartesian cuts                      |
| `include_3d_plots`         | false   | Large files — opt in explicitly             |
| `include_gain_tables`      | true    | Summary tables                              |
| `include_raw_data_tables`  | false   | Full per-angle data — large                 |
| `max_frequencies_in_table` | 10      | Limit rows                                  |
| `ai_executive_summary`     | true    | Auto-falls back to template if no provider  |
| `ai_section_analysis`      | true    |                                             |
| `ai_recommendations`       | true    |                                             |
| `ai_model`                 | `gpt-4o-mini` | Override default                       |

## When AI is unavailable

If no provider is configured (or the API errors out), report generation does **not** fail. Each AI section falls back to a template-only string, and the DOCX is generated as usual.

## Branding

Custom branding (logos, colors, footers) is partially implemented. See `plot_antenna/save.py:_build_branded_docx` and the [roadmap](status.md).

## Bulk reports

```python
process_folder("/path/to/captures", intent="passive", report=True)
```

Generates one report covering everything in the folder. For one-report-per-antenna, loop over subfolders:

```python
import os, pathlib
root = pathlib.Path("/path/to/antennas")
for child in sorted(root.iterdir()):
    if child.is_dir():
        process_folder(str(child), intent="passive", report=True,
                       report_path=str(child / "report.docx"))
```
