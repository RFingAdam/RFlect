# Quickstart

Get from a fresh install to a rendered radiation pattern in under five minutes.

## 1. Pick a scan type

Launch RFlect. The opening screen asks: **Active**, **Passive**, or **VNA**.

| You measured…                           | Pick      |
|----------------------------------------|-----------|
| TRP / total radiated power over the sphere | Active   |
| HPOL + VPOL patterns (gain reference)  | Passive   |
| S-parameters (S11, S21, group delay)   | VNA       |

## 2. Adjust settings

Settings are scan-type-specific. Reasonable defaults:

| Setting          | Default | Notes                                                  |
|------------------|---------|--------------------------------------------------------|
| Cable loss       | 0.0 dB  | Adds back loss between DUT and chamber receiver        |
| Conducted power  | 4 dBm   | Active only — needed for efficiency                    |
| Limit lines      | off     | VNA — show pass/fail bands                             |
| 3D Z-scale       | auto    | Or fix to a manual dB floor for batch comparison       |

## 3. Import files

`Ctrl+O` or **File → Import**. For passive scans pick the HPOL file; RFlect locates the matching VPOL by filename suffix.

## 4. View results

Plots render automatically. `Ctrl+R` (or `F5`) reprocesses with current settings — handy after tweaking cable loss.

## 5. Save / export

- **File → Save Plots** — PNG export of every visible figure (300 DPI)
- **File → Generate Report (DOCX)** — embeds plots, gain tables, and optional AI executive summary

## Driving RFlect from Claude (MCP)

Once the MCP server is registered (see [MCP installation](../mcp/installation.md)):

```
Process every passive pair in /home/me/lab/wifi_antenna and generate a report.
```

The agent will call `process_folder(folder_path, intent='passive', report=True)` and respond with the DOCX path plus a summary.

See [Recipes](../mcp/recipes.md) for more standard procedures.
