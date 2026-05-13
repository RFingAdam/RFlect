# MCP Overview

RFlect ships a [Model Context Protocol](https://modelcontextprotocol.io/) server so AI assistants (Claude Code, Cline, Continue, etc.) can drive RFlect programmatically — no GUI required.

## What you can do over MCP

- Import passive HPOL/VPOL pairs, active TRP files, S-parameter sweeps
- Run analysis: HPBW, F/B ratio, gain stats, polarization, UWB SFF, impedance bandwidth
- Generate DOCX reports with optional AI executive summaries
- Track calibration drift across time
- **Run a standard procedure on a folder with a single call** via [`process_folder`](recipes.md) — auto-detects intent, runs the right workflow, optionally generates a report

## Tool count

34 tools across 7 categories. See [Tools Reference](tools-reference.md) for the full list with signatures.

| Category          | Count |
|-------------------|------:|
| Import            | 6     |
| Analysis          | 5     |
| Reports           | 3     |
| Bulk              | 5     |
| UWB               | 3     |
| Cal Drift         | 8     |
| Orchestration     | 1     |
| Misc              | 3     |

## Why an orchestrator?

Before v4.2.0, an MCP client wanting to "process this folder and give me a report" had to:

1. `list_measurement_files(folder)` to see what's inside
2. Decide passive vs active based on filenames
3. `bulk_process_passive(folder, freqs)` or `bulk_process_active(folder)`
4. `import_passive_pair(...)` or `import_active_processed(...)` for each one
5. `generate_report(path, options)`

That's a five-step chain that every agent had to script. `process_folder` collapses all of it into one call:

```python
process_folder("/path/to/captures", intent="auto", report=True)
```

See [Recipes](recipes.md) for the full set of standard procedures.

## Where to next

- [Installation](installation.md) — wire RFlect into Claude Code / Cline / generic clients
- [Tools Reference](tools-reference.md) — every tool, signature, return shape
- [Recipes](recipes.md) — standard procedures for common workflows
- [Troubleshooting](troubleshooting.md) — what to do when it doesn't connect
