# Cal Drift

Track TRP-calibration files across time, compare any two runs, and flag setup-group mismatches.

## What it solves

Chamber calibrations drift. Cables age. Connectors loosen. Without a history, you can't tell whether the 5 dB difference you're seeing between today and 2024 is a real chamber drift or just a different methodology epoch.

## How it works

1. **Ingest** — point at a directory of archived `TRP Cal *.txt` files (and optional `TRP Cal Summary *.txt` siblings). Each is recorded with a unique `run_id`, a content SHA-256, and metadata pulled from the file header + filename.
2. **Compare** — pick any two runs as `baseline` and `current`. RFlect computes per-frequency ΔdB (H and V), a consistency-diff (did the gain standard change? same cable? same fixture?), and a missing-frequency audit.
3. **Report** — export markdown, PDF, or PNG drift reports for review or filing.

## Setup groups (methodology epochs)

A `setup_group` is a free-text tag like `pre-2024-cable-change` or `2026-v2-mount`. When you compare two runs in different groups, the consistency tab loudly flags the mismatch — that comparison may not be apples-to-apples.

```python
cal_drift_set_setup_group(run_id, "pre-2024-cable-change")
cal_drift_set_setup_group(run_id_2026, "2026-v2-mount")
# Comparing the two will surface a setup_group mismatch warning
```

## History storage

By default, RFlect writes history under your user profile (`~/.local/share/RFlect/cal_drift/` on Linux, equivalent on macOS/Windows). Override via:

```python
cal_drift_set_history_dir("/srv/shared/rflect_cal_history")
```

Or set the env var `RFLECT_CAL_DRIFT_DIR` (used by the test suite).

## Auto-capture

When you run Tools → Active Chamber Calibration to generate a new cal file, RFlect automatically records the result into the history. No manual ingest needed for fresh calibrations.

## MCP tools (8)

| Tool                                  | Purpose                                            |
|---------------------------------------|----------------------------------------------------|
| `cal_drift_ingest(directory)`         | Walk a directory of `TRP Cal *.txt` files          |
| `cal_drift_list_runs(antenna, band)`  | List recorded runs, optionally filtered            |
| `cal_drift_compare(baseline, current)`| Per-frequency ΔdB + consistency diff               |
| `cal_drift_report(baseline, current, path, format)` | Export markdown / pdf / png        |
| `cal_drift_history_dir()`             | Show current history directory                     |
| `cal_drift_set_history_dir(dir)`      | Persist new history directory                      |
| `cal_drift_set_setup_group(run_id, group)` | Tag a run with its methodology epoch           |
| `cal_drift_set_notes(run_id, text)`   | Free-text operator notes                           |

## Recipe

```python
# 1. Ingest the archived calibration directory once
process_folder("/path/to/historical_cal_archive", intent="cal_drift")

# 2. List what's now recorded
cal_drift_list_runs(antenna="BLPA", band="690-2700")

# 3. Compare two specific runs
cal_drift_compare(
    baseline_run_id="<early_id>",
    current_run_id="<recent_id>",
)

# 4. Export a PDF report
cal_drift_report(
    baseline_run_id="<early>",
    current_run_id="<recent>",
    output_path="/tmp/drift.pdf",
    format="pdf",
)
```

## See also

- `plot_antenna/cal_drift.py` — implementation
- `tests/test_cal_drift.py` — 18 tests covering ingest, compare, consistency flags, idempotency, exports
