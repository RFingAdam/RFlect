# Recipes — Standard Procedures

Standard RFlect procedures you can hand to Claude (or invoke programmatically). All driven by [`process_folder`](tools-reference.md#orchestration-1).

## Standard Passive Procedure

> "Process every HPOL/VPOL pair in this folder and generate a report."

```python
process_folder(
    "/path/to/wifi_antenna",
    intent="passive",
    report=True,
)
```

What it does:
- Pairs every `*_HPol.txt` with its matching `*_VPol.txt`
- Runs `batch_process_passive_scans` (gain, efficiency, directivity at every chamber-recorded frequency)
- Generates a DOCX report at `<folder>/process_folder_report.docx`

Restrict to specific frequencies:

```python
process_folder(
    "/path/to/wifi_antenna",
    intent="passive",
    freqs=[2400, 2450, 2500],
    report=True,
)
```

## Standard Active TRP Procedure

> "Process every TRP file in this folder."

```python
process_folder(
    "/path/to/trp_runs",
    intent="active",
    report=True,
)
```

What it does:
- Picks up every `*.txt` containing `TRP` (excluding `TRP Cal *.txt`)
- Runs `batch_process_active_scans` — TRP, H/V power, 2D/3D patterns, maritime stats
- Generates a DOCX report

## Cal-Drift Sweep

> "Ingest this archive of historical calibrations."

```python
process_folder("/path/to/historical_cal_archive", intent="cal_drift")
# returns: {ingested, skipped_duplicate, failed, run_ids: [...]}
```

Then compare runs:

```python
runs = cal_drift_list_runs(antenna="BLPA", band="690-2700")
cal_drift_compare(
    baseline_run_id=runs[0]["run_id"],
    current_run_id=runs[-1]["run_id"],
)
```

See [Cal Drift](../user-guide/cal-drift.md).

## UWB Characterization

> "Run UWB channel analysis on every S-parameter sweep here."

```python
process_folder("/path/to/uwb_sweep", intent="uwb")
# extra.results = [{file, analysis}, ...] with SFF, group delay, transfer function, impulse response
```

## Auto / "Just figure it out"

> "Process this folder for me."

```python
process_folder("/path/to/captures")  # intent="auto"
```

Auto-detect priority: `cal_drift > passive > active > uwb`. Mixed folders proceed with the winner and add a `mixed_intents_detected: …` warning to the result.

## Sample return shape

```json
{
  "intent_used": "passive",
  "files_scanned": 4,
  "files_processed": 4,
  "frequencies_loaded": [2400.0, 2450.0, 2500.0],
  "warnings": [],
  "report_path": "/path/to/wifi_antenna/process_folder_report.docx"
}
```

When something goes sideways (no match, missing folder, partial pair, per-file UWB failure, report write error) the call **does not raise** — warnings list the problems:

```json
{
  "intent_used": "passive",
  "files_scanned": 3,
  "files_processed": 2,
  "frequencies_loaded": [2400.0],
  "warnings": [
    "unmatched_hpol: ['PassiveTest_Z_HPol.txt']",
    "invalid_frequencies_dropped: [9999.0]"
  ],
  "report_path": null
}
```

## Prompt patterns for AI clients

If you're driving Claude / Cline / Continue with natural language, these work well:

- "Run the standard passive procedure on /Users/me/lab/wifi and give me a report."
- "Ingest the historical cal archive at /srv/cal_archive and tell me how many new runs were recorded."
- "Process the UWB folder and summarize the worst-fidelity angle."

The agent will translate these into `process_folder(...)` calls and follow up with the right secondary tools (e.g., `cal_drift_compare` for the drift case).

## Agent-authored report narrative

RFlect generates report prose **deterministically** by default — no LLM, no API key. Because the MCP client driving RFlect *is* an LLM, the higher-quality path is for the agent to read the computed analysis and author the narrative itself, then hand it to `generate_report`:

```python
# 1. Compute the facts (deterministic).
stats = get_gain_statistics(2450.0)
pol   = compare_polarizations(2450.0)

# 2. The agent writes the prose from those facts, then:
generate_report(
    "/tmp/report.docx",
    metadata={"title": "BLE Antenna Report", "antenna_type": "Chip"},
    narrative={
        "executive_summary": "Peak gain of +1.8 dBi at 2450 MHz with ...",
        "section_analysis":  {"WiFi-Main": "Pattern is near-omnidirectional ..."},
        "recommendations":   "- Re-tune match toward band center\n- Re-check 2480 MHz",
        "captions":          {"3D_TRP_total_2450.0MHz.png": "Total radiated power, 2450 MHz"},
    },
)
```

Any `narrative` key you omit falls back to RFlect's data-driven text, so you can author just the executive summary and let everything else stay deterministic. There is no `ai_*` option and no provider to configure.
