# Tools Reference

All 34 RFlect MCP tools, organized by category. Full signatures and return shapes.

## Import (6)

| Tool                       | Purpose                                                     |
|----------------------------|-------------------------------------------------------------|
| `import_antenna_file(file_path, scan_type)` | Single measurement file (auto-detects scan type from filename) |
| `import_antenna_folder(folder_path, pattern, scan_type)` | Every file in folder matching glob |
| `import_passive_pair(hpol_file, vpol_file, cable_loss, name)` | HPOL+VPOL → full passive calc pipeline |
| `import_active_processed(file_path, name)` | Single TRP file with full active calc pipeline |
| `list_loaded_data()`       | Currently loaded measurements                               |
| `clear_data()`             | Drop all loaded measurements                                |

## Analysis (5)

| Tool                       | Purpose                                                     |
|----------------------------|-------------------------------------------------------------|
| `list_frequencies()`       | Available frequencies across loaded data                    |
| `analyze_pattern(frequency, polarization)` | HPBW, F/B ratio, nulls, sidelobes                |
| `get_gain_statistics(frequency)` | Min / max / spherical avg gain                        |
| `compare_polarizations(frequency)` | AR / tilt / XPD / sense                              |
| `get_all_analysis(frequency)` | Combined pattern + gain + polarization                   |

## Reports (3)

| Tool                       | Purpose                                                     |
|----------------------------|-------------------------------------------------------------|
| `generate_report(output_path, options, title, metadata)` | DOCX with plots, tables, AI summary  |
| `preview_report(options)`  | What the report would contain — no file written             |
| `get_report_options()`     | All filtering / customization options                       |

## Bulk (5)

| Tool                       | Purpose                                                     |
|----------------------------|-------------------------------------------------------------|
| `list_measurement_files(folder_path)` | Scan folder, categorize HPOL/VPOL/TRP/VSWR      |
| `bulk_process_passive(folder_path, frequencies, cable_loss, save_path, datasheet_plots)` | Batch HPOL/VPOL pairs |
| `bulk_process_active(folder_path, save_path, interpolate)` | Batch TRP files                       |
| `validate_file_pair(hpol_path, vpol_path)` | Are these two files a valid pair?               |
| `convert_to_cst(hpol, vpol, vswr, frequency, cable_loss, output_path)` | Export CST `.ffs`        |

## UWB (3)

| Tool                       | Purpose                                                     |
|----------------------------|-------------------------------------------------------------|
| `calculate_sff_from_files(file_paths, pulse_type, min_freq_ghz, max_freq_ghz)` | SFF per angle from a list of files |
| `analyze_uwb_channel(file_path, distance_m, pulse_type)` | Full UWB analysis from one file       |
| `get_impedance_bandwidth(file_path, threshold_dB)` | S11 bandwidth metrics                       |

## Cal Drift (8)

| Tool                       | Purpose                                                     |
|----------------------------|-------------------------------------------------------------|
| `cal_drift_ingest(directory)` | Walk a directory of `TRP Cal *.txt` and record each     |
| `cal_drift_list_runs(antenna, band)` | List recorded runs, optionally filtered          |
| `cal_drift_compare(baseline_run_id, current_run_id, max_delta_rows)` | Per-frequency ΔdB + consistency  |
| `cal_drift_report(baseline_run_id, current_run_id, output_path, format)` | Export markdown / pdf / png |
| `cal_drift_history_dir()`  | Current history directory                                   |
| `cal_drift_set_history_dir(directory)` | Persist a new history directory                  |
| `cal_drift_set_setup_group(run_id, setup_group)` | Tag a run's methodology epoch          |
| `cal_drift_set_notes(run_id, notes)` | Free-text operator notes                          |

## Orchestration (1)

| Tool                       | Purpose                                                     |
|----------------------------|-------------------------------------------------------------|
| `process_folder(folder_path, intent, report, freqs, report_path)` | Scan folder, pick + run the right workflow |

Returns:
```python
{
  "intent_used":        "passive" | "active" | "cal_drift" | "uwb" | None,
  "files_scanned":      int,
  "files_processed":    int,
  "frequencies_loaded": list[float],
  "warnings":           list[str],     # never raises; failures here
  "report_path":        str | None,
  "extra":              dict           # intent-specific (uwb results, cal_drift run_ids, …)
}
```

See [Recipes](recipes.md) for usage patterns.

## Misc (3)

| Tool                       | Purpose                                                     |
|----------------------------|-------------------------------------------------------------|
| `get_measurement_details(measurement_name)` | Inspect one loaded measurement                |
| `batch_analyze_frequencies()` | Re-run analysis across all loaded frequencies            |
| `rflect://help`            | Help resource — server prints categorized tool list         |

## Source of truth

Registration entry points:

- `rflect-mcp/tools/import_tools.py` — `register_import_tools(mcp)`
- `rflect-mcp/tools/analysis_tools.py` — `register_analysis_tools(mcp)`
- `rflect-mcp/tools/report_tools.py` — `register_report_tools(mcp)`
- `rflect-mcp/tools/bulk_tools.py` — `register_bulk_tools(mcp)`
- `rflect-mcp/tools/uwb_tools.py` — `register_uwb_tools(mcp)`
- `rflect-mcp/tools/cal_drift_tools.py` — `register_cal_drift_tools(mcp)`
- `rflect-mcp/tools/orchestration.py` — `register_orchestration_tools(mcp)`

All registered in `rflect-mcp/server.py`.
