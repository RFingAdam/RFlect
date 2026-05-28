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

## Validation (1)

| Tool                       | Purpose                                                     |
|----------------------------|-------------------------------------------------------------|
| `analyze_iperf_angle_sweep(session_dir, reference_session_dir, out_dir, mean_threshold_mbps, worst_threshold_mbps)` | Per-angle throughput delta of an installed antenna vs a reference, across azimuth |

Compares two bench iperf sessions — an installed-antenna session and a matched reference-antenna session recorded at the same azimuth angles. For each `(channel, mode)` cell it computes the per-angle deltas (installed − reference) and a roll-up: mean, median, worst-angle, best-angle, spread, and p10/p90. It writes `summary.csv`, `summary.json`, a polar PNG per cell, and a markdown `report.md` with a configurable adequacy verdict.

Returns:
```python
{
  "summary_csv":  str | None,
  "summary_json": str | None,
  "report_md":    str | None,
  "polar_pngs":   list[str],
  "n_cells":      int,
  "warnings":     list[str],     # never raises; failures here
}
```

Reads only the documented bench `session.json` shape (`wifi_only` runs carrying `angle_deg` and `aggregate.overall_mbps`); it has no dependency on any specific bench harness. See [Recipes](recipes.md) for usage patterns.

## RF Analysis (6)

Deterministic wrappers over `plot_antenna` math (no LLM, no network). Each returns a structured dict and never raises; failures appear in a `warnings` list.

| Tool | Purpose |
|------|---------|
| `compare_antennas(measurement_names, reference, out_csv)` | Cross-measurement overlay: per-frequency peak gain/TRP, deltas vs a reference, best-per-frequency, best-overall, optional CSV |
| `analyze_s11(freq_hz, s11_db, threshold_db, include_vswr_curve)` | Return loss / impedance bandwidth / VSWR / resonance from an S11 sweep |
| `analyze_group_delay(freq_hz, phase_deg, band_start_hz, band_stop_hz)` | Group delay + in-band flatness from a transmission-phase sweep (phase only) |
| `estimate_link_budget(tx_power_dbm, rx_sensitivity_dbm, tx_gain_dbi, rx_gain_dbi, freq_mhz, ...)` | Max range + link margin via Friis / log-distance / ITU-indoor; optional Rayleigh/Rician fade margin and de-rated reliable range |
| `analyze_mimo_diversity(ecc, snr_db, snr_sweep_db, fading, rician_k)` | ECC → Vaughan-Andersen diversity gain, 2×2 capacity, isolation rating, optional capacity-vs-SNR curve |
| `generate_active_cal(power_measurement_file, gain_standard_file, hpol_file, vpol_file, freq_list, cable_loss)` | Generate an active chamber cal file + summary; auto-records into cal-drift history |

`compare_antennas` operates on the loaded-measurement store (import first). The others take agent-supplied arrays/scalars — pair them with measured values (e.g. use the peak gain from `get_gain_statistics` as `tx_gain_dbi`, or an ECC from measured patterns for `analyze_mimo_diversity`).

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
