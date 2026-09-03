# Tools Reference

**61** registered MCP tools. Registration is the source of truth:
`rflect-mcp/server.py` plus `register_*_tools` in `rflect-mcp/tools/`.
`rflect://status` and `rflect://help` are resources, not tools.

`batch_analyze_frequencies` appears in older docs; it is **not** registered.

## Import (7)

| Tool | Purpose |
|------|---------|
| `import_antenna_file(file_path, scan_type)` | Single measurement file (`scan_type` auto / passive / active) |
| `import_antenna_folder(folder_path, pattern, scan_type)` | Every file in folder matching glob |
| `import_passive_pair(hpol_file, vpol_file, cable_loss, name)` | HPOL+VPOL → full passive calc pipeline |
| `import_active_processed(file_path, name)` | Single TRP file with full active calc pipeline |
| `list_loaded_data()` | Currently loaded measurements |
| `clear_data()` | Drop all loaded measurements |
| `get_measurement_details(measurement_name)` | Inspect one loaded measurement |

## Analysis (7)

| Tool | Purpose |
|------|---------|
| `list_frequencies()` | Available frequencies across loaded data |
| `analyze_pattern(frequency, polarization)` | HPBW, F/B ratio, nulls, sidelobes |
| `get_gain_statistics(frequency)` | Min / max / spherical avg gain |
| `compare_polarizations(frequency)` | AR / tilt / XPD / sense |
| `get_all_analysis(frequency)` | Combined pattern + gain + polarization |
| `extrapolate_to_frequency(hpol_file, vpol_file, target_frequency, fit_degree)` | Polynomial gain extrapolation outside the measured band |
| `get_horizon_statistics(frequency, theta_min, theta_max, gain_threshold)` | Horizon-plane / maritime-band stats |

## Reports (3)

| Tool | Purpose |
|------|---------|
| `generate_report(output_path, options, title, metadata, narrative)` | DOCX with plots, tables, deterministic prose (optional agent `narrative`) |
| `preview_report(options)` | What the report would contain. No file written |
| `get_report_options()` | All filtering / customization options |

## Bulk (5)

| Tool | Purpose |
|------|---------|
| `list_measurement_files(folder_path)` | Scan folder, categorize HPOL/VPOL/TRP/VSWR |
| `bulk_process_passive(folder_path, frequencies, cable_loss, save_path, datasheet_plots)` | Batch HPOL/VPOL pairs |
| `bulk_process_active(folder_path, save_path, interpolate)` | Batch TRP files |
| `validate_file_pair(hpol_path, vpol_path)` | Are these two files a valid pair? |
| `convert_to_cst(hpol, vpol, vswr, frequency, cable_loss, output_path)` | Export CST `.ffs` |

## UWB (3)

| Tool | Purpose |
|------|---------|
| `calculate_sff_from_files(file_paths, pulse_type, min_freq_ghz, max_freq_ghz)` | SFF per angle from a list of files |
| `analyze_uwb_channel(file_path, distance_m, pulse_type)` | Full UWB analysis from one file |
| `get_impedance_bandwidth(file_path, threshold_dB)` | S11 bandwidth metrics |

## Cal Drift (12)

| Tool | Purpose |
|------|---------|
| `cal_drift_ingest(directory)` | Walk a directory of `TRP Cal *.txt` and record each |
| `cal_drift_list_runs(antenna, band)` | List recorded runs, optionally filtered |
| `cal_drift_compare(baseline_run_id, current_run_id, max_delta_rows)` | Per-frequency ΔdB + consistency |
| `cal_drift_report(baseline_run_id, current_run_id, output_path, format)` | Export markdown / pdf / png |
| `cal_drift_history_dir()` | Current history directory |
| `cal_drift_set_history_dir(directory)` | Persist a new history directory |
| `cal_drift_set_setup_group(run_id, setup_group)` | Tag a run's methodology epoch |
| `cal_drift_set_notes(run_id, notes)` | Free-text operator notes |
| `cal_drift_alert(...)` | Threshold alerts on a new run vs baseline |
| `cal_drift_monitor(...)` | Scheduled / repeated drift check |
| `cal_drift_recert_check(...)` | Gain-standard recertification reminder |
| `cal_drift_cable_loss_history(...)` | Cable-loss `.s2p` history |

## Orchestration (1)

| Tool | Purpose |
|------|---------|
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

| Tool | Purpose |
|------|---------|
| `analyze_iperf_angle_sweep(session_dir, reference_session_dir, out_dir, mean_threshold_mbps, worst_threshold_mbps)` | Per-angle throughput delta of an installed antenna vs a reference, across azimuth |

Compares two bench iperf sessions. An installed-antenna session and a matched reference-antenna session recorded at the same azimuth angles. For each `(channel, mode)` cell it computes the per-angle deltas (installed − reference) and a roll-up: mean, median, worst-angle, best-angle, spread, and p10/p90. It writes `summary.csv`, `summary.json`, a polar PNG per cell, and a markdown `report.md` with a configurable adequacy verdict.

Reads only the documented bench `session.json` shape (`wifi_only` runs carrying `angle_deg` and `aggregate.overall_mbps`); it has no dependency on any specific bench harness. See [Recipes](recipes.md).

## Comparison (2)

| Tool | Purpose |
|------|---------|
| `compare_antennas(measurement_names, reference, out_csv)` | Cross-measurement overlay: per-frequency peak gain/TRP, deltas vs a reference |
| `summarize_antennas()` | Roll-up of currently loaded measurements |

`compare_antennas` operates on the loaded-measurement store (import first).

## VNA (2)

| Tool | Purpose |
|------|---------|
| `analyze_s11(freq_hz, s11_db, threshold_db, include_vswr_curve)` | Return loss / impedance bandwidth / VSWR / resonance |
| `analyze_group_delay(freq_hz, phase_deg, band_start_hz, band_stop_hz)` | Group delay + in-band flatness from a transmission-phase sweep |

These take agent-supplied arrays (pair with measured S-parameter files via UWB/import tools as needed).

## Propagation, MIMO, calibration (3)

| Tool | Purpose |
|------|---------|
| `estimate_link_budget(...)` | Max range + link margin (Friis / log-distance / ITU-indoor); optional fade margin |
| `analyze_mimo_diversity(ecc, snr_db, ...)` | Vaughan-Andersen diversity gain, 2×2 capacity, isolation rating |
| `generate_active_cal(...)` | Active chamber cal file + summary; auto-records into cal-drift history |

## Compliance, uncertainty, S-parameters, statistics (5)

| Tool | Purpose |
|------|---------|
| `check_spec_compliance(...)` | PASS/FAIL against user-supplied limit lines |
| `check_regulatory_eirp(...)` | Measured EIRP vs built-in FCC/ETSI *reference* ceilings (not a substitute for the live rule text) |
| `compute_uncertainty_budget(...)` | RSS uncertainty budget (CTIA / ISO-GUM style) |
| `analyze_multiport_touchstone(...)` | `.s3p` / `.s4p` + mixed-mode S-parameters |
| `average_patterns(patterns)` | Average repeat pattern grids and report spread |

## RF methods (8)

Thin wrappers over `plot_antenna.rf_methods`. `ctia_test_plan_template` is a **scaffold** for a campaign, not a certification.

| Tool | Purpose |
|------|---------|
| `analyze_axial_ratio(mag_v, mag_h, phase_diff_deg)` | AR / tilt / sense from orthogonal V & H |
| `three_antenna_gain_method(...)` | Absolute gain from three pairwise S21 ratios |
| `synthesize_array(...)` | Uniform linear-array factor, steering, grating-lobe flag |
| `near_field_to_far_field(...)` | Planar NF→FF via 2D FFT |
| `time_gate_sparameters(...)` | IFFT window / FFT time-gating |
| `deembed_port_extension(...)` | Electrical-length / fixture delay removal |
| `calculate_tis(...)` | CTIA Total Isotropic Sensitivity from an EIS sphere |
| `ctia_test_plan_template(...)` | Starting TRP/TIS grid/channel plan; verify against the current CTIA revision |

## Instruments (2)

| Tool | Purpose |
|------|---------|
| `vna_read_trace(...)` | SCPI VNA sweep; in-memory mock unless `pyvisa` + a VISA resource |
| `positioner_scan_grid(...)` | Chamber positioner grid; mock unless serial backend |

Install live hardware backends with `pip install -e ".[instruments]"`.

## Source of truth

Every `register_*_tools` call in `rflect-mcp/server.py`. Inventory summary: [MCP_STATUS.md](../MCP_STATUS.md).
