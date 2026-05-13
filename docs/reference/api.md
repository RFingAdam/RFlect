# API Reference

Pure-Python entry points you can import from `plot_antenna`.

## `AntennaAnalyzer` — `plot_antenna.ai_analysis`

GUI-independent analysis engine. Same code path used by the AI chat assistant and the MCP analysis tools.

```python
from plot_antenna.ai_analysis import AntennaAnalyzer

data = {
    "phi":         phi_angles,      # 1-D array
    "theta":       theta_angles,    # 1-D array
    "total_gain":  gain_array,      # 2-D or flattened
    "h_gain":      h_pol_array,
    "v_gain":      v_pol_array,
}

analyzer = AntennaAnalyzer(
    measurement_data=data,
    scan_type="passive",            # or "active"
    frequencies=[2400.0, 2450.0],
)

stats   = analyzer.get_gain_statistics(frequency=2450.0)
pattern = analyzer.analyze_pattern(frequency=2450.0)
polz    = analyzer.compare_polarizations(frequency=2450.0)
```

## File parsers — `plot_antenna.file_utils`

| Function                         | Returns                                          |
|----------------------------------|--------------------------------------------------|
| `read_passive_file(path)`        | `(parsed_data, start_phi, stop_phi, inc_phi, start_theta, stop_theta, inc_theta)` |
| `read_active_file(path)`         | Dict with `Frequency`, axis ranges, `H_Power_dBm`, `V_Power_dBm` |
| `parse_2port_data(path)`         | DataFrame with `! Stimulus(Hz)`, `S11(dB)`, `S21(dB)`, `S21(s)` |
| `parse_touchstone_to_dataframe(path)` | Same DataFrame shape from `.s2p`            |
| `batch_process_passive_scans(...)`| Process every HPOL/VPOL pair in a folder        |
| `batch_process_active_scans(...)` | Process every TRP file in a folder              |
| `convert_HpolVpol_files(...)`     | Export to CST `.ffs`                            |
| `validate_hpol_vpol_files(...)`   | `(bool, error_msg)` tuple                       |

## Calculations — `plot_antenna.calculations`

| Function                         | Returns                                          |
|----------------------------------|--------------------------------------------------|
| `calculate_passive_variables(...)` | Total/H/V gain arrays + theta, phi grids       |
| `calculate_active_variables(...)`| TRP, H/V TRP, total power 2D, min/nom stats      |
| `extract_passive_frequencies(path)` | List of MHz available in a passive file       |

## UWB — `plot_antenna.uwb_analysis`

| Function                         | Returns                                          |
|----------------------------------|--------------------------------------------------|
| `calculate_sff(freq, s21, pulse_type)` | SFF + quality label + peak delay         |
| `calculate_sff_vs_angle(angle_data, pulse_type)` | Per-angle SFF + mean             |
| `compute_group_delay_from_s21(freq, s21)` | Group delay + variation + distance error |
| `extract_transfer_function(freq, s21, distance_m)` | `H(f)` mag/phase + flatness      |
| `compute_impulse_response(freq, H_complex)` | Pulse width + ringing                |
| `analyze_return_loss(freq, s11_dB, threshold_dB)` | Bandwidth + band edges + FBW %    |
| `parse_touchstone(path)`         | Dict with `freq_hz`, `s11`, `s21`                |

## Calibration drift — `plot_antenna.cal_drift`

| Function                         | Purpose                                          |
|----------------------------------|--------------------------------------------------|
| `record_run(cal_result, ...)`    | Record one calibration run with metadata         |
| `import_historical_dir(path)`    | Walk + record every `TRP Cal *.txt` in dir       |
| `list_runs(antenna, band)`       | DataFrame of recorded runs                       |
| `compute_drift(baseline_id, current_id)` | Drift result object                      |
| `result_to_dict(result, max_delta_rows)` | JSON-safe dict                           |
| `export_markdown(result, path)`  | Markdown drift report                            |
| `export_pdf(result, path)`       | PDF drift report                                 |
| `render_delta_plot(result, out_path)` | PNG drift plot                              |
| `history_dir()` / `set_history_dir(path)` | Where history lives                      |
| `set_setup_group(run_id, group)` | Tag a run's methodology epoch                    |
| `update_notes(run_id, text)`     | Operator notes                                   |

## LLM provider — `plot_antenna.llm_provider`

```python
from plot_antenna.llm_provider import get_provider, UnifiedMessage, ToolDefinition

provider = get_provider("openai")  # or "anthropic", "ollama"
response = provider.chat(
    messages=[UnifiedMessage(role="user", content="...")],
    tools=[ToolDefinition(name="get_gain_statistics", ...)],
)
```

See `plot_antenna/llm_provider.py` for `BaseLLMProvider`, `LLMResponse`, `ToolCall`.

## API key management — `plot_antenna.api_keys`

```python
from plot_antenna.api_keys import store_key, get_key, delete_key

store_key("openai", "sk-...")
key = get_key("openai")        # returns plaintext from keyring or Fernet store
delete_key("openai")
```

Keys are AES-128 (Fernet) encrypted at rest with PBKDF2-HMAC-SHA256 (600 K iterations) and a machine-ID-derived salt. Storage priority: OS keyring → Fernet file → environment variable → `.env`.
