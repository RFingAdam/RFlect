# RFlect MCP: Status & Tool Inventory

**Last Updated**: 2026-09-03
**Current Version**: v6.1.0
**Status**: Stable

---

## No LLM dependency

As of **v5.0.0**, RFlect makes **no outbound LLM/API calls and requires no API key or paid subscription.** It is a deterministic RF analysis + rendering toolkit. The in-app AI chat assistant, AI report generation, LLM provider abstraction, and encrypted API-key store were all removed.

When RFlect is driven over MCP, the AI agent (Claude Code, Cline, Continue, …) *is* the LLM. RFlect supplies computed data and rendering; the agent supplies any natural-language narrative. Report prose is data-driven by default, or authored by the agent and passed to `generate_report` via the `narrative` parameter.

## MCP server (62 tools)

`rflect-mcp/server.py` registers the groups below. Counts are from `@mcp.tool()` / `mcp.tool()` registrations. See `docs/mcp/tools-reference.md` for signatures.

| Category | Count | Examples |
|----------|------:|----------|
| Import | 7 | `import_antenna_file`, `import_passive_pair`, `get_measurement_details` |
| Analysis | 7 | `analyze_pattern`, `get_gain_statistics`, `get_horizon_statistics`, `extrapolate_to_frequency` |
| Reports | 3 | `generate_report`, `preview_report`, `get_report_options` |
| Bulk | 5 | `bulk_process_passive`, `bulk_process_active`, `convert_to_cst` |
| UWB | 3 | `calculate_sff_from_files`, `analyze_uwb_channel`, `get_impedance_bandwidth` |
| Calibration drift | 12 | `cal_drift_ingest`, `cal_drift_compare`, `cal_drift_alert`, `cal_drift_recert_check` |
| Orchestration | 1 | `process_folder` |
| Validation | 1 | `analyze_iperf_angle_sweep` |
| Comparison | 3 | `compare_antennas`, `summarize_antennas`, `compare_active_overlay` |
| VNA | 2 | `analyze_s11`, `analyze_group_delay` |
| Propagation | 1 | `estimate_link_budget` |
| MIMO | 1 | `analyze_mimo_diversity` |
| Calibration | 1 | `generate_active_cal` |
| Compliance | 2 | `check_spec_compliance`, `check_regulatory_eirp` |
| Uncertainty | 1 | `compute_uncertainty_budget` |
| S-parameters | 1 | `analyze_multiport_touchstone` |
| Statistics | 1 | `average_patterns` |
| RF methods | 8 | `synthesize_array`, `near_field_to_far_field`, `calculate_tis`, `ctia_test_plan_template` |
| Instruments | 2 | `vna_read_trace`, `positioner_scan_grid` |

`rflect://status` and `rflect://help` are MCP **resources**, not tools.

`batch_analyze_frequencies` is mentioned in older docs but is **not** a registered tool.

Smoke check (from the repo root, with the `mcp` extra installed):

```bash
.venv/bin/python -c "import sys, os; sys.path.insert(0, os.path.abspath('rflect-mcp')); import server; print('tools:', len(server.mcp._tool_manager._tools))"
```

Expected: `tools: 62`.

## Report narrative

`generate_report` produces deterministic, data-driven prose from the measurement
data with no LLM involved. A driving agent may override any of the following via
the `narrative` parameter (omitted keys fall back to the deterministic text):

- `executive_summary` (str)
- `section_analysis` ({measurement_name: str})
- `recommendations` (str)
- `captions` ({plot_filename: str})

## Deterministic analysis core

All analysis math lives in `plot_antenna/` (`analysis_engine.py`, `calculations.py`,
`rf_methods.py`, `uwb_analysis.py`, `cal_drift.py`, `file_utils.py`) as pure functions. The MCP tools
are thin wrappers that validate input, call these functions, and return structured
dicts. They never make network or LLM calls and never raise: failures surface in a
`warnings` list (or an `error` field on some RF-method tools).

Instrument tools (`vna_read_trace`, `positioner_scan_grid`) use an in-memory mock
unless optional `pyvisa` / `pyserial` backends are installed (`pip install -e ".[instruments]"`).

## History

- **v6.1.0**: 3D-render correctness; Windows packaging (no UPX, PE version resource). Tool count unchanged at 61.
- **v6.0.0**: RF-method, compliance, uncertainty, instrument, and related tools; MCP count grew to 61. `pip install rflect[mcp]` extra.
- **v5.0.0**: Removed the entire AI/LLM stack. Added 6 RF-analysis MCP tools. `generate_report` gained `narrative`. MCP tool count 35 → 41.
- **v4.3.0**: `analyze_iperf_angle_sweep`.
- **v4.2.0**: `process_folder` single-call folder orchestration.
