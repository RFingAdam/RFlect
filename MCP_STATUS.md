# RFlect MCP — Status & Tool Inventory

**Last Updated**: May 28, 2026
**Current Version**: v5.0.0
**Status**: Stable

---

## Zero-dependency by design

As of **v5.0.0**, RFlect makes **no outbound LLM/API calls and requires no API key or paid subscription.** It is a deterministic RF analysis + rendering toolkit. The in-app AI chat assistant, AI report generation, LLM provider abstraction, and encrypted API-key store were all removed.

When RFlect is driven over MCP, the AI agent (Claude Code, Cline, Continue, …) *is* the LLM. RFlect supplies computed data and rendering; the agent supplies any natural-language narrative. Report prose is data-driven by default, or authored by the agent and passed to `generate_report` via the `narrative` parameter.

## MCP server (41 tools, 9 categories)

`rflect-mcp/server.py` exposes the following tool groups. See `docs/mcp/tools-reference.md` for signatures and return shapes.

| Category | Count | Examples |
|----------|------:|----------|
| Import | 6 | `import_antenna_file`, `import_passive_pair`, `import_antenna_folder` |
| Analysis | 5 | `analyze_pattern`, `get_gain_statistics`, `compare_polarizations`, `get_horizon_statistics` |
| Reports | 3 | `generate_report`, `preview_report`, `get_report_options` |
| Bulk | 5 | `bulk_process_passive`, `bulk_process_active`, `convert_to_cst` |
| UWB | 3 | `calculate_sff_from_files`, `analyze_uwb_channel`, `get_impedance_bandwidth` |
| Calibration Drift | 8 | `cal_drift_ingest`, `cal_drift_compare`, `cal_drift_report` |
| Orchestration | 1 | `process_folder` *(v4.2.0)* |
| Validation | 1 | `analyze_iperf_angle_sweep` *(v4.3.0)* |
| RF Analysis | 6 | `compare_antennas`, `analyze_s11`, `analyze_group_delay`, `estimate_link_budget`, `analyze_mimo_diversity`, `generate_active_cal` *(v5.0.0)* |
| Misc | 3 | `get_measurement_details`, `validate_file_pair`, help resource |

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
`uwb_analysis.py`, `cal_drift.py`, `file_utils.py`) as pure functions. The MCP tools
are thin wrappers that validate input, call these functions, and return structured
dicts. They never make network or LLM calls and never raise — failures surface in a
`warnings` list.

## History

- **v5.0.0** — Removed the entire AI/LLM stack (chat, AI report generation, provider abstraction, API-key store). Added 6 RF-analysis MCP tools (comparison, S11, group delay, link budget, MIMO diversity, active-cal). `generate_report` gained the agent-authored `narrative` parameter. MCP tool count 35 → 41.
- **v4.3.0** — `analyze_iperf_angle_sweep` (multi-angle throughput validation).
- **v4.2.0** — `process_folder` single-call folder orchestration.
