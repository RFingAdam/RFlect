# Changelog

All notable changes to RFlect are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and the project adheres
to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

Full, detailed per-release notes live in
[RELEASE_NOTES.md](https://github.com/RFingAdam/RFlect/blob/main/RELEASE_NOTES.md); this file is the concise summary.

## [Unreleased] — v5.1 (Correctness & quick wins)

### Added
- Golden-reference regression tests locking the TRP/gain integration core
  (isotropic EIRP → itself; upper hemisphere → −3.01 dB).
- `compute_group_delay_dispersion()` — per-frequency group-delay variance/std
  across theta cuts (replaces the old dead TODO block).
- Test coverage for previously-untested modules: `plotting.py`, `save.py`,
  `groupdelay.py`, `plot_group_delay_cst.py`, `uwb_plotting.py`.

### Changed
- `cal_drift_report` MCP tool now returns `{output_path, format}` / `{error}`
  and never raises to the client.
- Slimmer install: `openai` moved to the `[ai]` extra; `keyring`/`cryptography`
  dropped from core deps (unused since the v5.0.0 AI removal); `pyinstaller`
  removed from the runtime requirements; numpy ceiling raised to `<3.0.0`.
- GUI settings save/load and data-processing failures are now surfaced
  (stderr / log) instead of being silently swallowed.

### Docs
- This changelog; expanded reference docs.

## [5.0.0] — 2026-05-28

**Zero-dependency, MCP-first relaunch.** Removed the entire in-app AI/LLM stack
(chat, AI report generation, provider abstraction, API-key store) — RFlect makes
no LLM/API calls and needs no key or subscription. `generate_report` is
deterministic by default with an optional agent-authored `narrative`. Added 6
MCP tools (compare_antennas, analyze_s11, analyze_group_delay,
estimate_link_budget, analyze_mimo_diversity, generate_active_cal); MCP tool
count 35 → 41. Renamed `ai_analysis.py` → `analysis_engine.py`.

## [4.3.0] — 2026-05-28

Multi-angle iperf throughput-validation MCP tool (`analyze_iperf_angle_sweep`).

## [4.2.0] — 2026-05-12

`process_folder` single-call folder-orchestration MCP tool; MkDocs Material docs
site.

## [4.1.9] — 2026-04-14

Patch: restored the Active Chamber Calibration routine.

---

Earlier releases (4.1.8 and prior) are detailed in
[RELEASE_NOTES.md](https://github.com/RFingAdam/RFlect/blob/main/RELEASE_NOTES.md).
