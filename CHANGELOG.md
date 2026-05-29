# Changelog

All notable changes to RFlect are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and the project adheres
to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

Full, detailed per-release notes live in
[RELEASE_NOTES.md](https://github.com/RFingAdam/RFlect/blob/main/RELEASE_NOTES.md); this file is the concise summary.

## [6.0.0] — 2026-05-28

Consolidated release bundling the **v5.1 (correctness & quick wins)**,
**v5.2 (RF analysis expansion)** and **v6.0 (automation & platform)** milestones
on top of v5.0.0 — 46 issues. MCP tool count grew to 61.

### Added — new RF methods & analysis (v5.2 / v6.0)
- Advanced RF methods in `plot_antenna/rf_methods.py`, exposed as MCP tools:
  axial-ratio / CP sense, 3-antenna absolute-gain method (#48), uniform-linear
  array factor with electronic steering / HPBW / SLL / grating-lobe flag (#46),
  planar near-field→far-field via 2D FFT (#47), S-parameter time-gating +
  port-extension de-embedding (#43), CTIA TIS + OTA test-plan templates (#42).
- Optional trapezoidal TRP quadrature with pole/phi-wrap half-cells (#35).
- Multiport Touchstone `.s3p`/`.s4p` + mixed-mode S-parameters (#31).
- Measurement-uncertainty budgets / error bars on TRP & gain (#30); regulatory
  spec-mask checks (FCC Part 15 / ETSI) (#29) with automated PASS/FAIL
  limit-lines in reports (#28); statistical pattern averaging across repeat
  measurements (#32); n-antenna Min/Max VSWR/Eff/Gain comparison tables (#7, #34).
- Exact Rice/Marcum-Q fade-margin model (#23); main-lobe guard on sidelobe
  detection (#22); `compute_group_delay_dispersion()` group-delay variance/std (#8).

### Added — automation & platform (v6.0)
- SCPI VNA control + chamber-positioner automation via a driver-Protocol +
  in-memory mock backend (CI-testable) + optional pyVISA/pyserial backends (#44, #45).
- Cal-drift: threshold alerts on new runs (#4), gain-standard recertification
  reminders (#3), scheduled monitoring (#33), passive-calibration support (#6),
  cable-loss `.s2p` history tracking (#5).
- Off-the-Tk-thread rendering helper (`async_render`) and report generation
  (#40, #24); determinate batch progress bars + dialog keyboard nav (#26).
- `pip install rflect[mcp]` extra; `[instruments]` extra for live hardware (#39).

### Changed — engineering & refactors (v5.1 / v6.0)
- Decomposed the giant modules into cohesive submodules (zero API change via
  re-export): `advanced_plots`, `extrapolation`, `advanced_analysis_config`,
  `docx_helpers`; `plotting.py` shrank 3825→2986 lines (#36, #37, #38, #41).
- Documented + test-locked the string return contract for the analysis/bulk MCP
  tools; `cal_drift_report` returns a dict and never raises (#9, #10).
- Slimmer install: `openai` → `[ai]` extra; dropped unused `keyring`/`cryptography`;
  `pyinstaller` out of runtime reqs; numpy ceiling raised to `<3.0.0` (#16, #17, #18).
- Cross-platform GUI fonts + themed dialogs (#25); GUI errors routed to
  log/messagebox instead of `print()` / silent `except…pass` (#15, #19).

### Fixed
- Clarified TRP docstring (input is EIRP; the 1/4π factor is correct) and added a
  golden-reference regression test locking the TRP/gain core (#11, #12).

### Docs
- Keep-a-Changelog `CHANGELOG.md`; expanded glossary / measurement-types / MCP
  overview (#20, #21); v6.0 RF-method example-figure gallery + reproducible
  generator (#27); test coverage for previously-untested modules (#13, #14).

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
