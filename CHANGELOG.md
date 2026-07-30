# Changelog

All notable changes to RFlect are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and the project adheres
to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

Full, detailed per-release notes live in
[RELEASE_NOTES.md](https://github.com/RFingAdam/RFlect/blob/main/RELEASE_NOTES.md); this file is the concise summary.

## [6.1.0] — 2026-07-30

3D radiation-pattern rendering correctness & robustness. The shared scaling
machinery (equal symmetric axis limits + `set_box_aspect([1,1,1])`) was verified
to render true proportions — a sphere renders as a sphere in both the passive
and active routes — and these changes fix an active-save data-selection bug and
harden the degenerate-input paths.

### Fixed
- **Active 3D saves now use per-polarization data.** `save_to_results_folder`
  passed the total-power arrays into the H-pol and V-pol active 3D plot calls,
  so the saved `3D_TRP_hpol` / `3D_TRP_vpol` images rendered the total-power
  pattern under a polarization label. They now receive `h_power_dBm_2d` /
  `v_power_dBm_2d` (matching the already-correct GUI display path).
- **Passive 3D rendering hardened against degenerate input.** A NaN-containing
  or constant-gain (`max == min`) interpolated pattern no longer NaN-poisons /
  divides-by-zero into a blank surface; NaN-robust min/max + a zero-radius
  fallback now mirror the active route.
- **3D axis setup guards degenerate extents.** `_setup_3d_axes` no longer lets a
  flat (extent 0) or all-NaN pattern collapse the bounding box to `lim=0` /
  `lim=NaN` (which blanked the axes); it falls back to a unit box.

### Changed
- `process_data` (3D render helper) now returns only the interpolated grid and
  its axes `(data_interp, theta_interp, phi_interp)`; it previously also built
  Cartesian `X/Y/Z` and a `db_to_linear` radius that every caller discarded.
- Removed dead code in `plot_passive_3d_component`: the unused `gain_normalized`
  local and the vestigial `shadowing_enabled` / `shadow_direction` parameters.
  Human-shadow fading is applied to the gain data upstream; these parameters had
  no effect inside the plot function (the shadow-cone overlay was never wired
  up). Re-add them if/when that overlay is implemented.

### Tests
- `tests/test_3d_scaling_fixes.py`: regression coverage for the active per-pol
  save, passive NaN/constant-gain robustness, `_setup_3d_axes` degenerate
  extents, and the `process_data` return contract.

### Packaging
- **Windows executable hardened against AV/EDR false positives** (Cylance and
  similar were blocking installs). `RFlect.spec`: disabled UPX compression
  (`upx=False` — a well-known AV/EDR heuristic trigger) and added a PE
  VERSIONINFO resource (`version_info.txt`) embedding Company/Product/File
  version metadata, previously absent. Also dropped a stale `anthropic`
  hidden-import left over from the removed AI/LLM stack. Code signing (so IT
  can allow-list RFlect by publisher certificate instead of per-build file
  hash) is scoped as follow-up work, not included here.

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
