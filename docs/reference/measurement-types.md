# Measurement Types & Analyses

Every measurement type and analysis RFlect supports, the input it expects, what it computes, and the MCP tool that drives it. RFlect is fully deterministic — every value here is computed from your data, with no LLM or API key involved.

## Scan types (primary measurement modes)

| Scan type | Input | What RFlect computes | Primary MCP tools |
|---|---|---|---|
| **Active TRP** | WTL chamber `.txt` (transmitting DUT) | Total Radiated Power, H/V power split, EIRP, 2D/3D power patterns | `import_antenna_file`, `analyze_pattern`, `get_gain_statistics`, `bulk_process_active` |
| **Passive gain** | WTL chamber HPOL + VPOL `.txt` pair | Total/H/V gain, efficiency, directivity, HPBW, sidelobes, polarization | `import_passive_pair`, `bulk_process_passive`, `get_gain_statistics`, `compare_polarizations` |
| **VNA / S-parameters** | Copper Mountain / generic VNA `.csv`, Touchstone `.s2p` | S11, VSWR, return loss, impedance bandwidth, resonances | `analyze_s11` |

## Analyses (the Tools menu + MCP surface)

| Analysis | Input | What it produces | MCP tool |
|---|---|---|---|
| **Bulk passive processing** | Folder of HPOL/VPOL pairs | Batch gain/efficiency/pattern outputs | `bulk_process_passive` |
| **Bulk active processing** | Folder of active TRP files | Batch TRP/pattern outputs | `bulk_process_active` |
| **Polarization analysis** | Passive HPOL/VPOL | Axial ratio, tilt, sense, XPD, H/V balance | `compare_polarizations` |
| **Antenna comparison** | 2+ loaded measurements | Per-frequency peak gain/TRP overlay, deltas vs reference, best-per-frequency | `compare_antennas` |
| **Group delay** | Transmission phase sweep (S21/S12) | Group delay vs frequency, peak-to-peak, in-band flatness, distance error | `analyze_group_delay` |
| **System Fidelity Factor (UWB)** | S2VNA `.csv` / `.s2p` pair | SFF, transfer function, impulse response | `calculate_sff_from_files`, `analyze_uwb_channel` |
| **Impedance bandwidth** | S11 sweep | Return loss, VSWR, fractional bandwidth | `analyze_s11`, `get_impedance_bandwidth` |
| **Horizon / maritime statistics** | Active/passive pattern | Horizon-plane gain stats, coverage | `get_horizon_statistics` |
| **Frequency extrapolation** | Loaded measurement | Estimated metrics at an unmeasured frequency | `extrapolate_to_frequency` |
| **MIMO / diversity** | Envelope correlation (ECC) | Vaughan-Andersen diversity gain, 2×2 capacity, isolation rating | `analyze_mimo_diversity` |
| **Link budget / range** | Measured gains + path-loss model | Max range, link margin, fade margin (Rayleigh/Rician) | `estimate_link_budget` |
| **Calibration drift** | TRP-Cal run archive | Drift vs prior epochs, setup-group mismatch flags | `cal_drift_ingest`, `cal_drift_compare`, `cal_drift_report` |
| **Active chamber calibration** | Power + gain-standard + HPOL/VPOL reference files | TRP cal file + summary; auto-recorded to drift history | `generate_active_cal` |
| **CST FFS conversion** | HPOL/VPOL data | CST-format far-field source | `convert_to_cst` |
| **Combined / branded report** | Any loaded measurements | DOCX with plots, gain tables, deterministic or agent-authored prose | `generate_report`, `preview_report` |
| **Folder orchestration** | A folder of any of the above | One-call auto-detect → import → analyze → optional report | `process_folder` |
| **Multi-angle iperf validation** | Two bench iperf sessions (installed + reference) at matched azimuths | Per-angle throughput delta, worst/best angle, polar plots | `analyze_iperf_angle_sweep` |

## Where the math lives

All analyses are pure, deterministic functions in `plot_antenna/` (`calculations.py`, `analysis_engine.py`, `uwb_analysis.py`, `cal_drift.py`, `file_utils.py`). The MCP tools in `rflect-mcp/tools/` are thin wrappers that validate inputs, call these functions, and return structured results — they never make network or LLM calls.

See the [Tools Reference](../mcp/tools-reference.md) for signatures and return shapes, and the [User Guide](../user-guide/active-trp.md) for the methodology behind each measurement type.
