# File Formats

All formats RFlect reads natively, with the parser entry point in `plot_antenna/file_utils.py`.

## WTL chamber `.txt`

The primary format. Plain-text, header + tabular data. RFlect supports both V5.02 and V5.03 layouts.

### Passive (HPOL or VPOL)

Header contains:
- `Frequency` (MHz) — one or more
- `Start Phi`, `Stop Phi`, `Inc Phi` (degrees)
- `Start Theta`, `Stop Theta`, `Inc Theta`
- `Polarization` — `Horizontal` or `Vertical`

Data block: gain (dBi) per ($\theta$, $\phi$) cell. RFlect's parser: `read_passive_file()`.

### Active TRP

Header contains:
- `Frequency` (MHz)
- Axis ranges as above
- `H_Power_dBm` and `V_Power_dBm` data blocks

RFlect's parser: `read_active_file()`. Downstream: `calculate_active_variables()` produces total-power 2D arrays plus `TRP_dBm` / `h_TRP_dBm` / `v_TRP_dBm`.

### TRP Calibration

`TRP Cal <antenna> <band> <Amp> <power> <range> <date>.txt`. Parsed by `plot_antenna/cal_drift.py:parse_trp_cal_file`.

Paired with a `TRP Cal Summary <…>.txt` when present (records the input files, gain standard, methodology).

## Touchstone `.s2p`

Industry-standard 2-port S-parameter format. Supports REAL/IMAG, MAGANGLE, and DB columns. Parsed by `plot_antenna/uwb_analysis.py:parse_touchstone`.

Used for:
- Group delay analysis
- UWB SFF / transfer function / impulse response
- Impedance bandwidth

## S2VNA `.csv`

Copper Mountain S2VNA software exports CSV with a fixed column convention RFlect understands:
- `! Stimulus(Hz)` — frequency
- `S11(dB)`, `S21(dB)` — magnitudes
- `S21(s)` — group delay (seconds)

Parsed by `parse_2port_data()`. Auto-detected from `.csv` extension + columns.

## Generic VNA `.csv`

Two-port S-parameter exports from arbitrary VNAs work as long as the column names match the S2VNA convention. Otherwise, convert to Touchstone `.s2p`.

## CST simulation `.txt`

Far-field exports from CST Studio Suite. Read by `plot_antenna/plot_group_delay_cst.py` for ECC and fidelity factor analysis.

## CST `.ffs` (output only)

RFlect can **export** to CST Farfield Source format via the `convert_to_cst` MCP tool — useful for moving measured patterns into a simulation toolchain.

## Auto-detection in `process_folder`

| Filename pattern                                             | Intent      |
|--------------------------------------------------------------|-------------|
| `TRP Cal *.txt` (not `TRP Cal Summary *.txt`)                | `cal_drift` |
| `*_HPol.txt` paired with `*_VPol.txt`                        | `passive`   |
| `*.txt` containing `TRP` (not `TRP Cal`)                     | `active`    |
| `*.s2p` OR `*.csv` with `S21` / `GroupDelay` / `*deg*` token | `uwb`       |

Priority on tie: `cal_drift > passive > active > uwb`. See [Recipes](../mcp/recipes.md).
