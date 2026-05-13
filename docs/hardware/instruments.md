# Instruments

RFlect doesn't talk to instruments directly — it reads their saved output. Below is the verified hardware/software the test suite has touched, but the format definitions in [File Formats](file-formats.md) are what really matters.

## Chambers

| Vendor          | Model                  | Tested |
|-----------------|------------------------|--------|
| Howland Company | 3100 Antenna Chamber   | yes (primary reference) |
| Any chamber producing WTL V5.02 / V5.03 `.txt` | — | yes (format-level) |

Open an issue with a sample file if your chamber's export differs.

## Vector network analyzers

| Vendor          | Software / format         | Tested |
|-----------------|---------------------------|--------|
| Copper Mountain | S2VNA `.csv`              | yes    |
| Any 2-port VNA  | Touchstone `.s2p`         | yes    |
| Any 2-port VNA  | CSV with S2VNA columns    | yes    |

`! Stimulus(Hz)`, `S11(dB)`, `S21(dB)`, `S21(s)` are the column names RFlect looks for.

## Simulation tools

| Vendor       | Format                    | Read | Write |
|--------------|---------------------------|------|-------|
| CST Studio   | Far-field `.txt`          | yes  | —     |
| CST Studio   | Farfield Source `.ffs`    | —    | yes (via `convert_to_cst` MCP tool) |

## Gain standards used in active calibration

The active-calibration routine has been verified with:

- Howland BLPA (broadband log-periodic)
- Howland HORN

The gain-standard's calibrated bands determine which TRP-cal frequencies route to "Missing Data" — e.g. the BLPA-19 has uncalibrated gaps at 960–1500 MHz, 1610–1710 MHz, and 2170–2300 MHz. See `tests/test_active_calibration.py`.

## What you do **not** need

- A specific OS — RFlect runs on Windows, Linux, and macOS (built from source)
- A specific instrument vendor — formats matter, brands don't
- A specific Python version beyond 3.11+
- Network access — AI features are optional and gracefully disabled when unconfigured
