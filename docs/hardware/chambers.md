# Chambers

RFlect is **chamber-vendor-agnostic** — it works with the file output, not the hardware. If your chamber exports a supported file format ([see File Formats](file-formats.md)), RFlect can read it.

## Primary supported workflow

The reference workflow we test against is the **WTL (Wireless Telecom Lab)** chamber output produced by the Howland Company 3100 Antenna Chamber. WTL exports plain-text `.txt` files in V5.02 and V5.03 layouts which RFlect parses natively.

## What you actually need

A chamber that produces, per measurement:

- A scan over $\theta$ (elevation) and $\phi$ (azimuth) at fixed frequencies
- Power or gain values per angle
- Polarization split (HPOL vs VPOL) for passive scans
- A summary header with frequency, axis ranges, increments, and (for active) the conducted power level

Most modern chambers can produce this format — either natively or via a post-processor.

## Filename conventions RFlect expects

| Measurement       | Filename pattern (example)                                  |
|-------------------|-------------------------------------------------------------|
| Passive HPOL      | `PassiveTest_BLE AP_HPol.txt` — must contain `_HPol`        |
| Passive VPOL      | `PassiveTest_BLE AP_VPol.txt` — must contain `_VPol`        |
| Active TRP        | `Active Test_BLE TRP.txt` — must contain `TRP` (any case)    |
| TRP Calibration   | `TRP Cal <antenna> <band> <date>.txt`                       |
| TRP Cal Summary   | `TRP Cal Summary <antenna> <band> <date>.txt`               |

Auto-detection in [`process_folder`](../mcp/recipes.md) relies on these patterns.

## Reference chamber: Howland 3100

The 3100 is the system RFlect is verified against. RFlect's TRP computation agrees with the chamber's own report to within 0.002 dB on reference measurements (see [TRP concept](../getting-started/concepts.md#trp-total-radiated-power)).

If you have a different chamber and it exports the WTL format directly (or you can convert), RFlect will work. Open a GitHub issue if you hit a parser edge case for a non-WTL export.

## Active vs passive in the chamber

- **Active** — DUT transmits at a known `conducted_power_dBm`. The chamber receiver records `H_Power_dBm` and `V_Power_dBm` per angle.
- **Passive** — Chamber transmitter illuminates the DUT. Receiver measures co-pol and cross-pol gain. Two separate scans (HPOL and VPOL) — RFlect pairs them by filename.

See [Active TRP](../user-guide/active-trp.md) and [Passive Gain](../user-guide/passive-gain.md) for the full workflow.
