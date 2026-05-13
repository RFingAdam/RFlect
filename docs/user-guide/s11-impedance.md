# S11 & Impedance Bandwidth

Return-loss and bandwidth analysis from VNA `.csv` or Touchstone `.s2p` files.

![VNA / S-parameter results](../assets/screenshots/vna.png){ .rflect-screenshot }

## What you need

- A 2-port (or 1-port) VNA capture with S11 in dB
- Optional: a `-10 dB` (or custom) threshold to define "in-band"

## In the GUI

1. Scan type: **VNA**
2. `Ctrl+O` and pick the `.csv` or `.s2p` file
3. The S11 plot appears with optional limit lines
4. **Set Limit Lines** dialog lets you mark pass/fail thresholds per band

## What you get

| Metric                      | Definition                                              |
|-----------------------------|---------------------------------------------------------|
| `min_s11_dB`                | Best (most negative) return loss in the sweep           |
| `bandwidth_mhz`             | Frequency span where S11 ≤ threshold (default -10 dB)   |
| `band_start_ghz` / `band_stop_ghz` | Edges of the matched band                        |
| `fractional_bandwidth_pct`  | $200 \cdot (f_2 - f_1) / (f_2 + f_1)$                  |
| VSWR                        | Voltage Standing Wave Ratio derived from S11           |

## Programmatic / MCP

```python
get_impedance_bandwidth("/path/to/sweep.s2p", threshold_dB=-10.0)
```

Returns the same metrics as JSON. See [tools reference](../mcp/tools-reference.md).

## Common gotchas

- **Calibration matters**. RFlect doesn't apply VNA calibration — bring already-calibrated data.
- **Threshold convention**. -10 dB ≈ 90% power match. Some specs use -6 dB or -3 dB; set explicitly.
- **Single-band assumption**. The bandwidth metric reports the widest contiguous matched band; multi-band antennas will need per-band analysis.

## See also

- [Group Delay](group-delay.md) — derived from S21 phase
- [UWB / SFF](uwb-sff.md) — full UWB channel characterization
