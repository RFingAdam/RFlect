# UWB / SFF Analysis

Ultra-Wideband antenna characterization. Gain alone doesn't capture pulse fidelity over wide bandwidths — you also need to know how well the antenna preserves pulse shape across angles.

## System Fidelity Factor (SFF)

Normalized cross-correlation between the transmitted pulse and the received pulse after the antenna chain:

$$\text{SFF} = \max_\tau \frac{\left|\int s_t(t)\, s_r(t-\tau)\, dt\right|}{\sqrt{\int |s_t|^2 dt \cdot \int |s_r|^2 dt}}$$

Range: 0 to 1. 0.95+ is typically considered "good" for UWB.

## What you need

- One or more 2-port captures (Touchstone `.s2p` or S2VNA `.csv` with `S21(s)`)
- For per-angle SFF: filenames containing the angle, e.g. `Antenna_45deg.csv`

## Pulse types supported

| `pulse_type`                  | Use case                                       |
|-------------------------------|------------------------------------------------|
| `gaussian_monocycle` (default)| Generic UWB pulse, baseline                    |
| `modulated_gaussian`          | Carrier-modulated UWB                          |
| `5th_derivative_gaussian`     | FCC UWB spectral mask compliance               |

## Programmatic / MCP

### Full channel analysis (one file)

```python
analyze_uwb_channel("/path/to/cap.s2p", distance_m=1.0)
```

Returns SFF, group delay, transfer function, impulse response, return loss.

### SFF vs angle (many files)

```python
calculate_sff_from_files(
    file_paths=["ant_0deg.csv", "ant_45deg.csv", "ant_90deg.csv", ...],
    pulse_type="gaussian_monocycle",
    min_freq_ghz=3.1,
    max_freq_ghz=10.6,
)
```

Returns mean SFF plus per-angle SFF with quality labels.

### Batch / orchestrator

```python
process_folder("/path/to/uwb_sweep", intent="uwb")
```

Runs `analyze_uwb_channel` on every `.s2p` / matching `.csv` in the folder. See [Recipes](../mcp/recipes.md#uwb-characterization).

## Frequency-domain transfer function

`extract_transfer_function` returns `H(f)` magnitude and complex form, plus a `flatness_dB` metric (peak-to-peak variation in the band).

## Impulse response

IFFT of the transfer function gives the time-domain impulse response. `compute_impulse_response` reports:

- `pulse_width_ps` — full-width-half-max of the main lobe
- `ringing_dB` — secondary peaks relative to the main pulse

## Common gotchas

- **Frequency band filter** — UWB sweeps are wide. Use `min_freq_ghz` / `max_freq_ghz` to restrict the analysis to your standard's band (3.1–10.6 GHz for FCC UWB).
- **Group delay column** — without `S21(s)`, RFlect falls back to magnitude-only SFF, which is less informative. Capture group delay if possible.
- **Angle parsing** — `calculate_sff_from_files` extracts angle from filename via regex `(\d+)(deg|DEG)`. If filenames don't follow that pattern, angles default to 0°, 45°, 90°, … by file order.
