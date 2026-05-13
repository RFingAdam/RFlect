# Group Delay

Group delay vs frequency, derived from S21 phase or read directly from an S2VNA capture.

![Group delay analysis](../assets/screenshots/group_delay.png){ .rflect-screenshot }

## What you need

- A 2-port VNA capture: Touchstone `.s2p` or S2VNA `.csv` with `S21(s)` column

## What "group delay" means

$$\tau_g = -\frac{d\phi_{21}}{d\omega}$$

The negative slope of S21 phase with respect to angular frequency. For an antenna pair, $\tau_g$ measures the propagation delay through the antennas plus the air path.

## Metrics RFlect computes

| Metric                | Definition                                                 |
|-----------------------|------------------------------------------------------------|
| `mean_ns`             | Average group delay over the band                          |
| `variation_ps`        | Peak-to-peak ripple in ps                                  |
| `distance_error_cm`   | Effective additional path length from group-delay ripple ($c \cdot \tau_{pp}$) |

## In the GUI

VNA scan type → import a 2-port capture with `S21(s)`. The group-delay plot appears alongside S11/S21.

## Programmatic / MCP

Full UWB analysis (group delay + SFF + transfer function + impulse response):

```python
analyze_uwb_channel("/path/to/cap.s2p", distance_m=1.0)
```

Just group delay is derived inside `analyze_uwb_channel` — there isn't a dedicated MCP tool for group delay alone because it's almost always wanted alongside the other UWB metrics. Use the full analysis and pluck `result["group_delay"]`.

## Common gotchas

- **Phase unwrapping** — `compute_group_delay_from_s21` handles this internally, but the sweep needs enough frequency density (≥ 200 points typical) to unwrap cleanly.
- **Antenna pair vs DUT** — measured group delay includes BOTH antennas plus the air gap. Subtract a reference air-only measurement if you only want the DUT contribution.
