# Active TRP

Total Radiated Power workflow for an active (transmitting) DUT.

![Active 2D azimuth cuts](../assets/screenshots/active_2d.png){ .rflect-screenshot }

## What you need

- A WTL `.txt` file with H/V power per ($\theta$, $\phi$) cell
- The DUT's **conducted power** in dBm (to compute efficiency)
- Optional: cable loss in dB

## In the GUI

1. Scan type: **Active**
2. Set `Conducted Power (dBm)`: defaults to 4 dBm; this is the level at the DUT antenna input, not at the source
3. Set `Cable Loss (dB)` if applicable: added back to measured power
4. `Ctrl+O` and pick the active-TRP file
5. Plots render: 2D azimuth/elevation cuts, 3D pattern with turbo colormap, datasheet-style summary

## What you get

| Metric                    | Where it appears                                       |
|---------------------------|--------------------------------------------------------|
| TRP_dBm                   | Status bar + DOCX summary table                        |
| H-TRP / V-TRP             | Polarization-split TRP                                 |
| Total efficiency          | TRP / conducted power (linear), expressed in %         |
| Max / min / avg gain      | Sin-weighted spherical statistics                      |
| 3D radiation pattern      | turbo colormap, DUT orientation triad (X=green, Y=red, Z=blue) |
| Maritime stats (if enabled) | Horizon-band gain, maritime power fraction          |

## Math

RFlect's TRP uses the CTIA / IEEE-149 discrete solid-angle form with the $\sin\theta$ Jacobian, on **EIRP-per-angle** chamber samples:

$$\text{TRP} = \frac{1}{4\pi} \int_0^{2\pi}\!\int_0^{\pi} P(\theta,\phi)\,\sin\theta\,d\theta\,d\phi$$

Golden-reference tests lock the isotropic oracle (constant EIRP integrates back to itself). That is a calculation check, not laboratory accreditation or a certified chamber report.

## Batch / MCP

Process every TRP file in a folder:

```python
process_folder("/path/to/trp_runs", intent="active", report=True)
```

This runs `batch_process_active_scans` over the folder and (optionally) generates a DOCX report. See [Recipes](../mcp/recipes.md#standard-active-trp-procedure).

## Common gotchas

- **Conducted power matters**. Efficiency is wrong if you enter the source-side power instead of the DUT-side power.
- **Cable loss is positive**. Loss is what the signal loses between DUT and receiver: add it back to compensate.
- **TRP at a single frequency only**. If you need TRP vs frequency, use multiple active scans.

## See also

- [Concepts → TRP](../getting-started/concepts.md#trp-total-radiated-power)
- [Maritime / Horizon plots](maritime-horizon.md)
- [Cal Drift](cal-drift.md): track active-cal stability over time
