# Maritime / Horizon Plots

Specialized plots for on-water antenna analysis — where the link almost exclusively lives in a narrow band around the horizon.

## What "horizon band" means

A horizon band is defined by two $\theta$ angles around 90° (equator of the sphere): e.g., $\theta = 85°$ to $\theta = 95°$ for a 10°-wide horizon. Everything outside that band wastes power for maritime links.

## Five plot types

| Plot                       | Purpose                                                |
|----------------------------|--------------------------------------------------------|
| Mercator heatmap           | Flat $\phi$-vs-$\theta$ map of gain                    |
| Conical cuts               | Gain vs $\phi$ at each $\theta$ in the horizon band    |
| Gain-over-azimuth          | Horizon-band average gain vs $\phi$ (a single curve)   |
| Horizon statistics table   | Numerical summary: avg, max, min, advantage            |
| 3D pattern with band highlight | 3D sphere with the horizon band visually emphasized |

## Maritime metrics

| Metric                       | Definition                                             |
|------------------------------|--------------------------------------------------------|
| `band_avg_dB`                | Sin-weighted average gain inside the horizon band      |
| `full_avg_dB`                | Sin-weighted average gain over the full sphere         |
| `band_advantage_dB`          | `band_avg_dB - full_avg_dB`. Positive = horizon-favored antenna |
| Maritime Power Fraction (%)  | Fraction of total radiated power inside the band. 50% = isotropic; > 50% = horizon-favored |
| Horizon Efficiency           | Same as `band_advantage_dB`, displayed alongside conducted-power efficiency when available |

## In the GUI

After loading a passive or active scan, the **Maritime** tab appears (if maritime plots are enabled in settings). Configure $\theta_{\text{start}}$ and $\theta_{\text{stop}}$ in degrees.

## Programmatic / MCP

`bulk_process_passive` and `bulk_process_active` both return maritime metrics in their result rows:

```python
process_folder("/path/to/lab/captures", intent="passive", report=True)
# Each result row includes band_advantage_dB and maritime power fraction.
```

## Why "Maritime Power Fraction" was added (v4.1.8)

The earlier "Horizon Efficiency" metric was confusingly close to 50% even for an isotropic antenna (because half the sphere's solid-angle is in any 180°-symmetric band). The dB-form `band_advantage_dB` is the right metric — 0 dB means isotropic, positive means horizon-favored. Both are shown for traceability.
