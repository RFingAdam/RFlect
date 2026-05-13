# Polarization

Axial Ratio, Tilt Angle, XPD, and circular-polarization sense from HPOL/VPOL data.

## Polarization ellipse

A radiated electric field can be decomposed into two orthogonal components $E_\theta$ and $E_\phi$ with a phase difference $\delta$. The tip of $\vec{E}$ traces an ellipse in the plane perpendicular to propagation.

The ellipse has:
- **Semi-major axis** $a$
- **Semi-minor axis** $b$  
- **Tilt angle** $\tau$ relative to a reference axis
- **Sense** — RH (clockwise looking from source) or LH

## Axial Ratio (AR)

$$\text{AR (dB)} = 20\log_{10}\!\left(\frac{a}{b}\right)$$

- AR = 0 dB → perfect circular polarization
- AR = ∞ dB → perfect linear polarization (b = 0)
- Typical CP antenna: AR ≤ 3 dB in the main beam

RFlect uses the full polarization-ellipse derivation, including the $\cos(2\delta)$ discriminant. Phase $\delta$ is recovered correctly — RFlect does NOT use the magnitude-only approximation.

## XPD (Cross-Polarization Discrimination)

$$\text{XPD (dB)} = 20\log_{10}\!\left(\frac{|E_{\text{co-pol}}|}{|E_{\text{cross-pol}}|}\right)$$

Note the **20 log** — XPD is a field ratio. Common mistake: using 10·log. RFlect uses the correct formula.

## RHCP / LHCP sense

Determined from the sign of the discriminant $\sin\delta$:

- $\sin\delta > 0$ → LHCP (Left-Hand Circular)
- $\sin\delta < 0$ → RHCP (Right-Hand Circular)

IEEE convention: looking in the direction of propagation.

## How to read it in RFlect

After a passive scan loads, the **Polarization** tab shows:
- AR plot vs frequency (or vs angle)
- Tilt angle map
- XPD plot
- Sense annotation per frequency

## Programmatic / MCP

```python
compare_polarizations(frequency=2450.0)
```

Returns AR, tilt, XPD, and sense stats. See `AntennaAnalyzer.compare_polarizations` in `plot_antenna/ai_analysis.py`.

## See also

- [Concepts → Polarization](../getting-started/concepts.md#polarization)
- [Passive Gain](passive-gain.md)
