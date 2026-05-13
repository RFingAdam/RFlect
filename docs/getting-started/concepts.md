# Concepts

A quick refresher on the RF measurement vocabulary RFlect uses.

## Active vs passive measurements

| Type    | What you control                            | What you measure                                 |
|---------|----------------------------------------------|-------------------------------------------------|
| Active  | The DUT transmits at a known conducted power | Radiated power per angle → TRP                  |
| Passive | The chamber transmits, DUT receives passively | Gain per angle (relative to a calibrated reference) |

Active TRP needs a known conducted-power level so you can compute efficiency:

$$\eta = \frac{\text{TRP}_{\text{radiated}}}{P_{\text{conducted}}}$$

Passive gain is already in dBi relative to isotropic.

## Polarization

Antennas radiate in two orthogonal polarizations (Ludwig-3 convention used here):

| Component  | RFlect calls it | Maps to                            |
|------------|-----------------|------------------------------------|
| $E_\phi$   | HPOL            | Azimuthal / "horizontal"           |
| $E_\theta$ | VPOL            | Elevation / "vertical"             |

Combined into **total gain** = $10 \log_{10}(|E_\theta|^2 + |E_\phi|^2)$ relative to isotropic.

Derived metrics:

- **Axial Ratio (AR)** — major/minor axis ratio of the polarization ellipse
- **Tilt Angle** — orientation of the polarization ellipse
- **XPD** — Cross-Polarization Discrimination, $20 \log_{10}(\text{co-pol field}/\text{cross-pol field})$
- **Sense** — RHCP vs LHCP (right- vs left-hand circular polarization)

## TRP — Total Radiated Power

IEEE-standard solid-angle integration with $\sin\theta$ Jacobian:

$$\text{TRP} = \frac{1}{4\pi} \int_0^{2\pi}\!\int_0^{\pi} P(\theta,\phi)\,\sin\theta\,d\theta\,d\phi$$

RFlect's TRP is verified to within 0.002 dB of the chamber's own report on reference measurements.

## Efficiency vs directivity

- **Efficiency** $\eta$ — radiated power / accepted power. Includes ohmic and mismatch losses.
- **Directivity** $D$ — peak gain divided by average gain over the sphere (the "shape" of the pattern).
- **Gain** = $\eta \cdot D$

## Beamwidth

- **HPBW** (Half-Power Beamwidth, aka -3 dB beamwidth) — angular width where gain drops to half-peak

RFlect computes HPBW with proper boundary wrapping at 0/360°.

## Cal-drift epochs

A "setup_group" tags a calibration run with its methodology epoch (e.g. `pre-2024-cable-change`, `2026-v2-mount`). Two runs in different groups are flagged on the cross-epoch consistency tab as not apples-to-apples — see [Cal Drift](../user-guide/cal-drift.md).

## UWB / SFF

For ultra-wideband antennas, gain isn't enough — you also care about preserving pulse shape across angles. **System Fidelity Factor (SFF)** = normalized cross-correlation between transmitted and received pulse. 1.0 = perfect; 0.95+ is typically good.
