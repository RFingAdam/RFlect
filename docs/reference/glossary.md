# Glossary

| Term | Definition |
|---|---|
| **AR** | Axial Ratio — ratio of semi-major to semi-minor axis of the polarization ellipse, expressed in dB. 0 dB = perfect circular; ∞ dB = perfect linear. |
| **CST** | Computer Simulation Technology — vendor for EM simulation; their `.ffs` Farfield Source format is supported as an RFlect export target. |
| **Directivity** | Peak gain divided by spherical-average gain. The "shape" of the pattern, independent of efficiency. |
| **ECC** | Envelope Correlation Coefficient — how correlated two antenna patterns are; matters for MIMO diversity. |
| **Efficiency (η)** | Radiated power / accepted power. Includes ohmic and mismatch losses. |
| **F/B** | Front-to-Back ratio. Gain at the peak direction minus gain at 180° from the peak. |
| **Fernet** | Symmetric AES-128 + HMAC-SHA256 encryption scheme used to encrypt API keys at rest. |
| **Gain** | Power radiated in a given direction per unit solid angle, relative to isotropic. `Gain = Efficiency × Directivity`. dBi units. |
| **HPBW** | Half-Power Beamwidth, aka -3 dB beamwidth. Angular width where gain drops to half-peak. |
| **HPOL / VPOL** | H- and V-polarization. Ludwig-3: HPOL → $E_\phi$, VPOL → $E_\theta$. |
| **Kraus** | Approximation for efficiency: $\eta \approx 32400 / (\text{HPBW}_E \cdot \text{HPBW}_H)$. Only valid for moderately directional antennas. |
| **LHCP / RHCP** | Left-/Right-Hand Circular Polarization. IEEE convention: looking in direction of propagation. |
| **MCP** | [Model Context Protocol](https://modelcontextprotocol.io/) — open spec for AI assistants to call tools on external servers. |
| **MEG** | Mean Effective Gain — mean of the antenna gain weighted by the angular power spectrum of the environment. |
| **PBKDF2** | Password-Based Key Derivation Function. RFlect uses PBKDF2-HMAC-SHA256 with 600 K iterations to derive the Fernet key. |
| **SFF** | System Fidelity Factor — normalized cross-correlation between transmitted and received UWB pulses. 1.0 = perfect; 0.95+ usually good. |
| **setup_group** | A free-text tag on a cal-drift run identifying its methodology epoch (e.g. `pre-2024-cable-change`). Mismatch is flagged when comparing across groups. |
| **TRP** | Total Radiated Power. Integral of radiated power over the sphere. IEEE solid-angle formula with $\sin\theta$ Jacobian. |
| **Touchstone** | Industry-standard `.s2p` / `.s1p` file format for S-parameters. RFlect's UWB pipeline reads `.s2p`. |
| **VSWR** | Voltage Standing Wave Ratio. Derived from S11 magnitude. 1.0 = perfect match; 2.0 ≈ -9.5 dB return loss. |
| **WTL** | Wireless Telecom Lab — the chamber-output format RFlect's parsers were built against (V5.02 / V5.03). |
| **XPD** | Cross-Polarization Discrimination — co-pol field divided by cross-pol field, in dB. Uses 20 log, not 10 log. |
