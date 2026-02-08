<p align="center">
  <img src="./assets/rflect_logo.svg" alt="RFlect Logo" width="160">
</p>

<h1 align="center">RFlect</h1>

<p align="center">
  <strong>Antenna measurement visualization and analysis for RF engineers.</strong>
</p>

<p align="center">
  <a href="https://github.com/RFingAdam/RFlect/releases"><img src="https://img.shields.io/badge/version-4.0.0-blue" alt="Version"></a>
  <img src="https://img.shields.io/badge/python-3.11+-green" alt="Python">
  <a href="LICENSE"><img src="https://img.shields.io/badge/license-GPL--3.0-orange" alt="License"></a>
  <img src="https://img.shields.io/badge/tests-227%20passing-brightgreen" alt="Tests">
</p>

---

RFlect processes antenna measurement data from the **Howland 3100 Antenna Chamber** (WTL format), **Copper Mountain VNA** exports, and **CST** far-field simulation files. It computes TRP, passive gain, polarization parameters, and efficiency metrics with IEEE-standard methods, then generates publication-ready 2D/3D radiation pattern plots.

## New in v4.0

- **11 RF engineering fixes** — Corrected diversity gain (Vaughan-Andersen), axial ratio, XPD, TRP integration, HPBW boundary wrapping, and more. All formulas verified against IEEE references and chamber data.
- **Modern dark GUI** — Complete visual overhaul with dark ttk theme, color-coded logs, keyboard shortcuts (`Ctrl+R`/`F5`), and WCAG AA contrast compliance across all dialogs.
- **Multi-provider AI** — Unified LLM abstraction supporting OpenAI, Anthropic, and Ollama with timeout/retry hardening. AI chat assistant, report generation, and vision-based analysis.
- **Secure API key storage** — Fernet AES-128 encryption with PBKDF2 (600K iterations), machine-ID binding, and OS keyring integration.
- **MCP server (20 tools)** — Programmatic antenna analysis for Claude Code and other AI assistants. End-to-end pipeline verified against chamber reference data.
- **227 tests** — Up from ~50 in v3.x. Integration tests with real BLE and LoRa chamber data.

See [RELEASE_NOTES.md](RELEASE_NOTES.md) for the full changelog.

## Quick Start

**Windows:** Download `RFlect_vX.X.X.exe` from the [latest release](https://github.com/RFingAdam/RFlect/releases).

**From source:**
```bash
git clone https://github.com/RFingAdam/RFlect.git
cd RFlect
python -m venv .venv && source .venv/bin/activate  # or .venv\Scripts\activate on Windows
pip install -r requirements.txt
python run_rflect.py
```

## Supported Data

| Scan Type | Input Format | Analysis |
|-----------|-------------|----------|
| **Active TRP** | WTL `.txt` (V5.02/V5.03) | TRP, H/V power, 2D/3D radiation patterns |
| **Passive Gain** | WTL HPOL + VPOL `.txt` pairs | Total/H/V gain, efficiency, directivity |
| **S-Parameters** | Copper Mountain `.csv` | S11, VSWR, return loss with limit lines |
| **Group Delay** | 2-port VNA `.csv` | Group delay vs frequency, peak-to-peak, distance error |
| **CST Far-Field** | `.txt` simulation files | ECC, fidelity factor, group delay |

## Usage

1. **Select scan type** (Active, Passive, or VNA)
2. **Adjust settings** if needed (cable loss, limit lines, frequency range)
3. **Import file(s)** via the Import button or `Ctrl+O`
4. **View results** — plots render automatically; use `Ctrl+R` to reprocess

<details>
<summary><strong>Screenshots (click to expand)</strong></summary>

### Passive Measurements

**G&D Comparison** — Efficiency, gain, and directivity across multiple scans:
![G&D Results](./assets/python_1d_results_g&d.png)

**HPOL/VPOL 1D** — Efficiency and total gain vs frequency:
![Passive 1D](./assets/python_1d_results.png)

**2D Azimuth Cuts** — Gain pattern for various theta angles:
![Passive 2D](./assets/python_passive_2d_results_azimuth.png)

**Datasheet Plots** — Peak gain per polarization, polar cuts at key planes:
![Datasheet 1D](./assets/python_1d_results_datasheet.png)
![Datasheet 2D](./assets/python_2d_results_datasheet.png)

**3D Radiation Patterns** — Phi, theta, and total gain:
![Passive 3D](./assets/python_passive_3d_results.png)

### Active TRP Measurements

**2D Power Cuts:**
![Active 2D](./assets/python_active_2d_results_azimuth.png)
![Active Datasheet](./assets/python_active_2d_results_datasheet.png)

### VNA / S-Parameters

**S-Parameter Overlay:**
![VNA Results](./assets/python_vna_results.png)

**Group Delay Analysis:**
![Group Delay](./assets/python_groupdelay_results.png)

</details>

## Key Features

- **Polarization Analysis** — Axial ratio, tilt angle, XPD, and polarization sense (LHCP/RHCP) from HPOL/VPOL data with interactive and batch export modes
- **Batch Processing** — Automatically find and process all HPOL/VPOL pairs or TRP files in a directory with organized per-pair output folders
- **Report Generation** — Export DOCX reports with embedded plots, measurement summaries, and optional AI-generated analysis
- **3D Visualization** — Turbo colormap, transparent panes, coordinate axes on top, manual or auto Z-axis scaling

## AI Features (Optional)

RFlect integrates with **OpenAI**, **Anthropic**, and **Ollama** for intelligent measurement analysis. Features include a chat assistant with function-calling tools, AI-powered report generation, and vision-based plot analysis. All AI features are optional — core functionality works without any provider configured.

API keys are stored securely using OS keyring or Fernet-encrypted files bound to your machine ID. Configure via **Tools > Manage API Keys**.

See [AI_STATUS.md](AI_STATUS.md) for provider details and supported models.

## MCP Server

RFlect includes an [MCP](https://modelcontextprotocol.io/) server with 20 tools for programmatic antenna analysis via AI assistants like Claude Code. Import measurements, run analysis, and generate reports without the GUI.

See [rflect-mcp/README.md](rflect-mcp/README.md) for setup and tool reference.

## Project Structure

```
RFlect/
  plot_antenna/           # Core application
    gui/                  #   GUI mixins (callbacks, tools, dialogs, AI chat)
    ai_analysis.py        #   RF analysis engine (gain stats, pattern, polarization)
    calculations.py       #   TRP, passive gain, efficiency computations
    file_utils.py         #   WTL/VNA file parsers
    plotting.py           #   2D/3D matplotlib rendering
    llm_provider.py       #   Multi-provider LLM abstraction
    api_keys.py           #   Secure key storage (keyring + Fernet)
    save.py               #   DOCX report generation
  rflect-mcp/             # MCP server for programmatic access
  tests/                  # 227 tests (pytest)
```

## Development

```bash
pip install -r requirements-dev.txt
python -m pytest tests/                    # run tests
pyinstaller RFlect.spec                    # build exe
```

See [CONTRIBUTING.md](CONTRIBUTING.md) for coding standards, architecture details, and test guidelines.

## License

[GPL-3.0](LICENSE)
