# RFlect

**The RF engineer's toolkit for antenna measurement visualization and analysis.**

RFlect takes raw antenna-chamber and VNA output and turns it into publication-ready 2D/3D radiation pattern plots, TRP calculations, polarization analysis, UWB characterization, and DOCX reports — all validated against IEEE-standard methods.

Whether you're characterizing a BLE chip antenna, qualifying a cellular array, or tracking calibration drift across multiple chamber sessions, RFlect handles the heavy lifting.

## At a glance

| What you have                                        | What RFlect produces                                                  |
|------------------------------------------------------|-----------------------------------------------------------------------|
| WTL chamber `.txt` (active TRP)                      | TRP, H/V power split, 2D/3D radiation patterns                        |
| WTL chamber HPOL + VPOL `.txt` pair (passive)        | Total/H/V gain, efficiency, directivity, polarization metrics         |
| Copper Mountain / generic VNA `.csv`                 | S11, VSWR, return loss with limit lines, impedance bandwidth          |
| 2-port VNA `.csv` or Touchstone `.s2p` (group delay) | Group delay vs frequency, peak-to-peak, distance error                |
| S2VNA `.csv` or Touchstone `.s2p` (UWB)              | SFF, transfer function, impulse response, impedance bandwidth         |
| CST simulation export                                | ECC, fidelity factor, group delay                                     |
| Folder of any of the above                           | One-call orchestration via the [`process_folder`](mcp/recipes.md) MCP tool |

## Built for the way RF labs actually work

- **GUI** — desktop app (Tk-based, dark theme) for interactive measurement review
- **MCP server** — 34 tools that let Claude Code, Cline, and other AI clients drive RFlect programmatically
- **AI-assisted reports** — DOCX reports with embedded plots, gain tables, and optional LLM-generated executive summaries (OpenAI / Anthropic / Ollama)
- **Cal-drift tracker** — record TRP-Cal runs over time, compare across epochs, flag setup-group mismatches

## Where to next

- New to RFlect? → [Install](getting-started/install.md) → [Quickstart](getting-started/quickstart.md)
- Already have measurements to process? → [User Guide](user-guide/active-trp.md)
- Want Claude / an agent to drive RFlect? → [MCP overview](mcp/overview.md) → [Recipes](mcp/recipes.md)
- Need to configure AI features? → [AI overview](ai/overview.md)
- Looking up a term or tool? → [Reference](reference/glossary.md)

## License

[GPL-3.0](https://github.com/RFingAdam/RFlect/blob/main/LICENSE)
