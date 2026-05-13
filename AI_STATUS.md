# RFlect AI Features — Status & Roadmap

**Last Updated**: May 12, 2026
**Current Version**: v4.2.0
**Status**: Beta / Enabled in GUI and MCP

---

## Overview

RFlect ships AI-powered features for intelligent antenna analysis. A unified provider abstraction (`plot_antenna/llm_provider.py`) supports **OpenAI**, **Anthropic (Claude)**, and **Ollama (local models)**, so users can choose between cloud and on-prem inference without changing the rest of the toolchain.

There are two integration surfaces:

1. **In-GUI** — AI Chat Assistant + AI-powered report generation, accessible from the desktop app.
2. **MCP server** — `rflect-mcp/server.py` exposes 34 tools an AI agent (Claude Code, Cline, etc.) can call programmatically.

## What Works

### 1. AI Chat Assistant
**Status**: ~85% Complete

- Real-time conversational analysis of loaded measurements
- Function calling: `get_gain_statistics`, `analyze_pattern`, `compare_polarizations`, `generate_2d_plot`, `generate_3d_plot`
- Rich context awareness (loaded files, frequencies, data shape, key metrics)
- Quick-action buttons in the GUI (Gain Stats, Pattern, Polarization, All Freqs)
- Multi-turn history, multi-provider support

### 2. AI-Powered Report Generation
**Status**: ~90% Complete

- Executive summary generation from loaded measurements
- Automatic gain statistics insertion
- Pattern type classification
- Vision-capable plot analysis (provider-dependent)
- Uses the same unified provider abstraction as the chat assistant

**Pending**:
- Custom branding integration (partially implemented)
- Multi-frequency comparison tables
- Automated figure captioning
- Compliance checklist generation (FCC, CE)

### 3. Multi-Provider Support (since v4.0.0)
**Status**: Complete

| Provider  | Tool Calling                              | Vision                | Notes                       |
|-----------|-------------------------------------------|-----------------------|-----------------------------|
| OpenAI    | GPT-4 (Chat Completions) + GPT-5 (Responses API) | GPT-4o+        | Default                     |
| Anthropic | Claude Messages API                       | All Claude models     | Via Tools → Manage API Keys |
| Ollama    | llama3.1+, qwen2.5+                       | llava, llama3.2-vision | Local, no API key needed   |

### 4. Secure API Key Management
**Status**: Complete

- Fernet AES-128 encryption with PBKDF2 key derivation (600K iterations)
- Machine-ID based encryption key (`/etc/machine-id`, `IOPlatformUUID`, `MachineGuid`)
- OS keyring integration (Credential Manager, Keychain)
- Restrictive file permissions (chmod 600 / Windows ACL)
- Keys stored in `_key_cache` dict instead of `os.environ`

### 5. MCP Server (34 tools)
**Status**: Complete

| Category       | Count | Examples                                                                          |
|----------------|-------|-----------------------------------------------------------------------------------|
| Import         | 6     | `import_antenna_file`, `import_antenna_folder`, `import_passive_pair`             |
| Analysis       | 5     | `analyze_pattern`, `get_gain_statistics`, `compare_polarizations`                 |
| Reports        | 3     | `generate_report`, `preview_report`, `get_report_options`                         |
| Bulk           | 5     | `bulk_process_passive`, `bulk_process_active`, `list_measurement_files`           |
| UWB            | 3     | `calculate_sff_from_files`, `analyze_uwb_channel`, `get_impedance_bandwidth`      |
| Calibration Drift | 8 | `cal_drift_ingest`, `cal_drift_compare`, `cal_drift_report`, `cal_drift_set_setup_group` |
| Orchestration  | 1     | `process_folder` *(new in v4.2.0)*                                                |
| Misc           | 3     | `get_measurement_details`, `validate_file_pair`, `convert_to_cst`                 |

#### `process_folder` (v4.2.0)

A single entry point that scans a folder, picks the right workflow (passive HPOL/VPOL pair, active TRP, cal-drift archive, or UWB sweep), runs it, and optionally produces a DOCX report. Replaces the previous `list → bulk → analyze → report` chain that MCP clients had to script manually.

```
process_folder(folder_path, intent='auto'|'passive'|'active'|'cal_drift'|'uwb',
               report=False, freqs=None, report_path=None)
```

Auto-detect priority: `cal_drift > passive > active > uwb`. Mixed folders proceed with the winner and surface a `mixed_intents_detected` warning. The tool never raises — every failure mode (missing folder, no match, partial pair, per-file UWB error, report write failure) returns as a structured warning.

---

## Known Limitations

1. **Pattern Analysis** — HPBW and F/B ratio implemented and verified (boundary wrapping + IEEE-validated TRP). No sidelobe-level detection or symmetry analysis yet. Pattern classification limited to 3 categories.
2. **Function Calling Compatibility** — works across all major model families. Older Ollama models may have limited tool support.
3. **Ollama Vision** — only vision-capable models (llava, llama3.2-vision, gemma3) support plot analysis; others fall back to text-only.

---

## Architecture

```
plot_antenna/
├── llm_provider.py         # Unified provider abstraction
├── ai_analysis.py          # Pure analysis logic (GUI-independent)
│   └── AntennaAnalyzer     # Reusable analysis class
├── gui/
│   └── ai_chat_mixin.py    # AI chat GUI integration
├── save.py                 # Report generation
└── api_keys.py             # Secure key management

rflect-mcp/
├── server.py
└── tools/
    ├── import_tools.py
    ├── analysis_tools.py
    ├── report_tools.py
    ├── bulk_tools.py
    ├── uwb_tools.py
    ├── cal_drift_tools.py
    └── orchestration.py    # process_folder (v4.2.0)
```

---

## Configuration

### Enabling AI

Pick one provider:
1. **OpenAI** — API key via Tools → Manage API Keys
2. **Anthropic** — `ANTHROPIC_API_KEY` env var
3. **Ollama** — running locally, no key required

Select provider via Tools → AI Settings.

### Disabling AI

If no provider is configured:
- AI Chat menu item is hidden
- `Generate Report with AI` falls back to template-only mode
- Core RFlect functionality continues to work normally

### Models Supported

- **OpenAI**: GPT-4o-mini (default), GPT-4o, GPT-5-nano, GPT-5-mini, GPT-5.2
- **Anthropic**: Claude Sonnet, Claude Opus
- **Ollama**: llama3.1, qwen2.5, mistral, llava (vision), and more

---

## Roadmap

### Shipped — v4.0.0 (Feb 2026)
- Architecture refactor (mixin-based GUI)
- Multi-provider AI (OpenAI, Anthropic, Ollama)
- Secure API-key management (Fernet, OS keyring, machine-ID binding)
- AntennaAnalyzer with HPBW, F/B ratio, batch analysis
- 11 RF engineering formula fixes
- MCP server with 20 tools

### Shipped — v4.1.x (Feb–Apr 2026)
- Maritime / horizon antenna plots (5 plot types)
- Advanced RF analysis suite (Link Budget, ITU-R, Multipath, MIMO, Wearable/SAR)
- Calibration drift tracker with cross-epoch comparison
- Active Chamber Calibration regression fix

### Shipped — v4.2.0 (May 2026)
- `process_folder` MCP orchestrator (single-call folder workflows)
- UWB analysis helper extracted from MCP wrapper for reuse
- MCP server expanded to 34 tools

### Planned — v4.3+
- AI datasheet extraction (vision-based parameter extraction from PDF/images)
- Sidelobe detection and reporting
- Automated figure insertion in reports
- Complete branding integration
- Multi-frequency comparison tables
- Simulation vs measurement comparison
- AI-powered anomaly detection
- Integration with electromagnetic simulation tools

---

## For Developers

### Testing AI Features

```python
from plot_antenna.ai_analysis import AntennaAnalyzer

data = {
    'phi': phi_angles,
    'theta': theta_angles,
    'total_gain': gain_array,
    'h_gain': h_pol_array,
    'v_gain': v_pol_array,
}

analyzer = AntennaAnalyzer(
    measurement_data=data,
    scan_type='passive',
    frequencies=[2400, 2450, 2500],
)

print(analyzer.get_gain_statistics(frequency=2400))
print(analyzer.analyze_pattern(frequency=2400))
print(analyzer.compare_polarizations(frequency=2400))
```

### Adding New Analysis Functions

1. Add function to `AntennaAnalyzer` in `ai_analysis.py`
2. Expose to the chat assistant via `gui/ai_chat_mixin.py`
3. Add tests under `tests/test_ai_analysis.py`
4. Wrap as an MCP tool under `rflect-mcp/tools/analysis_tools.py`
5. Update this document

### Contributing

See [CONTRIBUTING.md](https://github.com/RFingAdam/RFlect/blob/main/CONTRIBUTING.md).

---

## Support

- **Issues**: https://github.com/RFingAdam/RFlect/issues
- **Discussions**: https://github.com/RFingAdam/RFlect/discussions

Tag AI-feature issues with `ai`.

---

## Disclaimer

AI-generated analysis and recommendations are provided as guidance only. Always verify results with established RF engineering practices and measurements. RFlect's AI features are experimental and should not be relied upon for critical design decisions without human expert review.
