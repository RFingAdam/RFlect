# AI Overview

RFlect has optional AI integration on two surfaces:

1. **In-GUI** — AI Chat Assistant + AI-powered DOCX reports
2. **MCP** — agent clients (Claude Code, Cline, …) drive RFlect programmatically via 34 tools

Both are powered by the same unified provider abstraction in `plot_antenna/llm_provider.py`, which supports **OpenAI**, **Anthropic**, and **Ollama**.

## Always optional

AI features are **completely optional**. Without a configured provider:

- The AI Chat menu item is hidden
- `Generate Report with AI` falls back to template-only mode
- Core RFlect functionality (plotting, calculation, MCP tool calling) continues to work normally

## What AI is good at here

- **Executive summaries** in reports — "this antenna's peak gain is 5.2 dBi at 2450 MHz; the pattern is mildly directional with HPBW of 78° in E-plane"
- **Pattern interpretation** — comparing HPOL vs VPOL, calling out polarization mismatches
- **Conversational queries** over loaded measurements — "what's the worst-case gain in the 2.4 GHz band?"
- **Vision over rendered plots** — given a 3D pattern PNG, describe what the operator should notice

## What AI is **not** doing

- The math. All gain, TRP, polarization, and bandwidth calculations are pure-Python in `plot_antenna/calculations.py` and `plot_antenna/ai_analysis.py`. The AI layer only consumes the numbers.
- Hardware control. RFlect doesn't drive instruments.
- Replacing engineering judgment. The disclaimer in [AI Status](status.md) applies.

## Where to next

- [Status](status.md) — current state, what works, what's pending
- [Providers](providers.md) — configure OpenAI / Anthropic / Ollama
- [Chat](chat.md) — using the in-GUI AI Chat Assistant
- [Reports](reports.md) — AI-augmented DOCX reports
