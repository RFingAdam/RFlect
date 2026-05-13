# AI Chat Assistant

In-GUI conversational analysis of loaded measurements.

## Opening it

Tools → AI Chat (only visible when a provider is configured).

The chat window has:
- Conversation pane
- Input box
- Quick-action buttons: Gain Stats, Pattern, Polarization, All Freqs

## Function calling

The chat assistant has five callable functions wired to the AI:

| Function                    | Returns                                                |
|-----------------------------|--------------------------------------------------------|
| `get_gain_statistics()`     | Min / max / spherical avg gain                         |
| `analyze_pattern()`         | HPBW, F/B ratio, nulls, sidelobes                      |
| `compare_polarizations()`   | AR, tilt, XPD, sense                                   |
| `generate_2d_plot()`        | 2D pattern description (textual)                       |
| `generate_3d_plot()`        | 3D pattern description (textual)                       |

The model picks tools based on your question and chains them as needed.

## Multi-turn context

Conversation history is preserved across messages. The assistant remembers which measurement you loaded and which frequency you've been discussing.

## Example prompts

- "What's the peak gain at 2.45 GHz?"
- "Compare HPOL vs VPOL at this frequency."
- "Is the pattern more directional or more omnidirectional?"
- "Summarize the antenna's performance across all loaded frequencies."

## Vision

If your provider supports vision (GPT-4o+, Claude, llava/llama3.2-vision), you can ask the model to interpret a rendered plot. The chat window's "attach plot" button base64-encodes the current matplotlib figure and includes it in the message.

## Limits

- Older Ollama models may have limited tool-calling support
- Pattern analysis is currently limited to three pattern types
- No sidelobe-level detection yet

See [AI Status](status.md) for the full list.

## Programmatic access

The same analyses are available via MCP without the GUI:

```python
get_gain_statistics(2450.0)
analyze_pattern(2450.0, "total")
compare_polarizations(2450.0)
```

See [MCP Tools Reference](../mcp/tools-reference.md).
