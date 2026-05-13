# MCP Troubleshooting

## Server fails to start

```
ModuleNotFoundError: No module named 'plot_antenna'
```

The server adds the parent directory to `sys.path`, but only if it's launched from a Python that has access to the RFlect package. Check:

- Your MCP config's `command` points at the **right Python** (the venv where you ran `pip install -e .`)
- Or the server is launched from a Python that can `import plot_antenna` (try `python -c "import plot_antenna; print(plot_antenna.__version__)"`)

## "No data loaded"

You called an analysis or report tool before importing. Fix:

```python
import_passive_pair("/path/to/X_HPol.txt", "/path/to/X_VPol.txt")
list_loaded_data()  # confirm
get_all_analysis(2450.0)
```

Or skip the import step entirely with `process_folder("/path/to/dir", intent="passive")`.

## "AI Summary requires API key"

Report generation tried to use an AI provider but none is configured. Options:

- **GUI**: Tools → Manage API Keys → enter OpenAI/Anthropic key
- **Environment**: `export OPENAI_API_KEY=...` or `export ANTHROPIC_API_KEY=...`
- **Ollama**: run `ollama serve` locally, no key needed
- **Skip AI**: pass `"ai_executive_summary": false` in options

## Report too large / OOM

Use filtering:

```python
generate_report("/tmp/report.docx", {
    "frequencies":          [2450],
    "polarizations":        ["total"],
    "include_3d_plots":     False,
    "max_frequencies_in_table": 5,
})
```

## File import fails

```
Error importing file: …
```

Check:

- File is in a [supported format](../hardware/file-formats.md)
- Encoding is UTF-8 or ASCII
- For passive: filename has `_HPol` or `_VPol` (or you're using `import_passive_pair` explicitly)
- For active: filename contains `TRP` (any case)

Run `list_measurement_files(folder)` to see what RFlect detects before importing.

## `process_folder` returns `intent_used=null`

Two cases:

| Warning                               | Meaning                                                |
|---------------------------------------|--------------------------------------------------------|
| `folder_not_found: <path>`            | The directory doesn't exist or is a file               |
| `no_files_matched_intent: <intent>`   | Folder is empty or contains no recognized RFlect files |

Run `list_measurement_files(folder)` to see what's in the folder.

## Mixed-intents warning

```
"warnings": ["mixed_intents_detected: passive(2), active(1) — chose passive"]
```

Not an error — RFlect picked the highest-priority intent. If you want a different choice, pass `intent=` explicitly:

```python
process_folder(folder, intent="active")  # ignore the passive pair
```

## Cal-drift import is idempotent — why?

Each `TRP Cal *.txt` is content-hashed (SHA-256) on ingest. Re-running `cal_drift_ingest` on the same directory will report `skipped_duplicate` instead of double-recording.

## Tool count check

```bash
python -c "
import sys, os
sys.path.insert(0, os.path.abspath('rflect-mcp'))
from mcp.server.fastmcp import FastMCP
from tools.import_tools import register_import_tools
# ... import the rest ...
m = FastMCP('t')
# ... register them ...
print(len(m._tool_manager._tools))
"
```

Should print `34` on v4.2.0+.

## Logs

Run the server directly to see errors:

```bash
python rflect-mcp/server.py
```

It will block waiting for MCP stdio traffic; Ctrl+C to exit. Any import errors print to stderr before that point.

## Still stuck?

Open an issue with:

- Your MCP client (Claude Code / Cline / Continue / other) and version
- Your RFlect version (`python -c "import plot_antenna; print(plot_antenna.__version__)"`)
- The exact `mcp` config block
- Output of `python rflect-mcp/server.py` (any errors)

https://github.com/RFingAdam/RFlect/issues
