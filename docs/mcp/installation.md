# MCP Installation

Add RFlect's MCP server to your MCP client. The server is at `rflect-mcp/server.py`.

## Prerequisites

- RFlect installed (`pip install -e .` or pre-built binary)
- `rflect-mcp` dependencies installed:

```bash
cd rflect-mcp
pip install -r requirements.txt
```

## Claude Code

Add to `~/.claude/settings.json` (or use a project-local `.mcp.json`):

```json
{
  "mcpServers": {
    "rflect": {
      "command": "/absolute/path/to/RFlect/.venv/bin/python",
      "args": ["/absolute/path/to/RFlect/rflect-mcp/server.py"]
    }
  }
}
```

On Windows use forward slashes or escaped backslashes:

```json
{
  "mcpServers": {
    "rflect": {
      "command": "C:/Users/you/RFlect/.venv/Scripts/python.exe",
      "args": ["C:/Users/you/RFlect/rflect-mcp/server.py"]
    }
  }
}
```

Restart Claude Code; you should see 34 RFlect tools available.

## Cline

`.cline/mcp_settings.json` in your project root:

```json
{
  "mcpServers": {
    "rflect": {
      "command": "python",
      "args": ["/absolute/path/to/RFlect/rflect-mcp/server.py"],
      "env": {}
    }
  }
}
```

## Continue (VS Code)

Continue uses `~/.continue/config.json`. Add an MCP server entry:

```json
{
  "experimental": {
    "modelContextProtocolServers": [
      {
        "transport": {
          "type": "stdio",
          "command": "python",
          "args": ["/absolute/path/to/RFlect/rflect-mcp/server.py"]
        }
      }
    ]
  }
}
```

## Other MCP clients

Any client supporting stdio MCP can host the server. The launch command is always:

```
python /path/to/rflect-mcp/server.py
```

No special args required.

## Verifying

After registration, ask the assistant: "What RFlect tools do you have?" — it should list 34. Or run the smoke test from a shell:

```bash
python -c "
import sys, os
sys.path.insert(0, os.path.abspath('rflect-mcp'))
from mcp.server.fastmcp import FastMCP
from tools.import_tools import register_import_tools
from tools.bulk_tools import register_bulk_tools
from tools.uwb_tools import register_uwb_tools
from tools.cal_drift_tools import register_cal_drift_tools
from tools.report_tools import register_report_tools
from tools.analysis_tools import register_analysis_tools
from tools.orchestration import register_orchestration_tools
m = FastMCP('t')
for r in (register_import_tools, register_analysis_tools, register_report_tools,
          register_bulk_tools, register_uwb_tools, register_cal_drift_tools,
          register_orchestration_tools):
    r(m)
print(f'Tools registered: {len(m._tool_manager._tools)}')
print(f'process_folder present: {\"process_folder\" in m._tool_manager._tools}')
"
```

Expected: `Tools registered: 34` and `process_folder present: True`.

## Troubleshooting

If the server fails to connect, see [Troubleshooting](troubleshooting.md).
