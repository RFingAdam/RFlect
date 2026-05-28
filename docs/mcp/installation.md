# MCP Installation

Add RFlect's MCP server to your MCP client. The server entry point is
`rflect-mcp/server.py`. RFlect runs identically on **Linux, macOS, and Windows**
— the only per-OS differences are the Python path and where your client keeps
its config.

## 1. Install RFlect + MCP dependencies

=== "Linux / macOS"

    ```bash
    git clone https://github.com/RFingAdam/RFlect.git
    cd RFlect
    python3 -m venv .venv
    source .venv/bin/activate
    pip install -e .
    pip install -r rflect-mcp/requirements.txt
    ```

    Your Python interpreter is then `…/RFlect/.venv/bin/python`.

=== "Windows (PowerShell)"

    ```powershell
    git clone https://github.com/RFingAdam/RFlect.git
    cd RFlect
    py -m venv .venv
    .venv\Scripts\Activate.ps1
    pip install -e .
    pip install -r rflect-mcp\requirements.txt
    ```

    Your Python interpreter is then `…\RFlect\.venv\Scripts\python.exe`.

!!! note "No API key required"
    RFlect makes no LLM/API calls and needs no key or subscription. The MCP
    server runs fully offline.

## 2. Register with your MCP client

### Claude Code

Config file location:

| OS | Path |
|----|------|
| Linux / macOS | `~/.claude/settings.json` |
| Windows | `%USERPROFILE%\.claude\settings.json` |

(Or use a project-local `.mcp.json` in your repo root, which works the same on every OS.)

=== "Linux / macOS"

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

=== "Windows"

    Use forward slashes (or escaped `\\`) in JSON, and the `Scripts` venv path:

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

Restart Claude Code; you should see **41** RFlect tools available.

### Cline (VS Code)

`.cline/mcp_settings.json` in your project root (same shape on every OS — just
swap the Python path):

=== "Linux / macOS"

    ```json
    {
      "mcpServers": {
        "rflect": {
          "command": "/absolute/path/to/RFlect/.venv/bin/python",
          "args": ["/absolute/path/to/RFlect/rflect-mcp/server.py"],
          "env": {}
        }
      }
    }
    ```

=== "Windows"

    ```json
    {
      "mcpServers": {
        "rflect": {
          "command": "C:/Users/you/RFlect/.venv/Scripts/python.exe",
          "args": ["C:/Users/you/RFlect/rflect-mcp/server.py"],
          "env": {}
        }
      }
    }
    ```

### Continue (VS Code)

Continue uses `~/.continue/config.json` (Linux/macOS) or
`%USERPROFILE%\.continue\config.json` (Windows):

```json
{
  "experimental": {
    "modelContextProtocolServers": [
      {
        "transport": {
          "type": "stdio",
          "command": "/absolute/path/to/RFlect/.venv/bin/python",
          "args": ["/absolute/path/to/RFlect/rflect-mcp/server.py"]
        }
      }
    ]
  }
}
```

### Other MCP clients

Any client supporting stdio MCP can host the server. The launch command is
always your venv Python plus the server path — no special args:

```
<venv-python> /path/to/rflect-mcp/server.py
```

## 3. Verify

Ask the assistant *"What RFlect tools do you have?"* — it should list **41**.
Or run the smoke test from a shell (works on every OS):

=== "Linux / macOS"

    ```bash
    cd RFlect
    .venv/bin/python -c "import sys, os; sys.path.insert(0, os.path.abspath('rflect-mcp')); import server; print('tools:', len(server.mcp._tool_manager._tools))"
    ```

=== "Windows (PowerShell)"

    ```powershell
    cd RFlect
    .venv\Scripts\python.exe -c "import sys, os; sys.path.insert(0, os.path.abspath('rflect-mcp')); import server; print('tools:', len(server.mcp._tool_manager._tools))"
    ```

Expected: `tools: 41`.

## Troubleshooting

If the server fails to connect, see [Troubleshooting](troubleshooting.md). The
most common cause is pointing `command` at a Python that doesn't have the
`rflect-mcp/requirements.txt` dependencies installed — use the venv interpreter,
not the system `python`.
