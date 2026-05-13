# Install

RFlect runs on Windows, Linux, and macOS. Python 3.11 or newer.

## Pre-built binaries (recommended)

Grab the latest release from [GitHub Releases](https://github.com/RFingAdam/RFlect/releases).

=== "Windows"

    - `RFlect_Installer_vX.X.X.exe` — Inno Setup installer (registers app, creates shortcuts)
    - `RFlect_vX.X.X.exe` — standalone portable binary

=== "Linux"

    ```bash
    chmod +x RFlect_vX.X.X_linux
    ./RFlect_vX.X.X_linux
    ```

=== "macOS"

    Build from source — pre-built macOS binaries are not yet published.

## From source

```bash
git clone https://github.com/RFingAdam/RFlect.git
cd RFlect
python -m venv .venv
source .venv/bin/activate          # Linux/macOS
# or:  .venv\Scripts\activate      # Windows
pip install -r requirements.txt
python run_rflect.py
```

## Developer install

```bash
pip install -r requirements-dev.txt
python -m pytest tests/            # 450+ tests
pyinstaller RFlect.spec            # build exe
```

The repo also ships an editable install entry point through `pyproject.toml`:

```bash
pip install -e .
```

## MCP server (optional)

If you want Claude Code or Cline to drive RFlect, install the MCP layer too:

```bash
cd rflect-mcp
pip install -r requirements.txt
```

Configuration lives in your MCP client — see [MCP installation](../mcp/installation.md).

## Sanity check

After install, launch the GUI:

```bash
python run_rflect.py
```

You should see the scan-type selector. If you only need the MCP server (no GUI), run:

```bash
python rflect-mcp/server.py
```

It will sit waiting for stdio MCP traffic; close it with `Ctrl+C`.
