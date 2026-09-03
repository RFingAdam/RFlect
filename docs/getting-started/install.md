# Install

RFlect runs on Windows, Linux, and macOS. Python 3.11 or newer. The GUI needs Tk/tkinter (usually bundled with the official Windows/macOS Python installers; on Debian/Ubuntu install `python3-tk`).

## Pre-built binaries (recommended)

Grab the latest release from [GitHub Releases](https://github.com/RFingAdam/RFlect/releases).

=== "Windows"

    - `RFlect_Installer_vX.X.X.exe`: Inno Setup installer (registers app, creates shortcuts)
    - `RFlect_vX.X.X.exe`: standalone portable binary

=== "Linux"

    ```bash
    chmod +x RFlect_vX.X.X_linux
    ./RFlect_vX.X.X_linux
    ```

=== "macOS"

    Build from source: pre-built macOS binaries are not yet published.

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

After `pip install -e .` the same GUI is also the `rflect` console script.

## Developer install

```bash
pip install -r requirements-dev.txt
python -m pytest tests/
pyinstaller RFlect.spec            # build exe
```

`requirements-dev.txt` already includes the MCP SDK (`mcp>=1.0.0,<2.0.0`) so MCP integration tests can collect.

The repo also ships an editable install through `pyproject.toml`:

```bash
pip install -e .
pip install -e ".[mcp]"            # MCP extra
pip install -e ".[instruments]"    # optional pyvisa / pyserial
```

## MCP server (optional)

If you want Claude Code or Cline to drive RFlect:

```bash
pip install -e ".[mcp]"
pip install -r rflect-mcp/requirements.txt
python rflect-mcp/server.py
```

Configuration lives in your MCP client. See [MCP installation](../mcp/installation.md).

## Example data

This repo does not ship customer measurements. Synthetic cal-drift files live in `tests/fixtures/cal_drift/`. Point optional real-file tests at a local folder with `RFLECT_TEST_DATA_DIR`. Example plot screenshots are in `assets/` at the repo root.

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
