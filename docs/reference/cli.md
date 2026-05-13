# CLI

RFlect is primarily GUI- and MCP-driven, but there are a handful of useful command-line entry points.

## Launch GUI

```bash
python run_rflect.py
```

Or, after `pip install -e .`:

```bash
rflect
```

(See `[project.scripts]` in `pyproject.toml`.)

## Launch MCP server

```bash
python rflect-mcp/server.py
```

Blocks waiting for stdio MCP traffic. Usually launched by your MCP client, not by hand — see [MCP installation](../mcp/installation.md).

## Test suite

```bash
.venv/bin/python -m pytest tests/                 # full suite
.venv/bin/python -m pytest tests/test_calculations.py -v
.venv/bin/python -m pytest tests/test_mcp_process_folder.py -v
```

## Build distributable

PyInstaller spec is in the repo:

```bash
pyinstaller RFlect.spec
```

Hidden imports of note (already configured): `PIL._tkinter_finder` for Pillow logo loading inside the bundled exe.

## Windows installer

Inno Setup script `installer.iss`. Build with:

```bat
iscc installer.iss
```

Or override version inline:

```bat
iscc /DRFLECT_VERSION=4.2.0 installer.iss
```

## Version bumping

```bash
bump2version patch     # 4.2.0 → 4.2.1
bump2version minor     # 4.2.0 → 4.3.0
bump2version major     # 4.2.0 → 5.0.0
```

Sources of truth for the version (`.bumpversion.cfg`):

- `pyproject.toml` — `version = "..."`
- `plot_antenna/__init__.py` — `__version__ = "..."`
- `README.md` — version badge
- `installer.iss` — `RFLECT_VERSION`
- `settings.json` — `CURRENT_VERSION`
