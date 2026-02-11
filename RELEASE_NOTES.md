# RFlect - Release Notes

## Version 4.1.4 (02/11/2026)

**Feature release — horizon band TRP, efficiency calculations, and enhanced maritime statistics.**

### New Features
- **Horizon TRP**: Integrated radiated power over the horizon band (the "donut" between theta min/max), computed using sin(θ)-weighted numerical integration
- **Full Sphere TRP**: Total radiated power integrated over the entire measurement sphere for reference
- **Horizon Efficiency**: Percentage of total radiated power concentrated in the horizon band — the key figure of merit for maritime antennas
- **3D pattern statistics**: The 3D masked horizon plot now includes an annotation box with max/min/avg, horizon TRP, full TRP, and efficiency

### Improvements
- **Multi-cut polar plot**: Horizon statistics page now shows 3–5 polar cuts spanning the full horizon band instead of a single θ=90° cut
- **Enhanced statistics table**: Added Horizon TRP, Full Sphere TRP, and Horizon Efficiency rows with appropriate labels (Gain/Directivity for passive, Power/TRP for active)
- **Batch processing**: Disabled interactive matplotlib during batch jobs so figure windows no longer briefly pop up and interfere with queuing additional work

---

## Version 4.1.3 (02/10/2026)

**Patch release — coverage threshold and horizon statistics fixes.**

### Bug Fixes
- **Coverage threshold reference line**: The -3 dB reference line in conical cuts and GOA plots was hardcoded at `y=-3` on the Y-axis (meaningless for active dBm data). Now drawn relative to peak using the coverage threshold setting, and legend shows the absolute value (e.g., "-3 dB ref (17.5 dBm)")
- **Configurable threshold**: Changing the coverage threshold setting (e.g., to -6 dB) now applies to the reference line in all maritime Cartesian plots
- **MEG denominator bug**: Fixed incorrect array shape reference in MEG calculation (`gain_2d.shape[1]` vs `horizon_gain.shape[1]`)
- **MEG label for active data**: Horizon statistics table now shows "Avg EIRP (sin-θ weighted)" for active power data instead of "MEG" which is only meaningful for passive gain

---

## Version 4.1.2 (02/10/2026)

**Patch release — maritime plot title corrections.**

### Improvements
- **Maritime plot titles**: All maritime plot titles now dynamically show "Gain (dBi)" for passive or "Power (dBm)" for active measurements instead of hardcoded "Gain"
- **Theta range in titles**: Conical cuts and Gain/Power-over-Azimuth plots now display the theta cut range in the title
- **GOA summary annotation**: Added max/min/avg summary below the Gain/Power-over-Azimuth plot
- **Horizon statistics title**: Now includes data type and unit (e.g., "Horizon Gain Statistics @ 2400 MHz (dBi)")
- **3D masked pattern title**: Now shows "3D Gain Pattern" or "3D Power Pattern" based on data type

---

## Version 4.1.1 (02/09/2026)

**Patch release with 8 bug fixes for v4.1.0.**

### Bug Fixes
- **Settings dialog crash**: Added missing `SECTION_HEADER_FONT` import in dialogs_mixin — the maritime settings section crashed the entire settings dialog
- **HPOL/VPOL file matching**: Replaced brittle `[:-4]` filename slice with regex that correctly strips `AP_HPol`/`AP_VPol` suffixes
- **Matplotlib mainloop conflict**: Added `plt.ion()` to prevent "main thread is not in main loop" errors during passive processing
- **Batch processing warnings**: Suppressed noisy "GUI outside main thread" and "Tight layout not applied" warnings during bulk runs
- **3D axis labels**: Added white stroke outline so X/Y/Z labels are readable when occluded by the radiation pattern surface
- **Mercator summary overlap**: Moved gain summary annotation below X-axis labels
- **Bulk settings pass-through**: Maritime settings (theta min/max, cuts, gain threshold) and axis scaling now forwarded from GUI to batch routines
- **Black formatting**: Fixed CI formatting failures across 17 files

---

## Version 4.1.0 (02/09/2026)

**Maritime antenna plots, Windows installer overhaul, and startup crash fixes.**

### Maritime / Horizon Antenna Visualization
- **5 new plot types** for on-water and maritime antenna analysis, focusing on the horizon region (theta 60-120 deg):
  - Mercator heatmap of gain vs azimuth/elevation
  - Conical cuts at configurable elevation angles
  - Gain-over-azimuth at horizon
  - Horizon statistics table (min/max/mean gain, coverage)
  - 3D radiation pattern with horizon band highlighting
- Controlled by a settings toggle (off by default) in active/passive settings dialogs
- `get_horizon_statistics()` added to AntennaAnalyzer and MCP server
- Integrated into all 4 entry points (View/Save/Bulk Passive/Bulk Active)
- Desaturation-based 3D masking (avoids depth-sort alpha artifacts)
- 45 new tests (391 total, 0 regressions)

### Windows Installer Overhaul
- **App icon**: Smith chart icon on exe, Start Menu shortcuts, desktop shortcut, installer wizard, and uninstall entry
- **No console window**: Fixed `console=True` in PyInstaller spec — GUI-only with `runw.exe` bootloader
- **Upgrade handling**: `UsePreviousAppDir=yes` for seamless in-place upgrades; `[InstallDelete]` cleans up old `RFlect_v*.exe` from pre-v4 installs
- **Release notes**: Shown after install via `InfoAfterFile`
- **Uninstall cleanup**: Removes settings.json and assets directory
- **Smaller exe**: Excludes torch, tensorflow, jupyter, sklearn, and other unused packages
- **Consistent naming**: Release assets use `v` prefix (`RFlect_v4.1.0.exe`, `RFlect_v4.1.0_linux`)

### Bug Fixes
- **Startup crash fix**: `check_for_updates` no longer crashes when `log_text` widget isn't ready or when network requests fail
- **Network error handling**: `get_latest_release` catches `RequestException` instead of letting connection errors propagate
- **Release workflow**: Fixed Inno Setup Action parameters, awk release notes extraction, and YAML parsing

---

## Version 4.0.0 (02/08/2026) - MAJOR RELEASE

**Complete architecture refactoring, multi-provider AI support, 11 RF engineering fixes, secure API key management, MCP server with 23 tools, UWB analysis suite, and 346 tests.**

### RF Engineering Fixes
- **Diversity gain**: Vaughan-Andersen formula `DG = 10*sqrt(1 - ECC^2)` replacing incorrect log-based formula
- **Axial ratio**: Polarization ellipse semi-axes with phase difference delta (`cos(2*delta)` discriminant)
- **XPD from AR**: Field ratio uses `20*log10` (was incorrectly `10*log10`)
- **TRP calculation**: IEEE solid-angle integration verified to 0.002 dB of chamber reference
- **Average gain**: Linear domain averaging instead of dB domain
- **HPBW**: Modular arithmetic for 0/360 degree boundary wrapping
- **NaN propagation floor** in `Total_Gain_dB` (prevents `log(0)`)
- **angles_match**: `np.isclose()` with `bool()` wrapper for floating-point comparison
- **capacity_monte_carlo**: Enforces scalar ECC input
- **Frequency alignment**: Validates HPOL/VPOL frequency match at import
- **UTF-8 encoding** on file readers (`determine_polarization`, `extract_passive_frequencies`)

### Modern GUI Overhaul
- Custom dark ttk theme based on `clam` with styled widgets across the entire application
- Branded header bar with Smith chart logo, red "RFlect" title, subtitle, and version badge
- Dark theme applied to ALL settings Toplevel dialogs (active, passive, VSWR)
- Action buttons bar with flat, icon-prefixed buttons and hover effects
- Monospace output log panel with dark background
- Color-coded logs: Info (white), Success (green), Warning (amber), Error (red)
- Dark-themed menus, combobox dropdowns, tooltips, and scrollbars
- Increased default window size to 850x600 with 700x500 minimum
- WCAG AA contrast fix: `DISABLED_FG_COLOR #A0A0A0` (5.6:1 ratio on `#2E2E2E`)
- Ctrl+R / F5 keyboard shortcuts for process data
- Re-import confirmation dialog before overwriting loaded data
- Bulk processing runs in background thread with progress window
- Semantic version comparison for update checks
- Processing lock prevents double-clicks during data processing
- VSWR input validation with frequency range check
- Settings persistence for VSWR limits via `user_settings.json`

### Multi-Provider AI Support
- **Unified LLM Provider Abstraction** (`llm_provider.py`)
  - Common interface for OpenAI, Anthropic, and Ollama
  - Unified data types: `LLMMessage`, `ToolDefinition`, `ToolCall`, `LLMResponse`
  - Provider-agnostic tool calling loop
  - Factory function `create_provider()` and `get_available_providers()`
- **OpenAI Provider**: Chat Completions API (GPT-4) + Responses API (GPT-5) with vision
- **Anthropic Provider**: Messages API with tool use and vision
- **Ollama Provider**: Local LLM support (llama3.1, qwen2.5, llava for vision, etc.)
- **LLM timeout/retry**: OpenAI/Anthropic `timeout=30s` `max_retries=3`, Ollama `timeout=60s`
- **AI Chat Assistant**: Quick-action buttons, multi-turn conversations, rich measurement context
- **AI Settings Dialog**: Provider selection, model lists, Ollama URL field
- **Report Generation**: Provider-aware error messages, works with any configured provider

### Secure API Key Management
- **Fernet AES-128 encryption** (HMAC-SHA256) with PBKDF2 key derivation (600K iterations)
- **Machine-ID encryption key**: `/etc/machine-id` (Linux), `IOPlatformUUID` (macOS), `MachineGuid` (Windows)
- OS keyring integration (Windows Credential Manager, macOS Keychain, Linux keyring)
- Restrictive file permissions (`chmod 600` / Windows ACL)
- Multi-provider tabbed dialog (OpenAI, Anthropic, Ollama)
- Key validation via threaded "Test Connection" button
- Keys stored in `_key_cache` dict instead of `os.environ`
- Legacy base64 auto-migration from pre-4.0.0 storage

### AI Analysis Engine
- `AntennaAnalyzer` class: GUI-independent, reusable for MCP and programmatic access
- HPBW (Half-Power Beamwidth) calculation for E-plane and H-plane
- Front-to-back ratio with proper direction identification
- Batch frequency analysis: resonance detection, 3dB bandwidth, gain stability
- Pattern classification (omnidirectional, sectoral, directional)
- Antenna engineering domain knowledge in AI prompts

### Architecture Refactoring
- **GUI Refactored to Mixin-Based Design**
  - Migrated from monolithic `gui.py` (4,331 lines) to modular architecture:
    `main_window.py`, `dialogs_mixin.py`, `ai_chat_mixin.py`, `tools_mixin.py`, `callbacks_mixin.py`
- Consolidated duplicate code (`DualOutput`, utility functions, API key methods)
- Fixed 8 bare `except:` clauses with specific exception types
- Proper Python package with `__init__.py` and package metadata

### UWB Analysis Engine
- **System Fidelity Factor (SFF)**: Cross-correlation-based `SFF = max_τ |⟨s(t), r(t-τ)⟩| / (‖s‖·‖r‖)` with quality thresholds (Excellent/Very Good/Good/Fair/Poor)
- **Phase reconstruction**: `φ(f) = φ₀ - 2π∫τ_g(f')df'` from group delay via cumulative trapezoidal integration
- **Complex S21 from S2VNA data**: Reconstruct phase from S21(s) group delay + S21(dB) magnitude
- **UWB pulse library**: Gaussian monocycle, modulated Gaussian, 5th derivative Gaussian — auto-centered on measurement band
- **Transfer function extraction**: Free-space channel removal `H(f) = S21·(4πfd/c)·exp(j2πfd/c)`
- **Impulse response**: IFFT with Blackman window, pulse width and ringing metrics
- **S11/VSWR analysis**: Impedance bandwidth, fractional bandwidth, VSWR conversion
- **Multi-angle SFF**: SFF vs orientation with mean across all angles
- **Touchstone .s2p support**: Manual parser (no scikit-rf dependency) for RI/MA/DB formats and Hz/kHz/MHz/GHz units
- **UWB plots**: SFF vs angle, group delay, impulse response, transfer function, input/output pulse overlay, S11/VSWR, group delay variation
- Fixed broken `calculate_SFF_with_gaussian_pulse()` — was using magnitude-only IFFT, now uses proper phase-reconstructed complex S21 + cross-correlation

### MCP Server (23 Tools)
- FastMCP-based server for Claude Code, Cline, and other AI assistants
- Import tools: `import_antenna_file`, `import_antenna_folder`, `import_passive_pair`, `import_active_processed`
- Analysis tools: gain statistics, pattern analysis, polarization comparison
- Report tools: DOCX generation with AI summaries and YAML template engine
- Bulk tools: batch process passive/active folders, CST conversion, file validation
- UWB tools: `calculate_sff_from_files`, `analyze_uwb_channel`, `get_impedance_bandwidth`
- Thread-safe measurement store with `threading.Lock`
- End-to-end analysis pipeline verified against chamber reference data

### Plotting/Parser
- **turbo** colormap replaces **jet** for perceptual uniformity
- DPI 300 for saved figures (was default 100)
- `check_matching_files` uses keyword-based search instead of hardcoded line indices
- Infinite loop prevention in passive parser
- `_get_gain_grid` returns `None` on reshape mismatch instead of 1D data
- NF2FF caching by (frequency, file pair)
- Individual figure close on reset instead of `plt.close("all")`

### Release Infrastructure
- `requirements.txt` and `requirements-dev.txt` with versioned dependencies
- `pyproject.toml` following PEP 621 with optional dependency groups (dev, ai, exe)
- `.bumpversion.cfg` for automated version bumping
- GitHub Actions CI: multi-OS (Ubuntu, Windows, macOS), multi-Python (3.11, 3.12), coverage, linting
- GitHub Actions release workflow: Windows .exe build on version tags

### Testing
- 346 tests, all passing
- `test_mcp_integration.py`: 66 MCP integration tests covering all 20+ tools
- `test_uwb_analysis.py`: 33 synthetic UWB tests (93% coverage on `uwb_analysis.py`)
- `test_uwb_real_data.py`: 8 real data integration tests with GroupDelay measurement files
- `test_real_data_integration.py`: Real BLE and LoRa chamber data tests
- `test_api_keys.py`, `test_llm_provider.py`, `test_ai_analysis.py`
- 24% overall code coverage

### Bug Fixes
- Fixed mousewheel crash in scrollable dialogs (global binding persisted after dialog close)
- Fixed PyInstaller compatibility (upgraded to 6.18.0 for setuptools 80+)
- Reduced .exe size from ~3.1 GB to ~135 MB via targeted excludes
- Lambda scoping bug in exception handlers (Python 3 deletes `e` after except block)
- File parser IndexError protection for malformed TRP headers

### Migration Notes
- No user action required — update via installer
- Developers: Run `pip install -r requirements.txt` to update dependencies
- API key storage location unchanged (AppData/RFlect)
- All existing measurement files remain compatible

---

## Version 3.2.0 (11/18/2025)
- **Added Interactive Polarization Analysis Tool**
  - Calculate and visualize polarization parameters from HPOL/VPOL passive measurements
  - Two analysis modes available in Tools menu:
    - **Polarization Analysis (Export)**: Batch export all frequencies to CSV/TXT files
    - **Polarization Analysis (Interactive)**: Live visualization with frequency selection
  - Calculated parameters include:
    - Axial Ratio (AR) in dB - measure of polarization ellipticity
    - Tilt Angle in degrees - orientation of polarization ellipse
    - Polarization Sense - LHCP vs RHCP classification
    - Cross-Polarization Discrimination (XPD) in dB
  - **2D Visualizations** (6 subplots):
    - AR, Tilt, Sense, and XPD contour maps
    - Polar plots of AR and Tilt at horizon (θ=90°)
  - **3D Visualizations** (2 spherical plots):
    - AR sphere with color and radius varying by axial ratio
    - Polarization sense sphere (Red=LHCP, Blue=RHCP)
    - High-resolution interpolation (120×180 points) for smooth rendering
    - Enhanced shading and antialiasing for professional appearance
- **Enhanced 3D Plot Visualization**
  - Improved coordinate axis visibility in all 3D radiation pattern plots
  - Transparent background panes allow axes to show through the surface
  - Disabled depth shading on coordinate arrows (always render on top)
  - Increased arrow thickness (2.5px) for better visibility
  - Removed all axis tick labels for cleaner appearance
  - Applied to Active TRP, Passive Gain, and Polarization 3D plots
- **Professional GUI Enhancements**
  - Added comprehensive menu structure (File/Tools/Help)
  - **File Menu**: Import, Recent Files (5 file history), Clear Recent, Exit
  - **Help Menu**: About, API Key Management, AI Settings, Updates, GitHub links
  - Recent files automatically load correct scan type (Active/Passive)
  - Status bar at bottom shows real-time operation feedback
  - Keyboard shortcuts: Ctrl+O (import), Ctrl+Q (exit)
  - Professional About dialog with logo, version, license info
- **Advanced AI Configuration & Features**
  - **AI Settings Dialog** (Help → AI Settings):
    - Model selection dropdown (GPT-4o-mini, GPT-4o, GPT-5, O3, etc.)
    - Response style configuration (concise/detailed)
    - Max token/verbosity control
    - Reasoning effort levels for GPT-5 models
    - Settings saved to `config_local.py` for persistence
  - **AI Chat Assistant** (Tools → AI Chat Assistant):
    - **Fully integrated with OpenAI API** for real-time conversational analysis
    - **Complete Function-Calling Implementation**:
      - `generate_2d_plot` - Create 2D radiation pattern descriptions with gain statistics
      - `generate_3d_plot` - Create 3D spherical pattern descriptions with peak locations
      - `get_gain_statistics` - Calculate min/max/avg gain, standard deviation, frequency analysis
      - `analyze_pattern` - Comprehensive pattern analysis (nulls, beamwidth, front-to-back ratio, pattern type)
      - `compare_polarizations` - Full HPOL vs VPOL comparison with cross-pol discrimination (XPD)
    - **Interactive Analysis**: AI executes Python functions on demand based on user questions
    - **Detailed Metrics**: All functions return comprehensive JSON data for AI analysis
    - Context-aware responses based on current scan type and loaded files
    - Persistent chat history within session for follow-up questions
    - Automatic data context injection (frequencies, file names, scan types)
    - Interactive Q&A about radiation patterns, efficiency, and RF metrics
    - Supports all configured AI models (GPT-4, GPT-5, O3, etc.)
    - Clean error handling with descriptive messages
    - Shift+Enter for newline, Enter to send message
    - Clean error handling with helpful troubleshooting messages
  - **Secure API Key Storage**:
    - Keys stored in user's AppData folder (not in app directory)
    - Base64 obfuscation for security
    - Encrypted storage at %LOCALAPPDATA%\RFlect\.openai_key
    - GUI-based key management (no manual .env editing)
    - Works across Windows/macOS/Linux
  - OpenAI API key automatically enables AI features in Tools menu
- **Fixed Critical Task Manager Hanging Issue**
  - Added proper window close protocol handler (`WM_DELETE_WINDOW`) in `gui.py`
  - Implemented `on_closing()` cleanup method to properly shut down matplotlib figures and Tkinter
  - Application now terminates cleanly when closed, eliminating persistent background processes
- **Resolved Memory Leaks in Matplotlib Figure Management**
  - Added `plt.close('all')` before all `plt.show()` calls in `groupdelay.py` (5 locations)
  - Added `plt.close('all')` before all `plt.show()` calls in `plotting.py` (11 locations)
  - Prevents accumulation of matplotlib figures in memory during interactive plotting sessions
  - Significantly reduces memory footprint when generating multiple plots
- **Code Quality and Maintainability Improvements**
- **Enhanced AI-Powered Report Generation**
  - Dual-API support: Chat Completions API (GPT-4 family) and Responses API (GPT-5 family)
  - Configurable AI behavior via `config.py`: model selection, verbosity, reasoning effort, response style
  - Intelligent executive summaries and context-aware conclusions based on measurement analysis
- Replaced deprecated `scipy.interpolate.interp2d` with `RegularGridInterpolator` for SciPy 1.14.0+ compatibility
- Fixed deprecated NumPy function: replaced `np.trapz` with `np.trapezoid` for Python 3.12+ compatibility
- Corrected type checking errors in `file_utils.py` for robust null validation on angle parameters
- Fixed type checking errors in `gui.py`:
  - Fixed type mismatches for `zmin`/`zmax` parameters to accept float values
  - Added null checking for user input dialogs to prevent runtime errors
- Fixed type checking errors in `plotting.py`:
- Corrected 3D plotting axis for consistency with displaying
- Additional settings options for mix/max of active TRP measurements
- Added support for limit lines for VNA 2-port measurements
- Added batch processing for multiple passive scans in a folder with results saved in a subfolder
- Added batch processing for multiple active scans in a folder with results saved in a subfolder
- Added support for 3D plotting Autoscale or manual fixed scaling for comparison of multiple scans
- Added support for Group Delay/Fidelity/ECC analysis for CST files
- Added support for Envelope Correlation Coefficient (ECC) from 2 antenna measurements of HPOL/VPOL files

## Version 3.1.0 (2/5/2025)
- Corrected Active TRP Measurement Calculation before Phi=0/360 append
- Corrected Active Save Results to File for TRP Measurements
- Passive G&D File Alternate processing for Min/Max Gain & Efficiency in Tabled format
- VSWR or S11/S22/S21/S12 LogMag .csv file import for VNA measurements and tabled min/max

## Version 3.0.0 (10/23/2024)
- Implemented Active 3D plotting
- Implement Save Results to File for Active Scans
- Added Setting for Active & __TODO__ Passive 3D plotting to use a interpolated meshing
- Added Console in the GUI for Status Updates/Errors/Warnings/Messages
- Added Support for Report Generation to Word Documents via save.py
  - Implement Report Generation for an entire selected folder to capture all tested frequencies and results
  - Impelent Report Generation for both Active and Passive Scans
- Added Support for OpenAI API for Image Captioning for report generation
    - Create openapi.env file for API Key for option to appear .gitignore added. Source only
- __TODO__ Added placeholder for NF2FF conversion for Passive measurements in the nearfield for   investigation for low frequency measurements
  
## Version 2.2.2 (07/31/2024)
- Mirror Active & Passive 2D plots to appropriate conventions 
- Added Labeling for 2D plots to show the Phi angle on the plot.

## Version 2.2.1 (07/30/2024)
- Added Support For Hpol/Vpol Conversion to CST .ffs, Farfield results for Importing to CST
- Added Support for Active antenna chamber calibration file generation

## Version 2.1.1 (10/19/2023)
- Added Support For Agilent VNA Group Delay Measurements

## Version 2.1.0 (10/16/2023)
- Added Copper Mountain S2VNA/M5090 .csv export support (S11(dB), S22(dB), S21(dB) or S12(dB), S21(s) or S12(s))
- Fixed Matplotlib backend to display plots properly
- More Robust Group Delay File Import/Parsing Method
- Updated README

## Version 2.0.0 (10/13/2023)
- Started Active Scan Implementation
  - Azimuth Power Pattern Cuts vs. Phi for various values of Theta
  - Added Azimuth and Elevation Cuts, Theta=90deg plane, Phi = 0/180deg plane, and Phi = 90/270deg plane for TRP cuts
  - TODO 3D Plots for Phi, Theta and TRP
- Added Gain Summary for Passive 2D Azimuth Gain Cuts
- Fixed Issue with Save Results Overwriting Existing Files
- Added Group Delay, Peak Group Delay Difference, Max. Distance Error, from 2-port VNA measurements (S21(s) or S12(s))
- Total Gain not plotting in Save Results Routine

## Version 1.3.0 (10/03/2023)
- Updated 2D Passive Plots Formatting
- Added "Datasheet Plot" Setting for Passive Antenna Plots
  - Phi & Theta Gain Plots vs Frequency for Passive/Antenna Scans
    - Gain Summary for max peak Theta and Phi gain
  - Added Azimuth and Elevation Cuts, Theta=90deg plane, Phi = 0/180deg plane, and Phi = 90/270deg plane
- Formatting fixes on 3D plots

## Version 1.2.0 (09/06/2023)
- Passive Azimuth Cuts Wrap 360deg
- Updated 2D Passive Plots Formatting
- Fixed Interpolation on 3D Passive Plots, now set in config.py - higher resolution leads to long load times
- Save Results to Folder at Selected Frequency for Passive Scans

## Version 1.1.0 (09/01/2023)
- Codebase has been refactored into separate modules
- Added Docstrings
- Updated Images in README.md
- Added Passive 3D Plots, Theta, Phi, and Total Gain

## Version 1.0.3 (08/24/2023)
- Fixed Bug with Passive Scan Selection

## Version 1.0.2 (08/24/2023)
- Added Assets Folder for images
- Testing Automatic Version Updates

## Version 1.0.1 (08/24/2023)
- Automatic GitHub Update Checks & Download

## Version 1.0.0 (08/24/2023)

### Release Features:
- **Passive Scans Support**:
  - G&D files for 2D plotting results.
  - HPOL/VPOL files for 2D & 3D results.
  
- **VSWR/S11 LogMag .csv Files**:
  - Import and processing support.
  - VSWR/S11 settings with adjustable limit lines.
  - Graphical representation of VSWR/S11 data.

- **Graph Plotting**:
  - Metrics including: Gain, Efficiency, Directivity, Return Loss.

- **User-Friendly GUI**:
  - Easy data import.
  - Visualization of results.
  - Configuration of settings.

### Known Issues:
- Check Project Issues Page: 
  https://github.com/RFingAdam/RFlect/issues


