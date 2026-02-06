# RFlect - Release Notes

## Version 4.0.0 (02/04/2026) - MAJOR RELEASE

**This is a major release with complete architecture refactoring and professional release infrastructure.**

### 🏗️ Architecture Refactoring
- **GUI Refactored to Mixin-Based Design**
  - Migrated from monolithic `gui.py` (4,331 lines) to modular architecture
  - GUI organized into focused mixins:
    - `main_window.py` (585 lines) - Core AntennaPlotGUI class
    - `dialogs_mixin.py` (1,162 lines) - Dialog methods
    - `ai_chat_mixin.py` (1,492 lines) - AI Chat functionality
    - `tools_mixin.py` (1,006 lines) - Bulk processing & analysis tools
    - `callbacks_mixin.py` (900 lines) - File I/O & data processing
  - Benefits: Improved maintainability, testability, and extensibility
  - Archived legacy code to `_archive/` directory for reference

### 🧹 Code Quality & Cleanup
- **Eliminated Duplicate Code**
  - Consolidated `DualOutput` class (was duplicated in 2 locations)
  - Consolidated utility functions (`resource_path`, `get_user_data_dir`)
  - Consolidated API key management methods
  - Removed redundant `gui_components/` package
- **Improved Error Handling**
  - Fixed 8 bare `except:` clauses with specific exception types
  - Added proper logging for error conditions
  - Better error messages for user guidance
- **Security Improvements**
  - Standardized environment file naming (`.env`)
  - Enhanced `.gitignore` patterns for credential protection
  - Multi-layer API key storage: OS keyring → file → environment variables

### 🚀 Release Infrastructure
- **Dependency Management**
  - Added `requirements.txt` with production dependencies
  - Added `requirements-dev.txt` with testing and code quality tools
  - All dependencies properly versioned and documented
- **Modern Python Packaging**
  - Created `pyproject.toml` following PEP 621 standards
  - Entry point: `rflect` command for CLI usage
  - Optional dependencies for dev tools and PyInstaller builds
- **Testing Infrastructure**
  - Comprehensive test suite with pytest
  - Unit tests for RF calculations (diversity gain, MIMO capacity, TRP, process data)
  - AI analysis tests (gain stats, pattern analysis, polarization comparison)
  - MCP tools tests (dataclass creation, no-data error handling)
  - File parsing tests with import verification
- **CI/CD Automation**
  - GitHub Actions workflow for automated testing
    - Multi-OS (Ubuntu, Windows, macOS)
    - Multi-Python (3.11, 3.12)
    - Code coverage reporting to Codecov
    - Linting with flake8 and black
  - GitHub Actions workflow for automated releases
    - Builds Windows .exe on version tags
    - Creates GitHub releases with release notes
    - Uploads artifacts for distribution
- **Version Management**
  - Added `.bumpversion.cfg` for automated version bumping
  - Centralized version tracking across all files
  - Created `plot_antenna/__init__.py` with package metadata

### 🤖 AI Architecture Improvements
- **Extracted AI Analysis Logic**
  - Created `ai_analysis.py` module with `AntennaAnalyzer` class
  - GUI-independent, reusable for MCP server and programmatic access
  - Pure analysis functions that don't depend on GUI state
  - Added antenna engineering domain knowledge for AI prompts
- **Pattern Analysis Functions**
  - HPBW (Half-Power Beamwidth) calculation for E-plane and H-plane
  - Front-to-back ratio with proper direction identification
  - Batch frequency analysis: resonance detection, 3dB bandwidth, gain stability
  - Pattern classification (omnidirectional, sectoral, directional)
- **AI Feature Status Documentation**
  - Created `AI_STATUS.md` documenting feature completeness
  - Clearly marked AI features as experimental (~80-90% complete)
  - Documented what works and what needs improvement
- **Optional AI Features**
  - AI features require opt-in (OpenAI API key)
  - Core RFlect functionality works without AI
  - Clear status indicators in UI

### 🔌 MCP Server (Programmatic Access)
- **RFlect MCP Server** (`rflect-mcp/`)
  - FastMCP-based server for Claude Code, Cline, and other AI assistants
  - Import tools: load measurement files programmatically
  - Analysis tools: gain statistics, pattern analysis, polarization comparison
  - Report tools: generate DOCX reports with AI summaries and smart filtering
  - YAML-based report template engine (`rflect-mcp/templates/default.yaml`)
- **Bulk Processing MCP Tools**
  - `bulk_process_passive()`: Batch process HPOL/VPOL pairs in a folder
  - `bulk_process_active()`: Batch process TRP files in a folder
  - `validate_file_pair()`: Validate HPOL/VPOL file pairing
  - `convert_to_cst()`: Convert measurement files to CST .ffs format
  - `list_measurement_files()`: Scan folders for recognized file types

### 📚 Documentation
- **Developer Guidelines**
  - Created `CONTRIBUTING.md` with development setup
  - Code style guidelines (Black, 100 char lines)
  - Testing guidelines and coverage goals
  - PR process and release workflow
- **Updated README**
  - Added installation methods (standalone .exe, from source)
  - Clarified AI feature status (experimental)
  - Added development instructions
- **Project Organization**
  - Clear file structure documentation
  - Architecture explanations
  - Contribution guidelines and priorities

### 🔧 Technical Improvements
- **Package Structure**
  - Proper Python package with `__init__.py`
  - Version available programmatically: `import plot_antenna; print(plot_antenna.__version__)`
  - Better import organization
- **Build System**
  - PyInstaller configured for .exe builds
  - Automated in CI/CD pipeline
  - Version-tagged releases

### ⚠️ Breaking Changes
**None** - All existing workflows remain compatible. This is a major version bump due to architectural refactoring, but all features are preserved.

### 🎯 Future Roadmap
- **v4.1** (Planned Q2 2026)
  - Sidelobe detection and reporting
  - System Fidelity Factor calculation (#31)
  - Automated figure insertion in reports
  - Multi-frequency comparison tables
- **v4.2** (Planned Q3 2026)
  - Vision API integration (AI analyzes plots visually)
  - Simulation vs measurement comparison
  - AI-powered anomaly detection

### 📦 Installation
Download `RFlect_v4.0.0.exe` from [GitHub Releases](https://github.com/RFingAdam/RFlect/releases)

**For Developers**:
```bash
git clone https://github.com/RFingAdam/RFlect.git
cd RFlect
pip install -r requirements.txt
python -m plot_antenna.main
```

### 📝 Migration Notes
- No user action required - update via installer
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


