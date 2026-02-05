# RFlect - Antenna Plot Tool      <img src="./assets/smith_logo.png" alt="RFlect Logo" width="40">


**Version:** 3.2.0

RFlect is a comprehensive antenna plotting tool, currently designed specifically for visualizing and analyzing antenna measurements from the Howland Company 3100 Antenna Chamber and WTL Test Lab outputs. Additionally, it offers support for .csv VNA files from Copper Mountain RVNA and S2VNA software of S11/VSWR/Group Delay(S21(s)) measurements, making it a versatile choice for a wide range of antenna data processing needs. Through its user-friendly graphical interface, RFlect provides an intuitive way to handle various antenna metrics and visualize results.

## Installation
1. Run the provided installer from the latest release (`RFlect_vX.X.X.exe`)
2. Follow the on-screen instructions to complete the installation.

![Installation Steps](./assets/installation_steps.png)

## How to Use
### Select Scan Type:
- Choose from **Active**, **Passive**, or **.csv (VNA/S-Parameters)** scan.

  ![Scan Type Selection](./assets/scan_type_selection.png)

### Adjust Settings (if needed):
- Click the **Settings** button to open the settings window.
- Depending on your scan type selection, adjust the relevant settings.
  - For VNA/S11 LogMAG scans, you can set limit lines.
  - Select Group Delay & SFF(*future implementation*) for Group Delay Measurements, Peak to Peak Delay Difference, etc from 2-Port S2VNA .csv files
    ![VSWR Settings Window](./assets/vswr_settings_window.png)
  - For Passive scans, select between G&D or HPOL/VPOL.
  - Datasheet Plots Check Option will additionally plot Peak Phi & Theta Gain vs Frequency with Total Gain vs Freq. Also plots Elevation/Azimuth 2D cuts for 3-planes.
    ![Passive Settings Window](./assets/passive_settings_window.png)
  - For Active scans, *future implementation*

### Import Data:
- Click the **Import File(s)** button.
- Follow the on-screen prompts to select and import your data files.

### View Results:
- If you've selected a VPOL/HPOL Passive scan, or Active Scan, you can click **View Results** to visualize the data.
- For other scan types, results will be displayed after data import.

<details>
<summary><strong>Example Results (click to expand)</strong></summary>

Here are some examples of the results you can expect with this tool. Click on an image to view it in full size.

# Passive Routine
## G&D Files
### Passive G&D Comparison
![G&D File - 1D Results](./assets/python_1d_results_g&d.png)
Efficiency, Gain, and Directivity comparison of 'n' Number of G&D Files/Scans

## HPOL & VPOL Files
### Passive 1D Results
![Passive 1D Results](./assets/python_1d_results.png)
Eff(%) vs Freq., Eff(dB) vs Freq., and Total Gain vs Freq.

### Passive 2D Results
![Passive 2D Results](./assets/python_passive_2d_results_azimuth.png)

Gain Pattern Azimuth Cuts vs Phi for various Theta Angles

### Additional Passive "Datasheet" Plots
![Additional 1D Results](./assets/python_1d_results_datasheet.png)
![Additional 2D Results](./assets/python_2d_results_datasheet.png)
Peak gain for Phi & Theta Polarization in addition to Total Gain per IEEE Definition and Additional Polar plots for Azimuth, Theta=90deg, Elevation Phi=0deg&180deg, and Elevation Phi=90deg&270deg

### Passive 3D Results
![Passive 3D Results](./assets/python_passive_3d_results.png)
3D Gain Pattern for Phi, Theta, and Total Gains

# Active Routine
## TRP Files
### Active 2D Results
![Active 2D Results](./assets/python_active_2d_results_azimuth.png)
Azimuth Power Cuts vs Phi for various Theta Angles

![Additional 2D Results](./assets/python_active_2d_results_datasheet.png)
Additional Polar plots for Azimuth, Theta=90deg, Elevation Phi=0deg&180deg, and Elevation Phi=90deg&270deg

# VNA Routine
## Text/.csv Files
### 1 or 2-Port S-Parameters
![1 or 2-port S-Parameters](./assets/python_vna_results.png)
'n' number of S-Parameter Files Plotted

## 2-Port, Group Delay Measurements
![Group Delay](./assets/python_groupdelay_results.png)
Plots Group Delay vs Frequency for Various Theta (Azimuthal Rotation), Peak-to-peak Group Delay Difference over Theta, and Max Distance Error over Theta
</details>

## Additional Features

### File Management
- Save your results using the **Save Results to File** button
- Adjust the frequency and other parameters using the provided dropdown menus and input fields
- **Recent Files**: Access up to 5 recently opened files via File → Recent Files
- **Keyboard Shortcuts**: Ctrl+O (Import), Ctrl+Q (Exit)

### Menu Structure
RFlect features a professional menu bar with organized access to all functionality:

- **File Menu**
  - Import File(s)...
  - Recent Files (with history of 5 most recent files)
  - Clear Recent Files
  - Exit

- **Tools Menu**
  - Bulk Passive Processing
  - Bulk Active Processing
  - Polarization Analysis (Export)
  - Polarization Analysis (Interactive)
  - HPOL/VPOL → CST FFS Converter
  - Active Chamber Calibration
  - Generate Report
  - Generate Report with AI
  - AI Chat Assistant

- **Help Menu**
  - About RFlect
  - Manage OpenAI API Key
  - AI Settings
  - Check for Updates
  - View on GitHub
  - Report an Issue

### Polarization Analysis Tool
Calculate and visualize polarization parameters from HPOL/VPOL passive measurements:

**Two Analysis Modes (Tools Menu)**:
- **Polarization Analysis (Export)**: Batch export all frequencies to CSV/TXT files with complete polarization data
- **Polarization Analysis (Interactive)**: Live visualization with frequency selection dropdown

**Calculated Parameters**:
| Parameter | Description |
|-----------|-------------|
| Axial Ratio (AR) | Measure of polarization ellipticity in dB (Ludwig-3 definition) |
| Tilt Angle | Orientation of polarization ellipse in degrees |
| Polarization Sense | LHCP vs RHCP classification (+1/-1) |
| Cross-Pol Discrimination (XPD) | Ratio of co-pol to cross-pol power in dB |

**2D Visualizations** (6 subplots):
- AR, Tilt, Sense, and XPD contour maps
- Polar plots of AR and Tilt at horizon (θ=90°)

**3D Visualizations** (2 spherical plots):
- AR sphere with color and radius varying by axial ratio
- Polarization sense sphere (Red=LHCP, Blue=RHCP)
- High-resolution interpolation (120×180 points) for smooth rendering

### AI-Powered Features
> **⚠️ Status: Experimental** - AI features are functional but not production-ready (~80-90% complete). See [AI_STATUS.md](AI_STATUS.md) for details on what works and what needs improvement.

RFlect integrates OpenAI's API for intelligent antenna analysis:

**AI Chat Assistant** (Tools → AI Chat Assistant):
- Real-time conversational analysis of your antenna measurements
- Function-calling implementation for on-demand analysis:
  - `generate_2d_plot` - 2D radiation pattern descriptions with gain statistics
  - `generate_3d_plot` - 3D spherical pattern descriptions with peak locations
  - `get_gain_statistics` - Min/max/avg gain, standard deviation, frequency analysis
  - `analyze_pattern` - Nulls, beamwidth, front-to-back ratio, pattern type classification
  - `compare_polarizations` - HPOL vs VPOL comparison with XPD analysis
- Context-aware responses based on loaded data
- Persistent chat history within session
- Supports GPT-4o-mini, GPT-4o, GPT-5, and O3 models

**AI Settings** (Help → AI Settings):
| Setting | Options |
|---------|---------|
| Model Selection | GPT-4o-mini (recommended), GPT-4o, GPT-5-nano, GPT-5-mini, GPT-5.2 |
| Response Style | Concise (~80 words), Detailed (~200+ words) |
| Max Tokens | 50-1000 tokens |
| Reasoning Effort | none, low, medium, high, xhigh (GPT-5 models) |
| Text Verbosity | auto, low, medium, high |

**AI-Powered Report Generation**:
- Intelligent executive summaries based on measurement analysis
- Dual-API support: Chat Completions API (GPT-4) and Responses API (GPT-5)
- Context-aware conclusions and recommendations

**Secure API Key Storage**:
- Keys stored in user's AppData folder (not in app directory)
- Cross-platform support: Windows (%LOCALAPPDATA%\RFlect), macOS (~/Library/Application Support/RFlect), Linux (~/.config/RFlect)
- GUI-based key management (Help → Manage OpenAI API Key)
- Multiple storage backends: OS Keyring, user data file, environment variables

**Note**: AI features are **optional** and require a valid OpenAI API key. RFlect works fully without AI features enabled. See [AI_STATUS.md](AI_STATUS.md) for current status and roadmap.

### Batch Processing
Process multiple measurement files automatically:

**Bulk Passive Processing** (Tools Menu):
- Automatically finds all HPOL/VPOL pairs in a directory
- Processes multiple frequencies per antenna pair
- Generates 2D and 3D plots for each pair/frequency
- Saves results to organized subfolders

**Bulk Active Processing** (Tools Menu):
- Finds all TRP measurement files in a directory
- Generates 2D azimuth/elevation cuts
- Creates 3D TRP visualization plots
- Saves results to per-file subfolders

### Group Delay & Advanced Analysis
**CST Far-Field File Support**:
- Import CST simulation far-field .txt files
- Group Delay analysis from phase vs frequency data
- Fidelity factor calculations
- Envelope Correlation Coefficient (ECC) from 2-antenna HPOL/VPOL measurements

**VNA 2-Port Measurements**:
- Support for S2VNA and RVNA data formats
- Configurable limit lines on VSWR vs Frequency plots
- Frequency range filtering

### 3D Plot Enhancements
All 3D radiation pattern plots feature:
- Transparent background panes for improved axis visibility
- Coordinate axes rendered on top of surface (no depth shading)
- Increased arrow thickness (2.5px) for better visibility
- Cleaned tick labels for professional appearance
- Applied to Active TRP, Passive Gain, and Polarization 3D plots

<details>
<summary><strong>3D Plot Autoscale Options</strong></summary>

For comparison of multiple scans, configure 3D plotting via Settings:
- **Autoscale**: Automatic Z-axis scaling based on data range
- **Manual Fixed Scaling**: Set consistent min/max values across multiple plots
</details>

## Note to Users
- Always ensure the data you're importing is consistent with the scan type you've selected
- For best results, ensure that settings are appropriately adjusted before importing data
- AI features require a valid OpenAI API key (configure via Help → Manage OpenAI API Key)
- Status bar at the bottom of the window provides real-time operation feedback

## System Requirements
- Windows, macOS, or Linux
- Python 3.12+ (for source installation)
- SciPy 1.14.0+ (uses RegularGridInterpolator)
- NumPy with trapezoid function support

---

**The software is under active development, and additional features and improvements are expected in the future. Please refer to the release notes for version-specific details.**