# RFlect - Release Notes

## Version 1.4.0 (10/06/2023)
- Started Active Scan Implementation
  - Azimuth Power Pattern Cuts vs. Phi for various values of Theta
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

### Features:
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


