# RFlect - Release Notes

## Version 3.0.1 (11/8/2024)
- Corrected Active TRP Measurement Calculation before Phi=0/360 append
- Corrected Active Save Results to File for TRP Measurements

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


