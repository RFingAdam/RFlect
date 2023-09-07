"""
Configuration file for the RFlect application.

This module contains constants and configuration values used throughout the application.
"""

#3D Plotting Interpolation Resolution
PHI_RESOLUTION = 120 #default 360 = 1deg spacing
THETA_RESOLUTION = 60 #default 180 - 1deg spacing

#Set Min/Max for optional 2D polar plots on passive scans - set in settings
polar_dB_max = +5.0
polar_dB_min = -20.0

# GUI Settings
    # Colors
DARK_BG_COLOR = "#2E2E2E"
LIGHT_TEXT_COLOR = "#FFFFFF"
ACCENT_BLUE_COLOR = "#4A90E2"
BUTTON_COLOR = "#3A3A3A"
HOVER_COLOR = "#4A4A4A"
ACCENT_GREEN_COLOR = "#4CAF50"

    # Fonts
HEADER_FONT = ("Arial", 14, "bold")
LABEL_FONT = ("Arial", 12)
