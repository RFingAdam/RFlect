"""
RFlect - Antenna Measurement Visualization and Analysis Tool

A comprehensive tool for visualizing and analyzing antenna measurements from:
- Howland Company 3100 Antenna Chamber
- WTL Test Lab outputs
- VNA measurements (Copper Mountain, S2VNA)
- CST simulation exports

Features:
- 2D and 3D radiation pattern visualization
- TRP (Total Radiated Power) calculations
- Polarization analysis (Axial Ratio, Tilt Angle, XPD)
- AI-powered analysis with OpenAI integration
- Batch processing capabilities
- Professional report generation
"""

__version__ = "4.1.0"
__author__ = "Adam"
__license__ = "GPL-3.0"

from .main import main

__all__ = ["main", "__version__", "__author__", "__license__"]
