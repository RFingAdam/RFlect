"""
RFlect GUI Package

This package contains the refactored GUI components for RFlect.
The main AntennaPlotGUI class is assembled from multiple mixins for maintainability.

Structure:
- main_window.py: Core AntennaPlotGUI class combining all mixins
- base_protocol.py: Protocol/interface definitions for type checking
- dialogs_mixin.py: Dialog methods (About, Settings)
- tools_mixin.py: Bulk processing, polarization analysis, converters
- callbacks_mixin.py: File import, data processing, save operations
- utils.py: Shared utility functions (DualOutput, resource_path, etc.)
"""

from .main_window import AntennaPlotGUI
from . import utils

__all__ = ["AntennaPlotGUI", "utils"]
