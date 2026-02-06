"""
Main entry point for the RFlect application.

Initializes and runs the main application window.
"""

import sys
import os
import tkinter as tk

# Handle imports for different execution contexts:
# 1. PyInstaller frozen executable
# 2. Package mode (pip install -e . or pytest)
# 3. Direct script execution (python plot_antenna/main.py)

if getattr(sys, "frozen", False):
    # PyInstaller bundles modules with full package paths
    from plot_antenna.gui import AntennaPlotGUI
elif __name__ == "__main__" and __package__ is None:
    # Direct script execution - add parent to path
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from plot_antenna.gui import AntennaPlotGUI
else:
    # Package mode (pytest, pip install -e .)
    from .gui import AntennaPlotGUI


def main():
    root = tk.Tk()
    app = AntennaPlotGUI(root)
    root.mainloop()


if __name__ == "__main__":
    main()
