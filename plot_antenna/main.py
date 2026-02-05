"""
Main entry point for the RFlect application.

Initializes and runs the main application window.
"""

import sys
import os

# Ensure modules can be found when bundled by PyInstaller
if getattr(sys, 'frozen', False):
    # Running as compiled executable
    sys.path.insert(0, os.path.dirname(sys.executable))
else:
    # Running as script
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import tkinter as tk
from gui import AntennaPlotGUI


def main():
    root = tk.Tk()
    app = AntennaPlotGUI(root)
    root.mainloop()


if __name__ == "__main__":
    main()
