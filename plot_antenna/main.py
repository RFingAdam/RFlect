"""
Main entry point for the RFlect application.

Initializes and runs the main application window.
"""

import sys
import os
import tkinter as tk

# Add parent directory to path for direct script execution
if __name__ == "__main__" and __package__ is None:
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from plot_antenna.gui import AntennaPlotGUI
else:
    from .gui import AntennaPlotGUI


def main():
    root = tk.Tk()
    app = AntennaPlotGUI(root)
    root.mainloop()


if __name__ == "__main__":
    main()
