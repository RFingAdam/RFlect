"""
Main entry point for the RFlect application.

Initializes and runs the main application window.
"""

import tkinter as tk

# Handle both package import and direct script execution
try:
    from .gui import AntennaPlotGUI
except ImportError:
    from gui import AntennaPlotGUI


def main():
    root = tk.Tk()
    app = AntennaPlotGUI(root)
    root.mainloop()


if __name__ == "__main__":
    main()
