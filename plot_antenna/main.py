import tkinter as tk
from gui import AntennaPlotGUI

"""
    Main entry point for the RFlect application.

    Initializes and runs the main application window.
"""

def main():
    root = tk.Tk()
    app = AntennaPlotGUI(root)
    root.mainloop()

if __name__ == "__main__":
    main()