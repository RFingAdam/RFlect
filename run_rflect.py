"""
RFlect Launcher Script

This is the entry point for PyInstaller builds.
It imports and runs the main application from the plot_antenna package.
"""

if __name__ == "__main__":
    from plot_antenna.main import main
    main()
