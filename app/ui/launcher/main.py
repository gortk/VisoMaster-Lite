# main.py
# ---------------------------------------------------------------------------
# Entry point for VisoMaster Lite Launcher
# ---------------------------------------------------------------------------
# Performs basic sanity checks, applies the True-Dark theme,
# and initializes the main launcher window.
# ---------------------------------------------------------------------------

import sys
from PySide6 import QtWidgets
from .core import PATHS, must_exist, apply_theme_to_app
from .launcher_window import LauncherWindow


def main():
    """Initialize and run the VisoMaster-Lite Launcher."""
    sys.stdout.flush()

    try:
        # --- Sanity Checks ---
        must_exist(PATHS["PYTHON_EXE"], "portable venv python")
        must_exist(PATHS["GIT_EXE"], "portable git.exe")
        must_exist(PATHS["APP_DIR"], "VisoMaster-Lite directory")
        must_exist(PATHS["MAIN_PY"], "VisoMaster-Lite main.py")

        # --- Enable Ctrl+C in console ---
        import signal

        signal.signal(signal.SIGINT, signal.SIG_DFL)

        # --- Create and run UI ---
        app = QtWidgets.QApplication(sys.argv)
        apply_theme_to_app(app)

        win = LauncherWindow()
        win.show()

        sys.exit(app.exec())

    except Exception as e:
        print(f"[Launcher] Failed to start the VisoMaster-Lite Launcher: {e}")
        sys.exit(1)
