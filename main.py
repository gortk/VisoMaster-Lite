import os
import sys
import argparse
import traceback
from datetime import datetime
from pathlib import Path

# --- PyTorch VRAM Optimization ---
# MUST be set BEFORE importing torch (which occurs during _run_app).
# Instructs the caching allocator to release unused memory segments back to the CUDA driver.
# Dropped 'max_split_size_mb:128' to prevent heavy cudaMalloc/cudaFree churn and implicit
# device synchronizations when allocating large frame tensors (e.g., 4K, VR180).
# Left overridable so other values can be A/B'd against peak VRAM without a
# rebuild, e.g. PYTORCH_CUDA_ALLOC_CONF="max_split_size_mb:512,garbage_collection_threshold:0.8"
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "garbage_collection_threshold:0.8")

import torch

torch.set_grad_enabled(False)


def _write_crash_log(exc: BaseException) -> Path:
    """Persist a full traceback to disk so the diagnostic survives even if the
    console window closes before the user can copy it.

    Returns the path of the written log so the caller can print it.
    """
    log_dir = Path(__file__).resolve().parent / "crash_logs"
    log_dir.mkdir(exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = log_dir / f"crash_{stamp}.log"
    with open(log_path, "w", encoding="utf-8") as f:
        f.write(f"VisoMaster crash report — {datetime.now().isoformat()}\n")
        f.write("=" * 70 + "\n")
        try:
            import platform

            f.write(f"Python:   {sys.version}\n")
            f.write(f"Platform: {platform.platform()}\n")
        except Exception:
            pass
        f.write("=" * 70 + "\n\n")
        traceback.print_exception(type(exc), exc, exc.__traceback__, file=f)
    return log_path


def _show_early_splash():
    """Show the splash as soon as a QApplication exists, i.e. BEFORE the heavy
    application imports (onnxruntime, cv2, every processor/UI module).

    Returns the QSplashScreen, or None if the image is missing/unreadable.
    Purely cosmetic: any failure here is swallowed so it can never block start-up.
    """
    try:
        from PySide6 import QtCore, QtGui, QtWidgets

        splash_path = (
            Path(__file__).resolve().parent
            / "app"
            / "ui"
            / "core"
            / "media"
            / "splash.png"
        )
        pixmap = QtGui.QPixmap(str(splash_path))
        if pixmap.isNull():
            return None
        screen = QtWidgets.QApplication.primaryScreen()
        geo = screen.availableGeometry()
        dpr = screen.devicePixelRatio() or 1.0
        # Same sizing rules as MainWindow._show_loading_splash: real pixel size,
        # never larger than 80% of the screen.
        max_w = int(geo.width() * dpr * 0.8)
        max_h = int(geo.height() * dpr * 0.8)
        if pixmap.width() > max_w or pixmap.height() > max_h:
            pixmap = pixmap.scaled(
                max_w,
                max_h,
                QtCore.Qt.AspectRatioMode.KeepAspectRatio,
                QtCore.Qt.TransformationMode.SmoothTransformation,
            )
        pixmap.setDevicePixelRatio(dpr)
        splash = QtWidgets.QSplashScreen(pixmap)
        splash.setWindowFlag(QtCore.Qt.WindowType.FramelessWindowHint, True)
        splash.setWindowFlag(QtCore.Qt.WindowType.WindowStaysOnTopHint, False)
        splash.move(geo.center() - splash.rect().center())
        splash.show()
        QtWidgets.QApplication.processEvents()
        return splash
    except Exception:
        return None


def _run_app() -> None:
    """Boot the Qt app. Imports are inside the function so any startup error is
    captured by the outer try/except (otherwise a top-level import error would
    bypass the crash-log writer)."""
    from PySide6 import QtWidgets
    from app.ui.core.proxy_style import ProxyStyle

    parser = argparse.ArgumentParser(description="VisoMaster")
    parser.add_argument(
        "--gpu-id",
        type=int,
        default=0,
        help="CUDA GPU device ID to use (default: 0)",
    )
    args, remaining = parser.parse_known_args()

    # torch is already imported (module top, before PySide6 is loaded for the
    # first time), so the import order that was used before is unchanged.
    app = QtWidgets.QApplication(remaining)
    early_splash = _show_early_splash()

    try:
        # Heavy part: onnxruntime, cv2 and every app module. The splash is
        # already on screen while this runs.
        from app.ui import main_ui

        import qdarktheme

        app.setStyle(ProxyStyle())
        with open("app/ui/styles/true_dark_styles.qss", "r") as f:
            _style = f.read()
            _style = (
                qdarktheme.load_stylesheet(
                    theme="dark", custom_colors={"primary": "#4090a3"}
                )
                + "\n"
                + _style
            )
            app.setStyleSheet(_style)

        if early_splash is not None:
            # Hand over to MainWindow's own splash without a gap: close this one
            # right after MainWindow has shown its splash.
            _orig_show_splash = main_ui.MainWindow._show_loading_splash

            def _show_splash_then_close_early(self, *a, **kw):
                result = _orig_show_splash(self, *a, **kw)
                try:
                    early_splash.close()
                except Exception:
                    pass
                return result

            main_ui.MainWindow._show_loading_splash = _show_splash_then_close_early

        window = main_ui.MainWindow(gpu_id=args.gpu_id)
    finally:
        if early_splash is not None:
            try:
                early_splash.close()
            except Exception:
                pass
    from app.ui.widgets.actions import frame_step_actions

    frame_step_actions.add_frame_step_controls(window)
    window.show()
    app.exec()


if __name__ == "__main__":
    try:
        _run_app()
    except KeyboardInterrupt:
        pass
    except Exception as e:
        log_path = _write_crash_log(e)
        print("\n" + "=" * 70)
        print("[FATAL] VisoMaster crashed.")
        print("  Crash log written to:")
        print(f"    {log_path}")
        print(f"  Error: {e}")
        print("=" * 70)
        print("\nFull traceback:")
        traceback.print_exc()
        sys.exit(1)
