"""
Here are the rules for extension authors:

1. Basic File Rules

    Must be a .py file inside the extensions/ folder.

    Cannot start with an underscore (e.g., _test.py is ignored).

    Must define a def setup(main_window): function that returns a class instance (or object).
---------------------------------------------------------------------------------------------------------------------
2. The on_frame_processed Hook

    Format: The frame passed to this hook is a NumPy array in RGB format (shape: (H, W, 3), dtype: uint8).

    In-Place Modification: You must modify the array in-place (e.g., frame[:] = new_frame or using cv2.putText directly on it). If you just do frame = new_frame, it will do nothing.

    No I/O Blocking: Do not read/write to disk or use time.sleep() in this hook. It runs 30-60 times a second and will freeze the app.

    Use Existing Libraries: Do not require external libraries that the user has to pip install. Stick to cv2, numpy, torch, or PySide6 (which are already bundled).
---------------------------------------------------------------------------------------------------------------------
3. UI Modifications

    The Parameters tabs (Face Swap, Face Editor, Restorers, ...) are shown in a
    full-width tab row directly below the menu bar. The right-hand Parameters
    panel only shows the content of the selected tab and has no tab strip of
    its own.

    Add your own tab in setup() with:

        main_window.add_parameter_tab(widget, "Title")      # recommended

    (main_window.tabWidget.addTab(widget, "Title") still works and also shows
    up in the top row, but add_parameter_tab is the supported way.)

    - Keep the title SHORT (about 14 characters or less). The row is shared by
      all tabs; when the window is too narrow it scrolls with small left/right
      buttons and the mouse wheel, so long titles just make users scroll more.
    - Your widget is displayed inside the (narrow) right panel, so lay it out
      to wrap/shrink gracefully: use QVBoxLayout, word-wrapped labels and no
      hard-coded wide widgets. Do not rely on the tab strip being above it.
    - To switch to your tab from code use
      main_window.tabWidget.setCurrentIndex(main_window.tabWidget.indexOf(widget)).
      The visible tab row follows.
    - Do NOT use main_window.tabWidget.tabBar(): that is a hidden internal bar.
      The visible bar is main_window.topTabBar (read-only use, e.g. tooltips).
    - The tab row hides/shows together with the Parameters panel (View menu).
    - Do not delete or alter core UI elements (like the main video slider or
      the Play button), or the app will crash.
---------------------------------------------------------------------------------------------------------------------
4. Thread Safety

    The on_frame_processed hook runs on the GUI thread. Heavy processing (like running a new AI model) will drop the frame rate. If you need heavy processing, push it to a background thread and just paste the result here.
---------------------------------------------------------------------------------------------------------------------
5. Minimal Example of a Valid Extension:

python

import cv2
import numpy as np
from PySide6 import QtWidgets

class MyExtension:
    def __init__(self, main_window):
        self.main_window = main_window

    def on_frame_processed(self, frame):
        # Rule 2: Modify IN-PLACE. (Example: add a green tint)
        frame[:, :, 1] = np.clip(frame[:, :, 1].astype(int) + 50, 0, 255).astype(np.uint8)

    def on_app_close(self):
        print("[EXTENSION] Shutting down...")

def setup(main_window):
    ext = MyExtension(main_window)
    
    # Rule 3: Add UI safely (the tab appears in the top tab row)
    tab = QtWidgets.QWidget()
    layout = QtWidgets.QVBoxLayout(tab)
    layout.addWidget(QtWidgets.QLabel("My Extension Settings"))
    main_window.add_parameter_tab(tab, "My Ext")
    
    return ext

---------------------------------------------------------------------------------------------------------------------
What will crash the app:

    Modifying frame.shape (e.g., resizing or cropping). The downstream pipeline expects the exact original dimensions.

    Spawning infinite while True: loops without QApplication.processEvents().

    Throwing unhandled exceptions in __init__ (wrap your code in try/except).
---------------------------------------------------------------------------------------------------------------------   
Extensions run standard Python in the main process, meaning they can interact with any external application, local file system, or network.
How they can pipe to external apps:

Since it's raw Python, an extension can use subprocess, requests, sockets, or zmq. For example:

    ComfyUI / Stable Diffusion API: Send the current frame to an API via requests.post, wait for the processed image, and overwrite the frame array.

    OBS / StreamDeck: Use obsws-python to trigger scene changes when specific faces are detected.

    External CLI Tools: Run subprocess.run() to pass a frame to a command-line upscaler (like Topaz or Real-ESRGAN standalone) and read the output back.

    Webhooks: Send a ping to a Discord bot or webhook whenever a recording finishes.

How complex can they be?

Extremely complex. Since they receive the main_window object, they have total control over the app. Users can:

    Build custom UI tabs (with complex PySide6 widgets).

    Monitor the UI state (read main_window.control).

    Hook into other lifecycle events (e.g., on_app_close, on_recording_start).

    Spawn background QThreads to do heavy processing (like running a TensorRT model) without freezing the UI.

The only limit:

The UI thread.
If your extension sends an HTTP request to ComfyUI inside on_frame_processed, it will freeze the app for 2 seconds until the response returns.
Rule of thumb: For external apps, the extension must use a background worker thread to process the frame and a queue to pass the result back to the hook.    
    
"""

"""
=============================
EXAMPLE EXTENSION STARTS HERE
=============================
"""


from PySide6 import QtWidgets, QtGui
import cv2

class WatermarkExtension:
    def __init__(self, main_window):
        self.main_window = main_window
        self.enabled = True

    def on_frame_processed(self, frame):
        if not self.enabled:
            return
            
        # Add a red text watermark to the top-left of the frame
        cv2.putText(
            frame, 
            "EXTENSION ACTIVE", 
            (20, 50), 
            cv2.FONT_HERSHEY_SIMPLEX, 
            1, 
            (0, 0, 255), 
            2, 
            cv2.LINE_AA
        )

    def on_app_close(self):
        print("[EXTENSION] Watermark extension is shutting down.")

def setup(main_window):
    # This is the entry point called by the ExtensionManager
    ext = WatermarkExtension(main_window)
    
    # Optional: Add a custom tab to the UI for the extension
    tab = QtWidgets.QWidget()
    layout = QtWidgets.QVBoxLayout(tab)
    # Tab content is shown in the narrow right panel: wrap text, avoid wide widgets.
    label = QtWidgets.QLabel("Watermark Extension Settings")
    label.setWordWrap(True)
    layout.addWidget(label)
    layout.addStretch()
    
    # Add the tab to the main UI. It appears in the tab row below the menu
    # bar; keep the title short because that row is shared by all tabs.
    # Fall back to tabWidget.addTab for older builds without the top tab row.
    add_tab = getattr(main_window, "add_parameter_tab", None)
    if add_tab is not None:
        add_tab(tab, "Watermark")
    else:
        main_window.tabWidget.addTab(tab, "Watermark")
    
    return ext
    
    
    