"""Floating-window support for the right-pane accordion sections.

A ``CollapsibleSection`` can be dragged out of the Parameters pane into its own
window (``DetachedSectionWindow``).  While it is away, a slim
``DetachedSectionPlaceholder`` keeps its slot in the pane so the section goes
back to exactly the same position when it is docked again.

Only the two small widgets live here; the logic that moves a section in and out
lives on ``CollapsibleSection`` (see ``widget_components.py``).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from PySide6 import QtCore, QtGui, QtWidgets

if TYPE_CHECKING:
    from app.ui.main_ui import MainWindow
    from app.ui.widgets.widget_components import CollapsibleSection


class DetachedSectionPlaceholder(QtWidgets.QFrame):
    """Slim bar left in the pane where a section used to be."""

    activate_requested = QtCore.Signal()
    dock_requested = QtCore.Signal()

    def __init__(self, title: str, parent: QtWidgets.QWidget | None = None):
        super().__init__(parent)
        self.setObjectName("detachedSectionPlaceholder")
        self.setFixedHeight(24)
        self.setSizePolicy(
            QtWidgets.QSizePolicy.Policy.Expanding,
            QtWidgets.QSizePolicy.Policy.Fixed,
        )
        self.setCursor(QtCore.Qt.CursorShape.PointingHandCursor)
        self.setToolTip(
            "This section is open in its own window.\n"
            "Click to bring that window to the front."
        )
        self.setStyleSheet(
            "QFrame#detachedSectionPlaceholder {"
            " border: 1px dashed #4a4552; border-radius: 4px;"
            " background: transparent; }"
            "QFrame#detachedSectionPlaceholder QLabel {"
            " color: #8d8895; font-style: italic;"
            " border: none; background: transparent; }"
            "QFrame#detachedSectionPlaceholder QToolButton {"
            " border: none; background: transparent;"
            " color: #b9b4c2; padding: 0 6px; }"
            "QFrame#detachedSectionPlaceholder QToolButton:hover {"
            " color: #ffffff; }"
        )

        layout = QtWidgets.QHBoxLayout(self)
        layout.setContentsMargins(8, 0, 2, 0)
        layout.setSpacing(4)

        label = QtWidgets.QLabel(f"{title} \u2014 in a separate window", self)
        label.setSizePolicy(
            QtWidgets.QSizePolicy.Policy.Ignored,
            QtWidgets.QSizePolicy.Policy.Preferred,
        )
        dock_button = QtWidgets.QToolButton(self)
        dock_button.setText("Dock back")
        dock_button.setCursor(QtCore.Qt.CursorShape.PointingHandCursor)
        dock_button.setToolTip("Put this section back into the panel")
        dock_button.clicked.connect(lambda _checked=False: self.dock_requested.emit())

        layout.addWidget(label, 1)
        layout.addWidget(dock_button)

    def mousePressEvent(self, event: QtGui.QMouseEvent) -> None:
        if event.button() == QtCore.Qt.MouseButton.LeftButton:
            self.activate_requested.emit()
            event.accept()
            return
        super().mousePressEvent(event)


class DetachedSectionWindow(QtWidgets.QWidget):
    """A small tool window that hosts one ``CollapsibleSection``."""

    def __init__(
        self,
        main_window: "MainWindow",
        section: "CollapsibleSection",
        title: str,
    ):
        flags = (
            QtCore.Qt.WindowType.Tool
            | QtCore.Qt.WindowType.CustomizeWindowHint
            | QtCore.Qt.WindowType.WindowTitleHint
            | QtCore.Qt.WindowType.WindowCloseButtonHint
        )
        # Parenting to the main window keeps the app stylesheet/theme and keeps
        # the section windows on top of the main window.
        super().__init__(main_window, flags)
        self._main_window = main_window
        self._section = section
        self._discarding = False
        self._dragging = False
        self._drag_offset = QtCore.QPoint()

        self.setWindowTitle(title)
        self.setObjectName("detachedSectionWindow")
        self.setMinimumWidth(260)

        outer = QtWidgets.QVBoxLayout(self)
        outer.setContentsMargins(4, 4, 4, 4)
        outer.setSpacing(0)

        self._scroll = QtWidgets.QScrollArea(self)
        self._scroll.setWidgetResizable(True)
        self._scroll.setFrameShape(QtWidgets.QFrame.Shape.NoFrame)
        self._scroll.setHorizontalScrollBarPolicy(
            QtCore.Qt.ScrollBarPolicy.ScrollBarAsNeeded
        )
        self._scroll.setVerticalScrollBarPolicy(
            QtCore.Qt.ScrollBarPolicy.ScrollBarAsNeeded
        )
        self._container = QtWidgets.QWidget()
        self._container_layout = QtWidgets.QVBoxLayout(self._container)
        self._container_layout.setContentsMargins(0, 0, 6, 0)
        self._container_layout.setSpacing(0)
        self._scroll.setWidget(self._container)
        outer.addWidget(self._scroll)

    # ------------------------------------------------------------------ hosting
    def adopt_section(self, width: int) -> None:
        self._container_layout.addWidget(self._section)
        self._container_layout.addStretch(1)
        self._section.show()
        self._section.expanded_changed.connect(self._schedule_fit)
        self.resize(max(width, 260) + 22, 120)
        self._fit_height()

    def release_section(self) -> None:
        """Hand the section back (called by ``CollapsibleSection.dock``)."""
        self._end_drag()
        try:
            self._section.expanded_changed.disconnect(self._schedule_fit)
        except (TypeError, RuntimeError):
            pass
        self._container_layout.removeWidget(self._section)
        self._section.hide()
        self._section.setParent(None)

    def discard(self) -> None:
        self._discarding = True
        self._end_drag()
        self.hide()
        self.deleteLater()

    # ------------------------------------------------------------------- sizing
    def _schedule_fit(self, *_args) -> None:
        QtCore.QTimer.singleShot(0, self._fit_height)

    def _fit_height(self) -> None:
        if self._discarding:
            return
        self._container_layout.activate()
        margins = self.layout().contentsMargins()
        wanted = (
            self._section.sizeHint().height()
            + margins.top()
            + margins.bottom()
            + 2 * self._scroll.frameWidth()
            + 6
        )
        screen = self.screen()
        limit = int(screen.availableGeometry().height() * 0.9) if screen else 900
        self.resize(self.width(), max(min(wanted, limit), 40))

    # ---------------------------------------------------------------- placement
    def position_for_drag(self, global_pos: QtCore.QPoint) -> None:
        """Put the window so the cursor sits on its title bar."""
        offset_x = min(self.width() // 2, 80)
        self.move(global_pos.x() - offset_x, global_pos.y() + 10)

    def position_beside_main(self) -> None:
        main_frame = self._main_window.frameGeometry()
        screen = self._main_window.screen()
        avail = screen.availableGeometry() if screen else QtCore.QRect(0, 0, 1920, 1080)
        if main_frame.right() + 10 + self.width() <= avail.right():
            x = main_frame.right() + 10
        else:
            x = main_frame.left() + 60
        self.move(x, main_frame.top() + 80)

    def ensure_on_screen(self) -> None:
        """Pull the window back if a saved position is on a vanished monitor."""
        frame = self.frameGeometry()
        for screen in QtGui.QGuiApplication.screens():
            overlap = screen.availableGeometry().intersected(frame)
            if overlap.width() >= 80 and overlap.height() >= 30:
                return
        self.position_beside_main()

    # ----------------------------------------------------------------- dragging
    def begin_drag(self, global_pos: QtCore.QPoint) -> None:
        """Keep moving the window with the cursor until the button is released."""
        buttons = QtWidgets.QApplication.mouseButtons()
        if not (buttons & QtCore.Qt.MouseButton.LeftButton):
            return
        self._drag_offset = global_pos - self.frameGeometry().topLeft()
        handle = self.windowHandle()
        # Wayland compositors ignore client-side window moves; ask for a
        # compositor-driven move there instead.
        if (
            handle is not None
            and QtGui.QGuiApplication.platformName().startswith("wayland")
            and handle.startSystemMove()
        ):
            return
        self._dragging = True
        self.grabMouse()

    def _end_drag(self) -> None:
        if self._dragging:
            self._dragging = False
            try:
                self.releaseMouse()
            except RuntimeError:
                pass

    def mouseMoveEvent(self, event: QtGui.QMouseEvent) -> None:
        if self._dragging:
            if not (event.buttons() & QtCore.Qt.MouseButton.LeftButton):
                self._end_drag()
                return
            self.move(event.globalPosition().toPoint() - self._drag_offset)
            event.accept()
            return
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event: QtGui.QMouseEvent) -> None:
        if self._dragging:
            self._end_drag()
            event.accept()
            return
        super().mouseReleaseEvent(event)

    # ------------------------------------------------------------------- events
    def keyPressEvent(self, event: QtGui.QKeyEvent) -> None:
        # Unhandled keys reach the main window when the section is docked
        # (play/pause, frame step ...); keep that working from this window too.
        try:
            self._main_window.keyPressEvent(event)
        except Exception:
            super().keyPressEvent(event)

    def closeEvent(self, event: QtGui.QCloseEvent) -> None:
        if self._discarding:
            event.accept()
            return
        # The window's X button docks the section back instead of losing it.
        event.ignore()
        QtCore.QTimer.singleShot(0, self._section.dock)
