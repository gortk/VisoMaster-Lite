"""Full-width tab row for the Parameters panel.

The tabs of the right-hand Parameters panel (``MainWindow.tabWidget``) live in
a dedicated row right below the menu bar.  The ``QTabWidget`` itself is kept as
the page container (its pages, order, current index and the workspace
save/load code keep working untouched); only its own tab strip is hidden and
mirrored by a ``QTabBar`` that sits in a top tool bar spanning the window.

* The bar is sized dynamically: it shrinks with the window and, when the tabs
  do not fit, Qt's small left/right scroll buttons appear.
* The mouse wheel scrolls the tab strip (it does *not* change the active tab).
* Tabs can still be dragged to reorder; the order is mirrored back to the
  hidden ``QTabWidget`` tab bar so the saved workspace tab order still works.
* The row follows the visibility of the Parameters dock.
* Tabs added later (extensions) via ``tabWidget.addTab`` / ``insertTab`` /
  ``removeTab`` or the ``main_window.add_parameter_tab`` helper show up in the
  row automatically.
"""

from PySide6 import QtCore, QtWidgets


class ScrollableTabBar(QtWidgets.QTabBar):
    """QTabBar whose wheel scrolls the strip instead of switching tabs."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self._wheel_accum = 0
        self.setExpanding(False)
        self.setUsesScrollButtons(True)
        self.setElideMode(QtCore.Qt.TextElideMode.ElideNone)
        self.setDrawBase(False)
        self.setMovable(True)
        self.setDocumentMode(False)
        self.setFocusPolicy(QtCore.Qt.FocusPolicy.NoFocus)
        self.setSizePolicy(
            QtWidgets.QSizePolicy.Policy.Ignored, QtWidgets.QSizePolicy.Policy.Fixed
        )
        # Scroll buttons: keep them unobtrusive and theme-neutral.
        self.setStyleSheet(
            "QTabBar::tab { background: #202020; color: #f0f0f0;"
            " font-weight: bold; border: 1px solid #363636;"
            " border-bottom-color: #363636; padding: 4px 8px;"
            " border-top-left-radius: 4px; border-top-right-radius: 4px; }"
            "QTabBar::tab:selected { background: #6C5CE7; color: #f0f0f0;"
            " font-weight: bold; border-color: #4A3AB5;"
            " border-bottom-color: #6C5CE7; }"
            "QTabBar::tab:hover:!selected { background: #352E4C;"
            " color: #f0f0f0; font-weight: bold; border-color: #6C5CE7; }"
            "QTabBar QToolButton { background: transparent; border: none;"
            " margin: 0px; padding: 0px; }"
            "QTabBar QToolButton:hover:enabled { background: rgba(108,92,231,120);"
            " border-radius: 3px; }"
        )

    def _scroll_buttons(self):
        buttons = [b for b in self.findChildren(QtWidgets.QToolButton) if b.isVisible()]
        buttons.sort(key=lambda b: b.x())
        return buttons

    def wheelEvent(self, event):
        buttons = self._scroll_buttons()
        if len(buttons) < 2:
            # Everything fits: nothing to scroll (and never switch tabs here).
            event.accept()
            return
        delta = event.angleDelta()
        amount = delta.y() or delta.x()
        if amount == 0:
            event.accept()
            return
        self._wheel_accum += amount
        steps = int(self._wheel_accum / 120)
        if steps:
            self._wheel_accum -= steps * 120
            # Wheel up/left -> scroll towards the first tab.
            btn = buttons[0] if steps > 0 else buttons[-1]
            for _ in range(abs(steps)):
                if btn.isEnabled():
                    btn.click()
        event.accept()

    def sizeHint(self):
        # Height from the tabs, width irrelevant (policy is Ignored).
        hint = super().sizeHint()
        return QtCore.QSize(hint.width(), hint.height())

    def minimumSizeHint(self):
        hint = super().minimumSizeHint()
        return QtCore.QSize(0, hint.height())


class _KeepHidden(QtCore.QObject):
    """Event filter that keeps the QTabWidget's own tab strip hidden."""

    def eventFilter(self, obj, event):
        if event.type() == QtCore.QEvent.Type.Show:
            QtCore.QTimer.singleShot(0, obj.hide)
        return False


class _StackWatcher(QtCore.QObject):
    """Notices pages being added/removed in the QTabWidget's page stack."""

    _TYPES = (
        QtCore.QEvent.Type.ChildAdded,
        QtCore.QEvent.Type.ChildRemoved,
        QtCore.QEvent.Type.LayoutRequest,
    )

    def __init__(self, controller):
        super().__init__(controller)
        self._controller = controller

    def eventFilter(self, obj, event):
        if event.type() in self._TYPES:
            self._controller._queue_resync()
        return False


class TopTabBarController(QtCore.QObject):
    """Mirrors ``tab_widget`` into a ``ScrollableTabBar`` in a top tool bar."""

    def __init__(
        self,
        main_window: QtWidgets.QMainWindow,
        tab_widget: QtWidgets.QTabWidget,
        follow_widget: QtWidgets.QWidget | None = None,
    ):
        super().__init__(main_window)
        self._window = main_window
        self._tabs = tab_widget
        self._src = tab_widget.tabBar()
        self._busy = False

        self.bar = ScrollableTabBar()
        self.bar.setObjectName("topTabBar")
        self.bar.setFont(tab_widget.font())

        # Row: tabs (small side margins) + a full-width separator line below.
        self.row = QtWidgets.QWidget()
        self.row.setObjectName("topTabRow")
        row_layout = QtWidgets.QVBoxLayout(self.row)
        row_layout.setContentsMargins(0, 0, 0, 0)
        row_layout.setSpacing(0)
        tabs_holder = QtWidgets.QWidget(self.row)
        tabs_layout = QtWidgets.QHBoxLayout(tabs_holder)
        tabs_layout.setContentsMargins(5, 2, 5, 0)
        tabs_layout.setSpacing(0)
        tabs_layout.addWidget(self.bar, 1)
        row_layout.addWidget(tabs_holder)
        # Colour comes from the theme (#topTabSeparator in the .qss files).
        self.separator = QtWidgets.QFrame(self.row)
        self.separator.setObjectName("topTabSeparator")
        self.separator.setFrameShape(QtWidgets.QFrame.Shape.NoFrame)
        self.separator.setFixedHeight(2)
        row_layout.addWidget(self.separator)
        self.row.setSizePolicy(
            QtWidgets.QSizePolicy.Policy.Expanding, QtWidgets.QSizePolicy.Policy.Fixed
        )

        self.toolbar = QtWidgets.QToolBar("Parameter tabs", main_window)
        self.toolbar.setObjectName("topTabsToolBar")
        self.toolbar.setMovable(False)
        self.toolbar.setFloatable(False)
        self.toolbar.setAllowedAreas(QtCore.Qt.ToolBarArea.TopToolBarArea)
        self.toolbar.setContextMenuPolicy(QtCore.Qt.ContextMenuPolicy.PreventContextMenu)
        self.toolbar.layout().setContentsMargins(0, 0, 0, 0)
        self.toolbar.layout().setSpacing(0)
        self.toolbar.addWidget(self.row)
        main_window.addToolBar(QtCore.Qt.ToolBarArea.TopToolBarArea, self.toolbar)

        # Hide the QTabWidget's own strip (keep it alive: save/load use it).
        self._src.hide()
        self._hider = _KeepHidden(self)
        self._src.installEventFilter(self._hider)

        self._rebuild()
        self._connect(follow_widget)

    # -- mirroring -------------------------------------------------------
    def _rebuild(self):
        self._busy = True
        try:
            while self.bar.count():
                self.bar.removeTab(0)
            for i in range(self._tabs.count()):
                self.bar.addTab(self._tabs.tabText(i))
                self.bar.setTabToolTip(i, self._tabs.tabToolTip(i))
                self.bar.setTabEnabled(i, self._tabs.isTabEnabled(i))
            self.bar.setCurrentIndex(self._tabs.currentIndex())
        finally:
            self._busy = False

    def resync(self):
        """Bring the visible bar in line with the QTabWidget."""
        count = self._tabs.count()
        if self.bar.count() != count:
            self._rebuild()
            return
        self._busy = True
        try:
            for i in range(count):
                if self.bar.tabText(i) != self._tabs.tabText(i):
                    self.bar.setTabText(i, self._tabs.tabText(i))
                tip = self._tabs.tabToolTip(i)
                if self.bar.tabToolTip(i) != tip:
                    self.bar.setTabToolTip(i, tip)
                en = self._tabs.isTabEnabled(i)
                if self.bar.isTabEnabled(i) != en:
                    self.bar.setTabEnabled(i, en)
            if self.bar.currentIndex() != self._tabs.currentIndex():
                self.bar.setCurrentIndex(self._tabs.currentIndex())
        finally:
            self._busy = False

    def refresh_texts(self):
        """Backwards compatible alias of resync()."""
        self.resync()

    def _queue_resync(self):
        QtCore.QTimer.singleShot(0, self.resync)

    def _patch_tab_widget_setters(self):
        """Make Python-side text/enabled/tooltip changes show up immediately."""
        tabs = self._tabs
        for name in ("setTabText", "setTabEnabled", "setTabToolTip"):
            original = getattr(tabs, name)

            def wrapper(*args, _orig=original, **kwargs):
                result = _orig(*args, **kwargs)
                self.resync()
                return result

            setattr(tabs, name, wrapper)

    def add_tab(self, widget, title, index=None):
        """Add a page to the Parameters panel; the tab appears in the top row."""
        if index is None:
            idx = self._tabs.addTab(widget, title)
        else:
            idx = self._tabs.insertTab(index, widget, title)
        self.resync()
        return idx

    def _connect(self, follow_widget):
        # Tabs added/removed by anyone (extensions use tabWidget.addTab).
        self._stack = self._tabs.findChild(QtWidgets.QStackedWidget)
        if self._stack is not None:
            self._stack_filter = _StackWatcher(self)
            self._stack.installEventFilter(self._stack_filter)
        self._patch_tab_widget_setters()
        self.bar.currentChanged.connect(self._on_bar_current_changed)
        self._tabs.currentChanged.connect(self._on_tabs_current_changed)
        self.bar.tabMoved.connect(self._on_bar_moved)
        self._src.tabMoved.connect(self._on_src_moved)
        if follow_widget is not None:
            follow_widget.visibilityChanged.connect(
                lambda _v, w=follow_widget: self.toolbar.setVisible(not w.isHidden())
            )
            self.toolbar.setVisible(not follow_widget.isHidden())

    def _on_bar_current_changed(self, index):
        if self._busy or index < 0:
            return
        self._busy = True
        try:
            self._tabs.setCurrentIndex(index)
        finally:
            self._busy = False

    def _on_tabs_current_changed(self, index):
        if self._busy or index < 0:
            return
        self._busy = True
        try:
            self.bar.setCurrentIndex(index)
        finally:
            self._busy = False

    def _on_bar_moved(self, from_index, to_index):
        if self._busy:
            return
        self._busy = True
        try:
            self._src.moveTab(from_index, to_index)
        finally:
            self._busy = False

    def _on_src_moved(self, from_index, to_index):
        if self._busy:
            return
        self._busy = True
        try:
            self.bar.moveTab(from_index, to_index)
        finally:
            self._busy = False


def install_top_tab_bar(main_window, tab_widget, follow_widget=None):
    """Create the controller and keep it referenced on the main window."""
    controller = TopTabBarController(main_window, tab_widget, follow_widget)
    main_window.topTabBarController = controller
    main_window.topTabBar = controller.bar
    # Public helper for extensions (see extensions/_example_watermark.py).
    main_window.add_parameter_tab = controller.add_tab
    return controller
