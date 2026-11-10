from __future__ import annotations

from typing import TYPE_CHECKING

from PySide6 import QtCore, QtWidgets

from app.ui.widgets import sortable_widgets
from app.ui.widgets.actions import filter_actions

if TYPE_CHECKING:
    from app.ui.main_ui import MainWindow


def current_sort_needs_metadata(main_window: "MainWindow") -> bool:
    sort_mode = main_window.targetVideosSortComboBox.currentData()
    return sort_mode in {
        sortable_widgets.SortingMode.ImageDimension,
        sortable_widgets.SortingMode.ImagePixels,
        sortable_widgets.SortingMode.VideoLength,
    }


def sort_target_videos(main_window: "MainWindow", *args, **kwargs) -> None:
    list_widget = main_window.targetVideosList
    list_widget.setMinImageDimension(None)

    sort_order = (
        QtCore.Qt.SortOrder.DescendingOrder
        if main_window.sortTargetVideosDirectionButton.isChecked()
        else QtCore.Qt.SortOrder.AscendingOrder
    )
    list_widget.setSortingMode(main_window.targetVideosSortComboBox.currentData())

    filter_actions.filter_target_videos(main_window)
    list_widget.sortItems(sort_order)

    selected = main_window.selected_video_button
    selected_item = getattr(selected, "list_item", None) if selected else None
    if selected_item is not None:
        list_widget.scrollToItem(
            selected_item,
            QtWidgets.QAbstractItemView.ScrollHint.EnsureVisible,
        )


_THUMB_PX_PER_NOTCH = 8


class _ThumbnailZoomFilter(QtCore.QObject):
    def __init__(self, main_window: "MainWindow"):
        super().__init__(main_window)
        self._main_window = main_window
        self._size_f: float | None = None

    def eventFilter(self, watched, event):
        if not (
            event.type() == QtCore.QEvent.Type.Wheel
            and event.modifiers() & QtCore.Qt.KeyboardModifier.ControlModifier
        ):
            return False

        delta = event.angleDelta().y() or event.angleDelta().x()
        if delta:
            from app.ui.widgets.actions import list_view_actions

            mw = self._main_window
            current = list_view_actions._get_target_button_size(mw)[0]
            if self._size_f is None or round(self._size_f) != current:
                self._size_f = float(current)
            self._size_f = min(
                list_view_actions._TARGET_BUTTON_MAX,
                max(
                    list_view_actions._TARGET_BUTTON_MIN,
                    self._size_f + delta / 120.0 * _THUMB_PX_PER_NOTCH,
                ),
            )
            new_side = round(self._size_f)
            if new_side != current:
                list_view_actions.apply_target_thumbnail_size(mw, (new_side, new_side))
        event.accept()
        return True


def install_target_thumbnail_zoom(main_window: "MainWindow") -> None:
    if getattr(main_window, "_target_thumb_zoom_installed", False):
        return
    zoom_filter = _ThumbnailZoomFilter(main_window)
    list_widget = main_window.targetVideosList
    list_widget.installEventFilter(zoom_filter)
    list_widget.viewport().installEventFilter(zoom_filter)
    main_window._target_thumb_zoom_installed = True