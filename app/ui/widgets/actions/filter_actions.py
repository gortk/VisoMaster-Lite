from typing import TYPE_CHECKING

from PySide6 import QtCore, QtWidgets
from shiboken6 import isValid

if TYPE_CHECKING:
    from app.ui.main_ui import MainWindow


def filter_target_videos(main_window: "MainWindow", *args):
    if getattr(main_window, "_closing", False):
        return
    main_window.target_videos_filter_worker.stop_thread()

    search_box = getattr(main_window, "targetVideosSearchBox", None)
    search_text = search_box.text().lower() if search_box else ""

    if not search_text:
        for i in range(main_window.targetVideosList.count()):
            main_window.targetVideosList.item(i).setHidden(False)
        return

    items_snapshot = []
    for i in range(main_window.targetVideosList.count()):
        item = main_window.targetVideosList.item(i)
        item_widget = main_window.targetVideosList.itemWidget(item)
        if item_widget is None:
            continue
        items_snapshot.append((i, str(item_widget.media_path), item_widget.file_type, 0, 0))

    worker = main_window.target_videos_filter_worker
    worker.include_file_types = ["image", "video"]
    worker.min_image_size = (0, 0)
    worker.search_text = search_text
    worker.items_snapshot = items_snapshot
    worker.start()


def filter_input_faces(main_window: "MainWindow", *args):
    if getattr(main_window, "_closing", False):
        return
    main_window.input_faces_filter_worker.stop_thread()

    # Capture all Qt widget data in the main thread before starting the worker
    search_text = main_window.inputFacesSearchBox.text().lower()

    items_snapshot = []
    for i in range(main_window.inputFacesList.count()):
        item = main_window.inputFacesList.item(i)
        item_widget = main_window.inputFacesList.itemWidget(item)
        if item_widget is not None:
            items_snapshot.append((i, item_widget.media_path))

    worker = main_window.input_faces_filter_worker
    worker.search_text = search_text
    worker.items_snapshot = items_snapshot
    worker.start()


def ensure_embedding_filter_worker(main_window):
    """Create a filter lazily after clear/reset, or reuse its unchanged list."""
    worker = getattr(main_window, "merged_embeddings_filter_worker", None)
    if worker is None:
        from app.ui.widgets import ui_workers

        worker = ui_workers.FilterWorker(main_window, filter_list="merged_embeddings")
        main_window.merged_embeddings_filter_worker = worker
    return worker


def filter_merged_embeddings(main_window: "MainWindow", *args):
    if getattr(main_window, "_closing", False) or getattr(main_window, "_embedding_tabs_rebuilding", False):
        return
    worker = ensure_embedding_filter_worker(main_window)
    worker.stop_thread()

    # Capture all Qt widget data in the main thread before starting the worker
    search_text = main_window.inputEmbeddingsSearchBox.text().lower()

    items_snapshot = []
    for i in range(main_window.inputEmbeddingsList.count()):
        item = main_window.inputEmbeddingsList.item(i)
        item_widget = main_window.inputEmbeddingsList.itemWidget(item)
        if item_widget is not None:
            items_snapshot.append((i, item_widget.embedding_name))

    worker.search_text = search_text
    worker.items_snapshot = items_snapshot
    worker.start()


def update_filtered_list(
    main_window: "MainWindow",
    filter_list_widget: QtWidgets.QListWidget,
    visible_indices: list,
    snapshot_size: int = 0,
    *,
    is_current=None,
):
    if not isValid(filter_list_widget) or (is_current is not None and not is_current()):
        return
    # Defer hide/show work to the next event loop tick so pending paint events can
    # complete naturally without pumping the queue inside this function.
    sequence = getattr(filter_list_widget, "_pending_filter_sequence", 0) + 1
    filter_list_widget._pending_filter_sequence = sequence

    def apply_update():
        if not isValid(filter_list_widget) or (is_current is not None and not is_current()):
            return
        if getattr(filter_list_widget, "_pending_filter_sequence", None) != sequence:
            return

        filter_list_widget.setUpdatesEnabled(False)
        try:
            # Only manage items that existed at snapshot time; items added after the
            # snapshot was captured (index >= snapshot_size) are left visible so they
            # are not accidentally hidden by a stale filter result.
            limit = snapshot_size if snapshot_size > 0 else filter_list_widget.count()
            for i in range(min(limit, filter_list_widget.count())):
                filter_list_widget.item(i).setHidden(True)

            # Show only the items in the visible_indices list
            for i in visible_indices:
                if i < filter_list_widget.count():
                    filter_list_widget.item(i).setHidden(False)
        finally:
            filter_list_widget.setUpdatesEnabled(True)
            filter_list_widget.viewport().update()

    QtCore.QTimer.singleShot(0, apply_update)
