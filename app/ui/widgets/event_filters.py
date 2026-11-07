from typing import TYPE_CHECKING
from functools import partial

from PySide6 import QtWidgets, QtGui, QtCore
from app.ui.widgets.actions import list_view_actions
from app.ui.widgets.actions import video_control_actions
from app.ui.widgets import ui_workers
import app.helpers.miscellaneous as misc_helpers

if TYPE_CHECKING:
    from app.ui.main_ui import MainWindow

_HANDLED_EVENTS = frozenset({
    QtCore.QEvent.Type.MouseButtonPress,
    QtCore.QEvent.Type.DragEnter,
    QtCore.QEvent.Type.Drop,
})

class GraphicsViewEventFilter(QtCore.QObject):
    def __init__(self, main_window: "MainWindow", parent=None):
        super().__init__(parent)
        self.main_window = main_window

    def eventFilter(self, graphics_object: QtWidgets.QGraphicsView, event):
        if event.type() == QtCore.QEvent.Type.MouseButtonPress:
            if event.button() == QtCore.Qt.MouseButton.LeftButton:
                # Do not attempt to play if the loaded media is an image
                if getattr(self.main_window.video_processor, "file_type", None) == "image":
                    return False
                    
                self.main_window.buttonMediaPlay.click()
                return True  # Mark the event as handled
        return False  # Pass the event to the original handler

class videoSeekSliderLineEditEventFilter(QtCore.QObject):
    def __init__(self, main_window: "MainWindow", parent=None):
        super().__init__(parent)
        self.main_window = main_window

    def eventFilter(self, line_edit: QtWidgets.QLineEdit, event):
        if event.type() == QtCore.QEvent.KeyPress:
            # Check if the pressed key is Enter/Return
            if event.key() in (QtCore.Qt.Key_Enter, QtCore.Qt.Key_Return):
                if video_control_actions.is_issue_scan_active(self.main_window):
                    return True
                text = line_edit.text().strip()
                if not text:
                    new_value = int(self.main_window.videoSeekSlider.value())
                else:
                    try:
                        new_value = int(text)
                    except ValueError:
                        new_value = int(self.main_window.videoSeekSlider.value())
                    new_value = max(
                        0,
                        min(new_value, self.main_window.video_processor.max_frame_number),
                    )
                line_edit.setText(str(new_value))
                self.main_window.videoSeekSlider.setValue(new_value)
                self.main_window.video_processor.process_current_frame()
                return True
        return False


class VideoSeekSliderEventFilter(QtCore.QObject):
    def __init__(self, main_window: "MainWindow", parent=None):
        super().__init__(parent)
        self.main_window = main_window

    def eventFilter(self, slider, event):
        if event.type() == QtCore.QEvent.Type.KeyPress:
            if event.key() == QtCore.Qt.Key_Right:
                # Force strictly 1 frame advance through our controlled pipeline
                video_control_actions.advance_video_slider_by_n_frames(
                    self.main_window, 1
                )
                return True  # Stop QT from applying default values

            elif event.key() == QtCore.Qt.Key_Left:
                # Force strictly 1 frame rewind through our controlled pipeline
                video_control_actions.rewind_video_slider_by_n_frames(
                    self.main_window, 1
                )
                return True  # Stop QT from applying default values

        elif event.type() == QtCore.QEvent.Type.Wheel:
            # Intercept mousewheel to use FrameSkipStepSlider logic
            delta = event.angleDelta().y()
            if delta > 0:
                # If wheel up (Advance)
                video_control_actions.advance_video_slider_by_n_frames(self.main_window)
            elif delta < 0:
                # If wheel down (Rewind)
                video_control_actions.rewind_video_slider_by_n_frames(self.main_window)

            return True  # Stop QT from applying default values

        # For other events, use the default behavior
        return super().eventFilter(slider, event)


class ListWidgetEventFilter(QtCore.QObject):
    def __init__(self, main_window: "MainWindow", parent=None):
        super().__init__(parent)
        self.main_window = main_window
        self._pending_loads: dict[str, list[list[str]]] = {}

    def _spec_for(self, list_widget):
        if list_widget in (self.main_window.targetVideosList,
                           self.main_window.targetVideosList.viewport()):
            return {
                "loader_attr": "video_loader_worker",
                "loader_cls": ui_workers.TargetMediaLoaderWorker,
                "thumb_cb": list_view_actions.add_media_thumbnail_to_target_videos_list,
                "is_valid": lambda p: (
                    misc_helpers.is_image_file(p) or misc_helpers.is_video_file(p)
                ),
                "block_msg": "change target media",
            }
        if list_widget in (self.main_window.inputFacesList,
                           self.main_window.inputFacesList.viewport()):
            return {
                "loader_attr": "input_faces_loader_worker",
                "loader_cls": ui_workers.InputFacesLoaderWorker,
                "thumb_cb": list_view_actions.add_media_thumbnail_to_source_faces_list,
                "is_valid": misc_helpers.is_image_file,
                "block_msg": "load input faces",
            }
        return None

    def _drain_loader(self, spec):
        attr = spec["loader_attr"]
        current = getattr(self.main_window, attr, None)
        if current is not None and current.isRunning():
            return
        queue = self._pending_loads.get(attr)
        if not queue:
            return
        files = queue.pop(0)
        worker = spec["loader_cls"](
            main_window=self.main_window, folder_name=False, files_list=files,
        )
        worker.thumbnail_ready.connect(partial(spec["thumb_cb"], self.main_window))
        worker.finished.connect(lambda s=spec: self._drain_loader(s))
        setattr(self.main_window, attr, worker)
        worker.start()

    def _drop_media(self, event, spec):
        if video_control_actions.block_if_issue_scan_active(
            self.main_window, spec["block_msg"]
        ):
            event.ignore()
            return True
        paths = [
            u.toLocalFile()
            for u in event.mimeData().urls()
            if spec["is_valid"](u.toLocalFile())
        ]
        if paths:
            self._pending_loads.setdefault(spec["loader_attr"], []).append(paths)
            self._drain_loader(spec)
        event.acceptProposedAction()
        return True


    def eventFilter(
        self,
        list_widget: QtWidgets.QListWidget,
        event: QtCore.QEvent | QtGui.QDropEvent | QtGui.QMouseEvent,
    ):
        if event.type() not in _HANDLED_EVENTS:
            return False
        try:
            target_videos_list = self.main_window.targetVideosList
            target_videos_viewport = target_videos_list.viewport()
            input_faces_list = self.main_window.inputFacesList
            input_faces_viewport = input_faces_list.viewport()
        except RuntimeError:
            return False

        if list_widget == target_videos_list or list_widget == target_videos_viewport:
            if event.type() == QtCore.QEvent.Type.MouseButtonPress:
                if (
                    event.button() == QtCore.Qt.MouseButton.LeftButton
                    and not self.main_window.target_videos
                ):
                    list_view_actions.select_target_medias(self.main_window, "folder")

            elif event.type() == QtCore.QEvent.Type.DragEnter:
                if event.mimeData().hasUrls():
                    if any(
                        misc_helpers.is_image_file(u.toLocalFile())
                        or misc_helpers.is_video_file(u.toLocalFile())
                        for u in event.mimeData().urls()
                    ):
                        event.acceptProposedAction()
                        return True
                return False
            # Handle the drop event
            elif event.type() == QtCore.QEvent.Type.Drop:
                if event.mimeData().hasUrls():
                    return self._drop_media(event, self._spec_for(list_widget))

        elif list_widget == input_faces_list or list_widget == input_faces_viewport:
            if event.type() == QtCore.QEvent.Type.MouseButtonPress:
                if (
                    event.button() == QtCore.Qt.MouseButton.LeftButton
                    and not self.main_window.input_faces
                ):
                    list_view_actions.select_input_face_images(
                        self.main_window, "folder"
                    )

            elif event.type() == QtCore.QEvent.Type.DragEnter:
                if event.mimeData().hasUrls():
                    if any(
                        misc_helpers.is_image_file(u.toLocalFile())
                        for u in event.mimeData().urls()
                    ):
                        event.acceptProposedAction()
                        return True
                return False
            # Handle the drop event
            elif event.type() == QtCore.QEvent.Type.Drop:
                if event.mimeData().hasUrls():
                    return self._drop_media(event, self._spec_for(list_widget))
        return super().eventFilter(list_widget, event)
