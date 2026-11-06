from PySide6 import QtWidgets, QtGui, QtCore
from typing import TYPE_CHECKING
from app.ui.widgets.actions import video_control_actions

if TYPE_CHECKING:
    from app.ui.main_ui import MainWindow


def _item_belongs_to_scene(item, scene) -> bool:
    try:
        return item is not None and item.scene() is scene
    except RuntimeError:
        return False


def get_base_pixmap_item(main_window: "MainWindow"):
    """Return the live frame item."""
    scene = main_window.graphicsViewFrame.scene()
    pixmap_item = getattr(main_window, "_base_pixmap_item", None)
    if _item_belongs_to_scene(pixmap_item, scene):
        return pixmap_item

    for item in scene.items():
        if isinstance(item, QtWidgets.QGraphicsPixmapItem):
            main_window._base_pixmap_item = item
            return item
    return None


def update_graphics_view(
    main_window: "MainWindow",
    pixmap: QtGui.QPixmap,
    current_frame_number: int,
    reset_fit: bool = False,
    size_mode: str = "preserve_previous_pixmap_size",
):
    # Update the video seek slider and line edit safely to avoid recursive signal firing
    if main_window.videoSeekSlider.value() != current_frame_number:
        main_window.videoSeekSlider.blockSignals(True)
        main_window.videoSeekSlider.setValue(current_frame_number)
        main_window.videoSeekSlider.blockSignals(False)

    current_text = main_window.videoSeekLineEdit.text()
    if current_text != str(current_frame_number):
        main_window.videoSeekLineEdit.setText(str(current_frame_number))
    video_control_actions.update_video_time_line_edit(main_window, current_frame_number)

    scene = main_window.graphicsViewFrame.scene()
    pixmap_item = get_base_pixmap_item(main_window)

    # Resize the pixmap if necessary (e.g., face compare or mask compare mode)
    if pixmap_item and size_mode == "preserve_previous_pixmap_size":
        bounding_rect = pixmap_item.boundingRect()
        b_width = int(bounding_rect.width())
        b_height = int(bounding_rect.height())

        if b_width > pixmap.width() and b_height > pixmap.height():
            pixmap = pixmap.scaled(
                b_width,
                b_height,
                QtCore.Qt.AspectRatioMode.KeepAspectRatio,
                QtCore.Qt.TransformationMode.SmoothTransformation,
            )

    if pixmap_item:
        pixmap_item.setPixmap(pixmap)
    else:
        pixmap_item = QtWidgets.QGraphicsPixmapItem(pixmap)
        pixmap_item.setTransformationMode(
            QtCore.Qt.TransformationMode.SmoothTransformation
        )
        scene.addItem(pixmap_item)

    main_window._base_pixmap_item = pixmap_item

    scene_rect = pixmap_item.boundingRect()
    main_window.graphicsViewFrame.setSceneRect(scene_rect)

    if reset_fit:
        fit_image_to_view(main_window, pixmap_item, scene_rect)


def zoom_andfit_image_to_view_onchange(main_window: "MainWindow", new_transform):
    """Restore the previous transform (zoom and pan state) and update the view."""
    main_window.graphicsViewFrame.setTransform(new_transform, combine=False)


def fit_image_to_view(
    main_window: "MainWindow", pixmap_item: QtWidgets.QGraphicsPixmapItem, scene_rect
):
    """Reset the view and fit the image to the view, keeping the aspect ratio."""
    graphicsViewFrame = main_window.graphicsViewFrame
    graphicsViewFrame.resetTransform()
    graphicsViewFrame.setSceneRect(scene_rect)
    graphicsViewFrame.fitInView(pixmap_item, QtCore.Qt.AspectRatioMode.KeepAspectRatio)