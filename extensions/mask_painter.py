"""Mask Painter Extension — paint the swap mask by hand (SD Forge / inpainting style)."""
import cv2
import numpy as np
from PySide6 import QtWidgets, QtGui, QtCore


class MaskPainterExtension(QtCore.QObject):
    def __init__(self, main_window):
        super().__init__()
        self.main_window = main_window
        self.enabled = False
        self.mask = None            # uint8 (H,W): 255 = keep swap
        self.mask_frame = None
        self.brush_size = 60
        self.erase_mode = False
        self._painting = False
        self._last_mask_pos = None
        self._mask_item = None
        self._brush_item = None
        self._original_ref = None
        self._original_ref_frame = None

    # ---------- helpers ----------
    def _frame_number(self):
        return int(self.main_window.video_processor.current_frame_number)

    def _pixmap_item(self):
        item = getattr(self.main_window, "_base_pixmap_item", None)
        try:
            if item is not None and item.scene() is not None:
                return item
        except RuntimeError:
            return None
        return None

    def _mask_size(self):
        item = self._pixmap_item()
        if item is not None:
            r = item.boundingRect()
            return int(r.width()), int(r.height())
        vp = self.main_window.video_processor
        cf = vp.current_frame
        if isinstance(cf, np.ndarray) and cf.size > 0:
            return cf.shape[1], cf.shape[0]
        return None

    def _ensure_mask(self):
        size = self._mask_size()
        if size is None:
            return False
        w, h = size
        if self.mask is None or self.mask.shape != (h, w):
            self.mask = np.zeros((h, w), dtype=np.uint8)
            self.mask_frame = self._frame_number()
        return True

    def _capture_original_reference(self):
        vp = self.main_window.video_processor
        cache = getattr(vp, "_seek_cached_frame", None)
        if cache and cache[0] == vp.current_frame_number:
            self._original_ref = cache[1].copy()
            self._original_ref_frame = cache[0]
            return
        if vp.file_type == "image" and vp.media_path:
            try:
                from app.helpers import miscellaneous as mh
                frame = mh.read_image_file(vp.media_path)
                if frame is not None:
                    self._original_ref = frame
                    self._original_ref_frame = vp.current_frame_number
                    return
            except Exception:
                pass
        self._original_ref = None

    def _get_original_frame(self):
        vp = self.main_window.video_processor
        cache = getattr(vp, "_seek_cached_frame", None)
        if cache and cache[0] == vp.current_frame_number:
            return cache[1]
        if (self._original_ref is not None
                and self._original_ref_frame == vp.current_frame_number):
            return self._original_ref
        return None

    # ---------- coordinates ----------
    def _viewport_to_mask(self, viewport_pos):
        view = self.main_window.graphicsViewFrame
        scene_pos = view.mapToScene(viewport_pos)
        item = self._pixmap_item()
        if item is not None:
            local = item.mapFromScene(scene_pos)
            return int(local.x()), int(local.y())
        return int(scene_pos.x()), int(scene_pos.y())

    # ---------- painting ----------
    def _paint_segment(self, p1, p2, value):
        if not self._ensure_mask():
            return
        t = self.brush_size
        if p1 is None:
            cv2.circle(self.mask, p2, t // 2, value, -1)
        else:
            cv2.line(self.mask, p1, p2, value, t)
        self._update_overlay()

    def _update_overlay(self):
        if self._mask_item is None or self.mask is None:
            return
        h, w = self.mask.shape[:2]
        rgba = np.zeros((h, w, 4), dtype=np.uint8)
        rgba[..., 0] = 255
        rgba[..., 3] = (self.mask.astype(np.float32) * 0.45).astype(np.uint8)
        qimg = QtGui.QImage(
            rgba.data, w, h, 4 * w, QtGui.QImage.Format_RGBA8888
        ).copy()
        self._mask_item.setPixmap(QtGui.QPixmap.fromImage(qimg))

    def _install_overlay(self):
        item = self._pixmap_item()
        if item is None:
            return
        self._remove_overlay()
        self._mask_item = QtWidgets.QGraphicsPixmapItem(item)
        self._mask_item.setZValue(1000)
        self._mask_item.setAcceptedMouseButtons(QtCore.Qt.MouseButton.NoButton)

        self._brush_item = QtWidgets.QGraphicsEllipseItem(item)
        self._brush_item.setZValue(1001)
        pen = QtGui.QPen(QtGui.QColor(255, 255, 255, 200))
        pen.setWidth(1)
        self._brush_item.setPen(pen)
        self._brush_item.setAcceptedMouseButtons(QtCore.Qt.MouseButton.NoButton)
        self._update_overlay()

    def _remove_overlay(self):
        for attr in ("_mask_item", "_brush_item"):
            it = getattr(self, attr, None)
            if it is not None:
                try:
                    sc = it.scene()
                    if sc is not None:
                        sc.removeItem(it)
                except RuntimeError:
                    pass
                setattr(self, attr, None)

    def _update_brush_preview(self, mask_pos):
        if self._brush_item is None:
            return
        r = self.brush_size / 2.0
        self._brush_item.setRect(
            mask_pos[0] - r, mask_pos[1] - r, self.brush_size, self.brush_size
        )

    # ---------- event filter ----------
    def eventFilter(self, watched, event):
        if not self.enabled:
            return False
        view = self.main_window.graphicsViewFrame
        if watched is not view.viewport():
            return False
        et = event.type()
        shift = bool(event.modifiers() & QtCore.Qt.ShiftModifier) \
            if hasattr(event, "modifiers") else False
        value = 0 if (self.erase_mode or shift) else 255

        if et == QtCore.QEvent.Type.MouseButtonPress:
            if event.button() == QtCore.Qt.LeftButton:
                self._ensure_mask()
                self._painting = True
                p = self._viewport_to_mask(event.pos())
                self._last_mask_pos = p
                self._paint_segment(None, p, value)
                return True
        elif et == QtCore.QEvent.Type.MouseMove:
            p = self._viewport_to_mask(event.pos())
            self._update_brush_preview(p)
            if self._painting:
                self._paint_segment(self._last_mask_pos, p, value)
                self._last_mask_pos = p
                return True
        elif et == QtCore.QEvent.Type.MouseButtonRelease:
            if event.button() == QtCore.Qt.LeftButton and self._painting:
                self._painting = False
                self._last_mask_pos = None
                return True
        return False

    # ---------- pipeline hook ----------
    def on_frame_processed(self, frame):
        if not self.enabled or self.mask is None:
            return
        if self.mask_frame != self._frame_number():
            return
        original = self._get_original_frame()
        if original is None:
            return

        h, w = frame.shape[:2]
        if original.shape[:2] != (h, w):
            original = cv2.resize(original, (w, h))
        m = self.mask
        if m.shape != (h, w):
            m = cv2.resize(m, (w, h), interpolation=cv2.INTER_NEAREST)

        alpha = (m.astype(np.float32) / 255.0)[..., None]
        # Both frame and original are already in the same colorspace (BGR).
        blended = (frame.astype(np.float32) * alpha
                   + original.astype(np.float32) * (1.0 - alpha))
        frame[:] = np.clip(blended, 0, 255).astype(np.uint8)

    # ---------- public controls ----------
    def set_enabled(self, checked):
        self.enabled = bool(checked)
        view = self.main_window.graphicsViewFrame
        vp = view.viewport()
        if self.enabled:
            self._capture_original_reference()
            self._ensure_mask()
            self._install_overlay()
            vp.setMouseTracking(True)
            vp.setCursor(QtCore.Qt.CursorShape.CrossCursor)
            vp.installEventFilter(self)
        else:
            try:
                vp.removeEventFilter(self)
            except RuntimeError:
                pass
            vp.setMouseTracking(False)
            vp.unsetCursor()
            self._remove_overlay()
            self._painting = False
            self._last_mask_pos = None

    def clear_mask(self):
        if self.mask is not None:
            self.mask[:] = 0
            self._update_overlay()

    def invert_mask(self):
        if self.mask is not None:
            self.mask[:] = 255 - self.mask
            self._update_overlay()

    def refresh_reference(self):
        self._capture_original_reference()
        self._ensure_mask()
        self._install_overlay()

    def on_app_close(self):
        try:
            self.set_enabled(False)
        except Exception:
            pass


def setup(main_window):
    ext = MaskPainterExtension(main_window)

    tab = QtWidgets.QWidget()
    layout = QtWidgets.QVBoxLayout(tab)

    info = QtWidgets.QLabel(
        "<b>Paint the swap mask manually.</b><br><br>"
        "Left-drag on the preview to paint. Red overlay = pixels that keep "
        "the swapped face; unpainted pixels are restored to the original frame.<br><br>"
        "Hold <b>Shift</b> (or enable Erase mode) to remove paint.<br>"
        "The mask applies to the frame it was painted on. Seek, paint, then "
        "press <b>Swap</b> or frame-step to reprocess and see the result."
    )
    info.setWordWrap(True)
    layout.addWidget(info)

    enable_toggle = QtWidgets.QCheckBox("Enable Mask Painter")
    layout.addWidget(enable_toggle)

    erase_toggle = QtWidgets.QCheckBox("Erase Mode")
    erase_toggle.toggled.connect(lambda c: setattr(ext, "erase_mode", c))
    layout.addWidget(erase_toggle)

    refresh_btn = QtWidgets.QPushButton("Refresh Reference Frame")
    refresh_btn.clicked.connect(ext.refresh_reference)
    layout.addWidget(refresh_btn)

    brush_row = QtWidgets.QHBoxLayout()
    brush_row.addWidget(QtWidgets.QLabel("Brush:"))
    brush_slider = QtWidgets.QSlider(QtCore.Qt.Orientation.Horizontal)
    brush_slider.setRange(5, 300)
    brush_slider.setValue(60)
    brush_row.addWidget(brush_slider, 1)
    brush_val = QtWidgets.QLabel("60")
    brush_val.setFixedWidth(35)
    brush_row.addWidget(brush_val)
    layout.addLayout(brush_row)

    def on_brush(v):
        ext.brush_size = int(v)
        brush_val.setText(str(int(v)))
        if ext._brush_item is not None:
            r = ext.brush_size / 2.0
            c = ext._brush_item.rect().center()
            ext._brush_item.setRect(
                c.x() - r, c.y() - r, ext.brush_size, ext.brush_size
            )
    brush_slider.valueChanged.connect(on_brush)

    btn_row = QtWidgets.QHBoxLayout()
    clear_btn = QtWidgets.QPushButton("Clear Mask")
    clear_btn.clicked.connect(ext.clear_mask)
    btn_row.addWidget(clear_btn)
    invert_btn = QtWidgets.QPushButton("Invert Mask")
    invert_btn.clicked.connect(ext.invert_mask)
    btn_row.addWidget(invert_btn)
    layout.addLayout(btn_row)

    layout.addStretch()

    enable_toggle.toggled.connect(ext.set_enabled)

    main_window.tabWidget.addTab(tab, "Mask Painter")
    return ext