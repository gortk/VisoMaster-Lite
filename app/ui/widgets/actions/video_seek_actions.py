import cv2
import math
from typing import TYPE_CHECKING
from PySide6 import QtWidgets, QtCore, QtGui
import app.helpers.miscellaneous as misc_helpers
from app.helpers.miscellaneous import get_video_rotation
from app.ui.widgets.actions.video_control_actions import _get_marker_data_for_position

if TYPE_CHECKING:
    from app.ui.main_ui import MainWindow


class TimelineSeekSlider(QtWidgets.QSlider):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.main_window: "MainWindow" = None

        # 1. Initialize all state markers natively
        self.markers = set()
        self.markers_sorted = []
        self.issue_markers = set()
        self.issue_markers_sorted = []
        self.dropped_markers = set()
        self.dropped_markers_sorted = []

        self.setTickPosition(QtWidgets.QSlider.TickPosition.TicksBelow)

        # 2. Provisional State Tracking
        self._last_provisional_state = False
        self._last_polled_position = -1
        self._cached_baseline_params = {}
        self._cached_baseline_ctrl = {}
        self._connected_widgets = set()

        # P7-PERF: cache QPen / QColor / QFont so paintEvent (fired ~30x/sec
        # during playback) does not allocate them on every redraw.
        self._pen_gray = QtGui.QPen(QtGui.QColor("gray"), 3)
        self._pen_segment_a = QtGui.QPen(QtGui.QColor("#7e57c2"), 3)
        self._pen_segment_b = QtGui.QPen(QtGui.QColor("#42a5f5"), 3)
        self._pen_provisional = QtGui.QPen(QtGui.QColor("#e5c07b"), 3)
        self._pen_issue = QtGui.QPen(QtGui.QColor("#ff9800"), 3)
        self._pen_issue.setCapStyle(QtCore.Qt.PenCapStyle.SquareCap)
        self._pen_dropped = QtGui.QPen(QtGui.QColor("#e8483c"), 3)
        self._pen_bracket_start = QtGui.QPen(QtGui.QColor("#4CAF50"), 1)
        self._pen_bracket_end = QtGui.QPen(QtGui.QColor("#e8483c"), 1)
        self._pen_playhead_normal = QtGui.QPen(QtGui.QColor("white"), 1)
        self._pen_playhead_provisional = QtGui.QPen(QtGui.QColor("#e5c07b"), 1)
        self._brush_playhead_normal = QtGui.QBrush(QtGui.QColor("white"))
        self._brush_playhead_provisional = QtGui.QBrush(QtGui.QColor("#e5c07b"))
        self._bracket_font = QtGui.QFont("Arial", 16, QtGui.QFont.Bold)
        self._pen_segment = QtGui.QPen(QtGui.QColor("#9c27b0"), 1)
        self._brush_segment = QtGui.QBrush(QtGui.QColor(156, 39, 176, 90))
        self._pen_segment_recording = QtGui.QPen(QtGui.QColor("#4caf50"), 1)
        self._brush_segment_recording = QtGui.QBrush(QtGui.QColor(76, 175, 80, 90))
        self._pen_segment_exclude = QtGui.QPen(QtGui.QColor("#e8483c"), 1)
        self._brush_segment_exclude = QtGui.QBrush(QtGui.QColor(232, 72, 60, 90))
        self._pen_segment_drag = QtGui.QPen(QtGui.QColor("#9c27b0"), 1, QtCore.Qt.PenStyle.DashLine)
        self._brush_segment_drag = QtGui.QBrush(QtGui.QColor(156, 39, 176, 60))
        self._drag_start_frame = None
        self._drag_current_frame = None
        self._hover_segment_index = None
        self._resize_segment_index = None
        self._resize_edge = None
        self._move_segment_index = None
        self._move_orig_start = None
        self._move_orig_end = None
        self._move_drag_start_frame = None
        self._resize_border_px = 5
        # Playhead grab zone (px on each side of the playhead). The playhead always
        # wins over segment borders / segment creation so it can never be blocked.
        self._playhead_grab_px = 7
        self._playhead_dragging = False
        self.setMouseTracking(True)

    def setup(self, main_window: "MainWindow"):
        """Injects the main window dependency after UI creation."""
        self.main_window = main_window
        self.valueChanged.connect(self.trigger_check)
        self.valueChanged.connect(self._auto_scroll_viewport)
        QtCore.QTimer.singleShot(500, self.bind_provisional_events)
        self.setContextMenuPolicy(QtCore.Qt.ContextMenuPolicy.CustomContextMenu)
        self.customContextMenuRequested.connect(self._show_segment_context_menu)

    def setValue(self, value: int):
        """Override native setValue to catch programmatic updates even when signals are blocked."""
        super().setValue(value)
        # Manually trigger the auto-scroll check
        self._auto_scroll_viewport(value)

    def _auto_scroll_viewport(self, value: int):
        """Automatically page-scrolls the timeline if the playhead exits the viewport."""
        if not getattr(self, "main_window", None):
            return

        scroll_area = getattr(self.main_window, "timelineScrollArea", None)
        if not scroll_area:
            return

        # Only auto-scroll if we are actually zoomed in
        if self.width() <= scroll_area.viewport().width():
            return

        max_val = max(1, self.maximum())
        # Calculate the exact pixel coordinate of the playhead
        playhead_pixel_x = int(self.width() * (value / max_val))

        scrollbar = scroll_area.horizontalScrollBar()
        scroll_x = scrollbar.value()
        viewport_width = scroll_area.viewport().width()

        # Add a 10-pixel padding so it doesn't hug the absolute mathematical edge
        margin = 10

        if playhead_pixel_x > scroll_x + viewport_width:
            # Playback passed the right edge. Jump the scrollbar so the playhead is at the left edge.
            scrollbar.setValue(max(0, playhead_pixel_x - margin))

        elif playhead_pixel_x < scroll_x:
            # User scrubbed backwards past the left edge. Jump the scrollbar so the playhead is at the right edge.
            scrollbar.setValue(max(0, playhead_pixel_x - viewport_width + margin))

    # --- Marker Management Methods ---
    def add_marker_and_paint(self, value=None):
        if value is None or isinstance(value, bool):
            value = self.value()
        if self.minimum() <= value <= self.maximum() and value not in self.markers:
            self.markers.add(value)
            if value not in self.markers_sorted:
                self.markers_sorted.append(value)
                self.markers_sorted.sort()
            self._last_polled_position = -1
            self.update()
            self.trigger_check()

    def remove_marker_and_paint(self, value=None):
        if value is None or isinstance(value, bool):
            value = self.value()
        if value in self.markers:
            self.markers.remove(value)
            if value in self.markers_sorted:
                self.markers_sorted.remove(value)
            self._last_polled_position = -1
            self.update()
            self.trigger_check()

    def _add_sorted_marker(
        self, marker_set: set[int], marker_list: list[int], value: int
    ) -> bool:
        if value not in marker_set:
            marker_set.add(value)
            marker_list.append(value)
            marker_list.sort()
            return True
        return False

    def _remove_sorted_marker(
        self, marker_set: set[int], marker_list: list[int], value: int
    ) -> bool:
        if value in marker_set:
            marker_set.remove(value)
            if value in marker_list:
                marker_list.remove(value)
            return True
        return False

    def add_issue_marker_and_paint(self, value=None):
        if value is None or isinstance(value, bool):
            value = self.value()
        if self.minimum() <= value <= self.maximum() and self._add_sorted_marker(
            self.issue_markers, self.issue_markers_sorted, value
        ):
            self.update()

    def remove_issue_marker_and_paint(self, value=None):
        if value is None or isinstance(value, bool):
            value = self.value()
        if self._remove_sorted_marker(
            self.issue_markers, self.issue_markers_sorted, value
        ):
            self.update()

    def add_dropped_marker_and_paint(self, value=None):
        if value is None or isinstance(value, bool):
            value = self.value()
        if self.minimum() <= value <= self.maximum() and self._add_sorted_marker(
            self.dropped_markers, self.dropped_markers_sorted, value
        ):
            self.update()

    def remove_dropped_marker_and_paint(self, value=None):
        if value is None or isinstance(value, bool):
            value = self.value()
        if self._remove_sorted_marker(
            self.dropped_markers, self.dropped_markers_sorted, value
        ):
            self.update()

    # --- Provisional State Logic ---
    def is_state_provisional(self) -> bool:
        if not self.main_window:
            return False
        current_position = self.value()

        if not self.markers_sorted or current_position < self.markers_sorted[0]:
            return True

        if current_position != self._last_polled_position:
            self._last_polled_position = current_position
            marker_data = _get_marker_data_for_position(
                self.main_window, current_position
            )

            if not marker_data:
                return True

            self._cached_baseline_params = marker_data.get("parameters", {})
            self._cached_baseline_ctrl = marker_data.get("control", {})

        baseline_params = self._cached_baseline_params
        baseline_ctrl = self._cached_baseline_ctrl

        if self.main_window.parameters != baseline_params:
            return True

        protected_keys = {
            "TrackMarkersToggle",
            "OutputMediaFolder",
            "OutputToTargetLocationToggle",
            "PreserveOutputDirectoryStructureToggle",
            "ClusterOutputBySourceToggle",
        }

        curr_ctrl = self.main_window.control
        for k, v in curr_ctrl.items():
            if k in protected_keys:
                continue
            if k not in baseline_ctrl or baseline_ctrl[k] != v:
                return True
        for k in baseline_ctrl:
            if k in protected_keys:
                continue
            if k not in curr_ctrl:
                return True

        return False

    def check_provisional_changes(self):
        current_state = self.is_state_provisional()
        if current_state != self._last_provisional_state:
            self._last_provisional_state = current_state
            self.update()

    def trigger_check(self, *args, **kwargs):
        QtCore.QTimer.singleShot(150, self.check_provisional_changes)

    def _segment_creation_active(self) -> bool:
        return bool(getattr(self.main_window, "segment_creation_mode", False))

    def _frame_for_x(self, x: float) -> int:
        style = self.style()
        opt = QtWidgets.QStyleOptionSlider()
        self.initStyleOption(opt)
        groove = style.subControlRect(
            QtWidgets.QStyle.ComplexControl.CC_Slider,
            opt,
            QtWidgets.QStyle.SubControl.SC_SliderGroove,
        )
        if groove.width() <= 0:
            return self.minimum()
        ratio = (x - groove.left()) / groove.width()
        ratio = min(max(ratio, 0.0), 1.0)
        return int(self.minimum() + ratio * (self.maximum() - self.minimum()))

    def _x_for_frame(self, frame: int) -> float:
        style = self.style()
        opt = QtWidgets.QStyleOptionSlider()
        self.initStyleOption(opt)
        groove = style.subControlRect(
            QtWidgets.QStyle.ComplexControl.CC_Slider,
            opt,
            QtWidgets.QStyle.SubControl.SC_SliderGroove,
        )
        if groove.width() <= 0:
            return float(groove.left())
        span = max(1, self.maximum() - self.minimum())
        ratio = (frame - self.minimum()) / span
        return float(groove.left() + ratio * groove.width())

    def _is_on_playhead(self, x: float) -> bool:
        if self.maximum() <= self.minimum():
            return False
        return abs(x - self._x_for_frame(self.value())) <= self._playhead_grab_px

    def _border_at_x(self, x: float):
        if self.main_window is None:
            return None
        segments = getattr(self.main_window, "segments", [])
        if not segments:
            return None
        thr = self._resize_border_px
        for i, seg in enumerate(segments):
            if abs(x - self._x_for_frame(int(seg["start"]))) <= thr:
                return (i, "start")
            if abs(x - self._x_for_frame(int(seg["end"]))) <= thr:
                return (i, "end")
        return None

    def _segment_at_x(self, x: float):
        if self.main_window is None:
            return None
        segments = getattr(self.main_window, "segments", [])
        if not segments:
            return None
        frame = self._frame_for_x(x)
        for i, seg in enumerate(segments):
            if seg["start"] <= frame <= seg["end"]:
                return i
        return None

    def leaveEvent(self, event):
        if self._hover_segment_index is not None:
            self._hover_segment_index = None
            self.update()
        super().leaveEvent(event)

    def _draw_frame_flag(self, painter, x, text, anchor_left):
        font = QtGui.QFont("Arial", 8)
        font.setBold(True)
        fm = QtGui.QFontMetrics(font)
        text_w = fm.horizontalAdvance(text)
        text_h = fm.height()
        w = text_w + 10
        h = text_h + 4
        y = self.height() - h - 2
        if y < 0:
            y = 0
        if anchor_left:
            rx = int(x) + 2
        else:
            rx = int(x) - w - 2
        if rx < 0:
            rx = 0
        if rx + w > self.width():
            rx = max(0, self.width() - w)
        rect = QtCore.QRect(rx, y, w, h)
        painter.setFont(font)
        painter.setPen(QtGui.QPen(QtGui.QColor("#5a5a5a"), 1))
        painter.setBrush(QtGui.QBrush(QtGui.QColor(44, 44, 44, 240)))
        painter.drawRoundedRect(rect, 3, 3)
        painter.setPen(QtGui.QColor("#e0e0e0"))
        painter.drawText(rect, QtCore.Qt.AlignmentFlag.AlignCenter, text)

    def mousePressEvent(self, event):
        # Playhead has absolute priority over segments (resize / move / create).
        if (
            event.button() == QtCore.Qt.MouseButton.LeftButton
            and self.main_window is not None
            and self.maximum() > self.minimum()
            and self._is_on_playhead(event.position().x())
        ):
            self._playhead_dragging = True
            self._hover_segment_index = None
            self.setSliderDown(True)
            self.setValue(self._frame_for_x(event.position().x()))
            self.unsetCursor()
            self.update()
            event.accept()
            return
        if self.main_window is not None and self.maximum() > self.minimum():
            if event.button() == QtCore.Qt.MouseButton.MiddleButton:
                idx = self._segment_at_x(event.position().x())
                if idx is not None:
                    segs = self.main_window.segments
                    self._move_segment_index = idx
                    self._move_orig_start = int(segs[idx]["start"])
                    self._move_orig_end = int(segs[idx]["end"])
                    self._move_drag_start_frame = self._frame_for_x(event.position().x())
                    self.setCursor(QtCore.Qt.CursorShape.ClosedHandCursor)
                    event.accept()
                    return
            elif event.button() == QtCore.Qt.MouseButton.LeftButton:
                hit = self._border_at_x(event.position().x())
                if hit is not None:
                    self._resize_segment_index, self._resize_edge = hit
                    self.setCursor(QtCore.Qt.CursorShape.SizeHorCursor)
                    event.accept()
                    return
        if self._segment_creation_active() and event.button() == QtCore.Qt.MouseButton.LeftButton:
            frame = self._frame_for_x(event.position().x())
            self._drag_start_frame = frame
            self._drag_current_frame = frame
            self.update()
            event.accept()
            return
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event):
        if self._playhead_dragging:
            self.setValue(self._frame_for_x(event.position().x()))
            event.accept()
            return
        if self._segment_creation_active() and self._drag_start_frame is not None:
            self._drag_current_frame = self._frame_for_x(event.position().x())
            self.update()
            event.accept()
            return
        if self._move_segment_index is not None and self.main_window is not None:
            curr = self._frame_for_x(event.position().x())
            delta = curr - self._move_drag_start_frame
            segs = self.main_window.segments
            if 0 <= self._move_segment_index < len(segs):
                seg = segs[self._move_segment_index]
                seg_len = self._move_orig_end - self._move_orig_start
                new_start = self._move_orig_start + delta
                if new_start < 0:
                    new_start = 0
                if new_start + seg_len > self.maximum():
                    new_start = self.maximum() - seg_len
                seg["start"] = new_start
                seg["end"] = new_start + seg_len
                self.update()
            event.accept()
            return
        if self._resize_segment_index is not None and self.main_window is not None:
            curr = self._frame_for_x(event.position().x())
            segs = self.main_window.segments
            if 0 <= self._resize_segment_index < len(segs):
                seg = segs[self._resize_segment_index]
                if self._resize_edge == "start":
                    seg["start"] = max(0, min(curr, int(seg["end"]) - 1))
                else:
                    seg["end"] = min(self.maximum(), max(curr, int(seg["start"]) + 1))
                self.update()
            event.accept()
            return
        over_playhead = self._is_on_playhead(event.position().x())
        if self.main_window is not None and self.maximum() > self.minimum():
            hit = None if over_playhead else self._border_at_x(event.position().x())
            if hit is not None:
                self.setCursor(QtCore.Qt.CursorShape.SizeHorCursor)
            else:
                self.unsetCursor()
        if self.main_window is not None:
            new_hover = (
                None if over_playhead else self._segment_at_x(event.position().x())
            )
            if new_hover != self._hover_segment_index:
                self._hover_segment_index = new_hover
                self.update()
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event):
        if self._playhead_dragging:
            self._playhead_dragging = False
            self.setSliderDown(False)
            self.update()
            event.accept()
            return
        if self._move_segment_index is not None:
            self._move_segment_index = None
            self._move_orig_start = None
            self._move_orig_end = None
            self._move_drag_start_frame = None
            self._drag_start_frame = None
            self._drag_current_frame = None
            self.unsetCursor()
            self.update()
            event.accept()
            return
        if self._resize_segment_index is not None:
            self._resize_segment_index = None
            self._resize_edge = None
            self._drag_start_frame = None
            self._drag_current_frame = None
            self.unsetCursor()
            self.update()
            event.accept()
            return
        if self._segment_creation_active() and self._drag_start_frame is not None and event.button() == QtCore.Qt.MouseButton.LeftButton:
            end_frame = self._frame_for_x(event.position().x())
            start = min(self._drag_start_frame, end_frame)
            end = max(self._drag_start_frame, end_frame)
            self._drag_start_frame = None
            self._drag_current_frame = None
            if end - start >= 1 and self.main_window is not None:
                from app.ui.widgets.actions import video_control_actions
                video_control_actions.add_segment(self.main_window, start, end)
            self.update()
            event.accept()
            return
        super().mouseReleaseEvent(event)

    def _show_segment_context_menu(self, pos):
        if self.main_window is None:
            return
        idx = self._segment_at_x(pos.x())
        if idx is None:
            return
        from app.ui.widgets.actions import video_control_actions
        video_control_actions.show_segment_context_menu(
            self.main_window, idx, self.mapToGlobal(pos)
        )

    def bind_provisional_events(self):
        if hasattr(self.main_window, "parameter_widgets"):
            for name, widget in self.main_window.parameter_widgets.items():
                if name in self._connected_widgets:
                    continue
                try:
                    if isinstance(widget, QtWidgets.QSlider):
                        widget.valueChanged.connect(self.trigger_check)
                    elif isinstance(widget, QtWidgets.QComboBox):
                        widget.currentIndexChanged.connect(self.trigger_check)
                    elif isinstance(widget, QtWidgets.QAbstractButton):
                        widget.toggled.connect(self.trigger_check)
                    elif isinstance(widget, QtWidgets.QLineEdit):
                        widget.textChanged.connect(self.trigger_check)
                    self._connected_widgets.add(name)
                except Exception:
                    pass

    # --- Paint Event ---
    def paintEvent(self, event: QtGui.QPaintEvent):
        """Native paint event overriding QSlider's default."""
        if not self.main_window:
            return super().paintEvent(event)

        if (
            self.maximum() == self.minimum()
            or self.main_window.video_processor.file_type == "image"
        ):
            return super().paintEvent(event)

        painter = QtWidgets.QStylePainter(self)
        opt = QtWidgets.QStyleOptionSlider()
        self.initStyleOption(opt)
        style = self.style()

        groove_rect = style.subControlRect(
            QtWidgets.QStyle.ComplexControl.CC_Slider,
            opt,
            QtWidgets.QStyle.SubControl.SC_SliderGroove,
        )
        groove_y = (groove_rect.top() + groove_rect.bottom()) // 2
        groove_start = groove_rect.left()
        groove_end = groove_rect.right()
        groove_width = groove_end - groove_start

        def marker_x_for_value(value: int) -> float:
            marker_normalized_value = (value - self.minimum()) / max(
                1, (self.maximum() - self.minimum())
            )
            return groove_start + marker_normalized_value * groove_width

        normalized_value = (self.value() - self.minimum()) / max(
            1, (self.maximum() - self.minimum())
        )
        handle_center_x = groove_start + normalized_value * groove_width

        handle_width = 5
        handle_height = groove_rect.height()
        handle_left_x = int(handle_center_x - (handle_width // 2))
        handle_top_y = groove_y - (handle_height // 2)

        handle_rect = QtCore.QRect(
            handle_left_x, handle_top_y, handle_width, handle_height
        )

        has_provisional_changes = self._last_provisional_state

        # 1. Base Alternating Segments
        painter.setPen(self._pen_gray)
        if not self.markers_sorted:
            painter.drawLine(groove_start, groove_y, groove_end, groove_y)
        else:
            first_marker_x = marker_x_for_value(self.markers_sorted[0])
            painter.drawLine(groove_start, groove_y, int(first_marker_x), groove_y)

            for i in range(len(self.markers_sorted)):
                start_val = self.markers_sorted[i]
                end_val = (
                    self.markers_sorted[i + 1]
                    if i + 1 < len(self.markers_sorted)
                    else self.maximum()
                )
                start_x = marker_x_for_value(start_val)
                end_x = marker_x_for_value(end_val)

                painter.setPen(
                    self._pen_segment_a if i % 2 == 0 else self._pen_segment_b
                )
                painter.drawLine(int(start_x), groove_y, int(end_x), groove_y)
                painter.drawLine(
                    int(start_x), groove_rect.top(), int(start_x), groove_rect.bottom()
                )

        # 2. Provisional Changes Overlay
        if has_provisional_changes:
            provisional_start_val = self.value()
            provisional_end_val = self.maximum()

            for m in self.markers_sorted:
                if m > provisional_start_val:
                    provisional_end_val = m
                    break

            start_x = marker_x_for_value(provisional_start_val)
            end_x = marker_x_for_value(provisional_end_val)
            painter.setPen(self._pen_provisional)
            painter.drawLine(int(start_x), groove_y, int(end_x), groove_y)

        # 3. Issue markers
        if self.issue_markers:
            painter.setPen(self._pen_issue)
            issue_top = groove_y - 2
            issue_bottom = groove_y + 2
            for value in self.issue_markers_sorted:
                if value in self.dropped_markers:
                    continue
                marker_x = marker_x_for_value(value)
                painter.drawLine(int(marker_x), issue_top, int(marker_x), issue_bottom)

        # 4. Dropped markers
        if self.dropped_markers:
            painter.setPen(self._pen_dropped)
            for value in self.dropped_markers_sorted:
                marker_x = marker_x_for_value(value)
                painter.drawLine(
                    int(marker_x),
                    groove_rect.top(),
                    int(marker_x),
                    groove_rect.bottom(),
                )

        # 6. Segments
        segments = getattr(self.main_window, "segments", [])
        if segments:
            seg_top = groove_rect.top()
            seg_height = groove_rect.bottom() - groove_rect.top()
            for seg in segments:
                kind = seg.get("kind", "normal")
                if kind == "recording":
                    pen, brush = self._pen_segment_recording, self._brush_segment_recording
                elif kind == "exclude":
                    pen, brush = self._pen_segment_exclude, self._brush_segment_exclude
                else:
                    pen, brush = self._pen_segment, self._brush_segment
                painter.setPen(pen)
                painter.setBrush(brush)
                start_x = marker_x_for_value(int(seg["start"]))
                end_x = marker_x_for_value(int(seg["end"]))
                painter.drawRect(
                    QtCore.QRect(int(start_x), seg_top, max(1, int(end_x - start_x)), seg_height)
                )

        # 7. Drag preview
        _hl_pen = QtGui.QPen(QtGui.QColor("#ff9800"), 2)
        _int_segs = (
            getattr(self.main_window, "segments", [])
            if self.main_window is not None
            else []
        )
        if (
            self._resize_segment_index is not None
            and 0 <= self._resize_segment_index < len(_int_segs)
        ):
            _rs = _int_segs[self._resize_segment_index]
            _rs_x = marker_x_for_value(int(_rs["start"]))
            _re_x = marker_x_for_value(int(_rs["end"]))
            painter.setPen(_hl_pen)
            if self._resize_edge == "start":
                painter.drawLine(
                    int(_rs_x), groove_rect.top() - 3, int(_rs_x), groove_rect.bottom() + 3
                )
            else:
                painter.drawLine(
                    int(_re_x), groove_rect.top() - 3, int(_re_x), groove_rect.bottom() + 3
                )
            self._draw_frame_flag(painter, _rs_x, str(int(_rs["start"])), anchor_left=True)
            self._draw_frame_flag(painter, _re_x, str(int(_rs["end"])), anchor_left=False)
        elif (
            self._move_segment_index is not None
            and 0 <= self._move_segment_index < len(_int_segs)
        ):
            _ms = _int_segs[self._move_segment_index]
            _ms_x = marker_x_for_value(int(_ms["start"]))
            _me_x = marker_x_for_value(int(_ms["end"]))
            painter.setPen(_hl_pen)
            painter.setBrush(QtCore.Qt.BrushStyle.NoBrush)
            painter.drawRect(
                QtCore.QRect(
                    int(_ms_x),
                    groove_rect.top(),
                    max(1, int(_me_x - _ms_x)),
                    groove_rect.bottom() - groove_rect.top(),
                )
            )
            self._draw_frame_flag(painter, _ms_x, str(int(_ms["start"])), anchor_left=True)
            self._draw_frame_flag(painter, _me_x, str(int(_ms["end"])), anchor_left=False)
        elif self._drag_start_frame is not None and self._drag_current_frame is not None:
            painter.setPen(self._pen_segment_drag)
            painter.setBrush(self._brush_segment_drag)
            _drag_lo = min(self._drag_start_frame, self._drag_current_frame)
            _drag_hi = max(self._drag_start_frame, self._drag_current_frame)
            _drag_start_x = marker_x_for_value(_drag_lo)
            _drag_end_x = marker_x_for_value(_drag_hi)
            painter.drawRect(
                QtCore.QRect(
                    int(_drag_start_x),
                    groove_rect.top(),
                    max(1, int(_drag_end_x - _drag_start_x)),
                    groove_rect.bottom() - groove_rect.top(),
                )
            )
            painter.setBrush(QtCore.Qt.BrushStyle.NoBrush)
            self._draw_frame_flag(painter, _drag_start_x, str(_drag_lo), anchor_left=True)
            self._draw_frame_flag(painter, _drag_end_x, str(_drag_hi), anchor_left=False)
        elif self._hover_segment_index is not None:
            _hover_segs = getattr(self.main_window, "segments", [])
            if 0 <= self._hover_segment_index < len(_hover_segs):
                _hs = _hover_segs[self._hover_segment_index]
                _hs_x = marker_x_for_value(int(_hs["start"]))
                _he_x = marker_x_for_value(int(_hs["end"]))
                self._draw_frame_flag(painter, _hs_x, str(int(_hs["start"])), anchor_left=True)
                self._draw_frame_flag(painter, _he_x, str(int(_hs["end"])), anchor_left=False)

        # 8. Playhead Handle (drawn last so segments never cover it)
        if has_provisional_changes:
            painter.setPen(self._pen_playhead_provisional)
            painter.setBrush(self._brush_playhead_provisional)
        else:
            painter.setPen(self._pen_playhead_normal)
            painter.setBrush(self._brush_playhead_normal)
        painter.drawRect(handle_rect)

        painter.setBrush(QtCore.Qt.BrushStyle.NoBrush)


class ThumbnailExtractionWorker(QtCore.QThread):
    # Sends (frame_number, QImage) back to the main thread
    thumbnail_ready = QtCore.Signal(int, QtGui.QImage)

    def __init__(self, media_path, frame_intervals, target_height, parent=None):
        super().__init__(parent)
        self.media_path = media_path
        self.frame_intervals = frame_intervals
        self.target_height = target_height
        self._is_cancelled = False

    def cancel(self):
        self._is_cancelled = True

    def run(self):
        if not self.media_path or not self.frame_intervals:
            return

        rotation_angle = get_video_rotation(self.media_path)
        cap = cv2.VideoCapture(self.media_path)

        if hasattr(cv2, "CAP_PROP_ORIENTATION_AUTO"):
            cap.set(cv2.CAP_PROP_ORIENTATION_AUTO, 1)

        if not cap.isOpened():
            return

        for frame_num in self.frame_intervals:
            if self._is_cancelled:
                break

            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
            ret, frame = misc_helpers.read_frame(cap, rotation_angle)

            if ret and frame is not None and not self._is_cancelled:
                # Convert BGR to RGB
                frame_rgb = frame[..., ::-1]

                # Calculate new width maintaining aspect ratio
                h, w, _ = frame_rgb.shape
                aspect_ratio = w / max(1, h)
                target_width = int(self.target_height * aspect_ratio)

                # Resize using OpenCV (faster in background thread than Qt)
                resized = cv2.resize(
                    frame_rgb,
                    (target_width, self.target_height),
                    interpolation=cv2.INTER_AREA,
                )

                # Convert to Thread-Safe QImage
                bytes_per_line = 3 * target_width
                q_img = QtGui.QImage(
                    resized.data,
                    target_width,
                    self.target_height,
                    bytes_per_line,
                    QtGui.QImage.Format_RGB888,
                ).copy()  # Crucial: .copy() detaches memory from the numpy array

                self.thumbnail_ready.emit(frame_num, q_img)

        cap.release()


class ThumbnailTrackWidget(QtWidgets.QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFixedHeight(40)
        self.setCursor(QtCore.Qt.CursorShape.PointingHandCursor)

        self.main_window = None
        self.worker = None
        self.thumbnail_cache = {}
        self.current_media_path = None
        self.expected_intervals = []
        self.thumbnail_width = 71

        self.resize_timer = QtCore.QTimer(self)
        self.resize_timer.setSingleShot(True)
        self.resize_timer.setInterval(400)
        self.resize_timer.timeout.connect(self.request_thumbnails)

    def setup(self, main_window):
        self.main_window = main_window
        # Connect the scrollbar to our debounce timer so scrolling loads new thumbnails
        scroll_area = getattr(main_window, "timelineScrollArea", None)
        if scroll_area:
            scroll_area.horizontalScrollBar().valueChanged.connect(self.on_scroll)

    def on_scroll(self, value):
        if self.main_window:
            self.resize_timer.start()

    def resizeEvent(self, event):
        super().resizeEvent(event)
        # Restart the timer every time the widget resizes (zooms)
        if self.main_window:
            self.resize_timer.start()

    def request_thumbnails(self):
        if not self.main_window or not self.main_window.video_processor:
            return

        # Respect the setting — stop generation and hide when disabled
        if not self.main_window.control.get("ShowSeekBarThumbnailsToggle", True):
            self.thumbnail_cache.clear()
            self.expected_intervals.clear()
            self.setVisible(False)
            if self.worker and self.worker.isRunning():
                self.worker.cancel()
                self.worker.wait()
            self.update()
            return

        self.setVisible(True)

        vp = self.main_window.video_processor
        if vp.file_type != "video" or not vp.media_path or vp.max_frame_number <= 0:
            self.thumbnail_cache.clear()
            self.current_media_path = None  # Reset tracker
            self.update()
            return

        # --- CLEAR CACHE ON VIDEO CHANGE ---
        if self.current_media_path != vp.media_path:
            self.thumbnail_cache.clear()
            self.current_media_path = vp.media_path
            self.expected_intervals.clear()  # Force a full recalculation

        import math

        self.thumbnail_width = int(self.height() * 1.777)
        total_thumbnails = math.ceil(self.width() / max(1, self.thumbnail_width))

        # 1. Calculate ALL global intervals for the current zoom level
        # We need this to know where to draw placeholders across the entire widget
        all_intervals = []
        for i in range(total_thumbnails):
            x_pos = i * self.thumbnail_width
            frame_num = int((x_pos / max(1, self.width())) * vp.max_frame_number)
            all_intervals.append(min(frame_num, vp.max_frame_number))

        self.expected_intervals = all_intervals

        # 2. Determine which thumbnails are currently VISIBLE in the scroll area
        scroll_area = getattr(self.main_window, "timelineScrollArea", None)
        if scroll_area:
            scroll_x = scroll_area.horizontalScrollBar().value()
            viewport_w = scroll_area.viewport().width()

            # Calculate indices based on scroll position (Add a 1-thumbnail buffer on each side for smooth scrolling)
            start_idx = max(0, int(scroll_x // self.thumbnail_width) - 1)
            end_idx = min(
                total_thumbnails,
                int((scroll_x + viewport_w) // self.thumbnail_width) + 2,
            )

            visible_intervals = all_intervals[start_idx:end_idx]
        else:
            visible_intervals = all_intervals

        # 3. Clean up cache
        # We ONLY remove thumbnails from old zoom levels to save RAM.
        # We keep the scrolled ones cached so scrubbing back and forth is instantly responsive
        keys_to_remove = [
            k for k in self.thumbnail_cache.keys() if k not in all_intervals
        ]
        for k in keys_to_remove:
            del self.thumbnail_cache[k]

        # 4. Only request extraction for VISIBLE thumbnails we don't have yet
        missing_visible = [
            f for f in visible_intervals if f not in self.thumbnail_cache
        ]

        if missing_visible:
            # Cancel any active worker
            if self.worker and self.worker.isRunning():
                self.worker.cancel()
                self.worker.wait()

            self.worker = ThumbnailExtractionWorker(
                vp.media_path, missing_visible, self.height(), self
            )
            self.worker.thumbnail_ready.connect(self.on_thumbnail_ready)
            self.worker.start()

        self.update()  # Force repaint of visible area

    @QtCore.Slot(int, QtGui.QImage)
    def on_thumbnail_ready(self, frame_num, q_image):
        # Convert the thread-safe QImage back to a GUI-friendly QPixmap
        self.thumbnail_cache[frame_num] = QtGui.QPixmap.fromImage(q_image)
        # We received new data, request a repaint
        self.update()

    def mousePressEvent(self, event: QtGui.QMouseEvent):
        """Handle clicks on the thumbnail track."""
        if event.button() == QtCore.Qt.MouseButton.LeftButton:
            # Left-click: Jump to this position and start scrubbing
            slider = getattr(self.main_window, "videoSeekSlider", None)
            if slider:
                slider.setSliderDown(True)  # Tell the app we are "dragging"
            self._seek_to_mouse_pos(event.pos().x())

        elif event.button() == QtCore.Qt.MouseButton.RightButton:
            # Right-click: Center the viewport on the current playhead
            if not self.main_window:
                return

            slider = getattr(self.main_window, "videoSeekSlider", None)
            scroll_area = getattr(self.main_window, "timelineScrollArea", None)

            if slider and scroll_area:
                current_val = slider.value()
                max_val = max(1, slider.maximum())

                # Calculate the exact pixel coordinate of the playhead
                relative_pos = current_val / max_val
                playhead_pixel_x = self.width() * relative_pos

                # Calculate where the scrollbar needs to be to center that pixel
                viewport_width = scroll_area.viewport().width()
                target_scroll_pos = int(playhead_pixel_x - (viewport_width / 2))

                # Snap the scrollbar
                scroll_area.horizontalScrollBar().setValue(max(0, target_scroll_pos))

    def mouseMoveEvent(self, event: QtGui.QMouseEvent):
        """Allow the user to click and drag across the thumbnails to scrub."""
        if event.buttons() & QtCore.Qt.MouseButton.LeftButton:
            self._seek_to_mouse_pos(event.pos().x())

    def mouseReleaseEvent(self, event: QtGui.QMouseEvent):
        """Trigger final processing when the user lets go of the mouse."""
        if event.button() == QtCore.Qt.MouseButton.LeftButton:
            slider = getattr(self.main_window, "videoSeekSlider", None)
            if slider:
                slider.setSliderDown(False)  # We stopped dragging
                # Emit the released signal so the final AI processing runs
                slider.sliderReleased.emit()

    def _seek_to_mouse_pos(self, x_pos: int):
        """Convert the mouse X coordinate into a video frame and update the slider."""
        if not self.main_window or not self.main_window.video_processor:
            return

        vp = self.main_window.video_processor
        if vp.max_frame_number <= 0:
            return

        # Clamp the X position to the widget boundaries so dragging outside doesn't crash it
        x_pos = max(0, min(x_pos, self.width()))

        # Use the exact same math we used to draw the thumbnails to find the frame
        frame_num = int((x_pos / max(1, self.width())) * vp.max_frame_number)
        frame_num = min(frame_num, vp.max_frame_number)

        # Update the slider! This instantly triggers on_change_video_seek_slider
        # which safely handles OpenCV extraction, UI updates, and previewing.
        slider = getattr(self.main_window, "videoSeekSlider", None)
        if slider:
            slider.setValue(frame_num)

    def paintEvent(self, event: QtGui.QPaintEvent):
        painter = QtGui.QPainter(self)
        painter.fillRect(self.rect(), QtGui.QColor("#1e1e1e"))

        if not self.expected_intervals:
            return

        # Safely get the video's FPS to calculate timestamps
        vp = self.main_window.video_processor if self.main_window else None
        fps = float(getattr(vp, "fps", 0.0) or 0.0) if vp else 0.0

        # Setup a small, legible font for the timestamps
        font = painter.font()
        font.setPointSize(8)
        font.setBold(True)
        painter.setFont(font)

        # Draw the thumbnails along the track
        for i, frame_num in enumerate(self.expected_intervals):
            x_pos = i * self.thumbnail_width

            if frame_num in self.thumbnail_cache:
                pixmap = self.thumbnail_cache[frame_num]
                # In case the actual video aspect ratio is wider/narrower, we center it
                actual_width = pixmap.width()
                offset = max(0, (self.thumbnail_width - actual_width) // 2)
                painter.drawPixmap(x_pos + offset, 0, pixmap)

                # --- TIMESTAMP OVERLAY ---
                if fps > 0.0:
                    total_seconds = frame_num / fps
                    hours = int(total_seconds // 3600)
                    minutes = int((total_seconds % 3600) // 60)
                    seconds = int(total_seconds % 60)

                    if hours > 0:
                        time_str = f"{hours:02d}:{minutes:02d}:{seconds:02d}"
                    else:
                        time_str = f"{minutes:02d}:{seconds:02d}"

                    # Create a bounding box at the bottom of the thumbnail space (14px high)
                    text_rect = QtCore.QRect(
                        x_pos, self.height() - 14, self.thumbnail_width, 14
                    )

                    # Draw a semi-transparent black background strip so white text always pops
                    painter.fillRect(text_rect, QtGui.QColor(0, 0, 0, 160))

                    # Draw the white text centered in the strip
                    painter.setPen(QtGui.QColor("white"))
                    painter.drawText(
                        text_rect, QtCore.Qt.AlignmentFlag.AlignCenter, time_str
                    )

            else:
                # Draw a placeholder while loading
                painter.setPen(QtGui.QColor("#444444"))
                painter.drawRect(x_pos, 0, self.thumbnail_width - 1, self.height() - 1)


class FrameRulerWidget(QtWidgets.QWidget):
    """Frame ruler drawn under the thumbnails.

    Tick spacing adapts to the media length and the current zoom level so labels
    never overlap, and only the visible part of the (scrolled) timeline is painted.
    """

    LABEL_MIN_PX = 70  # minimum horizontal distance between two labelled ticks
    MINOR_MIN_PX = 5  # minimum distance between two minor ticks

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFixedHeight(16)
        self.setCursor(QtCore.Qt.CursorShape.PointingHandCursor)
        self.main_window = None
        self._slider = None
        self._bg = QtGui.QColor("#1e1e1e")
        self._pen_minor = QtGui.QPen(QtGui.QColor("#5f5f5f"), 1)
        self._pen_major = QtGui.QPen(QtGui.QColor("#a8a8a8"), 1)
        self._pen_text = QtGui.QPen(QtGui.QColor("#c8c8c8"), 1)
        self._pen_current = QtGui.QPen(QtGui.QColor("#ffffff"), 1)
        self._font = QtGui.QFont("Arial", 7)

    def setup(self, main_window, slider):
        self.main_window = main_window
        self._slider = slider
        slider.rangeChanged.connect(lambda *_: self.update())
        slider.valueChanged.connect(lambda *_: self.update())
        scroll_area = getattr(main_window, "timelineScrollArea", None)
        if scroll_area:
            scroll_area.horizontalScrollBar().valueChanged.connect(
                lambda *_: self.update()
            )

    @staticmethod
    def _nice_step(min_frames: float) -> int:
        """Smallest 1/2/5 x 10^n step that is >= min_frames."""
        if min_frames <= 1:
            return 1
        mag = 10 ** int(math.floor(math.log10(min_frames)))
        for m in (1, 2, 5, 10):
            if m * mag >= min_frames:
                return int(m * mag)
        return int(10 * mag)

    def _steps(self, px_per_frame: float):
        major = self._nice_step(self.LABEL_MIN_PX / max(px_per_frame, 1e-9))
        mantissa = major // (10 ** int(math.floor(math.log10(major))))
        divisor = 2 if mantissa == 2 else 5
        minor = max(1, major // divisor)
        if minor * px_per_frame < self.MINOR_MIN_PX or major % minor != 0:
            minor = major
        return major, minor

    def _seek(self, x: float):
        if self._slider is not None and self._slider.maximum() > self._slider.minimum():
            self._slider.setValue(self._slider._frame_for_x(x))

    def mousePressEvent(self, event):
        if event.button() == QtCore.Qt.MouseButton.LeftButton and self._slider:
            self._slider.setSliderDown(True)
            self._seek(event.position().x())

    def mouseMoveEvent(self, event):
        if event.buttons() & QtCore.Qt.MouseButton.LeftButton:
            self._seek(event.position().x())

    def mouseReleaseEvent(self, event):
        if event.button() == QtCore.Qt.MouseButton.LeftButton and self._slider:
            self._slider.setSliderDown(False)

    def paintEvent(self, event: QtGui.QPaintEvent):
        painter = QtGui.QPainter(self)
        painter.fillRect(self.rect(), self._bg)

        slider = self._slider
        if slider is None or self.main_window is None:
            return
        vp = getattr(self.main_window, "video_processor", None)
        if vp is not None and getattr(vp, "file_type", None) == "image":
            return
        lo, hi = slider.minimum(), slider.maximum()
        if hi <= lo:
            return

        x_lo = slider._x_for_frame(lo)
        x_hi = slider._x_for_frame(hi)
        px_per_frame = (x_hi - x_lo) / (hi - lo)
        if px_per_frame <= 0:
            return
        major, minor = self._steps(px_per_frame)

        # Only paint what is visible inside the scroll area
        vis_x0, vis_x1 = 0, self.width()
        scroll_area = getattr(self.main_window, "timelineScrollArea", None)
        if scroll_area is not None:
            vis_x0 = scroll_area.horizontalScrollBar().value()
            vis_x1 = vis_x0 + scroll_area.viewport().width()

        first = max(lo, int((vis_x0 - x_lo) / px_per_frame) - minor)
        last = min(hi, int((vis_x1 - x_lo) / px_per_frame) + minor)
        first = (first // minor) * minor

        painter.setFont(self._font)
        h = self.height()
        text_flags = (
            QtCore.Qt.AlignmentFlag.AlignVCenter | QtCore.Qt.AlignmentFlag.AlignLeft
        )
        for f in range(first, last + 1, minor):
            if f < lo:
                continue
            x = int(slider._x_for_frame(f))
            if f % major == 0:
                painter.setPen(self._pen_major)
                painter.drawLine(x, h - 8, x, h)
                painter.setPen(self._pen_text)
                painter.drawText(x + 3, 0, 60, h - 2, text_flags, str(f))
            else:
                painter.setPen(self._pen_minor)
                painter.drawLine(x, h - 4, x, h)

        # Current frame indicator
        cx = int(slider._x_for_frame(slider.value()))
        painter.setPen(self._pen_current)
        painter.drawLine(cx, 0, cx, h)


class CompositeTimelineWidget(QtWidgets.QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.layout = QtWidgets.QVBoxLayout(self)
        self.layout.setContentsMargins(0, 0, 0, 0)
        self.layout.setSpacing(2)

        self.thumbnail_track = ThumbnailTrackWidget(self)
        self.ruler = FrameRulerWidget(self)
        self.slider = TimelineSeekSlider(self)

        self.layout.addWidget(self.thumbnail_track)
        self.layout.addWidget(self.ruler)
        self.layout.addWidget(self.slider)

    def setup(self, main_window):
        """Passes the main window dependency down to both children."""
        self.slider.setup(main_window)
        self.thumbnail_track.setup(main_window)
        self.ruler.setup(main_window, self.slider)
