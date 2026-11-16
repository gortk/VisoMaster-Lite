"""Small popup that reports TensorRT engine generation.

Design notes
------------
* The dialog only ever lives in the GUI thread.  Everything that happens
  elsewhere (probe retry loop, worker threads doing a first-run build, DFM
  loads) talks to it through Qt signals, so the calls are queued onto the GUI
  thread and never touch widgets from a foreign thread.
* The elapsed time is *derived* from ``time.monotonic()`` on every tick instead
  of being incremented, so a late tick (heavy load) never makes the clock drift.
* Several builds can overlap (e.g. two worker threads finishing the lazy build
  of different models).  Each build is a "job"; the popup stays up until the
  last job has ended and closes by itself afterwards.
* A job can ask for a short show-delay.  A build that really has to generate an
  engine takes tens of seconds, a false alarm (engine already cached) is over
  in a moment, so with the delay the popup never flashes on plain model loads.
* While a build is running the user cannot close the popup (no close button,
  Esc / Alt+F4 ignored).  It closes itself when the last job ends, so it can
  never be left open without a build behind it.
"""

from __future__ import annotations

import re
import time
from typing import Dict

from PySide6 import QtCore, QtWidgets


_ANSI_RE = re.compile(r"\x1b(?:\[[0-?]*[ -/]*[@-~]|[@-Z\\-_])")
_CTRL_RE = re.compile(r"[\x00-\x08\x0b-\x1f\x7f]")
# "2026-11-16 09:32:18.812 [W:onnxruntime:, file.cc:123 fn] message" -> "message"
_ORT_PREFIX_RE = re.compile(r"^\s*\d{4}-\d{2}-\d{2}[ T][\d:.]+\s*(?:\[[^\]]*\]\s*)?")


def _clean_log_line(text: str) -> str:
    """Make a raw ONNX Runtime / TensorRT log line fit for a label."""
    text = _ANSI_RE.sub("", text)
    text = _CTRL_RE.sub("", text)
    text = _ORT_PREFIX_RE.sub("", text)
    return text.strip()


def _format_elapsed(seconds: float) -> str:
    total = max(0, int(seconds))
    hours, rest = divmod(total, 3600)
    minutes, secs = divmod(rest, 60)
    if hours:
        return f"{hours}:{minutes:02d}:{secs:02d}"
    return f"{minutes:02d}:{secs:02d}"


class _Job:
    __slots__ = ("model", "attempt", "max_attempts", "status", "started")

    def __init__(self, model: str, max_attempts: int) -> None:
        self.model = model
        self.attempt = 1
        self.max_attempts = max(1, int(max_attempts))
        self.status = ""
        self.started = time.monotonic()


class TrtBuildDialog(QtWidgets.QDialog):
    """Progress popup for TensorRT engine generation."""

    TICK_MS = 250

    def __init__(self, parent: QtWidgets.QWidget | None = None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Building TensorRT Engine")
        self.setWindowFlags(
            QtCore.Qt.WindowType.Dialog
            | QtCore.Qt.WindowType.CustomizeWindowHint
            | QtCore.Qt.WindowType.WindowTitleHint
        )  # title bar without a close button
        self.setWindowModality(QtCore.Qt.WindowModality.WindowModal)
        self.setMinimumWidth(440)

        self._jobs: Dict[str, _Job] = {}

        self._title_label = QtWidgets.QLabel("Generating TensorRT engine cache...")
        title_font = self._title_label.font()
        title_font.setBold(True)
        self._title_label.setFont(title_font)

        self._model_value = QtWidgets.QLabel("-")
        self._model_value.setTextInteractionFlags(
            QtCore.Qt.TextInteractionFlag.TextSelectableByMouse
        )
        self._attempt_value = QtWidgets.QLabel("-")
        self._elapsed_value = QtWidgets.QLabel("00:00")

        form = QtWidgets.QFormLayout()
        form.setLabelAlignment(QtCore.Qt.AlignmentFlag.AlignLeft)
        form.addRow("Model:", self._model_value)
        form.addRow("Attempt:", self._attempt_value)
        form.addRow("Elapsed:", self._elapsed_value)

        self._bar = QtWidgets.QProgressBar()
        self._bar.setRange(0, 0)  # TensorRT reports no percentage: busy bar
        self._bar.setTextVisible(False)

        self._status_label = QtWidgets.QLabel("")
        self._status_label.setEnabled(False)  # dimmed by the active theme
        self._queue_label = QtWidgets.QLabel("")
        self._hint_label = QtWidgets.QLabel(
            "This can take several minutes and only happens once per model."
        )
        self._hint_label.setWordWrap(True)

        layout = QtWidgets.QVBoxLayout(self)
        layout.setSpacing(8)
        layout.addWidget(self._title_label)
        layout.addLayout(form)
        layout.addWidget(self._bar)
        layout.addWidget(self._status_label)
        layout.addWidget(self._queue_label)
        layout.addWidget(self._hint_label)

        self._tick = QtCore.QTimer(self)
        self._tick.setInterval(self.TICK_MS)
        self._tick.timeout.connect(self._refresh)

        self._reveal_timer = QtCore.QTimer(self)
        self._reveal_timer.setSingleShot(True)
        self._reveal_timer.timeout.connect(self._reveal)

    # ------------------------------------------------------------------ API
    # All of these are meant to be connected to signals (queued onto the GUI
    # thread when emitted from another thread).

    @QtCore.Slot(str, str, int, int)
    def begin(
        self, job_id: str, model: str, max_attempts: int, show_delay_ms: int
    ) -> None:
        """Register a build. ``show_delay_ms`` > 0 postpones the popup."""
        self._jobs[job_id] = _Job(model, max_attempts)
        if not self._tick.isActive():
            self._tick.start()
        if self.isVisible():
            self._refresh()
        elif show_delay_ms <= 0:
            self._reveal()
        elif not self._reveal_timer.isActive():
            self._reveal_timer.start(show_delay_ms)

    @QtCore.Slot(str, int)
    def set_attempt(self, job_id: str, attempt: int) -> None:
        job = self._jobs.get(job_id)
        if job is None:
            return
        job.attempt = max(1, int(attempt))
        job.max_attempts = max(job.max_attempts, job.attempt)
        self._refresh()

    @QtCore.Slot(str, str)
    def set_status(self, job_id: str, text: str) -> None:
        job = self._jobs.get(job_id)
        if job is None:
            return
        cleaned = _clean_log_line(text)
        if not cleaned:
            return  # keep the previous line rather than blanking the label
        job.status = cleaned
        self._refresh()

    @QtCore.Slot(str)
    def end(self, job_id: str) -> None:
        """Unregister a build; closes the popup when none are left."""
        self._jobs.pop(job_id, None)
        if self._jobs:
            self._refresh()
            return
        self._reveal_timer.stop()
        self._tick.stop()
        self.hide()

    # ------------------------------------------------------------- internals

    def _current_job(self) -> _Job | None:
        # dicts keep insertion order: show the build that has been running longest
        for job in self._jobs.values():
            return job
        return None

    def _reveal(self) -> None:
        if not self._jobs:
            return
        self._refresh()
        if not self.isVisible():
            self.show()
        self.raise_()
        # Paint right away; the caller may be about to do something heavy.
        QtCore.QCoreApplication.processEvents()

    def _refresh(self) -> None:
        job = self._current_job()
        if job is None:
            return
        self._model_value.setText(job.model)
        self._attempt_value.setText(f"{job.attempt} of {job.max_attempts}")
        self._elapsed_value.setText(_format_elapsed(time.monotonic() - job.started))
        metrics = self._status_label.fontMetrics()
        self._status_label.setText(
            metrics.elidedText(
                job.status, QtCore.Qt.TextElideMode.ElideRight, 400
            )
        )
        others = len(self._jobs) - 1
        self._queue_label.setText(
            f"{others} more engine build{'s' if others != 1 else ''} running"
            if others > 0
            else ""
        )

    # --------------------------------------------------------- close guards

    def closeEvent(self, event) -> None:  # noqa: N802 (Qt naming)
        if self._jobs:
            event.ignore()  # a build is still running
        else:
            event.accept()

    def reject(self) -> None:
        if not self._jobs:
            super().reject()
