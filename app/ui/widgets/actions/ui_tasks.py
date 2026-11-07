"""Off-thread execution helper for blocking UI-adjacent work.

The GUI thread must own all Qt widgets and the event loop. This module runs
blocking payloads on a QThreadPool and delivers callbacks back on the GUI
thread.
"""

from __future__ import annotations

from typing import Any, Callable, Optional

from PySide6 import QtCore


class _Runnable(QtCore.QRunnable):
    def __init__(self, fn, args, kwargs, done_cb, err_cb):
        super().__init__()
        self._fn = fn
        self._args = args
        self._kwargs = kwargs
        self._done_cb = done_cb
        self._err_cb = err_cb

    def run(self) -> None:
        try:
            result = self._fn(*self._args, **self._kwargs)
        except BaseException as exc:  # noqa: BLE001
            self._err_cb(exc)
            return
        self._done_cb(result)


class TaskRunner(QtCore.QObject):
    """Runs blocking work off the GUI thread; delivers callbacks on the GUI thread."""

    _done_signal = QtCore.Signal(object, object)  # (callback, result)
    _err_signal = QtCore.Signal(object, object)   # (callback, exception)

    def __init__(self, parent: Optional[QtCore.QObject] = None) -> None:
        super().__init__(parent)
        self._pool = QtCore.QThreadPool(self)
        self._pool.setMaxThreadCount(4)
        self._done_signal.connect(self._apply_done)
        self._err_signal.connect(self._apply_err)

    def submit(self, fn: Callable[..., Any], *args: Any,
               on_done: Optional[Callable[[object], None]] = None,
               on_err: Optional[Callable[[BaseException], None]] = None,
               **kwargs: Any) -> None:
        runnable = _Runnable(
            fn, args, kwargs,
            done_cb=lambda result: self._done_signal.emit(on_done, result),
            err_cb=lambda exc: self._err_signal.emit(on_err, exc),
        )
        self._pool.start(runnable)

    @QtCore.Slot(object, object)
    def _apply_done(self, on_done, result) -> None:
        if callable(on_done):
            on_done(result)

    @QtCore.Slot(object, object)
    def _apply_err(self, on_err, exc) -> None:
        if callable(on_err):
            on_err(exc)

    def shutdown(self) -> None:
        self._pool.waitForDone(2000)