"""Opt-in timing counters for the frame pipeline (off by default, zero cost when off).

Enable with the environment variable  VISOMASTER_PROFILE_GPU=1  before starting the app.
Every VISOMASTER_PROFILE_INTERVAL frames (default 120) one summary line is printed:

  [PROFILE] 120 frames | frame 71.3 ms | in ORT 38.1 ms (53%) [pre-sync 4.0, gate 1.2,
  lock 3.3, run 14.0, post-sync 15.6] | outside ORT 33.2 ms (47%) | upload 2.1 ms |
  download 6.4 ms | 9.2 ORT calls/frame

Meaning (all numbers are per frame, averaged over the interval and over worker threads):
  frame        wall time of FrameWorker.process_frame
  in ORT       time the worker spent inside run_ort_with_iobinding
    pre-sync   wait for the worker's CUDA stream before launching the model
    gate/lock  waiting for the GFPGAN gate / the per-session lock (contention between threads)
    run        run_with_iobinding call
    post-sync  io_binding.synchronize_outputs()
  outside ORT  Python / torch / numpy work between model calls (where the GPU can starve)
  upload / download   host->device frame upload / device->host result copy
"""

import os
import threading
import time

ENABLED: bool = os.environ.get("VISOMASTER_PROFILE_GPU", "0") == "1"
try:
    _INTERVAL: int = max(1, int(os.environ.get("VISOMASTER_PROFILE_INTERVAL", "120")))
except ValueError:
    _INTERVAL = 120

perf = time.perf_counter

_lock = threading.Lock()
_tls = threading.local()
_acc: dict = {}
_frames = 0
_frame_total = 0.0
_frame_ort = 0.0
_ort_calls = 0


def add(name: str, seconds: float) -> None:
    if not ENABLED:
        return
    with _lock:
        _acc[name] = _acc.get(name, 0.0) + seconds


def ort_call(pre: float, gate: float, lock: float, run: float, post: float) -> None:
    """Record one run_ort_with_iobinding call (all arguments in seconds)."""
    global _ort_calls
    total = pre + gate + lock + run + post
    _tls.ort = getattr(_tls, "ort", 0.0) + total
    with _lock:
        for k, v in (
            ("pre-sync", pre),
            ("gate", gate),
            ("lock", lock),
            ("run", run),
            ("post-sync", post),
        ):
            _acc[k] = _acc.get(k, 0.0) + v
        _ort_calls += 1


def frame_begin() -> None:
    _tls.ort = 0.0


def frame_end(total_seconds: float) -> None:
    global _frames, _frame_total, _frame_ort, _ort_calls
    line = None
    with _lock:
        _frames += 1
        _frame_total += total_seconds
        _frame_ort += getattr(_tls, "ort", 0.0)
        if _frames >= _INTERVAL:
            n = _frames
            frame_ms = _frame_total / n * 1000.0
            ort_ms = _frame_ort / n * 1000.0
            out_ms = max(frame_ms - ort_ms, 0.0)
            pct = (lambda x: (x / frame_ms * 100.0) if frame_ms > 0 else 0.0)
            parts = ", ".join(
                f"{k} {_acc.get(k, 0.0) / n * 1000.0:.1f}"
                for k in ("pre-sync", "gate", "lock", "run", "post-sync")
            )
            line = (
                f"[PROFILE] {n} frames | frame {frame_ms:.1f} ms | "
                f"in ORT {ort_ms:.1f} ms ({pct(ort_ms):.0f}%) [{parts}] | "
                f"outside ORT {out_ms:.1f} ms ({pct(out_ms):.0f}%) | "
                f"upload {_acc.get('upload', 0.0) / n * 1000.0:.1f} ms | "
                f"download {_acc.get('download', 0.0) / n * 1000.0:.1f} ms | "
                f"{_ort_calls / n:.1f} ORT calls/frame"
            )
            _acc.clear()
            _frames = 0
            _frame_total = 0.0
            _frame_ort = 0.0
            _ort_calls = 0
    if line:
        print(line, flush=True)
