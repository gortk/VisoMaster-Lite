"""Separate storage for the open embedding tabs of ``last_workspace.json``.

Why this exists
---------------
``last_workspace.json`` used to contain every embedding of every open tab
(vectors, names, K/V-map paths).  That made the file huge and meant that sharing
it also shared all embeddings.  Now ``last_workspace.json`` only holds a tiny
pointer (file name, tab count, active tab) and the real tab contents live in a
file next to it:  ``last_workspace_embeddings.jsonl``.

File format (JSON Lines, one JSON document per line)
----------------------------------------------------
line 1      header  {"format": "embedding_tabs", "version": 1,
                     "active_index": 0, "tabs": [{"title": ..., "filename": ...}]}
line 2..N   one line per tab (same order as ``tabs``): a JSON list of
            {"id", "name", "embedding_store", "kv_map", "tags"?}

One line per tab is what makes lazy loading possible: on startup the file is only
*scanned* (no JSON parsing) to learn where each tab starts; only the active tab is
parsed and turned into widgets.  Every other tab is created as a "pending" tab and
is parsed the first time the user opens it.  When the file is written again,
the raw line of a tab that was never opened is copied byte-for-byte, so it is never
parsed or converted to numpy.
"""

import json
import os
import threading
import time
import traceback
import uuid
from pathlib import Path
from typing import TYPE_CHECKING, List, Tuple

import numpy as np
import torch
from PySide6 import QtCore, QtWidgets

if TYPE_CHECKING:
    from app.ui.main_ui import MainWindow

FORMAT_NAME = "embedding_tabs"
FORMAT_VERSION = 1

# Guards the file while it is being rewritten, so a lazy tab load can never read
# a half-replaced file (saves can be triggered from the video-processing thread).
EMBEDDING_TABS_LOCK = threading.RLock()


def sidecar_path_for(workspace_filename) -> Path:
    """last_workspace.json -> last_workspace_embeddings.jsonl (same folder)."""
    path = Path(workspace_filename)
    return path.with_name(f"{path.stem}_embeddings.jsonl")


# --------------------------------------------------------------------------- #
# Reading
# --------------------------------------------------------------------------- #


def scan_embedding_tabs_file(path) -> Tuple[dict, List[dict]]:
    """Parse only the header and locate every tab line (no tab JSON is parsed)."""
    header = None
    lines: List[dict] = []
    with open(path, "rb") as f:
        offset = 0
        for raw in f:
            length = len(raw)
            body = raw.rstrip(b"\r\n")
            if body.strip():
                if header is None:
                    header = json.loads(body.decode("utf-8"))
                    if (
                        not isinstance(header, dict)
                        or header.get("format") != FORMAT_NAME
                        or not isinstance(header.get("tabs"), list)
                    ):
                        raise ValueError("not an embedding tabs file")
                else:
                    lines.append({"offset": offset, "length": len(body)})
            offset += length
    if header is None:
        raise ValueError("embedding tabs file is empty")
    return header, lines


def _read_raw_line(pending: dict) -> bytes:
    with open(pending["path"], "rb") as f:
        f.seek(pending["offset"])
        raw = f.read(pending["length"])
    if len(raw) != pending["length"]:
        raise OSError(f"embedding tabs file is truncated: {pending['path']}")
    return raw


def read_pending_entries(pending: dict) -> List[dict]:
    """Parse the entries of one not-yet-opened tab."""
    with EMBEDDING_TABS_LOCK:
        raw = _read_raw_line(pending)
    entries = json.loads(raw.decode("utf-8"))
    if not isinstance(entries, list):
        raise ValueError("tab line is not a list")
    return [entry for entry in entries if isinstance(entry, dict)]


def _preserve_broken_file(path: Path) -> None:
    broken = f"{path}.broken.{int(time.time())}"
    try:
        os.rename(path, broken)
        print(f"[WARN] Unreadable embedding tabs file preserved as {broken}")
    except OSError as e:
        print(f"[WARN] Could not preserve unreadable embedding tabs file: {e}")


def _create_button_from_entry(main_window: "MainWindow", entry: dict, workspace_path):
    from app.ui.widgets.actions import list_view_actions
    from app.ui.widgets.actions import save_load_actions

    store = {
        model: np.array(embedding)
        for model, embedding in (entry.get("embedding_store") or {}).items()
    }
    name = entry.get("name", entry.get("embedding_name", "Unknown"))
    embedding_id = str(entry.get("id") or uuid.uuid1().int)
    if embedding_id in main_window.merged_embeddings:
        embedding_id = str(uuid.uuid1().int)

    list_view_actions.create_and_add_embed_button_to_list(
        main_window,
        name,
        store,
        embedding_id=embedding_id,
        tags=entry.get("tags"),
    )
    button = main_window.merged_embeddings.get(embedding_id)
    kv_map_path = entry.get("kv_map")
    if button is None or not (kv_map_path and os.path.exists(kv_map_path)):
        return
    try:
        payload = torch.load(kv_map_path, map_location="cpu", weights_only=False)
        if isinstance(payload, dict):
            if "kv_map_list" in payload:
                button.kv_map_list = payload["kv_map_list"]
            else:
                button.kv_map = payload.get("kv_map")
        else:
            button.kv_map = payload
        if workspace_path:
            save_load_actions._update_registry(
                main_window,
                Path(kv_map_path).name,
                str(workspace_path),
                "reference_kv_data",
            )
    except Exception as e:
        print(f"[ERROR] Error loading K/V map for embedding '{name}': {e}")


def materialize_pending_tab(main_window: "MainWindow", state: dict) -> bool:
    """Parse a pending tab and create its embedding buttons (once)."""
    pending = state.get("pending")
    if not pending:
        return True
    try:
        entries = read_pending_entries(pending)
    except Exception as e:
        # Keep it pending: the raw data is still copied on the next save.
        print(f"[ERROR] Could not read embeddings of this tab: {e}")
        return False

    list_widget = state["list_widget"]
    main_window.inputEmbeddingsList = list_widget
    main_window.merged_embeddings = state["embeddings"]

    QtWidgets.QApplication.setOverrideCursor(QtCore.Qt.CursorShape.WaitCursor)
    list_widget.setUpdatesEnabled(False)
    try:
        for entry in entries:
            try:
                _create_button_from_entry(
                    main_window, entry, pending.get("workspace")
                )
            except Exception:
                traceback.print_exc()
    finally:
        list_widget.setUpdatesEnabled(True)
        QtWidgets.QApplication.restoreOverrideCursor()
    state.pop("pending", None)
    return True


def restore_tabs_from_file(
    main_window: "MainWindow", path, workspace_filename, active_hint: int = 0
) -> bool:
    """Recreate the saved tabs.  Only the active tab is loaded into memory."""
    from app.ui.widgets.actions import list_view_actions

    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(
            f"Embedding tabs file not found, tabs were not restored:\n{path}"
        )
    try:
        with EMBEDDING_TABS_LOCK:
            header, lines = scan_embedding_tabs_file(path)
    except (OSError, ValueError) as e:
        _preserve_broken_file(path)
        raise RuntimeError(f"Embedding tabs file is unreadable: {e}") from e

    metas = header.get("tabs") or []
    count = min(len(metas), len(lines))
    if count == 0:
        return True

    tabs = main_window.embeddingTabs
    active = header.get("active_index", active_hint)
    try:
        active = int(active)
    except (TypeError, ValueError):
        active = 0
    active = min(max(0, active), count - 1)

    list_view_actions.stop_embedding_filter_worker(main_window)
    was_rebuilding = getattr(main_window, "_embedding_tabs_rebuilding", False)
    # While rebuilding, tab activation does not load pending tabs or start filters.
    main_window._embedding_tabs_rebuilding = True
    try:
        for i in range(count):
            meta = metas[i] if isinstance(metas[i], dict) else {}
            title = meta.get("title") or f"Embeddings {i + 1}"
            if i >= len(main_window.embedding_tab_states):
                list_view_actions.add_embedding_tab(main_window, title=title)
            state = main_window.embedding_tab_states[i]
            state["filename"] = meta.get("filename") or ""
            state["pending"] = {
                "path": str(path),
                "offset": lines[i]["offset"],
                "length": lines[i]["length"],
                "workspace": str(workspace_filename),
            }
            tabs.setTabText(i, title)
            tabs.setTabToolTip(i, state["filename"] or title)
        tabs.setCurrentIndex(active)
    finally:
        main_window._embedding_tabs_rebuilding = was_rebuilding

    if not was_rebuilding:
        # Loads the active tab only; the others stay pending until opened.
        list_view_actions._activate_embedding_tab(main_window, active)
    return True


# --------------------------------------------------------------------------- #
# Writing
# --------------------------------------------------------------------------- #


def _collect_tab_entries(
    main_window: "MainWindow", state: dict, parent_path: str
) -> List[dict]:
    from app.ui.widgets.actions import save_load_actions

    entries: List[dict] = []
    for embedding_id, button in list(state["embeddings"].items()):
        try:
            name = button.embedding_name
            store = button.embedding_store
            tags = list(getattr(button, "tags", None) or [])
        except RuntimeError:  # widget already deleted
            continue

        kv_map_path = None
        kv_list = getattr(button, "kv_map_list", None)
        kv_payload = kv_list or getattr(button, "kv_map", None)
        if kv_payload is not None:
            try:
                payload = (
                    {"kv_map_list": kv_payload} if kv_list else {"kv_map": kv_payload}
                )
                kv_map_path = save_load_actions._save_hashed_kv_payload(
                    main_window,
                    payload,
                    "reference_kv_data",
                    parent_file_paths=parent_path,
                )
            except Exception as e:
                print(
                    f"[ERROR] Error saving hashed K/V map for embedding {embedding_id}: {e}"
                )

        entry = {
            "id": str(embedding_id),
            "name": name,
            "embedding_store": {
                model: np.asarray(vector).tolist() for model, vector in store.items()
            },
            "kv_map": kv_map_path,
        }
        if tags:
            entry["tags"] = tags
        entries.append(entry)
    return entries


def write_embedding_tabs_file(main_window: "MainWindow", workspace_filename) -> bool:
    """Write all tabs next to the workspace file (atomic replace, no .bak files).

    Tabs that were never opened this session are copied as raw bytes.
    """
    states = list(getattr(main_window, "embedding_tab_states", None) or [])
    if not states:
        return False
    tabs = getattr(main_window, "embeddingTabs", None)
    path = sidecar_path_for(workspace_filename)
    parent_path = str(workspace_filename)

    header = {
        "format": FORMAT_NAME,
        "version": FORMAT_VERSION,
        "active_index": max(0, tabs.currentIndex()) if tabs is not None else 0,
        "tabs": [
            {
                "title": tabs.tabText(i) if tabs is not None and i < tabs.count() else "",
                "filename": state.get("filename") or "",
            }
            for i, state in enumerate(states)
        ],
    }

    tmp_path = f"{path}.{os.getpid()}.{uuid.uuid4().hex[:8]}.tmp"
    try:
        with EMBEDDING_TABS_LOCK:
            moved: List[Tuple[dict, int, int]] = []
            with open(tmp_path, "wb") as f:
                f.write(json.dumps(header).encode("utf-8") + b"\n")
                for state in states:
                    pending = state.get("pending")
                    offset = f.tell()
                    if pending:
                        raw = _read_raw_line(pending)
                    else:
                        entries = _collect_tab_entries(main_window, state, parent_path)
                        raw = json.dumps(entries, separators=(",", ":")).encode("utf-8")
                    f.write(raw + b"\n")
                    if pending:
                        moved.append((state, offset, len(raw)))
                f.flush()
                try:
                    os.fsync(f.fileno())
                except OSError:
                    pass

            _, written_lines = scan_embedding_tabs_file(tmp_path)
            if len(written_lines) != len(states):
                raise RuntimeError("written embedding tabs file failed validation")

            for attempt in range(10):
                try:
                    os.replace(tmp_path, path)
                    break
                except PermissionError:
                    if attempt == 9:
                        raise
                    time.sleep(0.1)

            # Tabs that were never opened now live at new offsets in the new file.
            for state, offset, length in moved:
                pending = state.get("pending")
                if pending:
                    pending.update(path=str(path), offset=offset, length=length)
        return True
    except Exception as e:
        try:
            os.remove(tmp_path)
        except OSError:
            pass
        print(f"[ERROR] Failed to save embedding tabs file {path}: {e}")
        return False
