import os
import json
import hashlib
import io
import time
import shutil
import threading
import weakref
from pathlib import Path
import uuid
import copy
from functools import partial
from typing import TYPE_CHECKING, Dict, Union, cast, Optional, List

from PySide6 import QtWidgets, QtCore
import numpy as np
import torch

from app.ui.widgets import widget_components
from app.ui.widgets.actions import common_actions as common_widget_actions
from app.ui.widgets.actions import card_actions
from app.ui.widgets.actions import list_view_actions
from app.ui.widgets.actions import video_control_actions
from app.ui.widgets.actions import control_actions
from app.ui.widgets.actions import layout_actions
from app.ui.widgets.actions import filter_actions
from app.ui.widgets.actions import embedding_tabs_store
from app.ui.widgets import ui_workers
from app.helpers.typing_helper import ParametersTypes, MarkerTypes
import app.helpers.miscellaneous as misc_helpers

if TYPE_CHECKING:
    from app.ui.main_ui import MainWindow

_KV_REGISTRY_LOCK = threading.Lock()
_KV_CACHE_LOCK = threading.RLock()
_WORKSPACE_WRITE_LOCK = threading.Lock()


def _repair_workspace_file_if_needed(data_filename):
    primary = str(data_filename)
    bak = primary + ".bak"
    bak2 = primary + ".bak2"

    primary_ok = False
    if os.path.isfile(primary):
        try:
            with open(primary, "r", encoding="utf-8") as f:
                json.load(f)
            primary_ok = True
        except (OSError, json.JSONDecodeError, ValueError):
            pass

    if primary_ok:
        return True

    for candidate in (bak, bak2):
        if not os.path.isfile(candidate):
            continue
        try:
            with open(candidate, "r", encoding="utf-8") as src:
                content = src.read()
            json.loads(content)
        except (OSError, json.JSONDecodeError, ValueError):
            continue
        try:
            with open(primary, "w", encoding="utf-8") as dst:
                dst.write(content)
            print(f"[WARN] Restored workspace from {candidate}")
            return True
        except OSError as e:
            print(f"[ERROR] Failed to restore workspace from {candidate}: {e}")
            return False

    return False


def _notify_workspace_load_done(main_window) -> None:
    main_window._workspace_load_in_progress = False
    cb = getattr(main_window, "_on_workspace_load_finished", None)
    if callable(cb):
        try:
            cb()
        except Exception as e:
            print(f"[WARN] workspace completion callback failed: {e}")


def _load_registry(registry_path: Path) -> Dict[str, List[str]]:
    """
    Loads the KV map registry safely.
    Returns an empty dictionary if the file is missing or corrupted.
    """
    if not registry_path.exists():
        return {}
    try:
        with open(registry_path, "r", encoding="utf-8") as f:
            return json.load(f)
    except (json.JSONDecodeError, IOError) as e:
        print(f"[WARN] Failed to read KV registry (might be corrupted), resetting: {e}")
        return {}


def _update_registry(
    main_window: "MainWindow",
    kv_filename: str,
    parent_file_paths: Union[str, List[str]],
    sub_folder: str = "reference_kv_data",
) -> None:
    """
    Thread-safe function to update the central KV registry.
    Adds one or more parent file paths (workspaces, jobs, or source image paths)
    to the list of dependents for the specified KV map.
    """
    if not parent_file_paths:
        return

    if isinstance(parent_file_paths, str):
        parent_file_paths = [parent_file_paths]

    registry_path = (
        main_window.project_root_path / "model_assets" / sub_folder / "kv_registry.json"
    )

    with _KV_REGISTRY_LOCK:
        registry = _load_registry(registry_path)

        if kv_filename not in registry:
            registry[kv_filename] = []

        registry_modified = False
        for path in parent_file_paths:
            if not path:
                continue
            normalized_path = str(Path(path).resolve())
            if normalized_path not in registry[kv_filename]:
                registry[kv_filename].append(normalized_path)
                registry_modified = True

        if registry_modified:
            try:
                with open(registry_path, "w", encoding="utf-8") as f:
                    json.dump(registry, f, indent=4)
            except IOError as e:
                print(f"[ERROR] Failed to write KV registry to {registry_path}: {e}")


_KV_PAYLOAD_KEYS = ("kv_map", "kv_map_list")
_KV_SIGNATURE_MAX_DEPTH = 8


def _kv_signature(value, leaves: list, depth: int = 0):
    """Cheap, content-free signature of a K/V structure, or None if unsupported.

    The structure (keys, list lengths, scalars) is compared by value. Every
    tensor contributes its version counter, shape, dtype, device, stride and
    requires_grad, and is appended to ``leaves`` so the caller can verify
    *object identity* separately (via weak references). Nothing here reads
    tensor data, so no GPU->CPU copy and no hashing is needed.
    """
    if depth > _KV_SIGNATURE_MAX_DEPTH:
        return None

    if isinstance(value, torch.Tensor):
        try:
            if value.layout != torch.strided:
                return None
            signature = (
                "T",
                value._version,  # raises on inference tensors -> not cached
                tuple(value.shape),
                str(value.dtype),
                str(value.device),
                tuple(value.stride()),
                bool(value.requires_grad),
            )
        except Exception:
            return None
        leaves.append(value)
        return signature

    if isinstance(value, dict):
        items = []
        for key, child in value.items():
            child_signature = _kv_signature(child, leaves, depth + 1)
            if child_signature is None:
                return None
            items.append((key, child_signature))
        return ("D", tuple(items))

    if isinstance(value, (list, tuple)):
        children = []
        for child in value:
            child_signature = _kv_signature(child, leaves, depth + 1)
            if child_signature is None:
                return None
            children.append(child_signature)
        return (type(value).__name__, tuple(children))

    if value is None or isinstance(value, (bool, int, float, str, bytes)):
        return (type(value).__name__, value)

    # Unknown objects can change without changing identity: never cache them.
    return None


def _kv_cache_target(payload):
    """(payload_key, value) for the one-key payloads the app saves, else None."""
    if isinstance(payload, dict) and len(payload) == 1:
        key, value = next(iter(payload.items()))
        if key in _KV_PAYLOAD_KEYS:
            return key, value
    return None


def _kv_cache_get(main_window, payload, sub_folder: str, kv_data_dir: Path):
    """Return the existing file for an unchanged payload, or None (cache miss)."""
    target = _kv_cache_target(payload)
    cache = getattr(main_window, "_kv_payload_save_cache", None)
    if target is None or not cache:
        return None
    payload_key, value = target

    with _KV_CACHE_LOCK:
        entry = cache.get((str(sub_folder), payload_key, id(value)))
    if entry is None:
        return None
    leaf_refs, signature, kv_map_file = entry

    leaves: list = []
    if _kv_signature(value, leaves) != signature or len(leaves) != len(leaf_refs):
        return None
    # Same *objects* (not merely same-shaped tensors). A dead weak reference
    # means the original tensor was freed, so this entry is stale.
    if any(ref() is not leaf for ref, leaf in zip(leaf_refs, leaves)):
        return None

    try:
        same_dir = kv_map_file.parent == kv_data_dir or (
            kv_map_file.parent.resolve() == kv_data_dir.resolve()
        )
        if same_dir and kv_map_file.is_file():
            return kv_map_file
    except OSError:
        pass
    return None


def _kv_cache_put(main_window, payload, sub_folder: str, kv_map_file: Path) -> None:
    """Remember that this exact set of tensors is stored in ``kv_map_file``.

    Only weak references to the tensors are kept, so caching never keeps K/V
    data (usually on the GPU) alive after a face or embedding is removed.
    """
    target = _kv_cache_target(payload)
    if target is None:
        return
    payload_key, value = target

    leaves: list = []
    signature = _kv_signature(value, leaves)
    if signature is None or not leaves:
        return
    try:
        leaf_refs = tuple(weakref.ref(leaf) for leaf in leaves)
    except TypeError:
        return

    with _KV_CACHE_LOCK:
        cache = getattr(main_window, "_kv_payload_save_cache", None)
        if cache is None:
            cache = {}
            main_window._kv_payload_save_cache = cache
        # Drop entries whose tensors have been freed.
        for key in [
            k for k, (refs, _sig, _path) in cache.items()
            if any(ref() is None for ref in refs)
        ]:
            del cache[key]
        cache[(str(sub_folder), payload_key, id(value))] = (
            leaf_refs,
            signature,
            Path(kv_map_file),
        )


def remember_loaded_kv_payload(
    main_window: "MainWindow",
    payload,
    kv_map_path,
    sub_folder: str = "reference_kv_data",
) -> None:
    """Call right after ``torch.load(kv_map_path)`` for a K/V payload.

    The loaded tensors are, by construction, what ``kv_map_path`` contains, so
    the next workspace save can reuse that file without re-serializing and
    hashing them (this is what makes the very first close after a launch fast).
    Safe to call with anything: unsupported payloads are simply ignored.
    """
    try:
        path = Path(kv_map_path)
        if not (path.name.startswith("kv_") and path.suffix == ".pt"):
            return
        _kv_cache_put(main_window, payload, sub_folder, path)
    except Exception:
        pass


def _save_hashed_kv_payload(
    main_window: "MainWindow",
    payload: dict,
    sub_folder: str = "reference_kv_data",
    parent_file_paths: Optional[Union[str, List[str]]] = None,
) -> Optional[str]:
    """Save a content-addressed K/V payload (``kv_<sha256>.pt``) and return its path.

    Serializing + hashing is the expensive part (and needs a GPU->CPU copy for
    GPU tensors), so unchanged payloads are recognised by tensor identity and
    version counters (see ``_kv_cache_get``) and skip it entirely. Registry
    updates still happen on every call, including cache hits.
    """
    try:
        kv_data_dir = main_window.project_root_path / "model_assets" / sub_folder

        kv_map_file = _kv_cache_get(main_window, payload, sub_folder, kv_data_dir)

        if kv_map_file is None:
            buffer = io.BytesIO()
            torch.save(payload, buffer)
            buffer_bytes = buffer.getvalue()
            payload_hash = hashlib.sha256(buffer_bytes).hexdigest()

            kv_data_dir.mkdir(parents=True, exist_ok=True)
            kv_map_file = kv_data_dir / f"kv_{payload_hash}.pt"

            if not kv_map_file.exists():
                with open(kv_map_file, "wb") as f:
                    f.write(buffer_bytes)
                print(f"[INFO] Saved new K/V map: {kv_map_file.name}")
            else:
                print(
                    f"[INFO] K/V map {kv_map_file.name} already exists. Skipping disk write."
                )

            _kv_cache_put(main_window, payload, sub_folder, kv_map_file)

        if parent_file_paths:
            _update_registry(
                main_window, kv_map_file.name, parent_file_paths, sub_folder
            )

        return str(kv_map_file)

    except Exception as e:
        print(f"[ERROR] Failed to hash and save K/V map payload: {e}")
        return None


def sanitize_markers_dictionary(
    main_window: "MainWindow", markers: dict | None
) -> dict:
    """
    Deep-sanitizes the markers dictionary, ensuring both 'control' and 'parameters'
    for every marker conform strictly to the current active schemas.
    """
    if not markers:
        return {}

    sanitized_markers = {}
    default_control = main_window.control
    default_params = cast(
        misc_helpers.ParametersDict, main_window.default_parameters
    ).data

    for marker_position, marker_data in markers.items():
        marker_payload = copy.deepcopy(marker_data)

        marker_payload["control"] = sanitize_state_dictionary(
            marker_payload.get("control", {}), default_control
        )

        if "parameters" in marker_payload:
            for face_id, face_params in marker_payload["parameters"].items():
                face_params_dict = (
                    face_params.data if hasattr(face_params, "data") else face_params
                )
                marker_payload["parameters"][face_id] = sanitize_state_dictionary(
                    face_params_dict, default_params
                )

        sanitized_markers[marker_position] = marker_payload

    return sanitized_markers


def sanitize_state_dictionary(loaded_dict: dict, reference_dict: dict) -> dict:
    """
    Sanitizes a loaded dictionary (parameters or controls) against a reference dictionary.
    Removes deprecated keys and initializes missing keys with default values.
    Ensures safe injection into PySide6 UI and CUDA/TensorRT worker pipelines.
    """
    sanitized_dict = {}
    if not isinstance(loaded_dict, dict):
        return copy.deepcopy(reference_dict)

    for key, default_value in reference_dict.items():
        if key in loaded_dict:
            sanitized_dict[key] = loaded_dict[key]
        else:
            sanitized_dict[key] = default_value

    obsolete_keys = set(loaded_dict.keys()) - set(reference_dict.keys())
    if obsolete_keys:
        print(
            f"[INFO] Sanitization: Removed obsolete keys during load: {obsolete_keys}"
        )

    return sanitized_dict


def _get_clamped_window_geometry(
    main_window: "MainWindow", x: int, y: int, width: int, height: int
) -> QtCore.QRect:
    app = QtWidgets.QApplication.instance()
    screens = app.screens() if app else []
    if not screens:
        return QtCore.QRect(x, y, width, height)

    saved_rect = QtCore.QRect(x, y, width, height)
    saved_center = saved_rect.center()

    target_screen = None
    for screen in screens:
        if screen.availableGeometry().contains(saved_center):
            target_screen = screen
            break

    if target_screen is None:
        target_screen = app.primaryScreen() or screens[0]

    available = target_screen.availableGeometry()
    clamped_width = min(max(1, width), available.width())
    clamped_height = min(max(1, height), available.height())
    max_x = available.x() + available.width() - clamped_width
    max_y = available.y() + available.height() - clamped_height
    clamped_x = min(max(x, available.x()), max_x)
    clamped_y = min(max(y, available.y()), max_y)

    return QtCore.QRect(clamped_x, clamped_y, clamped_width, clamped_height)


def _get_target_screen_for_rect(rect: QtCore.QRect):
    app = QtWidgets.QApplication.instance()
    screens = app.screens() if app else []
    if not screens:
        return None

    rect_center = rect.center()
    for screen in screens:
        if screen.availableGeometry().contains(rect_center):
            return screen

    return app.primaryScreen() or screens[0]


def _clamp_window_frame_to_available_geometry(main_window: "MainWindow"):
    frame_rect = main_window.frameGeometry()
    if not frame_rect.isValid():
        return

    target_screen = _get_target_screen_for_rect(frame_rect)
    if target_screen is None:
        return

    available = target_screen.availableGeometry()
    if (
        frame_rect.width() > available.width()
        or frame_rect.height() > available.height()
    ):
        excess_width = max(frame_rect.width() - available.width(), 0)
        excess_height = max(frame_rect.height() - available.height(), 0)
        new_width = max(1, main_window.width() - excess_width)
        new_height = max(1, main_window.height() - excess_height)
        if new_width != main_window.width() or new_height != main_window.height():
            main_window.resize(new_width, new_height)
            frame_rect = main_window.frameGeometry()

    max_x = available.x() + available.width() - frame_rect.width()
    max_y = available.y() + available.height() - frame_rect.height()
    clamped_x = min(max(frame_rect.x(), available.x()), max_x)
    clamped_y = min(max(frame_rect.y(), available.y()), max_y)

    if clamped_x != frame_rect.x() or clamped_y != frame_rect.y():
        main_window.move(clamped_x, clamped_y)


def _apply_workspace_window_state(
    main_window: "MainWindow", window_state: dict
) -> bool:
    is_maximized = window_state.get("isMaximized", False)

    if is_maximized:
        main_window.resize(main_window.sizeHint())
        main_window.showMaximized()
        main_window.menuBar().show()
        return False

    restored_rect = _get_clamped_window_geometry(
        main_window,
        window_state.get("x", main_window.x()),
        window_state.get("y", main_window.y()),
        window_state.get("width", main_window.width()),
        window_state.get("height", main_window.height()),
    )
    main_window.setGeometry(restored_rect)
    main_window.menuBar().show()
    return True


def open_embeddings_from_file(main_window: "MainWindow"):
    if video_control_actions.block_if_issue_scan_active(main_window, "load embeddings"):
        return

    embedding_filename, _ = QtWidgets.QFileDialog.getOpenFileName(
        main_window,
        filter="JSON (*.json)",
        dir=misc_helpers.get_dir_of_file(main_window.loaded_embedding_filename),
    )
    if embedding_filename:
        try:
            with open(embedding_filename, "r", encoding="utf-8") as embed_file:
                embeddings_list = json.load(embed_file)
                card_actions.clear_merged_embeddings(main_window)

                for _, target_face in main_window.target_faces.items():
                    target_face.assigned_merged_embeddings = {}
                    target_face.assigned_input_embedding = {}

                for embed_data in embeddings_list:
                    embedding_store = embed_data.get("embedding_store", {})
                    for recogn_model, embed in embedding_store.items():
                        embedding_store[recogn_model] = np.array(embed)

                    embedding_id = str(uuid.uuid1().int)

                    list_view_actions.create_and_add_embed_button_to_list(
                        main_window,
                        embed_data["name"],
                        embedding_store,
                        embedding_id=embedding_id,
                        tags=embed_data.get("tags"),
                    )

                    if embedding_id in main_window.merged_embeddings:
                        embed_button = main_window.merged_embeddings[embedding_id]
                        kv_map_path = embed_data.get("kv_map")
                        if kv_map_path and os.path.exists(kv_map_path):
                            try:
                                payload = torch.load(
                                    kv_map_path, map_location="cpu", weights_only=False
                                )
                                remember_loaded_kv_payload(main_window, payload, kv_map_path)
                                if isinstance(payload, dict):
                                    if "kv_map_list" in payload:
                                        embed_button.kv_map_list = payload[
                                            "kv_map_list"
                                        ]
                                    else:
                                        embed_button.kv_map = payload.get("kv_map")
                                else:
                                    embed_button.kv_map = payload
                                print(
                                    f"[INFO] Restored standalone K/V map for imported embedding: {embed_data['name']}"
                                )
                                _update_registry(
                                    main_window,
                                    Path(kv_map_path).name,
                                    str(embedding_filename),
                                    "reference_kv_data",
                                )
                            except Exception as e:
                                print(
                                    f"[ERROR] Error loading K/V map for imported embedding from {kv_map_path}: {e}"
                                )

        except (json.JSONDecodeError, KeyError, TypeError, Exception) as e:
            QtWidgets.QMessageBox.critical(
                main_window, "Error", f"Failed to load embeddings: {e}"
            )
            return

    main_window.loaded_embedding_filename = (
        embedding_filename or main_window.loaded_embedding_filename
    )

    tabs = getattr(main_window, "embeddingTabs", None)
    states = getattr(main_window, "embedding_tab_states", None)
    if tabs is not None and states:
        idx = tabs.currentIndex()
        if 0 <= idx < len(states):
            states[idx]["embeddings"] = main_window.merged_embeddings
            states[idx]["filename"] = main_window.loaded_embedding_filename or ""
            states[idx]["list_widget"] = main_window.inputEmbeddingsList
            list_view_actions.update_active_embedding_tab_title(main_window)


def load_embeddings_into_current_tab(
    main_window: "MainWindow", embedding_filename: str
) -> bool:
    """Load an embeddings JSON into the active tab (no file dialog)."""
    if not embedding_filename or not os.path.isfile(embedding_filename):
        return False

    try:
        with open(embedding_filename, "r", encoding="utf-8") as embed_file:
            embeddings_list = json.load(embed_file)
            card_actions.clear_merged_embeddings(main_window)

            for embed_data in embeddings_list:
                embedding_store = embed_data.get("embedding_store", {})
                for recogn_model, embed in embedding_store.items():
                    embedding_store[recogn_model] = np.array(embed)

                embedding_id = str(uuid.uuid1().int)
                list_view_actions.create_and_add_embed_button_to_list(
                    main_window,
                    embed_data["name"],
                    embedding_store,
                    embedding_id=embedding_id,
                    tags=embed_data.get("tags"),
                )

                if embedding_id in main_window.merged_embeddings:
                    embed_button = main_window.merged_embeddings[embedding_id]
                    kv_map_path = embed_data.get("kv_map")
                    if kv_map_path and os.path.exists(kv_map_path):
                        try:
                            payload = torch.load(
                                kv_map_path, map_location="cpu", weights_only=False
                            )
                            remember_loaded_kv_payload(main_window, payload, kv_map_path)
                            if isinstance(payload, dict):
                                if "kv_map_list" in payload:
                                    embed_button.kv_map_list = payload["kv_map_list"]
                                else:
                                    embed_button.kv_map = payload.get("kv_map")
                            else:
                                embed_button.kv_map = payload
                        except Exception as e:
                            print(
                                f"[ERROR] Error loading K/V map for "
                                f"{embed_data.get('name')}: {e}"
                            )

        main_window.loaded_embedding_filename = embedding_filename

        tabs = getattr(main_window, "embeddingTabs", None)
        states = getattr(main_window, "embedding_tab_states", None)
        if tabs is not None and states:
            idx = tabs.currentIndex()
            if 0 <= idx < len(states):
                states[idx]["embeddings"] = main_window.merged_embeddings
                states[idx]["filename"] = embedding_filename
                states[idx]["list_widget"] = main_window.inputEmbeddingsList
                list_view_actions.update_active_embedding_tab_title(main_window)
        return True
    except Exception as e:
        print(f"[ERROR] Failed to load embeddings from {embedding_filename}: {e}")
        return False


def save_embeddings_to_file(main_window: "MainWindow", save_as=False):
    if not main_window.merged_embeddings:
        common_widget_actions.create_and_show_messagebox(
            main_window,
            "Embeddings List Empty!",
            "No Embeddings available to save",
            parent_widget=main_window,
        )
        return

    embedding_filename = main_window.loaded_embedding_filename
    if (
        not embedding_filename
        or not misc_helpers.is_file_exists(embedding_filename)
        or save_as
    ):
        embedding_filename, _ = QtWidgets.QFileDialog.getSaveFileName(
            main_window, filter="JSON (*.json)"
        )
        if not embedding_filename:
            return
    elif (
        QtWidgets.QMessageBox.question(
            main_window,
            "Confirm Save",
            (
                "Save all embeddings to the current file?\n\n"
                f"This will overwrite:\n{embedding_filename}"
            ),
            QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.No,
            QtWidgets.QMessageBox.No,
        )
        != QtWidgets.QMessageBox.Yes
    ):
        return

    embeddings_list = []
    for embedding_id, embed_button in main_window.merged_embeddings.items():
        kv_map_path = None
        kv_payload_to_save = getattr(embed_button, "kv_map_list", None) or getattr(
            embed_button, "kv_map", None
        )

        if kv_payload_to_save is not None:
            try:
                payload = (
                    {"kv_map_list": kv_payload_to_save}
                    if getattr(embed_button, "kv_map_list", None)
                    else {"kv_map": kv_payload_to_save}
                )
                kv_map_path = _save_hashed_kv_payload(
                    main_window,
                    payload,
                    "reference_kv_data",
                    parent_file_paths=str(embedding_filename),
                )
            except Exception as e:
                print(
                    f"[ERROR] Error saving hashed K/V map for embedding {embedding_id}: {e}"
                )

        embedding_entry = {
            "name": embed_button.embedding_name,
            "embedding_store": {
                k: v.tolist() for k, v in embed_button.embedding_store.items()
            },
            "kv_map": kv_map_path,
        }
        if getattr(embed_button, "tags", None):
            embedding_entry["tags"] = list(embed_button.tags)
        embeddings_list.append(embedding_entry)

    if embedding_filename:
        with open(embedding_filename, "w", encoding="utf-8") as embed_file:
            embeddings_as_json = json.dumps(embeddings_list, indent=4)
            embed_file.write(embeddings_as_json)

            common_widget_actions.create_and_show_toast_message(
                main_window,
                "Embeddings Saved",
                f"Saved Embeddings to file: {embedding_filename}",
            )

        main_window.loaded_embedding_filename = embedding_filename


def convert_parameters_to_supported_type(
    main_window: "MainWindow",
    parameters: Union[dict, ParametersTypes],
    convert_type: type,
):
    if convert_type is dict:
        if isinstance(parameters, misc_helpers.ParametersDict):
            return parameters.data
    elif convert_type is misc_helpers.ParametersDict:
        if isinstance(parameters, dict):
            return misc_helpers.ParametersDict(
                parameters,
                cast(misc_helpers.ParametersDict, main_window.default_parameters).data,
            )
    return parameters


def convert_markers_to_supported_type(
    main_window: "MainWindow",
    markers: MarkerTypes,
    convert_type: type,
):
    for _, marker_data in markers.items():
        if "parameters" in marker_data:
            for target_face_id, target_parameters in marker_data["parameters"].items():
                marker_data["parameters"][target_face_id] = (
                    convert_parameters_to_supported_type(
                        main_window, target_parameters, convert_type
                    )
                )
    return markers


def save_current_parameters_and_control(main_window: "MainWindow", face_id):
    data_filename, _ = QtWidgets.QFileDialog.getSaveFileName(
        main_window, filter="JSON (*.json)"
    )

    default_params = cast(
        misc_helpers.ParametersDict, main_window.default_parameters
    ).data
    raw_params = convert_parameters_to_supported_type(
        main_window, main_window.parameters[face_id], dict
    )

    data = {
        "parameters": sanitize_state_dictionary(raw_params, default_params),
        "control": sanitize_state_dictionary(
            main_window.control.copy(), main_window.control
        ),
    }

    if data_filename:
        with open(data_filename, "w") as data_file:  # pylint: disable=unspecified-encoding
            data_as_json = json.dumps(data, indent=4)
            data_file.write(data_as_json)


def load_parameters_and_settings(
    main_window: "MainWindow", face_id, load_settings=False
):
    if video_control_actions.block_if_issue_scan_active(
        main_window,
        "load parameters and settings" if load_settings else "load parameters",
    ):
        return

    data_filename, _ = QtWidgets.QFileDialog.getOpenFileName(
        main_window, filter="JSON (*.json)"
    )
    if data_filename:
        with open(data_filename, "r") as data_file:  # pylint: disable=unspecified-encoding
            data = json.load(data_file)

            default_params = cast(
                misc_helpers.ParametersDict, main_window.default_parameters
            ).data
            sanitized_params = sanitize_state_dictionary(
                data.get("parameters", {}), default_params
            )

            main_window.parameters[face_id] = convert_parameters_to_supported_type(
                main_window, sanitized_params, misc_helpers.ParametersDict
            )

            if main_window.selected_target_face_id == face_id:
                common_widget_actions.set_widgets_values_using_face_id_parameters(
                    main_window, face_id
                )
            if load_settings:
                sanitized_control = sanitize_state_dictionary(
                    data.get("control", {}), main_window.control
                )

                main_window.control.update(sanitized_control)
                common_widget_actions.set_control_widgets_values(main_window)
            common_widget_actions.refresh_frame(main_window)


def _salvage_json_object(text: str) -> dict:
    """Recover every complete top-level key/value pair from a damaged JSON object."""
    decoder = json.JSONDecoder()
    out = {}
    i = text.find("{")
    if i < 0:
        return out
    i += 1
    n = len(text)
    ws = " \t\r\n"
    while i < n:
        while i < n and text[i] in ws + ",":
            i += 1
        if i >= n or text[i] == "}":
            break
        try:
            key, i = decoder.raw_decode(text, i)
            if not isinstance(key, str):
                break
            while i < n and text[i] in ws:
                i += 1
            if i >= n or text[i] != ":":
                break
            i += 1
            while i < n and text[i] in ws:
                i += 1
            value, i = decoder.raw_decode(text, i)
        except ValueError:
            break
        out[key] = value
    return out


def _read_workspace_candidate(path):
    """Returns (data, salvaged). Never modifies, renames or deletes anything."""
    try:
        with open(path, "r", encoding="utf-8") as f:
            text = f.read()
    except (OSError, UnicodeDecodeError):
        return None, False
    try:
        data = json.loads(text)
        if isinstance(data, dict):
            return data, False
        return None, False
    except ValueError:
        data = _salvage_json_object(text)
        return (data, True) if data else (None, False)


def _pick_workspace_data(data_filename: str):
    """Best readable data from the workspace file or its backups: (data, source, salvaged)."""
    candidates = (data_filename, f"{data_filename}.bak", f"{data_filename}.bak2")
    best_salvage = None
    for cand in candidates:
        if not os.path.isfile(cand):
            continue
        data, salvaged = _read_workspace_candidate(cand)
        if data is None:
            continue
        if not salvaged:
            return data, cand, False
        if best_salvage is None or len(data) > len(best_salvage[0]):
            best_salvage = (data, cand, True)
    if best_salvage:
        return best_salvage
    return None, None, False


def get_auto_load_workspace_toggle(
    main_window: "MainWindow", data_filename: str | bool = False
):
    if not data_filename:
        data_filename, _ = QtWidgets.QFileDialog.getOpenFileName(
            main_window, filter="JSON (*.json)"
        )
    if not isinstance(data_filename, str) or not data_filename:
        return False
    data, _src, _salv = _pick_workspace_data(data_filename)
    if not isinstance(data, dict):
        return False
    control = data.get("control", {})
    if not isinstance(control, dict):
        return False
    return control.get("AutoLoadWorkspaceToggle", False)


def _record_failure(main_window, label: str, keys=()):
    import traceback

    traceback.print_exc()
    main_window._workspace_failed_keys.update(keys)
    main_window._workspace_load_errors.append(f"{label}: {traceback.format_exc().strip().splitlines()[-1]}")


def _safe(main_window, label: str, fn, keys=()) -> bool:
    """Run one part of the load; a failure is recorded and the load goes on."""
    try:
        fn()
        return True
    except Exception:
        _record_failure(main_window, label, keys)
        return False


def _guard(main_window, label: str, keys, fn):
    def _runner():
        _safe(main_window, label, fn, keys)

    return _runner


def _schedule(main_window, fn, delay_ms: int = 0):
    QtCore.QTimer.singleShot(delay_ms, fn)


def load_saved_workspace(main_window, data_filename=False):
    if video_control_actions.block_if_issue_scan_active(main_window, "load a workspace"):
        return
    if not data_filename:
        data_filename, _ = QtWidgets.QFileDialog.getOpenFileName(
            main_window, filter="JSON (*.json)"
        )
    if not isinstance(data_filename, str) or not data_filename:
        return
    bak_path = f"{data_filename}.bak"
    bak2_path = f"{data_filename}.bak2"
    if not (
        Path(data_filename).is_file()
        or Path(bak_path).is_file()
        or Path(bak2_path).is_file()
    ):
        return

    data, source, salvaged = _pick_workspace_data(data_filename)
    if data is None:
        # Nothing readable. The files are left exactly as they are and saving over
        # them is blocked, so nothing is lost and nothing is reset.
        main_window._workspace_load_failed = True
        QtWidgets.QMessageBox.critical(
            main_window,
            "Error",
            "The workspace file could not be read.\n\n"
            f"It was left untouched:\n{data_filename}\n\n"
            "The app will not overwrite it on close.",
        )
        _notify_workspace_load_done(main_window)
        return

    if source != data_filename or salvaged:
        print(
            f"[WARN] Workspace read from {source}"
            + (" (damaged file, recovered every readable setting)" if salvaged else "")
        )
        # Keep a copy of the damaged file before a later save replaces it.
        try:
            if os.path.isfile(data_filename):
                shutil.copy2(data_filename, f"{data_filename}.damaged")
        except OSError as e:
            print(f"[WARN] Could not copy damaged workspace: {e}")

    main_window._workspace_load_failed = False
    main_window._workspace_original_data = copy.deepcopy(data)
    main_window._workspace_failed_keys = set()
    main_window._workspace_load_errors = []
    _workspace_load_begin(main_window, data, data_filename)


def _workspace_load_begin(main_window, data, data_filename):
    main_window._workspace_load_in_progress = True

    for label, fn in (
        ("stop input loading", lambda: list_view_actions.clear_stop_loading_input_media(main_window)),
        ("stop target loading", lambda: list_view_actions.clear_stop_loading_target_media(main_window)),
        ("clear target videos", lambda: setattr(main_window, "target_videos", {})),
        ("clear input faces", lambda: card_actions.clear_input_faces(main_window)),
        ("clear target faces", lambda: card_actions.clear_target_faces(main_window)),
        ("reset embedding tabs", lambda: list_view_actions.reset_embedding_tabs(main_window)),
    ):
        _safe(main_window, label, fn)

    def _apply_control():
        loaded = data.get("control", {})
        if not isinstance(loaded, dict):
            raise ValueError("'control' is not an object")
        for control_name in list(main_window.control.keys()):
            if control_name in loaded:
                main_window.control[control_name] = loaded[control_name]

    _safe(main_window, "control settings", _apply_control, ("control",))

    def _apply_current_params():
        # Must happen BEFORE the target media is clicked below: new target faces
        # copy their parameters from current_widget_parameters. Otherwise they are
        # created from defaults and the next save writes those defaults.
        raw = data.get("current_widget_parameters")
        if not isinstance(raw, dict) or not raw:
            return
        defaults = main_window.default_parameters
        default_data = defaults.data if hasattr(defaults, "data") else defaults
        sanitized = sanitize_state_dictionary(raw, default_data)
        main_window.current_widget_parameters = misc_helpers.ParametersDict(
            sanitized, default_data
        )

    _safe(main_window, "saved face parameters", _apply_current_params,
          ("current_widget_parameters",))

    def _start_target_loader():
        raw = data.get("target_medias_data", [])
        if not isinstance(raw, list):
            raise ValueError("'target_medias_data' is not a list")
        files, ids = [], []
        for m in raw:
            try:
                files.append(m["media_path"])
                ids.append(m["media_id"])
            except (KeyError, TypeError):
                main_window._workspace_failed_keys.add("target_medias_data")
                continue

        from app.ui.widgets.actions import target_videos_list_actions
        from app.ui.widgets import ui_workers

        main_window.video_loader_worker = ui_workers.TargetMediaLoaderWorker(
            main_window=main_window,
            folder_name=False,
            files_list=files,
            media_ids=ids,
            sort_files_list_by_name=False,
            metadata_enabled=target_videos_list_actions.current_sort_needs_metadata(main_window),
        )
        main_window.video_loader_worker.thumbnail_ready.connect(
            partial(list_view_actions.add_media_thumbnail_to_target_videos_list, main_window),
            QtCore.Qt.ConnectionType.QueuedConnection,
        )
        main_window.video_loader_worker.start()

    started = _safe(main_window, "target media", _start_target_loader, ("target_medias_data",))
    if started:
        _schedule(
            main_window,
            partial(_poll_target_thumbnails_then_continue, main_window, data, data_filename),
            50,
        )
    else:
        main_window.video_loader_worker = None
        _schedule(main_window, partial(_workspace_after_targets, main_window, data, data_filename))


def _poll_target_thumbnails_then_continue(main_window, data, data_filename):
    try:
        worker = main_window.video_loader_worker
        worker_running = worker is not None and worker.isRunning()
        pending = list_view_actions._has_pending_target_media_thumbnail_work(main_window)
    except Exception:
        _record_failure(main_window, "waiting for target media", ())
        worker_running = pending = False
    if worker_running or pending:
        _schedule(
            main_window,
            partial(_poll_target_thumbnails_then_continue, main_window, data, data_filename),
            50,
        )
        return
    _workspace_after_targets(main_window, data, data_filename)


def _workspace_after_targets(main_window, data, data_filename):
    def _select_media():
        selected_media_id = data.get("selected_media_id", False)
        if selected_media_id and main_window.target_videos.get(selected_media_id):
            main_window.target_videos[selected_media_id].click()

    _safe(main_window, "selected target media", _select_media, ("selected_media_id",))

    def _start_input_loader():
        raw = data.get("input_faces_data", {})
        if not isinstance(raw, dict):
            raise ValueError("'input_faces_data' is not an object")
        paths, ids = [], []
        for face_id, input_face_data in raw.items():
            try:
                paths.append(input_face_data["media_path"])
                ids.append(face_id)
            except (KeyError, TypeError):
                main_window._workspace_failed_keys.add("input_faces_data")
                continue

        from app.ui.widgets import ui_workers

        main_window.input_faces_loader_worker = ui_workers.InputFacesLoaderWorker(
            main_window=main_window,
            folder_name=False,
            files_list=paths,
            face_ids=ids,
        )
        main_window.input_faces_loader_worker.thumbnail_ready.connect(
            partial(list_view_actions.add_media_thumbnail_to_source_faces_list, main_window),
            QtCore.Qt.ConnectionType.QueuedConnection,
        )
        main_window.input_faces_loader_worker.finished.connect(
            partial(common_widget_actions.refresh_frame, main_window)
        )
        main_window.input_faces_loader_worker.start()

    started = _safe(main_window, "input faces", _start_input_loader, ("input_faces_data",))
    if started:
        _schedule(
            main_window,
            partial(_poll_input_thumbnails_then_continue, main_window, data, data_filename),
            50,
        )
    else:
        main_window.input_faces_loader_worker = None
        _schedule(main_window, partial(_workspace_after_inputs, main_window, data, data_filename))


def _poll_input_thumbnails_then_continue(main_window, data, data_filename):
    try:
        worker = main_window.input_faces_loader_worker
        worker_running = worker is not None and worker.isRunning()
        pending = bool(getattr(main_window, "_pending_input_face_thumbnails", None))
        timer = getattr(main_window, "_input_face_batch_timer", None)
        timer_active = timer is not None and timer.isActive()
    except Exception:
        _record_failure(main_window, "waiting for input faces", ())
        worker_running = pending = timer_active = False
    if worker_running or pending or timer_active:
        _schedule(
            main_window,
            partial(_poll_input_thumbnails_then_continue, main_window, data, data_filename),
            50,
        )
        return
    _workspace_after_inputs(main_window, data, data_filename)


def _workspace_after_inputs(main_window, data, data_filename):
    _safe(
        main_window,
        "input faces placeholder",
        lambda: main_window.placeholder_update_signal.emit(main_window.inputFacesList, False),
    )

    steps = []

    def _add(label, keys, fn):
        steps.append(_guard(main_window, label, keys, fn))

    # ---- input face K/V maps (one guarded step per face) ----
    def _load_input_kv(face_id, input_face_data):
        input_face_button = main_window.input_faces.get(face_id)
        if input_face_button is None:
            return
        kv_map_path = input_face_data.get("kv_map")
        media_path = input_face_data.get("media_path")
        if not (kv_map_path and os.path.exists(kv_map_path)):
            return
        payload = torch.load(kv_map_path, map_location="cpu", weights_only=False)
        remember_loaded_kv_payload(main_window, payload, kv_map_path)
        input_face_button.kv_map = (
            payload.get("kv_map") if isinstance(payload, dict) else payload
        )
        parents = [str(data_filename)]
        if media_path:
            parents.append(str(media_path))
        _update_registry(main_window, Path(kv_map_path).name, parents, "reference_kv_data")

    raw_inputs = data.get("input_faces_data", {})
    if isinstance(raw_inputs, dict):
        for face_id, input_face_data in raw_inputs.items():
            if isinstance(input_face_data, dict):
                _add(f"input face K/V {face_id}", ("input_faces_data",),
                     partial(_load_input_kv, face_id, input_face_data))

    # ---- embeddings ----
    def _load_embedding(embedding_id, embedding_data):
        embedding_store_loaded = {
            m: np.array(e) for m, e in embedding_data["embedding_store"].items()
        }
        embedding_name = embedding_data.get(
            "embedding_name", embedding_data.get("name", "Unknown")
        )
        list_view_actions.create_and_add_embed_button_to_list(
            main_window,
            embedding_name,
            embedding_store_loaded,
            embedding_id=embedding_id,
            tags=embedding_data.get("tags"),
        )
        embed_button = main_window.merged_embeddings.get(embedding_id)
        if embed_button is None:
            return
        kv_map_path = embedding_data.get("kv_map")
        if not (kv_map_path and os.path.exists(kv_map_path)):
            return
        try:
            payload = torch.load(kv_map_path, map_location="cpu", weights_only=False)
            remember_loaded_kv_payload(main_window, payload, kv_map_path)
            if isinstance(payload, dict):
                if "kv_map_list" in payload:
                    embed_button.kv_map_list = payload["kv_map_list"]
                else:
                    embed_button.kv_map = payload.get("kv_map")
            else:
                embed_button.kv_map = payload
            _update_registry(
                main_window, Path(kv_map_path).name, str(data_filename), "reference_kv_data"
            )
        except Exception as e:
            print(f"[ERROR] Error loading K/V map for embedding from {kv_map_path}: {e}")

    saved_tabs_state = data.get("embedding_tabs_state")
    external_tabs_file = (
        saved_tabs_state.get("external_file") if isinstance(saved_tabs_state, dict) else None
    )
    embedding_keys = ("embeddings_data", "embedding_tabs_state")
    if external_tabs_file:
        _add("embedding tabs file", embedding_keys,
             partial(_restore_embedding_tabs_from_file, main_window, data_filename, saved_tabs_state))
    else:
        raw_embeddings = data.get("embeddings_data", {})
        if isinstance(raw_embeddings, dict):
            for embedding_id, embedding_data in raw_embeddings.items():
                _add(f"embedding {embedding_id}", embedding_keys,
                     partial(_load_embedding, embedding_id, embedding_data))
        _add("embedding tabs", embedding_keys,
             partial(list_view_actions.restore_embedding_tabs_state, main_window,
                     saved_tabs_state if isinstance(saved_tabs_state, dict) else {}))

    # ---- segments / markers / issue frames (each part isolated) ----
    default_params = main_window.default_parameters

    def _restore_segments():
        loaded_segments = data.get("segments")
        if not isinstance(loaded_segments, list):
            loaded_segments = []
            for pair in data.get("job_marker_pairs", []) or []:
                try:
                    s, e = int(pair[0]), int(pair[1])
                except (TypeError, ValueError, IndexError):
                    continue
                if s < e:
                    loaded_segments.append({"start": s, "end": e, "kind": "normal"})
            if not loaded_segments:
                s = data.get("job_start_frame", None)
                e = data.get("job_end_frame", None)
                if s is not None and e is not None and int(s) < int(e):
                    loaded_segments = [{"start": int(s), "end": int(e), "kind": "normal"}]
        normalized_segments = []
        for s in loaded_segments:
            try:
                entry = {
                    "start": int(s["start"]),
                    "end": int(s["end"]),
                    "kind": s.get("kind", "normal"),
                    "parameters": None,
                    "control": s.get("control"),
                }
                raw_params = s.get("parameters")
                if raw_params is not None:
                    entry["parameters"] = {
                        str(fid): misc_helpers.ParametersDict(
                            pd if isinstance(pd, dict) else dict(pd), default_params.data
                        )
                        for fid, pd in raw_params.items()
                    }
                normalized_segments.append(entry)
            except Exception:
                _record_failure(main_window, "one segment", ("segments",))
        main_window.segments = normalized_segments
        main_window._in_normal_segment = False
        main_window._segments_base_parameters = main_window.parameters
        main_window._segments_base_control = main_window.control

    def _restore_markers():
        video_control_actions.remove_all_markers(main_window)
        raw_markers = data.get("markers", {})
        if not isinstance(raw_markers, dict):
            raise ValueError("'markers' is not an object")
        for marker_position, marker_data in raw_markers.items():
            try:
                single = sanitize_markers_dictionary(main_window, {marker_position: marker_data})
                single = convert_markers_to_supported_type(
                    main_window, cast(MarkerTypes, single), misc_helpers.ParametersDict
                )
                md = single[marker_position]
                video_control_actions.add_marker(
                    main_window, md["parameters"], md["control"], int(marker_position)
                )
            except Exception:
                _record_failure(main_window, f"marker {marker_position}", ("markers",))

    def _restore_issue_frames():
        loaded = data.get("issue_frames_by_face")
        if loaded is not None:
            video_control_actions.set_issue_frames_by_face(main_window, loaded)
        else:
            selected_face_id = getattr(main_window, "selected_target_face_id", None)
            if selected_face_id is None and getattr(main_window, "target_faces", {}):
                selected_face_id = str(next(iter(main_window.target_faces.keys())))
            if selected_face_id is not None:
                video_control_actions.set_issue_frames_for_face(
                    main_window, selected_face_id, data.get("issue_frames", [])
                )
            else:
                video_control_actions.set_issue_frames_by_face(main_window, {})

    def _restore_dropped_frames():
        video_control_actions.set_dropped_frames(main_window, data.get("dropped_frames", []))
        main_window.videoSeekSlider.update()
        video_control_actions.update_drop_frame_button_label(main_window)

    def _restore_folder_paths():
        main_window.last_target_media_folder_path = data.get("last_target_media_folder_path", "")
        main_window.last_input_media_folder_path = data.get("last_input_media_folder_path", "")
        main_window.targetVideosPathLineEdit.setText(main_window.last_target_media_folder_path)
        main_window.targetVideosPathLineEdit.setToolTip(main_window.last_target_media_folder_path)
        main_window.inputFacesPathLineEdit.setText(main_window.last_input_media_folder_path)
        main_window.inputFacesPathLineEdit.setToolTip(main_window.last_input_media_folder_path)
        if not data.get("embedding_tabs_state"):
            main_window.loaded_embedding_filename = data.get("loaded_embedding_filename", "")

    def _restore_control_widgets():
        common_widget_actions.set_control_widgets_values(main_window)

    def _restore_output_folder():
        output_folder = main_window.control.get("OutputMediaFolder", "")
        common_widget_actions.create_control(main_window, "OutputMediaFolder", output_folder)
        main_window.outputFolderLineEdit.setText(output_folder)

    def _restore_auto_watch():
        list_view_actions.set_target_folder_auto_watch(
            main_window, bool(main_window.control.get("AutoLoadTargetFolderToggle", False))
        )

    _add("segments", ("segments", "job_marker_pairs"), _restore_segments)
    _add("markers", ("markers",), _restore_markers)
    _add("issue frames", ("issue_frames_by_face", "issue_frames"), _restore_issue_frames)
    _add("dropped frames", ("dropped_frames",), _restore_dropped_frames)
    _add("folder paths", ("last_target_media_folder_path", "last_input_media_folder_path", "loaded_embedding_filename"), _restore_folder_paths)
    _add("control widgets", ("control",), _restore_control_widgets)
    _add("output folder", ("control",), _restore_output_folder)
    _add("auto watch", (), _restore_auto_watch)

    # ---- tab order / face selection / current parameters (isolated) ----
    def _restore_tab_order():
        tab_state = data.get("tab_state")
        if not isinstance(tab_state, dict):
            return
        tab_texts = {main_window.tabWidget.tabText(i): i for i in range(main_window.tabWidget.count())}
        for i, tab_info in enumerate(tab_state.get("tab_order", [])):
            tab_text = tab_info["text"]
            if tab_text in tab_texts:
                current_index = tab_texts[tab_text]
                if current_index != i:
                    main_window.tabWidget.tabBar().moveTab(current_index, i)
                    tab_texts = {main_window.tabWidget.tabText(j): j for j in range(main_window.tabWidget.count())}
        if "current_tab_index" in tab_state:
            main_window.tabWidget.setCurrentIndex(tab_state["current_tab_index"])

    def _restore_face_selection():
        if main_window.target_faces:
            saved_face_id = data.get("selected_target_face_id")
            first_face_id = (
                saved_face_id
                if saved_face_id in main_window.target_faces
                else list(main_window.target_faces.keys())[0]
            )
            main_window.selected_target_face_id = first_face_id
            first_face_button = main_window.target_faces.get(first_face_id)
            if first_face_button:
                first_face_button.setChecked(True)
                main_window.cur_selected_target_face_button = first_face_button
                for target_face_id, target_face_button in main_window.target_faces.items():
                    if target_face_id != first_face_id:
                        target_face_button.setChecked(False)
                card_actions.uncheck_all_input_faces(main_window)
                card_actions.uncheck_all_merged_embeddings(main_window)
                for input_face_id in first_face_button.assigned_input_faces.keys():
                    btn = main_window.input_faces.get(input_face_id)
                    if btn:
                        btn.setChecked(True)
                for embedding_id in first_face_button.assigned_merged_embeddings.keys():
                    btn = main_window.merged_embeddings.get(embedding_id)
                    if btn:
                        btn.setChecked(True)
                main_window.current_kv_tensors_map = getattr(first_face_button, "assigned_kv_map", None)
            video_control_actions.refresh_issue_frames_for_selected_face(main_window)
            video_control_actions.update_scan_review_button_states(main_window)
            common_widget_actions.set_widgets_values_using_face_id_parameters(main_window, face_id=first_face_id)
            main_window.current_widget_parameters = main_window.parameters[first_face_id].copy()
        else:
            raw_current_params = data.get("current_widget_parameters", default_params.data.copy())
            sanitized_current_params = sanitize_state_dictionary(raw_current_params, default_params.data)
            main_window.current_widget_parameters = cast(
                ParametersTypes,
                misc_helpers.ParametersDict(sanitized_current_params, default_params.data),
            )
            common_widget_actions.set_widgets_values_using_face_id_parameters(main_window, face_id=None)

    def _restore_edit_buttons():
        main_window.swapfacesButton.setChecked(data.get("swap_faces_enabled", False))
        main_window.editFacesButton.setChecked(data.get("edit_faces_enabled", False))
        control_actions.handle_face_editor_button_click(main_window)

    _add("tab order", ("tab_state",), _restore_tab_order)
    _add("face selection / parameters", ("current_widget_parameters",), _restore_face_selection)
    _add("swap/edit buttons", ("swap_faces_enabled", "edit_faces_enabled"), _restore_edit_buttons)

    # ---- window state (every part isolated) ----
    window_state = data.get("window_state_data", {})
    if not isinstance(window_state, dict):
        window_state = {}
    wkeys = ("window_state_data",)
    clamp_needed = {"v": False}

    def _ws_geometry():
        clamp_needed["v"] = _apply_workspace_window_state(main_window, window_state)

    def _ws_panels():
        panel_state_map = {
            "target_media": window_state.get("target_media", window_state.get("TargetMediaCheckBox", True)),
            "input_faces": window_state.get("input_faces", window_state.get("InputFacesCheckBox", True)),
            "faces": window_state.get("faces", window_state.get("facesPanelCheckBox", True)),
            "parameters": window_state.get("parameters", window_state.get("parametersPanelCheckBox", True)),
        }
        for panel_key, visible in panel_state_map.items():
            main_window._set_panel_visibility(panel_key, visible)

    def _ws_face_thumbs():
        size = window_state.get("face_thumbnail_size")
        if size == "small":
            list_view_actions.apply_face_thumbnail_size(main_window, list_view_actions._SMALL_FACE_BUTTON_SIZE)
        elif size == "large":
            list_view_actions.apply_face_thumbnail_size(main_window, list_view_actions._LARGE_FACE_BUTTON_SIZE)
        else:
            list_view_actions.apply_face_thumbnail_size(main_window, list_view_actions._FACE_BUTTON_SIZE)

    def _ws_target_thumbs():
        saved = window_state.get("target_thumbnail_size")
        if saved:
            list_view_actions.apply_target_thumbnail_size(main_window, tuple(saved))

    def _ws_scan_tools():
        if hasattr(main_window, "scanToolsToggleButton"):
            video_control_actions.set_scan_tools_expanded(
                main_window, window_state.get("scan_tools_expanded", False)
            )

    def _ws_sections():
        states = None
        if "parameter_section_states" in window_state:
            states = window_state["parameter_section_states"]
        elif "parameter_section_states" in data:
            states = data["parameter_section_states"]
        main_window.apply_parameter_section_states(states)
        if isinstance(window_state.get("detached_sections"), dict):
            main_window.apply_detached_sections_state(window_state["detached_sections"])

    def _ws_filter():
        filter_actions.filter_target_videos(main_window)

    def _ws_dock():
        dock_state_str = window_state.get("dock_state", data.get("dock_state", ""))
        if dock_state_str:
            ba = QtCore.QByteArray.fromBase64(dock_state_str.encode("utf-8"))
            main_window.restoreState(ba)
            main_window._refresh_panel_visibility_state_from_widgets()

    def _ws_clamp():
        if clamp_needed["v"]:
            QtCore.QTimer.singleShot(0, partial(_clamp_window_frame_to_available_geometry, main_window))

    for label, fn in (
        ("window geometry", _ws_geometry), ("panel visibility", _ws_panels),
        ("face thumbnail size", _ws_face_thumbs), ("target thumbnail size", _ws_target_thumbs),
        ("scan tools", _ws_scan_tools), ("parameter sections", _ws_sections),
        ("target filter", _ws_filter), ("dock layout", _ws_dock), ("window clamp", _ws_clamp),
    ):
        _add(label, wkeys, fn)

    _run_workspace_steps(main_window, steps, 0)


def _restore_embedding_tabs_from_file(main_window, data_filename, tabs_state):
    # Only a plain file name is accepted: the file always sits next to the workspace.
    file_name = Path(str(tabs_state.get("external_file"))).name
    embedding_tabs_store.restore_tabs_from_file(
        main_window,
        Path(str(data_filename)).parent / file_name,
        data_filename,
        active_hint=tabs_state.get("active_index", 0),
    )


def _add_pending_tabs_inline(
    main_window, tabs_state: dict, embeddings_data: dict, parent_path: str
) -> None:
    """Inline workspaces must contain tabs that were never opened this session too."""
    states = getattr(main_window, "embedding_tab_states", None) or []
    saved_tabs = tabs_state.get("tabs", [])
    for i, state in enumerate(states):
        pending = state.get("pending")
        if not pending or i >= len(saved_tabs):
            continue
        try:
            entries = embedding_tabs_store.read_pending_entries(pending)
        except Exception as e:
            print(f"[ERROR] Could not read unopened embedding tab {i + 1}: {e}")
            continue
        ids = []
        for entry in entries:
            embedding_id = str(entry.get("id") or uuid.uuid1().int)
            if embedding_id in embeddings_data:
                embedding_id = str(uuid.uuid1().int)
            kv_map_path = entry.get("kv_map")
            inline_entry = {
                "embedding_name": entry.get("name", entry.get("embedding_name", "Unknown")),
                "embedding_store": entry.get("embedding_store") or {},
                "kv_map": kv_map_path,
            }
            if entry.get("tags"):
                inline_entry["tags"] = list(entry["tags"])
            embeddings_data[embedding_id] = inline_entry
            ids.append(embedding_id)
            if kv_map_path:
                _update_registry(
                    main_window, Path(kv_map_path).name, parent_path, "reference_kv_data"
                )
        saved_tabs[i]["embedding_ids"] = ids


def _run_workspace_steps(main_window, steps, index, errors=None):
    if index >= len(steps):
        errors = list(getattr(main_window, "_workspace_load_errors", []))
        if errors:
            print(f"[WARN] Workspace loaded with {len(errors)} failed part(s).")
        _notify_workspace_load_done(main_window)
        if errors:
            QtWidgets.QMessageBox.warning(
                main_window,
                "Workspace partially loaded",
                f"{len(errors)} part(s) of the workspace could not be restored "
                "(details in the console). Everything else was loaded.\n\n"
                "The parts that failed are kept exactly as they were in "
                "last_workspace.json; they will not be overwritten or reset.\n\n"
                f"First error:\n{errors[0]}",
            )
        return
    try:
        steps[index]()
    except Exception:
        _record_failure(main_window, f"step {index}", ())
    QtCore.QTimer.singleShot(
        0, partial(_run_workspace_steps, main_window, steps, index + 1)
    )


def save_current_workspace(
    main_window: "MainWindow", data_filename: str | Path | bool = False
):
    if data_filename is False:
        dialog_filename, _ = QtWidgets.QFileDialog.getSaveFileName(
            main_window, filter="JSON (*.json)"
        )
        if not dialog_filename:
            return
        data_filename = dialog_filename

    resolved_parent_path = str(data_filename)
    # last_workspace.json keeps embeddings out of the file (privacy / size): they
    # are written to a separate file, see embedding_tabs_store.
    use_tabs_file = str(data_filename).endswith("last_workspace.json")

    embeddings_data = {}
    input_faces_data = {}
    target_medias_data = []

    saved_is_maximized = bool(main_window.isMaximized())
    saved_geometry = main_window.geometry()

    try:
        dock_state_data = main_window.saveState().toBase64().data().decode("utf-8")
    except Exception:
        dock_state_data = ""

    window_state_data = {
        "x": saved_geometry.x(),
        "y": saved_geometry.y(),
        "height": saved_geometry.height(),
        "width": saved_geometry.width(),
        "isMaximized": saved_is_maximized,
        "target_media": main_window.panel_visibility_state.get("target_media", True),
        "input_faces": main_window.panel_visibility_state.get("input_faces", True),
        "faces": main_window.panel_visibility_state.get("faces", True),
        "parameters": main_window.panel_visibility_state.get("parameters", True),
        "face_thumbnail_size": (
            "small"
            if getattr(
                main_window,
                "face_thumbnail_button_size",
                list_view_actions._FACE_BUTTON_SIZE,
            )
            == list_view_actions._SMALL_FACE_BUTTON_SIZE
            else "large"
        ),
        "target_thumbnail_size": getattr(
            main_window, "target_thumbnail_button_size", (85, 85)
        ),
        "scan_tools_expanded": getattr(main_window, "scan_tools_expanded", False),
        "parameter_section_states": {
            section_id: bool(expanded)
            for section_id, expanded in getattr(
                main_window, "parameter_section_states", {}
            ).items()
        },
        "dock_state": dock_state_data,
        "detached_sections": main_window.get_detached_sections_state(),
    }

    control = main_window.control
    is_denoiser_enabled = (
        control.get("DenoiserUNetEnableBeforeRestorersToggle", False)
        or control.get("DenoiserAfterFirstRestorerToggle", False)
        or control.get("DenoiserAfterRestorersToggle", False)
    )

    for media_id, target_media in main_window.target_videos.items():
        target_medias_data.append(
            {
                "media_path": target_media.media_path,
                "file_type": target_media.file_type,
                "media_id": media_id,
            }
        )

    for face_id, input_face in main_window.input_faces.items():
        kv_map_path = None
        if is_denoiser_enabled and getattr(input_face, "kv_map", None) is not None:
            try:
                payload = {"kv_map": input_face.kv_map}
                kv_map_path = _save_hashed_kv_payload(
                    main_window,
                    payload,
                    "reference_kv_data",
                    parent_file_paths=[
                        resolved_parent_path,
                        str(input_face.media_path),
                    ],
                )
            except Exception as e:
                print(
                    f"[ERROR] Error saving hashed K/V map for input face {input_face.face_id}: {e}"
                )

        input_faces_data[face_id] = {
            "media_path": input_face.media_path,
            "kv_map": kv_map_path,
        }

    inline_embedding_buttons = (
        {}
        if use_tabs_file
        else list_view_actions.get_all_merged_embeddings(main_window)
    )
    for embedding_id, embedding_button in inline_embedding_buttons.items():
        kv_map_path = None
        kv_payload_to_save = getattr(embedding_button, "kv_map_list", None) or getattr(
            embedding_button, "kv_map", None
        )

        if kv_payload_to_save is not None:
            try:
                payload = (
                    {"kv_map_list": kv_payload_to_save}
                    if getattr(embedding_button, "kv_map_list", None)
                    else {"kv_map": kv_payload_to_save}
                )
                kv_map_path = _save_hashed_kv_payload(
                    main_window,
                    payload,
                    "reference_kv_data",
                    parent_file_paths=resolved_parent_path,
                )
            except Exception as e:
                print(
                    f"[ERROR] Error saving hashed K/V map for embedding {embedding_id}: {e}"
                )

        embeddings_data[embedding_id] = {
            "embedding_name": embedding_button.embedding_name,
            "embedding_store": {
                model: emb.tolist()
                for model, emb in embedding_button.embedding_store.items()
            },
            "kv_map": kv_map_path,
        }
        if getattr(embedding_button, "tags", None):
            embeddings_data[embedding_id]["tags"] = list(embedding_button.tags)

    markers_to_save = sanitize_markers_dictionary(
        main_window,
        convert_markers_to_supported_type(
            main_window, cast(MarkerTypes, copy.deepcopy(main_window.markers)), dict
        ),
    )

    tab_state = {
        "current_tab_index": main_window.tabWidget.currentIndex(),
        "tab_order": [],
    }

    for i in range(main_window.tabWidget.count()):
        tab_state["tab_order"].append(
            {"text": main_window.tabWidget.tabText(i), "original_index": i}
        )

    if use_tabs_file:
        embedding_tabs_widget = getattr(main_window, "embeddingTabs", None)
        embedding_tabs_state_to_save = {
            "external_file": embedding_tabs_store.sidecar_path_for(data_filename).name,
            "tab_count": len(getattr(main_window, "embedding_tab_states", None) or []),
            "active_index": (
                max(0, embedding_tabs_widget.currentIndex())
                if embedding_tabs_widget is not None
                else 0
            ),
        }
    else:
        embedding_tabs_state_to_save = list_view_actions.get_embedding_tabs_state(
            main_window
        )
        _add_pending_tabs_inline(
            main_window, embedding_tabs_state_to_save, embeddings_data, resolved_parent_path
        )

    current_params_to_save = {}
    _params_src = main_window.current_widget_parameters
    _sel_face = getattr(main_window, "selected_target_face_id", None)
    if main_window.target_faces and _sel_face in main_window.parameters:
        # Save what the user actually sees for the selected face.
        _params_src = main_window.parameters[_sel_face]
    if isinstance(_params_src, misc_helpers.ParametersDict):
        current_params_to_save = _params_src.data.copy()
    elif isinstance(_params_src, dict):
        current_params_to_save = _params_src.copy()
    else:
        print(
            f"[WARN] Unexpected type for current widget parameters: {type(_params_src)}. Saving empty dict."
        )

    data = {
        "control": sanitize_state_dictionary(
            main_window.control.copy(), main_window.control
        ),
        "target_medias_data": target_medias_data,
        "selected_media_id": main_window.selected_video_button.media_id
        if isinstance(
            main_window.selected_video_button, widget_components.TargetMediaCardButton
        )
        else False,
        "swap_faces_enabled": main_window.swapfacesButton.isChecked(),
        "edit_faces_enabled": main_window.editFacesButton.isChecked(),
        "input_faces_data": input_faces_data,
        "embeddings_data": embeddings_data,
        "markers": markers_to_save,
        "issue_frames_by_face": {
            str(face_id): sorted(frames)
            for face_id, frames in main_window.issue_frames_by_face.items()
        },
        "dropped_frames": sorted(main_window.dropped_frames),
        "segments": [
            {
                "start": int(s["start"]),
                "end": int(s["end"]),
                "kind": s.get("kind", "normal"),
                "parameters": (
                    {
                        fid: pd.data.copy() if hasattr(pd, "data") else dict(pd)
                        for fid, pd in s["parameters"].items()
                    }
                    if s.get("parameters") is not None
                    else None
                ),
                "control": copy.deepcopy(s.get("control"))
                if s.get("control") is not None
                else None,
            }
            for s in getattr(main_window, "segments", [])
        ],
        "last_target_media_folder_path": main_window.last_target_media_folder_path,
        "last_input_media_folder_path": main_window.last_input_media_folder_path,
        # The path of an embeddings file can reveal names, so it is not written to
        # last_workspace.json; the separate embedding tabs file keeps it.
        "loaded_embedding_filename": (
            "" if use_tabs_file else main_window.loaded_embedding_filename
        ),
        "embedding_tabs_state": embedding_tabs_state_to_save,
        "current_widget_parameters": current_params_to_save,
        "tab_state": tab_state,
        "window_state_data": window_state_data,
    }

    if data_filename:
        _is_last_ws = str(data_filename).endswith("last_workspace.json")

        _failed_keys = set(getattr(main_window, "_workspace_failed_keys", ()) or ())
        _original = getattr(main_window, "_workspace_original_data", None)
        if _is_last_ws and _failed_keys and isinstance(_original, dict):
            # Parts that could not be loaded keep their value from the file.
            for _k in _failed_keys:
                if _k in _original:
                    data[_k] = copy.deepcopy(_original[_k])

        if _is_last_ws and getattr(main_window, "_workspace_load_in_progress", False):
            print("[WARN] Skipping last_workspace save: workspace load in progress")
            return

        if _is_last_ws and getattr(main_window, "_workspace_load_failed", False):
            print("[WARN] Skipping last_workspace save: previous load failed")
            return

        if use_tabs_file:
            if _failed_keys & {"embeddings_data", "embedding_tabs_state"}:
                print("[WARN] Embedding tabs failed to load; keeping the existing tabs file.")
            else:
                # Failure is logged inside; the workspace itself is still saved.
                embedding_tabs_store.write_embedding_tabs_file(main_window, data_filename)

        tmp_path = f"{data_filename}.{os.getpid()}.{uuid.uuid4().hex[:8]}.tmp"
        bak_path = f"{data_filename}.bak"
        bak2_path = f"{data_filename}.bak2"
        try:
            data_as_json = json.dumps(data, indent=4)
            with _WORKSPACE_WRITE_LOCK:
                with open(tmp_path, "w", encoding="utf-8") as data_file:
                    data_file.write(data_as_json)
                    data_file.flush()
                    try:
                        os.fsync(data_file.fileno())
                    except OSError:
                        pass
                with open(tmp_path, "r", encoding="utf-8") as _chk:
                    json.load(_chk)
                for _attempt in range(10):
                    try:
                        os.replace(tmp_path, data_filename)
                        break
                    except PermissionError:
                        if _attempt == 9:
                            raise
                        time.sleep(0.1)
                try:
                    with open(data_filename, "r", encoding="utf-8") as _post:
                        json.load(_post)
                except (OSError, json.JSONDecodeError, ValueError) as _post_err:
                    print(f"[ERROR] Post-write validation failed: {_post_err}")
                    _restored = False
                    for _src in (bak_path, bak2_path):
                        if not os.path.exists(_src):
                            continue
                        try:
                            with open(_src, "r", encoding="utf-8") as _src_f:
                                _src_content = _src_f.read()
                            json.loads(_src_content)
                            with open(data_filename, "w", encoding="utf-8") as _dst_f:
                                _dst_f.write(_src_content)
                            print(f"[WARN] Restored workspace from {_src}")
                            _restored = True
                            break
                        except (OSError, json.JSONDecodeError, ValueError) as _rst_err:
                            print(f"[ERROR] Restore from {_src} failed: {_rst_err}")
                    if not _restored:
                        raise
            if _is_last_ws:
                print(f"[INFO] Last workspace saved to: {data_filename}")
            else:
                common_widget_actions.create_and_show_toast_message(
                    main_window,
                    "Workspace Saved",
                    f"Saved Workspace to file: {data_filename}",
                )
        except Exception as e:
            try:
                os.remove(tmp_path)
            except OSError:
                pass
            print(f"[ERROR] Failed to save workspace {data_filename}: {e}")
            if not _is_last_ws:
                common_widget_actions.create_and_show_messagebox(
                    main_window,
                    "Save Error",
                    f"Failed to save workspace:\n{e}",
                    main_window,
                )


def purge_unused_kv_maps(main_window: "MainWindow"):
    """
    Scans the KV registry for missing parent files and orphaned .pt hashes.
    Prompts the user for verification before safely deleting unused tensors.
    """
    registry_path = (
        main_window.project_root_path
        / "model_assets"
        / "reference_kv_data"
        / "kv_registry.json"
    )
    kv_data_dir = main_window.project_root_path / "model_assets" / "reference_kv_data"

    with _KV_REGISTRY_LOCK:
        registry = _load_registry(registry_path)

    unique_parent_paths = set()
    for paths in registry.values():
        unique_parent_paths.update(paths)

    missing_paths = [p for p in unique_parent_paths if not Path(p).exists()]
    paths_to_remove = set()

    for missing_path in missing_paths:
        reply = QtWidgets.QMessageBox.question(
            main_window,
            "Missing File Detected",
            f"The saved workspace or job file cannot be found:\n{missing_path}\n\n"
            "Have you deleted or moved this file?\n\n"
            "Click 'Yes' if it's gone and you want to clean up its unused data.\n"
            "Click 'No' to keep its K/V maps safely.",
            QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.No,
            QtWidgets.QMessageBox.No,
        )
        if reply == QtWidgets.QMessageBox.Yes:
            paths_to_remove.add(missing_path)

    files_deleted_count = 0
    registry_modified = False
    keys_to_delete = []

    for kv_filename, paths in registry.items():
        original_len = len(paths)
        registry[kv_filename] = [p for p in paths if p not in paths_to_remove]

        if len(registry[kv_filename]) != original_len:
            registry_modified = True

        if not registry[kv_filename]:
            keys_to_delete.append(kv_filename)

    for kv_filename in keys_to_delete:
        pt_path = kv_data_dir / kv_filename
        try:
            if pt_path.exists():
                pt_path.unlink()
                print(f"[INFO] Purged unused K/V map: {kv_filename}")
                files_deleted_count += 1
        except Exception as e:
            print(f"[ERROR] Failed to delete {kv_filename}: {e}")

        del registry[kv_filename]
        registry_modified = True

    unregistered_pt_files = []
    if kv_data_dir.exists():
        for pt_file in kv_data_dir.glob("*.pt"):
            if pt_file.name not in registry:
                unregistered_pt_files.append(pt_file)

    legacy_files_deleted = 0
    if unregistered_pt_files:
        reply = QtWidgets.QMessageBox.question(
            main_window,
            "Legacy K/V Maps Detected",
            f"Found {len(unregistered_pt_files)} old K/V map files that are not tracked in the current registry.\n\n"
            "Do you want to permanently delete them to free up disk space?",
            QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.No,
            QtWidgets.QMessageBox.No,
        )
        if reply == QtWidgets.QMessageBox.Yes:
            for pt_file in unregistered_pt_files:
                try:
                    pt_file.unlink()
                    legacy_files_deleted += 1
                    print(f"[INFO] Purged legacy K/V map: {pt_file.name}")
                except Exception as e:
                    print(f"[ERROR] Failed to delete {pt_file.name}: {e}")

    if registry_modified:
        with _KV_REGISTRY_LOCK:
            try:
                with open(registry_path, "w", encoding="utf-8") as f:
                    json.dump(registry, f, indent=4)
            except IOError as e:
                print(f"[ERROR] Failed to write updated KV registry: {e}")

    total_deleted = files_deleted_count + legacy_files_deleted
    if total_deleted > 0:
        common_widget_actions.create_and_show_messagebox(
            main_window,
            "Storage Purge Complete",
            f"Successfully purged {total_deleted} unused K/V map(s).",
            main_window,
        )
    elif not missing_paths and not unregistered_pt_files:
        common_widget_actions.create_and_show_messagebox(
            main_window,
            "Storage Purge",
            "All tracked K/V maps are currently in use. Nothing to purge.",
            main_window,
        )