from typing import Dict, List, Any, Optional, Tuple, cast
from collections.abc import Mapping

import numpy
import torch

from app.helpers.miscellaneous import find_best_target_match


class SequentialDetector:
    def __init__(self, main_window):
        self.main_window = main_window
        self.last_detected_faces: List[Dict[str, Any]] = []
        self._smoothed_kps: Dict[int, numpy.ndarray] = {}
        self._smoothed_dense_kps: Dict[int, numpy.ndarray] = {}
        self._smoothed_dense_kps_203: Dict[int, numpy.ndarray] = {}
        self._temporal_memory: List[Dict[str, Any]] = []

    def reset_state(self) -> None:
        self.last_detected_faces.clear()
        self._smoothed_kps.clear()
        self._smoothed_dense_kps.clear()
        self._smoothed_dense_kps_203.clear()
        if hasattr(self, "_temporal_memory"):
            self._temporal_memory.clear()
        else:
            self._temporal_memory = []
        if hasattr(self.main_window, "function_worker") and hasattr(
            self.main_window.function_worker, "reset_face_tracker"
        ):
            self.main_window.function_worker.reset_face_tracker()

    def run(
        self,
        frame_rgb: numpy.ndarray,
        local_control_for_worker: dict,
        local_params_for_worker: Optional[dict] = None,
        is_master_edit_active: bool = False,
        frame_tensor: Optional[torch.Tensor] = None,
        detector_control_override: Optional[dict] = None,
        frame_number: int = -1,
    ) -> Tuple[numpy.ndarray, numpy.ndarray, numpy.ndarray, numpy.ndarray]:
        if local_control_for_worker.get("VR180ModeEnableToggle", False):
            return (
                numpy.empty((0, 4), dtype=numpy.float32),
                numpy.empty((0, 5, 2), dtype=numpy.float32),
                numpy.empty((0, 68, 2), dtype=numpy.float32),
                numpy.empty((0, 203, 2), dtype=numpy.float32),
            )

        control = detector_control_override or local_control_for_worker
        use_landmark = control.get("LandmarkDetectToggle", True)
        landmark_mode = control.get("LandmarkDetectModelSelection", "203")
        from_points = control.get("DetectFromPointsToggle", False)

        requires_203 = False
        if local_params_for_worker:
            for face_id, face_params in local_params_for_worker.items():
                if isinstance(face_params, Mapping):
                    is_face_editor_active = is_master_edit_active and face_params.get(
                        "FaceEditorEnableToggle", False
                    )
                    is_expression_active = face_params.get(
                        "FaceExpressionEnableBothToggle", False
                    )
                    is_auto_mouth_active = face_params.get(
                        "AutoMouthExpressionEnableToggle", False
                    )
                    is_makeup_active = is_master_edit_active and (
                        face_params.get("FaceMakeupEnableToggle", False)
                        or face_params.get("HairMakeupEnableToggle", False)
                        or face_params.get("EyeBrowsMakeupEnableToggle", False)
                        or face_params.get("LipsMakeupEnableToggle", False)
                    )
                    if (
                        is_face_editor_active
                        or is_expression_active
                        or is_auto_mouth_active
                        or is_makeup_active
                    ):
                        requires_203 = True
                        break

        device = self.main_window.models_processor.device
        owns_frame_tensor = frame_tensor is None

        try:
            target_faces = dict(self.main_window.target_faces)
        except Exception:
            target_faces = {}

        show_bboxes = control.get(
            "ShowAllDetectedFacesBBoxToggle", False
        ) or control.get("ShowByteTrackBBoxToggle", False)

        if len(target_faces) == 0 and not show_bboxes:
            if owns_frame_tensor and frame_tensor is not None:
                del frame_tensor
            self._smoothed_kps.clear()
            self._smoothed_dense_kps.clear()
            self._smoothed_dense_kps_203.clear()
            return (
                numpy.empty((0, 4), dtype=numpy.float32),
                numpy.empty((0, 5, 2), dtype=numpy.float32),
                numpy.empty((0, 68, 2), dtype=numpy.float32),
                numpy.empty((0, 203, 2), dtype=numpy.float32),
            )

        _upscale_rw, _upscale_rh = 1.0, 1.0
        _orig_h, _orig_w = frame_rgb.shape[0], frame_rgb.shape[1]
        _new_h, _new_w = _orig_h, _orig_w
        if _orig_w < 512 or _orig_h < 512:
            if _orig_w <= _orig_h:
                _new_h, _new_w = int(512 * _orig_h / _orig_w), 512
            else:
                _new_h, _new_w = 512, int(512 * _orig_w / _orig_h)
            _upscale_rw = _orig_w / _new_w
            _upscale_rh = _orig_h / _new_h

        from app.helpers.miscellaneous import get_scaling_transforms as _gst
        try:
            _scaleback_interp = _gst(control)[8]
        except Exception:
            from torchvision.transforms import v2 as _tv2
            _scaleback_interp = _tv2.InterpolationMode.BILINEAR

        if frame_tensor is None:
            if (_new_w, _new_h) != (_orig_w, _orig_h):
                import torchvision.transforms.v2 as _v2
                _t = (
                    torch.from_numpy(frame_rgb)
                    .to(device, non_blocking=True)
                    .permute(2, 0, 1)
                    .unsqueeze(0)
                )
                _t = _v2.functional.resize(
                    _t,
                    [_new_h, _new_w],
                    interpolation=_scaleback_interp,
                    antialias=False,
                )
                frame_tensor = _t.squeeze(0).contiguous()
            else:
                frame_tensor = (
                    torch.from_numpy(frame_rgb)
                    .to(device, non_blocking=True)
                    .permute(2, 0, 1)
                )

        bboxes, kpss_5, _ = self.main_window.function_worker.run_detect(
            frame_tensor,
            control.get("DetectorModelSelection", "RetinaFace"),
            max_num=int(control.get("MaxFacesToDetectSlider", 20)),
            score=control.get("DetectorScoreSlider", 50) / 100.0,
            input_size=(512, 512),
            use_landmark_detection=False,
            from_points=from_points,
            rotation_angles=[0]
            if not control.get("AutoRotationToggle", False)
            else [0, 90, 180, 270],
            use_mean_eyes=control.get("LandmarkMeanEyesToggle", False),
            control_override=control,
            bypass_bytetrack=False,
        )

        # NOTE: bboxes / kpss_5 intentionally stay in UPSCALED coordinates here,
        # the same space as frame_tensor. They are scaled back to original-frame
        # coordinates once, after the landmark stage (see below).

        valid_indices: list = []
        temporal_memory_this_frame: list = []

        try:
            target_faces = dict(self.main_window.target_faces)
        except Exception:
            target_faces = {}

        rec_model = str(control.get("RecognitionModelSelection", "ArcFace"))
        default_params = dict(self.main_window.default_parameters.data)

        if (
            isinstance(bboxes, numpy.ndarray)
            and bboxes.shape[0] > 0
            and len(target_faces) > 0
        ):
            unverified_current_indices = []
            unverified_embeddings = {}
            emb_by_index: dict = {}

            for i in range(len(kpss_5)):
                current_bbox = bboxes[i]
                current_kps5 = kpss_5[i]

                # Already in the same (upscaled) space as frame_tensor.
                _kps_for_rec = current_kps5
                face_emb, _ = self.main_window.function_worker.run_recognize_direct(
                    frame_tensor, _kps_for_rec, "Auto", rec_model
                )
                _params_zero = {}
                for _fid_z, _p_z in (local_params_for_worker or {}).items():
                    if hasattr(_p_z, "items"):
                        _new_p_z = dict(_p_z)
                        _new_p_z["SimilarityThresholdSlider"] = 0.0
                        _params_zero[_fid_z] = _new_p_z
                    else:
                        _params_zero[_fid_z] = _p_z
                _defaults_zero = dict(default_params)
                _defaults_zero["SimilarityThresholdSlider"] = 0.0
                match, _, _score = find_best_target_match(
                    face_emb,
                    self.main_window.function_worker,
                    target_faces,
                    _params_zero,
                    _defaults_zero,
                    rec_model,
                )

                if match is not None:
                    valid_indices.append(i)
                    emb_by_index[i] = face_emb
                    temporal_memory_this_frame.append(
                        {"bbox": current_bbox, "emb": face_emb}
                    )
                else:
                    unverified_current_indices.append(i)
                    unverified_embeddings[i] = face_emb

            if (
                len(unverified_current_indices) > 0
                and hasattr(self, "_temporal_memory")
                and len(self._temporal_memory) > 0
            ):
                curr_bboxes = bboxes[unverified_current_indices]
                prev_bboxes = numpy.array(
                    [p["bbox"] for p in self._temporal_memory], dtype=numpy.float32
                )

                xA = numpy.maximum(curr_bboxes[:, 0:1], prev_bboxes[:, 0])
                yA = numpy.maximum(curr_bboxes[:, 1:2], prev_bboxes[:, 1])
                xB = numpy.minimum(curr_bboxes[:, 2:3], prev_bboxes[:, 2])
                yB = numpy.minimum(curr_bboxes[:, 3:4], prev_bboxes[:, 3])

                interArea = numpy.maximum(0.0, xB - xA) * numpy.maximum(0.0, yB - yA)
                boxAArea = (curr_bboxes[:, 2] - curr_bboxes[:, 0]) * (
                    curr_bboxes[:, 3] - curr_bboxes[:, 1]
                )
                boxBArea = (prev_bboxes[:, 2] - prev_bboxes[:, 0]) * (
                    prev_bboxes[:, 3] - prev_bboxes[:, 1]
                )
                denominator = boxAArea[:, None] + boxBArea - interArea
                iou_matrix = numpy.where(denominator > 0, interArea / denominator, 0.0)

                rows, cols = numpy.where(iou_matrix > 0.40)
                iou_matches = [
                    (float(iou_matrix[r, c]), unverified_current_indices[r], int(c))
                    for r, c in zip(rows, cols)
                ]
                iou_matches.sort(key=lambda x: x[0], reverse=True)
                used_curr_indices = set()
                used_prev_indices = set()

                for iou, curr_idx, prev_idx in iou_matches:
                    if (
                        curr_idx not in used_curr_indices
                        and prev_idx not in used_prev_indices
                    ):
                        curr_emb = unverified_embeddings[curr_idx].flatten()
                        prev_emb = self._temporal_memory[prev_idx]["emb"].flatten()
                        sim = numpy.dot(curr_emb, prev_emb) / (
                            numpy.linalg.norm(curr_emb) * numpy.linalg.norm(prev_emb)
                            + 1e-8
                        )
                        if sim > 0.75:
                            valid_indices.append(curr_idx)
                            emb_by_index[curr_idx] = unverified_embeddings[curr_idx]
                            temporal_memory_this_frame.append(
                                {
                                    "bbox": bboxes[curr_idx],
                                    "emb": unverified_embeddings[curr_idx],
                                }
                            )
                            used_curr_indices.add(curr_idx)
                            used_prev_indices.add(prev_idx)

            self._temporal_memory = temporal_memory_this_frame

        elif isinstance(bboxes, numpy.ndarray) and bboxes.shape[0] > 0:
            force_swap = control.get("ForceSwapToggle", True)
            if not force_swap and not requires_203:
                valid_indices = []
                self._temporal_memory = []
            else:
                valid_indices = list(range(len(bboxes)))
                temporal_memory_this_frame = []
                for i in range(len(bboxes)):
                    dummy_emb = numpy.zeros((512,), dtype=numpy.float32)
                    temporal_memory_this_frame.append(
                        {"bbox": bboxes[i], "emb": dummy_emb}
                    )
                self._temporal_memory = temporal_memory_this_frame

        filtered_bboxes = (
            bboxes[valid_indices]
            if valid_indices
            else numpy.empty((0, 4), dtype=numpy.float32)
        )
        filtered_kpss_5 = (
            kpss_5[valid_indices]
            if valid_indices
            else numpy.empty((0, 5, 2), dtype=numpy.float32)
        )

        if len(filtered_bboxes) == 0:
            if owns_frame_tensor and frame_tensor is not None:
                del frame_tensor
            self._smoothed_kps.clear()
            self._smoothed_dense_kps.clear()
            self._smoothed_dense_kps_203.clear()
            return (
                numpy.empty((0, 4), dtype=numpy.float32),
                numpy.empty((0, 5, 2), dtype=numpy.float32),
                numpy.empty((0, 68, 2), dtype=numpy.float32),
                numpy.empty((0, 203, 2), dtype=numpy.float32),
            )

        num_targets = len(filtered_bboxes)
        filtered_kpss = []
        filtered_kpss_203 = []
        head_bboxes = None
        if use_landmark and landmark_mode == "hrffa" and num_targets > 0:
            head_bboxes = self.main_window.function_worker.run_detect_head_bboxes(
                frame_tensor
            )

        for i in range(num_targets):
            current_bbox = filtered_bboxes[i]
            current_kps5 = filtered_kpss_5[i]
            kps_standard = current_kps5.copy()
            kps_203_local = numpy.zeros((203, 2), dtype=numpy.float32)
            has_valid_203 = False

            if requires_203:
                lm_203_5, lm_203, _ = (
                    self.main_window.function_worker.run_detect_landmark(
                        frame_tensor,
                        current_bbox,
                        current_kps5,
                        detect_mode="203",
                        score=0.5,
                        use_mean_eyes=control.get("LandmarkMeanEyesToggle", False),
                        from_points=True,
                    )
                )
                if len(lm_203) > 0:
                    kps_203_local = lm_203
                    has_valid_203 = True
                    if landmark_mode == "203" and len(lm_203_5) > 0:
                        filtered_kpss_5[i] = lm_203_5
                filtered_kpss_203.append(kps_203_local)

            if use_landmark:
                if (
                    landmark_mode == "203"
                    and requires_203
                    and has_valid_203
                    and from_points
                ):
                    kps_standard = kps_203_local.copy()
                else:
                    lm_std_5, lm_kpss, _ = (
                        self.main_window.function_worker.run_detect_landmark(
                            frame_tensor,
                            current_bbox,
                            current_kps5,
                            detect_mode=landmark_mode,
                            score=control.get("LandmarkDetectScoreSlider", 50) / 100.0,
                            use_mean_eyes=control.get("LandmarkMeanEyesToggle", False),
                            from_points=from_points,
                            head_bboxes=head_bboxes,
                        )
                    )
                    if len(lm_kpss) > 0:
                        kps_standard = lm_kpss
                        if len(lm_std_5) > 0:
                            filtered_kpss_5[i] = lm_std_5
            filtered_kpss.append(kps_standard)

        bboxes = filtered_bboxes
        kpss_5 = filtered_kpss_5

        if len(filtered_kpss) > 0:
            try:
                kpss = numpy.array(filtered_kpss, dtype=numpy.float32)
            except Exception:
                kpss = numpy.array(filtered_kpss, dtype=object)
        else:
            kpss = numpy.empty((0, 5, 2), dtype=numpy.float32)

        if requires_203 and len(filtered_kpss_203) > 0:
            try:
                kpss_203 = numpy.array(filtered_kpss_203, dtype=numpy.float32)
            except Exception:
                kpss_203 = numpy.array(filtered_kpss_203, dtype=object)
        else:
            kpss_203 = numpy.empty((0, 203, 2), dtype=numpy.float32)

        if owns_frame_tensor:
            del frame_tensor
            frame_tensor = None

        if bboxes.shape[0] > 0:
            valid_mask = numpy.isfinite(bboxes).all(axis=1)
            if not valid_mask.all():
                bboxes = bboxes[valid_mask]
                kpss_5 = kpss_5[valid_mask]
                if isinstance(kpss, numpy.ndarray) and kpss.shape[0] == len(valid_mask):
                    kpss = kpss[valid_mask]
                if isinstance(kpss_203, numpy.ndarray) and kpss_203.shape[0] == len(
                    valid_mask
                ):
                    kpss_203 = kpss_203[valid_mask]

        if _upscale_rw != 1.0 or _upscale_rh != 1.0:
            # Scale EVERYTHING back to original-frame coordinates exactly once.
            if bboxes.shape[0] > 0:
                bboxes = bboxes.copy()
                bboxes[:, [0, 2]] *= _upscale_rw
                bboxes[:, [1, 3]] *= _upscale_rh
            if isinstance(kpss_5, numpy.ndarray) and kpss_5.shape[0] > 0:
                kpss_5 = kpss_5.copy()
                kpss_5[:, :, 0] *= _upscale_rw
                kpss_5[:, :, 1] *= _upscale_rh
            if isinstance(kpss, numpy.ndarray) and kpss.shape[0] > 0:
                try:
                    kpss = kpss.copy()
                    kpss[:, :, 0] *= _upscale_rw
                    kpss[:, :, 1] *= _upscale_rh
                except (AttributeError, ValueError):
                    pass
            if isinstance(kpss_203, numpy.ndarray) and kpss_203.shape[0] > 0:
                try:
                    kpss_203 = kpss_203.copy()
                    kpss_203[:, :, 0] *= _upscale_rw
                    kpss_203[:, :, 1] *= _upscale_rh
                except (AttributeError, ValueError):
                    pass

        detected_for_state = []
        if bboxes.shape[0] > 0:
            for i in range(len(bboxes)):
                detected_for_state.append({"bbox": bboxes[i], "score": 1.0})
        self.last_detected_faces = detected_for_state

        is_smoothing_enabled = local_control_for_worker.get(
            "KPSSmoothingEnableToggle", True
        )

        if is_smoothing_enabled:
            img_h_for_kps, img_w_for_kps = frame_rgb.shape[0], frame_rgb.shape[1]
            if isinstance(kpss_5, numpy.ndarray) and kpss_5.shape[0] > 0:
                kpss_5 = kpss_5.copy()
                n_faces = kpss_5.shape[0]
                new_smoothed_kps = {}
                new_smoothed_dense_kps = {}
                new_smoothed_dense_kps_203 = {}
                valid_kpss = cast(numpy.ndarray, kpss)
                valid_kpss_203 = cast(numpy.ndarray, kpss_203)
                dense_kps_count = (
                    int(kpss.shape[0]) if isinstance(kpss, numpy.ndarray) else 0
                )
                has_dense_kps = dense_kps_count > 0
                if has_dense_kps:
                    valid_kpss = valid_kpss.copy()
                    kpss = valid_kpss
                dense_kps_203_count = (
                    int(kpss_203.shape[0]) if isinstance(kpss_203, numpy.ndarray) else 0
                )
                has_dense_kps_203 = dense_kps_203_count > 0
                if has_dense_kps_203:
                    valid_kpss_203 = valid_kpss_203.copy()
                    kpss_203 = valid_kpss_203
                for _i in range(n_faces):
                    _raw = kpss_5[_i]
                    dense_kps_available = has_dense_kps and _i < dense_kps_count
                    dense_kps_203_available = (
                        has_dense_kps_203 and _i < dense_kps_203_count
                    )
                    if (
                        _raw is None
                        or _raw.size == 0
                        or numpy.any(numpy.isnan(_raw))
                        or numpy.any(numpy.isinf(_raw))
                    ):
                        continue
                    if (
                        numpy.any(_raw[:, 0] < 0)
                        or numpy.any(_raw[:, 0] >= img_w_for_kps)
                        or numpy.any(_raw[:, 1] < 0)
                        or numpy.any(_raw[:, 1] >= img_h_for_kps)
                    ):
                        continue
                    _centroid_raw = numpy.mean(_raw, axis=0)
                    _best_match_key = None
                    _min_dist = float("inf")
                    _face_w = bboxes[_i][2] - bboxes[_i][0]
                    _face_h = bboxes[_i][3] - bboxes[_i][1]
                    _adaptive_threshold = max(30.0, max(_face_w, _face_h) * 0.4)
                    for _k, _prev_kps in self._smoothed_kps.items():
                        _centroid_prev = numpy.mean(_prev_kps, axis=0)
                        _dist = numpy.linalg.norm(_centroid_raw - _centroid_prev)
                        if _dist < _adaptive_threshold and _dist < _min_dist:
                            _min_dist = float(_dist)
                            _best_match_key = _k
                    if _best_match_key is not None:
                        base_alpha = (
                            local_control_for_worker.get("KPSEmaAlphaSlider", 35)
                            / 100.0
                        )
                        movement_factor = min(1.0, _min_dist / 15.0)
                        dynamic_alpha = base_alpha + movement_factor * (
                            1.0 - base_alpha
                        )
                        new_smoothed_kps[_i] = (
                            dynamic_alpha * _raw
                            + (1.0 - dynamic_alpha)
                            * self._smoothed_kps[_best_match_key]
                        )
                        del self._smoothed_kps[_best_match_key]
                        if dense_kps_available:
                            if _best_match_key in self._smoothed_dense_kps:
                                new_smoothed_dense_kps[_i] = (
                                    dynamic_alpha * valid_kpss[_i]
                                    + (1.0 - dynamic_alpha)
                                    * self._smoothed_dense_kps[_best_match_key]
                                )
                                del self._smoothed_dense_kps[_best_match_key]
                            else:
                                new_smoothed_dense_kps[_i] = valid_kpss[_i].copy()
                        if has_dense_kps_203 and dense_kps_203_available:
                            is_valid_203 = not numpy.all(valid_kpss_203[_i] == 0)
                            if is_valid_203:
                                if _best_match_key in self._smoothed_dense_kps_203:
                                    new_smoothed_dense_kps_203[_i] = (
                                        dynamic_alpha * valid_kpss_203[_i]
                                        + (1.0 - dynamic_alpha)
                                        * self._smoothed_dense_kps_203[_best_match_key]
                                    )
                                    del self._smoothed_dense_kps_203[_best_match_key]
                                else:
                                    new_smoothed_dense_kps_203[_i] = valid_kpss_203[
                                        _i
                                    ].copy()
                            else:
                                if _best_match_key in self._smoothed_dense_kps_203:
                                    new_smoothed_dense_kps_203[_i] = (
                                        self._smoothed_dense_kps_203[
                                            _best_match_key
                                        ].copy()
                                    )
                                    valid_kpss_203[_i] = new_smoothed_dense_kps_203[_i]
                    else:
                        new_smoothed_kps[_i] = _raw.copy()
                        if dense_kps_available:
                            new_smoothed_dense_kps[_i] = valid_kpss[_i].copy()
                        if has_dense_kps_203 and dense_kps_203_available:
                            new_smoothed_dense_kps_203[_i] = valid_kpss_203[_i].copy()
                    kpss_5[_i] = new_smoothed_kps[_i]
                    if dense_kps_available:
                        valid_kpss[_i] = new_smoothed_dense_kps[_i]
                    if has_dense_kps_203 and dense_kps_203_available:
                        valid_kpss_203[_i] = new_smoothed_dense_kps_203[_i]
                self._smoothed_kps = new_smoothed_kps
                self._smoothed_dense_kps = new_smoothed_dense_kps
                self._smoothed_dense_kps_203 = new_smoothed_dense_kps_203
        else:
            self._smoothed_kps.clear()
            self._smoothed_dense_kps.clear()
            self._smoothed_dense_kps_203.clear()

        return bboxes, kpss_5, kpss, kpss_203