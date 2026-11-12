import math
import re
from math import ceil
from collections import OrderedDict, deque
from typing import TYPE_CHECKING, Any, cast
import threading
import numpy as np
import skimage.transform as trans
import torch
import torch.nn.functional as F
from torchvision.transforms import v2
import kornia.enhance as ke
import kornia.color as kc
import kornia.geometry.transform as kgm

from app.processors.utils import faceutil
from app.processors.utils import platform_support

if TYPE_CHECKING:
    from .frame_worker import FrameWorker

_COLOR_EMA_LOCK = threading.Lock()
_COLOR_STATS_EMA: OrderedDict[bytes, dict[str, torch.Tensor]] = OrderedDict()
_COLOR_STATS_EMA_MAX: int = 32
_COLOR_EMA_ALPHA: float = 0.15


class PipelineProcessor:
    def __init__(self, worker: "FrameWorker"):
        self.worker = worker
        self._gabor_kernels_expanded_cache: OrderedDict[tuple, torch.Tensor] = OrderedDict()
        self._gabor_kernels_cache: OrderedDict[tuple, torch.Tensor] = OrderedDict()
        self._kernel_lap: torch.Tensor | None = None
        self._kernel_sobel_x: torch.Tensor | None = None
        self._kernel_sobel_y: torch.Tensor | None = None
        self._mouth_action_score: float = 0.0

    @property
    def kernel_lap(self) -> torch.Tensor:
        if self._kernel_lap is None:
            device = self.worker.models_processor.device
            self._kernel_lap = torch.tensor(
                [[0, 1, 0], [1, -4, 1], [0, 1, 0]], device=device, dtype=torch.float32
            ).view(1, 1, 3, 3)
        return self._kernel_lap

    @property
    def kernel_sobel_x(self) -> torch.Tensor:
        if self._kernel_sobel_x is None:
            device = self.worker.models_processor.device
            self._kernel_sobel_x = torch.tensor(
                [[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], device=device, dtype=torch.float32
            ).view(1, 1, 3, 3)
        return self._kernel_sobel_x

    @property
    def kernel_sobel_y(self) -> torch.Tensor:
        if self._kernel_sobel_y is None:
            self._kernel_sobel_y = self.kernel_sobel_x.transpose(2, 3)
        return self._kernel_sobel_y

    def _apply_denoiser_pass(
        self,
        image_tensor_cxhxw_uint8: torch.Tensor,
        control: dict[str, Any],
        pass_suffix: str,
        kv_map: dict | None,
        color_mask: torch.Tensor | None = None,
        blend_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        use_exclusive_path = control.get("UseReferenceExclusivePathToggle", False)
        denoiser_seed_from_slider_val = int(control.get("DenoiserBaseSeedSlider", 1))
        denoiser_mode_val = control.get(
            f"DenoiserModeSelection{pass_suffix}", "Single Step (Fast)"
        )
        ddim_steps_val = int(control.get(f"DenoiserDDIMStepsSlider{pass_suffix}", 20))
        cfg_scale_val = float(
            control.get(f"DenoiserCFGScaleDecimalSlider{pass_suffix}", 1.0)
        )
        single_step_t_val = int(
            control.get(f"DenoiserSingleStepTimestepSlider{pass_suffix}", 1)
        )
        sharpen_val = float(
            control.get(f"DenoiserLatentSharpeningDecimalSlider{pass_suffix}", 0.0)
        )
        enable_color = control.get("DenoiserColorCorrectionToggle", True)
        coarse_grain_reduction_val: float = (
            float(control.get("DenoiserCoarseGrainAmountDecimalSlider", 0.85))
            if control.get("DenoiserCoarseGrainReductionToggle", True)
            else 0.0
        )
        micro_grain_val: float = (
            float(control.get("DenoiserMicroGrainStrengthDecimalSlider", 0.15))
            if control.get("DenoiserMicroGrainToggle", False)
            else 0.0
        )

        if not kv_map and use_exclusive_path:
            return image_tensor_cxhxw_uint8

        denoised_image = self.worker.function_worker.apply_denoiser_unet(
            image_tensor_cxhxw_uint8,
            reference_kv_map=kv_map,
            use_reference_exclusive_path=use_exclusive_path,
            denoiser_mode=denoiser_mode_val,
            base_seed=denoiser_seed_from_slider_val,
            denoiser_single_step_t=single_step_t_val,
            denoiser_ddim_steps=ddim_steps_val,
            denoiser_cfg_scale=cfg_scale_val,
            latent_sharpening_strength=sharpen_val,
            enable_color_correction=enable_color,
            color_mask=color_mask,
            coarse_grain_reduction=coarse_grain_reduction_val,
            micro_grain_strength=micro_grain_val,
        )

        denoised_image = torch.clamp(denoised_image, 0, 255)

        if blend_mask is not None:
            if (
                blend_mask.shape[-1] != denoised_image.shape[-1]
                or blend_mask.shape[-2] != denoised_image.shape[-2]
            ):
                blend_mask = v2.functional.resize(
                    blend_mask,
                    [denoised_image.shape[-2], denoised_image.shape[-1]],
                    interpolation=v2.InterpolationMode.BILINEAR,
                    antialias=False,
                )
            denoised_image = torch.lerp(
                image_tensor_cxhxw_uint8.float(), denoised_image.float(), blend_mask
            ).to(image_tensor_cxhxw_uint8.dtype)

        return denoised_image

    @staticmethod
    @torch.no_grad()
    def _apply_likeness(
        source_latent: torch.Tensor, target_latent: torch.Tensor, params: dict
    ) -> torch.Tensor:
        if not params.get("FaceLikenessEnableToggle", False):
            return source_latent

        factor: float = float(params.get("FaceLikenessFactorDecimalSlider", 0.0))
        strength_multiplier: float = float(
            params.get("FaceLikenessStrengthDecimalSlider", 1.0)
        )

        if factor == 0.0 and strength_multiplier == 1.0:
            return source_latent

        s_norm: torch.Tensor = torch.norm(source_latent)
        t_norm: torch.Tensor = torch.norm(target_latent)

        if s_norm < 1e-6 or t_norm < 1e-6:
            return source_latent

        s_dir: torch.Tensor = source_latent / s_norm
        t_dir: torch.Tensor = target_latent / t_norm

        blended_dir: torch.Tensor

        if factor < 0.0:
            t: float = 1.0 + factor
            cos_theta: torch.Tensor = torch.sum(s_dir * t_dir)
            cos_theta = torch.clamp(cos_theta, -0.9999, 0.9999)
            theta: torch.Tensor = torch.acos(cos_theta)
            sin_theta: torch.Tensor = torch.sin(theta)

            if sin_theta < 1e-3:
                blended_dir = (1.0 - t) * t_dir + t * s_dir
            else:
                weight_t: torch.Tensor = torch.sin((1.0 - t) * theta) / sin_theta
                weight_s: torch.Tensor = torch.sin(t * theta) / sin_theta
                blended_dir = weight_t * t_dir + weight_s * s_dir
        elif factor > 0.0:
            difference_vector: torch.Tensor = s_dir - t_dir
            blended_dir = s_dir + (factor * difference_vector)
        else:
            blended_dir = s_dir

        blended_dir = blended_dir / torch.norm(blended_dir)
        final_latent: torch.Tensor = blended_dir * (s_norm * strength_multiplier)
        return final_latent

    def get_affined_face_dim_and_swapping_latents(
        self,
        original_faces: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
        swapper_model: str,
        dfm_model_name: str | None,
        s_e: np.ndarray | None,
        t_e: np.ndarray | None,
        parameters: dict[str, Any],
        cmddebug: bool,
        tform: Any,
    ) -> tuple[torch.Tensor | None, Any | None, int, torch.Tensor | list | None]:
        original_face_512, original_face_384, original_face_256, original_face_128 = (
            original_faces
        )

        dfm_model_instance = None
        input_face_affined = None
        dim = 1
        latent = None

        if swapper_model == "Inswapper128":
            _s_latent_np = self.worker.function_worker.calc_inswapper_latent(s_e)
            _t_latent_np = self.worker.function_worker.calc_inswapper_latent(t_e)
            if _s_latent_np is None or _t_latent_np is None:
                print(
                    "[ERROR] calc_inswapper_latent returned None (emap unavailable). Skipping swap."
                )
                return input_face_affined, dfm_model_instance, dim, latent

            latent = (
                torch.from_numpy(_s_latent_np)
                .float()
                .to(self.worker.models_processor.device)
            )
            dst_latent = (
                torch.from_numpy(_t_latent_np)
                .float()
                .to(self.worker.models_processor.device)
            )
            latent = self._apply_likeness(latent, dst_latent, parameters)

            dim = 1
            if parameters["SwapperResSelection"] == "Auto":
                if tform.scale <= 1.00:
                    dim = 4
                    input_face_affined = original_face_512
                elif tform.scale <= 1.75:
                    dim = 3
                    input_face_affined = original_face_384
                elif tform.scale <= 2:
                    dim = 2
                    input_face_affined = original_face_256
                else:
                    dim = 1
                    input_face_affined = original_face_128
            else:
                if parameters["SwapperResSelection"] == "128":
                    dim = 1
                    input_face_affined = original_face_128
                elif parameters["SwapperResSelection"] == "256":
                    dim = 2
                    input_face_affined = original_face_256
                elif parameters["SwapperResSelection"] == "384":
                    dim = 3
                    input_face_affined = original_face_384
                elif parameters["SwapperResSelection"] == "512":
                    dim = 4
                    input_face_affined = original_face_512

        elif swapper_model == "AlphaFace":
            _s_latent_np = self.worker.function_worker.calc_swapper_latent_alphaface(
                s_e
            )
            if _s_latent_np is None:
                print(
                    "[ERROR] calc_swapper_latent_alphaface returned None. Skipping swap."
                )
                return input_face_affined, dfm_model_instance, dim, latent

            latent = (
                torch.from_numpy(_s_latent_np)
                .float()
                .to(self.worker.models_processor.device)
            )

            if parameters.get("FaceLikenessEnableToggle", False):
                _t_latent_np = (
                    self.worker.function_worker.calc_swapper_latent_alphaface(t_e)
                )
                if _t_latent_np is None:
                    print(
                        "[ERROR] calc_swapper_latent_alphaface returned None for the "
                        "target face. Skipping swap."
                    )
                    return input_face_affined, dfm_model_instance, dim, None
                dst_latent = (
                    torch.from_numpy(_t_latent_np)
                    .float()
                    .to(self.worker.models_processor.device)
                )
                latent = self._apply_likeness(latent, dst_latent, parameters)

            alpha_res = str(parameters.get("AlphaFaceResSelection", "256"))

            if alpha_res == "Auto":
                if tform.scale <= 1.10:
                    dim = 4
                    input_face_affined = original_face_512
                else:
                    dim = 2
                    input_face_affined = original_face_256
            elif alpha_res == "512":
                dim = 4
                input_face_affined = original_face_512
            else:
                dim = 2
                input_face_affined = original_face_256

        elif swapper_model in (
            "InStyleSwapper256 Version A",
            "InStyleSwapper256 Version B",
            "InStyleSwapper256 Version C",
        ):
            version = swapper_model[-1]
            latent = (
                torch.from_numpy(
                    self.worker.function_worker.calc_swapper_latent_iss(s_e, version)
                )
                .float()
                .to(self.worker.models_processor.device)
            )
            dst_latent = (
                torch.from_numpy(
                    self.worker.function_worker.calc_swapper_latent_iss(t_e, version)
                )
                .float()
                .to(self.worker.models_processor.device)
            )
            latent = self._apply_likeness(latent, dst_latent, parameters)

            if (
                (
                    parameters["SwapModelSelection"] == "InStyleSwapper256 Version A"
                    and parameters["InStyleResAEnableToggle"]
                )
                or (
                    parameters["SwapModelSelection"] == "InStyleSwapper256 Version B"
                    and parameters["InStyleResBEnableToggle"]
                )
                or (
                    parameters["SwapModelSelection"] == "InStyleSwapper256 Version C"
                    and parameters["InStyleResCEnableToggle"]
                )
            ):
                dim = 4
                input_face_affined = original_face_512
            else:
                dim = 2
                input_face_affined = original_face_256

        elif swapper_model == "SimSwap512":
            latent = (
                torch.from_numpy(
                    self.worker.function_worker.calc_swapper_latent_simswap512(s_e)
                )
                .float()
                .to(self.worker.models_processor.device)
            )
            dst_latent = (
                torch.from_numpy(
                    self.worker.function_worker.calc_swapper_latent_simswap512(t_e)
                )
                .float()
                .to(self.worker.models_processor.device)
            )
            latent = self._apply_likeness(latent, dst_latent, parameters)
            dim = 4
            input_face_affined = original_face_512

        elif swapper_model in self.worker.GHOSTFACE_MODELS:
            latent = (
                torch.from_numpy(
                    self.worker.function_worker.calc_swapper_latent_ghost(s_e)
                )
                .float()
                .to(self.worker.models_processor.device)
            )
            dst_latent = (
                torch.from_numpy(
                    self.worker.function_worker.calc_swapper_latent_ghost(t_e)
                )
                .float()
                .to(self.worker.models_processor.device)
            )
            latent = self._apply_likeness(latent, dst_latent, parameters)
            dim = 2
            input_face_affined = original_face_256

        elif swapper_model == "CSCS":
            latent = (
                torch.from_numpy(
                    self.worker.function_worker.calc_swapper_latent_cscs(s_e)
                )
                .float()
                .to(self.worker.models_processor.device)
            )
            dst_latent = (
                torch.from_numpy(
                    self.worker.function_worker.calc_swapper_latent_cscs(t_e)
                )
                .float()
                .to(self.worker.models_processor.device)
            )
            latent = self._apply_likeness(latent, dst_latent, parameters)
            dim = 2
            input_face_affined = original_face_256

        elif swapper_model == "DeepFaceLive (DFM)" and dfm_model_name:
            self.worker.function_worker.manage_dfm_swapper_model_state(
                "DeepFaceLive (DFM)"
            )
            dfm_model_instance = self.worker.models_processor.load_dfm_model(
                dfm_model_name
            )
            if dfm_model_instance is not None:
                dfm_model_instance._dfm_filename_fallback = dfm_model_name
            latent = []
            input_face_affined = original_face_512
            dim = 4

        return input_face_affined, dfm_model_instance, dim, latent
    def _fix_drift_and_texture(
        self,
        current_face: torch.Tensor,
        first_face: torch.Tensor,
    ) -> torch.Tensor:
        if first_face is None:
            return current_face

        device = current_face.device
        C, H, W = current_face.shape

        if not hasattr(self, "_drift_cache"):
            self._drift_cache: dict[tuple[int, int, str], dict[str, torch.Tensor]] = {}

        cache_key = (H, W, str(device))

        with torch.no_grad():
            curr_bchw = current_face.unsqueeze(0)
            first_bchw = first_face.unsqueeze(0)

            if cache_key not in self._drift_cache:
                y_coords = torch.arange(H, dtype=torch.float32, device=device)
                x_coords = torch.arange(W, dtype=torch.float32, device=device)
                Y_pix, X_pix = torch.meshgrid(y_coords, x_coords, indexing="ij")

                Y_norm = (Y_pix - H / 2) / (H / 2)
                X_norm = (X_pix - W / 2) / (W / 2)

                dist_norm_core = torch.sqrt(X_norm**2 + (Y_norm / 1.2) ** 2)
                core_mask = (
                    torch.exp(-0.5 * (dist_norm_core / 0.4) ** 2)
                    .unsqueeze(0)
                    .unsqueeze(0)
                )

                dist_norm_spatial = torch.sqrt(X_norm**2 + (Y_norm / 0.5) ** 2)
                spatial_mask = torch.exp(
                    -0.5 * (dist_norm_spatial / 0.5) ** 2
                ).unsqueeze(0)
                spatial_mask = torch.clamp(spatial_mask, 0.0, 1.0)

                weights = torch.tensor(
                    [0.299, 0.587, 0.114], dtype=torch.float32, device=device
                ).view(1, 3, 1, 1)

                k_size = 51
                sigma = 15.0
                x_cord = torch.arange(k_size, dtype=torch.float32, device=device)
                x_cord = x_cord - (k_size - 1) / 2.0
                g = torch.exp(-(x_cord**2) / (2 * sigma**2))
                g = g / g.sum()
                blob_kernel = (
                    (g.unsqueeze(1) * g.unsqueeze(0)).unsqueeze(0).unsqueeze(0)
                )

                base_M = torch.eye(3, dtype=torch.float32, device=device)[
                    :2, :
                ].unsqueeze(0)

                self._drift_cache[cache_key] = {
                    "Y_pix": Y_pix,
                    "X_pix": X_pix,
                    "core_mask": core_mask,
                    "spatial_mask": spatial_mask,
                    "weights": weights,
                    "blob_kernel": blob_kernel,
                    "base_M": base_M,
                }

            cache = self._drift_cache[cache_key]
            Y_pix = cache["Y_pix"]
            X_pix = cache["X_pix"]
            core_mask = cache["core_mask"]
            spatial_mask = cache["spatial_mask"]
            weights = cache["weights"]
            blob_kernel = cache["blob_kernel"]
            base_M = cache["base_M"]

            curr_gray = (curr_bchw * weights).sum(dim=1, keepdim=True)
            first_gray = (first_bchw * weights).sum(dim=1, keepdim=True)

            padding = blob_kernel.shape[-1] // 2
            curr_blob = F.conv2d(curr_gray, blob_kernel, padding=padding)
            first_blob = F.conv2d(first_gray, blob_kernel, padding=padding)

            curr_blob_masked = curr_blob * core_mask
            first_blob_masked = first_blob * core_mask

            curr_mass = curr_blob_masked.sum() + 1e-8
            first_mass = first_blob_masked.sum() + 1e-8

            curr_cx = (curr_blob_masked * X_pix).sum() / curr_mass
            curr_cy = (curr_blob_masked * Y_pix).sum() / curr_mass

            first_cx = (first_blob_masked * X_pix).sum() / first_mass
            first_cy = (first_blob_masked * Y_pix).sum() / first_mass

            max_drift = 5.0
            dx = torch.clamp(first_cx - curr_cx, -max_drift, max_drift)
            dy = torch.clamp(first_cy - curr_cy, -max_drift, max_drift)

            dx_view = dx.view(1, 1, 1)
            dy_view = dy.view(1, 1, 1)
            translations = torch.cat([dx_view, dy_view], dim=1)
            M = torch.cat([base_M[:, :, :2], translations], dim=2)

            curr_bchw = kgm.warp_affine(
                curr_bchw,
                M,
                dsize=(H, W),
                mode="bilinear",
                padding_mode="border",
                align_corners=False,
            )
            current_face = curr_bchw.squeeze(0)

            current_face = (current_face * spatial_mask) + (
                first_face * (1.0 - spatial_mask)
            )
            current_face = (current_face * 0.85) + (first_face * 0.15)

            curr_bchw_clamp = current_face.unsqueeze(0).clamp(0.0, 1.0)
            first_bchw_clamp = first_face.unsqueeze(0).clamp(0.0, 1.0)

            curr_lab = kc.rgb_to_lab(curr_bchw_clamp)
            first_lab = kc.rgb_to_lab(first_bchw_clamp)

            stat_mask = core_mask.expand_as(curr_lab)
            mask_sum = stat_mask.sum(dim=(2, 3), keepdim=True) + 1e-8

            mean_curr = (curr_lab * stat_mask).sum(dim=(2, 3), keepdim=True) / mask_sum
            mean_first = (first_lab * stat_mask).sum(
                dim=(2, 3), keepdim=True
            ) / mask_sum

            var_curr = (((curr_lab - mean_curr) ** 2) * stat_mask).sum(
                dim=(2, 3), keepdim=True
            ) / mask_sum
            var_first = (((first_lab - mean_first) ** 2) * stat_mask).sum(
                dim=(2, 3), keepdim=True
            ) / mask_sum

            std_curr = torch.sqrt(var_curr) + 1e-6
            std_first = torch.sqrt(var_first) + 1e-6

            matched_lab = (curr_lab - mean_curr) * (std_first / std_curr) + mean_first
            current_face = kc.lab_to_rgb(matched_lab).squeeze(0)

        return torch.clamp(current_face, 0.0, 1.0)

    def get_swapped_and_prev_face(
        self,
        output: torch.Tensor,
        input_face_affined: torch.Tensor,
        original_face_512: torch.Tensor,
        latent: torch.Tensor | list | None,
        itex: int,
        dim: int,
        swapper_model: str,
        dfm_model: Any | None,
        parameters: dict[str, Any],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if parameters["PreSwapSharpnessDecimalSlider"] != 1.0:
            input_face_affined = input_face_affined.permute(2, 0, 1)
            input_face_affined = v2.functional.adjust_sharpness(
                input_face_affined, parameters["PreSwapSharpnessDecimalSlider"]
            )
            input_face_affined = input_face_affined.permute(1, 2, 0)

        prev_face = input_face_affined
        first_pass_face = None
        use_mode_2 = parameters.get("StrengthMode2EnableToggle", False)

        if swapper_model == "Inswapper128":
            _use_batched = dim > 1

            for k in range(itex):
                prev_face = input_face_affined

                if _use_batched:
                    H_tgt = 128
                    W_tgt = 128

                    batch_input = (
                        input_face_affined.view(H_tgt, dim, W_tgt, dim, 3)
                        .permute(1, 3, 4, 0, 2)
                        .reshape(dim * dim, 3, H_tgt, W_tgt)
                        .contiguous()
                    )

                    batch_output = torch.empty_like(batch_input)

                    self.worker.function_worker.run_inswapper_batched(
                        batch_input, latent, batch_output
                    )

                    if self.worker.models_processor.device_type == "cuda":
                        platform_support.blocking_stream_sync()

                    tile_sums = batch_output.abs().sum(dim=(1, 2, 3), keepdim=True)
                    zero_mask = tile_sums < 1.0
                    batch_output = torch.where(zero_mask, batch_input, batch_output)

                    temp_output = (
                        batch_output.view(dim, dim, 3, H_tgt, W_tgt)
                        .permute(3, 0, 4, 1, 2)
                        .reshape(dim * H_tgt, dim * W_tgt, 3)
                        .contiguous()
                    )

                    if use_mode_2:
                        curr_chw = temp_output.permute(2, 0, 1)
                        if k == 0:
                            first_pass_face = curr_chw.clone()
                        else:
                            curr_chw = self._fix_drift_and_texture(
                                curr_chw, first_pass_face
                            )
                            temp_output = curr_chw.permute(1, 2, 0)

                    input_face_affined = temp_output
                    output = torch.clamp(temp_output * 255.0, 0, 255)

                else:
                    temp_output = input_face_affined.clone()
                    tile_inputs = []
                    tile_outputs = []
                    tile_coords = []

                    for j in range(dim):
                        for i in range(dim):
                            tile = input_face_affined[j::dim, i::dim]
                            t_in = tile.permute(2, 0, 1).contiguous().unsqueeze(0)
                            t_out = torch.empty_like(t_in)
                            tile_inputs.append(t_in)
                            tile_outputs.append(t_out)
                            tile_coords.append((j, i))

                    with torch.no_grad():
                        for idx in range(len(tile_inputs)):
                            self.worker.function_worker.run_inswapper(
                                tile_inputs[idx], latent, tile_outputs[idx]
                            )

                    if self.worker.models_processor.device_type == "cuda":
                        platform_support.blocking_stream_sync()

                    for idx, (j, i) in enumerate(tile_coords):
                        t_in = tile_inputs[idx]
                        t_out = tile_outputs[idx]
                        res = torch.where(t_out.abs().sum() < 1.0, t_in, t_out)
                        temp_output[j::dim, i::dim] = res.squeeze(0).permute(1, 2, 0)

                    if use_mode_2:
                        curr_chw = temp_output.permute(2, 0, 1)
                        if k == 0:
                            first_pass_face = curr_chw.clone()
                        else:
                            curr_chw = self._fix_drift_and_texture(
                                curr_chw, first_pass_face
                            )
                            temp_output = curr_chw.permute(1, 2, 0)

                    input_face_affined = temp_output
                    output = torch.clamp(temp_output * 255.0, 0, 255)

        elif swapper_model == "AlphaFace":
            _use_batched_512 = dim > 2

            for k in range(itex):
                prev_face = input_face_affined

                if _use_batched_512:
                    dim_res = dim // 2
                    H_tgt = 256
                    W_tgt = 256

                    batch_input = (
                        input_face_affined.view(H_tgt, dim_res, W_tgt, dim_res, 3)
                        .permute(1, 3, 4, 0, 2)
                        .reshape(dim_res * dim_res, 3, H_tgt, W_tgt)
                        .contiguous()
                    )

                    batch_output = torch.empty_like(batch_input)

                    if hasattr(
                        self.worker.function_worker, "run_swapper_alphaface_batched"
                    ):
                        self.worker.function_worker.run_swapper_alphaface_batched(
                            batch_input, latent, batch_output
                        )
                    else:
                        for b_i in range(dim_res * dim_res):
                            self.worker.function_worker.run_swapper_alphaface(
                                batch_input[b_i : b_i + 1],
                                latent,
                                batch_output[b_i : b_i + 1],
                            )

                    tile_sums = batch_output.abs().sum(dim=(1, 2, 3), keepdim=True)
                    zero_mask = tile_sums < 1e-4
                    batch_output = torch.where(zero_mask, batch_input, batch_output)

                    temp_output = (
                        batch_output.view(dim_res, dim_res, 3, H_tgt, W_tgt)
                        .permute(3, 0, 4, 1, 2)
                        .reshape(dim_res * H_tgt, dim_res * W_tgt, 3)
                        .contiguous()
                    )

                    if use_mode_2:
                        curr_chw = temp_output.permute(2, 0, 1)
                        if k == 0:
                            first_pass_face = curr_chw.clone()
                        else:
                            curr_chw = self._fix_drift_and_texture(
                                curr_chw, first_pass_face
                            )
                            temp_output = curr_chw.permute(1, 2, 0)

                    input_face_affined = temp_output
                    output = torch.clamp(temp_output * 255.0, 0, 255)

                else:
                    input_face_disc = (
                        input_face_affined.permute(2, 0, 1).unsqueeze(0).contiguous()
                    )
                    swapper_output = torch.empty(
                        (1, 3, 256, 256),
                        dtype=torch.float32,
                        device=self.worker.models_processor.device,
                    )

                    self.worker.function_worker.run_swapper_alphaface(
                        input_face_disc, latent, swapper_output
                    )

                    swapper_output = swapper_output.squeeze(0)
                    valid_output = torch.logical_and(
                        torch.isfinite(swapper_output).all(),
                        swapper_output.abs().mean() >= 1e-4,
                    )
                    swapper_output = torch.where(
                        valid_output,
                        swapper_output,
                        input_face_affined.permute(2, 0, 1),
                    )

                    if use_mode_2:
                        if k == 0:
                            first_pass_face = swapper_output.clone()
                        else:
                            swapper_output = self._fix_drift_and_texture(
                                swapper_output, first_pass_face
                            )

                    input_face_affined = swapper_output.permute(1, 2, 0)
                    output = torch.clamp(input_face_affined * 255.0, 0, 255)

        elif swapper_model in (
            "InStyleSwapper256 Version A",
            "InStyleSwapper256 Version B",
            "InStyleSwapper256 Version C",
        ):
            version = swapper_model[-1]
            dim_res = dim // 2
            for k in range(itex):
                prev_face = input_face_affined
                temp_output = input_face_affined.clone()
                tile_inputs = []
                tile_outputs = []
                tile_coords = []

                for j in range(dim_res):
                    for i in range(dim_res):
                        tile = input_face_affined[j::dim_res, i::dim_res]
                        t_in = tile.permute(2, 0, 1).contiguous().unsqueeze(0)
                        t_out = torch.empty_like(t_in)
                        tile_inputs.append(t_in)
                        tile_outputs.append(t_out)
                        tile_coords.append((j, i))

                with torch.no_grad():
                    for idx in range(len(tile_inputs)):
                        self.worker.function_worker.run_iss_swapper(
                            tile_inputs[idx], latent, tile_outputs[idx], version
                        )

                if self.worker.models_processor.device_type == "cuda":
                    platform_support.blocking_stream_sync()

                for idx, (j, i) in enumerate(tile_coords):
                    res = (
                        tile_inputs[idx]
                        if tile_outputs[idx].sum() < 1.0
                        else tile_outputs[idx]
                    )
                    temp_output[j::dim_res, i::dim_res] = res.squeeze(0).permute(
                        1, 2, 0
                    )

                if use_mode_2:
                    curr_chw = temp_output.permute(2, 0, 1)
                    if k == 0:
                        first_pass_face = curr_chw.clone()
                    else:
                        curr_chw = self._fix_drift_and_texture(
                            curr_chw, first_pass_face
                        )
                        temp_output = curr_chw.permute(1, 2, 0)

                input_face_affined = temp_output
                output = torch.clamp(temp_output * 255.0, 0, 255)

        elif swapper_model == "SimSwap512":
            for k in range(itex):
                prev_face = input_face_affined
                input_face_disc = (
                    input_face_affined.permute(2, 0, 1).unsqueeze(0).contiguous()
                )
                swapper_output = torch.empty(
                    (1, 3, 512, 512),
                    dtype=torch.float32,
                    device=self.worker.models_processor.device,
                ).contiguous()

                self.worker.function_worker.run_swapper_simswap512(
                    input_face_disc, latent, swapper_output
                )
                if self.worker.models_processor.device_type == "cuda":
                    platform_support.blocking_stream_sync()

                if swapper_output.abs().max() < 1e-4:
                    swapper_output = input_face_disc
                swapper_output = swapper_output.squeeze(0)

                if use_mode_2:
                    if k == 0:
                        first_pass_face = swapper_output.clone()
                    else:
                        swapper_output = self._fix_drift_and_texture(
                            swapper_output, first_pass_face
                        )

                swapper_output_hwc = swapper_output.permute(1, 2, 0)
                input_face_affined = swapper_output_hwc
                output = torch.clamp(swapper_output_hwc * 255.0, 0, 255)

        elif swapper_model in self.worker.GHOSTFACE_MODELS:
            for k in range(itex):
                prev_face = input_face_affined

                input_face_disc = torch.mul(input_face_affined, 255.0).permute(2, 0, 1)
                input_face_disc = torch.div(input_face_disc.float(), 127.5)
                input_face_disc = (
                    torch.sub(input_face_disc, 1).unsqueeze(0).contiguous()
                )

                swapper_output = torch.empty(
                    (1, 3, 256, 256),
                    dtype=torch.float32,
                    device=self.worker.models_processor.device,
                ).contiguous()

                self.worker.function_worker.run_swapper_ghostface(
                    input_face_disc, latent, swapper_output, swapper_model
                )
                if self.worker.models_processor.device_type == "cuda":
                    platform_support.blocking_stream_sync()

                swapper_output = swapper_output[0]
                if swapper_output.abs().mean() < 0.01:
                    swapper_output = input_face_affined.permute(2, 0, 1) * 2.0 - 1.0

                if use_mode_2:
                    swapper_output = torch.add(torch.mul(swapper_output, 127.5), 127.5)
                    curr_chw = torch.div(swapper_output, 255.0)

                    if k == 0:
                        first_pass_face = curr_chw.clone()
                    else:
                        curr_chw = self._fix_drift_and_texture(
                            curr_chw, first_pass_face
                        )

                    input_face_affined = curr_chw.permute(1, 2, 0)
                    output = torch.clamp(input_face_affined * 255.0, 0, 255)
                else:
                    swapper_output = swapper_output.permute(1, 2, 0)
                    swapper_output = torch.add(torch.mul(swapper_output, 127.5), 127.5)

                    input_face_affined = torch.div(swapper_output, 255.0)
                    output = torch.clamp(swapper_output, 0, 255)

        elif swapper_model == "CSCS":
            for k in range(itex):
                prev_face = input_face_affined
                input_face_disc = input_face_affined.permute(2, 0, 1)
                input_face_disc = v2.functional.normalize(
                    input_face_disc, (0.5, 0.5, 0.5), (0.5, 0.5, 0.5), inplace=False
                )
                input_face_disc = input_face_disc.unsqueeze(0).contiguous()

                swapper_output = torch.empty(
                    (1, 3, 256, 256),
                    dtype=torch.float32,
                    device=self.worker.models_processor.device,
                ).contiguous()

                self.worker.function_worker.run_swapper_cscs(
                    input_face_disc, latent, swapper_output
                )
                if self.worker.models_processor.device_type == "cuda":
                    platform_support.blocking_stream_sync()

                swapper_output = swapper_output.squeeze(0)
                swapper_output = torch.add(torch.mul(swapper_output, 0.5), 0.5)

                if use_mode_2:
                    if k == 0:
                        first_pass_face = swapper_output.clone()
                    else:
                        swapper_output = self._fix_drift_and_texture(
                            swapper_output, first_pass_face
                        )

                    input_face_affined = swapper_output.permute(1, 2, 0)
                    output = torch.clamp(input_face_affined * 255.0, 0, 255)
                else:
                    input_face_affined = swapper_output.permute(1, 2, 0)
                    output = torch.clamp(input_face_affined * 255.0, 0, 255)

        elif swapper_model == "DeepFaceLive (DFM)" and dfm_model:
            if hasattr(dfm_model, "_cached_dfm_res"):
                dfm_res = dfm_model._cached_dfm_res
            else:
                dfm_res = 256
                queue = deque([dfm_model])
                visited = set([id(dfm_model)])
                found_res = None
                while queue and not found_res:
                    current = queue.popleft()
                    if hasattr(current, "get_inputs") and callable(current.get_inputs):
                        try:
                            shape = current.get_inputs()[0].shape
                            for s in shape:
                                if isinstance(s, int) and s > 32 and s % 16 == 0:
                                    found_res = s
                                    break
                        except Exception:
                            pass
                    if not found_res and hasattr(current, "__dict__"):
                        for k, v in current.__dict__.items():
                            if id(v) not in visited and not k.startswith("__"):
                                visited.add(id(v))
                                queue.append(v)
                if found_res:
                    dfm_res = found_res
                elif hasattr(dfm_model, "_dfm_filename_fallback"):
                    match = re.search(
                        r"(128|192|224|256|320|384|448|512)",
                        dfm_model._dfm_filename_fallback,
                    )
                    if match:
                        dfm_res = int(match.group(1))
                dfm_model._cached_dfm_res = dfm_res

            if dfm_res != 512:
                dfm_input = v2.functional.resize(
                    original_face_512.float(),
                    [dfm_res, dfm_res],
                    interpolation=v2.InterpolationMode.BILINEAR,
                    antialias=True,
                ).to(original_face_512.dtype)
            else:
                dfm_input = original_face_512.clone()

            with self.worker.models_processor.dfm_inference_lock:
                if self.worker.models_processor.device_type == "cuda":
                    platform_support.blocking_stream_sync()
                out_celeb, _, _ = dfm_model.convert(
                    dfm_input,
                    parameters["DFMAmpMorphSlider"] / 100,
                    rct=parameters["DFMRCTColorToggle"],
                )

            if isinstance(out_celeb, np.ndarray):
                out_celeb = torch.from_numpy(out_celeb).to(original_face_512.device)

            if getattr(out_celeb, "ndim", 0) != 3 or out_celeb.shape[2] != 3:
                print(
                    f"[WARN] DFM output shape unexpected: {out_celeb.shape}. Proceeding anyway."
                )

            if out_celeb.max() > 2.0:
                out_celeb_float = out_celeb.float() / 255.0
            else:
                out_celeb_float = out_celeb.float()

            input_face_affined = out_celeb_float
            prev_face = out_celeb_float
            output = out_celeb_float * 255.0

        if output.abs().max() < 30.0:
            print(
                "[WARN] Swap model output near-zero for face — possible VRAM pressure"
            )

        output = output.permute(2, 0, 1)
        assert self.worker.t512 is not None, (
            "t512 transform must be initialized before swapping"
        )
        swap = self.worker.t512(output)
        return swap, prev_face

    _BORDER_MASK_KEYS = (
        "BordermaskEnableToggle",
        "BorderTopSlider",
        "BorderLeftSlider",
        "BorderRightSlider",
        "BorderBottomSlider",
        "TopBorderBlurSlider",
        "BottomBorderBlurSlider",
        "LeftBorderBlurSlider",
        "RightBorderBlurSlider",
    )
    _border_mask_cache: dict = {}

    def get_border_mask(self, parameters: dict) -> tuple[torch.Tensor, torch.Tensor]:
        # The border masks depend only on these slider values, so cache them
        # instead of rebuilding 4 gaussian blurs per face per frame.
        # Callers never modify the returned tensors in place.
        device = self.worker.models_processor.device
        cache_key = (str(device),) + tuple(
            parameters.get(k) for k in self._BORDER_MASK_KEYS
        )
        cached = PipelineProcessor._border_mask_cache.get(cache_key)
        if cached is not None:
            return cached
        result = self._build_border_mask(parameters)
        if len(PipelineProcessor._border_mask_cache) > 32:
            PipelineProcessor._border_mask_cache.clear()
        PipelineProcessor._border_mask_cache[cache_key] = result
        return result

    def _build_border_mask(
        self, parameters: dict
    ) -> tuple[torch.Tensor, torch.Tensor]:
        device = self.worker.models_processor.device

        base_mask = torch.ones((1, 128, 128), dtype=torch.float32, device=device)

        if not parameters.get("BordermaskEnableToggle", False):
            return base_mask, base_mask.clone()

        top: int = parameters["BorderTopSlider"]
        left: int = parameters["BorderLeftSlider"]
        right: int = 128 - parameters["BorderRightSlider"]
        bottom: int = 128 - parameters["BorderBottomSlider"]

        left = max(0, min(left, 128))
        right = max(left, min(right, 128))
        top = max(0, min(top, 128))
        bottom = max(top, min(bottom, 128))

        border_mask_calc = base_mask.clone()
        border_mask_calc[:, :top, :] = 0
        border_mask_calc[:, bottom:, :] = 0
        border_mask_calc[:, :, :left] = 0
        border_mask_calc[:, :, right:] = 0

        mask_top = base_mask.clone()
        mask_top[:, :top, :] = 0

        mask_bottom = base_mask.clone()
        mask_bottom[:, bottom:, :] = 0

        mask_left = base_mask.clone()
        mask_left[:, :, :left] = 0

        mask_right = base_mask.clone()
        mask_right[:, :, right:] = 0

        def apply_blur(mask: torch.Tensor, blur_amount: int) -> torch.Tensor:
            blur_kernel_size = blur_amount * 2 + 1
            if blur_kernel_size > 1:
                sigma_val = float(max(blur_amount * 0.15 + 0.1, 1e-6))
                return v2.functional.gaussian_blur(
                    mask,
                    kernel_size=[blur_kernel_size, blur_kernel_size],
                    sigma=[sigma_val, sigma_val],
                )
            return mask

        mask_top = apply_blur(mask_top, parameters.get("TopBorderBlurSlider", 8))
        mask_bottom = apply_blur(
            mask_bottom, parameters.get("BottomBorderBlurSlider", 8)
        )
        mask_left = apply_blur(mask_left, parameters.get("LeftBorderBlurSlider", 8))
        mask_right = apply_blur(mask_right, parameters.get("RightBorderBlurSlider", 8))

        border_mask = mask_top * mask_bottom * mask_left * mask_right

        return border_mask, border_mask_calc

    def get_dynamic_side_mask(
        self, yaw_deg, pitch_deg, height, width, device, parameters, kps_5, tform
    ):
        mask = torch.ones((1, height, width), dtype=torch.float32, device=device)
        if not parameters.get("ProfileAngleMaskEnableToggle", False):
            return mask

        start_angle = parameters.get("ProfileAngleMaskThresholdSlider", 20)
        max_strength = parameters.get("ProfileAngleMaskStrengthSlider", 100) / 100.0

        if tform is not None:
            kps_proj = tform(kps_5)
            le_x, re_x = kps_proj[0][0], kps_proj[1][0]
        else:
            le_x, re_x = width * 0.35, width * 0.65

        le_x_norm = np.clip(le_x / width, 0.0, 1.0)
        re_x_norm = np.clip(re_x / width, 0.0, 1.0)
        eye_safety_margin = 0.05

        abs_yaw = abs(yaw_deg)
        if abs_yaw > start_angle:
            angle_excess = max(0, abs_yaw - start_angle)
            strength_yaw = min(angle_excess / 45.0, 1.0) * max_strength
            linspace_x = torch.linspace(0, 1, width, device=device).view(1, 1, width)

            if yaw_deg > 0:
                fade_end = max(0.0, le_x_norm - eye_safety_margin)
                if fade_end > 0.05:
                    grad_yaw = torch.clamp(linspace_x / fade_end, 0, 1)
                    grad_yaw = 1.0 - (1.0 - grad_yaw) * strength_yaw
                    mask = mask * grad_yaw
            else:
                fade_start = min(1.0, re_x_norm + eye_safety_margin)
                if fade_start < 0.95:
                    grad_yaw = torch.clamp(
                        (linspace_x - fade_start) / (1.0 - fade_start), 0, 1
                    )
                    grad_yaw = 1.0 - grad_yaw
                    mask_r = torch.ones_like(linspace_x)
                    mask_r[linspace_x > fade_start] = 1.0 - (
                        (linspace_x[linspace_x > fade_start] - fade_start)
                        / (1.0 - fade_start)
                    )
                    grad_yaw = 1.0 - (1.0 - mask_r) * strength_yaw
                    mask = mask * grad_yaw
        return mask
    def _apply_restorer_with_auto(
        self,
        swap: torch.Tensor,
        swap2: torch.Tensor,
        swap_original2: torch.Tensor,
        original_face_512: torch.Tensor,
        mask_forcalc_512: torch.Tensor,
        parameters: dict,
        tform_scale: float,
        debug: bool,
        debug_info: dict,
        slot_id: int,
    ) -> torch.Tensor:
        debug_key = f"Restore2_{slot_id}"
        if parameters["FaceRestorerAutoEnable2Toggle"]:
            alpha_restorer2 = float(parameters["FaceRestorerBlend2Slider"]) / 100.0
            adjust_sharpness2 = float(parameters["FaceRestorerAutoSharpAdjust2Slider"])
            scale_factor2 = round(tform_scale, 2)
            automasktoggle2 = parameters["FaceRestorerAutoMask2EnableToggle"]
            automaskadjust2 = parameters[
                "FaceRestorerAutoSharpMask2AdjustDecimalSlider"
            ]
            automaskblur2 = 2
            restore_mask = mask_forcalc_512.clone()

            alpha_auto2, blur_value2 = self.face_restorer_auto(
                original_face_512.clone(),
                swap_original2.clone(),
                swap2,
                alpha_restorer2,
                adjust_sharpness2,
                scale_factor2,
                debug,
                restore_mask,
                automasktoggle2,
                automaskadjust2,
                automaskblur2,
            )

            if blur_value2 > 0:
                k_size = int(2 * blur_value2 + 1)
                s_val = float(blur_value2 * 0.1)
                swap = v2.functional.gaussian_blur(
                    swap_original2, kernel_size=[k_size, k_size], sigma=[s_val, s_val]
                )
                debug_info[debug_key] = f": {-blur_value2:.2f}"
            elif isinstance(alpha_auto2, torch.Tensor):
                swap = swap2 * alpha_auto2 + swap_original2 * (1 - alpha_auto2)
            elif alpha_auto2 != 0:
                swap = swap2 * alpha_auto2 + swap_original2 * (1 - alpha_auto2)
                if debug:
                    debug_info[debug_key] = f": {alpha_auto2 * 100:.2f}"
            else:
                swap = swap_original2
                if debug:
                    debug_info[debug_key] = f": {alpha_auto2 * 100:.2f}"
        else:
            alpha_restorer2 = float(parameters["FaceRestorerBlend2Slider"]) / 100.0
            swap = (
                torch.lerp(swap_original2.float(), swap2.float(), alpha_restorer2)
                .to(swap2.dtype)
                .contiguous()
            )
        return swap

    def _detect_mouth_action_score(
        self,
        face_bbox: "np.ndarray | None" = None,
        vr_crop_chw: "torch.Tensor | np.ndarray | None" = None,
    ) -> "float | None":
        from app.processors.mouth_action_detector import MouthActionDetector

        detector = MouthActionDetector.ensure_loaded_async(self.worker.main_window)
        if not detector.available:
            if not getattr(self, "_mouth_action_detector_warned", False):
                err = detector.load_error or "unknown error"
                print(f"[WARN] Mouth action detector unavailable: {err}")
                self._mouth_action_detector_warned = True
            return None

        if vr_crop_chw is not None:
            if isinstance(vr_crop_chw, torch.Tensor):
                img_chw_np = vr_crop_chw.cpu().numpy().astype(np.uint8)
            else:
                img_chw_np = np.asarray(vr_crop_chw, dtype=np.uint8)
        elif face_bbox is not None and self.worker.frame is not None:
            frame_hwc = self.worker.frame
            fh, fw = frame_hwc.shape[:2]
            x1, y1, x2, y2 = (
                float(face_bbox[0]),
                float(face_bbox[1]),
                float(face_bbox[2]),
                float(face_bbox[3]),
            )

            _bbox_max = max(x2, y2)
            if _bbox_max > max(fw, fh) + 8 and (fw < 512 or fh < 512):
                if fw <= fh:
                    _new_w, _new_h = 512, int(512 * fh / fw)
                else:
                    _new_h, _new_w = 512, int(512 * fw / fh)
                _ratio_w_down = fw / float(_new_w)
                _ratio_h_down = fh / float(_new_h)
                x1 *= _ratio_w_down
                x2 *= _ratio_w_down
                y1 *= _ratio_h_down
                y2 *= _ratio_h_down
            cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0
            half_w, half_h = x2 - x1, y2 - y1
            crop_x1, crop_y1 = max(0, int(cx - half_w)), max(0, int(cy - half_h))
            crop_x2, crop_y2 = min(fw, int(cx + half_w)), min(fh, int(cy + half_h))
            crop_hwc = frame_hwc[crop_y1:crop_y2, crop_x1:crop_x2]
            if crop_hwc.size == 0:
                return None
            img_chw_np = np.transpose(crop_hwc, (2, 0, 1))
        else:
            img_chw_np = np.transpose(self.worker.frame, (2, 0, 1))

        raw_score = detector.score(img_chw_np)
        return raw_score if raw_score > 0.0 else None

    def _apply_auto_mouth(
        self,
        params: dict,
        target_fb: Any,
        face_bbox: "np.ndarray | None" = None,
        vr_crop_chw: "torch.Tensor | np.ndarray | None" = None,
    ) -> dict:
        if not params.get("AutoMouthExpressionEnableToggle", False):
            return params

        from app.processors.mouth_openness import MouthOpennessState

        _alpha = params.get("AutoMouthEMAAlphaDecimalSlider", 0.65)
        _threshold = params.get("AutoMouthOpenThresholdDecimalSlider", 0.50)
        _ratio: "float | None" = self._detect_mouth_action_score(
            face_bbox=face_bbox, vr_crop_chw=vr_crop_chw
        )

        _state: "MouthOpennessState | None" = getattr(
            target_fb, "mouth_openness_state", None
        )
        if _state is None or self.worker.frame_number == 0:
            target_fb.mouth_openness_state = MouthOpennessState()
            _state = target_fb.mouth_openness_state

        _single_frame_mode = bool(getattr(self.worker, "is_single_frame", False))
        _auto_active, _ema_value = _state.update(
            _ratio, _alpha, _threshold, single_frame_mode=_single_frame_mode
        )

        _exclude_upper_teeth = bool(
            params.get("AutoMouthExcludeUpperTeethToggle", False)
        )
        _was_auto_active = bool(getattr(target_fb, "_auto_mouth_prev_active", False))
        target_fb._auto_mouth_prev_active = _auto_active

        if not _auto_active:
            return params

        _base_strength = params.get("AutoMouthExpressionStrengthDecimalSlider", 1.00)
        _normalize = params.get("AutoMouthNormalizeLipsToggle", True)
        _restore_mode = params.get(
            "AutoMouthRestoreModeSelection", "Expression Restorer"
        )
        _ramp_range = max(_threshold * 0.5, 0.04)
        _proportion = min(1.0, max(0.0, (_ema_value - _threshold) / _ramp_range))
        _strength = _base_strength * _proportion

        if _strength < 0.01:
            if _exclude_upper_teeth:
                _p = dict(params)
                _p["FaceParserEnableToggle"] = True
                _p["AutoMouthUpperTeethExcludeActive"] = True
                return _p
            return params

        _p = dict(params)

        _mouth_val = int(params.get("AutoMouthMouthParserSlider", 1))
        _upper_val = int(params.get("AutoMouthUpperLipParserSlider", 3))
        _lower_val = int(params.get("AutoMouthLowerLipParserSlider", 17))
        if _mouth_val > 0 or _upper_val > 0 or _lower_val > 0 or _exclude_upper_teeth:
            _p["FaceParserEnableToggle"] = True
        _p["MouthParserSlider"] = _mouth_val
        _p["UpperLipParserSlider"] = _upper_val
        _p["LowerLipParserSlider"] = _lower_val
        _p["AutoMouthUpperTeethExcludeActive"] = _exclude_upper_teeth

        if _restore_mode == "Face Parser Only":
            return _p

        _user_enabled = params.get("FaceExpressionEnableBothToggle", False)
        _user_mode = params.get("FaceExpressionModeSelection", "Advanced")
        _user_region = str(params.get("FaceExpressionAnimationRegionSelection", "all"))

        _p["FaceExpressionEnableBothToggle"] = True
        _p["FaceExpressionBeforeTypeSelection"] = "Beginning"

        if _user_enabled and _user_mode == "Advanced":
            _p["FaceExpressionModeSelection"] = "Advanced"
            _p["FaceExpressionLipsToggle"] = True
            _p["FaceExpressionFriendlyFactorLipsDecimalSlider"] = _strength
            _p["FaceExpressionRelativeLipsToggle"] = True
            _p["FaceExpressionRetargetingLipsBothEnableToggle"] = _normalize
            _p["FaceExpressionRetargetingLipsMultiplierBothDecimalSlider"] = 1.0

        elif _user_enabled and _user_mode == "Simple":
            _had_eyes = "eyes" in _user_region or "all" in _user_region
            _user_factor = float(
                params.get("FaceExpressionFriendlyFactorDecimalSlider", 1.0)
            )
            _p["FaceExpressionModeSelection"] = "Advanced"
            _p["FaceExpressionLipsToggle"] = True
            _p["FaceExpressionFriendlyFactorLipsDecimalSlider"] = _strength
            _p["FaceExpressionRelativeLipsToggle"] = True
            _p["FaceExpressionRetargetingLipsBothEnableToggle"] = _normalize
            _p["FaceExpressionRetargetingLipsMultiplierBothDecimalSlider"] = 1.0
            _p["FaceExpressionEyesToggle"] = _had_eyes
            if _had_eyes:
                _p["FaceExpressionFriendlyFactorEyesDecimalSlider"] = _user_factor
                _p["FaceExpressionRelativeEyesToggle"] = True
            _p["FaceExpressionBrowsToggle"] = False
            _p["FaceExpressionGeneralToggle"] = False

        else:
            _region = params.get("AutoMouthAnimationRegionSelection", "lips")
            _p["FaceExpressionModeSelection"] = "Simple"
            _p["FaceExpressionAnimationRegionSelection"] = _region
            _p["FaceExpressionFriendlyFactorDecimalSlider"] = _strength
            _p["FaceExpressionNormalizeLipsEnableToggle"] = _normalize
            _p["FaceExpressionNormalizeLipsThresholdDecimalSlider"] = 0.03
            _p["FaceExpressionNeutralDecimalSlider"] = 1.0
            _p["FaceExpressionLipsToggle"] = False
            _p["FaceExpressionEyesToggle"] = False
            _p["FaceExpressionBrowsToggle"] = False
            _p["FaceExpressionGeneralToggle"] = False

        return _p

    @torch.no_grad()
    def _blend_frequency_separation(
        self,
        primary_face: torch.Tensor,
        secondary_face: torch.Tensor,
        alpha: float,
        radius: int = 4,
        coring_limit: float = 16.0,
    ) -> torch.Tensor:
        if alpha <= 0.0:
            return primary_face

        device: torch.device = primary_face.device
        p_f: torch.Tensor = primary_face.float().clamp(0.0, 255.0) / 255.0
        s_f: torch.Tensor = secondary_face.float().clamp(0.0, 255.0) / 255.0

        p_lab: torch.Tensor = kc.rgb_to_lab(p_f.unsqueeze(0))
        s_lab: torch.Tensor = kc.rgb_to_lab(s_f.unsqueeze(0))

        L_p: torch.Tensor = p_lab[:, 0:1, :, :]
        L_s: torch.Tensor = s_lab[:, 0:1, :, :]
        AB_p: torch.Tensor = p_lab[:, 1:3, :, :]

        r: int = max(1, int(radius))
        k_size: int = int(r * 2 + 1)
        sigma_val: float = float(max(r * 0.5, 0.1))

        L_p_low: torch.Tensor = v2.functional.gaussian_blur(
            L_p, kernel_size=[k_size, k_size], sigma=[sigma_val, sigma_val]
        )
        L_s_low: torch.Tensor = v2.functional.gaussian_blur(
            L_s, kernel_size=[k_size, k_size], sigma=[sigma_val, sigma_val]
        )

        L_p_high: torch.Tensor = L_p - L_p_low
        L_s_high: torch.Tensor = L_s - L_s_low

        coring_scaled: float = float(coring_limit) * (100.0 / 255.0)
        L_s_high_cored: torch.Tensor = coring_scaled * torch.tanh(
            L_s_high / max(coring_scaled, 1e-4)
        )

        low_diff: torch.Tensor = torch.abs(L_p_low - L_s_low)
        diff_min: float = 10.0 * (100.0 / 255.0)
        diff_span: float = 20.0 * (100.0 / 255.0)
        disc_gate: torch.Tensor = torch.clamp(
            1.0 - (low_diff - diff_min) / diff_span, 0.0, 1.0
        )

        sobel_x: torch.Tensor = self.kernel_sobel_x.to(device)
        sobel_y: torch.Tensor = self.kernel_sobel_y.to(device)

        gx_p: torch.Tensor = F.conv2d(L_p_low / 100.0, sobel_x, padding=1)
        gy_p: torch.Tensor = F.conv2d(L_p_low / 100.0, sobel_y, padding=1)
        grad_p: torch.Tensor = torch.sqrt(gx_p * gx_p + gy_p * gy_p + 1e-6)

        gx_s: torch.Tensor = F.conv2d(L_s_low / 100.0, sobel_x, padding=1)
        gy_s: torch.Tensor = F.conv2d(L_s_low / 100.0, sobel_y, padding=1)
        grad_s: torch.Tensor = torch.sqrt(gx_s * gx_s + gy_s * gy_s + 1e-6)

        macro_edges: torch.Tensor = torch.maximum(grad_p, grad_s)
        edge_gate: torch.Tensor = torch.clamp(
            1.0 - (macro_edges - 0.10) / 0.20, 0.0, 1.0
        )

        transfer_mask: torch.Tensor = disc_gate * edge_gate
        transfer_mask = v2.functional.gaussian_blur(
            transfer_mask, kernel_size=[5, 5], sigma=[1.5, 1.5]
        )

        effective_alpha: torch.Tensor = float(alpha) * transfer_mask
        L_high_blended: torch.Tensor = torch.lerp(
            L_p_high, L_s_high_cored, effective_alpha
        )
        L_final: torch.Tensor = torch.clamp(L_p_low + L_high_blended, 0.0, 100.0)

        lab_final: torch.Tensor = torch.cat([L_final, AB_p], dim=1)
        rgb_final: torch.Tensor = kc.lab_to_rgb(lab_final).squeeze(0)

        return (
            torch.clamp(rgb_final * 255.0, 0.0, 255.0)
            .to(primary_face.dtype)
            .contiguous()
        )
    @torch.no_grad()
    def swap_core(
        self,
        img: torch.Tensor,
        kps_5: np.ndarray,
        kps: np.ndarray | None = None,
        kps_203: np.ndarray | None = None,
        s_e: np.ndarray | None = None,
        t_e: np.ndarray | None = None,
        parameters: dict[str, Any] | None = None,
        control: dict[str, Any] | None = None,
        dfm_model_name: str | None = None,
        is_perspective_crop: bool = False,
        kv_map: dict[str, Any] | None = None,
        yawnet_deg: float | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor | None]:
        valid_s_e = s_e if isinstance(s_e, np.ndarray) else None
        valid_t_e = t_e if isinstance(t_e, np.ndarray) else None
        parameters = parameters if parameters is not None else {}
        control = control if control is not None else {}
        swapper_model = parameters["SwapModelSelection"]
        itex = 1

        def _resize_func(
            tensor: torch.Tensor, target_shape: tuple[int, int], is_mask: bool = True
        ) -> torch.Tensor:
            return v2.functional.resize(
                tensor,
                [target_shape[0], target_shape[1]],
                interpolation=v2.InterpolationMode.BILINEAR,
                antialias=not is_mask,
            )

        debug = False
        debug_info: dict[str, str] = {}

        tform = self.worker.get_face_similarity_tform(swapper_model, kps_5)
        tform_original = tform

        t512_mask = self.worker.t512_mask
        t128_mask = self.worker.t128_mask
        assert t512_mask is not None, (
            "t512_mask must be initialized via set_scaling_transforms before swap_core"
        )
        assert t128_mask is not None, (
            "t128_mask must be initialized via set_scaling_transforms before swap_core"
        )

        _face_interp = (
            "bicubic"
            if parameters.get("FaceAlignmentInterpolation", "Bilinear") == "Bicubic"
            else "bilinear"
        )

        sec_enabled = parameters.get("SecondarySwapperEnableToggle", False)
        sec_model = parameters.get("SecondarySwapModelSelection", "AlphaFace")
        _only_256 = (swapper_model == "AlphaFace") and (
            not sec_enabled or sec_model == "AlphaFace"
        )

        original_face_512, original_face_384, original_face_256, original_face_128 = (
            self.worker.get_transformed_and_scaled_faces(
                tform, img, interp_mode=_face_interp, only_256=_only_256
            )
        )

        alphaface_scale_val = 1.0
        if parameters.get("FaceAdjEnableToggle", False):
            scale_val = 1.0 + (parameters.get("FaceScaleAmountSlider", 0) / 100.0)
            if scale_val != 1.0:
                if swapper_model == "AlphaFace":
                    alphaface_scale_val = scale_val
                else:
                    _, h, w = original_face_512.shape
                    new_h, new_w = (
                        max(1, int(h * scale_val)),
                        max(1, int(w * scale_val)),
                    )

                    resized = v2.functional.resize(
                        original_face_512,
                        [new_h, new_w],
                        interpolation=v2.InterpolationMode.BICUBIC,
                        antialias=True,
                    )

                    if scale_val < 1.0:
                        pad_l = (w - new_w) // 2
                        pad_r = w - new_w - pad_l
                        pad_t = (h - new_h) // 2
                        pad_b = h - new_h - pad_t
                        original_face_512 = F.pad(
                            resized,
                            (pad_l, pad_r, pad_t, pad_b),
                            mode="constant",
                            value=0,
                        )
                    else:
                        crop_t = (new_h - h) // 2
                        crop_l = (new_w - w) // 2
                        original_face_512 = v2.functional.crop(
                            resized, crop_t, crop_l, h, w
                        )

                    original_face_256 = _resize_func(
                        original_face_512, (256, 256), is_mask=False
                    )
                    if _only_256:
                        original_face_384 = original_face_512
                        original_face_128 = original_face_256
                    else:
                        original_face_384 = _resize_func(
                            original_face_512, (384, 384), is_mask=False
                        )
                        original_face_128 = _resize_func(
                            original_face_512, (128, 128), is_mask=False
                        )

                    scale_matrix = np.array(
                        [
                            [scale_val, 0.0, 256.0 * (1.0 - scale_val)],
                            [0.0, scale_val, 256.0 * (1.0 - scale_val)],
                            [0.0, 0.0, 1.0],
                        ],
                        dtype=np.float32,
                    )

                    tform = trans.SimilarityTransform(
                        matrix=scale_matrix @ tform.params
                    )

        kps_all_crop = None
        if kps_203 is not None and len(kps_203) == 203 and not np.all(kps_203 == 0):
            raw_kps_crop = tform(kps_203)
            kps_all_crop = np.array(raw_kps_crop, dtype=np.float32)
        elif kps is not None and len(kps) > 0 and not np.all(kps == 0):
            raw_kps_crop = tform(kps)
            kps_all_crop = np.array(raw_kps_crop, dtype=np.float32)

        M_ref = cast(np.ndarray, tform.params)[0:2]
        ones_column_ref = np.ones((kps_5.shape[0], 1), dtype=np.float32)
        kps_ref = np.hstack([kps_5, ones_column_ref]) @ M_ref.T

        apply_pre_swap_shaping = (
            parameters.get("FaceShapingEnableToggle", False)
            and parameters.get("FaceShapingPreSwapToggle", False)
            and self.worker.local_control_state_from_feeder.get("edit_enabled", True)
        )
        if apply_pre_swap_shaping:
            original_face_512 = self.worker.function_worker.apply_face_shaping_gpu(
                original_face_512, kps_ref, parameters
            )

            kps_all_crop = None

            original_face_256 = _resize_func(
                original_face_512, (256, 256), is_mask=False
            )
            if _only_256:
                original_face_384 = original_face_512
                original_face_128 = original_face_256
            else:
                original_face_384 = _resize_func(
                    original_face_512, (384, 384), is_mask=False
                )
                original_face_128 = _resize_func(
                    original_face_512, (128, 128), is_mask=False
                )

        original_faces = (
            original_face_512,
            original_face_384,
            original_face_256,
            original_face_128,
        )
        swap = original_face_512
        prev_face = torch.div(original_face_512.float(), 255.0).permute(1, 2, 0)

        if valid_s_e is not None or (
            swapper_model == "DeepFaceLive (DFM)" and dfm_model_name
        ):
            input_face_affined, dfm_model_instance, dim, latent = (
                self.get_affined_face_dim_and_swapping_latents(
                    original_faces,
                    swapper_model,
                    dfm_model_name,
                    valid_s_e,
                    valid_t_e,
                    parameters,
                    debug,
                    tform,
                )
            )

            if input_face_affined is None:
                swap = original_face_512
            else:
                itex = 1
                if parameters.get("StrengthEnableToggle", False):
                    itex = ceil(parameters["StrengthAmountSlider"] / 100.0)

                output_size = int(128 * dim)
                output = torch.zeros(
                    (output_size, output_size, 3),
                    dtype=torch.float32,
                    device=self.worker.models_processor.device,
                )
                input_face_affined = input_face_affined.permute(1, 2, 0).contiguous()
                input_face_affined = torch.div(input_face_affined, 255.0)

                swap, prev_face = self.get_swapped_and_prev_face(
                    output,
                    input_face_affined,
                    original_face_512,
                    latent,
                    itex,
                    dim,
                    swapper_model,
                    dfm_model_instance,
                    parameters,
                )

                if parameters.get("StrengthEnableToggle", False):
                    if itex == 0:
                        swap = original_face_512.clone()
                    else:
                        alpha = np.mod(parameters["StrengthAmountSlider"], 100) * 0.01
                        if alpha == 0:
                            alpha = 1.0
                        prev_face = (
                            torch.mul(prev_face, 255.0).clamp(0, 255).permute(2, 0, 1)
                        )
                        if prev_face.shape[-1] != swap.shape[-1]:
                            prev_face = _resize_func(
                                prev_face,
                                (swap.shape[-2], swap.shape[-1]),
                                is_mask=False,
                            )
                        swap = (
                            torch.lerp(prev_face.float(), swap.float(), alpha)
                            .to(swap.dtype)
                            .contiguous()
                        )

                _COMPATIBLE_ARCFACE_MODELS: frozenset[str] = frozenset(
                    {
                        "Inswapper128",
                        "AlphaFace",
                        "InStyleSwapper256 Version A",
                        "InStyleSwapper256 Version B",
                        "InStyleSwapper256 Version C",
                    }
                )

                if (
                    sec_enabled
                    and swapper_model in _COMPATIBLE_ARCFACE_MODELS
                    and sec_model in _COMPATIBLE_ARCFACE_MODELS
                ):
                    sec_parameters = dict(parameters)
                    sec_res_choice = parameters.get(
                        "SecondarySwapperResSelection", "Auto"
                    )
                    sec_parameters["SwapperResSelection"] = sec_res_choice
                    sec_parameters["AlphaFaceResSelection"] = sec_res_choice

                    strength_enabled = parameters.get("StrengthEnableToggle", False)
                    if strength_enabled:
                        sec_amount = parameters.get(
                            "SecondaryStrengthAmountSlider", 100
                        )
                    else:
                        sec_amount = 100

                    sec_parameters["StrengthEnableToggle"] = strength_enabled
                    sec_parameters["StrengthMode2EnableToggle"] = parameters.get(
                        "StrengthMode2EnableToggle", False
                    )
                    sec_parameters["StrengthAmountSlider"] = sec_amount

                    sec_input, sec_dfm, sec_dim, sec_latent = (
                        self.get_affined_face_dim_and_swapping_latents(
                            original_faces,
                            sec_model,
                            dfm_model_name,
                            valid_s_e,
                            valid_t_e,
                            sec_parameters,
                            debug,
                            tform,
                        )
                    )

                    if sec_input is not None:
                        sec_itex = ceil(sec_amount / 100.0)

                        if sec_itex == 0:
                            sec_swap = original_face_512.clone()
                            del sec_input, sec_latent
                        else:
                            sec_output_size = int(128 * sec_dim)
                            sec_output = torch.zeros(
                                (sec_output_size, sec_output_size, 3),
                                dtype=torch.float32,
                                device=self.worker.models_processor.device,
                            )
                            sec_input_chw = (
                                sec_input.permute(1, 2, 0).contiguous().div(255.0)
                            )

                            sec_swap, sec_prev_face = self.get_swapped_and_prev_face(
                                sec_output,
                                sec_input_chw,
                                original_face_512,
                                sec_latent,
                                sec_itex,
                                sec_dim,
                                sec_model,
                                sec_dfm,
                                sec_parameters,
                            )

                            if strength_enabled:
                                sec_alpha = np.mod(sec_amount, 100) * 0.01
                                if sec_alpha == 0:
                                    sec_alpha = 1.0

                                sec_prev_face = (
                                    torch.mul(sec_prev_face, 255.0)
                                    .clamp(0, 255)
                                    .permute(2, 0, 1)
                                )
                                if sec_prev_face.shape[-1] != sec_swap.shape[-1]:
                                    sec_prev_face = _resize_func(
                                        sec_prev_face,
                                        (sec_swap.shape[-2], sec_swap.shape[-1]),
                                        is_mask=False,
                                    )
                                sec_swap = (
                                    torch.lerp(
                                        sec_prev_face.float(),
                                        sec_swap.float(),
                                        sec_alpha,
                                    )
                                    .to(sec_swap.dtype)
                                    .contiguous()
                                )

                            del (
                                sec_output,
                                sec_input,
                                sec_input_chw,
                                sec_latent,
                                sec_prev_face,
                            )

                        if sec_swap.shape[-1] != swap.shape[-1]:
                            sec_swap = _resize_func(
                                sec_swap,
                                (swap.shape[-2], swap.shape[-1]),
                                is_mask=False,
                            )

                        blend_alpha = (
                            float(
                                parameters.get("SecondarySwapperBlendAmountSlider", 50)
                            )
                            / 100.0
                        )

                        if parameters.get("SecondaryTextureOnlyEnableToggle", False):
                            tex_radius = int(
                                parameters.get("SecondaryTextureRadiusSlider", 4)
                            )
                            tex_coring = float(
                                parameters.get(
                                    "SecondaryTextureCoringDecimalSlider", 16.0
                                )
                            )
                            swap = self._blend_frequency_separation(
                                primary_face=swap,
                                secondary_face=sec_swap,
                                alpha=blend_alpha,
                                radius=tex_radius,
                                coring_limit=tex_coring,
                            )
                        else:
                            swap = (
                                torch.lerp(swap.float(), sec_swap.float(), blend_alpha)
                                .to(swap.dtype)
                                .contiguous()
                            )
                        del sec_swap
                elif sec_enabled:
                    if debug:
                        debug_info["Sec_Swapper_Skipped"] = "Incompatible ArcFace Model"
        else:
            swap = original_face_512
        current_swap_h, current_swap_w = swap.shape[1], swap.shape[2]
        yaw_deg, pitch_deg = faceutil.calc_face_yaw_pitch(kps_5)

        if yawnet_deg is not None:
            guard_yaw_deg = self.worker.function_worker.yaw_from_frontal(yawnet_deg)
        else:
            guard_yaw_deg = min(abs(float(yaw_deg)), 90.0)
        side_mask = self.get_dynamic_side_mask(
            yaw_deg,
            pitch_deg,
            current_swap_h,
            current_swap_w,
            self.worker.models_processor.device,
            parameters,
            kps_5,
            tform,
        )

        if parameters.get("BordermaskEnableToggle", False):
            border_mask, border_mask_calc = self.get_border_mask(parameters)
            if (
                border_mask.shape[1] != current_swap_h
                or border_mask.shape[2] != current_swap_w
            ):
                border_mask = _resize_func(
                    border_mask, (current_swap_h, current_swap_w), is_mask=True
                )
                border_mask_calc = _resize_func(
                    border_mask_calc, (current_swap_h, current_swap_w), is_mask=True
                )
            border_mask = border_mask * side_mask
            border_mask_calc = border_mask_calc * side_mask
        else:
            border_mask = side_mask
            border_mask_calc = side_mask

        swap_mask = torch.ones(
            (1, current_swap_h, current_swap_w),
            dtype=torch.float32,
            device=self.worker.models_processor.device,
        )
        swap_mask_noFP = border_mask.clone()
        core_stats_mask = torch.ones(
            (1, current_swap_h, current_swap_w),
            dtype=torch.float32,
            device=self.worker.models_processor.device,
        )

        base_shape = (1, 512, 512)
        diff_mask = torch.ones(
            base_shape, dtype=torch.float32, device=self.worker.models_processor.device
        )
        texture_mask_view = torch.ones(
            base_shape, dtype=torch.float32, device=self.worker.models_processor.device
        )
        texture_exclude_512 = torch.ones(
            base_shape, dtype=torch.float32, device=self.worker.models_processor.device
        )

        calc_mask = core_stats_mask
        calc_mask_dill = core_stats_mask
        mask_forcalc_512 = core_stats_mask

        M_ref = cast(np.ndarray, tform.params)[0:2]
        ones_column_ref = np.ones((kps_5.shape[0], 1), dtype=np.float32)
        kps_ref = np.hstack([kps_5, ones_column_ref]) @ M_ref.T

        swap = torch.clamp(swap, 0.0, 255.0)

        if (
            parameters["FaceExpressionEnableBothToggle"]
            and (
                parameters["FaceExpressionLipsToggle"]
                or parameters["FaceExpressionEyesToggle"]
                or parameters["FaceExpressionBrowsToggle"]
                or parameters["FaceExpressionGeneralToggle"]
                or parameters.get("FaceExpressionModeSelection", "Advanced")
                in ("Simple", "Recast")
            )
            and parameters["FaceExpressionBeforeTypeSelection"] == "Beginning"
        ):
            if parameters.get("FaceExpressionModeSelection", "Advanced") == "Recast":
                swap = self.worker.function_worker.apply_perform_recast(
                    original_face_512,
                    swap,
                    cast(dict, parameters),
                    cast(dict, control),
                    driving_kps=kps_all_crop,
                )
            else:
                swap = self.worker.function_worker.apply_face_expression_restorer(
                    original_face_512,
                    swap,
                    cast(dict, parameters),
                    cast(dict, control),
                    driving_kps=kps_all_crop,
                )

        if (
            parameters["FaceEditorEnableToggle"]
            and self.worker.local_control_state_from_feeder.get("edit_enabled", True)
            and parameters["FaceEditorBeforeTypeSelection"] == "Beginning"
        ):
            editor_mask = swap_mask.clone()
            swap = (
                torch.lerp(original_face_512.float(), swap.float(), editor_mask)
                .to(swap.dtype)
                .contiguous()
            )
            swap = self.worker.function_worker.swap_edit_face_core(
                swap, swap, parameters, control
            )

        if (
            parameters.get("FaceShapingEnableToggle", False)
            and not parameters.get("FaceShapingPreSwapToggle", False)
            and self.worker.local_control_state_from_feeder.get("edit_enabled", True)
            and parameters.get("FaceEditorBeforeTypeSelection", "Beginning")
            == "Beginning"
        ):
            swap = self.worker.function_worker.apply_face_shaping_gpu(
                swap, kps_ref, parameters
            )

        if control.get("DenoiserUNetEnableBeforeRestorersToggle", False):
            swap = self._apply_denoiser_pass(
                swap,
                control,
                "Before",
                kv_map,
                color_mask=mask_forcalc_512,
                blend_mask=swap_mask,
            )

        paste_after_restorer = parameters.get("MouthParserStretchAfterToggle", False)
        if not paste_after_restorer:
            mouth_overlay_pkg = None
            if hasattr(self.worker.function_worker, "get_mouth_overlay"):
                mouth_overlay_pkg = self.worker.function_worker.get_mouth_overlay(
                    swap, original_face_512, parameters
                )

            if mouth_overlay_pkg is not None:
                overlay_rgb, overlay_mask = mouth_overlay_pkg
                if overlay_rgb is not None and overlay_mask is not None:
                    if overlay_rgb.shape[-1] != swap.shape[-1]:
                        overlay_rgb = _resize_func(
                            overlay_rgb, (swap.shape[-2], swap.shape[-1]), is_mask=False
                        )
                        overlay_mask = _resize_func(
                            overlay_mask.unsqueeze(0),
                            (swap.shape[-2], swap.shape[-1]),
                            is_mask=True,
                        ).squeeze(0)
                    swap = swap * (1.0 - overlay_mask) + overlay_rgb * overlay_mask

        swap_original = None
        if parameters["FaceRestorerEnableToggle"]:
            swap_original = swap.clone()
            swap_restorecalc = self.worker.function_worker.apply_facerestorer(
                swap,
                parameters["FaceRestorerDetTypeSelection"],
                parameters["FaceRestorerTypeSelection"],
                parameters["FaceRestorerBlendSlider"],
                parameters["FaceFidelityWeightDecimalSlider"],
                control["DetectorScoreSlider"],
                kps_ref,
                slot_id=1,
                osdface_timestep=parameters.get("OSDFaceTimestepSlider", 399),
                osdface_latent_strength=parameters.get(
                    "OSDFaceLatentStrengthDecimalSlider", 1.0
                ),
            )
        else:
            swap_restorecalc = swap.clone()

        if parameters["OccluderEnableToggle"]:
            mask = self.worker.function_worker.apply_occlusion(
                original_face_256,
                parameters["OccluderSizeSlider"],
                parameters=parameters,
                original_face_512=swap_restorecalc,
                yaw_deg=guard_yaw_deg,
            )
            if mask.shape[-1] != swap_mask.shape[-1]:
                mask = _resize_func(
                    mask, (swap_mask.shape[-2], swap_mask.shape[-1]), is_mask=True
                )
            swap_mask.mul_(mask)

            k_size = int(parameters["OccluderXSegBlurSlider"] * 2 + 1)
            s_val = float((parameters["OccluderXSegBlurSlider"] + 1) * 0.2)
            swap_mask = v2.functional.gaussian_blur(
                swap_mask, kernel_size=[k_size, k_size], sigma=[s_val, s_val]
            )

            if swap_mask_noFP.shape[-1] != swap_mask.shape[-1]:
                swap_mask_noFP = _resize_func(
                    swap_mask_noFP,
                    (swap_mask.shape[-2], swap_mask.shape[-1]),
                    is_mask=True,
                )
            swap_mask_noFP.mul_(swap_mask)

        need_any_parser = (
            parameters.get("FaceParserEnableToggle", False)
            or (
                parameters.get("DFLXSegEnableToggle", False)
                and (
                    (
                        parameters.get("XSegMouthEnableToggle", False)
                        and parameters.get("DFLXSegSizeSlider", 0)
                        != parameters.get("DFLXSeg2SizeSlider", 0)
                    )
                    or parameters.get("XSegExcludeInnerMouthToggle", False)
                )
            )
            or (
                (
                    parameters.get("TransferTextureEnableToggle", False)
                    or parameters.get("DifferencingEnableToggle", False)
                )
                and parameters.get("ExcludeMaskEnableToggle", False)
            )
        )

        FaceParser_mask = None
        mouth_512 = None
        mouth_debug_512 = None
        mouth_debug_teeth_512 = None
        inner_mouth_protection_512 = None

        if need_any_parser:
            out = self.worker.function_worker.process_masks_and_masks(
                swap_restorecalc, original_face_512, parameters, control
            )
            if not parameters.get("FaceParserEndToggle", False):
                FaceParser_mask = out.get("FaceParser_mask", None)
            texture_exclude_512 = out.get("texture_mask", texture_exclude_512)
            mouth_512 = out.get("mouth", None)
            mouth_debug_512 = out.get("mouth_debug", None)
            mouth_debug_teeth_512 = out.get("mouth_debug_teeth", None)
            inner_mouth_protection_512 = out.get("inner_mouth_protection", None)

        if FaceParser_mask is not None:
            if FaceParser_mask.shape[-1] != swap_mask.shape[-1]:
                FaceParser_mask = _resize_func(
                    FaceParser_mask,
                    (swap_mask.shape[-2], swap_mask.shape[-1]),
                    is_mask=True,
                )
            swap_mask.mul_(FaceParser_mask)

        if parameters.get("ClipEnableToggle", False):
            mask_clip = self.worker.function_worker.run_CLIPs(
                original_face_512,
                parameters["ClipText"],
                parameters["ClipAmountSlider"],
            )
            if mask_clip.shape[-1] != swap_mask.shape[-1]:
                mask_clip = _resize_func(
                    mask_clip, (swap_mask.shape[-2], swap_mask.shape[-1]), is_mask=True
                )
            swap_mask.mul_(mask_clip)
            if swap_mask_noFP.shape[-1] != mask_clip.shape[-1]:
                swap_mask_noFP = _resize_func(
                    swap_mask_noFP,
                    (mask_clip.shape[-2], mask_clip.shape[-1]),
                    is_mask=True,
                )
            swap_mask_noFP.mul_(mask_clip)

        if parameters.get("RestoreMouthEnableToggle", False) or parameters.get(
            "RestoreEyesEnableToggle", False
        ):
            M = cast(np.ndarray, tform.params)[0:2]
            ones_column = np.ones((kps_5.shape[0], 1), dtype=np.float32)
            dst_kps_5 = np.hstack([kps_5, ones_column]) @ M.T

            img_swap_mask = torch.ones(
                (1, 512, 512),
                dtype=torch.float32,
                device=self.worker.models_processor.device,
            )
            img_orig_mask = torch.zeros(
                (1, 512, 512),
                dtype=torch.float32,
                device=self.worker.models_processor.device,
            )

            if parameters.get("RestoreMouthEnableToggle", False):
                img_swap_mask = self.worker.function_worker.restore_mouth(
                    img_orig_mask,
                    img_swap_mask,
                    dst_kps_5,
                    parameters["RestoreMouthBlendAmountSlider"] / 100.0,
                    parameters["RestoreMouthFeatherBlendSlider"],
                    parameters["RestoreMouthSizeFactorSlider"] / 100.0,
                    parameters["RestoreXMouthRadiusFactorDecimalSlider"],
                    parameters["RestoreYMouthRadiusFactorDecimalSlider"],
                    parameters["RestoreXMouthOffsetSlider"],
                    parameters["RestoreYMouthOffsetSlider"],
                ).clamp(0, 1)

            if parameters.get("RestoreEyesEnableToggle", False):
                img_swap_mask = self.worker.function_worker.restore_eyes(
                    img_orig_mask,
                    img_swap_mask,
                    dst_kps_5,
                    parameters["RestoreEyesBlendAmountSlider"] / 100.0,
                    parameters["RestoreEyesFeatherBlendSlider"],
                    parameters["RestoreEyesSizeFactorDecimalSlider"],
                    parameters["RestoreXEyesRadiusFactorDecimalSlider"],
                    parameters["RestoreYEyesRadiusFactorDecimalSlider"],
                    parameters["RestoreXEyesOffsetSlider"],
                    parameters["RestoreYEyesOffsetSlider"],
                    parameters["RestoreEyesSpacingOffsetSlider"],
                ).clamp(0, 1)

            if parameters.get("RestoreEyesMouthBlurSlider", 0) > 0:
                b = parameters["RestoreEyesMouthBlurSlider"]
                k_size = int(b * 2 + 1)
                s_val = float((b + 1) * 0.2)
                img_swap_mask = v2.functional.gaussian_blur(
                    img_swap_mask, kernel_size=[k_size, k_size], sigma=[s_val, s_val]
                )

            if img_swap_mask.shape[-1] != swap_mask.shape[-1]:
                mask_resized = _resize_func(
                    img_swap_mask,
                    (swap_mask.shape[-2], swap_mask.shape[-1]),
                    is_mask=True,
                )
            else:
                mask_resized = img_swap_mask
            swap_mask = swap_mask * mask_resized

        t256_near = self.worker.t256_near
        assert t256_near is not None, (
            "t256_near must be initialized via set_scaling_transforms"
        )

        if parameters.get("DFLXSegEnableToggle", False):
            img_xseg_256 = t256_near(original_face_512)
            mouth_256 = None
            inner_mouth_protection_256 = None
            if (
                parameters.get("XSegMouthEnableToggle", False)
                and parameters.get("DFLXSegSizeSlider", 0)
                != parameters.get("DFLXSeg2SizeSlider", 0)
                and mouth_512 is not None
            ):
                mouth_256 = t256_near(mouth_512.unsqueeze(0))
            if (
                parameters.get("XSegExcludeInnerMouthToggle", False)
                and inner_mouth_protection_512 is not None
            ):
                inner_mouth_protection_256 = t256_near(
                    inner_mouth_protection_512.unsqueeze(0)
                ).squeeze(0)

            img_mask_256, mask_forcalc_256, mask_forcalc_dill_256, outpred_noFP_256 = (
                self.worker.function_worker.apply_dfl_xseg(
                    img_xseg_256,
                    -parameters["DFLXSegSizeSlider"],
                    mouth_256 if mouth_256 is not None else 0,
                    parameters,
                    inner_mouth_mask=inner_mouth_protection_256,
                    yaw_deg=guard_yaw_deg,
                )
            )

            if img_mask_256.shape[-1] != swap_mask.shape[-1]:
                img_mask_res = _resize_func(
                    img_mask_256,
                    (swap_mask.shape[-2], swap_mask.shape[-1]),
                    is_mask=True,
                )
                outpred_noFP_res = _resize_func(
                    outpred_noFP_256,
                    (swap_mask.shape[-2], swap_mask.shape[-1]),
                    is_mask=True,
                )
            else:
                img_mask_res = img_mask_256
                outpred_noFP_res = outpred_noFP_256

            if swap_mask_noFP.shape[-1] != outpred_noFP_res.shape[-1]:
                swap_mask_noFP = _resize_func(
                    swap_mask_noFP,
                    (outpred_noFP_res.shape[-2], outpred_noFP_res.shape[-1]),
                    is_mask=True,
                )

            if not parameters.get("OccluderEnableToggle", False):
                blur_amount = parameters.get("OccluderXSegBlurSlider", 0)
                if blur_amount > 0:
                    k_size = int(blur_amount * 2 + 1)
                    s_val = float((blur_amount + 1) * 0.2)
                    swap_mask = v2.functional.gaussian_blur(
                        swap_mask, kernel_size=[k_size, k_size], sigma=[s_val, s_val]
                    )
                    swap_mask_noFP = v2.functional.gaussian_blur(
                        swap_mask_noFP,
                        kernel_size=[k_size, k_size],
                        sigma=[s_val, s_val],
                    )

            swap_mask_noFP.mul_(1.0 - outpred_noFP_res)
            swap_mask.mul_(1.0 - img_mask_res)

            mask_forcalc_512 = t512_mask(mask_forcalc_256)
            mask_forcalc_dill_512 = t512_mask(mask_forcalc_dill_256)

            xseg_core = 1.0 - mask_forcalc_512
            xseg_core_dill = 1.0 - mask_forcalc_dill_512

            calc_mask = core_stats_mask * xseg_core
            calc_mask_dill = core_stats_mask * xseg_core_dill
            mask_forcalc_512 = calc_mask.clone()

        else:
            calc_mask = core_stats_mask.clone()
            calc_mask_dill = core_stats_mask.clone()
            mask_forcalc_512 = core_stats_mask.clone()

        mask_autocolor = calc_mask.clone()
        mask_autocolor = (mask_autocolor > 0.5).float()
        if (
            parameters["FaceRestorerEnableToggle"]
            and parameters["FaceRestorerAutoEnableToggle"]
        ):
            assert swap_original is not None, (
                "swap_original must be set when FaceRestorerEnableToggle is active"
            )
            alpha_restorer = float(parameters["FaceRestorerBlendSlider"]) / 100.0
            adjust_sharpness = float(parameters["FaceRestorerAutoSharpAdjustSlider"])
            scale_factor = round(tform.scale, 2)
            automasktoggle = parameters["FaceRestorerAutoMaskEnableToggle"]
            automaskadjust = parameters["FaceRestorerAutoSharpMaskAdjustDecimalSlider"]
            automaskblur = 2
            restore_mask = mask_forcalc_512

            alpha_auto, blur_value = self.face_restorer_auto(
                original_face_512,
                swap_original,
                swap_restorecalc,
                alpha_restorer,
                adjust_sharpness,
                scale_factor,
                debug,
                restore_mask,
                automasktoggle,
                automaskadjust,
                automaskblur,
            )

            if blur_value > 0:
                k_size = int(2 * blur_value + 1)
                s_val = float(blur_value * 0.1)
                swap = v2.functional.gaussian_blur(
                    swap_original, kernel_size=[k_size, k_size], sigma=[s_val, s_val]
                )
                debug_info["Restore1"] = f": {-blur_value:.2f}"
            elif isinstance(alpha_auto, torch.Tensor):
                swap = swap_restorecalc * alpha_auto + swap_original * (1 - alpha_auto)
            elif alpha_auto != 0:
                swap = swap_restorecalc * alpha_auto + swap_original * (1 - alpha_auto)
                if debug:
                    debug_info["Restore1"] = f": {alpha_auto * 100:.2f}"
            else:
                swap = swap_original
                if debug:
                    debug_info["Restore1"] = f": {alpha_auto * 100:.2f}"
        elif parameters["FaceRestorerEnableToggle"]:
            assert swap_original is not None, (
                "swap_original must be set when FaceRestorerEnableToggle is active"
            )
            alpha_restorer = float(parameters["FaceRestorerBlendSlider"]) / 100.0
            swap = (
                torch.lerp(
                    swap_original.float(), swap_restorecalc.float(), alpha_restorer
                )
                .to(swap_restorecalc.dtype)
                .contiguous()
            )

        if (
            parameters["FaceExpressionEnableBothToggle"]
            and (
                parameters["FaceExpressionLipsToggle"]
                or parameters["FaceExpressionEyesToggle"]
                or parameters["FaceExpressionBrowsToggle"]
                or parameters["FaceExpressionGeneralToggle"]
                or parameters.get("FaceExpressionModeSelection", "Advanced")
                in ("Simple", "Recast")
            )
            and parameters["FaceExpressionBeforeTypeSelection"]
            == "After First Restorer"
        ):
            if parameters.get("FaceExpressionModeSelection", "Advanced") == "Recast":
                swap = self.worker.function_worker.apply_perform_recast(
                    original_face_512,
                    swap,
                    cast(dict, parameters),
                    cast(dict, control),
                    driving_kps=kps_all_crop,
                )
            else:
                swap = self.worker.function_worker.apply_face_expression_restorer(
                    original_face_512,
                    swap,
                    cast(dict, parameters),
                    cast(dict, control),
                    driving_kps=kps_all_crop,
                )

        if (
            parameters["FaceEditorEnableToggle"]
            and self.worker.local_control_state_from_feeder.get("edit_enabled", True)
            and parameters["FaceEditorBeforeTypeSelection"] == "After First Restorer"
        ):
            editor_mask = swap_mask.clone()
            swap = (
                torch.lerp(original_face_512.float(), swap.float(), editor_mask)
                .to(swap.dtype)
                .contiguous()
            )
            swap = self.worker.function_worker.swap_edit_face_core(
                swap, swap_restorecalc, parameters, control
            )
            if swap_mask_noFP.shape[-1] != swap.shape[-1]:
                swap_mask = _resize_func(
                    swap_mask_noFP, (swap.shape[-2], swap.shape[-1]), is_mask=True
                )
            else:
                swap_mask = swap_mask_noFP

        if (
            parameters.get("FaceShapingEnableToggle", False)
            and not parameters.get("FaceShapingPreSwapToggle", False)
            and self.worker.local_control_state_from_feeder.get("edit_enabled", True)
            and parameters.get("FaceEditorBeforeTypeSelection", "Beginning")
            == "After First Restorer"
        ):
            swap = self.worker.function_worker.apply_face_shaping_gpu(
                swap, kps_ref, parameters
            )

        if control.get("DenoiserAfterFirstRestorerToggle", False):
            swap = self._apply_denoiser_pass(
                swap,
                control,
                "AfterFirst",
                kv_map,
                color_mask=mask_forcalc_512,
                blend_mask=swap_mask,
            )

        if (
            parameters["FaceRestorerEnable2Toggle"]
            and not parameters["FaceRestorerEnable2EndToggle"]
        ):
            swap_original2 = swap.clone()
            swap2 = self.worker.function_worker.apply_facerestorer(
                swap,
                parameters["FaceRestorerDetType2Selection"],
                parameters["FaceRestorerType2Selection"],
                parameters["FaceRestorerBlend2Slider"],
                parameters["FaceFidelityWeight2DecimalSlider"],
                control["DetectorScoreSlider"],
                kps_ref,
                slot_id=2,
                osdface_timestep=parameters.get("OSDFaceTimestep2Slider", 399),
                osdface_latent_strength=parameters.get(
                    "OSDFaceLatentStrength2DecimalSlider", 1.0
                ),
            )
            swap = self._apply_restorer_with_auto(
                swap,
                swap2,
                swap_original2,
                original_face_512,
                mask_forcalc_512,
                parameters,
                tform.scale,
                debug,
                debug_info,
                slot_id=2,
            )

        if (
            parameters["FaceExpressionEnableBothToggle"]
            and (
                parameters["FaceExpressionLipsToggle"]
                or parameters["FaceExpressionEyesToggle"]
                or parameters["FaceExpressionBrowsToggle"]
                or parameters["FaceExpressionGeneralToggle"]
                or parameters.get("FaceExpressionModeSelection", "Advanced")
                in ("Simple", "Recast")
            )
            and parameters["FaceExpressionBeforeTypeSelection"]
            == "After Second Restorer"
        ):
            if parameters.get("FaceExpressionModeSelection", "Advanced") == "Recast":
                swap = self.worker.function_worker.apply_perform_recast(
                    original_face_512,
                    swap,
                    cast(dict, parameters),
                    cast(dict, control),
                    driving_kps=kps_all_crop,
                )
            else:
                swap = self.worker.function_worker.apply_face_expression_restorer(
                    original_face_512,
                    swap,
                    cast(dict, parameters),
                    cast(dict, control),
                    driving_kps=kps_all_crop,
                )

        if (
            parameters["FaceEditorEnableToggle"]
            and self.worker.local_control_state_from_feeder.get("edit_enabled", True)
            and parameters["FaceEditorBeforeTypeSelection"] == "After Second Restorer"
        ):
            editor_mask = t512_mask(swap_mask).clone()
            swap = (
                torch.lerp(original_face_512.float(), swap.float(), editor_mask)
                .to(swap.dtype)
                .contiguous()
            )
            swap = self.worker.function_worker.swap_edit_face_core(
                swap, swap, parameters, control
            )
            if swap_mask_noFP.shape[-1] != swap.shape[-1]:
                swap_mask = _resize_func(
                    swap_mask_noFP, (swap.shape[-2], swap.shape[-1]), is_mask=True
                )
            else:
                swap_mask = swap_mask_noFP

        if (
            parameters.get("FaceShapingEnableToggle", False)
            and not parameters.get("FaceShapingPreSwapToggle", False)
            and self.worker.local_control_state_from_feeder.get("edit_enabled", True)
            and parameters.get("FaceEditorBeforeTypeSelection", "Beginning")
            == "After Second Restorer"
        ):
            swap = self.worker.function_worker.apply_face_shaping_gpu(
                swap, kps_ref, parameters
            )

        original_face_for_color = original_face_512
        if parameters.get("AutoColorEnableToggle", False) and valid_t_e is not None:
            _ema_key = valid_t_e.tobytes()
            _face_f = original_face_512.float()
            _curr_mean = _face_f.mean(dim=(1, 2), keepdim=True)
            _curr_std = _face_f.std(dim=(1, 2), keepdim=True) + 1e-6

            with _COLOR_EMA_LOCK:
                if self.worker.frame_number == 0 or self.worker.is_single_frame:
                    _COLOR_STATS_EMA.clear()
                _prev = _COLOR_STATS_EMA.get(_ema_key)
                if _prev is not None:
                    _COLOR_STATS_EMA.move_to_end(_ema_key)
                    _ema_mean = _COLOR_EMA_ALPHA * _curr_mean + (1.0 - _COLOR_EMA_ALPHA) * _prev["mean"]
                    _ema_std = _COLOR_EMA_ALPHA * _curr_std + (1.0 - _COLOR_EMA_ALPHA) * _prev["std"]
                else:
                    _ema_mean, _ema_std = _curr_mean, _curr_std
                    if len(_COLOR_STATS_EMA) >= _COLOR_STATS_EMA_MAX:
                        _COLOR_STATS_EMA.popitem(last=False)
                _COLOR_STATS_EMA[_ema_key] = {
                    "mean": _ema_mean.detach(),
                    "std": _ema_std.detach(),
                }

            original_face_for_color = (
                ((_face_f - _curr_mean) / _curr_std * _ema_std + _ema_mean)
                .clamp(0, 255)
                .to(original_face_512.dtype)
            )

        if parameters.get("AutoColorEnableToggle", False):
            if parameters["AutoColorTransferTypeSelection"] == "CDF Histogram":
                swap = faceutil.histogram_matching(
                    original_face_for_color,
                    swap,
                    parameters["AutoColorBlendAmountSlider"],
                )
            elif (
                parameters["AutoColorTransferTypeSelection"] == "CDF Histogram (Masked)"
            ):
                swap = faceutil.histogram_matching_withmask(
                    original_face_for_color,
                    swap,
                    mask_autocolor,
                    parameters["AutoColorBlendAmountSlider"],
                )
            elif parameters["AutoColorTransferTypeSelection"] == "Reinhard Transfer":
                swap = faceutil.apply_reinhard_color_transfer(
                    original_face_for_color,
                    swap,
                    parameters["AutoColorBlendAmountSlider"],
                )
            elif (
                parameters["AutoColorTransferTypeSelection"]
                == "Reinhard Transfer (Masked)"
            ):
                swap = faceutil.apply_reinhard_color_transfer(
                    original_face_for_color,
                    swap,
                    parameters["AutoColorBlendAmountSlider"],
                    mask_autocolor,
                )
            elif parameters["AutoColorTransferTypeSelection"] == "AdaIN (Core Masked)":
                swap = faceutil.apply_adain_color_transfer(
                    swap,
                    original_face_for_color,
                    swap_mask,
                    parameters["AutoColorBlendAmountSlider"],
                    calc_mask=mask_autocolor,
                )

        if parameters.get("TransferTextureEnableToggle", False):
            if swap.shape[-1] != 512:
                swap = t512_mask(swap)
                swap_mask = t512_mask(swap_mask)
                swap_mask_noFP = t512_mask(swap_mask_noFP)

            mask_input_vgg = t128_mask(calc_mask.clone())
            mask_vgg_512 = torch.ones(
                (1, 512, 512),
                dtype=torch.float32,
                device=self.worker.models_processor.device,
            )
            upper_thresh = parameters["TextureUpperLimitSlider"] / 100.0

            if parameters.get("ExcludeOriginalVGGMaskEnableToggle", False):
                thr = (
                    parameters["VGGMaskThresholdSlider"]
                    if parameters.get("ExcludeVGGMaskEnableToggle", False)
                    else 0
                )
                mask_vgg_raw, diff_norm_texture_raw = (
                    self.worker.function_worker.apply_vgg_mask_simple(
                        swap,
                        original_face_512,
                        mask_input_vgg,
                        center_pct=thr,
                        softness_pct=100,
                        feature_layer="combo_relu3_3_relu3_1",
                        mode="smooth",
                    )
                )
                mask_vgg_512 = t512_mask(mask_vgg_raw).clamp(0.0, 1.0)
                diff_norm_texture_512 = t512_mask(diff_norm_texture_raw).clamp(0.0, 1.0)

                if not parameters.get("ExcludeVGGMaskEnableToggle", False):
                    mask_vgg_512 = diff_norm_texture_512.clone()

                if parameters.get("TextureBlendAmountSlider", 0) > 0:
                    b = parameters["TextureBlendAmountSlider"]
                    k_size = int(b * 2 + 1)
                    s_val = float((b + 1) * 0.2)
                    mask_vgg_512 = v2.functional.gaussian_blur(
                        mask_vgg_512.float(),
                        kernel_size=[k_size, k_size],
                        sigma=[s_val, s_val],
                    )

            if parameters.get("ExcludeMaskEnableToggle", False):
                feature_mask = texture_exclude_512.clone().float()

                if parameters.get("ExcludeOriginalVGGMaskEnableToggle", False):
                    blur_val = parameters.get("FaceParserBlurTextureSlider", 0)
                    if blur_val > 0:
                        k_size = int(blur_val * 2 + 1)
                        s_val = float(max((blur_val + 1) * 0.2, 1e-6))
                        feature_mask = v2.functional.gaussian_blur(
                            feature_mask,
                            kernel_size=[k_size, k_size],
                            sigma=[s_val, s_val],
                        )

                if parameters.get("ExcludeOriginalVGGMaskEnableToggle", False):
                    mask_vgg_512 = torch.where(
                        mask_vgg_512 >= upper_thresh, upper_thresh, mask_vgg_512
                    )

                mask_final_512 = (
                    torch.max(mask_vgg_512 * (1.0 - feature_mask), 1.0 - calc_mask_dill)
                ).clamp(0.0, 1.0)
            elif parameters.get("ExcludeOriginalVGGMaskEnableToggle", False):
                mask_vgg_512 = torch.where(
                    mask_vgg_512 >= upper_thresh, upper_thresh, mask_vgg_512
                )
                mask_final_512 = torch.max(mask_vgg_512, 1.0 - calc_mask_dill).clamp(
                    0.0, 1.0
                )
            else:
                mask_final_512 = (1.0 - mask_forcalc_512).clamp(0.0, 1.0)

            if parameters.get("AutoColorEnableToggle", False):
                swap_texture_backup = swap.clone()
            else:
                swap_texture_backup = faceutil.apply_reinhard_color_transfer(
                    original_face_512, swap.clone(), 100, mask_autocolor
                )

            TransferTextureKernelSizeSlider = 12
            TransferTextureSigmaDecimalSlider = 4.00
            TransferTextureWeightSlider = 1
            TransferTexturePhiDecimalSlider = 9.7
            TransferTextureGammaDecimalSlider = 0.5

            if parameters.get("TransferTextureModeEnableToggle", False):
                TransferTextureLambdSlider = 8
                TransferTextureThetaSlider = 8
            else:
                TransferTextureLambdSlider = 2
                TransferTextureThetaSlider = 1

            clip_limit = (
                parameters["TransferTextureClipLimitDecimalSlider"]
                if parameters.get("TransferTextureClaheEnableToggle", False)
                else 0.0
            )
            alpha_clahe = parameters["TransferTextureAlphaClaheDecimalSlider"]
            grid_size = (4, 4)
            global_gamma = parameters["TransferTexturePreGammaDecimalSlider"]
            global_contrast = parameters["TransferTexturePreContrastDecimalSlider"]

            gradient_texture = self.gradient_magnitude(
                original_face_512,
                calc_mask_dill,
                TransferTextureKernelSizeSlider,
                TransferTextureWeightSlider,
                TransferTextureSigmaDecimalSlider,
                TransferTextureLambdSlider,
                TransferTextureGammaDecimalSlider,
                TransferTexturePhiDecimalSlider,
                TransferTextureThetaSlider,
                clip_limit,
                alpha_clahe,
                grid_size,
                global_gamma,
                global_contrast,
            )

            gradient_texture = faceutil.apply_reinhard_color_transfer(
                original_face_512, gradient_texture, 100, mask_autocolor
            )

            if parameters["FaceParserBlurTextureSlider"] > 0:
                orig = mask_final_512.clone()
                b_val = parameters["FaceParserBlurTextureSlider"]
                k_size = int(b_val * 2 + 1)
                s_val = float((b_val + 1) * 0.2)
                mask_final_512 = v2.functional.gaussian_blur(
                    mask_final_512.to(torch.float32),
                    kernel_size=[k_size, k_size],
                    sigma=[s_val, s_val],
                )
                mask_final_512 = torch.max(mask_final_512, orig).clamp(0.0, 1.0)

            alpha_t = parameters["TransferTextureBlendAmountSlider"] / 100.0
            w = alpha_t * (1.0 - mask_final_512)
            w = w.clamp(0.0, 1.0)

            swap = (swap_texture_backup * (1.0 - w) + gradient_texture * w).clamp(
                0, 255
            )
            texture_mask_view = (1.0 - mask_final_512).clone()

        if parameters.get("DifferencingEnableToggle", False):
            if swap.shape[-1] != 512:
                swap = t512_mask(swap)
                swap_mask = t512_mask(swap_mask)
                swap_mask_noFP = t512_mask(swap_mask_noFP)

            diff_mask_128 = t128_mask(calc_mask.clone())
            swapped_face_resized = swap.clone()
            original_face_resized = original_face_512.clone()
            FeatureLayerTypeSelection = "combo_relu3_3_relu3_1"

            lower_thresh = parameters["DifferencingLowerLimitThreshSlider"] / 100.0
            upper_thresh = parameters["DifferencingUpperLimitThreshSlider"] / 100.0
            middle_value = parameters["DifferencingMiddleLimitValueSlider"] / 100.0
            upper_value = parameters["DifferencingUpperLimitValueSlider"] / 100.0

            mask_diff_128, diff_norm_texture = (
                self.worker.function_worker.apply_perceptual_diff_onnx(
                    swapped_face_resized,
                    original_face_resized,
                    diff_mask_128,
                    lower_thresh,
                    0,
                    upper_thresh,
                    upper_value,
                    middle_value,
                    FeatureLayerTypeSelection,
                    False,
                )
            )

            eps = 1e-6
            inv_lower = 1.0 / max(lower_thresh, eps)
            inv_mid = 1.0 / max((upper_thresh - lower_thresh), eps)
            inv_high = 1.0 / max((1.0 - upper_thresh), eps)

            res_low = diff_norm_texture * inv_lower * middle_value
            res_mid = middle_value + (diff_norm_texture - lower_thresh) * inv_mid * (
                upper_value - middle_value
            )
            res_high = upper_value + (diff_norm_texture - upper_thresh) * inv_high * (
                1.0 - upper_value
            )

            piece = torch.where(
                diff_norm_texture < lower_thresh,
                res_low,
                torch.where(diff_norm_texture > upper_thresh, res_high, res_mid),
            )
            mask512 = t512_mask(piece)

            if parameters.get("DifferencingBlendAmountSlider", 0) > 0:
                b = parameters["DifferencingBlendAmountSlider"]
                k_size = int(b * 2 + 1)
                s_val = float((b + 1) * 0.2)
                mask512 = v2.functional.gaussian_blur(
                    mask512.float(), kernel_size=[k_size, k_size], sigma=[s_val, s_val]
                )

            mask512 = torch.max((mask512), 1 - calc_mask_dill).clamp_(0, 1)
            swap = (
                torch.lerp(original_face_512.float(), swap.float(), mask512)
                .clamp_(0, 255)
                .to(swap.dtype)
                .contiguous()
            )
            diff_mask = 1 - mask512.clone()

        if (
            parameters["FaceEditorEnableToggle"]
            and self.worker.local_control_state_from_feeder.get("edit_enabled", True)
            and parameters["FaceEditorBeforeTypeSelection"] == "After Texture Transfer"
        ):
            editor_mask = t512_mask(swap_mask).clone()
            if swap.shape[-1] != 512:
                swap = t512_mask(swap)

            swap = (
                torch.lerp(original_face_512.float(), swap.float(), editor_mask)
                .to(swap.dtype)
                .contiguous()
            )
            swap = self.worker.function_worker.swap_edit_face_core(
                swap, swap, parameters, control
            )
            if swap_mask_noFP.shape[-1] != swap.shape[-1]:
                swap_mask = _resize_func(
                    swap_mask_noFP, (swap.shape[-2], swap.shape[-1]), is_mask=True
                )
            else:
                swap_mask = swap_mask_noFP

        if (
            parameters.get("FaceShapingEnableToggle", False)
            and not parameters.get("FaceShapingPreSwapToggle", False)
            and self.worker.local_control_state_from_feeder.get("edit_enabled", True)
            and parameters.get("FaceEditorBeforeTypeSelection", "Beginning")
            == "After Texture Transfer"
        ):
            swap = self.worker.function_worker.apply_face_shaping_gpu(
                swap, kps_ref, parameters
            )
        if parameters["ColorEnableToggle"]:
            swap_pre_color = swap.clone()

            swap_adj = torch.unsqueeze(swap, 0).contiguous()
            swap_adj = v2.functional.adjust_gamma(
                swap_adj, parameters["ColorGammaDecimalSlider"], 1.0
            )
            swap_adj = torch.squeeze(swap_adj).permute(1, 2, 0).type(torch.float32)

            del_color = torch.tensor(
                [
                    parameters["ColorRedSlider"],
                    parameters["ColorGreenSlider"],
                    parameters["ColorBlueSlider"],
                ],
                device=self.worker.models_processor.device,
            )
            swap_adj += del_color
            swap_adj = (
                torch.clamp(swap_adj, min=0.0, max=255.0).permute(2, 0, 1) / 255.0
            )

            swap_adj = v2.functional.adjust_brightness(
                swap_adj, parameters["ColorBrightnessDecimalSlider"]
            )
            swap_adj = v2.functional.adjust_contrast(
                swap_adj, parameters["ColorContrastDecimalSlider"]
            )
            swap_adj = v2.functional.adjust_saturation(
                swap_adj, parameters["ColorSaturationDecimalSlider"]
            )
            swap_adj = v2.functional.adjust_sharpness(
                swap_adj, parameters["ColorSharpnessDecimalSlider"]
            )
            swap_adj = v2.functional.adjust_hue(
                swap_adj, parameters["ColorHueDecimalSlider"]
            )

            swap_adj = swap_adj * 255.0

            color_bounds_mask = v2.functional.gaussian_blur(
                (swap_pre_color.sum(dim=0, keepdim=True) > 0.05).float(),
                kernel_size=15,
                sigma=5.0,
            )
            swap = (
                torch.lerp(swap_pre_color.float(), swap_adj.float(), color_bounds_mask)
                .to(swap.dtype)
                .contiguous()
            )

        if (
            parameters["FaceRestorerEnable2Toggle"]
            and parameters["FaceRestorerEnable2EndToggle"]
        ):
            swap_original2 = swap.clone()
            swap2 = self.worker.function_worker.apply_facerestorer(
                swap,
                parameters["FaceRestorerDetType2Selection"],
                parameters["FaceRestorerType2Selection"],
                parameters["FaceRestorerBlend2Slider"],
                parameters["FaceFidelityWeight2DecimalSlider"],
                control["DetectorScoreSlider"],
                kps_ref,
                slot_id=2,
                osdface_timestep=parameters.get("OSDFaceTimestep2Slider", 399),
                osdface_latent_strength=parameters.get(
                    "OSDFaceLatentStrength2DecimalSlider", 1.0
                ),
            )
            swap = self._apply_restorer_with_auto(
                swap,
                swap2,
                swap_original2,
                original_face_512,
                mask_forcalc_512,
                parameters,
                tform.scale,
                debug,
                debug_info,
                slot_id=2,
            )

        if parameters.get("MouthParserStretchAfterToggle", False):
            mouth_overlay_pkg = None
            if hasattr(self.worker.function_worker, "get_mouth_overlay"):
                mouth_overlay_pkg = self.worker.function_worker.get_mouth_overlay(
                    swap, original_face_512, parameters
                )
            if mouth_overlay_pkg is not None:
                overlay_rgb, overlay_mask = mouth_overlay_pkg
                if overlay_rgb is not None and overlay_mask is not None:
                    if overlay_rgb.shape[-1] != swap.shape[-1]:
                        overlay_rgb = _resize_func(
                            overlay_rgb, (swap.shape[-2], swap.shape[-1]), is_mask=False
                        )
                        overlay_mask = _resize_func(
                            overlay_mask.unsqueeze(0),
                            (swap.shape[-2], swap.shape[-1]),
                            is_mask=True,
                        ).squeeze(0)
                    swap = swap * (1.0 - overlay_mask) + overlay_rgb * overlay_mask

        if parameters.get("FaceParserEnableToggle") and parameters.get(
            "FaceParserEndToggle"
        ):
            out = self.worker.function_worker.process_masks_and_masks(
                swap, original_face_512, parameters, control
            )
            FaceParser_mask = out.get("FaceParser_mask", None)
            mouth_debug_512 = out.get("mouth_debug", mouth_debug_512)
            mouth_debug_teeth_512 = out.get("mouth_debug_teeth", mouth_debug_teeth_512)
            if FaceParser_mask is not None:
                if FaceParser_mask.shape[-1] != swap_mask.shape[-1]:
                    FaceParser_mask = _resize_func(
                        FaceParser_mask, (swap.shape[-2], swap.shape[-1]), is_mask=True
                    )
                swap_mask.mul_(FaceParser_mask)

        mask_autocolor_end = mask_forcalc_512.clone()
        if (
            parameters.get("FaceParserEnableToggle")
            and parameters.get("FaceParserEndToggle")
            and FaceParser_mask is not None
        ):
            fp_core = (FaceParser_mask > 0.5).float()
            if fp_core.shape[-1] != mask_autocolor_end.shape[-1]:
                fp_core = _resize_func(
                    fp_core,
                    (mask_autocolor_end.shape[-2], mask_autocolor_end.shape[-1]),
                    is_mask=True,
                )
            mask_autocolor_end = mask_autocolor_end * fp_core
        mask_autocolor_end = (mask_autocolor_end > 0.5).float()

        if parameters.get("EndingColorTransferEnableToggle", False):
            if parameters["EndingColorTransferTypeSelection"] == "CDF Histogram":
                swap = faceutil.histogram_matching(
                    original_face_512, swap, parameters["EndingColorBlendAmountSlider"]
                )
            elif (
                parameters["EndingColorTransferTypeSelection"]
                == "CDF Histogram (Masked)"
            ):
                swap = faceutil.histogram_matching_withmask(
                    original_face_512,
                    swap,
                    mask_autocolor_end,
                    parameters["EndingColorBlendAmountSlider"],
                )
            elif parameters["EndingColorTransferTypeSelection"] == "Reinhard Transfer":
                swap = faceutil.apply_reinhard_color_transfer(
                    original_face_512, swap, parameters["EndingColorBlendAmountSlider"]
                )
            elif (
                parameters["EndingColorTransferTypeSelection"]
                == "Reinhard Transfer (Masked)"
            ):
                swap = faceutil.apply_reinhard_color_transfer(
                    original_face_512,
                    swap,
                    parameters["EndingColorBlendAmountSlider"],
                    mask_autocolor_end,
                )
            elif (
                parameters["EndingColorTransferTypeSelection"] == "AdaIN (Core Masked)"
            ):
                swap = faceutil.apply_adain_color_transfer(
                    swap,
                    original_face_512,
                    swap_mask,
                    parameters["EndingColorBlendAmountSlider"],
                    calc_mask=mask_autocolor_end,
                )

        if control.get("DenoiserAfterRestorersToggle", False):
            swap = self._apply_denoiser_pass(
                swap,
                control,
                "After",
                kv_map,
                color_mask=mask_autocolor_end,
                blend_mask=swap_mask,
            )

        if (
            parameters.get("FinalBlendAdjEnableToggle", False)
            and parameters.get("FinalBlendAmountSlider", 0) > 0
        ):
            final_blur_strength = parameters["FinalBlendAmountSlider"]
            k_size = int(2 * final_blur_strength + 1)
            s_val = float(max(final_blur_strength * 0.1, 1e-6))
            swap = v2.functional.gaussian_blur(
                swap, kernel_size=[k_size, k_size], sigma=[s_val, s_val]
            )

        if (
            parameters.get("FinalBorderBlendEnableToggle", False)
            and parameters.get("FinalBorderBlendAmountSlider", 0) > 0
        ):
            border_blur_strength = int(parameters["FinalBorderBlendAmountSlider"])

            kernel_size = int(2 * border_blur_strength + 1)
            padding = border_blur_strength
            sigma = float(max(border_blur_strength * 0.2, 1e-6))

            blurred_swap = v2.functional.gaussian_blur(
                swap, kernel_size=[kernel_size, kernel_size], sigma=[sigma, sigma]
            )

            dilation = F.max_pool2d(
                swap_mask, kernel_size=kernel_size, stride=1, padding=padding
            )
            erosion = -F.max_pool2d(
                -swap_mask, kernel_size=kernel_size, stride=1, padding=padding
            )

            raw_border_band = dilation - erosion

            smooth_border_mask = v2.functional.gaussian_blur(
                raw_border_band,
                kernel_size=[kernel_size, kernel_size],
                sigma=[sigma, sigma],
            ).clamp_(0.0, 1.0)

            swap = (
                torch.lerp(swap.float(), blurred_swap.float(), smooth_border_mask)
                .to(swap.dtype)
                .contiguous()
            )

        if parameters["JPEGCompressionEnableToggle"]:
            jpeg_q = int(parameters["JPEGCompressionAmountSlider"])
            if jpeg_q != 100:
                s = float(tform.scale)
                gamma = 0.60
                strength = 0.80
                q_min = 14
                q_max = 100

                jpeg_q_eff = faceutil._map_jpeg_quality(
                    base_q=jpeg_q,
                    face_scale=s,
                    gamma=gamma,
                    strength=strength,
                    q_min=q_min,
                    q_max=q_max,
                )
                if debug:
                    debug_info["JPEG Quality"] = f"{jpeg_q_eff}"

                swap_jpeg = faceutil.jpegBlur(swap, jpeg_q_eff)
                blend = parameters["JPEGCompressionBlendSlider"] / 100.0
                swap = (
                    torch.lerp(swap.float(), swap_jpeg.float(), blend)
                    .to(swap.dtype)
                    .contiguous()
                )

        if parameters["BlockShiftEnableToggle"]:
            base_quality = parameters["BlockShiftAmountSlider"]
            max_px = parameters["BlockShiftMaxAmountSlider"]

            swap_blockshift = self.apply_block_shift_gpu_jitter(
                swap,
                block_size=base_quality,
                max_amount_pixels=float(max_px),
                seed=1337,
            )

            block_shift_blend = parameters["BlockShiftBlendAmountSlider"] / 100.0
            swap = (
                torch.lerp(swap.float(), swap_blockshift.float(), block_shift_blend)
                .to(swap.dtype)
                .contiguous()
            )

        if parameters["ColorNoiseDecimalSlider"] > 0:
            swap = swap.to(torch.float32)
            noise = (
                (torch.rand_like(swap, dtype=torch.float32) - 0.5)
                * 2
                * parameters["ColorNoiseDecimalSlider"]
            )
            swap = torch.clamp(swap + noise, 0.0, 255.0)

        if is_perspective_crop:
            return t512_mask(swap), t512_mask(swap_mask), None

        b_val = parameters["OverallMaskBlendAmountSlider"]
        k_size = int(b_val * 2 + 1)
        s_val = float((b_val + 1) * 0.2)
        swap_mask = v2.functional.gaussian_blur(
            swap_mask, kernel_size=[k_size, k_size], sigma=[s_val, s_val]
        )

        if border_mask.shape[-1] != swap_mask.shape[-1]:
            border_mask = _resize_func(
                border_mask, (swap_mask.shape[-2], swap_mask.shape[-1]), is_mask=True
            )

        swap_mask = torch.mul(swap_mask, border_mask)

        if swap.shape[-1] != 512:
            swap = t512_mask(swap)
            swap_mask = t512_mask(swap_mask)

        swap = torch.mul(swap, swap_mask)

        original_face_512_clone = (
            original_face_512.clone().type(torch.uint8).permute(1, 2, 0)
            if self.worker.is_view_face_compare
            else None
        )

        swap_mask_clone = None
        if self.worker.is_view_face_mask:
            mask_show_type = parameters.get("MaskShowSelection", "swap_mask")
            if mask_show_type == "swap_mask":
                swap_mask_clone = (
                    torch.ones_like(swap_mask)
                    if (
                        parameters["FaceEditorEnableToggle"]
                        and self.worker.local_control_state_from_feeder.get(
                            "edit_enabled", True
                        )
                    )
                    else swap_mask.clone()
                )
            elif mask_show_type == "diff":
                swap_mask_clone = diff_mask.clone()
            elif mask_show_type == "texture":
                swap_mask_clone = texture_mask_view.clone()

            if swap_mask_clone is not None:
                if swap_mask_clone.shape[-1] != 512:
                    swap_mask_clone = t512_mask(swap_mask_clone)
                swap_mask_clone = torch.mul(
                    torch.cat((torch.sub(1, swap_mask_clone),) * 3, 0).permute(1, 2, 0),
                    255.0,
                ).type(torch.uint8)

        M_inv_np = cast(np.ndarray, tform_original.inverse.params)

        if alphaface_scale_val != 1.0:
            scale_matrix = np.array(
                [
                    [alphaface_scale_val, 0.0, 256.0 * (1.0 - alphaface_scale_val)],
                    [0.0, alphaface_scale_val, 256.0 * (1.0 - alphaface_scale_val)],
                    [0.0, 0.0, 1.0],
                ],
                dtype=np.float32,
            )
            M_inv_np = M_inv_np @ scale_matrix

        M_inv = (
            torch.from_numpy(M_inv_np[0:2])
            .float()
            .unsqueeze(0)
            .to(self.worker.models_processor.device)
        )
        dsize = (img.shape[1], img.shape[2])

        swap_full = kgm.warp_affine(
            swap.unsqueeze(0).float(),
            M_inv,
            dsize=dsize,
            mode="bilinear",
            padding_mode="zeros",
            align_corners=True,
        ).squeeze(0)
        swap_mask_full = kgm.warp_affine(
            swap_mask.unsqueeze(0).float(),
            M_inv,
            dsize=dsize,
            mode="bilinear",
            padding_mode="zeros",
            align_corners=True,
        ).squeeze(0)

        img = (
            (swap_full + (img.float() * (1.0 - swap_mask_full)))
            .clamp_(0, 255)
            .type(torch.uint8)
        )

        if parameters.get("AutoMouthShowDebugOutlineToggle", False):
            for _mask_512, _color in [
                (mouth_debug_512, [0, 255, 0]),
                (mouth_debug_teeth_512, [0, 255, 255]),
            ]:
                if _mask_512 is not None:
                    _mf = (
                        kgm.warp_affine(
                            _mask_512.unsqueeze(0).unsqueeze(0).float(),
                            M_inv,
                            dsize=dsize,
                            mode="bilinear",
                            padding_mode="zeros",
                            align_corners=True,
                        )
                        .squeeze(0)
                        .squeeze(0)
                    )
                    _bin = (_mf > 0.5).float().unsqueeze(0).unsqueeze(0)
                    _edge = (
                        F.max_pool2d(_bin, kernel_size=3, stride=1, padding=1)
                        - (-F.max_pool2d(-_bin, kernel_size=3, stride=1, padding=1))
                    ).squeeze(0).squeeze(0) > 0.0
                    img[0][_edge], img[1][_edge], img[2][_edge] = (
                        _color[0],
                        _color[1],
                        _color[2],
                    )

        return img, original_face_512_clone, swap_mask_clone

    @torch.no_grad()
    def gradient_magnitude(
        self,
        image: torch.Tensor,
        mask: torch.Tensor,
        kernel_size: int,
        weighting_strength: float,
        sigma: float,
        lambd: float,
        gamma: float,
        psi: float,
        theta_count: int,
        clip_limit: float,
        alpha_clahe: float,
        grid_size: tuple[int, int],
        global_gamma: float,
        global_contrast: float,
    ) -> torch.Tensor:
        C, H, W = image.shape
        image = image.float() / 255.0
        mask = mask.bool()

        if global_gamma != 1.0:
            image = image.pow(global_gamma)
        if global_contrast != 1.0:
            m_gc = image.mean((1, 2), keepdim=True)
            image = (image - m_gc) * global_contrast + m_gc

        if clip_limit > 0.0:
            image = image.unsqueeze(0).clamp(0, 1)
            mask_b3 = mask.unsqueeze(0)

            lab = kc.rgb_to_lab(image)
            L = lab[:, 0:1, :, :] / 100.0

            mb = mask_b3[:, 0:1, :, :]
            area_l = mb.sum((2, 3), keepdim=True).clamp(min=1)
            mean_l = (L * mb).sum((2, 3), keepdim=True) / area_l
            Lf = torch.where(mb, L, mean_l)
            Leq = ke.equalize_clahe(
                Lf,
                clip_limit=clip_limit,
                grid_size=grid_size,
                slow_and_differentiable=False,
            ).clamp(0, 1)
            L_blend = alpha_clahe * Leq + (1 - alpha_clahe) * L
            Lnew = torch.where(mb, L_blend, L)

            lab_eq = torch.cat([Lnew * 100.0, lab[:, 1:, :, :]], dim=1)
            x_eq = kc.lab_to_rgb(lab_eq)
            image = x_eq.squeeze(0)

        kernel_size = max(1, 2 * kernel_size - 1)
        if theta_count == 10:
            theta_values = torch.tensor([math.pi / 4], device=image.device)
        else:
            theta_values = torch.linspace(
                0, math.pi, theta_count + 1, device=image.device
            )[:-1]

        magnitude = self.apply_gabor_filter_torch(
            image, kernel_size, sigma, lambd, gamma, psi, theta_values
        )

        max_mv = magnitude.amax((1, 2), keepdim=True)
        inverted = max_mv - magnitude

        if weighting_strength > 0:
            img_m = image * mask
            weighted = inverted * (
                (1 - weighting_strength) + weighting_strength * img_m
            )
        else:
            weighted = inverted

        return weighted * 255

    def apply_gabor_filter_torch(
        self, image, kernel_size, sigma, lambd, gamma, psi, theta_values
    ):
        C, H, W = image.shape
        image = image.unsqueeze(0)

        N = theta_values.shape[0]

        kernels = self.get_gabor_kernels(
            kernel_size, sigma, lambd, gamma, psi, theta_values, image.device
        )

        expand_cache_key = (*kernels.shape, C)

        if expand_cache_key not in self._gabor_kernels_expanded_cache:
            MAX_GABOR_EXPANDED_CACHE = 16
            if len(self._gabor_kernels_expanded_cache) >= MAX_GABOR_EXPANDED_CACHE:
                self._gabor_kernels_expanded_cache.popitem(last=False)
            self._gabor_kernels_expanded_cache[expand_cache_key] = (
                kernels.repeat_interleave(C, dim=0)
            )

        weight = self._gabor_kernels_expanded_cache[expand_cache_key]
        out = F.conv2d(
            image,
            weight,
            padding=kernel_size // 2,
            groups=C,
        )

        out = out.squeeze(0).view(N, C, H, W)
        magnitudes = out.amax(dim=0)
        return magnitudes

    def get_gabor_kernels(
        self, kernel_size, sigma, lambd, gamma, psi, theta_values, device
    ):
        N = theta_values.shape[0]
        cache_key = (
            int(kernel_size),
            float(sigma),
            float(lambd),
            float(gamma),
            float(psi),
            int(N),
            str(device),
        )
        cached = self._gabor_kernels_cache.get(cache_key)
        if cached is not None:
            return cached

        half = kernel_size // 2
        y, x = torch.meshgrid(
            torch.linspace(-half, half, kernel_size, device=device),
            torch.linspace(-half, half, kernel_size, device=device),
            indexing="ij",
        )

        kernels = []
        for theta in theta_values:
            x_theta = x * torch.cos(theta) + y * torch.sin(theta)
            y_theta = -x * torch.sin(theta) + y * torch.cos(theta)

            gb = torch.exp(-0.5 * (x_theta**2 + (gamma**2) * y_theta**2) / sigma**2)
            gb *= torch.cos(2 * math.pi * x_theta / lambd + psi)
            kernels.append(gb)

        result = torch.stack(kernels).unsqueeze(1)
        MAX_GABOR_CACHE = 32
        if len(self._gabor_kernels_cache) >= MAX_GABOR_CACHE:
            self._gabor_kernels_cache.popitem(last=False)
        self._gabor_kernels_cache[cache_key] = result
        return result

    def face_restorer_auto(
        self,
        original_face_512,
        swap_original,
        swap,
        alpha,
        adjust_sharpness,
        scale_factor,
        debug,
        swap_mask,
        alpha_map_enable: bool = False,
        alpha_map_strength: float = 0.5,
        alpha_map_blur: int = 7,
    ):
        scores_original = self.sharpness_score(original_face_512)
        score_new_original = (
            scores_original["combined"].item() * 100 + adjust_sharpness / 10.0
        )

        alpha = 1.0
        max_iterations = 7
        alpha_min, alpha_max = 0.0, 1.0
        tolerance = 0.5
        min_alpha_change = 0.05
        iteration = 0
        prev_alpha = alpha
        iteration_blur = 0

        while iteration < max_iterations:
            swap2 = swap * alpha + swap_original * (1 - alpha)
            swap2_masked = swap2.clone()

            scores_swap = self.sharpness_score(swap2_masked)
            score_new_swap = scores_swap["combined"].item() * 100
            sharpness_diff = score_new_swap - score_new_original

            if abs(sharpness_diff) < tolerance:
                break

            if sharpness_diff < 0:
                if alpha > 0.99:
                    prev_alpha = alpha
                    break
                alpha_min = alpha
                alpha = (alpha + alpha_max) / 2.0
            else:
                alpha_max = alpha
                alpha = (alpha + alpha_min) / 2.0

            if sharpness_diff >= 0 and alpha < 0.07:
                prev_alpha = 0.0
                base = swap_original
                max_blur_strength = 10

                for bs in range(0, max_blur_strength + 1):
                    if bs == 0:
                        swap2_blurred = v2.functional.gaussian_blur(
                            base, kernel_size=[1, 1], sigma=[1e-6, 1e-6]
                        )
                    else:
                        k_size = int(2 * bs + 1)
                        s_val = float(max(bs, 1e-6))
                        swap2_blurred = v2.functional.gaussian_blur(
                            base, kernel_size=[k_size, k_size], sigma=[s_val, s_val]
                        )

                    scores_swap_b = self.sharpness_score(swap2_blurred)
                    score_new_swap_b = scores_swap_b["combined"].item() * 100.0
                    sharpness_diff_b = score_new_swap_b - score_new_original

                    if sharpness_diff_b < 0:
                        iteration_blur = 0 if bs == 0 else (bs - 1)
                        break
                    if abs(sharpness_diff_b) <= tolerance:
                        iteration_blur = bs
                        break
                    iteration_blur = bs
                break

            if abs(prev_alpha - alpha) < min_alpha_change:
                prev_alpha = (prev_alpha + alpha) / 2.0
                if abs(prev_alpha) <= 0.05:
                    prev_alpha = 0.0
                break

            prev_alpha = alpha
            iteration += 1

        if alpha_map_enable and (prev_alpha > 0.0):
            swap_final = swap * prev_alpha + swap_original * (1 - prev_alpha)

            s_map = self.sharpness_map(
                swap_final,
                mask=swap_mask,
                tenengrad_thresh=0.05,
                comb_weight=0.5,
                smooth_kernel=alpha_map_blur
                if (alpha_map_blur and alpha_map_blur % 2 == 1)
                else 0,
            )

            if swap_mask is not None:
                m = (
                    (swap_mask if swap_mask.dim() == 2 else swap_mask.squeeze(0))
                    .float()
                    .to(s_map.device)
                )
                denom = m.sum().clamp_min(1.0)
                mu = (s_map * m).sum() / denom
            else:
                mu = s_map.mean()

            dev = (s_map - mu).clamp(-1.0, 1.0)
            alpha_map = prev_alpha * (1.0 + alpha_map_strength * dev)
            alpha_map = alpha_map.clamp(0.0, 1.0)

            if swap_mask is not None:
                m = (
                    (swap_mask if swap_mask.dim() == 2 else swap_mask.squeeze(0))
                    .float()
                    .to(alpha_map.device)
                )
                alpha_map = alpha_map * m + prev_alpha * (1.0 - m)

            return alpha_map.unsqueeze(0), iteration_blur

        return prev_alpha, iteration_blur

    @torch.no_grad()
    def sharpness_score(
        self,
        image: torch.Tensor,
        mask: torch.Tensor = None,
        tenengrad_thresh: float = 0.05,
        comb_weight: float = 0.5,
    ) -> dict:
        image = image / 255.0

        gray = image.mean(dim=0, keepdim=True).unsqueeze(0)

        if mask is not None:
            m = mask.float()
            if m.dim() == 3:
                m = m.squeeze(0)
        else:
            m = None

        def valid_count(t):
            return m.sum().clamp(min=1.0) if m is not None else t.numel()

        L = F.conv2d(gray, self.kernel_lap, padding=1).squeeze()
        L2 = L.pow(2)
        if m is not None:
            L = L * m
            L2 = L2 * m
        cnt = valid_count(L2)
        mean_L2 = L2.sum() / cnt
        mean_L = L.sum() / cnt
        var_lap = (mean_L2 - mean_L.pow(2)).clamp(min=0.0)

        Gx = F.conv2d(gray, self.kernel_sobel_x, padding=1).squeeze()
        Gy = F.conv2d(gray, self.kernel_sobel_y, padding=1).squeeze()
        G = (Gx.pow(2) + Gy.pow(2)).sqrt()
        if m is not None:
            G = G * m
        total = cnt
        strong = (G > tenengrad_thresh).float().sum()
        ttengrad = strong / total

        combined = comb_weight * var_lap + (1 - comb_weight) * ttengrad

        return {"var_lap": var_lap, "ttengrad": ttengrad, "combined": combined}

    @torch.no_grad()
    def sharpness_map(
        self,
        image: torch.Tensor,
        mask: torch.Tensor | None = None,
        tenengrad_thresh: float = 0.05,
        comb_weight: float = 0.5,
        smooth_kernel: int = 5,
    ) -> torch.Tensor:
        eps = 1e-8
        device = image.device

        gray = (image / 255.0).mean(dim=0, keepdim=True).unsqueeze(0)

        lap = F.conv2d(gray, self.kernel_lap, padding=1).squeeze(0).squeeze(0)
        gx = (
            F.conv2d(gray, self.kernel_sobel_x, padding=1).squeeze(0).squeeze(0)
        )
        gy = F.conv2d(gray, self.kernel_sobel_y, padding=1).squeeze(0).squeeze(0)
        grad = (gx.pow(2) + gy.pow(2)).sqrt()

        def robust_norm(x, msk):
            if msk is not None:
                sel = x[msk > 0]
                if sel.numel() < 16:
                    sel = x.reshape(-1)
            else:
                sel = x.reshape(-1)
            p5 = (
                torch.quantile(sel, 0.05)
                if sel.numel() > 0
                else torch.tensor(0.0, device=device)
            )
            p95 = (
                torch.quantile(sel, 0.95)
                if sel.numel() > 0
                else torch.tensor(1.0, device=device)
            )
            y = (x - p5) / (p95 - p5 + eps)
            return y.clamp_(0, 1)

        m = None
        if mask is not None:
            m = (mask if mask.dim() == 2 else mask.squeeze(0)).float().to(device)

        lap_n = robust_norm(lap.abs(), m)
        grad_n = robust_norm(grad, m)

        smap = comb_weight * lap_n + (1.0 - comb_weight) * grad_n

        if smooth_kernel:
            if smooth_kernel % 2 == 0:
                smooth_kernel += 1
            smooth_kernel = max(3, smooth_kernel)
        if smooth_kernel and smooth_kernel >= 3:
            k = int(smooth_kernel)
            s_val = float(max(1, k // 2))
            smap3 = smap.unsqueeze(0).unsqueeze(0)

            smap = (
                v2.functional.gaussian_blur(
                    smap3, kernel_size=[k, k], sigma=[s_val, s_val]
                )
                .squeeze(0)
                .squeeze(0)
            )

        return smap.clamp(0, 1)

    @torch.no_grad()
    def apply_block_shift_gpu_jitter(
        self,
        img: torch.Tensor,
        block_size: int,
        max_amount_pixels: float,
        *,
        seed: int = 1337,
        pad_mode: str = "replicate",
        align_corners: bool = True,
    ) -> torch.Tensor:
        seed = seed + self.worker.frame_number * 17
        assert img.ndim == 3, "expected [C,H,W]"
        C, H, W = img.shape
        device = img.device
        dtype = img.dtype

        work = (
            img
            if img.dtype in (torch.float32, torch.float16, torch.bfloat16)
            else img.float()
        )

        B = max(1, int(block_size))
        H_pad = (B - (H % B)) % B
        W_pad = (B - (W % B)) % B
        if H_pad or W_pad:
            pad = (0, W_pad, 0, H_pad)
            mode = {
                "replicate": "replicate",
                "reflect": "reflect",
                "zeros": "constant",
            }[pad_mode]
            work = F.pad(work[None], pad=pad, mode=mode).squeeze(0)
        Hp, Wp = work.shape[-2:]

        nby = Hp // B
        nbx = Wp // B

        by_grid, bx_grid = torch.meshgrid(
            torch.arange(nby, device=device, dtype=torch.float32),
            torch.arange(nbx, device=device, dtype=torch.float32),
            indexing="ij",
        )
        h = torch.sin((bx_grid * 12.9898 + by_grid * 78.233 + float(seed)) * 43758.5453)
        frac = torch.frac(h * 0.5 + 0.5)

        _scaled_max = float(max_amount_pixels) / 4.0
        dx_base = ((frac) * 2.0 - 1.0) * _scaled_max

        h2 = torch.sin(
            (bx_grid * 96.233 + by_grid * 15.987 + (float(seed) + 101)) * 12345.6789
        )
        frac2 = torch.frac(h2 * 0.5 + 0.5)
        dy_base = ((frac2) * 2.0 - 1.0) * _scaled_max

        dx = torch.repeat_interleave(
            torch.repeat_interleave(dx_base, B, dim=0), B, dim=1
        )
        dy = torch.repeat_interleave(
            torch.repeat_interleave(dy_base, B, dim=0), B, dim=1
        )

        xs = torch.linspace(-1.0, 1.0, Wp, device=device)
        ys = torch.linspace(-1.0, 1.0, Hp, device=device)
        grid_y, grid_x = torch.meshgrid(ys, xs, indexing="ij")
        dx_norm = (2.0 * dx) / max(Wp - 1, 1)
        dy_norm = (2.0 * dy) / max(Hp - 1, 1)

        flow_x = grid_x + dx_norm
        flow_y = grid_y + dy_norm
        flow = torch.stack([flow_x, flow_y], dim=-1)

        warped = F.grid_sample(
            work[None],
            flow[None],
            mode="bilinear",
            padding_mode="border",
            align_corners=align_corners,
        ).squeeze(0)

        if H_pad or W_pad:
            warped = warped[..., :H, :W]

        if warped.dtype != dtype:
            warped = warped.to(dtype)

        return warped

    @torch.no_grad()
    def analyze_image(self, image):
        image = image.float() / 255.0
        C, H, W = image.shape
        grayscale = torch.mean(image, dim=0, keepdim=True)
        analysis = {}
        fft = torch.fft.fft2(grayscale)
        high_freq_energy = torch.mean(torch.abs(fft))
        analysis["jpeg_artifacts"] = min(high_freq_energy.item() / 50, 1.0)
        median_filtered = F.avg_pool2d(grayscale, 3, stride=1, padding=1)
        noise_map = torch.abs(grayscale - median_filtered)
        sp_noise = torch.mean((noise_map > 0.1).float())
        analysis["salt_pepper_noise"] = min(sp_noise.item() * 10, 1.0)
        local_var = F.avg_pool2d(grayscale**2, 5, stride=1, padding=2) - (
            F.avg_pool2d(grayscale, 5, stride=1, padding=2) ** 2
        )
        speckle_noise = torch.mean(local_var)
        analysis["speckle_noise"] = min(speckle_noise.item() * 50, 1.0)

        laplace_edges = F.conv2d(grayscale.unsqueeze(0), self.kernel_lap, padding=1)
        edge_strength = torch.mean(torch.abs(laplace_edges))
        analysis["blur"] = 1.0 - min(edge_strength.item() * 5, 1.0)
        contrast = grayscale.std()
        analysis["low_contrast"] = 1.0 - min(contrast.item() * 10, 1.0)
        return analysis