"""
Mapping registry linking AI models (ONNX/PyTorch) to their respective UI toggles.
Used by the ModelsProcessor.
"""

from typing import List, Dict
from enum import Enum
from dataclasses import dataclass


class ToggleScope(Enum):
    GLOBAL = "global"
    PER_FACE = "per_face"


@dataclass
class ToggleDef:
    key: str
    scope: ToggleScope
    # Optional gate on a selection widget that lives in the same parameter dict as
    # the toggle. The toggle only keeps the model "active" when the gate matches.
    selection_key: str | None = None
    selection_equals: str | None = None
    selection_not_equals: str | None = None
    requires_toggle: str | None = None


# The Face Expression Restorer has two back-ends that are chosen per face with
# FaceExpressionModeSelection: LivePortrait ("Simple" / "Advanced") and
# PerformRecast ("Recast"). Each back-end's models are only needed while at least
# one face actually uses it. (LivePortrait models stay protected by the other
# features that use them: Face Editor, Auto Mouth, makeups.)
_EXPR_NOT_RECAST = ToggleDef(
    "FaceExpressionEnableBothToggle",
    ToggleScope.PER_FACE,
    selection_key="FaceExpressionModeSelection",
    selection_not_equals="Recast",
)
_EXPR_RECAST_ONLY = ToggleDef(
    "FaceExpressionEnableBothToggle",
    ToggleScope.PER_FACE,
    selection_key="FaceExpressionModeSelection",
    selection_equals="Recast",
)


# Centralized dictionary mapping exact model_name (from models_data.py) to UI toggle definitions.
# If ANY of the mapped toggles is True in its respective scope, the model is allowed to run.
# Models missing from this list (like Core Detectors, Swappers, ArcFace or facial landmarks)
# are considered always active if requested, bypassing the UI state check.
MODELS_TOGGLE_MAP: Dict[str, List[ToggleDef]] = {
    # --- DENOISERS (ReF-LDM UNet & VAEs) ---
    "RefLDMVAEEncoder": [
        ToggleDef("DenoiserUNetEnableBeforeRestorersToggle", ToggleScope.GLOBAL),
        ToggleDef("DenoiserAfterFirstRestorerToggle", ToggleScope.GLOBAL),
        ToggleDef("DenoiserAfterRestorersToggle", ToggleScope.GLOBAL),
    ],
    "RefLDMVAEDecoder": [
        ToggleDef("DenoiserUNetEnableBeforeRestorersToggle", ToggleScope.GLOBAL),
        ToggleDef("DenoiserAfterFirstRestorerToggle", ToggleScope.GLOBAL),
        ToggleDef("DenoiserAfterRestorersToggle", ToggleScope.GLOBAL),
    ],
    "RefLDM_UNET_EXTERNAL_KV": [
        ToggleDef("DenoiserUNetEnableBeforeRestorersToggle", ToggleScope.GLOBAL),
        ToggleDef("DenoiserAfterFirstRestorerToggle", ToggleScope.GLOBAL),
        ToggleDef("DenoiserAfterRestorersToggle", ToggleScope.GLOBAL),
    ],
    "KVExtractor": [
        ToggleDef("DenoiserUNetEnableBeforeRestorersToggle", ToggleScope.GLOBAL),
        ToggleDef("DenoiserAfterFirstRestorerToggle", ToggleScope.GLOBAL),
        ToggleDef("DenoiserAfterRestorersToggle", ToggleScope.GLOBAL),
    ],
    # --- FRAME ENHANCERS (Background / Full Image) ---
    "RealEsrganx2Plus": [ToggleDef("FrameEnhancerEnableToggle", ToggleScope.GLOBAL)],
    "RealEsrganx4Plus": [ToggleDef("FrameEnhancerEnableToggle", ToggleScope.GLOBAL)],
    "RealEsrx4v3": [ToggleDef("FrameEnhancerEnableToggle", ToggleScope.GLOBAL)],
    "BSRGANx2": [ToggleDef("FrameEnhancerEnableToggle", ToggleScope.GLOBAL)],
    "BSRGANx4": [ToggleDef("FrameEnhancerEnableToggle", ToggleScope.GLOBAL)],
    "UltraSharpx4": [ToggleDef("FrameEnhancerEnableToggle", ToggleScope.GLOBAL)],
    "UltraMixx4": [ToggleDef("FrameEnhancerEnableToggle", ToggleScope.GLOBAL)],
    "DeoldifyArt": [ToggleDef("FrameEnhancerEnableToggle", ToggleScope.GLOBAL)],
    "DeoldifyStable": [ToggleDef("FrameEnhancerEnableToggle", ToggleScope.GLOBAL)],
    "DeoldifyVideo": [ToggleDef("FrameEnhancerEnableToggle", ToggleScope.GLOBAL)],
    "DDColorArt": [ToggleDef("FrameEnhancerEnableToggle", ToggleScope.GLOBAL)],
    "DDcolor": [ToggleDef("FrameEnhancerEnableToggle", ToggleScope.GLOBAL)],
    # --- FACE RESTORERS ---
    "GFPGANv1.4": [
        ToggleDef("FaceRestorerEnableToggle", ToggleScope.PER_FACE),
        ToggleDef("FaceRestorerEnable2Toggle", ToggleScope.PER_FACE),
    ],
    "GFPGAN1024": [
        ToggleDef("FaceRestorerEnableToggle", ToggleScope.PER_FACE),
        ToggleDef("FaceRestorerEnable2Toggle", ToggleScope.PER_FACE),
    ],
    "CodeFormer": [
        ToggleDef("FaceRestorerEnableToggle", ToggleScope.PER_FACE),
        ToggleDef("FaceRestorerEnable2Toggle", ToggleScope.PER_FACE),
    ],
    "GPENBFR256": [
        ToggleDef("FaceRestorerEnableToggle", ToggleScope.PER_FACE),
        ToggleDef("FaceRestorerEnable2Toggle", ToggleScope.PER_FACE),
    ],
    "GPENBFR512": [
        ToggleDef("FaceRestorerEnableToggle", ToggleScope.PER_FACE),
        ToggleDef("FaceRestorerEnable2Toggle", ToggleScope.PER_FACE),
    ],
    "GPENBFR1024": [
        ToggleDef("FaceRestorerEnableToggle", ToggleScope.PER_FACE),
        ToggleDef("FaceRestorerEnable2Toggle", ToggleScope.PER_FACE),
    ],
    "GPENBFR2048": [
        ToggleDef("FaceRestorerEnableToggle", ToggleScope.PER_FACE),
        ToggleDef("FaceRestorerEnable2Toggle", ToggleScope.PER_FACE),
    ],
    "RestoreFormerPlusPlus": [
        ToggleDef("FaceRestorerEnableToggle", ToggleScope.PER_FACE),
        ToggleDef("FaceRestorerEnable2Toggle", ToggleScope.PER_FACE),
    ],
    "VQFRv2": [
        ToggleDef("FaceRestorerEnableToggle", ToggleScope.PER_FACE),
        ToggleDef("FaceRestorerEnable2Toggle", ToggleScope.PER_FACE),
    ],
    "OSDFacePromptEncoder": [
        ToggleDef("FaceRestorerEnableToggle", ToggleScope.PER_FACE),
        ToggleDef("FaceRestorerEnable2Toggle", ToggleScope.PER_FACE),
    ],
    "OSDFaceVAEEncoder": [
        ToggleDef("FaceRestorerEnableToggle", ToggleScope.PER_FACE),
        ToggleDef("FaceRestorerEnable2Toggle", ToggleScope.PER_FACE),
    ],
    "OSDFaceUNet": [
        ToggleDef("FaceRestorerEnableToggle", ToggleScope.PER_FACE),
        ToggleDef("FaceRestorerEnable2Toggle", ToggleScope.PER_FACE),
    ],
    "OSDFaceVAEDecoder": [
        ToggleDef("FaceRestorerEnableToggle", ToggleScope.PER_FACE),
        ToggleDef("FaceRestorerEnable2Toggle", ToggleScope.PER_FACE),
    ],
    # --- RE-AGING ---
    "FaceReaging": [ToggleDef("FaceReagingEnableToggle", ToggleScope.PER_FACE)],
    # --- FACE MASKS / OCCLUDERS / TEXTURES ---
    "Occluder": [ToggleDef("OccluderEnableToggle", ToggleScope.PER_FACE)],
    "XSeg": [ToggleDef("DFLXSegEnableToggle", ToggleScope.PER_FACE)],
    "RD64ClipText": [ToggleDef("ClipEnableToggle", ToggleScope.PER_FACE)],
    "FaceParser": [
        ToggleDef("FaceParserEnableToggle", ToggleScope.PER_FACE),
        ToggleDef("RestoreEyesEnableToggle", ToggleScope.PER_FACE),
        ToggleDef("RestoreMouthEnableToggle", ToggleScope.PER_FACE),
        ToggleDef("MouthParserStretchToggle", ToggleScope.PER_FACE),
        ToggleDef("TransferTextureEnableToggle", ToggleScope.PER_FACE),
        ToggleDef("DifferencingEnableToggle", ToggleScope.PER_FACE),
        ToggleDef("DFLXSegEnableToggle", ToggleScope.PER_FACE),  # Pour le XSegMouth
    ],
    "combo_relu3_3_relu3_1": [
        ToggleDef("TransferTextureEnableToggle", ToggleScope.PER_FACE),
        ToggleDef("DifferencingEnableToggle", ToggleScope.PER_FACE),
    ],
    # --- FACE EDITORS & EXPRESSIONS (LivePortrait) ---
    "LivePortraitMotionExtractor": [
        _EXPR_NOT_RECAST,
        ToggleDef("AutoMouthExpressionEnableToggle", ToggleScope.PER_FACE),
        ToggleDef("FaceEditorEnableToggle", ToggleScope.PER_FACE),
        ToggleDef("FaceMakeupEnableToggle", ToggleScope.PER_FACE),
        ToggleDef("HairMakeupEnableToggle", ToggleScope.PER_FACE),
        ToggleDef("EyeBrowsMakeupEnableToggle", ToggleScope.PER_FACE),
        ToggleDef("LipsMakeupEnableToggle", ToggleScope.PER_FACE),
    ],
    "LivePortraitAppearanceFeatureExtractor": [
        _EXPR_NOT_RECAST,
        ToggleDef("AutoMouthExpressionEnableToggle", ToggleScope.PER_FACE),
        ToggleDef("FaceEditorEnableToggle", ToggleScope.PER_FACE),
        ToggleDef("FaceMakeupEnableToggle", ToggleScope.PER_FACE),
        ToggleDef("HairMakeupEnableToggle", ToggleScope.PER_FACE),
        ToggleDef("EyeBrowsMakeupEnableToggle", ToggleScope.PER_FACE),
        ToggleDef("LipsMakeupEnableToggle", ToggleScope.PER_FACE),
    ],
    "LivePortraitStitching": [
        _EXPR_NOT_RECAST,
        ToggleDef("AutoMouthExpressionEnableToggle", ToggleScope.PER_FACE),
        ToggleDef("FaceEditorEnableToggle", ToggleScope.PER_FACE),
        ToggleDef("FaceMakeupEnableToggle", ToggleScope.PER_FACE),
        ToggleDef("HairMakeupEnableToggle", ToggleScope.PER_FACE),
        ToggleDef("EyeBrowsMakeupEnableToggle", ToggleScope.PER_FACE),
        ToggleDef("LipsMakeupEnableToggle", ToggleScope.PER_FACE),
    ],
    "LivePortraitStitchingEye": [
        _EXPR_NOT_RECAST,
        ToggleDef("AutoMouthExpressionEnableToggle", ToggleScope.PER_FACE),
        ToggleDef("FaceEditorEnableToggle", ToggleScope.PER_FACE),
        ToggleDef("FaceMakeupEnableToggle", ToggleScope.PER_FACE),
        ToggleDef("HairMakeupEnableToggle", ToggleScope.PER_FACE),
        ToggleDef("EyeBrowsMakeupEnableToggle", ToggleScope.PER_FACE),
        ToggleDef("LipsMakeupEnableToggle", ToggleScope.PER_FACE),
        ToggleDef(
            "FaceExpressionRetargetingEyesBothEnableToggle",
            ToggleScope.PER_FACE,
            selection_key="FaceExpressionModeSelection",
            selection_equals="Advanced",
            requires_toggle="FaceExpressionEnableBothToggle",
        ),
    ],
    "LivePortraitStitchingLip": [
        _EXPR_NOT_RECAST,
        ToggleDef("AutoMouthExpressionEnableToggle", ToggleScope.PER_FACE),
        ToggleDef("FaceEditorEnableToggle", ToggleScope.PER_FACE),
        ToggleDef("FaceMakeupEnableToggle", ToggleScope.PER_FACE),
        ToggleDef("HairMakeupEnableToggle", ToggleScope.PER_FACE),
        ToggleDef("EyeBrowsMakeupEnableToggle", ToggleScope.PER_FACE),
        ToggleDef("LipsMakeupEnableToggle", ToggleScope.PER_FACE),
        ToggleDef(
            "FaceExpressionRetargetingLipsBothEnableToggle",
            ToggleScope.PER_FACE,
            selection_key="FaceExpressionModeSelection",
            selection_equals="Advanced",
            requires_toggle="FaceExpressionEnableBothToggle",
        ),
    ],
    "LivePortraitWarpingSpade": [
        _EXPR_NOT_RECAST,
        ToggleDef("AutoMouthExpressionEnableToggle", ToggleScope.PER_FACE),
        ToggleDef("FaceEditorEnableToggle", ToggleScope.PER_FACE),
        ToggleDef("FaceMakeupEnableToggle", ToggleScope.PER_FACE),
        ToggleDef("HairMakeupEnableToggle", ToggleScope.PER_FACE),
        ToggleDef("EyeBrowsMakeupEnableToggle", ToggleScope.PER_FACE),
        ToggleDef("LipsMakeupEnableToggle", ToggleScope.PER_FACE),
    ],
    "LivePortraitWarpingSpadeFix": [
        _EXPR_NOT_RECAST,
        ToggleDef("AutoMouthExpressionEnableToggle", ToggleScope.PER_FACE),
        ToggleDef("FaceEditorEnableToggle", ToggleScope.PER_FACE),
        ToggleDef("FaceMakeupEnableToggle", ToggleScope.PER_FACE),
        ToggleDef("HairMakeupEnableToggle", ToggleScope.PER_FACE),
        ToggleDef("EyeBrowsMakeupEnableToggle", ToggleScope.PER_FACE),
        ToggleDef("LipsMakeupEnableToggle", ToggleScope.PER_FACE),
    ],
    "PerformRecastAppearanceFeatureExtractor": [
        _EXPR_RECAST_ONLY
    ],
    "PerformRecastMotionExtractor": [
        _EXPR_RECAST_ONLY
    ],
    "PerformRecastWarpingModule": [
        _EXPR_RECAST_ONLY
    ],
    "PerformRecastSpadeGenerator": [
        _EXPR_RECAST_ONLY
    ],
}


def get_toggles_for_model(model_name: str) -> List[ToggleDef]:
    """
    Returns the list of UI toggle definitions associated with a specific model.
    """
    return MODELS_TOGGLE_MAP.get(model_name, [])
