import traceback
import gc

import torch
import torch.nn as nn
from tqdm import tqdm

from .merging_methods import MergingMethod
from diffusers import UNet2DConditionModel


# Simple wrapper so that bare nn.Modules can be passed to
# ComfyUI's CheckpointSave node which expects a MODEL type
# exposing a `.model` attribute and `get_model_object()` method.
class _SimpleWrapper:
    def __init__(self, unet=None, clip=None):
        dummy = type("SimplePipeline", (), {})()
        if unet is not None:
            dummy.unet = unet
            dummy.diffusion_model = unet
        if clip is not None:
            dummy.clip = clip
            dummy.text_encoder = clip
            dummy.text_encoder_1 = clip
        self.model = dummy
        self.clip = getattr(dummy, "clip", None)

    def get_model_object(self):
        return self.model


# ─── HELPERS ────────────────────────────────────────────────────────────────────

def _get_unet(wrapper):
    """
    Given a ComfyUI pipeline/Model wrapper, pull out the actual UNet2DConditionModel.
    """
    mdl = getattr(wrapper, "model", wrapper)
    if isinstance(mdl, UNet2DConditionModel):
        return mdl
    for attr in ("unet", "diffusion_model", "unet1", "unets"):
        cand = getattr(mdl, attr, None)
        if isinstance(cand, UNet2DConditionModel):
            return cand
    raise AttributeError(f"No U-Net found on {type(mdl).__name__}")


def _get_clip(wrapper):
    """
    Given a ComfyUI pipeline/Model wrapper, pull out its CLIP encoder (nn.Module).
    """
    mdl = getattr(wrapper, "model", wrapper)
    # If it's already an nn.Module without pipeline attrs, assume it's the CLIP encoder:
    if isinstance(mdl, nn.Module) and not any(
        hasattr(mdl, a) for a in ("clip", "text_encoder", "text_encoder_1")
    ):
        return mdl
    for attr in ("clip", "text_encoder", "text_encoder_1"):
        cand = getattr(mdl, attr, None)
        if isinstance(cand, nn.Module):
            return cand
    raise AttributeError(f"No CLIP encoder found on {type(mdl).__name__}")


# ─── MODEL/CLIP LIST NODES (unchanged) ──────────────────────────────────────────

class DonutMakeModelList2:
    class_type = "CUSTOM"; aux_id = "DonutsDelivery/ComfyUI-DonutNodes"
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"a": ("MODEL",), "b": ("MODEL",)}}
    RETURN_TYPES = ("MODELLIST",); FUNCTION = "execute"; CATEGORY = "merging"
    def execute(self, a, b): return ([a, b],)


class DonutAppendModelToList:
    class_type = "CUSTOM"; aux_id = "DonutsDelivery/ComfyUI-DonutNodes"
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"lst": ("MODELLIST",), "m": ("MODEL",)}}
    RETURN_TYPES = ("MODELLIST",); FUNCTION = "execute"; CATEGORY = "merging"
    def execute(self, lst, m): return (lst + [m],)


class DonutMergeModelLists:
    class_type = "CUSTOM"; aux_id = "DonutsDelivery/ComfyUI-DonutNodes"
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"x": ("MODELLIST",), "y": ("MODELLIST",)}}
    RETURN_TYPES = ("MODELLIST",); FUNCTION = "execute"; CATEGORY = "merging"
    def execute(self, x, y): return (x + y,)


class DonutMakeClipList2:
    class_type = "CUSTOM"; aux_id = "DonutsDelivery/ComfyUI-DonutNodes"
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"a": ("CLIP",), "b": ("CLIP",)}}
    RETURN_TYPES = ("CLIPLIST",); FUNCTION = "execute"; CATEGORY = "merging"
    def execute(self, a, b): return ([a, b],)


class DonutAppendClipToList:
    class_type = "CUSTOM"; aux_id = "DonutsDelivery/ComfyUI-DonutNodes"
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"lst": ("CLIPLIST",), "c": ("CLIP",)}}
    RETURN_TYPES = ("CLIPLIST",); FUNCTION = "execute"; CATEGORY = "merging"
    def execute(self, lst, c): return (lst + [c],)


class DonutMergeClipLists:
    class_type = "CUSTOM"; aux_id = "DonutsDelivery/ComfyUI-DonutNodes"
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"x": ("CLIPLIST",), "y": ("CLIPLIST",)}}
    RETURN_TYPES = ("CLIPLIST",); FUNCTION = "execute"; CATEGORY = "merging"
    def execute(self, x, y): return (x + y,)


# ─── WIDEN MERGE UNET ──────────────────────────────────────────────────────────

class DonutWidenMergeUNet:
    """Widen-merge a list of SDXL U-Nets in-place on the original pipeline wrapper."""
    class_type = "CUSTOM"; aux_id = "DonutsDelivery/ComfyUI-DonutNodes"

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "models":        ("MODELLIST",),
            "exclude_regex": ("STRING", {"default": ""}),
            "above_avg":     ("FLOAT",  {"default": 1.0, "min": 0.0}),
            "score_calib":   ("FLOAT",  {"default": 1.0, "min": 0.0}),
        }}

    RETURN_TYPES = ("MODEL",)
    FUNCTION     = "execute"
    CATEGORY     = "merging"

    def execute(self, models, exclude_regex, above_avg, score_calib):
        try:
            print(f"\n[DonutWidenMergeUNet] Starting merge of {len(models)} pipelines…")
            # 1) keep the original ComfyUI model wrapper
            orig_wrapper = models[0]

            # 2) extract every UNet2DConditionModel (in each wrapper)
            wrappers = models
            unets    = [_get_unet(w) for w in wrappers]

            # 3) grab devices
            cpu   = torch.device("cpu")
            gpu   = next(unets[0].parameters()).device

            # 4) offload all to CPU
            for u in unets:
                u.to(cpu)

            # 5) parse your regex list
            regexes = [r.strip() for r in exclude_regex.split(",") if r.strip()]

            # 6) run widen_merging on CPU
            merger = MergingMethod("widen_merging")
            with torch.no_grad():
                print("[DonutWidenMergeUNet] Invoking MergingMethod.widen_merging()…")
                merged_weights = merger.widen_merging(
                    merged_model=unets[0],
                    models_to_merge=unets[1:],
                    exclude_param_names_regex=regexes,
                    above_average_value_ratio=above_avg,
                    score_calibration_value=score_calib,
                )

            # 7) if any weights came back, load them
            if isinstance(merged_weights, dict) and merged_weights:
                print("[DonutWidenMergeUNet] Loading merged weights back onto original UNet…")
                unets[0].load_state_dict(merged_weights, strict=False)
            else:
                print("[DonutWidenMergeUNet] ⚠️ widen_merging returned no weights; skipping load.")

            # 8) move everything back up to GPU
            for u in unets:
                u.to(gpu)

            print("[DonutWidenMergeUNet] Merge complete!\n")

            # 9) cleanup intermediate modules
            del unets
            torch.cuda.empty_cache()
            gc.collect()

            # 10) return the wrapper. If the input was a bare UNet module,
            # wrap it so the CheckpointSave node receives a compatible object.
            if isinstance(orig_wrapper, UNet2DConditionModel):
                return (_SimpleWrapper(unet=orig_wrapper),)
            else:
                return (orig_wrapper,)

        except Exception:
            print("\n[DonutWidenMergeUNet] *** Exception during merge ***")
            traceback.print_exc()
            raise


# ─── WIDEN MERGE CLIP ──────────────────────────────────────────────────────────

class DonutWidenMergeCLIP:
    """Widen-merge a list of SDXL CLIP encoders in-place on the original wrapper."""
    class_type = "CUSTOM"; aux_id = "DonutsDelivery/ComfyUI-DonutNodes"

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "clips":         ("CLIPLIST",),
            "exclude_regex": ("STRING", {"default": ""}),
            "above_avg":     ("FLOAT",  {"default": 1.0, "min": 0.0}),
            "score_calib":   ("FLOAT",  {"default": 1.0, "min": 0.0}),
        }}

    RETURN_TYPES = ("CLIP",)
    FUNCTION     = "execute"
    CATEGORY     = "merging"

    def execute(self, clips, exclude_regex, above_avg, score_calib):
        try:
            print(f"\n[DonutWidenMergeCLIP] Starting merge of {len(clips)} CLIP encoders…")
            orig_wrapper = clips[0]
            wrappers     = clips
            encs         = [_get_clip(w) for w in wrappers]

            # offload
            cpu = torch.device("cpu")
            for c in encs:
                c.to(cpu)

            # parse
            regexes = [r.strip() for r in exclude_regex.split(",") if r.strip()]

            # merge
            merger = MergingMethod("widen_merging")
            with torch.no_grad():
                print("[DonutWidenMergeCLIP] Invoking MergingMethod.widen_merging()…")
                merged_weights = merger.widen_merging(
                    merged_model=encs[0],
                    models_to_merge=encs[1:],
                    exclude_param_names_regex=regexes,
                    above_average_value_ratio=above_avg,
                    score_calibration_value=score_calib,
                )

            # load back onto the first CLIP
            if isinstance(merged_weights, dict) and merged_weights:
                print("[DonutWidenMergeCLIP] Loading merged weights…")
                encs[0].load_state_dict(merged_weights, strict=False)
            else:
                print("[DonutWidenMergeCLIP] ⚠️ widen_merging returned no weights; skipping load.")

            # done
            print("[DonutWidenMergeCLIP] Merge complete!\n")

            # cleanup
            del encs
            torch.cuda.empty_cache()
            gc.collect()

            # return the wrapper. If the input was a bare CLIP module, wrap it
            # so the CheckpointSave node can use it.
            if isinstance(orig_wrapper, nn.Module) and not hasattr(orig_wrapper, "model"):
                return (_SimpleWrapper(clip=orig_wrapper),)
            else:
                return (orig_wrapper,)

        except Exception:
            print("\n[DonutWidenMergeCLIP] *** Exception during merge ***")
            traceback.print_exc()
            raise


# ─── EXPORT ───────────────────────────────────────────────────────────────────

NODE_CLASS_MAPPINGS = {
    "DonutMakeModelList2":    DonutMakeModelList2,
    "DonutAppendModelToList": DonutAppendModelToList,
    "DonutMergeModelLists":   DonutMergeModelLists,
    "DonutMakeClipList2":     DonutMakeClipList2,
    "DonutAppendClipToList":  DonutAppendClipToList,
    "DonutMergeClipLists":    DonutMergeClipLists,
    "DonutWidenMergeUNet":    DonutWidenMergeUNet,
    "DonutWidenMergeCLIP":    DonutWidenMergeCLIP,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    k: k.replace("Donut", "Donut ") for k in NODE_CLASS_MAPPINGS
}
