import traceback
import gc

import torch
import torch.nn as nn
from tqdm import tqdm

from .merging_methods import MergingMethod
from diffusers import UNet2DConditionModel


class _SimpleWrapper:
    def __init__(self, pipeline=None):
        """Wrap a pipeline or bare module so CheckpointSave can access all parts."""
        real_pipe = getattr(pipeline, "model", pipeline)

        # 2) Extract from the real pipeline, supporting bare modules
        if isinstance(real_pipe, UNet2DConditionModel):
            self._unet = real_pipe
        else:
            self._unet = getattr(real_pipe, "unet", None) or getattr(real_pipe, "diffusion_model", None)

        if isinstance(real_pipe, nn.Module) and self._unet is None and not any(
            hasattr(real_pipe, a) for a in ("clip", "text_encoder", "text_encoder_2", "vae")
        ):
            self._clip = real_pipe
        else:
            self._clip = getattr(real_pipe, "text_encoder", None) or getattr(real_pipe, "clip", None)

        self._vae = getattr(real_pipe, "vae", None)
        self._clip_vision = getattr(real_pipe, "text_encoder_2", None) or getattr(real_pipe, "clip_vision", None)

        if self._unet is None and isinstance(real_pipe, UNet2DConditionModel):
            self._unet = real_pipe
        if self._clip is None and isinstance(real_pipe, nn.Module) and self._unet is None:
            # treat bare nn.Module as a clip encoder when not a unet
            self._clip = real_pipe


        # Determine original device
        first_param = None
        for mdl in (self._unet, self._clip, self._vae, self._clip_vision):
            if mdl is not None:
                first_param = next(iter(mdl.parameters()), None)
                if first_param is not None:
                    break
        self.load_device = getattr(first_param, "device", torch.device("cpu"))

        # Build dummy model object
        dummy = type("SimplePipeline", (), {})()
        # expose the original pipeline so model_management can do self.model.model
        dummy.model = real_pipe

        # expose submodules on dummy
        if self._unet is not None:
            dummy.unet = self._unet
            dummy.diffusion_model = self._unet
        if self._clip is not None:
            dummy.clip = self._clip
            dummy.text_encoder = self._clip
            dummy.text_encoder_1 = self._clip
        if self._vae is not None:
            dummy.vae = self._vae
        if self._clip_vision is not None:
            dummy.clip_vision = self._clip_vision

        # expose the attributes CheckpointSave needs:
        dummy.parent          = getattr(pipeline, "parent", None)
        dummy.model_type      = getattr(pipeline, "model_type", None)
        dummy.model_sampling  = getattr(pipeline, "model_sampling", None)
        dummy.load_device     = self.load_device
        dummy.current_loaded_device = lambda: self.load_device
        dummy.model_size      = self.model_size
        dummy.loaded_size     = self.loaded_size
        dummy.model_patches_to= self.model_patches_to
        dummy.get_sd          = self.get_sd
        dummy.load_model      = lambda: dummy
        dummy.model_dtype     = self.model_dtype
        dummy.partially_load  = self.partially_load
        dummy.state_dict_for_saving = self.state_dict_for_saving
        dummy.get_model_object      = self.get_model_object
        dummy.model_load          = self.model_load
        dummy.model_memory_required = self.model_memory_required

        tok_func = getattr(real_pipe, "tokenize", None)
        self._tokenizer = None
        if callable(tok_func):
            dummy.tokenize = tok_func
        else:
            tok = getattr(real_pipe, "tokenizer", None) or getattr(real_pipe, "processor", None)
            if tok is not None and hasattr(tok, "__call__"):
                self._tokenizer = tok
                def _tok(text, **kw):
                    out = tok(text, return_tensors="pt", **kw)
                    return out["input_ids"] if isinstance(out, dict) else out
                dummy.tokenize = _tok

        # ---- INJECT encode_from_tokens_scheduled -----------------------------
        if hasattr(real_pipe, "encode_from_tokens_scheduled") and callable(real_pipe.encode_from_tokens_scheduled):
            dummy.encode_from_tokens_scheduled = real_pipe.encode_from_tokens_scheduled
        else:
            if hasattr(real_pipe, "get_text_features") and callable(real_pipe.get_text_features):
                dummy.encode_from_tokens_scheduled = lambda tokens, **kw: real_pipe.get_text_features(tokens, **kw)
            elif hasattr(real_pipe, "encode") and callable(real_pipe.encode):
                dummy.encode_from_tokens_scheduled = lambda tokens, **kw: real_pipe.encode(tokens, **kw)
            else:
                def _no_encode(tokens, **kw):
                    raise AttributeError(f"{type(self).__name__!r} wrapped object has no 'encode_from_tokens_scheduled' or fallback encode method.")
                dummy.encode_from_tokens_scheduled = _no_encode
        # ----------------------------------------------------------------------

        dummy.clone = lambda: _SimpleWrapper(pipeline=pipeline)

        self.model = dummy

    def tokenize(self, text, **kw):
        if hasattr(self.model, "tokenize"):
            return self.model.tokenize(text, **kw)
        if self._tokenizer is not None:
            out = self._tokenizer(text, return_tensors="pt", **kw)
            return out["input_ids"] if isinstance(out, dict) else out
        raise AttributeError(f"{type(self).__name__!r} has no attribute 'tokenize'")

    def clone(self):
        return _SimpleWrapper(pipeline=self.model.model)

    def model_load(self, lowvram_model_memory, force_patch_weights=False):
        # ComfyUI will call this to initialize model offloading/patching.
        # Simplest is to re-wrap everything on CPU or GPU as needed:
        return self.model_patches_to(self.load_device)

    def model_memory_required(self, device):
        # Used by ComfyUI to budget VRAM.
        # Return how much memory (in bytes) the *rest* of the model needs when offloaded.
        # A simple approximation is total size minus what’s already “loaded”:
        return self.model_size() - self.loaded_size()

    def get_model_object(self, name=None):
        if name is None:
            return self.model
        return getattr(self.model, name, None)

    def model_size(self):
        total = 0
        for mdl in (self._unet, self._clip, self._vae, self._clip_vision):
            if mdl is not None:
                total += sum(p.nelement() * p.element_size() for p in mdl.parameters())
        return total

    def loaded_size(self):
        # mirror of model_size
        return self.model_size()

    def model_patches_to(self, device):
        for mdl in (self._unet, self._clip, self._vae, self._clip_vision):
            if mdl is not None:
                mdl.to(device)
        self.load_device = device
        self.model.load_device = device
        return self

    def model_dtype(self):
        for mdl in (self._unet, self._clip, self._vae, self._clip_vision):
            if mdl is not None:
                try:
                    p = next(iter(mdl.parameters()))
                except StopIteration:
                    continue
                return p.dtype
        return torch.float32

    def partially_load(self, device, extra_memory, force_patch_weights=False):
        # minimal stub: just move everything to device
        return self.model_patches_to(device)

    def get_sd(self):
        # not used by CheckpointSave but present
        return {}

    def state_dict_for_saving(self, clip_sd=None, vae_sd=None, clip_vision_sd=None):
        sd = {}
        if self._unet:
            for k, v in self._unet.state_dict().items():
                sd[f"model.diffusion_model.{k}"] = v.half().cpu()
        if self._vae:
            for k, v in self._vae.state_dict().items():
                sd[f"first_stage_model.{k}"] = v.half().cpu()
        if self._clip:
            for k, v in self._clip.state_dict().items():
                sd[f"conditioner.embedders.0.transformer.{k}"] = v.half().cpu()
        if self._clip_vision:
            for k, v in self._clip_vision.state_dict().items():
                sd[f"conditioner.embedders.1.model.{k}"] = v.half().cpu()

        print(f"[SimpleWrapper] dumping {len(sd)} state_dict keys")
        return sd

    def __getattr__(self, name):
        # Avoid recursion during initialization
        if "model" not in self.__dict__:
            raise AttributeError(name)

        # fallback to dummy.model first
        if hasattr(self.model, name):
            return getattr(self.model, name)
        # then to submodules
        for attr in ("_unet", "_clip", "_vae", "_clip_vision"):
            mdl = getattr(self, attr)
            if mdl is not None and hasattr(mdl, name):
                return getattr(mdl, name)
        raise AttributeError(f"{type(self).__name__!r} has no attribute {name!r}")


# ─── HELPERS ──────────────────────────────────────────────────────────────────

def _get_unet(wrapper):
    mdl = getattr(wrapper, "model", wrapper)
    if isinstance(mdl, UNet2DConditionModel):
        return mdl
    for attr in ("unet", "diffusion_model", "unet1", "unets"):
        cand = getattr(mdl, attr, None)
        if isinstance(cand, UNet2DConditionModel):
            return cand
    raise AttributeError(f"No U-Net found on {type(mdl).__name__}")


def _get_clip(wrapper):
    mdl = getattr(wrapper, "model", wrapper)
    if isinstance(mdl, nn.Module) and not any(hasattr(mdl, a) for a in ("clip", "text_encoder", "text_encoder_1")):
        return mdl
    for attr in ("clip", "text_encoder", "text_encoder_1"):
        cand = getattr(mdl, attr, None)
        if isinstance(cand, nn.Module):
            return cand
    raise AttributeError(f"No CLIP encoder found on {type(mdl).__name__}")


def _unwrap_pipeline(obj, _seen=None):
    if _seen is None:
        _seen = set()
    if id(obj) in _seen:
        return obj
    _seen.add(id(obj))
    if isinstance(obj, _SimpleWrapper):
        return _unwrap_pipeline(obj.model, _seen)
    nxt = getattr(obj, "model", None)
    if nxt is not None and nxt is not obj:
        return _unwrap_pipeline(nxt, _seen)
    return obj


# ─── MERGE NODES ──────────────────────────────────────────────────────────────

class DonutWidenMergeUNet:
    class_type = "CUSTOM"; aux_id = "DonutsDelivery/ComfyUI-DonutNodes"
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "models": ("MODELLIST",),
            "exclude_regex": ("STRING", {"default": ""}),
            "above_avg": ("FLOAT", {"default": 1.0, "min": 0.0}),
            "score_calib": ("FLOAT", {"default": 1.0, "min": 0.0}),
        }}
    RETURN_TYPES = ("MODEL",)
    FUNCTION = "execute"
    CATEGORY = "merging"

    def execute(self, models, exclude_regex, above_avg, score_calib):
        try:
            orig = models[0]
            unets = [_get_unet(m) for m in models]

            cpu, gpu = torch.device("cpu"), next(unets[0].parameters()).device
            for u in unets: u.to(cpu)

            regexes = [r.strip() for r in exclude_regex.split(",") if r.strip()]
            merger = MergingMethod("widen_merging")
            with torch.no_grad():
                merged = merger.widen_merging(
                    merged_model=unets[0],
                    models_to_merge=unets[1:],
                    exclude_param_names_regex=regexes,
                    above_average_value_ratio=above_avg,
                    score_calibration_value=score_calib,
                )
            if isinstance(merged, dict) and merged:
                unets[0].load_state_dict(merged, strict=False)
            for u in unets:
                u.to(gpu)
            gc.collect()

            base_pipe = _unwrap_pipeline(orig)
            if hasattr(base_pipe, "unet"):
                base_pipe.unet = unets[0]
            return (_SimpleWrapper(pipeline=base_pipe),)

        except Exception:
            traceback.print_exc()
            raise


class DonutWidenMergeCLIP:
    class_type = "CUSTOM"; aux_id = "DonutsDelivery/ComfyUI-DonutNodes"
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "clips": ("CLIPLIST",),
            "exclude_regex": ("STRING", {"default": ""}),
            "above_avg": ("FLOAT", {"default": 1.0, "min": 0.0}),
            "score_calib": ("FLOAT", {"default": 1.0, "min": 0.0}),
        }}
    RETURN_TYPES = ("CLIP",)
    FUNCTION = "execute"
    CATEGORY = "merging"

    def execute(self, clips, exclude_regex, above_avg, score_calib):
        try:
            orig = clips[0]
            encs = [_get_clip(c) for c in clips]

            cpu, gpu = torch.device("cpu"), next(encs[0].parameters()).device
            for c in encs: c.to(cpu)

            regexes = [r.strip() for r in exclude_regex.split(",") if r.strip()]
            merger = MergingMethod("widen_merging")
            with torch.no_grad():
                merged = merger.widen_merging(
                    merged_model=encs[0],
                    models_to_merge=encs[1:],
                    exclude_param_names_regex=regexes,
                    above_average_value_ratio=above_avg,
                    score_calibration_value=score_calib,
                )
            if isinstance(merged, dict) and merged:
                encs[0].load_state_dict(merged, strict=False)
            for c in encs:
                c.to(gpu)
            gc.collect()

            base_pipe = _unwrap_pipeline(orig)
            if hasattr(base_pipe, "text_encoder"):
                base_pipe.text_encoder = encs[0]
            return (_SimpleWrapper(pipeline=base_pipe),)

        except Exception:
            traceback.print_exc()
            raise


# ─── EXPORT ───────────────────────────────────────────────────────────────────

NODE_CLASS_MAPPINGS = {
    "DonutWidenMergeUNet": DonutWidenMergeUNet,
    "DonutWidenMergeCLIP": DonutWidenMergeCLIP,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    k: k.replace("Donut", "Donut ") for k in NODE_CLASS_MAPPINGS
}
