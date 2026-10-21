"""HQ-SAM-inspired UNETR model for prompt-free three-channel prediction."""

from __future__ import annotations

from collections import OrderedDict
from collections.abc import Mapping
from os import PathLike
from typing import Any

import torch
from torch import nn
from torch.nn import functional as F

CHANNEL_LAYOUT = {
    "foreground": 0,
    "center_distances": 1,
    "boundary_distances": 2,
}
OUT_CHANNELS = len(CHANNEL_LAYOUT)
FREEZE_POLICIES = ("stage1", "lightft", "fullft", "head_only")
SAM_PIXEL_MEAN = (123.675, 116.28, 103.53)
SAM_PIXEL_STD = (58.395, 57.12, 57.375)


def _unwrap(module: nn.Module) -> nn.Module:
    return module.module if hasattr(module, "module") else module


def _set_trainable(module: nn.Module, trainable: bool) -> None:
    for parameter in module.parameters():
        parameter.requires_grad = bool(trainable)


def parameter_summary(module: nn.Module) -> dict[str, int]:
    """Return total and trainable parameter counts."""

    total = sum(int(parameter.numel()) for parameter in module.parameters())
    trainable = sum(int(parameter.numel()) for parameter in module.parameters() if parameter.requires_grad)
    return {"total": total, "trainable": trainable}


def _is_tiny_vit_encoder(image_encoder: nn.Module) -> bool:
    patch_embed = getattr(image_encoder, "patch_embed", None)
    return hasattr(image_encoder, "layers") and patch_embed is not None and not hasattr(patch_embed, "proj")


def _infer_early_dim(image_encoder: nn.Module) -> int:
    patch_embed = getattr(image_encoder, "patch_embed", None)
    projection = getattr(patch_embed, "proj", None)
    if projection is not None and hasattr(projection, "out_channels"):
        return int(projection.out_channels)
    if _is_tiny_vit_encoder(image_encoder):
        neck = getattr(image_encoder, "neck", None)
        first = neck[0] if neck is not None and len(neck) > 0 else None
        if first is not None and hasattr(first, "in_channels"):
            return int(first.in_channels)
    embed_dim = getattr(patch_embed, "embed_dim", None)
    if embed_dim is not None:
        return int(embed_dim)
    raise AttributeError("Could not infer the early SAM feature dimension")


def _infer_final_dim(image_encoder: nn.Module) -> int:
    neck = getattr(image_encoder, "neck", None)
    if neck is not None:
        for module in reversed(list(neck.modules())):
            if isinstance(module, nn.Conv2d):
                return int(module.out_channels)
            if hasattr(module, "out_channels"):
                return int(module.out_channels)
    raise AttributeError("Could not infer the final SAM feature dimension")


def _as_nchw(feature: torch.Tensor, expected_channels: int | None = None) -> torch.Tensor:
    if feature.ndim != 4:
        raise ValueError(f"Expected a four-dimensional feature tensor, got {tuple(feature.shape)}")
    if expected_channels is not None and int(feature.shape[1]) == int(expected_channels):
        return feature.contiguous()
    if expected_channels is not None and int(feature.shape[-1]) == int(expected_channels):
        return feature.permute(0, 3, 1, 2).contiguous()
    if int(feature.shape[1]) <= int(feature.shape[-1]):
        return feature.contiguous()
    return feature.permute(0, 3, 1, 2).contiguous()


def _collect_tiny_vit_features(
    image_encoder: nn.Module,
    images: torch.Tensor,
    selected_intermediate_index: int,
) -> tuple[torch.Tensor, torch.Tensor, dict[str, Any]]:
    features = image_encoder.patch_embed(images)
    layer_features = []
    for layer in image_encoder.layers:
        features = layer(features)
        layer_features.append(features)

    if features.ndim != 3:
        raise RuntimeError(f"Expected TinyViT token features, got {tuple(features.shape)}")
    batch_size, num_tokens, channels = features.shape
    spatial = int(round(float(num_tokens) ** 0.5))
    if spatial * spatial != int(num_tokens):
        raise RuntimeError(f"TinyViT token grid is not square: {num_tokens}")

    # MobileSAM changes channel width across stages. The final token map is the
    # stable local feature accepted by the public neck and residual projection.
    early = features.view(batch_size, spatial, spatial, channels).permute(0, 3, 1, 2).contiguous()
    final = image_encoder.neck(early)
    selected = max(0, min(int(selected_intermediate_index), len(layer_features) - 1)) if layer_features else 0
    info = {
        "encoder_family": "tiny_vit",
        "selected_intermediate_index": selected,
        "num_intermediate_features": len(layer_features),
        "early_shape": list(early.shape),
        "final_shape": list(final.shape),
    }
    return final, early, info


def collect_sam_features(
    image_encoder: nn.Module,
    images: torch.Tensor,
    *,
    selected_intermediate_index: int = 0,
) -> tuple[torch.Tensor, torch.Tensor, dict[str, Any]]:
    """Collect final embeddings and an early global-attention feature map."""

    if _is_tiny_vit_encoder(image_encoder):
        return _collect_tiny_vit_features(image_encoder, images, selected_intermediate_index)

    features = image_encoder.patch_embed(images)
    positional_embedding = getattr(image_encoder, "pos_embed", None)
    if positional_embedding is not None:
        features = features + positional_embedding

    intermediate_features = []
    intermediate_blocks = []
    for block_index, block in enumerate(image_encoder.blocks):
        features = block(features)
        if int(getattr(block, "window_size", 0) or 0) == 0:
            intermediate_features.append(features)
            intermediate_blocks.append(block_index)

    if not intermediate_features:
        intermediate_features = [features]
        intermediate_blocks = [-1]
    selected = max(0, min(int(selected_intermediate_index), len(intermediate_features) - 1))
    early = _as_nchw(intermediate_features[selected], _infer_early_dim(image_encoder))
    final = image_encoder.neck(features.permute(0, 3, 1, 2).contiguous())
    info = {
        "encoder_family": "sam_vit",
        "selected_intermediate_index": selected,
        "selected_block_index": int(intermediate_blocks[selected]),
        "num_intermediate_features": len(intermediate_features),
        "early_shape": list(early.shape),
        "final_shape": list(final.shape),
    }
    return final, early, info


class HQResidualHead(nn.Module):
    """Fuse early and final SAM features into a zero-initialized residual."""

    def __init__(
        self,
        *,
        final_dim: int,
        early_dim: int,
        hidden_dim: int = 256,
        out_channels: int = OUT_CHANNELS,
    ) -> None:
        super().__init__()
        self.early_projection = nn.Conv2d(int(early_dim), int(final_dim), kernel_size=1)
        self.fusion = nn.Sequential(
            nn.Conv2d(int(final_dim) * 2, int(hidden_dim), kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(int(hidden_dim), int(hidden_dim), kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(int(hidden_dim), int(out_channels), kernel_size=1),
        )
        nn.init.zeros_(self.fusion[-1].weight)
        nn.init.zeros_(self.fusion[-1].bias)

    def forward(self, final_embedding: torch.Tensor, early_feature: torch.Tensor) -> torch.Tensor:
        if early_feature.shape[-2:] != final_embedding.shape[-2:]:
            early_feature = F.interpolate(
                early_feature,
                final_embedding.shape[-2:],
                mode="bilinear",
                align_corners=False,
            )
        early = self.early_projection(early_feature)
        return self.fusion(torch.cat((final_embedding, early), dim=1))


def _strip_state_prefixes(state_dict: Mapping[str, Any]) -> OrderedDict[str, Any]:
    output: OrderedDict[str, Any] = OrderedDict()
    for key, value in state_dict.items():
        normalized = str(key)
        for prefix in ("_orig_mod.", "module."):
            while normalized.startswith(prefix):
                normalized = normalized[len(prefix) :]
        output[normalized] = value
    return output


def _base_unetr_state(decoder_state: Mapping[str, Any] | None) -> OrderedDict[str, Any] | None:
    if not decoder_state:
        return None
    stripped = _strip_state_prefixes(decoder_state)
    if not any(key.startswith(("baseline_decoder.", "hq_residual.")) for key in stripped):
        return OrderedDict((key, value) for key, value in stripped.items() if not key.startswith("encoder."))
    prefix = "baseline_decoder."
    return OrderedDict(
        (key[len(prefix) :], value)
        for key, value in stripped.items()
        if key.startswith(prefix)
    ) or None


class UniSEMHQUNETR(nn.Module):
    """Prompt-free SAM encoder with UNETR output and HQ residual fusion."""

    def __init__(
        self,
        *,
        image_encoder: nn.Module,
        baseline_decoder: nn.Module,
        selected_intermediate_index: int = 0,
        hidden_dim: int = 256,
        pixel_mean: tuple[float, float, float] = SAM_PIXEL_MEAN,
        pixel_std: tuple[float, float, float] = SAM_PIXEL_STD,
    ) -> None:
        super().__init__()
        self.encoder = image_encoder
        self.baseline_decoder = baseline_decoder
        self.hq_residual = HQResidualHead(
            final_dim=_infer_final_dim(image_encoder),
            early_dim=_infer_early_dim(image_encoder),
            hidden_dim=int(hidden_dim),
            out_channels=OUT_CHANNELS,
        )
        self.selected_intermediate_index = int(selected_intermediate_index)
        self.hidden_dim = int(hidden_dim)
        self.register_buffer(
            "pixel_mean",
            torch.tensor(pixel_mean, dtype=torch.float32).view(1, 3, 1, 1),
            persistent=False,
        )
        self.register_buffer(
            "pixel_std",
            torch.tensor(pixel_std, dtype=torch.float32).view(1, 3, 1, 1),
            persistent=False,
        )
        self.last_feature_info: dict[str, Any] = {}

    @staticmethod
    def fuse_logits(baseline: torch.Tensor, residual_logits: torch.Tensor) -> torch.Tensor:
        """Add the HQ residual in logit space and return probabilities."""

        if residual_logits.shape[-2:] != baseline.shape[-2:]:
            residual_logits = F.interpolate(
                residual_logits,
                baseline.shape[-2:],
                mode="bilinear",
                align_corners=False,
            )
        epsilon = torch.finfo(baseline.dtype).eps
        baseline_logits = torch.logit(baseline.clamp(float(epsilon), 1.0 - float(epsilon)))
        return torch.sigmoid(baseline_logits + residual_logits)

    def preprocess(self, images: torch.Tensor) -> tuple[torch.Tensor, tuple[int, int], tuple[int, int]]:
        if images.ndim != 4:
            raise ValueError(f"Expected BCHW input, got {tuple(images.shape)}")
        if int(images.shape[1]) not in (1, 3):
            raise ValueError(f"Expected one or three input channels, got {int(images.shape[1])}")
        original_shape = (int(images.shape[-2]), int(images.shape[-1]))
        image_size = int(getattr(self.encoder, "img_size", 1024))
        scale = float(image_size) / float(max(original_shape))
        input_shape = (
            int(original_shape[0] * scale + 0.5),
            int(original_shape[1] * scale + 0.5),
        )
        if tuple(images.shape[-2:]) != input_shape:
            images = F.interpolate(images, input_shape, mode="bilinear", align_corners=False, antialias=True)
        if int(images.shape[1]) == 1:
            images = images.expand(-1, 3, -1, -1)
        mean = self.pixel_mean.to(device=images.device, dtype=images.dtype)
        std = self.pixel_std.to(device=images.device, dtype=images.dtype)
        images = (images - mean) / std
        pad_height = image_size - int(images.shape[-2])
        pad_width = image_size - int(images.shape[-1])
        if pad_height < 0 or pad_width < 0:
            raise ValueError(f"Input exceeds SAM image size {image_size}: {tuple(images.shape)}")
        images = F.pad(images, (0, pad_width, 0, pad_height))
        return images, input_shape, original_shape

    def postprocess(
        self,
        predictions: torch.Tensor,
        input_shape: tuple[int, int],
        original_shape: tuple[int, int],
    ) -> torch.Tensor:
        image_size = int(getattr(self.encoder, "img_size", 1024))
        predictions = F.interpolate(predictions, (image_size, image_size), mode="bilinear", align_corners=False)
        predictions = predictions[..., : input_shape[0], : input_shape[1]]
        return F.interpolate(predictions, original_shape, mode="bilinear", align_corners=False)

    def _decode_baseline(self, final_embedding: torch.Tensor) -> torch.Tensor:
        implementation = getattr(self.baseline_decoder, "_forward_impl", None)
        if implementation is not None:
            return implementation(final_embedding)
        return self.baseline_decoder(final_embedding)

    def forward_preprocessed(
        self,
        images: torch.Tensor,
        input_shape: tuple[int, int],
        original_shape: tuple[int, int],
    ) -> torch.Tensor:
        final, early, info = collect_sam_features(
            self.encoder,
            images,
            selected_intermediate_index=self.selected_intermediate_index,
        )
        self.last_feature_info = info
        baseline = self._decode_baseline(final)
        if int(baseline.shape[1]) != OUT_CHANNELS:
            raise RuntimeError(f"UNETR must return {OUT_CHANNELS} channels, got {tuple(baseline.shape)}")
        residual = self.hq_residual(final, early)
        return self.postprocess(self.fuse_logits(baseline, residual), input_shape, original_shape)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        images, input_shape, original_shape = self.preprocess(images)
        return self.forward_preprocessed(images, input_shape, original_shape)

    def decoder_state_dict(self) -> OrderedDict[str, torch.Tensor]:
        return OrderedDict(
            (key, value)
            for key, value in self.state_dict().items()
            if not key.startswith("encoder.")
        )

    def load_decoder_state(self, decoder_state: Mapping[str, Any], *, strict: bool = True) -> None:
        stripped = _strip_state_prefixes(decoder_state)
        if not any(key.startswith(("baseline_decoder.", "hq_residual.")) for key in stripped):
            return
        current = self.state_dict()
        loadable = OrderedDict()
        mismatched = []
        for key, value in stripped.items():
            if key.startswith("encoder."):
                continue
            if key in current and hasattr(value, "shape") and tuple(value.shape) == tuple(current[key].shape):
                loadable[key] = value
            elif key in current:
                mismatched.append(key)
        missing = [
            key
            for key in current
            if not key.startswith("encoder.") and key not in loadable
        ]
        if strict and (missing or mismatched):
            raise RuntimeError(
                f"Incompatible decoder state: missing={missing[:10]}, mismatched={mismatched[:10]}"
            )
        self.load_state_dict(loadable, strict=False)


def apply_freeze_policy(model: nn.Module, policy: str) -> dict[str, Any]:
    """Apply Stage 1, LightFT, or FullFT trainability."""

    normalized = str(policy).lower()
    if normalized not in FREEZE_POLICIES:
        raise ValueError(f"Unknown freeze policy {policy!r}; expected one of {FREEZE_POLICIES}")
    core = _unwrap(model)
    required = ("encoder", "baseline_decoder", "hq_residual")
    if any(not hasattr(core, name) for name in required):
        raise TypeError("Freeze policies require encoder, baseline_decoder, and hq_residual modules")

    if normalized == "lightft":
        from .lora import LoRAQVConfig, inject_lora_qv

        config = LoRAQVConfig(**getattr(core.encoder, "_unisem_lora_config", {}))
        inject_lora_qv(core.encoder, config)
    else:
        if normalized in {"stage1", "fullft"} and any(".lora_" in name for name, _ in core.encoder.named_parameters()):
            raise ValueError("Stage 1 and FullFT require a model without injected LoRA branches")
        _set_trainable(core.encoder, normalized != "head_only")
    _set_trainable(core.baseline_decoder, True)
    _set_trainable(core.hq_residual, True)
    groups = {
        name: parameter_summary(getattr(core, name))
        for name in required
    }
    if normalized == "head_only" and groups["encoder"]["trainable"] != 0:
        raise RuntimeError("Head-only adaptation must freeze all encoder parameters")
    if normalized in {"stage1", "fullft"} and groups["encoder"]["trainable"] == 0:
        raise RuntimeError(f"{normalized} requires trainable encoder parameters")
    return {"policy": normalized, "groups": groups, "model": parameter_summary(core)}


def build_model(
    image_encoder: nn.Module,
    *,
    decoder_state: Mapping[str, Any] | None = None,
    selected_intermediate_index: int = 0,
    hidden_dim: int = 256,
    device: str | torch.device | None = None,
    flexible_load: bool = False,
    freeze_policy: str = "stage1",
) -> UniSEMHQUNETR:
    """Build UniSEM around an initialized SAM image encoder."""

    from micro_sam.instance_segmentation import DecoderAdapter, get_unetr

    unetr = get_unetr(
        image_encoder=image_encoder,
        decoder_state=_base_unetr_state(decoder_state),
        device=device,
        out_channels=OUT_CHANNELS,
        flexible_load_checkpoint=bool(flexible_load),
        final_activation="Sigmoid",
    )
    model = UniSEMHQUNETR(
        image_encoder=image_encoder,
        baseline_decoder=DecoderAdapter(unetr),
        selected_intermediate_index=int(selected_intermediate_index),
        hidden_dim=int(hidden_dim),
    )
    if decoder_state:
        model.load_decoder_state(decoder_state, strict=not bool(flexible_load))
    if device is not None:
        model.to(device)
    apply_freeze_policy(model, freeze_policy)
    return model


def _load_sam_image_encoder(model_type: str, checkpoint_path: str | PathLike[str] | None = None) -> nn.Module:
    """Build the official SAM architecture without hidden downloads or CUDA loading."""
    from segment_anything import sam_model_registry

    if model_type not in sam_model_registry:
        raise ValueError(f"Unsupported SAM architecture: {model_type}")
    sam = sam_model_registry[model_type](checkpoint=None)
    if checkpoint_path is not None:
        state = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
        sam.load_state_dict(state, strict=True)
    return sam.image_encoder


def load_model(
    checkpoint: Mapping[str, Any] | str | PathLike[str],
    *,
    image_encoder: nn.Module | None = None,
    model_type: str | None = None,
    sam_checkpoint_path: str | PathLike[str] | None = None,
    device: str | torch.device | None = None,
    strict: bool = True,
    freeze_policy: str | None = None,
    base_checkpoint_path: str | PathLike[str] | None = None,
) -> UniSEMHQUNETR:
    """Load a clean or legacy checkpoint into a UniSEM model."""

    from .checkpoints import (
        ADAPTER_SCHEMA_VERSION, INFERENCE_SCHEMA_VERSION, load_checkpoint,
        load_inference_checkpoint, reconstruct_adapter, sanitize_checkpoint,
    )

    payload = load_checkpoint(checkpoint)
    if payload.get("schema_version") == ADAPTER_SCHEMA_VERSION:
        if base_checkpoint_path is None:
            raise ValueError("An adapter requires its exact --base-checkpoint")
        payload = reconstruct_adapter(payload, base_checkpoint_path)
    if payload.get("schema_version") != INFERENCE_SCHEMA_VERSION:
        payload = sanitize_checkpoint(payload, model_type=model_type)
    payload = load_inference_checkpoint(payload)
    architecture = payload["architecture"]
    resolved_model_type = str(model_type or architecture["model_type"])

    if image_encoder is None:
        image_encoder = _load_sam_image_encoder(resolved_model_type, sam_checkpoint_path)

    adaptation = architecture.get("adaptation", {"mode": "none"})
    if adaptation["mode"] == "lora_qv":
        from .lora import LoRAQVConfig, inject_lora_qv

        inject_lora_qv(image_encoder, LoRAQVConfig(
            rank=int(adaptation["rank"]), alpha=float(adaptation["alpha"]),
            dropout=float(adaptation["dropout"]),
        ))
    image_encoder.load_state_dict(payload["encoder_state"], strict=bool(strict))
    model = build_model(
        image_encoder,
        decoder_state=payload["decoder_state"],
        selected_intermediate_index=int(architecture["selected_intermediate_index"]),
        hidden_dim=int(architecture["hidden_dim"]),
        device=device,
        flexible_load=not bool(strict),
        freeze_policy=freeze_policy or ("lightft" if adaptation["mode"] == "lora_qv" else "fullft"),
    )
    return model


__all__ = [
    "CHANNEL_LAYOUT",
    "FREEZE_POLICIES",
    "HQResidualHead",
    "UniSEMHQUNETR",
    "apply_freeze_policy",
    "build_model",
    "collect_sam_features",
    "load_model",
    "parameter_summary",
]
