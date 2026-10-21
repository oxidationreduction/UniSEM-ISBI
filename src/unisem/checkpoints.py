"""Inspection and sanitization helpers for public UniSEM checkpoints."""

from __future__ import annotations

from collections import OrderedDict
from collections.abc import Mapping, Sequence
from os import PathLike
from pathlib import Path
from typing import Any
import hashlib
import re

import torch

INFERENCE_SCHEMA_VERSION = "unisem.inference.v2"
ADAPTER_SCHEMA_VERSION = "unisem.adapter.v1"
_PRIVATE_ROOT_MARKERS = tuple(f"/{name}/" for name in ("home", "mnt", "opt"))
DEFAULT_FORBIDDEN_STRINGS = (
    "run_root",
    "tmux",
    *_PRIVATE_ROOT_MARKERS,
    "\\users\\",
)


def _mapping(value: Any, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{name} must be a mapping, got {type(value).__name__}")
    return value


def _strip_prefixes(state_dict: Mapping[str, Any]) -> OrderedDict[str, Any]:
    output: OrderedDict[str, Any] = OrderedDict()
    for key, value in state_dict.items():
        normalized = str(key)
        for prefix in ("_orig_mod.", "module."):
            while normalized.startswith(prefix):
                normalized = normalized[len(prefix) :]
        output[normalized] = value
    return output


def _cpu_tensor_state(state_dict: Mapping[str, Any], name: str) -> OrderedDict[str, torch.Tensor]:
    output: OrderedDict[str, torch.Tensor] = OrderedDict()
    for key, value in state_dict.items():
        if not isinstance(value, torch.Tensor):
            continue
        output[str(key)] = value.detach().cpu()
    if not output:
        raise ValueError(f"{name} does not contain tensor parameters")
    return output


def _checkpoint_mapping(checkpoint: Any) -> Mapping[str, Any]:
    if isinstance(checkpoint, Mapping):
        return checkpoint
    raise TypeError(f"Checkpoint must be a mapping, got {type(checkpoint).__name__}")


def load_checkpoint(source: Mapping[str, Any] | str | PathLike[str]) -> dict[str, Any]:
    """Load a checkpoint mapping on CPU without changing it."""

    if isinstance(source, Mapping):
        return dict(source)
    payload = torch.load(Path(source), map_location="cpu", weights_only=True)
    return dict(_checkpoint_mapping(payload))


def _walk_strings(
    value: Any,
    *,
    path: str,
    seen: set[int],
) -> list[tuple[str, str]]:
    if isinstance(value, str):
        return [(path, value)]
    if isinstance(value, bytes):
        return [(path, value.decode("utf-8", errors="replace"))]
    if isinstance(value, PathLike):
        return [(path, str(value))]
    if isinstance(value, torch.Tensor) or value is None or isinstance(value, (int, float, bool)):
        return []

    identity = id(value)
    if identity in seen:
        return []
    seen.add(identity)
    strings: list[tuple[str, str]] = []
    if isinstance(value, Mapping):
        for key, item in value.items():
            key_text = str(key)
            strings.append((f"{path}.<key>", key_text))
            strings.extend(_walk_strings(item, path=f"{path}.{key_text}", seen=seen))
    elif isinstance(value, Sequence):
        for index, item in enumerate(value):
            strings.extend(_walk_strings(item, path=f"{path}[{index}]", seen=seen))
    return strings


def scan_forbidden_strings(
    value: Any,
    *,
    forbidden_strings: Sequence[str] = DEFAULT_FORBIDDEN_STRINGS,
) -> list[dict[str, str]]:
    """Recursively locate private path, route, and orchestration strings."""

    tokens = tuple(str(token).lower() for token in forbidden_strings if str(token))
    hits = []
    for path, text in _walk_strings(value, path="$", seen=set()):
        lowered = text.lower()
        for original, token in zip(forbidden_strings, tokens):
            if token in lowered:
                hits.append({"path": path, "forbidden": str(original)})
    return hits


def _tensor_summary(state_dict: Mapping[str, Any]) -> dict[str, Any]:
    tensors = [value for value in state_dict.values() if isinstance(value, torch.Tensor)]
    return {
        "keys": len(state_dict),
        "tensors": len(tensors),
        "elements": sum(int(tensor.numel()) for tensor in tensors),
        "dtypes": sorted({str(tensor.dtype).replace("torch.", "") for tensor in tensors}),
    }


def _find_model_type(checkpoint: Mapping[str, Any], explicit: str | None) -> str:
    if explicit:
        return str(explicit)
    architecture = checkpoint.get("architecture")
    if isinstance(architecture, Mapping) and architecture.get("model_type"):
        return str(architecture["model_type"])
    if checkpoint.get("model_type"):
        return str(checkpoint["model_type"])
    train_config = checkpoint.get("train_config")
    if isinstance(train_config, Mapping) and train_config.get("model_type"):
        return str(train_config["model_type"])
    raise ValueError("model_type is required because it is not present in the checkpoint")


def _extract_encoder_state(checkpoint: Mapping[str, Any]) -> OrderedDict[str, torch.Tensor]:
    if isinstance(checkpoint.get("encoder_state"), Mapping):
        raw = _strip_prefixes(_mapping(checkpoint["encoder_state"], "encoder_state"))
        normalized = OrderedDict()
        for key, value in raw.items():
            for prefix in ("encoder.", "image_encoder.", "sam.image_encoder.", "model.image_encoder."):
                if key.startswith(prefix):
                    key = key[len(prefix) :]
                    break
            normalized[key] = value
        return _cpu_tensor_state(normalized, "encoder_state")

    source = checkpoint.get("model_state", checkpoint.get("model_state_dict", checkpoint.get("state_dict")))
    if source is None and all(isinstance(value, torch.Tensor) for value in checkpoint.values()):
        source = checkpoint
    raw = _strip_prefixes(_mapping(source, "model_state"))
    prefixes = ("image_encoder.", "sam.image_encoder.", "model.image_encoder.", "encoder.")
    encoder = OrderedDict()
    for key, value in raw.items():
        for prefix in prefixes:
            if key.startswith(prefix):
                encoder[key[len(prefix) :]] = value
                break
    return _cpu_tensor_state(encoder, "encoder_state")


def _extract_decoder_state(checkpoint: Mapping[str, Any]) -> OrderedDict[str, torch.Tensor]:
    source = checkpoint.get("decoder_state")
    if source is None:
        source = checkpoint.get("state_dict")
    if source is None and all(isinstance(value, torch.Tensor) for value in checkpoint.values()):
        source = checkpoint
    raw = _strip_prefixes(_mapping(source, "decoder_state"))
    decoder = OrderedDict()
    for key, value in raw.items():
        if key.startswith("encoder."):
            continue
        if key.startswith(("baseline_decoder.", "hq_residual.")):
            decoder[key] = value
        elif not any(name.startswith(("baseline_decoder.", "hq_residual.")) for name in raw):
            decoder[key] = value
    output = _cpu_tensor_state(decoder, "decoder_state")
    if not any(key.startswith("baseline_decoder.") for key in output):
        raise ValueError("decoder_state is missing baseline_decoder parameters")
    if not any(key.startswith("hq_residual.") for key in output):
        raise ValueError("decoder_state is missing hq_residual parameters")
    return output


def _validate_architecture(architecture: Mapping[str, Any]) -> None:
    if str(architecture.get("name")) != "UniSEM-HQ-UNETR":
        raise ValueError("Unsupported checkpoint architecture")
    if int(architecture.get("out_channels", -1)) != 3:
        raise ValueError("UniSEM inference checkpoints require three output channels")
    expected_layout = {
        "foreground": 0,
        "center_distances": 1,
        "boundary_distances": 2,
    }
    if dict(_mapping(architecture.get("channel_layout"), "channel_layout")) != expected_layout:
        raise ValueError("Unexpected UniSEM channel layout")
    adaptation = architecture.get("adaptation", {"mode": "none"})
    if adaptation.get("mode") not in {"none", "lora_qv"}:
        raise ValueError("Unsupported encoder adaptation")
    if adaptation["mode"] == "lora_qv":
        if int(adaptation.get("rank", 0)) < 1 or float(adaptation.get("alpha", 0)) <= 0:
            raise ValueError("Invalid LoRA rank or alpha")
        if adaptation.get("target_matrices") != ["q", "v"]:
            raise ValueError("Only Q/V LoRA is supported")


def _adaptation_from_state(state: Mapping[str, torch.Tensor]) -> dict[str, Any]:
    keys = [key for key in state if ".lora_" in key]
    if not keys:
        return {"mode": "none"}
    matches = [re.search(r"blocks\.(\d+)\.", key) for key in keys]
    if any(match is None for match in matches):
        raise ValueError("LoRA tensor key does not identify an encoder block")
    blocks = sorted({int(match.group(1)) for match in matches})
    expected = {
        f"blocks.{block}.attn.qkv.lora_{matrix}_{direction}.weight"
        for block in blocks for matrix in ("q", "v") for direction in ("down", "up")
    }
    if set(keys) != expected:
        raise ValueError("Incomplete or unexpected Q/V LoRA tensor set")
    ranks = {int(state[key].shape[0]) for key in keys if "_down." in key}
    if len(ranks) != 1:
        raise ValueError("Inconsistent LoRA ranks")
    rank = ranks.pop()
    return {"mode": "lora_qv", "rank": rank, "alpha": float(rank), "dropout": 0.0,
            "target_blocks": blocks, "target_matrices": ["q", "v"]}


def inspect_checkpoint(
    checkpoint: Mapping[str, Any] | str | PathLike[str],
    *,
    forbidden_strings: Sequence[str] = DEFAULT_FORBIDDEN_STRINGS,
) -> dict[str, Any]:
    """Return a non-mutating structural and privacy summary."""

    payload = load_checkpoint(checkpoint)
    report: dict[str, Any] = {
        "schema_version": payload.get("schema_version"),
        "top_level_keys": sorted(str(key) for key in payload),
        "forbidden_string_hits": scan_forbidden_strings(payload, forbidden_strings=forbidden_strings),
    }
    if "architecture" in payload:
        report["architecture"] = payload["architecture"]
    if "base_sha256" in payload:
        report["base_sha256"] = payload["base_sha256"]
    for name in ("encoder_state", "decoder_state", "lora_state", "model_state", "model_state_dict", "state_dict"):
        state = payload.get(name)
        if isinstance(state, Mapping):
            report[name] = _tensor_summary(state)
    return report


def sanitize_checkpoint(
    checkpoint: Mapping[str, Any] | str | PathLike[str],
    *,
    model_type: str | None = None,
    selected_intermediate_index: int = 0,
    hidden_dim: int = 256,
    forbidden_strings: Sequence[str] = DEFAULT_FORBIDDEN_STRINGS,
) -> dict[str, Any]:
    """Convert a training checkpoint into the public inference schema."""

    payload = load_checkpoint(checkpoint)
    architecture = payload.get("architecture")
    architecture = architecture if isinstance(architecture, Mapping) else {}
    clean = {
        "schema_version": INFERENCE_SCHEMA_VERSION,
        "architecture": {
            "name": "UniSEM-HQ-UNETR",
            "model_type": _find_model_type(payload, model_type),
            "out_channels": 3,
            "channel_layout": {
                "foreground": 0,
                "center_distances": 1,
                "boundary_distances": 2,
            },
            "selected_intermediate_index": int(
                architecture.get("selected_intermediate_index", selected_intermediate_index)
            ),
            "hidden_dim": int(architecture.get("hidden_dim", hidden_dim)),
            "fusion": "additive_logit_residual",
            "residual_output_zero_initialized": True,
            "adaptation": dict(architecture.get("adaptation", _adaptation_from_state(_extract_encoder_state(payload)))),
        },
        "encoder_state": _extract_encoder_state(payload),
        "decoder_state": _extract_decoder_state(payload),
    }
    hits = scan_forbidden_strings(clean, forbidden_strings=forbidden_strings)
    if hits:
        raise ValueError(f"Sanitized checkpoint still contains forbidden strings: {hits[:10]}")
    return clean


def load_inference_checkpoint(
    checkpoint: Mapping[str, Any] | str | PathLike[str],
    *,
    forbidden_strings: Sequence[str] = DEFAULT_FORBIDDEN_STRINGS,
) -> dict[str, Any]:
    """Load and validate a clean public inference checkpoint."""

    payload = load_checkpoint(checkpoint)
    if payload.get("schema_version") != INFERENCE_SCHEMA_VERSION:
        raise ValueError(f"Expected schema_version={INFERENCE_SCHEMA_VERSION!r}")
    architecture = _mapping(payload.get("architecture"), "architecture")
    _validate_architecture(architecture)
    encoder = _cpu_tensor_state(_mapping(payload.get("encoder_state"), "encoder_state"), "encoder_state")
    inferred = _adaptation_from_state(encoder)
    adaptation = architecture.get("adaptation", {"mode": "none"})
    if inferred["mode"] != adaptation["mode"]:
        raise ValueError("LoRA metadata and tensors disagree")
    if inferred["mode"] == "lora_qv":
        if inferred["rank"] != adaptation["rank"] or inferred["target_blocks"] != adaptation["target_blocks"]:
            raise ValueError("LoRA tensor coverage does not match architecture")
        if architecture["model_type"] == "vit_l" and inferred["target_blocks"] != list(range(24)):
            raise ValueError("The ISBI ViT-L model requires LoRA on all 24 encoder blocks")
    decoder_state = _cpu_tensor_state(
        _mapping(payload.get("decoder_state"), "decoder_state"),
        "decoder_state",
    )
    if not any(key.startswith("baseline_decoder.") for key in decoder_state):
        raise ValueError("decoder_state is missing baseline_decoder parameters")
    if not any(key.startswith("hq_residual.") for key in decoder_state):
        raise ValueError("decoder_state is missing hq_residual parameters")
    hits = scan_forbidden_strings(payload, forbidden_strings=forbidden_strings)
    if hits:
        raise ValueError(f"Inference checkpoint contains forbidden strings: {hits[:10]}")
    return payload


def sanitize_checkpoint_file(
    source: str | PathLike[str],
    destination: str | PathLike[str],
    **kwargs: Any,
) -> dict[str, Any]:
    """Sanitize and save a checkpoint, returning its inspection summary."""

    clean = sanitize_checkpoint(source, **kwargs)
    torch.save(clean, Path(destination))
    return inspect_checkpoint(clean)


def inference_checkpoint(
    model: Any,
    *,
    model_type: str,
    public_hyperparameters: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Export a live UniSEM model directly to the public inference schema."""

    core = model.module if hasattr(model, "module") else model
    architecture = {
        "name": "UniSEM-HQ-UNETR",
        "model_type": str(model_type),
        "out_channels": 3,
        "channel_layout": {
            "foreground": 0,
            "center_distances": 1,
            "boundary_distances": 2,
        },
        "selected_intermediate_index": int(core.selected_intermediate_index),
        "hidden_dim": int(core.hidden_dim),
        "fusion": "additive_logit_residual",
        "residual_output_zero_initialized": True,
        "adaptation": _adaptation_from_state(core.encoder.state_dict()),
    }
    clean = {
        "schema_version": INFERENCE_SCHEMA_VERSION,
        "architecture": architecture,
        "encoder_state": _cpu_tensor_state(core.encoder.state_dict(), "encoder_state"),
        "decoder_state": _cpu_tensor_state(core.decoder_state_dict(), "decoder_state"),
    }
    if public_hyperparameters:
        clean["public_hyperparameters"] = dict(public_hyperparameters)
    hits = scan_forbidden_strings(clean)
    if hits:
        raise ValueError(f"Exported checkpoint contains forbidden strings: {hits[:10]}")
    return clean


def file_sha256(path: str | PathLike[str]) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def make_adapter(
    checkpoint: Mapping[str, Any] | str | PathLike[str],
    base_checkpoint: Mapping[str, Any] | str | PathLike[str],
    *,
    base_sha256: str | None = None,
) -> dict[str, Any]:
    """Export full decoder/HQ state and Q/V branches relative to an exact base."""
    full = load_inference_checkpoint(checkpoint)
    base = load_inference_checkpoint(base_checkpoint)
    if full["architecture"]["adaptation"]["mode"] != "lora_qv":
        raise ValueError("Adapter export requires a LoRA checkpoint")
    if base["architecture"]["adaptation"]["mode"] != "none":
        raise ValueError("The source checkpoint must not already contain LoRA")
    if base_sha256 is None:
        if isinstance(base_checkpoint, Mapping):
            raise ValueError("A mapping base requires its serialized file SHA256")
        base_sha256 = file_sha256(base_checkpoint)
    if not re.fullmatch(r"[0-9a-f]{64}", base_sha256):
        raise ValueError("Expected an exact sanitized-base SHA256")
    frozen = {
        key.replace(".qkv.base.", ".qkv."): value
        for key, value in full["encoder_state"].items() if ".lora_" not in key
    }
    if set(frozen) != set(base["encoder_state"]) or any(
        not torch.equal(value, base["encoder_state"][key]) for key, value in frozen.items()
    ):
        raise ValueError("Frozen encoder parameters differ from the specified source base")
    return {
        "schema_version": ADAPTER_SCHEMA_VERSION,
        "architecture": dict(full["architecture"]),
        "base_sha256": base_sha256,
        "lora_state": OrderedDict((key, value) for key, value in full["encoder_state"].items() if ".lora_" in key),
        "decoder_state": full["decoder_state"],
    }


def reconstruct_adapter(
    adapter: Mapping[str, Any] | str | PathLike[str],
    base_checkpoint: str | PathLike[str],
) -> dict[str, Any]:
    """Rebuild an inference checkpoint after checking the exact public base file."""
    payload = load_checkpoint(adapter)
    if payload.get("schema_version") != ADAPTER_SCHEMA_VERSION:
        raise ValueError("Unsupported adapter schema")
    if file_sha256(base_checkpoint) != payload.get("base_sha256"):
        raise ValueError("Adapter base SHA256 mismatch")
    base = load_inference_checkpoint(base_checkpoint)
    architecture = _mapping(payload.get("architecture"), "architecture")
    _validate_architecture(architecture)
    if architecture.get("adaptation", {}).get("mode") != "lora_qv":
        raise ValueError("Expected Q/V LoRA adapter metadata")
    encoder = OrderedDict()
    blocks = set(architecture["adaptation"]["target_blocks"])
    for key, value in base["encoder_state"].items():
        match = re.match(r"blocks\.(\d+)\.attn\.qkv\.(weight|bias)$", key)
        if match and int(match.group(1)) in blocks:
            key = key.replace(".qkv.", ".qkv.base.")
        encoder[key] = value
    lora = _cpu_tensor_state(_mapping(payload.get("lora_state"), "lora_state"), "lora_state")
    if any(".lora_" not in key for key in lora):
        raise ValueError("Adapter contains unexpected frozen-encoder tensors")
    encoder.update(lora)
    clean = {"schema_version": INFERENCE_SCHEMA_VERSION, "architecture": dict(architecture),
             "encoder_state": encoder, "decoder_state": payload["decoder_state"]}
    return load_inference_checkpoint(clean)


__all__ = [
    "DEFAULT_FORBIDDEN_STRINGS",
    "ADAPTER_SCHEMA_VERSION",
    "INFERENCE_SCHEMA_VERSION",
    "inspect_checkpoint",
    "inference_checkpoint",
    "load_checkpoint",
    "load_inference_checkpoint",
    "sanitize_checkpoint",
    "sanitize_checkpoint_file",
    "scan_forbidden_strings",
    "make_adapter",
    "reconstruct_adapter",
    "file_sha256",
]
