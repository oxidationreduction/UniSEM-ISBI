"""UniSEM public API; heavy model dependencies load only when requested."""

from importlib import import_module

__version__ = "0.2.0"
_API_MODULES = {
    "model": ("CHANNEL_LAYOUT", "FREEZE_POLICIES", "HQResidualHead", "UniSEMHQUNETR",
              "apply_freeze_policy", "build_model", "collect_sam_features", "load_model", "parameter_summary"),
    "checkpoints": ("ADAPTER_SCHEMA_VERSION", "INFERENCE_SCHEMA_VERSION", "inference_checkpoint",
                    "inspect_checkpoint", "load_inference_checkpoint", "sanitize_checkpoint",
                    "sanitize_checkpoint_file", "scan_forbidden_strings", "make_adapter", "reconstruct_adapter"),
    "lora": ("LoRAQVConfig", "LoRAQKVLinear", "inject_lora_qv"),
}
__all__ = [name for names in _API_MODULES.values() for name in names]


def __getattr__(name):
    for module, names in _API_MODULES.items():
        if name in names:
            value = getattr(import_module(f".{module}", __name__), name)
            globals()[name] = value
            return value
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
