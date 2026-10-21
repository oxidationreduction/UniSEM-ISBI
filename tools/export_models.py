"""Export trusted training checkpoints into path-free CPU inference assets.

Run in the installed UniSEM environment. Source files are never modified.
Use only checkpoints and joblib files whose provenance you trust.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _equal_states(actual, expected) -> None:
    import torch

    if set(actual) != set(expected):
        raise ValueError("Export changed the tensor-key set")
    for key, tensor in expected.items():
        if (actual[key].dtype != tensor.dtype or actual[key].shape != tensor.shape
                or not torch.equal(actual[key], tensor)):
            raise ValueError(f"Tensor mismatch: {key}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--kind", choices=("stage1", "lightft", "fullft", "rf"), required=True)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--source-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--base", type=Path, help="Sanitized first-stage checkpoint for LightFT")
    args = parser.parse_args()
    if args.source.resolve() == args.output.resolve():
        parser.error("Output must differ from source")
    if args.output.exists():
        parser.error("Output already exists; choose a new output file")
    source_sha = sha256(args.source)
    if source_sha != args.source_sha256.lower():
        raise ValueError("Source SHA256 mismatch")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    report = {"kind": args.kind, "source_sha256": source_sha, "file": args.output.name}
    if args.kind == "rf":
        shutil.copyfile(args.source, args.output)
        if sha256(args.output) != source_sha:
            raise ValueError("RF byte-for-byte verification failed")
        report["byte_parity"] = True
    else:
        import torch
        from unisem.checkpoints import (
            load_checkpoint, make_adapter, reconstruct_adapter, sanitize_checkpoint,
            scan_forbidden_strings,
        )

        torch.set_num_threads(4)
        original = load_checkpoint(args.source)
        clean = sanitize_checkpoint(original, model_type="vit_l")
        clean["source_sha256"] = source_sha
        _equal_states(clean["decoder_state"], original["decoder_state"])
        expected_encoder = {
            key[len("image_encoder."):]: value
            for key, value in original["model_state"].items()
            if key.startswith("image_encoder.")
        }
        _equal_states(clean["encoder_state"], expected_encoder)
        if args.kind == "lightft":
            if args.base is None:
                parser.error("LightFT export requires --base")
            base = load_checkpoint(args.base)
            exported = make_adapter(clean, base, base_sha256=sha256(args.base))
            exported["source_sha256"] = source_sha
            restored = reconstruct_adapter(exported, args.base)
            for state_name in ("encoder_state", "decoder_state"):
                _equal_states(restored[state_name], clean[state_name])
            report["base_sha256"] = sha256(args.base)
            report["lora_tensors"] = len(exported["lora_state"])
            report["state_elements_including_buffers"] = sum(
                t.numel() for t in exported["lora_state"].values()
            ) + sum(t.numel() for t in exported["decoder_state"].values())
            report["updated_parameters"] = sum(
                t.numel() for t in exported["lora_state"].values()
            ) + sum(
                tensor.numel() for name, tensor in exported["decoder_state"].items()
                if not name.endswith(("running_mean", "running_var", "num_batches_tracked"))
            )
        else:
            exported = clean
        if scan_forbidden_strings(exported):
            raise ValueError("Private metadata survived export")
        torch.save(exported, args.output)
        reloaded = torch.load(args.output, map_location="cpu", weights_only=True)
        if args.kind == "lightft":
            restored = reconstruct_adapter(reloaded, args.base)
        else:
            restored = reloaded
        for state_name in ("encoder_state", "decoder_state"):
            _equal_states(restored[state_name], clean[state_name])
        report.update(
            schema_version=exported["schema_version"],
            encoder_tensors=len(clean["encoder_state"]),
            decoder_tensors=len(clean["decoder_state"]),
            tensor_parity=True,
            weights_only_load=True,
            private_metadata_hits=0,
        )
    report["sha256"] = sha256(args.output)
    report["bytes"] = args.output.stat().st_size
    args.output.with_suffix(args.output.suffix + ".json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
