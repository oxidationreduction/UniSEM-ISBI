"""Minimal strict model loading and single-image dense prediction example."""
from __future__ import annotations

import argparse
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--base-checkpoint")
    parser.add_argument("--input", type=Path, required=True, help="Grayscale 2D TIFF")
    parser.add_argument("--output", type=Path, required=True, help="Three-channel dense TIFF")
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    import numpy as np
    import tifffile
    import torch
    from unisem import load_model
    from unisem.inference import overlap_fusion

    raw = tifffile.imread(args.input)
    if raw.ndim != 2:
        parser.error("This minimal example accepts one grayscale 2D image; use unisem infer for stacks.")
    if not np.isfinite(raw).all():
        parser.error("The input contains nonfinite intensities.")
    values = raw.astype(np.float32, copy=False)
    low, high = np.percentile(values, [1, 99])
    normalized = np.zeros_like(values) if high <= low else np.clip((values - low) / (high - low), 0, 1)
    device = torch.device(args.device)
    model = load_model(args.checkpoint, base_checkpoint_path=args.base_checkpoint, device=device).eval()

    def predict(tile):
        tensor = torch.from_numpy(tile[None, None]).to(device)
        with torch.inference_mode():
            return model(tensor)[0].float().cpu().numpy()

    dense = overlap_fusion(normalized, predict, tile_shape=(384, 384), stride=(256, 256))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    tifffile.imwrite(args.output, dense, metadata={"axes": "CYX"}, photometric="minisblack")
    print(f"Saved F/C/B dense predictions: {args.output}")
    print("For the complete multiscale RF instance pipeline, use unisem infer.")


if __name__ == "__main__":
    main()
