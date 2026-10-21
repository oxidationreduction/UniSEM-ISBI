from __future__ import annotations

import argparse
import json

import numpy as np


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", action="append", required=True)
    parser.add_argument("--rag-checkpoint", action="append", required=True)
    parser.add_argument("--raw", required=True)
    parser.add_argument("--raw-key", default="main")
    parser.add_argument("--slice", type=int, default=0)
    parser.add_argument("--crop", type=int, nargs=4, default=(0, 256, 0, 256))
    args = parser.parse_args()
    if len(args.checkpoint) != len(args.rag_checkpoint):
        raise ValueError("Each dense checkpoint requires one matching RAG checkpoint")

    import h5py
    import torch
    from segment_anything import sam_model_registry

    from unisem.inference import CHANNEL_NAMES, overlap_fusion, seeded_watershed
    from unisem.model import load_model
    from unisem.rag import extract_edge_features, load_random_forest, merge_fragments

    with h5py.File(args.raw, "r") as handle:
        y0, y1, x0, x1 = args.crop
        raw = np.asarray(handle[args.raw_key][args.slice, y0:y1, x0:x1])
    device = torch.device("cuda:0")
    scales = {
        "tile512_overlap": ((512, 512), (384, 384)),
        "tile384_overlap": ((384, 384), (256, 256)),
        "tile256_overlap": ((256, 256), (192, 192)),
    }
    results = []
    for checkpoint, rag_checkpoint in zip(args.checkpoint, args.rag_checkpoint):
        encoder = sam_model_registry["vit_l"](checkpoint=None).image_encoder
        model = load_model(
            checkpoint,
            image_encoder=encoder,
            model_type="vit_l",
            device=device,
            strict=True,
            freeze_policy="fullft",
        ).eval()

        def predict(tile: np.ndarray) -> np.ndarray:
            tensor = torch.from_numpy(tile.astype(np.float32, copy=False)[None, None]).to(device)
            with torch.inference_mode(), torch.autocast("cuda", dtype=torch.float16):
                return model(tensor)[0].float().cpu().numpy()

        states = {}
        segmentations = {}
        for scale, (tile_shape, stride) in scales.items():
            prediction = overlap_fusion(raw, predict, tile_shape=tile_shape, stride=stride)
            states[scale] = {name: prediction[index] for index, name in enumerate(CHANNEL_NAMES)}
            segmentations[scale] = seeded_watershed(prediction)
        fragments = segmentations["tile384_overlap"]
        batch = extract_edge_features(fragments, states, segmentations, raw.astype(np.float32, copy=False))
        rf = load_random_forest(rag_checkpoint)
        merged = merge_fragments(fragments, batch.edges, batch.features, rf)
        results.append(
            {
                "checkpoint_schema": "legacy_or_public_loaded",
                "crop_shape": list(raw.shape),
                "candidate_edges": int(len(batch.edges)),
                "fragments": int(fragments.max()),
                "merged_instances": int(merged.max()),
                "finite_dense_maps": bool(
                    all(np.isfinite(values).all() for state in states.values() for values in state.values())
                ),
            }
        )
        del model, encoder
        torch.cuda.empty_cache()
    print(json.dumps({"ok": True, "results": results}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
