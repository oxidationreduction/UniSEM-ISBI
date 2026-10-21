from __future__ import annotations

import argparse
import json
import os


def _model(checkpoint: str, device: object) -> object:
    from segment_anything import sam_model_registry

    from unisem.model import build_model

    sam = sam_model_registry["vit_b"](checkpoint=checkpoint)
    return build_model(sam.image_encoder, device=device, freeze_policy="stage1")


def _step(model: object, device: object, policy: str) -> dict[str, object]:
    import numpy as np
    import torch
    from torch_em.loss import DiceBasedDistanceLoss

    from unisem.data import make_three_channel_target
    from unisem.model import apply_freeze_policy

    yy, xx = np.mgrid[:256, :256]
    labels = np.zeros((256, 256), dtype=np.uint32)
    labels[(yy - 80) ** 2 + (xx - 80) ** 2 < 32**2] = 1
    labels[(yy - 176) ** 2 + (xx - 176) ** 2 < 36**2] = 2
    image = torch.from_numpy((labels > 0).astype(np.float32)[None, None] * 180.0).to(device)
    target = torch.from_numpy(make_three_channel_target(labels, min_size=1)[None]).to(device)
    summary = apply_freeze_policy(model, policy)
    optimizer = torch.optim.AdamW(
        [parameter for parameter in model.parameters() if parameter.requires_grad],
        lr=1.0e-5,
        weight_decay=1.0e-4,
    )
    optimizer.zero_grad(set_to_none=True)
    prediction = model(image)
    loss = DiceBasedDistanceLoss(mask_distances_in_bg=True)(prediction.float(), target.float())
    loss.backward()
    optimizer.step()
    if prediction.shape != target.shape or not torch.isfinite(loss):
        raise RuntimeError("GPU training step produced an invalid result")
    return {"policy": policy, "loss": float(loss.detach().cpu()), "trainability": summary}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sam-checkpoint", required=True)
    parser.add_argument("--ddp", action="store_true")
    args = parser.parse_args()

    import torch

    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    device = torch.device(f"cuda:{local_rank}")
    torch.cuda.set_device(device)
    if args.ddp:
        torch.distributed.init_process_group("nccl")
    model = _model(args.sam_checkpoint, device)
    if args.ddp:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[local_rank])
        results = [_step(model, device, "fullft")]
        torch.distributed.barrier()
        torch.distributed.destroy_process_group()
    else:
        results = [_step(model, device, policy) for policy in ("stage1", "fullft", "head_only", "lightft")]
    if local_rank == 0:
        print(json.dumps({"ok": True, "world_size": world_size, "results": results}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
