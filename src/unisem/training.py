"""Source training and budget-constrained target specialization.

The ISBI target profiles use 2,500 samples per epoch, 15 epochs and an
 effective batch of four (9,375 optimizer updates). Monitor samples are drawn
from the same allowed labels and are not an independent validation set.
"""
from __future__ import annotations

import json
import os
import random
from pathlib import Path
from typing import Any, Sequence

PROFILE_NAMES = ("emneuron-stage1", "waferem-lightft", "waferem-fullft", "waferem-head-only")


def profile_from_protocol(protocol: dict[str, Any], name: str) -> dict[str, Any]:
    if name not in PROFILE_NAMES:
        raise ValueError(f"Unknown training profile: {name}")
    if name == "emneuron-stage1":
        source = protocol["emneuron"]["stage1_pretraining"]
        return {"dataset": "emneuron", "model_type": "vit_l", "freeze_policy": "stage1",
                "epochs": int(source["maximum_epochs"]), "patch_shape": source["patch_shape_yx"],
                "batch_size": int(source["batch_size_per_gpu"]), "num_workers": 4,
                "lr": float(source["optimizer"]["learning_rate"]),
                "weight_decay": float(source["optimizer"]["weight_decay"]),
                "gradient_accumulation_steps": 1, "samples_per_epoch": None}
    key = {"waferem-lightft": "lightft", "waferem-fullft": "fullft", "waferem-head-only": "head_only"}[name]
    common, source = protocol["waferem"]["specialization_common"], protocol["waferem"][key]
    return {"dataset": "waferem", "model_type": "vit_l", "freeze_policy": key,
            "epochs": int(common["maximum_epochs"]), "patch_shape": common["patch_shape_yx"],
            "batch_size": int(source["batch_size_per_gpu"]), "num_workers": 4,
            "lr": float(common["learning_rate"]), "weight_decay": float(common["weight_decay"]),
            "gradient_accumulation_steps": int(source["gradient_accumulation_steps"]),
            "samples_per_epoch": 2500}


def seed_everything(seed: int) -> None:
    import numpy as np
    import torch
    random.seed(int(seed))
    np.random.seed(int(seed))
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))


def _seed_worker(worker_id: int) -> None:
    import numpy as np
    import torch
    value = torch.initial_seed() % (2**32)
    random.seed(value)
    np.random.seed(value)


def label_selection(volume_depths: dict[str, int], budget: int, seed: int,
                    manifest: dict[str, Any] | None = None) -> dict[str, list[int]]:
    """Choose labels once, or validate exact frozen lists supplied by the caller."""
    import numpy as np
    if budget not in (160, 315, 625):
        raise ValueError("Paper target budgets are 160, 315 or 625 slices")
    if len(volume_depths) != 5:
        raise ValueError("The paper adaptation protocol uses five volumes")
    per_volume = {160: 32, 315: 63, 625: 125}[budget]
    if manifest is not None:
        source = manifest.get("label_z_lists", manifest)
        if set(source) != set(volume_depths):
            raise ValueError("Label manifest must cover exactly the five adaptation volumes")
        result = {name: [int(z) for z in source[name]] for name in volume_depths}
    else:
        rng = np.random.default_rng(int(seed))
        result = {name: sorted(rng.permutation(depth)[:per_volume].tolist())
                  for name, depth in sorted(volume_depths.items())}
    for name, indices in result.items():
        if len(indices) != per_volume or len(set(indices)) != per_volume:
            raise ValueError(f"Wrong or duplicate label count for {name}")
        if any(z < 0 or z >= volume_depths[name] for z in indices):
            raise ValueError(f"Label slice outside volume: {name}")
    return result


def _sample_to_tensors(raw: Any, labels: Any, patch_shape: tuple[int, int], *,
                       train: bool, component_pseudo: bool, augmentations: Any,
                       crop_origin: tuple[int, int] | None = None) -> tuple[Any, Any]:
    import numpy as np
    import torch
    from .data import make_three_channel_target
    raw, labels = np.asarray(raw), np.asarray(labels, dtype=np.uint32)
    patch_h, patch_w = patch_shape
    pad_h, pad_w = max(0, patch_h - raw.shape[0]), max(0, patch_w - raw.shape[1])
    if pad_h or pad_w:
        raw = np.pad(raw, ((0, pad_h), (0, pad_w)), mode="reflect")
        labels = np.pad(labels, ((0, pad_h), (0, pad_w)), mode="constant")
    limit_y, limit_x = raw.shape[0] - patch_h, raw.shape[1] - patch_w
    if crop_origin is not None:
        y0, x0 = crop_origin
    elif train:
        y0 = int(np.random.randint(limit_y + 1)) if limit_y else 0
        x0 = int(np.random.randint(limit_x + 1)) if limit_x else 0
    else:
        y0, x0 = limit_y // 2, limit_x // 2
    raw = raw[y0:y0 + patch_h, x0:x0 + patch_w].astype(np.float32, copy=False)
    labels = labels[y0:y0 + patch_h, x0:x0 + patch_w]
    if augmentations is not None:
        raw_tensor, label_tensor = augmentations(torch.from_numpy(raw[None]),
                                                 torch.from_numpy(labels.astype(np.int64, copy=False)[None]))
        while raw_tensor.ndim > 2:
            raw_tensor, label_tensor = raw_tensor[0], label_tensor[0]
        raw, labels = raw_tensor.numpy(), np.rint(label_tensor.numpy()).astype(np.uint32)
    from micro_sam.training.util import require_8bit
    raw = require_8bit(raw)
    target = make_three_channel_target(labels, component_pseudo=component_pseudo, min_size=25)
    return torch.from_numpy(raw[None]).float(), torch.from_numpy(target).float()


class _TiffSliceDataset:
    def __init__(self, root: Path, patch_shape: tuple[int, int], *, train: bool) -> None:
        from torch_em.transform import get_augmentations
        suffix = "Tr" if train else "Val"
        images, labels = root / f"images{suffix}", root / f"instances{suffix}"
        self.records = []
        for image in sorted(images.glob("*_0000.tif")):
            label = labels / f"{image.name[:-len('_0000.tif')]}.tif"
            if not label.is_file():
                raise FileNotFoundError(label)
            self.records.append((image, label))
        if not self.records:
            raise ValueError(f"No prepared slices below {images}")
        self.patch_shape, self.train = patch_shape, bool(train)
        self.augmentations = get_augmentations(ndim=2) if train else None

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> tuple[Any, Any]:
        import tifffile
        raw, labels = self.records[int(index)]
        return _sample_to_tensors(tifffile.imread(raw), tifffile.imread(labels), self.patch_shape,
                                  train=self.train, component_pseudo=False, augmentations=self.augmentations)


class _WaferSliceDataset:
    def __init__(self, root: Path, volume_ids: Sequence[str], patch_shape: tuple[int, int], *,
                 train: bool, label_z_lists: dict[str, list[int]] | None = None,
                 samples: int | None = None, seed: int = 12345) -> None:
        import h5py
        import numpy as np
        from torch_em.transform import get_augmentations
        manifest = json.loads((root / "waferem_ds2_manifest.json").read_text(encoding="utf-8"))
        selected = set(volume_ids)
        rows = [row for row in manifest["volumes"] if row["volume_id"] in selected]
        if {row["volume_id"] for row in rows} != selected:
            raise ValueError("Unknown adaptation volume")
        if "wafer4_ds2" in selected:
            raise ValueError("The test volume cannot enter neural training or monitoring")
        self.patch_shape, self.train = patch_shape, bool(train)
        self.augmentations = get_augmentations(ndim=2) if train else None
        self.records, shapes = [], []
        for row in rows:
            raw, labels = root / row["raw_path"], root / row["label_path"]
            key = row["h5_key"]
            with h5py.File(raw, "r") as handle:
                shape = tuple(handle[key].shape)
            zs = range(shape[0]) if label_z_lists is None else label_z_lists[row["volume_id"]]
            for z in zs:
                self.records.append((raw, labels, key, int(z)))
                shapes.append(shape[-2:])
        if not self.records:
            raise ValueError("Empty adaptation dataset")
        self.samples = len(self.records) if samples is None else int(samples)
        self.monitor_records = []
        if not train:
            rng = np.random.default_rng(int(seed))
            for _ in range(self.samples):
                record_index = int(rng.integers(len(self.records)))
                h, w = shapes[record_index]
                origin = (int(rng.integers(max(0, h - patch_shape[0]) + 1)),
                          int(rng.integers(max(0, w - patch_shape[1]) + 1)))
                self.monitor_records.append((record_index, origin))

    def __len__(self) -> int:
        return self.samples

    def __getitem__(self, index: int) -> tuple[Any, Any]:
        import h5py
        import numpy as np
        crop = None
        if self.train:
            record_index = int(np.random.randint(len(self.records)))
        else:
            record_index, crop = self.monitor_records[int(index)]
        raw_path, label_path, key, z = self.records[record_index]
        with h5py.File(raw_path, "r") as raw_file, h5py.File(label_path, "r") as label_file:
            raw, labels = np.asarray(raw_file[key][z]), np.asarray(label_file[key][z], dtype=np.uint32)
        return _sample_to_tensors(raw, labels, self.patch_shape, train=self.train,
                                  component_pseudo=True, augmentations=self.augmentations, crop_origin=crop)


def _waferem_datasets(root: Path, protocol: dict[str, Any], patch_shape: tuple[int, int], *,
                      budget: int, seed: int, label_manifest: dict[str, Any] | None) -> tuple[Any, Any, dict]:
    import h5py
    manifest = json.loads((root / "waferem_ds2_manifest.json").read_text(encoding="utf-8"))
    names = protocol["waferem"]["adaptation_volumes"]
    depths = {}
    for row in manifest["volumes"]:
        if row["volume_id"] in names:
            with h5py.File(root / row["raw_path"], "r") as handle:
                depths[row["volume_id"]] = int(handle[row["h5_key"]].shape[0])
    labels = label_selection(depths, budget, seed, label_manifest)
    common = dict(root=root, volume_ids=names, patch_shape=patch_shape, label_z_lists=labels, seed=seed)
    monitor_samples = int(protocol["waferem"]["specialization_common"]["monitor_tiles_by_budget"][str(budget)])
    return (_WaferSliceDataset(**common, train=True, samples=2500),
            _WaferSliceDataset(**common, train=False, samples=monitor_samples), labels)


def _mean_loss(model: Any, loader: Any, loss_fn: Any, device: Any, *, train_mode: bool,
               optimizer: Any, scaler: Any, max_steps: int = 0, accumulation: int = 1) -> tuple[float, int]:
    import torch
    model.train(train_mode)
    total = torch.zeros(2, device=device, dtype=torch.float64)
    updates = 0
    if train_mode:
        optimizer.zero_grad(set_to_none=True)
    context = torch.enable_grad if train_mode else torch.inference_mode
    with context():
        for micro_step, (image, target) in enumerate(loader, start=1):
            if train_mode and max_steps and updates >= max_steps:
                break
            image, target = image.to(device), target.to(device)
            if target.ndim != 4 or target.shape[1] != 3:
                raise ValueError("Expected BCHW three-channel targets")
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=device.type == "cuda"):
                loss = loss_fn(model(image).float(), target.float())
            if train_mode:
                scaler.scale(loss / accumulation).backward()
                if micro_step % accumulation == 0:
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad(set_to_none=True)
                    updates += 1
            total[0] += loss.detach().double()
            total[1] += 1
    if train_mode and len(loader) % accumulation and not (max_steps and updates >= max_steps):
        raise ValueError("Microbatch count is not divisible by gradient accumulation")
    if torch.distributed.is_initialized():
        torch.distributed.all_reduce(total)
    if total[1] == 0:
        raise ValueError("Empty training or monitor loader")
    return float((total[0] / total[1]).item()), updates


def train(*, profile_name: str, protocol: dict[str, Any], data_root: str | Path,
          output_dir: str | Path, sam_checkpoint: str | Path,
          init_checkpoint: str | Path | None = None, epochs: int | None = None,
          max_steps: int = 0, seed: int = 12345, label_layers: int = 625,
          label_manifest: str | Path | None = None) -> dict[str, Any]:
    import torch
    from torch.nn.parallel import DistributedDataParallel
    from torch.utils.data import DataLoader
    from torch.utils.data.distributed import DistributedSampler
    from torch_em.loss import DiceBasedDistanceLoss
    from .checkpoints import inference_checkpoint
    from .model import _load_sam_image_encoder, apply_freeze_policy, build_model, load_model
    profile = profile_from_protocol(protocol, profile_name)
    rank, local_rank, world_size = (int(os.environ.get(key, default)) for key, default in
                                    (("RANK", "0"), ("LOCAL_RANK", "0"), ("WORLD_SIZE", "1")))
    if profile["dataset"] == "waferem" and world_size != 1:
        raise ValueError("Paper target profiles use one process to preserve effective batch four")
    if world_size > 1 and not torch.distributed.is_initialized():
        torch.distributed.init_process_group("nccl" if torch.cuda.is_available() else "gloo")
    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.cuda.set_device(device)
    seed_everything(seed + rank)
    data_root, output_dir = Path(data_root).resolve(), Path(output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    patch_shape = tuple(profile["patch_shape"])
    labels = None
    if profile["dataset"] == "emneuron":
        train_data = _TiffSliceDataset(data_root, patch_shape, train=True)
        val_data = _TiffSliceDataset(data_root, patch_shape, train=False)
    else:
        exact = None if label_manifest is None else json.loads(Path(label_manifest).read_text(encoding="utf-8"))
        if exact is None:
            exact = protocol["waferem"].get("label_budgets", {}).get(str(seed), {}).get(str(label_layers))
        train_data, val_data, labels = _waferem_datasets(data_root, protocol, patch_shape,
            budget=label_layers, seed=seed, label_manifest=exact)
    generator = torch.Generator().manual_seed(seed + rank)
    sampler = DistributedSampler(train_data, seed=seed) if world_size > 1 else None
    kwargs = dict(batch_size=profile["batch_size"], num_workers=profile["num_workers"],
                  pin_memory=device.type == "cuda", worker_init_fn=_seed_worker, generator=generator)
    train_loader = DataLoader(train_data, sampler=sampler, shuffle=sampler is None, drop_last=True, **kwargs)
    val_loader = DataLoader(val_data, shuffle=False, **kwargs)
    image_encoder = _load_sam_image_encoder("vit_l", sam_checkpoint)
    if init_checkpoint is None:
        if profile["dataset"] != "emneuron":
            raise ValueError("Target specialization requires a Stage-1 --init-checkpoint")
        model = build_model(image_encoder, device=device, freeze_policy="stage1")
    else:
        model = load_model(init_checkpoint, image_encoder=image_encoder,
                           model_type="vit_l", device=device, freeze_policy=profile["freeze_policy"])
    trainability = apply_freeze_policy(model, profile["freeze_policy"])
    parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    optimizer = torch.optim.AdamW(parameters, lr=profile["lr"], weight_decay=profile["weight_decay"])
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.9, patience=3)
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
    wrapped = DistributedDataParallel(model, device_ids=[local_rank] if device.type == "cuda" else None) if world_size > 1 else model
    loss_fn = DiceBasedDistanceLoss(mask_distances_in_bg=True)
    total_epochs = profile["epochs"] if epochs is None else int(epochs)
    if total_epochs < 1 or max_steps < 0:
        raise ValueError("Epochs must be positive and max-steps non-negative")
    history, best, updates = [], float("inf"), 0
    for epoch in range(1, total_epochs + 1):
        if sampler is not None:
            sampler.set_epoch(epoch)
        remaining = max_steps - updates if max_steps else 0
        train_loss, steps = _mean_loss(wrapped, train_loader, loss_fn, device, train_mode=True,
            optimizer=optimizer, scaler=scaler, max_steps=remaining,
            accumulation=profile["gradient_accumulation_steps"])
        updates += steps
        monitor_loss, _ = _mean_loss(wrapped, val_loader, loss_fn, device, train_mode=False,
                                     optimizer=None, scaler=None)
        scheduler.step(monitor_loss)
        history.append({"epoch": epoch, "train_loss": train_loss,
                        "monitor_loss": monitor_loss, "optimizer_updates": updates})
        if rank == 0 and monitor_loss < best:
            best = monitor_loss
            torch.save(inference_checkpoint(wrapped, model_type="vit_l", public_hyperparameters={
                "profile": profile_name, "epoch": epoch, "seed": seed, "optimizer_updates": updates}),
                output_dir / "best_inference.pt")
        if max_steps and updates >= max_steps:
            break
    result = {"profile": profile_name, "seed": seed, "epochs": len(history), "world_size": world_size,
              "optimizer_updates": updates, "effective_batch": profile["batch_size"] * world_size * profile["gradient_accumulation_steps"],
              "monitor_role": "source_validation" if labels is None else "budget_internal_training_monitor",
              "trainability": trainability, "label_z_lists": labels, "history": history,
              "paper_profile_unmodified": epochs is None and not max_steps}
    if rank == 0:
        (output_dir / "training_summary.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    if torch.distributed.is_initialized():
        torch.distributed.barrier()
    return result


__all__ = ["PROFILE_NAMES", "profile_from_protocol", "train", "label_selection", "seed_everything"]
