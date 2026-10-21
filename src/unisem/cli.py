from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any, Sequence

import numpy as np


SCALE_DATASET_NAMES = {
    "large": "tile512_overlap",
    "medium": "tile384_overlap",
    "small": "tile256_overlap",
}


def _json(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _write_json(path: str | Path, payload: Any) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="ascii")


def _load_array(path: str | Path, key: str = "main") -> np.ndarray:
    source = Path(path)
    if source.suffix.lower() in {".tif", ".tiff"}:
        import tifffile

        return np.asarray(tifffile.imread(source))
    if source.suffix.lower() in {".h5", ".hdf5"}:
        import h5py

        with h5py.File(source, "r") as handle:
            if key not in handle:
                raise KeyError(f"Missing HDF5 dataset {key!r} in {source}")
            return np.asarray(handle[key])
    raise ValueError(f"Unsupported array file: {source}")


def _as_stack(array: np.ndarray) -> tuple[np.ndarray, bool]:
    values = np.asarray(array)
    if values.ndim == 2:
        return values[None], True
    if values.ndim == 3:
        return values, False
    raise ValueError(f"Expected a 2D image or ZYX stack, got {values.shape}")


def _watershed_kwargs(protocol: dict[str, Any], preset: str) -> dict[str, Any]:
    config = protocol["inference"]["seeded_watershed"][preset]
    return {
        "foreground_threshold": config["foreground_threshold"],
        "center_distance_threshold": config["center_distance_threshold"],
        "boundary_distance_threshold": config["boundary_distance_threshold"],
        "foreground_smoothing": config["foreground_smoothing_sigma"],
        "distance_smoothing": config["distance_smoothing_sigma"],
        "min_instance_area": config["minimum_instance_area"],
    }


def command_prepare_data(args: argparse.Namespace) -> dict[str, Any]:
    from .data import materialize_emneuron_volume_split, materialize_waferem_ds2

    protocol = _json(args.config)
    if args.dataset == "emneuron":
        split = protocol["emneuron"]["split"]
        return materialize_emneuron_volume_split(
            args.input_root,
            args.output_root,
            validation_fraction=float(split["validation_fraction"]),
            seed=int(split["seed"]),
            overwrite=args.overwrite,
        )
    return materialize_waferem_ds2(args.input_root, args.output_root, overwrite=args.overwrite)


def command_train(args: argparse.Namespace) -> dict[str, Any]:
    from .training import train

    return train(
        profile_name=args.profile,
        protocol=_json(args.config),
        data_root=args.data_root,
        output_dir=args.output_root,
        sam_checkpoint=args.sam_checkpoint,
        init_checkpoint=args.init_checkpoint,
        epochs=args.epochs,
        max_steps=args.max_steps,
        seed=args.seed,
        label_layers=args.label_layers,
        label_manifest=args.label_manifest,
    )


def _load_model(args: argparse.Namespace) -> tuple[Any, Any]:
    import torch

    from .model import load_model

    device = torch.device(args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu"))
    model = load_model(
        args.checkpoint,
        model_type=args.model_type,
        sam_checkpoint_path=args.sam_checkpoint,
        device=device,
        strict=True,
        base_checkpoint_path=args.base_checkpoint,
    )
    model.eval()
    return model, device


def _predict_scale(
    image: np.ndarray,
    model: Any,
    device: Any,
    scale: dict[str, Any],
) -> np.ndarray:
    import torch

    from .inference import overlap_fusion, normalize_raw

    def predict_tile(tile: np.ndarray) -> np.ndarray:
        values = tile.astype(np.float32, copy=False)
        tensor = torch.from_numpy(values[None, None]).to(device)
        with torch.inference_mode():
            prediction = model(tensor)
        return prediction[0].float().cpu().numpy()

    return overlap_fusion(
        normalize_raw(image),
        predict_tile,
        tile_shape=scale["tile_shape_yx"],
        stride=scale["stride_yx"],
    )


def command_infer(args: argparse.Namespace) -> dict[str, Any]:
    from .inference import CHANNEL_NAMES, seeded_watershed, write_h5
    from .rag import extract_edge_features, load_random_forest, merge_fragments

    protocol = _json(args.config)
    raw_stack, squeezed = _as_stack(_load_array(args.input, args.input_key))
    model, device = _load_model(args)
    rf = load_random_forest(args.rag_checkpoint) if args.rag_checkpoint else None
    if rf is None and not args.dense_only:
        raise ValueError("--rag-checkpoint is required unless --dense-only is set")
    preset = "emneuron_main_rag_fragment_default" if args.dataset == "emneuron" else "waferem_fixed"
    watershed_kwargs = _watershed_kwargs(protocol, preset)

    dense_stacks: dict[str, list[np.ndarray]] = {}
    segmentation_stacks: dict[str, list[np.ndarray]] = {}
    merged_stack: list[np.ndarray] = []
    for raw in raw_stack:
        states: dict[str, dict[str, np.ndarray]] = {}
        segmentations: dict[str, np.ndarray] = {}
        for scale in protocol["inference"]["scales"]:
            public_name = SCALE_DATASET_NAMES[scale["name"]]
            prediction = _predict_scale(raw, model, device, scale)
            states[public_name] = {name: prediction[index] for index, name in enumerate(CHANNEL_NAMES)}
            segmentations[public_name] = seeded_watershed(prediction, **watershed_kwargs)
            for index, channel in enumerate(CHANNEL_NAMES):
                dense_stacks.setdefault(f"{channel}_{public_name}", []).append(prediction[index])
            segmentation_stacks.setdefault(f"segmentation_{public_name}", []).append(segmentations[public_name])
        medium_name = SCALE_DATASET_NAMES["medium"]
        if rf is not None:
            batch = extract_edge_features(
                segmentations[medium_name],
                states,
                segmentations,
                raw.astype(np.float32, copy=False),
            )
            merged_stack.append(merge_fragments(segmentations[medium_name], batch.edges, batch.features, rf))

    datasets = {name: np.stack(values) for name, values in {**dense_stacks, **segmentation_stacks}.items()}
    medium_name = SCALE_DATASET_NAMES["medium"]
    for channel in CHANNEL_NAMES:
        datasets[channel] = datasets[f"{channel}_{medium_name}"]
    if merged_stack:
        datasets["final_merged_segmentation"] = np.stack(merged_stack)
    if squeezed:
        datasets = {name: values[0] for name, values in datasets.items()}
    write_h5(args.output, datasets)
    return {"output": str(args.output), "slices": int(len(raw_stack)), "merged": bool(merged_stack)}


def command_train_rag(args: argparse.Namespace) -> dict[str, Any]:
    import h5py

    from .data import component_pseudo_relabel_2d
    from .inference import CHANNEL_NAMES
    from .rag import extract_edge_features, fit_random_forest

    manifest_path = Path(args.manifest).resolve()
    manifest = _json(manifest_path)
    feature_rows = []
    label_rows = []
    selected_labels = {}
    seen_slices = set()
    expected_labels = manifest.get("label_z_lists")
    for record_index, record in enumerate(manifest["records"]):
        volume = str(record.get("volume", f"record_{record_index}"))
        dataset = str(record.get("dataset", manifest.get("dataset", "custom"))).lower()
        split = str(record.get("split", "train")).lower()
        if split in {"test", "final_test", "heldout", "held_out"} or volume.lower() in {"wafer4", "wafer4_ds2"}:
            raise ValueError("Test volumes cannot be used to fit a random forest")
        raw_stack, _ = _as_stack(_load_array(manifest_path.parent / record["raw"], record.get("raw_key", "main")))
        gt_stack, _ = _as_stack(_load_array(manifest_path.parent / record["ground_truth"], record.get("ground_truth_key", "main")))
        if raw_stack.shape != gt_stack.shape:
            raise ValueError(f"Raw/GT shape mismatch for {volume}: {raw_stack.shape} vs {gt_stack.shape}")
        if dataset.startswith("wafer") and "z_indices" not in record:
            raise ValueError("WaferEM RF fitting requires explicit budget-constrained z_indices")
        indices = record.get("z_indices", list(range(len(raw_stack))))
        if not indices or any(not isinstance(z, int) or isinstance(z, bool) for z in indices):
            raise ValueError("z_indices must be a non-empty integer list")
        if len(set(indices)) != len(indices) or any(z < 0 or z >= len(raw_stack) for z in indices):
            raise ValueError(f"Duplicate or out-of-range z_indices for {volume}")
        if expected_labels is not None and (volume not in expected_labels or not set(indices) <= set(expected_labels[volume])):
            raise ValueError(f"RF labels exceed the declared neural label budget for {volume}")
        for z in indices:
            if (volume, z) in seen_slices:
                raise ValueError(f"Duplicate RF training slice: {volume}/{z}")
            seen_slices.add((volume, z))
        selected_labels.setdefault(volume, []).extend(indices)
        with h5py.File(manifest_path.parent / record["prediction"], "r") as predictions:
            names = [f"{channel}_{scale}" for scale in SCALE_DATASET_NAMES.values() for channel in CHANNEL_NAMES]
            names += [f"segmentation_{scale}" for scale in SCALE_DATASET_NAMES.values()]
            for name in names:
                if name not in predictions:
                    raise ValueError(f"Missing prediction dataset: {name}")
                shape = predictions[name].shape
                stack_shape = (1, *shape) if len(shape) == 2 else shape
                if stack_shape != raw_stack.shape:
                    raise ValueError(f"Prediction shape mismatch for {volume}/{name}: {stack_shape} vs {raw_stack.shape}")
            def prediction_slice(name: str, index: int) -> np.ndarray:
                dataset = predictions[name]
                return np.asarray(dataset if dataset.ndim == 2 else dataset[index])

            for z in indices:
                raw, ground_truth = raw_stack[z], gt_stack[z]
                states = {
                    scale: {channel: prediction_slice(f"{channel}_{scale}", z) for channel in CHANNEL_NAMES}
                    for scale in SCALE_DATASET_NAMES.values()
                }
                segmentations = {
                    scale: prediction_slice(f"segmentation_{scale}", z)
                    for scale in SCALE_DATASET_NAMES.values()
                }
                medium = SCALE_DATASET_NAMES["medium"]
                batch = extract_edge_features(
                    segmentations[medium],
                    states,
                    segmentations,
                    raw.astype(np.float32, copy=False),
                    ground_truth=component_pseudo_relabel_2d(ground_truth),
                )
                if batch.labels is not None and len(batch.features):
                    feature_rows.append(batch.features)
                    label_rows.append(batch.labels)
    if not feature_rows:
        raise ValueError("RAG manifest did not yield any candidate edges")
    features = np.concatenate(feature_rows, axis=0)
    labels = np.concatenate(label_rows, axis=0)
    workers = max(1, min(8, int(args.workers)))
    metadata = {"label_z_lists": selected_labels, "label_slices": len(seen_slices),
                "neural_budget_membership_checked": expected_labels is not None}
    if "label_budget" in manifest and int(manifest["label_budget"]) != len(seen_slices):
        raise ValueError("Declared label_budget differs from selected RF training slices")
    fit_random_forest(features, labels, n_jobs=workers, output_path=args.output, training_metadata=metadata)
    return {
        "output": str(args.output),
        "edges": int(len(labels)),
        "positive_edges": int(labels.sum()),
        "negative_edges": int(len(labels) - labels.sum()),
        **metadata,
    }


def command_evaluate(args: argparse.Namespace) -> dict[str, Any]:
    from .metrics import aggregate_volumes, evaluate_per_slice

    if args.manifest:
        manifest_path = Path(args.manifest).resolve()
        records = _json(manifest_path)["volumes"]
        volumes = {}
        for record in records:
            name = str(record["volume"])
            if name in volumes:
                raise ValueError(f"Duplicate volume: {name}")
            prediction_path = manifest_path.parent / record["prediction"]
            gt_path = manifest_path.parent / record["ground_truth"]
            volumes[name] = evaluate_per_slice(
                _load_array(prediction_path, record.get("prediction_key", args.prediction_key)),
                _load_array(gt_path, record.get("ground_truth_key", args.ground_truth_key)),
            )
        expected = ["FAFB-1", "FAFB-2", "FIB25-1", "FIB25-2", "Kstr", "Harris", "Ionsem", "Microns[B]"] if args.dataset == "emneuron" else ["wafer4_ds2"]
        aggregate = aggregate_volumes(volumes, expected_volumes=expected)
        expected_slices = 1003 if args.dataset == "emneuron" else 125
        if aggregate["num_slices"] != expected_slices:
            raise ValueError(f"Expected {expected_slices} slices, got {aggregate['num_slices']}")
        _write_json(args.output, {"aggregate": aggregate, "volumes": volumes})
        return aggregate
    if not args.ground_truth:
        raise ValueError("Single-volume evaluation requires --ground-truth")

    prediction_stack, _ = _as_stack(_load_array(args.prediction, args.prediction_key))
    ground_truth_stack, _ = _as_stack(_load_array(args.ground_truth, args.ground_truth_key))
    if prediction_stack.shape != ground_truth_stack.shape:
        raise ValueError(f"Prediction/ground-truth shape mismatch: {prediction_stack.shape} vs {ground_truth_stack.shape}")
    evaluation = evaluate_per_slice(prediction_stack, ground_truth_stack)
    result = {"slices": len(evaluation["slices"]), "aggregate": evaluation["aggregate"], "per_slice": evaluation["slices"]}
    _write_json(args.output, result)
    return result["aggregate"]


def command_inspect_checkpoint(args: argparse.Namespace) -> dict[str, Any]:
    from .checkpoints import inspect_checkpoint

    result = inspect_checkpoint(args.checkpoint)
    if args.output:
        _write_json(args.output, result)
    return result


def command_sanitize_checkpoint(args: argparse.Namespace) -> dict[str, Any]:
    from .checkpoints import sanitize_checkpoint_file

    return sanitize_checkpoint_file(args.source, args.output, model_type=args.model_type)


def command_export_adapter(args: argparse.Namespace) -> dict[str, Any]:
    import torch
    from .checkpoints import make_adapter
    payload = make_adapter(args.source, args.base_checkpoint)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, output)
    return {"output": str(output), "base_sha256": payload["base_sha256"],
            "lora_tensors": len(payload["lora_state"]), "decoder_tensors": len(payload["decoder_state"])}


def command_smoke(args: argparse.Namespace) -> dict[str, Any]:
    from .checkpoints import load_inference_checkpoint, sanitize_checkpoint
    from .data import component_pseudo_relabel_2d, make_three_channel_target
    from .inference import seeded_watershed
    from .metrics import aggregate_slice_metrics, evaluate_segmentation

    import torch

    yy, xx = np.mgrid[:64, :64]
    labels = np.zeros((64, 64), dtype=np.uint32)
    labels[(yy - 20) ** 2 + (xx - 20) ** 2 < 100] = 1
    labels[(yy - 44) ** 2 + (xx - 44) ** 2 < 100] = 2
    target = make_three_channel_target(labels)
    segmentation = seeded_watershed(target, min_instance_area=4)
    metrics = evaluate_segmentation(segmentation, component_pseudo_relabel_2d(labels))
    legacy = {
        "model_type": "vit_b",
        "model_state": {"image_encoder.weight": torch.ones(1)},
        "decoder_state": {
            "baseline_decoder.weight": torch.ones(1),
            "hq_residual.weight": torch.ones(1),
        },
    }
    clean = sanitize_checkpoint(legacy)
    load_inference_checkpoint(clean)
    result = {
        "ok": True,
        "target_shape": list(target.shape),
        "instances": int(segmentation.max()),
        "metrics": aggregate_slice_metrics([metrics]),
    }
    if args.output:
        _write_json(args.output, result)
    return result


def _add_config(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--config", required=True, help="Path to paper_protocol.json")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="unisem")
    subparsers = parser.add_subparsers(dest="command", required=True)

    prepare = subparsers.add_parser("prepare-data")
    prepare.add_argument("--dataset", choices=("emneuron", "waferem"), required=True)
    prepare.add_argument("--input-root", required=True)
    prepare.add_argument("--output-root", required=True)
    prepare.add_argument("--overwrite", action="store_true")
    _add_config(prepare)
    prepare.set_defaults(func=command_prepare_data)

    train_parser = subparsers.add_parser("train")
    train_parser.add_argument("--profile", choices=("emneuron-stage1", "waferem-lightft", "waferem-fullft", "waferem-head-only"), required=True)
    train_parser.add_argument("--data-root", required=True)
    train_parser.add_argument("--output-root", required=True)
    train_parser.add_argument("--sam-checkpoint", required=True)
    train_parser.add_argument("--init-checkpoint")
    train_parser.add_argument("--seed", type=int, default=12345)
    train_parser.add_argument("--label-layers", type=int, choices=(160, 315, 625), default=625)
    train_parser.add_argument("--label-manifest", help="Exact slice lists; defaults to frozen paper lists for recorded seeds, otherwise seeded sampling")
    train_parser.add_argument("--epochs", type=int)
    train_parser.add_argument("--max-steps", type=int, default=0)
    _add_config(train_parser)
    train_parser.set_defaults(func=command_train)

    infer = subparsers.add_parser("infer")
    infer.add_argument("--dataset", choices=("emneuron", "waferem"), required=True)
    infer.add_argument("--input", required=True)
    infer.add_argument("--input-key", default="main")
    infer.add_argument("--output", required=True)
    infer.add_argument("--checkpoint", required=True)
    infer.add_argument("--base-checkpoint", help="Exact sanitized Stage-1 checkpoint required by an adapter")
    infer.add_argument("--sam-checkpoint", help="Optional local official SAM initialization; full released weights are self-contained")
    infer.add_argument("--rag-checkpoint")
    infer.add_argument("--dense-only", action="store_true")
    infer.add_argument("--model-type", default="vit_l")
    infer.add_argument("--device")
    _add_config(infer)
    infer.set_defaults(func=command_infer)

    train_rag = subparsers.add_parser("train-rag")
    train_rag.add_argument("--manifest", required=True)
    train_rag.add_argument("--output", required=True)
    train_rag.add_argument("--workers", type=int, default=8)
    train_rag.set_defaults(func=command_train_rag)

    evaluate = subparsers.add_parser("evaluate")
    eval_input = evaluate.add_mutually_exclusive_group(required=True)
    eval_input.add_argument("--prediction")
    eval_input.add_argument("--manifest", help="JSON volumes list for equal-volume evaluation")
    evaluate.add_argument("--dataset", choices=("emneuron", "waferem"), default="emneuron")
    evaluate.add_argument("--prediction-key", default="final_merged_segmentation")
    evaluate.add_argument("--ground-truth")
    evaluate.add_argument("--ground-truth-key", default="main")
    evaluate.add_argument("--output", required=True)
    evaluate.set_defaults(func=command_evaluate)

    inspect = subparsers.add_parser("inspect-checkpoint")
    inspect.add_argument("--checkpoint", required=True)
    inspect.add_argument("--output")
    inspect.set_defaults(func=command_inspect_checkpoint)

    sanitize = subparsers.add_parser("sanitize-checkpoint")
    sanitize.add_argument("--source", required=True)
    sanitize.add_argument("--output", required=True)
    sanitize.add_argument("--model-type", default="vit_l")
    sanitize.set_defaults(func=command_sanitize_checkpoint)

    adapter = subparsers.add_parser("export-adapter")
    adapter.add_argument("--source", required=True, help="Sanitized full LightFT checkpoint")
    adapter.add_argument("--base-checkpoint", required=True, help="Sanitized source checkpoint")
    adapter.add_argument("--output", required=True)
    adapter.set_defaults(func=command_export_adapter)

    smoke = subparsers.add_parser("smoke")
    smoke.add_argument("--output")
    smoke.set_defaults(func=command_smoke)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    os.environ.setdefault("OMP_NUM_THREADS", "8")
    os.environ.setdefault("MKL_NUM_THREADS", "8")
    args = build_parser().parse_args(argv)
    result = args.func(args)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
