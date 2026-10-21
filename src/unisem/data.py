from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Any, Sequence

import numpy as np


WAFEREM_CASES = (
    "wafer4",
    "wafer25",
    "wafer26",
    "wafer26_2",
    "wafer36",
    "wafer36_2",
)
WAFEREM_ADAPTATION_CASES = (
    "wafer25",
    "wafer26",
    "wafer26_2",
    "wafer36",
    "wafer36_2",
)
WAFEREM_FINAL_TEST_CASE = "wafer4"


def _labels_2d(labels: np.ndarray) -> np.ndarray:
    array = np.asarray(labels)
    if array.ndim != 2:
        raise ValueError(f"Expected a 2D label map, got shape {array.shape}")
    if not np.issubdtype(array.dtype, np.integer):
        rounded = np.rint(array)
        if not np.array_equal(array, rounded):
            raise ValueError("Label maps must contain integer values")
        array = rounded.astype(np.int64)
    if array.size and int(array.min()) < 0:
        raise ValueError("Label maps must not contain negative ids")
    return array


def component_pseudo_relabel_2d(
    labels: np.ndarray,
    *,
    min_component_area: int = 1,
) -> np.ndarray:
    """Relabel each 2D 8-connected component as a separate instance.

    Background remains zero. Components are assigned deterministic consecutive
    ids by original instance id and raster-scan component order.
    """

    from scipy import ndimage as ndi

    if int(min_component_area) < 1:
        raise ValueError("min_component_area must be at least 1")
    labels = _labels_2d(labels)
    output = np.zeros(labels.shape, dtype=np.uint32)
    structure = np.ones((3, 3), dtype=np.uint8)
    next_id = 1
    original_ids = sorted(
        int(value) for value in np.unique(labels) if int(value) != 0
    )
    for original_id in original_ids:
        components, count = ndi.label(labels == original_id, structure=structure)
        for component_id in range(1, int(count) + 1):
            mask = components == component_id
            if int(mask.sum()) < int(min_component_area):
                continue
            output[mask] = np.uint32(next_id)
            next_id += 1
    return output


def make_three_channel_target(
    labels: np.ndarray,
    *,
    component_pseudo: bool = False,
    min_component_area: int = 1,
    min_size: int = 1,
) -> np.ndarray:
    """Build foreground, center-distance, and boundary-distance targets.

    The channel order is ``foreground, center_distances,
    boundary_distances`` and follows torch-em's
    ``PerObjectDistanceTransform``.
    """

    from torch_em.transform.label import PerObjectDistanceTransform

    labels = _labels_2d(labels)
    if component_pseudo:
        labels = component_pseudo_relabel_2d(
            labels,
            min_component_area=min_component_area,
        )
    transform = PerObjectDistanceTransform(
        distances=True,
        boundary_distances=True,
        directed_distances=False,
        foreground=True,
        instances=False,
        min_size=int(min_size),
    )
    target = np.asarray(transform(labels), dtype=np.float32)
    expected_shape = (3, *labels.shape)
    if target.shape != expected_shape:
        raise RuntimeError(
            f"Unexpected torch-em target shape {target.shape}; expected {expected_shape}"
        )
    if not np.isfinite(target).all():
        raise RuntimeError("torch-em target contains non-finite values")
    return target


def _safe_case_id(relative_label_path: Path) -> str:
    suffix = "_MaskIns.tif"
    text = relative_label_path.as_posix()
    if not text.endswith(suffix):
        raise ValueError(f"Expected an EMNeuron label ending in {suffix}: {text}")
    text = text[: -len(suffix)]
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", text).strip("_")


def _tiff_stack_shape(path: Path) -> tuple[int, int, int]:
    import tifffile

    with tifffile.TiffFile(path) as handle:
        shape = tuple(int(value) for value in handle.series[0].shape)
    if len(shape) == 2:
        return (1, *shape)
    if len(shape) != 3:
        raise ValueError(f"Expected a 2D or 3D TIFF stack at {path}, got {shape}")
    return shape


def discover_emneuron_pairs(
    dataset_root: str | Path,
    *,
    excluded_directory_names: Sequence[str] = ("valid",),
) -> list[dict[str, Any]]:
    """Discover EMNeuron raw/``*_MaskIns.tif`` pairs below ``dataset_root``.

    All paths in the returned records are relative to ``dataset_root``.
    """

    dataset_root = Path(dataset_root).expanduser().resolve()
    if not dataset_root.is_dir():
        raise NotADirectoryError(dataset_root)
    excluded = {str(name) for name in excluded_directory_names}
    pairs: list[dict[str, Any]] = []
    for label_path in sorted(dataset_root.rglob("*_MaskIns.tif")):
        relative_label = label_path.relative_to(dataset_root)
        if excluded.intersection(relative_label.parts[:-1]):
            continue
        raw_path = label_path.with_name(
            label_path.name[: -len("_MaskIns.tif")] + ".tif"
        )
        if not raw_path.is_file():
            continue
        raw_shape = _tiff_stack_shape(raw_path)
        label_shape = _tiff_stack_shape(label_path)
        if raw_shape != label_shape:
            raise ValueError(
                f"Raw/label shape mismatch for {relative_label}: "
                f"{raw_shape} vs {label_shape}"
            )
        pairs.append(
            {
                "case_id": _safe_case_id(relative_label),
                "raw_path": raw_path.relative_to(dataset_root).as_posix(),
                "label_path": relative_label.as_posix(),
                "shape": list(raw_shape),
                "num_slices": int(raw_shape[0]),
            }
        )
    if not pairs:
        raise FileNotFoundError(
            f"No EMNeuron raw/label pairs ending in *_MaskIns.tif under {dataset_root}"
        )
    return pairs


def _read_tiff_stack(path: Path) -> np.ndarray:
    import tifffile

    array = np.asarray(tifffile.imread(path))
    if array.ndim == 2:
        array = array[None]
    if array.ndim != 3:
        raise ValueError(f"Expected a 2D or 3D TIFF stack at {path}, got {array.shape}")
    return array


def _write_tiff(path: Path, array: np.ndarray, *, overwrite: bool) -> None:
    import tifffile

    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and not overwrite:
        return
    tifffile.imwrite(path, np.asarray(array))


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="ascii",
    )
    temporary.replace(path)


def materialize_emneuron_volume_split(
    dataset_root: str | Path,
    output_root: str | Path,
    *,
    validation_fraction: float = 0.1,
    seed: int = 0,
    overwrite: bool = False,
    excluded_directory_names: Sequence[str] = ("valid",),
) -> dict[str, Any]:
    """Materialize a deterministic volume-level EMNeuron train/validation split.

    Complete source volumes, not individual slices, are assigned to validation.
    Materialized and source paths in the manifest are relative to their injected
    roots.
    """

    if not 0.0 < float(validation_fraction) < 1.0:
        raise ValueError("validation_fraction must be between 0 and 1")
    dataset_root = Path(dataset_root).expanduser().resolve()
    output_root = Path(output_root).expanduser().resolve()
    pairs = discover_emneuron_pairs(
        dataset_root,
        excluded_directory_names=excluded_directory_names,
    )
    if len(pairs) < 2:
        raise ValueError("A volume-level split requires at least two EMNeuron volumes")

    rng = np.random.default_rng(int(seed))
    order = np.arange(len(pairs))
    rng.shuffle(order)
    validation_count = max(1, int(round(len(pairs) * float(validation_fraction))))
    validation_count = min(len(pairs) - 1, validation_count)
    validation_indices = {int(index) for index in order[:validation_count]}

    volume_records: list[dict[str, Any]] = []
    for index, pair in enumerate(pairs):
        split = "validation" if index in validation_indices else "train"
        image_directory = output_root / (
            "imagesVal" if split == "validation" else "imagesTr"
        )
        label_directory = output_root / (
            "instancesVal" if split == "validation" else "instancesTr"
        )
        raw_stack = _read_tiff_stack(dataset_root / pair["raw_path"])
        label_stack = _read_tiff_stack(dataset_root / pair["label_path"])
        if raw_stack.shape != label_stack.shape:
            raise ValueError(
                f"Raw/label shape mismatch for {pair['case_id']}: "
                f"{raw_stack.shape} vs {label_stack.shape}"
            )
        if label_stack.size and int(label_stack.max()) > np.iinfo(np.uint32).max:
            raise ValueError(f"Label ids exceed uint32 for {pair['case_id']}")

        slices: list[dict[str, Any]] = []
        for z in range(int(raw_stack.shape[0])):
            slice_case_id = f"{pair['case_id']}_z{z:04d}"
            image_path = image_directory / f"{slice_case_id}_0000.tif"
            label_path = label_directory / f"{slice_case_id}.tif"
            _write_tiff(image_path, raw_stack[z], overwrite=overwrite)
            _write_tiff(
                label_path,
                label_stack[z].astype(np.uint32, copy=False),
                overwrite=overwrite,
            )
            slices.append(
                {
                    "case_id": slice_case_id,
                    "source_z": z,
                    "image_path": image_path.relative_to(output_root).as_posix(),
                    "label_path": label_path.relative_to(output_root).as_posix(),
                }
            )
        volume_records.append(
            {
                **pair,
                "split": split,
                "slices": slices,
            }
        )

    manifest = {
        "dataset": "EMNeuron",
        "split_unit": "volume",
        "validation_fraction": float(validation_fraction),
        "seed": int(seed),
        "path_bases": {
            "source_paths": "dataset_root argument",
            "materialized_paths": "manifest directory",
        },
        "num_volumes": len(volume_records),
        "num_train_volumes": sum(row["split"] == "train" for row in volume_records),
        "num_validation_volumes": sum(
            row["split"] == "validation" for row in volume_records
        ),
        "num_train_slices": sum(
            len(row["slices"]) for row in volume_records if row["split"] == "train"
        ),
        "num_validation_slices": sum(
            len(row["slices"])
            for row in volume_records
            if row["split"] == "validation"
        ),
        "volumes": volume_records,
    }
    _write_json(output_root / "split_manifest.json", manifest)
    return manifest


def _relative_source_path(root: Path, pattern: str, case: str) -> tuple[Path, str]:
    relative = Path(str(pattern).format(case=case))
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError("WaferEM input patterns must resolve below source_root")
    return root / relative, relative.as_posix()


def _h5_dataset_shape(path: Path, key: str) -> tuple[int, ...]:
    import h5py

    with h5py.File(path, "r") as handle:
        if key not in handle:
            raise KeyError(f"Missing HDF5 dataset {key!r} in {path}")
        return tuple(int(value) for value in handle[key].shape)


def _create_ds2_h5(
    source_path: Path,
    destination_path: Path,
    *,
    key: str,
    mode: str,
    source_relative_path: str,
    overwrite: bool,
) -> tuple[int, int, int]:
    import h5py

    source_shape = _h5_dataset_shape(source_path, key)
    if len(source_shape) != 3:
        raise ValueError(f"Expected a ZYX HDF5 volume at {source_path}, got {source_shape}")
    z, height, width = source_shape
    if height % 2 or width % 2:
        raise ValueError(
            f"WaferEM XY dimensions must be even for ds2, got {source_shape}"
        )
    output_shape = (z, height // 2, width // 2)
    if destination_path.is_file() and not overwrite:
        if _h5_dataset_shape(destination_path, key) != output_shape:
            raise ValueError(f"Existing output shape mismatch at {destination_path}")
        return output_shape

    destination_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination_path.with_suffix(
        destination_path.suffix + f".tmp.{os.getpid()}"
    )
    if temporary.exists():
        temporary.unlink()
    try:
        with h5py.File(source_path, "r") as source, h5py.File(
            temporary,
            "w",
        ) as output:
            source_dataset = source[key]
            dtype = np.float32 if mode == "raw_mean2x2" else np.uint32
            chunks = (1, min(256, output_shape[1]), min(256, output_shape[2]))
            destination = output.create_dataset(
                key,
                shape=output_shape,
                dtype=dtype,
                chunks=chunks,
                compression="gzip",
                compression_opts=4,
            )
            destination.attrs["source_file"] = source_relative_path
            destination.attrs["source_key"] = key
            destination.attrs["xy_downsample"] = (
                "mean2x2" if mode == "raw_mean2x2" else "stride2"
            )
            for slice_index in range(z):
                source_slice = np.asarray(source_dataset[slice_index])
                if mode == "raw_mean2x2":
                    destination[slice_index] = source_slice.astype(
                        np.float32,
                        copy=False,
                    ).reshape(
                        output_shape[1],
                        2,
                        output_shape[2],
                        2,
                    ).mean(axis=(1, 3))
                elif mode == "label_stride2":
                    sampled = source_slice[::2, ::2]
                    if sampled.size and int(sampled.max()) > np.iinfo(np.uint32).max:
                        raise ValueError(f"Label ids exceed uint32 in {source_path}")
                    destination[slice_index] = sampled.astype(np.uint32, copy=False)
                else:
                    raise ValueError(f"Unsupported ds2 mode: {mode}")
        temporary.replace(destination_path)
    finally:
        if temporary.exists():
            temporary.unlink()
    return output_shape


def materialize_waferem_ds2(
    source_root: str | Path,
    output_root: str | Path,
    *,
    input_pattern: str = "{case}_inputs.h5",
    label_pattern: str = "{case}_labels.h5",
    h5_key: str = "main",
    overwrite: bool = False,
) -> dict[str, Any]:
    """Materialize all six WaferEM volumes and their formal split manifest.

    Raw XY downsampling uses non-overlapping 2x2 means. Labels use stride-2
    sampling. Five volumes are marked for adaptation and ``wafer4`` is marked
    as final-test-only. Manifest paths are relative to ``output_root``.
    """

    source_root = Path(source_root).expanduser().resolve()
    output_root = Path(output_root).expanduser().resolve()
    if not source_root.is_dir():
        raise NotADirectoryError(source_root)

    volumes: list[dict[str, Any]] = []
    for case in WAFEREM_CASES:
        raw_source, raw_source_relative = _relative_source_path(
            source_root,
            input_pattern,
            case,
        )
        label_source, label_source_relative = _relative_source_path(
            source_root,
            label_pattern,
            case,
        )
        if not raw_source.is_file():
            raise FileNotFoundError(raw_source)
        if not label_source.is_file():
            raise FileNotFoundError(label_source)
        raw_shape = _h5_dataset_shape(raw_source, h5_key)
        label_shape = _h5_dataset_shape(label_source, h5_key)
        if raw_shape != label_shape:
            raise ValueError(
                f"WaferEM raw/label shape mismatch for {case}: "
                f"{raw_shape} vs {label_shape}"
            )

        volume_root = output_root / "volumes" / case
        raw_output = volume_root / "raw.h5"
        label_output = volume_root / "label.h5"
        output_shape = _create_ds2_h5(
            raw_source,
            raw_output,
            key=h5_key,
            mode="raw_mean2x2",
            source_relative_path=raw_source_relative,
            overwrite=overwrite,
        )
        label_output_shape = _create_ds2_h5(
            label_source,
            label_output,
            key=h5_key,
            mode="label_stride2",
            source_relative_path=label_source_relative,
            overwrite=overwrite,
        )
        if output_shape != label_output_shape:
            raise RuntimeError(f"WaferEM ds2 output mismatch for {case}")
        role = "final_test" if case == WAFEREM_FINAL_TEST_CASE else "adaptation"
        volumes.append(
            {
                "volume_id": f"{case}_ds2",
                "case": case,
                "role": role,
                "raw_path": raw_output.relative_to(output_root).as_posix(),
                "label_path": label_output.relative_to(output_root).as_posix(),
                "h5_key": h5_key,
                "shape": list(output_shape),
                "num_slices": int(output_shape[0]),
                "source_raw_path": raw_source_relative,
                "source_label_path": label_source_relative,
            }
        )

    adaptation_volume_ids = [
        f"{case}_ds2" for case in WAFEREM_ADAPTATION_CASES
    ]
    manifest = {
        "dataset": "WaferEM",
        "resolution": "ds2",
        "h5_key": h5_key,
        "raw_downsampling": "non-overlapping 2x2 mean in XY",
        "label_downsampling": "stride 2 in XY",
        "path_bases": {
            "source_paths": "source_root argument",
            "materialized_paths": "manifest directory",
        },
        "adaptation_volume_ids": adaptation_volume_ids,
        "final_test_volume_id": f"{WAFEREM_FINAL_TEST_CASE}_ds2",
        "final_test_policy": "excluded from training, selection, and calibration",
        "volumes": volumes,
    }
    if len(adaptation_volume_ids) != 5:
        raise RuntimeError(
            "WaferEM formal manifest must contain five adaptation volumes"
        )
    if manifest["final_test_volume_id"] in adaptation_volume_ids:
        raise RuntimeError("WaferEM final-test volume leaked into adaptation")
    _write_json(output_root / "waferem_ds2_manifest.json", manifest)
    return manifest


__all__ = [
    "WAFEREM_ADAPTATION_CASES",
    "WAFEREM_CASES",
    "WAFEREM_FINAL_TEST_CASE",
    "component_pseudo_relabel_2d",
    "discover_emneuron_pairs",
    "make_three_channel_target",
    "materialize_emneuron_volume_split",
    "materialize_waferem_ds2",
]
