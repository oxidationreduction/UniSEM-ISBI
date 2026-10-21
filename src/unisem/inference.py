"""Portable inference utilities for three-channel dense predictions."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from pathlib import Path

import numpy as np


CHANNEL_NAMES = ("foreground", "center_distances", "boundary_distances")
HANN_FLOOR = 1.0e-3

FOREGROUND_THRESHOLD = 0.5
CENTER_DISTANCE_THRESHOLD = 0.5
BOUNDARY_DISTANCE_THRESHOLD = 0.5
FOREGROUND_SMOOTHING = 1.0
DISTANCE_SMOOTHING = 1.6
MIN_INSTANCE_AREA = 16


def normalize_raw(image: np.ndarray) -> np.ndarray:
    """Apply the paper's per-slice 1st/99th percentile uint8 normalization."""
    values = np.asarray(image, dtype=np.float32)
    if values.ndim != 2 or not np.isfinite(values).all():
        raise ValueError("Expected one finite 2D grayscale image")
    lo, hi = np.percentile(values, (1.0, 99.0))
    if hi <= lo:
        lo, hi = float(values.min()), float(values.max())
    if hi <= lo:
        return np.zeros(values.shape, dtype=np.uint8)
    return np.round(np.clip((values - lo) / (hi - lo), 0.0, 1.0) * 255.0).astype(np.uint8)


def tile_starts(length: int, tile: int, stride: int, offset: int = 0) -> list[int]:
    """Return deterministic starts that cover an axis and include its far edge."""
    length = int(length)
    tile = int(tile)
    stride = int(stride)
    offset = int(offset)
    if length <= 0:
        raise ValueError("length must be positive")
    if tile <= 0:
        raise ValueError("tile must be positive")
    if stride <= 0:
        raise ValueError("stride must be positive")
    if length <= tile:
        return [0]

    final_start = length - tile
    starts: list[int] = []
    current = max(0, offset)
    while current < length:
        starts.append(min(current, final_start))
        if starts[-1] == final_start:
            break
        current += stride
    return sorted(set(starts))


def hann_window(height: int, width: int) -> np.ndarray:
    """Return a separable Hann window with a nonzero numerical floor."""
    height = int(height)
    width = int(width)
    if height <= 0 or width <= 0:
        raise ValueError("window dimensions must be positive")
    weight_y = np.hanning(max(3, height))[:height]
    weight_x = np.hanning(max(3, width))[:width]
    return np.maximum(np.outer(weight_y, weight_x), HANN_FLOOR).astype(np.float32)


def _prediction_to_chw(
    prediction: np.ndarray | Mapping[str, np.ndarray],
    expected_shape: tuple[int, int],
) -> np.ndarray:
    if isinstance(prediction, Mapping):
        if set(prediction) != set(CHANNEL_NAMES):
            raise ValueError(f"prediction mapping must contain exactly {CHANNEL_NAMES}")
        output = np.stack(
            [np.asarray(prediction[name], dtype=np.float32) for name in CHANNEL_NAMES],
            axis=0,
        )
    else:
        output = np.asarray(prediction, dtype=np.float32)
        expected = (len(CHANNEL_NAMES), *expected_shape)
        if output.shape == expected:
            pass
        elif output.shape == (expected_shape[0], expected_shape[1], len(CHANNEL_NAMES)):
            output = np.moveaxis(output, -1, 0)
    expected = (len(CHANNEL_NAMES), *expected_shape)
    if output.shape != expected:
        raise ValueError(f"prediction shape must be {expected} or {(*expected_shape, 3)}, got {output.shape}")
    if not np.all(np.isfinite(output)):
        raise ValueError("prediction contains non-finite values")
    return output


def overlap_fusion(
    image: np.ndarray,
    predict_tile: Callable[[np.ndarray], np.ndarray | Mapping[str, np.ndarray]],
    *,
    tile_shape: Sequence[int],
    stride: Sequence[int],
    offsets: Sequence[Sequence[int]] = ((0, 0),),
) -> np.ndarray:
    """Run three-channel tiled prediction and fuse overlaps with Hann weights.

    The callback receives a contiguous image tile and may return either a
    channel-first array, a channel-last array, or a mapping keyed by
    ``CHANNEL_NAMES``. The fused output is channel-first float32.
    """
    image = np.asarray(image)
    if image.ndim < 2:
        raise ValueError("image must have at least two dimensions")
    height, width = image.shape[:2]
    if len(tile_shape) != 2 or len(stride) != 2:
        raise ValueError("tile_shape and stride must have two elements")
    tile_height, tile_width = (int(value) for value in tile_shape)
    stride_y, stride_x = (int(value) for value in stride)
    if not offsets:
        raise ValueError("offsets must not be empty")

    accumulator = np.zeros((len(CHANNEL_NAMES), height, width), dtype=np.float32)
    coverage = np.zeros((height, width), dtype=np.float32)
    for offset in offsets:
        if len(offset) != 2:
            raise ValueError("each offset must have two elements")
        offset_y, offset_x = (int(value) for value in offset)
        for y0 in tile_starts(height, tile_height, stride_y, offset_y):
            for x0 in tile_starts(width, tile_width, stride_x, offset_x):
                y1 = min(height, y0 + tile_height)
                x1 = min(width, x0 + tile_width)
                image_tile = np.ascontiguousarray(image[y0:y1, x0:x1, ...])
                prediction = _prediction_to_chw(predict_tile(image_tile), (y1 - y0, x1 - x0))
                weight = hann_window(y1 - y0, x1 - x0)
                accumulator[:, y0:y1, x0:x1] += prediction * weight[None, ...]
                coverage[y0:y1, x0:x1] += weight

    if float(coverage.min()) <= 0.0:
        raise RuntimeError("tile configuration leaves uncovered pixels")
    return (accumulator / coverage[None, ...]).astype(np.float32, copy=False)


def _filter_and_relabel(segmentation: np.ndarray, min_area: int) -> np.ndarray:
    segmentation = np.asarray(segmentation)
    output = np.zeros(segmentation.shape, dtype=np.uint32)
    labels, counts = np.unique(segmentation, return_counts=True)
    next_label = 1
    for label, count in zip(labels.tolist(), counts.tolist()):
        if int(label) == 0 or int(count) < int(min_area):
            continue
        output[segmentation == label] = np.uint32(next_label)
        next_label += 1
    return output


def seeded_watershed(
    prediction: np.ndarray | Mapping[str, np.ndarray],
    *,
    foreground_threshold: float = FOREGROUND_THRESHOLD,
    center_distance_threshold: float = CENTER_DISTANCE_THRESHOLD,
    boundary_distance_threshold: float = BOUNDARY_DISTANCE_THRESHOLD,
    foreground_smoothing: float = FOREGROUND_SMOOTHING,
    distance_smoothing: float = DISTANCE_SMOOTHING,
    min_instance_area: int = MIN_INSTANCE_AREA,
) -> np.ndarray:
    """Decode dense predictions with an explicit fixed watershed preset."""
    from micro_sam.instance_segmentation import InstanceSegmentationWithDecoder

    if isinstance(prediction, Mapping):
        if set(prediction) != set(CHANNEL_NAMES):
            raise ValueError(f"prediction mapping must contain exactly {CHANNEL_NAMES}")
        first = np.asarray(prediction[CHANNEL_NAMES[0]])
        expected_shape = tuple(int(value) for value in first.shape)
    else:
        array = np.asarray(prediction)
        if array.ndim != 3:
            raise ValueError("prediction must be a three-dimensional array")
        expected_shape = tuple(int(value) for value in (array.shape[1:] if array.shape[0] == 3 else array.shape[:2]))
    if len(expected_shape) != 2:
        raise ValueError("prediction channels must be two-dimensional")
    foreground, center, boundary = _prediction_to_chw(prediction, expected_shape)

    instance_generator = object.__new__(InstanceSegmentationWithDecoder)
    InstanceSegmentationWithDecoder.set_state(instance_generator, {
        "foreground": np.ascontiguousarray(foreground, dtype=np.float32),
        "center_distances": np.ascontiguousarray(center, dtype=np.float32),
        "boundary_distances": np.ascontiguousarray(boundary, dtype=np.float32),
    })
    segmentation = InstanceSegmentationWithDecoder.generate(
        instance_generator,
        foreground_threshold=float(foreground_threshold),
        center_distance_threshold=float(center_distance_threshold),
        boundary_distance_threshold=float(boundary_distance_threshold),
        foreground_smoothing=float(foreground_smoothing),
        distance_smoothing=float(distance_smoothing),
        min_size=int(min_instance_area), output_mode="instance_segmentation", n_threads=1,
    )
    return _filter_and_relabel(segmentation, int(min_instance_area))


def write_h5(path: str | Path, datasets: Mapping[str, np.ndarray]) -> None:
    """Write named arrays to a gzip-compressed HDF5 file."""
    import h5py

    if not datasets:
        raise ValueError("datasets must not be empty")
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(output_path, "w") as handle:
        for name, values in datasets.items():
            if not name or "/" in name:
                raise ValueError(f"invalid top-level dataset name: {name!r}")
            array = np.asarray(values)
            if array.ndim < 2:
                raise ValueError(f"dataset {name!r} must have at least two dimensions")
            if any(int(size) <= 0 for size in array.shape):
                raise ValueError(f"dataset {name!r} must not have empty dimensions")
            chunks = (1, *array.shape[1:]) if array.ndim >= 3 else True
            handle.create_dataset(
                name,
                data=array,
                compression="gzip",
                chunks=chunks,
            )
