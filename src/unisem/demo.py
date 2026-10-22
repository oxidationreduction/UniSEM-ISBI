"""Input validation and result export for the Colab demonstration.

Neural inference and instance generation remain in ``unisem infer``.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from urllib.parse import quote
from urllib.request import urlopen
from zipfile import ZIP_DEFLATED, ZipFile

import numpy as np


DEMO_ASSETS = {
    "unisem_stage1_emneuron_vitl.pt": "62821463f1d11c5adcc172337b78cf7fab10665add023de3aaf35373a7f2c1fe",
    "unisem_stage1_emneuron_full55_rf.joblib": "13c35acb2d085e292c4b6a44c6bd7840fb9a6839fa7cab1d5d5f63ab8ea3e4df",
}


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def download_demo_assets(
    destination: str | Path,
    revision: str,
    *,
    provider: str = "huggingface",
) -> dict[str, Path]:
    """Download the fixed public pair from an approved provider and verify hashes."""
    if provider not in ("huggingface", "modelscope"):
        raise ValueError("Provider must be huggingface or modelscope")
    if len(revision) != 40 or any(character not in "0123456789abcdef" for character in revision):
        raise ValueError("Use the published 40-character model revision, not a moving branch")
    directory = Path(destination)
    directory.mkdir(parents=True, exist_ok=True)
    result = {}
    for filename, expected in DEMO_ASSETS.items():
        path = directory / filename
        if path.exists():
            if file_sha256(path) != expected:
                raise ValueError(f"Checksum mismatch for existing asset {filename}; replace it with the published file")
        else:
            partial = path.with_suffix(path.suffix + ".part")
            encoded_filename = quote(filename, safe="")
            if provider == "huggingface":
                url = f"https://huggingface.co/dreamBuilder/UniSEM/resolve/{revision}/{encoded_filename}"
            else:
                url = f"https://modelscope.cn/api/v1/models/dreamBuilder/UniSEM/repo?Revision={revision}&FilePath={encoded_filename}"
            with urlopen(url, timeout=180) as response, partial.open("wb") as output:
                while chunk := response.read(4 * 1024 * 1024):
                    output.write(chunk)
            if file_sha256(partial) != expected:
                raise ValueError(f"Checksum mismatch after downloading {filename}; the asset was not loaded")
            partial.replace(path)
        result[filename] = path
    return result


def load_input(path: str | Path, key: str = "main", axes: str = "auto") -> tuple[np.ndarray, dict]:
    """Read a finite grayscale YX image or explicitly identified ZYX stack.

    RGB TIFF metadata is always rejected. Ambiguous HDF5 arrays ending in three
    or four elements require an explicit ZYX selection instead of a silent guess.
    """
    source = Path(path)
    if axes not in {"auto", "YX", "ZYX"}:
        raise ValueError("Axes must be auto, YX or ZYX")
    recorded_axes = None
    if source.suffix.lower() in {".tif", ".tiff"}:
        import tifffile

        with tifffile.TiffFile(source) as handle:
            series = handle.series[0]
            recorded_axes = series.axes
            if any(page.photometric == 2 for page in series.pages) or "S" in recorded_axes or "C" in recorded_axes:
                raise ValueError("RGB/channel TIFF images are unsupported; upload grayscale EM data")
            values = np.asarray(series.asarray())
    elif source.suffix.lower() in {".h5", ".hdf5"}:
        import h5py

        with h5py.File(source, "r") as handle:
            if key not in handle or not isinstance(handle[key], h5py.Dataset):
                raise KeyError(f"HDF5 dataset {key!r} was not found")
            values = np.asarray(handle[key])
    else:
        raise ValueError("Upload a .tif, .tiff, .h5 or .hdf5 file")
    if values.ndim not in {2, 3} or any(size == 0 for size in values.shape):
        raise ValueError(f"Expected a nonempty YX image or ZYX stack; received {values.shape}")
    if values.dtype.kind not in "buif" or not np.isfinite(values).all():
        raise ValueError("The image must contain finite real numeric values")
    expected_axes = "YX" if values.ndim == 2 else "ZYX"
    if axes != "auto" and axes != expected_axes:
        raise ValueError(f"{axes} does not match array shape {values.shape}")
    if values.ndim == 3 and values.shape[-1] in {3, 4} and axes == "auto":
        raise ValueError("This shape could be RGB; confirm ZYX explicitly only for a grayscale stack")
    return values, {
        "filename": source.name,
        "shape": list(values.shape),
        "dtype": str(values.dtype),
        "axes": expected_axes,
        "file_axes": recorded_axes,
        "hdf5_key": key if source.suffix.lower() in {".h5", ".hdf5"} else None,
    }


def select_input(array: np.ndarray, scope: str = "slice", index: int = 0) -> np.ndarray:
    if scope not in {"slice", "stack"}:
        raise ValueError("Choose slice or stack")
    if array.ndim == 2:
        return array
    if not 0 <= index < len(array):
        raise ValueError(f"Slice index must be between 0 and {len(array) - 1}")
    return array[index] if scope == "slice" else array


def label_array(values: np.ndarray, shape: tuple[int, ...]) -> np.ndarray:
    labels = np.asarray(values)
    if labels.shape != shape:
        raise ValueError(f"Prediction shape {labels.shape} does not match input {shape}")
    if labels.dtype.kind not in "ui" or labels.min() < 0 or labels.max() > np.iinfo(np.uint32).max:
        raise ValueError("Instance labels must be nonnegative integers fitting uint32")
    return labels.astype(np.uint32, copy=False)


def overlay_image(raw: np.ndarray, labels: np.ndarray, opacity: float = 0.65, mode: str = "boundaries") -> np.ndarray:
    """Create a deterministic preview without changing model predictions."""
    from .inference import normalize_raw

    if raw.ndim != 2 or not 0 <= opacity <= 1 or mode not in {"boundaries", "instances"}:
        raise ValueError("Preview requires a 2D image, opacity in [0, 1], and boundaries/instances mode")
    labels = label_array(labels, raw.shape)
    background = np.repeat(normalize_raw(raw)[..., None], 3, axis=2).astype(np.float32)
    identifiers = labels.astype(np.uint64)
    colors = np.stack([48 + (identifiers * factor) % 208 for factor in (73, 151, 199)], axis=-1)
    mask = labels != 0
    if mode == "boundaries":
        edges = np.zeros(labels.shape, dtype=bool)
        edges[1:] |= labels[1:] != labels[:-1]
        edges[:-1] |= labels[:-1] != labels[1:]
        edges[:, 1:] |= labels[:, 1:] != labels[:, :-1]
        edges[:, :-1] |= labels[:, :-1] != labels[:, 1:]
        mask &= edges
    background[mask] = (1 - opacity) * background[mask] + opacity * colors[mask]
    return np.round(background).astype(np.uint8)


def export_results(raw: np.ndarray, result_h5: str | Path, destination: str | Path, run_info: dict) -> dict[str, Path]:
    """Package actual inference output and a matched raw/overlay preview."""
    import h5py
    import tifffile
    from PIL import Image

    from .inference import normalize_raw

    directory = Path(destination)
    directory.mkdir(parents=True, exist_ok=True)
    result_h5 = Path(result_h5)
    with h5py.File(result_h5, "r") as handle:
        if "final_merged_segmentation" not in handle:
            raise KeyError("Inference did not produce final_merged_segmentation")
        labels = label_array(np.asarray(handle["final_merged_segmentation"]), raw.shape)
    label_file = directory / "instances.tif"
    tifffile.imwrite(label_file, labels, photometric="minisblack", metadata={"axes": "YX" if labels.ndim == 2 else "ZYX"})
    preview_raw = raw if raw.ndim == 2 else raw[0]
    preview_labels = labels if labels.ndim == 2 else labels[0]
    side_by_side = np.concatenate((np.repeat(normalize_raw(preview_raw)[..., None], 3, axis=-1), overlay_image(preview_raw, preview_labels)), axis=1)
    preview_file = directory / "preview.png"
    Image.fromarray(side_by_side).save(preview_file)
    info_file = directory / "run.json"
    info = dict(run_info, output_shape=list(labels.shape), label_dtype="uint32", instance_dimension="2D per slice", preview_output_slice=0)
    info_file.write_text(json.dumps(info, indent=2) + "\n", encoding="utf-8")
    bundle = directory / "unisem-results.zip"
    with ZipFile(bundle, "w", compression=ZIP_DEFLATED) as archive:
        for path in (label_file, preview_file, info_file, result_h5):
            archive.write(path, path.name)
    return {"labels": label_file, "hdf5": result_h5, "preview": preview_file, "metadata": info_file, "download": bundle}
