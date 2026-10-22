from __future__ import annotations

import hashlib
import json
from io import BytesIO
from urllib.parse import quote
from zipfile import ZipFile

import h5py
import numpy as np
import pytest
import tifffile

import unisem.demo as demo
from unisem.demo import download_demo_assets, export_results, label_array, load_input, overlay_image, select_input
from unisem.inference import normalize_raw


def test_tiff_stack_retains_shape_and_values(tmp_path):
    data = np.arange(3 * 12 * 14, dtype=np.uint16).reshape(3, 12, 14)
    path = tmp_path / "input.tif"
    tifffile.imwrite(path, data, photometric="minisblack", metadata={"axes": "ZYX"})
    actual, info = load_input(path)
    np.testing.assert_array_equal(actual, data)
    assert info["axes"] == "ZYX" and info["dtype"] == "uint16"
    np.testing.assert_array_equal(select_input(actual, index=1), data[1])
    assert select_input(actual, scope="stack").shape == data.shape


def test_rgb_tiff_rejected_even_when_claimed_zyx(tmp_path):
    path = tmp_path / "rgb.tif"
    tifffile.imwrite(path, np.zeros((12, 14, 3), dtype=np.uint8), photometric="rgb")
    with pytest.raises(ValueError, match="RGB"):
        load_input(path, axes="ZYX")


def test_hdf5_key_and_ambiguous_shape_are_explicit(tmp_path):
    path = tmp_path / "input.h5"
    with h5py.File(path, "w") as handle:
        handle["raw/data"] = np.ones((12, 14, 3), dtype=np.uint8)
    with pytest.raises(KeyError, match="main"):
        load_input(path)
    with pytest.raises(ValueError, match="RGB"):
        load_input(path, key="raw/data")
    data, info = load_input(path, key="raw/data", axes="ZYX")
    assert data.shape == (12, 14, 3) and info["hdf5_key"] == "raw/data"


@pytest.mark.parametrize("data", [np.zeros((0, 8)), np.full((8, 8), np.nan), np.zeros((1, 2, 3, 4))])
def test_invalid_input_is_rejected(tmp_path, data):
    path = tmp_path / "bad.h5"
    with h5py.File(path, "w") as handle:
        handle["main"] = data
    with pytest.raises(ValueError):
        load_input(path)


def test_export_has_exact_uint32_labels_and_real_results(tmp_path):
    raw = np.arange(12 * 14, dtype=np.uint8).reshape(12, 14)
    labels = np.zeros(raw.shape, dtype=np.uint64)
    labels[3:8, 4:10] = 913
    result = tmp_path / "prediction.h5"
    with h5py.File(result, "w") as handle:
        handle["final_merged_segmentation"] = labels
    exported = export_results(raw, result, tmp_path / "download", {"model_revision": "a" * 40})
    output = tifffile.imread(exported["labels"])
    np.testing.assert_array_equal(output, labels)
    assert output.dtype == np.uint32
    assert json.loads(exported["metadata"].read_text())["output_shape"] == list(raw.shape)
    with ZipFile(exported["download"]) as archive:
        assert set(archive.namelist()) == {"instances.tif", "preview.png", "run.json", "prediction.h5"}
    np.testing.assert_array_equal(overlay_image(raw, labels, opacity=0), np.repeat(normalize_raw(raw)[..., None], 3, axis=-1))


def test_output_mismatch_does_not_export_plausible_labels(tmp_path):
    result = tmp_path / "prediction.h5"
    with h5py.File(result, "w") as handle:
        handle["final_merged_segmentation"] = np.zeros((3, 3), dtype=np.uint32)
    with pytest.raises(ValueError, match="does not match"):
        export_results(np.zeros((4, 4)), result, tmp_path / "download", {})


@pytest.mark.parametrize("labels", [np.full((2, 2), -1), np.full((2, 2), 2**32, dtype=np.uint64), np.ones((2, 2), dtype=float)])
def test_labels_cannot_silently_wrap_or_truncate(labels):
    with pytest.raises(ValueError, match="nonnegative integers"):
        label_array(labels, labels.shape)


def test_download_rejects_branch_and_existing_wrong_asset(tmp_path):
    with pytest.raises(ValueError, match="revision"):
        download_demo_assets(tmp_path, "main")
    (tmp_path / "unisem_stage1_emneuron_vitl.pt").write_bytes(b"wrong model")
    with pytest.raises(ValueError, match="Checksum mismatch"):
        download_demo_assets(tmp_path, "a" * 40)


@pytest.mark.parametrize("provider", [None, "huggingface", "modelscope"], ids=["default-huggingface", "explicit-huggingface", "explicit-modelscope"])
def test_download_uses_fixed_provider_urls_and_checks_both_assets(tmp_path, monkeypatch, provider):
    revision = "a" * 40
    # Tiny bytes exercise download, hashing and final filenames without model I/O.
    contents = {"model with space.pt": b"model", "rf+fixed.joblib": b"random forest"}
    monkeypatch.setattr(demo, "DEMO_ASSETS", {name: hashlib.sha256(data).hexdigest() for name, data in contents.items()})
    expected_urls = []
    for name in contents:
        encoded = quote(name, safe="")
        expected_urls.append(
            f"https://modelscope.cn/api/v1/models/dreamBuilder/UniSEM/repo?Revision={revision}&FilePath={encoded}"
            if provider == "modelscope" else
            f"https://huggingface.co/dreamBuilder/UniSEM/resolve/{revision}/{encoded}"
        )
    calls = []

    def response(url, timeout):
        assert timeout == 180
        calls.append(url)
        index = expected_urls.index(url)
        return BytesIO(list(contents.values())[index])

    monkeypatch.setattr(demo, "urlopen", response)
    downloaded = download_demo_assets(tmp_path, revision) if provider is None else download_demo_assets(tmp_path, revision, provider=provider)
    assert calls == expected_urls
    assert {name: path.read_bytes() for name, path in downloaded.items()} == contents
    assert not list(tmp_path.glob("*.part"))


@pytest.mark.parametrize("provider", ["unknown", "https://other.example", None])
def test_invalid_download_provider_does_not_access_network(tmp_path, monkeypatch, provider):
    def unexpected_network(*args, **kwargs):
        pytest.fail("An invalid provider must be rejected before network access")

    monkeypatch.setattr(demo, "urlopen", unexpected_network)
    destination = tmp_path / "assets"
    with pytest.raises(ValueError, match="Provider"):
        download_demo_assets(destination, "a" * 40, provider=provider)
    assert not destination.exists()
