from __future__ import annotations

import hashlib
import json
from http.client import IncompleteRead
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


class HTTPBytes(BytesIO):
    def __init__(self, data, status=200, headers=None, failure=None):
        super().__init__(data)
        self.status = status
        self.headers = headers or {}
        self.failure = failure

    def read(self, size=-1):
        data = super().read(size)
        if not data and self.failure is not None:
            raise self.failure
        return data


@pytest.fixture
def tiny_model(monkeypatch):
    content = b"abcdef"
    monkeypatch.setattr(demo, "DEMO_ASSETS", {"model.pt": hashlib.sha256(content).hexdigest()})
    monkeypatch.setattr(demo, "sleep", lambda _: None)
    return content


@pytest.mark.parametrize("failure", [None, BrokenPipeError("connection interrupted"), IncompleteRead(b"", 3)], ids=["truncated-content-length", "broken-pipe", "incomplete-read"])
def test_download_resumes_partial_response(tmp_path, monkeypatch, tiny_model, failure):
    calls = []
    replies = [
        HTTPBytes(b"abc", headers={"Content-Length": "6"}, failure=failure),
        HTTPBytes(b"def", status=206, headers={"Content-Length": "3", "Content-Range": "bytes 3-5/6"}),
    ]

    def respond(request, timeout):
        calls.append(request)
        return replies.pop(0)

    monkeypatch.setattr(demo, "urlopen", respond)
    result = download_demo_assets(tmp_path, "a" * 40, provider="modelscope")
    assert isinstance(calls[0], str)
    assert calls[1].get_header("Range") == "bytes=3-"
    assert result["model.pt"].read_bytes() == tiny_model
    assert not (tmp_path / "model.pt.part").exists()


def test_download_restarts_when_server_ignores_range(tmp_path, monkeypatch, tiny_model):
    (tmp_path / "model.pt.part").write_bytes(b"abc")

    def respond(request, timeout):
        assert request.get_header("Range") == "bytes=3-"
        return HTTPBytes(tiny_model, headers={"Content-Length": "6"})

    monkeypatch.setattr(demo, "urlopen", respond)
    result = download_demo_assets(tmp_path, "a" * 40)
    assert result["model.pt"].read_bytes() == tiny_model


def test_download_continues_when_range_response_is_only_part_of_file(tmp_path, monkeypatch, tiny_model):
    (tmp_path / "model.pt.part").write_bytes(b"abc")
    replies = [
        HTTPBytes(b"de", status=206, headers={"Content-Length": "2", "Content-Range": "bytes 3-4/6"}),
        HTTPBytes(b"f", status=206, headers={"Content-Length": "1", "Content-Range": "bytes 5-5/6"}),
    ]
    offsets = []

    def respond(request, timeout):
        offsets.append(request.get_header("Range"))
        return replies.pop(0)

    monkeypatch.setattr(demo, "urlopen", respond)
    assert download_demo_assets(tmp_path, "a" * 40)["model.pt"].read_bytes() == tiny_model
    assert offsets == ["bytes=3-", "bytes=5-"]


@pytest.mark.parametrize("content_range", ["bytes 2-4/6", "bytes 3-2/6", "not-a-range"])
def test_download_rejects_wrong_resume_offset_before_appending(tmp_path, monkeypatch, tiny_model, content_range):
    partial = tmp_path / "model.pt.part"
    partial.write_bytes(b"abc")
    monkeypatch.setattr(demo, "urlopen", lambda *args, **kwargs: HTTPBytes(b"def", status=206, headers={"Content-Range": content_range}))
    with pytest.raises(ValueError, match="Content-Range"):
        download_demo_assets(tmp_path, "a" * 40)
    assert partial.read_bytes() == b"abc"
    assert not (tmp_path / "model.pt").exists()


def test_download_failures_have_bounded_retries_and_keep_partial(tmp_path, monkeypatch, tiny_model):
    calls = []

    def respond(request, timeout):
        calls.append(request)
        raise BrokenPipeError("connection interrupted")

    partial = tmp_path / "model.pt.part"
    partial.write_bytes(b"abc")
    monkeypatch.setattr(demo, "urlopen", respond)
    with pytest.raises(RuntimeError, match="after 3 attempts for model.pt.part"):
        download_demo_assets(tmp_path, "a" * 40)
    assert len(calls) == 3
    assert partial.read_bytes() == b"abc"
    assert not (tmp_path / "model.pt").exists()


def test_complete_partial_is_verified_without_another_request(tmp_path, monkeypatch, tiny_model):
    (tmp_path / "model.pt.part").write_bytes(tiny_model)
    monkeypatch.setattr(demo, "urlopen", lambda *args, **kwargs: pytest.fail("A complete verified partial needs no network"))
    assert download_demo_assets(tmp_path, "a" * 40)["model.pt"].read_bytes() == tiny_model


def test_download_never_promotes_wrong_checksum(tmp_path, monkeypatch, tiny_model):
    monkeypatch.setattr(demo, "urlopen", lambda *args, **kwargs: HTTPBytes(b"wrong!", headers={"Content-Length": "6"}))
    with pytest.raises(ValueError, match="Checksum mismatch"):
        download_demo_assets(tmp_path, "a" * 40)
    assert not (tmp_path / "model.pt").exists()
