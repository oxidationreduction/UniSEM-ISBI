from __future__ import annotations

import json
from pathlib import Path

import h5py
import numpy as np
import tifffile

from unisem.training import _TiffSliceDataset, _WaferSliceDataset, profile_from_protocol


ROOT = Path(__file__).resolve().parents[1]


def test_training_profiles_and_prepared_tiff_dataset(tmp_path: Path) -> None:
    protocol = json.loads((ROOT / "configs" / "paper_protocol.json").read_text(encoding="utf-8"))
    assert profile_from_protocol(protocol, "emneuron-stage1")["freeze_policy"] == "stage1"
    assert profile_from_protocol(protocol, "waferem-lightft")["batch_size"] == 4
    assert profile_from_protocol(protocol, "waferem-fullft")["batch_size"] == 2

    (tmp_path / "imagesTr").mkdir()
    (tmp_path / "instancesTr").mkdir()
    raw = np.arange(64 * 64, dtype=np.uint16).reshape(64, 64)
    labels = np.zeros((64, 64), dtype=np.uint32)
    labels[8:32, 8:32] = 1
    labels[36:56, 36:56] = 2
    tifffile.imwrite(tmp_path / "imagesTr" / "case_0000.tif", raw)
    tifffile.imwrite(tmp_path / "instancesTr" / "case.tif", labels)

    image, target = _TiffSliceDataset(tmp_path, (32, 32), train=True)[0]
    assert tuple(image.shape) == (1, 32, 32)
    assert tuple(target.shape) == (3, 32, 32)
    assert np.isfinite(image.numpy()).all()
    assert np.isfinite(target.numpy()).all()

    wafer_root = tmp_path / "wafer"
    volume_root = wafer_root / "volumes" / "wafer25"
    volume_root.mkdir(parents=True)
    with h5py.File(volume_root / "raw.h5", "w") as handle:
        handle.create_dataset("main", data=raw[None])
    with h5py.File(volume_root / "label.h5", "w") as handle:
        handle.create_dataset("main", data=labels[None])
    (wafer_root / "waferem_ds2_manifest.json").write_text(
        json.dumps(
            {
                "volumes": [
                    {
                        "volume_id": "wafer25_ds2",
                        "raw_path": "volumes/wafer25/raw.h5",
                        "label_path": "volumes/wafer25/label.h5",
                        "h5_key": "main",
                    }
                ]
            }
        ),
        encoding="ascii",
    )
    wafer_image, wafer_target = _WaferSliceDataset(
        wafer_root,
        ["wafer25_ds2"],
        (32, 32),
        train=False,
    )[0]
    assert tuple(wafer_image.shape) == (1, 32, 32)
    assert tuple(wafer_target.shape) == (3, 32, 32)
