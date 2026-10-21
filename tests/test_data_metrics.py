from __future__ import annotations

import json
import math
from pathlib import Path

import h5py
import numpy as np
import tifffile

from unisem.data import (
    WAFEREM_ADAPTATION_CASES,
    WAFEREM_CASES,
    component_pseudo_relabel_2d,
    make_three_channel_target,
    materialize_emneuron_volume_split,
    materialize_waferem_ds2,
)
from unisem.metrics import aggregate_slice_metrics, cremi_score


def _write_h5(path: Path, data: np.ndarray) -> None:
    with h5py.File(path, "w") as handle:
        handle.create_dataset("main", data=data)


def test_materialize_emneuron_volume_split(tmp_path: Path) -> None:
    source_root = tmp_path / "emneuron_source"
    output_root = tmp_path / "emneuron_prepared"
    source_root.mkdir()

    source_stacks: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for volume_index in range(3):
        volume_root = source_root / f"volume_{volume_index}"
        volume_root.mkdir()
        raw = (
            np.arange(2 * 8 * 10, dtype=np.uint16).reshape(2, 8, 10)
            + volume_index * 1000
        )
        labels = np.zeros((2, 8, 10), dtype=np.uint32)
        labels[:, 1:4, 1:5] = volume_index + 1
        labels[:, 5:7, 6:9] = volume_index + 11
        tifffile.imwrite(volume_root / "sample.tif", raw)
        tifffile.imwrite(volume_root / "sample_MaskIns.tif", labels)
        source_stacks[f"volume_{volume_index}_sample"] = (raw, labels)

    manifest = materialize_emneuron_volume_split(
        source_root,
        output_root,
        validation_fraction=1.0 / 3.0,
        seed=12345,
    )

    assert manifest["split_unit"] == "volume"
    assert manifest["seed"] == 12345
    assert manifest["num_volumes"] == 3
    assert manifest["num_train_volumes"] == 2
    assert manifest["num_validation_volumes"] == 1
    assert manifest["num_train_slices"] == 4
    assert manifest["num_validation_slices"] == 2

    persisted = json.loads(
        (output_root / "split_manifest.json").read_text(encoding="ascii")
    )
    assert persisted == manifest

    for volume in manifest["volumes"]:
        expected_raw, expected_labels = source_stacks[volume["case_id"]]
        split_directories = {
            Path(row["image_path"]).parts[0] for row in volume["slices"]
        }
        expected_image_directory = (
            "imagesVal" if volume["split"] == "validation" else "imagesTr"
        )
        assert split_directories == {expected_image_directory}
        for row in volume["slices"]:
            z = row["source_z"]
            image = tifffile.imread(output_root / row["image_path"])
            labels = tifffile.imread(output_root / row["label_path"])
            np.testing.assert_array_equal(image, expected_raw[z])
            np.testing.assert_array_equal(labels, expected_labels[z])


def test_materialize_waferem_ds2_and_formal_roles(tmp_path: Path) -> None:
    source_root = tmp_path / "waferem_source"
    output_root = tmp_path / "waferem_prepared"
    source_root.mkdir()

    expected_raw: dict[str, np.ndarray] = {}
    expected_labels: dict[str, np.ndarray] = {}
    for case_index, case in enumerate(WAFEREM_CASES):
        raw = (
            np.arange(2 * 6 * 8, dtype=np.float32).reshape(2, 6, 8)
            + case_index * 100
        )
        labels = np.zeros((2, 6, 8), dtype=np.uint32)
        labels[:, :3, :4] = case_index + 1
        labels[:, 3:, 4:] = case_index + 21
        _write_h5(source_root / f"{case}_inputs.h5", raw)
        _write_h5(source_root / f"{case}_labels.h5", labels)
        expected_raw[case] = raw.reshape(2, 3, 2, 4, 2).mean(axis=(2, 4))
        expected_labels[case] = labels[:, ::2, ::2]

    manifest = materialize_waferem_ds2(source_root, output_root)

    assert manifest["adaptation_volume_ids"] == [
        f"{case}_ds2" for case in WAFEREM_ADAPTATION_CASES
    ]
    assert manifest["final_test_volume_id"] == "wafer4_ds2"
    assert manifest["final_test_volume_id"] not in manifest["adaptation_volume_ids"]
    assert len(manifest["volumes"]) == 6

    persisted = json.loads(
        (output_root / "waferem_ds2_manifest.json").read_text(encoding="ascii")
    )
    assert persisted == manifest

    for volume in manifest["volumes"]:
        case = volume["case"]
        expected_role = "final_test" if case == "wafer4" else "adaptation"
        assert volume["role"] == expected_role
        assert volume["shape"] == [2, 3, 4]
        with h5py.File(output_root / volume["raw_path"], "r") as handle:
            np.testing.assert_allclose(handle["main"][:], expected_raw[case])
            assert handle["main"].attrs["xy_downsample"] == "mean2x2"
        with h5py.File(output_root / volume["label_path"], "r") as handle:
            np.testing.assert_array_equal(handle["main"][:], expected_labels[case])
            assert handle["main"].attrs["xy_downsample"] == "stride2"


def test_component_pseudo_and_three_channel_target() -> None:
    labels = np.zeros((32, 40), dtype=np.uint32)
    labels[2:10, 3:12] = 7
    labels[20:29, 25:36] = 7
    labels[13:18, 15:21] = 9
    labels[0, 39] = 9

    relabeled = component_pseudo_relabel_2d(labels, min_component_area=2)

    assert relabeled.dtype == np.uint32
    assert set(np.unique(relabeled)) == {0, 1, 2, 3}
    assert relabeled[4, 5] != relabeled[24, 30]
    assert relabeled[15, 18] != 0
    assert relabeled[0, 39] == 0

    target = make_three_channel_target(
        labels,
        component_pseudo=True,
        min_component_area=2,
        min_size=1,
    )

    assert target.shape == (3, *labels.shape)
    assert target.dtype == np.float32
    assert np.isfinite(target).all()
    np.testing.assert_array_equal(target[0], relabeled > 0)
    assert np.all((target[1:] >= 0.0) & (target[1:] <= 1.0))
    assert target[1, 5, 6] < target[1, 2, 3]
    assert target[2, 5, 6] < target[2, 2, 3]


def test_cremi_aggregation_keeps_both_named_semantics() -> None:
    rows = [
        {
            "voi_split": 0.75,
            "voi_merge": 0.25,
            "adapted_rand_error": 1.0,
            "rand_index": 0.0,
            "cremi": cremi_score(0.75, 0.25, 1.0),
        },
        {
            "voi_split": 5.0,
            "voi_merge": 4.0,
            "adapted_rand_error": 1.0 / 9.0,
            "rand_index": 8.0 / 9.0,
            "cremi": cremi_score(5.0, 4.0, 1.0 / 9.0),
        },
    ]

    aggregate = aggregate_slice_metrics(rows)

    assert aggregate["num_slices"] == 2
    assert aggregate["mean_voi_sum"] == 5.0
    assert aggregate["mean_adapted_rand_error"] == 5.0 / 9.0
    assert aggregate["mean_per_slice_cremi"] == 1.0
    assert aggregate["cremi_from_aggregate_components"] == math.sqrt(25.0 / 9.0)
    assert (
        aggregate["cremi_from_aggregate_components"]
        != aggregate["mean_per_slice_cremi"]
    )
