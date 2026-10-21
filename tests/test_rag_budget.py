from __future__ import annotations

import argparse
import json
from pathlib import Path

import h5py
import joblib
import numpy as np
import pytest

from unisem.cli import command_train_rag


def _inputs(root: Path) -> dict:
    from test_cli import _rag_arrays

    raw, gt, _, predictions = _rag_arrays()
    for name, array in (("raw", raw), ("gt", gt)):
        array = array[0] if array.ndim == 3 else array
        with h5py.File(root / f"{name}.h5", "w") as handle:
            handle.create_dataset("main", data=np.stack([array] * 3))
    with h5py.File(root / "prediction.h5", "w") as handle:
        for name, array in predictions.items():
            array = array[0] if array.ndim == 3 else array
            handle.create_dataset(name, data=np.stack([array] * 3))
    return {"dataset": "waferem", "label_budget": 2, "label_z_lists": {"wafer25_ds2": [0, 2]},
            "records": [{"raw": "raw.h5", "ground_truth": "gt.h5", "prediction": "prediction.h5",
                         "volume": "wafer25_ds2", "split": "train", "z_indices": [0, 2]}]}


def _run(root: Path, manifest: dict) -> dict:
    path = root / "manifest.json"
    path.write_text(json.dumps(manifest))
    return command_train_rag(argparse.Namespace(manifest=str(path), output=str(root / "rf.joblib"), workers=1))


def test_rf_fits_only_explicit_labels_and_records_budget(tmp_path: Path) -> None:
    manifest = _inputs(tmp_path)
    result = _run(tmp_path, manifest)
    assert result["label_slices"] == 2
    assert result["edges"] == 4
    assert result["positive_edges"] == 2
    assert result["neural_budget_membership_checked"] is True
    payload = joblib.load(tmp_path / "rf.joblib")
    assert payload["training_metadata"]["label_z_lists"] == {"wafer25_ds2": [0, 2]}


@pytest.mark.parametrize("invalid", ["raw_gt_depth", "prediction_depth", "prediction_xy", "duplicate", "out_of_range", "test_split", "test_volume", "missing_indices", "outside_neural_budget"])
def test_rf_rejects_shapes_test_data_and_invalid_budgets(tmp_path: Path, invalid: str) -> None:
    manifest = _inputs(tmp_path)
    record = manifest["records"][0]
    if invalid == "raw_gt_depth":
        with h5py.File(tmp_path / "gt.h5", "r+") as handle:
            values = handle["main"][:2]
            del handle["main"]
            handle.create_dataset("main", data=values)
    elif invalid.startswith("prediction_"):
        with h5py.File(tmp_path / "prediction.h5", "r+") as handle:
            name = "foreground_tile512_overlap"
            values = handle[name][:2] if invalid == "prediction_depth" else handle[name][:, :-1]
            del handle[name]
            handle.create_dataset(name, data=values)
    elif invalid == "duplicate":
        record["z_indices"] = [0, 0]
    elif invalid == "out_of_range":
        record["z_indices"] = [0, 3]
    elif invalid == "test_split":
        record["split"] = "test"
    elif invalid == "test_volume":
        record["volume"] = "wafer4_ds2"
    elif invalid == "missing_indices":
        del record["z_indices"]
    else:
        record["z_indices"] = [0, 1]
    with pytest.raises(ValueError):
        _run(tmp_path, manifest)
    assert not (tmp_path / "rf.joblib").exists()
