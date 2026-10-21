from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import h5py
import numpy as np
import pytest
import tifffile
import torch

from unisem import cli
from unisem.data import WAFEREM_CASES
from unisem.inference import CHANNEL_NAMES
from unisem.rag import SCALE_NAMES


PUBLIC_COMMANDS = (
    "prepare-data",
    "train",
    "infer",
    "train-rag",
    "evaluate",
    "inspect-checkpoint",
    "sanitize-checkpoint",
    "smoke",
)
PACKAGE_ROOT = Path(__file__).resolve().parents[1]
PROTOCOL_PATH = PACKAGE_ROOT / "configs" / "paper_protocol.json"


@pytest.fixture(autouse=True)
def _limit_cpu_threads(monkeypatch: pytest.MonkeyPatch) -> None:
    for variable in (
        "OMP_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "MKL_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
    ):
        monkeypatch.setenv(variable, "1")


def _run_cli(capsys: pytest.CaptureFixture[str], *arguments: object) -> dict[str, object]:
    exit_code = cli.main([str(argument) for argument in arguments])
    captured = capsys.readouterr()
    assert exit_code == 0
    assert captured.err == ""
    return json.loads(captured.out)


def _write_h5(path: Path, data: np.ndarray, key: str = "main") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(path, "w") as handle:
        handle.create_dataset(key, data=data)


def _legacy_checkpoint(path: Path) -> Path:
    torch.save(
        {
            "model_type": "vit_b",
            "model_state": {"image_encoder.weight": torch.ones(1)},
            "decoder_state": {
                "baseline_decoder.weight": torch.ones(1),
                "hq_residual.weight": torch.zeros(1),
            },
        },
        path,
    )
    return path


def _rag_arrays() -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, np.ndarray]]:
    height, width = 24, 48
    fragments = np.zeros((height, width), dtype=np.uint32)
    fragments[:, :16] = 1
    fragments[:, 16:32] = 2
    fragments[:, 32:] = 3
    ground_truth = np.where(fragments < 3, 1, 2).astype(np.uint32)
    yy, xx = np.mgrid[:height, :width]
    raw = (0.25 * yy / (height - 1) + 0.75 * xx / (width - 1)).astype(np.float32)

    prediction_datasets: dict[str, np.ndarray] = {}
    for scale_index, scale in enumerate(SCALE_NAMES):
        offset = np.float32(scale_index * 0.05)
        prediction_datasets[f"foreground_{scale}"] = np.clip(
            0.8 - 0.1 * raw + offset,
            0.0,
            1.0,
        )[None]
        prediction_datasets[f"center_distances_{scale}"] = np.clip(
            0.2 + 0.3 * raw + offset,
            0.0,
            1.0,
        )[None]
        prediction_datasets[f"boundary_distances_{scale}"] = np.clip(
            0.1 + 0.5 * raw + offset,
            0.0,
            1.0,
        )[None]
        segmentation = fragments if scale == "tile384_overlap" else ground_truth
        prediction_datasets[f"segmentation_{scale}"] = segmentation[None]
    return raw[None], ground_truth[None], fragments, prediction_datasets


def test_parser_and_help_cover_all_public_commands(
    capsys: pytest.CaptureFixture[str],
) -> None:
    parser = cli.build_parser()
    with pytest.raises(SystemExit) as root_exit:
        parser.parse_args(["--help"])
    assert root_exit.value.code == 0
    root_help = capsys.readouterr().out
    for command in PUBLIC_COMMANDS:
        assert command in root_help

    for command in PUBLIC_COMMANDS:
        with pytest.raises(SystemExit) as command_exit:
            parser.parse_args([command, "--help"])
        assert command_exit.value.code == 0
        command_help = capsys.readouterr().out
        assert f"unisem {command}" in command_help


def test_prepare_data_runs_for_emneuron_and_waferem(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    emneuron_source = tmp_path / "emneuron-source"
    emneuron_source.mkdir()
    for index in range(2):
        volume_root = emneuron_source / f"volume-{index}"
        volume_root.mkdir()
        raw = np.arange(2 * 8 * 10, dtype=np.uint16).reshape(2, 8, 10) + index * 200
        labels = np.zeros_like(raw, dtype=np.uint32)
        labels[:, 1:6, 2:8] = index + 1
        tifffile.imwrite(volume_root / "sample.tif", raw)
        tifffile.imwrite(volume_root / "sample_MaskIns.tif", labels)

    emneuron_output = tmp_path / "emneuron-prepared"
    emneuron_result = _run_cli(
        capsys,
        "prepare-data",
        "--dataset",
        "emneuron",
        "--input-root",
        emneuron_source,
        "--output-root",
        emneuron_output,
        "--config",
        PROTOCOL_PATH,
    )
    assert emneuron_result["num_volumes"] == 2
    assert emneuron_result["num_train_volumes"] == 1
    assert emneuron_result["num_validation_volumes"] == 1
    assert (emneuron_output / "split_manifest.json").is_file()

    waferem_source = tmp_path / "waferem-source"
    waferem_source.mkdir()
    for index, case in enumerate(WAFEREM_CASES):
        raw = np.arange(1 * 8 * 10, dtype=np.float32).reshape(1, 8, 10) + index * 100
        labels = np.zeros((1, 8, 10), dtype=np.uint32)
        labels[:, :4, :6] = index + 1
        labels[:, 4:, 6:] = index + 11
        _write_h5(waferem_source / f"{case}_inputs.h5", raw)
        _write_h5(waferem_source / f"{case}_labels.h5", labels)

    waferem_output = tmp_path / "waferem-prepared"
    waferem_result = _run_cli(
        capsys,
        "prepare-data",
        "--dataset",
        "waferem",
        "--input-root",
        waferem_source,
        "--output-root",
        waferem_output,
        "--config",
        PROTOCOL_PATH,
    )
    assert len(waferem_result["adaptation_volume_ids"]) == 5
    assert waferem_result["final_test_volume_id"] == "wafer4_ds2"
    assert waferem_result["final_test_volume_id"] not in waferem_result["adaptation_volume_ids"]
    assert (waferem_output / "waferem_ds2_manifest.json").is_file()


def test_train_and_infer_cli_orchestration(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    train_call: dict[str, object] = {}

    def fake_train(**kwargs: object) -> dict[str, object]:
        train_call.update(kwargs)
        return {"ok": True, "profile": kwargs["profile_name"]}

    monkeypatch.setattr("unisem.training.train", fake_train)
    train_result = _run_cli(
        capsys,
        "train",
        "--profile",
        "waferem-lightft",
        "--data-root",
        tmp_path / "prepared",
        "--output-root",
        tmp_path / "training-output",
        "--sam-checkpoint",
        tmp_path / "sam.pth",
        "--init-checkpoint",
        tmp_path / "initial.pt",
        "--seed",
        23456,
        "--epochs",
        1,
        "--max-steps",
        1,
        "--config",
        PROTOCOL_PATH,
    )
    assert train_result == {"ok": True, "profile": "waferem-lightft"}
    assert train_call["profile_name"] == "waferem-lightft"
    assert train_call["seed"] == 23456
    assert train_call["epochs"] == 1
    assert train_call["max_steps"] == 1
    assert train_call["protocol"]["protocol_id"] == "unisem-isbi-2027-v2"

    raw = np.zeros((48, 64), dtype=np.float32)
    raw[6:22, 8:24] = 0.4
    raw[26:44, 38:58] = 0.7
    input_path = tmp_path / "raw.h5"
    output_path = tmp_path / "prediction.h5"
    _write_h5(input_path, raw)
    model_arguments: dict[str, object] = {}
    model_sentinel = object()
    device_sentinel = object()

    def fake_load_model(arguments: object) -> tuple[object, object]:
        model_arguments.update(vars(arguments))
        return model_sentinel, device_sentinel

    def fake_predict_scale(
        image: np.ndarray,
        model: object,
        device: object,
        scale: dict[str, object],
    ) -> np.ndarray:
        assert model is model_sentinel
        assert device is device_sentinel
        assert scale["name"] in {"large", "medium", "small"}
        foreground = np.zeros(image.shape, dtype=np.float32)
        foreground[5:23, 7:25] = 1.0
        foreground[25:45, 37:59] = 1.0
        center = np.ones(image.shape, dtype=np.float32)
        boundary = np.ones(image.shape, dtype=np.float32)
        center[11:17, 13:19] = 0.0
        boundary[11:17, 13:19] = 0.0
        center[32:38, 44:52] = 0.0
        boundary[32:38, 44:52] = 0.0
        return np.stack((foreground, center, boundary))

    monkeypatch.setattr(cli, "_load_model", fake_load_model)
    monkeypatch.setattr(cli, "_predict_scale", fake_predict_scale)
    monkeypatch.setattr("unisem.rag.load_random_forest", lambda path: SimpleNamespace(path=path))

    infer_result = _run_cli(
        capsys,
        "infer",
        "--dataset",
        "emneuron",
        "--input",
        input_path,
        "--output",
        output_path,
        "--checkpoint",
        tmp_path / "model.pt",
        "--sam-checkpoint",
        tmp_path / "sam.pth",
        "--rag-checkpoint",
        tmp_path / "rag.joblib",
        "--model-type",
        "vit_b",
        "--device",
        "cpu",
        "--config",
        PROTOCOL_PATH,
    )
    assert infer_result == {"merged": True, "output": str(output_path), "slices": 1}
    assert model_arguments["model_type"] == "vit_b"
    assert model_arguments["device"] == "cpu"
    assert model_arguments["base_checkpoint"] is None

    expected_keys = {
        *CHANNEL_NAMES,
        "final_merged_segmentation",
        *(f"{channel}_{scale}" for channel in CHANNEL_NAMES for scale in SCALE_NAMES),
        *(f"segmentation_{scale}" for scale in SCALE_NAMES),
    }
    with h5py.File(output_path, "r") as handle:
        assert set(handle.keys()) == expected_keys
        for name in expected_keys:
            assert handle[name].shape == raw.shape
            assert handle[name].compression == "gzip"
        for name in CHANNEL_NAMES:
            assert handle[name].dtype == np.dtype("float32")
        assert handle["final_merged_segmentation"].dtype == np.dtype("uint32")


def test_rag_evaluation_checkpoint_and_smoke_commands(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    raw, ground_truth, _, prediction_datasets = _rag_arrays()
    raw_path = tmp_path / "raw.h5"
    ground_truth_path = tmp_path / "ground-truth.h5"
    prediction_path = tmp_path / "dense-prediction.h5"
    _write_h5(raw_path, raw)
    _write_h5(ground_truth_path, ground_truth)
    with h5py.File(prediction_path, "w") as handle:
        for name, values in prediction_datasets.items():
            handle.create_dataset(name, data=values)

    rag_manifest = tmp_path / "rag-manifest.json"
    rag_manifest.write_text(
        json.dumps(
            {
                "records": [
                    {
                        "raw": str(raw_path),
                        "ground_truth": str(ground_truth_path),
                        "prediction": str(prediction_path),
                    }
                ]
            }
        ),
        encoding="ascii",
    )
    rag_output = tmp_path / "rag.joblib"
    rag_result = _run_cli(
        capsys,
        "train-rag",
        "--manifest",
        rag_manifest,
        "--output",
        rag_output,
        "--workers",
        1,
    )
    assert rag_result["edges"] == 2
    assert rag_result["positive_edges"] == 1
    assert rag_result["negative_edges"] == 1
    assert rag_output.is_file()

    evaluation_output = tmp_path / "evaluation.json"
    evaluation_result = _run_cli(
        capsys,
        "evaluate",
        "--prediction",
        ground_truth_path,
        "--prediction-key",
        "main",
        "--ground-truth",
        ground_truth_path,
        "--output",
        evaluation_output,
    )
    assert evaluation_result["cremi_from_aggregate_components"] == pytest.approx(0.0)
    assert evaluation_result["mean_per_slice_cremi"] == pytest.approx(0.0)
    persisted_evaluation = json.loads(evaluation_output.read_text(encoding="ascii"))
    assert persisted_evaluation["aggregate"] == evaluation_result

    source_checkpoint = _legacy_checkpoint(tmp_path / "legacy.pt")
    inspection_output = tmp_path / "inspection.json"
    inspection = _run_cli(
        capsys,
        "inspect-checkpoint",
        "--checkpoint",
        source_checkpoint,
        "--output",
        inspection_output,
    )
    assert inspection["top_level_keys"] == ["decoder_state", "model_state", "model_type"]
    assert json.loads(inspection_output.read_text(encoding="ascii")) == inspection

    sanitized_path = tmp_path / "sanitized.pt"
    sanitized_report = _run_cli(
        capsys,
        "sanitize-checkpoint",
        "--source",
        source_checkpoint,
        "--output",
        sanitized_path,
        "--model-type",
        "vit_b",
    )
    assert sanitized_path.is_file()
    assert sanitized_report["schema_version"] == "unisem.inference.v2"
    assert sanitized_report["forbidden_string_hits"] == []

    smoke_output = tmp_path / "smoke.json"
    smoke_result = _run_cli(capsys, "smoke", "--output", smoke_output)
    assert smoke_result["ok"] is True
    assert smoke_result["target_shape"] == [3, 64, 64]
    assert smoke_result["metrics"] == json.loads(smoke_output.read_text(encoding="ascii"))["metrics"]
