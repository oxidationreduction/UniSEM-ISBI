from __future__ import annotations

from pathlib import Path

import torch
from torch import nn

from unisem.checkpoints import (
    INFERENCE_SCHEMA_VERSION,
    inspect_checkpoint,
    load_inference_checkpoint,
    sanitize_checkpoint,
    sanitize_checkpoint_file,
    scan_forbidden_strings,
)
from unisem.model import UniSEMHQUNETR, apply_freeze_policy


class _PatchEmbedding(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.proj = nn.Conv2d(3, channels, kernel_size=4, stride=4)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        return self.proj(images).permute(0, 2, 3, 1)


class _GlobalBlock(nn.Module):
    window_size = 0

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.projection = nn.Linear(channels, channels)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return features + self.projection(features)


class _SyntheticEncoder(nn.Module):
    img_size = 16

    def __init__(self, channels: int = 8) -> None:
        super().__init__()
        self.patch_embed = _PatchEmbedding(channels)
        self.pos_embed = nn.Parameter(torch.zeros(1, 4, 4, channels))
        self.blocks = nn.ModuleList([_GlobalBlock(channels)])
        self.neck = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=1),
            nn.GELU(),
        )


class _SyntheticDecoder(nn.Module):
    def __init__(self, channels: int = 8) -> None:
        super().__init__()
        self.output = nn.Conv2d(channels, 3, kernel_size=1)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.output(features))


def _make_model() -> UniSEMHQUNETR:
    return UniSEMHQUNETR(
        image_encoder=_SyntheticEncoder(),
        baseline_decoder=_SyntheticDecoder(),
        hidden_dim=8,
    )


def _all_trainable(module: nn.Module) -> bool:
    parameters = list(module.parameters())
    return bool(parameters) and all(parameter.requires_grad for parameter in parameters)


def _all_frozen(module: nn.Module) -> bool:
    parameters = list(module.parameters())
    return bool(parameters) and all(not parameter.requires_grad for parameter in parameters)


def test_three_channel_forward_and_backward() -> None:
    torch.manual_seed(0)
    model = _make_model()
    images = torch.rand(2, 1, 13, 11) * 255.0
    target = torch.rand(2, 3, 13, 11)

    prediction = model(images)
    loss = nn.functional.mse_loss(prediction, target)
    loss.backward()

    assert prediction.shape == target.shape
    assert torch.isfinite(prediction).all()
    assert prediction.min() >= 0.0
    assert prediction.max() <= 1.0
    assert any(parameter.grad is not None for parameter in model.encoder.parameters())
    assert any(parameter.grad is not None for parameter in model.baseline_decoder.parameters())
    assert any(parameter.grad is not None for parameter in model.hq_residual.parameters())


def test_stage1_lightft_and_fullft_freeze_policies() -> None:
    model = _make_model()

    for policy in ("stage1", "fullft"):
        summary = apply_freeze_policy(model, policy)
        assert summary["policy"] == policy
        assert _all_trainable(model.encoder)
        assert _all_trainable(model.baseline_decoder)
        assert _all_trainable(model.hq_residual)
        assert summary["groups"]["encoder"]["trainable"] > 0

    summary = apply_freeze_policy(model, "head_only")
    assert summary["policy"] == "head_only"
    assert _all_frozen(model.encoder)
    assert _all_trainable(model.baseline_decoder)
    assert _all_trainable(model.hq_residual)
    assert summary["groups"]["encoder"]["trainable"] == 0


def test_checkpoint_inspect_sanitize_round_trip_and_privacy(tmp_path: Path) -> None:
    torch.manual_seed(1)
    model = _make_model().eval()
    private_path = "/" + "home" + "/anonymous/internal-run/checkpoint.pt"
    legacy = {
        "model_type": "vit_b",
        "model_state": model.state_dict(),
        "state_dict": model.state_dict(),
        "metadata": {"source": private_path, "host": "private-marker"},
    }

    legacy_report = inspect_checkpoint(legacy)
    assert legacy_report["forbidden_string_hits"]
    assert scan_forbidden_strings(legacy, forbidden_strings=("private-marker",))

    clean = sanitize_checkpoint(legacy, hidden_dim=8)
    assert clean["schema_version"] == INFERENCE_SCHEMA_VERSION
    assert clean["architecture"]["out_channels"] == 3
    assert clean["architecture"]["channel_layout"] == {
        "foreground": 0,
        "center_distances": 1,
        "boundary_distances": 2,
    }
    assert not scan_forbidden_strings(clean)
    assert not scan_forbidden_strings(clean, forbidden_strings=("private-marker",))

    clean_path = tmp_path / "public-inference.pt"
    report = sanitize_checkpoint_file(
        source=_save_checkpoint(tmp_path / "legacy.pt", legacy),
        destination=clean_path,
        hidden_dim=8,
    )
    loaded = load_inference_checkpoint(clean_path)
    assert report["forbidden_string_hits"] == []
    assert loaded["schema_version"] == INFERENCE_SCHEMA_VERSION

    restored = _make_model().eval()
    restored.encoder.load_state_dict(loaded["encoder_state"])
    restored.load_decoder_state(loaded["decoder_state"])
    images = torch.rand(1, 1, 12, 10) * 255.0
    with torch.inference_mode():
        expected = model(images)
        actual = restored(images)
    torch.testing.assert_close(actual, expected)


def _save_checkpoint(path: Path, checkpoint: dict[str, object]) -> Path:
    torch.save(checkpoint, path)
    return path
