from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import torch
from torch import nn

from unisem.checkpoints import inference_checkpoint, make_adapter, reconstruct_adapter
from unisem.lora import LoRAQKVLinear, LoRAQVConfig, inject_lora_qv
from unisem.metrics import aggregate_slice_metrics, aggregate_volumes, evaluate_segmentation
from unisem.training import _mean_loss, label_selection, profile_from_protocol, seed_everything


def test_official_sam_initialization_is_strict_cpu_and_does_not_download(tmp_path: Path, monkeypatch) -> None:
    from segment_anything import sam_model_registry
    from unisem.model import _load_sam_image_encoder

    class SmallSAM(nn.Module):
        def __init__(self):
            super().__init__()
            self.image_encoder = nn.Linear(2, 2)
            self.prompt_encoder = nn.Linear(1, 1)

    def build(checkpoint=None):
        assert checkpoint is None
        return SmallSAM()

    monkeypatch.setitem(sam_model_registry, "test_arch", build)
    original = SmallSAM()
    path = tmp_path / "official.pt"
    torch.save(original.state_dict(), path)
    restored = _load_sam_image_encoder("test_arch", path)
    for key, tensor in restored.state_dict().items():
        torch.testing.assert_close(tensor, original.image_encoder.state_dict()[key], rtol=0, atol=0)
        assert tensor.device.type == "cpu"
    torch.save({"incomplete": torch.zeros(1)}, path)
    with pytest.raises(RuntimeError):
        _load_sam_image_encoder("test_arch", path)


def test_qv_branches_preserve_initial_output_and_never_change_k() -> None:
    torch.manual_seed(12345)
    base = nn.Linear(8, 24)
    adapter = LoRAQKVLinear(base, LoRAQVConfig(rank=2, alpha=2))
    values = torch.randn(2, 3, 8)
    initial = base(values).detach()
    torch.testing.assert_close(adapter(values), initial)
    with torch.no_grad():
        adapter.lora_q_up.weight.fill_(0.1)
        adapter.lora_v_up.weight.fill_(0.2)
    adapted = adapter(values)
    torch.testing.assert_close(adapted[..., 8:16], initial[..., 8:16])
    assert not torch.equal(adapted[..., :8], initial[..., :8])
    assert not any(parameter.requires_grad for parameter in adapter.base.parameters())


def test_all24_qv_parameter_count_without_allocating_a_vitl() -> None:
    with torch.device("meta"):
        encoder = nn.Module()
        encoder.blocks = nn.ModuleList()
        for _ in range(24):
            block = nn.Module()
            block.attn = nn.Module()
            block.attn.qkv = nn.Linear(1024, 3072)
            encoder.blocks.append(block)
        summary = inject_lora_qv(encoder)
    assert summary["blocks"] == 24
    assert summary["tensors"] == 96
    assert summary["trainable"] == 1_572_864
    assert summary["trainable"] + 9_591_235 + 2_033_155 == 13_197_254
    assert all((".lora_" in name) == parameter.requires_grad for name, parameter in encoder.named_parameters())


def test_adapter_reconstruction_checks_base_and_preserves_all_head_buffers(tmp_path: Path) -> None:
    from test_model_training_checkpoint import _make_model

    model = _make_model()
    # Give the synthetic encoder a fused attention projection for the adapter test.
    block = model.encoder.blocks[0]
    block.attn = nn.Module()
    block.attn.qkv = nn.Linear(8, 24)
    base = inference_checkpoint(model, model_type="vit_b")
    base_path = tmp_path / "base.pt"
    torch.save(base, base_path)
    inject_lora_qv(model.encoder, LoRAQVConfig(rank=2, alpha=2))
    full = inference_checkpoint(model, model_type="vit_b")
    adapter = make_adapter(full, base_path)
    rebuilt = reconstruct_adapter(adapter, base_path)
    for group in ("encoder_state", "decoder_state"):
        assert set(rebuilt[group]) == set(full[group])
        for key in rebuilt[group]:
            torch.testing.assert_close(rebuilt[group][key], full[group][key], rtol=0, atol=0)
    assert len(adapter["decoder_state"]) == len(full["decoder_state"])
    corrupted = {**adapter, "base_sha256": "0" * 64}
    with pytest.raises(ValueError, match="SHA256"):
        reconstruct_adapter(corrupted, base_path)
    incomplete = {**adapter, "lora_state": dict(adapter["lora_state"])}
    incomplete["lora_state"].pop(next(iter(incomplete["lora_state"])))
    with pytest.raises(ValueError, match="LoRA"):
        reconstruct_adapter(incomplete, base_path)


def test_budget_lists_are_paired_nested_and_monitor_uses_only_allowed_labels() -> None:
    root = Path(__file__).resolve().parents[1]
    protocol = json.loads((root / "configs/paper_protocol.json").read_text())
    volumes = protocol["waferem"]["adaptation_volumes"]
    for seed in (12345, 23456, 34567):
        frozen = protocol["waferem"]["label_budgets"][str(seed)]
        lists = {budget: label_selection(dict.fromkeys(volumes, 125), budget, seed, frozen[str(budget)])
                 for budget in (160, 315, 625)}
        for name in volumes:
            assert set(lists[160][name]) <= set(lists[315][name]) <= set(lists[625][name])
        assert sum(map(len, lists[160].values())) == 160
    low = label_selection(dict.fromkeys(volumes, 125), 160, 77)
    high = label_selection(dict.fromkeys(volumes, 125), 315, 77)
    assert all(set(low[name]) <= set(high[name]) for name in volumes)
    light = profile_from_protocol(protocol, "waferem-lightft")
    full = profile_from_protocol(protocol, "waferem-fullft")
    assert light["batch_size"] * light["gradient_accumulation_steps"] == 4
    assert full["batch_size"] * full["gradient_accumulation_steps"] == 4
    assert 2500 // 4 * light["epochs"] == 9375
    with pytest.raises(ValueError, match="duplicate"):
        label_selection(dict.fromkeys(volumes, 125), 160, 77, dict.fromkeys(volumes, [0] * 32))


def test_gradient_accumulation_matches_an_effective_batch_of_four() -> None:
    from torch.utils.data import DataLoader, TensorDataset

    seed_everything(12)
    raw, target = torch.rand(4, 1, 5, 5), torch.rand(4, 3, 5, 5)
    model_a, model_b = nn.Conv2d(1, 3, 1), nn.Conv2d(1, 3, 1)
    model_b.load_state_dict(model_a.state_dict())
    for model, batch, accumulation in ((model_a, 4, 1), (model_b, 2, 2)):
        optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
        loader = DataLoader(TensorDataset(raw, target), batch_size=batch)
        _, updates = _mean_loss(model, loader, nn.MSELoss(), torch.device("cpu"), train_mode=True,
                               optimizer=optimizer, scaler=torch.amp.GradScaler("cuda", enabled=False),
                               accumulation=accumulation)
        assert updates == 1
    for actual, expected in zip(model_a.parameters(), model_b.parameters()):
        torch.testing.assert_close(actual, expected)


def test_two_sided_component_evaluation_and_equal_volume_aggregation() -> None:
    ground_truth = np.zeros((12, 12), dtype=np.uint32)
    ground_truth[1:4, 1:4], ground_truth[8:11, 8:11] = 1, 2
    prediction = np.where(ground_truth > 0, 17, 0).astype(np.uint32)
    result = evaluate_segmentation(prediction, ground_truth)
    assert result["voi_sum"] == pytest.approx(0, abs=1e-14)
    assert result["arand"] == pytest.approx(0, abs=1e-14)
    good = {"voi_split": 0.0, "voi_merge": 0.0, "adapted_rand_error": 0.0, "rand_index": 1.0, "cremi": 0.0}
    bad = {"voi_split": 1.0, "voi_merge": 0.0, "adapted_rand_error": 1.0, "rand_index": 0.0, "cremi": 1.0}
    volumes = {"small": aggregate_slice_metrics([good]), "large": aggregate_slice_metrics([bad] * 9)}
    aggregate = aggregate_volumes(volumes, expected_volumes=["small", "large"])
    assert aggregate["mean_per_slice_cremi"] == 0.5
    assert aggregate["num_slices"] == 10
    with pytest.raises(ValueError, match="volumes"):
        aggregate_volumes(volumes, expected_volumes=["small"])
