import json
from pathlib import Path

import pytest
import torch
import torch.nn as nn
from accelerate import Accelerator
from safetensors.torch import load_file

from sefi.training.lora import (
    ADAPTER_METADATA_FILENAME,
    DOUBLE_STREAM_TARGETS,
    EXPECTED_ATTENTION_COMPLETE_TARGET_COUNTS,
    SINGLE_STREAM_TARGETS,
    build_lora_metadata,
    count_trainable_parameters,
    create_lora_model,
    discover_attention_complete_targets,
    load_lora_adapter_weights,
    register_lora_accelerate_hooks,
    save_lora_adapter,
)


class _DoubleAttention(nn.Module):
    def __init__(self, width=4):
        super().__init__()
        self.to_q = nn.Linear(width, width, bias=False)
        self.to_k = nn.Linear(width, width, bias=False)
        self.to_v = nn.Linear(width, width, bias=False)
        self.to_out = nn.ModuleList([nn.Linear(width, width, bias=False)])
        self.add_q_proj = nn.Linear(width, width, bias=False)
        self.add_k_proj = nn.Linear(width, width, bias=False)
        self.add_v_proj = nn.Linear(width, width, bias=False)
        self.to_add_out = nn.Linear(width, width, bias=False)


class _SingleAttention(nn.Module):
    def __init__(self, width=4):
        super().__init__()
        self.to_qkv_mlp_proj = nn.Linear(width, width, bias=False)
        self.to_out = nn.Linear(width, width, bias=False)


class _Block(nn.Module):
    def __init__(self, attention):
        super().__init__()
        self.attn = attention


class _TinyFluxWrapper(nn.Module):
    def __init__(self, double_blocks, single_blocks, width=4):
        super().__init__()
        self.backbone = nn.Module()
        self.backbone.transformer_blocks = nn.ModuleList(
            [_Block(_DoubleAttention(width)) for _ in range(double_blocks)]
        )
        self.backbone.single_transformer_blocks = nn.ModuleList(
            [_Block(_SingleAttention(width)) for _ in range(single_blocks)]
        )
        self.unrelated_to_out = nn.ModuleList([nn.Linear(width, width)])

    def forward(self, x):
        # Enough of a forward for generic PeftModel/Accelerate unit tests.
        return self.backbone.transformer_blocks[0].attn.to_q(x)


SCALE_BLOCKS = {
    "1b": (4, 12),
    "2b": (4, 16),
    "5b": (6, 21),
}


@pytest.mark.parametrize("scale", ["1b", "2b", "5b"])
def test_attention_complete_target_discovery_has_release_scale_count(scale):
    double_blocks, single_blocks = SCALE_BLOCKS[scale]
    model = _TinyFluxWrapper(double_blocks, single_blocks)

    targets = discover_attention_complete_targets(model, scale=scale.upper())

    assert len(targets) == EXPECTED_ATTENTION_COMPLETE_TARGET_COUNTS[scale]
    assert all(isinstance(model.get_submodule(name), nn.Linear) for name in targets)
    assert "unrelated_to_out.0" not in targets
    for block in range(double_blocks):
        prefix = f"backbone.transformer_blocks.{block}.attn."
        assert {name.removeprefix(prefix) for name in targets if name.startswith(prefix)} == set(
            DOUBLE_STREAM_TARGETS
        )
    for block in range(single_blocks):
        prefix = f"backbone.single_transformer_blocks.{block}.attn."
        assert {name.removeprefix(prefix) for name in targets if name.startswith(prefix)} == set(
            SINGLE_STREAM_TARGETS
        )


def test_attention_complete_discovery_fails_on_missing_linear_and_wrong_scale():
    model = _TinyFluxWrapper(4, 12)
    model.backbone.transformer_blocks[2].attn.to_add_out = nn.Identity()

    with pytest.raises(ValueError, match="Incomplete attention-complete"):
        discover_attention_complete_targets(model, scale="1b")

    with pytest.raises(ValueError, match="expected 64"):
        discover_attention_complete_targets(_TinyFluxWrapper(4, 12), scale="2b")


def test_create_lora_model_uses_default_rank_and_freezes_base():
    model, targets = create_lora_model(_TinyFluxWrapper(4, 12), scale="1b")

    trainable, total = count_trainable_parameters(model)
    assert 0 < trainable < total
    assert model.peft_config["default"].r == 16
    assert model.peft_config["default"].lora_alpha == 16
    trainable_names = [name for name, parameter in model.named_parameters() if parameter.requires_grad]
    assert trainable_names
    assert all("lora_" in name for name in trainable_names)
    assert len(targets) == 56
    assert set(model.targeted_module_names) == set(targets)


def _metadata(targets):
    return build_lora_metadata(
        base_model_repo="SeFi-Image/SeFi-Image-1B-Base",
        requested_base_revision="main",
        resolved_base_revision="a" * 40,
        scale="1b",
        target_modules=targets,
        training_config={"tuning": {"mode": "lora", "rank": 16}},
    )


def test_adapter_save_is_adapter_only_and_loads_in_place(tmp_path):
    model, targets = create_lora_model(_TinyFluxWrapper(4, 12), scale="1b")
    for name, parameter in model.named_parameters():
        if "lora_" in name:
            nn.init.constant_(parameter, 0.125)

    adapter_dir = save_lora_adapter(model, tmp_path / "adapter", metadata=_metadata(targets))

    assert (adapter_dir / "adapter_config.json").is_file()
    assert (adapter_dir / ADAPTER_METADATA_FILENAME).is_file()
    saved = load_file(adapter_dir / "adapter_model.safetensors")
    assert saved
    assert all("lora_" in name for name in saved)
    assert not any("base_layer" in name for name in saved)
    metadata = json.loads((adapter_dir / ADAPTER_METADATA_FILENAME).read_text())
    assert metadata["target_count"] == 56
    assert metadata["config_hash"]
    assert metadata["resolved_base_revision"] == "a" * 40

    for name, parameter in model.named_parameters():
        if "lora_" in name:
            nn.init.zeros_(parameter)
    load_lora_adapter_weights(model, adapter_dir)
    restored = [
        parameter
        for name, parameter in model.named_parameters()
        if "lora_" in name and name.endswith(".weight")
    ]
    assert restored
    assert any(torch.count_nonzero(parameter).item() > 0 for parameter in restored)


class _FakeAccelerator:
    def __init__(self):
        self.is_main_process = True
        self.save_hook = None
        self.load_hook = None

    def unwrap_model(self, model):
        return model

    def register_save_state_pre_hook(self, hook):
        self.save_hook = hook
        return object()

    def register_load_state_pre_hook(self, hook):
        self.load_hook = hook
        return object()


def test_accelerate_hooks_replace_full_model_payload_with_adapter(tmp_path):
    accelerator = _FakeAccelerator()
    model, targets = create_lora_model(_TinyFluxWrapper(4, 12), scale="1b")
    register_lora_accelerate_hooks(accelerator, metadata=_metadata(targets))

    weights = [model.state_dict()]
    accelerator.save_hook([model], weights, str(tmp_path))

    assert weights == []
    assert (tmp_path / "adapter" / "adapter_model.safetensors").is_file()

    models = [model]
    accelerator.load_hook(models, str(tmp_path))
    assert models == []


def test_real_accelerate_state_round_trip_restores_adapter_without_full_model(
    tmp_path, monkeypatch
):
    # This CPU-only test environment has DeepSpeed installed but no usable
    # accelerator backend; prevent Accelerate's generic unwrap helper from
    # importing the DeepSpeed engine. Real full-mode tests cover that backend.
    monkeypatch.setattr("accelerate.utils.other.is_deepspeed_available", lambda: False)
    accelerator = Accelerator(cpu=True)
    model, targets = create_lora_model(_TinyFluxWrapper(4, 12), scale="1b")
    optimizer = torch.optim.AdamW(
        [parameter for parameter in model.parameters() if parameter.requires_grad],
        lr=1e-3,
    )
    model, optimizer = accelerator.prepare(model, optimizer)
    register_lora_accelerate_hooks(accelerator, metadata=_metadata(targets))

    loss = model(torch.ones(2, 4)).square().mean()
    accelerator.backward(loss)
    optimizer.step()
    optimizer.zero_grad()
    unwrapped = accelerator.unwrap_model(model)
    before = {
        name: parameter.detach().clone()
        for name, parameter in unwrapped.named_parameters()
        if "lora_" in name
    }
    accelerator.save_state(str(tmp_path))

    assert not (tmp_path / "model.safetensors").exists()
    assert (tmp_path / "adapter" / "adapter_model.safetensors").is_file()
    with torch.no_grad():
        for name, parameter in unwrapped.named_parameters():
            if "lora_" in name:
                parameter.add_(10)
    accelerator.load_state(str(tmp_path))

    after = {
        name: parameter.detach()
        for name, parameter in unwrapped.named_parameters()
        if "lora_" in name
    }
    assert before.keys() == after.keys()
    assert all(torch.equal(before[name], after[name]) for name in before)
