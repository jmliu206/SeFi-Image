from __future__ import annotations

import copy

import pytest
import torch
import torch.nn as nn

from sefi.training.ema import FullGpuEMA


class _Accelerator:
    def __init__(self):
        self.registered = []

    def unwrap_model(self, model):
        return model

    def register_for_checkpointing(self, value):
        self.registered.append(value)


def _model(dtype=torch.float32):
    model = nn.Sequential(nn.Linear(2, 3), nn.Linear(3, 1, bias=False))
    return model.to(dtype=dtype)


def test_ema_is_per_rank_fp32_registered_and_name_mapped():
    accelerator = _Accelerator()
    model = _model(dtype=torch.bfloat16)
    ema = FullGpuEMA(model, accelerator=accelerator)

    assert accelerator.registered == [ema]
    assert ema.param_names == tuple(name for name, _ in model.named_parameters())
    assert all(value.dtype == torch.float32 for value in ema.ema_model.shadow_params)
    assert all(
        value.device == next(model.parameters()).device
        for value in ema.ema_model.shadow_params
    )
    assert tuple(ema.shadow_state_dict()) == ema.param_names


def test_ema_updates_only_on_real_optimizer_steps_with_diffusers_schedule():
    model = nn.Linear(1, 1, bias=False)
    with torch.no_grad():
        model.weight.fill_(1.0)
    ema = FullGpuEMA(model, register_for_checkpointing=False)

    with torch.no_grad():
        model.weight.fill_(3.0)
    assert ema.step(did_optimizer_step=False) is False
    assert ema.optimization_step == 0
    torch.testing.assert_close(ema.ema_model.shadow_params[0], torch.tensor([[1.0]]))

    assert ema.step(did_optimizer_step=True) is True
    assert ema.optimization_step == 1
    assert ema.current_decay == 0.0
    torch.testing.assert_close(ema.ema_model.shadow_params[0], torch.tensor([[3.0]]))

    with torch.no_grad():
        model.weight.fill_(5.0)
    ema.step(did_optimizer_step=True)
    expected_decay = 2.0 / 11.0
    assert ema.optimization_step == 2
    assert ema.current_decay == pytest.approx(expected_decay)
    torch.testing.assert_close(
        ema.ema_model.shadow_params[0],
        torch.tensor([[3.0 * expected_decay + 5.0 * (1.0 - expected_decay)]]),
    )


def test_ema_state_roundtrip_restores_fp32_device_names_and_step():
    source_model = _model(dtype=torch.bfloat16)
    source = FullGpuEMA(source_model, register_for_checkpointing=False)
    with torch.no_grad():
        for parameter in source_model.parameters():
            parameter.add_(1)
    source.step(did_optimizer_step=True)
    state = copy.deepcopy(source.state_dict())
    # Simulate a checkpoint loader returning a different shadow dtype/device.
    state["ema"]["shadow_params"] = [
        value.double().cpu() for value in state["ema"]["shadow_params"]
    ]

    target_model = _model(dtype=torch.bfloat16)
    target = FullGpuEMA(target_model, register_for_checkpointing=False)
    target.load_state_dict(state)

    assert target.optimization_step == source.optimization_step
    assert target.param_names == source.param_names
    assert all(value.dtype == torch.float32 for value in target.ema_model.shadow_params)
    assert all(
        value.device == next(target_model.parameters()).device
        for value in target.ema_model.shadow_params
    )
    for expected, actual in zip(source.ema_model.shadow_params, target.ema_model.shadow_params):
        torch.testing.assert_close(actual, expected)


def test_ema_resume_rejects_parameter_name_or_shape_mismatch():
    source = FullGpuEMA(_model(), register_for_checkpointing=False)
    state = source.state_dict()

    bad_names = copy.deepcopy(state)
    bad_names["param_names"][0] = "wrong.name"
    with pytest.raises(ValueError, match="parameter mapping"):
        source.load_state_dict(bad_names)

    bad_shapes = copy.deepcopy(state)
    bad_shapes["param_shapes"][0] = [999]
    with pytest.raises(ValueError, match="shapes"):
        source.load_state_dict(bad_shapes)


def test_ema_apply_context_restores_online_model():
    model = nn.Linear(1, 1, bias=False)
    with torch.no_grad():
        model.weight.fill_(1.0)
    ema = FullGpuEMA(model, register_for_checkpointing=False)
    with torch.no_grad():
        model.weight.fill_(2.0)
    online = model.weight.detach().clone()

    with ema.apply_to():
        torch.testing.assert_close(model.weight, torch.tensor([[1.0]]))
    torch.testing.assert_close(model.weight, online)


def test_ema_param_manifest_is_stable(tmp_path):
    ema = FullGpuEMA(_model(), register_for_checkpointing=False)
    path = ema.write_param_names(tmp_path)
    assert path.name == "ema_param_names.json"
    text = path.read_text(encoding="utf-8")
    for name in ema.param_names:
        assert name in text
