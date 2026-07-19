import json
from collections import OrderedDict
from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn
from safetensors.torch import load_file

from sefi.training.checkpointing import (
    TrainerState,
    collect_portable_state_dict,
    dataloader_for_resume,
    export_full_transformer_checkpoint,
    export_sharded_safetensors,
    overlay_ema_parameters_by_name,
    read_trainer_state,
    write_trainer_state,
)


def test_trainer_state_round_trip_and_batch_position(tmp_path):
    state = TrainerState(global_step=7, data_epoch=3, batch_offset=4)

    path = write_trainer_state(tmp_path, state)
    restored = read_trainer_state(tmp_path)

    assert path.name == "trainer_state.json"
    assert restored == state
    assert restored.after_batch(8) == TrainerState(
        global_step=7, data_epoch=3, batch_offset=5
    )
    assert TrainerState(global_step=7, data_epoch=3, batch_offset=7).after_batch(8) == TrainerState(
        global_step=7, data_epoch=4, batch_offset=0
    )


def test_trainer_state_reads_legacy_iteration(tmp_path):
    (tmp_path / "trainer_state.json").write_text('{"iteration": 12}')

    state = read_trainer_state(tmp_path)

    assert state == TrainerState(global_step=12)


class _Sampler:
    def __init__(self):
        self.epoch = None

    def set_epoch(self, epoch):
        self.epoch = epoch


class _Loader:
    def __init__(self, length):
        self.length = length
        self.sampler = _Sampler()

    def __len__(self):
        return self.length


class _PreparedLoader(_Loader):
    def __init__(self, length):
        super().__init__(length)
        self.prepared_epoch = None

    def set_epoch(self, epoch):
        self.prepared_epoch = epoch


class _SkipAccelerator:
    def skip_first_batches(self, dataloader, count):
        return dataloader, count


def test_dataloader_resume_sets_epoch_and_skips_offset():
    loader = _Loader(10)

    resumed = dataloader_for_resume(
        _SkipAccelerator(), loader, TrainerState(data_epoch=5, batch_offset=3)
    )

    assert loader.sampler.epoch == 5
    assert resumed == (loader, 3)

    prepared = _PreparedLoader(10)
    dataloader_for_resume(
        _SkipAccelerator(), prepared, TrainerState(data_epoch=6, batch_offset=0)
    )
    assert prepared.prepared_epoch == 6
    assert prepared.sampler.epoch is None


def test_ema_overlay_is_exactly_name_keyed_and_preserves_buffers():
    state = OrderedDict(
        {
            "module.layer.weight": torch.zeros(2, 3),
            "module.layer.bias": torch.zeros(2),
            "module.buffer": torch.tensor([9], dtype=torch.int64),
        }
    )
    shadows = [torch.full((2,), 2.0), torch.full((2, 3), 3.0)]

    overlaid = overlay_ema_parameters_by_name(
        state,
        ema_param_names=["layer.bias", "layer.weight"],
        shadow_params=shadows,
    )

    assert torch.equal(overlaid["layer.bias"], shadows[0])
    assert torch.equal(overlaid["layer.weight"], shadows[1])
    assert torch.equal(overlaid["buffer"], torch.tensor([9]))
    with pytest.raises(ValueError, match="shape mismatch"):
        overlay_ema_parameters_by_name(
            state,
            ema_param_names=["layer.weight"],
            shadow_params=[torch.zeros(1)],
        )
    with pytest.raises(ValueError, match="missing from model"):
        overlay_ema_parameters_by_name(
            state,
            ema_param_names=["not_a_parameter"],
            shadow_params=[torch.zeros(1)],
        )


def _load_sharded_export(output_dir, index_name):
    index = json.loads((output_dir / index_name).read_text())
    loaded = {}
    for filename in sorted(set(index["weight_map"].values())):
        loaded.update(load_file(output_dir / filename))
    return index, loaded


def test_export_sharded_safetensors_writes_hf_index_and_casts_float(tmp_path):
    state = {
        "z.weight": torch.arange(100, dtype=torch.float32).reshape(10, 10),
        "a.weight": torch.arange(100, dtype=torch.float32).reshape(10, 10),
        "counter": torch.tensor([4], dtype=torch.int64),
    }

    result = export_sharded_safetensors(
        state,
        tmp_path,
        max_shard_size=300,
        dtype="bf16",
    )

    assert result.index_file == "diffusion_pytorch_model.safetensors.index.json"
    assert len(result.weight_files) >= 2
    index, loaded = _load_sharded_export(tmp_path, result.index_file)
    assert set(loaded) == set(state)
    assert loaded["a.weight"].dtype == torch.bfloat16
    assert loaded["counter"].dtype == torch.int64
    assert index["metadata"]["total_size"] == result.total_size
    with pytest.raises(FileExistsError):
        export_sharded_safetensors(state, tmp_path, max_shard_size=300)


class _ExportAccelerator:
    def __init__(self, state_dict, is_main=True):
        self.state_dict = state_dict
        self.is_main_process = is_main
        self.get_state_dict_calls = 0

    def get_state_dict(self, model):
        assert model is not None
        self.get_state_dict_calls += 1
        return self.state_dict


def test_full_transformer_export_uses_ema_names_and_expected_layout(tmp_path):
    state = OrderedDict(
        weight=torch.zeros(2, 2, dtype=torch.bfloat16),
        buffer=torch.tensor([1], dtype=torch.int64),
    )
    accelerator = _ExportAccelerator(state)
    ema = SimpleNamespace(shadow_params=[torch.full((2, 2), 5.0, dtype=torch.float32)])

    result = export_full_transformer_checkpoint(
        accelerator,
        nn.Linear(2, 2, bias=False),
        tmp_path,
        ema_model=ema,
        ema_param_names=["weight"],
        max_shard_size="1KB",
        dtype="bf16",
        transformer_config={"model_type": "sefi-test"},
        metadata={"scale": "1b"},
    )

    assert result is not None
    transformer_dir = tmp_path / "transformer"
    weights = load_file(transformer_dir / "diffusion_pytorch_model.safetensors")
    assert torch.equal(weights["weight"], torch.full((2, 2), 5.0, dtype=torch.bfloat16))
    assert torch.equal(weights["buffer"], torch.tensor([1]))
    assert json.loads((transformer_dir / "config.json").read_text())["model_type"] == "sefi-test"
    manifest = json.loads((tmp_path / "sefi_export_manifest.json").read_text())
    assert manifest["source"] == "ema"
    assert manifest["metadata"]["scale"] == "1b"
    assert accelerator.get_state_dict_calls == 1


def test_collect_portable_state_dict_requires_names_for_ema():
    accelerator = _ExportAccelerator({"weight": torch.zeros(1)})

    with pytest.raises(ValueError, match="stable ema_param_names"):
        collect_portable_state_dict(
            accelerator,
            nn.Linear(1, 1),
            ema_model=SimpleNamespace(shadow_params=[torch.zeros(1)]),
        )
