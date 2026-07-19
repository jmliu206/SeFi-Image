from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from demo.dit_finetune.train import (
    _assert_distributed_finite_loss,
    _configure_deepspeed_for_explicit_dataloader,
    _prepare_image_branches,
    _validate_ema_global_step,
    load_training_config,
    validate_training_config,
)


class _DeepSpeedPlugin:
    def __init__(self, config):
        self.deepspeed_config = dict(config)


class _SemanticCodec:
    def __init__(self):
        self.input_dtype = None
        self.input_values = None

    def preprocess_batch(self, pixel_values, *, input_range):
        assert input_range == "minus_one_one"
        self.input_dtype = pixel_values.dtype
        self.input_values = pixel_values.detach().clone()
        return pixel_values.float() + 10


class _LossAccelerator:
    device = torch.device("cpu")
    num_processes = 8

    def __init__(self, finite_ranks):
        self.finite_ranks = int(finite_ranks)

    def reduce(self, value, reduction):
        assert reduction == "sum"
        assert value.dtype == torch.int64
        return torch.tensor(self.finite_ranks, dtype=torch.int64)


def test_nonfinite_loss_is_reduced_before_every_rank_raises():
    _assert_distributed_finite_loss(
        _LossAccelerator(finite_ranks=8),
        torch.tensor(1.0),
        global_step=3,
    )
    with pytest.raises(FloatingPointError, match="finite_ranks=7/8"):
        _assert_distributed_finite_loss(
            _LossAccelerator(finite_ranks=7),
            torch.tensor(1.0),
            global_step=3,
        )


def test_image_branches_preserve_fp32_for_dino_and_cast_only_texture():
    codec = _SemanticCodec()
    source = torch.tensor([[[[0.1234567]]]], dtype=torch.float32)

    texture, semantic = _prepare_image_branches(
        source,
        semantic_codec=codec,
        device=torch.device("cpu"),
    )

    assert codec.input_dtype == torch.float32
    torch.testing.assert_close(codec.input_values, source, rtol=0, atol=0)
    assert texture.dtype == torch.bfloat16
    assert semantic.dtype == torch.float32
    torch.testing.assert_close(semantic, source + 10)


def test_explicit_dataloader_resolves_deepspeed_auto_micro_batch_and_clip():
    plugin = _DeepSpeedPlugin(
        {
            "train_micro_batch_size_per_gpu": "auto",
            "train_batch_size": "auto",
            "gradient_accumulation_steps": "auto",
            "gradient_clipping": "auto",
        }
    )

    _configure_deepspeed_for_explicit_dataloader(
        plugin,
        micro_batch_size=1,
        gradient_accumulation_steps=2,
        gradient_clip=1.0,
    )

    assert plugin.deepspeed_config["train_micro_batch_size_per_gpu"] == 1
    assert plugin.deepspeed_config["gradient_clipping"] == 1.0
    # Accelerate still resolves these after it knows the process count.
    assert plugin.deepspeed_config["train_batch_size"] == "auto"
    assert plugin.deepspeed_config["gradient_accumulation_steps"] == "auto"


@pytest.mark.parametrize(
    ("override", "message"),
    [
        ({"train_micro_batch_size_per_gpu": 2}, "train_micro_batch_size"),
        ({"gradient_accumulation_steps": 4}, "gradient_accumulation_steps"),
        ({"gradient_clipping": 0.5}, "gradient_clipping"),
    ],
)
def test_explicit_dataloader_rejects_deepspeed_recipe_mismatch(override, message):
    config = {
        "train_micro_batch_size_per_gpu": "auto",
        "gradient_accumulation_steps": "auto",
        "gradient_clipping": "auto",
    }
    config.update(override)
    with pytest.raises(ValueError, match=message):
        _configure_deepspeed_for_explicit_dataloader(
            _DeepSpeedPlugin(config),
            micro_batch_size=1,
            gradient_accumulation_steps=2,
            gradient_clip=1.0,
        )


def test_ema_step_must_match_real_optimizer_global_step():
    _validate_ema_global_step(None, 10)
    _validate_ema_global_step(SimpleNamespace(optimization_step=10), 10)
    with pytest.raises(RuntimeError, match="EMA/optimizer step mismatch"):
        _validate_ema_global_step(SimpleNamespace(optimization_step=9), 10)


@pytest.mark.parametrize(
    "name",
    [
        "lora_1b.yaml",
        "lora_2b.yaml",
        "lora_5b.yaml",
        "full_1b_zero2.yaml",
        "full_2b_zero2.yaml",
        "full_5b_zero2.yaml",
    ],
)
def test_all_public_recipes_merge_and_validate(name):
    config = load_training_config(f"demo/dit_finetune/configs/{name}")
    validate_training_config(config)
