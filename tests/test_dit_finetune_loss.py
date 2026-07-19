from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn

from sefi.training.dit_finetune import (
    SFDLossConfig,
    build_sfd_schedule,
    compose_semantic_texture_latents,
    compute_sfd_loss,
    encode_frozen_batch,
    sample_timestep_u,
)


class _Pipeline:
    @staticmethod
    def _prepare_latent_ids(latents):
        batch, _, height, width = latents.shape
        h, w = torch.meshgrid(torch.arange(height), torch.arange(width), indexing="ij")
        ids = torch.stack(
            (torch.zeros_like(h), h, w, torch.zeros_like(h)), dim=-1
        ).reshape(1, height * width, 4)
        return ids.expand(batch, -1, -1)

    @staticmethod
    def _pack_latents(latents):
        batch, channels, height, width = latents.shape
        return latents.reshape(batch, channels, height * width).permute(0, 2, 1)

    @staticmethod
    def _unpack_latents_with_ids(values, latent_ids):
        batch, _, channels = values.shape
        height = int(latent_ids[..., 1].max().item()) + 1
        width = int(latent_ids[..., 2].max().item()) + 1
        return values.permute(0, 2, 1).reshape(batch, channels, height, width)


class _Scheduler:
    def __init__(self):
        self.config = SimpleNamespace(num_train_timesteps=10)
        self.timesteps = torch.arange(10, 0, -1, dtype=torch.float32) * 100
        self.sigmas = torch.linspace(1.0, 0.1, 10)


class _RecordingModel(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.scale = nn.Parameter(torch.tensor(0.0))
        self.channels = channels
        self.last_call = None

    def forward(self, **kwargs):
        self.last_call = kwargs
        return kwargs["hidden_states"] * self.scale


def _encoded_batch():
    semantic = torch.tensor([[[[1.0, 2.0]], [[3.0, 4.0]]]])
    texture = torch.tensor([[[[5.0, 6.0]]]])
    composed = compose_semantic_texture_latents(
        semantic_latents=semantic,
        texture_latents=texture,
        pipeline_cls=_Pipeline,
        semantic_channels=2,
        texture_channels=1,
        expected_grid=(1, 2),
    )
    composed.update(
        {
            "prompt_embeds": torch.zeros(1, 2, 4),
            "text_ids": torch.zeros(1, 2, 4, dtype=torch.long),
            "text_drop_ratio": 0.25,
        }
    )
    return composed


def test_semantic_first_schedule_matches_reference_equations_and_lookups():
    config = SFDLossConfig(semantic_channels=2, texture_channels=1)
    schedule = build_sfd_schedule(
        noise_scheduler=_Scheduler(),
        config=config,
        batch_size=3,
        device="cpu",
        semantic_ndim=4,
        texture_ndim=4,
        semantic_dtype=torch.float32,
        texture_dtype=torch.float32,
        u=torch.tensor([0.0, 0.5, 1.0]),
        delta_t=torch.tensor([0.1, 0.1, 0.1]),
    )

    torch.testing.assert_close(schedule.u_sem, torch.tensor([0.0, 0.55, 1.0]))
    torch.testing.assert_close(schedule.u_tex, torch.tensor([0.0, 0.45, 1.0]))
    assert schedule.sigmas_sem.shape == (3, 1, 1, 1)
    assert schedule.sigmas_tex.shape == (3, 1, 1, 1)
    # floor(u * 9): semantic [0, 4, 9], texture [0, 4, 9]
    torch.testing.assert_close(schedule.timesteps_sem, torch.tensor([1000.0, 600.0, 100.0]))
    torch.testing.assert_close(schedule.timesteps_tex, torch.tensor([1000.0, 600.0, 100.0]))


def test_all_uniform_mixture_matches_reference_rng_consumption():
    config = SFDLossConfig(
        semantic_channels=2,
        texture_channels=1,
        t_sampling_scheme="logit_normal_with_uniform",
        uniform_prob=1.0,
    )
    actual_generator = torch.Generator().manual_seed(1234)
    reference_generator = torch.Generator().manual_seed(1234)
    actual = sample_timestep_u(
        config,
        batch_size=4,
        device="cpu",
        generator=actual_generator,
    )
    expected = torch.rand(4, generator=reference_generator)
    torch.testing.assert_close(actual, expected)


def test_fixed_tensor_loss_matches_reference_target_weighting_and_dual_times():
    encoded = _encoded_batch()
    model = _RecordingModel(channels=3)
    config = SFDLossConfig(
        semantic_channels=2,
        texture_channels=1,
        semantic_loss_weight=2.0,
        loss_weighting_scheme="none",
    )
    noise_sem = torch.tensor([[[[2.0, 1.0]], [[1.0, 8.0]]]])
    noise_tex = torch.tensor([[[[9.0, 4.0]]]])

    loss, metrics = compute_sfd_loss(
        encoded=encoded,
        model=model,
        noise_scheduler=_Scheduler(),
        pipeline_cls=_Pipeline,
        config=config,
        u=torch.tensor([0.5]),
        delta_t=torch.tensor([0.1]),
        noise_sem=noise_sem,
        noise_tex=noise_tex,
    )

    target_sem = noise_sem - encoded["semantic_latents"]
    target_tex = noise_tex - encoded["texture_latents"]
    mse = torch.cat((target_sem, target_tex), dim=1).square()
    expected = (
        mse * torch.tensor([2.0, 2.0, 1.0]).view(1, 3, 1, 1)
    ).mean()
    torch.testing.assert_close(loss, expected)
    assert metrics["loss_total"] == pytest.approx(float(expected))
    assert metrics["loss_sem"] == pytest.approx(float(target_sem.square().mean()))
    assert metrics["loss_tex"] == pytest.approx(float(target_tex.square().mean()))
    assert metrics["text_drop_ratio"] == 0.25
    # index floor(0.55 * 9) = 4 and floor(0.45 * 9) = 4
    torch.testing.assert_close(model.last_call["timestep_sem"], torch.tensor([0.6]))
    torch.testing.assert_close(model.last_call["timestep_tex"], torch.tensor([0.6]))

    loss.backward()
    assert model.scale.grad is not None
    assert torch.isfinite(model.scale.grad)


def test_schedule_uses_independent_semantic_and_texture_scheduler_indices():
    model = _RecordingModel(channels=3)
    compute_sfd_loss(
        encoded=_encoded_batch(),
        model=model,
        noise_scheduler=_Scheduler(),
        pipeline_cls=_Pipeline,
        config=SFDLossConfig(semantic_channels=2, texture_channels=1),
        u=torch.tensor([0.2]),
        delta_t=torch.tensor([0.3]),
        noise_sem=torch.zeros(1, 2, 1, 2),
        noise_tex=torch.zeros(1, 1, 1, 2),
    )
    # u_sem=.26 -> index 2 -> 0.8; u_tex=0 -> index 0 -> 1.0
    torch.testing.assert_close(model.last_call["timestep_sem"], torch.tensor([0.8]))
    torch.testing.assert_close(model.last_call["timestep_tex"], torch.tensor([1.0]))


def test_semantic_tokens_must_exactly_match_texture_grid_without_interpolation():
    with pytest.raises(ValueError, match="token length mismatch"):
        compose_semantic_texture_latents(
            semantic_latents=torch.zeros(1, 3, 2),
            texture_latents=torch.zeros(1, 1, 2, 2),
            pipeline_cls=_Pipeline,
            semantic_channels=2,
            expected_grid=(2, 2),
        )


class _TextureCodec:
    def __init__(self):
        self.grad_enabled = None

    def encode_texture(self, images, pipeline_cls):
        self.grad_enabled = torch.is_grad_enabled()
        return images[:, :1]


class _SemanticCodec:
    def __init__(self):
        self.grad_enabled = []

    def extract_features(self, values):
        self.grad_enabled.append(torch.is_grad_enabled())
        return values

    def compress_features(self, values, sample=False):
        assert sample is False
        self.grad_enabled.append(torch.is_grad_enabled())
        return values

    def normalize_latents(self, values):
        self.grad_enabled.append(torch.is_grad_enabled())
        return values


class _TextEncoder:
    def __init__(self):
        self.grad_enabled = None

    def encode(self, captions):
        self.grad_enabled = torch.is_grad_enabled()
        return torch.ones(len(captions), 1, 2), torch.zeros(len(captions), 1, 4)


def test_frozen_batch_encoding_is_no_grad_and_returns_detached_outputs():
    texture = _TextureCodec()
    semantic = _SemanticCodec()
    text = _TextEncoder()
    encoded = encode_frozen_batch(
        pixel_values=torch.ones(1, 1, 2, 2, requires_grad=True),
        vfm_pixel_values=torch.ones(1, 4, 2, requires_grad=True),
        captions=["caption"],
        texture_codec=texture,
        semantic_codec=semantic,
        text_encoder=text,
        pipeline_cls=_Pipeline,
        semantic_channels=2,
        texture_channels=1,
        expected_grid=(2, 2),
        drop_text_probability=0.0,
    )
    assert texture.grad_enabled is False
    assert semantic.grad_enabled == [False, False, False]
    assert text.grad_enabled is False
    for key in (
        "composite_latents",
        "semantic_latents",
        "texture_latents",
        "latent_ids",
        "prompt_embeds",
        "text_ids",
    ):
        assert encoded[key].requires_grad is False


def test_repa_output_is_rejected():
    class _RepaModel(_RecordingModel):
        def forward(self, **kwargs):
            return super().forward(**kwargs), torch.ones(1)

    with pytest.raises(RuntimeError, match="REPA"):
        compute_sfd_loss(
            encoded=_encoded_batch(),
            model=_RepaModel(3),
            noise_scheduler=_Scheduler(),
            pipeline_cls=_Pipeline,
            config=SFDLossConfig(semantic_channels=2, texture_channels=1),
            u=torch.tensor([0.5]),
            delta_t=torch.tensor([0.1]),
        )
