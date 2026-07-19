"""Shared semantic-first flow-distillation (SFD) fine-tuning core.

This module intentionally contains no Accelerate, DeepSpeed, PEFT, or dataset
policy.  LoRA and full fine-tuning call the same frozen encoders and loss
implementation, which keeps the numerically important training semantics in a
small, independently testable unit.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import torch
import torch.nn as nn
from torch import Tensor


FIXED_TRAIN_RESOLUTION = 1024
FIXED_LATENT_GRID = (64, 64)
_T_SAMPLING_SCHEMES = {
    "uniform",
    "logit_normal",
    "logit_normal_with_uniform",
    "mode",
}
_LOSS_WEIGHTING_SCHEMES = {"none", "sigma_sqrt", "cosmap"}


def _value(container: Any, key: str, default: Any = None) -> Any:
    if isinstance(container, Mapping):
        return container.get(key, default)
    return getattr(container, key, default)


@dataclass(frozen=True)
class SFDLossConfig:
    """Numerical configuration for semantic-first flow matching."""

    semantic_channels: int
    texture_channels: int
    semantic_loss_weight: float = 1.0
    delta_t_min: float = 0.1
    delta_t_max: float = 0.1
    t_sampling_scheme: str = "logit_normal"
    logit_mean: float = -1.532
    logit_std: float = 1.0
    uniform_prob: float = 0.0
    mode_scale: float = 1.29
    loss_weighting_scheme: str = "none"

    def __post_init__(self) -> None:
        if int(self.semantic_channels) <= 0:
            raise ValueError("semantic_channels must be positive.")
        if int(self.texture_channels) <= 0:
            raise ValueError("texture_channels must be positive.")
        if float(self.semantic_loss_weight) < 0:
            raise ValueError("semantic_loss_weight must be non-negative.")
        if not 0.0 <= float(self.delta_t_min) <= 1.0:
            raise ValueError("delta_t_min must be in [0, 1].")
        if not 0.0 <= float(self.delta_t_max) <= 1.0:
            raise ValueError("delta_t_max must be in [0, 1].")
        if float(self.delta_t_min) > float(self.delta_t_max):
            raise ValueError("delta_t_min must be <= delta_t_max.")

        scheme = str(self.t_sampling_scheme).strip().lower()
        if scheme not in _T_SAMPLING_SCHEMES:
            raise ValueError(
                f"t_sampling_scheme must be one of {sorted(_T_SAMPLING_SCHEMES)}, "
                f"got {self.t_sampling_scheme!r}."
            )
        if scheme in {"logit_normal", "logit_normal_with_uniform"}:
            if float(self.logit_std) <= 0:
                raise ValueError("logit_std must be positive for logit-normal sampling.")
        if not 0.0 <= float(self.uniform_prob) <= 1.0:
            raise ValueError("uniform_prob must be in [0, 1].")
        if scheme == "mode" and float(self.mode_scale) <= 0:
            raise ValueError("mode_scale must be positive for mode sampling.")

        weighting = str(self.loss_weighting_scheme).strip().lower()
        if weighting not in _LOSS_WEIGHTING_SCHEMES:
            raise ValueError(
                "loss_weighting_scheme must be one of "
                f"{sorted(_LOSS_WEIGHTING_SCHEMES)}, got "
                f"{self.loss_weighting_scheme!r}."
            )

    @classmethod
    def from_training_config(
        cls,
        training_config: Any,
        *,
        semantic_channels: int,
        texture_channels: int,
    ) -> "SFDLossConfig":
        """Build from the public YAML's ``training`` section.

        Keeping this adapter here avoids making the loss depend on a specific
        config library while retaining the reference config layout.
        """

        sfd = _value(training_config, "sfd", {})
        t_sampling = _value(training_config, "t_sampling", {})
        loss = _value(training_config, "loss", {})
        return cls(
            semantic_channels=int(semantic_channels),
            texture_channels=int(texture_channels),
            semantic_loss_weight=float(_value(sfd, "semantic_loss_weight", 1.0)),
            delta_t_min=float(_value(sfd, "delta_t_min", 0.1)),
            delta_t_max=float(_value(sfd, "delta_t_max", 0.1)),
            t_sampling_scheme=str(_value(t_sampling, "scheme", "logit_normal")),
            logit_mean=float(_value(t_sampling, "logit_mean", -1.532)),
            logit_std=float(_value(t_sampling, "logit_std", 1.0)),
            uniform_prob=float(_value(t_sampling, "uniform_prob", 0.0)),
            mode_scale=float(_value(t_sampling, "mode_scale", 1.29)),
            loss_weighting_scheme=str(_value(loss, "weighting_scheme", "none")),
        )


@dataclass(frozen=True)
class SFDSchedule:
    """Sampled semantic/texture times and scheduler lookups for one batch."""

    u: Tensor
    delta_t: Tensor
    u_sem: Tensor
    u_tex: Tensor
    timesteps_sem: Tensor
    timesteps_tex: Tensor
    sigmas_sem: Tensor
    sigmas_tex: Tensor


def freeze_module(module: nn.Module) -> nn.Module:
    """Put a frozen training component in eval mode and disable its gradients."""

    module.eval()
    module.requires_grad_(False)
    return module


def apply_text_dropout(
    captions: Sequence[str],
    *,
    drop_probability: float = 0.1,
    unconditional_prompt: str = "",
    generator: torch.Generator | None = None,
) -> tuple[list[str], float]:
    """Apply the reference CFG-style per-caption text dropout."""

    captions = [str(caption) for caption in captions]
    if not 0.0 <= float(drop_probability) <= 1.0:
        raise ValueError("drop_probability must be in [0, 1].")
    if not captions or float(drop_probability) == 0.0:
        return captions, 0.0

    keep_mask = torch.rand(len(captions), generator=generator) >= float(drop_probability)
    dropped = [
        caption if bool(keep_mask[index]) else str(unconditional_prompt)
        for index, caption in enumerate(captions)
    ]
    return dropped, 1.0 - float(keep_mask.float().mean().item())


def compose_semantic_texture_latents(
    *,
    semantic_latents: Tensor,
    texture_latents: Tensor,
    pipeline_cls: type,
    semantic_channels: int,
    texture_channels: int | None = None,
    expected_grid: tuple[int, int] | None = FIXED_LATENT_GRID,
) -> dict[str, Tensor]:
    """Reshape normalized SemVAE tokens and compose semantic-first latents.

    ``semantic_latents`` may be normalized tokens ``[B, L, C]`` or an already
    spatial tensor ``[B, C, H, W]``.  Interpolation is deliberately forbidden:
    the DINO/SemVAE token grid must exactly match the texture grid.
    """

    if texture_latents.ndim != 4:
        raise ValueError(
            "texture_latents must have shape [B, C, H, W], got "
            f"{tuple(texture_latents.shape)}."
        )
    batch_size, actual_texture_channels, height, width = texture_latents.shape
    if texture_channels is not None and actual_texture_channels != int(texture_channels):
        raise ValueError(
            f"Texture channels mismatch: got={actual_texture_channels}, "
            f"expected={int(texture_channels)}."
        )
    if expected_grid is not None and (height, width) != tuple(expected_grid):
        raise ValueError(
            f"Texture grid mismatch: got={height}x{width}, "
            f"expected={expected_grid[0]}x{expected_grid[1]}."
        )

    if semantic_latents.ndim == 3:
        sem_batch, seq_len, channels = semantic_latents.shape
        if sem_batch != batch_size:
            raise ValueError(
                f"Semantic/texture batch mismatch: {sem_batch} != {batch_size}."
            )
        if channels != int(semantic_channels):
            raise ValueError(
                f"Semantic channels mismatch: got={channels}, "
                f"expected={int(semantic_channels)}."
            )
        if seq_len != height * width:
            raise ValueError(
                f"Semantic token length mismatch: L={seq_len}, "
                f"expected={height * width} from texture grid {height}x{width}."
            )
        semantic_latents = semantic_latents.permute(0, 2, 1).reshape(
            batch_size, channels, height, width
        )
    elif semantic_latents.ndim == 4:
        expected_shape = (batch_size, int(semantic_channels), height, width)
        if tuple(semantic_latents.shape) != expected_shape:
            raise ValueError(
                f"Semantic spatial shape mismatch: got={tuple(semantic_latents.shape)}, "
                f"expected={expected_shape}."
            )
    else:
        raise ValueError(
            "semantic_latents must have shape [B, L, C] or [B, C, H, W], "
            f"got {tuple(semantic_latents.shape)}."
        )

    composite_latents = torch.cat([semantic_latents, texture_latents], dim=1)
    latent_ids = pipeline_cls._prepare_latent_ids(composite_latents).to(
        composite_latents.device
    )
    return {
        "composite_latents": composite_latents,
        "semantic_latents": semantic_latents,
        "texture_latents": texture_latents,
        "latent_ids": latent_ids,
    }


@torch.no_grad()
def encode_frozen_batch(
    *,
    pixel_values: Tensor,
    vfm_pixel_values: Tensor,
    captions: Sequence[str],
    texture_codec: Any,
    semantic_codec: Any,
    text_encoder: Any,
    pipeline_cls: type,
    semantic_channels: int,
    texture_channels: int | None = None,
    semantic_sample: bool = False,
    expected_grid: tuple[int, int] | None = FIXED_LATENT_GRID,
    drop_text_probability: float = 0.1,
    unconditional_prompt: str = "",
    dropout_generator: torch.Generator | None = None,
) -> dict[str, Tensor | float]:
    """Online-encode one batch using only frozen, no-grad components.

    The method uses the public SemVAE codec's low-level batch API and does not
    perform its demo-only reconstruction/cosine pass.  ``torch.no_grad`` is used
    instead of ``torch.inference_mode`` so outputs remain ordinary tensors that
    can safely be saved by the trainable DiT's autograd graph.
    """

    texture_latents = texture_codec.encode_texture(
        images=pixel_values,
        pipeline_cls=pipeline_cls,
    )
    features = semantic_codec.extract_features(vfm_pixel_values)
    semantic_tokens = semantic_codec.compress_features(
        features,
        sample=bool(semantic_sample),
    )
    semantic_tokens = semantic_codec.normalize_latents(semantic_tokens).float()

    composed = compose_semantic_texture_latents(
        semantic_latents=semantic_tokens,
        texture_latents=texture_latents,
        pipeline_cls=pipeline_cls,
        semantic_channels=int(semantic_channels),
        texture_channels=texture_channels,
        expected_grid=expected_grid,
    )

    dropped_captions, text_drop_ratio = apply_text_dropout(
        captions,
        drop_probability=float(drop_text_probability),
        unconditional_prompt=unconditional_prompt,
        generator=dropout_generator,
    )
    prompt_embeds, text_ids = text_encoder.encode(dropped_captions)

    # Be explicit about the frozen-boundary contract.  ``detach`` is cheap here
    # and also protects callers whose custom codec forgot its own no-grad guard.
    encoded: dict[str, Tensor | float] = {
        key: value.detach() for key, value in composed.items()
    }
    encoded.update(
        {
            "prompt_embeds": prompt_embeds.detach(),
            "text_ids": text_ids.detach(),
            "text_drop_ratio": float(text_drop_ratio),
        }
    )
    return encoded


def sample_timestep_u(
    config: SFDLossConfig,
    *,
    batch_size: int,
    device: torch.device | str,
    generator: torch.Generator | None = None,
) -> Tensor:
    """Sample continuous flow time with reference-repository semantics."""

    if int(batch_size) <= 0:
        raise ValueError("batch_size must be positive.")
    scheme = str(config.t_sampling_scheme).strip().lower()
    if scheme == "uniform":
        return torch.rand((int(batch_size),), device=device, generator=generator)

    if (
        scheme == "logit_normal_with_uniform"
        and float(config.uniform_prob) == 1.0
    ):
        # Match the reference RNG consumption: do not draw a logit-normal
        # sample when every row is selected from the uniform branch.
        return torch.rand((int(batch_size),), device=device, generator=generator)

    if scheme in {"logit_normal", "logit_normal_with_uniform"}:
        normal = torch.normal(
            mean=float(config.logit_mean),
            std=float(config.logit_std),
            size=(int(batch_size),),
            device=device,
            generator=generator,
        )
        logit_normal = torch.nn.functional.sigmoid(normal)
        if scheme == "logit_normal":
            return logit_normal
        if float(config.uniform_prob) == 0.0:
            return logit_normal
        use_uniform = (
            torch.rand((int(batch_size),), device=device, generator=generator)
            < float(config.uniform_prob)
        )
        uniform = torch.rand((int(batch_size),), device=device, generator=generator)
        return torch.where(use_uniform, uniform, logit_normal)

    uniform = torch.rand((int(batch_size),), device=device, generator=generator)
    return 1 - uniform - float(config.mode_scale) * (
        torch.cos(math.pi * uniform / 2).square() - 1 + uniform
    )


def sample_delta_t(
    config: SFDLossConfig,
    *,
    batch_size: int,
    device: torch.device | str,
    generator: torch.Generator | None = None,
) -> Tensor:
    if float(config.delta_t_min) == float(config.delta_t_max):
        return torch.full(
            (int(batch_size),),
            float(config.delta_t_min),
            device=device,
        )
    return float(config.delta_t_min) + torch.rand(
        (int(batch_size),), device=device, generator=generator
    ) * (float(config.delta_t_max) - float(config.delta_t_min))


def scheduler_timesteps_and_sigmas(
    noise_scheduler: Any,
    u_continuous: Tensor,
    *,
    n_dim: int,
    dtype: torch.dtype,
) -> tuple[Tensor, Tensor]:
    """Quantize continuous time using the exact reference scheduler lookup."""

    config = getattr(noise_scheduler, "config", None)
    num_steps = int(_value(config, "num_train_timesteps", 0))
    if num_steps <= 0:
        raise ValueError("noise_scheduler.config.num_train_timesteps must be positive.")
    if len(noise_scheduler.timesteps) < num_steps or len(noise_scheduler.sigmas) < num_steps:
        raise ValueError(
            "noise_scheduler timesteps/sigmas are shorter than num_train_timesteps."
        )

    indices = (u_continuous * (num_steps - 1)).long().clamp(0, num_steps - 1)
    timesteps = noise_scheduler.timesteps[indices.cpu()].to(u_continuous.device)
    sigmas = noise_scheduler.sigmas[indices.cpu()].to(
        device=u_continuous.device,
        dtype=dtype,
    )
    while sigmas.ndim < int(n_dim):
        sigmas = sigmas.unsqueeze(-1)
    return timesteps, sigmas


def build_sfd_schedule(
    *,
    noise_scheduler: Any,
    config: SFDLossConfig,
    batch_size: int,
    device: torch.device | str,
    semantic_ndim: int,
    texture_ndim: int,
    semantic_dtype: torch.dtype,
    texture_dtype: torch.dtype,
    generator: torch.Generator | None = None,
    u: Tensor | None = None,
    delta_t: Tensor | None = None,
) -> SFDSchedule:
    """Build the semantic-first dual-time schedule.

    The three schedule equations below are intentionally kept verbatim from the
    reference implementation and must not be replaced by a symmetric offset.
    """

    if u is None:
        u = sample_timestep_u(
            config,
            batch_size=int(batch_size),
            device=device,
            generator=generator,
        )
    else:
        u = u.to(device=device)
    if delta_t is None:
        delta_t = sample_delta_t(
            config,
            batch_size=int(batch_size),
            device=device,
            generator=generator,
        )
    else:
        delta_t = delta_t.to(device=device)
    if tuple(u.shape) != (int(batch_size),) or tuple(delta_t.shape) != (int(batch_size),):
        raise ValueError(
            f"u and delta_t must both have shape [{int(batch_size)}], "
            f"got {tuple(u.shape)} and {tuple(delta_t.shape)}."
        )

    u_sem_raw = u * (1 + delta_t)
    u_tex = torch.clamp(u_sem_raw - delta_t, min=0.0)
    u_sem = torch.clamp(u_sem_raw, max=1.0)

    timesteps_sem, sigmas_sem = scheduler_timesteps_and_sigmas(
        noise_scheduler,
        u_sem,
        n_dim=int(semantic_ndim),
        dtype=semantic_dtype,
    )
    timesteps_tex, sigmas_tex = scheduler_timesteps_and_sigmas(
        noise_scheduler,
        u_tex,
        n_dim=int(texture_ndim),
        dtype=texture_dtype,
    )
    return SFDSchedule(
        u=u,
        delta_t=delta_t,
        u_sem=u_sem,
        u_tex=u_tex,
        timesteps_sem=timesteps_sem,
        timesteps_tex=timesteps_tex,
        sigmas_sem=sigmas_sem,
        sigmas_tex=sigmas_tex,
    )


def compute_loss_weighting(weighting_scheme: str, sigmas: Tensor) -> Tensor:
    scheme = str(weighting_scheme).strip().lower()
    if scheme not in _LOSS_WEIGHTING_SCHEMES:
        raise ValueError(
            f"Unknown loss weighting scheme {weighting_scheme!r}; "
            f"expected one of {sorted(_LOSS_WEIGHTING_SCHEMES)}."
        )
    if scheme == "sigma_sqrt":
        return (sigmas**-2.0).float()
    if scheme == "cosmap":
        denominator = 1 - 2 * sigmas + 2 * sigmas.square()
        return 2 / (math.pi * denominator)
    return torch.ones_like(sigmas)


def _prediction_tensor(model_output: Any) -> Tensor:
    if torch.is_tensor(model_output):
        return model_output
    if isinstance(model_output, (tuple, list)):
        if not model_output or not torch.is_tensor(model_output[0]):
            raise TypeError("Transformer tuple output must start with a tensor.")
        if len(model_output) > 1 and model_output[1] is not None:
            raise RuntimeError("REPA output is not supported by the fine-tuning demo.")
        return model_output[0]
    sample = getattr(model_output, "sample", None)
    if torch.is_tensor(sample):
        return sample
    raise TypeError(
        "Transformer must return a tensor, a (tensor, None) tuple, or an object "
        "with a tensor `.sample`."
    )


def _validate_encoded_batch(
    encoded: Mapping[str, Any], config: SFDLossConfig
) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
    required = (
        "semantic_latents",
        "texture_latents",
        "latent_ids",
        "prompt_embeds",
        "text_ids",
    )
    missing = [key for key in required if key not in encoded]
    if missing:
        raise KeyError(f"Encoded SFD batch is missing keys: {missing}.")

    z_sem = encoded["semantic_latents"]
    z_tex = encoded["texture_latents"]
    latent_ids = encoded["latent_ids"]
    prompt_embeds = encoded["prompt_embeds"]
    text_ids = encoded["text_ids"]
    tensors = (z_sem, z_tex, latent_ids, prompt_embeds, text_ids)
    if not all(torch.is_tensor(value) for value in tensors):
        raise TypeError("All encoded SFD batch fields except metrics must be tensors.")
    if z_sem.ndim != 4 or z_tex.ndim != 4:
        raise ValueError("Semantic and texture latents must both be [B, C, H, W].")
    if z_sem.shape[0] != z_tex.shape[0] or z_sem.shape[2:] != z_tex.shape[2:]:
        raise ValueError(
            "Semantic/texture batch and grid must match exactly, got "
            f"{tuple(z_sem.shape)} and {tuple(z_tex.shape)}."
        )
    if z_sem.shape[1] != int(config.semantic_channels):
        raise ValueError(
            f"Semantic channels mismatch: got={z_sem.shape[1]}, "
            f"expected={int(config.semantic_channels)}."
        )
    if z_tex.shape[1] != int(config.texture_channels):
        raise ValueError(
            f"Texture channels mismatch: got={z_tex.shape[1]}, "
            f"expected={int(config.texture_channels)}."
        )
    return z_sem, z_tex, latent_ids, prompt_embeds, text_ids


def compute_sfd_loss(
    *,
    encoded: Mapping[str, Any],
    model: nn.Module,
    noise_scheduler: Any,
    pipeline_cls: type,
    config: SFDLossConfig,
    generator: torch.Generator | None = None,
    u: Tensor | None = None,
    delta_t: Tensor | None = None,
    noise_sem: Tensor | None = None,
    noise_tex: Tensor | None = None,
) -> tuple[Tensor, dict[str, float]]:
    """Compute the exact no-REPA semantic-first dual-timestep SFD loss.

    Optional fixed ``u``, ``delta_t``, and noise tensors are useful for parity
    tests; normal training leaves them unset.
    """

    z_sem, z_tex, latent_ids, prompt_embeds, text_ids = _validate_encoded_batch(
        encoded, config
    )
    batch_size = z_sem.shape[0]
    schedule = build_sfd_schedule(
        noise_scheduler=noise_scheduler,
        config=config,
        batch_size=batch_size,
        device=z_sem.device,
        semantic_ndim=z_sem.ndim,
        texture_ndim=z_tex.ndim,
        semantic_dtype=z_sem.dtype,
        texture_dtype=z_tex.dtype,
        generator=generator,
        u=u,
        delta_t=delta_t,
    )

    if noise_sem is None:
        noise_sem = torch.randn(
            z_sem.shape,
            device=z_sem.device,
            dtype=z_sem.dtype,
            generator=generator,
        )
    else:
        noise_sem = noise_sem.to(device=z_sem.device, dtype=z_sem.dtype)
    if noise_tex is None:
        noise_tex = torch.randn(
            z_tex.shape,
            device=z_tex.device,
            dtype=z_tex.dtype,
            generator=generator,
        )
    else:
        noise_tex = noise_tex.to(device=z_tex.device, dtype=z_tex.dtype)
    if noise_sem.shape != z_sem.shape or noise_tex.shape != z_tex.shape:
        raise ValueError("Provided semantic/texture noise must match latent shapes.")

    xt_sem = (1.0 - schedule.sigmas_sem) * z_sem + schedule.sigmas_sem * noise_sem
    xt_tex = (1.0 - schedule.sigmas_tex) * z_tex + schedule.sigmas_tex * noise_tex
    packed_noisy = pipeline_cls._pack_latents(torch.cat([xt_sem, xt_tex], dim=1))

    model_output = model(
        hidden_states=packed_noisy,
        timestep_sem=schedule.timesteps_sem / 1000,
        timestep_tex=schedule.timesteps_tex / 1000,
        encoder_hidden_states=prompt_embeds,
        txt_ids=text_ids,
        img_ids=latent_ids,
    )
    model_pred = _prediction_tensor(model_output)
    model_pred = model_pred[:, : packed_noisy.size(1)]
    model_pred = pipeline_cls._unpack_latents_with_ids(model_pred, latent_ids)

    target = torch.cat([noise_sem - z_sem, noise_tex - z_tex], dim=1)
    if model_pred.shape != target.shape:
        raise ValueError(
            f"Unpacked model prediction shape {tuple(model_pred.shape)} does not "
            f"match SFD target {tuple(target.shape)}."
        )
    mse = (model_pred.float() - target.float()).square()

    sigma_weight_sem = compute_loss_weighting(
        config.loss_weighting_scheme, schedule.sigmas_sem
    )
    sigma_weight_tex = compute_loss_weighting(
        config.loss_weighting_scheme, schedule.sigmas_tex
    )
    sigma_weight = torch.cat(
        [
            sigma_weight_sem.expand(-1, int(config.semantic_channels), -1, -1),
            sigma_weight_tex.expand(-1, int(config.texture_channels), -1, -1),
        ],
        dim=1,
    ).to(mse.dtype)

    channel_weights = torch.ones(
        (int(config.semantic_channels) + int(config.texture_channels),),
        device=z_sem.device,
        dtype=mse.dtype,
    )
    channel_weights[: int(config.semantic_channels)] = float(
        config.semantic_loss_weight
    )
    pred_loss = (
        mse * sigma_weight * channel_weights.view(1, -1, 1, 1)
    ).mean()
    sem_loss = mse[:, : int(config.semantic_channels)].mean()
    tex_loss = mse[:, int(config.semantic_channels) :].mean()

    metrics = {
        "loss_pred": float(pred_loss.detach().item()),
        "loss_sem": float(sem_loss.detach().item()),
        "loss_tex": float(tex_loss.detach().item()),
        "u_sem_mean": float(schedule.u_sem.mean().item()),
        "u_tex_mean": float(schedule.u_tex.mean().item()),
        "delta_t_mean": float(schedule.delta_t.mean().item()),
        "delta_t_min": float(schedule.delta_t.min().item()),
        "delta_t_max": float(schedule.delta_t.max().item()),
        "sigma_weight_sem_mean": float(sigma_weight_sem.mean().item()),
        "sigma_weight_tex_mean": float(sigma_weight_tex.mean().item()),
        "text_drop_ratio": float(encoded.get("text_drop_ratio", 0.0)),
        "loss_total": float(pred_loss.detach().item()),
    }
    return pred_loss, metrics
