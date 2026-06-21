"""Texture latent codec for SEFI-T2I."""

from __future__ import annotations

import torch
import torch.nn as nn
from torch import Tensor


class TextureLatentCodec(nn.Module):
    """Encode/decode and normalize texture latents for SEFI."""

    def __init__(
        self,
        texture_vae: nn.Module,
        texture_vae_name: str,
    ):
        super().__init__()
        self.texture_vae = texture_vae
        self.texture_vae_name = str(texture_vae_name)
        self._use_flux2_bn = self.texture_vae_name == "flux2"

        config = getattr(texture_vae, "config", None)
        latent_channels = getattr(config, "latent_channels", None)
        if latent_channels is None:
            raise ValueError(
                "Texture VAE config must provide latent_channels for channel derivation."
            )
        self.latent_channels = int(latent_channels)
        self.texture_channels = int(self.latent_channels * 4)

        if self._use_flux2_bn:
            if not hasattr(texture_vae, "bn"):
                raise ValueError(
                    f"Texture VAE '{self.texture_vae_name}' requires bn stats but no bn module found."
                )
            eps = float(getattr(config, "batch_norm_eps", 1e-6))
            bn_mean = texture_vae.bn.running_mean.view(1, -1, 1, 1).float()
            bn_std = torch.sqrt(texture_vae.bn.running_var.view(1, -1, 1, 1).float() + eps)
            self.register_buffer("texture_bn_mean", bn_mean, persistent=False)
            self.register_buffer("texture_bn_std", bn_std, persistent=False)
            self.scaling_factor = None
            self.shift_factor = None
        else:
            scaling_factor = float(getattr(config, "scaling_factor", 1.0))
            shift_factor = float(getattr(config, "shift_factor", 0.0) or 0.0)
            if scaling_factor <= 0:
                raise ValueError(
                    f"Invalid scaling_factor={scaling_factor} for texture VAE {self.texture_vae_name}."
                )
            self.scaling_factor = scaling_factor
            self.shift_factor = shift_factor

    @property
    def vae_dtype(self) -> torch.dtype:
        return next(self.texture_vae.parameters()).dtype

    @torch.no_grad()
    def _encode_raw(self, images: Tensor) -> Tensor:
        return self.texture_vae.encode(images.to(dtype=self.vae_dtype)).latent_dist.mode()

    def _normalize_raw(self, raw_latents: Tensor) -> Tensor:
        return (raw_latents - self.shift_factor) * self.scaling_factor

    def _denormalize_raw(self, normed_latents: Tensor) -> Tensor:
        return normed_latents / self.scaling_factor + self.shift_factor

    def _normalize_patchified(self, patchified_latents: Tensor) -> Tensor:
        bn_mean = self.texture_bn_mean.to(patchified_latents.device, patchified_latents.dtype)
        bn_std = self.texture_bn_std.to(patchified_latents.device, patchified_latents.dtype)
        return (patchified_latents - bn_mean) / bn_std

    def _denormalize_patchified(self, patchified_latents: Tensor) -> Tensor:
        bn_mean = self.texture_bn_mean.to(patchified_latents.device, patchified_latents.dtype)
        bn_std = self.texture_bn_std.to(patchified_latents.device, patchified_latents.dtype)
        return patchified_latents * bn_std + bn_mean

    @torch.no_grad()
    def encode_texture(self, images: Tensor, pipeline_cls) -> Tensor:
        raw_latents = self._encode_raw(images)
        if self._use_flux2_bn:
            patchified = pipeline_cls._patchify_latents(raw_latents)
            patchified = self._normalize_patchified(patchified)
        else:
            normed_raw = self._normalize_raw(raw_latents)
            patchified = pipeline_cls._patchify_latents(normed_raw)

        if patchified.shape[1] != self.texture_channels:
            raise ValueError(
                f"Texture channels mismatch: derived={self.texture_channels}, got={patchified.shape[1]}."
            )

        return patchified

    @torch.no_grad()
    def decode_texture(self, texture_latents: Tensor, pipeline_cls) -> Tensor:
        if self._use_flux2_bn:
            patchified = self._denormalize_patchified(texture_latents)
            raw_latents = pipeline_cls._unpatchify_latents(patchified)
        else:
            raw_normed = pipeline_cls._unpatchify_latents(texture_latents)
            raw_latents = self._denormalize_raw(raw_normed)
        return self.texture_vae.decode(raw_latents, return_dict=False)[0]
