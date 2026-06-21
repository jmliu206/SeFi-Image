"""Texture VAE factory for SEFI-T2I."""

from __future__ import annotations

from collections.abc import Mapping

import torch

from .vae_registry import load_vae_from_path

SUPPORTED_TEXTURE_VAE_NAMES = {
    "sd1.5",
    "flux1",
    "flux2",
}


def _normalize_texture_vae_name(name: str) -> str:
    normalized = str(name).strip().lower()
    if normalized not in SUPPORTED_TEXTURE_VAE_NAMES:
        raise ValueError(
            f"Unsupported model.texture_vae.name={name}. "
            f"Expected one of {sorted(SUPPORTED_TEXTURE_VAE_NAMES)}."
        )
    return normalized


def build_texture_vae(texture_vae_cfg: Mapping, *, torch_dtype: torch.dtype):
    """Build the final texture VAE packaged in a SEFI inference artifact."""
    name = _normalize_texture_vae_name(str(texture_vae_cfg.get("name", "")))
    base_path = str(texture_vae_cfg.get("base_path", "")).strip()
    if not base_path:
        raise ValueError("model.texture_vae.base_path is required.")

    load_name = "flux2" if name == "flux2" else name
    return load_vae_from_path(
        load_name,
        base_path,
        torch_dtype=torch_dtype,
        local_files_only=True,
    )
