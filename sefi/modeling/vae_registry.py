"""VAE loading helpers for SEFI inference artifacts."""

from __future__ import annotations

import torch


def load_vae_from_path(
    model_name: str,
    base_path: str,
    *,
    torch_dtype: torch.dtype | None = None,
    local_files_only: bool = True,
):
    """Load a final inference VAE from an explicit artifact path."""
    import diffusers

    kwargs = {"local_files_only": local_files_only}
    if torch_dtype is not None:
        kwargs["torch_dtype"] = torch_dtype

    if model_name == "sd1.5":
        return diffusers.models.AutoencoderKL.from_pretrained(base_path, **kwargs)

    if model_name in {"flux", "flux1"}:
        return diffusers.models.AutoencoderKL.from_pretrained(
            base_path,
            subfolder="vae",
            **kwargs,
        )

    if model_name == "flux2":
        return diffusers.models.AutoencoderKLFlux2.from_pretrained(
            base_path,
            subfolder="vae",
            **kwargs,
        )

    raise ValueError(
        f"Unsupported texture VAE model_name={model_name}. "
        "Supported values: sd1.5, flux1, flux2."
    )
