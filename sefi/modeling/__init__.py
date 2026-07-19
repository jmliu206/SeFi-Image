"""SEFI inference model components, loaded only when requested."""

from __future__ import annotations

from importlib import import_module
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .flux2_sefi_transformer import Flux2SEFITransformer2DModel
    from .qwen3vl_text_encoder import Qwen3VLTextEncoder
    from .semvae import (
        DiagonalGaussianDistribution,
        SemVAEConfig,
        SemanticVariationalAutoEncoder,
    )
    from .texture_latent_codec import TextureLatentCodec
    from .texture_vae_factory import build_texture_vae


__all__ = [
    "Flux2SEFITransformer2DModel",
    "Qwen3VLTextEncoder",
    "DiagonalGaussianDistribution",
    "SemVAEConfig",
    "SemanticVariationalAutoEncoder",
    "TextureLatentCodec",
    "build_texture_vae",
]


_LAZY_EXPORTS = {
    "Flux2SEFITransformer2DModel": (
        ".flux2_sefi_transformer",
        "Flux2SEFITransformer2DModel",
    ),
    "Qwen3VLTextEncoder": (".qwen3vl_text_encoder", "Qwen3VLTextEncoder"),
    "DiagonalGaussianDistribution": (".semvae", "DiagonalGaussianDistribution"),
    "SemVAEConfig": (".semvae", "SemVAEConfig"),
    "SemanticVariationalAutoEncoder": (".semvae", "SemanticVariationalAutoEncoder"),
    "TextureLatentCodec": (".texture_latent_codec", "TextureLatentCodec"),
    "build_texture_vae": (".texture_vae_factory", "build_texture_vae"),
}


def __getattr__(name: str):
    """Import one component without eagerly loading unrelated model stacks."""

    target = _LAZY_EXPORTS.get(name)
    if target is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module_name, attribute_name = target
    value = getattr(import_module(module_name, __name__), attribute_name)
    globals()[name] = value
    return value
