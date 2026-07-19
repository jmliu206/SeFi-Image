"""SEFI text-to-image inference package."""

from typing import TYPE_CHECKING

from .pipeline import SEFIInferencePipeline
from .registry import ModelSpec, infer_model_spec

if TYPE_CHECKING:
    from .semvae import SemVAEFeatureCodec, SemVAEOutput

__all__ = [
    "ModelSpec",
    "SEFIInferencePipeline",
    "infer_model_spec",
]


def __getattr__(name: str):
    """Load optional SemVAE components only when explicitly requested."""

    if name in {"SemVAEFeatureCodec", "SemVAEOutput"}:
        from .semvae import SemVAEFeatureCodec, SemVAEOutput

        exports = {
            "SemVAEFeatureCodec": SemVAEFeatureCodec,
            "SemVAEOutput": SemVAEOutput,
        }
        globals().update(exports)
        return exports[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
