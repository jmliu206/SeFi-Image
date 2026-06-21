"""Checkpoint-derived model metadata for SeFi-Image inference."""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from omegaconf import OmegaConf


ModelFamily = Literal["base", "rl", "turbo"]


@dataclass(frozen=True)
class ModelSpec:
    name: str
    family: ModelFamily
    scale: str
    default_height: int = 1024
    default_width: int = 1024
    default_steps: int = 50
    default_guidance_scale: float = 4.0
    default_delta_t: float | None = None
    default_timestep_shift_alpha: float = 1.0
    default_dtype: str = "bf16"

    @property
    def is_distilled(self) -> bool:
        return self.family == "turbo"


def _string_option(config, *keys: str) -> str:
    for key in keys:
        value = OmegaConf.select(config, key, default=None)
        if value is not None:
            text = str(value).strip()
            if text:
                return text
    return ""


def _int_option(config, *keys: str, default: int) -> int:
    for key in keys:
        value = OmegaConf.select(config, key, default=None)
        if value is not None:
            return int(value)
    return int(default)


def _float_option(config, *keys: str, default: float | None) -> float | None:
    for key in keys:
        value = OmegaConf.select(config, key, default=None)
        if value is not None:
            return float(value)
    return default


def _checkpoint_hint(checkpoint_uri: str, checkpoint_path: str) -> str:
    parts = [checkpoint_uri, checkpoint_path]
    path = Path(checkpoint_path)
    parts.extend(str(part) for part in path.parts[-4:])
    return " ".join(parts).lower()


def _normalize_family(value: str) -> ModelFamily | None:
    text = value.strip().lower().replace("_", "-")
    if text in {"base", "sft"}:
        return "base"
    if text in {"rl", "reward", "posttrain", "post-training"}:
        return "rl"
    if text in {"turbo", "distill", "distilled", "dmd", "dmd2"}:
        return "turbo"
    return None


def _infer_family(config, checkpoint_uri: str, checkpoint_path: str) -> ModelFamily:
    configured = _string_option(
        config,
        "inference.family",
        "inference.variant",
        "model.family",
        "model.variant",
    )
    if configured:
        family = _normalize_family(configured)
        if family is None:
            raise ValueError(
                "Unsupported SeFi-Image model family in config: "
                f"{configured}. Expected base, rl, or turbo."
            )
        return family

    hint = _checkpoint_hint(checkpoint_uri, checkpoint_path)
    normalized = re.sub(r"[^a-z0-9]+", "-", hint)
    if "turbo" in normalized or "distill" in normalized or "dmd" in normalized:
        return "turbo"
    if re.search(r"(^|-)rl($|-)", normalized) or "scalar" in normalized:
        return "rl"
    if "base" in normalized or "sft" in normalized:
        return "base"

    raise ValueError(
        "Could not infer SeFi-Image checkpoint family from checkpoint name. "
        "Use a checkpoint path or Hugging Face repo id containing Base, RL, or "
        "Turbo, or add inference.family to sefi_config.yaml."
    )


def _infer_scale(config, checkpoint_uri: str, checkpoint_path: str) -> str:
    configured = _string_option(config, "model.transformer_scale")
    if configured and configured != "custom":
        return configured.lower()

    model_name = _string_option(config, "model.model_name").lower()
    match = re.search(r"([0-9]+(?:p[0-9]+)?b)", model_name)
    if match:
        return match.group(1)

    hint = _checkpoint_hint(checkpoint_uri, checkpoint_path)
    match = re.search(r"([0-9]+(?:p[0-9]+)?b)", hint)
    if match:
        return match.group(1).lower()

    raise ValueError(
        "Could not infer SeFi-Image model scale from config or checkpoint name."
    )


def _default_name(family: ModelFamily, scale: str) -> str:
    public_scale = scale.upper().replace("P", ".")
    suffix = {"base": "Base", "rl": "RL", "turbo": "turbo"}[family]
    return f"SeFi-Image-{public_scale}-{suffix}"


def _default_steps(family: ModelFamily) -> int:
    return 4 if family == "turbo" else 50


def _default_guidance_scale(family: ModelFamily) -> float:
    return 1.0 if family == "turbo" else 4.0


def _default_timestep_shift_alpha(family: ModelFamily) -> float:
    return 0.3 if family in {"base", "rl"} else 1.0


def _default_dtype(config) -> str:
    dtype = _string_option(config, "inference.dtype", "training.mixed_precision")
    dtype = dtype.lower()
    if dtype in {"bf16", "bfloat16"}:
        return "bf16"
    if dtype in {"fp32", "float32", "no", "none"}:
        return "fp32"
    return "bf16"


def infer_model_spec(
    config,
    *,
    checkpoint_uri: str,
    checkpoint_path: str,
) -> ModelSpec:
    family = _infer_family(config, checkpoint_uri, checkpoint_path)
    scale = _infer_scale(config, checkpoint_uri, checkpoint_path)
    resolution = _int_option(config, "data.resolution", default=1024)
    height = _int_option(config, "inference.height", "data.height", default=resolution)
    width = _int_option(config, "inference.width", "data.width", default=resolution)
    name = _string_option(config, "inference.model_name", "model.display_name")

    return ModelSpec(
        name=name or _default_name(family, scale),
        family=family,
        scale=scale,
        default_height=height,
        default_width=width,
        default_steps=_int_option(
            config,
            "inference.steps",
            "inference.default_steps",
            default=_default_steps(family),
        ),
        default_guidance_scale=_float_option(
            config,
            "inference.guidance_scale",
            "inference.default_guidance_scale",
            default=_default_guidance_scale(family),
        )
        or _default_guidance_scale(family),
        default_delta_t=_float_option(
            config,
            "inference.delta_t",
            default=None,
        ),
        default_timestep_shift_alpha=_float_option(
            config,
            "inference.timestep_shift_alpha",
            default=_default_timestep_shift_alpha(family),
        )
        or _default_timestep_shift_alpha(family),
        default_dtype=_default_dtype(config),
    )
