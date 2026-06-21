"""Inference-only component builder for SEFI models."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass

import torch

from .modeling import (
    Flux2SEFITransformer2DModel,
    Qwen3VLTextEncoder,
    TextureLatentCodec,
    build_texture_vae,
)


SEFI_SCALE_PRESETS = {
    "0p5b": {
        "attention_head_dim": 128,
        "num_attention_heads": 12,
        "num_layers": 3,
        "num_single_layers": 10,
        "joint_attention_dim": 6144,
    },
    "1b": {
        "attention_head_dim": 128,
        "num_attention_heads": 16,
        "num_layers": 4,
        "num_single_layers": 12,
        "joint_attention_dim": 6144,
    },
    "2b": {
        "attention_head_dim": 128,
        "num_attention_heads": 20,
        "num_layers": 4,
        "num_single_layers": 16,
        "joint_attention_dim": 6144,
    },
    "3b": {
        "attention_head_dim": 128,
        "num_attention_heads": 22,
        "num_layers": 5,
        "num_single_layers": 18,
        "joint_attention_dim": 7680,
    },
    "4b": {
        "attention_head_dim": 128,
        "num_attention_heads": 24,
        "num_layers": 5,
        "num_single_layers": 20,
        "joint_attention_dim": 7680,
    },
    "5b": {
        "attention_head_dim": 128,
        "num_attention_heads": 26,
        "num_layers": 6,
        "num_single_layers": 21,
        "joint_attention_dim": 7680,
    },
    "6b": {
        "attention_head_dim": 128,
        "num_attention_heads": 28,
        "num_layers": 6,
        "num_single_layers": 22,
        "joint_attention_dim": 7680,
    },
    "8b": {
        "attention_head_dim": 128,
        "num_attention_heads": 30,
        "num_layers": 7,
        "num_single_layers": 24,
        "joint_attention_dim": 7680,
    },
    "9b": {
        "attention_head_dim": 128,
        "num_attention_heads": 32,
        "num_layers": 8,
        "num_single_layers": 24,
        "joint_attention_dim": 12288,
    },
}

SEFI_MODEL_NAME_TO_SCALE = {
    "flux2-klein-base-0p5b-sefi": "0p5b",
    "flux2-klein-base-1b-sefi": "1b",
    "flux2-klein-base-2b-sefi": "2b",
    "flux2-klein-base-3b-sefi": "3b",
    "flux2-klein-base-4b-sefi": "4b",
    "flux2-klein-base-5b-sefi": "5b",
    "flux2-klein-base-6b-sefi": "6b",
    "flux2-klein-base-8b-sefi": "8b",
    "flux2-klein-base-9b-sefi": "9b",
}

QWEN3VL_TEXT_HIDDEN_DIMS = {
    "qwen3vl_2b": 2048,
    "qwen3vl_4b": 2560,
    "qwen3vl_8b": 4096,
}

@dataclass
class SEFIComponents:
    transformer: torch.nn.Module
    text_encoder: torch.nn.Module
    texture_codec: torch.nn.Module
    noise_scheduler: object
    pipeline_cls: type
    semantic_channels: int
    texture_channels: int
    total_channels: int


def _resolve_transformer_scale(config) -> str:
    model_cfg = config.model
    scale = str(model_cfg.get("transformer_scale", "")).strip().lower()
    if scale:
        if scale not in set(SEFI_SCALE_PRESETS) | {"custom"}:
            raise ValueError(
                "model.transformer_scale must be one of "
                f"{list(SEFI_SCALE_PRESETS) + ['custom']}. Got: {scale}"
            )
        return scale

    model_name = str(model_cfg.model_name)
    try:
        return SEFI_MODEL_NAME_TO_SCALE[model_name]
    except KeyError as exc:
        raise ValueError(
            f"Unsupported SEFI model.model_name: {model_name}. "
            f"Expected one of {sorted(SEFI_MODEL_NAME_TO_SCALE)}."
        ) from exc


def _derive_semantic_channels(config) -> int:
    value = config.model.get("semantic_channels", None)
    if value is None:
        raise ValueError("Config requires model.semantic_channels for inference.")
    return int(value)


def _texture_vae_config_path(texture_vae_cfg) -> str:
    name = str(texture_vae_cfg.get("name", "")).strip().lower()
    base_path = str(texture_vae_cfg.get("base_path", "")).strip()
    if not base_path:
        raise ValueError("model.texture_vae.base_path is required.")
    if name == "sd1.5":
        return os.path.join(base_path, "config.json")
    if name in {"flux1", "flux2"}:
        return os.path.join(base_path, "vae", "config.json")
    raise ValueError(
        f"Unsupported model.texture_vae.name: {name}. "
        "Expected one of ['sd1.5', 'flux1', 'flux2']."
    )


def _derive_texture_channels(config) -> int:
    config_path = _texture_vae_config_path(config.model.texture_vae)
    if not os.path.isfile(config_path):
        raise FileNotFoundError(f"Texture VAE config not found: {config_path}")
    with open(config_path, "r", encoding="utf-8") as handle:
        texture_vae_config = json.load(handle)
    latent_channels = texture_vae_config.get("latent_channels", None)
    if latent_channels is None:
        raise ValueError(f"Texture VAE config must contain latent_channels: {config_path}")
    return int(latent_channels) * 4


def _derive_text_output_dim(config) -> int:
    text_cfg = config.model.text_encoder
    model_name = str(text_cfg.model_name)
    if model_name not in QWEN3VL_TEXT_HIDDEN_DIMS:
        raise ValueError(
            f"Unsupported SEFI text_encoder.model_name: {model_name}. "
            f"Expected one of {sorted(QWEN3VL_TEXT_HIDDEN_DIMS)}."
        )
    hidden_layers = tuple(int(x) for x in text_cfg.hidden_layers)
    return int(QWEN3VL_TEXT_HIDDEN_DIMS[model_name]) * len(hidden_layers)


def text_encoder_signature(config) -> tuple:
    text_cfg = config.model.text_encoder
    return (
        str(text_cfg.model_name),
        str(text_cfg.get("weights_root", "outputs/model_weights")),
        int(text_cfg.max_length),
        tuple(int(x) for x in text_cfg.hidden_layers),
    )


def build_transformer_config(config, *, total_channels: int, text_output_dim: int) -> dict:
    from diffusers import Flux2Transformer2DModel

    model_cfg = config.model
    transformer_cfg_path = str(model_cfg.assets.transformer_config_path)
    transformer_cfg = Flux2Transformer2DModel.load_config(
        transformer_cfg_path,
        subfolder="transformer",
        local_files_only=True,
    )
    transformer_cfg = dict(transformer_cfg)

    transformer_scale = _resolve_transformer_scale(config)
    if transformer_scale == "custom":
        overrides = model_cfg.get("transformer_overrides", {})
        required_keys = (
            "attention_head_dim",
            "num_attention_heads",
            "num_layers",
            "num_single_layers",
            "joint_attention_dim",
        )
        missing = [key for key in required_keys if key not in overrides]
        if missing:
            raise ValueError(
                "model.transformer_overrides is missing required keys for custom "
                f"SEFI model: {missing}"
            )
        for key in required_keys:
            transformer_cfg[key] = int(overrides[key])
        if "mlp_ratio" in overrides:
            transformer_cfg["mlp_ratio"] = float(overrides["mlp_ratio"])
    else:
        transformer_cfg.update(SEFI_SCALE_PRESETS[transformer_scale])

    joint_attention_dim = int(transformer_cfg["joint_attention_dim"])
    if joint_attention_dim != int(text_output_dim):
        raise ValueError(
            "Text dimension mismatch: "
            f"text_encoder output_dim={text_output_dim}, "
            f"transformer joint_attention_dim={joint_attention_dim}."
        )

    transformer_cfg["in_channels"] = int(total_channels)
    transformer_cfg["out_channels"] = int(total_channels)
    transformer_cfg["guidance_embeds"] = False
    return transformer_cfg


def build_lightweight_transformer(config, *, total_channels: int, text_output_dim: int):
    transformer_cfg = build_transformer_config(
        config,
        total_channels=total_channels,
        text_output_dim=text_output_dim,
    )
    return Flux2SEFITransformer2DModel(
        backbone_config=transformer_cfg,
        text_input_dim=int(text_output_dim),
    )


def build_components(config, *, component_dtype: torch.dtype) -> SEFIComponents:
    from diffusers import FlowMatchEulerDiscreteScheduler, Flux2KleinPipeline

    model_cfg = config.model

    texture_vae = build_texture_vae(
        model_cfg.texture_vae,
        torch_dtype=component_dtype,
    )
    texture_codec = TextureLatentCodec(
        texture_vae=texture_vae,
        texture_vae_name=str(model_cfg.texture_vae.name),
    )

    noise_scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(
        str(model_cfg.assets.scheduler_path),
        subfolder="scheduler",
        local_files_only=True,
    )

    semantic_channels = _derive_semantic_channels(config)
    texture_channels = int(texture_codec.texture_channels)
    total_channels = int(semantic_channels + texture_channels)

    text_cfg = model_cfg.text_encoder
    text_encoder = Qwen3VLTextEncoder(
        model_name=str(text_cfg.model_name),
        weights_root=str(text_cfg.get("weights_root", "outputs/model_weights")),
        max_length=int(text_cfg.max_length),
        hidden_layers=[int(x) for x in text_cfg.hidden_layers],
        torch_dtype=component_dtype,
    )

    transformer = build_lightweight_transformer(
        config,
        total_channels=total_channels,
        text_output_dim=int(text_encoder.output_dim),
    )

    return SEFIComponents(
        transformer=transformer,
        text_encoder=text_encoder,
        texture_codec=texture_codec,
        noise_scheduler=noise_scheduler,
        pipeline_cls=Flux2KleinPipeline,
        semantic_channels=semantic_channels,
        texture_channels=texture_channels,
        total_channels=total_channels,
    )
