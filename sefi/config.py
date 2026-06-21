"""Config loading helpers for SEFI inference."""

from __future__ import annotations

from pathlib import Path

from omegaconf import OmegaConf


def _resolve_relative_path(value, base_dir: Path) -> str:
    if value is None:
        return value
    raw = str(value).strip()
    if not raw:
        return raw
    path = Path(raw).expanduser()
    if path.is_absolute():
        return str(path)
    return str(base_dir / path)


def _patch_path(config, dotted_path: str, base_dir: Path) -> None:
    parts = dotted_path.split(".")
    node = config
    for part in parts[:-1]:
        if part not in node:
            return
        node = node[part]
    leaf = parts[-1]
    if leaf in node:
        node[leaf] = _resolve_relative_path(node[leaf], base_dir)


def _resolve_artifact_paths(config, base_dir: Path):
    for dotted_path in (
        "model.assets.transformer_config_path",
        "model.assets.scheduler_path",
        "model.texture_vae.base_path",
        "model.text_encoder.weights_root",
    ):
        _patch_path(config, dotted_path, base_dir)
    return config


def load_config(config_path: str | Path):
    path = Path(config_path).expanduser()
    if not path.is_file():
        raise FileNotFoundError(f"Config file not found: {path}")
    config = OmegaConf.load(path)
    config = _resolve_artifact_paths(config, path.parent)
    OmegaConf.resolve(config)
    return config
