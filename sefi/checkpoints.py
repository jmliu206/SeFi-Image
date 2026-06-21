"""Checkpoint staging helpers for SEFI inference."""

from __future__ import annotations

import os
from pathlib import Path


CONFIG_FILENAMES = ("sefi_config.yaml", "config.yaml")


def _download_hf_snapshot(
    repo_id: str,
    *,
    cache_dir: str | os.PathLike[str],
) -> str:
    try:
        from huggingface_hub import snapshot_download
    except ImportError as exc:
        raise RuntimeError(
            "Checkpoint is not a local path. Install huggingface_hub or pass a "
            "local --checkpoint path."
        ) from exc

    return snapshot_download(
        repo_id=repo_id,
        cache_dir=str(cache_dir),
        local_files_only=False,
    )


def checkpoint_root(path: str | os.PathLike[str]) -> Path:
    resolved = Path(path).expanduser()
    return resolved if resolved.is_dir() else resolved.parent


def resolve_config_path(
    checkpoint_path: str | os.PathLike[str],
    config_path: str | os.PathLike[str] | None = None,
) -> str:
    root = checkpoint_root(checkpoint_path)

    if config_path:
        candidate = Path(config_path).expanduser()
        if not candidate.is_absolute():
            rooted = root / candidate
            if rooted.is_file():
                return str(rooted)
        if candidate.is_file():
            return str(candidate)
        raise FileNotFoundError(f"Config file not found: {config_path}")

    for filename in CONFIG_FILENAMES:
        candidate = root / filename
        if candidate.is_file():
            return str(candidate)

    expected = ", ".join(CONFIG_FILENAMES)
    raise FileNotFoundError(
        f"SEFI config not found under checkpoint root {root}. "
        f"Expected one of: {expected}. Use --config to override."
    )


def ensure_local_path(
    checkpoint: str,
    *,
    cache_dir: str | os.PathLike[str],
) -> str:
    if not checkpoint:
        raise ValueError(
            "No checkpoint was provided. Pass a local path or Hugging Face repo id "
            "with --checkpoint."
        )

    path = Path(checkpoint).expanduser()
    if path.exists():
        return str(path)

    return _download_hf_snapshot(
        checkpoint,
        cache_dir=cache_dir,
    )


def resolve_checkpoint_to_local(
    *,
    checkpoint: str,
    cache_dir: str | os.PathLike[str],
) -> tuple[str, str]:
    local_path = ensure_local_path(
        checkpoint,
        cache_dir=cache_dir,
    )
    return local_path, checkpoint
