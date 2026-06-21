"""Runtime helpers for the bundled SEFI inference package."""

from __future__ import annotations

from typing import Any

from .config import load_config


def load_runtime_symbols() -> tuple[Any, Any]:
    from .runner import SEFIInferenceRunner

    return SEFIInferenceRunner, load_config
