"""Resolution helpers for SEFI inference."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ImageSize:
    height: int
    width: int


def resolve_image_size(
    *,
    height: int | None,
    width: int | None,
    default_height: int,
    default_width: int,
) -> ImageSize:
    resolved = ImageSize(
        height=int(height if height is not None else default_height),
        width=int(width if width is not None else default_width),
    )
    if resolved.height <= 0 or resolved.width <= 0:
        raise ValueError(f"Image size must be positive, got {resolved}.")
    if resolved.height % 16 != 0 or resolved.width % 16 != 0:
        raise ValueError(
            "SEFI image size must be divisible by 16, "
            f"got height={resolved.height}, width={resolved.width}."
        )
    return resolved
