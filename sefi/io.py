"""Prompt and output helpers for SEFI inference."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

from PIL import Image


@dataclass(frozen=True)
class GenerationItem:
    index: int
    prompt_index: int
    repeat_index: int
    prompt: str

    @property
    def file_stem(self) -> str:
        if self.repeat_index == 0:
            return f"{self.prompt_index:06d}"
        return f"{self.prompt_index:06d}_{self.repeat_index:02d}"


def load_prompts(*, prompt: str | None, prompt_file: str | None) -> list[str]:
    prompts: list[str] = []
    if prompt:
        prompts.append(prompt)
    if prompt_file:
        with open(prompt_file, "r", encoding="utf-8") as handle:
            prompts.extend(line.strip() for line in handle if line.strip())
    if not prompts:
        raise ValueError("Provide --prompt or --prompt-file.")
    return prompts


def expand_prompts(prompts: Iterable[str], num_images_per_prompt: int) -> list[GenerationItem]:
    if num_images_per_prompt <= 0:
        raise ValueError("num_images_per_prompt must be > 0.")

    items: list[GenerationItem] = []
    index = 0
    for prompt_index, prompt in enumerate(prompts):
        for repeat_index in range(num_images_per_prompt):
            items.append(
                GenerationItem(
                    index=index,
                    prompt_index=prompt_index,
                    repeat_index=repeat_index,
                    prompt=prompt,
                )
            )
            index += 1
    return items


def save_images(
    *,
    output_dir: str | Path,
    items: list[GenerationItem],
    images: list[Image.Image],
    rank: int = 0,
) -> None:
    if len(items) != len(images):
        raise ValueError(f"items/images length mismatch: {len(items)} != {len(images)}")

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    metadata_path = out / f"metadata_rank{rank:03d}.jsonl"
    with metadata_path.open("a", encoding="utf-8") as meta:
        for item, image in zip(items, images):
            image_path = out / f"{item.file_stem}.png"
            image.save(image_path)
            row = asdict(item)
            row["image"] = image_path.name
            meta.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_manifest(output_dir: str | Path, payload: dict) -> None:
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    with (out / "inference_manifest.json").open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True)
