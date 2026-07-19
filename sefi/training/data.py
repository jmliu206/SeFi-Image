"""Small, deterministic paired image/text loader for DiT fine-tuning demos."""

from __future__ import annotations

import hashlib
import io
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from PIL import Image
from torch import Tensor
from torch.utils.data import DataLoader, Dataset, DistributedSampler


DEFAULT_RESOLUTION = 1024
DEFAULT_DATASET_CACHE_DIR = "outputs/cache/huggingface/datasets"
DEFAULT_CAPTION_WEIGHTS = {"enhanced_prompt": 4.0, "prompt": 1.0}
CAPTION_MODES = {"weighted", "caption", "fallback"}


def _nonempty_text(value: Any) -> str | None:
    if not isinstance(value, str):
        return None
    value = value.strip()
    return value or None


def _caption_mapping(value: Any, *, plain_key: str) -> dict[str, str]:
    """Normalize a plain, mapping, or JSON caption field."""

    if isinstance(value, Mapping):
        return {
            str(key): text
            for key, candidate in value.items()
            if (text := _nonempty_text(candidate)) is not None
        }
    text = _nonempty_text(value)
    if text is None:
        return {}
    try:
        decoded = json.loads(text)
    except (json.JSONDecodeError, TypeError):
        return {plain_key: text}
    if isinstance(decoded, Mapping):
        return {
            str(key): normalized
            for key, candidate in decoded.items()
            if (normalized := _nonempty_text(candidate)) is not None
        }
    if isinstance(decoded, str) and decoded.strip():
        return {plain_key: decoded.strip()}
    return {plain_key: text}


def collect_caption_fields(row: Mapping[str, Any]) -> dict[str, str]:
    """Collect captions without allowing nested metadata to override columns."""

    fields: dict[str, str] = {}
    for source_key in ("text", "caption", "enhanced_prompt", "prompt"):
        parsed = _caption_mapping(row.get(source_key), plain_key=source_key)
        for key, value in parsed.items():
            fields.setdefault(key, value)
        direct = _nonempty_text(row.get(source_key))
        if direct is not None:
            # A JSON object is metadata, not a literal caption.
            try:
                decoded = json.loads(direct)
            except json.JSONDecodeError:
                decoded = None
            if not isinstance(decoded, Mapping):
                fields[source_key] = decoded.strip() if isinstance(decoded, str) else direct
    return fields


def stable_caption_choice(
    row: Mapping[str, Any],
    *,
    row_id: str,
    seed: int,
    epoch: int,
    mode: str = "weighted",
    weights: Mapping[str, float] | None = None,
) -> tuple[str, str]:
    """Select a caption without consuming process/worker RNG state."""

    mode = str(mode).strip().lower()
    if mode not in CAPTION_MODES:
        raise ValueError(f"Unsupported caption mode {mode!r}; expected {sorted(CAPTION_MODES)}.")
    fields = collect_caption_fields(row)

    if mode == "caption":
        caption = fields.get("caption")
        if caption is None:
            raise ValueError(f"Row {row_id!r} has no non-empty caption field.")
        return caption, "caption"

    if mode == "fallback":
        for key in ("caption", "enhanced_prompt", "prompt", "text"):
            if key in fields:
                return fields[key], key
        raise ValueError(f"Row {row_id!r} has no usable caption in fallback fields.")

    selected_weights = dict(DEFAULT_CAPTION_WEIGHTS if weights is None else weights)
    available: list[tuple[str, str, float]] = []
    for key, raw_weight in selected_weights.items():
        try:
            weight = float(raw_weight)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"Caption weight for {key!r} must be numeric.") from exc
        if weight <= 0:
            raise ValueError(f"Caption weight for {key!r} must be positive, got {weight}.")
        if key in fields:
            available.append((key, fields[key], weight))
    if not available:
        raise ValueError(
            f"Row {row_id!r} has none of the weighted caption fields "
            f"{list(selected_weights)}."
        )
    if len(available) == 1:
        key, caption, _weight = available[0]
        return caption, key

    digest = hashlib.sha256(
        f"sefi-caption-v1\0{int(seed)}\0{int(epoch)}\0{row_id}".encode("utf-8")
    ).digest()
    unit_interval = int.from_bytes(digest[:8], "big") / float(1 << 64)
    total_weight = sum(weight for _key, _caption, weight in available)
    threshold = unit_interval * total_weight
    cumulative = 0.0
    for key, caption, weight in available:
        cumulative += weight
        if threshold < cumulative:
            return caption, key
    key, caption, _weight = available[-1]
    return caption, key


def decode_image(value: Any) -> Image.Image:
    """Decode HF Image, ``{bytes,path}``, raw bytes, or PIL values."""

    if isinstance(value, Image.Image):
        value.load()
        return value.convert("RGB")
    if isinstance(value, Mapping):
        image_bytes = value.get("bytes")
        image_path = value.get("path")
        if image_bytes is not None:
            value = image_bytes
        elif image_path:
            value = Path(str(image_path)).expanduser()
        else:
            raise ValueError("Image mapping must contain non-empty 'bytes' or 'path'.")
    if isinstance(value, (bytes, bytearray, memoryview)):
        with Image.open(io.BytesIO(bytes(value))) as image:
            image.load()
            return image.convert("RGB")
    if isinstance(value, (str, Path)):
        path = Path(value).expanduser()
        if not path.is_file():
            raise FileNotFoundError(f"Image path does not exist: {path}")
        with Image.open(path) as image:
            image.load()
            return image.convert("RGB")
    raise TypeError(f"Unsupported image value type: {type(value).__name__}.")


class FixedSquareImageTransform:
    """Resize shortest side, center crop, and normalize RGB to ``[-1,1]``."""

    def __init__(self, resolution: int = DEFAULT_RESOLUTION):
        self.resolution = int(resolution)
        if self.resolution <= 0:
            raise ValueError(f"resolution must be positive, got {self.resolution}.")

    def __call__(self, image: Image.Image) -> Tensor:
        try:
            from torchvision import transforms
        except ImportError as exc:
            raise RuntimeError("Image preprocessing requires torchvision.") from exc
        transform = transforms.Compose(
            [
                transforms.Resize(
                    self.resolution,
                    interpolation=transforms.InterpolationMode.BICUBIC,
                ),
                transforms.CenterCrop(self.resolution),
                transforms.ToTensor(),
                transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
            ]
        )
        pixel_values = transform(image.convert("RGB"))
        if pixel_values.shape != (3, self.resolution, self.resolution):
            raise ValueError(
                "Unexpected fixed-resolution image shape: "
                f"got={tuple(pixel_values.shape)}, resolution={self.resolution}."
            )
        if not torch.isfinite(pixel_values).all():
            raise ValueError("Transformed image contains NaN or Inf values.")
        return pixel_values


class PairedImageTextDataset(Dataset):
    """Validated map-style paired dataset with epoch-stable caption selection."""

    def __init__(
        self,
        rows: Any,
        *,
        resolution: int = DEFAULT_RESOLUTION,
        caption_mode: str = "weighted",
        caption_weights: Mapping[str, float] | None = None,
        seed: int = 0,
        transform: Any | None = None,
    ):
        if not hasattr(rows, "__len__") or not hasattr(rows, "__getitem__"):
            raise TypeError("rows must be a map-style dataset with __len__ and __getitem__.")
        self.rows = rows
        self.transform = transform or FixedSquareImageTransform(resolution)
        self.resolution = int(resolution)
        self.caption_mode = caption_mode
        self.caption_weights = dict(
            DEFAULT_CAPTION_WEIGHTS if caption_weights is None else caption_weights
        )
        self.seed = int(seed)
        self.epoch = 0

    def __len__(self) -> int:
        return len(self.rows)

    def set_epoch(self, epoch: int) -> None:
        epoch = int(epoch)
        if epoch < 0:
            raise ValueError(f"epoch must be non-negative, got {epoch}.")
        self.epoch = epoch

    def state_dict(self) -> dict[str, int]:
        return {"epoch": self.epoch, "seed": self.seed}

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        saved_seed = int(state.get("seed", self.seed))
        if saved_seed != self.seed:
            raise ValueError(
                f"Dataset seed mismatch: checkpoint={saved_seed}, configured={self.seed}."
            )
        self.set_epoch(int(state.get("epoch", 0)))

    def __getitem__(self, index: int) -> dict[str, Any]:
        row = self.rows[index]
        if not isinstance(row, Mapping):
            raise TypeError(f"Dataset row {index} must be a mapping, got {type(row).__name__}.")
        row_id_value = row.get("id", row.get("source_sample_id"))
        row_id = _nonempty_text(str(row_id_value)) if row_id_value is not None else None
        if row_id is None:
            raise ValueError(f"Dataset row {index} has no stable id/source_sample_id.")
        try:
            image = decode_image(row.get("image"))
            pixel_values = self.transform(image)
        except Exception as exc:
            raise ValueError(f"Failed to decode/transform image for row {row_id!r}: {exc}") from exc
        caption, caption_key = stable_caption_choice(
            row,
            row_id=row_id,
            seed=self.seed,
            epoch=self.epoch,
            mode=self.caption_mode,
            weights=self.caption_weights,
        )
        return {
            "pixel_values": pixel_values,
            "caption": caption,
            "caption_key": caption_key,
            "sample_id": row_id,
            "row_index": int(index),
        }


def collate_paired_batch(samples: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    if not samples:
        raise ValueError("Cannot collate an empty paired batch.")
    pixel_values = torch.stack([sample["pixel_values"] for sample in samples])
    if not torch.isfinite(pixel_values).all():
        raise ValueError("Paired batch contains NaN or Inf image values.")
    return {
        "pixel_values": pixel_values,
        "captions": [str(sample["caption"]) for sample in samples],
        "caption_keys": [str(sample["caption_key"]) for sample in samples],
        "sample_ids": [str(sample["sample_id"]) for sample in samples],
        "row_indices": torch.tensor([int(sample["row_index"]) for sample in samples]),
    }


def _local_parquet_files(source: Path, split: str) -> list[str]:
    if source.is_file():
        if source.suffix.lower() != ".parquet":
            raise ValueError(f"Local dataset file must be Parquet, got: {source}")
        return [str(source)]
    if not source.is_dir():
        raise FileNotFoundError(f"Local dataset path does not exist: {source}")
    patterns = (f"data/{split}-*.parquet", f"{split}-*.parquet")
    for pattern in patterns:
        matches = sorted(source.glob(pattern))
        if matches:
            return [str(path) for path in matches]
    raise FileNotFoundError(f"No Parquet files for split {split!r} under {source}.")


def load_paired_rows(
    source: str | Path | Any,
    *,
    split: str = "train",
    revision: str | None = None,
    cache_dir: str | Path | None = DEFAULT_DATASET_CACHE_DIR,
    dataset_config_name: str | None = None,
) -> Any:
    """Load a local Parquet split or a pinned Hugging Face map-style split."""

    if not isinstance(source, (str, Path)):
        if hasattr(source, "__len__") and hasattr(source, "__getitem__"):
            return source
        raise TypeError("source must be a local path, HF repo id, or map-style dataset.")
    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise RuntimeError("Paired Parquet loading requires the 'datasets' package.") from exc

    candidate = Path(source).expanduser()
    if candidate.exists():
        files = _local_parquet_files(candidate, split)
        local_kwargs: dict[str, Any] = {}
        if cache_dir is not None:
            local_kwargs["cache_dir"] = str(cache_dir)
        return load_dataset(
            "parquet",
            data_files={split: files},
            split=split,
            **local_kwargs,
        )

    kwargs: dict[str, Any] = {"split": split}
    if revision is not None:
        kwargs["revision"] = revision
    if cache_dir is not None:
        kwargs["cache_dir"] = str(cache_dir)
    return load_dataset(str(source), dataset_config_name, **kwargs)


@dataclass
class DataCursor:
    """Checkpointable position of the next batch in a deterministic epoch."""

    epoch: int = 0
    batch_offset: int = 0
    sampler_seed: int = 0

    def __post_init__(self) -> None:
        self.epoch = int(self.epoch)
        self.batch_offset = int(self.batch_offset)
        self.sampler_seed = int(self.sampler_seed)
        if self.epoch < 0 or self.batch_offset < 0:
            raise ValueError("Data cursor epoch and batch_offset must be non-negative.")

    def state_dict(self) -> dict[str, int]:
        return {
            "epoch": self.epoch,
            "batch_offset": self.batch_offset,
            "sampler_seed": self.sampler_seed,
        }

    @classmethod
    def from_state_dict(cls, state: Mapping[str, Any]) -> "DataCursor":
        return cls(
            epoch=int(state.get("epoch", 0)),
            batch_offset=int(state.get("batch_offset", 0)),
            sampler_seed=int(state.get("sampler_seed", 0)),
        )

    def advance(self, *, batches: int = 1) -> None:
        batches = int(batches)
        if batches < 0:
            raise ValueError(f"batches must be non-negative, got {batches}.")
        self.batch_offset += batches

    def next_epoch(self) -> None:
        self.epoch += 1
        self.batch_offset = 0


def apply_data_cursor(dataloader: DataLoader, cursor: DataCursor) -> None:
    """Restore epoch semantics; the caller skips ``batch_offset`` batches."""

    sampler = dataloader.sampler
    configured_seed = getattr(sampler, "seed", cursor.sampler_seed)
    if int(configured_seed) != cursor.sampler_seed:
        raise ValueError(
            "Sampler seed mismatch: "
            f"checkpoint={cursor.sampler_seed}, configured={configured_seed}."
        )
    if hasattr(sampler, "set_epoch"):
        sampler.set_epoch(cursor.epoch)
    dataset = dataloader.dataset
    if hasattr(dataset, "set_epoch"):
        dataset.set_epoch(cursor.epoch)


def build_paired_dataloader(
    source: str | Path | Any,
    *,
    split: str = "train",
    revision: str | None = None,
    cache_dir: str | Path | None = DEFAULT_DATASET_CACHE_DIR,
    dataset_config_name: str | None = None,
    resolution: int = DEFAULT_RESOLUTION,
    batch_size: int = 1,
    num_workers: int = 0,
    caption_mode: str = "weighted",
    caption_weights: Mapping[str, float] | None = None,
    sampler_seed: int = 0,
    epoch: int = 0,
    rank: int = 0,
    world_size: int = 1,
    shuffle: bool = True,
    drop_last: bool = False,
    pin_memory: bool = True,
) -> tuple[DataLoader, DataCursor]:
    """Build the shared LoRA/full deterministic map-style dataloader."""

    batch_size = int(batch_size)
    world_size = int(world_size)
    rank = int(rank)
    if batch_size <= 0:
        raise ValueError(f"batch_size must be positive, got {batch_size}.")
    if world_size <= 0 or not 0 <= rank < world_size:
        raise ValueError(f"Invalid distributed rank/world_size: {rank}/{world_size}.")
    rows = load_paired_rows(
        source,
        split=split,
        revision=revision,
        cache_dir=cache_dir,
        dataset_config_name=dataset_config_name,
    )
    dataset = PairedImageTextDataset(
        rows,
        resolution=resolution,
        caption_mode=caption_mode,
        caption_weights=caption_weights,
        seed=sampler_seed,
    )
    sampler = DistributedSampler(
        dataset,
        num_replicas=world_size,
        rank=rank,
        shuffle=shuffle,
        seed=int(sampler_seed),
        drop_last=drop_last,
    )
    # Creating a DataLoader iterator draws a base seed even when
    # ``num_workers=0``. Keep that bookkeeping off the process-global CPU RNG
    # used by text dropout; a resumed process necessarily creates an extra
    # iterator before its next optimization step.
    dataloader_generator = torch.Generator()
    dataloader_generator.manual_seed(int(sampler_seed))
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=sampler,
        num_workers=int(num_workers),
        collate_fn=collate_paired_batch,
        pin_memory=bool(pin_memory),
        # Fresh workers observe dataset.set_epoch() on every new iterator. A
        # persistent worker would retain a stale copied epoch and break exact
        # caption replay after resume.
        persistent_workers=False,
        drop_last=drop_last,
        generator=dataloader_generator,
    )
    cursor = DataCursor(epoch=epoch, batch_offset=0, sampler_seed=sampler_seed)
    apply_data_cursor(dataloader, cursor)
    return dataloader, cursor
