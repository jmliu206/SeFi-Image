"""Training-state and portable transformer checkpoint helpers.

``Accelerator.save_state`` remains the source of truth for resumable optimizer,
scheduler, RNG and (for full tuning) DeepSpeed ZeRO-2 state.  The helpers here
add the dataloader position that Accelerate cannot infer from ``global_step``
and provide a separate, name-keyed safetensors export for inference.
"""

from __future__ import annotations

import hashlib
import json
import os
from collections import OrderedDict
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch
import torch.nn as nn


TRAINER_STATE_FILENAME = "trainer_state.json"
EXPORT_MANIFEST_FILENAME = "sefi_export_manifest.json"
TRANSFORMER_WEIGHT_PREFIX = "diffusion_pytorch_model"


@dataclass(frozen=True)
class TrainerState:
    """Minimal loop state needed for exact dataloader resume.

    ``batch_offset`` is the number of batches already consumed in
    ``data_epoch``.  ``global_step`` counts only synchronized optimizer steps.
    """

    global_step: int = 0
    data_epoch: int = 0
    batch_offset: int = 0
    schema_version: int = 1

    def __post_init__(self) -> None:
        for name in ("global_step", "data_epoch", "batch_offset"):
            value = int(getattr(self, name))
            if value < 0:
                raise ValueError(f"TrainerState.{name} must be non-negative; got {value}.")
            object.__setattr__(self, name, value)
        if int(self.schema_version) <= 0:
            raise ValueError("TrainerState.schema_version must be positive.")

    def to_dict(self) -> dict[str, int]:
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "TrainerState":
        if not isinstance(payload, Mapping):
            raise ValueError("trainer_state.json must contain a JSON object.")
        # The reference trainer wrote only ``iteration``.  Accept it so an old
        # checkpoint remains loadable, while making the missing data position
        # explicit through zero-valued defaults.
        global_step = payload.get("global_step", payload.get("iteration", 0))
        return cls(
            global_step=int(global_step),
            data_epoch=int(payload.get("data_epoch", 0)),
            batch_offset=int(payload.get("batch_offset", 0)),
            schema_version=int(payload.get("schema_version", 1)),
        )

    def with_global_step(self, global_step: int) -> "TrainerState":
        return replace(self, global_step=int(global_step))

    def after_batch(self, batches_per_epoch: int) -> "TrainerState":
        """Return the next data position after consuming one batch."""

        batches_per_epoch = int(batches_per_epoch)
        if batches_per_epoch <= 0:
            raise ValueError(
                f"batches_per_epoch must be positive; got {batches_per_epoch}."
            )
        next_offset = self.batch_offset + 1
        if next_offset < batches_per_epoch:
            return replace(self, batch_offset=next_offset)
        if next_offset == batches_per_epoch:
            return replace(self, data_epoch=self.data_epoch + 1, batch_offset=0)
        raise ValueError(
            f"Current batch_offset={self.batch_offset} is invalid for "
            f"batches_per_epoch={batches_per_epoch}."
        )

    def validate_for_dataloader(self, batches_per_epoch: int) -> None:
        batches_per_epoch = int(batches_per_epoch)
        if batches_per_epoch <= 0:
            raise ValueError("Cannot resume from an empty dataloader.")
        if self.batch_offset >= batches_per_epoch:
            raise ValueError(
                f"batch_offset={self.batch_offset} must be smaller than "
                f"batches_per_epoch={batches_per_epoch}."
            )


def _is_main_process(accelerator) -> bool:
    return accelerator is None or bool(accelerator.is_main_process)


def _wait_for_everyone(accelerator) -> None:
    if accelerator is not None:
        accelerator.wait_for_everyone()


def write_trainer_state(
    checkpoint_dir: str | os.PathLike[str],
    state: TrainerState,
    *,
    accelerator=None,
) -> Path:
    """Atomically persist loop state on the main process."""

    output = Path(checkpoint_dir)
    target = output / TRAINER_STATE_FILENAME
    if _is_main_process(accelerator):
        output.mkdir(parents=True, exist_ok=True)
        temporary = output / f".{TRAINER_STATE_FILENAME}.tmp"
        with temporary.open("w", encoding="utf-8") as handle:
            json.dump(state.to_dict(), handle, indent=2, sort_keys=True)
            handle.write("\n")
        os.replace(temporary, target)
    _wait_for_everyone(accelerator)
    return target


def read_trainer_state(
    checkpoint_dir: str | os.PathLike[str],
    *,
    allow_missing: bool = False,
) -> TrainerState:
    path = Path(checkpoint_dir) / TRAINER_STATE_FILENAME
    if not path.is_file():
        if allow_missing:
            return TrainerState()
        raise FileNotFoundError(f"Missing training loop state: {path}")
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    return TrainerState.from_dict(payload)


def save_training_checkpoint(
    accelerator,
    checkpoint_dir: str | os.PathLike[str],
    state: TrainerState,
) -> Path:
    """Save complete Accelerate/DeepSpeed resume state plus loop position."""

    output = Path(checkpoint_dir)
    if accelerator.is_main_process:
        output.mkdir(parents=True, exist_ok=True)
    accelerator.wait_for_everyone()
    accelerator.save_state(str(output))
    write_trainer_state(output, state, accelerator=accelerator)
    return output


def load_training_checkpoint(
    accelerator,
    checkpoint_dir: str | os.PathLike[str],
) -> TrainerState:
    """Restore complete Accelerate/DeepSpeed state and return loop position."""

    checkpoint = Path(checkpoint_dir)
    if not checkpoint.is_dir():
        raise FileNotFoundError(f"Training checkpoint directory not found: {checkpoint}")
    accelerator.load_state(str(checkpoint))
    state = read_trainer_state(checkpoint)
    accelerator.wait_for_everyone()
    return state


def _find_sampler(dataloader):
    sampler = getattr(dataloader, "sampler", None)
    if sampler is not None and hasattr(sampler, "set_epoch"):
        return sampler
    batch_sampler = getattr(dataloader, "batch_sampler", None)
    sampler = getattr(batch_sampler, "sampler", None)
    if sampler is not None and hasattr(sampler, "set_epoch"):
        return sampler
    return None


def dataloader_for_resume(accelerator, dataloader, state: TrainerState):
    """Set the distributed epoch and skip batches already consumed in it."""

    state.validate_for_dataloader(len(dataloader))
    if bool(getattr(dataloader, "use_stateful_dataloader", False)):
        # accelerator.load_state() has already restored torchdata's iterator
        # position. Applying batch_offset again would double-skip samples.
        return dataloader
    # Accelerate's prepared DataLoaderShard exposes set_epoch itself; regular
    # PyTorch dataloaders instead expose it through DistributedSampler.
    if hasattr(dataloader, "set_epoch"):
        dataloader.set_epoch(state.data_epoch)
    else:
        sampler = _find_sampler(dataloader)
        if sampler is not None:
            sampler.set_epoch(state.data_epoch)
    if state.batch_offset == 0:
        return dataloader
    return accelerator.skip_first_batches(dataloader, state.batch_offset)


def _strip_uniform_prefix(
    state_dict: Mapping[str, torch.Tensor], prefix: str
) -> OrderedDict[str, torch.Tensor]:
    if state_dict and all(str(name).startswith(prefix) for name in state_dict):
        return OrderedDict(
            (str(name)[len(prefix) :], tensor) for name, tensor in state_dict.items()
        )
    return OrderedDict((str(name), tensor) for name, tensor in state_dict.items())


def normalize_state_dict_names(
    state_dict: Mapping[str, torch.Tensor],
) -> OrderedDict[str, torch.Tensor]:
    """Remove only uniform runtime wrappers; never do fuzzy key matching."""

    normalized = _strip_uniform_prefix(state_dict, "module.")
    normalized = _strip_uniform_prefix(normalized, "_orig_mod.")
    if len(normalized) != len(state_dict):
        raise ValueError("Duplicate names appeared while normalizing state_dict keys.")
    for name, tensor in normalized.items():
        if not name:
            raise ValueError("Portable state_dict contains an empty parameter name.")
        if not torch.is_tensor(tensor):
            raise TypeError(
                f"Portable state_dict value for {name!r} is not a Tensor: "
                f"{type(tensor).__name__}."
            )
    return normalized


def overlay_ema_parameters_by_name(
    model_state_dict: Mapping[str, torch.Tensor],
    *,
    ema_param_names: Sequence[str],
    shadow_params: Sequence[torch.Tensor],
) -> OrderedDict[str, torch.Tensor]:
    """Replace model parameters with EMA shadows using exact stable names.

    Buffers remain sourced from ``model_state_dict``.  Position-only matching is
    intentionally rejected: names, counts and shapes all have to agree.
    """

    state = normalize_state_dict_names(model_state_dict)
    names = [str(name) for name in ema_param_names]
    shadows = list(shadow_params)
    if len(names) != len(shadows):
        raise ValueError(
            f"EMA name/shadow count mismatch: names={len(names)}, "
            f"shadows={len(shadows)}."
        )
    if len(set(names)) != len(names):
        raise ValueError("EMA parameter names contain duplicates.")

    missing = [name for name in names if name not in state]
    if missing:
        raise ValueError(f"EMA parameter names missing from model state: {missing[:20]}")

    output = OrderedDict(state)
    for name, shadow in zip(names, shadows, strict=True):
        if not torch.is_tensor(shadow):
            raise TypeError(f"EMA shadow for {name!r} is not a Tensor.")
        if tuple(shadow.shape) != tuple(state[name].shape):
            raise ValueError(
                f"EMA shape mismatch for {name}: shadow={tuple(shadow.shape)}, "
                f"model={tuple(state[name].shape)}."
            )
        if not shadow.is_floating_point():
            raise TypeError(f"EMA shadow for {name!r} must be floating point.")
        output[name] = shadow.detach()
    return output


def collect_portable_state_dict(
    accelerator,
    model: nn.Module,
    *,
    ema_model=None,
    ema_param_names: Sequence[str] | None = None,
) -> OrderedDict[str, torch.Tensor]:
    """Collect a name-keyed full state dict from DDP or DeepSpeed ZeRO-2."""

    # All ranks must call get_state_dict; this remains safe if the implementation
    # later moves from ZeRO-2 to a mode that requires collective consolidation.
    state = accelerator.get_state_dict(model)
    state = normalize_state_dict_names(state)
    if ema_model is None:
        if ema_param_names:
            raise ValueError("ema_param_names was provided without ema_model.")
        return state
    if ema_param_names is None:
        raise ValueError("EMA export requires stable ema_param_names.")
    shadow_params = getattr(ema_model, "shadow_params", None)
    if shadow_params is None:
        raise TypeError("ema_model must expose a shadow_params sequence.")
    return overlay_ema_parameters_by_name(
        state,
        ema_param_names=ema_param_names,
        shadow_params=shadow_params,
    )


def _resolve_export_dtype(dtype: torch.dtype | str | None) -> torch.dtype | None:
    if dtype is None:
        return None
    if isinstance(dtype, torch.dtype):
        return dtype
    aliases = {
        "bf16": torch.bfloat16,
        "bfloat16": torch.bfloat16,
        "fp16": torch.float16,
        "float16": torch.float16,
        "fp32": torch.float32,
        "float32": torch.float32,
    }
    try:
        return aliases[str(dtype).strip().lower()]
    except KeyError as exc:
        raise ValueError(f"Unsupported portable export dtype: {dtype!r}.") from exc


def _tensor_export_dtype(tensor: torch.Tensor, dtype: torch.dtype | None) -> torch.dtype:
    if dtype is not None and tensor.is_floating_point():
        return dtype
    return tensor.dtype


def _export_total_size(
    state_dict: Mapping[str, torch.Tensor], dtype: torch.dtype | None
) -> int:
    total = 0
    for tensor in state_dict.values():
        export_dtype = _tensor_export_dtype(tensor, dtype)
        total += tensor.numel() * torch.empty((), dtype=export_dtype).element_size()
    return int(total)


@dataclass(frozen=True)
class ShardedSafetensorsExport:
    output_dir: Path
    weight_files: tuple[str, ...]
    index_file: str | None
    total_size: int
    tensor_count: int


def export_sharded_safetensors(
    state_dict: Mapping[str, torch.Tensor],
    output_dir: str | os.PathLike[str],
    *,
    filename_prefix: str = TRANSFORMER_WEIGHT_PREFIX,
    max_shard_size: int | str = "5GB",
    dtype: torch.dtype | str | None = torch.bfloat16,
) -> ShardedSafetensorsExport:
    """Write a deterministic, HF-style sharded safetensors state dict.

    Conversion is performed one shard at a time, avoiding a second full CPU
    copy of a multi-billion-parameter EMA state.
    """

    try:
        from huggingface_hub import split_torch_state_dict_into_shards
        from safetensors.torch import save_file
    except ImportError as exc:  # pragma: no cover
        raise ImportError(
            "Portable export requires huggingface_hub and safetensors."
        ) from exc

    if not filename_prefix or "/" in filename_prefix or "\\" in filename_prefix:
        raise ValueError("filename_prefix must be a plain filename stem.")
    normalized = normalize_state_dict_names(state_dict)
    if not normalized:
        raise ValueError("Cannot export an empty state_dict.")
    normalized = OrderedDict(sorted(normalized.items()))
    for name, tensor in normalized.items():
        if tensor.layout != torch.strided:
            raise TypeError(f"Safetensors export requires dense tensor {name!r}.")

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    index_path = output / f"{filename_prefix}.safetensors.index.json"
    existing = list(output.glob(f"{filename_prefix}*.safetensors"))
    if index_path.exists() or existing:
        raise FileExistsError(
            f"Refusing to overwrite existing portable weights in {output}."
        )

    filename_pattern = f"{filename_prefix}{{suffix}}.safetensors"
    split = split_torch_state_dict_into_shards(
        normalized,
        filename_pattern=filename_pattern,
        max_shard_size=max_shard_size,
    )
    export_dtype = _resolve_export_dtype(dtype)
    weight_files: list[str] = []
    for filename, tensor_names in split.filename_to_tensors.items():
        shard: dict[str, torch.Tensor] = {}
        for name in tensor_names:
            tensor = normalized[name].detach()
            target_dtype = _tensor_export_dtype(tensor, export_dtype)
            # clone() prevents shared-storage aliases from violating safetensors'
            # format contract while bounding extra host memory to one shard.
            shard[name] = tensor.to(device="cpu", dtype=target_dtype).contiguous().clone()
        target = output / filename
        temporary = output / f".{filename}.tmp"
        save_file(shard, str(temporary), metadata={"format": "pt"})
        os.replace(temporary, target)
        weight_files.append(filename)

    total_size = _export_total_size(normalized, export_dtype)
    index_filename: str | None = None
    if split.is_sharded:
        index = {
            "metadata": {"total_size": total_size},
            "weight_map": dict(sorted(split.tensor_to_filename.items())),
        }
        temporary_index = output / f".{index_path.name}.tmp"
        with temporary_index.open("w", encoding="utf-8") as handle:
            json.dump(index, handle, indent=2, sort_keys=True)
            handle.write("\n")
        os.replace(temporary_index, index_path)
        index_filename = index_path.name

    return ShardedSafetensorsExport(
        output_dir=output,
        weight_files=tuple(sorted(weight_files)),
        index_file=index_filename,
        total_size=total_size,
        tensor_count=len(normalized),
    )


def _state_name_hash(state_dict: Mapping[str, torch.Tensor]) -> str:
    payload = "\n".join(sorted(state_dict)).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def export_full_transformer_checkpoint(
    accelerator,
    model: nn.Module,
    output_dir: str | os.PathLike[str],
    *,
    ema_model=None,
    ema_param_names: Sequence[str] | None = None,
    max_shard_size: int | str = "5GB",
    dtype: torch.dtype | str | None = torch.bfloat16,
    transformer_config: Mapping[str, Any] | None = None,
    metadata: Mapping[str, Any] | None = None,
) -> ShardedSafetensorsExport | None:
    """Export model or GPU EMA weights from a full ZeRO-2 training run.

    Every rank participates in state collection.  Only the main process writes
    ``transformer/diffusion_pytorch_model*.safetensors``-compatible files.
    """

    state = collect_portable_state_dict(
        accelerator,
        model,
        ema_model=ema_model,
        ema_param_names=ema_param_names,
    )
    if not accelerator.is_main_process:
        return None

    transformer_dir = Path(output_dir) / "transformer"
    result = export_sharded_safetensors(
        state,
        transformer_dir,
        filename_prefix=TRANSFORMER_WEIGHT_PREFIX,
        max_shard_size=max_shard_size,
        dtype=dtype,
    )
    if transformer_config is not None:
        config_path = transformer_dir / "config.json"
        with config_path.open("w", encoding="utf-8") as handle:
            json.dump(dict(transformer_config), handle, indent=2, sort_keys=True, default=str)
            handle.write("\n")

    manifest = {
        "schema_version": 1,
        "source": "ema" if ema_model is not None else "model",
        "dtype": str(_resolve_export_dtype(dtype)).replace("torch.", "") if dtype is not None else "source",
        "tensor_count": result.tensor_count,
        "total_size": result.total_size,
        "weight_files": list(result.weight_files),
        "index_file": result.index_file,
        "state_name_sha256": _state_name_hash(state),
    }
    if metadata:
        manifest["metadata"] = dict(metadata)
    manifest_path = Path(output_dir) / EXPORT_MANIFEST_FILENAME
    with manifest_path.open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2, sort_keys=True, default=str)
        handle.write("\n")
    return result
