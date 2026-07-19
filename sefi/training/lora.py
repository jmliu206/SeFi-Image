"""PEFT LoRA helpers for SEFI DiT fine-tuning.

The target discovery in this module deliberately returns *full* module names.
PEFT accepts suffixes, but short suffixes such as ``to_out`` are ambiguous in
Flux2 (and can also select a ``ModuleList`` instead of a ``Linear``).  Keeping
the resolved names in the adapter metadata also makes the training artifact
auditable against the exact backbone that produced it.
"""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch.nn as nn


DOUBLE_STREAM_TARGETS = (
    "to_q",
    "to_k",
    "to_v",
    "to_out.0",
    "add_q_proj",
    "add_k_proj",
    "add_v_proj",
    "to_add_out",
)
SINGLE_STREAM_TARGETS = (
    "to_qkv_mlp_proj",
    "to_out",
)
EXPECTED_ATTENTION_COMPLETE_TARGET_COUNTS = {
    "1b": 56,
    "2b": 64,
    "5b": 90,
}
DEFAULT_LORA_RANK = 16
DEFAULT_LORA_ALPHA = 16
DEFAULT_LORA_DROPOUT = 0.0
ADAPTER_METADATA_FILENAME = "sefi_adapter_config.json"
ADAPTER_SUBDIRECTORY = "adapter"


def _normalize_scale(scale: str) -> str:
    normalized = str(scale).strip().lower().replace("-", "").replace("_", "")
    if normalized not in EXPECTED_ATTENTION_COMPLETE_TARGET_COUNTS:
        raise ValueError(
            "Attention-complete LoRA supports scale 1B, 2B, or 5B; "
            f"got {scale!r}."
        )
    return normalized


def _split_attention_target(name: str) -> tuple[str, int, str] | None:
    """Return ``(stream, block_index, target)`` for a Flux2 attention Linear."""

    parts = name.split(".")
    for stream, container, targets in (
        ("double", "transformer_blocks", DOUBLE_STREAM_TARGETS),
        ("single", "single_transformer_blocks", SINGLE_STREAM_TARGETS),
    ):
        try:
            container_index = parts.index(container)
        except ValueError:
            continue
        tail = parts[container_index:]
        if len(tail) < 4 or tail[2] != "attn":
            continue
        try:
            block_index = int(tail[1])
        except ValueError:
            continue
        target = ".".join(tail[3:])
        if target in targets:
            return stream, block_index, target
    return None


def discover_attention_complete_targets(
    model: nn.Module,
    *,
    scale: str | None = None,
    expected_count: int | None = None,
) -> list[str]:
    """Resolve and validate all attention-bearing Flux2 Linear module names.

    Validation happens both per block and by total count.  The per-block check
    catches architecture drift that a count-only assertion could miss.  When a
    release scale is supplied, its known 1B/2B/5B count is always enforced.
    """

    if scale is not None:
        scale_count = EXPECTED_ATTENTION_COMPLETE_TARGET_COUNTS[_normalize_scale(scale)]
        if expected_count is not None and int(expected_count) != scale_count:
            raise ValueError(
                f"Conflicting target counts: scale={scale!r} requires {scale_count}, "
                f"but expected_count={expected_count}."
            )
        expected_count = scale_count
    if expected_count is not None and int(expected_count) <= 0:
        raise ValueError(f"expected_count must be positive; got {expected_count}.")

    targets: list[str] = []
    by_block: dict[tuple[str, int], set[str]] = {}
    for name, module in model.named_modules():
        if not isinstance(module, nn.Linear):
            continue
        parsed = _split_attention_target(name)
        if parsed is None:
            continue
        stream, block_index, target = parsed
        targets.append(name)
        by_block.setdefault((stream, block_index), set()).add(target)

    if not targets:
        raise ValueError(
            "No Flux2 attention-complete LoRA targets were found. Expected Linear "
            "modules below transformer_blocks.*.attn and "
            "single_transformer_blocks.*.attn."
        )

    missing_by_block: list[str] = []
    for (stream, block_index), found in sorted(by_block.items()):
        required = set(
            DOUBLE_STREAM_TARGETS if stream == "double" else SINGLE_STREAM_TARGETS
        )
        missing = sorted(required - found)
        extra = sorted(found - required)
        if missing or extra:
            missing_by_block.append(
                f"{stream}[{block_index}] missing={missing} extra={extra}"
            )
    if missing_by_block:
        raise ValueError(
            "Incomplete attention-complete LoRA target set: "
            + "; ".join(missing_by_block)
        )

    streams = {stream for stream, _ in by_block}
    if streams != {"double", "single"}:
        raise ValueError(
            "Attention-complete LoRA requires both Flux2 streams; "
            f"found {sorted(streams)}."
        )

    targets.sort()
    if expected_count is not None and len(targets) != int(expected_count):
        block_counts = {
            stream: len({index for (kind, index) in by_block if kind == stream})
            for stream in ("double", "single")
        }
        raise ValueError(
            "Unexpected attention-complete LoRA target count: "
            f"found {len(targets)}, expected {int(expected_count)}; "
            f"blocks={block_counts}. This usually means the checkpoint scale or "
            "Flux2 architecture does not match the training config."
        )
    return targets


def create_lora_model(
    model: nn.Module,
    *,
    scale: str,
    rank: int = DEFAULT_LORA_RANK,
    alpha: int = DEFAULT_LORA_ALPHA,
    dropout: float = DEFAULT_LORA_DROPOUT,
    adapter_name: str = "default",
):
    """Inject an attention-complete PEFT adapter and return model and targets."""

    if int(rank) <= 0:
        raise ValueError(f"LoRA rank must be positive; got {rank}.")
    if int(alpha) <= 0:
        raise ValueError(f"LoRA alpha must be positive; got {alpha}.")
    if not 0.0 <= float(dropout) < 1.0:
        raise ValueError(f"LoRA dropout must be in [0, 1); got {dropout}.")

    try:
        from peft import LoraConfig, get_peft_model
    except ImportError as exc:  # pragma: no cover - dependency error is actionable
        raise ImportError(
            "LoRA fine-tuning requires PEFT. Install the training dependencies "
            "before using tuning.mode=lora."
        ) from exc

    targets = discover_attention_complete_targets(model, scale=scale)
    config = LoraConfig(
        r=int(rank),
        lora_alpha=int(alpha),
        lora_dropout=float(dropout),
        target_modules=targets,
        bias="none",
        inference_mode=False,
    )
    peft_model = get_peft_model(model, config, adapter_name=adapter_name)
    injected = set(getattr(peft_model, "targeted_module_names", ()))
    if injected != set(targets):
        missing = sorted(set(targets) - injected)
        unexpected = sorted(injected - set(targets))
        raise RuntimeError(
            "PEFT did not inject the exact resolved attention-complete target set: "
            f"missing={missing[:20]}, unexpected={unexpected[:20]}."
        )
    return peft_model, targets


def count_trainable_parameters(model: nn.Module) -> tuple[int, int]:
    """Return ``(trainable, total)`` parameter counts."""

    total = sum(parameter.numel() for parameter in model.parameters())
    trainable = sum(
        parameter.numel() for parameter in model.parameters() if parameter.requires_grad
    )
    return trainable, total


def canonical_config_hash(config: Any) -> str:
    """Hash a JSON-compatible config using a stable canonical representation."""

    if hasattr(config, "to_container"):
        config = config.to_container(resolve=True)
    elif hasattr(config, "to_dict"):
        config = config.to_dict()
    payload = json.dumps(
        config,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        default=str,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


@dataclass(frozen=True)
class LoraAdapterMetadata:
    """SEFI-specific provenance missing from PEFT's generic adapter config."""

    base_model_repo: str
    scale: str
    target_modules: tuple[str, ...]
    rank: int = DEFAULT_LORA_RANK
    alpha: int = DEFAULT_LORA_ALPHA
    dropout: float = DEFAULT_LORA_DROPOUT
    resolution: int = 1024
    requested_base_revision: str | None = None
    resolved_base_revision: str | None = None
    config_hash: str | None = None
    peft_version: str | None = None
    diffusers_version: str | None = None
    schema_version: int = 1

    def __post_init__(self) -> None:
        normalized_scale = _normalize_scale(self.scale)
        object.__setattr__(self, "scale", normalized_scale)
        object.__setattr__(self, "target_modules", tuple(self.target_modules))
        expected = EXPECTED_ATTENTION_COMPLETE_TARGET_COUNTS[normalized_scale]
        if len(self.target_modules) != expected:
            raise ValueError(
                f"Adapter metadata for {normalized_scale} requires {expected} targets; "
                f"got {len(self.target_modules)}."
            )
        if len(set(self.target_modules)) != len(self.target_modules):
            raise ValueError("Adapter metadata target_modules contains duplicates.")
        if int(self.rank) <= 0 or int(self.alpha) <= 0:
            raise ValueError("Adapter metadata rank and alpha must be positive.")
        if int(self.resolution) <= 0:
            raise ValueError("Adapter metadata resolution must be positive.")

    @property
    def target_count(self) -> int:
        return len(self.target_modules)

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["target_modules"] = list(self.target_modules)
        payload["target_count"] = self.target_count
        return payload


def build_lora_metadata(
    *,
    base_model_repo: str,
    scale: str,
    target_modules: Sequence[str],
    rank: int = DEFAULT_LORA_RANK,
    alpha: int = DEFAULT_LORA_ALPHA,
    dropout: float = DEFAULT_LORA_DROPOUT,
    resolution: int = 1024,
    requested_base_revision: str | None = None,
    resolved_base_revision: str | None = None,
    training_config: Any | None = None,
) -> LoraAdapterMetadata:
    """Build complete adapter provenance, including installed library versions."""

    try:
        import peft

        peft_version = peft.__version__
    except ImportError:  # pragma: no cover
        peft_version = None
    try:
        import diffusers

        diffusers_version = diffusers.__version__
    except ImportError:  # pragma: no cover
        diffusers_version = None

    return LoraAdapterMetadata(
        base_model_repo=str(base_model_repo),
        scale=str(scale),
        target_modules=tuple(sorted(str(name) for name in target_modules)),
        rank=int(rank),
        alpha=int(alpha),
        dropout=float(dropout),
        resolution=int(resolution),
        requested_base_revision=requested_base_revision,
        resolved_base_revision=resolved_base_revision,
        config_hash=(
            canonical_config_hash(training_config)
            if training_config is not None
            else None
        ),
        peft_version=peft_version,
        diffusers_version=diffusers_version,
    )


def write_lora_metadata(
    output_dir: str | os.PathLike[str], metadata: LoraAdapterMetadata | Mapping[str, Any]
) -> Path:
    """Write the SEFI adapter manifest atomically."""

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    payload = metadata.to_dict() if isinstance(metadata, LoraAdapterMetadata) else dict(metadata)
    target = output / ADAPTER_METADATA_FILENAME
    temporary = output / f".{ADAPTER_METADATA_FILENAME}.tmp"
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True, ensure_ascii=False)
        handle.write("\n")
    os.replace(temporary, target)
    return target


def read_lora_metadata(adapter_dir: str | os.PathLike[str]) -> dict[str, Any]:
    path = Path(adapter_dir) / ADAPTER_METADATA_FILENAME
    if not path.is_file():
        raise FileNotFoundError(f"Missing SEFI LoRA metadata: {path}")
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"Invalid SEFI LoRA metadata in {path}: expected an object.")
    return payload


def _require_peft_model(model: nn.Module):
    try:
        from peft import PeftModel
    except ImportError as exc:  # pragma: no cover
        raise ImportError("Loading or saving a LoRA adapter requires PEFT.") from exc
    if not isinstance(model, PeftModel):
        raise TypeError(f"Expected a peft.PeftModel, got {type(model).__name__}.")
    return model


def save_lora_adapter(
    model: nn.Module,
    output_dir: str | os.PathLike[str],
    *,
    metadata: LoraAdapterMetadata | Mapping[str, Any],
    adapter_name: str = "default",
) -> Path:
    """Save a clean PEFT adapter-only artifact plus SEFI provenance."""

    peft_model = _require_peft_model(model)
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    peft_model.save_pretrained(
        output,
        selected_adapters=[adapter_name],
        safe_serialization=True,
    )
    write_lora_metadata(output, metadata)
    return output


def load_lora_adapter_weights(
    model: nn.Module,
    adapter_dir: str | os.PathLike[str],
    *,
    adapter_name: str = "default",
) -> None:
    """Load adapter weights into an already-injected PEFT model in place."""

    peft_model = _require_peft_model(model)
    try:
        from peft import set_peft_model_state_dict
        from peft.utils.save_and_load import load_peft_weights
    except ImportError as exc:  # pragma: no cover
        raise ImportError("Loading a LoRA adapter requires PEFT.") from exc

    adapter_path = Path(adapter_dir)
    if not (adapter_path / "adapter_config.json").is_file():
        raise FileNotFoundError(f"Missing PEFT adapter config: {adapter_path}")
    state_dict = load_peft_weights(str(adapter_path), device="cpu")
    result = set_peft_model_state_dict(
        peft_model,
        state_dict,
        adapter_name=adapter_name,
    )
    unexpected = list(getattr(result, "unexpected_keys", ()) or ())
    if unexpected:
        raise ValueError(f"Unexpected keys while loading LoRA adapter: {unexpected[:20]}")


def load_lora_adapter(
    base_model: nn.Module,
    adapter_dir: str | os.PathLike[str],
    *,
    adapter_name: str = "default",
    is_trainable: bool = False,
):
    """Attach a saved PEFT adapter to a fully initialized Base transformer."""

    try:
        from peft import PeftModel
    except ImportError as exc:  # pragma: no cover
        raise ImportError("Loading a LoRA adapter requires PEFT.") from exc
    return PeftModel.from_pretrained(
        base_model,
        str(adapter_dir),
        adapter_name=adapter_name,
        is_trainable=bool(is_trainable),
    )


def _find_single_peft_model(models: Sequence[nn.Module], accelerator) -> tuple[int, nn.Module]:
    matches: list[tuple[int, nn.Module]] = []
    for index, candidate in enumerate(models):
        unwrapped = accelerator.unwrap_model(candidate)
        try:
            _require_peft_model(unwrapped)
        except TypeError:
            continue
        matches.append((index, unwrapped))
    if len(matches) != 1:
        raise RuntimeError(
            "LoRA checkpoint hooks require exactly one PEFT model in Accelerator; "
            f"found {len(matches)}."
        )
    return matches[0]


def register_lora_accelerate_hooks(
    accelerator,
    *,
    metadata: LoraAdapterMetadata | Mapping[str, Any],
    adapter_name: str = "default",
    adapter_subdirectory: str = ADAPTER_SUBDIRECTORY,
):
    """Register adapter-only Accelerate save/load hooks.

    Accelerate still owns optimizer, scheduler, dataloader, scaler and RNG state.
    The hooks replace only its normal full-model payload with ``adapter/``.
    """

    subdirectory = str(adapter_subdirectory).strip()
    if not subdirectory or Path(subdirectory).is_absolute() or ".." in Path(subdirectory).parts:
        raise ValueError(
            "adapter_subdirectory must be a non-empty relative path without '..'."
        )

    def save_hook(models, weights, output_dir) -> None:
        model_index, peft_model = _find_single_peft_model(models, accelerator)
        if model_index >= len(weights):
            raise RuntimeError(
                "Accelerate did not provide a model state for the LoRA model. "
                "LoRA mode must use DDP/no-DeepSpeed."
            )
        if accelerator.is_main_process:
            save_lora_adapter(
                peft_model,
                Path(output_dir) / subdirectory,
                metadata=metadata,
                adapter_name=adapter_name,
            )
        # Prevent save_accelerator_state from serializing the complete frozen Base.
        weights.pop(model_index)

    def load_hook(models, input_dir) -> None:
        model_index, peft_model = _find_single_peft_model(models, accelerator)
        load_lora_adapter_weights(
            peft_model,
            Path(input_dir) / subdirectory,
            adapter_name=adapter_name,
        )
        # Prevent Accelerate from looking for a full model safetensors file.
        models.pop(model_index)

    save_handle = accelerator.register_save_state_pre_hook(save_hook)
    load_handle = accelerator.register_load_state_pre_hook(load_hook)
    return save_handle, load_handle
