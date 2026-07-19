#!/usr/bin/env python3
"""Eight-GPU 1024px SeFi-Image DiT fine-tuning demo.

LoRA and full fine-tuning intentionally share data preprocessing, frozen
encoders, and the numerical SFD loss.  Their only policy differences are the
trainable parameter set, DeepSpeed ZeRO-2, and full-model GPU FP32 EMA.
"""

from __future__ import annotations

import argparse
import gc
import json
import math
import os
import sys
import time
from pathlib import Path
from typing import Any

import torch
from omegaconf import DictConfig, OmegaConf


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from sefi.builder import build_components  # noqa: E402
from sefi.config import load_config as load_model_config  # noqa: E402
from sefi.runner import (  # noqa: E402
    _extract_state_dict,
    _load_checkpoint_payload,
    _resolve_checkpoint_file,
    _strip_prefix_if_needed,
)
from sefi.semvae import SemVAEFeatureCodec  # noqa: E402
from sefi.training.checkpointing import (  # noqa: E402
    TrainerState,
    dataloader_for_resume,
    export_full_transformer_checkpoint,
    load_training_checkpoint,
    save_training_checkpoint,
)
from sefi.training.data import build_paired_dataloader  # noqa: E402
from sefi.training.dit_finetune import (  # noqa: E402
    FIXED_TRAIN_RESOLUTION,
    SFDLossConfig,
    compute_sfd_loss,
    encode_frozen_batch,
    freeze_module,
)
from sefi.training.ema import FullGpuEMA  # noqa: E402
from sefi.training.lora import (  # noqa: E402
    build_lora_metadata,
    count_trainable_parameters,
    create_lora_model,
    register_lora_accelerate_hooks,
    save_lora_adapter,
)


DEFAULT_CONFIG = REPO_ROOT / "demo/dit_finetune/configs/lora_1b.yaml"
SUPPORTED_SCALES = {"1b", "2b", "5b"}
SUPPORTED_MODES = {"lora", "full"}


def _resolve_repo_path(value: str | os.PathLike[str]) -> Path:
    path = Path(value).expanduser()
    return path if path.is_absolute() else REPO_ROOT / path


def _load_config_tree(path: Path, stack: tuple[Path, ...] = ()) -> DictConfig:
    path = path.expanduser().resolve()
    if path in stack:
        chain = " -> ".join(str(item) for item in (*stack, path))
        raise ValueError(f"Recursive demo config defaults: {chain}")
    if not path.is_file():
        raise FileNotFoundError(f"Training config not found: {path}")

    overlay = OmegaConf.load(path)
    defaults = list(overlay.get("defaults", []))
    if "defaults" in overlay:
        del overlay["defaults"]
    merged = OmegaConf.create()
    for reference in defaults:
        if not isinstance(reference, str):
            raise TypeError(
                f"Config defaults entries must be filenames, got {reference!r} in {path}."
            )
        parent = Path(reference)
        if not parent.suffix:
            parent = parent.with_suffix(".yaml")
        if not parent.is_absolute():
            parent = path.parent / parent
        merged = OmegaConf.merge(merged, _load_config_tree(parent, (*stack, path)))
    merged = OmegaConf.merge(merged, overlay)
    OmegaConf.resolve(merged)
    return merged


def load_training_config(path: str | os.PathLike[str]) -> DictConfig:
    """Load one demo config with small, explicit YAML inheritance."""

    return _load_config_tree(_resolve_repo_path(path))


def validate_training_config(config: DictConfig) -> None:
    """Reject recipe drift that would invalidate this intentionally small demo."""

    scale = str(config.model.scale).strip().lower()
    mode = str(config.tuning.mode).strip().lower()
    resolution = int(config.data.resolution)
    if scale not in SUPPORTED_SCALES:
        raise ValueError(f"model.scale must be one of {sorted(SUPPORTED_SCALES)}; got {scale!r}.")
    if mode not in SUPPORTED_MODES:
        raise ValueError(f"tuning.mode must be one of {sorted(SUPPORTED_MODES)}; got {mode!r}.")
    if resolution != FIXED_TRAIN_RESOLUTION:
        raise ValueError(
            f"This demo is fixed at {FIXED_TRAIN_RESOLUTION}px; got data.resolution={resolution}."
        )
    if bool(config.training.repa.enabled):
        raise ValueError("REPA is intentionally unsupported by the DiT fine-tuning demo.")
    if str(config.training.mixed_precision).lower() != "bf16":
        raise ValueError("The supported 80GB-GPU recipe requires training.mixed_precision=bf16.")
    if int(config.training.total_steps) <= 0:
        raise ValueError("training.total_steps must be positive.")
    if int(config.training.gradient_accumulation_steps) <= 0:
        raise ValueError("training.gradient_accumulation_steps must be positive.")

    ema_enabled = bool(config.training.ema.enabled)
    deepspeed_enabled = bool(config.training.deepspeed.enabled)
    if mode == "lora" and (ema_enabled or deepspeed_enabled):
        raise ValueError("LoRA mode must keep EMA and DeepSpeed disabled.")
    if mode == "full" and (not ema_enabled or not deepspeed_enabled):
        raise ValueError("Full mode requires GPU FP32 EMA and DeepSpeed ZeRO-2.")

    if deepspeed_enabled:
        ds_path = _resolve_repo_path(str(config.training.deepspeed.config_file))
        with ds_path.open("r", encoding="utf-8") as handle:
            ds_config = json.load(handle)
        stage = int(ds_config.get("zero_optimization", {}).get("stage", -1))
        if stage != 2:
            raise ValueError(f"Full tuning requires DeepSpeed ZeRO stage 2; got stage={stage}.")
        zero_config = ds_config["zero_optimization"]
        if any(key.startswith("offload_") for key in zero_config):
            raise ValueError("The 80GB-GPU demo does not use CPU/NVMe offload.")


def _configure_deepspeed_for_explicit_dataloader(
    deepspeed_plugin,
    *,
    micro_batch_size: int,
    gradient_accumulation_steps: int,
    gradient_clip: float,
) -> None:
    """Resolve DS values that Accelerate cannot infer without a prepared loader.

    The demo intentionally keeps its explicitly rank-sharded ``DistributedSampler``
    outside ``accelerator.prepare``.  Passing that loader to ``prepare`` would shard
    it a second time.  Accelerate therefore cannot resolve an ``auto`` micro batch
    from a loader and requires us to provide the per-rank value up front.
    """

    micro_batch_size = int(micro_batch_size)
    gradient_accumulation_steps = int(gradient_accumulation_steps)
    gradient_clip = float(gradient_clip)
    if micro_batch_size <= 0:
        raise ValueError("DeepSpeed micro_batch_size must be positive.")
    if gradient_accumulation_steps <= 0:
        raise ValueError("DeepSpeed gradient_accumulation_steps must be positive.")
    if gradient_clip < 0:
        raise ValueError("DeepSpeed gradient_clip must be non-negative.")

    ds_config = deepspeed_plugin.deepspeed_config
    configured_micro_batch = ds_config.get("train_micro_batch_size_per_gpu", "auto")
    if configured_micro_batch == "auto":
        ds_config["train_micro_batch_size_per_gpu"] = micro_batch_size
    elif int(configured_micro_batch) != micro_batch_size:
        raise ValueError(
            "DeepSpeed train_micro_batch_size_per_gpu does not match data.batch_size: "
            f"{configured_micro_batch} != {micro_batch_size}."
        )

    configured_accumulation = ds_config.get("gradient_accumulation_steps", "auto")
    if configured_accumulation != "auto" and (
        int(configured_accumulation) != gradient_accumulation_steps
    ):
        raise ValueError(
            "DeepSpeed gradient_accumulation_steps does not match the training config: "
            f"{configured_accumulation} != {gradient_accumulation_steps}."
        )

    configured_clip = ds_config.get("gradient_clipping", "auto")
    if configured_clip == "auto":
        ds_config["gradient_clipping"] = gradient_clip
    elif not math.isclose(float(configured_clip), gradient_clip, rel_tol=0.0, abs_tol=1e-12):
        raise ValueError(
            "DeepSpeed gradient_clipping does not match training.gradient_clip: "
            f"{configured_clip} != {gradient_clip}."
        )


def _validate_ema_global_step(ema: FullGpuEMA | None, global_step: int) -> None:
    if ema is None:
        return
    if int(ema.optimization_step) != int(global_step):
        raise RuntimeError(
            "EMA/optimizer step mismatch: "
            f"ema={ema.optimization_step}, global_step={int(global_step)}."
        )


def apply_cli_overrides(config: DictConfig, args: argparse.Namespace) -> DictConfig:
    updates = {
        "data.source": args.dataset,
        "data.revision": args.dataset_revision,
        "data.num_workers": args.num_workers,
        "model.base_checkpoint": args.base_checkpoint,
        "model.semvae_checkpoint": args.semvae_checkpoint,
        "model.vfm_checkpoint": args.vfm_checkpoint,
        "checkpointing.output_dir": args.output_dir,
        "training.total_steps": args.max_train_steps,
        "training.optimizer.learning_rate": args.learning_rate,
        "training.scheduler.warmup_steps": args.warmup_steps,
        "checkpointing.save_every": args.save_every,
        "logging.log_every": args.log_every,
    }
    for key, value in updates.items():
        if value is not None:
            OmegaConf.update(config, key, value, merge=False)
    if args.smoke:
        if args.max_train_steps is None:
            config.training.total_steps = 2
        if args.warmup_steps is None:
            config.training.scheduler.warmup_steps = 0
        if args.num_workers is None:
            config.data.num_workers = 0
    OmegaConf.resolve(config)
    return config


def _stage_hf_or_local(
    source: str,
    *,
    revision: str | None,
    cache_dir: Path,
    accelerator,
) -> str:
    candidate = Path(source).expanduser()
    if not candidate.is_absolute():
        rooted = REPO_ROOT / candidate
        if rooted.exists():
            candidate = rooted
    if candidate.exists():
        return str(candidate.absolute())

    try:
        from huggingface_hub import snapshot_download
    except ImportError as exc:
        raise RuntimeError(
            f"{source!r} is not local; install huggingface_hub to download it."
        ) from exc

    cache_dir.mkdir(parents=True, exist_ok=True)
    local_path: str | None = None
    if accelerator.is_main_process:
        local_path = snapshot_download(
            repo_id=source,
            revision=revision,
            cache_dir=str(cache_dir),
            local_files_only=False,
        )
    accelerator.wait_for_everyone()
    if local_path is None:
        local_path = snapshot_download(
            repo_id=source,
            revision=revision,
            cache_dir=str(cache_dir),
            local_files_only=True,
        )
    return local_path


def _load_public_base_weights(transformer: torch.nn.Module, checkpoint_root: str) -> None:
    checkpoint_file = _resolve_checkpoint_file(checkpoint_root)
    payload = _load_checkpoint_payload(checkpoint_file)
    state_dict = _extract_state_dict(payload)
    state_dict = _strip_prefix_if_needed(state_dict, "module.")
    incompatible = transformer.load_state_dict(state_dict, strict=True)
    if incompatible.missing_keys or incompatible.unexpected_keys:
        raise ValueError(
            "Strict Base transformer load failed: "
            f"missing={incompatible.missing_keys[:10]}, "
            f"unexpected={incompatible.unexpected_keys[:10]}."
        )
    del state_dict, payload
    gc.collect()


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True, ensure_ascii=False, default=str)
        handle.write("\n")
    os.replace(temporary, path)


def _append_jsonl(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, sort_keys=True, ensure_ascii=False, default=str))
        handle.write("\n")


def _set_data_epoch(dataloader, epoch: int) -> None:
    if hasattr(dataloader.dataset, "set_epoch"):
        dataloader.dataset.set_epoch(epoch)


def _reduce_metrics(accelerator, metrics: dict[str, float]) -> dict[str, float]:
    names = sorted(metrics)
    values = torch.tensor(
        [float(metrics[name]) for name in names],
        device=accelerator.device,
        dtype=torch.float32,
    )
    reduced = accelerator.reduce(values, reduction="mean")
    return {name: float(value) for name, value in zip(names, reduced.tolist(), strict=True)}


def _prepare_image_branches(
    pixel_values: torch.Tensor,
    *,
    semantic_codec,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Keep the frozen DINO input FP32; cast only the texture branch to BF16."""

    shared_fp32 = pixel_values.to(
        device,
        dtype=torch.float32,
        non_blocking=True,
    )
    vfm_pixel_values = semantic_codec.preprocess_batch(
        shared_fp32,
        input_range="minus_one_one",
    )
    texture_pixel_values = shared_fp32.to(dtype=torch.bfloat16)
    return texture_pixel_values, vfm_pixel_values


def _assert_distributed_finite_loss(
    accelerator,
    loss: torch.Tensor,
    *,
    global_step: int,
) -> None:
    """Make every rank fail together instead of hanging one rank in backward."""

    local_finite = torch.isfinite(loss.detach()).to(
        device=accelerator.device,
        dtype=torch.int64,
    )
    finite_ranks = int(accelerator.reduce(local_finite, reduction="sum").item())
    if finite_ranks != int(accelerator.num_processes):
        local_value = float(loss.detach().float().item())
        raise FloatingPointError(
            "Non-finite SFD loss before optimizer step "
            f"{int(global_step)}: finite_ranks={finite_ranks}/"
            f"{int(accelerator.num_processes)}, local_loss={local_value}."
        )


def _checkpoint_path(output_dir: Path, global_step: int) -> Path:
    return output_dir / "checkpoints" / f"checkpoint-{int(global_step):08d}"


def _save_checkpoint_once(
    accelerator,
    model,
    output_dir: Path,
    state: TrainerState,
    *,
    ema: FullGpuEMA | None = None,
) -> Path:
    del model  # The prepared model is already registered with Accelerator.
    _validate_ema_global_step(ema, state.global_step)
    path = _checkpoint_path(output_dir, state.global_step)
    exists = path.exists() if accelerator.is_main_process else False
    exists_tensor = torch.tensor(int(exists), device=accelerator.device)
    exists = bool(accelerator.reduce(exists_tensor, reduction="max").item())
    if exists:
        raise FileExistsError(f"Refusing to overwrite training checkpoint: {path}")
    save_training_checkpoint(accelerator, path, state)
    if ema is not None and accelerator.is_main_process:
        ema.write_param_names(path)
    accelerator.wait_for_everyone()
    return path


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--dataset", default=None, help="Local dataset directory or HF dataset id.")
    parser.add_argument("--dataset-revision", default=None)
    parser.add_argument("--base-checkpoint", default=None)
    parser.add_argument("--semvae-checkpoint", default=None)
    parser.add_argument("--vfm-checkpoint", default=None)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--max-train-steps", type=int, default=None)
    parser.add_argument("--learning-rate", type=float, default=None)
    parser.add_argument("--warmup-steps", type=int, default=None)
    parser.add_argument("--num-workers", type=int, default=None)
    parser.add_argument("--save-every", type=int, default=None)
    parser.add_argument("--log-every", type=int, default=None)
    parser.add_argument("--resume", default=None, help="Accelerate/DeepSpeed checkpoint directory.")
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="Default to two steps, no warmup, no final checkpoint, and no export.",
    )
    parser.add_argument("--skip-final-checkpoint", action="store_true")
    parser.add_argument("--skip-export", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    config = apply_cli_overrides(load_training_config(args.config), args)
    validate_training_config(config)

    from accelerate import Accelerator, DeepSpeedPlugin
    from accelerate.utils import DistributedDataParallelKwargs, ProjectConfiguration, set_seed
    from diffusers.optimization import get_scheduler

    mode = str(config.tuning.mode).lower()
    scale = str(config.model.scale).lower()
    output_dir = _resolve_repo_path(str(config.checkpointing.output_dir))
    output_dir.mkdir(parents=True, exist_ok=True)

    deepspeed_plugin = None
    kwargs_handlers = []
    if mode == "full":
        deepspeed_plugin = DeepSpeedPlugin(
            hf_ds_config=str(_resolve_repo_path(str(config.training.deepspeed.config_file)))
        )
        _configure_deepspeed_for_explicit_dataloader(
            deepspeed_plugin,
            micro_batch_size=int(config.data.batch_size),
            gradient_accumulation_steps=int(config.training.gradient_accumulation_steps),
            gradient_clip=float(config.training.gradient_clip),
        )
    else:
        kwargs_handlers.append(DistributedDataParallelKwargs(find_unused_parameters=False))

    accelerator = Accelerator(
        mixed_precision="bf16",
        gradient_accumulation_steps=int(config.training.gradient_accumulation_steps),
        deepspeed_plugin=deepspeed_plugin,
        kwargs_handlers=kwargs_handlers,
        step_scheduler_with_optimizer=False,
        project_config=ProjectConfiguration(project_dir=str(output_dir)),
    )
    set_seed(int(config.seed), device_specific=True)

    expected_world_size = int(config.distributed.expected_world_size)
    if accelerator.num_processes != expected_world_size:
        accelerator.print(
            "Warning: this recipe is intended for "
            f"{expected_world_size} GPUs, but the current launch has "
            f"{accelerator.num_processes}."
        )
    if mode == "full" and accelerator.distributed_type.name != "DEEPSPEED":
        raise RuntimeError(
            "Full tuning must run through the configured DeepSpeed plugin. "
            f"Accelerate selected {accelerator.distributed_type}."
        )

    if accelerator.is_main_process:
        OmegaConf.save(config, output_dir / "resolved_training_config.yaml")
    accelerator.wait_for_everyone()

    model_cache = REPO_ROOT / "outputs/cache/huggingface/models"
    base_checkpoint = _stage_hf_or_local(
        str(config.model.base_checkpoint),
        revision=str(config.model.base_revision),
        cache_dir=model_cache,
        accelerator=accelerator,
    )
    semvae_checkpoint = _stage_hf_or_local(
        str(config.model.semvae_checkpoint),
        revision=str(config.model.semvae_revision),
        cache_dir=model_cache,
        accelerator=accelerator,
    )
    vfm_checkpoint = _stage_hf_or_local(
        str(config.model.vfm_checkpoint),
        revision=str(config.model.vfm_revision),
        cache_dir=model_cache,
        accelerator=accelerator,
    )

    accelerator.print(
        f"Initializing {scale.upper()} {mode} fine-tuning on "
        f"{accelerator.num_processes} process(es)."
    )
    model_config = load_model_config(Path(base_checkpoint) / "sefi_config.yaml")
    OmegaConf.update(
        model_config,
        "model.text_encoder.max_length",
        int(config.model.text_max_length),
        merge=False,
    )
    components = build_components(model_config, component_dtype=torch.bfloat16)
    transformer = components.transformer.to(dtype=torch.bfloat16)
    _load_public_base_weights(transformer, base_checkpoint)

    if bool(config.training.gradient_checkpointing):
        transformer.enable_gradient_checkpointing()

    lora_metadata = None
    if mode == "lora":
        lora_cfg = config.tuning.lora
        transformer, target_modules = create_lora_model(
            transformer,
            scale=scale,
            rank=int(lora_cfg.rank),
            alpha=int(lora_cfg.alpha),
            dropout=float(lora_cfg.dropout),
        )
        lora_metadata = build_lora_metadata(
            base_model_repo=str(config.model.base_checkpoint),
            scale=scale,
            target_modules=target_modules,
            rank=int(lora_cfg.rank),
            alpha=int(lora_cfg.alpha),
            dropout=float(lora_cfg.dropout),
            resolution=int(config.data.resolution),
            requested_base_revision=str(config.model.base_revision),
            resolved_base_revision=str(config.model.base_revision),
            training_config=config,
        )
    else:
        transformer.requires_grad_(True)

    trainable_parameters, total_parameters = count_trainable_parameters(transformer)
    if trainable_parameters <= 0:
        raise RuntimeError("No trainable transformer parameters were selected.")
    accelerator.print(
        f"Trainable parameters: {trainable_parameters:,} / {total_parameters:,} "
        f"({100.0 * trainable_parameters / total_parameters:.4f}%)."
    )

    texture_codec = freeze_module(components.texture_codec).to(
        accelerator.device, dtype=torch.bfloat16
    )
    text_encoder = freeze_module(components.text_encoder).to(
        accelerator.device, dtype=torch.bfloat16
    )
    semantic_codec = SemVAEFeatureCodec.from_pretrained(
        semvae_checkpoint,
        vfm_checkpoint=vfm_checkpoint,
        cache_dir=model_cache,
        device=accelerator.device,
        image_size=int(config.data.resolution),
    )

    data_source = str(config.data.source)
    candidate_source = Path(data_source).expanduser()
    if not candidate_source.is_absolute() and (REPO_ROOT / candidate_source).exists():
        data_source = str(REPO_ROOT / candidate_source)
    dataset_revision = config.data.get("revision", None)
    dataset_revision = None if dataset_revision is None else str(dataset_revision)
    with accelerator.main_process_first():
        dataloader, _ = build_paired_dataloader(
            data_source,
            split=str(config.data.split),
            revision=dataset_revision,
            cache_dir=_resolve_repo_path(str(config.data.cache_dir)),
            resolution=int(config.data.resolution),
            batch_size=int(config.data.batch_size),
            num_workers=int(config.data.num_workers),
            caption_mode=str(config.data.caption.mode),
            caption_weights=OmegaConf.to_container(config.data.caption.weights, resolve=True),
            sampler_seed=int(config.data.sampler_seed),
            rank=accelerator.process_index,
            world_size=accelerator.num_processes,
            shuffle=bool(config.data.shuffle),
            drop_last=bool(config.data.drop_last),
            pin_memory=True,
        )
    if len(dataloader) <= 0:
        raise ValueError("The paired training dataloader is empty.")

    optimizer_cfg = config.training.optimizer
    optimizer = torch.optim.AdamW(
        [parameter for parameter in transformer.parameters() if parameter.requires_grad],
        lr=float(optimizer_cfg.learning_rate),
        betas=tuple(float(value) for value in optimizer_cfg.betas),
        eps=float(optimizer_cfg.eps),
        weight_decay=float(optimizer_cfg.weight_decay),
    )
    scheduler = get_scheduler(
        str(config.training.scheduler.name),
        optimizer=optimizer,
        num_warmup_steps=int(config.training.scheduler.warmup_steps),
        num_training_steps=int(config.training.total_steps),
    )

    transformer, optimizer, scheduler = accelerator.prepare(transformer, optimizer, scheduler)
    if mode == "lora":
        register_lora_accelerate_hooks(accelerator, metadata=lora_metadata)

    ema = None
    if mode == "full":
        ema = FullGpuEMA(
            transformer,
            accelerator=accelerator,
            decay=float(config.training.ema.decay),
            use_ema_warmup=bool(config.training.ema.use_ema_warmup),
        )

    state = TrainerState()
    if args.resume:
        state = load_training_checkpoint(accelerator, _resolve_repo_path(args.resume))
        _validate_ema_global_step(ema, state.global_step)
        accelerator.print(
            f"Resumed at optimizer step {state.global_step}, "
            f"data epoch {state.data_epoch}, batch offset {state.batch_offset}."
        )

    loss_config = SFDLossConfig.from_training_config(
        config.training,
        semantic_channels=components.semantic_channels,
        texture_channels=components.texture_channels,
    )
    total_steps = int(config.training.total_steps)
    save_every = int(config.checkpointing.save_every)
    log_every = int(config.logging.log_every)
    if save_every < 0 or log_every <= 0:
        raise ValueError("checkpointing.save_every must be >= 0 and logging.log_every > 0.")

    metrics_path = output_dir / "metrics.jsonl"
    history: list[dict[str, Any]] = []
    start_time = time.monotonic()
    transformer.train()
    if accelerator.device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(accelerator.device)

    while state.global_step < total_steps:
        _set_data_epoch(dataloader, state.data_epoch)
        epoch_dataloader = dataloader_for_resume(accelerator, dataloader, state)
        consumed_any = False

        for batch in epoch_dataloader:
            consumed_any = True
            did_optimizer_step = False
            local_metrics: dict[str, float]
            with accelerator.accumulate(transformer):
                pixel_values, vfm_pixel_values = _prepare_image_branches(
                    batch["pixel_values"],
                    semantic_codec=semantic_codec,
                    device=accelerator.device,
                )
                encoded = encode_frozen_batch(
                    pixel_values=pixel_values,
                    vfm_pixel_values=vfm_pixel_values,
                    captions=batch["captions"],
                    texture_codec=texture_codec,
                    semantic_codec=semantic_codec,
                    text_encoder=text_encoder,
                    pipeline_cls=components.pipeline_cls,
                    semantic_channels=components.semantic_channels,
                    texture_channels=components.texture_channels,
                    semantic_sample=False,
                    drop_text_probability=float(
                        config.training.conditioning.drop_text_probability
                    ),
                    unconditional_prompt=str(
                        config.training.conditioning.unconditional_prompt
                    ),
                )
                loss, local_metrics = compute_sfd_loss(
                    encoded=encoded,
                    model=transformer,
                    noise_scheduler=components.noise_scheduler,
                    pipeline_cls=components.pipeline_cls,
                    config=loss_config,
                )
                _assert_distributed_finite_loss(
                    accelerator,
                    loss,
                    global_step=state.global_step,
                )
                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(
                        transformer.parameters(), float(config.training.gradient_clip)
                    )
                optimizer.step()
                did_optimizer_step = bool(
                    accelerator.sync_gradients
                    and not accelerator.optimizer_step_was_skipped
                )
                if did_optimizer_step:
                    scheduler.step()
                optimizer.zero_grad(set_to_none=True)

            if ema is not None:
                ema.step(did_optimizer_step=did_optimizer_step)

            state = state.after_batch(len(dataloader))
            if not did_optimizer_step:
                continue
            state = state.with_global_step(state.global_step + 1)

            local_metrics["learning_rate"] = float(scheduler.get_last_lr()[0])
            local_metrics["global_step"] = float(state.global_step)
            local_metrics["data_epoch"] = float(state.data_epoch)
            local_metrics["batch_offset"] = float(state.batch_offset)
            reduced_metrics = _reduce_metrics(accelerator, local_metrics)
            reduced_metrics["global_step"] = state.global_step
            reduced_metrics["data_epoch"] = state.data_epoch
            reduced_metrics["batch_offset"] = state.batch_offset
            reduced_metrics["elapsed_seconds"] = time.monotonic() - start_time
            if accelerator.device.type == "cuda":
                reduced_metrics["max_memory_gib"] = (
                    torch.cuda.max_memory_allocated(accelerator.device) / (1024**3)
                )

            if state.global_step % log_every == 0 or state.global_step == total_steps:
                if accelerator.is_main_process:
                    _append_jsonl(metrics_path, reduced_metrics)
                    history.append(reduced_metrics)
                accelerator.print(
                    f"step={state.global_step}/{total_steps} "
                    f"loss={reduced_metrics['loss_total']:.6f} "
                    f"sem={reduced_metrics['loss_sem']:.6f} "
                    f"tex={reduced_metrics['loss_tex']:.6f} "
                    f"lr={reduced_metrics['learning_rate']:.3e}"
                )

            if save_every > 0 and state.global_step % save_every == 0:
                _save_checkpoint_once(
                    accelerator,
                    transformer,
                    output_dir,
                    state,
                    ema=ema,
                )

            if state.global_step >= total_steps:
                break

        if not consumed_any:
            raise RuntimeError(
                "The resumed dataloader produced no batches. Check trainer_state.json "
                "and the current dataset/world-size configuration."
            )

    _validate_ema_global_step(ema, state.global_step)
    skip_final_checkpoint = bool(args.skip_final_checkpoint or args.smoke)
    skip_export = bool(args.skip_export or args.smoke or not config.export.enabled)
    final_checkpoint = None
    if bool(config.checkpointing.save_final) and not skip_final_checkpoint:
        expected_path = _checkpoint_path(output_dir, state.global_step)
        if not expected_path.exists():
            final_checkpoint = _save_checkpoint_once(
                accelerator,
                transformer,
                output_dir,
                state,
                ema=ema,
            )
        else:
            final_checkpoint = expected_path

    export_path = None
    if not skip_export:
        if mode == "lora":
            export_path = output_dir / "adapter"
            if accelerator.is_main_process:
                save_lora_adapter(
                    accelerator.unwrap_model(transformer),
                    export_path,
                    metadata=lora_metadata,
                )
            accelerator.wait_for_everyone()
        else:
            _validate_ema_global_step(ema, state.global_step)
            export_path = output_dir / "export"
            unwrapped = accelerator.unwrap_model(transformer)
            transformer_config = dict(unwrapped.backbone.config)
            export_full_transformer_checkpoint(
                accelerator,
                transformer,
                export_path,
                ema_model=ema,
                ema_param_names=ema.param_names,
                max_shard_size=str(config.export.max_shard_size),
                dtype=str(config.export.dtype),
                transformer_config=transformer_config,
                metadata={
                    "base_model_repo": str(config.model.base_checkpoint),
                    "base_revision": str(config.model.base_revision),
                    "scale": scale,
                    "resolution": int(config.data.resolution),
                    "global_step": state.global_step,
                },
            )
            accelerator.wait_for_everyone()

    if accelerator.is_main_process:
        finite_losses = [float(item["loss_total"]) for item in history]
        summary = {
            "schema_version": 1,
            "status": "completed",
            "mode": mode,
            "scale": scale,
            "world_size": accelerator.num_processes,
            "resolution": int(config.data.resolution),
            "global_step": state.global_step,
            "dataset_rows": len(dataloader.dataset),
            "trainable_parameters": trainable_parameters,
            "total_parameters": total_parameters,
            "all_logged_losses_finite": all(math.isfinite(value) for value in finite_losses),
            "first_logged_loss": finite_losses[0] if finite_losses else None,
            "last_logged_loss": finite_losses[-1] if finite_losses else None,
            "elapsed_seconds": time.monotonic() - start_time,
            "max_memory_gib": (
                torch.cuda.max_memory_allocated(accelerator.device) / (1024**3)
                if accelerator.device.type == "cuda"
                else None
            ),
            "final_checkpoint": str(final_checkpoint) if final_checkpoint else None,
            "export_path": str(export_path) if export_path else None,
        }
        _write_json(output_dir / "run_summary.json", summary)
    accelerator.wait_for_everyone()
    accelerator.print(f"Training completed at optimizer step {state.global_step}.")
    accelerator.end_training()


if __name__ == "__main__":
    main()
