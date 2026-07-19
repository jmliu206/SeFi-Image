"""SeFi-Image inference pipeline wrapper."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import torch
from PIL import Image

from .checkpoints import (
    resolve_config_path,
    resolve_checkpoint_to_local,
)
from .registry import ModelSpec, infer_model_spec
from .resolution import resolve_image_size
from .runtime import load_runtime_symbols


SUPPORTED_DISTILL_STEPS = {4, 8, 10}


class SEFIInferencePipeline:
    """Inference wrapper for SeFi-Image checkpoints."""

    def __init__(
        self,
        *,
        spec: ModelSpec,
        runner,
        checkpoint_path: str,
        checkpoint_uri: str,
        transformer_checkpoint_path: str = "",
        transformer_checkpoint_uri: str = "",
        adapter_path: str = "",
        adapter_uri: str = "",
    ) -> None:
        self.spec = spec
        self.runner = runner
        self.checkpoint_path = checkpoint_path
        self.checkpoint_uri = checkpoint_uri
        self.transformer_checkpoint_path = transformer_checkpoint_path
        self.transformer_checkpoint_uri = transformer_checkpoint_uri
        self.adapter_path = adapter_path
        self.adapter_uri = adapter_uri

    @classmethod
    def from_pretrained(
        cls,
        checkpoint: str,
        *,
        cache_dir: str | Path = "outputs/model_weights/sefi_inference",
        config: str | Path | None = None,
        device: str | None = None,
        dtype: str | None = None,
        transformer_checkpoint_path: str | Path | None = None,
        adapter_path: str | Path | None = None,
        delta_t: float | None = None,
        timestep_shift_alpha: float | None = None,
        debug_assert_schedule: bool = False,
        autoguidance_config: str | None = None,
        autoguidance_checkpoint: str | None = None,
        guidance_interval_sigma_lo: float | None = None,
        guidance_interval_sigma_hi: float | None = None,
    ) -> "SEFIInferencePipeline":
        runner_cls, load_config = load_runtime_symbols()

        local_checkpoint, checkpoint_uri = resolve_checkpoint_to_local(
            checkpoint=checkpoint,
            cache_dir=cache_dir,
        )
        resolved_config = load_config(resolve_config_path(local_checkpoint, config))
        spec = infer_model_spec(
            resolved_config,
            checkpoint_uri=checkpoint_uri,
            checkpoint_path=local_checkpoint,
        )
        local_transformer_checkpoint = ""
        transformer_checkpoint_uri = ""
        if transformer_checkpoint_path:
            (
                local_transformer_checkpoint,
                transformer_checkpoint_uri,
            ) = resolve_checkpoint_to_local(
                checkpoint=str(transformer_checkpoint_path),
                cache_dir=cache_dir,
            )
        local_adapter = ""
        adapter_uri = ""
        if adapter_path:
            if spec.family != "base":
                raise ValueError(
                    "PEFT adapters are supported only with SeFi Base checkpoints; "
                    f"got model family {spec.family!r}."
                )
            local_adapter, adapter_uri = resolve_checkpoint_to_local(
                checkpoint=str(adapter_path),
                cache_dir=cache_dir,
            )
        resolved_device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        resolved_dtype = dtype or spec.default_dtype
        resolved_delta_t = delta_t if delta_t is not None else spec.default_delta_t
        resolved_timestep_shift_alpha = (
            timestep_shift_alpha
            if timestep_shift_alpha is not None
            else spec.default_timestep_shift_alpha
        )

        runner_kwargs = dict(
            checkpoint_path=local_checkpoint,
            device=resolved_device,
            debug_assert_schedule=debug_assert_schedule,
            delta_t_override=resolved_delta_t,
            inference_dtype=resolved_dtype,
            timestep_shift_alpha=resolved_timestep_shift_alpha,
            autoguidance_config_path=autoguidance_config,
            autoguidance_checkpoint_path=autoguidance_checkpoint,
            guidance_interval_sigma_lo=guidance_interval_sigma_lo,
            guidance_interval_sigma_hi=guidance_interval_sigma_hi,
        )
        # Keep the no-adapter constructor call byte-for-byte compatible with
        # older runtime implementations that do not accept this new keyword.
        if local_adapter:
            runner_kwargs["adapter_path"] = local_adapter
        if local_transformer_checkpoint:
            runner_kwargs["transformer_checkpoint_path"] = local_transformer_checkpoint

        runner = runner_cls(
            resolved_config,
            **runner_kwargs,
        )

        return cls(
            spec=spec,
            runner=runner,
            checkpoint_path=local_checkpoint,
            checkpoint_uri=checkpoint_uri,
            transformer_checkpoint_path=local_transformer_checkpoint,
            transformer_checkpoint_uri=transformer_checkpoint_uri,
            adapter_path=local_adapter,
            adapter_uri=adapter_uri,
        )

    def __call__(
        self,
        prompts: str | Iterable[str],
        *,
        num_inference_steps: int | None = None,
        guidance_scale: float | None = None,
        height: int | None = None,
        width: int | None = None,
        batch_size: int | None = None,
        seed: int | None = None,
        generator: torch.Generator | None = None,
    ) -> list[Image.Image]:
        prompt_list = [prompts] if isinstance(prompts, str) else list(prompts)
        if not prompt_list:
            return []

        steps = int(
            num_inference_steps
            if num_inference_steps is not None
            else self.spec.default_steps
        )
        guidance = float(
            guidance_scale
            if guidance_scale is not None
            else self.spec.default_guidance_scale
        )
        size = resolve_image_size(
            height=height,
            width=width,
            default_height=self.spec.default_height,
            default_width=self.spec.default_width,
        )

        if self.spec.is_distilled:
            if steps not in SUPPORTED_DISTILL_STEPS:
                raise ValueError(
                    "SEFI Turbo models currently support "
                    f"{sorted(SUPPORTED_DISTILL_STEPS)} steps, got {steps}."
                )
            if guidance != 1.0:
                raise ValueError("SEFI Turbo models should run with guidance_scale=1.0.")

        bs = int(batch_size or len(prompt_list))
        if bs <= 0:
            raise ValueError("batch_size must be > 0.")

        gen = generator
        if gen is None and seed is not None:
            gen = torch.Generator(device=str(self.runner.device)).manual_seed(int(seed))

        images: list[Image.Image] = []
        for start in range(0, len(prompt_list), bs):
            chunk = prompt_list[start : start + bs]
            images.extend(
                self.runner.generate_batch(
                    prompts=chunk,
                    num_inference_steps=steps,
                    guidance_scale=guidance,
                    height=size.height,
                    width=size.width,
                    generator=gen,
                )
            )
        return images
