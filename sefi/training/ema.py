"""Per-rank GPU FP32 EMA for full DiT fine-tuning.

The moving-average schedule is delegated to Diffusers' ``EMAModel`` so its
step-offset and early-step decay behavior stay identical to the reference
training code.  This wrapper adds stable parameter-name mapping, strict resume
validation, and explicit post-load device restoration.
"""

from __future__ import annotations

import json
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
from torch import Tensor

from diffusers.training_utils import EMAModel


class FullGpuEMA:
    """A complete FP32 EMA shadow on every rank.

    Instantiate this *after* ``accelerator.prepare`` and after loading the full
    public Base checkpoint.  The wrapped model itself is not serialized by this
    object; Accelerate registers only the EMA state.
    """

    STATE_VERSION = 1

    def __init__(
        self,
        trainable_model: nn.Module,
        *,
        accelerator: Any | None = None,
        decay: float = 0.9999,
        use_ema_warmup: bool = False,
        register_for_checkpointing: bool = True,
    ) -> None:
        if not 0.0 <= float(decay) <= 1.0:
            raise ValueError(f"EMA decay must be in [0, 1], got {decay}.")
        self.trainable_model = trainable_model
        self.accelerator = accelerator

        named_parameters = self._current_named_parameters()
        if not named_parameters:
            raise ValueError("Cannot create EMA for a model with no parameters.")
        self.param_names = tuple(name for name, _ in named_parameters)
        self._param_shapes = tuple(tuple(param.shape) for _, param in named_parameters)
        model_device = named_parameters[0][1].device
        self.ema_model = EMAModel(
            [param for _, param in named_parameters],
            decay=float(decay),
            use_ema_warmup=bool(use_ema_warmup),
        )
        self.ema_model.to(device=model_device, dtype=torch.float32)
        self.validate()

        if accelerator is not None and register_for_checkpointing:
            accelerator.register_for_checkpointing(self)

    def _unwrap_model(self) -> nn.Module:
        if self.accelerator is None:
            return self.trainable_model
        return self.accelerator.unwrap_model(self.trainable_model)

    def _current_named_parameters(self) -> list[tuple[str, nn.Parameter]]:
        return list(self._unwrap_model().named_parameters())

    @property
    def optimization_step(self) -> int:
        return int(self.ema_model.optimization_step)

    @property
    def shadow_params(self) -> list[Tensor]:
        """Expose Diffusers shadows for the portable checkpoint helper."""

        return self.ema_model.shadow_params

    @property
    def current_decay(self) -> float | None:
        value = getattr(self.ema_model, "cur_decay_value", None)
        return None if value is None else float(value)

    @property
    def device(self) -> torch.device:
        return self.ema_model.shadow_params[0].device

    def get_decay(self, optimization_step: int) -> float:
        """Expose Diffusers' exact EMA schedule for logging/testing."""

        return float(self.ema_model.get_decay(int(optimization_step)))

    def validate(self) -> None:
        """Fail fast if names/order/shapes/dtype/device diverge from the model."""

        named_parameters = self._current_named_parameters()
        current_names = tuple(name for name, _ in named_parameters)
        if current_names != self.param_names:
            raise ValueError(
                "EMA parameter-name/order mismatch. "
                f"checkpoint={list(self.param_names)[:5]}, "
                f"model={list(current_names)[:5]}."
            )
        if len(self.ema_model.shadow_params) != len(named_parameters):
            raise ValueError(
                "EMA parameter count mismatch: "
                f"shadow={len(self.ema_model.shadow_params)}, "
                f"model={len(named_parameters)}."
            )

        expected_device = named_parameters[0][1].device
        for index, ((name, parameter), shadow) in enumerate(
            zip(named_parameters, self.ema_model.shadow_params)
        ):
            expected_shape = tuple(parameter.shape)
            if tuple(shadow.shape) != expected_shape:
                raise ValueError(
                    f"EMA shape mismatch for {name!r} at index {index}: "
                    f"shadow={tuple(shadow.shape)}, model={expected_shape}."
                )
            if shadow.dtype != torch.float32:
                raise ValueError(
                    f"EMA shadow {name!r} must be float32, got {shadow.dtype}."
                )
            if shadow.device != expected_device:
                raise ValueError(
                    f"EMA shadow {name!r} is on {shadow.device}, "
                    f"but model is on {expected_device}."
                )

    def restore_device_and_validate(self) -> None:
        """Move a loaded EMA state back to this rank's model device in FP32."""

        named_parameters = self._current_named_parameters()
        if not named_parameters:
            raise ValueError("Cannot restore EMA for a model with no parameters.")
        self.ema_model.to(
            device=named_parameters[0][1].device,
            dtype=torch.float32,
        )
        self.validate()

    @torch.no_grad()
    def step(
        self,
        *,
        did_optimizer_step: bool,
    ) -> bool:
        """Update only after a real optimizer step.

        Call with ``accelerator.sync_gradients and not
        accelerator.optimizer_step_was_skipped``.  Returning whether an update
        happened makes the train loop's global-step accounting explicit.
        """

        if not bool(did_optimizer_step):
            return False
        named_parameters = self._current_named_parameters()
        current_names = tuple(name for name, _ in named_parameters)
        if current_names != self.param_names:
            raise ValueError("EMA parameter mapping changed after initialization.")
        self.ema_model.step([param for _, param in named_parameters])
        # Diffusers preserves FP32 shadows during the mixed-dtype update; keep a
        # cheap strict check here because losing FP32 silently invalidates EMA.
        if any(shadow.dtype != torch.float32 for shadow in self.ema_model.shadow_params):
            raise RuntimeError("EMA update changed a shadow parameter away from float32.")
        return True

    def named_shadow_parameters(self) -> Iterator[tuple[str, Tensor]]:
        for name, shadow in zip(self.param_names, self.ema_model.shadow_params):
            yield name, shadow

    def shadow_state_dict(self, *, clone: bool = False) -> dict[str, Tensor]:
        """Return the EMA transformer weights keyed by stable model names."""

        if clone:
            return {
                name: tensor.detach().clone()
                for name, tensor in self.named_shadow_parameters()
            }
        return {
            name: tensor.detach() for name, tensor in self.named_shadow_parameters()
        }

    @torch.no_grad()
    def copy_to(self, model: nn.Module | None = None) -> None:
        """Copy EMA weights to a compatible model, validating names first."""

        model = self._unwrap_model() if model is None else model
        named_parameters = list(model.named_parameters())
        names = tuple(name for name, _ in named_parameters)
        if names != self.param_names:
            raise ValueError("Cannot copy EMA into a model with a different parameter mapping.")
        for (_, parameter), shadow in zip(named_parameters, self.ema_model.shadow_params):
            parameter.data.copy_(shadow.to(parameter.device, parameter.dtype))

    @contextmanager
    def apply_to(self, model: nn.Module | None = None):
        """Temporarily swap a compatible model to EMA weights for export/eval."""

        model = self._unwrap_model() if model is None else model
        named_parameters = list(model.named_parameters())
        names = tuple(name for name, _ in named_parameters)
        if names != self.param_names:
            raise ValueError("Cannot apply EMA to a model with a different parameter mapping.")
        original = [parameter.detach().clone() for _, parameter in named_parameters]
        self.copy_to(model)
        try:
            yield model
        finally:
            with torch.no_grad():
                for (_, parameter), value in zip(named_parameters, original):
                    parameter.data.copy_(value)

    def state_dict(self) -> dict[str, Any]:
        """Accelerate-compatible state including the stable name mapping."""

        return {
            "state_version": self.STATE_VERSION,
            "param_names": list(self.param_names),
            "param_shapes": [list(shape) for shape in self._param_shapes],
            "ema": self.ema_model.state_dict(),
        }

    def load_state_dict(self, state_dict: Mapping[str, Any]) -> None:
        """Load, map-check, then explicitly restore this rank's device/FP32."""

        version = int(state_dict.get("state_version", -1))
        if version != self.STATE_VERSION:
            raise ValueError(
                f"Unsupported EMA state_version={version}; expected {self.STATE_VERSION}."
            )
        loaded_names = tuple(str(name) for name in state_dict.get("param_names", ()))
        if loaded_names != self.param_names:
            raise ValueError(
                "EMA checkpoint parameter mapping does not match the current model. "
                f"checkpoint={list(loaded_names)[:5]}, "
                f"model={list(self.param_names)[:5]}."
            )
        loaded_shapes = tuple(
            tuple(int(value) for value in shape)
            for shape in state_dict.get("param_shapes", ())
        )
        if loaded_shapes != self._param_shapes:
            raise ValueError("EMA checkpoint parameter shapes do not match the current model.")
        inner_state = state_dict.get("ema")
        if not isinstance(inner_state, Mapping):
            raise ValueError("EMA checkpoint is missing the inner Diffusers EMA state.")
        self.ema_model.load_state_dict(dict(inner_state))
        self.restore_device_and_validate()

    def write_param_names(self, path: str | Path) -> Path:
        """Write the stable name/shape manifest next to a training checkpoint."""

        destination = Path(path)
        if destination.suffix != ".json":
            destination = destination / "ema_param_names.json"
        destination.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "state_version": self.STATE_VERSION,
            "param_names": list(self.param_names),
            "param_shapes": [list(shape) for shape in self._param_shapes],
        }
        destination.write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        return destination
