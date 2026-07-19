"""SEFI T2I inference runner with three-phase masked denoising."""

from __future__ import annotations

import json
import math
import os
from typing import Optional

import torch
from PIL import Image
from torch import Tensor

from .builder import (
    build_components,
    build_lightweight_transformer,
    _derive_semantic_channels,
    _derive_text_output_dim,
    _derive_texture_channels,
    _resolve_transformer_scale,
    text_encoder_signature,
)
from .config import load_config
from .modeling import Qwen3VLTextEncoder


def _resolve_weight_dtype(config, *, override: Optional[str] = None) -> torch.dtype:
    if override is not None:
        normalized = str(override).strip().lower()
        if normalized == "bf16":
            return torch.bfloat16
        if normalized in {"fp32", "float32"}:
            return torch.float32
        raise ValueError(
            f"Unsupported inference dtype: {override}. Expected one of ['bf16', 'fp32']."
        )

    precision = str(getattr(config.training, "mixed_precision", "bf16")).lower()
    if precision == "fp16":
        return torch.float16
    if precision in {"fp32", "float32", "no"}:
        return torch.float32
    return torch.bfloat16


def _training_sefi_cfg(config):
    cfg = config.training.get("sefi", None)
    if cfg is not None:
        return cfg
    raise ValueError("Config requires training.sefi section.")


def _apply_timestep_shift_unit_interval(u_unit: Tensor, alpha: float) -> Tensor:
    """Apply t' = alpha*t / (1 + (alpha-1)*t) on unit coordinate u in [0, 1]."""
    alpha = float(alpha)
    if alpha <= 0:
        raise ValueError(f"timestep_shift_alpha must be > 0, got {alpha}")
    if alpha == 1.0:
        return u_unit
    denominator = 1.0 + (alpha - 1.0) * u_unit
    return (alpha * u_unit) / denominator


def _combine_guided_velocity(base_pred: Tensor, cond_pred: Tensor, guidance_scale: float) -> Tensor:
    """Shared guidance formula: base + scale * (conditioned - base)."""
    return base_pred + float(guidance_scale) * (cond_pred - base_pred)


def _resolve_guidance_interval_sigma(
    sigma_lo: Optional[float],
    sigma_hi: Optional[float],
) -> tuple[Optional[float], Optional[float]]:
    if sigma_lo is None and sigma_hi is None:
        return None, None
    if sigma_lo is None or sigma_hi is None:
        raise ValueError(
            "Limited interval guidance requires both "
            "guidance_interval_sigma_lo and guidance_interval_sigma_hi, or neither."
        )

    sigma_lo = float(sigma_lo)
    sigma_hi = float(sigma_hi)
    if not math.isfinite(sigma_lo) or not math.isfinite(sigma_hi):
        raise ValueError("guidance interval sigma thresholds must be finite.")
    if sigma_lo < 0.0 or sigma_hi < 0.0:
        raise ValueError("guidance interval sigma thresholds must be >= 0.")
    if sigma_lo >= sigma_hi:
        raise ValueError("guidance_interval_sigma_lo must be < guidance_interval_sigma_hi.")
    return sigma_lo, sigma_hi


def _guidance_interval_is_active(
    sigma: Tensor | float,
    sigma_lo: Optional[float],
    sigma_hi: Optional[float],
) -> bool:
    if sigma_lo is None and sigma_hi is None:
        return True
    if sigma_lo is None or sigma_hi is None:
        raise ValueError("guidance interval sigma bounds must be paired.")

    sigma_value = float(sigma.item()) if isinstance(sigma, Tensor) else float(sigma)
    return float(sigma_lo) < sigma_value <= float(sigma_hi)


def _normalize_optional_path(path: Optional[str]) -> str:
    if path is None:
        return ""
    return str(path).strip()


def _resolve_autoguidance_paths(
    autoguidance_config_path: Optional[str],
    autoguidance_checkpoint_path: Optional[str],
) -> tuple[str, str]:
    config_path = _normalize_optional_path(autoguidance_config_path)
    checkpoint_path = _normalize_optional_path(autoguidance_checkpoint_path)
    if bool(config_path) != bool(checkpoint_path):
        raise ValueError(
            "AutoGuidance requires both --autoguidance_config and "
            "--autoguidance_checkpoint, or neither."
        )
    return config_path, checkpoint_path


def _resolve_peft_adapter_directory(adapter_path: str) -> str:
    """Resolve either a clean adapter root or a training checkpoint root."""

    root = os.path.abspath(os.path.expanduser(str(adapter_path)))
    if not os.path.isdir(root):
        raise FileNotFoundError(f"PEFT adapter directory not found: {adapter_path}")

    candidates = (root, os.path.join(root, "adapter"))
    for candidate in candidates:
        if os.path.isfile(os.path.join(candidate, "adapter_config.json")):
            return candidate
    raise FileNotFoundError(
        "PEFT adapter_config.json not found. Expected it directly under "
        f"{root} or under {os.path.join(root, 'adapter')}."
    )


def _validate_autoguidance_guidance_scale(enabled: bool, guidance_scale: float) -> None:
    if enabled and float(guidance_scale) <= 1.0:
        raise ValueError("AutoGuidance requires guidance_scale > 1.0.")


def _resolve_checkpoint_file(checkpoint_path: str) -> str:
    if os.path.isdir(checkpoint_path):
        transformer_dir = os.path.join(checkpoint_path, "transformer")
        sharded_safetensors = os.path.join(
            transformer_dir,
            "diffusion_pytorch_model.safetensors.index.json",
        )
        if os.path.isfile(sharded_safetensors):
            return sharded_safetensors

        safetensors_state = os.path.join(
            transformer_dir,
            "diffusion_pytorch_model.safetensors",
        )
        if os.path.isfile(safetensors_state):
            return safetensors_state

        torch_state = os.path.join(transformer_dir, "diffusion_pytorch_model.bin")
        if os.path.isfile(torch_state):
            return torch_state

        raise FileNotFoundError(
            f"Unsupported SEFI inference checkpoint directory: {checkpoint_path}. "
            "Expected transformer/diffusion_pytorch_model.safetensors or "
            "transformer/diffusion_pytorch_model.safetensors.index.json."
        )

    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint path not found: {checkpoint_path}")

    return checkpoint_path


def _extract_state_dict(checkpoint: dict) -> dict:
    if not isinstance(checkpoint, dict):
        raise ValueError("Checkpoint must be a dict-like object.")

    if "model_state_dict" in checkpoint and isinstance(checkpoint["model_state_dict"], dict):
        return checkpoint["model_state_dict"]
    if "module" in checkpoint and isinstance(checkpoint["module"], dict):
        return checkpoint["module"]
    if "state_dict" in checkpoint and isinstance(checkpoint["state_dict"], dict):
        return checkpoint["state_dict"]
    if checkpoint and all(isinstance(v, torch.Tensor) for v in checkpoint.values()):
        return checkpoint

    raise ValueError(
        "Unsupported checkpoint format. Expected one of: "
        "model_state_dict / module / state_dict / plain state_dict."
    )


def _load_checkpoint_payload(checkpoint_file: str):
    if checkpoint_file.endswith(".safetensors.index.json"):
        from safetensors.torch import load_file

        with open(checkpoint_file, "r", encoding="utf-8") as handle:
            index = json.load(handle)
        weight_map = index.get("weight_map", None)
        if not isinstance(weight_map, dict) or not weight_map:
            raise ValueError(f"Invalid safetensors index file: {checkpoint_file}")

        base_dir = os.path.dirname(checkpoint_file)
        state_dict = {}
        for shard_name in sorted(set(weight_map.values())):
            shard_path = os.path.join(base_dir, shard_name)
            if not os.path.isfile(shard_path):
                raise FileNotFoundError(f"Missing safetensors shard: {shard_path}")
            state_dict.update(load_file(shard_path))
        return state_dict

    if checkpoint_file.endswith(".safetensors"):
        from safetensors.torch import load_file

        return load_file(checkpoint_file)

    return torch.load(checkpoint_file, map_location="cpu")


def _strip_prefix_if_needed(state_dict: dict, prefix: str) -> dict:
    if state_dict and all(k.startswith(prefix) for k in state_dict):
        return {k[len(prefix) :]: v for k, v in state_dict.items()}
    return state_dict


def _load_transformer_state_dict_strict_shapes(
    transformer,
    checkpoint_path: str,
    *,
    label: str,
) -> None:
    checkpoint_file = _resolve_checkpoint_file(checkpoint_path)
    print(f"Loading {label} checkpoint from {checkpoint_file}")
    payload = _load_checkpoint_payload(checkpoint_file)
    state_dict = _extract_state_dict(payload)
    state_dict = _strip_prefix_if_needed(state_dict, "module.")

    target_state = transformer.state_dict()
    compatible_state = {}
    mismatched = []
    for key, value in state_dict.items():
        if key not in target_state:
            continue
        if tuple(value.shape) != tuple(target_state[key].shape):
            mismatched.append(
                f"{key}: checkpoint={tuple(value.shape)} vs model={tuple(target_state[key].shape)}"
            )
            continue
        compatible_state[key] = value

    if mismatched:
        raise ValueError(f"{label} checkpoint has shape-mismatched keys: {mismatched[:10]}")
    if not compatible_state:
        raise ValueError(
            f"{label} checkpoint has zero loadable parameters for the constructed model: "
            f"{checkpoint_path}"
        )

    missing, unexpected = transformer.load_state_dict(compatible_state, strict=False)
    if missing:
        print(f"  Warning - {label} missing keys: {missing[:10]}")
    if unexpected:
        print(f"  Warning - {label} unexpected keys: {unexpected[:10]}")


class SEFIInferenceRunner:
    """Inference runner for SEFI-T2I with three-phase masked denoising."""

    def __init__(
        self,
        config,
        *,
        checkpoint_path: str = "",
        device: str = "cuda",
        debug_assert_schedule: bool = False,
        delta_t_override: Optional[float] = None,
        inference_dtype: Optional[str] = None,
        transformer_checkpoint_path: Optional[str] = None,
        adapter_path: Optional[str] = None,
        timestep_shift_alpha: float = 1.0,
        autoguidance_config_path: Optional[str] = None,
        autoguidance_checkpoint_path: Optional[str] = None,
        guidance_interval_sigma_lo: Optional[float] = None,
        guidance_interval_sigma_hi: Optional[float] = None,
    ):
        from diffusers.pipelines.flux2.image_processor import Flux2ImageProcessor

        self.config = config
        self.device = torch.device(device)
        self.component_dtype = _resolve_weight_dtype(config)
        self.weight_dtype = _resolve_weight_dtype(config, override=inference_dtype)
        self.transformer_checkpoint_path = ""
        self.adapter_path = ""
        self.adapter_metadata = None
        (
            self.autoguidance_config_path,
            self.autoguidance_checkpoint_path,
        ) = _resolve_autoguidance_paths(
            autoguidance_config_path,
            autoguidance_checkpoint_path,
        )
        self.autoguidance_enabled = bool(self.autoguidance_config_path)
        self.autoguidance_transformer = None
        self.autoguidance_text_encoder = None
        self.autoguidance_reuse_main_text_encoder = True
        (
            self.guidance_interval_sigma_lo,
            self.guidance_interval_sigma_hi,
        ) = _resolve_guidance_interval_sigma(
            guidance_interval_sigma_lo,
            guidance_interval_sigma_hi,
        )
        self.guidance_interval_enabled = self.guidance_interval_sigma_lo is not None

        components = build_components(config, component_dtype=self.component_dtype)
        self.transformer = components.transformer.to(
            device=self.device,
            dtype=self.weight_dtype,
        ).eval()
        self.text_encoder = components.text_encoder.to(
            device=self.device,
            dtype=self.component_dtype,
        ).eval()
        self.texture_codec = components.texture_codec.to(
            device=self.device,
            dtype=self.component_dtype,
        ).eval()
        self.noise_scheduler = components.noise_scheduler
        self.pipeline_cls = components.pipeline_cls
        self.semantic_channels = int(components.semantic_channels)
        self.texture_channels = int(components.texture_channels)
        self.total_channels = int(components.total_channels)

        self.debug_assert_schedule = bool(debug_assert_schedule)
        self.timestep_shift_alpha = float(timestep_shift_alpha)
        if self.timestep_shift_alpha <= 0:
            raise ValueError(
                "timestep_shift_alpha must be > 0. "
                f"Got {self.timestep_shift_alpha}."
            )
        self._configure_delta_t(delta_t_override)
        shift_enabled = self.timestep_shift_alpha != 1.0
        print(
            "Inference timestep schedule: "
            f"timestep_shift_alpha={self.timestep_shift_alpha:.6f}, "
            f"delta_t={self.delta_t:.6f}, shift_enabled={shift_enabled}"
        )
        if self.guidance_interval_enabled:
            print(
                "Limited interval guidance enabled on base sigma: "
                f"({self.guidance_interval_sigma_lo:.6f}, "
                f"{self.guidance_interval_sigma_hi:.6f}]"
            )

        texture_vae_cfg = self.texture_codec.texture_vae.config
        self.vae_scale_factor = 2 ** (len(texture_vae_cfg.block_out_channels) - 1)
        self.image_processor = Flux2ImageProcessor(
            vae_scale_factor=self.vae_scale_factor * 2
        )

        for module in (self.transformer, self.text_encoder, self.texture_codec):
            for param in module.parameters():
                param.requires_grad = False

        if checkpoint_path:
            self.load_checkpoint(checkpoint_path)

        if transformer_checkpoint_path:
            self.load_checkpoint(
                transformer_checkpoint_path,
                label="full transformer override",
            )
            self.transformer_checkpoint_path = str(transformer_checkpoint_path)

        if adapter_path:
            self.load_adapter(adapter_path)

        if self.autoguidance_enabled:
            self._load_autoguidance_model()

    def _configure_delta_t(self, delta_t_override: Optional[float]) -> None:
        sefi_cfg = _training_sefi_cfg(self.config)

        delta_t_min_raw = sefi_cfg.get("delta_t_min", None)
        delta_t_max_raw = sefi_cfg.get("delta_t_max", None)
        if delta_t_min_raw is None or delta_t_max_raw is None:
            raise ValueError("training.sefi.delta_t_min and delta_t_max are required.")

        self.delta_t_min = float(delta_t_min_raw)
        self.delta_t_max = float(delta_t_max_raw)
        if self.delta_t_min < 0 or self.delta_t_min > 1:
            raise ValueError("training.sefi.delta_t_min must be in [0, 1].")
        if self.delta_t_max < 0 or self.delta_t_max > 1:
            raise ValueError("training.sefi.delta_t_max must be in [0, 1].")
        if self.delta_t_min > self.delta_t_max:
            raise ValueError("training.sefi.delta_t_min must be <= delta_t_max.")

        if delta_t_override is None:
            self.delta_t = self.delta_t_max
            print(
                "Warning: --delta-t not provided. "
                f"Using training.sefi.delta_t_max={self.delta_t_max:.6f} for inference."
            )
            return

        self.delta_t = float(delta_t_override)
        if self.delta_t < 0 or self.delta_t > 1:
            raise ValueError("inference delta_t must be in [0, 1].")
        if self.delta_t < self.delta_t_min or self.delta_t > self.delta_t_max:
            print(
                "Warning: inference delta_t is outside training range "
                f"[{self.delta_t_min:.6f}, {self.delta_t_max:.6f}]. "
                f"Got delta_t={self.delta_t:.6f}."
            )

    def load_checkpoint(self, checkpoint_path: str, *, label: str = "checkpoint"):
        ckpt_file = _resolve_checkpoint_file(checkpoint_path)
        print(f"Loading {label} from {ckpt_file}")
        ckpt = _load_checkpoint_payload(ckpt_file)
        state_dict = _extract_state_dict(ckpt)
        state_dict = _strip_prefix_if_needed(state_dict, "module.")

        missing, unexpected = self.transformer.load_state_dict(state_dict, strict=False)
        if missing:
            raise ValueError(f"Checkpoint is missing transformer keys: {missing[:10]}")
        if unexpected:
            raise ValueError(f"Checkpoint has unexpected transformer keys: {unexpected[:10]}")

    def load_adapter(self, adapter_path: str) -> None:
        """Attach a frozen PEFT adapter after loading the complete Base weights."""

        from .training.lora import (
            ADAPTER_METADATA_FILENAME,
            load_lora_adapter,
            read_lora_metadata,
        )

        resolved = _resolve_peft_adapter_directory(adapter_path)
        metadata_path = os.path.join(resolved, ADAPTER_METADATA_FILENAME)
        metadata = read_lora_metadata(resolved) if os.path.isfile(metadata_path) else None
        if metadata is not None:
            adapter_scale = str(metadata.get("scale", "")).strip().lower()
            model_scale = str(_resolve_transformer_scale(self.config)).strip().lower()
            if adapter_scale and adapter_scale != model_scale:
                raise ValueError(
                    "LoRA adapter/model scale mismatch: "
                    f"adapter={adapter_scale}, model={model_scale}."
                )
            target_modules = metadata.get("target_modules", None)
            target_count = metadata.get("target_count", None)
            if target_modules is not None and not isinstance(target_modules, list):
                raise ValueError("SEFI adapter metadata target_modules must be a list.")
            if target_count is not None and target_modules is not None:
                if int(target_count) != len(target_modules):
                    raise ValueError(
                        "SEFI adapter metadata target_count does not match "
                        f"target_modules: {target_count} != {len(target_modules)}."
                    )

        transformer = load_lora_adapter(
            self.transformer,
            resolved,
            is_trainable=False,
        )
        transformer = transformer.to(device=self.device, dtype=self.weight_dtype).eval()
        transformer.requires_grad_(False)

        if metadata is not None and metadata.get("target_modules") is not None:
            injected = set(getattr(transformer, "targeted_module_names", ()))
            expected = {str(name) for name in metadata["target_modules"]}
            if injected != expected:
                missing = sorted(expected - injected)
                unexpected = sorted(injected - expected)
                raise ValueError(
                    "Loaded PEFT targets do not match SEFI adapter metadata: "
                    f"missing={missing[:10]}, unexpected={unexpected[:10]}."
                )

        self.transformer = transformer
        self.adapter_path = resolved
        self.adapter_metadata = metadata
        print(f"Loaded PEFT adapter from {resolved}")

    def _load_autoguidance_model(self) -> None:
        autoguidance_config = load_config(self.autoguidance_config_path)
        self.autoguidance_config = autoguidance_config

        ag_semantic_channels = _derive_semantic_channels(autoguidance_config)
        ag_texture_channels = _derive_texture_channels(autoguidance_config)
        if ag_semantic_channels != self.semantic_channels:
            raise ValueError(
                "AutoGuidance semantic channel mismatch: "
                f"main={self.semantic_channels}, small={ag_semantic_channels}."
            )
        if ag_texture_channels != self.texture_channels:
            raise ValueError(
                "AutoGuidance texture channel mismatch: "
                f"main={self.texture_channels}, small={ag_texture_channels}."
            )

        ag_text_output_dim = _derive_text_output_dim(autoguidance_config)
        autoguidance_transformer = build_lightweight_transformer(
            autoguidance_config,
            total_channels=self.total_channels,
            text_output_dim=ag_text_output_dim,
        )
        _load_transformer_state_dict_strict_shapes(
            autoguidance_transformer,
            self.autoguidance_checkpoint_path,
            label="AutoGuidance",
        )
        self.autoguidance_transformer = autoguidance_transformer.to(
            device=self.device,
            dtype=self.weight_dtype,
        ).eval()
        for param in self.autoguidance_transformer.parameters():
            param.requires_grad = False

        self.autoguidance_reuse_main_text_encoder = (
            text_encoder_signature(self.config) == text_encoder_signature(autoguidance_config)
        )
        if self.autoguidance_reuse_main_text_encoder:
            print("AutoGuidance reuses main prompt embeddings.")
        else:
            text_cfg = autoguidance_config.model.text_encoder
            self.autoguidance_text_encoder = Qwen3VLTextEncoder(
                model_name=str(text_cfg.model_name),
                weights_root=str(text_cfg.get("weights_root", "outputs/model_weights")),
                max_length=int(text_cfg.max_length),
                hidden_layers=[int(x) for x in text_cfg.hidden_layers],
                torch_dtype=self.component_dtype,
            ).to(device=self.device, dtype=self.component_dtype).eval()
            if int(self.autoguidance_text_encoder.output_dim) != int(ag_text_output_dim):
                raise ValueError(
                    "AutoGuidance text encoder output dim mismatch: "
                    f"loaded={self.autoguidance_text_encoder.output_dim}, "
                    f"expected={ag_text_output_dim}."
                )
            for param in self.autoguidance_text_encoder.parameters():
                param.requires_grad = False
            print("AutoGuidance uses a separate small-model text encoder.")

        print(
            "Loaded AutoGuidance model: "
            f"config={self.autoguidance_config_path}, "
            f"checkpoint={self.autoguidance_checkpoint_path}"
        )

    def _timesteps_and_sigmas(
        self,
        u_continuous: Tensor,
        *,
        n_dim: int,
        dtype: torch.dtype,
    ) -> tuple[Tensor, Tensor]:
        num_steps = int(self.noise_scheduler.config.num_train_timesteps)
        indices = (u_continuous * (num_steps - 1)).long().clamp(0, num_steps - 1)

        timesteps = self.noise_scheduler.timesteps[indices.cpu()].to(self.device)
        sigmas = self.noise_scheduler.sigmas[indices.cpu()].to(
            device=self.device,
            dtype=dtype,
        )
        while sigmas.ndim < n_dim:
            sigmas = sigmas.unsqueeze(-1)
        return timesteps, sigmas

    def _assert_shifted_schedule(
        self,
        u_base_unit: Tensor,
        u_sem_raw_schedule: Tensor,
        eps: float = 1e-6,
    ) -> None:
        if u_base_unit.ndim != 1 or u_sem_raw_schedule.ndim != 1:
            raise ValueError("u_base_unit and u_sem_raw_schedule must be 1D tensors.")
        if u_base_unit.shape != u_sem_raw_schedule.shape:
            raise ValueError("u_base_unit and u_sem_raw_schedule must have the same shape.")

        expected_u_max = 1.0 + self.delta_t
        if abs(float(u_base_unit[0].item()) - 0.0) > eps:
            raise ValueError(
                f"Invalid u_base_unit[0], expected 0, got {float(u_base_unit[0].item()):.6f}"
            )
        if abs(float(u_base_unit[-1].item()) - 1.0) > eps:
            raise ValueError(
                f"Invalid u_base_unit[-1], expected 1, got {float(u_base_unit[-1].item()):.6f}"
            )
        if abs(float(u_sem_raw_schedule[0].item()) - 0.0) > eps:
            raise ValueError(
                "Invalid shifted schedule start, expected 0, "
                f"got {float(u_sem_raw_schedule[0].item()):.6f}"
            )
        if abs(float(u_sem_raw_schedule[-1].item()) - expected_u_max) > eps:
            raise ValueError(
                "Invalid shifted schedule end, expected 1+delta_t, "
                f"got {float(u_sem_raw_schedule[-1].item()):.6f}, "
                f"expected={expected_u_max:.6f}"
            )

        diffs = u_sem_raw_schedule[1:] - u_sem_raw_schedule[:-1]
        if torch.any(diffs < -eps):
            index = int(torch.nonzero(diffs < -eps, as_tuple=False)[0, 0].item())
            raise ValueError(
                "Shifted u_sem_raw schedule must be monotonic non-decreasing, "
                f"but got decrease at step {index}: "
                f"{float(u_sem_raw_schedule[index].item()):.6f} -> "
                f"{float(u_sem_raw_schedule[index + 1].item()):.6f}"
            )

    def _assert_dual_time_invariants(
        self,
        u_sem: Tensor,
        u_tex: Tensor,
        sigmas_sem: Tensor,
        sigmas_tex: Tensor,
        eps: float = 1e-6,
    ) -> None:
        u_violation = u_sem < u_tex
        if torch.any(u_violation):
            index = int(torch.nonzero(u_violation, as_tuple=False)[0, 0].item())
            raise ValueError(
                "Dual-time invariant violated: expected u_sem >= u_tex, got "
                f"u_sem[{index}]={float(u_sem[index].item()):.6f}, "
                f"u_tex[{index}]={float(u_tex[index].item()):.6f}."
            )

        sigma_violation = sigmas_sem > (sigmas_tex + eps)
        if torch.any(sigma_violation):
            index = int(torch.nonzero(sigma_violation, as_tuple=False)[0, 0].item())
            sigma_sem_flat = sigmas_sem.reshape(sigmas_sem.shape[0], -1)
            sigma_tex_flat = sigmas_tex.reshape(sigmas_tex.shape[0], -1)
            raise ValueError(
                "Dual-time invariant violated: expected sigmas_sem <= sigmas_tex, got "
                f"sigmas_sem[{index}]={float(sigma_sem_flat[index, 0].item()):.6f}, "
                f"sigmas_tex[{index}]={float(sigma_tex_flat[index, 0].item()):.6f}."
            )

    def _prepare_latents(
        self,
        *,
        batch_size: int,
        height: int,
        width: int,
        generator: Optional[torch.Generator],
    ) -> tuple[Tensor, Tensor, int, int]:
        height = 2 * (int(height) // (self.vae_scale_factor * 2))
        width = 2 * (int(width) // (self.vae_scale_factor * 2))

        latents = torch.randn(
            (batch_size, self.total_channels, height // 2, width // 2),
            generator=generator,
            device=self.device,
            dtype=self.weight_dtype,
        )
        latent_ids = self.pipeline_cls._prepare_latent_ids(latents).to(self.device)
        return latents, latent_ids, height, width

    def _predict_velocity(
        self,
        transformer,
        *,
        packed_latents: Tensor,
        timesteps_sem: Tensor,
        timesteps_tex: Tensor,
        encoder_hidden_states: Tensor,
        txt_ids: Tensor,
        img_ids: Tensor,
    ) -> Tensor:
        pred = transformer(
            hidden_states=packed_latents,
            timestep_sem=timesteps_sem / 1000,
            timestep_tex=timesteps_tex / 1000,
            encoder_hidden_states=encoder_hidden_states,
            txt_ids=txt_ids,
            img_ids=img_ids,
        )
        pred = pred[:, : packed_latents.size(1)]
        return self.pipeline_cls._unpack_latents_with_ids(pred, img_ids)

    @torch.no_grad()
    def generate_batch(
        self,
        *,
        prompts: list[str],
        num_inference_steps: int,
        guidance_scale: float,
        height: int,
        width: int,
        generator: Optional[torch.Generator] = None,
    ) -> list[Image.Image]:
        if num_inference_steps <= 0:
            raise ValueError("num_inference_steps must be > 0")

        batch_size = len(prompts)
        if batch_size == 0:
            return []

        prompt_embeds, text_ids = self.text_encoder.encode(prompts, dtype=self.weight_dtype)
        _validate_autoguidance_guidance_scale(
            self.autoguidance_enabled,
            guidance_scale,
        )

        if self.autoguidance_enabled:
            if self.autoguidance_reuse_main_text_encoder:
                autoguidance_prompt_embeds = prompt_embeds
                autoguidance_text_ids = text_ids
            else:
                autoguidance_prompt_embeds, autoguidance_text_ids = (
                    self.autoguidance_text_encoder.encode(
                        prompts,
                        dtype=self.weight_dtype,
                    )
                )
            neg_prompt_embeds = None
            neg_text_ids = None
        elif guidance_scale > 1.0:
            neg_prompts = [""] * batch_size
            neg_prompt_embeds, neg_text_ids = self.text_encoder.encode(
                neg_prompts,
                dtype=self.weight_dtype,
            )
            autoguidance_prompt_embeds = None
            autoguidance_text_ids = None
        else:
            autoguidance_prompt_embeds = None
            autoguidance_text_ids = None
            neg_prompt_embeds = None
            neg_text_ids = None

        latents, latent_ids, _, _ = self._prepare_latents(
            batch_size=batch_size,
            height=height,
            width=width,
            generator=generator,
        )

        u_base_unit = torch.linspace(
            0.0,
            1.0,
            steps=num_inference_steps + 1,
            device=self.device,
            dtype=torch.float32,
        )
        u_shifted_unit = _apply_timestep_shift_unit_interval(
            u_base_unit,
            self.timestep_shift_alpha,
        )
        _, base_sigmas_schedule = self._timesteps_and_sigmas(
            u_shifted_unit,
            n_dim=1,
            dtype=torch.float32,
        )
        u_sem_raw_schedule = u_shifted_unit * (1.0 + self.delta_t)
        if self.debug_assert_schedule:
            self._assert_shifted_schedule(
                u_base_unit=u_base_unit,
                u_sem_raw_schedule=u_sem_raw_schedule,
            )

        for step in range(num_inference_steps):
            u_sem_raw_cur = torch.full(
                (batch_size,),
                float(u_sem_raw_schedule[step].item()),
                device=self.device,
            )
            u_sem_raw_next = torch.full(
                (batch_size,),
                float(u_sem_raw_schedule[step + 1].item()),
                device=self.device,
            )

            u_tex_cur = torch.clamp(u_sem_raw_cur - self.delta_t, min=0.0, max=1.0)
            u_sem_cur = torch.clamp(u_sem_raw_cur, max=1.0)
            u_tex_next = torch.clamp(u_sem_raw_next - self.delta_t, min=0.0, max=1.0)
            u_sem_next = torch.clamp(u_sem_raw_next, max=1.0)

            timesteps_sem_cur, sigmas_sem_cur = self._timesteps_and_sigmas(
                u_sem_cur,
                n_dim=latents.ndim,
                dtype=latents.dtype,
            )
            timesteps_tex_cur, sigmas_tex_cur = self._timesteps_and_sigmas(
                u_tex_cur,
                n_dim=latents.ndim,
                dtype=latents.dtype,
            )
            _, sigmas_sem_next = self._timesteps_and_sigmas(
                u_sem_next,
                n_dim=latents.ndim,
                dtype=latents.dtype,
            )
            _, sigmas_tex_next = self._timesteps_and_sigmas(
                u_tex_next,
                n_dim=latents.ndim,
                dtype=latents.dtype,
            )
            if self.debug_assert_schedule:
                self._assert_dual_time_invariants(
                    u_sem_cur,
                    u_tex_cur,
                    sigmas_sem_cur,
                    sigmas_tex_cur,
                )

            guidance_active = _guidance_interval_is_active(
                base_sigmas_schedule[step],
                self.guidance_interval_sigma_lo,
                self.guidance_interval_sigma_hi,
            )
            packed_latents = self.pipeline_cls._pack_latents(latents)
            pred_cond = self._predict_velocity(
                self.transformer,
                packed_latents=packed_latents,
                timesteps_sem=timesteps_sem_cur,
                timesteps_tex=timesteps_tex_cur,
                encoder_hidden_states=prompt_embeds,
                txt_ids=text_ids,
                img_ids=latent_ids,
            )

            if not guidance_active:
                velocity = pred_cond
            elif self.autoguidance_enabled:
                pred_base = self._predict_velocity(
                    self.autoguidance_transformer,
                    packed_latents=packed_latents,
                    timesteps_sem=timesteps_sem_cur,
                    timesteps_tex=timesteps_tex_cur,
                    encoder_hidden_states=autoguidance_prompt_embeds,
                    txt_ids=autoguidance_text_ids,
                    img_ids=latent_ids,
                )
                velocity = _combine_guided_velocity(
                    pred_base,
                    pred_cond,
                    guidance_scale,
                )
            elif guidance_scale > 1.0:
                pred_uncond = self._predict_velocity(
                    self.transformer,
                    packed_latents=packed_latents,
                    timesteps_sem=timesteps_sem_cur,
                    timesteps_tex=timesteps_tex_cur,
                    encoder_hidden_states=neg_prompt_embeds,
                    txt_ids=neg_text_ids,
                    img_ids=latent_ids,
                )
                velocity = _combine_guided_velocity(
                    pred_uncond,
                    pred_cond,
                    guidance_scale,
                )
            else:
                velocity = pred_cond

            vel_sem = velocity[:, : self.semantic_channels]
            vel_tex = velocity[:, self.semantic_channels :]
            lat_sem = latents[:, : self.semantic_channels]
            lat_tex = latents[:, self.semantic_channels :]

            dt_sem = sigmas_sem_next - sigmas_sem_cur
            dt_tex = sigmas_tex_next - sigmas_tex_cur

            lat_sem = lat_sem + dt_sem * vel_sem
            lat_tex = lat_tex + dt_tex * vel_tex
            latents = torch.cat([lat_sem, lat_tex], dim=1)

        texture_latents = latents[:, self.semantic_channels :]
        decoded = self.texture_codec.decode_texture(
            texture_latents.to(dtype=self.component_dtype),
            pipeline_cls=self.pipeline_cls,
        )
        return self.image_processor.postprocess(decoded, output_type="pil")
