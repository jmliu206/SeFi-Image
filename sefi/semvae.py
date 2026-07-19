"""Public loading and feature-compression API for SeFi SemVAE."""

from __future__ import annotations

import gc
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torch import Tensor

from .checkpoints import ensure_local_path
from .modeling.semvae import SemVAEConfig, SemanticVariationalAutoEncoder


DEFAULT_SEMVAE_CHECKPOINT = "SeFi-Image/SeFi-Image-SemVAE"
DEFAULT_SEMVAE_VARIANT = "dinov2_vitl14_reg/transformer_ch16"
DEFAULT_DINO_CHECKPOINT = "facebook/dinov2-with-registers-large"
SEMVAE_CHECKPOINT_FILENAME = "checkpoint_01000000.pt"
LATENT_STATS_FILENAME = "latent_stats.pt"


def _resolve_device(device: str | torch.device | None) -> torch.device:
    if device is None or str(device) == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    resolved = torch.device(device)
    if resolved.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested, but torch.cuda.is_available() is false.")
    return resolved


def _safe_torch_load(path: Path) -> Any:
    """Load tensor-only artifacts without enabling arbitrary pickle globals."""

    kwargs = {"map_location": "cpu", "weights_only": True}
    try:
        return torch.load(path, mmap=True, **kwargs)
    except TypeError:
        # ``mmap`` is unavailable in older supported PyTorch releases.
        return torch.load(path, **kwargs)


def _extract_state_dict(payload: Any) -> dict[str, Tensor]:
    if isinstance(payload, Mapping):
        for key in ("model_state_dict", "state_dict", "module"):
            candidate = payload.get(key)
            if isinstance(candidate, Mapping):
                state_dict = dict(candidate)
                break
        else:
            if payload and all(torch.is_tensor(value) for value in payload.values()):
                state_dict = dict(payload)
            else:
                raise ValueError(
                    "Unsupported SemVAE checkpoint. Expected model_state_dict, "
                    "state_dict, module, or a plain tensor state dict."
                )
    else:
        raise ValueError(f"Unsupported SemVAE checkpoint payload: {type(payload).__name__}")

    if state_dict and all(key.startswith("module.") for key in state_dict):
        state_dict = {key[len("module.") :]: value for key, value in state_dict.items()}
    return state_dict


def _resolve_artifact_paths(
    checkpoint: str,
    *,
    variant: str,
    cache_dir: str | Path,
) -> tuple[Path, Path, Path]:
    if variant != DEFAULT_SEMVAE_VARIANT:
        raise ValueError(
            "This release supports only the public SemVAE variant "
            f"'{DEFAULT_SEMVAE_VARIANT}', got '{variant}'."
        )
    # Keep the user-visible path instead of resolving file symlinks into a Hub
    # blob directory; direct checkpoint paths still need their sibling stats.
    local = Path(ensure_local_path(checkpoint, cache_dir=cache_dir)).absolute()

    if local.is_file():
        checkpoint_path = local
        variant_root = local.parent.parent if local.parent.name == "checkpoints" else local.parent
    else:
        requested_variant = Path(variant)
        if requested_variant.is_absolute() or ".." in requested_variant.parts:
            raise ValueError(f"SemVAE variant must be a safe relative path, got: {variant}")
        nested_variant = local / requested_variant
        variant_root = nested_variant if nested_variant.is_dir() else local
        checkpoint_candidates = (
            variant_root / "checkpoints" / SEMVAE_CHECKPOINT_FILENAME,
            variant_root / SEMVAE_CHECKPOINT_FILENAME,
        )
        checkpoint_path = next(
            (candidate for candidate in checkpoint_candidates if candidate.is_file()),
            checkpoint_candidates[0],
        )

    stats_path = variant_root / LATENT_STATS_FILENAME
    if not checkpoint_path.is_file():
        raise FileNotFoundError(
            f"SemVAE checkpoint file not found: {checkpoint_path}. "
            f"Expected variant '{variant}' under {local}."
        )
    if not stats_path.is_file():
        raise FileNotFoundError(f"SemVAE latent statistics not found: {stats_path}")
    return checkpoint_path, stats_path, variant_root


def _load_latent_stats(path: Path, bottleneck_dim: int) -> tuple[Tensor, Tensor]:
    payload = _safe_torch_load(path)
    if not isinstance(payload, Mapping):
        raise ValueError(f"Expected a mapping in latent stats file: {path}")
    mean = payload.get("mean")
    if mean is None:
        mean = payload.get("mean_broadcast")
    std = payload.get("std")
    if std is None:
        std = payload.get("std_broadcast")
    if not torch.is_tensor(mean) or not torch.is_tensor(std):
        raise ValueError(f"Latent stats must contain tensor mean/std entries: {path}")

    mean = mean.float().reshape(1, 1, -1)
    std = std.float().reshape(1, 1, -1)
    if mean.shape[-1] != bottleneck_dim or std.shape[-1] != bottleneck_dim:
        raise ValueError(
            "SemVAE latent stats channel mismatch: "
            f"mean={tuple(mean.shape)}, std={tuple(std.shape)}, expected={bottleneck_dim}."
        )
    if not torch.isfinite(mean).all() or not torch.isfinite(std).all():
        raise ValueError("SemVAE latent statistics contain NaN or Inf values.")
    if torch.any(std <= 0):
        raise ValueError("SemVAE latent standard deviation must be positive.")
    return mean, std


class DINOv2FeatureExtractor(nn.Module):
    """Extract DINOv2-with-registers patch tokens using training-time preprocessing."""

    patch_size = 14
    feature_dim = 1024

    def __init__(
        self,
        checkpoint: str = DEFAULT_DINO_CHECKPOINT,
        *,
        device: str | torch.device | None = None,
        cache_dir: str | Path | None = None,
        image_size: int = 256,
    ):
        super().__init__()
        try:
            from transformers import AutoImageProcessor, AutoModel
        except ImportError as exc:
            raise RuntimeError(
                "DINOv2 loading requires transformers. Install the dependencies "
                "listed in the repository README."
            ) from exc

        checkpoint_path = Path(checkpoint).expanduser()
        self.checkpoint = str(checkpoint_path) if checkpoint_path.exists() else str(checkpoint)
        self.device = _resolve_device(device)
        self.image_size = int(image_size)
        if self.image_size <= 0:
            raise ValueError(f"image_size must be positive, got {self.image_size}.")
        self.dino_size = self.image_size * 7 // 8
        if self.dino_size % self.patch_size != 0:
            raise ValueError(
                f"DINO input size {self.dino_size} must be divisible by patch size "
                f"{self.patch_size}."
            )

        local_only = Path(self.checkpoint).expanduser().exists()
        load_kwargs: dict[str, Any] = {"local_files_only": local_only}
        if cache_dir is not None:
            load_kwargs["cache_dir"] = str(cache_dir)
        self.processor = AutoImageProcessor.from_pretrained(
            self.checkpoint,
            use_fast=False,
            **load_kwargs,
        )
        self.model = AutoModel.from_pretrained(
            self.checkpoint,
            **load_kwargs,
        )
        self.model.eval().to(self.device, dtype=torch.float32)
        self.model.requires_grad_(False)

        configured_dim = int(getattr(self.model.config, "hidden_size", -1))
        if configured_dim != self.feature_dim:
            raise ValueError(
                f"Expected DINOv2-L hidden size {self.feature_dim}, got {configured_dim}."
            )
        register_tokens = getattr(self.model.config, "num_register_tokens", None)
        if register_tokens is None or int(register_tokens) <= 0:
            raise ValueError(
                "Expected a DINOv2-with-registers checkpoint with num_register_tokens."
            )
        self.num_register_tokens = int(register_tokens)

        mean = getattr(self.processor, "image_mean", None)
        std = getattr(self.processor, "image_std", None)
        if mean is None or std is None or len(mean) != 3 or len(std) != 3:
            raise ValueError("DINOv2 processor must provide three-channel image_mean/image_std.")
        self.register_buffer(
            "image_mean",
            torch.tensor(mean, dtype=torch.float32).reshape(1, 3, 1, 1),
            persistent=False,
        )
        self.register_buffer(
            "image_std",
            torch.tensor(std, dtype=torch.float32).reshape(1, 3, 1, 1),
            persistent=False,
        )

    def preprocess(self, image: Image.Image) -> Tensor:
        """Preprocess one PIL image using the legacy 256px-compatible path."""

        return self.preprocess_batch([image])[0]

    def preprocess_batch(
        self,
        images: Sequence[Image.Image] | Tensor,
        *,
        input_range: str = "zero_one",
    ) -> Tensor:
        """Preprocess a PIL sequence or an RGB tensor batch for DINOv2.

        Tensor input accepts ``[C,H,W]`` or ``[B,C,H,W]``. ``input_range`` must
        explicitly describe whether it is in ``[0,1]`` or ``[-1,1]``; the
        latter matches the shared DiT texture input. Both paths apply the same
        resize-shortest-side, center-crop framing before resizing to the DINO
        patch grid.
        """

        try:
            from torchvision import transforms
            from torchvision.transforms import functional as tvf
        except ImportError as exc:
            raise RuntimeError("DINOv2 preprocessing requires torchvision.") from exc

        if torch.is_tensor(images):
            pixel_values = images
            if pixel_values.ndim == 3:
                pixel_values = pixel_values.unsqueeze(0)
            if pixel_values.ndim != 4 or pixel_values.shape[1] != 3:
                raise ValueError(
                    "Tensor images must have shape [C,H,W] or [B,C,H,W] with C=3, "
                    f"got {tuple(pixel_values.shape)}."
                )
            pixel_values = pixel_values.float()
            if not torch.isfinite(pixel_values).all():
                raise ValueError("Tensor images contain NaN or Inf values.")
            if input_range == "minus_one_one":
                pixel_values = (pixel_values + 1.0) / 2.0
            elif input_range != "zero_one":
                raise ValueError(
                    "input_range must be 'zero_one' or 'minus_one_one', "
                    f"got {input_range!r}."
                )
            if pixel_values.numel() and (
                pixel_values.min().item() < -1e-4 or pixel_values.max().item() > 1.0001
            ):
                raise ValueError(
                    f"Tensor values do not match input_range={input_range!r}; "
                    f"observed [{pixel_values.min().item():.4f}, "
                    f"{pixel_values.max().item():.4f}]."
                )
            pixel_values = tvf.resize(
                pixel_values,
                self.image_size,
                interpolation=transforms.InterpolationMode.BICUBIC,
                antialias=True,
            )
            pixel_values = tvf.center_crop(
                pixel_values,
                [self.image_size, self.image_size],
            )
            pixel_values = tvf.resize(
                pixel_values,
                [self.dino_size, self.dino_size],
                interpolation=transforms.InterpolationMode.BICUBIC,
                antialias=True,
            )
            mean = self.image_mean.to(pixel_values.device, pixel_values.dtype)
            std = self.image_std.to(pixel_values.device, pixel_values.dtype)
            return (pixel_values - mean) / std

        image_list = list(images)
        if not image_list:
            raise ValueError("At least one image is required for batch preprocessing.")
        if not all(isinstance(image, Image.Image) for image in image_list):
            raise TypeError("PIL batch preprocessing requires only PIL.Image.Image values.")
        mean = self.image_mean.flatten().tolist()
        std = self.image_std.flatten().tolist()
        transform = transforms.Compose(
            [
                transforms.Resize(
                    self.image_size,
                    interpolation=transforms.InterpolationMode.BICUBIC,
                ),
                transforms.CenterCrop(self.image_size),
                transforms.Resize(
                    (self.dino_size, self.dino_size),
                    interpolation=transforms.InterpolationMode.BICUBIC,
                ),
                transforms.ToTensor(),
                transforms.Normalize(mean=mean, std=std),
            ]
        )
        return torch.stack([transform(image.convert("RGB")) for image in image_list])

    @torch.no_grad()
    def forward(self, pixel_values: Tensor) -> Tensor:
        pixel_values = pixel_values.to(self.device, dtype=torch.float32)
        if pixel_values.ndim != 4 or pixel_values.shape[1] != 3:
            raise ValueError(
                f"DINO pixel values must be [B,3,H,W], got {tuple(pixel_values.shape)}."
            )
        input_h, input_w = (int(value) for value in pixel_values.shape[-2:])
        if input_h % self.patch_size or input_w % self.patch_size:
            raise ValueError(
                f"DINO input {(input_h, input_w)} must be divisible by {self.patch_size}."
            )
        output = self.model(pixel_values=pixel_values, return_dict=True)
        prefix_tokens = 1 + self.num_register_tokens
        patch_tokens = output.last_hidden_state[:, prefix_tokens:, :]
        expected_tokens = (input_h // self.patch_size) * (input_w // self.patch_size)
        if patch_tokens.shape[1] != expected_tokens:
            raise ValueError(
                "Unexpected DINO patch-token count after removing CLS/register tokens: "
                f"got={patch_tokens.shape[1]}, expected={expected_tokens}."
            )
        return patch_tokens.float()


@dataclass
class SemVAEOutput:
    """Tensors produced by one deterministic SemVAE compression pass."""

    features: Tensor
    latents: Tensor
    normalized_latents: Tensor
    reconstruction: Tensor
    token_cosine: Tensor


@dataclass
class SemVAEEncodedBatch:
    """Training-oriented batch output without reconstruction overhead."""

    features: Tensor
    latents: Tensor
    normalized_latents: Tensor


class SemVAEFeatureCodec:
    """Load the public SemVAE and expose feature extraction/compression methods."""

    def __init__(
        self,
        *,
        feature_extractor: DINOv2FeatureExtractor,
        semvae: SemanticVariationalAutoEncoder,
        latent_mean: Tensor,
        latent_std: Tensor,
        checkpoint_path: Path,
        stats_path: Path,
    ):
        self.feature_extractor = feature_extractor
        self.semvae = semvae
        self.device = feature_extractor.device
        self.latent_mean = latent_mean.to(self.device)
        self.latent_std = latent_std.to(self.device)
        self.checkpoint_path = checkpoint_path
        self.stats_path = stats_path

    @classmethod
    def from_pretrained(
        cls,
        checkpoint: str = DEFAULT_SEMVAE_CHECKPOINT,
        *,
        variant: str = DEFAULT_SEMVAE_VARIANT,
        vfm_checkpoint: str = DEFAULT_DINO_CHECKPOINT,
        cache_dir: str | Path = "outputs/model_weights/semvae",
        device: str | torch.device | None = None,
        image_size: int = 256,
    ) -> "SemVAEFeatureCodec":
        resolved_device = _resolve_device(device)
        checkpoint_path, stats_path, _ = _resolve_artifact_paths(
            checkpoint,
            variant=variant,
            cache_dir=cache_dir,
        )
        config = SemVAEConfig()
        semvae = SemanticVariationalAutoEncoder(config)
        payload = _safe_torch_load(checkpoint_path)
        state_dict = _extract_state_dict(payload)
        del payload
        semvae.load_state_dict(state_dict, strict=True)
        del state_dict
        gc.collect()
        semvae.eval().to(resolved_device)
        semvae.requires_grad_(False)

        latent_mean, latent_std = _load_latent_stats(
            stats_path,
            bottleneck_dim=config.bottleneck_dim,
        )
        feature_extractor = DINOv2FeatureExtractor(
            vfm_checkpoint,
            device=resolved_device,
            cache_dir=cache_dir,
            image_size=image_size,
        )
        return cls(
            feature_extractor=feature_extractor,
            semvae=semvae,
            latent_mean=latent_mean,
            latent_std=latent_std,
            checkpoint_path=checkpoint_path,
            stats_path=stats_path,
        )

    @property
    def compression_ratio(self) -> float:
        return self.semvae.input_dim / self.semvae.bottleneck_dim

    @property
    def image_size(self) -> int:
        return self.feature_extractor.image_size

    @property
    def dino_size(self) -> int:
        return self.feature_extractor.dino_size

    def preprocess_batch(
        self,
        images: Sequence[Image.Image] | Tensor,
        *,
        input_range: str = "zero_one",
    ) -> Tensor:
        """Prepare a PIL/tensor image batch for the frozen DINO encoder."""

        return self.feature_extractor.preprocess_batch(images, input_range=input_range)

    @torch.no_grad()
    def extract_features(self, pixel_values: Tensor) -> Tensor:
        return self.feature_extractor(pixel_values)

    @torch.no_grad()
    def compress_features(self, features: Tensor, *, sample: bool = False) -> Tensor:
        features = features.to(self.device, dtype=torch.float32)
        if features.shape[-1] != self.semvae.input_dim:
            raise ValueError(
                f"SemVAE expected feature width {self.semvae.input_dim}, "
                f"got {features.shape[-1]}."
            )
        posterior = self.semvae.posterior(features)
        return posterior.sample() if sample else posterior.mode()

    def normalize_latents(self, latents: Tensor) -> Tensor:
        mean = self.latent_mean.to(latents.device, latents.dtype)
        std = self.latent_std.to(latents.device, latents.dtype)
        return (latents - mean) / std

    def denormalize_latents(self, normalized_latents: Tensor) -> Tensor:
        mean = self.latent_mean.to(normalized_latents.device, normalized_latents.dtype)
        std = self.latent_std.to(normalized_latents.device, normalized_latents.dtype)
        return normalized_latents * std + mean

    @torch.no_grad()
    def encode_batch(self, pixel_values: Tensor, *, sample: bool = False) -> SemVAEEncodedBatch:
        """Extract, compress, and normalize a preprocessed DINO tensor batch."""

        features = self.extract_features(pixel_values)
        latents = self.compress_features(features, sample=sample)
        normalized_latents = self.normalize_latents(latents)
        tensors = (features, latents, normalized_latents)
        if not all(torch.isfinite(tensor).all() for tensor in tensors):
            raise ValueError("SemVAE encoded batch contains NaN or Inf values.")
        return SemVAEEncodedBatch(
            features=features,
            latents=latents,
            normalized_latents=normalized_latents,
        )

    @torch.no_grad()
    def encode_images(
        self,
        images: Sequence[Image.Image] | Tensor,
        *,
        input_range: str = "zero_one",
        sample: bool = False,
    ) -> SemVAEEncodedBatch:
        """Preprocess and encode an image batch without decoding features."""

        pixel_values = self.preprocess_batch(images, input_range=input_range)
        return self.encode_batch(pixel_values, sample=sample)

    @torch.no_grad()
    def decompress_latents(self, latents: Tensor, *, normalized: bool = False) -> Tensor:
        latents = latents.to(self.device, dtype=torch.float32)
        if normalized:
            latents = self.denormalize_latents(latents)
        return self.semvae.decode(latents)

    @torch.no_grad()
    def encode_image(self, image: Image.Image) -> SemVAEOutput:
        pixel_values = self.feature_extractor.preprocess(image).unsqueeze(0)
        encoded = self.encode_batch(pixel_values, sample=False)
        features = encoded.features
        latents = encoded.latents
        reconstruction = self.decompress_latents(latents)
        normalized_latents = encoded.normalized_latents
        token_cosine = F.cosine_similarity(
            reconstruction.float(),
            features.float(),
            dim=-1,
        )
        tensors = (features, latents, normalized_latents, reconstruction, token_cosine)
        if not all(torch.isfinite(tensor).all() for tensor in tensors):
            raise ValueError("SemVAE output contains NaN or Inf values.")
        return SemVAEOutput(
            features=features,
            latents=latents,
            normalized_latents=normalized_latents,
            reconstruction=reconstruction,
            token_cosine=token_cosine,
        )
