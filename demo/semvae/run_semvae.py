#!/usr/bin/env python3
"""Extract DINO features, compress them with SemVAE, and validate reconstruction."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from PIL import Image


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from sefi import SemVAEFeatureCodec  # noqa: E402
from sefi.semvae import (  # noqa: E402
    DEFAULT_DINO_CHECKPOINT,
    DEFAULT_SEMVAE_CHECKPOINT,
    DEFAULT_SEMVAE_VARIANT,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compress DINOv2 patch features with the public SeFi SemVAE and "
            "report reconstruction cosine similarity."
        )
    )
    parser.add_argument(
        "--checkpoint",
        default=DEFAULT_SEMVAE_CHECKPOINT,
        help="Hugging Face repo id, downloaded repo root, variant directory, or .pt file.",
    )
    parser.add_argument(
        "--variant",
        default=DEFAULT_SEMVAE_VARIANT,
        help="Supported variant directory inside the SemVAE repository.",
    )
    parser.add_argument(
        "--vfm-checkpoint",
        default=DEFAULT_DINO_CHECKPOINT,
        help="DINOv2-with-registers Hugging Face repo id or local directory.",
    )
    parser.add_argument("--image", required=True, help="Input image path.")
    parser.add_argument(
        "--output-dir",
        default="outputs/demo/semvae",
        help="Directory for metrics and optional latent tensors.",
    )
    parser.add_argument(
        "--cache-dir",
        default="outputs/model_weights/semvae",
        help="Hugging Face cache directory.",
    )
    parser.add_argument(
        "--device",
        default="auto",
        help="Torch device, for example auto, cuda, cuda:0, or cpu.",
    )
    parser.add_argument(
        "--min-cosine",
        type=float,
        default=0.80,
        help="Minimum mean token cosine required for a passing smoke test.",
    )
    parser.add_argument(
        "--no-save-latents",
        action="store_true",
        help="Do not save raw and DiT-normalized semantic latents.",
    )
    return parser.parse_args()


def _validate_shapes(features: torch.Tensor, latents: torch.Tensor, reconstruction: torch.Tensor):
    if tuple(reconstruction.shape) != tuple(features.shape):
        raise ValueError(
            f"Reconstruction shape {tuple(reconstruction.shape)} does not match "
            f"feature shape {tuple(features.shape)}."
        )
    if features.shape[:2] != latents.shape[:2]:
        raise ValueError(
            "SemVAE must preserve batch/token dimensions: "
            f"features={tuple(features.shape)}, latents={tuple(latents.shape)}."
        )
    if latents.shape[-1] >= features.shape[-1]:
        raise ValueError(
            "Semantic latent did not compress the feature dimension: "
            f"features={tuple(features.shape)}, latents={tuple(latents.shape)}."
        )


def main() -> None:
    args = parse_args()
    image_path = Path(args.image).expanduser()
    if not image_path.is_file():
        raise FileNotFoundError(f"Input image not found: {image_path}")
    output_dir = Path(args.output_dir).expanduser()
    output_dir.mkdir(parents=True, exist_ok=True)

    codec = SemVAEFeatureCodec.from_pretrained(
        args.checkpoint,
        variant=args.variant,
        vfm_checkpoint=args.vfm_checkpoint,
        cache_dir=args.cache_dir,
        device=args.device,
    )
    with Image.open(image_path) as image:
        original_size = list(image.size)
        result = codec.encode_image(image)

    _validate_shapes(result.features, result.latents, result.reconstruction)
    cosine = result.token_cosine.float()
    cosine_mean = float(cosine.mean().item())
    cosine_std = float(cosine.std(unbiased=False).item())
    reconstruction_mse = float(
        F.mse_loss(result.reconstruction.float(), result.features.float()).item()
    )
    passed = cosine_mean >= float(args.min_cosine)

    stem = image_path.stem
    metrics_path = output_dir / f"{stem}_metrics.json"
    latent_path = output_dir / f"{stem}_semantic_latents.pt"
    metrics = {
        "status": "pass" if passed else "fail",
        "image": str(image_path),
        "original_size_wh": original_size,
        "checkpoint": args.checkpoint,
        "checkpoint_file": str(codec.checkpoint_path),
        "vfm_checkpoint": args.vfm_checkpoint,
        "feature_shape": list(result.features.shape),
        "latent_shape": list(result.latents.shape),
        "normalized_latent_shape": list(result.normalized_latents.shape),
        "reconstruction_shape": list(result.reconstruction.shape),
        "feature_dimension_compression_ratio": codec.compression_ratio,
        "token_cosine_mean": cosine_mean,
        "token_cosine_std": cosine_std,
        "token_cosine_min": float(cosine.min().item()),
        "reconstruction_mse": reconstruction_mse,
        "minimum_cosine": float(args.min_cosine),
        "posterior": "mode",
    }
    metrics_path.write_text(
        json.dumps(metrics, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    if not args.no_save_latents:
        torch.save(
            {
                "raw_latents": result.latents.detach().cpu(),
                "normalized_latents": result.normalized_latents.detach().cpu(),
                "image": str(image_path),
                "posterior": "mode",
            },
            latent_path,
        )

    print(json.dumps(metrics, indent=2, sort_keys=True))
    print(f"Metrics: {metrics_path}")
    if not args.no_save_latents:
        print(f"Latents: {latent_path}")
    if not passed:
        raise SystemExit(
            f"SemVAE smoke test failed: mean cosine {cosine_mean:.6f} "
            f"< minimum {args.min_cosine:.6f}."
        )
    print(f"PASS: mean token cosine {cosine_mean:.6f} >= {args.min_cosine:.6f}")


if __name__ == "__main__":
    main()
