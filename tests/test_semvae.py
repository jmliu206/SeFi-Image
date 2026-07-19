import inspect
import subprocess
import sys
import textwrap
from pathlib import Path
from types import SimpleNamespace

import torch
import torch.nn as nn
from PIL import Image

from sefi.modeling.semvae import SemVAEConfig, SemanticVariationalAutoEncoder
from sefi.semvae import (
    DEFAULT_SEMVAE_VARIANT,
    DINOv2FeatureExtractor,
    SemVAEFeatureCodec,
    _extract_state_dict,
    _load_latent_stats,
    _resolve_artifact_paths,
)


def test_semvae_import_does_not_load_unrelated_model_stacks():
    script = textwrap.dedent(
        """
        import builtins

        real_import = builtins.__import__

        def blocked(name, *args, **kwargs):
            if name.split('.')[0] in {'diffusers', 'transformers'}:
                raise ImportError(f'blocked optional dependency: {name}')
            return real_import(name, *args, **kwargs)

        builtins.__import__ = blocked

        import sefi
        from sefi import SemVAEFeatureCodec
        from sefi.modeling import SemVAEConfig, SemanticVariationalAutoEncoder

        assert sefi.__all__ == [
            'ModelSpec',
            'SEFIInferencePipeline',
            'infer_model_spec',
        ]
        assert SemVAEFeatureCodec.__name__ == 'SemVAEFeatureCodec'
        assert SemVAEConfig.__name__ == 'SemVAEConfig'
        assert SemanticVariationalAutoEncoder.__name__ == 'SemanticVariationalAutoEncoder'
        """
    )

    subprocess.run(
        [sys.executable, "-c", script],
        cwd=Path(__file__).resolve().parents[1],
        check=True,
    )


def test_semvae_mode_is_deterministic_and_compresses_feature_width():
    config = SemVAEConfig(
        input_dim=32,
        bottleneck_dim=4,
        hidden_dim=16,
        num_heads=4,
        num_blocks=2,
    )
    model = SemanticVariationalAutoEncoder(config).eval()
    features = torch.randn(2, 9, config.input_dim)

    with torch.no_grad():
        first, first_kl = model.encode(features, sample=False)
        second, second_kl = model.encode(features, sample=False)
        sampled, sampled_kl = model.encode(features)
        reconstruction = model.decode(first)

    assert torch.equal(first, second)
    assert torch.equal(first_kl, second_kl)
    assert first.shape == (2, 9, config.bottleneck_dim)
    assert sampled.shape == first.shape
    assert sampled_kl.shape == (2,)
    assert reconstruction.shape == features.shape


def test_extract_state_dict_removes_distributed_prefix():
    payload = {
        "model_state_dict": {
            "module.encoder.weight": torch.ones(2, 2),
            "module.encoder.bias": torch.zeros(2),
        }
    }

    state_dict = _extract_state_dict(payload)

    assert set(state_dict) == {"encoder.weight", "encoder.bias"}


def test_resolve_artifacts_accepts_direct_checkpoint_file(tmp_path):
    variant_root = tmp_path / "variant"
    checkpoint_dir = variant_root / "checkpoints"
    checkpoint_dir.mkdir(parents=True)
    checkpoint = checkpoint_dir / "checkpoint_01000000.pt"
    checkpoint.touch()
    stats = variant_root / "latent_stats.pt"
    stats.touch()

    resolved_checkpoint, resolved_stats, resolved_root = _resolve_artifact_paths(
        str(checkpoint),
        variant=DEFAULT_SEMVAE_VARIANT,
        cache_dir=tmp_path / "cache",
    )

    assert resolved_checkpoint == checkpoint.absolute()
    assert resolved_stats == stats.absolute()
    assert resolved_root == variant_root.absolute()


class _FakeDINO(nn.Module):
    last_hidden_state = None

    def forward(self, *, pixel_values, return_dict):
        assert return_dict is True
        batch = pixel_values.shape[0]
        hidden = torch.arange(
            batch * (1 + 4 + 256) * 1024,
            dtype=torch.float32,
        ).reshape(batch, 1 + 4 + 256, 1024)
        self.last_hidden_state = hidden
        return SimpleNamespace(last_hidden_state=self.last_hidden_state)


def test_dino_extractor_removes_cls_and_register_tokens():
    extractor = DINOv2FeatureExtractor.__new__(DINOv2FeatureExtractor)
    nn.Module.__init__(extractor)
    extractor.device = torch.device("cpu")
    extractor.num_register_tokens = 4
    extractor.model = _FakeDINO()

    patch_tokens = extractor(torch.zeros(1, 3, 224, 224))

    assert patch_tokens.shape == (1, 256, 1024)
    assert patch_tokens.dtype == torch.float32
    assert torch.equal(patch_tokens, extractor.model.last_hidden_state[:, 5:, :])


def test_latent_stats_and_normalization_round_trip(tmp_path):
    stats_path = tmp_path / "latent_stats.pt"
    torch.save(
        {
            "mean": torch.arange(4, dtype=torch.float32).reshape(1, 1, 4),
            "std": torch.full((1, 1, 4), 2.0),
        },
        stats_path,
    )
    mean, std = _load_latent_stats(stats_path, bottleneck_dim=4)
    config = SemVAEConfig(
        input_dim=8,
        bottleneck_dim=4,
        hidden_dim=8,
        num_heads=2,
        num_blocks=1,
    )
    feature_extractor = SimpleNamespace(device=torch.device("cpu"))
    codec = SemVAEFeatureCodec(
        feature_extractor=feature_extractor,
        semvae=SemanticVariationalAutoEncoder(config),
        latent_mean=mean,
        latent_std=std,
        checkpoint_path=Path("checkpoint.pt"),
        stats_path=stats_path,
    )
    latents = torch.randn(2, 7, 4)

    reconstructed = codec.denormalize_latents(codec.normalize_latents(latents))

    assert torch.allclose(reconstructed, latents, atol=1e-6, rtol=1e-5)


def test_semvae_from_pretrained_keeps_legacy_256_default():
    parameter = inspect.signature(SemVAEFeatureCodec.from_pretrained).parameters["image_size"]

    assert parameter.default == 256


def _bare_preprocessor(image_size: int) -> DINOv2FeatureExtractor:
    extractor = DINOv2FeatureExtractor.__new__(DINOv2FeatureExtractor)
    nn.Module.__init__(extractor)
    extractor.image_size = image_size
    extractor.dino_size = image_size * 7 // 8
    extractor.register_buffer(
        "image_mean",
        torch.tensor([0.485, 0.456, 0.406]).reshape(1, 3, 1, 1),
        persistent=False,
    )
    extractor.register_buffer(
        "image_std",
        torch.tensor([0.229, 0.224, 0.225]).reshape(1, 3, 1, 1),
        persistent=False,
    )
    return extractor


def test_dino_batch_preprocess_supports_1024_pil_and_texture_tensors():
    extractor = _bare_preprocessor(1024)
    images = [Image.new("RGB", (1280, 1024), (64, 128, 192)) for _ in range(2)]

    pil_batch = extractor.preprocess_batch(images)
    texture_batch = torch.stack(
        [
            torch.full((3, 1024, 1024), -0.5),
            torch.full((3, 1024, 1024), 0.5),
        ]
    )
    tensor_batch = extractor.preprocess_batch(
        texture_batch,
        input_range="minus_one_one",
    )

    assert pil_batch.shape == (2, 3, 896, 896)
    assert tensor_batch.shape == (2, 3, 896, 896)
    assert torch.isfinite(pil_batch).all()
    assert torch.isfinite(tensor_batch).all()


class _FakeBatchExtractor(nn.Module):
    def __init__(self):
        super().__init__()
        self.device = torch.device("cpu")
        self.image_size = 1024
        self.dino_size = 896

    def forward(self, pixel_values):
        batch = pixel_values.shape[0]
        base = pixel_values.mean(dim=(1, 2, 3)).reshape(batch, 1, 1)
        return base.expand(batch, 16, 8) + torch.arange(8).reshape(1, 1, 8)


def test_semvae_training_batch_api_skips_decode_and_normalizes(tmp_path):
    config = SemVAEConfig(
        input_dim=8,
        bottleneck_dim=4,
        hidden_dim=8,
        num_heads=2,
        num_blocks=1,
    )
    codec = SemVAEFeatureCodec(
        feature_extractor=_FakeBatchExtractor(),
        semvae=SemanticVariationalAutoEncoder(config).eval(),
        latent_mean=torch.zeros(1, 1, 4),
        latent_std=torch.full((1, 1, 4), 2.0),
        checkpoint_path=tmp_path / "checkpoint.pt",
        stats_path=tmp_path / "stats.pt",
    )

    encoded = codec.encode_batch(torch.randn(2, 3, 896, 896), sample=False)

    assert codec.image_size == 1024
    assert codec.dino_size == 896
    assert encoded.features.shape == (2, 16, 8)
    assert encoded.latents.shape == (2, 16, 4)
    assert torch.equal(encoded.normalized_latents, encoded.latents / 2)
    assert all(
        torch.isfinite(tensor).all()
        for tensor in (encoded.features, encoded.latents, encoded.normalized_latents)
    )
