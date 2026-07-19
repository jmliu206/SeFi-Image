"""Transformer SemVAE used by the public SeFi semantic checkpoint."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
from torch import Tensor


class DiagonalGaussianDistribution:
    """Diagonal Gaussian posterior for sequence tensors shaped ``[B, L, D]``."""

    def __init__(self, parameters: Tensor, deterministic: bool = False):
        self.parameters = parameters
        self.mean, self.logvar = torch.chunk(parameters, 2, dim=-1)
        self.logvar = torch.clamp(self.logvar, -30.0, 20.0)
        self.deterministic = bool(deterministic)
        self.std = torch.exp(0.5 * self.logvar)
        self.var = torch.exp(self.logvar)
        if self.deterministic:
            self.std = torch.zeros_like(self.mean)
            self.var = torch.zeros_like(self.mean)

    def sample(self, generator: torch.Generator | None = None) -> Tensor:
        noise = torch.randn(
            self.mean.shape,
            generator=generator,
            device=self.parameters.device,
            dtype=self.parameters.dtype,
        )
        return self.mean + self.std * noise

    def mode(self) -> Tensor:
        return self.mean

    def kl(self) -> Tensor:
        if self.deterministic:
            return torch.zeros(self.mean.shape[0], device=self.mean.device)
        return 0.5 * torch.sum(
            self.mean.square() + self.var - 1.0 - self.logvar,
            dim=(1, 2),
        )


class TransformerBlock(nn.Module):
    """Pre-normalized self-attention block used by SemVAE."""

    def __init__(self, embed_dim: int, num_heads: int, mlp_ratio: float = 4.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)
        self.norm2 = nn.LayerNorm(embed_dim)
        mlp_hidden_dim = int(embed_dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, mlp_hidden_dim),
            nn.GELU(),
            nn.Linear(mlp_hidden_dim, embed_dim),
        )

    def forward(self, hidden_states: Tensor) -> Tensor:
        normed = self.norm1(hidden_states)
        attended, _ = self.attn(normed, normed, normed, need_weights=False)
        hidden_states = hidden_states + attended
        return hidden_states + self.mlp(self.norm2(hidden_states))


@dataclass(frozen=True)
class SemVAEConfig:
    """Architecture settings for a SemVAE checkpoint."""

    input_dim: int = 1024
    bottleneck_dim: int = 16
    hidden_dim: int = 1024
    num_heads: int = 8
    num_blocks: int = 4
    mlp_ratio: float = 4.0


class SemanticVariationalAutoEncoder(nn.Module):
    """Compress semantic patch features with a Transformer VAE."""

    def __init__(self, config: SemVAEConfig | None = None):
        super().__init__()
        self.config = config or SemVAEConfig()
        self.input_dim = int(self.config.input_dim)
        self.bottleneck_dim = int(self.config.bottleneck_dim)
        self.hidden_dim = int(self.config.hidden_dim)

        self.encoder = self._build_stack(
            input_dim=self.input_dim,
            output_dim=self.bottleneck_dim * 2,
        )
        self.decoder = self._build_stack(
            input_dim=self.bottleneck_dim,
            output_dim=self.input_dim,
        )

    def _build_stack(self, *, input_dim: int, output_dim: int) -> nn.Sequential:
        layers: list[nn.Module] = [nn.Linear(input_dim, self.hidden_dim)]
        layers.extend(
            TransformerBlock(
                self.hidden_dim,
                self.config.num_heads,
                self.config.mlp_ratio,
            )
            for _ in range(self.config.num_blocks)
        )
        layers.extend((nn.LayerNorm(self.hidden_dim), nn.Linear(self.hidden_dim, output_dim)))
        return nn.Sequential(*layers)

    def posterior(self, features: Tensor) -> DiagonalGaussianDistribution:
        return DiagonalGaussianDistribution(self.encoder(features))

    def encode(
        self,
        features: Tensor,
        *,
        sample: bool = True,
        generator: torch.Generator | None = None,
    ) -> tuple[Tensor, Tensor]:
        """Encode features using the reference training-time sampling semantics."""

        posterior = self.posterior(features)
        latents = posterior.sample(generator=generator) if sample else posterior.mode()
        return latents, posterior.kl()

    def decode(self, latents: Tensor) -> Tensor:
        return self.decoder(latents)

    def forward(
        self,
        features: Tensor,
        *,
        sample: bool = True,
        generator: torch.Generator | None = None,
    ) -> tuple[Tensor, Tensor, Tensor]:
        latents, kl = self.encode(
            features,
            sample=sample,
            generator=generator,
        )
        return self.decode(latents), latents, kl
