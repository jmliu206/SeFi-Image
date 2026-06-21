"""Flux2 SEFI transformer wrapper with explicit dual timestep embedding."""

import inspect
from typing import Any

import torch
import torch.nn as nn
from torch import Tensor

from diffusers import Flux2Transformer2DModel
from diffusers.models.embeddings import TimestepEmbedding, Timesteps

try:
    from diffusers.utils import USE_PEFT_BACKEND, scale_lora_layers, unscale_lora_layers
except Exception:  # pragma: no cover - compatibility fallback
    USE_PEFT_BACKEND = False

    def scale_lora_layers(model, scale):
        del model, scale

    def unscale_lora_layers(model, scale):
        del model, scale

class SEFIDualTimestepEmbeddings(nn.Module):
    """SEFI dual timestep embeddings: concat([emb_sem, emb_tex])."""

    def __init__(self, in_channels: int, embedding_dim: int, bias: bool = False):
        super().__init__()
        if embedding_dim % 2 != 0:
            raise ValueError(
                f"SEFI dual timestep embedding requires even embedding_dim, got {embedding_dim}."
            )

        half_dim = embedding_dim // 2
        self.time_proj = Timesteps(
            num_channels=int(in_channels),
            flip_sin_to_cos=True,
            downscale_freq_shift=0,
        )
        self.semantic_embedder = TimestepEmbedding(
            in_channels=int(in_channels),
            time_embed_dim=half_dim,
            sample_proj_bias=bias,
        )
        self.texture_embedder = TimestepEmbedding(
            in_channels=int(in_channels),
            time_embed_dim=half_dim,
            sample_proj_bias=bias,
        )

    def forward(self, timestep_sem: Tensor, timestep_tex: Tensor) -> Tensor:
        sem_proj = self.time_proj(timestep_sem)
        tex_proj = self.time_proj(timestep_tex)
        sem_emb = self.semantic_embedder(sem_proj.to(timestep_sem.dtype))
        tex_emb = self.texture_embedder(tex_proj.to(timestep_tex.dtype))
        return torch.cat([sem_emb, tex_emb], dim=-1)


class Flux2SEFITransformer2DModel(nn.Module):
    """Flux2 transformer wrapper for SEFI inference."""

    def __init__(
        self,
        backbone_config: dict,
        text_input_dim: int,
    ):
        super().__init__()

        self.backbone = Flux2Transformer2DModel.from_config(backbone_config)
        # SEFI handles semantic/texture timesteps explicitly and does not reuse guidance semantics.
        self.backbone.time_guidance_embed = nn.Identity()
        self._double_mod_img_kwarg, self._double_mod_txt_kwarg = (
            self._resolve_double_stream_modulation_kwargs()
        )
        self._single_mod_kwarg = self._resolve_single_stream_modulation_kwarg()

        self.dual_time_embed = SEFIDualTimestepEmbeddings(
            in_channels=int(self.backbone.config.timestep_guidance_channels),
            embedding_dim=int(self.backbone.inner_dim),
            bias=False,
        )

        expected_text_dim = int(self.backbone.config.joint_attention_dim)
        if int(text_input_dim) != expected_text_dim:
            raise ValueError(
                f"Text embedding dim mismatch: text={text_input_dim}, "
                f"transformer expects={expected_text_dim}."
            )

    def _resolve_double_stream_modulation_kwargs(self) -> tuple[str, str]:
        if not self.backbone.transformer_blocks:
            raise ValueError("Flux2 backbone must define at least one double-stream block.")
        params = inspect.signature(
            self.backbone.transformer_blocks[0].forward
        ).parameters
        if "temb_mod_img" in params and "temb_mod_txt" in params:
            return "temb_mod_img", "temb_mod_txt"
        if "temb_mod_params_img" in params and "temb_mod_params_txt" in params:
            return "temb_mod_params_img", "temb_mod_params_txt"
        raise ValueError(
            "Unsupported Flux2TransformerBlock.forward signature. "
            "Expected temb_mod_img/temb_mod_txt or "
            "temb_mod_params_img/temb_mod_params_txt."
        )

    def _resolve_single_stream_modulation_kwarg(self) -> str:
        if not self.backbone.single_transformer_blocks:
            raise ValueError("Flux2 backbone must define at least one single-stream block.")
        params = inspect.signature(
            self.backbone.single_transformer_blocks[0].forward
        ).parameters
        if "temb_mod" in params:
            return "temb_mod"
        if "temb_mod_params" in params:
            return "temb_mod_params"
        raise ValueError(
            "Unsupported Flux2SingleTransformerBlock.forward signature. "
            "Expected temb_mod or temb_mod_params."
        )

    def _format_single_stream_modulation(self, single_stream_mod):
        if self._single_mod_kwarg != "temb_mod_params":
            return single_stream_mod
        if (
            isinstance(single_stream_mod, tuple)
            and len(single_stream_mod) == 1
            and isinstance(single_stream_mod[0], tuple)
            and len(single_stream_mod[0]) == 3
        ):
            return single_stream_mod[0]
        return single_stream_mod

    def enable_gradient_checkpointing(self):
        self.backbone.enable_gradient_checkpointing()

    def forward(
        self,
        hidden_states: Tensor,
        timestep_sem: Tensor,
        timestep_tex: Tensor,
        encoder_hidden_states: Tensor,
        txt_ids: Tensor,
        img_ids: Tensor,
        joint_attention_kwargs: dict[str, Any] | None = None,
    ) -> Tensor:
        model_device = hidden_states.device
        model_dtype = next(self.backbone.parameters()).dtype

        hidden_states = hidden_states.to(device=model_device, dtype=model_dtype)

        encoder_hidden_states = encoder_hidden_states.to(
            device=model_device, dtype=model_dtype
        )

        timestep_sem = timestep_sem.to(device=model_device, dtype=model_dtype)
        timestep_tex = timestep_tex.to(device=model_device, dtype=model_dtype)

        if joint_attention_kwargs is not None:
            joint_attention_kwargs = joint_attention_kwargs.copy()
            lora_scale = joint_attention_kwargs.pop("scale", 1.0)
        else:
            lora_scale = 1.0

        # 0) LoRA scaling (keep semantics aligned with Flux2 forward).
        if USE_PEFT_BACKEND:
            scale_lora_layers(self.backbone, lora_scale)

        num_txt_tokens = encoder_hidden_states.shape[1]
        # 1) SEFI dual-time embedding + modulation parameters.
        temb = self.dual_time_embed(timestep_sem * 1000, timestep_tex * 1000)

        double_stream_mod_img = self.backbone.double_stream_modulation_img(temb)
        double_stream_mod_txt = self.backbone.double_stream_modulation_txt(temb)
        single_stream_mod = self.backbone.single_stream_modulation(temb)
        single_stream_block_mod = self._format_single_stream_modulation(single_stream_mod)

        # 2) Input projection for image/text streams.
        hidden_states = self.backbone.x_embedder(hidden_states)
        encoder_hidden_states = self.backbone.context_embedder(encoder_hidden_states)

        if img_ids.ndim == 3:
            img_ids = img_ids[0]
        if txt_ids.ndim == 3:
            txt_ids = txt_ids[0]

        image_rotary_emb = self.backbone.pos_embed(img_ids)
        text_rotary_emb = self.backbone.pos_embed(txt_ids)
        concat_rotary_emb = (
            torch.cat([text_rotary_emb[0], image_rotary_emb[0]], dim=0),
            torch.cat([text_rotary_emb[1], image_rotary_emb[1]], dim=0),
        )

        # 3) Double-stream transformer blocks.
        for block in self.backbone.transformer_blocks:
            if torch.is_grad_enabled() and self.backbone.gradient_checkpointing:
                encoder_hidden_states, hidden_states = self.backbone._gradient_checkpointing_func(
                    block,
                    hidden_states,
                    encoder_hidden_states,
                    double_stream_mod_img,
                    double_stream_mod_txt,
                    concat_rotary_emb,
                    joint_attention_kwargs,
                )
            else:
                encoder_hidden_states, hidden_states = block(
                    hidden_states=hidden_states,
                    encoder_hidden_states=encoder_hidden_states,
                    **{
                        self._double_mod_img_kwarg: double_stream_mod_img,
                        self._double_mod_txt_kwarg: double_stream_mod_txt,
                    },
                    image_rotary_emb=concat_rotary_emb,
                    joint_attention_kwargs=joint_attention_kwargs,
                )

        hidden_states = torch.cat([encoder_hidden_states, hidden_states], dim=1)

        # 4) Single-stream transformer blocks.
        for block in self.backbone.single_transformer_blocks:
            if torch.is_grad_enabled() and self.backbone.gradient_checkpointing:
                hidden_states = self.backbone._gradient_checkpointing_func(
                    block,
                    hidden_states,
                    None,
                    single_stream_block_mod,
                    concat_rotary_emb,
                    joint_attention_kwargs,
                )
            else:
                hidden_states = block(
                    hidden_states=hidden_states,
                    encoder_hidden_states=None,
                    **{self._single_mod_kwarg: single_stream_block_mod},
                    image_rotary_emb=concat_rotary_emb,
                    joint_attention_kwargs=joint_attention_kwargs,
                )

        # 5) Output layers.
        hidden_states = hidden_states[:, num_txt_tokens:, ...]
        hidden_states = self.backbone.norm_out(hidden_states, temb)
        model_pred = self.backbone.proj_out(hidden_states)

        if USE_PEFT_BACKEND:
            unscale_lora_layers(self.backbone, lora_scale)

        return model_pred
