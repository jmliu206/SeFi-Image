"""Qwen3-VL text encoder wrapper for SEFI T2I inference."""

import os
from typing import Sequence

import torch
import torch.nn as nn
from torch import Tensor
from transformers import AutoProcessor, Qwen3VLForConditionalGeneration


QWEN3VL_MODEL_PATHS = {
    "qwen3vl_2b": "Qwen3-VL-2B-Instruct",
    "qwen3vl_4b": "Qwen3-VL-4B-Instruct",
    "qwen3vl_8b": "Qwen3-VL-8B-Instruct",
}


def resolve_qwen3vl_model_path(
    model_name: str,
    weights_root: str = "outputs/model_weights",
) -> str:
    if model_name not in QWEN3VL_MODEL_PATHS:
        raise ValueError(
            f"Unsupported Qwen3-VL model: {model_name}. "
            f"Supported: {list(QWEN3VL_MODEL_PATHS.keys())}"
        )

    model_path = os.path.join(weights_root, QWEN3VL_MODEL_PATHS[model_name])
    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"Qwen3-VL model not found: {model_path}. "
            "Please download weights to outputs/model_weights first."
        )
    return model_path


class Qwen3VLTextEncoder(nn.Module):
    """Text embedding wrapper using Qwen3-VL language model."""

    def __init__(
        self,
        model_name: str,
        weights_root: str = "outputs/model_weights",
        max_length: int = 512,
        hidden_layers: Sequence[int] = (9, 18, 27),
        torch_dtype: torch.dtype = torch.bfloat16,
    ):
        super().__init__()

        self.model_name = model_name
        self.max_length = int(max_length)
        self.hidden_layers = tuple(int(x) for x in hidden_layers)

        model_path = resolve_qwen3vl_model_path(model_name, weights_root=weights_root)
        self.processor = AutoProcessor.from_pretrained(model_path, local_files_only=True)
        self.tokenizer = self.processor.tokenizer

        self.model = Qwen3VLForConditionalGeneration.from_pretrained(
            model_path,
            torch_dtype=torch_dtype,
            local_files_only=True,
            device_map="cpu",
        )

        # Keep only text tower to save memory.
        if hasattr(self.model, "model") and hasattr(self.model.model, "visual"):
            del self.model.model.visual

        self.model.eval()

        text_hidden_size = int(self.model.config.text_config.hidden_size)
        self.output_dim = text_hidden_size * len(self.hidden_layers)

    def _build_chat_text(self, caption: str) -> str:
        messages = [{"role": "user", "content": [{"type": "text", "text": caption}]}]
        try:
            return self.processor.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=False,
            )
        except TypeError:
            return self.processor.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )

    @staticmethod
    def _prepare_text_ids(x: Tensor, t_coord: Tensor | None = None) -> Tensor:
        batch, seq_len, _ = x.shape
        out_ids = []

        for i in range(batch):
            t = torch.arange(1) if t_coord is None else t_coord[i]
            h = torch.arange(1)
            w = torch.arange(1)
            l = torch.arange(seq_len)
            coords = torch.cartesian_prod(t, h, w, l)
            out_ids.append(coords)

        return torch.stack(out_ids)

    @torch.no_grad()
    def encode(self, captions: list[str], dtype: torch.dtype | None = None) -> tuple[Tensor, Tensor]:
        device = next(self.model.parameters()).device
        model_dtype = next(self.model.parameters()).dtype
        if dtype is None:
            dtype = model_dtype

        chat_texts = [self._build_chat_text(caption) for caption in captions]
        tokenized = self.tokenizer(
            chat_texts,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=self.max_length,
        )
        input_ids = tokenized["input_ids"].to(device)
        attention_mask = tokenized["attention_mask"].to(device)

        output = self.model.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
            use_cache=False,
            return_dict=True,
        )

        hidden_states = output.hidden_states
        max_idx = len(hidden_states) - 1
        for layer_idx in self.hidden_layers:
            if layer_idx > max_idx:
                raise ValueError(
                    f"Requested hidden layer {layer_idx}, but model only provides up to {max_idx}."
                )

        stacked = torch.stack([hidden_states[idx] for idx in self.hidden_layers], dim=1)
        stacked = stacked.to(dtype=dtype)

        batch, num_layers, seq_len, hidden_dim = stacked.shape
        prompt_embeds = stacked.permute(0, 2, 1, 3).reshape(batch, seq_len, num_layers * hidden_dim)
        text_ids = self._prepare_text_ids(prompt_embeds).to(device)

        return prompt_embeds, text_ids
