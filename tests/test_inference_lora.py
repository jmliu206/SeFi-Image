from __future__ import annotations

import sys
from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn
from omegaconf import OmegaConf
from peft import LoraConfig, PeftModel, get_peft_model
from safetensors.torch import save_file

import sefi.cli as cli_module
import sefi.pipeline as pipeline_module
from sefi.pipeline import SEFIInferencePipeline
from sefi.registry import ModelSpec
from sefi.runner import SEFIInferenceRunner, _resolve_peft_adapter_directory


class _FakeRunner:
    calls = []

    def __init__(self, config, **kwargs):
        self.config = config
        self.kwargs = kwargs
        self.device = torch.device(kwargs["device"])
        type(self).calls.append(kwargs)


def _patch_pipeline_runtime(monkeypatch, *, family="base"):
    resolved = []

    def resolve(*, checkpoint, cache_dir):
        resolved.append((checkpoint, str(cache_dir)))
        lowered = checkpoint.lower()
        if "adapter" in lowered:
            suffix = "adapter"
        elif "full" in lowered or "override" in lowered:
            suffix = "full_transformer"
        else:
            suffix = "base"
        return f"/cache/{suffix}", checkpoint

    spec = ModelSpec(
        name=f"test-{family}",
        family=family,
        scale="1b",
        default_dtype="fp32",
    )
    _FakeRunner.calls = []
    monkeypatch.setattr(pipeline_module, "resolve_checkpoint_to_local", resolve)
    monkeypatch.setattr(pipeline_module, "resolve_config_path", lambda *_: "/cache/config.yaml")
    monkeypatch.setattr(
        pipeline_module,
        "load_runtime_symbols",
        lambda: (_FakeRunner, lambda _: {"config": "sentinel"}),
    )
    monkeypatch.setattr(pipeline_module, "infer_model_spec", lambda *_, **__: spec)
    return resolved


def test_pipeline_without_adapter_preserves_old_runner_call(monkeypatch):
    resolved = _patch_pipeline_runtime(monkeypatch)

    pipe = SEFIInferencePipeline.from_pretrained(
        "SeFi-Image/SeFi-Image-1B-Base",
        device="cpu",
    )

    assert resolved == [
        ("SeFi-Image/SeFi-Image-1B-Base", "outputs/model_weights/sefi_inference")
    ]
    assert "adapter_path" not in _FakeRunner.calls[0]
    assert pipe.adapter_path == ""
    assert pipe.adapter_uri == ""
    assert pipe.transformer_checkpoint_path == ""
    assert pipe.transformer_checkpoint_uri == ""


def test_pipeline_stages_full_transformer_override_without_changing_base_config(
    monkeypatch,
):
    resolved = _patch_pipeline_runtime(monkeypatch)

    pipe = SEFIInferencePipeline.from_pretrained(
        "SeFi-Image/SeFi-Image-1B-Base",
        transformer_checkpoint_path="SeFi-Image/my-full-1b-override",
        device="cpu",
    )

    assert [item[0] for item in resolved] == [
        "SeFi-Image/SeFi-Image-1B-Base",
        "SeFi-Image/my-full-1b-override",
    ]
    assert _FakeRunner.calls[0]["checkpoint_path"] == "/cache/base"
    assert (
        _FakeRunner.calls[0]["transformer_checkpoint_path"]
        == "/cache/full_transformer"
    )
    assert pipe.transformer_checkpoint_path == "/cache/full_transformer"
    assert pipe.transformer_checkpoint_uri == "SeFi-Image/my-full-1b-override"


def test_pipeline_stages_hf_adapter_and_passes_local_path(monkeypatch):
    resolved = _patch_pipeline_runtime(monkeypatch)

    pipe = SEFIInferencePipeline.from_pretrained(
        "SeFi-Image/SeFi-Image-1B-Base",
        adapter_path="SeFi-Image/my-1b-adapter",
        device="cpu",
    )

    assert [item[0] for item in resolved] == [
        "SeFi-Image/SeFi-Image-1B-Base",
        "SeFi-Image/my-1b-adapter",
    ]
    assert _FakeRunner.calls[0]["adapter_path"] == "/cache/adapter"
    assert pipe.adapter_path == "/cache/adapter"
    assert pipe.adapter_uri == "SeFi-Image/my-1b-adapter"


def test_pipeline_rejects_adapter_for_non_base_checkpoint(monkeypatch):
    resolved = _patch_pipeline_runtime(monkeypatch, family="turbo")

    with pytest.raises(ValueError, match="only with SeFi Base"):
        SEFIInferencePipeline.from_pretrained(
            "SeFi-Image/SeFi-Image-1B-turbo",
            adapter_path="SeFi-Image/my-1b-adapter",
            device="cpu",
        )

    assert len(resolved) == 1


class _TinyBase(nn.Module):
    def __init__(self):
        super().__init__()
        self.proj = nn.Linear(4, 4, bias=False)

    def forward(self, x):
        return self.proj(x)


def _write_generic_adapter(path):
    model = get_peft_model(
        _TinyBase(),
        LoraConfig(
            r=2,
            lora_alpha=2,
            lora_dropout=0.0,
            target_modules=["proj"],
            bias="none",
        ),
    )
    with torch.no_grad():
        for name, parameter in model.named_parameters():
            if "lora_A" in name:
                parameter.fill_(0.25)
            elif "lora_B" in name:
                parameter.fill_(0.5)
    model.save_pretrained(path, safe_serialization=True)


@pytest.mark.parametrize("nested", [False, True])
def test_runner_loads_frozen_adapter_from_clean_or_training_checkpoint_dir(
    tmp_path, nested
):
    root = tmp_path / ("checkpoint" if nested else "clean_adapter")
    adapter_dir = root / "adapter" if nested else root
    adapter_dir.mkdir(parents=True)
    _write_generic_adapter(adapter_dir)

    runner = SEFIInferenceRunner.__new__(SEFIInferenceRunner)
    runner.config = OmegaConf.create({"model": {"transformer_scale": "1b"}})
    runner.device = torch.device("cpu")
    runner.weight_dtype = torch.float32
    runner.transformer = _TinyBase().eval()
    inputs = torch.ones(1, 4)
    before = runner.transformer(inputs).detach().clone()

    runner.load_adapter(str(root))

    assert isinstance(runner.transformer, PeftModel)
    assert runner.adapter_path == str(adapter_dir.resolve())
    assert runner.adapter_metadata is None
    assert not any(parameter.requires_grad for parameter in runner.transformer.parameters())
    assert not torch.equal(runner.transformer(inputs), before)


def test_runner_rejects_adapter_metadata_scale_mismatch(tmp_path):
    adapter_dir = tmp_path / "adapter"
    adapter_dir.mkdir()
    _write_generic_adapter(adapter_dir)
    (adapter_dir / "sefi_adapter_config.json").write_text(
        '{"schema_version": 1, "scale": "2b"}\n',
        encoding="utf-8",
    )
    runner = SEFIInferenceRunner.__new__(SEFIInferenceRunner)
    runner.config = OmegaConf.create({"model": {"transformer_scale": "1b"}})
    runner.device = torch.device("cpu")
    runner.weight_dtype = torch.float32
    runner.transformer = _TinyBase()

    with pytest.raises(ValueError, match="scale mismatch"):
        runner.load_adapter(str(adapter_dir))


class _DummyTextureCodec(nn.Module):
    def __init__(self):
        super().__init__()
        self.texture_vae = SimpleNamespace(
            config=SimpleNamespace(block_out_channels=[1, 2])
        )


def test_runner_load_order_is_base_then_full_override_then_adapter(monkeypatch):
    events = []
    components = SimpleNamespace(
        transformer=nn.Linear(2, 2),
        text_encoder=nn.Linear(2, 2),
        texture_codec=_DummyTextureCodec(),
        noise_scheduler=object(),
        pipeline_cls=object,
        semantic_channels=1,
        texture_channels=1,
        total_channels=2,
    )
    monkeypatch.setattr("sefi.runner.build_components", lambda *_, **__: components)

    def fake_load_checkpoint(self, path, *, label="checkpoint"):
        events.append(("checkpoint", path, label))

    def fake_load_adapter(self, path):
        events.append(("adapter", path))

    monkeypatch.setattr(SEFIInferenceRunner, "load_checkpoint", fake_load_checkpoint)
    monkeypatch.setattr(SEFIInferenceRunner, "load_adapter", fake_load_adapter)
    config = OmegaConf.create(
        {
            "training": {
                "mixed_precision": "fp32",
                "sefi": {"delta_t_min": 0.1, "delta_t_max": 0.1},
            }
        }
    )

    SEFIInferenceRunner(
        config,
        checkpoint_path="base",
        transformer_checkpoint_path="full",
        adapter_path="adapter",
        device="cpu",
        inference_dtype="fp32",
    )

    assert events == [
        ("checkpoint", "base", "checkpoint"),
        ("checkpoint", "full", "full transformer override"),
        ("adapter", "adapter"),
    ]


def test_runner_strictly_loads_portable_full_transformer_override(tmp_path):
    source = nn.Linear(3, 2)
    with torch.no_grad():
        source.weight.fill_(4.0)
        source.bias.fill_(-2.0)
    transformer_dir = tmp_path / "portable" / "transformer"
    transformer_dir.mkdir(parents=True)
    save_file(
        source.state_dict(),
        transformer_dir / "diffusion_pytorch_model.safetensors",
    )
    runner = SEFIInferenceRunner.__new__(SEFIInferenceRunner)
    runner.transformer = nn.Linear(3, 2)

    runner.load_checkpoint(
        str(tmp_path / "portable"),
        label="full transformer override",
    )

    torch.testing.assert_close(runner.transformer.weight, source.weight)
    torch.testing.assert_close(runner.transformer.bias, source.bias)

    incompatible_dir = tmp_path / "incompatible" / "transformer"
    incompatible_dir.mkdir(parents=True)
    save_file(
        {"weight": source.weight},
        incompatible_dir / "diffusion_pytorch_model.safetensors",
    )
    with pytest.raises(ValueError, match="missing transformer keys"):
        runner.load_checkpoint(str(tmp_path / "incompatible"))


def test_adapter_directory_error_lists_supported_layouts(tmp_path):
    with pytest.raises(FileNotFoundError, match="adapter_config.json"):
        _resolve_peft_adapter_directory(str(tmp_path))


def test_cli_accepts_optional_adapter_path(monkeypatch):
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "sefi-inference",
            "--checkpoint",
            "SeFi-Image/SeFi-Image-1B-Base",
            "--adapter-path",
            "SeFi-Image/my-1b-adapter",
            "--transformer-checkpoint",
            "SeFi-Image/my-full-1b-override",
        ],
    )

    args = cli_module._parse_args()

    assert args.adapter_path == "SeFi-Image/my-1b-adapter"
    assert args.transformer_checkpoint == "SeFi-Image/my-full-1b-override"
