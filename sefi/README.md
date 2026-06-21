# SeFi Python Package

Reusable Python package for SeFi-Image inference. See `../README.md` for
installation, model checkpoints, and generation examples.

The package includes:

- checkpoint-derived model metadata
- checkpoint staging
- pipeline wrapper
- prompt/output helpers
- command-line interface

Weights and model-specific config are loaded from a local checkpoint artifact or
Hugging Face repo id passed through `--checkpoint` or
`SEFIInferencePipeline.from_pretrained(...)`. The artifact root should include
`sefi_config.yaml`.
