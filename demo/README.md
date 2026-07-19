# SeFi-Image Demos

This directory contains small, user-facing examples and reference code.
Run all commands from the repository root so Python can import `sefi`.

## Demo Status

| Demo | Purpose | Status |
| :--- | :--- | :--- |
| [SemVAE](semvae/README.md) | Extract visual features, compress them into semantic latents, and check feature reconstruction with cosine similarity. | Runnable |

## Installation

Start with the environment in the repository [README](../README.md#installation).
Individual demos document any additional dependencies they need.

Model weights may be loaded from Hugging Face or from a local checkpoint. If a
repository requires authentication, log in once with `huggingface-cli login`.

## Public Files and Local Outputs

Files under `demo/` are intended to be tracked and published. Downloaded model
weights, input data, metrics, latents, logs, and checkpoints belong under
`outputs/`, which is ignored by Git. The suggested local layout is:

```text
outputs/
├── demo_data/
│   └── semvae/
├── demo/
│   └── semvae/
└── model_weights/
```

Demo data is hosted separately so downloaded binaries do not live in the code
repository:

- [SeFi-Image-SemVAE-Demo](https://huggingface.co/datasets/SeFi-Image/SeFi-Image-SemVAE-Demo)
  is published and contains eight standalone Fine-T2I synthetic smoke-test
  images plus an embedded-image Parquet `test` split.
