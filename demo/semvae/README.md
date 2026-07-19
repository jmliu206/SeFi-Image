# SemVAE Feature Compression Demo

This runnable demo executes the complete deterministic SemVAE path:

```text
image -> DINOv2 patch features -> SemVAE latent -> reconstructed features
```

It is intended both as a quick checkpoint smoke test and as readable reference
code for extracting and compressing semantic image representations.

## Setup

Install the dependencies from the repository [installation guide](../../README.md#installation).
The demo uses PyTorch, torchvision, Transformers, Hugging Face Hub, and Pillow;
no training dependencies are required.

Download one image from the dedicated public demo dataset:

```bash
hf download SeFi-Image/SeFi-Image-SemVAE-Demo \
  images/city_taxi.jpg \
  --repo-type dataset \
  --local-dir outputs/demo_data/semvae
```

You can also use any local RGB image instead.

## Quick Start

Run from the repository root:

```bash
python demo/semvae/run_semvae.py \
  --checkpoint SeFi-Image/SeFi-Image-SemVAE \
  --image outputs/demo_data/semvae/images/city_taxi.jpg \
  --output-dir outputs/demo/semvae \
  --device auto
```

`--checkpoint` accepts either the Hugging Face repository id above or a local
repository root, variant directory, or `.pt` checkpoint file for the same
`dinov2_vitl14_reg/transformer_ch16` architecture. A direct `.pt` path must keep
its matching `latent_stats.pt` one directory above its `checkpoints/` folder.
Other SemVAE architectures are not auto-inferred in this first release. The
default DINOv2 model is `facebook/dinov2-with-registers-large`. Both downloads
and all generated artifacts remain under `outputs/` when the default paths are
used.

CPU execution is supported but slower. To use already downloaded assets:

```bash
python demo/semvae/run_semvae.py \
  --checkpoint outputs/model_weights/SeFi-Image-SemVAE \
  --vfm-checkpoint outputs/model_weights/VFM/dinov2-with-registers-large \
  --image outputs/demo_data/semvae/images/city_taxi.jpg \
  --output-dir outputs/demo/semvae \
  --device cpu
```

## Expected Result

The public checkpoint pairs DINOv2-L/14 with registers (feature width 1024) and
a 16-channel SemVAE bottleneck. The demo prints:

- input feature, compressed latent, and reconstructed feature shapes;
- the feature-dimension compression ratio (`1024 / 16 = 64x`), which is not a
  serialized file compression ratio;
- token-wise reconstruction cosine similarity mean and standard deviation;
- basic finite-value and shape validation.

For the default preprocessing path, the representative shape progression is:

```text
features:       [batch, tokens, 1024]
semantic latent:[batch, tokens,   16]
reconstruction: [batch, tokens, 1024]
```

The reference implementation averages cosine similarity over the 256 patch
tokens. A correct run is normally around `0.9`; the CLI uses a deliberately
loose `0.80` smoke threshold to catch loading, token slicing, and preprocessing
errors without treating one image as a quality benchmark. On the public
`city_taxi.jpg` sample, one checked CUDA result is:

```text
feature shape:       [1, 256, 1024]
latent shape:        [1, 256, 16]
reconstruction:      [1, 256, 1024]
compression ratio:   64x
mean token cosine:   0.917828
reconstruction MSE:  0.087731
status:              pass
```

All eight published samples passed; their checked mean cosine range is
`0.886340`–`0.922122`. Exact floating-point results can vary slightly by
dependency version and hardware.

## Outputs

For an input named `example.png`, the command writes:

```text
outputs/demo/semvae/
├── example_metrics.json
└── example_semantic_latents.pt
```

The tensor file contains both `raw_latents`, which the SemVAE decoder expects,
and `normalized_latents`, which follow the normalization convention needed by
the SeFi DiT. Pass `--no-save-latents` if only metrics are needed. The command
exits nonzero when the mean cosine is below `--min-cosine`.

## Python API

The same reusable loader is available from the `sefi` package:

```python
from PIL import Image
from sefi import SemVAEFeatureCodec

codec = SemVAEFeatureCodec.from_pretrained(
    "SeFi-Image/SeFi-Image-SemVAE",
    device="cuda",
)

with Image.open("my_image.png") as image:
    result = codec.encode_image(image)

print(result.features.shape)            # [1, 256, 1024]
print(result.latents.shape)             # [1, 256, 16]
print(result.normalized_latents.shape)  # [1, 256, 16], DiT normalization
print(result.token_cosine.mean().item())
```

## Determinism and Latent Normalization

The smoke test uses the posterior mode rather than a random posterior sample,
with the model in evaluation mode and gradients disabled. This keeps repeated
runs on the same hardware and input reproducible.

`latent_stats.pt` contains the channel statistics used by the SeFi DiT latent
convention. Feature reconstruction must decode the raw latent, or first undo
normalization; a normalized latent must not be passed directly to the SemVAE
decoder.

## Demo Data

Images are not committed with this code. The dedicated
[SeFi-Image-SemVAE-Demo dataset](https://huggingface.co/datasets/SeFi-Image/SeFi-Image-SemVAE-Demo)
contains eight standalone JPEGs as both raw image files and an embedded-image
Parquet `test` split. They are sampled from the Apache-2.0
[`ma-xu/fine-t2i`](https://huggingface.co/datasets/ma-xu/fine-t2i)
`synthetic_enhanced_prompt_random_resolution` subset; its real-image `curated`
subset is not used. The dataset card pins the upstream revision and records the
source ids, captions, license, and SHA-256 hashes. It is only for loader, shape,
latent, and cosine smoke tests—not training or quality evaluation.

## Current Scope

This first example follows the square 256-pixel preprocessing path used by the
checkpoint: resize and center-crop to 256, then resize to the DINO 224-pixel
input. Free-aspect-ratio preprocessing is outside this demo's scope.
