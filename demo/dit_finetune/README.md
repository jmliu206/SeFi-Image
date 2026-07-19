# SeFi DiT Fine-tuning Demo

This demo uses one shared, fixed-1024 image/text pipeline for LoRA and full
fine-tuning. LoRA is the default 1B recipe; 2B/5B and full tuning are explicit
alternatives. Both modes use the same paired dataloader, online frozen
VAE/DINOv2-L/SemVAE/Qwen encoding, and semantic-first flow-matching loss. REPA
and dynamic resolution are intentionally absent.

The paired demo dataset is public at
[`SeFi-Image/SeFi-Image-DiT-Finetune-Demo`](https://huggingface.co/datasets/SeFi-Image/SeFi-Image-DiT-Finetune-Demo).
All released configs pin revision
`475e616299b8c623c8a36ea5222a0e26bae91f06`; they never depend on a floating
`main`. The local data scripts remain offline and never create a Hub repository,
upload files, commit code, or push a branch.

## Install training dependencies

From the repository root, create a Python 3.11 environment and install the
runtime plus training packages. Choose the PyTorch wheel index that matches
your machine:

```bash
python -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
python -m pip install \
  transformers accelerate safetensors huggingface_hub omegaconf pillow \
  peft datasets pyarrow deepspeed
python -m pip install \
  "git+https://github.com/huggingface/diffusers.git@277e3055898dd98c89cafb7df7c1d359b554df76"
```

The Git-pinned Diffusers build is the exact revision used for the smoke matrix
and provides the Flux2 classes required by the public Base checkpoints. The
other checked versions were PyTorch 2.8.0, torchvision 0.23.0, Transformers
4.57.1, Accelerate 1.11.0, PEFT 0.18.0, datasets 4.4.1, PyArrow 19.0.1, and
DeepSpeed 0.18.6. These versions document the verified environment rather than
an asserted minimum-version range.

The Base repositories may require accepting their terms and authenticating
with Hugging Face:

```bash
hf auth login
```

The checked recipe uses bf16 and is designed for one node with eight 80GB-or-
larger NVIDIA GPUs. Full tuning always uses DeepSpeed ZeRO-2 and a complete GPU
FP32 EMA on every rank. LoRA uses ordinary Accelerate DDP, with no DeepSpeed and
no EMA.

## Train

The default 1B LoRA smoke command downloads the pinned public dataset and model
assets into `outputs/`:

```bash
mkdir -p outputs/cache/triton

NCCL_DEBUG=WARN \
TRITON_CACHE_DIR="$PWD/outputs/cache/triton" \
torchrun --standalone --nproc_per_node=8 \
  demo/dit_finetune/train.py \
  --config demo/dit_finetune/configs/lora_1b.yaml \
  --smoke
```

`--smoke` runs two real 1024px optimization steps with online encoders, zero
warmup, and deliberately skips large checkpoints and exports. Remove it for a
normal run. Available recipes are:

| Mode | 1B default | 2B | 5B |
| --- | --- | --- | --- |
| LoRA / DDP | `lora_1b.yaml` | `lora_2b.yaml` | `lora_5b.yaml` |
| Full / ZeRO-2 / GPU EMA | `full_1b_zero2.yaml` | `full_2b_zero2.yaml` | `full_5b_zero2.yaml` |

For example, full 1B uses the identical command with:

```text
--config demo/dit_finetune/configs/full_1b_zero2.yaml
```

The configs pin all currently published model and dataset revisions. CLI
overrides support local or Hub paths for `--dataset`, `--base-checkpoint`,
`--semvae-checkpoint`, and `--vfm-checkpoint`.

Resume from a training checkpoint by keeping the same recipe/data/world size
and increasing the target step:

```bash
torchrun --standalone --nproc_per_node=8 \
  demo/dit_finetune/train.py \
  --config demo/dit_finetune/configs/lora_1b.yaml \
  --resume outputs/demo/dit_finetune/lora_1b/checkpoints/checkpoint-00001000 \
  --max-train-steps 2000
```

LoRA resume checkpoints contain only the adapter plus optimizer, scheduler,
RNG, and exact data position; they do not duplicate the frozen Base. Full
checkpoints contain ZeRO-2 state and FP32 EMA. Clean outputs are separate:

- LoRA: `OUTPUT/adapter/` (PEFT adapter only).
- Full: `OUTPUT/export/transformer/` (EMA transformer by default).

Use either clean artifact while retaining the public Base assets:

```bash
python inference.py \
  --checkpoint SeFi-Image/SeFi-Image-1B-Base \
  --adapter-path outputs/demo/dit_finetune/lora_1b/adapter \
  --prompt "A red fox sitting beside a mountain lake."

python inference.py \
  --checkpoint SeFi-Image/SeFi-Image-1B-Base \
  --transformer-checkpoint outputs/demo/dit_finetune/full_1b_zero2/export \
  --prompt "A red fox sitting beside a mountain lake."
```

The loading order is Base, optional full transformer override, then optional
LoRA adapter, preserving every pre-existing inference call when neither new
option is supplied.

## Checked smoke matrix

The final code was exercised against the exact published Parquet bytes on eight
NVIDIA H20 GPUs with batch size 1/GPU, fixed 1024px images, gradient
checkpointing, and two optimizer steps. These are engineering checks, not
convergence claims:

| Mode | Scale | Trainable parameters | Loss step 1 / 2 | Peak GiB/GPU |
| --- | ---: | ---: | ---: | ---: |
| LoRA | 1B | 7,995,392 | 0.3228 / 0.3695 | 8.79 |
| LoRA | 2B | 12,451,840 | 0.3180 / 0.3632 | 10.73 |
| LoRA | 5B | 21,884,928 | 0.3145 / 0.3584 | 21.08 |
| Full + EMA | 1B | 1,177,621,504 | 0.3229 / 0.3691 | 14.92 |
| Full + EMA | 2B | 2,176,538,624 | 0.3180 / 0.3632 | 22.30 |
| Full + EMA | 5B | 4,972,626,176 | 0.3146 / 0.3583 | 48.74 |

All semantic, texture, and total losses were finite. A real 1B LoRA
adapter-only save/resume/export and a real 1B full ZeRO-2 + EMA
save/resume/portable-export were also loaded through the inference API.

## Dataset contract

The standard demo build contains 64 paired rows:

- `train`: 56 rows;
- `validation`: 8 rows.

Both splits are divisible by eight for the default eight-GPU smoke run. Every
row has one embedded source JPEG and two non-empty English captions. Training
selects `enhanced_prompt` and `prompt` with deterministic 4:1 weights; the
choice is a SHA-256 function of sampler seed, epoch, and stable row id, so a
fresh-process resume reproduces the following caption sequence without relying
on worker RNG state.

The Parquet stores the original JPEG bytes. Source images must be square and at
least 1024px on each side. The shared runtime transform resizes the shortest
side, center-crops to 1024x1024, converts to RGB, and normalizes to `[-1, 1]`.
There are no aspect-ratio buckets, random crops, or dynamic resolutions.

The standard build is derived only from:

```text
repository: ma-xu/fine-t2i
revision:   28fdd5663ee202b5cafc01d6ed08a03f14957854
subset:     synthetic_enhanced_prompt_random_resolution
shard:      train-000000.tar
SHA-256:    7ce0e0bfc97f5493d457033e63f86346155c9a1d45715f9d1856e0fc3c98c738
```

The builder requires a complete `.jpg`/`.json`/`.txt` triple, keeps only rows
whose `image_generator` is exactly `Z-Image-Turbo`, rejects non-square or
sub-1024 images, excludes the eight canonical SemVAE demo source ids and the
stable quality-review denylist, and deduplicates id, image hash, and normalized
prompt. It then orders candidates by a fixed SHA-256 selection key and takes
the standard 56/8 split. On the pinned shard, the released filter produces 146
eligible candidates.

FLUX.2-dev rows are deliberately excluded. Fine-T2I declares Apache-2.0 at the
dataset level and the Z-Image-Turbo model page declares Apache-2.0, but those
labels do not replace a release review of the selected images, prompts, and
applicable terms.

## Prepare the pinned local source

Run all commands from the repository root. The scripts are intentionally
offline: obtain the pinned tar first, for example with the Hugging Face CLI:

```bash
hf download ma-xu/fine-t2i \
  synthetic_enhanced_prompt_random_resolution/train-000000.tar \
  --repo-type dataset \
  --revision 28fdd5663ee202b5cafc01d6ed08a03f14957854 \
  --local-dir outputs/demo_data/fine_t2i_upstream
```

Confirm the file before building:

```bash
sha256sum \
  outputs/demo_data/fine_t2i_upstream/synthetic_enhanced_prompt_random_resolution/train-000000.tar
```

The expected digest is the value recorded above. The builder also checks it
and fails before reading samples when it differs.

The data tools require Pillow and PyArrow; loading the resulting training split
also requires `datasets`, PyTorch, and torchvision. The checked build used
PyArrow 19.0.1 and datasets 4.4.1.

## Build locally

The standard command is:

```bash
python demo/dit_finetune/build_demo_dataset.py
```

It writes only under:

```text
outputs/hf_datasets/SeFi-Image-DiT-Finetune-Demo/
├── README.md
├── manifest.json
├── selection_manifest.json
└── data/
    ├── train-00000-of-00001.parquet
    └── validation-00000-of-00001.parquet
```

`outputs/` is ignored by Git. The builder stages files in a temporary sibling
directory and atomically installs the completed directory. It refuses to
replace an existing output. To compare a rebuild, provide another location
instead of deleting the first result:

```bash
python demo/dit_finetune/build_demo_dataset.py \
  --output-dir outputs/hf_datasets/SeFi-Image-DiT-Finetune-Demo-rebuild
```

`selection_manifest.json` records the selection algorithm, seed, filters, ids,
dimensions, and source member names. `manifest.json` records provenance,
filter counts, split sizes, caption policy, the exact SemVAE and quality
exclusion sets, and the row count and SHA-256 of each generated Parquet file.
The Parquet hashes are integrity checks for that build; pin the checked PyArrow
version when byte-identical files across machines are required.

The eight canonical SemVAE source ids are embedded in the script so the
standard command works in a clean clone and does not depend on an ignored
`outputs/` manifest. `--semvae-manifest` replaces that set and is intended for
fixture/audit work; it changes the recorded dataset contract and must not be
used for the standard release unnoticed. Likewise, row-count, selection-seed,
and expected-SHA overrides create a nonstandard build.

## Validate before training

Validate every row against the same pinned tar:

```bash
python demo/dit_finetune/validate_demo_dataset.py \
  --report outputs/validation/dit_finetune_dataset.json
```

The validator checks:

- standard 56/8 row counts and exact selection-manifest order;
- generated Parquet SHA-256 values before decoding rows;
- unique ids, image hashes, and prompts with no split overlap;
- no overlap with the canonical SemVAE demo or quality-review ids;
- byte-for-byte equality with each pinned upstream JPEG;
- upstream JSON generator, prompt, enhanced prompt, and `.txt` caption;
- square source dimensions of at least 1024px;
- decodability, finite fixed-1024 transform output, provenance, and text JSON;
- an explicit `release_review.status=pending` marker.

A passing report is an integrity and data-contract check, not publication
approval.

## Expected Hugging Face schema

The Parquet includes Hugging Face feature metadata, so `image` loads as an
`Image` when decoding is enabled and remains compatible with raw
`{"bytes", "path"}` consumers.

| Field | Type | Meaning |
| --- | --- | --- |
| `id` | string | Stable Fine-T2I sample id. |
| `image` | HF `Image` / struct | Embedded, unmodified JPEG bytes and null path. |
| `caption` | string | Enhanced prompt for generic dataset consumers. |
| `prompt` | string | Short source prompt. |
| `enhanced_prompt` | string | Detailed source prompt. |
| `origin_caption` | string | Pinned upstream `.txt` content. |
| `text` | JSON string | Exact `enhanced_prompt` and `prompt` mapping for selector compatibility. |
| `width`, `height` | int32 | Decoded source dimensions. |
| `image_sha256` | string | SHA-256 of embedded JPEG bytes. |
| `source_repo` | string | `ma-xu/fine-t2i`. |
| `source_revision` | string | Pinned 40-character upstream commit. |
| `source_subset`, `source_shard` | string | Pinned subset and tar shard. |
| `source_sample_id` | string | Source id; equal to `id`. |
| `image_generator` | string | Always `Z-Image-Turbo` in the standard build. |
| `prompt_generator` | string | Upstream prompt generator. |
| `style`, `prompt_category` | string | Upstream descriptive metadata. |
| `aesthetic_score` | float32 | Upstream aesthetic-predictor score. |
| `source_license` | string | Recorded upstream license label. |
| `license_evidence_url` | string | Z-Image-Turbo model/license evidence page. |
| `ai_generated` | bool | Always true. |

Load the pinned public result through the same API used by LoRA and full tuning:

```python
from sefi.training.data import build_paired_dataloader

dataloader, cursor = build_paired_dataloader(
    "SeFi-Image/SeFi-Image-DiT-Finetune-Demo",
    split="train",
    revision="475e616299b8c623c8a36ea5222a0e26bae91f06",
    resolution=1024,
    batch_size=1,
    sampler_seed=4237,
    rank=0,
    world_size=1,
)

batch = next(iter(dataloader))
print(batch["pixel_values"].shape)  # [1, 3, 1024, 1024]
print(batch["captions"], batch["caption_keys"], batch["sample_ids"])
print(cursor.state_dict())
```

For distributed resume, save `cursor.state_dict()`, reconstruct the dataloader,
call `apply_data_cursor(dataloader, restored_cursor)`, and skip exactly the
saved `batch_offset` batches. The deterministic sampler epoch and caption hash
then reproduce the next sample/caption sequence.

## Rebuilding the public fixture

The builder deliberately emits `release_review.status=pending`, and the
validator requires that marker. A rebuilt release must be reviewed for license,
IP, privacy, content safety, watermarks, malformed pairs, company identifiers,
credentials, and embedded metadata; uploaded privately first; reloaded from a
clean cache at its exact Hub SHA; and only then made public. Released configs
must pin the resulting immutable SHA.

Neither `build_demo_dataset.py` nor `validate_demo_dataset.py` implements or
attempts publication.
