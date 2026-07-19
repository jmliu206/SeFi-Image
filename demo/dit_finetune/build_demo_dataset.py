#!/usr/bin/env python3
"""Build the local, paired Fine-T2I demo Parquet dataset without uploading it."""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
import shutil
import sys
import tarfile
import tempfile
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any, Collection

from PIL import Image


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


DATASET_REPO_ID = "SeFi-Image/SeFi-Image-DiT-Finetune-Demo"
UPSTREAM_REPO_ID = "ma-xu/fine-t2i"
UPSTREAM_REVISION = "28fdd5663ee202b5cafc01d6ed08a03f14957854"
UPSTREAM_SUBSET = "synthetic_enhanced_prompt_random_resolution"
UPSTREAM_SHARD = "train-000000.tar"
UPSTREAM_SHARD_SHA256 = "7ce0e0bfc97f5493d457033e63f86346155c9a1d45715f9d1856e0fc3c98c738"
UPSTREAM_CARD_URL = (
    "https://huggingface.co/datasets/ma-xu/fine-t2i/blob/"
    f"{UPSTREAM_REVISION}/README.md"
)
Z_IMAGE_LICENSE_URL = "https://huggingface.co/Tongyi-MAI/Z-Image-Turbo"
EXPECTED_IMAGE_GENERATOR = "Z-Image-Turbo"
SELECTION_SEED = 20260719
DEFAULT_TRAIN_ROWS = 56
DEFAULT_VALIDATION_ROWS = 8
DEFAULT_TAR = (
    REPO_ROOT
    / "outputs/demo_data/fine_t2i_upstream"
    / UPSTREAM_SUBSET
    / UPSTREAM_SHARD
)
SEMVAE_DEMO_SOURCE_IDS = frozenset(
    {
        "00329147-fd9a-4ded-a6a4-48942165a41c",
        "00d58a94-744e-4875-8df8-923e9ff48eb1",
        "018a1a03-741c-4e71-8470-35ba52e3c8b0",
        "025656d4-ac70-436e-8c06-1075633cf6fe",
        "056b1b1b-f8e3-4cef-af03-d08056822fca",
        "05c88ac7-49fb-4d62-89f6-d4cb4c24a34d",
        "06c3d127-c3ea-495e-9bd2-10f9a02a3df3",
        "0741c70b-280a-4460-832b-f01a1eb4bd8d",
    }
)
# Public fixture quality review: exclude malformed/instruction-residue prompts,
# obvious image/text mismatches, and visible signature-like marks. Keeping the
# stable source ids here makes clean-clone rebuilds deterministic.
QUALITY_EXCLUDED_SOURCE_IDS = frozenset(
    {
        "0f49b199-719d-4a0a-bf05-735c386913fc",
        "38c25018-21c4-466a-a9fd-38c196739be2",
        "4ca798d1-f13f-43ba-a1ab-72a6f6e05a9c",
        "5b25aa78-2dc2-45dc-8bfb-9fc5f260862e",
        "a28a0431-9825-47ed-909c-884904a023cf",
        "a3e8a03a-9ccd-4247-aed7-1f0034f94af4",
        "a18e91c2-f91c-4d73-9af0-67aefe55e138",
        "a2ce1b0a-e83e-444e-b46b-1c3de48a87c2",
        "cdba46d8-7ddc-4af9-84dc-99a973ac33d4",
        "d4ced6e5-af80-4331-b5db-b691b890b05d",
        "e10c025c-8e1b-4330-9546-5a5112947a60",
    }
)
DEFAULT_SEMVAE_MANIFEST = None
DEFAULT_OUTPUT_DIR = REPO_ROOT / "outputs/hf_datasets/SeFi-Image-DiT-Finetune-Demo"


@dataclass(frozen=True)
class Candidate:
    sample_id: str
    image_member: str
    json_member: str
    text_member: str
    width: int
    height: int
    image_sha256: str
    origin_caption: str
    metadata: dict[str, Any]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_member(archive: tarfile.TarFile, member: tarfile.TarInfo) -> bytes:
    handle = archive.extractfile(member)
    if handle is None:
        raise ValueError(f"Could not read regular tar member {member.name!r}.")
    return handle.read()


def _index_triples(archive: tarfile.TarFile) -> dict[str, dict[str, tarfile.TarInfo]]:
    triples: dict[str, dict[str, tarfile.TarInfo]] = {}
    for member in archive.getmembers():
        if not member.isfile():
            continue
        path = PurePosixPath(member.name)
        if len(path.parts) != 1 or path.name in {"", ".", ".."}:
            raise ValueError(f"Fine-T2I tar must contain flat, safe member names: {member.name!r}.")
        suffix = path.suffix.lower()
        if suffix not in {".jpg", ".json", ".txt"}:
            continue
        sample_id = path.stem
        per_sample = triples.setdefault(sample_id, {})
        if suffix in per_sample:
            raise ValueError(f"Duplicate {suffix} member for sample {sample_id!r}.")
        per_sample[suffix] = member
    return triples


def load_excluded_semvae_ids(path: Path | None) -> set[str]:
    if path is None:
        return set(SEMVAE_DEMO_SOURCE_IDS)
    if not path.is_file():
        raise FileNotFoundError(
            f"SemVAE manifest is required to prevent demo overlap: {path}"
        )
    payload = json.loads(path.read_text(encoding="utf-8"))
    samples = payload.get("samples")
    if not isinstance(samples, list) or not samples:
        raise ValueError(f"SemVAE manifest has no non-empty samples list: {path}")
    excluded: set[str] = set()
    for sample in samples:
        if not isinstance(sample, dict):
            raise ValueError("Each SemVAE manifest sample must be an object.")
        sample_id = str(sample.get("source_sample_id", sample.get("id", ""))).strip()
        if not sample_id:
            raise ValueError("SemVAE manifest sample has no id/source_sample_id.")
        excluded.add(sample_id)
    return excluded


def collect_candidates(
    tar_path: Path,
    *,
    excluded_ids: set[str],
    quality_excluded_ids: Collection[str] = QUALITY_EXCLUDED_SOURCE_IDS,
    minimum_side: int = 1024,
) -> tuple[list[Candidate], dict[str, int]]:
    counters = {
        "tar_sample_ids": 0,
        "complete_triples": 0,
        "z_image_turbo": 0,
        "square_minimum_1024": 0,
        "excluded_semvae": 0,
        "excluded_quality": 0,
        "duplicate_id_or_hash_or_prompt": 0,
        "eligible": 0,
    }
    candidates: list[Candidate] = []
    seen_ids: set[str] = set()
    seen_hashes: set[str] = set()
    seen_prompts: set[str] = set()

    with tarfile.open(tar_path, mode="r:") as archive:
        triples = _index_triples(archive)
        counters["tar_sample_ids"] = len(triples)
        for sample_id in sorted(triples):
            members = triples[sample_id]
            if set(members) != {".jpg", ".json", ".txt"}:
                continue
            counters["complete_triples"] += 1
            try:
                metadata = json.loads(_read_member(archive, members[".json"]))
            except (UnicodeDecodeError, json.JSONDecodeError) as exc:
                raise ValueError(f"Invalid JSON metadata for sample {sample_id!r}.") from exc
            if not isinstance(metadata, dict):
                raise ValueError(f"Metadata for sample {sample_id!r} must be an object.")
            metadata_id = str(metadata.get("id", "")).strip()
            if metadata_id != sample_id:
                raise ValueError(
                    f"Metadata/tar id mismatch: member={sample_id!r}, metadata={metadata_id!r}."
                )
            if metadata.get("image_generator") != EXPECTED_IMAGE_GENERATOR:
                continue
            counters["z_image_turbo"] += 1
            if sample_id in excluded_ids:
                counters["excluded_semvae"] += 1
                continue
            if sample_id in quality_excluded_ids:
                counters["excluded_quality"] += 1
                continue
            prompt = str(metadata.get("prompt", "")).strip()
            enhanced_prompt = str(metadata.get("enhanced_prompt", "")).strip()
            origin_caption = _read_member(archive, members[".txt"]).decode("utf-8").strip()
            if not prompt or not enhanced_prompt or not origin_caption:
                continue
            image_bytes = _read_member(archive, members[".jpg"])
            try:
                with Image.open(io.BytesIO(image_bytes)) as image:
                    image.load()
                    width, height = image.size
            except Exception as exc:
                raise ValueError(f"Invalid JPEG for sample {sample_id!r}: {exc}") from exc
            if width != height or width < minimum_side:
                continue
            counters["square_minimum_1024"] += 1
            declared_resolution = metadata.get("image_resolution")
            if (
                not isinstance(declared_resolution, list)
                or len(declared_resolution) != 2
                or tuple(int(value) for value in declared_resolution) != (width, height)
            ):
                raise ValueError(
                    f"Image dimensions disagree with metadata for {sample_id!r}: "
                    f"decoded={(width, height)}, declared={declared_resolution!r}."
                )
            image_sha256 = hashlib.sha256(image_bytes).hexdigest()
            normalized_prompt = " ".join(prompt.casefold().split())
            if (
                sample_id in seen_ids
                or image_sha256 in seen_hashes
                or normalized_prompt in seen_prompts
            ):
                counters["duplicate_id_or_hash_or_prompt"] += 1
                continue
            seen_ids.add(sample_id)
            seen_hashes.add(image_sha256)
            seen_prompts.add(normalized_prompt)
            candidates.append(
                Candidate(
                    sample_id=sample_id,
                    image_member=members[".jpg"].name,
                    json_member=members[".json"].name,
                    text_member=members[".txt"].name,
                    width=width,
                    height=height,
                    image_sha256=image_sha256,
                    origin_caption=origin_caption,
                    metadata=metadata,
                )
            )
    counters["eligible"] = len(candidates)
    return candidates, counters


def select_candidates(
    candidates: list[Candidate],
    *,
    train_rows: int,
    validation_rows: int,
    seed: int = SELECTION_SEED,
) -> dict[str, list[Candidate]]:
    requested = int(train_rows) + int(validation_rows)
    if train_rows <= 0 or validation_rows <= 0:
        raise ValueError("Both train_rows and validation_rows must be positive.")
    if len(candidates) < requested:
        raise ValueError(f"Need {requested} eligible samples, found only {len(candidates)}.")

    def rank(candidate: Candidate) -> tuple[str, str]:
        digest = hashlib.sha256(
            f"sefi-dit-demo-v1\0{int(seed)}\0{candidate.sample_id}".encode("utf-8")
        ).hexdigest()
        return digest, candidate.sample_id

    selected = sorted(candidates, key=rank)[:requested]
    return {
        "train": selected[: int(train_rows)],
        "validation": selected[int(train_rows) :],
    }


def _hf_schema_metadata() -> dict[bytes, bytes]:
    value_fields = {
        "id": "string",
        "caption": "string",
        "prompt": "string",
        "enhanced_prompt": "string",
        "origin_caption": "string",
        "text": "string",
        "width": "int32",
        "height": "int32",
        "image_sha256": "string",
        "source_repo": "string",
        "source_revision": "string",
        "source_subset": "string",
        "source_shard": "string",
        "source_sample_id": "string",
        "image_generator": "string",
        "prompt_generator": "string",
        "style": "string",
        "prompt_category": "string",
        "aesthetic_score": "float32",
        "source_license": "string",
        "license_evidence_url": "string",
        "ai_generated": "bool",
    }
    features = {
        key: {"dtype": dtype, "_type": "Value"} for key, dtype in value_fields.items()
    }
    features["image"] = {"_type": "Image"}
    payload = {"info": {"features": features}}
    return {b"huggingface": json.dumps(payload, sort_keys=True).encode("utf-8")}


def _rows_for_split(
    archive: tarfile.TarFile,
    selected: list[Candidate],
) -> list[dict[str, Any]]:
    member_index = {member.name: member for member in archive.getmembers()}
    rows: list[dict[str, Any]] = []
    for candidate in selected:
        image_bytes = _read_member(archive, member_index[candidate.image_member])
        if hashlib.sha256(image_bytes).hexdigest() != candidate.image_sha256:
            raise ValueError(f"Image changed during tar read for {candidate.sample_id!r}.")
        metadata = candidate.metadata
        prompt = str(metadata["prompt"]).strip()
        enhanced_prompt = str(metadata["enhanced_prompt"]).strip()
        rows.append(
            {
                "id": candidate.sample_id,
                "image": {"bytes": image_bytes, "path": None},
                "caption": enhanced_prompt,
                "prompt": prompt,
                "enhanced_prompt": enhanced_prompt,
                "origin_caption": candidate.origin_caption,
                "text": json.dumps(
                    {"enhanced_prompt": enhanced_prompt, "prompt": prompt},
                    ensure_ascii=False,
                    sort_keys=True,
                ),
                "width": candidate.width,
                "height": candidate.height,
                "image_sha256": candidate.image_sha256,
                "source_repo": UPSTREAM_REPO_ID,
                "source_revision": UPSTREAM_REVISION,
                "source_subset": UPSTREAM_SUBSET,
                "source_shard": UPSTREAM_SHARD,
                "source_sample_id": candidate.sample_id,
                "image_generator": EXPECTED_IMAGE_GENERATOR,
                "prompt_generator": str(metadata.get("prompt_generator", "")),
                "style": str(metadata.get("style", "")),
                "prompt_category": str(metadata.get("prompt_category", "")),
                "aesthetic_score": float(metadata.get("aesthetic_predictor_v_2_5_score", 0.0)),
                "source_license": "Apache-2.0",
                "license_evidence_url": Z_IMAGE_LICENSE_URL,
                "ai_generated": True,
            }
        )
    return rows


def _write_parquet(path: Path, rows: list[dict[str, Any]]) -> None:
    try:
        import pyarrow as pa
        import pyarrow.parquet as pq
    except ImportError as exc:
        raise RuntimeError("Building the demo dataset requires pyarrow.") from exc
    schema = pa.schema(
        [
            pa.field("id", pa.string()),
            pa.field(
                "image",
                pa.struct([pa.field("bytes", pa.binary()), pa.field("path", pa.string())]),
            ),
            pa.field("caption", pa.string()),
            pa.field("prompt", pa.string()),
            pa.field("enhanced_prompt", pa.string()),
            pa.field("origin_caption", pa.string()),
            pa.field("text", pa.string()),
            pa.field("width", pa.int32()),
            pa.field("height", pa.int32()),
            pa.field("image_sha256", pa.string()),
            pa.field("source_repo", pa.string()),
            pa.field("source_revision", pa.string()),
            pa.field("source_subset", pa.string()),
            pa.field("source_shard", pa.string()),
            pa.field("source_sample_id", pa.string()),
            pa.field("image_generator", pa.string()),
            pa.field("prompt_generator", pa.string()),
            pa.field("style", pa.string()),
            pa.field("prompt_category", pa.string()),
            pa.field("aesthetic_score", pa.float32()),
            pa.field("source_license", pa.string()),
            pa.field("license_evidence_url", pa.string()),
            pa.field("ai_generated", pa.bool_()),
        ],
        metadata=_hf_schema_metadata(),
    )
    table = pa.Table.from_pylist(rows, schema=schema)
    pq.write_table(table, path, compression="zstd", use_dictionary=True)


def _selection_record(candidate: Candidate) -> dict[str, Any]:
    return {
        "id": candidate.sample_id,
        "image_sha256": candidate.image_sha256,
        "width": candidate.width,
        "height": candidate.height,
        "image_member": candidate.image_member,
        "json_member": candidate.json_member,
        "text_member": candidate.text_member,
    }


def _dataset_card(train_rows: int, validation_rows: int) -> str:
    return f"""---
license: apache-2.0
language:
- en
pretty_name: SeFi-Image DiT Fine-tuning Demo
size_categories:
- n<1K
tags:
- text-to-image
- fine-tuning
- synthetic
configs:
- config_name: default
  data_files:
  - split: train
    path: data/train-*
  - split: validation
    path: data/validation-*
---

# SeFi-Image DiT Fine-tuning Demo

This local build contains {train_rows} training and {validation_rows} validation
image-text pairs for code-path smoke tests. It is derived byte-for-byte from
`{UPSTREAM_REPO_ID}/{UPSTREAM_SUBSET}` at revision `{UPSTREAM_REVISION}` and
contains only rows whose `image_generator` is `{EXPECTED_IMAGE_GENERATOR}`.
FLUX.2-dev rows and the existing SemVAE demo ids are excluded.
Rows rejected during the fixture's prompt/image quality review are also
excluded by stable source id.

The default training caption selector chooses `enhanced_prompt` and `prompt`
with deterministic 4:1 weights. Images are square source JPEGs at least 1024px;
the training loader resizes them deterministically to 1024x1024.

**Do not publish this directory until the recorded license/IP/safety release
review is complete.** The engineering filters and upstream license metadata do
not replace that review.
"""


def build_dataset(
    *,
    tar_path: Path,
    output_dir: Path,
    semvae_manifest: Path | None = DEFAULT_SEMVAE_MANIFEST,
    expected_tar_sha256: str = UPSTREAM_SHARD_SHA256,
    train_rows: int = DEFAULT_TRAIN_ROWS,
    validation_rows: int = DEFAULT_VALIDATION_ROWS,
    selection_seed: int = SELECTION_SEED,
) -> dict[str, Any]:
    tar_path = tar_path.expanduser().absolute()
    output_dir = output_dir.expanduser().absolute()
    semvae_manifest = (
        semvae_manifest.expanduser().absolute() if semvae_manifest is not None else None
    )
    if not tar_path.is_file():
        raise FileNotFoundError(f"Pinned local Fine-T2I tar not found: {tar_path}")
    actual_tar_sha256 = sha256_file(tar_path)
    if actual_tar_sha256 != expected_tar_sha256:
        raise ValueError(
            "Fine-T2I tar SHA-256 mismatch: "
            f"expected={expected_tar_sha256}, actual={actual_tar_sha256}, path={tar_path}."
        )
    if output_dir.exists():
        raise FileExistsError(
            f"Output already exists: {output_dir}. Move it aside before rebuilding."
        )
    excluded_ids = load_excluded_semvae_ids(semvae_manifest)
    candidates, counters = collect_candidates(
        tar_path,
        excluded_ids=excluded_ids,
        minimum_side=1024,
    )
    selected = select_candidates(
        candidates,
        train_rows=train_rows,
        validation_rows=validation_rows,
        seed=selection_seed,
    )

    output_dir.parent.mkdir(parents=True, exist_ok=True)
    staging_dir = Path(
        tempfile.mkdtemp(prefix=f".{output_dir.name}.tmp-", dir=output_dir.parent)
    )
    try:
        data_dir = staging_dir / "data"
        data_dir.mkdir()
        with tarfile.open(tar_path, mode="r:") as archive:
            for split, split_candidates in selected.items():
                rows = _rows_for_split(archive, split_candidates)
                _write_parquet(data_dir / f"{split}-00000-of-00001.parquet", rows)

        excluded_ids_sha256 = hashlib.sha256(
            ("\n".join(sorted(excluded_ids)) + "\n").encode("utf-8")
        ).hexdigest()
        selection_manifest = {
            "version": 1,
            "algorithm": "sha256(sefi-dit-demo-v1, selection_seed, sample_id)",
            "selection_seed": int(selection_seed),
            "filters": {
                "complete_jpg_json_txt": True,
                "image_generator": EXPECTED_IMAGE_GENERATOR,
                "square": True,
                "minimum_side": 1024,
                "exclude_semvae_ids": len(excluded_ids),
                "exclude_semvae_ids_sha256": excluded_ids_sha256,
                "exclude_quality_ids": len(QUALITY_EXCLUDED_SOURCE_IDS),
                "exclude_quality_ids_sha256": hashlib.sha256(
                    ("\n".join(sorted(QUALITY_EXCLUDED_SOURCE_IDS)) + "\n").encode(
                        "utf-8"
                    )
                ).hexdigest(),
                "deduplicate": ["id", "image_sha256", "normalized_prompt"],
            },
            "splits": {
                split: [_selection_record(candidate) for candidate in split_candidates]
                for split, split_candidates in selected.items()
            },
        }
        artifacts = {
            f"data/{split}-00000-of-00001.parquet": {
                "rows": len(split_candidates),
                "sha256": sha256_file(
                    data_dir / f"{split}-00000-of-00001.parquet"
                ),
            }
            for split, split_candidates in selected.items()
        }
        manifest = {
            "dataset": DATASET_REPO_ID,
            "format_version": 1,
            "purpose": "paired 1024px DiT fine-tuning smoke test",
            "rows": {split: len(rows) for split, rows in selected.items()},
            "artifacts": artifacts,
            "caption_selection": {
                "mode": "weighted",
                "weights": {"enhanced_prompt": 4.0, "prompt": 1.0},
                "hash_inputs": ["sampler_seed", "epoch", "row_id"],
            },
            "release_review": {
                "status": "pending",
                "required_before_upload": [
                    "license",
                    "IP",
                    "privacy",
                    "content_safety",
                    "watermarks",
                    "company_identity",
                    "credentials",
                    "embedded_metadata",
                    "pair_quality",
                ],
            },
            "upstream": {
                "repo_id": UPSTREAM_REPO_ID,
                "revision": UPSTREAM_REVISION,
                "subset": UPSTREAM_SUBSET,
                "shard": UPSTREAM_SHARD,
                "shard_sha256": actual_tar_sha256,
                "dataset_card": UPSTREAM_CARD_URL,
            },
            "filter_counts": counters,
            "excluded_semvae_ids": sorted(excluded_ids),
            "excluded_semvae_ids_sha256": excluded_ids_sha256,
            "excluded_quality_ids": sorted(QUALITY_EXCLUDED_SOURCE_IDS),
            "excluded_quality_ids_sha256": hashlib.sha256(
                ("\n".join(sorted(QUALITY_EXCLUDED_SOURCE_IDS)) + "\n").encode(
                    "utf-8"
                )
            ).hexdigest(),
        }
        (staging_dir / "selection_manifest.json").write_text(
            json.dumps(selection_manifest, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        (staging_dir / "manifest.json").write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        (staging_dir / "README.md").write_text(
            _dataset_card(train_rows, validation_rows), encoding="utf-8"
        )
        os.replace(staging_dir, output_dir)
    except BaseException:
        if staging_dir.exists():
            shutil.rmtree(staging_dir)
        raise
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build the paired 1024px Fine-T2I demo dataset locally (no upload)."
    )
    parser.add_argument(
        "--tar",
        type=Path,
        default=DEFAULT_TAR,
        help="Pinned local Fine-T2I train-000000.tar; the script never downloads it.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="New local dataset directory; the builder refuses to overwrite it.",
    )
    parser.add_argument(
        "--semvae-manifest",
        type=Path,
        default=DEFAULT_SEMVAE_MANIFEST,
        help=(
            "Optional manifest whose source ids replace the embedded canonical "
            "SemVAE demo exclusion set."
        ),
    )
    parser.add_argument(
        "--expected-tar-sha256",
        default=UPSTREAM_SHARD_SHA256,
        help="Pinned standard SHA-256; overriding it creates a nonstandard build.",
    )
    parser.add_argument(
        "--train-rows",
        type=int,
        default=DEFAULT_TRAIN_ROWS,
        help="Standard release count is 56; other values are for fixtures only.",
    )
    parser.add_argument(
        "--validation-rows",
        type=int,
        default=DEFAULT_VALIDATION_ROWS,
        help="Standard release count is 8; other values are for fixtures only.",
    )
    parser.add_argument(
        "--selection-seed",
        type=int,
        default=SELECTION_SEED,
        help="Fixed deterministic selection seed; overriding changes dataset identity.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    manifest = build_dataset(
        tar_path=args.tar,
        output_dir=args.output_dir,
        semvae_manifest=args.semvae_manifest,
        expected_tar_sha256=args.expected_tar_sha256,
        train_rows=args.train_rows,
        validation_rows=args.validation_rows,
        selection_seed=args.selection_seed,
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))
    print(f"Built local dataset: {args.output_dir}")
    print("Upload intentionally not performed; release review remains pending.")


if __name__ == "__main__":
    main()
