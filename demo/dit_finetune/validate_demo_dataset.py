#!/usr/bin/env python3
"""Validate the local paired demo dataset against its pinned Fine-T2I tar."""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import sys
import tarfile
from pathlib import Path
from typing import Any

import torch
from PIL import Image


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from demo.dit_finetune.build_demo_dataset import (  # noqa: E402
    DEFAULT_OUTPUT_DIR,
    DEFAULT_SEMVAE_MANIFEST,
    DEFAULT_TAR,
    DEFAULT_TRAIN_ROWS,
    DEFAULT_VALIDATION_ROWS,
    EXPECTED_IMAGE_GENERATOR,
    QUALITY_EXCLUDED_SOURCE_IDS,
    UPSTREAM_REPO_ID,
    UPSTREAM_REVISION,
    UPSTREAM_SHARD,
    UPSTREAM_SHARD_SHA256,
    UPSTREAM_SUBSET,
    load_excluded_semvae_ids,
    sha256_file,
)
def _load_rows(path: Path) -> list[dict[str, Any]]:
    try:
        import pyarrow.parquet as pq
    except ImportError as exc:
        raise RuntimeError("Validating the demo dataset requires pyarrow.") from exc
    return pq.read_table(path).to_pylist()


def _image_bytes(value: Any) -> bytes:
    if isinstance(value, dict):
        value = value.get("bytes")
    if not isinstance(value, (bytes, bytearray, memoryview)):
        raise TypeError(f"Expected embedded image bytes, got {type(value).__name__}.")
    return bytes(value)


def _tar_selected_records(tar_path: Path, selected_ids: set[str]) -> dict[str, dict[str, bytes]]:
    result: dict[str, dict[str, bytes]] = {}
    with tarfile.open(tar_path, mode="r:") as archive:
        for member in archive.getmembers():
            path = Path(member.name)
            suffix = path.suffix.lower()
            if (
                not member.isfile()
                or suffix not in {".jpg", ".json", ".txt"}
                or path.stem not in selected_ids
            ):
                continue
            handle = archive.extractfile(member)
            if handle is None:
                raise ValueError(f"Could not read selected tar member {member.name!r}.")
            record = result.setdefault(path.stem, {})
            if suffix in record:
                raise ValueError(f"Duplicate selected {suffix} in tar: {path.stem!r}.")
            record[suffix] = handle.read()
    missing = selected_ids - result.keys()
    if missing:
        raise ValueError(f"Selected source records missing from tar: {sorted(missing)}")
    for sample_id, record in result.items():
        if set(record) != {".jpg", ".json", ".txt"}:
            raise ValueError(f"Selected source triple is incomplete for {sample_id!r}.")
    return result


def validate_dataset(
    *,
    dataset_dir: Path,
    tar_path: Path,
    semvae_manifest: Path | None = DEFAULT_SEMVAE_MANIFEST,
    expected_tar_sha256: str = UPSTREAM_SHARD_SHA256,
    expected_train_rows: int = DEFAULT_TRAIN_ROWS,
    expected_validation_rows: int = DEFAULT_VALIDATION_ROWS,
    resolution: int = 1024,
) -> dict[str, Any]:
    from sefi.training.data import FixedSquareImageTransform, decode_image

    dataset_dir = dataset_dir.expanduser().absolute()
    tar_path = tar_path.expanduser().absolute()
    if not dataset_dir.is_dir():
        raise FileNotFoundError(f"Dataset directory not found: {dataset_dir}")
    if not tar_path.is_file():
        raise FileNotFoundError(f"Pinned upstream tar not found: {tar_path}")
    actual_tar_sha256 = sha256_file(tar_path)
    if actual_tar_sha256 != expected_tar_sha256:
        raise ValueError(
            f"Upstream tar SHA mismatch: expected={expected_tar_sha256}, "
            f"actual={actual_tar_sha256}."
        )

    manifest_path = dataset_dir / "manifest.json"
    selection_path = dataset_dir / "selection_manifest.json"
    if not manifest_path.is_file() or not selection_path.is_file():
        raise FileNotFoundError("Dataset requires manifest.json and selection_manifest.json.")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    selection = json.loads(selection_path.read_text(encoding="utf-8"))
    if manifest.get("release_review", {}).get("status") != "pending":
        raise ValueError("Local pre-upload build must explicitly retain release_review=pending.")
    upstream = manifest.get("upstream", {})
    expected_upstream = {
        "repo_id": UPSTREAM_REPO_ID,
        "revision": UPSTREAM_REVISION,
        "subset": UPSTREAM_SUBSET,
        "shard": UPSTREAM_SHARD,
        "shard_sha256": actual_tar_sha256,
    }
    for key, expected in expected_upstream.items():
        if upstream.get(key) != expected:
            raise ValueError(
                f"Manifest upstream {key} mismatch: {upstream.get(key)!r} != {expected!r}."
            )

    split_expectations = {
        "train": int(expected_train_rows),
        "validation": int(expected_validation_rows),
    }
    rows_by_split: dict[str, list[dict[str, Any]]] = {}
    for split, expected_rows in split_expectations.items():
        parquet_path = dataset_dir / "data" / f"{split}-00000-of-00001.parquet"
        if not parquet_path.is_file():
            raise FileNotFoundError(f"Missing split Parquet: {parquet_path}")
        relative_path = f"data/{split}-00000-of-00001.parquet"
        artifact = manifest.get("artifacts", {}).get(relative_path, {})
        if artifact.get("rows") != expected_rows:
            raise ValueError(f"Manifest artifact row count mismatch for {relative_path!r}.")
        parquet_sha256 = sha256_file(parquet_path)
        if artifact.get("sha256") != parquet_sha256:
            raise ValueError(
                f"Parquet artifact SHA mismatch for {relative_path!r}: "
                f"manifest={artifact.get('sha256')!r}, actual={parquet_sha256!r}."
            )
        rows = _load_rows(parquet_path)
        if len(rows) != expected_rows:
            raise ValueError(f"Split {split!r} has {len(rows)} rows, expected {expected_rows}.")
        if manifest.get("rows", {}).get(split) != expected_rows:
            raise ValueError(f"Manifest row count mismatch for split {split!r}.")
        selected_ids = [str(item["id"]) for item in selection["splits"][split]]
        row_ids = [str(row["id"]) for row in rows]
        if row_ids != selected_ids:
            raise ValueError(f"Parquet row order/ids disagree with selection for {split!r}.")
        rows_by_split[split] = rows

    all_rows = rows_by_split["train"] + rows_by_split["validation"]
    semvae_manifest = (
        semvae_manifest.expanduser().absolute() if semvae_manifest is not None else None
    )
    excluded_ids = load_excluded_semvae_ids(semvae_manifest)
    excluded_ids_sha256 = hashlib.sha256(
        ("\n".join(sorted(excluded_ids)) + "\n").encode("utf-8")
    ).hexdigest()
    if manifest.get("excluded_semvae_ids") != sorted(excluded_ids):
        raise ValueError("Manifest SemVAE exclusion ids differ from the configured source.")
    if manifest.get("excluded_semvae_ids_sha256") != excluded_ids_sha256:
        raise ValueError("Manifest SemVAE exclusion hash is invalid.")
    quality_excluded_ids = set(QUALITY_EXCLUDED_SOURCE_IDS)
    quality_excluded_ids_sha256 = hashlib.sha256(
        ("\n".join(sorted(quality_excluded_ids)) + "\n").encode("utf-8")
    ).hexdigest()
    if manifest.get("excluded_quality_ids") != sorted(quality_excluded_ids):
        raise ValueError("Manifest quality exclusion ids differ from the build contract.")
    if manifest.get("excluded_quality_ids_sha256") != quality_excluded_ids_sha256:
        raise ValueError("Manifest quality exclusion hash is invalid.")
    selection_filters = selection.get("filters", {})
    if (
        selection_filters.get("exclude_semvae_ids") != len(excluded_ids)
        or selection_filters.get("exclude_semvae_ids_sha256") != excluded_ids_sha256
    ):
        raise ValueError("Selection manifest SemVAE exclusion contract is invalid.")
    if (
        selection_filters.get("exclude_quality_ids") != len(quality_excluded_ids)
        or selection_filters.get("exclude_quality_ids_sha256")
        != quality_excluded_ids_sha256
    ):
        raise ValueError("Selection manifest quality exclusion contract is invalid.")
    ids: set[str] = set()
    hashes: set[str] = set()
    prompts: set[str] = set()
    transform = FixedSquareImageTransform(resolution)
    selected_ids = {str(row["id"]) for row in all_rows}
    source_records = _tar_selected_records(tar_path, selected_ids)
    selected_records = {
        str(item["id"]): item
        for split in ("train", "validation")
        for item in selection["splits"][split]
    }

    for row in all_rows:
        sample_id = str(row.get("id", "")).strip()
        if not sample_id or row.get("source_sample_id") != sample_id:
            raise ValueError(f"Invalid id/source_sample_id for row {sample_id!r}.")
        if (
            sample_id in ids
            or sample_id in excluded_ids
            or sample_id in quality_excluded_ids
        ):
            raise ValueError(f"Duplicate or excluded id: {sample_id!r}.")
        ids.add(sample_id)
        source_record = source_records[sample_id]
        image_bytes = _image_bytes(row.get("image"))
        if image_bytes != source_record[".jpg"]:
            raise ValueError(f"Image bytes differ from pinned upstream for {sample_id!r}.")
        image_hash = hashlib.sha256(image_bytes).hexdigest()
        if row.get("image_sha256") != image_hash or image_hash in hashes:
            raise ValueError(f"Invalid or duplicate image hash for {sample_id!r}.")
        if selected_records[sample_id].get("image_sha256") != image_hash:
            raise ValueError(f"Selection manifest hash mismatch for {sample_id!r}.")
        hashes.add(image_hash)
        prompt = str(row.get("prompt", "")).strip()
        enhanced_prompt = str(row.get("enhanced_prompt", "")).strip()
        caption = str(row.get("caption", "")).strip()
        if not prompt or not enhanced_prompt or caption != enhanced_prompt:
            raise ValueError(f"Invalid paired captions for {sample_id!r}.")
        normalized_prompt = " ".join(prompt.casefold().split())
        if normalized_prompt in prompts:
            raise ValueError(f"Duplicate normalized prompt for {sample_id!r}.")
        prompts.add(normalized_prompt)
        try:
            text_payload = json.loads(row.get("text", ""))
        except json.JSONDecodeError as exc:
            raise ValueError(f"Invalid text JSON for {sample_id!r}.") from exc
        if text_payload != {"enhanced_prompt": enhanced_prompt, "prompt": prompt}:
            raise ValueError(f"Text JSON disagrees with prompt columns for {sample_id!r}.")
        try:
            source_metadata = json.loads(source_record[".json"])
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise ValueError(f"Invalid pinned source metadata for {sample_id!r}.") from exc
        if (
            source_metadata.get("id") != sample_id
            or source_metadata.get("image_generator") != EXPECTED_IMAGE_GENERATOR
            or str(source_metadata.get("prompt", "")).strip() != prompt
            or str(source_metadata.get("enhanced_prompt", "")).strip() != enhanced_prompt
            or source_record[".txt"].decode("utf-8").strip()
            != str(row.get("origin_caption", "")).strip()
        ):
            raise ValueError(f"Parquet metadata differs from pinned source for {sample_id!r}.")
        if row.get("image_generator") != EXPECTED_IMAGE_GENERATOR:
            raise ValueError(f"Disallowed image generator for {sample_id!r}.")
        if (
            row.get("source_repo") != UPSTREAM_REPO_ID
            or row.get("source_revision") != UPSTREAM_REVISION
            or row.get("source_subset") != UPSTREAM_SUBSET
            or row.get("source_shard") != UPSTREAM_SHARD
        ):
            raise ValueError(f"Invalid provenance columns for {sample_id!r}.")
        width, height = int(row.get("width", 0)), int(row.get("height", 0))
        with Image.open(io.BytesIO(image_bytes)) as image:
            image.load()
            if image.size != (width, height):
                raise ValueError(f"Stored dimensions disagree for {sample_id!r}.")
        if width != height or width < resolution:
            raise ValueError(f"Image is not square >= {resolution}: {sample_id!r}.")
        pixel_values = transform(decode_image(image_bytes))
        if pixel_values.shape != (3, resolution, resolution):
            raise ValueError(f"Unexpected transformed shape for {sample_id!r}.")
        if not torch.isfinite(pixel_values).all():
            raise ValueError(f"Non-finite transform output for {sample_id!r}.")

    if set(row["id"] for row in rows_by_split["train"]) & set(
        row["id"] for row in rows_by_split["validation"]
    ):
        raise ValueError("Train and validation ids overlap.")
    return {
        "status": "pass",
        "dataset_dir": str(dataset_dir),
        "upstream_tar_sha256": actual_tar_sha256,
        "rows": split_expectations,
        "unique_ids": len(ids),
        "unique_image_sha256": len(hashes),
        "resolution": int(resolution),
        "image_generator": EXPECTED_IMAGE_GENERATOR,
        "semvae_overlap": 0,
        "quality_exclusion_overlap": 0,
        "release_review": "pending",
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Validate local paired Parquet against the pinned Fine-T2I tar."
    )
    parser.add_argument(
        "--dataset-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Local dataset directory produced by build_demo_dataset.py.",
    )
    parser.add_argument(
        "--tar",
        type=Path,
        default=DEFAULT_TAR,
        help="Same pinned local Fine-T2I tar used for the build.",
    )
    parser.add_argument(
        "--semvae-manifest",
        type=Path,
        default=DEFAULT_SEMVAE_MANIFEST,
        help=(
            "Optional manifest used only when the build replaced the embedded "
            "canonical SemVAE exclusion set."
        ),
    )
    parser.add_argument(
        "--expected-tar-sha256",
        default=UPSTREAM_SHARD_SHA256,
        help="Expected source tar SHA-256; must match the build contract.",
    )
    parser.add_argument(
        "--expected-train-rows",
        type=int,
        default=DEFAULT_TRAIN_ROWS,
        help="Expected train rows; standard release count is 56.",
    )
    parser.add_argument(
        "--expected-validation-rows",
        type=int,
        default=DEFAULT_VALIDATION_ROWS,
        help="Expected validation rows; standard release count is 8.",
    )
    parser.add_argument(
        "--resolution",
        type=int,
        default=1024,
        help="Fixed transform resolution; standard dataset contract is 1024.",
    )
    parser.add_argument(
        "--report",
        type=Path,
        help="Optional path for the JSON validation report.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report = validate_dataset(
        dataset_dir=args.dataset_dir,
        tar_path=args.tar,
        semvae_manifest=args.semvae_manifest,
        expected_tar_sha256=args.expected_tar_sha256,
        expected_train_rows=args.expected_train_rows,
        expected_validation_rows=args.expected_validation_rows,
        resolution=args.resolution,
    )
    encoded = json.dumps(report, indent=2, sort_keys=True) + "\n"
    print(encoded, end="")
    if args.report is not None:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(encoded, encoding="utf-8")


if __name__ == "__main__":
    main()
