import hashlib
import io
import json
import tarfile
from pathlib import Path

import pytest
import torch
from PIL import Image

from demo.dit_finetune.build_demo_dataset import (
    QUALITY_EXCLUDED_SOURCE_IDS,
    SEMVAE_DEMO_SOURCE_IDS,
    build_dataset,
    load_excluded_semvae_ids,
)
from demo.dit_finetune.validate_demo_dataset import validate_dataset
from sefi.training.data import (
    DataCursor,
    PairedImageTextDataset,
    apply_data_cursor,
    build_paired_dataloader,
    collect_caption_fields,
    decode_image,
    stable_caption_choice,
)


def _jpeg_bytes(size=(32, 32), color=(20, 80, 160)) -> bytes:
    output = io.BytesIO()
    Image.new("RGB", size, color).save(output, format="JPEG", quality=90)
    return output.getvalue()


def _rows(count: int, *, size=(32, 32)) -> list[dict]:
    image = _jpeg_bytes(size)
    return [
        {
            "id": f"sample-{index:03d}",
            "image": image,
            "caption": f"enhanced {index}",
            "enhanced_prompt": f"enhanced {index}",
            "prompt": f"prompt {index}",
            "text": json.dumps(
                {"enhanced_prompt": f"enhanced {index}", "prompt": f"prompt {index}"}
            ),
        }
        for index in range(count)
    ]


def test_image_decoder_accepts_all_public_representations(tmp_path):
    image_bytes = _jpeg_bytes()
    path = tmp_path / "image.jpg"
    path.write_bytes(image_bytes)
    pil = Image.open(io.BytesIO(image_bytes))

    decoded = [
        decode_image(pil),
        decode_image(image_bytes),
        decode_image({"bytes": image_bytes, "path": None}),
        decode_image({"bytes": None, "path": str(path)}),
    ]

    assert [image.mode for image in decoded] == ["RGB"] * 4
    assert [image.size for image in decoded] == [(32, 32)] * 4


def test_clean_clone_uses_embedded_semvae_exclusion_ids():
    assert load_excluded_semvae_ids(None) == set(SEMVAE_DEMO_SOURCE_IDS)
    assert len(SEMVAE_DEMO_SOURCE_IDS) == 8
    assert len(QUALITY_EXCLUDED_SOURCE_IDS) == 11


def test_caption_fields_accept_plain_mapping_and_json_and_weighted_hash_is_stable():
    row = {
        "caption": "plain caption",
        "text": {"prompt": "short prompt"},
        "enhanced_prompt": json.dumps("long prompt"),
    }

    fields = collect_caption_fields(row)
    first = stable_caption_choice(row, row_id="row-1", seed=17, epoch=3)
    second = stable_caption_choice(row, row_id="row-1", seed=17, epoch=3)

    assert fields == {
        "caption": "plain caption",
        "prompt": "short prompt",
        "enhanced_prompt": "long prompt",
    }
    assert first == second
    assert first[0] in {"short prompt", "long prompt"}


def test_weighted_caption_distribution_is_deterministic_four_to_one():
    row = {"enhanced_prompt": "enhanced", "prompt": "prompt"}
    choices = [
        stable_caption_choice(row, row_id=f"row-{index}", seed=123, epoch=0)[1]
        for index in range(1000)
    ]

    assert 750 <= choices.count("enhanced_prompt") <= 850
    assert choices == [
        stable_caption_choice(row, row_id=f"row-{index}", seed=123, epoch=0)[1]
        for index in range(1000)
    ]


def test_map_style_distributed_resume_replays_sample_and_caption_sequence():
    rows = _rows(16)
    loader, cursor = build_paired_dataloader(
        rows,
        resolution=16,
        batch_size=2,
        sampler_seed=91,
        epoch=4,
        rank=0,
        world_size=2,
        pin_memory=False,
    )
    iterator = iter(loader)
    next(iterator)
    next(iterator)
    cursor.advance(batches=2)
    expected_remaining = [
        (batch["sample_ids"], batch["captions"]) for batch in iterator
    ]

    restored_loader, _ = build_paired_dataloader(
        rows,
        resolution=16,
        batch_size=2,
        sampler_seed=91,
        epoch=0,
        rank=0,
        world_size=2,
        pin_memory=False,
    )
    restored = DataCursor.from_state_dict(cursor.state_dict())
    apply_data_cursor(restored_loader, restored)
    restored_iterator = iter(restored_loader)
    for _ in range(restored.batch_offset):
        next(restored_iterator)
    actual_remaining = [
        (batch["sample_ids"], batch["captions"]) for batch in restored_iterator
    ]

    assert actual_remaining == expected_remaining


def test_dataloader_iterator_does_not_consume_training_cpu_rng():
    loader, _cursor = build_paired_dataloader(
        _rows(4),
        resolution=16,
        batch_size=1,
        num_workers=0,
        sampler_seed=91,
        rank=0,
        world_size=1,
        pin_memory=False,
    )

    torch.manual_seed(9876)
    expected = torch.rand(4)
    torch.manual_seed(9876)
    next(iter(loader))
    actual = torch.rand(4)

    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


def test_dataset_reports_row_id_for_bad_image_and_empty_weighted_caption():
    bad_image = PairedImageTextDataset(
        [{"id": "bad-image", "image": b"broken", "prompt": "ok"}], resolution=16
    )
    empty_caption = PairedImageTextDataset(
        [{"id": "bad-caption", "image": _jpeg_bytes()}], resolution=16
    )

    with pytest.raises(ValueError, match="bad-image"):
        bad_image[0]
    with pytest.raises(ValueError, match="bad-caption"):
        empty_caption[0]


def _add_tar_bytes(archive: tarfile.TarFile, name: str, payload: bytes) -> None:
    member = tarfile.TarInfo(name)
    member.size = len(payload)
    member.mtime = 0
    archive.addfile(member, io.BytesIO(payload))


def _make_fine_t2i_tar(path: Path, *, excluded_id: str) -> None:
    with tarfile.open(path, "w") as archive:
        specs = []
        for index in range(10):
            specs.append((f"eligible-{index:03d}", "Z-Image-Turbo", (1024, 1024)))
        specs.extend(
            [
                (excluded_id, "Z-Image-Turbo", (1024, 1024)),
                (next(iter(QUALITY_EXCLUDED_SOURCE_IDS)), "Z-Image-Turbo", (1024, 1024)),
                ("flux-row", "FLUX.2-dev", (1024, 1024)),
                ("rectangular-row", "Z-Image-Turbo", (1024, 1152)),
            ]
        )
        for index, (sample_id, generator, size) in enumerate(specs):
            image_bytes = _jpeg_bytes(size=size, color=(index, 40, 80))
            metadata = {
                "id": sample_id,
                "image_generator": generator,
                "image_resolution": list(size),
                "prompt": f"prompt {sample_id}",
                "enhanced_prompt": f"enhanced {sample_id}",
                "prompt_generator": "test-generator",
                "style": "test-style",
                "prompt_category": "test-category",
                "aesthetic_predictor_v_2_5_score": 6.0,
            }
            _add_tar_bytes(archive, f"{sample_id}.jpg", image_bytes)
            _add_tar_bytes(
                archive,
                f"{sample_id}.json",
                json.dumps(metadata).encode("utf-8"),
            )
            _add_tar_bytes(
                archive,
                f"{sample_id}.txt",
                f"enhanced {sample_id}".encode("utf-8"),
            )


def test_local_builder_and_validator_create_deterministic_paired_parquet(tmp_path):
    excluded_id = "semvae-overlap"
    tar_path = tmp_path / "train-000000.tar"
    _make_fine_t2i_tar(tar_path, excluded_id=excluded_id)
    tar_sha256 = hashlib.sha256(tar_path.read_bytes()).hexdigest()
    semvae_manifest = tmp_path / "semvae_manifest.json"
    semvae_manifest.write_text(
        json.dumps({"samples": [{"id": excluded_id}]}), encoding="utf-8"
    )
    output_dir = tmp_path / "dataset"
    second_output_dir = tmp_path / "dataset-repeat"

    manifest = build_dataset(
        tar_path=tar_path,
        output_dir=output_dir,
        semvae_manifest=semvae_manifest,
        expected_tar_sha256=tar_sha256,
        train_rows=4,
        validation_rows=2,
        selection_seed=7,
    )
    repeated_manifest = build_dataset(
        tar_path=tar_path,
        output_dir=second_output_dir,
        semvae_manifest=semvae_manifest,
        expected_tar_sha256=tar_sha256,
        train_rows=4,
        validation_rows=2,
        selection_seed=7,
    )
    report = validate_dataset(
        dataset_dir=output_dir,
        tar_path=tar_path,
        semvae_manifest=semvae_manifest,
        expected_tar_sha256=tar_sha256,
        expected_train_rows=4,
        expected_validation_rows=2,
    )
    dataloader, _cursor = build_paired_dataloader(
        output_dir,
        split="train",
        cache_dir=tmp_path / "datasets-cache",
        resolution=16,
        batch_size=2,
        pin_memory=False,
    )
    batch = next(iter(dataloader))

    assert manifest["rows"] == {"train": 4, "validation": 2}
    assert repeated_manifest["artifacts"] == manifest["artifacts"]
    assert manifest["release_review"]["status"] == "pending"
    assert report["status"] == "pass"
    assert report["unique_ids"] == 6
    assert batch["pixel_values"].shape == (2, 3, 16, 16)
    assert len(batch["captions"]) == 2
    selection = json.loads((output_dir / "selection_manifest.json").read_text())
    repeated_selection = json.loads(
        (second_output_dir / "selection_manifest.json").read_text()
    )
    assert repeated_selection == selection
    selected_ids = {
        item["id"]
        for split in selection["splits"].values()
        for item in split
    }
    assert excluded_id not in selected_ids
    assert selected_ids.isdisjoint(QUALITY_EXCLUDED_SOURCE_IDS)
    assert "flux-row" not in selected_ids
    assert "rectangular-row" not in selected_ids

    tampered_manifest = json.loads((output_dir / "manifest.json").read_text())
    tampered_manifest["artifacts"]["data/train-00000-of-00001.parquet"][
        "sha256"
    ] = "0" * 64
    (output_dir / "manifest.json").write_text(
        json.dumps(tampered_manifest), encoding="utf-8"
    )
    with pytest.raises(ValueError, match="Parquet artifact SHA mismatch"):
        validate_dataset(
            dataset_dir=output_dir,
            tar_path=tar_path,
            semvae_manifest=semvae_manifest,
            expected_tar_sha256=tar_sha256,
            expected_train_rows=4,
            expected_validation_rows=2,
        )
