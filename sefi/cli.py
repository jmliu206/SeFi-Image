"""Command line entry point for SeFi-Image inference."""

from __future__ import annotations

import argparse
from dataclasses import asdict

from .distributed import (
    build_rank_generator,
    setup_distributed,
    shard_indices_interleaved,
    wait_for_everyone,
)
from .io import expand_prompts, load_prompts, save_images, write_manifest
from .pipeline import SEFIInferencePipeline


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prompt", default="")
    parser.add_argument("--prompt-file", default="")
    parser.add_argument("--output-dir", default="outputs/inference")
    parser.add_argument("--cache-dir", default="outputs/model_weights/sefi_inference")
    parser.add_argument(
        "--checkpoint",
        required=True,
        help="Local checkpoint path or Hugging Face repo id.",
    )
    parser.add_argument(
        "--config",
        default="",
        help="Optional config path. Defaults to sefi_config.yaml under --checkpoint.",
    )
    parser.add_argument("--steps", type=int, default=None)
    parser.add_argument("--guidance-scale", type=float, default=None)
    parser.add_argument("--height", type=int, default=None)
    parser.add_argument("--width", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--num-images-per-prompt", type=int, default=1)
    parser.add_argument("--seed", type=int, default=20260616)
    parser.add_argument("--device", default="")
    parser.add_argument("--dtype", choices=("bf16", "fp32"), default="")
    parser.add_argument("--delta-t", type=float, default=None)
    parser.add_argument("--timestep-shift-alpha", type=float, default=None)
    parser.add_argument("--debug-assert-schedule", action="store_true")
    parser.add_argument("--autoguidance-config", default="")
    parser.add_argument("--autoguidance-checkpoint", default="")
    parser.add_argument("--guidance-interval-sigma-lo", type=float, default=None)
    parser.add_argument("--guidance-interval-sigma-hi", type=float, default=None)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    prompts = load_prompts(
        prompt=args.prompt or None,
        prompt_file=args.prompt_file or None,
    )
    items = expand_prompts(prompts, args.num_images_per_prompt)

    rank, world_size, device, is_main, accelerator = setup_distributed()
    local_indices = shard_indices_interleaved(len(items), rank, world_size)
    local_items = [items[index] for index in local_indices]
    local_prompts = [item.prompt for item in local_items]

    pipe = SEFIInferencePipeline.from_pretrained(
        args.checkpoint,
        cache_dir=args.cache_dir,
        config=args.config or None,
        device=args.device or str(device),
        dtype=args.dtype or None,
        delta_t=args.delta_t,
        timestep_shift_alpha=args.timestep_shift_alpha,
        debug_assert_schedule=args.debug_assert_schedule,
        autoguidance_config=args.autoguidance_config or None,
        autoguidance_checkpoint=args.autoguidance_checkpoint or None,
        guidance_interval_sigma_lo=args.guidance_interval_sigma_lo,
        guidance_interval_sigma_hi=args.guidance_interval_sigma_hi,
    )

    generator = build_rank_generator(device, args.seed, rank)
    images = pipe(
        local_prompts,
        num_inference_steps=args.steps,
        guidance_scale=args.guidance_scale,
        height=args.height,
        width=args.width,
        batch_size=args.batch_size,
        generator=generator,
    )
    save_images(output_dir=args.output_dir, items=local_items, images=images, rank=rank)
    wait_for_everyone(accelerator)

    if is_main:
        write_manifest(
            args.output_dir,
            {
                "model": pipe.spec.name,
                "model_spec": asdict(pipe.spec),
                "checkpoint_path": pipe.checkpoint_path,
                "checkpoint_uri": pipe.checkpoint_uri,
                "num_prompts": len(prompts),
                "num_images": len(items),
                "seed": args.seed,
                "world_size": world_size,
            },
        )


if __name__ == "__main__":
    main()
