"""Small distributed helpers for CLI inference."""

from __future__ import annotations

import torch


def setup_distributed():
    try:
        from accelerate import Accelerator
    except ModuleNotFoundError:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        return 0, 1, device, True, None

    accelerator = Accelerator()
    return (
        int(accelerator.process_index),
        int(accelerator.num_processes),
        accelerator.device,
        bool(accelerator.is_main_process),
        accelerator,
    )


def wait_for_everyone(accelerator) -> None:
    if accelerator is not None:
        accelerator.wait_for_everyone()


def shard_indices_interleaved(total: int, rank: int, world_size: int) -> list[int]:
    return list(range(int(rank), int(total), int(world_size)))


def build_rank_generator(device: torch.device, seed: int, rank: int) -> torch.Generator:
    return torch.Generator(device=str(device)).manual_seed(int(seed) + int(rank))
