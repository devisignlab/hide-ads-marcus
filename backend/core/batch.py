# backend/core/batch.py
from collections.abc import Sequence
from typing import TypeVar

import numpy as np
import torch

T = TypeVar("T")


def frames_to_tensor(frames: list[np.ndarray], device: torch.device) -> torch.Tensor:
    arr = np.stack(frames)
    tensor = torch.from_numpy(arr).float().div(255.0)
    tensor = tensor.permute(0, 3, 1, 2)
    return tensor.to(device)


def tensor_to_frames(tensor: torch.Tensor) -> list[np.ndarray]:
    tensor = tensor.detach().cpu().clamp(0, 1)
    tensor = tensor.permute(0, 2, 3, 1)
    arr = (tensor * 255).to(torch.uint8).numpy()
    return [arr[i] for i in range(arr.shape[0])]


def iter_batches(items: Sequence[T], batch_size: int) -> list[list[T]]:
    return [list(items[i : i + batch_size]) for i in range(0, len(items), batch_size)]
