# backend/attacks/temporal.py
import numpy as np


def select_keyframe_indices(total_frames: int, interval: int) -> list[int]:
    if interval <= 0:
        raise ValueError("interval must be >= 1")
    if interval == 1:
        return list(range(total_frames))
    indices = list(range(0, total_frames, interval))
    if indices[-1] != total_frames - 1:
        indices.append(total_frames - 1)
    return indices


def propagate_keyframes(originals: list[np.ndarray], perturbed_keys: dict[int, np.ndarray]) -> list[np.ndarray]:
    n = len(originals)
    key_indices = sorted(perturbed_keys.keys())
    result = [None] * n
    for idx in key_indices:
        result[idx] = perturbed_keys[idx]
    deltas = {}
    for idx in key_indices:
        deltas[idx] = perturbed_keys[idx].astype(np.float32) - originals[idx].astype(np.float32)
    for i in range(len(key_indices) - 1):
        start = key_indices[i]
        end = key_indices[i + 1]
        if end - start <= 1:
            continue
        d_start = deltas[start]
        d_end = deltas[end]
        for j in range(start + 1, end):
            t = (j - start) / (end - start)
            interpolated_delta = d_start * (1 - t) + d_end * t
            frame = originals[j].astype(np.float32) + interpolated_delta
            result[j] = np.clip(frame, 0, 255).astype(np.uint8)
    return result
