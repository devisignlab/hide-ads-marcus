# backend/attacks/lsb.py
import numpy as np


def lsb_cloak(frames: list[np.ndarray], intensity: float = 0.5, dark_threshold: int = 60) -> list[np.ndarray]:
    result = []
    for frame in frames:
        out = frame.copy()
        luminance = out.mean(axis=2)
        dark_mask = luminance < dark_threshold
        rng = np.random.default_rng()
        prob_mask = rng.random(dark_mask.shape) < intensity
        modify_mask = dark_mask & prob_mask
        for c in range(3):
            channel = out[:, :, c]
            channel[modify_mask] = channel[modify_mask] & 0xFE
        result.append(out)
    return result


def lsb_embed(frames: list[np.ndarray], seed: int = 42) -> list[np.ndarray]:
    rng = np.random.default_rng(seed)
    result = []
    for frame in frames:
        out = frame.copy()
        noise = rng.integers(0, 2, size=frame.shape, dtype=np.uint8)
        out = (out & 0xFE) | noise
        result.append(out)
    return result
