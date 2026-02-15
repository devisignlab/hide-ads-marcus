# backend/core/device.py
import torch

_cached_device: str | None = None


def detect_device() -> str:
    """Detect best available device with MPS smoke test."""
    global _cached_device
    if _cached_device is not None:
        return _cached_device

    if torch.backends.mps.is_available():
        try:
            t = torch.zeros(1, device="mps")
            _ = (t + t).item()
            _cached_device = "mps"
            return "mps"
        except Exception:
            pass

    if torch.cuda.is_available():
        _cached_device = "cuda"
        return "cuda"

    _cached_device = "cpu"
    return "cpu"


def get_device() -> torch.device:
    """Return torch.device for the best available backend."""
    return torch.device(detect_device())
