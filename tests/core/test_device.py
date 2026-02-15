# tests/core/test_device.py
import torch
from backend.core.device import detect_device, get_device


class TestDetectDevice:
    def test_returns_valid_device_string(self):
        device = detect_device()
        assert device in ("mps", "cuda", "cpu")

    def test_get_device_returns_torch_device(self):
        device = get_device()
        assert isinstance(device, torch.device)

    def test_smoke_test_runs_tensor_op(self):
        device = get_device()
        t = torch.ones(2, 2, device=device)
        result = (t + t).sum().item()
        assert result == 8.0
