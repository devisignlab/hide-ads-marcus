# tests/attacks/test_uap.py
import numpy as np
import pytest
from backend.attacks.uap import compute_uap, apply_uap


@pytest.fixture(scope="module")
def sample_frames():
    return [np.random.randint(0, 255, (240, 320, 3), dtype=np.uint8) for _ in range(4)]


class TestUAP:
    def test_compute_uap_returns_perturbation(self, sample_frames):
        uap = compute_uap(sample_frames[:2], target="yolo", steps=2, epsilon=4/255)
        assert isinstance(uap, np.ndarray)
        assert uap.shape == sample_frames[0].shape
        assert uap.dtype == np.float32

    def test_apply_uap_modifies_frames(self, sample_frames):
        uap = np.random.uniform(-4/255, 4/255, sample_frames[0].shape).astype(np.float32)
        result = apply_uap(sample_frames, uap)
        assert len(result) == 4
        assert result[0].dtype == np.uint8
        assert not np.array_equal(result[0], sample_frames[0])
