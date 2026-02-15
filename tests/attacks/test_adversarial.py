# tests/attacks/test_adversarial.py
import numpy as np
import pytest
from backend.attacks.adversarial import attack_yolo_fgsm, attack_yolo_pgd


@pytest.fixture(scope="module")
def sample_frames():
    return [np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8) for _ in range(4)]


class TestYOLOAttacks:
    def test_fgsm_returns_perturbed_frames(self, sample_frames):
        perturbed = attack_yolo_fgsm(sample_frames, epsilon=4/255)
        assert len(perturbed) == len(sample_frames)
        assert perturbed[0].shape == sample_frames[0].shape
        assert perturbed[0].dtype == np.uint8
        assert not np.array_equal(perturbed[0], sample_frames[0])

    def test_fgsm_respects_epsilon_bound(self, sample_frames):
        eps = 4 / 255
        perturbed = attack_yolo_fgsm(sample_frames, epsilon=eps)
        diff = np.abs(perturbed[0].astype(float) - sample_frames[0].astype(float)) / 255
        assert diff.max() <= eps + 1e-6

    def test_pgd_returns_perturbed_frames(self, sample_frames):
        perturbed = attack_yolo_pgd(sample_frames, epsilon=4/255, steps=3, alpha=1/255)
        assert len(perturbed) == len(sample_frames)
        assert not np.array_equal(perturbed[0], sample_frames[0])
