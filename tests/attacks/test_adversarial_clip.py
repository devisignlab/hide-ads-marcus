# tests/attacks/test_adversarial_clip.py
import numpy as np
import pytest
from backend.attacks.adversarial import attack_clip_fgsm, attack_clip_pgd


@pytest.fixture(scope="module")
def sample_frames():
    return [np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8) for _ in range(2)]


class TestCLIPAttacks:
    def test_fgsm_returns_perturbed_frames(self, sample_frames):
        perturbed = attack_clip_fgsm(sample_frames, target_text="a dolphin swimming in the ocean", epsilon=4/255)
        assert len(perturbed) == 2
        assert not np.array_equal(perturbed[0], sample_frames[0])

    def test_pgd_returns_perturbed_frames(self, sample_frames):
        perturbed = attack_clip_pgd(sample_frames, target_text="a dolphin swimming in the ocean", epsilon=4/255, steps=3, alpha=1/255)
        assert len(perturbed) == 2
        assert perturbed[0].dtype == np.uint8
