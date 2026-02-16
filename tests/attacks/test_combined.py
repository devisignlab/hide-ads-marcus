# tests/attacks/test_combined.py
import numpy as np
import pytest
from backend.attacks.combined import attack_combined_pgd


@pytest.fixture(scope="module")
def sample_frames():
    return [np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8) for _ in range(2)]


class TestCombinedAttack:
    def test_combined_returns_perturbed_frames(self, sample_frames):
        perturbed = attack_combined_pgd(
            sample_frames,
            target_text="an empty room",
            yolo_weight=0.5,
            clip_weight=0.5,
            epsilon=4/255,
            steps=3,
        )
        assert len(perturbed) == 2
        assert not np.array_equal(perturbed[0], sample_frames[0])

    def test_yolo_only_weight(self, sample_frames):
        perturbed = attack_combined_pgd(
            sample_frames,
            target_text="an empty room",
            yolo_weight=1.0,
            clip_weight=0.0,
            epsilon=4/255,
            steps=2,
        )
        assert len(perturbed) == 2
