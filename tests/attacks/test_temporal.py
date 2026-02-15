# tests/attacks/test_temporal.py
import numpy as np
from backend.attacks.temporal import propagate_keyframes, select_keyframe_indices


class TestKeyframeSelection:
    def test_select_indices(self):
        indices = select_keyframe_indices(total_frames=20, interval=5)
        assert indices == [0, 5, 10, 15, 19]

    def test_interval_1_selects_all(self):
        indices = select_keyframe_indices(total_frames=5, interval=1)
        assert indices == [0, 1, 2, 3, 4]


class TestPropagateKeyframes:
    def test_interpolates_between_keyframes(self):
        originals = [np.full((4, 4, 3), i * 10, dtype=np.uint8) for i in range(10)]
        perturbed_keys = {
            0: np.full((4, 4, 3), 100, dtype=np.uint8),
            5: np.full((4, 4, 3), 200, dtype=np.uint8),
            9: np.full((4, 4, 3), 150, dtype=np.uint8),
        }
        result = propagate_keyframes(originals, perturbed_keys)
        assert len(result) == 10
        assert np.array_equal(result[0], perturbed_keys[0])
        assert np.array_equal(result[5], perturbed_keys[5])
        assert not np.array_equal(result[2], originals[2])
