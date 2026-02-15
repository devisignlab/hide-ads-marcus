# tests/attacks/test_lsb.py
import numpy as np
from backend.attacks.lsb import lsb_cloak, lsb_embed


class TestLSBCloak:
    def test_cloak_returns_modified_frames(self):
        frames = [np.full((64, 64, 3), 100, dtype=np.uint8)]
        result = lsb_cloak(frames, intensity=0.5)
        assert len(result) == 1
        assert result[0].shape == (64, 64, 3)
        assert result[0].dtype == np.uint8

    def test_cloak_is_visually_similar(self):
        frames = [np.random.randint(50, 200, (64, 64, 3), dtype=np.uint8)]
        result = lsb_cloak(frames, intensity=0.5)
        diff = np.abs(result[0].astype(int) - frames[0].astype(int))
        assert diff.max() <= 1

    def test_cloak_biases_dark_regions_to_even(self):
        dark = np.full((64, 64, 3), 30, dtype=np.uint8)
        result = lsb_cloak([dark], intensity=1.0)
        lsb_sum = (result[0].astype(int) % 2).sum()
        total = result[0].size
        even_ratio = 1 - (lsb_sum / total)
        assert even_ratio > 0.6


class TestLSBEmbed:
    def test_embed_returns_modified_frames(self):
        frames = [np.full((64, 64, 3), 100, dtype=np.uint8)]
        result = lsb_embed(frames, seed=42)
        assert len(result) == 1
        assert not np.array_equal(result[0], frames[0])

    def test_embed_is_deterministic(self):
        frames = [np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8)]
        a = lsb_embed(frames, seed=42)
        b = lsb_embed(frames, seed=42)
        assert np.array_equal(a[0], b[0])
