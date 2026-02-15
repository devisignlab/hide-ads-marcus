# tests/core/test_batch.py
import numpy as np
import torch
from backend.core.batch import frames_to_tensor, tensor_to_frames, iter_batches


class TestFramesToTensor:
    def test_converts_uint8_frames_to_float_tensor(self):
        frames = [np.full((64, 64, 3), 128, dtype=np.uint8) for _ in range(4)]
        tensor = frames_to_tensor(frames, device=torch.device("cpu"))
        assert tensor.shape == (4, 3, 64, 64)
        assert tensor.dtype == torch.float32
        assert torch.allclose(tensor, torch.full_like(tensor, 128 / 255), atol=0.01)


class TestTensorToFrames:
    def test_converts_float_tensor_to_uint8_frames(self):
        tensor = torch.full((4, 3, 64, 64), 0.5)
        frames = tensor_to_frames(tensor)
        assert len(frames) == 4
        assert frames[0].shape == (64, 64, 3)
        assert frames[0].dtype == np.uint8
        assert np.allclose(frames[0], 128, atol=1)


class TestIterBatches:
    def test_yields_correct_batch_sizes(self):
        items = list(range(10))
        batches = list(iter_batches(items, batch_size=3))
        assert len(batches) == 4
        assert batches[0] == [0, 1, 2]
        assert batches[-1] == [9]
