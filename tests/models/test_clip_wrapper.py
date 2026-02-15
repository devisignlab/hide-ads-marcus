# tests/models/test_clip_wrapper.py
import numpy as np
import torch
import pytest
from backend.models.clip_wrapper import CLIPWrapper


@pytest.fixture(scope="module")
def clip_model():
    return CLIPWrapper()


class TestCLIPWrapper:
    def test_loads_model(self, clip_model):
        assert clip_model.model is not None

    def test_classify_returns_scores(self, clip_model):
        img = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
        labels = ["a dog", "a cat", "a car"]
        scores = clip_model.classify(img, labels)
        assert len(scores) == 3
        assert abs(sum(scores.values()) - 1.0) < 0.01

    def test_encode_image_preserves_grad(self, clip_model):
        tensor = torch.randn(1, 3, 224, 224, device=clip_model.device, requires_grad=True)
        embedding = clip_model.encode_image_differentiable(tensor)
        assert embedding.requires_grad
        assert embedding.shape[1] == 512
