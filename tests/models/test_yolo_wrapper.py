# tests/models/test_yolo_wrapper.py
import numpy as np
import pytest
from backend.models.yolo_wrapper import YOLOWrapper


@pytest.fixture(scope="module")
def yolo():
    return YOLOWrapper()


class TestYOLOWrapper:
    def test_loads_model(self, yolo):
        assert yolo.model is not None

    def test_detect_returns_list_of_dicts(self, yolo):
        img = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        results = yolo.detect(img)
        assert isinstance(results, list)
        if len(results) > 0:
            det = results[0]
            assert "label" in det
            assert "confidence" in det
            assert "bbox" in det

    def test_get_inner_model_returns_nn_module(self, yolo):
        import torch.nn as nn
        inner = yolo.get_inner_model()
        assert isinstance(inner, nn.Module)
