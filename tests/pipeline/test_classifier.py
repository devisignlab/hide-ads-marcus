# tests/pipeline/test_classifier.py
import numpy as np
import pytest
from backend.pipeline.classifier import classify_frames

DEFAULT_LABELS = ["a person", "a car", "a cat", "an empty scene", "a building"]


@pytest.fixture(scope="module")
def frames():
    return [np.random.randint(0, 255, (240, 320, 3), dtype=np.uint8) for _ in range(2)]


class TestClassifier:
    def test_classify_returns_structured_results(self, frames):
        results = classify_frames(frames, clip_labels=DEFAULT_LABELS)
        assert len(results) == 2
        r = results[0]
        assert "yolo_detections" in r
        assert "clip_scores" in r
        assert isinstance(r["yolo_detections"], list)
        assert isinstance(r["clip_scores"], dict)
        assert len(r["clip_scores"]) == len(DEFAULT_LABELS)
