# tests/models/test_cache.py
from backend.models.cache import get_yolo, get_clip


class TestModelCache:
    def test_yolo_returns_same_instance(self):
        a = get_yolo()
        b = get_yolo()
        assert a is b

    def test_clip_returns_same_instance(self):
        a = get_clip()
        b = get_clip()
        assert a is b
