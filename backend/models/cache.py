# backend/models/cache.py
from backend.models.yolo_wrapper import YOLOWrapper
from backend.models.clip_wrapper import CLIPWrapper

_yolo: YOLOWrapper | None = None
_clip: CLIPWrapper | None = None


def get_yolo() -> YOLOWrapper:
    global _yolo
    if _yolo is None:
        _yolo = YOLOWrapper()
    return _yolo


def get_clip() -> CLIPWrapper:
    global _clip
    if _clip is None:
        _clip = CLIPWrapper()
    return _clip
