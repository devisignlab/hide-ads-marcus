# backend/models/yolo_wrapper.py
import numpy as np
import torch
import torch.nn as nn
from ultralytics import YOLO

from backend.config import YOLO_MODEL
from backend.core.device import get_device


class YOLOWrapper:
    def __init__(self, model_name: str = YOLO_MODEL):
        self.device = get_device()
        self._yolo = YOLO(model_name)
        self._yolo.to(self.device)

    @property
    def model(self) -> YOLO:
        return self._yolo

    def get_inner_model(self) -> nn.Module:
        return self._yolo.model

    def detect(self, image: np.ndarray, conf: float = 0.25) -> list[dict]:
        results = self._yolo.predict(image, conf=conf, verbose=False)
        detections = []
        for r in results:
            boxes = r.boxes
            for i in range(len(boxes)):
                detections.append({
                    "label": r.names[int(boxes.cls[i])],
                    "confidence": float(boxes.conf[i]),
                    "bbox": boxes.xyxy[i].tolist(),
                })
        return detections

    def detect_batch(self, images: list[np.ndarray], conf: float = 0.25) -> list[list[dict]]:
        return [self.detect(img, conf=conf) for img in images]
