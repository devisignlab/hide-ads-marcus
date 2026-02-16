"""Run YOLO + CLIP classification on frames."""

import numpy as np
from backend.models.cache import get_yolo, get_clip


def classify_frames(
    frames: list[np.ndarray],
    clip_labels: list[str],
    yolo_conf: float = 0.25,
) -> list[dict]:
    """Classify each frame with YOLO (object detection) and CLIP (semantic).

    Returns list of dicts with keys: yolo_detections, clip_scores.
    """
    yolo = get_yolo()
    clip_model = get_clip()

    results = []
    for frame in frames:
        yolo_dets = yolo.detect(frame, conf=yolo_conf)
        clip_scores = clip_model.classify(frame, clip_labels)
        results.append({
            "yolo_detections": yolo_dets,
            "clip_scores": clip_scores,
        })
    return results
