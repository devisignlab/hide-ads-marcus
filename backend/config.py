# backend/config.py
from dataclasses import dataclass

ATTACK_METHODS = ("fgsm", "pgd", "lsb", "uap", "combined", "turbo", "anti_deepfake", "auto")

YOLO_MODEL = "yolov8n.pt"
CLIP_MODEL = "ViT-B/32"
YOLO_INPUT_SIZE = 640
CLIP_INPUT_SIZE = 224

MAX_UPLOAD_MB = 500
UPLOAD_DIR = "uploads"
OUTPUT_DIR = "outputs"

DEFAULT_EPSILON = 4 / 255
DEFAULT_ALPHA = 1 / 255


@dataclass(frozen=True)
class QualityPreset:
    name: str
    method: str
    pgd_steps: int
    keyframe_interval: int
    batch_size: int
    epsilon: float


_PRESETS: dict[str, QualityPreset] = {
    "preview": QualityPreset(
        name="preview", method="fgsm", pgd_steps=0,
        keyframe_interval=15, batch_size=8, epsilon=DEFAULT_EPSILON,
    ),
    "fast": QualityPreset(
        name="fast", method="pgd", pgd_steps=5,
        keyframe_interval=10, batch_size=4, epsilon=DEFAULT_EPSILON,
    ),
    "standard": QualityPreset(
        name="standard", method="pgd", pgd_steps=20,
        keyframe_interval=5, batch_size=4, epsilon=DEFAULT_EPSILON,
    ),
    "high": QualityPreset(
        name="high", method="pgd", pgd_steps=40,
        keyframe_interval=1, batch_size=2, epsilon=DEFAULT_EPSILON,
    ),
    "aggressive": QualityPreset(
        name="aggressive", method="pgd", pgd_steps=40,
        keyframe_interval=5, batch_size=1, epsilon=32 / 255,
    ),
}


def get_preset(name: str) -> QualityPreset:
    return _PRESETS[name]


# Memory safety: max frames to hold in RAM at once
MAX_FRAMES_IN_MEMORY = 3000  # ~3000 frames at 720p ≈ 7GB
