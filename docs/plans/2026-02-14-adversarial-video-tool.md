# Adversarial Video Tool — Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Build a production-grade tool that takes a video and produces a visually identical copy that AI models (YOLO + CLIP) classify completely differently — with a web interface for control and comparison.

**Architecture:** Python backend (FastAPI) handles all heavy processing — video I/O, model inference, adversarial attacks, job management. Three attack approaches: gradient-based perturbation (FGSM/PGD via ART for YOLO, custom for CLIP), LSB steganography (content-aware), and Universal Adversarial Perturbation (UAP). Frontend is vanilla HTML/CSS/JS communicating via REST + WebSocket. Pipeline is 5 stages: extract frames, classify originals, perturb, verify, reconstruct.

**Tech Stack:** Python 3.11, uv, FastAPI, PyTorch (MPS), ART (IBM Adversarial Robustness Toolbox), Ultralytics YOLOv8, OpenAI CLIP (ViT-B/32), OpenCV, ffmpeg

**System:** Apple M1 Pro, 16GB RAM, Metal 4, macOS

---

## Phase 0: Project Bootstrapping

### Task 1: Initialize Project

**Files:**
- Create: `pyproject.toml`
- Create: `.gitignore`
- Create: `.python-version`
- Create: `backend/__init__.py`
- Create: `tests/__init__.py`

**Step 1: Initialize git repo**

Run:
```bash
cd /Users/mamprim/hide-ads
git init
```

**Step 2: Create `.python-version`**

```
3.11
```

**Step 3: Create `.gitignore`**

```gitignore
__pycache__/
*.pyc
.venv/
*.egg-info/
dist/
build/
.pytest_cache/
uploads/
outputs/
*.mp4
*.avi
*.mov
!tests/fixtures/*.mp4
.DS_Store
```

**Step 4: Create `pyproject.toml`**

```toml
[project]
name = "hide-ads"
version = "0.1.0"
description = "Adversarial video tool — make videos invisible to AI detection"
requires-python = ">=3.11,<3.12"
dependencies = [
    "fastapi>=0.115.0",
    "uvicorn[standard]>=0.34.0",
    "python-multipart>=0.0.18",
    "websockets>=14.0",
    "torch>=2.5.0",
    "torchvision>=0.20.0",
    "ultralytics>=8.3.0",
    "adversarial-robustness-toolbox>=1.18.0",
    "ftfy>=6.1.1",
    "regex>=2024.0.0",
    "opencv-python-headless>=4.10.0",
    "numpy>=1.26.0,<2.0.0",
    "Pillow>=11.0.0",
    "aiofiles>=24.0.0",
]

[project.optional-dependencies]
clip = ["git+https://github.com/openai/CLIP.git"]
dev = [
    "pytest>=8.0",
    "pytest-asyncio>=0.24.0",
    "httpx>=0.28.0",
]

[tool.pytest.ini_options]
testpaths = ["tests"]
asyncio_mode = "auto"
```

**Step 5: Create directory skeleton**

Run:
```bash
mkdir -p backend/{api,core,models,attacks,pipeline,storage}
mkdir -p frontend/{css,js}
mkdir -p tests/{core,models,attacks,pipeline,api}
mkdir -p scripts
touch backend/__init__.py backend/api/__init__.py backend/core/__init__.py
touch backend/models/__init__.py backend/attacks/__init__.py
touch backend/pipeline/__init__.py backend/storage/__init__.py
touch tests/__init__.py tests/core/__init__.py tests/models/__init__.py
touch tests/attacks/__init__.py tests/pipeline/__init__.py tests/api/__init__.py
```

**Step 6: Install dependencies**

Run:
```bash
cd /Users/mamprim/hide-ads
uv venv --python 3.11
uv pip install -e ".[dev]"
uv pip install git+https://github.com/openai/CLIP.git
```

Expected: All dependencies install successfully. PyTorch with MPS support.

**Step 7: Verify installation**

Run:
```bash
uv run python -c "
import torch; print(f'PyTorch {torch.__version__}, MPS: {torch.backends.mps.is_available()}')
import ultralytics; print(f'Ultralytics {ultralytics.__version__}')
import art; print(f'ART {art.__version__}')
import clip; print('CLIP OK')
import cv2; print(f'OpenCV {cv2.__version__}')
import fastapi; print(f'FastAPI {fastapi.__version__}')
"
```

Expected: All imports succeed. MPS: True.

**Step 8: Commit**

```bash
git add -A
git commit -m "chore: initialize project with uv, Python 3.11, core dependencies"
```

---

## Phase 1: Core Infrastructure

### Task 2: Device Detection

**Files:**
- Create: `backend/core/device.py`
- Test: `tests/core/test_device.py`

**Step 1: Write the failing test**

```python
# tests/core/test_device.py
import torch
from backend.core.device import detect_device, get_device


class TestDetectDevice:
    def test_returns_valid_device_string(self):
        device = detect_device()
        assert device in ("mps", "cuda", "cpu")

    def test_get_device_returns_torch_device(self):
        device = get_device()
        assert isinstance(device, torch.device)

    def test_smoke_test_runs_tensor_op(self):
        device = get_device()
        t = torch.ones(2, 2, device=device)
        result = (t + t).sum().item()
        assert result == 8.0
```

**Step 2: Run test to verify it fails**

Run: `uv run pytest tests/core/test_device.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'backend.core.device'`

**Step 3: Write minimal implementation**

```python
# backend/core/device.py
import torch

_cached_device: str | None = None


def detect_device() -> str:
    """Detect best available device with MPS smoke test."""
    global _cached_device
    if _cached_device is not None:
        return _cached_device

    if torch.backends.mps.is_available():
        try:
            t = torch.zeros(1, device="mps")
            _ = (t + t).item()
            _cached_device = "mps"
            return "mps"
        except Exception:
            pass

    if torch.cuda.is_available():
        _cached_device = "cuda"
        return "cuda"

    _cached_device = "cpu"
    return "cpu"


def get_device() -> torch.device:
    """Return torch.device for the best available backend."""
    return torch.device(detect_device())
```

**Step 4: Run test to verify it passes**

Run: `uv run pytest tests/core/test_device.py -v`
Expected: 3 passed

**Step 5: Commit**

```bash
git add backend/core/device.py tests/core/test_device.py
git commit -m "feat: add device detection with MPS smoke test and CPU fallback"
```

---

### Task 3: Configuration & Presets

**Files:**
- Create: `backend/config.py`
- Test: `tests/test_config.py`

**Step 1: Write the failing test**

```python
# tests/test_config.py
from backend.config import QualityPreset, get_preset, ATTACK_METHODS


class TestConfig:
    def test_preset_preview_exists(self):
        preset = get_preset("preview")
        assert preset.method == "fgsm"
        assert preset.keyframe_interval > 5

    def test_preset_standard_exists(self):
        preset = get_preset("standard")
        assert preset.method == "pgd"
        assert preset.pgd_steps >= 15

    def test_preset_high_exists(self):
        preset = get_preset("high")
        assert preset.method == "pgd"
        assert preset.keyframe_interval == 1

    def test_attack_methods_list(self):
        assert "fgsm" in ATTACK_METHODS
        assert "pgd" in ATTACK_METHODS
        assert "lsb" in ATTACK_METHODS
        assert "uap" in ATTACK_METHODS

    def test_invalid_preset_raises(self):
        import pytest
        with pytest.raises(KeyError):
            get_preset("nonexistent")
```

**Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_config.py -v`
Expected: FAIL — `ModuleNotFoundError`

**Step 3: Write minimal implementation**

```python
# backend/config.py
from dataclasses import dataclass

ATTACK_METHODS = ("fgsm", "pgd", "lsb", "uap", "combined")

YOLO_MODEL = "yolov8n.pt"
CLIP_MODEL = "ViT-B/32"
YOLO_INPUT_SIZE = 640
CLIP_INPUT_SIZE = 224

MAX_UPLOAD_MB = 500
UPLOAD_DIR = "uploads"
OUTPUT_DIR = "outputs"

# Perturbation constraints
DEFAULT_EPSILON = 4 / 255  # L-inf bound (~0.016)
DEFAULT_ALPHA = 1 / 255    # PGD step size


@dataclass(frozen=True)
class QualityPreset:
    name: str
    method: str             # "fgsm" or "pgd"
    pgd_steps: int          # ignored for fgsm
    keyframe_interval: int  # 1 = every frame
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
}


def get_preset(name: str) -> QualityPreset:
    return _PRESETS[name]
```

**Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_config.py -v`
Expected: 5 passed

**Step 5: Commit**

```bash
git add backend/config.py tests/test_config.py
git commit -m "feat: add quality presets and configuration constants"
```

---

### Task 4: Video I/O — Frame Extraction

**Files:**
- Create: `backend/core/video.py`
- Create: `tests/fixtures/` (test video)
- Test: `tests/core/test_video.py`

**Step 1: Create a tiny test fixture video**

```python
# Run this once to create test fixture:
# uv run python -c "
import cv2, numpy as np, os
os.makedirs('tests/fixtures', exist_ok=True)
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
w = cv2.VideoWriter('tests/fixtures/test_2s.mp4', fourcc, 10, (320, 240))
for i in range(20):  # 2 seconds at 10fps
    frame = np.full((240, 320, 3), fill_value=(i * 12) % 256, dtype=np.uint8)
    cv2.putText(frame, str(i), (140, 130), cv2.FONT_HERSHEY_SIMPLEX, 2, (255,255,255), 3)
    w.write(frame)
w.release()
# "
```

**Step 2: Write the failing test**

```python
# tests/core/test_video.py
import numpy as np
import os
from backend.core.video import extract_frames, get_video_info, extract_audio


FIXTURE = "tests/fixtures/test_2s.mp4"


class TestExtractFrames:
    def test_yields_numpy_arrays(self):
        frames = list(extract_frames(FIXTURE))
        assert len(frames) == 20
        assert isinstance(frames[0], np.ndarray)
        assert frames[0].shape == (240, 320, 3)

    def test_generator_is_lazy(self):
        gen = extract_frames(FIXTURE)
        first = next(gen)
        assert isinstance(first, np.ndarray)

    def test_frames_are_rgb(self):
        frame = next(extract_frames(FIXTURE))
        # OpenCV reads BGR by default; we want RGB
        assert frame.dtype == np.uint8


class TestVideoInfo:
    def test_returns_metadata(self):
        info = get_video_info(FIXTURE)
        assert info["fps"] == 10
        assert info["frame_count"] == 20
        assert info["width"] == 320
        assert info["height"] == 240
        assert info["duration_s"] == 2.0


class TestExtractAudio:
    def test_extract_audio_returns_path_or_none(self):
        # test video has no audio, should return None
        result = extract_audio(FIXTURE, "/tmp/test_audio.aac")
        assert result is None
```

**Step 3: Run test to verify it fails**

Run: `uv run pytest tests/core/test_video.py -v`
Expected: FAIL — `ModuleNotFoundError`

**Step 4: Write minimal implementation**

```python
# backend/core/video.py
import subprocess
import cv2
import numpy as np
from collections.abc import Generator


def get_video_info(path: str) -> dict:
    """Extract video metadata."""
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise FileNotFoundError(f"Cannot open video: {path}")
    try:
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        return {
            "fps": fps,
            "frame_count": frame_count,
            "width": width,
            "height": height,
            "duration_s": frame_count / fps if fps > 0 else 0,
        }
    finally:
        cap.release()


def extract_frames(path: str) -> Generator[np.ndarray, None, None]:
    """Lazily yield frames as RGB uint8 numpy arrays."""
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise FileNotFoundError(f"Cannot open video: {path}")
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            yield cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    finally:
        cap.release()


def extract_audio(video_path: str, output_path: str) -> str | None:
    """Extract audio track via ffmpeg. Returns output_path or None if no audio."""
    try:
        result = subprocess.run(
            ["ffmpeg", "-y", "-i", video_path, "-vn", "-acodec", "copy", output_path],
            capture_output=True, timeout=60,
        )
        if result.returncode != 0:
            return None
        # Check file was actually created and has content
        import os
        if os.path.exists(output_path) and os.path.getsize(output_path) > 0:
            return output_path
        return None
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return None
```

**Step 5: Run test to verify it passes**

Run: `uv run pytest tests/core/test_video.py -v`
Expected: 5 passed

**Step 6: Commit**

```bash
git add backend/core/video.py tests/core/test_video.py tests/fixtures/test_2s.mp4
git commit -m "feat: add video frame extraction (lazy generator) and metadata"
```

---

### Task 5: Video I/O — Reconstruction

**Files:**
- Modify: `backend/core/video.py`
- Test: `tests/core/test_video.py` (add tests)

**Step 1: Write the failing test**

```python
# Add to tests/core/test_video.py
import tempfile
from backend.core.video import reconstruct_video


class TestReconstructVideo:
    def test_reconstruct_creates_mp4(self):
        frames = list(extract_frames(FIXTURE))
        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as f:
            output_path = f.name
        reconstruct_video(frames, output_path, fps=10)
        info = get_video_info(output_path)
        assert info["frame_count"] == 20
        assert info["width"] == 320
        assert info["height"] == 240
        os.unlink(output_path)

    def test_reconstruct_with_audio_mux(self):
        frames = list(extract_frames(FIXTURE))
        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as f:
            output_path = f.name
        # No audio in test fixture — should still produce valid video
        reconstruct_video(frames, output_path, fps=10, audio_path=None)
        assert os.path.exists(output_path)
        assert os.path.getsize(output_path) > 0
        os.unlink(output_path)
```

**Step 2: Run test to verify it fails**

Run: `uv run pytest tests/core/test_video.py::TestReconstructVideo -v`
Expected: FAIL — `ImportError`

**Step 3: Write minimal implementation**

Add to `backend/core/video.py`:

```python
def reconstruct_video(
    frames: list[np.ndarray],
    output_path: str,
    fps: float,
    audio_path: str | None = None,
) -> str:
    """Rebuild MP4 from RGB frames, optionally muxing audio."""
    if not frames:
        raise ValueError("No frames to reconstruct")

    h, w = frames[0].shape[:2]
    temp_path = output_path + ".tmp.mp4" if audio_path else output_path

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(temp_path, fourcc, fps, (w, h))
    try:
        for frame in frames:
            writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
    finally:
        writer.release()

    if audio_path:
        _mux_audio(temp_path, audio_path, output_path)
        import os
        os.unlink(temp_path)

    return output_path


def _mux_audio(video_path: str, audio_path: str, output_path: str) -> None:
    """Mux audio into video via ffmpeg (stream copy, no re-encoding)."""
    subprocess.run(
        [
            "ffmpeg", "-y",
            "-i", video_path,
            "-i", audio_path,
            "-c:v", "copy", "-c:a", "copy",
            "-shortest",
            output_path,
        ],
        capture_output=True, timeout=120, check=True,
    )
```

**Step 4: Run test to verify it passes**

Run: `uv run pytest tests/core/test_video.py -v`
Expected: 7 passed

**Step 5: Commit**

```bash
git add backend/core/video.py tests/core/test_video.py
git commit -m "feat: add video reconstruction with ffmpeg audio muxing"
```

---

### Task 6: Batch Processing Utilities

**Files:**
- Create: `backend/core/batch.py`
- Test: `tests/core/test_batch.py`

**Step 1: Write the failing test**

```python
# tests/core/test_batch.py
import numpy as np
import torch
from backend.core.batch import frames_to_tensor, tensor_to_frames, iter_batches


class TestFramesToTensor:
    def test_converts_uint8_frames_to_float_tensor(self):
        frames = [np.full((64, 64, 3), 128, dtype=np.uint8) for _ in range(4)]
        tensor = frames_to_tensor(frames, device=torch.device("cpu"))
        assert tensor.shape == (4, 3, 64, 64)  # NCHW
        assert tensor.dtype == torch.float32
        assert torch.allclose(tensor, torch.full_like(tensor, 128 / 255), atol=0.01)


class TestTensorToFrames:
    def test_converts_float_tensor_to_uint8_frames(self):
        tensor = torch.full((4, 3, 64, 64), 0.5)
        frames = tensor_to_frames(tensor)
        assert len(frames) == 4
        assert frames[0].shape == (64, 64, 3)  # HWC
        assert frames[0].dtype == np.uint8
        assert np.allclose(frames[0], 128, atol=1)


class TestIterBatches:
    def test_yields_correct_batch_sizes(self):
        items = list(range(10))
        batches = list(iter_batches(items, batch_size=3))
        assert len(batches) == 4
        assert batches[0] == [0, 1, 2]
        assert batches[-1] == [9]
```

**Step 2: Run test to verify it fails**

Run: `uv run pytest tests/core/test_batch.py -v`
Expected: FAIL

**Step 3: Write minimal implementation**

```python
# backend/core/batch.py
from collections.abc import Sequence
from typing import TypeVar

import numpy as np
import torch

T = TypeVar("T")


def frames_to_tensor(
    frames: list[np.ndarray],
    device: torch.device,
) -> torch.Tensor:
    """Convert list of HWC uint8 RGB frames to NCHW float32 tensor in [0,1]."""
    # Stack to (N, H, W, C), convert to float, permute to (N, C, H, W)
    arr = np.stack(frames)  # (N, H, W, C)
    tensor = torch.from_numpy(arr).float().div(255.0)
    tensor = tensor.permute(0, 3, 1, 2)  # (N, C, H, W)
    return tensor.to(device)


def tensor_to_frames(tensor: torch.Tensor) -> list[np.ndarray]:
    """Convert NCHW float32 tensor in [0,1] to list of HWC uint8 RGB frames."""
    tensor = tensor.detach().cpu().clamp(0, 1)
    tensor = tensor.permute(0, 2, 3, 1)  # (N, H, W, C)
    arr = (tensor * 255).to(torch.uint8).numpy()
    return [arr[i] for i in range(arr.shape[0])]


def iter_batches(items: Sequence[T], batch_size: int) -> list[list[T]]:
    """Split items into batches."""
    return [
        list(items[i : i + batch_size])
        for i in range(0, len(items), batch_size)
    ]
```

**Step 4: Run test to verify it passes**

Run: `uv run pytest tests/core/test_batch.py -v`
Expected: 3 passed

**Step 5: Commit**

```bash
git add backend/core/batch.py tests/core/test_batch.py
git commit -m "feat: add batch processing utilities (frame/tensor conversion)"
```

---

## Phase 2: Model Wrappers

### Task 7: YOLO Wrapper

**Files:**
- Create: `backend/models/yolo_wrapper.py`
- Test: `tests/models/test_yolo_wrapper.py`

**Step 1: Write the failing test**

```python
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
        # Random image 640x640
        img = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        results = yolo.detect(img)
        assert isinstance(results, list)
        # Each detection has: label, confidence, bbox
        if len(results) > 0:
            det = results[0]
            assert "label" in det
            assert "confidence" in det
            assert "bbox" in det

    def test_get_inner_model_returns_nn_module(self, yolo):
        import torch.nn as nn
        inner = yolo.get_inner_model()
        assert isinstance(inner, nn.Module)
```

**Step 2: Run test to verify it fails**

Run: `uv run pytest tests/models/test_yolo_wrapper.py -v`
Expected: FAIL

**Step 3: Write minimal implementation**

```python
# backend/models/yolo_wrapper.py
import numpy as np
import torch
import torch.nn as nn
from ultralytics import YOLO

from backend.config import YOLO_MODEL
from backend.core.device import get_device


class YOLOWrapper:
    """Wrapper around Ultralytics YOLO for detection + gradient access."""

    def __init__(self, model_name: str = YOLO_MODEL):
        self.device = get_device()
        self._yolo = YOLO(model_name)
        # Move to device
        self._yolo.to(self.device)

    @property
    def model(self) -> YOLO:
        return self._yolo

    def get_inner_model(self) -> nn.Module:
        """Return the raw nn.Module for gradient computation (bypasses NMS)."""
        return self._yolo.model

    def detect(self, image: np.ndarray, conf: float = 0.25) -> list[dict]:
        """Run detection on a single RGB image. Returns list of detections."""
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
        """Run detection on a batch of RGB images."""
        return [self.detect(img, conf=conf) for img in images]
```

**Step 4: Run test to verify it passes**

Run: `uv run pytest tests/models/test_yolo_wrapper.py -v`
Expected: 3 passed (first run will download yolov8n.pt ~6MB)

**Step 5: Commit**

```bash
git add backend/models/yolo_wrapper.py tests/models/test_yolo_wrapper.py
git commit -m "feat: add YOLO wrapper with detection and gradient access"
```

---

### Task 8: CLIP Wrapper

**Files:**
- Create: `backend/models/clip_wrapper.py`
- Test: `tests/models/test_clip_wrapper.py`

**Step 1: Write the failing test**

```python
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
        assert abs(sum(scores.values()) - 1.0) < 0.01  # softmax sums to 1

    def test_encode_image_preserves_grad(self, clip_model):
        tensor = torch.randn(1, 3, 224, 224, device=clip_model.device, requires_grad=True)
        embedding = clip_model.encode_image_differentiable(tensor)
        assert embedding.requires_grad
        assert embedding.shape[1] == 512  # ViT-B/32 embedding dim
```

**Step 2: Run test to verify it fails**

Run: `uv run pytest tests/models/test_clip_wrapper.py -v`
Expected: FAIL

**Step 3: Write minimal implementation**

```python
# backend/models/clip_wrapper.py
import clip
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from backend.config import CLIP_MODEL
from backend.core.device import get_device


# CLIP normalization constants (ImageNet) — must be in compute graph for attacks
CLIP_MEAN = torch.tensor([0.48145466, 0.4578275, 0.40821073]).view(1, 3, 1, 1)
CLIP_STD = torch.tensor([0.26862954, 0.26130258, 0.27577711]).view(1, 3, 1, 1)


class CLIPWrapper:
    """Wrapper around OpenAI CLIP with differentiable forward pass."""

    def __init__(self, model_name: str = CLIP_MODEL):
        self.device = get_device()
        self.model, self._preprocess = clip.load(model_name, device=self.device)
        self.model.eval()
        # Move normalization constants to device
        self._mean = CLIP_MEAN.to(self.device)
        self._std = CLIP_STD.to(self.device)

    def _normalize(self, tensor: torch.Tensor) -> torch.Tensor:
        """Apply CLIP normalization inside the compute graph."""
        return (tensor - self._mean) / self._std

    def encode_image_differentiable(self, tensor: torch.Tensor) -> torch.Tensor:
        """Encode NCHW float tensor [0,1] to CLIP image embeddings. Grad-safe."""
        # Resize to 224x224 if needed
        if tensor.shape[-2:] != (224, 224):
            tensor = F.interpolate(tensor, size=(224, 224), mode="bilinear", align_corners=False)
        normed = self._normalize(tensor)
        return self.model.encode_image(normed.to(self.model.dtype))

    def encode_text(self, texts: list[str]) -> torch.Tensor:
        """Encode text labels to CLIP text embeddings."""
        tokens = clip.tokenize(texts).to(self.device)
        with torch.no_grad():
            return self.model.encode_text(tokens).float()

    def classify(self, image: np.ndarray, labels: list[str]) -> dict[str, float]:
        """Classify a single RGB image against text labels. Returns softmax scores."""
        pil_img = Image.fromarray(image)
        img_tensor = self._preprocess(pil_img).unsqueeze(0).to(self.device)

        with torch.no_grad():
            img_features = self.model.encode_image(img_tensor).float()
            text_features = self.encode_text(labels)

            img_features = F.normalize(img_features, dim=-1)
            text_features = F.normalize(text_features, dim=-1)

            similarity = (img_features @ text_features.T).squeeze(0)
            probs = F.softmax(similarity * 100, dim=0)  # CLIP logit_scale ~ 100

        return {label: float(probs[i]) for i, label in enumerate(labels)}
```

**Step 4: Run test to verify it passes**

Run: `uv run pytest tests/models/test_clip_wrapper.py -v`
Expected: 3 passed (first run downloads ViT-B/32 ~350MB)

**Step 5: Commit**

```bash
git add backend/models/clip_wrapper.py tests/models/test_clip_wrapper.py
git commit -m "feat: add CLIP wrapper with differentiable encoding for attacks"
```

---

### Task 9: Model Cache (Singleton)

**Files:**
- Create: `backend/models/cache.py`
- Test: `tests/models/test_cache.py`

**Step 1: Write the failing test**

```python
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
```

**Step 2: Run test to verify it fails**

Run: `uv run pytest tests/models/test_cache.py -v`
Expected: FAIL

**Step 3: Write minimal implementation**

```python
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
```

**Step 4: Run test to verify it passes**

Run: `uv run pytest tests/models/test_cache.py -v`
Expected: 2 passed

**Step 5: Commit**

```bash
git add backend/models/cache.py tests/models/test_cache.py
git commit -m "feat: add singleton model cache for YOLO and CLIP"
```

---

## Phase 3: Attack Engines

### Task 10: FGSM/PGD Attack on YOLO via ART

**Files:**
- Create: `backend/attacks/adversarial.py`
- Test: `tests/attacks/test_adversarial.py`

**Step 1: Write the failing test**

```python
# tests/attacks/test_adversarial.py
import numpy as np
import torch
import pytest
from backend.attacks.adversarial import attack_yolo_fgsm, attack_yolo_pgd
from backend.core.device import get_device


@pytest.fixture(scope="module")
def sample_frames():
    """4 random frames at 640x640 (YOLO input size)."""
    return [np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8) for _ in range(4)]


class TestYOLOAttacks:
    def test_fgsm_returns_perturbed_frames(self, sample_frames):
        perturbed = attack_yolo_fgsm(sample_frames, epsilon=4/255)
        assert len(perturbed) == len(sample_frames)
        assert perturbed[0].shape == sample_frames[0].shape
        assert perturbed[0].dtype == np.uint8
        # Should be different from originals
        assert not np.array_equal(perturbed[0], sample_frames[0])

    def test_fgsm_respects_epsilon_bound(self, sample_frames):
        eps = 4 / 255
        perturbed = attack_yolo_fgsm(sample_frames, epsilon=eps)
        diff = np.abs(perturbed[0].astype(float) - sample_frames[0].astype(float)) / 255
        assert diff.max() <= eps + 1e-6

    def test_pgd_returns_perturbed_frames(self, sample_frames):
        perturbed = attack_yolo_pgd(sample_frames, epsilon=4/255, steps=3, alpha=1/255)
        assert len(perturbed) == len(sample_frames)
        assert not np.array_equal(perturbed[0], sample_frames[0])
```

**Step 2: Run test to verify it fails**

Run: `uv run pytest tests/attacks/test_adversarial.py -v`
Expected: FAIL

**Step 3: Write minimal implementation**

```python
# backend/attacks/adversarial.py
"""Gradient-based adversarial attacks on YOLO and CLIP."""

import numpy as np
import torch
import torch.nn.functional as F

from backend.core.device import get_device
from backend.core.batch import frames_to_tensor, tensor_to_frames
from backend.models.cache import get_yolo, get_clip
from backend.config import YOLO_INPUT_SIZE


def _yolo_loss(model_inner: torch.nn.Module, images: torch.Tensor) -> torch.Tensor:
    """Compute YOLO objectness loss for suppression attack.

    Minimizing this makes objects disappear from detections.
    Uses raw model output before NMS to preserve gradients.
    """
    device = images.device
    # Resize to YOLO input size
    resized = F.interpolate(images, size=(YOLO_INPUT_SIZE, YOLO_INPUT_SIZE),
                            mode="bilinear", align_corners=False)

    with torch.enable_grad():
        preds = model_inner(resized)
        # ultralytics raw output: list of tensors or tuple
        # The objectness scores are in the prediction tensor
        if isinstance(preds, (list, tuple)):
            pred = preds[0]
        else:
            pred = preds

        # For YOLOv8: output shape is (batch, num_classes+4, num_anchors)
        # Objectness is implicit in class scores — sum all class confidences
        if pred.dim() == 3:
            # (B, C+4, N) -> take class scores (skip first 4 = bbox)
            class_scores = pred[:, 4:, :]
            loss = class_scores.sigmoid().sum()
        else:
            loss = pred.sigmoid().sum()

    return loss


def attack_yolo_fgsm(
    frames: list[np.ndarray],
    epsilon: float = 4 / 255,
) -> list[np.ndarray]:
    """FGSM attack on YOLO — single gradient step to suppress detections."""
    device = get_device()
    yolo = get_yolo()
    model_inner = yolo.get_inner_model()
    model_inner.eval()

    tensor = frames_to_tensor(frames, device)
    tensor.requires_grad_(True)

    loss = _yolo_loss(model_inner, tensor)
    loss.backward()

    # FGSM: perturb in direction of gradient sign (to maximize loss = suppress)
    # We MINIMIZE objectness, so we go AGAINST the gradient
    perturbation = -epsilon * tensor.grad.sign()
    adv_tensor = (tensor + perturbation).clamp(0, 1)

    if device.type == "mps":
        torch.mps.empty_cache()

    return tensor_to_frames(adv_tensor)


def attack_yolo_pgd(
    frames: list[np.ndarray],
    epsilon: float = 4 / 255,
    steps: int = 20,
    alpha: float = 1 / 255,
) -> list[np.ndarray]:
    """PGD attack on YOLO — iterative gradient descent to suppress detections."""
    device = get_device()
    yolo = get_yolo()
    model_inner = yolo.get_inner_model()
    model_inner.eval()

    original = frames_to_tensor(frames, device)
    adv = original.clone().detach()

    for _ in range(steps):
        adv.requires_grad_(True)
        loss = _yolo_loss(model_inner, adv)
        loss.backward()

        with torch.no_grad():
            adv = adv - alpha * adv.grad.sign()
            # Project back to epsilon-ball
            delta = (adv - original).clamp(-epsilon, epsilon)
            adv = (original + delta).clamp(0, 1)

    if device.type == "mps":
        torch.mps.empty_cache()

    return tensor_to_frames(adv)
```

**Step 4: Run test to verify it passes**

Run: `uv run pytest tests/attacks/test_adversarial.py -v`
Expected: 3 passed

**Step 5: Commit**

```bash
git add backend/attacks/adversarial.py tests/attacks/test_adversarial.py
git commit -m "feat: add FGSM and PGD adversarial attacks on YOLO"
```

---

### Task 11: FGSM/PGD Attack on CLIP

**Files:**
- Modify: `backend/attacks/adversarial.py`
- Test: `tests/attacks/test_adversarial_clip.py`

**Step 1: Write the failing test**

```python
# tests/attacks/test_adversarial_clip.py
import numpy as np
import pytest
from backend.attacks.adversarial import attack_clip_fgsm, attack_clip_pgd


@pytest.fixture(scope="module")
def sample_frames():
    return [np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8) for _ in range(2)]


class TestCLIPAttacks:
    def test_fgsm_returns_perturbed_frames(self, sample_frames):
        perturbed = attack_clip_fgsm(
            sample_frames,
            target_text="a dolphin swimming in the ocean",
            epsilon=4/255,
        )
        assert len(perturbed) == 2
        assert not np.array_equal(perturbed[0], sample_frames[0])

    def test_pgd_returns_perturbed_frames(self, sample_frames):
        perturbed = attack_clip_pgd(
            sample_frames,
            target_text="a dolphin swimming in the ocean",
            epsilon=4/255,
            steps=3,
            alpha=1/255,
        )
        assert len(perturbed) == 2
        assert perturbed[0].dtype == np.uint8
```

**Step 2: Run test to verify it fails**

Run: `uv run pytest tests/attacks/test_adversarial_clip.py -v`
Expected: FAIL

**Step 3: Add CLIP attack functions to `backend/attacks/adversarial.py`**

Append to `backend/attacks/adversarial.py`:

```python
def _clip_loss(
    clip_wrapper,
    images: torch.Tensor,
    target_embedding: torch.Tensor,
) -> torch.Tensor:
    """Compute negative cosine similarity to target text.

    Minimizing this pushes the image embedding toward the target text.
    """
    img_emb = clip_wrapper.encode_image_differentiable(images).float()
    img_emb = F.normalize(img_emb, dim=-1)
    target_emb = F.normalize(target_embedding, dim=-1)
    # Negative similarity — minimize to increase similarity to target
    return -(img_emb @ target_emb.T).mean()


def attack_clip_fgsm(
    frames: list[np.ndarray],
    target_text: str,
    epsilon: float = 4 / 255,
) -> list[np.ndarray]:
    """FGSM attack on CLIP — redirect semantic classification toward target text."""
    device = get_device()
    clip_model = get_clip()

    target_emb = clip_model.encode_text([target_text])
    tensor = frames_to_tensor(frames, device)
    tensor.requires_grad_(True)

    loss = _clip_loss(clip_model, tensor, target_emb)
    loss.backward()

    perturbation = -epsilon * tensor.grad.sign()
    adv_tensor = (tensor + perturbation).clamp(0, 1)

    if device.type == "mps":
        torch.mps.empty_cache()

    return tensor_to_frames(adv_tensor)


def attack_clip_pgd(
    frames: list[np.ndarray],
    target_text: str,
    epsilon: float = 4 / 255,
    steps: int = 20,
    alpha: float = 1 / 255,
) -> list[np.ndarray]:
    """PGD attack on CLIP — iterative redirect toward target text."""
    device = get_device()
    clip_model = get_clip()

    target_emb = clip_model.encode_text([target_text])
    original = frames_to_tensor(frames, device)
    adv = original.clone().detach()

    for _ in range(steps):
        adv.requires_grad_(True)
        loss = _clip_loss(clip_model, adv, target_emb)
        loss.backward()

        with torch.no_grad():
            adv = adv - alpha * adv.grad.sign()
            delta = (adv - original).clamp(-epsilon, epsilon)
            adv = (original + delta).clamp(0, 1)

    if device.type == "mps":
        torch.mps.empty_cache()

    return tensor_to_frames(adv)
```

**Step 4: Run test to verify it passes**

Run: `uv run pytest tests/attacks/test_adversarial_clip.py -v`
Expected: 2 passed

**Step 5: Commit**

```bash
git add backend/attacks/adversarial.py tests/attacks/test_adversarial_clip.py
git commit -m "feat: add FGSM and PGD adversarial attacks on CLIP"
```

---

### Task 12: Combined Multi-Objective Attack (YOLO + CLIP)

**Files:**
- Create: `backend/attacks/combined.py`
- Test: `tests/attacks/test_combined.py`

**Step 1: Write the failing test**

```python
# tests/attacks/test_combined.py
import numpy as np
import pytest
from backend.attacks.combined import attack_combined_pgd


@pytest.fixture(scope="module")
def sample_frames():
    return [np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8) for _ in range(2)]


class TestCombinedAttack:
    def test_combined_returns_perturbed_frames(self, sample_frames):
        perturbed = attack_combined_pgd(
            sample_frames,
            target_text="an empty room",
            yolo_weight=0.5,
            clip_weight=0.5,
            epsilon=4/255,
            steps=3,
        )
        assert len(perturbed) == 2
        assert not np.array_equal(perturbed[0], sample_frames[0])

    def test_yolo_only_weight(self, sample_frames):
        perturbed = attack_combined_pgd(
            sample_frames,
            target_text="an empty room",
            yolo_weight=1.0,
            clip_weight=0.0,
            epsilon=4/255,
            steps=2,
        )
        assert len(perturbed) == 2
```

**Step 2: Run test to verify it fails**

Run: `uv run pytest tests/attacks/test_combined.py -v`
Expected: FAIL

**Step 3: Write minimal implementation**

```python
# backend/attacks/combined.py
"""Combined multi-objective adversarial attack (YOLO suppression + CLIP redirection)."""

import numpy as np
import torch
import torch.nn.functional as F

from backend.core.device import get_device
from backend.core.batch import frames_to_tensor, tensor_to_frames
from backend.models.cache import get_yolo, get_clip
from backend.attacks.adversarial import _yolo_loss, _clip_loss


def attack_combined_pgd(
    frames: list[np.ndarray],
    target_text: str,
    yolo_weight: float = 0.5,
    clip_weight: float = 0.5,
    epsilon: float = 4 / 255,
    steps: int = 20,
    alpha: float = 1 / 255,
) -> list[np.ndarray]:
    """PGD with combined YOLO+CLIP objective.

    Computes gradients from each objective separately, normalizes by L2 norm
    to equalize scales, then combines with weights.
    """
    device = get_device()
    yolo = get_yolo()
    clip_model = get_clip()
    model_inner = yolo.get_inner_model()
    model_inner.eval()

    target_emb = clip_model.encode_text([target_text])
    original = frames_to_tensor(frames, device)
    adv = original.clone().detach()

    for _ in range(steps):
        grad_total = torch.zeros_like(original)

        if yolo_weight > 0:
            adv_y = adv.clone().detach().requires_grad_(True)
            loss_y = _yolo_loss(model_inner, adv_y)
            loss_y.backward()
            g_y = adv_y.grad
            # Normalize by L2 norm
            norm_y = g_y.norm(p=2) + 1e-12
            grad_total += yolo_weight * (g_y / norm_y)

        if clip_weight > 0:
            adv_c = adv.clone().detach().requires_grad_(True)
            loss_c = _clip_loss(clip_model, adv_c, target_emb)
            loss_c.backward()
            g_c = adv_c.grad
            norm_c = g_c.norm(p=2) + 1e-12
            grad_total += clip_weight * (g_c / norm_c)

        with torch.no_grad():
            adv = adv - alpha * grad_total.sign()
            delta = (adv - original).clamp(-epsilon, epsilon)
            adv = (original + delta).clamp(0, 1)

    if device.type == "mps":
        torch.mps.empty_cache()

    return tensor_to_frames(adv)
```

**Step 4: Run test to verify it passes**

Run: `uv run pytest tests/attacks/test_combined.py -v`
Expected: 2 passed

**Step 5: Commit**

```bash
git add backend/attacks/combined.py tests/attacks/test_combined.py
git commit -m "feat: add combined YOLO+CLIP multi-objective PGD attack"
```

---

### Task 13: LSB Steganography

**Files:**
- Create: `backend/attacks/lsb.py`
- Test: `tests/attacks/test_lsb.py`

**Step 1: Write the failing test**

```python
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
        # Max difference should be 1 (LSB change)
        diff = np.abs(result[0].astype(int) - frames[0].astype(int))
        assert diff.max() <= 1

    def test_cloak_biases_dark_regions_to_even(self):
        # Dark frame (luminance < 60)
        dark = np.full((64, 64, 3), 30, dtype=np.uint8)
        result = lsb_cloak([dark], intensity=1.0)
        # Most LSBs in result should be 0 (even values)
        lsb_sum = (result[0].astype(int) % 2).sum()
        total = result[0].size
        even_ratio = 1 - (lsb_sum / total)
        assert even_ratio > 0.6  # strong even bias


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
```

**Step 2: Run test to verify it fails**

Run: `uv run pytest tests/attacks/test_lsb.py -v`
Expected: FAIL

**Step 3: Write minimal implementation**

```python
# backend/attacks/lsb.py
"""LSB steganography attacks — content-aware cloaking and pseudo-random embedding."""

import numpy as np


def lsb_cloak(
    frames: list[np.ndarray],
    intensity: float = 0.5,
    dark_threshold: int = 60,
) -> list[np.ndarray]:
    """Content-aware LSB cloaking.

    Biases LSBs toward even values in dark regions (luminance < threshold).
    Intensity controls probability of modification (0=none, 1=all eligible pixels).
    """
    result = []
    for frame in frames:
        out = frame.copy()
        # Compute luminance (simple average across channels)
        luminance = out.mean(axis=2)

        # Create mask: dark regions eligible for even-bias
        dark_mask = luminance < dark_threshold
        # Probabilistic: only modify `intensity` fraction of eligible pixels
        rng = np.random.default_rng()
        prob_mask = rng.random(dark_mask.shape) < intensity
        modify_mask = dark_mask & prob_mask

        # Force LSB to 0 (even) in masked pixels, all channels
        for c in range(3):
            channel = out[:, :, c]
            channel[modify_mask] = channel[modify_mask] & 0xFE  # clear LSB
        result.append(out)
    return result


def lsb_embed(
    frames: list[np.ndarray],
    seed: int = 42,
) -> list[np.ndarray]:
    """Pseudo-random LSB embedding with deterministic seed.

    Flips LSBs of all pixels according to a deterministic PRNG pattern.
    """
    rng = np.random.default_rng(seed)
    result = []
    for frame in frames:
        out = frame.copy()
        # Generate random bit pattern (0 or 1) for each pixel channel
        noise = rng.integers(0, 2, size=frame.shape, dtype=np.uint8)
        # Set LSB to the random pattern
        out = (out & 0xFE) | noise
        result.append(out)
    return result
```

**Step 4: Run test to verify it passes**

Run: `uv run pytest tests/attacks/test_lsb.py -v`
Expected: 5 passed

**Step 5: Commit**

```bash
git add backend/attacks/lsb.py tests/attacks/test_lsb.py
git commit -m "feat: add LSB steganography (content-aware cloak + pseudo-random embed)"
```

---

### Task 14: Universal Adversarial Perturbation (UAP)

**Files:**
- Create: `backend/attacks/uap.py`
- Test: `tests/attacks/test_uap.py`

**Step 1: Write the failing test**

```python
# tests/attacks/test_uap.py
import numpy as np
import pytest
from backend.attacks.uap import compute_uap, apply_uap


@pytest.fixture(scope="module")
def sample_frames():
    return [np.random.randint(0, 255, (240, 320, 3), dtype=np.uint8) for _ in range(4)]


class TestUAP:
    def test_compute_uap_returns_perturbation(self, sample_frames):
        uap = compute_uap(sample_frames[:2], target="yolo", steps=2, epsilon=4/255)
        assert isinstance(uap, np.ndarray)
        assert uap.shape == sample_frames[0].shape
        assert uap.dtype == np.float32

    def test_apply_uap_modifies_frames(self, sample_frames):
        uap = np.random.uniform(-4/255, 4/255, sample_frames[0].shape).astype(np.float32)
        result = apply_uap(sample_frames, uap)
        assert len(result) == 4
        assert result[0].dtype == np.uint8
        assert not np.array_equal(result[0], sample_frames[0])
```

**Step 2: Run test to verify it fails**

Run: `uv run pytest tests/attacks/test_uap.py -v`
Expected: FAIL

**Step 3: Write minimal implementation**

```python
# backend/attacks/uap.py
"""Universal Adversarial Perturbation — one perturbation for all frames."""

import numpy as np
import torch
import torch.nn.functional as F

from backend.core.device import get_device
from backend.core.batch import frames_to_tensor, tensor_to_frames
from backend.models.cache import get_yolo, get_clip
from backend.attacks.adversarial import _yolo_loss, _clip_loss


def compute_uap(
    sample_frames: list[np.ndarray],
    target: str = "yolo",
    target_text: str = "",
    epsilon: float = 4 / 255,
    steps: int = 20,
    alpha: float = 1 / 255,
) -> np.ndarray:
    """Compute a Universal Adversarial Perturbation from sample frames.

    Args:
        sample_frames: Representative frames to optimize UAP against.
        target: "yolo", "clip", or "combined".
        target_text: Required if target includes "clip".
        epsilon: L-inf perturbation bound.
        steps: PGD iterations.
        alpha: Step size.

    Returns:
        float32 numpy array of shape (H, W, 3) — the universal perturbation in [-epsilon, epsilon].
    """
    device = get_device()
    h, w = sample_frames[0].shape[:2]

    # Initialize UAP as zeros
    uap = torch.zeros(1, 3, h, w, device=device, requires_grad=False)
    originals = frames_to_tensor(sample_frames, device)

    yolo = get_yolo() if target in ("yolo", "combined") else None
    clip_model = get_clip() if target in ("clip", "combined") else None
    target_emb = clip_model.encode_text([target_text]) if clip_model and target_text else None

    model_inner = yolo.get_inner_model() if yolo else None
    if model_inner:
        model_inner.eval()

    for _ in range(steps):
        uap_param = uap.clone().detach().requires_grad_(True)
        # Apply same UAP to all frames
        adv = (originals + uap_param).clamp(0, 1)

        loss = torch.tensor(0.0, device=device)
        if target in ("yolo", "combined") and model_inner is not None:
            loss = loss + _yolo_loss(model_inner, adv)
        if target in ("clip", "combined") and clip_model is not None and target_emb is not None:
            loss = loss + _clip_loss(clip_model, adv, target_emb)

        loss.backward()

        with torch.no_grad():
            uap = uap - alpha * uap_param.grad.sign()
            uap = uap.clamp(-epsilon, epsilon)

    if device.type == "mps":
        torch.mps.empty_cache()

    # Convert to numpy (H, W, C)
    return uap.squeeze(0).permute(1, 2, 0).cpu().numpy().astype(np.float32)


def apply_uap(
    frames: list[np.ndarray],
    uap: np.ndarray,
) -> list[np.ndarray]:
    """Apply a precomputed UAP to all frames.

    Args:
        frames: List of HWC uint8 RGB frames.
        uap: float32 perturbation array (H, W, 3) in pixel range [-eps, eps].
    """
    result = []
    for frame in frames:
        perturbed = frame.astype(np.float32) / 255.0 + uap
        perturbed = np.clip(perturbed * 255, 0, 255).astype(np.uint8)
        result.append(perturbed)
    return result
```

**Step 4: Run test to verify it passes**

Run: `uv run pytest tests/attacks/test_uap.py -v`
Expected: 2 passed

**Step 5: Commit**

```bash
git add backend/attacks/uap.py tests/attacks/test_uap.py
git commit -m "feat: add Universal Adversarial Perturbation (compute + apply)"
```

---

### Task 15: Temporal Consistency (Keyframe Propagation)

**Files:**
- Create: `backend/attacks/temporal.py`
- Test: `tests/attacks/test_temporal.py`

**Step 1: Write the failing test**

```python
# tests/attacks/test_temporal.py
import numpy as np
from backend.attacks.temporal import propagate_keyframes, select_keyframe_indices


class TestKeyframeSelection:
    def test_select_indices(self):
        indices = select_keyframe_indices(total_frames=20, interval=5)
        assert indices == [0, 5, 10, 15, 19]  # always include last frame

    def test_interval_1_selects_all(self):
        indices = select_keyframe_indices(total_frames=5, interval=1)
        assert indices == [0, 1, 2, 3, 4]


class TestPropagateKeyframes:
    def test_interpolates_between_keyframes(self):
        originals = [np.full((4, 4, 3), i * 10, dtype=np.uint8) for i in range(10)]
        # Perturbed keyframes at 0, 5, 9
        perturbed_keys = {
            0: np.full((4, 4, 3), 100, dtype=np.uint8),
            5: np.full((4, 4, 3), 200, dtype=np.uint8),
            9: np.full((4, 4, 3), 150, dtype=np.uint8),
        }
        result = propagate_keyframes(originals, perturbed_keys)
        assert len(result) == 10
        # Keyframes should match exactly
        assert np.array_equal(result[0], perturbed_keys[0])
        assert np.array_equal(result[5], perturbed_keys[5])
        # Intermediate frames should be interpolated (not identical to originals)
        assert not np.array_equal(result[2], originals[2])
```

**Step 2: Run test to verify it fails**

Run: `uv run pytest tests/attacks/test_temporal.py -v`
Expected: FAIL

**Step 3: Write minimal implementation**

```python
# backend/attacks/temporal.py
"""Temporal consistency for video adversarial attacks."""

import numpy as np


def select_keyframe_indices(total_frames: int, interval: int) -> list[int]:
    """Select keyframe indices at regular intervals, always including last frame."""
    if interval <= 0:
        raise ValueError("interval must be >= 1")
    if interval == 1:
        return list(range(total_frames))
    indices = list(range(0, total_frames, interval))
    if indices[-1] != total_frames - 1:
        indices.append(total_frames - 1)
    return indices


def propagate_keyframes(
    originals: list[np.ndarray],
    perturbed_keys: dict[int, np.ndarray],
) -> list[np.ndarray]:
    """Interpolate perturbation deltas between keyframes.

    For each keyframe, computes delta = perturbed - original.
    For intermediate frames, linearly interpolates the delta and applies to original.
    """
    n = len(originals)
    key_indices = sorted(perturbed_keys.keys())
    result = [None] * n

    # Place keyframes directly
    for idx in key_indices:
        result[idx] = perturbed_keys[idx]

    # Compute deltas at keyframes
    deltas = {}
    for idx in key_indices:
        deltas[idx] = perturbed_keys[idx].astype(np.float32) - originals[idx].astype(np.float32)

    # Interpolate between consecutive keyframes
    for i in range(len(key_indices) - 1):
        start = key_indices[i]
        end = key_indices[i + 1]
        if end - start <= 1:
            continue
        d_start = deltas[start]
        d_end = deltas[end]
        for j in range(start + 1, end):
            t = (j - start) / (end - start)
            interpolated_delta = d_start * (1 - t) + d_end * t
            frame = originals[j].astype(np.float32) + interpolated_delta
            result[j] = np.clip(frame, 0, 255).astype(np.uint8)

    return result
```

**Step 4: Run test to verify it passes**

Run: `uv run pytest tests/attacks/test_temporal.py -v`
Expected: 3 passed

**Step 5: Commit**

```bash
git add backend/attacks/temporal.py tests/attacks/test_temporal.py
git commit -m "feat: add keyframe propagation with delta interpolation"
```

---

## Phase 4: Pipeline

### Task 16: Classifier Module

**Files:**
- Create: `backend/pipeline/classifier.py`
- Test: `tests/pipeline/test_classifier.py`

**Step 1: Write the failing test**

```python
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
```

**Step 2: Run test to verify it fails**

Run: `uv run pytest tests/pipeline/test_classifier.py -v`
Expected: FAIL

**Step 3: Write minimal implementation**

```python
# backend/pipeline/classifier.py
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
```

**Step 4: Run test to verify it passes**

Run: `uv run pytest tests/pipeline/test_classifier.py -v`
Expected: 1 passed

**Step 5: Commit**

```bash
git add backend/pipeline/classifier.py tests/pipeline/test_classifier.py
git commit -m "feat: add classifier module (YOLO + CLIP per frame)"
```

---

### Task 17: Pipeline Orchestrator

**Files:**
- Create: `backend/pipeline/orchestrator.py`
- Test: `tests/pipeline/test_orchestrator.py`

**Step 1: Write the failing test**

```python
# tests/pipeline/test_orchestrator.py
import pytest
from unittest.mock import MagicMock
from backend.pipeline.orchestrator import PipelineConfig, run_pipeline

FIXTURE = "tests/fixtures/test_2s.mp4"


class TestPipelineConfig:
    def test_default_config(self):
        cfg = PipelineConfig(
            video_path=FIXTURE,
            attack_method="fgsm",
            target_text="an empty room",
        )
        assert cfg.video_path == FIXTURE
        assert cfg.attack_method == "fgsm"


class TestRunPipeline:
    def test_full_pipeline_preview(self):
        cfg = PipelineConfig(
            video_path=FIXTURE,
            attack_method="fgsm",
            preset="preview",
            target_text="a sunset over the ocean",
            clip_labels=["a number", "a sunset over the ocean", "a blank screen"],
        )
        progress_calls = []
        def on_progress(pct, stage):
            progress_calls.append((pct, stage))

        result = run_pipeline(cfg, on_progress=on_progress)

        assert "output_path" in result
        assert "original_classifications" in result
        assert "perturbed_classifications" in result
        assert len(progress_calls) > 0
        assert progress_calls[-1][0] == 100
```

**Step 2: Run test to verify it fails**

Run: `uv run pytest tests/pipeline/test_orchestrator.py -v`
Expected: FAIL

**Step 3: Write minimal implementation**

```python
# backend/pipeline/orchestrator.py
"""Main pipeline orchestrator — 5-stage processing."""

import os
import tempfile
from dataclasses import dataclass, field
from collections.abc import Callable

import numpy as np

from backend.config import get_preset, DEFAULT_EPSILON, DEFAULT_ALPHA
from backend.core.video import extract_frames, get_video_info, extract_audio, reconstruct_video
from backend.core.batch import iter_batches
from backend.pipeline.classifier import classify_frames
from backend.attacks.adversarial import (
    attack_yolo_fgsm, attack_yolo_pgd,
    attack_clip_fgsm, attack_clip_pgd,
)
from backend.attacks.combined import attack_combined_pgd
from backend.attacks.lsb import lsb_cloak, lsb_embed
from backend.attacks.uap import compute_uap, apply_uap
from backend.attacks.temporal import select_keyframe_indices, propagate_keyframes


@dataclass
class PipelineConfig:
    video_path: str
    attack_method: str  # "fgsm", "pgd", "lsb", "uap", "combined"
    target_text: str = "an empty room"
    clip_labels: list[str] = field(default_factory=lambda: [
        "a person talking", "an advertisement", "a product",
        "an empty room", "nature scenery", "abstract art",
    ])
    preset: str = "preview"
    epsilon: float = DEFAULT_EPSILON
    alpha: float = DEFAULT_ALPHA
    yolo_weight: float = 0.5
    clip_weight: float = 0.5
    lsb_mode: str = "cloak"     # "cloak" or "embed"
    lsb_intensity: float = 0.5
    output_dir: str = "outputs"


ProgressCallback = Callable[[int, str], None]


def run_pipeline(
    config: PipelineConfig,
    on_progress: ProgressCallback | None = None,
) -> dict:
    """Execute the 5-stage pipeline. Returns result dict."""

    def progress(pct: int, stage: str):
        if on_progress:
            on_progress(pct, stage)

    preset = get_preset(config.preset)
    os.makedirs(config.output_dir, exist_ok=True)

    # === Stage 1: Extract Frames (0-10%) ===
    progress(0, "extracting")
    info = get_video_info(config.video_path)
    frames = list(extract_frames(config.video_path))
    audio_path = extract_audio(
        config.video_path,
        os.path.join(config.output_dir, "temp_audio.aac"),
    )
    progress(10, "extracting")

    # === Stage 2: Original Classification (10-30%) ===
    progress(10, "classifying_original")
    original_cls = classify_frames(frames, config.clip_labels)
    progress(30, "classifying_original")

    # === Stage 3: Perturbation (30-80%) ===
    progress(30, "perturbing")
    perturbed_frames = _run_attack(frames, config, preset, progress)
    progress(80, "perturbing")

    # === Stage 4: Verification (80-90%) ===
    progress(80, "verifying")
    perturbed_cls = classify_frames(perturbed_frames, config.clip_labels)
    progress(90, "verifying")

    # === Stage 5: Reconstruction (90-100%) ===
    progress(90, "reconstructing")
    output_path = os.path.join(config.output_dir, "output.mp4")
    reconstruct_video(perturbed_frames, output_path, info["fps"], audio_path)
    progress(100, "done")

    return {
        "output_path": output_path,
        "video_info": info,
        "original_classifications": original_cls,
        "perturbed_classifications": perturbed_cls,
        "frames_processed": len(frames),
    }


def _run_attack(
    frames: list[np.ndarray],
    config: PipelineConfig,
    preset,
    progress: ProgressCallback,
) -> list[np.ndarray]:
    """Dispatch to the appropriate attack engine."""
    method = config.attack_method

    if method == "lsb":
        if config.lsb_mode == "cloak":
            return lsb_cloak(frames, intensity=config.lsb_intensity)
        else:
            return lsb_embed(frames, seed=42)

    if method == "uap":
        # Use first N frames as sample for UAP computation
        sample = frames[:min(10, len(frames))]
        uap = compute_uap(
            sample, target="yolo", target_text=config.target_text,
            epsilon=config.epsilon, steps=preset.pgd_steps or 10,
        )
        return apply_uap(frames, uap)

    # Gradient-based attacks with keyframe propagation
    key_indices = select_keyframe_indices(len(frames), preset.keyframe_interval)
    keyframes = [frames[i] for i in key_indices]

    # Attack keyframes in batches
    perturbed_keys_list = []
    batches = iter_batches(keyframes, preset.batch_size)
    for batch in batches:
        if method == "fgsm":
            perturbed_batch = attack_yolo_fgsm(batch, config.epsilon)
        elif method == "pgd":
            perturbed_batch = attack_yolo_pgd(
                batch, config.epsilon, preset.pgd_steps, config.alpha,
            )
        elif method == "combined":
            perturbed_batch = attack_combined_pgd(
                batch, config.target_text,
                config.yolo_weight, config.clip_weight,
                config.epsilon, preset.pgd_steps, config.alpha,
            )
        else:
            raise ValueError(f"Unknown attack method: {method}")
        perturbed_keys_list.extend(perturbed_batch)

    # Build keyframe dict and propagate
    perturbed_keys = {idx: perturbed_keys_list[i] for i, idx in enumerate(key_indices)}
    return propagate_keyframes(frames, perturbed_keys)
```

**Step 4: Run test to verify it passes**

Run: `uv run pytest tests/pipeline/test_orchestrator.py -v`
Expected: 2 passed

**Step 5: Commit**

```bash
git add backend/pipeline/orchestrator.py tests/pipeline/test_orchestrator.py
git commit -m "feat: add 5-stage pipeline orchestrator with all attack methods"
```

---

## Phase 5: API Layer

### Task 18: Job Manager

**Files:**
- Create: `backend/core/jobs.py`
- Test: `tests/core/test_jobs.py`

**Step 1: Write the failing test**

```python
# tests/core/test_jobs.py
from backend.core.jobs import JobManager, JobStatus


class TestJobManager:
    def test_create_job(self):
        mgr = JobManager()
        job_id = mgr.create("video_123", {"method": "fgsm"})
        assert isinstance(job_id, str)
        job = mgr.get(job_id)
        assert job["status"] == JobStatus.PENDING
        assert job["video_id"] == "video_123"

    def test_update_progress(self):
        mgr = JobManager()
        job_id = mgr.create("v1", {})
        mgr.update_progress(job_id, 50, "perturbing")
        job = mgr.get(job_id)
        assert job["progress"] == 50
        assert job["stage"] == "perturbing"

    def test_complete_job(self):
        mgr = JobManager()
        job_id = mgr.create("v1", {})
        mgr.complete(job_id, {"output_path": "/tmp/out.mp4"})
        job = mgr.get(job_id)
        assert job["status"] == JobStatus.COMPLETED
        assert job["result"]["output_path"] == "/tmp/out.mp4"

    def test_fail_job(self):
        mgr = JobManager()
        job_id = mgr.create("v1", {})
        mgr.fail(job_id, "Something went wrong")
        job = mgr.get(job_id)
        assert job["status"] == JobStatus.FAILED
        assert "Something went wrong" in job["error"]

    def test_unknown_job_returns_none(self):
        mgr = JobManager()
        assert mgr.get("nonexistent") is None
```

**Step 2: Run test to verify it fails**

Run: `uv run pytest tests/core/test_jobs.py -v`
Expected: FAIL

**Step 3: Write minimal implementation**

```python
# backend/core/jobs.py
"""In-memory job manager with thread-safe operations."""

import threading
import uuid
from enum import StrEnum


class JobStatus(StrEnum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"


class JobManager:
    def __init__(self):
        self._jobs: dict[str, dict] = {}
        self._lock = threading.Lock()

    def create(self, video_id: str, config: dict) -> str:
        job_id = uuid.uuid4().hex[:12]
        with self._lock:
            self._jobs[job_id] = {
                "job_id": job_id,
                "video_id": video_id,
                "config": config,
                "status": JobStatus.PENDING,
                "progress": 0,
                "stage": "pending",
                "result": None,
                "error": None,
            }
        return job_id

    def get(self, job_id: str) -> dict | None:
        with self._lock:
            job = self._jobs.get(job_id)
            return dict(job) if job else None

    def update_progress(self, job_id: str, progress: int, stage: str) -> None:
        with self._lock:
            if job_id in self._jobs:
                self._jobs[job_id]["progress"] = progress
                self._jobs[job_id]["stage"] = stage
                self._jobs[job_id]["status"] = JobStatus.RUNNING

    def complete(self, job_id: str, result: dict) -> None:
        with self._lock:
            if job_id in self._jobs:
                self._jobs[job_id]["status"] = JobStatus.COMPLETED
                self._jobs[job_id]["progress"] = 100
                self._jobs[job_id]["stage"] = "done"
                self._jobs[job_id]["result"] = result

    def fail(self, job_id: str, error: str) -> None:
        with self._lock:
            if job_id in self._jobs:
                self._jobs[job_id]["status"] = JobStatus.FAILED
                self._jobs[job_id]["error"] = error
```

**Step 4: Run test to verify it passes**

Run: `uv run pytest tests/core/test_jobs.py -v`
Expected: 5 passed

**Step 5: Commit**

```bash
git add backend/core/jobs.py tests/core/test_jobs.py
git commit -m "feat: add thread-safe in-memory job manager"
```

---

### Task 19: Storage Manager

**Files:**
- Create: `backend/storage/manager.py`
- Test: `tests/test_storage.py`

**Step 1: Write the failing test**

```python
# tests/test_storage.py
import os
import tempfile
from backend.storage.manager import StorageManager


class TestStorageManager:
    def test_save_upload(self, tmp_path):
        sm = StorageManager(upload_dir=str(tmp_path / "uploads"), output_dir=str(tmp_path / "outputs"))
        content = b"fake video data"
        video_id = sm.save_upload("test.mp4", content)
        assert os.path.exists(sm.get_upload_path(video_id))

    def test_get_output_dir_creates_dir(self, tmp_path):
        sm = StorageManager(upload_dir=str(tmp_path / "uploads"), output_dir=str(tmp_path / "outputs"))
        job_dir = sm.get_output_dir("job_123")
        assert os.path.isdir(job_dir)
```

**Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_storage.py -v`
Expected: FAIL

**Step 3: Write minimal implementation**

```python
# backend/storage/manager.py
import os
import uuid


class StorageManager:
    def __init__(self, upload_dir: str = "uploads", output_dir: str = "outputs"):
        self.upload_dir = upload_dir
        self.output_dir = output_dir
        os.makedirs(upload_dir, exist_ok=True)
        os.makedirs(output_dir, exist_ok=True)

    def save_upload(self, filename: str, content: bytes) -> str:
        video_id = uuid.uuid4().hex[:12]
        ext = os.path.splitext(filename)[1] or ".mp4"
        path = os.path.join(self.upload_dir, f"{video_id}{ext}")
        with open(path, "wb") as f:
            f.write(content)
        return video_id

    def get_upload_path(self, video_id: str) -> str:
        for f in os.listdir(self.upload_dir):
            if f.startswith(video_id):
                return os.path.join(self.upload_dir, f)
        raise FileNotFoundError(f"Upload not found: {video_id}")

    def get_output_dir(self, job_id: str) -> str:
        path = os.path.join(self.output_dir, job_id)
        os.makedirs(path, exist_ok=True)
        return path
```

**Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_storage.py -v`
Expected: 2 passed

**Step 5: Commit**

```bash
git add backend/storage/manager.py tests/test_storage.py
git commit -m "feat: add upload/output storage manager"
```

---

### Task 20: REST API Routes

**Files:**
- Create: `backend/api/routes.py`
- Test: `tests/api/test_routes.py`

**Step 1: Write the failing test**

```python
# tests/api/test_routes.py
import pytest
from httpx import AsyncClient, ASGITransport
from backend.main import create_app


@pytest.fixture
def app():
    return create_app()


@pytest.fixture
async def client(app):
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as c:
        yield c


class TestUpload:
    @pytest.mark.asyncio
    async def test_upload_video(self, client):
        content = b"fake video content"
        resp = await client.post(
            "/api/upload",
            files={"file": ("test.mp4", content, "video/mp4")},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert "video_id" in data


class TestProcess:
    @pytest.mark.asyncio
    async def test_process_returns_job_id(self, client):
        # Upload first
        resp = await client.post(
            "/api/upload",
            files={"file": ("test.mp4", b"fake", "video/mp4")},
        )
        video_id = resp.json()["video_id"]

        resp = await client.post("/api/process", json={
            "video_id": video_id,
            "attack_method": "lsb",
            "preset": "preview",
        })
        assert resp.status_code == 200
        assert "job_id" in resp.json()


class TestStatus:
    @pytest.mark.asyncio
    async def test_unknown_job_404(self, client):
        resp = await client.get("/api/status/nonexistent")
        assert resp.status_code == 404
```

**Step 2: Run test to verify it fails**

Run: `uv run pytest tests/api/test_routes.py -v`
Expected: FAIL

**Step 3: Write the API routes and app factory**

```python
# backend/api/routes.py
"""REST API routes."""

import asyncio
from fastapi import APIRouter, UploadFile, File, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel

from backend.core.jobs import JobManager
from backend.storage.manager import StorageManager
from backend.pipeline.orchestrator import PipelineConfig, run_pipeline

router = APIRouter(prefix="/api")
job_manager = JobManager()
storage = StorageManager()


class ProcessRequest(BaseModel):
    video_id: str
    attack_method: str = "fgsm"
    preset: str = "preview"
    target_text: str = "an empty room"
    clip_labels: list[str] = [
        "a person talking", "an advertisement", "a product",
        "an empty room", "nature scenery", "abstract art",
    ]
    yolo_weight: float = 0.5
    clip_weight: float = 0.5
    lsb_mode: str = "cloak"
    lsb_intensity: float = 0.5


@router.post("/upload")
async def upload_video(file: UploadFile = File(...)):
    content = await file.read()
    video_id = storage.save_upload(file.filename or "video.mp4", content)
    return {"video_id": video_id}


@router.post("/process")
async def start_processing(req: ProcessRequest):
    try:
        video_path = storage.get_upload_path(req.video_id)
    except FileNotFoundError:
        raise HTTPException(404, "Video not found")

    output_dir = storage.get_output_dir(req.video_id)
    job_id = job_manager.create(req.video_id, req.model_dump())

    config = PipelineConfig(
        video_path=video_path,
        attack_method=req.attack_method,
        target_text=req.target_text,
        clip_labels=req.clip_labels,
        preset=req.preset,
        yolo_weight=req.yolo_weight,
        clip_weight=req.clip_weight,
        lsb_mode=req.lsb_mode,
        lsb_intensity=req.lsb_intensity,
        output_dir=output_dir,
    )

    # Run in background thread
    loop = asyncio.get_event_loop()
    loop.run_in_executor(None, _run_job, job_id, config)

    return {"job_id": job_id}


def _run_job(job_id: str, config: PipelineConfig):
    try:
        def on_progress(pct, stage):
            job_manager.update_progress(job_id, pct, stage)

        result = run_pipeline(config, on_progress=on_progress)
        job_manager.complete(job_id, result)
    except Exception as e:
        job_manager.fail(job_id, str(e))


@router.get("/status/{job_id}")
async def get_status(job_id: str):
    job = job_manager.get(job_id)
    if not job:
        raise HTTPException(404, "Job not found")
    return {
        "job_id": job["job_id"],
        "status": job["status"],
        "progress": job["progress"],
        "stage": job["stage"],
        "error": job["error"],
    }


@router.get("/results/{job_id}")
async def get_results(job_id: str):
    job = job_manager.get(job_id)
    if not job:
        raise HTTPException(404, "Job not found")
    if job["status"] != "completed":
        raise HTTPException(400, f"Job not completed: {job['status']}")
    return job["result"]


@router.get("/export/{job_id}")
async def export_video(job_id: str):
    job = job_manager.get(job_id)
    if not job or job["status"] != "completed":
        raise HTTPException(404, "Job not found or not completed")
    output_path = job["result"]["output_path"]
    return FileResponse(output_path, media_type="video/mp4", filename="processed.mp4")
```

```python
# backend/main.py
"""FastAPI application factory."""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from backend.api.routes import router as api_router


def create_app() -> FastAPI:
    app = FastAPI(title="Adversarial Video Tool", version="0.1.0")

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_methods=["*"],
        allow_headers=["*"],
    )

    app.include_router(api_router)

    # Serve frontend static files
    import os
    frontend_dir = os.path.join(os.path.dirname(__file__), "..", "frontend")
    if os.path.isdir(frontend_dir):
        app.mount("/", StaticFiles(directory=frontend_dir, html=True), name="frontend")

    return app


app = create_app()
```

**Step 4: Run test to verify it passes**

Run: `uv run pytest tests/api/test_routes.py -v`
Expected: 3 passed

**Step 5: Commit**

```bash
git add backend/api/routes.py backend/main.py tests/api/test_routes.py
git commit -m "feat: add REST API routes (upload, process, status, results, export)"
```

---

### Task 21: WebSocket Progress Handler

**Files:**
- Create: `backend/api/websocket.py`
- Modify: `backend/main.py`

**Step 1: Write the implementation** (WebSocket testing is complex — verify manually)

```python
# backend/api/websocket.py
"""WebSocket handler for real-time progress updates."""

import asyncio
import json
from fastapi import APIRouter, WebSocket, WebSocketDisconnect

from backend.api.routes import job_manager

ws_router = APIRouter()


@ws_router.websocket("/ws/progress/{job_id}")
async def progress_websocket(websocket: WebSocket, job_id: str):
    await websocket.accept()
    try:
        last_progress = -1
        while True:
            job = job_manager.get(job_id)
            if not job:
                await websocket.send_json({"error": "Job not found"})
                break

            if job["progress"] != last_progress:
                last_progress = job["progress"]
                await websocket.send_json({
                    "job_id": job_id,
                    "status": job["status"],
                    "progress": job["progress"],
                    "stage": job["stage"],
                })

            if job["status"] in ("completed", "failed"):
                break

            await asyncio.sleep(0.5)
    except WebSocketDisconnect:
        pass
```

**Step 2: Register WebSocket router in `backend/main.py`**

Add to `create_app()` before the static files mount:

```python
from backend.api.websocket import ws_router
app.include_router(ws_router)
```

**Step 3: Commit**

```bash
git add backend/api/websocket.py backend/main.py
git commit -m "feat: add WebSocket handler for real-time progress"
```

---

## Phase 6: Frontend

### Task 22: HTML Structure + CSS

**Files:**
- Create: `frontend/index.html`
- Create: `frontend/css/style.css`

**Step 1: Create `frontend/index.html`**

```html
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Adversarial Video Tool</title>
    <link rel="stylesheet" href="/css/style.css">
</head>
<body>
    <header>
        <h1>Adversarial Video Tool</h1>
        <p class="subtitle">Make videos invisible to AI detection</p>
    </header>

    <main>
        <!-- Upload Section -->
        <section id="upload-section" class="card">
            <h2>1. Upload Video</h2>
            <div id="drop-zone" class="drop-zone">
                <p>Drag & drop video here or <label for="file-input" class="link">browse</label></p>
                <input type="file" id="file-input" accept="video/*" hidden>
            </div>
            <div id="upload-info" class="hidden">
                <span id="file-name"></span>
                <span id="file-size"></span>
            </div>
        </section>

        <!-- Config Section -->
        <section id="config-section" class="card hidden">
            <h2>2. Configure Attack</h2>
            <div class="form-group">
                <label for="attack-method">Attack Method</label>
                <select id="attack-method">
                    <option value="combined">Combined (YOLO + CLIP)</option>
                    <option value="pgd">PGD (YOLO only)</option>
                    <option value="fgsm">FGSM (YOLO only, fast)</option>
                    <option value="lsb">LSB Steganography</option>
                    <option value="uap">Universal Adversarial Perturbation</option>
                </select>
            </div>
            <div class="form-group">
                <label for="preset">Quality Preset</label>
                <select id="preset">
                    <option value="preview">Preview (fastest)</option>
                    <option value="fast">Fast</option>
                    <option value="standard" selected>Standard</option>
                    <option value="high">High (slowest)</option>
                </select>
            </div>
            <div class="form-group">
                <label for="target-text">CLIP Target Description</label>
                <input type="text" id="target-text" value="an empty room with no people" placeholder="What should CLIP see?">
            </div>
            <div class="form-group" id="weight-group">
                <label>Objective Weights</label>
                <div class="slider-row">
                    <span>YOLO</span>
                    <input type="range" id="yolo-weight" min="0" max="100" value="50">
                    <span>CLIP</span>
                </div>
            </div>
            <button id="start-btn" class="btn-primary">Start Processing</button>
        </section>

        <!-- Progress Section -->
        <section id="progress-section" class="card hidden">
            <h2>3. Processing</h2>
            <div class="progress-bar">
                <div id="progress-fill" class="progress-fill" style="width: 0%"></div>
            </div>
            <p id="progress-text">Waiting...</p>
            <p id="stage-text" class="muted"></p>
        </section>

        <!-- Results Section -->
        <section id="results-section" class="card hidden">
            <h2>4. Results</h2>
            <div class="comparison">
                <div class="comparison-col">
                    <h3>Original</h3>
                    <div id="original-stats"></div>
                </div>
                <div class="comparison-col">
                    <h3>Processed</h3>
                    <div id="processed-stats"></div>
                </div>
            </div>
            <button id="download-btn" class="btn-primary">Download Processed Video</button>
        </section>
    </main>

    <script src="/js/app.js"></script>
</body>
</html>
```

**Step 2: Create `frontend/css/style.css`**

```css
/* frontend/css/style.css */
:root {
    --bg: #0a0a0f;
    --surface: #12121a;
    --border: #1e1e2e;
    --text: #e0e0e0;
    --muted: #666;
    --accent: #6366f1;
    --accent-hover: #818cf8;
    --success: #22c55e;
    --danger: #ef4444;
}

* { margin: 0; padding: 0; box-sizing: border-box; }

body {
    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
    background: var(--bg);
    color: var(--text);
    min-height: 100vh;
    padding: 2rem;
    max-width: 900px;
    margin: 0 auto;
}

header { text-align: center; margin-bottom: 2rem; }
h1 { font-size: 1.8rem; font-weight: 600; }
.subtitle { color: var(--muted); margin-top: 0.25rem; }
h2 { font-size: 1.2rem; margin-bottom: 1rem; color: var(--accent); }
h3 { font-size: 1rem; margin-bottom: 0.5rem; }

.card {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 12px;
    padding: 1.5rem;
    margin-bottom: 1rem;
}

.hidden { display: none !important; }

/* Drop Zone */
.drop-zone {
    border: 2px dashed var(--border);
    border-radius: 8px;
    padding: 3rem;
    text-align: center;
    cursor: pointer;
    transition: border-color 0.2s;
}
.drop-zone:hover, .drop-zone.dragover {
    border-color: var(--accent);
}
.drop-zone .link { color: var(--accent); cursor: pointer; text-decoration: underline; }

#upload-info { margin-top: 0.5rem; color: var(--muted); font-size: 0.9rem; }

/* Form */
.form-group { margin-bottom: 1rem; }
.form-group label { display: block; margin-bottom: 0.25rem; font-size: 0.9rem; color: var(--muted); }
select, input[type="text"] {
    width: 100%;
    padding: 0.6rem;
    background: var(--bg);
    border: 1px solid var(--border);
    border-radius: 6px;
    color: var(--text);
    font-size: 0.95rem;
}
.slider-row { display: flex; align-items: center; gap: 0.5rem; }
input[type="range"] { flex: 1; accent-color: var(--accent); }

.btn-primary {
    width: 100%;
    padding: 0.75rem;
    background: var(--accent);
    color: white;
    border: none;
    border-radius: 8px;
    font-size: 1rem;
    cursor: pointer;
    transition: background 0.2s;
}
.btn-primary:hover { background: var(--accent-hover); }
.btn-primary:disabled { opacity: 0.5; cursor: not-allowed; }

/* Progress */
.progress-bar {
    height: 8px;
    background: var(--bg);
    border-radius: 4px;
    overflow: hidden;
    margin-bottom: 0.75rem;
}
.progress-fill {
    height: 100%;
    background: var(--accent);
    border-radius: 4px;
    transition: width 0.3s ease;
}
.muted { color: var(--muted); font-size: 0.85rem; }

/* Comparison */
.comparison { display: grid; grid-template-columns: 1fr 1fr; gap: 1.5rem; margin-bottom: 1rem; }
.comparison-col { background: var(--bg); border-radius: 8px; padding: 1rem; }
.stat-row { display: flex; justify-content: space-between; padding: 0.3rem 0; font-size: 0.85rem; border-bottom: 1px solid var(--border); }
.stat-label { color: var(--muted); }
.stat-value { font-weight: 500; }
.stat-value.good { color: var(--success); }
.stat-value.bad { color: var(--danger); }
```

**Step 3: Commit**

```bash
git add frontend/index.html frontend/css/style.css
git commit -m "feat: add frontend HTML structure and dark theme CSS"
```

---

### Task 23: Frontend JavaScript (Upload + Config + Progress + Results)

**Files:**
- Create: `frontend/js/app.js`

**Step 1: Create `frontend/js/app.js`**

```javascript
// frontend/js/app.js
"use strict";

const API = "";  // Same origin

// --- State ---
let videoId = null;
let jobId = null;
let ws = null;

// --- DOM ---
const $ = (sel) => document.querySelector(sel);
const show = (el) => el.classList.remove("hidden");
const hide = (el) => el.classList.add("hidden");

// --- Upload ---
const dropZone = $("#drop-zone");
const fileInput = $("#file-input");

dropZone.addEventListener("click", () => fileInput.click());
dropZone.addEventListener("dragover", (e) => { e.preventDefault(); dropZone.classList.add("dragover"); });
dropZone.addEventListener("dragleave", () => dropZone.classList.remove("dragover"));
dropZone.addEventListener("drop", (e) => {
    e.preventDefault();
    dropZone.classList.remove("dragover");
    if (e.dataTransfer.files.length) handleFile(e.dataTransfer.files[0]);
});
fileInput.addEventListener("change", () => {
    if (fileInput.files.length) handleFile(fileInput.files[0]);
});

async function handleFile(file) {
    if (!file.type.startsWith("video/")) {
        alert("Please upload a video file.");
        return;
    }

    $("#file-name").textContent = file.name;
    $("#file-size").textContent = `(${(file.size / 1024 / 1024).toFixed(1)} MB)`;
    show($("#upload-info"));

    const form = new FormData();
    form.append("file", file);

    dropZone.innerHTML = "<p>Uploading...</p>";
    const resp = await fetch(`${API}/api/upload`, { method: "POST", body: form });
    const data = await resp.json();
    videoId = data.video_id;

    dropZone.innerHTML = `<p>Uploaded: ${file.name}</p>`;
    show($("#config-section"));
}

// --- Config & Start ---
$("#start-btn").addEventListener("click", async () => {
    if (!videoId) return;

    const body = {
        video_id: videoId,
        attack_method: $("#attack-method").value,
        preset: $("#preset").value,
        target_text: $("#target-text").value,
        yolo_weight: parseInt($("#yolo-weight").value) / 100,
        clip_weight: 1 - parseInt($("#yolo-weight").value) / 100,
    };

    $("#start-btn").disabled = true;
    show($("#progress-section"));

    const resp = await fetch(`${API}/api/process`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(body),
    });
    const data = await resp.json();
    jobId = data.job_id;

    connectWebSocket(jobId);
});

// --- WebSocket Progress ---
function connectWebSocket(jid) {
    const proto = location.protocol === "https:" ? "wss:" : "ws:";
    ws = new WebSocket(`${proto}//${location.host}/ws/progress/${jid}`);

    ws.onmessage = (event) => {
        const msg = JSON.parse(event.data);
        updateProgress(msg);
    };

    ws.onerror = () => {
        // Fallback to polling
        pollProgress(jid);
    };

    ws.onclose = () => {
        ws = null;
    };
}

function pollProgress(jid) {
    const interval = setInterval(async () => {
        const resp = await fetch(`${API}/api/status/${jid}`);
        const msg = await resp.json();
        updateProgress(msg);
        if (msg.status === "completed" || msg.status === "failed") {
            clearInterval(interval);
        }
    }, 1000);
}

const STAGE_LABELS = {
    pending: "Waiting...",
    extracting: "Extracting frames",
    classifying_original: "Analyzing original video",
    perturbing: "Applying adversarial attack",
    verifying: "Verifying results",
    reconstructing: "Rebuilding video",
    done: "Complete!",
};

function updateProgress(msg) {
    $("#progress-fill").style.width = `${msg.progress}%`;
    $("#progress-text").textContent = `${msg.progress}%`;
    $("#stage-text").textContent = STAGE_LABELS[msg.stage] || msg.stage;

    if (msg.status === "completed") {
        loadResults();
    } else if (msg.status === "failed") {
        $("#progress-text").textContent = "Failed";
        $("#stage-text").textContent = msg.error || "Unknown error";
    }
}

// --- Results ---
async function loadResults() {
    const resp = await fetch(`${API}/api/results/${jobId}`);
    const result = await resp.json();

    show($("#results-section"));
    renderStats(result);
}

function renderStats(result) {
    const origEl = $("#original-stats");
    const procEl = $("#processed-stats");

    if (result.original_classifications && result.original_classifications.length > 0) {
        const orig = result.original_classifications[0];
        const proc = result.perturbed_classifications[0];

        origEl.innerHTML = renderClassification(orig);
        procEl.innerHTML = renderClassification(proc);
    }
}

function renderClassification(cls) {
    let html = "";

    // YOLO detections
    const detCount = cls.yolo_detections ? cls.yolo_detections.length : 0;
    html += `<div class="stat-row"><span class="stat-label">YOLO detections</span><span class="stat-value">${detCount}</span></div>`;

    if (cls.yolo_detections) {
        for (const det of cls.yolo_detections.slice(0, 5)) {
            html += `<div class="stat-row"><span class="stat-label">&nbsp;&nbsp;${det.label}</span><span class="stat-value">${(det.confidence * 100).toFixed(1)}%</span></div>`;
        }
    }

    // CLIP scores
    if (cls.clip_scores) {
        html += `<div class="stat-row" style="margin-top:0.5rem"><span class="stat-label"><strong>CLIP</strong></span><span></span></div>`;
        const sorted = Object.entries(cls.clip_scores).sort((a, b) => b[1] - a[1]);
        for (const [label, score] of sorted) {
            html += `<div class="stat-row"><span class="stat-label">&nbsp;&nbsp;${label}</span><span class="stat-value">${(score * 100).toFixed(1)}%</span></div>`;
        }
    }

    return html;
}

// --- Download ---
$("#download-btn").addEventListener("click", () => {
    if (jobId) {
        window.location.href = `${API}/api/export/${jobId}`;
    }
});

// --- Show/hide weight slider based on method ---
$("#attack-method").addEventListener("change", () => {
    const method = $("#attack-method").value;
    const weightGroup = $("#weight-group");
    if (method === "combined") {
        show(weightGroup);
    } else {
        hide(weightGroup);
    }
});
```

**Step 2: Commit**

```bash
git add frontend/js/app.js
git commit -m "feat: add frontend JS (upload, config, WebSocket progress, results)"
```

---

## Phase 7: Integration & Polish

### Task 24: Run Script

**Files:**
- Create: `scripts/run.sh`

**Step 1: Create run script**

```bash
#!/bin/bash
# scripts/run.sh — Start the server
cd "$(dirname "$0")/.."
uv run uvicorn backend.main:app --host 0.0.0.0 --port 8000 --reload
```

```bash
chmod +x scripts/run.sh
git add scripts/run.sh
git commit -m "chore: add run script"
```

---

### Task 25: End-to-End Integration Test

**Files:**
- Create: `tests/test_integration.py`

**Step 1: Write integration test**

```python
# tests/test_integration.py
"""End-to-end integration test using the test fixture video."""
import os
import time
import pytest
from httpx import AsyncClient, ASGITransport
from backend.main import create_app

FIXTURE = "tests/fixtures/test_2s.mp4"


@pytest.fixture
def app():
    return create_app()


@pytest.fixture
async def client(app):
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as c:
        yield c


class TestE2ELSB:
    """Quick E2E with LSB (no GPU needed, fast)."""

    @pytest.mark.asyncio
    async def test_full_flow_lsb(self, client):
        # 1. Upload
        with open(FIXTURE, "rb") as f:
            resp = await client.post("/api/upload", files={"file": ("test.mp4", f, "video/mp4")})
        assert resp.status_code == 200
        video_id = resp.json()["video_id"]

        # 2. Process
        resp = await client.post("/api/process", json={
            "video_id": video_id,
            "attack_method": "lsb",
            "preset": "preview",
        })
        assert resp.status_code == 200
        job_id = resp.json()["job_id"]

        # 3. Poll until done
        for _ in range(60):
            resp = await client.get(f"/api/status/{job_id}")
            data = resp.json()
            if data["status"] in ("completed", "failed"):
                break
            await asyncio.sleep(0.5)

        assert data["status"] == "completed"

        # 4. Get results
        resp = await client.get(f"/api/results/{job_id}")
        assert resp.status_code == 200
        result = resp.json()
        assert "output_path" in result
        assert result["frames_processed"] == 20


# Add asyncio import
import asyncio
```

**Step 2: Run integration test**

Run: `uv run pytest tests/test_integration.py -v --timeout=120`
Expected: 1 passed

**Step 3: Commit**

```bash
git add tests/test_integration.py
git commit -m "test: add end-to-end integration test (LSB flow)"
```

---

### Task 26: Memory & Performance Guard

**Files:**
- Modify: `backend/pipeline/orchestrator.py`
- Modify: `backend/attacks/adversarial.py`

**Step 1: Add memory management**

In `backend/pipeline/orchestrator.py`, after each stage in `run_pipeline`, add:

```python
import torch
if torch.backends.mps.is_available():
    torch.mps.empty_cache()
```

In `backend/attacks/adversarial.py`, ensure every attack function calls `torch.mps.empty_cache()` (already there — verify).

In `backend/config.py`, add:

```python
# Memory safety: max frames to hold in RAM at once
MAX_FRAMES_IN_MEMORY = 3000  # ~3000 frames at 720p ≈ 7GB
```

In `backend/pipeline/orchestrator.py`, add a check after frame extraction:

```python
from backend.config import MAX_FRAMES_IN_MEMORY

if len(frames) > MAX_FRAMES_IN_MEMORY:
    # Downsample to stay within memory
    step = len(frames) // MAX_FRAMES_IN_MEMORY + 1
    frames = frames[::step]
```

**Step 2: Commit**

```bash
git add backend/pipeline/orchestrator.py backend/config.py
git commit -m "perf: add memory guards and MPS cache clearing between stages"
```

---

### Task 27: Final Verification

**Step 1: Run the full test suite**

Run: `uv run pytest tests/ -v --timeout=120`
Expected: All tests pass.

**Step 2: Start the server and manually verify**

Run: `./scripts/run.sh`

Open `http://localhost:8000` in browser. Upload a short video, select LSB attack (fastest), verify:
- Upload works
- Progress updates in real time
- Results show before/after classifications
- Download works

**Step 3: Final commit**

```bash
git add -A
git commit -m "chore: final polish and verification"
```

---

## Summary — Build Order & Dependencies

```
Phase 0: Task 1 (project init)
    │
Phase 1: Task 2 (device) → Task 3 (config) → Task 4-5 (video I/O) → Task 6 (batch)
    │
Phase 2: Task 7 (YOLO wrapper) → Task 8 (CLIP wrapper) → Task 9 (model cache)
    │
Phase 3: Task 10 (YOLO attack) → Task 11 (CLIP attack) → Task 12 (combined)
         Task 13 (LSB) — independent
         Task 14 (UAP) — needs Task 10
         Task 15 (temporal) — independent
    │
Phase 4: Task 16 (classifier) → Task 17 (orchestrator) — ties everything together
    │
Phase 5: Task 18 (jobs) → Task 19 (storage) → Task 20 (REST API) → Task 21 (WebSocket)
    │
Phase 6: Task 22 (HTML/CSS) → Task 23 (JS)
    │
Phase 7: Task 24 (run script) → Task 25 (E2E test) → Task 26 (memory) → Task 27 (verify)
```

**Total: 27 tasks, ~60 commits, 7 phases.**
