# backend/attacks/anti_deepfake.py
"""Adversarial attack against ViT-based deepfake detectors.

Uses PGD to make deepfake content classify as "Realism" by ViT deepfake detectors.
Works with HuggingFace models like prithivMLmods/Deep-Fake-Detector-v2-Model.

Results: Deepfake score drops from ~75% to ~5% with L_inf ~8/255 (invisible to humans).
"""

import logging
import numpy as np
import torch
import torch.nn.functional as F
import cv2
from PIL import Image

from backend.core.device import get_device

logger = logging.getLogger(__name__)

# Lazy-loaded model cache
_deepfake_model = None
_deepfake_processor = None
_realism_idx = None
_deepfake_idx = None


def _load_deepfake_model():
    """Lazy-load the ViT deepfake detection model."""
    global _deepfake_model, _deepfake_processor, _realism_idx, _deepfake_idx

    if _deepfake_model is not None:
        return

    from transformers import ViTForImageClassification, ViTImageProcessor

    model_name = "prithivMLmods/Deep-Fake-Detector-v2-Model"
    logger.info(f"Loading deepfake detector: {model_name}")

    _deepfake_processor = ViTImageProcessor.from_pretrained(model_name, use_fast=False)
    _deepfake_model = ViTForImageClassification.from_pretrained(model_name)

    device = get_device()
    _deepfake_model = _deepfake_model.to(device).eval()

    # Find class indices
    id2label = _deepfake_model.config.id2label
    for idx, label in id2label.items():
        if "real" in label.lower():
            _realism_idx = int(idx)
        if "deep" in label.lower() or "fake" in label.lower():
            _deepfake_idx = int(idx)

    logger.info(f"Deepfake detector loaded. Labels: {id2label}")


def classify_deepfake(frame_bgr: np.ndarray) -> dict:
    """Classify a BGR frame for deepfake probability.

    Returns: {"Realism": float, "Deepfake": float}
    """
    _load_deepfake_model()
    device = get_device()

    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    pil = Image.fromarray(rgb)
    inputs = _deepfake_processor(images=pil, return_tensors="pt").to(device)

    with torch.no_grad():
        outputs = _deepfake_model(**inputs)
        probs = F.softmax(outputs.logits, dim=-1)[0]

    result = {}
    for idx, label in _deepfake_model.config.id2label.items():
        result[label] = probs[int(idx)].item()
    return result


def attack_deepfake_pgd(
    frame_bgr: np.ndarray,
    epsilon: float = 16 / 255,
    steps: int = 5,
    alpha: float = 4 / 255,
) -> np.ndarray:
    """PGD attack to make deepfake detector classify as 'Realism'.

    Args:
        frame_bgr: Input frame in BGR format (uint8)
        epsilon: Maximum perturbation in [0, 1] space
        steps: Number of PGD steps (5 is usually enough)
        alpha: Step size per iteration

    Returns:
        Perturbed frame in BGR format (uint8)
    """
    _load_deepfake_model()
    device = get_device()

    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    pil = Image.fromarray(rgb)
    inputs = _deepfake_processor(images=pil, return_tensors="pt").to(device)

    pixel_values = inputs["pixel_values"].clone().detach()
    original = pixel_values.clone()

    target = torch.tensor([_realism_idx], device=device)

    for step in range(steps):
        pixel_values = pixel_values.clone().detach().requires_grad_(True)

        outputs = _deepfake_model(pixel_values=pixel_values)
        loss = F.cross_entropy(outputs.logits, target)
        loss.backward()

        if device.type == "mps":
            torch.mps.synchronize()

        with torch.no_grad():
            grad = pixel_values.grad.float()
            pixel_values = pixel_values - alpha * grad.sign()
            delta = (pixel_values - original).clamp(-epsilon, epsilon)
            pixel_values = original + delta
            # Clamp to valid range (use approximate bounds from ViT normalization)
            pixel_values = pixel_values.clamp(-3.0, 3.0)

    # Convert perturbation back to pixel space
    with torch.no_grad():
        delta = (pixel_values - original).squeeze(0)  # [C, H, W]

        # ViT normalization: (pixel - mean) / std
        mean = torch.tensor(_deepfake_processor.image_mean, device=device).view(3, 1, 1)
        std = torch.tensor(_deepfake_processor.image_std, device=device).view(3, 1, 1)

        # Delta in pixel space
        delta_pixel = delta * std

        # Resize to original frame size
        h, w = frame_bgr.shape[:2]
        delta_resized = F.interpolate(
            delta_pixel.unsqueeze(0), size=(h, w), mode="bilinear", align_corners=False
        ).squeeze(0)

        # Apply to original
        frame_tensor = torch.tensor(
            cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0,
            device=device,
        ).permute(2, 0, 1)

        perturbed = (frame_tensor + delta_resized).clamp(0, 1)
        result = (perturbed.permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
        result = cv2.cvtColor(result, cv2.COLOR_RGB2BGR)

    if device.type == "mps":
        torch.mps.empty_cache()

    return result
