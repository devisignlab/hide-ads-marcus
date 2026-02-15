# backend/attacks/adversarial.py
"""Gradient-based adversarial attacks on YOLO and CLIP."""

import numpy as np
import torch
import torch.nn.functional as F

from backend.core.device import get_device
from backend.core.batch import frames_to_tensor, tensor_to_frames
from backend.models.cache import get_yolo, get_clip
from backend.config import YOLO_INPUT_SIZE


def _clamp_frames_epsilon(
    originals: list[np.ndarray], perturbed: list[np.ndarray], epsilon: float,
) -> list[np.ndarray]:
    """Clamp perturbed frames so that the uint8 diff respects epsilon."""
    eps_int = int(np.ceil(epsilon * 255))
    result = []
    for orig, pert in zip(originals, perturbed):
        lo = np.clip(orig.astype(np.int16) - eps_int, 0, 255).astype(np.uint8)
        hi = np.clip(orig.astype(np.int16) + eps_int, 0, 255).astype(np.uint8)
        result.append(np.clip(pert, lo, hi))
    return result


def _yolo_loss(model_inner: torch.nn.Module, images: torch.Tensor) -> torch.Tensor:
    """Compute YOLO objectness loss for suppression attack."""
    device = images.device
    resized = F.interpolate(images, size=(YOLO_INPUT_SIZE, YOLO_INPUT_SIZE),
                            mode="bilinear", align_corners=False)
    with torch.enable_grad():
        preds = model_inner(resized)
        if isinstance(preds, (list, tuple)):
            pred = preds[0]
        else:
            pred = preds
        if pred.dim() == 3:
            class_scores = pred[:, 4:, :]
            loss = class_scores.sigmoid().sum()
        else:
            loss = pred.sigmoid().sum()
    return loss


def attack_yolo_fgsm(frames: list[np.ndarray], epsilon: float = 4 / 255) -> list[np.ndarray]:
    """FGSM attack on YOLO — single gradient step to suppress detections."""
    device = get_device()
    yolo = get_yolo()
    model_inner = yolo.get_inner_model()
    model_inner.eval()

    tensor = frames_to_tensor(frames, device).contiguous()
    tensor.requires_grad_(True)

    loss = _yolo_loss(model_inner, tensor)
    loss.backward()

    perturbation = -epsilon * tensor.grad.float().sign()
    adv_tensor = (tensor + perturbation).clamp(0, 1)

    if device.type == "mps":
        torch.mps.empty_cache()

    return _clamp_frames_epsilon(frames, tensor_to_frames(adv_tensor), epsilon)


def attack_yolo_pgd(
    frames: list[np.ndarray], epsilon: float = 4 / 255,
    steps: int = 20, alpha: float = 1 / 255,
) -> list[np.ndarray]:
    """PGD attack on YOLO — iterative gradient descent to suppress detections."""
    device = get_device()
    yolo = get_yolo()
    model_inner = yolo.get_inner_model()
    model_inner.eval()

    original = frames_to_tensor(frames, device).contiguous()
    adv = original.clone().detach()

    for _ in range(steps):
        adv.requires_grad_(True)
        loss = _yolo_loss(model_inner, adv)
        loss.backward()
        with torch.no_grad():
            adv = adv - alpha * adv.grad.float().sign()
            delta = (adv - original).clamp(-epsilon, epsilon)
            adv = (original + delta).clamp(0, 1)

    if device.type == "mps":
        torch.mps.empty_cache()

    return _clamp_frames_epsilon(frames, tensor_to_frames(adv), epsilon)


def _clip_loss(clip_wrapper, images: torch.Tensor, target_embedding: torch.Tensor) -> torch.Tensor:
    """Compute negative cosine similarity to target text."""
    img_emb = clip_wrapper.encode_image_differentiable(images).float()
    img_emb = F.normalize(img_emb, dim=-1)
    target_emb = F.normalize(target_embedding, dim=-1)
    return -(img_emb @ target_emb.T).mean()


def attack_clip_fgsm(
    frames: list[np.ndarray], target_text: str, epsilon: float = 4 / 255,
) -> list[np.ndarray]:
    """FGSM attack on CLIP — redirect semantic classification toward target text."""
    device = get_device()
    clip_model = get_clip()

    target_emb = clip_model.encode_text([target_text])
    tensor = frames_to_tensor(frames, device).contiguous()
    tensor.requires_grad_(True)

    loss = _clip_loss(clip_model, tensor, target_emb)
    loss.backward()

    perturbation = -epsilon * tensor.grad.float().sign()
    adv_tensor = (tensor + perturbation).clamp(0, 1)

    if device.type == "mps":
        torch.mps.empty_cache()

    return _clamp_frames_epsilon(frames, tensor_to_frames(adv_tensor), epsilon)


def attack_clip_pgd(
    frames: list[np.ndarray], target_text: str,
    epsilon: float = 4 / 255, steps: int = 20, alpha: float = 1 / 255,
) -> list[np.ndarray]:
    """PGD attack on CLIP — iterative redirect toward target text."""
    device = get_device()
    clip_model = get_clip()

    target_emb = clip_model.encode_text([target_text])
    original = frames_to_tensor(frames, device).contiguous()
    adv = original.clone().detach()

    for _ in range(steps):
        adv.requires_grad_(True)
        loss = _clip_loss(clip_model, adv, target_emb)
        loss.backward()
        with torch.no_grad():
            adv = adv - alpha * adv.grad.float().sign()
            delta = (adv - original).clamp(-epsilon, epsilon)
            adv = (original + delta).clamp(0, 1)

    if device.type == "mps":
        torch.mps.empty_cache()

    return _clamp_frames_epsilon(frames, tensor_to_frames(adv), epsilon)
