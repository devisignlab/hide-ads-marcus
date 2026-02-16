# backend/attacks/turbo.py
"""Turbo attack: UAP + FGSM + Typographic overlay.

Speed: ~50ms per frame (vs 4+ minutes with PGD).
Strategy:
  1. Compute Universal Adversarial Perturbation (UAP) against CLIP from sample frames (offline)
  2. For each frame: apply UAP + single FGSM step against YOLO + typographic overlay
"""

import logging
import os
import hashlib
import numpy as np
import torch
import torch.nn.functional as F
import cv2

from backend.core.device import get_device
from backend.core.batch import frames_to_tensor, tensor_to_frames
from backend.models.cache import get_yolo, get_clip
from backend.attacks.adversarial import _yolo_loss
from backend.config import YOLO_INPUT_SIZE

logger = logging.getLogger(__name__)

UAP_CACHE_DIR = "cache/uap"


def _uap_cache_key(target_text: str, repel_text: str, epsilon: float) -> str:
    """Generate a cache key based on attack parameters."""
    raw = f"{target_text}|{repel_text}|{epsilon:.6f}"
    return hashlib.sha256(raw.encode()).hexdigest()[:16]


def save_uap(uap: np.ndarray, target_text: str, repel_text: str, epsilon: float) -> str:
    """Save UAP to disk for reuse."""
    os.makedirs(UAP_CACHE_DIR, exist_ok=True)
    key = _uap_cache_key(target_text, repel_text, epsilon)
    path = os.path.join(UAP_CACHE_DIR, f"uap_{key}.npy")
    np.save(path, uap)
    logger.info(f"UAP saved to {path}")
    return path


def load_cached_uap(target_text: str, repel_text: str, epsilon: float) -> np.ndarray | None:
    """Load UAP from cache if it exists."""
    key = _uap_cache_key(target_text, repel_text, epsilon)
    path = os.path.join(UAP_CACHE_DIR, f"uap_{key}.npy")
    if os.path.exists(path):
        uap = np.load(path)
        logger.info(f"UAP loaded from cache: {path}")
        return uap
    return None


def get_or_compute_uap(
    sample_frames: list[np.ndarray],
    target_text: str,
    repel_text: str = "",
    epsilon: float = 32 / 255,
    steps: int = 100,
    alpha: float = 2 / 255,
) -> np.ndarray:
    """Load UAP from cache or compute it. Saves after computing."""
    cached = load_cached_uap(target_text, repel_text, epsilon)
    if cached is not None:
        return cached

    uap = compute_clip_uap(sample_frames, target_text, repel_text, epsilon, steps, alpha)
    save_uap(uap, target_text, repel_text, epsilon)
    return uap


def compute_clip_uap(
    sample_frames: list[np.ndarray],
    target_text: str,
    repel_text: str = "",
    epsilon: float = 32 / 255,
    steps: int = 100,
    alpha: float = 2 / 255,
) -> np.ndarray:
    """Compute a Universal Adversarial Perturbation optimized against CLIP.

    Uses MULTIPLE separate repel embeddings to prevent score redistribution
    between related NSFW concepts.

    Returns: numpy array of shape (H, W, 3) as float32 in range [-epsilon*255, +epsilon*255].
    """
    device = get_device()
    clip_model = get_clip()

    # Multiple attract targets for stronger pull
    attract_texts = [target_text]
    target_embs = clip_model.encode_text(attract_texts)  # [A, D]

    # CRITICAL: Split repel into INDIVIDUAL concepts so we push away from ALL of them
    repel_embs = None
    if repel_text:
        repel_parts = [t.strip() for t in repel_text.split(",") if t.strip()]
        if repel_parts:
            repel_embs = clip_model.encode_text(repel_parts)  # [R, D]
            logger.info(f"  Repel targets ({len(repel_parts)}): {repel_parts}")

    # Convert samples to tensor
    tensor = frames_to_tensor(sample_frames, device).contiguous()
    N, C, H, W = tensor.shape

    # Initialize UAP as zeros
    uap = torch.zeros(1, C, H, W, device=device, requires_grad=False)

    logger.info(f"Computing CLIP UAP: {steps} steps, {N} samples, eps={epsilon*255:.0f}/255")

    for step in range(steps):
        grad_accum = torch.zeros_like(uap)

        for si in range(N):
            sample = tensor[si:si+1]
            perturbed = (sample + uap).clamp(0, 1)

            perturbed_grad = perturbed.clone().detach().requires_grad_(True)
            img_emb = clip_model.encode_image_differentiable(perturbed_grad).float()
            img_emb_norm = F.normalize(img_emb, dim=-1)

            # Attract: maximize similarity to ALL safe targets
            target_norm = F.normalize(target_embs, dim=-1)
            attract_loss = -(img_emb_norm @ target_norm.T).mean()

            # Repel: minimize similarity to EACH NSFW concept individually
            repel_loss = torch.tensor(0.0, device=device)
            if repel_embs is not None:
                repel_norm = F.normalize(repel_embs, dim=-1)
                # Max over repel targets: push away from whichever NSFW concept is closest
                sims = (img_emb_norm @ repel_norm.T)  # [1, R]
                repel_loss = sims.max()  # Focus on the strongest NSFW match

            loss = attract_loss + 0.8 * repel_loss
            loss.backward()

            if device.type == "mps":
                torch.mps.synchronize()

            grad_accum += perturbed_grad.grad

        with torch.no_grad():
            grad_avg = grad_accum / N
            uap = uap - alpha * grad_avg.sign()
            uap = uap.clamp(-epsilon, epsilon)

        if step % 20 == 0:
            logger.info(f"  UAP step {step}/{steps}: attract={attract_loss.item():.4f} repel={repel_loss.item():.4f}")

    if device.type == "mps":
        torch.mps.empty_cache()

    uap_np = uap.squeeze(0).permute(1, 2, 0).cpu().numpy() * 255.0
    logger.info(f"  UAP computed. L_inf={np.abs(uap_np).max():.1f}/255")
    return uap_np


def apply_uap_to_frame(frame: np.ndarray, uap: np.ndarray) -> np.ndarray:
    """Apply pre-computed UAP to a single frame. <1ms."""
    # Resize UAP if frame size differs
    if frame.shape[:2] != uap.shape[:2]:
        uap_resized = cv2.resize(uap, (frame.shape[1], frame.shape[0]),
                                  interpolation=cv2.INTER_LINEAR)
    else:
        uap_resized = uap

    result = frame.astype(np.float32) + uap_resized
    return np.clip(result, 0, 255).astype(np.uint8)


def fgsm_yolo_single(frame: np.ndarray, epsilon: float = 16 / 255) -> np.ndarray:
    """Single FGSM step against YOLO for one frame. ~30-50ms."""
    device = get_device()
    yolo = get_yolo()
    model_inner = yolo.get_inner_model()
    model_inner.eval()

    tensor = frames_to_tensor([frame], device).contiguous()
    tensor.requires_grad_(True)

    loss = _yolo_loss(model_inner, tensor)
    loss.backward()

    if device.type == "mps":
        torch.mps.synchronize()

    with torch.no_grad():
        perturbation = -epsilon * tensor.grad.float().sign()
        adv = (tensor + perturbation).clamp(0, 1)

    result = tensor_to_frames(adv)[0]

    if device.type == "mps":
        torch.mps.empty_cache()

    return result


def add_typographic_overlay(
    frame: np.ndarray,
    text: str = "safe content",
    opacity: float = 0.05,
    font_scale: float = 0.35,
) -> np.ndarray:
    """Add semi-transparent text overlay to fool CLIP. <0.1ms.

    CLIP is known to prioritize text in images over visual content.
    Multiple overlapping texts at various positions maximize the effect.
    """
    overlay = frame.copy()
    h, w = frame.shape[:2]

    # Multiple safe phrases to reinforce the message
    phrases = [
        text,
        "product photo",
        "advertisement",
        "safe for work",
        "family content",
    ]

    # Grid of positions covering the entire frame
    positions = []
    for row in range(0, h - 10, h // 4):
        for col in range(0, w - 10, w // 3):
            positions.append((col + 5, row + 15))

    for i, (x, y) in enumerate(positions):
        phrase = phrases[i % len(phrases)]
        # Alternate white/black for visibility on any background
        color = (255, 255, 255) if i % 2 == 0 else (0, 0, 0)
        cv2.putText(overlay, phrase, (x, y), cv2.FONT_HERSHEY_SIMPLEX,
                    font_scale, color, 1, cv2.LINE_AA)

    # Blend — opacity 0.05 = nearly invisible to humans, but CLIP reads it
    result = cv2.addWeighted(frame, 1 - opacity, overlay, opacity, 0)
    return result


def fgsm_combined_single(
    frame: np.ndarray,
    target_text: str,
    repel_texts: list[str],
    epsilon: float = 48 / 255,
    steps: int = 5,
    yolo_weight: float = 0.2,
    clip_weight: float = 0.8,
) -> np.ndarray:
    """Mini-PGD with softmax classification attack.

    Uses cross-entropy loss to maximize the probability of target_text
    against all NSFW labels simultaneously. This prevents score redistribution.

    ~15s per frame on MPS with 5 steps.
    """
    device = get_device()
    yolo = get_yolo()
    clip_model = get_clip()
    model_inner = yolo.get_inner_model()
    model_inner.eval()

    # Build full label set: target at index 0, NSFW labels after
    all_labels = [target_text] + repel_texts
    all_embs = clip_model.encode_text(all_labels)

    original = frames_to_tensor([frame], device).contiguous()
    adv = original.clone().detach()
    alpha = epsilon / max(steps, 1)

    for step in range(steps):
        adv_input = adv.clone().detach().requires_grad_(True)

        loss = torch.tensor(0.0, device=device)

        # YOLO suppression
        if yolo_weight > 0:
            loss_y = _yolo_loss(model_inner, adv_input)
            loss = loss + yolo_weight * loss_y

        # CLIP softmax classification attack
        if clip_weight > 0:
            img_emb = clip_model.encode_image_differentiable(adv_input).float()
            img_norm = F.normalize(img_emb, dim=-1)
            all_norm = F.normalize(all_embs, dim=-1)
            logits = (img_norm @ all_norm.T) * 100.0
            target_idx = torch.zeros(1, dtype=torch.long, device=device)
            clip_loss = F.cross_entropy(logits, target_idx)
            loss = loss + clip_weight * clip_loss * 500

        loss.backward()

        if device.type == "mps":
            torch.mps.synchronize()

        with torch.no_grad():
            adv = adv - alpha * adv_input.grad.float().sign()
            delta = (adv - original).clamp(-epsilon, epsilon)
            adv = (original + delta).clamp(0, 1)

    result = tensor_to_frames(adv)[0]

    if device.type == "mps":
        torch.mps.empty_cache()

    return result


def turbo_attack_frame(
    frame: np.ndarray,
    uap: np.ndarray | None = None,
    yolo_epsilon: float = 16 / 255,
    typographic_text: str = "safe content",
    use_typographic: bool = False,
    use_yolo_fgsm: bool = True,
    target_text: str = "",
    repel_texts: list[str] | None = None,
    epsilon: float = 32 / 255,
) -> np.ndarray:
    """Full turbo attack on a single frame.

    If target_text + repel_texts provided: uses combined FGSM (~100ms).
    Otherwise falls back to UAP + FGSM (~50ms).
    """
    # Preferred: Combined FGSM (no precomputation needed)
    if target_text and repel_texts:
        return fgsm_combined_single(
            frame, target_text, repel_texts,
            epsilon=epsilon,
        )

    # Fallback: UAP + separate FGSM
    result = frame
    if uap is not None:
        result = apply_uap_to_frame(result, uap)
    if use_yolo_fgsm:
        result = fgsm_yolo_single(result, yolo_epsilon)
    if use_typographic:
        result = add_typographic_overlay(result, typographic_text)
    return result
