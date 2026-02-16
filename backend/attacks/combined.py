# backend/attacks/combined.py
"""Combined multi-objective adversarial attack (YOLO suppression + CLIP redirection + CLIP repulsion)."""

import numpy as np
import torch
import torch.nn.functional as F

from backend.core.device import get_device
from backend.core.batch import frames_to_tensor, tensor_to_frames
from backend.models.cache import get_yolo, get_clip
from backend.attacks.adversarial import _yolo_loss, _clip_loss


def _clip_repel_loss(clip_wrapper, images: torch.Tensor, repel_embedding: torch.Tensor) -> torch.Tensor:
    """Compute cosine similarity to repel text (we want to MAXIMIZE this loss = push away)."""
    img_emb = clip_wrapper.encode_image_differentiable(images).float()
    img_emb = F.normalize(img_emb, dim=-1)
    repel_emb = F.normalize(repel_embedding, dim=-1)
    return (img_emb @ repel_emb.T).mean()  # positive = close to NSFW, we minimize to push away


def attack_combined_pgd(
    frames: list[np.ndarray],
    target_text: str,
    yolo_weight: float = 0.5,
    clip_weight: float = 0.5,
    epsilon: float = 4 / 255,
    steps: int = 20,
    alpha: float = 1 / 255,
    repel_text: str | None = None,
    repel_weight: float = 0.3,
) -> list[np.ndarray]:
    """PGD with combined YOLO+CLIP objective + optional CLIP repulsion.

    Three objectives:
    1. YOLO suppression: minimize objectness scores
    2. CLIP attract: push embeddings toward target_text
    3. CLIP repel: push embeddings away from repel_text (NSFW labels)

    Processes one frame at a time to avoid MPS memory issues.
    """
    device = get_device()
    yolo = get_yolo()
    clip_model = get_clip()
    model_inner = yolo.get_inner_model()
    model_inner.eval()

    target_emb = clip_model.encode_text([target_text])

    repel_emb = None
    if repel_text:
        repel_emb = clip_model.encode_text([repel_text])
        # Redistribute weights: clip_weight splits into attract + repel
        attract_weight = clip_weight * (1 - repel_weight)
        actual_repel_weight = clip_weight * repel_weight
    else:
        attract_weight = clip_weight
        actual_repel_weight = 0.0

    # Process one frame at a time for MPS stability
    all_perturbed = []
    for fi, frame in enumerate(frames):
        original = frames_to_tensor([frame], device).contiguous()
        adv = original.clone().detach()

        for step in range(steps):
            grad_total = torch.zeros_like(original)

            # 1) YOLO suppression
            if yolo_weight > 0:
                adv_y = adv.clone().detach().requires_grad_(True)
                loss_y = _yolo_loss(model_inner, adv_y)
                loss_y.backward()
                g_y = adv_y.grad.float()
                norm_y = g_y.norm(p=2) + 1e-12
                grad_total += yolo_weight * (g_y / norm_y)

                if device.type == "mps":
                    torch.mps.synchronize()

            # 2) CLIP attract to safe content
            if attract_weight > 0:
                adv_c = adv.clone().detach().requires_grad_(True)
                loss_c = _clip_loss(clip_model, adv_c, target_emb)
                loss_c.backward()
                g_c = adv_c.grad.float()
                norm_c = g_c.norm(p=2) + 1e-12
                grad_total += attract_weight * (g_c / norm_c)

                if device.type == "mps":
                    torch.mps.synchronize()

            # 3) CLIP repel from NSFW content
            if repel_emb is not None and actual_repel_weight > 0:
                adv_r = adv.clone().detach().requires_grad_(True)
                loss_r = _clip_repel_loss(clip_model, adv_r, repel_emb)
                loss_r.backward()
                g_r = adv_r.grad.float()
                norm_r = g_r.norm(p=2) + 1e-12
                grad_total += actual_repel_weight * (g_r / norm_r)

                if device.type == "mps":
                    torch.mps.synchronize()

            # PGD step
            with torch.no_grad():
                adv = adv - alpha * grad_total.sign()
                delta = (adv - original).clamp(-epsilon, epsilon)
                adv = (original + delta).clamp(0, 1)

        all_perturbed.extend(tensor_to_frames(adv))

        # Free memory between frames
        del original, adv
        if device.type == "mps":
            torch.mps.empty_cache()
            torch.mps.synchronize()

    return all_perturbed
