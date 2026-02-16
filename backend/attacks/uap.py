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
        # Apply same UAP to all frames; .contiguous() needed for MPS view ops
        adv = (originals + uap_param).clamp(0, 1).contiguous()

        loss = torch.tensor(0.0, device=device)
        if target in ("yolo", "combined") and model_inner is not None:
            loss = loss + _yolo_loss(model_inner, adv)
        if target in ("clip", "combined") and clip_model is not None and target_emb is not None:
            loss = loss + _clip_loss(clip_model, adv, target_emb)

        loss.backward()

        with torch.no_grad():
            # .float() before .sign() handles float16 gradients from CLIP on MPS
            uap = uap - alpha * uap_param.grad.float().sign()
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
