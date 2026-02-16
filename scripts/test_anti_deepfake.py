#!/usr/bin/env python3
"""Test adversarial attack against HuggingFace deepfake detector.

Goal: Lower "Deepfake" confidence from ~75% to below 50%.
"""
import sys, os, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import logging
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from transformers import ViTForImageClassification, ViTImageProcessor

from backend.core.video import read_frames_at_indices, get_video_info
from backend.core.device import get_device

# Load deepfake detector
print("Loading deepfake detector...")
model_name = "prithivMLmods/Deep-Fake-Detector-v2-Model"
processor = ViTImageProcessor.from_pretrained(model_name, use_fast=False)
model = ViTForImageClassification.from_pretrained(model_name)
device = get_device()
model = model.to(device).eval()

# Get label mapping
id2label = model.config.id2label
print(f"Labels: {id2label}")

# Find the "Realism" class index (we want to maximize this)
realism_idx = None
deepfake_idx = None
for idx, label in id2label.items():
    if "real" in label.lower():
        realism_idx = int(idx)
    if "deep" in label.lower() or "fake" in label.lower():
        deepfake_idx = int(idx)
print(f"Realism idx={realism_idx}, Deepfake idx={deepfake_idx}")


def classify_frame(frame_bgr):
    """Classify a single BGR frame."""
    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    pil = Image.fromarray(rgb)
    inputs = processor(images=pil, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = model(**inputs)
        probs = F.softmax(outputs.logits, dim=-1)[0]
    result = {}
    for idx, label in id2label.items():
        result[label] = probs[int(idx)].item()
    return result


def adversarial_attack_deepfake(
    frame_bgr: np.ndarray,
    epsilon: float = 16/255,
    steps: int = 10,
    alpha: float = 2/255,
) -> np.ndarray:
    """PGD attack to make deepfake detector classify as 'Realism'.

    Uses the ViT deepfake detector's own gradients.
    """
    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    pil = Image.fromarray(rgb)
    inputs = processor(images=pil, return_tensors="pt").to(device)

    # Get the preprocessed pixel values
    pixel_values = inputs["pixel_values"].clone().detach()
    original = pixel_values.clone()

    target = torch.tensor([realism_idx], device=device)

    for step in range(steps):
        pixel_values = pixel_values.clone().detach().requires_grad_(True)

        outputs = model(pixel_values=pixel_values)
        loss = F.cross_entropy(outputs.logits, target)
        loss.backward()

        if device.type == "mps":
            torch.mps.synchronize()

        with torch.no_grad():
            # Minimize loss = maximize "Realism" probability
            grad = pixel_values.grad.float()
            pixel_values = pixel_values - alpha * grad.sign()

            # Project back to epsilon ball
            delta = (pixel_values - original).clamp(-epsilon, epsilon)
            pixel_values = (original + delta).clamp(
                pixel_values.min().item(), pixel_values.max().item()
            )

        if step % 5 == 0:
            with torch.no_grad():
                probs = F.softmax(model(pixel_values=pixel_values.detach()).logits, dim=-1)[0]
                fake_p = probs[deepfake_idx].item()
                real_p = probs[realism_idx].item()
                print(f"    Step {step}: Deepfake={fake_p*100:.1f}%, Realism={real_p*100:.1f}%, loss={loss.item():.4f}")

    # Convert perturbed pixel values back to image
    # We need to reverse the ViT preprocessing
    with torch.no_grad():
        # The processor normalizes: (pixel - mean) / std
        # We need to get the perturbation in pixel space
        # Approach: compute delta in preprocessed space, scale to pixel space
        delta = (pixel_values - original).squeeze(0)  # [C, H, W]

        # Get normalization params
        mean = torch.tensor(processor.image_mean, device=device).view(3, 1, 1)
        std = torch.tensor(processor.image_std, device=device).view(3, 1, 1)

        # Delta in pixel space (approximately)
        delta_pixel = delta * std  # [C, H, W]

        # Resize delta back to original frame size
        h, w = frame_bgr.shape[:2]
        delta_resized = F.interpolate(
            delta_pixel.unsqueeze(0), size=(h, w), mode="bilinear", align_corners=False
        ).squeeze(0)  # [C, H, W]

        # Apply to original frame
        frame_tensor = torch.tensor(
            cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0,
            device=device
        ).permute(2, 0, 1)  # [C, H, W]

        perturbed = (frame_tensor + delta_resized).clamp(0, 1)
        result = (perturbed.permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
        result = cv2.cvtColor(result, cv2.COLOR_RGB2BGR)

    return result


# === Main ===
video = "uploads/Top-06.mp4"
info = get_video_info(video)
indices = [0, 500, 1000, 2000]
indices = [i for i in indices if i < info["frame_count"]]

frames = read_frames_at_indices(video, indices)
print(f"\nLoaded {len(frames)} frames from {video}")

# Classify originals
print("\n=== ORIGINAL CLASSIFICATION ===")
for i, frame in enumerate(frames):
    cls = classify_frame(frame)
    print(f"  Frame {indices[i]}: {', '.join(f'{k}={v*100:.1f}%' for k,v in sorted(cls.items(), key=lambda x: -x[1]))}")

# Attack each frame
print("\n=== ADVERSARIAL ATTACK (PGD, eps=16/255, 20 steps) ===")
perturbed_frames = []
for i, frame in enumerate(frames):
    print(f"\n  Frame {indices[i]}:")
    t0 = time.time()
    result = adversarial_attack_deepfake(frame, epsilon=16/255, steps=20, alpha=2/255)
    elapsed = time.time() - t0
    perturbed_frames.append(result)
    print(f"    Time: {elapsed:.1f}s")

    # Measure pixel difference
    diff = np.abs(frame.astype(float) - result.astype(float))
    print(f"    Pixel diff: L_inf={diff.max():.1f}/255, mean={diff.mean():.2f}/255")

# Classify perturbed
print("\n=== PERTURBED CLASSIFICATION ===")
for i, frame in enumerate(perturbed_frames):
    cls = classify_frame(frame)
    print(f"  Frame {indices[i]}: {', '.join(f'{k}={v*100:.1f}%' for k,v in sorted(cls.items(), key=lambda x: -x[1]))}")

# Delta summary
print("\n=== DELTA SUMMARY ===")
for i in range(len(frames)):
    orig_cls = classify_frame(frames[i])
    pert_cls = classify_frame(perturbed_frames[i])
    for label in id2label.values():
        ov = orig_cls[label] * 100
        pv = pert_cls[label] * 100
        if abs(pv - ov) > 0.1:
            print(f"  Frame {indices[i]} {label:15s}: {ov:.1f}% -> {pv:.1f}% ({pv-ov:+.1f})")

# Save example perturbed frame
os.makedirs("outputs/analysis", exist_ok=True)
cv2.imwrite("outputs/analysis/original_frame0.png", frames[0])
cv2.imwrite("outputs/analysis/perturbed_frame0.png", perturbed_frames[0])
print("\nSaved example frames to outputs/analysis/")
