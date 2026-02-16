#!/usr/bin/env python3
"""Test CLIP-based NSFW attack on sensual content (Top-13).

Goal: Reduce sensual/NSFW CLIP scores while maintaining visual quality.
"""
import sys, os, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import logging
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

import cv2
import numpy as np
import torch

from backend.core.video import read_frames_at_indices, get_video_info
from backend.core.device import get_device

# CLIP setup
import clip
device = get_device()
print(f"Device: {device}")
print("Loading CLIP ViT-B/32...")
clip_model, clip_preprocess = clip.load("ViT-B/32", device=device)

# NSFW labels to test
NSFW_LABELS = [
    "a safe family friendly product advertisement, professional clean content",
    "explicit adult content, nudity, sexually explicit material",
    "sensual content, seductive poses, provocative imagery",
    "bikini model, swimwear, lingerie",
    "romantic intimate scene, kissing, physical intimacy",
    "professional news broadcast, educational content",
    "cartoon animation, kids content",
    "deepfake manipulated face, AI generated content",
]

REPEL_TEXTS = [
    "explicit adult content, nudity, sexually explicit material, naked body, pornography",
    "sensual content, seductive poses, provocative imagery, erotic",
    "bikini model, swimwear, lingerie, revealing clothing",
    "romantic intimate scene, kissing, physical intimacy",
]

TARGET_TEXT = "a safe family friendly product advertisement, professional clean content"


def clip_classify(frame_bgr, labels=NSFW_LABELS):
    """Classify a frame with CLIP against multiple text labels."""
    from PIL import Image
    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    pil = Image.fromarray(rgb)
    image = clip_preprocess(pil).unsqueeze(0).to(device)
    text = clip.tokenize(labels).to(device)

    with torch.no_grad():
        image_features = clip_model.encode_image(image)
        text_features = clip_model.encode_text(text)
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        similarity = (100.0 * image_features @ text_features.T).softmax(dim=-1)

    return {label: similarity[0][i].item() for i, label in enumerate(labels)}


def clip_attack_pgd(
    frame_bgr: np.ndarray,
    target_text: str = TARGET_TEXT,
    repel_texts: list = REPEL_TEXTS,
    epsilon: float = 32 / 255,
    steps: int = 10,
) -> np.ndarray:
    """PGD attack to make CLIP classify as safe content."""
    from PIL import Image
    alpha = epsilon / steps

    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    pil = Image.fromarray(rgb)
    image_input = clip_preprocess(pil).unsqueeze(0).to(device)
    original = image_input.clone()

    # Encode all text
    all_labels = [target_text] + repel_texts
    text_tokens = clip.tokenize(all_labels).to(device)
    with torch.no_grad():
        text_features = clip_model.encode_text(text_tokens).float()
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)

    target_idx = torch.tensor([0], device=device)

    for step in range(steps):
        image_input = image_input.clone().detach().requires_grad_(True)

        image_features = clip_model.encode_image(image_input).float()
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)

        logits = 100.0 * image_features @ text_features.T
        loss = torch.nn.functional.cross_entropy(logits, target_idx)
        loss.backward()

        if device.type == "mps":
            torch.mps.synchronize()

        with torch.no_grad():
            grad = image_input.grad.float()
            image_input = image_input - alpha * grad.sign()
            delta = (image_input - original).clamp(-epsilon, epsilon)
            image_input = original + delta

    # Convert back to pixel space
    with torch.no_grad():
        delta = (image_input - original).squeeze(0)  # [C, H, W]

        # CLIP normalization: mean=[0.48145466, 0.4578275, 0.40821073], std=[0.26862954, 0.26130258, 0.27577711]
        mean = torch.tensor([0.48145466, 0.4578275, 0.40821073], device=device).view(3, 1, 1)
        std = torch.tensor([0.26862954, 0.26130258, 0.27577711], device=device).view(3, 1, 1)

        delta_pixel = delta * std

        h, w = frame_bgr.shape[:2]
        delta_resized = torch.nn.functional.interpolate(
            delta_pixel.unsqueeze(0), size=(h, w), mode="bilinear", align_corners=False
        ).squeeze(0)

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


# === Main ===
video = "uploads/Top-13.mp4"
info = get_video_info(video)
total = info["frame_count"]
indices = [0, total // 4, total // 2, 3 * total // 4]
indices = [i for i in indices if i < total]

frames = read_frames_at_indices(video, indices)
print(f"\nLoaded {len(frames)} frames from {video} ({total} total)")

# Classify originals
print("\n=== ORIGINAL CLIP CLASSIFICATION ===")
for i, frame in enumerate(frames):
    cls = clip_classify(frame)
    print(f"\n  Frame {indices[i]}:")
    for label, score in sorted(cls.items(), key=lambda x: -x[1]):
        bar = "█" * int(score * 50)
        print(f"    {score*100:5.1f}% {bar} {label[:60]}")

# Attack each frame
print("\n=== CLIP PGD ATTACK (eps=32/255, 10 steps) ===")
perturbed_frames = []
for i, frame in enumerate(frames):
    print(f"\n  Frame {indices[i]}:")
    t0 = time.time()
    result = clip_attack_pgd(frame, epsilon=32/255, steps=10)
    elapsed = time.time() - t0
    perturbed_frames.append(result)
    print(f"    Time: {elapsed:.1f}s")

    diff = np.abs(frame.astype(float) - result.astype(float))
    print(f"    Pixel diff: L_inf={diff.max():.1f}/255, mean={diff.mean():.2f}/255")

# Classify perturbed
print("\n=== PERTURBED CLIP CLASSIFICATION ===")
for i, frame in enumerate(perturbed_frames):
    cls = clip_classify(frame)
    print(f"\n  Frame {indices[i]}:")
    for label, score in sorted(cls.items(), key=lambda x: -x[1]):
        bar = "█" * int(score * 50)
        print(f"    {score*100:5.1f}% {bar} {label[:60]}")

# Delta summary
print("\n=== DELTA SUMMARY ===")
for i in range(len(frames)):
    orig = clip_classify(frames[i])
    pert = clip_classify(perturbed_frames[i])
    print(f"\n  Frame {indices[i]}:")
    for label in NSFW_LABELS:
        ov = orig[label] * 100
        pv = pert[label] * 100
        delta = pv - ov
        if abs(delta) > 0.5:
            arrow = "↑" if delta > 0 else "↓"
            print(f"    {arrow} {label[:55]:55s}: {ov:5.1f}% -> {pv:5.1f}% ({delta:+.1f})")

# Also test anti-deepfake on same frames
print("\n\n=== BONUS: ANTI-DEEPFAKE on Top-13 ===")
from backend.attacks.anti_deepfake import classify_deepfake, attack_deepfake_pgd

for i, frame in enumerate(frames[:2]):  # Just 2 frames
    orig_cls = classify_deepfake(frame)
    t0 = time.time()
    attacked = attack_deepfake_pgd(frame, epsilon=16/255, steps=5)
    elapsed = time.time() - t0
    pert_cls = classify_deepfake(attacked)
    print(f"  Frame {indices[i]}: ", end="")
    for label in orig_cls:
        ov = orig_cls[label] * 100
        pv = pert_cls[label] * 100
        if abs(pv - ov) > 0.1:
            print(f"{label}: {ov:.1f}% -> {pv:.1f}%  ", end="")
    print(f"({elapsed:.1f}s)")

print("\nDone!")
