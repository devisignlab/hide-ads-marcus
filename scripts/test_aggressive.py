#!/usr/bin/env python3
"""Test aggressive dual-objective attack on a single frame (MPS-safe)."""
import sys, os, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
import torch.nn.functional as F
from backend.core.device import get_device
from backend.core.video import read_frames_at_indices, get_video_info
from backend.core.batch import frames_to_tensor, tensor_to_frames
from backend.models.cache import get_yolo, get_clip
from backend.attacks.adversarial import _yolo_loss
from backend.pipeline.classifier import classify_frames

video = sys.argv[1] if len(sys.argv) > 1 else "uploads/2d9fc3accd1c.mp4"
if not os.path.exists(video):
    print(f"Video not found: {video}")
    sys.exit(1)

labels = [
    "adult content", "nudity", "sexually explicit",
    "safe content", "a product advertisement", "a family photo",
    "nature scenery", "a person wearing clothes",
]

info = get_video_info(video)
print(f"Video: {info['width']}x{info['height']}, {info['frame_count']} frames, {info['fps']}fps")

# Pick 3 sample frames (less memory pressure on MPS)
total = info["frame_count"]
indices = [0, total // 2, total - 1]
frames = read_frames_at_indices(video, indices)
print(f"Loaded {len(frames)} sample frames")

# Classify originals
print("\n=== ORIGINAL ===")
orig = classify_frames(frames, labels)
for i, c in enumerate(orig):
    dets = len(c["yolo_detections"])
    top = sorted(c["clip_scores"].items(), key=lambda x: -x[1])[:5]
    print(f"  Frame {i}: YOLO={dets}")
    for lbl, sc in top:
        print(f"    {sc*100:5.1f}% {lbl}")

# === ATTACK ONE FRAME AT A TIME (MPS-safe) ===
device = get_device()
print(f"\nDevice: {device}")

clip_model = get_clip()
yolo_model = get_yolo()
model_inner = yolo_model.get_inner_model()
model_inner.eval()

# Encode text targets once
target_emb = clip_model.encode_text([
    "a safe family friendly product advertisement, professional clean content"
])
repel_emb = clip_model.encode_text([
    "explicit adult content, nudity, sexually explicit material, naked body, pornography"
])

epsilon = 32 / 255
alpha = 4 / 255
steps = 40

perturbed_frames = []

for fi, frame in enumerate(frames):
    print(f"\n--- Attacking frame {fi} ({steps} steps, eps={epsilon*255:.0f}/255) ---")
    start = time.time()

    # Single frame tensor [1, 3, H, W]
    original = frames_to_tensor([frame], device)
    adv = original.clone().detach()

    for step in range(steps):
        grad_total = torch.zeros_like(original)

        # 1) YOLO suppression (weight=0.2)
        adv_y = adv.clone().detach().requires_grad_(True)
        loss_y = _yolo_loss(model_inner, adv_y)
        loss_y.backward()
        g_y = adv_y.grad.float()
        norm_y = g_y.norm(p=2) + 1e-12
        grad_total += 0.2 * (g_y / norm_y)

        # Sync MPS to prevent hang
        if device.type == "mps":
            torch.mps.synchronize()

        # 2) CLIP attract to safe content (weight=0.5)
        adv_c = adv.clone().detach().requires_grad_(True)
        img_emb = clip_model.encode_image_differentiable(adv_c)
        loss_attract = -F.cosine_similarity(img_emb, target_emb).mean()
        loss_attract.backward()
        g_attract = adv_c.grad.float()
        norm_a = g_attract.norm(p=2) + 1e-12
        grad_total += 0.5 * (g_attract / norm_a)

        if device.type == "mps":
            torch.mps.synchronize()

        # 3) CLIP repel from NSFW (weight=0.3)
        adv_r = adv.clone().detach().requires_grad_(True)
        img_emb2 = clip_model.encode_image_differentiable(adv_r)
        loss_repel = F.cosine_similarity(img_emb2, repel_emb).mean()
        loss_repel.backward()
        g_repel = adv_r.grad.float()
        norm_r = g_repel.norm(p=2) + 1e-12
        grad_total += 0.3 * (g_repel / norm_r)

        if device.type == "mps":
            torch.mps.synchronize()

        # PGD step
        with torch.no_grad():
            adv = adv - alpha * grad_total.sign()
            delta = (adv - original).clamp(-epsilon, epsilon)
            adv = (original + delta).clamp(0, 1)

        if step % 10 == 0:
            print(f"  Step {step:2d}: attract={loss_attract.item():.4f} repel={loss_repel.item():.4f} yolo={loss_y.item():.4f}")

    elapsed = time.time() - start
    print(f"  Done in {elapsed:.1f}s")

    perturbed_frames.append(tensor_to_frames(adv)[0])

    # Free GPU memory between frames
    del original, adv, grad_total
    if device.type == "mps":
        torch.mps.empty_cache()
        torch.mps.synchronize()
    elif device.type == "cuda":
        torch.cuda.empty_cache()

# Classify perturbed
print("\n=== PERTURBED ===")
pert = classify_frames(perturbed_frames, labels)
for i, c in enumerate(pert):
    dets = len(c["yolo_detections"])
    top = sorted(c["clip_scores"].items(), key=lambda x: -x[1])[:5]
    print(f"  Frame {i}: YOLO={dets}")
    for lbl, sc in top:
        print(f"    {sc*100:5.1f}% {lbl}")

# Delta summary
print("\n=== DELTA ===")
for i in range(len(orig)):
    print(f"  Frame {i}:")
    for key in ["adult content", "nudity", "sexually explicit", "safe content", "a product advertisement"]:
        ov = orig[i]["clip_scores"].get(key, 0) * 100
        pv = pert[i]["clip_scores"].get(key, 0) * 100
        arrow = "v" if pv < ov else "^" if pv > ov else "="
        print(f"    {key:25s}: {ov:5.1f}% -> {pv:5.1f}% ({pv-ov:+.1f}) {arrow}")
    od = len(orig[i]["yolo_detections"])
    pd = len(pert[i]["yolo_detections"])
    print(f"    {'YOLO detections':25s}: {od} -> {pd}")
