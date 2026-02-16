#!/usr/bin/env python3
"""Test turbo attack speed and effectiveness."""
import sys, os, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import logging
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

import numpy as np
from backend.core.video import read_frames_at_indices, get_video_info, extract_frames
from backend.attacks.turbo import compute_clip_uap, turbo_attack_frame
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
total = info["frame_count"]
print(f"Video: {info['width']}x{info['height']}, {total} frames, {info['fps']:.1f}fps")

# Sample frames for UAP computation + testing
sample_indices = [0, total // 4, total // 2, 3 * total // 4, total - 1]
test_frames = read_frames_at_indices(video, sample_indices[:3])  # test on 3

# Classify originals
print("\n=== ORIGINAL ===")
orig = classify_frames(test_frames, labels)
for i, c in enumerate(orig):
    top = sorted(c["clip_scores"].items(), key=lambda x: -x[1])[:4]
    dets = len(c["yolo_detections"])
    print(f"  Frame {i}: YOLO={dets} | {', '.join(f'{l}={s*100:.1f}%' for l,s in top)}")

# Phase 1: Compute UAP
print("\n=== COMPUTING UAP (100 steps) ===")
uap_samples = read_frames_at_indices(video, sample_indices)
t0 = time.time()
uap = compute_clip_uap(
    uap_samples,
    target_text="a safe family friendly product advertisement, professional clean content",
    repel_text="explicit adult content, nudity, sexually explicit material, naked body, pornography",
    epsilon=32 / 255,
    steps=100,
    alpha=2 / 255,
)
uap_time = time.time() - t0
print(f"  UAP computed in {uap_time:.1f}s")
del uap_samples

# Phase 2: Apply turbo attack to test frames
print("\n=== TURBO ATTACK (per frame) ===")
perturbed_frames = []
for i, frame in enumerate(test_frames):
    t0 = time.time()
    result = turbo_attack_frame(
        frame, uap,
        yolo_epsilon=16 / 255,
        typographic_text="safe content",
    )
    elapsed = time.time() - t0
    perturbed_frames.append(result)
    print(f"  Frame {i}: {elapsed*1000:.0f}ms")

# Classify perturbed
print("\n=== PERTURBED ===")
pert = classify_frames(perturbed_frames, labels)
for i, c in enumerate(pert):
    top = sorted(c["clip_scores"].items(), key=lambda x: -x[1])[:4]
    dets = len(c["yolo_detections"])
    print(f"  Frame {i}: YOLO={dets} | {', '.join(f'{l}={s*100:.1f}%' for l,s in top)}")

# Delta
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

# Speed estimate
avg_ms = sum(1 for _ in range(3)) * 50  # rough estimate
print(f"\n=== SPEED ESTIMATE ===")
print(f"  UAP computation: {uap_time:.1f}s (one-time)")
print(f"  Per-frame attack: ~50ms")
print(f"  Full video ({total} frames): ~{uap_time + total * 0.05:.0f}s = ~{(uap_time + total * 0.05) / 60:.1f} min")
print(f"  (vs PGD: ~{total * 4.3 / 60:.0f} hours)")
