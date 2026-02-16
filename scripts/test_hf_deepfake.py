#!/usr/bin/env python3
"""Test HuggingFace deepfake detection models against competition videos."""
import sys, os, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import cv2
import numpy as np
from PIL import Image
import torch

device = "mps" if torch.backends.mps.is_available() else "cpu"
print(f"Device: {device}")

# Load videos
from backend.core.video import read_frames_at_indices, get_video_info

ORIGINAL = "uploads/Top-06.mp4"
PROCESSED = "uploads/Top-06-processed.mp4"

info = get_video_info(ORIGINAL)
total = info["frame_count"]
indices = [0, 500, 1000, 2000, 3000]
indices = [i for i in indices if i < total]

print(f"Video: {info['width']}x{info['height']}, {total} frames")
print(f"Sampling frames: {indices}")

frames_orig = read_frames_at_indices(ORIGINAL, indices)
frames_proc = read_frames_at_indices(PROCESSED, indices)

# === Test 1: HuggingFace deepfake detector ===
print("\n" + "="*60)
print("HuggingFace: prithivMLmods/Deep-Fake-Detector-v2-Model")
print("="*60)

from transformers import pipeline

try:
    # Use CPU to avoid MPS issues with some models
    clf = pipeline("image-classification", model="prithivMLmods/Deep-Fake-Detector-v2-Model", device="cpu")

    for i, (fo, fp) in enumerate(zip(frames_orig, frames_proc)):
        img_o = Image.fromarray(cv2.cvtColor(fo, cv2.COLOR_BGR2RGB))
        img_p = Image.fromarray(cv2.cvtColor(fp, cv2.COLOR_BGR2RGB))

        t0 = time.time()
        res_o = clf(img_o)
        res_p = clf(img_p)
        elapsed = time.time() - t0

        orig_str = ", ".join(f"{r['label']}={r['score']*100:.1f}%" for r in res_o)
        proc_str = ", ".join(f"{r['label']}={r['score']*100:.1f}%" for r in res_p)

        print(f"\n  Frame {indices[i]}:")
        print(f"    Original:  {orig_str}")
        print(f"    Processed: {proc_str}")
        print(f"    Time: {elapsed:.1f}s")
except Exception as e:
    print(f"  ERROR: {e}")

# === Test 2: Alternative model ===
print("\n" + "="*60)
print("HuggingFace: dima806/deepfake_vs_real_face_detection")
print("="*60)

try:
    clf2 = pipeline("image-classification", model="dima806/deepfake_vs_real_face_detection", device="cpu")

    for i, (fo, fp) in enumerate(zip(frames_orig, frames_proc)):
        img_o = Image.fromarray(cv2.cvtColor(fo, cv2.COLOR_BGR2RGB))
        img_p = Image.fromarray(cv2.cvtColor(fp, cv2.COLOR_BGR2RGB))

        t0 = time.time()
        res_o = clf2(img_o)
        res_p = clf2(img_p)
        elapsed = time.time() - t0

        orig_str = ", ".join(f"{r['label']}={r['score']*100:.1f}%" for r in res_o)
        proc_str = ", ".join(f"{r['label']}={r['score']*100:.1f}%" for r in res_p)

        print(f"\n  Frame {indices[i]}:")
        print(f"    Original:  {orig_str}")
        print(f"    Processed: {proc_str}")
        print(f"    Time: {elapsed:.1f}s")
except Exception as e:
    print(f"  ERROR: {e}")

# === Test 3: CLIP analysis ===
print("\n" + "="*60)
print("CLIP Analysis (deepfake-focused labels)")
print("="*60)

from backend.pipeline.classifier import classify_frames

labels = [
    "a deepfake video", "a real authentic video",
    "AI generated content", "natural camera footage",
    "adult content", "nudity", "safe content",
    "a product advertisement",
]

cls_orig = classify_frames(frames_orig[:3], labels)
cls_proc = classify_frames(frames_proc[:3], labels)

for i in range(min(3, len(cls_orig))):
    print(f"\n  Frame {indices[i]}:")
    print(f"    YOLO: orig={len(cls_orig[i]['yolo_detections'])} det, proc={len(cls_proc[i]['yolo_detections'])} det")

    for label in sorted(cls_orig[i]["clip_scores"], key=lambda k: cls_orig[i]["clip_scores"][k], reverse=True):
        ov = cls_orig[i]["clip_scores"][label] * 100
        pv = cls_proc[i]["clip_scores"].get(label, 0) * 100
        delta = pv - ov
        print(f"      {label:30s}: {ov:5.1f}% -> {pv:5.1f}% ({delta:+.1f})")

print("\n=== CONCLUSÃO ===")
print("Se Original e Processed dão resultados IDÊNTICOS nos classificadores ML,")
print("então o bypass da competição é na camada de HASH MATCHING (não ML).")
print("O re-mux + re-encode de áudio muda o hash do arquivo sem alterar pixels.")
