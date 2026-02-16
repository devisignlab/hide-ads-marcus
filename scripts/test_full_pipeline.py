#!/usr/bin/env python3
"""Full end-to-end pipeline test.

1. Smart-analyze all videos
2. Process one video with combined attack (CLIP + anti-deepfake + remux)
3. Compare before/after scores
"""
import sys, os, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import logging
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

import cv2
import numpy as np
import torch

from backend.core.video import (
    read_frames_at_indices, get_video_info, extract_frames,
    open_video_writer, write_frame
)
from backend.attacks.temporal import select_keyframe_indices
from backend.core.device import get_device
from backend.attacks.smart_analyze import analyze_frame
from backend.attacks.anti_deepfake import attack_deepfake_pgd, classify_deepfake

# === CLIP attack ===
import clip
device = get_device()
print(f"Device: {device}")
clip_model, clip_preprocess = clip.load("ViT-B/32", device=device)


def clip_attack_frame(
    frame_bgr: np.ndarray,
    repel_texts: list[str],
    target_text: str = "a safe family friendly product advertisement, professional clean content",
    epsilon: float = 24 / 255,
    steps: int = 8,
) -> np.ndarray:
    """CLIP PGD attack on a single frame."""
    from PIL import Image
    alpha = epsilon / steps

    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    pil = Image.fromarray(rgb)
    image_input = clip_preprocess(pil).unsqueeze(0).to(device)
    original = image_input.clone()

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

    with torch.no_grad():
        delta = (image_input - original).squeeze(0)
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

    return result


def combined_attack_frame(
    frame_bgr: np.ndarray,
    repel_texts: list[str],
    use_clip: bool = True,
    use_deepfake: bool = True,
) -> np.ndarray:
    """Apply both CLIP and anti-deepfake attacks to a single frame."""
    result = frame_bgr.copy()

    if use_deepfake:
        result = attack_deepfake_pgd(result, epsilon=16/255, steps=5, alpha=4/255)

    if use_clip and repel_texts:
        result = clip_attack_frame(result, repel_texts, epsilon=24/255, steps=8)

    return result


# ============================================================
# PHASE 1: Analyze all videos
# ============================================================
print("\n" + "=" * 60)
print("  PHASE 1: SMART ANALYSIS OF ALL VIDEOS")
print("=" * 60)

videos = []
for f in sorted(os.listdir("uploads")):
    if f.endswith(".mp4") and os.path.getsize(f"uploads/{f}") > 1000:
        videos.append(f"uploads/{f}")

for video in videos:
    info = get_video_info(video)
    print(f"\n--- {os.path.basename(video)} ({info['frame_count']} frames, {info['width']}x{info['height']}) ---")

    # Sample 2 frames for quick analysis
    total = info["frame_count"]
    indices = [total // 4, 3 * total // 4]
    indices = [min(i, total - 1) for i in indices]
    frames = read_frames_at_indices(video, indices)

    for i, frame in enumerate(frames):
        result = analyze_frame(frame)
        flags_str = ', '.join(result['flags']) if result['flags'] else 'clean'
        top_scores = sorted(result['scores'].items(), key=lambda x: -x[1])[:3]
        clips = ' | '.join(f"{k}={v*100:.0f}%" for k, v in top_scores)
        print(f"  Frame {indices[i]:5d}: [{result['risk_level']:8s}] deepfake={result['deepfake_score']*100:.0f}% | {flags_str}")
        print(f"                  CLIP: {clips}")

    if device.type == "mps":
        torch.mps.empty_cache()

# ============================================================
# PHASE 2: Process Top-13 end-to-end (shorter video)
# ============================================================
print("\n" + "=" * 60)
print("  PHASE 2: FULL PIPELINE - Top-13.mp4")
print("=" * 60)

video_path = "uploads/Top-13.mp4"
output_path = "outputs/Top-13-adversarial.mp4"
os.makedirs("outputs", exist_ok=True)

info = get_video_info(video_path)
total = info["frame_count"]
fps = info["fps"]
w, h = info["width"], info["height"]
print(f"\n  Input: {total} frames, {w}x{h}, {fps}fps")

# Step 1: Smart analyze to get repel texts
print("\n  Step 1: Smart analysis...")
sample_indices = [total // 4, total // 2, 3 * total // 4]
sample_frames = read_frames_at_indices(video_path, sample_indices)
all_repel = set()
use_deepfake = False
for frame in sample_frames:
    result = analyze_frame(frame)
    all_repel.update(result["repel_texts"])
    if "deepfake" in result["flags"]:
        use_deepfake = True
repel_texts = sorted(all_repel)
print(f"  Deepfake attack: {'YES' if use_deepfake else 'NO'}")
print(f"  CLIP attack: {'YES' if repel_texts else 'NO'}")
if repel_texts:
    print(f"  Repel texts: {repel_texts}")

# Step 2: Compute keyframe deltas (combined attack)
print(f"\n  Step 2: Computing keyframe perturbations...")
keyframe_interval = 15  # Every 15 frames
key_indices = select_keyframe_indices(total, keyframe_interval)
print(f"  Keyframes: {len(key_indices)} (every {keyframe_interval} frames)")

deltas = {}
t_start = time.time()
for ki, idx in enumerate(key_indices):
    frame = read_frames_at_indices(video_path, [idx])[0]
    perturbed = combined_attack_frame(frame, repel_texts, use_clip=bool(repel_texts), use_deepfake=use_deepfake)
    delta = perturbed.astype(np.int16) - frame.astype(np.int16)
    deltas[idx] = np.clip(delta, -127, 127).astype(np.int8)

    if device.type == "mps":
        torch.mps.empty_cache()

    if (ki + 1) % 10 == 0 or ki == 0:
        elapsed = time.time() - t_start
        rate = (ki + 1) / elapsed
        eta = (len(key_indices) - ki - 1) / rate
        print(f"    [{ki+1}/{len(key_indices)}] {elapsed:.0f}s elapsed, ~{eta:.0f}s remaining")

t_keyframes = time.time() - t_start
print(f"  Keyframes done in {t_keyframes:.0f}s ({len(key_indices)/t_keyframes:.1f} keys/s)")

# Step 3: Stream video with interpolated deltas
print(f"\n  Step 3: Streaming {total} frames with delta interpolation...")
sorted_keys = sorted(deltas.keys())

temp_raw = "outputs/Top-13-raw.mp4"
writer = open_video_writer(temp_raw, fps, w, h)
t_stream = time.time()

for frame_idx, frame in enumerate(extract_frames(video_path)):
    if frame_idx in deltas:
        result = np.clip(frame.astype(np.int16) + deltas[frame_idx].astype(np.int16), 0, 255).astype(np.uint8)
    else:
        # Interpolate between nearest keyframes
        prev_key = None
        next_key = None
        for k in sorted_keys:
            if k <= frame_idx:
                prev_key = k
            if k >= frame_idx and next_key is None:
                next_key = k

        if prev_key is not None and next_key is not None and prev_key != next_key:
            t = (frame_idx - prev_key) / (next_key - prev_key)
            delta = ((1 - t) * deltas[prev_key].astype(np.float32) +
                     t * deltas[next_key].astype(np.float32))
            result = np.clip(frame.astype(np.int16) + delta.astype(np.int16), 0, 255).astype(np.uint8)
        elif prev_key is not None:
            result = np.clip(frame.astype(np.int16) + deltas[prev_key].astype(np.int16), 0, 255).astype(np.uint8)
        elif next_key is not None:
            result = np.clip(frame.astype(np.int16) + deltas[next_key].astype(np.int16), 0, 255).astype(np.uint8)
        else:
            result = frame

    write_frame(writer, result)
    if (frame_idx + 1) % 500 == 0:
        elapsed = time.time() - t_stream
        print(f"    [{frame_idx+1}/{total}] {elapsed:.0f}s ({(frame_idx+1)/elapsed:.0f} fps)")

writer.release()
t_stream_total = time.time() - t_stream
print(f"  Streaming done in {t_stream_total:.0f}s ({total/t_stream_total:.0f} fps)")

# Step 4: Mux audio + adversarial remux
print(f"\n  Step 4: Mux audio + adversarial remux...")
import subprocess

# Mux with original audio
muxed = "outputs/Top-13-muxed.mp4"
subprocess.run([
    "ffmpeg", "-y", "-i", temp_raw, "-i", video_path,
    "-map", "0:v", "-map", "1:a", "-c:v", "copy", "-c:a", "copy",
    "-shortest", muxed
], capture_output=True, timeout=120)

# Adversarial remux (hash bypass)
subprocess.run([
    "ffmpeg", "-y", "-i", muxed,
    "-c:v", "copy", "-c:a", "aac", "-b:a", "128k",
    "-map_metadata", "-1", "-movflags", "+faststart",
    output_path
], capture_output=True, timeout=120)

# Cleanup
for tmp in [temp_raw, muxed]:
    if os.path.exists(tmp):
        os.unlink(tmp)

output_size = os.path.getsize(output_path) / 1024 / 1024
input_size = os.path.getsize(video_path) / 1024 / 1024
print(f"  Output: {output_path} ({output_size:.1f} MB, was {input_size:.1f} MB)")

# ============================================================
# PHASE 3: Verify results
# ============================================================
print("\n" + "=" * 60)
print("  PHASE 3: BEFORE / AFTER COMPARISON")
print("=" * 60)

verify_indices = [0, total // 4, total // 2, 3 * total // 4]
verify_indices = [min(i, total - 1) for i in verify_indices]
original_frames = read_frames_at_indices(video_path, verify_indices)
output_frames = read_frames_at_indices(output_path, verify_indices)

print(f"\n  {'Frame':>8s}  {'Metric':20s}  {'BEFORE':>10s}  {'AFTER':>10s}  {'DELTA':>10s}")
print("  " + "-" * 70)

for i in range(len(verify_indices)):
    idx = verify_indices[i]
    orig = original_frames[i]
    proc = output_frames[i]

    # Pixel diff
    diff = np.abs(orig.astype(float) - proc.astype(float))
    linf = diff.max()
    mean_diff = diff.mean()

    # Deepfake
    orig_df = classify_deepfake(orig)
    proc_df = classify_deepfake(proc)
    orig_fake = orig_df.get("Deepfake", 0) * 100
    proc_fake = proc_df.get("Deepfake", 0) * 100

    # CLIP NSFW
    orig_analysis = analyze_frame(orig)
    proc_analysis = analyze_frame(proc)

    orig_nsfw = max(orig_analysis["scores"].get(k, 0) for k in ["nudity", "sexual", "sensual", "lingerie"]) * 100
    proc_nsfw = max(proc_analysis["scores"].get(k, 0) for k in ["nudity", "sexual", "sensual", "lingerie"]) * 100

    orig_safe = orig_analysis["scores"].get("safe", 0) * 100
    proc_safe = proc_analysis["scores"].get("safe", 0) * 100

    print(f"  {idx:>8d}  {'Deepfake %':20s}  {orig_fake:>9.1f}%  {proc_fake:>9.1f}%  {proc_fake-orig_fake:>+9.1f}%")
    print(f"  {'':>8s}  {'Max NSFW %':20s}  {orig_nsfw:>9.1f}%  {proc_nsfw:>9.1f}%  {proc_nsfw-orig_nsfw:>+9.1f}%")
    print(f"  {'':>8s}  {'Safe content %':20s}  {orig_safe:>9.1f}%  {proc_safe:>9.1f}%  {proc_safe-orig_safe:>+9.1f}%")
    print(f"  {'':>8s}  {'Pixel L_inf':20s}  {'':>10s}  {linf:>9.1f}  {'':>10s}")
    print(f"  {'':>8s}  {'Risk level':20s}  {orig_analysis['risk_level']:>10s}  {proc_analysis['risk_level']:>10s}")
    print()

    if device.type == "mps":
        torch.mps.empty_cache()

total_time = time.time() - t_start
print(f"\n  TOTAL TIME: {total_time:.0f}s ({total_time/60:.1f} min)")
print(f"  Output: {output_path}")
print("\nDone!")
