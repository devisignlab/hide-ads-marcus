#!/usr/bin/env python3
"""Process Top-13 end-to-end with combined attack and verify results."""
import sys, os, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import logging
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

import cv2
import numpy as np
import torch
import subprocess

from backend.core.video import read_frames_at_indices, get_video_info, extract_frames, open_video_writer, write_frame
from backend.attacks.temporal import select_keyframe_indices
from backend.attacks.anti_deepfake import attack_deepfake_pgd, classify_deepfake
from backend.core.device import get_device

device = get_device()
print(f"Device: {device}")

# Load CLIP
import clip
clip_model, clip_preprocess = clip.load("ViT-B/32", device=device)
print("CLIP loaded")


def clip_attack_frame(frame_bgr, repel_texts, epsilon=24/255, steps=8):
    from PIL import Image
    alpha = epsilon / steps
    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    pil = Image.fromarray(rgb)
    image_input = clip_preprocess(pil).unsqueeze(0).to(device)
    original = image_input.clone()

    target_text = "a safe family friendly product advertisement, professional clean content"
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
            cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0, device=device
        ).permute(2, 0, 1)
        perturbed = (frame_tensor + delta_resized).clamp(0, 1)
        result = (perturbed.permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
        result = cv2.cvtColor(result, cv2.COLOR_RGB2BGR)
    return result


# === CONFIG ===
video_path = "uploads/Top-13.mp4"
output_path = "outputs/Top-13-adversarial.mp4"
os.makedirs("outputs", exist_ok=True)

info = get_video_info(video_path)
total = info["frame_count"]
fps = info["fps"]
w, h = info["width"], info["height"]
print(f"\nInput: {video_path}")
print(f"  {total} frames, {w}x{h}, {fps:.1f}fps, {os.path.getsize(video_path)/1024/1024:.1f}MB")

# Repel texts (from smart analysis)
repel_texts = [
    "explicit adult content, nudity, sexually explicit material, naked body, pornography",
    "sensual content, seductive poses, provocative imagery, erotic",
    "sexually explicit content, pornography, sex act",
]

# === STEP 1: Compute keyframe deltas ===
keyframe_interval = 15
key_indices = select_keyframe_indices(total, keyframe_interval)
print(f"\nStep 1: Computing {len(key_indices)} keyframe deltas (interval={keyframe_interval})...")

deltas = {}
t_start = time.time()
for ki, idx in enumerate(key_indices):
    frame = read_frames_at_indices(video_path, [idx])[0]

    # Combined attack: anti-deepfake + CLIP
    result = attack_deepfake_pgd(frame, epsilon=16/255, steps=5, alpha=4/255)
    result = clip_attack_frame(result, repel_texts, epsilon=24/255, steps=8)

    delta = result.astype(np.int16) - frame.astype(np.int16)
    deltas[idx] = np.clip(delta, -127, 127).astype(np.int8)

    if device.type == "mps":
        torch.mps.empty_cache()

    elapsed = time.time() - t_start
    rate = (ki + 1) / elapsed
    eta = (len(key_indices) - ki - 1) / rate if rate > 0 else 0
    if (ki + 1) % 5 == 0 or ki == 0:
        print(f"  [{ki+1}/{len(key_indices)}] {elapsed:.0f}s elapsed, ETA ~{eta:.0f}s ({rate:.2f} keys/s)")

t_keys = time.time() - t_start
print(f"  Done: {len(key_indices)} keyframes in {t_keys:.0f}s ({len(key_indices)/t_keys:.2f} keys/s)")

# === STEP 2: Stream video with interpolation ===
print(f"\nStep 2: Streaming {total} frames...")
sorted_keys = sorted(deltas.keys())

temp_raw = "outputs/Top-13-raw.mp4"
writer = open_video_writer(temp_raw, fps, w, h)
t_stream = time.time()

for frame_idx, frame in enumerate(extract_frames(video_path)):
    if frame_idx in deltas:
        out = np.clip(frame.astype(np.int16) + deltas[frame_idx].astype(np.int16), 0, 255).astype(np.uint8)
    else:
        prev_key = next_key = None
        for k in sorted_keys:
            if k <= frame_idx:
                prev_key = k
            if k >= frame_idx and next_key is None:
                next_key = k
        if prev_key is not None and next_key is not None and prev_key != next_key:
            t = (frame_idx - prev_key) / (next_key - prev_key)
            d = ((1 - t) * deltas[prev_key].astype(np.float32) + t * deltas[next_key].astype(np.float32))
            out = np.clip(frame.astype(np.int16) + d.astype(np.int16), 0, 255).astype(np.uint8)
        elif prev_key is not None:
            out = np.clip(frame.astype(np.int16) + deltas[prev_key].astype(np.int16), 0, 255).astype(np.uint8)
        elif next_key is not None:
            out = np.clip(frame.astype(np.int16) + deltas[next_key].astype(np.int16), 0, 255).astype(np.uint8)
        else:
            out = frame
    write_frame(writer, out)
    if (frame_idx + 1) % 500 == 0:
        print(f"  [{frame_idx+1}/{total}] {time.time()-t_stream:.0f}s")

writer.release()
print(f"  Done: {total} frames in {time.time()-t_stream:.0f}s")

# === STEP 3: Mux + remux ===
print(f"\nStep 3: Mux audio + adversarial remux...")
muxed = "outputs/Top-13-muxed.mp4"
subprocess.run([
    "ffmpeg", "-y", "-i", temp_raw, "-i", video_path,
    "-map", "0:v", "-map", "1:a", "-c:v", "copy", "-c:a", "copy", "-shortest", muxed
], capture_output=True, timeout=120)

subprocess.run([
    "ffmpeg", "-y", "-i", muxed,
    "-c:v", "copy", "-c:a", "aac", "-b:a", "128k",
    "-map_metadata", "-1", "-movflags", "+faststart", output_path
], capture_output=True, timeout=120)

for tmp in [temp_raw, muxed]:
    if os.path.exists(tmp):
        os.unlink(tmp)

print(f"  Output: {output_path} ({os.path.getsize(output_path)/1024/1024:.1f} MB)")

# === STEP 4: Verify ===
print(f"\n{'='*60}")
print(f"  BEFORE vs AFTER COMPARISON")
print(f"{'='*60}")

verify_indices = [0, total//4, total//2, 3*total//4]
orig_frames = read_frames_at_indices(video_path, verify_indices)
out_frames = read_frames_at_indices(output_path, verify_indices)

for i in range(len(verify_indices)):
    idx = verify_indices[i]
    o = orig_frames[i]
    p = out_frames[i]

    diff = np.abs(o.astype(float) - p.astype(float))

    # Deepfake
    o_df = classify_deepfake(o).get("Deepfake", 0) * 100
    p_df = classify_deepfake(p).get("Deepfake", 0) * 100

    # CLIP NSFW
    from backend.attacks.smart_analyze import analyze_frame
    o_a = analyze_frame(o)
    p_a = analyze_frame(p)
    o_nsfw = max(o_a["scores"].get(k, 0) for k in ["nudity", "sexual", "sensual", "lingerie"]) * 100
    p_nsfw = max(p_a["scores"].get(k, 0) for k in ["nudity", "sexual", "sensual", "lingerie"]) * 100
    o_safe = o_a["scores"].get("safe", 0) * 100
    p_safe = p_a["scores"].get("safe", 0) * 100

    print(f"\n  Frame {idx}:")
    print(f"    Deepfake:    {o_df:5.1f}% -> {p_df:5.1f}% ({p_df-o_df:+.1f})")
    print(f"    Max NSFW:    {o_nsfw:5.1f}% -> {p_nsfw:5.1f}% ({p_nsfw-o_nsfw:+.1f})")
    print(f"    Safe:        {o_safe:5.1f}% -> {p_safe:5.1f}% ({p_safe-o_safe:+.1f})")
    print(f"    Risk:        {o_a['risk_level']:8s} -> {p_a['risk_level']}")
    print(f"    Pixel diff:  L_inf={diff.max():.0f}/255, mean={diff.mean():.1f}/255")

    if device.type == "mps":
        torch.mps.empty_cache()

total_time = time.time() - t_start
print(f"\n{'='*60}")
print(f"  TOTAL: {total_time:.0f}s ({total_time/60:.1f} min)")
print(f"  Output: {output_path}")
print(f"{'='*60}")
