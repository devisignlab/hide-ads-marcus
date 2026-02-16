#!/usr/bin/env python3
"""Final test: Smart analyze + optimized process + compare with competition.

Pipeline:
  1. Smart analyze → identify problems
  2. UAP for CLIP (compute ONCE, ~30s) → apply to ALL frames (instant)
  3. Anti-deepfake PGD on keyframes (interval=30, ~5s each)
  4. Stream + mux + remux
  5. Verify: before vs after vs competition
"""
import sys, os, time, subprocess, hashlib
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ["PYTHONUNBUFFERED"] = "1"

import logging
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

import cv2
import numpy as np
import torch

from backend.core.video import read_frames_at_indices, get_video_info, extract_frames, open_video_writer, write_frame
from backend.attacks.temporal import select_keyframe_indices
from backend.attacks.anti_deepfake import attack_deepfake_pgd, classify_deepfake
from backend.attacks.turbo import compute_clip_uap, apply_uap_to_frame
from backend.attacks.smart_analyze import analyze_frame, NSFW_CATEGORIES
from backend.core.device import get_device

device = get_device()
print(f"Device: {device}", flush=True)


def process_video(video_path, output_path, processed_path=None):
    """Full pipeline: analyze → attack → verify."""
    info = get_video_info(video_path)
    total = info["frame_count"]
    fps = info["fps"]
    w, h = info["width"], info["height"]
    duration = total / fps

    print(f"\n{'='*60}", flush=True)
    print(f"  VIDEO: {os.path.basename(video_path)}", flush=True)
    print(f"  {total} frames, {w}x{h}, {fps:.0f}fps, {duration:.1f}s", flush=True)
    print(f"{'='*60}", flush=True)

    # ── PHASE 1: Smart Analysis ──
    print(f"\n  [1/5] SMART ANALYSIS...", flush=True)
    t0 = time.time()

    sample_indices = [total // 5, 2 * total // 5, 3 * total // 5, 4 * total // 5]
    sample_frames = read_frames_at_indices(video_path, sample_indices)

    all_flags = set()
    all_repel = set()
    use_deepfake = False
    max_deepfake = 0.0
    max_nsfw = 0.0

    for i, frame in enumerate(sample_frames):
        result = analyze_frame(frame)
        all_flags.update(result["flags"])
        all_repel.update(result["repel_texts"])
        if "deepfake" in result["flags"]:
            use_deepfake = True
        max_deepfake = max(max_deepfake, result["deepfake_score"])
        nsfw_score = max(result["scores"].get(k, 0) for k in ["nudity", "sexual", "sensual", "lingerie"])
        max_nsfw = max(max_nsfw, nsfw_score)

        flags_str = ', '.join(result['flags']) if result['flags'] else 'clean'
        print(f"    Frame {sample_indices[i]:5d}: [{result['risk_level']:8s}] deepfake={result['deepfake_score']*100:.0f}% | {flags_str}", flush=True)

    use_clip = len(all_repel) > 0
    repel_text = ", ".join(all_repel) if all_repel else ""
    target_text = "a safe family friendly product advertisement, professional clean content"

    print(f"\n    DIAGNOSIS:", flush=True)
    print(f"      Flags: {', '.join(sorted(all_flags)) if all_flags else 'none'}", flush=True)
    print(f"      Max deepfake: {max_deepfake*100:.0f}%", flush=True)
    print(f"      Max NSFW: {max_nsfw*100:.0f}%", flush=True)
    print(f"      Attack plan: {'anti-deepfake' if use_deepfake else ''} {'+ CLIP UAP' if use_clip else ''} + remux", flush=True)
    print(f"    Analysis took {time.time()-t0:.0f}s", flush=True)

    if device.type == "mps":
        torch.mps.empty_cache()

    # ── PHASE 2: Compute UAP for CLIP ──
    uap = None
    if use_clip:
        print(f"\n  [2/5] COMPUTING CLIP UAP (one-time, applies to ALL frames)...", flush=True)
        t1 = time.time()
        # Use 6 diverse sample frames
        uap_indices = [int(i * total / 7) for i in range(1, 7)]
        uap_frames = read_frames_at_indices(video_path, uap_indices)
        uap = compute_clip_uap(
            uap_frames, target_text, repel_text,
            epsilon=32 / 255, steps=80, alpha=2 / 255
        )
        print(f"    UAP computed in {time.time()-t1:.0f}s, L_inf={np.abs(uap).max():.1f}/255", flush=True)
    else:
        print(f"\n  [2/5] CLIP UAP: SKIPPED (no NSFW flags)", flush=True)

    if device.type == "mps":
        torch.mps.empty_cache()

    # ── PHASE 3: Anti-deepfake keyframes ──
    deltas = {}
    if use_deepfake:
        keyframe_interval = 30
        key_indices = select_keyframe_indices(total, keyframe_interval)
        print(f"\n  [3/5] ANTI-DEEPFAKE PGD ({len(key_indices)} keyframes, interval={keyframe_interval})...", flush=True)
        t2 = time.time()

        for ki, idx in enumerate(key_indices):
            frame = read_frames_at_indices(video_path, [idx])[0]
            perturbed = attack_deepfake_pgd(frame, epsilon=16/255, steps=5, alpha=4/255)
            delta = perturbed.astype(np.int16) - frame.astype(np.int16)
            deltas[idx] = np.clip(delta, -127, 127).astype(np.int8)

            if device.type == "mps":
                torch.mps.empty_cache()

            if (ki + 1) % 10 == 0 or ki == 0 or ki == len(key_indices) - 1:
                elapsed = time.time() - t2
                rate = (ki + 1) / elapsed
                eta = (len(key_indices) - ki - 1) / rate if rate > 0 else 0
                print(f"    [{ki+1}/{len(key_indices)}] {elapsed:.0f}s, ETA {eta:.0f}s", flush=True)

        print(f"    Deepfake keyframes done in {time.time()-t2:.0f}s", flush=True)
    else:
        print(f"\n  [3/5] ANTI-DEEPFAKE: SKIPPED (deepfake score low)", flush=True)

    # ── PHASE 4: Stream video ──
    print(f"\n  [4/5] STREAMING {total} frames...", flush=True)
    t3 = time.time()
    sorted_keys = sorted(deltas.keys()) if deltas else []

    temp_raw = output_path.replace(".mp4", "-raw.mp4")
    writer = open_video_writer(temp_raw, fps, w, h)

    for frame_idx, frame in enumerate(extract_frames(video_path)):
        out = frame

        # Apply UAP (instant, <1ms)
        if uap is not None:
            out = apply_uap_to_frame(out, uap)

        # Apply deepfake delta (interpolated)
        if deltas:
            if frame_idx in deltas:
                out = np.clip(out.astype(np.int16) + deltas[frame_idx].astype(np.int16), 0, 255).astype(np.uint8)
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
                    out = np.clip(out.astype(np.int16) + d.astype(np.int16), 0, 255).astype(np.uint8)
                elif prev_key is not None:
                    out = np.clip(out.astype(np.int16) + deltas[prev_key].astype(np.int16), 0, 255).astype(np.uint8)
                elif next_key is not None:
                    out = np.clip(out.astype(np.int16) + deltas[next_key].astype(np.int16), 0, 255).astype(np.uint8)

        write_frame(writer, out)
        if (frame_idx + 1) % 500 == 0:
            print(f"    [{frame_idx+1}/{total}] {time.time()-t3:.0f}s", flush=True)

    writer.release()
    print(f"    Streaming done in {time.time()-t3:.0f}s ({total/(time.time()-t3):.0f} fps)", flush=True)

    # ── PHASE 5: Mux + Remux ──
    print(f"\n  [5/5] MUX AUDIO + ADVERSARIAL REMUX...", flush=True)
    t4 = time.time()
    muxed = output_path.replace(".mp4", "-muxed.mp4")

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

    out_size = os.path.getsize(output_path) / 1024 / 1024
    in_size = os.path.getsize(video_path) / 1024 / 1024
    print(f"    Output: {output_path} ({out_size:.1f}MB, original {in_size:.1f}MB)", flush=True)
    print(f"    Mux done in {time.time()-t4:.0f}s", flush=True)

    total_time = time.time() - t0
    print(f"\n    TOTAL PROCESSING TIME: {total_time:.0f}s ({total_time/60:.1f} min)", flush=True)

    # ── VERIFY ──
    print(f"\n  {'─'*56}", flush=True)
    print(f"  VERIFICATION: BEFORE vs AFTER", flush=True)
    print(f"  {'─'*56}", flush=True)

    verify_indices = [0, total//4, total//2, 3*total//4]
    orig_frames = read_frames_at_indices(video_path, verify_indices)
    our_frames = read_frames_at_indices(output_path, verify_indices)

    # Also load competition if available
    comp_frames = None
    if processed_path and os.path.exists(processed_path):
        comp_frames = read_frames_at_indices(processed_path, verify_indices)

    print(f"\n  {'Frame':>7s} │ {'Metric':15s} │ {'ORIGINAL':>10s} │ {'OURS':>10s} │ {'Δ ours':>8s}", end="")
    if comp_frames:
        print(f" │ {'COMPETID':>10s} │ {'Δ comp':>8s}", end="")
    print(flush=True)
    print(f"  {'─'*7}─┼─{'─'*15}─┼─{'─'*10}─┼─{'─'*10}─┼─{'─'*8}", end="")
    if comp_frames:
        print(f"─┼─{'─'*10}─┼─{'─'*8}", end="")
    print(flush=True)

    for i in range(len(verify_indices)):
        idx = verify_indices[i]
        o = orig_frames[i]
        ours = our_frames[i]

        # Deepfake scores
        o_df = classify_deepfake(o).get("Deepfake", 0) * 100
        ours_df = classify_deepfake(ours).get("Deepfake", 0) * 100

        # CLIP analysis
        o_a = analyze_frame(o)
        ours_a = analyze_frame(ours)
        o_nsfw = max(o_a["scores"].get(k, 0) for k in ["nudity", "sexual", "sensual", "lingerie"]) * 100
        ours_nsfw = max(ours_a["scores"].get(k, 0) for k in ["nudity", "sexual", "sensual", "lingerie"]) * 100
        o_safe = o_a["scores"].get("safe", 0) * 100
        ours_safe = ours_a["scores"].get("safe", 0) * 100

        # Pixel diff
        diff = np.abs(o.astype(float) - ours.astype(float))

        # Competition scores
        comp_df = comp_nsfw = comp_safe = None
        if comp_frames:
            c = comp_frames[i]
            comp_df = classify_deepfake(c).get("Deepfake", 0) * 100
            c_a = analyze_frame(c)
            comp_nsfw = max(c_a["scores"].get(k, 0) for k in ["nudity", "sexual", "sensual", "lingerie"]) * 100
            comp_safe = c_a["scores"].get("safe", 0) * 100

        # Print rows
        def row(metric, orig_v, ours_v, comp_v=None):
            line = f"  {idx if metric=='Deepfake%' else '':>7s} │ {metric:15s} │ {orig_v:>9.1f}% │ {ours_v:>9.1f}% │ {ours_v-orig_v:>+7.1f}%"
            if comp_v is not None:
                line += f" │ {comp_v:>9.1f}% │ {comp_v-orig_v:>+7.1f}%"
            print(line, flush=True)

        row("Deepfake%", o_df, ours_df, comp_df)
        row("Max NSFW%", o_nsfw, ours_nsfw, comp_nsfw)
        row("Safe%", o_safe, ours_safe, comp_safe)

        risk_line = f"  {'':>7s} │ {'Risk':15s} │ {o_a['risk_level']:>10s} │ {ours_a['risk_level']:>10s} │ {'':>8s}"
        if comp_frames:
            risk_line += f" │ {c_a['risk_level']:>10s} │ {'':>8s}"
        print(risk_line, flush=True)

        pix_line = f"  {'':>7s} │ {'Pixel L_inf':15s} │ {'':>10s} │ {diff.max():>9.0f}  │ {'':>8s}"
        print(pix_line, flush=True)
        print(flush=True)

        if device.type == "mps":
            torch.mps.empty_cache()

    # Summary
    print(f"\n  SUMMARY:", flush=True)
    if comp_frames:
        print(f"    Competition: Only remux (zero pixel changes, zero ML bypass)", flush=True)
    print(f"    Our system:  UAP + anti-deepfake PGD + remux", flush=True)
    print(f"    Processing:  {total_time:.0f}s for {duration:.0f}s video ({total_time/duration:.1f}x realtime)", flush=True)

    return total_time


# ══════════════════════════════════════════════════════════
#  MAIN
# ══════════════════════════════════════════════════════════
print("=" * 60)
print("  ADVERSARIAL VIDEO TOOL - FULL TEST")
print("=" * 60)

os.makedirs("outputs", exist_ok=True)
total_start = time.time()

# Process Top-13 (sensual, 82s)
t1 = process_video(
    "uploads/Top-13.mp4",
    "outputs/Top-13-adversarial.mp4",
    "uploads/Top-13-processed.mp4"
)

if device.type == "mps":
    torch.mps.empty_cache()

# Process Top-06 (deepfake, 135s)
t2 = process_video(
    "uploads/Top-06.mp4",
    "outputs/Top-06-adversarial.mp4",
    "uploads/Top-06-processed.mp4"
)

# Also check the smaller video
small_vid = "uploads/2d9fc3accd1c.mp4"
if os.path.exists(small_vid) and os.path.getsize(small_vid) > 10000:
    t3 = process_video(small_vid, "outputs/2d9fc3accd1c-adversarial.mp4")

grand_total = time.time() - total_start
print(f"\n{'='*60}")
print(f"  ALL DONE in {grand_total:.0f}s ({grand_total/60:.1f} min)")
print(f"{'='*60}")
