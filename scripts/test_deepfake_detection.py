#!/usr/bin/env python3
"""Test multiple deepfake detection approaches against original and processed competition videos.

Goal: Find which classifier the competition's processed video defeats.
"""
import sys, os, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import logging
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

import numpy as np
import cv2
import torch
import torch.nn.functional as F
from PIL import Image

from backend.core.video import read_frames_at_indices, get_video_info

# ---------- Config ----------
ORIGINAL = "uploads/Top-06.mp4"
PROCESSED = "uploads/Top-06-processed.mp4"
SAMPLE_FRAMES = [0, 100, 500, 1000, 2000, 3000, 4000]
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

# ---------- Helpers ----------
def extract_faces(frames, min_face_size=60):
    """Extract faces using MTCNN from facenet-pytorch."""
    from facenet_pytorch import MTCNN
    mtcnn = MTCNN(
        image_size=224, margin=40, min_face_size=min_face_size,
        thresholds=[0.6, 0.7, 0.7], factor=0.709,
        post_process=False, device=device,
    )
    faces = []
    for frame in frames:
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil = Image.fromarray(rgb)
        face = mtcnn(pil)
        if face is not None:
            faces.append(face)
        else:
            # No face detected, use center crop
            h, w = frame.shape[:2]
            s = min(h, w) // 2
            cx, cy = w // 2, h // 2
            crop = rgb[cy-s:cy+s, cx-s:cx+s]
            crop = cv2.resize(crop, (224, 224))
            tensor = torch.tensor(crop, dtype=torch.float32).permute(2, 0, 1) / 255.0
            faces.append(tensor)
    return torch.stack(faces) if faces else None


# ---------- Method 1: EfficientNet-B7 (DFDC-style) ----------
def test_efficientnet_b7(frames_orig, frames_proc):
    """Use timm EfficientNet-B7 pretrained on ImageNet (proxy for DFDC architecture)."""
    print("\n" + "="*60)
    print("METHOD 1: EfficientNet-B7 (timm, ImageNet pretrained)")
    print("="*60)

    import timm
    from timm.data import resolve_data_config
    from timm.data.transforms_factory import create_transform

    # Load EfficientNet-B7
    model = timm.create_model('efficientnet_b7', pretrained=True)
    model = model.to(device).eval()

    config = resolve_data_config({}, model=model)
    transform = create_transform(**config)

    def get_features(frames):
        """Get penultimate layer features for frames."""
        features = []
        for frame in frames:
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pil = Image.fromarray(rgb)
            tensor = transform(pil).unsqueeze(0).to(device)
            with torch.no_grad():
                feat = model.forward_features(tensor)
                feat = model.global_pool(feat)
                features.append(feat.cpu().numpy().flatten())
        return np.array(features)

    t0 = time.time()
    feat_orig = get_features(frames_orig)
    feat_proc = get_features(frames_proc)
    elapsed = time.time() - t0

    # Compare feature distances
    dists = np.linalg.norm(feat_orig - feat_proc, axis=1)
    cos_sims = np.array([
        np.dot(feat_orig[i], feat_proc[i]) / (np.linalg.norm(feat_orig[i]) * np.linalg.norm(feat_proc[i]))
        for i in range(len(feat_orig))
    ])

    print(f"  Time: {elapsed:.1f}s")
    print(f"  Feature L2 distances (orig vs proc):")
    for i, (d, c) in enumerate(zip(dists, cos_sims)):
        print(f"    Frame {i}: L2={d:.4f}, cos_sim={c:.6f}")
    print(f"  Mean L2: {dists.mean():.4f}, Mean cos: {cos_sims.mean():.6f}")

    return {"l2_distances": dists, "cos_similarities": cos_sims}


# ---------- Method 2: Face-based analysis with MTCNN ----------
def test_face_analysis(frames_orig, frames_proc):
    """Extract faces and compare pixel-level differences."""
    print("\n" + "="*60)
    print("METHOD 2: Face Extraction + Pixel Analysis (MTCNN)")
    print("="*60)

    t0 = time.time()
    faces_orig = extract_faces(frames_orig)
    faces_proc = extract_faces(frames_proc)
    elapsed = time.time() - t0

    if faces_orig is None or faces_proc is None:
        print("  ERROR: Could not extract faces!")
        return None

    print(f"  Face extraction: {elapsed:.1f}s")
    print(f"  Faces extracted: {len(faces_orig)} orig, {len(faces_proc)} proc")

    # Compare face tensors
    n = min(len(faces_orig), len(faces_proc))
    for i in range(n):
        diff = (faces_orig[i] - faces_proc[i]).abs()
        l_inf = diff.max().item() * 255
        l2 = diff.norm().item()
        mean_diff = diff.mean().item() * 255
        print(f"    Face {i}: L_inf={l_inf:.1f}/255, L2={l2:.2f}, mean={mean_diff:.2f}/255")

    return {"faces_orig": faces_orig, "faces_proc": faces_proc}


# ---------- Method 3: Pixel-level analysis ----------
def test_pixel_analysis(frames_orig, frames_proc):
    """Direct pixel comparison between original and processed."""
    print("\n" + "="*60)
    print("METHOD 3: Pixel-Level Comparison")
    print("="*60)

    for i in range(len(frames_orig)):
        diff = frames_orig[i].astype(np.float32) - frames_proc[i].astype(np.float32)
        l_inf = np.abs(diff).max()
        l2 = np.linalg.norm(diff)
        mean_abs = np.abs(diff).mean()
        psnr = cv2.PSNR(frames_orig[i], frames_proc[i])

        # Frequency analysis
        gray_orig = cv2.cvtColor(frames_orig[i], cv2.COLOR_BGR2GRAY).astype(np.float32)
        gray_proc = cv2.cvtColor(frames_proc[i], cv2.COLOR_BGR2GRAY).astype(np.float32)
        fft_orig = np.fft.fft2(gray_orig)
        fft_proc = np.fft.fft2(gray_proc)
        fft_diff = np.abs(fft_orig - fft_proc)

        # High freq energy ratio (adversarial perturbations are often high-freq)
        h, w = gray_orig.shape
        mask = np.ones((h, w))
        mask[h//4:3*h//4, w//4:3*w//4] = 0  # high-freq mask
        high_freq_energy = (fft_diff * mask).sum() / (fft_diff.sum() + 1e-8)

        print(f"  Frame {i}: L_inf={l_inf:.1f}/255, L2={l2:.1f}, mean_abs={mean_abs:.2f}/255, PSNR={psnr:.1f}dB, HF_ratio={high_freq_energy:.3f}")

        # Check if perturbation is structured (adversarial) or random (recompression)
        if l_inf > 0:
            # Save difference visualization
            diff_vis = np.abs(diff).astype(np.uint8)
            diff_enhanced = cv2.normalize(diff_vis, None, 0, 255, cv2.NORM_MINMAX)
            os.makedirs("outputs/analysis", exist_ok=True)
            cv2.imwrite(f"outputs/analysis/diff_frame_{i}.png", diff_enhanced)

            # Channel analysis
            for c, name in enumerate(["B", "G", "R"]):
                ch_diff = np.abs(diff[:,:,c])
                print(f"    {name}: mean={ch_diff.mean():.2f}, max={ch_diff.max():.1f}, std={ch_diff.std():.2f}")


# ---------- Method 4: HuggingFace deepfake model ----------
def test_huggingface_deepfake(frames_orig, frames_proc):
    """Try HuggingFace deepfake detection models."""
    print("\n" + "="*60)
    print("METHOD 4: HuggingFace Deepfake Detection")
    print("="*60)

    from transformers import pipeline

    # Try a known deepfake detection model
    models_to_try = [
        "dima806/deepfake_vs_real_face_detection",
        "Organika/sdxl-detector",
    ]

    for model_name in models_to_try:
        try:
            print(f"\n  Model: {model_name}")
            clf = pipeline("image-classification", model=model_name, device=device)

            results_orig = []
            results_proc = []

            t0 = time.time()
            for i, (fo, fp) in enumerate(zip(frames_orig, frames_proc)):
                img_o = Image.fromarray(cv2.cvtColor(fo, cv2.COLOR_BGR2RGB))
                img_p = Image.fromarray(cv2.cvtColor(fp, cv2.COLOR_BGR2RGB))

                res_o = clf(img_o)
                res_p = clf(img_p)
                results_orig.append(res_o)
                results_proc.append(res_p)

                # Pretty print
                orig_str = ", ".join(f"{r['label']}={r['score']*100:.1f}%" for r in res_o)
                proc_str = ", ".join(f"{r['label']}={r['score']*100:.1f}%" for r in res_p)
                print(f"    Frame {i}:")
                print(f"      Original:  {orig_str}")
                print(f"      Processed: {proc_str}")

            elapsed = time.time() - t0
            print(f"    Time: {elapsed:.1f}s")

        except Exception as e:
            print(f"    ERROR: {e}")


# ---------- Method 5: CLIP-based analysis ----------
def test_clip_analysis(frames_orig, frames_proc):
    """CLIP-based analysis with deepfake-specific labels."""
    print("\n" + "="*60)
    print("METHOD 5: CLIP Analysis (deepfake labels)")
    print("="*60)

    from backend.pipeline.classifier import classify_frames

    labels = [
        "a deepfake video", "a real authentic video",
        "AI generated content", "natural camera footage",
        "face manipulation", "unedited photo",
        "synthetic media", "genuine recording",
        "adult content", "nudity", "safe content",
    ]

    t0 = time.time()
    cls_orig = classify_frames(frames_orig[:3], labels)
    cls_proc = classify_frames(frames_proc[:3], labels)
    elapsed = time.time() - t0

    print(f"  Time: {elapsed:.1f}s")
    for i in range(min(3, len(cls_orig))):
        print(f"\n  Frame {i}:")
        print(f"    YOLO: orig={len(cls_orig[i]['yolo_detections'])} det, proc={len(cls_proc[i]['yolo_detections'])} det")

        orig_scores = cls_orig[i]["clip_scores"]
        proc_scores = cls_proc[i]["clip_scores"]

        # Sort by original score descending
        for label in sorted(orig_scores, key=lambda k: orig_scores[k], reverse=True):
            ov = orig_scores[label] * 100
            pv = proc_scores.get(label, 0) * 100
            delta = pv - ov
            arrow = "v" if delta < -1 else "^" if delta > 1 else "="
            print(f"      {label:30s}: {ov:5.1f}% -> {pv:5.1f}% ({delta:+.1f}) {arrow}")


# ---------- Main ----------
if __name__ == "__main__":
    # Load videos
    info_orig = get_video_info(ORIGINAL)
    info_proc = get_video_info(PROCESSED)
    print(f"Original: {info_orig['width']}x{info_orig['height']}, {info_orig['frame_count']} frames")
    print(f"Processed: {info_proc['width']}x{info_proc['height']}, {info_proc['frame_count']} frames")

    # Use fewer sample frames to speed up
    indices = [i for i in SAMPLE_FRAMES if i < min(info_orig['frame_count'], info_proc['frame_count'])]
    print(f"\nSampling frames: {indices}")

    frames_orig = read_frames_at_indices(ORIGINAL, indices)
    frames_proc = read_frames_at_indices(PROCESSED, indices)
    print(f"Loaded {len(frames_orig)} frames from each video")

    # Run all tests
    test_pixel_analysis(frames_orig, frames_proc)
    test_face_analysis(frames_orig, frames_proc)
    test_clip_analysis(frames_orig, frames_proc)
    test_efficientnet_b7(frames_orig, frames_proc)
    test_huggingface_deepfake(frames_orig, frames_proc)
