#!/usr/bin/env python3
"""Verification script — compares YOLO+CLIP detection before and after adversarial attack."""

import sys
import os
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from backend.core.video import get_video_info, read_frames_at_indices
from backend.pipeline.classifier import classify_frames
from backend.pipeline.orchestrator import PipelineConfig, run_pipeline


def print_header(text: str):
    print(f"\n{'='*60}")
    print(f"  {text}")
    print(f"{'='*60}")


def print_classification(cls: dict, label: str):
    print(f"\n  [{label}]")

    # YOLO
    dets = cls.get("yolo_detections", [])
    print(f"  YOLO detections: {len(dets)}")
    for d in dets[:10]:
        print(f"    - {d['label']}: {d['confidence']*100:.1f}%")
    if not dets:
        print(f"    (none)")

    # CLIP
    scores = cls.get("clip_scores", {})
    if scores:
        print(f"  CLIP scores:")
        sorted_scores = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        for label_name, score in sorted_scores:
            bar = "#" * int(score * 40)
            print(f"    {score*100:5.1f}% {bar} {label_name}")


def main():
    video_path = sys.argv[1] if len(sys.argv) > 1 else "tests/fixtures/test_2s.mp4"
    method = sys.argv[2] if len(sys.argv) > 2 else "fgsm"

    if not os.path.exists(video_path):
        print(f"Video not found: {video_path}")
        sys.exit(1)

    clip_labels = [
        "a person", "a car", "an advertisement", "a product",
        "an empty room", "nature scenery", "a building",
        "adult content", "violence", "safe content",
    ]

    # Video info
    info = get_video_info(video_path)
    print_header("VIDEO INFO")
    print(f"  File: {video_path}")
    print(f"  Size: {os.path.getsize(video_path) / 1024 / 1024:.1f} MB")
    print(f"  Resolution: {info['width']}x{info['height']}")
    print(f"  FPS: {info['fps']}")
    print(f"  Frames: {info['frame_count']}")
    print(f"  Duration: {info['duration_s']:.1f}s")

    # Run pipeline
    print_header(f"RUNNING ATTACK: {method.upper()}")
    output_dir = "outputs/verify_test"
    os.makedirs(output_dir, exist_ok=True)

    config = PipelineConfig(
        video_path=video_path,
        attack_method=method,
        preset="preview",
        target_text="an empty room with nothing in it",
        clip_labels=clip_labels,
        output_dir=output_dir,
    )

    start = time.time()

    def on_progress(pct, stage):
        print(f"\r  [{pct:3d}%] {stage}...", end="", flush=True)

    result = run_pipeline(config, on_progress=on_progress)
    elapsed = time.time() - start
    print(f"\n  Done in {elapsed:.1f}s")

    # Show results
    print_header("ORIGINAL (before attack)")
    for i, cls in enumerate(result["original_classifications"]):
        print_classification(cls, f"Sample frame {i+1}")

    print_header("PERTURBED (after attack)")
    for i, cls in enumerate(result["perturbed_classifications"]):
        print_classification(cls, f"Sample frame {i+1}")

    # Summary
    print_header("SUMMARY")
    orig_dets = sum(len(c.get("yolo_detections", [])) for c in result["original_classifications"])
    pert_dets = sum(len(c.get("yolo_detections", [])) for c in result["perturbed_classifications"])
    print(f"  YOLO detections:  {orig_dets} -> {pert_dets}  {'SUPPRESSED' if pert_dets < orig_dets else 'no change'}")

    # CLIP top label comparison
    for i in range(len(result["original_classifications"])):
        orig_scores = result["original_classifications"][i].get("clip_scores", {})
        pert_scores = result["perturbed_classifications"][i].get("clip_scores", {})
        if orig_scores and pert_scores:
            orig_top = max(orig_scores, key=orig_scores.get)
            pert_top = max(pert_scores, key=pert_scores.get)
            changed = "REDIRECTED" if orig_top != pert_top else "same"
            print(f"  CLIP top (frame {i+1}): '{orig_top}' -> '{pert_top}'  {changed}")

    print(f"\n  Output: {result['output_path']}")
    print(f"  Frames processed: {result['frames_processed']}")
    print()


if __name__ == "__main__":
    main()
