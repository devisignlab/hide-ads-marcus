#!/usr/bin/env python3
"""Test smart content analyzer on competition videos."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import logging
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

from backend.attacks.smart_analyze import analyze_video_sample

videos = [
    "uploads/Top-06.mp4",
    "uploads/Top-13.mp4",
]

for video in videos:
    if not os.path.exists(video):
        print(f"SKIP: {video} not found")
        continue

    print(f"\n{'='*60}")
    print(f"  SMART ANALYSIS: {os.path.basename(video)}")
    print(f"{'='*60}")

    result = analyze_video_sample(video, num_samples=4)

    print(f"\n  Total frames: {result['total_frames']}")
    print(f"  Sampled: {result['sampled_frames']} frames")
    print(f"\n  RISK LEVEL: {result['risk_level'].upper()}")
    print(f"  Max deepfake: {result['max_deepfake_score']*100:.1f}%")
    print(f"\n  FLAGS: {', '.join(result['flags']) if result['flags'] else 'none'}")
    print(f"  RECOMMENDED ATTACKS: {', '.join(result['recommended_attacks'])}")

    if result['repel_texts']:
        print(f"\n  AUTO-GENERATED REPEL TEXTS:")
        for t in result['repel_texts']:
            print(f"    - {t}")

    print(f"\n  Per-frame breakdown:")
    for fr in result['frame_results']:
        flags_str = ', '.join(fr['flags']) if fr['flags'] else 'clean'
        deepfake_str = f"deepfake={fr['deepfake_score']*100:.0f}%"

        # Top 3 CLIP scores
        top3 = sorted(fr['scores'].items(), key=lambda x: -x[1])[:3]
        clips = ' | '.join(f"{k}={v*100:.0f}%" for k, v in top3)

        print(f"    Frame {fr['frame_idx']:5d}: [{fr['risk_level']:8s}] {deepfake_str:15s} {flags_str}")
        print(f"                  CLIP: {clips}")

print(f"\n{'='*60}")
print("  ANALYSIS COMPLETE")
print(f"{'='*60}")
