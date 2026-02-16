"""Streaming pipeline orchestrator — processes video without loading all frames into RAM.

Memory usage: O(batch_size) instead of O(total_frames).
For a 1080p video: ~200-400MB instead of 10-20GB.
"""

import os
import logging
from dataclasses import dataclass, field
from collections.abc import Callable

import cv2
import numpy as np
import torch

from backend.config import get_preset, DEFAULT_EPSILON, DEFAULT_ALPHA
from backend.core.video import (
    get_video_info,
    extract_audio,
    read_frames_at_indices,
    open_video_writer,
    write_frame,
    extract_frames,
    mux_audio,
)
from backend.core.batch import iter_batches
from backend.pipeline.classifier import classify_frames
from backend.attacks.adversarial import (
    attack_yolo_fgsm, attack_yolo_pgd,
    attack_clip_fgsm, attack_clip_pgd,
)
from backend.attacks.combined import attack_combined_pgd
from backend.attacks.lsb import lsb_cloak, lsb_embed
from backend.attacks.uap import compute_uap, apply_uap
from backend.attacks.turbo import fgsm_combined_single, compute_clip_uap, apply_uap_to_frame
from backend.attacks.anti_deepfake import attack_deepfake_pgd
from backend.attacks.remux import full_adversarial_remux
from backend.attacks.temporal import select_keyframe_indices

logger = logging.getLogger(__name__)

# How many frames to sample for before/after classification
CLASSIFICATION_SAMPLES = 5


@dataclass
class PipelineConfig:
    video_path: str
    attack_method: str  # "fgsm", "pgd", "lsb", "uap", "combined"
    target_text: str = "an empty room"
    repel_text: str = ""  # CLIP repulsion target (e.g. "explicit adult content, nudity")
    repel_weight: float = 0.3  # Weight for CLIP repulsion within clip_weight
    clip_labels: list[str] = field(default_factory=lambda: [
        "a person talking", "an advertisement", "a product",
        "an empty room", "nature scenery", "abstract art",
    ])
    preset: str = "preview"
    epsilon: float = DEFAULT_EPSILON
    alpha: float = DEFAULT_ALPHA
    yolo_weight: float = 0.5
    clip_weight: float = 0.5
    lsb_mode: str = "cloak"
    lsb_intensity: float = 0.5
    output_dir: str = "outputs"


ProgressCallback = Callable[[int, str], None]


def run_pipeline(
    config: PipelineConfig,
    on_progress: ProgressCallback | None = None,
) -> dict:
    """Execute the streaming 5-stage pipeline. Never loads all frames into RAM."""

    def progress(pct: int, stage: str):
        if on_progress:
            on_progress(pct, stage)

    preset = get_preset(config.preset)
    os.makedirs(config.output_dir, exist_ok=True)

    # === Stage 1: Video Info + Audio (0-5%) ===
    progress(0, "extracting")
    logger.info("Stage 1: Getting video info")
    info = get_video_info(config.video_path)
    total_frames = info["frame_count"]
    logger.info(f"Video: {total_frames} frames, {info['width']}x{info['height']}, {info['fps']}fps")

    audio_path = extract_audio(
        config.video_path,
        os.path.join(config.output_dir, "temp_audio.aac"),
    )
    progress(5, "extracting")

    # === Stage 2: Sample Classification (5-20%) ===
    progress(5, "classifying_original")
    logger.info("Stage 2: Classifying sample frames")
    sample_indices = _pick_sample_indices(total_frames, CLASSIFICATION_SAMPLES)
    sample_frames = read_frames_at_indices(config.video_path, sample_indices)
    original_cls = classify_frames(sample_frames, config.clip_labels)
    del sample_frames  # free memory
    progress(20, "classifying_original")
    _clear_gpu()

    # === Stage 3: Streaming Perturbation + Write (20-85%) ===
    progress(20, "perturbing")
    logger.info(f"Stage 3: Streaming {config.attack_method} attack")
    raw_output = os.path.join(config.output_dir, "output_raw.mp4")

    if config.attack_method == "lsb":
        frames_written = _stream_lsb(config, info, raw_output, progress)
    elif config.attack_method == "uap":
        frames_written = _stream_uap(config, info, preset, raw_output, progress)
    elif config.attack_method == "turbo":
        frames_written = _stream_turbo(config, info, raw_output, progress)
    elif config.attack_method == "anti_deepfake":
        frames_written = _stream_anti_deepfake(config, info, raw_output, progress)
    elif config.attack_method == "auto":
        frames_written = _stream_auto(config, info, raw_output, progress)
    else:
        frames_written = _stream_gradient_attack(config, info, preset, raw_output, progress)

    progress(85, "perturbing")
    _clear_gpu()

    # === Stage 4: Sample Verification (85-92%) ===
    progress(85, "verifying")
    logger.info("Stage 4: Verifying perturbed samples")
    perturbed_samples = read_frames_at_indices(raw_output, sample_indices)
    perturbed_cls = classify_frames(perturbed_samples, config.clip_labels)
    del perturbed_samples
    progress(92, "verifying")
    _clear_gpu()

    # === Stage 5: Mux Audio + Remux (92-100%) ===
    progress(92, "reconstructing")
    logger.info("Stage 5: Muxing audio")
    muxed_output = os.path.join(config.output_dir, "output_muxed.mp4")
    output_path = os.path.join(config.output_dir, "output.mp4")
    if audio_path:
        mux_audio(raw_output, audio_path, muxed_output)
        os.unlink(raw_output)
    else:
        os.rename(raw_output, muxed_output)

    # Anti-deepfake: also do adversarial remux (strip metadata, re-encode audio)
    if config.attack_method in ("anti_deepfake", "auto"):
        logger.info("Stage 5b: Adversarial remux (hash bypass)")
        full_adversarial_remux(muxed_output, output_path)
        os.unlink(muxed_output)
    else:
        os.rename(muxed_output, output_path)

    progress(100, "done")
    logger.info(f"Done! {frames_written} frames written to {output_path}")

    return {
        "output_path": output_path,
        "video_info": info,
        "original_classifications": original_cls,
        "perturbed_classifications": perturbed_cls,
        "frames_processed": frames_written,
    }


# ---------------------------------------------------------------------------
# Streaming attack implementations
# ---------------------------------------------------------------------------

def _stream_lsb(config, info, output_path, progress):
    """LSB: pure streaming, 1 frame at a time. RAM: ~10MB."""
    total = info["frame_count"]
    writer = open_video_writer(output_path, info["fps"], info["width"], info["height"], sar=info.get("sar"))
    try:
        for i, frame in enumerate(extract_frames(config.video_path)):
            if config.lsb_mode == "cloak":
                perturbed = lsb_cloak([frame], intensity=config.lsb_intensity)[0]
            else:
                perturbed = lsb_embed([frame], seed=42 + i)[0]
            write_frame(writer, perturbed)
            if i % 100 == 0:
                pct = 20 + int(65 * i / max(total, 1))
                progress(pct, "perturbing")
    finally:
        writer.release()
    return total


def _stream_uap(config, info, preset, output_path, progress):
    """UAP: compute perturbation from samples, then stream-apply. RAM: ~200MB."""
    total = info["frame_count"]

    # Read 10 sample frames for UAP computation
    sample_indices = _pick_sample_indices(total, min(10, total))
    samples = read_frames_at_indices(config.video_path, sample_indices)
    logger.info(f"Computing UAP from {len(samples)} sample frames")

    uap = compute_uap(
        samples, target="yolo", target_text=config.target_text,
        epsilon=config.epsilon, steps=preset.pgd_steps or 10,
    )
    del samples
    _clear_gpu()
    progress(40, "perturbing")

    # Stream apply UAP
    writer = open_video_writer(output_path, info["fps"], info["width"], info["height"], sar=info.get("sar"))
    try:
        for i, frame in enumerate(extract_frames(config.video_path)):
            perturbed = apply_uap([frame], uap)[0]
            write_frame(writer, perturbed)
            if i % 100 == 0:
                pct = 40 + int(45 * i / max(total, 1))
                progress(pct, "perturbing")
    finally:
        writer.release()
    return total


def _stream_turbo(config, info, output_path, progress):
    """Turbo: Mini-PGD softmax attack on keyframes + interpolation.

    Phase A: Attack keyframes with 5-step mini-PGD (~15s each), store deltas.
    Phase B: Stream video, apply deltas with interpolation.
    """
    total = info["frame_count"]
    repel_texts = [t.strip() for t in config.repel_text.split(",") if t.strip()] if config.repel_text else [
        "adult content", "nudity", "sexually explicit",
        "naked body", "pornography", "nsfw",
    ]

    # Phase A: Attack keyframes (20-60%)
    keyframe_interval = 5
    key_indices = select_keyframe_indices(total, keyframe_interval)
    logger.info(f"Turbo Phase A: Attacking {len(key_indices)} keyframes (interval={keyframe_interval})")

    deltas: dict[int, np.ndarray] = {}
    for ki, idx in enumerate(key_indices):
        frame = read_frames_at_indices(config.video_path, [idx])[0]
        perturbed = fgsm_combined_single(
            frame,
            target_text=config.target_text,
            repel_texts=repel_texts,
            epsilon=config.epsilon,
            steps=5,
        )
        delta = perturbed.astype(np.int16) - frame.astype(np.int16)
        deltas[idx] = np.clip(delta, -127, 127).astype(np.int8)

        _clear_gpu()
        pct = 20 + int(40 * (ki + 1) / len(key_indices))
        progress(pct, "perturbing")

        if ki % 10 == 0:
            logger.info(f"  Keyframe {ki+1}/{len(key_indices)} (frame {idx})")

    # Phase B: Stream video, apply interpolated deltas (60-85%)
    logger.info("Turbo Phase B: Streaming with interpolation")
    sorted_keys = sorted(deltas.keys())
    writer = open_video_writer(output_path, info["fps"], info["width"], info["height"], sar=info.get("sar"))
    try:
        for frame_idx, frame in enumerate(extract_frames(config.video_path)):
            if frame_idx in deltas:
                result = np.clip(
                    frame.astype(np.int16) + deltas[frame_idx].astype(np.int16),
                    0, 255,
                ).astype(np.uint8)
            else:
                result = _interpolate_frame(frame, frame_idx, sorted_keys, deltas)

            write_frame(writer, result)
            if frame_idx % 100 == 0:
                pct = 60 + int(25 * frame_idx / max(total, 1))
                progress(pct, "perturbing")
    finally:
        writer.release()

    del deltas
    return total


def _stream_anti_deepfake(config, info, output_path, progress):
    """Anti-deepfake attack: PGD against ViT deepfake detector + remux.

    Phase A: Attack keyframes with PGD against deepfake detector.
    Phase B: Stream video, apply deltas with interpolation.
    Phase C: Remux to change file hash.
    """
    total = info["frame_count"]

    # Attack keyframes (5 steps is enough to drop from 75% to 5%)
    keyframe_interval = 10
    key_indices = select_keyframe_indices(total, keyframe_interval)
    logger.info(f"Anti-deepfake: Attacking {len(key_indices)} keyframes")

    deltas: dict[int, np.ndarray] = {}
    for ki, idx in enumerate(key_indices):
        frame = read_frames_at_indices(config.video_path, [idx])[0]
        perturbed = attack_deepfake_pgd(
            frame, epsilon=config.epsilon, steps=5, alpha=config.epsilon / 3,
        )
        delta = perturbed.astype(np.int16) - frame.astype(np.int16)
        deltas[idx] = np.clip(delta, -127, 127).astype(np.int8)

        _clear_gpu()
        pct = 20 + int(40 * (ki + 1) / len(key_indices))
        progress(pct, "perturbing")

        if ki % 20 == 0:
            logger.info(f"  Keyframe {ki+1}/{len(key_indices)} (frame {idx})")

    # Stream with interpolation
    logger.info("Anti-deepfake Phase B: Streaming with interpolation")
    sorted_keys = sorted(deltas.keys())
    writer = open_video_writer(output_path, info["fps"], info["width"], info["height"], sar=info.get("sar"))
    try:
        for frame_idx, frame in enumerate(extract_frames(config.video_path)):
            if frame_idx in deltas:
                result = np.clip(
                    frame.astype(np.int16) + deltas[frame_idx].astype(np.int16),
                    0, 255,
                ).astype(np.uint8)
            else:
                result = _interpolate_frame(frame, frame_idx, sorted_keys, deltas)

            write_frame(writer, result)
            if frame_idx % 100 == 0:
                pct = 60 + int(25 * frame_idx / max(total, 1))
                progress(pct, "perturbing")
    finally:
        writer.release()

    del deltas
    return total


def _stream_auto(config: PipelineConfig, info: dict, raw_output: str, progress: ProgressCallback):
    """Auto mode: Analyze → frame-by-frame PGD deepfake + CLIP UAP + Remux.

    Phase A: Smart analysis (detect deepfake, NSFW, etc.)
    Phase B: Compute CLIP UAP if NSFW detected (fast, one-shot)
    Phase C: Stream all frames — apply PGD deepfake + UAP per frame
    """
    total = info["frame_count"]
    logger.info(f"Auto mode: analyzing {total} frames")

    # Phase A: Smart analysis (20-25%)
    progress(22, "analyzing_content")
    from backend.attacks.smart_analyze import analyze_video_sample
    analysis = analyze_video_sample(config.video_path, num_samples=4)

    use_clip = "clip" in analysis["recommended_attacks"]
    use_deepfake = "anti_deepfake" in analysis["recommended_attacks"]
    repel_texts = analysis.get("repel_texts", [])
    repel_text = ", ".join(repel_texts)

    logger.info(f"Auto: flags={analysis['flags']}, risk={analysis['risk_level']}")
    logger.info(f"Auto: use_deepfake={use_deepfake}, use_clip={use_clip}")
    if repel_texts:
        logger.info(f"Auto: repel_texts={repel_texts[:3]}")

    # Phase B: Compute CLIP UAP if NSFW detected (25-45%)
    uap = None
    if use_clip and repel_text:
        progress(25, "computing_uap")
        sample_indices = [int(i * total / 7) for i in range(1, 7)]
        sample_frames = read_frames_at_indices(config.video_path, sample_indices)

        target_text = config.target_text or "a safe family friendly product advertisement, professional clean content"
        uap = compute_clip_uap(
            sample_frames, target_text, repel_text,
            epsilon=32 / 255, steps=80, alpha=2 / 255
        )
        _clear_gpu()
        logger.info(f"Auto: UAP computed, L_inf={np.abs(uap).max():.1f}/255")

    progress(45, "perturbing")

    # Phase C: Stream all frames — PGD deepfake + UAP per frame (45-85%)
    logger.info(f"Auto: Streaming frame-by-frame attack")
    writer = open_video_writer(raw_output, info["fps"], info["width"], info["height"], sar=info.get("sar"))

    for frame_idx, frame in enumerate(extract_frames(config.video_path)):
        out = frame

        # PGD against deepfake detector (per frame, ~6% score)
        if use_deepfake:
            out = attack_deepfake_pgd(
                out,
                epsilon=config.epsilon if config.epsilon > 4/255 else 16/255,
                steps=10, alpha=4/255,
                use_eot=False, use_hash=False,
            )

        # Apply CLIP UAP for NSFW bypass (instant, <1ms per frame)
        if uap is not None:
            out = apply_uap_to_frame(out, uap)

        write_frame(writer, out)

        if (frame_idx + 1) % 100 == 0:
            pct = 45 + int(40 * (frame_idx + 1) / total)
            progress(min(pct, 84), "perturbing")
            logger.info(f"  Auto: {frame_idx+1}/{total} ({100*(frame_idx+1)/total:.0f}%)")

    writer.release()
    return total


def _stream_gradient_attack(config, info, preset, output_path, progress):
    """Gradient attacks (FGSM/PGD/Combined) with keyframe propagation.

    Phase A: Read & perturb keyframes in small batches → store deltas (int8).
    Phase B: Stream through video, interpolate deltas, write.

    RAM for deltas: ~5MB per keyframe at 1080p (int8).
    Preview preset (interval=15) on 3600 frames = 241 keyframes ≈ 1.2GB.
    """
    total = info["frame_count"]
    method = config.attack_method

    # Compute keyframe indices
    key_indices = select_keyframe_indices(total, preset.keyframe_interval)
    logger.info(f"Keyframes: {len(key_indices)} of {total} (interval={preset.keyframe_interval})")

    # Enforce minimum PGD steps for iterative methods
    steps = preset.pgd_steps
    if method in ("pgd", "combined") and steps < 3:
        steps = 5
        logger.info(f"Enforcing minimum {steps} PGD steps for {method}")

    # Phase A: Perturb keyframes in batches, store deltas
    deltas: dict[int, np.ndarray] = {}  # idx -> int8 delta
    batches = list(iter_batches(key_indices, preset.batch_size))

    for batch_num, batch_indices in enumerate(batches):
        batch_frames = read_frames_at_indices(config.video_path, batch_indices)

        if method == "fgsm":
            perturbed = attack_yolo_fgsm(batch_frames, config.epsilon)
        elif method == "pgd":
            perturbed = attack_yolo_pgd(
                batch_frames, config.epsilon, steps, config.alpha,
            )
        elif method == "combined":
            perturbed = attack_combined_pgd(
                batch_frames, config.target_text,
                config.yolo_weight, config.clip_weight,
                config.epsilon, steps, config.alpha,
                repel_text=config.repel_text or None,
                repel_weight=config.repel_weight,
            )
        else:
            raise ValueError(f"Unknown method: {method}")

        # Store delta as int8 (perturbation is ≤ ±4 pixels, fits in int8)
        for i, idx in enumerate(batch_indices):
            delta = perturbed[i].astype(np.int16) - batch_frames[i].astype(np.int16)
            deltas[idx] = np.clip(delta, -127, 127).astype(np.int8)

        del batch_frames, perturbed
        _clear_gpu()

        pct = 20 + int(40 * (batch_num + 1) / len(batches))
        progress(pct, "perturbing")
        logger.info(f"Perturbed batch {batch_num + 1}/{len(batches)}")

    # Phase B: Stream through video, apply interpolated deltas
    sorted_keys = sorted(deltas.keys())
    writer = open_video_writer(output_path, info["fps"], info["width"], info["height"], sar=info.get("sar"))
    try:
        for frame_idx, frame in enumerate(extract_frames(config.video_path)):
            if frame_idx in deltas:
                # Keyframe: apply delta directly
                result = np.clip(
                    frame.astype(np.int16) + deltas[frame_idx].astype(np.int16),
                    0, 255,
                ).astype(np.uint8)
            else:
                # Interpolate between nearest keyframes
                result = _interpolate_frame(frame, frame_idx, sorted_keys, deltas)

            write_frame(writer, result)

            if frame_idx % 100 == 0:
                pct = 60 + int(25 * frame_idx / max(total, 1))
                progress(pct, "perturbing")
    finally:
        writer.release()

    del deltas
    return total


def _interpolate_frame(
    frame: np.ndarray,
    frame_idx: int,
    sorted_keys: list[int],
    deltas: dict[int, np.ndarray],
) -> np.ndarray:
    """Linearly interpolate perturbation delta between surrounding keyframes."""
    # Find surrounding keyframes
    prev_key = sorted_keys[0]
    next_key = sorted_keys[-1]
    for k in sorted_keys:
        if k <= frame_idx:
            prev_key = k
        if k >= frame_idx:
            next_key = k
            break

    if prev_key == next_key:
        delta = deltas[prev_key].astype(np.float32)
    else:
        t = (frame_idx - prev_key) / (next_key - prev_key)
        delta = (
            (1 - t) * deltas[prev_key].astype(np.float32)
            + t * deltas[next_key].astype(np.float32)
        )

    return np.clip(frame.astype(np.float32) + delta, 0, 255).astype(np.uint8)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _pick_sample_indices(total: int, count: int) -> list[int]:
    """Pick evenly spaced frame indices for sampling."""
    if total <= count:
        return list(range(total))
    step = total / count
    return [int(i * step) for i in range(count)]


def _clear_gpu():
    """Free GPU cache if available."""
    if torch.backends.mps.is_available():
        torch.mps.empty_cache()
    elif torch.cuda.is_available():
        torch.cuda.empty_cache()
