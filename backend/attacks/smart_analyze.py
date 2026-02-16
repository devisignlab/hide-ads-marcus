# backend/attacks/smart_analyze.py
"""Smart content analyzer - detects WHY content would be flagged.

Runs multiple classifiers and returns a risk profile with specific
flags and recommended attacks for each risk.
"""

import logging
import numpy as np
import torch
import cv2
from PIL import Image

from backend.core.device import get_device

logger = logging.getLogger(__name__)

# CLIP labels for content classification
NSFW_CATEGORIES = {
    "nudity": "explicit nudity, naked body, nude person, genitalia",
    "sexual": "sexually explicit content, pornography, sex act",
    "sensual": "sensual content, seductive poses, provocative imagery, erotic",
    "lingerie": "bikini model, swimwear, lingerie, revealing clothing, underwear",
    "intimacy": "romantic intimate scene, kissing, physical intimacy, making out",
    "violence": "violence, blood, gore, fighting, assault, injury",
    "weapons": "guns, weapons, knife, firearm, ammunition",
    "drugs": "drugs, marijuana, cocaine, pills, drug use, smoking",
    "hate": "hate symbol, nazi, extremist, racist imagery",
    "child_safety": "child exploitation, underage, minor in danger",
    "safe": "a safe family friendly product advertisement, professional clean content",
    "news": "professional news broadcast, educational documentary content",
    "cartoon": "cartoon animation, illustrated art, kids content",
}

# Thresholds for flagging
FLAG_THRESHOLDS = {
    "nudity": 0.10,
    "sexual": 0.10,
    "sensual": 0.08,
    "lingerie": 0.08,
    "intimacy": 0.08,
    "violence": 0.10,
    "weapons": 0.10,
    "drugs": 0.08,
    "hate": 0.05,
    "child_safety": 0.05,
}

# Which attack to use for each flag
ATTACK_MAP = {
    "nudity": "clip",
    "sexual": "clip",
    "sensual": "clip",
    "lingerie": "clip",
    "intimacy": "clip",
    "violence": "clip",
    "weapons": "clip",
    "drugs": "clip",
    "hate": "clip",
    "child_safety": "clip",
    "deepfake": "anti_deepfake",
}

# Lazy-loaded CLIP
_clip_model = None
_clip_preprocess = None


def _load_clip():
    global _clip_model, _clip_preprocess
    if _clip_model is not None:
        return
    import clip
    device = get_device()
    _clip_model, _clip_preprocess = clip.load("ViT-B/32", device=device)
    logger.info("CLIP loaded for smart analysis")


def analyze_frame(frame_bgr: np.ndarray) -> dict:
    """Analyze a single frame and return risk profile.

    Returns:
        {
            "scores": {"nudity": 0.45, "sexual": 0.12, ...},
            "flags": ["nudity", "sexual", "sensual"],
            "deepfake_score": 0.75,
            "risk_level": "high",  # low/medium/high/critical
            "recommended_attacks": ["clip", "anti_deepfake"],
            "repel_texts": ["explicit nudity...", "sexually explicit..."],
        }
    """
    _load_clip()
    device = get_device()
    import clip

    # CLIP classification
    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    pil = Image.fromarray(rgb)
    image = _clip_preprocess(pil).unsqueeze(0).to(device)

    labels = list(NSFW_CATEGORIES.values())
    keys = list(NSFW_CATEGORIES.keys())
    text = clip.tokenize(labels).to(device)

    with torch.no_grad():
        image_features = _clip_model.encode_image(image)
        text_features = _clip_model.encode_text(text)
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        similarity = (100.0 * image_features @ text_features.T).softmax(dim=-1)

    scores = {keys[i]: similarity[0][i].item() for i in range(len(keys))}

    # Deepfake detection
    from backend.attacks.anti_deepfake import classify_deepfake
    deepfake_result = classify_deepfake(frame_bgr)
    deepfake_score = deepfake_result.get("Deepfake", 0.0)

    # Determine flags
    flags = []
    for category, threshold in FLAG_THRESHOLDS.items():
        if scores.get(category, 0) > threshold:
            flags.append(category)

    if deepfake_score > 0.30:
        flags.append("deepfake")

    # Determine risk level
    max_nsfw = max(scores.get(k, 0) for k in FLAG_THRESHOLDS)
    if deepfake_score > 0.7 or max_nsfw > 0.4:
        risk_level = "critical"
    elif deepfake_score > 0.5 or max_nsfw > 0.2:
        risk_level = "high"
    elif deepfake_score > 0.3 or max_nsfw > 0.1:
        risk_level = "medium"
    else:
        risk_level = "low"

    # Recommended attacks
    attacks = set()
    repel_texts = []
    for flag in flags:
        attack = ATTACK_MAP.get(flag, "clip")
        attacks.add(attack)
        if flag in NSFW_CATEGORIES and flag not in ("safe", "news", "cartoon"):
            repel_texts.append(NSFW_CATEGORIES[flag])

    if not attacks and risk_level != "low":
        attacks.add("clip")

    return {
        "scores": scores,
        "flags": flags,
        "deepfake_score": deepfake_score,
        "risk_level": risk_level,
        "recommended_attacks": sorted(attacks),
        "repel_texts": repel_texts,
    }


def analyze_video_sample(video_path: str, num_samples: int = 4) -> dict:
    """Analyze a video by sampling frames across its duration.

    Returns aggregated risk profile across all sampled frames.
    """
    from backend.core.video import read_frames_at_indices, get_video_info

    info = get_video_info(video_path)
    total = info["frame_count"]

    # Sample evenly across the video
    indices = [int(i * total / (num_samples + 1)) for i in range(1, num_samples + 1)]
    indices = [min(i, total - 1) for i in indices]

    frames = read_frames_at_indices(video_path, indices)

    all_flags = set()
    all_attacks = set()
    all_repel_texts = set()
    max_deepfake = 0.0
    frame_results = []

    for i, frame in enumerate(frames):
        result = analyze_frame(frame)
        frame_results.append({"frame_idx": indices[i], **result})

        all_flags.update(result["flags"])
        all_attacks.update(result["recommended_attacks"])
        all_repel_texts.update(result["repel_texts"])
        max_deepfake = max(max_deepfake, result["deepfake_score"])

    # Aggregate risk level
    if max_deepfake > 0.7 or any(r["risk_level"] == "critical" for r in frame_results):
        risk_level = "critical"
    elif max_deepfake > 0.5 or any(r["risk_level"] == "high" for r in frame_results):
        risk_level = "high"
    elif any(r["risk_level"] == "medium" for r in frame_results):
        risk_level = "medium"
    else:
        risk_level = "low"

    return {
        "video_path": video_path,
        "total_frames": total,
        "sampled_frames": len(frames),
        "flags": sorted(all_flags),
        "risk_level": risk_level,
        "max_deepfake_score": max_deepfake,
        "recommended_attacks": sorted(all_attacks),
        "repel_texts": sorted(all_repel_texts),
        "frame_results": frame_results,
    }
