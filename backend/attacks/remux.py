# backend/attacks/remux.py
"""Video remux/re-encode attack — bypasses hash-based content moderation.

Techniques:
1. Re-mux: Copy video stream, change container (changes file hash)
2. Re-encode audio: Different bitrate changes combined fingerprint
3. Strip metadata: Remove identifying tags (creation_time, encoder info)
4. Optional: Re-encode video at same quality (changes bitstream hash)

This replicates what the competition's "processing" does.
"""

import os
import logging
import subprocess
import tempfile

logger = logging.getLogger(__name__)


def remux_video(
    input_path: str,
    output_path: str,
    strip_metadata: bool = True,
    reencode_audio: bool = True,
    audio_bitrate: str = "128k",
    reencode_video: bool = False,
    video_crf: int = 18,
) -> dict:
    """Re-mux a video to change its file hash without changing visual content.

    Args:
        input_path: Path to input video
        output_path: Path for output video
        strip_metadata: Remove all metadata tags
        reencode_audio: Re-encode audio at different bitrate
        audio_bitrate: Target audio bitrate (e.g., "128k")
        reencode_video: Re-encode video (slower but changes bitstream)
        video_crf: CRF quality for video re-encode (lower = better, 18 = visually lossless)

    Returns:
        dict with processing info
    """
    cmd = ["ffmpeg", "-y", "-i", input_path]

    # Video codec
    if reencode_video:
        cmd.extend(["-c:v", "libx265", "-crf", str(video_crf), "-preset", "medium"])
        logger.info(f"Re-encoding video with CRF={video_crf}")
    else:
        cmd.extend(["-c:v", "copy"])
        logger.info("Copying video stream (no re-encode)")

    # Audio codec
    if reencode_audio:
        cmd.extend(["-c:a", "aac", "-b:a", audio_bitrate])
        logger.info(f"Re-encoding audio at {audio_bitrate}")
    else:
        cmd.extend(["-c:a", "copy"])

    # Strip metadata
    if strip_metadata:
        cmd.extend(["-map_metadata", "-1"])
        logger.info("Stripping metadata")

    cmd.append(output_path)

    logger.info(f"Running: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)

    if result.returncode != 0:
        logger.error(f"FFmpeg error: {result.stderr}")
        raise RuntimeError(f"FFmpeg failed: {result.stderr[:500]}")

    # Get file info
    orig_size = os.path.getsize(input_path)
    new_size = os.path.getsize(output_path)

    info = {
        "input_path": input_path,
        "output_path": output_path,
        "original_size": orig_size,
        "processed_size": new_size,
        "size_change_pct": (new_size - orig_size) / orig_size * 100,
        "strip_metadata": strip_metadata,
        "reencode_audio": reencode_audio,
        "reencode_video": reencode_video,
    }
    logger.info(f"Done: {orig_size} -> {new_size} bytes ({info['size_change_pct']:+.1f}%)")
    return info


def full_adversarial_remux(
    input_path: str,
    output_path: str,
    strip_metadata: bool = True,
    reencode_audio: bool = True,
    audio_bitrate: str = "128k",
) -> dict:
    """Adversarial remux — copy video stream, re-encode audio, strip metadata.

    Video is already H.264 from FFmpegWriter, so we just copy it.
    Only re-encodes audio and strips metadata to change file hash.
    """
    cmd = ["ffmpeg", "-y", "-i", input_path,
           "-c:v", "copy",
           "-movflags", "+faststart"]

    if reencode_audio:
        cmd.extend(["-c:a", "aac", "-b:a", audio_bitrate])
    else:
        cmd.extend(["-c:a", "copy"])

    if strip_metadata:
        cmd.extend(["-map_metadata", "-1"])

    cmd.append(output_path)

    logger.info(f"Remux: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
    if result.returncode != 0:
        raise RuntimeError(f"FFmpeg failed: {result.stderr[:500]}")

    orig_size = os.path.getsize(input_path)
    new_size = os.path.getsize(output_path)

    logger.info(f"Remux done: {orig_size} -> {new_size} bytes ({(new_size - orig_size) / orig_size * 100:+.1f}%)")
    return {
        "input_path": input_path,
        "output_path": output_path,
        "original_size": orig_size,
        "processed_size": new_size,
        "size_change_pct": (new_size - orig_size) / orig_size * 100,
        "techniques": ["video_copy", "audio_reencode" if reencode_audio else "audio_copy",
                       "metadata_strip" if strip_metadata else "", "faststart"],
    }
