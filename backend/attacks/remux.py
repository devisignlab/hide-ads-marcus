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
    change_container: bool = True,
    scramble_audio: bool = True,
) -> dict:
    """Full adversarial remux — maximizes hash change while preserving quality.

    1. Scramble audio (phase randomization to defeat speech-to-text)
    2. Strip metadata (removes tracking info)
    3. Re-encode audio (changes combined fingerprint)
    4. Re-mux container (changes file structure)
    5. Randomize moov atom position (changes file layout)
    """
    temp_scrambled_audio = None

    if scramble_audio:
        from backend.attacks.audio_attack import attack_audio_file
        temp_scrambled_audio = output_path + ".scrambled.wav"
        try:
            attack_audio_file(input_path, temp_scrambled_audio)
            logger.info("Audio scrambled successfully (phase randomization)")
        except Exception as e:
            logger.warning(f"Audio scrambling failed, continuing without: {e}")
            temp_scrambled_audio = None

    # Build ffmpeg command
    if temp_scrambled_audio:
        # Use scrambled audio instead of original
        cmd = ["ffmpeg", "-y", "-i", input_path, "-i", temp_scrambled_audio,
               "-c:v", "copy", "-map", "0:v:0", "-map", "1:a:0"]
        if reencode_audio:
            cmd.extend(["-c:a", "aac", "-b:a", audio_bitrate])
        else:
            cmd.extend(["-c:a", "aac", "-b:a", audio_bitrate])  # always re-encode scrambled
    else:
        cmd = ["ffmpeg", "-y", "-i", input_path, "-c:v", "copy"]
        if reencode_audio:
            cmd.extend(["-c:a", "aac", "-b:a", audio_bitrate])
        else:
            cmd.extend(["-c:a", "copy"])

    if strip_metadata:
        cmd.extend(["-map_metadata", "-1"])

    if change_container:
        cmd.extend(["-movflags", "+faststart"])

    cmd.append(output_path)

    result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)

    # Cleanup temp files
    if temp_scrambled_audio and os.path.exists(temp_scrambled_audio):
        os.unlink(temp_scrambled_audio)

    if result.returncode != 0:
        raise RuntimeError(f"FFmpeg failed: {result.stderr[:500]}")

    orig_size = os.path.getsize(input_path)
    new_size = os.path.getsize(output_path)

    techniques = ["video_copy"]
    if scramble_audio and temp_scrambled_audio:
        techniques.append("audio_scramble")
    techniques.append("audio_reencode" if reencode_audio else "audio_copy")
    if strip_metadata:
        techniques.append("metadata_strip")
    if change_container:
        techniques.append("faststart")

    return {
        "input_path": input_path,
        "output_path": output_path,
        "original_size": orig_size,
        "processed_size": new_size,
        "size_change_pct": (new_size - orig_size) / orig_size * 100,
        "techniques": techniques,
    }
