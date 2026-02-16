import os
import subprocess
import cv2
import numpy as np
from collections.abc import Generator


def get_video_info(path: str) -> dict:
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise FileNotFoundError(f"Cannot open video: {path}")
    try:
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        return {
            "fps": fps,
            "frame_count": frame_count,
            "width": width,
            "height": height,
            "duration_s": frame_count / fps if fps > 0 else 0,
        }
    finally:
        cap.release()


def extract_frames(path: str) -> Generator[np.ndarray, None, None]:
    """Lazy generator — yields one RGB frame at a time. Never loads all into RAM."""
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise FileNotFoundError(f"Cannot open video: {path}")
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            yield cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    finally:
        cap.release()


def read_frames_at_indices(path: str, indices: list[int]) -> list[np.ndarray]:
    """Read only specific frames by index using OpenCV seek. Memory: O(len(indices))."""
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise FileNotFoundError(f"Cannot open video: {path}")
    try:
        frames = []
        sorted_indices = sorted(set(indices))
        for idx in sorted_indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()
            if ret:
                frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            else:
                # If seek fails, use a black frame as placeholder
                w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                frames.append(np.zeros((h, w, 3), dtype=np.uint8))
        return frames
    finally:
        cap.release()


def open_video_writer(path: str, fps: float, width: int, height: int) -> cv2.VideoWriter:
    """Open a VideoWriter for incremental frame writing."""
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    return cv2.VideoWriter(path, fourcc, fps, (width, height))


def write_frame(writer: cv2.VideoWriter, frame: np.ndarray) -> None:
    """Write a single RGB frame to the VideoWriter."""
    writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))


def extract_audio(video_path: str, output_path: str) -> str | None:
    try:
        result = subprocess.run(
            ["ffmpeg", "-y", "-i", video_path, "-vn", "-acodec", "copy", output_path],
            capture_output=True,
            timeout=60,
        )
        if result.returncode != 0:
            return None
        if os.path.exists(output_path) and os.path.getsize(output_path) > 0:
            return output_path
        return None
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return None


def reconstruct_video(
    frames: list[np.ndarray],
    output_path: str,
    fps: float,
    audio_path: str | None = None,
) -> str:
    """Legacy: reconstruct from frame list. Used by tests."""
    if not frames:
        raise ValueError("No frames to reconstruct")
    h, w = frames[0].shape[:2]
    temp_path = output_path + ".tmp.mp4" if audio_path else output_path
    writer = open_video_writer(temp_path, fps, w, h)
    try:
        for frame in frames:
            write_frame(writer, frame)
    finally:
        writer.release()
    if audio_path:
        _mux_audio(temp_path, audio_path, output_path)
        os.unlink(temp_path)
    return output_path


def mux_audio(video_path: str, audio_path: str, output_path: str) -> None:
    """Mux audio track into video file using ffmpeg."""
    _mux_audio(video_path, audio_path, output_path)


def _mux_audio(video_path: str, audio_path: str, output_path: str) -> None:
    subprocess.run(
        [
            "ffmpeg", "-y",
            "-i", video_path,
            "-i", audio_path,
            "-c:v", "copy",
            "-c:a", "copy",
            "-shortest",
            output_path,
        ],
        capture_output=True,
        timeout=120,
        check=True,
    )
