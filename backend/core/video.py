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


def extract_audio(video_path: str, output_path: str) -> str | None:
    try:
        result = subprocess.run(
            ["ffmpeg", "-y", "-i", video_path, "-vn", "-acodec", "copy", output_path],
            capture_output=True,
            timeout=60,
        )
        if result.returncode != 0:
            return None
        import os

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
    if not frames:
        raise ValueError("No frames to reconstruct")
    h, w = frames[0].shape[:2]
    temp_path = output_path + ".tmp.mp4" if audio_path else output_path
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(temp_path, fourcc, fps, (w, h))
    try:
        for frame in frames:
            writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
    finally:
        writer.release()
    if audio_path:
        _mux_audio(temp_path, audio_path, output_path)
        import os

        os.unlink(temp_path)
    return output_path


def _mux_audio(video_path: str, audio_path: str, output_path: str) -> None:
    subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-i",
            video_path,
            "-i",
            audio_path,
            "-c:v",
            "copy",
            "-c:a",
            "copy",
            "-shortest",
            output_path,
        ],
        capture_output=True,
        timeout=120,
        check=True,
    )
