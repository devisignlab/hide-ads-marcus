import os
import logging
import subprocess
import cv2
import numpy as np
from collections.abc import Generator

logger = logging.getLogger(__name__)


def get_video_info(path: str) -> dict:
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise FileNotFoundError(f"Cannot open video: {path}")
    try:
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        info = {
            "fps": fps,
            "frame_count": frame_count,
            "width": width,
            "height": height,
            "duration_s": frame_count / fps if fps > 0 else 0,
            "sar": None,
        }
    finally:
        cap.release()

    # Get SAR via ffprobe (OpenCV doesn't expose it)
    try:
        result = subprocess.run(
            ["ffprobe", "-v", "quiet", "-print_format", "json",
             "-show_streams", path],
            capture_output=True, text=True, timeout=10,
        )
        if result.returncode == 0:
            import json
            data = json.loads(result.stdout)
            for stream in data.get("streams", []):
                if stream.get("codec_type") == "video":
                    sar = stream.get("sample_aspect_ratio", "1:1")
                    if sar and sar != "1:1" and sar != "0:1":
                        info["sar"] = sar
                        logger.info(f"Video SAR: {sar}")
                    break
    except Exception:
        pass

    return info


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


class FFmpegWriter:
    """Write video frames via pipe to ffmpeg — encodes directly to H.264.

    Replaces OpenCV VideoWriter which used MPEG4 (bad quality, large files).
    Frames are piped as raw RGB24 → ffmpeg encodes to H.264 with CRF control.
    """

    def __init__(self, path: str, fps: float, width: int, height: int,
                 crf: int = 18, preset: str = "medium", sar: str | None = None):
        self.width = width
        self.height = height
        self.path = path
        cmd = [
            "ffmpeg", "-y",
            "-f", "rawvideo",
            "-vcodec", "rawvideo",
            "-s", f"{width}x{height}",
            "-pix_fmt", "rgb24",
            "-r", str(fps),
            "-i", "-",
            "-c:v", "libx264",
            "-crf", str(crf),
            "-preset", preset,
            "-pix_fmt", "yuv420p",
        ]
        if sar:
            # Convert "9:16" to "9/16" for ffmpeg filter syntax
            sar_filter = sar.replace(":", "/")
            cmd.extend(["-vf", f"setsar={sar_filter}"])
        cmd.extend(["-movflags", "+faststart", path])
        logger.info(f"FFmpegWriter: {' '.join(cmd)}")
        self.proc = subprocess.Popen(
            cmd, stdin=subprocess.PIPE, stderr=subprocess.PIPE,
        )
        self.frames_written = 0

    def write(self, frame: np.ndarray) -> None:
        """Write a single RGB uint8 frame."""
        if frame.shape[1] != self.width or frame.shape[0] != self.height:
            frame = cv2.resize(frame, (self.width, self.height))
        self.proc.stdin.write(frame.tobytes())
        self.frames_written += 1

    def release(self) -> None:
        """Close pipe and wait for ffmpeg to finish."""
        self.proc.stdin.close()
        self.proc.wait()
        if self.proc.returncode != 0:
            stderr = self.proc.stderr.read().decode()
            logger.error(f"FFmpegWriter error: {stderr[:500]}")
        else:
            logger.info(f"FFmpegWriter: {self.frames_written} frames → {self.path}")


def open_video_writer(path: str, fps: float, width: int, height: int,
                      sar: str | None = None) -> FFmpegWriter:
    """Open an FFmpeg pipe writer for H.264 encoding. Drop-in replacement for OpenCV."""
    return FFmpegWriter(path, fps, width, height, crf=28, preset="medium", sar=sar)


def write_frame(writer: FFmpegWriter, frame: np.ndarray) -> None:
    """Write a single RGB frame. Works with FFmpegWriter (no BGR conversion needed)."""
    writer.write(frame)


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
