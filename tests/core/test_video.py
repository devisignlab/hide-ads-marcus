import numpy as np
import os
import tempfile
from backend.core.video import extract_frames, get_video_info, extract_audio, reconstruct_video

FIXTURE = "tests/fixtures/test_2s.mp4"


class TestExtractFrames:
    def test_yields_numpy_arrays(self):
        frames = list(extract_frames(FIXTURE))
        assert len(frames) == 20
        assert isinstance(frames[0], np.ndarray)
        assert frames[0].shape == (240, 320, 3)

    def test_generator_is_lazy(self):
        gen = extract_frames(FIXTURE)
        first = next(gen)
        assert isinstance(first, np.ndarray)

    def test_frames_are_rgb(self):
        frame = next(extract_frames(FIXTURE))
        assert frame.dtype == np.uint8


class TestVideoInfo:
    def test_returns_metadata(self):
        info = get_video_info(FIXTURE)
        assert info["fps"] == 10
        assert info["frame_count"] == 20
        assert info["width"] == 320
        assert info["height"] == 240
        assert info["duration_s"] == 2.0


class TestExtractAudio:
    def test_extract_audio_returns_path_or_none(self):
        result = extract_audio(FIXTURE, "/tmp/test_audio.aac")
        assert result is None


class TestReconstructVideo:
    def test_reconstruct_creates_mp4(self):
        frames = list(extract_frames(FIXTURE))
        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as f:
            output_path = f.name
        reconstruct_video(frames, output_path, fps=10)
        info = get_video_info(output_path)
        assert info["frame_count"] == 20
        assert info["width"] == 320
        assert info["height"] == 240
        os.unlink(output_path)

    def test_reconstruct_with_no_audio(self):
        frames = list(extract_frames(FIXTURE))
        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as f:
            output_path = f.name
        reconstruct_video(frames, output_path, fps=10, audio_path=None)
        assert os.path.exists(output_path)
        assert os.path.getsize(output_path) > 0
        os.unlink(output_path)
