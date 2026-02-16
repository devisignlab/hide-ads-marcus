import os
import tempfile
from backend.storage.manager import StorageManager


class TestStorageManager:
    def test_save_upload(self, tmp_path):
        sm = StorageManager(upload_dir=str(tmp_path / "uploads"), output_dir=str(tmp_path / "outputs"))
        content = b"fake video data"
        video_id = sm.save_upload("test.mp4", content)
        assert os.path.exists(sm.get_upload_path(video_id))

    def test_get_output_dir_creates_dir(self, tmp_path):
        sm = StorageManager(upload_dir=str(tmp_path / "uploads"), output_dir=str(tmp_path / "outputs"))
        job_dir = sm.get_output_dir("job_123")
        assert os.path.isdir(job_dir)
