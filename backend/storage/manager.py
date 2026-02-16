import os
import uuid


class StorageManager:
    def __init__(self, upload_dir: str = "uploads", output_dir: str = "outputs"):
        self.upload_dir = upload_dir
        self.output_dir = output_dir
        os.makedirs(upload_dir, exist_ok=True)
        os.makedirs(output_dir, exist_ok=True)

    def save_upload(self, filename: str, content: bytes) -> str:
        video_id = uuid.uuid4().hex[:12]
        ext = os.path.splitext(filename)[1] or ".mp4"
        path = os.path.join(self.upload_dir, f"{video_id}{ext}")
        with open(path, "wb") as f:
            f.write(content)
        return video_id

    def get_upload_path(self, video_id: str) -> str:
        for f in os.listdir(self.upload_dir):
            if f.startswith(video_id):
                return os.path.join(self.upload_dir, f)
        raise FileNotFoundError(f"Upload not found: {video_id}")

    def get_output_dir(self, job_id: str) -> str:
        path = os.path.join(self.output_dir, job_id)
        os.makedirs(path, exist_ok=True)
        return path
