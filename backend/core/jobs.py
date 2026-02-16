"""In-memory job manager with thread-safe operations."""

import threading
import uuid
from enum import StrEnum


class JobStatus(StrEnum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"


class JobManager:
    def __init__(self):
        self._jobs: dict[str, dict] = {}
        self._lock = threading.Lock()

    def create(self, video_id: str, config: dict) -> str:
        job_id = uuid.uuid4().hex[:12]
        with self._lock:
            self._jobs[job_id] = {
                "job_id": job_id,
                "video_id": video_id,
                "config": config,
                "status": JobStatus.PENDING,
                "progress": 0,
                "stage": "pending",
                "result": None,
                "error": None,
            }
        return job_id

    def get(self, job_id: str) -> dict | None:
        with self._lock:
            job = self._jobs.get(job_id)
            return dict(job) if job else None

    def update_progress(self, job_id: str, progress: int, stage: str) -> None:
        with self._lock:
            if job_id in self._jobs:
                self._jobs[job_id]["progress"] = progress
                self._jobs[job_id]["stage"] = stage
                self._jobs[job_id]["status"] = JobStatus.RUNNING

    def complete(self, job_id: str, result: dict) -> None:
        with self._lock:
            if job_id in self._jobs:
                self._jobs[job_id]["status"] = JobStatus.COMPLETED
                self._jobs[job_id]["progress"] = 100
                self._jobs[job_id]["stage"] = "done"
                self._jobs[job_id]["result"] = result

    def fail(self, job_id: str, error: str) -> None:
        with self._lock:
            if job_id in self._jobs:
                self._jobs[job_id]["status"] = JobStatus.FAILED
                self._jobs[job_id]["error"] = error
