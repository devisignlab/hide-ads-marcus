from backend.core.jobs import JobManager, JobStatus


class TestJobManager:
    def test_create_job(self):
        mgr = JobManager()
        job_id = mgr.create("video_123", {"method": "fgsm"})
        assert isinstance(job_id, str)
        job = mgr.get(job_id)
        assert job["status"] == JobStatus.PENDING
        assert job["video_id"] == "video_123"

    def test_update_progress(self):
        mgr = JobManager()
        job_id = mgr.create("v1", {})
        mgr.update_progress(job_id, 50, "perturbing")
        job = mgr.get(job_id)
        assert job["progress"] == 50
        assert job["stage"] == "perturbing"

    def test_complete_job(self):
        mgr = JobManager()
        job_id = mgr.create("v1", {})
        mgr.complete(job_id, {"output_path": "/tmp/out.mp4"})
        job = mgr.get(job_id)
        assert job["status"] == JobStatus.COMPLETED
        assert job["result"]["output_path"] == "/tmp/out.mp4"

    def test_fail_job(self):
        mgr = JobManager()
        job_id = mgr.create("v1", {})
        mgr.fail(job_id, "Something went wrong")
        job = mgr.get(job_id)
        assert job["status"] == JobStatus.FAILED
        assert "Something went wrong" in job["error"]

    def test_unknown_job_returns_none(self):
        mgr = JobManager()
        assert mgr.get("nonexistent") is None
