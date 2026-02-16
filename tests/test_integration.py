# tests/test_integration.py
"""End-to-end integration test using the test fixture video."""
import os
import time
import asyncio
import pytest
from httpx import AsyncClient, ASGITransport
from backend.main import create_app

FIXTURE = "tests/fixtures/test_2s.mp4"


@pytest.fixture
def app():
    return create_app()


@pytest.fixture
async def client(app):
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as c:
        yield c


class TestE2ELSB:
    """Quick E2E with LSB (no GPU needed, fast)."""

    @pytest.mark.asyncio
    async def test_full_flow_lsb(self, client):
        # 1. Upload
        with open(FIXTURE, "rb") as f:
            resp = await client.post("/api/upload", files={"file": ("test.mp4", f, "video/mp4")})
        assert resp.status_code == 200
        video_id = resp.json()["video_id"]

        # 2. Process
        resp = await client.post("/api/process", json={
            "video_id": video_id,
            "attack_method": "lsb",
            "preset": "preview",
        })
        assert resp.status_code == 200
        job_id = resp.json()["job_id"]

        # 3. Poll until done
        for _ in range(60):
            resp = await client.get(f"/api/status/{job_id}")
            data = resp.json()
            if data["status"] in ("completed", "failed"):
                break
            await asyncio.sleep(0.5)

        assert data["status"] == "completed"

        # 4. Get results
        resp = await client.get(f"/api/results/{job_id}")
        assert resp.status_code == 200
        result = resp.json()
        assert "output_path" in result
        assert result["frames_processed"] == 20
