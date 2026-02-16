import pytest
from httpx import AsyncClient, ASGITransport
from backend.main import create_app


@pytest.fixture
def app():
    return create_app()


@pytest.fixture
async def client(app):
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as c:
        yield c


class TestUpload:
    @pytest.mark.asyncio
    async def test_upload_video(self, client):
        content = b"fake video content"
        resp = await client.post(
            "/api/upload",
            files={"file": ("test.mp4", content, "video/mp4")},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert "video_id" in data


class TestProcess:
    @pytest.mark.asyncio
    async def test_process_returns_job_id(self, client):
        # Upload first
        resp = await client.post(
            "/api/upload",
            files={"file": ("test.mp4", b"fake", "video/mp4")},
        )
        video_id = resp.json()["video_id"]

        resp = await client.post("/api/process", json={
            "video_id": video_id,
            "attack_method": "lsb",
            "preset": "preview",
        })
        assert resp.status_code == 200
        assert "job_id" in resp.json()


class TestStatus:
    @pytest.mark.asyncio
    async def test_unknown_job_404(self, client):
        resp = await client.get("/api/status/nonexistent")
        assert resp.status_code == 404
