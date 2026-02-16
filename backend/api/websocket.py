"""WebSocket handler for real-time progress updates."""

import asyncio
import json
from fastapi import APIRouter, WebSocket, WebSocketDisconnect

from backend.api.routes import job_manager

ws_router = APIRouter()


@ws_router.websocket("/ws/progress/{job_id}")
async def progress_websocket(websocket: WebSocket, job_id: str):
    await websocket.accept()
    try:
        last_progress = -1
        while True:
            job = job_manager.get(job_id)
            if not job:
                await websocket.send_json({"error": "Job not found"})
                break

            if job["progress"] != last_progress:
                last_progress = job["progress"]
                await websocket.send_json({
                    "job_id": job_id,
                    "status": job["status"],
                    "progress": job["progress"],
                    "stage": job["stage"],
                })

            if job["status"] in ("completed", "failed"):
                break

            await asyncio.sleep(0.5)
    except WebSocketDisconnect:
        pass
