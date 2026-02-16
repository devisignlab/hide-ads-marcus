"""REST API routes."""

import asyncio
from fastapi import APIRouter, UploadFile, File, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel

from backend.core.jobs import JobManager
from backend.storage.manager import StorageManager
from backend.pipeline.orchestrator import PipelineConfig, run_pipeline

router = APIRouter(prefix="/api")
job_manager = JobManager()
storage = StorageManager()


class ProcessRequest(BaseModel):
    video_id: str
    attack_method: str = "fgsm"
    preset: str = "preview"
    target_text: str = "an empty room"
    repel_text: str = ""
    repel_weight: float = 0.3
    clip_labels: list[str] = [
        "a person talking", "an advertisement", "a product",
        "an empty room", "nature scenery", "abstract art",
    ]
    yolo_weight: float = 0.5
    clip_weight: float = 0.5
    epsilon: float = 0.0  # 0 = use preset default
    alpha: float = 0.0    # 0 = use preset default
    lsb_mode: str = "cloak"
    lsb_intensity: float = 0.5


@router.post("/upload")
async def upload_video(file: UploadFile = File(...)):
    content = await file.read()
    video_id = storage.save_upload(file.filename or "video.mp4", content)
    return {"video_id": video_id}


class AnalyzeResponse(BaseModel):
    flags: list[str]
    risk_level: str
    max_deepfake_score: float
    max_nsfw_score: float
    recommended_attacks: list[str]
    repel_texts: list[str]
    estimated_time_seconds: int
    frame_results: list[dict]

@router.post("/analyze/{video_id}")
async def analyze_video(video_id: str):
    """Run smart analysis on uploaded video to detect content risks."""
    try:
        video_path = storage.get_upload_path(video_id)
    except FileNotFoundError:
        raise HTTPException(404, "Video not found")

    import asyncio
    loop = asyncio.get_event_loop()

    def _run_analysis():
        from backend.attacks.smart_analyze import analyze_video_sample
        from backend.core.video import get_video_info
        return get_video_info(video_path), analyze_video_sample(video_path, num_samples=4)

    info, result = await loop.run_in_executor(None, _run_analysis)

    # Estimate processing time based on frame count and recommended attacks
    total_frames = info["frame_count"]
    fps = info["fps"]
    duration = total_frames / fps

    # UAP CLIP: ~240s fixed, Anti-deepfake: ~5s per keyframe (interval=30)
    time_est = 10  # analysis + mux
    if "clip" in result["recommended_attacks"]:
        time_est += 240  # UAP computation
    if "anti_deepfake" in result["recommended_attacks"]:
        num_keyframes = total_frames // 30 + 1
        time_est += num_keyframes * 5
    time_est += total_frames // 30  # streaming

    max_nsfw = 0.0
    for fr in result["frame_results"]:
        nsfw = max(fr["scores"].get(k, 0) for k in ["nudity", "sexual", "sensual", "lingerie"])
        max_nsfw = max(max_nsfw, nsfw)

    return {
        "flags": result["flags"],
        "risk_level": result["risk_level"],
        "max_deepfake_score": result["max_deepfake_score"],
        "max_nsfw_score": max_nsfw,
        "recommended_attacks": result["recommended_attacks"],
        "repel_texts": result["repel_texts"],
        "estimated_time_seconds": time_est,
        "frame_results": result["frame_results"],
        "video_info": {
            "width": info["width"],
            "height": info["height"],
            "fps": info["fps"],
            "frame_count": info["frame_count"],
            "duration": duration,
        }
    }


@router.post("/process")
async def start_processing(req: ProcessRequest):
    try:
        video_path = storage.get_upload_path(req.video_id)
    except FileNotFoundError:
        raise HTTPException(404, "Video not found")

    output_dir = storage.get_output_dir(req.video_id)
    job_id = job_manager.create(req.video_id, req.model_dump())

    # Resolve epsilon/alpha: use request values if set, else use preset defaults
    from backend.config import get_preset, DEFAULT_EPSILON, DEFAULT_ALPHA
    preset_obj = get_preset(req.preset)
    epsilon = req.epsilon if req.epsilon > 0 else preset_obj.epsilon
    alpha = req.alpha if req.alpha > 0 else max(epsilon / 40, DEFAULT_ALPHA)

    config = PipelineConfig(
        video_path=video_path,
        attack_method=req.attack_method,
        target_text=req.target_text,
        repel_text=req.repel_text,
        repel_weight=req.repel_weight,
        clip_labels=req.clip_labels,
        preset=req.preset,
        epsilon=epsilon,
        alpha=alpha,
        yolo_weight=req.yolo_weight,
        clip_weight=req.clip_weight,
        lsb_mode=req.lsb_mode,
        lsb_intensity=req.lsb_intensity,
        output_dir=output_dir,
    )

    # Run in background thread
    loop = asyncio.get_event_loop()
    loop.run_in_executor(None, _run_job, job_id, config)

    return {"job_id": job_id}


def _run_job(job_id: str, config: PipelineConfig):
    try:
        def on_progress(pct, stage):
            job_manager.update_progress(job_id, pct, stage)

        result = run_pipeline(config, on_progress=on_progress)
        job_manager.complete(job_id, result)
    except Exception as e:
        job_manager.fail(job_id, str(e))


@router.get("/status/{job_id}")
async def get_status(job_id: str):
    job = job_manager.get(job_id)
    if not job:
        raise HTTPException(404, "Job not found")
    return {
        "job_id": job["job_id"],
        "status": job["status"],
        "progress": job["progress"],
        "stage": job["stage"],
        "error": job["error"],
    }


@router.get("/results/{job_id}")
async def get_results(job_id: str):
    job = job_manager.get(job_id)
    if not job:
        raise HTTPException(404, "Job not found")
    if job["status"] != "completed":
        raise HTTPException(400, f"Job not completed: {job['status']}")
    return job["result"]


@router.get("/export-original/{video_id}")
async def export_original(video_id: str):
    """Serve the original uploaded video for comparison."""
    try:
        video_path = storage.get_upload_path(video_id)
    except FileNotFoundError:
        raise HTTPException(404, "Video not found")
    return FileResponse(video_path, media_type="video/mp4", filename="original.mp4")


@router.get("/export/{job_id}")
async def export_video(job_id: str):
    job = job_manager.get(job_id)
    if not job or job["status"] != "completed":
        raise HTTPException(404, "Job not found or not completed")
    output_path = job["result"]["output_path"]
    return FileResponse(output_path, media_type="video/mp4", filename="processed.mp4")
