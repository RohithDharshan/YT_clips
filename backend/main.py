from fastapi import FastAPI, UploadFile, File, BackgroundTasks, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from typing import Optional
import asyncio
import uuid
import os

from pipeline.downloader import download_youtube
from pipeline.transcriber import transcribe_video
from pipeline.scorer import score_segments
from pipeline.clipper import generate_clips
from pipeline.cache import get_cache, set_cache

app = FastAPI(title="Project Ray - AI Video Clipper")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount("/clips", StaticFiles(directory="../clips"), name="clips")

UPLOAD_DIR = "../cache/uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

jobs: dict = {}


class YouTubeRequest(BaseModel):
    url: str
    duration_preset: str = "<1min"  # "<30s", "<1min", "custom"
    duration_min: Optional[int] = None
    duration_max: Optional[int] = None
    aspect_ratio: str = "9:16"  # "9:16", "1:1", "16:9", "custom"
    ratio_w: Optional[int] = None
    ratio_h: Optional[int] = None
    num_clips: int = 5


class GenerateRequest(BaseModel):
    job_id: str
    duration_preset: str = "<1min"
    duration_min: Optional[int] = None
    duration_max: Optional[int] = None
    aspect_ratio: str = "9:16"
    ratio_w: Optional[int] = None
    ratio_h: Optional[int] = None
    num_clips: int = 5


@app.get("/")
async def root():
    return {"status": "Project Ray AI Clipper running"}


@app.post("/api/process/youtube")
async def process_youtube(req: YouTubeRequest, bg: BackgroundTasks):
    job_id = str(uuid.uuid4())
    jobs[job_id] = {"status": "queued", "progress": 0, "clips": []}
    bg.add_task(run_youtube_pipeline, job_id, req)
    return {"job_id": job_id}


@app.post("/api/process/upload")
async def process_upload(
    bg: BackgroundTasks,
    file: UploadFile = File(...),
    duration_preset: str = "<1min",
    duration_min: Optional[int] = None,
    duration_max: Optional[int] = None,
    aspect_ratio: str = "9:16",
    ratio_w: Optional[int] = None,
    ratio_h: Optional[int] = None,
    num_clips: int = 5,
):
    job_id = str(uuid.uuid4())
    jobs[job_id] = {"status": "queued", "progress": 0, "clips": []}

    ext = os.path.splitext(file.filename)[-1].lower()
    if ext not in [".mp4", ".mov", ".mkv", ".webm"]:
        raise HTTPException(400, "Unsupported file format")

    video_path = os.path.join(UPLOAD_DIR, f"{job_id}{ext}")
    content = await file.read()
    with open(video_path, "wb") as f:
        f.write(content)

    params = GenerateRequest(
        job_id=job_id,
        duration_preset=duration_preset,
        duration_min=duration_min,
        duration_max=duration_max,
        aspect_ratio=aspect_ratio,
        ratio_w=ratio_w,
        ratio_h=ratio_h,
        num_clips=num_clips,
    )
    bg.add_task(run_video_pipeline, job_id, video_path, params)
    return {"job_id": job_id}


@app.post("/api/regenerate")
async def regenerate(req: GenerateRequest, bg: BackgroundTasks):
    cached = get_cache(req.job_id)
    if not cached:
        raise HTTPException(404, "Job not found or cache expired")
    jobs[req.job_id] = {"status": "regenerating", "progress": 0, "clips": []}
    bg.add_task(run_clip_generation, req.job_id, cached, req)
    return {"job_id": req.job_id}


@app.get("/api/status/{job_id}")
async def get_status(job_id: str):
    if job_id not in jobs:
        raise HTTPException(404, "Job not found")
    return jobs[job_id]


async def run_youtube_pipeline(job_id: str, req: YouTubeRequest):
    try:
        jobs[job_id].update({"status": "downloading", "progress": 5})
        video_path = await asyncio.get_event_loop().run_in_executor(
            None, download_youtube, req.url, f"../cache/{job_id}"
        )
        await run_video_pipeline(job_id, video_path, req)
    except Exception as e:
        jobs[job_id] = {"status": "error", "error": str(e)}


async def run_video_pipeline(job_id: str, video_path: str, req):
    try:
        jobs[job_id].update({"status": "transcribing", "progress": 15})
        transcript, audio_path = await asyncio.get_event_loop().run_in_executor(
            None, transcribe_video, video_path
        )

        jobs[job_id].update({"status": "analyzing", "progress": 40})
        segments = await asyncio.get_event_loop().run_in_executor(
            None, score_segments, video_path, audio_path, transcript
        )

        analysis = {"transcript": transcript, "segments": segments, "video_path": video_path}
        set_cache(job_id, analysis)

        await run_clip_generation(job_id, analysis, req)
    except Exception as e:
        jobs[job_id] = {"status": "error", "error": str(e)}


async def run_clip_generation(job_id: str, analysis: dict, req):
    try:
        jobs[job_id].update({"status": "generating", "progress": 60})
        clips = await asyncio.get_event_loop().run_in_executor(
            None, generate_clips, job_id, analysis, req
        )
        jobs[job_id] = {"status": "done", "progress": 100, "clips": clips}
    except Exception as e:
        jobs[job_id] = {"status": "error", "error": str(e)}
