import asyncio
import json
import os
import time
import uuid
import zipfile

from fastapi import BackgroundTasks, FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from typing import Optional

from pipeline.cache import get_cache, set_cache
from pipeline.clipper import generate_clips, render_single_clip
from pipeline.downloader import download_youtube
from pipeline.focus import analyze_focus
from pipeline.scorer import score_segments
from pipeline.transcriber import transcribe_video

app = FastAPI(title="Project Ray - AI Video Clipper")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

CLIPS_DIR = os.path.abspath("../clips")
os.makedirs(CLIPS_DIR, exist_ok=True)
app.mount("/clips", StaticFiles(directory=CLIPS_DIR), name="clips")

UPLOAD_DIR = "../cache/uploads"
EXPORT_DIR = "../cache/exports"
JOBS_FILE = "../cache/jobs.json"
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(EXPORT_DIR, exist_ok=True)

jobs: dict = {}


def _load_jobs():
    global jobs
    try:
        with open(JOBS_FILE) as f:
            jobs = json.load(f)
        # Anything mid-flight when the server died is unrecoverable
        for job in jobs.values():
            if job.get("status") not in ("done", "error"):
                job["status"] = "error"
                job["error"] = "Server restarted during processing"
    except Exception:
        jobs = {}


def _save_jobs():
    try:
        with open(JOBS_FILE, "w") as f:
            json.dump(jobs, f)
    except Exception:
        pass


def _update_job(job_id: str, **fields):
    jobs.setdefault(job_id, {}).update(fields)
    _save_jobs()


_load_jobs()


class ClipSettings(BaseModel):
    duration_preset: str = "<1min"          # "<30s", "<1min", "custom"
    duration_min: Optional[int] = None
    duration_max: Optional[int] = None
    aspect_ratio: str = "9:16"              # "9:16", "1:1", "16:9", "custom"
    ratio_w: Optional[int] = None
    ratio_h: Optional[int] = None
    num_clips: int = 5
    framing: str = "auto"                   # "auto" (use analysis), "fit" (normal), "fill" (focus subject)
    caption_style: str = "karaoke"          # "karaoke", "bold", "minimal", "none"
    watermark: bool = True
    watermark_text: Optional[str] = None


class YouTubeRequest(ClipSettings):
    url: str


class GenerateRequest(ClipSettings):
    job_id: str


class RerenderRequest(ClipSettings):
    job_id: str
    rank: int
    start: float
    end: float


@app.get("/")
async def root():
    return {"status": "Project Ray AI Clipper running"}


@app.post("/api/process/youtube")
async def process_youtube(req: YouTubeRequest, bg: BackgroundTasks):
    job_id = str(uuid.uuid4())
    _update_job(job_id, status="queued", progress=0, detail="Queued", clips=[],
                source=req.url, created=time.time(),
                settings=req.model_dump(exclude={"url"}))
    bg.add_task(run_youtube_pipeline, job_id, req)
    return {"job_id": job_id}


@app.post("/api/process/upload")
async def process_upload(
    bg: BackgroundTasks,
    file: UploadFile = File(...),
    duration_preset: str = Form("<1min"),
    duration_min: Optional[int] = Form(None),
    duration_max: Optional[int] = Form(None),
    aspect_ratio: str = Form("9:16"),
    ratio_w: Optional[int] = Form(None),
    ratio_h: Optional[int] = Form(None),
    num_clips: int = Form(5),
    framing: str = Form("fit"),
    caption_style: str = Form("karaoke"),
    watermark: bool = Form(True),
    watermark_text: Optional[str] = Form(None),
):
    job_id = str(uuid.uuid4())

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
        framing=framing,
        caption_style=caption_style,
        watermark=watermark,
        watermark_text=watermark_text,
    )
    _update_job(job_id, status="queued", progress=0, detail="Queued", clips=[],
                source=file.filename, created=time.time(),
                settings=params.model_dump(exclude={"job_id"}))
    bg.add_task(run_video_pipeline, job_id, video_path, params)
    return {"job_id": job_id}


@app.post("/api/regenerate")
async def regenerate(req: GenerateRequest, bg: BackgroundTasks):
    cached = get_cache(req.job_id)
    if not cached:
        raise HTTPException(404, "Job not found or cache expired")
    existing = jobs.get(req.job_id, {})
    _update_job(req.job_id, status="regenerating", progress=0,
                detail="Regenerating", clips=[],
                source=existing.get("source"), created=existing.get("created", time.time()),
                focus=existing.get("focus") or cached.get("focus"),
                settings=req.model_dump(exclude={"job_id"}))
    bg.add_task(run_clip_generation, req.job_id, cached, req)
    return {"job_id": req.job_id}


@app.post("/api/clip/rerender")
async def rerender_clip(req: RerenderRequest):
    cached = get_cache(req.job_id)
    if not cached:
        raise HTTPException(404, "Job not found or cache expired")
    if not os.path.exists(cached.get("video_path", "")):
        raise HTTPException(410, "Source video no longer available — reprocess the job")

    try:
        clip = await asyncio.get_event_loop().run_in_executor(
            None, render_single_clip, req.job_id, cached, req.rank, req.start, req.end, req,
        )
    except ValueError as e:
        raise HTTPException(400, str(e))
    except Exception as e:
        raise HTTPException(500, f"Render failed: {e}")

    job = jobs.get(req.job_id)
    if job and job.get("clips"):
        for i, existing in enumerate(job["clips"]):
            if existing.get("rank") == req.rank:
                job["clips"][i] = clip
                break
        _save_jobs()

    return {"clip": clip}


@app.get("/api/transcript/{job_id}")
async def get_transcript(job_id: str):
    cached = get_cache(job_id)
    if not cached:
        raise HTTPException(404, "Job not found or cache expired")
    return {
        "transcript": cached.get("transcript", []),
        "segments": cached.get("segments", []),
        "duration": cached.get("duration"),
        "focus": cached.get("focus"),
    }


@app.get("/api/status/{job_id}")
async def get_status(job_id: str):
    if job_id not in jobs:
        raise HTTPException(404, "Job not found")
    return jobs[job_id]


@app.get("/api/jobs")
async def list_jobs():
    entries = []
    for job_id, job in jobs.items():
        entries.append({
            "job_id": job_id,
            "status": job.get("status"),
            "source": job.get("source"),
            "created": job.get("created"),
            "num_clips": len(job.get("clips") or []),
        })
    entries.sort(key=lambda e: e.get("created") or 0, reverse=True)
    return {"jobs": entries[:20]}


@app.get("/api/download/{job_id}.zip")
async def download_all(job_id: str):
    job = jobs.get(job_id)
    if not job or not job.get("clips"):
        raise HTTPException(404, "No clips for this job")

    zip_path = os.path.join(EXPORT_DIR, f"{job_id}.zip")
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_STORED) as zf:
        for clip in job["clips"]:
            filename = os.path.basename(clip["clip_url"])
            path = os.path.join(CLIPS_DIR, filename)
            if os.path.exists(path):
                safe_title = "".join(
                    c for c in clip.get("title", filename) if c.isalnum() or c in " -_"
                ).strip()[:50] or filename
                zf.write(path, f"clip{clip['rank']} - {safe_title}.mp4")

    return FileResponse(zip_path, media_type="application/zip",
                        filename="project-ray-clips.zip")


@app.delete("/api/job/{job_id}")
async def delete_job(job_id: str):
    job = jobs.pop(job_id, None)
    _save_jobs()
    if job:
        for clip in job.get("clips") or []:
            for url in (clip.get("clip_url"), clip.get("thumb_url")):
                if url:
                    path = os.path.join(CLIPS_DIR, os.path.basename(url))
                    if os.path.exists(path):
                        os.remove(path)
    return {"deleted": job is not None}


async def run_youtube_pipeline(job_id: str, req: YouTubeRequest):
    try:
        _update_job(job_id, status="downloading", progress=5, detail="Downloading video")
        video_path = await asyncio.get_event_loop().run_in_executor(
            None, download_youtube, req.url, f"../cache/{job_id}"
        )
        await run_video_pipeline(job_id, video_path, req)
    except Exception as e:
        _update_job(job_id, status="error", error=str(e))


async def run_video_pipeline(job_id: str, video_path: str, req):
    try:
        _update_job(job_id, status="transcribing", progress=15, detail="Transcribing audio")
        transcript, audio_path = await asyncio.get_event_loop().run_in_executor(
            None, transcribe_video, video_path
        )

        _update_job(job_id, status="analyzing", progress=35,
                    detail="Analyzing the video — what should we focus on?")
        focus = await asyncio.get_event_loop().run_in_executor(
            None, analyze_focus, video_path
        )
        _update_job(job_id, focus=focus, progress=42, detail=focus["label"])

        _update_job(job_id, progress=45, detail="Scoring highlights")
        segments = await asyncio.get_event_loop().run_in_executor(
            None, score_segments, video_path, audio_path, transcript
        )

        analysis = {"transcript": transcript, "segments": segments,
                    "video_path": video_path, "focus": focus}
        set_cache(job_id, analysis)

        await run_clip_generation(job_id, analysis, req)
    except Exception as e:
        _update_job(job_id, status="error", error=str(e))


async def run_clip_generation(job_id: str, analysis: dict, req):
    try:
        _update_job(job_id, status="generating", progress=55, detail="Rendering clips")

        def progress_cb(fraction, detail):
            _update_job(job_id, progress=int(55 + fraction * 43), detail=detail)

        clips = await asyncio.get_event_loop().run_in_executor(
            None, lambda: generate_clips(job_id, analysis, req, progress_cb)
        )
        _update_job(job_id, status="done", progress=100, detail="Done", clips=clips)
    except Exception as e:
        _update_job(job_id, status="error", error=str(e))
