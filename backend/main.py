import asyncio
import json
import logging
import os
import subprocess
import time
import uuid
import zipfile
from typing import Optional

from fastapi import (BackgroundTasks, Depends, FastAPI, File, Form, Header,
                     HTTPException, Request, UploadFile)
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
from slowapi.util import get_remote_address

import auth
import config
from pipeline.cache import get_cache, set_cache
from pipeline.clipper import generate_clips, render_single_clip
from pipeline.downloader import download_youtube
from pipeline.focus import analyze_focus
from pipeline.scorer import score_segments
from pipeline.transcriber import transcribe_video

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
)
log = logging.getLogger("clipmind")

limiter = Limiter(key_func=get_remote_address)

app = FastAPI(title="ClipMind API")
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

app.add_middleware(
    CORSMiddleware,
    allow_origins=config.ALLOWED_ORIGINS,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.exception_handler(Exception)
async def unhandled_exception_handler(request: Request, exc: Exception):
    log.exception("Unhandled error on %s %s", request.method, request.url.path)
    return JSONResponse(status_code=500, content={"detail": "Internal server error"})


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
        log.exception("Failed to persist jobs.json")


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
    framing: str = "auto"                   # "auto" (use analysis), "fit" (normal), "fill" (focus subject), "manual"
    caption_style: str = "karaoke"          # "karaoke", "bold", "minimal", "none"
    watermark: bool = True
    watermark_text: Optional[str] = None
    # manual framing controls (used when framing == "manual")
    crop_x: float = 0.5                     # crop-window center, fraction of source width
    crop_y: float = 0.5                     # crop-window center, fraction of source height
    zoom: float = 1.0                       # 1.0 (widest) … 4.0 (4x punch-in)
    rotate: float = 0.0                     # degrees, -45 … 45
    max_dim: int = 1080                     # server-set from plan; not user-controlled


class YouTubeRequest(ClipSettings):
    url: str


class GenerateRequest(ClipSettings):
    job_id: str


class RerenderRequest(ClipSettings):
    job_id: str
    rank: int
    start: float
    end: float


def _bearer(authorization: Optional[str]) -> str:
    return (authorization or "").removeprefix("Bearer ").strip()


async def require_user(authorization: Optional[str] = Header(None)) -> dict:
    user = auth.get_user(_bearer(authorization))
    if not user:
        raise HTTPException(401, "Login required")
    return user


def _own_job(job_id: str, user: dict) -> dict:
    """Fetch a job, 404ing (not 403 — avoid confirming job_id exists) if the
    caller doesn't own it."""
    job = jobs.get(job_id)
    if not job or job.get("user_id") != user["id"]:
        raise HTTPException(404, "Job not found")
    return job


def _apply_plan_limits(req: ClipSettings, user: dict) -> ClipSettings:
    limits = config.PLAN_LIMITS[user.get("plan", "free")]
    req.num_clips = min(req.num_clips, limits["max_clips_per_video"])
    req.max_dim = limits["max_resolution"]
    if limits["force_watermark"]:
        req.watermark = True
    return req


def _video_duration_minutes(path: str) -> float:
    try:
        out = subprocess.run(
            ["ffprobe", "-v", "error", "-show_entries", "format=duration",
             "-of", "default=noprint_wrappers=1:nokey=1", path],
            capture_output=True, text=True, timeout=30,
        )
        return float(out.stdout.strip()) / 60.0
    except Exception:
        return 0.0


class GoogleAuthRequest(BaseModel):
    id_token: str


@app.get("/")
async def root():
    return {"status": "ClipMind API running", "env": config.ENV}


@app.get("/healthz")
async def healthz():
    return {"ok": True, "time": time.time()}


@app.post("/api/auth/google")
@limiter.limit("15/minute")
async def api_auth_google(request: Request, req: GoogleAuthRequest):
    try:
        return auth.login_with_google(req.id_token)
    except auth.AuthError as e:
        raise HTTPException(401, str(e))


@app.post("/api/auth/logout")
async def api_logout(authorization: Optional[str] = Header(None)):
    auth.logout(_bearer(authorization))
    return {"ok": True}


@app.get("/api/auth/me")
async def api_me(user: dict = Depends(require_user)):
    return user


@app.delete("/api/auth/me")
async def api_delete_me(user: dict = Depends(require_user)):
    auth.delete_account(user["id"])
    return {"deleted": True}


@app.get("/api/billing")
async def api_billing(user: dict = Depends(require_user)):
    return {
        "plan": user["plan"],
        "limits": config.PLAN_LIMITS[user["plan"]],
        "usage": auth.get_usage(user["id"]),
        "pricing": config.PRICING,
    }


@app.post("/api/billing/upgrade")
async def api_billing_upgrade(user: dict = Depends(require_user)):
    if config.ENV == "production":
        raise HTTPException(
            501, "Payment processing isn't connected yet — checkout is coming soon.")
    # Development-only stub so the upgrade flow can be exercised before
    # Stripe (or another processor) is wired up with real API keys.
    auth.set_plan(user["id"], "pro")
    return {"plan": "pro"}


@app.post("/api/process/youtube")
@limiter.limit("10/minute")
async def process_youtube(request: Request, req: YouTubeRequest, bg: BackgroundTasks,
                          user: dict = Depends(require_user)):
    limits = config.PLAN_LIMITS[user["plan"]]
    usage = auth.get_usage(user["id"])
    if usage["videos"] >= limits["videos_per_month"]:
        raise HTTPException(
            402, f"You've used all {limits['videos_per_month']} videos on the "
                 f"{limits['label']} plan this month. Upgrade to Pro for more.")

    req = _apply_plan_limits(req, user)
    job_id = str(uuid.uuid4())
    _update_job(job_id, status="queued", progress=0, detail="Queued", clips=[],
                source=req.url, created=time.time(), user_id=user["id"],
                settings=req.model_dump(exclude={"url"}))
    bg.add_task(run_youtube_pipeline, job_id, req, user["id"])
    return {"job_id": job_id}


@app.post("/api/process/upload")
@limiter.limit("10/minute")
async def process_upload(
    request: Request,
    bg: BackgroundTasks,
    user: dict = Depends(require_user),
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
    limits = config.PLAN_LIMITS[user["plan"]]
    usage = auth.get_usage(user["id"])
    if usage["videos"] >= limits["videos_per_month"]:
        raise HTTPException(
            402, f"You've used all {limits['videos_per_month']} videos on the "
                 f"{limits['label']} plan this month. Upgrade to Pro for more.")

    ext = os.path.splitext(file.filename)[-1].lower()
    if ext not in [".mp4", ".mov", ".mkv", ".webm"]:
        raise HTTPException(400, "Unsupported file format")

    job_id = str(uuid.uuid4())
    video_path = os.path.join(UPLOAD_DIR, f"{job_id}{ext}")

    written = 0
    try:
        with open(video_path, "wb") as f:
            while chunk := await file.read(1024 * 1024):
                written += len(chunk)
                if written > config.MAX_UPLOAD_BYTES:
                    limit_mb = config.MAX_UPLOAD_BYTES / (1024 * 1024)
                    raise HTTPException(
                        413, f"File exceeds the {limit_mb:.1f} MB upload limit")
                f.write(chunk)
    except HTTPException:
        if os.path.exists(video_path):
            os.remove(video_path)
        raise

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
    params = _apply_plan_limits(params, user)
    _update_job(job_id, status="queued", progress=0, detail="Queued", clips=[],
                source=file.filename, created=time.time(), user_id=user["id"],
                settings=params.model_dump(exclude={"job_id"}))
    bg.add_task(run_video_pipeline, job_id, video_path, params, user["id"])
    return {"job_id": job_id}


@app.post("/api/regenerate")
@limiter.limit("20/minute")
async def regenerate(request: Request, req: GenerateRequest, bg: BackgroundTasks,
                     user: dict = Depends(require_user)):
    existing = _own_job(req.job_id, user)
    cached = get_cache(req.job_id)
    if not cached:
        raise HTTPException(404, "Job not found or cache expired")

    req = _apply_plan_limits(req, user)
    _update_job(req.job_id, status="regenerating", progress=0,
                detail="Regenerating", clips=[],
                source=existing.get("source"), created=existing.get("created", time.time()),
                user_id=user["id"], focus=existing.get("focus") or cached.get("focus"),
                settings=req.model_dump(exclude={"job_id"}))
    bg.add_task(run_clip_generation, req.job_id, cached, req, user["id"], False)
    return {"job_id": req.job_id}


@app.post("/api/clip/rerender")
@limiter.limit("30/minute")
async def rerender_clip(request: Request, req: RerenderRequest, user: dict = Depends(require_user)):
    _own_job(req.job_id, user)
    cached = get_cache(req.job_id)
    if not cached:
        raise HTTPException(404, "Job not found or cache expired")
    if not os.path.exists(cached.get("video_path", "")):
        raise HTTPException(410, "Source video no longer available — reprocess the job")

    req = _apply_plan_limits(req, user)
    try:
        clip = await asyncio.get_event_loop().run_in_executor(
            None, render_single_clip, req.job_id, cached, req.rank, req.start, req.end, req,
        )
    except ValueError as e:
        raise HTTPException(400, str(e))
    except Exception as e:
        log.exception("Re-render failed for job %s", req.job_id)
        raise HTTPException(500, f"Render failed: {e}")

    job = jobs.get(req.job_id)
    if job and job.get("clips"):
        for i, existing in enumerate(job["clips"]):
            if existing.get("rank") == req.rank:
                job["clips"][i] = clip
                break
        _save_jobs()

    return {"clip": clip}


@app.get("/api/frame/{job_id}")
async def get_source_frame(job_id: str, t: float = 0.0, user: dict = Depends(require_user)):
    """A single source-video frame, used by the manual crop editor."""
    _own_job(job_id, user)
    cached = get_cache(job_id)
    if not cached or not os.path.exists(cached.get("video_path", "")):
        raise HTTPException(404, "Source video not available")

    t = max(0.0, float(t))
    frame_path = os.path.join(EXPORT_DIR, f"{job_id}_frame_{t:.1f}.jpg")
    if not os.path.exists(frame_path):
        def extract():
            subprocess.run([
                "ffmpeg", "-y", "-ss", str(t), "-i", cached["video_path"],
                "-vframes", "1", "-vf", "scale=720:-2", "-q:v", "5", frame_path,
            ], check=True, capture_output=True)
        try:
            await asyncio.get_event_loop().run_in_executor(None, extract)
        except Exception:
            raise HTTPException(500, "Could not extract frame")

    return FileResponse(frame_path, media_type="image/jpeg")


@app.get("/api/transcript/{job_id}")
async def get_transcript(job_id: str, user: dict = Depends(require_user)):
    _own_job(job_id, user)
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
async def get_status(job_id: str, user: dict = Depends(require_user)):
    return _own_job(job_id, user)


@app.get("/api/jobs")
async def list_jobs(user: dict = Depends(require_user)):
    entries = []
    for job_id, job in jobs.items():
        if job.get("user_id") != user["id"]:
            continue
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
async def download_all(job_id: str, user: dict = Depends(require_user)):
    job = _own_job(job_id, user)
    if not job.get("clips"):
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
                        filename="clipmind-clips.zip")


@app.delete("/api/job/{job_id}")
async def delete_job(job_id: str, user: dict = Depends(require_user)):
    job = _own_job(job_id, user)
    jobs.pop(job_id, None)
    _save_jobs()
    for clip in job.get("clips") or []:
        for url in (clip.get("clip_url"), clip.get("thumb_url")):
            if url:
                path = os.path.join(CLIPS_DIR, os.path.basename(url))
                if os.path.exists(path):
                    os.remove(path)
    return {"deleted": True}


async def run_youtube_pipeline(job_id: str, req: YouTubeRequest, user_id: int):
    try:
        _update_job(job_id, status="downloading", progress=5, detail="Downloading video")
        video_path = await asyncio.get_event_loop().run_in_executor(
            None, download_youtube, req.url, f"../cache/{job_id}"
        )
        await run_video_pipeline(job_id, video_path, req, user_id)
    except Exception as e:
        log.exception("YouTube pipeline failed for job %s", job_id)
        _update_job(job_id, status="error", error=str(e))


async def run_video_pipeline(job_id: str, video_path: str, req, user_id: int):
    try:
        user = auth.get_user_by_id(user_id)
        limits = config.PLAN_LIMITS[user["plan"] if user else "free"]
        duration_min = await asyncio.get_event_loop().run_in_executor(
            None, _video_duration_minutes, video_path
        )
        if duration_min > limits["max_source_minutes"]:
            length_str = (f"{duration_min * 60:.0f} seconds" if duration_min < 1
                         else f"{duration_min:.1f} minutes")
            _update_job(
                job_id, status="error",
                error=f"This video is {length_str} — the {limits['label']} plan "
                      f"allows up to {limits['max_source_minutes']} minutes per video. "
                      f"Upgrade to Pro for longer videos.")
            return

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
                    "video_path": video_path, "focus": focus, "duration_min": duration_min}
        set_cache(job_id, analysis)

        await run_clip_generation(job_id, analysis, req, user_id, True)
    except Exception as e:
        log.exception("Video pipeline failed for job %s", job_id)
        _update_job(job_id, status="error", error=str(e))


async def run_clip_generation(job_id: str, analysis: dict, req, user_id: int, count_usage: bool):
    try:
        _update_job(job_id, status="generating", progress=55, detail="Rendering clips")

        def progress_cb(fraction, detail):
            _update_job(job_id, progress=int(55 + fraction * 43), detail=detail)

        clips = await asyncio.get_event_loop().run_in_executor(
            None, lambda: generate_clips(job_id, analysis, req, progress_cb)
        )
        _update_job(job_id, status="done", progress=100, detail="Done", clips=clips)

        if count_usage:
            duration_min = analysis.get("duration_min") or 0.0
            auth.record_usage(user_id, duration_min)
    except Exception as e:
        log.exception("Clip generation failed for job %s", job_id)
        _update_job(job_id, status="error", error=str(e))
