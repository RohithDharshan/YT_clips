# Project Ray — AI Video Clipper

An AI-powered short-form video clip generator. Paste a YouTube URL or upload a video and get ranked, captioned, watermarked clips ready to post.

## Quick Start

```bash
# 1. Install ffmpeg (required)
brew install ffmpeg

# 2. Run everything
./start.sh
```

Open http://localhost:3000 in your browser.

## Pipeline

1. **Download** — yt-dlp pulls the YouTube video (or you upload MP4/MOV/MKV)
2. **Transcribe** — Whisper `base` model generates word-level timestamps
3. **Score** — Each segment is scored on:
   - Audio energy (RMS volume peaks)
   - Scene cut frequency (shot boundary detection via OpenCV)
   - Text signals (keyword density, questions, punchlines)
4. **Clip** — Top-N segments are:
   - Trimmed at natural sentence boundaries
   - Reframed for the target aspect ratio with face-tracking pan (MediaPipe)
   - Burned-in word-by-word highlight captions (PIL)
   - Watermarked with "Project Ray" branding
5. **Cache** — transcript + scores are cached 6 hours so you can regenerate with different settings instantly

## Controls

| Setting | Options |
|---|---|
| Duration | <30s · <1min · Custom (min–max seconds) |
| Aspect Ratio | 9:16 · 1:1 · 16:9 · Custom W:H |
| Number of clips | 1–10 (default 5) |

## API Endpoints

| Method | Path | Description |
|---|---|---|
| POST | `/api/process/youtube` | Submit YouTube URL |
| POST | `/api/process/upload` | Upload video file |
| POST | `/api/regenerate` | Regenerate clips with new settings (uses cache) |
| GET | `/api/status/{job_id}` | Poll job status + results |

## Requirements

- Python 3.10+
- ffmpeg
- ~4 GB RAM (Whisper base model)
- GPU optional (CPU runs fine, slower)
