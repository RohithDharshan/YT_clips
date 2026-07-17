# ClipMind — Think Less. Create More.

An AI-powered short-form clip studio by **ROJIT ENTERPRISES PVT LTD**. Paste a YouTube URL or upload a video, and get ranked, reframed, captioned clips — then fine-tune each one in a built-in editor before you post.

## Pages

- **`index.html`** — landing page: cinematic logo intro (the ClipMind logo alone, then a scroll-driven zoom dive that dissolves into the home page), hero, features, how-it-works, CTA
- **`login.html` / `signup.html`** — email + password auth (validated forms, password strength meter)
- **`studio.html`** — the full clip editor, gated behind login

## Quick Start

```bash
# 1. Install ffmpeg (required)
brew install ffmpeg

# 2. Run everything
./start.sh
```

Open http://localhost:3000 in your browser. If port 8000 is busy, the API automatically starts on 8010 and the frontend finds it on its own.

## Pipeline

1. **Download** — yt-dlp pulls the YouTube video (or you upload MP4/MOV/MKV/WEBM)
2. **Transcribe** — Whisper `base` model generates word-level timestamps
3. **Subject analysis** — the video is analyzed *before* rendering to decide what to focus on:
   - Face presence across sampled frames (MediaPipe) → talking-head videos track the speaker
   - Dominant moving object vs. a median-frame background → cars, balls, action get tracked instead
   - No clear subject → full-frame (Normal) is recommended
   - The result is shown during processing and drives the **Auto** framing mode
4. **Score** — Each segment is scored on:
   - Audio energy + spectral flux (crowd roars, impacts)
   - Scene cut frequency & motion (OpenCV)
   - Text signals (keyword density, questions, punchlines)
   - Hook strength ("what if…", "here's why…", question openers)
   - Climax moments (wins, overtakes, goals, emotional peaks — heavily weighted)
   - Videos with no speech fall back to audio/motion-scored windows
5. **Render** — Top-N segments are rendered in a single pass:
   - **Framing**: `auto` (use the subject analysis), `fit` (Normal — full frame) or `fill` (Focus — shot-aware reframing: scene cuts are detected, each shot gets a stable crop locked on the median face/subject position, with slow keyframed panning only in long shots; shots whose action is wider than the crop window — two cars side by side, full-width graphics — automatically fall back to full-frame so nothing gets cut off. Steady even on rapid footage like sports)
   - **Captions**: word-by-word highlight styles — `karaoke`, `bold`, `minimal`, or `none`
   - **Watermark**: toggleable, custom text
   - Per-clip thumbnail + virality score breakdown
6. **Cache** — transcript + scores + subject analysis cached 24 h; jobs persist across server restarts

## The Editor

After processing you land in a full editor:

- **Clip rail** — thumbnails, rank badges, scores; click to switch clips
- **Preview player** — spacebar play/pause, playhead synced to the trim strip
- **Trim strip** — drag in/out handles or slide the whole selection, then *Apply & re-render* that one clip (uses the cached analysis, no re-transcribe)
- **Inspector** — per-clip caption style, framing, aspect ratio, watermark; *Regenerate all* applies everywhere
- **Transcript tab** — click a line to seek; clicking a line outside the clip extends the trim to include it
- **Analytics tab** — hook / climax / audio / scene / text score breakdown
- **Source heatmap** — the whole video's highlight scores with clip regions; click to jump between clips
- **Export** — download a single clip or all of them as a ZIP

## Controls

| Setting | Options |
|---|---|
| Duration | <30s · <1min · Custom (min–max seconds) |
| Aspect Ratio | 9:16 · 1:1 · 16:9 · Custom W:H |
| Framing | Auto (AI decides from analysis) · Normal (full frame) · Focus subject (tracked crop) · Manual (crop position, zoom 1–4×, 3×3 grid placement, rotation ±45°) |
| Captions | Karaoke · Bold · Minimal · None |
| Watermark | On/off + custom text |
| Number of clips | 1–10 (default 5) |

## API Endpoints

| Method | Path | Description |
|---|---|---|
| POST | `/api/auth/signup` | Create account (name, email, password ≥ 8 chars) |
| POST | `/api/auth/login` | Log in → session token (30-day expiry) |
| POST | `/api/auth/logout` | Invalidate the session token |
| GET | `/api/auth/me` | Current user (Bearer token) |
| POST | `/api/process/youtube` | Submit YouTube URL 🔒 |
| POST | `/api/process/upload` | Upload video file (multipart form) 🔒 |
| POST | `/api/regenerate` | Regenerate all clips with new settings (uses cache) 🔒 |
| POST | `/api/clip/rerender` | Re-render one clip with new trim/style 🔒 |
| GET | `/api/status/{job_id}` | Poll job status, progress detail + results |
| GET | `/api/transcript/{job_id}` | Full transcript + scored segments |
| GET | `/api/jobs` | Recent projects |
| GET | `/api/download/{job_id}.zip` | All clips as a ZIP |
| DELETE | `/api/job/{job_id}` | Delete a job and its clips 🔒 |

🔒 = requires `Authorization: Bearer <token>`. Passwords are stored as PBKDF2-HMAC-SHA256 (200k iterations, per-user salt) in SQLite.

## Requirements

- Python 3.10–3.12
- ffmpeg
- ~4 GB RAM (Whisper base model)
- GPU optional (CPU runs fine, slower)
