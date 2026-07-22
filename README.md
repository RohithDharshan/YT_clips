# ClipMind — Think Less. Create More.

An AI-powered short-form clip studio by **ROJIT ENTERPRISES PVT LTD**. Paste a YouTube URL or upload a video, and get ranked, reframed, captioned clips — then fine-tune each one in a built-in editor before you post.

## Pages

- **`index.html`** — landing page: cinematic logo intro (the ClipMind logo alone, then a scroll-driven zoom dive that dissolves into the home page), hero, features, how-it-works, CTA
- **`login.html` / `signup.html`** — Google Sign-In only (no passwords stored anywhere)
- **`pricing.html`** — Free vs Pro plan comparison
- **`studio.html`** — the full clip editor, gated behind login

## Quick Start

```bash
# 1. Install ffmpeg (required)
brew install ffmpeg

# 2. Set up Google Sign-In (see below) and your .env
cp backend/.env.example backend/.env      # then fill in GOOGLE_CLIENT_ID
cp frontend/config.example.js frontend/config.js  # then fill in the same Client ID

# 3. Run everything
./start.sh
```

Open http://localhost:3000 in your browser. If port 8000 is busy, the API automatically starts on 8010 and the frontend finds it on its own.

## Google Sign-In setup

ClipMind accepts **Google accounts only** (restricted to `@gmail.com` by default — configurable via `ALLOWED_EMAIL_DOMAIN`). You need a free Google Cloud OAuth Client ID; the Client ID is public by design (it's shipped in frontend JS) — no secret key is needed for this flow.

1. Go to [console.cloud.google.com](https://console.cloud.google.com) → create a project (or pick an existing one).
2. **APIs & Services → OAuth consent screen** → configure it (External, fill in app name/logo/support email). Publishing status can stay in "Testing" while you develop.
3. **APIs & Services → Credentials → Create Credentials → OAuth client ID** → Application type **Web application**.
4. Under **Authorized JavaScript origins**, add every origin you'll load the frontend from, e.g.:
   - `http://localhost:3005` (dev)
   - `https://yourdomain.com` (production)
5. Copy the generated **Client ID** (looks like `123...apps.googleusercontent.com`) into:
   - `backend/.env` → `GOOGLE_CLIENT_ID=...`
   - `frontend/config.js` → `window.__GOOGLE_CLIENT_ID__ = "..."`
6. Restart the backend. Until this is set, `login.html`/`signup.html` show a clear "Google sign-in isn't configured yet" message instead of a broken button.

No Client Secret is needed — the backend verifies the ID token's signature and audience directly against Google's public keys (`google-auth` library), it never talks to Google with a secret.

## Pricing

| | Free | Pro |
|---|---|---|
| Price | $0 | **$15/month** ($12/mo billed annually) |
| Videos / month | 5 | 50 |
| Max source length | 15 min | 90 min |
| Clips per video | 3 | 10 |
| Export resolution | 720p | 1080p |
| Watermark | Forced on | Off |

**Why these numbers:** the pipeline is fully CPU-bound (Whisper + OpenCV/MediaPipe + ffmpeg, no GPU needed), so marginal compute cost per video is small — a few cents even for a 10-minute video on a modest VPS. Limits exist to bound worst-case load and give Free users a real reason to upgrade, not because the compute is expensive. $15/mo matches the market (Opus.pro, Klap, Vidyo.ai all sit in the $15–30/mo range) for a generous, simple two-tier plan.

Limits live in `backend/config.py` (`PLAN_LIMITS`, `PRICING`) — change the numbers there, no other code changes needed. They're mirrored in `frontend/pricing.html`'s `LIMITS`/`PRICE` constants for display; keep both in sync if you tune them.

**Billing is not connected to a real payment processor yet.** `POST /api/billing/upgrade` is a **development-only** stub that flips a user to Pro directly (it 501s in production — see `ENV` below). Wiring up real payments (Stripe Checkout is the natural fit) needs your own Stripe account and API keys.

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
   - **Watermark**: toggleable, custom text (forced on for Free plan)
   - **Resolution**: capped per plan (720p Free / 1080p Pro)
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
- **Header** — plan badge, monthly usage, avatar, Upgrade CTA (Free plan only), log out

## Controls

| Setting | Options |
|---|---|
| Duration | <30s · <1min · Custom (min–max seconds) |
| Aspect Ratio | 9:16 · 1:1 · 16:9 · Custom W:H |
| Framing | Auto (AI decides from analysis) · Normal (full frame) · Focus subject (tracked crop) · Manual (crop position, zoom 1–4×, 3×3 grid placement, rotation ±45°) |
| Captions | Karaoke · Bold · Minimal · None |
| Watermark | On/off + custom text (Pro only — Free always watermarks) |
| Number of clips | 1–10, capped by plan (default 5) |

## API Endpoints

| Method | Path | Description |
|---|---|---|
| POST | `/api/auth/google` | Sign in with a Google ID token → session token |
| POST | `/api/auth/logout` | Invalidate the session token |
| GET | `/api/auth/me` | Current user 🔒 |
| DELETE | `/api/auth/me` | Delete your account and all its data 🔒 |
| GET | `/api/billing` | Current plan, usage, limits, pricing 🔒 |
| POST | `/api/billing/upgrade` | Dev-only plan upgrade stub (501 in production) 🔒 |
| POST | `/api/process/youtube` | Submit YouTube URL 🔒 |
| POST | `/api/process/upload` | Upload video file (multipart form) 🔒 |
| POST | `/api/regenerate` | Regenerate all clips with new settings (uses cache) 🔒 |
| POST | `/api/clip/rerender` | Re-render one clip with new trim/style 🔒 |
| GET | `/api/status/{job_id}` | Poll job status, progress detail + results 🔒 |
| GET | `/api/transcript/{job_id}` | Full transcript + scored segments 🔒 |
| GET | `/api/frame/{job_id}` | Source-video frame for the manual crop editor 🔒 |
| GET | `/api/jobs` | Your recent projects 🔒 |
| GET | `/api/download/{job_id}.zip` | All clips as a ZIP 🔒 |
| DELETE | `/api/job/{job_id}` | Delete a job and its clips 🔒 |
| GET | `/healthz` | Liveness check (for uptime monitors / load balancers) |

🔒 = requires `Authorization: Bearer <token>`, and (where a `job_id` is involved) that the token's user owns that job — every job-scoped endpoint 404s rather than leaking another user's data.

## Production checklist

Everything below is already wired up; the table is what to configure/verify before a real deploy.

| Concern | What's in place |
|---|---|
| Auth | Google Sign-In only, ID tokens verified server-side against Google's public keys, domain-restricted, no passwords stored |
| Session tokens | 256-bit random, 30-day expiry, revocable via logout |
| Authorization | Every job-scoped endpoint checks `job.user_id` against the caller — set this up per `.env` before going live |
| CORS | `ALLOWED_ORIGINS` in `.env` — **must** list your real domain(s); `*` is not accepted with credentialed requests anyway |
| Rate limiting | `slowapi`, per-IP: 15/min on sign-in, 10/min on processing endpoints, 20–30/min on regenerate/rerender |
| Upload limits | Streamed to disk in 1 MB chunks with a hard cap (`MAX_UPLOAD_BYTES`, default 1 GB) — never buffered fully in memory |
| Plan limits | Enforced server-side before any compute starts (video count, source length) and during rendering (clip count, resolution, watermark) — never trust the client |
| Error handling | Unhandled exceptions are logged server-side and return a generic 500 to the client — no stack traces leak |
| Logging | Structured `logging` module output; wire your platform's log drain to it |
| Disk retention | `backend/cleanup.py` removes clips/uploads/exports/analysis-cache older than `RETENTION_DAYS` (default 30) — cron it daily |
| Health check | `GET /healthz` — point your load balancer / uptime monitor at this |
| Secrets | `backend/.env` (gitignored) — never commit `GOOGLE_CLIENT_ID`-adjacent secrets; the Client ID itself is fine to expose (it's public by design) |

**Known architecture limit — single process only.** Job state (`jobs` dict) and the rate limiter live in the process's memory, not a shared store. Running multiple uvicorn/gunicorn **worker processes** (or multiple instances behind a load balancer) means a request can land on a worker that never saw the job get created. Keep this to **one process** (you can still use `--workers 1` explicitly) until job state is moved to Redis or the database — a real scaling need, not a v1 concern.

### Environment variables (`backend/.env`)

| Variable | Default | Purpose |
|---|---|---|
| `GOOGLE_CLIENT_ID` | *(empty — required)* | OAuth Client ID from Google Cloud Console |
| `ALLOWED_EMAIL_DOMAIN` | `gmail.com` | Only accounts on this domain may sign in |
| `ALLOWED_ORIGINS` | `http://localhost:3000,http://localhost:3005` | Comma-separated CORS allowlist |
| `ENV` | `development` | Set to `production` to disable the billing-upgrade dev stub |
| `MAX_UPLOAD_BYTES` | `1073741824` (1 GB) | Hard cap on uploaded file size |
| `RETENTION_DAYS` | `30` | How long `cleanup.py` keeps rendered clips/uploads |

## Requirements

- Python 3.10–3.12
- ffmpeg
- ~4 GB RAM (Whisper base model)
- GPU optional (CPU runs fine, slower)
