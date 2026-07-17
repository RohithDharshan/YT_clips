import os
import re
import subprocess
import tempfile

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

try:
    import mediapipe as mp
    _MP_AVAILABLE = True
except Exception:
    _MP_AVAILABLE = False


CLIPS_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../clips"))
DEFAULT_WATERMARK = "ClipMind"

DURATION_PRESETS = {
    "<30s": (15, 30),
    "<1min": (30, 60),
}

RATIO_PRESETS = {
    "9:16": (9, 16),
    "1:1": (1, 1),
    "16:9": (16, 9),
}

# Caption style presets. Sizes are fractions of output height.
CAPTION_STYLES = {
    "karaoke": {
        "size": 0.045, "bold": True, "upper": False,
        "base_color": (255, 255, 255), "active_color": (255, 214, 10),
        "stroke": 3, "bg_box": False, "max_words": 5,
    },
    "bold": {
        "size": 0.055, "bold": True, "upper": True,
        "base_color": (255, 255, 255), "active_color": (57, 255, 136),
        "stroke": 5, "bg_box": False, "max_words": 4,
    },
    "minimal": {
        "size": 0.034, "bold": False, "upper": False,
        "base_color": (240, 240, 240), "active_color": (255, 255, 255),
        "stroke": 0, "bg_box": True, "max_words": 6,
    },
}

_FONT_CANDIDATES_BOLD = [
    ("/System/Library/Fonts/Supplemental/Arial Black.ttf", 0),
    ("/System/Library/Fonts/Supplemental/Arial Bold.ttf", 0),
    ("/System/Library/Fonts/Helvetica.ttc", 1),
    ("/System/Library/Fonts/Helvetica.ttc", 0),
    ("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 0),
]
_FONT_CANDIDATES_REGULAR = [
    ("/System/Library/Fonts/Helvetica.ttc", 0),
    ("/System/Library/Fonts/Supplemental/Arial.ttf", 0),
    ("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 0),
]


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def generate_clips(job_id: str, analysis: dict, req, progress_cb=None) -> list:
    """Render the top-N candidate clips. progress_cb(fraction, detail) is optional."""
    os.makedirs(CLIPS_DIR, exist_ok=True)

    duration_min, duration_max = _resolve_duration(req)
    ratio_w, ratio_h = _resolve_ratio(req)

    segments = analysis["segments"]
    video_path = analysis["video_path"]
    transcript = analysis["transcript"]

    video_duration = _get_video_duration(video_path)
    candidates = _select_candidates(segments, duration_min, duration_max, video_duration, req.num_clips)

    results = []
    total = max(len(candidates), 1)
    for rank, cand in enumerate(candidates, 1):
        if progress_cb:
            progress_cb((rank - 1) / total, f"Rendering clip {rank}/{len(candidates)}")

        clip = _render_clip(
            job_id=job_id,
            rank=rank,
            start=cand["start"],
            end=cand["end"],
            transcript=transcript,
            video_path=video_path,
            ratio_w=ratio_w,
            ratio_h=ratio_h,
            req=req,
            score=cand.get("score", 0.0),
            score_breakdown={
                "audio": cand.get("audio_score", 0.0),
                "scene": cand.get("scene_score", 0.0),
                "text": cand.get("text_score", 0.0),
                "climax": cand.get("climax_score", 0.0),
                "hook": cand.get("hook_score", 0.0),
            },
            fallback_text=cand.get("text", ""),
            focus=analysis.get("focus"),
        )
        results.append(clip)

    if progress_cb:
        progress_cb(1.0, "Finalizing")
    return results


def render_single_clip(job_id: str, analysis: dict, rank: int, start: float, end: float, req) -> dict:
    """Re-render one clip with explicit boundaries and current style settings."""
    os.makedirs(CLIPS_DIR, exist_ok=True)
    ratio_w, ratio_h = _resolve_ratio(req)
    video_path = analysis["video_path"]
    transcript = analysis["transcript"]

    duration = _get_video_duration(video_path)
    start = max(0.0, float(start))
    end = min(duration, float(end))
    if end - start < 2:
        raise ValueError("Clip must be at least 2 seconds long")

    # Best-overlap segment supplies the score shown in the UI
    score, breakdown, text = 0.0, {}, ""
    best_overlap = 0.0
    for seg in analysis["segments"]:
        overlap = min(end, seg["end"]) - max(start, seg["start"])
        if overlap > best_overlap:
            best_overlap = overlap
            score = seg.get("score", 0.0)
            text = seg.get("text", "")
            breakdown = {
                "audio": seg.get("audio_score", 0.0),
                "scene": seg.get("scene_score", 0.0),
                "text": seg.get("text_score", 0.0),
                "climax": seg.get("climax_score", 0.0),
                "hook": seg.get("hook_score", 0.0),
            }

    return _render_clip(
        job_id=job_id, rank=rank, start=start, end=end,
        transcript=transcript, video_path=video_path,
        ratio_w=ratio_w, ratio_h=ratio_h, req=req,
        score=score, score_breakdown=breakdown, fallback_text=text,
        focus=analysis.get("focus"),
    )


# ---------------------------------------------------------------------------
# Core rendering
# ---------------------------------------------------------------------------

def _render_clip(job_id, rank, start, end, transcript, video_path,
                 ratio_w, ratio_h, req, score, score_breakdown, fallback_text,
                 focus=None):
    clip_id = f"{job_id}_clip{rank}"
    clip_path = os.path.join(CLIPS_DIR, f"{clip_id}.mp4")
    thumb_path = os.path.join(CLIPS_DIR, f"{clip_id}.jpg")

    words = _get_words_in_range(transcript, start, end)
    full_text = " ".join(w["word"] for w in words) or fallback_text
    title = _generate_title(words, fallback_text)

    caption_style = getattr(req, "caption_style", "karaoke") or "karaoke"
    framing = getattr(req, "framing", "auto") or "auto"
    watermark_on = getattr(req, "watermark", True)
    watermark_text = (getattr(req, "watermark_text", None) or DEFAULT_WATERMARK).strip()

    focus = focus or {}
    focus_subject = focus.get("subject", "face")
    if framing == "auto":
        # Use the subject analysis: focus when there is something to focus on
        framing = focus.get("recommend", "fit")

    manual = {
        "crop_x": getattr(req, "crop_x", 0.5),
        "crop_y": getattr(req, "crop_y", 0.5),
        "zoom": getattr(req, "zoom", 1.0),
        "rotate": getattr(req, "rotate", 0.0),
    }

    trimmed = _trim_video(video_path, start, end)
    try:
        _compose(
            src=trimmed, dst=clip_path,
            ratio_w=ratio_w, ratio_h=ratio_h,
            framing=framing,
            words=words if caption_style != "none" else [],
            caption_style=caption_style,
            watermark_text=watermark_text if watermark_on else None,
            focus_subject=focus_subject,
            manual=manual,
        )
    finally:
        if os.path.exists(trimmed):
            os.remove(trimmed)

    _make_thumbnail(clip_path, thumb_path, end - start)

    return {
        "rank": rank,
        "clip_url": f"/clips/{clip_id}.mp4",
        "thumb_url": f"/clips/{clip_id}.jpg" if os.path.exists(thumb_path) else None,
        "title": title,
        "caption": (full_text or "")[:220],
        "score": score,
        "breakdown": {k: round(v, 3) for k, v in (score_breakdown or {}).items()},
        "start": round(start, 2),
        "end": round(end, 2),
        "duration": round(end - start, 1),
        "settings": {
            "aspect_ratio": f"{ratio_w}:{ratio_h}",
            "framing": framing,  # resolved: "fit", "fill" or "manual", never "auto"
            "focus_subject": focus_subject,
            "caption_style": caption_style,
            "watermark": bool(watermark_on),
            "manual": {k: round(float(v or 0), 3) for k, v in manual.items()},
        },
    }


def _compose(src, dst, ratio_w, ratio_h, framing, words, caption_style, watermark_text,
             focus_subject="face", manual=None):
    """Single cv2 pass: reframe + captions + watermark, then mux audio + encode."""
    out_w, out_h = _compute_output_size_from_ratio(ratio_w, ratio_h)

    cap = cv2.VideoCapture(src)
    src_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    src_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0

    tracker = None
    if framing == "fill":
        tracker = _SubjectTracker(src, src_w, src_h, out_w, out_h, subject=focus_subject)
    elif framing == "manual":
        manual = manual or {}
        tracker = _ManualCropper(
            src_w, src_h, out_w, out_h,
            crop_x=manual.get("crop_x", 0.5), crop_y=manual.get("crop_y", 0.5),
            zoom=manual.get("zoom", 1.0), rotate=manual.get("rotate", 0.0))

    pages = _build_caption_pages(words, caption_style) if words else []
    overlays = _prerender_caption_overlays(pages, caption_style, out_w, out_h, ratio_w, ratio_h)
    wm_overlay = _prerender_watermark(watermark_text, out_w, out_h) if watermark_text else None

    tmp_video = tempfile.mktemp(suffix="_composed.mp4")
    writer = cv2.VideoWriter(tmp_video, cv2.VideoWriter_fourcc(*"mp4v"), fps, (out_w, out_h))

    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        t = frame_idx / fps

        if tracker is not None:
            frame = tracker.crop(frame, frame_idx)
        else:
            frame = _fit_pad(frame, src_w, src_h, out_w, out_h)

        overlay = _active_overlay(overlays, t)
        if overlay is not None:
            _blend(frame, overlay)
        if wm_overlay is not None:
            _blend(frame, wm_overlay)

        writer.write(frame)
        frame_idx += 1

    cap.release()
    writer.release()
    if tracker is not None:
        tracker.close()

    subprocess.run([
        "ffmpeg", "-y",
        "-i", tmp_video,
        "-i", src,
        "-map", "0:v:0", "-map", "1:a:0?",
        "-c:v", "libx264", "-preset", "fast", "-crf", "19",
        "-pix_fmt", "yuv420p",
        "-c:a", "aac", "-b:a", "160k",
        "-movflags", "+faststart",
        "-shortest",
        dst,
    ], check=True, capture_output=True)
    os.remove(tmp_video)


class _ManualCropper:
    """User-defined crop window: center position, zoom and rotation.

    crop_x/crop_y are the window center as fractions of the source frame,
    zoom scales the window down from the largest crop that fits the target
    ratio (1.0 = widest view, 4.0 = 4x punch-in), rotate is in degrees.
    """

    def __init__(self, src_w, src_h, out_w, out_h,
                 crop_x=0.5, crop_y=0.5, zoom=1.0, rotate=0.0):
        self.src_w, self.src_h = src_w, src_h
        self.out_w, self.out_h = out_w, out_h

        target_r = out_w / out_h
        src_r = src_w / src_h
        if src_r > target_r:
            base_h, base_w = src_h, src_h * target_r
        else:
            base_w, base_h = src_w, src_w / target_r

        zoom = min(max(float(zoom or 1.0), 1.0), 4.0)
        self.crop_w = max(int(base_w / zoom), 16)
        self.crop_h = max(int(base_h / zoom), 16)

        cx = min(max(float(crop_x if crop_x is not None else 0.5), 0.0), 1.0) * src_w
        cy = min(max(float(crop_y if crop_y is not None else 0.5), 0.0), 1.0) * src_h
        self.x0 = int(np.clip(cx - self.crop_w / 2, 0, src_w - self.crop_w))
        self.y0 = int(np.clip(cy - self.crop_h / 2, 0, src_h - self.crop_h))

        self.rot_matrix = None
        rotate = float(rotate or 0.0)
        if abs(rotate) > 0.1:
            center = (self.x0 + self.crop_w / 2, self.y0 + self.crop_h / 2)
            # Scale up slightly so rotation doesn't reveal black corners
            pad_scale = 1.0 + abs(np.sin(np.radians(rotate))) * 0.8
            self.rot_matrix = cv2.getRotationMatrix2D(center, rotate, pad_scale)

    def crop(self, frame, frame_idx):
        if self.rot_matrix is not None:
            frame = cv2.warpAffine(frame, self.rot_matrix, (self.src_w, self.src_h),
                                   flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
        window = frame[self.y0:self.y0 + self.crop_h, self.x0:self.x0 + self.crop_w]
        return cv2.resize(window, (self.out_w, self.out_h), interpolation=cv2.INTER_AREA)

    def close(self):
        pass


def _fit_pad(frame, src_w, src_h, out_w, out_h):
    scale = min(out_w / src_w, out_h / src_h)
    new_w, new_h = int(src_w * scale), int(src_h * scale)
    resized = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_AREA)
    canvas = np.zeros((out_h, out_w, 3), dtype=np.uint8)
    x = (out_w - new_w) // 2
    y = (out_h - new_h) // 2
    canvas[y:y + new_h, x:x + new_w] = resized
    return canvas


class _SubjectTracker:
    """Shot-aware subject reframing (how pro auto-reframe behaves).

    Instead of chasing detections frame-by-frame (which makes fast footage
    like sports/F1 swim and drift), the clip is pre-analyzed once:

      1. Scene cuts are detected, splitting the clip into shots.
      2. Each shot gets a stable crop center — the median of face /
         moving-object detections inside that shot. Long shots with a
         drifting subject get a few keyframes and pan slowly between them.
      3. Shots with no reliable subject stay center-framed.
      4. If the action is WIDER than the crop window — two cars side by
         side, a full-width on-screen graphic — cropping would cut it off,
         so that shot is rendered full-frame (letterboxed) instead.

    The crop is locked within a shot, jumps instantly at cuts, and pan
    speed is clamped — the result is steady and watchable even when the
    source is rapid and chaotic.
    """

    MODE_FILL = 0.0  # crop on the subject
    MODE_FIT = 1.0   # subject/action too wide to crop — letterbox this shot

    STRIDE = 3            # analyze every 3rd frame
    CUT_DIFF_MIN = 18.0   # floor for the adaptive scene-cut threshold
    MIN_SHOT_SEC = 0.7    # merge cuts closer than this (fast montages)
    MAX_DETECT = 8        # detection frames per shot
    KEYFRAME_SEC = 2.0    # sub-segment length for long-shot keyframes
    MAX_PAN_FRAC = 0.015  # max pan per frame, fraction of crop size

    def __init__(self, src, src_w, src_h, out_w, out_h, subject="face"):
        self.src_w, self.src_h = src_w, src_h
        self.out_w, self.out_h = out_w, out_h
        self.subject = subject

        target_r = out_w / out_h
        src_r = src_w / src_h
        if src_r > target_r:
            self.crop_h = src_h
            self.crop_w = int(src_h * target_r)
        else:
            self.crop_w = src_w
            self.crop_h = int(src_w / target_r)

        self._small_w = 320
        self._small_h = max(int(320 * src_h / src_w), 1)

        self.detector = None
        if _MP_AVAILABLE:
            try:
                self.detector = mp.solutions.face_detection.FaceDetection(
                    model_selection=1, min_detection_confidence=0.4)
            except Exception:
                self.detector = None

        self.plan = self._build_plan(src)  # (n_frames, 2) crop centers

        if self.detector is not None:
            self.detector.close()
            self.detector = None

    # -- analysis pass -----------------------------------------------------

    def _build_plan(self, src):
        cap = cv2.VideoCapture(src)
        fps = cap.get(cv2.CAP_PROP_FPS) or 25.0

        samples = []  # (frame_idx, gray_small, bgr_small)
        n_frames = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            if n_frames % self.STRIDE == 0:
                small = cv2.resize(frame, (self._small_w, self._small_h))
                samples.append((n_frames, cv2.cvtColor(small, cv2.COLOR_BGR2GRAY), small))
            n_frames += 1
        cap.release()

        default = np.array([self.src_w / 2, self.src_h / 2, self.MODE_FILL])
        if n_frames == 0 or len(samples) < 2:
            return np.tile(default, (max(n_frames, 1), 1))

        shots = self._detect_shots(samples, fps)

        plan = np.tile(default, (n_frames, 1))
        for s_start, s_end in shots:  # sample indices, inclusive
            shot_samples = samples[s_start:s_end + 1]
            keyframes, mode = self._analyze_shot(shot_samples, fps)
            f0 = shot_samples[0][0]
            f1 = shot_samples[-1][0] + self.STRIDE - 1
            f1 = min(f1, n_frames - 1)
            plan[f0:f1 + 1, 2] = mode
            if mode == self.MODE_FIT or not keyframes:
                continue  # letterboxed or center-framed
            kf_frames = np.array([k[0] for k in keyframes], dtype=float)
            kf_cx = np.array([k[1] for k in keyframes])
            kf_cy = np.array([k[2] for k in keyframes])
            frames = np.arange(f0, f1 + 1, dtype=float)
            plan[f0:f1 + 1, 0] = np.interp(frames, kf_frames, kf_cx)
            plan[f0:f1 + 1, 1] = np.interp(frames, kf_frames, kf_cy)
            self._clamp_pan_speed(plan, f0, f1)

        return plan

    def _detect_shots(self, samples, fps):
        # Color diff between consecutive samples catches chroma-only cuts
        # that grayscale misses
        diffs = np.array([
            cv2.absdiff(samples[i][2], samples[i - 1][2]).mean()
            for i in range(1, len(samples))
        ])
        if len(diffs) == 0:
            return [(0, len(samples) - 1)]

        # Adaptive threshold: a cut must stand well above the clip's own
        # motion baseline, so constant fast motion (sports pans) doesn't
        # register as cuts while real cuts still spike far above it
        baseline = float(np.median(diffs))
        threshold = max(self.CUT_DIFF_MIN, baseline * 3.5)

        min_gap = int(self.MIN_SHOT_SEC * fps / self.STRIDE)
        boundaries = [0]
        for i in range(1, len(samples)):
            if diffs[i - 1] > threshold and (i - boundaries[-1]) >= max(min_gap, 2):
                boundaries.append(i)
        boundaries.append(len(samples))
        return [(boundaries[i], boundaries[i + 1] - 1)
                for i in range(len(boundaries) - 1)]

    def _analyze_shot(self, shot_samples, fps):
        """Detect the subject in a shot.

        Returns (keyframes, mode): keyframes is [(frame_idx, cx, cy), ...]
        (empty → center-framed), mode is MODE_FILL or MODE_FIT. FIT means the
        action is too wide to crop — e.g. two cars side by side or a
        full-width on-screen graphic — so the shot must stay full-frame.
        """
        if len(shot_samples) < 2:
            return [], self.MODE_FILL

        k = min(self.MAX_DETECT, len(shot_samples))
        picks = [shot_samples[i] for i in
                 np.linspace(0, len(shot_samples) - 1, k).astype(int)]

        detections = []  # (frame_idx, cx, cy, width)
        face_hits = 0
        for frame_idx, _, bgr in picks:
            pos = self._detect_face(bgr)
            if pos is not None:
                face_hits += 1
                detections.append((frame_idx, pos[0], pos[1], pos[2]))

        # Not a face shot — measure the union of ALL moving objects against
        # this shot's own background (median of its frames, so cuts don't
        # leak in). Union, not just the largest blob: two cars racing side
        # by side are ONE piece of action and must be framed together.
        if face_hits < max(2, k // 2) and self.subject != "face":
            detections = []
            grays = [g for _, g, _ in shot_samples[:24]]
            if len(grays) >= 3:
                background = np.median(np.stack(grays), axis=0).astype(np.uint8)
                for frame_idx, gray, _ in picks:
                    box = self._detect_moving_union(gray, background)
                    if box is not None:
                        x0, y0, x1, y1 = box
                        detections.append((frame_idx, (x0 + x1) / 2,
                                           (y0 + y1) / 2, x1 - x0))

        # Too few reliable detections → keep the shot center-framed
        if len(detections) < max(2, int(0.25 * k)):
            return [], self.MODE_FILL

        frames = np.array([d[0] for d in detections], dtype=float)
        xs = np.array([d[1] for d in detections])
        ys = np.array([d[2] for d in detections])
        widths = np.array([d[3] for d in detections])

        # Would cropping cut the action off? Needed width = the subject's
        # own width plus how much its center moves around the locked frame.
        subject_w = float(np.median(widths))
        wander = float(np.percentile(xs, 85) - np.percentile(xs, 15))
        if subject_w + 0.35 * wander > 0.92 * self.crop_w:
            return [], self.MODE_FIT

        shot_sec = (frames[-1] - frames[0]) / fps
        if shot_sec <= 4.0:
            # Short shot: one locked center for the whole shot
            return [(frames[0], float(np.median(xs)), float(np.median(ys)))], self.MODE_FILL

        # Long shot: a keyframe every ~KEYFRAME_SEC so a walking/drifting
        # subject is followed slowly instead of chased
        keyframes = []
        chunk = self.KEYFRAME_SEC * fps
        t = frames[0]
        while t <= frames[-1]:
            mask = (frames >= t) & (frames < t + chunk)
            if mask.sum() >= 1:
                keyframes.append((float(frames[mask].mean()),
                                  float(np.median(xs[mask])),
                                  float(np.median(ys[mask]))))
            t += chunk
        if not keyframes:
            keyframes = [(frames[0], float(np.median(xs)), float(np.median(ys)))]
        return keyframes, self.MODE_FILL

    def _clamp_pan_speed(self, plan, f0, f1):
        max_dx = self.crop_w * self.MAX_PAN_FRAC
        max_dy = self.crop_h * self.MAX_PAN_FRAC
        for f in range(f0 + 1, f1 + 1):
            plan[f, 0] = plan[f - 1, 0] + np.clip(plan[f, 0] - plan[f - 1, 0], -max_dx, max_dx)
            plan[f, 1] = plan[f - 1, 1] + np.clip(plan[f, 1] - plan[f - 1, 1], -max_dy, max_dy)

    # -- detectors -----------------------------------------------------------

    def _detect_face(self, small_bgr):
        if self.detector is None:
            return None
        try:
            result = self.detector.process(cv2.cvtColor(small_bgr, cv2.COLOR_BGR2RGB))
        except Exception:
            return None
        if result and result.detections:
            det = max(result.detections,
                      key=lambda d: d.location_data.relative_bounding_box.width)
            bbox = det.location_data.relative_bounding_box
            return ((bbox.xmin + bbox.width / 2) * self.src_w,
                    (bbox.ymin + bbox.height / 2) * self.src_h,
                    bbox.width * self.src_w)
        return None

    def _detect_moving_union(self, gray, background):
        """Union bounding box of all significant moving objects, src coords."""
        diff = cv2.absdiff(gray, background)
        _, mask = cv2.threshold(diff, 25, 255, cv2.THRESH_BINARY)
        mask = cv2.morphologyEx(
            mask, cv2.MORPH_OPEN,
            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)))

        area = self._small_w * self._small_h
        if cv2.countNonZero(mask) > area * 0.6:
            return None  # whole-scene change, not objects

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        boxes = [cv2.boundingRect(c) for c in contours
                 if 0.002 * area <= cv2.contourArea(c) <= 0.5 * area]
        if not boxes:
            return None

        x0 = min(b[0] for b in boxes)
        y0 = min(b[1] for b in boxes)
        x1 = max(b[0] + b[2] for b in boxes)
        y1 = max(b[1] + b[3] for b in boxes)
        sx = self.src_w / self._small_w
        sy = self.src_h / self._small_h
        return (x0 * sx, y0 * sy, x1 * sx, y1 * sy)

    # -- render pass ----------------------------------------------------------

    def crop(self, frame, frame_idx):
        cx, cy, mode = self.plan[min(frame_idx, len(self.plan) - 1)]
        if mode == self.MODE_FIT:
            # Action wider than the crop window — show the full frame
            return _fit_pad(frame, self.src_w, self.src_h, self.out_w, self.out_h)
        x0 = int(np.clip(cx - self.crop_w / 2, 0, self.src_w - self.crop_w))
        # Bias slightly above center so faces/heads stay in frame
        y0 = int(np.clip(cy - self.crop_h * 0.45, 0, self.src_h - self.crop_h))
        cropped = frame[y0:y0 + self.crop_h, x0:x0 + self.crop_w]
        return cv2.resize(cropped, (self.out_w, self.out_h), interpolation=cv2.INTER_AREA)

    def close(self):
        if self.detector is not None:
            self.detector.close()


# ---------------------------------------------------------------------------
# Captions
# ---------------------------------------------------------------------------

def _build_caption_pages(words, style_name):
    """Group words into short caption pages, breaking on pauses and page size."""
    style = CAPTION_STYLES.get(style_name, CAPTION_STYLES["karaoke"])
    max_words = style["max_words"]

    pages = []
    current = []
    for w in words:
        if current:
            gap = w["start"] - current[-1]["end"]
            if len(current) >= max_words or gap > 0.7:
                pages.append(current)
                current = []
        current.append(w)
    if current:
        pages.append(current)

    result = []
    for i, page in enumerate(pages):
        start = page[0]["start"]
        # Hold the page through short pauses so captions don't flicker out
        end = page[-1]["end"] + 1.2
        if i + 1 < len(pages):
            end = min(end, pages[i + 1][0]["start"])
        result.append({"start": start, "end": end, "words": page})
    return result


def _load_font(size, bold):
    candidates = _FONT_CANDIDATES_BOLD if bold else _FONT_CANDIDATES_REGULAR
    for path, index in candidates:
        try:
            return ImageFont.truetype(path, size=size, index=index)
        except Exception:
            continue
    return ImageFont.load_default()


def _prerender_caption_overlays(pages, style_name, out_w, out_h, ratio_w, ratio_h):
    """Pre-render every (page, active word) variant as an RGBA strip.

    Returns a flat list of {start, end, img (H,W,4 uint8), y} sorted by start.
    """
    if not pages:
        return []
    style = CAPTION_STYLES.get(style_name, CAPTION_STYLES["karaoke"])
    font_size = max(int(out_h * style["size"]), 14)
    font = _load_font(font_size, style["bold"])
    active_font = font
    stroke = style["stroke"]

    strip_h = int(font_size * 2.2)
    y_frac = 0.70 if ratio_w <= ratio_h else 0.82
    strip_y = int(out_h * y_frac)
    if strip_y + strip_h > out_h:
        strip_y = out_h - strip_h

    overlays = []
    for page in pages:
        pwords = page["words"]
        # Time slices: one per active word, plus trailing slice with no active word
        slices = []
        for i, w in enumerate(pwords):
            s = w["start"] if i > 0 else page["start"]
            e = pwords[i + 1]["start"] if i + 1 < len(pwords) else w["end"]
            slices.append((s, min(e, page["end"]), i))
        if pwords[-1]["end"] < page["end"]:
            slices.append((pwords[-1]["end"], page["end"], None))

        cache = {}
        for s, e, active_idx in slices:
            if e <= s:
                continue
            if active_idx not in cache:
                cache[active_idx] = _draw_caption_strip(
                    pwords, active_idx, style, font, active_font,
                    stroke, out_w, strip_h)
            overlays.append({"start": s, "end": e, "img": cache[active_idx], "y": strip_y})

    overlays.sort(key=lambda o: o["start"])
    return overlays


def _draw_caption_strip(pwords, active_idx, style, font, active_font, stroke, out_w, strip_h):
    img = Image.new("RGBA", (out_w, strip_h), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)

    texts = [(w["word"].upper() if style["upper"] else w["word"]) for w in pwords]
    space_w = draw.textlength(" ", font=font)
    widths = [draw.textlength(t, font=font) for t in texts]
    total_w = sum(widths) + space_w * (len(texts) - 1)

    # Wrap-safe: shrink font if the line is wider than 92% of frame
    if total_w > out_w * 0.92:
        scale = (out_w * 0.92) / total_w
        font = _load_font(max(int(font.size * scale), 12), style["bold"])
        space_w = draw.textlength(" ", font=font)
        widths = [draw.textlength(t, font=font) for t in texts]
        total_w = sum(widths) + space_w * (len(texts) - 1)

    x = (out_w - total_w) / 2
    y = (strip_h - font.size) / 2

    if style["bg_box"]:
        pad = font.size * 0.45
        draw.rounded_rectangle(
            [x - pad, y - pad * 0.6, x + total_w + pad, y + font.size + pad * 0.6],
            radius=int(font.size * 0.35), fill=(0, 0, 0, 150))

    for i, (t, w_px) in enumerate(zip(texts, widths)):
        color = style["active_color"] if i == active_idx else style["base_color"]
        if stroke:
            draw.text((x, y), t, font=font, fill=color + (255,),
                      stroke_width=stroke, stroke_fill=(0, 0, 0, 255))
        else:
            draw.text((x + 1, y + 1), t, font=font, fill=(0, 0, 0, 160))
            draw.text((x, y), t, font=font, fill=color + (255,))
        x += w_px + space_w

    return np.array(img)  # RGBA


def _prerender_watermark(text, out_w, out_h):
    font = _load_font(max(int(out_h * 0.024), 12), bold=True)
    strip_h = int(font.size * 1.8)
    img = Image.new("RGBA", (out_w, strip_h), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)
    x = int(out_w * 0.04)
    y = (strip_h - font.size) // 2
    draw.text((x + 1, y + 1), text, font=font, fill=(0, 0, 0, 140))
    draw.text((x, y), text, font=font, fill=(255, 255, 255, 185))
    return {"start": -1, "end": 1e12, "img": np.array(img), "y": int(out_h * 0.035)}


def _active_overlay(overlays, t):
    for o in overlays:
        if o["start"] <= t < o["end"]:
            return o
        if o["start"] > t:
            break
    return None


def _blend(frame, overlay):
    """Alpha-blend an RGBA overlay strip onto a BGR frame in place."""
    img = overlay["img"]
    y = overlay["y"]
    h = img.shape[0]
    h = min(h, frame.shape[0] - y)
    if h <= 0:
        return
    rgba = img[:h]
    alpha = rgba[:, :, 3:4].astype(np.float32) / 255.0
    if alpha.max() == 0:
        return
    rgb = rgba[:, :, :3][:, :, ::-1].astype(np.float32)  # RGBA -> BGR
    region = frame[y:y + h].astype(np.float32)
    frame[y:y + h] = (region * (1 - alpha) + rgb * alpha).astype(np.uint8)


# ---------------------------------------------------------------------------
# Selection / helpers
# ---------------------------------------------------------------------------

def _resolve_duration(req):
    if req.duration_preset == "custom":
        return req.duration_min or 15, req.duration_max or 60
    return DURATION_PRESETS.get(req.duration_preset, (30, 60))


def _resolve_ratio(req):
    if req.aspect_ratio == "custom":
        return req.ratio_w or 9, req.ratio_h or 16
    return RATIO_PRESETS.get(req.aspect_ratio, (9, 16))


def _trim_video(video_path, start, end):
    tmp = tempfile.mktemp(suffix="_trimmed.mp4")
    subprocess.run([
        "ffmpeg", "-y",
        "-ss", str(start), "-to", str(end),
        "-i", video_path,
        "-c:v", "libx264", "-preset", "fast", "-crf", "18",
        "-c:a", "aac",
        "-avoid_negative_ts", "make_zero",
        tmp,
    ], check=True, capture_output=True)
    return tmp


def _compute_output_size_from_ratio(ratio_w, ratio_h):
    if ratio_w < ratio_h:
        out_h = 1080
        out_w = int(out_h * ratio_w / ratio_h)
    elif ratio_w == ratio_h:
        out_w = out_h = 1080
    else:
        out_w = 1920
        out_h = int(out_w * ratio_h / ratio_w)
    # libx264 requires even dimensions
    return out_w + (out_w % 2), out_h + (out_h % 2)


def _get_video_duration(video_path: str) -> float:
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    cap.release()
    return frames / fps if fps else 0.0


def _select_candidates(segments, dur_min, dur_max, video_duration, num_clips):
    used_ranges = []
    results = []

    for seg in segments:
        if len(results) >= num_clips:
            break

        seg_dur = seg["end"] - seg["start"]
        start, end = seg["start"], seg["end"]
        if seg_dur < dur_min:
            expand = (dur_min - seg_dur) / 2
            start = max(0, start - expand)
            end = min(video_duration, end + expand)
        if (end - start) > dur_max:
            end = start + dur_max

        overlap = any(not (end <= r[0] or start >= r[1]) for r in used_ranges)
        if overlap:
            continue

        seg_copy = dict(seg)
        seg_copy["start"] = round(start, 2)
        seg_copy["end"] = round(end, 2)
        results.append(seg_copy)
        used_ranges.append((start, end))

    return results


def _get_words_in_range(transcript, start, end):
    words = []
    for seg in transcript:
        if seg["end"] < start or seg["start"] > end:
            continue
        for w in seg.get("words", []):
            if w["start"] >= start and w["end"] <= end:
                words.append({
                    "word": w["word"],
                    "start": w["start"] - start,
                    "end": w["end"] - start,
                })
    return words


def _generate_title(words, fallback_text):
    if not words:
        return (fallback_text or "Highlight")[:60]
    title = " ".join(w["word"] for w in words[:8])
    return re.sub(r"[^\w\s']", "", title).strip()[:60] or "Highlight"


def _make_thumbnail(clip_path, thumb_path, duration):
    try:
        subprocess.run([
            "ffmpeg", "-y",
            "-ss", str(max(duration * 0.2, 0.5)),
            "-i", clip_path,
            "-vframes", "1",
            "-vf", "scale=360:-2",
            "-q:v", "4",
            thumb_path,
        ], check=True, capture_output=True)
    except Exception:
        pass
