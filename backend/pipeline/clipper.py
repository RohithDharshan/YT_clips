import os
import subprocess
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import mediapipe as mp
import tempfile
import re


CLIPS_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../clips"))
WATERMARK_TEXT = "Project Ray"

DURATION_PRESETS = {
    "<30s": (15, 30),
    "<1min": (30, 60),
}

RATIO_PRESETS = {
    "9:16": (9, 16),
    "1:1": (1, 1),
    "16:9": (16, 9),
}


def generate_clips(job_id: str, analysis: dict, req) -> list:
    os.makedirs(CLIPS_DIR, exist_ok=True)

    duration_min, duration_max = _resolve_duration(req)
    ratio_w, ratio_h = _resolve_ratio(req)

    segments = analysis["segments"]
    video_path = analysis["video_path"]
    transcript = analysis["transcript"]

    video_duration = _get_video_duration(video_path)
    candidates = _select_candidates(segments, duration_min, duration_max, video_duration, req.num_clips)

    results = []
    for rank, cand in enumerate(candidates, 1):
        clip_id = f"{job_id}_clip{rank}"
        clip_path = os.path.join(CLIPS_DIR, f"{clip_id}.mp4")

        words_in_range = _get_words_in_range(transcript, cand["start"], cand["end"])
        title = _generate_title(words_in_range, cand["text"])

        temp_path = _trim_video(video_path, cand["start"], cand["end"])
        reframed_path = _reframe(temp_path, ratio_w, ratio_h)
        _add_watermark(reframed_path, clip_path, ratio_w, ratio_h)

        for p in [temp_path, reframed_path]:
            if os.path.exists(p) and p != clip_path:
                os.remove(p)

        results.append({
            "rank": rank,
            "clip_url": f"/clips/{clip_id}.mp4",
            "title": title,
            "caption": cand["text"][:200],
            "score": cand["score"],
            "start": cand["start"],
            "end": cand["end"],
            "duration": round(cand["end"] - cand["start"], 1),
        })

    return results


def _resolve_duration(req):
    if req.duration_preset == "custom":
        return req.duration_min or 15, req.duration_max or 60
    return DURATION_PRESETS.get(req.duration_preset, (30, 60))


def _resolve_ratio(req):
    if req.aspect_ratio == "custom":
        return req.ratio_w or 9, req.ratio_h or 16
    w, h = RATIO_PRESETS.get(req.aspect_ratio, (9, 16))
    return w, h


def _get_video_duration(video_path: str) -> float:
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    cap.release()
    return frames / fps


def _select_candidates(segments, dur_min, dur_max, video_duration, num_clips):
    used_ranges = []
    results = []

    for seg in segments:
        if len(results) >= num_clips:
            break

        seg_dur = seg["end"] - seg["start"]

        # Expand short segments by merging neighbors
        start, end = seg["start"], seg["end"]
        if seg_dur < dur_min:
            expand = (dur_min - seg_dur) / 2
            start = max(0, start - expand)
            end = min(video_duration, end + expand)

        # Trim long segments from the end
        if (end - start) > dur_max:
            end = start + dur_max

        # Skip if overlaps an already-used range
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
                words.append({"word": w["word"], "start": w["start"] - start, "end": w["end"] - start})
    return words


def _generate_title(words, fallback_text):
    if not words:
        return fallback_text[:60]
    title = " ".join(w["word"] for w in words[:8])
    return re.sub(r"[^\w\s']", "", title).strip()[:60]


def _trim_video(video_path, start, end):
    tmp = tempfile.mktemp(suffix="_trimmed.mp4")
    subprocess.run([
        "ffmpeg", "-y",
        "-ss", str(start), "-to", str(end),
        "-i", video_path,
        "-c:v", "libx264", "-c:a", "aac",
        "-avoid_negative_ts", "make_zero",
        tmp
    ], check=True, capture_output=True)
    return tmp


def _reframe(video_path, ratio_w, ratio_h):
    """Scale the full video to fit target ratio with black bars — no cropping, no distortion."""
    target_w, target_h = _compute_output_size_from_ratio(ratio_w, ratio_h)
    tmp_out = tempfile.mktemp(suffix="_reframed.mp4")

    # ffmpeg pad filter: scale to fit inside target box, then pad with black to fill
    vf = (
        f"scale={target_w}:{target_h}:force_original_aspect_ratio=decrease,"
        f"pad={target_w}:{target_h}:(ow-iw)/2:(oh-ih)/2:black"
    )
    subprocess.run([
        "ffmpeg", "-y", "-i", video_path,
        "-vf", vf,
        "-c:v", "libx264", "-preset", "fast", "-crf", "18",
        "-c:a", "aac",
        tmp_out
    ], check=True, capture_output=True)

    return tmp_out


def _detect_hud_zones(video_path, orig_w, orig_h):
    """Sample frames to find persistent static regions (overlays/tickers/scoreboards)."""
    SCALE = 0.25
    sw, sh = int(orig_w * SCALE), int(orig_h * SCALE)
    cap = cv2.VideoCapture(video_path)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0

    samples = []
    step = max(1, total // 20)
    for i in range(0, min(total, step * 20), step):
        cap.set(cv2.CAP_PROP_POS_FRAMES, i)
        ret, frame = cap.read()
        if ret:
            samples.append(cv2.resize(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), (sw, sh)).astype(float))
    cap.release()

    if len(samples) < 3:
        return np.zeros((sh, sw), dtype=np.uint8)

    stack = np.stack(samples, axis=0)
    # Pixels with very low variance across frames = static overlay
    variance = stack.var(axis=0)
    static_mask = (variance < 8).astype(np.uint8) * 255
    # Dilate slightly so nearby pixels are also excluded
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (9, 9))
    static_mask = cv2.dilate(static_mask, kernel)
    return static_mask  # at SCALE resolution


def _subject_bbox(small, bg_sub, hud_mask, sw, sh, orig_w, orig_h):
    """Return (cx, cy, zoom) for the main moving subject, excluding HUD zones."""
    mask = bg_sub.apply(small)
    # Remove HUD areas
    if hud_mask is not None and hud_mask.shape == mask.shape:
        mask = cv2.bitwise_and(mask, cv2.bitwise_not(hud_mask))

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_DILATE, kernel)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        # Largest foreground contour = main subject
        largest = max(contours, key=cv2.contourArea)
        bx, by, bw, bh = cv2.boundingRect(largest)

        cx = int((bx + bw / 2) / sw * orig_w)
        cy = int((by + bh / 2) / sh * orig_h)

        # Zoom = subject covers ~60% of frame width — pad to 1.8× subject size
        subject_w_full = bw / sw * orig_w
        zoom = min(max((subject_w_full * 1.8) / orig_w, 0.35), 1.0)
        return (cx, cy, zoom)

    return (orig_w // 2, orig_h // 2, 0.85)


def _detect_face_bbox(frame, mp_face):
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = mp_face.process(rgb)
    if result.detections:
        det = result.detections[0]
        bbox = det.location_data.relative_bounding_box
        h, w = frame.shape[:2]
        fx = int((bbox.xmin + bbox.width / 2) * w)
        fy = int((bbox.ymin + bbox.height / 2) * h)
        fw = int(bbox.width * w)
        fh = int(bbox.height * h)
        return (fx, fy, fw, fh)
    return None


def _compute_output_size(orig_w, orig_h, ratio_w, ratio_h):
    if ratio_w < ratio_h:
        out_h = 1080
        out_w = int(out_h * ratio_w / ratio_h)
    elif ratio_w == ratio_h:
        out_w = out_h = 1080
    else:
        out_w = 1920
        out_h = int(out_w * ratio_h / ratio_w)
    return out_w, out_h


def _compute_output_size_from_ratio(ratio_w, ratio_h):
    if ratio_w < ratio_h:
        out_h = 1080
        out_w = int(out_h * ratio_w / ratio_h)
    elif ratio_w == ratio_h:
        out_w = out_h = 1080
    else:
        out_w = 1920
        out_h = int(out_w * ratio_h / ratio_w)
    # Ensure both dimensions are even (required by libx264)
    out_w = out_w + (out_w % 2)
    out_h = out_h + (out_h % 2)
    return out_w, out_h


def _burn_captions(video_path, words, ratio_w, ratio_h):
    if not words:
        return video_path

    cap = cv2.VideoCapture(video_path)
    out_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    out_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    cap.release()

    tmp_video = tempfile.mktemp(suffix="_captioned_noaudio.mp4")
    tmp_out = tempfile.mktemp(suffix="_captioned.mp4")

    try:
        font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", size=int(out_h * 0.045))
    except Exception:
        font = ImageFont.load_default()

    cap = cv2.VideoCapture(video_path)
    writer = cv2.VideoWriter(tmp_video, cv2.VideoWriter_fourcc(*"mp4v"), fps, (out_w, out_h))

    cap_y = int(out_h * 0.78) if ratio_w <= ratio_h else int(out_h * 0.85)
    frame_idx = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        t = frame_idx / fps
        active_words = [w for w in words if w["start"] <= t <= w["end"]]
        window_words = [w for w in words if t - 3 < w["start"] <= t]

        if window_words:
            img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            draw = ImageDraw.Draw(img)

            line = " ".join(w["word"] for w in window_words)
            active_set = {w["word"] for w in active_words}

            x_cursor = out_w // 2
            total_w = sum(draw.textlength(w["word"] + " ", font=font) for w in window_words)
            x_cursor = int((out_w - total_w) / 2)

            for w in window_words:
                word_w = draw.textlength(w["word"] + " ", font=font)
                color = (255, 220, 0) if w["word"] in active_set else (255, 255, 255)
                # Shadow
                draw.text((x_cursor + 2, cap_y + 2), w["word"], font=font, fill=(0, 0, 0, 180))
                draw.text((x_cursor, cap_y), w["word"], font=font, fill=color)
                x_cursor += int(word_w)

            frame = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)

        writer.write(frame)
        frame_idx += 1

    cap.release()
    writer.release()

    subprocess.run([
        "ffmpeg", "-y",
        "-i", tmp_video,
        "-i", video_path,
        "-c:v", "copy", "-c:a", "aac",
        "-map", "0:v:0", "-map", "1:a:0",
        "-shortest",
        tmp_out
    ], check=True, capture_output=True)

    os.remove(tmp_video)
    return tmp_out


def _add_watermark(input_path, output_path, ratio_w, ratio_h):
    cap = cv2.VideoCapture(input_path)
    out_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    out_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    cap.release()

    tmp_video = tempfile.mktemp(suffix="_wm_noaudio.mp4")

    try:
        font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", size=int(out_h * 0.025))
    except Exception:
        font = ImageFont.load_default()

    wm_x = int(out_w * 0.04)
    wm_y = int(out_h * 0.04)

    cap = cv2.VideoCapture(input_path)
    writer = cv2.VideoWriter(tmp_video, cv2.VideoWriter_fourcc(*"mp4v"), fps, (out_w, out_h))

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        draw = ImageDraw.Draw(img)
        draw.text((wm_x + 1, wm_y + 1), WATERMARK_TEXT, font=font, fill=(0, 0, 0, 160))
        draw.text((wm_x, wm_y), WATERMARK_TEXT, font=font, fill=(255, 255, 255, 200))
        writer.write(cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR))

    cap.release()
    writer.release()

    subprocess.run([
        "ffmpeg", "-y",
        "-i", tmp_video,
        "-i", input_path,
        "-c:v", "libx264", "-c:a", "aac",
        "-map", "0:v:0", "-map", "1:a:0",
        "-shortest",
        output_path
    ], check=True, capture_output=True)

    os.remove(tmp_video)
