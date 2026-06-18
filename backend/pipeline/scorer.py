import cv2
import numpy as np
import librosa


# Generic excitement keywords
HIGHLIGHT_KEYWORDS = [
    "but", "however", "actually", "wait", "stop", "listen", "honestly",
    "secret", "never", "always", "truth", "mistake", "wrong", "imagine",
    "shocking", "reveal", "surprise", "incredible", "amazing", "insane",
    "what if", "the thing is", "here's why", "this is why",
]

# Climax / winning-moment keywords — heavily weighted
CLIMAX_KEYWORDS = [
    # Race wins / finish
    "wins", "winner", "victory", "champion", "championship",
    "checkered", "takes the win", "wins the race", "wins a grand prix",
    "wins the grand prix", "wins a race", "grand prix victory",
    "finishes first", "takes p1", "takes the lead", "podium",
    "fastest lap", "new record", "history", "legend",
    # Drama / incidents
    "overtake", "overtakes", "passes", "goes past", "move of the race",
    "incredible move", "what a move", "unbelievable", "extraordinary",
    "crashes", "collision", "safety car", "red flag", "out of nowhere",
    "drama", "dramatic", "stunning", "sensational", "retire", "retires",
    # Generic sports
    "goal", "scores", "touchdown", "knockout", "finish",
    # Emotional peaks
    "dream", "proud", "couldn't do this", "thank you", "love you",
    "what a moment", "moment",
]


def score_segments(video_path: str, audio_path: str, transcript: list) -> list:
    audio_scores, excitement_scores = _score_audio(audio_path, transcript)
    scene_scores = _score_scenes(video_path, transcript)
    text_scores, climax_scores = _score_text(transcript)

    scored = []
    for i, seg in enumerate(transcript):
        start, end = seg["start"], seg["end"]
        duration = end - start

        if duration < 2:
            continue

        audio_s = audio_scores.get(i, 0.0)
        excite_s = excitement_scores.get(i, 0.0)
        scene_s = scene_scores.get(i, 0.0)
        text_s = text_scores.get(i, 0.0)
        climax_s = climax_scores.get(i, 0.0)

        # Climax score dominates — winning/dramatic moments always rise to top
        score = (
            audio_s   * 0.15 +
            excite_s  * 0.15 +
            scene_s   * 0.10 +
            text_s    * 0.10 +
            climax_s  * 0.50
        )

        scored.append({
            "seg_index": i,
            "start": start,
            "end": end,
            "text": seg["text"],
            "words": seg.get("words", []),
            "score": round(score, 4),
            "audio_score": round(audio_s, 4),
            "scene_score": round(scene_s, 4),
            "text_score": round(text_s, 4),
            "climax_score": round(climax_s, 4),
        })

    return sorted(scored, key=lambda x: x["score"], reverse=True)


def _score_audio(audio_path: str, transcript: list) -> tuple[dict, dict]:
    y, sr = librosa.load(audio_path, sr=16000, mono=True)

    # RMS energy
    rms = librosa.feature.rms(y=y, frame_length=512, hop_length=256)[0]
    times = librosa.frames_to_time(np.arange(len(rms)), sr=sr, hop_length=256)
    rms_norm = (rms - rms.min()) / (rms.max() - rms.min() + 1e-8)

    # Spectral flux — measures sudden audio energy changes (crowd roar, impact sounds)
    S = np.abs(librosa.stft(y, hop_length=256))
    flux = np.sqrt(np.mean(np.diff(S, axis=1) ** 2, axis=0))
    flux = np.pad(flux, (0, 1))
    flux_norm = (flux - flux.min()) / (flux.max() - flux.min() + 1e-8)
    flux_times = librosa.frames_to_time(np.arange(len(flux_norm)), sr=sr, hop_length=256)

    rms_scores, excite_scores = {}, {}
    for i, seg in enumerate(transcript):
        mask = (times >= seg["start"]) & (times <= seg["end"])
        rms_scores[i] = float(np.percentile(rms_norm[mask], 80)) if mask.sum() else 0.0

        fmask = (flux_times >= seg["start"]) & (flux_times <= seg["end"])
        excite_scores[i] = float(np.percentile(flux_norm[fmask], 90)) if fmask.sum() else 0.0

    return rms_scores, excite_scores


def _score_scenes(video_path: str, transcript: list) -> dict:
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0

    # Sample every 5th frame for speed
    cut_times = []
    motion_energy = {}  # time -> mean frame diff
    prev_gray = None
    frame_idx = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if frame_idx % 5 == 0:
            gray = cv2.resize(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), (320, 180))
            if prev_gray is not None:
                diff = cv2.absdiff(gray, prev_gray).mean()
                t = frame_idx / fps
                motion_energy[t] = diff
                if diff > 20:
                    cut_times.append(t)
            prev_gray = gray
        frame_idx += 1

    cap.release()

    scores = {}
    for i, seg in enumerate(transcript):
        cuts = sum(1 for t in cut_times if seg["start"] <= t <= seg["end"])
        duration = max(seg["end"] - seg["start"], 1)
        cut_rate = min(cuts / duration / 2.0, 1.0)

        # Also score by average motion energy in this segment
        seg_motion = [v for t, v in motion_energy.items() if seg["start"] <= t <= seg["end"]]
        motion_s = min(np.mean(seg_motion) / 30.0, 1.0) if seg_motion else 0.0

        scores[i] = cut_rate * 0.5 + motion_s * 0.5

    return scores


def _score_text(transcript: list) -> tuple[dict, dict]:
    text_scores, climax_scores = {}, {}

    for i, seg in enumerate(transcript):
        text_lower = seg["text"].lower()

        kw_hits = sum(1 for kw in HIGHLIGHT_KEYWORDS if kw in text_lower)
        climax_hits = sum(1 for kw in CLIMAX_KEYWORDS if kw in text_lower)

        has_question = "?" in seg["text"]
        has_exclamation = "!" in seg["text"]
        word_count = len(seg["text"].split())
        density = min(word_count / 30.0, 1.0)

        raw = (kw_hits * 0.10) + (has_question * 0.15) + (has_exclamation * 0.15) + (density * 0.60)
        text_scores[i] = min(raw, 1.0)

        # Climax score: first hit = 0.7, second = 1.0, capped at 1.0
        climax_scores[i] = min(climax_hits * 0.50, 1.0)

    return text_scores, climax_scores
