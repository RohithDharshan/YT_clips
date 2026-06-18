import os
import subprocess
from faster_whisper import WhisperModel


def transcribe_video(video_path: str) -> tuple[list, str]:
    audio_path = video_path.rsplit(".", 1)[0] + ".wav"
    if not os.path.exists(audio_path):
        subprocess.run(
            ["ffmpeg", "-y", "-i", video_path, "-ac", "1", "-ar", "16000", audio_path],
            check=True, capture_output=True,
        )

    model = WhisperModel("base", device="cpu", compute_type="int8")
    raw_segments, _ = model.transcribe(audio_path, word_timestamps=True)

    segments = []
    for i, seg in enumerate(raw_segments):
        words = [
            {"word": w.word.strip(), "start": w.start, "end": w.end}
            for w in (seg.words or [])
        ]
        segments.append({
            "id": i,
            "start": seg.start,
            "end": seg.end,
            "text": seg.text.strip(),
            "words": words,
        })

    return segments, audio_path
