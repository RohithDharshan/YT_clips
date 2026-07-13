"""Subject analysis — figures out what the video should focus on.

Samples frames across the whole video and measures:
  - face presence (MediaPipe face detection)
  - dominant moving object (frame differencing, largest contour)

The result drives the "auto" framing mode and is shown to the user so they
can choose between a normal full-frame render and a subject-focused crop.
"""

import cv2
import numpy as np

try:
    import mediapipe as mp
    _MP_AVAILABLE = True
except Exception:
    _MP_AVAILABLE = False

SAMPLES = 24
FACE_RATE_THRESHOLD = 0.30    # faces in ≥30% of sampled frames → talking-head video
OBJECT_HIT_THRESHOLD = 0.40   # a dominant blob present in ≥40% of samples → trackable subject
OBJECT_MIN_FRAC = 0.004       # blob must cover ≥0.4% of the frame ...
OBJECT_MAX_FRAC = 0.45        # ... but not half the frame (that's a scene change / camera move)


def analyze_focus(video_path: str) -> dict:
    cap = cv2.VideoCapture(video_path)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total < 2:
        cap.release()
        return _result("none", 0.0, 0.0)

    idxs = np.linspace(0, total - 1, min(SAMPLES, total)).astype(int)

    detector = None
    if _MP_AVAILABLE:
        try:
            detector = mp.solutions.face_detection.FaceDetection(
                model_selection=1, min_detection_confidence=0.5)
        except Exception:
            detector = None

    face_hits = 0
    grays = []
    smalls = []

    for i in idxs:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(i))
        ret, frame = cap.read()
        if not ret:
            continue

        h, w = frame.shape[:2]
        small_w = 320
        small_h = max(int(small_w * h / w), 1)
        small = cv2.resize(frame, (small_w, small_h))
        smalls.append(small)
        grays.append(cv2.cvtColor(small, cv2.COLOR_BGR2GRAY))

        if detector is not None:
            try:
                res = detector.process(cv2.cvtColor(small, cv2.COLOR_BGR2RGB))
                if res.detections:
                    face_hits += 1
            except Exception:
                pass

    cap.release()
    if detector is not None:
        detector.close()

    if len(grays) < 3:
        return _result("none", 0.0, 0.0)

    face_rate = face_hits / len(grays)

    # Median frame ≈ static background; a moving subject stands out fully in
    # each frame no matter how slowly it moves (consecutive-frame diffs only
    # catch its edges). Backgrounds are built per *chunk* of neighboring
    # samples so multi-shot videos (sports broadcasts) don't blend unrelated
    # scenes into one meaningless background.
    area = grays[0].shape[0] * grays[0].shape[1]
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))

    CHUNK = 6
    valid_hits = 0
    tested = 0
    for c0 in range(0, len(grays), CHUNK):
        chunk = grays[c0:c0 + CHUNK]
        if len(chunk) < 3:
            continue
        background = np.median(np.stack(chunk), axis=0).astype(np.uint8)
        for g in chunk:
            tested += 1
            diff = cv2.absdiff(g, background)
            _, th = cv2.threshold(diff, 25, 255, cv2.THRESH_BINARY)
            th = cv2.morphologyEx(th, cv2.MORPH_OPEN, kernel)
            contours, _ = cv2.findContours(th, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if not contours:
                continue
            frac = max(cv2.contourArea(c) for c in contours) / area
            if OBJECT_MIN_FRAC <= frac <= OBJECT_MAX_FRAC:
                valid_hits += 1

    object_rate = valid_hits / tested if tested else 0.0

    if face_rate >= FACE_RATE_THRESHOLD:
        subject = "face"
    elif object_rate >= OBJECT_HIT_THRESHOLD:
        subject = "object"
    else:
        subject = "none"

    return _result(subject, face_rate, object_rate)


def _result(subject: str, face_rate: float, object_rate: float) -> dict:
    labels = {
        "face": f"Faces detected in {int(face_rate * 100)}% of the video — Focus mode will track the speaker",
        "object": "Moving subject detected — Focus mode will track the action",
        "none": "No clear subject — full-frame (Normal) recommended",
    }
    return {
        "subject": subject,
        "face_rate": round(face_rate, 3),
        "object_rate": round(object_rate, 4),
        "recommend": "fill" if subject in ("face", "object") else "fit",
        "label": labels[subject],
    }
