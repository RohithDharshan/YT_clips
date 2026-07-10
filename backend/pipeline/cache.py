import json
import os
import time

CACHE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../cache/analysis"))
TTL = 3600 * 24  # 24 hours — keeps the editor usable for a full working day


def _path(job_id: str) -> str:
    os.makedirs(CACHE_DIR, exist_ok=True)
    return os.path.join(CACHE_DIR, f"{job_id}.json")


def set_cache(job_id: str, data: dict):
    payload = {"ts": time.time(), "data": data}
    with open(_path(job_id), "w") as f:
        json.dump(payload, f)


def get_cache(job_id: str) -> dict | None:
    p = _path(job_id)
    if not os.path.exists(p):
        return None
    with open(p) as f:
        payload = json.load(f)
    if time.time() - payload["ts"] > TTL:
        os.remove(p)
        return None
    return payload["data"]
