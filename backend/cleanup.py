"""Retention cleanup for ClipMind's local disk state.

Run periodically (e.g. a daily cron hitting `python cleanup.py`) to bound
disk usage: rendered clips, uploaded source videos, exported ZIPs/frames, and
downloaded YouTube source videos all accumulate indefinitely otherwise. The
analysis cache already expires itself via pipeline/cache.py's TTL; this
handles everything else.
"""

import os
import re
import shutil
import time

RETENTION_DAYS = int(os.environ.get("RETENTION_DAYS", 30))
BASE = os.path.dirname(__file__)
CACHE_DIR = os.path.abspath(os.path.join(BASE, "../cache"))
# Deliberately excludes the cache/ root's own files (jobs.json, users.db,
# backend.log live there) — only transient, regenerable artifacts are swept.
DIRS = ["../clips", "../cache/uploads", "../cache/exports", "../cache/analysis"]

# run_youtube_pipeline() downloads each job to cache/<job_id>/ — a UUID-named
# directory sitting directly in cache/, separate from the DIRS above. These
# hold the full downloaded source video and are easy to miss in a sweep that
# only looks at known subdirectory names.
_UUID_RE = re.compile(r"^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$")


def _is_stale(path: str, cutoff: float) -> bool:
    try:
        return os.path.getmtime(path) < cutoff
    except OSError:
        return False


def _dir_size(path: str) -> int:
    total = 0
    for dirpath, _, filenames in os.walk(path):
        for f in filenames:
            try:
                total += os.path.getsize(os.path.join(dirpath, f))
            except OSError:
                pass
    return total


def clean():
    cutoff = time.time() - RETENTION_DAYS * 86400
    removed, freed = 0, 0

    for rel in DIRS:
        d = os.path.abspath(os.path.join(BASE, rel))
        if not os.path.isdir(d):
            continue
        for name in os.listdir(d):
            path = os.path.join(d, name)
            if os.path.isfile(path) and _is_stale(path, cutoff):
                try:
                    freed += os.path.getsize(path)
                    os.remove(path)
                    removed += 1
                except OSError:
                    pass

    if os.path.isdir(CACHE_DIR):
        for name in os.listdir(CACHE_DIR):
            if not _UUID_RE.match(name):
                continue
            path = os.path.join(CACHE_DIR, name)
            if os.path.isdir(path) and _is_stale(path, cutoff):
                try:
                    freed += _dir_size(path)
                    shutil.rmtree(path)
                    removed += 1
                except OSError:
                    pass

    print(f"cleanup: removed {removed} items, freed {freed / (1024*1024):.1f} MB "
          f"(older than {RETENTION_DAYS} days)")


if __name__ == "__main__":
    clean()
