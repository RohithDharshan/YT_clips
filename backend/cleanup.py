"""Retention cleanup for ClipMind's local disk state.

Run periodically (e.g. a daily cron hitting `python cleanup.py`) to bound
disk usage: rendered clips, uploaded source videos, and exported ZIPs/frames
all accumulate indefinitely otherwise. The analysis cache already expires
itself via pipeline/cache.py's TTL; this handles everything else.
"""

import os
import time

RETENTION_DAYS = int(os.environ.get("RETENTION_DAYS", 30))
BASE = os.path.dirname(__file__)
# Deliberately excludes the cache/ root itself (jobs.json, users.db live
# there) — only transient, regenerable artifacts are swept.
DIRS = ["../clips", "../cache/uploads", "../cache/exports", "../cache/analysis"]


def _is_stale(path: str, cutoff: float) -> bool:
    try:
        return os.path.getmtime(path) < cutoff
    except OSError:
        return False


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

    print(f"cleanup: removed {removed} files, freed {freed / (1024*1024):.1f} MB "
          f"(older than {RETENTION_DAYS} days)")


if __name__ == "__main__":
    clean()
