import os
import shutil
import subprocess
import sys

_BROWSERS = ["chrome", "firefox", "edge", "chromium", "safari"]

# Resolve yt-dlp binary (always use the venv one)
_YTDLP = os.path.join(os.path.dirname(sys.executable), "yt-dlp")
if not os.path.exists(_YTDLP):
    _YTDLP = shutil.which("yt-dlp") or "yt-dlp"

# Node.js for JS challenge solving
_NODE = (
    shutil.which("node")
    or "/Users/mrohithdharshan/.local/bin/node"
    or "/usr/local/bin/node"
    or "node"
)

_FORMAT = (
    "best[height<=1080][ext=mp4]"
    "/bestvideo[height<=1080][ext=mp4]+bestaudio[ext=m4a]"
    "/best[height<=1080]"
    "/best"
)


def _base_cmd(output_template: str) -> list[str]:
    return [
        _YTDLP,
        "--format", _FORMAT,
        "--output", output_template,
        "--merge-output-format", "mp4",
        "--js-runtimes", f"node:{_NODE}",
        "--remote-components", "ejs:github",
        "--retries", "5",
        "--fragment-retries", "5",
        "--quiet",
        "--no-warnings",
    ]


def download_youtube(url: str, output_dir: str) -> str:
    os.makedirs(output_dir, exist_ok=True)
    output_template = os.path.join(output_dir, "video.%(ext)s")

    cookies_file = os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..", "youtube_cookies.txt")
    )

    # Try order: each browser → cookies file → no cookies
    attempts: list[tuple[str, list[str]]] = []
    for browser in _BROWSERS:
        attempts.append((f"browser:{browser}", ["--cookies-from-browser", browser]))
    if os.path.exists(cookies_file):
        attempts.append(("cookiefile", ["--cookies", cookies_file]))
    attempts.append(("no-cookies", []))

    last_error = ""
    for label, cookie_args in attempts:
        cmd = _base_cmd(output_template) + cookie_args + [url]
        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=300,
                env={**os.environ, "PATH": f"/Users/mrohithdharshan/.local/bin:/usr/local/bin:/opt/homebrew/bin:/usr/bin:/bin:{os.environ.get('PATH','')}"},
            )
            if result.returncode == 0:
                for f in os.listdir(output_dir):
                    if f.startswith("video") and (f.endswith(".mp4") or f.endswith(".mkv")):
                        return os.path.join(output_dir, f)
                # File might have been converted — check again
                for f in os.listdir(output_dir):
                    if not f.endswith(".part"):
                        return os.path.join(output_dir, f)

            stderr = result.stderr + result.stdout
            retryable = any(w in stderr for w in [
                "Sign In", "sign in", "bot", "Bot", "confirm", "403",
                "login", "age", "private", "Operation not permitted",
                "decrypt", "Permission"
            ])
            if retryable:
                last_error = stderr.strip().splitlines()[-1] if stderr.strip() else "Auth error"
                continue
            # Non-auth error — raise immediately
            raise RuntimeError(f"Download failed [{label}]: {stderr.strip()[-300:]}")

        except subprocess.TimeoutExpired:
            last_error = "Download timed out"
            continue
        except OSError as e:
            last_error = str(e)
            continue

    raise RuntimeError(
        f"YouTube blocked this download after trying all cookie sources. "
        f"Make sure you are logged into YouTube in Chrome. Last error: {last_error}"
    )
