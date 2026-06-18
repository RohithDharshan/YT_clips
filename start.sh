#!/bin/bash
set -e

DIR="$(cd "$(dirname "$0")" && pwd)"
BACKEND="$DIR/backend"
FRONTEND="$DIR/frontend"
CLIPS="$DIR/clips"
CACHE="$DIR/cache"

mkdir -p "$CLIPS" "$CACHE/uploads" "$CACHE/analysis"

echo "🔍 Checking dependencies..."

# Prefer Python 3.12 (ML libraries require ≤3.12)
PYTHON=""
for candidate in python3.12 python3.11 python3.10 python3; do
  if command -v "$candidate" &>/dev/null; then
    ver=$("$candidate" -c "import sys; print(sys.version_info[:2])")
    if [[ "$ver" == "(3, 10)" || "$ver" == "(3, 11)" || "$ver" == "(3, 12)" ]]; then
      PYTHON="$candidate"
      break
    fi
  fi
done

if [ -z "$PYTHON" ]; then
  echo "❌ Python 3.10–3.12 required (torch/whisper don't support 3.13+)."
  echo "   Install via: brew install python@3.12"
  exit 1
fi
echo "🐍 Using $($PYTHON --version)"

# Check ffmpeg
if ! command -v ffmpeg &>/dev/null; then
  echo "⚙️  ffmpeg not found — installing via brew..."
  brew install ffmpeg
fi

# Check/create venv
if [ ! -d "$BACKEND/.venv" ]; then
  echo "📦 Creating virtual environment..."
  "$PYTHON" -m venv "$BACKEND/.venv"
fi

source "$BACKEND/.venv/bin/activate"

echo "📦 Installing Python dependencies..."
pip install -q --upgrade pip setuptools wheel
# faster-whisper has prebuilt wheels — install before requirements.txt
pip install -q faster-whisper
pip install -q -U yt-dlp  # keep current — YouTube changes frequently
# Install torch CPU build (needs special index)
pip install -q torch torchaudio --index-url https://download.pytorch.org/whl/cpu
# Install remaining dependencies
pip install -q -r "$BACKEND/requirements.txt"

echo "🚀 Starting Project Ray API..."
cd "$BACKEND"
uvicorn main:app --host 0.0.0.0 --port 8000 --reload &
API_PID=$!

echo "🌐 Starting frontend server..."
cd "$FRONTEND"
python3 -m http.server 3000 &
FRONTEND_PID=$!

echo ""
echo "✅ Project Ray is running!"
echo "   Frontend → http://localhost:3000"
echo "   API      → http://localhost:8000"
echo ""
echo "Press Ctrl+C to stop."

cleanup() {
  kill $API_PID $FRONTEND_PID 2>/dev/null
  echo "Stopped."
}
trap cleanup INT TERM
wait
