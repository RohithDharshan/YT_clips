"""Central configuration: plan limits, CORS, and environment-driven secrets."""

import os

from dotenv import load_dotenv

load_dotenv(os.path.join(os.path.dirname(__file__), ".env"))

ENV = os.environ.get("ENV", "development")

# Comma-separated list of allowed browser origins. In production this MUST
# be set explicitly — "*" cannot be combined with credentialed requests
# (Authorization headers) in real browsers, and is a wide-open CORS hole.
_origins = os.environ.get("ALLOWED_ORIGINS", "")
ALLOWED_ORIGINS = [o.strip() for o in _origins.split(",") if o.strip()] or [
    "http://localhost:3000", "http://localhost:3005",
]

MAX_UPLOAD_BYTES = int(os.environ.get("MAX_UPLOAD_BYTES", 1024 * 1024 * 1024))  # 1 GB

PLAN_LIMITS = {
    "free": {
        "videos_per_month": 5,
        "max_source_minutes": 15,
        "max_clips_per_video": 3,
        "max_resolution": 720,     # long-edge cap in px
        "force_watermark": True,
        "label": "Free",
    },
    "pro": {
        "videos_per_month": 50,
        "max_source_minutes": 90,
        "max_clips_per_video": 10,
        "max_resolution": 1080,
        "force_watermark": False,
        "label": "Pro",
    },
}

PRICING = {
    "free": {"price_usd": 0, "period": "forever"},
    # Deliberately low — the priority right now is reach, not revenue.
    # Actual display pricing is regional/PPP-adjusted; see
    # frontend/pricing.html's REGION_PRICE (standard/regional/lower tiers:
    # $7 / $5 / $3 per month, $5 / $4 / $2 annual). These backend numbers
    # are the "standard" tier and only matter once real billing is wired up.
    "pro": {"price_usd": 7, "period": "month", "annual_price_usd": 5},
}
