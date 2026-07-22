"""Google Sign-In authentication for ClipMind.

Users authenticate exclusively via Google (no passwords stored anywhere).
The frontend obtains a Google ID token via Google Identity Services; we
verify it server-side against Google's public keys, restrict it to a
configured email domain (gmail.com by default), and mint our own opaque
session token — the ID token itself is never treated as a session credential.
"""

import os
import secrets
import sqlite3
import time

from google.auth.transport import requests as google_requests
from google.oauth2 import id_token as google_id_token

DB_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "../cache/users.db"))
TOKEN_TTL = 30 * 86400  # 30 days

GOOGLE_CLIENT_ID = os.environ.get("GOOGLE_CLIENT_ID", "")
ALLOWED_EMAIL_DOMAIN = os.environ.get("ALLOWED_EMAIL_DOMAIN", "gmail.com").lower().lstrip("@")

_google_request = google_requests.Request()


class AuthError(Exception):
    """Raised with a user-facing message."""


def _migrate_legacy_schema():
    """One-time move-aside of the old password-based schema.

    Earlier versions stored pw_hash/salt directly on users. Google Sign-In
    replaces that entirely, so an old database is incompatible (NOT NULL
    columns we no longer populate) rather than upgradeable in place. Those
    were dev/test accounts, not production user data, so we move the file
    aside instead of mutating it in place.
    """
    if not os.path.exists(DB_PATH):
        return
    conn = sqlite3.connect(DB_PATH)
    try:
        cols = {row[1] for row in conn.execute("PRAGMA table_info(users)")}
    finally:
        conn.close()
    if "pw_hash" in cols:
        backup = DB_PATH + f".legacy-{int(time.time())}"
        os.rename(DB_PATH, backup)


_migrate_legacy_schema()


def _conn():
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    conn.execute("""CREATE TABLE IF NOT EXISTS users(
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        email TEXT UNIQUE NOT NULL,
        name TEXT NOT NULL,
        avatar_url TEXT,
        google_sub TEXT UNIQUE NOT NULL,
        plan TEXT NOT NULL DEFAULT 'free',
        created REAL NOT NULL,
        last_login REAL NOT NULL)""")
    conn.execute("""CREATE TABLE IF NOT EXISTS sessions(
        token TEXT PRIMARY KEY,
        user_id INTEGER NOT NULL,
        expires REAL NOT NULL)""")
    conn.execute("""CREATE TABLE IF NOT EXISTS usage(
        user_id INTEGER NOT NULL,
        period TEXT NOT NULL,
        videos INTEGER NOT NULL DEFAULT 0,
        minutes REAL NOT NULL DEFAULT 0,
        PRIMARY KEY (user_id, period))""")
    return conn


def _new_session(conn, user_id: int) -> str:
    token = secrets.token_hex(32)
    conn.execute("INSERT INTO sessions(token, user_id, expires) VALUES(?,?,?)",
                 (token, user_id, time.time() + TOKEN_TTL))
    conn.commit()
    return token


def login_with_google(id_token_str: str) -> dict:
    """Verify a Google ID token and return {token, name, email, avatar, plan}."""
    if not GOOGLE_CLIENT_ID:
        raise AuthError("Google sign-in is not configured on this server yet")
    if not id_token_str:
        raise AuthError("Missing Google credential")

    try:
        payload = google_id_token.verify_oauth2_token(
            id_token_str, _google_request, GOOGLE_CLIENT_ID)
    except ValueError:
        raise AuthError("Invalid or expired Google sign-in — please try again")

    if not payload.get("email_verified", False):
        raise AuthError("This Google account's email is not verified")

    email = (payload.get("email") or "").strip().lower()
    domain = email.rsplit("@", 1)[-1] if "@" in email else ""
    if ALLOWED_EMAIL_DOMAIN and domain != ALLOWED_EMAIL_DOMAIN:
        raise AuthError(f"Please sign in with a @{ALLOWED_EMAIL_DOMAIN} account")

    sub = payload["sub"]
    name = payload.get("name") or email.split("@")[0]
    avatar = payload.get("picture")
    now = time.time()

    conn = _conn()
    row = conn.execute("SELECT id FROM users WHERE google_sub=?", (sub,)).fetchone()
    if row:
        user_id = row[0]
        conn.execute(
            "UPDATE users SET name=?, avatar_url=?, last_login=? WHERE id=?",
            (name, avatar, now, user_id))
    else:
        try:
            cur = conn.execute(
                """INSERT INTO users(email, name, avatar_url, google_sub, plan, created, last_login)
                   VALUES(?,?,?,?,?,?,?)""",
                (email, name, avatar, sub, "free", now, now))
            user_id = cur.lastrowid
        except sqlite3.IntegrityError:
            conn.close()
            raise AuthError("An account with this email already exists")

    token = _new_session(conn, user_id)
    plan = conn.execute("SELECT plan FROM users WHERE id=?", (user_id,)).fetchone()[0]
    conn.commit()
    conn.close()
    return {"token": token, "name": name, "email": email, "avatar": avatar, "plan": plan}


def logout(token: str):
    conn = _conn()
    conn.execute("DELETE FROM sessions WHERE token=?", (token,))
    conn.commit()
    conn.close()


def get_user(token: str) -> dict | None:
    if not token:
        return None
    conn = _conn()
    row = conn.execute(
        """SELECT u.id, u.name, u.email, u.avatar_url, u.plan, s.expires
           FROM sessions s JOIN users u ON u.id = s.user_id WHERE s.token=?""",
        (token,)).fetchone()
    if row and row[5] < time.time():
        conn.execute("DELETE FROM sessions WHERE token=?", (token,))
        conn.commit()
        row = None
    conn.close()
    if not row:
        return None
    return {"id": row[0], "name": row[1], "email": row[2], "avatar": row[3], "plan": row[4]}


def get_user_by_id(user_id: int) -> dict | None:
    conn = _conn()
    row = conn.execute(
        "SELECT id, name, email, avatar_url, plan FROM users WHERE id=?",
        (user_id,)).fetchone()
    conn.close()
    if not row:
        return None
    return {"id": row[0], "name": row[1], "email": row[2], "avatar": row[3], "plan": row[4]}


def delete_account(user_id: int):
    conn = _conn()
    conn.execute("DELETE FROM sessions WHERE user_id=?", (user_id,))
    conn.execute("DELETE FROM usage WHERE user_id=?", (user_id,))
    conn.execute("DELETE FROM users WHERE id=?", (user_id,))
    conn.commit()
    conn.close()


# ---------------------------------------------------------------------------
# Usage / plan
# ---------------------------------------------------------------------------

def _period() -> str:
    return time.strftime("%Y-%m")


def get_usage(user_id: int) -> dict:
    conn = _conn()
    row = conn.execute(
        "SELECT videos, minutes FROM usage WHERE user_id=? AND period=?",
        (user_id, _period())).fetchone()
    conn.close()
    return {"videos": row[0] if row else 0, "minutes": row[1] if row else 0.0}


def record_usage(user_id: int, minutes: float):
    conn = _conn()
    period = _period()
    conn.execute(
        """INSERT INTO usage(user_id, period, videos, minutes) VALUES(?,?,1,?)
           ON CONFLICT(user_id, period) DO UPDATE SET
             videos = videos + 1, minutes = minutes + excluded.minutes""",
        (user_id, period, minutes))
    conn.commit()
    conn.close()


def set_plan(user_id: int, plan: str):
    if plan not in ("free", "pro"):
        raise AuthError("Unknown plan")
    conn = _conn()
    conn.execute("UPDATE users SET plan=? WHERE id=?", (plan, user_id))
    conn.commit()
    conn.close()
