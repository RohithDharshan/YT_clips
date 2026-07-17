"""Email + password authentication for ClipMind.

SQLite-backed users and sessions; passwords hashed with PBKDF2-HMAC-SHA256
(200k iterations, per-user salt). Session tokens are 256-bit random values
with a 30-day expiry.
"""

import hashlib
import os
import re
import secrets
import sqlite3
import time

DB_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "../cache/users.db"))
TOKEN_TTL = 30 * 86400  # 30 days
EMAIL_RE = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]{2,}$")


class AuthError(Exception):
    """Raised with a user-facing message."""


def _conn():
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    conn.execute("""CREATE TABLE IF NOT EXISTS users(
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        email TEXT UNIQUE NOT NULL,
        name TEXT NOT NULL,
        pw_hash BLOB NOT NULL,
        salt BLOB NOT NULL,
        created REAL NOT NULL)""")
    conn.execute("""CREATE TABLE IF NOT EXISTS sessions(
        token TEXT PRIMARY KEY,
        user_id INTEGER NOT NULL,
        expires REAL NOT NULL)""")
    return conn


def _hash(password: str, salt: bytes) -> bytes:
    return hashlib.pbkdf2_hmac("sha256", password.encode(), salt, 200_000)


def _new_session(conn, user_id: int) -> str:
    token = secrets.token_hex(32)
    conn.execute("INSERT INTO sessions(token, user_id, expires) VALUES(?,?,?)",
                 (token, user_id, time.time() + TOKEN_TTL))
    conn.commit()
    return token


def signup(name: str, email: str, password: str) -> dict:
    name = (name or "").strip()
    email = (email or "").strip().lower()
    if not name or len(name) > 80:
        raise AuthError("Please enter your name")
    if not EMAIL_RE.match(email):
        raise AuthError("Please enter a valid email address")
    if len(password or "") < 8:
        raise AuthError("Password must be at least 8 characters")

    salt = secrets.token_bytes(16)
    conn = _conn()
    try:
        cur = conn.execute(
            "INSERT INTO users(email, name, pw_hash, salt, created) VALUES(?,?,?,?,?)",
            (email, name, _hash(password, salt), salt, time.time()))
    except sqlite3.IntegrityError:
        conn.close()
        raise AuthError("An account with this email already exists")
    token = _new_session(conn, cur.lastrowid)
    conn.close()
    return {"token": token, "name": name, "email": email}


def login(email: str, password: str) -> dict:
    email = (email or "").strip().lower()
    conn = _conn()
    row = conn.execute(
        "SELECT id, name, pw_hash, salt FROM users WHERE email=?", (email,)).fetchone()
    if not row or not secrets.compare_digest(row[2], _hash(password or "", row[3])):
        conn.close()
        raise AuthError("Incorrect email or password")
    token = _new_session(conn, row[0])
    conn.close()
    return {"token": token, "name": row[1], "email": email}


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
        """SELECT u.id, u.name, u.email, s.expires FROM sessions s
           JOIN users u ON u.id = s.user_id WHERE s.token=?""", (token,)).fetchone()
    if row and row[3] < time.time():
        conn.execute("DELETE FROM sessions WHERE token=?", (token,))
        conn.commit()
        row = None
    conn.close()
    if not row:
        return None
    return {"id": row[0], "name": row[1], "email": row[2]}
