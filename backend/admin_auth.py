"""Admin authentication — deliberately separate from the Google-only user auth.

Users sign in exclusively via Google (see auth.py); admins are operators of
the platform (currently: you), not customers, so a username/password login
makes sense here even though it doesn't for the customer-facing app.
Passwords are PBKDF2-HMAC-SHA256 hashed (200k iterations, per-admin salt) —
never stored or transmitted in plaintext after creation.
"""

import hashlib
import os
import secrets
import sqlite3
import time

DB_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "../cache/users.db"))
TOKEN_TTL = 7 * 86400  # shorter-lived than user sessions — re-auth weekly


class AdminAuthError(Exception):
    pass


def _conn():
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    conn.execute("""CREATE TABLE IF NOT EXISTS admins(
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        username TEXT UNIQUE NOT NULL,
        pw_hash BLOB NOT NULL,
        salt BLOB NOT NULL,
        created REAL NOT NULL)""")
    conn.execute("""CREATE TABLE IF NOT EXISTS admin_sessions(
        token TEXT PRIMARY KEY,
        admin_id INTEGER NOT NULL,
        expires REAL NOT NULL)""")
    return conn


def _hash(password: str, salt: bytes) -> bytes:
    return hashlib.pbkdf2_hmac("sha256", password.encode(), salt, 200_000)


def bootstrap_admin() -> tuple[str, str] | None:
    """Create the first admin account if none exists yet.

    Returns (username, plaintext_password) exactly once — the moment of
    creation is the only time the plaintext password exists anywhere.
    Returns None if an admin already exists (idempotent — safe to call on
    every server startup).
    """
    conn = _conn()
    exists = conn.execute("SELECT 1 FROM admins LIMIT 1").fetchone()
    if exists:
        conn.close()
        return None

    username = "admin"
    password = secrets.token_urlsafe(12)  # ~16 chars, URL-safe, high entropy
    salt = secrets.token_bytes(16)
    conn.execute(
        "INSERT INTO admins(username, pw_hash, salt, created) VALUES(?,?,?,?)",
        (username, _hash(password, salt), salt, time.time()))
    conn.commit()
    conn.close()
    return username, password


def login(username: str, password: str) -> str:
    conn = _conn()
    row = conn.execute(
        "SELECT id, pw_hash, salt FROM admins WHERE username=?", (username,)).fetchone()
    if not row or not secrets.compare_digest(row[1], _hash(password or "", row[2])):
        conn.close()
        raise AdminAuthError("Incorrect username or password")
    token = secrets.token_hex(32)
    conn.execute("INSERT INTO admin_sessions(token, admin_id, expires) VALUES(?,?,?)",
                 (token, row[0], time.time() + TOKEN_TTL))
    conn.commit()
    conn.close()
    return token


def logout(token: str):
    conn = _conn()
    conn.execute("DELETE FROM admin_sessions WHERE token=?", (token,))
    conn.commit()
    conn.close()


def get_admin(token: str) -> dict | None:
    if not token:
        return None
    conn = _conn()
    row = conn.execute(
        """SELECT a.id, a.username, s.expires FROM admin_sessions s
           JOIN admins a ON a.id = s.admin_id WHERE s.token=?""", (token,)).fetchone()
    if row and row[2] < time.time():
        conn.execute("DELETE FROM admin_sessions WHERE token=?", (token,))
        conn.commit()
        row = None
    conn.close()
    if not row:
        return None
    return {"id": row[0], "username": row[1]}


def change_password(admin_id: int, new_password: str):
    if len(new_password or "") < 12:
        raise AdminAuthError("Password must be at least 12 characters")
    salt = secrets.token_bytes(16)
    conn = _conn()
    conn.execute("UPDATE admins SET pw_hash=?, salt=? WHERE id=?",
                 (_hash(new_password, salt), salt, admin_id))
    conn.commit()
    conn.close()
