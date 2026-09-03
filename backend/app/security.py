"""Authentication: password hashing, opaque session tokens, FastAPI guards.

PBKDF2-HMAC-SHA256 from the standard library rather than bcrypt/argon2 so the
install has no compiler dependency on Windows.  310k iterations matches the
current OWASP guidance for PBKDF2-SHA256.
"""
from __future__ import annotations

import base64
import hashlib
import hmac
import re
import secrets
import time
from typing import Any

from fastapi import Depends, Header, HTTPException, Request, status

from . import db
from .config import SETTINGS

PBKDF2_ROUNDS = 310_000
EMAIL_RE = re.compile(r"^[^@\s]+@[^@\s]+\.[a-zA-Z]{2,}$")


# ------------------------------------------------------------------ password

def hash_password(password: str) -> str:
    salt = secrets.token_bytes(16)
    dk = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt, PBKDF2_ROUNDS)
    return "pbkdf2$%d$%s$%s" % (
        PBKDF2_ROUNDS,
        base64.b64encode(salt).decode(),
        base64.b64encode(dk).decode(),
    )


def verify_password(password: str, stored: str) -> bool:
    try:
        scheme, rounds, salt_b64, dk_b64 = stored.split("$")
        if scheme != "pbkdf2":
            return False
        salt = base64.b64decode(salt_b64)
        expected = base64.b64decode(dk_b64)
        dk = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt, int(rounds))
        return hmac.compare_digest(dk, expected)
    except (ValueError, TypeError):
        return False


def password_problems(password: str) -> list[str]:
    out = []
    if len(password) < 8:
        out.append("La contrasena debe tener al menos 8 caracteres.")
    if password.isdigit():
        out.append("La contrasena no puede ser solo numeros.")
    return out


def valid_email(email: str) -> bool:
    return bool(EMAIL_RE.match(email or ""))


# ------------------------------------------------------------------ sessions

def create_session(user_id: str, user_agent: str = "") -> tuple[str, float]:
    token = secrets.token_urlsafe(32)
    created = db.now()
    expires = created + SETTINGS.session_days * 86400
    db.execute(
        "INSERT INTO sessions(token,user_id,created_at,expires_at,user_agent) "
        "VALUES(?,?,?,?,?)",
        (token, user_id, created, expires, (user_agent or "")[:300]),
    )
    return token, expires


def destroy_session(token: str) -> None:
    db.execute("DELETE FROM sessions WHERE token=?", (token,))


def purge_expired_sessions() -> None:
    db.execute("DELETE FROM sessions WHERE expires_at < ?", (db.now(),))


def user_for_token(token: str) -> dict | None:
    if not token:
        return None
    row = db.q1(
        "SELECT u.*, s.token AS session_token, s.expires_at AS session_expires "
        "FROM sessions s JOIN users u ON u.id = s.user_id "
        "WHERE s.token=? AND s.expires_at > ?",
        (token, db.now()),
    )
    return db.row_to_dict(row)


def _token_from_request(request: Request, authorization: str | None) -> str:
    if authorization and authorization.lower().startswith("bearer "):
        return authorization[7:].strip()
    cookie = request.cookies.get("pr_session")
    if cookie:
        return cookie
    return request.headers.get("x-session-token", "")


# ---------------------------------------------------------------- FastAPI deps

def current_user(request: Request,
                 authorization: str | None = Header(default=None)) -> dict:
    token = _token_from_request(request, authorization)
    user = user_for_token(token)
    if not user:
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "Sesion no valida o expirada")
    if user.get("status") == "suspended":
        raise HTTPException(status.HTTP_403_FORBIDDEN, "Cuenta suspendida")
    return user


def active_user(user: dict = Depends(current_user)) -> dict:
    """A user who has been approved by an administrator."""
    if user.get("status") != "active":
        raise HTTPException(
            status.HTTP_403_FORBIDDEN,
            "Tu cuenta esta pendiente de aprobacion por el administrador",
        )
    return user


def admin_user(user: dict = Depends(current_user)) -> dict:
    if user.get("role") != "admin":
        raise HTTPException(status.HTTP_403_FORBIDDEN, "Solo administradores")
    return user


def optional_user(request: Request,
                  authorization: str | None = Header(default=None)) -> dict | None:
    try:
        return current_user(request, authorization)
    except HTTPException:
        return None


# ------------------------------------------------------------- rate limiting

_BUCKETS: dict[str, list[float]] = {}


def rate_limit(key: str, limit: int, window_s: float) -> bool:
    """Tiny in-process sliding window.  Enough for a single node deployment."""
    now = time.time()
    bucket = [t for t in _BUCKETS.get(key, []) if now - t < window_s]
    if len(bucket) >= limit:
        _BUCKETS[key] = bucket
        return False
    bucket.append(now)
    _BUCKETS[key] = bucket
    return True


def client_ip(request: Request) -> str:
    fwd = request.headers.get("x-forwarded-for")
    if fwd:
        return fwd.split(",")[0].strip()
    return request.client.host if request.client else "unknown"


def public_user(user: dict[str, Any]) -> dict:
    """Strip anything that must never reach the browser."""
    safe = {k: v for k, v in user.items()
            if k not in {"password_hash", "session_token", "session_expires"}}
    return safe
