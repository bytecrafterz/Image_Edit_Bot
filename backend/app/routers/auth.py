"""Registration, login, session.

The first account ever created becomes the administrator and is active at once;
every later account waits for approval.  That makes a fresh install usable by
one person immediately while keeping the multi-person future the client asked
for from being open to anyone who finds the URL.
"""
from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, Request, Response, status
from pydantic import BaseModel, Field

from .. import db, security
from ..config import SETTINGS
from ..services import billing

router = APIRouter(prefix="/auth", tags=["auth"])

COOKIE = "pr_session"


class RegisterBody(BaseModel):
    email: str
    password: str
    display_name: str = Field(default="", max_length=80)


class LoginBody(BaseModel):
    email: str
    password: str


class PasswordBody(BaseModel):
    current_password: str
    new_password: str


def _set_cookie(response: Response, token: str) -> None:
    response.set_cookie(
        COOKIE, token, max_age=SETTINGS.session_days * 86400,
        httponly=True, samesite="lax", path="/",
    )


@router.post("/register")
def register(body: RegisterBody, request: Request, response: Response) -> dict:
    ip = security.client_ip(request)
    if not security.rate_limit(f"register:{ip}", 5, 900):
        raise HTTPException(status.HTTP_429_TOO_MANY_REQUESTS,
                            "Demasiados intentos. Espera unos minutos.")

    email = (body.email or "").strip().lower()
    if not security.valid_email(email):
        raise HTTPException(400, "Ese correo no parece valido.")
    problems = security.password_problems(body.password or "")
    if problems:
        raise HTTPException(400, " ".join(problems))
    if db.q1("SELECT id FROM users WHERE email=?", (email,)):
        raise HTTPException(409, "Ya existe una cuenta con ese correo.")

    first = db.q1("SELECT COUNT(*) AS n FROM users")
    is_first = int(first["n"] or 0) == 0 if first else True

    user_id = db.new_id("usr")
    now = db.now()
    db.execute(
        "INSERT INTO users(id,email,password_hash,display_name,role,status,"
        "daily_limit_usd,monthly_limit_usd,created_at,approved_at) "
        "VALUES(?,?,?,?,?,?,?,?,?,?)",
        (user_id, email, security.hash_password(body.password),
         (body.display_name or "").strip() or email.split("@")[0],
         "admin" if is_first else "user",
         "active" if is_first else "pending",
         SETTINGS.limits.default_daily_usd, SETTINGS.limits.default_monthly_usd,
         now, now if is_first else None),
    )
    db.audit("auth.register", user_id, email=email, first=is_first)

    user = db.row_to_dict(db.q1("SELECT * FROM users WHERE id=?", (user_id,)))
    if not is_first:
        return {"user": security.public_user(user), "needs_approval": True,
                "message": ("Tu cuenta se ha creado. Un administrador tiene que "
                            "aprobarla antes de que puedas entrar.")}

    token, expires = security.create_session(
        user_id, request.headers.get("user-agent", ""))
    _set_cookie(response, token)
    return {"user": security.public_user(user), "token": token,
            "expires_at": expires, "needs_approval": False}


@router.post("/login")
def login(body: LoginBody, request: Request, response: Response) -> dict:
    ip = security.client_ip(request)
    email = (body.email or "").strip().lower()
    if not (security.rate_limit(f"login:{ip}", 10, 300)
            and security.rate_limit(f"login:{email}", 10, 300)):
        raise HTTPException(status.HTTP_429_TOO_MANY_REQUESTS,
                            "Demasiados intentos. Espera unos minutos.")

    row = db.q1("SELECT * FROM users WHERE email=?", (email,))
    user = db.row_to_dict(row)
    if not user or not security.verify_password(body.password or "",
                                                user["password_hash"]):
        raise HTTPException(401, "Correo o contrasena incorrectos.")
    if user["status"] == "suspended":
        raise HTTPException(403, "Tu cuenta esta suspendida.")
    if user["status"] == "pending":
        raise HTTPException(403, "Tu cuenta esta pendiente de aprobacion.")

    token, expires = security.create_session(
        user["id"], request.headers.get("user-agent", ""))
    db.execute("UPDATE users SET last_login_at=? WHERE id=?",
               (db.now(), user["id"]))
    security.purge_expired_sessions()
    _set_cookie(response, token)
    db.audit("auth.login", user["id"])
    return {"user": security.public_user(user), "token": token,
            "expires_at": expires}


@router.post("/logout")
def logout(request: Request, response: Response,
           user: dict = Depends(security.current_user)) -> dict:
    token = user.get("session_token")
    if token:
        security.destroy_session(token)
    response.delete_cookie(COOKIE, path="/")
    return {"ok": True}


@router.get("/me")
def me(user: dict = Depends(security.current_user)) -> dict:
    profile = db.row_to_dict(db.q1(
        "SELECT id, person_name, status FROM profiles WHERE user_id=? "
        "AND deleted_at IS NULL ORDER BY is_default DESC, updated_at DESC LIMIT 1",
        (user["id"],)))
    unread = db.q1("SELECT COUNT(*) AS n FROM alerts WHERE user_id=? "
                   "AND read_at IS NULL", (user["id"],))
    return {
        "user": security.public_user(user),
        "balances": billing.all_balances(user["id"]),
        "alerts_unread": int(unread["n"] or 0) if unread else 0,
        "default_profile": profile,
    }


@router.post("/password")
def change_password(body: PasswordBody,
                    user: dict = Depends(security.current_user)) -> dict:
    if not security.verify_password(body.current_password or "",
                                    user["password_hash"]):
        raise HTTPException(403, "La contrasena actual no es correcta.")
    problems = security.password_problems(body.new_password or "")
    if problems:
        raise HTTPException(400, " ".join(problems))
    db.execute("UPDATE users SET password_hash=? WHERE id=?",
               (security.hash_password(body.new_password), user["id"]))
    db.execute("DELETE FROM sessions WHERE user_id=? AND token<>?",
               (user["id"], user.get("session_token") or ""))
    db.audit("auth.password_changed", user["id"])
    return {"ok": True, "message": "Contrasena actualizada."}
