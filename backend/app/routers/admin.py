"""Administration.

An administrator manages accounts, approvals, limits and the health of the
installation.  What they deliberately cannot do here is look at anyone else's
photographs: every endpoint returns counts and metadata, never image bytes.
"""
from __future__ import annotations

import secrets

from pathlib import Path

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel

from .. import config, db, security
from ..catalog import seed as seed_mod
from ..generation import orchestrator as orchestrator_mod
from ..identity import onboarding as onboarding_mod
from ..services import billing, storage

router = APIRouter(prefix="/admin", tags=["admin"])


# The providers that hold money.  A recharge on the free engine is a number
# with nothing behind it, so it is refused rather than written.
PAID_PROVIDERS = ("fal", "anthropic")


class AdminRecharge(BaseModel):
    provider: str
    amount_usd: float
    note: str = ""


class PatchUser(BaseModel):
    role: str | None = None
    plan: str | None = None
    status: str | None = None
    daily_limit_usd: float | None = None
    monthly_limit_usd: float | None = None
    free_quota_daily: int | None = None
    notes: str | None = None


class PurgeBody(BaseModel):
    days: int = 30


def _user_row(user_id: str) -> dict:
    row = db.row_to_dict(db.q1("SELECT * FROM users WHERE id=?", (user_id,)))
    if not row:
        raise HTTPException(404, "Ese usuario no existe.")
    return row


@router.get("/users")
def list_users(q: str = "", status: str = "",
               admin: dict = Depends(security.admin_user)) -> dict:
    sql = "SELECT * FROM users WHERE 1=1"
    params: list = []
    if q:
        sql += " AND (email LIKE ? OR display_name LIKE ?)"
        params += [f"%{q}%", f"%{q}%"]
    if status in ("pending", "active", "suspended"):
        sql += " AND status=?"
        params.append(status)
    sql += " ORDER BY created_at DESC"

    users = []
    for row in db.rows_to_dicts(db.q(sql, params)):
        stats = db.q1(
            "SELECT (SELECT COUNT(*) FROM originals WHERE user_id=? "
            "AND deleted_at IS NULL) AS originals, "
            "(SELECT COUNT(*) FROM images WHERE user_id=? AND deleted_at IS NULL) "
            "AS images, (SELECT COALESCE(SUM(cost_usd),0) FROM attempts "
            "WHERE user_id=? AND created_at >= ?) AS spend_30d",
            (row["id"], row["id"], row["id"], db.now() - 30 * 86400))
        users.append({
            **security.public_user(row),
            "originals": int(stats["originals"] or 0) if stats else 0,
            "images": int(stats["images"] or 0) if stats else 0,
            "spend_30d": round(float(stats["spend_30d"] or 0.0), 4) if stats else 0,
            # What each account can still spend, next to what it spent: the
            # administrator tops up from this screen and has to see the
            # number she is topping up.
            "balances": {p: round(billing.balance(row["id"], p), 4)
                         for p in PAID_PROVIDERS},
        })
    return {"users": users, "total": len(users)}


@router.post("/users/{user_id}/approve")
def approve(user_id: str, admin: dict = Depends(security.admin_user)) -> dict:
    _user_row(user_id)
    db.execute("UPDATE users SET status='active', approved_at=? WHERE id=?",
               (db.now(), user_id))
    db.audit("admin.approve", user_id, actor=admin["email"])
    return {"ok": True}


@router.post("/users/{user_id}/suspend")
def suspend(user_id: str, admin: dict = Depends(security.admin_user)) -> dict:
    if user_id == admin["id"]:
        raise HTTPException(400, "No puedes suspender tu propia cuenta.")
    _user_row(user_id)
    db.execute("UPDATE users SET status='suspended' WHERE id=?", (user_id,))
    db.execute("DELETE FROM sessions WHERE user_id=?", (user_id,))
    db.audit("admin.suspend", user_id, actor=admin["email"])
    return {"ok": True}


@router.post("/users/{user_id}/reactivate")
def reactivate(user_id: str, admin: dict = Depends(security.admin_user)) -> dict:
    """Undo a suspension.  The sessions were deleted on suspend; she logs in."""
    row = _user_row(user_id)
    if row.get("status") != "suspended":
        raise HTTPException(400, "Esa cuenta no esta suspendida.")
    db.execute("UPDATE users SET status='active' WHERE id=?", (user_id,))
    db.audit("admin.reactivate", user_id, actor=admin["email"])
    return {"ok": True}


@router.post("/users/{user_id}/recharge")
def recharge_user(user_id: str, body: AdminRecharge,
                  admin: dict = Depends(security.admin_user)) -> dict:
    """Write down money added at the provider for ANOTHER account.

    Same ledger row the account's own /settings/recharge writes, same rule:
    nothing here moves money, it records money that was already moved on the
    provider's website.  The note says who wrote it, because a balance that
    appeared with no name on it is a question the administrator will be
    asked later.
    """
    _user_row(user_id)
    provider = body.provider.strip().lower()
    if provider not in PAID_PROVIDERS:
        raise HTTPException(400, "Solo se anota saldo en %s."
                            % " o ".join(PAID_PROVIDERS))
    note = "anotado por %s" % admin["email"]
    if body.note.strip():
        note += ": " + body.note.strip()[:200]
    try:
        result = billing.recharge(user_id, provider, float(body.amount_usd),
                                  note=note)
    except ValueError as exc:
        raise HTTPException(400, str(exc))
    db.audit("admin.recharge", user_id, actor=admin["email"],
             provider=provider, amount=float(body.amount_usd))
    return result


@router.patch("/users/{user_id}")
def patch_user(user_id: str, body: PatchUser,
               admin: dict = Depends(security.admin_user)) -> dict:
    _user_row(user_id)
    fields, params = [], []
    if body.role is not None:
        if body.role not in ("user", "admin"):
            raise HTTPException(400, "Rol no valido.")
        if user_id == admin["id"] and body.role != "admin":
            raise HTTPException(400, "No puedes quitarte a ti mismo el rol de "
                                     "administrador.")
        fields.append("role=?")
        params.append(body.role)
    if body.plan is not None:
        if body.plan not in ("free", "paid"):
            raise HTTPException(400, "Plan no valido.")
        fields.append("plan=?")
        params.append(body.plan)
    if body.status is not None:
        if body.status not in ("pending", "active", "suspended"):
            raise HTTPException(400, "Estado no valido.")
        fields.append("status=?")
        params.append(body.status)
    for key in ("daily_limit_usd", "monthly_limit_usd", "free_quota_daily"):
        value = getattr(body, key)
        if value is not None:
            if float(value) < 0:
                raise HTTPException(400, "Los limites no pueden ser negativos.")
            fields.append(f"{key}=?")
            params.append(value)
    if body.notes is not None:
        fields.append("notes=?")
        params.append(body.notes[:1000])
    if not fields:
        raise HTTPException(400, "Nada que cambiar.")
    params.append(user_id)
    db.execute(f"UPDATE users SET {', '.join(fields)} WHERE id=?", params)
    db.audit("admin.patch_user", user_id, actor=admin["email"])
    return {"ok": True, "user": security.public_user(_user_row(user_id))}


@router.post("/users/{user_id}/reset-password")
def reset_password(user_id: str,
                   admin: dict = Depends(security.admin_user)) -> dict:
    _user_row(user_id)
    temporary = secrets.token_urlsafe(9)
    db.execute("UPDATE users SET password_hash=? WHERE id=?",
               (security.hash_password(temporary), user_id))
    db.execute("DELETE FROM sessions WHERE user_id=?", (user_id,))
    db.audit("admin.reset_password", user_id, actor=admin["email"])
    return {"ok": True, "temporary_password": temporary,
            "message": "Comparte esta contrasena una sola vez. Pide que la cambie."}


@router.delete("/users/{user_id}")
def delete_user(user_id: str,
                admin: dict = Depends(security.admin_user)) -> dict:
    if user_id == admin["id"]:
        raise HTTPException(400, "No puedes eliminar tu propia cuenta.")
    _user_row(user_id)
    for table in ("originals", "images"):
        for row in db.q(f"SELECT path, thumb_path FROM {table} WHERE user_id=?",
                        (user_id,)):
            storage.delete_file(row["path"])
            storage.delete_file(row["thumb_path"])
    db.execute("DELETE FROM users WHERE id=?", (user_id,))
    db.audit("admin.delete_user", user_id, actor=admin["email"])
    return {"ok": True}


@router.post("/users/{user_id}/rebuild-profile")
def rebuild_profile(user_id: str, admin: dict = Depends(security.admin_user)) -> dict:
    """Build - or rebuild - an account's identity profile from its photographs.

    The repair for the account that generated without a profile on
    2026-09-10, and the button for any account whose photographs changed.
    Free: the measuring is local.  Answers whether a face signature came out
    of it, because that is the thing every paid image is checked against.
    """
    user_row = _user_row(user_id)
    prof = db.row_to_dict(db.q1(
        "SELECT * FROM profiles WHERE user_id=? AND deleted_at IS NULL "
        "ORDER BY is_default DESC, updated_at DESC LIMIT 1", (user_id,))) or {}
    if not prof:
        prof = orchestrator_mod._ensure_default_profile(user_row)
    if not prof:
        raise HTTPException(400, "No se pudo crear el perfil.")
    try:
        report = onboarding_mod.build_first_run(user_row, prof["id"], force=True)
    except PermissionError as exc:
        raise HTTPException(400, str(exc))
    except Exception as exc:                              # noqa: BLE001
        raise HTTPException(500, "No se pudo construir el perfil: %s" % str(exc)[:160])
    prof = orchestrator_mod._profile_for(user_row, prof["id"])
    firma = bool((prof.get("face") or {}).get("embedding_mean"))
    fotos = int(prof.get("n_sources") or 0)
    db.audit("admin.rebuild_profile", user_id, actor=admin["email"],
             profile_id=prof["id"], fotos=fotos, firma=firma)
    return {"ok": True, "profile_id": prof["id"], "fotos": fotos,
            "firma_facial": firma, "resumen": str((report or {}).get("resumen") or "")}


@router.get("/users/{user_id}/images")
def user_images(user_id: str, kind: str = "", limit: int = 60, offset: int = 0,
                admin: dict = Depends(security.admin_user)) -> dict:
    """The images one account has generated, for the administrator.

    Same rows and same shape the account's own album gets (album._payload),
    read across the ownership line an ordinary route never crosses - so it is
    written to the audit, with who looked and at whom, every time.  Thumbnails
    travel by the same signed link the album uses; that link names a file,
    not an account, which is what lets a different browser open it.
    """
    from .album import _payload
    _user_row(user_id)
    sql = "SELECT * FROM images WHERE user_id=? AND deleted_at IS NULL"
    params: list = [user_id]
    if kind in ("preview", "final", "repair"):
        sql += " AND kind=?"
        params.append(kind)
    total = db.q1(sql.replace("SELECT *", "SELECT COUNT(*) AS n", 1), params)
    sql += " ORDER BY created_at DESC LIMIT ? OFFSET ?"
    params += [max(1, min(int(limit), 200)), max(0, int(offset))]
    rows = db.rows_to_dicts(db.q(sql, params))
    # ``kind`` is db.audit's own first parameter; the payload key is ``vista``.
    db.audit("admin.view_images", user_id, actor=admin["email"],
             vista=kind or "all", n=len(rows))
    return {"images": [_payload(r) for r in rows],
            "total": int(total["n"] or 0) if total else 0}


@router.get("/users/{user_id}/originals")
def user_originals(user_id: str, limit: int = 120, offset: int = 0,
                   admin: dict = Depends(security.admin_user)) -> dict:
    """The reference photographs one account uploaded, for the administrator.

    These are the most private files the installation holds, and the audit
    row says who looked at whose, every time.  Same signed thumbnail link the
    account's own "Mis fotos" uses.
    """
    _user_row(user_id)
    total = db.q1("SELECT COUNT(*) AS n FROM originals WHERE user_id=? "
                  "AND deleted_at IS NULL", (user_id,))
    rows = db.rows_to_dicts(db.q(
        "SELECT * FROM originals WHERE user_id=? AND deleted_at IS NULL "
        "ORDER BY sort_order, created_at LIMIT ? OFFSET ?",
        (user_id, max(1, min(int(limit), 300)), max(0, int(offset)))))
    out = []
    for row in rows:
        quality = row.get("quality") if isinstance(row.get("quality"), dict) else {}
        out.append({
            "id": row["id"], "filename": row.get("filename"),
            "shot_type": row.get("shot_type"),
            "width": row.get("width"), "height": row.get("height"),
            "quality": round(float(quality.get("score") or 0.0), 3),
            "issues": list(quality.get("issues") or []),
            "url": storage.public_url(row["id"], "full"),
            "thumb_url": storage.public_url(row["id"], "thumb"),
            "created_at": row.get("created_at"),
        })
    db.audit("admin.view_originals", user_id, actor=admin["email"], n=len(out))
    return {"originals": out, "total": int(total["n"] or 0) if total else 0}


@router.get("/users/{user_id}/originals/{original_id}/download")
def user_original_download(user_id: str, original_id: str,
                           admin: dict = Depends(security.admin_user)):
    row = db.row_to_dict(db.q1(
        "SELECT * FROM originals WHERE id=? AND user_id=?", (original_id, user_id)))
    if not row or not Path(str(row.get("path") or "")).is_file():
        raise HTTPException(404, "Esa foto no existe.")
    db.audit("admin.download_original", user_id, actor=admin["email"],
             original_id=original_id)
    return FileResponse(row["path"], filename=row.get("filename") or f"{original_id}.jpg")


@router.get("/users/{user_id}/images/{image_id}/download")
def user_image_download(user_id: str, image_id: str,
                        admin: dict = Depends(security.admin_user)):
    row = db.row_to_dict(db.q1(
        "SELECT * FROM images WHERE id=? AND user_id=?", (image_id, user_id)))
    if not row or not Path(str(row.get("path") or "")).is_file():
        raise HTTPException(404, "Esa imagen no existe.")
    db.audit("admin.download_image", user_id, actor=admin["email"],
             image_id=image_id)
    return FileResponse(row["path"], media_type="image/jpeg",
                        filename=f"{image_id}.jpg")


@router.get("/stats")
def stats(admin: dict = Depends(security.admin_user)) -> dict:
    users = db.rows_to_dicts(db.q(
        "SELECT status, COUNT(*) AS n FROM users GROUP BY status"))
    images = db.rows_to_dicts(db.q(
        "SELECT kind, COUNT(*) AS n FROM images WHERE deleted_at IS NULL "
        "GROUP BY kind"))
    spend = db.rows_to_dicts(db.q(
        "SELECT provider, COALESCE(SUM(cost_usd),0) AS cost, COUNT(*) AS n "
        "FROM attempts GROUP BY provider"))
    reasons = db.rows_to_dicts(db.q(
        "SELECT reject_reason AS reason, COUNT(*) AS count FROM attempts "
        "WHERE status='rejected' AND reject_reason<>'' GROUP BY reason "
        "ORDER BY count DESC LIMIT 15"))
    totals = db.q1(
        "SELECT COUNT(*) AS attempts, "
        "SUM(CASE WHEN status='accepted' THEN 1 ELSE 0 END) AS accepted "
        "FROM attempts")
    attempts = int(totals["attempts"] or 0) if totals else 0
    accepted = int(totals["accepted"] or 0) if totals else 0
    originals = db.q1("SELECT COUNT(*) AS n FROM originals WHERE deleted_at IS NULL")
    return {
        "users_by_status": {r["status"]: r["n"] for r in users},
        "images_by_kind": {r["kind"]: r["n"] for r in images},
        "spend_by_provider": spend,
        "originals": int(originals["n"] or 0) if originals else 0,
        "attempts": attempts,
        "accepted": accepted,
        "attempts_per_photo": round(attempts / accepted, 2) if accepted else None,
        "reject_reasons": reasons,
    }


@router.get("/audit")
def audit(limit: int = 100, offset: int = 0,
          admin: dict = Depends(security.admin_user)) -> dict:
    rows = db.rows_to_dicts(db.q(
        "SELECT * FROM audit ORDER BY created_at DESC LIMIT ? OFFSET ?",
        (max(1, min(int(limit), 500)), max(0, int(offset)))))
    total = db.q1("SELECT COUNT(*) AS n FROM audit")
    return {"events": rows, "total": int(total["n"] or 0) if total else 0}


@router.get("/providers")
def providers(admin: dict = Depends(security.admin_user)) -> dict:
    from ..providers import registry
    keys = config.key_status()
    availability = registry.availability()
    for name, info in availability.items():
        key_name = info.get("key_name") or name
        info["key"] = keys.get(key_name, {"present": False, "hint": None,
                                          "from_env": False})
    return {"providers": availability, "keys": keys}


@router.post("/maintenance/reseed-catalog")
def reseed(admin: dict = Depends(security.admin_user)) -> dict:
    result = seed_mod.reseed(force=True)
    db.audit("admin.reseed", admin["id"], actor=admin["email"])
    return {"ok": True, **result}


@router.post("/maintenance/purge-deleted")
def purge(body: PurgeBody,
          admin: dict = Depends(security.admin_user)) -> dict:
    cutoff = db.now() - max(1, int(body.days)) * 86400
    removed = 0
    for table in ("images", "originals"):
        rows = db.q(f"SELECT id, path, thumb_path FROM {table} "
                    f"WHERE deleted_at IS NOT NULL AND deleted_at < ?", (cutoff,))
        for row in rows:
            storage.delete_file(row["path"])
            storage.delete_file(row["thumb_path"])
            db.execute(f"DELETE FROM {table} WHERE id=?", (row["id"],))
            removed += 1
    db.audit("admin.purge", admin["id"], actor=admin["email"], removed=removed)
    return {"ok": True, "removed": removed}
