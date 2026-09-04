"""The album: everything the robot has produced and kept."""
from __future__ import annotations

from pathlib import Path

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel

from .. import db, security
from ..generation import learning
from ..services import storage

router = APIRouter(prefix="/album", tags=["album"])


class FeedbackBody(BaseModel):
    verdict: str
    reason: str = ""


def _payload(row: dict) -> dict:
    verdict = row.get("verdict") or {}
    meta = row.get("meta") or {}
    return {
        "id": row["id"], "kind": row.get("kind"), "run_id": row.get("run_id"),
        "url": storage.public_url(row["id"], "full"),
        "thumb_url": storage.public_url(row["id"], "thumb"),
        "width": row.get("width"), "height": row.get("height"),
        "score": round(float(row.get("score") or 0.0), 3),
        "cost_usd": round(float(row.get("cost_usd") or 0.0), 5),
        "provider": row.get("provider"), "model": row.get("model"),
        "is_favorite": bool(row.get("is_favorite")),
        "summary": verdict.get("summary", ""),
        "choices": meta.get("choices") or {},
        "created_at": row.get("created_at"),
    }


@router.get("")
def list_images(kind: str | None = None, run_id: str | None = None,
                favorites: bool = False, limit: int = 40, offset: int = 0,
                order: str = "desc",
                user: dict = Depends(security.active_user)) -> dict:
    sql = "SELECT * FROM images WHERE user_id=? AND deleted_at IS NULL"
    params: list = [user["id"]]
    if kind in ("preview", "final", "repair"):
        sql += " AND kind=?"
        params.append(kind)
    if run_id:
        sql += " AND run_id=?"
        params.append(run_id)
    if favorites:
        sql += " AND is_favorite=1"

    total_row = db.q1(sql.replace("SELECT *", "SELECT COUNT(*) AS n", 1), params)
    total = int(total_row["n"] or 0) if total_row else 0

    sql += " ORDER BY created_at " + ("ASC" if order == "asc" else "DESC")
    sql += " LIMIT ? OFFSET ?"
    params += [max(1, min(int(limit), 100)), max(0, int(offset))]
    rows = db.rows_to_dicts(db.q(sql, params))
    return {"images": [_payload(r) for r in rows], "total": total,
            "limit": limit, "offset": offset}


@router.get("/stats")
def stats(user: dict = Depends(security.active_user)) -> dict:
    row = db.q1(
        "SELECT COUNT(*) AS n, "
        "SUM(CASE WHEN kind='final' THEN 1 ELSE 0 END) AS finals, "
        "SUM(CASE WHEN is_favorite=1 THEN 1 ELSE 0 END) AS favorites, "
        "COALESCE(SUM(cost_usd),0) AS cost, COALESCE(AVG(score),0) AS score "
        "FROM images WHERE user_id=? AND deleted_at IS NULL", (user["id"],))
    return {
        "images": int(row["n"] or 0) if row else 0,
        "finals": int(row["finals"] or 0) if row else 0,
        "favorites": int(row["favorites"] or 0) if row else 0,
        "total_usd": round(float(row["cost"] or 0.0), 4) if row else 0.0,
        "avg_score": round(float(row["score"] or 0.0), 3) if row else 0.0,
    }


def _own(image_id: str, user: dict) -> dict:
    row = db.row_to_dict(db.q1(
        "SELECT * FROM images WHERE id=? AND user_id=?", (image_id, user["id"])))
    if not row:
        raise HTTPException(404, "Esa imagen no existe.")
    return row


@router.get("/{image_id}/download")
def download(image_id: str, user: dict = Depends(security.active_user)):
    row = _own(image_id, user)
    path = Path(row["path"])
    if not path.is_file():
        raise HTTPException(404, "El archivo ya no esta disponible.")
    import time
    stamp = time.strftime("%Y%m%d", time.localtime(float(row.get("created_at") or 0)))
    name = f"photorobot_{stamp}_{image_id[-6:]}.jpg"
    return FileResponse(path, media_type="image/jpeg", filename=name)


@router.post("/{image_id}/feedback")
def feedback(image_id: str, body: FeedbackBody,
             user: dict = Depends(security.active_user)) -> dict:
    _own(image_id, user)
    if body.verdict not in ("like", "dislike", "selected", "discarded"):
        raise HTTPException(400, "Valoracion no valida.")
    learning.record_feedback(user["id"], image_id, body.verdict, body.reason)
    return {"ok": True}


@router.post("/{image_id}/final")
def mark_final(image_id: str, user: dict = Depends(security.active_user)) -> dict:
    """Move an image she has already paid for into "Finales", for 0.00 USD.

    Until now the only way into that tab was ``run_final``, which re-renders the
    chosen previews and bills again: on 2026-09-04 the two images bought as the
    project's final delivery (0.08 USD, identity 0.7375 and 0.5958, both
    approved) sat in "Previas" while "Finales" read zero, and promoting them
    honestly would have cost another 0.08 USD out of a 2.58 USD balance.  A
    higher tier buys a more faithful model, not a bigger file - the pixels are
    already the pixels - so when the image in hand has passed every check there
    is nothing left to buy, and charging for the label would be charging twice.

    Two conditions, both about not lying to her.  The file has to be on disk,
    because "Finales" is where she goes to download.  And the verdict has to
    have PASSED: an image the robot could not confirm is her must never be
    relabelled as the finished work, whoever asks.  Anything else still goes
    through ``run_final`` and is priced there.
    """
    row = _own(image_id, user)
    if row.get("deleted_at"):
        raise HTTPException(410, "Esa imagen esta en la papelera.")
    if not Path(row["path"]).is_file():
        raise HTTPException(410, "El archivo ya no esta en el disco.")
    if str(row.get("kind")) == "final":
        return {"ok": True, "kind": "final", "cost_usd": 0.0,
                "mensaje": "Esa imagen ya estaba en Finales."}
    verdict = row.get("verdict") or {}
    if not verdict.get("passed"):
        raise HTTPException(
            409, "Esa imagen no paso todas las comprobaciones, asi que no se "
                 "puede marcar como final: %s" % (verdict.get("summary")
                                                  or "no fue aprobada."))
    db.execute("UPDATE images SET kind='final' WHERE id=? AND user_id=?",
               (image_id, user["id"]))
    db.audit("album.mark_final", user["id"], image_id=image_id)
    return {"ok": True, "kind": "final", "cost_usd": 0.0,
            "mensaje": "Ya esta en Finales. No se ha cobrado nada: es el mismo "
                       "archivo que ya pagaste."}


class BulkDeleteBody(BaseModel):
    image_ids: list[str]


@router.post("/bulk-delete")
def bulk_delete(body: BulkDeleteBody,
                user: dict = Depends(security.active_user)) -> dict:
    """Delete a selection in one request.

    Deleting from the browser one call at a time is slow on a phone and leaves
    the gallery half deleted if the connection drops midway; one request either
    removes the selection or does not.  Ownership is filtered in SQL, so an id
    belonging to someone else is simply not matched.
    """
    ids = [str(i) for i in (body.image_ids or []) if str(i).strip()][:500]
    if not ids:
        raise HTTPException(400, "No has elegido ninguna imagen.")

    placeholders = ",".join("?" * len(ids))
    rows = db.rows_to_dicts(db.q(
        f"SELECT id, path, thumb_path FROM images WHERE user_id=? "
        f"AND deleted_at IS NULL AND id IN ({placeholders})",
        (user["id"], *ids)))
    if not rows:
        return {"ok": True, "deleted": 0}

    db.execute(
        f"UPDATE images SET deleted_at=? WHERE user_id=? AND id IN ({placeholders})",
        (db.now(), user["id"], *ids))
    for row in rows:
        storage.delete_file(row.get("path"))
        storage.delete_file(row.get("thumb_path"))
    db.audit("album.bulk_delete", user["id"], n=len(rows))
    return {"ok": True, "deleted": len(rows)}


@router.delete("/{image_id}")
def delete(image_id: str, user: dict = Depends(security.active_user)) -> dict:
    row = _own(image_id, user)
    db.execute("UPDATE images SET deleted_at=? WHERE id=?", (db.now(), image_id))
    storage.delete_file(row.get("path"))
    storage.delete_file(row.get("thumb_path"))
    db.audit("album.delete", user["id"], image_id=image_id)
    return {"ok": True}


@router.post("/{image_id}/restore")
def restore(image_id: str, user: dict = Depends(security.active_user)) -> dict:
    row = _own(image_id, user)
    if not Path(row["path"]).is_file():
        raise HTTPException(410, "El archivo ya se borro del disco.")
    db.execute("UPDATE images SET deleted_at=NULL WHERE id=?", (image_id,))
    return {"ok": True}
