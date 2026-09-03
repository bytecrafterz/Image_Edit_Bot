"""Favourites.  A flag on the image row, never a second copy of the file."""
from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel

from .. import db, security
from ..generation import learning
from ..services import storage

router = APIRouter(prefix="/favorites", tags=["favorites"])


class BulkBody(BaseModel):
    image_ids: list[str]
    favorite: bool = True


def _payload(row: dict) -> dict:
    return {
        "id": row["id"], "kind": row.get("kind"),
        "url": storage.public_url(row["id"], "full"),
        "thumb_url": storage.public_url(row["id"], "thumb"),
        "score": round(float(row.get("score") or 0.0), 3),
        "cost_usd": round(float(row.get("cost_usd") or 0.0), 5),
        "created_at": row.get("created_at"),
        "is_favorite": True,
    }


def _own(image_id: str, user: dict) -> dict:
    row = db.row_to_dict(db.q1(
        "SELECT * FROM images WHERE id=? AND user_id=? AND deleted_at IS NULL",
        (image_id, user["id"])))
    if not row:
        raise HTTPException(404, "Esa imagen no existe.")
    return row


@router.get("")
def list_favorites(user: dict = Depends(security.active_user)) -> dict:
    rows = db.rows_to_dicts(db.q(
        "SELECT * FROM images WHERE user_id=? AND is_favorite=1 "
        "AND deleted_at IS NULL ORDER BY created_at DESC", (user["id"],)))
    return {"images": [_payload(r) for r in rows], "total": len(rows)}


@router.post("/bulk")
def bulk(body: BulkBody, user: dict = Depends(security.active_user)) -> dict:
    """Mark or unmark a whole selection.

    Declared BEFORE /{image_id}: FastAPI matches in definition order, so with
    the literal route last, "/favorites/bulk" was captured by the parameterised
    route as image_id="bulk" and answered 404 "Esa imagen no existe" - the bulk
    action never worked, and said so in a message about a missing image.
    """
    if not body.image_ids:
        raise HTTPException(400, "No has elegido ninguna imagen.")
    ids = [str(i) for i in body.image_ids if str(i).strip()][:500]
    if not ids:
        raise HTTPException(400, "No has elegido ninguna imagen.")
    placeholders = ",".join("?" * len(ids))
    db.execute(
        f"UPDATE images SET is_favorite=? WHERE user_id=? AND deleted_at IS NULL "
        f"AND id IN ({placeholders})",
        (1 if body.favorite else 0, user["id"], *ids))
    if body.favorite:
        for image_id in ids:
            learning.record_feedback(user["id"], image_id, "like", "favorito")
    return {"ok": True, "n": len(ids), "favorite": body.favorite}


@router.post("/{image_id}")
def add(image_id: str, user: dict = Depends(security.active_user)) -> dict:
    _own(image_id, user)
    db.execute("UPDATE images SET is_favorite=1 WHERE id=?", (image_id,))
    learning.record_feedback(user["id"], image_id, "like", "favorito")
    return {"ok": True, "is_favorite": True}


@router.delete("/{image_id}")
def remove(image_id: str, user: dict = Depends(security.active_user)) -> dict:
    _own(image_id, user)
    db.execute("UPDATE images SET is_favorite=0 WHERE id=?", (image_id,))
    return {"ok": True, "is_favorite": False}
