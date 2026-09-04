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
    # Which of those ids are actually hers, resolved BEFORE anything is written.
    # The UPDATE was already scoped by user_id, so nobody else's flag ever
    # moved - but the two things that followed were not scoped: the answer said
    # "n": len(ids) whether or not a single row matched, and record_feedback ran
    # once per id, writing a feedback row under HER user_id carrying SOMEONE
    # ELSE's image_id.  The isolation audit caught exactly that: account A
    # posted account B's image id here, B's favourites stayed at 0 and the reply
    # still said n=1, and a feedback row referring to B's image appeared under
    # A.  learning.record_feedback then refuses to learn from it ("never learn
    # from another user's image"), so the row was pure litter in her own
    # history.  Now the ids are matched first and only what matched is touched
    # or counted.
    placeholders = ",".join("?" * len(ids))
    mine = [r["id"] for r in db.q(
        f"SELECT id FROM images WHERE user_id=? AND deleted_at IS NULL "
        f"AND id IN ({placeholders})", (user["id"], *ids))]
    if not mine:
        return {"ok": True, "n": 0, "favorite": body.favorite,
                "mensaje": "Ninguna de esas imagenes es tuya."}
    marks = ",".join("?" * len(mine))
    db.execute(
        f"UPDATE images SET is_favorite=? WHERE user_id=? AND id IN ({marks})",
        (1 if body.favorite else 0, user["id"], *mine))
    if body.favorite:
        for image_id in mine:
            learning.record_feedback(user["id"], image_id, "like", "favorito")
    return {"ok": True, "n": len(mine), "favorite": body.favorite}


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
