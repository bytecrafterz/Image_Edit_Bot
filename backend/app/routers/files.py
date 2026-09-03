"""Serving image bytes.

Two doors, both checked.  A logged in client may ask for an image by id and the
row's owner is verified; an <img> tag, which cannot send a header, presents a
signed token that names exactly one image.  There is no third door that takes a
path.
"""
from __future__ import annotations

import mimetypes
from pathlib import Path

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import FileResponse

from .. import db, security
from ..services import storage

router = APIRouter(prefix="/files", tags=["files"])

CACHE = "private, max-age=86400"


def _serve(path_str: str | None, filename: str) -> FileResponse:
    if not path_str:
        raise HTTPException(404, "Ese archivo ya no esta disponible.")
    path = Path(path_str)
    if not path.is_file():
        raise HTTPException(404, "Ese archivo ya no esta disponible.")
    media, _ = mimetypes.guess_type(path.name)
    return FileResponse(path, media_type=media or "image/jpeg",
                        filename=filename,
                        headers={"Cache-Control": CACHE})


def _image_row(image_id: str) -> dict:
    row = db.row_to_dict(db.q1(
        "SELECT * FROM images WHERE id=? AND deleted_at IS NULL", (image_id,)))
    if not row:
        raise HTTPException(404, "Esa imagen no existe.")
    return row


@router.get("/original/{original_id}")
def original_file(original_id: str, variant: str = "full",
                  user: dict = Depends(security.current_user)):
    row = db.row_to_dict(db.q1(
        "SELECT * FROM originals WHERE id=? AND deleted_at IS NULL",
        (original_id,)))
    if not row:
        raise HTTPException(404, "Esa foto no existe.")
    if row["user_id"] != user["id"] and user.get("role") != "admin":
        raise HTTPException(403, "No puedes ver esa foto.")
    path = row.get("thumb_path") if variant == "thumb" else row["path"]
    return _serve(path or row["path"], row["filename"])


@router.get("/id/{image_id}")
def image_by_id(image_id: str, variant: str = "full",
                user: dict = Depends(security.current_user)):
    row = _image_row(image_id)
    if row["user_id"] != user["id"] and user.get("role") != "admin":
        raise HTTPException(403, "No puedes ver esa imagen.")
    path = row.get("thumb_path") if variant == "thumb" else row["path"]
    return _serve(path or row["path"], f"{image_id}.jpg")


@router.get("/{token}")
def image_by_token(token: str):
    """Signed access, so an <img> src works without an Authorization header."""
    resolved = storage.resolve_token(token)
    if not resolved:
        raise HTTPException(404, "Enlace no valido.")
    image_id, variant = resolved
    row = db.row_to_dict(db.q1(
        "SELECT * FROM images WHERE id=? AND deleted_at IS NULL", (image_id,)))
    if row:
        path = row.get("thumb_path") if variant == "thumb" else row["path"]
        return _serve(path or row["path"], f"{image_id}.jpg")
    orig = db.row_to_dict(db.q1(
        "SELECT * FROM originals WHERE id=? AND deleted_at IS NULL", (image_id,)))
    if orig:
        path = orig.get("thumb_path") if variant == "thumb" else orig["path"]
        return _serve(path or orig["path"], orig["filename"])
    raise HTTPException(404, "Esa imagen no existe.")
