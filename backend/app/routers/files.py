"""Serving image bytes.

Two doors, both checked.  A logged in client may ask for an image by id and the
row's owner is verified; an <img> tag, which cannot send a header, presents a
signed token that names exactly one image.  There is no third door that takes a
path.

OWNERSHIP IS THE WHOLE CHECK, AND AN ADMINISTRATOR IS NOT AN EXCEPTION.  Both
id routes used to end ``and user.get("role") != "admin"``, which meant the
administrator account could fetch the raw bytes of any other person's reference
photographs and generated images.  Measured on 2026-09-04 with two real
accounts in a throwaway installation: the admin asked for the other woman's
uploaded PNG by id and got 200 with the file, and the same for her generated
JPEG - while routers/admin.py's own docstring promises that what an
administrator "deliberately cannot do here is look at anyone else's
photographs".  These are photographs of a person's body; no support task needs
them, and the two counts the admin screen shows come from COUNT(*), not from
the pixels.  The route now answers 404 - the same answer as a row that does not
exist, so it is not an oracle for which ids are real either.
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


def _image_row(image_id: str, user_id: str) -> dict:
    row = db.row_to_dict(db.q1(
        "SELECT * FROM images WHERE id=? AND user_id=? AND deleted_at IS NULL",
        (image_id, user_id)))
    if not row:
        raise HTTPException(404, "Esa imagen no existe.")
    return row


@router.get("/original/{original_id}")
def original_file(original_id: str, variant: str = "full",
                  user: dict = Depends(security.current_user)):
    row = db.row_to_dict(db.q1(
        "SELECT * FROM originals WHERE id=? AND user_id=? AND deleted_at IS NULL",
        (original_id, user["id"])))
    if not row:
        raise HTTPException(404, "Esa foto no existe.")
    path = row.get("thumb_path") if variant == "thumb" else row["path"]
    return _serve(path or row["path"], row["filename"])


@router.get("/id/{image_id}")
def image_by_id(image_id: str, variant: str = "full",
                user: dict = Depends(security.current_user)):
    row = _image_row(image_id, user["id"])
    path = row.get("thumb_path") if variant == "thumb" else row["path"]
    return _serve(path or row["path"], f"{image_id}.jpg")


@router.get("/{token}")
def image_by_token(token: str):
    """Signed access, so an <img> src works without an Authorization header.

    The token is an HMAC over one id with the installation secret, so it cannot
    be guessed or altered and it names exactly one file.  What it is NOT is a
    per-user check: whoever holds the string can read that one file, and it
    never expires.  That is the deliberate trade for a gallery of eighty
    thumbnails on a phone, and the only way anyone gets a token for someone
    else's photograph is by already holding the secret - at which point they
    hold the whole installation.  The audit reaches this route only by minting
    the token with that secret in-process, which is not a door a browser has.
    """
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
