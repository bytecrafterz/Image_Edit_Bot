"""Reference photographs: upload, list, order, delete, analyse.

The client photographs herself with a phone, so uploads arrive as several large
JPEGs at once, sometimes named .HEIC.  Everything here is built for that: many
files per request, generous limits, duplicate detection by hash, and a quick
reading of each photo stored immediately so the interface can say "medio cuerpo,
buena calidad" without waiting for the full analysis.
"""
from __future__ import annotations

import logging
from pathlib import Path

from fastapi import (APIRouter, Depends, File, HTTPException, Request,
                     UploadFile, status)
from pydantic import BaseModel

from .. import db, security
from ..analysis import loader
from ..config import SETTINGS
from ..safety import guard as guard_mod
from ..services import storage

log = logging.getLogger("photorobot.originals")
router = APIRouter(prefix="/originals", tags=["originals"])

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp", ".heic", ".heif", ".bmp"}


class PatchBody(BaseModel):
    sort_order: int | None = None
    profile_id: str | None = None
    tags: str | None = None


class ImportBody(BaseModel):
    path: str
    profile_id: str | None = None
    person_name: str | None = None


def _quick_read(path: str) -> dict:
    """Shot type and technical quality only: fast enough to run on upload."""
    from ..analysis import (face as face_mod, pose as pose_mod,
                            quality as quality_mod, shot as shot_mod)
    try:
        img = loader.load_image(path, max_side=1024)
        pose_d = pose_mod.detect_pose(img)
        face_d = face_mod.detect_face(img)
        shot = shot_mod.classify_shot(img, pose_d, face_d)
        qual = quality_mod.assess_quality(img, path)
        return {"shot_type": shot.get("shot_type", "unknown"), "quality": qual}
    except Exception as exc:                              # noqa: BLE001
        log.warning("Lectura rapida fallida en %s: %s", path, exc)
        return {"shot_type": "unknown", "quality": {}}


def _ingest(user: dict, filename: str, data: bytes,
            profile_id: str | None) -> dict:
    limit = SETTINGS.limits.max_upload_mb * 1024 * 1024
    if len(data) > limit:
        raise HTTPException(413, "%s pesa mas de %d MB."
                            % (filename, SETTINGS.limits.max_upload_mb))

    sha = storage.sha256_bytes(data)
    existing = storage.find_duplicate(user["id"], sha)
    if existing:
        return {**existing, "duplicate": True}

    stored = storage.store_upload(user["id"], filename, data)
    info = loader.image_info(stored["path"])
    thumb = Path(stored["path"]).with_name(Path(stored["path"]).stem + "_thumb.jpg")
    try:
        loader.make_thumb(stored["path"], thumb, 512)
    except Exception:                                     # noqa: BLE001
        thumb = None

    read = _quick_read(stored["path"])
    flags = guard_mod.check_upload(stored["path"])

    original_id = db.new_id("org")
    row = db.q1("SELECT COALESCE(MAX(sort_order), 0) AS m FROM originals "
                "WHERE user_id=?", (user["id"],))
    order = int(row["m"] or 0) + 10 if row else 0

    db.execute(
        "INSERT INTO originals(id,user_id,profile_id,filename,path,thumb_path,"
        "width,height,bytes,sha256,shot_type,quality_json,sort_order,created_at) "
        "VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
        (original_id, user["id"], profile_id, stored["filename"], stored["path"],
         str(thumb) if thumb else None, int(info.get("width") or 0),
         int(info.get("height") or 0), stored["bytes"], sha,
         read["shot_type"], db.dumps(read["quality"]), order, db.now()),
    )
    out = db.row_to_dict(db.q1("SELECT * FROM originals WHERE id=?",
                               (original_id,)))
    out["duplicate"] = False
    out["upload_flags"] = flags.get("flags") or []
    out["upload_note"] = flags.get("reason") or ""
    return out


def _decorate(row: dict) -> dict:
    row = dict(row)
    row["url"] = storage.public_url(row["id"], "full")
    row["thumb_url"] = storage.public_url(row["id"], "thumb")
    return row


@router.get("")
def list_originals(profile_id: str | None = None,
                   user: dict = Depends(security.active_user)) -> dict:
    sql = ("SELECT * FROM originals WHERE user_id=? AND deleted_at IS NULL")
    params: list = [user["id"]]
    if profile_id:
        sql += " AND profile_id=?"
        params.append(profile_id)
    sql += " ORDER BY sort_order, created_at"
    rows = db.rows_to_dicts(db.q(sql, params))
    counts: dict[str, int] = {"closeup": 0, "half": 0, "full": 0, "unknown": 0}
    for row in rows:
        key = row.get("shot_type") or "unknown"
        counts[key if key in counts else "unknown"] += 1
    return {"originals": [_decorate(r) for r in rows], "total": len(rows),
            "by_shot_type": counts}


@router.post("")
async def upload(request: Request, files: list[UploadFile] = File(...),
                 user: dict = Depends(security.active_user)) -> dict:
    profile_id = request.query_params.get("profile_id")
    row = db.q1("SELECT COUNT(*) AS n FROM originals WHERE user_id=? "
                "AND deleted_at IS NULL", (user["id"],))
    have = int(row["n"] or 0) if row else 0
    if have + len(files) > SETTINGS.limits.max_originals_per_user:
        raise HTTPException(400, "Has alcanzado el maximo de fotos guardadas.")

    stored, skipped = [], []
    for upload_file in files:
        suffix = Path(upload_file.filename or "").suffix.lower()
        if suffix and suffix not in IMAGE_SUFFIXES:
            skipped.append({"filename": upload_file.filename,
                            "reason": "no es una imagen"})
            continue
        data = await upload_file.read()
        if not data:
            skipped.append({"filename": upload_file.filename,
                            "reason": "archivo vacio"})
            continue
        try:
            stored.append(_decorate(_ingest(user, upload_file.filename or "foto.jpg",
                                            data, profile_id)))
        except HTTPException as exc:
            skipped.append({"filename": upload_file.filename,
                            "reason": exc.detail})
        except Exception as exc:                          # noqa: BLE001
            log.error("Fallo al guardar %s: %s", upload_file.filename, exc)
            skipped.append({"filename": upload_file.filename,
                            "reason": "no se pudo leer la imagen"})

    db.audit("originals.upload", user["id"], n=len(stored), skipped=len(skipped))
    return {"originals": stored, "skipped": skipped,
            "added": sum(1 for s in stored if not s.get("duplicate"))}


@router.get("/{original_id}/analysis")
def analysis(original_id: str,
             user: dict = Depends(security.active_user)) -> dict:
    from ..generation import orchestrator

    row = db.row_to_dict(db.q1(
        "SELECT * FROM originals WHERE id=? AND user_id=? AND deleted_at IS NULL",
        (original_id, user["id"])))
    if not row:
        raise HTTPException(404, "Esa foto no existe.")
    report = orchestrator.analyse_original(row)
    body = report.get("body") or {}
    return {
        "original_id": original_id,
        "shot_type": report.get("shot_type"),
        "quality": report.get("quality"),
        "has_face": report.get("has_face"),
        "measurable_body": bool(body.get("ok")),
        "body_reason": body.get("reason", ""),
        "metrics": body.get("metrics") or {},
        "unreliable": body.get("unreliable") or [],
        "defects": report.get("defects") or [],
    }


@router.patch("/{original_id}")
def patch(original_id: str, body: PatchBody,
          user: dict = Depends(security.active_user)) -> dict:
    row = db.q1("SELECT id FROM originals WHERE id=? AND user_id=?",
                (original_id, user["id"]))
    if not row:
        raise HTTPException(404, "Esa foto no existe.")
    fields, params = [], []
    for key in ("sort_order", "profile_id", "tags"):
        value = getattr(body, key)
        if value is not None:
            fields.append(f"{key}=?")
            params.append(value)
    if not fields:
        raise HTTPException(400, "Nada que cambiar.")
    params.append(original_id)
    db.execute(f"UPDATE originals SET {', '.join(fields)} WHERE id=?", params)
    return _decorate(db.row_to_dict(db.q1(
        "SELECT * FROM originals WHERE id=?", (original_id,))))


@router.delete("/{original_id}")
def delete(original_id: str,
           user: dict = Depends(security.active_user)) -> dict:
    row = db.row_to_dict(db.q1(
        "SELECT * FROM originals WHERE id=? AND user_id=?",
        (original_id, user["id"])))
    if not row:
        raise HTTPException(404, "Esa foto no existe.")
    db.execute("UPDATE originals SET deleted_at=? WHERE id=?",
               (db.now(), original_id))
    storage.delete_file(row.get("path"))
    storage.delete_file(row.get("thumb_path"))
    db.audit("originals.delete", user["id"], original_id=original_id)
    return {"ok": True}


@router.post("/import-folder")
def import_folder(body: ImportBody,
                  user: dict = Depends(security.admin_user)) -> dict:
    """Ingest a server side folder.  How the developer loads a batch at once."""
    root = Path(body.path)
    if not root.exists() or not root.is_dir():
        raise HTTPException(400, "Esa carpeta no existe en el servidor.")

    profile_id = body.profile_id
    if not profile_id and body.person_name:
        profile_id = db.new_id("prf")
        now = db.now()
        db.execute(
            "INSERT INTO profiles(id,user_id,person_name,status,is_default,"
            "created_at,updated_at) VALUES(?,?,?,'draft',1,?,?)",
            (profile_id, user["id"], body.person_name.strip(), now, now))

    added, skipped = [], []
    for path in sorted(root.rglob("*")):
        if not path.is_file() or path.suffix.lower() not in IMAGE_SUFFIXES:
            continue
        try:
            result = _ingest(user, path.name, path.read_bytes(), profile_id)
            (skipped if result.get("duplicate") else added).append(path.name)
        except Exception as exc:                          # noqa: BLE001
            log.warning("No se pudo importar %s: %s", path, exc)
            skipped.append(path.name)

    db.audit("originals.import_folder", user["id"], path=str(root),
             added=len(added))
    return {"added": len(added), "skipped": len(skipped),
            "profile_id": profile_id, "files": added[:100]}
