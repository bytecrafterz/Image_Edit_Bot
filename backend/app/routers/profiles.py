"""Identity profiles: create, measure, consent, forget the originals.

The measurement step is the slow one (a few seconds per photograph), so with a
real reference set it runs as a background job and the phone polls, exactly like
a generation run.
"""
from __future__ import annotations

import logging

from fastapi import APIRouter, Depends, HTTPException, Request
from pydantic import BaseModel

from .. import db, security
from ..identity import profile as profile_mod
from ..safety import consent as consent_mod
from ..services import jobs, storage

log = logging.getLogger("photorobot.profiles")
router = APIRouter(prefix="/profiles", tags=["profiles"])

INLINE_LIMIT = 8          # photographs analysed inside the request


class CreateBody(BaseModel):
    person_name: str
    is_default: bool = False


class BuildBody(BaseModel):
    original_ids: list[str] | None = None


class ConsentBody(BaseModel):
    relationship: str = "self"
    granted_by: str = ""
    statement: str = ""
    evidence_note: str = ""
    scope: list[str] | None = None


class ForgetBody(BaseModel):
    confirm: bool = False


def _own(profile_id: str, user: dict) -> dict:
    row = db.row_to_dict(db.q1(
        "SELECT * FROM profiles WHERE id=? AND user_id=? AND deleted_at IS NULL",
        (profile_id, user["id"])))
    if not row:
        raise HTTPException(404, "Ese perfil no existe.")
    return row


def _decorate(row: dict) -> dict:
    row = dict(row)
    counts = db.q1("SELECT COUNT(*) AS n FROM originals WHERE profile_id=? "
                   "AND deleted_at IS NULL", (row["id"],))
    row["n_originals"] = int(counts["n"] or 0) if counts else 0
    row["consent_ok"] = consent_mod.has_valid_consent(row["id"])
    row["consent_problem"] = consent_mod.consent_problem(row["id"])
    return row


@router.get("")
def list_profiles(user: dict = Depends(security.active_user)) -> dict:
    rows = db.rows_to_dicts(db.q(
        "SELECT * FROM profiles WHERE user_id=? AND deleted_at IS NULL "
        "ORDER BY is_default DESC, updated_at DESC", (user["id"],)))
    return {"profiles": [_decorate(r) for r in rows]}


@router.post("")
def create(body: CreateBody,
           user: dict = Depends(security.active_user)) -> dict:
    name = (body.person_name or "").strip()
    if not name:
        raise HTTPException(400, "Escribe un nombre para el perfil.")
    profile_id = db.new_id("prf")
    now = db.now()
    existing = db.q1("SELECT COUNT(*) AS n FROM profiles WHERE user_id=? "
                     "AND deleted_at IS NULL", (user["id"],))
    is_default = body.is_default or int(existing["n"] or 0) == 0
    if is_default:
        db.execute("UPDATE profiles SET is_default=0 WHERE user_id=?",
                   (user["id"],))
    db.execute(
        "INSERT INTO profiles(id,user_id,person_name,status,is_default,"
        "created_at,updated_at) VALUES(?,?,?,'draft',?,?,?)",
        (profile_id, user["id"], name, 1 if is_default else 0, now, now))
    db.audit("profile.create", user["id"], profile_id=profile_id)
    return _decorate(db.row_to_dict(db.q1("SELECT * FROM profiles WHERE id=?",
                                          (profile_id,))))


def _store_profile(profile_id: str, built: dict) -> None:
    coverage = built.get("coverage") or {}
    status = "ready" if coverage.get("ready_for_body_check") else "draft"
    db.execute(
        "UPDATE profiles SET n_sources=?, coverage_json=?, face_json=?, "
        "body_json=?, skin_json=?, hair_json=?, marks_json=?, thresholds_json=?, "
        "status=?, updated_at=? WHERE id=?",
        (int(built.get("n_sources") or 0), db.dumps(coverage),
         db.dumps(built.get("face") or {}), db.dumps(built.get("body") or {}),
         db.dumps(built.get("skin") or {}), db.dumps(built.get("hair") or {}),
         db.dumps(built.get("marks") or []),
         db.dumps(built.get("thresholds") or {}), status, db.now(), profile_id))


@router.post("/{profile_id}/build")
def build(profile_id: str, body: BuildBody,
          user: dict = Depends(security.active_user)) -> dict:
    _own(profile_id, user)
    if body.original_ids:
        placeholders = ",".join("?" * len(body.original_ids))
        rows = db.q(
            f"SELECT path FROM originals WHERE user_id=? AND deleted_at IS NULL "
            f"AND id IN ({placeholders})",
            (user["id"], *body.original_ids))
    else:
        rows = db.q(
            "SELECT path FROM originals WHERE user_id=? AND deleted_at IS NULL "
            "AND (profile_id=? OR profile_id IS NULL) ORDER BY sort_order",
            (user["id"], profile_id))
    paths = [r["path"] for r in rows]
    if not paths:
        raise HTTPException(400, "No hay fotos para construir el perfil. "
                                 "Sube algunas en Mis fotos.")

    name = _own(profile_id, user)["person_name"]

    if len(paths) <= INLINE_LIMIT:
        built = profile_mod.build_profile(paths, name)
        _store_profile(profile_id, built)
        db.execute("UPDATE originals SET profile_id=? WHERE user_id=? "
                   "AND profile_id IS NULL", (profile_id, user["id"]))
        return {"ok": True, "async": False,
                "profile": _decorate(db.row_to_dict(db.q1(
                    "SELECT * FROM profiles WHERE id=?", (profile_id,))))}

    run_id = db.new_id("run")
    db.execute(
        "INSERT INTO runs(id,user_id,profile_id,mode,status,n_requested,"
        "created_at,stage) VALUES(?,?,?,'profile','queued',?,?,?)",
        (run_id, user["id"], profile_id, len(paths), db.now(),
         "Midiendo %d fotos" % len(paths)))

    def work() -> dict:
        built = profile_mod.build_profile(paths, name)
        _store_profile(profile_id, built)
        db.execute("UPDATE originals SET profile_id=? WHERE user_id=? "
                   "AND profile_id IS NULL", (profile_id, user["id"]))
        db.execute("UPDATE runs SET status='done', progress=1.0, finished_at=?, "
                   "stage='Perfil actualizado' WHERE id=?", (db.now(), run_id))
        return {"ok": True}

    jobs.submit(run_id, work)
    return {"ok": True, "async": True, "run_id": run_id,
            "message": "Estamos midiendo tus fotos, tardara un momento."}


@router.get("/{profile_id}")
def get_profile(profile_id: str,
                user: dict = Depends(security.active_user)) -> dict:
    return _decorate(_own(profile_id, user))


@router.post("/{profile_id}/consent")
def consent(profile_id: str, body: ConsentBody, request: Request,
            user: dict = Depends(security.active_user)) -> dict:
    _own(profile_id, user)
    try:
        record = consent_mod.record_consent(user["id"], profile_id, {
            "relationship": body.relationship,
            "granted_by": body.granted_by,
            "statement": body.statement,
            "evidence_note": body.evidence_note,
            "scope": body.scope,
            "ip": security.client_ip(request),
        })
    except ValueError as exc:
        raise HTTPException(400, str(exc))
    return {"ok": True, "consent": record,
            "profile": _decorate(_own(profile_id, user))}


@router.post("/{profile_id}/forget-originals")
def forget_originals(profile_id: str, body: ForgetBody,
                     user: dict = Depends(security.active_user)) -> dict:
    """Delete the photographs, keep the measurements.

    This is what makes the multi-person plan defensible: once a person has been
    measured, the system does not need to keep her pictures to keep working.
    """
    profile = _own(profile_id, user)
    if not body.confirm:
        raise HTTPException(400, "Confirma la accion para continuar.")
    if profile.get("status") != "ready" and not (profile.get("body") or {}):
        raise HTTPException(400, "Construye el perfil antes de borrar las fotos: "
                                 "si no, se perderian las medidas.")
    rows = db.rows_to_dicts(db.q(
        "SELECT * FROM originals WHERE profile_id=? AND user_id=? "
        "AND deleted_at IS NULL", (profile_id, user["id"])))
    for row in rows:
        storage.delete_file(row.get("path"))
        storage.delete_file(row.get("thumb_path"))
        db.execute("UPDATE originals SET deleted_at=?, path='', thumb_path=NULL "
                   "WHERE id=?", (db.now(), row["id"]))
    db.audit("profile.forget_originals", user["id"], profile_id=profile_id,
             deleted=len(rows))
    return {"ok": True, "deleted": len(rows),
            "message": ("Se han borrado %d fotos. Las medidas del perfil se "
                        "mantienen y el sistema sigue funcionando." % len(rows))}


@router.delete("/{profile_id}")
def delete(profile_id: str,
           user: dict = Depends(security.active_user)) -> dict:
    _own(profile_id, user)
    db.execute("UPDATE profiles SET deleted_at=? WHERE id=?",
               (db.now(), profile_id))
    db.execute("UPDATE images SET deleted_at=? WHERE profile_id=? "
               "AND deleted_at IS NULL", (db.now(), profile_id))
    db.audit("profile.delete", user["id"], profile_id=profile_id)
    return {"ok": True}
