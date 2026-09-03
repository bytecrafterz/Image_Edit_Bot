"""The generation endpoints.

Deliberately split into two calls.  ``/analyze`` plans and prices a run and
spends nothing; ``/run`` is the one that costs money, and it only ever acts on a
run the user has already seen an estimate for.  That separation is the whole
reason she can trust the cost figure on screen.
"""
from __future__ import annotations

import logging

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field

from .. import db, security
from ..config import SETTINGS
from ..generation import orchestrator
from ..services import billing, jobs, storage

log = logging.getLogger("photorobot.generate")
router = APIRouter(prefix="/generate", tags=["generate"])


class AnalyzeBody(BaseModel):
    original_id: str
    profile_id: str | None = None
    style: str | None = None
    options: dict = Field(default_factory=dict)
    n_previews: int = 6
    quality: str = "preview"


class RunBody(BaseModel):
    run_id: str


class FinalBody(BaseModel):
    run_id: str
    image_ids: list[str]
    quality: str = "high"


def _own_run(run_id: str, user: dict) -> dict:
    row = db.row_to_dict(db.q1("SELECT * FROM runs WHERE id=? AND user_id=?",
                               (run_id, user["id"])))
    if not row:
        raise HTTPException(404, "Ese trabajo no existe.")
    return row


def _image_payload(row: dict) -> dict:
    verdict = row.get("verdict") or {}
    return {
        "id": row["id"], "kind": row.get("kind"),
        "url": storage.public_url(row["id"], "full"),
        "thumb_url": storage.public_url(row["id"], "thumb"),
        "score": round(float(row.get("score") or 0.0), 3),
        "cost_usd": round(float(row.get("cost_usd") or 0.0), 5),
        "provider": row.get("provider"), "model": row.get("model"),
        "summary": verdict.get("summary", ""),
        "is_favorite": bool(row.get("is_favorite")),
        "created_at": row.get("created_at"),
    }


@router.post("/analyze")
def analyze(body: AnalyzeBody,
            user: dict = Depends(security.active_user)) -> dict:
    if not 1 <= int(body.n_previews) <= SETTINGS.limits.max_previews_per_run:
        raise HTTPException(400, "Puedes pedir entre 1 y %d vistas previas."
                            % SETTINGS.limits.max_previews_per_run)
    try:
        return orchestrator.prepare_run(
            user, body.original_id, body.options, body.n_previews,
            body.quality, body.profile_id, body.style)
    except PermissionError as exc:
        raise HTTPException(400, str(exc))
    except ValueError as exc:
        raise HTTPException(404, str(exc))


@router.post("/run")
def run(body: RunBody, user: dict = Depends(security.active_user)) -> dict:
    row = _own_run(body.run_id, user)
    if row.get("status") != "queued":
        raise HTTPException(409, "Ese trabajo ya se ha ejecutado.")
    result = jobs.submit(body.run_id,
                         lambda: orchestrator.run_previews(user, body.run_id))
    if not result.get("ok"):
        raise HTTPException(409, result.get("reason", "No se pudo iniciar."))
    return {"run_id": body.run_id, "status": "running"}


@router.get("/status/{run_id}")
def status(run_id: str, user: dict = Depends(security.active_user)) -> dict:
    row = _own_run(run_id, user)
    live = jobs.status(run_id)
    images = db.rows_to_dicts(db.q(
        "SELECT * FROM images WHERE run_id=? AND deleted_at IS NULL "
        "ORDER BY created_at", (run_id,)))
    counters = db.q1(
        "SELECT COUNT(*) AS attempts, "
        "SUM(CASE WHEN status='rejected' THEN 1 ELSE 0 END) AS rejected, "
        "COALESCE(SUM(cost_usd),0) AS cost FROM attempts WHERE run_id=?",
        (run_id,))
    reasons = db.rows_to_dicts(db.q(
        "SELECT reject_reason AS reason, COUNT(*) AS count FROM attempts "
        "WHERE run_id=? AND status='rejected' AND reject_reason<>'' "
        "GROUP BY reason ORDER BY count DESC", (run_id,)))

    payload = {
        "run_id": run_id,
        "status": live.get("status") or row.get("status"),
        "progress": float(row.get("progress") or 0.0),
        "stage": row.get("stage") or "",
        "error": row.get("error"),
        "images": [_image_payload(i) for i in images],
        "accepted": int(row.get("n_accepted") or 0),
        "rejected": int(counters["rejected"] or 0) if counters else 0,
        "attempts": int(counters["attempts"] or 0) if counters else 0,
        "spent_usd": round(float(counters["cost"] or 0.0), 5) if counters else 0.0,
        "est_cost_usd": round(float(row.get("est_cost_usd") or 0.0), 5),
        "discard_reasons": reasons,
    }
    if payload["status"] in ("done", "failed", "cancelled", "stopped_no_balance"):
        payload["report"] = orchestrator.build_report(run_id)
    if payload["status"] == "stopped_no_balance":
        provider = "fal"
        payload["balance_help"] = {
            "provider": provider,
            "balance": billing.balance(user["id"], provider),
            "recommended_topup": billing.recommended_topup(user["id"], provider),
        }
    return payload


@router.post("/final")
def final(body: FinalBody, user: dict = Depends(security.active_user)) -> dict:
    parent = _own_run(body.run_id, user)
    if not body.image_ids:
        raise HTTPException(400, "Elige al menos una imagen.")

    placeholders = ",".join("?" * len(body.image_ids))
    rows = db.q(f"SELECT id FROM images WHERE user_id=? AND deleted_at IS NULL "
                f"AND id IN ({placeholders})", (user["id"], *body.image_ids))
    valid = [r["id"] for r in rows]
    if not valid:
        raise HTTPException(404, "Esas imagenes ya no estan disponibles.")

    options = dict(parent.get("options") or {})
    options["selected_image_ids"] = valid
    options["quality"] = body.quality

    child_id = db.new_id("run")
    db.execute(
        "INSERT INTO runs(id,user_id,original_id,profile_id,parent_run_id,mode,"
        "status,options_json,plan_json,n_requested,created_at) "
        "VALUES(?,?,?,?,?,'final','queued',?,?,?,?)",
        (child_id, user["id"], parent.get("original_id"),
         parent.get("profile_id"), parent["id"], db.dumps(options),
         db.dumps(parent.get("plan") or {}), len(valid), db.now()))

    jobs.submit(child_id, lambda: orchestrator.run_final(user, child_id))
    return {"run_id": child_id, "status": "running", "n": len(valid)}


@router.post("/cancel/{run_id}")
def cancel(run_id: str, user: dict = Depends(security.active_user)) -> dict:
    _own_run(run_id, user)
    orchestrator.cancel(run_id)
    jobs.cancel(run_id)
    return {"ok": True, "message": "Deteniendo el trabajo."}


@router.get("/report/{run_id}")
def report(run_id: str, user: dict = Depends(security.active_user)) -> dict:
    _own_run(run_id, user)
    return orchestrator.build_report(run_id)


@router.get("/runs")
def recent_runs(limit: int = 20,
                user: dict = Depends(security.active_user)) -> dict:
    rows = db.rows_to_dicts(db.q(
        "SELECT * FROM runs WHERE user_id=? AND mode<>'profile' "
        "ORDER BY created_at DESC LIMIT ?",
        (user["id"], max(1, min(int(limit), 100)))))
    out = []
    for row in rows:
        out.append({
            "run_id": row["id"], "mode": row.get("mode"),
            "status": row.get("status"),
            "accepted": int(row.get("n_accepted") or 0),
            "rejected": int(row.get("n_rejected") or 0),
            "cost_usd": round(float(row.get("cost_usd") or 0.0), 5),
            "created_at": row.get("created_at"),
            "stage": row.get("stage") or "",
        })
    return {"runs": out}
