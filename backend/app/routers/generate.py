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
    # One of the two: a photograph of hers, or (source_image_id) a result.
    original_id: str | None = None
    profile_id: str | None = None
    style: str | None = None
    options: dict = Field(default_factory=dict)
    n_previews: int = 6
    quality: str = "preview"
    # Which edit engine draws it (a provider role such as identity_banana);
    # empty means the provider's own choice.  See providers/fal MODELS.
    engine: str | None = None
    # Edit one of the robot's own results instead of a photograph: "no,
    # cambia el escote" starts from the image she is looking at.
    source_image_id: str | None = None


class InterpretBody(BaseModel):
    texto: str = ""
    original_id: str | None = None
    source_image_id: str | None = None
    referencia_value: str | None = None


class RunBody(BaseModel):
    run_id: str
    # SAYING YES TO A COMBINATION THE RECORD HAS NEVER SEEN WORK.  It defaults
    # to False so that an old client, or a script, cannot spend on one by
    # forgetting to send the field: the safe answer to a missing flag is the
    # one that does not charge.
    confirmar_riesgo: bool = False


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


@router.post("/interpretar")
def interpretar(body: InterpretBody, user: dict = Depends(security.active_user)) -> dict:
    """Her words - typed or dictated - into choices, before anything is priced.

    Claude reads the catalogue, her sentence and, when she attached one, the
    picture of the garment or of a previous result, and answers with existing
    values where they fit and new ones where the catalogue had nothing; the new
    ones become values of her own (options.add_user_value) so the plan, the
    guard and the record see nothing unusual.  Costs a fraction of a cent of
    Anthropic usage and no image generation.
    """
    from ..catalog import options as options_mod
    from ..providers import registry
    texto = (body.texto or "").strip()
    if not texto and not body.referencia_value:
        raise HTTPException(400, "Dime que quieres cambiar, o elige una prenda.")
    try:
        vision = registry.get_vision_provider("claude")
    except Exception:                                     # noqa: BLE001
        vision = None
    if not vision or not getattr(vision, "available", lambda: False)() \
            or not hasattr(vision, "interpret_request"):
        raise HTTPException(400, "Para escribir o dictar lo que quieres hace "
                            "falta la clave de Anthropic en Ajustes.")
    source = orchestrator._source_row(user["id"], body.source_image_id or body.original_id or "")
    shot = "unknown"
    if source and not source.get("es_resultado"):
        shot = str(orchestrator.analyse_original(source).get("shot_type") or "unknown")
    groups = options_mod.groups_for_shot(shot)
    catalogo = {g["group_key"]: {"label_es": g["label_es"],
                                 "values": [{"value_key": v["value_key"], "label_es": v["label_es"]}
                                            for v in g["values"]]
                                           + [{"value_key": k, "label_es": k} for k in
                                              sorted(options_mod.user_value_keys(user["id"], g["group_key"]))]}
                for g in groups}
    garment = None
    if body.referencia_value:
        row = db.row_to_dict(db.q1("SELECT * FROM options WHERE user_id=? AND value_key=?",
                                   (user["id"], body.referencia_value)))
        garment = ((row or {}).get("params") or {}).get("garment_image")
    result = vision.interpret_request(
        texto or "Quiero llevar la prenda de la foto de referencia.", catalogo,
        garment_image=garment,
        current_image=(source or {}).get("path") if (source or {}).get("es_resultado") else None,
        context={"plano": shot})
    if not result.get("ok"):
        raise HTTPException(502, str(result.get("error") or "No se pudo interpretar."))
    choices: dict[str, list[str]] = {}
    nuevas: list[dict] = []
    for group, items in (result.get("elecciones") or {}).items():
        if group not in catalogo or not isinstance(items, list):
            continue
        keys: list[str] = []
        for item in items:
            if not isinstance(item, dict):
                continue
            if item.get("valor"):
                key = str(item["valor"])
                if any(v["value_key"] == key for v in catalogo[group]["values"]):
                    keys.append(key)
            elif isinstance(item.get("nueva"), dict) and (item["nueva"].get("prompt") or "").strip():
                nv = item["nueva"]
                row = options_mod.add_user_value(
                    user["id"], group, str(nv.get("label_es") or "a tu manera"),
                    str(nv["prompt"]), str(nv.get("negative") or ""))
                keys.append(row["value_key"]); nuevas.append(row)
        if keys:
            choices[group] = keys
    if body.referencia_value and "clothing" not in choices:
        choices["clothing"] = [body.referencia_value]
    db.audit("generate.interpretar", user["id"], texto=texto[:200], grupos=sorted(choices),
             nuevas=len(nuevas), coste=result.get("cost_usd"))
    return {"choices": choices, "resumen": result.get("resumen") or "",
            "rechazado": result.get("rechazado") or "", "nuevas": nuevas,
            "vistas": result.get("vistas") or 0, "coste_usd": result.get("cost_usd") or 0.0}


@router.post("/analyze")
def analyze(body: AnalyzeBody,
            user: dict = Depends(security.active_user)) -> dict:
    if not (body.original_id or body.source_image_id):
        raise HTTPException(400, "Elige una foto tuya o una imagen del album.")
    if not 1 <= int(body.n_previews) <= SETTINGS.limits.max_previews_per_run:
        raise HTTPException(400, "Puedes pedir entre 1 y %d vistas previas."
                            % SETTINGS.limits.max_previews_per_run)
    try:
        return orchestrator.prepare_run(
            user, body.source_image_id or body.original_id, body.options, body.n_previews,
            body.quality, body.profile_id, body.style,
            engine=body.engine)
    except PermissionError as exc:
        raise HTTPException(400, str(exc))
    except ValueError as exc:
        raise HTTPException(404, str(exc))


@router.post("/run")
def run(body: RunBody, user: dict = Depends(security.active_user)) -> dict:
    row = _own_run(body.run_id, user)
    if row.get("status") != "queued":
        raise HTTPException(409, "Ese trabajo ya se ha ejecutado.")
    # THE ROBOT DOES NOT QUIETLY TAKE THE MONEY.  When the estimate found a
    # combination that has failed every single time it was tried, it wrote the
    # sentence with the numbers onto the plan; this is where that sentence
    # becomes a refusal instead of a paragraph nobody read.  Read off the
    # STORED plan and not recomputed here on purpose: the record can grow
    # between the estimate and the button, and the client must be answering the
    # question she was actually shown.
    warning = str((row.get("plan") or {}).get("riesgo_confirmar") or "")
    if warning and not body.confirmar_riesgo:
        raise HTTPException(409, warning)
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

    # THE CALLS THAT NEVER CAME BACK AS A PICTURE, counted apart from the
    # robot's own verdicts.  ``discard_reasons`` above only reads 'rejected'
    # rows - images the gate looked at and refused - and the screen described
    # both cases with the same sentence, so a provider that charged for a black
    # file was reported to the client as "no ha salido ninguna buena", which
    # blames her photographs for something no robot ever saw.  The prefix is
    # written by the orchestrator when fal answers content_filter.
    blocked = db.q1(
        "SELECT COUNT(*) AS n, COALESCE(SUM(cost_usd),0) AS usd FROM attempts "
        "WHERE run_id=? AND status='error' "
        "AND reject_reason LIKE 'bloqueada por el proveedor%'", (run_id,))

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
        "bloqueadas": {"n": int(blocked["n"] or 0) if blocked else 0,
                       "usd": round(float(blocked["usd"] or 0.0), 5)
                       if blocked else 0.0},
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
