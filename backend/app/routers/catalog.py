"""The option and style menus, tailored to one photograph.

When an ``original_id`` is given the groups come back ordered for that specific
picture with a Spanish reason attached, which is the behaviour the developer
promised: the system looks at the photo and proposes what fits, instead of
showing one fixed list forever.
"""
from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, File, Form, UploadFile

from .. import db, security
from ..catalog import options as options_mod
from ..catalog import styles as styles_mod

router = APIRouter(prefix="/catalog", tags=["catalog"])


def _user_values(user_id: str, group_key: str) -> list[dict]:
    rows = db.rows_to_dicts(db.q(
        "SELECT * FROM options WHERE user_id=? AND group_key=? AND enabled=1 "
        "ORDER BY sort_order", (user_id, group_key)))
    return [{"value_key": r["value_key"], "label_es": r["label_es"],
             "label_en": r.get("label_en") or "",
             "prompt_fragment": r.get("prompt_fragment") or "",
             "negative_fragment": r.get("negative_fragment") or "",
             "shot_types": r.get("shot_types") or "closeup,half,full",
             "custom": True}
            for r in rows]


@router.get("/options")
def get_options(shot_type: str = "unknown", original_id: str | None = None,
                user: dict = Depends(security.active_user)) -> dict:
    analysis: dict = {}
    if original_id:
        row = db.row_to_dict(db.q1(
            "SELECT * FROM originals WHERE id=? AND user_id=? "
            "AND deleted_at IS NULL", (original_id, user["id"])))
        if not row:
            raise HTTPException(404, "Esa foto no existe.")
        from ..generation import orchestrator
        analysis = orchestrator.analyse_original(row)
        shot_type = analysis.get("shot_type") or shot_type

    suggestion = options_mod.suggest_for_analysis(
        analysis or {"shot_type": shot_type})

    groups = []
    for group in suggestion["groups"]:
        values = list(group["values"]) + _user_values(user["id"],
                                                      group["group_key"])
        groups.append({
            "group_key": group["group_key"],
            "label_es": group["label_es"],
            "label_en": group.get("label_en", ""),
            "multi": bool(group.get("multi", True)),
            "priority": group.get("priority", 99),
            "suggested": suggestion["suggested"].get(group["group_key"], []),
            "reason": suggestion["reason"].get(group["group_key"], ""),
            "values": [{k: v for k, v in value.items() if k != "local"}
                       for value in values],
        })
    return {"shot_type": shot_type, "groups": groups,
            "suggested": suggestion["suggested"],
            "reason": suggestion["reason"]}


@router.post("/referencia")
async def referencia(file: UploadFile = File(...), texto: str = Form(""),
                     user: dict = Depends(security.active_user)) -> dict:
    """A picture of a garment (or a place) she likes becomes a value of hers.

    Stored apart from her own photographs - it is of somebody else and must
    never enter her identity profile - described by Claude cut by cut, and
    saved as a clothing (or scene) value that carries the picture, so the
    engine receives the garment image itself alongside the words.
    """
    from ..config import DATA_DIR
    from ..providers import registry
    try:
        vision = registry.get_vision_provider("claude")
    except Exception:                                     # noqa: BLE001
        vision = None
    if not vision or not getattr(vision, "available", lambda: False)() \
            or not hasattr(vision, "describe_garment"):
        raise HTTPException(400, "Para usar una foto de referencia hace falta la "
                            "clave de Anthropic en Ajustes.")
    data = await file.read()
    if not data:
        raise HTTPException(400, "El archivo esta vacio.")
    folder = DATA_DIR / "referencias" / str(user["id"])
    folder.mkdir(parents=True, exist_ok=True)
    ref_id = db.new_id("ref")
    path = folder / f"{ref_id}.jpg"
    path.write_bytes(data)
    desc = vision.describe_garment(str(path), texto)
    if not desc.get("ok"):
        raise HTTPException(502, str(desc.get("error") or "No se pudo leer la prenda."))
    row = options_mod.add_user_value(
        user["id"], desc["grupo"], desc["label_es"], desc["prompt"], desc.get("negative", ""),
        params={"garment_image": str(path)} if desc["grupo"] == "clothing" else {})
    db.audit("catalog.referencia", user["id"], value=row["value_key"], grupo=desc["grupo"],
             coste=desc.get("cost_usd"))
    return {"value": row, "grupo": desc["grupo"], "descripcion": desc["prompt"],
            "coste_usd": desc.get("cost_usd") or 0.0}


@router.get("/styles")
def get_styles(shot_type: str = "unknown",
               user: dict = Depends(security.active_user)) -> dict:
    builtin = styles_mod.styles_for_shot(shot_type)
    custom = db.rows_to_dicts(db.q(
        "SELECT * FROM styles WHERE user_id=? AND enabled=1 ORDER BY sort_order",
        (user["id"],)))
    out = [{"key": s["key"], "name_es": s["name_es"],
            "name_en": s.get("name_en", ""),
            "description": s.get("description", ""),
            "shot_types": s.get("shot_types", ""),
            "defaults": s.get("defaults") or {}, "custom": False}
           for s in builtin]
    out += [{"key": s["key"], "name_es": s["name_es"],
             "name_en": s.get("name_en") or "",
             "description": s.get("description") or "",
             "shot_types": s.get("shot_types") or "",
             "defaults": s.get("defaults") or {}, "custom": True}
            for s in custom]
    return {"styles": out, "shot_type": shot_type}
