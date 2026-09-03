"""Settings: keys, limits, balances, usage and the user's own catalogue.

Two rules that are not negotiable here.  An API key is never returned to the
browser - only whether one exists and a masked hint.  And ``/recharge`` writes a
number down; it charges nothing, and its response says so in words the client
will read.
"""
from __future__ import annotations

import logging

import httpx
from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel

from .. import config, db, security
from ..config import SETTINGS
from ..services import billing

log = logging.getLogger("photorobot.settings")
router = APIRouter(prefix="/settings", tags=["settings"])

DEFAULTS: dict = {
    "default_n_previews": 6,
    "default_quality": "preview",
    "default_provider": "auto",
    "autorepair": True,
    "max_repair_rounds": SETTINGS.limits.max_repair_rounds,
    "max_retries": SETTINGS.limits.max_retries_per_variant,
    "strictness": "normal",
    "locale": "es",
    "notify_low_balance": True,
    "low_balance_threshold_usd": SETTINGS.limits.low_balance_usd,
    "daily_budget_usd": SETTINGS.limits.default_daily_usd,
    "monthly_budget_usd": SETTINGS.limits.default_monthly_usd,
}

STRICTNESS_ES = {
    "suave": "Acepta mas imagenes. Menos descartes, algun fallo puede pasar.",
    "normal": "El equilibrio recomendado entre exigencia y coste.",
    "estricto": "Descarta a la minima. Mas fieles, pero gastaras mas intentos.",
}


class KeyBody(BaseModel):
    provider: str
    key: str


class RechargeBody(BaseModel):
    provider: str
    amount_usd: float


class OptionBody(BaseModel):
    group_key: str
    value_key: str
    label_es: str
    prompt_fragment: str = ""
    negative_fragment: str = ""
    shot_types: str = "closeup,half,full"
    enabled: bool = True


def _get_settings(user_id: str) -> dict:
    rows = db.q("SELECT key, value_json FROM user_settings WHERE user_id=?",
                (user_id,))
    stored = {r["key"]: db.loads(r["value_json"], None) for r in rows}
    return {**DEFAULTS, **{k: v for k, v in stored.items() if v is not None}}


def _validate(key: str, value):
    if key in ("default_n_previews",):
        value = int(value)
        if not 1 <= value <= SETTINGS.limits.max_previews_per_run:
            raise ValueError("Las vistas previas van de 1 a %d."
                             % SETTINGS.limits.max_previews_per_run)
    elif key in ("max_repair_rounds", "max_retries"):
        value = int(value)
        if not 0 <= value <= 3:
            raise ValueError("Ese valor va de 0 a 3.")
    elif key == "default_quality":
        if value not in ("draft", "preview", "standard", "high", "max"):
            raise ValueError("Calidad no valida.")
    elif key == "strictness":
        if value not in STRICTNESS_ES:
            raise ValueError("La estrictez debe ser suave, normal o estricto.")
    elif key in ("autorepair", "notify_low_balance"):
        value = bool(value)
    elif key in ("low_balance_threshold_usd", "daily_budget_usd",
                 "monthly_budget_usd"):
        value = float(value)
        if value < 0 or value > 10000:
            raise ValueError("Importe fuera de rango.")
    elif key == "locale":
        if value not in ("es", "en", "pt"):
            raise ValueError("Idioma no valido.")
    elif key == "default_provider":
        if value not in ("auto", "local", "fal"):
            raise ValueError("Proveedor no valido.")
    return value


@router.get("")
def get_settings(user: dict = Depends(security.active_user)) -> dict:
    settings = _get_settings(user["id"])
    prices = {tier: billing.price_of_next_image(user["id"], tier)
              for tier in ("draft", "preview", "standard", "high", "max")}
    used_today = db.q1(
        "SELECT COUNT(*) AS n FROM attempts WHERE user_id=? AND created_at >= ?",
        (user["id"], billing._day_start()))
    return {
        "settings": settings,
        "keys": config.key_status(),
        "limits": {
            "daily_usd": float(user.get("daily_limit_usd") or 0.0),
            "monthly_usd": float(user.get("monthly_limit_usd") or 0.0),
            "free_quota_daily": int(user.get("free_quota_daily") or 0),
            "free_used_today": int(used_today["n"] or 0) if used_today else 0,
            "max_previews": SETTINGS.limits.max_previews_per_run,
        },
        "balances": billing.all_balances(user["id"]),
        "prices": prices,
        "plan": user.get("plan"),
        "strictness_help": STRICTNESS_ES,
    }


@router.put("")
def put_settings(body: dict, user: dict = Depends(security.active_user)) -> dict:
    unknown = [k for k in body if k not in DEFAULTS]
    if unknown:
        raise HTTPException(400, "Ajustes desconocidos: %s. Permitidos: %s"
                            % (", ".join(unknown), ", ".join(sorted(DEFAULTS))))
    for key, raw in body.items():
        try:
            value = _validate(key, raw)
        except (ValueError, TypeError) as exc:
            raise HTTPException(400, "%s: %s" % (key, exc))
        db.execute(
            "INSERT INTO user_settings(user_id,key,value_json,updated_at) "
            "VALUES(?,?,?,?) ON CONFLICT(user_id,key) DO UPDATE SET "
            "value_json=excluded.value_json, updated_at=excluded.updated_at",
            (user["id"], key, db.dumps(value), db.now()))
        if key == "daily_budget_usd":
            db.execute("UPDATE users SET daily_limit_usd=? WHERE id=?",
                       (float(value), user["id"]))
        elif key == "monthly_budget_usd":
            db.execute("UPDATE users SET monthly_limit_usd=? WHERE id=?",
                       (float(value), user["id"]))
    return {"ok": True, "settings": _get_settings(user["id"])}


def _probe(provider: str, key: str) -> dict:
    """Cheap authenticated call, so a typo is caught here and not mid-run."""
    try:
        if provider == "anthropic":
            resp = httpx.post(
                "https://api.anthropic.com/v1/messages",
                headers={"x-api-key": key, "anthropic-version": "2023-06-01",
                         "content-type": "application/json"},
                json={"model": "claude-sonnet-5", "max_tokens": 1,
                      "messages": [{"role": "user", "content": "hi"}]},
                timeout=20.0)
            if resp.status_code in (200, 201):
                return {"ok": True, "message": "Clave valida."}
            if resp.status_code in (401, 403):
                return {"ok": False, "message": "La clave no es valida."}
            if resp.status_code == 400:
                return {"ok": True, "message": "Clave aceptada."}
            return {"ok": False, "message": "Respuesta inesperada (%d)."
                    % resp.status_code}
        if provider == "fal":
            resp = httpx.get("https://rest.alpha.fal.ai/tokens/",
                             headers={"Authorization": f"Key {key}"},
                             timeout=20.0)
            if resp.status_code in (200, 201, 405):
                return {"ok": True, "message": "Clave valida."}
            if resp.status_code in (401, 403):
                return {"ok": False, "message": "La clave no es valida."}
            return {"ok": True, "message": "Clave guardada (no se pudo verificar)."}
    except httpx.HTTPError as exc:
        log.warning("No se pudo verificar la clave de %s: %s", provider, exc)
        return {"ok": True, "message": "Guardada, pero no se pudo verificar ahora."}
    return {"ok": True, "message": "Guardada."}


@router.post("/keys")
def set_key(body: KeyBody, user: dict = Depends(security.admin_user)) -> dict:
    provider = body.provider.strip().lower()
    if provider not in ("anthropic", "fal", "openai", "replicate", "stability"):
        raise HTTPException(400, "Proveedor no reconocido.")
    key = (body.key or "").strip()
    if len(key) < 12:
        raise HTTPException(400, "Esa clave parece incompleta.")
    result = _probe(provider, key)
    if not result["ok"]:
        raise HTTPException(400, result["message"])
    config.set_api_key(provider, key)
    db.audit("settings.key_set", user["id"], provider=provider)
    return {"ok": True, "provider": provider, "message": result["message"],
            "keys": config.key_status()}


@router.delete("/keys/{provider}")
def delete_key(provider: str,
               user: dict = Depends(security.admin_user)) -> dict:
    config.set_api_key(provider, None)
    db.audit("settings.key_removed", user["id"], provider=provider)
    return {"ok": True, "keys": config.key_status()}


@router.get("/usage")
def usage(days: int = 30, user: dict = Depends(security.active_user)) -> dict:
    return billing.usage(user["id"], max(1, min(int(days), 365)))


@router.post("/recharge")
def recharge(body: RechargeBody,
             user: dict = Depends(security.active_user)) -> dict:
    # Records money the user already added at the provider's own website.
    # Nothing in this call moves money, and the response says so.
    try:
        return billing.recharge(user["id"], body.provider.strip().lower(),
                                float(body.amount_usd), note="registro manual")
    except ValueError as exc:
        raise HTTPException(400, str(exc))


@router.get("/alerts")
def alerts(limit: int = 50,
           user: dict = Depends(security.active_user)) -> dict:
    rows = db.rows_to_dicts(db.q(
        "SELECT * FROM alerts WHERE user_id=? ORDER BY created_at DESC LIMIT ?",
        (user["id"], max(1, min(int(limit), 200)))))
    return {"alerts": rows,
            "unread": sum(1 for r in rows if not r.get("read_at"))}


@router.post("/alerts/{alert_id}/read")
def read_alert(alert_id: str,
               user: dict = Depends(security.active_user)) -> dict:
    db.execute("UPDATE alerts SET read_at=? WHERE id=? AND user_id=?",
               (db.now(), alert_id, user["id"]))
    return {"ok": True}


@router.post("/alerts/read-all")
def read_all(user: dict = Depends(security.active_user)) -> dict:
    db.execute("UPDATE alerts SET read_at=? WHERE user_id=? AND read_at IS NULL",
               (db.now(), user["id"]))
    return {"ok": True}


@router.get("/options")
def my_options(user: dict = Depends(security.active_user)) -> dict:
    rows = db.rows_to_dicts(db.q(
        "SELECT * FROM options WHERE user_id=? ORDER BY group_key, sort_order",
        (user["id"],)))
    return {"options": rows}


@router.post("/options")
def add_option(body: OptionBody,
               user: dict = Depends(security.active_user)) -> dict:
    from ..catalog.options import GROUPS_BY_KEY
    if body.group_key not in GROUPS_BY_KEY:
        raise HTTPException(400, "Ese grupo no existe.")
    value_key = body.value_key.strip().lower().replace(" ", "_")
    if not value_key or not body.label_es.strip():
        raise HTTPException(400, "Falta el nombre de la opcion.")
    db.execute(
        "INSERT INTO options(id,user_id,group_key,value_key,label_es,label_en,"
        "prompt_fragment,negative_fragment,shot_types,enabled,sort_order,"
        "builtin,created_at) VALUES(?,?,?,?,?,'',?,?,?,?,0,0,?) "
        "ON CONFLICT(user_id,group_key,value_key) DO UPDATE SET "
        "label_es=excluded.label_es, prompt_fragment=excluded.prompt_fragment, "
        "negative_fragment=excluded.negative_fragment, enabled=excluded.enabled",
        (db.new_id("opt"), user["id"], body.group_key, value_key,
         body.label_es.strip(), body.prompt_fragment.strip(),
         body.negative_fragment.strip(), body.shot_types,
         1 if body.enabled else 0, db.now()))
    return {"ok": True}


@router.delete("/options/{option_id}")
def delete_option(option_id: str,
                  user: dict = Depends(security.active_user)) -> dict:
    row = db.q1("SELECT id FROM options WHERE id=? AND user_id=?",
                (option_id, user["id"]))
    if not row:
        raise HTTPException(404, "Esa opcion no existe.")
    db.execute("DELETE FROM options WHERE id=?", (option_id,))
    return {"ok": True}
