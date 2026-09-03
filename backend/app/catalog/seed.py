"""Write the built in catalogue into the database on every start.

Upserts, never deletes: rows the user created (non-null user_id) are untouched,
so a restart refreshes the shipped catalogue without destroying anyone's own
clothing entries.
"""
from __future__ import annotations

import logging

from .. import db
from .options import BUILTIN_OPTIONS
from .styles import BUILTIN_STYLES

log = logging.getLogger("photorobot.catalog")


def seed_options(force: bool = False) -> int:
    n = 0
    now = db.now()
    # The upsert below targets (user_id, group_key, value_key), and for builtin
    # rows user_id is NULL.  SQLite treats every NULL as distinct in a UNIQUE
    # constraint, so the conflict clause never fires and each restart inserted a
    # fresh copy of the whole catalogue.  Builtin rows are a mirror of the Python
    # constants and own no user data, so the honest fix is to rewrite them wholesale.
    db.execute("DELETE FROM options WHERE builtin=1 AND user_id IS NULL")
    for group in BUILTIN_OPTIONS:
        group_key = group["group_key"]
        shot_types = group.get("shot_types", "closeup,half,full")
        for order, value in enumerate(group["values"]):
            params = dict(value.get("params") or {})
            if value.get("local"):
                params["local"] = value["local"]
            params["group_label_es"] = group["label_es"]
            params["group_label_en"] = group.get("label_en", "")
            params["multi"] = bool(group.get("multi", True))
            params["group_sort"] = int(group.get("sort_order", 0))
            db.execute(
                "INSERT INTO options(id,user_id,group_key,value_key,label_es,"
                "label_en,prompt_fragment,negative_fragment,params_json,"
                "shot_types,enabled,sort_order,builtin,created_at) "
                "VALUES(?,NULL,?,?,?,?,?,?,?,?,1,?,1,?) "
                "ON CONFLICT(user_id,group_key,value_key) DO UPDATE SET "
                "label_es=excluded.label_es, label_en=excluded.label_en, "
                "prompt_fragment=excluded.prompt_fragment, "
                "negative_fragment=excluded.negative_fragment, "
                "params_json=excluded.params_json, "
                "shot_types=excluded.shot_types, sort_order=excluded.sort_order",
                (db.new_id("opt"), group_key, value["value_key"],
                 value["label_es"], value.get("label_en", ""),
                 value.get("prompt_fragment", ""),
                 value.get("negative_fragment", ""), db.dumps(params),
                 value.get("shot_types", shot_types), order * 10, now),
            )
            n += 1
    return n


def seed_styles(force: bool = False) -> int:
    n = 0
    now = db.now()
    db.execute("DELETE FROM styles WHERE builtin=1 AND user_id IS NULL")
    for style in BUILTIN_STYLES:
        db.execute(
            "INSERT INTO styles(id,user_id,key,name_es,name_en,description,"
            "shot_types,prompt_template,negative_template,defaults_json,"
            "params_json,enabled,sort_order,builtin,created_at) "
            "VALUES(?,NULL,?,?,?,?,?,?,?,?,?,1,?,1,?) "
            "ON CONFLICT(user_id,key) DO UPDATE SET "
            "name_es=excluded.name_es, name_en=excluded.name_en, "
            "description=excluded.description, shot_types=excluded.shot_types, "
            "prompt_template=excluded.prompt_template, "
            "negative_template=excluded.negative_template, "
            "defaults_json=excluded.defaults_json, "
            "params_json=excluded.params_json, sort_order=excluded.sort_order",
            (db.new_id("sty"), style["key"], style["name_es"],
             style.get("name_en", ""), style.get("description", ""),
             style.get("shot_types", "closeup,half,full"),
             style.get("prompt_template", ""),
             style.get("negative_template", ""),
             db.dumps(style.get("defaults") or {}),
             db.dumps(style.get("params") or {}),
             int(style.get("sort_order", 0)), now),
        )
        n += 1
    return n


def seed_all() -> dict:
    try:
        options = seed_options()
        styles = seed_styles()
        log.info("Catalogo: %d opciones, %d estilos", options, styles)
        return {"options": options, "styles": styles}
    except Exception as exc:                              # noqa: BLE001
        # A catalogue problem must never stop the server from starting.
        log.error("No se pudo sembrar el catalogo: %s", exc)
        return {"options": 0, "styles": 0, "error": str(exc)}


def reseed(force: bool = False) -> dict:
    if force:
        db.execute("DELETE FROM options WHERE builtin=1 AND user_id IS NULL")
        db.execute("DELETE FROM styles WHERE builtin=1 AND user_id IS NULL")
    return seed_all()
