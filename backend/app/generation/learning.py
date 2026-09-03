"""Learning from corrections.

The client's yardstick is "five manual attempts must become one or two".  Part
of that is measurement, part of it is memory: when she keeps discarding the same
background, or the same defect keeps coming back, the robot has to change what
it plans - by itself, and visibly.

The memory is deliberately small and readable.  One row per user and scope in
the ``learning`` table holding three dictionaries: a weight per option value, a
weight per generation parameter, and a decaying counter per defect type.  Every
weight moves with the same exponential rule (w = w*0.8 + 0.2*outcome) so a
single bad image never erases a history, and nothing is ever changed silently:
each adjustment appends a sentence in Spanish to ``plan["notes"]``.
"""
from __future__ import annotations

from typing import Any

from .. import db

ALPHA = 0.2                 # exponential update rate, per CONTRACTS section 3
DEFAULT_W = 0.5             # an option nobody has judged yet is neutral
LOW_W = 0.25                # below this the value is actively disliked
HIGH_W = 0.60               # above this it is worth promoting
MIN_N_FOR_SWAP = 3          # never rewrite a plan on one single opinion

STRENGTH_STEP = 0.03        # per recent occurrence of a defect
STRENGTH_FLOOR = 0.25       # below this img2img stops being a photo edit
DEFECT_DECAY = 0.9          # each new judgement ages the old defect counters
DEFECT_DROP = 0.05          # counters below this are forgotten
MAX_DEFECT_TOTAL = 8.0      # cap: -0.24 of strength at most from defects

PARAM_KEYS = ("strength", "guidance", "steps", "identity_weight")

POSITIVE = ("like", "liked", "selected", "select", "favorite", "favourite",
            "favorito", "keep", "kept", "good", "up", "accepted", "aceptada",
            "aprobada", "ok")
NEGATIVE = ("dislike", "disliked", "discard", "discarded", "descartada",
            "reject", "rejected", "rechazada", "bad", "down", "delete",
            "deleted", "borrada")

DEFECT_ES: dict[str, str] = {
    "hand_malformed": "manos deformadas",
    "extra_limb": "extremidades de mas",
    "extra_person": "personas de mas",
    "face_distorted": "rostro alterado",
    "eye_asymmetry": "ojos asimetricos",
    "missing_limb": "falta una extremidad",
    "texture_smear": "textura emborronada",
    "duplicated_feature": "elementos duplicados",
    "border_artifact": "artefactos en el borde",
    "oversmoothed_skin": "piel demasiado suavizada",
}


# ------------------------------------------------------------------ helpers

def _f(value: Any, default: float = 0.0) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return default
    if out != out or out in (float("inf"), float("-inf")):
        return default
    return out


def _text(value: Any) -> str:
    return str(value).strip() if value is not None else ""


def empty_weights() -> dict:
    return {"options": {}, "params": {}, "defects": {}, "n": 0}


def _normalise(weights: Any) -> dict:
    out = empty_weights()
    if not isinstance(weights, dict):
        return out
    for section in ("options", "params", "defects"):
        raw = weights.get(section)
        if not isinstance(raw, dict):
            continue
        for key, value in raw.items():
            name = _text(key)
            if name:
                out[section][name] = _f(value, 0.0)
    out["n"] = int(_f(weights.get("n"), 0.0))
    return out


def outcome_of(verdict: Any) -> float | None:
    """1.0 for like/selected, 0.0 for dislike/discarded, None for anything else."""
    word = _text(verdict).lower()
    if word in POSITIVE:
        return 1.0
    if word in NEGATIVE:
        return 0.0
    return None


def _defect_types(source: Any) -> list[str]:
    out: list[str] = []
    if isinstance(source, dict):
        source = source.get("defects")
    if isinstance(source, (list, tuple)):
        for item in source:
            kind = _text(item.get("type")) if isinstance(item, dict) else _text(item)
            if kind and kind not in out:
                out.append(kind)
    return out


def _choices_from_meta(meta: Any) -> dict[str, str]:
    """Which option values produced this image.

    The orchestrator writes the variant into ``images.meta_json``; several
    shapes are accepted so a later refactor there cannot silently stop the
    learning from attributing credit.
    """
    if not isinstance(meta, dict):
        return {}
    for key in ("choices", "options", "selection"):
        raw = meta.get(key)
        if isinstance(raw, dict) and raw:
            break
    else:
        variant = meta.get("variant")
        raw = variant.get("choices") if isinstance(variant, dict) else None
    out: dict[str, str] = {}
    if isinstance(raw, dict):
        for group, value in raw.items():
            if isinstance(value, dict):
                value = value.get("value") or value.get("value_key")
            if isinstance(value, (list, tuple)):
                value = value[0] if value else None
            group_key, value_key = _text(group), _text(value)
            if group_key and value_key:
                out[group_key] = value_key
    return out


def _params_from_meta(meta: Any) -> dict[str, float]:
    if not isinstance(meta, dict):
        return {}
    raw = meta.get("params")
    if not isinstance(raw, dict):
        variant = meta.get("variant")
        raw = variant.get("params") if isinstance(variant, dict) else None
    out: dict[str, float] = {}
    if isinstance(raw, dict):
        for key in PARAM_KEYS:
            if key in raw:
                out[key] = _f(raw[key], 0.0)
    return out


# --------------------------------------------------------------- persistence

def get_weights(user_id: str, scope: str = "global") -> dict:
    """Stored weights for one user and scope; neutral defaults when empty."""
    try:
        row = db.q1("SELECT weights_json FROM learning WHERE user_id=? AND scope=?",
                    (_text(user_id), _text(scope) or "global"))
    except Exception:
        return empty_weights()
    rec = db.row_to_dict(row)
    return _normalise((rec or {}).get("weights"))


def _get_stats(user_id: str, scope: str) -> dict:
    try:
        row = db.q1("SELECT stats_json FROM learning WHERE user_id=? AND scope=?",
                    (user_id, scope))
    except Exception:
        return {}
    rec = db.row_to_dict(row)
    stats = (rec or {}).get("stats")
    return stats if isinstance(stats, dict) else {}


def _save(user_id: str, scope: str, weights: dict, stats: dict) -> bool:
    try:
        db.execute(
            "INSERT INTO learning(id,user_id,scope,weights_json,stats_json,updated_at) "
            "VALUES(?,?,?,?,?,?) ON CONFLICT(user_id,scope) DO UPDATE SET "
            "weights_json=excluded.weights_json, stats_json=excluded.stats_json, "
            "updated_at=excluded.updated_at",
            (db.new_id("lrn"), user_id, scope, db.dumps(weights),
             db.dumps(stats), db.now()))
        return True
    except Exception:
        return False


def _apply_update(weights: dict, outcome: float | None,
                  choices: dict[str, str], params: dict[str, float],
                  defect_types: list[str]) -> dict:
    """The exponential rule, applied to every value that produced the image."""
    out = _normalise(weights)
    if outcome is not None:
        for group, value in choices.items():
            key = "%s:%s" % (group, value)
            current = out["options"].get(key, DEFAULT_W)
            out["options"][key] = round(current * (1.0 - ALPHA) + ALPHA * outcome, 6)
        for name in params:
            current = out["params"].get(name, DEFAULT_W)
            out["params"][name] = round(current * (1.0 - ALPHA) + ALPHA * outcome, 6)
        out["n"] = int(out["n"]) + 1

    if defect_types or outcome is not None:
        aged = {}
        for kind, count in out["defects"].items():
            value = round(count * DEFECT_DECAY, 6)
            if value >= DEFECT_DROP:
                aged[kind] = value
        out["defects"] = aged
    # Only a bad outcome teaches "this defect keeps coming back"; a defect the
    # user accepted anyway must not push every future render away from her photo.
    if defect_types and (outcome is None or outcome < 0.5):
        for kind in defect_types:
            out["defects"][kind] = round(
                min(MAX_DEFECT_TOTAL, out["defects"].get(kind, 0.0) + 1.0), 6)
    return out


def _scopes_for(image: dict | None) -> list[str]:
    scopes = ["global"]
    profile_id = _text((image or {}).get("profile_id"))
    if profile_id:
        scopes.append("profile:" + profile_id)
    return scopes


def record_feedback(user_id: str, image_id: str, verdict: str,
                    reason: str = "") -> None:
    """Write the feedback row and move every weight the image is responsible for."""
    uid = _text(user_id)
    iid = _text(image_id)
    verdict_word = _text(verdict).lower()
    outcome = outcome_of(verdict_word)

    image: dict | None = None
    try:
        image = db.row_to_dict(db.q1("SELECT * FROM images WHERE id=?", (iid,)))
    except Exception:
        image = None
    if image is not None and _text(image.get("user_id")) != uid:
        image = None                      # never learn from another user's image

    meta = (image or {}).get("meta")
    choices = _choices_from_meta(meta)
    params = _params_from_meta(meta)
    defects = _defect_types((image or {}).get("verdict")) or _defect_types(meta)

    context = {
        "choices": choices, "params": params, "defects": defects,
        "provider": _text((image or {}).get("provider")),
        "model": _text((image or {}).get("model")),
        "score": _f((image or {}).get("score"), 0.0),
        "outcome": outcome,
    }
    try:
        db.execute(
            "INSERT INTO feedback(id,user_id,image_id,run_id,verdict,reason,"
            "context_json,created_at) VALUES(?,?,?,?,?,?,?,?)",
            (db.new_id("fb"), uid, iid or None, _text((image or {}).get("run_id")) or None,
             verdict_word, _text(reason), db.dumps(context), db.now()))
    except Exception:
        pass

    if outcome is None and not defects:
        return
    for scope in _scopes_for(image):
        weights = _apply_update(get_weights(uid, scope), outcome, choices,
                                params, defects)
        stats = _get_stats(uid, scope)
        stats["likes"] = int(_f(stats.get("likes"), 0.0)) + (1 if outcome == 1.0 else 0)
        stats["dislikes"] = int(_f(stats.get("dislikes"), 0.0)) + (1 if outcome == 0.0 else 0)
        stats["last_verdict"] = verdict_word
        stats["last_image_id"] = iid
        stats["updated_at"] = db.now()
        _save(uid, scope, weights, stats)


def record_defects(user_id: str, defect_types: list[str],
                   scope: str = "global") -> None:
    """Optional hook for the orchestrator: a rejected attempt is also evidence."""
    kinds = [_text(k) for k in (defect_types or []) if _text(k)]
    if not kinds:
        return
    uid = _text(user_id)
    weights = _apply_update(get_weights(uid, scope), None, {}, {}, kinds)
    _save(uid, scope, weights, _get_stats(uid, scope))


# ------------------------------------------------------------- applying it

def defect_summary_es(weights: dict) -> tuple[float, str]:
    """Total recent defect pressure plus a readable Spanish enumeration."""
    counts = _normalise(weights)["defects"]
    total = 0.0
    parts: list[str] = []
    for kind, count in sorted(counts.items(), key=lambda kv: (-kv[1], kv[0])):
        if count < DEFECT_DROP:
            continue
        total += count
        parts.append("%s x%d" % (DEFECT_ES.get(kind, kind), max(1, int(round(count)))))
    return min(total, MAX_DEFECT_TOTAL), ", ".join(parts[:4])


def apply_weights_to_params(params: dict, weights: dict) -> tuple[dict, list[str]]:
    """Push img2img strength down while the same defects keep reappearing."""
    out = dict(params or {})
    notes: list[str] = []
    w = _normalise(weights)
    strength = _f(out.get("strength"), 0.0)
    if strength <= 0.0:
        return out, notes

    total, detail = defect_summary_es(w)
    new_strength = strength
    if total > 0.0:
        new_strength = max(STRENGTH_FLOOR, strength - STRENGTH_STEP * total)
        if new_strength < strength - 1e-9:
            notes.append(
                "Se baja la fuerza de %.2f a %.2f porque se repiten defectos "
                "(%s): una fuerza menor mantiene la imagen mas cerca de tu foto "
                "real." % (strength, new_strength, detail or "varios"))

    param_w = w["params"].get("strength")
    if param_w is not None and w["n"] >= 5 and param_w < 0.35:
        adjusted = max(STRENGTH_FLOOR, new_strength - 0.02)
        if adjusted < new_strength - 1e-9:
            notes.append(
                "Se baja la fuerza otro 0.02 porque las imagenes con esta "
                "fuerza no te gustaron (%d valoraciones)." % w["n"])
            new_strength = adjusted

    if abs(new_strength - strength) > 1e-9:
        out["strength"] = round(new_strength, 4)
    return out, notes


def _variant_score(variant: dict, options_w: dict) -> float:
    choices = variant.get("choices") if isinstance(variant, dict) else None
    if not isinstance(choices, dict) or not choices:
        return DEFAULT_W
    values = [options_w.get("%s:%s" % (g, v), DEFAULT_W) for g, v in choices.items()]
    return sum(values) / float(len(values))


def _best_alternative(group: str, options_w: dict, taken: set[str]) -> str:
    best, best_w = "", HIGH_W
    prefix = group + ":"
    for key, weight in sorted(options_w.items()):
        if not key.startswith(prefix):
            continue
        value = key[len(prefix):]
        if value in taken or weight < best_w:
            continue
        if weight > best_w or (weight == best_w and not best):
            best, best_w = value, weight
    return best


def apply_learning(plan: dict, weights: dict) -> dict:
    """Reweight an existing plan toward what this user actually keeps."""
    if not isinstance(plan, dict):
        return {"variants": [], "locked": {}, "varied": [], "notes": [],
                "learning_applied": True}
    out = dict(plan)
    notes = [str(n) for n in (plan.get("notes") or [])]
    out["notes"] = notes
    variants = [dict(v) for v in (plan.get("variants") or []) if isinstance(v, dict)]
    out["variants"] = variants

    if plan.get("learning_applied"):
        return out
    out["learning_applied"] = True

    w = _normalise(weights)
    if not w["options"] and not w["defects"] and not w["params"]:
        notes.append("Aun no hay valoraciones tuyas suficientes: se planifica "
                     "sin ajustes aprendidos.")
        return out
    if not variants:
        return out

    locked = plan.get("locked") if isinstance(plan.get("locked"), dict) else {}
    options_w = w["options"]

    # 1. Replace values this user reliably discards, never a locked choice.
    if w["n"] >= MIN_N_FOR_SWAP:
        taken: dict[str, set[str]] = {}
        for variant in variants:
            for group, value in (variant.get("choices") or {}).items():
                taken.setdefault(str(group), set()).add(str(value))
        for variant in variants:
            choices = dict(variant.get("choices") or {})
            for group, value in list(choices.items()):
                if group in locked:
                    continue
                key = "%s:%s" % (group, value)
                if options_w.get(key, DEFAULT_W) > LOW_W:
                    continue
                alt = _best_alternative(str(group), options_w,
                                        taken.get(str(group), set()))
                if not alt:
                    continue
                choices[group] = alt
                taken.setdefault(str(group), set()).add(alt)
                taken[str(group)].discard(str(value))
                variant["why"] = (str(variant.get("why") or "").strip()
                                  + " Ajustado por tus valoraciones.").strip()
                notes.append(
                    "En la vista %s se cambio '%s' por '%s' porque casi siempre "
                    "descartas ese valor." % (variant.get("index", "?"), value, alt))
            variant["choices"] = choices

    # 2. Parameters: the same defect coming back means less denoise.
    param_notes: list[str] = []
    for variant in variants:
        new_params, changed_notes = apply_weights_to_params(
            variant.get("params") or {}, w)
        variant["params"] = new_params
        for note in changed_notes:
            if note not in param_notes:
                param_notes.append(note)
    notes.extend(param_notes)

    # 3. Show first what she usually keeps.
    order = sorted(range(len(variants)),
                   key=lambda i: (-_variant_score(variants[i], options_w), i))
    if order != list(range(len(variants))):
        variants = [variants[i] for i in order]
        for position, variant in enumerate(variants):
            variant["index"] = position
        out["variants"] = variants
        notes.append("Se reordenaron las vistas previas: primero las "
                     "combinaciones que sueles elegir.")
    return out
