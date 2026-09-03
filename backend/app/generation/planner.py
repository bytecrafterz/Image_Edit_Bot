"""Variant planning - deciding what the N previews should actually differ in.

The rule comes straight from the client.  A group where she picked exactly one
value is a decision: it is locked and identical in every preview.  A group where
she picked several is a question: the previews cross those values so she can
compare them.  A group she left alone is an opportunity: the robot varies it on
its own, otherwise the previews come back as five copies of the same photograph
and she has spent money to learn nothing.

Everything here is reproducible.  Seeds are derived from the run id and the
variant index, never from the clock, so the same run replans to the same images
and the report can be trusted.
"""
from __future__ import annotations

import hashlib
import itertools
import json
from typing import Any

from ..config import SETTINGS
from . import learning as learning_mod
from . import prompt as prompt_mod

PRODUCT_CAP = 4096          # bound on the cartesian product we will enumerate
MAX_FREE_GROUPS = 2         # more than two free axes and nothing is comparable
NEVER_FREE = ("body", "style", "framing")

# How much a group changes what the photograph looks like, per shot type.  Used
# only to pick which untouched groups are worth varying by ourselves.
PRIORITY: dict[str, dict[str, float]] = {
    "closeup": {"expression": 5.0, "lighting": 4.6, "background": 4.2,
                "makeup": 3.6, "hair": 3.4, "camera": 3.0, "color": 2.6,
                "mood": 2.4, "pose": 2.2, "accessories": 2.0, "outfit": 1.4},
    "half": {"outfit": 5.0, "background": 4.6, "pose": 4.2, "lighting": 4.0,
             "expression": 3.2, "camera": 2.8, "color": 2.6, "hair": 2.4,
             "accessories": 2.2, "mood": 2.2, "makeup": 1.6},
    "full": {"background": 5.0, "location": 4.8, "outfit": 4.6, "pose": 4.4,
             "lighting": 3.8, "camera": 2.8, "color": 2.6, "footwear": 2.2,
             "accessories": 2.0, "mood": 2.0, "hair": 1.6},
    "unknown": {"background": 4.6, "lighting": 4.0, "outfit": 3.8, "pose": 3.4,
                "expression": 3.0, "camera": 2.6, "color": 2.4, "mood": 2.0},
}
DEFAULT_PRIORITY = 1.0


# ------------------------------------------------------------------ helpers

def _text(value: Any) -> str:
    return str(value).strip() if value is not None else ""


def _priority(group: str, shot_type: str) -> float:
    table = PRIORITY.get(shot_type) or PRIORITY["unknown"]
    return table.get(prompt_mod.canon_group(group), DEFAULT_PRIORITY)


def _body_edit_allowed(profile: dict) -> bool:
    """Body morphing is refused unless the profile explicitly permits it."""
    for source in (profile, profile.get("policy") if isinstance(profile, dict) else None):
        if isinstance(source, dict):
            for key in ("allow_body_changes", "allow_body_edit"):
                if key in source:
                    return bool(source[key])
    return False


def _run_id(brief: dict, options: Any, n_previews: int) -> str:
    for key in ("run_id", "id"):
        value = _text(brief.get(key))
        if value:
            return value
    payload = json.dumps(
        {"original": _text(brief.get("original_id")) or _text(brief.get("sha256"))
         or _text(brief.get("path")),
         "options": options if isinstance(options, (dict, list)) else str(options),
         "n": n_previews},
        sort_keys=True, default=str)
    return "auto_" + hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


def variant_seed(run_id: str, index: int) -> int:
    """Deterministic, positive, and stable across processes and machines."""
    raw = "%s|%d" % (run_id, int(index))
    digest = hashlib.sha256(raw.encode("utf-8")).hexdigest()
    return int(digest[:8], 16) & 0x7FFFFFFF


def _combos(crossed: dict[str, list[dict]], count: int) -> tuple[list[dict], bool, bool]:
    """Combinations of the crossed groups.

    Round robin first, which guarantees every chosen value appears before any
    value repeats, then the remaining cartesian product in a fixed order.
    Returns (combos, some_values_left_out, fewer_combos_than_asked).
    """
    if not crossed:
        return [], False, False
    groups = [(g, crossed[g]) for g in sorted(crossed)]
    combos: list[dict] = []
    seen: set[tuple] = set()

    max_len = max(len(values) for _, values in groups)
    for i in range(max_len):
        combo = {g: values[i % len(values)]["value"] for g, values in groups}
        key = tuple(sorted(combo.items()))
        if key in seen:
            continue
        seen.add(key)
        combos.append(combo)

    left_out = max_len > count
    if len(combos) < count:
        product = itertools.product(*[values for _, values in groups])
        for tup in itertools.islice(product, PRODUCT_CAP):
            combo = {g: opt["value"] for (g, _), opt in zip(groups, tup)}
            key = tuple(sorted(combo.items()))
            if key in seen:
                continue
            seen.add(key)
            combos.append(combo)
            if len(combos) >= count:
                break
    return combos[:count], left_out, len(combos) < count


def _why(varying: list[tuple[str, dict]], duplicate: bool) -> str:
    if not varying:
        if duplicate:
            return ("Misma combinacion que otra vista; solo cambia la semilla "
                    "para darte otra toma.")
        return "Todas tus elecciones estan fijas: esta vista las aplica tal cual."
    bits = ["%s: %s" % (prompt_mod.group_label_es(group),
                        opt.get("label") or opt.get("value"))
            for group, opt in varying]
    text = "Cambia " + "; ".join(bits) + "."
    if duplicate:
        text += " Repite combinacion, cambia la semilla."
    return text


# -------------------------------------------------------------------- plan

def plan_run(brief: dict, options: dict, n_previews: int, profile: dict,
             style: dict, learning: dict | None = None) -> dict:
    """Turn one set of chosen options into N reproducible, distinct variants."""
    brf = brief if isinstance(brief, dict) else {}
    prof = profile if isinstance(profile, dict) else {}
    sty = style if isinstance(style, dict) else {}
    user_id = _text(brf.get("user_id")) or None
    shot = prompt_mod.shot_type_of(brf)
    notes: list[str] = []

    try:
        count = int(n_previews)
    except (TypeError, ValueError):
        count = 1
    ceiling = int(SETTINGS.limits.max_previews_per_run)
    if count > ceiling:
        notes.append("Se limitaron las vistas previas a %d, el maximo por tanda."
                     % ceiling)
    count = max(1, min(ceiling, count))

    chosen = prompt_mod.normalize_options(options, user_id)
    body_allowed = _body_edit_allowed(prof)

    # ---- policy filter ------------------------------------------------
    allowed: dict[str, list[dict]] = {}
    for group in sorted(chosen):
        kept: list[dict] = []
        for opt in chosen[group]:
            reason = prompt_mod.body_change_reason(opt)
            if reason and not body_allowed:
                notes.append(
                    "Se ignoro '%s' del grupo '%s' porque %s: el perfil protege "
                    "tus proporciones reales." % (opt.get("label") or opt["value"],
                                                  prompt_mod.group_label_es(group),
                                                  reason))
                continue
            if not prompt_mod.applies_to_shot(opt, shot):
                notes.append(
                    "'%s' no esta pensado para este encuadre (%s), pero se "
                    "respeta porque lo elegiste tu."
                    % (opt.get("label") or opt["value"], shot))
            kept.append(opt)
        allowed[group] = kept

    locked_opts = {g: v[0] for g, v in allowed.items() if len(v) == 1}
    locked = {g: opt["value"] for g, opt in locked_opts.items()}
    crossed = {g: v for g, v in allowed.items() if len(v) > 1}
    for group, opt in locked_opts.items():
        notes.append("'%s' fijo en '%s': igual en todas las vistas."
                     % (prompt_mod.group_label_es(group),
                        opt.get("label") or opt["value"]))

    # ---- free groups the robot varies by itself -----------------------
    catalog = prompt_mod.catalog_groups(shot, user_id)
    free_candidates: list[tuple[str, list[dict]]] = []
    for group, values in catalog.items():
        if group in locked or group in crossed:
            continue
        if prompt_mod.canon_group(group) in NEVER_FREE:
            continue
        usable = [opt for opt in values
                  if body_allowed or not prompt_mod.body_change_reason(opt)]
        if len(usable) >= 2:
            free_candidates.append((group, usable))
    free_candidates.sort(key=lambda item: (-_priority(item[0], shot), item[0]))

    combos, left_out, short = _combos(crossed, count)
    base = len(combos) or 1
    free_used: list[tuple[str, list[dict]]] = []
    if base < count:
        reach = base
        for group, values in free_candidates:
            if len(free_used) >= MAX_FREE_GROUPS:
                break
            free_used.append((group, values))
            reach *= len(values)
            if reach >= count:
                break

    for group, values in free_used:
        notes.append("Nadie eligio '%s': el robot lo varia (%d valores) para que "
                     "las vistas no se parezcan entre si."
                     % (prompt_mod.group_label_es(group), len(values)))
    if base < count and not free_used:
        notes.append("No hay grupos libres en el catalogo para variar: algunas "
                     "vistas repetiran combinacion y solo cambiara la semilla.")
    if left_out:
        notes.append("Elegiste mas valores que vistas previas: en esta tanda no "
                     "apareceran todos. Sube el numero de vistas para verlos.")
    if short and free_used:
        notes.append("Tus elecciones dan menos combinaciones que vistas pedidas; "
                     "se completan variando los grupos libres.")
    for group, values in allowed.items():
        if not values and group not in {g for g, _ in free_used}:
            notes.append("Dejaste '%s' sin elegir y el catalogo no ofrece "
                         "alternativas para este encuadre."
                         % prompt_mod.group_label_es(group))

    # ---- build the variants -------------------------------------------
    run_id = _run_id(brf, options, count)
    overrides = brf.get("params") if isinstance(brf.get("params"), dict) else None
    by_value: dict[tuple[str, str], dict] = {}
    for group, values in allowed.items():
        for opt in values:
            by_value[(group, opt["value"])] = opt

    variants: list[dict] = []
    seen_combo: set[tuple] = set()
    duplicates = 0
    for index in range(count):
        choices: dict[str, str] = dict(locked)
        used_opts: list[dict] = list(locked_opts.values())
        varying: list[tuple[str, dict]] = []

        if combos:
            combo = combos[index % len(combos)]
            for group, value in combo.items():
                choices[group] = value
                opt = by_value.get((group, value))
                if opt:
                    used_opts.append(opt)
                    varying.append((group, opt))

        stride = 1
        for group, values in free_used:
            opt = dict(values[(index // stride) % len(values)])
            opt["auto"] = True
            stride *= len(values)
            choices[group] = opt["value"]
            used_opts.append(opt)
            varying.append((group, opt))

        key = tuple(sorted(choices.items()))
        duplicate = key in seen_combo
        if duplicate:
            duplicates += 1
        seen_combo.add(key)

        params = prompt_mod.merge_params(sty, used_opts, learning=None,
                                         overrides=overrides)
        # ``options`` is shaped exactly like the input of build_prompt, so the
        # orchestrator can hand this variant straight to the prompt author.
        resolved: dict[str, list[dict]] = {}
        for opt in used_opts:
            resolved.setdefault(opt.get("group") or "", []).append(opt)
        resolved.pop("", None)
        variants.append({
            "index": index,
            "choices": choices,
            "seed": variant_seed(run_id, index),
            "params": params,
            "why": _why(varying, duplicate),
            "options": resolved,
        })

    if duplicates:
        notes.append("%d vista(s) repiten combinacion porque no hay mas opciones "
                     "distintas; cambian solo la semilla." % duplicates)

    varied: list[str] = []
    for group in sorted({g for v in variants for g in v["choices"]}):
        values = {v["choices"].get(group) for v in variants}
        if len(values) > 1:
            varied.append(group)
        elif count > 1 and group not in locked:
            notes.append("'%s' quedo constante en esta tanda."
                         % prompt_mod.group_label_es(group))

    plan = {
        "variants": variants,
        "locked": locked,
        "varied": varied,
        "notes": notes,
        "run_id": run_id,
        "shot_type": shot,
        "n_requested": count,
    }
    if isinstance(learning, dict) and learning:
        plan = learning_mod.apply_learning(plan, learning)
    return plan
