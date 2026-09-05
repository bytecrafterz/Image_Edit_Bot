"""Prompt authoring - the part of the robot that replaces typing by hand.

The client can already write a prompt.  What she cannot do, forty times a day,
is retype the same twelve safeguards in the same order, derived from what each
particular photograph actually contains.  This module does exactly that: it
reads the measured identity profile plus the analysis of the original and
composes the positive prompt in a fixed professional order - who the subject is,
what is being changed, what must be preserved, the scene, the camera, the
finish - and the negative prompt from four mandatory blocks.

Three invariants turn a text template into a control system.  The identity
clause is never optional and names every attribute that must survive the edit;
the negative prompt always carries the no-beautify block, because the failure
the client actually lived through was a tool that quietly slimmed her and
reshaped her face; and a request that changes what she is wearing always says
what the whole outfit is and always forbids underwear showing under it, because
the second failure she lived through was a shirt painted over the lingerie of
the source photograph with bare legs below it.  ``tokens`` reports every
fragment that went in, so the report page can explain why the prompt says what
it says.
"""
from __future__ import annotations

import re
from typing import Any

from .. import db
from ..catalog import options as catalog_mod
from . import learning as learning_mod

# --------------------------------------------------------------- mandatory

# Verbatim from docs/CONTRACTS.md section 3.  Do not edit without editing that.
NO_BEAUTIFY = (
    "slimmer body, slimmed waist, narrowed shoulders, reshaped face, "
    "airbrushed skin, plastic skin, beauty filter, face slimming, "
    "body slimming, changed skin tone, removed tattoos, altered breast size, "
    "different person"
)

ANATOMY_NEGATIVE = (
    "extra fingers, fused fingers, missing fingers, extra limbs, "
    "deformed hands, duplicated person, distorted face, warped anatomy, "
    "extra arms, extra legs, malformed feet, twisted joints, broken wrist, "
    "asymmetric eyes, misplaced eyes, floating limbs"
)

ARTEFACT_NEGATIVE = (
    "blurry, out of focus, low resolution, jpeg artifacts, compression noise, "
    "banding, halo, oversharpened, seam, ghosting, smeared texture, "
    "watermark, text, logo, signature, border, cartoon, illustration, "
    "painting, 3d render, cgi, doll, mannequin, waxy skin"
)

# The fourth mandatory block, and every term in it was read off one delivered
# image rather than imagined.  Attempt att_d6fb1c97f9874a82b38d35c3 asked for
# "change only: a crisp white cotton poplin shirt, sleeves lightly rolled" over
# a photograph of the client in lingerie.  Its negative prompt is stored beside
# it and it is 62 terms long - slimming, fingers, seams, watermarks - and does
# not contain the words underwear, lingerie, legs or trousers anywhere.  The
# image came back exactly as asked: the shirt painted on top of the lingerie
# that was already there, bare legs below it, a dark lace edge showing at the
# hem.  The three blocks above forbid a different person and a broken hand;
# nothing forbade a half dressed one, so this block does.
#
# It is applied whenever the request changes what she is wearing, in every
# framing, because underwear showing under a new garment is wrong on a close up
# too.  ``NO_BARE_LEGS`` and ``NO_BARE_HIPS`` are the part that depends on the
# garment and only go in when the lower half is in frame - forbidding "bare
# legs" under a summer dress would be forbidding the dress.
NO_UNDERWEAR = (
    "visible underwear, exposed lingerie, bra visible under the garment, "
    "knickers, thong, lace underwear showing at the hem, "
    "new garment worn over lingerie, new garment layered over the source "
    "clothing, the source clothing still visible under the new outfit, "
    "half dressed"
)

NO_BARE_LEGS = (
    "bare legs, bare thighs, naked lower body, missing trousers, no trousers, "
    "top worn without bottoms, underwear instead of trousers, "
    "shirt worn as a dress, swimwear, bikini"
)

NO_BARE_HIPS = (
    "naked lower body, missing skirt, no skirt, top worn without bottoms, "
    "underwear instead of a skirt, swimwear, bikini"
)

# The same failure read backwards.  Six bottoms were added to the wardrobe so
# that a user could overrule the automatic trousers, and a bottom chosen on its
# own arrives here as a complete outfit request with no upper half in it: of the
# 600 one and two garment selections the catalogue allows, 42 (the 21 made only
# of bottoms, at both framings) named nothing above the waist while the request
# still said "in nothing else" and no longer preserved the source clothing.
# This block and DEFAULT_TOP are the answer, and it goes in only where the chest
# is genuinely unnamed - the chest is in frame at every framing, unlike the legs.
NO_BARE_TORSO = (
    "topless, bare chest, bare breasts, exposed nipples, naked upper body, "
    "missing top, no shirt, bra as outerwear, underwear instead of a top"
)

# Requirement 3 of the brief, in the words a diffusion model acts on.  The
# source photographs this product is built from are of a person in minimal
# clothing, and "change only: a shirt" is an instruction to ADD a shirt to what
# is already in the frame - which is precisely what the engine did.  Saying
# replace rather than add costs one sentence.
# Said forwards, and that is deliberate.  The sentence used to end "leave no
# underwear and no part of the source clothing visible anywhere in the frame",
# which is the requirement stated backwards - by naming the thing it forbids.
# On the FLUX endpoints there is no negative_prompt field, so this text is the
# text that goes on the wire, and providers/fal.py measured what that costs:
# the product's own safety/guard.py refuses this very string
# (is_intimate_request -> True on 'underwear'), and the only two paid calls
# that ever carried this vocabulary are the two fal reviewed and returned
# black.  Complete coverage is the same instruction and a stronger one - it
# says what the picture must contain instead of what it must not.
OUTFIT_REPLACE = (
    "dress the subject in this outfit and in nothing else: paint the complete "
    "garment onto the body, replace whatever the source photograph shows "
    "instead of layering the new garment over it, and let this outfit be the "
    "only clothing visible anywhere in the frame, opaque fabric, covering the "
    "torso, the hips and the legs completely"
)

IDENTITY_CLAUSE = (
    "the exact same person as in the source photograph, identity locked: "
    "same face structure and jawline, same eyes, same nose, same mouth, "
    "same hair colour, length and hairline, same skin tone and real skin "
    "texture with pores and blemishes, same body shape and proportions, "
    "same bust, waist and hip proportions, same shoulder width, same hands "
    "and fingers, same tattoos, moles and marks in the same places, "
    "unretouched and unfiltered"
)

QUALITY_LANGUAGE = (
    "professional photograph, natural skin texture with visible pores and fine "
    "lines, individual hair strands, true to life colour, realistic fabric "
    "texture, correct anatomy, sharp focus on the eyes, natural depth of "
    "field, high dynamic range, straight out of camera, no digital retouching"
)

# ------------------------------------------------------------------ params

BASE_PARAMS: dict[str, float] = {
    "strength": 0.50,       # img2img denoise; lower stays closer to the real photo
    "guidance": 4.5,
    "steps": 30,
    "identity_weight": 0.85,
}

PARAM_BOUNDS: dict[str, tuple[float, float]] = {
    "strength": (0.20, 0.92),
    "guidance": (1.0, 14.0),
    "steps": (8, 60),
    "identity_weight": (0.30, 1.0),
}

INT_PARAMS = ("steps",)

# ------------------------------------------------------------- vocabularies

GROUP_CANON: dict[str, str] = {
    "background": "background", "fondo": "background", "backdrop": "background",
    "scene": "scene", "escena": "scene", "setting": "scene", "entorno": "scene",
    "location": "location", "lugar": "location", "place": "location",
    "outfit": "outfit", "ropa": "outfit", "clothing": "outfit",
    "garment": "outfit", "vestuario": "outfit", "wardrobe": "outfit",
    # A trouser choice is not a separate subject from a shirt choice: both are
    # the outfit, and treating them as one group is what lets an explicit
    # bottom silence the automatic one.
    "clothing_bottom": "outfit", "bottoms": "outfit", "bottom": "outfit",
    "pantalon": "outfit", "falda": "outfit",
    "footwear": "footwear", "calzado": "footwear", "shoes": "footwear",
    "accessories": "accessories", "accesorios": "accessories",
    "pose": "pose", "postura": "pose",
    "expression": "expression", "expresion": "expression", "gesto": "expression",
    "hair": "hair", "peinado": "hair", "hairstyle": "hair", "cabello": "hair",
    "makeup": "makeup", "maquillaje": "makeup",
    "lighting": "lighting", "luz": "lighting", "light": "lighting",
    "iluminacion": "lighting",
    "camera": "camera", "camara": "camera", "lens": "camera", "objetivo": "camera",
    "framing": "framing", "encuadre": "framing", "angle": "framing",
    "angulo": "framing", "crop": "framing",
    "color": "color", "color_grade": "color", "grading": "color", "tono": "color",
    "mood": "mood", "ambiente": "mood", "atmosfera": "mood",
    "style": "style", "estilo": "style",
    "time_of_day": "time_of_day", "hora": "time_of_day", "momento": "time_of_day",
    "season": "season", "temporada": "season", "estacion": "season",
    "weather": "weather", "clima": "weather",
    "props": "props", "objetos": "props", "atrezzo": "props",
    "body": "body", "cuerpo": "body", "figura": "body", "silueta": "body",
}

GROUP_ES: dict[str, str] = {
    "background": "fondo", "scene": "escena", "location": "lugar",
    "outfit": "ropa", "footwear": "calzado", "accessories": "accesorios",
    "pose": "pose", "expression": "expresion", "hair": "peinado",
    "makeup": "maquillaje", "lighting": "luz", "camera": "camara",
    "framing": "encuadre", "color": "color", "mood": "ambiente",
    "style": "estilo", "time_of_day": "hora del dia", "season": "temporada",
    "weather": "clima", "props": "objetos", "body": "cuerpo",
}

SHOT_FRAMING_EN: dict[str, str] = {
    "closeup": "tight head and shoulders portrait, same head size in frame as "
               "the source photograph",
    "half": "waist up framing, same camera distance as the source photograph",
    "full": "full body in frame from head to feet, nothing cropped, same "
            "camera distance as the source photograph",
    "unknown": "same framing and camera distance as the source photograph",
}

CAMERA_BY_SHOT: dict[str, str] = {
    "closeup": "shot on a full frame camera with an 85mm portrait lens at "
               "f/2.0, eye level, natural facial perspective, eyes tack sharp",
    "half": "shot on a full frame camera with a 50mm lens at f/2.8, chest "
            "height, natural perspective, clean subject separation",
    "full": "shot on a full frame camera with a 35mm lens at f/4, camera at "
            "chest height, natural body proportions, no wide angle distortion",
    "unknown": "shot on a full frame camera with a 50mm lens at f/2.8, eye "
               "level, natural perspective",
}

HAIR_LENGTH_EN: dict[str, str] = {
    "short": "short", "medium": "shoulder length", "long": "long",
}

REGION_EN: dict[str, str] = {
    "face": "face", "neck": "neck", "chest": "chest", "abdomen": "abdomen",
    "left_torso": "left side of the torso", "right_torso": "right side of the torso",
    "left_upper_arm": "left upper arm", "right_upper_arm": "right upper arm",
    "left_forearm": "left forearm", "right_forearm": "right forearm",
    "left_hand": "left hand", "right_hand": "right hand",
    "left_thigh": "left thigh", "right_thigh": "right thigh",
    "left_shin": "left shin", "right_shin": "right shin",
    "left_foot": "left foot", "right_foot": "right foot",
    "left_arm": "left arm", "right_arm": "right arm",
    "hands": "hands", "arms": "arms", "legs": "legs", "hair": "hair",
    "upper_body": "upper body", "lower_body": "lower body",
    "background": "background", "unknown": "body",
}

REGION_ES: dict[str, str] = {
    "face": "el rostro", "neck": "el cuello", "chest": "el pecho",
    "abdomen": "el abdomen", "left_hand": "la mano izquierda",
    "right_hand": "la mano derecha", "left_forearm": "el antebrazo izquierdo",
    "right_forearm": "el antebrazo derecho", "left_upper_arm": "el brazo izquierdo",
    "right_upper_arm": "el brazo derecho", "left_thigh": "el muslo izquierdo",
    "right_thigh": "el muslo derecho", "left_shin": "la pierna izquierda",
    "right_shin": "la pierna derecha", "left_foot": "el pie izquierdo",
    "right_foot": "el pie derecho", "hands": "las manos", "arms": "los brazos",
    "legs": "las piernas", "hair": "el pelo", "background": "el fondo",
    "unknown": "la zona senalada",
}

MARK_EN: dict[str, str] = {
    "tattoo": "tattoo", "mole": "mole", "scar": "scar",
    "birthmark": "birthmark", "freckles": "freckles", "mark": "mark",
}

# ------------------------------------------------------- body change policy

# A body morphing request is the one thing this product refuses by default: it
# is precisely what the client's previous tool did to her without asking.
BODY_GROUPS = ("body",)

BODY_CHANGE_RE = re.compile(
    r"(slim(mer|ming)?\b|thinner|skinny|lose weight|weight loss|narrow(er)? waist|"
    r"smaller waist|waist reduction|bigger (bust|breast|hips|butt)|"
    r"breast (enlarge|reduc|augment)|butt lift|hourglass figure|curvier|"
    r"reshape (the )?body|body reshap|liposuction|six pack|flat stomach|"
    r"adelgaz|delgad|mas flac|cintura mas (estrecha|fina)|reducir cintura|"
    r"aumentar (busto|pecho|caderas)|busto mas|pecho mas grande|"
    r"caderas mas anchas|vientre plano|estilizar (el )?cuerpo|"
    r"cuerpo mas (delgado|estilizado))",
    re.IGNORECASE,
)


# ------------------------------------------------------------------ helpers

def _norm_ws(value: Any) -> str:
    if value is None:
        return ""
    return re.sub(r"\s+", " ", str(value)).strip().strip(" ,;.")


def _f(value: Any, default: float | None = 0.0) -> float | None:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return default
    if out != out or out in (float("inf"), float("-inf")):
        return default
    return out


def _lab(values: Any) -> list[float] | None:
    if not isinstance(values, (list, tuple)) or len(values) < 3:
        return None
    out = [_f(v, None) for v in list(values)[:3]]
    return [float(v) for v in out] if all(v is not None for v in out) else None


def _join(parts: list[str], sep: str = ", ") -> str:
    seen: set[str] = set()
    out: list[str] = []
    for part in parts:
        text = _norm_ws(part)
        if not text:
            continue
        key = text.lower()
        if key in seen:
            continue
        seen.add(key)
        out.append(text)
    return sep.join(out)


def _join_terms(parts: list[str]) -> str:
    """Comma list of unique terms.  The first block keeps its exact wording,
    which is what makes the no-beautify block verbatim rather than merged."""
    seen: set[str] = set()
    out: list[str] = []
    for part in parts:
        for term in str(part or "").split(","):
            text = _norm_ws(term)
            if not text or text.lower() in seen:
                continue
            seen.add(text.lower())
            out.append(text)
    return ", ".join(out)


def _sentences(parts: list[str]) -> str:
    body = [p for p in (_norm_ws(x) for x in parts) if p]
    return (". ".join(body) + ".") if body else ""


def canon_group(group: Any) -> str:
    key = _norm_ws(group).lower().replace(" ", "_").replace("-", "_")
    return GROUP_CANON.get(key, key)


def group_label_es(group: Any) -> str:
    key = canon_group(group)
    return GROUP_ES.get(key, _norm_ws(group).replace("_", " ") or "grupo")


def _strip_name(text: str, profile: dict) -> str:
    """Never echo a person's name into a provider prompt."""
    name = _norm_ws((profile or {}).get("person_name"))
    if not name or len(name) < 3:
        return text
    out = text
    for token in name.split():
        if len(token) < 3:
            continue
        out = re.sub(r"\b%s\b" % re.escape(token), "the subject", out,
                     flags=re.IGNORECASE)
    return _norm_ws(out)


def _shot_ok(shot_types: Any, shot_type: str) -> bool:
    raw = _norm_ws(shot_types)
    if not raw:
        return True
    parts = [p.strip().lower() for p in raw.split(",") if p.strip()]
    if not parts or "all" in parts or "todos" in parts:
        return True
    return (shot_type or "unknown").lower() in parts


def applies_to_shot(option: Any, shot_type: str) -> bool:
    """Does this catalog value make sense for this framing?"""
    if not isinstance(option, dict):
        return True
    return _shot_ok(option.get("shot_types"), shot_type)


def shot_type_of(brief: dict) -> str:
    for key in ("shot_type", "shot"):
        raw = (brief or {}).get(key)
        if isinstance(raw, dict):
            raw = raw.get("shot_type")
        value = _norm_ws(raw).lower()
        if value in ("closeup", "half", "full"):
            return value
    return "unknown"


# ---------------------------------------------------------- option handling

def _looks_like_option(value: Any) -> bool:
    if not isinstance(value, dict):
        return False
    keys = ("value", "value_key", "key", "prompt_fragment", "prompt",
            "label_es", "label", "id")
    return any(k in value for k in keys)


def _option_from_row(row: dict) -> dict:
    params = row.get("params")
    return {
        "group": _norm_ws(row.get("group_key")),
        "value": _norm_ws(row.get("value_key")),
        "label": _norm_ws(row.get("label_es")) or _norm_ws(row.get("value_key")),
        "label_en": _norm_ws(row.get("label_en")),
        "prompt": _norm_ws(row.get("prompt_fragment")),
        "negative": _norm_ws(row.get("negative_fragment")),
        "params": dict(params) if isinstance(params, dict) else {},
        "shot_types": _norm_ws(row.get("shot_types")),
        "sort_order": int(_f(row.get("sort_order"), 0) or 0),
        "source": "catalog",
    }


def catalog_option(group_key: str, value_key: str,
                   user_id: str | None = None) -> dict | None:
    """One catalog row, user owned rows winning over the built in ones."""
    sql = ("SELECT * FROM options WHERE group_key=? AND value_key=? AND enabled=1 "
           "AND (user_id IS NULL OR user_id=?) "
           "ORDER BY (user_id IS NULL) ASC, sort_order ASC LIMIT 1")
    try:
        row = db.q1(sql, (str(group_key), str(value_key), user_id))
    except Exception:
        return None
    rec = db.row_to_dict(row)
    return _option_from_row(rec) if rec else None


def catalog_groups(shot_type: str = "",
                   user_id: str | None = None) -> dict[str, list[dict]]:
    """Every enabled option value that applies to this shot type, by group."""
    sql = ("SELECT * FROM options WHERE enabled=1 AND (user_id IS NULL OR user_id=?) "
           "ORDER BY group_key ASC, sort_order ASC, value_key ASC")
    try:
        rows = db.q(sql, (user_id,))
    except Exception:
        return {}
    out: dict[str, list[dict]] = {}
    owned: dict[tuple[str, str], bool] = {}
    for raw in rows:
        rec = db.row_to_dict(raw) or {}
        if shot_type and not _shot_ok(rec.get("shot_types"), shot_type):
            continue
        opt = _option_from_row(rec)
        if not opt["group"] or not opt["value"]:
            continue
        bucket = out.setdefault(opt["group"], [])
        key = (opt["group"], opt["value"])
        if key in owned:
            if not owned[key] and rec.get("user_id"):
                for i, existing in enumerate(bucket):
                    if existing["value"] == opt["value"]:
                        bucket[i] = opt
                        owned[key] = True
                        break
            continue
        owned[key] = bool(rec.get("user_id"))
        bucket.append(opt)
    return out


def _coerce_option(group: str, value: Any, user_id: str | None) -> dict | None:
    """Accept a bare value key, a catalog row, or an already resolved option."""
    if value is None:
        return None
    if isinstance(value, (str, int, float)) and not isinstance(value, bool):
        key = _norm_ws(value)
        if not key:
            return None
        found = catalog_option(group, key, user_id)
        if found:
            return found
        return {"group": group, "value": key,
                "label": key.replace("_", " "), "label_en": "",
                "prompt": "", "negative": "", "params": {},
                "shot_types": "", "sort_order": 0, "source": "given"}
    if not isinstance(value, dict):
        return None

    key = _norm_ws(value.get("value") or value.get("value_key")
                   or value.get("key") or value.get("id"))
    base = catalog_option(group, key, user_id) if key else None
    opt = base or {"group": group, "value": key,
                   "label": key.replace("_", " "), "label_en": "",
                   "prompt": "", "negative": "", "params": {},
                   "shot_types": "", "sort_order": 0, "source": "given"}
    opt = dict(opt)
    given_prompt = _norm_ws(value.get("prompt") or value.get("prompt_fragment"))
    given_neg = _norm_ws(value.get("negative") or value.get("negative_fragment"))
    given_label = _norm_ws(value.get("label") or value.get("label_es"))
    given_label_en = _norm_ws(value.get("label_en"))
    given_params = value.get("params")
    if given_label_en:
        opt["label_en"] = given_label_en
    if given_prompt:
        opt["prompt"] = given_prompt
    if given_neg:
        opt["negative"] = given_neg
    if given_label:
        opt["label"] = given_label
    if isinstance(given_params, dict):
        merged = dict(opt.get("params") or {})
        merged.update(given_params)
        opt["params"] = merged
    if value.get("auto"):
        opt["auto"] = True
    if not opt["value"] and not opt["prompt"]:
        return None
    if not opt["value"]:
        opt["value"] = re.sub(r"[^a-z0-9]+", "_", opt["prompt"].lower())[:32]
    if not opt["label"]:
        opt["label"] = opt["value"].replace("_", " ")
    return opt


def normalize_options(options: Any, user_id: str | None = None) -> dict[str, list[dict]]:
    """``{group: [option, ...]}`` from anything the API or the planner passes.

    An empty list is meaningful: it marks a group the user left free, which the
    planner is then allowed to vary on its own.
    """
    if not isinstance(options, dict):
        return {}
    out: dict[str, list[dict]] = {}
    for raw_group, raw_value in options.items():
        group = _norm_ws(raw_group)
        if not group or group.startswith("_"):
            continue
        values: list[Any]
        if raw_value is None:
            values = []
        elif isinstance(raw_value, (list, tuple, set)):
            values = list(raw_value)
        elif isinstance(raw_value, dict):
            if _looks_like_option(raw_value):
                values = [raw_value]
            else:
                values = []
                for key, item in raw_value.items():
                    if isinstance(item, dict):
                        entry = dict(item)
                        entry.setdefault("value", key)
                        values.append(entry)
                    elif item is True:
                        values.append(key)
                if not values:
                    continue          # a params style dict, not an option group
        else:
            values = [raw_value]

        chosen: list[dict] = []
        for item in values:
            opt = _coerce_option(group, item, user_id)
            if opt and not any(o["value"] == opt["value"] for o in chosen):
                chosen.append(opt)
        out[group] = chosen
    return out


def body_change_reason(option: dict) -> str:
    """Spanish reason when an option would modify the real body, else ''."""
    if not isinstance(option, dict):
        return ""
    if option.get("changes_body") or option.get("body_change"):
        return "modifica el cuerpo real"
    if canon_group(option.get("group")) in BODY_GROUPS:
        return "pertenece al grupo cuerpo"
    haystack = " ".join([_norm_ws(option.get("value")),
                         _norm_ws(option.get("label")),
                         _norm_ws(option.get("label_en")),
                         _norm_ws(option.get("prompt"))])
    if BODY_CHANGE_RE.search(haystack):
        return "pide cambiar la silueta"
    return ""


# ------------------------------------------------------------------ params

def merge_params(style: dict | None, chosen: list[dict] | None,
                 learning: dict | None = None,
                 overrides: dict | None = None) -> dict:
    """Style defaults, then option params, then learning, then explicit values."""
    params = dict(BASE_PARAMS)
    sty = style if isinstance(style, dict) else {}
    defaults = sty.get("defaults") if isinstance(sty.get("defaults"), dict) else {}
    # A flat ``defaults`` may carry unrelated keys, so only known params are read
    # from it; an explicit params dict is taken whole (providers accept extras).
    params.update({k: v for k, v in defaults.items()
                   if k in PARAM_BOUNDS and v is not None})
    for source in (defaults.get("params"), sty.get("params")):
        if isinstance(source, dict):
            params.update({k: v for k, v in source.items() if v is not None})
    for opt in (chosen or []):
        extra = opt.get("params") if isinstance(opt, dict) else None
        if isinstance(extra, dict):
            params.update({k: v for k, v in extra.items() if v is not None})
    if isinstance(learning, dict) and learning:
        params, _ = learning_mod.apply_weights_to_params(params, learning)
    if isinstance(overrides, dict):
        params.update({k: v for k, v in overrides.items() if v is not None})
    return clamp_params(params)


def clamp_params(params: dict | None) -> dict:
    out: dict[str, Any] = {}
    for key, value in (params or {}).items():
        num = _f(value, None)
        if num is None:
            out[key] = value
            continue
        lo, hi = PARAM_BOUNDS.get(key, (None, None))
        if lo is not None:
            num = max(lo, min(hi, num))
        out[key] = int(round(num)) if key in INT_PARAMS else round(float(num), 4)
    return out


# -------------------------------------------------- descriptions from pixels

def hair_description(profile: dict) -> str:
    hair = (profile or {}).get("hair") or {}
    lab = _lab(hair.get("lab_mean"))
    length = HAIR_LENGTH_EN.get(_norm_ws(hair.get("length")).lower(), "")
    colour = _hair_colour(lab)
    parts = [p for p in (length, colour) if p]
    return (" ".join(parts) + " hair") if parts else ""


def _hair_colour(lab: list[float] | None) -> str:
    if lab is None:
        return ""
    lightness, red_green, blue_yellow = lab
    if lightness >= 45.0 and abs(red_green) < 4.0 and abs(blue_yellow) < 5.0:
        return "grey"
    if lightness < 22.0:
        base = "black"
    elif lightness < 34.0:
        base = "dark brown"
    elif lightness < 46.0:
        base = "medium brown"
    elif lightness < 58.0:
        base = "light brown"
    elif lightness < 68.0:
        base = "dark blonde"
    else:
        base = "blonde"
    if red_green >= 18.0 and blue_yellow >= 10.0:
        return "auburn " + base if base != "black" else "very dark auburn"
    if red_green >= 12.0 and blue_yellow >= 18.0:
        return "warm chestnut " + base
    if blue_yellow >= 24.0 and base in ("dark blonde", "blonde"):
        return "golden " + base
    if blue_yellow <= 6.0 and base not in ("black",):
        return "ash " + base
    return base


def skin_description(profile: dict) -> str:
    """Neutral, non flattering: this is a measurement, not a compliment."""
    skin = (profile or {}).get("skin") or {}
    lab = _lab(skin.get("lab_mean"))
    ita = _f(skin.get("ita_deg"), None)
    if lab is None and ita is None:
        return ""
    if ita is None and lab is not None:
        ita = 45.0 if lab[0] > 60 else 20.0
    if ita > 55.0:
        tone = "very light"
    elif ita > 41.0:
        tone = "light"
    elif ita > 28.0:
        tone = "light intermediate"
    elif ita > 10.0:
        tone = "tan intermediate"
    elif ita > -30.0:
        tone = "brown"
    else:
        tone = "deep brown"
    undertone = ""
    if lab is not None:
        if lab[2] >= 20.0:
            undertone = "warm golden undertone"
        elif lab[2] <= 12.0:
            undertone = "neutral cool undertone"
        else:
            undertone = "neutral undertone"
    return _join([tone + " skin tone", undertone,
                  "real skin texture with visible pores, moles and blemishes "
                  "kept as they are"])


def _protect_mod():
    """generation/protect.py, imported at call time and never at load.

    protect.py imports THIS module, so the dependency can only be taken in one
    direction at import time.  It is needed all the same: protect is the only
    place that knows which body zones a garment covers, and a prompt that does
    not know that asks for things the mask is about to paint over.
    """
    from . import protect as mod
    return mod


def _garment_cover(chosen: Any) -> dict:
    """Which body zones the requested garment hides, or {} if nothing new is worn."""
    try:
        return _protect_mod().garment_cover(chosen) or {}
    except Exception:                                     # noqa: BLE001
        # A prompt that cannot reach protect.py is not a prompt that fails: it
        # simply asks for every mark, exactly as it did before this existed.
        return {}


# The words that make a preserve entry a piece of clothing rather than a piece
# of her.  Deliberately narrow: it has to survive "necklace with pendant",
# "shoulder tattoo with text and heart symbols", "hair length and dark red
# color", "face shape and features" and "skin tone", which are the other five
# entries the reading of the new photograph proposes.
_GARMENT_WORDS = (
    "top", "shirt", "t-shirt", "tshirt", "blouse", "tank", "camisole", "vest",
    "sweater", "jumper", "cardigan", "hoodie", "jacket", "blazer", "coat",
    "dress", "gown", "skirt", "trousers", "pants", "jeans", "shorts",
    "leggings", "bodysuit", "jumpsuit", "romper", "suit", "outfit", "clothes",
    "clothing", "garment", "sleeve", "sleeves", "neckline", "bra", "lingerie",
    "underwear", "swimsuit", "bikini", "uniform", "robe", "kimono", "poncho",
    "overalls", "waistcoat", "tie", "scarf", "belt",
)
_GARMENT_RE = re.compile(
    r"(?<!\w)(?:%s)(?!\w)" % "|".join(re.escape(w) for w in _GARMENT_WORDS),
    re.IGNORECASE)


def _names_a_garment(text: str) -> bool:
    return bool(_GARMENT_RE.search(str(text or "")))


def marks_description(profile: dict, limit: int = 4, cover: Any = None) -> str:
    """Her permanent marks, minus the ones the new garment will be over.

    WHY A GARMENT ARGUMENT BELONGS IN A SENTENCE ABOUT TATTOOS.  The prompt of
    the paid call of 2026-09-05 asked fal, in the same breath, to paint "a
    crisp white cotton poplin shirt... covering the torso... completely" and to
    keep "the tattoo on the chest... unchanged and in the same place".  Both
    sentences were sincere and together they are impossible: a shirt over a
    tattoo covers it, and the mask had already decided to repaint those very
    pixels.  Asking a model for a contradiction does not get half of each - it
    gets whatever the model settles on, and on this path what it settles on is
    a chest the shirt does not properly cover, which is the picture an output
    reviewer flags.
    So the request is trimmed to what the mask can actually deliver: a mark in
    a zone the garment leaves bare is asked for, a mark under the new garment
    is not mentioned at all.  ``cover`` is protect.garment_cover's answer; with
    no garment in the request nothing is covered and every mark is asked for,
    which is the behaviour this function always had.
    """
    marks = (profile or {}).get("marks")
    if not isinstance(marks, list) or not marks:
        return ""
    zones = cover if isinstance(cover, dict) else {}
    zone_of: dict = {}
    if zones:
        try:
            zone_of = _protect_mod().MARK_ZONES
        except Exception:                                 # noqa: BLE001
            zone_of = {}
    parts: list[str] = []
    for mark in marks:
        if not isinstance(mark, dict):
            continue
        raw = _norm_ws(mark.get("region")).lower()
        # "desconocida" is not "cubierta": a garment nobody described might
        # leave the mark showing, and dropping it from the prompt would throw
        # away the only chance of keeping it.  Only a zone the wardrobe says
        # is covered is silent here.
        if zone_of and str(zones.get(zone_of.get(raw, ""), "")) == "cubierta":
            continue
        kind = MARK_EN.get(_norm_ws(mark.get("type")).lower(), "mark")
        region = REGION_EN.get(raw, "body")
        parts.append("the %s on the %s" % (kind, region))
        if len(parts) >= limit:
            break
    if not parts:
        return ""
    text = _join(parts)
    if len(parts) >= limit:
        # Only when the LIMIT is what stopped the list.  It used to count the
        # profile's marks, which after the filter above would promise "every
        # other visible mark" and mean the ones under the shirt.
        text += ", and every other visible mark"
    return text + " unchanged and in the same place"


def _body_clause(profile: dict) -> str:
    body = (profile or {}).get("body")
    measured = isinstance(body, dict) and bool(body)
    base = ("the real body proportions of the source photograph: shoulder "
            "width, bust, waist and hips exactly as they are, same height to "
            "torso ratio, no slimming and no reshaping")
    if measured:
        base += ", these proportions are measured and verified after generation"
    return base


def _vision_text(brief: dict, key: str, profile: dict) -> str:
    value = (brief or {}).get(key)
    if isinstance(value, (list, tuple)):
        value = ", ".join(str(v) for v in value)
    return _strip_name(_norm_ws(value), profile)


# ------------------------------------------------------------ outfit plan

# A lower half is only worth naming when there is a lower half in the picture.
# On a head and shoulders portrait the trousers are an instruction about pixels
# that do not exist, and every extra instruction is one more chance for the
# engine to reframe in order to obey it.
LOWER_BODY_SHOTS = ("half", "full", "unknown")

# What the garment ends up wearing below the waist decides which negative is
# honest.  A midi dress that shows a shin is the dress working; a shirt that
# shows a thigh is the failure this whole block exists for.
_SKIRT_LOWERS = ("skirt", "dress")


def outfit_plan(chosen: dict, shot: str) -> dict:
    """Work out what the lower half of a requested outfit is wearing.

    The rule the client asked for, in one place: an explicit bottom wins, a
    complete outfit needs nothing added, and only a top left without a bottom
    gets one filled in automatically.  Returns the extra English phrase for the
    prompt, the negatives the chosen garment makes honest, and the Spanish
    tokens that let the report say why it says what it says.
    """
    tops: list = []
    completes: list = []
    bottoms: list = []
    unclassified: list = []
    for group, opts in (chosen or {}).items():
        if canon_group(group) != "outfit":
            continue
        for opt in (opts or []):
            info = catalog_mod.garment_info(opt)
            kind = _norm_ws(info.get("kind")).lower()
            entry = (opt, info)
            if kind == "bottom":
                bottoms.append(entry)
            elif kind == "complete":
                completes.append(entry)
            elif kind == "top":
                tops.append(entry)
            else:
                unclassified.append(entry)

    plan = {"changed": bool(tops or completes or bottoms or unclassified),
            "lower_phrase": "", "upper_phrase": "", "negatives": [],
            "tokens": [], "instruction": "", "auto": False, "lower": ""}
    if not plan["changed"]:
        return plan

    # Requirement 3.  This goes in whatever was chosen: a gown painted over
    # lingerie is the same picture as a shirt painted over lingerie.
    plan["instruction"] = OUTFIT_REPLACE
    plan["negatives"].append(NO_UNDERWEAR)
    plan["tokens"].append(
        "ropa: se pide vestir el conjunto completo y sustituir la ropa de la "
        "foto original, no superponerlo encima")
    plan["tokens"].append(
        "negativo: nada de ropa interior a la vista ni prenda superpuesta "
        "sobre la ropa de la foto original")

    visible = (shot or "unknown") in LOWER_BODY_SHOTS

    def _name(option):
        return _norm_ws(option.get("label")) or _norm_ws(option.get("value"))

    # 1. An explicit bottom wins outright and nothing is added: its own prompt
    #    fragment is already in the change list.
    if bottoms:
        opt, info = bottoms[0]
        plan["lower"] = _norm_ws(info.get("lower")) or "trousers"
        plan["tokens"].append(
            "ropa: la prenda de abajo la elegiste tu (%s), no se anade ninguna"
            % _name(opt))
        # And the other half of her.  A bottom on its own says "dress her in
        # these trousers and in nothing else" over a photograph in lingerie,
        # which is the delivered failure with the halves swapped; measured over
        # every selection the wardrobe allows it is the only case that reaches
        # here with nothing named above the waist.  The chest is in frame in
        # every framing, so this does not wait for the lower body to be visible.
        if not tops and not completes:
            plan["upper_phrase"] = catalog_mod.DEFAULT_TOP
            plan["auto"] = True
            plan["negatives"].append(NO_BARE_TORSO)
            plan["tokens"].append(
                "ropa: %s solo viste de cintura para abajo, asi que se anade "
                "automaticamente una prenda de arriba (%s). Si eliges una tu, "
                "manda la tuya." % (_name(opt), catalog_mod.DEFAULT_TOP))
            plan["tokens"].append(
                "negativo: nada de torso desnudo ni falta de prenda de arriba")
    # 2. A complete outfit dresses the whole person; it only gets a phrase when
    #    its own fragment never named the lower half.
    elif completes:
        opt, info = completes[0]
        plan["lower"] = _norm_ws(info.get("lower")) or "trousers"
        extra = _norm_ws(info.get("bottom"))
        if extra and visible:
            plan["lower_phrase"] = extra
            plan["auto"] = True
            plan["tokens"].append(
                "ropa: %s es un conjunto completo y se nombra su parte de "
                "abajo (%s) para que no quede a interpretacion"
                % (_name(opt), extra))
        else:
            plan["tokens"].append(
                "ropa: %s ya viste el cuerpo entero, no hace falta anadir nada"
                % _name(opt))
    # 3. A top with no bottom: the case that produced the delivered image, and
    #    the only one where the robot chooses for her.
    elif tops:
        opt, info = tops[0]
        plan["lower"] = _norm_ws(info.get("lower")) or "trousers"
        extra = _norm_ws(info.get("bottom")) or catalog_mod.DEFAULT_BOTTOM
        if visible:
            plan["lower_phrase"] = extra
            plan["auto"] = True
            plan["tokens"].append(
                "ropa: %s solo viste de cintura para arriba, asi que se anade "
                "automaticamente la prenda de abajo (%s). Si eliges una tu, "
                "manda la tuya." % (_name(opt), extra))
        else:
            plan["tokens"].append(
                "ropa: %s solo viste de cintura para arriba, pero en un primer "
                "plano no se ve la parte de abajo y no se pide" % _name(opt))
    # 4. A garment nobody classified - a wardrobe entry the user wrote herself.
    #    Inventing trousers for a garment this code cannot read would be
    #    overruling her, so it does not, and the report says so out loud.
    else:
        opt, info = unclassified[0]
        plan["negatives"].append(NO_BARE_TORSO)
        plan["tokens"].append(
            "ropa: no se ha podido clasificar %s, asi que no se anade prenda "
            "de abajo; se exige igualmente que no se vea ropa interior ni "
            "quede el torso desnudo" % _name(opt))

    if visible and plan["lower"]:
        if plan["lower"] in _SKIRT_LOWERS:
            plan["negatives"].append(NO_BARE_HIPS)
            plan["tokens"].append(
                "negativo: nada de cuerpo desnudo de cintura para abajo bajo "
                "la falda o el vestido pedido")
        else:
            plan["negatives"].append(NO_BARE_LEGS)
            plan["tokens"].append(
                "negativo: nada de piernas desnudas ni falta de pantalon bajo "
                "la prenda pedida")
    return plan


# ------------------------------------------------------------ main builders

def build_prompt(brief: dict, profile: dict, style: dict, options: dict) -> dict:
    """Author one prompt.  Sections always arrive in the same fixed order."""
    brf = brief if isinstance(brief, dict) else {}
    prof = profile if isinstance(profile, dict) else {}
    sty = style if isinstance(style, dict) else {}
    user_id = _norm_ws(brf.get("user_id")) or None

    chosen = normalize_options(options, user_id)
    tokens: list[str] = []
    shot = shot_type_of(brf)

    # ---- 1. subject and identity ------------------------------------
    subject_bits: list[str] = []
    subject = _vision_text(brf, "subject", prof)
    subject_bits.append(subject or "the same adult person from the source photograph")
    changed_groups = {canon_group(g) for g, v in chosen.items() if v}
    if "hair" not in changed_groups:
        subject_bits.append(hair_description(prof))
    subject_bits.append(skin_description(prof))
    subject_bits.append(SHOT_FRAMING_EN.get(shot, SHOT_FRAMING_EN["unknown"]))
    subject_clause = _join(["photograph of " + subject_bits[0]] + subject_bits[1:])
    tokens.append("sujeto: " + subject_clause)

    identity_clause = IDENTITY_CLAUSE
    tokens.append("identidad: clausula obligatoria de preservacion")

    # ---- 2. what is being changed -----------------------------------
    change_bits: list[str] = []
    negatives: list[str] = []
    used_options: list[dict] = []
    # Only the garments that survive the body-change refusal below, because a
    # garment that was refused never reaches the prompt and must not be able to
    # answer "what are the legs wearing" either.
    kept: dict[str, list[dict]] = {}
    for group in sorted(chosen, key=lambda g: (canon_group(g), g)):
        for opt in chosen[group]:
            reason = body_change_reason(opt)
            if reason:
                tokens.append("bloqueado [%s]: %s - %s, no se permite cambiar "
                              "el cuerpo real" % (group_label_es(group),
                                                  opt.get("label") or opt["value"],
                                                  reason))
                continue
            fragment = (opt.get("prompt") or opt.get("label_en")
                        or opt["value"].replace("_", " "))
            fragment = _norm_ws(fragment)
            if fragment:
                change_bits.append(fragment)
            if opt.get("negative"):
                negatives.append(opt["negative"])
            used_options.append(opt)
            kept.setdefault(group, []).append(opt)
            tokens.append("cambio [%s]: %s%s" % (
                group_label_es(group), fragment,
                " (elegido por el robot)" if opt.get("auto") else ""))

    # The lower half of a requested outfit, decided once and in one place.  It
    # runs after the loop because it needs every garment that was chosen, and
    # its phrase joins "change only" so the trousers are asked for rather than
    # merely not forbidden.
    outfit = outfit_plan(kept, shot)
    if outfit["lower_phrase"]:
        change_bits.append("with " + outfit["lower_phrase"])
    if outfit["upper_phrase"]:
        change_bits.append("with " + outfit["upper_phrase"])
    negatives.extend(outfit["negatives"])
    tokens.extend(outfit["tokens"])

    # ---- 3. what must be preserved ----------------------------------
    preserve_bits: list[str] = [_body_clause(prof)]
    if "outfit" not in changed_groups:
        clothing = _vision_text(brf, "clothing", prof)
        # AND NOT THE ATTRIBUTE THE REQUEST IS CHANGING.  A colour change is
        # not an outfit change - canon_group("clothing_color") is its own
        # group - so this sentence still fired, and the prompt built for it
        # read "change only: in deep black. keep unchanged: ... a ribbed
        # sleeveless knit top in warm grey with a black skirt, same cut, colour
        # and fit".  It orders the colour changed and the colour kept in the
        # same breath.  The garment is still preserved; the word being altered
        # comes out of the promise.
        # Only the attribute actually being altered comes out.  A sheerness
        # change needs nothing removed - cut, colour and fit are all still
        # promised, and opacity was never in the promise - so "transparency"
        # is deliberately not in this list.
        holds = ["cut", "colour", "fit"]
        if "clothing_color" in changed_groups or "color" in changed_groups:
            holds.remove("colour")
        same = ("same " + ", ".join(holds[:-1]) + " and " + holds[-1]
                if len(holds) > 1 else "same " + holds[0])
        preserve_bits.append(
            ("the same garment as in the source photograph: " + clothing +
             ", " + same) if clothing else
            ("the same garment as in the source photograph, " + same))
    if "hair" not in changed_groups:
        hair_style = _vision_text(brf, "hair", prof)
        preserve_bits.append("the same hairstyle" + (": " + hair_style
                                                     if hair_style else ""))
    if "makeup" not in changed_groups:
        preserve_bits.append("the same level of makeup as in the source photograph")
    # The garment decides which of her marks can still be asked for; see
    # marks_description.  ``kept`` is what was really chosen, after the
    # catalogue resolved it, which is the same object outfit_plan just read.
    marks = marks_description(prof, cover=_garment_cover(kept))
    if marks:
        preserve_bits.append(marks)
    extra_preserve = (brf.get("preserve") if isinstance(brf.get("preserve"), list)
                      else [])
    # AND THE READING'S OWN LIST, WHICH DOES NOT KNOW WHAT WAS ASKED FOR.
    # Claude's reading of the photograph proposes what is worth keeping, and on
    # the new source it proposes "beige tank top".  Copied into an outfit
    # change that already says "change only: a crisp white cotton poplin
    # shirt", it made the paid prompt of 2026-09-05 order two different tops in
    # one sentence.  Measured on her 25 readings this drops garment entries and
    # nothing else: her necklace, her tattoo, her hair colour and her skin tone
    # are all kept, because none of them is something she takes off.
    for item in extra_preserve[:6]:
        clean = _strip_name(_norm_ws(item), prof)
        if "outfit" in changed_groups and _names_a_garment(clean):
            tokens.append("conserva: se retira \"%s\" de lo que hay que "
                          "conservar, porque es justo la prenda que has "
                          "pedido cambiar" % clean)
            continue
        preserve_bits.append(clean)
    for bit in preserve_bits:
        if bit:
            tokens.append("conserva: " + bit)

    # ---- 4. scene and lighting --------------------------------------
    scene_bits: list[str] = []
    if "background" not in changed_groups and "scene" not in changed_groups \
            and "location" not in changed_groups:
        setting = _vision_text(brf, "setting", prof)
        scene_bits.append(("the same setting as the source photograph: " + setting)
                          if setting else
                          "the same setting as the source photograph")
    if "lighting" not in changed_groups:
        light = _vision_text(brf, "lighting", prof)
        scene_bits.append(("the same lighting as the source photograph: " + light)
                          if light else
                          "the same soft natural lighting as the source photograph, "
                          "same direction and colour temperature")
    style_scene = _style_scene(sty, brf, prof)
    if style_scene:
        scene_bits.append(style_scene)
        tokens.append("estilo: " + style_scene)
    scene_clause = _join(scene_bits)
    if scene_clause:
        tokens.append("escena: " + scene_clause)

    # ---- 5. camera and lens -----------------------------------------
    camera_clause = _norm_ws(sty.get("camera")) or CAMERA_BY_SHOT.get(
        shot, CAMERA_BY_SHOT["unknown"])
    if "camera" in changed_groups or "framing" in changed_groups:
        camera_clause = _join([camera_clause,
                               "keep the requested framing above"])
    tokens.append("camara: " + camera_clause)

    # ---- 6. photographic finish -------------------------------------
    quality_clause = QUALITY_LANGUAGE
    tokens.append("acabado: fotografia real, textura de piel y pelo natural")

    prompt = _sentences([
        subject_clause,
        identity_clause,
        ("change only: " + _join(change_bits)) if change_bits else "",
        outfit["instruction"],
        "keep unchanged: " + _join(preserve_bits),
        scene_clause,
        camera_clause,
        quality_clause,
    ])

    negative_prompt = _join([NO_BEAUTIFY, ANATOMY_NEGATIVE, ARTEFACT_NEGATIVE,
                             _norm_ws(sty.get("negative_template")),
                             _norm_ws(sty.get("negative"))] + negatives)
    tokens.append("negativo: bloque anti-embellecimiento obligatorio")
    tokens.append("negativo: anatomia y artefactos")

    params = merge_params(sty, used_options,
                          learning=brf.get("learning"),
                          overrides=brf.get("params"))
    tokens.append("parametros: " + " ".join(
        "%s=%s" % (k, params[k]) for k in sorted(params)))

    return {
        "prompt": prompt,
        "negative_prompt": negative_prompt,
        "identity_clause": identity_clause,
        "params": params,
        "tokens": tokens,
    }


def _style_scene(style: dict, brief: dict, profile: dict) -> str:
    """The style contributes scene and finish language, never the whole order."""
    template = _norm_ws(style.get("prompt_template")) or _norm_ws(style.get("prompt"))
    if not template:
        return ""
    if "{" in template:
        mapping = {
            "subject": _vision_text(brief, "subject", profile),
            "identity": "", "changes": "", "preserve": "",
            "scene": _vision_text(brief, "setting", profile),
            "lighting": _vision_text(brief, "lighting", profile),
            "camera": "", "quality": "",
            "hair": hair_description(profile),
            "skin": skin_description(profile),
            "shot": shot_type_of(brief),
        }
        try:
            template = template.format_map(_Blank(mapping))
        except (ValueError, IndexError):
            template = re.sub(r"\{[^{}]*\}", "", template)
    return _strip_name(_norm_ws(template), profile)


class _Blank(dict):
    def __missing__(self, key: str) -> str:  # unknown placeholder -> nothing
        return ""


# ------------------------------------------------------------------ repair

DEFECT_FIX_EN: dict[str, str] = {
    "hand_malformed": "a natural human {where} with exactly five fingers, "
                      "correct finger proportions and natural joints, "
                      "anatomically correct thumb",
    "extra_limb": "clean natural anatomy with only the limbs the person really "
                  "has, the extra limb removed and the background continued "
                  "behind it",
    "extra_person": "only one person in the frame, the duplicated person "
                    "removed and the background continued behind it",
    "face_distorted": "a natural undistorted human face, correct facial "
                      "geometry, aligned features, same identity as the rest "
                      "of the photograph",
    "eye_asymmetry": "two natural symmetric human eyes, aligned on the same "
                     "line, same iris colour and size, natural catchlights",
    "missing_limb": "the complete natural {where}, correct length and anatomy, "
                    "attached naturally to the body",
    "texture_smear": "sharp real detail, real fabric weave and real skin "
                     "texture, no smearing",
    "duplicated_feature": "a single natural feature, nothing duplicated",
    "border_artifact": "a clean continuous edge, no seam, no halo, the "
                       "background continuing naturally",
    "oversmoothed_skin": "real skin texture with visible pores, fine lines and "
                         "natural blemishes, no smoothing",
}

DEFECT_NEG_EN: dict[str, str] = {
    "hand_malformed": "extra fingers, six fingers, seven fingers, four fingers, "
                      "fused fingers, missing fingers, bent fingers, claw hand, "
                      "double thumb, melted hand, blurry hand, mitten hand",
    "extra_limb": "extra arm, extra leg, third hand, floating limb, "
                  "disconnected limb",
    "extra_person": "second person, duplicated person, crowd, reflection of a "
                    "person, ghost figure",
    "face_distorted": "warped face, melted face, asymmetric jaw, crooked nose, "
                      "double face, different person, deformed features",
    "eye_asymmetry": "crossed eyes, misaligned eyes, different eye sizes, "
                     "different iris colours, lazy eye, extra eye",
    "missing_limb": "amputated limb, cropped limb, stump, missing hand",
    "texture_smear": "smeared texture, painted look, plastic surface, blurry "
                     "patch, watercolour",
    "duplicated_feature": "duplicated feature, mirrored feature, cloned patch, "
                          "repeated pattern",
    "border_artifact": "visible seam, hard edge, halo, colour shift, "
                       "rectangular patch",
    "oversmoothed_skin": "airbrushed skin, plastic skin, beauty filter, "
                         "blurred skin, doll skin",
}

# Structural defects need real denoise inside the mask; identity sensitive and
# texture defects need far less, or the repair invents a new person.
REPAIR_STRENGTH: dict[str, float] = {
    "hand_malformed": 0.85, "extra_limb": 0.88, "extra_person": 0.90,
    "missing_limb": 0.88, "duplicated_feature": 0.82,
    "face_distorted": 0.60, "eye_asymmetry": 0.58,
    "texture_smear": 0.50, "border_artifact": 0.45, "oversmoothed_skin": 0.42,
}


def repair_prompt(defect: dict, brief: dict, profile: dict) -> dict:
    """A tight instruction for one region only - never for the whole image."""
    dfc = defect if isinstance(defect, dict) else {}
    brf = brief if isinstance(brief, dict) else {}
    prof = profile if isinstance(profile, dict) else {}

    kind = _norm_ws(dfc.get("type")).lower() or "texture_smear"
    where_key = _norm_ws(dfc.get("where")).lower()
    where_en = REGION_EN.get(where_key, where_key.replace("_", " ") or "region")
    where_es = REGION_ES.get(where_key, "la zona senalada")

    fix = DEFECT_FIX_EN.get(kind, "natural, anatomically correct detail")
    fix = fix.replace("{where}", where_en)

    skin = skin_description(prof)
    bits = [fix,
            "matching the skin tone and lighting of the photograph",
            "same colour temperature, same grain and same sharpness as the "
            "surrounding pixels",
            "photorealistic, seamless with the rest of the image"]
    if skin and kind in ("hand_malformed", "missing_limb", "extra_limb",
                         "face_distorted", "oversmoothed_skin", "texture_smear"):
        bits.insert(1, skin)
    if kind in ("face_distorted", "eye_asymmetry"):
        bits.insert(1, IDENTITY_CLAUSE)

    prompt = _sentences([_join(bits)])
    negative = _join([DEFECT_NEG_EN.get(kind, ""), NO_BEAUTIFY,
                      ANATOMY_NEGATIVE, ARTEFACT_NEGATIVE])

    params = clamp_params({
        "strength": REPAIR_STRENGTH.get(kind, 0.6),
        "guidance": 5.0,
        "steps": 32,
        "identity_weight": 0.95 if kind in ("face_distorted", "eye_asymmetry")
        else 0.85,
    })

    severity = _f(dfc.get("severity"), 0.0) or 0.0
    tokens = [
        "reparacion: %s en %s" % (kind, where_es),
        "gravedad medida: %.2f" % severity,
        "instruccion: " + fix,
        "negativo especifico: " + (DEFECT_NEG_EN.get(kind, "") or "generico"),
        "negativo: bloque anti-embellecimiento obligatorio",
        "parametros: " + " ".join("%s=%s" % (k, params[k]) for k in sorted(params)),
    ]
    _ = brf  # the brief only shapes the full prompt; a repair stays local

    return {
        "prompt": prompt,
        "negative_prompt": negative,
        "identity_clause": IDENTITY_CLAUSE,
        "params": params,
        "tokens": tokens,
        "type": kind,
        "where": where_key,
        "summary_es": "Se repara solo %s (%s)." % (where_es, kind),
    }
