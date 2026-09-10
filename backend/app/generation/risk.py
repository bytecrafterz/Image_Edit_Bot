"""What is likely to go wrong with THIS request, said before the money moves.

The client's instruction was one sentence: "minimiza los errores de rechazo, y
si ocurre uno, ajusta las opciones para que no vuelva a ocurrir".  Read as
engineering that is three obligations, and this file is where the first and the
third of them live:

  (a) BEFORE spending, know which option combinations have already failed and
      say so - or offer the change that avoids them;
  (b) when a rejection happens, work out WHICH option caused it (the
      orchestrator's job, informed by the counters below);
  (c) REMEMBER, so the same mistake is not paid for twice, on the next run or
      by the next person.

WHY A NEW FILE AND NOT ANOTHER LINE IN router.py.  ``router.option_history``
already counts something like this, and measuring it is what made a separate
module necessary rather than optional:

  * it counts only rows that carry an ``identity_face`` verdict at the current
    0.45 bar, and a call the provider BLOCKED has no verdict at all - so the 14
    blocked calls in this installation (0.65 USD, 19% of every paid call) are
    invisible to the only warning the estimate screen renders;
  * it counts identity passes and nothing else, which currently makes
    ('clothing','deportiva_elegante') read "7 de 7" and print as "si ha
    funcionado casi siempre" while 6 of those 7 images were REJECTED for
    anatomy.  A warning that certifies the images the client threw away is
    worse than no warning.

So the memory here is per FAILURE KIND, it counts provider blocks, and it
counts a denominator per kind rather than one denominator for everything.

THE DENOMINATOR IS THE HARD PART, and getting it wrong is how a table starts
lying.  Two rulers in this product changed mid-history:

  * ``identity_face`` was a geometric descriptor at 0.72 that scored 0.99 on a
    different woman, and is now an SFace embedding at 0.45.  Both wrote rows.
    Counting the old ones would certify strangers, so a stored identity check
    is only evidence when its threshold is still the one in force.
  * ``body_proportions`` is SKIPPED when the two figures cannot be compared,
    and a skip is stored as threshold 1.0.  It was actually measured on only 34
    of the 60 paid calls that drew anything.  Counting a skip as a pass turns
    "postura caminando nunca movio el cuerpo" into a sentence about 13 calls
    where nobody looked.

Hence ``kinds_of``: every record answers TWO questions, "which checks failed"
and "which checks were really measured", and every rate below is failures over
things that were measured.

WHAT IS DELIBERATELY NOT HERE.  No rule keyed on a person, a photograph or a
name: the counters are keyed on option group, option value, request feature and
request fingerprint, all of which are catalogue vocabulary.  The person enters
only as the SCOPE the counters are stored under.
"""
from __future__ import annotations

import hashlib
import json
import time
from typing import Any

from .. import db
from ..catalog import options as options_mod

# ------------------------------------------------------------------ vocabulary

# One name per way an image can cost money and not arrive.  ``bloqueo`` is the
# provider refusing to hand over a picture it already drew and billed; the rest
# are our own verdict rejecting the picture we did get.
KINDS = ("bloqueo", "identidad", "cuerpo", "anatomia", "piel", "calidad")

CHECK_KIND: dict[str, str] = {
    "identity_face": "identidad",
    "body_proportions": "cuerpo",
    "anatomy": "anatomia",
    "skin_tone": "piel",
    "quality": "calidad",
}

KIND_ES: dict[str, str] = {
    "bloqueo": "el proveedor no entrego la imagen y la cobro igual",
    "identidad": "la cara que salio no era la tuya",
    "cuerpo": "el cuerpo salio con otra forma",
    "anatomia": "algo salio mal dibujado (una mano, o la piel demasiado lisa)",
    "piel": "la piel salio de otro tono",
    "calidad": "la imagen salio borrosa",
}

# The short form, for the headline of a warning.
KIND_CORTO: dict[str, str] = {
    "bloqueo": "la imagen no llega y se cobra igual",
    "identidad": "sale otra cara",
    "cuerpo": "cambia tu figura",
    "anatomia": "salen manos o piel mal dibujadas",
    "piel": "cambia tu tono de piel",
    "calidad": "sale borrosa",
}

# THE GROUPS THAT MAKE THE ENGINE REDRAW THE PERSON.  Same list as
# router.GENERATIVE_CHANGES, repeated here rather than imported because router
# imports this module and the cycle would be a real one.  It matters for the
# group-level rule below: a scene or a light does not move her shoulders, so a
# background that happens to correlate with a body failure in a small sample is
# a coincidence, not a lever the client can pull.  Restricting the group rule to
# these five raised the recall of the whole assessment from 74% to 84% of the
# recorded failures WITHOUT costing a single extra false alarm, because the
# spurious "grupo:scene" findings it removed were all shadowing a real one.
REDRAW_GROUPS = ("clothing", "pose", "expression", "hair", "transparency")

# What the free local engine can really do: it transforms the photograph it is
# handed - background, garment colour, light, grade, crop - and cannot invent a
# garment or a new body position.  A request made only of these is a request
# nobody has to pay for, and that is worth shouting about: in this installation
# the free engine has 48 recorded attempts, 0.00 USD, and zero provider blocks.
FREE_GROUPS = ("scene", "lighting", "clothing_color", "framing", "grade",
               "treatment")

# The feature flags of a request that are switchable in Ajustes and that the
# record shows can decide the outcome on their own.
FEATURES = ("texto_cobertura", "rostro_repintado", "fotos_referencia")


# ------------------------------------------------------------------ thresholds

# Three paid images is the least that can tell a bad option from bad luck; it
# is the same bar router.RISK_MIN_N already uses and there is no reason for the
# two screens to disagree about what counts as evidence.
MIN_N = 3
# Above this share of the measurements the option is worth a sentence.
HIGH_RATE = 0.40
# ...but only if it is also this much worse than the alternative.  A rate on its
# own convicts whatever was popular: 'clothing' is on 74 of 74 paid calls, so
# its failure rate IS the installation's failure rate.  What makes a finding
# actionable is the contrast with the requests that did not carry it.
MIN_GAP = 0.30
# Failed every single time, with enough tries that "every time" means
# something.  This is the only level that stops and asks.  Measured on the
# record: at 4 it puts a confirmation in front of 15 calls that failed and 1
# that succeeded; at 3 it catches 23 failures and stops 2 successes.  Four buys
# most of the protection for half of the friction.
CONFIRM_N = 4

# How much history it takes to conclude that an ADJUSTMENT the robot itself
# recommended does not work.  Two, not the three MIN_N demands of an option,
# because they are not equal cases: an option value is one of many a person
# might pick for reasons of her own, while an adjustment is a recommendation
# this code made on purpose.  Being wrong about it twice is enough to stop
# making it, and the fallback - not changing anything - is free.  It lives here
# rather than in generation/adjust.py because BOTH sides need it: the button on
# the estimate screen must not keep offering a change that the run has already
# tried twice and watched fail.
ADJ_MIN_N = 2

# How many request fingerprints a scope remembers.  One per variant of a run,
# so a hundred runs of six variants fit comfortably; the oldest are dropped.
MAX_FINGERPRINTS = 600
# THE PHOTOGRAPH IS THE REQUEST'S FOUNDATION, and until 2026-09-10 the record
# had no key for it.  Measured that day over the 46 paid kontext/multi attempts
# judged by the SFace ruler: one source photograph produced a face that was
# hers in 17 of 17 images, another in 3 of 10 - and the estimate for the
# eleventh request on that second photograph read "aviso: ''", because every
# counter here was about OPTIONS, and the fingerprint keyed the whole request
# on the photograph's file PATH, which the move to Linux had just renamed.
# The client paid 0.04 USD to be told, again, what the record already knew.
#
# So the photograph gets its own cell, ``src:<sha256[:16]>``, keyed on the
# file's CONTENT so that a copy, a rename or a second machine keep the record,
# and it carries the filename so the sentence can name a better one.  The store
# is versioned: a pool written before this key existed is rebuilt from the
# attempts table, which also retires the path-keyed fingerprints the rename
# orphaned.
STORE_VERSION = 2
SRC_KEY_LEN = 16


# ------------------------------------------------------------------- helpers

def _text(value: Any) -> str:
    return str(value).strip() if value is not None else ""


def _f(value: Any, default: float = 0.0) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return default
    if out != out or out in (float("inf"), float("-inf")):
        return default
    return out


def _cell() -> dict:
    return {"n": 0, "ok": 0, "f": {}, "m": {}, "t": 0.0}


def _bump(store: dict, key: str, failed, measured, accepted: bool) -> None:
    cell = store.setdefault(key, _cell())
    cell["n"] = int(cell.get("n", 0)) + 1
    cell["ok"] = int(cell.get("ok", 0)) + (1 if accepted else 0)
    cell["t"] = time.time()
    for kind in measured:
        cell["m"][kind] = int(cell["m"].get(kind, 0)) + 1
    for kind in failed:
        cell["f"][kind] = int(cell["f"].get(kind, 0)) + 1


def _prefer(shared: dict, mine: dict) -> dict:
    """Her own record where she has one, the installation's underneath.

    Key by key, NOT added together: every attempt of hers is written to both
    pools, so summing them would count her own images twice and make three
    images read as six.  Once she has ``MIN_N`` measurements of a key those are
    what decide - they were measured on her face - and until then she inherits
    what the robot already paid to learn from everybody.
    """
    out = {k: {"n": int(_f(v.get("n"))), "ok": int(_f(v.get("ok"))),
               "f": dict(v.get("f") or {}), "m": dict(v.get("m") or {}),
               "t": _f(v.get("t")), "mio": False,
               "nombre": _text(v.get("nombre"))}
           for k, v in (shared or {}).items() if isinstance(v, dict)}
    for key, cell in (mine or {}).items():
        if not isinstance(cell, dict):
            continue
        if int(_f(cell.get("n"))) < MIN_N and key in out:
            continue
        out[key] = {"n": int(_f(cell.get("n"))), "ok": int(_f(cell.get("ok"))),
                    "f": dict(cell.get("f") or {}), "m": dict(cell.get("m") or {}),
                    "t": _f(cell.get("t")), "mio": True,
                    "nombre": _text(cell.get("nombre")) or _text((out.get(key) or {}).get("nombre"))}
    return out


def _rate(cell: dict | None, kind: str) -> tuple[int, int]:
    """(failures, measurements) of one kind on one key."""
    if not isinstance(cell, dict):
        return 0, 0
    return (int(_f((cell.get("f") or {}).get(kind))),
            int(_f((cell.get("m") or {}).get(kind))))


def adjustment_ok(pool: dict, key: str) -> bool:
    """Is this change still worth recommending, or has it been tried and failed?

    An ``adj:`` cell is written by the orchestrator every time a change it chose
    was paid for, so this is the robot scoring its OWN advice rather than the
    option the advice moved to - a different question, and the only one that
    can make the recommendation itself improve.
    """
    cell = (pool or {}).get(_text(key))
    if not isinstance(cell, dict):
        return True
    return not (int(_f(cell.get("n"))) >= ADJ_MIN_N
                and not int(_f(cell.get("ok"))))


# ------------------------------------------------------- reading an outcome

# The identity ruler in force.  Read from the profile module rather than
# written here, so that recalibrating the check automatically invalidates the
# rows measured by the old one instead of silently keeping them.
def _identity_bar() -> float:
    try:
        from ..identity.profile import DEFAULT_THRESHOLDS
        return _f(DEFAULT_THRESHOLDS.get("face_embed_min"), 0.45)
    except Exception:                                     # noqa: BLE001
        return 0.45


def kinds_of(status: str, reason: str, verdict: Any) -> tuple[set, set]:
    """(what failed, what was actually measured) for one finished attempt.

    See the module docstring: a rate is only honest over the checks that a
    comparable ruler really read.

    ALL 14 UNDELIVERED CALLS ARE ONE EVENT, and it is worth saying why, because
    the database makes them look like two.  13 rows say "fal.ai ha revisado la
    imagen que acababa de dibujar" and 1 says "bloqueada por el proveedor";
    that is not two failure modes, it is providers/fal.py rewording the SAME
    exception on 2026-09-06 (commit 48a450f, which removed the claim that fal
    had judged the picture because the timings did not support it).  Every one
    of the 13 old-wording rows is from 09-05 and every one of them carried the
    coverage clause; the single new-wording row is from 09-06 and did not.  So
    the reason string is a clock, not a diagnosis: splitting on it would have
    taught the robot that a message it no longer writes is the dangerous one.
    What really separates the deterministic block from the unlucky one is the
    REQUEST - the coverage clause is on 13 of 13 blocked prompts and 0 of the
    61 others - and that is counted as ``feat:texto_cobertura`` below, where it
    can be measured instead of assumed.
    """
    state = _text(status).lower()
    if state == "error":
        return {"bloqueo"}, {"bloqueo"}

    # Every call that reached the engine is evidence about blocking, because
    # not being blocked is the outcome we want to be able to compare against.
    measured: set = {"bloqueo"}
    failed: set = set()
    checks = (verdict or {}).get("checks") if isinstance(verdict, dict) else None
    bar = _identity_bar()
    for check in (checks or []):
        if not isinstance(check, dict):
            continue
        kind = CHECK_KIND.get(_text(check.get("name")))
        if not kind:
            continue
        threshold = check.get("threshold")
        if kind == "identidad":
            # A verdict written by the old descriptor is not a measurement of
            # the thing this check now measures.
            if threshold is None or abs(_f(threshold, -1.0) - bar) > 1e-6:
                continue
        if kind == "cuerpo":
            # 1.0 is how "no se pudieron medir proporciones comparables" is
            # stored.  A skip is not a pass.
            if threshold is None or _f(threshold, 1.0) >= 1.0 - 1e-9:
                continue
        measured.add(kind)
        if not check.get("passed", True):
            failed.add(kind)
    return failed, measured


# ------------------------------------------------------------ the request key

def _choices_of(plan: Any) -> dict[str, str]:
    """Every option value this plan asks for, one value per group.

    A plan varies some groups across its previews, so a group can carry several
    values; the assessment is about the request as a whole, so all of them are
    collected and the risk of the request is the risk of its worst value.
    """
    out: dict[str, set] = {}
    plan_d = plan if isinstance(plan, dict) else {}
    sources = list(plan_d.get("variants") or []) + [{"choices": plan_d.get("locked") or {}}]
    for variant in sources:
        for group, value in ((variant or {}).get("choices") or {}).items():
            values = value if isinstance(value, (list, tuple, set)) else [value]
            for item in values:
                if _text(item) and isinstance(item, str):
                    out.setdefault(_text(group), set()).add(_text(item))
    return {g: sorted(v) for g, v in out.items() if v}


def features_of(plan: Any) -> set:
    """Switchable properties of the request that the record can convict."""
    envio = (plan or {}).get("envio") if isinstance(plan, dict) else None
    envio = envio if isinstance(envio, dict) else {}
    out: set = set()
    if envio.get("outfit_coverage_text"):
        out.add("texto_cobertura")
    if envio.get("masked_inpaint"):
        out.add("rostro_repintado")
    if int(_f(envio.get("reference_photos"))) > 0:
        out.add("fotos_referencia")
    return out


def fingerprint(source: Any, choices: dict, quality: str, style: Any,
                endpoint: str = "") -> str:
    """The identity of a REQUEST, so the same one is never bought twice blind.

    Source photograph, the options asked for, the tier, the style and the
    endpoint that will run it.  Not the seed and not the strength: the record
    is unambiguous that lowering the strength does not rescue a request the
    engine has already answered - 0 of 5 identity retries and 0 of 5 blocked
    retries - so two attempts that differ only in strength are the same
    request, and remembering them as two would be remembering nothing.
    """
    payload = json.dumps(
        {"src": _text(source),
         "ch": sorted((g, sorted(v) if isinstance(v, (list, tuple, set)) else [_text(v)])
                      for g, v in (choices or {}).items()),
         "q": _text(quality), "st": _text(style), "ep": _text(endpoint)},
        sort_keys=True, default=str)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


def source_key(sha256: Any) -> str:
    """The record's name for a photograph: the start of its content hash."""
    text = _text(sha256).lower()
    return text[:SRC_KEY_LEN] if len(text) >= SRC_KEY_LEN else ""


def _plan_source(plan_d: dict) -> tuple[str, str]:
    """(source key, filename) for the plan being priced.

    The plan carries them from 2026-09-10 on; an older plan only has the
    original's id, which is looked up.  Read-only, never raises: a missing
    photograph simply has no record.
    """
    key = source_key(plan_d.get("source_sha"))
    name = _text(plan_d.get("source_name"))
    if key:
        return key, name
    oid = _text(plan_d.get("original_id"))
    if not oid:
        return "", ""
    try:
        row = db.q1("SELECT sha256, filename FROM originals WHERE id=?", (oid,))
    except Exception:                                     # noqa: BLE001
        return "", ""
    rec = db.row_to_dict(row) or {}
    return source_key(rec.get("sha256")), _text(rec.get("filename"))


# ------------------------------------------------------------------- the seed

# WHAT A BRAND NEW ACCOUNT INHERITS.  Every paid call this installation has
# ever made, counted by option value, by option group, by "the group was not
# asked for" and by request feature, with the denominators described at the top
# of the file.  74 paid calls, 3.05 USD.  It contains no person, no photograph
# and no name - only catalogue vocabulary - and it is a SEED: the shared pool in
# the ``learning`` table is initialised from the installation's own attempts
# table when it is first read, and from this table when there are none, and
# every finished attempt afterwards is added to it.
#
# The two findings it exists to carry, both of which cost real money to learn:
#   grp:pose vs sin:pose - asking for a different body position moved the
#     figure on 8 of the 18 requests where the proportions could be measured,
#     and on 0 of the 17 where no pose was asked for.
#   feat:texto_cobertura - the clause demanding full coverage was returned as a
#     black charged file 13 times out of 13, and 0 of the 61 prompts without it
#     were ever blocked that way.
PRIOR: dict[str, dict] = {
    "feat:texto_cobertura": {"n": 13, "ok": 0, "f": {"bloqueo": 13}, "m": {"bloqueo": 13}},
    "grp:clothing": {"n": 74, "ok": 31, "f": {"bloqueo": 14, "identidad": 10, "cuerpo": 8, "anatomia": 20, "piel": 3}, "m": {"bloqueo": 74, "identidad": 27, "cuerpo": 35, "anatomia": 60, "piel": 60, "calidad": 60}},
    "grp:clothing_color": {"n": 36, "ok": 21, "f": {"identidad": 5, "cuerpo": 5, "anatomia": 9, "piel": 1}, "m": {"bloqueo": 36, "identidad": 7, "cuerpo": 18, "anatomia": 36, "piel": 36, "calidad": 36}},
    "grp:lighting": {"n": 21, "ok": 15, "f": {"cuerpo": 3, "anatomia": 4}, "m": {"bloqueo": 21, "cuerpo": 9, "anatomia": 21, "piel": 21, "calidad": 21}},
    "grp:pose": {"n": 41, "ok": 21, "f": {"bloqueo": 4, "identidad": 4, "cuerpo": 8, "anatomia": 9}, "m": {"bloqueo": 41, "identidad": 7, "cuerpo": 18, "anatomia": 37, "piel": 37, "calidad": 37}},
    "grp:scene": {"n": 52, "ok": 21, "f": {"bloqueo": 6, "identidad": 10, "cuerpo": 8, "anatomia": 16, "piel": 3}, "m": {"bloqueo": 52, "identidad": 16, "cuerpo": 27, "anatomia": 46, "piel": 46, "calidad": 46}},
    "opt:clothing:blazer_oversize": {"n": 2, "ok": 2, "f": {}, "m": {"bloqueo": 2, "cuerpo": 2, "anatomia": 2, "piel": 2, "calidad": 2}},
    "opt:clothing:blusa_seda": {"n": 1, "ok": 1, "f": {}, "m": {"bloqueo": 1, "identidad": 1, "anatomia": 1, "piel": 1, "calidad": 1}},
    "opt:clothing:camisa_blanca": {"n": 7, "ok": 2, "f": {"bloqueo": 5}, "m": {"bloqueo": 7, "identidad": 1, "cuerpo": 2, "anatomia": 2, "piel": 2, "calidad": 2}},
    "opt:clothing:chaqueta_cuero": {"n": 1, "ok": 1, "f": {}, "m": {"bloqueo": 1, "identidad": 1, "cuerpo": 1, "anatomia": 1, "piel": 1, "calidad": 1}},
    "opt:clothing:deportiva_elegante": {"n": 8, "ok": 1, "f": {"bloqueo": 1, "anatomia": 6}, "m": {"bloqueo": 8, "identidad": 7, "cuerpo": 4, "anatomia": 7, "piel": 7, "calidad": 7}},
    "opt:clothing:falda_larga": {"n": 2, "ok": 0, "f": {"bloqueo": 2}, "m": {"bloqueo": 2}},
    "opt:clothing:gabardina": {"n": 8, "ok": 1, "f": {"cuerpo": 2, "anatomia": 6}, "m": {"bloqueo": 8, "cuerpo": 5, "anatomia": 8, "piel": 8, "calidad": 8}},
    "opt:clothing:jersey_cachemira": {"n": 3, "ok": 2, "f": {"bloqueo": 1}, "m": {"bloqueo": 3, "identidad": 1, "anatomia": 2, "piel": 2, "calidad": 2}},
    "opt:clothing:pantalon_ancho": {"n": 1, "ok": 1, "f": {}, "m": {"bloqueo": 1, "identidad": 1, "cuerpo": 1, "anatomia": 1, "piel": 1, "calidad": 1}},
    "opt:clothing:traje_sastre": {"n": 1, "ok": 1, "f": {}, "m": {"bloqueo": 1, "cuerpo": 1, "anatomia": 1, "piel": 1, "calidad": 1}},
    "opt:clothing:vaqueros_camiseta": {"n": 23, "ok": 16, "f": {"bloqueo": 5, "anatomia": 2}, "m": {"bloqueo": 23, "identidad": 3, "cuerpo": 5, "anatomia": 18, "piel": 18, "calidad": 18}},
    "opt:clothing:vestido_noche": {"n": 6, "ok": 0, "f": {"identidad": 4, "cuerpo": 4, "anatomia": 2}, "m": {"bloqueo": 6, "identidad": 6, "cuerpo": 4, "anatomia": 6, "piel": 6, "calidad": 6}},
    "opt:clothing:vestido_rojo": {"n": 6, "ok": 0, "f": {"identidad": 6, "anatomia": 4, "piel": 3}, "m": {"bloqueo": 6, "identidad": 6, "cuerpo": 6, "anatomia": 6, "piel": 6, "calidad": 6}},
    "opt:clothing:vestido_verano": {"n": 5, "ok": 3, "f": {"cuerpo": 2}, "m": {"bloqueo": 5, "cuerpo": 4, "anatomia": 5, "piel": 5, "calidad": 5}},
    "opt:clothing_color:blanco": {"n": 24, "ok": 18, "f": {"identidad": 3, "cuerpo": 3, "anatomia": 1}, "m": {"bloqueo": 24, "identidad": 4, "cuerpo": 9, "anatomia": 24, "piel": 24, "calidad": 24}},
    "opt:clothing_color:gris": {"n": 5, "ok": 1, "f": {"cuerpo": 1, "anatomia": 4}, "m": {"bloqueo": 5, "cuerpo": 3, "anatomia": 5, "piel": 5, "calidad": 5}},
    "opt:clothing_color:rojo": {"n": 6, "ok": 1, "f": {"identidad": 2, "cuerpo": 1, "anatomia": 4, "piel": 1}, "m": {"bloqueo": 6, "identidad": 3, "cuerpo": 5, "anatomia": 6, "piel": 6, "calidad": 6}},
    "opt:clothing_color:verde_oliva": {"n": 1, "ok": 1, "f": {}, "m": {"bloqueo": 1, "cuerpo": 1, "anatomia": 1, "piel": 1, "calidad": 1}},
    "opt:lighting:ventana_der": {"n": 8, "ok": 7, "f": {"anatomia": 1}, "m": {"bloqueo": 8, "cuerpo": 2, "anatomia": 8, "piel": 8, "calidad": 8}},
    "opt:lighting:ventana_izq": {"n": 13, "ok": 8, "f": {"cuerpo": 3, "anatomia": 3}, "m": {"bloqueo": 13, "cuerpo": 7, "anatomia": 13, "piel": 13, "calidad": 13}},
    "opt:pose:brazos_cruzados": {"n": 5, "ok": 3, "f": {"cuerpo": 2}, "m": {"bloqueo": 5, "cuerpo": 4, "anatomia": 5, "piel": 5, "calidad": 5}},
    "opt:pose:caminando": {"n": 18, "ok": 15, "f": {"bloqueo": 2, "anatomia": 1}, "m": {"bloqueo": 18, "identidad": 1, "cuerpo": 3, "anatomia": 16, "piel": 16, "calidad": 16}},
    "opt:pose:de_pie_frontal": {"n": 4, "ok": 1, "f": {"identidad": 3, "cuerpo": 1}, "m": {"bloqueo": 4, "identidad": 3, "cuerpo": 2, "anatomia": 4, "piel": 4, "calidad": 4}},
    "opt:pose:mano_cadera": {"n": 3, "ok": 0, "f": {"identidad": 1, "cuerpo": 3, "anatomia": 2}, "m": {"bloqueo": 3, "identidad": 3, "cuerpo": 3, "anatomia": 3, "piel": 3, "calidad": 3}},
    "opt:pose:sentada": {"n": 10, "ok": 1, "f": {"bloqueo": 2, "cuerpo": 2, "anatomia": 6}, "m": {"bloqueo": 10, "cuerpo": 5, "anatomia": 8, "piel": 8, "calidad": 8}},
    "opt:pose:tres_cuartos": {"n": 1, "ok": 1, "f": {}, "m": {"bloqueo": 1, "cuerpo": 1, "anatomia": 1, "piel": 1, "calidad": 1}},
    "opt:scene:ciclorama_blanco": {"n": 7, "ok": 1, "f": {"bloqueo": 2, "cuerpo": 1, "anatomia": 4}, "m": {"bloqueo": 7, "cuerpo": 3, "anatomia": 5, "piel": 5, "calidad": 5}},
    "opt:scene:ciudad_noche": {"n": 15, "ok": 12, "f": {"bloqueo": 2, "anatomia": 1}, "m": {"bloqueo": 15, "identidad": 1, "cuerpo": 1, "anatomia": 13, "piel": 13, "calidad": 13}},
    "opt:scene:estudio_gris": {"n": 6, "ok": 1, "f": {"bloqueo": 2, "cuerpo": 1, "anatomia": 2}, "m": {"bloqueo": 6, "cuerpo": 3, "anatomia": 4, "piel": 4, "calidad": 4}},
    "opt:scene:estudio_terracota": {"n": 1, "ok": 1, "f": {}, "m": {"bloqueo": 1, "cuerpo": 1, "anatomia": 1, "piel": 1, "calidad": 1}},
    "opt:scene:playa_atardecer": {"n": 23, "ok": 6, "f": {"identidad": 10, "cuerpo": 6, "anatomia": 9, "piel": 3}, "m": {"bloqueo": 23, "identidad": 15, "cuerpo": 19, "anatomia": 23, "piel": 23, "calidad": 23}},
    "sin:clothing_color": {"n": 38, "ok": 10, "f": {"bloqueo": 14, "identidad": 5, "cuerpo": 3, "anatomia": 11, "piel": 2}, "m": {"bloqueo": 38, "identidad": 20, "cuerpo": 17, "anatomia": 24, "piel": 24, "calidad": 24}},
    "sin:lighting": {"n": 53, "ok": 16, "f": {"bloqueo": 14, "identidad": 10, "cuerpo": 5, "anatomia": 16, "piel": 3}, "m": {"bloqueo": 53, "identidad": 27, "cuerpo": 26, "anatomia": 39, "piel": 39, "calidad": 39}},
    "sin:pose": {"n": 33, "ok": 10, "f": {"bloqueo": 10, "identidad": 6, "anatomia": 11, "piel": 3}, "m": {"bloqueo": 33, "identidad": 20, "cuerpo": 17, "anatomia": 23, "piel": 23, "calidad": 23}},
    "sin:scene": {"n": 22, "ok": 10, "f": {"bloqueo": 8, "anatomia": 4}, "m": {"bloqueo": 22, "identidad": 11, "cuerpo": 8, "anatomia": 14, "piel": 14, "calidad": 14}},
    "sinfeat:fotos_referencia": {"n": 74, "ok": 31, "f": {"bloqueo": 14, "identidad": 10, "cuerpo": 8, "anatomia": 20, "piel": 3}, "m": {"bloqueo": 74, "identidad": 27, "cuerpo": 35, "anatomia": 60, "piel": 60, "calidad": 60}},
    "sinfeat:rostro_repintado": {"n": 74, "ok": 31, "f": {"bloqueo": 14, "identidad": 10, "cuerpo": 8, "anatomia": 20, "piel": 3}, "m": {"bloqueo": 74, "identidad": 27, "cuerpo": 35, "anatomia": 60, "piel": 60, "calidad": 60}},
    "sinfeat:texto_cobertura": {"n": 61, "ok": 31, "f": {"bloqueo": 1, "identidad": 10, "cuerpo": 8, "anatomia": 20, "piel": 3}, "m": {"bloqueo": 61, "identidad": 27, "cuerpo": 35, "anatomia": 60, "piel": 60, "calidad": 60}},
}


# --------------------------------------------------------------- persistence

# The shared pool: what every account starts from and what every account adds
# to.  ``learning.user_id`` has no foreign key, so a reserved id that no
# registration can ever produce (``db.new_id`` always prefixes) is a valid row
# and keeps the shared record out of anybody's personal one.
SHARED_USER = "*"
SCOPE = "riesgo"


def _scope_for(profile_id: Any) -> str:
    """Per person, so the memory improves for the person it is about.

    Keyed on the PROFILE and not on the account: a household can share a login,
    and telling one person that a garment always fails because it failed on
    somebody else's face is exactly the mistake this module is for.
    """
    pid = _text(profile_id)
    return "%s:%s" % (SCOPE, pid) if pid else SCOPE


def _load(user_id: str, scope: str) -> dict:
    try:
        row = db.q1("SELECT weights_json FROM learning WHERE user_id=? AND scope=?",
                    (_text(user_id), scope))
    except Exception:                                     # noqa: BLE001
        return {}
    rec = db.row_to_dict(row) or {}
    data = rec.get("weights")
    return data if isinstance(data, dict) else {}


def _save(user_id: str, scope: str, data: dict) -> bool:
    try:
        db.execute(
            "INSERT INTO learning(id,user_id,scope,weights_json,stats_json,"
            "updated_at) VALUES(?,?,?,?,?,?) ON CONFLICT(user_id,scope) DO "
            "UPDATE SET weights_json=excluded.weights_json, "
            "updated_at=excluded.updated_at",
            (db.new_id("lrn"), _text(user_id), scope, db.dumps(data),
             db.dumps({}), db.now()))
        return True
    except Exception:                                     # noqa: BLE001
        return False


def _trim(store: dict) -> dict:
    """Keep the fingerprint memory bounded, oldest first."""
    prints = {k: v for k, v in store.items() if k.startswith("fp:")}
    if len(prints) <= MAX_FINGERPRINTS:
        return store
    order = sorted(prints.items(), key=lambda kv: _f(kv[1].get("t")))
    for key, _cellv in order[:len(prints) - MAX_FINGERPRINTS]:
        store.pop(key, None)
    return store


def _backfill_shared() -> dict:
    """Build the shared pool from this installation's own paid attempts.

    Run once, the first time the shared row is read empty.  It is what makes an
    installation that has been paying for months stop repeating its mistakes
    immediately instead of only learning from the next attempt onwards - and it
    is why the seed above is a fallback and not the source of truth: the
    database always knows more than a table frozen into a file.
    """
    store: dict = {}
    try:
        rows = db.q(
            "SELECT a.variant_index, a.status, a.reject_reason, a.verdict_json,"
            " a.prompt, a.model, r.plan_json, r.options_json,"
            " o.sha256 AS src_sha, o.filename AS src_name "
            "FROM attempts a JOIN runs r ON r.id=a.run_id "
            "LEFT JOIN originals o ON o.id=r.original_id WHERE a.cost_usd>0")
    except Exception:                                     # noqa: BLE001
        return {}
    groups_seen: set = set()
    records: list[tuple] = []
    for row in (rows or []):
        plan = db.loads(row["plan_json"], None) or {}
        index = row["variant_index"]
        variant = next((v for v in (plan.get("variants") or [])
                        if isinstance(v, dict)
                        and int(v.get("index", -1)) == int(index if index is not None else -1)),
                       None)
        choices = {}
        for group, value in ((variant or {}).get("choices") or {}).items():
            if isinstance(value, str) and value.strip():
                choices[_text(group)] = _text(value)
        for group, value in (plan.get("locked") or {}).items():
            if isinstance(value, str) and value.strip():
                choices.setdefault(_text(group), _text(value))
        if not choices:
            continue
        failed, measured = kinds_of(row["status"], row["reject_reason"],
                                    db.loads(row["verdict_json"], None) or {})
        # The one request feature the record can convict, read off the prompt
        # that was really sent rather than off a setting that may have changed
        # since.  The phrase is the tail of prompt.OUTFIT_REPLACE.
        feats = set()
        if "only clothing visible anywhere in the frame" in _text(row["prompt"]).lower():
            feats.add("texto_cobertura")
        groups_seen |= set(choices)
        # The same two things observe() writes for a live attempt, so a
        # rebuilt pool is the pool the live one would have grown into: the
        # request fingerprint keyed on the photograph's CONTENT, and the
        # photograph's own cell.
        options = db.loads(row["options_json"], None) or {}
        src = source_key(row["src_sha"])
        fp = fingerprint(src or "", choices, options.get("quality"),
                         options.get("style"), row["model"]) if src else ""
        records.append((choices, failed, measured,
                        _text(row["status"]).lower() == "accepted", feats,
                        fp, src, _text(row["src_name"])))
    for choices, failed, measured, accepted, feats, fp, src, name in records:
        _observe_into(store, choices, feats, groups_seen, failed, measured,
                      accepted, fp, src, name)
    return store


def _observe_into(store: dict, choices: dict, feats: set, groups_seen: set,
                  failed, measured, accepted: bool, fp: str,
                  src: str = "", src_name: str = "") -> None:
    for group, value in (choices or {}).items():
        _bump(store, "opt:%s:%s" % (group, value), failed, measured, accepted)
        _bump(store, "grp:%s" % group, failed, measured, accepted)
    for group in sorted(set(groups_seen) - set(choices or {})):
        _bump(store, "sin:%s" % group, failed, measured, accepted)
    for name in FEATURES:
        key = ("feat:%s" if name in (feats or set()) else "sinfeat:%s") % name
        _bump(store, key, failed, measured, accepted)
    if fp:
        _bump(store, "fp:%s" % fp, failed, measured, accepted)
    if src:
        _bump(store, "src:%s" % src, failed, measured, accepted)
        if src_name:
            store["src:%s" % src]["nombre"] = src_name
    store["_grupos"] = sorted(set(store.get("_grupos") or []) | set(choices or {}))


def shared_pool() -> dict:
    """The installation's record: read, backfilled once, seeded if brand new."""
    store = _load(SHARED_USER, SCOPE)
    if store and int(_f(store.get("_v"))) >= STORE_VERSION:
        return store
    # Either brand new, or written before the photograph had a cell: rebuilt
    # from the attempts table, which is the only place the truth was all along.
    store = _backfill_shared()
    if not store:
        store = {k: dict(v, f=dict(v["f"]), m=dict(v["m"])) for k, v in PRIOR.items()}
        store["_grupos"] = sorted({k.split(":")[1] for k in PRIOR
                                   if k.startswith("grp:")})
    store["_v"] = STORE_VERSION
    _save(SHARED_USER, SCOPE, store)
    return store


def person_pool(user_id: str, profile_id: Any) -> dict:
    store = _load(_text(user_id), _scope_for(profile_id))
    if store and int(_f(store.get("_v"))) < STORE_VERSION:
        # Her fingerprints were keyed on a file path that the move to Linux
        # renamed; they can never match again and would only crowd the memory.
        # Her option counters are still hers and are kept.
        store = {k: v for k, v in store.items() if not k.startswith("fp:")}
        store["_v"] = STORE_VERSION
        _save(_text(user_id), _scope_for(profile_id), store)
    return store


def observe(user_id: str, profile_id: Any, choices: dict, feats: set,
            fp: str, status: str, reason: str, verdict: Any,
            src: str = "", src_name: str = "") -> None:
    """Fold one finished PAID attempt into the person's pool and the shared one.

    Called from the orchestrator the moment an attempt is recorded, whichever
    way it ended.  Failures cost money and are the whole point, but successes
    are written too: without them there is no denominator and every rate below
    would read 100%.
    """
    failed, measured = kinds_of(status, reason, verdict)
    accepted = _text(status).lower() == "accepted"
    clean = {_text(g): _text(v) for g, v in (choices or {}).items()
             if _text(g) and _text(v)}
    if not clean:
        return
    for uid, scope in ((_text(user_id), _scope_for(profile_id)),
                       (SHARED_USER, SCOPE)):
        if not uid:
            continue
        store = _load(uid, scope)
        if not store and uid == SHARED_USER:
            store = shared_pool()
        seen = set(store.get("_grupos") or []) | set(clean)
        _observe_into(store, clean, feats, seen, failed, measured, accepted, fp,
                      src, src_name)
        _save(uid, scope, _trim(store))


def source_records(user_id: str = "", profile_id: Any = "") -> dict[str, dict]:
    """What the paid record says about each photograph, keyed by source_key.

    For the photo tiles: ``{"pagadas", "buenas", "cara_fallos", "cara_medidas"}``
    per key.  Read-only and never raises - a tile without a record is a tile
    without a badge, not an error.
    """
    try:
        pool = pool_for(user_id, profile_id)
    except Exception:                                     # noqa: BLE001
        return {}
    out: dict[str, dict] = {}
    for key, cell in (pool or {}).items():
        if not key.startswith("src:") or not isinstance(cell, dict):
            continue
        fails, meas = _rate(cell, "identidad")
        out[key[4:]] = {"pagadas": int(_f(cell.get("n"))),
                        "buenas": int(_f(cell.get("ok"))),
                        "cara_fallos": fails, "cara_medidas": meas,
                        "nombre": _text(cell.get("nombre"))}
    return out


def pool_for(user_id: str, profile_id: Any) -> dict:
    """Everything known about options, for this person, ready to read.

    The one reader both halves of the feature share.  ``assess`` builds this
    before the money moves and ``adjust.decide`` builds it after an image has
    been thrown away, and they must be looking at the same numbers or the robot
    will warn about one thing and then fix another.
    """
    try:
        return _prefer(shared_pool(), person_pool(user_id, profile_id))
    except Exception:                                     # noqa: BLE001
        return {}


def note_adjustment(user_id: str, profile_id: Any, key: str, status: str,
                    reason: str, verdict: Any) -> None:
    """Remember how an ADJUSTMENT this robot chose actually turned out.

    Kept beside the option counters, in the same two pools, under an ``adj:``
    key - so a change that keeps working is made faster and a change that keeps
    failing is stopped making, which is the only way the loop can get better
    with use instead of merely more insistent.  Without this row the robot
    would be scoring its own recommendations by the record of the option it
    moved TO, which is a different question: 'vaqueros_camiseta' being accepted
    16 times out of 23 says nothing about whether swapping a dress for it
    rescues a run that has already lost her face once.
    """
    if not _text(key):
        return
    failed, measured = kinds_of(status, reason, verdict)
    accepted = _text(status).lower() == "accepted"
    for uid, scope in ((_text(user_id), _scope_for(profile_id)),
                       (SHARED_USER, SCOPE)):
        if not uid:
            continue
        store = _load(uid, scope)
        if not store and uid == SHARED_USER:
            store = shared_pool()
        _bump(store, _text(key), failed, measured, accepted)
        _save(uid, scope, _trim(store))


def failed_before(pool: dict, fp: str) -> int:
    """How many times this EXACT request was bought and never once worked.

    0 means "never bought, or bought and accepted at least once".  It is the
    literal reading of the client's "que el error no vuelva a ocurrir": a
    request whose fingerprint is already in the pool with no acceptance behind
    it is one the robot has proof about, and proof is cheaper than 0.04 USD.
    """
    cell = (pool or {}).get("fp:%s" % _text(fp))
    if not isinstance(cell, dict):
        return 0
    n = int(_f(cell.get("n")))
    return n if n > 0 and not int(_f(cell.get("ok"))) else 0


# --------------------------------------------------------------- the verdict

def _label(group: str, value: str) -> str:
    option = options_mod.value_of(group, value) or {}
    return _text(option.get("label_es")) or value.replace("_", " ")


def _group_label(group: str) -> str:
    row = options_mod.GROUPS_BY_KEY.get(group) or {}
    return _text(row.get("label_es")) or group.replace("_", " ")


def _best_source(pool: dict, avoid: str, kind: str) -> tuple[str, int, int]:
    """The photograph that best survives ``kind``: (name, passed, measured).

    Judged on the SAME check the finding is about and under the SAME ruler -
    never on the accepted count.  ``ok`` counts every acceptance this
    installation ever recorded, including the ones the old descriptor waved
    through on 2026-09-03, four of which were a different woman; a
    measurement that ``kinds_of`` refused to compare cannot pad a
    recommendation either.  Ties go to the photograph measured more often.
    """
    best = ("", 0, 0)
    best_rate = -1.0
    for key, cell in (pool or {}).items():
        if not key.startswith("src:") or key[4:] == avoid or not isinstance(cell, dict):
            continue
        fails, meas = _rate(cell, kind)
        if meas < MIN_N:
            continue
        rate = (meas - fails) / float(meas)
        if (rate, meas) > (best_rate, best[2]):
            best, best_rate = (_text(cell.get("nombre")), meas - fails, meas), rate
    return best


def _findings(pool: dict, choices: dict[str, list], feats: set,
              src: str = "", src_name: str = "") -> list[dict]:
    """Every measured reason to hesitate about this request.

    Three rules, and each one is a CONTRAST rather than a bare rate, because a
    bare rate convicts whatever happens to be popular.
    """
    out: list[dict] = []

    # 1. A switchable feature of the request that has never once survived.
    #    This is the rule that would have saved 0.61 USD: the coverage clause
    #    was blocked 13 times out of 13 while 0 of 61 prompts without it ever
    #    were, and the robot only found out by paying for all thirteen.
    for name in sorted(feats or set()):
        cell = pool.get("feat:%s" % name)
        fails, meas = _rate(cell, "bloqueo")
        if meas >= 1 and fails == meas:
            # One block out of one is worth saying and is NOT worth stopping
            # for: "ha fallado 1 de 1 veces" is a sentence a coin toss can
            # produce.  Three is the same bar every other rule here uses.
            out.append({"tipo": "ajuste", "clave": name, "fallo": "bloqueo",
                        "veces": fails, "de": meas, "base": 0.0,
                        "nivel": ("confirmar" if meas >= MIN_N else "aviso")})

    # 2. One option value against the other values of its own group.
    for group in sorted(choices):
        gcell = pool.get("grp:%s" % group)
        for value in choices[group]:
            cell = pool.get("opt:%s:%s" % (group, value))
            if not cell or not gcell:
                continue
            for kind in KINDS:
                fails, meas = _rate(cell, kind)
                if meas < MIN_N:
                    continue
                rate = fails / float(meas)
                if rate < HIGH_RATE:
                    continue
                gf, gm = _rate(gcell, kind)
                other_m, other_f = gm - meas, gf - fails
                base = (other_f / float(other_m)) if other_m >= MIN_N else None
                if base is not None and rate < base + MIN_GAP:
                    continue
                out.append({"tipo": "valor", "grupo": group, "valor": value,
                            "fallo": kind, "veces": fails, "de": meas,
                            "base": base,
                            "nivel": ("confirmar" if fails == meas
                                      and meas >= CONFIRM_N else "aviso")})

    # 3. A whole group, against the requests that did not ask for it.  Only the
    #    groups that make the engine redraw the person: see REDRAW_GROUPS.
    for group in sorted(set(choices) & set(REDRAW_GROUPS)):
        gcell, ncell = pool.get("grp:%s" % group), pool.get("sin:%s" % group)
        if not gcell or not ncell:
            continue
        for kind in KINDS:
            fails, meas = _rate(gcell, kind)
            nf, nm = _rate(ncell, kind)
            if meas < MIN_N or nm < MIN_N:
                continue
            rate, base = fails / float(meas), nf / float(nm)
            if rate < HIGH_RATE or rate < base + MIN_GAP:
                continue
            out.append({"tipo": "grupo", "grupo": group, "fallo": kind,
                        "veces": fails, "de": meas, "base": base,
                        "nivel": "aviso"})

    # 4. The photograph itself, against her other photographs.  Same contrast
    #    as rule 2 - a bare rate would convict whichever photo she uses most -
    #    but the level is stricter: an option is one of several things in the
    #    request, the photograph is all of it, and 7 lost faces in 10 on one
    #    file (measured 2026-09-10, against 0 in 17 on another) is not a coin
    #    toss whatever the other options were.
    cell = pool.get("src:%s" % src) if src else None
    if isinstance(cell, dict):
        for kind in KINDS:
            fails, meas = _rate(cell, kind)
            if meas < MIN_N:
                continue
            rate = fails / float(meas)
            if rate < HIGH_RATE:
                continue
            of, om = 0, 0
            for key, other in pool.items():
                if key.startswith("src:") and key[4:] != src and isinstance(other, dict):
                    f2, m2 = _rate(other, kind)
                    of, om = of + f2, om + m2
            base = (of / float(om)) if om >= MIN_N else None
            if base is not None and rate < base + MIN_GAP:
                continue
            mejor = _best_source(pool, src, kind)
            out.append({"tipo": "foto", "fallo": kind, "veces": fails,
                        "de": meas, "base": base,
                        "nombre": src_name or _text(cell.get("nombre")),
                        "mejor": mejor[0], "mejor_ok": mejor[1],
                        "mejor_n": mejor[2],
                        "nivel": ("confirmar" if rate >= 0.6 and meas >= CONFIRM_N
                                  else "aviso")})
    return out


def _de_cuantas(item: dict) -> str:
    """The denominator, said for what it really is.

    Every rate in this module is failures over the images a COMPARABLE ruler
    actually measured, which is the only way to count when a check can be
    skipped - and for ``body_proportions`` it is skipped often: it was read on
    35 of the 60 paid calls that drew anything, because the two figures could
    not always be compared.  Calling that denominator "imagenes pagadas" turns
    a true 3-of-6-comparable into a false 3-of-13-paid in the client's head and
    doubles the rate she is being asked to act on.  The number does not change;
    what it is counting is now said out loud.
    """
    if item["fallo"] == "cuerpo":
        return ("%d imagenes en las que se pudo comparar tu figura"
                % item["de"])
    return "%d imagenes pagadas" % item["de"]


def _finding_text(item: dict) -> str:
    """One finding, in the words of what the client would see happen."""
    kind = KIND_CORTO.get(item["fallo"], item["fallo"])
    veces, de = item["veces"], item["de"]
    if item["tipo"] == "foto":
        base = item.get("base")
        contra = ("" if base is None else
                  (" frente a ninguna con tus otras fotos" if base <= 0.0
                   else " frente al %d%% con tus otras fotos" % round(100 * base)))
        mejor = ""
        if item.get("mejor"):
            mejor = (" Con %s no ha pasado en %d de %d: elige esa, u otra "
                     "donde se te vea entera y con buena luz."
                     % (item["mejor"], item["mejor_n"] - item["mejor_ok"],
                        item["mejor_n"]))
            if item["mejor_ok"] == item["mejor_n"]:
                mejor = (" Con %s ha salido bien en las %d: elige esa, u otra "
                         "donde se te vea entera y con buena luz."
                         % (item["mejor"], item["mejor_n"]))
        return ("Con esta foto (%s) %s en %d de las %d imagenes pagadas%s.%s"
                % (item.get("nombre") or "la elegida", kind, veces, de, contra,
                   mejor))
    if item["tipo"] == "ajuste":
        return ("El texto que exige tapar el cuerpo entero: el proveedor ha "
                "devuelto un archivo en negro y lo ha cobrado en %s. Se "
                "desactiva en Ajustes y no hace falta cambiar nada de lo que "
                "has elegido."
                % ("la unica imagen que lo llevaba" if de == 1
                   else "las %d imagenes que lo llevaban" % de))
    if item["tipo"] == "grupo":
        base = _f(item.get("base"))
        contra = ("y en ninguna de las que no lo pedian" if base <= 0.0
                  else "frente al %d%% de las que no lo pedian" % round(100 * base))
        return ("Cambiar %s obliga al motor a redibujarte entero: %s en %d de "
                "las %d peticiones %s %s. Quitarlo deja el resto igual y "
                "cuesta lo mismo."
                % (_group_label(item["grupo"]).lower(), kind, veces, de,
                   ("en las que se pudo medirlo" if item["fallo"] == "cuerpo"
                    else "que lo pedian"), contra))
    label = _label(item["grupo"], item["valor"])
    base = item.get("base")
    tail = ("" if base is None else
            ", frente al %d%% con las demas opciones de %s"
            % (round(100 * base), _group_label(item["grupo"]).lower()))
    return "%s: %s en %d de %s%s." % (label, kind, veces, _de_cuantas(item), tail)


# ------------------------------------------------------------- the suggestion

def _best_alternative(pool: dict, group: str, kind: str, avoid: set) -> str:
    """The value of this group that the record likes best, or "".

    Ranked by how often it was ACCEPTED, not by how often it survived one
    check: an option that keeps the face and loses the hands is not a safer
    option, and calling it one is the defect the module docstring describes.

    And never a swap this robot has already made twice and watched fail: the
    button on the estimate screen and the change the run makes after a
    rejection are the same recommendation, so they have to be able to be wrong
    about it in the same way and stop offering it at the same moment.
    """
    best, best_score = "", None
    for key, cell in pool.items():
        if not key.startswith("opt:%s:" % group):
            continue
        value = key[len("opt:%s:" % group):]
        if value in avoid:
            continue
        if not adjustment_ok(pool, "adj:%s:cambiar:%s:%s" % (kind, group, value)):
            continue
        n = int(_f(cell.get("n")))
        if n < MIN_N:
            continue
        fails, meas = _rate(cell, kind)
        if meas and fails / float(meas) >= HIGH_RATE:
            continue
        score = int(_f(cell.get("ok"))) / float(n)
        if best_score is None or score > best_score:
            best, best_score = value, score
    if best and best_score is not None and best_score >= 0.5:
        return best
    return ""


def _suggest(pool: dict, findings: list[dict], choices: dict[str, list],
             feats: set) -> dict:
    """ONE concrete change the client can accept with a tap.

    Not advice.  The record says the cheapest fix is always an option change
    and never another attempt - 0 of 5 strength retries rescued an identity
    failure, 0 of 5 rescued a blocked prompt - so what is offered here is the
    request itself, adjusted, at the same price.
    """
    if not findings:
        return {}
    order = {"confirmar": 0, "aviso": 1}
    ranked = sorted(findings, key=lambda f: (order.get(f["nivel"], 2),
                                             -f["veces"], -f["de"]))
    top = ranked[0]

    if top["tipo"] == "foto":
        if top.get("mejor"):
            texto = ("Con %s ha salido bien en %d de %d imagenes pagadas. Se "
                     "manda la misma ropa y el mismo escenario sobre esa foto "
                     "y cuesta exactamente lo mismo."
                     % (top["mejor"], top["mejor_ok"], top["mejor_n"]))
        else:
            texto = ("Ninguna otra foto tuya tiene todavia tres imagenes "
                     "pagadas con las que compararla: elige una donde se te "
                     "vea entera y con buena luz.")
        return {"titulo": "Elegir otra foto tuya", "texto": texto,
                "quitar": [], "cambiar": {}, "ajustes": [], "precio": "mismo",
                "foto": top.get("mejor") or ""}

    if top["tipo"] == "ajuste":
        return {"titulo": "Quitar el texto que exige tapar el cuerpo entero",
                "texto": ("Es una frase del texto que se envia, no una opcion "
                          "tuya: se desactiva en Ajustes y la peticion sigue "
                          "siendo la misma. Sin ella el proveedor no ha "
                          "bloqueado ninguna imagen."),
                "quitar": [], "cambiar": {}, "ajustes": ["texto_cobertura"],
                "precio": "mismo"}

    group = top["grupo"]
    name = _group_label(group).lower()
    resto = _join_rest(choices, group)
    if top["tipo"] == "grupo" or group not in choices:
        return {"titulo": "Quitar el cambio de %s" % name,
                "texto": ("Se queda %s que ya tiene tu foto y se manda todo lo "
                          "demas igual%s. Cuesta exactamente lo mismo."
                          % (_the(group), resto)),
                "quitar": [group], "cambiar": {}, "ajustes": [],
                "precio": "mismo"}

    avoid = set(choices.get(group) or [])
    alt = _best_alternative(pool, group, top["fallo"], avoid)
    if alt:
        return {"titulo": "Cambiar %s por %s" % (_the(group), _label(group, alt)),
                "texto": ("%s es lo que mejor ha salido de ese grupo en las "
                          "imagenes que ya se han pagado. Se manda todo lo "
                          "demas igual%s y cuesta exactamente lo mismo."
                          % (_label(group, alt), resto)),
                "quitar": [], "cambiar": {group: alt}, "ajustes": [],
                "precio": "mismo"}
    return {"titulo": "Quitar el cambio de %s" % name,
            "texto": ("Todavia no hay ninguna opcion de %s que haya salido "
                      "bien las veces suficientes como para recomendarla, asi "
                      "que lo mas barato es no cambiar %s en esta tirada."
                      % (name, _the(group))),
            "quitar": [group], "cambiar": {}, "ajustes": [], "precio": "mismo"}


# Spanish reads badly without them and the client reads every one of these
# sentences on a phone, so the article travels with the group name.
_ARTICLE = {"clothing": "la ropa", "clothing_color": "el color de la ropa",
            "pose": "la postura", "scene": "el escenario",
            "lighting": "la luz", "hair": "el peinado",
            "expression": "la expresion", "framing": "el encuadre",
            "grade": "el acabado de color", "transparency": "el tejido",
            "treatment": "el tratamiento"}


def _the(group: str) -> str:
    return _ARTICLE.get(group) or _group_label(group).lower()


def _join_rest(choices: dict, group: str) -> str:
    """"(la ropa y el escenario)" - what she keeps, named so she can see it."""
    rest = [_the(g) for g in sorted(choices) if g != group and _the(g)]
    if not rest:
        return ""
    if len(rest) == 1:
        return " (%s)" % rest[0]
    return " (%s y %s)" % (", ".join(rest[:-1]), rest[-1])


# The same three label helpers under public names, because generation/adjust.py
# has to say "se ha cambiado la ropa por Vaqueros y camiseta" in exactly the
# words this screen uses.  Two spellings of the same garment on two screens is
# how a client stops believing either of them.
label, group_label, the = _label, _group_label, _the


# ------------------------------------------------------------------- the API

def assess(plan: Any, user_id: str = "", profile_id: Any = "",
           endpoint: str = "", quality: str = "", style: Any = "",
           free_engine: bool = False) -> dict:
    """The risk assessment that travels on the estimate.

    Reads only.  Every failure here is swallowed: this sentence is worth money
    but it is never worth a run.
    """
    empty = {"nivel": "ninguno", "titulo": "", "resumen": "", "motivos": [],
             "ajuste": {}, "confirmacion": "", "gratis": {}, "aprendido": {}}
    try:
        plan_d = plan if isinstance(plan, dict) else {}
        choices = _choices_of(plan_d)
        feats = features_of(plan_d)
        if not choices:
            return empty

        # A RUN THAT COSTS NOTHING IS NOT WORTH NAGGING ABOUT, and the paid
        # record is not evidence about it either: the free engine transforms
        # her photograph instead of redrawing her, so an option that lost her
        # face on kontext says nothing about what a colour grade will do.  The
        # loud sentence stays, the six warnings go.
        if free_engine:
            gratis = _free_note(choices, True)
            return dict(empty, gratis=gratis, resumen=gratis.get("texto") or "")

        shared = shared_pool()
        mine = person_pool(user_id, profile_id)
        # HER OWN RECORD FIRST, THE INSTALLATION'S UNDERNEATH.  A new person
        # has nothing of her own and inherits everything the robot has already
        # paid to learn; once she has her own three images of an option, hers
        # are what decide, because they were measured on her face.
        pool = _prefer(shared, mine)
        own_n = sum(int(_f(c.get("n"))) for k, c in mine.items()
                    if k.startswith("opt:"))

        src_key, src_name = _plan_source(plan_d)
        findings = _findings(pool, choices, feats, src_key, src_name)

        # And the one thing no counter can know: has this EXACT request been
        # bought before and come back wrong?  26 paid calls, 1.07 USD, repeated
        # a fingerprint that had already failed, and 21 of them failed again.
        # Nothing in the code stopped it, because the only memory of a failure
        # lived in a local variable for the life of one variant of one run.
        #
        # ONE FINGERPRINT PER IMAGE THIS PLAN WILL BUY, not one for the plan.
        # The orchestrator writes the fingerprint of the VARIANT it paid for -
        # that variant's own choices - while this used to compute a single
        # fingerprint over the union of every variant's choices.  The two are
        # only the same string when the plan has exactly one variant, so a
        # four-up preview could re-offer a combination that had already been
        # bought and rejected and nothing would say a word.  That is precisely
        # the case the client's "que no vuelva a ocurrir" is about, so every
        # variant is asked, and the worst answer is the one she is shown.
        prints = {fingerprint(src_key or plan_d.get("source_path"),
                              {g: v for g, v in ((var or {}).get("choices") or {}).items()},
                              quality, style, endpoint)
                  for var in (plan_d.get("variants") or [])
                  if (var or {}).get("choices")}
        prints.add(fingerprint(src_key or plan_d.get("source_path"), choices, quality,
                               style, endpoint))
        worst = max((pool.get("fp:%s" % f) for f in prints),
                    key=lambda c: (int(_f((c or {}).get("n")))
                                   if isinstance(c, dict)
                                   and not int(_f(c.get("ok"))) else 0),
                    default=None)
        repeat = worst if isinstance(worst, dict) else None
        if repeat and int(_f(repeat.get("n"))) > 0 and not int(_f(repeat.get("ok"))):
            times = int(_f(repeat.get("n")))
            blocked = int(_f((repeat.get("f") or {}).get("bloqueo")))
            findings.insert(0, {
                "tipo": "peticion", "clave": "identica", "fallo": "repeticion",
                "veces": times, "de": times, "base": None,
                "nivel": ("confirmar" if times >= 2 or blocked else "aviso")})

        if not findings:
            gratis = _free_note(choices, free_engine)
            return dict(empty, gratis=gratis,
                        aprendido={"tuyas": own_n,
                                   "del_robot": sum(int(_f(c.get("n")))
                                                    for k, c in shared.items()
                                                    if k.startswith("opt:"))},
                        resumen=(gratis.get("texto") or ""))

        level = ("confirmar" if any(f["nivel"] == "confirmar" for f in findings)
                 else "aviso")
        motivos = []
        for item in findings:
            texto = (("Esta peticion exacta - la misma foto, las mismas "
                      "opciones, la misma calidad - ya se pago %s y no salio "
                      "ninguna imagen buena."
                      % ("una vez" if item["veces"] == 1
                         else "%d veces" % item["veces"]))
                     if item["tipo"] == "peticion" else _finding_text(item))
            motivos.append({"texto": texto, "tipo": item["tipo"],
                            "fallo": item["fallo"], "veces": item["veces"],
                            "de": item["de"], "nivel": item["nivel"],
                            "grupo": item.get("grupo", ""),
                            "valor": item.get("valor", "")})

        suggestion = _suggest(pool, [f for f in findings
                                     if f["tipo"] != "peticion"], choices, feats)
        hard = [f for f in findings if f["nivel"] == "confirmar"]
        confirmacion = ""
        if hard:
            worst = max(hard, key=lambda f: (f["de"], f["veces"]))
            confirmacion = (("Con esta foto han fallado %d de %d imagenes "
                             "pagadas. Si aun asi quieres pagarla, confirmalo.")
                            if worst["tipo"] == "foto" else
                            ("Esta combinacion ha fallado %d de %d veces. Si "
                             "aun asi quieres pagarla, confirmalo.")) % (worst["veces"], worst["de"])
        titulo = ("Esto ya ha fallado antes" if level == "confirmar"
                  else "Lo que suele salir mal con esta peticion")
        return {"nivel": level, "titulo": titulo,
                "resumen": " ".join(m["texto"] for m in motivos[:3]),
                "motivos": motivos, "ajuste": suggestion,
                "confirmacion": confirmacion,
                "gratis": _free_note(choices, free_engine),
                "aprendido": {"tuyas": own_n,
                              "del_robot": sum(int(_f(c.get("n")))
                                               for k, c in shared.items()
                                               if k.startswith("opt:"))}}
    except Exception:                                     # noqa: BLE001
        return empty


def _free_note(choices: dict, free_engine: bool) -> dict:
    """Said loudly when it applies, because it is the only free answer there is.

    The free engine transforms her photograph instead of redrawing her, and in
    this installation it has never been blocked by a provider and has never
    produced an anatomy defect in 48 attempts.  It also cannot put a garment on
    anybody, so the offer is only made when every group asked for is one it can
    really perform - promising it for a clothing change is how a run was
    announced as "sin coste" and then delivered the wrong images.
    """
    groups = set(choices or {})
    if not groups or not groups.issubset(set(FREE_GROUPS)):
        return {"posible": False, "texto": ""}
    if free_engine:
        return {"posible": True, "ya": True,
                "texto": ("Esta peticion la hace el motor gratuito: solo "
                          "cambias escenario, luz, color y encuadre, y eso se "
                          "hace transformando tu propia foto. No se gasta "
                          "nada y el proveedor no puede bloquearla.")}
    return {"posible": True, "ya": False,
            "texto": ("Todo lo que pides - escenario, luz, color, encuadre - "
                      "lo puede hacer el motor gratuito sobre tu propia foto, "
                      "sin pagar y sin que nadie pueda bloquearla. Elige "
                      "'motor local' en Ajustes si no quieres gastar.")}
