"""WHICH option caused this rejection, and the smallest change that fixes it.

The client's sentence was "si ocurre un error de rechazo, ajusta las opciones
para que no vuelva a ocurrir".  ``generation/risk.py`` is the half that runs
BEFORE the money moves; this is the half that runs after an image has already
been paid for and thrown away, and it answers one question: given what failed,
what is the smallest change to the REQUEST that the record says would have
helped - and is it worth another 0.04 USD at all?

WHY NOT MORE STRENGTH, LESS STRENGTH, ANOTHER SEED.  That is what this loop did
until today: ``params["strength"] -= 0.08`` and ``seed += 977``, the same
request bought again with the dials moved.  The record says exactly what that
is worth, over 25 paid retries and 1.03 USD:

    first failure was            retries bought   then accepted
    identity                            5               0
    a blocked prompt (coverage clause)  5               0
    body proportions                    6               2
    anatomy                             8               2
    a block WITHOUT the clause          1               1   (transient)

Identity is the flat zero that matters: run_16e276b5 lowered the strength from
0.52 to 0.44 to 0.36 across three paid attempts of one variant and read 0.3507,
0.3753 and 0.3643 against a line at 0.45.  A dial that moves the number by less
than its own noise is not an adjustment, it is the same request bought three
times.

WHAT DOES MOVE IT is an option, and which option is a measurement rather than an
opinion.  Every rule below is read out of the same counters the estimate screen
reads - ``risk._findings``, with the same MIN_N / HIGH_RATE / MIN_GAP contrast -
so the sentence the client saw before she paid and the change the robot makes
after she paid cannot disagree.  What this module adds on top of those counters
is only the mapping from a failed CHECK to the levers worth pulling for it:

  identidad  ->  the garment first, then whichever redraw group the counters
                 convict.  Under the current recogniser a full length dress
                 lost her face in 10 of 12 paid images and every other garment
                 in 0 of 15.
  cuerpo     ->  the pose.  Asking for a different body position moved the
                 figure in 8 of the 18 requests where the proportions could be
                 measured, and in 0 of the 17 that did not ask for one.
  anatomia   ->  only when the defect is a real body part.  Smoothed skin is
                 not bought out of: ``orchestrator._restore_texture`` already
                 gives the grain back from her own photograph for nothing, and
                 the 8 paid anatomy retries rescued 2.
  bloqueo    ->  the coverage clause, switched off.  13 of 13 prompts carrying
                 it came back as a charged black file and 0 of the 61 without
                 it ever did.  When it is already off there is nothing here to
                 adjust: the one block that happened without it was retried
                 byte identical and SUCCEEDED, so that case is left to the
                 orchestrator's existing "two blocks and stop".
  piel,
  calidad    ->  nothing.  The quality check has never rejected a paid image
                 (0 of 60) and skin tone was never the sole reason (3 of 60),
                 so there is no measured lever and no attempt is bought.

AND THE ANSWER IS ALLOWED TO BE "NO".  ``decide`` refuses far more often than
it changes something, because an adjusted retry is only worth 0.04 USD when the
record expects it to help.  A refusal is not this module failing, it is the
point of it: 1.81 USD of the 3.05 USD this installation has spent bought no
usable image, and 1.03 USD of that was retries.
"""
from __future__ import annotations

from typing import Any

from . import risk as risk_mod

# The order a rejection is acted on when several checks failed at once.  A face
# that is not hers cannot be repaired by fixing the pose, and an image the
# provider never delivered has no defects to read at all.
PRIORITY = ("bloqueo", "identidad", "cuerpo", "anatomia", "piel", "calidad")

# The kinds this module has a measured lever for.  ``piel`` and ``calidad`` are
# deliberately absent - see the module docstring.
ACTIONABLE = ("bloqueo", "identidad", "cuerpo", "anatomia")

# Which groups are worth pulling for which failure, most specific first.  These
# are the levers CONSIDERED; whether one is actually pulled is decided by the
# counters, so a group named here with nothing measured against it changes
# nothing.  Only groups that make the engine redraw the person appear: a scene
# or a light does not move her shoulders, however a small sample happens to
# read.
LEVERS: dict[str, tuple] = {
    "identidad": ("clothing", "pose", "expression", "hair", "transparency"),
    "cuerpo": ("pose", "clothing", "expression"),
    "anatomia": ("pose", "clothing"),
}

# A defect that is a real malformed body part, as opposed to skin the engine
# sanded.  ``verify`` reports both under the one check name ``anatomy`` and they
# are two unrelated problems: of the 20 anatomy rejections in this installation
# 12 are hands and 14 are smoothing, and only the first is worth buying a
# different request for.
BODY_DEFECTS = ("hand_malformed", "missing_limb", "duplicated_feature",
                "face_distorted", "extra_limb", "texture_smear")

# How much history it takes to conclude that an adjustment does not work.  The
# rule lives in ``risk`` because the estimate screen's one-tap button obeys it
# too - a change the run has already tried twice and watched fail must stop
# being offered on both screens at the same moment, or the robot would be
# arguing with itself in front of the client.
ADJ_MIN_N = risk_mod.ADJ_MIN_N

# What the client is told happened, in her words rather than the check's.
OPENING: dict[str, str] = {
    "bloqueo": "no ha llegado (el proveedor la ha devuelto en negro y la cobra igual)",
    "identidad": "no se parecia a ti",
    "cuerpo": "te cambiaba la figura",
    "anatomia_manos": "tenia una mano mal dibujada",
    "anatomia_piel": "tenia la piel demasiado lisa",
    "piel": "ha salido con otro tono de piel",
    "calidad": "ha salido borrosa",
}


def diagnose(status: str, reason: str, verdict: Any,
             defects: Any = None) -> tuple[str, str]:
    """(the failure to act on, the name the client is told).

    Read through ``risk.kinds_of`` rather than off the verdict directly, so a
    check written by a ruler that has since been recalibrated is never acted
    on: the identity score in this database means two different things either
    side of 2026-09-04 and only one of them is a measurement of who is in the
    picture.
    """
    failed, _measured = risk_mod.kinds_of(status, reason, verdict)
    kind = next((k for k in PRIORITY if k in failed), "")
    if kind != "anatomia":
        return kind, kind
    types = {str((d or {}).get("type") or "") for d in (defects or [])
             if isinstance(d, dict)}
    if types & set(BODY_DEFECTS):
        return kind, "anatomia_manos"
    # Nothing but smoothed skin.  The free grain transfer has already run on
    # this file by the time this is reached, so a second paid draw would be
    # asked to fix the thing that was already fixed for nothing.
    return kind, "anatomia_piel"


def _candidates(pool: dict, kind: str, choices: dict, feats: set) -> list[dict]:
    """The findings that could explain THIS failure, best supported first.

    ``risk._findings`` is the same reader the estimate screen uses, restricted
    here to the one kind that actually went wrong.  Ranking is by the CONTRAST -
    how much worse this value or this group is than the requests without it -
    and never by the bare rate, because a bare rate convicts whatever is
    popular: 'clothing' rides on 74 of 74 paid calls, so its failure rate is
    simply the installation's failure rate.
    """
    asked = {g: [str(v)] for g, v in (choices or {}).items() if v}
    found = [f for f in risk_mod._findings(pool, asked, set(feats or ()))
             if f.get("fallo") == kind]
    levers = LEVERS.get(kind, ())
    # THE LEVER TABLE IS A FILTER AND NOT ONLY AN ORDER.  Until this line it
    # was only the sort key, so whenever no lever group was convicted the loop
    # fell through to whatever else the counters happened to name and bought a
    # different request on the strength of it.  Measured against the shipped
    # seed, that is four wrong purchases of 0.04 USD each:
    #   identidad + scene=playa_atardecer  -> "se ha cambiado el escenario",
    #     on 10 of 15, a count that is perfectly confounded with the 12 dress
    #     calls and that no mechanism connects to a face;
    #   cuerpo + lighting=ventana_izq      -> "se ha cambiado la luz", 3 of 7;
    #   anatomia + scene=ciclorama_blanco  -> "se ha cambiado el escenario",
    #     4 of 5, the same two runs the hand cluster is made of;
    #   bloqueo + clothing=camisa_blanca   -> "se ha cambiado la ropa", 5 of 7,
    #     where all 5 of those blocks are coverage-clause blocks and the
    #     garment is standing in for the phrase.  LEVERS has no entry for a
    #     block on purpose: the only block ever seen WITHOUT the clause was
    #     retried byte identical and succeeded, so there is nothing here the
    #     record supports changing, and now nothing is changed.
    # The feature findings (``tipo == "ajuste"``) are not group findings and
    # are kept: the coverage clause is the one adjustment with 13 of 13 behind
    # it and it belongs to no option group at all.
    found = [f for f in found
             if f.get("tipo") == "ajuste" or str(f.get("grupo") or "") in levers]

    def gap(item: dict) -> float:
        base = item.get("base")
        rate = item["veces"] / float(max(1, item["de"]))
        return rate - (0.0 if base is None else float(base))

    def rank(item: dict) -> tuple:
        group = str(item.get("grupo") or "")
        # A lever this failure has a mechanism for beats one it does not.  It
        # is what separates the two readings of the same eight images: pose
        # 'sentada' and garment 'gabardina' both carry 6 malformed hands out of
        # 8 because they are the SAME two runs seen twice, and re-posing her is
        # the smaller change of the two.
        known = levers.index(group) if group in levers else len(levers) + 1
        hard = 0 if item.get("nivel") == "confirmar" else 1
        return (hard, known, -gap(item), -item["veces"])

    return sorted(found, key=rank)


def _adjustment_ok(pool: dict, key: str) -> bool:
    """Has this exact adjustment been made before and never once worked?

    THIS IS THE HALF THAT LETS A BAD IDEA BE LEARNED AWAY.  Without it the
    module could only ever grow more confident: every adjustment it made would
    be counted as evidence about the OPTION it moved to and never as evidence
    about the recommendation itself, so a change that reliably made things
    worse would be made again on every run, forever, and the client would pay
    0.04 USD for it every time.
    """
    return risk_mod.adjustment_ok(pool, key)


def _evidence(item: dict) -> str:
    """The numbers behind one adjustment, in a sentence she can check."""
    base = item.get("base")
    tail = ("" if base is None else
            ", frente al %d%% de las demas" % round(100 * float(base)))
    if item["tipo"] == "ajuste":
        return ("esa frase ha vuelto en negro en %d de las %d imagenes que la "
                "llevaban" % (item["veces"], item["de"]))
    # The same denominator the estimate screen uses, and said the same way: a
    # body_proportions rate is over the images where the two figures could be
    # compared (35 of the 60 that drew), not over every image paid for, and the
    # sentence beside the retry must not round that up any more than the
    # sentence beside the price does.
    de = risk_mod._de_cuantas(item)
    if item["tipo"] == "grupo":
        return ("pedir ese cambio ha salido mal en %d de las %d imagenes %s%s"
                % (item["veces"], item["de"],
                   ("en las que se pudo comparar tu figura"
                    if item["fallo"] == "cuerpo" else "pagadas que lo pedian"),
                   tail))
    return ("%s ha salido mal en %d de las %s%s"
            % (risk_mod.label(str(item["grupo"]), str(item["valor"])),
               item["veces"], de, tail))


def decide(kind: str, shown: str, choices: dict, feats: set, pool: dict,
           done: Any = ()) -> dict:
    """The one change to make before buying another image, or the reason not to.

    ``done`` is what has already been adjusted on this variant.  It is normally
    empty and never holds more than one entry: see the orchestrator, which buys
    at most one adjusted attempt per variant because the record is clear that
    the second retry is where the money goes and the images do not.
    """
    done = set(done or ())
    no = {"ok": False, "clave": "", "accion": "", "texto": "", "motivo": "",
          "grupo": "", "valor": "", "por": "", "kind": kind, "mostrado": shown}
    if kind not in ACTIONABLE:
        return dict(no, parar=("No hay ningun cambio de opciones que el "
                               "historial respalde para esto, asi que no se "
                               "paga otro intento."))
    if shown == "anatomia_piel":
        # Measured, and the reason this is a stop and not a retry: the grain
        # transfer that runs for free on every generated file has already been
        # applied to this image, and the 8 anatomy retries this installation
        # paid for rescued 2 of them.
        return dict(no, parar=("La piel demasiado lisa ya se corrige gratis "
                               "con la textura de tu propia foto y eso ya se "
                               "ha hecho; pagar otra imagen por ese motivo ha "
                               "funcionado 2 de 8 veces, asi que no se paga."))

    for item in _candidates(pool, kind, choices, feats):
        # 1. A switchable phrase in the text that is sent, not one of her
        #    choices: the cheapest fix there is, because nothing she asked for
        #    changes at all.
        if item["tipo"] == "ajuste" and item.get("clave") == "texto_cobertura":
            key = "adj:%s:texto_cobertura" % kind
            if key in done or not _adjustment_ok(pool, key):
                continue
            return {"ok": True, "clave": key, "accion": "ajuste",
                    "grupo": "", "valor": "", "por": "",
                    "kind": kind, "mostrado": shown,
                    "texto": ("se ha quitado del texto la frase que exigia "
                              "taparte el cuerpo entero"),
                    "motivo": _evidence(item)}

        group = str(item.get("grupo") or "")
        if not group or group not in (choices or {}):
            continue
        value = str(choices[group])

        # 2. The same group, a different value: the smallest change that still
        #    gives her the kind of thing she asked for.  Ranked by how often a
        #    value was ACCEPTED and not by how often it survived this one
        #    check, because an option that keeps the face and loses the hands
        #    is not a safer option.
        if item["tipo"] == "valor":
            alt = risk_mod._best_alternative(pool, group, kind, {value})
            if alt:
                key = "adj:%s:cambiar:%s:%s" % (kind, group, alt)
                if key in done or not _adjustment_ok(pool, key):
                    continue
                return {"ok": True, "clave": key, "accion": "cambiar",
                        "grupo": group, "valor": value, "por": alt,
                        "kind": kind, "mostrado": shown,
                        "texto": ("se ha cambiado %s por %s"
                                  % (risk_mod.the(group),
                                     risk_mod.label(group, alt))),
                        "motivo": _evidence(item)}

        # 3. Nothing in that group is worth recommending, or the whole group is
        #    what the counters convict: drop it and keep what her own
        #    photograph already has.  Same price, and it is the change with the
        #    cleanest support in the record - 0 of the 17 requests that asked
        #    for no pose ever moved her figure.
        #
        #    NEVER THE LAST GROUP.  A request with nothing left in it is not a
        #    cheaper request, it is a paid image of the photograph she already
        #    owns; if the only thing she asked for is the thing that failed,
        #    the honest answer is to stop and say so.
        if len(choices or {}) <= 1:
            continue
        key = "adj:%s:quitar:%s" % (kind, group)
        if key in done or not _adjustment_ok(pool, key):
            continue
        return {"ok": True, "clave": key, "accion": "quitar",
                "grupo": group, "valor": value, "por": "",
                "kind": kind, "mostrado": shown,
                "texto": ("se ha quitado el cambio de %s"
                          % risk_mod.group_label(group).lower()),
                "motivo": _evidence(item)}

    return dict(no, parar=("Ningun cambio de opciones tiene detras suficientes "
                           "imagenes pagadas como para esperar que arregle "
                           "esto, asi que no se paga otro intento."))


def apply_to(choices: dict, coverage_text: bool,
             plan: dict) -> tuple[dict, bool]:
    """The adjusted request.  Returns a NEW choices dict and never mutates."""
    out = dict(choices or {})
    cover = bool(coverage_text)
    action = str((plan or {}).get("accion") or "")
    group = str((plan or {}).get("grupo") or "")
    if action == "quitar" and group:
        out.pop(group, None)
    elif action == "cambiar" and group and plan.get("por"):
        out[group] = str(plan["por"])
    elif action == "ajuste":
        cover = False
    return out, cover


def sentence(index: int, plan: dict) -> str:
    """What the client reads: what went wrong, what was changed, and why.

    One sentence, in the order she would ask the questions in.  The numbers
    travel with it because "se ha cambiado la ropa" on its own is the robot
    asking to be trusted, and the whole argument of this product is that it
    does not have to be.
    """
    opening = OPENING.get(str(plan.get("mostrado") or ""), "no se pudo aceptar")
    return ("La imagen %d %s, asi que %s y se ha vuelto a intentar (%s). El "
            "segundo intento cuesta lo mismo que el primero."
            % (index, opening, plan.get("texto") or "se ajusto la peticion",
               plan.get("motivo") or "medido sobre las imagenes ya pagadas"))


def stop_sentence(index: int, shown: str, why: str) -> str:
    """What she reads when the honest answer is "no se paga otro intento"."""
    opening = OPENING.get(shown, "no se pudo dar por buena")
    return "La imagen %d %s. %s" % (index, opening, why)
