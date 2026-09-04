"""How many photographs a new person has to give, and what is measured on them.

Two things live here, and both exist because the product is no longer for one
woman: the RULE about how many reference photographs an account must upload
before it may generate, and the FIRST RUN REPORT that measures what can and
cannot be verified for that particular person before she spends anything.

Nothing in this module knows who anybody is.  Every number it prints is read
off the account's own photographs at the moment it runs.


=========================  WHY FIVE  =========================================

The minimum was measured, not chosen.  Corpus: the 24 photographs of the
person this installation already holds, embedded with the same SFace path
production uses (identity/embedding.py), plus eight photographs of eight other
women as impostors.  Two of the ten files in sample/ turned out to be HER
(0.8267 and 0.7565 against her own mean, 0.82 against each other) and were
taken out of the impostor bank rather than left in to flatter the numbers.

For each gallery size k, every subset of k photographs was made into a profile
mean exactly the way build_profile makes one; her REMAINING photographs were
scored against it as positives, the eight strangers as negatives, and the
margin is the worst positive minus the best impostor - the quantity that
decides whether a threshold exists at all.  3000 random subsets per size where
there were more than that.

    k    peor foto suya (min/mediana)   mejor impostora   MARGEN (min/mediana)
    1        0.4125 / 0.5060                0.2133          0.1985 / 0.2965
    2        0.4663 / 0.5706                0.2070          0.2553 / 0.3670
    3        0.4967 / 0.6044                0.2062          0.2759 / 0.3987
    5        0.5434 / 0.6351                0.2048          0.3158 / 0.4290
    8        0.5715 / 0.6542                0.2039          0.3356 / 0.4490
    12       0.6047 / 0.6724                0.2040          0.3709 / 0.4686
    20       0.6335 / 0.7358                0.2046          0.4175 / 0.5302
    24       0.6565 (leave-one-out)         0.2054          0.4511

What each added photograph buys, in median margin: the 2nd +0.0705, the 3rd
+0.0317, the 4th and 5th +0.0152 each, the 6th to 8th +0.0067 each, the 9th to
12th +0.0049 each.  The gain halves at 3 and halves again at 5; after 5 it is
under a tenth of what the second photograph was worth.  THAT is where the
curve flattens, and it is why the minimum is 5 rather than 3 or 8.

Two independent measurements agree with it and are the reason 5 wins over 3:

* FALSE REJECTION.  With ONE photograph in the profile, 12.5% of possible
  profiles reject at least one of her own photographs outright - the worst
  positive is 0.4125 against a 0.45 line.  From two photographs on it is 0.0%
  over every subset tried (6072 at k=2, 42504 at k=3).  One photograph is not
  a profile, it is a coin toss.
* WHAT CAN BE CHECKED AT ALL.  Her gallery is 13 closeups, 7 full length and 4
  half length.  Drawing k at random from that mix, the chance of having at
  least one photograph the body ruler can measure is 45.8% at k=1, 71.7% at 2,
  85.9% at 3, 97.0% at 5, 99.9% at 8; the chance of the two shot types
  gallery.choose_references needs for a real reference trio is 0% at 1, 62.0%
  at 2, 83.9% at 3, 97.0% at 5.  Five is the first size where both are at 97%.

So: FIVE is refused below, EIGHT is what the screen asks for, and everything
above eight is welcome and worth about half a point of margin per photograph.
The counts are person-agnostic; a second account's own numbers are measured by
first_run_report and shown to her before she spends.
"""
from __future__ import annotations

import logging
import time
from typing import Any

from .. import db
from . import gallery as gallery_mod
from . import profile as profile_mod

log = logging.getLogger("photorobot.onboarding")

MIN_PHOTOS = 5          # refused below this: see the table above
RECOMMENDED_PHOTOS = 8  # what the interface asks for
REFERENCE_COUNT = 3     # photographs sent with a paid request

# How many photographs the hand pass reads on the first run.  Each one costs
# about two seconds (pose + segmentation + the anomaly scan), the readings are
# cached per photograph afterwards, and the question it answers - "can this
# person's hands be judged at all" - is settled long before the twelfth
# photograph.  A person with forty photographs must not wait eighty seconds for
# an answer that stops changing at eight.
HAND_SAMPLE = 8

SHOT_ES = {"closeup": "primer plano", "half": "medio cuerpo",
           "full": "cuerpo entero", "unknown": "sin identificar"}

# What the account is told, in her language, about why the robot wants several
# photographs and what happens to them.  Kept here rather than in the page so
# the API, the report and the screen cannot drift apart.
WHY_ES: tuple[str, ...] = (
    "Con una sola foto el robot no sabe distinguirte: medido sobre 24 fotos de "
    "una persona real, un perfil de una sola foto rechaza una de sus propias "
    "fotos el 12,5% de las veces. Con cinco, ninguna.",
    "Cada foto separa mas tu cara de la de otra persona. La segunda foto es la "
    "que mas suma (+0,07 de margen), la tercera +0,03, la cuarta y la quinta "
    "+0,015 cada una; a partir de la quinta la mejora es diez veces menor.",
    "Hacen falta encuadres distintos: con solo primeros planos no se pueden "
    "medir tus proporciones, y es justo lo que evita que una imagen te "
    "adelgace sin que te des cuenta.",
    "De tus fotos se eligen tres para enviarlas juntas cuando haya que dibujar "
    "tu rostro. Esas tres cubren tu peor foto y ademas abaratan la imagen.",
)

FATE_ES: tuple[str, ...] = (
    "Tus fotos se guardan en este servidor, en tu carpeta, y solo las ve tu "
    "cuenta. Ni el administrador puede abrirlas.",
    "Se usan para medirte: tu rostro, tu tono de piel, tus proporciones y tus "
    "marcas. Con eso se comprueba que cada imagen generada sigues siendo tu.",
    "No se publican, no se envian a nadie salvo al motor que dibuja la imagen "
    "que tu pidas, y nunca se usan para entrenar nada.",
    "Cuando tengas el perfil hecho puedes pulsar 'Olvidar fotos': se borran los "
    "archivos y se quedan solo las medidas. El sistema sigue funcionando.",
    "Puedes borrar cualquier foto, o tu cuenta entera, cuando quieras.",
)

ADVICE_ES: tuple[str, ...] = (
    "Al menos dos de cuerpo entero, de la cabeza a los pies.",
    "Al menos una de medio cuerpo.",
    "Al menos una de cara, de cerca.",
    "Sin filtros ni retoque de belleza: si la foto lleva filtro, el robot "
    "aprende la version filtrada de ti.",
    "Con luz distinta y ropa distinta entre unas y otras.",
)


# ----------------------------------------------------------------- utilities

def _f(value: Any, default: float = 0.0) -> float:
    try:
        out = float(value)
        return out if out == out else default            # noqa: PLR0124  (NaN)
    except (TypeError, ValueError):
        return default


def _originals(user_id: str, profile_id: str = "") -> list[dict]:
    """Her live reference photographs, optionally those of one profile.

    Always scoped by user_id.  A profile_id that belongs to somebody else
    simply matches nothing rather than reaching across accounts.
    """
    if profile_id:
        rows = db.q(
            "SELECT * FROM originals WHERE user_id=? AND deleted_at IS NULL "
            "AND (profile_id=? OR profile_id IS NULL) ORDER BY sort_order, created_at",
            (user_id, profile_id))
    else:
        rows = db.q(
            "SELECT * FROM originals WHERE user_id=? AND deleted_at IS NULL "
            "ORDER BY sort_order, created_at", (user_id,))
    return db.rows_to_dicts(rows)


def _by_shot(rows: list[dict]) -> dict[str, int]:
    counts = {"closeup": 0, "half": 0, "full": 0, "unknown": 0}
    for row in rows:
        key = str(row.get("shot_type") or "unknown")
        counts[key if key in counts else "unknown"] += 1
    return counts


# --------------------------------------------------------------- the minimum

def readiness(user_id: str, profile_id: str = "") -> dict:
    """Can this account generate yet, and what is still missing.

    Answers in whole photographs and in Spanish, because it is rendered to
    somebody who has just registered and has no idea what an embedding is.
    """
    rows = _originals(user_id, profile_id)
    counts = _by_shot(rows)
    n = len(rows)
    missing = max(0, MIN_PHOTOS - n)
    body_ready = (counts["full"] + counts["half"]) >= 1
    shots = sum(1 for key in ("closeup", "half", "full") if counts[key] > 0)

    if missing == 1:
        message = ("Te falta 1 foto para poder generar: el robot necesita al "
                   "menos %d fotos tuyas de verdad y ahora tiene %d."
                   % (MIN_PHOTOS, n))
    elif missing:
        message = ("Te faltan %d fotos para poder generar: el robot necesita "
                   "al menos %d fotos tuyas de verdad y ahora tiene %d."
                   % (missing, MIN_PHOTOS, n))
    elif n < RECOMMENDED_PHOTOS:
        message = ("Ya puedes generar con %d fotos. Con %d el robot te "
                   "distingue bastante mejor: sube alguna mas cuando puedas."
                   % (n, RECOMMENDED_PHOTOS))
    else:
        message = ("Tienes %d fotos: suficientes para medirte bien." % n)

    pending: list[str] = []
    if missing:
        pending.append("Sube 1 foto mas." if missing == 1
                       else "Sube %d fotos mas." % missing)
    if counts["full"] < 2:
        pending.append("Faltan fotos de cuerpo entero (tienes %d, hacen falta 2): "
                       "sin ellas no se pueden medir tus proporciones."
                       % counts["full"])
    if counts["half"] < 1:
        pending.append("Falta alguna foto de medio cuerpo.")
    if counts["closeup"] < 1:
        pending.append("Falta alguna foto de cara, de cerca.")

    # Whether the one-off thorough reading has already been done for the
    # default profile, so the estimate screen can warn that the first one takes
    # a while instead of looking hung.
    done = bool(db.q1(
        "SELECT 1 FROM profiles WHERE user_id=? AND deleted_at IS NULL "
        "AND first_run_json LIKE '%\"hecho\": true%'", (user_id,)))

    return {
        "fotos": n,
        "analisis_hecho": done,
        "minimo": MIN_PHOTOS,
        "recomendado": RECOMMENDED_PHOTOS,
        "faltan": missing,
        "puede_generar": missing == 0,
        "por_plano": counts,
        "tipos_de_plano": shots,
        "cuerpo_medible_posible": body_ready,
        "mensaje": message,
        "pendiente": pending,
        "porque": list(WHY_ES),
        "que_pasa_con_tus_fotos": list(FATE_ES),
        "como_hacerlas": list(ADVICE_ES),
    }


def blocking_reason(user_id: str, profile_id: str = "") -> str:
    """The sentence that refuses a generation, or "" when there is none."""
    state = readiness(user_id, profile_id)
    if state["puede_generar"]:
        return ""
    return (
        "%s Sube unas cuantas en 'Mis fotos' y vuelve. El robot compara cada "
        "imagen que produce con tus fotos reales para comprobar que sigues "
        "siendo tu, y con menos de %d no puede hacerlo con fiabilidad: con una "
        "sola foto llega a rechazar tus propias fotos el 12,5%% de las veces."
        % (state["mensaje"], MIN_PHOTOS))


# ------------------------------------------------------- storing the profile

def store_profile(profile_id: str, built: dict) -> None:
    """Write a freshly measured profile onto its row.

    Lives here rather than in routers/profiles.py because two callers now need
    it - the Mis fotos screen and the first run analysis - and a second copy of
    this UPDATE is a second place for the column list to go stale.
    """
    coverage = built.get("coverage") or {}
    status = "ready" if coverage.get("ready_for_body_check") else "draft"
    db.execute(
        "UPDATE profiles SET n_sources=?, coverage_json=?, face_json=?, "
        "body_json=?, skin_json=?, hair_json=?, marks_json=?, thresholds_json=?, "
        "status=?, updated_at=? WHERE id=?",
        (int(built.get("n_sources") or 0), db.dumps(coverage),
         db.dumps(built.get("face") or {}), db.dumps(built.get("body") or {}),
         db.dumps(built.get("skin") or {}), db.dumps(built.get("hair") or {}),
         db.dumps(built.get("marks") or []),
         db.dumps(built.get("thresholds") or {}), status, db.now(), profile_id))


def _profile_row(profile_id: str, user_id: str) -> dict:
    return db.row_to_dict(db.q1(
        "SELECT * FROM profiles WHERE id=? AND user_id=? AND deleted_at IS NULL",
        (profile_id, user_id))) or {}


def stored_report(profile_id: str, user_id: str) -> dict:
    row = _profile_row(profile_id, user_id)
    report = row.get("first_run") if isinstance(row, dict) else None
    return report if isinstance(report, dict) and report.get("hecho") else {}


# --------------------------------------------------------- the first run scan

def _face_block(profile: dict, threshold: float) -> dict:
    """What her own photographs say about how well she can be recognised.

    ``embedding_self`` is her worst photograph scored against the mean of the
    others - leave one out, which is exactly the comparison a generated image
    will face - so the margin printed here is the real room above the line for
    THIS person, not a number borrowed from anybody else.
    """
    face = (profile.get("face") or {})
    self_c = face.get("embedding_self") or {}
    n = int(_f(face.get("embedding_n")))
    worst = _f(self_c.get("min"))
    mean = _f(self_c.get("mean"))
    block = {"fotos_con_rostro": n, "peor_foto": round(worst, 4),
             "media": round(mean, 4), "limite": round(threshold, 4),
             "margen": round(worst - threshold, 4), "medible": bool(n >= 2),
             # Measurable and TRUSTWORTHY are two different questions, and the
             # second one is the one that costs her images.  The second account
             # in the multi-user test is the case: five photographs, all five
             # readable, and a worst photograph of 0.3924 against a 0.45 line -
             # so the check works perfectly and will reject a picture that
             # really is her.  Without this flag the report listed "we can
             # check your face is yours" and said nothing about the negative
             # margin two lines above it.
             "fiable": bool(n >= 2 and worst >= threshold)}
    if not face.get("embedding_mean"):
        block["veredicto"] = ("No se ha podido guardar tu firma facial, asi que "
                              "el control de identidad no podra comprobar que "
                              "eres tu. Sube fotos donde se te vea la cara.")
    elif n < 2:
        block["veredicto"] = ("Solo se te ve la cara en %d foto, asi que no hay "
                              "con que comparar. Sube alguna mas." % n)
    elif worst < threshold:
        block["veredicto"] = (
            "Tu foto mas dificil puntua %.2f y el limite es %.2f: alguna imagen "
            "tuya podria rechazarse por parecerse poco a las demas. Anade fotos "
            "con luz e inclinacion parecidas al resto." % (worst, threshold))
    else:
        block["veredicto"] = (
            "El robot te reconoce en las %d fotos. La mas dificil puntua %.2f "
            "sobre un limite de %.2f, asi que hay %.2f de margen antes de que "
            "una imagen tuya se rechace por error."
            % (n, worst, threshold, worst - threshold))
    return block


def _body_block(profile: dict, counts: dict) -> dict:
    """What her photographs allow the silhouette gate to do.

    The gate that really protects her is the paired one: the generated image is
    compared with THE PHOTOGRAPH SHE GENERATED FROM, in head lengths, against a
    limit of HEAD_TOL.  So what matters is not only how many body photographs
    the profile holds but whether the one she picks as a source has a
    measurable body at all - on a closeup that ruler abstains and says so.  The
    stored profile bands are the fallback for when the caller cannot say what
    the image was made from, and they usually cannot reject anything: their own
    band is capped at +/-12% and _aggregate_body marks such a band
    ``gated: False`` on purpose, because on this installation the uncapped
    version rejected 12 of the 14 untouched photographs it could judge.  The
    report says which of the two is available rather than promising a
    protection that will abstain.
    """
    from .verify import HEAD_TOL

    coverage = profile.get("coverage") or {}
    body = profile.get("body") or {}
    metrics = body.get("metrics") if isinstance(body, dict) else {}
    metrics = metrics if isinstance(metrics, dict) else {}
    gated = [k for k, v in metrics.items()
             if isinstance(v, dict) and v.get("gated")]
    usable = int(_f(coverage.get("usable_body_shots")))
    with_body = counts.get("full", 0) + counts.get("half", 0)
    ready = bool(coverage.get("ready_for_body_check"))
    block = {
        "medible": bool(with_body),
        "perfil_listo": ready,
        "fotos_utiles": usable,
        "fotos_con_cuerpo": with_body,
        "cuerpo_entero": counts.get("full", 0),
        "medio_cuerpo": counts.get("half", 0),
        "limite_cabezas": round(HEAD_TOL, 4),
        "medidas_que_pueden_rechazar": sorted(gated),
    }
    if with_body:
        block["veredicto"] = (
            "Tus proporciones se pueden medir en %d de tus %d fotos. Cuando "
            "generes a partir de una de ellas, la imagen se compara con esa "
            "misma foto en alturas de cabeza y se rechaza si te estrecha mas "
            "de un %d%%. Si generas a partir de un primer plano, ese control "
            "se informa pero no rechaza, porque no hay silueta que medir."
            % (with_body, counts.get("full", 0) + counts.get("half", 0)
               + counts.get("closeup", 0) + counts.get("unknown", 0),
               round(HEAD_TOL * 100)))
        if gated:
            block["veredicto"] += (
                " Ademas hay %d medida(s) tuyas con fotos suficientes para "
                "rechazar por si solas." % len(gated))
    else:
        block["veredicto"] = (
            "Con estas fotos NO se pueden medir tus proporciones: todas son "
            "primeros planos. Las imagenes se haran igual y tu cara se "
            "comprobara igual, pero nadie podra avisarte si te estrechan la "
            "silueta. Anade dos fotos de cuerpo entero y esto se activa solo.")
    return block


def _hand_block(rows: list[dict], profile: dict) -> dict:
    """How many of her photographs show a hand the scanner can even read.

    Read on at most HAND_SAMPLE photographs; see the constant for why.  A
    person whose hands never appear is told so instead of being promised a
    check that will silently abstain on every image she pays for.
    """
    sample = rows[:HAND_SAMPLE]
    seen = 0
    worst = 0.0
    for row in sample:
        try:
            read = gallery_mod.reading(row, need=("hands",))
        except Exception as exc:                          # noqa: BLE001
            log.warning("Lectura de manos fallida en %s: %s", row.get("id"), exc)
            continue
        hands = (read.get("hands") or {})
        if hands.get("hands"):
            seen += 1
            worst = max(worst, _f(hands.get("severity")))
    block = {"fotos_leidas": len(sample), "con_manos": seen,
             "peor_lectura": round(worst, 3)}
    if seen:
        block["veredicto"] = (
            "En %d de %d fotos miradas se te ven las manos, asi que el robot "
            "tiene con que comparar cuando revise las manos de una imagen "
            "generada." % (seen, len(sample)))
    else:
        block["veredicto"] = (
            "En ninguna de las %d fotos miradas se te ven las manos. El control "
            "de manos seguira funcionando sobre la imagen generada, pero no "
            "podra compararlas con las tuyas." % len(sample))
    return block


def _skin_block(profile: dict) -> dict:
    skin = profile.get("skin") or {}
    ok = bool(skin.get("ok") if "ok" in skin else skin)
    return {"medible": ok,
            "veredicto": ("Tu tono de piel esta medido y se comprobara en cada "
                          "imagen." if ok else
                          "No se ha podido medir tu tono de piel en estas fotos.")}


def build_first_run(user: dict, profile_id: str, force: bool = False) -> dict:
    """The thorough reading of one person's photographs, done once.

    Called on the FIRST generation for a profile (and from the onboarding
    screen, so she can see it before she even picks a photograph).  It reuses
    the parts that already exist rather than measuring anything twice:
    profile.build_profile does the per photograph analysis and the aggregation,
    gallery.choose_references picks the trio by the same rule the paid path
    uses, and the readings are cached per photograph so the second run pays
    nothing.

    The result is stored on the profile row, so what she was shown before
    spending is the same text a support question can be answered from later.
    """
    user_id = str(user.get("id") or "")
    prof = _profile_row(profile_id, user_id)
    if not prof:
        raise ValueError("Ese perfil no existe.")

    started = time.perf_counter()
    rows = _originals(user_id, profile_id)
    counts = _by_shot(rows)
    n = len(rows)

    # The cache is keyed on the photograph COUNT, not merely on existing.  A
    # report that was true of five photographs is not a report about the nine
    # she has now, and the margin it prints is the number she decides to spend
    # on: uploading more photographs has to change it.
    if not force:
        cached = prof.get("first_run")
        if isinstance(cached, dict) and cached.get("hecho")                 and int(_f(cached.get("fotos"))) == n:
            return cached
    if n < MIN_PHOTOS:
        raise PermissionError(blocking_reason(user_id, profile_id))

    # 1. MEASURE HER.  Rebuilt whenever the profile has never been built or
    #    photographs have been added since, because a report about a stale
    #    profile would be a report about photographs she has already replaced.
    built_now = False
    if force or not (prof.get("face") or {}).get("embedding_mean") \
            or int(_f(prof.get("n_sources"))) != n:
        built = profile_mod.build_profile([r["path"] for r in rows],
                                          str(prof.get("person_name") or "Yo"))
        store_profile(profile_id, built)
        db.execute("UPDATE originals SET profile_id=? WHERE user_id=? "
                   "AND profile_id IS NULL", (profile_id, user_id))
        prof = _profile_row(profile_id, user_id)
        built_now = True

    thresholds = prof.get("thresholds") or {}
    threshold = _f(thresholds.get("face_embed_min"),
                   _f(profile_mod.DEFAULT_THRESHOLDS.get("face_embed_min"), 0.45))

    # 2. CHOOSE THE TRIO the paid path will send, by the same measured rule.
    try:
        picked = gallery_mod.choose_references(prof, REFERENCE_COUNT)
    except Exception as exc:                              # noqa: BLE001
        log.warning("Eleccion de referencias fallida: %s", exc)
        picked = {"paths": [], "reason": "", "detail": {}}

    face = _face_block(prof, threshold)
    body = _body_block(prof, counts)
    hands = _hand_block(rows, prof)
    skin = _skin_block(prof)

    can: list[str] = []
    cannot: list[str] = []
    avisos: list[str] = []
    (can if face["medible"] else cannot).append(
        "Que la cara de cada imagen es la tuya.")
    if face["medible"] and not face["fiable"]:
        avisos.append(
            "Tus fotos se parecen poco entre si: la mas dificil puntua %.2f y "
            "el limite para aprobar una imagen es %.2f. El control funcionara, "
            "pero puede rechazar imagenes que de verdad eres tu y eso gasta "
            "intentos. Sube 3 o 4 fotos mas, tomadas de frente y con luz "
            "parecida, y este margen sube solo."
            % (face["peor_foto"], face["limite"]))
    (can if body["medible"] else cannot).append(
        "Que nadie te cambia las proporciones del cuerpo.")
    (can if skin["medible"] else cannot).append("Que tu tono de piel no cambia.")
    (can if hands["con_manos"] else cannot).append(
        "Comparar las manos generadas con las tuyas.")
    can.append("Que la imagen no tenga fallos anatomicos ni este emborronada: "
               "eso se mide sobre la imagen generada y no necesita tus fotos.")

    refs = list(picked.get("paths") or [])
    by_path = {str(r.get("path")): r for r in rows}
    ref_names = [str((by_path.get(p) or {}).get("filename") or p) for p in refs]

    report = {
        "hecho": True,
        "fecha": db.now(),
        "persona": str(prof.get("person_name") or ""),
        "perfil_id": profile_id,
        "fotos": n,
        "por_plano": counts,
        "minimo": MIN_PHOTOS,
        "recomendado": RECOMMENDED_PHOTOS,
        "perfil_reconstruido": built_now,
        "rostro": face,
        "cuerpo": body,
        "manos": hands,
        "piel": skin,
        "referencias": {
            "n": len(refs),
            "fotos": ref_names,
            "motivo": str(picked.get("reason") or ""),
            "detalle": picked.get("detail") or {},
        },
        "se_puede_comprobar": can,
        "no_se_puede_comprobar": cannot,
        "avisos": avisos,
        "consejos": [] if body["medible"] else list(ADVICE_ES),
        "segundos": round(time.perf_counter() - started, 1),
    }
    if len(refs) < REFERENCE_COUNT:
        avisos.append(
            "Solo %d de tus fotos sirven como referencia y lo ideal son %d: con "
            "%d el robot te dibuja mas parecida y la imagen cuesta menos."
            % (len(refs), REFERENCE_COUNT, REFERENCE_COUNT))
    resumen = ("Se han mirado tus %d fotos (%s). %s %s"
               % (n,
                  ", ".join("%d %s" % (counts[k], SHOT_ES[k])
                            for k in ("full", "half", "closeup") if counts[k]),
                  face["veredicto"], body["veredicto"]))
    report["resumen"] = resumen

    db.execute("UPDATE profiles SET first_run_json=?, updated_at=? WHERE id=? "
               "AND user_id=?",
               (db.dumps(report), db.now(), profile_id, user_id))
    db.audit("profile.first_run", user_id, profile_id=profile_id, fotos=n,
             segundos=report["segundos"])
    return report
