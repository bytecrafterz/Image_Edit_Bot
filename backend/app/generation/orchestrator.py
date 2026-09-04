"""The robot.

Everything else in this package is a component; this file is the thing the
client is actually buying.  It does, without supervision, the loop she performs
by hand today:

    analizar -> decidir que se conserva -> escribir el prompt -> elegir el modelo
    -> comprobar el presupuesto -> generar -> revisar -> reparar solo lo roto
    -> reintentar con parametros corregidos -> aceptar -> aprender

Four rules shape the code:

* **Nothing is spent silently.**  ``prepare_run`` costs nothing and returns an
  estimate; money only moves after the user presses the button, and every call
  reserves its price through ``billing.reserve`` first - the same gate as
  ``can_spend``, held so that calls in flight cannot promise the same dollar
  twice.  A refusal stops the whole run at once and writes an alert, because
  carrying on would burn her balance failing.
* **A rejection must be explainable.**  Every attempt stores the prompt, the
  model, the cost, the verdict numbers and the reason, so ``build_report`` can
  show her exactly why an image was thrown away.
* **One bad variant is not a failed run.**  Each variant is wrapped; a crash
  becomes an error attempt row and the loop continues.
* **Several at a time, one purse.**  Nearly all of a run's wall clock is the
  provider's queue, so the variants wait in it side by side instead of one
  after another; the money is reserved before each call and settled after it,
  which is what keeps a shared balance exact while several calls are open.
"""
from __future__ import annotations

import logging
import os
import threading
import time
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from pathlib import Path

from .. import db
from ..analysis import loader
from ..catalog import options as options_mod
from ..catalog import styles as styles_mod
from ..config import PREVIEW_DIR, OUTPUT_DIR, SETTINGS
from ..identity import gallery as gallery_mod
from ..identity import onboarding as onboarding_mod
from ..identity import verify as verify_mod
from ..safety import consent as consent_mod
from ..safety import guard as guard_mod
from ..services import billing, jobs
from . import correct as correct_mod
from . import learning, planner, prompt as prompt_mod, repair as repair_mod
from . import protect as protect_mod
from . import retouch as retouch_mod
from . import router as router_mod

log = logging.getLogger("photorobot.robot")

_cancel_lock = threading.Lock()
_cancelled: set[str] = set()
# Variants that run side by side both append to runs.plan_json, which is a
# read-modify-write of one column: without this lock one of the notes is lost.
_notes_lock = threading.Lock()


# ------------------------------------------------------------------- helpers

def _set(run_id: str, **fields) -> None:
    if not fields:
        return
    cols = ", ".join(f"{k}=?" for k in fields)
    db.execute(f"UPDATE runs SET {cols} WHERE id=?",
               (*fields.values(), run_id))


def _stage(run_id: str, text: str, progress: float | None = None) -> None:
    if progress is None:
        _set(run_id, stage=text)
    else:
        _set(run_id, stage=text, progress=max(0.0, min(1.0, float(progress))))


def cancel(run_id: str) -> bool:
    with _cancel_lock:
        _cancelled.add(run_id)
    return True


def _is_cancelled(run_id: str) -> bool:
    with _cancel_lock:
        if run_id in _cancelled:
            return True
    return jobs.is_cancelled(run_id)


def _clear_cancel(run_id: str) -> None:
    with _cancel_lock:
        _cancelled.discard(run_id)
    jobs.clear_cancel(run_id)


def _default_provider(user_id: str) -> str | None:
    """The engine the user pinned in Ajustes, or None for 'decide tu'.

    The read lives in the router, next to the code that acts on it, because the
    balance page has to price the same engine this run will use: two readers of
    one setting is how a screen ends up quoting an engine that never runs.
    """
    return router_mod.pinned_provider(user_id)


def _is_generative_provider(provider) -> bool:
    """Does this engine invent pixels, or only transform the photograph?

    ``verify`` weighs a measured difference in her proportions by what the
    engine could physically have done, and it asks the caller because only the
    caller knows which engine ran.  The answer is read from the provider's own
    capabilities and never from its name, so a new engine is classified for
    free and anything written before the flag existed keeps the strict default.
    """
    try:
        caps = provider.capabilities()
    except Exception:                                     # noqa: BLE001
        return True
    return bool(getattr(caps, "generative", True))


TEXTURE_NOTE = ("Se devolvio la textura real de la piel: el motor la habia "
                "suavizado y se ha vuelto a aplicar la de su foto.")

# How many photographs of her travel with one generation.  Kontext multi takes
# three images in ``image_urls`` and charges the same 0.040 USD as the single
# image endpoint at preview - and HALF of identity_max's 0.080 USD at high and
# max - so the pooled reference is free or cheaper, never dearer.  The
# photograph being edited is one of the three; the other two are chosen by
# identity/gallery.py to go with it.  The number itself is defined in the
# router, where the price that depends on it is computed.
REFERENCE_COUNT = router_mod.REFERENCE_COUNT

# JPEG quality for an image composed from her photograph and a repainted zone.
# The pixel array outside the mask is identical to her file - 0 of 5 554 727
# protected pixels differ on IMG_7871 - and this is the number that decides how
# much of that survives the encoder: measured on that same composite, the
# protected region comes back off disk with a mean error of 1.14/255 at q92,
# 0.69 at q96 and 0.10 at q100, and identity_face reads 0.7918, 0.8006 and
# 0.8044 against 0.8177 on her own photograph.  100 costs 3.5 MB a preview
# against 2.2 MB, which is the cheapest thing in this whole product.
COMPOSE_QUALITY = 100

# Defects the free correction path owns.  A paid repaint must not be bought for
# any of them: a hand repainted by the fill endpoint measured 0.000 severity in
# 6 of 6 rehearsals because the hand had been deleted, a face is a different
# woman and cannot be patched at all, and smoothed skin is already given back
# from her own photograph before anything is verified.
FREE_DEFECTS = ("hand_malformed", "face_distorted", "oversmoothed_skin")

# When is a second paid attempt worth buying?  Measured on run_16e276b5: three
# paid attempts on one variant scored identity 0.3507, 0.3753 and 0.3643 - a
# spread of 0.0246 around a line at 0.45 - and lowering the strength did not
# move it.  That is not luck, it is the engine drawing a different person, and
# every further attempt is 0.040 USD for the same answer.  So when one check
# fails twice with readings this close, the variant stops.  The bands are twice
# the scatter measured on each ruler, which is the least that cannot be
# mistaken for noise.
STOP_NOISE: dict[str, float] = {
    "identity_face": 0.05,
    "body_proportions": 0.02,
    "skin_tone": 0.02,
    "anatomy": 0.05,
    "quality": 0.05,
}


def _smoothed_skin(image_path: str) -> str:
    """Which detector, if any, says the engine sanded her skin.  '' if none.

    Both detectors compare facial skin against the rest of the skin in the
    same frame, which catches a beauty filter and is blind to a model that
    smooths the whole picture evenly - and that is what these engines do: on
    the client's own samples the fine band is 40% down and neither of them
    fires.  So they are read as a positive signal and never as a veto.  The
    question "is anything actually missing" can only be answered against her
    photograph, and it is, inside ``restore_skin_texture``, which measures the
    deficit and changes nothing at all when there is none.
    """
    from ..analysis import (anomaly as anomaly_mod, face as face_mod,
                            pose as pose_mod, quality as quality_mod,
                            segment as segment_mod)
    try:
        img = loader.load_image(image_path, max_side=1600)
        face_d = face_mod.detect_face(img)
        qual = quality_mod.assess_quality(img, image_path, face_d)
        if qual.get("beauty_filter_suspected"):
            return "filtro de belleza"
        if qual.get("beauty_ratio") is not None:
            # It measured and said no.  The anomaly scan reads the same
            # comparison, so paying for a pose, a segmentation and a full
            # scan to be told the same thing would only slow every image
            # down; verify_image runs that scan afterwards regardless.
            return ""
        pose_d = pose_mod.detect_pose(img)
        regions = segment_mod.region_masks(img, pose_d, None) or {}
        scan = anomaly_mod.scan_anomalies(img, pose_d, face_d, regions)
        for defect in (scan.get("defects") or []):
            if defect.get("type") == "oversmoothed_skin":
                return "oversmoothed_skin"
    except Exception as exc:                              # noqa: BLE001
        log.debug("Deteccion de suavizado fallida: %s", exc)
    return ""


SKIN_MAX_PROBES = 6


def _skin_candidates(profile: dict, image_path: str, source_path: str) -> list:
    """Her photographs, in the order the transfer should be tried.

    The run's source photograph is NOT privileged.  Measured over 27 generated
    frames, the fixed source succeeds on 11 of 26 (42%) because
    ``restore_skin_texture`` reads both files at full resolution and her
    3024x4032 photograph against a 1024 px render gives a warp scale of 0.16
    against retouch.SCALE_MIN 0.20 - the very refusal this rehearsal printed on
    every image before this function existed.  Trying her photographs in
    ascending alignment residual and keeping the first the module itself
    accepts reaches 19 of 27 (70%), median 3 probes, and costs nothing but a
    few seconds of local work.  The source is in that ordering like any other
    photograph of hers, so nothing is lost by not going first.
    """
    ordered: list[str] = []
    try:
        donor = gallery_mod.choose_donor(profile, image_path, "skin")
        ordered = [p for p in ([donor.get("path")] +
                               list(donor.get("alternatives") or [])) if p]
    except Exception as exc:                              # noqa: BLE001
        log.warning("Eleccion de donante de piel fallida: %s", exc)
    if source_path and source_path not in ordered:
        # A profile that could not be read still deserves the old behaviour.
        ordered.append(source_path)
    return ordered[:SKIN_MAX_PROBES]


def _gate_texture_loss(image_path: str, source_path: str) -> float | None:
    """The grain loss the anatomy gate will read on this file, or None.

    ``restore_skin_texture`` grades itself with ``gain`` - the fine band it
    added over the fine band it found, on the pixels it touched.  The gate that
    decides whether the image is delivered or thrown away reads something else:
    ``verify._texture_loss``, her face against the same face in her own
    photograph, both brought to one width.  The two can disagree, so the only
    honest way to know whether a free correction helped is to ask the ruler
    that judges it.
    """
    if not source_path:
        return None
    from ..analysis import face as face_mod          # as _smoothed_skin does
    try:
        img = loader.load_image(image_path, verify_mod.VERIFY_MAX_SIDE)
        face_d = face_mod.detect_face(img)
        band = verify_mod._texture_loss(img, face_d,
                                        {"source_path": str(source_path)})
    except Exception as exc:                              # noqa: BLE001
        log.debug("Medida del grano fallida: %s", exc)
        return None
    return float(band.get("loss")) if band.get("ok") else None


def _restore_texture(run_id: str, image_path: str, source_path: str,
                     profile: dict | None = None) -> dict:
    """Put her own skin grain back on a generated file, in place.

    Returns what happened, for the attempt row and the ficha.  It never raises
    and it never loses the picture: the module writes only when the transfer
    succeeded, and the write itself is atomic, so a failure of any kind leaves
    exactly the file the provider returned - already paid for.  What is new is
    that a refusal is not the end: her other photographs are tried in the
    measured order until one of them is accepted by the module's own gates.

    And one more gate, because the module's own numbers were not enough.  The
    transfer rewrites the whole paid JPEG, and one more JPEG generation moves
    the gate's own reading by up to 0.03 of severity in either direction -
    measured 2026-09-04 by re-encoding one untouched file at several
    qualities: 0.387 at q90, 0.424 at q95, 0.454 at q96, 0.430 at q98, 0.423 at
    q100, and 0.424 through a lossless PNG, which is the file itself.  On
    run_713536e2/v1_a1 that cost more than the transfer gave: a paid image the
    robot ACCEPTED untouched (anatomy 0.310, grain loss 0.132) came back at
    0.451 after a correction that reported gain x1.13, failed the smoothing
    gate at 0.45 and would have bought another 0.040 USD attempt to replace an
    image that was already good.  Raising the save quality only moves the luck
    around - q96 measured better than q98 on 5 of the 9 files where the gate
    could read at all - so the fix is not a constant but a check: keep the
    corrected bytes only when the ruler that decides says the grain really
    improved, and otherwise put the provider's own bytes back and try the next
    photograph.  Measured over the twelve frames on disk the transfer treats:
    ten keep their correction with the identical gain (x1.21..x1.55), the
    regression above disappears, and the one transfer that reported x1.01 for
    a change of 4/255 on 1410 pixels is reverted instead of announced.
    """
    signal = _smoothed_skin(image_path)
    candidates = _skin_candidates(profile or {}, image_path, source_path)
    before_loss = _gate_texture_loss(image_path, source_path)
    try:
        paid_bytes = Path(image_path).read_bytes()
    except Exception as exc:                              # noqa: BLE001
        log.warning("No se pudo leer el archivo pagado: %s", exc)
        paid_bytes = b""
    got: dict = {}
    tried = 0
    reverted = 0
    revert_reason = ""
    for candidate in candidates:
        tried += 1
        try:
            got = retouch_mod.restore_skin_texture(image_path, candidate,
                                                   image_path)
        except Exception as exc:                          # noqa: BLE001
            log.warning("Restauracion de textura fallida: %s", exc)
            return {"aplicada": False, "motivo": str(exc)[:120],
                    "senal": signal, "intentos": tried}
        if got.get("ok"):
            after_loss = _gate_texture_loss(image_path, source_path)
            if (paid_bytes and before_loss is not None
                    and after_loss is not None and after_loss > before_loss):
                # The paid file measured better before we touched it.  Put it
                # back byte for byte and keep looking; nothing is spent either
                # way, and an image that was already good must not be lost to
                # a correction it did not need.  Written beside the file and
                # moved into place, like every other write on a paid image: a
                # half-written restore would destroy the very thing this branch
                # exists to protect.
                spare = Path(image_path).with_suffix(".pagada.tmp")
                spare.write_bytes(paid_bytes)
                os.replace(str(spare), image_path)
                reverted += 1
                revert_reason = ("tu foto %s no mejoraba el grano medido "
                                 "(%.3f -> %.3f): se deja la imagen tal como "
                                 "vino" % (Path(candidate).name,
                                           before_loss, after_loss))
                got = {"ok": False, "reason": revert_reason}
                continue
            break
        # "The image already keeps your skin texture" is an answer about the
        # IMAGE, not about this donor, so there is nothing to be gained by
        # asking another photograph the same question.
        if "ya conserva" in str(got.get("reason") or ""):
            break

    # A revert is the reason the paid file is the one she gets, so it wins the
    # sentence over whatever the last photograph in the queue happened to say.
    applied = bool(got.get("ok"))
    if applied:
        motivo = str(got.get("reason") or "")
    else:
        motivo = str(revert_reason or got.get("reason")
                     or "sin fotos con las que comparar la piel")
    record = {"aplicada": applied,
              "motivo": motivo[:160],
              "senal": signal,
              "ganancia": round(float(got.get("gain") or 1.0), 3),
              "zonas": int(got.get("regions") or 0),
              "intentos": tried,
              "revertidas": reverted,
              "grano_antes": None if before_loss is None else round(before_loss, 4),
              "foto": Path(candidates[tried - 1]).name if tried else ""}
    if got.get("ok"):
        _plan_note(run_id, TEXTURE_NOTE)
        log.info("Textura devuelta (x%.2f) en %s desde %s",
                 record["ganancia"], image_path, record["foto"])
    return record


def _failed_checks(verdict: dict) -> dict[str, float]:
    """Which checks failed and what they read, so a repeat can be recognised."""
    out: dict[str, float] = {}
    for check in (verdict.get("checks") or []):
        if not check.get("passed", True):
            try:
                out[str(check.get("name"))] = float(check.get("value") or 0.0)
            except (TypeError, ValueError):
                out[str(check.get("name"))] = 0.0
    return out


def _repeat_failure(history: dict, verdict: dict) -> str:
    """The reason to stop buying attempts, or "".

    A check that fails twice with the same reading is not bad luck; it is the
    engine's answer to this request.  Recorded per check, because a run that
    fails identity once and anatomy once has learnt two different things and
    deserves its retry.
    """
    for name, value in _failed_checks(verdict).items():
        seen = history.setdefault(name, [])
        band = STOP_NOISE.get(name, 0.05)
        for before in seen:
            if abs(before - value) <= band:
                return ("%s ha fallado dos veces con el mismo resultado (%.3f y "
                        "%.3f): no se compra otro intento, el motor esta dando "
                        "siempre la misma respuesta."
                        % (verify_mod.CHECK_ES.get(name, name), before, value))
        seen.append(value)
    return ""


def _correct_free(run_id: str, image_path: str, profile: dict, brief: dict,
                  verdict: dict, defects: list[dict]) -> tuple[dict, list, list]:
    """Try to fix what failed WITHOUT buying anything, then measure again.

    The order is the order of the money already spent: the image exists, it is
    paid for, and the only question left is whether her own photographs can
    correct it.  Each correction refuses on its own numbers - see
    generation/correct.py - so what comes back here is either a better file or
    an untouched one with a sentence explaining why it was left alone.  The
    verdict is only recomputed when a file really changed, because verifying
    the same bytes twice would only cost time and could not change the answer.
    """
    failing = set(_failed_checks(verdict))
    log: list[dict] = []
    notes: list[str] = []
    changed = False
    if not failing:
        return verdict, log, notes

    jobs_to_do: list = []
    if "identity_face" in failing:
        jobs_to_do.append(("rostro",
                           lambda: correct_mod.restore_face(image_path, profile,
                                                            brief)))
    if "body_proportions" in failing:
        jobs_to_do.append(("proporciones",
                           lambda: correct_mod.restore_body(image_path, profile,
                                                            brief)))
    if "anatomy" in failing and any(
            d.get("type") == "hand_malformed" for d in (defects or [])):
        jobs_to_do.append(("manos",
                           lambda: correct_mod.fix_hands(image_path, profile,
                                                         defects, brief)))
    for label, run in jobs_to_do:
        try:
            got = run()
        except Exception as exc:                          # noqa: BLE001
            log.append({"que": label, "aplicada": False,
                        "motivo": str(exc)[:160]})
            logging.getLogger("photorobot.robot").warning(
                "Correccion de %s fallida: %s", label, exc)
            continue
        log.append({"que": label, "aplicada": bool(got.get("ok")),
                    "motivo": str(got.get("reason") or "")[:200]})
        if got.get("ok"):
            changed = True
            note = str(got.get("nota") or got.get("reason") or "")
            if note:
                notes.append(note)
                _plan_note(run_id, note)
        else:
            log_it = ("No se pudo corregir %s sin gastar mas: %s"
                      % (label, got.get("reason")))
            _plan_note(run_id, log_it)
    if changed:
        verdict = verify_mod.verify_image(image_path, profile, brief)
    return verdict, log, notes


def _plan_note(run_id: str, note: str) -> None:
    """Keep a routing warning on the run, where notes already live.

    ``runs.plan_json`` has carried a ``notes`` list since the planner wrote
    it and the estimate screen already renders that list, so this needs no
    migration and no new column: build_report reads it back as ``avisos``.
    """
    if not note:
        return
    with _notes_lock:
        row = db.q1("SELECT plan_json FROM runs WHERE id=?", (run_id,))
        if not row:
            return
        plan = db.loads(row["plan_json"], {}) or {}
        if not isinstance(plan, dict):
            return
        notes = [n for n in (plan.get("notes") or []) if isinstance(n, str)]
        if note in notes:
            return
        notes.append(note)
        plan["notes"] = notes
        db.execute("UPDATE runs SET plan_json=? WHERE id=?",
                   (db.dumps(plan), run_id))


def _profile_for(user: dict, profile_id: str | None) -> dict:
    row = None
    if profile_id:
        row = db.q1("SELECT * FROM profiles WHERE id=? AND user_id=? "
                    "AND deleted_at IS NULL", (profile_id, user["id"]))
    if row is None:
        row = db.q1("SELECT * FROM profiles WHERE user_id=? AND deleted_at IS NULL "
                    "ORDER BY is_default DESC, updated_at DESC LIMIT 1",
                    (user["id"],))
    profile = db.row_to_dict(row) or {}
    if profile:
        # profiles store each block in its own column; verify wants one dict
        profile.setdefault("thresholds", profile.get("thresholds") or {})
    return profile


def analyse_original(original: dict) -> dict:
    """Cached full reading of the source photograph."""
    from ..analysis import (anomaly, face as face_mod, pose as pose_mod,
                            quality as quality_mod, segment, shot as shot_mod,
                            skin as skin_mod, body as body_mod)

    cached = db.loads(original.get("analysis"), None)
    if isinstance(cached, dict) and cached.get("ok"):
        return cached

    img = loader.load_image(original["path"], max_side=1600)
    pose_d = pose_mod.detect_pose(img)
    face_d = face_mod.detect_face(img)
    seg = segment.person_mask(img)
    mask = seg.get("mask") if seg.get("ok") else None
    regions = segment.region_masks(img, pose_d, mask) or {}
    # The face goes in because the head-length body ruler hangs its rows from
    # the chin of the mesh: this reading is cached and handed to the gate as
    # source_body, and without it that ruler - the only one that survives the
    # engine's reframing - would be missing from every pair and verify would
    # have to measure the photograph a second time to get it.
    body_d = body_mod.measure_body(img, pose_d, mask, face_d)
    report = {
        "ok": True,
        "shot_type": shot_mod.classify_shot(img, pose_d, face_d).get("shot_type"),
        "quality": quality_mod.assess_quality(img, original["path"]),
        "skin": skin_mod.skin_stats(img, pose_d, face_d),
        "body": body_d,
        "has_face": bool(face_d.get("ok")),
        "defects": (anomaly.scan_anomalies(img, pose_d, face_d, regions)
                    .get("defects") or []),
        "width": int(img.shape[1]), "height": int(img.shape[0]),
    }
    db.execute("UPDATE originals SET analysis_json=?, shot_type=? WHERE id=?",
               (db.dumps(report), report["shot_type"], original["id"]))
    return report


def _brief_from(analysis: dict, original: dict, choices: dict) -> dict:
    from ..providers import registry

    brief = {
        "shot_type": analysis.get("shot_type") or "unknown",
        "source_path": original["path"],
        "source_body": analysis.get("body") or {},
        # And the skin of that same photograph.  identity/verify judges the
        # skin tone against the picture this one was made from, not against the
        # pooled gallery mean: measured over 168 honest changes of her 24
        # photographs the pooled reference false-alarmed 3 times (twice on a
        # lighting change she had asked for) and the paired one never did.  It
        # is already measured and cached here, so handing it over costs
        # nothing and saves verify a second reading of the photograph.
        "source_skin": analysis.get("skin") or {},
        "expects_face": bool(analysis.get("has_face")),
        "choices": choices,
    }
    vision = registry.get_vision_provider("auto")
    if vision is not None:
        try:
            described = vision.describe_photo(original["path"],
                                              {"shot_type": brief["shot_type"]})
            for key in ("subject", "clothing", "hair", "expression", "pose",
                        "setting", "lighting", "camera", "colors", "preserve"):
                if described.get(key):
                    brief[key] = described[key]
            brief["vision_cost_usd"] = float(described.get("cost_usd") or 0.0)
            brief["vision_provider"] = vision.name
        except Exception as exc:                          # noqa: BLE001
            log.warning("Lectura de la foto fallida (%s): %s", vision.name, exc)
    return brief


# ----------------------------------------------------------------- preparing

def prepare_run(user: dict, original_id: str, choices: dict, n_previews: int,
                quality: str, profile_id: str | None = None,
                style_key: str | None = None) -> dict:
    """Plan and price a run.  Spends nothing."""
    original = db.row_to_dict(db.q1(
        "SELECT * FROM originals WHERE id=? AND user_id=? AND deleted_at IS NULL",
        (original_id, user["id"])))
    if not original:
        raise ValueError("Esa foto no existe o fue eliminada.")

    # ENOUGH REAL PHOTOGRAPHS FIRST, before anything is measured or priced.
    # The whole product rests on comparing what comes back against photographs
    # of the person, and identity/onboarding.py measures what a gallery of each
    # size is worth: with a single photograph 12.5% of possible profiles reject
    # one of her OWN photographs, and the margin against other women is 0.2965
    # where five photographs give 0.4290.  Refusing here rather than at the
    # button is deliberate - it costs her nothing and it is the only moment at
    # which the answer is still "upload four more" instead of "that image was
    # discarded".
    blocked = onboarding_mod.blocking_reason(user["id"])
    if blocked:
        raise PermissionError(blocked)

    analysis = analyse_original(original)
    shot = analysis.get("shot_type") or "unknown"
    profile = _profile_for(user, profile_id)
    warnings: list[str] = []

    # THE FIRST GENERATION FOR A PERSON READS HER PHOTOGRAPHS PROPERLY.  The
    # client asked for a detailed analysis on the first composite, and this is
    # the only place that is both before the money and after she has a profile.
    # It runs once: the report is stored on the profile row and every later
    # estimate reads it back, so the cost - profile.build_profile over her
    # photographs plus a hand reading on at most eight of them - is paid on the
    # first estimate and never again.  A failure here must not block a run, so
    # it is logged and the estimate goes on without the section.
    first_run: dict = {}
    if profile:
        first_run = onboarding_mod.stored_report(profile["id"], user["id"])
        if not first_run:
            try:
                first_run = onboarding_mod.build_first_run(user, profile["id"])
            except Exception as exc:                      # noqa: BLE001
                log.warning("Analisis inicial fallido: %s", exc)
                first_run = {}
            else:
                profile = _profile_for(user, profile["id"])

    if not profile:
        warnings.append("Todavia no has creado tu perfil: no se podran comprobar "
                        "tus proporciones. Ve a Mis fotos y pulsa Crear perfil.")
    else:
        problem = consent_mod.consent_problem(profile["id"])
        if problem:
            raise PermissionError(problem)
        coverage = profile.get("coverage") or {}
        if not coverage.get("ready_for_body_check"):
            warnings.append("Tu perfil aun no tiene fotos de cuerpo entero "
                            "suficientes, asi que el control de proporciones "
                            "sera limitado.")

    # The guard runs on what the user ASKED FOR, before the catalogue filter.
    # Filtering first would quietly drop a blocked value as "unknown option" and
    # let the request through sanitised, which is the wrong answer twice over:
    # the user is not told, and the refusal never happens.
    verdict = guard_mod.check_request({"shot_type": shot, **analysis}, choices,
                                      profile, user)
    if not verdict["allowed"]:
        raise PermissionError(verdict["reason"])

    clean = options_mod.resolve_choices(choices, shot)
    verdict = guard_mod.check_request({"shot_type": shot, **analysis}, clean,
                                      profile, user)
    if not verdict["allowed"]:
        raise PermissionError(verdict["reason"])

    style = (styles_mod.get_style(style_key) if style_key
             else styles_mod.default_style(shot))
    n = max(1, min(int(n_previews or 6), SETTINGS.limits.max_previews_per_run))

    run_id = db.new_id("run")
    weights = learning.get_weights(user["id"])
    brief = _brief_from(analysis, original, clean)
    plan = planner.plan_run(brief, clean, n, profile, style, weights)
    plan["run_id"] = run_id
    plan = learning.apply_learning(plan, weights)

    # The run obeys the provider pinned in Ajustes, so the estimate has to
    # price that one too: otherwise the screen says "sin coste" with the free
    # engine and the run charges the pinned one, which is the surprise bill
    # this module exists to prevent.
    pinned = _default_provider(user["id"])
    if pinned:
        plan["provider"] = pinned
    # The shape of her photograph travels with the plan, because the estimate
    # has to build the same request the run will send and the size is part of
    # it: her 2316x3088 is a 3:4, and asking any engine for a square instead is
    # what cropped or squeezed her body in every paid image so far.
    plan["source_path"] = original["path"]
    plan["source_size"] = [int(original.get("width") or 0),
                           int(original.get("height") or 0)]

    # Three photographs of her, not one.  The engine was being shown a single
    # picture and, at high quality, was even being sent that same picture twice
    # - as the image to edit and as its own "identity reference" - which is not
    # evidence about anybody.  The two companions are chosen by measurement
    # (identity/gallery.py) to cover her worst photograph and to differ in
    # framing and face size, and they are picked here, at planning time, so the
    # price on the screen is the price of the call that will really be sent:
    # with references fal bills identity_multi at 0.040 USD instead of
    # identity_max at 0.080 USD on the top tiers.
    plan["reference_paths"] = []
    if profile:
        try:
            picked = gallery_mod.choose_references(
                profile, REFERENCE_COUNT, must_include=original["path"])
        except Exception as exc:                          # noqa: BLE001
            log.warning("Eleccion de referencias fallida: %s", exc)
            picked = {"paths": [], "reason": ""}
        plan["reference_paths"] = [p for p in (picked.get("paths") or [])
                                   if p and p != original["path"]]
        plan["reference_detail"] = picked.get("detail") or {}
        note = str(picked.get("reason") or "")
        if note:
            notes = [n for n in (plan.get("notes") or []) if isinstance(n, str)]
            if note not in notes:
                notes.append(note)
            plan["notes"] = notes

    # WILL HER FACE BE GENERATED?  Decided here, on the option groups alone,
    # because the answer has to be on the screen with the price and not
    # discovered afterwards in a report.  It is the same call _run_variant
    # makes, from the same module, so the sentence she reads is the decision
    # the run takes.  A plan whose variants disagree - because the robot varied
    # the pose by itself on some of them - says how many.
    shields = [protect_mod.plan_mask((v or {}).get("choices") or {})
               for v in (plan.get("variants") or [{}])]
    safe_n = sum(1 for s in shields if s.get("safe"))
    plan["face_protected"] = [bool(s.get("safe")) for s in shields]
    if safe_n == len(shields):
        warnings.append("Tu rostro NO se va a generar en ninguna de estas "
                        "imagenes: se repinta solo la zona que cambias y tu "
                        "cara y tus manos se copian de tu propia foto.")
    elif safe_n:
        warnings.append("En %d de las %d imagenes tu rostro se copia de tu "
                        "foto; en las otras %d hay que volver a dibujarlo. %s"
                        % (safe_n, len(shields), len(shields) - safe_n,
                           next(s["reason"] for s in shields if not s.get("safe"))))
    else:
        warnings.append(next(s["reason"] for s in shields))

    # The same limits the run will obey, read once and handed to the estimate,
    # so the ceiling she is shown is the ceiling _run_variant can reach.
    estimate = router_mod.estimate_run_cost(plan, quality or "preview",
                                            router_mod.user_limits(user["id"]),
                                            user["id"])
    balances = billing.all_balances(user["id"])
    provider_name = estimate.get("provider") or "local"
    if provider_name not in billing.FREE_PROVIDERS:
        bal = balances.get(provider_name, {}).get("balance") or 0.0
        if bal < estimate.get("total_usd", 0.0):
            warnings.append("El saldo de %s (%.2f USD) no cubre esta tirada "
                            "(%.2f USD). Registra una recarga en Ajustes."
                            % (provider_name, bal, estimate["total_usd"]))

    if first_run:
        for line in (first_run.get("no_se_puede_comprobar") or []):
            note = ("En tus fotos no hay con que comprobar esto: %s" % line)
            if note not in warnings:
                warnings.append(note)
        # A warning about her own gallery is worth more here than anywhere
        # else: it is the only screen where the answer is still "upload three
        # more photographs" instead of "that attempt was discarded".
        for line in (first_run.get("avisos") or []):
            if line not in warnings:
                warnings.append(line)

    # The run writes the same warning into the plan notes, but by then the
    # images exist; here it lands on the estimate screen, which is the last
    # moment at which she can still change what she asked for.
    aviso = str(estimate.get("aviso") or "")
    if aviso and aviso not in warnings:
        warnings.append(aviso)

    # Same reasoning for the pixels: 'alta' buys a more faithful model, not a
    # bigger file, and she is told which one she is buying before she pays.
    aviso_res = str(estimate.get("aviso_resolucion") or "")
    if aviso_res and aviso_res not in warnings:
        warnings.append(aviso_res)

    # And the ceiling.  The estimate is what a run costs when it goes well; a
    # run that keeps being rejected pays for every retry and every repainted
    # zone, and on 2026-09-03 one draft variant settled 0.2687 USD against a
    # 0.0084 USD quote.  Both numbers now reach the screen she decides on.
    aviso_coste = str(estimate.get("aviso_coste") or "")
    if aviso_coste and aviso_coste not in warnings:
        warnings.append(aviso_coste)

    # What her own paid images say about the options she has just chosen.  It
    # is the cheapest warning in the product: changing a scene costs nothing,
    # and playa_atardecer came back as somebody else on 8 of 11 paid attempts.
    aviso_opciones = str(estimate.get("aviso_opciones") or "")
    if aviso_opciones and aviso_opciones not in warnings:
        warnings.append(aviso_opciones)

    db.execute(
        "INSERT INTO runs(id,user_id,original_id,profile_id,mode,status,"
        "options_json,plan_json,n_requested,est_cost_usd,created_at) "
        "VALUES(?,?,?,?,'preview','queued',?,?,?,?,?)",
        (run_id, user["id"], original_id, (profile or {}).get("id"),
         db.dumps({"choices": clean, "quality": quality or "preview",
                   "style": style["key"], "brief": _jsonable_brief(brief)}),
         db.dumps(plan), len(plan.get("variants") or []),
         float(estimate.get("total_usd") or 0.0), db.now()),
    )
    db.audit("run.prepared", user["id"], run_id=run_id, n=n, est=estimate)

    return {
        "run_id": run_id,
        "estimate": estimate,
        "plan_summary": {
            "locked": plan.get("locked") or {},
            "varied": plan.get("varied") or [],
            "n_variants": len(plan.get("variants") or []),
            "notes": plan.get("notes") or [],
            "style": {"key": style["key"], "name": style["name_es"]},
        },
        "warnings": warnings,
        "analisis_inicial": first_run or None,
        "analysis": {"shot_type": shot,
                     "quality": (analysis.get("quality") or {}).get("score"),
                     "issues": (analysis.get("quality") or {}).get("issues") or []},
    }


def _jsonable_brief(brief: dict) -> dict:
    """The brief carries numpy-flavoured measurements; keep only what stores."""
    out = {}
    for key, value in brief.items():
        if key == "source_body":
            continue
        if isinstance(value, (str, int, float, bool, list, dict)) or value is None:
            out[key] = value
    return out


# ----------------------------------------------------------------- executing

class _Batch:
    """The variants of one run while they are in flight together.

    Sending them one at a time cost one provider queue per image, and a single
    slow entry in that queue stalled everything behind it.  Sending a few at
    once turns three things that used to be trivially safe into shared state:
    the decision to stop, the count of what has finished, and the sentence the
    phone reads every 1.5 seconds.  They live here behind one lock so the
    workers themselves stay as simple as they were.
    """

    def __init__(self, run_id: str, total: int, label: str, workers: int):
        self.run_id = run_id
        self.total = max(1, int(total))
        self.label = label                    # "Generando" | "Alta calidad"
        self.workers = max(1, int(workers))
        self.serial = self.workers <= 1
        self._lock = threading.Lock()
        self._finished = 0
        self.stop_reason = ""
        self.cancelled = False

    # ------------------------------------------------------------ decisions
    def aborted(self) -> bool:
        """True when nothing further may be sent to a provider.

        Asked before a task is queued, before every provider call and before
        every repair round, so that a cancel or a refusal stops the work that
        has not started yet and not merely the work that was never planned.
        A call already inside the provider is left to finish: it is paid for.
        """
        with self._lock:
            if self.stop_reason or self.cancelled:
                return True
        if _is_cancelled(self.run_id):
            with self._lock:
                self.cancelled = True
            return True
        return False

    def stop(self, reason: str) -> None:
        """Record the first refusal; a later one must not overwrite it."""
        with self._lock:
            if not self.stop_reason:
                self.stop_reason = reason or "Sin saldo suficiente."

    # -------------------------------------------------- what the phone reads
    def opening(self) -> None:
        if not self.serial:
            _stage(self.run_id,
                   "%s: %d imagenes a la vez" % (self.label, self.workers), 0.0)

    def starting(self, position: int) -> None:
        """One at a time still announces the image it is about to make."""
        if self.serial:
            _stage(self.run_id,
                   "%s %d de %d" % (self.label, position + 1, self.total),
                   position / self.total)

    def finished(self) -> None:
        """Progress is what has really finished, in whatever order it did.

        The write stays inside the lock.  Counting under it and writing
        outside it lets two variants that finish together reach the UPDATE in
        the opposite order, and the phone then reads "4 de 6" and, 1.5
        seconds later, "3 de 6" - a bar that walks backwards, which reads as a
        fault in the robot.  One short UPDATE per finished variant is a cheap
        price for a number that only ever grows.
        """
        with self._lock:
            self._finished += 1
            done = self._finished
            _stage(self.run_id,
                   "%s %d de %d" % (self.label, min(done + 1, self.total),
                                    self.total),
                   done / self.total)

    def detail(self, text: str) -> None:
        """Commentary on one image, only when there is one image to talk about.

        With several in flight these lines would overwrite each other every
        1.5 seconds and describe whichever worker wrote last, which reads as a
        glitch rather than as information.
        """
        if self.serial:
            _stage(self.run_id, text)


def _parallel_limit(n_tasks: int) -> int:
    """How many variants of this run may wait on a provider at the same time.

    Bounded by the configured limit and by the work that actually exists, so a
    single image never leaves the calling thread and behaves exactly as it
    always did.  This limit is per run; whole runs are still bounded by
    ``max_concurrent_jobs`` in the jobs service.
    """
    limit = int(getattr(SETTINGS.limits, "max_parallel_generations", 1) or 1)
    return max(1, min(limit, max(1, int(n_tasks))))


def _run_batch(user: dict, run_id: str, variants: list[dict], brief: dict,
               profile: dict, style: dict, quality: str, out_dir: Path,
               original: dict | None, label: str,
               kind: str) -> tuple[list[dict], _Batch]:
    """Run the planned variants, a few at a time, and answer in planned order.

    Completion order is whatever the provider's queue decides; the results
    list is indexed by position, so the accepted images are still counted,
    reported and shown in the order she asked for.  One variant is still one
    variant: a crash inside it becomes an error attempt row and the rest of
    the run carries on.
    """
    batch = _Batch(run_id, len(variants), label, _parallel_limit(len(variants)))
    results: list[dict] = [{} for _ in variants]

    def _one(position: int, variant: dict) -> dict:
        index = int(variant.get("index", position))
        try:
            return _run_variant(user, run_id, variant, brief, profile, style,
                                quality, out_dir, original, batch, kind)
        except Exception as exc:                          # noqa: BLE001
            log.error("Variante %d fallida: %s", index, exc, exc_info=True)
            _record_attempt(run_id, user["id"], index, 1, "", "", "generate",
                            "", "", {}, {}, [], "error", str(exc)[:200], 0.0, 0)
            return {"accepted": False, "cost": 0.0, "attempts": 0,
                    "repaired": 0, "failed": True}
        finally:
            batch.finished()

    batch.opening()
    if batch.serial:
        for position, variant in enumerate(variants):
            if batch.aborted():
                break
            batch.starting(position)
            results[position] = _one(position, variant)
        return results, batch

    # Queued in waves rather than all at once, so a stop or a cancel keeps the
    # variants it has not reached yet from ever being sent.
    queue = list(enumerate(variants))
    pending: dict = {}
    with ThreadPoolExecutor(max_workers=batch.workers,
                            thread_name_prefix="photorobot-var") as pool:
        while queue or pending:
            while queue and len(pending) < batch.workers and not batch.aborted():
                position, variant = queue.pop(0)
                pending[pool.submit(_one, position, variant)] = position
            if not pending:
                break
            done, _ = wait(list(pending), return_when=FIRST_COMPLETED)
            for future in done:
                results[pending.pop(future)] = future.result()
    return results, batch


def _tally(results: list[dict]) -> dict:
    """Add the variants up in planned order, counting what has always counted."""
    out = {"accepted": 0, "rejected": 0, "repaired": 0, "attempts": 0,
           "cost": 0.0}
    for result in results:
        if not result:
            continue
        out["attempts"] += int(result.get("attempts") or 0)
        out["cost"] += float(result.get("cost") or 0.0)
        out["repaired"] += int(result.get("repaired") or 0)
        # A variant that crashed, that never ran, or that hit the money gate is
        # not a rejected image: it was never judged.
        if result.get("stopped") or result.get("aborted") or result.get("failed"):
            continue
        if result.get("accepted"):
            out["accepted"] += 1
        else:
            out["rejected"] += 1
    return out


def _order_images_by_variant(run_id: str) -> None:
    """Show the accepted images in variant order, not in finish order.

    The album and the status screen sort by ``created_at``, which used to be
    the same thing because the variants ran in order.  Now the sixth can come
    back before the second, so the timestamps of this run's own images are
    permuted among themselves: every row keeps a moment this run really
    produced, and the order on screen is the order she planned.
    """
    rows = db.q(
        "SELECT i.id AS id, i.created_at AS created_at, "
        "a.variant_index AS variant_index, a.attempt_no AS attempt_no "
        "FROM images i JOIN attempts a ON a.id = i.attempt_id "
        "WHERE i.run_id=? AND i.deleted_at IS NULL", (run_id,))
    if len(rows) < 2:
        return
    stamps = sorted(float(r["created_at"]) for r in rows)
    planned = sorted(rows, key=lambda r: (int(r["variant_index"]),
                                          int(r["attempt_no"])))
    previous = 0.0
    for stamp, row in zip(stamps, planned):
        # Two images finished inside the same clock tick share a timestamp,
        # and SQLite then breaks the tie by insertion order - which is exactly
        # the finish order this function exists to undo.  A tenth of a
        # millisecond apart is still a moment this run produced, and it is an
        # order.
        stamp = max(stamp, previous + 0.0001)
        previous = stamp
        if float(row["created_at"]) != stamp:
            db.execute("UPDATE images SET created_at=? WHERE id=?",
                       (stamp, row["id"]))


def run_previews(user: dict, run_id: str) -> dict:
    run = db.row_to_dict(db.q1("SELECT * FROM runs WHERE id=? AND user_id=?",
                               (run_id, user["id"])))
    if not run:
        raise ValueError("Ese trabajo no existe.")
    _clear_cancel(run_id)

    opts = run.get("options") or {}
    plan = run.get("plan") or {}
    quality = str(opts.get("quality") or "preview")
    variants = plan.get("variants") or []
    original = db.row_to_dict(db.q1("SELECT * FROM originals WHERE id=?",
                                    (run["original_id"],)))
    if not original:
        _set(run_id, status="failed", error="La foto original ya no existe.",
             finished_at=db.now())
        return {"ok": False, "error": "La foto original ya no existe."}

    analysis = analyse_original(original)
    profile = _profile_for(user, run.get("profile_id"))
    style = styles_mod.get_style(opts.get("style")) or styles_mod.default_style(
        analysis.get("shot_type") or "unknown")
    brief = dict(opts.get("brief") or {})
    brief["source_path"] = original["path"]
    brief["source_body"] = analysis.get("body") or {}
    brief["source_skin"] = analysis.get("skin") or {}
    # The references chosen when the run was priced, so the call that is sent
    # is the call that was quoted; repair.py reads the same list, which is why
    # it lives on the brief and not in a second variable.
    brief["reference_paths"] = [p for p in (plan.get("reference_paths") or [])
                                if p and Path(p).exists()]

    out_dir = PREVIEW_DIR / str(user["id"]) / run_id
    out_dir.mkdir(parents=True, exist_ok=True)

    started = time.time()
    results, batch = _run_batch(user, run_id, variants, brief, profile, style,
                                quality, out_dir, original, "Generando",
                                "preview")
    tally = _tally(results)
    accepted, rejected = tally["accepted"], tally["rejected"]
    repaired, attempts_used = tally["repaired"], tally["attempts"]
    spent = tally["cost"]
    _order_images_by_variant(run_id)

    if batch.stop_reason:
        # The refusal has already written the alert, so the run must carry the
        # reason that produced it - the images that were in flight when it
        # happened are still counted and still paid for.
        _set(run_id, status="stopped_no_balance", error=batch.stop_reason,
             finished_at=db.now(), cost_usd=round(spent, 6),
             n_accepted=accepted, n_rejected=rejected, n_repaired=repaired,
             attempts_used=attempts_used,
             stage="Detenido por falta de saldo")
        return {"ok": False, "stopped": True, "reason": batch.stop_reason,
                "accepted": accepted, "cost_usd": round(spent, 4)}
    if batch.cancelled:
        # Whatever finished before she pressed stop was really produced and
        # really charged, so the run says so instead of reporting zero.
        _set(run_id, status="cancelled", finished_at=db.now(),
             cost_usd=round(spent, 6), n_accepted=accepted,
             n_rejected=rejected, n_repaired=repaired,
             attempts_used=attempts_used, stage="Detenido")
        return {"ok": True, "cancelled": True, "accepted": accepted,
                "cost_usd": round(spent, 4)}

    _set(run_id, status="done", progress=1.0, finished_at=db.now(),
         n_accepted=accepted, n_rejected=rejected, n_repaired=repaired,
         attempts_used=attempts_used, cost_usd=round(spent, 6),
         stage="Listo: %d imagenes" % accepted)
    log.info("Run %s: %d aceptadas, %d descartadas, %.4f USD, %.1fs, %d a la vez",
             run_id, accepted, rejected, spent, time.time() - started,
             batch.workers)
    return {"ok": True, "accepted": accepted, "rejected": rejected,
            "repaired": repaired, "cost_usd": round(spent, 4),
            "seconds": round(time.time() - started, 1)}


def _run_variant(user: dict, run_id: str, variant: dict, brief: dict,
                 profile: dict, style: dict, quality: str, out_dir: Path,
                 original: dict, batch: "_Batch", kind: str = "preview") -> dict:
    """One planned image, including its repairs and retries.

    ``kind`` is what the album will call the result - 'preview' for a preview
    run, 'final' for a high quality render - and it has to be told, not
    guessed: both runs go through this same function, and labelling every
    output 'preview' leaves the album's "Finales" tab permanently empty.

    Retries and repairs stay inside this call on purpose: a variant that needs
    a second attempt must not hold up a different variant that does not.  The
    batch is here for the two things that are not private to one image - the
    decision to stop, and the money that several images are spending at once.
    """
    from ..providers.base import InsufficientBalance, ProviderError

    index = int(variant.get("index", 0))
    choices = variant.get("choices") or {}
    cost = 0.0
    repaired = 0
    attempts = 0
    # Read once per variant, not per attempt: the settings cannot change
    # halfway through a retry loop and one query each is enough.
    prefer = _default_provider(user["id"])
    # Her own limits, not the installation's.  Until 2026-09-04 both numbers
    # below came from SETTINGS.limits, so "0 reintentos, 0 reparaciones" on the
    # Ajustes screen changed nothing and the run still bought up to three
    # attempts and two repaint rounds.  ``plan_run`` prices the estimate off
    # this same reader, so the ceiling on the screen and the money this loop
    # can spend stay one number.
    limits = router_mod.user_limits(user["id"])
    max_retries = int(limits["max_retries"])
    max_repair_rounds = int(limits["max_repair_rounds"])
    params = dict(variant.get("params") or {})
    seed = int(variant.get("seed") or 0)
    extra_negatives: list[str] = []
    # Her other photographs, already chosen and priced when the run was
    # planned.  The photograph being edited is never repeated here: it travels
    # as the source and sending it twice is what the old code did.
    references = [p for p in (brief.get("reference_paths") or [])
                  if p and p != original.get("path") and Path(p).exists()]
    # What each check has read on the attempts that failed, so the loop can
    # stop buying answers the engine has already given.
    seen_failures: dict[str, list[float]] = {}
    # The face shield, built at most once for this variant and reused by every
    # retry.  Empty means "this image is made whole"; ``mask_note`` says why,
    # in her language, whichever way it went.
    mask_path = ""
    mask_note = ""
    mask_cover = 0.0

    for attempt_no in range(1, max_retries + 2):
        # Between attempts, and always before anything reaches a provider: a
        # cancel or another variant's money refusal ends this one here.
        if batch.aborted():
            return {"accepted": False, "cost": cost, "attempts": attempts,
                    "repaired": repaired, "aborted": True}
        attempts += 1
        built = prompt_mod.build_prompt({**brief, "choices": choices}, profile,
                                        style, choices)
        negative = built["negative_prompt"]
        if extra_negatives:
            negative = negative + ", " + ", ".join(sorted(set(extra_negatives)))
        merged = {**built.get("params", {}), **params}

        # Build the image BEFORE choosing who makes it and what it costs.  A
        # provider reads the request to pick its model - source, references,
        # mask, size - so a request built afterwards is a different request,
        # and pricing that one is how the estimate, the reservation and the
        # invoice came to disagree about the same photograph.  Her own shape
        # comes from the original row; a chosen framing overrides it, because
        # a 9:16 story really is a different picture.
        hints = _local_hints(choices)

        # THE FACE, AT GENERATION.  A request that only changes the clothes or
        # the background does not need her face redrawn, and a face that is
        # never redrawn cannot come back as somebody else.  This is the whole
        # of the client's first requirement, and it is not a check - it is
        # arithmetic: generation/protect.py masks the regions the request
        # really touches, keeps her face and her hands out of the mask, and
        # puts her own pixels back outside it at full resolution afterwards.
        # It matters because the repair side cannot do this job: restore_face
        # wrote 0 of 9 failing images, since a face measured 0.29-0.40 against
        # her is not a damaged copy of her face but a different woman's.
        # The mask is built once per variant and reused by every retry: it
        # depends on her photograph and on what she asked for, neither of which
        # changes between attempts, and building it costs about 1.4 s.
        shield = protect_mod.plan_mask(choices)
        if shield.get("safe") and not mask_path and not mask_note:
            mask_path, mask_note, mask_cover = _face_shield(
                original, choices, out_dir, index)
        masked = bool(mask_path)
        request = router_mod.build_request(
            "inpaint" if masked else "generate", quality,
            prompt=built["prompt"], negative_prompt=negative,
            source_path=original["path"],
            source_size=[int(original.get("width") or 0),
                         int(original.get("height") or 0)],
            mask_path=mask_path or None,
            # The fill endpoint takes no reference images at all (see
            # providers/fal MODELS['inpaint']['knobs']), and on this path it
            # needs none: the face it would have to be taught is the face it is
            # forbidden to touch.
            reference_paths=[] if masked else references,
            framing=hints.get("framing") or choices,
            seed=seed + attempt_no - 1,
            strength=float(merged.get("strength", 0.5)),
            guidance=float(merged.get("guidance", 4.0)),
            steps=int(merged.get("steps", 28)),
            identity_weight=float(merged.get("identity_weight", 0.85)),
            extra=hints,
        )

        # The router needs both halves of the question: who she prefers, and
        # what she actually asked to change.  Without the second half a free
        # engine that cannot change clothes or pose wins on price and
        # produces images that could never have satisfied the request.
        provider, model, why = router_mod.choose_provider(
            "inpaint" if masked else "generate", quality, budget_usd=None,
            prefer=prefer, changes=choices, request=request)
        if masked and not _is_generative_provider(provider):
            # The local engine's inpaint fills a hole from the pixels around
            # it: handed the mask of a torso it would sample the wall, not sew
            # a suit.  Masking is only worth anything against an engine that
            # can actually paint the garment, so this falls back to the whole
            # image path before a single cent is committed.
            masked, mask_path = False, ""
            mask_note = ("El motor gratuito no sabe repintar solo una zona, "
                         "asi que esta imagen se hace entera.")
            request = router_mod.build_request(
                "generate", quality,
                prompt=built["prompt"], negative_prompt=negative,
                source_path=original["path"],
                source_size=[int(original.get("width") or 0),
                             int(original.get("height") or 0)],
                reference_paths=references,
                framing=hints.get("framing") or choices,
                seed=seed + attempt_no - 1,
                strength=float(merged.get("strength", 0.5)),
                guidance=float(merged.get("guidance", 4.0)),
                steps=int(merged.get("steps", 28)),
                identity_weight=float(merged.get("identity_weight", 0.85)),
                extra=hints,
            )
            provider, model, why = router_mod.choose_provider(
                "generate", quality, budget_usd=None, prefer=prefer,
                changes=choices, request=request)
        # What was decided about her face, in her language, on the attempt row
        # and on the run's notes - before the image exists, not after.
        merged["rostro"] = {
            "protegido": masked,
            "motivo": (mask_note or shield.get("reason") or ""),
            "zona": round(float(mask_cover), 4),
            "referencias": 0 if masked else len(references),
        }
        unsupported = router_mod.unsupported_changes(provider, choices)
        # Priced on the object that is about to be sent, and on nothing else.
        price = provider.estimate_cost(request)
        # The decision travels with the attempt row - params_json already
        # exists, so the ficha gains it without a migration - and a warning
        # is written once onto the run's plan notes, which the report reads
        # back as 'avisos'.
        merged["router"] = why
        if unsupported:
            _plan_note(run_id, why)

        # The reviewer has to be told which engine made the image: an engine
        # that only composites, recolours and crops cannot have moved a
        # shoulder, so a difference measured on its output is the instrument's
        # scatter and not her body.  Set on a copy, and set unconditionally,
        # so the answer always comes from the provider actually chosen and can
        # never be relaxed by anything stored with the run.
        checked = dict(brief)
        checked["generative"] = _is_generative_provider(provider)

        target = out_dir / f"v{index}_a{attempt_no}.jpg"
        # On the masked path what the engine returns is not the image she gets:
        # it is the repainted half, which is then put back inside the mask over
        # her own photograph.  Both files are kept, because the raw answer is
        # the only evidence of what was actually bought.
        painted_target = (out_dir / f"v{index}_a{attempt_no}_pintado.jpg"
                          if masked else target)

        # Reserve, do not merely ask.  "Can I afford this?" is only true
        # until the next worker asks the same question about the same last
        # dollar, so the estimate is held before the call leaves and settled
        # against the real price when it comes back.  It is taken here, as
        # the very last thing before the try/finally that hands it back:
        # anything that could raise in between - the plan note is a database
        # write, and running several variants at once makes 'database is
        # locked' reachable - would strand the hold and quietly shrink her
        # balance for the rest of the process's life.
        gate = billing.reserve(user["id"], provider.name, price,
                               ref=f"{run_id}:v{index}:a{attempt_no}")
        if not gate["ok"]:
            alert = gate.get("alert") or {}
            if alert:
                # kind and level are positional here and would arrive twice if
                # the whole payload were splatted in - which raised TypeError
                # and turned the one refusal that must stop the run into a
                # crashed variant that the loop simply walked past.
                extra = {k: v for k, v in alert.items()
                         if k not in ("kind", "level")}
                billing.raise_alert(user["id"], alert["kind"], alert["level"],
                                    "El robot se ha detenido",
                                    gate["reason"], **extra)
            batch.stop(gate["reason"])
            return {"accepted": False, "cost": cost, "attempts": attempts,
                    "repaired": repaired, "stopped": True,
                    "reason": gate["reason"]}
        hold = gate["hold_id"]

        # Ask once more, with the reservation already in hand.  The gate reads
        # the ledger under a lock and can wait on the database, and she can
        # press Detener while it does; without this the run would pay for one
        # more image per worker after it was told to stop.  The hold is given
        # straight back, so nothing is left promised.
        if batch.aborted():
            billing.release(hold)
            # Nothing reached the provider, so this does not count as a try.
            return {"accepted": False, "cost": cost, "attempts": attempts - 1,
                    "repaired": repaired, "aborted": True}

        settled = False
        try:
            try:
                result = provider.generate(request, painted_target)
            except InsufficientBalance as exc:
                billing.raise_alert(user["id"], "zero_balance", "critical",
                                    "Sin saldo en %s" % provider.name, str(exc),
                                    provider=provider.name)
                batch.stop(str(exc))
                return {"accepted": False, "cost": cost, "attempts": attempts,
                        "repaired": repaired, "stopped": True,
                        "reason": str(exc)}
            except ProviderError as exc:
                _record_attempt(run_id, user["id"], index, attempt_no,
                                provider.name, model, "generate",
                                built["prompt"], negative, merged,
                                {}, [], "error", str(exc)[:200], 0.0, 0)
                if not exc.retryable:
                    return {"accepted": False, "cost": cost,
                            "attempts": attempts, "repaired": repaired}
                continue

            if not result.ok or not result.image_path:
                _record_attempt(run_id, user["id"], index, attempt_no,
                                provider.name, model, "generate",
                                built["prompt"], negative, merged,
                                {}, [], "error", result.error[:200], 0.0,
                                result.latency_ms)
                continue

            real_cost = float(result.cost_usd or 0.0)
            cost += real_cost
            billing.settle(user["id"], provider.name, hold, real_cost,
                           ref=f"{run_id}:v{index}:a{attempt_no}",
                           note="generacion")
            settled = True
        finally:
            # Every way out of that block that did not spend the money must
            # hand the reservation back - an error, a retry, a return - or the
            # rest of the run would be gated against money nobody will spend.
            if not settled:
                billing.release(hold)

        # THE GUARANTEE, ENFORCED HERE AND NOT ASKED OF THE ENGINE.  fal is
        # told which pixels it may repaint; this puts her own back everywhere
        # else, at the full resolution of her photograph, and counts the pixels
        # outside the mask that differ before writing anything.  Measured on
        # her IMG_7871 against the delivered render: 0 of 5 554 727 protected
        # pixels differ, and identity_face reads 0.8044 against 0.8177 on the
        # photograph itself - where the same render judged whole reads 0.7375
        # and fails the verdict.
        if masked:
            batch.detail("Devolviendo tu rostro a la imagen %d" % (index + 1))
            put = protect_mod.compose(original["path"], result.image_path,
                                      mask_path, str(target),
                                      quality=COMPOSE_QUALITY)
            merged["rostro"] = {**merged.get("rostro", {}),
                                "fuera_cambiado": put.get("fuera_cambiado"),
                                # The two numbers that can actually fail, beside
                                # the one that cannot: 'comprobante' says the
                                # mask was drawn on THIS photograph (a mask from
                                # another frame repaints her face while
                                # fuera_cambiado still reads 0), and
                                # 'fuera_pintado' says how much of the answer
                                # fal returned outside the zone it was allowed
                                # to touch, before that half was discarded.
                                "comprobante": put.get("comprobante"),
                                "fuera_pintado": put.get("fuera_pintado"),
                                "compuesta": bool(put.get("ok")),
                                "pintado": Path(str(result.image_path)).name}
            if put.get("ok"):
                result.image_path = put["image_path"]
            else:
                # The image is already bought, so it is judged as it came back
                # rather than thrown away - but the promise about her face was
                # not kept, and the run says so instead of implying it was.
                merged["rostro"]["protegido"] = False
                note = ("No se pudo devolver tu rostro a la imagen %d (%s): se "
                        "revisa la imagen tal como llego."
                        % (index + 1, put.get("reason") or "sin motivo"))
                _plan_note(run_id, note)
                log.warning(note)

        # What was ordered and what arrived, on the attempt row, before
        # anything downstream can resize or repaint the file and make the two
        # agree by accident.
        shape = _shape_facts(result)
        merged.update(shape["params"])
        if shape["note"]:
            _plan_note(run_id, shape["note"])

        # The engine keeps her shape and her colour and sands off the band
        # underneath: her pores, her fine lines, the grain her camera really
        # recorded.  Give it back from her own photograph before anything is
        # measured, so the verdict, the album and the file she downloads are
        # all the same picture.  Only when this engine invents pixels - a
        # compositor cannot have removed a texture it never touched.
        # Everything the robot did to this file, in her language, so the album
        # can never present a corrected image as one that came out right.
        corrections: list = []
        correction_notes: list = []
        # Never on the masked path: the grain transfer works on every skin
        # pixel it finds, including the face, and on that path the face is
        # already her own photograph at full resolution.  Rewriting it would
        # be repairing her camera's grain with her camera's grain, and it
        # would break the one promise this path makes about those pixels.
        if masked:
            merged["textura"] = {"aplicada": False,
                                 "motivo": ("no hace falta: tu piel y tu rostro "
                                            "vienen de tu propia foto")}
        elif checked["generative"] and original and original.get("path"):
            batch.detail("Devolviendo la textura de la piel")
            # Recorded whether or not it changed anything: "se midio y no
            # hacia falta" is an answer she is entitled to read.
            merged["textura"] = _restore_texture(run_id, result.image_path,
                                                 original["path"], profile)
            if merged["textura"].get("aplicada"):
                correction_notes.append(
                    "Se devolvio la textura real de tu piel desde tu foto %s "
                    "(x%.2f): el motor la habia suavizado."
                    % (merged["textura"].get("foto") or "de referencia",
                       merged["textura"].get("ganancia") or 1.0))

        batch.detail("Revisando la imagen %d" % (index + 1))
        verdict = verify_mod.verify_image(result.image_path, profile, checked)

        # The image is bought.  Before anything else is considered - and before
        # a single further cent can be spent - her own photographs are asked to
        # correct what failed, for free, and the result is measured again.  Each
        # correction refuses on its own numbers rather than doing something
        # cosmetic, so this is either a better file or the same file with a
        # sentence saying why it was left alone.
        if not verdict.get("passed"):
            batch.detail("Corrigiendo sin gastar la imagen %d" % (index + 1))
            verdict, corrections, notes = _correct_free(
                run_id, result.image_path, profile, checked, verdict,
                verdict.get("defects") or [])
            correction_notes.extend(notes)
            merged["correcciones"] = corrections
            if notes:
                merged["corregida"] = True
                if masked:
                    # The free corrections work on the whole file, and on this
                    # path most of that file is her own photograph: restore_body
                    # warps the silhouette, restore_face pastes a donor face,
                    # and either would rewrite pixels this run promised not to
                    # touch - the very slimming the client complained about,
                    # done by the robot instead of by the engine.  So the
                    # promise is imposed again over whatever they produced: what
                    # they improved INSIDE the repainted zone is kept, what they
                    # changed outside it goes back to her photograph.
                    again = protect_mod.compose(
                        original["path"], result.image_path, mask_path,
                        str(target), quality=COMPOSE_QUALITY)
                    if again.get("ok"):
                        result.image_path = again["image_path"]
                        verdict = verify_mod.verify_image(result.image_path,
                                                          profile, checked)
                        correction_notes.append(
                            "Las correcciones se aplicaron solo dentro de la "
                            "zona repintada: fuera de ella la imagen sigue "
                            "siendo tu foto.")
                    else:
                        _plan_note(run_id,
                                   "No se pudo reimponer tu rostro tras las "
                                   "correcciones en la imagen %d (%s)."
                                   % (index + 1, again.get("reason") or ""))

        # Only what the free path does NOT own may be bought as a repaint.  A
        # hand repainted by the fill endpoint measures 0.000 severity because
        # the hand has been deleted, and a face that is a different woman is
        # not a region defect at all; paying 0.050 USD a zone for either was
        # buying damage.
        defects = [d for d in (verdict.get("repairable_defects") or [])
                   if str(d.get("type")) not in FREE_DEFECTS]

        if (not verdict.get("passed") and defects
                and max_repair_rounds and not batch.aborted()):
            # A repair is more calls to the provider, so it is held and
            # settled like them.  A refusal here only skips the repair: the
            # image in hand is already paid for and deserves its verdict, and
            # the next generation is where the run really stops.  It is priced
            # as what it is - an inpaint on the fill endpoint, 0.050 USD where
            # a Kontext edit is 0.040 - through the same builder, carrying the
            # real identity references the repair sends.  Reserving the
            # generation price instead was the same defect as the estimate: a
            # hold that is not the bill.
            #
            # And a repair is not ONE call: repair() repaints each failing
            # region separately, up to repair.MAX_REGIONS of them, and the
            # provider charges for every one.  In the rehearsal of 2026-09-03
            # two regions billed 0.100 USD against a 0.050 USD hold - twice
            # what the gate had approved - and the run settled 0.600 USD with
            # only 0.550 USD ever reserved.  So the ceiling is reserved and the
            # truth is settled: over-reserving costs nothing (the hold lives
            # for one call) while under-reserving lets a run spend money the
            # balance gate refused.  Defects merge into regions, so their count
            # is the upper bound on the calls.
            repair_price = provider.estimate_cost(router_mod.build_request(
                "inpaint", quality, source_path=result.image_path,
                reference_paths=[original["path"]],
                source_size=[int(original.get("width") or 0),
                             int(original.get("height") or 0)],
                framing=hints.get("framing") or choices))
            repair_calls = max(1, min(repair_mod.MAX_REGIONS, len(defects)))
            repair_gate = billing.reserve(user["id"], provider.name,
                                          repair_price * repair_calls,
                                          ref=f"{run_id}:v{index}:repair")
            repair_settled = False
            try:
                if not repair_gate["ok"]:
                    log.info("Reparacion omitida por saldo: %s",
                             repair_gate["reason"])
                    fixed = {}
                else:
                    batch.detail("Corrigiendo %s" % _defect_label(defects))
                    try:
                        fixed = repair_mod.repair(result.image_path, defects,
                                                  brief, profile, provider,
                                                  str(target))
                    except InsufficientBalance as exc:
                        # The account ran dry in the middle of the repair.  The
                        # zones painted before that are already on fal's
                        # invoice - 0.0500 USD measured with a provider that
                        # dies on the second of two zones, 0.1000 USD with the
                        # three MAX_REGIONS allows - so they are settled here
                        # before the run stops, exactly like a reverted
                        # repaint.  Then the same hard stop the generation
                        # side does: alert her, halt the batch, and do not let
                        # the other variants keep asking a dry account.
                        spent = float(getattr(exc, "cost_usd", 0.0) or 0.0)
                        if spent > 0.0:
                            cost += spent
                            billing.settle(user["id"], provider.name,
                                           repair_gate["hold_id"], spent,
                                           ref=f"{run_id}:v{index}:repair",
                                           note="reparacion")
                            repair_settled = True
                        billing.raise_alert(user["id"], "zero_balance",
                                            "critical",
                                            "Sin saldo en %s" % provider.name,
                                            str(exc), provider=provider.name)
                        batch.stop(str(exc))
                        return {"accepted": False, "cost": cost,
                                "attempts": attempts, "repaired": repaired,
                                "stopped": True, "reason": str(exc)}
                # The provider charges for a repaint that was reverted exactly
                # like one that was kept, so what is settled is what it really
                # cost - not only the repairs that worked.  Settling on
                # ``ok`` alone dropped 0.050 USD of a 0.650 USD rehearsal from
                # the ledger on 2026-09-03: her balance page would drift from
                # fal's own dashboard by every repair the scanner rejected.
                repair_cost = float(fixed.get("cost_usd") or 0.0)
                if repair_cost > 0.0:
                    cost += repair_cost
                    billing.settle(user["id"], provider.name,
                                   repair_gate["hold_id"], repair_cost,
                                   ref=f"{run_id}:v{index}:repair",
                                   note="reparacion")
                    repair_settled = True
                if fixed.get("ok"):
                    repaired += 1
                    result.image_path = fixed.get("image_path") or result.image_path
                    if masked:
                        # Same reasoning as after the free corrections: a paid
                        # repaint is aimed at a box, and on this path a box can
                        # land on her own face or her own hand.  Whatever it
                        # improved inside the repainted zone is kept; outside
                        # it, her photograph wins.  Then, and only then, is the
                        # file judged.
                        again = protect_mod.compose(
                            original["path"], result.image_path, mask_path,
                            str(target), quality=COMPOSE_QUALITY)
                        if again.get("ok"):
                            result.image_path = again["image_path"]
                    verdict = verify_mod.verify_image(result.image_path,
                                                      profile, checked)
            finally:
                if not repair_settled:
                    billing.release(repair_gate["hold_id"])

        if verdict.get("passed"):
            # A repaired image is not the same claim as one that came out
            # right, and the client has to be able to read which one she is
            # looking at: the sentences travel on the verdict, which the album
            # and the ficha already show.
            if correction_notes:
                verdict = dict(verdict)
                verdict["correcciones"] = list(correction_notes)
                verdict["summary"] = (str(verdict.get("summary") or "").strip()
                                      + " " + " ".join(correction_notes)).strip()
            image_id = _store_image(user, run_id, original, profile, result,
                                    verdict, kind, choices)
            _record_attempt(run_id, user["id"], index, attempt_no, provider.name,
                            result.model or model, "generate", built["prompt"],
                            negative, merged, verdict,
                            verdict.get("defects") or [], "accepted", "",
                            real_cost, result.latency_ms, image_id)
            return {"accepted": True, "cost": cost, "attempts": attempts,
                    "repaired": repaired, "corrected": bool(correction_notes)}

        reason = _reject_reason(verdict)
        # The file STAYS.  Deleting it is the most expensive answer available:
        # 0.84 USD of the 2.34 USD spent on this installation bought two runs
        # whose images no longer exist, so what they proved cannot be looked at
        # again and the same mistake was paid for twice.  Nothing further is
        # spent on it - it is not put in the album either, because an image that
        # failed the identity check is not her and must never be handed over as
        # if it were - but it is kept, and the run says where.
        merged["archivo_conservado"] = str(result.image_path)
        _record_attempt(run_id, user["id"], index, attempt_no, provider.name,
                        result.model or model, "generate", built["prompt"],
                        negative, merged, verdict, verdict.get("defects") or [],
                        "rejected", reason, real_cost, result.latency_ms)
        _plan_note(run_id,
                   "La imagen %d no se pudo aprobar (%s) y NO se ha borrado: "
                   "ya esta pagada y se conserva en %s. No se ha gastado nada "
                   "mas en ella."
                   % (index + 1, reason, Path(result.image_path).name))

        # Stop buying answers the engine has already given.  This is the rule
        # that would have saved 0.27 USD on run_16e276b5, where three paid
        # attempts on one variant came back at identity 0.3507, 0.3753 and
        # 0.3643 against a line at 0.45.
        stop = _repeat_failure(seen_failures, verdict)
        if stop:
            _plan_note(run_id, "Variante %d: %s" % (index + 1, stop))
            return {"accepted": False, "cost": cost, "attempts": attempts,
                    "repaired": repaired, "stopped_repeating": True,
                    "reason": stop}

        # Adapt before trying again: hold closer to the real photograph and
        # name the observed failure in the negative prompt.
        params["strength"] = max(0.25, float(merged.get("strength", 0.5)) - 0.08)
        for defect in (verdict.get("defects") or []):
            extra_negatives.append(str(defect.get("type", "")).replace("_", " "))
        seed += 977

    return {"accepted": False, "cost": cost, "attempts": attempts,
            "repaired": repaired}


def _shape_facts(result) -> dict:
    """The shape ordered, the shape delivered, and one sentence when they differ.

    This is the half of the 1:1 defect that was never fixed.  request_geometry
    now orders her 3:4 instead of a square, but NOTHING read the answer: the
    two images she rejected on 2026-09-04 are 1024x1024 rows whose meta_json
    says only {choices, seed}, so which ratio was sent could not be told from
    the database at all and had to be inferred from the endpoint that billed
    them.  The provider reports both numbers now; here they are turned into a
    row on the attempt and, when they disagree, into one Spanish line on the
    run - because a shape she was promised and did not get is a defect she is
    entitled to read about without measuring the file herself.

    A rehearsal is excluded from the FIRST of the two warnings on purpose:
    MODO ENSAYO answers from a folder of old squares, and letting that accuse
    fal of ignoring the request would be a warning about a call that never
    happened.  The second one is a property of the request itself - the
    endpoint has no such shape to sell - so it is true with or without a
    network, and a rehearsal must show it.
    """
    meta = getattr(result, "meta", None) or {}
    asked = str(meta.get("aspect_ratio") or "")
    got = str(meta.get("delivered_aspect") or "")
    params: dict = {}
    if asked:
        params["forma_pedida"] = asked
    if got:
        params["forma_entregada"] = got
    note = ""
    if asked and got and asked != got and not meta.get("replay"):
        note = ("Aviso: se pidio la imagen en %s y %s la ha devuelto en %s "
                "(%sx%s px). El recorte no es el que elegiste."
                % (asked, result.provider, got, meta.get("width"),
                   meta.get("height")))
    elif asked and meta.get("aspect_exact") is False:
        note = ("Aviso: %s no acepta la forma exacta del encuadre elegido "
                "(%.2f); se pide %s, la mas parecida que ofrece, asi que la "
                "imagen sale un poco mas estrecha que la vista previa del "
                "motor gratuito."
                % (result.provider, float(meta.get("aspect_asked") or 0.0),
                   asked))
    return {"params": params, "note": note}


def _local_hints(choices: dict) -> dict:
    """Translate catalogue choices into the free engine's knobs."""
    extra: dict = {}
    for group, value_key in (choices or {}).items():
        value = options_mod.value_of(group, value_key if isinstance(value_key, str)
                                     else str(value_key))
        if value and value.get("local"):
            extra.update(value["local"])
    return extra


def _face_shield(original: dict, choices: dict, out_dir: Path,
                 index: int) -> tuple[str, str, float]:
    """The mask that keeps her face out of this generation, or why there isn't one.

    Returns ``(mask_path, sentence, cover)``.  An empty path is not an error:
    9 of her 24 photographs are closeups from which no torso can be built, and
    on those there is nothing to repaint around the face, so the image is made
    whole and the sentence says so.  Everything here is free and local; the
    money is only committed afterwards.
    """
    source = str((original or {}).get("path") or "")
    if not source:
        return "", "", 0.0
    mask_file = out_dir / ("v%d_mask.png" % int(index))
    try:
        built = protect_mod.build_mask(source, choices, str(mask_file))
    except Exception as exc:                              # noqa: BLE001
        log.warning("No se pudo preparar la mascara de proteccion: %s", exc)
        return "", ("No se pudo aislar la zona a cambiar, asi que la imagen se "
                    "hace entera y se comprueba la identidad."), 0.0
    if not built.get("ok"):
        return "", ("Tu rostro se volvera a generar: %s. Se comprobara la "
                    "identidad antes de ensenarte la imagen."
                    % (built.get("reason") or "no se pudo aislar la zona")), 0.0
    return str(built["mask_path"]), str(built.get("reason") or ""), \
        float(built.get("cover") or 0.0)


def _defect_label(defects: list[dict]) -> str:
    labels = {"hand_malformed": "una mano", "face_distorted": "el rostro",
              "eye_asymmetry": "los ojos", "extra_limb": "una extremidad",
              "texture_smear": "una zona borrosa",
              "oversmoothed_skin": "la piel"}
    for defect in defects:
        if defect.get("type") in labels:
            return labels[defect["type"]]
    return "un detalle"


def _reject_reason(verdict: dict) -> str:
    failed = [c for c in (verdict.get("checks") or []) if not c.get("passed", True)]
    if not failed:
        return "no supero la revision"
    return "; ".join(verify_mod.FAIL_ES.get(c["name"], c["name"]) for c in failed)


def _store_image(user: dict, run_id: str, original: dict, profile: dict,
                 result, verdict: dict, kind: str, choices: dict) -> str:
    image_id = db.new_id("img")
    path = Path(result.image_path)
    thumb = path.with_name(path.stem + "_thumb.jpg")
    try:
        loader.make_thumb(path, thumb, 512)
    except Exception:                                     # noqa: BLE001
        thumb = None
    info = loader.image_info(path)
    # An image produced in a rehearsal must never be mistakable for one that
    # was bought.  The provider is the only one that knows: it says so in its
    # own meta when it answered from the replay folder instead of the network
    # (providers/fal, MODO ENSAYO), and that mark is written beside the choices
    # so the album row carries it for the life of the file.
    meta = {"choices": choices, "seed": result.seed}
    if (getattr(result, "meta", None) or {}).get("replay"):
        meta["ensayo"] = True
    db.execute(
        "INSERT INTO images(id,user_id,run_id,attempt_id,original_id,profile_id,"
        "kind,path,thumb_path,width,height,bytes,sha256,provider,model,cost_usd,"
        "score,verdict_json,meta_json,created_at) "
        "VALUES(?,?,?,NULL,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
        (image_id, user["id"], run_id, original["id"], (profile or {}).get("id"),
         kind, str(path), str(thumb) if thumb else None,
         int(info.get("width") or 0), int(info.get("height") or 0),
         int(info.get("bytes") or 0), str(info.get("sha256") or ""),
         result.provider, result.model, float(result.cost_usd or 0.0),
         float(verdict.get("score") or 0.0), db.dumps(verdict),
         db.dumps(meta), db.now()),
    )
    return image_id


def _record_attempt(run_id: str, user_id: str, index: int, attempt_no: int,
                    provider: str, model: str, operation: str, prompt: str,
                    negative: str, params: dict, verdict: dict,
                    defects: list, status: str, reason: str, cost: float,
                    latency: int, image_id: str | None = None) -> str:
    attempt_id = db.new_id("att")
    db.execute(
        "INSERT INTO attempts(id,run_id,user_id,variant_index,attempt_no,"
        "provider,model,operation,prompt,negative_prompt,params_json,"
        "verdict_json,defects_json,status,reject_reason,cost_usd,latency_ms,"
        "image_id,created_at) VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
        (attempt_id, run_id, user_id, index, attempt_no, provider, model,
         operation, prompt[:4000], negative[:2000], db.dumps(params),
         db.dumps(verdict), db.dumps(defects), status, reason[:300],
         float(cost or 0.0), int(latency or 0), image_id, db.now()),
    )
    if image_id:
        db.execute("UPDATE images SET attempt_id=? WHERE id=?",
                   (attempt_id, image_id))
    return attempt_id


# -------------------------------------------------------------------- finals

def run_final(user: dict, run_id: str) -> dict:
    """Re-render the chosen previews at full quality from the original photo."""
    run = db.row_to_dict(db.q1("SELECT * FROM runs WHERE id=? AND user_id=?",
                               (run_id, user["id"])))
    if not run:
        raise ValueError("Ese trabajo no existe.")
    _clear_cancel(run_id)

    opts = run.get("options") or {}
    selected = list(opts.get("selected_image_ids") or [])
    quality = str(opts.get("quality") or "high")
    if not selected:
        _set(run_id, status="failed", error="No elegiste ninguna imagen.",
             finished_at=db.now())
        return {"ok": False, "error": "No elegiste ninguna imagen."}

    original = db.row_to_dict(db.q1("SELECT * FROM originals WHERE id=?",
                                    (run["original_id"],)))
    analysis = analyse_original(original) if original else {}
    profile = _profile_for(user, run.get("profile_id"))
    style = styles_mod.get_style(opts.get("style")) or styles_mod.default_style(
        analysis.get("shot_type") or "unknown")
    brief = dict(opts.get("brief") or {})
    brief["reference_paths"] = [
        p for p in ((run.get("plan") or {}).get("reference_paths") or [])
        if p and Path(p).exists()]
    if original:
        brief["source_path"] = original["path"]
        brief["source_body"] = analysis.get("body") or {}
        brief["source_skin"] = analysis.get("skin") or {}

    out_dir = OUTPUT_DIR / str(user["id"]) / run_id
    out_dir.mkdir(parents=True, exist_ok=True)

    # The chosen images become variants first, so the finals go through the
    # same bounded pool as the previews: a final costs one provider queue too,
    # and three of them in a row cost three.
    variants: list[dict] = []
    for i, image_id in enumerate(selected):
        source = db.row_to_dict(db.q1("SELECT * FROM images WHERE id=? AND user_id=?",
                                      (image_id, user["id"])))
        if not source:
            continue
        meta = source.get("meta") or {}
        variants.append({"index": i, "choices": meta.get("choices") or {},
                         "seed": int(meta.get("seed") or 0), "params": {}})

    results, batch = _run_batch(user, run_id, variants, brief, profile, style,
                                quality, out_dir, original, "Alta calidad",
                                "final")
    tally = _tally(results)
    accepted = tally["accepted"]
    spent = tally["cost"]
    _order_images_by_variant(run_id)

    if batch.stop_reason:
        _set(run_id, status="stopped_no_balance", finished_at=db.now(),
             error=batch.stop_reason, cost_usd=round(spent, 6),
             n_accepted=accepted, stage="Detenido por falta de saldo")
        return {"ok": False, "stopped": True, "reason": batch.stop_reason}
    if batch.cancelled:
        _set(run_id, status="cancelled", finished_at=db.now(),
             cost_usd=round(spent, 6), n_accepted=accepted, stage="Detenido")
        return {"ok": True, "cancelled": True, "accepted": accepted,
                "cost_usd": round(spent, 4)}

    _set(run_id, status="done", progress=1.0, finished_at=db.now(),
         n_accepted=accepted, cost_usd=round(spent, 6),
         stage="Listo: %d en alta calidad" % accepted)
    return {"ok": True, "accepted": accepted, "cost_usd": round(spent, 4)}


# -------------------------------------------------------------------- report

def build_report(run_id: str) -> dict:
    """The ficha promised to the client, in her language, from real rows."""
    run = db.row_to_dict(db.q1("SELECT * FROM runs WHERE id=?", (run_id,)))
    if not run:
        return {"ok": False, "reason": "Ese trabajo no existe."}
    attempts = db.rows_to_dicts(db.q(
        "SELECT * FROM attempts WHERE run_id=? ORDER BY variant_index, attempt_no",
        (run_id,)))
    images = db.rows_to_dicts(db.q(
        "SELECT * FROM images WHERE run_id=? AND deleted_at IS NULL "
        "ORDER BY created_at", (run_id,)))

    accepted = [a for a in attempts if a["status"] == "accepted"]
    discarded = [a for a in attempts if a["status"] == "rejected"]
    errors = [a for a in attempts if a["status"] == "error"]

    detected: dict[str, int] = {}
    for attempt in attempts:
        for defect in (attempt.get("defects") or []):
            name = verify_mod.DEFECT_ES.get(defect.get("type"),
                                            defect.get("type") or "")
            detected[name] = detected.get(name, 0) + 1

    seconds = None
    if run.get("finished_at") and run.get("started_at"):
        seconds = round(float(run["finished_at"]) - float(run["started_at"]), 1)

    models = sorted({a["provider"] + (":" + a["model"] if a["model"] else "")
                     for a in attempts if a.get("provider")})
    # What the robot gave back, not only what it threw away: how many results
    # got her real skin grain returned, and by how much on average.  It rides
    # on params_json, which every attempt already carries, so the ficha gains
    # it without a migration.
    restored = [(a.get("params") or {}).get("textura") for a in attempts]
    restored = [t for t in restored if isinstance(t, dict) and t.get("aplicada")]
    ganancia = (round(sum(float(t.get("ganancia") or 1.0) for t in restored)
                      / len(restored), 3) if restored else None)
    # Routing warnings written during the run: which requested changes the
    # chosen engine was unable to make, which photographs were sent as
    # references, what was corrected for free and which paid files were kept
    # instead of being deleted.  She reads them next to the cost.
    avisos = [n for n in ((run.get("plan") or {}).get("notes") or [])
              if isinstance(n, str)]
    # What the robot fixed by itself, per image, in her own language.  It rides
    # on params_json like the texture record, so no migration is needed.
    corregidas = []
    for attempt in attempts:
        applied = [c for c in ((attempt.get("params") or {}).get("correcciones")
                               or []) if c.get("aplicada")]
        refused = [c for c in ((attempt.get("params") or {}).get("correcciones")
                               or []) if not c.get("aplicada")]
        if applied or refused:
            corregidas.append({
                "variante": attempt["variant_index"],
                "intento": attempt["attempt_no"],
                "aplicadas": [c.get("que") for c in applied],
                "no_aplicadas": [{"que": c.get("que"), "motivo": c.get("motivo")}
                                 for c in refused],
            })
    conservadas = [(a.get("params") or {}).get("archivo_conservado")
                   for a in attempts]
    conservadas = [c for c in conservadas if c]

    return {
        "ok": True,
        "run_id": run_id,
        "estado": run.get("status"),
        "intentos": len(attempts),
        "aceptadas": len(accepted),
        "descartadas": len(discarded),
        "errores": len(errors),
        "reparadas": int(run.get("n_repaired") or 0),
        "intentos_por_foto": (round(len(attempts) / len(accepted), 2)
                              if accepted else None),
        "segundos": seconds,
        "coste_usd": round(float(run.get("cost_usd") or 0.0), 4),
        "coste_por_foto_usd": (round(float(run.get("cost_usd") or 0.0)
                                     / len(accepted), 4) if accepted else None),
        "modelos": models,
        "avisos": avisos,
        "textura_restaurada": len(restored),
        "textura_ganancia": ganancia,
        "correcciones": corregidas,
        "archivos_conservados": conservadas,
        "defectos_detectados": detected,
        "motivos_descarte": [
            {"variante": a["variant_index"], "intento": a["attempt_no"],
             "motivo": a.get("reject_reason") or "",
             "detalle": [c.get("detail") for c in
                         (a.get("verdict") or {}).get("checks", [])
                         if not c.get("passed", True)]}
            for a in discarded],
        "imagenes": [
            {"id": img["id"], "score": img.get("score"),
             "coste_usd": img.get("cost_usd"),
             "resumen": (img.get("verdict") or {}).get("summary", ""),
             "comprobaciones": [
                 {"nombre": verify_mod.CHECK_ES.get(c["name"], c["name"]),
                  "valor": c.get("value"), "limite": c.get("threshold"),
                  "paso": c.get("passed"), "detalle": c.get("detail")}
                 for c in (img.get("verdict") or {}).get("checks", [])]}
            for img in images],
    }
