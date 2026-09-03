"""The quality gate every generated image has to pass.

This is where the robot earns its name.  A generator can be asked politely not
to slim someone down; only a measurement can prove it did not.  ``verify_image``
re-measures the produced pixels with the same analysers that built the profile
and answers five questions with numbers: is it the same face, the same body, the
same skin, is the anatomy sane, is the file technically good enough.

Two rules shape the code.  First, a check that could not be computed never fails
the person - missing data is reported as skipped, not as a defect.  Second, the
verdict must be readable: the details are written in the Spanish the client will
actually read, naming what changed and by how much.
"""
from __future__ import annotations

import math
import time
from typing import Any

import numpy as np

from ..analysis import anomaly as anomaly_mod
from ..analysis import body as body_mod
from ..analysis import face as face_mod
from ..analysis import loader
from ..analysis import pose as pose_mod
from ..analysis import quality as quality_mod
from ..analysis import segment as segment_mod
from ..analysis import skin as skin_mod
from .profile import DEFAULT_THRESHOLDS, GATED_METRICS, usable_metrics

VERIFY_MAX_SIDE = 1600
QUALITY_MIN = 0.45
ANATOMY_SEVERITY_MAX = 0.6
BODY_CONF_MIN = 0.5           # below this, measurements only inform, never gate
YAW_BAND_SLACK = 1.4          # extra tolerance for a yaw-corrected width
PAIRED_TOL = 0.08             # floor for the shape change vs the source photo
PROFILE_MIN_SAMPLES = 4       # torso heights needed before the profile counts

# A pose landmark is not a tape measure.  Measuring one untouched photograph
# again at a different scale moves the shoulder and hip widths by several
# percent, and the torso length that divides them moves with them, so a fixed 8%
# limit is a gate more precise than its own instrument: it rejects images whose
# body was never touched.  The limit is therefore estimated for each pair - the
# source photograph is read again at the size of the result and once more
# smaller, and the spread between those readings is this pair's noise floor.
PAIRED_NOISE_FACTOR = 1.5     # a real change must stand this far above the noise
NOISE_RESCALE = 0.8           # the second look at the source, at 80% of the first
NOISE_MIN_SIDE = 320          # below this a body measurement is not worth taking
# The cap is what keeps a noisy pair from quietly switching the protection off.
# The client's complaint was a slim-down of about 12%, and the profile builder
# refuses to learn a band wider than +/-12% of the mean for the same reason, so
# a limit past 16% would forgive precisely the damage this module exists to
# catch.  When the measured noise asks for more, the limit stops here and the
# detail says the pair was too noisy to judge instead of widening in silence.
PAIRED_TOL_MAX = 0.16
# The silhouette profile is compared as a whole, so it carries its own noise
# entry rather than borrowing one from a single width.
WIDTH_PROFILE_KEY = "width_profile"
# A difference the engine physically could not have caused is not evidence about
# the body.  It must not reward the image either, so the check scores it in the
# middle instead of dragging the run's score down with a measurement error.
NON_GENERATIVE_SCORE = 0.75

# Read off the silhouette, so clothing moves them; the rest come from the
# skeleton and survive a change of outfit.
SILHOUETTE_METRICS = ("waist_w_over_torso", "bust_w_over_torso")

CHECK_WEIGHTS: dict[str, float] = {
    "identity_face": 0.30,
    "body_proportions": 0.25,
    "skin_tone": 0.15,
    "anatomy": 0.20,
    "quality": 0.10,
}

# metric -> (noun, word when it shrank, word when it grew)
METRIC_ES: dict[str, tuple[str, str, str]] = {
    "shoulder_w_over_torso": ("hombros", "mas estrechos", "mas anchos"),
    "hip_w_over_torso": ("caderas", "mas estrechas", "mas anchas"),
    "waist_w_over_torso": ("cintura", "mas estrecha", "mas ancha"),
    "bust_w_over_torso": ("busto", "mas estrecho", "mas ancho"),
    "head_h_over_torso": ("cabeza", "mas pequena", "mas grande"),
    "neck_w_over_torso": ("cuello", "mas estrecho", "mas ancho"),
    "arm_len_over_torso": ("brazos", "mas cortos", "mas largos"),
    "leg_len_over_torso": ("piernas", "mas cortas", "mas largas"),
    "shoulder_over_hip": ("relacion hombros-caderas", "menor", "mayor"),
}

DEFECT_ES: dict[str, str] = {
    "hand_malformed": "mano deformada",
    "extra_limb": "extremidad de mas",
    "extra_person": "persona de mas",
    "face_distorted": "rostro alterado",
    "eye_asymmetry": "ojos asimetricos",
    "missing_limb": "falta una extremidad",
    "texture_smear": "textura emborronada",
    "duplicated_feature": "elemento duplicado",
    "border_artifact": "artefacto en el borde",
    "oversmoothed_skin": "piel demasiado suavizada",
}

CHECK_ES: dict[str, str] = {
    "identity_face": "el rostro",
    "body_proportions": "las proporciones",
    "skin_tone": "el tono de piel",
    "anatomy": "la anatomia",
    "quality": "la calidad tecnica",
}

FAIL_ES: dict[str, str] = {
    "identity_face": "el rostro no coincide con el tuyo",
    "body_proportions": "cambiaron tus proporciones",
    "skin_tone": "cambio tu tono de piel",
    "anatomy": "hay errores anatomicos",
    "quality": "la calidad tecnica es baja",
}


# ------------------------------------------------------------------ helpers

def _safe(fn, *args) -> Any:
    try:
        return fn(*args)
    except Exception:
        return None


def _ok_dict(value: Any, reason: str) -> dict:
    return value if isinstance(value, dict) else {"ok": False, "reason": reason}


def _f(value: Any, default: Any = 0.0) -> Any:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return default
    if math.isnan(out) or math.isinf(out):
        return default
    return out


def _flist(value: Any, n: int) -> list[float] | None:
    if not isinstance(value, (list, tuple, np.ndarray)) or len(value) < n:
        return None
    out: list[float] = []
    for item in list(value)[:n]:
        val = _f(item, None)
        if val is None:
            return None
        out.append(float(val))
    return out


def _clamp01(value: Any) -> float:
    val = _f(value, 0.0)
    return 0.0 if val < 0.0 else (1.0 if val > 1.0 else float(val))


def _int_bbox(value: Any) -> list[int]:
    box = _flist(value, 4)
    if box is None or box[2] <= 0 or box[3] <= 0:
        return []
    return [int(round(box[0])), int(round(box[1])),
            int(round(box[2])), int(round(box[3]))]


def _ms(started: float) -> int:
    return int(round((time.perf_counter() - started) * 1000.0))


def _mk(name: str, value: Any, threshold: Any, passed: bool, detail: str) -> dict:
    return {"name": name, "value": round(_f(value, 0.0), 4),
            "threshold": round(_f(threshold, 0.0), 4), "passed": bool(passed),
            "weight": CHECK_WEIGHTS[name], "detail": detail}


def _thresholds(profile: dict) -> dict[str, float]:
    out = dict(DEFAULT_THRESHOLDS)
    src = profile.get("thresholds")
    if isinstance(src, dict):
        for key, value in src.items():
            val = _f(value, None)
            if val is not None:
                out[str(key)] = float(val)
    return out


def _defect(raw: Any) -> dict | None:
    if not isinstance(raw, dict):
        return None
    kind = str(raw.get("type") or "").strip()
    if not kind:
        return None
    return {"type": kind, "where": str(raw.get("where") or ""),
            "bbox": _int_bbox(raw.get("bbox")),
            "severity": round(_clamp01(raw.get("severity")), 3),
            "repairable": bool(raw.get("repairable")),
            "detail": str(raw.get("detail") or "")}


def _severity_es(severity: float) -> str:
    if severity >= 0.8:
        return "grave"
    if severity >= ANATOMY_SEVERITY_MAX:
        return "importante"
    if severity >= 0.35:
        return "moderado"
    return "leve"


def _expects_face(brief: dict) -> bool:
    """Does the brief say the shot must show a face?  Unknown means no claim."""
    for key in ("expects_face", "face_visible", "needs_face", "show_face"):
        if key in brief:
            return bool(brief.get(key))
    return str(brief.get("shot_type") or "").lower() in ("closeup", "half", "full")


def _is_generative(brief: dict) -> bool:
    """Can the engine that produced this image invent anatomy at all?

    Some providers here are not generators: they replace the background,
    recolour a garment, relight and crop.  Such an engine cannot make a woman
    slimmer even if it were asked to, so any proportion difference measured on
    its output is an artefact of the measurement by construction.  The caller
    knows which engine ran and says so; when nobody says, the answer is yes,
    because assuming a generator is the assumption that keeps the gate honest.
    """
    for key in ("generative", "provider_generative"):
        value = brief.get(key)
        if value is not None:
            return bool(value)
    return True


def _source_had_face(brief: dict) -> bool:
    """Did the photograph this image was made from actually show a face?

    Only worth asking on the non-generative path, and only when the generated
    image lost its face, so the detection here is paid for at most once per
    verification and never on the common route.
    """
    src = brief.get("source_face")
    if isinstance(src, dict):
        return bool(src.get("ok"))
    for key in ("source_has_face", "original_has_face"):
        value = brief.get(key)
        if value is not None:
            return bool(value)
    path = _source_path(brief)
    if not path:
        return False
    img = _safe(loader.load_image, path, VERIFY_MAX_SIDE)
    if not isinstance(img, np.ndarray) or img.size == 0:
        return False
    found = _ok_dict(_safe(face_mod.detect_face, img), "rostro no detectado")
    return bool(found.get("ok"))


def _unreliable_metrics(body: dict) -> set[str]:
    """Metrics measure_body itself distrusts; they inform but never gate.

    One exception, and it is the difference between a working proportion gate
    and a decorative one: a width flagged only because the subject was turned
    has already had its foreshortening undone by measure_body, so it is an
    approximation rather than a ruin.  Those are gated, with a wider tolerance
    applied in _check_body to pay for the correction's own error.  Widths ruined
    by an arm across the scanline stay out.
    """
    out: set[str] = set()
    raw = body.get("unreliable")
    if isinstance(raw, (list, tuple, set)):
        out.update(str(v) for v in raw)
    reasons = body.get("reliability")
    corrected = {str(v) for v in (body.get("corrected") or [])}
    if isinstance(reasons, dict):
        for name in list(out):
            if reasons.get(name) == "yaw" and name in corrected:
                out.discard(name)
    flags = body.get("flags")
    if isinstance(flags, dict):
        for name, flag in flags.items():
            if isinstance(flag, dict):
                if flag.get("unreliable"):
                    out.add(str(name))
            elif str(flag).lower() in ("unreliable", "low", "poco_fiable"):
                out.add(str(name))
    # A low confidence measurement is unreliable by definition.
    if _f(body.get("confidence")) < BODY_CONF_MIN:
        out.update(str(k) for k in (body.get("metrics") or {}))
    return out


# ------------------------------------------------------------------- checks

def _check_identity_face(img: np.ndarray, face_d: dict, profile: dict,
                         thresholds: dict, brief: dict,
                         defects_out: list[dict]) -> tuple[dict, float, bool]:
    name = "identity_face"
    face_min = _f(thresholds.get("face_min"), 0.72)
    reference = (profile.get("face") or {}).get("descriptor") or []
    if not isinstance(reference, (list, tuple, np.ndarray)) or len(reference) < 8:
        return (_mk(name, face_min, face_min, True,
                    "El perfil no guarda descriptor facial, comprobacion omitida."),
                1.0, False)

    descriptor = face_d.get("descriptor") if face_d.get("ok") else None
    if face_d.get("ok") and not descriptor:
        descriptor = _safe(face_mod.face_descriptor, img, face_d)
    if not face_d.get("ok") or not isinstance(descriptor, (list, tuple, np.ndarray)) \
            or len(descriptor) < 8:
        if _expects_face(brief):
            # An engine that only composites, recolours and crops never draws a
            # face: every face pixel in its output came from her own photograph.
            # When such an engine returns something the detector cannot read - a
            # hard crop, a dark relight, a strong colour grade - the honest
            # reading is that the detector failed, not that she was replaced by
            # someone else, so it is reported instead of rejected.  The hard
            # failure below is untouched for engines that do invent pixels,
            # which is the case it was written for.
            if not _is_generative(brief) and _source_had_face(brief):
                return (_mk(name, face_min, face_min, True,
                            "No se detecto rostro en la imagen generada, pero "
                            "este motor no dibuja caras: los pixeles del rostro "
                            "vienen de tu propia foto. Se avisa y no se rechaza."),
                        1.0, False)
            defects_out.append({
                "type": "face_distorted", "where": "face", "bbox": [],
                "severity": 0.9, "repairable": False,
                "detail": "No hay un rostro reconocible donde el encuadre lo exige.",
            })
            return (_mk(name, 0.0, face_min, False,
                        "No se detecto ningun rostro en la imagen generada, "
                        "pero el encuadre pedido si lo lleva."), 0.0, True)
        return (_mk(name, face_min, face_min, True,
                    "No se detecto rostro y el encuadre no exige que se vea; "
                    "comprobacion omitida."), 1.0, False)

    similarity = _f(_safe(face_mod.compare_faces, list(descriptor), list(reference)), None)
    if similarity is None:
        return (_mk(name, face_min, face_min, True,
                    "No se pudo comparar el rostro, comprobacion omitida."), 1.0, False)
    similarity = _clamp01(similarity)
    passed = similarity >= face_min
    if not passed:
        bbox = _int_bbox(face_d.get("bbox"))
        severity = min(1.0, 0.55 + (face_min - similarity) * 1.5)
        defects_out.append({
            "type": "face_distorted", "where": "face", "bbox": bbox,
            "severity": round(severity, 3), "repairable": bool(bbox),
            "detail": "Los rasgos del rostro no coinciden con los del perfil.",
        })
    detail = ("Parecido facial %.2f (minimo %.2f). %s"
              % (similarity, face_min,
                 "El rostro se mantiene." if passed
                 else "El rostro fue modificado respecto a tus fotos."))
    return _mk(name, similarity, face_min, passed, detail), similarity, True


def _source_path(brief: dict) -> str:
    path = brief.get("source_path") or brief.get("original_path")
    return str(path) if path else ""


def _measure_source(path: str, max_side: int) -> tuple[dict, int]:
    """Measure one photograph end to end at a chosen resolution.

    The longest side actually used comes back with the measurement, because the
    noise estimate below must not pay twice for a reading it already has.
    """
    img = _safe(loader.load_image, str(path), int(max_side))
    if not isinstance(img, np.ndarray) or img.size == 0:
        return {}, 0
    side = int(max(img.shape[0], img.shape[1]))
    pose_s = _ok_dict(_safe(pose_mod.detect_pose, img), "pose no disponible")
    seg_s = _ok_dict(_safe(segment_mod.person_mask, img), "segmentacion no disponible")
    mask = seg_s.get("mask") if seg_s.get("ok") else None
    mask = mask if isinstance(mask, np.ndarray) else None
    return _ok_dict(_safe(body_mod.measure_body, img, pose_s, mask),
                    "medidas no disponibles"), side


def _source_body(brief: dict) -> tuple[dict, int]:
    """Measurements of the photograph this image was generated from.

    The orchestrator already measures the original before planning, so it passes
    the result through the brief and nothing is measured twice.  A caller that
    only knows the path still gets the paired test, at the cost of one analysis.
    Returns the measurement and the resolution it was taken at; a measurement
    handed over in the brief reports 0, since its scale is not knowable here.
    """
    src = brief.get("source_body")
    if isinstance(src, dict) and src.get("ok"):
        return src, 0
    path = _source_path(brief)
    if not path:
        return {}, 0
    return _measure_source(path, VERIFY_MAX_SIDE)


def _profile_ratio(gen_profile: Any, src_profile: Any) -> dict | None:
    """Median width ratio between two silhouette profiles of the same torso.

    Both profiles are lists of ``[t, width_over_torso]`` sampled at the same
    fractions of torso height, so the pairing is exact and no interpolation is
    needed.  The median rather than the mean, because one bad scanline - a hand
    resting on a hip, a sleeve - should not carry the verdict.
    """
    if not isinstance(gen_profile, (list, tuple)) or not isinstance(src_profile, (list, tuple)):
        return None
    src_map: dict[float, float] = {}
    for row in src_profile:
        if isinstance(row, (list, tuple)) and len(row) >= 2:
            t, val = _f(row[0], None), _f(row[1], None)
            if t is not None and val is not None and val > 1e-6:
                src_map[round(float(t), 2)] = float(val)
    ratios: list[float] = []
    for row in gen_profile:
        if not (isinstance(row, (list, tuple)) and len(row) >= 2):
            continue
        t, val = _f(row[0], None), _f(row[1], None)
        if t is None or val is None:
            continue
        ref = src_map.get(round(float(t), 2))
        if ref is None or ref <= 1e-6:
            continue
        ratios.append(float(val) / ref)
    if len(ratios) < PROFILE_MIN_SAMPLES:
        return None
    return {"median": float(np.median(ratios)), "n": len(ratios),
            "spread": float(np.std(ratios))}


def _clothing_changed(brief: dict) -> bool:
    """Did the user ASK for different clothes in this generation?

    Only an explicit choice counts.  The brief also carries a description of
    what she was already wearing, read off the source photograph, and treating
    that as a change would switch the silhouette check off on every single run -
    quietly disabling the protection this whole module exists to provide.
    """
    choices = brief.get("choices")
    if isinstance(choices, dict):
        for group in ("clothing", "clothing_color", "transparency"):
            if choices.get(group):
                return True
    return bool(brief.get("clothing_changed"))


def _paired_noise(brief: dict, src_body: dict, gen_side: int,
                  src_side: int) -> dict[str, float]:
    """How far this pair's numbers move while nothing about the body moves.

    The source photograph is read again at the size of the result and once more
    at 80% of what that reading really got.  Nothing about the woman changed
    between those readings, so whatever the metrics do is the instrument's own
    scatter, and a difference smaller than that scatter says nothing at all
    about her shape.  Two extra measurements is the entire budget: enough for a
    range, cheap enough to run on every verification, one of them skipped when
    the reading already in hand was taken at that very size, and the whole
    thing skipped when the caller never said which photograph this image came
    from.
    """
    path = _source_path(brief)
    if not path or gen_side <= 0 or not src_body.get("ok"):
        return {}

    samples = [src_body]
    seen = {int(src_side)} if src_side > 0 else set()
    first_used = 0
    # The second look steps down from the size the first look actually got, not
    # from the size it asked for.  The loader never enlarges, so a photograph
    # smaller than the result answers both requests with the very same pixels:
    # the readings would agree perfectly, the noise would be reported as zero
    # and the fixed 8% limit would come back for exactly the small uploads that
    # need the widening most.
    for position in (0, 1):
        if position == 0:
            side = int(gen_side)
        else:
            base = first_used if first_used > 0 else int(gen_side)
            side = int(round(base * NOISE_RESCALE))
        if side < NOISE_MIN_SIDE or side in seen:
            continue
        seen.add(side)
        again, used = _measure_source(path, side)
        seen.add(int(used))
        if position == 0:
            first_used = int(used)
        if isinstance(again, dict) and again.get("ok"):
            samples.append(again)
    if len(samples) < 2:
        return {}

    # usable_metrics is the profile builder's own opinion of which numbers may
    # be trusted from one photograph; asking it here keeps the noise floor and
    # the comparison arguing about the same set of metrics.
    trusted = [usable_metrics(sample) for sample in samples]

    noise: dict[str, float] = {}
    for metric in GATED_METRICS:
        values: list[float] = []
        for sample, ok in zip(samples, trusted):
            if metric not in ok:
                continue
            value = _f((sample.get("metrics") or {}).get(metric), None)
            if value is not None and abs(value) > 1e-9:
                values.append(float(value))
        if len(values) < 2:
            continue
        mean = float(np.mean(values))
        if abs(mean) < 1e-9:
            continue
        noise[metric] = float(max(values) - min(values)) / abs(mean)

    # The silhouette is compared as a median of ratios, so its noise is measured
    # the same way it is used: one source reading against another, where the
    # honest answer is exactly 1.0.
    spreads: list[float] = []
    for i in range(len(samples)):
        for j in range(i + 1, len(samples)):
            ratio = _profile_ratio(samples[j].get("width_profile"),
                                   samples[i].get("width_profile"))
            if ratio is not None:
                spreads.append(abs(ratio["median"] - 1.0))
    if spreads:
        noise[WIDTH_PROFILE_KEY] = float(max(spreads))
    return noise


def _noise_aware_tol(base_tol: float, observed_noise: Any) -> tuple[float, bool]:
    """The limit for one metric: never under the floor, never over the cap.

    Also reports whether the cap had to bite, so the verdict can admit that what
    could not be resolved was the instrument rather than the body.
    """
    value = _f(observed_noise, None)
    if value is None or value <= 0.0:
        return float(base_tol), False
    wanted = PAIRED_NOISE_FACTOR * float(value)
    if wanted <= base_tol:
        return float(base_tol), False
    return float(min(wanted, PAIRED_TOL_MAX)), bool(wanted > PAIRED_TOL_MAX)


def _check_body_paired(gen_body: dict, src_body: dict, thresholds: dict,
                       brief: dict | None = None, gen_side: int = 0,
                       src_side: int = 0
                       ) -> tuple[dict, float, bool, list[str]] | None:
    """Compare the result against the very photograph it was made from.

    This is the sensitive test, and it is available almost every time, because
    the robot always transforms a real source image rather than inventing one.
    Measuring the same body in the same pose twice cancels nearly all of the
    noise that makes a population band blunt: framing, camera distance, how far
    the subject was turned, where her arms were.  What survives the subtraction
    is what the generator actually did to her shape.

    What does not survive the subtraction is the instrument's own scatter, and
    that scatter is wider than the fixed 8% this check used to insist on, which
    is why every metric is judged against a limit sized from the noise measured
    on this very pair instead of against a constant.

    Returns None when there is no usable source measurement, so the caller can
    fall back to the profile bands.
    """
    if not (gen_body.get("ok") and isinstance(src_body, dict) and src_body.get("ok")):
        return None

    brf = brief if isinstance(brief, dict) else {}
    base_tol = _f(thresholds.get("paired_tol"), PAIRED_TOL)
    noise = _paired_noise(brf, src_body, gen_side, src_side)
    gen_metrics = gen_body.get("metrics") or {}
    src_metrics = src_body.get("metrics") or {}
    gen_usable = _usable(gen_body)
    src_usable = _usable(src_body)

    offenders: list[str] = []
    notes: list[str] = []
    records: list[tuple[float, float]] = []   # (deviation, limit applied to it)
    widened = False
    too_noisy = False
    compared = 0

    dressed_now = _clothing_changed(brf)
    for metric in GATED_METRICS:
        # Waist and bust are read off the silhouette, so new clothes move them
        # for honest reasons; shoulders, hips and head come from the skeleton.
        if dressed_now and metric in SILHOUETTE_METRICS:
            continue
        gen_val = _f(gen_metrics.get(metric), None)
        src_val = _f(src_metrics.get(metric), None)
        if gen_val is None or src_val is None or abs(src_val) < 1e-9:
            continue
        if metric not in gen_usable or metric not in src_usable:
            continue
        ratio = gen_val / src_val
        deviation = abs(ratio - 1.0)
        tol, capped = _noise_aware_tol(base_tol, noise.get(metric))
        widened = widened or tol > base_tol + 1e-9
        too_noisy = too_noisy or capped
        compared += 1
        records.append((deviation, tol))
        if deviation <= tol:
            continue
        label, down, up = METRIC_ES.get(metric, (metric, "menor", "mayor"))
        pct = int(round(deviation * 100.0))
        offenders.append("%s un %d%% %s que en tu foto original"
                         % (label, pct, down if ratio < 1.0 else up))

    # The silhouette profile: the same torso sampled at nine heights in both
    # images.  Individually each sample is noisy, but a uniform slim-down moves
    # all of them the same way, so the median ratio is a far steadier signal
    # than any single width - and a uniform slim-down is exactly the failure the
    # client experienced.
    # A wool coat makes a person wider, and that is not the generator taking a
    # liberty with her body.  When the request changed the clothes, the
    # silhouette is expected to move, so the verdict rests on the skeletal
    # ratios instead - shoulder and hip widths come from pose landmarks, which
    # clothing barely shifts.  Gating the silhouette here would reject every
    # jacket she ever asks for and teach her to distrust the check that exists
    # to protect her.
    dressed = _clothing_changed(brf)
    prof_ratio = None if dressed else _profile_ratio(
        gen_body.get("width_profile"), src_body.get("width_profile"))
    if dressed:
        notes.append("Ropa distinta: la silueta no se compara, solo el esqueleto.")
    if prof_ratio is not None:
        deviation = abs(prof_ratio["median"] - 1.0)
        tol, capped = _noise_aware_tol(base_tol, noise.get(WIDTH_PROFILE_KEY))
        widened = widened or tol > base_tol + 1e-9
        too_noisy = too_noisy or capped
        compared += prof_ratio["n"]
        records.append((deviation, tol))
        if deviation > tol:
            pct = int(round(deviation * 100.0))
            offenders.append(
                "tu silueta es un %d%% %s en toda la altura del torso"
                % (pct, "mas estrecha" if prof_ratio["median"] < 1.0 else "mas ancha"))
        else:
            notes.append("Silueta comparada en %d alturas del torso."
                         % prof_ratio["n"])

    if compared == 0:
        return None

    # After the noise correction the limits are no longer all the same, so the
    # measurement worth reporting is the one that came closest to its own limit.
    # Both halves must come from that same measurement: pairing the largest
    # deviation with a different metric's limit prints a difference above its
    # own limit on an image that passed, which reads as a contradiction to the
    # only person who matters here.
    worst_rec = (max(records, key=lambda r: r[0] / max(r[1], 1e-6))
                 if records else (0.0, base_tol))
    worst_dev, worst_tol = worst_rec[0], worst_rec[1]
    score = _clamp01(1.0 - (worst_dev / max(worst_tol, 1e-6)) / 2.0)

    generative = _is_generative(brf)
    # A difference the engine cannot physically have produced is a measurement
    # artefact, not a change to her body, so it is reported and not held against
    # the image.  Engines that really do generate keep the rejection.
    degraded = bool(offenders) and not generative
    passed = (not offenders) or degraded

    if degraded:
        detail = ("Diferencia medida respecto a la foto original: %s. Este motor "
                  "no genera cuerpo (solo cambia fondo, color, luz y encuadre), "
                  "asi que no pudo tocar tus proporciones: la diferencia es ruido "
                  "de medicion y no rechaza la imagen."
                  % "; ".join(offenders[:4]))
        score = max(score, NON_GENERATIVE_SCORE)
    elif passed:
        detail = ("Tus proporciones se mantienen respecto a la foto original "
                  "(%d medidas comparadas, la mas ajustada difiere un %d%% "
                  "sobre un limite del %d%%)."
                  % (compared, int(round(worst_dev * 100)),
                     int(round(worst_tol * 100))))
    else:
        detail = ("Cambiaron tus proporciones respecto a la foto original: %s."
                  % "; ".join(offenders[:4]))
    if widened:
        notes.append("Limite ajustado al ruido de medicion de esta pareja.")
    if too_noisy:
        notes.append("La medicion de esta pareja es demasiado ruidosa para "
                     "afinar mas; el limite se queda en el %d%% maximo."
                     % int(round(PAIRED_TOL_MAX * 100)))
    if notes:
        detail += " " + " ".join(notes)
    check = _mk(name_body(), worst_dev, worst_tol, passed, detail)
    check["mode"] = "pareado"
    check["compared"] = compared
    if degraded:
        check["advisory"] = True
    return check, score, True, offenders


def name_body() -> str:
    return "body_proportions"


def _usable(body: dict) -> set[str]:
    """Metrics from one measurement that may be trusted, yaw correction included."""
    metrics = {str(k) for k in (body.get("metrics") or {})}
    unreliable = _unreliable_metrics(body)
    return metrics - unreliable


def _check_body(gen_body: dict, profile: dict,
                brief: dict | None = None) -> tuple[dict, float, bool, list[str]]:
    name = "body_proportions"
    bands = profile.get("body") or {}
    metrics = (gen_body.get("metrics") or {}) if gen_body.get("ok") else {}
    unreliable = _unreliable_metrics(gen_body) if gen_body.get("ok") else set()
    corrected = ({str(v) for v in (gen_body.get("corrected") or [])}
                 if gen_body.get("ok") else set())

    offenders: list[str] = []       # gated, reliable, out of band -> these fail
    notes: list[str] = []           # reported but never penalised
    gated_total = 0
    gated_ok = 0
    closeness: list[float] = []

    for metric, band in bands.items():
        value = _f(metrics.get(metric), None)
        mean = _f((band or {}).get("mean"), None)
        lo = _f((band or {}).get("lo"), None)
        hi = _f((band or {}).get("hi"), None)
        if value is None or mean is None or lo is None or hi is None or abs(mean) < 1e-9:
            continue
        # A band the profile itself marked ungated (too few usable photos to be
        # anything but a guess) reports but never rejects.
        gated = (metric in GATED_METRICS and metric not in unreliable
                 and bool((band or {}).get("gated", True)))
        if metric in corrected:
            # Pay for the yaw correction's own error before judging anyone.
            half = (hi - lo) / 2.0
            lo, hi = mean - half * YAW_BAND_SLACK, mean + half * YAW_BAND_SLACK
        inside = lo <= value <= hi
        if gated:
            gated_total += 1
            gated_ok += 1 if inside else 0
            tol = max((hi - lo) / 2.0, 1e-6)
            closeness.append(_clamp01(1.0 - abs(value - mean) / tol))
        if inside:
            continue
        label, down, up = METRIC_ES.get(metric, (metric, "menor", "mayor"))
        pct = int(round(abs((value - mean) / abs(mean)) * 100.0))
        phrase = "%s un %d%% %s que en tus fotos" % (label, pct,
                                                     down if value < mean else up)
        if gated:
            offenders.append(phrase)
        elif metric in unreliable:
            notes.append(phrase + " (medida poco fiable, no penaliza)")
        else:
            notes.append(phrase + " (medida informativa, no penaliza)")

    if gated_total == 0:
        detail = "No se pudieron medir proporciones comparables, comprobacion omitida."
        if notes:
            detail += " Observado: " + "; ".join(notes[:4]) + "."
        return _mk(name, 1.0, 1.0, True, detail), 1.0, False, []

    inside_ratio = gated_ok / float(gated_total)
    # Same reasoning as the paired check: an engine that only composites,
    # recolours and crops cannot have moved a shoulder, so a band it falls
    # outside of is describing the measurement, not the woman.
    degraded = bool(offenders) and not _is_generative(brief or {})
    passed = (not offenders) or degraded
    if degraded:
        detail = ("Diferencia medida respecto a tus fotos: %s. Este motor no "
                  "genera cuerpo (solo cambia fondo, color, luz y encuadre), "
                  "asi que la diferencia es ruido de medicion y no rechaza la "
                  "imagen." % "; ".join(offenders[:4]))
    elif offenders:
        detail = ("Tu cuerpo cambio: %s. Comparadas %d medidas principales."
                  % ("; ".join(offenders[:4]), gated_total))
    else:
        detail = ("Proporciones dentro de tu rango medido (%d medidas comparadas)."
                  % gated_total)
    if notes:
        detail += " Ademas: " + "; ".join(notes[:3]) + "."
    score = float(np.mean(closeness)) if closeness else inside_ratio
    if degraded:
        score = max(score, NON_GENERATIVE_SCORE)
    check = _mk(name, inside_ratio, 1.0, passed, detail)
    if degraded:
        check["advisory"] = True
    return check, _clamp01(score), True, offenders


def _check_skin(gen_skin: dict, profile: dict,
                thresholds: dict) -> tuple[dict, float, bool]:
    name = "skin_tone"
    delta_max = _f(thresholds.get("delta_e_max"), 8.0)
    reference = _flist((profile.get("skin") or {}).get("lab_mean"), 3)
    generated = _flist(gen_skin.get("lab_mean"), 3) if gen_skin.get("ok") else None
    if reference is None or generated is None:
        return (_mk(name, 0.0, delta_max, True,
                    "No se pudo medir el tono de piel en ambas imagenes, "
                    "comprobacion omitida."), 1.0, False)
    prof_skin = profile.get("skin") or {}
    comparison = _safe(skin_mod.compare_skin, gen_skin,
                       {"ok": True, "lab_mean": reference,
                        "lab_std": _flist(prof_skin.get("lab_std"), 3) or [0.0, 0.0, 0.0],
                        "ita_deg": _f(prof_skin.get("ita_deg"), 0.0)})
    if isinstance(comparison, dict) and _f(comparison.get("delta_e"), None) is not None:
        delta_e = _f(comparison.get("delta_e"))
        similarity = _f(comparison.get("similarity"), None)
    else:  # CIE76 fallback keeps the check alive if compare_skin is unavailable
        delta_e = float(np.sqrt(sum((a - b) ** 2 for a, b in zip(generated, reference))))
        similarity = None
    if similarity is None:
        similarity = _clamp01(1.0 - delta_e / 25.0)

    # The decision is taken on chroma, not on the full CIE76 distance.  Exposure
    # and white balance move L far more than a real change of skin tone moves a
    # and b, and gating on L rejects the person's own untouched photographs.
    # Limits come from the profile, widened to that person's own spread.
    chroma_max = _f(thresholds.get("chroma_max_eff"), None)
    if chroma_max is None:
        chroma_max = _f(thresholds.get("chroma_max"), 6.0)
    lightness_max = _f(thresholds.get("delta_l_max_eff"), None)
    if lightness_max is None:
        lightness_max = _f(thresholds.get("delta_l_max"), 22.0)

    d_chroma = float(np.sqrt((generated[1] - reference[1]) ** 2
                             + (generated[2] - reference[2]) ** 2))
    d_light = abs(float(generated[0] - reference[0]))
    passed = d_chroma <= chroma_max and d_light <= lightness_max

    if passed:
        detail = ("Tono de piel practicamente igual (color %.1f de %.1f permitido, "
                  "luminosidad %.1f de %.1f)." % (d_chroma, chroma_max, d_light,
                                                  lightness_max))
    elif d_chroma > chroma_max:
        detail = ("Cambio el color de tu piel (%.1f, el maximo para ti es %.1f). "
                  "Esto no es cuestion de luz: es el tono."
                  % (d_chroma, chroma_max))
    else:
        detail = ("Te aclararon u oscurecieron la piel (%.1f de luminosidad, "
                  "maximo %.1f)." % (d_light, lightness_max))
    # The reported value stays delta E so the report page keeps one familiar
    # number, but the threshold shown is the one actually applied.
    check = _mk(name, round(d_chroma, 3), round(chroma_max, 3), passed, detail)
    check["delta_e"] = round(delta_e, 3)
    check["delta_l"] = round(d_light, 3)
    return check, _clamp01(similarity), True


def _check_anatomy(anomalies: dict, defects: list[dict]) -> tuple[dict, float, bool]:
    name = "anatomy"
    if not anomalies.get("ok"):
        return (_mk(name, 0.0, ANATOMY_SEVERITY_MAX, True,
                    "No se pudo revisar la anatomia, comprobacion omitida."), 1.0, False)
    worst = max([_f(d.get("severity")) for d in defects], default=0.0)
    passed = worst < ANATOMY_SEVERITY_MAX
    if not defects:
        detail = "Sin anomalias anatomicas detectadas."
    else:
        listed = sorted(defects, key=lambda d: -_f(d.get("severity")))[:3]
        parts = ["%s (%s)" % (DEFECT_ES.get(d["type"], d["type"]),
                              _severity_es(_f(d.get("severity")))) for d in listed]
        detail = "Se detectaron %d problemas: %s." % (len(defects), ", ".join(parts))
    return (_mk(name, worst, ANATOMY_SEVERITY_MAX, passed, detail),
            _clamp01(1.0 - worst), True)


def _check_quality(qual: dict) -> tuple[dict, float, bool]:
    name = "quality"
    if not qual.get("ok"):
        return (_mk(name, QUALITY_MIN, QUALITY_MIN, True,
                    "No se pudo evaluar la calidad tecnica, comprobacion omitida."),
                1.0, False)
    score = _clamp01(qual.get("score"))
    passed = score >= QUALITY_MIN
    detail = "Calidad tecnica %.2f (minimo %.2f)." % (score, QUALITY_MIN)
    issues = [str(i) for i in (qual.get("issues") or [])][:3]
    if issues and not passed:
        detail += " Problemas: " + ", ".join(issues) + "."
    if qual.get("beauty_filter_suspected"):
        detail += " Se sospecha filtro de belleza sobre la piel."
    return _mk(name, score, QUALITY_MIN, passed, detail), score, True


# ------------------------------------------------------------------- verdict

def _join_es(items: list[str]) -> str:
    if not items:
        return ""
    if len(items) == 1:
        return items[0]
    return ", ".join(items[:-1]) + " y " + items[-1]


def _summary(passed: bool, checks: list[dict], repairable: list[dict],
             body_offenders: list[str], skipped: list[str]) -> str:
    """One or two sentences the client will actually read."""
    if passed:
        # A check that only reported - because the engine could not have caused
        # what it measured - must not be sold to her as a confirmation.  It
        # passed; it did not verify anything, and the summary says so by leaving
        # it out of the list while its own detail explains what was seen.
        advisory = {c["name"] for c in checks if c.get("advisory")}
        verified = [CHECK_ES[n] for n in ("identity_face", "body_proportions",
                                          "skin_tone")
                    if n not in skipped and n not in advisory]
        if verified:
            text = ("Aprobada: %s %s con tus fotos."
                    % (_join_es(verified),
                       "coincide" if len(verified) == 1 else "coinciden"))
        else:
            text = "Aprobada, aunque no se pudo comparar tu identidad en esta imagen."
        if skipped:
            text += (" No se pudo comprobar %s."
                     % _join_es([CHECK_ES[s] for s in skipped if s in CHECK_ES]))
        elif repairable:
            text += (" Quedan detalles menores que se pueden retocar sin volver "
                     "a generar la imagen.")
        return text
    reasons: list[str] = []
    for check in checks:
        if check["passed"]:
            continue
        reason = FAIL_ES.get(check["name"], check["name"])
        if check["name"] == "body_proportions" and body_offenders:
            reason += " (" + body_offenders[0] + ")"
        reasons.append(reason)
    text = "Rechazada: " + ", ".join(reasons) + "."
    text += (" Se puede reparar solo la zona afectada sin regenerar toda la foto."
             if repairable else " Conviene generar de nuevo con el ajuste corregido.")
    return text


def verify_image(image_path: str, profile: dict, brief: dict | None = None) -> dict:
    """Measure a generated image against a stored identity profile."""
    started = time.perf_counter()
    prof = profile if isinstance(profile, dict) else {}
    brf = brief if isinstance(brief, dict) else {}
    thresholds = _thresholds(prof)

    img = _safe(loader.load_image, str(image_path), VERIFY_MAX_SIDE)
    if not isinstance(img, np.ndarray) or img.size == 0:
        return {
            "passed": False, "score": 0.0,
            "checks": [_mk("quality", 0.0, QUALITY_MIN, False,
                           "No se pudo abrir el archivo generado.")],
            "defects": [], "repairable_defects": [],
            "summary": "No se pudo leer la imagen generada, no hay nada que verificar.",
            "elapsed_ms": _ms(started),
        }

    pose_d = _ok_dict(_safe(pose_mod.detect_pose, img), "pose no disponible")
    face_d = _ok_dict(_safe(face_mod.detect_face, img), "rostro no detectado")
    seg = _ok_dict(_safe(segment_mod.person_mask, img), "segmentacion no disponible")
    person_raw = seg.get("mask") if seg.get("ok") else None
    person_arg = person_raw if isinstance(person_raw, np.ndarray) else None
    regions = _safe(segment_mod.region_masks, img, pose_d, person_arg)
    if not isinstance(regions, dict):
        regions = {}
    masks = dict(regions)
    if person_arg is not None:
        masks.setdefault("person", person_arg)

    body_d = _ok_dict(_safe(body_mod.measure_body, img, pose_d, person_arg),
                      "medidas no disponibles")
    skin_d = _ok_dict(_safe(skin_mod.skin_stats, img, pose_d, face_d), "piel no medible")
    qual_d = _ok_dict(_safe(quality_mod.assess_quality, img, str(image_path)),
                      "calidad no evaluable")
    anom_d = _ok_dict(_safe(anomaly_mod.scan_anomalies, img, pose_d, face_d, masks),
                      "anomalias no evaluadas")

    scan_defects = [d for d in (_defect(x) for x in (anom_d.get("defects") or []))
                    if d is not None]
    identity_defects: list[dict] = []

    checks: list[dict] = []
    scored: list[tuple[float, float]] = []   # (weight, normalised score), computed only
    skipped: list[str] = []                  # measured nothing -> never a failure

    face_check, face_score, face_done = _check_identity_face(
        img, face_d, prof, thresholds, brf, identity_defects)
    # Prefer the paired test against the source photograph; the profile bands are
    # the fallback for when the caller did not say what this was made from.
    # The noise floor is estimated at the size the result actually has, so the
    # comparison and its error bar are read off the same scale.
    gen_side = int(max(img.shape[0], img.shape[1]))
    src_body, src_side = _source_body(brf)
    paired = _check_body_paired(body_d, src_body, thresholds, brf,
                                gen_side, src_side)
    if paired is not None:
        body_check, body_score, body_done, offenders = paired
    else:
        body_check, body_score, body_done, offenders = _check_body(body_d, prof, brf)
    skin_check, skin_score, skin_done = _check_skin(skin_d, prof, thresholds)
    anat_check, anat_score, anat_done = _check_anatomy(anom_d, scan_defects)
    qual_check, qual_score, qual_done = _check_quality(qual_d)

    for chk, chk_score, computed in ((face_check, face_score, face_done),
                                     (body_check, body_score, body_done),
                                     (skin_check, skin_score, skin_done),
                                     (anat_check, anat_score, anat_done),
                                     (qual_check, qual_score, qual_done)):
        checks.append(chk)
        if computed:
            scored.append((chk["weight"], chk_score))
        else:
            skipped.append(chk["name"])

    # The scan may already have reported a distorted face; do not say it twice.
    if any(d["type"] == "face_distorted" for d in scan_defects):
        identity_defects = [d for d in identity_defects if d["type"] != "face_distorted"]
    defects = [d for d in (_defect(x) for x in identity_defects) if d is not None]
    defects.extend(scan_defects)
    repairable = [d for d in defects if d["repairable"] and d["bbox"]]

    total_weight = sum(w for w, _ in scored)
    score = (sum(w * s for w, s in scored) / total_weight) if total_weight > 0 else 0.0
    passed = all(c["passed"] for c in checks if c["weight"] > 0)

    # "Never fail someone for what could not be measured" is right for one
    # missing metric and catastrophic as a blanket rule: an image damaged badly
    # enough that no face, no pose and no body can be found skips EVERY identity
    # check and sails through with a high score.  That is precisely the image
    # that must not reach the client.  So a subject that cannot be located at
    # all is a failure, not an abstention.
    lost_subject = (not face_d.get("ok")) and (not pose_d.get("ok")) \
        and (not body_d.get("ok"))
    identity_skipped = {"identity_face", "body_proportions", "skin_tone"} <= set(skipped)
    if lost_subject or (identity_skipped and not face_d.get("ok")):
        passed = False
        score = min(score, 0.2)
        for chk in checks:
            if chk["name"] == "identity_face":
                chk["passed"] = False
                chk["detail"] = ("No se encuentra a la persona en la imagen "
                                 "generada: no hay rostro ni cuerpo reconocibles, "
                                 "asi que no se puede garantizar que seas tu.")
        defects.append({
            "type": "face_distorted", "where": "image", "bbox": [],
            "severity": 0.95, "repairable": False,
            "detail": "sujeto irreconocible en la imagen generada",
        })

    return {
        "passed": bool(passed),
        "score": round(_clamp01(score), 4),
        "checks": checks,
        "defects": defects,
        "repairable_defects": repairable,
        "summary": _summary(passed, checks, repairable, offenders, skipped),
        "elapsed_ms": _ms(started),
    }
