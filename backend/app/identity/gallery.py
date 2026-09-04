"""Which of her photographs to use, and for what.

Until now a run picked ONE photograph and used it for everything: as the image
the engine edits, as the only reference for what she looks like, and as the
donor for every local correction.  Measured on this installation's own files,
that single choice is the most expensive decision in the product.

* **At generation.**  fal's Kontext multi endpoint accepts up to three
  reference images in ``image_urls`` and costs the SAME 0.040 USD as the single
  image endpoint at preview, and HALF of the 0.080 USD ``identity_max`` at high
  and max.  Pooling references is not a luxury, it is cheaper.  It is also
  measurably better at saying who she is: her 24 photographs sit 0.6849..0.8879
  from the mean of all of them but 0.4125..0.9562 from each OTHER, so two of
  her own pictures can be further apart than a stranger is from the average.
  The worst-case margin (worst photograph of her minus best of eight other
  women) is 0.1985..0.4241 for a single reference and 0.4795 for the pooled
  mean - 62% wider - and the triple this rule chooses measured 0.5294.
* **At repair.**  Which photograph is the best donor depends on the defect, and
  the three answers disagree: the closest face size wins for nothing (2 of 30
  transfers), the lowest alignment residual wins the face, and the grain
  transfer only fires at all when retouch.py's own gates are applied at FULL
  resolution before ranking - a 3024x4032 closeup carries a 1043 px face
  against a 1024 px render's 129 px, which is a scale of 0.12 against
  retouch.SCALE_MIN 0.20, so the ranking must not be done on the analysis-scale
  face size.

Two traps in this data cost two selections before they were found, and both are
handled here rather than in the callers:

* ``face.detect_face`` returns yaw=pitch=roll=(0,0,0) when it could not build a
  mesh (IMG_8949 on this installation), which is a failure sentinel and not a
  frontal head; those readings never feed a choice.
* roll comes back near +/-180 degrees for an upright head.  Unwrapped it is the
  largest number in any distance, and it turned the "deliberately diverse"
  three into three closeups with no size diversity at all.

Every function degrades honestly.  Too few photographs, or no photograph that
suits the job, and the answer is empty with a Spanish sentence saying so; the
caller then does without instead of being handed a donor that will make the
picture worse.
"""
from __future__ import annotations

import itertools
import math
from typing import Any

import cv2
import numpy as np

from .. import db
from ..analysis import face as face_mod
from ..analysis import loader
from ..analysis import pose as pose_mod
from . import embedding as embedding_mod

# ------------------------------------------------------------------ constants

# Readings are taken at the resolution every other measurement in the product
# uses, so a face width measured here means the same thing as one measured by
# verify or by the profile builder.
READ_MAX_SIDE = 1600

# The cache lives in the ``meta`` table, one row per photograph, and not inside
# ``originals.analysis_json``: that column is rewritten wholesale by
# ``orchestrator.analyse_original`` and a second writer would silently lose the
# readings of whichever of the two saved first.
CACHE_PREFIX = "gallery:"
CACHE_VERSION = 3

# A reference the engine cannot read is not a reference.  Her smallest face is
# 130 px at 1600 px analysis scale (IMG_8898), and that photograph is what
# carries the outdoor half of her gallery and her body proportions, so the
# floor is set just below it deliberately - but whether Kontext can copy a face
# from a 130 px reference is the one thing in this module that cannot be
# measured without spending, and the paid check is what settles it.
MIN_REF_FACE_PX = 120.0

# Her own photographs measured against the mean of the OTHER 23: 0.6565..0.8763,
# median 0.8218.  One of them sits far enough out to be a liability as a
# reference (0.6565).  The floor is relative as well as absolute so it keeps
# meaning something on somebody else's gallery: a photograph is dropped only
# when it is both under the absolute floor and clearly under the rest of hers.
LOO_ABS_FLOOR = 0.68
LOO_REL_DROP = 0.12                 # below (median - this) is an outlier

# "Diverse" has to be measurable or it is decoration.  Her gallery spans a
# factor of 4.16 in face size (130..541 px) and is two clusters, not a spread:
# 11 warm indoor closeups at 387-541 px and a cooler outdoor batch at
# 130-252 px, with nothing between 252 and 344.  A trio drawn from one cluster
# describes one distance and one light.
REF_SIZE_SPREAD_MIN = 1.8           # max face px / min face px inside the trio
REF_SHOT_TYPES_MIN = 2              # closeup / half / full: at least two

# Face graft donors.  A similarity transform cannot correct yaw, and her
# library spans -41.3..+35.4 degrees, so a donor whose head is turned
# differently is not a donor at all whatever its landmarks measure.
FACE_YAW_TOL = 8.0
FACE_ROLL_TOL = 8.0
# Every measured residual was 0.017..0.061 against retouch.py's 0.12 limit, so
# this gate is deliberately tighter than retouch's: alignment was never the
# bottleneck, and half of retouch's limit still admits every donor that ever
# won its comparison.
FACE_RESIDUAL_MAX = 0.06

# Skin donors.  These are retouch.py's own gates, applied here at FULL
# resolution before ranking, because that is the resolution
# restore_skin_texture reads its two files at.  Ranked without them the rule
# fires on 2 of 30 files; with them, and a fall-through on refusal, the same
# corpus reaches 18-19 of 30 - one short of the oracle.
SKIN_SCALE_MIN, SKIN_SCALE_MAX = 0.20, 5.00
SKIN_MAX_CANDIDATES = 10            # median 3 probes, never more than 8 needed

# Body consensus.  A global median across every measurable photograph spreads
# 47-66% and that spread is framing, not her: conditioned on shot type the same
# shoulder ratio falls from 66% to 10%.  So the consensus is taken inside a
# shot type whenever there are enough photographs of it.
BODY_MIN_SAMPLES = 3
BODY_METRICS = ("shoulder_w_over_torso", "hip_w_over_torso",
                "waist_w_over_torso", "bust_w_over_torso",
                "shoulder_over_hip", "arm_len_over_torso")

# Hand donors.  Only 4 of her 24 photographs have both a torso frame and hands
# the anomaly scan reads as clean (IMG_8918, IMG_8944, IMG_8946, IMG_8947);
# the unconstrained nearest donor is 25% closer but hands over IMG_8825's own
# 0.57 hand, so the clean rule costs +38% distance and is the only one that
# cannot make the picture worse.  Under 0.15 Procrustes distance the two hands
# are held the same way; only 2 of 7 measured defects found such a donor.
HAND_CLEAN_MAX = 0.30               # her own clean photographs read 0.000
HAND_MATCH_MAX = 0.15
_HAND_POINTS = ("wrist", "thumb", "index", "pinky")

_PURPOSES = ("face", "skin", "body", "hands")


# -------------------------------------------------------------------- helpers

def _f(value: Any, default: float = 0.0) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return default
    if out != out or math.isinf(out):
        return default
    return out


def _wrap_roll(angle: Any) -> float:
    """Roll folded into +/-90 degrees.

    ``detect_face`` reports an upright head as +/-175 degrees on this
    installation.  Left unwrapped it is the largest number in any distance and
    it hijacked two reference selections before it was found.
    """
    return float(((_f(angle) + 90.0) % 180.0) - 90.0)


def _pose_ok(face: dict) -> bool:
    """False for the (0, 0, 0) sentinel ``_head_pose`` returns with no mesh."""
    if not isinstance(face, dict) or not face.get("mesh"):
        return False
    return not (_f(face.get("yaw")) == 0.0 and _f(face.get("pitch")) == 0.0
                and _f(face.get("roll")) == 0.0)


def _cache_get(original_id: str) -> dict:
    row = db.q1("SELECT value FROM meta WHERE key=?",
                (CACHE_PREFIX + str(original_id),))
    got = db.loads(row["value"], {}) if row else {}
    if not isinstance(got, dict) or int(_f(got.get("v"))) != CACHE_VERSION:
        return {}
    return got


def _cache_put(original_id: str, payload: dict) -> None:
    payload = dict(payload)
    payload["v"] = CACHE_VERSION
    db.execute("INSERT OR REPLACE INTO meta(key,value) VALUES(?,?)",
               (CACHE_PREFIX + str(original_id), db.dumps(payload)))


def _rows_for(profile: dict) -> list[dict]:
    """Her photographs: everything of hers this user has not deleted.

    Deliberately NOT restricted to the originals tagged with this profile.  On
    this installation 20 of her 24 photographs carry the tag and four do not -
    they were imported after the profile was built - and those four include the
    413 px closeup that competes for a reference slot.  A photograph tagged
    with ANOTHER profile does belong to another person and is excluded; an
    untagged one belongs to nobody else, and throwing away a sixth of the
    evidence because of an import order is not a measurement.
    """
    prof = profile if isinstance(profile, dict) else {}
    user_id = str(prof.get("user_id") or "")
    if not user_id:
        return []
    profile_id = str(prof.get("id") or "")
    return db.rows_to_dicts(db.q(
        "SELECT id, path, filename, shot_type FROM originals "
        "WHERE user_id=? AND deleted_at IS NULL "
        "AND (profile_id IS NULL OR profile_id=?) "
        "ORDER BY sort_order, created_at", (user_id, profile_id)))


# ------------------------------------------------------------------- readings

def _grain(img: np.ndarray, bbox: Any) -> float:
    """Fine texture of the facial skin, read at a fixed face width.

    The fine band is not a property of skin: over her 24 photographs it tracks
    how wide the face happens to be (Pearson r = 0.89, identity/verify), so
    comparing donors by their raw amplitude compares their framing.  The crop is
    normalised to the same 520 px face retouch.py normalises to and read at the
    same sigma 1.4.
    """
    try:
        box = [float(v) for v in list(bbox or [])[:4]]
    except (TypeError, ValueError):
        return 0.0
    if len(box) != 4 or box[2] < 8 or box[3] < 8:
        return 0.0
    height, width = img.shape[:2]
    x0 = int(max(0, box[0] + 0.20 * box[2]))
    y0 = int(max(0, box[1] + 0.30 * box[3]))
    x1 = int(min(width, box[0] + 0.80 * box[2]))
    y1 = int(min(height, box[1] + 0.85 * box[3]))
    if x1 - x0 < 12 or y1 - y0 < 12:
        return 0.0
    crop = img[y0:y1, x0:x1]
    factor = 520.0 / float(box[2])
    if factor < 1.0:
        crop = cv2.resize(crop, None, fx=factor, fy=factor,
                          interpolation=cv2.INTER_AREA)
    luma = cv2.cvtColor(crop, cv2.COLOR_BGR2LAB)[:, :, 0].astype(np.float32)
    band = luma - cv2.GaussianBlur(luma, (0, 0), 1.4)
    return round(float(np.std(band)), 4)


def _face_reading(path: str) -> dict:
    """Everything about one photograph a choice can be made on."""
    out: dict[str, Any] = {"ok": False, "reason": "", "path": str(path)}
    full_side = 0.0
    try:
        info = loader.image_info(path) or {}
        full_side = float(max(int(info.get("width") or 0),
                              int(info.get("height") or 0)))
    except Exception:                                     # noqa: BLE001
        full_side = 0.0
    try:
        img = loader.load_image(str(path), READ_MAX_SIDE)
    except Exception as exc:                              # noqa: BLE001
        out["reason"] = "no se pudo abrir la foto: %s" % str(exc)[:80]
        return out
    if not isinstance(img, np.ndarray) or img.size == 0:
        out["reason"] = "foto no utilizable"
        return out
    height, width = img.shape[:2]
    read_side = float(max(height, width))
    face = face_mod.detect_face(img)
    if not face.get("ok"):
        out["reason"] = "no se detecto el rostro"
        return out
    bbox = [_f(v) for v in list(face.get("bbox") or [0, 0, 0, 0])[:4]]
    quality: dict = {}
    try:
        from ..analysis import quality as quality_mod
        quality = quality_mod.assess_quality(img, str(path), face) or {}
    except Exception:                                     # noqa: BLE001
        quality = {}
    embedding = list(embedding_mod.face_embedding(img, face) or [])
    out.update({
        "ok": True,
        "width": int(width), "height": int(height),
        # Full resolution over analysis resolution: retouch.py reads both of
        # its files at full size, so its scale gate has to be evaluated there.
        "full_factor": round(full_side / read_side, 4) if read_side else 1.0,
        "bbox": [round(v, 2) for v in bbox],
        "face_px": round(bbox[2], 1),
        "yaw": round(_f(face.get("yaw")), 2),
        "pitch": round(_f(face.get("pitch")), 2),
        "roll": round(_wrap_roll(face.get("roll")), 2),
        "pose_ok": _pose_ok(face),
        "mesh": face.get("mesh") or [],
        "landmarks": face.get("landmarks") or {},
        "beauty": bool(quality.get("beauty_filter_suspected")),
        "quality": round(_f(quality.get("score")), 4),
        "grain": _grain(img, bbox),
        "embedding": ([round(float(v), 6) for v in embedding]
                      if len(embedding) == embedding_mod.DIMS else []),
    })
    return out


def _body_reading(path: str) -> dict:
    """Her proportions in this photograph, measured the way verify measures."""
    from ..analysis import body as body_mod
    from ..analysis import segment as segment_mod

    out: dict[str, Any] = {"ok": False, "metrics": {}, "confidence": 0.0}
    try:
        img = loader.load_image(str(path), READ_MAX_SIDE)
    except Exception:                                     # noqa: BLE001
        return out
    if not isinstance(img, np.ndarray) or img.size == 0:
        return out
    pose_d = pose_mod.detect_pose(img)
    face_d = face_mod.detect_face(img)
    seg = segment_mod.person_mask(img, pose_d)
    mask = seg.get("mask") if seg.get("ok") else None
    mask = mask if isinstance(mask, np.ndarray) else None
    body = body_mod.measure_body(img, pose_d, mask, face_d)
    if not isinstance(body, dict) or not body.get("ok"):
        return out
    metrics = {str(k): _f(v) for k, v in (body.get("metrics") or {}).items()
               if _f(v) > 0.0}
    out.update({"ok": True, "metrics": metrics,
                "confidence": round(_f(body.get("confidence")), 4),
                "unreliable": [str(u) for u in (body.get("unreliable") or [])]})
    return out


def _hand_points(pose_d: dict) -> dict:
    """The two hands as four points each, normalised by the torso length.

    Normalising is what makes two photographs comparable at all: her hand is
    150 px across in a closeup and 30 px in a full length frame, and a raw
    distance would rank donors by how far away the camera was.
    """
    landmarks = (pose_d or {}).get("landmarks") or {}

    def point(name: str):
        got = landmarks.get(name)
        if not isinstance(got, dict) or _f(got.get("v")) < 0.4:
            return None
        return (_f(got.get("x")), _f(got.get("y")))

    shoulders = [point("left_shoulder"), point("right_shoulder")]
    hips = [point("left_hip"), point("right_hip")]
    if any(p is None for p in shoulders + hips):
        return {}
    mid_sh = ((shoulders[0][0] + shoulders[1][0]) / 2.0,
              (shoulders[0][1] + shoulders[1][1]) / 2.0)
    mid_hip = ((hips[0][0] + hips[1][0]) / 2.0, (hips[0][1] + hips[1][1]) / 2.0)
    torso = math.hypot(mid_sh[0] - mid_hip[0], mid_sh[1] - mid_hip[1])
    if torso < 1e-4:
        return {}
    out: dict[str, list] = {}
    for side in ("left", "right"):
        names = ["%s_%s" % (side, key) for key in _HAND_POINTS]
        points = [point(name) for name in names]
        if any(p is None for p in points):
            continue
        out[side] = [[round((p[0] - mid_sh[0]) / torso, 4),
                      round((p[1] - mid_sh[1]) / torso, 4)] for p in points]
    return out


def _hands_reading(path: str) -> dict:
    """Her hands in this photograph, and how clean the scanner says they are."""
    from ..analysis import anomaly as anomaly_mod
    from ..analysis import segment as segment_mod

    out: dict[str, Any] = {"ok": False, "hands": {}, "severity": 1.0}
    try:
        img = loader.load_image(str(path), READ_MAX_SIDE)
    except Exception:                                     # noqa: BLE001
        return out
    if not isinstance(img, np.ndarray) or img.size == 0:
        return out
    pose_d = pose_mod.detect_pose(img)
    face_d = face_mod.detect_face(img)
    seg = segment_mod.person_mask(img, pose_d)
    mask = seg.get("mask") if seg.get("ok") else None
    regions = segment_mod.region_masks(
        img, pose_d, mask if isinstance(mask, np.ndarray) else None) or {}
    scan = anomaly_mod.scan_anomalies(img, pose_d, face_d, regions) or {}
    worst = 0.0
    for defect in (scan.get("defects") or []):
        if str(defect.get("type")) == "hand_malformed":
            worst = max(worst, _f(defect.get("severity")))
    out.update({"ok": bool(pose_d.get("ok")), "hands": _hand_points(pose_d),
                "severity": round(worst, 4)})
    return out


def reading(row: dict, need: tuple[str, ...] = ("face",)) -> dict:
    """One photograph's cached reading, computing only the parts asked for.

    The face block is about 0.45 s of work; body and hands are several seconds
    each, so they are measured only when a body or a hand correction actually
    asks for them, and never on the generation path.
    """
    original_id = str(row.get("id") or "")
    path = str(row.get("path") or "")
    cached = _cache_get(original_id) if original_id else {}
    changed = False
    for key, fn in (("face", _face_reading), ("body", _body_reading),
                    ("hands", _hands_reading)):
        if key in need and key not in cached:
            cached[key] = fn(path)
            changed = True
    if changed and original_id:
        _cache_put(original_id, cached)
    out = {"id": original_id, "path": path,
           "filename": str(row.get("filename") or ""),
           "shot_type": str(row.get("shot_type") or "unknown")}
    for key in ("face", "body", "hands"):
        if key in cached:
            out[key] = cached[key]
    return out


def photographs(profile: dict, need: tuple[str, ...] = ("face",)) -> list[dict]:
    """Her whole gallery, read once and cached."""
    return [reading(row, need) for row in _rows_for(profile)]


# ------------------------------------------------------------------ selection

def _cos(a: Any, b: Any) -> float:
    va = np.asarray(list(a), dtype=np.float64)
    vb = np.asarray(list(b), dtype=np.float64)
    if va.shape != vb.shape or va.size == 0:
        return 0.0
    na, nb = float(np.linalg.norm(va)), float(np.linalg.norm(vb))
    if na < 1e-9 or nb < 1e-9:
        return 0.0
    return float(np.dot(va, vb) / (na * nb))


def _leave_one_out(vectors: list[list[float]]) -> list[float]:
    """Each photograph against the mean of the OTHERS.

    Against the mean of ALL of them a photograph is compared with something it
    is itself part of, which flatters exactly the outlier this looks for.
    """
    if len(vectors) < 3:
        return [1.0] * len(vectors)
    matrix = np.asarray(vectors, dtype=np.float64)
    total = matrix.sum(axis=0)
    out: list[float] = []
    for i in range(len(vectors)):
        rest = (total - matrix[i]) / float(len(vectors) - 1)
        out.append(_cos(matrix[i], rest))
    return out


def choose_references(profile: dict, n: int = 3, must_include: str = "") -> dict:
    """Up to ``n`` deliberately different photographs of her, by measurement.

    ``must_include`` is the photograph the run is editing.  It travels to the
    engine anyway - Kontext multi puts the source first in ``image_urls`` and
    accepts three images in all - so the honest question is not "which three
    photographs" but "which two go WITH this one", and forcing it into every
    combination is what asks that question.  Sending it a second time as its
    own reference, which is what the code did before, buys nothing at all.

    The rule, in order:

    1. A candidate must have a face SFace can read, a face at least
       MIN_REF_FACE_PX wide, a head pose that is not the failure sentinel, and
       no beauty filter - two of her photographs carry one, and a reference of
       a smoothed face teaches the engine the smoothing.
    2. A candidate must not be her gallery's own outlier: the photograph that
       sits furthest from the other 23 makes the pooled reference worse rather
       than more diverse.
    3. Among every combination of the survivors, the one whose mean best covers
       her WORST photograph wins - that minimum is what the identity gate reads
       on a bad day, and it is what the margin against other women is made of.
       Ties go to the most different trio, and a trio is only accepted when it
       really spans her gallery: two shot types and a face-size spread.

    Returns ``{"paths", "reason", "detail"}``.  ``paths`` is empty when she has
    too few usable photographs, and the caller then generates without
    references exactly as it did before - which is honest, and the same
    0.040 USD either way.
    """
    n = max(1, int(n or 1))
    rows = photographs(profile, need=("face",))
    usable = [r for r in rows if (r.get("face") or {}).get("ok")
              and (r["face"].get("embedding") or [])]
    if len(usable) < 2:
        return {"paths": [], "detail": {"medibles": len(usable)},
                "reason": ("No hay suficientes fotos tuyas medibles para "
                           "usarlas como referencia (%d)." % len(usable))}

    vectors = [r["face"]["embedding"] for r in usable]
    loo = _leave_one_out(vectors)
    median_loo = float(np.median(loo)) if loo else 1.0
    candidates: list[dict] = []
    dropped: list[str] = []
    for row, own in zip(usable, loo):
        face = row["face"]
        why = ""
        if face.get("beauty"):
            why = "filtro de belleza"
        elif _f(face.get("face_px")) < MIN_REF_FACE_PX:
            why = "rostro de %d px" % int(_f(face.get("face_px")))
        elif not face.get("pose_ok"):
            why = "sin malla facial"
        elif own < LOO_ABS_FLOOR and own < median_loo - LOO_REL_DROP:
            why = "se parece poco al resto de tus fotos (%.2f)" % own
        if why:
            dropped.append("%s: %s" % (row.get("filename") or row["path"], why))
            continue
        candidates.append({**row, "loo": round(own, 4)})

    forced = str(must_include or "")
    forced_index = -1
    if forced:
        for index, row in enumerate(candidates):
            if str(row.get("path")) == forced:
                forced_index = index
                break
        if forced_index < 0:
            # The photograph being edited is used whether or not it would have
            # been chosen as a reference: the engine is looking at it anyway.
            for row in usable:
                if str(row.get("path")) == forced:
                    candidates.append({**row, "loo": 0.0})
                    forced_index = len(candidates) - 1
                    break

    if not candidates:
        return {"paths": [], "detail": {"descartadas": dropped},
                "reason": ("Ninguna de tus fotos sirve como referencia: %s."
                           % "; ".join(dropped[:3]))}
    # Fewer usable photographs than references asked for is not a failure: two
    # good references still pool better than one.
    n = min(n, len(candidates))

    best = None
    best_relaxed = None
    for combo in itertools.combinations(range(len(candidates)), n):
        if forced_index >= 0 and forced_index not in combo:
            continue
        picked = [candidates[i] for i in combo]
        mean = embedding_mod.gallery_mean(
            [p["face"]["embedding"] for p in picked])
        if not mean:
            continue
        coverage = min(_cos(v, mean) for v in vectors)
        inner = [_cos(picked[a]["face"]["embedding"],
                      picked[b]["face"]["embedding"])
                 for a in range(len(picked)) for b in range(a + 1, len(picked))]
        diversity = 1.0 - (sum(inner) / len(inner)) if inner else 0.0
        sizes = [_f(p["face"].get("face_px")) for p in picked]
        spread = (max(sizes) / max(1.0, min(sizes))) if sizes else 1.0
        shots = len({p.get("shot_type") for p in picked})
        row = {"combo": picked, "coverage": coverage, "diversity": diversity,
               "spread": spread, "shots": shots}
        key = (round(coverage, 4), round(diversity, 4))
        if best_relaxed is None or key > (round(best_relaxed["coverage"], 4),
                                          round(best_relaxed["diversity"], 4)):
            best_relaxed = row
        if n >= 2 and (shots < min(REF_SHOT_TYPES_MIN, n)
                       or spread < REF_SIZE_SPREAD_MIN):
            continue
        if best is None or key > (round(best["coverage"], 4),
                                  round(best["diversity"], 4)):
            best = row

    relaxed = best is None
    chosen = best or best_relaxed
    if chosen is None:
        return {"paths": [], "detail": {},
                "reason": "No se pudo comparar tus fotos entre si."}

    picked = chosen["combo"]
    names = [p.get("filename") or p["path"] for p in picked]
    sizes = [int(_f(p["face"].get("face_px"))) for p in picked]
    reason = ("Se envian %d fotos tuyas como referencia (%s): cubren tu peor "
              "foto al %.2f y son distintas entre si (rostro de %d a %d px, "
              "%d tipos de plano)."
              % (len(picked), ", ".join(names), chosen["coverage"],
                 min(sizes), max(sizes), chosen["shots"]))
    if relaxed:
        reason += (" No hay %d fotos tuyas con encuadres y tamanos distintos, "
                   "asi que se han elegido las que mejor te representan aunque "
                   "se parezcan entre si." % n)
    return {
        "paths": [p["path"] for p in picked],
        "reason": reason,
        "detail": {
            "fotos": names,
            "cobertura": round(chosen["coverage"], 4),
            "diversidad": round(chosen["diversity"], 4),
            "rango_rostro": round(chosen["spread"], 2),
            "planos": chosen["shots"],
            "tamanos_px": sizes,
            "candidatas": len(candidates),
            "descartadas": dropped,
            "relajado": relaxed,
        },
    }


# --------------------------------------------------------------------- donors

def _generated_reading(generated_image: Any) -> dict:
    """The generated frame read exactly like one of her photographs."""
    if isinstance(generated_image, dict):
        return generated_image
    return _face_reading(str(generated_image))


def _residual(donor: dict, target: dict) -> tuple[float, float, float]:
    """Alignment of one photograph onto the generated frame.

    Reuses retouch.py's own geometry rather than a second copy of it: the
    number that decides which donor is best has to be the number the transfer
    will then be judged by.
    """
    from ..generation import retouch as retouch_mod

    face_s = {"mesh": donor.get("mesh") or [],
              "landmarks": donor.get("landmarks") or {},
              "bbox": donor.get("bbox") or []}
    face_g = {"mesh": target.get("mesh") or [],
              "landmarks": target.get("landmarks") or {},
              "bbox": target.get("bbox") or []}
    size_s = (int(_f(donor.get("width"))), int(_f(donor.get("height"))))
    size_g = (int(_f(target.get("width"))), int(_f(target.get("height"))))
    if min(size_s + size_g) <= 0:
        return 1.0, 0.0, 0.0
    src, dst, _kind = retouch_mod._correspondences(face_s, size_s,
                                                   face_g, size_g)
    if src is None:
        return 1.0, 0.0, 0.0
    interocular = retouch_mod._interocular(face_g, size_g[0], size_g[1])
    if interocular < 8.0:
        return 1.0, 0.0, 0.0
    matrix, inliers, residual, scale = retouch_mod._estimate(src, dst,
                                                             interocular)
    if matrix is None:
        return 1.0, 0.0, 0.0
    return residual / interocular, inliers, scale


def _full_scale(donor: dict, target: dict) -> float:
    """The scale retouch.py will compute when it reads both files FULL size.

    This is the gate that silently refused nearly every transfer: a 3024x4032
    photograph carries a 1043 px face against a 1024 px render's 129 px, and
    0.12 is below retouch.SCALE_MIN.  Ranked at analysis scale the number looks
    fine and the transfer then refuses; evaluated here, the donor is simply not
    offered.
    """
    donor_px = _f(donor.get("face_px")) * _f(donor.get("full_factor"), 1.0)
    target_px = _f(target.get("face_px")) * _f(target.get("full_factor"), 1.0)
    if donor_px < 1.0 or target_px < 1.0:
        return 0.0
    return target_px / donor_px


def _procrustes(a: Any, b: Any) -> float:
    """Distance between two hand shapes after translation, scale and rotation.

    What matters is whether the hand is held the same way, not where it is in
    the frame or how big it is.
    """
    pa = np.asarray(list(a), dtype=np.float64)
    pb = np.asarray(list(b), dtype=np.float64)
    if pa.shape != pb.shape or pa.ndim != 2 or pa.shape[0] < 3:
        return 9.9
    pa = pa - pa.mean(axis=0)
    pb = pb - pb.mean(axis=0)
    na, nb = float(np.linalg.norm(pa)), float(np.linalg.norm(pb))
    if na < 1e-6 or nb < 1e-6:
        return 9.9
    pa, pb = pa / na, pb / nb
    u, _s, vt = np.linalg.svd(pa.T @ pb)
    return float(np.linalg.norm(pa @ (u @ vt) - pb))


def choose_donor(profile: dict, generated_image: Any, purpose: str) -> dict:
    """The photograph of hers that best fixes THIS defect in THIS image.

    ``purpose`` is one of face | skin | body | hands, and each one has its own
    measured rule - they disagree, which is the whole reason for choosing per
    defect instead of using the run's source photograph for everything.

    Returns ``{"path", "alternatives", "reason", "detail"}``.  ``path`` is ""
    when nothing suits, and the caller must then leave the image alone: a donor
    that fails these gates makes the picture worse, not better.  ``alternatives``
    exists for the skin transfer, whose measured rule is "try them in order and
    keep the first the module itself accepts" (19 of 27 files, median 3 probes).
    """
    kind = str(purpose or "").lower()
    if kind not in _PURPOSES:
        return {"path": "", "alternatives": [], "detail": {},
                "reason": "proposito desconocido: %s" % purpose}
    if kind == "body":
        return _body_consensus(profile, generated_image)
    if kind == "hands":
        return _hand_donor(profile, generated_image)

    target = _generated_reading(generated_image)
    if not target.get("ok"):
        return {"path": "", "alternatives": [], "detail": {},
                "reason": "no se pudo leer el rostro de la imagen generada"}

    rows = photographs(profile, need=("face",))
    ranked: list[tuple] = []
    rejected: list[str] = []
    for row in rows:
        face = row.get("face") or {}
        if not face.get("ok") or not face.get("mesh"):
            continue
        if str(face.get("path") or "") == str(target.get("path") or ""):
            continue
        residual, inliers, scale = _residual(face, target)
        if residual >= 1.0:
            continue
        name = row.get("filename") or row.get("path") or ""
        if kind == "face":
            if not face.get("pose_ok") or not target.get("pose_ok"):
                continue
            dyaw = abs(_f(face.get("yaw")) - _f(target.get("yaw")))
            droll = abs(_f(face.get("roll")) - _f(target.get("roll")))
            if dyaw > FACE_YAW_TOL or droll > FACE_ROLL_TOL:
                rejected.append("%s: la cabeza esta girada %.0f grados de mas"
                                % (name, max(dyaw - FACE_YAW_TOL,
                                             droll - FACE_ROLL_TOL)))
                continue
            if residual > FACE_RESIDUAL_MAX:
                rejected.append("%s: no encaja (error %.3f)" % (name, residual))
                continue
            ranked.append((residual, row, {"residual": round(residual, 4),
                                           "inliers": round(inliers, 3),
                                           "escala": round(scale, 3),
                                           "yaw": _f(face.get("yaw")),
                                           "dyaw": round(dyaw, 1)}))
        else:                                             # skin
            full = _full_scale(face, target)
            if not (SKIN_SCALE_MIN <= full <= SKIN_SCALE_MAX):
                rejected.append("%s: escala real %.2f fuera de rango"
                                % (name, full))
                continue
            ranked.append((residual, row, {"residual": round(residual, 4),
                                           "escala_real": round(full, 3),
                                           "grano": _f(face.get("grain"))}))

    if not ranked:
        return {"path": "", "alternatives": [],
                "detail": {"descartadas": rejected[:6]},
                "reason": ("Ninguna de tus fotos encaja con esta imagen para "
                           "%s (%s)."
                           % ("el rostro" if kind == "face" else "la piel",
                              rejected[0] if rejected
                              else "no hay puntos faciales comunes"))}

    ranked.sort(key=lambda item: item[0])
    if kind == "skin":
        ranked = ranked[:SKIN_MAX_CANDIDATES]
    first = ranked[0]
    return {
        "path": first[1]["path"],
        "alternatives": [r[1]["path"] for r in ranked[1:]],
        "reason": ("Se usa %s: es la foto tuya que mejor encaja con esta "
                   "imagen (error de alineacion %.3f)."
                   % (first[1].get("filename") or first[1]["path"], first[0])),
        "detail": {"elegida": first[1].get("filename"),
                   "candidatas": len(ranked), "descartadas": len(rejected),
                   **first[2]},
    }


def _body_consensus(profile: dict, generated_image: Any) -> dict:
    """Her proportions as the MEDIAN of her photographs, by shot type.

    Judging a body against one photograph judges it against that photograph's
    lean.  But a global median over every measurable photograph is not the
    answer either: it spreads 47-66% and that spread is framing, not her - the
    same shoulder ratio spreads 66% over the whole gallery and 10% inside the
    closeups.  So the consensus is taken inside the shot type of the generated
    frame whenever there are BODY_MIN_SAMPLES photographs of it, it says which
    scope it used, and it hands back nothing when there are too few.
    """
    shot = ""
    if isinstance(generated_image, dict):
        shot = str(generated_image.get("shot_type") or "")
    rows = photographs(profile, need=("face", "body"))
    measured = [r for r in rows if (r.get("body") or {}).get("ok")]
    pool = [r for r in measured if r.get("shot_type") == shot] if shot else []
    scope = "mismo tipo de plano (%s)" % shot
    if len(pool) < BODY_MIN_SAMPLES:
        pool, scope = measured, "todas tus fotos medibles"
    if len(pool) < BODY_MIN_SAMPLES:
        return {"path": "", "alternatives": [],
                "detail": {"medibles": len(pool)},
                "reason": ("Solo hay %d fotos tuyas con medidas de cuerpo: no "
                           "hay consenso con el que comparar." % len(pool))}

    consensus: dict[str, float] = {}
    spread: dict[str, float] = {}
    for metric in BODY_METRICS:
        values = [_f(r["body"]["metrics"].get(metric)) for r in pool
                  if _f(r["body"]["metrics"].get(metric)) > 0.0]
        if len(values) < BODY_MIN_SAMPLES:
            continue
        median = float(np.median(values))
        if median <= 0.0:
            continue
        consensus[metric] = round(median, 5)
        spread[metric] = round((max(values) - min(values)) / median, 4)
    if not consensus:
        return {"path": "", "alternatives": [],
                "detail": {"medibles": len(pool)},
                "reason": "Tus fotos no dan ninguna medida de cuerpo comparable."}
    return {
        "path": "",
        "alternatives": [r["path"] for r in pool],
        "reason": ("Tus proporciones se comparan con la mediana de %d fotos "
                   "tuyas (%s), no con una sola." % (len(pool), scope)),
        "detail": {"consenso": consensus, "dispersion": spread,
                   "n": len(pool), "ambito": scope,
                   "fotos": [r.get("filename") for r in pool]},
    }


def _hand_donor(profile: dict, generated_image: Any) -> dict:
    """The photograph whose hand is held most like the broken one.

    Restricted to photographs whose own hands the scanner reads as clean.  Her
    IMG_8825 reads hand severity 0.57 - higher than any generated frame on this
    installation - and it is the nearest donor for 6 of 8 broken hands, so the
    unconstrained rule hands over a hand that is itself a defect.
    """
    if isinstance(generated_image, dict):
        target_hands = generated_image.get("hands") or {}
    else:
        target_hands = (_hands_reading(str(generated_image))
                        or {}).get("hands") or {}
    if not target_hands:
        return {"path": "", "alternatives": [], "detail": {},
                "reason": "no se ven las manos en la imagen generada"}

    rows = photographs(profile, need=("face", "hands"))
    ranked: list[tuple] = []
    for row in rows:
        hands = row.get("hands") or {}
        if not hands.get("ok") or not hands.get("hands"):
            continue
        if _f(hands.get("severity")) > HAND_CLEAN_MAX:
            continue
        best = 9.9
        for side, points in (hands.get("hands") or {}).items():
            other = target_hands.get(side)
            if other:
                best = min(best, _procrustes(points, other))
        if best < 9.0:
            ranked.append((best, row))
    if not ranked:
        return {"path": "", "alternatives": [], "detail": {},
                "reason": ("Ninguna foto tuya tiene las manos limpias y en una "
                           "posicion comparable.")}
    ranked.sort(key=lambda item: item[0])
    distance, row = ranked[0]
    usable = distance <= HAND_MATCH_MAX
    return {
        "path": row["path"] if usable else "",
        "alternatives": [r[1]["path"] for r in ranked[1:3]],
        "reason": (("Se usa %s: su mano esta en la misma posicion (distancia "
                    "%.3f)." % (row.get("filename"), distance)) if usable else
                   ("La mano mas parecida de tus fotos esta a %.3f y el limite "
                    "para copiarla es %.2f: no se toca la imagen."
                    % (distance, HAND_MATCH_MAX))),
        "detail": {"distancia": round(distance, 4), "limite": HAND_MATCH_MAX,
                   "candidata": row.get("filename"), "n": len(ranked)},
    }
