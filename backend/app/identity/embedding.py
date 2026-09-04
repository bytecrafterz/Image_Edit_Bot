"""The face signature that can actually tell her from somebody else.

WHY THIS MODULE EXISTS.  ``analysis/face.py`` builds a 64 float geometric and
photometric descriptor, and its own docstring is honest about what that is: a
consistency check, not face recognition.  Measured against the stored profile of
Nayane it is worse than that - it is blind.  Her own 24 photographs scored
0.9832 .. 0.9993 and eight photographs of eight OTHER women scored 0.9577 ..
0.9945, so the two populations overlap and the gate at 0.72 sat 0.24 below both
of them and could never fire.  The two paid results the client rejected on
sight, visibly a different woman, scored 0.9905 and 0.9960 and were approved.
The client's entire requirement is "it has to look exactly like her", so a
signature that answers "this is a human face" instead of "this is HER face" is
the single defect that matters most.

WHAT REPLACED IT.  SFace (Zhong & Deng, 2021), an ArcFace-style network that
maps a face to 128 floats trained so that two crops of one person point the same
way and two people do not.  It ships in OpenCV itself - ``cv2.FaceRecognizerSF``
- so nothing new had to be installed; only the weights are downloaded.  On the
same populations the cosine against her profile mean reads 0.6362 .. 0.8785 for
her own photographs and 0.0408 .. 0.2886 for the impostors and the two rejected
paid results: a gap of +0.3476 where the old descriptor had +0.0003.

MODEL FILES, both from the OpenCV Zoo (github.com/opencv/opencv_zoo, Apache-2.0),
stored under ``backend/app/models`` and fetched by ``scripts/fetch_face_model.py``:

  face_recognition_sface_2021dec.onnx   38,696,353 bytes
    https://github.com/opencv/opencv_zoo/raw/main/models/face_recognition_sface/face_recognition_sface_2021dec.onnx
    sha256 0ba9fbfa01b5270c96627c4ef784da859931e02f04419c829e83484087c34e79
  face_detection_yunet_2023mar.onnx        232,589 bytes
    https://github.com/opencv/opencv_zoo/raw/main/models/face_detection_yunet/face_detection_yunet_2023mar.onnx
    sha256 8f2383e4dd3cfbb4553ea8718107fc0423210dc964f9f4280604804ed2552fa4

Nothing here calls a network at runtime and no image ever leaves the machine.

Like the rest of the analysis layer this module degrades instead of raising: a
missing weights file makes ``available()`` False and every embedding None, and
the caller is expected to say it could not judge rather than to pass in silence.
"""
from __future__ import annotations

import os
import threading

import cv2
import numpy as np

MODEL_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                         "models")
SFACE_FILE = "face_recognition_sface_2021dec.onnx"
YUNET_FILE = "face_detection_yunet_2023mar.onnx"
DIMS = 128

# The head is cut out of the frame and resampled so the face is about this wide
# before anything is measured.  SFace reads a 112 px crop, so a face 60 px
# across in a full-length photograph has to be enlarged before the alignment
# rather than after it, and a 900 px closeup has to be shrunk once with a proper
# area filter instead of being point-sampled by the warp.
FACE_TARGET_PX = 260.0
# Half-width of that crop in face widths.  YuNet is trained on boxes with some
# hair and chin around them and finds nothing on a face cropped at the skin.
CROP_MARGIN = 0.85
CROP_MAX_PX = 1400
# YuNet's own confidence floor.  Low on purpose: this detector is not being
# asked to find faces in a crowd, only to say which way up an already located
# head is, and the four rotations compete on score so a weak true face still
# beats a weaker false one.
DETECT_CONF = 0.5
DETECT_NMS = 0.3

# The four right-angle rotations are tried and the most confident one wins.
# This is not a nicety.  Six of her 24 photographs are phone selfies whose
# pixels are stored rotated, and on three of them (IMG_8948, IMG_8949,
# IMG_8950) MediaPipe fitted the mesh upside down, so the aligned crop came out
# inverted and SFace read a stranger: 0.1549, 0.1554 and 0.1835 against a
# profile of her own other photographs, below every impostor in the negative
# set.  Choosing the orientation YuNet is most confident about lifted those
# three to 0.7447, 0.8435 and 0.7180 and lifted the worst photograph in the
# whole set from 0.1549 to 0.6362.
_ROTATIONS = ((0, None),
              (90, cv2.ROTATE_90_CLOCKWISE),
              (180, cv2.ROTATE_180),
              (270, cv2.ROTATE_90_COUNTERCLOCKWISE))

# How sure the detector has to be about the head the mesh cut out before that
# crop is measured without a second opinion.  Every clean image on this disk
# clears it: her 24 photographs read 0.8984 .. 0.9498 and the 38 generated
# results 0.9328 .. 0.9600.  Under it the mesh has usually landed on something
# that is not a face at all, and the embedding taken there is of a stranger -
# measured on her own IMG_8798 turned a quarter circle (mesh 0.530, similarity
# 0.0263 against her other 23 photographs) and shrunk to 256 px (mesh 0.660,
# 0.0631), while a look at the whole frame found her at 0.925 and 0.909 and
# read 0.6694 and 0.6124.  A near-zero score on her own face is not a strict
# gate; it is the gate measuring the wrong pixels and then calling her a
# stranger, which is the one thing this check must never do.
MESH_TRUSTED_CONF = 0.90
# The second opinion only wins when it is clearly better, not merely better.
# Over 504 (photograph, degradation) pairs every margin from 0.05 to 0.20 gives
# the same result - 17 rejections instead of 20, worst score 0.0859 instead of
# 0.0263, fifth percentile 0.5010 instead of 0.4883, and not one pair made
# worse - while taking the more confident crop unconditionally (margin 0.00)
# repairs three cases and breaks three others.
RESCUE_MARGIN = 0.10

# FaceMesh points that stand in for YuNet's five, used only when the detector
# finds nothing at any rotation.  Eye centres are averaged over the lid ring
# rather than taken from one corner: on a 1024 px photograph the ring centre
# landed within 1 px of YuNet's eye point while the outer corner sat 19 px away.
_EYE_R = (33, 133, 159, 145, 7, 163, 144, 153)
_EYE_L = (263, 362, 386, 374, 249, 390, 373, 380)
_NOSE_TIP = 1
_MOUTH_R, _MOUTH_L = 61, 291

_LOCK = threading.Lock()
# The two nets are ONE object each for the whole process, and they are
# stateful: YuNet is told the size of the picture with ``setInputSize`` and
# only then asked to detect, and SFace aligns a crop inside itself before it
# measures it.  Nothing serialised those two-step sequences, and the run makes
# up to ``max_parallel_generations`` images at a time (3 on this installation),
# so one variant's ``setInputSize`` landed between another's and its
# ``detect``.  Measured on 2026-09-04 with three threads over three files: the
# same frame that reads 0.7173 three times in a row on its own came back at
# 0.0106 and 0.1407 in parallel, and a second frame that reads 0.5279 alone
# read 0.1022 twice.  That is not a face check being strict, it is a face check
# measuring the wrong pixels - and every one of those readings would have
# rejected an image of her that she had already paid for, at 0.040 USD a
# regeneration.  One lock around the inference costs about 0.3 s per image on a
# three-way run and makes the number reproducible, which is the whole claim of
# this module.
_INFER = threading.RLock()
_RECOGNIZER = None
_DETECTOR = None
_FAILED = ""


def _paths() -> tuple[str, str]:
    return (os.path.join(MODEL_DIR, SFACE_FILE),
            os.path.join(MODEL_DIR, YUNET_FILE))


def _load() -> tuple:
    """Create both nets once and keep them; never raises."""
    global _RECOGNIZER, _DETECTOR, _FAILED
    if _RECOGNIZER is not None and _DETECTOR is not None:
        return _RECOGNIZER, _DETECTOR
    if _FAILED:
        return None, None
    with _LOCK:
        if _RECOGNIZER is not None and _DETECTOR is not None:
            return _RECOGNIZER, _DETECTOR
        if _FAILED:
            return None, None
        sface, yunet = _paths()
        missing = [os.path.basename(p) for p in (sface, yunet) if not os.path.exists(p)]
        if missing:
            _FAILED = ("faltan los pesos del reconocedor facial (%s); ejecuta "
                       "scripts/fetch_face_model.py" % ", ".join(missing))
            return None, None
        try:
            rec = cv2.FaceRecognizerSF.create(sface, "")
            det = cv2.FaceDetectorYN.create(yunet, "", (320, 320),
                                            DETECT_CONF, DETECT_NMS, 5000)
        except Exception as exc:           # a corrupt download, an old OpenCV
            _FAILED = "no se pudo cargar el reconocedor facial (%s)" % exc
            return None, None
        _RECOGNIZER, _DETECTOR = rec, det
        return rec, det


def available() -> bool:
    """True when a real embedding can be computed on this machine."""
    rec, det = _load()
    return rec is not None and det is not None


def unavailable_reason() -> str:
    """Spanish sentence naming why not, empty when it is available."""
    _load()
    return _FAILED


def _unit(vec) -> list[float] | None:
    arr = np.asarray(vec, dtype=np.float64).reshape(-1)
    if arr.size != DIMS or not np.all(np.isfinite(arr)):
        return None
    norm = float(np.linalg.norm(arr))
    if norm < 1e-9:
        return None
    return [round(float(v), 6) for v in (arr / norm)]


def _feature(rec, image: np.ndarray, row: np.ndarray) -> list[float] | None:
    try:
        with _INFER:
            return _unit(rec.feature(rec.alignCrop(image, row))[0])
    except Exception:
        return None


def _best_rotation(det, image: np.ndarray):
    """The most confident YuNet face over the four right-angle rotations."""
    best = None
    for _deg, code in _ROTATIONS:
        view = image if code is None else cv2.rotate(image, code)
        try:
            # setInputSize and detect are one indivisible operation: they are
            # two calls into one stateful net, and interleaving them across
            # threads makes the detector read a frame at another frame's size.
            with _INFER:
                det.setInputSize((int(view.shape[1]), int(view.shape[0])))
                _n, faces = det.detect(view)
        except Exception:
            continue
        if faces is None or len(faces) == 0:
            continue
        idx = int(np.argmax(faces[:, 14]))
        score = float(faces[idx, 14])
        if best is None or score > best[0]:
            best = (score, view, faces[idx])
    return best


def _head_crop(img: np.ndarray, mesh) -> tuple:
    """Cut the head out and bring it to FACE_TARGET_PX, mesh points included."""
    height, width = img.shape[:2]
    points = np.asarray(mesh, dtype=np.float64)
    if points.ndim != 2 or points.shape[0] < 468 or points.shape[1] < 2:
        return None, None
    px = points[:, :2] * [width, height]
    if not np.all(np.isfinite(px)):
        return None, None
    lo, hi = px.min(axis=0), px.max(axis=0)
    face_w = float(max(hi[0] - lo[0], hi[1] - lo[1]))
    if face_w < 8.0:
        return None, None
    half = face_w * CROP_MARGIN
    cx, cy = (lo[0] + hi[0]) / 2.0, (lo[1] + hi[1]) / 2.0
    side = int(round(2.0 * half * min(FACE_TARGET_PX / face_w, 4.0)))
    side = max(48, min(side, CROP_MAX_PX))
    scale = side / (2.0 * half)
    x0, y0 = cx - half, cy - half
    matrix = np.array([[scale, 0.0, -scale * x0], [0.0, scale, -scale * y0]],
                      dtype=np.float64)
    # BORDER_REPLICATE and not a black margin: a hard black edge beside a cheek
    # is a step the detector reads as structure, and a head at the edge of the
    # frame is the common case in a phone selfie.
    crop = cv2.warpAffine(img, matrix, (side, side), flags=cv2.INTER_CUBIC,
                          borderMode=cv2.BORDER_REPLICATE)
    keys = np.array([px[list(_EYE_R)].mean(axis=0), px[list(_EYE_L)].mean(axis=0),
                     px[_NOSE_TIP], px[_MOUTH_R], px[_MOUTH_L]], dtype=np.float64)
    keys = keys * scale - [scale * x0, scale * y0]
    return crop, keys


def _row_from_points(side: int, keys: np.ndarray) -> np.ndarray:
    """A YuNet-shaped detection row so alignCrop can use the mesh points.

    The five points are in YuNet's order and convention - the subject's right
    eye first, which sits on the LEFT of a frontal image - and MediaPipe's mesh
    uses the same anatomical convention, so the indices map straight across.
    Checked on one photograph at 1024 px: the mesh eye centres landed at
    (297.5, 291.1) and (430.0, 286.6) against YuNet's (296.8, 291.9) and
    (421.0, 279.5).
    """
    row = np.zeros((15,), dtype=np.float32)
    row[0:4] = [0.0, 0.0, float(side), float(side)]
    row[4:14] = keys.reshape(-1)
    row[14] = 1.0
    return row


def _whole_frame(det, img: np.ndarray):
    """The best face anywhere in the frame, looked for at two sizes.

    Shrinking to 1024 px is what makes a 3088 px photograph affordable, but a
    face that is small inside a full-length shot can fall under the detector's
    reach at that size, so a second, larger look follows when the first finds
    nothing at all.
    """
    height, width = img.shape[:2]
    for side in (1024, 1600):
        scale = min(1.0, float(side) / float(max(width, height)))
        view = (cv2.resize(img, None, fx=scale, fy=scale,
                           interpolation=cv2.INTER_AREA) if scale < 1.0 else img)
        found = _best_rotation(det, view)
        if found is not None:
            return found
    return None


def face_embedding(img_bgr, face_d: dict | None = None) -> list[float] | None:
    """128 unit-length floats describing WHOSE face this is, or None.

    ``face_d`` is a ``analysis.face.detect_face`` result and is only used for
    its mesh, to cut the head out at a workable size; when it is missing or
    meshless the whole frame is searched with YuNet instead.
    """
    rec, det = _load()
    if rec is None or det is None:
        return None
    img = img_bgr
    if not isinstance(img, np.ndarray) or img.ndim != 3 or img.size == 0:
        return None
    if img.dtype != np.uint8:
        img = np.clip(img, 0, 255).astype(np.uint8)

    mesh = (face_d or {}).get("mesh") if isinstance(face_d, dict) else None
    if mesh and (face_d or {}).get("ok"):
        crop, keys = _head_crop(img, mesh)
        if crop is not None:
            found = _best_rotation(det, crop)
            if found is not None:
                if found[0] >= MESH_TRUSTED_CONF:
                    out = _feature(rec, found[1], found[2])
                    if out is not None:
                        return out
                else:
                    # The mesh landed somewhere the detector is not sure is a
                    # face, so it does not get the last word.  See
                    # MESH_TRUSTED_CONF for what that saves and what it costs.
                    whole = _whole_frame(det, img)
                    if whole is not None and whole[0] - found[0] >= RESCUE_MARGIN:
                        out = _feature(rec, whole[1], whole[2])
                        if out is not None:
                            return out
                    out = _feature(rec, found[1], found[2])
                    if out is not None:
                        return out
            # The mesh is the fallback and not the first choice because it
            # carries MediaPipe's own orientation mistake into the alignment,
            # which is exactly the failure the rotation search exists to undo.
            out = _feature(rec, crop, _row_from_points(crop.shape[0], keys))
            if out is not None:
                return out

    whole = _whole_frame(det, img)
    if whole is not None:
        out = _feature(rec, whole[1], whole[2])
        if out is not None:
            return out
    return None


def _matrix(rows) -> np.ndarray | None:
    if not isinstance(rows, (list, tuple)) or not rows:
        return None
    kept = []
    for row in rows:
        vec = _unit(row) if isinstance(row, (list, tuple, np.ndarray)) else None
        if vec is not None:
            kept.append(vec)
    if not kept:
        return None
    return np.asarray(kept, dtype=np.float64)


def gallery_mean(rows) -> list[float] | None:
    """The direction a person's photographs point in, as one unit vector.

    The mean beats every alternative measured on her 24 photographs, scored
    leave-one-out against the other 23 (positives) and against eight other
    women plus the two rejected paid results (negatives).  Worst positive
    against best negative:

        mean of the gallery   0.6362 vs 0.2886   gap +0.3476
        best single photo     0.6558 vs 0.3664   gap +0.2894
        mean of the top 3     0.6243 vs 0.3361   gap +0.2881
        median of the photos  0.5219 vs 0.2333   gap +0.2886

    It is also the cheapest: one dot product at verification time instead of
    one per stored photograph.  The individual embeddings are kept in the
    profile anyway, so a better rule can be fitted later without asking her for
    her photographs again.
    """
    mat = _matrix(rows)
    if mat is None:
        return None
    return _unit(mat.mean(axis=0))


def similarity(embedding, reference) -> float | None:
    """Cosine between a result and a profile reference, or None.

    Clamped at zero on the way out: a negative cosine means the two faces point
    away from each other, which is already as far from "her" as the measurement
    goes, and the check's contract is that higher is better on 0..1.
    """
    one = _unit(embedding)
    two = _unit(reference)
    if one is None or two is None:
        return None
    value = float(np.dot(np.asarray(one), np.asarray(two)))
    if not np.isfinite(value):
        return None
    return round(max(0.0, min(1.0, value)), 6)


def self_consistency(rows) -> dict:
    """How far apart her own photographs are, measured the way results will be.

    Each stored photograph is scored against the mean of the others, so the
    number reported is the same quantity the gate reads and the profile can say
    on its own report page how much room there is between her worst photograph
    and the line.
    """
    mat = _matrix(rows)
    if mat is None or mat.shape[0] < 2:
        return {"n": 0 if mat is None else int(mat.shape[0]),
                "min": 0.0, "mean": 0.0}
    scores = []
    for i in range(mat.shape[0]):
        rest = np.delete(mat, i, axis=0).mean(axis=0)
        norm = float(np.linalg.norm(rest))
        if norm < 1e-9:
            continue
        scores.append(float(np.dot(mat[i], rest / norm)))
    if not scores:
        return {"n": int(mat.shape[0]), "min": 0.0, "mean": 0.0}
    return {"n": int(mat.shape[0]), "min": round(float(min(scores)), 4),
            "mean": round(float(sum(scores) / len(scores)), 4)}
