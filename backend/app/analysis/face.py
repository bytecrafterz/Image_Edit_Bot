"""Face geometry, head pose and the identity signature.

This is the module that answers the client's real question: "is this still my
face?".  It measures the face instead of describing it, because a prompt that
asks a model to preserve someone's features is a wish, while a ratio between
two landmarks is evidence.

IMPORTANT - what the descriptor is and is not.  ``face_descriptor`` builds a
64 float GEOMETRIC + PHOTOMETRIC signature: distance ratios between mesh points
normalised by the inter-ocular distance, plus colour and texture statistics of
a handful of regions.  It is NOT a deep face recognition embedding and must
never be used to identify a stranger or to match one person against a database
of people.  Its only job is consistency checking of one person against that
same person's own stored profile: "the generated face has the same proportions
as the original", which is exactly the check the client wants and is a much
weaker (and, for privacy, much safer) claim than face recognition.

Everything degrades instead of raising.  If MediaPipe is missing we still hand
back a bounding box from an OpenCV Haar cascade so framing, skin sampling and
shot classification keep working; the descriptor is simply empty and callers
skip the identity check.
"""
from __future__ import annotations

import math
import os
import threading

import cv2
import numpy as np

try:
    import mediapipe as mp
    MEDIAPIPE_AVAILABLE = True
except ImportError:  # optional at runtime - the app must still boot without it
    mp = None
    MEDIAPIPE_AVAILABLE = False

MAX_SIDE = 1600           # detection only; landmarks come back normalised
# Side the head crop is enlarged to before re-meshing a small, distant face.
_CROP_MESH_SIDE = 640.0
GEO_DIMS = 40
PHOTO_DIMS = 24
DESCRIPTOR_DIMS = GEO_DIMS + PHOTO_DIMS      # 64, per CONTRACTS.md

_REASON_NO_MP = "mediapipe no disponible"

# --- canonical 468 point FaceMesh indices ---------------------------------
# Left/right follow MediaPipe's own convention, which is anatomical: index 33
# belongs to the subject's RIGHT eye and therefore sits on the LEFT side of a
# frontal image.  Same convention as analysis/pose.py.
I_NOSE_TIP = 1
I_SUBNASALE = 2
I_NASION = 168
I_FOREHEAD = 10
I_CHIN = 152
I_EAR_R, I_EAR_L = 234, 454
I_TEMPLE_R, I_TEMPLE_L = 127, 356
I_JAW_HI_R, I_JAW_HI_L = 93, 323
I_JAW_MID_R, I_JAW_MID_L = 58, 288
I_JAW_LO_R, I_JAW_LO_L = 172, 397
I_EYE_OUT_R, I_EYE_IN_R, I_EYE_TOP_R, I_EYE_BOT_R = 33, 133, 159, 145
I_EYE_OUT_L, I_EYE_IN_L, I_EYE_TOP_L, I_EYE_BOT_L = 263, 362, 386, 374
I_IRIS_R, I_IRIS_L = 468, 473            # only when refine_landmarks is on
I_MOUTH_R, I_MOUTH_L = 61, 291
I_LIP_TOP, I_LIP_BOT = 0, 17
I_LIP_IN_TOP, I_LIP_IN_BOT = 13, 14
I_ALAR_R, I_ALAR_L = 129, 358
I_BROW_OUT_R, I_BROW_IN_R, I_BROW_TOP_R = 46, 55, 105
I_BROW_OUT_L, I_BROW_IN_L, I_BROW_TOP_L = 276, 285, 334
_BROW_R = (46, 53, 52, 65, 55, 70, 63, 105, 66, 107)
_BROW_L = (276, 283, 282, 295, 285, 300, 293, 334, 296, 336)

# solvePnP reference head, expressed in the OpenCV camera frame (x right,
# y down, z away from the camera) so the frontal solution is near identity.
_MODEL_POINTS = np.array([
    (0.0, 0.0, 0.0),            # nose tip                   (1)
    (0.0, 330.0, -65.0),        # chin                       (152)
    (-225.0, -170.0, -135.0),   # eye corner on image left    (33)
    (225.0, -170.0, -135.0),    # eye corner on image right  (263)
    (-150.0, 150.0, -125.0),    # mouth corner on image left  (61)
    (150.0, 150.0, -125.0),     # mouth corner on image right (291)
], dtype=np.float64)
_PNP_INDICES = (I_NOSE_TIP, I_CHIN, I_EYE_OUT_R, I_EYE_OUT_L,
                I_MOUTH_R, I_MOUTH_L)

_POPCOUNT = np.array([bin(i).count("1") for i in range(256)], dtype=np.int32)
_LBP_SIDE = 32            # texture patches are resampled to this before coding

_LOCK = threading.Lock()
_MESH = None
_DETECT = None
_CASCADE = None
_MESH_FAILED = False
_DETECT_FAILED = False
_CASCADE_FAILED = False


# ------------------------------------------------------------ image helpers

def _blank(reason: str, backend: str = "none") -> dict:
    """One shape for every failure so callers never have to branch."""
    return {
        "ok": False,
        "bbox": [],
        "bbox_norm": [],
        "landmarks": {},
        "mesh": [],
        "descriptor": [],
        "yaw": 0.0,
        "pitch": 0.0,
        "roll": 0.0,
        "n_faces": 0,
        "n_points": 0,
        "backend": backend,
        "reason": reason,
    }


def _as_bgr(img) -> np.ndarray | None:
    if not isinstance(img, np.ndarray) or img.size == 0 or img.ndim not in (2, 3):
        return None
    if img.dtype != np.uint8:
        img = np.clip(img, 0, 255).astype(np.uint8)
    if img.ndim == 2:
        return cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    if img.shape[2] == 4:
        return cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
    if img.shape[2] == 3:
        return img
    return None


def _downscale(img: np.ndarray, max_side: int) -> np.ndarray:
    h, w = img.shape[:2]
    longest = max(h, w)
    if longest <= max_side or longest == 0:
        return img
    scale = max_side / float(longest)
    return cv2.resize(
        img,
        (max(1, int(round(w * scale))), max(1, int(round(h * scale)))),
        interpolation=cv2.INTER_AREA,
    )


def _mesh_px(mesh, width: int, height: int) -> np.ndarray:
    """Normalised mesh -> pixel coordinates of *this* image.

    The mesh is stored normalised precisely so it survives resizing, but it is
    only meaningful against an image with the same framing it was measured on.
    """
    arr = np.asarray(mesh, dtype=np.float64)
    if arr.ndim != 2 or arr.shape[0] < 468 or arr.shape[1] < 2:
        raise ValueError("malla facial incompleta")
    out = np.empty((arr.shape[0], 2), dtype=np.float64)
    out[:, 0] = arr[:, 0] * float(width)
    out[:, 1] = arr[:, 1] * float(height)
    return out


def _eye_centers(px: np.ndarray):
    """Iris centres when refine_landmarks gave them, eyelid centroids if not."""
    if len(px) >= 478:
        return np.array(px[I_IRIS_R], dtype=np.float64), \
               np.array(px[I_IRIS_L], dtype=np.float64)
    if len(px) < 468:
        return None
    right = px[[I_EYE_OUT_R, I_EYE_IN_R, I_EYE_TOP_R, I_EYE_BOT_R]].mean(axis=0)
    left = px[[I_EYE_OUT_L, I_EYE_IN_L, I_EYE_TOP_L, I_EYE_BOT_L]].mean(axis=0)
    return np.asarray(right, dtype=np.float64), np.asarray(left, dtype=np.float64)


def _clamp01(value: float) -> float:
    return 0.0 if value < 0.0 else (1.0 if value > 1.0 else value)


def _bbox_from_points(px: np.ndarray, width: int, height: int) -> list[int]:
    x0 = int(max(0, math.floor(float(px[:, 0].min()))))
    y0 = int(max(0, math.floor(float(px[:, 1].min()))))
    x1 = int(min(width, math.ceil(float(px[:, 0].max()))))
    y1 = int(min(height, math.ceil(float(px[:, 1].max()))))
    return [x0, y0, max(0, x1 - x0), max(0, y1 - y0)]


def _bbox_norm(bbox: list[int], width: int, height: int) -> list[float]:
    if not bbox or width <= 0 or height <= 0:
        return []
    x, y, bw, bh = bbox
    return [round(_clamp01(x / float(width)), 5),
            round(_clamp01(y / float(height)), 5),
            round(_clamp01(bw / float(width)), 5),
            round(_clamp01(bh / float(height)), 5)]


def _point_entry(point, width: int, height: int) -> dict:
    x = float(point[0])
    y = float(point[1])
    return {
        "x": round(_clamp01(x / float(width)), 5),
        "y": round(_clamp01(y / float(height)), 5),
        "px": [int(round(x)), int(round(y))],
    }


# ------------------------------------------------------------ mediapipe glue

def _face_mesh():
    """Lazy singleton.  Caller must already hold ``_LOCK``."""
    global _MESH, _MESH_FAILED
    if _MESH is not None or _MESH_FAILED:
        return _MESH
    try:
        _MESH = mp.solutions.face_mesh.FaceMesh(
            static_image_mode=True,
            max_num_faces=3,          # counting faces feeds the extra_person check
            refine_landmarks=True,    # adds the 10 iris points -> 478 total
            min_detection_confidence=0.5,
        )
    except Exception:
        _MESH_FAILED = True
        _MESH = None
    return _MESH


def _face_detector():
    """Lazy singleton for the bbox-only fallback.  Caller holds ``_LOCK``."""
    global _DETECT, _DETECT_FAILED
    if _DETECT is not None or _DETECT_FAILED:
        return _DETECT
    try:
        _DETECT = mp.solutions.face_detection.FaceDetection(
            model_selection=1, min_detection_confidence=0.4)
    except Exception:
        _DETECT_FAILED = True
        _DETECT = None
    return _DETECT


def _cascade():
    """Haar cascade shipped inside the OpenCV wheel.  Caller holds ``_LOCK``."""
    global _CASCADE, _CASCADE_FAILED
    if _CASCADE is not None or _CASCADE_FAILED:
        return _CASCADE
    try:
        data = getattr(cv2, "data", None)
        folder = getattr(data, "haarcascades", "") if data is not None else ""
        path = os.path.join(folder, "haarcascade_frontalface_default.xml")
        if not folder or not os.path.exists(path):
            _CASCADE_FAILED = True
            return None
        candidate = cv2.CascadeClassifier(path)
        if candidate.empty():
            _CASCADE_FAILED = True
            return None
        _CASCADE = candidate
    except Exception:
        _CASCADE_FAILED = True
        _CASCADE = None
    return _CASCADE


def _run_mesh(rgb: np.ndarray):
    """-> (normalised Nx2 list of the largest face, n_faces, reason)."""
    with _LOCK:
        mesh = _face_mesh()
        if mesh is None:
            return None, 0, "mediapipe face mesh no se pudo inicializar"
        try:
            result = mesh.process(rgb)
        except Exception as exc:
            return None, 0, "fallo la malla facial: %s" % exc

    faces = getattr(result, "multi_face_landmarks", None)
    if not faces:
        return None, 0, "no se detecto rostro"

    try:
        best = None
        best_area = -1.0
        for candidate in faces:
            xs = [lm.x for lm in candidate.landmark]
            ys = [lm.y for lm in candidate.landmark]
            area = (max(xs) - min(xs)) * (max(ys) - min(ys))
            if area > best_area:
                best_area, best = area, candidate
        points = [[float(lm.x), float(lm.y)] for lm in best.landmark]
    except (AttributeError, TypeError, ValueError, IndexError):
        return None, 0, "resultado de malla facial ilegible"
    return points, len(faces), ""


def _run_detection(rgb: np.ndarray):
    """Bounding box plus six key points when the full mesh refuses."""
    with _LOCK:
        detector = _face_detector()
        if detector is None:
            return None
        try:
            result = detector.process(rgb)
        except Exception:
            return None

    detections = getattr(result, "detections", None)
    if not detections:
        return None

    try:
        best = None
        best_area = -1.0
        for det in detections:
            box = det.location_data.relative_bounding_box
            area = float(box.width) * float(box.height)
            if area > best_area:
                best_area, best = area, det
        box = best.location_data.relative_bounding_box
        keys = [(float(k.x), float(k.y))
                for k in list(best.location_data.relative_keypoints)]
        return ((float(box.xmin), float(box.ymin), float(box.width),
                 float(box.height)), keys, len(detections))
    except (AttributeError, TypeError, ValueError, IndexError):
        return None


# --------------------------------------------------------------- head pose

def _wrap_deg(angle: float) -> float:
    angle = math.fmod(angle + 180.0, 360.0)
    if angle < 0.0:
        angle += 360.0
    return angle - 180.0


def _head_pose(px: np.ndarray, width: int, height: int) -> tuple:
    """yaw, pitch, roll in degrees from solvePnP; (0,0,0) when it will not solve.

    Signs follow the OpenCV camera frame: positive roll rotates the face
    clockwise in the image, positive yaw turns the nose toward the image left,
    positive pitch tips the chin down.
    """
    try:
        image_points = np.array([px[i] for i in _PNP_INDICES], dtype=np.float64)
        if not np.all(np.isfinite(image_points)):
            return 0.0, 0.0, 0.0
        focal = float(max(width, height))
        camera = np.array([[focal, 0.0, width / 2.0],
                           [0.0, focal, height / 2.0],
                           [0.0, 0.0, 1.0]], dtype=np.float64)
        dist = np.zeros((4, 1), dtype=np.float64)
        ok, rvec, _tvec = cv2.solvePnP(_MODEL_POINTS, image_points, camera, dist,
                                       flags=cv2.SOLVEPNP_ITERATIVE)
        if not ok:
            return 0.0, 0.0, 0.0
        rmat, _ = cv2.Rodrigues(rvec)
        sy = math.hypot(float(rmat[0, 0]), float(rmat[1, 0]))
        if sy > 1e-6:
            pitch = math.atan2(float(rmat[2, 1]), float(rmat[2, 2]))
            yaw = math.atan2(-float(rmat[2, 0]), sy)
            roll = math.atan2(float(rmat[1, 0]), float(rmat[0, 0]))
        else:  # gimbal lock: roll and yaw are degenerate, keep roll at zero
            pitch = math.atan2(-float(rmat[1, 2]), float(rmat[1, 1]))
            yaw = math.atan2(-float(rmat[2, 0]), sy)
            roll = 0.0
        return (round(_wrap_deg(math.degrees(yaw)), 2),
                round(_wrap_deg(math.degrees(pitch)), 2),
                round(_wrap_deg(math.degrees(roll)), 2))
    except (cv2.error, ValueError, IndexError, TypeError):
        return 0.0, 0.0, 0.0


# ------------------------------------------------------------------- public

def _mesh_from_head_crop(img_bgr):
    """Locate the head cheaply, enlarge it, and mesh that instead.

    Returns mesh points normalised to the ORIGINAL frame, or None.  Purely a
    resolution rescue: it changes where the mesh is measured, never what the
    rest of the module believes the coordinate system to be.
    """
    height, width = img_bgr.shape[:2]
    if height < 2 or width < 2:
        return None

    rgb_small = cv2.cvtColor(_downscale(img_bgr, MAX_SIDE), cv2.COLOR_BGR2RGB)
    box = None
    detected = _run_detection(rgb_small)
    if detected is not None:
        bx, by, bw, bh = detected[0]
        box = (bx * width, by * height, bw * width, bh * height)
    else:
        haar = _haar_face(img_bgr, "")
        found = haar.get("bbox") or []
        if len(found) >= 4 and found[2] > 0 and found[3] > 0:
            box = tuple(float(v) for v in found[:4])
    if box is None:
        return None

    x, y, bw, bh = box
    if bw < 8 or bh < 8:
        return None
    pad = 0.5 * max(bw, bh)
    x0 = int(max(0, round(x - pad)))
    y0 = int(max(0, round(y - pad)))
    x1 = int(min(width, round(x + bw + pad)))
    y1 = int(min(height, round(y + bh + pad)))
    if x1 - x0 < 16 or y1 - y0 < 16:
        return None

    crop = img_bgr[y0:y1, x0:x1]
    crop_h, crop_w = crop.shape[:2]
    scale = _CROP_MESH_SIDE / float(max(crop_h, crop_w))
    if scale > 1.0:
        crop = cv2.resize(crop, (max(1, int(round(crop_w * scale))),
                                 max(1, int(round(crop_h * scale)))),
                          interpolation=cv2.INTER_CUBIC)

    mesh, _, _ = _run_mesh(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))
    if mesh is None or len(mesh) < 468:
        return None

    # crop-normalised -> crop pixels -> original pixels -> original normalised
    span_x = float(x1 - x0)
    span_y = float(y1 - y0)
    out = []
    for point in mesh:
        px = x0 + float(point[0]) * span_x
        py = y0 + float(point[1]) * span_y
        out.append((px / float(width), py / float(height)))
    return out


def detect_face(img_bgr) -> dict:
    """Locate the primary face, its mesh, its named points and its head pose.

    Backends, in order of preference: ``mediapipe_facemesh`` (468 points plus
    iris), ``mediapipe_detection`` (box plus six key points), ``haar`` (box
    only).  ``reason`` explains any downgrade; ``ok`` only means "a face was
    located", so always check ``descriptor`` before using it.
    """
    img = _as_bgr(img_bgr)
    if img is None:
        return _blank("imagen invalida")
    height, width = img.shape[:2]

    if not MEDIAPIPE_AVAILABLE:
        return _haar_face(img, _REASON_NO_MP)

    rgb = cv2.cvtColor(_downscale(img, MAX_SIDE), cv2.COLOR_BGR2RGB)
    mesh, n_faces, reason = _run_mesh(rgb)

    # On a full length photograph the head is a small fraction of the frame and
    # the mesh either fails or comes back too coarse to describe, so the
    # identity check reads "similarity 0" and throws away a perfectly good
    # picture of the right person.  Re-running the mesh on an enlarged crop of
    # the head recovers the full 468 points; the landmarks are then mapped back
    # into the original frame so everything downstream is unaffected.
    if mesh is None or len(mesh) < 468:
        recovered = _mesh_from_head_crop(img)
        if recovered is not None:
            mesh, n_faces, reason = recovered, max(1, n_faces), ""

    if mesh is not None:
        try:
            px = _mesh_px(mesh, width, height)
        except ValueError as exc:
            return _haar_face(img, str(exc))
        bbox = _bbox_from_points(px, width, height)
        yaw, pitch, roll = _head_pose(px, width, height)
        face = {
            "ok": True,
            "bbox": bbox,
            "bbox_norm": _bbox_norm(bbox, width, height),
            "landmarks": _named_points(px, width, height),
            "mesh": [[round(p[0], 5), round(p[1], 5)] for p in mesh],
            "yaw": yaw,
            "pitch": pitch,
            "roll": roll,
            "n_faces": int(n_faces),
            "n_points": len(mesh),
            "backend": "mediapipe_facemesh",
            "reason": "",
        }
        face["descriptor"] = face_descriptor(img, face)
        if not face["descriptor"]:
            face["reason"] = "descriptor no calculable"
        return face

    detected = _run_detection(rgb)
    if detected is not None:
        (bx, by, bw, bh), keys, count = detected
        bbox = [int(max(0, round(bx * width))), int(max(0, round(by * height))),
                int(max(0, round(bw * width))), int(max(0, round(bh * height)))]
        bbox[2] = min(bbox[2], max(0, width - bbox[0]))
        bbox[3] = min(bbox[3], max(0, height - bbox[1]))
        named: dict[str, dict] = {}
        # MediaPipe key point order: right eye, left eye, nose tip, mouth
        # centre, right ear tragion, left ear tragion.
        order = ("right_eye", "left_eye", "nose_tip", "mouth_center",
                 "right_ear", "left_ear")
        for name, key in zip(order, keys):
            named[name] = _point_entry((key[0] * width, key[1] * height),
                                       width, height)
        roll = 0.0
        if "left_eye" in named and "right_eye" in named:
            lx, ly = named["left_eye"]["px"]
            rx, ry = named["right_eye"]["px"]
            roll = round(_wrap_deg(math.degrees(math.atan2(ly - ry, lx - rx))), 2)
        out = _blank(reason or "malla facial no disponible", "mediapipe_detection")
        out.update({"ok": True, "bbox": bbox,
                    "bbox_norm": _bbox_norm(bbox, width, height),
                    "landmarks": named, "roll": roll,
                    "n_faces": int(count)})
        return out

    return _haar_face(img, reason or "no se detecto rostro")


def _named_points(px: np.ndarray, width: int, height: int) -> dict:
    """The small named subset every other module actually reads."""
    eyes = _eye_centers(px)
    named: dict[str, dict] = {}
    if eyes is not None:
        named["right_eye"] = _point_entry(eyes[0], width, height)
        named["left_eye"] = _point_entry(eyes[1], width, height)
    for name, index in (("nose_tip", I_NOSE_TIP), ("mouth_left", I_MOUTH_L),
                        ("mouth_right", I_MOUTH_R), ("chin", I_CHIN),
                        ("left_ear", I_EAR_L), ("right_ear", I_EAR_R),
                        ("forehead", I_FOREHEAD)):
        if index < len(px):
            named[name] = _point_entry(px[index], width, height)
    return named


def _haar_face(img: np.ndarray, reason: str) -> dict:
    """Last resort: a box, nothing else, so framing and skin sampling survive."""
    height, width = img.shape[:2]
    with _LOCK:
        cascade = _cascade()
        if cascade is None:
            return _blank(reason)
        try:
            gray = cv2.equalizeHist(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY))
            side = max(24, int(min(height, width) * 0.06))
            faces = cascade.detectMultiScale(gray, scaleFactor=1.1,
                                             minNeighbors=5,
                                             minSize=(side, side))
        except cv2.error:
            return _blank(reason)

    if faces is None or len(faces) == 0:
        return _blank(reason, "haar")

    x, y, bw, bh = max(faces, key=lambda f: int(f[2]) * int(f[3]))
    bbox = [int(x), int(y), int(bw), int(bh)]
    out = _blank(reason, "haar")
    out.update({"ok": True, "bbox": bbox,
                "bbox_norm": _bbox_norm(bbox, width, height),
                "n_faces": int(len(faces))})
    return out


# ------------------------------------------------------------- descriptor

def _ratio(numerator: float, denominator: float) -> float:
    if abs(denominator) < 1e-6:
        return 0.0
    return max(-8.0, min(8.0, numerator / denominator))


def _dist(a, b) -> float:
    return float(math.hypot(float(a[0]) - float(b[0]), float(a[1]) - float(b[1])))


def _mid(q: np.ndarray, i: int, j: int) -> np.ndarray:
    return (q[i] + q[j]) * 0.5


def _fit(values: list[float], size: int) -> list[float]:
    """Length is part of the contract, so force it instead of trusting it."""
    out = [0.0 if not math.isfinite(v) else max(-8.0, min(8.0, float(v)))
           for v in values[:size]]
    if len(out) < size:
        out.extend([0.0] * (size - len(out)))
    return out


def _canonical(px: np.ndarray, right_eye, left_eye, interocular: float,
               yaw: float) -> np.ndarray:
    """Put the mesh in a frame where scale, in-plane rotation and yaw are gone.

    Translate to the eye midpoint, rotate the eye line flat, then divide by the
    inter-ocular distance.  Yaw foreshortens horizontal distances by cos(yaw)
    while leaving vertical ones alone, so the same cosine is applied to y - the
    single anisotropic scale makes every distance, including diagonals, roughly
    comparable across +/- 25 degrees of head turn.
    """
    mid = (right_eye + left_eye) * 0.5
    theta = math.atan2(float(left_eye[1] - right_eye[1]),
                       float(left_eye[0] - right_eye[0]))
    cos_t, sin_t = math.cos(theta), math.sin(theta)
    rotation = np.array([[cos_t, sin_t], [-sin_t, cos_t]], dtype=np.float64)
    q = (px - mid) @ rotation.T
    if not math.isfinite(yaw):
        yaw = 0.0
    cos_yaw = max(0.55, math.cos(math.radians(max(-40.0, min(40.0, yaw)))))
    q[:, 0] /= interocular
    q[:, 1] *= cos_yaw / interocular
    return q


def _geometric_features(q: np.ndarray) -> list[float]:
    """40 scale and rotation invariant ratios, in inter-ocular units."""
    nose_w = _dist(q[I_ALAR_R], q[I_ALAR_L])
    nose_len = _dist(q[I_NASION], q[I_SUBNASALE])
    nose_prot = _dist(q[I_NOSE_TIP], _mid(q, I_ALAR_R, I_ALAR_L))

    mouth_w = _dist(q[I_MOUTH_R], q[I_MOUTH_L])
    mouth_h = _dist(q[I_LIP_TOP], q[I_LIP_BOT])
    upper_lip = _dist(q[I_LIP_TOP], q[I_LIP_IN_TOP])
    lower_lip = _dist(q[I_LIP_IN_BOT], q[I_LIP_BOT])
    mouth_mid = _mid(q, I_MOUTH_R, I_MOUTH_L)

    temple_w = _dist(q[I_TEMPLE_R], q[I_TEMPLE_L])
    cheek_w = _dist(q[I_EAR_R], q[I_EAR_L])          # widest point, ear to ear
    jaw_hi = _dist(q[I_JAW_HI_R], q[I_JAW_HI_L])
    jaw_mid = _dist(q[I_JAW_MID_R], q[I_JAW_MID_L])
    jaw_lo = _dist(q[I_JAW_LO_R], q[I_JAW_LO_L])

    face_h = _dist(q[I_FOREHEAD], q[I_CHIN])
    brow_eye_r = _dist(q[I_BROW_TOP_R], q[I_EYE_TOP_R])
    brow_eye_l = _dist(q[I_BROW_TOP_L], q[I_EYE_TOP_L])
    eye_w_r = _dist(q[I_EYE_OUT_R], q[I_EYE_IN_R])
    eye_w_l = _dist(q[I_EYE_OUT_L], q[I_EYE_IN_L])
    eye_h_r = _dist(q[I_EYE_TOP_R], q[I_EYE_BOT_R])
    eye_h_l = _dist(q[I_EYE_TOP_L], q[I_EYE_BOT_L])
    intercanthal = _dist(q[I_EYE_IN_R], q[I_EYE_IN_L])

    philtrum = _dist(q[I_SUBNASALE], q[I_LIP_TOP])
    chin_h = _dist(q[I_LIP_BOT], q[I_CHIN])
    mouth_chin = _dist(mouth_mid, q[I_CHIN])
    # The eye midpoint is the origin of the canonical frame.
    eye_nose = float(math.hypot(float(q[I_NOSE_TIP, 0]), float(q[I_NOSE_TIP, 1])))
    eye_mouth = float(math.hypot(float(mouth_mid[0]), float(mouth_mid[1])))
    nose_mouth = _dist(q[I_SUBNASALE], mouth_mid)
    forehead_h = _dist(q[I_FOREHEAD], q[I_NASION])
    brow_w_r = _dist(q[I_BROW_OUT_R], q[I_BROW_IN_R])
    brow_w_l = _dist(q[I_BROW_OUT_L], q[I_BROW_IN_L])

    return _fit([
        nose_w, nose_len, _ratio(nose_w, nose_len), nose_prot,
        mouth_w, mouth_h, _ratio(mouth_h, mouth_w), upper_lip, lower_lip,
        _ratio(upper_lip, lower_lip),
        temple_w, cheek_w, jaw_hi, jaw_mid, jaw_lo,
        _ratio(jaw_lo, cheek_w), _ratio(cheek_w, jaw_mid),
        face_h, _ratio(cheek_w, face_h),
        brow_eye_r, brow_eye_l, abs(brow_eye_r - brow_eye_l),
        eye_w_r, eye_w_l, eye_h_r, eye_h_l,
        _ratio(eye_h_r, eye_w_r), _ratio(eye_h_l, eye_w_l),
        intercanthal, philtrum, chin_h, mouth_chin,
        eye_nose, eye_mouth, nose_mouth, forehead_h,
        brow_w_r, brow_w_l,
        _ratio(nose_mouth + mouth_chin, face_h),
        _ratio(forehead_h, face_h),
    ], GEO_DIMS)


def _patch(img: np.ndarray, cx: float, cy: float,
           half_w: float, half_h: float) -> np.ndarray | None:
    """Axis aligned crop, clipped to the frame; None when nothing usable is left."""
    if not (math.isfinite(cx) and math.isfinite(cy)):
        return None
    height, width = img.shape[:2]
    half_w = max(1.0, float(half_w))
    half_h = max(1.0, float(half_h))
    x0 = max(0, int(round(cx - half_w)))
    y0 = max(0, int(round(cy - half_h)))
    x1 = min(width, int(round(cx + half_w)))
    y1 = min(height, int(round(cy + half_h)))
    if x1 - x0 < 2 or y1 - y0 < 2:
        return None
    return np.ascontiguousarray(img[y0:y1, x0:x1])


def _lab_mean(patch) -> list[float]:
    """Mean CIE Lab of a patch, rescaled so all three live around 0..1 / -1..1."""
    if patch is None or patch.size == 0:
        return [0.0, 0.0, 0.0]
    try:
        lab = cv2.cvtColor(patch, cv2.COLOR_BGR2LAB)
    except cv2.error:
        return [0.0, 0.0, 0.0]
    mean = lab.reshape(-1, 3).mean(axis=0)
    return [float(mean[0]) / 255.0,
            (float(mean[1]) - 128.0) / 128.0,
            (float(mean[2]) - 128.0) / 128.0]


def _lab_mean_of(patches: list) -> list[float]:
    values = [_lab_mean(p) for p in patches if p is not None]
    if not values:
        return [0.0, 0.0, 0.0]
    return [sum(v[i] for v in values) / len(values) for i in range(3)]


def _lbp6(patch) -> list[float]:
    """Six bin local binary pattern histogram of a skin patch.

    Bins are populated by bit count, not by code value, which makes the
    histogram invariant to in-plane rotation - the same property the geometric
    block has.  It is a texture fingerprint: heavy retouching collapses it
    toward the low bins.
    """
    if patch is None or patch.shape[0] < 8 or patch.shape[1] < 8:
        return [0.0] * 6
    try:
        # Resample to a fixed size first: texture must be measured per unit of
        # face, not per pixel, or a 4000px original and a 1024px render would
        # never agree.
        interp = cv2.INTER_AREA if patch.shape[0] > _LBP_SIDE else cv2.INTER_LINEAR
        square = cv2.resize(patch, (_LBP_SIDE, _LBP_SIDE), interpolation=interp)
        gray = cv2.cvtColor(square, cv2.COLOR_BGR2GRAY).astype(np.int16)
    except cv2.error:
        return [0.0] * 6
    centre = gray[1:-1, 1:-1]
    neighbours = (gray[0:-2, 0:-2], gray[0:-2, 1:-1], gray[0:-2, 2:],
                  gray[1:-1, 2:], gray[2:, 2:], gray[2:, 1:-1],
                  gray[2:, 0:-2], gray[1:-1, 0:-2])
    codes = np.zeros(centre.shape, dtype=np.int32)
    for bit, neighbour in enumerate(neighbours):
        codes += (neighbour >= centre).astype(np.int32) << bit
    counts = np.bincount(_POPCOUNT[codes].ravel(), minlength=9).astype(np.float64)
    total = float(counts.sum())
    if total <= 0.0:
        return [0.0] * 6
    counts /= total
    return [float(counts[0] + counts[1]), float(counts[2]), float(counts[3]),
            float(counts[4]), float(counts[5]),
            float(counts[6] + counts[7] + counts[8])]


def _photometric_features(img: np.ndarray, px: np.ndarray,
                          interocular: float) -> list[float]:
    """18 colour values over six regions plus a 6 bin texture histogram."""
    eyes = _eye_centers(px)
    if eyes is None:
        return _fit([], PHOTO_DIMS)
    right_eye, left_eye = eyes

    # --- iris: radius from the iris ring when present, eye width otherwise
    if len(px) >= 478:
        radius_r = float(np.mean([_dist(px[I_IRIS_R], px[i])
                                  for i in (469, 470, 471, 472)]))
        radius_l = float(np.mean([_dist(px[I_IRIS_L], px[i])
                                  for i in (474, 475, 476, 477)]))
    else:
        radius_r = 0.22 * _dist(px[I_EYE_OUT_R], px[I_EYE_IN_R])
        radius_l = 0.22 * _dist(px[I_EYE_OUT_L], px[I_EYE_IN_L])
    iris = _lab_mean_of([
        _patch(img, right_eye[0], right_eye[1], 0.7 * radius_r, 0.7 * radius_r),
        _patch(img, left_eye[0], left_eye[1], 0.7 * radius_l, 0.7 * radius_l),
    ])

    # --- lips: two thin bands avoid sampling the mouth opening
    mouth_w = _dist(px[I_MOUTH_R], px[I_MOUTH_L])
    upper = _mid(px, I_LIP_TOP, I_LIP_IN_TOP)
    lower = _mid(px, I_LIP_IN_BOT, I_LIP_BOT)
    lips = _lab_mean_of([
        _patch(img, upper[0], upper[1], 0.22 * mouth_w,
               max(2.0, 0.35 * _dist(px[I_LIP_TOP], px[I_LIP_IN_TOP]))),
        _patch(img, lower[0], lower[1], 0.22 * mouth_w,
               max(2.0, 0.35 * _dist(px[I_LIP_IN_BOT], px[I_LIP_BOT]))),
    ])

    # --- eyebrows: shrunken bounding box of each brow contour
    brow_patches = []
    for indices in (_BROW_R, _BROW_L):
        pts = px[list(indices)]
        cx = float(pts[:, 0].mean())
        cy = float(pts[:, 1].mean())
        half_w = 0.35 * float(pts[:, 0].max() - pts[:, 0].min())
        half_h = 0.35 * float(pts[:, 1].max() - pts[:, 1].min())
        brow_patches.append(_patch(img, cx, cy, max(2.0, half_w), max(2.0, half_h)))
    brows = _lab_mean_of(brow_patches)

    # --- forehead skin: between the brow line and the top of the face oval
    brow_y = min(float(px[I_BROW_TOP_R, 1]), float(px[I_BROW_TOP_L, 1]))
    top_y = float(px[I_FOREHEAD, 1])
    forehead = [0.0, 0.0, 0.0]
    if brow_y > top_y + 4.0:
        span = brow_y - top_y
        forehead = _lab_mean(_patch(img, float(px[I_NOSE_TIP, 0]),
                                    top_y + 0.55 * span,
                                    0.35 * interocular,
                                    max(2.0, 0.15 * span)))

    # --- cheeks: midway between the outer eye corner and the mouth corner
    cheek_r = (px[I_EYE_OUT_R] + px[I_MOUTH_R]) * 0.5
    cheek_l = (px[I_EYE_OUT_L] + px[I_MOUTH_L]) * 0.5
    half = max(3.0, 0.16 * interocular)
    patch_r = _patch(img, cheek_r[0], cheek_r[1], half, half)
    patch_l = _patch(img, cheek_l[0], cheek_l[1], half, half)
    cheek = _lab_mean_of([patch_r, patch_l])

    # --- hair: a band above the forehead, absent when the crop cuts the head
    face_h = _dist(px[I_FOREHEAD], px[I_CHIN])
    hair_top = top_y - 0.42 * face_h
    hair_bottom = top_y - 0.12 * face_h
    hair = [0.0, 0.0, 0.0]
    if hair_bottom > 0.0 and hair_bottom > hair_top:
        hair = _lab_mean(_patch(img, float(px[I_FOREHEAD, 0]),
                                0.5 * (hair_top + hair_bottom),
                                0.30 * interocular,
                                max(2.0, 0.5 * (hair_bottom - hair_top))))

    texture = _lbp6(patch_r if patch_r is not None else patch_l)

    return _fit(iris + lips + brows + forehead + cheek + hair + texture,
                PHOTO_DIMS)


def face_descriptor(img_bgr, face: dict) -> list[float]:
    """64 float identity signature: 40 geometric ratios then 24 photometric.

    Geometric block: every distance is measured after the mesh has been
    de-rotated by the eye line and divided by the inter-ocular distance, so the
    block is invariant to scale and to in-plane rotation by construction.
    Photometric block: mean Lab of iris, lips, eyebrows, forehead, cheek and
    hair, plus a rotation invariant texture histogram of the cheek.

    Returns ``[]`` - never raises - when the mesh is missing or degenerate.
    ``img_bgr`` must be the same framing the mesh was measured on.
    """
    img = _as_bgr(img_bgr)
    if img is None or not isinstance(face, dict):
        return []
    mesh = face.get("mesh")
    try:                             # a caller may hand back an ndarray mesh
        if mesh is None or len(mesh) < 468:
            return []
    except TypeError:
        return []
    height, width = img.shape[:2]
    try:
        px = _mesh_px(mesh, width, height)
    except (ValueError, TypeError):
        return []
    if not np.all(np.isfinite(px)):
        return []

    eyes = _eye_centers(px)
    if eyes is None:
        return []
    right_eye, left_eye = eyes
    interocular = _dist(right_eye, left_eye)
    if interocular < 2.0:            # face too small to measure anything from
        return []

    yaw = face.get("yaw", 0.0)
    try:
        yaw = float(yaw)
    except (TypeError, ValueError):
        yaw = 0.0

    canonical = _canonical(px, right_eye, left_eye, interocular, yaw)
    values = _geometric_features(canonical) + _photometric_features(img, px,
                                                                    interocular)
    vector = np.asarray(_fit(values, DESCRIPTOR_DIMS), dtype=np.float64)
    vector = np.nan_to_num(vector, nan=0.0, posinf=0.0, neginf=0.0)
    norm = float(np.linalg.norm(vector))
    if norm < 1e-9:
        return []
    return [round(float(v), 6) for v in (vector / norm)]


def _as_vector(descriptor):
    if descriptor is None:
        return None
    try:
        arr = np.asarray(descriptor, dtype=np.float64).ravel()
    except (TypeError, ValueError):
        return None
    if arr.size != DESCRIPTOR_DIMS or not np.all(np.isfinite(arr)):
        return None
    return arr


def _block_similarity(a: np.ndarray, b: np.ndarray):
    norm_a = float(np.linalg.norm(a))
    norm_b = float(np.linalg.norm(b))
    if norm_a < 1e-9 or norm_b < 1e-9:
        return None
    cosine = float(np.dot(a, b) / (norm_a * norm_b))
    cosine = max(-1.0, min(1.0, cosine))
    return (cosine + 1.0) / 2.0


def compare_faces(desc_a, desc_b) -> float:
    """0..1 similarity, 1 = identical.

    The two blocks are compared separately and then mixed 0.7 geometric /
    0.3 photometric: bone geometry is what identity actually rests on, while
    colour statistics move with lighting and makeup and only get a vote.
    A block that is entirely zero (region crops that fell outside the frame)
    is dropped and the remaining weight is renormalised, so a partially
    measurable face still yields a usable number instead of a false mismatch.
    """
    a = _as_vector(desc_a)
    b = _as_vector(desc_b)
    if a is None or b is None:
        return 0.0
    parts = []
    geometric = _block_similarity(a[:GEO_DIMS], b[:GEO_DIMS])
    if geometric is not None:
        parts.append((0.7, geometric))
    photometric = _block_similarity(a[GEO_DIMS:], b[GEO_DIMS:])
    if photometric is not None:
        parts.append((0.3, photometric))
    if not parts:
        return 0.0
    total = sum(weight for weight, _ in parts)
    score = sum(weight * value for weight, value in parts) / total
    return round(max(0.0, min(1.0, float(score))), 6)
