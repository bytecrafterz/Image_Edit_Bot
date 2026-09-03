"""Body pose landmarks - the anchor points every body measurement is built on.

The client's complaint was that a previous tool silently slimmed her down.  The
only way to answer that objectively is to measure the same skeletal points in
the original and in the generated image, so this module exists to produce those
points reliably and to never, under any circumstance, take the whole pipeline
down when MediaPipe is absent or finds nothing.

MediaPipe solution objects wrap a stateful C++ calculator graph: they are not
thread safe and they cost roughly a second to build.  Both facts are handled
here and nowhere else - one lazily created singleton, every call serialised
through one module level lock, so the rest of the application can call this
from the job thread pool without knowing any of it.
"""
from __future__ import annotations

import threading

import cv2
import numpy as np

try:
    import mediapipe as mp
    MEDIAPIPE_AVAILABLE = True
except ImportError:  # optional at runtime - the app must still boot without it
    mp = None
    MEDIAPIPE_AVAILABLE = False

# MediaPipe Pose always emits exactly 33 landmarks in this order.  The names in
# CONTRACTS.md are a subset of these; the extras (mouth, hand tips, eye corners)
# are kept because the anomaly and hand checks need them.
POSE_LANDMARK_NAMES: tuple[str, ...] = (
    "nose",
    "left_eye_inner", "left_eye", "left_eye_outer",
    "right_eye_inner", "right_eye", "right_eye_outer",
    "left_ear", "right_ear",
    "mouth_left", "mouth_right",
    "left_shoulder", "right_shoulder",
    "left_elbow", "right_elbow",
    "left_wrist", "right_wrist",
    "left_pinky", "right_pinky",
    "left_index", "right_index",
    "left_thumb", "right_thumb",
    "left_hip", "right_hip",
    "left_knee", "right_knee",
    "left_ankle", "right_ankle",
    "left_heel", "right_heel",
    "left_foot_index", "right_foot_index",
)

# Detection runs on a downscaled copy; landmarks come back normalised, so this
# is pure speed and changes nothing about the result.
MAX_SIDE = 1280
VISIBILITY_MIN = 0.5

_REASON_NO_MP = "mediapipe no disponible"

_LOCK = threading.Lock()
_POSE = None
_INIT_FAILED = False


# ------------------------------------------------------------------- helpers

def _blank(reason: str, backend: str = "none") -> dict:
    """Every failure path returns the same shape so callers never branch."""
    return {
        "ok": False,
        "landmarks": {},
        "visible_count": 0,
        "bbox_norm": [],
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


def _detector():
    """Lazy singleton.  Caller must already hold ``_LOCK``."""
    global _POSE, _INIT_FAILED
    if _POSE is not None or _INIT_FAILED:
        return _POSE
    try:
        _POSE = mp.solutions.pose.Pose(
            static_image_mode=True,
            model_complexity=2,
            enable_segmentation=False,
            min_detection_confidence=0.5,
        )
    except Exception:  # broken install, missing model asset, out of memory
        _INIT_FAILED = True
        _POSE = None
    return _POSE


def _clamp01(value: float) -> float:
    if value < 0.0:
        return 0.0
    if value > 1.0:
        return 1.0
    return value


def _bbox_norm(points: list[tuple[float, float]]) -> list[float]:
    if not points:
        return []
    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    x0, x1 = _clamp01(min(xs)), _clamp01(max(xs))
    y0, y1 = _clamp01(min(ys)), _clamp01(max(ys))
    return [round(x0, 5), round(y0, 5), round(max(0.0, x1 - x0), 5),
            round(max(0.0, y1 - y0), 5)]


# -------------------------------------------------------------------- public

def detect_pose(img_bgr) -> dict:
    """Locate the 33 body landmarks of the most prominent person.

    Returns ``{"ok", "landmarks", "visible_count", "bbox_norm", "backend",
    "reason"}``.  Landmark coordinates are normalised to the frame and clamped
    to 0..1 so a consumer can multiply by width/height and index pixels without
    a bounds check; ``v`` (visibility) is the honest signal for "trust this
    point" - MediaPipe happily extrapolates limbs that left the frame.
    """
    if not MEDIAPIPE_AVAILABLE:
        return _blank(_REASON_NO_MP)

    img = _as_bgr(img_bgr)
    if img is None:
        return _blank("imagen invalida")

    rgb = cv2.cvtColor(_downscale(img, MAX_SIDE), cv2.COLOR_BGR2RGB)

    with _LOCK:
        detector = _detector()
        if detector is None:
            return _blank("mediapipe pose no se pudo inicializar")
        try:
            result = detector.process(rgb)
        except Exception as exc:  # a bad frame must not kill the run
            return _blank("fallo la deteccion de pose: %s" % exc, "mediapipe")

    landmark_list = getattr(result, "pose_landmarks", None)
    raw = list(getattr(landmark_list, "landmark", None) or [])
    if not raw:
        return _blank("no se detecto ninguna persona", "mediapipe")

    landmarks: dict[str, dict[str, float]] = {}
    visible: list[tuple[float, float]] = []
    every: list[tuple[float, float]] = []
    visible_count = 0

    try:
        for index, name in enumerate(POSE_LANDMARK_NAMES):
            if index >= len(raw):
                break
            point = raw[index]
            x = _clamp01(float(point.x))
            y = _clamp01(float(point.y))
            v = float(getattr(point, "visibility", 0.0))
            landmarks[name] = {
                "x": round(x, 5),
                "y": round(y, 5),
                "z": round(float(getattr(point, "z", 0.0)), 5),
                "v": round(v, 4),
            }
            every.append((x, y))
            if v >= VISIBILITY_MIN:
                visible_count += 1
                visible.append((x, y))
    except (AttributeError, TypeError, ValueError):
        return _blank("resultado de pose ilegible", "mediapipe")

    if not landmarks:
        return _blank("no se detecto ninguna persona", "mediapipe")

    # Four confident points make a meaningful box; below that use everything
    # MediaPipe guessed rather than reporting no box at all.
    box_source = visible if len(visible) >= 4 else every

    return {
        "ok": True,
        "landmarks": landmarks,
        "visible_count": visible_count,
        "bbox_norm": _bbox_norm(box_source),
        "backend": "mediapipe",
        "reason": "",
    }
