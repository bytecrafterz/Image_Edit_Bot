"""Shot type and framing of a photograph.

Everything downstream branches on this: which catalog options apply, which body
metrics can honestly be measured (leg length needs feet in the frame), how the
prompt describes the crop, and what the planner is allowed to vary.  Getting it
from pose landmarks rather than from the face size alone matters, because the
face occupies a very different fraction of the frame in a wide angle selfie
than in a portrait taken from three metres away.  Anatomy is the reliable
signal: if the ankles are in the picture it is a full shot, whatever the lens
was doing.
"""
from __future__ import annotations

import math
from typing import Any

import numpy as np

__all__ = ["classify_shot"]

_VIS_STRONG = 0.60      # a joint we are willing to classify on
_VIS_ANY = 0.40         # a joint we are willing to include in the person box
_CLOSEUP_FACE = 0.28    # face box height / frame height
_FACE_DOMINATES = 0.32  # above this the face alone settles it: a hip landmark
                        # at this scale is geometrically impossible


def _clamp(value: float, lo: float = 0.0, hi: float = 1.0) -> float:
    if not math.isfinite(value):
        return lo
    return float(max(lo, min(hi, value)))


def _landmarks(pose: Any) -> dict:
    if not isinstance(pose, dict) or not pose.get("ok"):
        return {}
    lms = pose.get("landmarks")
    return lms if isinstance(lms, dict) else {}


def _point(lms: dict, name: str, min_vis: float,
           y_max: float = 1.0) -> tuple[float, float, float] | None:
    """A landmark counts only when it is confident *and* inside the frame.
    MediaPipe keeps emitting joints it has extrapolated past the crop, and a
    hallucinated hip is exactly what turns a closeup into a wrong 'half'."""
    raw = lms.get(name)
    if not isinstance(raw, dict):
        return None
    try:
        x = float(raw.get("x"))
        y = float(raw.get("y"))
        v = float(raw.get("v", 1.0))
    except (TypeError, ValueError):
        return None
    if not (math.isfinite(x) and math.isfinite(y) and math.isfinite(v)):
        return None
    if v < min_vis:
        return None
    if not (-0.02 <= x <= 1.02 and -0.02 <= y <= y_max):
        return None
    return x, y, v


def _group(lms: dict, names: tuple[str, ...], min_vis: float,
           y_max: float = 1.0) -> list[tuple[float, float, float]]:
    found = []
    for name in names:
        point = _point(lms, name, min_vis, y_max)
        if point is not None:
            found.append(point)
    return found


def _mean_vis(points: list[tuple[float, float, float]]) -> float:
    return sum(p[2] for p in points) / len(points) if points else 0.0


def _confidence(base: float, span: float, mean_vis: float) -> float:
    """Slide from base to base+span as the deciding joints get more confident."""
    return round(_clamp(base + span * _clamp((mean_vis - 0.60) / 0.40), 0.0, 0.97), 3)


def _face_norm(face: Any, width: int, height: int) -> tuple[float, float, float, float] | None:
    """Face box as (x, y, w, h) normalised to the frame."""
    if not isinstance(face, dict) or not face.get("ok", True):
        return None
    norm = face.get("bbox_norm")
    values: list[float] = []
    if isinstance(norm, (list, tuple)) and len(norm) == 4:
        try:
            values = [float(v) for v in norm]
        except (TypeError, ValueError):
            values = []
    if (not values or values[2] <= 0 or values[3] <= 0) and width > 0 and height > 0:
        box = face.get("bbox")
        if isinstance(box, (list, tuple)) and len(box) == 4:
            try:
                bx, by, bw, bh = (float(v) for v in box)
                values = [bx / width, by / height, bw / width, bh / height]
            except (TypeError, ValueError, ZeroDivisionError):
                values = []
    if len(values) != 4 or not all(math.isfinite(v) for v in values):
        return None
    if values[2] <= 0.0 or values[3] <= 0.0:
        return None
    return values[0], values[1], values[2], values[3]


def _head_top(lms: dict, face_box: tuple[float, float, float, float] | None) -> float | None:
    """Normalised y of the crown of the head.

    A face detector's box starts around the brow line, so the crown sits above
    it; from pose only, the eye-to-shoulder span gives the head scale (eyes sit
    about halfway between crown and shoulder line).
    """
    if face_box is not None:
        return face_box[1] - 0.20 * face_box[3]

    eyes = _group(lms, ("left_eye", "right_eye"), _VIS_ANY, y_max=1.05)
    if not eyes:
        nose = _point(lms, "nose", _VIS_ANY, y_max=1.05)
        eyes = [nose] if nose else []
    shoulders = _group(lms, ("left_shoulder", "right_shoulder"), _VIS_ANY, y_max=1.05)
    if not eyes:
        return None
    eye_y = sum(p[1] for p in eyes) / len(eyes)
    if shoulders:
        shoulder_y = sum(p[1] for p in shoulders) / len(shoulders)
        span = shoulder_y - eye_y
        if span > 0.01:
            return eye_y - 0.50 * span
    ears = _group(lms, ("left_ear", "right_ear"), _VIS_ANY, y_max=1.05)
    if len(ears) == 2:
        head_w = abs(ears[0][0] - ears[1][0])
        if head_w > 0.005:
            return eye_y - 0.55 * head_w
    return None


def _person_box(lms: dict, face_box: tuple[float, float, float, float] | None,
                head_top: float | None) -> tuple[list[float], str] | tuple[None, str]:
    xs: list[float] = []
    ys: list[float] = []
    for name in lms:
        point = _point(lms, name, _VIS_ANY, y_max=1.02)
        if point is not None:
            xs.append(point[0])
            ys.append(point[1])
    source = "pose" if len(xs) >= 4 else ""

    if face_box is not None:
        fx, fy, fw, fh = face_box
        xs.extend([fx, fx + fw])
        ys.extend([fy, fy + fh])
        source = source or "face"
    if head_top is not None:
        ys.append(head_top)
    if not xs or not ys:
        return None, ""

    x0, x1 = min(xs), max(xs)
    y0, y1 = min(ys), max(ys)
    if source == "pose":
        # Joints are the skeleton, not the silhouette: widen a little so the
        # fill figure is not systematically under the truth.
        pad_x = 0.08 * (x1 - x0)
        pad_y = 0.02 * (y1 - y0)
        x0, x1 = x0 - pad_x, x1 + pad_x
        y0, y1 = y0 - pad_y, y1 + pad_y
    return [_clamp(x0), _clamp(y0), _clamp(x1), _clamp(y1)], source


def classify_shot(img_bgr: Any, pose: dict, face: dict) -> dict:
    """Decide closeup / half / full from anatomy first, face size second."""
    if isinstance(img_bgr, np.ndarray) and img_bgr.ndim >= 2 and img_bgr.size:
        height, width = int(img_bgr.shape[0]), int(img_bgr.shape[1])
    else:
        height = width = 0

    if width > 0 and height > 0:
        ratio = width / float(height)
        orientation = ("portrait" if ratio < 0.95
                       else "landscape" if ratio > 1.05 else "square")
    else:
        orientation = "unknown"

    lms = _landmarks(pose)
    face_box = _face_norm(face, width, height)
    face_ratio = face_box[3] if face_box is not None else None

    feet = _group(lms, ("left_ankle", "right_ankle", "left_heel", "right_heel",
                        "left_foot_index", "right_foot_index"), _VIS_STRONG)
    knees = _group(lms, ("left_knee", "right_knee"), _VIS_STRONG)
    hips = _group(lms, ("left_hip", "right_hip"), _VIS_STRONG, y_max=0.99)
    shoulders = _group(lms, ("left_shoulder", "right_shoulder"), _VIS_ANY, y_max=1.02)

    shot = "unknown"
    confidence = 0.0
    reason = "no pose and no face"

    if face_ratio is not None and face_ratio >= _FACE_DOMINATES:
        # The face fills a third of the frame: nothing below the chest can be
        # in shot, so any hip landmark here is an extrapolation.
        shot = "closeup"
        confidence = 0.62 if hips else 0.86
        reason = "face box dominates the frame"
    elif feet:
        shot = "full"
        confidence = _confidence(0.72, 0.23, _mean_vis(feet))
        reason = "feet in frame"
    elif knees:
        shot = "full"
        confidence = _confidence(0.55, 0.17, _mean_vis(knees))
        reason = "knees in frame, feet cropped"
    elif hips:
        shot = "half"
        bonus = 0.03 if shoulders else 0.0
        confidence = round(min(0.92, _confidence(0.68, 0.20, _mean_vis(hips)) + bonus), 3)
        reason = "hips in frame, legs cropped"
    elif face_ratio is not None and face_ratio > _CLOSEUP_FACE:
        shot = "closeup"
        confidence = 0.80
        reason = "large face, no hips"
    elif face_ratio is not None:
        # Fallback: size of the head against the frame, deliberately uncertain.
        if face_ratio >= 0.13:
            shot, confidence, reason = "half", 0.45, "face size heuristic"
        elif face_ratio >= 0.05:
            shot, confidence, reason = "full", 0.40, "face size heuristic"
        else:
            shot, confidence, reason = "full", 0.30, "very small face, wide frame"
    elif shoulders and _point(lms, "nose", _VIS_ANY, y_max=1.02):
        shot, confidence, reason = "closeup", 0.35, "head and shoulders only"
    elif shoulders:
        shot, confidence, reason = "half", 0.25, "torso landmarks only"
    else:
        shot, confidence = "unknown", 0.0

    head_top = _head_top(lms, face_box)
    box, box_source = _person_box(lms, face_box, head_top)

    head_room = None if head_top is None else round(_clamp(head_top), 4)
    subject_fill = None
    if box is not None:
        subject_fill = round(_clamp((box[2] - box[0]) * (box[3] - box[1])), 4)

    return {
        "shot_type": shot,
        "confidence": float(confidence),
        "framing": {"head_room": head_room, "subject_fill": subject_fill},
        "orientation": orientation,
        "reason": reason,
        "face_ratio": None if face_ratio is None else round(float(face_ratio), 4),
        "person_bbox_norm": box if box is not None else [],
        "fill_source": box_source,
        "landmarks_used": {"feet": len(feet), "knees": len(knees),
                           "hips": len(hips), "shoulders": len(shoulders)},
    }
