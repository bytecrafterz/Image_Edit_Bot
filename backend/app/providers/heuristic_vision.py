"""The free vision provider: the robot's eyes when there is no API key.

Everything here is measured from pixels by the analysis package - no network,
no account, no cost.  It is what makes the whole pipeline demonstrable at zero
cost and, more importantly, it is what makes identity preservation *measured*
rather than requested: describe_photo reports what the photograph actually
contains, and critique_result converts the numeric identity verdict and the
anatomy scan into the same defect contract a paid vision model would return.

Every analysis module is imported lazily and every call is wrapped: a missing
or half written sibling module degrades this provider to fewer findings, never
to an exception.
"""
from __future__ import annotations

import importlib
import logging
from typing import Any

import cv2
import numpy as np

from .base import VisionProvider

log = logging.getLogger("photorobot.providers.heuristic")

WORK_SIDE = 1280          # measurements are ratios, so a downscale is free
_MIN_MASK_PX = 250

# Canonical MediaPipe FaceMesh indices, repeated here so this module does not
# reach into the private constants of analysis/face.py.
_M_MOUTH_R, _M_MOUTH_L = 61, 291
_M_LIP_TOP, _M_LIP_BOT = 0, 17
_M_LIP_IN_TOP, _M_LIP_IN_BOT = 13, 14
_M_EYE_OUT_R, _M_EYE_IN_R, _M_EYE_TOP_R, _M_EYE_BOT_R = 33, 133, 159, 145
_M_EYE_OUT_L, _M_EYE_IN_L, _M_EYE_TOP_L, _M_EYE_BOT_L = 263, 362, 386, 374

_MODULES: dict[str, Any] = {}

_HAIR_SHORT = 0.15        # hair bottom below the chin, in torso lengths
_HAIR_MEDIUM = 0.70


def _mod(dotted: str):
    """Import an analysis/identity module once.  None when unavailable."""
    if dotted in _MODULES:
        return _MODULES[dotted]
    try:
        module = importlib.import_module(dotted, __package__)
    except Exception as exc:
        log.warning("Modulo %s no disponible: %s", dotted, exc)
        module = None
    _MODULES[dotted] = module
    return module


def _safe(fn, *args, **kwargs):
    if fn is None:
        return None
    try:
        return fn(*args, **kwargs)
    except Exception as exc:
        log.warning("%s fallo: %s", getattr(fn, "__name__", fn), exc)
        return None


def _d(value: Any) -> dict:
    return value if isinstance(value, dict) else {}


def _f(value: Any, default: float = 0.0) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return default
    return default if out != out else out


def _clamp01(value: float) -> float:
    return 0.0 if value < 0.0 else 1.0 if value > 1.0 else value


# -------------------------------------------------------------------- masks

def _bool_mask(mask: Any, shape: tuple[int, int]) -> np.ndarray | None:
    if not isinstance(mask, np.ndarray) or mask.size == 0:
        return None
    arr = mask[:, :, 0] if mask.ndim == 3 else mask
    if arr.shape[:2] != shape:
        try:
            arr = cv2.resize(arr.astype(np.uint8), (shape[1], shape[0]),
                             interpolation=cv2.INTER_NEAREST)
        except cv2.error:
            return None
    if arr.dtype == bool:
        out = arr
    else:
        peak = float(arr.max()) if arr.size else 0.0
        out = arr > (0.5 if peak <= 1.0 else 127)
    return out if out.any() else None


# ------------------------------------------------------------------ colours

_HUE_NAMES = ((12, "red"), (45, "orange"), (70, "yellow"), (170, "green"),
              (200, "teal"), (255, "blue"), (290, "purple"), (330, "magenta"),
              (345, "pink"), (361, "red"))


def _colour_name(bgr: Any) -> str:
    """A plain English colour name - this text goes straight into a prompt."""
    try:
        vals = [int(max(0, min(255, round(float(c))))) for c in list(bgr)[:3]]
    except (TypeError, ValueError):
        return ""
    if len(vals) < 3:
        return ""
    hsv = cv2.cvtColor(np.uint8([[vals]]), cv2.COLOR_BGR2HSV)[0][0]
    hue = float(hsv[0]) * 2.0
    sat = float(hsv[1]) / 255.0
    val = float(hsv[2]) / 255.0

    if val < 0.10:
        return "black"
    if sat < 0.14:
        if val < 0.28:
            return "near black"
        if val < 0.45:
            return "dark grey"
        if val < 0.68:
            return "grey"
        if val < 0.88:
            return "light grey"
        return "white"

    if 8.0 <= hue < 48.0:                       # the skin, wood and fabric band
        if val < 0.42:
            return "dark brown"
        if sat < 0.32:
            return "beige" if val > 0.62 else "taupe"
        if val < 0.66:
            return "brown"
        if sat < 0.55:
            return "tan"
        return "orange" if hue >= 22.0 else "terracotta"

    name = "red"
    for edge, label in _HUE_NAMES:
        if hue < edge:
            name = label
            break
    if name == "red" and val > 0.78 and sat < 0.55:
        return "pink"
    if val < 0.35:
        return f"deep {name}"
    if val > 0.86 and sat < 0.45:
        return f"pale {name}"
    if sat > 0.75 and val > 0.62:
        return f"bright {name}"
    return name


def _sample_pixels(img: np.ndarray, mask: np.ndarray | None,
                   limit: int = 4000) -> np.ndarray | None:
    pixels = img[mask] if mask is not None else img.reshape(-1, img.shape[-1])
    if pixels.shape[0] == 0:
        return None
    if pixels.shape[0] > limit:
        idx = np.linspace(0, pixels.shape[0] - 1, limit).astype(np.int64)
        pixels = pixels[idx]
    return pixels.astype(np.float32)


def _dominant_names(img: np.ndarray, mask: np.ndarray | None = None,
                    k: int = 3) -> list[str]:
    """Dominant colours by k-means, ordered by how much of the area they own."""
    pixels = _sample_pixels(img, mask)
    if pixels is None:
        return []
    clusters = int(max(1, min(k, pixels.shape[0])))
    criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_MAX_ITER, 12, 1.0)
    try:
        _, labels, centers = cv2.kmeans(pixels, clusters, None, criteria, 3,
                                        cv2.KMEANS_PP_CENTERS)
    except cv2.error:
        return [n for n in [_colour_name(pixels.mean(axis=0))] if n]
    counts = np.bincount(labels.reshape(-1), minlength=clusters)
    floor = max(1.0, pixels.shape[0] * 0.08)
    names: list[str] = []
    for index in np.argsort(-counts):
        if counts[index] < floor:
            continue
        name = _colour_name(centers[index])
        if name and name not in names:
            names.append(name)
    return names


# ----------------------------------------------------------------- geometry

def _lm(pose_d: dict, name: str, width: int, height: int,
        min_v: float = 0.35) -> tuple[float, float] | None:
    marks = _d(pose_d).get("landmarks")
    if not isinstance(marks, dict):
        return None
    point = marks.get(name)
    if not isinstance(point, dict):
        return None
    if _f(point.get("v"), 1.0) < min_v:
        return None
    return (_f(point.get("x")) * width, _f(point.get("y")) * height)


def _mid(a, b):
    if a is None or b is None:
        return None
    return ((a[0] + b[0]) / 2.0, (a[1] + b[1]) / 2.0)


def _dist(a, b) -> float:
    if a is None or b is None:
        return 0.0
    return float(np.hypot(a[0] - b[0], a[1] - b[1]))


def _torso_px(pose_d: dict, width: int, height: int) -> float:
    shoulders = _mid(_lm(pose_d, "left_shoulder", width, height),
                     _lm(pose_d, "right_shoulder", width, height))
    hips = _mid(_lm(pose_d, "left_hip", width, height),
                _lm(pose_d, "right_hip", width, height))
    return _dist(shoulders, hips)


def _face_box(face_d: dict) -> list[float]:
    box = _d(face_d).get("bbox")
    if isinstance(box, (list, tuple)) and len(box) == 4:
        vals = [_f(v) for v in box]
        if vals[2] > 1 and vals[3] > 1:
            return vals
    return []


def _mesh_pt(face_d: dict, index: int, width: int,
             height: int) -> tuple[float, float] | None:
    mesh = _d(face_d).get("mesh")
    if not isinstance(mesh, list) or index >= len(mesh):
        return None
    point = mesh[index]
    if not isinstance(point, (list, tuple)) or len(point) < 2:
        return None
    return (_f(point[0]) * width, _f(point[1]) * height)


# ----------------------------------------------------------------- lighting

def _region_stats(lab_l: np.ndarray, mask: np.ndarray | None) -> dict:
    values = lab_l[mask] if mask is not None else lab_l.reshape(-1)
    if values.size < 50:
        return {}
    p05, p50, p95 = np.percentile(values, [5, 50, 95])
    return {"mean": float(values.mean()), "median": float(p50),
            "p05": float(p05), "p95": float(p95)}


def _lighting(img: np.ndarray, subject: np.ndarray | None,
              face_box: list[float]) -> dict:
    """Direction and hardness from the luminance field, not from a guess."""
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    lum = lab[:, :, 0].astype(np.float32)
    warmth = float(lab[:, :, 2].mean()) - 128.0
    height, width = lum.shape[:2]

    mask = subject
    if mask is None and face_box:
        mask = np.zeros((height, width), dtype=bool)
        x, y, w, h = (int(round(v)) for v in face_box)
        mask[max(0, y):min(height, y + h), max(0, x):min(width, x + w)] = True
        if not mask.any():
            mask = None

    stats = _region_stats(lum, mask)
    if not stats:
        return {"direction": "unknown", "hardness": "unknown",
                "key": "unknown", "warmth": "neutral", "text": ""}

    if mask is not None:
        ys, xs = np.nonzero(mask)
        y0, y1 = int(ys.min()), int(ys.max()) + 1
        x0, x1 = int(xs.min()), int(xs.max()) + 1
        sub_mask = mask[y0:y1, x0:x1]
    else:
        y0, y1, x0, x1 = 0, height, 0, width
        sub_mask = np.ones((height, width), dtype=bool)
    sub_lum = lum[y0:y1, x0:x1]
    mid_x = sub_mask.shape[1] // 2
    mid_y = sub_mask.shape[0] // 2

    def _half(rows: slice, cols: slice) -> float:
        piece = sub_lum[rows, cols][sub_mask[rows, cols]]
        return float(piece.mean()) if piece.size > 30 else float("nan")

    left = _half(slice(None), slice(0, mid_x))
    right = _half(slice(None), slice(mid_x, None))
    top = _half(slice(0, mid_y), slice(None))
    bottom = _half(slice(mid_y, None), slice(None))

    base = max(1.0, stats["mean"])
    direction = "frontal and even"
    if mask is not None:
        background = _region_stats(lum, ~mask)
        if background and background["mean"] - stats["mean"] > 22.0:
            direction = "backlit, background brighter than the subject"
    if direction == "frontal and even" and left == left and right == right:
        delta = (right - left) / base
        if delta > 0.08:
            direction = "from the right of the frame"
        elif delta < -0.08:
            direction = "from the left of the frame"
    if direction == "frontal and even" and top == top and bottom == bottom:
        if (top - bottom) / base > 0.12:
            direction = "from above"
        elif (bottom - top) / base > 0.12:
            direction = "from below"

    spread = (stats["p95"] - stats["p05"]) / 255.0
    hardness = "soft"
    if spread > 0.62:
        hardness = "hard, with defined shadows"
    elif spread < 0.28:
        hardness = "flat, almost shadowless"
    level = stats["mean"] / 255.0
    key = "balanced exposure"
    if level > 0.68:
        key = "bright, high key"
    elif level < 0.32:
        key = "dark, low key"
    temperature = "warm" if warmth > 6.0 else "cool" if warmth < -6.0 else "neutral"

    return {"direction": direction, "hardness": hardness, "key": key,
            "warmth": temperature, "spread": round(spread, 3),
            "level": round(level, 3),
            "text": f"{hardness} light {direction}, {key}, {temperature} tone"}


# --------------------------------------------------------------- appearance

def _hair_text(img: np.ndarray, regions: dict, face_d: dict,
               pose_d: dict) -> tuple[str, dict]:
    height, width = img.shape[:2]
    hair = _bool_mask(regions.get("hair"), (height, width))
    if hair is None or int(hair.sum()) < _MIN_MASK_PX:
        return "", {}
    names = _dominant_names(img, hair, k=2)
    colour = names[0] if names else _colour_name(img[hair].mean(axis=0))

    # The hair region mask is built from the area above the face, so its lowest
    # row sits at the chin no matter how long the hair really is - every subject
    # came out "short", and that word went straight into the prompt telling the
    # generator to cut her hair.  identity.profile grows the same colour down
    # through the silhouette to find the real end of the hair; reuse it rather
    # than keep a second, wronger copy of the measurement here.
    grown = hair
    try:
        from ..identity.profile import _hair_sample

        measured = _hair_sample(img, regions, face_d, pose_d)
        if measured.get("ok") and measured.get("length_ratio") is not None:
            ratio = float(measured["length_ratio"])
            length = ("short" if ratio <= _HAIR_SHORT else
                      "medium length" if ratio <= _HAIR_MEDIUM else "long")
            text = " ".join(p for p in (length, colour, "hair") if p)
            return text.strip(), {"colour": colour, "length": length,
                                  "length_ratio": round(ratio, 3)}
    except Exception:                                  # noqa: BLE001
        pass                                           # fall through to the local estimate

    rows = np.flatnonzero(grown.any(axis=1))
    bottom = float(rows[-1]) if rows.size else 0.0
    box = _face_box(face_d)
    chin = box[1] + box[3] if box else None
    torso = _torso_px(pose_d, width, height)
    length = ""
    ratio = None
    if chin is not None and torso > 1.0:
        ratio = (bottom - chin) / torso
        length = ("short" if ratio <= _HAIR_SHORT else
                  "medium length" if ratio <= _HAIR_MEDIUM else "long")
    elif chin is not None and box:
        ratio = (bottom - chin) / max(1.0, box[3])
        length = ("short" if ratio <= 0.5 else
                  "medium length" if ratio <= 2.0 else "long")
    text = " ".join(p for p in (length, colour, "hair") if p)
    return text.strip(), {"colour": colour, "length": length,
                          "length_ratio": None if ratio is None else round(ratio, 3)}


def _clothing_text(img: np.ndarray, garment: np.ndarray | None, regions: dict,
                   pose_d: dict) -> tuple[str, dict]:
    height, width = img.shape[:2]
    cloth = _bool_mask(garment, (height, width))
    if cloth is None:
        cloth = _bool_mask(regions.get("upper_body"), (height, width))
    if cloth is None or int(cloth.sum()) < _MIN_MASK_PX:
        return "", {}

    hips = _mid(_lm(pose_d, "left_hip", width, height),
                _lm(pose_d, "right_hip", width, height))
    split = int(hips[1]) if hips else None
    upper = cloth.copy()
    lower = None
    if split is not None and 0 < split < height - 8:
        upper = cloth.copy()
        upper[split:, :] = False
        lower = cloth.copy()
        lower[:split, :] = False
        if int(lower.sum()) < _MIN_MASK_PX:
            lower = None
        if int(upper.sum()) < _MIN_MASK_PX:
            upper = cloth

    upper_names = _dominant_names(img, upper, k=2)
    lower_names = _dominant_names(img, lower, k=2) if lower is not None else []

    sleeves = ""
    arms = _bool_mask(regions.get("arms"), (height, width))
    covered = None
    if arms is not None:
        overlap = float(np.logical_and(arms, cloth).sum())
        covered = overlap / float(max(1, int(arms.sum())))
        sleeves = ("long sleeved" if covered > 0.55 else
                   "sleeveless" if covered < 0.15 else "short sleeved")

    parts = []
    if upper_names:
        parts.append(" ".join(p for p in (sleeves, upper_names[0],
                                          "top") if p))
    if lower_names:
        parts.append(f"{lower_names[0]} lower garment")
    text = " and ".join(parts)
    detail = {"upper_colours": upper_names, "lower_colours": lower_names,
              "sleeves": sleeves,
              "arm_coverage": None if covered is None else round(covered, 3)}
    return text, detail


def _expression_text(face_d: dict, width: int, height: int) -> str:
    if not _d(face_d).get("ok"):
        return ""
    corner_r = _mesh_pt(face_d, _M_MOUTH_R, width, height)
    corner_l = _mesh_pt(face_d, _M_MOUTH_L, width, height)
    lip_top = _mesh_pt(face_d, _M_LIP_TOP, width, height)
    lip_bot = _mesh_pt(face_d, _M_LIP_BOT, width, height)
    inner_top = _mesh_pt(face_d, _M_LIP_IN_TOP, width, height)
    inner_bot = _mesh_pt(face_d, _M_LIP_IN_BOT, width, height)

    parts: list[str] = []
    mouth_w = _dist(corner_r, corner_l)
    if mouth_w > 1.0 and inner_top and inner_bot and lip_top and lip_bot:
        opening = _dist(inner_top, inner_bot) / mouth_w
        lift = (((lip_top[1] + lip_bot[1]) / 2.0)
                - ((corner_r[1] + corner_l[1]) / 2.0)) / mouth_w
        if lift > 0.06:
            parts.append("smiling" if opening > 0.14 else "soft closed smile")
        elif opening > 0.16:
            parts.append("mouth open")
        else:
            parts.append("neutral mouth")

    eye_ratio = []
    for out_i, in_i, top_i, bot_i in (
            (_M_EYE_OUT_R, _M_EYE_IN_R, _M_EYE_TOP_R, _M_EYE_BOT_R),
            (_M_EYE_OUT_L, _M_EYE_IN_L, _M_EYE_TOP_L, _M_EYE_BOT_L)):
        span = _dist(_mesh_pt(face_d, out_i, width, height),
                     _mesh_pt(face_d, in_i, width, height))
        lid = _dist(_mesh_pt(face_d, top_i, width, height),
                    _mesh_pt(face_d, bot_i, width, height))
        if span > 1.0:
            eye_ratio.append(lid / span)
    if eye_ratio:
        mean_ratio = sum(eye_ratio) / len(eye_ratio)
        parts.append("eyes narrowed" if mean_ratio < 0.18 else "eyes open")

    yaw = abs(_f(face_d.get("yaw")))
    pitch = abs(_f(face_d.get("pitch")))
    if yaw < 12.0 and pitch < 15.0:
        parts.append("looking at the camera")
    return ", ".join(parts)


def _pose_text(pose_d: dict, face_d: dict, width: int, height: int) -> str:
    parts: list[str] = []
    yaw = abs(_f(face_d.get("yaw")))
    if _d(face_d).get("ok"):
        parts.append("facing the camera" if yaw < 15.0 else
                     "three quarter view" if yaw < 50.0 else "profile view")
    if not _d(pose_d).get("ok"):
        return ", ".join(parts)

    torso = _torso_px(pose_d, width, height)
    hips = _mid(_lm(pose_d, "left_hip", width, height),
                _lm(pose_d, "right_hip", width, height))
    knees = _mid(_lm(pose_d, "left_knee", width, height),
                 _lm(pose_d, "right_knee", width, height))
    ankles = _mid(_lm(pose_d, "left_ankle", width, height),
                  _lm(pose_d, "right_ankle", width, height))
    if hips and knees and torso > 1.0:
        drop = (knees[1] - hips[1]) / torso
        if ankles and drop > 0.55:
            parts.append("standing upright")
        elif drop < 0.35:
            parts.append("seated or crouched")

    shoulders = _mid(_lm(pose_d, "left_shoulder", width, height),
                     _lm(pose_d, "right_shoulder", width, height))
    wrists = [p for p in (_lm(pose_d, "left_wrist", width, height),
                          _lm(pose_d, "right_wrist", width, height)) if p]
    if wrists and shoulders and torso > 1.0:
        above = sum(1 for w in wrists if w[1] < shoulders[1])
        if above == len(wrists):
            parts.append("arms raised")
        elif hips and all(w[1] > hips[1] - 0.15 * torso for w in wrists):
            parts.append("arms down at the sides")
        else:
            parts.append("hands in front of the body")
    return ", ".join(parts)


def _setting_text(img: np.ndarray, person: np.ndarray | None) -> tuple[str, dict]:
    height, width = img.shape[:2]
    background = ~person if person is not None else None
    if background is not None and int(background.sum()) < _MIN_MASK_PX:
        background = None

    names = _dominant_names(img, background, k=3)
    small = cv2.resize(img, (min(width, 320), min(height, 320)),
                       interpolation=cv2.INTER_AREA)
    grey = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
    texture = float(cv2.Laplacian(grey, cv2.CV_32F).var())

    hsv = cv2.cvtColor(small, cv2.COLOR_BGR2HSV)
    hue = hsv[:, :, 0].astype(np.float32) * 2.0
    sat = hsv[:, :, 1].astype(np.float32) / 255.0
    val = hsv[:, :, 2].astype(np.float32) / 255.0
    green = float(np.mean((hue >= 70) & (hue < 170) & (sat > 0.25)))
    band = max(1, small.shape[0] // 3)          # only the top of the frame
    sky = float(np.mean((hue[:band] >= 185) & (hue[:band] < 260)
                        & (val[:band] > 0.55)))

    colour = names[0] if names else ""
    if green > 0.15 or sky > 0.25:
        text = "outdoors, natural surroundings"
        if sky > 0.25:
            text += " with visible sky"
    elif texture < 45.0:
        text = f"plain {colour} background".strip()
    else:
        text = f"indoor scene, {colour} tones".strip(", ")
    return text, {"texture": round(texture, 2), "green": round(green, 3),
                  "sky": round(sky, 3), "colours": names}


def _camera_text(img: np.ndarray, shot_d: dict, face_d: dict) -> str:
    height, width = img.shape[:2]
    orientation = str(_d(shot_d).get("orientation") or
                      ("portrait" if height > width else "landscape"))
    shot = str(_d(shot_d).get("shot_type") or "unknown")
    framing = _d(_d(shot_d).get("framing"))
    parts = [f"{orientation} framing"]
    if shot != "unknown":
        parts.append({"closeup": "head and shoulders close up",
                      "half": "waist up", "full": "full body"}[shot]
                     if shot in ("closeup", "half", "full") else shot)
    fill = _f(framing.get("subject_fill"), 0.0)
    if fill > 0.0:
        parts.append(f"subject fills about {int(round(fill * 100))} percent of the frame")
    pitch = abs(_f(face_d.get("pitch")))
    parts.append("eye level" if pitch <= 12.0 else "slightly angled camera")
    return ", ".join(parts)


# ----------------------------------------------------------------- provider

class HeuristicVision(VisionProvider):
    """VisionProvider implemented entirely from local computer vision."""

    name = "heuristic"

    def available(self) -> bool:
        return True

    def estimate_cost(self, n_images: int = 1) -> float:
        return 0.0

    # ------------------------------------------------------------ describe

    def describe_photo(self, image_path: str, context: dict | None = None) -> dict:
        out = {"shot_type": "unknown", "subject": "", "clothing": "", "hair": "",
               "expression": "", "pose": "", "setting": "", "lighting": "",
               "camera": "", "colors": [], "preserve": [], "notes": "",
               "measurements": {}, "cost_usd": 0.0}

        loader = _mod("..analysis.loader")
        img = _safe(getattr(loader, "load_image", None), str(image_path), WORK_SIDE)
        if not isinstance(img, np.ndarray) or img.size == 0:
            out["notes"] = "No se pudo abrir la imagen."
            return out
        height, width = img.shape[:2]

        pose_mod = _mod("..analysis.pose")
        face_mod = _mod("..analysis.face")
        shot_mod = _mod("..analysis.shot")
        quality_mod = _mod("..analysis.quality")
        segment_mod = _mod("..analysis.segment")
        skin_mod = _mod("..analysis.skin")
        body_mod = _mod("..analysis.body")

        pose_d = _d(_safe(getattr(pose_mod, "detect_pose", None), img))
        face_d = _d(_safe(getattr(face_mod, "detect_face", None), img))
        shot_d = _d(_safe(getattr(shot_mod, "classify_shot", None), img, pose_d, face_d))
        qual_d = _d(_safe(getattr(quality_mod, "assess_quality", None), img,
                          str(image_path)))

        person_raw = None
        regions: dict = {}
        garment = None
        if segment_mod is not None:
            seg = _d(_safe(getattr(segment_mod, "person_mask", None), img))
            if seg.get("ok"):
                person_raw = seg.get("mask")
            found = _safe(getattr(segment_mod, "region_masks", None), img, pose_d,
                          person_raw if isinstance(person_raw, np.ndarray) else None)
            regions = found if isinstance(found, dict) else {}
            garment = _safe(getattr(segment_mod, "garment_mask", None), img, pose_d,
                            person_raw if isinstance(person_raw, np.ndarray) else None)
        person = _bool_mask(person_raw, (height, width))

        skin_d = _d(_safe(getattr(skin_mod, "skin_stats", None), img, pose_d, face_d))
        body_d = _d(_safe(getattr(body_mod, "measure_body", None), img, pose_d,
                          person_raw if isinstance(person_raw, np.ndarray) else None))

        shot = str(shot_d.get("shot_type") or "unknown")
        out["shot_type"] = shot if shot in ("closeup", "half", "full") else "unknown"

        # Every composer below is wrapped: one unlucky mask must not cost the
        # caller the whole reading.
        view = _safe(_pose_text, pose_d, face_d, width, height) or ""
        subject_bits = ["one person"]
        if out["shot_type"] != "unknown":
            subject_bits.append({"closeup": "framed head and shoulders",
                                 "half": "framed from the waist up",
                                 "full": "full body in frame"}[out["shot_type"]])
        if view:
            subject_bits.append(view.split(",")[0])
        out["subject"] = ", ".join(subject_bits)
        out["pose"] = view
        out["expression"] = _safe(_expression_text, face_d, width, height) or ""

        hair_text, hair_detail = _safe(_hair_text, img, regions, face_d,
                                       pose_d) or ("", {})
        out["hair"] = hair_text
        cloth_text, cloth_detail = _safe(_clothing_text, img, garment, regions,
                                         pose_d) or ("", {})
        out["clothing"] = cloth_text
        setting_text, setting_detail = _safe(_setting_text, img, person) or ("", {})
        out["setting"] = setting_text
        light = _d(_safe(_lighting, img, person, _face_box(face_d)))
        out["lighting"] = light.get("text") or ""
        out["camera"] = _safe(_camera_text, img, shot_d, face_d) or ""
        out["colors"] = _safe(_dominant_names, img, None, 4) or []

        out["preserve"] = _safe(self._preserve, face_d, body_d, skin_d,
                                hair_detail, cloth_detail, regions,
                                (height, width)) or []
        out["measurements"] = {
            "hair": hair_detail, "clothing": cloth_detail,
            "lighting": {k: light[k] for k in
                         ("direction", "hardness", "key", "warmth", "spread",
                          "level") if k in light},
            "setting": setting_detail,
            "quality": {"score": round(_f(qual_d.get("score")), 3),
                        "beauty_filter_suspected":
                            bool(qual_d.get("beauty_filter_suspected"))},
            "body_metrics": {str(k): round(_f(v), 4) for k, v in
                             _d(body_d.get("metrics")).items()},
            "skin": {"ita_deg": round(_f(skin_d.get("ita_deg")), 2),
                     "lab_mean": [round(_f(v), 2) for v in
                                  (skin_d.get("lab_mean")
                                   if isinstance(skin_d.get("lab_mean"), (list, tuple))
                                   else [])]},
        }

        notes = []
        if not face_d.get("ok"):
            notes.append("no se detecto rostro")
        if not pose_d.get("ok"):
            notes.append("no se detecto cuerpo")
        if person is None:
            notes.append("sin segmentacion de persona")
        if qual_d.get("beauty_filter_suspected"):
            notes.append("la foto original parece llevar filtro de belleza")
        out["notes"] = ("Lectura local: " + ", ".join(notes) + "."
                        if notes else "Lectura local completa.")
        return out

    def _preserve(self, face_d: dict, body_d: dict, skin_d: dict,
                  hair: dict, clothing: dict, regions: dict,
                  shape: tuple[int, int]) -> list[str]:
        """What a regenerated version must not change, from what was measured."""
        items: list[str] = []
        if face_d.get("ok"):
            items.append("exact face shape, features and bone structure")
            if _d(face_d).get("mesh"):
                items.append("eye shape, eyebrow shape and lip shape")
        metrics = _d(body_d.get("metrics"))
        gated = [n for n in ("shoulder_w_over_torso", "waist_w_over_torso",
                             "hip_w_over_torso", "bust_w_over_torso")
                 if n in metrics]
        if gated:
            items.append("body proportions exactly as photographed: "
                         "shoulder width, waist width, hips and bust")
        if skin_d.get("ok"):
            ita = _f(skin_d.get("ita_deg"), None)
            items.append("skin tone unchanged"
                         + (f" (ITA {ita:.0f} degrees)" if ita is not None else ""))
        items.append("skin texture, pores, freckles, scars and moles")
        items.append("every visible tattoo and birthmark")
        if hair.get("colour"):
            length = hair.get("length") or ""
            items.append(" ".join(p for p in (length, hair["colour"],
                                              "hair, same length and colour") if p))
        if clothing.get("upper_colours"):
            items.append(f"garment colour ({clothing['upper_colours'][0]}) and cut")
        if _bool_mask(regions.get("hands"), shape) is not None:
            items.append("both hands with five fingers each")
        return items

    # ------------------------------------------------------------ critique

    def critique_result(self, image_path: str, brief: dict,
                        reference_path: str | None = None) -> dict:
        out = {"ok": False, "score": 0.0, "defects": [], "identity_notes": "",
               "cost_usd": 0.0}

        profile = None
        if isinstance(brief, dict):
            for key in ("profile", "identity_profile"):
                candidate = brief.get(key)
                if isinstance(candidate, dict) and candidate:
                    profile = candidate
                    break

        # With a profile the numeric verdict is the answer: it already runs the
        # identity, proportion, skin and anatomy checks the client cares about.
        verify_mod = _mod("..identity.verify")
        if profile is not None and verify_mod is not None:
            verdict = _d(_safe(getattr(verify_mod, "verify_image", None),
                               str(image_path), profile,
                               brief if isinstance(brief, dict) else None))
            if verdict.get("checks") is not None:
                defects = [self._clean_defect(d) for d in (verdict.get("defects") or [])]
                out["ok"] = bool(verdict.get("passed"))
                out["score"] = round(_clamp01(_f(verdict.get("score"))), 4)
                out["defects"] = [d for d in defects if d]
                out["identity_notes"] = str(verdict.get("summary") or "")
                out["checks"] = verdict.get("checks")
                return out

        loader = _mod("..analysis.loader")
        img = _safe(getattr(loader, "load_image", None), str(image_path), WORK_SIDE)
        if not isinstance(img, np.ndarray) or img.size == 0:
            out["identity_notes"] = "No se pudo abrir la imagen generada."
            return out
        height, width = img.shape[:2]

        pose_mod = _mod("..analysis.pose")
        face_mod = _mod("..analysis.face")
        quality_mod = _mod("..analysis.quality")
        segment_mod = _mod("..analysis.segment")
        anomaly_mod = _mod("..analysis.anomaly")
        skin_mod = _mod("..analysis.skin")

        pose_d = _d(_safe(getattr(pose_mod, "detect_pose", None), img))
        face_d = _d(_safe(getattr(face_mod, "detect_face", None), img))
        qual_d = _d(_safe(getattr(quality_mod, "assess_quality", None), img,
                          str(image_path)))
        regions: dict = {}
        person_raw = None
        if segment_mod is not None:
            seg = _d(_safe(getattr(segment_mod, "person_mask", None), img))
            if seg.get("ok"):
                person_raw = seg.get("mask")
            found = _safe(getattr(segment_mod, "region_masks", None), img, pose_d,
                          person_raw if isinstance(person_raw, np.ndarray) else None)
            regions = found if isinstance(found, dict) else {}

        scan = _d(_safe(getattr(anomaly_mod, "scan_anomalies", None), img, pose_d,
                        face_d, regions))
        defects = [self._clean_defect(d) for d in (scan.get("defects") or [])]
        defects = [d for d in defects if d]

        similarity = None
        skin_delta = None
        if reference_path:
            compared = _safe(self._compare_reference, img, face_d, pose_d,
                             str(reference_path), face_mod, loader, pose_mod,
                             skin_mod)
            if compared is not None:
                similarity, skin_delta, extra = compared
                defects.extend(extra)

        worst = max((d["severity"] for d in defects), default=0.0)
        penalty = min(0.85, sum(d["severity"] for d in defects) * 0.35)
        quality_score = _clamp01(_f(qual_d.get("score"), 0.5)) if qual_d.get("ok") else 0.5
        score = 0.35 * quality_score + 0.65 * (1.0 - penalty)
        if similarity is not None:
            score = 0.5 * score + 0.5 * similarity

        out["score"] = round(_clamp01(score), 4)
        out["defects"] = defects
        out["ok"] = bool(out["score"] >= 0.62 and worst < 0.55
                         and (similarity is None or similarity >= 0.72))

        notes = []
        if similarity is not None:
            notes.append(f"parecido facial {similarity:.2f} sobre el original")
        if skin_delta is not None:
            notes.append(f"diferencia de tono de piel dE {skin_delta:.1f}")
        if profile is None:
            notes.append("sin perfil de identidad: solo anatomia y calidad")
        if defects:
            notes.append(f"{len(defects)} defecto(s) detectado(s)")
        else:
            notes.append("sin defectos detectados")
        out["identity_notes"] = "Revision local: " + ", ".join(notes) + "."
        return out

    def _compare_reference(self, img: np.ndarray, face_d: dict, pose_d: dict,
                           reference_path: str, face_mod, loader, pose_mod,
                           skin_mod) -> tuple:
        """Face and skin comparison against the untouched original."""
        defects: list[dict] = []
        ref = _safe(getattr(loader, "load_image", None), reference_path, WORK_SIDE)
        if not isinstance(ref, np.ndarray) or ref.size == 0:
            return None, None, defects

        ref_face = _d(_safe(getattr(face_mod, "detect_face", None), ref))
        similarity = None
        desc_a = ref_face.get("descriptor")
        desc_b = face_d.get("descriptor")
        if not desc_a:
            desc_a = _safe(getattr(face_mod, "face_descriptor", None), ref, ref_face)
        if not desc_b:
            desc_b = _safe(getattr(face_mod, "face_descriptor", None), img, face_d)
        if desc_a is not None and desc_b is not None:
            value = _safe(getattr(face_mod, "compare_faces", None), desc_a, desc_b)
            if value is not None:
                similarity = _clamp01(_f(value))

        if similarity is not None and similarity < 0.72:
            box = _face_box(face_d)
            defects.append({
                "type": "face_distorted", "where": "face",
                "bbox": [int(round(v)) for v in box] if box else [],
                "severity": round(_clamp01((0.72 - similarity) + 0.35), 3),
                "repairable": bool(box),
                "detail": f"El rostro no coincide con el original "
                          f"(parecido {similarity:.2f}, minimo 0.72).",
            })

        skin_delta = None
        if skin_mod is not None:
            ref_pose = _d(_safe(getattr(pose_mod, "detect_pose", None), ref))
            a = _d(_safe(getattr(skin_mod, "skin_stats", None), ref, ref_pose, ref_face))
            b = _d(_safe(getattr(skin_mod, "skin_stats", None), img, pose_d, face_d))
            if a.get("ok") and b.get("ok"):
                cmp = _d(_safe(getattr(skin_mod, "compare_skin", None), a, b))
                if cmp:
                    skin_delta = _f(cmp.get("delta_e"), None)
                    if skin_delta is not None and skin_delta > 8.0:
                        defects.append({
                            "type": "skin_tone_changed", "where": "global",
                            "bbox": [], "severity": round(min(1.0, skin_delta / 20.0), 3),
                            "repairable": False,
                            "detail": f"El tono de piel cambio (dE {skin_delta:.1f}, "
                                      f"maximo 8.0).",
                        })
        return similarity, skin_delta, defects

    def _clean_defect(self, item: Any) -> dict | None:
        if not isinstance(item, dict):
            return None
        kind = str(item.get("type") or "").strip()
        if not kind:
            return None
        bbox: list[int] = []
        raw = item.get("bbox")
        if isinstance(raw, (list, tuple)) and len(raw) == 4:
            vals = [int(round(_f(v))) for v in raw]
            if vals[2] > 1 and vals[3] > 1:
                bbox = vals
        return {"type": kind,
                "where": str(item.get("where") or "global"),
                "bbox": bbox,
                "severity": round(_clamp01(_f(item.get("severity"), 0.5)), 3),
                "repairable": bool(item.get("repairable")) and bool(bbox),
                "detail": str(item.get("detail") or "")}
