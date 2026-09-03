"""Finding what the generator got wrong, and where exactly it is.

The economics of the product live in this file.  If a defect is found and
localised, the repair step repaints a few hundred pixels; if it is missed, the
whole image is regenerated and the client pays again for something she already
paid for.  So every detector here returns a pixel box wherever the fault is
local, and only genuinely global faults report an empty box.

Every detector is defensive on purpose: a scan that cannot run contributes no
defects instead of taking the whole analysis down with it.  A false positive
costs an unnecessary inpaint, so thresholds are set where a human would agree
the image is broken, not where a metric first twitches.
"""
from __future__ import annotations

import math

import cv2
import numpy as np

from .body import landmarks_px, torso_frame
from .skin import named_point_px, skin_mask_ycrcb

try:  # optional heavy dependency; every detector degrades without it
    import mediapipe as mp
except Exception:                                     # noqa: BLE001
    mp = None

# How much each defect type removes from a perfect score.
SEVERITY_WEIGHT = {
    "extra_person": 1.00,
    "extra_limb": 0.90,
    "face_distorted": 0.90,
    "hand_malformed": 0.80,
    "texture_smear": 0.60,
    "duplicated_feature": 0.60,
    "eye_asymmetry": 0.50,
    "oversmoothed_skin": 0.50,
    "missing_limb": 0.50,
    "border_artifact": 0.40,
}

# Below this size a hand cannot support digit-ratio geometry; the flags it
# produces are landmark noise rather than deformity.  See the hand detector.
HAND_MIN_PX = 170.0
HAND_MIN_TORSO_FRAC = 0.26

# Oversmoothed skin: what the measured face/body texture ratio means.
# A face carrying 45% of the fine texture of the same person's other skin is
# where the difference first becomes visible; at 20% the grain is essentially
# gone.  Severity is the position between those two, and it is only a
# measurement: this file no longer decides what happens next, because the
# consequence depends on who made the image - every phone camera smooths a
# face, so the same number means "her camera" on a photograph she took and
# "the robot retouched her" on a generated one.  identity/verify.py knows
# which, and gates accordingly.
SMOOTH_RATIO_TRIGGER = 0.45
SMOOTH_RATIO_SEVERE = 0.20

# Fine-band measurement of facial skin, used by the caller to compare a result
# with the photographs the person actually took.  The face is scaled to a
# common width first so the number means the same thing on a closeup and on a
# full-length shot, and the band is the one a beauty filter destroys first:
# pores, fine lines, camera grain.
FACE_TEXTURE_REF_PX = 520.0
_FINE_SIGMA = 1.4
_MID_SIGMA = 5.0

# MediaPipe Hands topology.
_DIGITS = {
    "thumb": (1, 2, 3, 4),
    "index": (5, 6, 7, 8),
    "middle": (9, 10, 11, 12),
    "ring": (13, 14, 15, 16),
    "pinky": (17, 18, 19, 20),
}
# Digit length over palm length, with bounds wide enough that foreshortening
# alone never trips them - only real deformity does.
_DIGIT_BOUNDS = {
    "thumb": (0.30, 1.50),
    "index": (0.30, 1.35),
    "middle": (0.32, 1.45),
    "ring": (0.30, 1.35),
    "pinky": (0.20, 1.10),
}

# Face mesh: mirror pairs and the midline.
_MESH_PAIRS = ((33, 263), (133, 362), (61, 291), (70, 300), (234, 454),
               (93, 323), (132, 361), (172, 397), (58, 288), (215, 435))
_MESH_MIDLINE = (10, 168, 6, 4, 1, 0, 17, 152)
_EYE_A = (33, 160, 158, 133, 153, 144)
_EYE_B = (362, 385, 387, 263, 373, 380)


# ------------------------------------------------------------------ helpers

def _clamp(v: float, lo: float, hi: float) -> float:
    return lo if v < lo else (hi if v > hi else v)


def _defect(kind: str, where: str, bbox, severity: float,
            repairable: bool, detail: str) -> dict:
    return {"type": kind, "where": where,
            "bbox": [int(round(t)) for t in bbox] if bbox else [],
            "severity": round(_clamp(float(severity), 0.0, 1.0), 3),
            "repairable": bool(repairable), "detail": detail}


def _box_from_points(pts, pad: float, w: int, h: int):
    xs = [p[0] for p in pts]
    ys = [p[1] for p in pts]
    x0 = _clamp(min(xs) - pad, 0, w - 1)
    y0 = _clamp(min(ys) - pad, 0, h - 1)
    x1 = _clamp(max(xs) + pad, 0, w - 1)
    y1 = _clamp(max(ys) + pad, 0, h - 1)
    if x1 - x0 < 2 or y1 - y0 < 2:
        return []
    return [x0, y0, x1 - x0, y1 - y0]


def _poly_area(pts) -> float:
    total = 0.0
    n = len(pts)
    for i in range(n):
        x1, y1 = pts[i]
        x2, y2 = pts[(i + 1) % n]
        total += x1 * y2 - x2 * y1
    return abs(total) / 2.0


def _ccw(a, b, c) -> bool:
    return (c[1] - a[1]) * (b[0] - a[0]) > (b[1] - a[1]) * (c[0] - a[0])


def _segments_cross(a, b, c, d) -> bool:
    return (_ccw(a, c, d) != _ccw(b, c, d)) and (_ccw(a, b, c) != _ccw(a, b, d))


def _dist(a, b) -> float:
    return math.hypot(a[0] - b[0], a[1] - b[1])


def _person_of(masks: dict, img_bgr, pose, h: int, w: int):
    """Person silhouette from the region masks, or computed if absent."""
    if isinstance(masks, dict) and masks:
        got = masks.get("person")
        if isinstance(got, np.ndarray) and got.size:
            return _resize_mask(got, h, w)
        bg = masks.get("background")
        if isinstance(bg, np.ndarray) and bg.size:
            return cv2.bitwise_not(_resize_mask(bg, h, w))
        union = None
        for name in ("upper_body", "lower_body", "arms", "legs", "face", "hair", "hands"):
            part = masks.get(name)
            if isinstance(part, np.ndarray) and part.size:
                part = _resize_mask(part, h, w)
                union = part if union is None else cv2.bitwise_or(union, part)
        if union is not None and np.count_nonzero(union) > 64:
            return union
    try:
        from . import segment
        got = segment.person_mask(img_bgr, pose)
        if got.get("ok") and isinstance(got.get("mask"), np.ndarray):
            return _resize_mask(got["mask"], h, w)
    except Exception:                                 # noqa: BLE001
        pass
    return None


def _resize_mask(mask, h: int, w: int):
    m = mask[:, :, 0] if mask.ndim == 3 else mask
    if m.dtype != np.uint8:
        m = (m > (0.5 if m.dtype.kind == "f" else 0)).astype(np.uint8) * 255
    elif int(m.max()) == 1:
        m = m * 255
    if m.shape[:2] != (h, w):
        m = cv2.resize(m, (w, h), interpolation=cv2.INTER_NEAREST)
    return m


def _energy(gray):
    """Fine (pore/hair scale) and coarse (feature scale) local energy."""
    b1 = cv2.GaussianBlur(gray, (0, 0), 1.5)
    b2 = cv2.GaussianBlur(gray, (0, 0), 8.0)
    fine = cv2.boxFilter((gray - b1) ** 2, -1, (11, 11))
    coarse = cv2.boxFilter((b1 - b2) ** 2, -1, (21, 21))
    return fine, coarse


def _median_in(values, mask):
    sel = values[mask > 0]
    if sel.size < 64:
        return None
    return float(np.median(sel))


# ------------------------------------------------------------------- hands

def _hand_geometry(pts) -> dict:
    """Everything measurable about one detected hand, in pixels."""
    wrist = pts[0]
    palm_len = _dist(wrist, pts[9])
    palm_w = _dist(pts[5], pts[17])
    out = {"palm_len": palm_len, "palm_w": palm_w, "flags": [], "extended": 0}
    if palm_len < 6.0 or palm_w < 4.0:
        out["flags"].append("degenerate")
        return out

    for name, chain in _DIGITS.items():
        length = sum(_dist(pts[a], pts[b]) for a, b in zip(chain, chain[1:]))
        ratio = length / palm_len
        lo, hi = _DIGIT_BOUNDS[name]
        if ratio < lo or ratio > hi:
            out["flags"].append("digit_%s_%.2f" % (name, ratio))

    # A distal phalanx longer than the proximal one is not a human hand.
    broken = 0
    for name in ("index", "middle", "ring", "pinky"):
        a, b, c, d = _DIGITS[name]
        s1, s2, s3 = _dist(pts[a], pts[b]), _dist(pts[b], pts[c]), _dist(pts[c], pts[d])
        if s1 > 1e-6 and (s3 > 1.5 * s1 or s2 > 1.4 * s1):
            broken += 1
    if broken >= 2:
        out["flags"].append("phalanx_order")

    # Extended-finger count.  A closed fist legitimately counts zero, so the
    # count is only evidence when the hand is open in the first place.
    reach = []
    for name in ("index", "middle", "ring", "pinky"):
        mcp, pip, _dip, tip = _DIGITS[name]
        reach.append(_dist(pts[tip], wrist) / palm_len)
        if _dist(pts[tip], wrist) > 1.12 * _dist(pts[pip], wrist):
            out["extended"] += 1
    if _dist(pts[4], pts[17]) > 1.10 * _dist(pts[2], pts[17]):
        out["extended"] += 1
    out["openness"] = sum(reach) / max(len(reach), 1)
    if out["openness"] > 1.45 and out["extended"] != 5:
        out["flags"].append("finger_count_%d" % out["extended"])

    # Crossing finger axes only mean something on an open hand: curled fingers
    # legitimately fold over each other.
    if out["openness"] > 1.30:
        axes = [(pts[_DIGITS[n][0]], pts[_DIGITS[n][3]])
                for n in ("index", "middle", "ring", "pinky")]
        for i in range(len(axes)):
            for j in range(i + 1, len(axes)):
                if _segments_cross(axes[i][0], axes[i][1], axes[j][0], axes[j][1]):
                    out["flags"].append("axes_cross")
                    break
            if "axes_cross" in out["flags"]:
                break

    area = _poly_area([pts[i] for i in (0, 5, 9, 13, 17)])
    fill = area / max(palm_len * palm_w, 1e-6)
    aspect = palm_len / max(palm_w, 1e-6)
    if fill < 0.20 or fill > 1.05:
        out["flags"].append("palm_area_%.2f" % fill)
    if aspect < 0.45 or aspect > 2.20:
        out["flags"].append("palm_aspect_%.2f" % aspect)
    return out


def _read_hands(detector, img_bgr, ox: float, oy: float) -> list[dict]:
    h, w = img_bgr.shape[:2]
    rgb = np.ascontiguousarray(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB))
    res = detector.process(rgb)
    out = []
    lms = getattr(res, "multi_hand_landmarks", None) or []
    handed = getattr(res, "multi_handedness", None) or []
    for i, hand in enumerate(lms):
        pts = [(ox + p.x * w, oy + p.y * h) for p in hand.landmark]
        if len(pts) < 21:
            continue
        label, score = "", 0.0
        if i < len(handed):
            try:
                cls = handed[i].classification[0]
                label, score = str(cls.label).lower(), float(cls.score)
            except Exception:                         # noqa: BLE001
                pass
        cx = sum(p[0] for p in pts) / len(pts)
        cy = sum(p[1] for p in pts) / len(pts)
        out.append({"pts": pts, "label": label, "score": score, "centre": (cx, cy)})
    return out


def _hand_defects(img_bgr, lm: dict, frame, h: int, w: int) -> list[dict]:
    if mp is None:
        return []
    torso = frame["torso"] if frame else 0.12 * max(h, w)
    wrists = {}
    for side in ("left", "right"):
        p = lm.get(side + "_wrist")
        if p is not None and p[3] >= 0.5:
            el = lm.get(side + "_elbow")
            forearm = _dist((p[0], p[1]), (el[0], el[1])) if el and el[3] >= 0.4 else 0.0
            wrists[side] = {"pt": (p[0], p[1]),
                            "forearm": forearm if forearm > 8.0 else 0.35 * torso}

    defects: list[dict] = []
    try:
        with mp.solutions.hands.Hands(static_image_mode=True, max_num_hands=4,
                                      min_detection_confidence=0.4) as detector:
            hands = _read_hands(detector, img_bgr, 0.0, 0.0)

            # Small hands in a wide frame are missed by a full-frame pass; look
            # again around each wrist before calling a hand missing.
            for side, info in wrists.items():
                near = min((_dist(info["pt"], hd["pts"][0]) for hd in hands), default=1e9)
                if near <= max(0.6 * info["forearm"], 0.12 * torso):
                    continue
                side_px = max(48.0, 2.4 * info["forearm"])
                x0 = int(_clamp(info["pt"][0] - side_px / 2, 0, w - 2))
                y0 = int(_clamp(info["pt"][1] - side_px / 2, 0, h - 2))
                x1 = int(_clamp(x0 + side_px, x0 + 2, w))
                y1 = int(_clamp(y0 + side_px, y0 + 2, h))
                crop = img_bgr[y0:y1, x0:x1]
                if crop.shape[0] < 24 or crop.shape[1] < 24:
                    continue
                for found in _read_hands(detector, np.ascontiguousarray(crop),
                                         float(x0), float(y0)):
                    if all(_dist(found["centre"], hd["centre"]) > 0.25 * torso
                           for hd in hands):
                        hands.append(found)
    except Exception:                                 # noqa: BLE001
        return []

    # A hand running off the edge of the frame is the commonest thing in a phone
    # selfie - the hand holding the phone - and it is unjudgeable, not deformed:
    # the fingers that would make the geometry add up are simply not in the
    # picture.  Measured on this client's own 24 untouched photographs, treating
    # those as defects produced six false "mano deformada" verdicts at severity
    # up to 0.95, which would have thrown away good images and paid to remake
    # them.  Truncated hands are skipped, and the caller is told why.
    edge_x, edge_y = 0.02 * w, 0.02 * h
    for hand in hands:
        pts = hand["pts"]
        box = _box_from_points(pts, 0.12 * max(_dist(pts[0], pts[9]), 8.0), w, h)
        if not box or min(box[2], box[3]) < 12:
            continue
        xs = [p[0] for p in pts]
        ys = [p[1] for p in pts]
        truncated = (min(xs) <= edge_x or max(xs) >= w - edge_x
                     or min(ys) <= edge_y or max(ys) >= h - edge_y)
        if truncated:
            continue
        geo = _hand_geometry(pts)
        flags = geo["flags"]
        if not flags:
            continue
        strong = any(f.startswith(("digit_", "phalanx", "axes_cross")) for f in flags)
        if not strong and len(flags) < 2:
            continue

        # Confidence in hand geometry scales with how many pixels the hand
        # occupies.  A digit-length ratio is a quotient of two short distances,
        # so on a hand 70 px across a two-pixel landmark error is a 30% error in
        # the ratio, and every distant hand looks deformed.  On this client's
        # untouched photographs that produced six confident "deformed hand"
        # verdicts at severity up to 0.95, all of them wrong.  Small hands are
        # still reported - they stay below the severity that rejects an image,
        # so the information survives without costing a regeneration.
        hand_px = float(min(box[2], box[3]))
        floor_px = max(HAND_MIN_PX, HAND_MIN_TORSO_FRAC * torso) if torso > 1 else HAND_MIN_PX
        size_conf = _clamp(hand_px / max(floor_px, 1.0), 0.0, 1.0)

        where = "left_hand" if hand["label"].startswith("left") else (
            "right_hand" if hand["label"].startswith("right") else "hand")
        severity = _clamp(0.45 + 0.13 * len(flags), 0.45, 0.92)
        if hand["score"] >= 0.9 and strong:
            severity = min(0.95, severity + 0.05)
        severity = _clamp(severity * size_conf, 0.15, 0.95)
        defects.append(_defect(
            "hand_malformed", where, box, severity, True,
            "geometria de mano fuera de rango: " + ", ".join(flags[:4])))

    # More hands than wrists means a limb was invented.
    if wrists and len(hands) > len(wrists):
        extra = max(hands, key=lambda hd: min(_dist(hd["pts"][0], i["pt"])
                                              for i in wrists.values()))
        box = _box_from_points(extra["pts"], 0.15 * torso, w, h)
        defects.append(_defect(
            "extra_limb", "hands", box, 0.75, True,
            "se detectaron %d manos para %d munecas" % (len(hands), len(wrists))))

    # A wrist well inside the frame with nothing on the end of it.
    margin_x, margin_y = 0.06 * w, 0.06 * h
    for side, info in wrists.items():
        pt = info["pt"]
        if not (margin_x < pt[0] < w - margin_x and margin_y < pt[1] < h - margin_y):
            continue
        if info["forearm"] < 0.03 * math.hypot(w, h):
            continue
        near = min((_dist(pt, hd["pts"][0]) for hd in hands), default=1e9)
        if near <= max(0.6 * info["forearm"], 0.12 * torso):
            continue
        radius = 0.7 * info["forearm"]
        bx = _clamp(pt[0] - radius, 0, w - 2)
        by = _clamp(pt[1] - radius, 0, h - 2)
        box = [bx, by, min(2 * radius, w - bx), min(2 * radius, h - by)]
        defects.append(_defect(
            "missing_limb", side + "_hand", box, 0.35, True,
            "muneca " + side + " visible sin mano reconocible"))
    return defects


# ------------------------------------------------------------ extra person

def _extra_person(img_bgr, face: dict, h: int, w: int) -> list[dict]:
    if mp is None:
        return []
    try:
        rgb = np.ascontiguousarray(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB))
        with mp.solutions.face_detection.FaceDetection(
                model_selection=1, min_detection_confidence=0.5) as det:
            res = det.process(rgb)
    except Exception:                                 # noqa: BLE001
        return []

    boxes = []
    for found in (getattr(res, "detections", None) or []):
        try:
            rel = found.location_data.relative_bounding_box
            box = [rel.xmin * w, rel.ymin * h, rel.width * w, rel.height * h]
        except Exception:                             # noqa: BLE001
            continue
        if box[2] <= 2 or box[3] <= 2:
            continue
        if (box[2] * box[3]) / float(w * h) < 0.03:
            continue
        if any(_iou(box, kept) > 0.5 for kept in boxes):
            continue
        boxes.append(box)
    if len(boxes) < 2:
        return []

    main = None
    ref = face.get("bbox") if isinstance(face, dict) else None
    if isinstance(ref, (list, tuple)) and len(ref) == 4:
        try:
            main = max(boxes, key=lambda b: _iou(b, [float(t) for t in ref]))
        except (TypeError, ValueError):
            main = None
    if main is None:
        main = max(boxes, key=lambda b: b[2] * b[3])
    others = [b for b in boxes if b is not main]
    worst = max(others, key=lambda b: b[2] * b[3])
    return [_defect("extra_person", "background", worst, 0.85, False,
                    "hay %d caras en la imagen; solo debe haber una" % len(boxes))]


def _iou(a, b) -> float:
    ax0, ay0, ax1, ay1 = a[0], a[1], a[0] + a[2], a[1] + a[3]
    bx0, by0, bx1, by1 = b[0], b[1], b[0] + b[2], b[1] + b[3]
    ix = max(0.0, min(ax1, bx1) - max(ax0, bx0))
    iy = max(0.0, min(ay1, by1) - max(ay0, by0))
    inter = ix * iy
    union = a[2] * a[3] + b[2] * b[3] - inter
    return inter / union if union > 1e-6 else 0.0


# -------------------------------------------------------------------- face

def _mesh_points(face: dict, img_bgr, h: int, w: int):
    """468-point mesh in pixels: reuse what face.py found, else measure again."""
    mesh = face.get("mesh") if isinstance(face, dict) else None
    try:
        if mesh is not None and len(mesh) >= 468:
            arr = np.asarray(mesh, dtype=np.float64)
            if arr.ndim == 2 and arr.shape[1] >= 2:
                scale = (float(w), float(h)) if float(np.abs(arr[:, :2]).max()) <= 1.6 \
                    else (1.0, 1.0)
                return {i: (float(arr[i, 0]) * scale[0], float(arr[i, 1]) * scale[1])
                        for i in range(arr.shape[0])}
    except (TypeError, ValueError):
        pass

    lms = face.get("landmarks") if isinstance(face, dict) else None
    if isinstance(lms, dict) and len(lms) >= 100:
        pts = {}
        span = 0.0
        for key, val in lms.items():
            if not isinstance(val, dict):
                continue
            idx = _index_of(key)
            if idx is None:
                continue
            try:
                x, y = float(val.get("x", 0.0)), float(val.get("y", 0.0))
            except (TypeError, ValueError):
                continue
            pts[idx] = (x, y)
            span = max(span, abs(x), abs(y))
        if len(pts) >= 100:
            if span <= 1.6:
                pts = {k: (v[0] * w, v[1] * h) for k, v in pts.items()}
            return pts
    if mp is None:
        return None
    try:
        rgb = np.ascontiguousarray(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB))
        with mp.solutions.face_mesh.FaceMesh(static_image_mode=True, max_num_faces=1,
                                             refine_landmarks=False,
                                             min_detection_confidence=0.5) as mesh:
            res = mesh.process(rgb)
        found = getattr(res, "multi_face_landmarks", None) or []
        if not found:
            return None
        return {i: (p.x * w, p.y * h) for i, p in enumerate(found[0].landmark)}
    except Exception:                                 # noqa: BLE001
        return None


def _index_of(key):
    if isinstance(key, int):
        return key
    text = str(key)
    digits = "".join(ch for ch in text if ch.isdigit())
    if digits and digits == text.lstrip("plmesh_"):
        try:
            return int(digits)
        except ValueError:
            return None
    if text.isdigit():
        return int(text)
    return None


def _face_defects(img_bgr, face: dict, h: int, w: int) -> list[dict]:
    face = face if isinstance(face, dict) else {}
    pts = _mesh_points(face, img_bgr, h, w)
    if not pts or not all(i in pts for i in (33, 263, 133, 362)):
        return _face_defects_coarse(face, w, h)

    eye_a = [pts[i] for i in _EYE_A if i in pts]
    eye_b = [pts[i] for i in _EYE_B if i in pts]
    if len(eye_a) < 6 or len(eye_b) < 6:
        return _face_defects_coarse(face, w, h)
    ca = (sum(p[0] for p in eye_a) / 6.0, sum(p[1] for p in eye_a) / 6.0)
    cb = (sum(p[0] for p in eye_b) / 6.0, sum(p[1] for p in eye_b) / 6.0)
    iod = _dist(ca, cb)
    if iod < 12.0:
        return []

    ang = math.atan2(cb[1] - ca[1], cb[0] - ca[0])
    cos_a, sin_a = math.cos(-ang), math.sin(-ang)
    origin = ((ca[0] + cb[0]) / 2.0, (ca[1] + cb[1]) / 2.0)

    def rot(p):
        dx, dy = p[0] - origin[0], p[1] - origin[1]
        return (dx * cos_a - dy * sin_a, dx * sin_a + dy * cos_a)

    mid_pts = [rot(pts[i]) for i in _MESH_MIDLINE if i in pts]
    if len(mid_pts) < 3:
        return []
    midline = sum(p[0] for p in mid_pts) / len(mid_pts)

    # Yaw makes a healthy face asymmetric in projection; measure it from the
    # mesh itself and stay quiet when the head is genuinely turned.
    half_a = abs(midline - rot(pts[33])[0])
    half_b = abs(rot(pts[263])[0] - midline)
    balance = min(half_a, half_b) / max(half_a, half_b, 1e-6)
    yaw = abs(float(face.get("yaw") or 0.0))
    turned = balance < 0.78 or yaw > 18.0

    defects: list[dict] = []
    box = _box_from_points(list(pts.values()), 0.10 * iod, w, h)

    if not turned:
        offs, vmis = [], []
        for a, b in _MESH_PAIRS:
            if a not in pts or b not in pts:
                continue
            ra, rb = rot(pts[a]), rot(pts[b])
            offs.append(abs((ra[0] + rb[0]) / 2.0 - midline) / iod)
            vmis.append(abs(ra[1] - rb[1]) / iod)
        if len(offs) >= 5:
            asym = sum(offs) / len(offs)
            drift = sum(vmis) / len(vmis)
            if asym > 0.10 or drift > 0.055:
                sev = _clamp(max((asym - 0.10) / 0.12, (drift - 0.055) / 0.06),
                             0.25, 0.9)
                defects.append(_defect(
                    "face_distorted", "face", box, sev, True,
                    "rasgos asimetricos respecto a la linea media "
                    "(desvio %.3f, deriva %.3f)" % (asym, drift)))

        area_a = _poly_area([rot(p) for p in eye_a])
        area_b = _poly_area([rot(p) for p in eye_b])
        if min(area_a, area_b) > 4.0:
            ratio = min(area_a, area_b) / max(area_a, area_b)
            if ratio < 0.62:
                eye_box = _box_from_points(eye_a + eye_b, 0.25 * iod, w, h)
                defects.append(_defect(
                    "eye_asymmetry", "eyes", eye_box,
                    _clamp((0.62 - ratio) / 0.5, 0.2, 0.85), True,
                    "un ojo mide %.0f%% del otro" % (100.0 * ratio)))
    return defects


def _face_defects_coarse(face: dict, w: int, h: int) -> list[dict]:
    """Key-point fallback: only an extreme imbalance is reported."""
    box = face.get("bbox") if isinstance(face, dict) else None
    if abs(float(face.get("yaw") or 0.0)) > 15.0:
        return []
    pn = None
    for name in ("nose_tip", "nose"):
        pn = named_point_px(face, name, w, h)
        if pn is not None:
            break
    pl = named_point_px(face, "left_eye", w, h)
    pr = named_point_px(face, "right_eye", w, h)
    if pn is None or pl is None or pr is None:
        return []
    da, db = _dist(pn, pl), _dist(pn, pr)
    if min(da, db) < 1e-6:
        return []
    ratio = min(da, db) / max(da, db)
    if ratio >= 0.62:
        return []
    bbox = [float(t) for t in box] if isinstance(box, (list, tuple)) and len(box) == 4 else []
    return [_defect("face_distorted", "face", bbox,
                    _clamp((0.62 - ratio) / 0.5, 0.25, 0.8), True,
                    "nariz descentrada respecto a los ojos (%.0f%%)" % (100.0 * ratio))]


# ------------------------------------------------------------ texture work

def _texture_defects(small, person_s, masks_s: dict, scale: float) -> list[dict]:
    gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY).astype(np.float32)
    fine, coarse = _energy(gray)
    defects: list[dict] = []

    area_person = float(np.count_nonzero(person_s))
    if area_person < 400:
        return defects
    med_fine = _median_in(fine, person_s)
    med_coarse = _median_in(coarse, person_s)

    # "Melted" regions: no fine detail at all, yet clear structure at a coarse
    # scale.  Plain fabric has neither; a smear has only the second.
    if med_fine is not None and med_coarse is not None:
        flat = fine < max(0.22 * med_fine, 3.0)
        structured = coarse > max(1.5 * med_coarse, 20.0)
        cand = (flat & structured & (person_s > 0)).astype(np.uint8) * 255
        cand = cv2.morphologyEx(cand, cv2.MORPH_OPEN,
                                cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)))
        n, labels, stats, _ = cv2.connectedComponentsWithStats(cand, 8)
        blobs = []
        for i in range(1, n):
            area = float(stats[i, cv2.CC_STAT_AREA])
            if area < max(400.0, 0.015 * area_person):
                continue
            blobs.append((area, stats[i]))
        for area, st in sorted(blobs, key=lambda t: -t[0])[:2]:
            frac = area / area_person
            box = [st[cv2.CC_STAT_LEFT] / scale, st[cv2.CC_STAT_TOP] / scale,
                   st[cv2.CC_STAT_WIDTH] / scale, st[cv2.CC_STAT_HEIGHT] / scale]
            defects.append(_defect(
                "texture_smear", "body", box, _clamp(frac * 4.0, 0.2, 0.75), True,
                "zona sin micro textura que ocupa el %.0f%% de la figura"
                % (100.0 * frac)))

    # Beauty-filter signature: face skin far smoother than the rest of the skin.
    face_m = masks_s.get("face")
    skin = skin_mask_ycrcb(small)
    if face_m is not None and med_fine is not None:
        face_skin = cv2.bitwise_and(face_m, skin)
        body_skin = cv2.bitwise_and(cv2.bitwise_and(skin, person_s),
                                    cv2.bitwise_not(face_m))
        hair_m = masks_s.get("hair")
        if hair_m is not None:
            body_skin = cv2.bitwise_and(body_skin, cv2.bitwise_not(hair_m))
        face_e = _median_in(fine, face_skin)
        body_e = _median_in(fine, body_skin)
        if face_e is not None and body_e is not None and body_e > 2.0:
            ratio = face_e / body_e
            if ratio < SMOOTH_RATIO_TRIGGER:
                fb = cv2.boundingRect(face_m)
                box = [fb[0] / scale, fb[1] / scale, fb[2] / scale, fb[3] / scale]
                # Report what was measured, at its real size.  The old code
                # capped this at 0.55 so that it could never reject an image,
                # which made the worst case - a face with no grain left at all -
                # indistinguishable from a mild one and silently threw away the
                # only number the caller could have judged.  The cap is gone;
                # the reason it existed lives in identity/verify.py, which knows
                # whether a camera or a generator produced these pixels.
                span = max(SMOOTH_RATIO_TRIGGER - SMOOTH_RATIO_SEVERE, 1e-6)
                sev = 0.20 + 0.75 * (SMOOTH_RATIO_TRIGGER - ratio) / span
                defects.append(_defect(
                    "oversmoothed_skin", "face", box,
                    _clamp(sev, 0.2, 0.95), True,
                    "la piel del rostro conserva el %.0f%% de la textura del "
                    "resto de la piel (ratio %.2f)" % (100.0 * ratio, ratio)))
    return defects


# ------------------------------------------------- facial texture, absolute

def face_box_px(img_bgr, face: dict) -> list[float]:
    """[x, y, w, h] of the detected face in pixels, or [] when unusable.

    Kept apart because two callers need the same box and must not disagree
    about it: the texture measurement below, and identity/verify.py, which has
    to shrink one image until its face is as wide as another's before the two
    bands can be compared at all.
    """
    if not isinstance(img_bgr, np.ndarray) or img_bgr.ndim != 3 or img_bgr.size == 0:
        return []
    h, w = img_bgr.shape[:2]
    box = face.get("bbox") if isinstance(face, dict) else None
    vals: list[float] = []
    if isinstance(box, (list, tuple)) and len(box) == 4:
        try:
            vals = [float(t) for t in box]
        except (TypeError, ValueError):
            vals = []
    if len(vals) != 4 or vals[2] <= 1 or vals[3] <= 1:
        norm = face.get("bbox_norm") if isinstance(face, dict) else None
        if isinstance(norm, (list, tuple)) and len(norm) == 4:
            try:
                nx, ny, nw, nh = (float(t) for t in norm)
                vals = [nx * w, ny * h, nw * w, nh * h]
            except (TypeError, ValueError):
                vals = []
    if len(vals) != 4 or not all(math.isfinite(t) for t in vals):
        return []
    if vals[2] < 24.0 or vals[3] < 24.0:
        return []
    return vals


def face_skin_texture(img_bgr, face: dict,
                      ref_px: float = FACE_TEXTURE_REF_PX) -> dict:
    """Fine-band amplitude of facial skin, on a face scaled to a common width.

    The within-image ratio above answers "is her face smoother than her arms".
    This answers a different question: "how much grain does this face carry at
    all".  The cheek band is used - below the eyes, inside the middle of the
    face - because it is skin and nothing else: no eyelashes, no lips, no
    hairline, so what is measured is pores and camera grain rather than
    features.

    {"ok", "fine", "mid", "face_px", "reason"}.

    ``fine`` is NOT an absolute property of a person's skin and must never be
    compared against a stored constant.  Scaling the cheek to ``ref_px`` puts
    the band at one scale, but it cannot put back grain the sensor never
    resolved: measured over one person's 24 photographs ``fine`` tracks
    ``face_px`` with a Pearson r of 0.89, running from 0.71 on a face 130 px
    wide to 3.09 on the same skin at 540 px.  Two readings mean something only
    when both faces were the same width to begin with, which is why
    ``face_px`` comes back with the number - the caller has to match it.
    """
    out = {"ok": False, "fine": 0.0, "mid": 0.0, "face_px": 0.0, "reason": ""}
    if not isinstance(img_bgr, np.ndarray) or img_bgr.ndim != 3 or img_bgr.size == 0:
        out["reason"] = "imagen invalida"
        return out
    h, w = img_bgr.shape[:2]
    vals = face_box_px(img_bgr, face)
    if not vals:
        out["reason"] = "sin rostro medible"
        return out
    fx, fy, fw, fh = vals

    x0 = int(_clamp(fx + 0.18 * fw, 0, w - 2))
    x1 = int(_clamp(fx + 0.82 * fw, x0 + 2, w))
    y0 = int(_clamp(fy + 0.45 * fh, 0, h - 2))
    y1 = int(_clamp(fy + 0.80 * fh, y0 + 2, h))
    patch = img_bgr[y0:y1, x0:x1]
    if patch.size == 0 or min(patch.shape[:2]) < 8:
        out["reason"] = "sin rostro medible"
        return out

    scale = float(ref_px) / fw
    if abs(scale - 1.0) > 0.01:
        patch = cv2.resize(patch, None, fx=scale, fy=scale,
                           interpolation=cv2.INTER_CUBIC if scale > 1
                           else cv2.INTER_AREA)
    if min(patch.shape[:2]) < 12:
        out["reason"] = "rostro demasiado pequeno"
        return out

    skin = skin_mask_ycrcb(patch) > 0
    # The mask is a filter, not a decision: when it finds too little skin the
    # cheek crop is used whole rather than reporting nothing.
    if int(np.count_nonzero(skin)) < 400:
        skin = np.ones(patch.shape[:2], bool)
    gray = cv2.cvtColor(patch, cv2.COLOR_BGR2GRAY).astype(np.float32)
    blur_fine = cv2.GaussianBlur(gray, (0, 0), _FINE_SIGMA)
    blur_mid = cv2.GaussianBlur(gray, (0, 0), _MID_SIGMA)
    out["fine"] = round(float(np.std((gray - blur_fine)[skin])), 4)
    out["mid"] = round(float(np.std((blur_fine - blur_mid)[skin])), 4)
    out["face_px"] = round(float(fw), 1)
    out["ok"] = True
    return out


def _border_defects(small, person_s, scale: float) -> list[dict]:
    """A composite rim: a thin band whose brightness belongs to neither side."""
    cov = float(np.count_nonzero(person_s)) / float(person_s.size)
    if cov < 0.03 or cov > 0.97:
        return []
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    er2 = cv2.erode(person_s, k, iterations=2)
    er5 = cv2.erode(person_s, k, iterations=5)
    di2 = cv2.dilate(person_s, k, iterations=2)
    di6 = cv2.dilate(person_s, k, iterations=6)
    edge = cv2.subtract(di2, er2)
    inner = cv2.subtract(er2, er5)
    outer = cv2.subtract(di6, di2)
    gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY).astype(np.float32)

    cell = 24
    hs, ws = person_s.shape[:2]
    total = 0
    bad = []
    for y in range(0, hs - cell + 1, cell):
        for x in range(0, ws - cell + 1, cell):
            e = edge[y:y + cell, x:x + cell]
            if int(np.count_nonzero(e)) < 30:
                continue
            i = inner[y:y + cell, x:x + cell]
            o = outer[y:y + cell, x:x + cell]
            if int(np.count_nonzero(i)) < 20 or int(np.count_nonzero(o)) < 20:
                continue
            g = gray[y:y + cell, x:x + cell]
            me = float(g[e > 0].mean())
            mi = float(g[i > 0].mean())
            mo = float(g[o > 0].mean())
            total += 1
            halo = min(abs(me - mi), abs(me - mo)) - 0.5 * abs(mi - mo)
            if halo > 10.0:
                bad.append((x, y))
    if total < 8 or not bad:
        return []
    frac = len(bad) / float(total)
    if frac < 0.12:
        return []
    xs = [b[0] for b in bad]
    ys = [b[1] for b in bad]
    box = [min(xs) / scale, min(ys) / scale,
           (max(xs) + cell - min(xs)) / scale, (max(ys) + cell - min(ys)) / scale]
    return [_defect("border_artifact", "silhouette", box,
                    _clamp(frac, 0.15, 0.7), True,
                    "borde de recorte visible en el %.0f%% del contorno"
                    % (100.0 * frac))]


def _duplicate_defects(img_bgr, person, h: int, w: int) -> list[dict]:
    """Two near-identical detailed patches far apart: a copy-pasted feature.

    A repeating fabric print produces hundreds of such pairs, so a large count
    is evidence of texture, not of duplication, and is ignored.
    """
    side = 320
    scale = side / float(max(h, w))
    if scale >= 1.0:
        scale = 1.0
    nw, nh = max(64, int(w * scale)), max(64, int(h * scale))
    small = cv2.cvtColor(cv2.resize(img_bgr, (nw, nh), interpolation=cv2.INTER_AREA),
                         cv2.COLOR_BGR2GRAY).astype(np.float32)
    if person is not None:
        pm = cv2.resize(person, (nw, nh), interpolation=cv2.INTER_NEAREST)
    else:
        pm = np.full((nh, nw), 255, np.uint8)

    size, stride = 24, 12
    coords, vecs, var = [], [], []
    for y in range(0, nh - size + 1, stride):
        for x in range(0, nw - size + 1, stride):
            if pm[y + size // 2, x + size // 2] == 0:
                continue
            patch = small[y:y + size, x:x + size].ravel()
            v = float(patch.var())
            if v < 25.0:
                continue
            coords.append((x, y))
            vecs.append(patch)
            var.append(v)
    if len(coords) < 12:
        return []
    order = np.argsort(np.array(var))[::-1][:400]
    coords = [coords[i] for i in order]
    mat = np.stack([vecs[i] for i in order]).astype(np.float32)
    mat -= mat.mean(axis=1, keepdims=True)
    norms = np.linalg.norm(mat, axis=1, keepdims=True)
    norms[norms < 1e-6] = 1e-6
    mat /= norms
    corr = mat @ mat.T

    n = len(coords)
    pos = np.array(coords, np.float32)
    ii, jj = np.triu_indices(n, k=1)
    strong = corr[ii, jj] >= 0.97
    ii, jj = ii[strong], jj[strong]
    if ii.size:
        gap = np.hypot(pos[ii, 0] - pos[jj, 0], pos[ii, 1] - pos[jj, 1])
        far = gap >= 3.0 * size
        ii, jj = ii[far], jj[far]
    if ii.size == 0 or ii.size > 12:
        return []
    best = int(np.argmax(corr[ii, jj]))
    i, j = int(ii[best]), int(jj[best])
    score = float(corr[i, j])
    inv = 1.0 / scale
    x0 = min(coords[i][0], coords[j][0]) * inv
    y0 = min(coords[i][1], coords[j][1]) * inv
    x1 = (max(coords[i][0], coords[j][0]) + size) * inv
    y1 = (max(coords[i][1], coords[j][1]) + size) * inv
    box = [coords[j][0] * inv, coords[j][1] * inv, size * inv, size * inv]
    detail = ("dos zonas casi identicas (correlacion %.2f) separadas %d px"
              % (score, int(math.hypot(x1 - x0, y1 - y0))))
    return [_defect("duplicated_feature", "body", box, 0.5, True, detail)]


# ------------------------------------------------------------------- public

def scan_anomalies(img_bgr, pose: dict, face: dict, masks: dict) -> dict:
    """Full anatomical and texture audit of one generated image."""
    out = {"ok": False, "defects": [], "score": 0.0, "reason": "", "checks": []}
    if not isinstance(img_bgr, np.ndarray) or img_bgr.ndim != 3 or img_bgr.size == 0:
        out["reason"] = "imagen invalida"
        return out
    h, w = img_bgr.shape[:2]
    pose = pose if isinstance(pose, dict) else {}
    face = face if isinstance(face, dict) else {}
    masks = masks if isinstance(masks, dict) else {}

    lm = landmarks_px(pose, w, h) if pose.get("ok") else {}
    frame = torso_frame(lm) if lm else None
    person = _person_of(masks, img_bgr, pose, h, w)

    # Texture work runs on a bounded copy so thresholds mean the same thing on
    # a phone snapshot and on a 4k render.
    scale = min(1.0, 768.0 / float(max(h, w)))
    sw, sh = max(32, int(w * scale)), max(32, int(h * scale))
    small = cv2.resize(img_bgr, (sw, sh), interpolation=cv2.INTER_AREA)
    person_s = (cv2.resize(person, (sw, sh), interpolation=cv2.INTER_NEAREST)
                if person is not None else None)
    masks_s = {}
    for name in ("face", "hair"):
        got = masks.get(name)
        if isinstance(got, np.ndarray) and got.size:
            masks_s[name] = cv2.resize(_resize_mask(got, h, w), (sw, sh),
                                       interpolation=cv2.INTER_NEAREST)
    if "face" not in masks_s:
        box = face.get("bbox")
        bx, by, bw, bh = 0.0, 0.0, 0.0, 0.0
        if isinstance(box, (list, tuple)) and len(box) == 4:
            try:
                bx, by, bw, bh = [float(t) * scale for t in box]
            except (TypeError, ValueError):
                bw = bh = 0.0
            if bw > 6 and bh > 6:
                oval = np.zeros((sh, sw), np.uint8)
                cv2.ellipse(oval, (int(bx + bw / 2), int(by + bh / 2)),
                            (int(bw / 2), int(bh / 2)), 0, 0, 360, 255, -1)
                if np.count_nonzero(oval):
                    masks_s["face"] = oval

    defects: list[dict] = []
    jobs = [
        ("hands", lambda: _hand_defects(img_bgr, lm, frame, h, w)),
        ("extra_person", lambda: _extra_person(img_bgr, face, h, w)),
        ("face", lambda: _face_defects(img_bgr, face, h, w)),
        ("duplicate", lambda: _duplicate_defects(img_bgr, person, h, w)),
    ]
    if person_s is not None:
        jobs.append(("texture", lambda: _texture_defects(small, person_s, masks_s,
                                                         float(sw) / float(w))))
        jobs.append(("border", lambda: _border_defects(small, person_s,
                                                       float(sw) / float(w))))
    for name, job in jobs:
        try:
            found = job() or []
        except Exception:                             # noqa: BLE001
            continue
        out["checks"].append(name)
        defects.extend(d for d in found if isinstance(d, dict))

    defects.sort(key=lambda d: -float(d.get("severity", 0.0)))
    penalty = sum(SEVERITY_WEIGHT.get(d["type"], 0.5) * float(d["severity"])
                  for d in defects)
    out["defects"] = defects
    out["score"] = round(_clamp(1.0 - penalty, 0.0, 1.0), 3)
    out["ok"] = True
    if person is None:
        out["reason"] = "sin silueta: textura y borde no evaluados"
    return out
