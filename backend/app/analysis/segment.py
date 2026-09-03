"""Where the person is, and which part of her is which.

Everything downstream needs pixel ownership: the body measurer needs a
silhouette to find the waist, the repair step needs a mask so it can repaint a
hand without touching the face, and the garment logic needs to know cloth from
skin.  This module answers those questions with three levels of fallback, so a
machine without MediaPipe still produces a usable mask instead of an exception.

Region masks are anatomical constructions - capsules and polygons built around
the pose skeleton with radii proportional to torso length, then cut down to the
person mask.  They are approximate by design: their job is to bound a repair,
not to trace a contour.
"""
from __future__ import annotations

import math

import cv2
import numpy as np

from .body import landmarks_px, torso_frame
from .skin import skin_mask_ycrcb

try:  # heavy, optional; every path below degrades without it
    import mediapipe as mp
except Exception:                                     # noqa: BLE001
    mp = None

SEG_THRESHOLD = 0.5
MIN_COVERAGE = 0.005


# ------------------------------------------------------------------ helpers

def _bin(mask, h: int, w: int):
    """Any mask-ish array -> uint8 0/255 of shape (h, w), or None."""
    if not isinstance(mask, np.ndarray) or mask.size == 0:
        return None
    m = mask[:, :, 0] if mask.ndim == 3 else mask
    if m.dtype == np.bool_:
        m = m.astype(np.uint8) * 255
    elif m.dtype.kind == "f":
        peak = float(np.nanmax(m)) if np.isfinite(m).any() else 0.0
        m = ((m >= 0.5) if peak <= 1.001 else (m >= 128)).astype(np.uint8) * 255
    else:
        m = m.astype(np.uint8)
        m = ((m >= 128) if int(m.max()) > 1 else (m > 0)).astype(np.uint8) * 255
    if m.shape[:2] != (h, w):
        m = cv2.resize(m, (w, h), interpolation=cv2.INTER_NEAREST)
    return m


def _kernel(h: int, w: int, frac: float = 0.006):
    k = max(3, int(min(h, w) * frac))
    return cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k | 1, k | 1))


def _largest(mask):
    n, labels, stats, _ = cv2.connectedComponentsWithStats((mask > 0).astype(np.uint8), 8)
    if n <= 1:
        return None
    idx = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    return (labels == idx).astype(np.uint8) * 255


def _clean(mask, h: int, w: int):
    k = _kernel(h, w)
    m = cv2.morphologyEx(mask, cv2.MORPH_OPEN, k)
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, k)
    return _largest(m)


def _capsule(canvas, p1, p2, radius: float, value: int = 255) -> None:
    r = int(max(1, round(radius)))
    a = (int(round(p1[0])), int(round(p1[1])))
    b = (int(round(p2[0])), int(round(p2[1])))
    cv2.line(canvas, a, b, value, r * 2)
    cv2.circle(canvas, a, r, value, -1)
    cv2.circle(canvas, b, r, value, -1)


def _grown_poly(points, radius: float):
    """Convex hull of the points pushed outward by ``radius`` from its centre."""
    if len(points) < 3:
        return None
    arr = np.array([[float(p[0]), float(p[1])] for p in points], np.float32)
    hull = cv2.convexHull(arr).reshape(-1, 2)
    centre = hull.mean(axis=0)
    grown = []
    for p in hull:
        d = p - centre
        n = float(math.hypot(float(d[0]), float(d[1])))
        grown.append(p + d / n * radius if n > 1e-6 else p)
    return np.array(grown, np.int32)


def _pose_for(img_bgr, pose):
    """Use the pose we were handed; otherwise ask the pose module for one."""
    if isinstance(pose, dict) and pose.get("ok"):
        return pose
    try:
        from . import pose as pose_module
        got = pose_module.detect_pose(img_bgr)
        return got if isinstance(got, dict) and got.get("ok") else None
    except Exception:                                 # noqa: BLE001
        return None


def _visible(lm: dict, name: str, t: float = 0.4):
    p = lm.get(name)
    return (p[0], p[1]) if p is not None and p[3] >= t else None


# ------------------------------------------------------------- person mask

def _mediapipe_person(img_bgr, h: int, w: int):
    if mp is None:
        return None
    try:
        rgb = np.ascontiguousarray(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB))
        with mp.solutions.selfie_segmentation.SelfieSegmentation(
                model_selection=1) as seg:
            res = seg.process(rgb)
        raw = getattr(res, "segmentation_mask", None)
        if raw is None:
            return None
        m = (np.asarray(raw, np.float32) >= SEG_THRESHOLD).astype(np.uint8) * 255
        if m.shape[:2] != (h, w):
            m = cv2.resize(m, (w, h), interpolation=cv2.INTER_NEAREST)
        return _clean(m, h, w)
    except Exception:                                 # noqa: BLE001
        return None


def _grabcut_person(img_bgr, lm: dict, h: int, w: int):
    """Colour cut seeded by the pose bounding box."""
    pts = [(p[0], p[1]) for p in lm.values() if p[3] >= 0.2] if lm else []
    if len(pts) >= 4:
        xs = [p[0] for p in pts]
        ys = [p[1] for p in pts]
        pad = 0.10 * max(w, h)
        x0 = int(max(0, min(xs) - pad))
        y0 = int(max(0, min(ys) - 1.4 * pad))
        x1 = int(min(w, max(xs) + pad))
        y1 = int(min(h, max(ys) + pad))
    else:
        x0, y0 = int(0.15 * w), int(0.05 * h)
        x1, y1 = int(0.85 * w), int(0.98 * h)
    if x1 - x0 < 32 or y1 - y0 < 32:
        return None
    try:
        gc = np.zeros((h, w), np.uint8)
        bgd = np.zeros((1, 65), np.float64)
        fgd = np.zeros((1, 65), np.float64)
        cv2.grabCut(np.ascontiguousarray(img_bgr), gc,
                    (x0, y0, x1 - x0, y1 - y0), bgd, fgd, 4,
                    cv2.GC_INIT_WITH_RECT)
    except Exception:                                 # noqa: BLE001
        return None
    m = np.where((gc == cv2.GC_FGD) | (gc == cv2.GC_PR_FGD), 255, 0).astype(np.uint8)
    return _clean(m, h, w)


def _hull_person(lm: dict, h: int, w: int):
    pts = [(p[0], p[1]) for p in lm.values() if p[3] >= 0.2] if lm else []
    if len(pts) < 3:
        return None
    try:
        frame = torso_frame(lm)
        radius = 0.18 * frame["torso"] if frame else 0.05 * max(h, w)
        poly = _grown_poly(pts, radius)
        if poly is None:
            return None
        m = np.zeros((h, w), np.uint8)
        cv2.fillConvexPoly(m, poly, 255)
    except Exception:                                 # noqa: BLE001
        return None
    return m


def person_mask(img_bgr, pose: dict | None = None) -> dict:
    """Silhouette of the subject. Always returns a mask array, ok says if it
    is trustworthy."""
    out = {"ok": False, "mask": None, "coverage": 0.0, "backend": "none",
           "reason": ""}
    if not isinstance(img_bgr, np.ndarray) or img_bgr.ndim != 3 or img_bgr.size == 0:
        out["reason"] = "imagen invalida"
        out["mask"] = np.zeros((1, 1), np.uint8)
        return out
    h, w = img_bgr.shape[:2]

    mask = _mediapipe_person(img_bgr, h, w)
    backend = "mediapipe"
    if mask is None or float(np.count_nonzero(mask)) / (h * w) < MIN_COVERAGE:
        got = _pose_for(img_bgr, pose)
        lm = landmarks_px(got, w, h) if got else {}
        mask = _grabcut_person(img_bgr, lm, h, w)
        backend = "grabcut"
        if mask is None or float(np.count_nonzero(mask)) / (h * w) < MIN_COVERAGE:
            mask = _hull_person(lm, h, w)
            backend = "hull"

    if mask is None:
        out["mask"] = np.zeros((h, w), np.uint8)
        out["reason"] = "no se pudo aislar a la persona"
        return out

    coverage = float(np.count_nonzero(mask)) / float(h * w)
    out["mask"] = mask
    out["coverage"] = round(coverage, 4)
    out["backend"] = backend
    out["ok"] = coverage >= MIN_COVERAGE
    if not out["ok"]:
        out["reason"] = "silueta demasiado pequena"
    elif backend == "hull":
        out["reason"] = "silueta aproximada por envolvente de landmarks"
    return out


# ------------------------------------------------------------ region masks

def _face_ellipse(lm: dict, frame: dict | None, h: int, w: int):
    """(centre, axes, angle_deg) of the head, from ears or eyes."""
    ears = (_visible(lm, "left_ear", 0.3), _visible(lm, "right_ear", 0.3))
    eyes = (_visible(lm, "left_eye", 0.3), _visible(lm, "right_eye", 0.3))
    if all(ears):
        span = math.hypot(ears[0][0] - ears[1][0], ears[0][1] - ears[1][1])
        centre = ((ears[0][0] + ears[1][0]) / 2.0, (ears[0][1] + ears[1][1]) / 2.0)
        axis = (ears[0][0] - ears[1][0], ears[0][1] - ears[1][1])
    elif all(eyes):
        span = 2.2 * math.hypot(eyes[0][0] - eyes[1][0], eyes[0][1] - eyes[1][1])
        centre = ((eyes[0][0] + eyes[1][0]) / 2.0, (eyes[0][1] + eyes[1][1]) / 2.0)
        axis = (eyes[0][0] - eyes[1][0], eyes[0][1] - eyes[1][1])
    else:
        return None
    if span < 10.0 or span > 2.0 * max(h, w):
        return None
    ang = math.degrees(math.atan2(axis[1], axis[0]))
    if frame is not None:
        vx, vy = frame["v"]
        centre = (centre[0] + 0.10 * span * vx, centre[1] + 0.10 * span * vy)
    else:
        centre = (centre[0], centre[1] + 0.10 * span)
    return (centre, (0.62 * span, 0.95 * span), ang)


def region_masks(img_bgr, pose: dict, person=None) -> dict:
    """Anatomical regions as uint8 0/255 masks; unmeasurable regions omitted.

    Partial results are kept: a region that cannot be built is left out rather
    than taking the rest of the dictionary with it.
    """
    out: dict = {}
    try:
        _build_regions(out, img_bgr, pose, person)
    except Exception:                                 # noqa: BLE001
        pass
    return out


def _build_regions(out: dict, img_bgr, pose: dict, person) -> None:
    if not isinstance(img_bgr, np.ndarray) or img_bgr.ndim != 3 or img_bgr.size == 0:
        return
    h, w = img_bgr.shape[:2]

    per = _bin(person, h, w)
    if per is None:
        got = person_mask(img_bgr, pose)
        per = got["mask"] if got.get("ok") else None
    if per is None:
        per = np.full((h, w), 255, np.uint8)
    out["background"] = cv2.bitwise_not(per)

    got_pose = pose if isinstance(pose, dict) and pose.get("ok") else None
    lm = landmarks_px(got_pose, w, h) if got_pose else {}
    if not lm:
        return
    frame = torso_frame(lm)
    torso = frame["torso"] if frame else 0.0

    def clipped(canvas):
        m = cv2.bitwise_and(canvas, per)
        return m if np.count_nonzero(m) >= 16 else None

    # ------------------------------------------------------------- head
    ell = _face_ellipse(lm, frame, h, w)
    if ell is not None:
        centre, axes, ang = ell
        face_c = np.zeros((h, w), np.uint8)
        cv2.ellipse(face_c, (int(round(centre[0])), int(round(centre[1]))),
                    (int(round(axes[0])), int(round(axes[1]))), ang, 0, 360, 255, -1)
        face = clipped(face_c)
        if face is not None:
            out["face"] = face

        big = np.zeros((h, w), np.uint8)
        cv2.ellipse(big, (int(round(centre[0])), int(round(centre[1]))),
                    (int(round(axes[0] * 1.32)), int(round(axes[1] * 1.30))),
                    ang, 0, 360, 255, -1)
        # Hair lives above the face line, is inside the person, and is not skin.
        up = np.zeros((h, w), np.uint8)
        if frame is not None:
            vx, vy = frame["v"]
            far = min(2.0 * max(h, w), 12000.0)   # stay inside cv2 fixed point
            px, py = -vy, vx
            base = (centre[0] + 0.10 * axes[1] * vx, centre[1] + 0.10 * axes[1] * vy)
            quad = np.array([
                [base[0] - px * far, base[1] - py * far],
                [base[0] + px * far, base[1] + py * far],
                [base[0] + px * far - vx * far, base[1] + py * far - vy * far],
                [base[0] - px * far - vx * far, base[1] - py * far - vy * far],
            ], np.int32)
            cv2.fillConvexPoly(up, quad, 255)
        else:
            up[:max(0, int(centre[1] + 0.10 * axes[1])), :] = 255
        hair_c = cv2.bitwise_and(cv2.bitwise_and(big, up), cv2.bitwise_not(face_c))
        hair_c = cv2.bitwise_and(hair_c, cv2.bitwise_not(skin_mask_ycrcb(img_bgr)))
        hair = clipped(hair_c)
        if hair is not None:
            out["hair"] = hair

    if frame is None:
        return

    # -------------------------------------------------------- upper body
    quad = [frame["ls"], frame["rs"], frame["rh"], frame["lh"]]
    poly = _grown_poly(quad, 0.14 * torso)
    if poly is not None:
        canvas = np.zeros((h, w), np.uint8)
        cv2.fillConvexPoly(canvas, poly, 255)
        _capsule(canvas, frame["S"], frame["H"], 0.26 * torso)
        upper = clipped(canvas)
        if upper is not None:
            out["upper_body"] = upper

    # -------------------------------------------------------- lower body
    vx, vy = frame["v"]
    knees = [_visible(lm, s + "_knee", 0.35) for s in ("left", "right")]
    bottom = [k for k in knees if k]
    if not bottom:
        bottom = [(frame["lh"][0] + 0.95 * torso * vx, frame["lh"][1] + 0.95 * torso * vy),
                  (frame["rh"][0] + 0.95 * torso * vx, frame["rh"][1] + 0.95 * torso * vy)]
    poly = _grown_poly([frame["lh"], frame["rh"]] + bottom, 0.13 * torso)
    if poly is not None:
        canvas = np.zeros((h, w), np.uint8)
        cv2.fillConvexPoly(canvas, poly, 255)
        lower = clipped(canvas)
        if lower is not None:
            out["lower_body"] = lower

    # ------------------------------------------------------- limbs, hands
    arms = np.zeros((h, w), np.uint8)
    hands = np.zeros((h, w), np.uint8)
    legs = np.zeros((h, w), np.uint8)
    for side in ("left", "right"):
        sh = _visible(lm, side + "_shoulder", 0.35)
        el = _visible(lm, side + "_elbow", 0.35)
        wr = _visible(lm, side + "_wrist", 0.35)
        if sh and el:
            _capsule(arms, sh, el, 0.14 * torso)
        if el and wr:
            _capsule(arms, el, wr, 0.11 * torso)
            # The hand continues past the wrist along the forearm direction.
            dx, dy = wr[0] - el[0], wr[1] - el[1]
            n = math.hypot(dx, dy)
            if n > 1e-6:
                tip = (wr[0] + dx / n * 0.20 * torso, wr[1] + dy / n * 0.20 * torso)
                _capsule(hands, wr, tip, 0.13 * torso)
        elif wr:
            cv2.circle(hands, (int(wr[0]), int(wr[1])), int(max(3, 0.15 * torso)), 255, -1)

        hip = _visible(lm, side + "_hip", 0.35)
        kn = _visible(lm, side + "_knee", 0.35)
        an = _visible(lm, side + "_ankle", 0.35)
        if hip and kn:
            _capsule(legs, hip, kn, 0.15 * torso)
        if kn and an:
            _capsule(legs, kn, an, 0.11 * torso)
            foot = _visible(lm, side + "_foot_index", 0.35)
            if foot:
                _capsule(legs, an, foot, 0.09 * torso)

    for name, canvas in (("arms", arms), ("hands", hands), ("legs", legs)):
        got = clipped(canvas)
        if got is not None:
            out[name] = got


def garment_mask(img_bgr, pose: dict, person_mask=None) -> np.ndarray:
    """Clothing: the torso and legs regions with every skin pixel removed."""
    if not isinstance(img_bgr, np.ndarray) or img_bgr.ndim != 3 or img_bgr.size == 0:
        return np.zeros((1, 1), np.uint8)
    h, w = img_bgr.shape[:2]
    regions = region_masks(img_bgr, pose, person_mask)
    canvas = np.zeros((h, w), np.uint8)
    found = False
    for name in ("upper_body", "lower_body"):
        got = regions.get(name)
        if got is not None:
            canvas = cv2.bitwise_or(canvas, got)
            found = True
    if not found:
        return canvas
    skin = skin_mask_ycrcb(img_bgr)
    skin = cv2.dilate(skin, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)))
    canvas = cv2.bitwise_and(canvas, cv2.bitwise_not(skin))
    return cv2.morphologyEx(canvas, cv2.MORPH_OPEN, _kernel(h, w, 0.004))


def bbox_of(mask) -> list[int]:
    """[x, y, w, h] of the set pixels, [] when the mask is empty."""
    if not isinstance(mask, np.ndarray) or mask.size == 0:
        return []
    m = mask[:, :, 0] if mask.ndim == 3 else mask
    ys, xs = np.nonzero(m)
    if xs.size == 0:
        return []
    x0, x1 = int(xs.min()), int(xs.max())
    y0, y1 = int(ys.min()), int(ys.max())
    return [x0, y0, x1 - x0 + 1, y1 - y0 + 1]
