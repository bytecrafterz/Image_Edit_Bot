"""Skin tone as a number, so "they changed my colour" stops being an opinion.

Sampling is the whole problem.  A naive average over a face box measures hair,
lipstick, shadow and the ring light before it measures skin, and then two
photographs of the same person disagree by more than two different people do.
So this module samples named anatomical sites (forehead, both cheeks, the
neck/chest triangle, the upper arms), throws away every pixel that is not
skin-like, every specular highlight and every deep shadow, and only then
averages in CIE Lab where distance means something to the eye.

ITA (individual typology angle) is the dermatological summary of the result and
is what actually catches a whitened or tanned render.
"""
from __future__ import annotations

import math

import cv2
import numpy as np

from .body import landmarks_px, torso_frame

# YCrCb skin envelope, plus guards: grey pixels are cloth or wall, oversaturated
# pixels are fabric, and Lab L outside this band is a highlight or a shadow.
CR_RANGE = (133, 176)
CB_RANGE = (77, 128)
SAT_RANGE = (25, 190)
VAL_MIN = 40
L_HI = 95.0
L_LO = 20.0

MIN_SITE_PX = 24
MIN_SITE_FRACTION = 0.20
DELTA_E_FULL = 25.0     # contract: similarity = 1 - dE/25


# ------------------------------------------------------------------ helpers

def _clamp(v: float, lo: float, hi: float) -> float:
    return lo if v < lo else (hi if v > hi else v)


def skin_mask_ycrcb(img_bgr) -> np.ndarray:
    """Loose skin envelope over a whole frame; uint8 0/255.

    Public because segmentation needs it to separate hair from forehead and
    clothing from bare arms.  Deliberately permissive: it is a filter, never a
    decision on its own.
    """
    if not isinstance(img_bgr, np.ndarray) or img_bgr.ndim != 3:
        return np.zeros((1, 1), np.uint8)
    ycc = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2YCrCb)
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    cr, cb = ycc[:, :, 1], ycc[:, :, 2]
    sat, val = hsv[:, :, 1], hsv[:, :, 2]
    ok = ((cr >= CR_RANGE[0]) & (cr <= CR_RANGE[1]) &
          (cb >= CB_RANGE[0]) & (cb <= CB_RANGE[1]) &
          (sat >= SAT_RANGE[0]) & (sat <= SAT_RANGE[1]) &
          (val >= VAL_MIN))
    out = (ok.astype(np.uint8) * 255)
    kern = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    return cv2.morphologyEx(out, cv2.MORPH_OPEN, kern)


def _disc_pixels(img_bgr, cx: float, cy: float, r: float):
    """Skin-like, well-exposed Lab pixels inside one circular sample site."""
    h, w = img_bgr.shape[:2]
    r = float(max(3.0, r))
    x0, x1 = int(max(0, cx - r)), int(min(w, cx + r + 1))
    y0, y1 = int(max(0, cy - r)), int(min(h, cy + r + 1))
    if x1 - x0 < 5 or y1 - y0 < 5:
        return None
    patch = img_bgr[y0:y1, x0:x1]
    yy, xx = np.mgrid[y0:y1, x0:x1]
    disc = ((xx - cx) ** 2 + (yy - cy) ** 2) <= r * r
    if int(disc.sum()) < MIN_SITE_PX:
        return None

    ycc = cv2.cvtColor(patch, cv2.COLOR_BGR2YCrCb)
    hsv = cv2.cvtColor(patch, cv2.COLOR_BGR2HSV)
    lab = cv2.cvtColor(patch.astype(np.float32) / 255.0, cv2.COLOR_BGR2LAB)
    cr, cb = ycc[:, :, 1], ycc[:, :, 2]
    sat, val = hsv[:, :, 1], hsv[:, :, 2]
    lch = lab[:, :, 0]
    keep = (disc &
            (cr >= CR_RANGE[0]) & (cr <= CR_RANGE[1]) &
            (cb >= CB_RANGE[0]) & (cb <= CB_RANGE[1]) &
            (sat >= SAT_RANGE[0]) & (sat <= SAT_RANGE[1]) &
            (val >= VAL_MIN) & (lch < L_HI) & (lch > L_LO))
    n = int(keep.sum())
    if n < MIN_SITE_PX or n < MIN_SITE_FRACTION * float(disc.sum()):
        return None
    return {"lab": lab[keep].reshape(-1, 3).astype(np.float64),
            "bgr": patch[keep].reshape(-1, 3).astype(np.float64),
            "n": n}


def _unit(dx: float, dy: float):
    n = math.hypot(dx, dy)
    return (dx / n, dy / n) if n > 1e-6 else (1.0, 0.0)


def named_point_px(face: dict, name: str, w: int, h: int):
    """One named face landmark in pixels, whichever convention it arrived in.

    face.py publishes ``px`` alongside coordinates normalised to the whole
    frame; both are accepted, and pixel-only producers work too.
    """
    lms = face.get("landmarks") if isinstance(face, dict) else None
    if not isinstance(lms, dict):
        return None
    val = lms.get(name)
    if not isinstance(val, dict):
        return None
    px = val.get("px")
    if isinstance(px, (list, tuple)) and len(px) >= 2:
        try:
            return (float(px[0]), float(px[1]))
        except (TypeError, ValueError):
            return None
    try:
        x, y = float(val["x"]), float(val["y"])
    except (KeyError, TypeError, ValueError):
        return None
    if not (math.isfinite(x) and math.isfinite(y)):
        return None
    return (x * w, y * h) if max(abs(x), abs(y)) <= 1.6 else (x, y)


def _eye_pair(lm: dict, face: dict, w: int, h: int):
    """Eye centres in pixels from pose, or from the face module's landmarks."""
    le, re = lm.get("left_eye"), lm.get("right_eye")
    if le is not None and re is not None and le[3] >= 0.3 and re[3] >= 0.3:
        return (le[0], le[1]), (re[0], re[1])
    a = named_point_px(face, "left_eye", w, h)
    b = named_point_px(face, "right_eye", w, h)
    if a is not None and b is not None:
        return a, b
    return None


def _bbox_px(face: dict):
    """[x, y, w, h] in pixels, or None."""
    box = face.get("bbox") if isinstance(face, dict) else None
    if not (isinstance(box, (list, tuple)) and len(box) == 4):
        return None
    try:
        x, y, bw, bh = [float(t) for t in box]
    except (TypeError, ValueError):
        return None
    if bw <= 8.0 or bh <= 8.0:
        return None
    return (x, y, bw, bh)


def _face_sites(lm: dict, face: dict, w: int, h: int) -> list[dict]:
    """Forehead and both cheeks, placed in the eye-line frame so head tilt
    does not walk the sites off the face."""
    sites: list[dict] = []
    pair = _eye_pair(lm, face, w, h)
    if pair is not None:
        (lx, ly), (rx, ry) = pair
        iod = math.hypot(lx - rx, ly - ry)
        if iod >= 8.0:
            ex, ey = _unit(lx - rx, ly - ry)          # along the eye line
            dx, dy = -ey, ex                          # perpendicular
            mid = ((lx + rx) / 2.0, (ly + ry) / 2.0)
            chin = None
            nose = lm.get("nose")
            if nose is not None and nose[3] >= 0.3:
                chin = (nose[0], nose[1])
            else:
                for name in ("chin", "nose_tip"):
                    chin = named_point_px(face, name, w, h)
                    if chin is not None:
                        break
            if chin is not None:
                if (chin[0] - mid[0]) * dx + (chin[1] - mid[1]) * dy < 0:
                    dx, dy = -dx, -dy                 # (dx, dy) points at the chin
            elif dy < 0:
                dx, dy = -dx, -dy
            sites.append({"region": "face", "name": "forehead",
                          "x": mid[0] - 0.75 * iod * dx,
                          "y": mid[1] - 0.75 * iod * dy,
                          "r": 0.22 * iod})
            for tag, (px_, py_) in (("cheek_left", (lx, ly)), ("cheek_right", (rx, ry))):
                sites.append({"region": "face", "name": tag,
                              "x": px_ + 0.55 * iod * dx + 0.14 * (mid[0] - px_),
                              "y": py_ + 0.55 * iod * dy + 0.14 * (mid[1] - py_),
                              "r": 0.20 * iod})
            return sites

    box = _bbox_px(face)
    if box is not None:
        x, y, bw, bh = box
        sites.append({"region": "face", "name": "forehead",
                      "x": x + 0.50 * bw, "y": y + 0.22 * bh, "r": 0.10 * bw})
        sites.append({"region": "face", "name": "cheek_left",
                      "x": x + 0.24 * bw, "y": y + 0.62 * bh, "r": 0.09 * bw})
        sites.append({"region": "face", "name": "cheek_right",
                      "x": x + 0.76 * bw, "y": y + 0.62 * bh, "r": 0.09 * bw})
    return sites


def _body_sites(lm: dict) -> list[dict]:
    sites: list[dict] = []
    frame = torso_frame(lm)
    if frame is not None:
        sx, sy = frame["S"]
        vx, vy = frame["v"]
        torso = frame["torso"]
        sites.append({"region": "chest", "name": "neck",
                      "x": sx - 0.10 * torso * vx, "y": sy - 0.10 * torso * vy,
                      "r": 0.07 * torso})
        sites.append({"region": "chest", "name": "chest",
                      "x": sx + 0.18 * torso * vx, "y": sy + 0.18 * torso * vy,
                      "r": 0.11 * torso})
        scale = torso
    else:
        scale = 0.0

    for side in ("left", "right"):
        sh = lm.get(side + "_shoulder")
        el = lm.get(side + "_elbow")
        if sh is None or el is None or sh[3] < 0.45 or el[3] < 0.45:
            continue
        arm = math.hypot(sh[0] - el[0], sh[1] - el[1])
        if arm < 12.0:
            continue
        sites.append({"region": "arm", "name": side + "_upper_arm",
                      "x": (sh[0] + el[0]) / 2.0, "y": (sh[1] + el[1]) / 2.0,
                      "r": max(6.0, 0.16 * arm if scale <= 0 else 0.09 * scale)})
    return sites


# ------------------------------------------------------------------- public

def skin_stats(img_bgr, pose: dict, face: dict) -> dict:
    """Mean skin colour of one photograph, in Lab, from named sites only."""
    out = {"ok": False, "lab_mean": [], "lab_std": [], "rgb_mean": [],
           "ita_deg": 0.0, "samples": 0, "regions": {}, "reason": ""}
    if not isinstance(img_bgr, np.ndarray) or img_bgr.ndim != 3 or img_bgr.size == 0:
        out["reason"] = "imagen invalida"
        return out
    h, w = img_bgr.shape[:2]
    pose = pose if isinstance(pose, dict) else {}
    face = face if isinstance(face, dict) else {}

    lm = landmarks_px(pose, w, h) if pose.get("ok") else {}
    sites = _face_sites(lm, face, w, h) + _body_sites(lm)
    if not sites:
        out["reason"] = "sin puntos de referencia para muestrear piel"
        return out

    per_region: dict = {}
    taken: list[dict] = []
    all_lab: list[np.ndarray] = []
    all_bgr: list[np.ndarray] = []
    for site in sites:
        try:
            got = _disc_pixels(img_bgr, site["x"], site["y"], site["r"])
        except Exception:                             # noqa: BLE001
            got = None
        if got is None:
            continue
        per_region.setdefault(site["region"], []).append(got["lab"])
        all_lab.append(got["lab"])
        all_bgr.append(got["bgr"])
        taken.append({"name": site["name"], "region": site["region"],
                      "n": got["n"]})
    if not all_lab:
        out["reason"] = "ninguna muestra paso el filtro de piel"
        return out

    lab = np.concatenate(all_lab, axis=0)
    bgr = np.concatenate(all_bgr, axis=0)
    mean = lab.mean(axis=0)
    std = lab.std(axis=0)
    bgr_mean = bgr.mean(axis=0)

    b_star = float(mean[2])
    if abs(b_star) < 1e-6:
        ita = 90.0 if mean[0] >= 50.0 else -90.0
    else:
        ita = math.degrees(math.atan2(float(mean[0]) - 50.0, b_star))

    out["ok"] = True
    out["lab_mean"] = [round(float(v), 3) for v in mean]
    out["lab_std"] = [round(float(v), 3) for v in std]
    out["rgb_mean"] = [round(float(bgr_mean[2]), 2), round(float(bgr_mean[1]), 2),
                       round(float(bgr_mean[0]), 2)]
    out["ita_deg"] = round(ita, 2)
    out["samples"] = len(taken)
    out["n_px"] = int(lab.shape[0])
    out["regions"] = {k: [round(float(v), 3) for v in np.concatenate(vals, axis=0).mean(axis=0)]
                      for k, vals in per_region.items()}
    out["sites"] = taken
    return out


def compare_skin(a: dict, b: dict) -> dict:
    """CIE76 distance between two skin measurements."""
    out = {"delta_e": 0.0, "delta_L": 0.0, "ita_delta": 0.0,
           "similarity": 0.0, "ok": False, "reason": ""}
    la = a.get("lab_mean") if isinstance(a, dict) else None
    lb = b.get("lab_mean") if isinstance(b, dict) else None
    if not (isinstance(la, (list, tuple)) and isinstance(lb, (list, tuple))
            and len(la) >= 3 and len(lb) >= 3):
        out["reason"] = "falta la medida de piel en una de las dos imagenes"
        return out
    try:
        va = [float(t) for t in la[:3]]
        vb = [float(t) for t in lb[:3]]
    except (TypeError, ValueError):
        out["reason"] = "medida de piel invalida"
        return out

    de = math.sqrt(sum((x - y) ** 2 for x, y in zip(va, vb)))
    out["ok"] = True
    out["delta_e"] = round(de, 3)
    out["delta_L"] = round(va[0] - vb[0], 3)
    out["ita_delta"] = round(float(a.get("ita_deg", 0.0)) - float(b.get("ita_deg", 0.0)), 2)
    out["similarity"] = round(_clamp(1.0 - de / DELTA_E_FULL, 0.0, 1.0), 4)
    return out
