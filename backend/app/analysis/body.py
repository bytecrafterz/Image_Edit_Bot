"""Body proportions, measured the way a tailor would.

The client was made slimmer by another tool and had no way to prove it.  The
answer is arithmetic.  Every distance here is divided by TORSO LENGTH (shoulder
midpoint to hip midpoint), so cropping, resizing or re-framing the photograph
does not move the numbers - only the body itself does.  Width at the waist and
at the chest is read off the silhouette rather than off the landmarks, because
a landmark skeleton survives a slim-down filter untouched while the silhouette
does not.

Two systematic errors are corrected instead of ignored: in-plane roll (the
measuring frame is rotated so the shoulder line is horizontal) and yaw (a
turned subject is foreshortened, so the affected widths are reported as
unreliable rather than passed off as truth).

Nothing in here raises.  An image that cannot be measured comes back as
{"ok": False, "reason": ...} and the caller decides what that means.
"""
from __future__ import annotations

import math

import cv2
import numpy as np

# --------------------------------------------------------------- constants

VIS_MIN = 0.35          # below this a landmark is a guess, not a measurement
WAIST_T = 0.55          # position along the shoulder->hip axis
BUST_T = 0.22
YAW_SUSPECT_DEG = 15.0  # widths start shrinking measurably from here
YAW_REJECT_DEG = 45.0   # ... and stop meaning anything here
MIN_TORSO_PX = 24.0
SHAPE_SAMPLES = 12      # heights sampled across the silhouette

WIDTH_METRICS = ("shoulder_w_over_torso", "hip_w_over_torso",
                 "waist_w_over_torso", "bust_w_over_torso",
                 "neck_w_over_torso")

_LIMB_CHAINS = {
    "arm": (("left_shoulder", "left_elbow", "left_wrist"),
            ("right_shoulder", "right_elbow", "right_wrist")),
    "leg": (("left_hip", "left_knee", "left_ankle"),
            ("right_hip", "right_knee", "right_ankle")),
}


# ----------------------------------------------------------------- helpers

def _clamp(v: float, lo: float, hi: float) -> float:
    return lo if v < lo else (hi if v > hi else v)


def _dist(a, b) -> float:
    return math.hypot(a[0] - b[0], a[1] - b[1])


def _mid(a, b):
    return ((a[0] + b[0]) / 2.0, (a[1] + b[1]) / 2.0)


def landmarks_px(pose: dict, width: int, height: int,
                 min_v: float = 0.0) -> dict:
    """Pose landmarks in pixels: {name: (x_px, y_px, z, visibility)}.

    The contract says landmarks arrive normalised 0..1; pixel input is accepted
    too and detected by magnitude, because a wrong guess here would silently
    corrupt every measurement downstream.  ``z`` is always returned in the same
    normalised units as x (MediaPipe's own convention) so yaw can be recovered
    from it.
    """
    out: dict = {}
    if not isinstance(pose, dict):
        return out
    lms = pose.get("landmarks")
    if not isinstance(lms, dict) or not lms:
        return out

    span = 0.0
    for val in lms.values():
        if isinstance(val, dict):
            try:
                span = max(span, abs(float(val.get("x", 0.0))),
                           abs(float(val.get("y", 0.0))))
            except (TypeError, ValueError):
                continue
    normalised = span <= 1.6
    wf = float(max(int(width), 1))
    hf = float(max(int(height), 1))

    for name, val in lms.items():
        if not isinstance(val, dict):
            continue
        try:
            x = float(val.get("x", 0.0))
            y = float(val.get("y", 0.0))
            z = float(val.get("z", 0.0))
            vis = val.get("v", val.get("visibility", 1.0))
            vis = float(1.0 if vis is None else vis)
        except (TypeError, ValueError):
            continue
        if not (math.isfinite(x) and math.isfinite(y)):
            continue
        if not math.isfinite(z):
            z = 0.0
        if not math.isfinite(vis):
            vis = 0.0
        if normalised:
            x *= wf
            y *= hf
        else:
            z /= wf
        if vis < min_v:
            continue
        out[str(name)] = (x, y, z, _clamp(vis, 0.0, 1.0))
    return out


def torso_frame(lm: dict, vis: float = VIS_MIN) -> dict | None:
    """Roll-corrected measuring frame built on the torso.

    ``u`` runs along the shoulder line (the "horizontal" axis after roll
    correction), ``v`` is perpendicular to it and always points from the
    shoulders towards the hips, so the head is at negative ``v`` whatever the
    subject is doing.  Shared with skin/segment so every module measures in the
    same frame.
    """
    need = ("left_shoulder", "right_shoulder", "left_hip", "right_hip")
    pts = {}
    for name in need:
        p = lm.get(name)
        if p is None or p[3] < vis:
            return None
        pts[name] = p
    ls, rs = pts["left_shoulder"], pts["right_shoulder"]
    lh, rh = pts["left_hip"], pts["right_hip"]
    s = _mid(ls, rs)
    hip = _mid(lh, rh)
    torso = _dist(s, hip)
    if torso < MIN_TORSO_PX:
        return None
    dx, dy = ls[0] - rs[0], ls[1] - rs[1]
    n = math.hypot(dx, dy)
    if n < 1e-6:
        return None
    ux, uy = dx / n, dy / n
    vx, vy = -uy, ux
    if (hip[0] - s[0]) * vx + (hip[1] - s[1]) * vy < 0.0:
        ux, uy, vx, vy = -ux, -uy, -vx, -vy
    return {
        "S": s, "H": hip, "torso": torso,
        "u": (ux, uy), "v": (vx, vy),
        "shoulder_w": _dist(ls, rs), "hip_w": _dist(lh, rh),
        "roll_deg": math.degrees(math.atan2(uy, ux)),
        "ls": (ls[0], ls[1]), "rs": (rs[0], rs[1]),
        "lh": (lh[0], lh[1]), "rh": (rh[0], rh[1]),
        "z_shoulder": (ls[2], rs[2]), "z_hip": (lh[2], rh[2]),
        "vis_mean": (ls[3] + rs[3] + lh[3] + rh[3]) / 4.0,
    }


def _as_mask(mask, h: int, w: int):
    """Accept bool / 0-1 / 0-255 masks of any size, return uint8 0/255 HxW."""
    if mask is None or not isinstance(mask, np.ndarray) or mask.size == 0:
        return None
    m = mask
    if m.ndim == 3:
        m = m[:, :, 0]
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


def _scan_run(sil, cx: float, cy: float, ux: float, uy: float,
              max_r: float, gap: int) -> dict | None:
    """Length of the silhouette run crossing (cx, cy) along the u axis.

    Small holes up to ``gap`` pixels are bridged; a run that walks out of the
    frame while still on the subject is marked clipped, because a cropped body
    yields a width that is a lie.
    """
    h, w = sil.shape[:2]
    ir = int(max(4.0, max_r))

    def on(t: float):
        x = int(round(cx + ux * t))
        y = int(round(cy + uy * t))
        if x < 0 or y < 0 or x >= w or y >= h:
            return None
        return bool(sil[y, x])

    centre = 0.0
    if on(0.0) is not True:
        found = None
        for t in range(1, int(0.3 * ir) + 1):
            if on(float(t)) is True:
                found = float(t)
                break
            if on(float(-t)) is True:
                found = float(-t)
                break
        if found is None:
            return None
        centre = found

    reach = {1: 0.0, -1: 0.0}
    clipped = False
    for sign in (1, -1):
        miss = 0
        last = 0.0
        t = 0
        while t < ir:
            t += 1
            state = on(centre + sign * t)
            if state is None:
                clipped = clipped or miss == 0
                break
            if state:
                miss = 0
                last = float(t)
            else:
                miss += 1
                if miss > gap:
                    break
        reach[sign] = last
    return {"width": reach[1] + reach[-1] + 1.0, "centre": centre,
            "clipped": clipped, "left": reach[-1], "right": reach[1]}


def _expanded_hull(points, radius: float):
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


def _colour_silhouette(img, lm: dict, frame: dict):
    """Colour-model silhouette for when no segmentation mask was supplied.

    GrabCut seeded from the skeleton: the torso capsule is definite foreground,
    a grown convex hull of the landmarks is probable foreground, and the padded
    border of the crop is background only on the sides where padding actually
    fitted inside the image.  Returns None rather than a guess - a failed cut
    must not be dressed up as a measurement.
    """
    h, w = img.shape[:2]
    torso = frame["torso"]
    pts = [(p[0], p[1]) for p in lm.values() if p[3] >= 0.2]
    if len(pts) < 4:
        return None
    xs = [p[0] for p in pts]
    ys = [p[1] for p in pts]
    padx = 0.9 * torso
    pady_top = 1.3 * torso
    rx0, rx1 = int(min(xs) - padx), int(max(xs) + padx) + 1
    ry0, ry1 = int(min(ys) - pady_top), int(max(ys) + padx) + 1
    x0, x1 = max(0, rx0), min(w, rx1)
    y0, y1 = max(0, ry0), min(h, ry1)
    if x1 - x0 < 24 or y1 - y0 < 24:
        return None
    roi = np.ascontiguousarray(img[y0:y1, x0:x1])

    gc = np.full(roi.shape[:2], cv2.GC_PR_BGD, np.uint8)
    band = max(2, int(0.04 * min(roi.shape[0], roi.shape[1])))
    if ry0 >= 0:
        gc[:band, :] = cv2.GC_BGD
    if ry1 <= h:
        gc[-band:, :] = cv2.GC_BGD
    if rx0 >= 0:
        gc[:, :band] = cv2.GC_BGD
    if rx1 <= w:
        gc[:, -band:] = cv2.GC_BGD

    shifted = [(p[0] - x0, p[1] - y0) for p in pts]
    hull = _expanded_hull(shifted, 0.30 * torso)
    if hull is None:
        return None
    cv2.fillConvexPoly(gc, hull, cv2.GC_PR_FGD)

    quad = np.array([[frame["ls"][0] - x0, frame["ls"][1] - y0],
                     [frame["rs"][0] - x0, frame["rs"][1] - y0],
                     [frame["rh"][0] - x0, frame["rh"][1] - y0],
                     [frame["lh"][0] - x0, frame["lh"][1] - y0]], np.float32)
    centre = quad.mean(axis=0)
    core = ((quad - centre) * 0.6 + centre).astype(np.int32)
    cv2.fillConvexPoly(gc, core, cv2.GC_FGD)
    for chains in _LIMB_CHAINS.values():
        for chain in chains:
            for a, b in zip(chain, chain[1:]):
                pa, pb = lm.get(a), lm.get(b)
                if pa is None or pb is None or pa[3] < 0.5 or pb[3] < 0.5:
                    continue
                cv2.line(gc, (int(pa[0] - x0), int(pa[1] - y0)),
                         (int(pb[0] - x0), int(pb[1] - y0)),
                         int(cv2.GC_FGD), max(2, int(0.06 * torso)))

    if not (gc == cv2.GC_BGD).any() or not (gc == cv2.GC_FGD).any():
        return None
    try:
        bgd = np.zeros((1, 65), np.float64)
        fgd = np.zeros((1, 65), np.float64)
        cv2.grabCut(roi, gc, None, bgd, fgd, 3, cv2.GC_INIT_WITH_MASK)
    except Exception:                                 # noqa: BLE001
        return None

    cut = np.where((gc == cv2.GC_FGD) | (gc == cv2.GC_PR_FGD), 255, 0).astype(np.uint8)
    k = max(3, int(0.02 * torso) | 1)
    kern = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
    cut = cv2.morphologyEx(cut, cv2.MORPH_OPEN, kern)
    cut = cv2.morphologyEx(cut, cv2.MORPH_CLOSE, kern)
    cut = _largest_component(cut)
    if cut is None:
        return None
    fill = float(cut.mean()) / 255.0
    if fill < 0.03 or fill > 0.92:
        return None

    full = np.zeros((h, w), np.uint8)
    full[y0:y1, x0:x1] = cut
    mid = _mid(frame["S"], frame["H"])
    cx_i, cy_i = int(round(mid[0])), int(round(mid[1]))
    if not (0 <= cy_i < h and 0 <= cx_i < w) or full[cy_i, cx_i] == 0:
        return None
    return full


def _largest_component(mask):
    n, labels, stats, _ = cv2.connectedComponentsWithStats((mask > 0).astype(np.uint8), 8)
    if n <= 1:
        return None
    idx = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    return ((labels == idx).astype(np.uint8) * 255)


def _shot_type(lm: dict) -> str:
    def vis(name, t=0.4):
        p = lm.get(name)
        return p is not None and p[3] >= t

    feet = any(vis(n) for n in ("left_ankle", "right_ankle",
                                "left_foot_index", "right_foot_index"))
    hips = vis("left_hip") or vis("right_hip")
    head = any(vis(n) for n in ("nose", "left_eye", "right_eye"))
    if feet and hips:
        return "full"
    if hips:
        return "half"
    if head or vis("left_shoulder") or vis("right_shoulder"):
        return "closeup"
    return "unknown"


def _yaw_from_depth(dz: float, width_norm: float) -> float:
    """Depth split between a symmetric landmark pair -> rotation off-camera."""
    if width_norm <= 1e-6:
        return 0.0
    return _clamp(math.degrees(math.atan2(abs(dz), width_norm)), 0.0, 90.0)


def _chain_length(lm: dict, chain, vis: float) -> float | None:
    total = 0.0
    for a, b in zip(chain, chain[1:]):
        pa, pb = lm.get(a), lm.get(b)
        if pa is None or pb is None or pa[3] < vis or pb[3] < vis:
            return None
        total += _dist(pa, pb)
    return total if total > 1.0 else None


def _head_scan(sil, frame: dict, head_pt, gap: int):
    """Vertex-to-neck height and neck width, read off the silhouette.

    Deliberately silhouette-only: a landmark-based head estimate would use a
    different definition of "head", and mixing the two across a profile and a
    generated image would fabricate a proportion change out of nothing.
    """
    sx, sy = frame["S"]
    ux, uy = frame["u"]
    vx, vy = frame["v"]
    torso = frame["torso"]
    a0 = (head_pt[0] - sx) * ux + (head_pt[1] - sy) * uy

    h, w = sil.shape[:2]
    rows = []          # (b, width, clipped)
    misses = 0
    limit = max(6, int(0.03 * torso))
    top_xy = None
    for k in range(2, int(1.5 * torso) + 1):
        b = -float(k)
        cx = sx + a0 * ux + b * vx
        cy = sy + a0 * uy + b * vy
        run = _scan_run(sil, cx, cy, ux, uy, 1.3 * torso, gap)
        if run is None or run["width"] < 0.04 * torso:
            misses += 1
            if misses > limit:
                break
            continue
        misses = 0
        top_xy = (cx, cy)
        rows.append((b, run["width"], run["clipped"]))
    if len(rows) < max(8, int(0.12 * torso)) or top_xy is None:
        return None
    # A head that runs off the top of the frame measures short, and a short
    # head would read as a slimmed body once divided by the torso.
    if (top_xy[0] < 3 or top_xy[1] < 3 or top_xy[0] > w - 4 or top_xy[1] > h - 4):
        return None

    b_top = rows[-1][0]
    clean = [r for r in rows if not r[2]]
    neck_band = [r for r in clean if -0.45 * torso <= r[0] <= -0.04 * torso]
    if not neck_band:
        return None
    b_neck, neck_w = neck_band[0][0], neck_band[0][1]
    for b_i, wd, _c in neck_band:
        if wd < neck_w:
            b_neck, neck_w = b_i, wd
    head_band = [r for r in clean if r[0] < b_neck]
    if len(head_band) < 3:
        return None
    head_max = max(r[1] for r in head_band)
    if neck_w > 0.85 * head_max or neck_w > 0.75 * frame["shoulder_w"]:
        return None                          # no constriction: hair or scarf
    head_h = b_neck - b_top
    if not (0.25 * torso <= head_h <= 1.15 * torso):
        return None
    return {"head_h": head_h, "neck_w": neck_w,
            "b_top": b_top, "b_neck": b_neck}


def _limb_crosses(lm: dict, frame: dict, b_scan: float, half_w: float) -> bool:
    """True when an arm lies on the scanline and inflates the silhouette run."""
    sx, sy = frame["S"]
    ux, uy = frame["u"]
    vx, vy = frame["v"]
    torso = frame["torso"]
    for name in ("left_elbow", "right_elbow", "left_wrist", "right_wrist"):
        p = lm.get(name)
        if p is None or p[3] < 0.4:
            continue
        a = (p[0] - sx) * ux + (p[1] - sy) * uy
        b = (p[0] - sx) * vx + (p[1] - sy) * vy
        if abs(b - b_scan) < 0.20 * torso and abs(a) < half_w + 0.10 * torso:
            return True
    return False


# ------------------------------------------------------------------- public

def shape_profile(mask, n: int = SHAPE_SAMPLES) -> list:
    """Body width at n heights, measured against the silhouette's own height.

    This is the ruler the proportion check should be using, and the reason is
    arithmetic rather than taste.  Every other metric in this module divides by
    TORSO LENGTH, taken from two pose landmarks - and a difference of two jittery
    points is the noisiest denominator available.  Re-measuring one untouched
    photograph at different resolutions moves those ratios by 12.6% (shoulders)
    and 13.8% (hips), while a real 12% slim-down moves them 12.8%.  Signal and
    noise are the same size, so the gate is blind: it cannot tell a slimmed
    picture from a re-encoded one.

    The silhouette is a region, not a pair of points, so its extent barely moves:
    the same sweep gives 1.0% spread (3.0% at the worst height) against the same
    12.5% signal.  Twelve times the discrimination, from the same pixels.

    Returns [[t, width_over_height], ...] with t the fraction of the way down the
    figure, or [] when the mask cannot support the measurement.  Nothing from the
    pose enters here on purpose.
    """
    m = mask
    if m is None or not isinstance(m, np.ndarray) or m.size == 0:
        return []
    if m.ndim == 3:
        m = m[:, :, 0]
    body = m > 127 if m.dtype != np.bool_ else m
    rows = np.flatnonzero(body.any(axis=1))
    if rows.size < 20:
        return []
    top, bottom = int(rows[0]), int(rows[-1])
    height = float(bottom - top)
    if height < 20.0:
        return []

    out: list = []
    for index in range(1, int(n) + 1):
        y = int(top + height * index / (n + 1))
        if y < 0 or y >= body.shape[0]:
            continue
        cols = np.flatnonzero(body[y])
        if cols.size < 2:
            continue
        # Full extent at this height: the outline is what a slim-down moves.
        width = float(cols[-1] - cols[0])
        if width <= 0.0:
            continue
        out.append([round(index / (n + 1), 3), round(width / height, 5)])
    return out


def measure_body(img_bgr, pose: dict, mask=None) -> dict:
    """Measure one photograph. Every metric is a ratio against torso length."""
    out = {"ok": False, "shot_type": "unknown", "metrics": {}, "px": {},
           "confidence": 0.0, "reason": "", "unreliable": [],
           "reliability": {}, "corrected": [], "width_profile": [],
           "shape_profile": [],
           "yaw_estimate": 0.0}

    if not isinstance(img_bgr, np.ndarray) or img_bgr.ndim < 2 or img_bgr.size == 0:
        out["reason"] = "imagen invalida"
        return out
    h, w = img_bgr.shape[:2]
    img = img_bgr if img_bgr.ndim == 3 else cv2.cvtColor(img_bgr, cv2.COLOR_GRAY2BGR)

    if not isinstance(pose, dict) or not pose.get("ok"):
        out["reason"] = "sin pose detectada"
        return out
    lm = landmarks_px(pose, w, h)
    if len(lm) < 4:
        out["reason"] = "landmarks insuficientes"
        return out
    out["shot_type"] = _shot_type(lm)

    frame = torso_frame(lm)
    if frame is None:
        out["reason"] = "torso no medible: hombros o caderas no visibles"
        return out

    torso = frame["torso"]
    ux, uy = frame["u"]
    vx, vy = frame["v"]
    sx, sy = frame["S"]
    hx, hy = frame["H"]

    # Yaw from the depth split of the symmetric pairs.  A turned subject is
    # foreshortened by cos(yaw) on every horizontal measurement.
    yaw_sh = _yaw_from_depth(frame["z_shoulder"][0] - frame["z_shoulder"][1],
                             frame["shoulder_w"] / float(max(w, 1)))
    yaw_hp = _yaw_from_depth(frame["z_hip"][0] - frame["z_hip"][1],
                             frame["hip_w"] / float(max(w, 1)))
    yaw = max(yaw_sh, yaw_hp)
    out["yaw_estimate"] = round(yaw_sh, 2)

    sil = _as_mask(mask, h, w)
    sil_src = "provided"
    if sil is None:
        try:
            sil = _colour_silhouette(img, lm, frame)
        except Exception:                             # noqa: BLE001
            sil = None
        sil_src = "colour" if sil is not None else "none"
    elif not sil.any():
        sil = None
        sil_src = "none"

    metrics: dict = {}
    px: dict = {"image_w": int(w), "image_h": int(h),
                "torso_len": round(torso, 2),
                "shoulder_w": round(frame["shoulder_w"], 2),
                "hip_w": round(frame["hip_w"], 2),
                "roll_deg": round(frame["roll_deg"], 2),
                "yaw_shoulder_deg": round(yaw_sh, 2),
                "yaw_hip_deg": round(yaw_hp, 2),
                "silhouette": sil_src,
                "pose_backend": str(pose.get("backend") or "")}
    unreliable: list[str] = []
    notes: list[str] = []
    corrected: list[str] = []
    profile: list[list[float]] = []   # [t, width/torso] along the torso axis

    metrics["shoulder_w_over_torso"] = frame["shoulder_w"] / torso
    metrics["hip_w_over_torso"] = frame["hip_w"] / torso
    # Reported because the client asks for it, never gated on: a uniform
    # slim-down scales shoulders and hips by the same factor and leaves this
    # ratio identical.  Width over torso length is what actually moves.
    if frame["hip_w"] > 1.0:
        metrics["shoulder_over_hip"] = frame["shoulder_w"] / frame["hip_w"]

    # ---------------------------------------------------------- silhouette
    gap = max(1, int(0.012 * torso))
    if sil is not None:
        for name, t in (("bust_w_over_torso", BUST_T), ("waist_w_over_torso", WAIST_T)):
            cx = sx + (hx - sx) * t
            cy = sy + (hy - sy) * t
            run = _scan_run(sil, cx, cy, ux, uy, 1.3 * torso, gap)
            key = name.split("_w_")[0]
            if run is None:
                notes.append("silueta no cruzada en " + key)
                continue
            width = run["width"]
            if run["clipped"]:
                notes.append("cuerpo cortado por el borde en " + key)
                continue
            if width < 0.12 * torso or width > 2.5 * torso:
                notes.append("ancho implausible en " + key)
                continue
            b_scan = (cy - sy) * vy + (cx - sx) * vx
            if _limb_crosses(lm, frame, b_scan, width / 2.0):
                unreliable.append(name)
                notes.append("brazo sobre la linea de " + key)
            metrics[name] = width / torso
            px[key + "_w"] = round(width, 2)
            px[key + "_at"] = [round(cx, 1), round(cy, 1)]

        # The full silhouette profile, not just two heights.  Two scanlines are
        # two samples of a noisy measurement: whether the mask edge lands one
        # pixel in or out swings the ratio by several percent, which is the same
        # order as the slimming we are trying to catch.  Sampling the torso at
        # nine heights and comparing the whole profile against the source photo
        # averages that noise down and is what actually makes the comparison
        # sensitive enough to be worth running.
        for step in range(1, 10):
            t = step / 10.0
            cx = sx + (hx - sx) * t
            cy = sy + (hy - sy) * t
            run = _scan_run(sil, cx, cy, ux, uy, 1.3 * torso, gap)
            if run is None or run["clipped"]:
                continue
            width = run["width"]
            if width < 0.12 * torso or width > 2.5 * torso:
                continue
            b_scan = (cy - sy) * vy + (cx - sx) * vx
            limb = 1 if _limb_crosses(lm, frame, b_scan, width / 2.0) else 0
            # Scanlines crossed by an arm are kept but marked.  For an absolute
            # measurement an arm is contamination; for a comparison against the
            # same person in the same pose it is not, because the arm sits in
            # the same place in both images and cancels in the ratio.  Dropping
            # them outright left one or two usable samples on a woman standing
            # with her arms down - which is to say, on almost every photograph.
            profile.append([round(t, 2), round(width / torso, 4), limb])

        head_pt = None
        nose = lm.get("nose")
        if nose is not None and nose[3] >= 0.3:
            head_pt = (nose[0], nose[1])
        else:
            for pair in (("left_eye", "right_eye"), ("left_ear", "right_ear")):
                a, b = lm.get(pair[0]), lm.get(pair[1])
                if a is not None and b is not None and a[3] >= 0.3 and b[3] >= 0.3:
                    head_pt = _mid(a, b)
                    break
        if head_pt is not None:
            head = _head_scan(sil, frame, head_pt, gap)
            if head is not None:
                metrics["head_h_over_torso"] = head["head_h"] / torso
                metrics["neck_w_over_torso"] = head["neck_w"] / torso
                px["head_h"] = round(head["head_h"], 2)
                px["neck_w"] = round(head["neck_w"], 2)
            else:
                notes.append("cabeza no medible en la silueta")
    else:
        notes.append("sin silueta: cintura, busto, cuello y cabeza omitidos")

    # ------------------------------------------------------------- limbs
    # Length along the limb (shoulder->elbow->wrist), not the straight line:
    # a bent elbow would otherwise report a shorter arm than a straight one.
    for kind, chains in _LIMB_CHAINS.items():
        lens = [v for v in (_chain_length(lm, c, 0.45) for c in chains) if v]
        if not lens:
            continue
        value = sum(lens) / len(lens)
        metrics[kind + "_len_over_torso"] = value / torso
        px[kind + "_len"] = round(value, 2)
        px[kind + "_sides"] = len(lens)

    # ------------------------------------------------------- reliability
    # Every flag records WHY, because the two causes are not equivalent.  An arm
    # lying across the scanline contaminates that one width with something that
    # is not the body, and nothing can recover it.  A turned torso, by contrast,
    # foreshortens every width by a factor that is close to cos(yaw) - a known,
    # correctable geometry rather than lost information.  Discarding both alike
    # was measured (scripts/calibrate_identity.py) to disable the proportion
    # gate entirely on ordinary half-body photographs, which is where the
    # client's own pictures live.
    reasons: dict[str, str] = {}

    def _flag(name: str, why: str) -> None:
        if name in metrics and name not in unreliable:
            unreliable.append(name)
        reasons.setdefault(name, why)

    for name in list(unreliable):
        reasons.setdefault(name, "limb")

    if yaw > YAW_REJECT_DEG:
        for name in WIDTH_METRICS:
            metrics.pop(name, None)
        notes.append("sujeto de perfil: anchos descartados")
    elif yaw > YAW_SUSPECT_DEG:
        cos_yaw = math.cos(math.radians(min(yaw, YAW_REJECT_DEG)))
        for name in WIDTH_METRICS:
            if name not in metrics:
                continue
            if reasons.get(name) == "limb":
                continue                      # contaminated, not merely turned
            # Undo the foreshortening so the number is comparable with a band
            # measured from photographs taken at other angles.
            metrics[name] = metrics[name] / max(cos_yaw, 0.35)
            _flag(name, "yaw")
            corrected.append(name)
        notes.append("sujeto girado %.0f grados: anchos corregidos" % yaw)
    if yaw > 25.0:
        for name in ("arm_len_over_torso", "leg_len_over_torso"):
            _flag(name, "yaw_limb")
    if sil_src == "colour":
        for name in ("waist_w_over_torso", "bust_w_over_torso",
                     "neck_w_over_torso", "head_h_over_torso"):
            _flag(name, "silhouette")
        notes.append("silueta estimada por color")
    if out["shot_type"] != "full":
        metrics.pop("leg_len_over_torso", None)

    conf = _clamp(frame["vis_mean"], 0.3, 1.0)
    if sil is None:
        conf *= 0.55
    elif sil_src == "colour":
        conf *= 0.75
    if yaw > YAW_SUSPECT_DEG:
        conf *= max(0.30, math.cos(math.radians(min(yaw, 80.0))))
    if torso < 60.0:
        conf *= 0.75
    if unreliable:
        conf *= 0.85

    out["metrics"] = {k: round(float(v), 4) for k, v in metrics.items()
                      if math.isfinite(float(v))}
    out["px"] = px
    out["unreliable"] = [n for n in unreliable if n in out["metrics"]]
    out["reliability"] = {n: reasons.get(n, "unknown") for n in out["unreliable"]}
    # Widths whose foreshortening was undone: still approximations, but they are
    # comparable with the profile and may be gated with a wider tolerance.
    out["corrected"] = [n for n in corrected if n in out["metrics"]]
    out["width_profile"] = profile
    # The stable ruler: silhouette width against silhouette height, no landmark
    # anywhere in it.  Measured spread on an unchanged photo is 1.0% against a
    # 12.5% signal, where the landmark ratios manage 12.6% against 12.8%.
    out["shape_profile"] = shape_profile(sil if sil is not None else mask)
    out["ok"] = bool(out["metrics"])
    out["confidence"] = round(_clamp(conf, 0.0, 1.0), 3) if out["ok"] else 0.0
    out["reason"] = "; ".join(notes) if notes else ""
    if not out["ok"]:
        out["reason"] = out["reason"] or "ninguna metrica calculable"
    return out
