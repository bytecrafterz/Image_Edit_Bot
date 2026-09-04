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

# ------------------------------------------------------------ hand rulers
#
# Every number below is a ratio between two things measured in the SAME
# picture, on a crop scaled so the hand is always HAND_CANON_PX across.  That
# is not decoration: the previous version measured digit lengths in raw pixels
# and its verdict moved with the load size, which makes it unable to gate
# anything.  Measured 2026-09-04 on this client's own 24 photographs, the worst
# hand read severity 0.40 at max_side 1024 and 0.57 at 1600 (IMG_8825), and two
# others went the other way, 0.28 at 1024 down to 0.00 at 1600 (IMG_8798).
# Same pixels, same hand, three different answers.
#
# A hand is judgeable when it occupies enough of the FRAME - a fraction, which
# no resize can change - rather than enough pixels.  0.055 of the diagonal is
# where her own hands stop resolving fingers at all: her judgeable hands run
# 0.06 to 0.26 of the diagonal at every load size tested.
HAND_CANON_PX = 256.0
HAND_JUDGE_FRAC = 0.055

# The melted hand: the one failure this corpus actually contains, and the only
# one anything here can reject.  The fine grain of the hand skin is compared
# with the fine grain of the same arm in the same crop at the same zoom, so
# lens, camera, light and load size all divide out.
#
# Measured through this exact code path: 61 readings of her own hands (her 24
# photographs at max_side 768, 1024 and 1600, plus 10 re-encoded controls) never
# fall below 0.845, and their fifth percentile is 0.909.  Hands melted on
# purpose on copies of those same photographs reach 0.251, and 10 of the 44
# readable hands in the 26 generated images on disk sit below 0.60 - which is
# the client's complaint, measured.
#
# Reporting starts at 0.82, just under her worst real hand, and the severity
# only reaches the 0.60 that rejects an image at 0.62: 36% below anything her
# own photographs have ever produced, at any load size.  That margin is the
# instruction "do not make it oversensitive" written as a number.
HAND_MELT_REPORT = 0.82
HAND_MELT_SEVERE = 0.49

# Digit length over palm length in MediaPipe's metric 3D frame, which is what
# the 2D ratios should always have been: it is immune to foreshortening, where
# the 2D ones are not - her own untouched IMG_7880 measures a 2D palm aspect of
# 17.42 (7.9x outside the old bound) simply because the hand is edge-on, while
# its 3D reading is an unremarkable 1.77.  The bounds are human anatomy with
# roughly 40% of margin on her measured envelope (thumb 0.78-1.15, index
# 0.44-1.06, middle 0.60-1.19, ring 0.51-0.93, pinky 0.35-0.89), because the
# next user's hands are not hers and the envelope must not be fitted to one
# person.
HAND_DIGIT_BOUNDS_3D = {
    "thumb": (0.55, 1.60),
    "index": (0.32, 1.45),
    "middle": (0.42, 1.60),
    "ring": (0.34, 1.35),
    "pinky": (0.22, 1.25),
}
# Above this the 3D palm fit itself has collapsed - a palm cannot be five times
# longer than it is wide - so the reading is unusable rather than damning.  Her
# hands read 1.38-3.12; a generated hand a person can see is fine reads 6.29.
HAND_ASPECT_UNRELIABLE = 4.6

# A wrist with no recognised hand on it is only a missing hand when there is
# nothing there.  Measured on her photographs, all 7 of the wrists that used to
# be reported as "missing" have 0.75-1.00 skin in the disc just beyond them:
# the hand is in the picture, the model simply did not name it.
LIMB_SKIN_PRESENT = 0.35

# How much of the forearm patch has to actually be arm before the grain ruler
# above is allowed to divide by it.
#
# The melt ruler is hand grain over ARM grain, and it was proved invariant to
# the load size.  It is not invariant to the FRAME: the arm patch is placed
# from the pose landmarks, and the pose is fitted to whatever shape of picture
# arrives.  Her own IMG_8918, untouched and merely cropped at the sides to 72%
# of its width - her whole body still in frame, not one pixel of her hand or
# arm changed - moved that patch off her forearm.  Skin inside the disc fell
# from 88% to 38%, the grain it measured was a lace bodice and a wall instead
# of her arm, the ratio fell from 1.85 to 0.47, and the image was REJECTED with
# "la mano conserva el 47% de la textura fina del brazo: dedos fundidos o
# borrados" at severity 0.95, on a hand that is perfect.
#
# Over 161 hand readings - her 24 photographs under five framings, plus the
# whole broken-hand truth set - only six patches fall under 60% skin, and four
# of those are 0% (no patch at all, already reported as unjudged).  The two
# real ones are both that reframe of IMG_8918, at 0.385 and 0.577; the next
# reading up anywhere in the corpus is 0.666.  The limit sits in that gap: 56%
# above the reading that produced the false rejection and 10% below the
# faintest honest one.
#
# It costs almost nothing that was working.  Every melted hand the ruler
# rejects - 6 of 15 in the truth set - is measured against a patch over 80%
# skin, so the guard removes 0 of those at any limit up to 0.80, and on the
# paid images on disk it changes exactly one verdict.
#
# That one verdict is the reason this guard matters most, not an argument
# against it.  data/final/nayane_final.jpg was being REJECTED at severity 0.95
# for "la mano conserva el 10% de la textura fina del brazo".  In that image
# she is wearing a long sleeved shirt buttoned at the cuff, and the patch the
# ruler divided by contains 1.1% and 0.0% skin: it sits on the sleeve.  That
# number was hand grain over WOVEN COTTON grain, and cotton at that scale
# carries far more fine energy than skin, so the ratio had to come out low
# whatever the hand looked like.  It was never a measurement of her hand.
#
# Nothing in the calibration corpus could show this: all 24 of her photographs
# are bare armed, and the broken-hand set is built on them.  Half the wardrobe
# is not - blazer, camisa, abrigo, gabardina, chaqueta de cuero, jersey, traje
# and esmoquin all cover the forearm - so without this guard every long sleeved
# outfit she orders is a hand the ruler calls melted.
#
# A hand whose reference cannot be trusted is now reported as unjudged, and the
# verdict tells her which hands nobody checked and to look at them herself.
# That is the honest answer, and it is the one the check already knows how to
# give.
ARM_PATCH_SKIN_MIN = 0.60

# ------------------------------------------------------------ face rulers
#
# A face turned away from the camera is asymmetric in projection, and the old
# code handled that with an on/off guard at balance 0.78.  Her IMG_8949 lands on
# that edge: balance 0.739 at max_side 768, 0.782 at 1024, 0.692 at 1600, so at
# 1024 alone it was called "rasgos asimetricos" at severity 0.90 - a rejection,
# on her own photograph, decided by the resize.  The guard is now a slope
# instead of a step: the asymmetry a turn explains grows with the turn.  Over
# her 24 photographs at three sizes the worst honest case needs a slope of 1.30
# (asym 0.383 at turn 0.218); the slope used is 2.2, and for the vertical drift
# 0.60 against a worst honest 0.274.
FACE_ASYM_BASE = 0.10
FACE_ASYM_TURN = 2.2
FACE_DRIFT_BASE = 0.055
FACE_DRIFT_TURN = 0.60
# Past this much turn the projection dominates and nothing here can be read.
FACE_BALANCE_MIN = 0.35
# One eye is legitimately smaller than the other the moment the head turns:
# hers read down to 0.280 when turned and never below 0.662 when near frontal,
# so the eye check only speaks about a near-frontal face.
FACE_EYE_FRONTAL = 0.85
FACE_EYE_RATIO = 0.55

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
# The views one hand is measured from.  A landmark set that survives a mirror,
# a small rotation and a change of scale is one the pixels support; one that
# does not is a guess, and a guess must not reject a paid image.

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

def _world_ratios(world) -> dict | None:
    """Digit length over palm length in the metric 3D frame, plus palm aspect.

    This is the same question the old pixel geometry asked, asked where the
    answer means something.  Measured on her 24 photographs at three load
    sizes, one hand's 2D digit ratios ranged over a factor of eight between
    sizes (IMG_8918 read 2.28 at 768 and 8.51 at 1024 for the same index
    finger) while its 3D ratios moved by 0.1 or less.
    """
    if world is None or len(world) < 21:
        return None
    palm = _dist3(world[0], world[9])
    if palm < 1e-9:
        return None
    out = {}
    for name, chain in _DIGITS.items():
        out[name] = sum(_dist3(world[a], world[b])
                        for a, b in zip(chain, chain[1:])) / palm
    out["aspect"] = palm / max(_dist3(world[5], world[17]), 1e-9)
    return out


def _dist3(a, b) -> float:
    return math.sqrt((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2 + (a[2] - b[2]) ** 2)


def _hand_views(crop):
    """One hand seen six ways, with the map that puts each back in the crop."""
    h, w = crop.shape[:2]
    out = [(crop, lambda p: p)]
    out.append((cv2.flip(crop, 1),
                lambda p: np.stack([w - 1 - p[:, 0], p[:, 1]], 1)))
    for ang in (-12.0, 12.0):
        M = cv2.getRotationMatrix2D((w / 2.0, h / 2.0), ang, 1.0)
        turned = cv2.warpAffine(crop, M, (w, h), flags=cv2.INTER_CUBIC,
                                borderMode=cv2.BORDER_REPLICATE)
        Mi = cv2.invertAffineTransform(M)
        out.append((turned,
                    lambda p, Mi=Mi: np.c_[p, np.ones(len(p))] @ Mi.T))
    for zoom in (0.75, 1.3):
        sized = cv2.resize(crop, None, fx=zoom, fy=zoom,
                           interpolation=cv2.INTER_AREA if zoom < 1
                           else cv2.INTER_CUBIC)
        out.append((sized, lambda p, zoom=zoom: p / zoom))
    return out


def _fine_energy(gray, mask) -> float | None:
    """Amplitude of the pore-and-grain band inside a mask."""
    if mask is None or int(np.count_nonzero(mask)) < 64:
        return None
    band = np.abs(gray - cv2.GaussianBlur(gray, (0, 0), 1.4))
    sel = band[mask > 0]
    return float(np.median(sel)) if sel.size >= 64 else None


def _canonical_hand(img_bgr, pts, box, detector, forearm) -> dict:
    """Measure one hand at a fixed size, from six views, against its own arm.

    Returns what the picture supports rather than what one detection claimed:
    the 3D digit ratios agreed on across the views, how far the landmarks moved
    between them, and the grain of the hand skin over the grain of the forearm
    in the same crop.
    """
    out = {"n_views": 0, "g3": None, "spread": None, "melt": None, "zoom": 0.0}
    h, w = img_bgr.shape[:2]
    span = max(box[2], box[3])
    pad = 0.5 * span
    x0 = int(_clamp(box[0] - pad, 0, w - 2))
    y0 = int(_clamp(box[1] - pad, 0, h - 2))
    x1 = int(_clamp(box[0] + box[2] + pad, x0 + 2, w))
    y1 = int(_clamp(box[1] + box[3] + pad, y0 + 2, h))
    crop = img_bgr[y0:y1, x0:x1]
    if crop.size == 0 or min(crop.shape[:2]) < 16:
        return out
    zoom = float(_clamp(HAND_CANON_PX / max(span, 1e-6), 0.25, 8.0))
    big = np.ascontiguousarray(cv2.resize(
        crop, None, fx=zoom, fy=zoom,
        interpolation=cv2.INTER_CUBIC if zoom > 1 else cv2.INTER_AREA))
    out["zoom"] = round(zoom, 2)

    seen, ratios = [], []
    for view, inverse in _hand_views(big):
        found = _read_hands(detector, view, 0.0, 0.0)
        if not found:
            continue
        best = max(found, key=lambda f: f["score"])
        seen.append(inverse(np.asarray(best["pts"], np.float64)))
        got = _world_ratios(best.get("world"))
        if got:
            ratios.append(got)
    out["n_views"] = len(seen)
    if len(seen) >= 2:
        stack = np.stack(seen)
        median = np.median(stack, axis=0)
        reach = max(float(np.ptp(median[:, 0])), float(np.ptp(median[:, 1])), 1e-6)
        out["spread"] = round(float(np.median(
            np.linalg.norm(stack - median[None], axis=2) / reach)), 4)
    if ratios:
        out["g3"] = {k: float(np.median([r[k] for r in ratios]))
                     for k in ratios[0]}

    # Grain of the hand against grain of the arm it is attached to.
    local = (np.asarray(pts, np.float64) - np.array([x0, y0])) * zoom
    gray = cv2.cvtColor(big, cv2.COLOR_BGR2GRAY).astype(np.float32)
    skin = skin_mask_ycrcb(big)
    hull = np.zeros(big.shape[:2], np.uint8)
    cv2.fillConvexPoly(hull, cv2.convexHull(local.astype(np.int32)), 255)
    hand_e = _fine_energy(gray, cv2.bitwise_and(hull, skin))
    arm_e = None
    if forearm is not None:
        fx = (forearm[0] - x0) * zoom
        fy = (forearm[1] - y0) * zoom
        disc = np.zeros(big.shape[:2], np.uint8)
        cv2.circle(disc, (int(fx), int(fy)), int(0.25 * HAND_CANON_PX), 255, -1)
        on_arm = cv2.bitwise_and(disc, skin)
        area = float(cv2.countNonZero(disc))
        # The denominator has to BE an arm.  When the pose puts this disc on a
        # bodice or on the wall, the grain measured there is not her arm's and
        # the ratio is not about her hand - see ARM_PATCH_SKIN_MIN.
        covered = (float(cv2.countNonZero(on_arm)) / area) if area > 0 else 0.0
        out["arm_skin"] = round(covered, 3)
        if covered >= ARM_PATCH_SKIN_MIN:
            arm_e = _fine_energy(gray, on_arm)
    if hand_e is not None and arm_e is not None and arm_e > 1e-6:
        out["melt"] = round(hand_e / arm_e, 3)
    return out


def _hand_side(hand: dict) -> str:
    label = str(hand.get("label") or "")
    return "left_hand" if label.startswith("left") else (
        "right_hand" if label.startswith("right") else "hand")


def _nearest_arm(hand, wrists: dict):
    """The forearm patch belonging to this hand, or None if the pose lost it."""
    best, gap = None, 1e18
    for info in wrists.values():
        d = _dist(hand["pts"][0], info["pt"])
        if d < gap:
            gap, best = d, info
    return best["arm"] if best else None


def _judge_hands(img_bgr, hands, wrists, torso, h, w, detector,
                 unjudged) -> list[dict]:
    """One verdict per hand, from rulers that cannot move with the load size."""
    defects: list[dict] = []
    # A hand running off the edge of the frame is the commonest thing in a
    # phone selfie - the hand holding the phone - and it is unjudgeable, not
    # deformed: the fingers that would make the geometry add up are simply not
    # in the picture.  Treating those as defects produced six false "mano
    # deformada" verdicts at severity up to 0.95 on her own photographs.
    edge_x, edge_y = 0.02 * w, 0.02 * h
    diag = math.hypot(w, h)
    for hand in hands:
        pts = hand["pts"]
        box = _box_from_points(pts, 0.12 * max(_dist(pts[0], pts[9]), 8.0), w, h)
        if not box or min(box[2], box[3]) < 12:
            continue
        xs = [p[0] for p in pts]
        ys = [p[1] for p in pts]
        truncated = (min(xs) <= edge_x or max(xs) >= w - edge_x
                     or min(ys) <= edge_y or max(ys) >= h - edge_y)
        got = _canonical_hand(img_bgr, pts, box, detector,
                              _nearest_arm(hand, wrists))
        where = _hand_side(hand)

        # The grain ruler runs first and runs on every hand, whatever the model
        # made of it.  It has to: a hand melted badly enough stops being
        # re-findable at all, so anything ordered behind a "could we measure
        # this" test would let exactly the worst hands through - which is what
        # the first cut of this file did, reporting the melted hands of
        # run_ecbc8543 as merely unjudged.  It needs no landmarks, only the box
        # they gave and a patch of the arm.
        melt = got["melt"]
        if melt is None:
            _note_unjudged(unjudged, hand, box,
                           "sin piel del brazo con la que comparar la textura")
        elif melt < HAND_MELT_REPORT:
            reach = max(HAND_MELT_REPORT - HAND_MELT_SEVERE, 1e-6)
            sev = (HAND_MELT_REPORT - melt) / reach
            defects.append(_defect(
                "hand_malformed", where, box, _clamp(sev, 0.15, 0.95), True,
                "la mano conserva el %d%% de la textura fina del brazo: dedos "
                "fundidos o borrados" % int(round(100.0 * melt))))

        # Everything from here down counts fingers, and counting fingers needs
        # the whole hand, resolved.  Being cut by the edge of the frame stops
        # only that: the hand holding the phone in a selfie is not deformed, it
        # is half absent, and treating it as deformed produced six false "mano
        # deformada" verdicts on her own photographs.  The grain ruler above
        # does not care, and must not be put behind this test - the worst hand
        # in the whole corpus, the one delivered in nayane_final.jpg with 10% of
        # the grain of its own arm, is a hand that touches the bottom edge.
        if truncated:
            _note_unjudged(unjudged, hand, box, "sale del encuadre")
            continue

        # Finger geometry also needs fingers that are actually resolved.  How
        # much of the frame the hand occupies survives any resize; how many
        # pixels it happens to have does not, which is the whole difference
        # between this check and the pixel floor it replaces.
        frac = max(box[2], box[3]) / max(diag, 1e-6)
        if frac < HAND_JUDGE_FRAC:
            _note_unjudged(unjudged, hand, box,
                           "demasiado pequena en el encuadre para medirle los "
                           "dedos (%.1f%% de la imagen, hace falta %.1f%%)"
                           % (100.0 * frac, 100.0 * HAND_JUDGE_FRAC))
            continue
        if got["n_views"] < 2:
            # The pixels stop supporting the skeleton as soon as the crop is
            # mirrored or turned a little.  That happens on a melted hand - the
            # grain ruler above has already had its say - and it also happens on
            # her own closed fists (IMG_8918 at all three sizes), so here it
            # names the hand instead of condemning it.
            _note_unjudged(unjudged, hand, box,
                           "no se pudo volver a medir en primer plano")
            continue

        g3 = got["g3"]
        if not g3:
            _note_unjudged(unjudged, hand, box, "sin geometria 3D medible")
        elif g3.get("aspect", 0.0) > HAND_ASPECT_UNRELIABLE:
            _note_unjudged(unjudged, hand, box,
                           "la palma no se pudo reconstruir en 3D")
        else:
            worst, name, value = 0.0, "", 0.0
            for digit, (lo, hi) in HAND_DIGIT_BOUNDS_3D.items():
                got_v = g3.get(digit)
                if got_v is None or got_v <= 1e-6:
                    continue
                excess = max(got_v / hi - 1.0, lo / got_v - 1.0)
                if excess > worst:
                    worst, name, value = excess, digit, got_v
            if worst > 0.0:
                defects.append(_defect(
                    "hand_malformed", where, box,
                    _clamp(0.30 + 1.6 * worst, 0.30, 0.90), True,
                    "el dedo %s mide %.2f veces la palma, fuera de lo que "
                    "puede medir una mano" % (name, value)))
    return defects


def _note_unjudged(sink: list | None, hand: dict, box, why: str) -> None:
    """Record a hand the geometry could not honestly judge."""
    if sink is None:
        return
    label = str(hand.get("label") or "")
    where = "left_hand" if label.startswith("left") else (
        "right_hand" if label.startswith("right") else "hand")
    sink.append({"where": where, "reason": why,
                 "bbox": [int(round(t)) for t in box] if box else []})


def _read_hands(detector, img_bgr, ox: float, oy: float) -> list[dict]:
    h, w = img_bgr.shape[:2]
    rgb = np.ascontiguousarray(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB))
    res = detector.process(rgb)
    out = []
    lms = getattr(res, "multi_hand_landmarks", None) or []
    worlds = getattr(res, "multi_hand_world_landmarks", None) or []
    handed = getattr(res, "multi_handedness", None) or []
    for i, hand in enumerate(lms):
        pts = [(ox + p.x * w, oy + p.y * h) for p in hand.landmark]
        if len(pts) < 21:
            continue
        world = None
        if i < len(worlds):
            world = [(p.x, p.y, p.z) for p in worlds[i].landmark]
        label, score = "", 0.0
        if i < len(handed):
            try:
                cls = handed[i].classification[0]
                label, score = str(cls.label).lower(), float(cls.score)
            except Exception:                         # noqa: BLE001
                pass
        cx = sum(p[0] for p in pts) / len(pts)
        cy = sum(p[1] for p in pts) / len(pts)
        out.append({"pts": pts, "label": label, "score": score,
                    "centre": (cx, cy), "world": world})
    return out


def _hand_defects(img_bgr, lm: dict, frame, h: int, w: int,
                  unjudged: list | None = None) -> list[dict]:
    """Hands that are wrong, plus - in ``unjudged`` - hands nobody could judge.

    What this can and cannot do, measured rather than assumed.  The truth set
    was her own 24 photographs (56 hand readings at max_side 768, 1024 and
    1600, plus 10 re-encoded controls), against 37 hands broken on purpose on
    copies of those same photographs - a finger erased, a sixth finger grafted
    on, a finger stretched to 1.9x, a finger bent 70 degrees at the joint, the
    fingers melted into a mitten - every one of them looked at at high zoom
    before it was called broken, and the ones whose damage landed on a finger
    that was hidden anyway were thrown out rather than labelled.

    THE MELTED HAND IS CAUGHT.  Hand grain over arm grain never drops below
    0.880 on a real hand and reaches 0.208 on a melted one; that is the ruler
    that rejects, and it needs 0.52 to do it.

    A SIXTH FINGER, A MISSING FINGER AND A BENT FINGER ARE NOT CAUGHT, and no
    threshold here can be moved to catch them.  MediaPipe fits a plausible
    skeleton over the damage and reports it with confidence 0.99: a hand with
    a finger grafted onto it reads 3D digit ratios of 0.87/0.91/0.84 against
    her own 0.82/0.91/0.82.  It is not a resolution problem - the same test at
    native 3088 px, where the hands are 96-320 px wide, gives the same answer.
    Landmark spread across six views of the same crop does not separate them
    either (her hands 0.004-0.054, broken hands 0.003-0.033, complete overlap),
    nor does a silhouette fingertip count, which cannot even find the hand
    outline: the arm, the thigh and a beige wall are all the same colour.

    So those hands are reported and not rejected, which is exactly the client's
    instruction: do not be oversensitive, do not throw away paid images.  What
    must never happen again is the silence - the second paid image of the day
    was delivered under "Sin anomalias anatomicas detectadas" with both hands
    melted - so a hand nobody could measure leaves its name in ``unjudged``.
    """
    if mp is None:
        return []
    torso = frame["torso"] if frame else 0.12 * max(h, w)
    wrists = {}
    for side in ("left", "right"):
        p = lm.get(side + "_wrist")
        if p is not None and p[3] >= 0.5:
            el = lm.get(side + "_elbow")
            forearm = _dist((p[0], p[1]), (el[0], el[1])) if el and el[3] >= 0.4 else 0.0
            towards = None
            if el is not None and el[3] >= 0.4 and forearm > 8.0:
                # a patch of the arm, close enough to the hand to share its
                # light and its focus, which is what the grain is compared to
                towards = (0.65 * p[0] + 0.35 * el[0], 0.65 * p[1] + 0.35 * el[1])
            wrists[side] = {"pt": (p[0], p[1]), "arm": towards,
                            "elbow": (el[0], el[1]) if el is not None and el[3] >= 0.4
                            else None,
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

            defects.extend(_judge_hands(img_bgr, hands, wrists, torso, h, w,
                                        detector, unjudged))
    except Exception:                                 # noqa: BLE001
        return []

    # More hands than a person has means a limb was invented.  The comparison
    # used to be against the number of WRISTS the pose reported, and that is a
    # visibility count, not an anatomy count: her own IMG_8841, cropped at the
    # sides to 72% of its width, gave the pose one confident wrist while
    # MediaPipe found both of her hands, and the image was rejected for
    # "extremidad de mas" at severity 0.75 with both hands present and correct.
    # Two hands can never be one too many, so the floor is two.  Over 456
    # verifications of her own photographs this rule fired exactly once, on
    # that crop, and it fires zero times on the 144 readings of the broken-hand
    # truth set - so raising the floor loses nothing that ever worked.
    if wrists and len(hands) > max(len(wrists), 2):
        extra = max(hands, key=lambda hd: min(_dist(hd["pts"][0], i["pt"])
                                              for i in wrists.values()))
        box = _box_from_points(extra["pts"], 0.15 * torso, w, h)
        defects.append(_defect(
            "extra_limb", "hands", box, 0.75, True,
            "se detectaron %d manos para %d munecas" % (len(hands), len(wrists))))

    # A wrist well inside the frame with nothing on the end of it - and
    # "nothing" has to be checked, not assumed.  All seven of the wrists this
    # used to report on her own photographs have skin right where the hand
    # should be (0.75 to 1.00 of the disc beyond the wrist): the hand is in the
    # picture, behind her back or in a fist the model would not name.  Calling
    # that a missing limb is a false statement in a report the client reads, so
    # the skin is looked at first, and when it is there the hand is listed as
    # one nobody could check.
    margin_x, margin_y = 0.06 * w, 0.06 * h
    skin = None
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

        centre = pt
        elbow = info.get("elbow")
        if elbow is not None:
            dx, dy = pt[0] - elbow[0], pt[1] - elbow[1]
            reach = math.hypot(dx, dy)
            if reach > 1e-6:
                centre = (pt[0] + 0.55 * info["forearm"] * dx / reach,
                          pt[1] + 0.55 * info["forearm"] * dy / reach)
        if skin is None:
            skin = skin_mask_ycrcb(img_bgr)
        disc = np.zeros((h, w), np.uint8)
        cv2.circle(disc, (int(centre[0]), int(centre[1])),
                   int(max(6.0, 0.45 * info["forearm"])), 255, -1)
        area = float(cv2.countNonZero(disc))
        covered = (float(cv2.countNonZero(cv2.bitwise_and(disc, skin))) / area
                   if area > 0 else 0.0)
        if covered >= LIMB_SKIN_PRESENT:
            if unjudged is not None:
                unjudged.append({"where": side + "_hand",
                                 "reason": "hay mano pero no se pudo reconocer",
                                 "bbox": [int(round(t)) for t in box]})
            continue
        defects.append(_defect(
            "missing_limb", side + "_hand", box, 0.35, True,
            "muneca " + side + " visible y sin mano: solo el %d%% del sitio "
            "donde deberia estar tiene piel" % int(round(100.0 * covered))))
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

    # Yaw makes a healthy face asymmetric in projection.  How much turn there
    # is comes from the mesh itself, and the asymmetry a turn is allowed to
    # explain grows with it - see FACE_ASYM_TURN for why this is a slope and no
    # longer a step at balance 0.78.
    half_a = abs(midline - rot(pts[33])[0])
    half_b = abs(rot(pts[263])[0] - midline)
    balance = min(half_a, half_b) / max(half_a, half_b, 1e-6)
    turn = _clamp(1.0 - balance, 0.0, 1.0)

    defects: list[dict] = []
    box = _box_from_points(list(pts.values()), 0.10 * iod, w, h)
    if balance < FACE_BALANCE_MIN:
        return defects

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
        asym_max = FACE_ASYM_BASE + FACE_ASYM_TURN * turn
        drift_max = FACE_DRIFT_BASE + FACE_DRIFT_TURN * turn
        if asym > asym_max or drift > drift_max:
            sev = _clamp(max((asym - asym_max) / 0.12, (drift - drift_max) / 0.06),
                         0.25, 0.9)
            defects.append(_defect(
                "face_distorted", "face", box, sev, True,
                "rasgos asimetricos respecto a la linea media (desvio %.3f "
                "sobre %.3f, deriva %.3f sobre %.3f)"
                % (asym, asym_max, drift, drift_max)))

    # One eye is smaller than the other the moment the head turns, so this only
    # speaks about a face that is looking at the camera.
    if balance >= FACE_EYE_FRONTAL:
        area_a = _poly_area([rot(p) for p in eye_a])
        area_b = _poly_area([rot(p) for p in eye_b])
        if min(area_a, area_b) > 4.0:
            ratio = min(area_a, area_b) / max(area_a, area_b)
            if ratio < FACE_EYE_RATIO:
                eye_box = _box_from_points(eye_a + eye_b, 0.25 * iod, w, h)
                defects.append(_defect(
                    "eye_asymmetry", "eyes", eye_box,
                    _clamp((FACE_EYE_RATIO - ratio) / 0.5, 0.2, 0.85), True,
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
    # This one is information, not a verdict.  It fires on 4 of her 24 untouched
    # photographs at correlation 0.97-0.98 - a tiled floor, a repeated lace
    # pattern - and not once in this corpus on anything a person would call a
    # duplicated feature, so it is pinned below the severity that can reject an
    # image however sure the correlation looks.
    inv = 1.0 / scale
    x0 = min(coords[i][0], coords[j][0]) * inv
    y0 = min(coords[i][1], coords[j][1]) * inv
    x1 = (max(coords[i][0], coords[j][0]) + size) * inv
    y1 = (max(coords[i][1], coords[j][1]) + size) * inv
    box = [coords[j][0] * inv, coords[j][1] * inv, size * inv, size * inv]
    detail = ("dos zonas casi identicas (correlacion %.2f) separadas %d px; "
              "suele ser una textura repetida" % (score,
                                                  int(math.hypot(x1 - x0, y1 - y0))))
    return [_defect("duplicated_feature", "body", box, 0.30, True, detail)]


# ------------------------------------------------------------------- public

def scan_anomalies(img_bgr, pose: dict, face: dict, masks: dict) -> dict:
    """Full anatomical and texture audit of one generated image."""
    out = {"ok": False, "defects": [], "score": 0.0, "reason": "", "checks": [],
           "unjudged": []}
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
    unjudged: list[dict] = []
    jobs = [
        ("hands", lambda: _hand_defects(img_bgr, lm, frame, h, w, unjudged)),
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

    # What was found, and - just as important for a client who is paying for
    # these - what nobody could look at.  See _hand_defects.  One hand is one
    # entry: a named side reaches this list from two places, the hand pass and
    # the wrist pass, and telling her "no se han podido comprobar 3 manos"
    # about a photograph of one person is the kind of sentence that makes her
    # stop believing the rest of the report.
    seen_sides: set[str] = set()
    kept = []
    for note in unjudged:
        side = str(note.get("where") or "")
        if side and side != "hand":
            if side in seen_sides:
                continue
            seen_sides.add(side)
        kept.append(note)
    out["unjudged"] = kept
    defects.sort(key=lambda d: -float(d.get("severity", 0.0)))
    penalty = sum(SEVERITY_WEIGHT.get(d["type"], 0.5) * float(d["severity"])
                  for d in defects)
    out["defects"] = defects
    out["score"] = round(_clamp(1.0 - penalty, 0.0, 1.0), 3)
    out["ok"] = True
    if person is None:
        out["reason"] = "sin silueta: textura y borde no evaluados"
    return out
