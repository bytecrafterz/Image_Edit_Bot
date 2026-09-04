"""Identity profile builder.

The client's complaint about her previous tool was that it silently made her
slimmer and altered her face.  Answering that complaint needs numbers, not a
sentence inside a prompt: this module turns a handful of real photographs into
a measured description of one person - a face signature, a per-metric band for
every body proportion, skin and hair colour, and the marks that must survive
every edit.  Aggregation is deliberately robust (median / MAD trimming) because
one bad frame - a filtered selfie, an odd angle - must not move the bands.

Once the profile exists the originals may be deleted: the numbers here are what
``identity/verify.py`` compares every generated image against.
"""
from __future__ import annotations

import math
from collections import Counter
from typing import Any, Sequence

import cv2
import numpy as np

from ..analysis import body as body_mod
from ..analysis import face as face_mod
from ..analysis import loader
from ..analysis import pose as pose_mod
from ..analysis import quality as quality_mod
from ..analysis import segment as segment_mod
from ..analysis import shot as shot_mod
from ..analysis import skin as skin_mod
from . import embedding as embedding_mod

# ------------------------------------------------------------------- tuning

ANALYSIS_MAX_SIDE = 1600      # metrics are ratios, so a downscale costs nothing
QUALITY_FLOOR = 0.25          # below this a photo cannot support a measurement
BODY_CONF_MIN = 0.5           # measure_body confidence needed to feed a band
DROP_FRACTION = 0.2           # share of outlier samples dropped before averaging
BAND_MAX_REL = 0.12           # widest useful band: +/-12% of the measured mean
GATE_MIN_SAMPLES = 3          # usable photos needed before a metric may reject

DEFAULT_THRESHOLDS: dict[str, float] = {
    # Kept because stored profiles carry it and the report page prints it, but
    # nothing gates on it any more: the geometric descriptor it belongs to
    # cannot separate her from anybody (0.9832..0.9993 on her own photographs
    # against 0.9577..0.9945 on eight other women, so 0.72 sits a quarter of the
    # scale below both populations and can never fire).
    "face_min": 0.72,
    # The line the identity check actually reads: cosine between the SFace
    # embedding of a result and the mean of her own photographs.  Calibrated,
    # not chosen - measured on this profile with the photographs held out of
    # the profile they are scored against:
    #
    #   her 24 photographs, leave-one-out   0.6362 .. 0.8785
    #   8 photographs of 8 other women      0.0408 .. 0.1962
    #   the 2 paid results she rejected     0.1867 and 0.2886
    #
    # Every value in (0.2886, 0.6362] separates the two perfectly, and the next
    # generated images above the negatives sit at 0.2943, 0.3262, 0.3682 and
    # 0.3990 - four more paid results whose faces are visibly not hers - with
    # nothing at all between 0.3990 and 0.5305.  0.45 lands in the middle of
    # that empty band: 0.19 of room below her worst photograph, which is what
    # pays for a faithful generation being noisier than a photograph, and 0.05
    # above the worst face the check has to reject.
    "face_embed_min": 0.45,
    # Skin is gated on chroma, not on raw CIE76 distance: exposure moves L by
    # far more than a real change of skin tone moves a and b.  The effective
    # limit is widened per person from their own measured spread, so a woman
    # photographed across several lighting sessions is not accused of having
    # changed colour.  See _skin_tolerance().
    "delta_e_max": 8.0,
    "chroma_max": 6.0,
    "chroma_sigma": 2.0,
    "delta_l_max": 22.0,
    "metric_tol_sigma": 2.5,
    "metric_tol_floor": 0.06,
    "metric_band_max_rel": BAND_MAX_REL,
    "metric_gate_min_n": GATE_MIN_SAMPLES,
}

# Metrics that can make a generated image fail.  The rest are reported only:
# gating a ratio that is noisy by nature would reject perfectly good images.
GATED_METRICS: tuple[str, ...] = (
    "shoulder_w_over_torso",
    "hip_w_over_torso",
    "waist_w_over_torso",
    "bust_w_over_torso",
    "head_h_over_torso",
)

METRIC_ORDER: tuple[str, ...] = (
    "shoulder_w_over_torso", "hip_w_over_torso", "waist_w_over_torso",
    "bust_w_over_torso", "head_h_over_torso", "neck_w_over_torso",
    "arm_len_over_torso", "leg_len_over_torso", "shoulder_over_hip",
)

# Mark detection
MARK_DEV_SIGMA = 3.0          # Lab distance from the local skin distribution
MARK_DELTA_E = 10.0           # region vs surrounding skin ring, CIE76
MARK_ENERGY_PCT = 80.0        # texture energy percentile inside skin
MARK_MATCH_DIST = 0.12        # centre distance (person box units) for "same mark"
MAX_MARKS_PER_PHOTO = 8

HAIR_SHORT_MAX = 0.15         # hair bottom below the chin, in torso lengths
HAIR_LAB_TOL = 26.0           # Lab distance still counted as the same hair
HAIR_CORRIDOR = 1.9           # half-width of the hair corridor, in face widths
HAIR_MEDIUM_MAX = 0.70

# Shown when the profile cannot support a body check yet.  Written the way the
# client will read it on a phone: one instruction per line, no jargon.
REFERENCE_ADVICE: tuple[str, ...] = (
    "Apoya el telefono a 2 o 3 metros de distancia, a la altura del pecho.",
    "Usa la camara trasera, no la frontal.",
    "Coloca el telefono en vertical (retrato).",
    "Activa el temporizador y entra en el encuadre.",
    "Sal de cuerpo entero: la cabeza y los pies dentro del cuadro.",
    "Brazos relajados a los lados, sin cruzarlos.",
    "Ropa relativamente ajustada para que se vea tu silueta real.",
    "Luz natural de ventana, suave y de frente. Sin flash.",
    "Sin filtros y sin retoque de belleza.",
    "Haz seis fotos: dos de frente, dos de perfil y dos de tres cuartos.",
)


# ------------------------------------------------------------------ helpers

def _safe(fn, *args) -> Any:
    """The analysis layer promises to degrade; this makes sure it cannot break us."""
    try:
        return fn(*args)
    except Exception:
        return None


def _ok_dict(value: Any, reason: str) -> dict:
    return value if isinstance(value, dict) else {"ok": False, "reason": reason}


def _f(value: Any, default: Any = 0.0) -> Any:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return default
    if math.isnan(out) or math.isinf(out):
        return default
    return out


def _flist(value: Any, n: int) -> list[float] | None:
    if not isinstance(value, (list, tuple, np.ndarray)) or len(value) < n:
        return None
    out: list[float] = []
    for item in list(value)[:n]:
        val = _f(item, None)
        if val is None:
            return None
        out.append(float(val))
    return out


def _round_list(values: Sequence[Any], digits: int = 6) -> list[float]:
    return [round(float(v), digits) for v in values]


def _clamp01(value: float) -> float:
    return 0.0 if value < 0.0 else (1.0 if value > 1.0 else float(value))


def _as_bool_mask(mask: Any, shape: tuple[int, int]) -> np.ndarray | None:
    """Accept 0/255 or 0/1 masks, single or multi channel; reject shape mismatch."""
    if not isinstance(mask, np.ndarray) or mask.size == 0:
        return None
    arr = mask[..., 0] if mask.ndim == 3 else mask
    if arr.ndim != 2 or arr.shape != shape:
        return None
    if arr.dtype == np.bool_:
        return arr
    return arr > (127 if float(arr.max()) > 1.5 else 0.5)


def _lab_float(img_bgr: np.ndarray) -> np.ndarray:
    """OpenCV 8 bit Lab rescaled to real CIE ranges (L 0..100, a/b centred on 0)."""
    lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
    lab[..., 0] *= 100.0 / 255.0
    lab[..., 1] -= 128.0
    lab[..., 2] -= 128.0
    return lab


def _bbox_from_bool(mask: np.ndarray | None) -> list[float] | None:
    if mask is None or not mask.any():
        return None
    rows = np.flatnonzero(mask.any(axis=1))
    cols = np.flatnonzero(mask.any(axis=0))
    return [float(cols[0]), float(rows[0]),
            float(cols[-1] - cols[0] + 1), float(rows[-1] - rows[0] + 1)]


def _robust_mean(values: Sequence[Any],
                 drop_frac: float = DROP_FRACTION) -> tuple[float, float, int]:
    """Mean/std after discarding the samples furthest from the median.

    Population std (ddof=0) on purpose: with three photos an unbiased estimator
    is noise, and the relative band floor is what protects the user anyway.
    """
    clean = [v for v in (_f(x, None) for x in values) if v is not None]
    arr = np.asarray(clean, dtype=np.float64)
    if arr.size == 0:
        return 0.0, 0.0, 0
    if arr.size <= 2:
        return float(arr.mean()), float(arr.std()), int(arr.size)
    med = float(np.median(arr))
    mad = float(np.median(np.abs(arr - med))) * 1.4826
    scale = mad if mad > 1e-9 else max(abs(med) * 1e-3, 1e-9)
    dist = np.abs(arr - med) / scale
    keep = np.argsort(dist, kind="stable")[: arr.size - int(arr.size * drop_frac)]
    sub = arr[keep]
    return float(sub.mean()), float(sub.std()), int(sub.size)


def _robust_mean_rows(rows: np.ndarray,
                      drop_frac: float = DROP_FRACTION) -> tuple[np.ndarray, np.ndarray, int]:
    """Vector version: distance to the median measured in per-dimension MADs."""
    if rows.ndim != 2 or rows.shape[0] == 0:
        return np.zeros(0), np.zeros(0), 0
    if rows.shape[0] <= 2:
        return rows.mean(axis=0), rows.std(axis=0), int(rows.shape[0])
    med = np.median(rows, axis=0)
    mad = np.median(np.abs(rows - med), axis=0) * 1.4826
    scale = np.maximum(mad, 1e-6)
    dist = np.mean(np.abs(rows - med) / scale, axis=1)
    n = int(rows.shape[0])
    keep = np.argsort(dist, kind="stable")[: n - int(n * drop_frac)]
    sub = rows[keep]
    return sub.mean(axis=0), sub.std(axis=0), int(sub.shape[0])


# ------------------------------------------------------------ pose geometry

def _landmark_points(pose: dict, width: int, height: int,
                     min_vis: float = 0.3) -> dict[str, tuple[float, float]]:
    out: dict[str, tuple[float, float]] = {}
    marks = (pose or {}).get("landmarks")
    if not isinstance(marks, dict):
        return out
    for name, data in marks.items():
        if not isinstance(data, dict):
            continue
        if _f(data.get("v"), 1.0) < min_vis:
            continue
        x = _f(data.get("x"), None)
        y = _f(data.get("y"), None)
        if x is None or y is None:
            continue
        # Landmarks are normalised; anything far outside the frame is noise.
        if not (-0.5 <= x <= 1.5 and -0.5 <= y <= 1.5):
            continue
        out[str(name)] = (x * width, y * height)
    return out


def _mid(a: tuple[float, float] | None,
         b: tuple[float, float] | None) -> tuple[float, float] | None:
    if a is None or b is None:
        return None
    return ((a[0] + b[0]) / 2.0, (a[1] + b[1]) / 2.0)


def _dist(a: tuple[float, float], b: tuple[float, float]) -> float:
    return math.hypot(a[0] - b[0], a[1] - b[1])


def _torso_len_px(pose: dict, width: int, height: int) -> float | None:
    pts = _landmark_points(pose, width, height)
    shoulders = _mid(pts.get("left_shoulder"), pts.get("right_shoulder"))
    hips = _mid(pts.get("left_hip"), pts.get("right_hip"))
    if shoulders is None or hips is None:
        return None
    length = _dist(shoulders, hips)
    return length if length > 4.0 else None


def _point_segment_distance(p: tuple[float, float], a: tuple[float, float],
                            b: tuple[float, float]) -> float:
    vx, vy = b[0] - a[0], b[1] - a[1]
    len2 = vx * vx + vy * vy
    if len2 < 1e-6:
        return _dist(p, a)
    t = ((p[0] - a[0]) * vx + (p[1] - a[1]) * vy) / len2
    t = 0.0 if t < 0.0 else (1.0 if t > 1.0 else t)
    return math.hypot(p[0] - (a[0] + t * vx), p[1] - (a[1] + t * vy))


def _body_segments(pts: dict[str, tuple[float, float]]) -> list[tuple[str, Any, Any]]:
    """Named skeleton segments used to say where on the body a mark sits."""
    shoulders = _mid(pts.get("left_shoulder"), pts.get("right_shoulder"))
    hips = _mid(pts.get("left_hip"), pts.get("right_hip"))
    waist = _mid(shoulders, hips)
    pairs: list[tuple[str, Any, Any]] = [
        ("face", pts.get("nose"), pts.get("nose")),
        ("neck", pts.get("nose"), shoulders),
        ("chest", shoulders, waist),
        ("abdomen", waist, hips),
        ("left_torso", pts.get("left_shoulder"), pts.get("left_hip")),
        ("right_torso", pts.get("right_shoulder"), pts.get("right_hip")),
        ("left_upper_arm", pts.get("left_shoulder"), pts.get("left_elbow")),
        ("right_upper_arm", pts.get("right_shoulder"), pts.get("right_elbow")),
        ("left_forearm", pts.get("left_elbow"), pts.get("left_wrist")),
        ("right_forearm", pts.get("right_elbow"), pts.get("right_wrist")),
        ("left_hand", pts.get("left_wrist"), pts.get("left_wrist")),
        ("right_hand", pts.get("right_wrist"), pts.get("right_wrist")),
        ("left_thigh", pts.get("left_hip"), pts.get("left_knee")),
        ("right_thigh", pts.get("right_hip"), pts.get("right_knee")),
        ("left_shin", pts.get("left_knee"), pts.get("left_ankle")),
        ("right_shin", pts.get("right_knee"), pts.get("right_ankle")),
        ("left_foot", pts.get("left_ankle"), pts.get("left_foot_index")),
        ("right_foot", pts.get("right_ankle"), pts.get("right_foot_index")),
    ]
    return [(name, a, b) for name, a, b in pairs if a is not None and b is not None]


def _nearest_region(point: tuple[float, float], pose: dict,
                    width: int, height: int) -> str:
    pts = _landmark_points(pose, width, height)
    segments = _body_segments(pts)
    if not segments:
        return "unknown"
    torso = _torso_len_px(pose, width, height) or (0.35 * min(width, height))
    best_name, best_d = "unknown", float("inf")
    for name, a, b in segments:
        d = _point_segment_distance(point, a, b)
        if d < best_d:
            best_name, best_d = name, d
    return best_name if best_d <= 0.45 * torso else "unknown"


# --------------------------------------------------------- marks and hair

def _norm_bbox(bbox: Sequence[Any], person_bbox: Sequence[Any]) -> list[float]:
    x, y, w, h = [float(v) for v in list(bbox)[:4]]
    px, py, pw, ph = [float(v) for v in list(person_bbox)[:4]]
    pw = pw if pw > 1.0 else 1.0
    ph = ph if ph > 1.0 else 1.0
    return [round(_clamp01((x - px) / pw), 4), round(_clamp01((y - py) / ph), 4),
            round(_clamp01(w / pw), 4), round(_clamp01(h / ph), 4)]


def _detect_marks(img_bgr: np.ndarray, pose: dict, person_bool: np.ndarray | None,
                  regions: dict, person_bbox: Sequence[Any]) -> list[dict]:
    """Candidate tattoos and birthmarks: skin all around, something alien inside.

    A mark is a compact island whose Lab statistics sit far outside the local
    skin distribution *and* whose texture energy is above the skin surrounding
    it - the second test is what separates ink from a shadow or a highlight.
    """
    if person_bool is None or not person_bool.any():
        return []
    height, width = img_bgr.shape[:2]

    skin_area = person_bool.copy()
    hair = _as_bool_mask((regions or {}).get("hair"), (height, width))
    if hair is not None:
        skin_area &= ~hair
    person_u8 = (person_bool.astype(np.uint8)) * 255
    garment = _as_bool_mask(_safe(segment_mod.garment_mask, img_bgr, pose, person_u8),
                            (height, width))
    if garment is not None and 0 < int(garment.sum()) < int(person_bool.sum()):
        skin_area &= ~garment
    if int(skin_area.sum()) < 800:
        return []

    lab = _lab_float(img_bgr)
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    energy = cv2.GaussianBlur(np.abs(cv2.Laplacian(gray, cv2.CV_32F, ksize=3)),
                              (0, 0), 2.0)

    sel = lab[skin_area]
    if sel.shape[0] > 200000:      # the median does not need every skin pixel
        sel = sel[:: int(sel.shape[0] // 200000) + 1]
    med = np.median(sel, axis=0).astype(np.float32)
    mad = (np.median(np.abs(sel - med), axis=0) * 1.4826).astype(np.float32)
    sd = np.maximum(mad, np.array([2.5, 1.2, 1.2], dtype=np.float32))
    dev = np.sqrt((((lab - med) / sd) ** 2).sum(axis=2))
    e_thr = max(float(np.percentile(energy[skin_area], MARK_ENERGY_PCT)), 1.0)

    cand = (skin_area & (dev > MARK_DEV_SIGMA) & (energy > e_thr)).astype(np.uint8) * 255
    k3 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    k7 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    cand = cv2.morphologyEx(cand, cv2.MORPH_OPEN, k3)
    cand = cv2.morphologyEx(cand, cv2.MORPH_CLOSE, k7)

    n_labels, labels, stats, _ = cv2.connectedComponentsWithStats(cand, connectivity=8)
    person_area = max(1.0, float(person_bbox[2]) * float(person_bbox[3]))
    min_area = max(60.0, 0.0006 * person_area)
    max_area = 0.10 * person_area
    order = [i for i in range(1, n_labels)
             if min_area <= float(stats[i, cv2.CC_STAT_AREA]) <= max_area]
    order.sort(key=lambda i: -float(stats[i, cv2.CC_STAT_AREA]))

    found: list[dict] = []
    for idx in order[:40]:
        comp = labels == idx
        ring = cv2.dilate((comp.astype(np.uint8)) * 255, k7, iterations=2) > 127
        ring &= ~comp
        ring_skin = ring & skin_area
        n_ring = int(ring.sum())
        n_ring_skin = int(ring_skin.sum())
        if n_ring < 20 or n_ring_skin < 30 or n_ring_skin / float(n_ring) < 0.5:
            continue
        c_lab = lab[comp].mean(axis=0)
        r_lab = lab[ring_skin].mean(axis=0)
        delta_e = float(np.sqrt(((c_lab - r_lab) ** 2).sum()))
        if delta_e < MARK_DELTA_E:
            continue
        c_energy = float(energy[comp].mean())
        r_energy = float(energy[ring_skin].mean())
        if c_energy < 1.15 * max(r_energy, 0.5):
            continue
        x = int(stats[idx, cv2.CC_STAT_LEFT])
        y = int(stats[idx, cv2.CC_STAT_TOP])
        w = int(stats[idx, cv2.CC_STAT_WIDTH])
        h = int(stats[idx, cv2.CC_STAT_HEIGHT])
        area = float(stats[idx, cv2.CC_STAT_AREA])
        score = (0.7 * min(1.0, delta_e / 30.0)
                 + 0.3 * min(1.0, c_energy / max(2.0 * r_energy, 1.0)))
        darker = float(c_lab[0] - r_lab[0]) < -6.0
        found.append({
            "type": "tattoo" if (darker and area >= 0.0015 * person_area) else "mark",
            "region": _nearest_region((x + w / 2.0, y + h / 2.0), pose, width, height),
            "bbox_norm": _norm_bbox([x, y, w, h], person_bbox),
            "score": round(float(score), 3),
        })
    found.sort(key=lambda m: -m["score"])
    return found[:MAX_MARKS_PER_PHOTO]


def _chin_y(face: dict, height: int) -> float | None:
    bbox = _flist((face or {}).get("bbox"), 4)
    if bbox is not None and bbox[3] > 1.0:
        return bbox[1] + bbox[3]
    norm = _flist((face or {}).get("bbox_norm"), 4)
    if norm is not None and norm[3] > 0.0:
        return (norm[1] + norm[3]) * height
    return None


def _hair_sample(img_bgr: np.ndarray, regions: dict, face: dict, pose: dict) -> dict:
    """Mean hair colour plus how far the hair actually falls below the chin.

    The region mask for hair is built from the area above the face box, which is
    fine for colour and useless for length: by construction it stops at the chin,
    so every person came out "short".  That single wrong word then travelled into
    every prompt as "short dark brown hair", instructing the generator to cut the
    hair of a woman who wears it long - the opposite of preserving identity.

    So the seed mask supplies the colour, and the length is measured by growing
    that colour downwards through the person silhouette: hair-coloured, not skin,
    connected to the head, within a sane horizontal corridor either side of it.
    """
    height, width = img_bgr.shape[:2]
    seed = _as_bool_mask((regions or {}).get("hair"), (height, width))
    if seed is None or int(seed.sum()) < 200:
        return {"ok": False, "reason": "sin mascara de pelo"}
    lab = _lab_float(img_bgr)
    mean = lab[seed].mean(axis=0)

    grown = seed
    # region_masks publishes "background", not "person"; accept either so this
    # keeps working whichever the caller hands over.
    person = _as_bool_mask((regions or {}).get("person"), (height, width))
    if person is None:
        background = _as_bool_mask((regions or {}).get("background"), (height, width))
        person = ~background if background is not None else None
    fbox = _flist((face or {}).get("bbox"), 4)
    if person is not None and fbox is not None and fbox[2] > 1.0:
        # Colour distance to the measured hair colour, in Lab.
        delta = np.linalg.norm(lab - mean.reshape(1, 1, 3), axis=2)
        spread = float(np.std(np.linalg.norm(lab[seed] - mean, axis=1))) or 1.0
        near_hair = delta <= max(HAIR_LAB_TOL, 2.0 * spread)
        skin = _as_bool_mask((regions or {}).get("skin"), (height, width))
        candidate = person & near_hair
        if skin is not None:
            candidate &= ~skin
        # Hair hangs beside the head, not across the whole frame.
        cx = fbox[0] + fbox[2] / 2.0
        half = HAIR_CORRIDOR * fbox[2]
        cols = np.arange(width, dtype=np.float32)
        corridor = (np.abs(cols - cx) <= half).reshape(1, width)
        candidate &= corridor
        candidate |= seed
        try:
            n_lab, labels = cv2.connectedComponents(
                candidate.astype(np.uint8), connectivity=8)
            if n_lab > 1:
                keep = np.unique(labels[seed])
                keep = keep[keep > 0]
                if keep.size:
                    grown = np.isin(labels, keep)
        except cv2.error:
            grown = candidate

    rows = np.flatnonzero(grown.any(axis=1))
    bottom = float(rows[-1]) if rows.size else 0.0
    chin = _chin_y(face, height)
    torso = _torso_len_px(pose, width, height)
    ratio = None
    if chin is not None and torso:
        ratio = float((bottom - chin) / torso)
    return {"ok": True, "lab_mean": _round_list(mean, 3), "length_ratio": ratio,
            "coverage": round(float(grown.sum()) / float(width * height), 5)}


def _hair_label(ratio: float | None) -> str:
    if ratio is None:
        return "medium"
    if ratio <= HAIR_SHORT_MAX:
        return "short"
    if ratio <= HAIR_MEDIUM_MAX:
        return "medium"
    return "long"


# ---------------------------------------------------------- per photo pass

def analyse_source(path: str) -> dict:
    """Measure one original.  Never raises: a bad photo comes back rejected."""
    out: dict[str, Any] = {
        "path": str(path), "sha256": "", "shot_type": "unknown",
        "ok": False, "rejected": "", "reject_code": "",
    }
    info = _safe(loader.image_info, str(path))
    if isinstance(info, dict):
        out["sha256"] = str(info.get("sha256") or "")

    img = _safe(loader.load_image, str(path), ANALYSIS_MAX_SIDE)
    if not isinstance(img, np.ndarray) or img.size == 0:
        out["rejected"] = "No se pudo abrir la imagen."
        out["reject_code"] = "unreadable"
        return out
    height, width = img.shape[:2]
    out["width"], out["height"] = int(width), int(height)

    pose_d = _ok_dict(_safe(pose_mod.detect_pose, img), "pose no disponible")
    face_d = _ok_dict(_safe(face_mod.detect_face, img), "rostro no detectado")
    qual_d = _ok_dict(_safe(quality_mod.assess_quality, img, str(path)),
                      "calidad no evaluable")
    shot_d = _ok_dict(_safe(shot_mod.classify_shot, img, pose_d, face_d),
                      "encuadre desconocido")

    shot_type = str(shot_d.get("shot_type") or "unknown")
    if shot_type not in ("closeup", "half", "full"):
        shot_type = "unknown"
    out["shot_type"] = shot_type
    out["shot"] = {"shot_type": shot_type,
                   "confidence": _f(shot_d.get("confidence")),
                   "orientation": str(shot_d.get("orientation") or "")}

    q_score = _f(qual_d.get("score"), None) if qual_d.get("ok") else None
    out["quality"] = {
        "ok": bool(qual_d.get("ok")),
        "score": float(q_score) if q_score is not None else 0.0,
        "beauty_filter_suspected": bool(qual_d.get("beauty_filter_suspected")),
        "issues": [str(i) for i in (qual_d.get("issues") or [])][:8],
    }
    out["pose"] = {"ok": bool(pose_d.get("ok")),
                   "visible_count": int(_f(pose_d.get("visible_count"), 0)),
                   "backend": str(pose_d.get("backend") or "")}

    descriptor = face_d.get("descriptor")
    if face_d.get("ok") and not descriptor:
        descriptor = _safe(face_mod.face_descriptor, img, face_d)
    if not isinstance(descriptor, (list, tuple, np.ndarray)):
        descriptor = []
    # The descriptor above says "this is a human face"; the embedding says
    # WHOSE.  Both are kept: the geometric one still feeds the report page and
    # the anomaly notes, the embedding is what identity/verify.py gates on.
    # See identity/embedding.py for the measurement that made the swap
    # necessary - the old signature scored eight other women 0.958..0.995
    # against her own 0.983..0.999.
    embedding = _safe(embedding_mod.face_embedding, img, face_d) or []
    out["face"] = {
        "ok": bool(face_d.get("ok")),
        "descriptor": [round(_f(v), 6) for v in list(descriptor)],
        "embedding": list(embedding),
        "yaw": _f(face_d.get("yaw")), "pitch": _f(face_d.get("pitch")),
        "roll": _f(face_d.get("roll")),
        "bbox": _flist(face_d.get("bbox"), 4) or [],
    }

    # Rejection happens after the cheap reads so the UI can explain itself.
    if q_score is not None and q_score < QUALITY_FLOOR:
        out["rejected"] = ("Calidad demasiado baja para medir la identidad "
                           "(nitidez o exposicion insuficientes).")
        out["reject_code"] = "low_quality"
        return out
    if not face_d.get("ok") and not pose_d.get("ok"):
        out["rejected"] = "No se detecto ni rostro ni cuerpo en la foto."
        out["reject_code"] = "no_subject"
        return out

    seg = _ok_dict(_safe(segment_mod.person_mask, img), "segmentacion no disponible")
    person_raw = seg.get("mask") if seg.get("ok") else None
    person_arg = person_raw if isinstance(person_raw, np.ndarray) else None
    person_bool = _as_bool_mask(person_raw, (height, width))
    regions = _safe(segment_mod.region_masks, img, pose_d, person_arg)
    if not isinstance(regions, dict):
        regions = {}

    body_d = _ok_dict(_safe(body_mod.measure_body, img, pose_d, person_arg),
                      "medidas corporales no disponibles")
    skin_d = _ok_dict(_safe(skin_mod.skin_stats, img, pose_d, face_d), "piel no medible")

    metrics: dict[str, float] = {}
    for name, value in (body_d.get("metrics") or {}).items():
        val = _f(value, None)
        if val is not None:
            metrics[str(name)] = float(val)
    # The reliability verdict has to travel with the numbers.  Without it the
    # aggregator cannot tell a clean width from one measured across a crossed
    # arm, and the bands end up describing the poses rather than the body.
    out["body"] = {"ok": bool(body_d.get("ok")),
                   "confidence": _f(body_d.get("confidence")),
                   "shot_type": str(body_d.get("shot_type") or shot_type),
                   "metrics": metrics,
                   "unreliable": [str(u) for u in (body_d.get("unreliable") or [])],
                   "reliability": {str(k): str(v) for k, v
                                   in (body_d.get("reliability") or {}).items()},
                   "corrected": [str(c) for c in (body_d.get("corrected") or [])],
                   "yaw_estimate": _f(body_d.get("yaw_estimate"))}
    out["skin"] = {"ok": bool(skin_d.get("ok")),
                   "lab_mean": _flist(skin_d.get("lab_mean"), 3) or [],
                   "lab_std": _flist(skin_d.get("lab_std"), 3) or [],
                   "ita_deg": _f(skin_d.get("ita_deg"), None),
                   "samples": int(_f(skin_d.get("samples"), 0))}

    person_bbox = None
    if isinstance(person_raw, np.ndarray):
        box = _safe(segment_mod.bbox_of, person_raw)
        if isinstance(box, (list, tuple)) and len(box) == 4 and _f(box[2]) > 1:
            person_bbox = [float(v) for v in box]
    if person_bbox is None:
        person_bbox = (_bbox_from_bool(person_bool)
                       or [0.0, 0.0, float(width), float(height)])

    out["person_bbox"] = [round(float(v), 2) for v in person_bbox]
    out["person_coverage"] = _f(seg.get("coverage"))
    out["marks"] = _detect_marks(img, pose_d, person_bool, regions, person_bbox)
    out["hair"] = _hair_sample(img, regions, face_d, pose_d)
    out["ok"] = True
    return out


# ------------------------------------------------------------ aggregation

def usable_metrics(body: dict) -> set[str]:
    """Metric names from one photograph that may legitimately feed a band.

    A width flagged only because the torso was turned has already had its
    foreshortening undone by measure_body, so it is an approximation worth
    keeping.  A width measured across a crossed arm, or read off a silhouette
    that was guessed from colour, is describing something other than the body
    and is discarded.  Coverage and band building must agree on this, so both
    call here rather than each keeping their own opinion.
    """
    if not isinstance(body, dict) or not body.get("ok"):
        return set()
    metrics = {str(k) for k in (body.get("metrics") or {})}
    unreliable = {str(u) for u in (body.get("unreliable") or [])}
    reasons = body.get("reliability") or {}
    corrected = {str(c) for c in (body.get("corrected") or [])}
    recovered = {name for name in unreliable
                 if str(reasons.get(name)) == "yaw" and name in corrected}
    return metrics - (unreliable - recovered)


def _coverage(accepted: list[dict]) -> dict:
    counts = {"closeup": 0, "half": 0, "full": 0, "unknown": 0}
    usable = 0
    for item in accepted:
        shot = str(item.get("shot_type") or "unknown")
        counts[shot if shot in counts else "unknown"] += 1
        if shot not in ("full", "half"):
            continue
        body = item.get("body") or {}
        if not body.get("ok") or _f(body.get("confidence")) < BODY_CONF_MIN:
            continue
        # Counting photos that merely produced a number was too generous: a
        # width measured across a crossed arm, or on a torso turned far enough
        # to foreshorten it, cannot feed a band.  Only measurements that would
        # actually survive into a gate count towards readiness, otherwise the
        # profile reports itself ready while the protection is inert.
        gate_worthy = [m for m in usable_metrics(body) if m in GATED_METRICS]
        if len(gate_worthy) >= 2:
            usable += 1
    ready = usable >= GATE_MIN_SAMPLES
    missing: list[str] = []
    if counts["full"] < 2:
        missing.append("full")
    if counts["half"] < 1:
        missing.append("half")
    if counts["closeup"] < 1:
        missing.append("closeup")
    if not ready and "full" not in missing:
        missing.append("full_measurable")
    return {**counts, "ready_for_body_check": ready, "missing": missing,
            "advice": [] if ready else list(REFERENCE_ADVICE),
            "usable_body_shots": usable}


def _aggregate_face(accepted: list[dict]) -> dict:
    """The face half of the profile: the old descriptor and the real signature.

    Every photograph's embedding is kept, not just their average.  Three
    reasons, and the first one is the only one that had to be measured: the
    average is what the gate reads (it separated her from the impostors by
    +0.3476, where the best single photograph managed +0.2894 - see
    embedding.gallery_mean), but keeping the individuals costs 24 x 128 floats,
    about 30 kB of JSON, and buys back the ability to fit a better rule, to
    drop one bad photograph, or to tell her which of her own photographs is the
    odd one out, none of which is possible once they have been averaged and the
    originals deleted.  ``self_consistency`` is stored for that last purpose:
    it is her worst photograph measured exactly the way a generated image will
    be, so the report page can show how much room is left above the line.
    """
    rows: list[list[float]] = []
    embeddings: list[list[float]] = []
    yaws: list[float] = []
    for item in accepted:
        face = item.get("face") or {}
        if not face.get("ok"):
            continue
        raw = face.get("descriptor") or []
        if isinstance(raw, (list, tuple, np.ndarray)) and len(raw) >= 8:
            vals = [_f(v, None) for v in list(raw)]
            if all(v is not None for v in vals):
                rows.append([float(v) for v in vals])
        emb = face.get("embedding") or []
        if isinstance(emb, (list, tuple, np.ndarray)) \
                and len(emb) == embedding_mod.DIMS:
            vals = [_f(v, None) for v in list(emb)]
            if all(v is not None for v in vals):
                embeddings.append([float(v) for v in vals])
        yaw = _f(face.get("yaw"), None)
        if yaw is not None:
            yaws.append(float(yaw))

    mean_emb = embedding_mod.gallery_mean(embeddings) or []
    out = {
        "embeddings": [list(e) for e in embeddings],
        "embedding_mean": list(mean_emb),
        "embedding_n": len(embeddings),
        "embedding_model": embedding_mod.SFACE_FILE if mean_emb else "",
        "embedding_self": embedding_mod.self_consistency(embeddings),
    }
    if not rows:
        out.update({"descriptor": [], "descriptor_std": [], "n": 0,
                    "yaw_range": [0.0, 0.0]})
        return out
    modal = Counter(len(r) for r in rows).most_common(1)[0][0]
    arr = np.asarray([r for r in rows if len(r) == modal], dtype=np.float64)
    mean, std, n = _robust_mean_rows(arr)
    norm = float(np.linalg.norm(mean))
    if norm > 1e-9:  # std rescaled with the mean so both live in one unit system
        mean = mean / norm
        std = std / norm
    out.update({"descriptor": _round_list(mean, 6),
                "descriptor_std": _round_list(std, 6), "n": int(n),
                "yaw_range": ([round(min(yaws), 2), round(max(yaws), 2)]
                              if yaws else [0.0, 0.0])})
    return out


def _aggregate_body(accepted: list[dict], thresholds: dict) -> dict:
    """Turn per-photo measurements into one accepted band per metric.

    Two rules here decide whether the anti-slimming gate works at all, and both
    were established by measurement (scripts/calibrate_identity.py), not taste:

    1. A measurement that measure_body itself flagged unreliable - an arm lying
       across the waist scanline, a turned torso whose widths are foreshortened -
       must never feed a band.  Letting it in inflates the standard deviation
       with pose noise rather than body variation, and since the tolerance grows
       with sigma, the band widens until nothing can fail it.
    2. The band is capped in relative terms.  A band of +/-35% is arithmetically
       defensible and practically useless: the client was slimmed by roughly a
       tenth and never noticed, so a gate that only reacts beyond a third of her
       width is decoration.  Past the cap the honest move is to stop gating and
       say so, which is what "gated" records.
    """
    samples: dict[str, list[float]] = {}
    dropped: dict[str, int] = {}
    for item in accepted:
        body = item.get("body") or {}
        if not body.get("ok") or _f(body.get("confidence")) < BODY_CONF_MIN:
            continue
        usable = usable_metrics(body)
        for name, value in (body.get("metrics") or {}).items():
            key = str(name)
            if key not in usable:
                dropped[key] = dropped.get(key, 0) + 1
                continue
            val = _f(value, None)
            if val is not None:
                samples.setdefault(key, []).append(float(val))

    floor = _f(thresholds.get("metric_tol_floor"), 0.06)
    sigma = _f(thresholds.get("metric_tol_sigma"), 2.5)
    band_max = _f(thresholds.get("metric_band_max_rel"), BAND_MAX_REL)
    min_n = int(_f(thresholds.get("metric_gate_min_n"), GATE_MIN_SAMPLES))

    out: dict[str, dict] = {}
    ordered = sorted(samples, key=lambda k: (METRIC_ORDER.index(k)
                                             if k in METRIC_ORDER else 99, k))
    for name in ordered:
        mean, std, n = _robust_mean(samples[name])
        if n == 0 or abs(mean) < 1e-9:
            continue
        # The relative floor defines the band when there are few photos; sigma
        # widens it when the person genuinely varies; the cap stops it from
        # widening into uselessness.
        tol = max(abs(mean) * floor, sigma * std)
        capped = min(tol, abs(mean) * band_max)
        out[name] = {
            "mean": round(mean, 5),
            "std": round(std, 5),
            "n": int(n),
            "lo": round(max(0.0, mean - capped), 5),
            "hi": round(mean + capped, 5),
            "spread": round(float(std / abs(mean)), 5),
            "dropped": int(dropped.get(name, 0)),
            # A band built on one or two usable photos is a guess.  It is still
            # reported, so the report page can show it, but it cannot reject.
            # Nor can a band the cap had to narrow: rule 2 above says that past
            # the cap the honest move is to stop gating, and until now "gated"
            # did not actually record it.  It was not academic - with Nayane's
            # 20 sources every gated band came out capped (shoulder asks for
            # 2.5 * 0.1358 = 43% of a mean of 0.790 and is clipped to 12%), so
            # the band was narrower than the very photographs it was learned
            # from and rejected 12 of the 14 of her own untouched photographs
            # it could judge - against 12 of 15 for the same photographs
            # slimmed by 12%, which is no discrimination at all.
            "gated": bool(n >= min_n and name in GATED_METRICS
                          and tol <= abs(mean) * band_max),
            "band_capped": bool(tol > abs(mean) * band_max),
        }
    return out


def _aggregate_skin(accepted: list[dict]) -> dict:
    means: list[list[float]] = []
    within: list[list[float]] = []
    itas: list[float] = []
    for item in accepted:
        skin = item.get("skin") or {}
        if not skin.get("ok"):
            continue
        lab = _flist(skin.get("lab_mean"), 3)
        if lab is None:
            continue
        means.append(lab)
        std = _flist(skin.get("lab_std"), 3)
        if std is not None:
            within.append([v * v for v in std])
        ita = _f(skin.get("ita_deg"), None)
        if ita is not None:
            itas.append(float(ita))
    if not means:
        return {"lab_mean": [], "lab_std": [], "ita_deg": 0.0, "n": 0}
    mean, between, n = _robust_mean_rows(np.asarray(means, dtype=np.float64))
    # Total spread = between photos + the average spread inside each photo.
    within_var = (np.asarray(within, dtype=np.float64).mean(axis=0)
                  if within else np.zeros(3, dtype=np.float64))
    std = np.sqrt(np.maximum(0.0, between ** 2 + within_var))
    lightness, blue_yellow = float(mean[0]), float(mean[2])
    if abs(blue_yellow) > 1e-6:
        ita = math.degrees(math.atan2(lightness - 50.0, blue_yellow))
    else:
        ita = _robust_mean(itas)[0] if itas else 0.0
    return {"lab_mean": _round_list(mean, 3), "lab_std": _round_list(std, 3),
            "ita_deg": round(float(ita), 2), "n": int(n)}


def skin_tolerance(skin: dict, thresholds: dict | None = None) -> dict:
    """Per-person limits for the skin check.

    A fixed CIE76 limit over the full Lab triple cannot work here.  Lightness
    moves with exposure, white balance and the time of day: across this client's
    own untouched photographs L varies by more than 15 units, which alone
    exceeds any sane fixed limit and would reject her real face as an impostor.
    Chroma - the a/b pair - is what actually describes skin colour, and it is
    comparatively stable under exposure changes.

    So the gate is built on chroma, with the limit widened to whatever spread the
    person genuinely shows in their own reference photographs.  Lightness is
    still checked, but with a limit loose enough to catch deliberate lightening
    or darkening rather than a different lamp.
    """
    base = dict(thresholds or DEFAULT_THRESHOLDS)
    std = _flist(skin.get("lab_std"), 3) if isinstance(skin, dict) else None
    floor_c = _f(base.get("chroma_max"), 6.0)
    k = _f(base.get("chroma_sigma"), 2.0)
    floor_l = _f(base.get("delta_l_max"), 22.0)
    if not std:
        return {"chroma_max_eff": round(floor_c, 3), "delta_l_max_eff": round(floor_l, 3)}
    observed_chroma = math.sqrt(float(std[1]) ** 2 + float(std[2]) ** 2)
    chroma_eff = max(floor_c, k * observed_chroma)
    # Cap it: a person whose reference set is genuinely inconsistent should not
    # end up with a limit so wide that a recoloured skin passes unnoticed.
    chroma_eff = min(chroma_eff, 18.0)
    lightness_eff = max(floor_l, 1.5 * float(std[0]))
    lightness_eff = min(lightness_eff, 45.0)
    return {"chroma_max_eff": round(chroma_eff, 3),
            "delta_l_max_eff": round(lightness_eff, 3)}


def _aggregate_hair(accepted: list[dict]) -> dict:
    labs: list[list[float]] = []
    ratios: list[float] = []
    for item in accepted:
        hair = item.get("hair") or {}
        if not hair.get("ok"):
            continue
        lab = _flist(hair.get("lab_mean"), 3)
        if lab is not None:
            labs.append(lab)
        ratio = _f(hair.get("length_ratio"), None)
        if ratio is not None:
            ratios.append(float(ratio))
    if not labs:
        return {"lab_mean": [], "length": "medium", "n": 0, "length_ratio": None}
    mean, _, n = _robust_mean_rows(np.asarray(labs, dtype=np.float64))
    ratio = float(np.median(np.asarray(ratios, dtype=np.float64))) if ratios else None
    return {"lab_mean": _round_list(mean, 3), "length": _hair_label(ratio),
            "n": int(n),
            "length_ratio": round(ratio, 3) if ratio is not None else None}


def _aggregate_marks(accepted: list[dict]) -> list[dict]:
    """Keep only what more than one photograph agrees on: ink, not shadows."""
    clusters: list[dict] = []
    for index, item in enumerate(accepted):
        for mark in (item.get("marks") or []):
            bbox = _flist(mark.get("bbox_norm"), 4)
            if bbox is None:
                continue
            cx, cy = bbox[0] + bbox[2] / 2.0, bbox[1] + bbox[3] / 2.0
            region = str(mark.get("region") or "unknown")
            best, best_d = None, MARK_MATCH_DIST
            for cluster in clusters:
                if cluster["region"] != region or index in cluster["photos"]:
                    continue
                d = math.hypot(cx - cluster["cx"], cy - cluster["cy"])
                if d < best_d:
                    best, best_d = cluster, d
            if best is None:
                clusters.append({"region": region, "cx": cx, "cy": cy,
                                 "boxes": [bbox], "photos": {index},
                                 "types": [str(mark.get("type") or "mark")]})
            else:
                best["boxes"].append(bbox)
                best["photos"].add(index)
                best["types"].append(str(mark.get("type") or "mark"))
                arr = np.asarray(best["boxes"], dtype=np.float64)
                best["cx"] = float((arr[:, 0] + arr[:, 2] / 2.0).mean())
                best["cy"] = float((arr[:, 1] + arr[:, 3] / 2.0).mean())
    out: list[dict] = []
    for cluster in clusters:
        seen = len(cluster["photos"])
        if seen < 2:
            continue
        box = np.asarray(cluster["boxes"], dtype=np.float64).mean(axis=0)
        out.append({"type": Counter(cluster["types"]).most_common(1)[0][0],
                    "region": cluster["region"],
                    "bbox_norm": _round_list(box, 4),
                    "seen_in": int(seen)})
    out.sort(key=lambda m: (-m["seen_in"], m["region"]))
    return out[:12]


# ------------------------------------------------------------------- public

def profile_from_analyses(analyses: list[dict], person_name: str) -> dict:
    """Aggregation half: turn per photo measurements into one measured identity."""
    items = [a for a in (analyses or []) if isinstance(a, dict)]
    accepted = [a for a in items if not a.get("rejected")]
    rejected = [{"path": str(a.get("path") or ""),
                 "reason": str(a.get("rejected") or ""),
                 "code": str(a.get("reject_code") or ""),
                 "shot_type": str(a.get("shot_type") or "unknown")}
                for a in items if a.get("rejected")]
    thresholds = dict(DEFAULT_THRESHOLDS)
    skin = _aggregate_skin(accepted)
    thresholds.update(skin_tolerance(skin, thresholds))
    return {
        "person_name": str(person_name or "").strip() or "Sin nombre",
        "n_sources": len(accepted),
        "coverage": _coverage(accepted),
        "face": _aggregate_face(accepted),
        "body": _aggregate_body(accepted, thresholds),
        "skin": skin,
        "hair": _aggregate_hair(accepted),
        "marks": _aggregate_marks(accepted),
        "thresholds": thresholds,
        "sources": [{"path": str(a.get("path") or ""),
                     "sha256": str(a.get("sha256") or ""),
                     "shot_type": str(a.get("shot_type") or "unknown")}
                    for a in accepted],
        "rejected": rejected,
    }


def build_profile(image_paths: list[str], person_name: str) -> dict:
    """Measure every photograph, then aggregate.  The profile outlives the
    originals: only ``sources`` remembers which files it was built from."""
    analyses = [analyse_source(str(path)) for path in (image_paths or [])]
    return profile_from_analyses(analyses, person_name)
