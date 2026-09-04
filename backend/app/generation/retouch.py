"""Put her own skin back into a generated image.

The client's sentence is the whole specification: "elimina por completo
cualquier retoque y manipula la imagen para que se parezca exactamente a ella".
A generative model does not disobey that by changing her face - the identity
score of a Kontext render against her photograph is 0.99 - it disobeys by
sanding the surface off.  Measured on her own pictures, the fine band of her
facial skin (pores, fine lines, the grain her camera really recorded) drops
between 32% and 48% in every generated frame.  Shape survives, colour survives,
she stops looking photographed.

The band that dies is exactly the band that can be recovered, because the run
is always img2img from her own photograph: the same person, the same pose, the
same light, one reframe apart.  So:

    align her photograph onto the generated frame  ->  take the high
    frequency band out of it  ->  add back only the part the model removed,
    to luminance only, on skin only.

Three rules keep this honest rather than cosmetic:

* **Never invent.**  The grain comes from her file, never from a noise
  generator, and the result is capped at what her own photograph measures on
  the same skin; a picture that already has her texture is returned untouched.
* **Never blend blindly.**  If the two frames cannot be aligned - and a
  reframed generation is never pixel aligned - the image is left exactly as
  the provider returned it.  A misplaced pore band is worse than a smooth one.
* **Never move colour.**  Only L of Lab is touched, and the mean Lab of the
  treated pixels is compared before and after; a result that drifted is thrown
  away rather than saved.
"""
from __future__ import annotations

import math

import cv2
import numpy as np

from ..analysis import face as face_mod
from ..analysis import loader
from ..analysis import pose as pose_mod
from ..analysis import segment as segment_mod
from ..analysis.skin import skin_mask_ycrcb

# The evidence script normalises a face to 520 px wide and reads the band at
# sigma 1.4.  Scaling sigma with the measured face keeps "fine texture"
# meaning the same physical band of her skin on a 512 px crop and on a 2048 px
# render, instead of meaning "1.4 pixels" - which is a different thing at each
# size and would transfer the wrong band.
FACE_REF_W = 520.0
SIGMA_REF = 1.4
SIGMA_MIN, SIGMA_MAX = 0.8, 6.0

# Alignment gates.  A similarity transform is the right model for a reframe
# (crop, scale, small rotation); allowing shear would let landmark noise and a
# changed expression stretch her grain across the cheek.
MIN_POINTS = 5
# Deliberately not stricter: a generation is allowed to change her expression,
# and a mouth that opened puts a third of the mesh outside a tight RANSAC
# threshold on a closeup while the forehead and the cheeks are perfectly
# placed.  This gate only has to reject a transform that is wrong outright -
# whether her photograph really matches a given patch is decided again, patch
# by patch, by MIN_CORRESPONDENCE below.
MIN_INLIER_FRACTION = 0.35
MAX_RESIDUAL_FRACTION = 0.12       # median landmark error / interocular
SCALE_MIN, SCALE_MAX = 0.20, 5.00

MIN_SKIN_PX = 1200                 # below this the deficit is noise talking
MIN_REGION_PX = 400                # and one region needs at least this much
MIN_CORRESPONDENCE = 0.25          # NCC of the mid band: same content or not
MID_SIGMA_RATIO = 3.5              # the mid band spans a factor of 3.5
MID_MIN_FRACTION = 0.025           # and starts no finer than this of interocular
MIN_DEFICIT = 0.04                 # under 4% missing is not worth a rewrite
MIN_GAIN_STEP = 0.02               # and neither is an amplitude this small
FACTOR_MAX = 2.0                   # hard guard: never scale her band past this
BAND_CLIP_SIGMA = 3.0              # a hard edge cannot punch through a seam
MAX_COLOR_SHIFT = 2.0              # CIE76 dE on the mean of treated pixels
SAVE_QUALITY = 96                  # JPEG at 95 eats part of what we just added
MASK_MAX_SIDE = 1024               # anatomical regions are approximate anyway

# MediaPipe FaceMesh contours.  Grain belongs on skin: an iris, a lip or an
# eyebrow with pores added to it reads as a rendering fault immediately.
_EYE_R = (33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160,
          161, 246)
_EYE_L = (263, 249, 390, 373, 374, 380, 381, 382, 362, 398, 384, 385, 386, 387,
          388, 466)
_BROW_R = (46, 53, 52, 65, 55, 70, 63, 105, 66, 107)
_BROW_L = (276, 283, 282, 295, 285, 300, 293, 334, 296, 336)
_LIPS = (61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291, 409, 270, 269, 267,
         0, 37, 39, 40, 185)
_NOSTRILS = (98, 327)
_EXCLUDE_GROUPS = (_EYE_R, _EYE_L, _BROW_R, _BROW_L, _LIPS)

_I_EYE_OUT_R, _I_EYE_OUT_L = 33, 263

# Regions that are skin on a person and are not hair, clothing or background.
_SKIN_REGIONS = ("face", "upper_body", "lower_body", "arms", "hands", "legs")

# The small named subset detect_face always returns, mesh or no mesh.
_NAMED_KEYS = ("right_eye", "left_eye", "nose_tip", "mouth_left", "mouth_right",
               "chin", "left_ear", "right_ear", "forehead")


# ------------------------------------------------------------------ results

def _result(ok: bool, out_path: str, reason: str, gain: float = 1.0,
            aligned: bool = False, regions: int = 0, **extra) -> dict:
    """One shape for every exit, so the caller never has to branch on failure."""
    payload = {
        "ok": bool(ok),
        "out_path": str(out_path),
        "gain": round(float(gain), 4),
        "reason": reason,
        "aligned": bool(aligned),
        "regions": int(regions),
    }
    payload.update(extra)
    return payload


# ------------------------------------------------------------------ geometry

def _mesh_px(face: dict, width: int, height: int):
    """The 468 point mesh in the pixels of this frame, or None."""
    mesh = face.get("mesh") if isinstance(face, dict) else None
    if not isinstance(mesh, (list, tuple)) or len(mesh) < 468:
        return None
    arr = np.asarray(mesh, dtype=np.float64)
    if arr.ndim != 2 or arr.shape[1] < 2 or not np.isfinite(arr).all():
        return None
    out = np.empty((arr.shape[0], 2), dtype=np.float64)
    out[:, 0] = arr[:, 0] * float(width)
    out[:, 1] = arr[:, 1] * float(height)
    return out


def _named_px(face: dict, width: int, height: int) -> dict:
    """The named landmarks in pixels; the fallback when the mesh is short."""
    out: dict[str, tuple[float, float]] = {}
    landmarks = face.get("landmarks") if isinstance(face, dict) else None
    if not isinstance(landmarks, dict):
        return out
    for name in _NAMED_KEYS:
        point = landmarks.get(name)
        if not isinstance(point, dict):
            continue
        try:
            x = float(point["x"]) * float(width)
            y = float(point["y"]) * float(height)
        except (KeyError, TypeError, ValueError):
            continue
        if math.isfinite(x) and math.isfinite(y):
            out[name] = (x, y)
    return out


def _correspondences(face_s: dict, size_s: tuple[int, int],
                     face_g: dict, size_g: tuple[int, int]):
    """Matching points source -> generated, mesh when both have one.

    Returns (src Nx2, dst Nx2, kind) or (None, None, reason).  The mesh is
    preferred because 468 spread points make RANSAC able to ignore the handful
    that moved with the expression; the named points are enough for a scale
    and a shift when a full length frame only gave a detection box.
    """
    mesh_s = _mesh_px(face_s, size_s[0], size_s[1])
    mesh_g = _mesh_px(face_g, size_g[0], size_g[1])
    if mesh_s is not None and mesh_g is not None:
        n = min(len(mesh_s), len(mesh_g))
        if n >= 468:
            return mesh_s[:n], mesh_g[:n], "malla"

    named_s = _named_px(face_s, size_s[0], size_s[1])
    named_g = _named_px(face_g, size_g[0], size_g[1])
    shared = [k for k in _NAMED_KEYS if k in named_s and k in named_g]
    if len(shared) >= MIN_POINTS:
        src = np.asarray([named_s[k] for k in shared], dtype=np.float64)
        dst = np.asarray([named_g[k] for k in shared], dtype=np.float64)
        return src, dst, "puntos"
    return None, None, "no hay puntos faciales comunes"


def _interocular(face: dict, width: int, height: int) -> float:
    """Distance between the eye corners: the one length that scales with her
    face and not with the framing."""
    mesh = _mesh_px(face, width, height)
    if mesh is not None and len(mesh) > _I_EYE_OUT_L:
        value = float(np.hypot(*(mesh[_I_EYE_OUT_L] - mesh[_I_EYE_OUT_R])))
        if value > 4.0:
            return value
    named = _named_px(face, width, height)
    if "left_eye" in named and "right_eye" in named:
        value = float(math.hypot(named["left_eye"][0] - named["right_eye"][0],
                                 named["left_eye"][1] - named["right_eye"][1]))
        if value > 4.0:
            return value
    bbox = face.get("bbox") or []
    if len(bbox) == 4 and float(bbox[2]) > 8.0:
        # A face box is about 2.2 interocular distances wide.
        return float(bbox[2]) / 2.2
    return 0.0


def _estimate(src: np.ndarray, dst: np.ndarray, interocular: float):
    """Similarity source -> generated, with the residual that judges it.

    RANSAC because a generation is allowed to move her mouth a little and a
    least squares fit over 468 points would let those few drag the whole
    transform sideways.
    """
    threshold = max(2.0, 0.05 * interocular)
    matrix, inliers = cv2.estimateAffinePartial2D(
        src.reshape(-1, 1, 2).astype(np.float32),
        dst.reshape(-1, 1, 2).astype(np.float32),
        method=cv2.RANSAC, ransacReprojThreshold=float(threshold),
        maxIters=5000, confidence=0.995, refineIters=20)
    if matrix is None or not np.isfinite(matrix).all():
        return None, 0.0, 0.0, 0.0
    inlier_fraction = (float(np.count_nonzero(inliers)) / float(len(src))
                       if inliers is not None and len(src) else 0.0)
    moved = src @ matrix[:, :2].T + matrix[:, 2]
    residual = float(np.median(np.hypot(moved[:, 0] - dst[:, 0],
                                        moved[:, 1] - dst[:, 1])))
    scale = float(math.sqrt(abs(float(np.linalg.det(matrix[:, :2])))))
    return matrix, inlier_fraction, residual, scale


def _warp_source(src_bgr: np.ndarray, matrix: np.ndarray, scale: float,
                 size_g: tuple[int, int]):
    """Her photograph resampled into the generated frame, plus its coverage.

    Her photograph is usually far larger than the render - 2316 px against
    1024 on this client's own files - and the minification has to average
    before it samples.  It is tempting not to: the average visibly lowers her
    band, and the cubic kernel alone leaves a much larger number to hand back.
    That number is not her skin.  A 1024 px frame cannot carry a frequency
    that needs 2316 px to exist; sampling it anyway folds her sensor grain
    down as alias - noise at the wrong scale wearing her amplitude.  Measured
    on run_713/v2_a1 (scale 0.482), inside the same face mask: the unfiltered
    cubic warp reads her band as 3.053 and the averaged one as 2.022, so 51%
    of it was invented, and pure noise of her amplitude pushed through the
    unfiltered path reads 1.912 - nearly the whole difference.  Believing the
    inflated figure made the module hand back more grain than she has: two of
    two treated faces finished above her own photograph on the project's own
    cheek measurement (+27% and +8%).

    So the reduction is done properly, with INTER_AREA, down to exactly the
    pitch the render prints at, and the warp then runs at 1:1.  That is also
    the resampling the evidence script uses to establish her reference level,
    which is the level this result is judged against.
    """
    width_g, height_g = size_g
    work = src_bgr
    moved = matrix.copy()
    pre = min(1.0, float(scale))
    if pre < 0.999:
        work = cv2.resize(src_bgr, None, fx=pre, fy=pre,
                          interpolation=cv2.INTER_AREA)
        if work.size == 0:
            return None, None
        # x_small = pre * x_source, so the linear part absorbs 1 / pre.
        moved[:, :2] = matrix[:, :2] / float(pre)
    warped = cv2.warpAffine(work, moved, (width_g, height_g),
                            flags=cv2.INTER_CUBIC,
                            borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    cover = cv2.warpAffine(np.full(work.shape[:2], 255, np.uint8), moved,
                           (width_g, height_g), flags=cv2.INTER_NEAREST,
                           borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    return warped, cover


# --------------------------------------------------------------------- masks

def _fill_hull(canvas: np.ndarray, points: np.ndarray) -> None:
    if len(points) < 3:
        return
    hull = cv2.convexHull(points.astype(np.int32).reshape(-1, 1, 2))
    cv2.fillConvexPoly(canvas, hull, 255)


def _feature_exclusions(mesh_g, size_g: tuple[int, int], interocular: float,
                        face_box) -> np.ndarray:
    """Eyes, eyebrows, lips and nostrils, grown a little past their edges.

    Grown by a small fraction of the interocular distance and no more.  On a
    full length frame her face is 85 px wide, and a margin that looks modest
    in absolute pixels swallows the cheek the grain is meant for: the first
    version of this cut the treated area of the measured patch to 44%.

    Every one of those outlines comes from the mesh, and ``detect_face``
    returns ok=True with an empty mesh on two of its three backends
    (``mediapipe_detection``, ``haar``) - a box and at most six named points.
    Read literally, "no mesh" used to mean "nothing to exclude", which is the
    opposite of the truth: with the detection backend forced on run_713/v2_a1
    the transfer wrote grain onto 75% of her lips, 88% of her eyes and 82% of
    her eyebrows.  Without the outlines a cheek cannot be told from a lip, so
    the whole head is withdrawn from treatment and only body skin is left.
    """
    width, height = size_g
    canvas = np.zeros((height, width), np.uint8)
    if mesh_g is None:
        box = list(face_box or [])
        if len(box) == 4 and float(box[2]) > 0 and float(box[3]) > 0:
            x, y, w, h = [float(v) for v in box]
            pad = 0.15 * max(w, h)
            cv2.rectangle(canvas,
                          (int(round(x - pad)), int(round(y - pad))),
                          (int(round(x + w + pad)), int(round(y + h + pad))),
                          255, -1)
        return canvas
    limit = len(mesh_g)
    for group in _EXCLUDE_GROUPS:
        points = np.asarray([mesh_g[i] for i in group if i < limit],
                            dtype=np.float64)
        _fill_hull(canvas, points)
    # The nostrils are two holes, not the whole underside of the nose: the
    # bridge and the wings are skin and keep her pores like any other cheek.
    radius = int(max(2, round(0.07 * interocular)))
    for index in _NOSTRILS:
        if index < limit:
            cv2.circle(canvas, (int(round(mesh_g[index][0])),
                                int(round(mesh_g[index][1]))), radius, 255, -1)
    grow = int(max(2, round(0.035 * interocular)))
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,
                                       (grow | 1, grow | 1))
    return cv2.dilate(canvas, kernel)


def _skin_regions(gen_bgr: np.ndarray, mesh_g, interocular: float,
                  face_box) -> list:
    """Where grain may be added, kept region by region rather than merged.

    They stay separate because a generation is only her photograph again in
    the places it actually kept - she asked for a different outfit, so the
    torso of the result and the torso of her photograph are not the same
    pixels at all - and that has to be decided per region, later, by
    measurement.

    The anatomical regions are built on a reduced copy on purpose: they are
    capsules and ellipses around the pose skeleton, approximate by design, and
    a full resolution segmentation buys nothing here.  The skin gate itself
    runs at full resolution, because that boundary is the one that shows.
    """
    height, width = gen_bgr.shape[:2]
    longest = max(height, width)
    if longest > MASK_MAX_SIDE:
        factor = MASK_MAX_SIDE / float(longest)
        small = cv2.resize(gen_bgr, (max(1, int(round(width * factor))),
                                     max(1, int(round(height * factor)))),
                           interpolation=cv2.INTER_AREA)
    else:
        small = gen_bgr

    pose_d = pose_mod.detect_pose(small)
    person = segment_mod.person_mask(small, pose_d)
    person_mask = person.get("mask") if person.get("ok") else None
    regions = segment_mod.region_masks(small, pose_d, person_mask) or {}

    def full(mask):
        if not isinstance(mask, np.ndarray) or mask.size == 0:
            return None
        if mask.shape[:2] != (height, width):
            return cv2.resize(mask, (width, height),
                              interpolation=cv2.INTER_NEAREST)
        return mask

    # Skin, never hair, never a feature.  Everything below is cut down to it.
    base = skin_mask_ycrcb(gen_bgr)
    hair = full(regions.get("hair"))
    if hair is not None:
        base = cv2.bitwise_and(base, cv2.bitwise_not(hair))
    base = cv2.bitwise_and(
        base, cv2.bitwise_not(_feature_exclusions(mesh_g, (width, height),
                                                  interocular, face_box)))

    out: list = []
    mesh_face = None
    if mesh_g is not None:
        # The mesh knows the face far better than an ellipse fitted to the
        # pose does, and on a closeup the pose often finds nothing at all.
        hull = np.zeros((height, width), np.uint8)
        _fill_hull(hull, np.asarray(mesh_g, dtype=np.float64))
        mesh_face = cv2.bitwise_and(hull, base)
        if np.count_nonzero(mesh_face) < MIN_REGION_PX:
            mesh_face = None

    for name in _SKIN_REGIONS:
        if name == "face" and mesh_face is not None:
            continue                      # the mesh hull replaces it outright
        got = full(regions.get(name))
        if got is None:
            continue
        got = cv2.bitwise_and(got, base)
        if np.count_nonzero(got) >= MIN_REGION_PX:
            out.append((name, got))
    if mesh_face is not None:
        # Painted last so that where it overlaps a body region, the face wins.
        out.append(("face", mesh_face))
    return out


def _correspondence(mid_gen: np.ndarray, mid_src: np.ndarray,
                    selected: np.ndarray) -> float:
    """Do these two frames actually show the same thing here?  0..1.

    Normalised cross correlation of the mid band - the band a generator
    reproduces faithfully.  It is the one honest test of whether her
    photograph may lend this patch its grain: a torso wearing a different
    jacket scores near zero and is left alone, her cheek scores well and is
    restored.
    """
    a = mid_gen[selected]
    b = mid_src[selected]
    if a.size < MIN_REGION_PX:
        return 0.0
    a = a - float(a.mean())
    b = b - float(b.mean())
    norm = float(math.sqrt(float((a * a).sum()) * float((b * b).sum())))
    if norm < 1e-9:
        return 0.0
    return float((a * b).sum() / norm)


def _feather_sigma(interocular: float) -> float:
    """Soft enough that no edge shows, narrow enough to leave a cheek."""
    return float(max(1.0, 0.035 * interocular))


def _interior(mask: np.ndarray, interocular: float) -> np.ndarray:
    """The inside of a mask, as a bool array.

    Every measurement is taken here and never on the whole mask.  A mask edge
    is a hairline, a lip border or a jaw: pure high frequency that belongs to
    a feature and not to skin.  Measured on her own samples those few pixels
    lift the generated face's fine band from 2.32 to 3.59 and hide a real 25%
    loss of texture completely.
    """
    pull = int(max(1, round(_feather_sigma(interocular))))
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,
                                       (pull * 2 | 1, pull * 2 | 1))
    return cv2.erode(mask, kernel) > 0


def _feathered(mask: np.ndarray, interocular: float):
    """A soft edge, so nothing anywhere shows a line where grain starts.

    The mask is pulled in by the same amount it is then blurred by, which
    keeps the whole feather inside the skin: blurring alone would spill grain
    onto the lip and the iris the exclusions were built to protect.
    """
    sigma = _feather_sigma(interocular)
    core = _interior(mask, interocular).astype(np.float32)
    if not np.count_nonzero(core):
        core = (mask > 0).astype(np.float32)
    return np.clip(cv2.GaussianBlur(core, (0, 0), sigma), 0.0, 1.0)


# ---------------------------------------------------------------- measuring

def _band(luma: np.ndarray, sigma: float) -> np.ndarray:
    """The fine band: what a beauty filter removes first."""
    return luma - cv2.GaussianBlur(luma, (0, 0), sigma)


def _std_in(values: np.ndarray, selected: np.ndarray) -> float:
    if not np.any(selected):
        return 0.0
    return float(np.std(values[selected]))


def _lab_mean(lab: np.ndarray, selected: np.ndarray) -> np.ndarray:
    """Mean colour of the treated pixels in real Lab units, for the guard."""
    picked = lab[selected].astype(np.float64)
    return np.array([picked[:, 0].mean() * 100.0 / 255.0,
                     picked[:, 1].mean() - 128.0,
                     picked[:, 2].mean() - 128.0])


# ------------------------------------------------------------------- public

def restore_skin_texture(generated_path, source_path, out_path,
                         strength: float = 0.7) -> dict:
    """Give the generated image back the skin texture of her own photograph.

    ``strength`` is the fraction of the measured deficit that is returned;
    0.7 lands close to her level without ever passing it.  ``gain`` is the
    fine texture of the result over the fine texture of the input, measured on
    the same skin pixels, so the caller can log what was actually achieved.

    Nothing is written unless the transfer succeeded: on any failure the file
    at ``generated_path`` is left exactly as the provider returned it.
    """
    generated_path = str(generated_path)
    source_path = str(source_path)
    out_path = str(out_path)
    strength = float(max(0.0, min(1.0, strength)))

    try:
        gen = loader.load_image(generated_path)
        src = loader.load_image(source_path)
    except Exception as exc:                              # noqa: BLE001
        return _result(False, generated_path,
                       "no se pudo abrir la imagen: %s" % exc)
    if gen.ndim != 3 or src.ndim != 3 or gen.size == 0 or src.size == 0:
        return _result(False, generated_path, "imagen no utilizable")

    height_g, width_g = gen.shape[:2]
    height_s, width_s = src.shape[:2]

    face_g = face_mod.detect_face(gen)
    face_s = face_mod.detect_face(src)
    if not face_g.get("ok") or not face_s.get("ok"):
        return _result(False, generated_path,
                       "no se detecto el rostro en las dos imagenes")

    interocular = _interocular(face_g, width_g, height_g)
    if interocular < 8.0:
        return _result(False, generated_path, "rostro demasiado pequeno")

    # ---------------------------------------------------------- 1. alignment
    src_pts, dst_pts, kind = _correspondences(face_s, (width_s, height_s),
                                              face_g, (width_g, height_g))
    if src_pts is None:
        return _result(False, generated_path, kind)
    matrix, inlier_fraction, residual, scale = _estimate(src_pts, dst_pts,
                                                         interocular)
    if matrix is None:
        return _result(False, generated_path, "no se pudo alinear la foto")
    relative = residual / interocular
    # Blending a badly aligned band would smear her pores across the cheek;
    # returning the provider's image untouched is the safe answer, and the
    # reason has to say which of the three tests refused it.
    if not (SCALE_MIN <= scale <= SCALE_MAX):
        return _result(False, generated_path,
                       "el encuadre no encaja (escala %.2f)" % scale)
    if relative > MAX_RESIDUAL_FRACTION:
        return _result(False, generated_path,
                       "alineacion insuficiente (error %.2f del ojo a ojo)"
                       % relative)
    if inlier_fraction < MIN_INLIER_FRACTION:
        return _result(False, generated_path,
                       "pocos puntos coinciden (%.0f%% de la malla)"
                       % (100.0 * inlier_fraction))

    warped, cover = _warp_source(src, matrix, scale, (width_g, height_g))
    if warped is None:
        return _result(False, generated_path, "no se pudo encajar la foto")

    # -------------------------------------------------------------- 2. skin
    mesh_g = _mesh_px(face_g, width_g, height_g)
    candidates = _skin_regions(gen, mesh_g, interocular, face_g.get("bbox"))
    if not candidates:
        return _result(False, generated_path, "no se pudo delimitar la piel",
                       aligned=True)

    # ------------------------------------------------------------- 3. bands
    # The evidence reads the band at sigma 1.4 on a face normalised to 520 px,
    # so sigma follows the measured face and the band always means the same
    # thing.  The floor is not cosmetic: on a small face the pore band falls
    # under the pixel grid, and the finest band the file can carry is then the
    # honest thing to restore - it is still the grain her camera recorded.
    face_width = float((face_g.get("bbox") or [0, 0, 0, 0])[2]) or interocular * 2.2
    sigma = float(np.clip(SIGMA_REF * face_width / FACE_REF_W,
                          SIGMA_MIN, SIGMA_MAX))

    lab_gen = cv2.cvtColor(gen, cv2.COLOR_BGR2LAB)
    luma_gen = lab_gen[:, :, 0].astype(np.float32)
    luma_src = cv2.cvtColor(warped, cv2.COLOR_BGR2LAB)[:, :, 0].astype(np.float32)

    band_gen = _band(luma_gen, sigma)
    band_src = np.nan_to_num(_band(luma_src, sigma), nan=0.0,
                             posinf=0.0, neginf=0.0)
    # The band that answers "is this still the same thing" is a band of
    # anatomy, so it is tied to the face and not to the working sigma: at the
    # finest scale a two pixel reframe already decorrelates her own cheek from
    # itself and every region would be thrown away.
    mid_lo = max(sigma, MID_MIN_FRACTION * interocular)
    mid_gen = (cv2.GaussianBlur(luma_gen, (0, 0), mid_lo)
               - cv2.GaussianBlur(luma_gen, (0, 0), mid_lo * MID_SIGMA_RATIO))
    mid_src = (cv2.GaussianBlur(luma_src, (0, 0), mid_lo)
               - cv2.GaussianBlur(luma_src, (0, 0), mid_lo * MID_SIGMA_RATIO))

    # -------------------------------------------------- 4. region by region
    # Each region is judged on its own: does her photograph still show the
    # same thing here, and how much of her grain is missing from it.  One
    # figure for the whole body would let an untouched forearm cancel out a
    # sanded face - measured on her own samples, a changed outfit turns a real
    # 25% loss on the cheek into 2% over the body, and nothing gets repaired.
    union = np.zeros((height_g, width_g), np.uint8)
    keep: list[tuple] = []
    # The largest deficit measured on a region that could NOT be paired.  The
    # amplitudes are read before the correspondence test purely so this number
    # exists; nothing else about the loop changed, and a region that fails the
    # test is skipped exactly as before.
    unverified = 0.0
    for name, region in candidates:
        region = cv2.bitwise_and(region, cover)
        selected = _interior(region, interocular)
        if int(np.count_nonzero(selected)) < MIN_REGION_PX:
            continue
        here_gen = _std_in(band_gen, selected)
        here_src = _std_in(band_src, selected)
        if here_src < 1e-3 or here_gen < 1e-6:
            continue
        deficit = 1.0 - (here_gen / here_src)
        match = _correspondence(mid_gen, mid_src, selected)
        if match < MIN_CORRESPONDENCE:
            # She asked for another outfit, so this is not her arm any more.
            unverified = max(unverified, deficit)
            continue
        if deficit <= MIN_DEFICIT:
            continue
        keep.append((name, region, selected, match, deficit, here_gen, here_src))
        union = cv2.bitwise_or(union, region)

    if not keep:
        # Two very different silences used to say the same sentence, and one of
        # them was a lie told to her about her own photograph.  Measured over
        # the fifteen paid FLUX files on disk (scratchpad k2_why.py): the module
        # declines on nine, and on seven of those her own face failed the
        # correspondence test carrying a measured deficit of 37, 58, 60, 60,
        # 62, 64 and 65%.  On five of those seven the gate's
        # own ruler reads 17.7 to 19.9% of her grain gone, over its 14%
        # rejection line.  Telling her "the image already keeps your skin
        # texture" there contradicts what this very function just measured.  So
        # that sentence is now reserved for the case it describes - every region
        # paired and none of them missing anything - and a frame whose skin
        # could not be paired says so instead.  Neither branch touches a pixel:
        # blending a band that failed the correspondence test is still refused.
        if unverified > MIN_DEFICIT:
            return _result(False, generated_path,
                           "no se pudo comprobar tu piel contra tu foto en esta "
                           "imagen (le falta hasta un %d%% de grano, pero no "
                           "coincide lo suficiente para devolverselo): se deja "
                           "tal cual" % int(round(unverified * 100.0)),
                           gain=1.0, aligned=True, regions=0)
        # Exactly the case the client is paying for us NOT to touch: the
        # engine kept her skin, so adding grain would be our own retouch.
        return _result(False, generated_path,
                       "la imagen ya conserva su textura de piel",
                       gain=1.0, aligned=True, regions=0)

    weight = _feathered(union, interocular)
    core = weight > 0.5
    if int(np.count_nonzero(core)) < MIN_SKIN_PX:
        return _result(False, generated_path, "hay muy poca piel medible",
                       aligned=True, regions=0)

    # A mole, a stray hair or a jaw shadow survives in the band as a hard
    # edge; one pixel of misalignment there would print a ghost of it on the
    # cheek.  Clipping keeps the pores and drops the ghosts.
    limit = BAND_CLIP_SIGMA * _std_in(band_src, core)
    band_src = np.clip(band_src, -limit, limit)
    offer = band_src * weight

    # ------------------------------------------------- 5. how much to add
    # The amplitude that lands on the target is measured, not assumed.  Her
    # band is already high passed, so passing it through the same filter again
    # to read the result costs part of it - a quarter, on these files - and a
    # factor derived from her raw amplitude stops a third short of her level.
    # ``offer`` is put through exactly the measurement the result will face,
    # and the factor is solved against that.  The ceiling stays where the
    # client put it: her own photograph, never past it.
    factors = np.zeros((height_g, width_g), np.float32)
    treated: list[dict] = []
    band_offer = _band(offer, sigma)
    for name, region, selected, match, deficit, here_gen, here_src in keep:
        gives = _std_in(band_offer, selected)
        if gives < 1e-6:
            continue
        target = min(here_src, here_gen + strength * (here_src - here_gen))
        # Her grain and whatever the engine left are independent signals, so
        # their amplitudes add in quadrature and not linearly; a linear guess
        # overshoots her own level by a third.
        factor = math.sqrt(max(0.0, target * target - here_gen * here_gen))
        factor = float(min(FACTOR_MAX, max(0.0, factor / gives)))
        if factor < MIN_GAIN_STEP:
            continue
        factors[selected] = factor
        treated.append({"zona": name, "parecido": round(match, 3),
                        "deficit": round(deficit, 4),
                        "factor": round(factor, 4)})
    if not treated:
        return _result(False, generated_path,
                       "la diferencia de textura es despreciable",
                       gain=1.0, aligned=True, regions=0)
    # Soften the step where two regions with different deficits meet; the
    # feather above is what keeps the result inside the mask.
    factors = cv2.GaussianBlur(factors, (0, 0), _feather_sigma(interocular))

    fine_gen = _std_in(band_gen, core)
    fine_src = _std_in(band_src, core)

    # ------------------------------------------------------------- 6. apply
    luma_out = luma_gen + factors * offer
    luma_out = np.nan_to_num(luma_out, nan=0.0, posinf=255.0, neginf=0.0)

    # The ceiling is the whole point of the module and until now nothing
    # enforced it on the result, only on the per region arithmetic that leads
    # to it: the feather, the blur of the factors and the overlap between
    # regions all land somewhere the solver did not measure.  So the finished
    # luminance is measured once, on the same pixels, against what her own
    # photograph carries there, and the one free scalar is solved down if it
    # came out above her - again in quadrature, because her grain and what the
    # engine left are independent.  Only her level caps it: if the render was
    # already above her the module has nothing to add and says so.
    ceiling = max(fine_src, fine_gen)
    fine_new = _std_in(_band(luma_out, sigma), core)
    if fine_new > ceiling:
        room = math.sqrt(max(0.0, ceiling * ceiling - fine_gen * fine_gen))
        added = math.sqrt(max(1e-12, fine_new * fine_new - fine_gen * fine_gen))
        luma_out = luma_gen + (factors * float(room / added)) * offer
        luma_out = np.nan_to_num(luma_out, nan=0.0, posinf=255.0, neginf=0.0)
        fine_new = _std_in(_band(luma_out, sigma), core)
        if fine_new > ceiling * 1.02:
            return _result(False, generated_path,
                           "habria pasado de la textura de su propia foto",
                           aligned=True, regions=len(treated))

    lab_out = lab_gen.copy()
    lab_out[:, :, 0] = np.clip(np.rint(luma_out), 0, 255).astype(np.uint8)

    # a and b are copied untouched, so her colour and the model's lighting
    # cannot move; this only checks that clipping L did not drag the mean.
    shift = float(np.linalg.norm(_lab_mean(lab_out, core)
                                 - _lab_mean(lab_gen, core)))
    if not math.isfinite(shift) or shift > MAX_COLOR_SHIFT:
        return _result(False, generated_path,
                       "el color se habria desplazado, no se toca la imagen",
                       aligned=True, regions=len(treated))

    # Only the pixels whose L really moved are rebuilt from Lab.  The round
    # trip is not free: converting the untouched frame back through LAB2BGR
    # moved 73% of the pixels outside the treated skin and drifted their a and
    # b by 0.42 and 0.35, which is exactly the colour the module promises not
    # to move.  Everything the transfer did not reach keeps the provider's own
    # bytes.
    out = gen.copy()
    changed = lab_out[:, :, 0] != lab_gen[:, :, 0]
    if not np.any(changed):
        # The solver scaled the whole offer away - the render already carries
        # her level.  Re-encoding the file to store identical pixels would
        # only cost her one more JPEG generation.
        return _result(False, generated_path,
                       "la imagen ya conserva su textura de piel",
                       gain=1.0, aligned=True, regions=0)
    out[changed] = cv2.cvtColor(lab_out, cv2.COLOR_LAB2BGR)[changed]
    try:
        written = loader.save_image(out, out_path, quality=SAVE_QUALITY)
    except Exception as exc:                              # noqa: BLE001
        return _result(False, generated_path,
                       "no se pudo guardar el resultado: %s" % exc,
                       aligned=True, regions=len(treated))

    # Measured on the file that was actually written, not on the array in
    # memory: JPEG is part of what reaches her, so it is part of the number.
    try:
        saved = loader.load_image(written)
        luma_saved = cv2.cvtColor(saved, cv2.COLOR_BGR2LAB)[:, :, 0]
        if luma_saved.shape[:2] != (height_g, width_g):
            raise ValueError("el archivo guardado cambio de tamano")
    except Exception:                                     # noqa: BLE001
        luma_saved = lab_out[:, :, 0]
    fine_after = _std_in(_band(luma_saved.astype(np.float32), sigma), core)

    return _result(True, written, "se devolvio la textura real de la piel",
                   gain=fine_after / max(fine_gen, 1e-6), aligned=True,
                   regions=len(treated),
                   fine_before=round(fine_gen, 4),
                   fine_after=round(fine_after, 4),
                   fine_source=round(fine_src, 4),
                   sigma=round(sigma, 3),
                   points=kind,
                   residual=round(relative, 4),
                   color_shift=round(shift, 3),
                   skin_px=int(np.count_nonzero(core)),
                   zonas=treated)
