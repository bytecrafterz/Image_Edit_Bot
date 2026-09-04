"""Correct a paid image for free - or refuse, and say exactly why.

``retouch.restore_skin_texture`` is the model this file is built on: measure the
defect, take the correction from HER OWN photographs, measure the result, and
leave the provider's file untouched whenever the correction cannot be proved to
have helped.  What is new here is the reason for existing.  Until now a
generated image that failed a check was deleted: 0.84 USD of this
installation's 2.34 USD bought two runs whose files no longer exist at all, and
what they proved cannot be looked at again.  Deleting is the most expensive
possible answer, because the money is already spent and the picture may only
need a crop.

Three corrections, three very different verdicts from the measurement, and the
code says so rather than pretending they are equally useful:

* **The face cannot be corrected.**  Grafting the best-aligning donor across
  the six identity failures on disk lifted every one of them over the 0.45 line
  (0.29 -> 0.71, 0.32 -> 0.77, 0.40 -> 0.72 ...) with seam metrics inside the
  band her own photographs read, and 0 of 6 looked photographed when the crops
  were actually viewed.  Both rulers - the identity score and the seam metric -
  passed the single worst-looking composite in the set.  So the identity graft
  keeps the gates that a graft could ever be honest under (same head angle,
  same makeup, same light) and refuses everything else, which on that corpus is
  everything.  A face that is a different woman is a regeneration with better
  references, not a patch.  And because those three gates are all about how the
  patch would LOOK, there is a fourth about who is underneath it: a graft is
  only attempted on a near miss, within 0.05 of the bar, which is the scatter
  her own photographs show when they are re-encoded.  Without it, matching the
  lipstick is enough to graft her face onto a stranger and have the result
  score 0.45-0.59 - measured, 21 of 25 pairs.
* **The body may be nudged, never slimmed.**  The client's sentence is
  "elimina por completo cualquier retoque".  A warp that reaches the 0.85
  factor the first version clipped at IS the retouch she is paying us to
  refuse, and it is not even self-consistent: narrowing a waist by 0.907 moved
  the measured deviation from +0.102 to +0.380, so the ruler's own scatter is
  the size of the correction.  Correction is therefore limited to +/-5%, the
  whole row is rescaled instead of cutting the silhouette out and inpainting
  the hole (the Telea fill left a ghost duplicate forearm), and the result must
  measure better by more than the ruler's noise or it is thrown away.
* **A hand is fixed by not showing it, or not at all.**  Local inpainting
  drives hand severity from 0.310-0.549 to 0.000 in 6 of 6 frames by deleting
  the hand: the arm ends in a smeared blur and the scanner reads zero because
  there is nothing left to judge.  Any code that inpaints and then re-scans
  certifies its own vandalism, so this module never inpaints.  It crops the
  hand out of the frame when the crop is cheap and the picture survives it -
  1 of 7 measured cases - and otherwise refuses and names the hand.

Every function returns the same shape and, when it changes a file, a Spanish
sentence for the ficha: a repaired image is not the same claim as one that came
out right, and the client has to be able to read which one she is looking at.
"""
from __future__ import annotations

import math
from typing import Any

import cv2
import numpy as np

from ..analysis import face as face_mod
from ..analysis import loader
from ..analysis import pose as pose_mod
from ..identity import embedding as embedding_mod
from ..identity import gallery as gallery_mod
from ..identity.profile import DEFAULT_THRESHOLDS
from . import retouch as retouch_mod

# ------------------------------------------------------------------ constants

# --- face ---------------------------------------------------------------
# The two gates that decide whether a graft can be honest at all.  A similarity
# transform cannot correct yaw and cannot repaint makeup, and her library spans
# -41.3..+35.4 degrees of yaw and includes both bare lips and red lipstick, so
# these are the two ways a metrically perfect graft still reads as a mask.
FACE_MIN_GAIN = 0.05           # the identity score must really move
LIP_DELTA_E_MAX = 12.0         # CIE76 between the two mouths: lipstick guard
SKIN_DELTA_E_MAX = 10.0        # and between the two cheeks: light and grade
# The floor under the whole graft, and the only gate here that is about WHO is
# in the picture rather than about how the patch would look.  Measured
# 2026-09-04: with the two colour gates satisfied, grafting her face onto the
# eight photographs of other women in sample/ lifts them from 0.019..0.195 to
# 0.45..0.59 - over the line - in 21 of the 25 (frame, donor) pairs that align
# at all.  On the untouched files nothing gets through, but only because the
# strangers happened to wear other lipstick: the lip and cheek dE gates ask
# whether the patch would LOOK pasted, never whether pasting is legitimate, and
# they go quiet exactly when the engine does what three identity references
# push it to do - return somebody else lit and made up like her photographs.
# The near miss the graft was written for is separable from that.  Re-encoding
# and rescaling her own 24 photographs moves the reading by at most 0.0531
# (median 0.0141, p95 0.0373), so a shortfall inside 0.05 of the bar can
# honestly be her read badly, and anything further down cannot.  Every frame
# below the bar this installation owns reads at most 0.3968 - the eight
# strangers 0.019..0.195, the four paid frames the client called a different
# woman 0.292..0.397 - while her own photographs never read under 0.6738.  So
# the floor refuses all twelve and all twenty-one forgeries, and still leaves
# the 0.40..0.45 band the correction exists for.
FACE_FLOOR_MARGIN = 0.05
# The seam ruler: median |grad L| in a ring straddling the mask edge over the
# same measure in a ring just inside it.  Her own 23 mesh-bearing photographs
# read 0.768..1.627 on the same mask (median 1.271), so a composite outside
# that band has an edge her photographs do not have.  Inside it proves nothing
# on its own - it passed the worst composite measured - which is why it is the
# last gate here and never the only one.
SEAM_MIN, SEAM_MAX = 0.75, 1.65
FACE_MASK_ERODE = 0.20         # of the interocular distance: the inner face

# --- body ---------------------------------------------------------------
# What counts as a deviation worth correcting is verify.py's own paired
# tolerance, so the module never fires on something the gate would have
# accepted.
BODY_TOL = 0.08
# The ceiling on the correction itself.  Outside this band the honest answer is
# that the engine drew a different body and the image has to be made again: at
# the 0.85 the first version clipped to, the waist is visibly cinched, which is
# precisely the retouch this product exists to refuse.
BODY_FACTOR_MIN, BODY_FACTOR_MAX = 0.95, 1.05
# The correction must beat the ruler's own scatter, not merely move the number:
# re-measuring an untouched photograph at a different scale already moves these
# ratios by several percent.
BODY_MIN_IMPROVEMENT = 0.30    # relative drop in mean |deviation|
BODY_CONF_MIN = 0.5
# A metric may only drive a correction when her own photographs agree about it.
# Measured over her gallery, the same ratio spreads 47-66% across all of her
# photographs and 10% inside one shot type: where the spread is wide, the
# difference between a render and the median is her framing and not her body,
# and correcting it would be inventing a shape from an artefact.  Her half-body
# photographs read shoulder_w_over_torso 1.1438 where every generated half
# frame reads about 0.70 - a 39% "deviation" that no warp should ever act on.
BODY_SPREAD_MAX = 0.15
BODY_IDENTITY_DROP_MAX = 0.02  # a warp that costs her face is not a fix
# Rows the widths belong to, as a fraction of the shoulder-to-hip span.
_BODY_ROWS = {"bust_w_over_torso": 0.25, "waist_w_over_torso": 0.60,
              "hip_w_over_torso": 1.00, "shoulder_w_over_torso": 0.00}

# --- hands --------------------------------------------------------------
# Only a hand the gate actually failed on is worth reframing for.  Her own
# photographs reach 0.569 (IMG_8825) on the same scanner, so a threshold under
# that would reject her own pictures.
HAND_ACT_MIN = 0.60
HAND_CROP_MAX = 0.20           # a crop that costs more than a fifth is a
                               # different photograph, not a repair
HAND_FACE_MARGIN = 0.35        # keep this much around the face, in face widths
HAND_WORSE_TOL = 0.15          # the crop must not reveal a worse defect
HAND_IDENTITY_DROP_MAX = 0.02

SAVE_QUALITY = 96              # the same as retouch.py: 95 eats what we added


# -------------------------------------------------------------------- results

def _result(ok: bool, out_path: str, reason: str, nota: str = "",
            **extra) -> dict:
    """One shape for every exit, so the caller never branches on failure."""
    payload = {"ok": bool(ok), "out_path": str(out_path), "reason": reason,
               "nota": nota}
    payload.update(extra)
    return payload


def _f(value: Any, default: float = 0.0) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return default
    if out != out or math.isinf(out):
        return default
    return out


def _face_min(profile: dict) -> float:
    thresholds = (profile or {}).get("thresholds") or {}
    return _f(thresholds.get("face_embed_min"),
              float(DEFAULT_THRESHOLDS["face_embed_min"]))


def _reference(profile: dict) -> list:
    reference = ((profile or {}).get("face") or {}).get("embedding_mean") or []
    if isinstance(reference, (list, tuple)) \
            and len(reference) == embedding_mod.DIMS:
        return list(reference)
    return []


def _identity(img: np.ndarray, profile: dict, face_d: dict | None = None):
    """Her own similarity score for this frame, or None when it cannot be read."""
    reference = _reference(profile)
    if not reference:
        return None
    face = face_d if isinstance(face_d, dict) else face_mod.detect_face(img)
    if not face.get("ok"):
        return None
    embedding = embedding_mod.face_embedding(img, face)
    if not embedding:
        return None
    value = embedding_mod.similarity(embedding, reference)
    return None if value is None else float(value)


def _reading_of(img: np.ndarray, face_d: dict, path: str) -> dict:
    """The generated frame described the way gallery.py describes a photograph."""
    height, width = img.shape[:2]
    bbox = [_f(v) for v in list(face_d.get("bbox") or [0, 0, 0, 0])[:4]]
    return {"ok": bool(face_d.get("ok")), "path": str(path),
            "width": int(width), "height": int(height), "full_factor": 1.0,
            "bbox": bbox, "face_px": bbox[2],
            "yaw": _f(face_d.get("yaw")), "pitch": _f(face_d.get("pitch")),
            "roll": gallery_mod._wrap_roll(face_d.get("roll")),
            "pose_ok": gallery_mod._pose_ok(face_d),
            "mesh": face_d.get("mesh") or [],
            "landmarks": face_d.get("landmarks") or {}}


def _lab_mean(img: np.ndarray, selected: np.ndarray):
    """Mean colour of the selected pixels in real Lab units."""
    if not np.any(selected):
        return None
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)[selected].astype(np.float64)
    return np.array([lab[:, 0].mean() * 100.0 / 255.0,
                     lab[:, 1].mean() - 128.0, lab[:, 2].mean() - 128.0])


def _delta_e(a, b) -> float:
    if a is None or b is None:
        return 99.0
    return float(np.linalg.norm(np.asarray(a) - np.asarray(b)))


def _seam_ratio(img: np.ndarray, mask: np.ndarray, radius: int) -> float:
    """How visible the edge of a patch is, as her own photographs measure.

    A ring straddling the mask edge against a ring just inside it, both read as
    the median gradient of L.  A real face has structure everywhere, so the
    ratio hovers around one; a pasted patch prints a step exactly on the edge
    and drives it up.
    """
    radius = max(2, int(radius))
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,
                                       (radius * 2 | 1, radius * 2 | 1))
    grown = cv2.dilate(mask, kernel)
    shrunk = cv2.erode(mask, kernel)
    deeper = cv2.erode(shrunk, kernel)
    edge = cv2.bitwise_and(grown, cv2.bitwise_not(shrunk)) > 0
    inside = cv2.bitwise_and(shrunk, cv2.bitwise_not(deeper)) > 0
    if np.count_nonzero(edge) < 50 or np.count_nonzero(inside) < 50:
        return 0.0
    luma = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)[:, :, 0].astype(np.float32)
    gx = cv2.Sobel(luma, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(luma, cv2.CV_32F, 0, 1, ksize=3)
    magnitude = cv2.magnitude(gx, gy)
    inner = float(np.median(magnitude[inside]))
    if inner < 1e-6:
        return 0.0
    return float(np.median(magnitude[edge]) / inner)


def _mesh_hull(mesh_px: np.ndarray, shape: tuple[int, int],
               erode_px: int) -> np.ndarray:
    canvas = np.zeros(shape, np.uint8)
    hull = cv2.convexHull(mesh_px.astype(np.int32).reshape(-1, 1, 2))
    cv2.fillConvexPoly(canvas, hull, 255)
    if erode_px > 0:
        kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (erode_px * 2 | 1, erode_px * 2 | 1))
        canvas = cv2.erode(canvas, kernel)
    return canvas


def _region_mask(mesh_px: np.ndarray, indices, shape: tuple[int, int],
                 grow: int) -> np.ndarray:
    canvas = np.zeros(shape, np.uint8)
    points = np.asarray([mesh_px[i] for i in indices if i < len(mesh_px)],
                        dtype=np.float64)
    if len(points) >= 3:
        cv2.fillConvexPoly(canvas,
                           cv2.convexHull(points.astype(np.int32).reshape(-1, 1, 2)),
                           255)
    if grow > 0:
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,
                                           (grow * 2 | 1, grow * 2 | 1))
        canvas = cv2.dilate(canvas, kernel)
    return canvas


# MediaPipe FaceMesh outlines: the mouth, and two cheek patches that are skin
# on any face and are never hair, lip or iris.
_LIPS = (61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291, 409, 270, 269, 267,
         0, 37, 39, 40, 185)
_CHEEK_R = (50, 101, 118, 117, 123, 147, 187, 205, 36)
_CHEEK_L = (280, 330, 347, 346, 352, 376, 411, 425, 266)


# ---------------------------------------------------------------------- face

def restore_face(image_path: str, profile: dict, brief: dict | None = None,
                 out_path: str | None = None) -> dict:
    """Put her own face back - only when a graft could be honest.

    Measured on the six identity failures this installation still has on disk,
    the graft is 100% effective by the numbers and 0% acceptable to the eye, so
    the gates below are the ones that separate the two: the donor's head must be
    turned the same way (a similarity transform cannot rotate a head), its mouth
    must be the same colour (the best-aligning donor imported red lipstick onto
    a bare-lipped frame) and its cheeks must be lit the same way.  When those
    hold, the patch is Poisson-blended over the INNER face only and the result
    has to gain identity, keep her colour and print no edge her own photographs
    do not print.  Anything else and the paid file is left exactly as it came.
    """
    image_path = str(image_path)
    out_path = str(out_path or image_path)
    face_min = _face_min(profile)
    if not _reference(profile):
        return _result(False, image_path,
                       "el perfil no guarda tu firma facial: no se puede "
                       "comprobar ni corregir el rostro")
    try:
        gen = loader.load_image(image_path)
    except Exception as exc:                              # noqa: BLE001
        return _result(False, image_path,
                       "no se pudo abrir la imagen: %s" % str(exc)[:80])
    face_g = face_mod.detect_face(gen)
    if not face_g.get("ok") or not face_g.get("mesh"):
        return _result(False, image_path,
                       "no hay un rostro con malla en la imagen generada: no "
                       "se puede injertar nada")
    before = _identity(gen, profile, face_g)
    if before is None:
        return _result(False, image_path,
                       "no se pudo medir el parecido facial de esta imagen")
    if before >= face_min:
        return _result(False, image_path,
                       "el rostro ya es el tuyo (%.2f), no se toca" % before,
                       antes=round(before, 4))
    # Before anything is measured about how the patch would look: is this
    # picture even nearly her?  See FACE_FLOOR_MARGIN.  A graft always ends up
    # scoring like her - it IS her face - so the score after the graft proves
    # nothing about the picture underneath, and the only moment this can be
    # judged is now, on the file as it came from the engine.
    floor = face_min - FACE_FLOOR_MARGIN
    if before < floor:
        return _result(False, image_path,
                       "esta imagen no es de ti (%.2f, y por debajo de %.2f ya "
                       "no es que hayas salido mal, es otra persona): pegar tu "
                       "cara encima seria un montaje, no una correccion. Se "
                       "conserva el archivo y hay que generarla de nuevo."
                       % (before, floor),
                       antes=round(before, 4), suelo=round(floor, 4),
                       donante="")

    height, width = gen.shape[:2]
    target = _reading_of(gen, face_g, image_path)
    donor = gallery_mod.choose_donor(profile, target, "face")
    if not donor.get("path"):
        return _result(False, image_path,
                       "el rostro no es el tuyo (%.2f) y no se puede corregir "
                       "aqui: %s" % (before, donor.get("reason") or ""),
                       antes=round(before, 4), donante="")

    try:
        src = loader.load_image(donor["path"], gallery_mod.READ_MAX_SIDE)
    except Exception as exc:                              # noqa: BLE001
        return _result(False, image_path,
                       "no se pudo abrir tu foto de referencia: %s"
                       % str(exc)[:80])
    face_s = face_mod.detect_face(src)
    if not face_s.get("ok"):
        return _result(False, image_path,
                       "no se detecto tu rostro en la foto elegida")

    src_pts, dst_pts, kind = retouch_mod._correspondences(
        face_s, (src.shape[1], src.shape[0]), face_g, (width, height))
    if src_pts is None:
        return _result(False, image_path, kind)
    interocular = retouch_mod._interocular(face_g, width, height)
    if interocular < 12.0:
        return _result(False, image_path,
                       "el rostro es demasiado pequeno para injertar nada "
                       "(%.0f px entre los ojos)" % interocular)
    matrix, inliers, residual, scale = retouch_mod._estimate(src_pts, dst_pts,
                                                             interocular)
    if matrix is None or inliers < retouch_mod.MIN_INLIER_FRACTION:
        return _result(False, image_path,
                       "tu foto no se puede alinear con esta imagen")
    warped, cover = retouch_mod._warp_source(src, matrix, scale,
                                             (width, height))
    if warped is None:
        return _result(False, image_path, "no se pudo encajar tu foto")

    mesh_g = retouch_mod._mesh_px(face_g, width, height)
    if mesh_g is None:
        return _result(False, image_path, "no hay malla facial en el resultado")
    erode = int(max(2, round(FACE_MASK_ERODE * interocular)))
    mask = _mesh_hull(mesh_g, (height, width), erode)
    mask = cv2.bitwise_and(mask, cover)
    if int(np.count_nonzero(mask)) < 400:
        return _result(False, image_path,
                       "la zona del rostro es demasiado pequena para injertar")

    # --- the two gates that a metric alone cannot see ---------------------
    grow = int(max(1, round(0.03 * interocular)))
    lips = cv2.bitwise_and(_region_mask(mesh_g, _LIPS, (height, width), grow),
                           cover) > 0
    cheeks = cv2.bitwise_or(
        _region_mask(mesh_g, _CHEEK_R, (height, width), 0),
        _region_mask(mesh_g, _CHEEK_L, (height, width), 0))
    cheeks = cv2.bitwise_and(cheeks, cover) > 0
    lip_de = _delta_e(_lab_mean(gen, lips), _lab_mean(warped, lips))
    skin_de = _delta_e(_lab_mean(gen, cheeks), _lab_mean(warped, cheeks))
    if lip_de > LIP_DELTA_E_MAX:
        return _result(False, image_path,
                       "tu foto tiene los labios de otro color que esta imagen "
                       "(diferencia %.0f): pegar el rostro se veria como una "
                       "mascara, no se toca" % lip_de,
                       antes=round(before, 4), labios=round(lip_de, 1),
                       donante=donor["path"])
    if skin_de > SKIN_DELTA_E_MAX:
        return _result(False, image_path,
                       "tu foto esta iluminada de otra manera que esta imagen "
                       "(diferencia %.0f): pegar el rostro se veria pegado, no "
                       "se toca" % skin_de,
                       antes=round(before, 4), piel=round(skin_de, 1),
                       donante=donor["path"])

    # --- the graft --------------------------------------------------------
    moments = cv2.moments(mask, binaryImage=True)
    if moments["m00"] < 1.0:
        return _result(False, image_path, "no se pudo situar el injerto")
    centre = (int(moments["m10"] / moments["m00"]),
              int(moments["m01"] / moments["m00"]))
    try:
        blended = cv2.seamlessClone(warped, gen, mask, centre, cv2.NORMAL_CLONE)
    except Exception as exc:                              # noqa: BLE001
        return _result(False, image_path,
                       "no se pudo fundir el rostro: %s" % str(exc)[:80])

    after = _identity(blended, profile)
    seam = _seam_ratio(blended, mask, max(2, int(round(0.05 * interocular))))
    if after is None:
        return _result(False, image_path,
                       "tras el injerto no se pudo medir el rostro: se deja la "
                       "imagen como vino")
    if after < face_min or after < before + FACE_MIN_GAIN:
        return _result(False, image_path,
                       "el injerto no consigue que seas tu (%.2f -> %.2f, "
                       "minimo %.2f): se deja la imagen como vino"
                       % (before, after, face_min),
                       antes=round(before, 4), despues=round(after, 4),
                       costura=round(seam, 3), donante=donor["path"])
    if not (SEAM_MIN <= seam <= SEAM_MAX):
        return _result(False, image_path,
                       "el injerto deja un borde visible (costura %.2f, tus "
                       "fotos dan de %.2f a %.2f): se deja la imagen como vino"
                       % (seam, SEAM_MIN, SEAM_MAX),
                       antes=round(before, 4), despues=round(after, 4),
                       costura=round(seam, 3), donante=donor["path"])

    try:
        written = loader.save_image(blended, out_path, quality=SAVE_QUALITY)
    except Exception as exc:                              # noqa: BLE001
        return _result(False, image_path,
                       "no se pudo guardar el resultado: %s" % str(exc)[:80])
    return _result(True, written,
                   "se injerto tu rostro desde tu propia foto",
                   nota=("Se corrigio el rostro con tu foto %s: el parecido "
                         "paso de %.2f a %.2f. La imagen esta retocada por el "
                         "robot, no salio asi del motor."
                         % (donor.get("detail", {}).get("elegida")
                            or donor["path"], before, after)),
                   antes=round(before, 4), despues=round(after, 4),
                   costura=round(seam, 3), labios=round(lip_de, 1),
                   piel=round(skin_de, 1), donante=donor["path"],
                   puntos=kind, residual=round(residual / interocular, 4))


# ---------------------------------------------------------------------- body

def _measure_body(img: np.ndarray) -> tuple[dict, dict, Any]:
    from ..analysis import body as body_mod
    from ..analysis import segment as segment_mod

    pose_d = pose_mod.detect_pose(img)
    face_d = face_mod.detect_face(img)
    seg = segment_mod.person_mask(img, pose_d)
    mask = seg.get("mask") if seg.get("ok") else None
    mask = mask if isinstance(mask, np.ndarray) else None
    body = body_mod.measure_body(img, pose_d, mask, face_d)
    return (body if isinstance(body, dict) else {}), pose_d, face_d


def _deviations(metrics: dict, consensus: dict) -> dict:
    out: dict[str, float] = {}
    for name, target in (consensus or {}).items():
        got = _f(metrics.get(name))
        if got > 0.0 and _f(target) > 0.0:
            out[name] = got / float(target) - 1.0
    return out


def _row_ys(pose_d: dict, height: int) -> tuple[float, float]:
    """Shoulder line and hip line in pixels, or (0, 0)."""
    landmarks = (pose_d or {}).get("landmarks") or {}

    def y(name: str):
        got = landmarks.get(name)
        return _f(got.get("y")) * height if isinstance(got, dict) else None

    tops = [y("left_shoulder"), y("right_shoulder")]
    bottoms = [y("left_hip"), y("right_hip")]
    if any(v is None for v in tops + bottoms):
        return 0.0, 0.0
    return float(np.mean(tops)), float(np.mean(bottoms))


def _width_warp(img: np.ndarray, centre_x: float, profile_factor: np.ndarray
                ) -> np.ndarray:
    """Rescale each row horizontally about the body's centre line.

    The whole row moves, background included.  That is the point: cutting the
    silhouette out, narrowing it and filling the gap is what left a ghost
    duplicate forearm against the wall, and inventing pixels to make a
    measurement pass is the opposite of what this product does.  At the +/-5%
    this module allows, a row rescale is invisible and invents nothing.
    """
    height, width = img.shape[:2]
    xs = np.arange(width, dtype=np.float32)[None, :]
    factors = profile_factor.reshape(-1, 1).astype(np.float32)
    map_x = centre_x + (xs - centre_x) / np.maximum(factors, 1e-3)
    map_x = np.clip(map_x, 0.0, float(width - 1))
    map_y = np.repeat(np.arange(height, dtype=np.float32)[:, None], width, 1)
    return cv2.remap(img, map_x, map_y, interpolation=cv2.INTER_LINEAR,
                     borderMode=cv2.BORDER_REPLICATE)


def restore_body(image_path: str, profile: dict, brief: dict | None = None,
                 out_path: str | None = None) -> dict:
    """Bring her proportions back towards the CONSENSUS of her photographs.

    Not towards one photograph: judging a body against a single picture judges
    it against that picture's lean.  And only within +/-5%: past that the engine
    drew a different body and the answer is to make the image again, because a
    correction big enough to be visible is a retouch, which is the thing the
    client came here to stop.  The result is re-measured and thrown away unless
    it improved by more than the ruler's own scatter and kept her face.
    """
    image_path = str(image_path)
    out_path = str(out_path or image_path)
    try:
        gen = loader.load_image(image_path)
    except Exception as exc:                              # noqa: BLE001
        return _result(False, image_path,
                       "no se pudo abrir la imagen: %s" % str(exc)[:80])
    body, pose_d, face_d = _measure_body(gen)
    if not body.get("ok") or _f(body.get("confidence")) < BODY_CONF_MIN:
        return _result(False, image_path,
                       "no se pueden medir tus proporciones en esta imagen: no "
                       "se toca")
    metrics = {k: _f(v) for k, v in (body.get("metrics") or {}).items()}
    unreliable = {str(u) for u in (body.get("unreliable") or [])}

    shot = str(body.get("shot_type") or "")
    if not shot:
        try:
            from ..analysis import shot as shot_mod
            shot = str((shot_mod.classify_shot(gen, pose_d, face_d)
                        or {}).get("shot_type") or "")
        except Exception:                                 # noqa: BLE001
            shot = ""
    consensus_res = gallery_mod.choose_donor(profile, {"shot_type": shot},
                                             "body")
    consensus = (consensus_res.get("detail") or {}).get("consenso") or {}
    if not consensus:
        return _result(False, image_path, consensus_res.get("reason") or
                       "no hay consenso de tus proporciones")

    spread = (consensus_res.get("detail") or {}).get("dispersion") or {}
    usable: dict[str, float] = {}
    skipped: list[str] = []
    for name, value in consensus.items():
        if name not in _BODY_ROWS:
            continue                      # only widths can be corrected at all
        if name in unreliable:
            skipped.append("%s no se midio con fiabilidad aqui" % name)
            continue
        if _f(spread.get(name), 1.0) > BODY_SPREAD_MAX:
            skipped.append("tus fotos discrepan un %d%% en %s"
                           % (int(round(_f(spread.get(name)) * 100)), name))
            continue
        usable[name] = value
    deviations = _deviations(metrics, usable)
    if not deviations:
        return _result(False, image_path,
                       "ninguna medida de esta imagen se puede comparar con "
                       "tus fotos (%s): no se toca"
                       % ("; ".join(skipped[:2]) if skipped
                          else "no hay medidas de ancho comunes"),
                       omitidas=skipped)
    worst = max(abs(v) for v in deviations.values())
    if worst <= BODY_TOL:
        return _result(False, image_path,
                       "tus proporciones ya coinciden con tus fotos (la mayor "
                       "diferencia es del %d%%)" % int(round(worst * 100)),
                       desviaciones={k: round(v, 4)
                                     for k, v in deviations.items()})

    factors = {name: 1.0 / (1.0 + dev) for name, dev in deviations.items()}
    outside = {name: value for name, value in factors.items()
               if not (BODY_FACTOR_MIN <= value <= BODY_FACTOR_MAX)}
    if outside:
        name, value = sorted(outside.items(), key=lambda kv: -abs(kv[1] - 1.0))[0]
        return _result(False, image_path,
                       "esta imagen se aparta un %d%% de tus proporciones "
                       "(%s): corregirlo seria estrecharte o ensancharte, que "
                       "es justo el retoque que no se hace. Hay que generarla "
                       "de nuevo."
                       % (int(round(abs(1.0 / max(value, 1e-3) - 1.0) * 100)),
                          name),
                       desviaciones={k: round(v, 4)
                                     for k, v in deviations.items()})

    top, bottom = _row_ys(pose_d, gen.shape[0])
    if bottom - top < 8.0:
        return _result(False, image_path,
                       "no se ve bastante cuerpo para corregir nada")
    height, width = gen.shape[:2]
    rows = np.arange(height, dtype=np.float64)
    anchors = sorted(((top + _BODY_ROWS[name] * (bottom - top), value)
                      for name, value in factors.items()),
                     key=lambda item: item[0])
    ys = np.asarray([a[0] for a in anchors], dtype=np.float64)
    vs = np.asarray([a[1] for a in anchors], dtype=np.float64)
    curve = np.interp(rows, ys, vs, left=vs[0], right=vs[-1])
    # Smooth over a tenth of the torso so no row is scaled differently from the
    # one above it in a way an eye could see.
    span = max(3, int(round((bottom - top) * 0.10)) | 1)
    curve = cv2.GaussianBlur(curve.astype(np.float32).reshape(-1, 1),
                             (1, span), 0).reshape(-1)
    centre_x = float(width) / 2.0
    landmarks = (pose_d.get("landmarks") or {})
    xs = [_f((landmarks.get(name) or {}).get("x")) * width
          for name in ("left_shoulder", "right_shoulder", "left_hip",
                       "right_hip")
          if isinstance(landmarks.get(name), dict)]
    if xs:
        centre_x = float(np.mean(xs))

    warped = _width_warp(gen, centre_x, curve)
    before_identity = _identity(gen, profile, face_d)
    after_body, _pose2, face2 = _measure_body(warped)
    if not after_body.get("ok"):
        return _result(False, image_path,
                       "tras corregir no se pudieron volver a medir tus "
                       "proporciones: se deja la imagen como vino")
    after_dev = _deviations({k: _f(v) for k, v
                             in (after_body.get("metrics") or {}).items()},
                            usable)
    shared = [k for k in deviations if k in after_dev]
    if not shared:
        return _result(False, image_path,
                       "tras corregir no hay medidas comparables: se deja la "
                       "imagen como vino")
    mean_before = float(np.mean([abs(deviations[k]) for k in shared]))
    mean_after = float(np.mean([abs(after_dev[k]) for k in shared]))
    if mean_after > mean_before * (1.0 - BODY_MIN_IMPROVEMENT):
        return _result(False, image_path,
                       "la correccion no mejora de verdad (%.3f -> %.3f, y el "
                       "propio instrumento se mueve mas que eso): se deja la "
                       "imagen como vino" % (mean_before, mean_after),
                       antes=round(mean_before, 4), despues=round(mean_after, 4))
    after_identity = _identity(warped, profile, face2)
    if before_identity is not None and after_identity is not None \
            and after_identity < before_identity - BODY_IDENTITY_DROP_MAX:
        return _result(False, image_path,
                       "la correccion te cambiaria la cara (%.2f -> %.2f): se "
                       "deja la imagen como vino"
                       % (before_identity, after_identity))
    try:
        written = loader.save_image(warped, out_path, quality=SAVE_QUALITY)
    except Exception as exc:                              # noqa: BLE001
        return _result(False, image_path,
                       "no se pudo guardar el resultado: %s" % str(exc)[:80])
    worst_name = max(shared, key=lambda k: abs(deviations[k]))
    return _result(True, written, "se corrigieron las proporciones",
                   nota=("Se corrigieron tus proporciones un %d%% como maximo "
                         "(%s), comparando con la mediana de tus fotos: la "
                         "diferencia media paso de %.3f a %.3f. La imagen esta "
                         "corregida por el robot."
                         % (int(round(max(abs(1.0 - v) for v in factors.values())
                                      * 100)), worst_name, mean_before,
                            mean_after)),
                   antes=round(mean_before, 4), despues=round(mean_after, 4),
                   factores={k: round(v, 4) for k, v in factors.items()},
                   ambito=(consensus_res.get("detail") or {}).get("ambito", ""))


# --------------------------------------------------------------------- hands

def _hand_defects(defects: list[dict]) -> list[dict]:
    return [d for d in (defects or [])
            if isinstance(d, dict) and str(d.get("type")) == "hand_malformed"
            and d.get("bbox")]


def _side_es(box: list[int], width: int) -> str:
    centre = box[0] + box[2] / 2.0
    return "izquierda" if centre < width / 2.0 else "derecha"


def _crop_box(frame: tuple[int, int], keep: list[int],
              avoid: list[list[int]]) -> list[int] | None:
    """The largest crop of the same shape that keeps ``keep`` and drops
    ``avoid``.

    Only the four straight cuts are considered - a hand is at an edge or it is
    not croppable - and the cut that costs the least frame is chosen.
    """
    width, height = frame
    best = None
    for box in avoid:
        x0, y0, x1, y1 = box[0], box[1], box[0] + box[2], box[1] + box[3]
        options = [
            [x1, 0, width - x1, height],                  # cut from the left
            [0, 0, x0, height],                           # cut from the right
            [0, y1, width, height - y1],                  # cut from the top
            [0, 0, width, y0],                            # cut from the bottom
        ]
        for option in options:
            ox, oy, ow, oh = option
            if ow < 32 or oh < 32:
                continue
            if not (ox <= keep[0] and oy <= keep[1]
                    and ox + ow >= keep[0] + keep[2]
                    and oy + oh >= keep[1] + keep[3]):
                continue
            if any(not (bx >= ox + ow or bx + bw <= ox or by >= oy + oh
                        or by + bh <= oy)
                   for bx, by, bw, bh in
                   [(b[0], b[1], b[2], b[3]) for b in avoid]):
                continue
            area = ow * oh
            if best is None or area > best[0]:
                best = (area, option)
    return best[1] if best else None


def fix_hands(image_path: str, profile: dict, defects: list[dict],
              brief: dict | None = None, out_path: str | None = None) -> dict:
    """Take a broken hand out of the frame, or leave the picture alone.

    Never inpaints.  Painting over a hand and re-scanning the same pixels reads
    0.000 severity in 6 of 6 measured frames because the hand is gone, not
    because it is fixed - the arm ends in a smear.  A crop is the only local
    change that removes a defect without inventing anything, and it is only
    worth it when it is cheap: measured reframing costs were 17-43% of the
    frame and the face survived in 5 of 7 boxes, so exactly one of those seven
    would have been repaired here.
    """
    image_path = str(image_path)
    out_path = str(out_path or image_path)
    hands = _hand_defects(defects)
    if not hands:
        return _result(False, image_path, "no hay ninguna mano marcada")
    worst = max(_f(d.get("severity")) for d in hands)
    if worst < HAND_ACT_MIN:
        return _result(False, image_path,
                       "la mano no llega al limite que obliga a corregir "
                       "(%.2f de %.2f)" % (worst, HAND_ACT_MIN))
    try:
        gen = loader.load_image(image_path)
    except Exception as exc:                              # noqa: BLE001
        return _result(False, image_path,
                       "no se pudo abrir la imagen: %s" % str(exc)[:80])
    height, width = gen.shape[:2]
    face_d = face_mod.detect_face(gen)
    if not face_d.get("ok"):
        return _result(False, image_path,
                       "sin rostro localizado no se puede recortar sin riesgo")
    fb = [_f(v) for v in list(face_d.get("bbox") or [0, 0, 0, 0])[:4]]
    margin = HAND_FACE_MARGIN * max(fb[2], fb[3])
    keep = [int(max(0, fb[0] - margin)), int(max(0, fb[1] - margin)),
            int(min(width, fb[0] + fb[2] + margin)
                - max(0, fb[0] - margin)),
            int(min(height, fb[1] + fb[3] + margin)
                - max(0, fb[1] - margin))]

    boxes: list[list[int]] = []
    from .repair import _pixel_box
    for defect in hands:
        box = _pixel_box(defect.get("bbox"), width, height)
        if box:
            boxes.append(box)
    if not boxes:
        return _result(False, image_path, "la mano no trae una zona utilizable")
    side = _side_es(boxes[0], width)

    crop = _crop_box((width, height), keep, boxes)
    if crop is None:
        return _result(False, image_path,
                       "no se puede sacar del encuadre la mano %s sin perder "
                       "tu cara: la imagen se conserva tal cual y hay que "
                       "generarla de nuevo" % side,
                       mano=side, gravedad=round(worst, 3))
    cost = 1.0 - (crop[2] * crop[3]) / float(width * height)
    if cost > HAND_CROP_MAX:
        return _result(False, image_path,
                       "para quitar la mano %s habria que recortar el %d%% de "
                       "la foto: se conserva la imagen entera y hay que "
                       "generarla de nuevo" % (side, int(round(cost * 100))),
                       mano=side, recorte=round(cost, 3),
                       gravedad=round(worst, 3))

    cut = gen[crop[1]:crop[1] + crop[3], crop[0]:crop[0] + crop[2]].copy()
    before_identity = _identity(gen, profile, face_d)
    face_c = face_mod.detect_face(cut)
    if not face_c.get("ok"):
        return _result(False, image_path,
                       "al recortar la mano %s se pierde tu cara: no se toca"
                       % side, mano=side)
    after_identity = _identity(cut, profile, face_c)
    if before_identity is not None and after_identity is not None \
            and after_identity < before_identity - HAND_IDENTITY_DROP_MAX:
        return _result(False, image_path,
                       "el recorte empeoraria el parecido (%.2f -> %.2f): no "
                       "se toca" % (before_identity, after_identity),
                       mano=side)

    # The defect must be gone because it is OUTSIDE the frame now, and nothing
    # worse may have appeared at the new edges.
    from ..analysis import anomaly as anomaly_mod
    from ..analysis import segment as segment_mod
    pose_c = pose_mod.detect_pose(cut)
    seg_c = segment_mod.person_mask(cut, pose_c)
    mask_c = seg_c.get("mask") if seg_c.get("ok") else None
    regions_c = segment_mod.region_masks(
        cut, pose_c, mask_c if isinstance(mask_c, np.ndarray) else None) or {}
    scan = anomaly_mod.scan_anomalies(cut, pose_c, face_c, regions_c) or {}
    after_defects = [d for d in (scan.get("defects") or [])
                     if str(d.get("type")) != "oversmoothed_skin"]
    after_hand = max([_f(d.get("severity")) for d in after_defects
                      if str(d.get("type")) == "hand_malformed"], default=0.0)
    after_worst = max([_f(d.get("severity")) for d in after_defects],
                      default=0.0)
    if after_hand >= HAND_ACT_MIN:
        return _result(False, image_path,
                       "la mano sigue en el encuadre despues de recortar "
                       "(%.2f): no se toca" % after_hand, mano=side)
    if after_worst > worst + HAND_WORSE_TOL:
        return _result(False, image_path,
                       "el recorte deja a la vista otro fallo peor (%.2f): no "
                       "se toca" % after_worst, mano=side)

    try:
        written = loader.save_image(cut, out_path, quality=SAVE_QUALITY)
    except Exception as exc:                              # noqa: BLE001
        return _result(False, image_path,
                       "no se pudo guardar el recorte: %s" % str(exc)[:80])
    return _result(True, written, "se saco la mano del encuadre",
                   nota=("Se recorto el %d%% de la imagen para dejar fuera la "
                         "mano %s, que salio mal (gravedad %.2f). No se ha "
                         "repintado nada: la imagen es mas pequena, no "
                         "inventada." % (int(round(cost * 100)), side, worst)),
                   mano=side, recorte=round(cost, 3),
                   gravedad_antes=round(worst, 3),
                   gravedad_despues=round(after_hand, 3),
                   tamano=[crop[2], crop[3]])
