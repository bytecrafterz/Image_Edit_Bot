"""Repaint what she asked to change, and NOT her face.

The client's first requirement is the face, and the measurements say the face
cannot be rescued after the fact: generation/correct.py's restore_face wrote 0
of 9 failing images, because a Kontext render whose face reads 0.29 to 0.40
against her profile is not a damaged copy of her face, it is somebody else's,
and no local transfer can put her back.  So the face has to be right at
generation time, and the cheapest way to make a face right is not to redraw it.

That is what this module does.  A request that only changes the clothes, or
only the background, does not need the face repainted at all: fal's
flux-pro/v1/fill repaints the white part of a mask and leaves the black part
alone, and whatever it hands back, compose() below puts her own pixels back
outside the mask at full resolution.  The guarantee is therefore not a promise
made by the model - it is arithmetic done here, and it is checked on every
image before the file is written.

WHEN THIS IS POSSIBLE, MEASURED.  A mask can only carry a change that lives
inside it, so each option group was placed by asking where its pixels land:

* Clothes, garment colour, sheerness and background land inside torso, legs,
  arms or background - all of them regions this module can draw, all of them
  outside the face.  Over her 24 photographs the mask built for a clothing
  change covers 12.9% to 44.5% of the frame and overlaps the detected face
  rectangle in exactly ZERO pixels, on all 15 photographs where the pose gives
  a torso at all.
* A POSE change cannot.  Measured over the 7 photographs of her that a torso
  can be built from, changing pose moves the head within the frame by a median
  1.23 head lengths (0.37 to 2.80), while the protected ellipse only reaches
  0.62 head lengths sideways: in 19 of those 21 pairs the head would have to be
  painted outside the zone the mask forbids.  Relative to her own shoulders the
  head barely moves (median 0.22 head lengths) - it is the whole person that
  travels - which is exactly why the mask cannot follow it.
* A FRAMING change cannot either, and for a blunter reason: the fill endpoint
  has no aspect knob at all (see providers/fal MODELS['inpaint']['knobs']), so
  a masked call cannot reframe anything.
* LIGHT, GRADE and TREATMENT cannot.  Painting a global change inside the mask
  only puts the entire change on the mask boundary as a step: applying a
  +14.0 L* warm grade to her 14 measurable photographs and compositing it
  inside the clothing mask added 13.6 to 14.0 L* of step at that edge - 97% to
  100% of the change - with her face still lit the old way.
* EXPRESSION, MAKEUP and HAIR cannot, because their pixels are the face and its
  outline, which is the very thing being protected.

So the rule is not a preference, it is a consequence: if every group in the
request paints inside a maskable region, the safe path is taken automatically;
if any group needs the face or the whole frame, the request goes down the
ordinary whole-image path AND the estimate says so before the money is spent.

WHAT IT COSTS.  fal bills the fill endpoint at 0.050 USD against 0.040 USD for
Kontext multi, and that endpoint accepts no reference images, so the safe path
gives up the three-photograph identity pool that widens her margin from 0.2965
to 0.5294.  It gives it up for something strictly better: on the safe path the
face is not generated at all, so there is no margin left to defend.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from ..analysis import face as face_mod
from ..analysis import loader
from ..analysis import pose as pose_mod
from ..analysis import segment as segment_mod
from . import prompt as prompt_mod

# The image is analysed at this size and the mask is drawn on it, then scaled
# to whatever the composite needs.  1600 is what identity/verify and
# identity/gallery already read at, so the regions protected here are the
# regions those modules measured.
ANALYSE_MAX_SIDE = 1600

# How far the protected core is grown beyond the face, the hair and the hands,
# as a fraction of the shorter side of the frame.  0.03 is 36 px on her
# 1200x1600 analysis frame and 69 px on her 2316x3088 original.  Measured over
# her 24 photographs at this value the mask overlaps the detected face
# rectangle in 0 pixels on every one of the 15 that produce a torso.
PROTECT_MARGIN = 0.03

# The feather is HALF the protection margin on purpose: a blur that reached the
# protected core would blend generated pixels into her face, which is the one
# thing this module exists to prevent.  The core is zeroed again after the blur
# anyway, so this is belt and braces rather than the guarantee itself.
FEATHER_FRACTION = PROTECT_MARGIN / 2.0

# Below this the mask is not worth sending: there would be nothing to repaint,
# and the call would be paid for and answered with the photograph.  2% of the
# frame is well under the 12.9% her tightest real clothing mask covers.
MIN_COVER = 0.02

# And above this the "mask" is the whole picture, so masking buys nothing and
# the ordinary path - cheaper, and able to use her three reference photographs
# - is the honest answer.
MAX_COVER = 0.92

# Which regions from analysis/segment.region_masks each option group repaints.
# A group that is not in this table cannot be carried by a mask.
REGIONS_BY_GROUP: dict[str, tuple[str, ...]] = {
    "outfit": ("upper_body", "lower_body", "legs", "arms"),
    "clothing_color": ("upper_body", "lower_body", "legs", "arms"),
    "transparency": ("upper_body", "lower_body", "legs", "arms"),
    "footwear": ("legs",),
    "accessories": ("upper_body", "arms"),
    "scene": ("background",),
    "background": ("background",),
    "location": ("background",),
    "props": ("background",),
}

# Groups that change nothing in the picture itself - the delivery tier, the
# house style - and so never stand in the way of the safe path.
NEUTRAL_GROUPS = ("resolution", "style")

# Why each of the others cannot be masked, in her language, carrying the number
# that says so.  These sentences reach the estimate screen.
BLOCKED_REASON: dict[str, str] = {
    "pose": "cambiar la pose mueve a la persona dentro de la foto (medido en "
            "tus fotos: la cabeza se desplaza 1,23 cabezas de mediana), asi "
            "que el rostro tiene que volver a dibujarse",
    "framing": "cambiar el encuadre cambia la foto entera, y el motor que "
               "repinta por zonas no sabe reencuadrar",
    "camera": "cambiar la camara cambia la perspectiva de toda la foto",
    "expression": "cambiar la expresion es cambiar el rostro",
    "makeup": "el maquillaje esta en el rostro",
    "hair": "el peinado cambia el contorno de la cabeza, pegado al rostro",
    "lighting": "la luz cambia la foto entera: si solo se repinta el cuerpo, "
                "el rostro se queda con la luz vieja y se ve el corte",
    "color": "el color de la imagen se aplica a toda la foto",
    "grade": "el tratamiento de color se aplica a toda la foto",
    "treatment": "el acabado se aplica a toda la foto",
    "mood": "el ambiente cambia la foto entera",
    "time_of_day": "la hora del dia cambia la luz de toda la foto",
    "weather": "el clima cambia la luz de toda la foto",
    "season": "la temporada cambia la luz y el fondo de toda la foto",
    "body": "cambiar el cuerpo mueve la silueta bajo el rostro",
}

REGION_ES: dict[str, str] = {
    "upper_body": "el torso", "lower_body": "la cadera", "legs": "las piernas",
    "arms": "los brazos", "background": "el fondo",
    "persona": "tu cuerpo entero menos la cabeza y las manos",
}

# Hands, asked of the hand model and not only of the skeleton.  The protected
# core used to be built from segment.region_masks, whose "hands" region is a
# capsule drawn from the wrist along the forearm and only 0.13 of a torso wide.
# Measured against MediaPipe Hands over her 24 photographs, that capsule leaves
# real hands partly INSIDE the repaint zone: 45% of the detected hand of
# IMG_8798 and 10% of IMG_7619 were pixels fal was being asked to repaint, on a
# request whose own sentence promises "tu cara y tus manos se copian de tu
# propia foto".  Worse, on the 9 closeups where MediaPipe finds no pose at all,
# region_masks returns no hands region whatsoever, so nothing hand-shaped was
# protected there.  Adding the hull of every hand the hand model finds takes
# both readings to 0%, costs 0.12 s per mask (the mask itself costs about
# 1.4 s), never removes more than 0.52 points of repaintable area (IMG_8798,
# 28.05% -> 27.53%) and finds nothing at all - so changes nothing - on the 13
# photographs where no hand is visible.
HAND_CONFIDENCE = 0.5
HAND_MAX = 4

# The mask is bound to the photograph it was drawn on, and compose() checks the
# binding before it trusts a single pixel.  Its own ``fuera_cambiado`` cannot:
# that number is counted AFTER her pixels have been written back, so it is 0 by
# construction and stays 0 even when the mask belongs to a different frame.
# Measured: handing compose the mask of IMG_8949 with IMG_7871 as the source
# repaints 100% of the face box (142191 of 142191 pixels) and still reports
# ok=True and fuera_cambiado=0.  A mask that does not match the photograph is
# the only way this guarantee can break, so it is the one thing worth checking,
# and a sha256 of the file costs about 10 ms.
BINDING_EXT = ".json"

# The regions that are part of HER rather than of the scene.  A request that
# only repaints one of these can fall back to "the whole person minus the
# protected head" when the pose gives no torso; a background change cannot and
# must not.
BODY_REGIONS = ("upper_body", "lower_body", "legs", "arms")


# ------------------------------------------------------------------ helpers

def _safe(fn, *args, **kwargs) -> Any:
    try:
        return fn(*args, **kwargs)
    except Exception:                                    # noqa: BLE001
        return None


def _text(value: Any) -> str:
    return str(value).strip() if value is not None else ""


def _region_label(name: str) -> str:
    return REGION_ES.get(name, name)


def _groups(choices: Any) -> list[str]:
    """The canonical option groups a request actually changes."""
    out: list[str] = []
    if isinstance(choices, dict):
        items = list(choices.items())
    elif isinstance(choices, (list, tuple, set)):
        items = [(g, "x") for g in choices]
    else:
        return out
    for group, value in items:
        if not _text(group):
            continue
        if isinstance(value, (list, tuple, set)):
            if not [v for v in value if _text(v)]:
                continue
        elif value is None or not _text(value):
            continue
        key = prompt_mod.canon_group(group)
        if key not in out:
            out.append(key)
    return out


def _hand_hulls(img_bgr: np.ndarray) -> np.ndarray | None:
    """Every hand the hand model can see in this frame, as one filled mask.

    Guarded like analysis/anomaly.py: mediapipe is optional at import time, and
    a missing model must fall back to the skeleton capsule rather than take the
    whole face shield down with it.
    """
    try:
        import mediapipe as mp
    except Exception:                                    # noqa: BLE001
        return None
    height, width = img_bgr.shape[:2]
    canvas = np.zeros((height, width), np.uint8)
    try:
        rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        with mp.solutions.hands.Hands(
                static_image_mode=True, max_num_hands=HAND_MAX,
                min_detection_confidence=HAND_CONFIDENCE) as detector:
            found = detector.process(rgb)
            for hand in (getattr(found, "multi_hand_landmarks", None) or []):
                pts = np.array([[lm.x * width, lm.y * height]
                                for lm in hand.landmark], np.int32)
                cv2.fillConvexPoly(canvas, cv2.convexHull(pts), 255)
    except Exception:                                    # noqa: BLE001
        return None
    return canvas if np.count_nonzero(canvas) else None


def _binding_path(mask_path: Any) -> Path:
    return Path(str(mask_path)).with_suffix(BINDING_EXT)


def _write_binding(mask_path: Any, source_path: Any, size: list) -> None:
    """Record which photograph this mask was drawn on, beside the mask."""
    data = {"fuente": Path(str(source_path)).name,
            "sha256": loader.file_sha256(str(source_path)),
            "completa": [int(size[0]), int(size[1])]}
    _binding_path(mask_path).write_text(json.dumps(data), encoding="utf-8")


def _check_binding(mask_path: Any, source_path: Any,
                   size: tuple[int, int]) -> dict:
    """Does this mask belong to this photograph?  The only real check there is.

    A missing record is not a refusal: masks written before this existed, and
    any caller that builds one by hand, still have to work - they are simply
    reported as unverified.  A record that DISAGREES is a refusal, because the
    alternative is repainting her face and reporting 0 pixels changed.
    """
    record = _binding_path(mask_path)
    if not record.is_file():
        return {"ok": True, "estado": "sin comprobante", "reason": ""}
    try:
        data = json.loads(record.read_text(encoding="utf-8"))
        digest = str(data.get("sha256") or "")
        shape = [int(v) for v in (data.get("completa") or [0, 0])]
    except Exception:                                    # noqa: BLE001
        return {"ok": True, "estado": "comprobante ilegible", "reason": ""}
    if shape and shape != [int(size[0]), int(size[1])]:
        return {"ok": False, "estado": "no coincide",
                "reason": ("la mascara se dibujo sobre una foto de %dx%d y "
                           "esta es de %dx%d" % (shape[0], shape[1],
                                                 size[0], size[1]))}
    mine = _safe(loader.file_sha256, str(source_path))
    if digest and mine and digest != mine:
        return {"ok": False, "estado": "no coincide",
                "reason": ("la mascara no se dibujo sobre %s, asi que su zona "
                           "negra no es el rostro de esta foto"
                           % Path(str(source_path)).name)}
    return {"ok": True, "estado": "verificado", "reason": ""}


def _odd(value: float, low: int = 3, high: int = 401) -> int:
    k = int(max(low, min(high, round(value))))
    return k if k % 2 else k + 1


# --------------------------------------------------------------- the decision

def plan_mask(choices: Any) -> dict:
    """Can this request keep her face?  Decided from the option groups alone.

    No pixels are read, because this answer is needed on the estimate screen -
    before a run exists and before anything is paid for - and it has to be the
    same answer the run will act on.  Returns ``safe`` (the masked path is
    possible), ``regions`` (what the mask will cover), ``blocked`` (the groups
    that forbid it) and ``reason``, a sentence for the user.
    """
    groups = _groups(choices)
    regions: list[str] = []
    blocked: list[str] = []
    for group in groups:
        if group in NEUTRAL_GROUPS:
            continue
        wanted = REGIONS_BY_GROUP.get(group)
        if wanted:
            for name in wanted:
                if name not in regions:
                    regions.append(name)
            continue
        blocked.append(group)

    if not regions and not blocked:
        return {"safe": False, "regions": [], "blocked": [], "groups": groups,
                "reason": "No has pedido ningun cambio sobre la foto."}
    if blocked:
        parts = []
        for group in blocked[:2]:
            why = BLOCKED_REASON.get(group, "cambia la foto entera")
            parts.append("%s: %s" % (prompt_mod.group_label_es(group), why))
        return {
            "safe": False, "regions": regions, "blocked": blocked,
            "groups": groups,
            "reason": ("Tu rostro se va a volver a generar porque %s. Se "
                       "enviaran tus fotos de referencia y se comprobara la "
                       "identidad antes de ensenarte nada."
                       % "; ".join(parts)),
        }
    return {
        "safe": True, "regions": regions, "blocked": [], "groups": groups,
        "reason": ("Tu rostro NO se va a generar: solo se repinta %s, y tu "
                   "cara y tus manos se copian de tu propia foto."
                   % ", ".join(_region_label(r) for r in regions)),
    }


# ------------------------------------------------------------------ the mask

def build_mask(source_path: str, choices: Any, out_path: str,
               regions: Any = None) -> dict:
    """Draw the mask this request needs: white repaints, black stays hers.

    White is every region the requested change lives in; black is everything
    else and, above all, the face, the hair and the hands, grown by
    PROTECT_MARGIN and then forced back to black after the feather so that no
    blend can ever reach them.

    The mask is written at the FULL resolution of her photograph, because the
    composite is done there: a mask stored at the analysis size and stretched
    afterwards would move its own edge by a couple of pixels, and those pixels
    are on the boundary of her face.
    """
    plan = plan_mask(choices) if regions is None else {
        "safe": True, "regions": list(regions), "blocked": [], "reason": ""}
    result: dict[str, Any] = {"ok": False, "mask_path": "", "cover": 0.0,
                              "reason": "", "detail": {},
                              "regions": list(plan.get("regions") or [])}
    if not plan.get("safe"):
        result["reason"] = plan.get("reason") or "el cambio pedido no se puede aislar"
        return result

    small = _safe(loader.load_image, str(source_path), ANALYSE_MAX_SIDE)
    if not isinstance(small, np.ndarray) or small.size == 0:
        result["reason"] = "no se pudo leer tu foto para preparar la mascara"
        return result
    height, width = small.shape[:2]

    pose_d = _safe(pose_mod.detect_pose, small) or {}
    face_d = _safe(face_mod.detect_face, small) or {}
    seg = _safe(segment_mod.person_mask, small, pose_d) or {}
    person = seg.get("mask") if seg.get("ok") else None
    person = person if isinstance(person, np.ndarray) else None
    found = _safe(segment_mod.region_masks, small, pose_d, person)
    found = dict(found) if isinstance(found, dict) else {}

    change = np.zeros((height, width), np.uint8)
    used: list[str] = []
    for name in (plan.get("regions") or []):
        got = found.get(name)
        if isinstance(got, np.ndarray) and got.size:
            change = cv2.bitwise_or(change, got)
            used.append(name)
    if not used and person is not None and any(
            name in BODY_REGIONS for name in (plan.get("regions") or [])):
        # A closeup has no torso: 10 of her 24 photographs give MediaPipe no
        # pose at all, so region_masks returns only background, face and hair
        # and the clothing change had nowhere to land.  It has somewhere: what
        # she is wearing in a closeup is simply the part of HER that is not
        # face, hair or hands.  Measured on those 10 photographs the person
        # mask covers 59.3% to 77.1% of the frame and the protected head 11.7%
        # to 22.8%, which leaves a garment zone of 38.7% to 56.1% - and its
        # overlap with the detected face rectangle is 0 pixels on all ten.
        # Without this the face shield would be unavailable on exactly the
        # photographs where the face is largest.
        change = person.copy()
        used = ["persona"]
    if not used:
        result["reason"] = ("no se pueden localizar %s en tu foto"
                            % ", ".join(_region_label(r)
                                        for r in (plan.get("regions") or [])))
        result["detail"] = {"regiones_encontradas": sorted(found.keys())}
        return result

    # What is never repainted.  The hair goes in even when the request does not
    # touch it: it is the outline of her head and it sits against the face.
    protect = np.zeros((height, width), np.uint8)
    for name in ("face", "hair", "hands"):
        got = found.get(name)
        if isinstance(got, np.ndarray) and got.size:
            protect = cv2.bitwise_or(protect, got)
    # And the hands as the hand model sees them, not only as the skeleton
    # guesses them: see HAND_CONFIDENCE above for the two photographs where the
    # skeleton capsule left 45% and 10% of a real hand inside the repaint zone.
    hulls = _hand_hulls(small)
    if hulls is not None:
        protect = cv2.bitwise_or(protect, hulls)
    # The face rectangle from the mesh as well as the ellipse from the pose:
    # the two disagree on a turned head, and their union is the honest answer.
    box = face_d.get("bbox") if isinstance(face_d, dict) else None
    if isinstance(box, (list, tuple)) and len(box) >= 4:
        x, y, w, h = [float(v) for v in list(box)[:4]]
        if max(x, y, w, h) <= 1.5:
            x, y, w, h = x * width, y * height, w * width, h * height
        cv2.rectangle(protect, (int(x), int(y)), (int(x + w), int(y + h)), 255, -1)
    if not np.count_nonzero(protect):
        result["reason"] = ("no se encuentra tu rostro en la foto, asi que no "
                            "se puede proteger")
        return result

    margin = _odd(PROTECT_MARGIN * min(height, width))
    core = protect.copy()
    grown = cv2.dilate(protect, cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (margin, margin)))
    mask = cv2.bitwise_and(change, cv2.bitwise_not(grown))

    cover = float(np.count_nonzero(mask)) / float(height * width)
    if cover < MIN_COVER:
        result["reason"] = ("la zona a repintar es demasiado pequena (%.1f%% de "
                            "la foto)" % (100.0 * cover))
        result["cover"] = round(cover, 4)
        return result
    if cover > MAX_COVER:
        result["reason"] = ("habria que repintar casi toda la foto (%.1f%%), "
                            "asi que no compensa" % (100.0 * cover))
        result["cover"] = round(cover, 4)
        return result

    feather = _odd(FEATHER_FRACTION * min(height, width))
    soft = cv2.GaussianBlur(mask, (feather, feather), 0)
    # The blur may spread outwards but never into the protected core.
    soft[core > 0] = 0

    # Up to her real resolution, and the core is cleared again at that size: an
    # interpolated edge must not be able to reintroduce a single grey pixel
    # over her face.
    full = _safe(loader.load_image, str(source_path), 0)
    if not isinstance(full, np.ndarray) or full.size == 0:
        result["reason"] = "no se pudo leer tu foto a tamano completo"
        return result
    fh, fw = full.shape[:2]
    big = cv2.resize(soft, (fw, fh), interpolation=cv2.INTER_LINEAR)
    core_big = cv2.resize(core, (fw, fh), interpolation=cv2.INTER_NEAREST)
    core_big = cv2.dilate(core_big, cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (5, 5)))
    big[core_big > 0] = 0

    saved = _safe(loader.save_image, big, str(out_path), 100)
    if not saved:
        result["reason"] = "no se pudo guardar la mascara"
        return result
    # Which photograph this mask was drawn on, written beside the mask, so that
    # compose() can refuse one that belongs to a different frame instead of
    # repainting her face and reporting that nothing changed.
    _safe(_write_binding, saved, source_path, [fw, fh])

    result.update({
        "ok": True,
        "mask_path": str(saved),
        "cover": round(float(np.count_nonzero(big)) / float(fh * fw), 4),
        "regions": used,
        "reason": ("Se repinta solo %s (%.0f%% de la foto). Tu rostro y tus "
                   "manos se copian de tu foto original."
                   % (", ".join(_region_label(r) for r in used),
                      100.0 * cover)),
        "detail": {
            "analisis": [width, height],
            "completa": [fw, fh],
            "margen_px": margin,
            "difuminado_px": feather,
            "protegido_px": int(np.count_nonzero(core_big)),
            "manos_del_modelo": int(0 if hulls is None
                                    else np.count_nonzero(hulls)),
            "regiones": used,
        },
    })
    return result


# ------------------------------------------------------------- the guarantee

def compose(source_path: str, painted_path: str, mask_path: str,
            out_path: str, quality: int = 96) -> dict:
    """Put her own pixels back everywhere the mask is black, and check it.

    fal is asked to honour the mask, and this does not trust it to.  The file
    that comes back is scaled to her photograph and blended in only where the
    mask is not zero; outside it her own pixels are written back.

    Three numbers come out, and only two of them are evidence:

    * ``fuera_cambiado`` is 0 BY CONSTRUCTION - it is counted after her pixels
      have been assigned back, so it can never be anything else.  It is kept
      because the run and the rehearsal already read it, and it is worth
      exactly what it says: the arithmetic here did what it says it does.
    * ``comprobante`` is the number that can actually fail: whether the mask
      was drawn on THIS photograph (see BINDING_EXT).  A mask from another
      frame protects the wrong pixels, and ``fuera_cambiado`` stays 0 while it
      does - measured, 100% of her face box repainted with ok=True.
    * ``fuera_pintado`` is how far the engine's own answer strayed outside the
      mask, counted before any of it is thrown away: it says whether fal
      honoured the mask it was sent.  It is reported and never gates, because
      what a real fill endpoint returns outside the mask cannot be calibrated
      without buying images, and those pixels are discarded either way.
    """
    out: dict[str, Any] = {"ok": False, "image_path": "", "reason": "",
                           "fuera_cambiado": -1, "cover": 0.0}
    source = _safe(loader.load_image, str(source_path), 0)
    painted = _safe(loader.load_image, str(painted_path), 0)
    mask = _safe(loader.load_image, str(mask_path), 0)
    if not isinstance(source, np.ndarray) or not isinstance(painted, np.ndarray) \
            or not isinstance(mask, np.ndarray):
        out["reason"] = "no se pudo leer la imagen o la mascara"
        return out

    height, width = source.shape[:2]
    bind = _check_binding(mask_path, source_path, (width, height))
    out["comprobante"] = bind["estado"]
    if not bind["ok"]:
        out["reason"] = ("la mascara no corresponde a tu foto: %s"
                         % bind["reason"])
        return out
    if mask.ndim == 3:
        mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
    if mask.shape[:2] != (height, width):
        mask = cv2.resize(mask, (width, height), interpolation=cv2.INTER_LINEAR)
    if painted.shape[:2] != (height, width):
        # The fill endpoint hands back about one megapixel whatever it was
        # given, so the repainted clothes are enlarged to her frame.  That
        # softness is inside the mask, on the garment she asked to change; her
        # face keeps every pixel her camera recorded.
        painted = cv2.resize(painted, (width, height),
                             interpolation=cv2.INTER_LANCZOS4)

    # What the engine did outside the zone it was allowed to touch, measured
    # before anything is discarded.  8/255 is the encoder floor: her own
    # photograph re-encoded at q100 moves by at most 10/255 and by 0.10 on
    # average, so a smaller difference would be measuring the JPEG and not fal.
    keep = mask == 0
    strayed = int(np.count_nonzero(np.max(
        np.abs(painted[keep].astype(np.int16) - source[keep].astype(np.int16)),
        axis=-1) > 8)) if bool(np.any(keep)) else 0
    out["fuera_pintado"] = strayed
    out["fuera_total"] = int(np.count_nonzero(keep))

    alpha = (mask.astype(np.float32) / 255.0)[..., None]
    blended = np.clip(source.astype(np.float32) * (1.0 - alpha)
                      + painted.astype(np.float32) * alpha, 0, 255).astype(np.uint8)
    # Outside the mask the arithmetic above already returns the source, but
    # float rounding is not a guarantee; this is.
    blended[keep] = source[keep]

    changed = int(np.count_nonzero(np.any(blended[keep] != source[keep], axis=-1)))
    out["fuera_cambiado"] = changed
    out["cover"] = round(float(np.count_nonzero(mask)) / float(height * width), 4)
    if changed:
        out["reason"] = ("la imagen compuesta cambio %d pixeles fuera de la "
                         "mascara" % changed)
        return out

    saved = _safe(loader.save_image, blended, str(out_path), quality)
    if not saved:
        out["reason"] = "no se pudo guardar la imagen compuesta"
        return out
    out["ok"] = True
    out["image_path"] = str(saved)
    out["tamano"] = [width, height]
    return out
