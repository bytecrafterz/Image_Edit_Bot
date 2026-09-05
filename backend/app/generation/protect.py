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

import hashlib
import json
import threading
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from ..analysis import face as face_mod
from ..analysis import loader
from ..analysis import pose as pose_mod
from ..analysis import segment as segment_mod
from ..catalog import options as catalog_mod
from ..identity import profile as profile_mod
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

# What is being repainted, said as the user asked for it rather than as the
# segmenter names it.  "se repinta solo la ropa" is the sentence she has to be
# able to act on; "se repinta el torso, la cadera, las piernas y los brazos" is
# the same fact told in the language of a mask, and she never asked for a mask.
CHANGE_ES: dict[str, str] = {
    "outfit": "la ropa", "clothing_color": "el color de la ropa",
    "transparency": "la transparencia de la tela",
    "footwear": "el calzado", "accessories": "los complementos",
    "scene": "el fondo", "background": "el fondo", "location": "el fondo",
    "props": "el fondo",
}

# And the same for the groups that FORBID the mask, in the form the sentence
# needs: "porque pediste cambiar la postura", not "porque pose".
BLOCKED_ES_NOUN: dict[str, str] = {
    "pose": "la postura", "framing": "el encuadre", "camera": "la camara",
    "expression": "la expresion", "makeup": "el maquillaje",
    "hair": "el peinado", "lighting": "la luz", "color": "el color",
    "grade": "el tratamiento de color", "treatment": "el acabado",
    "mood": "el ambiente", "time_of_day": "la hora del dia",
    "weather": "el clima", "season": "la temporada", "body": "el cuerpo",
}

# One mask belongs to one photograph and one set of regions, and BOTH the
# estimate and the run need it: the estimate to price the endpoint that will
# really be called, the run to send it.  Drawing it twice is drawing it twice
# with two chances of a different answer - which is the whole defect this
# module is being wired against - so it is drawn once, into a file named after
# the photograph and the regions, and whoever asks second reads it back.  The
# lock is here because _run_batch runs the variants in parallel threads and two
# variants that change the same thing land on the same file name.
_MASK_LOCK = threading.Lock()

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

# ---------------------------------------------------------------- her marks
#
# HER TATTOOS ARE IDENTITY, NOT CLOTHING, so they belong in the same protected
# set as her face and her hands.  Until now they were not in it: on the
# delivered image of 2026-09-04 all three - the script and two hearts below the
# left collarbone, the script down the forearm, the rose sleeve across the
# thigh - sat INSIDE the repaint zone and came back gone, and the only thing
# this module did about it was warn (see marks_at_risk below).
#
# WHERE THE MARK IS, MEASURED ON THIS PHOTOGRAPH.  It cannot come from the
# profile's stored boxes: profile.marks records ``bbox_norm`` in units of the
# PERSON BOX, and the person box is the whole frame on a closeup and a
# head-to-calf strip on a half shot, so the same chest tattoo lands somewhere
# different in every framing.  What the profile does supply, and what makes
# this work at all, is HER SKIN: ``skin.lab_mean`` plus the two tolerances
# ``chroma_max_eff`` / ``delta_l_max_eff`` that identity/verify already gates
# a generated image on.  A mark is then what it is in plain words - something
# that is not her skin, sitting inside her skin.
#
# WHY NOT profile._detect_marks ITSELF.  It builds its skin distribution from
# "the person minus whatever segment.garment_mask found", and on the new
# clothed photograph that subtraction removes only 62 816 of 606 054 person
# pixels: the warm grey top and the black skirt stay in the distribution, the
# deviation it derives from them is huge, and nothing is 3 sigma from anything
# any more.  Measured, it returns [] on that photograph - it does not see the
# chest script that is plainly there.  Handing the SAME four tests her
# profile's skin envelope instead finds that script at (628,752,74,52) and its
# two hearts at (697,757,31,27) on the analysis frame.
MARK_ZONES: dict[str, str] = {
    "chest": "torso", "left_torso": "torso", "right_torso": "torso",
    "abdomen": "torso", "neck": "neck",
    "left_upper_arm": "upper_arm", "right_upper_arm": "upper_arm",
    "left_forearm": "forearm", "right_forearm": "forearm",
    "left_thigh": "thigh", "right_thigh": "thigh",
    "left_shin": "shin", "right_shin": "shin",
}

MARK_ZONE_ES: dict[str, str] = {
    "torso": "el torso", "neck": "el cuello", "upper_arm": "el brazo",
    "forearm": "el antebrazo", "thigh": "el muslo", "shin": "la pierna",
}

# The same four tests profile._detect_marks uses, at the same values, because
# what is being changed here is the skin area they are asked about, nothing
# else.
MARK_DEV_SIGMA = 3.0
MARK_ENERGY_PCT = 80.0
MARK_DELTA_E = 10.0
MARK_ENERGY_RATIO = 1.15

# Her skin grown by this much of the square root of the person box, so that the
# island it surrounds falls inside the area being searched.  2.0% is 19 px on
# the new photograph's person box of 673x1283.
MARK_GROW_FRAC = 0.020

# A mark is small.  0.04% of the person box is the floor (345 px on the new
# photograph, where the two hearts read 371) and 3.0% the ceiling - her thigh
# rose is the largest real thing here at 2.63% (IMG_8798), 2.36% (IMG_8898)
# and 2.09% (IMG_8918), while the blobs this rejects are the hair falling over
# her chest on IMG_7818 (2.31%) and IMG_7825 (1.20%).
MARK_MIN_FRAC = 0.0004
MARK_MAX_FRAC = 0.030

# CORROBORATION.  The profile is what knows which parts of her carry marks at
# all - _aggregate_marks keeps only what two or more photographs agreed on -
# so a candidate in one of those zones is believed on the ordinary tests.  A
# candidate somewhere the profile knows nothing about has to be far more
# clearly ink.  Measured over her 25 photographs this second door admits the
# forearm script on IMG_7871 (dE 18.6, texture 1.98x) and IMG_7880 (16.6,
# 2.00x) - which the profile misses because it was seen in only one photograph
# when the profile was built - and rejects the hair over her chest on
# IMG_7818, IMG_7825, IMG_7771 and IMG_7778 and the pendant on IMG_7760.
MARK_STRICT_DELTA_E = 15.0
MARK_STRICT_ENERGY_RATIO = 1.50
MARK_STRICT_MAX_FRAC = 0.004

# A mark is a COMPACT patch: ink, a mole, a scar.  Its own area over the area
# of its convex hull separates it from the one false positive that survived
# everything else - the lace front of her bodysuit on IMG_7880, a spidery
# skeleton seen through its own cut-outs, which reads 0.15 and 0.17 while every
# one of the 20 real marks measured on her 25 photographs reads 0.30 or better
# (the lowest is the thigh rose of IMG_7871 at 0.30).  0.25 sits in that gap.
MARK_MIN_SOLIDITY = 0.25

# A mark is only in danger when the repaint zone really covers it.
MARK_INSIDE_MIN = 0.10

# The halo kept around a protected mark, as a fraction of the shorter side, so
# that the feather cannot nibble its edge: 0.008 is 10 px on a 1200 px frame.
MARK_PAD_FRAC = 0.008

# And a ceiling on the whole protected set, because a detector that ran away
# would eat the repaint zone and turn the request into a refusal.  Measured on
# the worst case her wardrobe allows - a colour change, which dresses her in
# nothing new and so protects every mark found - the largest protected set over
# her 25 photographs is 2.62% of the frame (IMG_8798, her thigh rose), then
# 2.37% (IMG_8918) and 1.27% (IMG_7880).  4% leaves those alone and still stops
# a runaway.
MARK_TOTAL_MAX = 0.04

# ------------------------------------------------------------------ her hair
#
# HER HAIR IS IDENTITY AND THE GUARANTEE WAS TRUE OF THE WRONG PIXELS.  The
# protected set has always contained the region segment.region_masks calls
# "hair", and that region is a construction: an ellipse 1.32x the face,
# everything ABOVE the face line, minus skin.  By construction it stops at the
# hairline.  On the new clothed photograph it is 7907 px - 0.41% of the frame,
# a sliver at the crown - while her hair, grown from that same sliver by her
# own colour, is 57 032 px.  Over her 25 photographs the seed runs 0.03% to
# 3.14% of the frame and the hair 0.93% to 39.14%, a median 13.9x more hair
# than was being kept, and OF THE HAIR THE OLD SHIELD DID NOT ALREADY COVER
# between 15.5% and 100% of it (mean 72.4%) lay inside the repaint zone: her
# long auburn hair falling over her left shoulder and chest was going to be
# painted over, on a request that only asked to change the clothes.  Her hair
# is the second thing a client checks after the face - its colour, its length
# and where it falls - so it belongs in the same protected core as the face and
# the hands, not in a sentence.
#
# WHAT THE DETECTOR IS.  The seed is trusted for one thing only, its COLOUR:
# those 7907 px really are her hair, and their median Lab is 1.7 to 9.7 away
# from the hair colour her profile was built from over all 25 photographs.
# Everything else is her own photograph: the pixels of that colour, inside her
# silhouette, that are not her skin, that hang together with the head, and that
# stop where her measured hair length stops.
HAIR_SEED_MIN = 200

# HER COLOUR, IN TWO PIECES RATHER THAN ONE BALL.  A single Lab distance around
# the seed's median is the wrong shape for hair: hair varies enormously in
# LIGHTNESS between the shadowed crown the seed sits on and the lit strands
# over her shoulder, and hardly at all in hue.  Measured on IMG_7580: narrow
# the lightness window to the chroma tolerance, which is the same thing as a
# ball, and the layer falls from 209 675 px to 83 363 - and the 126 312 px
# difference is the lit half of her hair.  So lightness gets a wide window (a
# strand may be 30 L above the crown or 40 below it) and chroma a tight one: 8
# units from THIS photograph's reference, whose own median chroma is 4.4 where
# she is shot in shade (IMG_8946) and 19.0 where the light is on her
# (IMG_7825).
HAIR_L_LIGHTER = 30.0
HAIR_L_DARKER = 40.0
HAIR_CHROMA_TOL = 8.0

# And the seed is checked against the profile before it is believed, because a
# hat, a hood or a bad face ellipse would hand this a reference that is not her
# hair at all.  Measured over her 25 photographs the gap is 1.7 (IMG_7760) to
# 9.7 (IMG_7839); 20 is twice the worst.
HAIR_PROFILE_MAX = 20.0

# WHERE HER HAIR ENDS, from her own profile.  ``hair.length_ratio`` is how far
# below the chin her hair reaches in TORSO LENGTHS, the median of 20 of her
# photographs (1.226), measured by identity/profile._hair_sample - so the bound
# is read in the same unit it was written in, using the same torso length.  The
# extra 0.45 of a torso is slack for a photograph where her hair falls further
# than the median.  MEASURED, IT NEVER FIRES ON HER 25: a torso can be measured
# on 15 of them, the bound lands inside the frame on 9, and removing it changes
# the layer on none - the colour and the mesh test have already stopped the
# growing higher up.  It is kept because it is the only thing in here that says
# her hair ENDS somewhere: without it a dark trouser leg or a shadow low in the
# frame is one contiguous dark region away from the head.  It is a row bound,
# so on the photographs where she is lying with her head at the bottom of the
# frame it cannot bind at all.
HAIR_REACH = 0.45

# THE ONE THING THAT TELLS HER HAIR FROM HER BLACK LACE.  On 9 of her 25
# photographs she is wearing lace lingerie whose Lab is her hair's Lab: on
# IMG_8946 the pixels this detector accepts have a median chroma of 4.4, which
# is the black lace and the shadowed hair together, and no colour rule can
# separate them (IMG_8947 reads 4.3).  Texture can, some of the time: lace is a
# MESH, so at strand scale it jumps between black thread and lit skin, while
# hair does not.  The
# test is the local standard deviation of lightness over a window of 0.8% of
# the shorter side (10 px on her 1200 px frame), and the limit is 1.3x the 90th
# percentile of that same statistic INSIDE THE SEED - her own hair's texture,
# on this photograph, rather than a number from somewhere else - with a floor
# of 4 L units so a perfectly smooth crown cannot drive the limit to zero.
# What survives is grown back by 0.8% of the shorter side, because the test
# erodes the edge strands it should keep.  Measured, it bites on 14 of her 25,
# it takes the mean share of the garment zone the hair layer swallows from
# 23.5% to 17.6%, and on the three photographs where it matters most it takes
# that share from 36.0% to 1.1% (IMG_8825), 35.5% to 1.7% (IMG_8841) and 26.2%
# to 2.2% (IMG_8898).  It does NOT rescue the six photographs where her hair
# lies directly on the lace (IMG_8944 to IMG_8950): there the two are the same
# colour AND the same texture, and that is said out loud in hair_note_es
# instead of being hidden.
HAIR_TEX_FRAC = 0.008
HAIR_TEX_PCT = 90.0
HAIR_TEX_K = 1.3
HAIR_TEX_FLOOR = 4.0
HAIR_TEX_GROW = 0.008

# Strands leave holes; this closes the ones smaller than 0.6% of the shorter
# side so the layer is a lock of hair rather than a comb.
HAIR_CLOSE_FRAC = 0.006

# WHEN THE PROTECTION COSTS THE REQUEST SOMETHING, SAY SO.  There is no cap
# here on purpose - capping would mean choosing to repaint part of her hair to
# buy back repaint area, and her hair is the thing being protected.  What there
# is, is a number above which the client is told: over her 25 photographs the
# hair layer takes 1.1% to 51.6% of the garment zone (mean 17.6%), and the
# repaint zone that survives is never smaller than 12.43% of the frame against
# a MIN_COVER of 2%, so nothing is ever refused for this.  A tenth of the
# garment is the line because a tenth of a shirt left as it was is visible in
# the delivered image; it is crossed on 15 of her 25 photographs, and the
# sentence it triggers is one she can act on - the same request on a
# photograph where her hair is not lying on the clothes does not pay it.
HAIR_ZONE_WARN = 0.10
# And when most of what it took is near-neutral - chroma under 4, which is
# black cloth and not auburn hair - the reason is named too: on IMG_8947 68.2%
# of what the layer takes out of the garment zone is that black lace, against
# 0.0% on the new clothed photograph and on 9 others.
HAIR_NEUTRAL_CHROMA = 4.0
HAIR_NEUTRAL_WARN = 0.40


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


def _lab_f32(img_bgr: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(img_bgr.astype(np.float32) / 255.0, cv2.COLOR_BGR2Lab)


def _region_bool(regions: Any, name: str, shape: tuple) -> np.ndarray | None:
    got = (regions or {}).get(name)
    if isinstance(got, np.ndarray) and got.shape[:2] == shape:
        return got > 127
    return None


def _face_box_px(face: Any, width: int, height: int) -> list[float] | None:
    """The detected face rectangle as [x, y, w, h] in this frame's pixels."""
    box = (face or {}).get("bbox") if isinstance(face, dict) else None
    if not isinstance(box, (list, tuple)) or len(box) < 4:
        return None
    x, y, w, h = [float(v) for v in list(box)[:4]]
    if max(x, y, w, h) <= 1.5:
        x, y, w, h = x * width, y * height, w * width, h * height
    return [x, y, w, h] if w > 1.0 and h > 1.0 else None


def _her_skin(img_bgr: np.ndarray, person: np.ndarray, regions: Any,
              profile: Any):
    """The pixels that are HER skin, by her own profile's numbers.

    The same reference and the same two tolerances identity/verify.py gates a
    generated image on, used in the other direction: what in this photograph is
    the skin the profile was built from.  On the new photograph that is 13.3%
    of the frame - the warm grey top fails on chroma (21.8 against a limit of
    10.59) and the black skirt on lightness - which is exactly why a mark
    detector fed this can see anything at all.
    """
    lab = _lab_f32(img_bgr)
    prof = profile if isinstance(profile, dict) else {}
    skin_d = prof.get("skin") if isinstance(prof.get("skin"), dict) else {}
    ref = [float(v) for v in list(skin_d.get("lab_mean") or [])[:3]]
    if len(ref) < 3:
        return None, lab
    th = prof.get("thresholds") if isinstance(prof.get("thresholds"), dict) else {}
    chroma = float(th.get("chroma_max_eff") or th.get("chroma_max") or 6.0)
    light = float(th.get("delta_l_max_eff") or th.get("delta_l_max") or 22.0)
    d_chroma = np.sqrt((lab[:, :, 1] - ref[1]) ** 2 + (lab[:, :, 2] - ref[2]) ** 2)
    d_light = np.abs(lab[:, :, 0] - ref[0])
    skin = (person > 127) & (d_chroma <= chroma) & (d_light <= light)
    for name in ("hair", "face", "hands"):
        got = _region_bool(regions, name, skin.shape)
        if got is not None:
            skin &= ~got
    return skin, lab


def visible_marks(img_bgr: np.ndarray, pose: Any, person: Any, regions: Any,
                  profile: Any, person_bbox: Any = None, hands: Any = None):
    """Every permanent mark this photograph actually shows, with its pixels.

    Returns ``(marks, labels)``.  Each mark carries its box, its zone and the
    label index into ``labels`` that holds its exact pixels, so that a caller
    protects the ink and not a rectangle drawn around it.
    """
    if not isinstance(person, np.ndarray) or not isinstance(img_bgr, np.ndarray):
        return [], None
    skin, lab = _her_skin(img_bgr, person, regions, profile)
    if skin is None or int(skin.sum()) < 800:
        return [], None
    height, width = img_bgr.shape[:2]
    box = [float(v) for v in list(person_bbox or [0, 0, width, height])[:4]]
    person_area = max(1.0, box[2] * box[3])

    grow = _odd(MARK_GROW_FRAC * (person_area ** 0.5), low=5)
    area = cv2.dilate(skin.astype(np.uint8) * 255,
                      cv2.getStructuringElement(cv2.MORPH_ELLIPSE,
                                                (grow, grow))) > 127
    area &= person > 127
    for name in ("hair", "face", "hands"):
        got = _region_bool(regions, name, area.shape)
        if got is not None:
            area &= ~got
    # And the hands as the hand model really sees them, not only as the
    # skeleton capsule guesses them.  A hand is black in the mask anyway, so a
    # "mark" found on one means nothing - and on IMG_8798 that is precisely
    # what happened: the hand holding her hair scored 51% inside the detected
    # hand and was about to be protected as a torso mark.
    if isinstance(hands, np.ndarray) and hands.shape[:2] == area.shape:
        area &= ~(hands > 127)

    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    energy = cv2.GaussianBlur(np.abs(cv2.Laplacian(gray, cv2.CV_32F, ksize=3)),
                              (0, 0), 2.0)
    sample = lab[skin]
    if sample.shape[0] > 200000:
        sample = sample[:: int(sample.shape[0] // 200000) + 1]
    med = np.median(sample, axis=0).astype(np.float32)
    mad = (np.median(np.abs(sample - med), axis=0) * 1.4826).astype(np.float32)
    sd = np.maximum(mad, np.array([2.5, 1.2, 1.2], np.float32))
    dev = np.sqrt((((lab - med) / sd) ** 2).sum(axis=2))
    e_thr = max(float(np.percentile(energy[skin], MARK_ENERGY_PCT)), 1.0)

    cand = (area & (dev > MARK_DEV_SIGMA) & (energy > e_thr)).astype(np.uint8) * 255
    k3 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    k7 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    cand = cv2.morphologyEx(cand, cv2.MORPH_OPEN, k3)
    cand = cv2.morphologyEx(cand, cv2.MORPH_CLOSE, k7)
    n_labels, labels, stats, _ = cv2.connectedComponentsWithStats(cand, connectivity=8)

    known: set[str] = set()
    for entry in ((profile or {}).get("marks") or []):
        if isinstance(entry, dict):
            zone = MARK_ZONES.get(str(entry.get("region") or ""))
            if zone:
                known.add(zone)

    found: list[dict] = []
    for idx in range(1, n_labels):
        size = float(stats[idx, cv2.CC_STAT_AREA])
        if size < max(40.0, MARK_MIN_FRAC * person_area) or \
                size > MARK_MAX_FRAC * person_area:
            continue
        comp = labels == idx
        ring = cv2.dilate(comp.astype(np.uint8) * 255, k7, iterations=2) > 127
        ring &= ~comp
        ring_skin = ring & skin
        n_ring, n_ring_skin = int(ring.sum()), int(ring_skin.sum())
        if n_ring < 20 or n_ring_skin < 30 or n_ring_skin / float(n_ring) < 0.5:
            continue
        c_lab = lab[comp].mean(axis=0)
        r_lab = lab[ring_skin].mean(axis=0)
        delta_e = float(np.sqrt(((c_lab - r_lab) ** 2).sum()))
        if delta_e < MARK_DELTA_E:
            continue
        c_energy = float(energy[comp].mean())
        r_energy = float(energy[ring_skin].mean())
        ratio = c_energy / max(r_energy, 0.5)
        if ratio < MARK_ENERGY_RATIO:
            continue
        solidity = 1.0
        shapes = _safe(cv2.findContours, comp.astype(np.uint8),
                       cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        outline = max((shapes or [[]])[0] or [], key=cv2.contourArea, default=None)
        if outline is not None and len(outline) >= 3:
            hull_area = float(cv2.contourArea(cv2.convexHull(outline)))
            if hull_area > 0:
                solidity = float(cv2.contourArea(outline)) / hull_area
        if solidity < MARK_MIN_SOLIDITY:
            continue
        x = int(stats[idx, cv2.CC_STAT_LEFT])
        y = int(stats[idx, cv2.CC_STAT_TOP])
        w = int(stats[idx, cv2.CC_STAT_WIDTH])
        h = int(stats[idx, cv2.CC_STAT_HEIGHT])
        region = _safe(profile_mod._nearest_region, (x + w / 2.0, y + h / 2.0),
                       pose or {}, width, height) or "unknown"
        zone = MARK_ZONES.get(str(region), "")
        share = size / person_area
        if zone and zone in known:
            origin = "perfil"
        elif (zone and delta_e >= MARK_STRICT_DELTA_E
              and ratio >= MARK_STRICT_ENERGY_RATIO
              and share <= MARK_STRICT_MAX_FRAC):
            origin = "foto"
        else:
            continue
        found.append({"label": int(idx), "zona": zone, "region": str(region),
                      "caja": [x, y, w, h], "px": int(size),
                      "parte": round(share, 5), "delta_e": round(delta_e, 1),
                      "textura": round(ratio, 2), "solidez": round(solidity, 2),
                      "origen": origin})
    found.sort(key=lambda m: -m["px"])
    return found, labels


def her_hair(img_bgr: np.ndarray, person: Any, regions: Any, face: Any,
             pose: Any, profile: Any, change: Any = None):
    """The hair anyone can SEE, grown out of the sliver the segmenter finds.

    Returns ``(layer, info)``: a uint8 0/255 mask to add to the protected core,
    or ``None`` when this photograph cannot support the measurement - in which
    case ``info["motivo"]`` says why, and the caller keeps the old sliver
    rather than pretending.

    ``change`` is the repaint zone this request wants, and it is passed in so
    that the price of the protection - how much of the garment her hair takes
    out of it, and how much of THAT is near-neutral black cloth rather than
    auburn hair - is measured here, in the one place that knows what a hair
    pixel is.
    """
    info: dict[str, Any] = {"px": 0, "foto": 0.0, "zona": 0.0, "neutro": 0.0,
                            "motivo": ""}
    if not isinstance(img_bgr, np.ndarray) or not isinstance(person, np.ndarray):
        info["motivo"] = "sin foto o sin silueta"
        return None, info
    height, width = img_bgr.shape[:2]
    seed = _region_bool(regions, "hair", (height, width))
    if seed is None or int(seed.sum()) < HAIR_SEED_MIN:
        info["motivo"] = "no se ve el nacimiento del pelo"
        return None, info
    inside = person > 127
    lab = _lab_f32(img_bgr)
    ref = np.median(lab[seed], axis=0).astype(np.float32)
    info["color"] = [round(float(v), 1) for v in ref]

    prof = profile if isinstance(profile, dict) else {}
    hair_p = prof.get("hair") if isinstance(prof.get("hair"), dict) else {}
    ref_prof = [float(v) for v in list(hair_p.get("lab_mean") or [])[:3]]
    if len(ref_prof) == 3:
        gap = float(np.linalg.norm(ref - np.asarray(ref_prof, np.float32)))
        info["contra_perfil"] = round(gap, 1)
        if gap > HAIR_PROFILE_MAX:
            info["motivo"] = ("lo que hay sobre tu cara no tiene tu color de "
                              "pelo (%.0f de distancia)" % gap)
            return None, info

    d_light = lab[:, :, 0] - float(ref[0])
    d_chroma = np.sqrt((lab[:, :, 1] - float(ref[1])) ** 2
                       + (lab[:, :, 2] - float(ref[2])) ** 2)
    cand = (inside & (d_light <= HAIR_L_LIGHTER) & (d_light >= -HAIR_L_DARKER)
            & (d_chroma <= HAIR_CHROMA_TOL))
    # Her skin by her own profile, out: the same envelope visible_marks reads,
    # so hair and marks cannot both claim the same pixel.
    skin, _lab = _her_skin(img_bgr, person, regions, profile)
    if skin is not None:
        cand &= ~skin

    # Where her hair ends, from the profile's measured length.
    box = _face_box_px(face, width, height)
    torso = _safe(profile_mod._torso_len_px, pose, width, height)
    ratio = hair_p.get("length_ratio")
    if box is not None and torso and ratio is not None:
        limit = box[1] + box[3] + (float(ratio) + HAIR_REACH) * float(torso)
        rows = np.arange(height, dtype=np.float32).reshape(height, 1)
        cand &= rows <= limit
        info["fondo_fila"] = int(limit)

    # And the mesh test that tells the lace from the hair.
    window = _odd(HAIR_TEX_FRAC * min(height, width))
    light = np.ascontiguousarray(lab[:, :, 0])
    mean = cv2.blur(light, (window, window))
    var = cv2.blur(light * light, (window, window)) - mean * mean
    rough = np.sqrt(np.maximum(var, 0.0))
    rough_max = max(HAIR_TEX_FLOOR,
                    HAIR_TEX_K * float(np.percentile(rough[seed], HAIR_TEX_PCT)))
    info["textura"] = round(rough_max, 2)
    grow = _odd(HAIR_TEX_GROW * min(height, width))
    smooth = cv2.dilate((cand & (rough <= rough_max)).astype(np.uint8),
                        cv2.getStructuringElement(cv2.MORPH_ELLIPSE,
                                                  (grow, grow))) > 0
    cand &= smooth

    # Hair hangs together and it hangs from the head: whatever is left has to
    # reach the seed to count.
    cand |= seed
    close = _odd(HAIR_CLOSE_FRAC * min(height, width))
    solid = cv2.morphologyEx(cand.astype(np.uint8), cv2.MORPH_CLOSE,
                             cv2.getStructuringElement(cv2.MORPH_ELLIPSE,
                                                       (close, close)))
    _count, labels = cv2.connectedComponents(solid, 8)
    keep = np.unique(labels[seed])
    keep = keep[keep > 0]
    hair = np.isin(labels, keep) if keep.size else seed

    info["px"] = int(hair.sum())
    info["foto"] = round(float(hair.sum()) / float(height * width), 4)
    if isinstance(change, np.ndarray) and change.shape[:2] == hair.shape:
        eaten = hair & (change > 127)
        area = float(np.count_nonzero(change))
        info["zona"] = round(float(eaten.sum()) / max(1.0, area), 4)
        chroma = np.sqrt(lab[:, :, 1] ** 2 + lab[:, :, 2] ** 2)
        info["neutro"] = round(float((eaten & (chroma < HAIR_NEUTRAL_CHROMA)).sum())
                               / max(1.0, float(eaten.sum())), 4)
    return hair.astype(np.uint8) * 255, info


def hair_note_es(hair: Any) -> str:
    """What the mask does with her hair, and what that costs the garment.

    THE HONESTY RULE THE MARKS ALREADY FOLLOW.  A mark under the new garment is
    repainted on purpose and said out loud; hair is the other way round - it is
    never repainted - so what has to be said out loud is the price: the part of
    the garment her hair sits on cannot be repainted either, and on the six
    photographs where her hair and her lace are the same colour that part is
    large.  A garment that is MEANT to cover her hair - a hood, a high collar,
    a scarf - is a real conflict rather than a cost, and it is named as one.
    """
    data = hair if isinstance(hair, dict) else {}
    if not data:
        return ""
    if not int(data.get("px") or 0):
        motivo = _text(data.get("motivo"))
        return ("En esta foto no se puede aislar tu pelo%s, asi que solo queda "
                "fuera de la zona el de la cabeza y el que te cae sobre la "
                "ropa se repinta."
                % (" (%s)" % motivo if motivo else ""))
    parts = ["Tu pelo no se repinta: ocupa el %.1f%% de la foto y se copia "
             "entero de tu original." % (100.0 * float(data.get("foto") or 0.0))]
    zone = float(data.get("zona") or 0.0)
    if zone >= HAIR_ZONE_WARN:
        parts.append("Como te cae por encima de la ropa, el %.0f%% de la zona "
                     "de ropa se queda como esta." % (100.0 * zone))
        if float(data.get("neutro") or 0.0) >= HAIR_NEUTRAL_WARN:
            parts.append("Aqui tu pelo y tu ropa tienen el mismo tono oscuro y "
                         "no se pueden separar por color: si quieres que la "
                         "ropa cambie del todo, elige una foto donde el pelo "
                         "no te caiga sobre ella.")
    if data.get("cuello_cubierto"):
        parts.append("Ademas pediste una prenda de cuello alto o con capucha: "
                     "como tu pelo no se repinta, la prenda no va a poder "
                     "taparlo y el pelo va a quedar por encima.")
    return " ".join(parts)

# ------------------------------------------------- what the garment will hide
#
# A MARK UNDER THE NEW GARMENT CANNOT SURVIVE, and protecting it would be worse
# than losing it: the mask would keep her own pixels there, so the delivered
# image would show a crisp white shirt with a hole in it and her bare chest
# inside the hole.  A shirt covers a chest tattoo.  So the rule is: protect
# what the requested garment leaves bare, and SAY what it covers.
#
# The words below are read off the garment's own English fragment, the same
# text the engine is sent, plus the automatic completion prompt.outfit_plan
# adds - a top chosen alone gets DEFAULT_BOTTOM under it, a bottom chosen alone
# gets DEFAULT_TOP over it - because a mark does not care whether the trousers
# were her idea or the robot's.
_SLEEVE_BARE = ("sleeveless", "strapless", "tank top", "camisole", "spaghetti",
                "halter", "bandeau", "tube top", "sin mangas", "tirantes")
_SLEEVE_SHORT = ("t-shirt", "tshirt", "short sleeve", "short-sleeve", "tee",
                 "polo", "camiseta", "manga corta")
_SLEEVE_LONG = ("long sleeve", "long-sleeve", "coat", "jacket", "blazer",
                "sweater", "jumper", "cardigan", "suit", "tuxedo", "trench",
                "shirt", "blouse", "kimono", "poncho", "abrigo", "chaqueta",
                "camisa", "blusa", "jersey", "gabardina", "esmoquin", "sastre",
                "manga larga")
_NECK_COVER = ("turtleneck", "polo neck", "high neck", "roll neck", "hood",
               "scarf", "cuello alto", "bufanda", "capucha")
_LEG_LONG = ("floor length", "full length", "to the ankle", "to the floor",
             "at the ankle", "wide leg", "maxi", "trousers", "jeans",
             "pantalon", "vaquero", "largo")
_LEG_KNEE = ("midi", "below the knee", "pencil skirt", "knee length",
             "por la rodilla")
_LEG_SHORT = ("mini", "above the knee", "shorts", "hot pants", "bermuda",
              "corto")

_MARK_ZONE_ORDER = ("torso", "neck", "upper_arm", "forearm", "thigh", "shin")


def _word_in(text: str, roots) -> bool:
    return any(root in text for root in roots)


def garment_cover(choices: Any) -> dict:
    """Which body zones the requested garment will hide, in three states.

    ``cubierta`` the garment is over it, ``descubierta`` it stays bare,
    ``desconocida`` the wardrobe entry does not say.  Unknown is treated as
    covered everywhere a decision has to be taken, because the mask can promise
    a mark only when it knows the garment leaves it showing; promising on a
    guess is how a hole gets painted into a dress.
    """
    state = {zone: "descubierta" for zone in _MARK_ZONE_ORDER}
    chosen = _safe(prompt_mod.normalize_options, choices) or {}
    garments: list[dict] = []
    for group, opts in chosen.items():
        if prompt_mod.canon_group(group) != "outfit":
            continue
        for opt in (opts or []):
            if isinstance(opt, dict):
                garments.append(opt)
    if not garments:
        # Nothing new is worn: a colour change, a sheerer fabric, shoes or an
        # accessory all leave her skin where it is, so every mark that shows
        # now still shows afterwards.
        return state

    kinds, bits = set(), []
    for opt in garments:
        info = _safe(catalog_mod.garment_info, opt) or {}
        kinds.add(str(info.get("kind") or "unknown"))
        for field in ("prompt", "label_en", "label", "value"):
            value = opt.get(field)
            if value:
                bits.append(str(value))
        if info.get("bottom"):
            bits.append(str(info["bottom"]))
        if info.get("lower"):
            bits.append(str(info["lower"]))
    # The halves prompt.outfit_plan fills in when the request only dressed one.
    if kinds and kinds <= {"top"}:
        bits.append(catalog_mod.DEFAULT_BOTTOM)
    if kinds and kinds <= {"bottom"}:
        bits.append(catalog_mod.DEFAULT_TOP)
    text = " ".join(bits).lower()

    # The torso: every path through outfit_plan ends with something over the
    # chest - a top, a complete outfit, or DEFAULT_TOP added to a bare bottom -
    # and the prompt itself says "dress the subject in this outfit and in
    # nothing else".  Only a garment nobody could classify leaves it open.
    state["torso"] = "desconocida" if kinds == {"unknown"} else "cubierta"
    state["neck"] = "cubierta" if _word_in(text, _NECK_COVER) else "descubierta"

    # Order matters, and it cost a measurement to notice: "t-shirt" contains
    # "shirt", so testing the long-sleeve words first filed
    # vaqueros_camiseta - jeans and a plain white cotton t-shirt - as covering
    # the forearm, and her forearm script would have been repainted under a
    # garment that leaves the forearm bare.  An explicit "long sleeve" still
    # wins over everything, because that is the phrase that means it.
    if _word_in(text, _SLEEVE_BARE):
        state["upper_arm"] = state["forearm"] = "descubierta"
    elif _word_in(text, ("long sleeve", "long-sleeve", "manga larga")):
        state["upper_arm"] = state["forearm"] = "cubierta"
    elif _word_in(text, _SLEEVE_SHORT):
        state["upper_arm"], state["forearm"] = "cubierta", "descubierta"
    elif _word_in(text, _SLEEVE_LONG):
        state["upper_arm"] = state["forearm"] = "cubierta"
    else:
        state["upper_arm"] = state["forearm"] = "desconocida"

    if _word_in(text, _LEG_SHORT):
        state["thigh"] = state["shin"] = "descubierta"
    elif _word_in(text, _LEG_LONG):
        state["thigh"] = state["shin"] = "cubierta"
    elif _word_in(text, _LEG_KNEE):
        state["thigh"], state["shin"] = "cubierta", "descubierta"
    else:
        state["thigh"] = state["shin"] = "desconocida"
    return state


def upload_report(mask_path: Any, img_bgr: Any, person: Any, regions: Any,
                  profile: Any, full_size: Any) -> dict:
    """The picture the provider will really review, measured before paying.

    THE MEASUREMENT THAT WAS TAKEN ON THE WRONG PICTURE, TAKEN ON THE RIGHT
    ONE.  Before the paid call of 2026-09-05 the case for spending was written
    out in full and every number in it described her whole photograph: bare
    skin inside the repaint zone down from 11.78% of the frame to 7.47%, a
    36.6% relative improvement from choosing a clothed source.  providers/fal
    did not send her whole photograph.  It cut the upload to the mask's
    bounding box - which is the clothing zone, the most skin-dense part of any
    picture of a person, and which starts at her chin - so what fal reviewed
    was a headless torso holding 8.56% of her head box and 0% of her face
    rectangle, and the improvement it really saw was 27.50% bare skin down to
    20.09%, not 11.78% down to 7.00%.

    THE CROP IS GONE (see "THERE IS NO CROP" in providers/fal.py), so what is
    measured here is now her whole frame, and that is the point of still
    measuring it: the numbers on the estimate screen and the numbers the
    reviewer sees are the same numbers again, which is the property that was
    missing when the money moved.  It needs the mask, her photograph and her
    profile's own skin envelope - all three already in hand when the mask is
    drawn - so it costs nothing extra and is written beside the mask for the
    estimate to read back for free.
    """
    path = Path(str(mask_path))
    mask = _safe(cv2.imread, str(path), cv2.IMREAD_GRAYSCALE)
    if not isinstance(mask, np.ndarray) or mask.size == 0:
        # No mask on disk is no upload to describe.  Empty, and the caller
        # says nothing rather than guessing.
        return {}
    fh, fw = mask.shape[:2]
    zone = mask > 127
    out: dict[str, Any] = {
        "completa": [int(fw), int(fh)],
        "enviado": [int(fw), int(fh)],
        "envio_completo": True,
        "envio_frac": 1.0,
        "zona_envio": round(float(np.count_nonzero(zone))
                            / float(fw * fh), 4),
    }
    # Her rectangles, from the record this mask was written with.  They are
    # read from disk rather than recomputed because the record is what the rest
    # of the system trusts, and because a second reading is a second chance to
    # disagree with the first.
    head, face = [], []
    try:
        rec = json.loads(_binding_path(path).read_text(encoding="utf-8"))
        head = [int(v) for v in (rec.get("cabeza") or [])]
        # "cara" meant the HEAD box in records written before the two were
        # told apart, and reading one of those as a face rectangle would say
        # "0 pixels of her face are repainted" about her hairline.  A record
        # that carries "cabeza" is a record that knows the difference; one
        # that does not is treated as not knowing where her face is, which is
        # the honest answer and costs only a sentence on the estimate.
        face = ([int(v) for v in (rec.get("cara") or [])]
                if rec.get("cabeza") else [])
    except Exception:                                     # noqa: BLE001
        head, face = [], []

    def _box(name, box):
        if len(box) != 4:
            return
        area = max(1, (box[2] - box[0]) * (box[3] - box[1]))
        out["%s_dentro_px" % name] = int(area)
        # It all travels now, so this is 1.0 by construction; it is written
        # anyway because it is the number that was 0.0856 when the crop was
        # announced as keeping her face at home.
        out["%s_dentro_frac" % name] = 1.0
        out["%s_repintada_px" % name] = int(np.count_nonzero(
            zone[box[1]:box[3], box[0]:box[2]]))

    out["cara_conocida"] = bool(len(face) == 4)
    _box("cara", face)
    _box("cabeza", head)
    if not isinstance(img_bgr, np.ndarray) or not isinstance(person, np.ndarray):
        # Reading her skin costs 0.4 s and needs her photograph; a mask being
        # read back off disk has neither, so the geometry is answered alone
        # rather than making the estimate wait for a number it can survive
        # without.
        out["aviso"] = upload_note_es(out)
        return out
    skin, _lab = _her_skin(img_bgr, person, regions, profile)
    if skin is None:
        out["aviso"] = upload_note_es(out)
        return out
    sh, sw = skin.shape[:2]
    small_zone = cv2.resize(mask, (sw, sh),
                            interpolation=cv2.INTER_NEAREST) > 127
    area = float(sw * sh)
    whole = float(np.count_nonzero(skin)) / area
    out["piel_foto"] = round(whole, 4)
    # What travels IS the photograph now, so these two are the same number by
    # construction.  Both are kept because the pair is the evidence: when they
    # differed by 2.8x nobody noticed, and a row where they differ again is a
    # crop that has come back.
    out["piel_envio"] = out["piel_foto"]
    out["concentracion"] = 1.0
    out["piel_zona"] = round(
        float(np.count_nonzero(skin & small_zone)) / area, 4)
    out["aviso"] = upload_note_es(out)
    return out


def upload_note_es(envio: Any) -> str:
    """What leaves the machine and what it looks like, in her language.

    Said before she pays, because the one thing the two blocked calls proved is
    that the provider judges the picture it is handed and nothing else.
    """
    data = envio if isinstance(envio, dict) else {}
    if not data.get("envio_completo"):
        return ""
    parts = ["Sale tu foto entera, con tu cabeza y tus hombros: es lo que "
             "el modelo necesita para ajustar la ropa a tu cuerpo, y es lo "
             "que revisa el proveedor."]
    if "zona_envio" in data:
        parts.append("De esa foto se repinta el %.0f%%."
                     % (100.0 * float(data.get("zona_envio") or 0.0)))
    if "piel_envio" in data:
        parts.append("Tu piel descubierta es el %.0f%% de lo que se envia, y "
                     "el %.0f%% queda dentro de la zona que se repinta."
                     % (100.0 * float(data.get("piel_envio") or 0.0),
                        100.0 * float(data.get("piel_zona") or 0.0)))
    if data.get("cara_conocida"):
        # Counted on the DETECTED FACE RECTANGLE, not on the head box: the
        # head box includes her hair, the mask really does repaint hair, and a
        # promise about her face must not be measured on her hairline.
        repaint = int(data.get("cara_repintada_px") or 0)
        if repaint:
            parts.append("Aviso: la zona a repintar toca %d pixeles del "
                         "recuadro de tu cara. Eso no deberia pasar y se "
                         "revisa antes de enviar nada." % repaint)
        else:
            parts.append("Tu cara viaja entera y no se repinta: la mascara la "
                         "deja fuera por completo (0 pixeles del recuadro de "
                         "tu cara dentro de la zona) y encima se vuelven a "
                         "poner tus propios pixeles.")
    return " ".join(parts)


def marks_note_es(marks: Any) -> str:
    """What the mask keeps of her marks and what it cannot, in her language."""
    kept, hidden, doubt, over = [], [], [], []
    for mark in (marks or []):
        if not isinstance(mark, dict):
            continue
        label = MARK_ZONE_ES.get(str(mark.get("zona") or ""), "tu piel")
        state = str(mark.get("estado") or "")
        if state == "protegida" and label not in kept:
            kept.append(label)
        elif state == "tapada" and label not in hidden:
            hidden.append(label)
        elif state == "dudosa" and label not in doubt:
            doubt.append(label)
        elif state == "excedida" and label not in over:
            over.append(label)
    parts: list[str] = []
    if kept:
        parts.append("Tus marcas en %s se conservan: %s no se repinta%s, se "
                     "copia%s de tu propia foto."
                     % (_join_es(kept), "esa zona" if len(kept) == 1
                        else "esas zonas", "" if len(kept) == 1 else "n",
                        "" if len(kept) == 1 else "n"))
    if hidden:
        parts.append("La ropa que pediste tapa %s, asi que la marca que llevas "
                     "ahi no puede sobrevivir: una prenda encima de un tatuaje "
                     "lo cubre, y no se te va a decir lo contrario. Si quieres "
                     "verla, elige una prenda que deje esa zona al aire."
                     % _join_es(hidden))
    if doubt:
        parts.append("De la prenda elegida no se sabe si deja a la vista %s, "
                     "asi que esa zona se repinta y la marca que llevas ahi "
                     "puede perderse." % _join_es(doubt))
    if over:
        parts.append("En esta foto hay demasiada superficie marcada para "
                     "protegerla entera (el limite es el 3%% del cuadro), asi "
                     "que las marcas mas pequenas de %s se repintan."
                     % _join_es(over))
    return " ".join(parts)


def _join_es(items: Any) -> str:
    names = [str(i) for i in (items or []) if str(i or "").strip()]
    if not names:
        return ""
    if len(names) == 1:
        return names[0]
    return "%s y %s" % (", ".join(names[:-1]), names[-1])


def _binding_path(mask_path: Any) -> Path:
    return Path(str(mask_path)).with_suffix(BINDING_EXT)


def _write_binding(mask_path: Any, source_path: Any, size: list,
                   cover: float = 0.0, regions: Any = (),
                   head: Any = None, marks: Any = None,
                   envio: Any = None, face: Any = None,
                   pelo: Any = None) -> None:
    """Record which photograph this mask was drawn on, beside the mask.

    ``cover`` and ``regions`` ride along so that a mask found on disk can be
    REUSED without redrawing it: the estimate needs the covered fraction for
    its sentence and the run needs the regions for its ficha, and reading them
    back out of the PNG would mean recomputing what has already been measured.

    ``head`` is the rectangle her face and hair occupy in this photograph, in
    the same full resolution pixels as the mask, and it is written here for a
    reason that cost 0.100 USD to learn.  The upload used to be cut to the
    mask's bounding box, and that cut was announced as "her face never leaves
    the machine" and then measured afterwards at 8.56% of her face box: her
    mouth, chin, jaw and both earrings, because the zone starts at her chin.
    The crop is gone and her whole photograph travels, so this rectangle is no
    longer a clamp - it is the ruler.  ``upload_report`` reads it back to say
    how much of her face is in the picture fal reviews (all of it) and how
    much of it the mask would repaint (0 px, on all 25 of her photographs),
    and a number nobody can read is a number nobody checks.
    """
    data = {"fuente": Path(str(source_path)).name,
            "sha256": loader.file_sha256(str(source_path)),
            "completa": [int(size[0]), int(size[1])],
            "cubre": round(float(cover or 0.0), 4),
            "regiones": [str(r) for r in (regions or [])]}
    if head:
        # Her face AND her hair: the outline of her head, which is what the
        # protected core is built from.
        data["cabeza"] = [int(v) for v in head]
    if face:
        # And the DETECTED FACE RECTANGLE alone, which is a different
        # rectangle and the only one the guarantee is about.  Writing one and
        # calling it the other is how "her face is not repainted" ended up
        # being said over a number counted on her hairline: measured on
        # IMG_8949 the head box holds 333366 mask pixels - her hair, which the
        # mask really does repaint - while the face rectangle holds 0, on all
        # 25 of her photographs.  Two rectangles, two numbers, no rounding of
        # one into the other.
        data["cara"] = [int(v) for v in face]
    if marks:
        # So that a mask read back off disk can still say which of her marks it
        # kept and which the garment covers, without measuring the photograph
        # a second time and possibly answering differently.
        data["marcas"] = [{"zona": str(m.get("zona") or ""),
                           "estado": str(m.get("estado") or ""),
                           "caja": [int(v) for v in (m.get("caja") or [])],
                           "origen": str(m.get("origen") or "")}
                          for m in marks if isinstance(m, dict)]
    if envio:
        # What the provider will really be shown (upload_report above), stored
        # so the estimate can quote it without measuring the photograph again:
        # a mask read back off disk costs 0.005 s and this costs 0.4 s, and a
        # number the estimate cannot afford is a number nobody sees before the
        # money moves - which is exactly how it went wrong the first time.
        data["envio"] = envio
    if pelo:
        # And what it does with her hair, for the same reason: the sentence the
        # client reads about her hair is measured on her photograph once, and a
        # mask reused from disk has to say the same thing.
        data["pelo"] = pelo
    _binding_path(mask_path).write_text(json.dumps(data), encoding="utf-8")


def _binding_data(mask_path: Any, source_path: Any) -> dict:
    """The record beside an existing mask, but only if it is THIS photograph's.

    Empty means "there is no mask here that can be trusted", which is the only
    answer that may lead to reusing one.  The digest is the same check compose()
    makes before it believes a single pixel; making it here as well is what lets
    the run reuse the estimate's mask without ever reusing somebody else's.
    """
    path = Path(str(mask_path))
    record = _binding_path(path)
    if not path.is_file() or not record.is_file():
        return {}
    try:
        data = json.loads(record.read_text(encoding="utf-8"))
    except Exception:                                    # noqa: BLE001
        return {}
    if not isinstance(data, dict):
        return {}
    digest = str(data.get("sha256") or "")
    mine = _safe(loader.file_sha256, str(source_path))
    if digest and mine and digest != mine:
        return {}
    return data


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
            parts.append("%s: %s" % (
                BLOCKED_ES_NOUN.get(group, prompt_mod.group_label_es(group)),
                why))
        return {
            "safe": False, "regions": regions, "blocked": blocked,
            "groups": groups,
            "reason": ("Tu rostro se va a volver a generar porque pediste "
                       "cambiar %s. Se comprobara la identidad antes de "
                       "ensenarte nada." % "; ".join(parts)),
        }
    return {
        "safe": True, "regions": regions, "blocked": [], "groups": groups,
        "reason": ("Tu rostro NO se va a generar: solo se repinta %s, y tu "
                   "cara y tus manos se copian de tu propia foto."
                   % ", ".join(_region_label(r) for r in regions)),
    }


def change_label_es(groups: Any, regions: Any = ()) -> str:
    """What is being repainted, in one noun phrase: "la ropa", "el fondo"."""
    wanted = [str(g) for g in (groups or [])]
    if "outfit" in wanted:
        # The garment itself is being replaced, so "la ropa y el color de la
        # ropa y la transparencia de la tela" is one change said three times.
        wanted = [g for g in wanted
                  if g not in ("clothing_color", "transparency")]
    names: list[str] = []
    for group in wanted:
        label = CHANGE_ES.get(str(group))
        if label and label not in names:
            names.append(label)
    if not names:
        names = [_region_label(r) for r in (regions or [])] or \
            ["la zona que cambias"]
    if len(names) == 1:
        return names[0]
    return "%s y %s" % (", ".join(names[:-1]), names[-1])


def masked_sentence(groups: Any, cover: float, regions: Any = ()) -> str:
    """The sentence for the safe path, with the number that makes it checkable."""
    zone = change_label_es(groups, regions)
    if cover > 0:
        return ("Tu rostro no se va a volver a generar: se repinta solo %s "
                "(%.0f%% de la foto) y tu cara y tus manos se copian de tu "
                "propia foto." % (zone, 100.0 * float(cover)))
    return ("Tu rostro no se va a volver a generar: se repinta solo %s y tu "
            "cara y tus manos se copian de tu propia foto." % zone)


# WHAT THE MASK CANNOT KEEP, SAID OUT LOUD.
# The face shield keeps her face, her hair, her hands and the earrings that
# hang beside them, because those are black in the mask and compose() writes
# them back from her own file.  Nothing keeps a mark that lies on the body:
# her three tattoos - script and a heart below the left collarbone, script down
# the inside of the right forearm, a rose sleeve across the left thigh - are
# all INSIDE the repaint zone, and the delivered image of 2026-09-04 came back
# without any of them.  The prompt asks for them anyway ("same tattoos, moles
# and marks in the same places"), which is asking a model to reproduce
# something it was never shown: on the masked path those pixels are the ones
# painted over.
# This is the OTHER half of the answer, and it is the older one.  ``visible_marks``
# above now measures her marks on the photograph itself and forces the ones the
# requested garment leaves bare to black, so those really do come back; what is
# left over is the mark the garment covers, which no mask can rescue.  This
# reads the same danger off the reading's ``preserve`` text - useful because the
# text names marks the pixels did not resolve - and ``kept`` takes back out of
# it whatever the mask has just promised to keep, so the two halves cannot
# contradict each other on the same screen.
_MARK_WORDS = ("tattoo", "tatuaje", "mole", "lunar", "scar", "cicatriz",
               "birthmark", "freckle", "peca", "piercing", "mark", "marca")

# Where a mark sits, in the words the reader uses, mapped to the region that
# repaints it.  A mark on a part nobody is repainting survives, and saying it
# would be a false alarm.
_MARK_PLACE = {
    "upper_body": ("chest", "pecho", "breast", "collarbone", "clavicula",
                   "torso", "stomach", "abdomen", "belly", "vientre", "back",
                   "espalda", "shoulder", "hombro", "rib", "costilla",
                   "sternum", "neckline", "escote"),
    "arms": ("arm", "brazo", "forearm", "antebrazo", "wrist", "muneca",
             "elbow", "codo", "bicep", "sleeve", "manga"),
    "lower_body": ("hip", "cadera", "waist", "cintura", "groin", "ingle",
                   "buttock", "gluteo"),
    "legs": ("leg", "pierna", "thigh", "muslo", "calf", "pantorrilla",
             "knee", "rodilla", "ankle", "tobillo", "shin", "espinilla"),
    "persona": ("chest", "pecho", "torso", "arm", "brazo", "forearm",
                "antebrazo", "leg", "pierna", "thigh", "muslo", "shoulder",
                "hombro", "back", "espalda", "stomach", "abdomen"),
}


# The same places again, keyed by the zone the mask now protects, so that a
# mark the mask KEEPS is not also announced as lost.  Before this, a run that
# protected the forearm script still printed "lo que la mascara NO puede
# conservar: tattoo on the forearm", which is now simply untrue.
_MARK_ZONE_WORDS: dict[str, tuple[str, ...]] = {
    "torso": ("chest", "pecho", "breast", "collarbone", "clavicula", "torso",
              "stomach", "abdomen", "belly", "vientre", "back", "espalda",
              "shoulder", "hombro", "rib", "costilla", "sternum", "neckline",
              "escote"),
    "neck": ("neck", "cuello", "throat", "garganta"),
    "upper_arm": ("upper arm", "bicep", "biceps", "hombro", "sleeve", "manga"),
    "forearm": ("forearm", "antebrazo", "wrist", "muneca", "elbow", "codo",
                "arm", "brazo"),
    "thigh": ("thigh", "muslo", "hip", "cadera", "groin", "ingle", "buttock",
              "gluteo"),
    "shin": ("calf", "pantorrilla", "shin", "espinilla", "knee", "rodilla",
             "ankle", "tobillo", "leg", "pierna"),
}


def marks_at_risk(preserve: Any, regions: Any, kept: Any = ()) -> list[str]:
    """The marks the reader found that this mask is about to paint over.

    Both halves have to match before anything is claimed: the entry has to name
    a MARK (a tattoo, a mole, a scar) and it has to name a PLACE that one of
    the regions being repainted actually covers.  "earrings" and "face shape"
    are in the same list and are kept by the shield, so neither is reported;
    "tattoo on forearm with text" is reported when the arms are repainted and
    not when only the background is.
    """
    wanted = {str(r) for r in (regions or [])}
    places: set[str] = set()
    for region in wanted:
        places.update(_MARK_PLACE.get(region, ()))
    if not places:
        return []
    # Whatever the mask really keeps is taken back out of the warning.
    safe: set[str] = set()
    for zone in (kept or []):
        safe.update(_MARK_ZONE_WORDS.get(str(zone), ()))
    found: list[str] = []
    for entry in (preserve or []):
        text = str(entry or "").strip().lower()
        if not text or not any(w in text for w in _MARK_WORDS):
            continue
        if safe and any(word in text for word in safe):
            continue
        if any(place in text for place in places):
            clean = str(entry).strip()
            if clean not in found:
                found.append(clean)
    return found


def marks_sentence(marks: Any) -> str:
    """The warning, in her language, or an empty string when there is none.

    THE HAIR CAME OUT OF THIS SENTENCE AND HAS GONE BACK IN, WITH THE PIXELS
    UNDER IT.  It used to promise "tu cara, tu pelo, tus manos y tus pendientes
    si se copian de tu foto" while the only hair the mask kept was the sliver
    segment.region_masks finds above the face line - 7907 px on the new
    photograph - so the long hair falling over her shoulder and chest was
    repainted and the sentence was wrong about the one thing a reader checks by
    eye.  It was then rewritten to say the opposite, that the hair over the
    clothing IS repainted, which was true of the code as it stood.  Since
    her_hair the code no longer stands that way: her hair is measured on her
    own photograph and forced black beside her face and her hands, so the
    promise is made again - and what it costs the garment is said by
    hair_note_es, on the same screen, measured on the same photograph.
    """
    items = [str(m).strip() for m in (marks or []) if str(m or "").strip()]
    if not items:
        return ""
    return ("Lo que la mascara NO puede conservar: %s. Esas marcas estan "
            "dentro de la zona que se repinta, asi que el motor las pinta "
            "encima y no puede devolverlas; tu cara, tu pelo, tus manos y "
            "tus pendientes si se copian de tu foto, tambien el pelo que te "
            "cae sobre la ropa. Si quieres conservar una marca, "
            "elige una prenda que la deje fuera (manga corta para el brazo, "
            "escote para el pecho) o pide solo un cambio de fondo."
            % "; ".join(items))


def mask_name(source_path: Any, regions: Any, bare: Any = ()) -> str:
    """The file name a mask for this photograph and these regions must have.

    Named after WHAT IT IS rather than after which variant asked for it: six
    previews that all change the clothes need one mask, not six identical ones,
    and the estimate that priced them needs the very same file so that the
    price it quotes is the price of a call that can really be made.  It keeps
    the ``v*_mask.png`` shape the rehearsal already looks for.
    """
    # ``bare`` is the set of body zones the requested garment leaves showing,
    # and it belongs in the name because it changes the mask: her marks are
    # black inside those zones and repainted outside them, so two requests that
    # touch the same regions with different garments are two different masks.
    # Without it a shirt would reuse the mask drawn for a sleeveless dress and
    # keep the forearm the shirt is meant to cover.
    raw = "%s|%s|%s" % (Path(str(source_path)).name,
                        ",".join(sorted(str(r) for r in (regions or []))),
                        ",".join(sorted(str(z) for z in (bare or []))))
    return "v%s_mask.png" % hashlib.sha1(raw.encode("utf-8")).hexdigest()[:12]


def shield_for(source_path: Any, choices: Any, work_dir: Any = None,
               profile: Any = None) -> dict:
    """WILL HER FACE BE GENERATED?  The one place that answers, for everybody.

    This is the fix to a defect that had already been fixed once from the other
    end.  The answer was being taken twice: ``plan_mask`` above decided it from
    the option groups for the estimate, and ``_run_variant`` decided it again
    from whether a mask could actually be DRAWN on this photograph - which is a
    different question, because 9 of her 24 photographs are closeups where the
    regions are found differently, and because a mask under MIN_COVER or over
    MAX_COVER is refused on pixels the groups know nothing about.  When the two
    disagreed the client was quoted one endpoint (fill, 0.050 USD) and billed
    another (kontext/multi, 0.040 USD with three references), and the sentence
    on the screen promised a face that would not be redrawn while the run
    redrew it.

    So both callers ask this, and this draws the mask ONCE - into ``work_dir``,
    under a name made of the photograph and the regions - and hands the second
    caller the same file, the same coverage and the same sentence.  Without a
    ``work_dir`` there is nowhere to keep it, and the honest answer is the group
    one, marked ``sin dibujar`` so nobody mistakes it for a measured one.
    """
    plan = plan_mask(choices)
    cover_state = garment_cover(choices)
    bare = sorted(z for z, s in cover_state.items() if s == "descubierta")
    out: dict[str, Any] = {
        "masked": False, "mask_path": "", "cover": 0.0,
        "regions": list(plan.get("regions") or []),
        "blocked": list(plan.get("blocked") or []),
        "groups": list(plan.get("groups") or []),
        "reason": _text(plan.get("reason")), "estado": "bloqueado",
        "marks": [], "marks_note": "", "prenda_cubre": cover_state,
        # What the mask does with her hair, and what that costs the garment.
        "hair": {}, "hair_note": "",
        # What will really be uploaded, so the estimate can say it before the
        # money moves.  Empty until a mask exists, because until then there is
        # no repaint zone to measure it against.
        "envio": {}, "envio_note": "",
    }
    if not plan.get("safe"):
        return out

    source = _text(source_path)
    folder = Path(_text(work_dir)) if _text(work_dir) else None
    if not source or folder is None:
        out["masked"] = True
        out["estado"] = "sin dibujar"
        out["reason"] = masked_sentence(out["groups"], 0.0, out["regions"])
        return out

    mask_file = folder / mask_name(source, plan.get("regions") or [], bare)
    with _MASK_LOCK:
        record = _binding_data(mask_file, source)
        if record:
            out.update({
                "masked": True, "mask_path": str(mask_file),
                "cover": float(record.get("cubre") or 0.0),
                "regions": [str(r) for r in (record.get("regiones")
                                             or out["regions"])],
                "marks": list(record.get("marcas") or []),
                "hair": dict(record.get("pelo") or {}),
                "envio": dict(record.get("envio") or {}),
                "estado": "reutilizada"})
            if not out["envio"]:
                # A mask drawn before any of this was measured.  The skin
                # fractions need her photograph and cost 0.4 s, which is 80x
                # what reading a mask back costs, so only the geometry is
                # recovered here - the rectangle and how much of her head is
                # in it, both read from the mask PNG alone.  Half an answer
                # said for free beats a whole one nobody waits for.
                out["envio"] = _safe(upload_report, mask_file, None, None,
                                     None, profile, [0, 0]) or {}
            out["marks_note"] = marks_note_es(out["marks"])
            out["hair_note"] = hair_note_es(out["hair"])
            out["envio_note"] = upload_note_es(out["envio"])
            out["reason"] = masked_sentence(out["groups"], out["cover"],
                                            out["regions"])
            return out
        try:
            folder.mkdir(parents=True, exist_ok=True)
            built = build_mask(source, choices, str(mask_file),
                               regions=plan.get("regions") or [],
                               profile=profile)
        except Exception as exc:                          # noqa: BLE001
            built = {"ok": False,
                     "reason": "no se pudo preparar la mascara (%s)" % exc}

    if not built.get("ok"):
        # Not an error: on a closeup there may be no torso to repaint at all.
        # It IS money, though - this is the branch that pays kontext/multi
        # instead of fill - so it is said in her language, once, here.
        out["estado"] = "sin zona"
        out["reason"] = ("Tu rostro se va a volver a generar: %s. Se comprobara "
                         "la identidad antes de ensenarte nada."
                         % (built.get("reason") or "no se pudo aislar la zona"))
        return out
    out.update({"masked": True, "mask_path": str(built["mask_path"]),
                "cover": float(built.get("cover") or 0.0),
                "regions": list(built.get("regions") or out["regions"]),
                "marks": list(built.get("marks") or []),
                "marks_note": _text(built.get("marks_note")),
                "hair": dict(built.get("hair") or {}),
                "hair_note": _text(built.get("hair_note")),
                "envio": dict(built.get("envio") or {}),
                "estado": "dibujada"})
    out["envio_note"] = upload_note_es(out["envio"])
    out["reason"] = masked_sentence(out["groups"], out["cover"], out["regions"])
    return out


# ------------------------------------------------------------------ the mask

def build_mask(source_path: str, choices: Any, out_path: str,
               regions: Any = None, profile: Any = None) -> dict:
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
                              "reason": "", "detail": {}, "marks": [],
                              "marks_note": "",
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
    # And the same thing again without the hands.  The hands are protected
    # pixels but they live in the middle of the body, so a rectangle drawn
    # round them would span the torso and be useless to anyone asking "where
    # is her face"; upload_report measures against the head alone.
    head = np.zeros((height, width), np.uint8)
    for name in ("face", "hair", "hands"):
        got = found.get(name)
        if isinstance(got, np.ndarray) and got.size:
            protect = cv2.bitwise_or(protect, got)
            if name != "hands":
                head = cv2.bitwise_or(head, got)
    # And the hands as the hand model sees them, not only as the skeleton
    # guesses them: see HAND_CONFIDENCE above for the two photographs where the
    # skeleton capsule left 45% and 10% of a real hand inside the repaint zone.
    hulls = _hand_hulls(small)
    if hulls is not None:
        protect = cv2.bitwise_or(protect, hulls)
    # The face rectangle from the mesh as well as the ellipse from the pose:
    # the two disagree on a turned head, and their union is the honest answer.
    box = face_d.get("bbox") if isinstance(face_d, dict) else None
    # And kept, in the analysis frame, so the finished mask can be measured
    # against the rectangle the guarantee is actually about.  See _write_binding.
    face_rect: list[float] = []
    if isinstance(box, (list, tuple)) and len(box) >= 4:
        x, y, w, h = [float(v) for v in list(box)[:4]]
        if max(x, y, w, h) <= 1.5:
            x, y, w, h = x * width, y * height, w * width, h * height
        cv2.rectangle(protect, (int(x), int(y)), (int(x + w), int(y + h)), 255, -1)
        cv2.rectangle(head, (int(x), int(y)), (int(x + w), int(y + h)), 255, -1)
        face_rect = [x, y, x + w, y + h]
    if not np.count_nonzero(protect):
        result["reason"] = ("no se encuentra tu rostro en la foto, asi que no "
                            "se puede proteger")
        return result

    # HER HAIR, ALL OF IT THIS PHOTOGRAPH SHOWS.  The "hair" region added above
    # is the sliver at the crown - 7907 px on the new photograph - and the long
    # hair falling over her shoulder and chest was inside the repaint zone.
    # See her_hair for what that sliver is grown into and for what the growing
    # costs the garment; it goes into the same core as the face and the hands,
    # so the feather cannot reach it either.
    hair_layer, hair_info = _safe(her_hair, small, person, found, face_d,
                                  pose_d, profile, change) or (None, {})
    if isinstance(hair_layer, np.ndarray) and np.count_nonzero(hair_layer):
        protect = cv2.bitwise_or(protect, hair_layer)

    margin = _odd(PROTECT_MARGIN * min(height, width))
    core = protect.copy()
    grown = cv2.dilate(protect, cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (margin, margin)))

    # HER MARKS, INTO THE SAME PROTECTED SET AS THE FACE AND THE HANDS - but
    # only the ones the requested garment will really leave showing.  A mark
    # under the new garment is repainted on purpose: keeping her own pixels
    # there would put a hole in the shirt.  What is lost is not lost silently,
    # it is named in ``marks_note`` and reaches the plan note.
    person_bbox: list[float] = [0.0, 0.0, float(width), float(height)]
    if person is not None:
        got = _safe(segment_mod.bbox_of, person)
        if isinstance(got, (list, tuple)) and len(got) == 4 and float(got[2]) > 1:
            person_bbox = [float(v) for v in got]
    marks, labels = ([], None)
    if profile:
        marks, labels = _safe(visible_marks, small, pose_d, person, found,
                              profile, person_bbox, hulls) or ([], None)
    # Asked once and used twice: the marks need to know what the garment
    # covers, and so does the hair note - a hood or a high collar is a garment
    # that MEANS to cover her hair, which the mask now refuses to repaint.
    cover_state = _safe(garment_cover, choices) or {}
    mark_layer = np.zeros((height, width), np.uint8)
    pad = _odd(MARK_PAD_FRAC * min(height, width), low=3)
    pad_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (pad, pad))
    budget = MARK_TOTAL_MAX * float(height * width)
    for mark in marks:
        comp = (labels == int(mark["label"])) if labels is not None else None
        if comp is None or not bool(comp.any()):
            mark["estado"] = "sin pixeles"
            continue
        piece = cv2.dilate((comp.astype(np.uint8)) * 255, pad_kernel)
        inside = float(np.count_nonzero(cv2.bitwise_and(piece, change)))
        mark["dentro"] = round(inside / max(1.0, float(np.count_nonzero(piece))), 3)
        if mark["dentro"] < MARK_INSIDE_MIN:
            # Nothing is repainting it, so there is nothing to promise.
            mark["estado"] = "fuera"
            continue
        state = str(cover_state.get(str(mark.get("zona") or ""), "desconocida"))
        if state != "descubierta":
            mark["estado"] = "tapada" if state == "cubierta" else "dudosa"
            continue
        if float(np.count_nonzero(mark_layer)) + inside > budget:
            # Never reached on her 25 photographs - the largest protected set
            # measured is 1.42% of the frame (IMG_8798) against this 3% - but a
            # detector that ran away would otherwise eat the repaint zone, and
            # the honest thing is to stop and say which marks were dropped.
            mark["estado"] = "excedida"
            continue
        mark_layer = cv2.bitwise_or(mark_layer, piece)
        mark["estado"] = "protegida"
    if np.count_nonzero(mark_layer):
        grown = cv2.bitwise_or(grown, mark_layer)
        core = cv2.bitwise_or(core, mark_layer)

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
    # repainting her face and reporting that nothing changed - and what it
    # covers, so that shield_for() can hand the same numbers to the estimate
    # and to the run without drawing the mask a second time.
    cover_full = round(float(np.count_nonzero(big)) / float(fh * fw), 4)
    # The head, measured on the analysis image and carried up to her real
    # pixels the same way the mask itself is.  Rounded OUTWARDS on every side,
    # because a rectangle that is meant to keep her face at home is only
    # honest if it errs by being too large.
    head_box: list[int] = []
    sx, sy = float(fw) / float(width), float(fh) / float(height)
    ys, xs = np.nonzero(head)
    if xs.size:
        head_box = [max(0, int(np.floor(xs.min() * sx))),
                    max(0, int(np.floor(ys.min() * sy))),
                    min(fw, int(np.ceil((xs.max() + 1) * sx))),
                    min(fh, int(np.ceil((ys.max() + 1) * sy)))]
    # The face rectangle alone, carried up the same way and rounded outwards
    # too, so that a mask read back off disk can be checked against the
    # guarantee - 0 mask pixels inside it - without her photograph in hand.
    face_box: list[int] = []
    if face_rect:
        face_box = [max(0, int(np.floor(face_rect[0] * sx))),
                    max(0, int(np.floor(face_rect[1] * sy))),
                    min(fw, int(np.ceil(face_rect[2] * sx))),
                    min(fh, int(np.ceil(face_rect[3] * sy)))]
    # And what will really be uploaded, measured now while her photograph, her
    # skin envelope and the finished mask are all in hand.
    # THE RECORD IS WRITTEN TWICE, ON PURPOSE.  upload_report answers "how much
    # of her face is in the picture fal reviews, and how much of it does the
    # mask repaint" by reading the head rectangle out of THIS record - because
    # one rectangle read from one place cannot disagree with itself, and the
    # estimate reads the same file later without her photograph in hand.  So
    # the head goes to disk first, the upload is measured against the record,
    # and the record is completed with the measurement.
    if isinstance(hair_info, dict) and hair_info:
        hair_info["cuello_cubierto"] = (
            str(cover_state.get("neck") or "") == "cubierta")
    hair_note = hair_note_es(hair_info)
    _safe(_write_binding, saved, source_path, [fw, fh], cover_full, used,
          head_box, marks, None, face_box, pelo=hair_info)
    envio = _safe(upload_report, saved, small, person, found, profile,
                  (fw, fh)) or {}
    _safe(_write_binding, saved, source_path, [fw, fh], cover_full, used,
          head_box, marks, envio, face_box, pelo=hair_info)

    kept_marks = [m for m in marks if str(m.get("estado")) == "protegida"]
    result.update({
        "ok": True,
        "mask_path": str(saved),
        "cover": cover_full,
        "regions": used,
        "marks": marks,
        "marks_note": marks_note_es(marks),
        "hair": dict(hair_info) if isinstance(hair_info, dict) else {},
        "hair_note": hair_note,
        "envio": envio,
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
            "cabeza": head_box,
            "cara": face_box,
            "regiones": used,
            "marcas_vistas": len(marks),
            "marcas_protegidas": len(kept_marks),
            "marcas_px": int(np.count_nonzero(mark_layer)),
            "marcas_zonas": sorted({str(m.get("zona") or "") for m in kept_marks}),
            "pelo_px": int(0 if hair_layer is None
                           else np.count_nonzero(hair_layer)),
            "pelo_zona": float((hair_info or {}).get("zona") or 0.0),
            "prenda_cubre": dict(cover_state),
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
