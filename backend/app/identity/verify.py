"""The quality gate every generated image has to pass.

This is where the robot earns its name.  A generator can be asked politely not
to slim someone down; only a measurement can prove it did not.  ``verify_image``
re-measures the produced pixels with the same analysers that built the profile
and answers five questions with numbers: is it the same face, the same body, the
same skin, is the anatomy sane, is the file technically good enough.

Two rules shape the code.  First, a check that could not be computed never fails
the person - missing data is reported as skipped, not as a defect.  Second, the
verdict must be readable: the details are written in the Spanish the client will
actually read, naming what changed and by how much.
"""
from __future__ import annotations

import math
import os
import time
from typing import Any

import cv2
import numpy as np

from ..analysis import anomaly as anomaly_mod
from ..analysis import body as body_mod
from ..analysis import face as face_mod
from ..analysis import loader
from ..analysis import pose as pose_mod
from ..analysis import quality as quality_mod
from ..analysis import segment as segment_mod
from ..analysis import skin as skin_mod
from . import embedding as embedding_mod
from .profile import DEFAULT_THRESHOLDS, GATED_METRICS, usable_metrics

VERIFY_MAX_SIDE = 1600
QUALITY_MIN = 0.45
ANATOMY_SEVERITY_MAX = 0.6

# --- oversmoothed skin -------------------------------------------------------
# The defect the client actually came here with: "it removes any retouching and
# manipulates the image to look exactly like her".  A generator keeps her shape
# and her colour - face similarity stays at 0.99 - and quietly deletes the band
# that makes a photograph a photograph: pores, fine lines, the grain her camera
# recorded.  It is judged apart from the rest of the anatomy, with its own
# threshold, for one reason: every phone camera smooths a face too, so the same
# measurement means "her camera" on a photograph she took and "the robot
# retouched her" on a generated one.  Only the caller knows which, so only the
# generative context is gated - see _smoothing_verdict.
#
# What is compared is a RATIO between two measurements of the same person,
# never a stored amplitude.  The fine band is not a property of skin: over her
# 24 photographs it tracks how wide the face happens to be, Pearson r = 0.89 -
# the same cheek reads 0.71 at 130 px and 3.09 at 540 px, because grain the
# sensor recorded at one pixel is simply not there once the face is 100 px
# across.  Comparing a result against a fixed amplitude therefore measures
# framing: built from her own photographs, a fixed reference of 1.09 rejected
# 12 of 24 perfectly faithful full-length results (100 px face) and caught 0 of
# 17 closeups with 40% of the grain deliberately deleted (260 px face).
#
# So the reference is her own source photograph, and both images are shrunk
# until their faces are the same width before either is measured - downwards
# only, since shrinking removes grain that exists while enlarging invents none.
# What survives is the fraction of HER grain this engine kept, at her framing.
#
# Shrinking is itself a measurement, and it has to be the same measurement on
# both sides.  The first version resized only the larger face and read the
# other one as it came, and its zero was not zero: a faithful copy of her own
# photograph (loaded at 1400 px, saved again as JPEG) read a loss of -0.23,
# 23% MORE grain than the file it was made from, because the side left alone
# kept its own resize and JPEG noise while the other side was box-filtered
# clean.  The zero also moved with how far that side was shrunk - an exact 2:1
# INTER_AREA passes far more near-Nyquist noise than a 0.44 weighted average -
# reading -0.10 with the target at half the narrower face and +0.01 at 0.8 of
# it.  So both faces are now brought to TEXTURE_MATCH_FACTOR of the narrower
# one, which resamples both sides every time, and each side is first blurred
# by a Gaussian of TEXTURE_PREFILTER pixels measured at the target scale
# (sigma = TEXTURE_PREFILTER / k in its own pixels): the anti-alias is then
# the same filter on both sides whatever pair of shrink factors the two
# framings happen to produce.
#
# The two lines below are read off that ratio, measured on cases built from
# her own photographs: a faithful copy saved again as JPEG at four sizes, so
# the pair of shrink factors varies, and the same photograph with a known
# fraction of the fine band deleted (cv2.bilateralFilter 9/60/9, blended).
# Measured loss, mean and [min..max] over 8 photographs:
#
#   faithful copy 1600 px    -0.015  [-0.032 .. 0.010]
#   faithful copy 1300 px    -0.001  [-0.018 .. 0.021]
#   faithful copy 1000 px     0.027  [-0.005 .. 0.065]
#   faithful copy  800 px     0.036  [ 0.015 .. 0.081]
#   -25% of the band          0.119  [ 0.068 .. 0.175]
#   -50% of the band          0.214  [ 0.116 .. 0.301]
#
# A faithful copy never exceeded 0.081 and a face with half its band erased
# never read under 0.116, so 0.14 rejects nothing that kept her grain and
# catches every half-erased face measured; the 40% FLUX Kontext was measured
# deleting from her lies between those two populations.  Below 0.09 nothing
# is said at all; between the two the loss is reported so the texture step can
# act on it, without throwing the picture away.
#
# The honest limit: a 25% cut reads 0.12 on average - under the line - so a
# quarter of her grain can go missing and only be reported.  The gate is
# deliberately the safety net and not the cure, because the cure already ran:
# the orchestrator puts her own grain back with generation/retouch.py before
# anything is verified.  Rejecting a faithful result would charge her for a
# regeneration she does not need, which is the trade this line is set on.
SMOOTH_TEXTURE_LOSS_MAX = 0.14
SMOOTH_TEXTURE_LOSS_MIN = 0.09
# Both faces are shrunk to this fraction of the narrower one, so that neither
# side is ever the unresampled one; and each side is anti-aliased by this many
# pixels at the target scale before its own INTER_AREA - see the note above.
TEXTURE_MATCH_FACTOR = 0.8
TEXTURE_PREFILTER = 0.5
# Below this the source photograph carries so little grain of its own that the
# ratio would be dividing noise by noise, and nothing is claimed either way.
SMOOTH_TEXTURE_MIN_REF = 0.30
# Severity is linear in that loss, so this constant IS the loss at the line
# (SMOOTH_TEXTURE_LOSS_MAX) expressed as a severity; it is deliberately not
# ANATOMY_SEVERITY_MAX, which answers a different question about broken hands
# and invented limbs.
SMOOTH_SEVERITY_MAX = 0.45
SMOOTH_SEVERITY_CAP = 0.95
BODY_CONF_MIN = 0.5           # below this, measurements only inform, never gate
YAW_BAND_SLACK = 1.4          # extra tolerance for a yaw-corrected width
PAIRED_TOL = 0.08             # torso rulers: the size a difference has to reach
                              # before it is worth printing.  Since the ruler
                              # shoot-out above HEAD_TOL it reports only and
                              # rejects nothing.
PROFILE_MIN_SAMPLES = 4       # torso heights needed before the profile counts

# A pose landmark is not a tape measure.  Measuring one untouched photograph
# again at a different scale moves the shoulder and hip widths by several
# percent, and the torso length that divides them moves with them, so a fixed 8%
# limit is a gate more precise than its own instrument: it rejects images whose
# body was never touched.  The limit is therefore estimated for each pair - the
# source photograph is read again at the size of the result and once more
# smaller, and the spread between those readings is this pair's noise floor.
PAIRED_NOISE_FACTOR = 1.5     # a real change must stand this far above the noise
NOISE_RESCALE = 0.8           # the second look at the source, at 80% of the first
NOISE_MIN_SIDE = 320          # below this a body measurement is not worth taking
# The cap is what keeps a noisy pair from quietly switching the protection off.
# The client's complaint was a slim-down of about 12%, and the profile builder
# refuses to learn a band wider than +/-12% of the mean for the same reason, so
# a limit past 16% would forgive precisely the damage this module exists to
# catch.  When the measured noise asks for more, the limit stops here and the
# detail says the pair was too noisy to judge instead of widening in silence.
PAIRED_TOL_MAX = 0.16
# The silhouette profile is compared as a whole, so it carries its own noise
# entry rather than borrowing one from a single width.
WIDTH_PROFILE_KEY = "width_profile"
# THE RULER.  Three of them used to take part here and they contradicted each
# other on the same picture, so they were shot against ground truth that could
# be constructed rather than argued about: her 24 photographs re-encoded and
# re-read (truth: nothing changed), the same photographs re-read at 1024 px
# instead of 1600 (truth: nothing changed), cropped to 62% of their height
# (truth: nothing changed, the FRAME changed), widened 6% by the same warp that
# slims - a stand-in for the bulk a garment adds, which is the only direction
# clothing can move a silhouette (truth: her body did not change) - and slimmed
# by a known 6, 12 and 18%, plus a 62% crop of the 12% slim.  Fourteen of the
# twenty-four can be measured at all; 112 measured pairs.  What each ruler read
# (scratchpad rulers.py / an.py, 2026-09-04), as a percentage away from "no
# change":
#
#   ruler       untouched   reframed     +6% fabric   -6% slim   -12%    -18%
#   skeletal    0.7..14.4   13.5         1.5..15.3    2.6..19.8  2.7..13.4  3.1..25.7
#   width prof  0.0.. 2.6   14.2..16.9   0.4.. 7.5    2.8.. 7.8  3.1..12.5  3.3..19.9
#   head prof   0.0.. 1.2    0.3.. 1.6   4.1.. 6.7    4.5.. 7.5 11.4..13.5 17.2..19.0
#
# The first two do not measure the body.  The skeletal ratios read 14.4% on a
# photograph that was only re-encoded and 2.7% on another one slimmed by 12%:
# at no limit at all do they reach 10% false alarm together with 80% detection.
# The width profile reads the crop - 14..17% on a picture nobody touched - and
# missed a real 12% slim under that crop entirely (0.4%).  The head profile
# reads the truth: it divides by forehead-to-chin of the face mesh on rows hung
# from the chin, so a crop cannot move it, and it came back within 1.6% of zero
# on every untouched and every reframed photograph while reporting each slim
# within about two points of the amount actually applied.  So the head profile
# is the gate and the other two are demoted to reported-only in
# _check_body_paired.  They are not averaged with it: averaging a ruler that
# measures the framing into one that measures the body is how two contradictory
# percentages ended up in the same verdict.
#
# WHERE THE LINE GOES.  The client's instruction is that a small difference must
# not cost her a paid image, and that the slim-down she originally complained
# about - about 12% - must still be caught.  At 8%:
#   false alarm  0 of 25  (7 re-encoded + 7 re-read smaller + 4 reframed
#                          + 7 widened 6% by fabric; worst reading 6.7%)
#   6% slim      0 of 7 caught  (readings 4.5..7.5%)
#   12% slim     7 of 7 caught  (readings 11.4..13.5%)
#   18% slim     7 of 7 caught  (readings 17.2..19.0%)
#   12% slim under a 62% crop   4 of 4 caught  (8.1..12.3%)
# The 6% row is not a limit that could simply be tightened: a 6% slim reads
# 4.5..7.5% and 6% of added fabric reads 4.1..6.7%, and those two overlap, so
# at that size no threshold separates a slimmed body from a jumper.  Only the
# SIGN separates them, and a pose change moves the silhouette both ways - over
# the 17 generated images on disk this ruler could measure, the worst honest
# NARROWING was 4.2% - so the sign is not solid enough to gate on either.  A
# 6% difference is therefore measured, written into the verdict in plain
# Spanish, and not charged to her, which is what she asked for.
#
# The per-pair widening in _noise_aware_tol still applies on top of this floor,
# capped at HEAD_TOL_MAX and not at the wider PAIRED_TOL_MAX: the faintest real
# 12% slim read 11.4%, so a limit allowed past that would forgive the very
# damage this module exists to catch.  Measured head-ruler noise is 1.6% at
# worst, so 1.5 x noise never reaches 8% and the widening is inert here in
# practice; the cap is the guarantee, not the expectation.
HEAD_PROFILE_KEY = "head_profile"
HEAD_TOL = 0.08
HEAD_TOL_MAX = 0.10
# Before a rejection is spent, the heights have to agree with each other, and
# this is the size of disagreement past which they do not.
#
# The failure this exists to stop was measured, not imagined.  Her own
# IMG_8841, untouched and merely cropped to the top 80% of its own height, was
# REJECTED as "tu figura es un 20% mas estrecha de arriba abajo".  Nothing
# about her had moved; the person segmentation had.  In the cropped picture the
# mask stops at her waist and leaves her hips and thighs out, so every row down
# to two heads below the chin reads the same as the source and every row below
# it reads 20..40% narrower.  The median of the eleven rows is a 19.5% slim the
# generator never performed - in the one direction, narrowing, that this whole
# module exists to catch, on a photograph nobody had touched.
#
# The rows themselves say which case they are in, because a slim-down is a
# single multiplication: it moves EVERY height by the same factor, while a mask
# that finds a different amount of body in the two pictures moves some heights
# and leaves the rest alone.  Measured over the 7 photographs this ruler can
# judge, as the median absolute deviation of the per-row ratios about their own
# median, relative to that median (91 readings, 13 conditions):
#   HONEST, nothing about her moved
#     re-encode 0.000..0.006   at 1024 0.001..0.005   at 768 0.001..0.004
#     re-saved at q75 0.000..0.006      upscaled x2 0.000..0.005
#     lit +12 L* 0.000..0.004           +6% of garment bulk 0.003..0.019
#     crop to 62% of the height 0.005..0.015   sides to 72% 0.004..0.025
#     turned 3 degrees 0.003..0.037  <- the loudest honest reading
#   HER BODY REALLY NARROWED
#     6% 0.002..0.025      12% 0.001..0.015      18% 0.002..0.014
#   THE MASK COLLAPSED
#     the 80% crop of IMG_8841                             0.195
# Every genuine reading - honest or slimmed - stays under 0.037, and no real
# slim ever passed 0.025; the broken mask reads 7.8 times that.  The limit sits
# at twice the loudest real slim and a quarter of the failure.
#
# It can only ever turn a rejection into a report, never the other way round,
# so its worst case is a slim described instead of rejected, never a paid image
# discarded.  Over the whole corpus it changes exactly one verdict out of 15:
# the false alarm above.  All 14 rejections of a real 12% or 18% slim survive
# it untouched.
HEAD_ROWS_MAD_MAX = 0.05
# The width profile and the head profile read the very same mask, so the ratio
# between their two medians is not about her width at all: it is how much the
# torso unit (pose landmarks) moved against the head unit (face mesh) between
# the two images.  A reframe moves only the torso unit: a 62% crop of an
# untouched photograph read the width profile 14..17% while the head profile
# stayed within 1.6%.  This number no longer switches any ruler off - since the
# shoot-out above, nothing that divides by torso length is allowed to reject
# anything - and it is kept for one purpose: it is the cheapest honest
# explanation of WHY the torso figures printed in the verdict disagree with the
# verdict, and she is owed that sentence rather than two contradictory
# percentages with no account of which one to believe.
UNIT_SHIFT_MAX = 0.06
# The pixel width measure_body stored for each width metric, so the paired
# test can divide the raw width by the raw torso length on both sides.  The
# yaw correction the metrics carry exists to compare one photograph with a
# band built from others taken at other angles; between two images of the
# same pose the foreshortening is the same on both sides and cancels, and
# what the correction adds to a pair is only the scatter of the yaw estimate.
# That scatter is not small: one untouched photograph read yaw 32.6 degrees at
# 1600 px and 13.6 degrees at 1200 px, so one side was divided by cos(33) and
# the other left alone, and shoulders, hips and bust came out 16..19%
# narrower on a faithful copy while the raw ratios were 0.99, 0.96 and 1.00
# and the head ruler read 1.000.
_PX_WIDTH: dict[str, str] = {
    "shoulder_w_over_torso": "shoulder_w",
    "hip_w_over_torso": "hip_w",
    "waist_w_over_torso": "waist_w",
    "bust_w_over_torso": "bust_w",
    "neck_w_over_torso": "neck_w",
}
# A difference the engine physically could not have caused is not evidence about
# the body.  It must not reward the image either, so the check scores it in the
# middle instead of dragging the run's score down with a measurement error.
NON_GENERATIVE_SCORE = 0.75

# The skin check used to be the last identity ruler still measured against the
# POPULATION profile - the pooled mean of her whole gallery - while
# body_proportions had already moved to comparing the image with the very
# photograph it was made from.  Measured over her 24 photographs under seven
# honest changes (168 readings: re-encode, three reframes, a 3 degree rotation,
# a declared lighting change, 6% of added garment bulk):
#
#     reference                        falsas alarmas   peor lectura
#     el perfil de la galeria (antes)      3 de 168      1.101 del limite
#     su propia foto de origen (ahora)     0 de 168      0.884 del limite
#
# Two of the three false alarms were a LIGHTING change she had asked for, and
# the third a 62% reframe that simply dropped a brightly lit thigh out of the
# sample: the pooled mean cannot tell "this crop shows less leg" from "somebody
# lightened her".  Her own photograph can, because both readings sample the
# same person under the same light.
#
# The population profile is not thrown away, because a paired reading has one
# blind spot the pooled one does not: if the engine shifts the whole frame it
# shifts source and result alike.  So it stays as a LOOSER veto, at
# SKIN_PROFILE_VETO times its own limit.  Sized off the same 168 honest
# readings, whose worst reading against the profile is 1.101: 1.25 leaves 14%
# of headroom above the loudest honest picture.  Over 120 images with the skin
# really changed the pair beats the old rule on both counts at once -
# 0.0% false alarms against 1.8%, and 58.3% detection against 55.8% - so
# nothing was traded away for the quiet.
SKIN_PROFILE_VETO = 1.25

# Two lists used to live here, SILHOUETTE_METRICS and WIDTH_METRICS, naming
# which torso-length ratios a change of clothes was allowed to move and in
# which direction.  They are gone because the ruler they qualified is gone:
# since the shoot-out above HEAD_TOL nothing that divides by torso length can
# reject an image, so there is nothing left to excuse.  The reasoning they
# carried survives where it is still load-bearing - fabric is worn on top of a
# person, every garment in the catalogue adds bulk and none removes any, so a
# figure that comes back WIDER may be the coat while one that comes back
# NARROWER cannot be - and it now qualifies the head-length ruler instead, in
# _check_body_paired.

CHECK_WEIGHTS: dict[str, float] = {
    "identity_face": 0.30,
    "body_proportions": 0.25,
    "skin_tone": 0.15,
    "anatomy": 0.20,
    "quality": 0.10,
}

# metric -> (noun, word when it shrank, word when it grew)
METRIC_ES: dict[str, tuple[str, str, str]] = {
    "shoulder_w_over_torso": ("hombros", "mas estrechos", "mas anchos"),
    "hip_w_over_torso": ("caderas", "mas estrechas", "mas anchas"),
    "waist_w_over_torso": ("cintura", "mas estrecha", "mas ancha"),
    "bust_w_over_torso": ("busto", "mas estrecho", "mas ancho"),
    "head_h_over_torso": ("cabeza", "mas pequena", "mas grande"),
    "neck_w_over_torso": ("cuello", "mas estrecho", "mas ancho"),
    "arm_len_over_torso": ("brazos", "mas cortos", "mas largos"),
    "leg_len_over_torso": ("piernas", "mas cortas", "mas largas"),
    "shoulder_over_hip": ("relacion hombros-caderas", "menor", "mayor"),
}

DEFECT_ES: dict[str, str] = {
    "hand_malformed": "mano deformada",
    "extra_limb": "extremidad de mas",
    "extra_person": "persona de mas",
    "face_distorted": "rostro alterado",
    "eye_asymmetry": "ojos asimetricos",
    "missing_limb": "falta una extremidad",
    "texture_smear": "textura emborronada",
    "duplicated_feature": "elemento duplicado",
    "border_artifact": "artefacto en el borde",
    "oversmoothed_skin": "piel demasiado suavizada",
}

# Body parts the anatomy scan could locate but not honestly judge.
UNJUDGED_ES: dict[str, str] = {
    "left_hand": "tu mano izquierda",
    "right_hand": "tu mano derecha",
    "hand": "una mano",
}

CHECK_ES: dict[str, str] = {
    "identity_face": "el rostro",
    "body_proportions": "las proporciones",
    "skin_tone": "el tono de piel",
    "anatomy": "la anatomia",
    "quality": "la calidad tecnica",
}

FAIL_ES: dict[str, str] = {
    "identity_face": "el rostro no coincide con el tuyo",
    "body_proportions": "cambiaron tus proporciones",
    "skin_tone": "cambio tu tono de piel",
    "anatomy": "hay errores anatomicos",
    "quality": "la calidad tecnica es baja",
}


# ------------------------------------------------------------------ helpers

def _safe(fn, *args) -> Any:
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


def _clamp01(value: Any) -> float:
    val = _f(value, 0.0)
    return 0.0 if val < 0.0 else (1.0 if val > 1.0 else float(val))


def _int_bbox(value: Any) -> list[int]:
    box = _flist(value, 4)
    if box is None or box[2] <= 0 or box[3] <= 0:
        return []
    return [int(round(box[0])), int(round(box[1])),
            int(round(box[2])), int(round(box[3]))]


def _ms(started: float) -> int:
    return int(round((time.perf_counter() - started) * 1000.0))


def _mk(name: str, value: Any, threshold: Any, passed: bool, detail: str) -> dict:
    return {"name": name, "value": round(_f(value, 0.0), 4),
            "threshold": round(_f(threshold, 0.0), 4), "passed": bool(passed),
            "weight": CHECK_WEIGHTS[name], "detail": detail}


def _thresholds(profile: dict) -> dict[str, float]:
    out = dict(DEFAULT_THRESHOLDS)
    src = profile.get("thresholds")
    if isinstance(src, dict):
        for key, value in src.items():
            val = _f(value, None)
            if val is not None:
                out[str(key)] = float(val)
    return out


def _defect(raw: Any) -> dict | None:
    if not isinstance(raw, dict):
        return None
    kind = str(raw.get("type") or "").strip()
    if not kind:
        return None
    return {"type": kind, "where": str(raw.get("where") or ""),
            "bbox": _int_bbox(raw.get("bbox")),
            "severity": round(_clamp01(raw.get("severity")), 3),
            "repairable": bool(raw.get("repairable")),
            "detail": str(raw.get("detail") or "")}


def _severity_es(severity: float) -> str:
    if severity >= 0.8:
        return "grave"
    if severity >= ANATOMY_SEVERITY_MAX:
        return "importante"
    if severity >= 0.35:
        return "moderado"
    return "leve"


def _expects_face(brief: dict) -> bool:
    """Does the brief say the shot must show a face?  Unknown means no claim."""
    for key in ("expects_face", "face_visible", "needs_face", "show_face"):
        if key in brief:
            return bool(brief.get(key))
    return str(brief.get("shot_type") or "").lower() in ("closeup", "half", "full")


def _is_generative(brief: dict) -> bool:
    """Can the engine that produced this image invent anatomy at all?

    Some providers here are not generators: they replace the background,
    recolour a garment, relight and crop.  Such an engine cannot make a woman
    slimmer even if it were asked to, so any proportion difference measured on
    its output is an artefact of the measurement by construction.  The caller
    knows which engine ran and says so; when nobody says, the answer is yes,
    because assuming a generator is the assumption that keeps the gate honest.
    """
    for key in ("generative", "provider_generative"):
        value = brief.get(key)
        if value is not None:
            return bool(value)
    return True


def _is_source_photograph(image_path: str, brief: dict) -> bool:
    """Is the file being verified the very photograph the brief points at?

    A photograph she took is not something the robot produced, whatever the
    engine flag says, and the smoothing every phone applies to it must never be
    charged to the generator.  This is the case the calibration harness runs -
    a real photograph handed in as both source and result - and the case the
    old severity cap existed to protect.
    """
    src = _source_path(brief)
    if not src or not image_path:
        return False
    try:
        if os.path.exists(src) and os.path.exists(image_path) \
                and os.path.samefile(src, str(image_path)):
            return True
    except OSError:
        pass
    return (os.path.normcase(os.path.abspath(src))
            == os.path.normcase(os.path.abspath(str(image_path))))


def _source_had_face(brief: dict) -> bool:
    """Did the photograph this image was made from actually show a face?

    Only worth asking on the non-generative path, and only when the generated
    image lost its face, so the detection here is paid for at most once per
    verification and never on the common route.
    """
    src = brief.get("source_face")
    if isinstance(src, dict):
        return bool(src.get("ok"))
    for key in ("source_has_face", "original_has_face"):
        value = brief.get(key)
        if value is not None:
            return bool(value)
    path = _source_path(brief)
    if not path:
        return False
    img = _safe(loader.load_image, path, VERIFY_MAX_SIDE)
    if not isinstance(img, np.ndarray) or img.size == 0:
        return False
    found = _ok_dict(_safe(face_mod.detect_face, img), "rostro no detectado")
    return bool(found.get("ok"))


def _unreliable_metrics(body: dict) -> set[str]:
    """Metrics measure_body itself distrusts; they inform but never gate.

    One exception, and it is the difference between a working proportion gate
    and a decorative one: a width flagged only because the subject was turned
    has already had its foreshortening undone by measure_body, so it is an
    approximation rather than a ruin.  Those are gated, with a wider tolerance
    applied in _check_body to pay for the correction's own error.  Widths ruined
    by an arm across the scanline stay out.
    """
    out: set[str] = set()
    raw = body.get("unreliable")
    if isinstance(raw, (list, tuple, set)):
        out.update(str(v) for v in raw)
    reasons = body.get("reliability")
    corrected = {str(v) for v in (body.get("corrected") or [])}
    if isinstance(reasons, dict):
        for name in list(out):
            if reasons.get(name) == "yaw" and name in corrected:
                out.discard(name)
    flags = body.get("flags")
    if isinstance(flags, dict):
        for name, flag in flags.items():
            if isinstance(flag, dict):
                if flag.get("unreliable"):
                    out.add(str(name))
            elif str(flag).lower() in ("unreliable", "low", "poco_fiable"):
                out.add(str(name))
    # A low confidence measurement is unreliable by definition.
    if _f(body.get("confidence")) < BODY_CONF_MIN:
        out.update(str(k) for k in (body.get("metrics") or {}))
    return out


# ------------------------------------------------------------------- checks

def _check_identity_face(img: np.ndarray, face_d: dict, profile: dict,
                         thresholds: dict, brief: dict,
                         defects_out: list[dict]) -> tuple[dict, float, bool]:
    """Is this HER face, measured against the embeddings of her photographs.

    What this check reads changed, and the reason is the worst defect the
    product had.  It used to compare the 64 float geometric descriptor from
    analysis/face.py, and that descriptor cannot tell one woman from another:
    her own 24 photographs scored 0.9832 .. 0.9993 against the stored profile
    and eight photographs of eight OTHER women scored 0.9577 .. 0.9945, so both
    populations sat about a quarter of the scale above the 0.72 line and the
    gate could not fire even in principle.  Two paid results the client
    rejected on sight - visibly a different woman, different face shape,
    different hair - scored 0.9905 and 0.9960 and were approved.  It now reads
    the SFace embedding instead (see identity/embedding.py), where the very
    same images come out 0.6362 .. 0.8785 for her and 0.0408 .. 0.2886 for the
    strangers.  The check keeps its shape and value is still higher-is-better
    on 0..1, so the report page and the album need no change; only the number
    underneath is real.

    Degrading is deliberate and it is never a silent pass.  No weights on disk,
    no signature in the profile, or no face SFace can read, and the check is
    reported as NOT computed: it stays out of the weighted score and the
    summary tells her "no se pudo comprobar el rostro" instead of claiming she
    was recognised.  It does not fall back to the geometric descriptor either.
    A fallback that cannot separate anybody is worse than an admission, because
    it prints a passing identity check on an image nobody verified - which is
    precisely how the two rejected results got through.
    """
    name = "identity_face"
    face_min = _f(thresholds.get("face_embed_min"),
                  DEFAULT_THRESHOLDS["face_embed_min"])
    reference = (profile.get("face") or {}).get("embedding_mean") or []
    if not isinstance(reference, (list, tuple, np.ndarray)) \
            or len(reference) != embedding_mod.DIMS:
        reason = embedding_mod.unavailable_reason()
        return (_mk(name, 0.0, face_min, True,
                    ("No se pudo comprobar que seas tu: %s." % reason) if reason
                    else ("El perfil no guarda tu firma facial, asi que no se "
                          "ha podido comprobar que seas tu. Vuelve a construir "
                          "el perfil con tus fotos.")), 0.0, False)

    embedding = _safe(embedding_mod.face_embedding, img, face_d)
    if not embedding:
        # "The recogniser is not installed" and "there is no face in this
        # picture" are two different sentences and only one of them is about
        # her.  Without this branch a machine with no weights on disk still
        # carries a profile that HAS a signature, so every single image falls
        # into the no-face path below and is rejected with "no se detecto
        # ningun rostro" - a rejection the client cannot act on, about a face
        # that is plainly there.
        if not embedding_mod.available():
            return (_mk(name, 0.0, face_min, True,
                        "No se pudo comprobar que seas tu: %s."
                        % embedding_mod.unavailable_reason()), 0.0, False)
        if _expects_face(brief):
            # An engine that only composites, recolours and crops never draws a
            # face: every face pixel in its output came from her own photograph.
            # When such an engine returns something the detector cannot read - a
            # hard crop, a dark relight, a strong colour grade - the honest
            # reading is that the detector failed, not that she was replaced by
            # someone else, so it is reported instead of rejected.  The hard
            # failure below is untouched for engines that do invent pixels,
            # which is the case it was written for.
            if not _is_generative(brief) and _source_had_face(brief):
                return (_mk(name, 0.0, face_min, True,
                            "No se detecto rostro en la imagen generada, pero "
                            "este motor no dibuja caras: los pixeles del rostro "
                            "vienen de tu propia foto. Se avisa y no se rechaza."),
                        0.0, False)
            defects_out.append({
                "type": "face_distorted", "where": "face", "bbox": [],
                "severity": 0.9, "repairable": False,
                "detail": "No hay un rostro reconocible donde el encuadre lo exige.",
            })
            return (_mk(name, 0.0, face_min, False,
                        "No se detecto ningun rostro en la imagen generada, "
                        "pero el encuadre pedido si lo lleva."), 0.0, True)
        return (_mk(name, 0.0, face_min, True,
                    "No se detecto rostro y el encuadre no exige que se vea; "
                    "comprobacion omitida."), 0.0, False)

    similarity = _f(embedding_mod.similarity(embedding, reference), None)
    if similarity is None:
        return (_mk(name, 0.0, face_min, True,
                    "No se pudo comparar el rostro con tu firma facial, "
                    "comprobacion omitida."), 0.0, False)
    similarity = _clamp01(similarity)
    passed = similarity >= face_min
    if not passed:
        bbox = _int_bbox(face_d.get("bbox"))
        # The distance below the line is worth far more than it was against the
        # old descriptor, whose entire population lived inside 0.04 of scale:
        # here the two rejected paid results sit 0.16 and 0.26 under the line,
        # so this reads 0.79 and 0.94 instead of everything piling up at 0.55.
        # Not repairable, and that is the point: a local patch can put grain
        # back on a cheek, but a different woman's face has to be generated
        # again.
        severity = min(1.0, 0.55 + (face_min - similarity) * 1.5)
        defects_out.append({
            "type": "face_distorted", "where": "face", "bbox": bbox,
            "severity": round(severity, 3), "repairable": False,
            "detail": "El rostro no es el tuyo: no coincide con tus fotos.",
        })
    detail = ("Parecido facial %.2f (minimo %.2f), medido sobre la firma de "
              "tus %d fotos. %s"
              % (similarity, face_min,
                 int(_f((profile.get("face") or {}).get("embedding_n"), 0)),
                 "Eres tu." if passed
                 else "Esta cara no es la tuya: cambian los rasgos, no solo el "
                      "peinado o la luz."))
    return _mk(name, similarity, face_min, passed, detail), similarity, True


def _source_path(brief: dict) -> str:
    path = brief.get("source_path") or brief.get("original_path")
    return str(path) if path else ""


def _measure_source(path: str, max_side: int) -> tuple[dict, int]:
    """Measure one photograph end to end at a chosen resolution.

    The longest side actually used comes back with the measurement, because the
    noise estimate below must not pay twice for a reading it already has.
    """
    img = _safe(loader.load_image, str(path), int(max_side))
    if not isinstance(img, np.ndarray) or img.size == 0:
        return {}, 0
    side = int(max(img.shape[0], img.shape[1]))
    pose_s = _ok_dict(_safe(pose_mod.detect_pose, img), "pose no disponible")
    seg_s = _ok_dict(_safe(segment_mod.person_mask, img), "segmentacion no disponible")
    mask = seg_s.get("mask") if seg_s.get("ok") else None
    mask = mask if isinstance(mask, np.ndarray) else None
    # The face is what the head-length ruler hangs its rows from; without it
    # measure_body returns the torso rulers only and the paired test loses the
    # one profile that survives a reframe.
    face_s = _ok_dict(_safe(face_mod.detect_face, img), "rostro no detectado")
    return _ok_dict(_safe(body_mod.measure_body, img, pose_s, mask, face_s),
                    "medidas no disponibles"), side


def _source_skin(brief: dict) -> dict:
    """Skin of the photograph this image was generated from.

    The orchestrator already measures it before planning and hands it over in
    the brief, so nothing is measured twice.  A caller that only knows the path
    still gets the paired test, at the cost of one reading of one photograph.
    """
    src = brief.get("source_skin")
    if isinstance(src, dict) and src.get("ok"):
        return src
    path = _source_path(brief)
    if not path:
        return {}
    img = _safe(loader.load_image, str(path), VERIFY_MAX_SIDE)
    if not isinstance(img, np.ndarray) or img.size == 0:
        return {}
    pose_s = _ok_dict(_safe(pose_mod.detect_pose, img), "pose no disponible")
    face_s = _ok_dict(_safe(face_mod.detect_face, img), "rostro no detectado")
    return _ok_dict(_safe(skin_mod.skin_stats, img, pose_s, face_s),
                    "piel no medible")


def _source_body(brief: dict) -> tuple[dict, int]:
    """Measurements of the photograph this image was generated from.

    The orchestrator already measures the original before planning, so it passes
    the result through the brief and nothing is measured twice.  A caller that
    only knows the path still gets the paired test, at the cost of one analysis.
    Returns the measurement and the resolution it was taken at; a measurement
    handed over in the brief reports 0, since its scale is not knowable here.

    One exception to "nothing is measured twice".  The head-length profile did
    not exist when most stored analyses were written, and a reading taken
    without a face carries an empty one, so a handed-over measurement with no
    head profile is read again from the photograph when the path is known.
    An empty list cannot be told apart from a closeup where the ruler abstains
    for good reason, and for such a source the second reading is one analysis
    spent to learn that again; the alternative - trusting the empty list -
    would switch the only reframe-proof ruler off, silently, for every source
    photograph analysed before the ruler was built.
    """
    src = brief.get("source_body")
    path = _source_path(brief)
    if isinstance(src, dict) and src.get("ok"):
        if src.get(HEAD_PROFILE_KEY) or not path:
            return src, 0
    if not path:
        return {}, 0
    return _measure_source(path, VERIFY_MAX_SIDE)


def _profile_ratio(gen_profile: Any, src_profile: Any) -> dict | None:
    """Median width ratio between two silhouette profiles of the same person.

    Both profiles are lists of ``[position, width_over_unit]`` and the three
    that measure_body returns share this one comparison: the width profile
    (fractions of torso height, widths over torso length), the shape profile
    (fractions of silhouette height, widths over that height) and the head
    profile (head lengths below the chin, widths over head length).  Rows are
    sampled at the same positions in both images, so the pairing is exact and
    no interpolation is needed; rows present on one side only - a crop took
    them - are simply left out.  The median rather than the mean, because one
    bad scanline - a hand resting on a hip, a sleeve, a stray fragment of mask
    - should not carry the verdict.
    """
    if not isinstance(gen_profile, (list, tuple)) or not isinstance(src_profile, (list, tuple)):
        return None
    src_map: dict[float, float] = {}
    for row in src_profile:
        if isinstance(row, (list, tuple)) and len(row) >= 2:
            t, val = _f(row[0], None), _f(row[1], None)
            if t is not None and val is not None and val > 1e-6:
                src_map[round(float(t), 2)] = float(val)
    ratios: list[float] = []
    for row in gen_profile:
        if not (isinstance(row, (list, tuple)) and len(row) >= 2):
            continue
        t, val = _f(row[0], None), _f(row[1], None)
        if t is None or val is None:
            continue
        ref = src_map.get(round(float(t), 2))
        if ref is None or ref <= 1e-6:
            continue
        ratios.append(float(val) / ref)
    if len(ratios) < PROFILE_MIN_SAMPLES:
        return None
    arr = np.asarray(ratios, dtype=float)
    med = float(np.median(arr))
    # How far the rows disagree with EACH OTHER, as a fraction of what they
    # agree on.  The median above says how much the silhouette changed; this
    # says whether the rows are telling one story or several, which is the
    # difference between a body that was narrowed and a mask that found a
    # different amount of body.  See HEAD_ROWS_MAD_MAX for the measurements.
    mad = float(np.median(np.abs(arr - med))) / max(abs(med), 1e-6)
    return {"median": med, "n": len(ratios),
            "spread": float(np.std(arr)), "mad": mad}


def _paired_metric(body: dict, metric: str) -> float | None:
    """One metric as the paired test reads it: raw width over raw torso.

    Only for metrics measure_body kept - a width it discarded (a subject in
    profile) stays discarded - and only where the pixel widths are on record;
    anything else comes back as the stored metric.  See _PX_WIDTH for why the
    yaw-corrected value is the wrong number to hand a paired comparison.
    """
    if not isinstance(body, dict):
        return None
    stored = _f((body.get("metrics") or {}).get(metric), None)
    if stored is None:
        return None
    key = _PX_WIDTH.get(metric)
    if key:
        px = body.get("px") or {}
        width = _f(px.get(key), None)
        torso = _f(px.get("torso_len"), None)
        if width is not None and torso is not None and width > 0.0 and torso > 1e-6:
            return float(width) / float(torso)
    return float(stored)


def _clothing_changed(brief: dict) -> bool:
    """Did the user ASK for different clothes in this generation?

    Only an explicit choice counts.  The brief also carries a description of
    what she was already wearing, read off the source photograph, and treating
    that as a change would switch the silhouette check off on every single run -
    quietly disabling the protection this whole module exists to provide.
    """
    choices = brief.get("choices")
    if isinstance(choices, dict):
        for group in ("clothing", "clothing_color", "transparency"):
            if choices.get(group):
                return True
    return bool(brief.get("clothing_changed"))


def _paired_noise(brief: dict, src_body: dict, gen_side: int,
                  src_side: int) -> dict[str, float]:
    """How far this pair's numbers move while nothing about the body moves.

    The source photograph is read again at the size of the result and once more
    at 80% of what that reading really got.  Nothing about the woman changed
    between those readings, so whatever the metrics do is the instrument's own
    scatter, and a difference smaller than that scatter says nothing at all
    about her shape.  Two extra measurements is the entire budget: enough for a
    range, cheap enough to run on every verification, one of them skipped when
    the reading already in hand was taken at that very size, and the whole
    thing skipped when the caller never said which photograph this image came
    from.

    Only the head-profile entry is read now that the head profile is the only
    ruler that can reject anything; the rest are left in because they cost
    nothing beyond arithmetic on readings already taken, and because the entry
    a caller most wants when a verdict looks wrong is the one for the ruler it
    did not use.
    """
    path = _source_path(brief)
    if not path or gen_side <= 0 or not src_body.get("ok"):
        return {}

    samples = [src_body]
    seen = {int(src_side)} if src_side > 0 else set()
    first_used = 0
    # The second look steps down from the size the first look actually got, not
    # from the size it asked for.  The loader never enlarges, so a photograph
    # smaller than the result answers both requests with the very same pixels:
    # the readings would agree perfectly, the noise would be reported as zero
    # and the fixed 8% limit would come back for exactly the small uploads that
    # need the widening most.
    for position in (0, 1):
        if position == 0:
            side = int(gen_side)
        else:
            base = first_used if first_used > 0 else int(gen_side)
            side = int(round(base * NOISE_RESCALE))
        if side < NOISE_MIN_SIDE or side in seen:
            continue
        seen.add(side)
        again, used = _measure_source(path, side)
        seen.add(int(used))
        if position == 0:
            first_used = int(used)
        if isinstance(again, dict) and again.get("ok"):
            samples.append(again)
    if len(samples) < 2:
        return {}

    # usable_metrics is the profile builder's own opinion of which numbers may
    # be trusted from one photograph; asking it here keeps the noise floor and
    # the comparison arguing about the same set of metrics.
    trusted = [usable_metrics(sample) for sample in samples]

    noise: dict[str, float] = {}
    for metric in GATED_METRICS:
        values: list[float] = []
        for sample, ok in zip(samples, trusted):
            if metric not in ok:
                continue
            value = _paired_metric(sample, metric)
            if value is not None and abs(value) > 1e-9:
                values.append(float(value))
        if len(values) < 2:
            continue
        mean = float(np.mean(values))
        if abs(mean) < 1e-9:
            continue
        noise[metric] = float(max(values) - min(values)) / abs(mean)

    # The silhouette profiles are compared as a median of ratios, so their
    # noise is measured the same way they are used: one source reading against
    # another, where the honest answer is exactly 1.0.  The head profile gets
    # its own entry because it is a different ruler with a different scatter.
    for key in (WIDTH_PROFILE_KEY, HEAD_PROFILE_KEY):
        spreads: list[float] = []
        for i in range(len(samples)):
            for j in range(i + 1, len(samples)):
                ratio = _profile_ratio(samples[j].get(key), samples[i].get(key))
                if ratio is not None:
                    spreads.append(abs(ratio["median"] - 1.0))
        if spreads:
            noise[key] = float(max(spreads))
    return noise


def _noise_aware_tol(base_tol: float, observed_noise: Any,
                     max_tol: float = PAIRED_TOL_MAX) -> tuple[float, bool]:
    """The limit for one metric: never under the floor, never over the cap.

    Also reports whether the cap had to bite, so the verdict can admit that what
    could not be resolved was the instrument rather than the body.

    The cap is a parameter because the one ruler that still gates - the head
    profile - has its own.  The faintest 12% slim measured on it read 11.4%, so
    a limit that widened past HEAD_TOL_MAX would quietly forgive the change the
    client complained about, where the wider PAIRED_TOL_MAX was sized for rulers
    that no longer reject anything.
    """
    value = _f(observed_noise, None)
    if value is None or value <= 0.0:
        return float(base_tol), False
    wanted = PAIRED_NOISE_FACTOR * float(value)
    if wanted <= base_tol:
        return float(base_tol), False
    return float(min(wanted, max_tol)), bool(wanted > max_tol)


def _check_body_paired(gen_body: dict, src_body: dict, thresholds: dict,
                       brief: dict | None = None, gen_side: int = 0,
                       src_side: int = 0
                       ) -> tuple[dict, float, bool, list[str]] | None:
    """Compare the result against the very photograph it was made from.

    This is the sensitive test, and it is available almost every time, because
    the robot always transforms a real source image rather than inventing one.
    Measuring the same body in the same pose twice cancels nearly all of the
    noise that makes a population band blunt: framing, camera distance, how far
    the subject was turned, where her arms were.  What survives the subtraction
    is what the generator actually did to her shape.

    What does not survive the subtraction is the instrument's own scatter, and
    that scatter is wider than the fixed 8% this check used to insist on, which
    is why every metric is judged against a limit sized from the noise measured
    on this very pair instead of against a constant.

    Three rulers are measured and exactly ONE of them decides.  The skeletal
    ratios and the width profile divide by torso length; tested against
    constructed ground truth they report the framing rather than the body - the
    numbers are in the note above HEAD_TOL - so they are written into every
    verdict in plain Spanish and never reject anything.  The head profile
    divides by the length of her face and hangs its rows from the chin, came
    back within 1.6% of zero on every untouched and every reframed photograph,
    and read a known slim within about two points of the truth; it is the gate.

    The two kinds are not averaged.  A ruler that measures the crop and a ruler
    that measures the body do not disagree about her shape, they disagree about
    what they are measuring, and the mean of the two is a number about nothing.

    Returns None when there is no usable source measurement and nothing at all
    could be read, so the caller can fall back to the profile bands.  When
    something was measured but the head profile had no rows to pair, the check
    comes back with computed=False: it reports what it saw and neither rejects
    the image nor moves its score.
    """
    if not (isinstance(src_body, dict) and src_body.get("ok")):
        return None
    if not (gen_body.get("ok") or gen_body.get(HEAD_PROFILE_KEY)):
        return None

    brf = brief if isinstance(brief, dict) else {}
    base_tol = _f(thresholds.get("paired_tol"), PAIRED_TOL)
    gen_usable = _usable(gen_body)
    src_usable = _usable(src_body)

    offenders: list[str] = []
    notes: list[str] = []
    # What the two demoted rulers read.  It goes into the verdict in plain
    # Spanish every time so she can see the number, and it never rejects the
    # image - see the shoot-out written above HEAD_TOL for why neither of them
    # is allowed to spend one of her paid generations.
    observed: list[str] = []
    excused_widths: list[str] = []
    widened = False
    too_noisy = False

    # A wool coat makes a person wider, and that is not the generator taking a
    # liberty with her body.  New clothes can only ADD fabric over her body:
    # every garment in the catalogue is something worn on top of the skin and
    # not one of them takes anything away, so a figure that comes back WIDER
    # may be the jacket, while a figure that comes back NARROWER cannot be.
    # The head-length ruler is therefore kept when the request changed the
    # clothes and judged one-sided, instead of being switched off - switching
    # it off is what let the paid engine slim her unnoticed.  The measurement
    # backing the one-sided rule is in the HEAD_TOL note: a 6% garment-bulk
    # proxy read 4.1..6.7% on that ruler, under the 8% line, so the excuse now
    # only has to cover a genuinely bulky coat rather than every shirt.
    dressed = _clothing_changed(brf)
    head_one_sided = dressed
    prof_ratio = _profile_ratio(gen_body.get(WIDTH_PROFILE_KEY),
                                src_body.get(WIDTH_PROFILE_KEY))
    head_ratio = _profile_ratio(gen_body.get(HEAD_PROFILE_KEY),
                                src_body.get(HEAD_PROFILE_KEY))
    # Said here only when the ruler came back inside its limit.  When it did
    # not, the sentence built for the verdict below already says all of this
    # about a concrete measured percentage, and saying it twice reads like the
    # check arguing with itself.
    dressed_note = ("Ropa distinta: la ropa nueva puede ensancharte, asi que "
                    "solo se te compara por si el motor te ha estrechado.")

    # The width profile and the head profile read the very same mask, so the
    # ratio between their two medians is how far the torso unit (pose
    # landmarks) moved against the head unit (face mesh) between the two
    # images.  It no longer switches anything off, because nothing that divides
    # by torso length gates any more; it is kept because it is the cheapest
    # available explanation of WHY the torso numbers below disagree with the
    # verdict, and she is owed that explanation rather than two contradictory
    # percentages.
    unit_shift = None
    if prof_ratio is not None and head_ratio is not None \
            and head_ratio["median"] > 1e-6:
        unit_shift = abs(prof_ratio["median"] / head_ratio["median"] - 1.0)

    # ---------------------------------------------------------------- demoted
    # Ruler one: the skeletal ratios, each width divided by torso length.
    # Reported when it moves more than the old 8% line, never a rejection.
    for metric in GATED_METRICS:
        gen_val = _paired_metric(gen_body, metric)
        src_val = _paired_metric(src_body, metric)
        if gen_val is None or src_val is None or abs(src_val) < 1e-9:
            continue
        if metric not in gen_usable or metric not in src_usable:
            continue
        ratio = gen_val / src_val
        deviation = abs(ratio - 1.0)
        if deviation <= base_tol:
            continue
        label, down, up = METRIC_ES.get(metric, (metric, "menor", "mayor"))
        observed.append("%s un %d%% %s" % (label, int(round(deviation * 100.0)),
                                           down if ratio < 1.0 else up))

    # Ruler two: the same silhouette sampled at nine heights of the torso.
    if prof_ratio is not None:
        deviation = abs(prof_ratio["median"] - 1.0)
        if deviation > base_tol:
            observed.append("tu silueta un %d%% %s medida sobre el torso"
                            % (int(round(deviation * 100.0)),
                               "mas estrecha" if prof_ratio["median"] < 1.0
                               else "mas ancha"))
    if unit_shift is not None and unit_shift > UNIT_SHIFT_MAX:
        notes.append("El encuadre cambio: el torso mide un %d%% distinto "
                     "respecto a tu cabeza, asi que lo medido sobre el torso "
                     "habla del encuadre y solo se informa."
                     % int(round(unit_shift * 100.0)))

    # ------------------------------------------------------------- the verdict
    # Only the head-length profile decides, and only when it has rows to pair.
    # When it has none - a headshot with no body under the chin, a result with
    # no face mesh - there is no ruler here that survives a reframe, and the
    # honest move under the client's instruction is to report what was seen and
    # charge her nothing for it.  The check comes back uncomputed so it neither
    # rejects the image nor moves its score.
    if head_ratio is None:
        detail = ("No se pudo medir tu figura en cabezas en esta imagen (hace "
                  "falta el rostro y algo de cuerpo por debajo de la barbilla), "
                  "asi que las proporciones se informan y no rechazan.")
        if observed:
            detail += (" Medido sobre el torso: %s. Esa regla se mueve con el "
                       "encuadre, asi que no decide."
                       % "; ".join(observed[:4]))
        if dressed:
            notes.append(dressed_note)
        if notes:
            detail += " " + " ".join(notes)
        if not observed and not notes:
            return None
        check = _mk(name_body(), 0.0, HEAD_TOL, True, detail)
        check["mode"] = "pareado"
        check["compared"] = 0
        check["advisory"] = True
        return check, 1.0, False, []

    # The noise floor costs two more readings of the source photograph, so it is
    # bought only once it is known there is a verdict to size.  It is measured
    # for the ruler that gates and nothing else: on her photographs it never
    # reaches the 5.3% that would widen an 8% limit at all, but a new user's
    # uploads are not her uploads, and a limit that cannot admit its own
    # instrument is the mistake the demoted rulers were making.
    deviation = abs(head_ratio["median"] - 1.0)
    noise = _paired_noise(brf, src_body, gen_side, src_side)
    tol, capped = _noise_aware_tol(HEAD_TOL, noise.get(HEAD_PROFILE_KEY),
                                   HEAD_TOL_MAX)
    widened = tol > HEAD_TOL + 1e-9
    too_noisy = capped
    compared = head_ratio["n"]
    pct = int(round(deviation * 100.0))
    wider = head_ratio["median"] > 1.0
    # A widening the new clothes can explain is reported and not held against
    # the image, and it is not evidence that her shape survived either, so the
    # summary is told below that only half a test ran.
    excused = head_one_sided and wider
    # A reading the rows themselves do not support cannot spend one of her paid
    # images.  It is still measured, still printed and still says which way it
    # went; it just does not reject.  See HEAD_ROWS_MAD_MAX.
    rows_mad = _f(head_ratio.get("mad"))
    disputed = bool(deviation > tol and not excused
                    and rows_mad > HEAD_ROWS_MAD_MAX)
    if deviation > tol and excused:
        excused_widths.append("tu figura un %d%% mas ancha de arriba abajo" % pct)
    elif disputed:
        pass
    elif deviation > tol:
        offenders.append("tu figura es un %d%% %s de arriba abajo (silueta "
                         "medida en %d alturas, en cabezas)"
                         % (pct, "mas ancha" if wider else "mas estrecha",
                            head_ratio["n"]))

    score = _clamp01(1.0 - (deviation / max(tol, 1e-6)) / 2.0)

    generative = _is_generative(brf)
    # A difference the engine cannot physically have produced is a measurement
    # artefact, not a change to her body, so it is reported and not held against
    # the image.  Engines that really do generate keep the rejection.
    degraded = bool(offenders) and not generative
    passed = (not offenders) or degraded

    if degraded:
        detail = ("Diferencia medida respecto a la foto original: %s. Este motor "
                  "no genera cuerpo (solo cambia fondo, color, luz y encuadre), "
                  "asi que no pudo tocar tus proporciones: la diferencia es ruido "
                  "de medicion y no rechaza la imagen."
                  % "; ".join(offenders[:4]))
        score = max(score, NON_GENERATIVE_SCORE)
    elif excused_widths:
        # The reading is OVER the limit and the image passes anyway, so the
        # sentence must say that and not "tus proporciones se mantienen".
        # Printing a difference of 11% against a limit of 8% under the words
        # "se mantiene" is the contradiction this whole check was rewritten to
        # stop producing.
        detail = ("Sales un %d%% mas ancha de arriba abajo que en tu foto "
                  "original (medida en %d alturas sobre el tamano de tu "
                  "cabeza, limite del %d%%). Eso lo puede hacer la ropa que "
                  "pediste - ninguna prenda te estrecha - asi que se informa y "
                  "no se rechaza la imagen. Lo unico que se ha comprobado aqui "
                  "es que el motor no te haya estrechado."
                  % (pct, compared, int(round(tol * 100))))
    elif disputed:
        # Over the limit and NOT rejected, so the sentence has to say both and
        # say why: printing "tus proporciones se mantienen" over a 20% reading
        # is the contradiction this check was rewritten to stop producing.
        score = max(score, NON_GENERATIVE_SCORE)
        detail = ("Medida un %d%% mas %s de arriba abajo, pero las %d alturas "
                  "no se ponen de acuerdo entre ellas (discrepan un %d%%, "
                  "maximo %d%%): unas te miden igual que en tu foto y otras "
                  "mucho mas %s. Eso lo hace un encuadre que deja fuera parte "
                  "de tu cuerpo, no un adelgazamiento: al estrecharte se "
                  "mueven todas las alturas por igual. Se informa la "
                  "diferencia y no se rechaza la imagen."
                  % (pct, "ancha" if wider else "estrecha", compared,
                     int(round(rows_mad * 100)),
                     int(round(HEAD_ROWS_MAD_MAX * 100)),
                     "ancha" if wider else "estrecha"))
    elif passed:
        detail = ("Tu figura se mantiene respecto a la foto original: difiere "
                  "un %d%% sobre un limite del %d%%, medida en %d alturas "
                  "sobre el tamano de tu cabeza."
                  % (pct, int(round(tol * 100)), compared))
    else:
        detail = ("Cambiaron tus proporciones respecto a la foto original: %s."
                  % "; ".join(offenders[:4]))
    if dressed and not excused_widths:
        notes.append(dressed_note)
    # Always, on every verdict: the number the demoted rulers read, and why it
    # is not the verdict.  Two contradictory percentages with no explanation is
    # what made the old check impossible to believe.
    if observed:
        notes.append("Medido ademas sobre el torso: %s. Esa regla se mueve con "
                     "el encuadre y con la ropa, asi que se informa y no "
                     "rechaza." % "; ".join(observed[:4]))
    if widened:
        notes.append("Limite ajustado al ruido de medicion de esta pareja.")
    if too_noisy:
        notes.append("La medicion de esta pareja es demasiado ruidosa para "
                     "afinar mas; el limite se queda en el %d%% maximo."
                     % int(round(HEAD_TOL_MAX * 100)))
    if notes:
        detail += " " + " ".join(notes)
    check = _mk(name_body(), deviation, tol, passed, detail)
    check["mode"] = "pareado"
    check["compared"] = compared
    check["rows_mad"] = round(rows_mad, 4)
    if degraded or disputed:
        check["advisory"] = True
    if disputed:
        # The album shows the one-line summary, not this detail, and an image
        # whose silhouette nobody could compare must not be presented as one
        # whose proportions were checked.
        check["parcial_es"] = ("no se ha podido comparar tu silueta: el "
                               "encuadre deja fuera parte de tu cuerpo")
    # When the clothes changed, a widening is excused and only a NARROWING can
    # still fail, so what survived is half a test.  The detail says so; the
    # one-line summary the album actually shows used to say "las proporciones
    # coinciden con tus fotos", which claims more than was measured.
    if (passed and not degraded and not disputed
            and (excused_widths or head_one_sided)):
        check["parcial_es"] = ("solo se ha comprobado que el motor no te haya "
                               "estrechado, porque la ropa que pediste puede "
                               "ensancharte")
    return check, score, True, offenders


def name_body() -> str:
    return "body_proportions"


def _usable(body: dict) -> set[str]:
    """Metrics from one measurement that may be trusted, yaw correction included."""
    metrics = {str(k) for k in (body.get("metrics") or {})}
    unreliable = _unreliable_metrics(body)
    return metrics - unreliable


def _check_body(gen_body: dict, profile: dict,
                brief: dict | None = None) -> tuple[dict, float, bool, list[str]]:
    name = "body_proportions"
    bands = profile.get("body") or {}
    metrics = (gen_body.get("metrics") or {}) if gen_body.get("ok") else {}
    unreliable = _unreliable_metrics(gen_body) if gen_body.get("ok") else set()
    corrected = ({str(v) for v in (gen_body.get("corrected") or [])}
                 if gen_body.get("ok") else set())

    offenders: list[str] = []       # gated, reliable, out of band -> these fail
    notes: list[str] = []           # reported but never penalised
    gated_total = 0
    gated_ok = 0
    compared_bands = 0              # bands that had a value to judge, gated or not
    closeness: list[float] = []

    for metric, band in bands.items():
        value = _f(metrics.get(metric), None)
        mean = _f((band or {}).get("mean"), None)
        lo = _f((band or {}).get("lo"), None)
        hi = _f((band or {}).get("hi"), None)
        if value is None or mean is None or lo is None or hi is None or abs(mean) < 1e-9:
            continue
        compared_bands += 1
        # A band the profile itself marked ungated (too few usable photos to be
        # anything but a guess) reports but never rejects.  Neither can a band
        # whose width had to be capped.  profile._aggregate_body asks for 2.5
        # sigma of the scatter of her own photographs and then clips the answer
        # at +/-12% of the mean, and its docstring already says that past the
        # cap the honest move is to stop gating - the flag it wrote down,
        # band_capped, was simply never read here.  It matters: in the stored
        # profile every gated band is capped (shoulder_w_over_torso wants
        # 2.5 * 0.1358 = 43% of a mean of 0.790 and gets 12%, so the band
        # [0.696, 0.885] is narrower than the six photographs it was learned
        # from, two of which - 0.954 and 0.611 - fall outside it).  Measured
        # over her 24 originals as faithful q95 copies at 1200 px, this band
        # rejected 12 of the 14 photographs it could judge, all of them her
        # own and untouched, and 12 of 15 of the same photographs slimmed by
        # 12%: a verdict that says the same thing whether or not the body was
        # touched is not evidence about her body.  The paired test against the
        # source photograph is the gate that works (0 false alarms over the 24,
        # the 12% slim caught on every photograph it could judge); the band is
        # the fallback for when there is no source to compare against, and
        # there it must report what it sees without condemning her for it.
        capped = bool((band or {}).get("band_capped", False))
        gated = (metric in GATED_METRICS and metric not in unreliable
                 and bool((band or {}).get("gated", True)) and not capped)
        if metric in corrected:
            # Pay for the yaw correction's own error before judging anyone.
            half = (hi - lo) / 2.0
            lo, hi = mean - half * YAW_BAND_SLACK, mean + half * YAW_BAND_SLACK
        inside = lo <= value <= hi
        if gated:
            gated_total += 1
            gated_ok += 1 if inside else 0
            tol = max((hi - lo) / 2.0, 1e-6)
            closeness.append(_clamp01(1.0 - abs(value - mean) / tol))
        if inside:
            continue
        label, down, up = METRIC_ES.get(metric, (metric, "menor", "mayor"))
        pct = int(round(abs((value - mean) / abs(mean)) * 100.0))
        phrase = "%s un %d%% %s que en tus fotos" % (label, pct,
                                                     down if value < mean else up)
        if gated:
            offenders.append(phrase)
        elif metric in unreliable:
            notes.append(phrase + " (medida poco fiable, no penaliza)")
        else:
            notes.append(phrase + " (medida informativa, no penaliza)")

    if gated_total == 0:
        # Two different silences, and telling her the wrong one is a small lie:
        # nothing measurable in the picture, or measurements taken against a
        # range too wide to judge them with.
        if compared_bands:
            detail = ("Proporciones medidas, pero tus fotos varian demasiado "
                      "entre si para fijar un rango que pueda rechazar "
                      "(%d medidas comparadas, se informan sin penalizar)."
                      % compared_bands)
        else:
            detail = ("No se pudieron medir proporciones comparables, "
                      "comprobacion omitida.")
        if notes:
            detail += " Observado: " + "; ".join(notes[:4]) + "."
        return _mk(name, 1.0, 1.0, True, detail), 1.0, False, []

    inside_ratio = gated_ok / float(gated_total)
    # Same reasoning as the paired check: an engine that only composites,
    # recolours and crops cannot have moved a shoulder, so a band it falls
    # outside of is describing the measurement, not the woman.
    degraded = bool(offenders) and not _is_generative(brief or {})
    passed = (not offenders) or degraded
    if degraded:
        detail = ("Diferencia medida respecto a tus fotos: %s. Este motor no "
                  "genera cuerpo (solo cambia fondo, color, luz y encuadre), "
                  "asi que la diferencia es ruido de medicion y no rechaza la "
                  "imagen." % "; ".join(offenders[:4]))
    elif offenders:
        detail = ("Tu cuerpo cambio: %s. Comparadas %d medidas principales."
                  % ("; ".join(offenders[:4]), gated_total))
    else:
        detail = ("Proporciones dentro de tu rango medido (%d medidas comparadas)."
                  % gated_total)
    if notes:
        detail += " Ademas: " + "; ".join(notes[:3]) + "."
    score = float(np.mean(closeness)) if closeness else inside_ratio
    if degraded:
        score = max(score, NON_GENERATIVE_SCORE)
    check = _mk(name, inside_ratio, 1.0, passed, detail)
    if degraded:
        check["advisory"] = True
    return check, _clamp01(score), True, offenders


def _check_skin(gen_skin: dict, profile: dict, thresholds: dict,
                src_skin: dict | None = None) -> tuple[dict, float, bool]:
    """Has her skin tone changed?

    Measured against the photograph this image was made from whenever the
    caller says what that was, and against the pooled gallery profile only as
    the fallback.  See SKIN_PROFILE_VETO for the 168 readings that decided it.
    """
    name = "skin_tone"
    delta_max = _f(thresholds.get("delta_e_max"), 8.0)
    reference = _flist((profile.get("skin") or {}).get("lab_mean"), 3)
    generated = _flist(gen_skin.get("lab_mean"), 3) if gen_skin.get("ok") else None
    if reference is None or generated is None:
        return (_mk(name, 0.0, delta_max, True,
                    "No se pudo medir el tono de piel en ambas imagenes, "
                    "comprobacion omitida."), 1.0, False)
    prof_skin = profile.get("skin") or {}
    comparison = _safe(skin_mod.compare_skin, gen_skin,
                       {"ok": True, "lab_mean": reference,
                        "lab_std": _flist(prof_skin.get("lab_std"), 3) or [0.0, 0.0, 0.0],
                        "ita_deg": _f(prof_skin.get("ita_deg"), 0.0)})
    if isinstance(comparison, dict) and _f(comparison.get("delta_e"), None) is not None:
        delta_e = _f(comparison.get("delta_e"))
        similarity = _f(comparison.get("similarity"), None)
    else:  # CIE76 fallback keeps the check alive if compare_skin is unavailable
        delta_e = float(np.sqrt(sum((a - b) ** 2 for a, b in zip(generated, reference))))
        similarity = None
    if similarity is None:
        similarity = _clamp01(1.0 - delta_e / 25.0)

    # The decision is taken on chroma, not on the full CIE76 distance.  Exposure
    # and white balance move L far more than a real change of skin tone moves a
    # and b, and gating on L rejects the person's own untouched photographs.
    # Limits come from the profile, widened to that person's own spread.
    chroma_max = _f(thresholds.get("chroma_max_eff"), None)
    if chroma_max is None:
        chroma_max = _f(thresholds.get("chroma_max"), 6.0)
    lightness_max = _f(thresholds.get("delta_l_max_eff"), None)
    if lightness_max is None:
        lightness_max = _f(thresholds.get("delta_l_max"), 22.0)

    def _apart(from_lab):
        """Distance of the generated skin from one reference, in both senses."""
        return (float(np.sqrt((generated[1] - from_lab[1]) ** 2
                              + (generated[2] - from_lab[2]) ** 2)),
                abs(float(generated[0] - from_lab[0])))

    # The pooled profile first, because it is the fallback and it is always
    # available; then her own source photograph, which takes over the verdict.
    prof_chroma, prof_light = _apart(reference)
    source = _flist((src_skin or {}).get("lab_mean"), 3)         if isinstance(src_skin, dict) and src_skin.get("ok") else None

    if source is not None:
        d_chroma, d_light = _apart(source)
        paired = True
    else:
        d_chroma, d_light = prof_chroma, prof_light
        paired = False

    over = d_chroma > chroma_max or d_light > lightness_max
    # The veto: a departure so large that no reframe and no light explains it
    # is still a rejection even when the source photograph forgave it, because
    # an engine that shifts the whole frame shifts both sides of a pair alike.
    veto = paired and (prof_chroma > chroma_max * SKIN_PROFILE_VETO
                       or prof_light > lightness_max * SKIN_PROFILE_VETO)
    passed = not (over or veto)

    ref_es = "tu foto de origen" if paired else "el conjunto de tus fotos"
    if passed:
        detail = ("Tono de piel practicamente igual a %s (color %.1f de %.1f "
                  "permitido, luminosidad %.1f de %.1f)."
                  % (ref_es, d_chroma, chroma_max, d_light, lightness_max))
        if paired and (prof_chroma > chroma_max or prof_light > lightness_max):
            # Measured and dismissed, with the reason, rather than silently
            # dropped: the pooled mean disagreeing with her own photograph is
            # the normal case for a reframe or a light she asked for.
            detail += (" Frente al conjunto de tus fotos la diferencia sube a "
                       "color %.1f y luminosidad %.1f, pero eso lo mueve el "
                       "encuadre y la luz, no tu piel, asi que se informa y no "
                       "rechaza." % (prof_chroma, prof_light))
    elif veto and not over:
        detail = ("Tu piel se aleja demasiado de todas tus fotos (color %.1f, "
                  "luminosidad %.1f, frente a %.1f y %.1f permitidos): es "
                  "demasiado para que lo explique el encuadre o la luz."
                  % (prof_chroma, prof_light, chroma_max, lightness_max))
    elif d_chroma > chroma_max:
        detail = ("Cambio el color de tu piel respecto a %s (%.1f, el maximo "
                  "para ti es %.1f). Esto no es cuestion de luz: es el tono."
                  % (ref_es, d_chroma, chroma_max))
    else:
        detail = ("Te aclararon u oscurecieron la piel respecto a %s (%.1f de "
                  "luminosidad, maximo %.1f)."
                  % (ref_es, d_light, lightness_max))
    # The reported value stays delta E so the report page keeps one familiar
    # number, but the threshold shown is the one actually applied.
    check = _mk(name, round(d_chroma, 3), round(chroma_max, 3), passed, detail)
    check["delta_e"] = round(delta_e, 3)
    check["delta_l"] = round(d_light, 3)
    check["referencia"] = "foto de origen" if paired else "perfil"
    return check, _clamp01(similarity), True


def _smooth_severity(loss: float) -> float:
    """A texture loss expressed as a severity, linear through the rejection."""
    if loss <= 0.0:
        return 0.0
    scaled = SMOOTH_SEVERITY_MAX * loss / max(SMOOTH_TEXTURE_LOSS_MAX, 1e-6)
    return round(min(scaled, SMOOTH_SEVERITY_CAP), 3)


def _fine_at_face_px(img: np.ndarray, face: dict, target_px: float) -> float | None:
    """Fine band of the cheek with the face first brought down to target_px.

    Only ever downwards.  Shrinking discards grain the sensor really recorded,
    a loss both sides of the comparison can be made to suffer equally;
    enlarging would invent none and the two readings would not be commensurate.

    The anti-alias is applied by hand, before the resize and on every call,
    because it is the part of resampling that decides how much near-Nyquist
    noise survives: a Gaussian of TEXTURE_PREFILTER pixels at the target scale
    is the same filter on both sides whatever each side's own shrink factor,
    and INTER_AREA alone is not - measured in the note above
    SMOOTH_TEXTURE_LOSS_MAX.
    """
    box = anomaly_mod.face_box_px(img, face)
    if not box:
        return None
    k = min(1.0, float(target_px) / box[2])
    if k <= 0.0:
        return None
    work = _safe(cv2.GaussianBlur, img, (0, 0), TEXTURE_PREFILTER / k)
    if not isinstance(work, np.ndarray) or work.size == 0:
        return None
    if k < 0.995:
        wide = max(8, int(round(img.shape[1] * k)))
        high = max(8, int(round(img.shape[0] * k)))
        small = _safe(cv2.resize, work, (wide, high), None, 0, 0, cv2.INTER_AREA)
        if not isinstance(small, np.ndarray) or small.size == 0:
            return None
        work, box = small, [v * k for v in box]
    band = _ok_dict(_safe(anomaly_mod.face_skin_texture, work, {"bbox": box}),
                    "rostro no medible")
    return float(_f(band.get("fine"))) if band.get("ok") else None


def _texture_loss(img: np.ndarray, face_d: dict, brief: dict) -> dict:
    """How much of HER OWN grain this image kept, measured at one face width.

    {"ok", "loss", "fine", "ref"}.  The reference is the photograph the brief
    says this image was made from, so the answer is about this person and this
    camera; a hard-coded amplitude would be about whoever the constant was
    measured on, and this product has more than one user.  Both faces are
    reduced below the narrower of the two before either band is read, because
    over one person's own photographs that width explains the reading almost
    by itself - and reduced BOTH, so that neither side keeps a resampling noise
    the other one lost.
    """
    out = {"ok": False, "loss": 0.0, "fine": 0.0, "ref": 0.0}
    src_path = _source_path(brief)
    if not src_path:
        return out
    gen_box = anomaly_mod.face_box_px(img, face_d)
    if not gen_box:
        return out
    src_img = _safe(loader.load_image, src_path, VERIFY_MAX_SIDE)
    if not isinstance(src_img, np.ndarray) or src_img.size == 0:
        return out
    src_face = _ok_dict(_safe(face_mod.detect_face, src_img), "rostro no detectado")
    src_box = anomaly_mod.face_box_px(src_img, src_face) if src_face.get("ok") else []
    if not src_box:
        return out

    target = TEXTURE_MATCH_FACTOR * min(gen_box[2], src_box[2])
    fine = _fine_at_face_px(img, face_d, target)
    ref = _fine_at_face_px(src_img, src_face, target)
    if fine is None or ref is None or ref < SMOOTH_TEXTURE_MIN_REF:
        return out
    out["ok"] = True
    out["fine"] = round(fine, 4)
    out["ref"] = round(ref, 4)
    out["loss"] = round(1.0 - fine / ref, 4)
    return out


def _smoothing_verdict(image_path: str, img: np.ndarray, face_d: dict,
                       brief: dict, defects: list[dict]) -> dict:
    """What an oversmoothed face means for THIS image, given who made it.

    Returns {"failed", "severity", "loss", "detail"} and, when a generator did
    it, writes the measured loss into the defect list so the texture repair has
    a box to work on.

    Two measurements are involved and they answer different questions.  The
    scan's ratio - facial skin against the rest of her skin in the same frame -
    is reported, never gated: measured over her own 24 photographs it fires on
    8 of them (a bare arm in daylight carries far more fine energy than a
    cheek), and over 11 FLUX results it fires on 1, so on its own it would
    reject her real photographs and miss the retouching.  The gate compares the
    fine band of this face against the fine band of the same face in her source
    photograph, both brought to the same width first - see _texture_loss.
    """
    out = {"failed": False, "severity": 0.0, "loss": None, "detail": ""}
    existing = next((d for d in defects if d.get("type") == "oversmoothed_skin"),
                    None)

    # Her camera, not the robot: report it and leave the image alone.  This is
    # the whole reason the detector used to cap itself.
    if not _is_generative(brief) or _is_source_photograph(image_path, brief):
        if existing is not None:
            out["detail"] = ("La piel sale suavizada, pero la ha suavizado la "
                             "camara y no el robot: se informa, no se rechaza.")
        return out

    band = _texture_loss(img, face_d, brief)
    if not band.get("ok"):
        # Nothing measured is never a failure; the scan's own report survives.
        # No source photograph means no reference, and without a reference this
        # gate stays quiet rather than borrowing somebody else's skin.
        return out

    fine = _f(band.get("fine"))
    ref = _f(band.get("ref"))
    loss = _f(band.get("loss"))
    out["loss"] = round(loss, 3)
    if loss < SMOOTH_TEXTURE_LOSS_MIN:
        return out

    severity = _smooth_severity(loss)
    out["severity"] = severity
    out["failed"] = loss >= SMOOTH_TEXTURE_LOSS_MAX
    kept = 100.0 * max(1.0 - loss, 0.0)
    # Say the measured number and nothing more.  A rejection at the line means
    # 86% of her grain survived, so "casi ha desaparecido" would be us
    # exaggerating to her about her own photograph.
    if out["failed"]:
        out["detail"] = ("Te han suavizado la piel: el rostro solo conserva el "
                         "%.0f%% del grano de tu foto." % kept)
    else:
        out["detail"] = ("La piel sale algo mas lisa que en tu foto: conserva "
                         "el %.0f%% del grano." % kept)

    detail = ("piel suavizada por el generador: el rostro conserva el %.0f%% "
              "del grano de su foto de origen (%.2f frente a %.2f, medidos con "
              "el rostro al mismo tamano)" % (kept, fine, ref))
    # A defect without a box cannot be repaired locally, so fall back to the
    # normalised face box before giving up on the cheaper fix.
    bbox = _int_bbox(existing.get("bbox") if existing else None) \
        or _int_bbox(face_d.get("bbox"))
    if not bbox:
        norm = face_d.get("bbox_norm")
        if isinstance(norm, (list, tuple)) and len(norm) == 4:
            h, w = img.shape[:2]
            bbox = _int_bbox([_f(norm[0]) * w, _f(norm[1]) * h,
                              _f(norm[2]) * w, _f(norm[3]) * h])
    if existing is None:
        # The scan misses this in almost every real case, so the measurement
        # that did see it has to carry the defect itself - otherwise the repair
        # step is never asked to put the grain back.
        defects.append({"type": "oversmoothed_skin", "where": "face",
                        "bbox": bbox, "severity": severity,
                        "repairable": True, "detail": detail})
    elif severity > _f(existing.get("severity")):
        existing["severity"] = severity
        existing["detail"] = detail
        existing["repairable"] = True
        if bbox:
            existing["bbox"] = bbox
    return out


def _check_anatomy(anomalies: dict, defects: list[dict],
                   smoothing: dict | None = None) -> tuple[dict, float, bool]:
    name = "anatomy"
    smooth = smoothing if isinstance(smoothing, dict) else {}
    if not anomalies.get("ok") and not smooth.get("failed"):
        return (_mk(name, 0.0, ANATOMY_SEVERITY_MAX, True,
                    "No se pudo revisar la anatomia, comprobacion omitida."), 1.0, False)
    # Oversmoothed skin has already been judged, in context, by
    # _smoothing_verdict; the anatomical threshold applies to the rest.
    others = [d for d in defects if d.get("type") != "oversmoothed_skin"]
    worst = max([_f(d.get("severity")) for d in others], default=0.0)
    smooth_sev = _f(smooth.get("severity"))
    failed = bool(smooth.get("failed"))
    passed = worst < ANATOMY_SEVERITY_MAX and not failed
    if not defects:
        detail = "Sin anomalias anatomicas detectadas."
    else:
        listed = sorted(defects, key=lambda d: -_f(d.get("severity")))[:3]
        parts = ["%s (%s)" % (DEFECT_ES.get(d["type"], d["type"]),
                              _severity_es(_f(d.get("severity")))) for d in listed]
        detail = "Se detectaron %d problemas: %s." % (len(defects), ", ".join(parts))
    # A hand the scan could not judge is not a hand that passed.  On
    # 2026-09-04 the second paid image of the day was delivered under "Sin
    # anomalias anatomicas detectadas" with both hands visibly melted against
    # her thighs: one was cut by the bottom edge and the other measured 75 px
    # against the 170 px that digit geometry needs, so neither could produce a
    # severity that reaches the 0.60 gate.  Nothing here rejects the image for
    # it - there is no free ruler that separates those hands from her own
    # photographs' (see analysis/anomaly._hand_defects) - but the sentence the
    # client reads now says which part of her image nobody checked.
    unjudged = [u for u in (anomalies.get("unjudged") or []) if isinstance(u, dict)]
    if unjudged:
        names = _join_es([UNJUDGED_ES.get(str(u.get("where")), "una mano")
                          + ", " + str(u.get("reason") or "no medible")
                          for u in unjudged[:3]])
        detail += (" No se ha podido comprobar %s: %s. Miralas tu antes de "
                   "darla por buena." % ("una mano" if len(unjudged) == 1
                                         else "%d manos" % len(unjudged), names))
    if smooth.get("detail"):
        detail = smooth["detail"] + " " + detail
    value = smooth_sev if failed else worst
    threshold = SMOOTH_SEVERITY_MAX if failed else ANATOMY_SEVERITY_MAX
    check = _mk(name, value, threshold, passed, detail)
    if failed:
        check["fail_es"] = "te han suavizado la piel"
    if unjudged:
        # Carried separately so the one-line summary can repeat it: the album
        # shows the summary, not the check detail, and this is the one thing
        # about an approved image the client has to check with her own eyes.
        check["unjudged_es"] = ("una mano" if len(unjudged) == 1
                                else "%d manos" % len(unjudged))
    return check, _clamp01(1.0 - max(worst, smooth_sev)), True


def _check_quality(qual: dict) -> tuple[dict, float, bool]:
    name = "quality"
    if not qual.get("ok"):
        return (_mk(name, QUALITY_MIN, QUALITY_MIN, True,
                    "No se pudo evaluar la calidad tecnica, comprobacion omitida."),
                1.0, False)
    score = _clamp01(qual.get("score"))
    passed = score >= QUALITY_MIN
    detail = "Calidad tecnica %.2f (minimo %.2f)." % (score, QUALITY_MIN)
    issues = [str(i) for i in (qual.get("issues") or [])][:3]
    if issues and not passed:
        detail += " Problemas: " + ", ".join(issues) + "."
    if qual.get("beauty_filter_suspected"):
        detail += " Se sospecha filtro de belleza sobre la piel."
    return _mk(name, score, QUALITY_MIN, passed, detail), score, True


# ------------------------------------------------------------------- verdict

def _join_es(items: list[str]) -> str:
    if not items:
        return ""
    if len(items) == 1:
        return items[0]
    return ", ".join(items[:-1]) + " y " + items[-1]


def _summary(passed: bool, checks: list[dict], repairable: list[dict],
             body_offenders: list[str], skipped: list[str]) -> str:
    """One or two sentences the client will actually read."""
    if passed:
        # A check that only reported - because the engine could not have caused
        # what it measured - must not be sold to her as a confirmation.  It
        # passed; it did not verify anything, and the summary says so by leaving
        # it out of the list while its own detail explains what was seen.
        advisory = {c["name"] for c in checks if c.get("advisory")}
        # A check that only tested half of what its name promises is not a
        # confirmation of the whole thing either; it is named in its own clause
        # below instead of being folded into "coinciden con tus fotos".
        partial = {c["name"]: str(c.get("parcial_es") or "")
                   for c in checks if c.get("parcial_es")}
        verified = [CHECK_ES[n] for n in ("identity_face", "body_proportions",
                                          "skin_tone")
                    if n not in skipped and n not in advisory
                    and n not in partial]
        if verified:
            text = ("Aprobada: %s %s con tus fotos."
                    % (_join_es(verified),
                       "coincide" if len(verified) == 1 else "coinciden"))
        else:
            text = "Aprobada, aunque no se pudo comparar tu identidad en esta imagen."
        for check_name, why in partial.items():
            text += (" Sobre %s, %s." % (CHECK_ES.get(check_name, check_name), why))
        if skipped:
            text += (" No se pudo comprobar %s."
                     % _join_es([CHECK_ES[s] for s in skipped if s in CHECK_ES]))
        elif repairable:
            text += (" Quedan detalles menores que se pueden retocar sin volver "
                     "a generar la imagen.")
        # And the part of an approved image that was never checked.  See
        # _check_anatomy: at the size hands come out of these renders no ruler
        # can tell a good one from a melted one, so this sentence is the only
        # honest thing the robot can say about them.
        hands = next((str(c.get("unjudged_es") or "") for c in checks
                      if c["name"] == "anatomy"), "")
        if hands:
            text += (" Revisa %s: no se ha podido comprobar." % hands)
        return text
    reasons: list[str] = []
    for check in checks:
        if check["passed"]:
            continue
        reason = str(check.get("fail_es") or FAIL_ES.get(check["name"], check["name"]))
        if check["name"] == "body_proportions" and body_offenders:
            reason += " (" + body_offenders[0] + ")"
        reasons.append(reason)
    text = "Rechazada: " + ", ".join(reasons) + "."
    text += (" Se puede reparar solo la zona afectada sin regenerar toda la foto."
             if repairable else " Conviene generar de nuevo con el ajuste corregido.")
    return text


def verify_image(image_path: str, profile: dict, brief: dict | None = None) -> dict:
    """Measure a generated image against a stored identity profile."""
    started = time.perf_counter()
    prof = profile if isinstance(profile, dict) else {}
    brf = brief if isinstance(brief, dict) else {}
    thresholds = _thresholds(prof)

    img = _safe(loader.load_image, str(image_path), VERIFY_MAX_SIDE)
    if not isinstance(img, np.ndarray) or img.size == 0:
        return {
            "passed": False, "score": 0.0,
            "checks": [_mk("quality", 0.0, QUALITY_MIN, False,
                           "No se pudo abrir el archivo generado.")],
            "defects": [], "repairable_defects": [],
            "summary": "No se pudo leer la imagen generada, no hay nada que verificar.",
            "elapsed_ms": _ms(started),
        }

    pose_d = _ok_dict(_safe(pose_mod.detect_pose, img), "pose no disponible")
    face_d = _ok_dict(_safe(face_mod.detect_face, img), "rostro no detectado")
    seg = _ok_dict(_safe(segment_mod.person_mask, img), "segmentacion no disponible")
    person_raw = seg.get("mask") if seg.get("ok") else None
    person_arg = person_raw if isinstance(person_raw, np.ndarray) else None
    regions = _safe(segment_mod.region_masks, img, pose_d, person_arg)
    if not isinstance(regions, dict):
        regions = {}
    masks = dict(regions)
    if person_arg is not None:
        masks.setdefault("person", person_arg)

    body_d = _ok_dict(_safe(body_mod.measure_body, img, pose_d, person_arg, face_d),
                      "medidas no disponibles")
    skin_d = _ok_dict(_safe(skin_mod.skin_stats, img, pose_d, face_d), "piel no medible")
    qual_d = _ok_dict(_safe(quality_mod.assess_quality, img, str(image_path)),
                      "calidad no evaluable")
    anom_d = _ok_dict(_safe(anomaly_mod.scan_anomalies, img, pose_d, face_d, masks),
                      "anomalias no evaluadas")

    scan_defects = [d for d in (_defect(x) for x in (anom_d.get("defects") or []))
                    if d is not None]
    identity_defects: list[dict] = []

    checks: list[dict] = []
    scored: list[tuple[float, float]] = []   # (weight, normalised score), computed only
    skipped: list[str] = []                  # measured nothing -> never a failure

    face_check, face_score, face_done = _check_identity_face(
        img, face_d, prof, thresholds, brf, identity_defects)
    # Prefer the paired test against the source photograph; the profile bands are
    # the fallback for when the caller did not say what this was made from.
    # The noise floor is estimated at the size the result actually has, so the
    # comparison and its error bar are read off the same scale.
    gen_side = int(max(img.shape[0], img.shape[1]))
    src_body, src_side = _source_body(brf)
    paired = _check_body_paired(body_d, src_body, thresholds, brf,
                                gen_side, src_side)
    if paired is not None:
        body_check, body_score, body_done, offenders = paired
    else:
        body_check, body_score, body_done, offenders = _check_body(body_d, prof, brf)
    skin_check, skin_score, skin_done = _check_skin(
        skin_d, prof, thresholds, _source_skin(brf))
    # Smoothing is decided before the anatomy check reads the defects, because
    # it is the one defect whose meaning depends on who produced the pixels.
    smoothing = _smoothing_verdict(str(image_path), img, face_d, brf, scan_defects)
    anat_check, anat_score, anat_done = _check_anatomy(anom_d, scan_defects, smoothing)
    qual_check, qual_score, qual_done = _check_quality(qual_d)

    for chk, chk_score, computed in ((face_check, face_score, face_done),
                                     (body_check, body_score, body_done),
                                     (skin_check, skin_score, skin_done),
                                     (anat_check, anat_score, anat_done),
                                     (qual_check, qual_score, qual_done)):
        checks.append(chk)
        if computed:
            scored.append((chk["weight"], chk_score))
        else:
            skipped.append(chk["name"])

    # The scan may already have reported a distorted face; do not say it twice.
    if any(d["type"] == "face_distorted" for d in scan_defects):
        identity_defects = [d for d in identity_defects if d["type"] != "face_distorted"]
    defects = [d for d in (_defect(x) for x in identity_defects) if d is not None]
    defects.extend(scan_defects)
    repairable = [d for d in defects if d["repairable"] and d["bbox"]]

    total_weight = sum(w for w, _ in scored)
    score = (sum(w * s for w, s in scored) / total_weight) if total_weight > 0 else 0.0
    passed = all(c["passed"] for c in checks if c["weight"] > 0)

    # "Never fail someone for what could not be measured" is right for one
    # missing metric and catastrophic as a blanket rule: an image damaged badly
    # enough that no face, no pose and no body can be found skips EVERY identity
    # check and sails through with a high score.  That is precisely the image
    # that must not reach the client.  So a subject that cannot be located at
    # all is a failure, not an abstention.
    lost_subject = (not face_d.get("ok")) and (not pose_d.get("ok")) \
        and (not body_d.get("ok"))
    identity_skipped = {"identity_face", "body_proportions", "skin_tone"} <= set(skipped)
    if lost_subject or (identity_skipped and not face_d.get("ok")):
        passed = False
        score = min(score, 0.2)
        for chk in checks:
            if chk["name"] == "identity_face":
                chk["passed"] = False
                chk["detail"] = ("No se encuentra a la persona en la imagen "
                                 "generada: no hay rostro ni cuerpo reconocibles, "
                                 "asi que no se puede garantizar que seas tu.")
        defects.append({
            "type": "face_distorted", "where": "image", "bbox": [],
            "severity": 0.95, "repairable": False,
            "detail": "sujeto irreconocible en la imagen generada",
        })

    return {
        "passed": bool(passed),
        "score": round(_clamp01(score), 4),
        "checks": checks,
        "defects": defects,
        "repairable_defects": repairable,
        "summary": _summary(passed, checks, repairable, offenders, skipped),
        "elapsed_ms": _ms(started),
    }
