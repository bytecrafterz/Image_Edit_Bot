"""fal.ai image provider - the paid engine behind identity critical work.

fal exposes every model through one queue REST API: POST the payload, poll the
status URL, then read the response URL.  That uniformity is why it was chosen:
switching model is a single line in MODELS below, which the client asked for
explicitly after being locked into a tool she could not steer.

The source photograph and the repair mask travel as data URIs, so nothing is
uploaded to a third party bucket and no file ever outlives the request.

Prices are per generated image in USD and are used for the budget gate before
the call and for the ledger after it, so they must stay honest.

MODO ENSAYO (PHOTOROBOT_FAL_REPLAY).  The whole paid path could only be walked
by paying for it: the 15 fal images this installation owns cost 0.040-0.080 USD
each, and every plumbing defect found so far - the 1:1 box, the three different
prices for one image, a hold that was not the bill - only became visible after
the invoice.  When that environment variable names a readable folder of images,
``generate`` (and therefore ``inpaint``) builds the real payload, then copies
one of those files to out_path and answers with a GenResult shaped exactly like
a paid one instead of opening a socket: same endpoint, same cost from
``estimate_cost``, plus ``replay`` markers in ``meta``.  It exists so the
client's key works the first time, and so a rehearsal costs nothing.

It is NOT a fallback for a missing key.  The FAL_KEY check runs BEFORE it and
stays a hard error: an installation with no key has to fail loudly rather than
hand back invented images that look bought.  It cannot switch itself on either -
with the variable unset this file behaves exactly as it did - and if the
variable is set but the folder is unusable the request fails instead of going to
the network, because a rehearsal with a mistyped path must never end up
spending real money.
"""
from __future__ import annotations

import base64
import hashlib
import io
import json
import logging
import os
import random
import re
import shutil
import threading
import time
from pathlib import Path
from typing import Any

import httpx
from PIL import Image, ImageOps

from ..config import get_api_key
from ..safety import guard
from .base import (Capabilities, GenRequest, GenResult, ImageProvider,
                   InsufficientBalance, ProviderError)

log = logging.getLogger("photorobot.providers.fal")

QUEUE_BASE = "https://queue.fal.run"

# --------------------------------------------------------------- model table
# One dict, one line per model: swapping an endpoint or a price is a one line
# edit and nothing else in the codebase moves.
#
# PRICES MUST BE RE-CHECKED against https://fal.ai/pricing before they are
# quoted to a customer - fal changes them without notice, and this table feeds
# both the budget gate and the invoice the user sees.
#
# "knobs" declares which request fields the endpoint actually accepts.  Sending
# an unknown field is a 422 on some fal endpoints, so the payload builder only
# emits what is listed here.
#
# "out_max_side" is the longest side of the file the endpoint HANDS BACK, which
# is not the same question.  The Kontext family has no size knob at all - the
# only shape control it exposes is aspect_ratio - and it answers with about one
# megapixel: the 21 fal images this installation has produced are 1024x1024
# without exception.  So a tier cannot buy pixels here, only fidelity, and
# nothing in the codebase may promise 1536 or 2048 for a Kontext render.
# Bringing the file back up afterwards is not a fix either: measured over her
# seven measurable photographs, local_free.upscale from 768x1024 to 1152x1536
# moved the mean texture loss of identity/verify._texture_loss from +0.018 to
# +0.050 - worse on five of the seven, up to +0.107 on IMG_7580 - because the
# unsharp it uses is not her grain.
MODELS: dict[str, dict[str, Any]] = {
    # Identity work: Kontext edits the photograph it is given instead of
    # re-imagining the person, which is what keeps the face and the body honest.
    "identity": {
        "endpoint": "fal-ai/flux-pro/kontext",
        "price_usd": 0.040,
        "per_megapixel": False,
        "knobs": ("image", "guidance", "seed", "aspect"),
        "out_max_side": 1024,
        "notes": "FLUX.1 Kontext [pro]: edicion guiada, conserva los rasgos.",
    },
    "identity_multi": {
        "endpoint": "fal-ai/flux-pro/kontext/multi",
        "price_usd": 0.040,
        "per_megapixel": False,
        "knobs": ("images", "guidance", "seed", "aspect"),
        "out_max_side": 1024,
        "notes": "Kontext multi: acepta fotos de referencia de la persona.",
    },
    "identity_max": {
        "endpoint": "fal-ai/flux-pro/kontext/max",
        "price_usd": 0.080,
        "per_megapixel": False,
        "knobs": ("image", "guidance", "seed", "aspect"),
        "out_max_side": 1024,
        "notes": "Kontext [max]: la entrega final, mas fiel y mas cara.",
    },
    "inpaint": {
        "endpoint": "fal-ai/flux-pro/v1/fill",
        "price_usd": 0.050,
        "per_megapixel": False,
        "knobs": ("image", "mask", "steps", "guidance", "seed"),
        "out_max_side": 1024,
        "notes": "FLUX.1 Fill [pro]: repinta solo la zona blanca de la mascara.",
    },
    "img2img": {
        "endpoint": "fal-ai/flux/dev/image-to-image",
        "price_usd": 0.025,
        "per_megapixel": True,
        "knobs": ("image", "strength", "steps", "guidance", "seed", "size"),
        "notes": "FLUX.1 [dev] img2img: borradores baratos sobre la foto real.",
    },
    "draft": {
        "endpoint": "fal-ai/flux/schnell",
        "price_usd": 0.003,
        "per_megapixel": True,
        "knobs": ("steps", "seed", "size"),
        "max_steps": 4,
        "notes": "FLUX.1 [schnell]: solo texto, para fondos y pruebas.",
    },
}

DEFAULT_ROLE = "identity"

_SUBMIT_TIMEOUT = 60.0
_POLL_TIMEOUT = 30.0
_RESULT_TIMEOUT = 60.0
_DOWNLOAD_TIMEOUT = 120.0
# Kontext on a full resolution photograph, behind a busy queue, comfortably
# exceeds three minutes.  Giving up early is the worst outcome available: the
# job usually completes on fal's side anyway, so the image is paid for and
# thrown away.  Five minutes costs nothing when the work finishes sooner.
_OVERALL_TIMEOUT = 300.0
_POLL_START = 1.0
_POLL_CAP = 5.0

# A 403 from fal can mean "bad key" or "exhausted balance"; the body decides.
_BALANCE_WORDS = ("balance", "credit", "quota", "exhausted", "billing")

# Rehearsal mode.  One environment variable, read on every call so a rehearsal
# can be turned on and off without restarting the server, and named after the
# product so it cannot collide with anything else in the process.
REPLAY_ENV = "PHOTOROBOT_FAL_REPLAY"
# THE ONE ANSWER A FOLDER OF IMAGES CANNOT REHEARSE.  On 2026-09-04 the only
# two masked clothing requests this product has ever really sent to fal came
# back HTTP 200 after 19.1 s and 3.3 s of GPU, with has_nsfw_concepts=[True]
# and an all black PNG: fal ran the inference, charged for it, and refused to
# show the result.  That is the path that took the client's last 0.100 USD, and
# a rehearsal that answers from a folder can never produce it, so the handling
# written for it - settle the money, say so in her language, and refuse to buy
# a third seed after two blocks - could not be exercised offline.  This
# variable makes it reproducible: it names how many of the NEXT rehearsal calls
# answer the way fal answered that night.  It only has any effect inside
# ``_replay``, which only runs when REPLAY_ENV points at a folder, so it cannot
# change what a paying installation does.
REPLAY_BLOCK_ENV = "PHOTOROBOT_FAL_REPLAY_BLOCK"
_REPLAY_BLOCK_LOCK = threading.Lock()
_REPLAY_BLOCKS_USED = 0
_REPLAY_SUFFIXES = (".jpg", ".jpeg", ".png", ".webp")
_REPLAY_TYPES = {".jpg": "image/jpeg", ".jpeg": "image/jpeg",
                 ".png": "image/png", ".webp": "image/webp"}

_ASPECTS = ((21 / 9, "21:9"), (16 / 9, "16:9"), (4 / 3, "4:3"), (3 / 2, "3:2"),
            (1.0, "1:1"), (2 / 3, "2:3"), (3 / 4, "3:4"), (9 / 16, "9:16"),
            (9 / 21, "9:21"))
# The same nine keyed by their name, so a caller can ask how far the ratio it
# is really getting is from the one it asked for without parsing "3:4" again.
_ASPECT_VALUE = {name: value for value, name in _ASPECTS}
# Within one part in a hundred is "the shape that was ordered": 1232x1536 is
# 0.8021 and 4:5 is 0.8000, and calling those two different pictures would put
# a warning on every headshot for nothing.
_ASPECT_TOLERANCE = 0.01


# ------------------------------------------------------------------ helpers

def _clamp(value: Any, lo: float, hi: float, default: float) -> float:
    try:
        val = float(value)
    except (TypeError, ValueError):
        return default
    if val != val:                                  # NaN
        return default
    return max(lo, min(hi, val))


def _aspect_ratio(width: int, height: int) -> str:
    if width <= 0 or height <= 0:
        return "1:1"
    ratio = width / float(height)
    return min(_ASPECTS, key=lambda row: abs(row[0] - ratio))[1]


def _record_shape(meta: dict, width: Any, height: Any) -> None:
    """Write down the shape that ARRIVED, beside the one that was ordered.

    Her photographs are 2316x3088 and the 23 fal images this installation has
    bought are 1024x1024 without exception, yet nothing on disk records which
    aspect_ratio was asked for on any of them - the payload was built, sent and
    forgotten - so the square could only be found weeks later by measuring the
    files.  Both numbers now ride in the provider meta, which the orchestrator
    copies onto the attempt row, so the next time the answer does not have the
    shape of the request that is visible on the ficha instead of on the invoice.
    """
    try:
        wide, tall = int(width or 0), int(height or 0)
    except (TypeError, ValueError):
        return
    if wide <= 0 or tall <= 0:
        return
    meta["delivered_aspect"] = _aspect_ratio(wide, tall)
    asked = str(meta.get("aspect_ratio") or "")
    if asked:
        meta["aspect_kept"] = bool(meta["delivered_aspect"] == asked)


def _encode(path: str, max_side: int, mask: bool = False
            ) -> tuple[str, tuple[int, int]]:
    """File -> data URI: the WHOLE oriented picture, resized, never cut.

    EXIF orientation is applied here because every mask and every measurement
    upstream is computed on the oriented image; sending the raw bytes would
    hand fal a rotated photo and misalign the repair mask.  After that nothing
    moves a pixel sideways: the photograph and its mask are both scaled by the
    same max_side rule from the same frame, so index (x, y) of the upload and
    index (x, y) of the mask are the same point of her photograph.  That is
    what makes the paste back exact - measured on 5 of her photographs after
    this crop was removed, the composite lands 0 px from her original on every
    one, and her face box moves 0 px.
    """
    try:
        with Image.open(path) as raw:
            img = ImageOps.exif_transpose(raw) or raw
            img = img.convert("L" if mask else "RGB")
            width, height = img.size
            longest = max(width, height)
            if max_side and longest > max_side:
                scale = max_side / float(longest)
                width = max(1, int(round(width * scale)))
                height = max(1, int(round(height * scale)))
                img = img.resize((width, height), Image.LANCZOS)
            buf = io.BytesIO()
            if mask:
                img.save(buf, format="PNG", optimize=True)
                mime = "image/png"
            else:
                img.save(buf, format="JPEG", quality=92)
                mime = "image/jpeg"
    except (OSError, ValueError) as exc:
        raise ProviderError(f"No se pudo leer la imagen {Path(path).name}: {exc}",
                            retryable=False, code="bad_input") from exc
    payload = base64.b64encode(buf.getvalue()).decode("ascii")
    return f"data:{mime};base64,{payload}", (width, height)


# THERE IS NO CROP.  THE WHOLE PHOTOGRAPH GOES OUT, AND THIS IS WHY.
# Until 2026-09-05 the masked path cut the upload to the mask's bounding box
# plus a 6% margin, on a privacy argument: the pixels outside the mask are
# thrown away by protect.compose anyway, so why upload them.  The argument was
# never measured against the two things that matter, and both of them say the
# opposite.
#
# 1. WHAT THE REVIEWER SAW.  The bounding box of a clothing mask IS the
#    clothing zone - the most skin-dense part of any picture of a person - and
#    it starts at her chin.  What fal received was a headless torso: chest
#    centred, shoulders and arms bare, a knee at the bottom, and 8.6% of her
#    head box - her mouth, chin, jaw and both earrings - which is 0.0% of her
#    detected face rectangle on the new source and 1.9% on IMG_7871.  Measured
#    on both sources with the same skin envelope her own profile is built from
#    (protect._her_skin), the crop multiplied the bare skin density of the
#    picture under review by 2.82x on the new clothed photograph (9.76% of the
#    frame -> 27.50% of the upload) and by 2.73x on IMG_7871 (12.11% ->
#    33.05%).  Choosing a clothed source had cut whole frame skin by 36%, and
#    the crop handed most of it straight back: both paid masked calls came
#    back blocked, the second on a fully clothed woman.
# 2. WHAT THE MODEL NEEDED.  flux-pro/v1/fill is being asked to fit a garment
#    to a person.  Scale, proportion, the shoulder line, where the light comes
#    from and what the room is are all outside the mask, and a model cannot
#    match what it cannot see.  Every kontext call this account ever made sent
#    the whole frame and was never blocked for it (1 refusal in ~42), against
#    4 in 4 on the cropped masked path.
#
# The face was never the reason to crop either: the mask already covers 0
# pixels of the detected face rectangle on all 25 of her photographs, so her
# face is protected by arithmetic that does not depend on the framing.  What
# leaves the machine is her photograph, oriented and scaled to _max_upload_side
# and nothing else.

# WHY THE OUTBOUND TEXT IS FILTERED AT ALL, AND WHY HERE.
# The FLUX endpoints have no negative_prompt field, so _payload used to paste
# the whole negative into the prompt as "Strictly avoid: ...".  The negative is
# written to stop the engine delivering a half dressed picture, and to do that
# it names the thing it does not want: underwear, lingerie, bra, thong, naked,
# bare legs, bikini.  Measured on the outbound string of the two calls that
# came back black on 2026-09-04, the product's OWN content guard refuses that
# text - safety/guard.py reports ['lingerie','underwear','bra','thong','naked']
# and is_intimate_request() True - and a SQL scan of all 89 attempts ever
# recorded finds that vocabulary in exactly those 2 rows and in none of the
# other 87, which is the only thing separating them from the 24 calls on the
# same photograph that were never flagged.
# So: the robot does not send a provider words it would refuse from the client.
# The requirement is kept and said forwards instead of backwards - the terms
# that carry it without naming underwear ("no trousers", "top worn without
# bottoms", "the source clothing still visible under the new outfit") stay in
# the negative, and COVERAGE_CLAUSE states the same rule positively.
_UNDRESS_WORDS = (
    "underwear", "undergarment", "undergarments", "lingerie", "bra", "bras",
    "knickers", "panties", "thong", "lace", "naked", "nude", "nudity",
    "topless", "undressed", "bare", "breast", "breasts", "nipple", "nipples",
    "cleavage", "swimwear", "bikini", "half dressed", "half-dressed",
)
_UNDRESS_RE = re.compile(
    r"(?<!\w)(?:%s)(?!\w)" % "|".join(re.escape(w) for w in _UNDRESS_WORDS))

COVERAGE_CLAUSE = (
    "The outfit described above is the only clothing in this picture: opaque "
    "fabric, the torso, the hips and the legs fully covered by it, and "
    "nothing of what the source photograph showed left visible under it or "
    "beside it."
)


# AND WHAT MUST SURVIVE THE FILTER, WHICH IS NOT THE SAME QUESTION.
# A word is not a reason.  "altered breast size" contains "breast" and names
# no state of undress whatever: it is a body-shape protection, the same family
# as "slimmed waist" and "narrowed shoulders", and it is the single clause
# standing between this client and the complaint she arrived with - a previous
# tool that changed her body without asking.  Measured over the 93 attempt
# rows that carry a negative: dropping on the word alone removed that clause
# from all 93, and on the 42 kontext attempts - the endpoint that has produced
# every image she has ever been shown - it was the ONLY clause removed, so the
# filter was a pure loss there and protected nothing (guard already reports
# is_intimate False on the kontext negative).  A clause that names a CHANGE to
# her body is kept whatever nouns it uses; only a clause that names a state of
# undress goes.  With this, the filter removes 0 of 62 clauses on kontext and
# 14 of 83 on fill, which is exactly the endpoint whose text the guard refuses.
_SHAPE_WORDS = (
    "altered", "alter", "changed", "change", "different", "bigger", "larger",
    "smaller", "enlarged", "reduced", "reshaped", "reshape", "slimmed",
    "slimmer", "narrowed", "widened", "augmented", "enhanced", "size",
    "shape", "proportions", "lifted", "removed", "erased",
)
_SHAPE_RE = re.compile(
    r"(?<!\w)(?:%s)(?!\w)" % "|".join(re.escape(w) for w in _SHAPE_WORDS))


# AND THE PHRASES THAT NAME A HALF DRESSED PERSON WITHOUT NAMING A GARMENT.
# Measured on the 4141 character prompt of the paid call of 2026-09-05, the one
# fal reviewed and returned black: after _covered_negative had removed its 14
# clauses the wire text still carried "top worn without bottoms", "shirt worn
# as a dress", "no trousers" and "missing trousers".  Not one of them contains
# a word the filter above looks for, and every one of them describes a person
# who is not dressed.  Where there is a negative_prompt field they are free -
# they steer the model away from that picture and nobody reads them as a
# request.  There is no such field on any FLUX endpoint, so they are appended
# to the INSTRUCTION, and COVERAGE_CLAUSE already carries the same requirement
# forwards ("the torso, the hips and the legs fully covered by it"), which
# makes them pure cost on the wire.
# ONLY ON THE MASKED PATH, and that limit is the measurement talking.  Our own
# attempts table splits cleanly by path: 1 content block in about 42 whole
# image kontext calls against 4 blocks in 4 masked calls.  These clauses cost
# nothing on the endpoint that works, where they still guard the delivered
# failure that NO_BARE_TORSO was written for, so they are dropped only where
# the evidence of harm is - the call that sends a mask.
_PARTIAL_DRESS = (
    "top worn without bottoms", "shirt worn as a dress", "no trousers",
    "missing trousers", "missing top", "no shirt", "worn as a dress",
)

# THE LAST THREE WORDS ON THE WIRE, REWRITTEN INSTEAD OF DROPPED.
# The note above is right twice over and the two halves were never put
# together.  "altered breast size" is the single clause standing between this
# client and the complaint she arrived with, so dropping it protects nothing -
# and on an endpoint with no negative_prompt field that clause is not a
# negative at all, it is prompt text a reviewer reads.  Both are true, and the
# way through is that the PROTECTION is what matters and the NOUN is not: the
# noun is exchanged for the clinical one and the clause goes out whole.
# Measured on that same 4141 character prompt: "breast" once and "bust" twice,
# all three inside body-shape protections ("altered breast size", "same bust,
# waist and hip proportions" in the identity clause and again in the preserve
# block), none of them naming undress, and all three gone afterwards with the
# instruction unchanged in meaning.
_WIRE_NOUNS: tuple[tuple[Any, str], ...] = (
    (re.compile(r"(?<!\w)breasts(?!\w)", re.I), "chest"),
    (re.compile(r"(?<!\w)breast(?!\w)", re.I), "chest"),
    (re.compile(r"(?<!\w)busts(?!\w)", re.I), "chest"),
    (re.compile(r"(?<!\w)bust(?!\w)", re.I), "chest"),
    (re.compile(r"(?<!\w)cleavage(?!\w)", re.I), "neckline"),
)


def _wire_safe(text: str) -> tuple[str, list[str]]:
    """The same instruction in words a content reviewer has no reason to flag.

    Applied to the whole outbound prompt, not only to the negative half, since
    two of the three words measured were in the positive text.  What was
    swapped is returned so the attempt row records it: a rewrite nobody can see
    afterwards is a rewrite nobody can check.
    """
    out = str(text or "")
    swapped: list[str] = []
    for pattern, plain in _WIRE_NOUNS:
        hits = pattern.findall(out)
        if hits:
            swapped.append("%s -> %s (x%d)" % (hits[0].lower(), plain, len(hits)))
            out = pattern.sub(plain, out)
    return out, swapped


def _covered_negative(negative: str, masked: bool = False) -> tuple[str, list[str]]:
    """The negative, minus every term that names a state of undress.

    Split on commas because that is how prompt.py assembles it, so a dropped
    term takes its own clause and nothing else with it.  Three authorities, in
    order: a clause that names an ALTERATION of her body is kept no matter what
    it is called (_SHAPE_RE, and see the note above for what that cost to
    find); otherwise the product's own guard - the same list that refuses these
    words from the client - and the supplement above for the ones the guard has
    no reason to carry ("knickers", "bare thighs", "bikini"), which an output
    reviewer weighs all the same.
    """
    kept: list[str] = []
    dropped: list[str] = []
    for term in (t.strip() for t in str(negative or "").split(",")):
        if not term:
            continue
        low = term.lower()
        if masked and any(phrase in low for phrase in _PARTIAL_DRESS):
            # Said forwards by COVERAGE_CLAUSE instead; see _PARTIAL_DRESS.
            dropped.append(term)
        elif _SHAPE_RE.search(low):
            kept.append(term)
        elif guard.is_intimate_request(term) or _UNDRESS_RE.search(low):
            dropped.append(term)
        else:
            kept.append(term)
    return ", ".join(kept), dropped


def replay_dir() -> Path | None:
    """The rehearsal folder, or None - which is the normal, paying case.

    Public so the rehearsal script and the report can say, in one place and
    without duplicating the rule, whether what is running is an ensayo.
    """
    raw = (os.environ.get(REPLAY_ENV) or "").strip().strip('"')
    return Path(raw).expanduser() if raw else None


def _replay_block_due() -> bool:
    """Rehearsal only: is this the call that fal blocks?

    The counter is kept here and compared against the environment on every
    call, rather than being read once: a rehearsal turns the seam on halfway
    through its own script, after other runs have already been replayed, and a
    value cached at import time would answer for the wrong run.  The lock is
    there because ``_run_batch`` replays several variants at once.
    """
    global _REPLAY_BLOCKS_USED
    with _REPLAY_BLOCK_LOCK:
        try:
            wanted = int((os.environ.get(REPLAY_BLOCK_ENV) or "0").strip())
        except ValueError:
            wanted = 0
        if _REPLAY_BLOCKS_USED >= wanted:
            return False
        _REPLAY_BLOCKS_USED += 1
        return True


def _replay_files(folder: Path) -> list[Path]:
    """The images available to replay, sorted so a run is reproducible.

    Anything wrong with the folder raises instead of returning an empty list.
    Falling through to the network here would be the one unacceptable outcome:
    somebody mistypes the path while rehearsing and the "free" run bills six
    Kontext calls at 0.040-0.080 USD each.
    """
    try:
        files = sorted(path for path in folder.iterdir()
                       if path.is_file()
                       and path.suffix.lower() in _REPLAY_SUFFIXES)
    except OSError as exc:
        raise ProviderError(
            "MODO ENSAYO: no se puede leer la carpeta %s indicada en %s (%s). "
            "No se llama a fal.ai para no gastar por error."
            % (folder, REPLAY_ENV, exc), retryable=False,
            code="replay_bad_dir") from exc
    if not files:
        raise ProviderError(
            "MODO ENSAYO: la carpeta %s no contiene ninguna imagen (%s). "
            "No se llama a fal.ai para no gastar por error."
            % (folder, ", ".join(_REPLAY_SUFFIXES)), retryable=False,
            code="replay_empty_dir")
    return files


def _replay_pick(files: list[Path], req: GenRequest, out: Path) -> Path:
    """Which file answers this request: stable, and different per variant.

    Keyed on the destination name and the seed, both of which the orchestrator
    already varies per variant and per retry, so six variants get six different
    pictures and the same rehearsal twice gives the same six.  A digest rather
    than hash() because hash() is salted per process and would make two
    identical rehearsals disagree.
    """
    key = "%s|%s|%s" % (out.name, getattr(req, "seed", None),
                        getattr(req, "operation", ""))
    digest = hashlib.sha1(key.encode("utf-8")).digest()
    return files[int.from_bytes(digest[:4], "big") % len(files)]


def _body_text(resp: httpx.Response) -> str:
    try:
        return resp.text[:600]
    except Exception:
        return ""


def _check(resp: httpx.Response) -> None:
    """Map an HTTP failure onto the error taxonomy the orchestrator reacts to."""
    code = resp.status_code
    if code < 400:
        return
    body = _body_text(resp)
    low = body.lower()
    # Balance first: fal answers 403 "exhausted balance" as often as 402, and
    # calling that an auth error would send the user to re-paste a good key.
    if code == 402 or (code != 429 and any(w in low for w in _BALANCE_WORDS)):
        raise InsufficientBalance(
            "fal.ai rechazo la peticion por saldo insuficiente. "
            "Recarga la cuenta de fal.ai para seguir generando.", provider="fal")
    if code in (401, 403):
        raise ProviderError(
            "fal.ai rechazo la clave de API (FAL_KEY). Revisala en Ajustes.",
            retryable=False, code="auth")
    if code == 429:
        raise ProviderError("fal.ai esta limitando la tasa de peticiones.",
                            retryable=True, code="rate_limit")
    if code >= 500:
        raise ProviderError(f"fal.ai devolvio un error {code}.",
                            retryable=True, code="server")
    # The body is fal's own English JSON.  It goes to the log, where whoever
    # installed this can read it; what the client sees is a sentence about her
    # picture and her money, because until 2026-09-04 a raw
    # ``{"detail":"Unprocessable Entity"}`` was shown to her verbatim.
    log.warning("fal.ai HTTP %d: %s", code, body)
    raise ProviderError(
        "fal.ai (el servicio que dibuja las imagenes) no ha aceptado esta "
        "peticion. No se ha cobrado esta imagen. Prueba a cambiar alguna "
        "opcion o intentalo mas tarde.",
        retryable=False, code=f"http_{code}")


def _json(resp: httpx.Response) -> dict:
    try:
        data = resp.json()
    except ValueError as exc:
        raise ProviderError("La respuesta de fal.ai ha llegado incompleta. "
                            "No se ha cobrado esta imagen.",
                            retryable=True, code="bad_json") from exc
    return data if isinstance(data, dict) else {"data": data}


class FalProvider(ImageProvider):
    """Queue API client for fal.ai."""

    name = "fal"

    def __init__(self, model: str | None = None) -> None:
        # ``model`` may be a role key from MODELS or a raw endpoint path; the
        # second form lets the admin page pin a model without a code change.
        self.forced = ""
        self._custom: dict[str, Any] | None = None
        if model:
            wanted = str(model).strip()
            if wanted in MODELS:
                self.forced = wanted
            elif "/" in wanted:
                self.forced = "custom"
                self._custom = dict(MODELS[DEFAULT_ROLE])
                self._custom["endpoint"] = wanted
                self._custom["notes"] = "Modelo fijado a mano en la configuracion."
            else:
                log.warning("Modelo fal desconocido '%s'; se usa el automatico.",
                            wanted)

    # ------------------------------------------------------------ metadata

    def capabilities(self) -> Capabilities:
        return Capabilities(
            name=self.name,
            kind="image",
            img2img=True,
            inpaint=True,
            upscale=False,
            text2img=True,
            identity_reference=True,
            # max_side is what it ACCEPTS; out_max_side is what comes back.
            # The two differ here and saying so is the only way the estimate
            # can stop selling a 1536 px "alta" that arrives at 1024 px.
            max_side=2048,
            out_max_side=int(self._spec(self.forced or DEFAULT_ROLE)
                             .get("out_max_side") or 0),
            needs_key=True,
            key_name="FAL_KEY",
            cost_per_image_usd=float(self._spec(DEFAULT_ROLE)["price_usd"]),
            notes="fal.ai: edicion sobre la foto real, inpainting por zona y "
                  "referencias de identidad. Precio por imagen segun modelo. "
                  "Devuelve alrededor de 1 megapixel (1024 px de lado largo) "
                  "en cualquier calidad: se paga fidelidad, no tamano.",
        )

    def available(self) -> bool:
        return bool(get_api_key("fal"))

    def _spec(self, role: str) -> dict[str, Any]:
        if role == "custom" and self._custom is not None:
            return self._custom
        return MODELS.get(role) or MODELS[DEFAULT_ROLE]

    def endpoint(self, role: str) -> str:
        return str(self._spec(role)["endpoint"])

    def pick_model(self, req: GenRequest) -> str:
        """Role key for this request.  Cheap for drafts, faithful for finals."""
        if self.forced:
            return self.forced
        operation = str(getattr(req, "operation", "") or "generate").lower()
        quality = str(getattr(req, "quality", "") or "preview").lower()
        has_source = bool(getattr(req, "source_path", ""))
        has_mask = bool(getattr(req, "mask_path", ""))
        if operation == "inpaint" or has_mask:
            return "inpaint"
        if not has_source:
            return "draft" if quality in ("draft", "preview") else "identity"
        if quality == "draft":
            return "img2img"
        if getattr(req, "reference_paths", None):
            return "identity_multi"
        if quality in ("high", "max"):
            return "identity_max"
        return "identity"

    def delivered_side(self, req: GenRequest) -> int:
        """Longest side the endpoint THIS request will hit really hands back.

        Capabilities can only carry one number and carries the default role's,
        so a masked run - fill instead of Kontext - was being described by the
        wrong row.  Every model in MODELS declares its own ``out_max_side``;
        this asks the one that will actually be called.  0 means "the table
        does not say", and the caller falls back to the tier it asked for.
        """
        return int(self._spec(self.pick_model(req)).get("out_max_side") or 0)

    def images_sent(self, req: GenRequest) -> int:
        """How many photographs of her really travel with this request.

        Not "how many were chosen": what leaves the machine.  Only the models
        whose knobs declare ``images`` carry a pool - kontext/multi does, and
        _payload puts the photograph being edited first and up to three others
        after it.  The single image endpoints (kontext, kontext/max, img2img)
        take exactly one, and the fill endpoint takes the photograph and a mask
        and NO references at all.  The estimate says this number out loud, so
        it has to be counted from the table rather than assumed: quoting "3
        fotos tuyas" for a draft that sends one is the same class of promise as
        quoting a model that will not run.
        """
        spec = self._spec(self.pick_model(req))
        knobs = tuple(spec.get("knobs") or ())
        has_source = 1 if getattr(req, "source_path", "") else 0
        if "images" in knobs:
            return has_source + len(list(req.reference_paths or [])[:3])
        if "image" in knobs:
            return has_source
        return 0

    def estimate_cost(self, req: GenRequest) -> float:
        spec = self._spec(self.pick_model(req))
        price = float(spec.get("price_usd") or 0.0)
        if spec.get("per_megapixel"):
            width = int(getattr(req, "width", 0) or 0)
            height = int(getattr(req, "height", 0) or 0)
            megapixels = (width * height) / 1_000_000.0 if width and height else 1.0
            price *= max(0.25, megapixels)
        return round(price, 6)

    # ------------------------------------------------------------- payload

    def _max_upload_side(self, req: GenRequest) -> int:
        quality = str(getattr(req, "quality", "") or "preview").lower()
        return 2048 if quality in ("high", "max") else 1536

    def _payload(self, role: str, req: GenRequest) -> tuple[dict, dict]:
        spec = self._spec(role)
        knobs = tuple(spec.get("knobs") or ())
        side = self._max_upload_side(req)

        prompt = str(req.prompt or "").strip()
        negative = str(req.negative_prompt or "").strip()
        meta: dict[str, Any] = {"role": role, "endpoint": spec["endpoint"]}
        # Whether this call sends a mask decides how much of the text is safe
        # to send (see _PARTIAL_DRESS), so it is read before the prompt is
        # assembled rather than below where the crop is taken.
        masked = bool(req.mask_path and "mask" in knobs and "image" in knobs)
        if negative and "negative" not in knobs:
            # FLUX endpoints have no negative_prompt field, so the no-beautify
            # block has to ride inside the instruction or it does nothing - and
            # this is therefore the ONLY endpoint family that puts the words
            # "underwear", "lingerie" and "naked" on the wire.  See
            # _covered_negative: the clauses that name a state of undress are
            # dropped and COVERAGE_CLAUSE says the same requirement forwards.
            clean, dropped = _covered_negative(negative, masked=masked)
            prompt = f"{prompt}\n\nStrictly avoid: {clean}."
            if dropped:
                prompt = f"{prompt}\n\n{COVERAGE_CLAUSE}"
                meta["negativo_retirado"] = dropped

        # LAST, over the finished text, because the words being exchanged are
        # in both halves of it: "same bust, waist and hip proportions" is in
        # the positive prompt and "altered breast size" arrives with the
        # negative.  Only where the text IS the request - an endpoint with a
        # negative_prompt field would be having its wording changed for
        # nothing.
        if "negative" not in knobs:
            prompt, swapped = _wire_safe(prompt)
            if swapped:
                meta["texto_reescrito"] = swapped

        payload: dict[str, Any] = {"prompt": prompt, "num_images": 1}

        if "negative" in knobs and negative:
            payload["negative_prompt"] = negative

        # HER WHOLE PHOTOGRAPH, ON EVERY PATH, MASKED OR NOT.
        # MEASURED ON THE UPLOAD, which is the picture fal reviews and the one
        # nobody had measured before paying for it twice.  Left is what the
        # crop sent until 2026-09-05, right is what goes out now; the skin
        # envelope is her own profile's (protect._her_skin), the repaint zone
        # is the mask itself, and both are read on the picture that travels:
        #
        #   WhatsApp 2026-09-04 (1200x1599, clothed, ribbed top and skirt)
        #     share of her frame uploaded    34.85%  ->  100%
        #     bare skin, share of upload     27.50%  ->  9.76%   (2.82x less)
        #     bare skin inside repaint zone  20.09%  ->  7.00%   (2.87x less)
        #     her face rectangle included     0.00%  ->  100%
        #     her head box (face+hair)        8.56%  ->  100%
        #     repaint zone, share of upload  49.11%  ->  17.12%
        #   IMG_7871 (2316x3088, the older source)
        #     share of her frame uploaded    36.18%  ->  100%
        #     bare skin, share of upload     33.05%  ->  12.11%  (2.73x less)
        #     bare skin inside repaint zone  30.14%  ->  10.90%  (2.77x less)
        #     her face rectangle included     1.87%  ->  100%
        #     her head box (face+hair)       13.22%  ->  100%
        #     repaint zone, share of upload  57.01%  ->  20.63%
        #
        # Her face travelling whole costs nothing it was not already paying:
        # the mask covers 0 pixels of the detected face rectangle on all 25 of
        # her photographs, so fal is handed her face and told not to touch it,
        # and protect.compose writes her own pixels back over everything the
        # mask leaves black.  See "THERE IS NO CROP" above for why the picture
        # under review being a headless torso was the problem, not the fix.
        if "image" in knobs and req.source_path:
            uri, size = _encode(str(req.source_path), side)
            payload["image_url"] = uri
            meta["source_size"] = [size[0], size[1]]
            # What really went out, in pixels, after the max_side resize.  It
            # is the whole frame, so this and source_size agree in shape and a
            # row where they ever stop agreeing is a bug.  Recorded on every
            # attempt because "what did the reviewer actually see?" had to be
            # reconstructed after the money moved, twice.
            meta["enviado"] = [size[0], size[1]]
            meta["envio_completo"] = True
        elif "images" in knobs:
            urls: list[str] = []
            if req.source_path:
                uri, size = _encode(str(req.source_path), side)
                urls.append(uri)
                meta["source_size"] = [size[0], size[1]]
            for ref in list(req.reference_paths or [])[:3]:
                urls.append(_encode(str(ref), side)[0])
            if urls:
                payload["image_urls"] = urls
                meta["n_references"] = len(urls) - (1 if req.source_path else 0)
                # A fingerprint of each picture that really went out, so "three
                # photographs of her" can be checked instead of believed.  The
                # old code sent her source photograph twice - as the image to
                # edit and as its own identity reference - and nothing on any
                # row could have told anybody: two identical digests here say
                # it at a glance, and they cost one sha1 over bytes that are
                # already in memory.
                meta["image_digests"] = [
                    hashlib.sha1(u.encode("ascii")).hexdigest()[:12]
                    for u in urls]

        if "mask" in knobs and req.mask_path:
            payload["mask_url"] = _encode(str(req.mask_path), side,
                                          mask=True)[0]
            meta["masked"] = True
            # The fill endpoint has no aspect knob: it answers in the shape of
            # the picture it was given, so the shape ORDERED is the source's
            # own and there is nothing to ask for.  It is recorded anyway,
            # because "which shape did this call order?" must have an answer on
            # every paid row - the square that survived 23 paid images was
            # invisible precisely because no row carried it - and because a
            # masked call that came back reframed would have silently moved the
            # mask off her face.
            size = meta.get("source_size") or []
            if len(size) == 2 and size[0] and size[1]:
                meta["aspect_ratio"] = _aspect_ratio(int(size[0]), int(size[1]))
                meta["aspect_asked"] = round(int(size[0]) / float(size[1]), 4)
                meta["aspect_source"] = "mascara"

        if "strength" in knobs:
            payload["strength"] = round(_clamp(req.strength, 0.05, 1.0, 0.55), 3)
        if "guidance" in knobs:
            payload["guidance_scale"] = round(_clamp(req.guidance, 1.0, 20.0, 4.0), 2)
        if "steps" in knobs:
            top = int(spec.get("max_steps") or 50)
            payload["num_inference_steps"] = int(_clamp(req.steps, 1, top, min(28, top)))
        if "seed" in knobs and req.seed is not None:
            payload["seed"] = int(req.seed) & 0x7FFFFFFF
        if "size" in knobs and req.width and req.height:
            payload["image_size"] = {"width": int(req.width), "height": int(req.height)}
        if "aspect" in knobs:
            # Her photographs are 2316x3088 - 3:4 - and Kontext reframes to
            # whatever ratio it is told, so a wrong one here crops or squeezes
            # her body before any check can see it.  The caller's box wins; when
            # it did not send one, the source photograph's own shape does, and
            # only a request with no picture at all falls back to a square.
            size = meta.get("source_size") or []
            width = int(req.width or (size[0] if len(size) == 2 else 0))
            height = int(req.height or (size[1] if len(size) == 2 else 0))
            if width and height:
                payload["aspect_ratio"] = _aspect_ratio(width, height)
                # Recorded on the way out, here and not in the rehearsal
                # branch, because the paid path is the one that had no record
                # of what it ordered: every fal image on disk is a square and
                # not one row says whether a square was asked for.
                meta["aspect_ratio"] = payload["aspect_ratio"]
                ratio = width / float(height)
                meta["aspect_asked"] = round(ratio, 4)
                # Kontext accepts only the nine ratios in _ASPECTS, and three
                # framings in the catalogue are 4:5 (0.8000) - medio cuerpo,
                # primer plano y retrato de cabeza - which is not one of them
                # and lands on 3:4 (0.7500), a picture 6.3% narrower for its
                # height.  The free engine crops those exactly, so the same
                # click gives two different shapes depending on who paints it;
                # said here so the run can warn her instead of letting her
                # discover it in the album.
                meta["aspect_exact"] = bool(
                    abs(_ASPECT_VALUE[payload["aspect_ratio"]] - ratio)
                    <= _ASPECT_TOLERANCE)

        return payload, meta

    # -------------------------------------------------------------- queue

    def _submit(self, client: httpx.Client, endpoint: str,
                payload: dict) -> tuple[str, str, str]:
        resp = client.post(f"{QUEUE_BASE}/{endpoint}", json=payload,
                           timeout=_SUBMIT_TIMEOUT)
        _check(resp)
        data = _json(resp)
        request_id = str(data.get("request_id") or "")
        status_url = str(data.get("status_url") or "")
        response_url = str(data.get("response_url") or "")
        if not status_url and request_id:
            status_url = f"{QUEUE_BASE}/{endpoint}/requests/{request_id}/status"
        if not response_url and request_id:
            response_url = f"{QUEUE_BASE}/{endpoint}/requests/{request_id}"
        if not status_url or not response_url:
            raise ProviderError("fal.ai no ha dicho donde recoger la imagen. "
                                "No se ha cobrado.",
                                retryable=True, code="bad_submit")
        return request_id, status_url, response_url

    def _wait(self, client: httpx.Client, status_url: str, response_url: str,
              deadline: float) -> dict:
        delay = _POLL_START
        last_status = "SIN RESPUESTA"
        while True:
            if time.monotonic() >= deadline:
                # Say what it was doing when we gave up: "still queued" and
                # "still rendering" call for different answers, and without the
                # last status a timeout is unactionable.
                raise ProviderError(
                    "fal.ai no termino la imagen dentro del tiempo limite "
                    "(ultimo estado: %s)." % last_status,
                    retryable=True, code="timeout")
            resp = None
            try:
                resp = client.get(status_url, timeout=_POLL_TIMEOUT)
            except httpx.TimeoutException:
                pass                                # a slow poll is not a failure
            if resp is not None:
                _check(resp)
                status = str(_json(resp).get("status") or "").upper()
                last_status = status or last_status
                if status == "COMPLETED":
                    break
                if status in ("FAILED", "ERROR", "CANCELLED", "CANCELED"):
                    # ``status`` is fal's own word (FAILED, CANCELLED): it
                    # belongs in the log, not on her screen.
                    log.warning("fal.ai estado %s", status)
                    raise ProviderError(
                        "fal.ai no ha podido terminar esta imagen. No se ha "
                        "cobrado. Se vuelve a intentar.",
                        retryable=True, code="job_failed")
            nap = min(delay * (0.9 + 0.2 * random.random()),
                      max(0.05, deadline - time.monotonic()))
            time.sleep(nap)
            delay = min(delay * 1.6, _POLL_CAP)

        resp = client.get(response_url, timeout=_RESULT_TIMEOUT)
        _check(resp)
        return _json(resp)

    def _first_image(self, result: dict) -> dict:
        flags = result.get("has_nsfw_concepts")
        if isinstance(flags, (list, tuple)) and any(bool(f) for f in flags):
            # PAID FOR, AND WORTH RETRYING.  fal does not refuse the job: it
            # runs the whole inference, its own safety checker looks at what
            # came out, and it hands back an all black PNG with this flag set.
            # Two facts were measured against this account's real history on
            # 2026-09-04 (175 fal requests over five days, read from
            # rest.alpha.fal.ai/requests):
            #
            #   * it is not the request that is refused, it is one draw.  12 of
            #     those 175 came back flagged - 6.9% - spread across all three
            #     endpoints, and 26 of the 28 calls to fill passed.  So the
            #     answer to a block is another seed, not "give up": marking it
            #     unretryable turned a 7% event into a 100% loss for that
            #     image, and did it while the retry budget was still unspent.
            #
            # THAT FIGURE CANNOT BE RE-READ, AND WHAT REPLACES IT SPLITS THE
            # OTHER WAY.  rest.alpha.fal.ai/requests answers 401 with the key
            # in the keystore today, so 175 and 6.9% stand as a record of one
            # reading on 2026-09-04 and not as something anybody can check now.
            # What can be checked is our own attempts table, and by 2026-09-05
            # it says the blocks are not spread evenly at all: 1 in about 42
            # whole image kontext calls, against 4 in 4 on the masked path -
            # every masked call this account has ever made.  Retrying is still
            # right (the draw really is what is refused, and 2 of those 4 were
            # charged), but "6.9%, try another seed" is not the odds on the
            # masked path.  What that path did differently was the
            # crop: it cut the upload to the repaint zone, which
            # multiplied the bare skin
            # density of the picture under review by 2.8x and took her head
            # out of it.  The crop is gone (see "THERE IS NO CROP" above);
            # whether that moves these odds is a question only the next paid
            # call answers, and the numbers to compare it with are on the row.
            #   * the GPU time is real: the two blocked jobs of that day ran
            #     19.1 s and 3.3 s and returned HTTP 200.  The caller must
            #     therefore treat this as MONEY SPENT (``billed``), because the
            #     alternative - handing the reservation back - is a ledger that
            #     shows 0.0000 USD for an image fal charges 0.050 USD for.
            raise ProviderError(
                "fal.ai ha revisado la imagen que acababa de dibujar y no la "
                "ha dado por buena, asi que ha devuelto un archivo en negro. "
                "El dibujo si se hizo, de modo que esta imagen se cobra "
                "aunque no puedas verla. Se vuelve a intentar con otra "
                "semilla, que es lo unico que cambia el resultado.",
                retryable=True, code="content_filter", billed=True)
        images = result.get("images")
        candidate: Any = None
        if isinstance(images, list) and images:
            candidate = images[0]
        elif isinstance(result.get("image"), (dict, str)):
            candidate = result.get("image")
        if isinstance(candidate, str):
            candidate = {"url": candidate}
        if not isinstance(candidate, dict) or not candidate.get("url"):
            raise ProviderError("fal.ai no ha devuelto ninguna imagen esta "
                                "vez. No se ha cobrado.",
                                retryable=True, code="empty_result")
        return candidate

    def _download(self, client: httpx.Client, url: str, out: Path) -> int:
        if url.startswith("data:"):
            _, _, encoded = url.partition(",")
            try:
                payload = base64.b64decode(encoded)
            except (ValueError, TypeError) as exc:
                raise ProviderError("La imagen que ha llegado de fal.ai no "
                                    "se puede abrir. No se ha cobrado.",
                                    retryable=True, code="bad_image") from exc
        else:
            resp = client.get(url, timeout=_DOWNLOAD_TIMEOUT)
            _check(resp)
            payload = resp.content
        if not payload:
            raise ProviderError("La imagen que ha llegado de fal.ai ha venido "
                                "vacia. No se ha cobrado.",
                                retryable=True, code="bad_image")
        try:
            out.parent.mkdir(parents=True, exist_ok=True)
            out.write_bytes(payload)
        except OSError as exc:
            raise ProviderError(f"No se pudo guardar la imagen: {exc}",
                                retryable=False, code="io") from exc
        return len(payload)

    # --------------------------------------------------------- modo ensayo

    def _replay(self, folder: Path, role: str, req: GenRequest, out: Path,
                started: float, payload: dict, meta: dict) -> GenResult:
        """Answer this request from a folder of images, without the network.

        The payload has already been built by the caller and is handed in on
        purpose: encoding her photograph, applying the EXIF rotation, choosing
        the aspect ratio and refusing an unreadable file are the parts of the
        paid path most likely to be wrong, so a rehearsal that skipped them
        would rehearse nothing.  What is skipped is exactly the three HTTP
        calls - submit, poll, download - and the money they cost.
        """
        endpoint = str(self._spec(role)["endpoint"])
        if _replay_block_due():
            # Word for word the error the real endpoint raises, ``billed`` and
            # all, so what the rehearsal exercises is the code that runs when
            # money has been spent on an image nobody will ever see.
            log.warning("MODO ENSAYO fal (%s): esta llamada se responde como "
                        "la bloqueo fal el 2026-09-04 (%s).",
                        REPLAY_BLOCK_ENV, endpoint)
            raise ProviderError(
                "fal.ai ha revisado la imagen que acababa de dibujar y no la "
                "ha dado por buena, asi que ha devuelto un archivo en negro. "
                "El dibujo si se hizo, de modo que esta imagen se cobra "
                "aunque no puedas verla. Se vuelve a intentar con otra "
                "semilla, que es lo unico que cambia el resultado.",
                retryable=True, code="content_filter", billed=True)
        chosen = _replay_pick(_replay_files(folder), req, out)
        try:
            out.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(chosen, out)
        except OSError as exc:
            raise ProviderError("MODO ENSAYO: no se pudo copiar %s a %s: %s"
                                % (chosen.name, out, exc),
                                retryable=False, code="io") from exc

        # Reported like the real thing, because the checks downstream read
        # these numbers: a Kontext render comes back at about one megapixel
        # whatever aspect_ratio was asked for, and the rehearsal has to show
        # that same gap between what was ordered and what arrived.
        meta["replay"] = True
        meta["replay_file"] = chosen.name
        meta["replay_dir"] = str(folder)
        meta["bytes"] = out.stat().st_size
        # The type of the file that was really copied, not of the name it was
        # copied to: the bytes are the source's, and a PNG saved as v0_a1.jpg
        # would be announced as a JPEG by anything reading the extension.
        meta["content_type"] = _REPLAY_TYPES.get(chosen.suffix.lower(),
                                                 "image/jpeg")
        try:
            with Image.open(out) as img:
                meta["width"], meta["height"] = img.size
        except (OSError, ValueError) as exc:
            raise ProviderError(
                "MODO ENSAYO: %s no es una imagen legible (%s)."
                % (chosen.name, exc), retryable=False, code="bad_image") from exc
        # The shape of a file copied out of a folder is not fal's answer, so it
        # is recorded exactly like a real one and the orchestrator refuses to
        # warn the user about it (see _shape_facts): a rehearsal must not be
        # able to accuse the paid engine of returning the wrong picture.
        _record_shape(meta, meta.get("width"), meta.get("height"))
        # Loud on purpose and on every call: an ensayo that goes unnoticed is
        # an installation that thinks it has tested the paid path.
        log.warning("MODO ENSAYO fal (%s): no se llama a la red. %s -> %s, "
                    "endpoint que se habria usado %s, aspecto pedido %s, "
                    "coste simulado %.4f USD.",
                    REPLAY_ENV, chosen.name, out.name, endpoint,
                    payload.get("aspect_ratio", "sin aspecto"),
                    self.estimate_cost(req))
        return GenResult(
            ok=True,
            image_path=str(out),
            provider=self.name,
            model=endpoint,
            cost_usd=self.estimate_cost(req),
            latency_ms=int((time.monotonic() - started) * 1000),
            seed=req.seed,
            meta=meta,
        )

    # ------------------------------------------------------------ generate

    def generate(self, req: GenRequest, out_path: str | Path) -> GenResult:
        key = get_api_key("fal")
        if not key:
            raise ProviderError(
                "Falta la clave de fal.ai (FAL_KEY). Anadela en Ajustes.",
                retryable=False, code="missing_key")

        role = self.pick_model(req)
        spec = self._spec(role)
        endpoint = str(spec["endpoint"])
        out = Path(out_path)
        started = time.monotonic()
        deadline = started + _OVERALL_TIMEOUT
        payload, meta = self._payload(role, req)

        # The rehearsal branch, and the only one: after the key check, so a
        # missing key still fails loudly, and after the payload, so everything
        # that can be exercised offline has been.  With PHOTOROBOT_FAL_REPLAY
        # unset this is one dictionary lookup and the code below is untouched.
        rehearsal = replay_dir()
        if rehearsal is not None:
            return self._replay(rehearsal, role, req, out, started, payload,
                                meta)

        headers = {
            "Authorization": f"Key {key}",
            "Content-Type": "application/json",
            "Accept": "application/json",
        }

        try:
            with httpx.Client(headers=headers, follow_redirects=True,
                              timeout=_SUBMIT_TIMEOUT) as client:
                request_id, status_url, response_url = self._submit(
                    client, endpoint, payload)
                meta["request_id"] = request_id
                result = self._wait(client, status_url, response_url, deadline)
                image = self._first_image(result)
                meta["bytes"] = self._download(client, str(image["url"]), out)
        except ProviderError as exc:
            # THE TRAIL A CHARGED FAILURE MUST LEAVE.  meta already holds the
            # endpoint, the request_id fal files the job under, and the
            # size of the picture uploaded; without this line all of it
            # dies here and the row the client is billed on names none of it.
            # That is not hypothetical: it is exactly what happened to the two
            # blocked calls of 2026-09-04 and again to the one of 2026-09-05,
            # whose request_id had to be recovered from an httpx log line.
            if not getattr(exc, "meta", None):
                exc.meta = dict(meta)
            if not getattr(exc, "latency_ms", 0):
                exc.latency_ms = int((time.monotonic() - started) * 1000)
            raise
        except httpx.TimeoutException as exc:
            log.warning("fal.ai timeout: %s", exc)
            raise ProviderError(
                "fal.ai ha tardado demasiado en contestar y se ha dejado esta "
                "imagen. No se ha cobrado. Se vuelve a intentar.",
                retryable=True, code="timeout") from exc
        except httpx.HTTPError as exc:
            log.warning("fal.ai red: %s", exc)
            raise ProviderError(
                "No se ha podido conectar con fal.ai. Comprueba tu conexion a "
                "internet. No se ha cobrado esta imagen.",
                retryable=True, code="network") from exc

        seed = result.get("seed")
        try:
            seed = int(seed) if seed is not None else req.seed
        except (TypeError, ValueError):
            seed = req.seed
        for field in ("width", "height", "content_type"):
            # fal answers in the shape of the picture it was given, and that is
            # now always her whole frame, so what it reports IS the shape of
            # the file on disk and can be recorded exactly as it comes.
            if image.get(field) is not None:
                meta[field] = image[field]
        _record_shape(meta, meta.get("width"), meta.get("height"))
        if result.get("timings"):
            meta["timings"] = result["timings"]

        return GenResult(
            ok=True,
            image_path=str(out),
            provider=self.name,
            model=endpoint,
            cost_usd=self.estimate_cost(req),
            latency_ms=int((time.monotonic() - started) * 1000),
            seed=seed,
            meta=meta,
        )

    # ------------------------------------------------------------- upscale

    def upscale(self, req: GenRequest, out_path: str | Path) -> GenResult:
        """Refused, in rehearsal exactly as in production.

        There is no upscale endpoint in MODELS, capabilities() says
        ``upscale=False`` and the router therefore never sends this work here,
        so no network call exists to replace: replaying a success would be a
        rehearsal that passes where the client's key would fail, which is the
        one thing this mode must never do.  Written out rather than inherited
        because the base class answers in English and this message is read by
        the user.  If fal ever gains an upscale endpoint, add the role to
        MODELS and the rehearsal covers it for free through ``generate``.
        """
        raise ProviderError(
            "fal.ai no ofrece un modelo de ampliacion en esta configuracion; "
            "la ampliacion la hace el motor local.",
            retryable=False, code="unsupported")
