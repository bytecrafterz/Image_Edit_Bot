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
import logging
import os
import random
import shutil
import time
from pathlib import Path
from typing import Any

import httpx
from PIL import Image, ImageOps

from ..config import get_api_key
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


def _encode(path: str, max_side: int, mask: bool = False) -> tuple[str, tuple[int, int]]:
    """File -> data URI.

    EXIF orientation is applied here because every mask and every measurement
    upstream is computed on the oriented image; sending the raw bytes would
    hand fal a rotated photo and misalign the repair mask.
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


def replay_dir() -> Path | None:
    """The rehearsal folder, or None - which is the normal, paying case.

    Public so the rehearsal script and the report can say, in one place and
    without duplicating the rule, whether what is running is an ensayo.
    """
    raw = (os.environ.get(REPLAY_ENV) or "").strip().strip('"')
    return Path(raw).expanduser() if raw else None


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
    raise ProviderError(f"fal.ai devolvio {code}: {body}",
                        retryable=False, code=f"http_{code}")


def _json(resp: httpx.Response) -> dict:
    try:
        data = resp.json()
    except ValueError as exc:
        raise ProviderError("fal.ai devolvio una respuesta ilegible.",
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
        if negative and "negative" not in knobs:
            # FLUX endpoints have no negative_prompt field, so the no-beautify
            # block has to ride inside the instruction or it does nothing.
            prompt = f"{prompt}\n\nStrictly avoid: {negative}."

        payload: dict[str, Any] = {"prompt": prompt, "num_images": 1}
        meta: dict[str, Any] = {"role": role, "endpoint": spec["endpoint"]}

        if "negative" in knobs and negative:
            payload["negative_prompt"] = negative

        if "image" in knobs and req.source_path:
            uri, size = _encode(str(req.source_path), side)
            payload["image_url"] = uri
            meta["source_size"] = [size[0], size[1]]
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
            payload["mask_url"] = _encode(str(req.mask_path), side, mask=True)[0]
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
            raise ProviderError("fal.ai no devolvio la URL de la peticion.",
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
                    raise ProviderError(f"fal.ai marco la peticion como {status}.",
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
            raise ProviderError(
                "El filtro de contenido de fal.ai bloqueo el resultado. "
                "Cambia el encuadre o la ropa descrita en las opciones.",
                retryable=False, code="content_filter")
        images = result.get("images")
        candidate: Any = None
        if isinstance(images, list) and images:
            candidate = images[0]
        elif isinstance(result.get("image"), (dict, str)):
            candidate = result.get("image")
        if isinstance(candidate, str):
            candidate = {"url": candidate}
        if not isinstance(candidate, dict) or not candidate.get("url"):
            raise ProviderError("fal.ai no devolvio ninguna imagen.",
                                retryable=True, code="empty_result")
        return candidate

    def _download(self, client: httpx.Client, url: str, out: Path) -> int:
        if url.startswith("data:"):
            _, _, encoded = url.partition(",")
            try:
                payload = base64.b64decode(encoded)
            except (ValueError, TypeError) as exc:
                raise ProviderError("fal.ai devolvio una imagen ilegible.",
                                    retryable=True, code="bad_image") from exc
        else:
            resp = client.get(url, timeout=_DOWNLOAD_TIMEOUT)
            _check(resp)
            payload = resp.content
        if not payload:
            raise ProviderError("La imagen descargada de fal.ai esta vacia.",
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
        except ProviderError:
            raise
        except httpx.TimeoutException as exc:
            raise ProviderError(f"fal.ai no respondio a tiempo: {exc}",
                                retryable=True, code="timeout") from exc
        except httpx.HTTPError as exc:
            raise ProviderError(f"Fallo de red hablando con fal.ai: {exc}",
                                retryable=True, code="network") from exc

        seed = result.get("seed")
        try:
            seed = int(seed) if seed is not None else req.seed
        except (TypeError, ValueError):
            seed = req.seed
        for field in ("width", "height", "content_type"):
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
