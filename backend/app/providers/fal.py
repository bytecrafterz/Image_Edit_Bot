"""fal.ai image provider - the paid engine behind identity critical work.

fal exposes every model through one queue REST API: POST the payload, poll the
status URL, then read the response URL.  That uniformity is why it was chosen:
switching model is a single line in MODELS below, which the client asked for
explicitly after being locked into a tool she could not steer.

The source photograph and the repair mask travel as data URIs, so nothing is
uploaded to a third party bucket and no file ever outlives the request.

Prices are per generated image in USD and are used for the budget gate before
the call and for the ledger after it, so they must stay honest.
"""
from __future__ import annotations

import base64
import io
import logging
import random
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
MODELS: dict[str, dict[str, Any]] = {
    # Identity work: Kontext edits the photograph it is given instead of
    # re-imagining the person, which is what keeps the face and the body honest.
    "identity": {
        "endpoint": "fal-ai/flux-pro/kontext",
        "price_usd": 0.040,
        "per_megapixel": False,
        "knobs": ("image", "guidance", "seed", "aspect"),
        "notes": "FLUX.1 Kontext [pro]: edicion guiada, conserva los rasgos.",
    },
    "identity_multi": {
        "endpoint": "fal-ai/flux-pro/kontext/multi",
        "price_usd": 0.040,
        "per_megapixel": False,
        "knobs": ("images", "guidance", "seed", "aspect"),
        "notes": "Kontext multi: acepta fotos de referencia de la persona.",
    },
    "identity_max": {
        "endpoint": "fal-ai/flux-pro/kontext/max",
        "price_usd": 0.080,
        "per_megapixel": False,
        "knobs": ("image", "guidance", "seed", "aspect"),
        "notes": "Kontext [max]: la entrega final, mas fiel y mas cara.",
    },
    "inpaint": {
        "endpoint": "fal-ai/flux-pro/v1/fill",
        "price_usd": 0.050,
        "per_megapixel": False,
        "knobs": ("image", "mask", "steps", "guidance", "seed"),
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

_ASPECTS = ((21 / 9, "21:9"), (16 / 9, "16:9"), (4 / 3, "4:3"), (3 / 2, "3:2"),
            (1.0, "1:1"), (2 / 3, "2:3"), (3 / 4, "3:4"), (9 / 16, "9:16"),
            (9 / 21, "9:21"))


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
            max_side=2048,
            needs_key=True,
            key_name="FAL_KEY",
            cost_per_image_usd=float(self._spec(DEFAULT_ROLE)["price_usd"]),
            notes="fal.ai: edicion sobre la foto real, inpainting por zona y "
                  "referencias de identidad. Precio por imagen segun modelo.",
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

        if "mask" in knobs and req.mask_path:
            payload["mask_url"] = _encode(str(req.mask_path), side, mask=True)[0]
            meta["masked"] = True

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
        if "aspect" in knobs and req.width and req.height:
            payload["aspect_ratio"] = _aspect_ratio(int(req.width), int(req.height))

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
