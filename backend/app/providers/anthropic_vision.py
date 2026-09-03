"""Claude as the robot's eyes and its prompt writer.

Three jobs, all of them the ones a human otherwise does by hand: read the
source photograph and say what must be preserved, turn a brief into a precise
prompt, and inspect a generated image for the defects that make a render
unusable.  The numeric identity verdict is NOT delegated here - identity is
measured in identity/verify.py.  Claude is asked for what a measurement cannot
see: hands, smeared textures, plastic skin, a face that is not the same person.

Raw HTTP over httpx on purpose: the Anthropic SDK is deliberately not in
requirements.txt, so this module speaks the Messages API directly.  It never
raises - a missing key, a dead network or an unparseable answer all degrade to
the empty contract dict, and the caller falls back to the free local vision.
"""
from __future__ import annotations

import base64
import io
import json
import logging
import time
from typing import Any

import httpx
from PIL import Image, ImageOps

from ..config import get_api_key
from .base import VisionProvider

log = logging.getLogger("photorobot.providers.claude")

API_URL = "https://api.anthropic.com/v1/messages"
API_VERSION = "2023-06-01"

DEFAULT_MODEL = "claude-sonnet-5"

# USD per token.  RE-CHECK against https://www.anthropic.com/pricing before
# billing a customer with them: these move, and the ledger must not lie.
# (Published rates are per million tokens: Sonnet 5 = 2 / 10, Opus 5 = 5 / 25,
# Haiku 4.5 = 1 / 5, Sonnet 4.6 = 3 / 15.)
PRICES: dict[str, dict[str, float]] = {
    "claude-sonnet-5": {"in": 2.00e-6, "out": 10.00e-6},
    "claude-opus-5": {"in": 5.00e-6, "out": 25.00e-6},
    "claude-sonnet-4-6": {"in": 3.00e-6, "out": 15.00e-6},
    "claude-haiku-4-5": {"in": 1.00e-6, "out": 5.00e-6},
}
_FALLBACK_PRICE = {"in": 3.00e-6, "out": 15.00e-6}

# Cache writes cost 1.25x an input token, cache reads 0.1x.
_CACHE_WRITE_FACTOR = 1.25
_CACHE_READ_FACTOR = 0.10

MAX_SIDE = 1024          # an image costs about (w*h)/750 tokens; 1024 caps it
_TIMEOUT = 90.0
_ATTEMPTS = 2

# Rough per-call token shape, used only by estimate_cost() before the call.
_EST_INPUT_TOKENS = 2100
_EST_OUTPUT_TOKENS = 700

DEFECT_TYPES = (
    # the contract enum from docs/CONTRACTS.md ...
    "hand_malformed", "extra_limb", "extra_person", "face_distorted",
    "eye_asymmetry", "missing_limb", "texture_smear", "duplicated_feature",
    "border_artifact", "oversmoothed_skin",
    # ... plus three the client's complaint requires and pixels alone cannot
    # localise.  They are never repairable by inpainting: a slimmed body or a
    # shifted skin tone means the whole render is wrong.
    "body_reshaped", "skin_tone_changed", "mark_removed",
)

_NON_REPAIRABLE = ("body_reshaped", "skin_tone_changed", "mark_removed")

_DESCRIBE_SYSTEM = """You are the vision stage of an automated photo studio.
You look at ONE source photograph of a real person and report what a
regenerated version of it must keep identical.

Answer with ONE JSON object and nothing else. No prose, no markdown, no code
fences. Every key is required; use "" or [] when you cannot tell.

{"shot_type": "closeup" | "half" | "full" | "unknown",
 "subject": "who and how framed, no names, no age or ethnicity guesses",
 "clothing": "garments, colours, sleeve and neckline shape",
 "hair": "length, colour, texture, how it is worn",
 "expression": "mouth, eyes, gaze direction",
 "pose": "body position, arms, hands, orientation to the camera",
 "setting": "background and location",
 "lighting": "direction, hardness, colour temperature",
 "camera": "framing, angle, apparent focal length, depth of field",
 "colors": ["2 to 4 dominant colours as plain English names"],
 "preserve": ["short phrases naming what must not change"],
 "notes": "one sentence in Spanish, no accented characters"}

Rules:
- Every value except "notes" is English prompt material: short, literal, no
  adjectives of praise, no invented details.
- "preserve" is the important one. List face shape and features, body
  proportions, skin tone, visible tattoos or scars, hair length and colour,
  glasses, and garment colour - only the ones actually visible.
- Never state or guess a name, an age or an ethnicity.
"""

_CRITIQUE_SYSTEM = """You are the quality inspector of an automated photo
studio. You are shown a GENERATED image and must find the defects that make it
unusable. Be strict and literal; a clean image gets an empty defect list.

Answer with ONE JSON object and nothing else. No prose, no markdown, no code
fences.

{"score": 0.0 to 1.0,
 "identity_notes": "one or two sentences in Spanish, no accented characters",
 "defects": [{"type": "<one of the types below>",
              "where": "left_hand | right_hand | face | eyes | torso | arms |
                        legs | hair | background | global",
              "severity": 0.0 to 1.0,
              "repairable": true if repainting only that box would fix it,
              "bbox": [x, y, w, h] normalised 0..1 of the whole image,
              "detail": "short explanation in Spanish, no accented characters"}]}

Inspect for these, in this order, and use exactly these type values:
1. hand_malformed - wrong finger count, fused, bent or duplicated digits.
   Count the fingers on every visible hand.
2. body_reshaped - the waist or shoulders are slimmer or narrower than a real
   photograph of this body would be, or limb lengths are altered.
3. skin_tone_changed - the skin is lighter, darker or a different hue than the
   rest of the person implies.
4. mark_removed - a tattoo, scar, mole or birthmark that belongs on this body
   is missing or smoothed away.
5. oversmoothed_skin - plastic, airbrushed, poreless or beauty filtered skin.
6. face_distorted - the face is not the same person, or is warped, melted or
   asymmetric beyond what a camera does.
7. eye_asymmetry, extra_limb, extra_person, missing_limb, duplicated_feature,
   texture_smear, border_artifact - the usual generator failures.

Rules:
- body_reshaped, skin_tone_changed and mark_removed are NEVER repairable:
  set "repairable": false for them.
- bbox is normalised 0..1 as [x, y, w, h]; use [0,0,1,1] for a global defect.
- score is your overall usability of the image, where 1.0 is a photograph
  nobody would question.
- Report only what you can actually see. Do not invent defects to be helpful.
"""

_PROMPT_SYSTEM = """You write prompts for an image model that will edit a real
photograph of a real person. Identity is the product: the face, the body
proportions, the skin tone and every mark must survive the edit.

Answer with ONE JSON object and nothing else. No prose, no markdown, no code
fences.

{"prompt": "the positive prompt, English, one paragraph",
 "negative_prompt": "comma separated things to avoid, English",
 "identity_clause": "the sentence inside the prompt that pins the identity",
 "tokens": ["the key phrases you used"],
 "notes": "one sentence in Spanish, no accented characters"}

Rules:
- The prompt describes the desired result, not the process. No camera brand
  names, no artist names, no quality spam.
- The negative prompt must contain, verbatim: slimmer body, slimmed waist,
  narrowed shoulders, reshaped face, airbrushed skin, plastic skin, beauty
  filter, face slimming, body slimming, changed skin tone, removed tattoos,
  altered breast size, different person.
- Never invent physical traits that the brief does not state.
"""


# ------------------------------------------------------------------ parsing

def _extract_json(text: str) -> dict | None:
    """Pull the outermost JSON object out of a model answer.

    Tolerates code fences, a leading apology and trailing commentary, because
    a parse failure must degrade to "no opinion", never to an exception.
    """
    if not text:
        return None
    cleaned = text.strip()
    if "```" in cleaned:
        parts = cleaned.split("```")
        best = ""
        for i in range(1, len(parts), 2):
            chunk = parts[i]
            if chunk[:4].lower() == "json":
                chunk = chunk[4:]
            if len(chunk.strip()) > len(best):
                best = chunk.strip()
        if best:
            cleaned = best
    start = cleaned.find("{")
    if start < 0:
        return None
    depth = 0
    in_string = False
    escaped = False
    for i in range(start, len(cleaned)):
        ch = cleaned[i]
        if in_string:
            if escaped:
                escaped = False
            elif ch == "\\":
                escaped = True
            elif ch == '"':
                in_string = False
            continue
        if ch == '"':
            in_string = True
        elif ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                try:
                    data = json.loads(cleaned[start:i + 1])
                except ValueError:
                    return None
                return data if isinstance(data, dict) else None
    try:                                    # truncated answer: last resort
        data = json.loads(cleaned[start:])
    except ValueError:
        return None
    return data if isinstance(data, dict) else None


def _s(value: Any, limit: int = 400) -> str:
    if value is None or isinstance(value, (dict, list, tuple)):
        return ""
    text = str(value).strip()
    return text[:limit]


def _str_list(value: Any, limit: int = 12) -> list[str]:
    if isinstance(value, str):
        value = [p.strip() for p in value.split(",")]
    if not isinstance(value, (list, tuple)):
        return []
    out = []
    for item in value:
        text = _s(item, 160)
        if text:
            out.append(text)
        if len(out) >= limit:
            break
    return out


def _f(value: Any, default: float = 0.0) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return default
    if out != out:
        return default
    return out


def _clamp01(value: float) -> float:
    return 0.0 if value < 0.0 else 1.0 if value > 1.0 else value


def _blank_description(note: str = "") -> dict:
    return {"shot_type": "unknown", "subject": "", "clothing": "", "hair": "",
            "expression": "", "pose": "", "setting": "", "lighting": "",
            "camera": "", "colors": [], "preserve": [], "notes": note,
            "cost_usd": 0.0}


def _blank_critique(note: str = "") -> dict:
    return {"ok": False, "score": 0.0, "defects": [], "identity_notes": note,
            "cost_usd": 0.0}


# ------------------------------------------------------------------ images

def _encode_image(path: str) -> tuple[str, str, int, int] | None:
    """(base64, media_type, full_width, full_height), downscaled to MAX_SIDE.

    The full size is returned because Claude answers with normalised boxes and
    the repair stage needs them in the pixels of the untouched image.  EXIF
    orientation is applied so those pixels match every other measurement.
    """
    try:
        with Image.open(path) as raw:
            img = ImageOps.exif_transpose(raw) or raw
            img = img.convert("RGB")
            full_w, full_h = img.size
            longest = max(full_w, full_h)
            if longest > MAX_SIDE:
                scale = MAX_SIDE / float(longest)
                img = img.resize((max(1, int(round(full_w * scale))),
                                  max(1, int(round(full_h * scale)))),
                                 Image.LANCZOS)
            buf = io.BytesIO()
            img.save(buf, format="JPEG", quality=85)
    except (OSError, ValueError) as exc:
        log.warning("No se pudo codificar %s para Claude: %s", path, exc)
        return None
    return (base64.b64encode(buf.getvalue()).decode("ascii"), "image/jpeg",
            int(full_w), int(full_h))


def _image_block(encoded: tuple[str, str, int, int]) -> dict:
    return {"type": "image",
            "source": {"type": "base64", "media_type": encoded[1],
                       "data": encoded[0]}}


def _context_text(payload: Any, limit: int = 1200) -> str:
    try:
        blob = json.dumps(payload, ensure_ascii=False, default=str)
    except (TypeError, ValueError):
        return ""
    return blob[:limit]


class ClaudeVision(VisionProvider):
    """Anthropic Messages API client, JSON in and strict JSON out."""

    name = "claude"

    def __init__(self, model: str | None = None) -> None:
        self.model = str(model).strip() if model else DEFAULT_MODEL

    # ------------------------------------------------------------ metadata

    def available(self) -> bool:
        return bool(get_api_key("anthropic"))

    def _price(self) -> dict[str, float]:
        return PRICES.get(self.model, _FALLBACK_PRICE)

    def estimate_cost(self, n_images: int = 1) -> float:
        price = self._price()
        per_call = (_EST_INPUT_TOKENS * price["in"]
                    + _EST_OUTPUT_TOKENS * price["out"])
        return round(per_call * max(1, int(n_images)), 6)

    def _cost_of(self, usage: Any) -> float:
        if not isinstance(usage, dict):
            return 0.0
        price = self._price()
        tokens_in = _f(usage.get("input_tokens"))
        tokens_in += _f(usage.get("cache_creation_input_tokens")) * _CACHE_WRITE_FACTOR
        tokens_in += _f(usage.get("cache_read_input_tokens")) * _CACHE_READ_FACTOR
        tokens_out = _f(usage.get("output_tokens"))
        return round(tokens_in * price["in"] + tokens_out * price["out"], 6)

    # ------------------------------------------------------------- request

    def _ask(self, system: str, blocks: list, max_tokens: int) -> tuple:
        """(parsed dict | None, cost_usd, note).  Never raises."""
        key = get_api_key("anthropic")
        if not key:
            return None, 0.0, "Falta la clave de Anthropic."
        headers = {"x-api-key": key, "anthropic-version": API_VERSION,
                   "content-type": "application/json"}
        body = {"model": self.model, "max_tokens": int(max_tokens),
                "system": system,
                "messages": [{"role": "user", "content": blocks}]}

        note = ""
        cost = 0.0
        for attempt in range(_ATTEMPTS):
            try:
                with httpx.Client(timeout=_TIMEOUT) as client:
                    resp = client.post(API_URL, headers=headers, json=body)
            except httpx.HTTPError as exc:
                note = f"Fallo de red hablando con Claude: {exc}"
                log.warning(note)
                if attempt + 1 < _ATTEMPTS:
                    time.sleep(1.5 * (attempt + 1))
                    continue
                return None, 0.0, note

            code = resp.status_code
            if code in (401, 403):
                return None, 0.0, "Anthropic rechazo la clave de API."
            if code == 429 or code >= 500:
                note = f"Anthropic devolvio {code}."
                if attempt + 1 < _ATTEMPTS:
                    time.sleep(1.5 * (attempt + 1))
                    continue
                return None, 0.0, note
            if code >= 400:
                return None, 0.0, f"Anthropic devolvio {code}: {resp.text[:200]}"

            try:
                data = resp.json()
            except ValueError:
                return None, 0.0, "Anthropic devolvio una respuesta ilegible."
            cost = self._cost_of(data.get("usage"))
            if str(data.get("stop_reason") or "") == "refusal":
                return None, cost, "Claude declino analizar esta imagen."
            text = "".join(
                str(block.get("text") or "")
                for block in (data.get("content") or [])
                if isinstance(block, dict) and block.get("type") == "text")
            parsed = _extract_json(text)
            if parsed is None:
                log.warning("Respuesta de Claude no parseable (%d caracteres)",
                            len(text))
                return None, cost, "Claude no devolvio JSON valido."
            return parsed, cost, ""
        return None, cost, note or "Claude no respondio."

    # ------------------------------------------------------------ describe

    def describe_photo(self, image_path: str, context: dict | None = None) -> dict:
        encoded = _encode_image(str(image_path))
        if encoded is None:
            return _blank_description("No se pudo leer la imagen.")

        blocks: list[dict] = [_image_block(encoded)]
        hint = ""
        if isinstance(context, dict):
            slim = {k: context[k] for k in
                    ("person_name", "shot_type", "marks", "known_tattoos",
                     "intent", "brief") if k in context}
            hint = _context_text(slim)
        text = "Read this source photograph and answer with the JSON object."
        if hint:
            text += "\nKnown context (JSON, may be incomplete): " + hint
        blocks.append({"type": "text", "text": text})

        parsed, cost, note = self._ask(_DESCRIBE_SYSTEM, blocks, 1600)
        if parsed is None:
            out = _blank_description(note)
            out["cost_usd"] = cost
            return out

        shot = _s(parsed.get("shot_type"), 20).lower()
        out = {
            "shot_type": shot if shot in ("closeup", "half", "full") else "unknown",
            "subject": _s(parsed.get("subject")),
            "clothing": _s(parsed.get("clothing")),
            "hair": _s(parsed.get("hair")),
            "expression": _s(parsed.get("expression")),
            "pose": _s(parsed.get("pose")),
            "setting": _s(parsed.get("setting")),
            "lighting": _s(parsed.get("lighting")),
            "camera": _s(parsed.get("camera")),
            "colors": _str_list(parsed.get("colors"), 6),
            "preserve": _str_list(parsed.get("preserve"), 14),
            "notes": _s(parsed.get("notes"), 300),
            "cost_usd": cost,
        }
        return out

    # ------------------------------------------------------------ critique

    def critique_result(self, image_path: str, brief: dict,
                        reference_path: str | None = None) -> dict:
        encoded = _encode_image(str(image_path))
        if encoded is None:
            return _blank_critique("No se pudo leer la imagen generada.")

        blocks: list[dict] = []
        reference = _encode_image(str(reference_path)) if reference_path else None
        if reference is not None:
            blocks.append({"type": "text",
                           "text": "Image 1 is the ORIGINAL photograph of this "
                                   "person, for identity comparison only."})
            blocks.append(_image_block(reference))
            blocks.append({"type": "text", "text": "Image 2 is the GENERATED "
                                                   "image you must inspect."})
        blocks.append(_image_block(encoded))

        text = "Inspect the generated image and answer with the JSON object."
        if isinstance(brief, dict) and brief:
            slim = {k: brief[k] for k in
                    ("intent", "prompt", "options", "preserve", "shot_type",
                     "person_name") if k in brief}
            hint = _context_text(slim)
            if hint:
                text += "\nWhat was asked for (JSON): " + hint
        blocks.append({"type": "text", "text": text})

        parsed, cost, note = self._ask(_CRITIQUE_SYSTEM, blocks, 1600)
        if parsed is None:
            out = _blank_critique(note)
            out["cost_usd"] = cost
            return out

        width, height = encoded[2], encoded[3]
        defects = []
        for item in (parsed.get("defects") or []):
            defect = self._defect(item, width, height)
            if defect is not None:
                defects.append(defect)
        worst = max((d["severity"] for d in defects), default=0.0)
        if parsed.get("score") is None:      # derive rather than fail a clean image
            score = _clamp01(1.0 - sum(d["severity"] for d in defects) * 0.4)
        else:
            score = _clamp01(_f(parsed.get("score"), 0.0))
        return {
            "ok": bool(score >= 0.62 and worst < 0.55),
            "score": round(score, 4),
            "defects": defects,
            "identity_notes": _s(parsed.get("identity_notes"), 500),
            "cost_usd": cost,
        }

    def _defect(self, item: Any, width: int, height: int) -> dict | None:
        if not isinstance(item, dict):
            return None
        kind = _s(item.get("type"), 40).lower().replace(" ", "_")
        if kind not in DEFECT_TYPES:
            return None
        severity = _clamp01(_f(item.get("severity"), 0.5))
        repairable = bool(item.get("repairable"))
        if kind in _NON_REPAIRABLE:
            repairable = False
        bbox: list[int] = []
        raw = item.get("bbox")
        if isinstance(raw, (list, tuple)) and len(raw) == 4:
            x, y, w, h = (_f(v) for v in raw)
            if max(x, y, w, h) > 1.5:
                # Answered in pixels of the downscaled image it was shown, so
                # undo that downscale to land on the full resolution frame.
                seen = min(1.0, MAX_SIDE / float(max(width, height, 1)))
                scale_x = scale_y = 1.0 / seen if seen > 0 else 1.0
            else:
                scale_x, scale_y = float(width), float(height)
            px = int(round(max(0.0, x) * scale_x))
            py = int(round(max(0.0, y) * scale_y))
            pw = int(round(w * scale_x))
            ph = int(round(h * scale_y))
            pw = min(pw, width - px)
            ph = min(ph, height - py)
            if pw > 1 and ph > 1:
                bbox = [px, py, pw, ph]
        if not bbox:
            repairable = False                  # nothing to repaint
        return {"type": kind, "where": _s(item.get("where"), 40) or "global",
                "bbox": bbox, "severity": round(severity, 3),
                "repairable": repairable, "detail": _s(item.get("detail"), 300)}

    # -------------------------------------------------------- prompt writer

    def write_prompt(self, brief: dict) -> dict:
        """Author a prompt from a brief.  {} means 'no opinion, use the
        deterministic builder in generation/prompt.py'."""
        if not isinstance(brief, dict) or not brief:
            return {}
        blocks = [{"type": "text",
                   "text": "Brief (JSON):\n" + _context_text(brief, 4000)
                           + "\n\nAnswer with the JSON object."}]
        parsed, cost, note = self._ask(_PROMPT_SYSTEM, blocks, 1200)
        if parsed is None:
            log.info("Claude no escribio el prompt (%s); se usa el builder local.",
                     note)
            return {}
        prompt = _s(parsed.get("prompt"), 2000)
        if not prompt:
            return {}
        return {"prompt": prompt,
                "negative_prompt": _s(parsed.get("negative_prompt"), 1200),
                "identity_clause": _s(parsed.get("identity_clause"), 600),
                "tokens": _str_list(parsed.get("tokens"), 24),
                "notes": _s(parsed.get("notes"), 300),
                "cost_usd": cost}
