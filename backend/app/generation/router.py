"""Provider routing and cost estimation.

The orchestrator must never name a vendor, and the client must never be
surprised by a bill.  Those two requirements meet here: this module asks the
registry which providers exist, keeps only the ones that can actually perform
the operation at hand and have a key, picks the cheapest that fits the money
available, and - when nothing fits, or nothing is configured - falls back to the
free local engine instead of failing the run.  Every decision comes back with a
sentence in Spanish, because "porque si" is not an acceptable answer when the
user is being charged.
"""
from __future__ import annotations

from typing import Any

from .. import db
from ..catalog import options as options_mod
from ..config import SETTINGS
from ..providers import registry as registry_mod
from ..providers.base import GenRequest, ImageProvider, ProviderError

# Planning factor: about one image in three needs a repair pass or a retry.
# Documented here so the number on the estimate screen can be explained.
REPAIR_FACTOR = 1.35

# What that repair pass really costs.  The factor used to multiply the
# GENERATION price, as if a repair were a fraction of an image; it is not.  A
# rejected attempt is repainted region by region on the fill endpoint, and each
# region is a separate paid call: 25 runs of her IMG_7580 through the real gate
# on 2026-09-03 opened a repair round on 10 of them and EVERY round billed two
# inpaints, 0.100 USD - sixteen times the 0.00625 USD draft it was repairing.
# Quoted at 1.35 x generation, that draft run was announced at 0.0084 USD and
# settled at 0.1062 USD, and with the shipped retry limit at 0.2687 USD: a
# thirty-twofold surprise on the one number the client said she cared about.
# So the repair is priced as what it is - an inpaint per region - and the two
# regions below are the measured typical round, not a guess.
REPAIR_REGIONS_TYPICAL = 2

# How many photographs of her travel with one generation, the one being edited
# included.  It lives here, with the pricing, because the number changes which
# model runs and therefore what an image costs: fal picks identity_multi
# (0.040 USD) when references are attached and identity_max (0.080 USD) when
# they are not, so the balance page, the estimate and the run have to agree
# about it or they will quote three different prices again.
REFERENCE_COUNT = 3

# How many attempts a variant TYPICALLY makes.  The orchestrator stops as soon
# as the same check fails twice with the same reading (STOP_NOISE there), and
# that is what a run measured against the engine's actual behaviour will do:
# three paid attempts on one variant read identity 0.3507 / 0.3753 / 0.3643 -
# the same answer three times, 0.12 USD for it.
#
# It is NOT the bound, and quoting it as one made the ceiling a lie.  The stop
# rule only recognises a REPEATED reading; a check whose value moves between
# attempts - anatomy severity 0.62, then 0.95, then 0.72, which is ordinary
# scatter on that ruler - buys the third attempt ``max_retries_per_variant``
# allows.  Measured 2026-09-04 in the adversarial rehearsal: three variants
# failing that way settled 1.2600 USD (9 generations at 0.040 + 18 repaints at
# 0.050) against a ceiling announced as 1.1400 USD, while the run's own
# reservations added up to 1.7100 USD - the billing gate was already holding
# for three attempts while the screen promised two.  So the ceiling below is
# computed from the hard limit and this number is only quoted as the usual
# stopping point.
BOUNDED_ATTEMPTS = 2

# What the paid history says about the options themselves.  This used to be a
# hand-written table and it went stale the moment the next run finished: it
# still announced playa_atardecer as "3 de 11" after that scene had been
# measured at 1 of 5.  So it is now a SEED plus a live count, and the rule that
# turns counts into a warning is written once, below, instead of being applied
# by hand into two frozen dictionaries.
#
# THE SEED.  Re-measured 2026-09-04 by scoring every paid image FILE still on
# disk (17 of them) with the live SFace recogniser against her current profile
# signature, at the 0.45 bar.  The previous seed counted 21 attempts, four of
# which no longer have a file and could only be read from records written by
# the old blind descriptor that scored 0.99 on a different woman; those are
# dropped, which is the whole difference between "3 de 11" and "1 de 5".
#
#   ropa    camisa_blanca      2 de 2 (0.7256, 0.7396 - las dos mejores del corpus)
#   ropa    jersey_cachemira   2 de 2      ropa   blazer_oversize   1 de 1
#   ropa    gabardina          1 de 1      ropa   vaqueros_camiseta 7 de 8
#   ropa    vestido_verano     0 de 3
#   postura caminando          7 de 8      postura sentada          1 de 1
#   postura brazos_cruzados    0 de 3
#   escena  ciudad_noche       6 de 6      escena ciclorama_blanco  1 de 1
#   escena  playa_atardecer    1 de 5 (0.3221, 0.3365, 0.3580, 0.3853, 0.6323)
#   color   blanco             7 de 11     luz ventana_izq 4 de 7, ventana_der 4 de 5
#
# n is small and the cells are partly confounded (scene with clothing), but the
# beach signal survives the obvious control: the same garment and pose scored
# 6 of 6 in ciudad_noche and 1 of 2 on the beach.  It is a warning, not a law,
# and it is worded as one.  The colour and lighting cells are counted and named
# here on purpose - they are the ones the rule below correctly declines to warn
# about, and leaving them out of the record would make the table look cleaner
# than the measurement was.
SEED_HISTORY: dict[tuple[str, str], tuple[int, int]] = {
    ("clothing", "camisa_blanca"): (2, 2),
    ("clothing", "jersey_cachemira"): (2, 2),
    ("clothing", "blazer_oversize"): (1, 1),
    ("clothing", "gabardina"): (1, 1),
    ("clothing", "vaqueros_camiseta"): (7, 8),
    ("clothing", "vestido_verano"): (0, 3),
    ("clothing_color", "blanco"): (7, 11),
    ("clothing_color", "gris"): (1, 1),
    ("lighting", "ventana_izq"): (4, 7),
    ("lighting", "ventana_der"): (4, 5),
    ("pose", "caminando"): (7, 8),
    ("pose", "sentada"): (1, 1),
    ("pose", "brazos_cruzados"): (0, 3),
    ("scene", "ciudad_noche"): (6, 6),
    ("scene", "ciclorama_blanco"): (1, 1),
    ("scene", "playa_atardecer"): (1, 5),
}
# Every paid attempt in this installation was made on or before this instant,
# so anything the live query finds after it is new evidence and adds to the
# seed instead of double-counting it.
SEED_UNTIL = 1788514580.1

# When a tally is worth putting on the screen.  Three paid images is the least
# that can distinguish a bad option from bad luck, and half of them coming back
# as somebody else is not luck; four at 85% is the mirror image of that.  The
# measured cells the rule declines to warn about are exactly the confounded
# ones - colour blanco at 7 of 11 and ventana_izq at 4 of 7 - which is the
# behaviour wanted: those numbers are carried by whatever scene they were shot
# in, not by the colour.
RISK_MIN_N = 3
RISK_MAX_RATE = 0.50
SAFE_MIN_N = 4
SAFE_MIN_RATE = 0.85

QUALITIES = ("draft", "preview", "standard", "high", "max")
QUALITY_MIN_SIDE = {"draft": 0, "preview": 0, "standard": 1024,
                    "high": 1536, "max": 2048}
# Longest side of the box each tier asks for.  It lives here, next to the
# routing, because the same number has to reach the two places that must
# agree: the request that is priced and the request that is sent.
QUALITY_LONGEST_SIDE = {"draft": 512, "preview": 768, "standard": 1024,
                        "high": 1536, "max": 2048}
TOP_QUALITY = ("high", "max")

_OPERATION_ALIASES = {
    "generate": "generate", "generar": "generate", "img2img": "generate",
    "image": "generate", "preview": "generate", "final": "generate",
    "render": "generate", "text2img": "generate",
    "inpaint": "inpaint", "repair": "inpaint", "reparar": "inpaint",
    "fix": "inpaint", "mask": "inpaint",
    "upscale": "upscale", "escalar": "upscale", "enlarge": "upscale",
}

_NAME_ALIASES = {
    "fal.ai": "fal", "fal-ai": "fal", "falai": "fal",
    "openai": "openai", "dalle": "openai", "gpt-image": "openai",
    "stability": "stability", "stabilityai": "stability",
    "replicate": "replicate", "local": "local", "libre": "local",
    "gratis": "local", "free": "local", "offline": "local",
}

_LOCAL_HINTS = ("local", "free", "gratis", "offline", "builtin", "cpu")

# Option groups that can only be honoured by inventing pixels of the person:
# another garment, another body position, other face muscles, other hair,
# other fabric.  A provider whose Capabilities say ``generative=False``
# transforms the photograph it is handed - background, garment colour, light,
# grade, crop - so it cannot do any of these however cheap it is, and being
# cheap is not a reason to give it a job it is unable to perform.
GENERATIVE_CHANGES = ("clothing", "pose", "expression", "hair",
                      "transparency")


# ------------------------------------------------------------------ helpers

def _text(value: Any) -> str:
    return str(value).strip() if value is not None else ""


def _f(value: Any, default: float = 0.0) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return default
    if out != out:
        return default
    return out


def _operation(req_kind: Any) -> str:
    return _OPERATION_ALIASES.get(_text(req_kind).lower(), "generate")


# The same word, for the one message a client reads.  "inpaint" is the name on
# fal's price list and belongs on the estimate card next to the money; it does
# not belong in a sentence that tells her why nothing can be made right now.
_OPERATION_ES: dict[str, str] = {
    "generate": "crear la imagen",
    "inpaint": "repintar solo la zona que cambias",
    "upscale": "ampliar la imagen",
    "edit": "editar la imagen",
}


def _operation_es(operation: Any) -> str:
    return _OPERATION_ES.get(_text(operation).lower(), "hacer este cambio")


def _quality(quality: Any) -> str:
    word = _text(quality).lower()
    return word if word in QUALITIES else "preview"


def _alias(name: Any) -> str:
    word = _text(name).lower()
    return _NAME_ALIASES.get(word, word)


def _iter_providers(source: Any) -> list:
    items: list = []
    if isinstance(source, dict):
        items = list(source.values())
    elif isinstance(source, (list, tuple, set)):
        items = list(source)
    elif source is not None:
        items = [source]
    out = []
    for item in items:
        if isinstance(item, ImageProvider):
            out.append(item)
            continue
        if isinstance(item, type):
            try:
                built = item()
            except Exception:
                continue
            if isinstance(built, ImageProvider):
                out.append(built)
            continue
        if hasattr(item, "capabilities") and callable(item.capabilities):
            out.append(item)
    return out


def _all_image_providers() -> list:
    """Whatever accessor the registry exposes, this returns image providers."""
    found: list = []
    for name in ("image_providers", "all_image_providers", "list_image_providers",
                 "providers", "all_providers", "list_providers"):
        fn = getattr(registry_mod, name, None)
        if not callable(fn):
            continue
        try:
            found = _iter_providers(fn())
        except Exception:
            found = []
        if found:
            break
    if not found:
        for attr in ("IMAGE_PROVIDERS", "PROVIDERS", "REGISTRY"):
            found = _iter_providers(getattr(registry_mod, attr, None))
            if found:
                break

    out: list = []
    seen: set[str] = set()
    for provider in found:
        caps = _caps(provider)
        if caps is None or _text(caps.kind).lower() not in ("", "image"):
            continue
        key = _alias(caps.name or getattr(provider, "name", ""))
        if key in seen:
            continue
        seen.add(key)
        out.append(provider)
    return out


def _caps(provider: Any):
    try:
        caps = provider.capabilities()
    except Exception:
        return None
    return caps if hasattr(caps, "name") else None


def _name_of(provider: Any) -> str:
    caps = _caps(provider)
    return _text(getattr(caps, "name", "")) or _text(getattr(provider, "name", "")) \
        or provider.__class__.__name__


def _available(provider: Any) -> bool:
    try:
        return bool(provider.available())
    except Exception:
        return False


def _supports(provider: Any, operation: str) -> bool:
    caps = _caps(provider)
    if caps is None:
        return False
    if operation == "inpaint":
        return bool(caps.inpaint)
    if operation == "upscale":
        return bool(caps.upscale)
    return bool(caps.img2img or caps.text2img)


def _dim(value: float) -> int:
    """A side in pixels: never below 64, always a multiple of eight."""
    return max(64, int(round(_f(value, 64.0) / 8.0)) * 8)


def _oriented_size(path: str) -> tuple[int, int]:
    """The photograph's size AS IT IS SEEN, orientation tag applied.

    Her camera writes 3088x2316 with EXIF orientation 6, which is a portrait
    2316x3088 on screen and in every mask this app computes.  Reading the raw
    header would order a landscape 4:3 of a portrait photograph - the square
    bug again, only sideways - so the tag is honoured here.
    """
    try:
        from PIL import Image
        with Image.open(path) as img:
            width, height = img.size
            tag = img.getexif().get(274)
    except Exception:
        return 0, 0
    if tag in (5, 6, 7, 8):
        width, height = height, width
    return int(width or 0), int(height or 0)


def framing_key(framing: Any) -> str:
    """The framing as the engines name it ('portrait_full'), or "".

    Accepts the catalogue value key the user actually clicked
    ('cuerpo_entero'), the engine key the local hints already carry, or the
    whole ``choices`` dict, because all three are what the callers hold.
    """
    from ..providers.local_free import FRAMING_ASPECT

    raw = framing.get("framing") if isinstance(framing, dict) else framing
    if isinstance(raw, (list, tuple, set)):
        raw = next(iter(raw), "")
    key = _text(raw)
    if not key or key in FRAMING_ASPECT:
        return key
    value = options_mod.value_of("framing", key) or {}
    return _text((value.get("local") or {}).get("framing"))


def request_geometry(quality: str, source_path: Any = None,
                     source_size: Any = None,
                     framing: Any = None) -> tuple[int, int]:
    """The output box for one image: HER shape, at the size of the tier.

    This is the fix to the defect that mattered most.  The orchestrator used to
    send ``width=height=QUALITY_LONGEST_SIDE[quality]`` and fal turns that pair
    into an aspect_ratio string, so every paid image was ordered as a square:
    all 21 fal renders on this installation are 1024x1024 while her
    photographs are 2316x3088.  Measured on her seven measurable photographs,
    delivering that square instead of her own 3:4 moves the body ruler of
    identity/verify from a mean 0.017 to 0.279 against tolerances of 0.04 to
    0.08, and the gate rejects six of the seven; her 3:4 passes all seven.

    The chosen framing wins when there is one - she asked for a 9:16 story, she
    gets 9:16 - and otherwise the source photograph decides.  (0, 0) means
    "follow the source", which is what GenRequest already documents, and is
    only reached when nothing at all is known about the shape.
    """
    from ..providers.local_free import FRAMING_ASPECT

    side = QUALITY_LONGEST_SIDE.get(_quality(quality), 768)
    ratio = FRAMING_ASPECT.get(framing_key(framing)) if framing else None
    if not ratio:
        width = height = 0
        if isinstance(source_size, (list, tuple)) and len(source_size) == 2:
            width, height = int(_f(source_size[0])), int(_f(source_size[1]))
        if not (width > 0 and height > 0) and _text(source_path):
            width, height = _oriented_size(_text(source_path))
        ratio = (width / float(height)) if width > 0 and height > 0 else 0.0
    if ratio <= 0:
        return 0, 0
    if ratio >= 1.0:
        return _dim(side), _dim(side / ratio)
    return _dim(side * ratio), _dim(side)


def build_request(operation: str, quality: str, *, prompt: str = "",
                  source_path: Any = None, reference_paths: Any = None,
                  mask_path: Any = None, framing: Any = None,
                  source_size: Any = None, **fields) -> GenRequest:
    """The one builder for a request that is BOTH priced and sent.

    Three numbers used to exist for the same image at quality 'high': the
    estimate priced a probe with no references and got fal's 'identity_max' at
    0.080, the orchestrator sent the source photograph as its own identity
    reference and was billed 'identity_multi' at 0.040, and the balance gate
    reserved a hard-coded 0.055.  A provider picks its model FROM THE REQUEST -
    source present, references present, mask present, size - so the only way
    the quote, the reservation and the invoice can agree is for all three to
    talk about the same object.  That is what this function returns.
    """
    width, height = request_geometry(quality, source_path=source_path,
                                     source_size=source_size, framing=framing)
    refs = [_text(r) for r in (reference_paths or []) if _text(r)]
    return GenRequest(prompt=prompt, operation=operation,
                      quality=_quality(quality),
                      source_path=_text(source_path) or None,
                      mask_path=_text(mask_path) or None,
                      reference_paths=refs, width=width, height=height,
                      **fields)


def _probe_request(operation: str, quality: str, **kw) -> GenRequest:
    """A request shaped like the one that will really be sent.

    This matters for the price.  Every image this product makes starts from a
    photograph of the person, and providers choose a cheaper text-only model
    when no source is attached - so pricing an empty request quoted the client
    0.003 USD for work that actually bills at 0.040, a thirteenfold
    understatement on the one number she said she cared about most.  The probe
    therefore carries a source path, exactly like the real call; when the
    caller knows the real photograph it hands that over, with its shape and
    its framing, because the cheap tiers are billed per megapixel.
    """
    if not kw.get("source_path"):
        kw["source_path"] = "probe.jpg"
    return build_request(operation, quality, **kw)


def _cost(provider: Any, operation: str, quality: str,
          request: GenRequest | None = None) -> float:
    """Price of one image.  ``request`` is the real one whenever there is one:
    pricing a probe that differs from what is sent is how the three numbers
    diverged in the first place."""
    req = request if isinstance(request, GenRequest) \
        else _probe_request(operation, quality)
    try:
        return max(0.0, _f(provider.estimate_cost(req), 0.0))
    except Exception:
        caps = _caps(provider)
        return max(0.0, _f(getattr(caps, "cost_per_image_usd", 0.0), 0.0))


def delivered_side(provider: Any, quality: str,
                   request: GenRequest | None = None) -> int:
    """Longest side this provider really hands back FOR THIS REQUEST.

    The provider is asked about the call that will be made whenever there is
    one, because the answer is per endpoint and not per vendor: fal's
    Capabilities can only carry one number and it carries the default role's,
    so a masked run - a different endpoint, a different file - was being
    described by the Kontext row.
    """
    asked = QUALITY_LONGEST_SIDE.get(_quality(quality), 768)
    ceiling = 0
    fn = getattr(provider, "delivered_side", None)
    if callable(fn) and isinstance(request, GenRequest):
        try:
            ceiling = int(_f(fn(request), 0.0))
        except Exception:                                # noqa: BLE001
            ceiling = 0
    if not ceiling:
        caps = _caps(provider)
        if caps is None:
            return 0
        ceiling = int(_f(getattr(caps, "out_max_side", 0), 0.0)) or \
            int(_f(getattr(caps, "max_side", 0), 0.0))
    return min(asked, ceiling) if ceiling else asked


def photos_sent(provider: Any, request: GenRequest | None = None,
                references: Any = None) -> int:
    """How many photographs of her this call really carries.

    Asked of the provider, because only it knows which of its endpoints has a
    pool: kontext/multi takes the photograph plus up to three others, kontext
    and img2img take one, and the fill endpoint takes none besides the one it
    is repainting.  The fallback is what the caller planned, for a provider
    that has never been asked this question.
    """
    fn = getattr(provider, "images_sent", None)
    if callable(fn) and isinstance(request, GenRequest):
        try:
            return max(0, int(_f(fn(request), 0.0)))
        except Exception:                                # noqa: BLE001
            pass
    return min(REFERENCE_COUNT, 1 + len(list(references or [])))


def resolution_note(provider: Any, quality: str,
                    request: GenRequest | None = None,
                    source_size: Any = None) -> str:
    """One Spanish sentence about the pixels of the endpoint THAT WILL RUN.

    fal's Kontext endpoints expose no size knob at all - see MODELS in
    providers/fal - and answer with about one megapixel whatever was paid.
    Silence here is what let 'alta' and 'maxima' sell 1536 and 2048 px files
    that arrive at 1024 px.  The tier still buys a more faithful model; it just
    stops promising pixels, and putting the local upscaler in between would not
    buy them either - it moved the measured texture loss from +0.018 to +0.050
    over her seven photographs.

    ON THE MASKED PATH THAT SENTENCE IS SIMPLY FALSE, which is why the request
    is an argument now.  fal's fill endpoint answers with about one megapixel
    as well, but that answer is not the file she receives: protect.compose puts
    it back INSIDE the mask over her own photograph at its full size, so the
    image the album stores is her camera's 2316x3088 - her own pixels
    everywhere the mask is black, her face included.  Saying "entrega como
    maximo 1024 px" about that file would understate it threefold in each
    direction, so what is said instead is where the softness really is: inside
    the zone that was repainted and enlarged.
    """
    asked = QUALITY_LONGEST_SIDE.get(_quality(quality), 768)
    gets = delivered_side(provider, quality, request)
    masked = bool(isinstance(request, GenRequest) and request.mask_path)
    if masked:
        # NOT clamped by the tier.  The fill endpoint has no size knob at all,
        # so it answers with about one megapixel whatever was paid, and
        # min(asked, ceiling) was announcing 512 px for a draft that comes back
        # at 1024 px exactly like the top tier does.
        raw = 0
        fn = getattr(provider, "delivered_side", None)
        if callable(fn):
            try:
                raw = int(_f(fn(request), 0.0))
            except Exception:                            # noqa: BLE001
                raw = 0
        gets = raw or gets
        size = ""
        if isinstance(source_size, (list, tuple)) and len(source_size) == 2 \
                and int(_f(source_size[0])) > 0 and int(_f(source_size[1])) > 0:
            size = " (%dx%d px)" % (int(_f(source_size[0])),
                                    int(_f(source_size[1])))
        return ("Aviso: como solo se repinta la zona que cambias, el archivo "
                "que recibes conserva el tamano de tu foto%s. %s devuelve esa "
                "zona a %d px de lado largo y se amplia para pegarla, asi que "
                "solo la parte repintada pierde algo de detalle; tu rostro "
                "conserva todos los pixeles de tu camara."
                % (size, _name_of(provider), gets or asked))
    if not gets or gets >= asked:
        return ""
    return ("Aviso: %s entrega como maximo %d px de lado largo en esta "
            "calidad, no %d. La calidad alta paga un modelo mas fiel a tus "
            "rasgos, no un archivo mas grande."
            % (_name_of(provider), gets, asked))


def _model_name(provider: Any, quality: str,
                request: GenRequest | None = None) -> str:
    # pick_model expects a request, not a quality string.  Handing it the bare
    # string used to "work" only because attribute lookups on a str all miss and
    # it fell through to the no-source default, which is how the draft model
    # ended up named in every estimate.  The real request is handed over when
    # there is one: the model named on the screen must be the model billed, and
    # a reference photograph is enough to change which one runs.
    probe = request if isinstance(request, GenRequest) \
        else _probe_request("generate", quality)
    for attr in ("model_for", "pick_model", "model_name"):
        fn = getattr(provider, attr, None)
        if not callable(fn):
            continue
        for args in ((probe,), (quality,), ()):
            try:
                value = fn(*args)
            except Exception:
                continue
            if _text(value):
                return _text(value)
    for attr in ("MODELS", "models"):
        table = getattr(provider, attr, None)
        if isinstance(table, dict):
            value = table.get(quality) or table.get("default")
            if _text(value):
                return _text(value)
    for attr in ("model", "default_model"):
        value = getattr(provider, attr, None)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return _name_of(provider)


def endpoint_name(provider: Any, request: GenRequest | None = None,
                  quality: str = "preview") -> str:
    """The vendor URL this request will really be posted to, when it is known.

    The role key ('inpaint', 'identity_multi') is what the codebase routes on;
    it is not what anybody can look up on a price list.  The estimate carries
    both so that "fill a 0.050 USD" and "kontext/multi a 0.040 USD con tres
    fotos" can be read off the screen and checked against fal's own pricing
    page, which is the only way the client can audit the number she is shown.
    """
    role = _model_name(provider, quality, request)
    fn = getattr(provider, "endpoint", None)
    if callable(fn):
        try:
            value = _text(fn(role))
            if value:
                return value
        except Exception:                                # noqa: BLE001
            pass
    return role


def _quality_score(provider: Any) -> float:
    """How much identity work this provider can be trusted with."""
    caps = _caps(provider)
    if caps is None:
        return 0.0
    score = _f(getattr(caps, "max_side", 0), 0.0) / 2048.0
    if getattr(caps, "identity_reference", False):
        score += 2.0
    if getattr(caps, "img2img", False):
        score += 1.0
    if getattr(caps, "inpaint", False):
        score += 0.5
    if getattr(caps, "upscale", False):
        score += 0.25
    return score


def _meets_bar(provider: Any, quality: str) -> bool:
    caps = _caps(provider)
    need = QUALITY_MIN_SIDE.get(quality, 0)
    return caps is not None and _f(getattr(caps, "max_side", 0), 0.0) >= need


def _is_local(provider: Any) -> bool:
    caps = _caps(provider)
    if caps is None:
        return False
    if not getattr(caps, "needs_key", True) and _f(
            getattr(caps, "cost_per_image_usd", 0.0), 0.0) <= 0.0:
        return True
    return any(hint in _name_of(provider).lower() for hint in _LOCAL_HINTS)


def _find_named(name: str, providers: list):
    wanted = _alias(name)
    if not wanted:
        return None
    for accessor in ("get_image_provider", "get_provider", "image_provider", "get"):
        fn = getattr(registry_mod, accessor, None)
        if not callable(fn):
            continue
        try:
            found = fn(name)
        except Exception:
            found = None
        got = _iter_providers(found)
        if got:
            return got[0]
    for provider in providers:
        if _alias(_name_of(provider)) == wanted:
            return provider
        caps = _caps(provider)
        if caps is not None and _alias(getattr(caps, "key_name", "")) == wanted:
            return provider
    return None


def _local_provider(providers: list):
    for accessor in ("local_provider", "local", "free_provider", "fallback"):
        fn = getattr(registry_mod, accessor, None)
        if callable(fn):
            try:
                got = _iter_providers(fn())
            except Exception:
                got = []
            if got:
                return got[0]
    for provider in providers:
        if _is_local(provider):
            return provider
    return None


def _money(value: float) -> str:
    return ("%.4f" % value).rstrip("0").rstrip(".") if value else "0"


def _join_es(items: list[str], word: str = "y") -> str:
    """Spanish enumeration: 'ropa', 'ropa y postura', 'ropa, postura y pelo'."""
    if not items:
        return ""
    if len(items) == 1:
        return items[0]
    return "%s %s %s" % (", ".join(items[:-1]), word, items[-1])


def _changes_set(changes: Any) -> set[str]:
    """Normalise what the caller asked to change into a set of group keys.

    Accepts the variant's ``choices`` dict as well as a plain iterable, because
    the orchestrator already holds that dict and copying it into a second shape
    would only invite the two to drift.  A group mapped to nothing was not
    asked for, so it is not a requested change.
    """
    if changes is None:
        return set()
    if isinstance(changes, dict):
        items = [k for k, v in changes.items() if v not in (None, "", [], {})]
    elif isinstance(changes, (list, tuple, set, frozenset)):
        items = list(changes)
    else:
        items = [changes]
    return {_text(k).lower() for k in items if _text(k)}


def _change_labels(groups: list[str]) -> list[str]:
    """Spanish names of the option groups, taken from the catalogue she reads."""
    out: list[str] = []
    for key in groups:
        group = options_mod.GROUPS_BY_KEY.get(key) or {}
        label = (_text(group.get("label_es")) or _text(key)).lower()
        if label and label not in out:
            out.append(label)
    return out


def _cannot_do(provider: Any, changes: set[str]) -> list[str]:
    """Requested groups this provider is honestly unable to perform.

    The answer comes from the provider's own Capabilities and never from its
    name, so a future non generative engine is filtered for free and any
    provider written before the flag existed keeps the permissive default.
    """
    if not changes:
        return []
    caps = _caps(provider)
    if caps is None or bool(getattr(caps, "generative", True)):
        return []
    return [group for group in GENERATIVE_CHANGES if group in changes]


def _unapplied_note(provider: Any, changes: set[str]) -> str:
    """One Spanish sentence naming the changes this engine will not make."""
    missing = _cannot_do(provider, changes)
    if not missing:
        return ""
    return ("Aviso: %s no genera imagen nueva, solo transforma la foto real, "
            "asi que estos cambios NO se aplicaran: %s. Configura la clave "
            "de un motor que sepa dibujar para poder pedirlos."
            % (_name_of(provider), _join_es(_change_labels(missing))))


def _plan_changes(plan: dict) -> set[str]:
    """Which option groups a planned run actually changes."""
    out: set[str] = set()
    for variant in (plan.get("variants") or []):
        if isinstance(variant, dict):
            out |= _changes_set(variant.get("choices"))
    out |= _changes_set(plan.get("locked"))
    out |= _changes_set(plan.get("varied"))
    return out


# ------------------------------------------------------------------ routing

def unsupported_changes(provider: Any, changes: Any = None) -> list[str]:
    """Which of the requested option groups this provider cannot perform.

    Public because the orchestrator has to record the same answer the router
    acted on, and reading it back out of a Spanish sentence would be a way to
    get it wrong the first time somebody rewords the sentence.
    """
    return _cannot_do(provider, _changes_set(changes))


def choose_provider(req_kind: str, quality: str, budget_usd: float,
                    prefer: str | None = None,
                    changes: Any = None,
                    request: GenRequest | None = None
                    ) -> tuple[ImageProvider, str, str]:
    """Pick who makes this image, and say why in one Spanish sentence.

    ``changes`` is what the user asked to change: the option group keys of the
    variant, or the variant's ``choices`` dict.  It matters because an engine
    that cannot perform a requested change is not a cheaper way of doing the
    job, it is a different job - routing on price alone is exactly how a run
    produced six previews wearing the original clothes in the original pose.

    ``request`` is the GenRequest the caller is about to send, from
    ``build_request``.  Every price in the sentence below is measured on it, so
    the figure she reads is the figure her ledger will carry; without it the
    quote was taken on a probe that differed from the call in exactly the
    fields a provider uses to choose its model.
    """
    operation = _operation(req_kind)
    qual = _quality(quality)
    # None means "the caller set no ceiling"; 0 means "there is no money".
    budget = float("inf") if budget_usd is None else _f(budget_usd, 0.0)
    wanted = _changes_set(changes)
    providers = _all_image_providers()
    local = _local_provider(providers)
    skipped: list[str] = []

    if not providers:
        raise ProviderError("No hay ningun motor de imagen instalado en esta "
                            "aplicacion.", retryable=False, code="no_provider")

    preferred = _text(prefer) or _text(SETTINGS.default_image_provider)
    explicit = bool(_text(prefer))
    if _alias(preferred) not in ("", "auto"):
        target = _find_named(preferred, providers)
        label = _text(preferred)
        if target is None:
            skipped.append("%s no esta registrado" % label)
        elif not _supports(target, operation):
            skipped.append("%s no puede %s" % (_name_of(target), operation))
        elif not _available(target):
            skipped.append("%s no tiene clave configurada" % _name_of(target))
        else:
            cost = _cost(target, operation, qual, request)
            if cost > budget + 1e-9:
                skipped.append("%s cuesta %s USD por imagen y el presupuesto es "
                               "%s USD" % (_name_of(target), _money(cost),
                                           _money(budget)))
            else:
                why = ("lo pediste explicitamente" if explicit
                       else "es el motor que elegiste en Ajustes")
                # The name on fal's price list, not the internal role: "Se usa
                # fal (inpaint)" told the client a word from this codebase,
                # while the estimate card beside it said
                # fal-ai/flux-pro/v1/fill.  One name, and it is the one she can
                # look up.
                reason = ("Se usa %s (%s) porque %s. Coste por imagen: %s "
                          "USD." % (_name_of(target),
                                    endpoint_name(target, request, qual), why,
                                    _money(cost)))
                # A stated preference is an instruction and is obeyed, but it
                # never buys silence: if that engine cannot make one of her
                # changes she is told now, not after six images.
                warning = _unapplied_note(target, wanted)
                if warning:
                    reason += " " + warning
                return target, _model_name(target, qual, request), reason

    candidates = []
    for provider in providers:
        if not _supports(provider, operation):
            continue
        if not _available(provider):
            name = _name_of(provider)
            if not _is_local(provider):
                note = "%s no tiene clave configurada" % name
                if note not in skipped:
                    skipped.append(note)
            continue
        missing = _cannot_do(provider, wanted)
        if missing:
            note = "%s no puede cambiar %s" % (
                _name_of(provider), _join_es(_change_labels(missing), "ni"))
            if note not in skipped:
                skipped.append(note)
            continue
        cost = _cost(provider, operation, qual, request)
        if cost > budget + 1e-9:
            note = "%s cuesta %s USD y no cabe en el presupuesto" % (
                _name_of(provider), _money(cost))
            if note not in skipped:
                skipped.append(note)
            continue
        candidates.append((provider, cost))

    if candidates:
        if qual in TOP_QUALITY:
            # On a final render capability comes first: this is the image she keeps.
            candidates.sort(key=lambda item: (
                0 if _meets_bar(item[0], qual) else 1,
                -_quality_score(item[0]), item[1], _name_of(item[0])))
            criterion = "es el de mas calidad dentro del presupuesto"
        else:
            candidates.sort(key=lambda item: (
                0 if _meets_bar(item[0], qual) else 1,
                item[1], -_quality_score(item[0]), _name_of(item[0])))
            criterion = "es el mas barato que puede hacerlo"
        provider, cost = candidates[0]
        model = _model_name(provider, qual, request)
        reason = "Se usa %s (%s) porque %s. Coste por imagen: %s USD." % (
            _name_of(provider), endpoint_name(provider, request, qual),
            criterion, _money(cost))
        if skipped:
            reason += " Descartados: " + "; ".join(skipped[:3]) + "."
        return provider, model, reason

    if local is not None and _supports(local, operation) and _available(local):
        reason = ("; ".join(skipped[:3]) + ", se usa el motor local gratuito."
                  if skipped else
                  "No hay ningun motor de pago disponible, se usa el motor local "
                  "gratuito.")
        reason = reason[0].upper() + reason[1:]
        # The run never fails for want of an engine - but a free image that
        # ignores half of the request is only acceptable when she is told
        # which half was ignored.
        warning = _unapplied_note(local, wanted)
        if warning:
            reason += " " + warning
        return local, _model_name(local, qual, request), reason

    for provider in providers:
        if _supports(provider, operation) and _available(provider):
            reason = ("Sin alternativas dentro del presupuesto: se usa %s, el "
                      "unico motor disponible que puede hacerlo."
                      % _name_of(provider))
            warning = _unapplied_note(provider, wanted)
            if warning:
                reason += " " + warning
            return provider, _model_name(provider, qual, request), reason

    # ``operation`` is the internal word (generate, inpaint) and ``skipped``
    # names engines by their key: both were being shown to a client who does
    # not know what an inpaint is.  The engine names stay, because she chooses
    # one in Ajustes and pays for it; the operation is said in Spanish.
    raise ProviderError(
        "Ahora mismo no hay ningun motor que pueda %s (%s). Revisa en Ajustes "
        "que la clave este puesta, o elige 'que decida el robot'."
        % (_operation_es(operation),
           "; ".join(skipped[:3]) or "no hay ningun motor activo"),
        retryable=False, code="no_provider")


# ----------------------------------------------------------------- estimate

def plan_references(plan: dict) -> list[str]:
    """The photographs of her that will really travel with each generation.

    Filtered by ``exists`` exactly as ``run_previews`` filters them, because
    the count is what chooses the model: with references fal bills
    identity_multi at 0.040 USD, without them identity_max at 0.080 USD on the
    top tiers.  Counting a file that is no longer on disk would quote 0.040 and
    settle 0.080 - an UNDER-quote, the one direction that is not safe.
    """
    from pathlib import Path as _Path

    plan_d = plan if isinstance(plan, dict) else {}
    out: list[str] = []
    for ref in (plan_d.get("reference_paths") or []):
        name = _text(ref)
        if not name or name in out:
            continue
        try:
            if not _Path(name).exists():
                continue
        except OSError:
            continue
        out.append(name)
    return out


def plan_shields(plan: dict) -> list[dict]:
    """Will her face be generated?  One answer per variant, from ONE place.

    ``protect.shield_for`` is the same call ``_run_variant`` makes, on the same
    photograph, into the same folder - so it draws the mask once and the run
    finds it already there.  That is the whole point: the estimate used to ask
    ``plan_mask``, which reads the option groups, while the run asked whether a
    mask could actually be DRAWN on this photograph, and the two are different
    questions.  When they disagreed the screen priced fal's fill endpoint at
    0.050 USD for a call that went to kontext/multi at 0.040 USD with three
    reference photographs, and promised a face that would not be redrawn while
    the run redrew it.
    """
    from . import protect as protect_mod

    plan_d = plan if isinstance(plan, dict) else {}
    variants = plan_d.get("variants")
    rows = variants if isinstance(variants, list) and variants else [{}]
    source_path = plan_d.get("source_path")
    work_dir = plan_d.get("work_dir")
    # Her skin, her tolerances and the marks two or more of her photographs
    # agreed on, carried on the plan by ``prepare_run`` - the estimate has to
    # draw the SAME mask the run will send, and since 2026-09-05 that mask
    # keeps her visible marks black wherever the requested garment leaves them
    # showing.  Without this block the estimate would draw a mask with no
    # marks in it and the run would draw a different one.
    profile = plan_d.get("mark_profile")
    return [protect_mod.shield_for(source_path,
                                   (v.get("choices") if isinstance(v, dict)
                                    else {}) or {}, work_dir, profile=profile)
            for v in rows]


def plan_requests(plan: dict, quality: str,
                  shields: list[dict] | None = None) -> list[GenRequest]:
    """The requests this plan will really send, one per variant.

    Built here so the estimate screen and the run cannot drift: same builder,
    same source photograph, same framing, same size, same mask file.  A variant
    that changes the framing changes the shape of its image and, on the per
    megapixel tiers, its price - so each one is asked separately instead of
    multiplying the first one by six.

    ``shields`` is ``plan_shields`` above; it is an argument so that a caller
    which already has the answers does not ask for them twice.
    """
    plan_d = plan if isinstance(plan, dict) else {}
    variants = plan_d.get("variants")
    rows = variants if isinstance(variants, list) and variants else [{}]
    source_path = plan_d.get("source_path")
    source_size = plan_d.get("source_size")
    references = plan_references(plan_d)
    found = shields if isinstance(shields, list) and len(shields) == len(rows) \
        else plan_shields(plan_d)

    out: list[GenRequest] = []
    for variant, shield in zip(rows, found):
        choices = variant.get("choices") if isinstance(variant, dict) else {}
        masked = bool(shield.get("masked"))
        # The real mask file when there is one.  Nothing here is invented: the
        # request that is priced carries the very path the request that is sent
        # will carry, so provider.pick_model cannot answer differently.
        mask = _text(shield.get("mask_path")) or ("mask.png" if masked else "")
        out.append(_probe_request("inpaint" if masked else "generate", quality,
                                  source_path=source_path,
                                  source_size=source_size,
                                  mask_path=mask or None,
                                  reference_paths=[] if masked else references,
                                  framing=(choices or {})))
    return out


def option_history(user_id: str = "") -> dict[tuple[str, str], tuple[int, int]]:
    """Paid images per option value, and how many of them were really her.

    The seed above plus everything paid for since, so the warning learns.  Only
    attempts whose stored verdict was written by the CURRENT recogniser count:
    ``identity_face.threshold`` is 0.45 for the SFace check and 0.72 for the
    old geometric descriptor that read 0.99 on a stranger, and counting the
    second kind is how a table ends up certifying the images the client
    rejected on sight.  Which option each attempt was made under comes from the
    run's own plan, matched on the variant index, because the run's options
    hold every value that was asked for and not the one this image used.

    Reads only rows; a failure here costs a sentence on a screen, never a run.
    """
    counts = {k: list(v) for k, v in SEED_HISTORY.items()}
    if not user_id:
        return {k: (v[0], v[1]) for k, v in counts.items()}
    try:
        rows = db.q(
            "SELECT a.variant_index, a.verdict_json, r.plan_json "
            "FROM attempts a JOIN runs r ON r.id=a.run_id "
            "WHERE a.user_id=? AND a.cost_usd>0 AND a.created_at>?",
            (user_id, SEED_UNTIL))
    except Exception:                                     # noqa: BLE001
        rows = []
    for row in (rows or []):
        verdict = db.loads(row["verdict_json"], None) or {}
        check = next((c for c in (verdict.get("checks") or [])
                      if isinstance(c, dict) and c.get("name") == "identity_face"), None)
        if not check:
            continue
        threshold = _f(check.get("threshold"), None)
        value = _f(check.get("value"), None)
        if threshold is None or value is None or abs(threshold - 0.45) > 1e-6:
            continue
        plan = db.loads(row["plan_json"], None) or {}
        index = row["variant_index"]
        variant = next((v for v in (plan.get("variants") or [])
                        if isinstance(v, dict)
                        and int(v.get("index", -1)) == int(index if index is not None else -1)),
                       None)
        for group, picked in ((variant or {}).get("choices") or {}).items():
            if not _text(picked) or not isinstance(picked, str):
                continue
            cell = counts.setdefault((_text(group), _text(picked)), [0, 0])
            cell[1] += 1
            if value >= threshold:
                cell[0] += 1
    return {k: (v[0], v[1]) for k, v in counts.items()}


def _history_note(plan: dict, user_id: str = "") -> str:
    """What the images already paid for say about these very options.

    The client has spent 2.42 USD, and the most useful thing that money bought
    is a table of which requests came back as somebody else.  It is worth
    exactly one sentence on the screen where the request can still be changed
    for free.
    """
    history = option_history(user_id)
    chosen: set[tuple[str, str]] = set()
    for variant in (plan.get("variants") or []):
        for group, value in ((variant or {}).get("choices") or {}).items():
            values = value if isinstance(value, (list, tuple, set)) else [value]
            for item in values:
                if _text(item):
                    chosen.add((_text(group), _text(item)))
    for group, value in (plan.get("locked") or {}).items():
        values = value if isinstance(value, (list, tuple, set)) else [value]
        for item in values:
            if _text(item):
                chosen.add((_text(group), _text(item)))

    risky = [(key, history[key]) for key in sorted(chosen)
             if key in history and history[key][1] >= RISK_MIN_N
             and history[key][0] <= RISK_MAX_RATE * history[key][1]]
    if not risky:
        return ""

    def label(group: str, value: str) -> str:
        option = options_mod.value_of(group, value) or {}
        return _text(option.get("label_es")) or value

    parts = ["%s (%d de %d aprobadas)" % (label(*key), ok, total)
             for key, (ok, total) in risky]
    safe = [label(*key) for key in sorted(chosen)
            if key in history and history[key][1] >= SAFE_MIN_N
            and history[key][0] >= SAFE_MIN_RATE * history[key][1]]
    note = ("Aviso: de las imagenes que ya has pagado, estas opciones son las "
            "que mas veces han salido con otra cara: %s. Cambiarlas es la "
            "forma mas barata de no pagar por imagenes que no eres tu."
            % _join_es(parts))
    if safe:
        note += " En cambio %s si ha funcionado casi siempre." % _join_es(safe)
    return note


def _face_note(shields: list[dict], endpoint: str, per_image: float,
               n_photos: int) -> str:
    """Which of the two things is about to happen to her face, in one sentence.

    This is the difference between an image that will look like her and one
    that may not, so it is said before the money and not in the report
    afterwards.  The endpoint and the price are in the same sentence on
    purpose: they are the checkable half of it - a run that promises "no se
    vuelve a generar" while quoting kontext/multi is contradicting itself, and
    the client can now see that without reading any code.
    """
    rows = [sh for sh in (shields or []) if isinstance(sh, dict)]
    if not rows:
        return ""
    safe = [sh for sh in rows if sh.get("masked")]
    n_refs = max(0, int(n_photos))
    price = ("%s (%s USD por imagen)" % (endpoint, _money(per_image))
             if endpoint else "%s USD por imagen" % _money(per_image))
    if len(safe) == len(rows):
        return "%s Se paga %s, que repinta por zonas." % (
            safe[0].get("reason") or "", price)
    blocked = next(sh for sh in rows if not sh.get("masked"))
    if not safe:
        # The count is what leaves the machine, not what was chosen: at draft
        # the pose change goes to img2img, which takes ONE photograph, so
        # promising three would be promising a defence that is not there.
        carried = ("Viaja 1 foto tuya con cada imagen" if n_refs == 1 else
                   "Viajan %d fotos tuyas con cada imagen" % n_refs)
        return "%s %s y se paga %s." % (blocked.get("reason") or "",
                                        carried, price)
    return ("En %d de las %d imagenes tu rostro se copia de tu foto; en las "
            "otras %d hay que volver a dibujarlo. %s"
            % (len(safe), len(rows), len(rows) - len(safe),
               blocked.get("reason") or ""))


def _cost_ceiling_note(total: float, total_max: float, repair_unit: float,
                       attempts: int) -> str:
    """One Spanish sentence with the worst this run can cost.

    The estimate is a planning number - one image in three needs a second pass
    - and a planning number is honest only next to the ceiling it can reach.
    Without this the screen said 0.0084 USD for a draft run that settled at
    0.2687 USD (measured 2026-09-03, her IMG_7580 with the framing that the
    gate keeps rejecting): the money plumbing was right, only the sentence was
    missing.  Silent above a fifth over the estimate, because a ceiling that is
    printed on every run is a ceiling nobody reads.
    """
    if total <= 0.0 or total_max <= total * 1.2:
        return ""
    # The number of attempts said here is the hard one.  Saying "como mucho 2"
    # while the loop can buy 3 turned this sentence into the defect it exists
    # to prevent: 1.2600 USD settled under a 1.1400 USD promise, measured
    # 2026-09-04.  The stop rule is still worth naming, but as what it is -
    # what normally happens - and never as the limit.
    # The stop rule is worth naming only when there is something for it to
    # stop.  With retries disabled the loop makes one attempt, and "como mucho
    # 1 veces (normalmente 2)" is the sentence contradicting itself.
    usual = ("" if attempts <= BOUNDED_ATTEMPTS else
             " (normalmente %d, porque el robot deja de pagar intentos en "
             "cuanto la misma comprobacion falla dos veces con el mismo "
             "resultado)" % BOUNDED_ATTEMPTS)
    return ("Aviso: la estimacion cuenta con que 1 de cada 3 imagenes se "
            "repita. En el peor caso esta tirada llega a %s USD: cada imagen "
            "se intenta como mucho %d veces%s y solo se repintan por zonas (a "
            "%s USD cada una) los fallos que no se pueden corregir gratis. Las "
            "manos, el rostro y la piel se corrigen sin gastar. El saldo se "
            "comprueba antes de cada llamada, asi que no se gasta mas de lo "
            "que tienes."
            % (_money(total_max), attempts, usual, _money(repair_unit)))


def estimate_run_cost(plan: dict, quality: str, limits: dict | None = None,
                      user_id: str = "") -> dict:
    """What this plan will cost, repairs and retries included.

    ``limits`` is what ``user_limits`` read off her Ajustes screen.  It is an
    argument rather than another lookup because the caller has to hand the SAME
    dict to the run: an estimate that prices two retries beside a run that buys
    three is the defect this file was already fixed for once, and reading the
    setting twice is how it comes back.
    """
    plan_d = plan if isinstance(plan, dict) else {}
    qual = _quality(quality)
    # ONE decision about her face, taken here and reused by the run: which
    # endpoint each variant will hit, with which mask and with how many
    # photographs of her.  Everything below - the model named on the screen,
    # the price per image, the total, the ceiling, the resolution warning and
    # the sentence about her face - is computed from these objects and from
    # nothing else.
    shields = plan_shields(plan_d)
    requests = plan_requests(plan_d, qual, shields)
    references = plan_references(plan_d)
    n_images = len(requests)
    budget = plan_d.get("budget_usd")
    budget_usd = _f(budget, float("inf")) if budget is not None else float("inf")

    changes = _plan_changes(plan_d)
    try:
        # The estimate has to price the engine that will really run this
        # plan.  Pricing the free one for work it cannot do is how a run was
        # announced as "sin coste" and then delivered the wrong images.
        # The operation is read off the request that will really be sent: a
        # variant whose face is protected is an inpaint, and asking the router
        # to price a 'generate' while sending an 'inpaint' is how the quote and
        # the invoice came apart in the first place.
        provider, _chosen, reason = choose_provider(
            _text(requests[0].operation) or "generate", qual, budget_usd,
            plan_d.get("provider"), changes=changes, request=requests[0])
    except ProviderError as exc:
        return {"total_usd": 0.0, "per_image_usd": 0.0, "provider": "",
                "model": "", "endpoint": "", "modelos": [], "breakdown": [],
                "free": True,
                "n_images": n_images, "quality": qual,
                "factor": REPAIR_FACTOR, "reason": str(exc), "aviso": "",
                "aviso_resolucion": "", "aviso_coste": "", "aviso_rostro": "",
                "rostro": {"protegidas": 0, "de": n_images, "referencias": 0,
                           "detalle": []},
                "aviso_opciones": _history_note(plan_d, user_id),
                "intentos_maximos": 0, "total_max_usd": 0.0}

    # THE SAME FALLBACK THE RUN MAKES, made here so the quote is never of a
    # call nobody will attempt.  The free engine's inpaint fills a hole from
    # the pixels around it: handed the mask of a torso it would sample the wall
    # rather than sew a suit, so ``_run_variant`` drops the mask and makes the
    # whole image whenever the chosen engine is not generative.  Quoting the
    # masked path against that engine would put "tu rostro no se va a volver a
    # generar" on the screen for a run that will not honour it.
    caps = _caps(provider)
    if any(sh.get("masked") for sh in shields) and caps is not None \
            and not bool(getattr(caps, "generative", True)):
        shields = [{**sh, "masked": False, "mask_path": "", "cover": 0.0,
                    "estado": "motor no generativo",
                    "reason": "El motor gratuito no sabe repintar solo una "
                              "zona, asi que esta imagen se hace entera y se "
                              "comprueba la identidad."}
                   for sh in shields]
        requests = plan_requests(plan_d, qual, shields)

    costs = [round(_cost(provider, _text(req.operation) or "generate", qual,
                         req), 6) for req in requests]
    generation = round(sum(costs), 6)
    per_image = round(generation / float(max(1, n_images)), 6)

    # A repair is priced on the endpoint that repaints, not as a slice of the
    # endpoint that generates: the same builder, operation 'inpaint', so the
    # provider picks its fill model exactly as the run will.  On fal that is
    # 0.050 USD a region against a 0.00625 USD draft image, which is why a
    # percentage of the generation could never have covered it.
    repair_unit = round(_cost(provider, "inpaint", qual, _probe_request(
        "inpaint", qual, source_path=plan_d.get("source_path"),
        source_size=plan_d.get("source_size"))), 6)
    round_price = round(repair_unit * REPAIR_REGIONS_TYPICAL, 6)
    # One image in three needs that pass, and the pass is a retry AND the
    # repaint of the attempt that failed - both bill.  But only the ones the
    # limits in force still allow: _run_variant's loop is
    # ``max_retries_per_variant + 1`` attempts long, so with 0 retries there is
    # no second attempt to price, and its repair block is skipped outright when
    # ``max_repair_rounds`` is 0, so there is no zone to repaint either.
    # Quoting money the configuration forbids is the same defect as promising a
    # ceiling the run can exceed, taken from the other side: measured
    # 2026-09-04 with both limits at 0, two images at 0.040 USD were announced
    # at 0.1780 USD under a 0.3800 USD ceiling when 0.0800 USD was the only
    # spend the run could make.  With the shipped limits (2 retries, 2 rounds)
    # both numbers below are exactly what they were.
    lim = limits if isinstance(limits, dict) else {}
    rounds = int(lim.get("max_repair_rounds",
                         SETTINGS.limits.max_repair_rounds))
    attempts = int(lim.get("max_retries",
                           SETTINGS.limits.max_retries_per_variant)) + 1
    retry_risk = generation if attempts > 1 else 0.0
    repaint_risk = n_images * round_price if rounds > 0 else 0.0
    extra = round((REPAIR_FACTOR - 1.0) * (retry_risk + repaint_risk), 6)
    total = round(generation + extra, 6)

    # The ceiling the hard limits allow, so the screen can promise a number
    # nothing can exceed: every attempt this run may make, each one repainted
    # in as many regions as repair.MAX_REGIONS permits.  Measured on
    # 2026-09-03 with the shipped limits, one draft variant that kept failing
    # billed 0.2687 USD against a 0.0084 USD quote; this ceiling is 0.4688.
    from .repair import MAX_REGIONS

    # Bounded by the HARD limit, which is the only number that cannot be
    # exceeded.  The stop rule usually ends a variant after the second reading,
    # but it only fires on a repeated one, so a check that scatters buys the
    # third attempt - and a ceiling that assumed two was 0.1200 USD short of a
    # 1.2600 USD run on 2026-09-04.  With the free corrections owning hands,
    # face and skin, a repaint is also no longer bought for the defects that
    # used to open most repair rounds - but the ceiling still counts them,
    # because a run that meets three genuinely repaintable regions may still
    # buy them.  This is exactly what the run's own reservations add up to, so
    # the promise on the screen and the money the gate is willing to hold are
    # now the same number.
    regions = MAX_REGIONS if rounds > 0 else 0
    total_max = round(attempts * (generation
                                  + n_images * regions * repair_unit), 6)

    # WHAT WILL REALLY BE CALLED, named on the screen.  The role key is what
    # the code routes on; the endpoint is what fal's price list calls it, and
    # it is the only form of the answer the client can check herself.
    roles = [_model_name(provider, qual, req) for req in requests]
    endpoints = [endpoint_name(provider, req, qual) for req in requests]
    model = roles[0] if len(set(roles)) <= 1 else " + ".join(sorted(set(roles)))
    endpoint = endpoints[0] if len(set(endpoints)) <= 1 \
        else " + ".join(sorted(set(endpoints)))
    by_model: dict[tuple, dict] = {}
    for role, url, unit, shield, req in zip(roles, endpoints, costs, shields,
                                            requests):
        row = by_model.setdefault((role, url, unit), {
            "modelo": role, "endpoint": url, "usd": unit, "n": 0,
            "rostro_protegido": bool(shield.get("masked")),
            # What really leaves the machine with this call, counted by the
            # provider from the endpoint's own knobs.
            "fotos": photos_sent(provider, req, references)})
        row["n"] += 1
    modelos = list(by_model.values())
    masked_n = sum(1 for sh in shields if sh.get("masked"))

    detail = ("%d variante(s) x %s USD" % (n_images, _money(per_image))
              if len(set(costs)) <= 1 else
              "%d variante(s), de %s a %s USD segun lo que cambia cada una"
              % (n_images, _money(min(costs)), _money(max(costs))))
    breakdown = [
        {"item": "generacion", "detail": detail,
         "n": n_images, "unit_usd": per_image, "usd": generation},
        {"item": "reparaciones y reintentos",
         "detail": "factor %.2f: 1 de cada 3 imagenes se repite y se repinta "
                   "(%s USD por zona, %d zonas por reparacion)"
                   % (REPAIR_FACTOR, _money(repair_unit),
                      REPAIR_REGIONS_TYPICAL),
         "n": 0, "unit_usd": round_price, "usd": extra},
    ]
    # The very sentence the run would later write into the plan notes, handed
    # back now - while the estimate is still on a button she has not pressed.
    # A change that will not be applied has to be known before it is paid for,
    # not found afterwards in a report next to six images that ignore it.
    return {
        "aviso": _unapplied_note(provider, changes),
        # What the tier can really deliver in pixels, said before she pays for
        # it rather than discovered in the file properties afterwards - and
        # said about the endpoint that will run, because a masked call hands
        # back her own photograph at her own size and the Kontext sentence
        # would understate it threefold.
        "aviso_resolucion": resolution_note(provider, qual, requests[0],
                                            plan_d.get("source_size")),
        # And the answer to the question the client actually asked first: will
        # this look like her?  It is the same decision the run acts on, said in
        # her language, with the endpoint and the price it implies.
        "aviso_rostro": _face_note(shields, endpoint, per_image,
                                   photos_sent(provider, requests[0],
                                               references)),
        "rostro": {"protegidas": masked_n, "de": n_images,
                   "fotos_enviadas": photos_sent(provider, requests[0],
                                                 references),
                   "referencias": (0 if masked_n == n_images
                                   else min(REFERENCE_COUNT,
                                            1 + len(references))),
                   "detalle": [{"protegido": bool(sh.get("masked")),
                                "zona": round(float(sh.get("cover") or 0.0), 4),
                                "estado": sh.get("estado") or "",
                                "motivo": sh.get("reason") or ""}
                               for sh in shields]},
        # And the worst case, in the same breath as the estimate.  She is
        # never surprised by a bill only if the bill cannot exceed a number
        # she was shown.
        "aviso_coste": _cost_ceiling_note(total, total_max, repair_unit,
                                          attempts),
        # And what her own paid history says about the options she just picked,
        # while picking a different one is still free.
        "aviso_opciones": _history_note(plan_d, user_id),
        "intentos_maximos": attempts,
        "total_max_usd": total_max,
        "total_usd": total,
        "per_image_usd": per_image,
        "provider": _name_of(provider),
        "model": model,
        "endpoint": endpoint,
        "modelos": modelos,
        "breakdown": breakdown,
        "free": total <= 0.0,
        "n_images": n_images,
        "quality": qual,
        "factor": REPAIR_FACTOR,
        "reason": reason,
    }


# -------------------------------------------------------------------- price

def pinned_provider(user_id: str) -> str | None:
    """The engine the user pinned in Ajustes, or None for 'decide tu'.

    Stored as the ``default_provider`` user setting; 'auto' and an absent row
    both mean no preference, which is what ``choose_provider`` already reads as
    'choose for me'.  It lives here because everything that has to agree about
    the price - the estimate, the run and the balance page - has to agree first
    about which engine is going to run.
    """
    row = db.q1("SELECT value_json FROM user_settings WHERE user_id=? "
                "AND key='default_provider'", (user_id,))
    value = db.loads(row["value_json"], "") if row else ""
    name = str(value or "").strip().lower()
    return None if name in ("", "auto") else name


def user_limits(user_id: str) -> dict:
    """The retry and repair limits this user set in Ajustes, clamped.

    They were dead settings.  ``routers/settings.py`` has stored
    ``max_retries``, ``max_repair_rounds`` and ``autorepair`` per user since the
    screen was written, and nothing ever read them: ``_run_variant`` and the
    repair block both went to ``SETTINGS.limits``, which is global.  Measured
    2026-09-04 on this installation, a user who set "0 reparaciones" and "0
    reintentos" still got a run that would buy up to three attempts and two
    repaint rounds - up to 1.71 USD against a balance of 2.58.  A money setting
    that does nothing is worse than no setting at all.

    It lives here, beside ``pinned_provider``, for the same reason: the
    estimate, the ceiling and the run must read one number, not three.  The
    stored value is clamped by the configured maximum, so a user can only ever
    ask for LESS than the installation allows, never more.
    """
    rows = db.q("SELECT key, value_json FROM user_settings WHERE user_id=? "
                "AND key IN ('max_retries','max_repair_rounds','autorepair')",
                (user_id,))
    stored = {r["key"]: db.loads(r["value_json"], None) for r in (rows or [])}

    def _pick(key: str, ceiling: int) -> int:
        raw = stored.get(key)
        if raw is None:
            return int(ceiling)
        try:
            return max(0, min(int(raw), int(ceiling)))
        except (TypeError, ValueError):
            return int(ceiling)

    rounds = _pick("max_repair_rounds", SETTINGS.limits.max_repair_rounds)
    # "Autorepair off" and "zero repair rounds" are the same instruction said
    # two ways, and the screen offers both; whichever she used has to hold.
    if stored.get("autorepair") is False:
        rounds = 0
    return {"max_retries": _pick("max_retries",
                                 SETTINGS.limits.max_retries_per_variant),
            "max_repair_rounds": rounds}


def price_per_image(quality: str, prefer: str | None = None, *,
                    operation: str = "generate", source_path: Any = None,
                    source_size: Any = None, framing: Any = None,
                    changes: Any = None, references: int = 0) -> float:
    """What one image of this tier costs on the engine that would run it.

    The balance page used to read a hard-coded table - 0.055 USD for 'high' -
    while fal billed 0.040 or 0.080 depending on which model the request chose.
    Three numbers for one image is one too many by two, so there is now a
    single source: the provider's own estimate_cost, measured on a request
    built exactly like the one the run would send.  ``changes`` is what the
    image has to change, because that is half of the routing: a background swap
    runs free on the local engine and a new garment cannot.  Returns 0.0 when
    no engine can be chosen, which is also what the free local engine answers.
    """
    qual = _quality(quality)
    # ``references`` is a COUNT, not a list of files: what changes the model -
    # and therefore the price - is whether the request carries references at
    # all, and this function is called from the balance page, which has no
    # business opening her photographs to answer "what does an image cost".
    # A run that sends none quotes none, so the two still agree.
    refs = ["referencia_%d.jpg" % i for i in range(max(0, int(references)))]
    req = _probe_request(operation, qual, source_path=source_path,
                         source_size=source_size, framing=framing,
                         reference_paths=refs)
    try:
        provider, _model, _why = choose_provider(operation, qual, None, prefer,
                                                 changes=changes, request=req)
    except ProviderError:
        return 0.0
    return round(_cost(provider, _operation(operation), qual, req), 6)
