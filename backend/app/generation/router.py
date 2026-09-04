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


def delivered_side(provider: Any, quality: str) -> int:
    """Longest side this provider really hands back at this tier."""
    caps = _caps(provider)
    if caps is None:
        return 0
    ceiling = int(_f(getattr(caps, "out_max_side", 0), 0.0)) or \
        int(_f(getattr(caps, "max_side", 0), 0.0))
    asked = QUALITY_LONGEST_SIDE.get(_quality(quality), 768)
    return min(asked, ceiling) if ceiling else asked


def resolution_note(provider: Any, quality: str) -> str:
    """One Spanish sentence when a tier cannot deliver the pixels it names.

    fal's Kontext endpoints expose no size knob at all - see MODELS in
    providers/fal - and answer with about one megapixel whatever was paid.
    Silence here is what let 'alta' and 'maxima' sell 1536 and 2048 px files
    that arrive at 1024 px.  The tier still buys a more faithful model; it just
    stops promising pixels, and putting the local upscaler in between would not
    buy them either - it moved the measured texture loss from +0.018 to +0.050
    over her seven photographs.
    """
    asked = QUALITY_LONGEST_SIDE.get(_quality(quality), 768)
    gets = delivered_side(provider, quality)
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
            "de un proveedor generativo para poder pedirlos."
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
        raise ProviderError("No hay ningun proveedor de imagen registrado.",
                            retryable=False, code="no_provider")

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
                       else "es tu proveedor por defecto")
                reason = ("Se usa %s (%s) porque %s. Coste por imagen: %s "
                          "USD." % (_name_of(target),
                                    _model_name(target, qual, request), why,
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
            _name_of(provider), model, criterion, _money(cost))
        if skipped:
            reason += " Descartados: " + "; ".join(skipped[:3]) + "."
        return provider, model, reason

    if local is not None and _supports(local, operation) and _available(local):
        reason = ("; ".join(skipped[:3]) + ", se usa el motor local gratuito."
                  if skipped else
                  "No hay proveedor de pago disponible, se usa el motor local "
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
                      "unico proveedor disponible que puede hacerlo."
                      % _name_of(provider))
            warning = _unapplied_note(provider, wanted)
            if warning:
                reason += " " + warning
            return provider, _model_name(provider, qual, request), reason

    raise ProviderError(
        "Ningun proveedor disponible puede %s ahora mismo (%s)."
        % (operation, "; ".join(skipped[:3]) or "sin motores activos"),
        retryable=False, code="no_provider")


# ----------------------------------------------------------------- estimate

def plan_requests(plan: dict, quality: str) -> list[GenRequest]:
    """The requests this plan will really send, one per variant.

    Built here so the estimate screen and the run cannot drift: same builder,
    same source photograph, same framing, same size.  A variant that changes
    the framing changes the shape of its image and, on the per megapixel
    tiers, its price - so each one is asked separately instead of multiplying
    the first one by six.
    """
    plan_d = plan if isinstance(plan, dict) else {}
    variants = plan_d.get("variants")
    rows = variants if isinstance(variants, list) and variants else [{}]
    source_path = plan_d.get("source_path")
    source_size = plan_d.get("source_size")
    out: list[GenRequest] = []
    for variant in rows:
        choices = variant.get("choices") if isinstance(variant, dict) else {}
        out.append(_probe_request("generate", quality, source_path=source_path,
                                  source_size=source_size,
                                  framing=(choices or {})))
    return out


def _cost_ceiling_note(total: float, total_max: float,
                       repair_unit: float) -> str:
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
    return ("Aviso: la estimacion cuenta con que 1 de cada 3 imagenes se "
            "repita. Si todas fallan a la primera, esta tirada puede llegar a "
            "%s USD, porque cada intento rechazado se repinta por zonas a %s "
            "USD cada una. El saldo se comprueba antes de cada llamada, asi "
            "que no se gasta mas de lo que tienes."
            % (_money(total_max), _money(repair_unit)))


def estimate_run_cost(plan: dict, quality: str) -> dict:
    """What this plan will cost, repairs and retries included."""
    plan_d = plan if isinstance(plan, dict) else {}
    qual = _quality(quality)
    requests = plan_requests(plan_d, qual)
    n_images = len(requests)
    budget = plan_d.get("budget_usd")
    budget_usd = _f(budget, float("inf")) if budget is not None else float("inf")

    changes = _plan_changes(plan_d)
    try:
        # The estimate has to price the engine that will really run this
        # plan.  Pricing the free one for work it cannot do is how a run was
        # announced as "sin coste" and then delivered the wrong images.
        provider, model, reason = choose_provider(
            "generate", qual, budget_usd, plan_d.get("provider"),
            changes=changes, request=requests[0])
    except ProviderError as exc:
        return {"total_usd": 0.0, "per_image_usd": 0.0, "provider": "",
                "model": "", "breakdown": [], "free": True,
                "n_images": n_images, "quality": qual,
                "factor": REPAIR_FACTOR, "reason": str(exc), "aviso": "",
                "aviso_resolucion": "", "aviso_coste": "",
                "total_max_usd": 0.0}

    costs = [round(_cost(provider, "generate", qual, req), 6) for req in requests]
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
    # repaint of the attempt that failed - both bill.
    extra = round((REPAIR_FACTOR - 1.0) * (generation + n_images * round_price), 6)
    total = round(generation + extra, 6)

    # The ceiling the hard limits allow, so the screen can promise a number
    # nothing can exceed: every attempt this run may make, each one repainted
    # in as many regions as repair.MAX_REGIONS permits.  Measured on
    # 2026-09-03 with the shipped limits, one draft variant that kept failing
    # billed 0.2687 USD against a 0.0084 USD quote; this ceiling is 0.4688.
    from .repair import MAX_REGIONS

    attempts = int(SETTINGS.limits.max_retries_per_variant) + 1
    total_max = round(attempts * (generation
                                  + n_images * MAX_REGIONS * repair_unit), 6)

    detail = ("%d variante(s) x %s USD" % (n_images, _money(per_image))
              if len(set(costs)) <= 1 else
              "%d variante(s), de %s a %s USD segun el encuadre"
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
        # it rather than discovered in the file properties afterwards.
        "aviso_resolucion": resolution_note(provider, qual),
        # And the worst case, in the same breath as the estimate.  She is
        # never surprised by a bill only if the bill cannot exceed a number
        # she was shown.
        "aviso_coste": _cost_ceiling_note(total, total_max, repair_unit),
        "total_max_usd": total_max,
        "total_usd": total,
        "per_image_usd": per_image,
        "provider": _name_of(provider),
        "model": model,
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


def price_per_image(quality: str, prefer: str | None = None, *,
                    operation: str = "generate", source_path: Any = None,
                    source_size: Any = None, framing: Any = None,
                    changes: Any = None) -> float:
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
    req = _probe_request(operation, qual, source_path=source_path,
                         source_size=source_size, framing=framing)
    try:
        provider, _model, _why = choose_provider(operation, qual, None, prefer,
                                                 changes=changes, request=req)
    except ProviderError:
        return 0.0
    return round(_cost(provider, _operation(operation), qual, req), 6)
