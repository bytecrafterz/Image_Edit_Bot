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

from ..catalog import options as options_mod
from ..config import SETTINGS
from ..providers import registry as registry_mod
from ..providers.base import GenRequest, ImageProvider, ProviderError

# Planning factor: about one image in three needs a repair pass or a retry.
# Documented here so the number on the estimate screen can be explained.
REPAIR_FACTOR = 1.35

QUALITIES = ("draft", "preview", "standard", "high", "max")
QUALITY_MIN_SIDE = {"draft": 0, "preview": 0, "standard": 1024,
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


def _probe_request(operation: str, quality: str) -> GenRequest:
    """A request shaped like the one that will really be sent.

    This matters for the price.  Every image this product makes starts from a
    photograph of the person, and providers choose a cheaper text-only model
    when no source is attached - so pricing an empty request quoted the client
    0.003 USD for work that actually bills at 0.040, a thirteenfold
    understatement on the one number she said she cared about most.  The probe
    therefore carries a source path, exactly like the real call.
    """
    return GenRequest(prompt="", operation=operation, quality=quality,
                      source_path="probe.jpg")


def _cost(provider: Any, operation: str, quality: str) -> float:
    req = _probe_request(operation, quality)
    try:
        return max(0.0, _f(provider.estimate_cost(req), 0.0))
    except Exception:
        caps = _caps(provider)
        return max(0.0, _f(getattr(caps, "cost_per_image_usd", 0.0), 0.0))


def _model_name(provider: Any, quality: str) -> str:
    # pick_model expects a request, not a quality string.  Handing it the bare
    # string used to "work" only because attribute lookups on a str all miss and
    # it fell through to the no-source default, which is how the draft model
    # ended up named in every estimate.
    probe = _probe_request("generate", quality)
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
                    changes: Any = None) -> tuple[ImageProvider, str, str]:
    """Pick who makes this image, and say why in one Spanish sentence.

    ``changes`` is what the user asked to change: the option group keys of the
    variant, or the variant's ``choices`` dict.  It matters because an engine
    that cannot perform a requested change is not a cheaper way of doing the
    job, it is a different job - routing on price alone is exactly how a run
    produced six previews wearing the original clothes in the original pose.
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
            cost = _cost(target, operation, qual)
            if cost > budget + 1e-9:
                skipped.append("%s cuesta %s USD por imagen y el presupuesto es "
                               "%s USD" % (_name_of(target), _money(cost),
                                           _money(budget)))
            else:
                why = ("lo pediste explicitamente" if explicit
                       else "es tu proveedor por defecto")
                reason = ("Se usa %s (%s) porque %s. Coste por imagen: %s "
                          "USD." % (_name_of(target),
                                    _model_name(target, qual), why,
                                    _money(cost)))
                # A stated preference is an instruction and is obeyed, but it
                # never buys silence: if that engine cannot make one of her
                # changes she is told now, not after six images.
                warning = _unapplied_note(target, wanted)
                if warning:
                    reason += " " + warning
                return target, _model_name(target, qual), reason

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
        cost = _cost(provider, operation, qual)
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
        reason = "Se usa %s (%s) porque %s. Coste por imagen: %s USD." % (
            _name_of(provider), _model_name(provider, qual), criterion,
            _money(cost))
        if skipped:
            reason += " Descartados: " + "; ".join(skipped[:3]) + "."
        return provider, _model_name(provider, qual), reason

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
        return local, _model_name(local, qual), reason

    for provider in providers:
        if _supports(provider, operation) and _available(provider):
            reason = ("Sin alternativas dentro del presupuesto: se usa %s, el "
                      "unico proveedor disponible que puede hacerlo."
                      % _name_of(provider))
            warning = _unapplied_note(provider, wanted)
            if warning:
                reason += " " + warning
            return provider, _model_name(provider, qual), reason

    raise ProviderError(
        "Ningun proveedor disponible puede %s ahora mismo (%s)."
        % (operation, "; ".join(skipped[:3]) or "sin motores activos"),
        retryable=False, code="no_provider")


# ----------------------------------------------------------------- estimate

def estimate_run_cost(plan: dict, quality: str) -> dict:
    """What this plan will cost, repairs and retries included."""
    plan_d = plan if isinstance(plan, dict) else {}
    variants = plan_d.get("variants")
    n_images = len(variants) if isinstance(variants, list) and variants else 1
    qual = _quality(quality)
    budget = plan_d.get("budget_usd")
    budget_usd = _f(budget, float("inf")) if budget is not None else float("inf")

    changes = _plan_changes(plan_d)
    try:
        # The estimate has to price the engine that will really run this
        # plan.  Pricing the free one for work it cannot do is how a run was
        # announced as "sin coste" and then delivered the wrong images.
        provider, model, reason = choose_provider(
            "generate", qual, budget_usd, plan_d.get("provider"),
            changes=changes)
    except ProviderError as exc:
        return {"total_usd": 0.0, "per_image_usd": 0.0, "provider": "",
                "model": "", "breakdown": [], "free": True,
                "n_images": n_images, "quality": qual,
                "factor": REPAIR_FACTOR, "reason": str(exc), "aviso": ""}

    per_image = round(_cost(provider, "generate", qual), 6)
    generation = round(per_image * n_images, 6)
    extra = round(generation * (REPAIR_FACTOR - 1.0), 6)
    total = round(generation + extra, 6)

    breakdown = [
        {"item": "generacion",
         "detail": "%d variante(s) x %s USD" % (n_images, _money(per_image)),
         "n": n_images, "unit_usd": per_image, "usd": generation},
        {"item": "reparaciones y reintentos",
         "detail": "factor de planificacion %.2f sobre la generacion"
                   % REPAIR_FACTOR,
         "n": 0, "unit_usd": 0.0, "usd": extra},
    ]
    # The very sentence the run would later write into the plan notes, handed
    # back now - while the estimate is still on a button she has not pressed.
    # A change that will not be applied has to be known before it is paid for,
    # not found afterwards in a report next to six images that ignore it.
    return {
        "aviso": _unapplied_note(provider, changes),
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
