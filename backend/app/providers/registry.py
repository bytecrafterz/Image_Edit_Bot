"""Provider registry - the only module that knows which vendors exist.

Everything above this layer asks for "an image provider" or "the best vision
provider available"; nothing above it names fal.ai or Anthropic.  Replacing a
vendor is therefore an edit to SPECS, which is exactly the guarantee the client
asked for.

Two invariants:

  * each family always contains a free, keyless provider, so the whole robot is
    demonstrable with no accounts at all;
  * key presence is re-read on every availability() call, because the user can
    paste a key into the settings page while the process is running.

Providers are imported lazily and defensively: a broken optional provider is
logged and skipped, never allowed to take the application down at startup.
"""
from __future__ import annotations

import importlib
import logging
import threading
from dataclasses import dataclass

from .base import Capabilities, ImageProvider, ProviderError, VisionProvider

log = logging.getLogger("photorobot.providers")


@dataclass(frozen=True)
class _Spec:
    """Static declaration of one provider.  Cheap to build, never imported."""
    name: str
    kind: str          # image | vision
    module: str        # module path relative to this package
    cls: str
    rank: int          # capability rank; higher wins when the caller says "auto"
    key_name: str      # environment variable name, "" when no key is needed
    notes: str


# Order here is documentation only; the public functions sort explicitly.
SPECS: tuple[_Spec, ...] = (
    _Spec("local", "image", ".local_free", "LocalFreeProvider", 10, "",
          "Compositor local: recorte, fondo, color y retoque. Coste cero."),
    _Spec("fal", "image", ".fal", "FalProvider", 60, "FAL_KEY",
          "fal.ai: img2img e inpainting con conservacion de identidad."),
    _Spec("heuristic", "vision", ".heuristic_vision", "HeuristicVision", 10, "",
          "Lectura local por vision por computador. Coste cero."),
    _Spec("claude", "vision", ".anthropic_vision", "ClaudeVision", 60,
          "ANTHROPIC_API_KEY",
          "Claude: lee la foto, redacta el prompt y critica el resultado."),
)

_AUTO_NAMES = ("", "auto", "default", "best", "mejor")
_FREE_NAMES = ("local", "free", "gratis", "libre")

_LOCK = threading.Lock()
_CACHE: dict[str, object] = {}
_FAILED: dict[str, str] = {}


# ------------------------------------------------------------------ loading

def _spec_by_name(kind: str, name: str) -> _Spec | None:
    for spec in SPECS:
        if spec.kind == kind and spec.name == name:
            return spec
    return None


def _free_spec(kind: str) -> _Spec | None:
    """The keyless provider of a family - the one that must always work."""
    for spec in SPECS:
        if spec.kind == kind and not spec.key_name:
            return spec
    return None


def _instance(spec: _Spec):
    """Import and construct once.  Returns None when the provider is broken."""
    with _LOCK:
        cached = _CACHE.get(spec.name)
        if cached is not None:
            return cached
        if spec.name in _FAILED:
            return None
    try:
        module = importlib.import_module(spec.module, __package__)
        obj = getattr(module, spec.cls)()
    except Exception as exc:                       # never fatal: log and skip
        reason = f"{type(exc).__name__}: {exc}"
        with _LOCK:
            _FAILED[spec.name] = reason
        log.warning("Proveedor '%s' no se pudo cargar (%s)", spec.name, reason)
        return None
    with _LOCK:
        _CACHE[spec.name] = obj
    return obj


def _is_available(obj) -> bool:
    if obj is None:
        return False
    try:
        return bool(obj.available())
    except Exception as exc:
        log.warning("available() fallo en %r: %s", obj, exc)
        return False


def _caps(obj, spec: _Spec) -> Capabilities:
    """Capabilities for either family; vision providers do not declare them."""
    if spec.kind == "image":
        try:
            caps = obj.capabilities()
            if isinstance(caps, Capabilities):
                return caps
        except Exception as exc:
            log.warning("capabilities() fallo en '%s': %s", spec.name, exc)
    cost = 0.0
    if spec.kind == "vision":
        try:
            cost = float(obj.estimate_cost(1))
        except Exception:
            cost = 0.0
    return Capabilities(name=spec.name, kind=spec.kind,
                        needs_key=bool(spec.key_name), key_name=spec.key_name,
                        cost_per_image_usd=cost, notes=spec.notes)


def _ordered(kind: str, cheapest_first: bool) -> list:
    """Live instances of one family, filtered by availability and ordered."""
    found: list[tuple[float, int, str, object]] = []
    for spec in SPECS:
        if spec.kind != kind:
            continue
        obj = _instance(spec)
        if not _is_available(obj):
            continue
        cost = float(_caps(obj, spec).cost_per_image_usd or 0.0)
        found.append((cost, spec.rank, spec.name, obj))
    if cheapest_first:
        found.sort(key=lambda row: (row[0], -row[1], row[2]))
    else:
        found.sort(key=lambda row: (-row[1], row[0], row[2]))
    return [row[3] for row in found]


# ------------------------------------------------------------------- public

def image_providers() -> list[ImageProvider]:
    """Every usable image provider, cheapest capable first."""
    return _ordered("image", cheapest_first=True)


def vision_providers() -> list[VisionProvider]:
    """Every usable vision provider, cheapest capable first."""
    return _ordered("vision", cheapest_first=True)


def _pick(kind: str, name: str | None):
    wanted = (name or "").strip().lower()

    if wanted in _AUTO_NAMES:
        best = _ordered(kind, cheapest_first=False)
        if best:
            return best[0]
        free = _free_spec(kind)
        obj = _instance(free) if free else None
        if obj is not None:
            return obj
        raise ProviderError(
            f"No hay ningun proveedor de tipo '{kind}' disponible.",
            retryable=False, code="no_provider")

    if wanted in _FREE_NAMES:
        spec = _free_spec(kind)
        obj = _instance(spec) if spec else None
        if obj is None:
            raise ProviderError(
                f"El proveedor local de '{kind}' no se pudo cargar.",
                retryable=False, code="no_provider")
        return obj

    spec = _spec_by_name(kind, wanted)
    if spec is None:
        raise ProviderError(f"Proveedor desconocido: {wanted or name}.",
                            retryable=False, code="unknown_provider")
    obj = _instance(spec)
    if obj is None:
        raise ProviderError(
            f"El proveedor {spec.name} no se pudo cargar: "
            + _FAILED.get(spec.name, "error desconocido"),
            retryable=False, code="load_failed")
    if not _is_available(obj):
        missing = spec.key_name or "configuracion"
        raise ProviderError(
            f"El proveedor {spec.name} no esta configurado (falta {missing}).",
            retryable=False, code="missing_key")
    return obj


def get_image_provider(name: str | None = None) -> ImageProvider:
    """By name; 'auto' picks the best available, 'local' forces the free one."""
    return _pick("image", name)


def get_vision_provider(name: str | None = None) -> VisionProvider:
    """By name; 'auto' picks the best available, 'local' forces the free one."""
    return _pick("vision", name)


def availability() -> dict:
    """Matrix for /api/health and the admin page.

    Key presence is re-evaluated on every call: the user can paste an API key
    into the settings page and expect the next refresh to show the provider on.
    """
    out: dict[str, dict] = {}
    for spec in SPECS:
        obj = _instance(spec)
        if obj is None:
            out[spec.name] = {
                "available": False,
                "needs_key": bool(spec.key_name),
                "key_name": spec.key_name,
                "kind": spec.kind,
                "cost_per_image_usd": 0.0,
                "notes": "No se pudo cargar: "
                         + _FAILED.get(spec.name, "error desconocido"),
            }
            continue
        caps = _caps(obj, spec)
        out[spec.name] = {
            "available": _is_available(obj),
            "needs_key": bool(spec.key_name),
            "key_name": spec.key_name,
            "kind": spec.kind,
            "cost_per_image_usd": round(float(caps.cost_per_image_usd or 0.0), 6),
            "notes": caps.notes or spec.notes,
        }
    return out
